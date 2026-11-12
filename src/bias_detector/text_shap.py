# @title
# Módulo text_shap
"""SHAP de grupos de texto para Longformer congelado + MLP-MIL.

Explica una detección positiva manteniendo fijas las otras ventanas del bag.
No altera umbrales ni utiliza etiquetas de referencia.
Requiere la conversación original y ventanas/embeddings guardados.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


HEADER = re.compile(r"\[(USER|LLM)\]")


@dataclass(frozen=True)
class ShapTextConfig:
    model_name: str = "allenai/longformer-base-4096"
    max_features: int = 8
    nsamples: int = 64
    batch_size: int = 1
    fp16: bool = True
    seed: int = 42


def _group_tokens(full_text, offsets, messages, token_start, token_end, max_features):
    """Genera grupos de contenido, inicialmente por mensaje; excluye cabeceras de rol."""
    groups = []
    for msg in messages:
        h = HEADER.match(full_text, msg["char_start"])
        if h is None:
            raise ValueError(f"Cabecera normalizada no encontrada en turno {msg['turn_id']}")
        content_start, content_end = h.end(), msg["char_end"]
        positions = [i - token_start + 1 for i in range(token_start, token_end)
                     if offsets[i][1] > content_start and offsets[i][0] < content_end
                     and offsets[i][1] > offsets[i][0]]
        if positions:
            groups.append({"positions": positions, "turn_ids": [msg["turn_id"]]})
    if not groups:
        raise ValueError("No hay tokens de contenido para explicar")
    # Subdividir primero los mensajes más extensos (mejora la resolución).
    while len(groups) < max_features:
        idx = int(np.argmax([len(g["positions"]) for g in groups]))
        g = groups[idx]
        if len(g["positions"]) < 2:
            break
        middle = len(g["positions"]) // 2
        left = {"positions": g["positions"][:middle], "turn_ids": list(g["turn_ids"])}
        right = {"positions": g["positions"][middle:], "turn_ids": list(g["turn_ids"])}
        groups[idx:idx+1] = [left, right]
    # Si hay demasiados mensajes, fusionar grupos vecinos hasta max_features.
    while len(groups) > max_features:
        idx = int(np.argmin([len(groups[i]["positions"]) + len(groups[i+1]["positions"])
                             for i in range(len(groups)-1)]))
        a, b = groups[idx:idx+2]
        merged = {"positions": a["positions"] + b["positions"],
                  "turn_ids": list(dict.fromkeys(a["turn_ids"] + b["turn_ids"]))}
        groups[idx:idx+2] = [merged]
    for g in groups:
        p = np.asarray(g["positions"], dtype=int)
        g["char_start"] = int(min(offsets[token_start + int(i)-1][0] for i in p))
        g["char_end"] = int(max(offsets[token_start + int(i)-1][1] for i in p))
        g["text"] = full_text[g["char_start"]:g["char_end"]]
    return groups


class MILTextShapExplainer:
    def __init__(self, mlp, windows: pd.DataFrame, embeddings: np.ndarray,
                 window_keys: np.ndarray, *, config: ShapTextConfig = ShapTextConfig(),
                 device: str | None = None, encoder=None):
        self.config = config
        if not 1 <= config.max_features <= 16 or config.nsamples < 2 or config.batch_size < 1:
            raise ValueError("max_features entre 1 y 16, nsamples >=2, batch_size >=1")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if encoder is None:
            # Funcionamiento independiente:
            # cargar Longformer como hacíamos anteriormente.
            self.encoder, self.device = load_encoder(
                config.model_name,
                device=self.device
            )
        else:
            # Funcionamiento integrado:
            # reutilizar el Longformer del pipeline.
            self.encoder = encoder.to(self.device)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        if not self.tokenizer.is_fast or self.tokenizer.mask_token_id is None:
            raise ValueError("Se necesita tokenizer fast con token MASK")
        self.mlp = mlp.to(self.device).eval()
        self.mlp.requires_grad_(False)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        keys = np.asarray(window_keys).astype(str)
        if len(keys) != len(self.embeddings):
            raise ValueError("Cantidad de window_keys/embeddings incompatible")
        self.key_to_index = {k: i for i, k in enumerate(keys)}
        if len(self.key_to_index) != len(keys):
            raise ValueError("Claves duplicadas")
        df = windows.copy()
        df["conversation_id"] = df["conversation_id"].astype(str)
        df["window_key"] = df["conversation_id"] + "::" + df["window_id"].astype(str)
        self.window_by_key = df.set_index("window_key")
        if set(keys) != set(self.window_by_key.index):
            raise ValueError("Ventanas y embeddings no coinciden")

    def _evaluate_masks(self, mask_matrix, base_ids, groups, bias_idx, other_max_logit):
        masks = np.asarray(mask_matrix, dtype=float)
        if masks.ndim == 1:
            masks = masks[None, :]
        output = []
        for start in range(0, len(masks), self.config.batch_size):
            sub = masks[start:start+self.config.batch_size]
            sequences = np.tile(base_ids, (len(sub), 1))
            for row_idx, switches in enumerate(sub):
                for j, visible in enumerate(switches):
                    if visible < 0.5:
                        sequences[row_idx, groups[j]["positions"]] = self.tokenizer.mask_token_id
            ids = torch.tensor(sequences, dtype=torch.long, device=self.device)
            attention = torch.ones_like(ids)
            global_attention = torch.zeros_like(ids)
            global_attention[:, 0] = 1
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(self.config.fp16 and self.device.startswith("cuda"))):
                    states = self.encoder(input_ids=ids, attention_mask=attention,
                                          global_attention_mask=global_attention).last_hidden_state[:, 0, :]
                logits = self.mlp.classifier(states.float())[:, bias_idx]
                if other_max_logit is not None:
                    logits = torch.maximum(logits, torch.tensor(other_max_logit,
                                                dtype=logits.dtype, device=logits.device))
                output.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.asarray(output, dtype=np.float64)

    def explain(self, *, conversation_id, bias: str, bias_names, original_text: str,
                winning_window_id: int, nsamples: int | None = None):
        import shap
        cid = str(conversation_id)
        bias_names = tuple(bias_names)
        if bias not in bias_names:
            raise ValueError("Sesgo desconocido")
        j = bias_names.index(bias)
        key = f"{cid}::{int(winning_window_id)}"
        if key not in self.key_to_index:
            raise KeyError(f"Ventana no encontrada: {key}")
        window = self.window_by_key.loc[key]
        normalized, messages = normalize_conversation(original_text)
        tokenized = self.tokenizer(normalized, truncation=False, add_special_tokens=False,
                                   return_offsets_mapping=True, verbose=False)
        all_ids = tokenized["input_ids"]
        a, b = int(window["token_start"]), int(window["token_end"])
        expected = [self.tokenizer.cls_token_id] + all_ids[a:b] + [self.tokenizer.sep_token_id]
        stored_ids = list(window["input_ids"])
        if expected != stored_ids:
            raise ValueError("Los tokens reconstruidos no coinciden con la ventana guardada")
        groups = _group_tokens(normalized, tokenized["offset_mapping"], messages,
                               a, b, self.config.max_features)
        # Resto de ventanas de esta conversación: logits originales congelados.
        indices = [idx for k, idx in self.key_to_index.items() if k.startswith(cid + "::") and k != key]
        other_max = None
        if indices:
            with torch.inference_mode():
                X = torch.as_tensor(self.embeddings[indices], device=self.device)
                other_max = float(self.mlp.classifier(X)[:, j].max().cpu())
        def model_fn(masks):
            return self._evaluate_masks(masks, np.asarray(stored_ids, dtype=np.int64),
                                        groups, j, other_max)
        n = len(groups)
        np.random.seed(self.config.seed)
        explainer = shap.KernelExplainer(model_fn, np.zeros((1, n)), link="identity")
        shap_raw = explainer.shap_values(np.ones((1, n)),
                                         nsamples=nsamples or self.config.nsamples,
                                         silent=True)
        values = np.asarray(shap_raw).reshape(-1)
        if len(values) != n:
            raise ValueError(f"Dimensiones SHAP inesperadas: {np.asarray(shap_raw).shape}")
        original_score = float(model_fn(np.ones((1,n)))[0])
        baseline_score = float(model_fn(np.zeros((1,n)))[0])
        expected_score = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        records = [{"conversation_id": cid, "bias": bias, "window_id": int(winning_window_id),
                    "group_id": i, "turn_ids": g["turn_ids"], "text": g["text"],
                    "char_start_normalized": g["char_start"], "char_end_normalized": g["char_end"],
                    "n_tokens": len(g["positions"]), "shap_value": float(values[i])}
                   for i, g in enumerate(groups)]
        metadata = {"conversation_id": cid, "bias": bias, "window_id": int(winning_window_id),
                    "n_groups": n, "nsamples": nsamples or self.config.nsamples,
                    "original_score_recomputed": original_score,
                    "baseline_score_masked": baseline_score,
                    "expected_value": expected_score,
                    "sum_shap": float(values.sum()),
                    "additivity_error": float(abs(expected_score + values.sum() - original_score)),
                    "max_other_window_logit": other_max,
                    "masking_method": "replace_content_token_ids_with_mask_token_keep_roles_and_length",
                    "model_name": self.config.model_name}
        return pd.DataFrame(records), metadata


def save_shap_explanation(rows: pd.DataFrame, metadata: dict, output_dir: str | Path):
    """Guarda explicación individual; evita sobrescritura del resultado previo."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^a-zA-Z0-9_-]", "_", str(metadata["conversation_id"]))
    name = f"{key}__{metadata['bias']}__w{metadata['window_id']}"
    csv_path, json_path = folder / f"{name}.csv", folder / f"{name}.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"Ya hay una explicación guardada: {name}")
    rows.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path