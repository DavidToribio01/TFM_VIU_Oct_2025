# @title
# Módulo embedding_extractor
"""Persistencia reproducible de ventanas y extracción reanudable de embeddings Longformer.

Requisitos: pip install torch transformers pandas pyarrow numpy
No realiza fine-tuning ni utiliza etiquetas.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel


KEYS = ["conversation_id", "window_id"]
COLUMNS = ["conversation_id", "window_id", "input_ids", "attention_mask", "n_tokens",
           "token_start", "token_end", "char_start", "char_end", "turn_ids", "window_text"]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _normalize_windows(df_windows: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df_windows.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if df_windows.empty:
        raise ValueError("No hay ventanas")
    if df_windows[KEYS].isna().any().any() or df_windows.duplicated(KEYS).any():
        raise ValueError("Las claves conversation_id + window_id deben ser únicas y no nulas")
    df = df_windows[COLUMNS].copy()
    # ID convertido a texto de forma coherente para persistencia y reanudación.
    df["conversation_id"] = df["conversation_id"].astype(str)
    if df.duplicated(KEYS).any():
        raise ValueError("ID de conversación duplicado después de convertir a texto")
    df["window_id"] = df["window_id"].astype(int)
    df = df.sort_values(KEYS, kind="stable").reset_index(drop=True)
    for row in df.itertuples(index=False):
        if len(row.input_ids) != row.n_tokens or len(row.attention_mask) != row.n_tokens:
            raise ValueError(f"Longitud incoherente en {row.conversation_id}/{row.window_id}")
        if row.n_tokens < 3 or any(x != 1 for x in row.attention_mask):
            raise ValueError("Ventana vacía o attention_mask inválida")
    return df


def save_windows(df_windows: pd.DataFrame, folder: str | Path, segmentation_config: Any) -> Path:
    """Guarda df_windows y configuración en carpeta persistente (idealmente Google Drive).

    Falla si ya hay artefactos, para no invalidar embeddings parciales.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    parquet = folder / "windows.parquet"
    manifest = folder / "windows_manifest.json"
    if parquet.exists() or manifest.exists():
        raise FileExistsError(f"Ya hay una segmentación guardada en {folder}. Utiliza load_windows().")
    df = _normalize_windows(df_windows)
    cfg = asdict(segmentation_config) if hasattr(segmentation_config, "__dataclass_fields__") else dict(segmentation_config)
    temporary = folder / "windows.parquet.tmp"
    try:
        df.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, parquet)
        _atomic_json(manifest, {
            "format_version": 1,
            "num_windows": len(df),
            "num_conversations": int(df.conversation_id.nunique()),
            "segmentation_config": cfg,
            "sha256": _sha256_file(parquet),
        })
    finally:
        temporary.unlink(missing_ok=True)
    return parquet


def load_windows(folder: str | Path) -> tuple[pd.DataFrame, dict]:
    folder = Path(folder)
    with (folder / "windows_manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    parquet = folder / "windows.parquet"
    if _sha256_file(parquet) != manifest["sha256"]:
        raise ValueError("El archivo de ventanas no coincide con el manifest")
    df = _normalize_windows(pd.read_parquet(parquet, engine="pyarrow"))
    if len(df) != manifest["num_windows"]:
        raise ValueError("Número de ventanas incoherente")
    return df, manifest


def load_encoder(model_name: str = "allenai/longformer-base-4096", *, device: str | None = None):
    """Carga el codificador congelado, sin cabeza de clasificación."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name)
    model.requires_grad_(False)
    model.eval().to(device)
    return model, device


def _encode_batch(model, batch: pd.DataFrame, device: str, fp16: bool) -> np.ndarray:
    # Padding dinámico dentro de cada micro-lote.
    sequences = batch["input_ids"].tolist()
    max_len = max(map(len, sequences))
    ids = torch.zeros((len(sequences), max_len), dtype=torch.long)
    mask = torch.zeros_like(ids)
    global_mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        n = len(seq)
        ids[i, :n] = torch.tensor(seq, dtype=torch.long)
        mask[i, :n] = 1
        global_mask[i, 0] = 1  # Atención global para el primer token <s>.
    ids, mask, global_mask = ids.to(device), mask.to(device), global_mask.to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(fp16 and device.startswith("cuda"))):
            out = model(input_ids=ids, attention_mask=mask, global_attention_mask=global_mask)
            vectors = out.last_hidden_state[:, 0, :]
    return vectors.float().cpu().numpy()


def extract_embeddings(
    df_windows: pd.DataFrame,
    windows_manifest: dict,
    output_dir: str | Path,
    *,
    model_name: str = "allenai/longformer-base-4096",
    batch_size: int = 1,
    shard_size: int = 128,
    fp16: bool = True,
    device: str | None = None,
) -> Path:
    """Guarda embeddings en bloques .npz, retomando desde el último bloque válido.

    Cada bloque incluye vectores y claves de ventana; no se reentrena Longformer.
    Se exige el manifest del archivo parquet de ventanas para proteger la reanudación.
    """
    if batch_size < 1 or shard_size < 1:
        raise ValueError("batch_size y shard_size deben ser mayores que cero")
    df = _normalize_windows(df_windows)
    if len(df) != windows_manifest["num_windows"]:
        raise ValueError("La tabla y el manifest de segmentación no coinciden")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "embeddings_manifest.json"
    settings = {
        "version": 1, "model_name": model_name, "pooling": "first_token_global_attention",
        "dtype_saved": "float32", "num_windows": len(df), "shard_size": shard_size,
        "windows_sha256": windows_manifest["sha256"],
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != settings:
            raise ValueError("Configuración distinta: usa otra carpeta de embeddings")
    else:
        if list(output_dir.glob("shard_*.npz")):
            raise ValueError("Hay bloques sin manifest; no se puede reanudar de forma segura")
        _atomic_json(manifest_path, settings)

    model = None
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_shards = (len(df) + shard_size - 1) // shard_size
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        stop = min(len(df), start + shard_size)
        batch_df = df.iloc[start:stop]
        output = output_dir / f"shard_{shard_idx:05d}.npz"
        if output.exists():
            try:
                with np.load(output, allow_pickle=False) as existing:
                    values = existing["embeddings"]
                    keys = existing["window_keys"]
                    expected = np.array([f"{r.conversation_id}::{r.window_id}" for r in batch_df.itertuples()], dtype=str)
                    if values.shape[0] == len(batch_df) and np.array_equal(keys, expected) and np.isfinite(values).all():
                        continue
            except (OSError, ValueError, KeyError):
                pass
            raise ValueError(f"Bloque previo inválido: {output}")

        if model is None:
            model, resolved_device = load_encoder(model_name, device=resolved_device)
            print(f"Encoder cargado en {resolved_device}")
        pieces = []
        for k in range(0, len(batch_df), batch_size):
            mini = batch_df.iloc[k:k + batch_size]
            pieces.append(_encode_batch(model, mini, resolved_device, fp16))
        matrix = np.concatenate(pieces, axis=0)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Embeddings no finitos en bloque {shard_idx}")
        keys = np.array([f"{r.conversation_id}::{r.window_id}" for r in batch_df.itertuples()], dtype=str)
        temporary = output.with_suffix(".npz.tmp")
        with temporary.open("wb") as file:
            np.savez(file, embeddings=matrix, window_keys=keys)
        os.replace(temporary, output)
        print(f"Guardado bloque {shard_idx + 1}/{num_shards} ({stop}/{len(df)} ventanas)")
    return output_dir



def load_embeddings(output_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruye la matriz de embeddings a partir de los shards.

    Returns:
        embeddings: Matriz de forma (n_windows, 768).
        window_keys: Identificadores de las ventanas en el
                     mismo orden que los embeddings.
    """
    output_dir = Path(output_dir)

    # Cargar manifiesto
    settings = json.loads(
        (output_dir / "embeddings_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    n_shards = (
        settings["num_windows"] + settings["shard_size"] - 1
    ) // settings["shard_size"]

    vectors, keys = [], []

    # Recuperar los bloques de embeddings
    for idx in range(n_shards):

        shard_path = output_dir / f"shard_{idx:05d}.npz"

        if not shard_path.exists():
            raise FileNotFoundError(
                f"No se encuentra el bloque: {shard_path}"
            )

        with np.load(shard_path, allow_pickle=False) as file:
            vectors.append(file["embeddings"])
            keys.append(file["window_keys"])

    # Reconstruir matriz completa
    result = np.concatenate(vectors, axis=0)
    joined_keys = np.concatenate(keys, axis=0)

    # Validar cantidad y unicidad
    if (
        len(result) != settings["num_windows"]
        or len(set(joined_keys)) != len(joined_keys)
    ):
        raise ValueError(
            "Los embeddings recuperados están incompletos "
            "o contienen identificadores duplicados."
        )

    # Validar dimensiones
    if result.ndim != 2 or result.shape[1] != 768:
        raise ValueError(
            f"Dimensiones inesperadas: {result.shape}"
        )

    # Comprobar valores numéricos
    if not np.isfinite(result).all():
        raise ValueError(
            "Los embeddings contienen valores NaN o infinitos."
        )

    return result, joined_keys