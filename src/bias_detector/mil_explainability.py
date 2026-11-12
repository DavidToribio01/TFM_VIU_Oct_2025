# @title
# Módulo mil_explainability

"""Etapa 1 de explicabilidad MIL: localización de la ventana dominante por sesgo.

No calcula SHAP y no utiliza las anotaciones de referencia.
Entradas: embeddings ya extraídos, claves de ventana, tabla de ventanas,
checkpoint de la MLP-MIL y umbrales seleccionados en validación.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_BIASES = (
    "anchoring_bias", "availability_bias", "confirmation_bias", "framing_effect"
)


class MLPMaxMILClassifier(nn.Module):
    """Misma topología (y nombres de capas) que la MLP-MIL entrenada.

    Este modelo se incluye para poder reconstruir los checkpoints en un notebook
    nuevo; si la clase utilizada en el entrenamiento difiere, hay que usar esa.
    """

    def __init__(self, input_dim=768, hidden_dim=256, bottleneck_dim=64,
                 num_labels=4, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_labels),
        )

    def forward(self, embeddings: torch.Tensor, bag_ids: torch.Tensor) -> torch.Tensor:
        window_logits = self.classifier(embeddings)
        unique_bags, inverse = torch.unique(bag_ids, sorted=True, return_inverse=True)
        return torch.stack([
            window_logits[inverse == i].max(dim=0).values
            for i in range(len(unique_bags))
        ], dim=0)


def load_mil_model(checkpoint_dir: str | Path, *, device: str = "cpu",
                   biases: Sequence[str] = DEFAULT_BIASES):
    """Carga estrictamente el checkpoint v3 en la topología MLP conocida."""
    path = Path(checkpoint_dir) / "best_model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved_biases = tuple(checkpoint["biases"])
    if saved_biases != tuple(biases):
        raise ValueError(f"Orden de sesgos distinto: {saved_biases} vs {tuple(biases)}")
    if checkpoint.get("training_config", {}).get("weight_strategy") != "quarter":
        raise ValueError("El checkpoint no corresponde a la estrategia 'quarter' (v3).")
    model = MLPMaxMILClassifier(num_labels=len(biases))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _align_window_metadata(window_keys: np.ndarray, df_windows: pd.DataFrame) -> pd.DataFrame:
    """Ordena metadatos en el orden exacto de embeddings; rechaza desajustes."""
    required = {"conversation_id", "window_id", "window_text", "turn_ids",
                "token_start", "token_end", "char_start", "char_end"}
    missing = required - set(df_windows.columns)
    if missing:
        raise ValueError(f"Columnas ausentes: {sorted(missing)}")
    keys = np.asarray(window_keys).astype(str)
    if len(keys) == 0 or len(set(keys)) != len(keys):
        raise ValueError("window_keys vacías o duplicadas")
    df = df_windows.copy()
    df["conversation_id"] = df["conversation_id"].astype(str)
    df["window_id"] = df["window_id"].astype(int)
    df["window_key"] = df["conversation_id"] + "::" + df["window_id"].astype(str)
    if df["window_key"].duplicated().any():
        raise ValueError("Ventanas duplicadas")
    aligned = df.set_index("window_key", drop=False).reindex(keys)
    if aligned["conversation_id"].isna().any() or len(aligned) != len(df):
        raise ValueError("Claves de ventanas y embeddings no coinciden exactamente")
    return aligned.reset_index(drop=True)


def compute_window_scores(model: nn.Module, embeddings: np.ndarray,
                          *, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """Devuelve puntuaciones sigmoides [n_ventanas, n_sesgos].

    El pooling MIL se aplica después, no aquí. Dropout desactivado con eval().
    """
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 768 or not np.isfinite(X).all():
        raise ValueError("Embeddings inválidos o dimensión distinta de 768")
    if batch_size < 1:
        raise ValueError("batch_size debe ser positivo")
    model = model.to(device).eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            logits = model.classifier(batch)
            pieces.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(pieces, axis=0)


def locate_positive_windows(
    window_scores: np.ndarray,
    window_keys: np.ndarray,
    df_windows: pd.DataFrame,
    thresholds: Sequence[float],
    biases: Sequence[str] = DEFAULT_BIASES,
    *,
    selected_conversation_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identifica ventana dominante por sesgo y detecciones positivas.

    Devuelve:
      conversation_scores: una fila por conversación, con *_score y *_pred;
      positive_windows: una fila por par (conversación, sesgo) positivo,
        ventana ganadora, texto y metadatos para su posterior explicación.

    Las puntuaciones de conversación se obtienen con max por sesgo, coincidiendo
    con MIL. Las posiciones originales de ventana no se alteran.
    """
    scores = np.asarray(window_scores, dtype=np.float64)
    bias_names = tuple(biases)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(bias_names):
        raise ValueError("Forma de puntuaciones incompatible con los sesgos")
    if thresholds.shape != (len(bias_names),) or not np.isfinite(thresholds).all():
        raise ValueError("Umbrales incompatibles")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Las puntuaciones deben estar en [0, 1]")
    aligned = _align_window_metadata(window_keys, df_windows)
    if len(aligned) != len(scores):
        raise ValueError("Número de embeddings y ventanas distinto")
    if selected_conversation_ids is not None:
        requested = set(map(str, selected_conversation_ids))
        available = set(aligned["conversation_id"])
        if requested - available:
            raise ValueError(f"Conversaciones no encontradas: {len(requested - available)}")
    else:
        requested = None

    by_id = aligned.groupby("conversation_id", sort=False).indices
    summary, winners = [], []
    for conv_id, row_indices in by_id.items():
        if requested is not None and conv_id not in requested:
            continue
        row_indices = np.asarray(row_indices, dtype=int)
        local_scores = scores[row_indices]
        maxima = local_scores.max(axis=0)
        record = {"conversation_id": conv_id, "n_windows": len(row_indices)}
        for j, bias in enumerate(bias_names):
            prediction = int(maxima[j] >= thresholds[j])
            record[f"{bias}_score"] = float(maxima[j])
            record[f"{bias}_pred"] = prediction
            if prediction:
                local_idx = int(np.argmax(local_scores[:, j]))
                global_idx = int(row_indices[local_idx])
                window = aligned.iloc[global_idx]
                winners.append({
                    "conversation_id": conv_id,
                    "bias": bias,
                    "score": float(maxima[j]),
                    "threshold": float(thresholds[j]),
                    "window_id": int(window["window_id"]),
                    "window_key": str(window["window_key"]),
                    "embedding_index": global_idx,
                    "n_windows": len(row_indices),
                    "turn_ids": list(window["turn_ids"]),
                    "token_start": int(window["token_start"]),
                    "token_end": int(window["token_end"]),
                    "char_start": int(window["char_start"]),
                    "char_end": int(window["char_end"]),
                    "window_text": window["window_text"],
                })
        summary.append(record)
    return pd.DataFrame(summary), pd.DataFrame(winners)


def verify_against_saved_predictions(
    conversation_scores: pd.DataFrame,
    predictions_csv: str | Path,
    biases: Sequence[str] = DEFAULT_BIASES,
    *,
    atol: float = 1e-5,
) -> None:
    """Verifica scores y decisiones frente al CSV inmutable de prueba.

    Permite pasar a locate_positive_windows solo el subconjunto de test,
    evitando comparar entren/val con el archivo de test.
    """
    saved = pd.read_csv(predictions_csv, dtype={"conversation_id": str})
    calculated = conversation_scores.copy()
    calculated["conversation_id"] = calculated["conversation_id"].astype(str)
    if calculated["conversation_id"].duplicated().any() or saved["conversation_id"].duplicated().any():
        raise ValueError("ID de conversación repetido")
    if set(calculated["conversation_id"]) != set(saved["conversation_id"]):
        raise ValueError("El conjunto de conversaciones no coincide con el CSV de prueba")
    merged = calculated.merge(saved, on="conversation_id", suffixes=("_computed", "_saved"), validate="one_to_one")
    for bias in biases:
        for suffix in ("score", "pred"):
            left = merged[f"{bias}_{suffix}_computed"].to_numpy(dtype=float)
            right = merged[f"{bias}_{suffix}_saved"].to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=0, atol=atol if suffix == "score" else 0):
                raise ValueError(f"Desajuste con evaluación final: {bias}_{suffix}")


def save_window_attributions(output_dir: str | Path, conversation_scores: pd.DataFrame,
                             positive_windows: pd.DataFrame) -> tuple[Path, Path]:
    """Guarda trazabilidad MIL (no SHAP), sin sobrescribir resultados anteriores."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "conversation_scores_mil.csv"
    winner_path = output / "positive_windows_mil.parquet"
    if score_path.exists() or winner_path.exists():
        raise FileExistsError("Ya existen resultados de localización MIL en esa carpeta")
    conversation_scores.to_csv(score_path, index=False)
    positive_windows.to_parquet(winner_path,index=False)
    return score_path, winner_path