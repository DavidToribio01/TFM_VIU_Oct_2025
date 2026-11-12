"""Clasificador multietiqueta Multiple Instance Learning (MIL) por max pooling.

Longformer se utiliza previamente como extractor congelado de embeddings.
Este módulo no realiza entrenamiento ni selecciona umbrales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class MILConfig:
    embedding_dim: int = 768
    biases: tuple[str, ...] = (
        "anchoring_bias",
        "availability_bias",
        "confirmation_bias",
        "framing_effect",
    )



import torch
import torch.nn as nn


class MaxMILClassifier(nn.Module):
    """
    Clasificador MIL multietiqueta con MLP.

    Entrada:
        embeddings: Tensor [n_windows, 768]
        bag_ids: Tensor [n_windows]

    Salida:
        logits: Tensor [n_conversations, 4]
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        bottleneck_dim=64,
        num_labels=4,
        dropout=0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(bottleneck_dim, num_labels)
        )

    def forward(self, embeddings, bag_ids):

        # Puntuaciones por ventana
        window_logits = self.classifier(embeddings)

        # Agregación máxima por conversación y sesgo
        unique_bags, inverse = torch.unique(
            bag_ids,
            sorted=True,
            return_inverse=True
        )

        bag_logits = []

        for idx in range(len(unique_bags)):
            mask = inverse == idx

            bag_logits.append(
                window_logits[mask].max(dim=0).values
            )

        return torch.stack(bag_logits, dim=0)


def build_label_matrix(
    annotations: pd.DataFrame,
    biases: Sequence[str] = MILConfig().biases,
) -> pd.DataFrame:
    """Una fila por conversación y una columna 0/1 por sesgo.

    Exige cobertura completa: una anotación PRESENT/ABSENT por par conversación-sesgo.
    """
    required = {"conversation_id", "bias", "final_label"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Faltan columnas: {required - set(annotations.columns)}")
    data = annotations[["conversation_id", "bias", "final_label"]].copy()
    if data["conversation_id"].isna().any():
        raise ValueError("Hay conversation_id vacíos")
    if data.duplicated(["conversation_id", "bias"]).any():
        raise ValueError("Hay anotaciones duplicadas para conversación y sesgo")
    if not data["final_label"].isin(["PRESENT", "ABSENT"]).all():
        raise ValueError("Etiquetas diferentes de PRESENT/ABSENT")
    unknown = set(data["bias"]) - set(biases)
    if unknown:
        raise ValueError(f"Sesgos desconocidos: {sorted(unknown)}")
    data["label"] = (data["final_label"] == "PRESENT").astype(np.int64)
    matrix = data.pivot(index="conversation_id", columns="bias", values="label")
    matrix = matrix.reindex(columns=list(biases))
    if matrix.isna().any().any():
        raise ValueError("Faltan anotaciones para uno o varios sesgos en ciertas conversaciones")
    return matrix.astype(np.float32)


def align_embeddings_and_labels(
    embeddings: np.ndarray,
    window_keys: np.ndarray,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alinea ventanas y etiquetas por conversation_id.

    window_keys debe seguir el convenio '<conversation_id>::<window_id>'.
    Devuelve X, bag_ids, y, conversation_ids, manteniendo el orden de primera
    aparición de las conversaciones en window_keys.
    """
    X = np.asarray(embeddings, dtype=np.float32)
    keys = np.asarray(window_keys).astype(str)
    if X.ndim != 2 or len(X) != len(keys):
        raise ValueError("Embeddings y window_keys no están alineados")
    if len(set(keys)) != len(keys):
        raise ValueError("Hay window_keys duplicadas")
    if not np.isfinite(X).all():
        raise ValueError("Embeddings con valores no finitos")
    if any("::" not in key for key in keys):
        raise ValueError("Formato de window_keys inválido")
    conversation_ids = [key.rsplit("::", 1)[0] for key in keys]
    if labels.index.has_duplicates:
        raise ValueError("La matriz de etiquetas tiene IDs duplicados")
    label_lookup = labels.copy()
    label_lookup.index = label_lookup.index.map(str)
    if label_lookup.index.has_duplicates:
        raise ValueError("IDs duplicados tras convertirlos a texto")
    unique_conversations = list(dict.fromkeys(conversation_ids))
    missing = set(unique_conversations) - set(label_lookup.index)
    extra = set(label_lookup.index) - set(unique_conversations)
    if missing or extra:
        raise ValueError(f"Desajuste IDs: sin etiqueta={len(missing)}, sin ventanas={len(extra)}")
    ids_to_bag = {cid: i for i, cid in enumerate(unique_conversations)}
    bag_ids = np.array([ids_to_bag[cid] for cid in conversation_ids], dtype=np.int64)
    y = label_lookup.loc[unique_conversations].to_numpy(dtype=np.float32)
    return X, bag_ids, y, np.asarray(unique_conversations)
