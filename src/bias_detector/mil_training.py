"""Entrenamiento de clasificadores MIL sobre embeddings congelados."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 12
    min_delta: float = 1e-4
    seed: int = 42
    weight_strategy: str = "sqrt"
    grad_clip: float = 1.0
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. PREPARACIÓN DE DATOS
# ============================================================

def partition_arrays(
    X: np.ndarray,
    bag_ids: np.ndarray,
    y: np.ndarray,
    conversation_ids: np.ndarray,
    splits: pd.DataFrame,
    split: str
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:

    """Extrae las conversaciones correspondientes a una partición."""

    if split not in ("train", "val", "test"):
        raise ValueError("Partición desconocida")

    X = np.asarray(X, dtype=np.float32)
    bags = np.asarray(bag_ids, dtype=np.int64)
    y = np.asarray(y, dtype=np.float32)
    ids = np.asarray(conversation_ids).astype(str)

    if X.ndim != 2 or bags.ndim != 1 or len(X) != len(bags):
        raise ValueError("Forma inconsistente de embeddings o bag_ids")

    if y.ndim != 2 or len(y) != len(ids):
        raise ValueError("y y conversation_ids deben estar alineados")

    if (
        not np.isfinite(X).all()
        or not np.isfinite(y).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Datos no finitos o etiquetas no binarias")

    if len(bags) == 0 or bags.min() < 0 or bags.max() >= len(ids):
        raise ValueError("bag_ids fuera de rango")

    if set(np.unique(bags)) != set(range(len(ids))):
        raise ValueError("Alguna conversación carece de ventanas")

    if splits["conversation_id"].astype(str).duplicated().any():
        raise ValueError("conversation_id duplicado en splits")

    selected = set(
        splits.loc[
            splits["split"] == split,
            "conversation_id"
        ].astype(str)
    )

    indices = [
        i for i, cid in enumerate(ids)
        if cid in selected
    ]

    if len(indices) != len(selected):
        raise ValueError(f"Faltan conversaciones en partición {split}")

    groups = [
        X[bags == i].copy()
        for i in indices
    ]

    if any(len(g) == 0 for g in groups):
        raise ValueError("Bolsa vacía")

    return (
        groups,
        y[indices].copy(),
        ids[indices].copy()
    )


class BagDataset(Dataset):
    """Dataset de conversaciones representadas por múltiples ventanas."""

    def __init__(
        self,
        groups: list[np.ndarray],
        labels: np.ndarray
    ):

        if len(groups) == 0 or len(groups) != len(labels):
            raise ValueError("Número de bolsas y etiquetas inválido")

        self.groups = [
            torch.as_tensor(g, dtype=torch.float32)
            for g in groups
        ]

        self.labels = torch.as_tensor(
            labels,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx], self.labels[idx]


def collate_bags(batch):
    """Construye lotes de conversaciones y sus ventanas."""

    arrays, labels = zip(*batch)

    X = torch.cat(arrays, dim=0)

    bag_ids = torch.repeat_interleave(
        torch.arange(len(arrays), dtype=torch.long),
        torch.tensor(
            [len(a) for a in arrays],
            dtype=torch.long
        )
    )

    return X, bag_ids, torch.stack(labels)


# ============================================================
# 3. PONDERACIÓN DE CLASES
# ============================================================

def compute_pos_weight(
    train_labels: np.ndarray,
    strategy: str = "sqrt"
) -> torch.Tensor:

    """
    Calcula los pesos de las clases positivas usando
    exclusivamente el conjunto de entrenamiento.

    Estrategias:
        none:    peso = 1
        quarter: peso = (negativos / positivos) ** 0.25
        sqrt:    peso = (negativos / positivos) ** 0.5
        full:    peso = negativos / positivos
    """

    y = np.asarray(train_labels, dtype=np.float32)

    positive = y.sum(axis=0)
    negative = len(y) - positive

    if np.any(positive == 0):
        raise ValueError(
            "Hay sesgos sin positivos en entrenamiento"
        )

    ratio = negative / positive

    if strategy == "none":
        values = np.ones_like(ratio)

    elif strategy == "quarter":
        values = np.power(ratio, 0.25)

    elif strategy == "sqrt":
        values = np.sqrt(ratio)

    elif strategy == "full":
        values = ratio

    else:
        raise ValueError(
            "weight_strategy debe ser: "
            "none, quarter, sqrt o full"
        )

    return torch.tensor(values, dtype=torch.float32)


# ============================================================
# 4. ENTRENAMIENTO Y VALIDACIÓN POR ÉPOCA
# ============================================================

def _epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
    grad_clip=1.0
):

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total = 0

    for X, bags, targets in loader:

        X = X.to(device)
        bags = bags.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            logits = model(X, bags)

            if logits.shape != targets.shape:
                raise ValueError(
                    f"Forma de logits {logits.shape} "
                    f"!= etiquetas {targets.shape}"
                )

            loss = loss_fn(logits, targets)

            if training:
                loss.backward()

                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip
                    )

                optimizer.step()

        total_loss += float(loss.detach()) * len(targets)
        total += len(targets)

    return total_loss / total


def _atomic_save_torch(obj, path: Path):
    """Guarda el checkpoint de forma atómica."""

    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)

    finally:
        tmp.unlink(missing_ok=True)


# ============================================================
# 5. ENTRENAMIENTO PRINCIPAL
# ============================================================

def train_mil(
    model: nn.Module,
    train_groups: list[np.ndarray],
    y_train: np.ndarray,
    val_groups: list[np.ndarray],
    y_val: np.ndarray,
    output_dir: str | Path,
    config: TrainingConfig = TrainingConfig(),
    biases: tuple[str, ...] = (
        "anchoring_bias",
        "availability_bias",
        "confirmation_bias",
        "framing_effect"
    )
) -> pd.DataFrame:

    """
    Entrena el clasificador MIL y guarda el mejor modelo
    según la pérdida de validación.

    Incluye:
        - Ponderación configurable.
        - AdamW.
        - Gradient clipping.
        - Early stopping.
        - Checkpoint del mejor modelo.
        - Historial de entrenamiento.
    """

    if (
        len(biases) != y_train.shape[1]
        or y_val.shape[1] != len(biases)
    ):
        raise ValueError(
            "Incompatibilidad entre número de sesgos y etiquetas"
        )

    if (
        config.epochs < 1
        or config.batch_size < 1
        or config.patience < 1
    ):
        raise ValueError(
            "epochs, batch_size y patience deben ser positivos"
        )

    set_seed(config.seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = out / "best_model.pt"
    history_file = out / "training_history.csv"

    if checkpoint.exists() or history_file.exists():
        raise FileExistsError(
            "Ya existe un entrenamiento en output_dir. "
            "Utiliza otra carpeta para preservar resultados."
        )

    # Datasets
    train_dataset = BagDataset(train_groups, y_train)
    val_dataset = BagDataset(val_groups, y_val)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_bags,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_bags,
        num_workers=0
    )

    device = torch.device(config.device)
    model = model.to(device)

    # Ponderación de clases
    weights = compute_pos_weight(
        y_train,
        config.weight_strategy
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=weights
    )

    # Optimizador
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    history = []
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, config.epochs + 1):

        train_loss = _epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            config.grad_clip
        )

        val_loss = _epoch(
            model,
            val_loader,
            loss_fn,
            device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.5f} | "
            f"val={val_loss:.5f}"
        )

        # Guardar mejor modelo
        if val_loss < best_loss - config.min_delta:

            best_loss = val_loss
            bad_epochs = 0

            _atomic_save_torch(
                {
                    "model_state_dict": copy.deepcopy(
                        model.state_dict()
                    ),
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "biases": list(biases),
                    "training_config": asdict(config),
                    "pos_weight": weights.detach().cpu()
                },
                checkpoint
            )

        else:
            bad_epochs += 1

        # Guardar historial
        pd.DataFrame(history).to_csv(
            history_file,
            index=False
        )

        # Early stopping
        if bad_epochs >= config.patience:

            print(
                f"Early stopping: "
                f"{config.patience} épocas sin mejora."
            )

            break

    return pd.DataFrame(history)


# ============================================================
# 6. RECUPERACIÓN DEL MEJOR MODELO
# ============================================================

def load_best_model(
    model: nn.Module,
    output_dir: str | Path,
    device: str = "cpu"
) -> tuple[nn.Module, dict]:

    """Recupera los mejores pesos del modelo entrenado."""

    checkpoint_path = (
        Path(output_dir) / "best_model.pt"
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.to(device).eval()

    checkpoint_info = {
        k: v
        for k, v in ckpt.items()
        if k != "model_state_dict"
    }

    return model, checkpoint_info


# ============================================================
# 7. PREDICCIÓN
# ============================================================

def predict_bags(
    model: nn.Module,
    groups: list[np.ndarray],
    batch_size: int = 32,
    device: str = "cpu"
) -> np.ndarray:

    """
    Genera puntuaciones para las conversaciones.

    Devuelve una matriz:
        [n_conversations, n_biases]

    Conserva el orden original de groups.
    """

    if not groups:
        raise ValueError("No hay bolsas")

    dummy = np.zeros(
        (len(groups), 1),
        dtype=np.float32
    )

    loader = DataLoader(
        BagDataset(groups, dummy),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_bags
    )

    model.to(device).eval()

    chunks = []

    with torch.inference_mode():

        for X, bag_ids, _ in loader:

            logits = model(
                X.to(device),
                bag_ids.to(device)
            )

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

            chunks.append(probabilities)

    return np.concatenate(chunks, axis=0)
