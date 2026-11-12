"""División reproducible de conversaciones multietiqueta (70/15/15).

La unidad de partición es conversation_id, nunca la ventana.
Requiere: pip install iterative-stratification
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


def _validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Valida matriz de etiquetas y normaliza índices a texto."""
    if labels.empty or labels.shape[1] == 0:
        raise ValueError("La matriz de etiquetas está vacía")
    if labels.index.isna().any() or labels.index.has_duplicates:
        raise ValueError("conversation_id vacío o duplicado")
    data = labels.copy()
    data.index = data.index.map(str)
    if data.index.has_duplicates:
        raise ValueError("IDs duplicados tras convertirlos a texto")
    if data.isna().any().any() or not data.isin([0, 1]).all().all():
        raise ValueError("Las etiquetas deben ser binarias (0/1) sin ausencias")
    return data.astype(np.int8)


def create_splits(labels: pd.DataFrame, config: SplitConfig = SplitConfig()) -> pd.DataFrame:
    """Devuelve conversation_id, split; preserva la estratificación multietiqueta.

    La estratificación es aproximada: no garantiza positivos en cada clase y partición.
    """
    if any(x <= 0 for x in (config.train_ratio, config.val_ratio, config.test_ratio)):
        raise ValueError("Las tres proporciones deben ser positivas")
    if not np.isclose(config.train_ratio + config.val_ratio + config.test_ratio, 1.0):
        raise ValueError("Las proporciones deben sumar 1")
    data = _validate_labels(labels)
    if len(data) < 3:
        raise ValueError("Se necesitan al menos tres conversaciones")

    y = data.to_numpy(dtype=np.int8)
    indices = np.arange(len(data))
    stage1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1 - config.train_ratio, random_state=config.seed
    )
    train_pos, holdout_pos = next(stage1.split(np.zeros((len(data), 1)), y))

    holdout_y = y[holdout_pos]
    stage2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
        random_state=config.seed + 1,
    )
    val_local, test_local = next(
        stage2.split(np.zeros((len(holdout_pos), 1)), holdout_y)
    )

    assignments = np.empty(len(data), dtype=object)
    assignments[indices[train_pos]] = "train"
    assignments[holdout_pos[val_local]] = "val"
    assignments[holdout_pos[test_local]] = "test"
    result = pd.DataFrame({
        "conversation_id": data.index.to_numpy(), "split": assignments
    })
    validate_splits(result, data)
    return result


def validate_splits(splits: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Comprueba que cada conversación aparece una sola vez en un conjunto válido."""
    data = _validate_labels(labels)
    if not {"conversation_id", "split"}.issubset(splits.columns):
        raise ValueError("Faltan columnas conversation_id o split")
    ids = splits["conversation_id"].astype(str)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("IDs de partición duplicados")
    if set(ids) != set(data.index):
        raise ValueError("Las particiones no cubren exactamente las conversaciones anotadas")
    if not splits["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Nombre de partición desconocido")
    if set(splits["split"]) != {"train", "val", "test"}:
        raise ValueError("Alguna partición está vacía")


def split_report(splits: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Cuenta conversaciones y etiquetas positivas por partición."""
    data = _validate_labels(labels)
    validate_splits(splits, data)
    merged = splits.copy()
    merged["conversation_id"] = merged["conversation_id"].astype(str)
    merged = merged.join(data, on="conversation_id", validate="one_to_one")
    rows = []
    for name in ("train", "val", "test"):
        subset = merged.loc[merged["split"] == name]
        row = {"split": name, "n_conversations": len(subset),
               "pct_corpus": 100 * len(subset) / len(merged)}
        for bias in data.columns:
            row[f"{bias}_positive"] = int(subset[bias].sum())
            row[f"{bias}_pct"] = round(100 * subset[bias].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def save_splits(splits: pd.DataFrame, labels: pd.DataFrame,
                folder: str | Path, config: SplitConfig = SplitConfig()) -> None:
    """Guarda particiones y manifiesto sin sobreescribir una partición existente."""
    data = _validate_labels(labels)
    validate_splits(splits, data)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "conversation_splits.csv"
    manifest = folder / "split_manifest.json"
    if target.exists() or manifest.exists():
        raise FileExistsError("Ya existe una partición guardada; reutiliza load_splits()")
    report = split_report(splits, data)
    info = {"config": asdict(config), "biases": list(data.columns),
            "n_conversations": len(data), "report": report.to_dict(orient="records")}
    # Escribir CSV y manifiesto mediante nombres temporales y renombrado.
    csv_tmp = target.with_suffix(".csv.tmp")
    json_tmp = manifest.with_suffix(".json.tmp")
    try:
        splits.to_csv(csv_tmp, index=False)
        json_tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
        os.replace(csv_tmp, target)
        os.replace(json_tmp, manifest)
    finally:
        csv_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)


def load_splits(folder: str | Path, labels: pd.DataFrame) -> pd.DataFrame:
    """Recupera las particiones y valida cobertura, columnas y tamaño."""
    folder = Path(folder)
    manifest = json.loads((folder / "split_manifest.json").read_text(encoding="utf-8"))
    splits = pd.read_csv(folder / "conversation_splits.csv", dtype={"conversation_id": str})
    data = _validate_labels(labels)
    validate_splits(splits, data)
    if manifest["biases"] != list(data.columns) or manifest["n_conversations"] != len(data):
        raise ValueError("Manifiesto incompatible con la matriz de etiquetas")
    return splits


def select_bags(bag_ids: np.ndarray, conversation_ids: np.ndarray,
                splits: pd.DataFrame, partition: str) -> np.ndarray:
    """Máscara de ventanas pertenecientes a las conversaciones de una partición.

    bag_ids usa los índices de conversation_ids (orden de salida de
    align_embeddings_and_labels). Aplicar también a X; y se filtra por IDs de bolsa.
    """
    if partition not in {"train", "val", "test"}:
        raise ValueError("Partición desconocida")
    split_ids = set(splits.loc[splits["split"] == partition, "conversation_id"].astype(str))
    bag_ids = np.asarray(bag_ids)
    conversation_ids = np.asarray(conversation_ids).astype(str)
    if bag_ids.ndim != 1 or not np.issubdtype(bag_ids.dtype, np.integer):
        raise ValueError("bag_ids debe ser vector de enteros")
    if len(bag_ids) and (bag_ids.min() < 0 or bag_ids.max() >= len(conversation_ids)):
        raise ValueError("bag_ids fuera de rango")
    return np.isin(conversation_ids[bag_ids], list(split_ids))