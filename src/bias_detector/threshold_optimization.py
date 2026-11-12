# @title
# Módulo threshold_optimization
"""Optimización reproducible de umbrales multietiqueta a nivel conversación.

Utilizar únicamente predicciones de VALIDACIÓN para seleccionar umbrales.
Las puntuaciones deben corresponder a las mismas conversaciones y orden que y_true.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdConfig:
    # 'exact' considera los puntos de cambio de las predicciones (cada score distinto).
    # En un empate de F1, preferimos mayor precisión y luego el umbral más alto.
    method: str = "exact"


def _validate_arrays(y_true, scores, bias_names):
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    biases = tuple(bias_names)
    if y.ndim != 2 or s.ndim != 2 or y.shape != s.shape:
        raise ValueError(f"y_true y scores deben tener misma forma (n, k): {y.shape}, {s.shape}")
    if y.shape[0] == 0 or y.shape[1] != len(biases):
        raise ValueError("Número de sesgos incompatible o dataset vacío.")
    if len(set(biases)) != len(biases):
        raise ValueError("Nombres de sesgos duplicados.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Las etiquetas deben ser binarias 0/1.")
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("Las puntuaciones deben estar en [0,1] y ser finitas.")
    return y.astype(np.int8), s, biases


def _optimize_one(y, scores):
    """Optimización exacta de F1 en O(n log n) ordenando los scores.

    La regla de detección utilizada en todo el proyecto es score >= threshold.
    Si varios umbrales maximizan F1: más precisión y después umbral mayor.
    """
    n_pos = int(y.sum())
    n = len(y)
    if n_pos == 0 or n_pos == n:
        raise ValueError("Se necesitan ejemplos PRESENT y ABSENT en validación para cada sesgo.")

    order = np.argsort(-scores, kind="stable")
    sorted_s = scores[order]
    sorted_y = y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_s) != 0), n - 1]
    tp = np.cumsum(sorted_y)[ends].astype(np.int64)
    n_pred = ends + 1
    fp = n_pred - tp
    fn = n_pos - tp
    precision = tp / n_pred
    recall = tp / n_pos
    f1 = 2 * tp / (n_pred + n_pos)  # equivale a 2TP/(2TP+FP+FN)
    thresholds = sorted_s[ends]

    # Incluimos la opción de no emitir ningún positivo, con umbral > 1.
    # Con positivos reales su F1 siempre es 0, así que nunca será mejor que algún corte.
    best_f1 = float(f1.max())
    eligible = np.flatnonzero(np.isclose(f1, best_f1, rtol=1e-12, atol=1e-12))
    best_precision = precision[eligible].max()
    eligible = eligible[np.isclose(precision[eligible], best_precision, rtol=1e-12, atol=1e-12)]
    best = eligible[np.argmax(thresholds[eligible])]
    return float(thresholds[best])


def metrics_at_thresholds(y_true, scores, bias_names, thresholds):
    """Métricas por sesgo y agregadas usando umbrales FIJOS."""
    y, s, biases = _validate_arrays(y_true, scores, bias_names)
    t = np.asarray(thresholds, dtype=np.float64)
    if t.shape != (len(biases),) or not np.isfinite(t).all():
        raise ValueError("Se requiere un umbral finito por sesgo.")
    pred = (s >= t[None, :]).astype(np.int8)
    rows = []
    for j, bias in enumerate(biases):
        tn, fp, fn, tp = confusion_matrix(y[:, j], pred[:, j], labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[:, j], pred[:, j], average="binary", zero_division=0
        )
        rows.append({
            "bias": bias, "threshold": float(t[j]), "n_positive": int(y[:, j].sum()),
            "predicted_positive": int(pred[:, j].sum()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "average_precision": float(average_precision_score(y[:, j], s[:, j])) if y[:, j].sum() else np.nan,
            "roc_auc": float(roc_auc_score(y[:, j], s[:, j])) if np.unique(y[:, j]).size == 2 else np.nan,
        })
    report = pd.DataFrame(rows)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y.ravel(), pred.ravel(), average="binary", zero_division=0
    )
    overall = {
        "macro_precision": float(report["precision"].mean()),
        "macro_recall": float(report["recall"].mean()),
        "macro_f1": float(report["f1"].mean()),
        "micro_precision": float(micro_p), "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
    }
    return report, overall


def optimize_thresholds(y_val, val_probabilities, bias_names, config=None):
    """Selecciona umbrales por máximo F1 sobre validación; devuelve vector y reportes."""
    cfg = config or ThresholdConfig()
    if cfg.method != "exact":
        raise ValueError("Actualmente solo se implementa method='exact'.")
    y, s, biases = _validate_arrays(y_val, val_probabilities, bias_names)
    thresholds = np.array([_optimize_one(y[:, j], s[:, j]) for j in range(len(biases))])
    report, overall = metrics_at_thresholds(y, s, biases, thresholds)
    return thresholds, report, overall


def apply_thresholds(probabilities, thresholds):
    """Devuelve matriz binaria de predicciones, sin modificar umbrales."""
    s = np.asarray(probabilities, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    if s.ndim != 2 or t.shape != (s.shape[1],):
        raise ValueError("Dimensiones incompatibles entre puntuaciones y umbrales.")
    if not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError("Se encontraron puntuaciones o umbrales no finitos.")
    return (s >= t[None, :]).astype(np.int8)


def save_thresholds(folder, thresholds, bias_names, report, overall, *, metadata=None):
    """Guarda los umbrales y métricas de validación. Rechaza sobrescrituras."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / "thresholds.json", folder / "validation_metrics.csv"]
    if any(p.exists() for p in paths):
        raise FileExistsError(f"Ya existe una optimización en {folder}. Usa otra carpeta.")
    biases = list(bias_names)
    t = np.asarray(thresholds, dtype=float)
    if len(biases) != len(t):
        raise ValueError("Número de umbrales distinto al de sesgos.")
    payload = {
        "selection_dataset": "validation", "selection_metric": "per_label_f1",
        "decision_rule": "score >= threshold", "tie_break": "higher_precision_then_higher_threshold",
        "biases": biases,
        "thresholds": {b: float(v) for b, v in zip(biases, t)},
        "overall_validation_metrics": overall,
        "metadata": metadata or {},
    }
    paths[0].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report.to_csv(paths[1], index=False)
    return paths


def load_thresholds(folder, bias_names):
    """Carga umbrales garantizando el orden esperado de sesgos."""
    payload = json.loads((Path(folder) / "thresholds.json").read_text(encoding="utf-8"))
    expected = list(bias_names)
    if payload["biases"] != expected:
        raise ValueError(f"Orden de sesgos incompatible: {payload['biases']} vs {expected}")
    return np.array([payload["thresholds"][b] for b in expected], dtype=float), payload


def plot_precision_recall(y_true, scores, bias_names, thresholds):
    """Curvas PR de validación con el punto seleccionado en cada sesgo."""
    import matplotlib.pyplot as plt
    y, s, biases = _validate_arrays(y_true, scores, bias_names)
    t = np.asarray(thresholds)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for j, ax in enumerate(axes.flat):
        p, r, _ = precision_recall_curve(y[:, j], s[:, j])
        ax.plot(r, p, label="Curva PR")
        y_pred = (s[:, j] >= t[j]).astype(np.int8)
        tp = int(((y[:, j] == 1) & (y_pred == 1)).sum())
        predicted = int(y_pred.sum())
        selected_p = tp / predicted if predicted else 0.0
        selected_r = tp / max(int(y[:, j].sum()), 1)
        ax.scatter([selected_r], [selected_p], label=f"Umbral={t[j]:.4f}", zorder=3)
        ax.axhline(y[:, j].mean(), linestyle=":", label="Prevalencia")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title=biases[j])
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(y_true, scores, bias_names, thresholds):
    """Matrices de confusión con los umbrales seleccionados."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    y, s, biases = _validate_arrays(y_true, scores, bias_names)
    pred = apply_thresholds(s, thresholds)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for j, ax in enumerate(axes.flat):
        cm = confusion_matrix(y[:, j], pred[:, j], labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["ABSENT", "PRESENT"]).plot(
            ax=ax, colorbar=False, values_format="d"
        )
        ax.set_title(biases[j])
    fig.tight_layout()
    return fig