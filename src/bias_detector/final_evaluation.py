"""Evaluación final multietiqueta de un clasificador MIL, sin ajuste de umbrales.

Todas las métricas se calculan a nivel CONVERSACIÓN. Los umbrales tienen que
proceder del conjunto de validación y permanecer fijos durante la evaluación.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, jaccard_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_multilabel(y_true, scores, thresholds, bias_names, beta=2.0):
    """Devuelve (reporte_por_sesgo, métricas_globales, matrices, predicciones).

    - Jaccard por sesgo = TP/(TP+FP+FN).
    - Jaccard samples: promedio del Jaccard entre conjuntos de etiquetas de cada
      conversación; para dos conjuntos vacíos se asigna 0 (zero_division=0).
    - Subset accuracy exige coincidencia exacta de las cuatro etiquetas.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    names = tuple(bias_names)
    if y.ndim != 2 or s.shape != y.shape or y.shape[0] == 0:
        raise ValueError("y_true y scores deben tener forma (n_conversaciones, n_sesgos).")
    if y.shape[1] != len(names) or thresholds.shape != (len(names),):
        raise ValueError("Incompatibilidad en el número u orden de sesgos/umbrales.")
    if len(set(names)) != len(names):
        raise ValueError("Nombres de sesgos duplicados.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Las etiquetas deben ser 0/1.")
    if not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError("Las puntuaciones deben estar entre 0 y 1 y ser finitas.")
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Los umbrales deben estar en [0,1].")
    if beta <= 1:
        raise ValueError("Para este estudio se requiere beta > 1.")
    y = y.astype(np.int8)
    pred = (s >= thresholds[None, :]).astype(np.int8)
    rows, matrices = [], {}
    for j, name in enumerate(names):
        yt, yp = y[:, j], pred[:, j]
        tn, fp, fn, tp = (int(v) for v in confusion_matrix(yt, yp, labels=[0, 1]).ravel())
        matrices[name] = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        rows.append({
            "bias": name, "threshold": float(thresholds[j]),
            "support_positive": int(yt.sum()), "predicted_positive": int(yp.sum()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall_sensitivity": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "f_beta": float(fbeta_score(yt, yp, beta=beta, zero_division=0)),
            "jaccard": float(jaccard_score(yt, yp, zero_division=0)),
            "average_precision": float(average_precision_score(yt, s[:, j])) if yt.sum() else float("nan"),
            "roc_auc": float(roc_auc_score(yt, s[:, j])) if len(np.unique(yt)) == 2 else float("nan"),
        })
    per_label = pd.DataFrame(rows)
    overall = {
        "n_conversations": int(len(y)), "n_labels": int(len(names)),
        "f_beta_beta": float(beta),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y, pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f_beta": float(fbeta_score(y, pred, beta=beta, average="macro", zero_division=0)),
        "micro_f_beta": float(fbeta_score(y, pred, beta=beta, average="micro", zero_division=0)),
        "weighted_f_beta": float(fbeta_score(y, pred, beta=beta, average="weighted", zero_division=0)),
        "jaccard_macro": float(jaccard_score(y, pred, average="macro", zero_division=0)),
        "jaccard_micro": float(jaccard_score(y, pred, average="micro", zero_division=0)),
        "jaccard_samples": float(jaccard_score(y, pred, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y, pred)),
        "exact_matches": int(np.all(y == pred, axis=1).sum()),
        "total_tp": int(per_label["tp"].sum()),
        "total_fp": int(per_label["fp"].sum()),
        "total_fn": int(per_label["fn"].sum()),
        "total_tn": int(per_label["tn"].sum()),
    }
    return per_label, overall, matrices, pred


def predictions_dataframe(conversation_ids, y_true, scores, predictions, bias_names):
    """Una fila por conversación, con verdad, score y decisión por sesgo."""
    ids = pd.Series(conversation_ids).astype(str).to_numpy()
    y, s, p = map(np.asarray, (y_true, scores, predictions))
    if len(ids) != len(y) or len(set(ids)) != len(ids) or y.shape != s.shape or y.shape != p.shape:
        raise ValueError("IDs únicos y matrices alineadas son obligatorios.")
    df = pd.DataFrame({"conversation_id": ids})
    for j, name in enumerate(bias_names):
        df[f"{name}_true"] = y[:, j].astype(np.int8)
        df[f"{name}_score"] = s[:, j].astype(np.float64)
        df[f"{name}_pred"] = p[:, j].astype(np.int8)
    return df


def save_final_evaluation(folder, per_label, overall, matrices, predictions_df, *, metadata):
    """Guarda evaluación completa sin sobrescribir una anterior."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    targets = [folder / n for n in (
        "test_metrics_by_bias.csv", "test_metrics_overall.json",
        "test_confusion_matrices.json", "test_predictions.csv", "evaluation_manifest.json"
    )]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Hay resultados anteriores en {folder}; utiliza otra carpeta.")
    per_label.to_csv(targets[0], index=False)
    targets[1].write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
    targets[2].write_text(json.dumps({k: v.tolist() for k, v in matrices.items()}, indent=2), encoding="utf-8")
    predictions_df.to_csv(targets[3], index=False, float_format="%.17g")
    targets[4].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return targets


def plot_confusion_matrices(matrices):
    """Dibuja matrices 2x2, con filas reales y columnas predichas."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)