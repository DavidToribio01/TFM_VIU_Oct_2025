# @title
"""Inferencia retrospectiva de extremo a extremo para el TFM.

Conversación [TURN n | USER/LLM] -> segmentación -> Longformer congelado ->
MLP-MIL v3 -> umbrales validados -> SHAP por detección -> alertas.

No consulta etiquetas de referencia, no entrena, ni modifica umbrales.

Requiere:
    segmentation.py
    embedding_extractor.py
    mil_explainability.py
    text_shap.py
    threshold_optimization.py
    alert_generator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping
import json

import numpy as np
import pandas as pd
import torch

# from segmentation import (
#     SegmentationConfig,
#     create_tokenizer,
#     segment_conversation,
# )

# from embedding_extractor import (
#     load_encoder,
#     _encode_batch,
# )

# from mil_explainability import (
#     DEFAULT_BIASES,
#     load_mil_model,
#     compute_window_scores,
# )

# from threshold_optimization import load_thresholds

# from text_shap import (
#     ShapTextConfig,
#     MILTextShapExplainer,
# )

# from alert_generator import (
#     BIASES,
#     generate_alerts,
# )


from .segmentation import SegmentationConfig, create_tokenizer, segment_conversation
from .embedding_extractor import load_encoder, _encode_batch
from .mil_explainability import DEFAULT_BIASES, load_mil_model, compute_window_scores
from .threshold_optimization import load_thresholds
from .text_shap import ShapTextConfig, MILTextShapExplainer
from .alert_generator import BIASES, generate_alerts

@dataclass(frozen=True)
class PipelineConfig:
    """Configuración del flujo de inferencia."""

    model_name: str = "allenai/longformer-base-4096"

    # Segmentación
    max_length: int = 2048
    overlap: int = 128
    min_boundary_fraction: float = 0.75

    # Explicabilidad
    shap_max_features: int = 8
    shap_nsamples: int = 64
    shap_batch_size: int = 1

    # Inferencia
    use_fp16: bool = True
    embedding_batch_size: int = 1

    # Validaciones
    score_tolerance: float = 0.02
    shap_additivity_tolerance: float = 0.02

    # Protección operativa: no trunca conversaciones.
    max_windows: int = 64


class BiasDetectionPipeline:
    """
    Orquesta todos los componentes del sistema.

    Carga una sola vez:
        - Modelo MLP-MIL v3.
        - Umbrales optimizados en validación.
        - Tokenizador Longformer.
        - Codificador Longformer congelado.

    Para cada conversación:
        1. Segmenta el texto.
        2. Extrae los embeddings.
        3. Calcula las puntuaciones MIL.
        4. Detecta los sesgos que superan su umbral.
        5. Calcula SHAP para cada detección.
        6. Genera alertas explicables.

    Si una explicación SHAP falla, no devuelve
    un resultado parcial.

    La instancia no está diseñada para procesar
    peticiones concurrentes.
    """

    def __init__(
        self,
        experiment_dir: str | Path,
        *,
        config: PipelineConfig = PipelineConfig(),
        device: str | None = None,
    ):

        self.config = config

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if (
            self.device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "Se solicitó CUDA, pero no está disponible."
            )

        # Verificar el orden de las etiquetas
        if tuple(DEFAULT_BIASES) != tuple(BIASES):
            raise ValueError(
                "Orden de etiquetas incoherente entre módulos."
            )

        # Validar configuración operativa
        if (
            config.max_windows < 1
            or config.embedding_batch_size < 1
        ):
            raise ValueError(
                "max_windows y embedding_batch_size "
                "deben ser positivos."
            )

        if (
            config.score_tolerance < 0
            or config.shap_additivity_tolerance < 0
        ):
            raise ValueError(
                "Las tolerancias no pueden ser negativas."
            )

        # --------------------------------------------------
        # 1. Configuración del segmentador
        # --------------------------------------------------

        self.segment_config = SegmentationConfig(
            model_name=config.model_name,
            max_length=config.max_length,
            overlap=config.overlap,
            min_boundary_fraction=config.min_boundary_fraction,
        )

        self.tokenizer = create_tokenizer(
            self.segment_config
        )

        # --------------------------------------------------
        # 2. Recuperar MLP-MIL v3
        # --------------------------------------------------

        self.mlp, self.checkpoint = load_mil_model(
            experiment_dir,
            device=self.device,
            biases=BIASES,
        )

        # --------------------------------------------------
        # 3. Recuperar umbrales definitivos
        # --------------------------------------------------

        threshold_values, self.threshold_metadata = (
            load_thresholds(
                Path(experiment_dir) / "thresholds",
                bias_names=BIASES,
            )
        )

        if (
            self.threshold_metadata.get("selection_dataset")
            != "validation"
        ):
            raise ValueError(
                "Los umbrales no proceden de validación."
            )

        if isinstance(threshold_values, Mapping):
            self.thresholds = {
                bias: float(threshold_values[bias])
                for bias in BIASES
            }
        else:
            values = np.asarray(threshold_values).reshape(-1)

            if len(values) != len(BIASES):
                raise ValueError(
                    "Número de umbrales incompatible con las etiquetas"
                )

            self.thresholds = {
                bias: float(value)
                for bias, value in zip(BIASES, values)
            }

        # --------------------------------------------------
        # 4. Cargar Longformer congelado
        # --------------------------------------------------

        self.encoder, self.device = load_encoder(
            config.model_name,
            device=self.device,
        )

    def run(
        self,
        *,
        conversation_id: str,
        conversation_text: str,
    ) -> dict[str, Any]:
        """
        Procesa una conversación finalizada.

        Devuelve un resultado completo únicamente cuando
        todas las explicaciones requeridas son válidas.

        Si no hay detecciones:
            status = "no_alerts"
            alerts = []

        Si hay detecciones:
            status = "alerts_generated"
            alerts contiene una alerta por sesgo positivo.

        Si SHAP falla:
            Propaga la excepción sin generar alertas
            parcialmente explicadas.
        """

        # ==================================================
        # 1. Validación de entrada
        # ==================================================

        cid = str(conversation_id).strip()

        if not cid:
            raise ValueError(
                "conversation_id no puede estar vacío."
            )

        if (
            not isinstance(conversation_text, str)
            or not conversation_text.strip()
        ):
            raise ValueError(
                "conversation_text debe ser una cadena no vacía."
            )

        # ==================================================
        # 2. Segmentación de la conversación
        # ==================================================

        window_rows = segment_conversation(
            cid,
            conversation_text,
            self.tokenizer,
            self.segment_config,
        )

        if len(window_rows) > self.config.max_windows:
            raise ValueError(
                f"La conversación produce "
                f"{len(window_rows)} ventanas; "
                f"límite operativo: "
                f"{self.config.max_windows}. "
                "No se ha truncado la conversación."
            )

        windows = pd.DataFrame(window_rows)

        # ==================================================
        # 3. Extracción de embeddings
        # ==================================================

        embedding_chunks = [
            _encode_batch(
                self.encoder,
                windows.iloc[
                    start:
                    start + self.config.embedding_batch_size
                ],
                self.device,
                self.config.use_fp16,
            )

            for start in range(
                0,
                len(windows),
                self.config.embedding_batch_size,
            )
        ]

        embeddings = np.concatenate(
            embedding_chunks,
            axis=0,
        ).astype(np.float32)

        window_keys = np.array(
            [
                f"{cid}::{int(w)}"
                for w in windows["window_id"]
            ],
            dtype=str,
        )

        # Validar forma e integridad
        if (
            embeddings.shape != (len(windows), 768)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(
                "Embeddings nuevos inválidos."
            )

        # ==================================================
        # 4. Clasificación mediante MLP-MIL
        # ==================================================

        window_scores = compute_window_scores(
            self.mlp,
            embeddings,
            device=self.device,
        )

        # Ventana dominante para cada sesgo
        winning_indices = np.argmax(
            window_scores,
            axis=0,
        )

        # Máximo MIL para cada sesgo
        maxima = np.max(
            window_scores,
            axis=0,
        )

        scores = {
            bias: float(maxima[j])
            for j, bias in enumerate(BIASES)
        }

        # ==================================================
        # 5. Aplicación de los umbrales
        # ==================================================

        positive_biases = [
            bias
            for bias in BIASES
            if scores[bias] >= self.thresholds[bias]
        ]

        winners = {
            bias: int(
                windows.iloc[
                    int(winning_indices[j])
                ]["window_id"]
            )

            for j, bias in enumerate(BIASES)
        }

        explanations = {}
        fragments_by_bias = {}

        # ==================================================
        # 6. SHAP para cada sesgo positivo
        # ==================================================

        if positive_biases:

            explain_config = ShapTextConfig(
                model_name=self.config.model_name,
                max_features=self.config.shap_max_features,
                nsamples=self.config.shap_nsamples,
                batch_size=self.config.shap_batch_size,
                fp16=self.config.use_fp16,
            )

            # Reutilizar el mismo Longformer cargado
            # para evitar duplicar memoria en la GPU.
            explainer = MILTextShapExplainer(
                mlp=self.mlp,
                windows=windows,
                embeddings=embeddings,
                window_keys=window_keys,
                config=explain_config,
                device=self.device,
                encoder=self.encoder,
            )

            for bias in positive_biases:

                # Obtener atribuciones y metadatos
                attrs, metadata = explainer.explain(
                    conversation_id=cid,
                    bias=bias,
                    bias_names=BIASES,
                    original_text=conversation_text,
                    winning_window_id=winners[bias],
                )

                # ------------------------------------------
                # Validar la reconstrucción de la puntuación
                # ------------------------------------------

                diff = abs(
                    metadata["original_score_recomputed"]
                    - scores[bias]
                )

                additivity = float(
                    metadata["additivity_error"]
                )

                if diff > self.config.score_tolerance:
                    raise ValueError(
                        f"SHAP no reconstruye la "
                        f"puntuación de {bias}: "
                        f"diferencia={diff:.6g}"
                    )

                if (
                    additivity
                    > self.config.shap_additivity_tolerance
                ):
                    raise ValueError(
                        f"Error de aditividad excesivo "
                        f"para {bias}: {additivity:.6g}"
                    )

                # ------------------------------------------
                # Conservar los fragmentos atribuidos
                # ------------------------------------------

                fragments_by_bias[bias] = attrs[
                    [
                        "group_id",
                        "turn_ids",
                        "text",
                        "shap_value",
                    ]
                ].to_dict(
                    orient="records"
                )

                # ------------------------------------------
                # Metadatos de la explicación
                # ------------------------------------------

                explanations[bias] = {
                    "winning_window_id": winners[bias],

                    "score_recomputed": float(
                        metadata[
                            "original_score_recomputed"
                        ]
                    ),

                    "score_difference": float(diff),

                    "additivity_error": additivity,

                    "baseline_score_masked": float(
                        metadata[
                            "baseline_score_masked"
                        ]
                    ),

                    "n_groups": int(
                        metadata["n_groups"]
                    ),

                    "nsamples": int(
                        metadata["nsamples"]
                    ),

                    "masking_method": str(
                        metadata["masking_method"]
                    ),
                }

        # ==================================================
        # 7. Generación de alertas explicables
        # ==================================================

        # La generación es atómica: si falta alguna
        # explicación SHAP, generate_alerts lanzará
        # una excepción y no devolverá alertas parciales.

        alerts = generate_alerts(
            conversation_id=cid,
            scores=scores,
            thresholds=self.thresholds,
            shap_fragments=fragments_by_bias,
        )

        # ==================================================
        # 8. Resultado estructurado
        # ==================================================

        return {
            "conversation_id": cid,

            "status": (
                "alerts_generated"
                if alerts
                else "no_alerts"
            ),

            "n_windows": len(windows),

            "scores": scores,

            "thresholds": dict(self.thresholds),

            "detected_biases": positive_biases,

            "winning_windows": {
                bias: winners[bias]
                for bias in positive_biases
            },

            "explanations": explanations,

            "alerts": alerts,

            "model_version": "mlp_mil_v3",

            "evaluation_mode": "retrospective",
        }


def save_pipeline_result(
    result: dict[str, Any],
    destination: str | Path,
) -> Path:
    """
    Guarda un resultado completo en formato JSON.

    No sobrescribe resultados anteriores.
    """

    path = Path(destination)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # El modo "x" evita sobrescrituras.
    with path.open(
        "x",
        encoding="utf-8",
    ) as output:

        json.dump(
            result,
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    return path