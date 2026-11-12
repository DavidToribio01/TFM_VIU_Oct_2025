# @title
# Módulo alert_generator.py

"""Generación determinista de alertas explicables para MLP-MIL v3.

El módulo NO ejecuta clasificación ni SHAP. Para cada sesgo positivo exige
atribuciones SHAP textuales previamente calculadas; si faltan, lanza
MissingExplanationError y no devuelve alertas incompletas.

Una atribución positiva explica influencia sobre la puntuación del modelo,
no demuestra la presencia del sesgo ni la intención de manipular.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


BIASES = (
    "anchoring_bias",
    "availability_bias",
    "confirmation_bias",
    "framing_effect",
)


TEMPLATES = {
    "anchoring_bias": {
        "title": "Posible sesgo de anclaje",
        "summary": (
            "El sistema ha identificado un patrón potencialmente relacionado "
            "con la influencia de una referencia inicial sobre valoraciones "
            "posteriores."
        ),
        "review_question": (
            "¿Una cifra o idea inicial ha condicionado las valoraciones "
            "posteriores? ¿Se consideraron otras referencias?"
        ),
    },
    "availability_bias": {
        "title": "Posible sesgo de disponibilidad",
        "summary": (
            "El sistema ha identificado un patrón potencialmente relacionado "
            "con una importancia excesiva otorgada a ejemplos llamativos "
            "o fáciles de recordar."
        ),
        "review_question": (
            "¿Los ejemplos utilizados son representativos? "
            "¿Se han considerado datos y alternativas relevantes?"
        ),
    },
    "confirmation_bias": {
        "title": "Posible sesgo de confirmación",
        "summary": (
            "El sistema ha identificado un patrón potencialmente relacionado "
            "con el refuerzo de ideas previas sin suficiente contraste "
            "con perspectivas alternativas."
        ),
        "review_question": (
            "¿El agente ha contrastado las premisas y justificado sus "
            "respuestas? ¿Consideró alternativas cuando correspondía?"
        ),
    },
    "framing_effect": {
        "title": "Posible efecto marco",
        "summary": (
            "El sistema ha identificado un patrón potencialmente relacionado "
            "con la influencia de la forma en que se presenta una "
            "información o una decisión."
        ),
        "review_question": (
            "¿Presentar la misma información de otra forma podría "
            "modificar tu interpretación o decisión?"
        ),
    },
}


DISCLAIMER = (
    "Alerta orientativa: puede ser incorrecta. La puntuación no es una "
    "probabilidad calibrada ni prueba la existencia de un sesgo o una "
    "intención manipulativa. Las atribuciones SHAP muestran influencia "
    "sobre la predicción, no evidencias concluyentes; se limitan a los "
    "grupos de texto analizados en la ventana dominante y pueden omitir "
    "pasajes relevantes de otras ventanas."
)


class MissingExplanationError(ValueError):
    """No puede completarse una alerta positiva sin fragmentos SHAP válidos."""


@dataclass(frozen=True)
class Alert:
    conversation_id: str
    bias: str
    title: str
    summary: str
    score: float
    threshold: float
    shap_fragments: list[dict[str, Any]]
    n_shap_fragments: int
    review_question: str
    disclaimer: str
    explanation_method: str = "SHAP_text_masking_winning_MIL_window"
    evaluation_mode: str = "retrospective"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probability(value: Any, name: str) -> float:
    """Valida una puntuación o umbral comprendido entre 0 y 1."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} debe ser numérico") from exc

    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} debe estar en [0, 1]")

    return number


def _turn_ids(value: Any, bias: str, group: Any) -> list[int]:
    """Normaliza los identificadores de turno."""

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"turn_ids inválidos en {bias}, grupo {group}"
            ) from exc

    if not isinstance(value, (list, tuple, set)):
        raise ValueError(
            f"turn_ids deben ser una secuencia en {bias}, grupo {group}"
        )

    result = []

    for turn in value:

        if isinstance(turn, bool) or not isinstance(turn, (int, str)):
            raise ValueError(f"ID de turno inválido: {turn!r}")

        try:
            n = int(turn)
        except ValueError as exc:
            raise ValueError(
                f"ID de turno inválido: {turn!r}"
            ) from exc

        if n < 1 or str(turn).strip() != str(n):
            raise ValueError(f"ID de turno inválido: {turn!r}")

        result.append(n)

    return sorted(set(result))


def _positive_fragments(
    fragments: Sequence[Mapping[str, Any]],
    bias: str
) -> list[dict[str, Any]]:
    """
    Recupera todos los grupos con contribución SHAP positiva.

    No limita el número de grupos ni trunca sus textos.
    """

    if (
        isinstance(fragments, (str, bytes))
        or not isinstance(fragments, Sequence)
    ):
        raise MissingExplanationError(
            f"SHAP de {bias} debe ser una lista de grupos"
        )

    selected: list[dict[str, Any]] = []
    seen_groups: set[int] = set()

    for i, item in enumerate(fragments):

        if not isinstance(item, Mapping):
            raise ValueError(
                f"Grupo SHAP {i} de {bias} no es un diccionario"
            )

        try:
            value = float(item["shap_value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shap_value ausente o inválido en {bias}, grupo {i}"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"shap_value no finito en {bias}, grupo {i}"
            )

        # Los valores no positivos se excluyen de la alerta.
        if value <= 0:
            continue

        try:
            group_id = int(item["group_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"group_id ausente o inválido en {bias}, grupo {i}"
            ) from exc

        if group_id < 0 or group_id in seen_groups:
            raise ValueError(
                f"group_id duplicado o negativo en {bias}: {group_id}"
            )

        seen_groups.add(group_id)

        text = item.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Texto vacío o inválido en {bias}, grupo {group_id}"
            )

        turns = _turn_ids(
            item.get("turn_ids", []),
            bias,
            group_id
        )

        selected.append({
            "group_id": group_id,
            "turn_ids": turns,
            "text": text,
            "shap_value": value,
        })

    # Ordenar por contribución descendente
    selected.sort(
        key=lambda f: (-f["shap_value"], f["group_id"])
    )

    if not selected:
        raise MissingExplanationError(
            f"La detección positiva de {bias} no tiene fragmentos SHAP "
            "con contribución positiva; alerta no generada."
        )

    return selected


def generate_alerts(
    *,
    conversation_id: str,
    scores: Mapping[str, float],
    thresholds: Mapping[str, float],
    shap_fragments: Mapping[
        str,
        Sequence[Mapping[str, Any]]
    ],
) -> list[dict[str, Any]]:
    """
    Genera alertas explicables utilizando plantillas deterministas.

    Parámetros
    ----------
    conversation_id:
        Identificador de la conversación analizada.

    scores:
        Diccionario con las cuatro puntuaciones del clasificador.

    thresholds:
        Diccionario con los cuatro umbrales optimizados
        previamente sobre validación.

    shap_fragments:
        Diccionario con los grupos SHAP correspondientes
        a cada sesgo detectado.

    Retorno
    -------
    list[dict]:
        Lista de alertas serializables a JSON.

    Comportamiento
    --------------
    - Solo genera alertas para los sesgos que superan su umbral.
    - Requiere SHAP para todas las detecciones positivas.
    - Conserva todos los grupos con contribución positiva.
    - Si falta alguna explicación, no devuelve alertas parciales.
    - Si no hay detecciones, devuelve una lista vacía.
    """

    # Validar identificador
    if (
        not isinstance(conversation_id, str)
        or not conversation_id.strip()
    ):
        raise ValueError(
            "conversation_id debe ser una cadena no vacía"
        )

    # Validar diccionarios
    if not isinstance(scores, Mapping) or not isinstance(
        thresholds, Mapping
    ):
        raise ValueError(
            "scores y thresholds deben ser diccionarios"
        )

    if set(scores) != set(BIASES) or set(thresholds) != set(BIASES):
        raise ValueError(
            f"scores y thresholds deben contener exactamente {BIASES}"
        )

    if not isinstance(shap_fragments, Mapping):
        raise ValueError(
            "shap_fragments debe ser un diccionario"
        )

    if set(shap_fragments) - set(BIASES):
        raise ValueError(
            "shap_fragments contiene sesgos desconocidos"
        )

    # Validar puntuaciones
    validated_scores = {
        bias: _probability(
            scores[bias],
            f"score de {bias}"
        )
        for bias in BIASES
    }

    # Validar umbrales
    validated_thresholds = {
        bias: _probability(
            thresholds[bias],
            f"threshold de {bias}"
        )
        for bias in BIASES
    }

    # Identificar sesgos detectados
    positive = [
        bias
        for bias in BIASES
        if validated_scores[bias] >= validated_thresholds[bias]
    ]

    # Preparar todas las explicaciones antes de generar alertas
    prepared: dict[str, list[dict[str, Any]]] = {}

    for bias in positive:

        if bias not in shap_fragments:
            raise MissingExplanationError(
                "Falta explicación SHAP para la detección "
                f"positiva: {bias}"
            )

        prepared[bias] = _positive_fragments(
            shap_fragments[bias],
            bias
        )

    # Generación determinista mediante plantillas
    alerts = [
        Alert(
            conversation_id=conversation_id,
            bias=bias,
            title=TEMPLATES[bias]["title"],
            summary=TEMPLATES[bias]["summary"],
            score=validated_scores[bias],
            threshold=validated_thresholds[bias],
            shap_fragments=prepared[bias],
            n_shap_fragments=len(prepared[bias]),
            review_question=TEMPLATES[bias]["review_question"],
            disclaimer=DISCLAIMER,
        ).to_dict()

        for bias in positive
    ]

    return alerts