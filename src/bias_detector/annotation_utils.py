"""Herramientas reutilizables de anotación; sin llamadas a servicios externos."""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path


def load_prompt_templates(prompt_files: Mapping[str, str | Path], *, validate_placeholder: bool = True) -> dict[str, str]:
    """Carga prompts UTF-8 y comprueba el marcador de conversación."""
    templates = {}
    for bias, value in prompt_files.items():
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el prompt de {bias}: {path}")
        template = path.read_text(encoding="utf-8")
        if validate_placeholder and "{{CONVERSATION}}" not in template:
            raise ValueError(f"El prompt de {bias} carece de {{CONVERSATION}}")
        templates[bias] = template
    return templates


def format_annotation_prompt(templates: Mapping[str, str], bias: str, conversation_text: str) -> str:
    """Inserta una conversación en la plantilla, validando las claves."""
    if bias not in templates:
        raise ValueError(f"Sesgo desconocido: {bias}")
    template = templates[bias]
    if "{{CONVERSATION}}" not in template:
        raise ValueError(f"La plantilla de {bias} no tiene marcador de conversación")
    return template.replace("{{CONVERSATION}}", conversation_text)


def validate_annotation_turns(annotation: object, expected_bias: str, n_messages: int) -> bool:
    """Comprueba la categoría y que los turnos indicados existan."""
    if annotation.bias != expected_bias:
        raise ValueError(f"Sesgo incorrecto: esperado {expected_bias}; recibido {annotation.bias}")
    turn_ids = list(annotation.trigger_turns or []) + list(annotation.evidence_turns or [])
    invalid = [turn for turn in turn_ids if not isinstance(turn, int) or not (1 <= turn <= n_messages)]
    if invalid:
        raise ValueError(f"Turnos inexistentes: {invalid}; mensajes: {n_messages}")
    return True
