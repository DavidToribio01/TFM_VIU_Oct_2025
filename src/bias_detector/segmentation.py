"""Segmentación reversible a nivel de tokens para conversaciones USER/LLM.

No calcula embeddings ni usa anotaciones del corpus.
"""

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


TURN_RE = re.compile(r"\[TURN\s+(\d+)\s*\|\s*(USER|LLM)\]", re.IGNORECASE)


@dataclass(frozen=True)
class SegmentationConfig:
    model_name: str = "allenai/longformer-base-4096"
    max_length: int = 2048
    overlap: int = 128
    min_boundary_fraction: float = 0.75


def create_tokenizer(config: SegmentationConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Se necesita un tokenizador rápido para recuperar offsets.")
    payload_size = config.max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not (0 <= config.overlap < payload_size):
        raise ValueError("overlap debe ser >= 0 y menor que la capacidad útil.")
    if not (0 < config.min_boundary_fraction <= 1):
        raise ValueError("min_boundary_fraction debe estar en (0, 1].")
    if config.max_length > tokenizer.model_max_length:
        raise ValueError("max_length supera el límite del tokenizador.")
    return tokenizer


def normalize_conversation(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Conserva íntegro el contenido textual, cambia solo las cabeceras.

    Devuelve texto normalizado y spans de caracteres de cada mensaje.
    """
    if not isinstance(text, str):
        raise TypeError("conversation_text debe ser una cadena.")
    matches = list(TURN_RE.finditer(text))
    if not matches:
        raise ValueError("No se encontraron cabeceras [TURN n | USER/LLM].")
    if text[:matches[0].start()].strip():
        raise ValueError("Se encontró texto antes de la primera cabecera.")

    parts, messages, cursor, length = [], [], 0, 0
    for match in matches:
        preceding = text[cursor:match.start()]
        parts.append(preceding)
        length += len(preceding)
        header = f"[{match.group(2).upper()}]"
        start = length
        parts.append(header)
        length += len(header)
        messages.append({
            "turn_id": int(match.group(1)),
            "role": match.group(2).upper(),
            "char_start": start,
        })
        cursor = match.end()
    parts.append(text[cursor:])
    normalized = "".join(parts)
    for i, msg in enumerate(messages):
        msg["char_end"] = messages[i + 1]["char_start"] if i + 1 < len(messages) else len(normalized)

    turn_ids = [m["turn_id"] for m in messages]
    if len(turn_ids) != len(set(turn_ids)):
        raise ValueError("Hay identificadores TURN duplicados.")
    return normalized, messages


def _tokenize_full(text: str, tokenizer):
    encoded = tokenizer(
        text, add_special_tokens=False, truncation=False,
        return_offsets_mapping=True, verbose=False
    )
    return encoded["input_ids"], encoded["offset_mapping"]


def _find_boundaries(offsets, messages):
    """Índices de token en los que comienza una cabecera de mensaje."""
    boundaries = []
    j = 0
    for msg in messages[1:]:
        char_start = msg["char_start"]
        while j < len(offsets) and offsets[j][0] < char_start:
            j += 1
        if j < len(offsets):
            boundaries.append(j)
    return sorted(set(boundaries))


def _choose_window_end(start, total, capacity, boundaries, min_fraction):
    target = min(start + capacity, total)
    if target == total:
        return target
    earliest = start + max(1, int(capacity * min_fraction))
    candidates = [b for b in boundaries if earliest <= b <= target]
    return candidates[-1] if candidates else target


def segment_conversation(conversation_id, text, tokenizer, config: SegmentationConfig):
    """Genera ventanas sin perder tokens, con metadatos de trazabilidad."""
    normalized, messages = normalize_conversation(text)
    token_ids, offsets = _tokenize_full(normalized, tokenizer)
    if not token_ids:
        raise ValueError(f"Conversación vacía: {conversation_id}")

    specials = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = config.max_length - specials
    boundaries = _find_boundaries(offsets, messages)
    rows = []
    start = 0

    while start < len(token_ids):
        end = _choose_window_end(
            start, len(token_ids), capacity, boundaries,
            config.min_boundary_fraction
        )
        chunk = token_ids[start:end]
        input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        if len(input_ids) > config.max_length:
            raise AssertionError("Ventana superior a max_length")

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        included = [
            msg for msg in messages
            if msg["char_end"] > char_start and msg["char_start"] < char_end
        ]
        rows.append({
            "conversation_id": conversation_id,
            "window_id": len(rows),
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "n_tokens": len(input_ids),
            "token_start": start,
            "token_end": end,
            "char_start": char_start,
            "char_end": char_end,
            "turn_ids": [m["turn_id"] for m in included],
            "window_text": normalized[char_start:char_end],
        })

        if end == len(token_ids):
            break
        next_start = max(start + 1, end - config.overlap)
        start = next_start

    # Verificación: cada token de contenido está cubierto por al menos una ventana.
    if rows[0]["token_start"] != 0 or rows[-1]["token_end"] != len(token_ids):
        raise AssertionError("Cobertura incompleta de tokens")
    if any(a["token_end"] < b["token_start"] for a, b in zip(rows, rows[1:])):
        raise AssertionError("Hueco entre ventanas")
    return rows


def segment_corpus(df_corpus: pd.DataFrame, tokenizer, config: SegmentationConfig):
    """Produce una tabla con una fila por ventana."""
    required = {"conversation_id", "conversation_text"}
    if not required.issubset(df_corpus.columns):
        raise ValueError(f"Faltan columnas: {required - set(df_corpus.columns)}")
    if df_corpus["conversation_id"].duplicated().any():
        raise ValueError("conversation_id debe ser único.")
    rows = []
    for conversation_id, text in df_corpus[["conversation_id", "conversation_text"]].itertuples(index=False, name=None):
        rows.extend(segment_conversation(conversation_id, text, tokenizer, config))
    return pd.DataFrame(rows)
