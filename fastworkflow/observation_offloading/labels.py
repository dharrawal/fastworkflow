"""Useful pointers to persisted observations, without payload-sized metadata."""
from __future__ import annotations

import math
import re

CHARS_PER_TOKEN = 4
OFFLOAD_MARK = "Use search_memory tool to search inside Observation "
LABEL_RE = re.compile(r"^Use search_memory tool to search inside Observation (O[1-9]\d*) returned by ")


def estimated_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def output_description(response: str) -> str:
    """Truthful fallback when a command has no authored Output description."""
    heading = next((line.strip() for line in response.splitlines() if line.strip()), "")
    return f"command output beginning with: {heading[:200]}" if heading else "an empty command result"


def offload_label(*, alias: str, command_name: str, response: str,
                  description: str = "") -> str:
    description = description.strip() or output_description(response)
    return (
        f"{OFFLOAD_MARK}{alias} returned by {command_name}. "
        f"It was offloaded to memory and contains {description.rstrip('.')}. "
    ).rstrip()


def is_offload_label(text: str) -> bool:
    return LABEL_RE.match(text) is not None


def label_alias(text: str) -> str | None:
    match = LABEL_RE.match(text)
    return match.group(1) if match else None


def replacement_saves_space(original: str, replacement: str) -> bool:
    return (len(replacement) < len(original)
            and len(replacement.encode('utf-8')) < len(original.encode('utf-8')))
