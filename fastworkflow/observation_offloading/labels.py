"""Offload-label text written into a ReAct trajectory slot."""
from __future__ import annotations

import hashlib
import math

CHARS_PER_TOKEN = 4
OFFLOAD_MARK = "full saved observation offloaded"


def estimated_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def offload_label(*, alias: str, command_name: str, response: str) -> str:
    digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
    return (
        f"{alias} — {command_name}; {OFFLOAD_MARK} "
        f"({len(response)} chars, {len(response.encode('utf-8'))} UTF-8 bytes, "
        f"~{estimated_tokens(response)} tokens; sha256 {digest}). "
        f"Ask a focused evidence question against handle {alias}; an exhaustive "
        "internal walk of that handle may be required. Source completeness: unknown."
    )


def is_offload_label(text: str) -> bool:
    return OFFLOAD_MARK in text
