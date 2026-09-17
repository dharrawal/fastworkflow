"""Record observation slots on fw.llm.call instead of a system-prompt prefix."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Optional

from fastworkflow import tracing
from fastworkflow.observation_offloading.labels import (
    canonical_response,
    is_offload_label,
    observation_alias,
)

CONTRACT = "fastworkflow-trajectory-manifest/1"
DSPY_OBS_RE = re.compile(
    r"\[\[ ## (observation_(\d+)) ## \]\]\s*(.*?)(?=\n\[\[ ## |\Z)",
    re.S,
)
ALIAS_RE = re.compile(r"^(O\d+)\s+—")

_installed = False
_original_capped = tracing._capped


def observation_row(key: str, text: str) -> dict[str, Any]:
    """One observation slot of the prompt, described without carrying it.

    Two digests, because the slot and the evidence are not the same bytes
    (``ido-sll``). ``sha256`` is the PROMPT SLOT exactly as the model received
    it, header and all, and is what a reader has to hash to prove what was
    sent. ``response_sha256`` is the command response inside that slot --
    ``canonical_response`` takes our handle line and its escape back off -- and
    is what ``fw.agent.step`` recorded, because that span closes before the
    completion hook annotates. Comparing the first with the second is comparing
    non-equivalent bytes, and it reported unchanged resident evidence as
    mismatched.

    ``alias`` is read from either line this package prints, not from the
    offload label alone: since the handle line became unconditional the inline
    case IS the normal case, and it was the one reporting no alias.
    ``alias_source`` says which line named it, so a resident observation and a
    pointer to one are still told apart by the row rather than by inference.

    One normalisation, used by every field that reads the text's shape: the
    alias, the kind and the response were previously read off differently
    normalised copies, so a slot with leading whitespace could report an alias
    and call itself text in the same row.
    """
    probe = text.lstrip()
    alias, source = observation_alias(probe)
    response = canonical_response(probe)
    encoded = text.encode("utf-8")
    return {
        "key": key,
        "alias": alias,
        "alias_source": source,
        "kind": "label" if is_offload_label(probe) else "text",
        "chars": len(text),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "response_sha256": (
            None if response is None
            else hashlib.sha256(response.encode("utf-8")).hexdigest()
        ),
    }


def _message_blobs(messages_json: str) -> tuple[list[str], int]:
    try:
        parsed = json.loads(messages_json)
    except (TypeError, ValueError):
        return [str(messages_json or "")], 0
    if not isinstance(parsed, list):
        return [messages_json], 0
    blobs: list[str] = []
    system_prompt_chars = 0
    for message in parsed:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            text = "\n".join(parts)
        if message.get("role") == "system" and system_prompt_chars == 0:
            system_prompt_chars = len(text)
        if text:
            blobs.append(text)
    return blobs, system_prompt_chars


def manifest_from_messages_json(messages_json: str) -> Optional[dict[str, Any]]:
    if not isinstance(messages_json, str) or not messages_json:
        return None
    blobs, system_prompt_chars = _message_blobs(messages_json)
    by_index: dict[int, dict[str, Any]] = {}
    for blob in blobs:
        for match in DSPY_OBS_RE.finditer(blob):
            by_index[int(match.group(2))] = observation_row(match.group(1), match.group(3))
    if not by_index:
        return None
    observations = [by_index[index] for index in sorted(by_index)]
    return {
        "contract": CONTRACT,
        "system_prompt_chars": system_prompt_chars,
        "messages_original_length": len(messages_json.encode("utf-8")),
        "observation_count": len(observations),
        "observations": observations,
    }


def _comparable_digests(row: Mapping[str, Any]) -> set[str]:
    """Digests of this row that a step's own evidence can honestly be equal to.

    The canonical response first -- that is the raw tool return the step span
    recorded -- and the prompt-slot digest too, because a slot this package
    never annotated (an older recording, a non-execute tool) has only that one
    and the two are then the same bytes anyway.
    """
    return {
        value
        for value in (row.get("response_sha256"), row.get("sha256"))
        if isinstance(value, str) and value
    }


def classify_against_steps(
    manifest: Mapping[str, Any],
    step_sha256_by_index: Mapping[int, str],
) -> dict[str, list[int]]:
    """Each step's recorded observation digest against the manifest's rows.

    ``step_sha256_by_index`` is the digest of the RAW tool return from
    ``fw.agent.step`` -- recorded before the completion hook prints the handle
    line -- so residency is decided against the response inside the slot, not
    against the annotated slot (``ido-sll``).

    ``mismatched`` therefore means the evidence genuinely differs, and it still
    can: a rehydrated listing carries its own response plus the stored rows
    behind its result handle, which is an intentional transformation of the
    slot and not the step's bytes. A rehydrated offload label, whose archived
    response comes back whole, is resident -- the transformation is the header,
    and the header is no longer counted against it.
    """
    by_index: dict[int, dict[str, Any]] = {}
    for row in manifest.get("observations") or []:
        key = str(row.get("key") or "")
        if not key.startswith("observation_"):
            continue
        try:
            index = int(key.removeprefix("observation_"))
        except ValueError:
            continue
        by_index[index] = row
    resident: list[int] = []
    labelled: list[int] = []
    absent: list[int] = []
    mismatched: list[int] = []
    for index, digest in sorted(step_sha256_by_index.items()):
        row = by_index.get(index)
        if row is None:
            absent.append(index)
        elif row.get("kind") == "label":
            labelled.append(index)
        elif digest in _comparable_digests(row):
            resident.append(index)
        else:
            mismatched.append(index)
    return {
        "resident": resident,
        "labelled": labelled,
        "absent": absent,
        "mismatched": mismatched,
    }


def _capped_with_manifest(attributes: Optional[dict[str, Any]]) -> dict[str, Any]:
    capped = _original_capped(attributes)
    if not attributes:
        return capped
    messages = attributes.get("messages")
    if not isinstance(messages, str):
        return capped
    manifest = manifest_from_messages_json(messages)
    if not manifest:
        return capped
    capped["trajectory_manifest"] = manifest
    stored = capped.get("messages")
    if isinstance(stored, dict) and stored.get("truncated"):
        stored["value"] = ""
        stored["replaced_by"] = "trajectory_manifest"
    return capped


def install_span_policy() -> None:
    global _installed
    if _installed:
        return
    tracing._capped = _capped_with_manifest
    _installed = True


def uninstall_span_policy() -> None:
    global _installed
    tracing._capped = _original_capped
    _installed = False
