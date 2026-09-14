"""Record observation slots on fw.llm.call instead of a system-prompt prefix."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Optional

from fastworkflow import tracing
from fastworkflow.observation_offloading.labels import is_offload_label, label_alias

CONTRACT = "fastworkflow-trajectory-manifest/1"
DSPY_OBS_RE = re.compile(
    r"\[\[ ## (observation_(\d+)) ## \]\]\s*(.*?)(?=\n\[\[ ## |\Z)",
    re.S,
)
ALIAS_RE = re.compile(r"^(O\d+)\s+—")

_installed = False
_original_capped = tracing._capped


def observation_row(key: str, text: str) -> dict[str, Any]:
    alias = label_alias(text.lstrip())
    encoded = text.encode("utf-8")
    return {
        "key": key,
        "alias": alias,
        "kind": "label" if is_offload_label(text) else "text",
        "chars": len(text),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
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


def classify_against_steps(
    manifest: Mapping[str, Any],
    step_sha256_by_index: Mapping[int, str],
) -> dict[str, list[int]]:
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
        elif row.get("sha256") == digest:
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
