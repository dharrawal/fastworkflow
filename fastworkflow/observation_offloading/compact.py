"""Oldest-first packed-trajectory compaction (Arm D)."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Optional

from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.labels import (
    estimated_tokens,
    is_offload_label,
    offload_label,
)
from fastworkflow.observation_offloading.state import (
    archive,
    default_scope,
    evict_hot_handles,
    hot_handle_max_bytes_from_env,
    hot_payload_bytes,
    record_event,
    remember_handle,
)

ELIGIBILITY_THRESHOLD_TOKENS = 1_000
RECENT_OBSERVATIONS_PROTECTED = 5
PACKED_TARGET_BYTES = 28_000
TRAJECTORY_MAX_BYTES_ENV = "FW_TRAJECTORY_MAX_BYTES"


def packed_target_bytes_from_env(default: int = PACKED_TARGET_BYTES) -> int:
    raw = os.environ.get(TRAJECTORY_MAX_BYTES_ENV, "").strip()
    return default if not raw else int(raw)


def execute_ordinals(trajectory: Mapping[str, Any]) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    ordinal = 0
    index = 0
    while f"tool_name_{index}" in trajectory or f"observation_{index}" in trajectory:
        if str(trajectory.get(f"tool_name_{index}") or "") == "execute_workflow_query":
            ordinal += 1
            found.append((index, ordinal))
        index += 1
    return found


def _over_packed_target(
    text: str,
    *,
    packed_target_bytes: int,
    packed_target_tokens: Optional[int],
) -> bool:
    if packed_target_tokens is not None:
        return estimated_tokens(text) > packed_target_tokens
    return len(text.encode("utf-8")) > packed_target_bytes


def compact_trajectory(
    trajectory: dict[str, Any],
    *,
    eligibility_threshold_tokens: int = ELIGIBILITY_THRESHOLD_TOKENS,
    recent_observations_protected: int = RECENT_OBSERVATIONS_PROTECTED,
    packed_target_tokens: Optional[int] = None,
    packed_target_bytes: int = PACKED_TARGET_BYTES,
    hot_handle_max_bytes: Optional[int] = None,
    scope: Optional[RuntimeHandleScope] = None,
    selected_archive: Optional[RuntimeHandleArchive] = None,
) -> list[dict[str, Any]]:
    """Mutate trajectory observations in place. Return offload decisions."""

    selected_scope = scope or default_scope()
    store = selected_archive or archive()
    if packed_target_tokens is None:
        packed_target_bytes = packed_target_bytes_from_env(packed_target_bytes)
    if hot_handle_max_bytes is None:
        hot_handle_max_bytes = hot_handle_max_bytes_from_env()
    if hot_handle_max_bytes < 0:
        raise ValueError("hot_handle_max_bytes cannot be negative")
    executes = execute_ordinals(trajectory)
    if not executes:
        return []
    protected_from = max(1, len(executes) - recent_observations_protected + 1)
    packed_text = json.dumps(trajectory, ensure_ascii=False, default=str)
    decisions: list[dict[str, Any]] = []
    for step_index, ordinal in executes:
        key = f"observation_{step_index}"
        response = trajectory.get(key)
        if not isinstance(response, str):
            continue
        alias = f"O{ordinal}"
        size = {
            "characters": len(response),
            "utf8_bytes": len(response.encode("utf-8")),
            "estimated_tokens": estimated_tokens(response),
        }
        recency_protected = ordinal >= protected_from
        already_label = is_offload_label(response)
        decision = {
            "alias": alias,
            "step_index": step_index,
            "action": "kept",
            "reason": "below_threshold",
            "recency_protected": recency_protected,
            "response_size": size,
        }
        if already_label:
            decision["reason"] = "already_label"
            decisions.append(decision)
            continue
        eligible = (
            size["estimated_tokens"] > eligibility_threshold_tokens and not recency_protected
        )
        if recency_protected:
            decision["reason"] = "recent_observation_protected"
        elif not eligible:
            decision["reason"] = "below_threshold"
        elif not _over_packed_target(
            packed_text,
            packed_target_bytes=packed_target_bytes,
            packed_target_tokens=packed_target_tokens,
        ):
            decision["reason"] = "eligible_but_target_already_met"
        else:
            args = trajectory.get(f"tool_args_{step_index}")
            command = ""
            if isinstance(args, Mapping):
                command = str(args.get("command") or "")
            label = offload_label(
                alias=alias,
                command_name=command or "execute_workflow_query",
                response=response,
            )
            digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
            packed_utf8_bytes_before = len(packed_text.encode("utf-8"))
            try:
                store.persist(
                    selected_scope,
                    alias=alias,
                    offload_order=ordinal,
                    command_name=command,
                    step_index=step_index,
                    text=response,
                    text_sha256=digest,
                )
            except Exception as error:  # noqa: BLE001
                decision["reason"] = "persistence_failed_original_retained"
                decision["persistence_error"] = type(error).__name__
                record_event(
                    {
                        "kind": "offload_refused",
                        "scope_id": selected_scope.scope_id,
                        "alias": alias,
                        "reason": decision["reason"],
                        "error": type(error).__name__,
                    }
                )
                decisions.append(decision)
                continue
            remember_handle(
                selected_scope,
                {
                    "alias": alias,
                    "text": response,
                    "text_sha256": digest,
                    "command": command,
                    "step_index": step_index,
                    "offload_order": ordinal,
                },
            )
            evicted = evict_hot_handles(
                selected_scope, hot_handle_max_bytes=hot_handle_max_bytes
            )
            trajectory[key] = label
            decision["action"] = "offloaded"
            decision["reason"] = "oldest_eligible_until_target"
            decision["label"] = label
            decision["text_sha256"] = digest
            decision["persisted_before_label"] = True
            decision["hot_evictions"] = evicted
            decision["hot_payload_bytes"] = hot_payload_bytes(selected_scope)
            decision["packed_utf8_bytes_before"] = packed_utf8_bytes_before
            packed_text = json.dumps(trajectory, ensure_ascii=False, default=str)
            record_event(
                {
                    "kind": "offload",
                    "scope_id": selected_scope.scope_id,
                    "alias": alias,
                    "step_index": step_index,
                    "text_sha256": digest,
                    "estimated_tokens": size["estimated_tokens"],
                    "packed_utf8_bytes_before": packed_utf8_bytes_before,
                    "packed_utf8_bytes_after": len(packed_text.encode("utf-8")),
                    "hot_payload_bytes": hot_payload_bytes(selected_scope),
                    "hot_evictions": evicted,
                }
            )
        decisions.append(decision)
    return decisions
