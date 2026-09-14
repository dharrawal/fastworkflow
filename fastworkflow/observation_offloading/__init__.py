"""Arm D observation offloading: compact, archive, search_memory, continuation.

Enabled by default. Set ``FW_OBSERVATION_OFFLOADING=0`` to restore stock ReAct.
"""
from __future__ import annotations

from fastworkflow.observation_offloading.agent import build_tool_agent, enabled
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import (
    PACKED_TARGET_BYTES,
    RECENT_OBSERVATIONS_PROTECTED,
    annotate_execute_observations,
    compact_trajectory,
    execute_ordinals,
)
from fastworkflow.observation_offloading.continuation import (
    MAX_FORCED_REPLANS,
    REPLAN_OBSERVATION_MAX_BYTES,
    StructuredContinuationReAct,
    replan_trajectory_skeleton,
)
from fastworkflow.observation_offloading.labels import (
    alias_line,
    offload_label,
    printed_alias,
    strip_alias_line,
)
from fastworkflow.observation_offloading.manifest import (
    classify_against_steps,
    install_span_policy,
    uninstall_span_policy,
)
from fastworkflow.observation_offloading.search import search_memory
from fastworkflow.observation_offloading.state import (
    clear_hot_handles,
    hot_payload_bytes,
    reset_runtime_state,
    stored_handles,
)

__all__ = [
    "MAX_FORCED_REPLANS",
    "PACKED_TARGET_BYTES",
    "PersistenceError",
    "RECENT_OBSERVATIONS_PROTECTED",
    "REPLAN_OBSERVATION_MAX_BYTES",
    "RuntimeHandleArchive",
    "RuntimeHandleScope",
    "StructuredContinuationReAct",
    "alias_line",
    "annotate_execute_observations",
    "build_tool_agent",
    "classify_against_steps",
    "clear_hot_handles",
    "compact_trajectory",
    "enabled",
    "execute_ordinals",
    "hot_payload_bytes",
    "install_span_policy",
    "offload_label",
    "printed_alias",
    "replan_trajectory_skeleton",
    "reset_runtime_state",
    "search_memory",
    "stored_handles",
    "strip_alias_line",
    "uninstall_span_policy",
]
