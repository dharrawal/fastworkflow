"""Arm D observation offloading: compact, archive, search_memory, continuation.

Enabled by default. Set ``FW_OBSERVATION_OFFLOADING=0`` to restore stock ReAct.
"""
from __future__ import annotations

from fastworkflow.observation_offloading.agent import enabled, maybe_wrap_tool_agent
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import (
    PACKED_TARGET_BYTES,
    RECENT_OBSERVATIONS_PROTECTED,
    compact_trajectory,
    execute_ordinals,
)
from fastworkflow.observation_offloading.continuation import (
    MAX_FORCED_REPLANS,
    REPLAN_OBSERVATION_MAX_BYTES,
    StructuredContinuationReAct,
    replan_trajectory_skeleton,
)
from fastworkflow.observation_offloading.labels import offload_label
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
    "classify_against_steps",
    "clear_hot_handles",
    "compact_trajectory",
    "enabled",
    "execute_ordinals",
    "hot_payload_bytes",
    "install_span_policy",
    "maybe_wrap_tool_agent",
    "offload_label",
    "replan_trajectory_skeleton",
    "reset_runtime_state",
    "search_memory",
    "stored_handles",
    "uninstall_span_policy",
]
