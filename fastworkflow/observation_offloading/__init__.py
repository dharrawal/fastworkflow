"""Observation offloading: compact, archive, search_memory, offloading ReAct.

This is how fastWorkflow runs a tool agent; there is no flag to turn it off.
``build_tool_agent`` always returns an ``OffloadingReAct``, execute
observations always carry their canonical ``O{step_index}`` alias, compaction
always swaps an observation that is no longer worth its residency for its own
label, and the text behind every label stays reachable through ``search_memory``
and through answer-time rehydration. See ``docs/observation_search.md``.
"""
from __future__ import annotations

from fastworkflow.observation_offloading.agent import (
    build_tool_agent,
    open_handle_archive,
)
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
    UnavailableHandleArchive,
)
from fastworkflow.observation_offloading.compact import (
    MIN_OFFLOAD_SAVING_BYTES,
    PACKED_TARGET_BYTES,
    RECENT_OBSERVATIONS_PROTECTED,
    annotate_execute_observations,
    archive_execute_observations,
    compact_trajectory,
    execute_step_indexes,
    min_offload_saving_bytes_from_env,
)
from fastworkflow.observation_offloading.offloading_react import (
    DEFAULT_MAX_ITERS,
    OffloadingReAct,
)
from fastworkflow.observation_offloading.labels import (
    alias_line,
    annotated_observation,
    context_clause,
    escape_response,
    is_search_answer_key,
    offload_label,
    offload_saving_bytes,
    printed_alias,
    printed_context,
    search_answer_key,
    strip_alias_line,
)
from fastworkflow.observation_offloading.manifest import (
    classify_against_steps,
    install_span_policy,
    uninstall_span_policy,
)
from fastworkflow.observation_offloading.search import (
    SEARCH_ANSWER_MAX_BYTES,
    archived_search_answer,
    bounded_answer_marking,
    declaring_subject,
    present_answer,
    search_answer_max_bytes_from_env,
    search_memory,
)
from fastworkflow.agent_runtime import reclaim_scope, reset_runtime_state
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    clear_hot_handles,
    context_clause_of,
    durable_archive,
    forget_context_clause,
    hot_payload_bytes,
    observation_inline,
    record_context_clause,
    stored_handles,
)

__all__ = [
    "DEFAULT_MAX_ITERS",
    "MIN_OFFLOAD_SAVING_BYTES",
    "PACKED_TARGET_BYTES",
    "PersistenceError",
    "RECENT_OBSERVATIONS_PROTECTED",
    "RuntimeHandleArchive",
    "RuntimeHandleScope",
    "SEARCH_ANSWER_MAX_BYTES",
    "OffloadingReAct",
    "UnavailableHandleArchive",
    "alias_line",
    "archive_for_path",
    "annotate_execute_observations",
    "annotated_observation",
    "context_clause",
    "context_clause_of",
    "declaring_subject",
    "durable_archive",
    "forget_context_clause",
    "record_context_clause",
    "archive_execute_observations",
    "archived_search_answer",
    "bounded_answer_marking",
    "build_tool_agent",
    "classify_against_steps",
    "clear_hot_handles",
    "compact_trajectory",
    "escape_response",
    "execute_step_indexes",
    "hot_payload_bytes",
    "install_span_policy",
    "is_search_answer_key",
    "min_offload_saving_bytes_from_env",
    "observation_inline",
    "offload_label",
    "offload_saving_bytes",
    "open_handle_archive",
    "present_answer",
    "printed_alias",
    "printed_context",
    "reclaim_scope",
    "reset_runtime_state",
    "search_answer_key",
    "search_answer_max_bytes_from_env",
    "search_memory",
    "stored_handles",
    "strip_alias_line",
    "uninstall_span_policy",
]
