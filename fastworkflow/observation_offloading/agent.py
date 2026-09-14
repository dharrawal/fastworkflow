"""Wire Arm D offloading onto an initialized workflow tool agent."""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import fastworkflow
from fastworkflow import state_paths, tracing
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.continuation import (
    DEFAULT_MAX_ITERS,
    StructuredContinuationReAct,
    max_forced_replans_from_env,
)
from fastworkflow.observation_offloading.manifest import install_span_policy
from fastworkflow.observation_offloading.search import search_memory
from fastworkflow.observation_offloading.state import HANDLE_ARCHIVE_ENV, record_event

ENABLED_ENV = "FW_OBSERVATION_OFFLOADING"

logger = logging.getLogger(__name__)


def enabled() -> bool:
    raw = os.environ.get(ENABLED_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _callable_from_tool(tool: Any) -> Callable[..., Any]:
    if callable(tool) and not hasattr(tool, "func"):
        return tool
    func = getattr(tool, "func", None)
    if callable(func):
        return func
    if callable(tool):
        return tool
    raise TypeError(f"cannot unwrap tool {tool!r}")


def _scope_for_session(chat_session: Any) -> RuntimeHandleScope:
    claim = tracing.get_experiment_claim(chat_session)
    channel_id = str(tracing.get_channel_id(chat_session) or "unbound")
    turn_key = str(tracing.get_turn_key(chat_session) or channel_id)
    sink = tracing.get_sink(chat_session)
    sink_store = getattr(sink, "store", None)
    identity_value = getattr(sink_store, "store_identity", None)
    if callable(identity_value):
        identity_value = identity_value()
    getter = getattr(chat_session, "get_active_workflow", None)
    active_workflow = getter() if callable(getter) else None
    workflow_path = str(getattr(active_workflow, "folderpath", "") or "")
    store_identity = str(
        identity_value
        or getattr(sink, "store_identity", None)
        or state_paths.observability_db(workflow_path)
    )
    return RuntimeHandleScope(
        store_identity=store_identity,
        channel_id=channel_id,
        experiment_id=str(claim.get("experiment_id") or "unbound"),
        task_id=str(claim.get("task_id") or "unbound"),
        attempt=int(claim.get("attempt") or 0),
        turn_key=turn_key,
    )


def build_compacting_step(
    agent_ref: Callable[[], Any],
    *,
    fallback_scope: RuntimeHandleScope,
    selected_archive: RuntimeHandleArchive,
    on_step_complete: Optional[Callable[[int, dict[str, Any]], bool]] = None,
    describe_output: Optional[Callable[[str, str], str]] = None,
) -> Callable[[int, dict[str, Any]], bool]:
    """The ReAct on_step_complete hook: compact, then defer to the caller's hook.

    ``_run_loop`` invokes this with no try/except of its own, so anything raised
    here would turn a successful tool call into a full-turn abort. Offloading is
    an optimisation; a failure to compact is logged and recorded, and the step
    continues with its observation left inline.
    """

    def compacting_step(idx: int, trajectory: dict[str, Any]) -> bool:
        agent = agent_ref()
        scope = getattr(agent, "continuation_scope", None) or fallback_scope
        try:
            compact_trajectory(
                trajectory,
                scope=scope,
                selected_archive=selected_archive,
                ordinal_offset=int(getattr(agent, "truncated_execute_steps", 0) or 0),
                describe_output=describe_output,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "observation offloading skipped compaction at step %d: %s: %s",
                idx, type(error).__name__, error,
            )
            record_event(
                {
                    "kind": "compaction_failed",
                    "scope_id": scope.scope_id,
                    "step_index": idx,
                    "error": type(error).__name__,
                    "detail": str(error)[:300],
                }
            )
        if on_step_complete is not None and not on_step_complete(idx, trajectory):
            return False
        return True

    return compacting_step


def current_search_reasoning(agent: Any) -> str:
    """Read the current search step, not an earlier completed tool's thought."""
    trajectory = agent.current_trajectory
    indexes = [int(key.removeprefix("tool_name_")) for key in trajectory
               if key.startswith("tool_name_") and key.removeprefix("tool_name_").isdigit()]
    if not indexes:
        return ""
    index = max(indexes)
    if trajectory.get(f"tool_name_{index}") != "search_memory":
        return ""
    return str(trajectory.get(f"thought_{index}") or "")


def describe_command_output(chat_session: Any, command: str, response: str) -> str:
    """Resolve authored output fields from the command that actually produced this text."""
    core = getattr(chat_session, "_core", chat_session)
    records = getattr(core, "action_log", [])
    record = next((r for r in reversed(records)
                   if r.get("command") == command and r.get("response") == response), None)
    if record is None:
        return ""
    try:
        workflow = chat_session.get_active_workflow()
        routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        metadata = CommandMetadataAPI._extract_signature_info(
            record["command_name"], routing, routing)
        fields = metadata.get("outputs", [])
        return "; ".join(f"{field['name']}: {field['description']}"
                         for field in fields if field.get("description"))
    except Exception:
        # Missing metadata must never prevent persistence or turn success.
        return ""


def maybe_wrap_tool_agent(
    chat_session: Any,
    agent: Any,
    *,
    max_iters: int,
    on_step_complete=None,
) -> Any:
    if not enabled():
        return agent
    install_span_policy()
    # The scope is re-resolved by the rebuilt agent at every forward(), so the
    # turn_key it carries is the turn actually running. This one is only the
    # fallback for a step that fires before the first forward() bound a scope.
    scope = _scope_for_session(chat_session)
    archive_path = os.environ.get(HANDLE_ARCHIVE_ENV, "").strip()
    if not archive_path:
        getter = getattr(chat_session, "get_active_workflow", None)
        active_workflow = getter() if callable(getter) else None
        workflow_path = str(getattr(active_workflow, "folderpath", "") or "")
        archive_path = state_paths.observability_db(workflow_path) + ".offload-handles.sqlite3"
    selected_archive = RuntimeHandleArchive(archive_path)
    rebuilt: Any = None

    compacting_step = build_compacting_step(
        lambda: rebuilt,
        fallback_scope=scope,
        selected_archive=selected_archive,
        on_step_complete=on_step_complete,
        describe_output=lambda command, response: describe_command_output(chat_session, command, response),
    )

    def scoped_search_memory(question: str, alias: str) -> str:
        """Answer a question inside one offloaded observation. alias is required (e.g. O8)."""

        current = getattr(rebuilt, "continuation_scope", None) or scope
        return search_memory(
            question, alias, reasoning=current_search_reasoning(rebuilt),
            scope=current, selected_archive=selected_archive
        )

    scoped_search_memory.__name__ = "search_memory"
    callables: list[Callable[..., Any]] = []
    for name, tool in agent.tools.items():
        if name == "finish":
            continue
        callables.append(_callable_from_tool(tool))
    callables.append(scoped_search_memory)
    rebuilt = StructuredContinuationReAct(
        agent.signature,
        tools=callables,
        max_iters=int(max_iters or DEFAULT_MAX_ITERS),
        on_step_complete=compacting_step,
        scope_factory=lambda: _scope_for_session(chat_session),
    )
    rebuilt.continuation_scope_id = scope.scope_id
    rebuilt.observation_archive = selected_archive
    rebuilt.describe_output = lambda command, response: describe_command_output(chat_session, command, response)
    record_event(
        {
            "kind": "agent_installed",
            "max_iters": rebuilt.max_iters,
            "max_forced_replans": max_forced_replans_from_env(),
            "tools": sorted(rebuilt.tools),
            "scope_id": scope.scope_id,
        }
    )
    return rebuilt
