"""Process-local hot cache and optional event log."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)

from fastworkflow import context_budget

#: The hot-cache cap at the reference context window. The effective cap is a
#: fraction of the model's window (``context_budget.OFFLOAD_HOT``).
HOT_HANDLE_MAX_BYTES = context_budget.REFERENCE_OFFLOAD_HOT_MAX_BYTES
HOT_HANDLE_MAX_BYTES_ENV = context_budget.OFFLOAD_HOT.override_env
#: The diagnostic event log. A DESTINATION, not a feature switch: the events are
#: always recorded in process (``snapshot_events``), and this says where a copy
#: is appended for a run that wants one on disk. It is what the evaluation
#: harness reads every offloading, coverage and search measure out of.
EVENTS_ENV = "FW_OFFLOAD_EVENTS"
#: How many diagnostic events the process keeps in memory. The in-process log is
#: a RING, not a ledger: the durable copy is the ``FW_OFFLOAD_EVENTS`` file, and
#: what stays in memory is only what a live turn (or a test) reads back through
#: ``snapshot_events``. Unbounded it grew with lifetime traffic and held search
#: questions, reasoning and full answers long after the turns that produced them
#: had ended (ido-1ew). A scope's own events go the moment the scope is
#: reclaimed; this cap is the backstop for the events no scope owns and for a
#: single turn that talks more than the whole process should remember.
EVENT_BUFFER_MAX_ENV = "FW_OFFLOAD_EVENT_BUFFER_MAX"
DEFAULT_EVENT_BUFFER_MAX = 2000

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_handles: dict[str, dict[str, Any]] = {}
_archived: dict[str, dict[str, Any]] = {}
_search_answers: dict[str, int] = {}
#: ido-8ps.13. ``handle_key(scope, alias) -> context clause``, written by
#: ``CommandExecutor.invoke_command`` BEFORE the command runs and read when
#: the alias line is printed. Turn-scoped like everything else here.
_context_clauses: dict[str, str] = {}
_events: list[dict[str, Any]] = []
_event_log_failures: set[str] = set()
_event_cap_warnings: set[str] = set()
_default_archive: Optional[RuntimeHandleArchive] = None
_default_scope = RuntimeHandleScope(
    store_identity=f"process-{os.getpid()}",
    channel_id=f"process-{os.getpid()}",
    experiment_id="unbound",
    task_id="unbound",
    attempt=0,
    turn_key=f"process-{os.getpid()}",
)


def event_buffer_max_from_env() -> int:
    """How many events this process keeps in memory. ``0`` is not allowed."""
    raw = os.environ.get(EVENT_BUFFER_MAX_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
        if raw not in _event_cap_warnings:
            _event_cap_warnings.add(raw)
            logger.warning(
                "ignoring %s=%s: expected a positive integer",
                EVENT_BUFFER_MAX_ENV, raw,
            )
    return DEFAULT_EVENT_BUFFER_MAX


def hot_handle_max_bytes_from_env() -> int:
    """The hot-cache cap for this run. See ``fastworkflow.context_budget``."""
    return context_budget.offload_hot_max_bytes()


def archive() -> RuntimeHandleArchive:
    """The process-default archive, for a caller with no archive of its own.

    ``build_tool_agent`` always passes the workflow's archive, beside its
    observability database; this per-process file is the fallback for a
    direct call from a command frame before an agent has been built.
    """
    global _default_archive
    if _default_archive is None:
        _default_archive = RuntimeHandleArchive(os.path.join(
            tempfile.gettempdir(), f"fw-offload-handles-{os.getpid()}.sqlite3"
        ))
    return _default_archive


def scope_for_host(host: Any) -> RuntimeHandleScope:
    """The turn scope of the session this call is running under.

    ``build_tool_agent`` resolves the same scope for the ReAct loop; this is the
    same computation reached from a command's own frame, where the only handle
    on the session is the trace host ``CommandExecutor.invoke_command`` bound.
    Keeping one implementation matters: a scope computed two ways is two scopes
    the moment either changes, and a handle stored under one of them would be
    unreachable under the other.
    """
    from fastworkflow import state_paths, tracing

    claim = tracing.get_experiment_claim(host)
    channel_id = str(tracing.get_channel_id(host) or "unbound")
    turn_key = str(tracing.get_turn_key(host) or channel_id)
    sink = tracing.get_sink(host)
    sink_store = getattr(sink, "store", None)
    identity_value = getattr(sink_store, "store_identity", None)
    if callable(identity_value):
        identity_value = identity_value()
    getter = getattr(host, "get_active_workflow", None)
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


def handle_key(scope: RuntimeHandleScope, alias: str) -> str:
    return f"{scope.scope_id}:{alias}"


def default_scope() -> RuntimeHandleScope:
    return _default_scope


def record_event(event: Mapping[str, Any]) -> None:
    """Append to the in-memory log and, when configured, the event file.

    The file write is best effort: offloading is an optimisation, so a full
    disk or a revoked permission on the event log must not abort the agent
    step that produced the event. The first failure per path is logged.
    """
    item = dict(event)
    with _lock:
        _events.append(item)
        # The ring closes here, inside the same lock that appended, so two
        # threads recording at once cannot both skip the trim. A list is kept
        # rather than a deque because callers read this buffer as a list.
        overflow = len(_events) - event_buffer_max_from_env()
        if overflow > 0:
            del _events[:overflow]
        path = os.environ.get(EVENTS_ENV, "").strip()
        if not path:
            return
        try:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                    + "\n"
                )
        except (OSError, TypeError, ValueError) as error:
            if path not in _event_log_failures:
                _event_log_failures.add(path)
                logger.warning(
                    "observation offloading could not append to %s=%s: %s",
                    EVENTS_ENV, path, error,
                )


def snapshot_events() -> list[dict[str, Any]]:
    with _lock:
        return list(_events)


def mark_archived(
    scope: RuntimeHandleScope, alias: str, *, text_sha256: str, inline: bool = True
) -> None:
    """Remember that this alias is durable, and whether it is still inline.

    The digest lets the eager archiver skip a re-write of text it already wrote
    in this process (compaction revisits every execute step at every step), and
    the ``inline`` flag lets a search event separate a miss on a handle that was
    never printed from a miss on an observation the agent could still read.
    """
    with _lock:
        _archived[handle_key(scope, alias)] = {
            "text_sha256": text_sha256,
            "inline": inline,
        }


def mark_offloaded(scope: RuntimeHandleScope, alias: str) -> None:
    """The trajectory now carries a label for this alias instead of its text."""
    key = handle_key(scope, alias)
    with _lock:
        entry = _archived.get(key)
        if entry is None:
            _archived[key] = {"text_sha256": None, "inline": False}
        else:
            entry["inline"] = False


def archived_digest(scope: RuntimeHandleScope, alias: str) -> Optional[str]:
    with _lock:
        entry = _archived.get(handle_key(scope, alias))
    return None if entry is None else entry["text_sha256"]


def observation_inline(scope: RuntimeHandleScope, alias: str) -> Optional[bool]:
    """True while the observation is inline, False once labelled, None if unknown."""
    with _lock:
        entry = _archived.get(handle_key(scope, alias))
    return None if entry is None else bool(entry["inline"])


def next_search_answer_sequence(scope: RuntimeHandleScope) -> int:
    """The next ordinal for an archived search answer in this scope.

    Only used to build a record key (``labels.search_answer_key``) when an
    answer had to be bounded, so repeated searches of the same observation each
    keep their own complete text. It is not an observation ordinal and never
    enters the agent-visible ``O`` namespace.
    """
    with _lock:
        _search_answers[scope.scope_id] = _search_answers.get(scope.scope_id, 0) + 1
        return _search_answers[scope.scope_id]


def record_context_clause(scope: RuntimeHandleScope, alias: str, clause: str) -> None:
    """Remember the context an execute step's command RAN IN (``ido-8ps.13``).

    Written at dispatch, from the context handle taken BEFORE the command
    executes, because that is the fact the observation is evidence about: a
    command that MOVES the context produced its output in the context it ran
    in, not in the one it entered. The alias line is printed later, in the
    ``on_step_complete`` hook, by which time the current context has already
    moved -- so the fact has to be carried, not recomputed.

    An empty clause is stored as an empty clause: "this ran at the root" is a
    fact, and it must not read as "nothing was captured".
    """
    with _lock:
        _context_clauses[handle_key(scope, alias)] = str(clause or "")


def context_clause_of(scope: RuntimeHandleScope, alias: str) -> Optional[str]:
    """The recorded clause for *alias*, ``""`` at the root, None if unrecorded."""
    with _lock:
        return _context_clauses.get(handle_key(scope, alias))


def forget_context_clause(scope: RuntimeHandleScope, alias: str) -> None:
    """Drop the clause recorded for *alias*, so it reads as UNRECORDED again.

    ``ido-8ps.29``. The dispatch-time stamp is a good default and a bad answer
    for one kind of step: a page of a result handle declared somewhere else. If
    the declaring subject turns out to be unknown, "no subject recorded" is the
    truth and the context the agent happened to be standing in is not -- and
    "unrecorded" is a state every reader already handles, where a wrong clause
    is one every reader believes.
    """
    with _lock:
        _context_clauses.pop(handle_key(scope, alias), None)


def reclaim_scope(scope: "RuntimeHandleScope | str") -> None:
    """Drop every process-local cache one FINISHED turn scope holds (``ido-1ew``).

    Residency, never evidence: the archive and the result-handle tables keep
    every row, so a scope reclaimed here is still fully readable from disk --
    which is exactly what the cold-resume path already does in a process that
    never saw the turn at all.

    This is deliberately NOT a global reset. Everything the offloading runtime
    remembers is keyed by ``scope_id``, so one turn's state can be released
    while every other live turn in the process keeps its own. A reset that took
    the lot would invalidate the turns running beside this one.

    The caller decides what "finished" means, and only two places may: a turn
    that is over because the agent has bound the NEXT one
    (``StructuredContinuationReAct.bind_scope``), and a session that is over
    because its execution context was closed or evicted
    (``WorkflowExecutionContext.close``). Neither fires for a SUSPENDED turn,
    because a suspension is state that must outlive the process, not state to
    reclaim.
    """
    scope_id = scope if isinstance(scope, str) else scope.scope_id
    prefix = f"{scope_id}:"
    with _lock:
        for registry in (_handles, _archived, _context_clauses):
            for key in [key for key in registry if key.startswith(prefix)]:
                del registry[key]
        _search_answers.pop(scope_id, None)
        kept = [
            item for item in _events
            if str(item.get("scope_id") or "") != scope_id
        ]
        if len(kept) != len(_events):
            _events[:] = kept
    from fastworkflow import auto_navigation, result_handles

    result_handles.release_scope(scope_id)
    auto_navigation.forget_scope(scope_id)


def reset_runtime_state() -> None:
    """Drop every process-local cache the offloading runtime holds.

    The result-handle caches go with them: they are keyed by the same scope and
    hold rows for the same turn, so leaving them behind would let a new turn
    read a previous one's hot copy. Stored SQLite rows are untouched on both
    sides — this resets residency, never evidence.

    The auto-navigation registry goes with them for the same reason: it is
    turn-scoped by contract (ido-8ps.9), so a handle written in one turn must
    never resolve to a context instance another turn entered.
    """
    global _default_archive
    with _lock:
        _handles.clear()
        _archived.clear()
        _search_answers.clear()
        _context_clauses.clear()
        _events.clear()
        _event_log_failures.clear()
        _default_archive = None
    from fastworkflow import auto_navigation, result_handles

    result_handles.reset_result_handle_state()
    auto_navigation.reset_auto_navigation_state()


def stored_handles(scope: Optional[RuntimeHandleScope] = None) -> dict[str, dict[str, Any]]:
    selected = scope or _default_scope
    prefix = f"{selected.scope_id}:"
    with _lock:
        return {
            str(payload["alias"]): dict(payload)
            for key, payload in _handles.items()
            if key.startswith(prefix)
        }


def remember_handle(scope: RuntimeHandleScope, payload: dict[str, Any]) -> None:
    with _lock:
        _handles[handle_key(scope, str(payload["alias"]))] = payload


def hot_payload_bytes(scope: Optional[RuntimeHandleScope] = None) -> int:
    selected = scope or _default_scope
    prefix = f"{selected.scope_id}:"
    with _lock:
        return sum(
            len(str(value["text"]).encode("utf-8"))
            for key, value in _handles.items()
            if key.startswith(prefix)
        )


def clear_hot_handles(scope: Optional[RuntimeHandleScope] = None) -> None:
    selected = scope or _default_scope
    prefix = f"{selected.scope_id}:"
    with _lock:
        for key in [key for key in _handles if key.startswith(prefix)]:
            del _handles[key]


def evict_hot_handles(scope: RuntimeHandleScope, *, hot_handle_max_bytes: int) -> list[str]:
    evicted: list[str] = []
    prefix = f"{scope.scope_id}:"
    while hot_payload_bytes(scope) > hot_handle_max_bytes:
        with _lock:
            oldest_key = next((key for key in _handles if key.startswith(prefix)), None)
            if oldest_key is None:
                break
            payload = _handles.pop(oldest_key)
        alias = str(payload["alias"])
        evicted.append(alias)
        record_event(
            {
                "kind": "hot_evict",
                "scope_id": scope.scope_id,
                "alias": alias,
                "text_sha256": payload["text_sha256"],
                "hot_payload_bytes": hot_payload_bytes(scope),
            }
        )
    return evicted
