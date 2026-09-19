"""Process-local hot cache and optional event log."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import threading
from contextlib import closing
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
#: One archive object per sidecar FILE, on ``result_handles.store``'s pattern
#: and for the same reason (ido-pg2): two spellings of one path must be one
#: object, and a caller that holds only a store path must be able to reach the
#: durable subject rows in the same file without re-creating the schema on
#: every call.
_archives_by_path: dict[str, RuntimeHandleArchive] = {}
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


def archive_for_path(db_path: str) -> RuntimeHandleArchive:
    """The archive object for one sidecar file, created once per path.

    ``ResultHandleStore`` and ``RuntimeHandleArchive`` are two views of the same
    file by construction, so a caller holding one can reach the other's tables
    here instead of opening a second connection per call.
    """
    key = os.path.abspath(os.path.expanduser(str(db_path)))
    with _lock:
        existing = _archives_by_path.get(key)
    if existing is not None:
        return existing
    created = RuntimeHandleArchive(key)
    with _lock:
        return _archives_by_path.setdefault(key, created)


def durable_archive(selected_archive: Any = None) -> Any:
    """Where this call's DURABLE subject and navigation records belong.

    The caller's archive when it named one; otherwise the archive the running
    agent writes its observations to, so a turn's subject metadata lands in the
    same sidecar as the observations it describes; otherwise the process
    default, which is what every other write in this module falls back to.

    Never raises: durability of presentation metadata must not be able to fail a
    command, so an unresolvable archive is reported as ``None`` and the caller
    keeps working out of its process-local cache alone.
    """
    if selected_archive is not None:
        return selected_archive
    try:
        from fastworkflow.result_handles.paging import _current_agent

        found = getattr(_current_agent(), "observation_archive", None)
        if found is not None:
            return found
    except Exception:  # noqa: BLE001 - no agent, no tracing host, no archive
        logger.debug("no agent archive for durable subject metadata", exc_info=True)
    try:
        return archive()
    except Exception:  # noqa: BLE001
        logger.debug("no default archive for durable subject metadata", exc_info=True)
        return None


def archive() -> RuntimeHandleArchive:
    """The process-default archive, for a caller with no archive of its own.

    ``build_tool_agent`` always passes the workflow's archive, beside its
    observability database; this per-process file is the fallback for a
    direct call from a command frame before an agent has been built.
    """
    global _default_archive
    if _default_archive is None:
        _default_archive = RuntimeHandleArchive(default_archive_path())
    return _default_archive


def default_archive_path() -> str:
    """Where the per-process fallback sidecar lives.

    Named after the pid and in the temp directory, so it is process-local by
    construction. Spelled once, because ``clear_default_cold_records`` has to
    be able to reach the file whether or not this process has instantiated the
    archive object for it.
    """
    return os.path.join(
        tempfile.gettempdir(), f"fw-offload-handles-{os.getpid()}.sqlite3"
    )


#: The tables that exist so a turn can be read back after a restart
#: (``ido-dhw``). They are the only two whose rows a process-local reset has to
#: reach: everything else in the sidecar is evidence a reset never owned.
COLD_RESTART_TABLES = ("observation_subjects", "observation_context_entries")


def clear_default_cold_records(*tables: str) -> None:
    """Empty the PROCESS-DEFAULT sidecar's cold-restart tables.

    The durable half of a process-local reset (``ido-dhw``). Subject clauses and
    navigation entries are now read THROUGH to the sidecar when the in-memory
    registry misses, so a reset that cleared only memory would be answered from
    disk by the very rows it meant to drop. Every caller that never named an
    archive of its own shares this one file AND one ``default_scope``, so those
    rows are exactly the state the reset owns.

    No archive a CALLER named is touched, and nothing but these tables is:
    a real sidecar holds real turns, and this is not an erasure path.
    """
    wanted = tables or COLD_RESTART_TABLES
    path = default_archive_path()
    if not os.path.exists(path):
        return
    try:
        with closing(sqlite3.connect(path, timeout=30.0)) as conn:
            present = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for table in wanted:
                if table in present:
                    conn.execute(f'DELETE FROM "{table}"')
            conn.commit()
    except Exception:  # noqa: BLE001 - a reset must not fail on a temp file
        logger.debug("could not clear the default sidecar's cold-restart records",
                     exc_info=True)


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


def record_context_clause(
    scope: RuntimeHandleScope,
    alias: str,
    clause: str,
    *,
    selected_archive: Any = None,
) -> None:
    """Remember the context an execute step's command RAN IN (``ido-8ps.13``).

    Written at dispatch, from the context handle taken BEFORE the command
    executes, because that is the fact the observation is evidence about: a
    command that MOVES the context produced its output in the context it ran
    in, not in the one it entered. The alias line is printed later, in the
    ``on_step_complete`` hook, by which time the current context has already
    moved -- so the fact has to be carried, not recomputed.

    An empty clause is stored as an empty clause: "this ran at the root" is a
    fact, and it must not read as "nothing was captured".

    ``ido-dhw`` (F3): it is also written THROUGH to the sidecar, because this
    map is turn-scoped process memory and the subject of an observation has to
    outlive the process that saw it. The durable write is best effort on the
    same terms as everything else on this path -- a sidecar that cannot be
    written keeps the observation and loses only its cold-restart subject, and
    says so in the event log rather than failing the command.
    """
    text = str(clause or "")
    with _lock:
        _context_clauses[handle_key(scope, alias)] = text
    _write_subject(scope, alias, text, selected_archive)


def _write_subject(
    scope: RuntimeHandleScope, alias: str, clause: str, selected_archive: Any
) -> None:
    store = durable_archive(selected_archive)
    if store is None:
        return
    try:
        store.put_subject(scope, alias, clause)
    except Exception as error:  # noqa: BLE001 - metadata must never fail a turn
        record_event(
            {
                "kind": "subject_persist_refused",
                "scope_id": scope.scope_id,
                "alias": alias,
                "error": type(error).__name__,
            }
        )


def context_clause_of(
    scope: RuntimeHandleScope, alias: str, *, selected_archive: Any = None
) -> Optional[str]:
    """The recorded clause for *alias*, ``""`` at the root, None if unrecorded.

    Process memory first, then the sidecar (``ido-dhw``, F3). The second tier is
    what makes a subject survive a restart: a rehydrated label, a cross-context
    page stamp, the attribution check and observation search all read the
    subject through here, and in a process that only imported a suspension the
    map is empty while the rows are still on disk. A durable hit refills the map
    -- the bounded runtime cache is REBUILT from the durable record rather than
    kept a second way -- so the read is paid for once per alias per process.

    ``None`` still means UNRECORDED, and it is what an alias stamped before this
    table existed reads as. Nothing here ever invents a subject.
    """
    key = handle_key(scope, alias)
    with _lock:
        if key in _context_clauses:
            return _context_clauses[key]
    store = durable_archive(selected_archive)
    if store is None:
        return None
    try:
        clause = store.get_subject(scope, alias)
    except Exception:  # noqa: BLE001 - an unreadable sidecar is an unrecorded one
        logger.debug("could not read the stored subject of %s", alias, exc_info=True)
        return None
    if clause is None:
        return None
    with _lock:
        _context_clauses.setdefault(key, clause)
    return clause


def forget_context_clause(
    scope: RuntimeHandleScope, alias: str, *, selected_archive: Any = None
) -> None:
    """Drop the clause recorded for *alias*, so it reads as UNRECORDED again.

    ``ido-8ps.29``. The dispatch-time stamp is a good default and a bad answer
    for one kind of step: a page of a result handle declared somewhere else. If
    the declaring subject turns out to be unknown, "no subject recorded" is the
    truth and the context the agent happened to be standing in is not -- and
    "unrecorded" is a state every reader already handles, where a wrong clause
    is one every reader believes.

    The durable row goes with it (``ido-dhw``): a correction that only reached
    process memory would be undone by the next restart, which is the failure
    mode this whole pair exists to prevent.
    """
    with _lock:
        _context_clauses.pop(handle_key(scope, alias), None)
    store = durable_archive(selected_archive)
    if store is None:
        return
    try:
        store.forget_subject(scope, alias)
    except Exception:  # noqa: BLE001 - metadata must never fail a turn
        logger.debug("could not drop the stored subject of %s", alias, exc_info=True)


def seal_scope(
    scope: RuntimeHandleScope, *, selected_archive: Any = None
) -> dict[str, Any]:
    """Seal one FINISHED turn's stored evidence into its redacted form (``ido-6sc``).

    The owner's decision made redaction a turn-COMPLETION step rather than a
    write-time transform: nothing is redacted while a turn is in flight, and in
    flight is the whole life of the turn, an ask_user wait and every
    serialize/deserialize round trip included. This is the moment the rest of
    that decision is paid for.

    It deliberately runs only through the runtime owner's two finished-turn
    paths, immediately before aggregate release. That is not a coincidence: it is the
    requirement: the guards those callers already carry --
    ``WorkflowExecutionContext._reclaim_offloading_scope`` returning early when
    ``self._awaiting_user`` or ``agent.export_suspended() is not None``, and
    ``StructuredContinuationReAct.bind_scope`` skipping while ``self._suspended
    is not None`` -- already mean exactly "this turn is not finished", which is
    the property a seal needs. Inventing a second notion of over would be
    inventing a second way to be wrong about a suspension.

    Both callers also run strictly LATER than the turn's summary. The
    conversation summary that feeds the next turn's query refinement is
    produced inside ``WorkflowExecutionContext._finalize_agent_output``, out of
    the in-memory ``_action_log`` whose ``response`` was captured at execution
    time and is never read back from this archive -- so the summary sees raw
    text by ordering, and the ordering is pinned by a test.

    The scope's hot observations go with the seal. They hold the raw copy, the
    turn that could read it is over, and leaving them would mean memory and
    disk disagreeing for a scope nobody may read again. ``reclaim_scope`` drops
    them anyway at both callers; doing it here too is what makes a seal reached
    any other way -- the sweep, a test -- leave nothing raw behind in process.

    Never raises. Failing to seal is a fidelity and exposure problem to report,
    never a reason to fail a session close or the turn that is starting.
    """
    store = durable_archive(selected_archive)
    if store is None:
        return {"sealed": 0, "redacted": 0, "failed": 0, "aliases": [],
                "errors": ["no_archive"]}
    try:
        result = store.seal_scope(scope)
    except Exception as error:  # noqa: BLE001 - a seal must not fail a turn
        record_event(
            {
                "kind": "seal_refused",
                "scope_id": scope.scope_id,
                "error": type(error).__name__,
            }
        )
        logger.warning(
            "could not seal the evidence of scope %s: %s", scope.scope_id, error
        )
        return {"sealed": 0, "redacted": 0, "failed": 0, "aliases": [],
                "errors": [type(error).__name__]}
    clear_hot_handles(scope)
    if result.get("sealed") or result.get("failed"):
        record_event(
            {
                "kind": "observations_sealed",
                "scope_id": scope.scope_id,
                "sealed": int(result.get("sealed") or 0),
                "redacted": int(result.get("redacted") or 0),
                "failed": int(result.get("failed") or 0),
                "aliases": list(result.get("aliases") or ()),
            }
        )
    return result


def release_scope(scope: "RuntimeHandleScope | str") -> None:
    """Drop this component's process-local cache for one finished scope.

    Residency, never evidence: the archive and the result-handle tables keep
    every row, so a scope reclaimed here is still fully readable from disk --
    which is exactly what the cold-resume path already does in a process that
    never saw the turn at all. That now includes the subject clauses and the
    navigation entries dropped below (``ido-dhw``): both tiers are dropped from
    memory and neither row is deleted, so a later read of a reclaimed scope
    rebuilds from the sidecar rather than answering "unrecorded".

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

    ``seal_scope`` runs immediately before this at both of them (``ido-6sc``),
    on the strength of those same two guards: the earliest honest moment to
    release a turn's residency is also the earliest honest moment to redact its
    evidence, and one notion of "over" serves both.
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


def reset_observation_state() -> None:
    """Drop every process-local cache the offloading runtime holds.

    Result-handle and auto-navigation caches are reset by the runtime owner,
    which calls each component's reset hook. Stored SQLite rows are untouched:
    this resets residency, never evidence.

    The PROCESS-DEFAULT sidecar's durable subject and navigation rows go too
    (ido-dhw). That file is ``fw-offload-handles-<pid>.sqlite3`` in the temp
    directory -- process-local by construction, named after this process, and
    shared by every caller that never passed an archive of its own, all of whom
    also share one ``default_scope``. Leaving those two tables behind would let
    a reset process read back the subject and the navigation entries of the
    state it just dropped. Nothing else in the file is touched, and no archive a
    CALLER named is touched at all: those are real sidecars holding real turns.
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
        _archives_by_path.clear()
    clear_default_cold_records()


def reclaim_scope(scope: "RuntimeHandleScope | str") -> None:
    """Compatibility import; aggregate ownership lives in ``agent_runtime``."""
    from fastworkflow.agent_runtime import reclaim_scope as runtime_reclaim_scope

    runtime_reclaim_scope(scope)


def reset_runtime_state() -> None:
    """Compatibility import; aggregate ownership lives in ``agent_runtime``."""
    from fastworkflow.agent_runtime import reset_runtime_state as runtime_reset

    runtime_reset()


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
