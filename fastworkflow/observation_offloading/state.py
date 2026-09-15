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

HOT_HANDLE_MAX_BYTES = 262_144
HANDLE_ARCHIVE_ENV = "FW_OFFLOAD_HANDLE_ARCHIVE"
HOT_HANDLE_MAX_BYTES_ENV = "FW_OFFLOAD_HOT_MAX_BYTES"
EVENTS_ENV = "FW_OFFLOAD_EVENTS"

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_handles: dict[str, dict[str, Any]] = {}
_archived: dict[str, dict[str, Any]] = {}
_search_answers: dict[str, int] = {}
_events: list[dict[str, Any]] = []
_event_log_failures: set[str] = set()
_default_archive: Optional[RuntimeHandleArchive] = None
_default_scope = RuntimeHandleScope(
    store_identity=f"process-{os.getpid()}",
    channel_id=f"process-{os.getpid()}",
    experiment_id="unbound",
    task_id="unbound",
    attempt=0,
    turn_key=f"process-{os.getpid()}",
)


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Read a non-negative integer knob, falling back to ``default`` on bad input.

    These parsers run on the compaction hot path of every agent step, so a
    mistyped export must degrade to the default with a warning rather than
    abort the turn.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using default %d", name, raw, default
        )
        return default
    if value < minimum:
        logger.warning(
            "%s=%d is below the minimum %d; using default %d",
            name, value, minimum, default,
        )
        return default
    return value


def hot_handle_max_bytes_from_env(default: int = HOT_HANDLE_MAX_BYTES) -> int:
    return env_int(HOT_HANDLE_MAX_BYTES_ENV, default)


def archive() -> RuntimeHandleArchive:
    global _default_archive
    if _default_archive is None:
        raw = os.environ.get(HANDLE_ARCHIVE_ENV, "").strip()
        path = raw or os.path.join(
            tempfile.gettempdir(), f"fw-offload-handles-{os.getpid()}.sqlite3"
        )
        _default_archive = RuntimeHandleArchive(path)
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
