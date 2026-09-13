"""Process-local hot cache and optional event log."""
from __future__ import annotations

import json
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

_lock = threading.Lock()
_handles: dict[str, dict[str, Any]] = {}
_events: list[dict[str, Any]] = []
_default_archive: Optional[RuntimeHandleArchive] = None
_default_scope = RuntimeHandleScope(
    store_identity=f"process-{os.getpid()}",
    channel_id=f"process-{os.getpid()}",
    experiment_id="unbound",
    task_id="unbound",
    attempt=0,
    turn_key=f"process-{os.getpid()}",
)


def hot_handle_max_bytes_from_env(default: int = HOT_HANDLE_MAX_BYTES) -> int:
    raw = os.environ.get(HOT_HANDLE_MAX_BYTES_ENV, "").strip()
    return default if not raw else int(raw)


def archive() -> RuntimeHandleArchive:
    global _default_archive
    if _default_archive is None:
        raw = os.environ.get(HANDLE_ARCHIVE_ENV, "").strip()
        path = raw or os.path.join(
            tempfile.gettempdir(), f"fw-offload-handles-{os.getpid()}.sqlite3"
        )
        _default_archive = RuntimeHandleArchive(path)
    return _default_archive


def handle_key(scope: RuntimeHandleScope, alias: str) -> str:
    return f"{scope.scope_id}:{alias}"


def default_scope() -> RuntimeHandleScope:
    return _default_scope


def record_event(event: Mapping[str, Any]) -> None:
    item = dict(event)
    with _lock:
        _events.append(item)
        path = os.environ.get(EVENTS_ENV, "").strip()
        if not path:
            return
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def snapshot_events() -> list[dict[str, Any]]:
    with _lock:
        return list(_events)


def reset_runtime_state() -> None:
    global _default_archive
    with _lock:
        _handles.clear()
        _events.clear()
        _default_archive = None


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
