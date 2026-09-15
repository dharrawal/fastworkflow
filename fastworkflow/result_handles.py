"""Paging and literal lookup over a listing a command already ran.

A listing command answers with every row it materialised — 477 `uid  label`
lines — and the ReAct trajectory then carries all of it for the rest of the
turn. Observation offloading can move that text out of the prompt after the
fact, but the agent still has to re-read the whole listing through
``search_memory`` to reach the five people an utterance actually named.

This module stores the listing instead: the rendered rows, the *serialisable*
description of the query that produced them, and (ido-986.14.2) the ability to
continue that query against the backend. A workflow reaches it through two
calls: ``declare`` from the command that produced the listing, and
``fetch_page`` from the workflow's own fetch command.

Identity. A handle is the canonical ``O`` alias of the execute step that
declared it — the alias observation offloading (A1) prints on that step's
observation — so there is no second agent-visible namespace to be confused with
ReAct step numbers. A fetch call is itself an execute step with its own ``O``
alias: its page observation is immutable, it is archived under that alias like
any other execute observation (A2), and it links internally back to the listing
handle it paged, so the parent is discoverable from the page.

Nothing callable is ever persisted. A resolver is registered in-process under a
name; the stored descriptor only names it, alongside the view, params, verified
filter columns, page size and ordering policy needed to re-issue the query.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import unicodedata
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.observation_offloading.state import (
    default_scope,
    env_int,
    record_event,
    scope_for_host,
)

logger = logging.getLogger(__name__)

#: A page observation is a listing observation, so it gets the listing budget:
#: 3 KB of the ReAct prompt, header line included. Rows that do not fit are not
#: dropped — they stay in the store and the next cursor returns them.
RESULT_PAGE_MAX_BYTES = 3_072
RESULT_PAGE_MAX_BYTES_ENV = "FW_RESULT_PAGE_MAX_BYTES"
#: Below this the header alone would consume the budget it is describing.
RESULT_PAGE_MIN_BYTES = 512

#: Rows held in this process. The durable copy is SQLite, so eviction costs a
#: re-read and never loses a stored page.
HOT_ROWS_MAX_BYTES = 262_144
HOT_ROWS_MAX_BYTES_ENV = "FW_RESULT_HANDLE_HOT_MAX_BYTES"

DEFAULT_PAGE_SIZE = 25

#: The only ordering policy a descriptor may name. B0 (ido-gqv.6) measured an
#: explicit ``sort`` combined with offset paging silently dropping 20 of 540
#: group members while returning exactly ``total`` rows, so a stored descriptor
#: cannot express a sorted walk at all: there is no field to put one in.
UNSORTED_OFFSET = "unsorted-offset"

CURSOR_VERSION = 1

#: ``%`` and ``_`` are LIKE wildcards on the portal and ``*`` behaves as one
#: too; backslash escaping does not work (B0 §b). A literal filter therefore
#: cannot be delivered by passing the agent's text through, so the characters
#: are removed and the observation says which literal was really sent.
WILDCARD_CHARACTERS = "%_*"

#: The model emits U+00A0 and U+2011 inside these very names, and the portal
#: answers a Unicode-contaminated filter with a confident, silent zero.
_SPACE_LIKE = {0x00A0: " ", 0x2007: " ", 0x202F: " ", 0x2009: " ", 0x2011: "-"}
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")

_ALIAS_RE = re.compile(r"^(?:O[1-9]\d*|D[1-9]\d*)$")

#: How the producer renders a row: `uid` then two spaces then the label. Kept
#: identical to the listing text the command returned, because a filter has to
#: be able to find "Alan Cooper" in the row the agent was shown.
ROW_SEPARATOR = "  "


class ResultHandleError(RuntimeError):
    """A handle, cursor or filter a caller can act on — never a crash.

    Unknown handle, a cursor written for another query scope, a descriptor that
    names an unregistered resolver: each is a reached decision, declined by
    name, so the calling command can refuse in the agent's own terms.
    """


# ---------------------------------------------------------------------------
# Serialisable source description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDescriptor:
    """Everything needed to re-issue the producing query, and nothing callable.

    ``resolver`` names a resolver registered in this process (see
    ``register_resolver``); the rest is JSON. There is deliberately no ``sort``
    field and no ``timeslot`` value other than ``None``: B0 established that
    the views C1 pages have a stable, complete, repeatable default order with
    no timeslot sent, and that an explicit sort is what breaks offset paging.
    ``timeslot`` is carried explicitly as ``None`` so evidence records that the
    read had no pin rather than leaving the question open.
    """

    resolver: str
    view: str
    params: Mapping[str, Any] = field(default_factory=dict)
    #: Columns the filter may be mapped to, already verified against this view.
    #: Empty means literal filtering is unsupported for this handle — a filter
    #: sent without columns is silently ignored by the portal and returns the
    #: whole scope, so that pair must be impossible to emit.
    filter_columns: Sequence[str] = ()
    uid_field: str = ""
    label_fields: Sequence[str] = ()
    page_size: int = DEFAULT_PAGE_SIZE
    ordering: str = UNSORTED_OFFSET
    #: Backend offset the producer's own first row came from.
    start_offset: int = 0
    #: Rows the producer already materialised from ``start_offset``. The walk
    #: continues at ``start_offset + materialized``.
    materialized: int = 0
    timeslot: None = None
    role: Optional[str] = None
    count_only: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resolver:
            raise ResultHandleError("a source descriptor must name a resolver")
        if self.ordering != UNSORTED_OFFSET:
            raise ResultHandleError(
                "ordering %r is not available: C1 walks offsets in the view's "
                "default order only (ido-gqv.6 B0)" % (self.ordering,)
            )
        if self.timeslot is not None:
            raise ResultHandleError(
                "no timeslot pin exists for these views; the descriptor records "
                "timeslot=None (ido-986.14.1)"
            )
        if int(self.page_size) < 1:
            raise ResultHandleError("page_size must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filter_columns"] = list(self.filter_columns)
        payload["label_fields"] = list(self.label_fields)
        payload["params"] = dict(self.params)
        payload["extra"] = dict(self.extra)
        return payload

    @property
    def digest(self) -> str:
        return _digest(_canonical_json(self.as_dict()))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceDescriptor":
        known = {key: payload[key] for key in payload if key in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ResultHandleSpec:
    """What a producing command declares about the listing it just rendered.

    The field names are the ones the producing command already uses for its own
    rendering, so a workflow declares what it showed rather than translating it.
    ``items`` are the rendered ``uid  label`` lines exactly as the response
    carried them: a literal filter has to be able to find a name in the row the
    agent read.
    """

    kind: str
    summary: str = ""
    items: Sequence[str] = ()
    ordering: str = UNSORTED_OFFSET
    total: int = 0
    source_complete: bool = True
    page_size: int = DEFAULT_PAGE_SIZE
    classification: str = "user-text"
    presentation: bool = True
    filters: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resolver registry (in-process; never persisted)
# ---------------------------------------------------------------------------

_resolvers: dict[str, Callable[..., Any]] = {}


def register_resolver(kind_or_workflow: str, resolver: Callable[..., Any]) -> None:
    """Bind a name a descriptor may reference to the callable that executes it.

    The registry is process-local and is never written to SQLite: a stored
    descriptor that names a resolver this process has not registered is refused
    by name, which is a recoverable answer, where a persisted callable would be
    an unsafe one.
    """
    if not kind_or_workflow:
        raise ResultHandleError("a resolver needs a non-empty name")
    if not callable(resolver):
        raise ResultHandleError("a resolver must be callable")
    _resolvers[str(kind_or_workflow)] = resolver


def unregister_resolver(kind_or_workflow: str) -> None:
    _resolvers.pop(str(kind_or_workflow), None)


def registered_resolvers() -> tuple[str, ...]:
    return tuple(sorted(_resolvers))


def resolver_for(name: str) -> Callable[..., Any]:
    try:
        return _resolvers[str(name)]
    except KeyError:
        raise ResultHandleError(
            "no resolver named %r is registered in this process; the stored "
            "rows are still readable, but this handle cannot be continued"
            % (name,)
        ) from None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str
    ).encode("utf-8")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ResultHandleStore:
    """Turn-scoped SQLite for declarations and immutable raw page records.

    It lives in the same database file as ``observation_offload_handles`` and
    follows the same pattern — scope-keyed rows, digest-verified payloads,
    insert-or-nothing writes — in its own tables. The offload table is not
    touched.

    Retention. Rows are never deleted by this module. They live exactly as long
    as the archive file that holds the turn's observations, which is what makes
    a page reconstructable for evaluation after the live turn has ended; the hot
    cache bound is a residency bound and not a retention bound.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_handle_declarations (
                    scope_id TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    ordering TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    materialized INTEGER NOT NULL,
                    source_complete INTEGER NOT NULL,
                    page_size INTEGER NOT NULL,
                    classification TEXT NOT NULL,
                    presentation INTEGER NOT NULL,
                    filters_json TEXT NOT NULL,
                    descriptor_json TEXT NOT NULL,
                    descriptor_sha256 TEXT NOT NULL,
                    columns_json TEXT NOT NULL,
                    sample_row_json TEXT NOT NULL,
                    parent_alias TEXT NOT NULL,
                    query_scope TEXT NOT NULL,
                    cursor_position INTEGER NOT NULL,
                    declared_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_handle_pages (
                    scope_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    query_scope TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    limit_requested INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    backend_total INTEGER,
                    record_json BLOB NOT NULL,
                    record_sha256 TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias, query_scope, start_offset)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS result_handle_pages_walk
                ON result_handle_pages(scope_id, alias, query_scope, start_offset)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    # -- declarations ------------------------------------------------------

    def put_declaration(
        self, scope: RuntimeHandleScope, alias: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Write a declaration once. A redeclaration of the same query is a no-op.

        Two different queries under one alias would make the alias ambiguous —
        the agent would ask for O42 and get whichever was written last — so the
        second one is refused by name instead.
        """
        scope_json = json.dumps(
            asdict(scope), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO result_handle_declarations (
                    scope_id, scope_json, alias, kind, summary, ordering, total,
                    materialized, source_complete, page_size, classification,
                    presentation, filters_json, descriptor_json,
                    descriptor_sha256, columns_json, sample_row_json,
                    parent_alias, query_scope, cursor_position, declared_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias) DO NOTHING
                """,
                (
                    scope.scope_id,
                    scope_json,
                    alias,
                    str(payload["kind"]),
                    str(payload["summary"]),
                    str(payload["ordering"]),
                    int(payload["total"]),
                    int(payload["materialized"]),
                    1 if payload["source_complete"] else 0,
                    int(payload["page_size"]),
                    str(payload["classification"]),
                    1 if payload["presentation"] else 0,
                    json.dumps(dict(payload["filters"]), sort_keys=True),
                    json.dumps(payload["descriptor"], sort_keys=True),
                    str(payload["descriptor_sha256"]),
                    json.dumps(payload.get("columns") or {}, sort_keys=True),
                    json.dumps(payload.get("sample_row") or {}, sort_keys=True),
                    str(payload.get("parent_alias") or ""),
                    str(payload.get("query_scope") or ""),
                    int(payload.get("cursor_position") or 0),
                    _now(),
                ),
            )
            conn.commit()
        stored = self.get_declaration(scope, alias)
        if stored is None:
            raise ResultHandleError("result handle %s could not be stored" % alias)
        if stored["descriptor_sha256"] != payload["descriptor_sha256"]:
            raise ResultHandleError(
                "result handle %s already describes a different query in this "
                "scope; an alias identifies one observation" % alias
            )
        return stored

    def get_declaration(
        self, scope: RuntimeHandleScope, alias: str
    ) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM result_handle_declarations "
                "WHERE scope_id = ? AND alias = ?",
                (scope.scope_id, alias),
            ).fetchone()
        return None if row is None else self._decode_declaration(row)

    def list_declarations(self, scope: RuntimeHandleScope) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM result_handle_declarations WHERE scope_id = ? "
                "ORDER BY declared_at, alias",
                (scope.scope_id,),
            ).fetchall()
        return [self._decode_declaration(row) for row in rows]

    def set_verified_columns(
        self,
        scope: RuntimeHandleScope,
        alias: str,
        *,
        columns: Mapping[str, str],
        sample_row: Mapping[str, Any],
    ) -> None:
        """Record the column names/types and one sample row from the first page.

        Written once: the evidence is what the *first* page actually carried, so
        a later page with a different shape must not overwrite the record of
        what was verified.
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE result_handle_declarations
                SET columns_json = ?, sample_row_json = ?
                WHERE scope_id = ? AND alias = ? AND columns_json IN ('', '{}')
                """,
                (
                    json.dumps(dict(columns), sort_keys=True, default=str),
                    json.dumps(dict(sample_row), sort_keys=True, default=str),
                    scope.scope_id,
                    alias,
                ),
            )
            conn.commit()

    @staticmethod
    def _decode_declaration(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "alias": str(row["alias"]),
            "kind": str(row["kind"]),
            "summary": str(row["summary"]),
            "ordering": str(row["ordering"]),
            "total": int(row["total"]),
            "materialized": int(row["materialized"]),
            "source_complete": bool(row["source_complete"]),
            "page_size": int(row["page_size"]),
            "classification": str(row["classification"]),
            "presentation": bool(row["presentation"]),
            "filters": json.loads(row["filters_json"]),
            "descriptor": json.loads(row["descriptor_json"]),
            "descriptor_sha256": str(row["descriptor_sha256"]),
            "columns": json.loads(row["columns_json"] or "{}"),
            "sample_row": json.loads(row["sample_row_json"] or "{}"),
            "parent_alias": str(row["parent_alias"]),
            "query_scope": str(row["query_scope"]),
            "cursor_position": int(row["cursor_position"]),
            "declared_at": str(row["declared_at"]),
        }

    # -- immutable raw pages ----------------------------------------------

    def put_page(
        self,
        scope: RuntimeHandleScope,
        *,
        alias: str,
        query_scope: str,
        start_offset: int,
        limit_requested: int,
        source: str,
        record: Mapping[str, Any],
        backend_total: Optional[int],
    ) -> dict[str, Any]:
        """Append one raw page. Re-fetching an offset returns the stored page.

        Append-only and idempotent by construction: the insert cannot overwrite,
        and the read-back is the value returned, so a retry of the same offset
        can never produce a second row or a different answer than the first
        attempt already recorded.
        """
        payload = _canonical_json(dict(record))
        digest = _digest(payload)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO result_handle_pages (
                    scope_id, alias, query_scope, start_offset, limit_requested,
                    source, row_count, backend_total, record_json,
                    record_sha256, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias, query_scope, start_offset) DO NOTHING
                """,
                (
                    scope.scope_id,
                    alias,
                    query_scope,
                    int(start_offset),
                    int(limit_requested),
                    source,
                    int(len(record.get("rows") or [])),
                    None if backend_total is None else int(backend_total),
                    payload,
                    digest,
                    _now(),
                ),
            )
            conn.commit()
        stored = self.get_page(scope, alias=alias, query_scope=query_scope,
                               start_offset=start_offset)
        if stored is None:
            raise ResultHandleError(
                "page at offset %d of %s could not be stored" % (start_offset, alias)
            )
        return stored

    def get_page(
        self,
        scope: RuntimeHandleScope,
        *,
        alias: str,
        query_scope: str,
        start_offset: int,
    ) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT * FROM result_handle_pages
                WHERE scope_id = ? AND alias = ? AND query_scope = ?
                  AND start_offset = ?
                """,
                (scope.scope_id, alias, query_scope, int(start_offset)),
            ).fetchone()
        return None if row is None else self._decode_page(row)

    def list_pages(
        self, scope: RuntimeHandleScope, *, alias: str, query_scope: str
    ) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM result_handle_pages
                WHERE scope_id = ? AND alias = ? AND query_scope = ?
                ORDER BY start_offset
                """,
                (scope.scope_id, alias, query_scope),
            ).fetchall()
        return [self._decode_page(row) for row in rows]

    @staticmethod
    def _decode_page(row: sqlite3.Row) -> dict[str, Any]:
        payload = bytes(row["record_json"])
        digest = _digest(payload)
        if digest != row["record_sha256"]:
            raise ResultHandleError(
                "stored page %s@%s failed digest verification"
                % (row["alias"], row["start_offset"])
            )
        return {
            "alias": str(row["alias"]),
            "query_scope": str(row["query_scope"]),
            "start_offset": int(row["start_offset"]),
            "limit_requested": int(row["limit_requested"]),
            "source": str(row["source"]),
            "row_count": int(row["row_count"]),
            "backend_total": (None if row["backend_total"] is None
                              else int(row["backend_total"])),
            "record": json.loads(payload.decode("utf-8")),
            "record_sha256": digest,
            "fetched_at": str(row["fetched_at"]),
        }


# ---------------------------------------------------------------------------
# Process-local state: the store handle, the hot rows, the per-turn counters
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_default_store: Optional[ResultHandleStore] = None
_hot: "dict[str, dict[str, Any]]" = {}
_pages_served: "dict[str, set[int]]" = {}
_local_sequence: "dict[str, int]" = {}


def hot_rows_max_bytes_from_env(default: int = HOT_ROWS_MAX_BYTES) -> int:
    return env_int(HOT_ROWS_MAX_BYTES_ENV, default)


def page_max_bytes_from_env(default: int = RESULT_PAGE_MAX_BYTES) -> int:
    return env_int(RESULT_PAGE_MAX_BYTES_ENV, default, minimum=RESULT_PAGE_MIN_BYTES)


def store() -> ResultHandleStore:
    """The database the running turn's handles live in.

    The same file the observation archive uses, so a page and the observation
    that showed it survive together: one file to keep, one file to read back
    when an experiment is scored.
    """
    global _default_store
    with _lock:
        if _default_store is not None:
            return _default_store
    path = _default_store_path()
    created = ResultHandleStore(path)
    with _lock:
        if _default_store is None:
            _default_store = created
        return _default_store


def _default_store_path() -> str:
    from fastworkflow.observation_offloading import state as offload_state

    agent = _current_agent()
    archive = getattr(agent, "observation_archive", None)
    path = getattr(archive, "db_path", "")
    if path:
        return str(path)
    return offload_state.archive().db_path


def reset_result_handle_state() -> None:
    """Drop the process-local caches. Stored rows are untouched by design."""
    global _default_store
    with _lock:
        _default_store = None
        _hot.clear()
        _pages_served.clear()
        _local_sequence.clear()


def _hot_key(scope: RuntimeHandleScope, alias: str, query_scope: str) -> str:
    return "%s:%s:%s" % (scope.scope_id, alias, query_scope)


def _hot_bytes() -> int:
    return sum(int(entry.get("bytes") or 0) for entry in _hot.values())


def _remember_walk(key: str, walk: dict[str, Any]) -> list[str]:
    """Cache a walk and evict oldest-first when over the hot bound.

    Eviction is free of consequence: every row in a walk came from a stored page
    and is rebuilt from SQLite on the next read, so the bound controls memory
    and never reachability.
    """
    evicted: list[str] = []
    with _lock:
        walk["bytes"] = sum(len(record["line"].encode("utf-8")) for record in walk["records"])
        _hot[key] = walk
        limit = hot_rows_max_bytes_from_env()
        while _hot_bytes() > limit and len(_hot) > 1:
            oldest = next(iter(_hot))
            if oldest == key:
                # Never evict the walk being built: the rest of this call still
                # needs it, and it will be rebuilt from SQLite next time anyway.
                oldest = next((candidate for candidate in _hot if candidate != key), None)
                if oldest is None:
                    break
            _hot.pop(oldest, None)
            evicted.append(oldest)
    if evicted:
        record_event({"kind": "result_handle_hot_evict", "walks": evicted})
    return evicted


def _cached_walk(key: str) -> Optional[dict[str, Any]]:
    with _lock:
        return _hot.get(key)


# ---------------------------------------------------------------------------
# Identity: scope and the canonical execute alias
# ---------------------------------------------------------------------------


def _current_agent() -> Any:
    """The ReAct agent running this command, when there is one."""
    from fastworkflow import tracing

    host = tracing.current_host()
    if host is None:
        return None
    agent = getattr(host, "workflow_tool_agent", None)
    if agent is None:
        core = getattr(host, "_core", None)
        agent = getattr(core, "workflow_tool_agent", None)
    return agent


def current_scope() -> RuntimeHandleScope:
    """The scope a handle declared right now belongs to.

    The live agent's own ``continuation_scope`` first: that is the scope its
    observations are archived under for this turn, and a handle filed anywhere
    else would be a handle the same turn could not read back. Then the trace
    host (a command running outside the ReAct loop), then the process default.
    """
    from fastworkflow import tracing

    agent = _current_agent()
    scope = getattr(agent, "continuation_scope", None)
    if isinstance(scope, RuntimeHandleScope):
        return scope
    host = tracing.current_host()
    if host is not None:
        try:
            return scope_for_host(host)
        except Exception:  # noqa: BLE001
            logger.debug("result handles could not resolve a host scope", exc_info=True)
    return default_scope()


def current_execute_alias(agent: Any = None) -> Optional[str]:
    """The ``O`` alias of the execute step this command is running inside.

    ReAct writes ``tool_name_{idx}`` before it calls the tool and
    ``observation_{idx}`` after it returns, so during a command the in-flight
    step is the last one with no observation. Its ordinal is the number of
    execute steps in ``current_trajectory`` — which is never truncated — and
    that is exactly the alias ``annotate_execute_observations`` will print on
    this step's observation when it completes.

    ``None`` when there is no agent step in flight: a direct user command, a
    non-execute tool, or offloading turned off. There is no agent-visible
    namespace in that case, so there is no alias to be wrong about.
    """
    agent = agent if agent is not None else _current_agent()
    trajectory = getattr(agent, "current_trajectory", None)
    if not isinstance(trajectory, Mapping) or not trajectory:
        return None
    indexes = [
        int(key.removeprefix("tool_name_"))
        for key in trajectory
        if key.startswith("tool_name_") and key.removeprefix("tool_name_").isdigit()
    ]
    if not indexes:
        return None
    latest = max(indexes)
    if str(trajectory.get(f"tool_name_{latest}") or "") != "execute_workflow_query":
        return None
    if f"observation_{latest}" in trajectory:
        # The step already completed; this call is not inside it.
        return None
    ordinal = sum(
        1
        for index in indexes
        if str(trajectory.get(f"tool_name_{index}") or "") == "execute_workflow_query"
    )
    return f"O{ordinal}" if ordinal else None


def _local_alias(scope: RuntimeHandleScope) -> str:
    """A store key for a declaration made outside an agent step.

    Deliberately not an ``O``: the ``O`` namespace is execute ordinals, and this
    key exists only where there is no agent to show it to.
    """
    with _lock:
        _local_sequence[scope.scope_id] = _local_sequence.get(scope.scope_id, 0) + 1
        return "D%d" % _local_sequence[scope.scope_id]


# ---------------------------------------------------------------------------
# Literal normalisation and query scoping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Literal:
    """A filter literal as it will really be matched, plus what changed."""

    raw: str
    text: str
    notes: tuple[str, ...] = ()

    @property
    def scope(self) -> str:
        return "" if not self.text else "c:" + _digest(self.text.casefold().encode("utf-8"))[:16]


def normalize_literal(raw: Optional[str]) -> Literal:
    """NFKC, space-like and zero-width repair, whitespace collapse, wildcards out.

    The portal normalises nothing: an NBSP inside "Alan Cooper", a fullwidth C,
    a stray zero-width space each return a confident, silent zero, and the model
    emits exactly those characters. ``%``, ``_`` and ``*`` are LIKE wildcards
    with no working escape, so a filter described to the agent as literal cannot
    pass them through; they are removed and the observation reports the literal
    that was actually used.
    """
    if raw is None:
        return Literal(raw="", text="")
    text = str(raw)
    notes: list[str] = []
    folded = unicodedata.normalize("NFKC", text).translate(_SPACE_LIKE)
    folded = _ZERO_WIDTH.sub("", folded)
    folded = " ".join(folded.split())
    if folded != text:
        notes.append("normalised")
    if any(character in folded for character in WILDCARD_CHARACTERS):
        folded = "".join(
            character for character in folded if character not in WILDCARD_CHARACTERS
        )
        folded = " ".join(folded.split())
        notes.append(
            "wildcards %s removed: the backend treats them as LIKE wildcards and "
            "no escape works, so they cannot be matched literally"
            % " ".join(WILDCARD_CHARACTERS)
        )
    return Literal(raw=text, text=folded, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Cursors
# ---------------------------------------------------------------------------


def encode_cursor(*, alias: str, query_scope: str, position: int, descriptor_sha256: str) -> str:
    payload = {
        "v": CURSOR_VERSION,
        "h": alias,
        "q": query_scope,
        "p": int(position),
        "d": descriptor_sha256[:16],
    }
    return base64.urlsafe_b64encode(_canonical_json(payload)).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> dict[str, Any]:
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception as error:  # noqa: BLE001
        raise ResultHandleError(
            "this cursor is not readable by this build (%s); omit cursor to "
            "start the same query at its first page" % type(error).__name__
        ) from error
    if not isinstance(payload, dict) or int(payload.get("v") or 0) != CURSOR_VERSION:
        raise ResultHandleError(
            "this cursor was written by a different version of the page store; "
            "omit cursor to start the same query at its first page"
        )
    return payload


def _describe_scope(query_scope: str, literal: Literal) -> str:
    if not query_scope:
        return "the unfiltered listing"
    return 'the filter contains="%s"' % literal.text


def _check_cursor(
    payload: Mapping[str, Any],
    *,
    alias: str,
    query_scope: str,
    literal: Literal,
    descriptor_sha256: str,
) -> int:
    """Refuse a cursor from another handle, filter or descriptor, by name."""
    if str(payload.get("h") or "") != alias:
        raise ResultHandleError(
            "this cursor belongs to result handle %s, not %s"
            % (payload.get("h"), alias)
        )
    if str(payload.get("q") or "") != query_scope:
        raise ResultHandleError(
            "this cursor belongs to a different query on %s (%s); omit cursor to "
            "start %s at its first page"
            % (
                alias,
                "the unfiltered listing" if not payload.get("q") else "another filter",
                _describe_scope(query_scope, literal),
            )
        )
    if str(payload.get("d") or "") != descriptor_sha256[:16]:
        raise ResultHandleError(
            "this cursor was written for a different query descriptor on %s; "
            "omit cursor to start again at the first page" % alias
        )
    position = int(payload.get("p") or 0)
    if position < 0:
        raise ResultHandleError("this cursor names a negative page position")
    return position


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _uid_of_line(line: str) -> str:
    return line.split(ROW_SEPARATOR, 1)[0].strip()


def _records_from_items(items: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for line in items:
        text = str(line)
        records.append({"uid": _uid_of_line(text), "line": text, "row": None})
    return records


def _record_of(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"uid": str(record["uid"]), "line": str(record["line"]),
            "row": record.get("row")}


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


@dataclass
class ResultPage:
    """One rendered page of a stored handle, and the truth about its coverage.

    Every count is about the query that was actually run: with a filter,
    ``matched`` is the filtered population and ``total`` is still the whole
    relation, so a page can never present a filtered count as a population or a
    partial walk as a complete one.
    """

    handle: str
    kind: str
    summary: str
    rows: list[str]
    matched: int
    total: int
    materialized: int
    source_complete: bool
    matched_complete: bool
    continuation: str
    incomplete_reason: Optional[str]
    next_cursor: Optional[str]
    outcome: str
    position: int
    page_index: int
    page_alias: Optional[str] = None
    parent_alias: Optional[str] = None
    literal: Optional[str] = None
    filter_columns: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    observation: str = ""

    def as_observation(self) -> str:
        return self.observation

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filter_columns"] = list(self.filter_columns)
        payload["warnings"] = list(self.warnings)
        payload["notes"] = list(self.notes)
        return payload


def _header(page: "ResultPage") -> str:
    parts = [
        "result_handle=%s" % page.handle,
        "page %d" % page.page_index,
        (
            "rows %d-%d of %d"
            % (page.position + 1, page.position + len(page.rows), page.matched)
            if page.rows
            else "rows 0 of %d" % page.matched
        ),
        "matched=%d" % page.matched,
        "materialized=%d" % page.materialized,
        "total=%d" % page.total,
        "source_complete=%s" % str(page.source_complete).lower(),
        "matched_complete=%s" % str(page.matched_complete).lower(),
        "continuation=%s" % page.continuation,
        "has_more=%s" % str(page.next_cursor is not None).lower(),
    ]
    if page.literal:
        parts.insert(1, 'filter="%s"' % page.literal)
        if page.filter_columns:
            parts.insert(2, "filter_columns=%s" % ",".join(page.filter_columns))
    if page.next_cursor:
        parts.append("next_cursor=%s" % page.next_cursor)
    if page.incomplete_reason:
        parts.append("incomplete_reason=%s" % page.incomplete_reason)
    return " ".join(parts)


def _render(
    page: "ResultPage",
    *,
    budget_bytes: int,
    warnings: Sequence[str] = (),
    reported_budget: Optional[int] = None,
) -> tuple[str, list[str], tuple[str, ...]]:
    """Pack whole rows under the byte budget, header and notes included.

    Rows are never split and never skipped: the packer stops at the last row
    that fits and the next cursor starts at the first one that did not. A single
    row wider than the whole budget is emitted whole with a warning, because
    dropping it would lose evidence and truncating it would invent a row that
    was never returned.
    """
    lines = [line for line in page.rows]
    warnings = list(warnings)
    fixed = [_header(page)]
    if page.summary:
        fixed.append(page.summary)
    for note in page.notes:
        fixed.append(note)
    overhead = len("\n".join(fixed + [""]).encode("utf-8")) if fixed else 0
    for warning in warnings:
        overhead += len(warning.encode("utf-8")) + 1
    shown: list[str] = []
    used = 0
    for line in lines:
        cost = len(line.encode("utf-8")) + 1
        if shown and overhead + used + cost > budget_bytes:
            break
        shown.append(line)
        used += cost
    if shown and overhead + used > budget_bytes:
        # Whole rows or nothing: a cut row is a row that was never returned, and
        # a dropped one is evidence lost. Report the overage and continue.
        warnings.append(
            "Warning: this page is over its %d-byte observation budget. A row is "
            "never cut or skipped, so it is shown whole and the overage is "
            "reported instead." % (reported_budget or budget_bytes)
        )
    return "\n".join(fixed + shown + warnings), shown, tuple(warnings)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def declare(
    spec: ResultHandleSpec,
    *,
    source: Any = None,
    scope: Optional[RuntimeHandleScope] = None,
    selected_store: Optional[ResultHandleStore] = None,
    alias: Optional[str] = None,
    parent_alias: str = "",
    query_scope: str = "",
    cursor_position: int = 0,
) -> dict[str, Any]:
    """Store what a command just listed, and return the payload for its artifacts.

    The command keeps returning its own full text as the command response; this
    only adds a stored, pageable copy plus the handle the agent already sees
    printed on the observation.

    A storage failure is never allowed to fail the command that produced real
    output: it is recorded and the payload says ``declared: false``. A malformed
    spec is a programming error and raises.
    """
    if not isinstance(spec, ResultHandleSpec):
        raise ResultHandleError("declare() takes a ResultHandleSpec")
    descriptor: Optional[SourceDescriptor] = None
    if source is not None:
        descriptor = (
            source
            if isinstance(source, SourceDescriptor)
            else SourceDescriptor.from_mapping(source)
        )
    selected_scope = scope or current_scope()
    items = [str(item) for item in (spec.items or ())]
    handle = alias or current_execute_alias() or _local_alias(selected_scope)
    descriptor_payload = descriptor.as_dict() if descriptor is not None else {}
    descriptor_sha256 = _digest(_canonical_json(descriptor_payload))
    payload = {
        "result_handle": handle,
        "kind": spec.kind,
        "summary": spec.summary,
        "ordering": spec.ordering,
        "total": int(spec.total or len(items)),
        "materialized": len(items),
        "source_complete": bool(spec.source_complete),
        "page_size": int(spec.page_size or DEFAULT_PAGE_SIZE),
        "classification": spec.classification,
        "presentation": bool(spec.presentation),
        "filters": dict(spec.filters or {}),
        "descriptor": descriptor_payload,
        "descriptor_sha256": descriptor_sha256,
        "parent_alias": parent_alias,
        "query_scope": query_scope,
        "cursor_position": int(cursor_position),
        "scope_id": selected_scope.scope_id,
    }
    try:
        store_ = selected_store or store()
        store_.put_declaration(selected_scope, handle, payload)
        store_.put_page(
            selected_scope,
            alias=handle,
            query_scope="",
            start_offset=int(descriptor.start_offset) if descriptor else 0,
            limit_requested=len(items),
            source="producer",
            record={"rows": [], "records": _records_from_items(items)},
            backend_total=int(spec.total or len(items)),
        )
    except ResultHandleError:
        raise
    except Exception as error:  # noqa: BLE001
        record_event(
            {
                "kind": "result_handle_declare_refused",
                "scope_id": selected_scope.scope_id,
                "alias": handle,
                "error": type(error).__name__,
                "reason": "persistence_failed_response_retained",
            }
        )
        return {**payload, "declared": False, "error": type(error).__name__}
    record_event(
        {
            "kind": "result_handle_declared",
            "scope_id": selected_scope.scope_id,
            "alias": handle,
            "kind_name": spec.kind,
            "total": payload["total"],
            "materialized": payload["materialized"],
            "source_complete": payload["source_complete"],
            "descriptor_sha256": descriptor_sha256,
            "parent_alias": parent_alias,
        }
    )
    return {**payload, "declared": True}


def handle_declaration(
    handle: str,
    *,
    scope: Optional[RuntimeHandleScope] = None,
    selected_store: Optional[ResultHandleStore] = None,
) -> dict[str, Any]:
    """The stored declaration for a handle, or a refusal naming what is stored."""
    selected_scope = scope or current_scope()
    store_ = selected_store or store()
    alias = str(handle or "").strip()
    if not alias:
        raise ResultHandleError("no result handle was given")
    declaration = store_.get_declaration(selected_scope, alias)
    if declaration is None:
        known = [row["alias"] for row in store_.list_declarations(selected_scope)]
        raise ResultHandleError(
            "no stored result handle %s in this turn%s"
            % (
                alias,
                (" (stored: %s)" % ", ".join(known)) if known else
                "; re-run the command that produced the list",
            )
        )
    return declaration


def parent_handle(
    handle: str,
    *,
    scope: Optional[RuntimeHandleScope] = None,
    selected_store: Optional[ResultHandleStore] = None,
) -> Optional[str]:
    """The listing a page observation came from, or None for a listing itself."""
    declaration = handle_declaration(
        handle, scope=scope, selected_store=selected_store
    )
    return declaration["parent_alias"] or None


def _walk_records(
    scope: RuntimeHandleScope,
    store_: ResultHandleStore,
    declaration: Mapping[str, Any],
    query_scope: str,
) -> dict[str, Any]:
    """Rebuild the traversal for one query scope from its stored pages.

    Distinct uids in first-seen order: a page that repeats a uid the walk has
    already passed adds nothing to the sequence, which is what keeps a duplicate
    from displacing a row that has not been shown yet.
    """
    alias = declaration["alias"]
    key = _hot_key(scope, alias, query_scope)
    cached = _cached_walk(key)
    if cached is not None:
        return cached
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    next_offset = 0
    backend_total: Optional[int] = None
    for page in store_.list_pages(scope, alias=alias, query_scope=query_scope):
        record = page["record"]
        for entry in record.get("records") or []:
            item = _record_of(entry)
            if item["uid"] and item["uid"] in seen:
                continue
            seen.add(item["uid"])
            records.append(item)
        next_offset = max(next_offset, page["start_offset"] + page["limit_requested"])
        if page["backend_total"] is not None:
            backend_total = page["backend_total"]
    walk = {
        "alias": alias,
        "query_scope": query_scope,
        "records": records,
        "seen": seen,
        "next_offset": next_offset,
        "complete": bool(declaration["source_complete"]) and not query_scope,
        "backend_total": backend_total,
        "reconciled": None,
        "incomplete_reason": None,
        "bytes": 0,
    }
    _remember_walk(key, walk)
    return walk


def _filter_records(
    records: Sequence[Mapping[str, Any]], literal: Literal
) -> list[dict[str, Any]]:
    needle = literal.text.casefold()
    return [
        _record_of(record)
        for record in records
        if needle in str(record["line"]).casefold()
    ]


def fetch_page(
    handle: str,
    cursor: Optional[str] = None,
    contains: Optional[str] = None,
    *,
    scope: Optional[RuntimeHandleScope] = None,
    selected_store: Optional[ResultHandleStore] = None,
    budget_bytes: Optional[int] = None,
) -> ResultPage:
    """One page of a stored handle — the next page, or a filtered one.

    fastWorkflow registers no core fetch command: a core command joins every
    workflow's command surface, its ``what_can_i_do`` output and its trained
    intent model, which would make "a workflow that did not opt in is untouched"
    false for every workflow that never opts in. A workflow that opts in
    declares its own command and calls this from it.

    ``handle`` is the ``O`` alias printed on the producing observation. A page's
    own alias is accepted too and resolves to the listing it came from, so the
    parent is reachable from the page the agent is looking at.
    """
    selected_scope = scope or current_scope()
    store_ = selected_store or store()
    declaration = handle_declaration(
        handle, scope=selected_scope, selected_store=store_
    )
    if declaration["parent_alias"]:
        declaration = handle_declaration(
            declaration["parent_alias"], scope=selected_scope, selected_store=store_
        )
    alias = declaration["alias"]
    literal = normalize_literal(contains)
    query_scope = literal.scope
    descriptor_sha256 = declaration["descriptor_sha256"]
    position = 0
    if cursor:
        position = _check_cursor(
            decode_cursor(cursor),
            alias=alias,
            query_scope=query_scope,
            literal=literal,
            descriptor_sha256=descriptor_sha256,
        )
    base = _walk_records(selected_scope, store_, declaration, "")
    materialized = len(base["records"])
    total = int(declaration["total"])
    source_complete = bool(declaration["source_complete"])
    descriptor = declaration["descriptor"] or {}
    filter_columns = tuple(descriptor.get("filter_columns") or ())
    notes: list[str] = list(literal.notes)
    warnings: list[str] = []
    incomplete_reason: Optional[str] = None
    outcome = "rows"

    if query_scope:
        if not source_complete:
            # A filter over a partial local copy is a partial answer wearing a
            # whole-relation answer's clothes. Say it is unsupported instead.
            return _unsupported_page(
                declaration=declaration,
                literal=literal,
                materialized=materialized,
                total=total,
                budget_bytes=budget_bytes,
                scope=selected_scope,
                store_=store_,
                reason="producer_materialized_subset",
                message=(
                    "This handle holds %d of %d rows, so a filter over it could "
                    "not speak for the whole relation. Re-run the producing "
                    "command with a narrower query."
                    % (materialized, total)
                ),
            )
        records = _filter_records(base["records"], literal)
        matched = len(records)
        matched_complete = True
    else:
        records = base["records"]
        matched = materialized
        matched_complete = source_complete
        if not source_complete:
            incomplete_reason = "producer_materialized_subset"

    available = [str(record["line"]) for record in records[position:]]
    page = ResultPage(
        handle=alias,
        kind=declaration["kind"],
        summary=declaration["summary"],
        rows=available,
        matched=matched,
        total=total,
        materialized=materialized,
        source_complete=source_complete,
        matched_complete=matched_complete,
        continuation="complete",
        incomplete_reason=incomplete_reason,
        next_cursor=None,
        outcome=outcome,
        position=position,
        page_index=_page_index(selected_scope, alias, position),
        parent_alias=alias,
        page_alias=current_execute_alias(),
        literal=literal.text or None,
        filter_columns=filter_columns if literal.text else (),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )
    budget = budget_bytes or page_max_bytes_from_env()
    # Pack against a budget reduced by the longest cursor this page could carry.
    # The header is built before the packer knows how many rows fit, and the
    # cursor is built after, so without the reserve a full page would overrun
    # the budget by exactly the length of its own continuation.
    reserve = len(" next_cursor=") + len(
        encode_cursor(
            alias=alias,
            query_scope=query_scope,
            position=max(len(records), 1),
            descriptor_sha256=descriptor_sha256,
        )
    )
    base_warnings = tuple(warnings)
    observation, shown, warned = _render(
        page, budget_bytes=max(budget - reserve, 1), warnings=base_warnings,
        reported_budget=budget,
    )
    page.rows = shown
    page.warnings = warned
    remaining = len(records) - position - len(shown)
    if remaining > 0:
        page.continuation = "cursor"
        page.next_cursor = encode_cursor(
            alias=alias,
            query_scope=query_scope,
            position=position + len(shown),
            descriptor_sha256=descriptor_sha256,
        )
    elif not page.matched_complete or not page.source_complete:
        page.continuation = "source-incomplete"
        page.incomplete_reason = page.incomplete_reason or "producer_materialized_subset"
    else:
        page.continuation = "complete"
    if not shown and matched == 0:
        page.outcome = "complete-zero" if page.matched_complete else "partial"
        notes.append(_zero_message(literal, filter_columns, declaration))
        page.notes = tuple(notes)
    elif not page.matched_complete or not page.source_complete:
        page.outcome = "partial"
    page.observation, page.rows, page.warnings = _render(
        page, budget_bytes=budget, warnings=base_warnings, reported_budget=budget
    )
    _record_page_event(selected_scope, page, literal)
    _link_page(selected_scope, store_, page, declaration, query_scope)
    return page


def _zero_message(
    literal: Literal, filter_columns: Sequence[str], declaration: Mapping[str, Any]
) -> str:
    """A complete zero is a fact about this query, never about the world."""
    if literal.text:
        return (
            'No rows matched the literal "%s" in these fields: %s. That is a '
            "complete zero for this literal in this listing; it is not evidence "
            "that the person or object does not exist."
            % (literal.text, ", ".join(filter_columns) or "the rendered rows")
        )
    return (
        "This listing is empty: the command that produced it returned no rows."
    )


def _unsupported_page(
    *,
    declaration: Mapping[str, Any],
    literal: Literal,
    materialized: int,
    total: int,
    budget_bytes: Optional[int],
    scope: RuntimeHandleScope,
    store_: ResultHandleStore,
    reason: str,
    message: str,
) -> ResultPage:
    page = ResultPage(
        handle=declaration["alias"],
        kind=declaration["kind"],
        summary=declaration["summary"],
        rows=[],
        matched=0,
        total=total,
        materialized=materialized,
        source_complete=bool(declaration["source_complete"]),
        matched_complete=False,
        continuation="source-incomplete",
        incomplete_reason=reason,
        next_cursor=None,
        outcome="unsupported",
        position=0,
        page_index=_page_index(scope, declaration["alias"], 0),
        parent_alias=declaration["alias"],
        page_alias=current_execute_alias(),
        literal=literal.text or None,
        notes=(("Filtering is unsupported for this handle. " + message),),
    )
    budget = budget_bytes or page_max_bytes_from_env()
    page.observation, page.rows, page.warnings = _render(
        page, budget_bytes=budget, reported_budget=budget
    )
    _record_page_event(scope, page, literal)
    return page


def _page_index(scope: RuntimeHandleScope, alias: str, position: int) -> int:
    """How many distinct pages of this handle have been served in this turn.

    Counted by start position rather than by call, so a retried cursor is the
    same page it was the first time and cannot inflate the count.
    """
    key = "%s:%s" % (scope.scope_id, alias)
    with _lock:
        served = _pages_served.setdefault(key, set())
        served.add(int(position))
        return len(served)


def _record_page_event(
    scope: RuntimeHandleScope, page: ResultPage, literal: Literal
) -> None:
    record_event(
        {
            "kind": "result_handle_page",
            "scope_id": scope.scope_id,
            "alias": page.handle,
            "page_alias": page.page_alias,
            "page_index": page.page_index,
            "position": page.position,
            "rows_shown": len(page.rows),
            "matched": page.matched,
            "total": page.total,
            "materialized": page.materialized,
            "outcome": page.outcome,
            "continuation": page.continuation,
            "incomplete_reason": page.incomplete_reason,
            "literal": literal.text or None,
            "observation_bytes": len(page.observation.encode("utf-8")),
        }
    )


def _link_page(
    scope: RuntimeHandleScope,
    store_: ResultHandleStore,
    page: ResultPage,
    declaration: Mapping[str, Any],
    query_scope: str,
) -> None:
    """File the page observation's own alias against the listing it paged.

    The fetch command is an execute step, so its observation gets its own ``O``
    alias, is archived under it and is searchable like any other. This records
    the internal link, so a search that lands on the page can find the listing
    it came from without the page pretending to be that listing.
    """
    if not page.page_alias or page.page_alias == declaration["alias"]:
        return
    try:
        store_.put_declaration(
            scope,
            page.page_alias,
            {
                "kind": "%s-page" % declaration["kind"],
                "summary": page.summary,
                "ordering": declaration["ordering"],
                "total": page.total,
                "materialized": len(page.rows),
                "source_complete": page.source_complete,
                "page_size": declaration["page_size"],
                "classification": declaration["classification"],
                "presentation": declaration["presentation"],
                "filters": declaration["filters"],
                "descriptor": declaration["descriptor"],
                "descriptor_sha256": declaration["descriptor_sha256"],
                "parent_alias": declaration["alias"],
                "query_scope": query_scope,
                "cursor_position": page.position,
            },
        )
    except Exception as error:  # noqa: BLE001
        record_event(
            {
                "kind": "result_handle_link_refused",
                "scope_id": scope.scope_id,
                "alias": page.page_alias,
                "parent_alias": declaration["alias"],
                "error": type(error).__name__,
            }
        )


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "HOT_ROWS_MAX_BYTES",
    "HOT_ROWS_MAX_BYTES_ENV",
    "Literal",
    "RESULT_PAGE_MAX_BYTES",
    "RESULT_PAGE_MAX_BYTES_ENV",
    "ResultHandleError",
    "ResultHandleSpec",
    "ResultHandleStore",
    "ResultPage",
    "SourceDescriptor",
    "UNSORTED_OFFSET",
    "WILDCARD_CHARACTERS",
    "current_execute_alias",
    "current_scope",
    "declare",
    "decode_cursor",
    "encode_cursor",
    "fetch_page",
    "handle_declaration",
    "normalize_literal",
    "page_max_bytes_from_env",
    "parent_handle",
    "register_resolver",
    "registered_resolvers",
    "reset_result_handle_state",
    "resolver_for",
    "store",
    "unregister_resolver",
]
