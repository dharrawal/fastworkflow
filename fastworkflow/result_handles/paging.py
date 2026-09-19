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

import logging
import os
import re
import sqlite3
import threading
import unicodedata
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from fastworkflow import context_budget
from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.observation_offloading.state import (
    default_scope,
    record_event,
    scope_for_host,
)
from fastworkflow.result_handles.common import (
    CURSOR_TOKEN_EXAMPLE,
    DEFAULT_PAGE_SIZE,
    FIRST_CURSOR_PAGE,
    MAX_ALIAS_DIGITS,
    MAX_CURSOR_PAGE,
    UNSORTED_OFFSET,
    ResultHandleError,
    canonical_json as _canonical_json,
    digest as _digest,
)

logger = logging.getLogger(__name__)

#: A page observation is a listing observation, so it gets the listing budget:
#: 3 KB of the ReAct prompt, header line included. Rows that do not fit are not
#: dropped — they stay in the store and the next cursor returns them.
RESULT_PAGE_MAX_BYTES = context_budget.REFERENCE_RESULT_PAGE_MAX_BYTES
RESULT_PAGE_MAX_BYTES_ENV = context_budget.RESULT_PAGE.override_env
#: Below this the header alone would consume the budget it is describing.
RESULT_PAGE_MIN_BYTES = context_budget.RESULT_PAGE.floor

#: Rows held in this process. The durable copy is SQLite, so eviction costs a
#: re-read and never loses a stored page.
HOT_ROWS_MAX_BYTES = context_budget.REFERENCE_RESULT_HANDLE_HOT_MAX_BYTES
HOT_ROWS_MAX_BYTES_ENV = context_budget.RESULT_HANDLE_HOT.override_env


#: The only ordering policy a descriptor may name. B0 (ido-gqv.6) measured an
#: explicit ``sort`` combined with offset paging silently dropping 20 of 540
#: group members while returning exactly ``total`` rows, so a stored descriptor
#: cannot express a sorted walk at all: there is no field to put one in.

#: Marker key of the raw-page REFERENCE a declaration returns for its artifacts
#: (bead ido-986.14.3, D). It is deliberately NOT the observability store's
#: ``__fw_artifact_ref__`` envelope, which points at a row of the turn's own
#: ``artifacts`` table: that envelope's lifetime is the turn record and its
#: payload is an opaque blob, while a listing page is owned by THIS store, is
#: keyed by ``(scope_id, alias, query_scope, start_offset)``, and outlives the
#: turn exactly as long as the archive file does. Pointing one at the other
#: would let a mutable listing collection masquerade as an immutable blob, which
#: is the thing ido-986.14.3 forbids in as many words.
RESULT_PAGES_REF_KEY = "__fw_result_pages__"


def result_pages_reference(
    *,
    scope_id: str,
    alias: str,
    descriptor_sha256: str,
    pages: "Sequence[Mapping[str, Any]]" = (),
) -> dict[str, Any]:
    """The durable pointer a listing artifact carries instead of copies of rows.

    Everything in it is a KEY or a DIGEST: nothing that has to be kept in step
    with the rows. Reading it back is `ResultHandleStore.get_page` plus a
    sha256, which is what makes a missing or edited page a refusal rather than a
    silently shorter listing.
    """
    return {
        RESULT_PAGES_REF_KEY: True,
        "scope_id": str(scope_id),
        "alias": str(alias),
        "descriptor_sha256": str(descriptor_sha256),
        "pages": [dict(page) for page in pages],
    }

#: ``CURSOR_TOKEN_EXAMPLE`` is defined in ``result_handles.common`` and imported
#: at the top of this module; it is named in ``__all__`` because it is part of
#: the released surface. A page token is short enough to read off a page and
#: type back without transcription error: the handle, an optional traversal tag,
#: then the page ordinal. ``O7/p2`` is page 2 of handle O7; ``O7/f1p2`` is page
#: 2 of the first filtered traversal of O7. C1 (exp-ido-gqv-8) measured a
#: 150-byte opaque base64 cursor re-typed by hand in 4 of 15 fetch calls, and
#: the corruption decoded to a DIFFERENT valid handle. Here the handle is
#: literal in the token and the ordinal resolves only through the store, so a
#: mistyped token is refused by name instead of quietly serving another
#: listing's rows.
#:
#: ``MAX_CURSOR_PAGE`` is ``common``'s too, and is the bound ``cursors``
#: enforces. (ido-1de, F22) It is the largest page ordinal a token may name. An
#: unbounded ordinal was not merely useless, it was a crash a model could type:
#: 20 digits reached SQLite as an out-of-range INTEGER (OverflowError) and 4300
#: digits hit CPython's int() digit limit (ValueError), and neither is a
#: ResultHandleError the command can turn into a refusal. No traversal reaches a
#: millionth page, so anything past this is garbage and is named as such.

#: The token grammar itself belongs to ``result_handles.common`` and is applied
#: by ``result_handles.cursors``: the digit caps, the token pattern, the
#: overlong-ordinal pattern and the trim set all live there. Paging still
#: enforces the alias shape, so it imports ``MAX_ALIAS_DIGITS`` from ``common``
#: instead of restating it. Why those caps exist, recorded here because this
#: module is where the failures were measured:
#:
#: ``MAX_CURSOR_PAGE_DIGITS`` -- digits the token pattern itself accepts.
#: Comfortably wider than the bound above - the bound is what refuses a large
#: ordinal, with a message about pages - but narrow enough that int() on the
#: match is always cheap and always fits a SQLite INTEGER.
#:
#: ``MAX_ALIAS_DIGITS`` -- (ido-bdo) Digits of the HANDLE ordinal a token may
#: name, and of the alias a declaration may be filed under: the two are one
#: pattern, so a token can name every alias ``declare`` can create and nothing
#: else. F22 capped the page ordinal and left this group open. It never crashed
#: -- an alias is only ever a string lookup, with no ``int()`` to overflow --
#: but a 5,000-digit alias still reached the store and the "no page token has
#: been issued for ..." message it builds, so it is capped for symmetry. An
#: ``O`` is an execute ordinal in one turn; nine digits is past any turn that
#: has ever run.
#:
#: ``MAX_TAG_DIGITS`` -- (ido-h0c, F28) Digits of a traversal tag.
#: ``cursor_tag`` hands out ``f1``, ``f2``, ... per distinct filter on one
#: handle in one turn and never stopped at three digits, so a handle filtered a
#: thousand times printed ``f1000`` in its own next_cursor and then refused to
#: parse it. Six digits is past any turn, and the tag is a string lookup, so no
#: numeric bound is owed.
#:
#: ``CURSOR_TOKEN_OVERLONG_RE`` -- the token shape with an ordinal too long for
#: the pattern above, so a token whose only fault is an absurd page number is
#: refused for THAT, rather than falling through to "this is not a page token".
#: ``CURSOR_TOKEN_TRIM`` -- quoting and punctuation a model wraps a copied
#: value in.

#: Backend pages one fetch call may read before it warns and hands the rest to
#: the next cursor. A bound on one call, never a cap on enumeration.
#:
#: (ido-2y3, F6) It bounds the whole of ONE ``fetch_page``, across every fill
#: round that call makes, because the budget used to restart at zero on each
#: round: a round that spent all eight calls and still had not filled the
#: observation simply got eight more, so a fetch could read many times the
#: advertised number of backend pages. ``_PageCallBudget`` is what makes the
#: number mean one fetch.
#:
#: What it counts is SOURCE PAGES. The independent ``countOnly`` of
#: ``_reconcile`` is deliberately outside it: it is at most one call per fetch
#: (the walk is marked reconciled and never re-proves itself), it does not grow
#: with the number of pages read, and it is the coverage proof itself - charging
#: it against the page budget would let a fetch that read exactly eight pages
#: silently lose the one call that decides whether the enumeration was complete.
MAX_RESOLVER_CALLS_PER_FETCH = 8
#: How many times one call may widen its read to fill the byte budget.
MAX_FILL_ROUNDS = 4
#: Pages of one handle in a turn after which the observation suggests a literal
#: filter. It suggests; it never refuses and never narrows anything itself.
PAGE_WARNING_AFTER = 3

#: ``%`` and ``_`` are LIKE wildcards on the portal and ``*`` behaves as one
#: too; backslash escaping does not work (B0 §b). A literal filter therefore
#: cannot be delivered by passing the agent's text through, so the characters
#: are removed and the observation says which literal was really sent.
WILDCARD_CHARACTERS = "%_*"

#: The model emits U+00A0 and U+2011 inside these very names, and the portal
#: answers a Unicode-contaminated filter with a confident, silent zero.
_SPACE_LIKE = {0x00A0: " ", 0x2007: " ", 0x202F: " ", 0x2009: " ", 0x2011: "-"}
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")

#: (ido-h0c, F28) What an alias may be, ENFORCED by ``declare``. It was dead
#: code: ``declare(alias="handle-x")`` was accepted, page 1 served and printed
#: ``handle-x/p2`` as its own next_cursor, and that token could never parse,
#: so the listing was unpageable and said so only on the second call. It is the
#: same shape ``common.CURSOR_TOKEN_RE`` accepts -- an ``O`` execute ordinal or
#: the ``D`` key used where there is no agent step -- so an alias that declares
#: is an alias a token can name.
_ALIAS_RE = re.compile(r"^[OD][1-9]\d{0,%d}$" % (MAX_ALIAS_DIGITS - 1))

#: How the producer renders a row: `uid` then two spaces then the label. Kept
#: identical to the listing text the command returned, because a filter has to
#: be able to find "Alan Cooper" in the row the agent was shown.
ROW_SEPARATOR = "  "

#: (ido-56z, F26) Characters that would end the line a row is rendered on. A
#: row is user text by contract, and a page observation is a header line
#: followed by one line per row: a label carrying a newline is therefore not a
#: long row, it is a second line the reader has no way to tell from a header
#: this module wrote. The observed case printed
#: ``result_handle=O1 page 1 ... outcome=rows`` out of a backend label. They are
#: escaped rather than stripped, because the bytes a source returned are
#: evidence and deleting them would invent a shorter label.
_LINE_BREAK_ESCAPES = {
    "\n": "\\n",
    "\r": "\\r",
    "\v": "\\v",
    "\f": "\\f",
    "\x85": "\\x85",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}
_LINE_BREAK_RE = re.compile("[%s]" % "".join(_LINE_BREAK_ESCAPES))


def one_line(text: Any) -> str:
    """*text* as a single line: every line break spelled, none removed."""
    return _LINE_BREAK_RE.sub(
        lambda match: _LINE_BREAK_ESCAPES[match.group(0)], str(text)
    )


# ---------------------------------------------------------------------------
# Serialisable source description
# ---------------------------------------------------------------------------


from fastworkflow.result_handles.models import ResultHandleSpec, SourceDescriptor

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


from fastworkflow.result_handles.store import ResultHandleStore


# ---------------------------------------------------------------------------
# Process-local state: the store handle, the hot rows, the per-turn counters
# ---------------------------------------------------------------------------

_lock = threading.Lock()
#: One store per archive file, keyed by the normalised path. (ido-pg2) Keying
#: this by first use instead made the whole process write into whichever
#: workflow happened to run first: a second workflow's declarations, pages and
#: cursors landed in the first workflow's file while its observations were
#: archived in its own, and a later process that opened the second file could
#: not find them. The path is resolved on every call, so the file a handle is
#: written to is always the file the current agent archives its observations in.
_stores: "dict[str, ResultHandleStore]" = {}
_hot: "dict[str, dict[str, Any]]" = {}
_pages_served: "dict[str, set[int]]" = {}
_local_sequence: "dict[str, int]" = {}
#: Tokens this process has issued or resolved, "<scope_id>|<token>" -> payload.
#: A write-through mirror of ``result_handle_cursors``: it makes the hot path a
#: dict read and keeps a token usable for the rest of the turn even if the store
#: write failed. It is never the only copy that matters - the SQLite row is what
#: survives a restart, and the tests read tokens back through a fresh store.
_cursor_tokens: "dict[str, dict[str, Any]]" = {}
#: Which turn scopes a cached store is serving, ``db_path -> {scope_id}``.
#: (ido-1ew) ``_stores`` grows by one entry per archive file the process ever
#: touches, so something has to say when an entry may go. The rule is
#: OWNERSHIP BY USE: a scope owns the store whose file its cached rows name,
#: recorded where a scope and a store meet to form a cache key, and the LAST
#: owner released drops the store. A file no scope ever claimed -- the
#: per-process fallback a command frame uses before any agent exists -- has no
#: owner to release it and is kept. Dropping an entry closes nothing: a store
#: is a path, and ``_connect`` opens and closes a connection per statement, so
#: a reopened store is the same store.
_store_scopes: "dict[str, set[str]]" = {}


def hot_rows_max_bytes_from_env() -> int:
    """The hot-row cache cap for this run. See ``fastworkflow.context_budget``."""
    return context_budget.result_handle_hot_max_bytes()


def page_max_bytes_from_env() -> int:
    """The page-observation budget for this run. See ``fastworkflow.context_budget``.

    (ido-h0c, F28) Bounded at both ends here. ``context_budget`` refuses an
    override below the budget's floor and accepts anything at all above it, so
    ``FW_RESULT_PAGE_MAX_BYTES=99999999999999999999999`` was taken at its word
    and a page would be packed until the rows ran out. One page observation
    cannot usefully be larger than the whole prompt it has to fit in, so the
    ceiling is the resolved context window in bytes: an override past it is a
    typo, is warned about, and the ceiling stands. This bounds the budget this
    module reads; an explicit ``budget_bytes=`` argument is a caller measuring
    its own observation and is left alone.
    """
    budget = context_budget.result_page_max_bytes()
    ceiling = (
        int(context_budget.context_window_tokens()[0])
        * context_budget.BYTES_PER_TOKEN
    )
    if budget > ceiling:
        logger.warning(
            "%s=%d is larger than the whole %d-byte context window; using %d",
            RESULT_PAGE_MAX_BYTES_ENV, budget, ceiling, ceiling,
        )
        return ceiling
    return budget


def store() -> ResultHandleStore:
    """The database the running turn's handles live in.

    The same file the observation archive uses, so a page and the observation
    that showed it survive together: one file to keep, one file to read back
    when an experiment is scored.

    The active archive path is resolved first and the cache is keyed by it, so
    two agents alive in one process each write into their own workflow's file,
    and a call made before any agent exists is pinned to the per-process
    fallback file only for as long as that is the file it names.
    """
    agent = _current_agent()
    runtime = getattr(agent, "turn_runtime", None)
    bound = runtime.get_result_store() if runtime is not None else None
    if bound is not None:
        return bound
    return store_for_path(_default_store_path())


def store_for_path(path: str) -> ResultHandleStore:
    """Return the component-owned lazy store for an archive path."""
    key = _store_key(path)
    with _lock:
        existing = _stores.get(key)
    if existing is not None:
        return existing
    created = ResultHandleStore(key)
    with _lock:
        return _stores.setdefault(key, created)


def _store_key(path: str) -> str:
    """The normalised path two spellings of one archive file agree on."""
    return os.path.abspath(os.path.expanduser(str(path)))


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
    with _lock:
        _stores.clear()
        _store_scopes.clear()
        _hot.clear()
        _pages_served.clear()
        _local_sequence.clear()
    _cursor_component.reset_cursor_state()


def release_scope(scope_id: str) -> None:
    """Drop one finished turn scope's process-local rows, counters and tokens.

    ``ido-1ew``. Stored rows are untouched, by the same design that makes hot
    eviction free of consequence: everything dropped here is rebuildable from
    the store file, which is what a resume in another process already does.

    ``TurnRuntime`` and the released standalone coordinator call this only
    after their existing guards have decided the turn is finished.
    """
    hot_prefix = "%s:" % scope_id
    namespace_prefix = "%s@" % scope_id
    with _lock:
        for key in [key for key in _hot if key.startswith(hot_prefix)]:
            _hot.pop(key, None)
        for registry in (_pages_served,):
            for key in [
                key for key in registry if key.startswith(namespace_prefix)
            ]:
                registry.pop(key, None)
        _local_sequence.pop(scope_id, None)
        for path in list(_store_scopes):
            owners = _store_scopes[path]
            owners.discard(scope_id)
            if owners:
                continue
            del _store_scopes[path]
            _stores.pop(path, None)
    _cursor_component.release_scope(scope_id)


def _note_store_owner(store_: "ResultHandleStore", scope: RuntimeHandleScope) -> None:
    """Record that *scope* is using *store_*, so its release can free the store."""
    with _lock:
        _store_scopes.setdefault(store_.db_path, set()).add(scope.scope_id)


def _hot_key(
    store_: "ResultHandleStore",
    scope: RuntimeHandleScope,
    alias: str,
    query_scope: str,
) -> str:
    """One walk per traversal per store file.

    (ido-pg2) The store file is part of the key because one process can hold two
    workflows whose turn scopes agree, and a walk rebuilt from one file must
    never be served for the other.
    """
    _note_store_owner(store_, scope)
    return "%s:%s:%s@%s" % (scope.scope_id, alias, query_scope, store_.db_path)


def _cache_namespace(scope: RuntimeHandleScope, store_: "ResultHandleStore") -> str:
    """The process-local cache namespace of one turn scope in one store file."""
    _note_store_owner(store_, scope)
    return "%s@%s" % (scope.scope_id, store_.db_path)


#: What one cached walk record costs beyond the text it carries. (ido-5b5, F23)
#: Measured on CPython 3.13: the two-key record dict (184), the list slot that
#: holds it (8), its entry in the walk's ``seen`` set (66 at a 2,000-uid load
#: factor) and the object header of each of the two strings (41 apiece) come to
#: 340 bytes; the strings' own bytes are added on top of it. It is an accounting
#: figure, not an allocation, and it exists so the bound measures what the cache
#: RETAINS. Counting rendered line bytes alone undercounted a 30-column
#: relation by 229x, which is how 32,890 accounted bytes passed a 262,144 byte
#: cap while the walk held 7.5 MB.
_HOT_RECORD_OVERHEAD_BYTES = 340


def _walk_retained_bytes(walk: Mapping[str, Any]) -> int:
    """What one cached walk holds, counted to the nearest object header.

    Every field a walk keeps per row is counted: the record dict, its slot in
    the records list, its uid in the ``seen`` set, and the two strings. The rest
    of a walk - offsets, flags, the stop reason - is a fixed handful of bytes
    per walk and the eviction loop is not sensitive to it.
    """
    total = 0
    for record in walk["records"]:
        total += (
            _HOT_RECORD_OVERHEAD_BYTES
            + len(str(record["uid"]).encode("utf-8"))
            + len(str(record["line"]).encode("utf-8"))
        )
    return total


def _hot_bytes() -> int:
    return sum(int(entry.get("bytes") or 0) for entry in _hot.values())


def _remember_walk(key: str, walk: dict[str, Any]) -> list[str]:
    """Cache a walk and evict least-recently-used first until under the bound.

    **On return ``_hot_bytes() <= hot_rows_max_bytes_from_env()``, always.** The
    bound has no exemption, not for a big walk and not for the walk this call is
    building. Eviction is free of consequence: every row in a walk came from a
    stored page and is rebuilt from SQLite on the next read, so the bound
    controls memory and never reachability, and the caller that just handed its
    walk over still holds it for the rest of the call.

    (ido-7ce, F8) The loop used to stop while one walk was left and skip the
    walk being built, so a single enumerated relation could exceed the cap by
    any amount and stay cached after the fetch returned - a zero cap still
    retained it. The walk being built is now the LAST one evicted rather than
    the one never evicted, which keeps it out of the way of walks the turn has
    finished with while still making it answer to the bound.

    (ido-1r0) Re-assigning an existing key leaves its position in the dict where
    first use put it, which made insertion order first-use order and evicted the
    walk being paged right now before walks nothing had touched in a while. The
    key is dropped before it is written so that every remembered walk moves to
    the end, and ``next(iter(_hot))`` is genuinely the coldest one.
    """
    evicted: list[str] = []
    oversized = False
    with _lock:
        walk["bytes"] = _walk_retained_bytes(walk)
        _hot.pop(key, None)
        _hot[key] = walk
        limit = hot_rows_max_bytes_from_env()
        while _hot_bytes() > limit and _hot:
            oldest = next(iter(_hot))
            if oldest == key and len(_hot) > 1:
                # Coldest first, but the walk being built goes last: the rest of
                # this call still reads it out of the local it was handed from.
                oldest = next(candidate for candidate in _hot if candidate != key)
            _hot.pop(oldest, None)
            evicted.append(oldest)
            oversized = oversized or oldest == key
    if evicted:
        record_event({"kind": "result_handle_hot_evict", "walks": evicted})
    if oversized:
        # A walk that does not fit the cache on its own. Worth saying out loud:
        # it is rebuilt from stored pages on every fetch from here on, which
        # costs SQLite reads and no resolver call, and the cap is the reason.
        record_event({
            "kind": "result_handle_hot_oversized",
            "walk": key,
            "bytes": int(walk["bytes"]),
            "limit": int(hot_rows_max_bytes_from_env()),
            "records": len(walk["records"]),
        })
    return evicted


def _cached_walk(key: str) -> Optional[dict[str, Any]]:
    """The cached walk, moved to the end of the eviction order by the read.

    (ido-1r0) Reading a walk is using it: without this a walk that is paged on
    every call but never rebuilt keeps the position its first use gave it and is
    evicted ahead of walks nothing has touched since.
    """
    with _lock:
        walk = _hot.get(key)
        if walk is not None:
            _hot.pop(key, None)
            _hot[key] = walk
        return walk


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

    The live agent's turn runtime first, when it has one: ``TurnRuntime`` is the
    component that binds a turn to its scope, so its answer is the scope this
    turn's observations are archived under, and a handle filed anywhere else
    would be a handle the same turn could not read back. The agent's own
    ``continuation_scope`` is the same answer for an agent built without a
    runtime, and is consulted next. Then the trace host (a command running
    outside the ReAct loop), then the process default.
    """
    from fastworkflow import tracing

    agent = _current_agent()
    runtime = getattr(agent, "turn_runtime", None)
    runtime_scope = getattr(runtime, "scope", None)
    if isinstance(runtime_scope, RuntimeHandleScope):
        return runtime_scope
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
    step is the last one with no observation. Which step is in flight is read
    from ``current_trajectory``; what that step is CALLED is read from the
    agent's ``execute_ordinal_by_step`` ledger, which numbered it just before
    dispatch and is the same ledger ``annotate_execute_observations`` is given
    when the step completes.

    ido-7qd: counting ``current_trajectory`` was that answer, and it is only
    right when the mirror holds the whole turn. A process that imported a
    suspension has an empty mirror and a resumed trajectory with N executes
    already in it, so the first command after the resume declared and stamped
    O1 while its observation was printed O(N+1) -- a collision against the real
    O1, or a printed handle nothing had declared. The ledger is restored with
    the suspension, so both sides now read one number.

    The count stands in only for an agent with no ledger (a duck-typed host, a
    plain ReAct), where the mirror is the whole turn by construction.

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
    ledger = getattr(agent, "execute_ordinal_by_step", None)
    if isinstance(ledger, Mapping) and latest in ledger:
        ordinal = int(ledger[latest])
    else:
        ordinal = sum(
            1
            for index in indexes
            if str(trajectory.get(f"tool_name_{index}") or "")
            == "execute_workflow_query"
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


from fastworkflow.result_handles.cursors import (
    _cursor_tag as _cursor_tag_impl,
    _parse_cursor_token,
    cursor_placeholder,
    cursor_token,
    decode_cursor as _decode_cursor_impl,
    encode_cursor as _encode_cursor_impl,
)
from fastworkflow.result_handles import cursors as _cursor_component

# One registry object, owned by cursors and released by this component's
# lifecycle hook. There is no process-global configure step or provider swap.
_cursor_tokens = _cursor_component._cursor_tokens


def _cursor_dependencies(
    scope: Optional[RuntimeHandleScope],
    selected_store: Optional[ResultHandleStore],
) -> tuple[RuntimeHandleScope, ResultHandleStore]:
    """The ambient scope and store the cursor component is not allowed to find.

    Resolving is not free and it is not read-only: ``store()`` OPENS the turn's
    archive file, which creates it if it is not there, and ``_cache_namespace``
    records the scope as an owner of it. So this is called only where the caller
    is really going to reach the store, never merely to satisfy a signature.
    """
    selected_scope = scope or current_scope()
    store_ = selected_store or store()
    _cache_namespace(selected_scope, store_)
    return selected_scope, store_


def encode_cursor(*, alias: str, query_scope: str, position: int,
                  descriptor_sha256: str,
                  scope: Optional[RuntimeHandleScope] = None,
                  selected_store: Optional[ResultHandleStore] = None) -> str:
    selected_scope, store_ = _cursor_dependencies(scope, selected_store)
    return _encode_cursor_impl(
        alias=alias, query_scope=query_scope, position=position,
        descriptor_sha256=descriptor_sha256, scope=selected_scope,
        selected_store=store_,
    )


def decode_cursor(cursor: str, *, alias: Optional[str] = None,
                  scope: Optional[RuntimeHandleScope] = None,
                  selected_store: Optional[ResultHandleStore] = None) -> dict[str, Any]:
    # The GRAMMAR runs before the dependencies do, which is the order the
    # single-module implementation had and is the order that matters: a string
    # that is not a page token is refused by ``_parse_cursor_token`` without the
    # turn's archive file being opened, and therefore created, and without the
    # scope being recorded as an owner of it. Resolving first made every
    # mistyped cursor leave a ~100 KB SQLite file behind on the way to a
    # refusal that never reads a row. ``_decode_cursor_impl`` parses again as
    # its own first step; the token is at most a few dozen characters and the
    # parse is one regex, so the second one costs nothing worth keeping.
    _parse_cursor_token(cursor)
    selected_scope, store_ = _cursor_dependencies(scope, selected_store)
    return _decode_cursor_impl(
        cursor, alias=alias, scope=selected_scope, selected_store=store_
    )


def _cursor_tag(scope: RuntimeHandleScope, store_: ResultHandleStore,
                alias: str, query_scope: str) -> str:
    _cache_namespace(scope, store_)
    return _cursor_tag_impl(scope, store_, alias, query_scope)


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
        # (ido-56z, F26) One item is one row, so a producer item that carries a
        # newline is flattened here rather than becoming two lines of a page.
        text = one_line(line)
        records.append({"uid": _uid_of_line(text), "line": text, "row": None})
    return records


def _stored_row_count(record: Mapping[str, Any]) -> int:
    """How many rows one stored page carries. (ido-h0c, F28)

    ``records`` is the rendered sequence a page is served from and is what every
    reader of a page means by its rows; ``rows`` is the backend's raw reply and
    a producer has none. They are the same length whenever both are present --
    ``records`` is one ``_render_row`` per row -- so this is the count of the
    page for a producer page and for a resolver page alike.
    """
    return len(record.get("records") or record.get("rows") or [])


def _dedupe_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Distinct uids in first-seen order. (ido-h0c, F28)

    The same rule ``_walk_records`` and ``_extend_walk`` apply when they read a
    stored page back, stated once and applied where a listing is first stored,
    so a declaration's ``materialized`` and its pages' ``matched`` are counts of
    the same rows. A row with no uid is never a duplicate of anything: there is
    nothing to compare it on.
    """
    seen: set[str] = set()
    distinct: list[dict[str, Any]] = []
    for record in records:
        uid = str(record["uid"])
        if uid and uid in seen:
            continue
        seen.add(uid)
        distinct.append(dict(record))
    return distinct


def _record_of(record: Mapping[str, Any]) -> dict[str, Any]:
    """One stored record as the WALK keeps it: the identity and the text.

    (ido-5b5, F23) The backend row this line was rendered from is deliberately
    not carried here. Nothing reads it back: rendering happened when the page
    was stored, a local filter matches the rendered line, the uid is the
    identity, and every caller of a walk reads ``uid`` or ``line`` and nothing
    else. Keeping it made a walk hold every column of every row it had ever
    paged for the length of a turn - 7.5 MB for a 2,000-row walk of 30-column
    rows - while the hot bound counted only the rendered text.

    The stored page is untouched: ``record_json`` still carries the row, byte
    for byte, in the shape it has always had, so this changes what the process
    holds and never what a later process reads.
    """
    return {"uid": str(record["uid"]), "line": str(record["line"])}


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


from fastworkflow.result_handles.rendering import ResultPage, _assemble, _pack


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
    handle = alias or current_execute_alias() or _local_alias(selected_scope)
    if not _ALIAS_RE.match(str(handle)):
        # (ido-h0c, F28) A programming error in the opting-in workflow, raised
        # like a malformed spec: an alias a page token cannot name is a listing
        # whose second page can never be asked for, and finding that out on the
        # second fetch is worse than finding it out here.
        raise ResultHandleError(
            "%r is not a usable result handle alias. A handle is the O alias of "
            "the execute step that declared it (O1, O2, ...), or a D key where "
            "there is no agent step; a page token names the handle literally, "
            "so an alias of any other shape has no second page." % (str(handle)[:40],)
        )
    # (ido-h0c, F28) One uid is one row. The walk keeps distinct uids in
    # first-seen order, so a producer that filed the same uid twice had its
    # duplicate dropped on the way back out and the declaration went on
    # advertising a count no page of it could ever show -- declare said
    # materialized=6 where every page said 3. Deduplicated HERE instead, once,
    # so the number the declaration reports is the number the rows amount to.
    # ``total`` is untouched: that is the producer's claim about the relation,
    # not about what it filed.
    declared_items = [str(item) for item in (spec.items or ())]
    records = _dedupe_records(_records_from_items(declared_items))
    items = [str(record["line"]) for record in records]
    if len(items) != len(declared_items):
        record_event(
            {
                "kind": "result_handle_duplicate_items_dropped",
                "scope_id": selected_scope.scope_id,
                "alias": handle,
                "declared": len(declared_items),
                "distinct": len(items),
            }
        )
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
    stored_pages: list[dict[str, Any]] = []
    try:
        # Only when there are rows. A zero-row producer page stored at the
        # walk's first offset would be read back as the empty page that ends a
        # walk, and the walk would stop before it started.
        start_offset = int(descriptor.start_offset) if descriptor else 0
        producer_record = {"rows": [], "records": records} if records else None
        # (ido-ecd, F19) Digested here, before anything is written, because it
        # is half of the identity `put_declaration` refuses a redeclaration on.
        # It is the digest of the bytes `put_page` would store, so it compares
        # directly against `record_sha256` of the page a first declaration left
        # behind.
        producer_sha256 = (
            None if producer_record is None
            else _digest(_canonical_json(dict(producer_record)))
        )
        store_ = selected_store or store()
        store_.put_declaration(
            selected_scope,
            handle,
            payload,
            first_page_offset=start_offset,
            first_page_sha256=producer_sha256,
        )
        if producer_record is not None:
            stored = store_.put_page(
                selected_scope,
                alias=handle,
                query_scope="",
                start_offset=start_offset,
                limit_requested=len(items),
                source="producer",
                record=producer_record,
                backend_total=int(spec.total or len(items)),
            )
            # `put_page` reads the row back and re-digests it, so this sha256 is
            # the STORED bytes and not the bytes this process meant to store.
            stored_pages.append(
                {
                    "query_scope": "",
                    "start_offset": start_offset,
                    "limit_requested": len(items),
                    "records": len(items),
                    "source": "producer",
                    "sha256": str(stored["record_sha256"]),
                }
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
        return {
            **payload,
            "declared": False,
            "error": type(error).__name__,
            "raw_pages": result_pages_reference(
                scope_id=selected_scope.scope_id,
                alias=handle,
                descriptor_sha256=descriptor_sha256,
                pages=stored_pages,
            ),
        }
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
    return {
        **payload,
        "declared": True,
        # D (ido-986.14.3): the artifact's pointer at the immutable rows. The
        # declaration payload keeps its own `descriptor` because
        # `put_declaration` reads it out of this dict; what the ARTIFACT keeps
        # is the digest beside this reference, never a second copy of either.
        "raw_pages": result_pages_reference(
            scope_id=selected_scope.scope_id,
            alias=handle,
            descriptor_sha256=descriptor_sha256,
            pages=stored_pages,
        ),
    }


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


def declaring_alias(
    handle: str,
    *,
    scope: Optional[RuntimeHandleScope] = None,
    selected_store: Optional[ResultHandleStore] = None,
) -> str:
    """The handle a page ULTIMATELY pages: ``parent_alias`` followed to the root.

    ``ido-8ps.29``. A page of a page of a listing is still evidence about the
    listing, so the walk goes all the way up rather than one step. A handle with
    no parent is its own declaring alias, and an alias with no stored
    declaration at all is returned unchanged -- this is a lookup, not a
    validator, and it is read on the paging path of every fetch.
    """
    selected_scope = scope or current_scope()
    store_ = selected_store or store()
    current = str(handle or "").strip()
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        declaration = store_.get_declaration(selected_scope, current)
        parent = str((declaration or {}).get("parent_alias") or "")
        if not parent:
            return current
        current = parent
    return current


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


@dataclass(frozen=True)
class SourceRequest:
    """One call to a resolver: one offset window of one query, or its count.

    The descriptor arrives as the JSON that was stored, so a resolver reads
    exactly what the evidence records — not an object assembled here. ``filter``
    and ``filter_columns`` travel together and are never separable: the portal
    silently ignores a filter that names no columns and hands back the whole
    scope, which is a search that did not run wearing the answer of one that
    did.
    """

    descriptor: Mapping[str, Any]
    start: int
    limit: int
    contains: Optional[str] = None
    filter_columns: tuple[str, ...] = ()
    count_only: bool = False


def _call_resolver(resolver: Callable[..., Any], request: SourceRequest) -> dict[str, Any]:
    """Normalise whatever a resolver returns into rows / total / count / columns."""
    response = resolver(request)
    if response is None:
        return {"rows": []}
    if isinstance(response, Mapping):
        return dict(response)
    return {
        "rows": list(getattr(response, "rows", []) or []),
        "total": getattr(response, "total", None),
        "count": getattr(response, "count", None),
        "columns": getattr(response, "columns", None),
    }


class MalformedResolverResponse(ValueError):
    """A resolver answered, but not with a shape a page can be built from.

    (ido-94h, F20) A malformed reply is the SOURCE failing, exactly like a
    resolver that raised, so it is raised where the resolver call is already
    guarded and becomes the same typed ``resolver_error`` outcome. It used to
    escape ``fetch_page`` as a bare ValueError/TypeError from ``dict(row)`` or
    ``int(total)``; callers catch ResultHandleError, so the command errored
    instead of serving the rows it already had.
    """


def _coerce_rows(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``response["rows"]`` as a list of row dicts, or a typed refusal.

    Raised, not repaired: a resolver that answers with a string, or with rows
    that are not mappings, has not returned rows, and guessing at what it meant
    would put invented rows in front of the agent.
    """
    raw = response.get("rows")
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, bytearray)) or isinstance(raw, Mapping):
        raise MalformedResolverResponse(
            "rows is %s, not a sequence of row mappings" % type(raw).__name__
        )
    try:
        items = list(raw)
    except TypeError as error:
        raise MalformedResolverResponse(
            "rows is not a sequence of row mappings (%s)" % error
        ) from None
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(items):
        if not isinstance(row, Mapping):
            raise MalformedResolverResponse(
                "row %d is %s, not a mapping" % (index, type(row).__name__)
            )
        rows.append(dict(row))
        unstorable = _unstorable_value(rows[-1])
        if unstorable is not None:
            # (ido-bdo) Refused on the spot, with the rest of this reply. A page
            # is the stored evidence of what the source returned, and
            # ``_canonical_json`` serialises with ``default=str``: a value with
            # no string form of its own therefore stored as its MEMORY ADDRESS,
            # so the same query stored a different page - a different
            # record_sha256 - on every call. Nothing about that is repairable
            # here, and it used to surface as the walk being unable to make
            # progress: the row was stored, its uid deduplicated against the
            # identical one from the previous offset, the walk advanced without
            # growing, and the fetch spent its whole shared resolver-call purse
            # before giving up with ``resolver_call_limit``. Eight calls to say
            # what the first reply already said.
            raise MalformedResolverResponse(
                "row %d field %r is a %s with no value a stored page could "
                "keep: it has no string form of its own, so it would be "
                "recorded as its memory address"
                % (index, unstorable[0], type(unstorable[1]).__name__)
            )
    return rows


#: How deep into a row's own containers the check below looks. A row is one
#: record of a relation, not a document; past this it is opaque either way.
_UNSTORABLE_SCAN_DEPTH = 6


def _unstorable_value(
    row: Mapping[str, Any], depth: int = _UNSTORABLE_SCAN_DEPTH
) -> Optional[tuple[str, Any]]:
    """The first field of *row* holding a value only its address could name.

    (ido-bdo) Narrow on purpose. It is not "JSON cannot take this": a
    ``datetime``, a ``Decimal`` and a ``UUID`` all serialise through
    ``default=str`` to the value they mean, and a resolver returning one has
    returned data. What is refused is the value that falls back to
    ``object.__repr__`` -- ``<object object at 0x7f...>`` -- which is not the
    row's content, is different in the next process, and makes an immutable
    page unreproducible.
    """
    for name, value in row.items():
        if _is_addressless(value, depth):
            return str(name), value
    return None


def _is_addressless(value: Any, depth: int) -> bool:
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return False
    if depth > 0:
        if isinstance(value, Mapping):
            return any(_is_addressless(item, depth - 1) for item in value.values())
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(_is_addressless(item, depth - 1) for item in value)
    kind = type(value)
    return kind.__str__ is object.__str__ and kind.__repr__ is object.__repr__


def _coerce_row_count(value: Any, field: str) -> Optional[int]:
    """A resolver's ``total``/``count`` as an int, or a typed refusal."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise MalformedResolverResponse(
            "%s is %r, which is not a number of rows" % (field, value)
        ) from None


def _render_row(row: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """A backend row as the producer would have rendered it: ``uid  label``.

    The rendering has to match the producer's, because a stored listing and its
    continuation are one sequence of rows to the agent, and a filter matches the
    row text it was shown.
    """
    uid_field = str(descriptor.get("uid_field") or "")
    if not uid_field:
        uid_field = next(iter(row), "")
    # (ido-56z, F26) Escaped as they are read out of the row, so the record the
    # walk keeps and the line the page prints are the same single line. The row
    # itself is stored verbatim in ``record_json``: this changes how a label is
    # SHOWN, never what the source is recorded as having returned.
    uid = "" if uid_field not in row else one_line(row.get(uid_field) or "")
    label = ""
    for field_name in descriptor.get("label_fields") or ():
        value = row.get(field_name)
        if value not in (None, ""):
            label = one_line(value)
            break
    line = (uid + ROW_SEPARATOR + label) if label else uid
    return {"uid": uid, "line": line, "row": dict(row)}


def _columns_of(response: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The column names and types this view really returned on its first page."""
    declared = response.get("columns")
    if isinstance(declared, Mapping) and declared:
        return {str(name): str(kind) for name, kind in declared.items()}
    if not rows:
        return {}
    return {str(name): type(value).__name__ for name, value in rows[0].items()}


def _walk_records(
    scope: RuntimeHandleScope,
    store_: ResultHandleStore,
    declaration: Mapping[str, Any],
    query_scope: str,
) -> dict[str, Any]:
    """Rebuild the traversal for one query scope from its stored pages.

    Distinct uids in first-seen order: a page that repeats a uid the walk has
    already passed adds nothing to the sequence, which is what keeps a duplicate
    from displacing a row that has not been shown yet. Rebuilding from SQLite is
    what makes hot eviction free of consequence.
    """
    alias = declaration["alias"]
    key = _hot_key(store_, scope, alias, query_scope)
    cached = _cached_walk(key)
    if cached is not None:
        return cached
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    # (ido-oon, F24) Where this traversal begins. The descriptor's start_offset
    # is an offset into the RELATION, and it is the right origin for the
    # unfiltered walk, which re-walks the relation the producer paged. It is the
    # wrong origin for a filtered one: the backend applies `contains` first, so
    # `start` there counts MATCHES. Seeding a filtered walk at 100 asked for the
    # matches after the hundredth one, a relation with 68 of them answered
    # nothing at all, and the walk called itself finished with zero. A filter
    # begins at the first match, whatever part of the relation the handle was
    # declared over.
    next_offset = (
        0 if query_scope
        else int((declaration["descriptor"] or {}).get("start_offset") or 0)
    )
    backend_total: Optional[int] = None
    terminal_offset: Optional[int] = None
    # (ido-7ce, F8) Page at a time, not the whole traversal at once: see
    # ``iter_pages``. Only the uid and the line of each row outlive the page.
    with closing(
        store_.iter_pages(scope, alias=alias, query_scope=query_scope)
    ) as pages:
        for page in pages:
            record = page["record"]
            entries = record.get("records") or []
            if (not entries and not (record.get("rows") or [])
                    and page["source"] != "producer"):
                # (ido-1r0) The stored end of the walk, recognised here exactly
                # as ``_extend_walk`` recognises it live. Seeding the rebuilt
                # walk PAST it is what made every rebuild ask the source to find
                # the end again, one page further out, and store another empty
                # page.
                terminal_offset = page["start_offset"]
                break
            for entry in entries:
                item = _record_of(entry)
                if item["uid"] and item["uid"] in seen:
                    continue
                seen.add(item["uid"])
                records.append(item)
            next_offset = max(
                next_offset, page["start_offset"] + page["limit_requested"]
            )
            if page["backend_total"] is not None:
                backend_total = page["backend_total"]
    # Without a descriptor the producer's own rows are all there will ever be,
    # so the producer's own claim about coverage is the walk's.
    complete = bool(declaration["source_complete"]) and not query_scope
    count_only: Optional[int] = None
    stop_reason: Optional[str] = None
    reconciled = False
    verdict = store_.get_walk_terminal(scope, alias=alias, query_scope=query_scope)
    if verdict is not None and verdict["count_only"] is None:
        # A row without a count is not a verdict: the end was recorded but
        # nothing proved what it covered, so the reconciliation still owes a
        # call. Nothing this module writes looks like that; a hand-written or
        # migrated row might.
        verdict = None
    if verdict is not None:
        # The end was reached and judged against the source's own count in some
        # earlier call or process. Nothing about that is worth asking twice.
        terminal_offset = verdict["terminal_offset"]
        next_offset = verdict["terminal_offset"]
        complete = verdict["complete"]
        count_only = verdict["count_only"]
        stop_reason = verdict["stop_reason"] or None
        reconciled = True
    elif terminal_offset is not None:
        # The end is stored but was never judged: resume AT the empty page, so
        # the reconciliation runs once off a stored page and costs no fetch.
        next_offset = terminal_offset
    walk = {
        "alias": alias,
        "query_scope": query_scope,
        "records": records,
        "seen": seen,
        "next_offset": next_offset,
        "complete": complete,
        "backend_total": backend_total,
        "count_only": count_only,
        "stop_reason": stop_reason,
        "terminal_offset": terminal_offset,
        "terminal_stop_reason": stop_reason,
        "reconciled": reconciled,
        "error": None,
        "bytes": 0,
    }
    _remember_walk(key, walk)
    return walk


class _PageCallBudget:
    """Source-page resolver calls left in ONE ``fetch_page``. (ido-2y3, F6)

    Mutable and shared: ``fetch_page`` makes one and hands the same object to
    every ``_extend_walk`` of that call, so the fill rounds spend from one purse
    instead of each opening a fresh one. Count-only reconciliation is not
    charged here - see ``MAX_RESOLVER_CALLS_PER_FETCH``.
    """

    __slots__ = ("remaining",)

    def __init__(self, calls: int = MAX_RESOLVER_CALLS_PER_FETCH) -> None:
        self.remaining = max(0, int(calls))

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def spend(self) -> None:
        self.remaining -= 1


def _extend_walk(
    scope: RuntimeHandleScope,
    store_: ResultHandleStore,
    declaration: Mapping[str, Any],
    walk: dict[str, Any],
    *,
    descriptor: Mapping[str, Any],
    query_scope: str,
    literal: "Literal",
    needed: int,
    budget: Optional["_PageCallBudget"] = None,
) -> dict[str, Any]:
    """Walk offsets until ``needed`` rows are known or the walk ends.

    **The walk stops on an empty page and on nothing else.** B0 measured a
    sorted offset walk on ``ido_groupDetail_identity`` returning exactly
    ``total`` rows while 20 of 540 members were never shown, so a pager that
    stops at ``rows == total`` reports a complete enumeration that is missing
    people. ``rows == total`` is not a stop condition here and is not a
    completeness proof anywhere; the proof is distinct uids reconciled against
    ``countOnly`` (see ``_reconcile``).

    Never sends a sort: the descriptor has no field for one.

    A refusal on the way is not an exception the agent cannot act on: the rows
    already stored stay served, and the walk carries a typed stop reason the
    page turns into ``incomplete_reason``.
    """
    # (ido-2y3, F6) No budget passed means this is a walk of its own and gets a
    # whole fetch's worth; fetch_page passes ITS budget, so its fill rounds
    # share one.
    budget = _PageCallBudget() if budget is None else budget
    walk["stop_reason"] = None
    walk["error"] = None
    if walk.get("reconciled"):
        # (ido-1r0) The walk reached its end and the source's own count has
        # already judged it. Re-walking would re-prove a stored fact, and on a
        # mismatch it would do so on every fetch for the life of the handle.
        walk["stop_reason"] = walk.get("terminal_stop_reason")
        return walk
    if walk["complete"]:
        return walk
    alias = declaration["alias"]
    key = _hot_key(store_, scope, alias, query_scope)
    try:
        resolver = resolver_for(str(descriptor.get("resolver") or ""))
    except ResultHandleError as error:
        walk["stop_reason"] = "resolver_unavailable"
        walk["error"] = str(error)
        _remember_walk(key, walk)
        return walk
    limit = max(1, int(descriptor.get("page_size") or DEFAULT_PAGE_SIZE))
    columns_for = tuple(descriptor.get("filter_columns") or ()) if query_scope else ()
    while len(walk["records"]) < needed:
        start = int(walk["next_offset"])
        try:
            stored = store_.get_page(
                scope, alias=alias, query_scope=query_scope, start_offset=start
            )
        except (sqlite3.Error, OSError) as error:
            _refuse_for_store(scope, walk, alias, query_scope, start, error)
            break
        if stored is None:
            if budget.exhausted:
                # Over-limit warns and continues: the cursor still advances, so
                # the next call resumes exactly here.
                walk["stop_reason"] = "resolver_call_limit"
                break
            # Charged before the call, so a page the source refused still costs
            # what it cost the source.
            budget.spend()
            try:
                response = _call_resolver(
                    resolver,
                    SourceRequest(
                        descriptor=dict(descriptor),
                        start=start,
                        limit=limit,
                        contains=literal.text or None if query_scope else None,
                        filter_columns=columns_for,
                    ),
                )
                # (ido-94h, F20) Reading the reply is part of the resolver
                # call, not part of the caller's bookkeeping: a malformed rows
                # list or a non-numeric total is refused HERE, inside this
                # guard, so it reaches the agent as resolver_error with the
                # rows already stored still served.
                rows = _coerce_rows(response)
                backend_total = _coerce_row_count(response.get("total"), "total")
            except Exception as error:  # noqa: BLE001
                walk["stop_reason"] = "resolver_error"
                walk["error"] = "%s: %s" % (type(error).__name__, error)
                record_event(
                    {
                        "kind": "result_handle_resolver_error",
                        "scope_id": scope.scope_id,
                        "alias": alias,
                        "query_scope": query_scope,
                        "start": start,
                        "error": type(error).__name__,
                        "detail": str(error)[:300],
                    }
                )
                break
            records = [_render_row(row, descriptor) for row in rows]
            # (ido-2mk, F21) A page is served from what was STORED, so a store
            # that cannot be written ends this walk here rather than out of the
            # process. A read-only file, a full disk and a writer holding the
            # file past the busy timeout all arrive as the same sqlite3 error,
            # and all three mean the same thing to the agent: the rows already
            # walked are still served, this call stopped, and the reason says
            # the store and not the source. The offset is NOT advanced, so the
            # same page is asked for again when the store comes back, and no
            # cursor is offered for a walk whose continuation could not be
            # recorded either.
            try:
                stored = store_.put_page(
                    scope,
                    alias=alias,
                    query_scope=query_scope,
                    start_offset=start,
                    limit_requested=limit,
                    source="resolver",
                    record={"rows": rows, "records": records,
                            "columns": _columns_of(response, rows)},
                    backend_total=backend_total,
                )
                if rows:
                    store_.set_verified_columns(
                        scope, alias,
                        columns=_columns_of(response, rows),
                        sample_row=rows[0],
                    )
            except (sqlite3.Error, OSError) as error:
                _refuse_for_store(scope, walk, alias, query_scope, start, error)
                break
        record = stored["record"]
        page_records = record.get("records") or []
        if not page_records and not (record.get("rows") or []):
            if stored["source"] == "producer":
                # Nothing the producer stored; the backend has not been asked.
                walk["next_offset"] = start + max(1, int(stored["limit_requested"]))
                continue
            # THE stop condition, and the only one.
            walk["complete"] = True
            walk["terminal_offset"] = start
            break
        for entry in page_records:
            item = _record_of(entry)
            if item["uid"] and item["uid"] in walk["seen"]:
                continue
            walk["seen"].add(item["uid"])
            walk["records"].append(item)
        walk["next_offset"] = start + limit
        if stored["backend_total"] is not None:
            walk["backend_total"] = stored["backend_total"]
    if walk["complete"]:
        _reconcile(
            scope, store_, declaration, walk, descriptor=descriptor,
            query_scope=query_scope, literal=literal, resolver=resolver,
        )
    _remember_walk(key, walk)
    return walk


def _refuse_for_store(
    scope: RuntimeHandleScope,
    walk: dict[str, Any],
    alias: str,
    query_scope: str,
    start: int,
    error: BaseException,
) -> None:
    """Stop a walk on a store failure the way it stops on a source failure. (ido-2mk, F21)

    Typed, not raised: the caller has rows in hand that cost the source real
    calls, and losing them to a traceback no command catches is the worst of the
    available outcomes.
    """
    walk["stop_reason"] = "store_unavailable"
    walk["error"] = "%s: %s" % (type(error).__name__, error)
    record_event(
        {
            "kind": "result_handle_store_unavailable",
            "scope_id": scope.scope_id,
            "alias": alias,
            "query_scope": query_scope,
            "start": start,
            "error": type(error).__name__,
            "detail": str(error)[:300],
        }
    )
    logger.warning(
        "result handle page store unavailable at %s@%d: %s", alias, start, error
    )


def _reconcile(
    scope: RuntimeHandleScope,
    store_: ResultHandleStore,
    declaration: Mapping[str, Any],
    walk: dict[str, Any],
    *,
    descriptor: Mapping[str, Any],
    query_scope: str,
    literal: "Literal",
    resolver: Callable[..., Any],
) -> None:
    """Prove coverage by distinct uids against an independent ``countOnly``.

    An empty page ends the walk; it does not prove the walk saw everything.
    ``countOnly`` honours the filter, so this is as available for a filtered
    query as for the whole relation. Completeness is claimed only when the two
    numbers agree — a mismatch leaves the walk incomplete and says so, which is
    exactly the case a ``rows == total`` pager reports as finished.
    """
    walk["count_only"] = None
    if not query_scope and int(descriptor.get("start_offset") or 0) != 0:
        # (ido-oon, F24) The unfiltered proof only. A walk that starts partway
        # into the relation can never account for the rows before it, so it is
        # never complete. A FILTERED walk now starts at the first match
        # regardless (see `_walk_records`), so the same origin says nothing
        # about it, and reporting it here said the query was unprovable when it
        # had in fact just been asked wrongly.
        walk["complete"] = False
        # Read off the descriptor, so it is free to decide again and needs no
        # stored verdict; marking it settled keeps the walk from re-walking.
        _settle_walk(walk, "offset_origin_not_zero")
        return
    if not descriptor.get("count_only", True):
        walk["complete"] = False
        _settle_walk(walk, "countonly_unavailable")
        return
    try:
        response = _call_resolver(
            resolver,
            SourceRequest(
                descriptor=dict(descriptor),
                start=0,
                limit=0,
                contains=literal.text or None if query_scope else None,
                filter_columns=(tuple(descriptor.get("filter_columns") or ())
                                if query_scope else ()),
                count_only=True,
            ),
        )
        # (ido-94h, F20) The count is read inside the guard too: a countOnly
        # that answers "n/a" is a source failure, reported as countonly_error,
        # not a ValueError out of the command.
        count = _coerce_row_count(response.get("count"), "count")
        if count is None:
            count = _coerce_row_count(response.get("total"), "total")
    except Exception as error:  # noqa: BLE001
        walk["complete"] = False
        walk["stop_reason"] = "countonly_error"
        walk["error"] = "%s: %s" % (type(error).__name__, error)
        return
    if count is None:
        walk["complete"] = False
        walk["stop_reason"] = "countonly_unavailable"
        return
    distinct = len(walk["records"])
    walk["count_only"] = int(count)
    if int(count) != distinct:
        walk["complete"] = False
        _settle_walk(walk, "countonly_mismatch")
    else:
        _settle_walk(walk, None)
    _record_walk_terminal(scope, store_, declaration, walk, query_scope=query_scope)
    record_event(
        {
            "kind": "result_handle_reconciled",
            "scope_id": scope.scope_id,
            "alias": declaration["alias"],
            "query_scope": query_scope,
            "distinct_uids": distinct,
            "count_only": int(count),
            "complete": bool(walk["complete"]),
        }
    )


def _settle_walk(walk: dict[str, Any], stop_reason: Optional[str]) -> None:
    """Mark a walk judged: this is its end and this is why, until it changes."""
    walk["stop_reason"] = stop_reason
    walk["terminal_stop_reason"] = stop_reason
    walk["reconciled"] = True


def _record_walk_terminal(
    scope: RuntimeHandleScope,
    store_: ResultHandleStore,
    declaration: Mapping[str, Any],
    walk: dict[str, Any],
    *,
    query_scope: str,
) -> None:
    """Persist the verdict so the next process inherits it instead of re-proving it.

    Best effort, like every other write on a page's path: a walk whose verdict
    could not be written still serves its rows and is still right for this
    process; it only pays for the proof again next time.
    """
    if walk.get("terminal_offset") is None or walk.get("count_only") is None:
        return
    try:
        store_.put_walk_terminal(
            scope,
            alias=declaration["alias"],
            query_scope=query_scope,
            terminal_offset=int(walk["terminal_offset"]),
            complete=bool(walk["complete"]),
            count_only=int(walk["count_only"]),
            distinct_uids=len(walk["records"]),
            stop_reason=str(walk.get("terminal_stop_reason") or ""),
        )
    except Exception as error:  # noqa: BLE001
        record_event({"kind": "result_handle_walk_terminal_failed",
                      "scope_id": scope.scope_id,
                      "alias": declaration["alias"],
                      "query_scope": query_scope,
                      "error": type(error).__name__})
        logger.warning("result handle walk terminal could not be stored: %s", error)


def _filter_records(
    records: Sequence[Mapping[str, Any]], literal: "Literal"
) -> list[dict[str, Any]]:
    """Case-insensitive literal over the rendered rows, uid and label included."""
    needle = literal.text.casefold()
    return [
        _record_of(record)
        for record in records
        if needle in str(record["line"]).casefold()
    ]


_STOP_REASON_NOTES = {
    "resolver_error": "the source refused a page of this query",
    "resolver_unavailable": "this process cannot reach the source that produced these rows",
    "resolver_call_limit": "this call reached its backend-page limit; ask again to continue",
    "countonly_mismatch": "the walk and the source's own count disagree",
    "countonly_unavailable": "the source offers no independent count to prove coverage",
    "countonly_error": "the source refused the count that would prove coverage",
    "offset_origin_not_zero": "this handle starts partway into the relation",
    "store_unavailable": "the page store could not be written, so this walk stopped where it was",
    "producer_materialized_subset": "the producing command did not materialise every row",
}


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

    ``contains`` is a literal, not a question: it is normalised and matched, and
    it is never split into tokens to be intersected. A named lookup is one
    filtered call at any page position; paging is for enumeration.

    ``cursor`` is the page token printed on the page it continues (``O7/p2``,
    or ``O7/f1p2`` for a filtered traversal). It is issued, never composed: a
    token this turn did not issue, or one issued for another handle or another
    filter, is refused by name.
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
    descriptor = declaration["descriptor"] or {}
    filter_columns = tuple(descriptor.get("filter_columns") or ())
    descriptor_sha256 = declaration["descriptor_sha256"]
    # (ido-56z, F26) A filter that normalisation empties is refused HERE, before
    # a plan is chosen. ``contains="%"`` and ``contains="   "`` both strip to
    # nothing, and an empty literal has an empty ``scope``, so the call fell
    # through to the unfiltered enumeration and printed a page with no
    # ``filter=`` in its header at all: the agent asked a question, got every
    # row of the listing, and nothing on the page said its filter had been
    # dropped. Refused rather than repaired -- there is no literal to guess at.
    # It never ran, so it is never tagged, never stored and never a page of a
    # traversal: ``_unsupported_page`` files no page declaration, which is what
    # keeps it out of the zero-match echo marker (``page_matched_nothing``).
    if contains is not None and str(contains) != "" and not literal.text:
        return _unsupported_page(
            declaration=declaration,
            literal=literal,
            materialized=int(declaration["materialized"]),
            total=int(declaration["total"]),
            budget_bytes=budget_bytes,
            scope=selected_scope,
            store_=store_,
            reason="empty_filter_literal",
            message=(
                "contains=%r leaves no literal to match: after normalisation "
                "and the removal of the characters the backend treats as LIKE "
                "wildcards there is nothing left of it. This filter was NOT "
                "run, so no rows are shown and none is a zero. Pass a "
                "literal with at least one character the backend can "
                "match, or omit contains to page the listing."
                % (str(contains)[:60],)
            ),
            notes=literal.notes,
        )
    position = 0
    if cursor:
        position = _check_cursor(
            decode_cursor(cursor, alias=alias, scope=selected_scope,
                          selected_store=store_),
            alias=alias,
            query_scope=query_scope,
            literal=literal,
            descriptor_sha256=descriptor_sha256,
        )
    budget = budget_bytes or page_max_bytes_from_env()
    page_size = max(1, int(descriptor.get("page_size") or declaration["page_size"]
                           or DEFAULT_PAGE_SIZE))
    notes: list[str] = list(literal.notes)
    warnings: list[str] = []

    base = _walk_records(selected_scope, store_, declaration, "")
    base_complete = bool(declaration["source_complete"]) or bool(base["complete"])

    # Which query this call actually runs, and whether it can be run at all.
    if query_scope:
        if base_complete:
            # Every row of the relation is already here, so a literal over the
            # rendered rows IS a whole-relation search, not a partial one.
            plan = "local-filter"
        elif descriptor and filter_columns:
            plan = "backend-filter"
        else:
            return _unsupported_page(
                declaration=declaration,
                literal=literal,
                materialized=len(base["records"]),
                total=int(declaration["total"]),
                budget_bytes=budget,
                scope=selected_scope,
                store_=store_,
                reason=("no_verified_filter_columns" if descriptor
                        else "producer_materialized_subset"),
                message=(
                    "Filtering is unsupported for this handle. "
                    "This handle holds %d of %d rows and has no verified "
                    "filterable columns for this view, so a filter over it could "
                    "not speak for the whole relation. Page it, or re-run the "
                    "producing command with a narrower query."
                    % (len(base["records"]), int(declaration["total"]))
                ),
            )
    else:
        plan = "backend-walk" if descriptor else "local"

    walk = base
    if plan == "backend-filter":
        walk = _walk_records(selected_scope, store_, declaration, query_scope)

    # The traversal this page's tokens belong to, resolved once the query is
    # known to be runnable: an unsupported filter never gets a tag, because it
    # never gets a page to continue.
    tag = _cursor_tag(selected_scope, store_, alias, query_scope)

    # How many rows this page can show is decided once, by the packer, after
    # every line above the rows is known. Deciding it twice is how a page skips
    # a row: a header that grew by a cursor would push out a row the previous
    # cursor had already counted as shown.
    rounds = 0
    # (ido-2y3, F6) One purse for this fetch. Every fill round below spends from
    # it, so MAX_RESOLVER_CALLS_PER_FETCH bounds the call, not the round.
    call_budget = _PageCallBudget()
    while True:
        if plan in ("backend-walk", "backend-filter"):
            # Ask for enough rows to fill the observation, not for one backend
            # page: with a small page size a page-at-a-time fill would return a
            # three-line observation and call it a page.
            _extend_walk(
                selected_scope, store_, declaration, walk,
                descriptor=descriptor, query_scope=query_scope, literal=literal,
                needed=position + _rows_wanted(walk, budget, page_size) * (rounds + 1),
                budget=call_budget,
            )
            records: list[dict[str, Any]] = walk["records"]
        elif plan == "local-filter":
            records = _filter_records(base["records"], literal)
        else:
            records = base["records"]
        available = [str(record["line"]) for record in records[position:]]
        probe = _provisional_page(
            declaration=declaration, alias=alias, rows=available, records=records,
            base=base, walk=walk, plan=plan, literal=literal,
            filter_columns=filter_columns, descriptor=descriptor,
            position=position, scope=selected_scope, store_=store_, notes=notes,
            warnings=warnings,
            placeholder_cursor=cursor_placeholder(
                alias, tag, pages_at_most=len(records) + 2
            ),
        )
        shown, _ = _pack(probe, budget_bytes=budget)
        if (
            not plan.startswith("backend")
            or walk["complete"]
            or walk.get("stop_reason")
            or len(shown) < len(available)
            or rounds >= MAX_FILL_ROUNDS
        ):
            break
        rounds += 1

    page = probe
    matched = page.matched
    # Outcome classes, distinguished. None of them is a stand-in for another,
    # and each is decided on rows that EXIST for this query, not on rows that
    # happened to fit in this observation.
    error_reason = page.incomplete_reason in (
        "resolver_error", "resolver_unavailable", "countonly_error",
        # (ido-2mk, F21) A page that could not be stored is a failure of this
        # call, not a property of the query, so an empty one must not read as a
        # zero.
        "store_unavailable",
    )
    if error_reason and not available:
        page.outcome = "error"
        notes.append(
            "%s (%s). Rows already stored are still readable; nothing here "
            "shows what the unread rows contain."
            % (
                "This query's pages could not be stored, so it did not run"
                if page.incomplete_reason == "store_unavailable"
                else "The source refused this query",
                walk.get("error") or page.incomplete_reason,
            )
        )
    elif matched == 0 and page.matched_complete:
        page.outcome = "complete-zero"
        notes.append(_zero_message(literal, page.filter_columns, declaration))
    elif matched == 0:
        page.outcome = "partial"
        notes.append(
            "No rows matched here, and this query is not complete (%s), so this "
            "is not a zero: it is an unfinished search."
            % _STOP_REASON_NOTES.get(page.incomplete_reason or "", "incomplete")
        )
    elif not page.matched_complete:
        page.outcome = "partial"
        notes.append(
            "Coverage is not proven for this query (%s), so treat these rows as "
            "some of the matches, never as all of them."
            % _STOP_REASON_NOTES.get(page.incomplete_reason or "", "incomplete")
        )
    else:
        page.outcome = "rows"
    if page.incomplete_reason == "countonly_mismatch":
        notes.append(
            "The walk reached %d distinct rows and the source's own count says "
            "%s. Rows retrieved equalling the reported total is not coverage; "
            "the disagreement is reported rather than resolved."
            % (len(walk["records"]), walk.get("count_only"))
        )
    if page.page_index >= PAGE_WARNING_AFTER and not literal.text:
        warnings.append(
            "Note: this is page %d of %s in this turn. For a named lookup one "
            "filtered call finds the row at any page position — pass "
            "contains=<name>. Paging still works and is not being restricted."
            % (page.page_index, alias)
        )
    page.notes = tuple(notes)
    page.warnings = tuple(warnings)

    shown, over_budget = _pack(page, budget_bytes=budget)
    remaining = len(available) - len(shown)
    walk_can_continue = (
        plan.startswith("backend")
        and not walk["complete"]
        # A refusal or a call bound stops THIS call, not the enumeration: the
        # cursor resumes exactly where the walk stopped, and the stored pages
        # cost nothing to pass again. A resolver this process cannot reach, or a
        # walk that ended without proving coverage, is not continuable, and the
        # page says so rather than offering a cursor that would not move.
        and walk.get("stop_reason") in (None, "resolver_call_limit", "resolver_error")
    )
    page.rows = list(shown)
    if remaining > 0 or walk_can_continue:
        page.continuation = "cursor"
        page.next_cursor = encode_cursor(
            alias=alias, query_scope=query_scope, position=position + len(shown),
            descriptor_sha256=descriptor_sha256, scope=selected_scope,
            selected_store=store_,
        )
    elif page.matched_complete:
        # Completeness is a property of the query that ran. A filter the backend
        # applied to the whole relation is complete even when the base listing
        # this handle materialised is not; the header reports both numbers.
        page.continuation = "complete"
        page.next_cursor = None
    else:
        page.continuation = "source-incomplete"
        page.next_cursor = None
    page.observation, page.warnings = _assemble(
        page, shown, over_budget=over_budget, budget_bytes=budget
    )
    _record_page_event(selected_scope, page, literal)
    _link_page(selected_scope, store_, page, declaration, query_scope)
    return page


def _rows_wanted(walk: Mapping[str, Any], budget: int, page_size: int) -> int:
    """How many rows it would take to fill one observation at this row width."""
    records = walk["records"]
    if records:
        widths = [len(str(record["line"]).encode("utf-8")) + 1 for record in records]
        estimate = max(16, sum(widths) // len(widths))
    else:
        estimate = 64
    return max(page_size, -(-budget // estimate))


def _provisional_page(
    *,
    declaration: Mapping[str, Any],
    alias: str,
    rows: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    base: Mapping[str, Any],
    walk: Mapping[str, Any],
    plan: str,
    literal: "Literal",
    filter_columns: tuple[str, ...],
    descriptor: Mapping[str, Any],
    position: int,
    scope: RuntimeHandleScope,
    store_: "ResultHandleStore",
    notes: Sequence[str],
    warnings: Sequence[str],
    placeholder_cursor: str,
) -> ResultPage:
    """The page as it will be, with the widest header it could carry.

    Packing happens against this: ``source-incomplete`` is the longest
    continuation word and the cursor is the widest token this traversal could
    print, so the real header is never wider than the one the rows were measured
    against.
    """
    source_complete = bool(declaration["source_complete"]) or bool(base["complete"])
    matched_complete = (bool(walk["complete"]) if plan == "backend-filter"
                        else source_complete)
    stop_reason = walk.get("stop_reason") if plan.startswith("backend") else None
    if not source_complete and not stop_reason and not descriptor:
        stop_reason = "producer_materialized_subset"
    page = ResultPage(
        handle=alias,
        kind=declaration["kind"],
        summary=declaration["summary"],
        rows=list(rows),
        matched=len(records),
        total=int(declaration["total"]),
        materialized=len(base["records"]),
        source_complete=source_complete,
        matched_complete=matched_complete,
        continuation="source-incomplete",
        # Placeholders chosen as the LONGEST each field can become, so the
        # header the rows were packed against is never narrower than the header
        # the page finally carries. Both are overwritten before assembly.
        outcome="complete-zero",
        incomplete_reason=stop_reason,
        next_cursor=placeholder_cursor,
        position=position,
        page_index=_page_index(scope, store_, alias, position),
        parent_alias=alias,
        page_alias=current_execute_alias(),
        literal=literal.text or None,
        filter_columns=filter_columns if plan == "backend-filter" else (),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )
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
    notes: Sequence[str] = (),
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
        page_index=_page_index(scope, store_, declaration["alias"], 0),
        parent_alias=declaration["alias"],
        page_alias=current_execute_alias(),
        literal=literal.text or None,
        notes=(message,) + tuple(notes),
    )
    budget = budget_bytes or page_max_bytes_from_env()
    shown, over_budget = _pack(page, budget_bytes=budget)
    page.rows = shown
    page.observation, page.warnings = _assemble(
        page, shown, over_budget=over_budget, budget_bytes=budget
    )
    _record_page_event(scope, page, literal)
    return page


def _page_index(
    scope: RuntimeHandleScope, store_: "ResultHandleStore", alias: str, position: int
) -> int:
    """How many distinct pages of this handle have been served in this turn.

    Counted by start position rather than by call, so a retried cursor is the
    same page it was the first time and cannot inflate the count.
    """
    key = "%s:%s" % (_cache_namespace(scope, store_), alias)
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

    It is also where a page's own coverage is recorded (``ido-3f8``). The three
    fields below say, per page observation: which listing this is a page OF
    (``parent_alias``), which query it ran (``query_scope``, empty for the
    unfiltered listing and otherwise the content-addressed name of the filter
    literal) and how many rows it carried (``materialized``). A filtered page
    with no rows is a page that retrieved nothing, and saying so in three
    columns is what keeps a reader from having to infer it from the prose the
    page or its backend wrote -- see :func:`page_matched_nothing`.
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
                # The rows THIS page carried, never the listing's count: a zero
                # here under a filtered query scope is the zero-match marker
                # (``ido-3f8``).
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
        return
    _stamp_page_clause(scope, store_, page.page_alias, declaration["alias"])


# ---------------------------------------------------------------------------
# What a page observation proves about a name (ido-3f8)
# ---------------------------------------------------------------------------

#: The ``filter="..."`` span :func:`_header` prints. Read back HERE, by the
#: module that writes it, so the one place that knows the header's shape is the
#: one place that parses it.
_HEADER_FILTER_RE = re.compile(r'filter="([^"\n]*)"')


def page_matched_nothing(declaration: Optional[Mapping[str, Any]]) -> bool:
    """Is this the declaration of a FILTERED page observation that carried no rows?

    (``ido-3f8``) The marker a coverage reader needs, and the page layer already
    files it: ``_link_page`` records, for every page observation's own ``O``
    alias, the listing it paged (``parent_alias``), the traversal it ran
    (``query_scope``, the content-addressed name of the filter literal — see
    :attr:`Literal.scope`) and the rows that page actually carried
    (``materialized``). Those three columns are the structured fact, so nothing
    new is written and nothing new is persisted; what was missing was a reader
    and a name for it.

    The three states a reader has to tell apart are exactly the three answers
    here. A page nobody fetched has no declaration at all, so there is nothing
    to ask and the answer is False. An unfiltered page has an empty
    ``query_scope`` — it ran no literal, and no literal of its can be echoed —
    so it is False. A filtered page with no rows is True.

    Zero rows, not "zero matches proven": a filtered page that carried no rows
    retrieved nothing whatever the reason (a complete zero, a search that
    stopped early, a refused resolver, a cursor past the last match), and the
    only use of this answer is to say that such a page cannot be the reason a
    name looks retrieved. That is sound for every one of those reasons, and
    reading it as "the thing does not exist" would not be sound for any of them.
    It is deliberately NOT read that way: see
    :func:`fastworkflow.answer_coverage.drop_zero_match_echo`.
    """
    if not declaration:
        return False
    if not str(declaration.get("parent_alias") or ""):
        return False
    if not str(declaration.get("query_scope") or ""):
        return False
    return int(declaration.get("materialized") or 0) == 0


def echoed_literal(
    declaration: Optional[Mapping[str, Any]], text: Any
) -> str:
    """The literal a zero-row filtered page ran, or ``""`` (``ido-3f8``).

    Taken from the page's own header and PROVED against the stored declaration:
    the candidate counts only when its digest is the ``query_scope`` the page
    was filed under, so this can never return a literal some other text happened
    to spell. The store keeps that digest and not the literal text
    (``answer_rehydration.stored_rows_block``), which is why the text is asked
    for the spelling and the store is asked whether it is the right one.

    Normalised, because the digest is over the normalised literal: what comes
    back is the literal as it was really matched, which is also the form a
    reader will look for.
    """
    if not page_matched_nothing(declaration):
        return ""
    query_scope = str((declaration or {}).get("query_scope") or "")
    for candidate in _HEADER_FILTER_RE.findall(str(text or "")):
        literal = normalize_literal(candidate)
        if literal.text and literal.scope == query_scope:
            return literal.text
    return ""


def _stamp_page_clause(
    scope: RuntimeHandleScope,
    store_: ResultHandleStore,
    page_alias: str,
    parent_alias: str,
) -> None:
    """Give a page observation the subject its HANDLE was declared for (ido-8ps.29).

    ``CommandExecutor._remember_execute_context`` stamps every execute step with
    the context the command RAN IN, taken before dispatch. That is the right
    fact for a command that produces new output and the wrong one for a command
    that re-serves output produced elsewhere: an agent that opens Christopher
    Hubbard's identity and then pages Alan Cooper's entitlement listing gets
    Cooper's rows stamped "Identity ... Christopher Hubbard", and every reader
    of the clause -- the alias line, the answer-time evidence sentence
    (``answer_coverage``), the attribution check (``answer_attribution``) --
    then reads another subject's rows as this subject's evidence.

    A page is evidence about the handle it pages, wherever it is fetched from,
    so the clause follows the declaring handle. If the declaring handle carries
    no recorded clause the dispatch-time stamp is DROPPED rather than kept:
    unrecorded is a state every reader handles, and a wrong subject is one they
    all believe.

    Best effort from end to end: a presentation fact must never fail a fetch.
    """
    try:
        from fastworkflow.observation_offloading.state import (
            context_clause_of,
            forget_context_clause,
            record_context_clause,
        )

        from fastworkflow.observation_offloading.state import durable_archive

        # The same archive ``record_context_clause`` writes a dispatch-time
        # stamp to, resolved the same way, because a subject read out of one
        # file and corrected in another is two subjects. In a running workflow
        # that file is the store's own -- ``store()`` is keyed by the agent's
        # archive path -- so the declaring handle's recorded subject is read
        # back even in a process that only RESUMED the turn, where the clause
        # map starts empty and this page's subject would otherwise be dropped
        # (ido-dhw, F3).
        subjects = durable_archive()
        root = declaring_alias(parent_alias, scope=scope, selected_store=store_)
        declaring = context_clause_of(scope, root, selected_archive=subjects)
        stamped = context_clause_of(scope, page_alias, selected_archive=subjects)
        if declaring == stamped:
            return
        if declaring is None:
            forget_context_clause(scope, page_alias, selected_archive=subjects)
        else:
            record_context_clause(
                scope, page_alias, declaring, selected_archive=subjects)
        record_event(
            {
                "kind": "result_handle_page_clause",
                "scope_id": scope.scope_id,
                "page_alias": page_alias,
                "parent_alias": parent_alias,
                "declaring_alias": root,
                "clause_at_dispatch": stamped,
                "clause_recorded": declaring,
            }
        )
    except Exception as error:  # noqa: BLE001 - presentation must never fail a fetch
        record_event(
            {
                "kind": "result_handle_page_clause_refused",
                "scope_id": scope.scope_id,
                "page_alias": page_alias,
                "parent_alias": parent_alias,
                "error": type(error).__name__,
            }
        )


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "HOT_ROWS_MAX_BYTES",
    "HOT_ROWS_MAX_BYTES_ENV",
    "Literal",
    "MalformedResolverResponse",
    "RESULT_PAGE_MAX_BYTES",
    "RESULT_PAGE_MAX_BYTES_ENV",
    "RESULT_PAGES_REF_KEY",
    "ResultHandleError",
    "ResultHandleSpec",
    "ResultHandleStore",
    "ResultPage",
    "SourceDescriptor",
    "SourceRequest",
    "UNSORTED_OFFSET",
    "WILDCARD_CHARACTERS",
    "CURSOR_TOKEN_EXAMPLE",
    "FIRST_CURSOR_PAGE",
    "MAX_CURSOR_PAGE",
    "current_execute_alias",
    "current_scope",
    "cursor_placeholder",
    "cursor_token",
    "declare",
    "decode_cursor",
    "encode_cursor",
    "fetch_page",
    "handle_declaration",
    "hot_rows_max_bytes_from_env",
    "normalize_literal",
    "one_line",
    "echoed_literal",
    "page_matched_nothing",
    "declaring_alias",
    "page_max_bytes_from_env",
    "parent_handle",
    "MAX_RESOLVER_CALLS_PER_FETCH",
    "PAGE_WARNING_AFTER",
    "register_resolver",
    "registered_resolvers",
    "result_pages_reference",
    "release_scope",
    "reset_result_handle_state",
    "resolver_for",
    "store",
    "unregister_resolver",
]
