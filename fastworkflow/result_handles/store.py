"""Persistence for result declarations, pages, walks, and cursors."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict
from typing import Any, Iterator, Mapping, Optional

from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.result_handles.common import (
    FIRST_CURSOR_PAGE,
    ResultHandleError,
    canonical_json as _canonical_json,
    digest as _digest,
    utc_now as _now,
)


def _stored_row_count(record: Mapping[str, Any]) -> int:
    return len(record.get("records") or record.get("rows") or [])


class ResultHandleStore:
    """Turn-scoped SQLite for declarations and immutable raw page records.

    It lives in the same database file as ``observation_offload_handles`` and
    follows the same pattern — scope-keyed rows, digest-verified payloads,
    insert-or-nothing writes — in its own tables. The offload table is not
    touched.

    Retention. Rows are not deleted by this module, and that is a division of
    labour, not an exemption (ido-gls). They live exactly as long as the archive
    file that holds the turn's observations, which is what makes a page
    reconstructable for evaluation after the live turn has ended; the hot cache
    bound is a residency bound and not a retention bound. What deletes them is
    ``fastworkflow.observation_offloading.erasure``, which owns erasure and
    retention for every scope-keyed table in this file -- including the ones
    this class adds -- and which discovers those tables structurally, so a table
    added here is erased with its channel without that module being edited.
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
            # (ido-1r0) Where a traversal ended and what proved it. The empty
            # page that ends a walk is stored like any other page, but the
            # countOnly reconciliation that turns "the pages ran out" into
            # "every row was seen" was memory only: after an eviction or a
            # restart the walk asked the backend to prove its end again, stored
            # another empty page one offset further on, and did it again on the
            # next fetch. This row is that proof, written once the two numbers
            # are known, so the end of a walk costs the source nothing twice.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_handle_walks (
                    scope_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    query_scope TEXT NOT NULL,
                    terminal_offset INTEGER NOT NULL,
                    complete INTEGER NOT NULL,
                    count_only INTEGER,
                    distinct_uids INTEGER NOT NULL,
                    stop_reason TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias, query_scope)
                )
                """
            )
            # (ido-986.14.11) The tokens themselves. A token carries no payload:
            # everything the old base64 cursor spelled out - query scope, offset,
            # descriptor digest - lives in these rows, so the agent-visible
            # string can be five characters and still cannot be mangled into a
            # different query. They live in SQLite beside the pages, so a token
            # printed before a hot-cache eviction or a process restart still
            # resolves.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_handle_cursor_tags (
                    scope_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    query_scope TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias, query_scope)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_handle_cursors (
                    scope_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    query_scope TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    descriptor_sha256 TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias, tag, page)
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS result_handle_cursors_position
                ON result_handle_cursors(
                    scope_id, alias, tag, position, descriptor_sha256
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        """Every statement in this class goes through here. It is a SEAM.

        (fix-iq53.2.10, F5b) A subclass that overrides this one method changes
        how the whole store reaches SQLite without touching a query. That is
        how ``ReadOnlyResultHandleStore`` below opens a file it must not write,
        and how a caller that knows more about a particular file than the
        framework ever can -- whether it is frozen evidence or a live run
        directory -- routes the same reads through its own connection policy.
        The seam is the reason this class takes no ``immutable`` parameter, and
        no other parameter, for read-only opens.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def open_readonly(cls, db_path: str) -> "ReadOnlyResultHandleStore":
        """Open an EXISTING store without writing a byte to it.

        The named entry point for reading a historical store. See
        ``ReadOnlyResultHandleStore`` for why the ordinary constructor is not
        safe to point at one.
        """
        return ReadOnlyResultHandleStore(db_path)

    # -- declarations ------------------------------------------------------

    def put_declaration(
        self,
        scope: RuntimeHandleScope,
        alias: str,
        payload: Mapping[str, Any],
        *,
        first_page_offset: Optional[int] = None,
        first_page_sha256: Optional[str] = None,
    ) -> dict[str, Any]:
        """Write a declaration once. A redeclaration of the same query is a no-op.

        Two different queries under one alias would make the alias ambiguous —
        the agent would ask for O42 and get whichever was written last — so the
        second one is refused by name instead.

        (ido-ecd, F19) The descriptor digest alone is not that identity. Every
        descriptor-less handle shares one digest, and re-running the same query
        against a changed backend shares it too, so a second, different listing
        declared under an alias a restarted sequence handed out again reported
        itself declared with its own total while the alias went on serving the
        first listing's rows. What a listing actually IS, for this purpose, is
        its first page of rows: pass ``first_page_offset`` with the digest of
        the producer page this declaration is about to write (``None`` for a
        listing that materialised nothing), and a redeclaration whose first page
        is not the stored first page is refused by name like any other different
        query. Callers that file something other than a producer listing -- a
        page observation's own alias -- pass neither and are unaffected.
        """
        scope_json = json.dumps(
            asdict(scope), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
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
            # Nothing inserted means the alias was already taken by an earlier
            # declaration, which is the only case the identity below judges.
            redeclaration = not cursor.rowcount
            conn.commit()
        stored = self.get_declaration(scope, alias)
        if stored is None:
            raise ResultHandleError("result handle %s could not be stored" % alias)
        if stored["descriptor_sha256"] != payload["descriptor_sha256"]:
            raise ResultHandleError(
                "result handle %s already describes a different query in this "
                "scope; an alias identifies one observation" % alias
            )
        if redeclaration and first_page_offset is not None:
            kept = self._producer_page_digest(
                scope, alias=alias, start_offset=int(first_page_offset)
            )
            if kept != first_page_sha256:
                raise ResultHandleError(
                    "result handle %s already holds a different listing in this "
                    "scope: its first page is not the one being declared. An "
                    "alias identifies one observation" % alias
                )
        return stored

    def _producer_page_digest(
        self, scope: RuntimeHandleScope, *, alias: str, start_offset: int
    ) -> Optional[str]:
        """The digest of the rows the PRODUCER filed at this alias, if any.

        Only a producer page answers: a page the walk stored at the same offset
        is the backend's account of the same query, not the listing's own first
        page, and a handle that materialised nothing has no first page at all.
        """
        page = self.get_page(
            scope, alias=alias, query_scope="", start_offset=int(start_offset)
        )
        if page is None or page["source"] != "producer":
            return None
        return str(page["record_sha256"])

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

        (ido-h0c, F28) ``row_count`` is the count of the RECORDS the page
        carries -- see ``_stored_row_count``. Reading it off ``rows`` alone made
        the column read 0 for every producer page ever written, because a
        producer files its rendered lines under ``records`` and leaves ``rows``
        empty. Rows written before that fix keep their 0 and are not rewritten:
        this is an append-only table and a stored page is evidence. A reader
        that spans the change therefore sees 0 on old producer pages and the
        real count on new ones, and the way to tell them apart is that
        ``record_json`` was always right -- ``len(record["records"] or
        record["rows"])`` is the count for a page of either vintage, and is what
        a reader wanting one number across the boundary should use.
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
                    _stored_row_count(record),
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

    def iter_pages(
        self, scope: RuntimeHandleScope, *, alias: str, query_scope: str
    ) -> "Iterator[dict[str, Any]]":
        """The stored pages of one traversal, in offset order, ONE AT A TIME.

        (ido-7ce, F8) ``list_pages`` decodes every page of a walk before the
        caller sees the first one, so rebuilding a large traversal held every
        page's rows - which the stored payload carries twice, once in ``rows``
        and once inside each record - in memory at the same moment. A rebuild
        reads each page, takes the uid and the line out of it and has no further
        use for it, so this yields them and lets each one go. What a rebuild
        holds is then the walk it is building, which the hot bound measures,
        plus one page.

        The caller may stop early - a rebuild stops at the stored end of the
        walk - so close the generator (``contextlib.closing``) to put the
        connection back rather than leaving it to the collector.
        """
        conn = self._connect()
        try:
            for row in conn.execute(
                """
                SELECT * FROM result_handle_pages
                WHERE scope_id = ? AND alias = ? AND query_scope = ?
                ORDER BY start_offset
                """,
                (scope.scope_id, alias, query_scope),
            ):
                yield self._decode_page(row)
        finally:
            conn.close()

    def list_pages(
        self, scope: RuntimeHandleScope, *, alias: str, query_scope: str
    ) -> list[dict[str, Any]]:
        """Every stored page of one traversal at once. See ``iter_pages``."""
        with closing(
            self.iter_pages(scope, alias=alias, query_scope=query_scope)
        ) as pages:
            return list(pages)

    def list_page_query_scopes(
        self, scope: RuntimeHandleScope, *, alias: str
    ) -> list[str]:
        """Every traversal that has stored pages for *alias*, base one first.

        A handle's rows are stored per query scope: the unfiltered walk under
        ``""`` and one scope per literal filter run against it. Reading a handle
        back whole at answer time (``ido-8ps.18``) has to know which scopes
        exist, and the scope is the only key ``list_pages`` cannot supply itself.
        Read-only, like every other list method here.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT query_scope FROM result_handle_pages
                WHERE scope_id = ? AND alias = ?
                ORDER BY query_scope
                """,
                (scope.scope_id, alias),
            ).fetchall()
        scopes = [str(row["query_scope"]) for row in rows]
        return [value for value in scopes if not value] + [
            value for value in scopes if value
        ]

    # -- where a walk ended, and what proved it ---------------------------

    def put_walk_terminal(
        self,
        scope: RuntimeHandleScope,
        *,
        alias: str,
        query_scope: str,
        terminal_offset: int,
        complete: bool,
        count_only: Optional[int],
        distinct_uids: int,
        stop_reason: str,
    ) -> None:
        """Record the offset a walk ended at and the count that judged it.

        Written only when the source returned a real count, because that is the
        only verdict a later process can trust without asking again: the empty
        page proves the pages ran out, and the count proves nothing was missed.
        A verdict is replaceable — rows the walk did not have when it was
        written would make a new one — so this is an upsert, unlike a page.
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO result_handle_walks (
                    scope_id, alias, query_scope, terminal_offset, complete,
                    count_only, distinct_uids, stop_reason, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias, query_scope) DO UPDATE SET
                    terminal_offset = excluded.terminal_offset,
                    complete = excluded.complete,
                    count_only = excluded.count_only,
                    distinct_uids = excluded.distinct_uids,
                    stop_reason = excluded.stop_reason,
                    recorded_at = excluded.recorded_at
                """,
                (
                    scope.scope_id,
                    alias,
                    query_scope,
                    int(terminal_offset),
                    1 if complete else 0,
                    None if count_only is None else int(count_only),
                    int(distinct_uids),
                    str(stop_reason or ""),
                    _now(),
                ),
            )
            conn.commit()

    def get_walk_terminal(
        self, scope: RuntimeHandleScope, *, alias: str, query_scope: str
    ) -> Optional[dict[str, Any]]:
        """The stored verdict for one traversal, or ``None`` if it has none."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT * FROM result_handle_walks
                WHERE scope_id = ? AND alias = ? AND query_scope = ?
                """,
                (scope.scope_id, alias, query_scope),
            ).fetchone()
        if row is None:
            return None
        return {
            "alias": str(row["alias"]),
            "query_scope": str(row["query_scope"]),
            "terminal_offset": int(row["terminal_offset"]),
            "complete": bool(row["complete"]),
            "count_only": (None if row["count_only"] is None
                           else int(row["count_only"])),
            "distinct_uids": int(row["distinct_uids"]),
            "stop_reason": str(row["stop_reason"] or ""),
            "recorded_at": str(row["recorded_at"]),
        }

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

    # -- cursor tokens (ido-986.14.11) -------------------------------------

    def cursor_tag(
        self, scope: RuntimeHandleScope, *, alias: str, query_scope: str
    ) -> str:
        """The short tag that stands for this query scope on this handle.

        The base traversal has no tag at all (``O7/p2``), because that is the
        one the agent pages most and the one it has to type. A filtered
        traversal gets ``f1``, ``f2``, ... in the order the filters were first
        seen on this handle in this turn (``O7/f1p2``). The tag is an index into
        this table, never a hash of the literal: the token has to stay short,
        and a filter is identified by the row, not by the string.
        """
        if not query_scope:
            return ""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT tag FROM result_handle_cursor_tags
                WHERE scope_id = ? AND alias = ? AND query_scope = ?
                """,
                (scope.scope_id, alias, query_scope),
            ).fetchone()
            if row is not None:
                conn.commit()
                return str(row["tag"])
            used = conn.execute(
                """
                SELECT COUNT(*) AS used FROM result_handle_cursor_tags
                WHERE scope_id = ? AND alias = ?
                """,
                (scope.scope_id, alias),
            ).fetchone()
            tag = "f%d" % (int(used["used"] or 0) + 1)
            conn.execute(
                """
                INSERT INTO result_handle_cursor_tags (
                    scope_id, alias, query_scope, tag, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias, query_scope) DO NOTHING
                """,
                (scope.scope_id, alias, query_scope, tag, _now()),
            )
            conn.commit()
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT tag FROM result_handle_cursor_tags
                WHERE scope_id = ? AND alias = ? AND query_scope = ?
                """,
                (scope.scope_id, alias, query_scope),
            ).fetchone()
        return "" if row is None else str(row["tag"])

    def issue_cursor(
        self,
        scope: RuntimeHandleScope,
        *,
        alias: str,
        tag: str,
        query_scope: str,
        position: int,
        descriptor_sha256: str,
    ) -> int:
        """The page ordinal for this resumption point, allocated once.

        Idempotent by position: the same offset of the same traversal is always
        the same ordinal, so a page re-rendered or a cursor re-issued prints the
        token the agent already has. Ordinals only ever go up, so a token that
        was printed keeps meaning what it meant.
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT page FROM result_handle_cursors
                WHERE scope_id = ? AND alias = ? AND tag = ? AND position = ?
                  AND descriptor_sha256 = ?
                """,
                (scope.scope_id, alias, tag, int(position),
                 str(descriptor_sha256)),
            ).fetchone()
            if row is not None:
                conn.commit()
                return int(row["page"])
            top = conn.execute(
                """
                SELECT MAX(page) AS top FROM result_handle_cursors
                WHERE scope_id = ? AND alias = ? AND tag = ?
                """,
                (scope.scope_id, alias, tag),
            ).fetchone()
            page = int(top["top"] or (FIRST_CURSOR_PAGE - 1)) + 1
            conn.execute(
                """
                INSERT INTO result_handle_cursors (
                    scope_id, alias, tag, page, query_scope, position,
                    descriptor_sha256, issued_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias, tag, page) DO NOTHING
                """,
                (scope.scope_id, alias, tag, page, query_scope, int(position),
                 str(descriptor_sha256), _now()),
            )
            conn.commit()
        return page

    def get_cursor(
        self, scope: RuntimeHandleScope, *, alias: str, tag: str, page: int
    ) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT * FROM result_handle_cursors
                WHERE scope_id = ? AND alias = ? AND tag = ? AND page = ?
                """,
                (scope.scope_id, alias, tag, int(page)),
            ).fetchone()
        if row is None:
            return None
        return {
            "alias": str(row["alias"]),
            "tag": str(row["tag"]),
            "page": int(row["page"]),
            "query_scope": str(row["query_scope"]),
            "position": int(row["position"]),
            "descriptor_sha256": str(row["descriptor_sha256"]),
            "issued_at": str(row["issued_at"]),
        }

    def list_cursors(
        self, scope: RuntimeHandleScope, *, alias: str
    ) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM result_handle_cursors
                WHERE scope_id = ? AND alias = ?
                ORDER BY tag, page
                """,
                (scope.scope_id, alias),
            ).fetchall()
        return [
            {
                "alias": str(row["alias"]),
                "tag": str(row["tag"]),
                "page": int(row["page"]),
                "query_scope": str(row["query_scope"]),
                "position": int(row["position"]),
                "descriptor_sha256": str(row["descriptor_sha256"]),
            }
            for row in rows
        ]


class ReadOnlyResultHandleStore(ResultHandleStore):
    """Read-only view of an existing handle store. Never creates, migrates, or
    writes the file — a store that has been archived is evidence, and reading
    evidence must not change it. Construction raises when the file is absent or
    unopenable (``sqlite3.OperationalError``); callers degrade gracefully.

    Same shape, deliberately, as ``ReadOnlyObservabilityStore`` in
    ``fastworkflow/observability/store.py``: an ``__init__`` that skips the
    parent's create/migrate block entirely, and a ``_connect`` override that
    goes through ``file:<path>?mode=ro``. Inherited writers are not overridden
    — SQLite refuses them by itself with "attempt to write a readonly
    database", which is the loud failure and needs no help from here.

    WHY THIS EXISTS AT ALL (fix-iq53.2.10, F5b). ``ResultHandleStore.__init__``
    runs five ``CREATE TABLE IF NOT EXISTS`` statements over a read-write
    connection, so merely CONSTRUCTING one over a frozen artifact rewrites it:
    measured on a copy, a 135 168-byte store became 192 512 bytes with five
    tables added and a different sha256. "Additive" describes the SCHEMA delta,
    not the FILE delta. A read-only open is the mechanism that actually holds;
    new table names are a second line, not the first.

    WHY THERE IS NO ``immutable`` PARAMETER, and why ``_connect`` is a seam
    instead. A plain ``mode=ro`` connection to a WAL-mode database STILL
    CREATES an ``-shm`` sidecar beside it (and a ``-wal``, when none is there);
    that is how 17 sidecars once appeared under a frozen evidence root, an
    incident and not a precaution. ``immutable=1`` suppresses both, but it also
    makes SQLite ignore the WAL, so against a store with an uncheckpointed WAL
    it silently reads a TRUNCATED database. So the two options are each wrong
    for some file, and which one is wrong depends on whether that path is
    frozen evidence or a directory being written while it is read. The
    framework cannot know that, and should not learn it. The MECHANISM lives
    here; the per-file JUDGMENT lives with the caller, which expresses it by
    overriding ``_connect`` — the way IDO's evaluation layer already subclasses
    ``ReadOnlyObservabilityStore`` to route it through its own sidecar-safe
    policy. This is the owner's ruling on the question Revision 4 §9.3 left
    open, and it is none of that section's three options: no new framework
    parameter is added, because a parameter is the overreach that got two
    earlier revisions rejected.
    """

    def __init__(self, db_path: str) -> None:
        # Normalized like the parent so ``db_path`` means the same thing on
        # both classes. No ``makedirs``: the parent creates the containing
        # directory, and doing that beside an archive would be a write to the
        # very tree this class exists to leave alone.
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        # Prove the file is there and is a database now, rather than on the
        # first query, and close again immediately: an open connection is the
        # one thing that could still take a lock on it.
        conn = self._connect()
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn
