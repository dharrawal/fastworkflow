"""Turn-scoped SQLite archive for persist-before-label offloads."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional


class PersistenceError(RuntimeError):
    """Handle text and digest disagree, or an alias collides."""


@dataclass(frozen=True)
class RuntimeHandleScope:
    store_identity: str
    channel_id: str
    experiment_id: str
    task_id: str
    attempt: int
    turn_key: str

    @property
    def scope_id(self) -> str:
        encoded = json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class RuntimeHandleArchive:
    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_offload_handles (
                    scope_id TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    offload_order INTEGER NOT NULL,
                    command_name TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    text_utf8 BLOB NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    persisted_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS observation_offload_handles_order
                ON observation_offload_handles(scope_id, offload_order)
                """
            )
            # ido-8ps.15. The context clause an observation was produced in,
            # beside the text rather than inside it. A SEPARATE table, not a
            # column on observation_offload_handles: the archived response and
            # its text_sha256 are the evidence record, they are compared across
            # experiments, and metadata that reads as part of the text would
            # change what a digest means. Same (scope_id, alias) key, so a row
            # here is exactly "what the alias line printed above that text".
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_offload_context (
                    scope_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    context_clause TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def persist(
        self,
        scope: RuntimeHandleScope,
        *,
        alias: str,
        offload_order: int,
        command_name: str,
        step_index: int,
        text: str,
        text_sha256: str,
        context: Optional[str] = None,
    ) -> None:
        """Store one observation, and optionally the context it was produced in.

        ``context`` is the ``ido-8ps.13`` clause exactly as the alias line
        printed it. ``None`` means "no clause was captured for this alias" and
        records nothing; ``""`` means "this ran at the root context", which is a
        fact and is stored as one. The clause never touches ``text`` or
        ``text_sha256``: an observation archived with a clause is byte-identical
        to the same observation archived before clauses existed.
        """
        payload = text.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != text_sha256:
            raise PersistenceError("runtime handle digest does not match its text")
        scope_json = json.dumps(
            asdict(scope), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO observation_offload_handles (
                    scope_id, scope_json, alias, offload_order, command_name,
                    step_index, text_utf8, text_sha256, persisted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias) DO NOTHING
                """,
                (
                    scope.scope_id,
                    scope_json,
                    alias,
                    offload_order,
                    command_name,
                    step_index,
                    payload,
                    text_sha256,
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            if context is not None:
                conn.execute(
                    """
                    INSERT INTO observation_offload_context (
                        scope_id, alias, context_clause, recorded_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(scope_id, alias) DO NOTHING
                    """,
                    (
                        scope.scope_id,
                        alias,
                        str(context),
                        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ),
                )
            conn.commit()
        stored = self.get(scope, alias)
        if stored is None or stored["text_sha256"] != text_sha256:
            raise PersistenceError(
                "runtime handle alias collides with different text in this turn"
            )

    def record_context(self, scope: RuntimeHandleScope, alias: str,
                       context: str) -> None:
        """File the clause for an alias whose text is already stored.

        Insert-or-nothing, like the handle row: an alias is archived once in a
        turn, so the first clause recorded for it is the one the reader saw.
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO observation_offload_context (
                    scope_id, alias, context_clause, recorded_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(scope_id, alias) DO NOTHING
                """,
                (scope.scope_id, alias, str(context),
                 datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            )
            conn.commit()

    def context_clause(self, scope: RuntimeHandleScope,
                       alias: str) -> Optional[str]:
        """The recorded clause, ``""`` at the root, ``None`` if unrecorded."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT context_clause FROM observation_offload_context
                WHERE scope_id = ? AND alias = ?
                """,
                (scope.scope_id, alias),
            ).fetchone()
        return None if row is None else str(row["context_clause"])

    def context_clauses(self, scope: RuntimeHandleScope) -> dict[str, str]:
        """Every recorded clause in this scope, by alias."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT alias, context_clause FROM observation_offload_context
                WHERE scope_id = ?
                """,
                (scope.scope_id,),
            ).fetchall()
        return {str(row["alias"]): str(row["context_clause"]) for row in rows}

    def get(self, scope: RuntimeHandleScope, alias: str) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT h.alias, h.offload_order, h.command_name, h.step_index,
                       h.text_utf8, h.text_sha256, c.context_clause
                FROM observation_offload_handles AS h
                LEFT JOIN observation_offload_context AS c
                  ON c.scope_id = h.scope_id AND c.alias = h.alias
                WHERE h.scope_id = ? AND h.alias = ?
                """,
                (scope.scope_id, alias),
            ).fetchone()
        return None if row is None else self._decode_row(row)

    def list(self, scope: RuntimeHandleScope, alias: str = "") -> list[dict[str, Any]]:
        query = """
            SELECT h.alias, h.offload_order, h.command_name, h.step_index,
                   h.text_utf8, h.text_sha256, c.context_clause
            FROM observation_offload_handles AS h
            LEFT JOIN observation_offload_context AS c
              ON c.scope_id = h.scope_id AND c.alias = h.alias
            WHERE h.scope_id = ?
        """
        params: list[Any] = [scope.scope_id]
        if alias:
            query += " AND h.alias = ?"
            params.append(alias)
        query += " ORDER BY h.offload_order, h.alias"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._decode_row(row) for row in rows]

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        payload = bytes(row["text_utf8"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row["text_sha256"]:
            raise PersistenceError("runtime archive text failed digest verification")
        clause = row["context_clause"] if "context_clause" in row.keys() else None
        return {
            "alias": str(row["alias"]),
            "offload_order": int(row["offload_order"]),
            "command": str(row["command_name"]),
            "step_index": int(row["step_index"]),
            "text": payload.decode("utf-8"),
            "text_sha256": digest,
            # ido-8ps.15: metadata beside the text. ``None`` when no clause was
            # recorded for this alias, ``""`` when it ran at the root context.
            "context": None if clause is None else str(clause),
        }
