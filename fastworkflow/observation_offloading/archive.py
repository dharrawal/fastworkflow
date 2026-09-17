"""Turn-scoped SQLite archive for persist-before-label offloads.

Deletion lives next door, in ``observation_offloading.erasure``: this
module writes a turn's raw response bytes and never removes them, and that
module owns erasure and retention for every scope-keyed table in the file,
including the result-handle tables written beside this one. The two halves
meet at ``scope_json``, which is why this module stores the whole scope and
not only its digest: a row must be able to say which channel it came from,
and whether it belongs to an experiment run, long after the process that
wrote it is gone (ido-gls).
"""
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
    #: This one holds a file it can read and write. See
    #: ``UnavailableHandleArchive`` for the one that does not.
    available = True

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
            # (ido-dhw, F3) The SUBJECT an observation is evidence about: the
            # context clause ``CommandExecutor._remember_execute_context``
            # captured before the command ran, and that ``_stamp_page_clause``
            # may correct to the clause its declaring handle was declared for.
            # It lived only in ``state._context_clauses``, so a process that
            # restarted rehydrated every observation without its subject: the
            # handle line lost its clause, attribution could not say whose
            # evidence a listing was, and observation search was handed a
            # permission table with nothing saying whose permissions it held.
            #
            # Its own table rather than a column on the handle row, because a
            # clause is recorded at DISPATCH and the handle row is written when
            # the step COMPLETES -- and because an alias can carry a subject
            # with no archived observation at all (a step whose archive was
            # refused, a page alias stamped before its own observation exists).
            #
            # Additive and created on open, so an existing sidecar gains the
            # table the first time new code opens it and needs no migration. A
            # row written before this existed simply has no entry here, which
            # reads back as UNRECORDED -- never as a guessed subject.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_subjects (
                    scope_id TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    context_clause TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias)
                )
                """
            )
            # (ido-dhw, F3) The auto-navigation entry registry, durably. Rule 3
            # resolves an ``O`` handle the agent writes to the context instance
            # that handle denotes, and the registry holding that mapping was
            # process-local: after a restart the handle the same turn had
            # printed resolved to nothing. The contract stored here is exactly
            # what ``ContextEntry`` publishes -- the context, the entering
            # command, its parameter values, the alias, and which of those
            # parameters the entry contract declared REQUIRED (ido-nx6).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_context_entries (
                    scope_id TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    context TEXT NOT NULL,
                    command_name TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    required_parameters_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, sequence)
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
    ) -> None:
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
            conn.commit()
        stored = self.get(scope, alias)
        if stored is None or stored["text_sha256"] != text_sha256:
            raise PersistenceError(
                "runtime handle alias collides with different text in this turn"
            )

    def get(self, scope: RuntimeHandleScope, alias: str) -> Optional[dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT alias, offload_order, command_name, step_index,
                       text_utf8, text_sha256
                FROM observation_offload_handles
                WHERE scope_id = ? AND alias = ?
                """,
                (scope.scope_id, alias),
            ).fetchone()
        return None if row is None else self._decode_row(row)

    def list(self, scope: RuntimeHandleScope, alias: str = "") -> list[dict[str, Any]]:
        query = """
            SELECT alias, offload_order, command_name, step_index,
                   text_utf8, text_sha256
            FROM observation_offload_handles
            WHERE scope_id = ?
        """
        params: list[Any] = [scope.scope_id]
        if alias:
            query += " AND alias = ?"
            params.append(alias)
        query += " ORDER BY offload_order, alias"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._decode_row(row) for row in rows]

    # -- subject metadata (ido-dhw, F3) ------------------------------------

    def _scope_json(self, scope: RuntimeHandleScope) -> str:
        return json.dumps(
            asdict(scope), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )

    def put_subject(
        self, scope: RuntimeHandleScope, alias: str, context_clause: str
    ) -> None:
        """Record the subject *alias* is evidence about, durably.

        An UPSERT, not insert-or-nothing: the dispatch-time stamp is a first
        answer and ``result_handles._stamp_page_clause`` may replace it with the
        clause the page's DECLARING handle carries, which is the fact a reader
        of that page needs. The empty string is a real value -- "this ran at the
        workflow root" -- and is stored as one; absence of the row is the only
        thing that means "no subject was recorded".
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO observation_subjects (
                    scope_id, scope_json, alias, context_clause, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias) DO UPDATE SET
                    context_clause = excluded.context_clause,
                    recorded_at = excluded.recorded_at
                """,
                (
                    scope.scope_id,
                    self._scope_json(scope),
                    str(alias),
                    str(context_clause or ""),
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            conn.commit()

    def get_subject(self, scope: RuntimeHandleScope, alias: str) -> Optional[str]:
        """The recorded clause, ``""`` at the root, ``None`` when UNRECORDED.

        ``None`` is what a row written before this table existed reads as, and
        it is the same answer an alias nobody stamped gives. Neither is ever
        upgraded to a guess.
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT context_clause FROM observation_subjects "
                "WHERE scope_id = ? AND alias = ?",
                (scope.scope_id, str(alias)),
            ).fetchone()
        return None if row is None else str(row["context_clause"])

    def list_subjects(self, scope: RuntimeHandleScope) -> dict[str, str]:
        """Every recorded subject in this scope, ``alias -> clause``."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT alias, context_clause FROM observation_subjects "
                "WHERE scope_id = ? ORDER BY alias",
                (scope.scope_id,),
            ).fetchall()
        return {str(row["alias"]): str(row["context_clause"]) for row in rows}

    def forget_subject(self, scope: RuntimeHandleScope, alias: str) -> None:
        """Drop the recorded subject, so *alias* reads as UNRECORDED again.

        The durable half of ``state.forget_context_clause``: a dispatch-time
        stamp that turns out to be the wrong subject must not survive on disk
        after the process that corrected it is gone.
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM observation_subjects WHERE scope_id = ? AND alias = ?",
                (scope.scope_id, str(alias)),
            )
            conn.commit()

    # -- auto-navigation entries (ido-dhw, F3) -----------------------------

    def put_context_entry(
        self,
        scope: RuntimeHandleScope,
        *,
        sequence: int,
        context: str,
        command_name: str,
        parameters: dict[str, str],
        alias: str,
        required_parameters: tuple[str, ...],
    ) -> None:
        """Record one context instance this turn entered, and what entered it."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO observation_context_entries (
                    scope_id, scope_json, sequence, context, command_name,
                    parameters_json, alias, required_parameters_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, sequence) DO NOTHING
                """,
                (
                    scope.scope_id,
                    self._scope_json(scope),
                    int(sequence),
                    str(context),
                    str(command_name),
                    json.dumps(dict(parameters), ensure_ascii=False, sort_keys=True),
                    str(alias or ""),
                    json.dumps(list(required_parameters), ensure_ascii=False),
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            conn.commit()

    def list_context_entries(
        self, scope: "RuntimeHandleScope | str"
    ) -> list[dict[str, Any]]:
        """Every recorded entry of one turn scope, in the order it recorded them."""
        scope_id = scope if isinstance(scope, str) else scope.scope_id
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT sequence, context, command_name, parameters_json, alias,
                       required_parameters_json
                FROM observation_context_entries
                WHERE scope_id = ? ORDER BY sequence
                """,
                (scope_id,),
            ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "context": str(row["context"]),
                "command_name": str(row["command_name"]),
                "parameters": json.loads(row["parameters_json"] or "{}"),
                "alias": str(row["alias"]) or None,
                "required_parameters": tuple(
                    json.loads(row["required_parameters_json"] or "[]")
                ),
            }
            for row in rows
        ]

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        payload = bytes(row["text_utf8"])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row["text_sha256"]:
            raise PersistenceError("runtime archive text failed digest verification")
        return {
            "alias": str(row["alias"]),
            "offload_order": int(row["offload_order"]),
            "command": str(row["command_name"]),
            "step_index": int(row["step_index"]),
            "text": payload.decode("utf-8"),
            "text_sha256": digest,
        }


class UnavailableHandleArchive:
    """The archive this process could not open, inert and honest about it.

    Opening or creating the sidecar can fail for reasons that have nothing to do
    with the turn about to run: a read-only state root, a permission bit, a path
    that holds something which is not a database. Evidence storage is an
    availability optimisation, and the surrounding design already says what a
    storage failure costs -- ``archive_execute_observations`` and
    ``compact_trajectory`` record the refusal and leave the observation inline.
    Before ``ido-t5x`` an INITIALISATION failure cost the whole turn instead,
    because it raised out of the agent's constructor and the persist-before-label
    recovery never got to run.

    So the failure is degraded to the policy the writes already have, rather
    than to a second one. Every write refuses with the ``PersistenceError`` the
    write path expects, so the caller records ``archive_refused`` /
    ``offload_refused`` per alias and keeps the original text; every read answers
    "nothing stored here", which is the same answer as an alias that was never
    archived, so ``search_memory`` reports a miss instead of raising. Nothing is
    ever marked archived, so no part of the runtime claims durability this
    object cannot provide.

    ``db_path`` is the path that was WANTED, not a substitute: no evidence is
    silently redirected to another file, and the sidecar stays absent, which is
    exactly what ``observation_offloading.erasure`` reports empty for.
    """

    available = False

    def __init__(self, db_path: str, error: BaseException) -> None:
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self.error = error
        self.reason = f"{type(error).__name__}: {error}"

    def persist(self, scope: RuntimeHandleScope, **_: Any) -> None:
        raise PersistenceError(
            f"runtime handle archive unavailable at {self.db_path}: {self.reason}"
        )

    def get(self, scope: RuntimeHandleScope, alias: str) -> Optional[dict[str, Any]]:
        return None

    def list(self, scope: RuntimeHandleScope, alias: str = "") -> list[dict[str, Any]]:
        return []

    # The subject and navigation surface, inert for the same reason the rest of
    # this class is: nothing may claim a durability this object cannot provide,
    # and every read answers "nothing recorded here" -- which is exactly the
    # UNRECORDED answer readers already handle.
    def put_subject(
        self, scope: RuntimeHandleScope, alias: str, context_clause: str
    ) -> None:
        raise PersistenceError(
            f"runtime handle archive unavailable at {self.db_path}: {self.reason}"
        )

    def get_subject(self, scope: RuntimeHandleScope, alias: str) -> Optional[str]:
        return None

    def list_subjects(self, scope: RuntimeHandleScope) -> dict[str, str]:
        return {}

    def forget_subject(self, scope: RuntimeHandleScope, alias: str) -> None:
        return None

    def put_context_entry(self, scope: RuntimeHandleScope, **_: Any) -> None:
        raise PersistenceError(
            f"runtime handle archive unavailable at {self.db_path}: {self.reason}"
        )

    def list_context_entries(
        self, scope: "RuntimeHandleScope | str"
    ) -> list[dict[str, Any]]:
        return []
