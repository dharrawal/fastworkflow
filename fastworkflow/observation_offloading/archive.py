"""Turn-scoped SQLite archive for persist-before-label offloads.

Deletion lives next door, in ``observation_offloading.erasure``: this
module writes a turn's response bytes and never removes them, and that
module owns erasure and retention for every scope-keyed table in the file,
including the result-handle tables written beside this one. The two halves
meet at ``scope_json``, which is why this module stores the whole scope and
not only its digest: a row must be able to say which channel it came from,
and whether it belongs to an experiment run, long after the process that
wrote it is gone (ido-gls).

Redaction lives next door as well, in the sense that matters: the bytes
``persist`` stores are the bytes ``observability.store`` would have stored for
the same text, because they are produced by calling into that module's own
scrub-and-capture pipeline rather than by a second one written here (ido-zlm).
The toggle, its default and the full policy are stated in
``observation_offloading.erasure``'s docstring, beside the retention policy,
because a reader deciding what this file may keep needs both at once. The
mechanism is below, in ``redaction_mode`` and ``capture_record_for``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    """Handle text and digest disagree, or an alias collides."""


# ---------------------------------------------------------------------------
# Redaction policy (ido-zlm)
# ---------------------------------------------------------------------------

REDACTION_ENV = "FW_OFFLOAD_EVIDENCE_REDACTION"
#: Route stored response bytes through the credential and capture pipeline.
#: The default, and what an unconfigured deployment gets.
REDACTION_ON = "on"
#: Store exactly what the command returned. For development and optimisation,
#: where the archive's job is to reproduce what the agent actually read.
REDACTION_OFF = "off"
_REDACTION_MODES = (REDACTION_ON, REDACTION_OFF)
#: Spellings an operator is likely to reach for, folded onto the two modes.
#: Anything else warns and falls back to the default, by the same rule as
#: ``erasure.preservation_mode``: a typo must not quietly change policy.
_REDACTION_ALIASES = {
    "on": REDACTION_ON, "1": REDACTION_ON, "true": REDACTION_ON,
    "yes": REDACTION_ON, "enabled": REDACTION_ON,
    "off": REDACTION_OFF, "0": REDACTION_OFF, "false": REDACTION_OFF,
    "no": REDACTION_OFF, "disabled": REDACTION_OFF,
}

#: Recorded per row when no capture policy was consulted, so ``debug`` (a real
#: profile that happens to be inert) and "not asked" never read the same.
PROFILE_NOT_CONSULTED = ""
#: What ``capture_record`` answers for a row written before this table existed.
UNKNOWN_CAPTURE_RECORD = None

_warned_redaction: set[str] = set()


def redaction_mode(override: Optional[str] = None) -> str:
    """The active mode: the argument, else the environment, else the default.

    ON by default and on purpose. An unconfigured deployment gets the safe
    direction, because the cost of the wrong default here is an archive that
    reproduces the agent's reading less exactly -- recoverable by re-running --
    while the cost in the other direction is a credential written verbatim to
    disk, which no later configuration change undoes.
    """
    raw = override if override is not None else os.environ.get(REDACTION_ENV, "")
    value = str(raw or "").strip().lower()
    if not value:
        return REDACTION_ON
    resolved = _REDACTION_ALIASES.get(value)
    if resolved is not None:
        return resolved
    if value not in _warned_redaction:
        _warned_redaction.add(value)
        logger.warning(
            "ignoring %s=%s: expected one of %s; using %r",
            REDACTION_ENV, raw, ", ".join(_REDACTION_MODES), REDACTION_ON,
        )
    return REDACTION_ON


def redaction_enabled(override: Optional[str] = None) -> bool:
    return redaction_mode(override) == REDACTION_ON


def _observability_store() -> Any:
    """The module that owns the credential scrub and the capture policy.

    Imported on use, not at module scope, for the mirror image of the reason
    ``store._offload_erasure`` is: that module reaches into this package, and
    this package is imported from the ReAct agent it also reaches into.
    """
    from fastworkflow.observability import store as observability_store

    return observability_store


def capture_record_for(text: str, *, mode: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    """The bytes to store for *text*, and the record that describes their fidelity.

    The record is what makes an archive auditable about itself. It carries the
    capture-policy CONTRACT version in force at the write, the profile that was
    consulted (empty when redaction was off and none was), the toggle state, and
    -- the part a reader actually needs -- whether the stored bytes DIFFER from
    what the command returned. Version plus toggle says which rules applied;
    ``redacted`` says whether they had anything to act on, which is how a reader
    tells a row that was redacted from a row that never contained a secret.

    A failure inside the pipeline stores nothing rather than storing the raw
    text: evidence capture is an optimisation and the write path already knows
    how to keep an observation inline, but silently downgrading to verbatim
    would turn a broken dependency into a credential on disk.
    """
    active = redaction_mode(mode)
    observability_store = _observability_store()
    record: dict[str, Any] = {
        "capture_policy_version": str(observability_store.CAPTURE_POLICY_VERSION),
        "capture_profile": PROFILE_NOT_CONSULTED,
        "redaction": active,
        "redacted": False,
        "raw_utf8_bytes": len(text.encode("utf-8")),
    }
    if active == REDACTION_OFF:
        return text, record
    policy = observability_store.resolve_capture_policy()
    record["capture_profile"] = str(policy.profile)
    record["capture_policy_version"] = str(policy.policy_version)
    stored = observability_store.protect_offload_observation(text)
    stored = text if stored is None else str(stored)
    record["redacted"] = stored != text
    return stored, record


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
            # (ido-zlm) What produced the bytes of each archived observation:
            # the capture-policy contract version in force, the profile that
            # was consulted, the redaction toggle, and whether the stored bytes
            # actually differ from what the command returned. An archive that
            # cannot say this cannot be read as evidence -- a full-fidelity row
            # and a redacted one look alike, and so do a redacted row and one
            # that simply never contained a secret.
            #
            # Its own table rather than columns on the handle row, so that an
            # existing sidecar gains it the first time new code opens the file
            # and NO migration, ALTER or script ever runs against a store that
            # holds real evidence. A row written before this existed has no
            # entry here and reads back as UNKNOWN -- never as a guess that it
            # was full fidelity, which is the guess that would matter.
            #
            # ``scope_id``/``scope_json`` are not decoration: they are what
            # makes ``erasure.evidence_tables`` discover this table
            # structurally, so a channel's fidelity record is erased with the
            # channel and cannot outlive the evidence it describes.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_capture_policy (
                    scope_id TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    capture_policy_version TEXT NOT NULL,
                    capture_profile TEXT NOT NULL,
                    redaction TEXT NOT NULL,
                    redacted INTEGER NOT NULL,
                    raw_utf8_bytes INTEGER NOT NULL,
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
    ) -> dict[str, Any]:
        """Store one observation and return the row as the archive now holds it.

        ``text``/``text_sha256`` are checked against each other first and always:
        that pair is the caller's integrity claim about what the command
        returned, and it is verified against the RAW text whatever the redaction
        toggle says.

        What is then STORED may be shorter (ido-zlm). With redaction on, the
        bytes go through ``capture_record_for`` -- the observability store's own
        credential scrub and capture policy -- and ``text_sha256`` in the row
        covers the bytes the archive actually holds, because that column is what
        ``_decode_row`` verifies a read against and a digest that covers
        something other than the bytes beside it is a digest that fails every
        reader. The RAW digest keeps its meaning where its meaning is needed and
        is unchanged: ``compact`` computes it, hands it in here, and remembers it
        in ``state.mark_archived`` as the key that says this text is already
        written. It is deliberately not persisted beside the redacted bytes; a
        digest of unredacted text stored next to the redaction is a confirmation
        oracle for the credential the redaction just removed, which is
        `_protected_text`'s own reasoning applied one file over.

        The returned row is the archive's answer to "what did you keep?", which
        is what a caller should put in a hot cache so that the same alias reads
        the same way whichever tier serves it.
        """
        payload = text.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != text_sha256:
            raise PersistenceError("runtime handle digest does not match its text")
        stored_text, capture = capture_record_for(text)
        stored_payload = stored_text.encode("utf-8")
        stored_sha256 = hashlib.sha256(stored_payload).hexdigest()
        scope_json = json.dumps(
            asdict(scope), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
                    stored_payload,
                    stored_sha256,
                    recorded_at,
                ),
            )
            # DO NOTHING here too, and in the same transaction: the fidelity
            # record must describe the bytes that are actually in the file. If
            # the insert above kept an existing row, this one must keep the
            # record that describes it rather than overwrite it with a claim
            # about bytes that were not stored.
            conn.execute(
                """
                INSERT INTO observation_capture_policy (
                    scope_id, scope_json, alias, capture_policy_version,
                    capture_profile, redaction, redacted, raw_utf8_bytes,
                    recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias) DO NOTHING
                """,
                (
                    scope.scope_id,
                    scope_json,
                    alias,
                    capture["capture_policy_version"],
                    capture["capture_profile"],
                    capture["redaction"],
                    1 if capture["redacted"] else 0,
                    int(capture["raw_utf8_bytes"]),
                    recorded_at,
                ),
            )
            conn.commit()
        stored = self.get(scope, alias)
        if stored is None or stored["text_sha256"] != stored_sha256:
            raise PersistenceError(
                "runtime handle alias collides with different text in this turn"
            )
        return stored

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

    # -- capture fidelity (ido-zlm) ----------------------------------------

    def capture_record(
        self, scope: RuntimeHandleScope, alias: str
    ) -> Optional[dict[str, Any]]:
        """How this alias's stored bytes were produced, or ``None`` if UNKNOWN.

        ``None`` is what a row written before this table existed reads as, and
        it is the only honest answer for one: nothing in the file says whether
        those bytes are full fidelity, so nothing here claims they are.
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT capture_policy_version, capture_profile, redaction,
                       redacted, raw_utf8_bytes, recorded_at
                FROM observation_capture_policy
                WHERE scope_id = ? AND alias = ?
                """,
                (scope.scope_id, str(alias)),
            ).fetchone()
        if row is None:
            return UNKNOWN_CAPTURE_RECORD
        return {
            "capture_policy_version": str(row["capture_policy_version"]),
            "capture_profile": str(row["capture_profile"]),
            "redaction": str(row["redaction"]),
            "redacted": bool(row["redacted"]),
            "raw_utf8_bytes": int(row["raw_utf8_bytes"]),
            "recorded_at": str(row["recorded_at"]),
        }

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

    def persist(self, scope: RuntimeHandleScope, **_: Any) -> dict[str, Any]:
        raise PersistenceError(
            f"runtime handle archive unavailable at {self.db_path}: {self.reason}"
        )

    def capture_record(
        self, scope: RuntimeHandleScope, alias: str
    ) -> Optional[dict[str, Any]]:
        return None

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
