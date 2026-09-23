"""Turn-scoped SQLite archive for persist-before-label offloads.

Deletion lives next door, in ``observation_offloading.erasure``: this
module writes a turn's response bytes and never removes them, and that
module owns erasure and retention for every scope-keyed table in the file.
The two halves meet at ``scope_json``, which is why this module stores the
whole scope and not only its digest: a row must be able to say which channel
it came from, and whether it belongs to an experiment run, long after the
process that wrote it is gone.

Redaction lives next door as well, in the sense that matters: the bytes a
SEALED row holds are the bytes ``observability.store`` would have stored for
the same text, because they are produced by calling into that module's own
scrub-and-capture pipeline rather than by a second one written here.
The toggle, its default and the full policy are stated in
``observation_offloading.erasure``'s docstring, beside the retention policy,
because a reader deciding what this file may keep needs both at once. The
mechanism is below, in ``redaction_mode`` and ``capture_record_for``.

WHEN that happens is a deliberate choice. Redaction is not a write-time
transform: ``persist`` stores what the command returned, VERBATIM, and the row
is SEALED into its redacted form when the turn that produced it is genuinely
over. Nothing is redacted while a turn is in flight, and in flight means the
whole life of the turn -- an ask_user wait and any serialize/deserialize round
trip included -- so every read an agent can make during its own turn returns
raw: the live trajectory, ``search_memory``, rehydration, and those same reads
after a resume in a fresh process. The trade is raw bytes on disk for the
duration of a turn, taken because redaction mid-turn would change what the
agent reads back and therefore what it answers.

Two consequences live in this file. ``seal_scope`` is the completion step, and
it is only ever reached through the two places that already know a turn is over
(``StructuredContinuationReAct.bind_scope`` and
``WorkflowExecutionContext._reclaim_offloading_scope``). ``sweep_unsealed`` is
the backstop for a process that died mid-turn, so the raw-on-disk window is
bounded rather than open-ended.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

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


# ---------------------------------------------------------------------------
# Seal timing (ido-6sc)
# ---------------------------------------------------------------------------

#: Raw bytes are on disk and the seal is OWED. The state every row is written
#: in while its turn is in flight, under redaction ``on``.
SEAL_PENDING = "pending"
#: ``capture_record_for`` has run over this row's bytes and rewritten them.
SEAL_SEALED = "sealed"
#: Redaction was ``off`` at the write, so nothing is owed and the verbatim
#: bytes are the policy rather than a window. Distinct from ``sealed`` on
#: purpose: a reader must be able to tell "the pipeline ran and found nothing"
#: from "the pipeline was never going to run".
SEAL_NOT_REQUIRED = "not_required"
#: No seal row at all: written before ido-6sc, by code that redacted at write
#: time. Nothing is owed, and nothing here claims to know which it is.
SEAL_UNKNOWN = "unknown"
_SEAL_STATES = (SEAL_PENDING, SEAL_SEALED, SEAL_NOT_REQUIRED, SEAL_UNKNOWN)

#: How long after a row was opened raw the crash sweep may seal it without the
#: turn ever having said it was over. It is the bound on the owner's accepted
#: risk: a process that dies mid-turn leaves raw bytes for at most this long
#: after the write, once any process opens the sidecar again.
SEAL_GRACE_ENV = "FW_OFFLOAD_SEAL_GRACE_SECONDS"
#: A day. Long enough that a user who walks away from an ask_user question and
#: comes back after lunch still resumes into raw evidence, short enough that a
#: crashed turn's bytes are not a permanent exposure. Spelled as a constant
#: because the two mistakes are not symmetric: too short seals a live turn and
#: costs answer quality, which is the thing the owner is protecting.
DEFAULT_SEAL_GRACE_SECONDS = 86_400
#: Spellings that turn the sweep OFF. An operator who wants raw evidence kept
#: until a turn says it is over, and nothing else, says so here.
_SWEEP_DISABLED = ("off", "never", "disabled", "no", "none")

_warned_grace: set[str] = set()

#: This process's identity, for the one question the sweep has to answer: is
#: this pending row MINE. A pid alone is not enough -- pids are reused, and a
#: sidecar outlives the process that wrote it -- so a random token per process
#: is appended. It is written into the seal row and compared, never parsed.
_OWNER_ID = f"pid-{os.getpid()}-{uuid.uuid4().hex[:12]}"


def owner_id() -> str:
    """The token this process stamps on the rows it opens raw."""
    return _OWNER_ID


def seal_grace_seconds(override: Optional[str] = None) -> Optional[int]:
    """The sweep's horizon in seconds, or ``None`` when the sweep is OFF.

    An unrecognised value warns once and falls back to the default, by the same
    rule as ``redaction_mode`` and ``erasure.preservation_mode``: a typo must
    not quietly leave a crashed turn's credentials on disk forever, and it must
    not quietly seal a live turn either.
    """
    raw = override if override is not None else os.environ.get(SEAL_GRACE_ENV, "")
    value = str(raw or "").strip().lower()
    if not value:
        return DEFAULT_SEAL_GRACE_SECONDS
    if value in _SWEEP_DISABLED:
        return None
    try:
        seconds = int(value)
    except ValueError:
        seconds = -1
    if seconds >= 0:
        return seconds
    if value not in _warned_grace:
        _warned_grace.add(value)
        logger.warning(
            "ignoring %s=%s: expected a non-negative integer of seconds or one "
            "of %s; using %d",
            SEAL_GRACE_ENV, raw, ", ".join(_SWEEP_DISABLED),
            DEFAULT_SEAL_GRACE_SECONDS,
        )
    return DEFAULT_SEAL_GRACE_SECONDS


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
            # (ido-6sc) Whether this row's bytes are still the RAW ones the
            # command returned, and who left them that way. The owner's
            # decision moved redaction off the write path and onto turn
            # COMPLETION, so an archived observation now has two honest states
            # and a reader must be able to tell them apart -- an unsealed row
            # reproduces the agent's reading exactly and is a live exposure; a
            # sealed one is neither.
            #
            # Its own table, additive and created on open, for the third time
            # on this branch and for the same reason: an existing sidecar gains
            # it the first time new code opens the file and no migration, ALTER
            # or script ever runs against a store holding real evidence. A row
            # written before this table existed has no entry, reads back as
            # ``unknown``, and is never swept -- the code that wrote it redacted
            # at write time, so it owes nothing and guessing otherwise would
            # rewrite bytes that are already final.
            #
            # ``owner_id`` is what makes the crash sweep safe: a pending row
            # stamped by a process that is not this one, and older than the
            # grace horizon, belongs to a turn that will never say it is over.
            # A row this process opened is never swept by this process, so the
            # sweep cannot reach a turn that is still running.
            #
            # ``opened_at`` is deliberately the FIRST ``_at`` column here, so
            # ``erasure.evidence_tables`` dates this table by the moment the
            # row was written rather than by the moment it was sealed, and a
            # scope's retention horizon stays the moment its turn began.
            #
            # No raw digest is stored. While the row is pending its own
            # ``text_sha256`` already covers the raw bytes sitting beside it;
            # once sealed, a digest of the unredacted text would be exactly the
            # confirmation oracle ``persist`` refused to write in ido-zlm.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observation_seal_state (
                    scope_id TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    seal_state TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    sealed_at TEXT NOT NULL,
                    PRIMARY KEY (scope_id, alias)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS observation_seal_state_pending
                ON observation_seal_state(seal_state, opened_at)
                """
            )
        # (ido-6sc) Opening the sidecar is the crash-recovery trigger: see
        # ``sweep_unsealed``. It must never fail the open, for the same reason
        # ``UnavailableHandleArchive`` exists -- a turn does not stop because
        # an evidence optimisation could not tidy up after a previous one.
        #: What the sweep on THIS open sealed, so a caller that opened the file
        #: in order to sweep it (``erasure.prune``) can report the whole count
        #: rather than nothing -- the open already did the work.
        self.open_sweep: dict[str, Any] = {"sealed": 0, "redacted": 0,
                                           "failed": 0, "aliases": [],
                                           "errors": [], "swept": False}
        try:
            self.open_sweep = self.sweep_unsealed()
        except Exception:  # noqa: BLE001
            logger.debug(
                "could not sweep unsealed observations in %s", self.db_path,
                exc_info=True,
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
        """Store one observation VERBATIM and return the row as it now holds it.

        ``text``/``text_sha256`` are checked against each other first and
        always: that pair is the caller's integrity claim about what the command
        returned.

        What is stored is that same text, byte for byte, whatever the redaction
        toggle says. The turn that produced it is in flight, and
        nothing is redacted while a turn is in flight -- so the row's
        ``text_sha256`` is the caller's RAW digest, ``_decode_row`` verifies a
        read against the raw bytes beside it, and the hot cache a caller fills
        from the returned row agrees with the agent's own prompt. With redaction
        ``on`` the row is also marked ``pending``: the seal is OWED, and
        ``seal_scope`` pays it when the turn is over. With redaction ``off`` it
        is marked ``not_required``, which is that toggle's whole meaning.

        Idempotence survives the seal. The insert is insert-or-nothing, so a
        re-persist of the same alias is a readback -- and when the row it reads
        back has already been sealed, its digest covers the SEALED bytes and no
        longer matches the caller's raw digest. That is not a collision, and
        ``_agrees_after_a_seal`` re-derives the answer rather than patching it:
        it seals the candidate text the same way the row was sealed and
        compares. Nothing raw is persisted to make that comparison possible, so
        the file never becomes an oracle that confirms unredacted text.
        """
        payload = text.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != text_sha256:
            raise PersistenceError("runtime handle digest does not match its text")
        # The policy is resolved here, at the write, because the row has to be
        # able to say which rules it is OWED while it is still pending. The
        # seal re-records whichever rules actually ran.
        prospective = capture_record_for(text)[1]
        pending = prospective["redaction"] == REDACTION_ON
        scope_json = json.dumps(
            asdict(scope), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        recorded_at = _utc_now()
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
                    prospective["capture_policy_version"],
                    prospective["capture_profile"],
                    prospective["redaction"],
                    # Nothing has been redacted yet, and the row must not claim
                    # it has. The seal sets this to what actually happened.
                    0,
                    int(prospective["raw_utf8_bytes"]),
                    recorded_at,
                ),
            )
            # And DO NOTHING a third time, for the reason above squared: the
            # seal state describes the bytes in the file, so a re-persist that
            # kept an existing row must not reopen a sealed one as pending.
            conn.execute(
                """
                INSERT INTO observation_seal_state (
                    scope_id, scope_json, alias, seal_state, owner_id,
                    opened_at, sealed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, alias) DO NOTHING
                """,
                (
                    scope.scope_id,
                    scope_json,
                    alias,
                    SEAL_PENDING if pending else SEAL_NOT_REQUIRED,
                    _OWNER_ID,
                    recorded_at,
                    "" if pending else recorded_at,
                ),
            )
            conn.commit()
        stored = self.get(scope, alias)
        if stored is None:
            raise PersistenceError(
                "runtime handle alias collides with different text in this turn"
            )
        if stored["text_sha256"] != text_sha256 and not self._agrees_after_a_seal(
            scope, alias, text, stored["text_sha256"]
        ):
            raise PersistenceError(
                "runtime handle alias collides with different text in this turn"
            )
        return stored

    def _agrees_after_a_seal(
        self, scope: RuntimeHandleScope, alias: str, text: str, stored_sha256: str
    ) -> bool:
        """Whether *text* is what the SEALED row at *alias* was sealed from.

        The one case where a stored digest may legitimately differ from the
        caller's raw digest. Asked only when the digests already
        disagree, and answered by re-deriving: seal the candidate the way the
        row was sealed, and compare. A real collision -- two different
        observations under one alias -- still fails, because two different
        texts do not seal to the same bytes unless the redaction removed the
        only thing that told them apart, in which case the archive genuinely
        cannot distinguish them and refusing would be a false alarm about
        evidence it no longer holds.
        """
        if self.seal_state(scope, alias) != SEAL_SEALED:
            return False
        sealed_text, _ = capture_record_for(text)
        digest = hashlib.sha256(sealed_text.encode("utf-8")).hexdigest()
        return digest == stored_sha256

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
        answer and a later, better-informed writer may replace it. The empty
        string is a real value -- "this ran at the workflow root" -- and is
        stored as one; absence of the row is the only thing that means "no
        subject was recorded".
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

        ``seal_state`` is the field that makes the other five
        readable now that redaction happens at turn completion. ``pending``
        means the bytes are still the raw ones and ``redacted`` is ``False``
        because nothing has run yet, not because there was nothing to find;
        ``sealed`` means the pipeline has run and ``redacted`` says whether it
        found anything; ``not_required`` means the toggle was off; ``unknown``
        means the row predates the seal ledger. Those are three different rows
        and they used to be indistinguishable.
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
            seal = self._seal_row(conn, scope.scope_id, str(alias))
        if row is None:
            return UNKNOWN_CAPTURE_RECORD
        return {
            "capture_policy_version": str(row["capture_policy_version"]),
            "capture_profile": str(row["capture_profile"]),
            "redaction": str(row["redaction"]),
            "redacted": bool(row["redacted"]),
            "raw_utf8_bytes": int(row["raw_utf8_bytes"]),
            "recorded_at": str(row["recorded_at"]),
            "seal_state": SEAL_UNKNOWN if seal is None else str(seal["seal_state"]),
            "sealed_at": "" if seal is None else str(seal["sealed_at"]),
        }

    # -- seal timing (ido-6sc) ---------------------------------------------

    @staticmethod
    def _seal_row(
        conn: sqlite3.Connection, scope_id: str, alias: str
    ) -> Optional[sqlite3.Row]:
        try:
            return conn.execute(
                """
                SELECT seal_state, owner_id, opened_at, sealed_at
                FROM observation_seal_state
                WHERE scope_id = ? AND alias = ?
                """,
                (scope_id, alias),
            ).fetchone()
        except sqlite3.OperationalError:
            # A sidecar opened read-only, or one whose schema this process did
            # not create. An absent ledger is an UNKNOWN row, never a pending
            # one, so nothing is swept on the strength of a missing table.
            return None

    def seal_state(self, scope: RuntimeHandleScope, alias: str) -> str:
        """``pending``, ``sealed``, ``not_required`` or ``unknown``."""
        with closing(self._connect()) as conn:
            row = self._seal_row(conn, scope.scope_id, str(alias))
        return SEAL_UNKNOWN if row is None else str(row["seal_state"])

    def pending_aliases(self, scope: RuntimeHandleScope) -> list[str]:
        """Every alias of one scope whose bytes are still raw, in order."""
        with closing(self._connect()) as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT alias FROM observation_seal_state
                    WHERE scope_id = ? AND seal_state = ?
                    ORDER BY alias
                    """,
                    (scope.scope_id, SEAL_PENDING),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [str(row["alias"]) for row in rows]

    def seal_scope(
        self, scope: RuntimeHandleScope, *, mode: Optional[str] = None
    ) -> dict[str, Any]:
        """Seal every pending observation of one FINISHED turn.

        The completion step for deferred redaction. It is reached only
        through the two callers that already know a turn is over -- see
        ``state.seal_scope`` -- so a suspended turn, whose whole point is that
        it is not over, is never sealed.

        Per alias, in one transaction: the raw bytes are read back and verified
        against their own digest, run through ``capture_record_for``, and
        written back with a digest that covers the bytes now in the file. The
        fidelity record is rewritten to name the policy that actually ran and
        whether it found anything, and the seal row moves to ``sealed``.

        Idempotent by state, not by digest: only ``pending`` rows are touched,
        so sealing a scope twice is a no-op and sealing one whose turn ran
        under redaction ``off`` is too. A per-alias failure is recorded and the
        rest of the scope is still sealed -- one unreadable row must not leave
        the whole turn raw.
        """
        return self._seal_aliases(
            [(scope.scope_id, alias) for alias in self.pending_aliases(scope)],
            mode=mode,
        )

    def _seal_aliases(
        self, keys: Iterable[tuple[str, str]], *, mode: Optional[str] = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"sealed": 0, "redacted": 0, "failed": 0,
                                  "aliases": [], "errors": []}
        for scope_id, alias in keys:
            try:
                changed = self._seal_one(scope_id, str(alias), mode=mode)
            except Exception as error:  # noqa: BLE001 - one row must not stop the rest
                result["failed"] += 1
                result["errors"].append(f"{alias}: {type(error).__name__}")
                logger.warning(
                    "could not seal observation %s in %s: %s",
                    alias, self.db_path, error,
                )
                continue
            if changed is None:
                continue
            result["sealed"] += 1
            result["aliases"].append(str(alias))
            if changed:
                result["redacted"] += 1
        return result

    def _seal_one(
        self, scope_id: str, alias: str, *, mode: Optional[str] = None
    ) -> Optional[bool]:
        """Seal one pending row; ``None`` if it was not pending after all.

        The read, the redaction and the two writes are one ``BEGIN IMMEDIATE``
        transaction, and the ``seal_state = 'pending'`` predicate is re-checked
        inside it, so two processes sweeping the same file cannot both seal the
        same row and cannot seal a row a turn-completion call just sealed.
        """
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT h.text_utf8 AS text_utf8, h.text_sha256 AS text_sha256
                    FROM observation_seal_state AS s
                    JOIN observation_offload_handles AS h
                      ON h.scope_id = s.scope_id AND h.alias = s.alias
                    WHERE s.scope_id = ? AND s.alias = ? AND s.seal_state = ?
                    """,
                    (scope_id, alias, SEAL_PENDING),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return None
                raw_payload = bytes(row["text_utf8"])
                if hashlib.sha256(raw_payload).hexdigest() != row["text_sha256"]:
                    raise PersistenceError(
                        "runtime archive text failed digest verification"
                    )
                raw_text = raw_payload.decode("utf-8")
                sealed_text, capture = capture_record_for(raw_text, mode=mode)
                sealed_payload = sealed_text.encode("utf-8")
                sealed_sha256 = hashlib.sha256(sealed_payload).hexdigest()
                sealed_at = _utc_now()
                conn.execute(
                    """
                    UPDATE observation_offload_handles
                    SET text_utf8 = ?, text_sha256 = ?
                    WHERE scope_id = ? AND alias = ?
                    """,
                    (sealed_payload, sealed_sha256, scope_id, alias),
                )
                conn.execute(
                    """
                    UPDATE observation_capture_policy
                    SET capture_policy_version = ?, capture_profile = ?,
                        redaction = ?, redacted = ?
                    WHERE scope_id = ? AND alias = ?
                    """,
                    (
                        capture["capture_policy_version"],
                        capture["capture_profile"],
                        capture["redaction"],
                        1 if capture["redacted"] else 0,
                        scope_id,
                        alias,
                    ),
                )
                conn.execute(
                    """
                    UPDATE observation_seal_state
                    SET seal_state = ?, sealed_at = ?
                    WHERE scope_id = ? AND alias = ?
                    """,
                    (SEAL_SEALED, sealed_at, scope_id, alias),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return bool(capture["redacted"])

    def sweep_unsealed(
        self,
        *,
        grace_seconds: Optional[str] = None,
        now: Optional[datetime] = None,
        mode: Optional[str] = None,
    ) -> dict[str, Any]:
        """Seal rows whose turn never said it was over.

        The bound on how long raw bytes can sit on disk. A turn seals its own evidence
        at completion, but a process that is killed between the write and the
        completion leaves raw bytes behind with nobody left to seal them, and
        "until the next deletion request" is not a bound.

        TRIGGERED from two places, both of them moments when somebody is
        already touching the file. Opening the sidecar, which is what the next
        process to run a turn against this store does, and which makes recovery
        automatic rather than operational. And ``erasure.prune``, the retention
        job, which is the only thing that runs against a store whose processes
        are all long gone.

        WHAT IT LOOKS FOR: a row still ``pending`` whose ``owner_id`` is not
        this process's, and whose ``opened_at`` is older than the grace
        horizon. Both halves matter. The owner check is what makes the sweep
        unable to reach a turn THIS process is running, which is the case
        deferred redaction is actually about. The horizon is what makes it unable
        to reach a turn ANOTHER live process is running, since the sweep cannot
        ask another process whether its turn is still in flight -- only how
        long ago it started.

        WHAT IT CANNOT COVER, stated here because a bound nobody can see is not
        a bound. A sidecar that no process ever opens again and that retention
        never visits keeps its raw bytes. The horizon itself is an exposure
        window, by construction. A turn that outlives the horizon -- including
        an ask_user question nobody answers for a day -- is swept while it is
        arguably still in flight, so a very late resume reads sealed evidence;
        that is the deliberate trade, and it is why the horizon is long and
        configurable. And a seal rewrites a row in place, which does not scrub
        the freed page space the old bytes occupied until something VACUUMs the
        file, which ``erasure`` does and this does not.
        """
        horizon_seconds = seal_grace_seconds(grace_seconds)
        if horizon_seconds is None:
            return {"sealed": 0, "redacted": 0, "failed": 0, "aliases": [],
                    "errors": [], "swept": False}
        moment = now or datetime.now(timezone.utc)
        horizon = (moment - timedelta(seconds=horizon_seconds)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with closing(self._connect()) as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT scope_id, alias FROM observation_seal_state
                    WHERE seal_state = ? AND owner_id <> ? AND opened_at < ?
                    ORDER BY opened_at, scope_id, alias
                    """,
                    (SEAL_PENDING, _OWNER_ID, horizon),
                ).fetchall()
            except sqlite3.OperationalError:
                return {"sealed": 0, "redacted": 0, "failed": 0, "aliases": [],
                        "errors": [], "swept": False}
        result = self._seal_aliases(
            [(str(row["scope_id"]), str(row["alias"])) for row in rows], mode=mode
        )
        result["swept"] = True
        result["horizon"] = horizon
        if result["sealed"]:
            logger.info(
                "sealed %d observation(s) left unsealed by a turn that never "
                "completed in %s", result["sealed"], self.db_path,
            )
        return result

    def compact(self) -> None:
        """VACUUM the file, so a sealed row's old bytes leave its free pages.

        A seal is an in-place UPDATE, and SQLite does not zero what a shorter
        blob stopped using. ``erasure`` already vacuums after it deletes; this
        is the same call, reachable from a sweep that sealed something.
        """
        with closing(self._connect()) as conn:
            conn.execute("VACUUM")

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
    Without this stand-in an INITIALISATION failure costs the whole turn instead,
    because it raises out of the agent's constructor and the persist-before-label
    recovery never gets to run.

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

    # Nothing was stored, so nothing is owed and nothing can be swept. The
    # seal surface answers that rather than raising, on this class's own rule:
    # every read says "nothing recorded here", and nothing claims a durability
    # this object cannot provide (ido-6sc).
    def seal_state(self, scope: RuntimeHandleScope, alias: str) -> str:
        return SEAL_UNKNOWN

    def pending_aliases(self, scope: RuntimeHandleScope) -> list[str]:
        return []

    def seal_scope(
        self, scope: RuntimeHandleScope, *, mode: Optional[str] = None
    ) -> dict[str, Any]:
        return {"sealed": 0, "redacted": 0, "failed": 0, "aliases": [],
                "errors": [], "unavailable": self.reason}

    def sweep_unsealed(self, **_: Any) -> dict[str, Any]:
        return {"sealed": 0, "redacted": 0, "failed": 0, "aliases": [],
                "errors": [], "swept": False, "unavailable": self.reason}

    def compact(self) -> None:
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
