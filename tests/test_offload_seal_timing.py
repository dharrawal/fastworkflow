"""Nothing is redacted while a turn is in flight; the seal is at the end.

Redaction decides WHETHER the evidence sidecar scrubs what it stored; the seal
decides WHEN. With redaction on, a turn's stored observations are written
VERBATIM and sealed into their redacted form only when the turn is genuinely
over. In flight means the whole life of the turn, explicitly including an
ask_user wait and any serialize/deserialize round trip, so every read an agent
can make during its own turn returns raw -- the live trajectory,
``search_memory``, rehydration, and those same reads after a resume in a fresh
process.

The cost of that design is raw bytes on disk for the duration of a turn. Two
things keep it bounded rather than open-ended, and both are under test here: a
completing turn seals its own evidence, and a turn whose process DIED is swept.

The claims are made in BYTES wherever a leak or a degradation is the thing
being denied, on ``test_offload_capture_redaction``'s rule: a row read back
through the archive's API proves what the API returns, and the file is what a
stolen disk is about. Every scan opens the database (and its journal, if any)
and searches the raw bytes.

Everything runs against databases and workflows created in this test's own
temporary directory. Nothing here reads or writes a store it did not create.
No model, no backend, no network.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dspy

import fastworkflow
from fastworkflow import tracing
from fastworkflow.answer_rehydration import archived_observation
from fastworkflow.observation_offloading import archive as archive_module
from fastworkflow.observation_offloading import erasure
from fastworkflow.observation_offloading import state as offload_state
from fastworkflow.observation_offloading.agent import build_tool_agent
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import archive_execute_observations
from fastworkflow.observation_offloading.state import (
    record_context_clause,
    reset_runtime_state,
    stored_handles,
)
from fastworkflow.observation_offloading.state import (
    current_execute_alias,
    current_scope,
)
from fastworkflow.utils.react import AskUserSuspend
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

# Read off the module rather than imported, for the reason the redaction cases
# do it: a regression check has to be RUNNABLE against a revision that lacks the
# fix. Where these names are absent the shims below answer for them, and the
# cases FAIL on their assertions -- a credential absent from a mid-turn file, a
# redacted read inside a live turn -- rather than erroring on an import.
SEAL_PENDING = getattr(archive_module, "SEAL_PENDING", "pending")
SEAL_SEALED = getattr(archive_module, "SEAL_SEALED", "sealed")
SEAL_NOT_REQUIRED = getattr(archive_module, "SEAL_NOT_REQUIRED", "not_required")
SEAL_UNKNOWN = getattr(archive_module, "SEAL_UNKNOWN", "unknown")
SEAL_GRACE_ENV = getattr(
    archive_module, "SEAL_GRACE_ENV", "FW_OFFLOAD_SEAL_GRACE_SECONDS"
)
REDACTION_ENV = archive_module.REDACTION_ENV
REDACTION_ON = archive_module.REDACTION_ON
REDACTION_OFF = archive_module.REDACTION_OFF

needs_the_seal = unittest.skipIf(
    not hasattr(archive_module, "SEAL_PENDING"),
    "the turn-completion seal is not in this revision",
)

SIDECAR_SUFFIX = ".offload-handles.sqlite3"
#: A credential shape ``Redactor._SECRET_PATTERNS`` recognises with no help
#: from the environment.
SK_TOKEN = "sk-livekey1234567890abcdef"
REDACTED = "[REDACTED]"


def response_with_credential(command: str = "sync") -> str:
    """One command response of the shape a workflow really returns."""
    return (
        "connector: okta-prod\n"
        f"api_key: {SK_TOKEN}\n"
        f"subject: {command}\n"
        "rows: 3 users synchronised\n"
    )


def seal_scope(archive, scope):
    """Complete the turn. ``None`` on a revision with no seal."""
    sealer = getattr(archive, "seal_scope", None)
    return None if sealer is None else sealer(scope)


def seal_state(archive, scope, alias: str) -> str:
    reader = getattr(archive, "seal_state", None)
    return SEAL_UNKNOWN if reader is None else reader(scope, alias)


def sweep(archive, **kwargs):
    sweeper = getattr(archive, "sweep_unsealed", None)
    return None if sweeper is None else sweeper(**kwargs)


def chatbot_scope(channel: str = "chat", turn: str = "turn-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="unbound",
        task_id="unbound", attempt=0, turn_key=turn,
    )


def experiment_scope(channel: str = "exp", turn: str = "turn-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="exp-7",
        task_id="task-3", attempt=1, turn_key=turn,
    )


class SealFixture(unittest.TestCase):
    """A sidecar in a temporary directory, and a clean configuration."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._restore_env: dict[str, str | None] = {}
        for name in (
            REDACTION_ENV, SEAL_GRACE_ENV, "FW_OBS_CAPTURE_PROFILE",
            "FW_OFFLOAD_EVIDENCE_PRESERVATION", "FW_OFFLOAD_EVENTS",
        ):
            self._restore_env[name] = os.environ.pop(name, None)
        self.addCleanup(self._restore_environment)
        for name in ("_warned_redaction", "_warned_grace"):
            warned = getattr(archive_module, name, None)
            if warned is not None:
                warned.clear()
                self.addCleanup(warned.clear)
        self.db_path = os.path.join(self.temp.name, "observability.sqlite3")
        self.sidecar = self.db_path + SIDECAR_SUFFIX

    def _restore_environment(self) -> None:
        for name, value in self._restore_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # -- helpers ---------------------------------------------------------

    def file_bytes(self, path: str | None = None) -> bytes:
        """Every byte the sidecar occupies, journal included."""
        blob = b""
        base = path or self.sidecar
        for suffix in ("", "-wal", "-journal"):
            if os.path.exists(base + suffix):
                with open(base + suffix, "rb") as handle:
                    blob += handle.read()
        return blob

    def persist(self, text: str, *, scope=None, alias: str = "O1", order: int = 1,
                archive=None):
        scope = scope or chatbot_scope()
        target = archive or RuntimeHandleArchive(self.sidecar)
        stored = target.persist(
            scope, alias=alias, offload_order=order,
            command_name="execute_workflow_query", step_index=order,
            text=text, text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        return target, stored


# ---------------------------------------------------------------------------
# Requirement 1 and 5: what "over" means, and the toggle
# ---------------------------------------------------------------------------


class InFlightTests(SealFixture):
    """While the turn is running, every read is the raw response."""

    def test_a_mid_turn_row_is_the_raw_response_on_disk(self) -> None:
        """The in-flight cost of the design, asserted rather than assumed."""
        text = response_with_credential()
        archive, stored = self.persist(text)
        scope = chatbot_scope()

        self.assertEqual(stored["text"], text)
        self.assertEqual(archive.get(scope, "O1")["text"], text)
        self.assertIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_PENDING)

    def test_the_seal_is_what_redacts_and_it_is_idempotent(self) -> None:
        text = response_with_credential()
        archive, _ = self.persist(text)
        scope = chatbot_scope()

        first = seal_scope(archive, scope)
        self.assertEqual(first["sealed"], 1)
        self.assertEqual(first["redacted"], 1)
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_SEALED)
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        self.assertIn(REDACTED, archive.get(scope, "O1")["text"])
        # Everything that was not a secret survives, which is what keeps a
        # sealed archive worth reading.
        self.assertIn("rows: 3 users synchronised", archive.get(scope, "O1")["text"])

        again = seal_scope(archive, scope)
        self.assertEqual(again["sealed"], 0)
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_SEALED)

    def test_a_seal_only_touches_the_scope_it_names(self) -> None:
        """One turn completing must not redact the turn running beside it."""
        mine = chatbot_scope("mine", "turn-1")
        theirs = chatbot_scope("theirs", "turn-1")
        archive, _ = self.persist(response_with_credential(), scope=mine)
        self.persist(response_with_credential(), scope=theirs, archive=archive)

        seal_scope(archive, mine)

        self.assertIn(REDACTED, archive.get(mine, "O1")["text"])
        self.assertEqual(archive.get(theirs, "O1")["text"],
                         response_with_credential())
        self.assertEqual(seal_state(archive, theirs, "O1"), SEAL_PENDING)

    @needs_the_seal
    def test_redaction_off_owes_no_seal_and_never_redacts(self) -> None:
        """Requirement 5: off still means never redact."""
        os.environ[REDACTION_ENV] = REDACTION_OFF
        text = response_with_credential()
        archive, stored = self.persist(text)
        scope = chatbot_scope()

        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_NOT_REQUIRED)
        result = seal_scope(archive, scope)
        self.assertEqual(result["sealed"], 0)
        self.assertEqual(archive.get(scope, "O1")["text"], text)
        self.assertIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        # And the sweep leaves it alone too, whatever its age.
        with sqlite3.connect(self.sidecar) as conn:
            conn.execute("UPDATE observation_seal_state SET opened_at=?, owner_id=?",
                         ("2000-01-01T00:00:00Z", "pid-dead"))
            conn.commit()
        sweep(archive)
        self.assertEqual(archive.get(scope, "O1")["text"], text)


# ---------------------------------------------------------------------------
# Requirement 6 and 7: the four states, erasure, retention, preservation
# ---------------------------------------------------------------------------


class FidelityRecordTests(SealFixture):
    """A reader can tell an unsealed row from a sealed one, and both from a
    row that never held a secret."""

    @needs_the_seal
    def test_the_four_states_are_distinguishable(self) -> None:
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.sidecar)
        # pending: raw on disk, seal owed.
        self.persist(response_with_credential(), scope=scope, alias="O1",
                     archive=archive)
        # not_required: the toggle was off at the write.
        os.environ[REDACTION_ENV] = REDACTION_OFF
        self.persist("nothing secret here\n", scope=scope, alias="O2", order=2,
                     archive=archive)
        os.environ.pop(REDACTION_ENV)
        # unknown: a row written before the seal ledger existed.
        legacy = "a row from an older revision\n"
        with sqlite3.connect(self.sidecar) as conn:
            conn.execute(
                "INSERT INTO observation_offload_handles ("
                "scope_id, scope_json, alias, offload_order, command_name, "
                "step_index, text_utf8, text_sha256, persisted_at"
                ") VALUES (?,?,?,?,?,?,?,?,?)",
                (scope.scope_id, "{}", "O3", 3, "execute_workflow_query", 3,
                 legacy.encode("utf-8"),
                 hashlib.sha256(legacy.encode("utf-8")).hexdigest(),
                 "2026-01-01T00:00:00Z"),
            )
            conn.commit()

        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_PENDING)
        self.assertEqual(seal_state(archive, scope, "O2"), SEAL_NOT_REQUIRED)
        self.assertEqual(seal_state(archive, scope, "O3"), SEAL_UNKNOWN)
        # A pending row says redacted=False because nothing has RUN, not
        # because there was nothing to find. The seal is what settles that.
        self.assertFalse(archive.capture_record(scope, "O1")["redacted"])
        self.assertEqual(
            archive.capture_record(scope, "O1")["seal_state"], SEAL_PENDING
        )
        seal_scope(archive, scope)
        sealed = archive.capture_record(scope, "O1")
        self.assertEqual(sealed["seal_state"], SEAL_SEALED)
        self.assertTrue(sealed["redacted"])
        self.assertTrue(sealed["sealed_at"])
        # The row that never held a secret is sealed and says redacted=False,
        # which is the distinction the fidelity table exists for.
        self.persist("no secret at all\n", scope=scope, alias="O4", order=4,
                     archive=archive)
        seal_scope(archive, scope)
        clean = archive.capture_record(scope, "O4")
        self.assertEqual(clean["seal_state"], SEAL_SEALED)
        self.assertFalse(clean["redacted"])
        # And nothing rewrote the row the ledger knows nothing about.
        self.assertEqual(archive.get(scope, "O3")["text"], legacy)
        self.assertIsNone(archive.capture_record(scope, "O3"))

    @needs_the_seal
    def test_a_sidecar_written_before_this_change_opens_and_reads(self) -> None:
        """Additive state is created on open; there is no migration step.

        The file is created with only the four tables and row shapes an
        older sidecar wrote, then opened by the current code.
        """
        scope = chatbot_scope()
        text = "an older revision's row\n"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.sidecar) as conn:
            conn.execute(
                "CREATE TABLE observation_offload_handles ("
                "scope_id TEXT NOT NULL, scope_json TEXT NOT NULL, "
                "alias TEXT NOT NULL, offload_order INTEGER NOT NULL, "
                "command_name TEXT NOT NULL, step_index INTEGER NOT NULL, "
                "text_utf8 BLOB NOT NULL, text_sha256 TEXT NOT NULL, "
                "persisted_at TEXT NOT NULL, PRIMARY KEY (scope_id, alias))"
            )
            conn.execute(
                "INSERT INTO observation_offload_handles VALUES (?,?,?,?,?,?,?,?,?)",
                (scope.scope_id, json.dumps({"channel_id": "chat"}), "O1", 1,
                 "execute_workflow_query", 1, text.encode("utf-8"), digest,
                 "2026-02-01T00:00:00Z"),
            )
            conn.commit()

        archive = RuntimeHandleArchive(self.sidecar)

        self.assertEqual(archive.get(scope, "O1")["text"], text)
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_UNKNOWN)
        self.assertIsNone(archive.capture_record(scope, "O1"))
        # Sealing the scope leaves it alone -- nothing is owed on a row whose
        # bytes were already final when it was written.
        self.assertEqual(seal_scope(archive, scope)["sealed"], 0)
        self.assertEqual(archive.get(scope, "O1")["text"], text)
        # And a new row in the same file gets the new ledger.
        self.persist(response_with_credential(), scope=scope, alias="O2", order=2,
                     archive=archive)
        self.assertEqual(seal_state(archive, scope, "O2"), SEAL_PENDING)


class ErasureAndRetentionTests(SealFixture):
    """Requirement 6: both work on sealed and unsealed rows alike."""

    def test_a_channel_is_erased_in_either_state(self) -> None:
        for state, do_seal in (("unsealed", False), ("sealed", True)):
            with self.subTest(state=state):
                sidecar = os.path.join(self.temp.name, f"{state}.sqlite3")
                scope = chatbot_scope("erase")
                archive = RuntimeHandleArchive(sidecar)
                self.persist(response_with_credential(), scope=scope,
                             archive=archive)
                if do_seal:
                    seal_scope(archive, scope)

                deleted = erasure.forget_channel(sidecar, "erase")

                self.assertEqual(deleted["observation_offload_handles"], 1)
                # Discovered structurally, with no name in the erasure module.
                self.assertEqual(deleted.get("observation_seal_state", 0), 1)
                self.assertIsNone(archive.get(scope, "O1"))
                blob = self.file_bytes(sidecar)
                self.assertNotIn(b"okta-prod", blob)
                self.assertNotIn(SK_TOKEN.encode("ascii"), blob)

    def test_retention_prunes_a_scope_in_either_state(self) -> None:
        for state, do_seal in (("unsealed", False), ("sealed", True)):
            with self.subTest(state=state):
                sidecar = os.path.join(self.temp.name, f"prune-{state}.sqlite3")
                scope = chatbot_scope("old")
                archive = RuntimeHandleArchive(sidecar)
                self.persist(response_with_credential(), scope=scope,
                             archive=archive)
                if do_seal:
                    seal_scope(archive, scope)
                with sqlite3.connect(sidecar) as conn:
                    for table, column in (
                        ("observation_offload_handles", "persisted_at"),
                        ("observation_capture_policy", "recorded_at"),
                        ("observation_seal_state", "opened_at"),
                    ):
                        try:
                            conn.execute(f'UPDATE "{table}" SET "{column}"=?',
                                         ("2000-01-01T00:00:00Z",))
                        except sqlite3.OperationalError:
                            pass
                    conn.commit()

                deleted = erasure.prune(sidecar, retention_days=30,
                                        max_bytes=1_000_000_000)

                self.assertEqual(deleted["scopes"], 1)
                self.assertIsNone(archive.get(scope, "O1"))

    def test_a_preserved_experiment_scope_survives_in_either_state(self) -> None:
        for state, do_seal in (("unsealed", False), ("sealed", True)):
            with self.subTest(state=state):
                sidecar = os.path.join(self.temp.name, f"keep-{state}.sqlite3")
                kept = experiment_scope("exp-channel")
                erasable = chatbot_scope("chat-channel", turn="turn-2")
                archive = RuntimeHandleArchive(sidecar)
                self.persist(response_with_credential(), scope=kept,
                             archive=archive)
                self.persist(response_with_credential(), scope=erasable,
                             archive=archive)
                if do_seal:
                    seal_scope(archive, kept)
                    seal_scope(archive, erasable)

                deleted = erasure.forget_all_channels(sidecar)

                self.assertEqual(deleted["preserved_scopes"], 1)
                self.assertIsNotNone(archive.get(kept, "O1"))
                self.assertIsNone(archive.get(erasable, "O1"))


# ---------------------------------------------------------------------------
# Requirement 4: the digests, across a seal
# ---------------------------------------------------------------------------


class DigestsAcrossASealTests(SealFixture):
    """Three digests, three meanings, and idempotence survives the rewrite."""

    @needs_the_seal
    def test_the_archiver_is_idempotent_across_a_seal(self) -> None:
        """A re-persist of a sealed alias is a readback, not a collision.

        The in-process memo that normally makes the archiver skip a step it has
        already written is dropped with the turn's residency, so a process that
        re-visits a sealed scope reaches ``persist`` with the RAW digest of text
        whose stored row now covers SEALED bytes. That has to be recognised as
        the same observation, and it is re-derived rather than looked up: no raw
        digest is persisted beside the sealed bytes.
        """
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.sidecar)
        text = response_with_credential()
        trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_connector"},
            "observation_0": text,
        }
        archive_execute_observations(trajectory, scope=scope,
                                     selected_archive=archive)
        seal_scope(archive, scope)
        sealed = archive.get(scope, "O1")
        # Exactly what a session close or the next turn's bind leaves behind.
        offload_state.reclaim_scope(scope)

        again = archive_execute_observations(trajectory, scope=scope,
                                             selected_archive=archive)

        # No refusal was recorded, the row was not rewritten, and the alias is
        # readable from the archive that holds it.
        self.assertEqual([row["alias"] for row in again], ["O1"])
        self.assertEqual(archive.get(scope, "O1")["text"], sealed["text"])
        self.assertEqual(archive.get(scope, "O1")["text_sha256"],
                         sealed["text_sha256"])
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_SEALED)
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        # And the hot copy the second pass cached is what the archive kept.
        self.assertEqual(stored_handles(scope)["O1"]["text"], sealed["text"])

    @needs_the_seal
    def test_the_raw_digest_is_never_written_beside_sealed_bytes(self) -> None:
        """The refusal to leave a confirmation oracle, kept across the seal.

        While the row is pending its own ``text_sha256`` covers the raw bytes
        that are sitting right beside it, so it tells an attacker nothing they
        cannot already read. Once sealed, that digest must be gone from the
        file: a digest of unredacted text stored next to the redaction is a
        confirmation oracle for the credential the redaction just removed.
        """
        scope = chatbot_scope()
        text = response_with_credential()
        raw_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        archive, _ = self.persist(text, scope=scope)

        self.assertIn(raw_digest.encode("ascii"), self.file_bytes())
        seal_scope(archive, scope)

        blob = self.file_bytes()
        self.assertNotIn(raw_digest.encode("ascii"), blob)
        self.assertNotIn(SK_TOKEN.encode("ascii"), blob)
        stored = archive.get(scope, "O1")
        self.assertEqual(
            stored["text_sha256"],
            hashlib.sha256(stored["text"].encode("utf-8")).hexdigest(),
        )

    @needs_the_seal
    def test_a_different_text_is_still_refused_after_a_seal(self) -> None:
        scope = chatbot_scope()
        archive, _ = self.persist(response_with_credential(), scope=scope)
        seal_scope(archive, scope)
        other = "a completely different observation with no secret\n"
        with self.assertRaises(PersistenceError):
            archive.persist(
                scope, alias="O1", offload_order=1,
                command_name="execute_workflow_query", step_index=1,
                text=other,
                text_sha256=hashlib.sha256(other.encode("utf-8")).hexdigest(),
            )

    @needs_the_seal
    def test_a_sealed_row_still_reads_back_through_its_own_digest(self) -> None:
        """``_decode_row`` verifies every read, so a seal that left the digest
        behind would turn every later read into a permanent failure."""
        scope = chatbot_scope()
        archive, _ = self.persist(response_with_credential(), scope=scope)
        seal_scope(archive, scope)
        reopened = RuntimeHandleArchive(self.sidecar)
        self.assertIsNotNone(reopened.get(scope, "O1"))
        self.assertEqual(len(reopened.list(scope)), 1)


# ---------------------------------------------------------------------------
# Requirement 3: the crash sweep
# ---------------------------------------------------------------------------


class CrashSweepTests(SealFixture):
    """A process that dies mid-turn must not leave raw bytes forever."""

    def age_the_row(self, *, owner: str = "pid-dead-1234", seconds: int = 7_200,
                    sidecar: str | None = None) -> None:
        """Rewrite the seal row as one a DIFFERENT, now-dead process opened."""
        stamp = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with sqlite3.connect(sidecar or self.sidecar) as conn:
            conn.execute(
                "UPDATE observation_seal_state SET owner_id=?, opened_at=?",
                (owner, stamp),
            )
            conn.commit()

    @needs_the_seal
    def test_a_row_a_dead_process_left_raw_is_sealed_when_the_file_reopens(self) -> None:
        """The acceptance case: opening the sidecar is the recovery trigger."""
        scope = chatbot_scope()
        archive, _ = self.persist(response_with_credential(), scope=scope)
        self.age_the_row(seconds=2 * 86_400)
        self.assertIn(SK_TOKEN.encode("ascii"), self.file_bytes())

        # Exactly what the next process to run a turn against this store does.
        reopened = RuntimeHandleArchive(self.sidecar)

        self.assertEqual(seal_state(reopened, scope, "O1"), SEAL_SEALED)
        self.assertIn(REDACTED, reopened.get(scope, "O1")["text"])
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        self.assertTrue(reopened.capture_record(scope, "O1")["redacted"])

    @needs_the_seal
    def test_the_sweep_never_touches_a_row_this_process_opened(self) -> None:
        """The guard that makes the sweep unable to reach a LIVE turn."""
        scope = chatbot_scope()
        archive, _ = self.persist(response_with_credential(), scope=scope)
        # Old enough for the horizon, but still owned by this process, which is
        # what a turn that has been running a long time looks like.
        self.age_the_row(owner=archive_module.owner_id(), seconds=5 * 86_400)

        result = sweep(archive)

        self.assertEqual(result["sealed"], 0)
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_PENDING)
        self.assertEqual(archive.get(scope, "O1")["text"],
                         response_with_credential())

    @needs_the_seal
    def test_the_sweep_leaves_a_row_inside_the_grace_window_alone(self) -> None:
        """A turn in another live process is not a crashed one."""
        scope = chatbot_scope()
        archive, _ = self.persist(response_with_credential(), scope=scope)
        self.age_the_row(seconds=60)  # a minute old, default horizon is a day

        result = sweep(archive)

        self.assertEqual(result["sealed"], 0)
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_PENDING)

    @needs_the_seal
    def test_the_horizon_is_configurable_and_the_sweep_can_be_turned_off(self) -> None:
        self.assertEqual(archive_module.seal_grace_seconds(),
                         archive_module.DEFAULT_SEAL_GRACE_SECONDS)
        self.assertEqual(archive_module.seal_grace_seconds("30"), 30)
        self.assertIsNone(archive_module.seal_grace_seconds("off"))
        # A typo warns once and falls back to the default rather than either
        # disabling the sweep or sealing everything.
        with self.assertLogs(archive_module.logger, level="WARNING") as logs:
            self.assertEqual(archive_module.seal_grace_seconds("sometimes"),
                             archive_module.DEFAULT_SEAL_GRACE_SECONDS)
        self.assertIn(SEAL_GRACE_ENV, "\n".join(logs.output))

        scope = chatbot_scope()
        archive, _ = self.persist(response_with_credential(), scope=scope)
        self.age_the_row(seconds=5 * 86_400)
        os.environ[SEAL_GRACE_ENV] = "off"
        self.assertEqual(sweep(archive)["sealed"], 0)
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_PENDING)
        os.environ[SEAL_GRACE_ENV] = "3600"
        self.assertEqual(sweep(archive)["sealed"], 1)
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_SEALED)

    @needs_the_seal
    def test_retention_sweeps_a_store_whose_processes_are_all_gone(self) -> None:
        """The second trigger: the only job that visits an abandoned store.

        The scope is young enough that retention deletes nothing, so the only
        thing that can remove the credential from the file is the sweep.
        """
        scope = chatbot_scope()
        archive, _ = self.persist(response_with_credential(), scope=scope)
        self.age_the_row(seconds=3 * 86_400)
        os.environ[SEAL_GRACE_ENV] = "3600"
        self.assertIn(SK_TOKEN.encode("ascii"), self.file_bytes())

        result = erasure.prune(self.sidecar, retention_days=3650,
                               max_bytes=1_000_000_000)

        self.assertEqual(result["scopes"], 0)
        self.assertEqual(result["sealed_by_sweep"], 1)
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())
        self.assertIsNotNone(archive.get(scope, "O1"))

    @needs_the_seal
    def test_retention_sweeps_a_preserved_file_too(self) -> None:
        """Sealing is a CAPTURE decision; preservation governs DELETION.

        A preserved evaluation corpus must not keep a crashed turn's credential
        forever just because its rows may not be deleted.
        """
        scope = experiment_scope("exp-channel")
        archive, _ = self.persist(response_with_credential(), scope=scope)
        self.age_the_row(seconds=3 * 86_400)
        os.environ[SEAL_GRACE_ENV] = "3600"
        Path(erasure.preserve_sentinel_path(self.sidecar)).write_text("keep")

        result = erasure.prune(self.sidecar, retention_days=0,
                               max_bytes=1)

        self.assertEqual(result["file_preserved"], 1)
        self.assertEqual(result["sealed_by_sweep"], 1)
        self.assertIsNotNone(archive.get(scope, "O1"))
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())

    @needs_the_seal
    def test_a_row_with_no_seal_ledger_entry_is_never_swept(self) -> None:
        """Requirement 7 again: a pre-change row owes nothing and is left alone."""
        scope = chatbot_scope()
        text = "an older revision's row, already final\n"
        archive = RuntimeHandleArchive(self.sidecar)
        with sqlite3.connect(self.sidecar) as conn:
            conn.execute(
                "INSERT INTO observation_offload_handles ("
                "scope_id, scope_json, alias, offload_order, command_name, "
                "step_index, text_utf8, text_sha256, persisted_at"
                ") VALUES (?,?,?,?,?,?,?,?,?)",
                (scope.scope_id, "{}", "O1", 1, "execute_workflow_query", 1,
                 text.encode("utf-8"),
                 hashlib.sha256(text.encode("utf-8")).hexdigest(),
                 "2020-01-01T00:00:00Z"),
            )
            conn.commit()

        self.assertEqual(sweep(archive)["sealed"], 0)
        self.assertEqual(archive.get(scope, "O1")["text"], text)


# ---------------------------------------------------------------------------
# Requirements 1 and 2: the production lifecycle, with a real session
# ---------------------------------------------------------------------------


class Signature(dspy.Signature):
    user_query: str = dspy.InputField()
    final_answer: str = dspy.OutputField()


class LiveTurnFixture(unittest.TestCase):
    """A real session: a workflow, an execution context and a built tool agent.

    Modelled on ``test_offload_state_reclamation``'s fixture, and for the same
    reason: the seal under test has to be reached by the production lifecycle,
    because a seal only a test can trigger is not the fix.
    """

    workflow_path = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self._restore_env: dict[str, str | None] = {}
        for name in (REDACTION_ENV, SEAL_GRACE_ENV, "FW_OBS_CAPTURE_PROFILE",
                     "FW_OFFLOAD_EVENTS"):
            self._restore_env[name] = os.environ.pop(name, None)
        os.environ["FASTWORKFLOW_STATE_ROOT"] = os.path.join(self.temp.name, "state")
        fastworkflow.init({"FASTWORKFLOW_STATE_ROOT":
                           os.path.join(self.temp.name, "state")})
        self.open_sessions: list[tuple] = []

    def tearDown(self) -> None:
        for ctx, workflow in list(self.open_sessions):
            try:
                ctx.close()
                workflow.close()
            except Exception:  # noqa: BLE001
                pass
        reset_runtime_state()
        os.environ.pop("FASTWORKFLOW_STATE_ROOT", None)
        for name, value in self._restore_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.temp.cleanup()

    def make_session(self, *, channel: str, turn: str):
        workflow = fastworkflow.Workflow.create(
            self.workflow_path, workflow_id_str=f"{channel}-{turn}-{uuid.uuid4().hex}"
        )
        ctx = WorkflowExecutionContext(run_as_agent=False, session_key=channel)
        ctx.bind_app_workflow(workflow)
        ctx.bind_observability_identity(channel_id=channel)
        ctx._turn_key = turn
        ctx.push_active_workflow(workflow)

        def execute_workflow_query(command: str) -> str:
            alias = current_execute_alias()
            record_context_clause(current_scope(), alias, "Fixture " + command)
            return response_with_credential(command)

        def ask_user(question: str) -> str:
            raise AskUserSuspend(question)

        agent = build_tool_agent(ctx, Signature,
                                 [execute_workflow_query, ask_user], max_iters=8)
        ctx._workflow_tool_agent = agent
        agent.extract = lambda **kwargs: dspy.Prediction(final_answer="done")
        self.open_sessions.append((ctx, workflow))
        return ctx, workflow, agent

    @staticmethod
    def script(agent, steps) -> None:
        queue = iter(steps)

        def decide(**kwargs):
            name, arguments = next(queue)
            return dspy.Prediction(next_thought="scripted", next_tool_name=name,
                                   next_tool_args=arguments)

        agent.react = decide

    def close_session(self, ctx, workflow) -> None:
        ctx.pop_active_workflow()
        ctx.close()
        workflow.close()
        if (ctx, workflow) in self.open_sessions:
            self.open_sessions.remove((ctx, workflow))

    # -- the reads an agent can make -------------------------------------

    def reads(self, agent, scope, alias: str = "O1") -> dict[str, str]:
        """Every read path an agent has, at one moment."""
        archive = agent.observation_archive
        # search_memory's own two-tier resolution: hot cache, then SQLite.
        served = stored_handles(scope).get(alias) or archive.get(scope, alias)
        return {
            "search_memory": None if served is None else served["text"],
            "rehydration": archived_observation(alias, scope=scope,
                                                archive=archive),
            "archive_row": (archive.get(scope, alias) or {}).get("text"),
        }

    def assert_all_raw(self, reads: dict[str, str], where: str) -> None:
        for path, text in reads.items():
            with self.subTest(path=path, moment=where):
                self.assertIsNotNone(text, f"{path} read nothing {where}")
                self.assertIn(SK_TOKEN, text,
                              f"{path} was degraded {where}")
                self.assertNotIn(REDACTED, text)


class TurnCompletionSealsTests(LiveTurnFixture):
    """Requirement 1: the seal reuses the existing notion of "over"."""

    def test_binding_the_next_scope_seals_the_previous_turn(self) -> None:
        ctx, workflow, agent = self.make_session(channel="chan", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "first"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        first = agent.continuation_scope
        archive = agent.observation_archive
        # Mid-turn, every read is raw and the file says so.
        self.assert_all_raw(self.reads(agent, first), "during the turn")
        self.assertEqual(seal_state(archive, first, "O1"), SEAL_PENDING)

        # The agent binding the NEXT turn is it saying the previous one is over.
        ctx._turn_key = "turn-2"
        self.script(agent, [("execute_workflow_query", {"command": "second"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        second = agent.continuation_scope

        self.assertNotEqual(first, second)
        self.assertEqual(seal_state(archive, first, "O1"), SEAL_SEALED)
        self.assertIn(REDACTED, archive.get(first, "O1")["text"])
        # And the turn that is actually running kept its raw evidence.
        self.assert_all_raw(self.reads(agent, second), "in the second turn")
        self.assertEqual(seal_state(archive, second, "O1"), SEAL_PENDING)

    def test_closing_the_session_seals_the_turn(self) -> None:
        ctx, workflow, agent = self.make_session(channel="closed", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "only"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        path = agent.observation_archive.db_path
        self.assertEqual(seal_state(agent.observation_archive, scope, "O1"),
                         SEAL_PENDING)

        self.close_session(ctx, workflow)

        reopened = RuntimeHandleArchive(path)
        self.assertEqual(seal_state(reopened, scope, "O1"), SEAL_SEALED)
        self.assertIn(REDACTED, reopened.get(scope, "O1")["text"])
        # The hot copy went with it, so memory cannot serve raw text for a
        # turn whose disk copy is sealed.
        self.assertEqual(stored_handles(scope), {})

    def test_the_awaiting_user_guard_skips_the_seal(self) -> None:
        """A turn waiting on the user is not over, so its evidence is not sealed."""
        ctx, workflow, agent = self.make_session(channel="susp", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "one"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            prediction = agent.forward(user_query="fixture")
        self.assertTrue(prediction.suspended)
        scope = agent.continuation_scope
        archive = agent.observation_archive

        ctx._awaiting_user = True
        ctx.pop_active_workflow()
        ctx.close()

        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_PENDING)
        self.assert_all_raw(self.reads(agent, scope), "after a suspended close")

    def test_an_exported_suspension_skips_the_seal(self) -> None:
        """The second half of the guard: ``export_suspended() is not None``."""
        ctx, workflow, agent = self.make_session(channel="susp2", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "one"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        archive = agent.observation_archive
        self.assertIsNotNone(agent.export_suspended())

        # _awaiting_user was never set -- only the agent knows it is suspended.
        self.assertFalse(ctx._awaiting_user)
        ctx.pop_active_workflow()
        ctx.close()

        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_PENDING)

    def test_the_bind_scope_guard_skips_a_still_suspended_agent(self) -> None:
        """``bind_scope`` is skipped while ``self._suspended is not None``."""
        ctx, workflow, agent = self.make_session(channel="susp3", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "one"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        archive = agent.observation_archive
        self.assertIsNotNone(agent._suspended)

        # A caller binding a scope by hand over a still-suspended agent.
        ctx._turn_key = "turn-2"
        with tracing.host_scope(ctx):
            agent.bind_scope()

        self.assertNotEqual(agent.continuation_scope, scope)
        self.assertEqual(seal_state(archive, scope, "O1"), SEAL_PENDING)


class SuspensionRoundTripTests(LiveTurnFixture):
    """Requirement 2: a suspended turn is never sealed, across a round trip."""

    def test_a_suspension_survives_serialization_and_still_reads_raw(self) -> None:
        """The acceptance case for the owner's "including ask_user waits and
        serialization/deserialization" clause.

        The turn suspends, its payload goes through ``json.dumps``/``loads``
        exactly as a session state file carries it, the session is CLOSED (the
        eviction), every process-local registry is emptied so nothing below can
        be answered out of memory, and a FRESH agent imports it and resumes.
        Both reads of the same alias must still be the raw response.
        """
        ctx, workflow, agent = self.make_session(channel="rt", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "first"}),
                            ("execute_workflow_query", {"command": "second"}),
                            ("ask_user", {"question": "continue?"})])
        with tracing.host_scope(ctx):
            prediction = agent.forward(user_query="fixture")
        self.assertTrue(prediction.suspended)
        scope = agent.continuation_scope
        path = agent.observation_archive.db_path
        self.assert_all_raw(self.reads(agent, scope, "O1"), "during the turn")

        blob = json.loads(json.dumps(agent.export_suspended()))
        ctx._awaiting_user = True
        ctx.pop_active_workflow()
        ctx.close()
        # As close to a fresh process as one interpreter allows.
        reset_runtime_state()
        self.assertEqual(stored_handles(scope), {})

        ctx2, workflow2, agent2 = self.make_session(channel="rt", turn="turn-1")
        agent2.import_suspended(blob)
        self.script(agent2, [("execute_workflow_query", {"command": "third"}),
                             ("finish", {})])
        with tracing.host_scope(ctx2):
            resumed = agent2.resume("go on")

        self.assertEqual(agent2.continuation_scope, scope)
        self.assertEqual(agent2.observation_archive.db_path, path)
        for alias in ("O1", "O2", "O3"):
            self.assert_all_raw(self.reads(agent2, scope, alias),
                                f"after the round trip ({alias})")
        self.assertIn(SK_TOKEN, resumed.trajectory["observation_3"])
        # Nothing was sealed by the round trip, and nothing was sealed by the
        # resume either: the turn is only now finishing.
        for alias in ("O1", "O2", "O3"):
            self.assertEqual(
                seal_state(agent2.observation_archive, scope, alias),
                SEAL_PENDING,
            )

        self.close_session(ctx2, workflow2)
        reopened = RuntimeHandleArchive(path)
        for alias in ("O1", "O2", "O3"):
            self.assertEqual(seal_state(reopened, scope, alias), SEAL_SEALED)
            self.assertIn(REDACTED, reopened.get(scope, alias)["text"])


# ---------------------------------------------------------------------------
# The conversation summary that feeds the NEXT turn
# ---------------------------------------------------------------------------


class SummaryOrderingTests(LiveTurnFixture):
    """The summary fed to the next turn must be built from RAW text.

    ``_finalize_agent_output`` summarises ``self._action_log``, whose
    ``response`` is the command's ``response_text`` captured at execution time
    and never read back from the evidence sidecar, and the summary it produces
    is what ``_refine_user_query`` feeds the LLM that refines the next turn's
    query. So the requirement is met by ORDERING and by SOURCE, and both are
    pinned here: if anyone later routes the action log through the capture
    pipeline, or moves the seal earlier than the summary, these fail loudly.
    """

    def finalize(self, ctx, agent, scope):
        """Run the real ``_finalize_agent_output``, capturing what it summarises."""
        seen: dict[str, object] = {}
        archive = agent.observation_archive

        def spy(user_query, workflow_actions, final_agent_response):
            seen["user_query"] = user_query
            seen["workflow_actions"] = json.loads(json.dumps(workflow_actions))
            seen["final_agent_response"] = final_agent_response
            # What the archive held AT THE MOMENT the summary was produced.
            row = archive.get(scope, "O1")
            seen["archive_text_at_summary_time"] = None if row is None else row["text"]
            seen["seal_state_at_summary_time"] = seal_state(archive, scope, "O1")
            return "a summary mentioning the api key", json.dumps({"seen": True})

        ctx._extract_conversation_summary = spy
        return seen, ctx._finalize_agent_output(
            "what is the connector's api key?",
            dspy.Prediction(final_answer=f"the key is {SK_TOKEN}"),
        )

    def test_the_summary_is_built_from_raw_text(self) -> None:
        ctx, workflow, agent = self.make_session(channel="sum", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "show"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        # The action log the real dispatch path appends, with the real response.
        ctx.append_action_log({
            "command": "show_connector",
            "command_name": "show_connector",
            "parameters": {},
            "response": response_with_credential("show"),
        })

        seen, output = self.finalize(ctx, agent, scope)

        actions = seen["workflow_actions"]
        self.assertEqual(len(actions), 1)
        self.assertIn(SK_TOKEN, actions[0]["response"])
        self.assertNotIn(REDACTED, actions[0]["response"])
        self.assertIn(SK_TOKEN, seen["final_agent_response"])
        # The turn was recorded, and the entry the next turn's refinement reads
        # is the one the summary produced.
        self.assertEqual(
            output.command_response.artifacts["conversation_summary"],
            "a summary mentioning the api key",
        )
        newest = ctx.conversation_history.messages[-1]
        self.assertIn("a summary mentioning the api key", json.dumps(newest))
        # And the action log itself is never rewritten by the offloading
        # runtime, which is the other way this could regress.
        self.assertIn(SK_TOKEN, ctx.action_log[0]["response"])

    def test_the_seal_happens_after_the_summary(self) -> None:
        """Move the seal earlier and this breaks; that is its whole job."""
        ctx, workflow, agent = self.make_session(channel="order", turn="turn-1")
        self.script(agent, [("execute_workflow_query", {"command": "show"}),
                            ("finish", {})])
        with tracing.host_scope(ctx):
            agent.forward(user_query="fixture")
        scope = agent.continuation_scope
        path = agent.observation_archive.db_path
        ctx.append_action_log({
            "command": "show_connector",
            "command_name": "show_connector",
            "parameters": {},
            "response": response_with_credential("show"),
        })

        seen, _ = self.finalize(ctx, agent, scope)

        # At summary time the evidence was still raw and still unsealed.
        self.assertEqual(seen["seal_state_at_summary_time"], SEAL_PENDING)
        self.assertIn(SK_TOKEN, seen["archive_text_at_summary_time"])
        # Only the session close, which is strictly later, seals it.
        self.close_session(ctx, workflow)
        reopened = RuntimeHandleArchive(path)
        self.assertEqual(seal_state(reopened, scope, "O1"), SEAL_SEALED)
        self.assertIn(REDACTED, reopened.get(scope, "O1")["text"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
