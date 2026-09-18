"""ido-zlm: the evidence sidecar's response bytes ride the trace sink's pipeline.

Before this change, ``RuntimeHandleArchive.persist`` wrote exactly the bytes a
command returned. A credential in a command response therefore landed verbatim
in ``<observability.sqlite3>.offload-handles.sqlite3``, while the same text
inside a span attribute was scrubbed by ``observability.store.Redactor`` on its
way into the main database. The sidecar bypassed that pipeline entirely.

Redaction is now a toggle, ``FW_OFFLOAD_EVIDENCE_REDACTION``, and it is ON by
default. Both states are first-class: a developer turns it off because the
archive's whole job is to reproduce what the agent read, and devops leave it on
so a secret is not written to disk. Each archived observation records which
mode produced it, so an archive stays auditable about its own fidelity.

The claims here are made in BYTES wherever a leak is the thing being denied: a
row read back through the archive's own API proves what the API returns, not
what is in the file, and the file is what a deletion request and a stolen disk
are about. Every scan therefore opens the database (and its write-ahead log, if
any) and searches the raw bytes.

Everything runs against databases created in this test's own temporary
directory. Nothing here reads or writes a store it did not create.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest

from fastworkflow.observability import store as obs
from fastworkflow.observation_offloading import archive as archive_module
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import archive_execute_observations
from fastworkflow.observation_offloading.state import (
    archived_digest,
    reset_runtime_state,
    stored_handles,
)

try:  # The module that owns deletion, and now states the redaction policy.
    from fastworkflow.observation_offloading import erasure
except ImportError:  # pragma: no cover - only on a revision before ido-zlm
    erasure = None

# The toggle's own names are read off the module rather than imported, and the
# archive's new methods are reached through the shims below, for the reason
# ``test_offload_evidence_erasure`` reads ``erasure`` that way: a regression
# check has to be RUNNABLE against the revision that had the defect, or it
# proves nothing about the fix. On that revision these cases fail on their
# assertions -- a credential in the file, no recorded policy version -- instead
# of erroring on an import, and only the cases that test this change's own
# policy surface skip.
REDACTION_ENV = getattr(
    archive_module, "REDACTION_ENV", "FW_OFFLOAD_EVIDENCE_REDACTION"
)
REDACTION_ON = getattr(archive_module, "REDACTION_ON", "on")
REDACTION_OFF = getattr(archive_module, "REDACTION_OFF", "off")

needs_the_toggle = unittest.skipIf(
    not hasattr(archive_module, "redaction_mode"),
    "the sidecar redaction toggle is not in this revision",
)


def redaction_mode(*args, **kwargs) -> str:
    return archive_module.redaction_mode(*args, **kwargs)


def capture_record(archive, scope, alias: str):
    """The fidelity record for one archived observation, or ``None``.

    ``None`` on a revision that records none at all is the same answer as
    ``None`` for a row written before the table existed, and both are what the
    cases below refuse to accept where a record is required.
    """
    reader = getattr(archive, "capture_record", None)
    return None if reader is None else reader(scope, alias)

SIDECAR_SUFFIX = ".offload-handles.sqlite3"

#: A credential shape ``Redactor._SECRET_PATTERNS`` recognises with no help
#: from the environment, so this case does not depend on how the process was
#: started.
SK_TOKEN = "sk-livekey1234567890abcdef"

#: ...and one the scrub only knows about because the variable's NAME marks it
#: as secret. Included because it is the half of the pipeline a hand-rolled
#: regex in this package would not have had, and therefore the evidence that
#: the real one is being reused rather than imitated.
API_KEY_VAR = "IDOZLM_PLANTED_SERVICE_API_KEY"
ENV_SECRET = "planted-env-secret-value-zlm"

REDACTED = "[REDACTED]"


def response_with_credential() -> str:
    """One command response of the shape a workflow really returns."""
    return (
        "connector: okta-prod\n"
        f"api_key: {SK_TOKEN}\n"
        f"fallback_key: {ENV_SECRET}\n"
        "rows: 3 users synchronised\n"
    )


def innocent_response() -> str:
    return (
        "connector: okta-prod\n"
        "rows: 3 users synchronised\n"
    )


def chatbot_scope(channel: str = "chat", turn: str = "turn-1") -> RuntimeHandleScope:
    """A scope exactly as ``scope_for_host`` builds it with no experiment claim."""
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="unbound",
        task_id="unbound", attempt=0, turn_key=turn,
    )


def experiment_scope(channel: str = "chat", turn: str = "turn-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id=channel, experiment_id="exp-7",
        task_id="task-3", attempt=1, turn_key=turn,
    )


class RedactionFixture(unittest.TestCase):
    """A sidecar in a temporary directory, and a clean configuration."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self._restore_env: dict[str, str | None] = {}
        for name in (
            REDACTION_ENV,
            "FW_OBS_CAPTURE_PROFILE",
            "FW_OFFLOAD_EVIDENCE_PRESERVATION",
            "FW_OFFLOAD_EVENTS",
            API_KEY_VAR,
        ):
            self._restore_env[name] = os.environ.pop(name, None)
        self.addCleanup(self._restore_environment)
        # Warn-once state is module-level and would otherwise leak between
        # cases in either direction.
        warned = getattr(archive_module, "_warned_redaction", None)
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

    def plant_env_secret(self) -> None:
        """A loaded secret, set BEFORE any Redactor is built (it snapshots)."""
        os.environ[API_KEY_VAR] = ENV_SECRET

    def file_bytes(self) -> bytes:
        """Every byte the sidecar occupies, journal included."""
        blob = b""
        for suffix in ("", "-wal", "-journal"):
            path = self.sidecar + suffix
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    blob += handle.read()
        return blob

    def persist(self, text: str, *, scope=None, alias: str = "O1"):
        scope = scope or chatbot_scope()
        archive = RuntimeHandleArchive(self.sidecar)
        stored = archive.persist(
            scope, alias=alias, offload_order=1,
            command_name="execute_workflow_query", step_index=1,
            text=text, text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        # A revision whose ``persist`` returns nothing still stored a row; read
        # it back so the byte-level claims below are about that revision too.
        return archive, stored if stored is not None else archive.get(scope, alias)


class DefaultIsOnTests(RedactionFixture):
    """An unconfigured deployment is the safe one."""

    @needs_the_toggle
    def test_nothing_configured_means_redaction_on(self) -> None:
        self.assertNotIn(REDACTION_ENV, os.environ)
        self.assertEqual(redaction_mode(), REDACTION_ON)

    @needs_the_toggle
    def test_an_empty_value_is_not_a_configuration(self) -> None:
        os.environ[REDACTION_ENV] = "   "
        self.assertEqual(redaction_mode(), REDACTION_ON)

    @needs_the_toggle
    def test_an_unrecognised_value_warns_and_uses_the_default(self) -> None:
        os.environ[REDACTION_ENV] = "of"  # the typo that matters most
        with self.assertLogs(archive_module.logger, level="WARNING") as logs:
            self.assertEqual(redaction_mode(), REDACTION_ON)
        self.assertIn("FW_OFFLOAD_EVIDENCE_REDACTION=of", "\n".join(logs.output))
        # Warned once, not once per archived observation.
        self.assertEqual(redaction_mode(), REDACTION_ON)

    def test_a_typo_does_not_leave_a_credential_in_the_file(self) -> None:
        """The warning is not the point; what it falls back to is."""
        os.environ[REDACTION_ENV] = "yes-please"
        with self.assertLogs(archive_module.logger, level="WARNING"):
            self.persist(response_with_credential())
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())

    @needs_the_toggle
    def test_the_operator_spellings_resolve(self) -> None:
        for value, expected in (
            ("on", REDACTION_ON), ("ON", REDACTION_ON), ("1", REDACTION_ON),
            ("true", REDACTION_ON), ("off", REDACTION_OFF), ("0", REDACTION_OFF),
            ("false", REDACTION_OFF), ("No", REDACTION_OFF),
        ):
            with self.subTest(value=value):
                os.environ[REDACTION_ENV] = value
                self.assertEqual(redaction_mode(), expected)


class StoredBytesTests(RedactionFixture):
    """What is on disk, in both states, scanned as bytes."""

    def test_a_credential_is_not_stored_verbatim_by_default(self) -> None:
        self.plant_env_secret()
        text = response_with_credential()
        archive, stored = self.persist(text)

        blob = self.file_bytes()
        self.assertNotIn(SK_TOKEN.encode("ascii"), blob)
        self.assertNotIn(ENV_SECRET.encode("ascii"), blob)
        # Not silence, and not a stub: everything that was not a secret is
        # still there, which is what keeps a redacted archive worth reading.
        self.assertIn(b"rows: 3 users synchronised", blob)
        self.assertIn(REDACTED, stored["text"])
        self.assertIn("connector: okta-prod", stored["text"])
        self.assertEqual(archive.get(chatbot_scope(), "O1")["text"], stored["text"])

    def test_redaction_off_stores_the_response_verbatim(self) -> None:
        self.plant_env_secret()
        os.environ[REDACTION_ENV] = REDACTION_OFF
        text = response_with_credential()
        archive, stored = self.persist(text)

        blob = self.file_bytes()
        self.assertIn(SK_TOKEN.encode("ascii"), blob)
        self.assertIn(ENV_SECRET.encode("ascii"), blob)
        # And the archive reproduces the observation exactly, which is the
        # reason a developer turns the toggle off at all.
        self.assertEqual(stored["text"], text)
        self.assertEqual(archive.get(chatbot_scope(), "O1")["text"], text)

    def test_a_response_with_no_secret_is_stored_byte_identical_under_both(self) -> None:
        text = innocent_response()
        for mode in (REDACTION_ON, REDACTION_OFF):
            with self.subTest(mode=mode):
                reset_runtime_state()
                os.environ[REDACTION_ENV] = mode
                sidecar = os.path.join(self.temp.name, f"{mode}.sqlite3")
                archive = RuntimeHandleArchive(sidecar)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                archive.persist(
                    chatbot_scope(), alias="O1", offload_order=1,
                    command_name="execute_workflow_query", step_index=1,
                    text=text, text_sha256=digest,
                )
                stored = archive.get(chatbot_scope(), "O1")
                self.assertEqual(stored["text"], text)
                self.assertEqual(stored["text_sha256"], digest)


class CapturePolicyRecordTests(RedactionFixture):
    """An archive that can be asked about its own fidelity."""

    def test_the_policy_version_is_recorded_with_redaction_on(self) -> None:
        self.plant_env_secret()
        archive, _ = self.persist(response_with_credential())
        record = capture_record(archive, chatbot_scope(), "O1")
        self.assertEqual(
            record["capture_policy_version"], obs.CAPTURE_POLICY_VERSION
        )
        self.assertEqual(record["capture_profile"], "debug")
        self.assertEqual(record["redaction"], REDACTION_ON)
        self.assertTrue(record["redacted"])
        self.assertEqual(
            record["raw_utf8_bytes"],
            len(response_with_credential().encode("utf-8")),
        )

    def test_the_policy_version_is_recorded_with_redaction_off(self) -> None:
        os.environ[REDACTION_ENV] = REDACTION_OFF
        archive, _ = self.persist(response_with_credential())
        record = capture_record(archive, chatbot_scope(), "O1")
        self.assertEqual(
            record["capture_policy_version"], obs.CAPTURE_POLICY_VERSION
        )
        self.assertEqual(record["redaction"], REDACTION_OFF)
        self.assertFalse(record["redacted"])
        # No policy was consulted, and that is not the same as the `debug`
        # profile having been consulted and done nothing.
        self.assertEqual(record["capture_profile"], "")

    def test_a_redacted_row_is_distinguishable_from_one_with_no_secret(self) -> None:
        """The requirement the version alone does not meet."""
        archive, _ = self.persist(response_with_credential(), alias="O1")
        archive.persist(
            chatbot_scope(), alias="O2", offload_order=2,
            command_name="execute_workflow_query", step_index=2,
            text=innocent_response(),
            text_sha256=hashlib.sha256(
                innocent_response().encode("utf-8")
            ).hexdigest(),
        )
        leaky = capture_record(archive, chatbot_scope(), "O1")
        clean = capture_record(archive, chatbot_scope(), "O2")
        self.assertEqual(leaky["redaction"], clean["redaction"], REDACTION_ON)
        self.assertEqual(
            leaky["capture_policy_version"], clean["capture_policy_version"]
        )
        self.assertTrue(leaky["redacted"])
        self.assertFalse(clean["redacted"])

    def test_a_row_written_before_this_change_reads_as_unknown(self) -> None:
        """Never a guess, and never an error.

        The row is inserted the way the pre-change writer wrote one: into the
        handles table alone, with no fidelity record beside it. Nothing in the
        file says whether those bytes are full fidelity, so nothing may claim
        they are -- least of all by defaulting to "not redacted".
        """
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.sidecar)
        text = innocent_response()
        with sqlite3.connect(self.sidecar) as conn:
            conn.execute(
                "INSERT INTO observation_offload_handles ("
                "scope_id, scope_json, alias, offload_order, command_name, "
                "step_index, text_utf8, text_sha256, persisted_at"
                ") VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    scope.scope_id, "{}", "O9", 9, "execute_workflow_query", 9,
                    text.encode("utf-8"),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "2026-01-01T00:00:00Z",
                ),
            )
            conn.commit()
        # The observation itself still reads.
        self.assertEqual(archive.get(scope, "O9")["text"], text)
        self.assertIsNone(capture_record(archive, scope, "O9"))
        # And an alias the file has never heard of reads the same way, which is
        # the answer that was already correct before this table existed.
        self.assertIsNone(capture_record(archive, scope, "O404"))

    def test_the_record_describes_the_bytes_that_were_actually_kept(self) -> None:
        """A second persist of the same alias does not rewrite either half."""
        os.environ[REDACTION_ENV] = REDACTION_OFF
        archive, _ = self.persist(response_with_credential())
        os.environ[REDACTION_ENV] = REDACTION_ON
        with self.assertRaises(PersistenceError):
            archive.persist(
                chatbot_scope(), alias="O1", offload_order=1,
                command_name="execute_workflow_query", step_index=1,
                text=response_with_credential(),
                text_sha256=hashlib.sha256(
                    response_with_credential().encode("utf-8")
                ).hexdigest(),
            )
        # The stored bytes are the first write's, so the record must still be
        # the first write's too, or it describes bytes that are not there.
        self.assertEqual(
            archive.get(chatbot_scope(), "O1")["text"], response_with_credential()
        )
        self.assertEqual(
            capture_record(archive, chatbot_scope(), "O1")["redaction"], REDACTION_OFF
        )


class DigestMeaningTests(RedactionFixture):
    """Two digests, and neither quietly becomes the other."""

    def test_the_callers_digest_is_still_checked_against_the_raw_text(self) -> None:
        archive = RuntimeHandleArchive(self.sidecar)
        with self.assertRaises(PersistenceError):
            archive.persist(
                chatbot_scope(), alias="O1", offload_order=1,
                command_name="execute_workflow_query", step_index=1,
                text=response_with_credential(),
                text_sha256=hashlib.sha256(b"something else").hexdigest(),
            )

    def test_the_stored_digest_covers_the_stored_bytes(self) -> None:
        self.plant_env_secret()
        archive, stored = self.persist(response_with_credential())
        self.assertEqual(
            stored["text_sha256"],
            hashlib.sha256(stored["text"].encode("utf-8")).hexdigest(),
        )
        # A read verifies the row against that column, so a redacted row is
        # readable rather than a permanent digest failure.
        self.assertIsNotNone(archive.get(chatbot_scope(), "O1"))
        self.assertEqual(len(archive.list(chatbot_scope())), 1)

    def test_the_raw_digest_still_means_the_raw_response(self) -> None:
        """Through the real archiver, which is where both digests are made."""
        self.plant_env_secret()
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.sidecar)
        text = response_with_credential()
        trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_connector"},
            "observation_0": text,
        }
        archived = archive_execute_observations(
            trajectory, scope=scope, selected_archive=archive
        )
        raw_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual([row["alias"] for row in archived], ["O1"])
        # The archiver's own record of "this text is already written" is the
        # digest of what the COMMAND returned, unchanged by redaction. It is an
        # idempotence key; computing it from the stored bytes would make the
        # archiver rewrite every step at every step.
        self.assertEqual(archived[0]["text_sha256"], raw_digest)
        self.assertEqual(archived_digest(scope, "O1"), raw_digest)
        # The archive's own digest covers what the archive kept.
        stored = archive.get(scope, "O1")
        self.assertNotEqual(stored["text_sha256"], raw_digest)
        self.assertEqual(
            stored["text_sha256"],
            hashlib.sha256(stored["text"].encode("utf-8")).hexdigest(),
        )
        # And the hot cache holds what the archive kept, so one alias reads the
        # same way whether it is served from memory or from SQLite.
        hot = stored_handles(scope)["O1"]
        self.assertEqual(hot["text"], stored["text"])
        self.assertEqual(hot["text_sha256"], stored["text_sha256"])
        self.assertNotIn(SK_TOKEN.encode("ascii"), self.file_bytes())

    def test_with_redaction_off_every_digest_is_the_same_digest(self) -> None:
        os.environ[REDACTION_ENV] = REDACTION_OFF
        scope = chatbot_scope()
        archive = RuntimeHandleArchive(self.sidecar)
        text = response_with_credential()
        trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_connector"},
            "observation_0": text,
        }
        archive_execute_observations(
            trajectory, scope=scope, selected_archive=archive
        )
        raw_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(archived_digest(scope, "O1"), raw_digest)
        self.assertEqual(archive.get(scope, "O1")["text_sha256"], raw_digest)
        self.assertEqual(stored_handles(scope)["O1"]["text_sha256"], raw_digest)


@unittest.skipIf(erasure is None, "observation_offloading.erasure is not in this revision")
class ErasureStillWorksTests(RedactionFixture):
    """The toggle changes exposure duration, not erasability."""

    def test_a_redacted_row_and_its_fidelity_record_are_erased_together(self) -> None:
        scope = chatbot_scope("erase")
        self.persist(response_with_credential(), scope=scope)
        archive = RuntimeHandleArchive(self.sidecar)
        self.assertIsNotNone(capture_record(archive, scope, "O1"))

        deleted = erasure.forget_channel(self.sidecar, "erase")

        self.assertEqual(deleted["observation_offload_handles"], 1)
        # Discovered structurally, with no name in the erasure module: a record
        # of what was kept must not outlive what it describes.
        self.assertEqual(deleted["observation_capture_policy"], 1)
        self.assertIsNone(archive.get(scope, "O1"))
        self.assertIsNone(capture_record(archive, scope, "O1"))
        blob = self.file_bytes()
        self.assertNotIn(b"okta-prod", blob)
        self.assertNotIn(scope.scope_id.encode("ascii"), blob)

    def test_an_experiment_scope_is_still_preserved(self) -> None:
        kept = experiment_scope("exp-channel")
        erasable = chatbot_scope("chat-channel", turn="turn-2")
        self.persist(response_with_credential(), scope=kept)
        self.persist(response_with_credential(), scope=erasable, alias="O1")

        deleted = erasure.forget_all_channels(self.sidecar)

        self.assertEqual(deleted["preserved_scopes"], 1)
        archive = RuntimeHandleArchive(self.sidecar)
        self.assertIsNotNone(archive.get(kept, "O1"))
        self.assertIsNotNone(capture_record(archive, kept, "O1"))
        self.assertIsNone(archive.get(erasable, "O1"))

    def test_retention_ages_a_redacted_scope_by_its_fidelity_record_too(self) -> None:
        """The new table carries a timestamp, so it cannot orphan a scope."""
        scope = chatbot_scope("old")
        self.persist(response_with_credential(), scope=scope)
        with sqlite3.connect(self.sidecar) as conn:
            for table in ("observation_offload_handles", "observation_capture_policy"):
                column = (
                    "persisted_at" if table.endswith("handles") else "recorded_at"
                )
                conn.execute(f'UPDATE "{table}" SET "{column}"=?',
                             ("2000-01-01T00:00:00Z",))
            conn.commit()

        deleted = erasure.prune(
            self.sidecar, retention_days=30, max_bytes=1_000_000_000
        )

        self.assertEqual(deleted["scopes"], 1)
        self.assertEqual(deleted["observation_capture_policy"], 1)
        archive = RuntimeHandleArchive(self.sidecar)
        self.assertIsNone(archive.get(scope, "O1"))
        self.assertIsNone(capture_record(archive, scope, "O1"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
