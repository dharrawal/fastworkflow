"""Channel erasure and retention reach the offload sidecar.

Every execute response is persisted in
``<observability.sqlite3>.offload-handles.sqlite3``. An erasure path that
deleted a channel's turn from the main store and left that file untouched
would leave the channel's complete responses -- and the ``scope_json`` naming
the channel -- fully recoverable, with no retention knob reaching the file at
all.

Everything here runs against databases created in this test's own temporary
directory. Nothing in this module reads or writes a store it did not create.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from fastworkflow.observability import store as obs
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.state import reset_runtime_state
from fastworkflow.run_chatbot.server import run_forget_channel

try:  # The module under test.
    from fastworkflow.observation_offloading import erasure
except ImportError:  # pragma: no cover - only on a revision before ido-gls
    # Deliberate: a regression check has to be runnable against the revision
    # that had the defect. The cases that go through the PUBLIC erasure and
    # retention paths then fail there on their assertions, which is the proof
    # that they test the fix; only the cases that address this module's own
    # policy surface skip.
    erasure = None

#: The sidecar's name, spelled out here rather than imported, for the same
#: reason: this file must mean the same thing on either revision.
SIDECAR_SUFFIX = ".offload-handles.sqlite3"

#: Every scope-keyed table the sidecar is known to hold today. The production
#: code discovers them structurally; this list exists so a table added without
#: a ``scope_id`` column -- which would slip past that discovery -- is noticed
#: here instead of in a breach report.
KNOWN_EVIDENCE_TABLES = {
    "observation_offload_handles": "persisted_at",
    "result_handle_declarations": "declared_at",
    # (fix-iq53.2.9, F5a) Both renamed, and both old names kept: this dict is
    # intersected with the tables a revision actually created, so an entry for a
    # table this revision does not create simply sits idle. Keeping the old names
    # is what lets these cases still run against the revision each table arrived
    # in, which is the property the file's header comment claims.
    "result_handle_pages": "fetched_at",
    "result_handle_batches": "fetched_at",
    "result_handle_walks": "recorded_at",
    "result_handle_walk_terminals": "recorded_at",
    "result_handle_cursor_tags": "created_at",
    "result_handle_cursors": "issued_at",
    # ido-dhw (F3). Both carry scope_id and scope_json, so both are discovered
    # structurally and erased with their channel without erasure.py knowing
    # their names -- which is exactly the property this constant checks.
    "observation_subjects": "recorded_at",
    "observation_context_entries": "recorded_at",
    # ido-zlm. The fidelity record of each archived observation: which capture
    # policy produced its stored bytes, and whether they were redacted. It
    # names its channel like every other evidence row, so it is discovered
    # structurally and goes with the channel -- a record of what was kept must
    # not outlive what it describes.
    "observation_capture_policy": "recorded_at",
    # ido-6sc. Whether an observation's bytes are still the RAW ones the
    # command returned (the turn is in flight, or its process died before it
    # completed) or have been sealed into their redacted form. Scope-keyed for
    # the same reason, and dated by ``opened_at`` rather than ``sealed_at`` so
    # a scope's retention horizon stays the moment its turn began.
    "observation_seal_state": "opened_at",
}

needs_erasure_module = unittest.skipIf(
    erasure is None, "observation_offloading.erasure is not in this revision"
)


def chatbot_scope(channel: str, turn: str = "turn-1") -> RuntimeHandleScope:
    """A scope exactly as ``scope_for_host`` builds it with no experiment claim."""
    return RuntimeHandleScope(
        store_identity="store",
        channel_id=channel,
        experiment_id="unbound",
        task_id="unbound",
        attempt=0,
        turn_key=turn,
    )


def experiment_scope(
    channel: str, turn: str = "turn-1", experiment: str = "exp-7"
) -> RuntimeHandleScope:
    """A scope as it is built when the session carries an experiment claim."""
    return RuntimeHandleScope(
        store_identity="store",
        channel_id=channel,
        experiment_id=experiment,
        task_id="task-3",
        attempt=1,
        turn_key=turn,
    )


def rows(marker: str, count: int = 30) -> list[str]:
    return ["%s-%03d  confidential row %d" % (marker, i, i) for i in range(count)]


def present_tables(conn: sqlite3.Connection) -> set[str]:
    """The tables this revision actually created in the file."""
    return {
        name
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def counts(db_path: str, scope_id: str | None = None) -> dict[str, int]:
    """Row counts per evidence table, optionally for one scope."""
    with sqlite3.connect(db_path) as conn:
        names = sorted(
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            if name in KNOWN_EVIDENCE_TABLES
        )
        clause = " WHERE scope_id=?" if scope_id else ""
        args = (scope_id,) if scope_id else ()
        return {
            name: conn.execute(
                f"SELECT count(*) FROM {name}{clause}", args
            ).fetchone()[0]
            for name in names
        }


class EvidenceFixture(unittest.TestCase):
    """A sidecar populated through the real writers, in a temporary directory."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(reset_runtime_state)
        self._restore_env: dict[str, str | None] = {}
        for name in ("FW_OFFLOAD_EVIDENCE_PRESERVATION", "FW_OFFLOAD_EVENTS"):
            self._restore_env[name] = os.environ.pop(name, None)
        self.addCleanup(self._restore_environment)
        self.db_path = os.path.join(self.temp.name, "observability.sqlite3")
        self.sidecar = self.db_path + SIDECAR_SUFFIX

    def _restore_environment(self) -> None:
        for name, value in self._restore_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # -- population ------------------------------------------------------

    def populate(
        self, scope: RuntimeHandleScope, marker: str, *, sidecar: str | None = None
    ) -> None:
        """One turn's worth of evidence, written by the production writers.

        An archived observation, its recorded subject and a context entry --
        one row in each of the tables a turn can still reach.
        """
        path = sidecar or self.sidecar
        text = (
            "Observation O1 (execute_workflow_query)\n"
            + "\n".join(rows(marker))
        )
        archive = RuntimeHandleArchive(path)
        # ido-dhw (F3): the cold-restart records are evidence too. The clause
        # names the subject of a listing and the entry names an instance the
        # turn opened, so both carry exactly the kind of text a deletion
        # request is about and both must go with the channel.
        archive.put_subject(scope, "O1", "Fixture " + marker)
        archive.put_context_entry(
            scope, sequence=1, context="Fixture", command_name="open_fixture",
            parameters={"uid": marker}, alias="O1",
            required_parameters=("uid",),
        )
        archive.persist(
            scope,
            alias="O1",
            offload_order=1,
            command_name="execute_workflow_query",
            step_index=1,
            text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def seed_turn(self, scope: RuntimeHandleScope) -> None:
        """The main-store turn record that names the same channel."""
        store = obs.ObservabilityStore(self.db_path)
        row = dict(
            turn_key=scope.turn_key, channel_id=scope.channel_id,
            conversation_id=None, ordinal=None, user_message="fixture",
            refined_user_message=None, entry_workflow_name="fixture",
            entry_context="", status="completed", success=1,
            failure_reason=None, answer="fixture answer",
            conversation_summary=None, conversation_traces=None,
            started_at=None, completed_at=None, suspended_ms=0,
            continuation_of=None, record_version=1, record_json="{}",
        )
        with store._connect() as conn:
            self.assertTrue(store.upsert_turn_row(conn, row, [], obs.Redactor()))

    def age_rows(self, days: int, *, scope_id: str | None = None) -> None:
        """Backdate every timestamp column, for one scope or for all of them."""
        stamp = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        with sqlite3.connect(self.sidecar) as conn:
            present = present_tables(conn)
            for table, column in KNOWN_EVIDENCE_TABLES.items():
                # A table this revision does not create is simply absent, which
                # is what keeps these cases runnable against the revision each
                # table arrived in (``observation_seal_state`` is the newest).
                if table not in present:
                    continue
                clause = " WHERE scope_id=?" if scope_id else ""
                conn.execute(
                    f'UPDATE "{table}" SET "{column}"=?{clause}',
                    (stamp, scope_id) if scope_id else (stamp,),
                )
            conn.commit()

    def assert_readable(self, scope: RuntimeHandleScope, marker: str):
        """This scope's evidence is still there and still usable."""
        archive = RuntimeHandleArchive(self.sidecar)
        recovered = archive.get(scope, "O1")
        self.assertIsNotNone(recovered)
        self.assertIn(marker, recovered["text"])
        # ido-dhw: the cold-restart records survive with everything else, so a
        # preserved turn can still say whose listing O1 was and still resolve
        # the handle of the instance it opened.
        self.assertEqual(archive.get_subject(scope, "O1"), "Fixture " + marker)
        self.assertEqual(len(archive.list_context_entries(scope)), 1)

    def assert_erased(self, scope: RuntimeHandleScope, marker: str):
        """Nothing of this scope is left anywhere in the file."""
        self.assertEqual(
            set(counts(self.sidecar, scope.scope_id).values()), {0},
            counts(self.sidecar, scope.scope_id),
        )
        self.assertIsNone(RuntimeHandleArchive(self.sidecar).get(scope, "O1"))
        with open(self.sidecar, "rb") as handle:
            blob = handle.read()
        self.assertNotIn(marker.encode("utf-8"), blob)
        self.assertNotIn(scope.scope_id.encode("ascii"), blob)


class ChannelErasureTests(EvidenceFixture):
    """Forgetting a channel takes its evidence with it, and only its own."""

    def test_public_forget_channel_erases_the_channels_sidecar_evidence(self):
        erased = chatbot_scope("erase")
        kept = chatbot_scope("keep", turn="turn-2")
        self.populate(erased, "confidential-erase")
        self.populate(kept, "confidential-keep")
        self.seed_turn(erased)
        self.seed_turn(kept)
        before = counts(self.sidecar, erased.scope_id)
        # Every known evidence table THIS revision creates: a table that
        # arrived later is absent rather than empty, which is what keeps this
        # case runnable against the revision each one arrived in.
        with sqlite3.connect(self.sidecar) as conn:
            known_present = sorted(set(KNOWN_EVIDENCE_TABLES) & present_tables(conn))
        self.assertEqual(sorted(before), known_present)
        self.assertTrue(all(value == 1 for value in before.values()), before)

        deleted = run_forget_channel(self.db_path, "erase")

        # The claim first, in the terms a deletion request is made in: none of
        # this channel's evidence is left anywhere in the file.
        self.assert_erased(erased, "confidential-erase")
        self.assert_readable(kept, "confidential-keep")
        self.assertEqual(deleted["turns"], 1)
        self.assertEqual(deleted["offload_scopes"], 1)
        self.assertEqual(deleted["offload_preserved_scopes"], 0)
        for table in KNOWN_EVIDENCE_TABLES:
            if f"offload_{table}" not in deleted:
                continue  # a table this revision does not create
            self.assertEqual(deleted[f"offload_{table}"], 1, table)
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(
                [row[0] for row in conn.execute("SELECT channel_id FROM turns")],
                ["keep"],
            )

    @needs_erasure_module
    def test_every_scope_keyed_table_is_discovered_not_listed(self):
        self.populate(chatbot_scope("erase"), "marker")
        with sqlite3.connect(self.sidecar) as conn:
            discovered = erasure.evidence_tables(conn)
            known_present = set(KNOWN_EVIDENCE_TABLES) & present_tables(conn)
            # A table added later -- as result_handle_walks was in 173e14b --
            # is erased with the channel without this module being edited.
            conn.execute(
                "CREATE TABLE result_handle_future ("
                "scope_id TEXT NOT NULL, scope_json TEXT NOT NULL, "
                "payload TEXT NOT NULL, recorded_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO result_handle_future VALUES (?,?,?,?)",
                (
                    chatbot_scope("erase").scope_id,
                    json.dumps({"channel_id": "erase", "experiment_id": "unbound"}),
                    "secret", "2020-01-01T00:00:00Z",
                ),
            )
            conn.commit()
        self.assertEqual(
            set(discovered), set(KNOWN_EVIDENCE_TABLES) & known_present
        )
        # The six `result_handle_*` tables went with the result-handle package
        # and are simply absent now; `KNOWN_EVIDENCE_TABLES` is intersected with
        # what the revision created, so their entries sit idle. Discovery is
        # unchanged, which is what `result_handle_future` below demonstrates:
        # a table this module has never heard of is found by its `scope_id`
        # column, and erasure.py was not edited for the removal either.
        self.assertIn("observation_subjects", discovered)
        self.assertEqual(
            discovered["observation_subjects"]["timestamp"], "recorded_at")

        deleted = erasure.forget_channel(self.sidecar, "erase")

        self.assertEqual(deleted["result_handle_future"], 1)

    def test_clear_conversations_reaches_the_sidecar(self):
        first = chatbot_scope("one")
        second = chatbot_scope("two", turn="turn-2")
        self.populate(first, "confidential-one")
        self.populate(second, "confidential-two")
        deleted = obs.ObservabilityStore(self.db_path).clear_conversations()
        self.assert_erased(first, "confidential-one")
        self.assert_erased(second, "confidential-two")
        self.assertEqual(deleted["offload_scopes"], 2)

    def test_an_absent_sidecar_reports_nothing_rather_than_creating_one(self):
        self.seed_turn(chatbot_scope("erase"))
        deleted = run_forget_channel(self.db_path, "erase")
        self.assertFalse([key for key in deleted if key.startswith("offload_")])
        self.assertFalse(os.path.exists(self.sidecar))


class PreservationTests(EvidenceFixture):
    """The mode: experiment evidence is preserved, and ambiguity preserves."""

    @needs_erasure_module
    def test_the_signal_is_the_persisted_experiment_id(self):
        self.assertIs(
            erasure.is_experiment_scope('{"experiment_id":"unbound"}'), False
        )
        self.assertIs(erasure.is_experiment_scope('{"experiment_id":""}'), False)
        self.assertIs(erasure.is_experiment_scope('{"experiment_id":"exp-7"}'), True)
        # Fail safe: absent, silent or malformed is not a licence to delete.
        for unknown in ("", "   ", "not json", "{}", '{"channel_id":"c"}', None):
            self.assertIsNone(erasure.is_experiment_scope(unknown), unknown)
            self.assertFalse(erasure.scope_is_erasable(unknown), unknown)

    def test_an_experiment_run_survives_a_forget_aimed_at_another_channel(self):
        experiment = experiment_scope("measured")
        self.populate(experiment, "experiment-evidence")
        self.populate(chatbot_scope("erase", turn="turn-2"), "confidential-erase")
        self.seed_turn(experiment)
        self.seed_turn(chatbot_scope("erase", turn="turn-2"))

        run_forget_channel(self.db_path, "erase")

        # Every one of its six rows is still there -- checked before the read,
        # because reading a further page legitimately writes one more.
        self.assertTrue(
            all(
                value == 1
                for value in counts(self.sidecar, experiment.scope_id).values()
            ),
            counts(self.sidecar, experiment.scope_id),
        )
        self.assert_readable(experiment, "experiment-evidence")

    def test_an_experiment_run_survives_a_forget_of_its_own_channel(self):
        experiment = experiment_scope("measured")
        self.populate(experiment, "experiment-evidence")
        self.seed_turn(experiment)

        deleted = run_forget_channel(self.db_path, "measured")

        # The main store's turn record goes -- that erasure is unchanged and is
        # the operator's to ask for -- but the measurement's evidence does not.
        self.assert_readable(experiment, "experiment-evidence")
        self.assertEqual(deleted["turns"], 1)
        self.assertEqual(deleted["offload_scopes"], 0)
        self.assertEqual(deleted["offload_preserved_scopes"], 1)

    def test_an_experiment_run_survives_a_retention_pass(self):
        experiment = experiment_scope("measured")
        self.populate(experiment, "experiment-evidence")
        chatbot = chatbot_scope("chat", turn="turn-2")
        self.populate(chatbot, "confidential-chat")
        self.age_rows(400)

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=1, max_bytes=1
        )

        self.assert_erased(chatbot, "confidential-chat")
        self.assert_readable(experiment, "experiment-evidence")
        self.assertEqual(deleted["offload_preserved_scopes"], 1)
        self.assertEqual(deleted["offload_scopes"], 1)
        # Over the cap and staying there: the bytes left are preserved.
        self.assertEqual(deleted["offload_over_cap"], 1)

    @needs_erasure_module
    def test_a_scope_whose_json_cannot_be_read_is_preserved(self):
        scope = chatbot_scope("erase")
        self.populate(scope, "unclassifiable")
        with sqlite3.connect(self.sidecar) as conn:
            conn.execute("UPDATE observation_offload_handles SET scope_json='{'")
            conn.commit()
        self.age_rows(400)

        forgotten = erasure.forget_channel(self.sidecar, "erase")
        pruned = erasure.prune(self.sidecar, retention_days=1, max_bytes=1)

        self.assertEqual(forgotten["scopes"], 0)
        self.assertEqual(pruned["scopes"], 0)
        self.assertEqual(pruned["preserved_scopes"], 1)
        self.assertTrue(
            all(
                value == 1
                for value in counts(self.sidecar, scope.scope_id).values()
            ),
            counts(self.sidecar, scope.scope_id),
        )

    @needs_erasure_module
    def test_preserve_all_mode_stops_erasure_and_retention_entirely(self):
        scope = chatbot_scope("erase")
        self.populate(scope, "confidential-erase")
        self.age_rows(400)
        os.environ["FW_OFFLOAD_EVIDENCE_PRESERVATION"] = erasure.PRESERVE_ALL

        self.assertEqual(
            erasure.forget_channel(self.sidecar, "erase")["file_preserved"], 1
        )
        self.assertEqual(
            erasure.prune(self.sidecar, retention_days=1, max_bytes=1)[
                "file_preserved"
            ],
            1,
        )
        self.assert_readable(scope, "confidential-erase")

    @needs_erasure_module
    def test_the_sentinel_file_preserves_a_whole_store(self):
        scope = chatbot_scope("erase")
        self.populate(scope, "confidential-erase")
        self.age_rows(400)
        open(erasure.preserve_sentinel_path(self.sidecar), "w").close()

        self.assertEqual(
            erasure.forget_channel(self.sidecar, "erase")["file_preserved"], 1
        )
        self.assertEqual(
            erasure.prune(self.sidecar, retention_days=1, max_bytes=1)[
                "file_preserved"
            ],
            1,
        )
        self.assert_readable(scope, "confidential-erase")

    @needs_erasure_module
    def test_preserve_none_lets_an_operator_erase_experiment_evidence(self):
        scope = experiment_scope("measured")
        self.populate(scope, "experiment-evidence")
        os.environ["FW_OFFLOAD_EVIDENCE_PRESERVATION"] = erasure.PRESERVE_NONE

        deleted = erasure.forget_channel(self.sidecar, "measured")

        self.assertEqual(deleted["scopes"], 1)
        self.assert_erased(scope, "experiment-evidence")

    @needs_erasure_module
    def test_an_unrecognised_mode_falls_back_to_the_documented_default(self):
        os.environ["FW_OFFLOAD_EVIDENCE_PRESERVATION"] = "yes-please"
        self.assertEqual(erasure.preservation_mode(), erasure.PRESERVE_EXPERIMENTS)


class RetentionTests(EvidenceFixture):
    """Age and size retention reach the sidecar, one whole turn at a time."""

    def test_the_horizon_drops_an_old_turn_whole_and_keeps_a_recent_one(self):
        old = chatbot_scope("chat", turn="turn-old")
        recent = chatbot_scope("chat", turn="turn-recent")
        self.populate(old, "confidential-old")
        self.populate(recent, "confidential-recent")
        self.age_rows(400, scope_id=old.scope_id)

        deleted = obs.ObservabilityStore(self.db_path).prune(
            retention_days=30, max_bytes=1_000_000_000
        )

        self.assert_erased(old, "confidential-old")
        self.assert_readable(recent, "confidential-recent")
        self.assertEqual(deleted["offload_scopes"], 1)
        self.assertEqual(deleted["offload_size_scopes"], 0)
        for table in KNOWN_EVIDENCE_TABLES:
            if f"offload_{table}" not in deleted:
                continue  # a table this revision does not create
            self.assertEqual(deleted[f"offload_{table}"], 1, table)

    @needs_erasure_module
    def test_a_turn_is_aged_by_its_earliest_row_so_it_goes_whole(self):
        scope = chatbot_scope("chat")
        self.populate(scope, "confidential")
        # Only the archived observation is old: it is the first row a turn
        # writes, so the turn began before the horizon and goes entirely.
        with sqlite3.connect(self.sidecar) as conn:
            conn.execute(
                "UPDATE observation_offload_handles SET persisted_at=?",
                ("2000-01-01T00:00:00Z",),
            )
            conn.commit()

        deleted = erasure.prune(
            self.sidecar, retention_days=1, max_bytes=1_000_000_000
        )

        self.assertEqual(deleted["scopes"], 1)
        self.assertEqual(set(counts(self.sidecar).values()), {0})

    @needs_erasure_module
    def test_the_size_cap_evicts_the_oldest_turns_first(self):
        scopes = [chatbot_scope("chat", turn=f"turn-{index}") for index in range(3)]
        for index, scope in enumerate(scopes):
            self.populate(scope, f"confidential-{index}")
            self.age_rows(100 - index, scope_id=scope.scope_id)

        deleted = erasure.prune(self.sidecar, retention_days=3650, max_bytes=1)

        self.assertEqual(deleted["scopes"], 0)
        self.assertEqual(deleted["size_scopes"], 3)
        self.assertEqual(set(counts(self.sidecar).values()), {0})

    def test_retention_does_not_create_a_sidecar_for_a_workflow_without_one(self):
        deleted = obs.ObservabilityStore(self.db_path).prune(retention_days=1)
        self.assertEqual(set(deleted), {"spans", "artifacts"})
        self.assertFalse(os.path.exists(self.sidecar))


class EventLogErasureTests(EvidenceFixture):
    """The plaintext event sink is part of the channel, so erasure reaches it."""

    def events_file(self, *entries: dict) -> str:
        path = os.path.join(self.temp.name, "offload-events.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        os.environ["FW_OFFLOAD_EVENTS"] = path
        return path

    @needs_erasure_module
    def test_forgetting_a_channel_removes_its_lines_and_keeps_the_others(self):
        erased = chatbot_scope("erase")
        kept = chatbot_scope("keep", turn="turn-2")
        self.populate(erased, "confidential-erase")
        self.populate(kept, "confidential-keep")
        path = self.events_file(
            {"kind": "search_memory", "scope_id": erased.scope_id,
             "question": "what is the secret", "answer": "confidential-erase"},
            {"kind": "search_memory", "scope_id": kept.scope_id,
             "question": "keep this", "answer": "confidential-keep"},
            {"kind": "result_handle_hot_evict", "walks": 1},
            "not json at all",
        )

        deleted = erasure.forget_channel(self.sidecar, "erase")

        self.assertEqual(deleted["events_removed"], 1)
        remaining = open(path, encoding="utf-8").read()
        self.assertNotIn("confidential-erase", remaining)
        self.assertNotIn(erased.scope_id, remaining)
        # A line this module cannot attribute is kept, by the same rule that
        # keeps an unattributable row.
        self.assertIn("confidential-keep", remaining)
        self.assertIn("result_handle_hot_evict", remaining)
        self.assertIn("not json at all", remaining)

    def events_file_lines(self, path: str) -> int:
        return len(open(path, encoding="utf-8").read().splitlines())

    @needs_erasure_module
    def test_no_event_log_is_not_an_erasure_failure(self):
        self.populate(chatbot_scope("erase"), "confidential-erase")
        os.environ.pop("FW_OFFLOAD_EVENTS", None)
        deleted = erasure.forget_channel(self.sidecar, "erase")
        self.assertEqual(deleted["scopes"], 1)
        self.assertEqual(deleted["events_removed"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
