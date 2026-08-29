"""Phase 2 observability: SQLite store + background writer (bead fix-kw7.3).

Store contract tests run against real SQLite in tmp_path (design §6 — no
mocks for stores/serialization); the WEC end-to-end tests reuse the fixture
patterns of tests/test_tracing_phase1.py (real todo_list_workflow, fakes only
at the NLU/agent boundary).
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow import TurnStatus, tracing
from fastworkflow import observability_store as obs
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


@pytest.fixture
def sink(db_path):
    s = obs.SQLiteTraceSink(db_path)
    yield s
    s.close()


def _make_assistant_ctx(todo_workflow_path, monkeypatch, sink):
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"obs-assist-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(wf)

    def fake_invoke(cls, session, command: str):
        return fastworkflow.CommandOutput(
            command_name=command.split()[0] if command else "",
            command_response=fastworkflow.CommandResponse(response=f"ok:{command}"),
        )

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(fake_invoke))
    return ctx, wf


def _make_agent_ctx(todo_workflow_path, monkeypatch, sink):
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"obs-agent-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do", lambda session: "commands"
    )
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        ctx,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )
    return ctx, wf


def _rows(db_path: str, query: str, params=()):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Schema, versioning, file posture
# ----------------------------------------------------------------------


class TestSchema:
    def test_schema_created_with_version_and_vacuum(self, db_path):
        store = obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
            # 2 = INCREMENTAL [R12], set at creation before any table
            assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()
        assert {
            "conversations",
            "turns",
            "feedback",
            "spans",
            "artifacts",
            "train_runs",
            "diagnostics",
        } <= tables
        assert store.db_size_bytes() > 0

    def test_file_posture(self, db_path):
        obs.ObservabilityStore(db_path)
        mode = stat.S_IMODE(os.stat(db_path).st_mode)
        assert mode == 0o600  # [R4]
        dir_mode = stat.S_IMODE(os.stat(os.path.dirname(db_path)).st_mode)
        assert dir_mode == 0o700

    def test_refuses_newer_schema(self, db_path):
        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()
        with pytest.raises(obs.IncompatibleObservabilityDB):
            obs.ObservabilityStore(db_path)  # [R11]


# ----------------------------------------------------------------------
# Identity: conversation-id minting [R1] and labels [R15]
# ----------------------------------------------------------------------


class TestConversationIdentity:
    def test_mint_is_sequential_per_channel(self, db_path):
        store = obs.ObservabilityStore(db_path)
        assert store.mint_conversation_id("chan-a") == 1
        assert store.mint_conversation_id("chan-a") == 2
        assert store.mint_conversation_id("chan-b") == 1
        assert store.mint_conversation_id("chan-a") == 3

    def test_record_conversation_label_upserts(self, db_path):
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan-a")
        store.record_conversation_label("chan-a", conv, "Groceries", "Bought milk")
        store.record_conversation_label("chan-a", conv, "Groceries v2", "Bought more")
        rows = store.list_conversations("chan-a")
        assert len(rows) == 1
        assert rows[0]["topic"] == "Groceries v2"
        # Label for a conversation the store never minted still lands [R15]
        store.record_conversation_label("chan-c", 5, "Restored", "s")
        assert store.list_conversations("chan-c")[0]["conversation_id"] == 5


# ----------------------------------------------------------------------
# End-to-end: WEC turn -> DB rows
# ----------------------------------------------------------------------


class TestEndToEnd:
    def test_turn_and_spans_written(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch, db_path, sink
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink)
        conv = sink.store.mint_conversation_id("chan-e2e")
        ctx.bind_observability_identity(channel_id="chan-e2e", conversation_id=conv)

        turn_output = ctx.process_turn("add_todo buy milk")
        assert sink.flush()

        turns = _rows(db_path, "SELECT * FROM turns WHERE turn_key=?", (turn_output.turn_key,))
        assert len(turns) == 1
        row = turns[0]
        assert row["status"] == "completed"
        assert row["success"] == 1
        assert row["channel_id"] == "chan-e2e"
        assert row["conversation_id"] == conv
        assert row["ordinal"] == 1  # store-assigned [R1]
        assert row["user_message"] == "add_todo buy milk"
        record = json.loads(row["record_json"])
        assert record["turn_output"]["turn_key"] == turn_output.turn_key

        spans = _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_output.turn_key,))
        names = {s["name"] for s in spans}
        assert tracing.SPAN_TURN in names
        assert tracing.SPAN_AGENT_TOOL_CALL in names
        root = next(s for s in spans if s["name"] == tracing.SPAN_TURN)
        assert root["end_ns"] is not None  # close upserted over the open emission
        assert root["status"] == "completed"

        # Second turn gets ordinal 2
        second = ctx.process_turn("list_todos")
        assert sink.flush()
        row2 = _rows(db_path, "SELECT ordinal FROM turns WHERE turn_key=?", (second.turn_key,))[0]
        assert row2["ordinal"] == 2

    def test_awaiting_then_terminal_transition(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch, db_path, sink
    ):
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch, sink)
        ctx.bind_observability_identity(channel_id="chan-susp")

        suspended = SimpleNamespace(suspended=True, clarification="Which task?")
        completed = SimpleNamespace(final_answer="All done")
        mock_agent = MagicMock()
        mock_agent.return_value = suspended
        mock_agent.resume.return_value = completed
        ctx._workflow_tool_agent = mock_agent
        ctx._intent_clarification_agent = MagicMock()

        first = ctx.process_turn("clean up")
        assert sink.flush()
        row = _rows(db_path, "SELECT status, success FROM turns WHERE turn_key=?", (first.turn_key,))[0]
        assert row["status"] == "awaiting_user"  # INSERT at first emission [R2]
        assert row["success"] == 0

        second = ctx.process_turn("the urgent one")
        assert second.turn_key == first.turn_key
        assert sink.flush()
        row = _rows(db_path, "SELECT status, success FROM turns WHERE turn_key=?", (first.turn_key,))[0]
        assert row["status"] == "completed"  # guarded transition [R2]

        # fw.ask_user span closed with the human wait
        ask = _rows(
            db_path,
            "SELECT * FROM spans WHERE trace_id=? AND name=?",
            (first.turn_key, tracing.SPAN_ASK_USER),
        )
        assert len(ask) == 1
        assert ask[0]["end_ns"] is not None
        assert ask[0]["kind"] == "human_wait"

    def test_terminal_row_is_write_once(self, db_path, sink):
        store = sink.store
        redactor = obs.Redactor()
        base = {
            "turn_key": "20260825T000000.000000Z-aaaaaaaaaaaa",
            "channel_id": "c",
            "conversation_id": None,
            "ordinal": None,
            "user_message": "m",
            "refined_user_message": None,
            "entry_workflow_name": "",
            "entry_context": "",
            "status": "completed",
            "success": 1,
            "failure_reason": None,
            "answer": "a",
            "conversation_summary": None,
            "conversation_traces": None,
            "started_at": None,
            "completed_at": None,
            "suspended_ms": 0,
            "continuation_of": None,
            "record_version": 1,
            "record_json": "{}",
        }
        conn = store._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            assert store.upsert_turn_row(conn, dict(base), [], redactor) is True
            # identical retry claims idempotent success
            assert store.upsert_turn_row(conn, dict(base), [], redactor) is True
            # conflicting content against a terminal row is refused
            conflicting = dict(base, record_json='{"x": 1}')
            assert store.upsert_turn_row(conn, conflicting, [], redactor) is False
            conn.commit()
        finally:
            conn.close()
        assert store.get_turn(base["turn_key"])["record_json"] == "{}"


# ----------------------------------------------------------------------
# Span idempotent upsert [R6]
# ----------------------------------------------------------------------


class TestSpanUpsert:
    def test_open_then_close_converges(self, db_path, sink):
        key = "20260825T000000.000000Z-bbbbbbbbbbbb"
        span_id = tracing.root_span_id(key)
        open_span = tracing.Span(
            span_id=span_id, trace_id=key, name="fw.turn", start_ns=100,
            status="open", attributes={"a": 1},
        )
        sink.emit_span(open_span)
        closed = tracing.Span(
            span_id=span_id, trace_id=key, name="fw.turn", start_ns=100,
            end_ns=200, status="completed", attributes={"a": 1, "b": 2},
        )
        sink.emit_span(closed)
        # A late open re-emission must not reopen the closed span
        sink.emit_span(open_span)
        assert sink.flush()

        rows = _rows(db_path, "SELECT * FROM spans WHERE span_id=?", (span_id,))
        assert len(rows) == 1
        assert rows[0]["end_ns"] == 200
        assert rows[0]["status"] == "completed"
        assert json.loads(rows[0]["attributes"])["b"] == 2


# ----------------------------------------------------------------------
# Size policy [R10], redaction [R20], traceback gate
# ----------------------------------------------------------------------


class TestSerializationPolicies:
    def _turn_result(self, artifacts: dict, channel="c"):
        response = fastworkflow.CommandResponse(response="done", artifacts=artifacts)
        output = fastworkflow.CommandOutput(
            command_name="x", command_response=response
        )
        turn_output = fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="done",
            command_outputs=[output],
        )
        return fastworkflow.TurnResult(
            turn_output=turn_output, channel_id=channel, user_message="msg"
        )

    def test_oversized_artifact_offloaded_with_envelope(
        self, db_path, sink, monkeypatch
    ):
        monkeypatch.setenv("FW_OBS_INLINE_ARTIFACT_BYTES", "64")
        big = "y" * 500
        turn_result = self._turn_result({"big_blob": big, "small": "ok"})
        sink.emit_turn_record(turn_result)
        assert sink.flush()

        row = _rows(db_path, "SELECT * FROM turns")[0]
        record = json.loads(row["record_json"])
        artifacts = record["turn_output"]["command_outputs"][0]["command_response"]["artifacts"]
        assert artifacts["small"] == "ok"  # inline below the limit
        envelope = artifacts["big_blob"]
        assert envelope["__fw_artifact_ref__"]
        assert envelope["size"] > 64

        stored = _rows(db_path, "SELECT * FROM artifacts")[0]
        assert stored["artifact_id"] == envelope["__fw_artifact_ref__"]
        assert stored["key"] == "big_blob"
        assert json.loads(stored["inline_value"].decode()) == big

    def test_redaction_of_env_secret_and_shapes(self, db_path, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY_TEST", "hunter2secretvalue")
        sink = obs.SQLiteTraceSink(db_path)
        try:
            turn_result = self._turn_result(
                {"leak": "the key is hunter2secretvalue and sk-abcdefghijklmnopqrstu"}
            )
            sink.emit_turn_record(turn_result)
            assert sink.flush()
        finally:
            sink.close()
        row = _rows(db_path, "SELECT record_json FROM turns")[0]
        assert "hunter2secretvalue" not in row["record_json"]
        assert "sk-abcdefghijklmnopqrstu" not in row["record_json"]
        assert "[REDACTED]" in row["record_json"]

    def test_traceback_suppressed_by_default(self, db_path, sink):
        turn_result = self._turn_result({"traceback": "Traceback (most recent...)"})
        sink.emit_turn_record(turn_result)
        assert sink.flush()
        row = _rows(db_path, "SELECT record_json FROM turns")[0]
        assert "most recent" not in row["record_json"]
        assert "FW_OBS_CAPTURE_TRACEBACKS" in row["record_json"]

    def test_unserializable_value_becomes_placeholder(self):
        turn_result = self._turn_result({"weird": object()})
        turn_row, _ = obs.serialize_turn_result(turn_result)
        record = json.loads(turn_row["record_json"])
        artifacts = record["turn_output"]["command_outputs"][0]["command_response"]["artifacts"]
        assert artifacts["weird"]["__fw_unserializable__"] == "object"


# ----------------------------------------------------------------------
# Conversation memory round trip (fix-24f.1)
#
# Every read below filters on a non-NULL conversation_summary. The serializer
# used to hardcode both memory columns to None, which made all of them return
# empty against real traffic while the seeded-row tests above stayed green —
# so these go through emit_turn_record rather than writing rows directly.
# ----------------------------------------------------------------------


class TestConversationMemoryRoundTrip:
    def _turn_result(self, summary, traces, conversation_id=1, channel="c"):
        output = fastworkflow.CommandOutput(
            command_name="x",
            command_response=fastworkflow.CommandResponse(response="done"),
        )
        turn_output = fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="done",
            command_outputs=[output],
        )
        return fastworkflow.TurnResult(
            turn_output=turn_output,
            channel_id=channel,
            conversation_id=conversation_id,
            user_message="msg",
            conversation_summary=summary,
            conversation_traces=traces,
        )

    def test_stamped_turn_is_readable_as_memory(self, db_path, sink):
        sink.emit_turn_record(self._turn_result("first turn", '{"a": 1}'))
        sink.emit_turn_record(self._turn_result("second turn", '{"b": 2}'))
        assert sink.flush()

        store = obs.ObservabilityStore(db_path)
        assert store.count_usable_turns("c", 1) == 2
        window = store.get_memory_window("c", 1, max_turns=10)
        assert [entry["conversation summary"] for entry in window] == [
            "first turn",
            "second turn",
        ]
        assert window[0]["conversation_traces"] == '{"a": 1}'
        assert store.conversation_summaries("c", 1) == [
            {"conversation summary": "first turn"},
            {"conversation summary": "second turn"},
        ]
        assert store.conversation_label_state("c", 1) == ("", 2)
        assert store.get_last_completed_turn_key("c", 1) is not None
        assert [c["conversation_id"] for c in store.list_conversation_summaries("c", 10)] == [1]
        assert len(store.dump_all_conversations("c")[0]["turns"]) == 2

    def test_unstamped_turn_is_a_trace_not_memory(self, db_path, sink):
        """A turn that appended no history entry stays out of every memory read
        even though its row exists (ruling I4's usable-rows invariant)."""
        sink.emit_turn_record(self._turn_result(None, None))
        assert sink.flush()

        store = obs.ObservabilityStore(db_path)
        assert _rows(db_path, "SELECT * FROM turns")  # the row is there
        assert store.count_usable_turns("c", 1) == 0
        assert store.get_memory_window("c", 1, max_turns=10) == []
        assert store.list_conversation_summaries("c", 10) == []

    def test_terminal_emission_fills_columns_an_awaiting_user_row_left_null(
        self, db_path, sink
    ):
        suspended = self._turn_result(None, None)
        suspended.turn_output.status = TurnStatus.AWAITING_USER
        sink.emit_turn_record(suspended)
        assert sink.flush()
        store = obs.ObservabilityStore(db_path)
        assert store.count_usable_turns("c", 1) == 0

        resumed = self._turn_result("the resumed turn", "{}")
        resumed.turn_output.turn_key = suspended.turn_output.turn_key
        sink.emit_turn_record(resumed)
        assert sink.flush()

        assert store.count_usable_turns("c", 1) == 1
        window = store.get_memory_window("c", 1, max_turns=10)
        assert window[0]["conversation summary"] == "the resumed turn"

    def test_feedback_joins_into_the_memory_window(self, db_path, sink):
        turn_result = self._turn_result("a turn with feedback", "{}")
        sink.emit_turn_record(turn_result)
        assert sink.flush()

        store = obs.ObservabilityStore(db_path)
        store.upsert_feedback(
            turn_result.turn_output.turn_key, json.dumps({"nl_feedback": "helpful"})
        )
        window = store.get_memory_window("c", 1, max_turns=10)
        assert window[0]["feedback"] == {"nl_feedback": "helpful"}


# ----------------------------------------------------------------------
# Writer discipline [R13]: drops counted, failures never propagate
# ----------------------------------------------------------------------


class TestWriterDiscipline:
    def test_span_queue_overflow_drops_and_counts(self, db_path, monkeypatch):
        monkeypatch.setenv("FW_OBS_QUEUE_MAX", "1")
        sink = obs.SQLiteTraceSink(db_path)
        try:
            # Stall the writer by holding the DB write lock so the queue fills.
            blocker = sqlite3.connect(db_path, timeout=30.0)
            blocker.execute("BEGIN IMMEDIATE")
            for i in range(50):
                sink.emit_span(
                    tracing.Span(
                        span_id=f"s{i}", trace_id="t", name="fw.agent.tool_call",
                        start_ns=i, status="ok",
                    )
                )
            blocker.rollback()
            blocker.close()
            sink.flush()
        finally:
            sink.close()
        health = obs.ObservabilityStore(db_path).writer_health()
        assert health is not None
        assert health["spans_dropped"] > 0

    def test_store_failure_never_raises_to_emitter(self, db_path, sink, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(sink.store, "upsert_turn_row", broken)
        turn_output = fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(), status=TurnStatus.COMPLETED
        )
        turn_result = fastworkflow.TurnResult(turn_output=turn_output, user_message="m")
        sink.emit_turn_record(turn_result)  # must not raise
        sink.flush()
        health = sink.store.writer_health()
        assert health["write_errors"] > 0
        assert "disk on fire" in (health["last_error"] or "")

    def test_unopenable_db_yields_no_sink_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "root"))
        workflow_path = str(tmp_path / "wf")
        os.makedirs(workflow_path, exist_ok=True)
        db = fastworkflow.state_paths.observability_db(workflow_path)
        obs.ObservabilityStore(db)  # create it
        os.chmod(db, 0o400)  # unwritable -> schema ensure fails
        try:
            assert obs.get_observability_sink(workflow_path) is None
        finally:
            os.chmod(db, 0o600)

    def test_close_drains_pending_writes(self, db_path):
        sink = obs.SQLiteTraceSink(db_path)
        for i in range(20):
            sink.emit_span(
                tracing.Span(
                    span_id=f"c{i}", trace_id="t", name="fw.agent.tool_call",
                    start_ns=i, status="ok",
                )
            )
        sink.close()
        assert len(_rows(db_path, "SELECT * FROM spans")) == 20
        # Emissions after close are dropped silently
        sink.emit_span(
            tracing.Span(span_id="late", trace_id="t", name="fw.turn", start_ns=1, status="open")
        )


# ----------------------------------------------------------------------
# Maintenance [R12] and erasure [R21]
# ----------------------------------------------------------------------


class TestMaintenance:
    def test_prune_deletes_old_spans_keeps_turns(self, db_path, sink):
        old_ns = int((time.time() - 90 * 86_400) * 1_000_000_000)
        sink.emit_span(
            tracing.Span(span_id="old", trace_id="t-old", name="fw.turn", start_ns=old_ns, status="ok")
        )
        sink.emit_span(
            tracing.Span(
                span_id="new", trace_id="t-new", name="fw.turn",
                start_ns=int(time.time() * 1e9), status="ok",
            )
        )
        assert sink.flush()

        deleted = sink.store.prune(retention_days=30)
        assert deleted["spans"] == 1
        remaining = _rows(db_path, "SELECT span_id FROM spans")
        assert [r["span_id"] for r in remaining] == ["new"]

    def test_forget_channel_erases_across_tables(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch, db_path, sink
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink)
        conv = sink.store.mint_conversation_id("chan-erase")
        ctx.bind_observability_identity(channel_id="chan-erase", conversation_id=conv)
        ctx.process_turn("add_todo x")

        ctx2, _wf2 = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink)
        ctx2.bind_observability_identity(channel_id="chan-keep")
        ctx2.process_turn("list_todos")
        assert sink.flush()

        deleted = sink.store.forget_channel("chan-erase")
        assert deleted["turns"] == 1
        assert deleted["conversations"] == 1
        assert deleted["spans"] > 0

        assert _rows(db_path, "SELECT * FROM turns WHERE channel_id='chan-erase'") == []
        assert len(_rows(db_path, "SELECT * FROM turns WHERE channel_id='chan-keep'")) == 1
        assert _rows(db_path, "SELECT * FROM spans WHERE channel_id='chan-erase'") == []


# ----------------------------------------------------------------------
# Factory / FW_OBSERVABILITY gating [R4]
# ----------------------------------------------------------------------


class TestFactory:
    def test_gating(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "root"))
        workflow_path = str(tmp_path / "wf")
        os.makedirs(workflow_path, exist_ok=True)

        monkeypatch.setenv("FW_OBSERVABILITY", "0")
        assert obs.get_observability_sink(workflow_path) is None

        monkeypatch.delenv("FW_OBSERVABILITY", raising=False)
        # Entry points default ON; embedders default OFF
        assert obs.get_observability_sink(workflow_path, entry_point=False) is None
        sink = obs.get_observability_sink(workflow_path)
        try:
            assert sink is not None
            # Cached: same sink per DB path
            assert obs.get_observability_sink(workflow_path) is sink
        finally:
            if sink is not None:
                sink.close()

    def test_train_runs_roundtrip(self, db_path):
        store = obs.ObservabilityStore(db_path)
        store.record_train_run(
            "run-1", "fp", "2026-08-25T00:00:00Z", "2026-08-25T00:10:00Z",
            {"contexts": {"global": {"f1": 0.97}}},
        )
        runs = store.list_train_runs()
        assert len(runs) == 1
        assert json.loads(runs[0]["metrics_json"])["contexts"]["global"]["f1"] == 0.97


# ----------------------------------------------------------------------
# Ruling C2: conversation ids come from a per-channel counter that never
# decreases, so no erasure or prune can cause an id to be reused; and a mint
# that cannot run degrades instead of failing the caller.
#
# The Phase-A `legacy_floor` half of C2 is gone with the legacy store: it
# existed to stop a fresh observability DB re-issuing an id that already named
# one of the channel's per-channel-DB conversations, which mattered only while
# BOTH stores were written. Nothing reads those files now.
# ----------------------------------------------------------------------


class TestMinting:
    def _runtime(self, sink, channel_id):
        bound: dict = {}

        class _Ctx:
            trace_sink = sink

            def bind_observability_identity(self, **kwargs):
                bound.update(kwargs)

        runtime = SimpleNamespace(
            execution_context=_Ctx(),
            channel_id=channel_id,
        )
        return runtime, bound

    def test_reserve_conversation_id_binds_the_minted_id(self, db_path):
        from fastworkflow.run_fastapi_mcp.utils import reserve_conversation_id

        sink = obs.SQLiteTraceSink(db_path)
        try:
            runtime, bound = self._runtime(sink, "chanX")
            assert reserve_conversation_id(runtime) == 1
            assert bound == {"conversation_id": 1}
            assert reserve_conversation_id(runtime) == 2
        finally:
            sink.close()

    def test_an_id_is_never_reused_after_the_channel_is_forgotten(self, db_path):
        """Erasure must not roll the counter back (ruling C2).

        A MAX-derived mint would hand the next conversation an id that names a
        deleted one, so anything still holding the old id — a checkpoint, a
        client's history list — would silently point at the new conversation.
        """
        from fastworkflow.run_fastapi_mcp.utils import reserve_conversation_id

        sink = obs.SQLiteTraceSink(db_path)
        try:
            runtime, _bound = self._runtime(sink, "chanZ")
            first = reserve_conversation_id(runtime)
            second = reserve_conversation_id(runtime)
            sink.store.forget_channel("chanZ")
            after_erasure = reserve_conversation_id(runtime)
        finally:
            sink.close()

        assert (first, second) == (1, 2)
        assert after_erasure > second, (
            f"minting reused id {after_erasure} after the channel was forgotten"
        )

    def test_reserve_degrades_to_zero_when_the_mint_fails(self, db_path, monkeypatch):
        """A wedged DB must not fail /initialize or a turn.

        Zero is the same value a never-reserved channel carries, so every caller
        already handles it as "no active conversation".
        """
        from fastworkflow.run_fastapi_mcp.utils import reserve_conversation_id

        sink = obs.SQLiteTraceSink(db_path)
        try:
            def _wedged(*args, **kwargs):
                raise sqlite3.OperationalError("database is locked")

            monkeypatch.setattr(sink.store, "mint_conversation_id", _wedged)
            runtime, bound = self._runtime(sink, "chanY")
            conv_id = reserve_conversation_id(runtime)  # must not raise
        finally:
            sink.close()

        assert conv_id == 0
        assert bound == {}, "a failed mint bound an id onto the context anyway"


# ----------------------------------------------------------------------
# Gate 2 (§2.4, rulings I1/I6/C8/C9): sync-first turn records
# ----------------------------------------------------------------------


class TestSyncFirstTurnRecords:
    def _turn_result(self, summary="a turn", conversation_id=1, status=None):
        turn_output = fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=status or TurnStatus.COMPLETED,
            answer="ok",
        )
        return fastworkflow.TurnResult(
            turn_output=turn_output,
            channel_id="c",
            conversation_id=conversation_id,
            user_message="msg",
            conversation_summary=summary,
        )

    def test_a_healthy_emit_is_durable_before_it_returns(self, db_path, sink):
        turn_result = self._turn_result()
        assert sink.emit_turn_record(turn_result) is True
        # Deliberately NO flush: the ack promises the row is already there.
        assert obs.ObservabilityStore(db_path).get_turn(
            turn_result.turn_output.turn_key
        ) is not None
        assert sink.pending_retry_depth() == 0

    def test_an_open_breaker_degrades_to_the_queue_and_reports_it(self, db_path, sink):
        sink._sync_breaker_until = time.monotonic() + 300
        turn_result = self._turn_result()
        assert sink.emit_turn_record(turn_result) is False
        assert sink.pending_retry_depth() == 1
        # The row is not durable yet, which is exactly what the ack said.
        assert obs.ObservabilityStore(db_path).get_turn(
            turn_result.turn_output.turn_key
        ) is None
        assert sink.flush()
        assert obs.ObservabilityStore(db_path).get_turn(
            turn_result.turn_output.turn_key
        ) is not None

    def test_a_degraded_record_keeps_its_chronological_ordinal(self, db_path, sink):
        """Ruling I6: the ordinal is reserved synchronously before the enqueue.

        Without that, a turn written while the DB was briefly wedged would sort
        after turns that happened later.
        """
        assert sink.emit_turn_record(self._turn_result("first")) is True
        sink._sync_breaker_until = time.monotonic() + 300
        assert sink.emit_turn_record(self._turn_result("second")) is False
        sink._sync_breaker_until = 0.0
        assert sink.emit_turn_record(self._turn_result("third")) is True
        assert sink.flush()

        window = obs.ObservabilityStore(db_path).get_memory_window("c", 1, 10)
        assert [entry["conversation summary"] for entry in window] == [
            "first",
            "second",
            "third",
        ]

    def test_the_pending_ring_is_bounded(self, db_path, sink):
        sink._sync_breaker_until = time.monotonic() + 300
        for i in range(obs._PENDING_RETRY_MAX + 10):
            sink.emit_turn_record(self._turn_result(f"turn-{i}"))
        assert sink.pending_retry_depth() == obs._PENDING_RETRY_MAX
        health = sink._health
        assert health["records_dropped"] >= 10
        assert health["sync_fallbacks"] >= obs._PENDING_RETRY_MAX

    def test_the_breaker_rearms_only_after_a_successful_probe(self, db_path, sink):
        sink._trip_sync_breaker(RuntimeError("wedged"))
        assert sink._sync_available() is False

        # Cooldown elapsed, but the breaker stays shut until a probe succeeds.
        sink._sync_breaker_until = time.monotonic() - 1
        sink._maybe_rearm_sync_breaker()
        assert sink._sync_available() is True
        assert sink._health["sync_breaker_open"] is False
        assert sink.emit_turn_record(self._turn_result()) is True

    def test_writer_health_records_the_sync_path(self, db_path, sink):
        sink.emit_turn_record(self._turn_result())
        assert sink.flush()
        health = obs.ObservabilityStore(db_path).writer_health()
        assert health is not None
        assert health["sync_writes"] >= 1
        assert health["sync_write_ms_max"] >= 0
        assert health["sync_breaker_open"] is False

    def test_an_awaiting_user_emission_is_also_sync(self, db_path, sink):
        """Ruling I6: awaiting_user and terminal take the SAME path.

        Mixing them was what let one logical turn split across the sync and
        queued paths and produce spurious refused-terminal-write noise.
        """
        suspended = self._turn_result(
            summary=None, status=TurnStatus.AWAITING_USER
        )
        assert sink.emit_turn_record(suspended) is True
        row = obs.ObservabilityStore(db_path).get_turn(
            suspended.turn_output.turn_key
        )
        assert row is not None and row["status"] == "awaiting_user"
        # A suspended row is not a pending-retry obligation: it is not terminal.
        assert sink.pending_retry_depth() == 0


# ----------------------------------------------------------------------
# Distillation storage foundation (distillation design §§8-11, fix-sb8.2 /
# fix-sb8.13 substrate). Real SQLite in tmp_path, no mocks.
# ----------------------------------------------------------------------

_DISTILL_TABLES = (
    "distillation_runs",
    "distillation_passes",
    "distillation_divergences",
    "distillation_insights",
    "distillation_insight_citations",
    "distillation_verdicts",
)

# The v3.2.0 schema, verbatim from before the distillation columns/tables: an
# "old DB" the migration has to upgrade in place.
_PRE_DISTILLATION_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS conversations (
        channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
        topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,
        started_at TEXT, last_turn_at TEXT, updated_at TEXT,
        PRIMARY KEY (channel_id, conversation_id))""",
    """CREATE TABLE IF NOT EXISTS conversation_counters (
        channel_id TEXT PRIMARY KEY, next_id INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS turns (
        turn_key TEXT PRIMARY KEY,
        channel_id TEXT NOT NULL, conversation_id INTEGER, ordinal INTEGER,
        user_message TEXT NOT NULL, refined_user_message TEXT,
        entry_workflow_name TEXT, entry_context TEXT,
        status TEXT NOT NULL, success INTEGER NOT NULL,
        failure_reason TEXT, answer TEXT,
        conversation_summary TEXT, conversation_traces TEXT,
        started_at TEXT, completed_at TEXT, suspended_ms INTEGER,
        continuation_of TEXT, record_version INTEGER NOT NULL,
        record_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS feedback (
        turn_key TEXT PRIMARY KEY, feedback_json TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS spans (
        span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
        parent_span_id TEXT, name TEXT NOT NULL,
        kind TEXT NOT NULL,
        channel_id TEXT,
        command_name TEXT, context TEXT,
        start_ns INTEGER NOT NULL, end_ns INTEGER,
        status TEXT NOT NULL, attributes TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY, turn_key TEXT NOT NULL,
        channel_id TEXT,
        span_id TEXT, key TEXT NOT NULL, content_type TEXT,
        size_bytes INTEGER, sha256 TEXT,
        inline_value BLOB, error TEXT)""",
    """CREATE TABLE IF NOT EXISTS train_runs (
        run_id TEXT PRIMARY KEY, workflow_fingerprint TEXT, started_at TEXT,
        completed_at TEXT, metrics_json TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS diagnostics (
        key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)""",
]


def _write_pre_distillation_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA journal_mode=WAL")
        for statement in _PRE_DISTILLATION_SCHEMA:
            conn.execute(statement)
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            "INSERT INTO spans (span_id, trace_id, name, kind, start_ns, status, "
            "attributes) VALUES ('legacy-span', 'legacy-trace', 'fw.turn', "
            "'internal', 1, 'ok', '{}')"
        )
        conn.execute(
            "INSERT INTO turns (turn_key, channel_id, user_message, status, "
            "success, record_version, record_json) VALUES "
            "('legacy-turn', 'chan-old', 'hi', 'completed', 1, 1, '{}')"
        )
        conn.commit()
    finally:
        conn.close()


def _distill_run_payload(run_id: str, turn_key: str, **overrides):
    payload = {
        "run_id": run_id,
        "turn_key": turn_key,
        "channel_id": "chan-distill",
        "user_message": "add a todo",
        "comparable": 1,
        "started_at": _iso_days_ago(0),
        "run_json": json.dumps({"run_id": run_id}),
    }
    payload.update(overrides)
    return payload


def _iso_days_ago(days: float) -> str:
    from datetime import datetime, timedelta, timezone as _tz

    return (datetime.now(_tz.utc) - timedelta(days=days)).isoformat()


def _seed_run(
    store,
    run_id,
    turn_key,
    *,
    pinned=0,
    diverged=1,
    started_days_ago=0.0,
    with_insight=True,
):
    """One complete run: row, one pass, one divergence, one insight, one
    citation — written the way a live turn writes them ([DR46] shape).

    `with_insight=False` gives the other shape §10.3 distinguishes: a run that
    produced no insight, which is the only thing the negative-pin window is
    allowed to release (`fix-sb8.17`)."""
    redactor = obs.Redactor()
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_distillation_row(
            conn,
            "run",
            _distill_run_payload(
                run_id,
                turn_key,
                pinned=pinned,
                planning_diverged=diverged,
                exec_diverged=diverged,
                started_at=_iso_days_ago(started_days_ago),
            ),
            redactor,
        )
        for label, seq in (("teacher", 0), ("student", 1)):
            store.upsert_distillation_row(
                conn,
                "pass",
                {
                    "run_id": run_id,
                    "pass_label": label,
                    "role": label,
                    "seq": seq,
                    "trace_id": turn_key,
                    "entry_inputs_json": json.dumps({"user_message": "add a todo"}),
                },
                redactor,
            )
        store.upsert_distillation_row(
            conn,
            "divergence",
            {
                "divergence_id": f"{run_id}-d0",
                "run_id": run_id,
                "level": "action",
                "left_pass": "teacher",
                "right_pass": "student",
                "align_index": 0,
                "kind": "same-command-different-params",
                "material": 1,
                "detail_json": json.dumps({"left": {}, "right": {}}),
            },
            redactor,
        )
        if with_insight:
            store.upsert_distillation_row(
                conn,
                "insight",
                {
                    "insight_id": f"ins-{run_id}",
                    "run_id": run_id,
                    "kind": "execution",
                    "text": "prefer the id form",
                    "text_hash": "deadbeef",
                    "created_at": _iso_days_ago(0),
                },
                redactor,
            )
            store.upsert_distillation_row(
                conn,
                "citation",
                {"insight_id": f"ins-{run_id}", "divergence_id": f"{run_id}-d0"},
                redactor,
            )
        conn.commit()


def _seed_spans(store, trace_id, count, *, days_ago=0.0, channel_id=None, base=0):
    now_ns = int((time.time() - days_ago * 86_400) * 1_000_000_000)
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for i in range(count):
            conn.execute(
                "INSERT INTO spans (span_id, trace_id, name, kind, channel_id, "
                "start_ns, status, attributes) VALUES (?, ?, 'fw.turn', "
                "'internal', ?, ?, 'ok', ?)",
                (
                    f"{trace_id}-s{base + i}",
                    trace_id,
                    channel_id,
                    now_ns + base + i,
                    json.dumps({"pad": "x" * 64}),
                ),
            )
        conn.commit()


class TestDistillationSchema:
    def test_six_tables_and_indexes_on_a_fresh_db(self, db_path):
        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            # Queryable, not merely present.
            for table in _DISTILL_TABLES:
                assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
            span_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(spans)").fetchall()
            }
        finally:
            conn.close()
        assert set(_DISTILL_TABLES) <= tables
        assert "distillation_pass" in span_cols
        assert {
            "idx_spans_trace_pass",
            "idx_distill_runs_turn",
            "idx_distill_runs_channel",
            "idx_distill_runs_replay",
            "idx_distill_runs_pinned",
            "idx_distill_passes_run",
            "idx_distill_passes_trace",
            "idx_distill_div_run",
            "idx_distill_div_kind",
            "idx_distill_insights_run",
            "idx_distill_insights_hash",
            "idx_distill_citations_div",
            "idx_distill_verdicts_insight",
        } <= indexes

    def test_old_db_upgraded_in_place_without_a_version_bump(self, db_path):
        """[DR28]: additive migration, SCHEMA_VERSION stays 1, no data loss."""
        _write_pre_distillation_db(db_path)

        pre = obs.ReadOnlyObservabilityStore(db_path)
        assert pre.has_feature("distillation_v1") is False

        store = obs.ObservabilityStore(db_path)
        assert store.has_feature("distillation_v1") is True
        assert obs.SCHEMA_VERSION == 1

        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            span = conn.execute(
                "SELECT span_id, distillation_pass FROM spans"
            ).fetchall()
            turns = conn.execute("SELECT turn_key FROM turns").fetchall()
            features = json.loads(
                conn.execute(
                    "SELECT value FROM diagnostics WHERE key='schema_features'"
                ).fetchone()[0]
            )
        finally:
            conn.close()
        assert set(_DISTILL_TABLES) <= tables
        assert span == [("legacy-span", None)]  # migrated in place, not dropped
        assert turns == [("legacy-turn",)]
        assert "distillation_v1" in features

    def test_schema_features_marker_is_merged_not_overwritten(self, db_path):
        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "UPDATE diagnostics SET value=? WHERE key='schema_features'",
                (json.dumps(["someone_elses_v9"]),),
            )
            conn.commit()
        finally:
            conn.close()
        obs.ObservabilityStore(db_path)
        value = _rows(
            db_path, "SELECT value FROM diagnostics WHERE key='schema_features'"
        )[0]["value"]
        assert set(json.loads(value)) == {"someone_elses_v9", "distillation_v1"}

    def test_has_feature_is_false_for_unknown_names(self, db_path):
        store = obs.ObservabilityStore(db_path)
        assert store.has_feature("distillation_v2") is False


class TestDistillationWritePath:
    def test_records_round_trip_through_the_sink(self, db_path, sink):
        """[DR46]: every kind lands via the existing writer thread."""
        run_id = "run-1"
        sink.emit_distillation_record(
            "run", _distill_run_payload(run_id, "20260101T000000.000000Z-abc")
        )
        sink.emit_distillation_record(
            "pass",
            {
                "run_id": run_id,
                "pass_label": "teacher",
                "role": "teacher",
                "seq": 0,
                "trace_id": "20260101T000000.000000Z-abc",
            },
        )
        sink.emit_distillation_record(
            "divergence",
            {
                "divergence_id": "d-1",
                "run_id": run_id,
                "level": "action",
                "left_pass": "teacher",
                "right_pass": "student",
                "align_index": 0,
                "kind": "missing-in-student",
                "detail_json": "{}",
            },
        )
        sink.emit_distillation_record(
            "insight",
            {
                "insight_id": "ins-1",
                "run_id": run_id,
                "kind": "planning",
                "text": "plan before acting",
                "text_hash": "abc123",
                "created_at": _iso_days_ago(0),
            },
        )
        sink.emit_distillation_record(
            "citation", {"insight_id": "ins-1", "divergence_id": "d-1"}
        )
        assert sink.flush()

        assert _rows(db_path, "SELECT run_id FROM distillation_runs") == [
            {"run_id": run_id}
        ]
        assert len(_rows(db_path, "SELECT * FROM distillation_passes")) == 1
        assert len(_rows(db_path, "SELECT * FROM distillation_divergences")) == 1
        assert len(_rows(db_path, "SELECT * FROM distillation_insights")) == 1
        assert len(_rows(db_path, "SELECT * FROM distillation_insight_citations")) == 1
        health = sink.store.writer_health() or {}
        assert not health.get("write_errors")

    def test_run_row_is_upserted_not_duplicated(self, db_path, sink):
        run_id = "run-upsert"
        sink.emit_distillation_record(
            "run", _distill_run_payload(run_id, "t-k", completed_at=None)
        )
        sink.emit_distillation_record(
            "run",
            {
                "run_id": run_id,
                "turn_key": "t-k",
                "user_message": "add a todo",
                "comparable": 1,
                "run_json": "{}",
                "completed_at": "2026-08-28T00:00:00+00:00",
                "material_divergences": 3,
            },
        )
        assert sink.flush()
        rows = _rows(db_path, "SELECT * FROM distillation_runs")
        assert len(rows) == 1
        assert rows[0]["completed_at"] == "2026-08-28T00:00:00+00:00"
        assert rows[0]["material_divergences"] == 3

    def test_unknown_kind_is_counted_not_raised(self, db_path, sink):
        sink.emit_distillation_record("verdict", {"verdict_id": "v"})
        assert sink.flush()
        health = sink.store.writer_health() or {}
        assert int(health.get("write_errors") or 0) >= 1
        assert _rows(db_path, "SELECT * FROM distillation_verdicts") == []

    def test_a_malformed_record_does_not_lose_its_batch(self, db_path, sink):
        sink.emit_distillation_record("run", {"turn_key": "no-run-id"})
        sink.emit_distillation_record(
            "run", _distill_run_payload("run-ok", "t-ok")
        )
        assert sink.flush()
        assert [r["run_id"] for r in _rows(db_path, "SELECT run_id FROM distillation_runs")] == [
            "run-ok"
        ]

    def test_span_distillation_pass_round_trips(self, db_path, sink):
        """The single edit most likely to be missed: emit_span rebuilds the
        Span field by field, so an omitted field is a silent no-op."""
        sink.emit_span(
            tracing.Span(
                span_id="sp-teacher",
                trace_id="t-1",
                name="fw.command.execute",
                start_ns=int(time.time() * 1e9),
                status="ok",
                distillation_pass="teacher",
            )
        )
        sink.emit_span(
            tracing.Span(
                span_id="sp-plain",
                trace_id="t-1",
                name="fw.command.execute",
                start_ns=int(time.time() * 1e9),
                status="ok",
            )
        )
        assert sink.flush()
        rows = {
            r["span_id"]: r["distillation_pass"]
            for r in _rows(db_path, "SELECT span_id, distillation_pass FROM spans")
        }
        assert rows == {"sp-teacher": "teacher", "sp-plain": None}

    def test_a_close_cannot_relabel_a_spans_pass(self, db_path, sink):
        """distillation_pass is write-once at open: it is not in the upsert's
        DO UPDATE set, like trace_id and parent_span_id."""
        start = int(time.time() * 1e9)
        sink.emit_span(
            tracing.Span(
                span_id="sp-x",
                trace_id="t-1",
                name="fw.ask_user",
                start_ns=start,
                status="open",
                distillation_pass="teacher",
            )
        )
        sink.emit_span(
            tracing.Span(
                span_id="sp-x",
                trace_id="t-1",
                name="fw.ask_user",
                start_ns=start,
                end_ns=start + 10,
                status="ok",
                distillation_pass="student",
            )
        )
        assert sink.flush()
        row = _rows(db_path, "SELECT distillation_pass, status FROM spans")[0]
        assert row["distillation_pass"] == "teacher"
        assert row["status"] == "ok"  # the close still landed


class TestDistillationErasure:
    """[DR44]: the six tables hold verbatim user content, so erasure must
    reach them. Assert row counts, not the absence of an exception."""

    def _seed_channel(self, store, channel_id, run_id, turn_key):
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO turns (turn_key, channel_id, user_message, status, "
                "success, record_version, record_json) VALUES (?, ?, ?, 'completed', "
                "1, 1, '{}')",
                (turn_key, channel_id, "add a todo"),
            )
            conn.commit()
        _seed_run(store, run_id, turn_key)
        _seed_spans(store, turn_key, 2, channel_id=channel_id)
        # A replay trace carries no channel_id of its own [DR41].
        _seed_spans(store, f"{turn_key}~replay.1", 2)
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO distillation_verdicts (verdict_id, insight_id, verdict, "
                "actor, created_at) VALUES (?, ?, 'supported', 'human', ?)",
                (f"v-{run_id}", f"ins-{run_id}", _iso_days_ago(0)),
            )
            conn.commit()

    def test_forget_channel_leaves_no_distillation_row_behind(self, db_path):
        store = obs.ObservabilityStore(db_path)
        self._seed_channel(store, "chan-gone", "run-gone", "20260101T000000.000000Z-a")
        self._seed_channel(store, "chan-stays", "run-stays", "20260101T000001.000000Z-b")

        deleted = store.forget_channel("chan-gone")
        assert deleted["distillation_runs"] == 1

        # _seed_run writes two passes per run and one of everything else.
        expected = {table: 1 for table in _DISTILL_TABLES}
        expected["distillation_passes"] = 2
        for table in _DISTILL_TABLES:
            remaining = _rows(db_path, f"SELECT * FROM {table}")
            assert len(remaining) == expected[table], (
                f"{table} kept the wrong number of rows"
            )
            assert "gone" not in json.dumps(remaining), table
        # The replay trace goes with the erased turn; the other channel's does not.
        traces = {r["trace_id"] for r in _rows(db_path, "SELECT trace_id FROM spans")}
        assert traces == {
            "20260101T000001.000000Z-b",
            "20260101T000001.000000Z-b~replay.1",
        }

    def test_clear_conversations_leaves_no_distillation_row_behind(self, db_path):
        store = obs.ObservabilityStore(db_path)
        self._seed_channel(store, "chan-a", "run-a", "20260101T000000.000000Z-a")
        self._seed_channel(store, "chan-b", "run-b", "20260101T000001.000000Z-b")

        deleted = store.clear_conversations()
        assert deleted["distillation_runs"] == 2
        assert deleted["distillation_passes"] == 4
        for table in _DISTILL_TABLES:
            assert _rows(db_path, f"SELECT * FROM {table}") == []
        assert _rows(db_path, "SELECT * FROM spans") == []
        # Counters survive a clear, as they always have.
        store.mint_conversation_id("chan-a")


class TestDistillationRetention:
    def test_pinned_run_survives_a_prune_that_deletes_its_neighbour(self, db_path):
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-pin", "t-pin", started_days_ago=90)
        _seed_run(store, "run-free", "t-free", started_days_ago=90)
        _seed_spans(store, "t-pin", 3, days_ago=90)
        _seed_spans(store, "t-free", 3, days_ago=90)
        assert store.pin_distillation_run("run-pin") is True

        store.prune(retention_days=30)

        traces = {r["trace_id"] for r in _rows(db_path, "SELECT trace_id FROM spans")}
        assert traces == {"t-pin"}
        # The unpinned run loses the bulk and keeps its conclusions [DR52].
        divergences = {
            r["run_id"] for r in _rows(db_path, "SELECT run_id FROM distillation_divergences")
        }
        assert divergences == {"run-pin"}
        runs = {
            r["run_id"]: r["evidence_pruned"]
            for r in _rows(db_path, "SELECT run_id, evidence_pruned FROM distillation_runs")
        }
        assert runs == {"run-pin": 0, "run-free": 1}
        assert len(_rows(db_path, "SELECT * FROM distillation_insights")) == 2
        entry_inputs = {
            r["run_id"]: r["entry_inputs_json"]
            for r in _rows(
                db_path, "SELECT run_id, entry_inputs_json FROM distillation_passes"
            )
        }
        assert entry_inputs["run-free"] is None
        assert entry_inputs["run-pin"] is not None

    def test_an_all_pinned_batch_does_not_stop_eviction(self, db_path, monkeypatch):
        """[DR52], the :1178-1179 trap: with the pin predicate on the OUTER
        DELETE, the first batch whose spans are all pinned deletes nothing,
        rowcount is 0, and the size-cap loop breaks with the DB still over cap
        and evictable spans still present."""
        monkeypatch.setattr(obs, "_PRUNE_BATCH_ROWS", 2)
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-pin", "t-pin")
        _seed_spans(store, "t-pin", 4, days_ago=1)  # oldest, and all pinned
        _seed_spans(store, "t-loose", 3, days_ago=0)
        assert store.pin_distillation_run("run-pin") is True

        # pin_max_fraction is held out of the way: this asserts the eviction
        # loop makes progress, not the ceiling (tested separately below).
        store.prune(retention_days=30, max_bytes=1, pin_max_fraction=10**9)

        traces = [r["trace_id"] for r in _rows(db_path, "SELECT trace_id FROM spans")]
        assert traces.count("t-loose") == 0, "eviction stopped on an all-pinned batch"
        assert traces.count("t-pin") == 4

    def test_size_cap_eviction_never_half_deletes_a_trace(self, db_path, monkeypatch):
        """`[DR27]`: whole traces, or none of them.

        Deleting spans by `start_ns` with a row limit cuts across trace
        boundaries by construction, and the surviving half renders as a
        waterfall with silently missing rows — a turn that reads as though the
        agent skipped steps it actually took. That is worse than losing the
        trace outright, because nothing about it looks lossy.
        """
        # ONE batch, so eviction stops partway and what it left is visible.
        # An unbounded run against a 1-byte cap empties the table under either
        # behaviour, which is why the obvious version of this test proves
        # nothing. A span-ordered batch of 4 takes four of `t-a`'s six spans
        # and leaves the other two — the half-deleted trace.
        monkeypatch.setattr(obs, "_PRUNE_MAX_BATCHES", 1)
        monkeypatch.setattr(obs, "_PRUNE_BATCH_ROWS", 4)
        monkeypatch.setattr(obs, "_EVICT_TRACES_PER_BATCH", 1)
        store = obs.ObservabilityStore(db_path)
        for index, trace in enumerate(("t-a", "t-b", "t-c")):
            _seed_spans(store, trace, 6, base=index * 100)

        store.prune(retention_days=3650, max_bytes=1)

        surviving = {
            row["trace_id"]: row["n"]
            for row in _rows(
                db_path, "SELECT trace_id, COUNT(*) AS n FROM spans GROUP BY trace_id"
            )
        }
        assert surviving == {"t-b": 6, "t-c": 6}, (
            f"eviction was not trace-atomic: {surviving}"
        )
        marker = _rows(
            db_path, "SELECT value FROM diagnostics WHERE key='span_evictions'"
        )
        assert marker, "eviction left no marker, so the loss is only inferable"
        assert json.loads(marker[0]["value"])["reason"] == "size-cap"

    def test_size_cap_eviction_marks_a_distillation_run_evidence_pruned(
        self, db_path
    ):
        """A run whose trace is evicted must SAY its evidence is gone, or the
        UI renders an empty diff instead of "the trace behind this is gone"."""
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-evict", "t-evict", diverged=1, with_insight=False)
        _seed_spans(store, "t-evict", 4)

        store.prune(retention_days=3650, max_bytes=1)

        assert _rows(db_path, "SELECT * FROM spans WHERE trace_id='t-evict'") == []
        run = _rows(
            db_path,
            "SELECT evidence_pruned FROM distillation_runs WHERE run_id='run-evict'",
        )[0]
        assert run["evidence_pruned"] == 1

    def test_the_pinned_set_ceiling_evicts_pinned_traces_loudly(self, db_path):
        """[DR52]: the cap wins over the pin, with a diagnostic marker."""
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-pin", "t-pin")
        _seed_spans(store, "t-pin", 4)
        assert store.pin_distillation_run("run-pin") is True

        result = store.prune(retention_days=30, max_bytes=1, pin_max_fraction=0.5)

        assert result["pinned_traces_evicted"] == 1
        assert _rows(db_path, "SELECT * FROM spans") == []
        over_cap = json.loads(
            _rows(db_path, "SELECT value FROM diagnostics WHERE key='distill_pin_over_cap'")[0][
                "value"
            ]
        )
        assert over_cap["evicting_pinned"] is True
        assert over_cap["pinned_bytes"] > 0
        marker = json.loads(
            _rows(db_path, "SELECT value FROM diagnostics WHERE key='span_evictions'")[0]["value"]
        )
        assert marker["traces_evicted"] == 1
        # The run is still there, and says its evidence is gone.
        shortfall = store.distillation_evidence_shortfall("run-pin")
        assert shortfall["evidence_pruned"] is True
        assert shortfall["incomplete"] is True

    def test_negative_pin_is_released_after_its_window(self, db_path):
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-neg", "t-neg", diverged=0, with_insight=False)
        _seed_spans(store, "t-neg", 2, days_ago=90)
        store.pin_distillation_run("run-neg")
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE distillation_runs SET pinned_at=? WHERE run_id='run-neg'",
                (_iso_days_ago(120),),
            )
            conn.commit()

        result = store.prune(retention_days=30, negative_pin_days=90)

        assert result["distillation_pins_released"] == 1
        assert _rows(db_path, "SELECT pinned FROM distillation_runs")[0]["pinned"] == 0
        # Released this pass means prunable this pass.
        assert _rows(db_path, "SELECT * FROM spans") == []

    def test_a_no_divergence_run_stays_pinned_inside_its_window(self, db_path):
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-neg", "t-neg", diverged=0, with_insight=False)
        _seed_spans(store, "t-neg", 2, days_ago=90)
        store.pin_distillation_run("run-neg")

        store.prune(retention_days=30, negative_pin_days=90)

        assert _rows(db_path, "SELECT pinned FROM distillation_runs")[0]["pinned"] == 1
        assert len(_rows(db_path, "SELECT * FROM spans")) == 2

    def test_an_agreeing_run_that_produced_an_insight_survives_the_window(
        self, db_path
    ):
        """`fix-sb8.17`: the two §10.3 pin classes partition on "produced an
        insight", not on "diverged".

        The release arm selected `pinned=1 AND planning_diverged=0 AND
        exec_diverged=0` with no insight guard, so a run that AGREED and still
        yielded an insight fell into the 90-day negative window and lost the
        evidence behind a permanently-pinned rule — AC9, for the most
        interesting insight class there is. Nothing writes that row today
        (extraction is gated on divergence), but `fix-sb8.11`'s replay is
        exactly an agreeing run that says something about an insight.
        """
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-agree", "t-agree", diverged=0)
        _seed_spans(store, "t-agree", 2, days_ago=90)
        store.pin_distillation_run("run-agree")
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE distillation_runs SET pinned_at=? WHERE run_id='run-agree'",
                (_iso_days_ago(120),),
            )
            conn.commit()

        result = store.prune(retention_days=30, negative_pin_days=90)

        assert result.get("distillation_pins_released", 0) == 0
        assert _rows(db_path, "SELECT pinned FROM distillation_runs")[0]["pinned"] == 1
        assert len(_rows(db_path, "SELECT * FROM spans")) == 2, (
            "the 90-day negative window pruned the evidence behind an insight"
        )

    def test_an_agreeing_run_whose_insight_is_rejected_is_still_released(
        self, db_path
    ):
        """The other half of the partition: once every insight on that run is
        rejected, the rejected-only arm — which does not look at divergence at
        all — is what releases it."""
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-agree", "t-agree", diverged=0)
        _seed_spans(store, "t-agree", 2, days_ago=90)
        store.pin_distillation_run("run-agree")
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO distillation_verdicts (verdict_id, insight_id, "
                "verdict, actor, superseded, created_at) VALUES "
                "('v-rej', 'ins-run-agree', 'overfit-to-single-turn', 'me', 0, ?)",
                (_iso_days_ago(0),),
            )
            conn.commit()

        result = store.prune(retention_days=30, negative_pin_days=90)

        assert result["distillation_pins_released"] == 1
        assert _rows(db_path, "SELECT pinned FROM distillation_runs")[0]["pinned"] == 0

    def test_a_rejected_only_run_is_unpinned_by_the_sweep(self, db_path):
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-rej", "t-rej")
        _seed_run(store, "run-sup", "t-sup")
        store.pin_distillation_run("run-rej")
        store.pin_distillation_run("run-sup")
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for verdict_id, insight_id, verdict in (
                ("v-old", "ins-run-rej", "supported"),
                ("v-new", "ins-run-rej", "overfit-to-single-turn"),
                ("v-sup", "ins-run-sup", "supported"),
            ):
                conn.execute(
                    "INSERT INTO distillation_verdicts (verdict_id, insight_id, "
                    "verdict, actor, superseded, created_at) VALUES (?, ?, ?, "
                    "'human', ?, ?)",
                    (
                        verdict_id,
                        insight_id,
                        verdict,
                        1 if verdict_id == "v-old" else 0,
                        _iso_days_ago(1 if verdict_id == "v-old" else 0),
                    ),
                )
            conn.commit()

        store.prune(retention_days=30)

        pinned = {
            r["run_id"]: r["pinned"]
            for r in _rows(db_path, "SELECT run_id, pinned FROM distillation_runs")
        }
        assert pinned == {"run-rej": 0, "run-sup": 1}

    def test_pin_records_the_span_count_and_reports_a_shortfall(self, db_path):
        """[DR43]: the pin binds only builds carrying the predicate, so a loss
        caused by another build must be detectable."""
        store = obs.ObservabilityStore(db_path)
        _seed_run(store, "run-pin", "t-pin")
        _seed_spans(store, "t-pin", 5)
        store.pin_distillation_run("run-pin")

        intact = store.distillation_evidence_shortfall("run-pin")
        assert intact["pinned_span_count"] == 5
        assert intact["live_span_count"] == 5
        assert intact["incomplete"] is False

        # An older build's prune(), which has no pin predicate.
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM spans WHERE span_id='t-pin-s0'")
            conn.commit()

        lost = store.distillation_evidence_shortfall("run-pin")
        assert lost["missing_span_count"] == 1
        assert lost["incomplete"] is True
        assert store.distillation_evidence_shortfall("nope") is None

    def test_retention_env_defaults(self, db_path, monkeypatch):
        monkeypatch.delenv("FW_OBS_DISTILL_NEGATIVE_PIN_DAYS", raising=False)
        monkeypatch.delenv("FW_OBS_DISTILL_PIN_MAX_FRACTION", raising=False)
        assert obs._env_int("FW_OBS_DISTILL_NEGATIVE_PIN_DAYS", 90) == 90
        assert obs._env_float("FW_OBS_DISTILL_PIN_MAX_FRACTION", 0.5) == 0.5
        monkeypatch.setenv("FW_OBS_DISTILL_NEGATIVE_PIN_DAYS", "7")
        monkeypatch.setenv("FW_OBS_DISTILL_PIN_MAX_FRACTION", "0.25")
        assert obs._env_int("FW_OBS_DISTILL_NEGATIVE_PIN_DAYS", 90) == 7
        assert obs._env_float("FW_OBS_DISTILL_PIN_MAX_FRACTION", 0.5) == 0.25


# ----------------------------------------------------------------------
# [R21]/[DR44] erasure: reach AND cost. forget_channel is on the ordinary
# turn path, not the distillation path, so its shape is everyone's problem.
# ----------------------------------------------------------------------


class _CountingStore(obs.ObservabilityStore):
    """The real store with SQLite's own statement tracer attached.

    Not a stand-in for the store — every statement still runs against the real
    file. Counting the statements SQLite actually prepares is the only way to
    assert the *shape* of a delete (constant vs one-per-turn) rather than its
    wall clock, which is what the regression was.
    """

    def __init__(self, db_path: str) -> None:
        super().__init__(db_path)
        self.statements: list[str] = []

    def _connect(self, timeout: float = 30.0):
        conn = super()._connect(timeout)
        conn.set_trace_callback(self.statements.append)
        return conn

    def span_deletes(self) -> list[str]:
        return [s for s in self.statements if "DELETE FROM spans" in s]


def _seed_channel_turns(store, channel_id: str, n_turns: int) -> list[str]:
    """n_turns turns, each with one turn-root span and one replay span.

    The replay span carries no channel_id of its own ([DR41]), so only the
    derived-trace arm of forget_channel can reach it.
    """
    keys = [f"20260101T{i:06d}.000000Z-{channel_id}" for i in range(n_turns)]
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for i, key in enumerate(keys):
            conn.execute(
                "INSERT INTO turns (turn_key, channel_id, user_message, status, "
                "success, record_version, record_json) VALUES (?, ?, 'm', "
                "'completed', 1, 1, '{}')",
                (key, channel_id),
            )
            conn.execute(
                "INSERT INTO spans (span_id, trace_id, name, kind, channel_id, "
                "start_ns, status, attributes) VALUES (?, ?, 'fw.turn', "
                "'internal', ?, ?, 'ok', '{}')",
                (f"{key}-s", key, channel_id, i),
            )
            conn.execute(
                "INSERT INTO spans (span_id, trace_id, name, kind, channel_id, "
                "start_ns, status, attributes) VALUES (?, ?, 'fw.turn', "
                "'internal', NULL, ?, 'ok', '{}')",
                (f"{key}-r", f"{key}~replay.1", i),
            )
        conn.commit()
    return keys


class TestErasureCost:
    """[DR44] made erasure reach the six new tables and the '~replay.<n>'
    derived traces. The obvious way to get that reach — one
    `DELETE FROM spans WHERE trace_id LIKE key || '~%'` per turn key — is a
    full scan of `spans` per turn of the channel, inside one BEGIN IMMEDIATE:
    quadratic in channel size, and it locks the sink's writer out for the
    duration. Both properties have to hold together, so they are asserted
    together."""

    def test_the_derived_trace_delete_does_not_grow_with_the_channel(self, tmp_path):
        """The regression, stated as a shape rather than a stopwatch."""
        counts = {}
        for n_turns in (20, 400):
            store = _CountingStore(str(tmp_path / f"obs-{n_turns}.sqlite3"))
            _seed_channel_turns(store, "chan-cost", n_turns)
            _seed_channel_turns(store, "chan-other", 5)
            store.statements.clear()

            store.forget_channel("chan-cost")
            counts[n_turns] = len(store.span_deletes())

        assert counts[20] == counts[400], (
            "forget_channel issues one DELETE against `spans` per turn key: "
            f"{counts[20]} statements at 20 turns, {counts[400]} at 400. Each "
            "one is a full scan (a LIKE with ESCAPE cannot use "
            "idx_spans_trace), so erasure is quadratic in channel size."
        )

    def test_a_large_channel_is_erased_quickly_and_completely(self, tmp_path):
        """Timing AND [DR44] reach, on one channel big enough for the old shape
        to be visibly quadratic.

        Measured with the per-key shape: 0.59s at 2000 turns, 3.06s at 4000 —
        8x the turns for 78x the work. The set-based shape is 0.017s at 4000.
        The ceiling below sits an order of magnitude above the fixed cost and
        an order of magnitude below the regressed one.
        """
        db_path = str(tmp_path / "obs-big.sqlite3")
        store = obs.ObservabilityStore(db_path)
        keys = _seed_channel_turns(store, "chan-big", 4000)
        # A neighbouring channel of the same size, so the rows each per-key
        # scan would walk are still there when the derived-trace arm runs: the
        # channel_id arm above it has already deleted this channel's own.
        _seed_channel_turns(store, "chan-kept", 4000)
        # ...and a full distillation closure on one of its turns, so the six
        # new tables are in scope for the same call.
        _seed_run(store, "run-big", keys[0])
        _seed_spans(store, f"{keys[0]}~replay.2", 3)

        started = time.perf_counter()
        deleted = store.forget_channel("chan-big")
        elapsed = time.perf_counter() - started

        assert elapsed < 1.5, (
            f"forget_channel took {elapsed:.2f}s on a 4000-turn channel; the "
            "per-key LIKE shape it replaced took 3.06s at that size and grows "
            "quadratically, all of it inside one BEGIN IMMEDIATE"
        )
        assert deleted["distillation_runs"] == 1

        # Every one of the six tables, emptied of this channel.
        for table in _DISTILL_TABLES:
            remaining = _rows(db_path, f"SELECT * FROM {table}")
            assert remaining == [], f"{table} survived the erasure"
        # Both span sets: the turn-root traces and the derived '~replay.<n>'
        # ones, which carry no channel_id of their own.
        surviving = {r["trace_id"] for r in _rows(db_path, "SELECT trace_id FROM spans")}
        assert not any(t.startswith("20260101") and t.endswith("chan-big") for t in surviving)
        assert not any("chan-big" in t for t in surviving), (
            f"derived traces survived: {sorted(t for t in surviving if 'chan-big' in t)[:5]}"
        )
        # The neighbouring channel is untouched.
        assert (
            len(_rows(db_path, "SELECT * FROM turns WHERE channel_id='chan-kept'"))
            == 4000
        )
        assert len(surviving) == 8000

    def test_the_conversationless_prune_arm_still_takes_replay_spans(
        self, db_path
    ):
        """The same per-key LIKE lived in `prune(include_conversationless_turns
        =True)` (ruling C10), up to _PRUNE_BATCH_ROWS scans per batch, and was
        hoisted with it. That arm had no test at all, so the hoist would have
        been an untested edit to a delete path.
        """
        store = obs.ObservabilityStore(db_path)
        gone = "20250101T000000.000000Z-gone"
        kept = "20250101T000001.000000Z-kept"
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO turns (turn_key, channel_id, conversation_id, "
                "user_message, status, success, record_version, record_json) "
                "VALUES (?, 'chan', NULL, 'm', 'completed', 1, 1, '{}')",
                (gone,),
            )
            conn.execute(
                "INSERT INTO turns (turn_key, channel_id, conversation_id, "
                "user_message, status, success, record_version, record_json) "
                "VALUES (?, 'chan', 1, 'm', 'completed', 1, 1, '{}')",
                (kept,),
            )
            conn.commit()
        # Recent spans on old turn keys: only the conversationless arm can
        # reach them, not the retention horizon.
        for trace in (gone, f"{gone}~replay.1", kept, f"{kept}~replay.1"):
            _seed_spans(store, trace, 2)

        store.prune(retention_days=30, include_conversationless_turns=True)

        traces = {r["trace_id"] for r in _rows(db_path, "SELECT trace_id FROM spans")}
        assert traces == {kept, f"{kept}~replay.1"}
        assert [r["turn_key"] for r in _rows(db_path, "SELECT turn_key FROM turns")] == [
            kept
        ]

    def test_the_derived_trace_delete_leaves_a_neighbours_replay_spans_alone(
        self, db_path
    ):
        """A set-based delete that matched too much would be a silent data-loss
        bug in exactly the direction erasure is supposed to be careful about."""
        store = obs.ObservabilityStore(db_path)
        _seed_channel_turns(store, "chan-x", 3)
        kept = _seed_channel_turns(store, "chan-y", 3)

        store.forget_channel("chan-x")

        traces = {r["trace_id"] for r in _rows(db_path, "SELECT trace_id FROM spans")}
        assert traces == {k for k in kept} | {f"{k}~replay.1" for k in kept}


# ----------------------------------------------------------------------
# [DR49] §7.5: flush() is a DURABILITY barrier, not a liveness one.
# ----------------------------------------------------------------------


class TestFlushIsADurabilityBarrier:
    """§7.5 makes `sink.flush()` the gate between "the aligner holds these Span
    objects" and "the table holds the rows the divergence record is about to
    cite". A flush that reports success for a batch that rolled back lets a run
    be stored `comparable = 1` while the spans its divergence rows name are
    gone."""

    def _span(self, span_id: str, trace_id: str = "t-barrier", start_ns: int = 1):
        return tracing.Span(
            span_id=span_id,
            trace_id=trace_id,
            name="fw.turn",
            start_ns=start_ns,
            status="ok",
        )

    def test_a_rolled_back_batch_makes_flush_report_failure(
        self, db_path, monkeypatch
    ):
        """The SQLITE_BUSY arm [R8]: the batch is discarded and its spans are
        dropped outright, so the barrier was not taken."""
        sink = obs.SQLiteTraceSink(db_path)
        try:
            def busy(*args, **kwargs):
                raise sqlite3.OperationalError("database is locked")

            monkeypatch.setattr(sink.store, "upsert_span_rows", busy)
            sink.emit_span(self._span("s-lost"))
            assert sink.flush(timeout=10.0) is False, (
                "flush() reported success for a batch that rolled back; a run "
                "written on the strength of it is comparable=1 with no spans"
            )
        finally:
            sink.close()

        assert _rows(db_path, "SELECT * FROM spans WHERE span_id='s-lost'") == []
        health = obs.ObservabilityStore(db_path).writer_health() or {}
        assert int(health.get("spans_dropped") or 0) >= 1

    def test_a_write_error_makes_flush_report_failure(self, db_path, monkeypatch):
        """The generic arm is worse than the busy one: nothing is requeued and
        `spans_dropped` never moves, so the barrier's bool is the only signal
        that exists."""
        sink = obs.SQLiteTraceSink(db_path)
        try:
            def broken(*args, **kwargs):
                raise RuntimeError("disk on fire")

            monkeypatch.setattr(sink.store, "upsert_span_rows", broken)
            sink.emit_span(self._span("s-gone"))
            assert sink.flush(timeout=10.0) is False
        finally:
            sink.close()

        assert _rows(db_path, "SELECT * FROM spans WHERE span_id='s-gone'") == []

    def test_a_loss_before_the_call_is_still_reported_to_the_next_barrier(
        self, db_path, monkeypatch
    ):
        """The batch that fails need not be the one carrying the sentinel.

        `_drain_pending` batches opportunistically, so the spans a caller is
        about to cite are routinely committed in an EARLIER batch than the
        flush it takes afterwards. A barrier that only reported its own batch
        would call that loss a success.
        """
        sink = obs.SQLiteTraceSink(db_path)
        try:
            def broken(*args, **kwargs):
                raise RuntimeError("disk on fire")

            monkeypatch.setattr(sink.store, "upsert_span_rows", broken)
            sink.emit_span(self._span("s-early"))
            # Let the writer pick that span up and fail on it by itself, well
            # before the barrier is enqueued.
            deadline = time.time() + 10.0
            while time.time() < deadline:
                health = sink.store.writer_health() or {}
                if int(health.get("write_errors") or 0) >= 1:
                    break
                time.sleep(0.02)
            monkeypatch.setattr(
                sink.store, "upsert_span_rows", type(sink.store).upsert_span_rows.__get__(sink.store)
            )
            assert sink.flush(timeout=10.0) is False
        finally:
            sink.close()

    def test_the_barrier_recovers_after_reporting_a_loss(self, db_path, monkeypatch):
        """A hiccup must be charged to one barrier, not to every later one:
        `comparable = 0` on every subsequent run would be its own corruption of
        §15's corpus."""
        sink = obs.SQLiteTraceSink(db_path)
        try:
            real = sink.store.upsert_span_rows

            def broken(*args, **kwargs):
                raise RuntimeError("disk on fire")

            monkeypatch.setattr(sink.store, "upsert_span_rows", broken)
            sink.emit_span(self._span("s-bad"))
            assert sink.flush(timeout=10.0) is False

            monkeypatch.setattr(sink.store, "upsert_span_rows", real)
            sink.emit_span(self._span("s-good", start_ns=2))
            assert sink.flush(timeout=10.0) is True
        finally:
            sink.close()

        kept = {r["span_id"] for r in _rows(db_path, "SELECT span_id FROM spans")}
        assert kept == {"s-good"}

    def test_the_barrier_waits_for_the_whole_span_backlog_not_one_batch(
        self, db_path
    ):
        """`_next_item` takes the record queue first and `_drain_pending` caps a
        batch at 512, while the span queue holds 10,000. Completing the barrier
        in the batch the sentinel lands in therefore lets the caller cite spans
        that are still queued — which is the same failure as citing dropped
        ones, minus the counter.
        """
        n_spans = 4000
        sink = obs.SQLiteTraceSink(db_path)
        blocker = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        released = threading.Event()

        def release() -> None:
            # Long enough that flush()'s sentinel is enqueued BEHIND the whole
            # backlog while the writer is still stuck in BEGIN IMMEDIATE.
            time.sleep(0.5)
            blocker.rollback()
            blocker.close()
            released.set()

        try:
            blocker.execute("BEGIN IMMEDIATE")
            for i in range(n_spans):
                sink.emit_span(self._span(f"s{i}", start_ns=i))
            releaser = threading.Thread(target=release, daemon=True)
            releaser.start()

            assert sink.flush(timeout=60.0) is True
            landed = _rows(db_path, "SELECT COUNT(*) AS n FROM spans")[0]["n"]
            releaser.join(30.0)
        finally:
            released.wait(30.0)
            sink.close()

        health = obs.ObservabilityStore(db_path).writer_health() or {}
        assert int(health.get("spans_dropped") or 0) == 0, "the queue overflowed"
        assert landed == n_spans, (
            f"flush() returned True with {n_spans - landed} of {n_spans} spans "
            "still unwritten"
        )

    def test_a_barrier_settles_under_continuous_concurrent_emission(self, db_path):
        """`fix-sb8.15`: the barrier must not require GLOBAL sink quiescence.

        `get_observability_sink` caches one sink — one writer thread — per
        workflow DB and shares it across every channel, so a second channel
        emitting spans steadily is the ordinary case, not an exotic one. A
        both-queues-empty barrier never settles under that load: it burns its
        whole timeout and returns False, which `DistillationSession.read_barrier`
        turns into `comparable = 0` / `evidence-incomplete` and a run that AC9
        then declines to pin.

        The barrier is scoped to the trace the caller is about to cite, so it
        waits for that trace's spans and for nothing the other channel piled up
        afterwards.
        """
        sink = obs.SQLiteTraceSink(db_path)
        stop = threading.Event()

        def noisy() -> None:
            i = 0
            while not stop.is_set():
                sink.emit_span(
                    self._span(f"noise-{i}", trace_id="t-other", start_ns=1000 + i)
                )
                i += 1
                # No sleep: a quiescence barrier survives a producer that
                # pauses, so pausing would not exercise the bug. Measured
                # against the old rule this starves flush() for its full
                # 10s timeout; against the watermark it settles immediately.

        noise = threading.Thread(target=noisy, daemon=True)
        try:
            sink.emit_span(self._span("s-mine", start_ns=1))
            noise.start()
            # Give the noise thread a head start so the queues are genuinely
            # never simultaneously empty for the duration of the barrier.
            time.sleep(0.1)
            started = time.monotonic()
            ok = sink.flush(timeout=10.0, trace_id="t-barrier")
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            noise.join(10.0)
            sink.close()

        assert ok is True, (
            "flush() was starved by an unrelated producer on the shared sink"
        )
        assert elapsed < 5.0, (
            f"the barrier took {elapsed:.2f}s under concurrent emission, which "
            "is the starvation signature rather than a settled watermark"
        )
        assert _rows(db_path, "SELECT span_id FROM spans WHERE span_id='s-mine'")

    def test_the_watermark_still_reports_a_loss_under_concurrent_emission(
        self, db_path, monkeypatch
    ):
        """Anti-starvation must not cost the durability property (finding 3):
        a rolled-back batch still has to report False, noise or no noise."""
        sink = obs.SQLiteTraceSink(db_path)
        stop = threading.Event()

        def broken(*args, **kwargs):
            raise RuntimeError("disk on fire")

        def noisy() -> None:
            i = 0
            while not stop.is_set():
                sink.emit_span(
                    self._span(f"noise-{i}", trace_id="t-other", start_ns=2000 + i)
                )
                i += 1

        noise = threading.Thread(target=noisy, daemon=True)
        try:
            monkeypatch.setattr(sink.store, "upsert_span_rows", broken)
            sink.emit_span(self._span("s-lost-noisy", start_ns=1))
            noise.start()
            time.sleep(0.1)
            assert sink.flush(timeout=10.0, trace_id="t-barrier") is False
        finally:
            stop.set()
            noise.join(10.0)
            sink.close()

        assert _rows(
            db_path, "SELECT span_id FROM spans WHERE span_id='s-lost-noisy'"
        ) == []
