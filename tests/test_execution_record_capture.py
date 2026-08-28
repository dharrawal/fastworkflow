"""TurnResult.execution_records join command_call_id to spans (arch §12.1–§12.2).

These tests measure the durable correlation skeleton on ``TurnResult``: each
``ExecutionRecordRef`` must name the ``command_call_id`` minted for one dispatch
and the ``span_id`` of the span that covers it, so a reader holding
``record_json`` can join outcomes to trace rows without guessing from ordering.

Fixtures follow ``tests/test_command_call_id.py`` and
``tests/test_turn_result_additive.py``: real ``todo_list_workflow``, real
``WorkflowExecutionContext``, real ``CommandExecutor``, and a real
``TraceSink`` implementation writing real SQLite in ``tmp_path``. The only
stand-in is the CME wildcard hop on the prose path, because the test workflow
ships no trained intent models — and it performs a real nested dispatch rather
than returning a canned result.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import TurnResult, tracing
from fastworkflow import observability_store as obs
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.execution_recorder import record_execution
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

LIST_COMMAND = "TodoListManager/list_todo_lists"
CREATE_COMMAND = "TodoListManager/create_todo_list"


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


class RecordingTraceSink:
    """Real TraceSink implementation that records everything it receives."""

    def __init__(self):
        self.spans: list[tracing.Span] = []
        self.turn_records: list[TurnResult] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        self.turn_records.append(record)
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


def _make_ctx(todo_workflow_path: str, tmp_path, sink) -> WorkflowExecutionContext:
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"execrec-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    return ctx


@pytest.fixture
def sink() -> RecordingTraceSink:
    return RecordingTraceSink()


@pytest.fixture
def ctx(initialized_fastworkflow, todo_workflow_path, tmp_path, sink):
    context = _make_ctx(todo_workflow_path, tmp_path, sink)
    yield context
    with suppress(Exception):
        context.close()


def _action(command_name: str = LIST_COMMAND, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=command_name, command="do it", parameters=parameters
    )


def _nesting_cme_hop(monkeypatch, app_workflow, nested_command: str = LIST_COMMAND):
    """Stand in for the untrained CME wildcard hop, keeping the nesting real."""
    real_perform_action = CommandExecutor.perform_action

    def cme_hop(cls, workflow, action):
        command_output = real_perform_action(
            app_workflow, _action(nested_command)
        )
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = nested_command
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(cme_hop))
    return real_perform_action


def _rows(path: str, sql: str) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def test_direct_action_turn_populates_execution_records(ctx, sink):
    """One real dispatch yields one correlation ref joined to its span."""
    ctx.process_action_turn(_action())

    assert sink.turn_records, "the real turn emitted no turn record"
    turn_result = sink.turn_records[-1]
    assert turn_result.execution_records, "execution_records stayed empty"

    record = turn_result.execution_records[0]
    tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
    span_call_id = tool_call.attributes[tracing.ATTR_COMMAND_CALL_ID]

    assert record.command_call_id == span_call_id
    assert record.span_id == tool_call.span_id
    assert record.command_ordinal == 0
    assert record.parent_call_id is None
    assert ctx._turn_outputs[-1].command_call_id == span_call_id


def test_execution_records_survive_turn_record_serialization(ctx, sink):
    """The join must reach record_json, not only in-memory TurnResult."""
    ctx.process_action_turn(_action())

    turn_row, _artifacts = obs.serialize_turn_result(sink.turn_records[-1])
    persisted = json.loads(turn_row["record_json"])

    span_call_id = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0].attributes[
        tracing.ATTR_COMMAND_CALL_ID
    ]
    assert persisted["execution_records"] == [
        {
            "contract_version": persisted["execution_records"][0]["contract_version"],
            "command_call_id": span_call_id,
            "parent_call_id": None,
            "command_ordinal": 0,
            "span_id": sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0].span_id,
        }
    ]


def test_execution_records_survive_sqlite_round_trip(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """Through real SQLite, because that is where a reader actually joins."""
    db_path = str(tmp_path / "observability.sqlite3")
    store_sink = obs.SQLiteTraceSink(db_path)
    context = _make_ctx(todo_workflow_path, tmp_path, store_sink)
    try:
        context.process_action_turn(_action())
        assert store_sink.flush()
    finally:
        with suppress(Exception):
            context.close()
        store_sink.close()

    stored = json.loads(_rows(db_path, "SELECT record_json FROM turns")[0]["record_json"])
    restored = TurnResult.model_validate(stored)

    assert restored.execution_records
    assert restored.execution_records[0].command_call_id
    assert restored.execution_records[0].span_id


def test_prose_path_records_nested_child_calls(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """Inner CME/core hops with no span of their own appear as child refs."""
    context = _make_ctx(todo_workflow_path, tmp_path, sink)
    try:
        _nesting_cme_hop(monkeypatch, context.app_workflow)
        context.process_turn("list my todo lists")

        turn_result = sink.turn_records[-1]
        assert len(turn_result.execution_records) >= 2

        parent = turn_result.execution_records[0]
        child = turn_result.execution_records[1]
        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]

        assert parent.command_call_id == execute.attributes[tracing.ATTR_COMMAND_CALL_ID]
        assert parent.span_id == execute.span_id
        assert child.parent_call_id == parent.command_call_id
        assert child.command_call_id != parent.command_call_id
        assert child.span_id is None
    finally:
        with suppress(Exception):
            context.close()


def test_no_sink_means_no_execution_records(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """Capture projections are sink-gated; the id on CommandOutput is not."""
    context = _make_ctx(todo_workflow_path, tmp_path, sink=None)
    try:
        context.process_action_turn(_action())
        assert context._turn_outputs[-1].command_call_id
        assert getattr(context, "_execution_recorder", None) is None
    finally:
        with suppress(Exception):
            context.close()


def test_record_execution_with_no_recorder_is_cheap():
    """FW-NFR-005 sanity: record_execution(recorder=None) is a no-op, not work.

    No strict timing budget — only that 100k calls finish in well under a second,
    matching the spirit of test_span_contract_versioning.py's stamping-cost test.
    """
    start = time.perf_counter()
    for _ in range(100_000):
        record_execution(
            None,
            command_call_id="deadbeef",
            parent_call_id=None,
            span_id=None,
            child_calls=[{"call_id": "c1", "parent_call_id": "deadbeef"}],
        )
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"record_execution(None) took {elapsed:.2f}s for 100k calls"
