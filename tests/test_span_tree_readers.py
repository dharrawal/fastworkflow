"""The readers of the store read the span tree of contract 63 (bd fix-qjne.34 step 8).

A turn of contract 63 is ``fw.turn > fw.split / fw.resolve / fw.review / fw.run > fw.run.part > fw.plan, fw.step >
fw.command, fw.question / fw.answer``. Each reader below is checked on such a turn: the reconstruction for
distillation and its feedback anchors, the replay of a step's command, the plan segments of a consistency
comparison, the diagnosis's known names, and the debug page (a jsdom check, run by
``python -m tests.browser_validation``). A store written before contract 63 is not read (owner decision: no legacy
path in the framework)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from fastworkflow import distillation_trace, policy_replay, state_paths, tracing
from fastworkflow.observability import comparison, consistency, diagnosis, turn_derivations
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server

T0 = 1_800_000_000_000_000_000
NEW_TURN = "20261125T100100-new-contract"


def _span(span_id, trace_id, name, parent, t0, *, command_name=None, attributes=None, kind="internal", dur=1_000_000):
    return {"span_id": span_id, "trace_id": trace_id, "parent_span_id": parent, "name": name, "kind": kind,
            "channel_id": "chan", "command_name": command_name, "context": None, "start_ns": t0,
            "end_ns": t0 + dur, "status": "ok", "attributes": attributes or {}}


def new_turn(key: str = NEW_TURN) -> list[dict]:
    """A turn as span contract 63 records it: the stages, the run part, the plan beside its step, the step with its
    command as a child, the question under the part, the answer with the summary call."""
    return [
        _span("n-root", key, "fw.turn", None, T0, dur=90_000_000, attributes={"user_message": "find Ann"}),
        _span("n-split", key, "fw.split", "n-root", T0 + 500_000, attributes={"parts": 1}),
        _span("n-gate", key, "fw.split.gate", "n-split", T0 + 600_000, attributes={"label": "one"}),
        _span("n-gate-llm", key, "fw.llm.call", "n-gate", T0 + 700_000, kind="llm", attributes={"role": "JEV"}),
        _span("n-resolve", key, "fw.resolve", "n-root", T0 + 800_000, attributes={"parts": 1}),
        _span("n-rpart", key, "fw.resolve.part", "n-resolve", T0 + 850_000, command_name="find_person",
              attributes={"part": 1, "outcome": "bound", "command": "find_person"}),
        _span("n-intent", key, "fw.intent", "n-rpart", T0 + 860_000,
              attributes={"decided_by": "local", "command_name": "find_person", "part": 1}),
        _span("n-run", key, "fw.run", "n-root", T0 + 900_000, dur=80_000_000, attributes={"parts": 1, "ran": 1}),
        _span("n-part", key, "fw.run.part", "n-run", T0 + 950_000, dur=75_000_000, command_name="find_person",
              attributes={"part": 1, "means": "ad_hoc", "outcome": "ran", "agent_input": "find Ann"}),
        _span("n-plan", key, "fw.plan", "n-part", T0 + 1_000_000,
              attributes={"plan": "1. find\n2. answer", "trigger": "initial"}),
        _span("n-plan-llm", key, "fw.llm.call", "n-plan", T0 + 1_100_000, kind="llm",
              attributes={"reasoning": "find first", "role": "LLM_PLANNER"}),
        _span("n-step", key, "fw.step", "n-part", T0 + 3_000_000, dur=20_000_000,
              attributes={"run_by": "agent", "kind": "command", "plan_span_id": "n-plan", "thought": "look her up",
                          "tool_name": "execute_workflow_query", "tool_args": {"command": "find_person Ann"},
                          "observation": "Ann, 42", "raw_command": "find_person Ann", "success": True,
                          "command_call_id": "c-1", "response_text": "Ann, 42"}),
        _span("n-step-llm", key, "fw.llm.call", "n-step", T0 + 3_100_000, kind="llm",
              attributes={"module_chain": "fastWorkflowReAct > Predict", "role": "LLM_AGENT"}),
        _span("n-cmd", key, "fw.command", "n-step", T0 + 4_100_000, kind="tool", command_name="find_person",
              attributes={"raw_command": "find_person Ann", "parameters": {"name": "Ann"}, "success": True,
                          "command_call_id": "c-1", "response_text": "Ann, 42", "resolved_by": "exact_name",
                          "parameters_read_by": "typed", "parameters_valid": True}),
        _span("n-ask", key, "fw.question", "n-part", T0 + 70_000_000, kind="human_wait",
              attributes={"kind": "agent", "asked": "which Ann?", "user_response": "the first"}),
        _span("n-answer", key, "fw.answer", "n-root", T0 + 85_000_000, attributes={"written_by": "agent"}),
        _span("n-sum", key, "fw.llm.call", "n-answer", T0 + 86_000_000, kind="llm",
              attributes={"purpose": "conversation_summary", "role": "LLM_PLANNER"}),
    ]


# ----------------------------------------------------------------------
# The names
# ----------------------------------------------------------------------


def test_the_readers_restate_the_emitter_s_names():
    """The stdlib-only readers restate the names they read; they are pinned here to the emitter's constants."""
    assert turn_derivations.SPAN_COMMAND == comparison.SPAN_COMMAND == tracing.SPAN_COMMAND
    assert turn_derivations.SPAN_QUESTION == tracing.SPAN_QUESTION
    assert turn_derivations.SPAN_INTENT == tracing.SPAN_INTENT
    assert consistency.SPAN_PLAN == tracing.SPAN_PLAN


def test_diagnosis_knows_the_contract_s_names_only():
    assert {span["name"] for span in new_turn()} <= diagnosis.KNOWN_SPAN_NAMES
    assert diagnosis.KNOWN_SPAN_NAMES == frozenset(tracing.SPAN_CONTRACTS)
    assert "fw.command.execute" not in diagnosis.KNOWN_SPAN_NAMES


def test_a_contract_is_asked_by_its_name_only():
    assert tracing.SPAN_CONTRACTS.get("fw.command.execute") is None
    assert tracing.SPAN_CONTRACTS[tracing.SPAN_COMMAND].version == 6


# ----------------------------------------------------------------------
# distillation_trace: the reconstruction and the feedback anchors
# ----------------------------------------------------------------------


def test_a_turn_is_reconstructed_from_its_steps_commands_and_plans():
    new = distillation_trace.read_turn({"turn_key": NEW_TURN, "ordinal": 1}, new_turn())
    assert new.trajectory["tool_name_0"] == "execute_workflow_query" and new.trajectory["thought_0"] == "look her up"
    assert new.trajectory["observation_0"] == "Ann, 42"
    assert new.actions == [{"command": "find_person Ann", "command_name": "find_person",
                            "parameters": {"name": "Ann"}, "response": "Ann, 42"}]
    assert [(s.user_query, s.generated_plan, s.reasoning) for s in new.planning_steps] == [
        ("find Ann", ["1. find", "2. answer"], "find first")]
    anchors = distillation_trace._span_ids_by_phase(new_turn())
    assert anchors[distillation_trace.PHASE_COMMANDS] == ["n-cmd"]
    assert anchors[distillation_trace.PHASE_STEPS] == ["n-cmd", "n-step-llm"]
    assert anchors[distillation_trace.PHASE_PLANNING] == ["n-plan-llm"]
    assert "n-ask" in anchors[distillation_trace.PHASE_EXECUTION]


def test_a_second_plan_reads_as_a_replan_by_its_trigger():
    spans = new_turn() + [_span("n-replan", NEW_TURN, "fw.plan", "n-part", T0 + 80_000_000,
                                attributes={"plan": "1. ask", "trigger": "ask_user_response"})]
    steps = distillation_trace.read_turn({"turn_key": NEW_TURN, "ordinal": 1}, spans).planning_steps
    assert [step.user_query for step in steps] == ["find Ann", "[replan: ask_user_response]"]
    segments = consistency._plan_segments(0, NEW_TURN, spans)
    assert [segment.boundary for segment in segments] == ["[turn 1 plan]", "[turn 1 replan after ask_user_response]"]


# ----------------------------------------------------------------------
# policy_replay: the command a step dispatched
# ----------------------------------------------------------------------


def test_the_command_a_step_dispatched_is_its_own_command_child():
    spans = new_turn()
    step = next(span for span in spans if span["span_id"] == "n-step")
    assert policy_replay._executed(step, policy_replay._children(spans)) == (
        "find_person", {"name": "Ann"}, True, "Ann, 42")


# ----------------------------------------------------------------------
# The debug page (jsdom)
# ----------------------------------------------------------------------


def _row(key, spans_of):
    return {"turn_key": key, "channel_id": "chan", "conversation_id": None, "ordinal": 1,
            "user_message": "find Ann", "refined_user_message": None, "entry_workflow_name": "span-tree",
            "entry_context": "test", "status": "completed", "success": 1, "failure_reason": None, "answer": "Ann, 42",
            "conversation_summary": None, "conversation_traces": None, "started_at": "2026-11-25T10:00:00+00:00",
            "completed_at": "2026-11-25T10:00:01+00:00", "suspended_ms": 0, "continuation_of": None,
            "record_version": 1, "experiment_id": None, "task_id": None, "attempt": None, "claim_epoch": None,
            "server_incarnation": None,
            "record_json": json.dumps({"turn_output": {"turn_key": key, "success": True},
                                       "execution_records": [{"contract_version": 1, "command_call_id": "c-1",
                                                              "parent_call_id": None, "command_ordinal": 0,
                                                              "span_id": spans_of}],
                                       "routing_events": []})}


def _as_span(span):
    return tracing.Span(span_id=span["span_id"], trace_id=span["trace_id"], name=span["name"], kind=span["kind"],
                        channel_id=span["channel_id"], parent_span_id=span["parent_span_id"],
                        command_name=span["command_name"], context=span["context"], start_ns=span["start_ns"],
                        end_ns=span["end_ns"], status=span["status"], attributes=span["attributes"])


@pytest.fixture
def tree_server(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "span_tree_workflow"
    workflow.mkdir()
    db_path = state_paths.observability_db(str(workflow))
    store = obs.ObservabilityStore(db_path)
    redactor = store._store_redactor()
    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(conn, _row(NEW_TURN, "n-cmd"), [], redactor)
        store.upsert_span_rows(conn, [_as_span(span) for span in new_turn()], redactor)
        conn.commit()
    finally:
        conn.close()
    srv = run_chatbot_server.ChatbotServer(db_path, port=0, workflow_path=str(workflow))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def test_span_tree_dom(tree_server):
    """The debug page draws a turn by its stages, with its parts, numbered steps, commands and the summary call
    labelled by its purpose; a span node's feedback anchor is its own id; the turn-wide record sits on the Run stage."""
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_span_tree_dom.cjs")
    result = subprocess.run(
        ["node", str(script), dependency, f"http://127.0.0.1:{tree_server.port}/?token={tree_server.token}",
         NEW_TURN],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
