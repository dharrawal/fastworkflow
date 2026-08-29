"""Phase 7 §2.7 / ruling I8: distillation reads the in-process action log.

The cwd `action.jsonl` mirror is retired. Distillation now compares teacher and
student passes off `WorkflowExecutionContext.action_log`, which is a single live
list cleared between passes. That makes two things load-bearing, and both are
covered here:

* **Snapshot discipline** — a pass's actions must be copied out (`list(...)`).
  An aliased reference would be emptied by the next pass's clear, silently
  making every student trajectory equal to the teacher's (no divergence ever
  detected, no insights ever extracted).
* **Clear-point discipline** — the log must be cleared at distillation entry
  and between passes, or the previous turn's actions leak into the teacher pass.

These drive a real WorkflowExecutionContext bound to a real workflow, and append
through the production `_append_action_record` path.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.distillation import DistillationSession, distill_message
from fastworkflow.workflow_agent import _append_action_record
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
def distillation_session(initialized_fastworkflow, todo_workflow_path):
    """A DistillationSession over a real WEC bound to the todo workflow."""
    ctx = WorkflowExecutionContext(run_as_agent=True)
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"distill-actionlog-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(workflow)
    # Distillation reads the workflow off the active-workflow stack, which the
    # run loop normally pushes.
    ctx.push_active_workflow(workflow)
    yield DistillationSession(ctx), ctx
    ctx.clear_workflow_stack()
    ctx.close()


def _action(command_name: str, **params) -> dict:
    return {
        "command": command_name,
        "command_name": command_name,
        "parameters": params,
        "response": f"ran {command_name}",
    }


def test_pass_snapshot_survives_the_next_pass_clear(distillation_session):
    """`list(ctx.action_log)` must detach from the live list (ruling I8)."""
    ds, ctx = distillation_session
    initial = ds.snapshot_workflow_state()

    # Teacher pass: two actions, snapshotted the way _run_agent_pass does.
    _append_action_record(ctx, _action("list_tasks"))
    _append_action_record(ctx, _action("complete_task", id="1"))
    teacher_actions = list(ctx.action_log)

    # Between passes distillation restores state, which clears the live log.
    ds.restore_workflow_state(initial)

    assert ctx.action_log == []
    # The snapshot is a copy: an alias would be empty here, and the student pass
    # would then refill it, making teacher == student for every comparison.
    assert len(teacher_actions) == 2
    assert [a["command_name"] for a in teacher_actions] == [
        "list_tasks",
        "complete_task",
    ]


def test_per_pass_action_counts_stay_disjoint(distillation_session):
    """Teacher and student snapshots hold only their own pass's actions."""
    ds, ctx = distillation_session
    initial = ds.snapshot_workflow_state()

    _append_action_record(ctx, _action("list_tasks"))
    _append_action_record(ctx, _action("complete_task", id="1"))
    teacher_actions = list(ctx.action_log)

    ds.restore_workflow_state(initial)

    _append_action_record(ctx, _action("list_tasks"))
    student_actions = list(ctx.action_log)

    assert len(teacher_actions) == 2
    assert len(student_actions) == 1
    # Divergence is detectable precisely because the passes did not merge.
    diverged, _summary = ds.compare_trajectories(teacher_actions, student_actions)
    assert diverged


def test_distill_message_sheds_prior_turn_actions_at_entry(
    distillation_session, monkeypatch
):
    """Entry clear: turn N-1's actions must not be attributed to the teacher pass.

    The distillation branch bypasses `_run_agent` (the only other clear point),
    so `distill_message` clears on the way in. The agent passes are scripted here
    to keep the test LLM-free; each records how many actions were already in the
    live log when it started, which is what the clear points govern.
    """
    ds, ctx = distillation_session
    entry_lengths: list[int] = []

    def scripted_pass(self, message, **kwargs):
        # Deliberately does NOT clear: the observed length reflects only what the
        # caller (distill_message / restore_workflow_state) left behind.
        entry_lengths.append(len(self.chat_session.action_log))
        _append_action_record(self.chat_session, _action("list_tasks"))
        response = fastworkflow.CommandResponse(response="done")
        return (
            fastworkflow.CommandOutput(command_response=response),
            {},
            list(self.chat_session.action_log),
            [],
        )

    monkeypatch.setattr(DistillationSession, "_run_agent_pass", scripted_pass)

    # Residue from the previous turn, as the live log would hold it.
    _append_action_record(ctx, _action("stale_previous_turn_action"))
    assert len(ctx.action_log) == 1

    result = distill_message(ctx, "list my tasks")

    assert result.command_output.command_response.response == "done"
    # Both passes started clean: entry clear before the teacher, restore-driven
    # clear before the student.
    assert entry_lengths == [0, 0]


class _ScriptedAgent:
    """Stands in for the DSPy ReAct agent, appending the actions its pass ran."""

    def __init__(self, chat_session, command_names: list[str]):
        self._chat_session = chat_session
        self._command_names = command_names
        self.current_trajectory: dict = {}

    def __call__(self, **_kwargs):
        for name in self._command_names:
            _append_action_record(self._chat_session, _action(name))
        self.current_trajectory = {"thought_0": "scripted"}
        return type("AgentResult", (), {"final_answer": "done"})()


def test_run_agent_pass_returns_a_detached_action_snapshot(
    distillation_session, monkeypatch
):
    """Covers the production snapshot in `_run_agent_pass` (ruling I8).

    Only the LLM boundaries are scripted — the clear at pass start and the
    `list(...)` snapshot at pass end are the real lines under test. Returning the
    live list instead of a copy would leave the teacher holding the student's
    actions, so the two passes below would compare equal.
    """
    ds, ctx = distillation_session
    pass_commands = [["list_tasks", "complete_task"], ["list_tasks"]]

    def scripted_agent(chat_session, **_kwargs):
        return _ScriptedAgent(chat_session, pass_commands.pop(0))

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.initialize_workflow_tool_agent", scripted_agent
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do", lambda session: "commands"
    )
    monkeypatch.setattr(
        "fastworkflow.utils.dspy_utils.get_lm", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        ctx, "_call_agent_with_retry", lambda agent_call, lm=None: agent_call()
    )
    monkeypatch.setattr(
        ctx, "summarize_and_record_turn", lambda *args, **kwargs: ("summary", None)
    )

    initial = ds.snapshot_workflow_state()
    _, _, teacher_actions, _ = ds._run_agent_pass(
        "list my tasks",
        agent_lm_role="LLM_TEACHER_AGENT",
        agent_api_key_role="LITELLM_API_KEY_TEACHER_AGENT",
        planner_lm_role="LLM_TEACHER_PLANNER",
        planner_api_key_role="LITELLM_API_KEY_TEACHER_PLANNER",
    )
    ds.restore_workflow_state(initial)
    _, _, student_actions, _ = ds._run_agent_pass(
        "list my tasks",
        agent_lm_role="LLM_STUDENT_AGENT",
        agent_api_key_role="LITELLM_API_KEY_STUDENT_AGENT",
        planner_lm_role="LLM_STUDENT_PLANNER",
        planner_api_key_role="LITELLM_API_KEY_STUDENT_PLANNER",
    )

    assert [a["command_name"] for a in teacher_actions] == [
        "list_tasks",
        "complete_task",
    ]
    assert [a["command_name"] for a in student_actions] == ["list_tasks"]
    diverged, _summary = ds.compare_trajectories(teacher_actions, student_actions)
    assert diverged


def test_action_records_never_write_action_jsonl(
    distillation_session, tmp_path, monkeypatch
):
    """The cwd action.jsonl mirror is gone, including the no-log-available path."""
    _ds, ctx = distillation_session
    monkeypatch.chdir(tmp_path)

    _append_action_record(ctx, _action("list_tasks"))
    # An object exposing neither the WEC nor a core used to fall back to a file
    # append; it now no-ops.
    _append_action_record(object(), _action("list_tasks"))

    assert len(ctx.action_log) == 1
    assert not (tmp_path / "action.jsonl").exists()
    strays = list(tmp_path.iterdir())
    assert not strays, f"appending an action record wrote to the cwd: {strays}"


# ---------------------------------------------------------------------------
# Pass separation and the run record — fix-kw7.11 / fix-sb8.2
#
# The origin symptom of kw7.11 is that both passes emitted into one trace with
# nothing marking the boundary. The fix is two things at once (design §8): the
# indexed `spans.distillation_pass` column, and `fw.distill.run` /
# `fw.distill.pass` as REAL spans that parent each pass's work, so the pass is
# a structural fact in the waterfall rather than only a filter [DR2]. Both
# halves are asserted here against a real SQLite sink, and so is the run record
# they key on.
#
# Only the LLM boundaries are scripted; `distill_message`, `_run_agent_pass`,
# `tracing.start_span` and the store's writer thread are the real code.
# ---------------------------------------------------------------------------

import json
import sqlite3

from fastworkflow import tracing


class _SpanEmittingScriptedAgent(_ScriptedAgent):
    """`_ScriptedAgent` that also emits its pass's agent span for real.

    It stands exactly where DSPy's ReAct call stands, so the emission runs
    through `tracing.start_span` — the one line that stamps
    `spans.distillation_pass` — instead of a hand-written span row.
    """

    def __call__(self, **kwargs):
        span = tracing.start_span(
            self._chat_session,
            tracing.SPAN_AGENT_EXECUTE,
            attributes={"agent_input": "scripted"},
        )
        try:
            return super().__call__(**kwargs)
        finally:
            tracing.end_span(self._chat_session, span)


def _rows(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


@pytest.fixture
def observed_wec(initialized_fastworkflow, todo_workflow_path, tmp_path):
    """A real WEC over the todo workflow, writing to a real observability DB."""
    from fastworkflow.observability_store import SQLiteTraceSink

    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    ctx = WorkflowExecutionContext(run_as_agent=True)
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"distill-obs-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(workflow)
    ctx.push_active_workflow(workflow)
    ctx.set_trace_sink(sink)
    ctx.bind_observability_identity(channel_id="distill-pass-channel")
    try:
        yield ctx, sink, db_path
    finally:
        ctx.clear_workflow_stack()
        ctx.close()
        sink.close()


def _script_llm_boundaries(monkeypatch, ctx, agent_factory):
    """Script only the LLM-touching boundaries of an agent pass."""
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.initialize_workflow_tool_agent", agent_factory
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do", lambda session: "commands"
    )
    monkeypatch.setattr(
        "fastworkflow.utils.dspy_utils.get_lm", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        ctx, "_call_agent_with_retry", lambda agent_call, lm=None: agent_call()
    )
    monkeypatch.setattr(
        ctx, "summarize_and_record_turn", lambda *args, **kwargs: ("summary", None)
    )


@pytest.fixture
def distilled_turn(observed_wec, monkeypatch):
    """One real two-pass distillation over an open turn, flushed to the DB.

    Both passes run the same command, so the comparison finds no divergence and
    no extraction LLM is reached.
    """
    ctx, sink, db_path = observed_wec
    _script_llm_boundaries(
        monkeypatch,
        ctx,
        lambda chat_session, **_kwargs: _SpanEmittingScriptedAgent(
            chat_session, ["list_tasks"]
        ),
    )

    ctx._begin_turn("list my tasks")
    turn_key = ctx.current_turn_key
    result = distill_message(ctx, "list my tasks")
    sink.close()  # flush the writer thread before anything reads the DB
    return ctx, db_path, turn_key, result


def test_a_distilled_turn_writes_one_run_row_and_two_pass_rows(distilled_turn):
    """fix-sb8.2: one distillation_runs row per compared message, one
    distillation_passes row per pass — the shape §4 grows to N passes by
    adding rows, never columns."""
    _ctx, db_path, turn_key, result = distilled_turn

    runs = _rows(db_path, "SELECT * FROM distillation_runs")
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == result.run_id
    assert run["turn_key"] == turn_key  # == spans.trace_id == turns.turn_key
    assert run["user_message"] == "list my tasks"
    assert run["channel_id"] == "distill-pass-channel"
    assert run["started_at"] and run["completed_at"]
    assert json.loads(run["run_json"])["run_id"] == result.run_id
    # Both passes now record a state_fingerprint (fix-sb8.3), and they entered
    # on the same one — so the verdict is comparable rather than the
    # "evidence incomplete" this asserted while the fingerprints were missing.
    assert (run["comparable"], run["comparable_reason"]) == (1, None)
    assert run["fingerprint_teacher"] == run["fingerprint_student"]
    assert run["fingerprint_teacher"]

    passes = _rows(db_path, "SELECT * FROM distillation_passes ORDER BY seq")
    assert len(passes) == 2
    assert [p["pass_label"] for p in passes] == ["teacher", "student"]
    assert [p["role"] for p in passes] == ["teacher", "student"]
    assert [p["seq"] for p in passes] == [0, 1]
    for row in passes:
        assert row["run_id"] == result.run_id
        assert row["trace_id"] == turn_key
        assert row["wall_ms"] is not None
        assert row["first_span_id"]


def test_every_span_of_a_pass_carries_that_passs_label(distilled_turn):
    """[DR8]: the two passes' span sets are disjoint and both non-empty —
    the assertion (a)'s advocate wanted, in the language this repo can test."""
    _ctx, db_path, turn_key, _result = distilled_turn

    spans = _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_key,))
    by_pass: dict = {}
    for span in spans:
        by_pass.setdefault(span["distillation_pass"], []).append(span)

    teacher = {s["span_id"] for s in by_pass.get("teacher", [])}
    student = {s["span_id"] for s in by_pass.get("student", [])}
    assert teacher and student
    assert not (teacher & student)

    # The pass's own agent work carries the label, not just the wrapper.
    assert {
        s["name"] for s in by_pass["teacher"]
    } == {tracing.SPAN_DISTILL_PASS, tracing.SPAN_AGENT_EXECUTE}
    assert {
        s["name"] for s in by_pass["student"]
    } == {tracing.SPAN_DISTILL_PASS, tracing.SPAN_AGENT_EXECUTE}

    # The run-level spans stay NULL: fw.distill.run and the fw.turn root belong
    # to the turn, not to either pass ([DR7]).
    run_level = {s["name"] for s in by_pass.get(None, [])}
    assert tracing.SPAN_DISTILL_RUN in run_level
    assert tracing.SPAN_TURN in run_level


def test_no_span_of_one_pass_hangs_under_the_others_pass_span(distilled_turn):
    """The structural half of §8: each pass's work descends from its own
    fw.distill.pass span, which descends from the one fw.distill.run span."""
    _ctx, db_path, turn_key, _result = distilled_turn

    spans = _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_key,))
    by_id = {s["span_id"]: s for s in spans}

    run_spans = [s for s in spans if s["name"] == tracing.SPAN_DISTILL_RUN]
    assert len(run_spans) == 1
    assert run_spans[0]["parent_span_id"] == tracing.root_span_id(turn_key)

    pass_spans = {
        s["distillation_pass"]: s
        for s in spans
        if s["name"] == tracing.SPAN_DISTILL_PASS
    }
    assert set(pass_spans) == {"teacher", "student"}
    for label, span in pass_spans.items():
        assert span["parent_span_id"] == run_spans[0]["span_id"]
        # Deterministic [DR51], so the ask_user close can recompute it.
        assert span["span_id"] == tracing.distill_pass_span_id(turn_key, label)

    def ancestors(span: dict) -> set:
        seen = set()
        parent = span["parent_span_id"]
        while parent and parent in by_id and parent not in seen:
            seen.add(parent)
            parent = by_id[parent]["parent_span_id"]
        return seen

    for label, other in (("teacher", "student"), ("student", "teacher")):
        work = [
            s
            for s in spans
            if s["distillation_pass"] == label and s["name"] != tracing.SPAN_DISTILL_PASS
        ]
        assert work
        for span in work:
            chain = ancestors(span)
            assert pass_spans[label]["span_id"] in chain
            assert pass_spans[other]["span_id"] not in chain


def test_an_ordinary_turn_is_completely_unaffected(observed_wec):
    """Distillation costs the 99.9% of turns that are not distilled nothing:
    distillation_pass stays NULL and no distillation row is written."""
    ctx, sink, db_path = observed_wec

    ctx._begin_turn("plain turn")
    turn_key = ctx.current_turn_key
    span = tracing.start_span(ctx, tracing.SPAN_AGENT_EXECUTE, attributes={"x": 1})
    tracing.end_span(ctx, span)
    sink.close()

    spans = _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_key,))
    assert spans
    assert all(s["distillation_pass"] is None for s in spans)
    assert _rows(db_path, "SELECT * FROM distillation_runs") == []
    assert _rows(db_path, "SELECT * FROM distillation_passes") == []


def test_a_teacher_that_raises_still_writes_its_run_row_and_still_raises(
    observed_wec, monkeypatch
):
    """The half run (§18, fix-sb8.2): an unset LLM_TEACHER_* makes get_lm raise
    before the student ever runs. That run is still a fact worth seeing in the
    list, and the raise must reach the caller exactly as it does today."""
    ctx, sink, db_path = observed_wec
    _script_llm_boundaries(
        monkeypatch,
        ctx,
        lambda chat_session, **_kwargs: _SpanEmittingScriptedAgent(
            chat_session, ["list_tasks"]
        ),
    )

    def raising_get_lm(model_env_var, *args, **kwargs):
        raise ValueError(
            f"DSPy Language Model not provided. Set {model_env_var} environment variable."
        )

    monkeypatch.setattr("fastworkflow.utils.dspy_utils.get_lm", raising_get_lm)

    ctx._begin_turn("list my tasks")
    turn_key = ctx.current_turn_key
    with pytest.raises(ValueError, match="LLM_TEACHER_PLANNER"):
        distill_message(ctx, "list my tasks")
    sink.close()

    runs = _rows(db_path, "SELECT * FROM distillation_runs")
    assert len(runs) == 1
    assert runs[0]["turn_key"] == turn_key
    assert (runs[0]["comparable"], runs[0]["comparable_reason"]) == (
        0,
        "teacher-raised",
    )
    # A run that could not start never completed.
    assert runs[0]["completed_at"] is None

    # The teacher pass row is still written, and the student never ran.
    passes = _rows(db_path, "SELECT * FROM distillation_passes")
    assert [p["pass_label"] for p in passes] == ["teacher"]

    # The label did not leak past the failed pass.
    assert ctx.current_distillation_pass is None
