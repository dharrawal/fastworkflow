"""Unit tests for distillation's pure comparison/formatting/persistence logic.

These cover the LLM-free core of fastworkflow/distillation.py — trajectory and
plan comparison, trajectory formatting for the insight LLM, and numbered-insight
persistence — without any network/LLM dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastworkflow.distillation import (
    DistillationSession,
    PlanningStep,
    DistillationResult,
)


@pytest.fixture
def session() -> DistillationSession:
    # The comparison/formatting methods don't touch WEC state, so build an
    # instance without running __init__ (no WEC needed).
    return DistillationSession.__new__(DistillationSession)


# ---------------------------------------------------------------------------
# _action_signature / _format_action
# ---------------------------------------------------------------------------

def test_action_signature_is_order_independent_for_params():
    a1 = {"command_name": "cancel_order", "parameters": {"id": "1", "reason": "x"}}
    a2 = {"command_name": "cancel_order", "parameters": {"reason": "x", "id": "1"}}
    # Same command + same params (different dict order) -> identical signature.
    assert DistillationSession._action_signature(a1) == DistillationSession._action_signature(a2)


def test_action_signature_differs_on_params():
    a1 = {"command_name": "cancel_order", "parameters": {"id": "1"}}
    a2 = {"command_name": "cancel_order", "parameters": {"id": "2"}}
    assert DistillationSession._action_signature(a1) != DistillationSession._action_signature(a2)


def test_format_action_with_and_without_params():
    assert DistillationSession._format_action(
        {"command_name": "get_order", "parameters": {"id": "1"}}
    ) == 'get_order({"id": "1"})'
    # No params -> bare command name.
    assert DistillationSession._format_action(
        {"command_name": "list_all", "parameters": {}}
    ) == "list_all"


# ---------------------------------------------------------------------------
# compare_trajectories (action-level divergence + is_valid_action filter)
# ---------------------------------------------------------------------------

def test_compare_trajectories_identical_no_divergence(session):
    actions = [
        {"command_name": "find_user", "parameters": {"email": "a@b.com"}},
        {"command_name": "get_order", "parameters": {"id": "1"}},
    ]
    diverged, summary = session.compare_trajectories(list(actions), list(actions))
    assert diverged is False
    assert summary == ""


def test_compare_trajectories_detects_extra_student_action(session):
    teacher = [{"command_name": "find_user", "parameters": {"email": "a@b.com"}}]
    student = [
        {"command_name": "find_user", "parameters": {"email": "a@b.com"}},
        {"command_name": "get_user_details", "parameters": {"user_id": "u1"}},
    ]
    diverged, summary = session.compare_trajectories(teacher, student)
    assert diverged is True
    assert "get_user_details" in summary


def test_compare_trajectories_filters_ask_user_and_error_correction(session):
    # ask_user records (agent_query key) and ErrorCorrection/* must be excluded
    # from the command-level divergence comparison.
    teacher = [
        {"agent_query": "email?", "user_response": "a@b.com"},           # ask_user
        {"command_name": "ErrorCorrection/abort", "parameters": {}},     # abort
        {"command_name": "find_user", "parameters": {"email": "a@b.com"}},
    ]
    student = [
        {"command_name": "find_user", "parameters": {"email": "a@b.com"}},
    ]
    # After filtering, both reduce to the same single find_user action.
    diverged, summary = session.compare_trajectories(teacher, student)
    assert diverged is False
    assert summary == ""


# ---------------------------------------------------------------------------
# compare_planning_traces
# ---------------------------------------------------------------------------

def test_compare_planning_traces_identical(session):
    t = [PlanningStep(0, "q", ["step a", "step b"])]
    s = [PlanningStep(0, "q", ["step a", "step b"])]
    diverged, summary = session.compare_planning_traces(t, s)
    assert diverged is False


def test_compare_planning_traces_detects_difference(session):
    t = [PlanningStep(0, "q", ["step a", "step b"])]
    s = [PlanningStep(0, "q", ["step a", "different"])]
    diverged, summary = session.compare_planning_traces(t, s)
    assert diverged is True
    assert "Step 0" in summary


def test_compare_planning_traces_empty_both(session):
    assert session.compare_planning_traces([], []) == (False, "")


# ---------------------------------------------------------------------------
# _format_trajectory_for_llm (must surface ask_user steps + observations)
# ---------------------------------------------------------------------------

def test_format_trajectory_includes_ask_user_and_observations():
    trajectory = {
        "thought_0": "need email",
        "tool_name_0": "ask_user",
        "tool_args_0": {"clarification_request": "email?"},
        "observation_0": "mia@example.com",       # user's answer
        "thought_1": "look up",
        "tool_name_1": "execute_workflow_query",
        "tool_args_1": {"command": "find_user"},
        "observation_1": "user id: u1",
    }
    out = DistillationSession._format_trajectory_for_llm(trajectory)
    # ask_user step and the user's answer must reach the insight LLM.
    assert "ask_user" in out
    assert "mia@example.com" in out
    assert "execute_workflow_query" in out
    assert "user id: u1" in out


def test_format_trajectory_truncates_long_observations():
    trajectory = {
        "thought_0": "t",
        "tool_name_0": "x",
        "tool_args_0": {},
        "observation_0": "A" * 900,
    }
    out = DistillationSession._format_trajectory_for_llm(trajectory)
    assert "[truncated]" in out


# ---------------------------------------------------------------------------
# _append_numbered_insights (numbering continuity across appends)
# ---------------------------------------------------------------------------

def test_append_numbered_insights_starts_at_one_then_continues(tmp_path: Path):
    f = tmp_path / "insights.md"
    fmt = "{num}. {insight}\n"
    pattern = r"^(\d+)\.\s"
    header = "# Insights\n\n"

    DistillationSession._append_numbered_insights(["first"], f, header, pattern, fmt)
    DistillationSession._append_numbered_insights(["second", "third"], f, header, pattern, fmt)

    content = f.read_text(encoding="utf-8")
    assert "1. first" in content
    assert "2. second" in content
    assert "3. third" in content
    # Header written exactly once.
    assert content.count("# Insights") == 1


def test_append_numbered_planning_insights_uses_its_own_pattern(tmp_path: Path):
    f = tmp_path / "planning.md"
    fmt = "## {num}. {insight}\n\n"
    pattern = r"^## (\d+)\."
    header = "# Planning\n\n"

    DistillationSession._append_numbered_insights(["a"], f, header, pattern, fmt)
    DistillationSession._append_numbered_insights(["b"], f, header, pattern, fmt)

    content = f.read_text(encoding="utf-8")
    assert "## 1. a" in content
    assert "## 2. b" in content


# ---------------------------------------------------------------------------
# DistillationResult
# ---------------------------------------------------------------------------

def test_distillation_result_total_insights():
    r = DistillationResult(
        command_output=None,
        planning_insights_extracted=2,
        execution_insights_extracted=3,
    )
    assert r.insights_extracted == 5


# ---------------------------------------------------------------------------
# Comparability evidence — fix-sb8.3 (design §5, §6)
#
# A divergence is evidence of a student mistake only if both passes started
# identical. These cover the two fingerprint projections `[DR47]`, the deep
# copy `snapshot_workflow_state` needs before a fingerprint means anything
# (§5), the two `restore_ok_*` columns and their two distinct baselines (§6.2),
# the cache-asymmetry confound `[DR16]`, and `[DR48]`'s rule that fix-sb8.3
# may only ever write NULL or 0 into `isolation_verified`.
#
# Real WEC over `tests/todo_list_workflow`, real `SQLiteTraceSink`, real
# `distill_message` / `_run_agent_pass` / `tracing.start_span` / writer thread.
# Only the LLM boundaries are scripted, extending the precedent in
# `tests/test_distillation_action_log.py`.
# ---------------------------------------------------------------------------

import json
import sqlite3
import uuid

import fastworkflow
from fastworkflow import tracing
from fastworkflow.distillation import (
    _canonical,
    distill_message,
    prompt_fingerprint,
    state_fingerprint,
)
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


def _bound_wec(todo_workflow_path: str) -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=True)
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"distill-fingerprint-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(workflow)
    ctx.push_active_workflow(workflow)
    return ctx


@pytest.fixture
def wec(initialized_fastworkflow, todo_workflow_path):
    """A real WEC bound to the todo workflow, with no observability sink."""
    ctx = _bound_wec(todo_workflow_path)
    yield ctx
    ctx.clear_workflow_stack()
    ctx.close()


@pytest.fixture
def observed_wec(initialized_fastworkflow, todo_workflow_path, tmp_path):
    """A real WEC writing to a real observability DB."""
    from fastworkflow.observability_store import SQLiteTraceSink

    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    ctx = _bound_wec(todo_workflow_path)
    ctx.set_trace_sink(sink)
    ctx.bind_observability_identity(channel_id="distill-comparability-channel")
    try:
        yield ctx, sink, db_path
    finally:
        ctx.clear_workflow_stack()
        ctx.close()
        sink.close()


def _rows(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _action(command_name: str, **params) -> dict:
    return {
        "command": command_name,
        "command_name": command_name,
        "parameters": params,
        "response": f"ran {command_name}",
    }


class _PassAgent:
    """Stands where DSPy's ReAct call stands, emitting its pass's spans for real.

    The `fw.llm.call` span carries exactly the attribute shape
    `utils/dspy_logger.py` writes — `model` (:374), `usage` as a JSON *string*
    (:425), `cost` (:426), `cache_hit` (:429-430) — because §6.3's per-pass
    rollup is a query over those attributes and nothing else.

    Each command it "runs" emits a real `fw.command.execute` span in
    `CommandExecutor.invoke_command`'s exact shape — `raw_command` at open,
    `command_name`/`context` plus `parameters`/`response_text`/`success` at
    close (`command_executor.py:44-48,116-122,158-160`) — as well as the
    action record. [DR17] aligns over those spans, so a double that emitted
    only the action record would make every pass look actionless.
    """

    def __init__(self, chat_session, command_names, *, cache_hit, mutate=None):
        self._chat_session = chat_session
        self._command_names = command_names
        self._cache_hit = cache_hit
        self._mutate = mutate
        self.current_trajectory: dict = {}

    def __call__(self, **_kwargs):
        span = tracing.start_span(
            self._chat_session,
            tracing.SPAN_LLM_CALL,
            kind=tracing.KIND_LLM,
            attributes={"model": "scripted/model", "messages": "[]"},
        )
        tracing.end_span(
            self._chat_session,
            span,
            attributes={
                "usage": json.dumps({"total_tokens": 11}),
                "cost": 0.25,
                "cache_hit": self._cache_hit,
            },
        )
        for name in self._command_names:
            command_span = tracing.start_span(
                self._chat_session,
                tracing.SPAN_COMMAND_EXECUTE,
                kind=tracing.KIND_TOOL,
                attributes={"raw_command": name},
            )
            tracing.end_span(
                self._chat_session,
                command_span,
                command_name=name,
                attributes={
                    "parameters": {},
                    "response_text": f"ran {name}",
                    "success": True,
                },
            )
            _append_action_record(self._chat_session, _action(name))
        if self._mutate is not None:
            self._mutate(self._chat_session)
        self.current_trajectory = {"thought_0": "scripted"}
        return type("AgentResult", (), {"final_answer": "done"})()


def _script_llm_boundaries(monkeypatch, ctx, agent_factory):
    """Script only the LLM-touching boundaries of a distillation run."""
    from fastworkflow.distillation import DistillationSession

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
    # The two insight extractors are LLM calls like any other; a run that
    # diverges must still reach its state-restoration step without one.
    monkeypatch.setattr(
        DistillationSession, "extract_insights", lambda *a, **k: []
    )
    monkeypatch.setattr(
        DistillationSession, "extract_planning_insights", lambda *a, **k: []
    )


def _run_distillation(ctx, sink, monkeypatch, agents, message="list my tasks"):
    """One real two-pass distillation, flushed to the DB."""
    queued = list(agents)
    _script_llm_boundaries(
        monkeypatch, ctx, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )
    ctx._begin_turn(message)
    turn_key = ctx.current_turn_key
    result = distill_message(ctx, message)
    sink.close()
    return turn_key, result


# ---------------------------------------------------------------------------
# _canonical / _digest — the BLOCKING `default=str` finding [DR47]
# ---------------------------------------------------------------------------

class _Widget:
    """A live application object, exactly as a context dict may hold one."""


def test_canonical_renders_a_live_object_as_a_type_token_not_an_address():
    # `fastworkflow/workflow.py` defines no __repr__/__str__, so `default=str`
    # rendered such a value as '<... object at 0x...>' — a heap address, which
    # no two runs could ever agree on and no replay could ever reproduce.
    rendered = _canonical(_Widget())
    assert rendered == "<type:_Widget>"
    assert "0x" not in rendered


def test_state_fingerprint_is_blind_to_object_identity(wec):
    workflow = wec.get_active_workflow()
    workflow.context["widget"] = _Widget()
    first = state_fingerprint(wec)
    # A different object, at a different address, holding the same (absent)
    # data. An address-bearing digest would differ here on every run.
    workflow.context["widget"] = _Widget()
    assert state_fingerprint(wec) == first


def test_state_fingerprint_drops_the_live_app_workflow_handle(wec):
    # cme._context always carries the live Workflow (`workflow_execution_context
    # .py:1036`); it is named in §6.1's exclusion list so the structural-token
    # rule is not the only thing between the fingerprint and an address.
    assert "app_workflow" in wec.cme_workflow.context
    payload = _canonical(wec.cme_workflow.context)
    assert "app_workflow" not in payload


def test_canonical_drops_wall_clock_and_entry_written_keys():
    payload = _canonical(
        {
            "kept": "yes",
            "created_at": "2026-01-01",
            "run_ts": 1,
            "started_ns": 2,
            "updated_by": "x",
            "raw_user_message": "hello",
            "is_user_command": True,
            "stored_parameters": {},
            "NLU_Pipeline_Stage": "one",
        }
    )
    assert payload == {"kept": "yes"}


def test_canonical_survives_a_cycle_and_renders_numbers_as_strings():
    node: dict = {"n": 3, "f": 1.5}
    node["self"] = node
    payload = _canonical(node)
    assert payload["n"] == "3"
    assert payload["f"] == "1.5"
    assert payload["self"] == "<cycle>"


# ---------------------------------------------------------------------------
# The two projections [DR47]
# ---------------------------------------------------------------------------

def test_state_fingerprint_excludes_conversation_history(wec):
    before = state_fingerprint(wec)
    # Every pass appends its own LLM-generated summary from inside the pass, so
    # a history-bearing hash could never compare equal across passes — which
    # would make `material` and `restore_ok` constants.
    wec.conversation_history.messages.append({"conversation summary": "teacher's"})
    assert state_fingerprint(wec) == before


def test_prompt_fingerprint_is_taken_at_an_explicit_history_bound(wec):
    bound = len(wec.conversation_history.messages)
    before = prompt_fingerprint(wec, history_bound=bound)
    wec.conversation_history.messages.append({"conversation summary": "teacher's"})
    # At the pass's own entry bound the appended summary is invisible...
    assert prompt_fingerprint(wec, history_bound=bound) == before
    # ...and at the grown bound it is not.
    assert prompt_fingerprint(wec, history_bound=bound + 1) != before


# ---------------------------------------------------------------------------
# (c) the deep copy `snapshot_workflow_state` needs (§5)
# ---------------------------------------------------------------------------

def test_snapshot_deep_copies_the_context_so_a_restore_actually_restores(wec):
    """Would FAIL without the deepcopy: `Workflow._to_dict()` returns
    `self._context` by reference (`workflow.py:450`), so the snapshot would
    alias the live dict, the restore would be a no-op, and any fingerprint
    across it would report agreement by construction."""
    ds = DistillationSession(wec)
    workflow = wec.get_active_workflow()
    workflow.context["pre_existing"] = "kept"

    initial = ds.snapshot_workflow_state()
    entry_fingerprint = state_fingerprint(wec)
    assert initial["workflow_dict"]["workflow_context"] is not workflow._context

    # The teacher pass writes into the live context.
    workflow.context["teacher_wrote"] = "this"
    assert state_fingerprint(wec) != entry_fingerprint

    ds.restore_workflow_state(initial)

    assert "teacher_wrote" not in wec.get_active_workflow().context
    assert wec.get_active_workflow().context["pre_existing"] == "kept"
    assert state_fingerprint(wec) == entry_fingerprint


def test_snapshot_keeps_live_handles_by_reference(wec):
    """The deep copy must not clone `app_workflow`: `intent_detection.py:34`
    reads that object back, and a clone would re-register a duplicate Workflow
    under the same id on restore."""
    ds = DistillationSession(wec)
    workflow = wec.get_active_workflow()
    assert wec.cme_workflow.context["app_workflow"] is workflow

    snapshot = ds.snapshot_workflow_state()
    assert snapshot["cme_dict"]["workflow_context"]["app_workflow"] is workflow

    ds.restore_workflow_state(snapshot)
    assert wec.cme_workflow.context["app_workflow"] is workflow


# ---------------------------------------------------------------------------
# (d) two restore_ok columns, two baselines (§6.2)
# ---------------------------------------------------------------------------

def test_the_two_restore_baselines_are_different_assertions(wec):
    """`restore_ok_pre_student` is measured against the PRE-teacher entry
    state; `restore_ok_post_compare` against the teacher's EXIT state. At each
    restore site exactly one of them can hold, which is why revision 1's single
    column would have reported 0 on every divergent run."""
    ds = DistillationSession(wec)
    workflow = wec.get_active_workflow()

    initial = ds.snapshot_workflow_state()
    pre_teacher = state_fingerprint(wec)

    workflow.context["teacher_wrote"] = "this"
    teacher_exit = state_fingerprint(wec)
    teacher_final = ds.snapshot_workflow_state()
    assert teacher_exit != pre_teacher

    # Site 1 (distillation.py: restore toward the pre-teacher snapshot).
    ds.restore_workflow_state(initial)
    assert ds._restore_matches(pre_teacher) == 1
    assert ds._restore_matches(teacher_exit) == 0

    # Site 2 (the divergence / student-failure restores).
    ds.restore_workflow_state(teacher_final)
    assert ds._restore_matches(teacher_exit) == 1
    assert ds._restore_matches(pre_teacher) == 0


# ---------------------------------------------------------------------------
# (a) + (e) end to end: equal entry fingerprints, comparable, isolation NULL
# ---------------------------------------------------------------------------

def test_two_passes_from_one_entry_state_record_equal_entry_fingerprints(
    observed_wec, monkeypatch
):
    ctx, sink, db_path = observed_wec
    turn_key, result = _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )

    passes = _rows(db_path, "SELECT * FROM distillation_passes ORDER BY seq")
    assert [p["pass_label"] for p in passes] == ["teacher", "student"]
    entries = {p["entry_fingerprint"] for p in passes}
    assert len(entries) == 1 and entries != {None}
    for row in passes:
        assert row["exit_fingerprint"]
        assert row["entry_prompt_fingerprint"] == row["exit_prompt_fingerprint"]
        assert row["history_bound"] is not None
        assert json.loads(row["entry_inputs_json"])["raw_user_message"] == (
            "list my tasks"
        )

    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["run_id"] == result.run_id
    assert (run["comparable"], run["comparable_reason"]) == (1, None)
    assert run["fingerprint_teacher"] == run["fingerprint_student"]
    assert run["fingerprint_teacher"] == passes[0]["entry_fingerprint"]
    # The restore toward the pre-teacher state landed; the post-compare site
    # never ran (the passes agreed), so its column stays NULL.
    assert run["restore_ok_pre_student"] == 1
    assert run["restore_ok_post_compare"] is None
    assert run["turn_key"] == turn_key


def test_a_run_whose_passes_entered_differently_is_not_comparable(wec, monkeypatch):
    """A deliberately mutated entry state must be visible as
    `comparable = 0 / 'fingerprint-differs'`, never as silence."""
    ds = DistillationSession(wec)
    workflow = wec.get_active_workflow()

    def mutate(chat_session):
        # Survives into the next pass precisely because no restore intervenes.
        chat_session.get_active_workflow().context["teacher_wrote"] = "this"

    agents = [
        lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False, mutate=mutate),
        lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
    ]
    queued = list(agents)
    _script_llm_boundaries(
        monkeypatch, wec, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )

    for label, role, seq, keys in (
        ("teacher", "teacher", 0, ("LLM_TEACHER_AGENT", "LLM_TEACHER_PLANNER")),
        ("student", "student", 1, ("LLM_STUDENT_AGENT", "LLM_STUDENT_PLANNER")),
    ):
        ds._run_agent_pass(
            "list my tasks",
            agent_lm_role=keys[0],
            agent_api_key_role=f"LITELLM_API_KEY_{role.upper()}_AGENT",
            planner_lm_role=keys[1],
            planner_api_key_role=f"LITELLM_API_KEY_{role.upper()}_PLANNER",
            pass_label=label,
            role=role,
            seq=seq,
        )

    assert workflow.context["teacher_wrote"] == "this"
    assert ds.pass_fingerprint("teacher", "entry_fingerprint") != (
        ds.pass_fingerprint("student", "entry_fingerprint")
    )
    verdict = ds.comparability_fields()
    assert verdict["comparable"] == 0
    assert verdict["comparable_reason"] == "fingerprint-differs"
    assert verdict["fingerprint_teacher"] != verdict["fingerprint_student"]


def test_a_pass_that_never_reached_a_boundary_is_evidence_incomplete(wec):
    """Silence is never read as agreement ([DR40]): one pass, or a missing
    fingerprint, is `evidence-incomplete` rather than comparable."""
    ds = DistillationSession(wec)
    ds._passes["teacher"] = {"seq": 0, "entry_fingerprint": "abc"}
    assert ds.comparability_fields()["comparable_reason"] == "evidence-incomplete"

    ds._passes["student"] = {"seq": 1, "entry_fingerprint": None}
    assert ds.comparability_fields()["comparable_reason"] == "evidence-incomplete"

    # Equal fingerprints, but the writer dropped spans inside a pass: the
    # evidence the divergence records rest on is incomplete ([DR49]).
    ds._passes["student"] = {
        "seq": 1,
        "entry_fingerprint": "abc",
        "spans_dropped_delta": 2,
    }
    assert ds.comparability_fields()["comparable_reason"] == "evidence-incomplete"

    ds._passes["student"]["spans_dropped_delta"] = 0
    assert ds.comparability_fields() == {
        "fingerprint_teacher": "abc",
        "fingerprint_student": "abc",
        "comparable": 1,
        "comparable_reason": None,
    }


# ---------------------------------------------------------------------------
# (f) cache asymmetry [DR16] — a query over existing fw.llm.call attributes
# ---------------------------------------------------------------------------

def test_cache_asymmetry_is_flagged_when_one_pass_hit_and_the_other_missed(
    observed_wec, monkeypatch
):
    ctx, sink, db_path = observed_wec
    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=True),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )

    passes = {
        row["pass_label"]: row
        for row in _rows(db_path, "SELECT * FROM distillation_passes")
    }
    assert (passes["teacher"]["cache_hits"], passes["teacher"]["cache_misses"]) == (1, 0)
    assert (passes["student"]["cache_hits"], passes["student"]["cache_misses"]) == (0, 1)
    # The nested json_extract on `usage` is what makes this non-NULL; the
    # single-level form returns NULL silently.
    assert passes["teacher"]["tokens"] == 11
    assert passes["student"]["cost_usd"] == pytest.approx(0.25)

    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["cache_asymmetric"] == 1
    # A cache hit returns the same completion, so the trajectory is still
    # comparable — it is a cost confound, not a comparability one.
    assert run["comparable"] == 1


def test_two_passes_that_both_missed_the_cache_are_not_asymmetric(
    observed_wec, monkeypatch
):
    ctx, sink, db_path = observed_wec
    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )
    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["cache_asymmetric"] == 0


# ---------------------------------------------------------------------------
# (e) [DR48] — fix-sb8.3 may write NULL or 0 into isolation_verified, never 1
# ---------------------------------------------------------------------------

def test_isolation_verified_is_never_1(observed_wec, monkeypatch):
    ctx, sink, db_path = observed_wec
    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )
    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    # NULL until fix-35m.3's read-only surface check exists. A NULL is not
    # readable as a pass: the promotion view and replay both refuse on it.
    assert run["isolation_verified"] is None


def test_the_run_writer_refuses_an_isolation_verified_of_1(observed_wec):
    """Belt and braces on [DR48]: even asked directly, sb8 does not assert
    isolation it did not check."""
    ctx, sink, db_path = observed_wec
    ctx._begin_turn("list my tasks")
    ds = DistillationSession(ctx)
    ds._emit_run_record(
        user_message="list my tasks",
        comparable=0,
        comparable_reason="evidence-incomplete",
        isolation_verified=1,
        run_json="{}",
    )
    sink.close()

    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["isolation_verified"] is None


# ---------------------------------------------------------------------------
# The divergent path: restore_ok_post_compare has its own baseline
# ---------------------------------------------------------------------------

def test_a_divergent_run_restores_toward_the_teachers_exit_state(
    observed_wec, monkeypatch
):
    ctx, sink, db_path = observed_wec

    def teacher_mutation(chat_session):
        chat_session.get_active_workflow().context["teacher_wrote"] = "this"

    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(
                cs, ["list_tasks", "complete_task"], cache_hit=False,
                mutate=teacher_mutation,
            ),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )

    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["exec_diverged"] == 1
    # Both sites executed on this path, and each landed on its own baseline.
    assert run["restore_ok_pre_student"] == 1
    assert run["restore_ok_post_compare"] == 1
    # The teacher's exit state is what the world was left in.
    assert ctx.get_active_workflow().context["teacher_wrote"] == "this"
    # ...and the passes still entered identically, so the divergence counts.
    assert (run["comparable"], run["comparable_reason"]) == (1, None)


# ---------------------------------------------------------------------------
# entry_inputs_json — prompt inputs, never restorable state [DR45]
# ---------------------------------------------------------------------------

def test_entry_inputs_carry_prompt_inputs_and_a_labelled_diagnostic_snapshot(wec):
    ds = DistillationSession(wec)
    wec.get_active_workflow().context["visible"] = "in the diagnostic snapshot"
    payload = json.loads(ds._entry_inputs_json("list my tasks", 0))

    assert payload["raw_user_message"] == "list my tasks"
    assert payload["history_bound"] == 0
    # The context dicts are here to EXPLAIN a divergence to a reader, never to
    # be loaded back into a Workflow — hence the explicit label.
    assert payload["context_snapshot"]["diagnostic_only"] is True
    assert payload["context_snapshot"]["workflow_context"]["visible"] == (
        "in the diagnostic snapshot"
    )
    # The corpora go in by size and hash, never by body (§8).
    assert set(payload["insight_set"]) == {"planning", "execution"}


def test_the_refined_query_and_plan_are_folded_in_at_pass_exit():
    """Both are produced INSIDE the pass, so they cannot be captured at the
    entry boundary — but [DR45] names them as that pass's prompt inputs."""
    entry = json.dumps({"v": 1, "refined_user_message": None, "plan": None})
    step = PlanningStep(0, "list the tasks that are open", ["list_tasks"])
    folded = json.loads(
        DistillationSession._with_pass_prompt_inputs(entry, [step])
    )
    assert folded["refined_user_message"] == "list the tasks that are open"
    assert folded["plan"] == ["list_tasks"]
    # No planning step captured (the planner produced no next steps): the entry
    # inputs are returned untouched rather than half-filled.
    assert DistillationSession._with_pass_prompt_inputs(entry, []) == entry


def test_the_run_row_records_the_insight_corpora_as_loaded_at_agent_init(
    observed_wec, monkeypatch
):
    """[DR34]: `_planning_insights` / `_execution_insights` are loaded once and
    never reloaded, so the corpus a run used is not the file's current
    contents. Stored by size and hash."""
    ctx, sink, db_path = observed_wec
    ctx._planning_insights = "1. always list before completing\n"
    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )
    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    insight_set = json.loads(run["insight_set_json"])
    assert insight_set["planning"]["bytes"] == len(
        "1. always list before completing\n".encode()
    )
    assert len(insight_set["planning"]["sha256"]) == 64
    assert insight_set["execution"] is None


# ---------------------------------------------------------------------------
# Structured divergence records — fix-sb8.4 (design §7)
#
# [DR17] moved the comparison off `chat_session.action_log` and onto the
# `fw.command.execute` / `fw.ask_user` SPANS, which is what gives every stored
# record a span id that resolves and what stops `is_valid_action` from
# swallowing every failed command. [DR49] fixes when those spans are read —
# in process, behind a `sink.flush()` barrier — because a merely LATE span
# would otherwise become a fabricated `missing-in-student` divergence that
# §10.3 then pins forever. §7.6 makes the prose the extractor receives a
# RENDERING of the stored records rather than a second, disagreeing summary.
#
# Same harness as above: real WEC over `tests/todo_list_workflow`, real
# `SQLiteTraceSink`, real `distill_message` / `tracing.start_span` / writer
# thread; only the LLM boundaries are scripted.
# ---------------------------------------------------------------------------

from fastworkflow import distillation_alignment as alignment


def _record_from_row(row: dict) -> alignment.DivergenceRecord:
    """Rebuild a `DivergenceRecord` from the row `distillation_divergences` holds.

    §7.6's claim is that the prose is renderable from the STORED record; the
    only way to assert that is to render from the row and nothing else.
    """
    return alignment.DivergenceRecord(
        divergence_id=row["divergence_id"],
        run_id=row["run_id"],
        level=row["level"],
        left_pass=row["left_pass"],
        right_pass=row["right_pass"],
        align_index=row["align_index"],
        kind=row["kind"],
        material=row["material"],
        replayable=row["replayable"],
        command_key=row["command_key"],
        command_name=row["command_name"],
        context=row["context"],
        left_step_key=row["left_step_key"],
        right_step_key=row["right_step_key"],
        left_span_id=row["left_span_id"],
        right_span_id=row["right_span_id"],
        param_diff=(
            json.loads(row["param_diff_json"]) if row["param_diff_json"] else None
        ),
        detail=json.loads(row["detail_json"]),
    )


def _divergences(db_path: str) -> list[dict]:
    return _rows(
        db_path,
        "SELECT * FROM distillation_divergences ORDER BY level, align_index",
    )


def _teacher_wrote(chat_session):
    """A teacher-only state change, so the two passes EXIT differently."""
    chat_session.get_active_workflow().context["teacher_wrote"] = "this"


def test_a_two_pass_run_persists_records_whose_span_ids_resolve_to_real_spans(
    observed_wec, monkeypatch
):
    """[DR17]'s payoff: provenance for free. Every stored span id is the
    PRIMARY KEY of a real `fw.command.execute` row, in the right pass."""
    ctx, sink, db_path = observed_wec
    _turn_key, result = _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(
                cs, ["list_tasks", "complete_task"], cache_hit=False,
                mutate=_teacher_wrote,
            ),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )

    records = _divergences(db_path)
    assert [r["kind"] for r in records] == ["identical", "missing-in-student"]
    assert {r["run_id"] for r in records} == {result.run_id}
    assert {r["level"] for r in records} == {"action"}
    assert [r["left_pass"] for r in records] == ["teacher", "teacher"]
    assert [r["right_pass"] for r in records] == ["student", "student"]
    assert [r["align_index"] for r in records] == [0, 1]
    assert [r["command_name"] for r in records] == ["list_tasks", "complete_task"]

    spans = {
        row["span_id"]: row
        for row in _rows(db_path, "SELECT * FROM spans")
    }
    for record in records:
        for side, expected_pass in (("left", "teacher"), ("right", "student")):
            span_id = record[f"{side}_span_id"]
            if span_id is None:
                continue
            assert span_id in spans, f"{side}_span_id does not resolve"
            assert spans[span_id]["name"] == tracing.SPAN_COMMAND_EXECUTE
            assert spans[span_id]["distillation_pass"] == expected_pass

    # The teacher-only step is unmatched, so it has no student span at all.
    assert records[1]["right_span_id"] is None
    assert records[1]["left_span_id"] in spans

    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["exec_diverged"] == 1
    # The aligner's OWN step counts [DR49], so a reader can detect a stored
    # sequence truncated by retention.
    assert (run["left_steps"], run["right_steps"]) == (2, 1)
    # The passes exited in different states, so the unmatched step is material
    # and the matched one is not (`identical` is never material).
    assert run["material_divergences"] == 1
    assert [r["material"] for r in records] == [0, 1]


def test_identical_action_sequences_store_all_identical_records_and_no_material(
    observed_wec, monkeypatch
):
    """`identical` records ARE stored (§7.3) — they are what makes the aligned
    diff renderable without recomputation and the denominator of every rate in
    fix-sb8.10 — and the same end state means nothing is material ([DR20])."""
    ctx, sink, db_path = observed_wec
    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(
                cs, ["list_tasks", "complete_task"], cache_hit=False
            ),
            lambda cs: _PassAgent(
                cs, ["list_tasks", "complete_task"], cache_hit=False
            ),
        ],
    )

    records = _divergences(db_path)
    assert len(records) == 2
    assert {r["kind"] for r in records} == {"identical"}
    assert all(r["material"] == 0 for r in records)
    # Nothing to highlight on an identical pair, so the column is NULL rather
    # than an empty diff object.
    assert all(r["param_diff_json"] is None for r in records)
    # Both steps matched, so both step keys are present and equal.
    assert all(r["left_step_key"] == r["right_step_key"] for r in records)

    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["exec_diverged"] == 0
    assert run["material_divergences"] == 0
    assert (run["left_steps"], run["right_steps"]) == (2, 2)


def test_the_prose_the_extractor_receives_is_rendered_from_the_stored_records(
    observed_wec, monkeypatch
):
    """§7.6: one source of truth for the extractor, the UI and the aggregates.

    The summary handed to `extract_insights` must be reproducible from the
    rows alone — nothing computed alongside them, and nothing the UI cannot
    read back.
    """
    ctx, sink, db_path = observed_wec
    captured: list[str] = []

    queued = [
        lambda cs: _PassAgent(
            cs, ["list_tasks", "complete_task"], cache_hit=False,
            mutate=_teacher_wrote,
        ),
        lambda cs: _PassAgent(cs, ["get_task", "list_tasks"], cache_hit=False),
    ]
    _script_llm_boundaries(
        monkeypatch, ctx, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )
    monkeypatch.setattr(
        DistillationSession,
        "extract_insights",
        lambda self, t, s, divergence_summary, q: captured.append(
            divergence_summary
        ) or [],
    )

    ctx._begin_turn("list my tasks")
    distill_message(ctx, "list my tasks")
    sink.close()

    assert len(captured) == 1
    stored = [_record_from_row(row) for row in _divergences(db_path)]
    assert stored, "a diverged run must have stored its records"
    # Rendered from the rows, byte for byte what the extractor was given.
    assert alignment.render_divergence_summary(stored) == captured[0]
    # ...and it is a real summary, not an empty string that trivially agrees.
    assert "list_tasks" in captured[0] or "get_task" in captured[0]


def test_a_non_comparable_run_stores_material_null_on_every_record(
    observed_wec, monkeypatch
):
    """[DR49] + [DR20]: a pass whose spans_dropped counter moved makes the run
    `evidence-incomplete`, and materiality is unknowable on a run whose passes
    may not have started from the same place — NULL, never a 0 that reads as
    'checked and found harmless'."""
    ctx, sink, db_path = observed_wec

    def drop_a_span(chat_session):
        # Exactly what `SQLiteTraceSink.emit_span` does on `queue.Full`.
        tracing.get_sink(chat_session)._count("spans_dropped")

    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(
                cs, ["list_tasks", "complete_task"], cache_hit=False,
                mutate=drop_a_span,
            ),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )

    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert (run["comparable"], run["comparable_reason"]) == (
        0,
        "evidence-incomplete",
    )
    records = _divergences(db_path)
    # The records are still WRITTEN — a non-comparable run is recorded rather
    # than deleted (§6.2 obligation 5) — but quarantined by a NULL material.
    assert records
    assert all(r["material"] is None for r in records)
    assert run["material_divergences"] == 0


def test_a_failed_command_is_a_step_instead_of_being_dropped(
    observed_wec, monkeypatch
):
    """§7.1's headline: `is_valid_action` drops every record with a falsy
    `command_name`, which is every FAILED command. Aligning over spans
    inherits none of that, and [DR50] keys the step off `raw_command`."""
    ctx, sink, db_path = observed_wec

    class _FailingCommandAgent(_PassAgent):
        """A pass whose command fails the way `command_executor` fails one.

        `_invoke_command_impl` returns early on `not success` BEFORE assigning
        `command_name`/`context`, so the close writes NULL into both and only
        the `raw_command` recorded at open survives.
        """

        def __call__(self, **kwargs):
            span = tracing.start_span(
                self._chat_session,
                tracing.SPAN_COMMAND_EXECUTE,
                kind=tracing.KIND_TOOL,
                attributes={"raw_command": "complete task 99"},
            )
            tracing.end_span(
                self._chat_session,
                span,
                status=tracing.STATUS_ERROR,
                attributes={"response_text": "no such task", "success": False},
            )
            return super().__call__(**kwargs)

    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _FailingCommandAgent(cs, ["list_tasks"], cache_hit=False),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )

    records = _divergences(db_path)
    kinds = [r["kind"] for r in records]
    assert "missing-in-student" in kinds
    failed = next(r for r in records if r["kind"] == "missing-in-student")
    # Keyed off raw_command, so it is legible in the UI and in the
    # idx_distill_div_kind rollups rather than collapsing onto every other
    # failure in the run.
    assert failed["command_name"] == "raw:complete task 99"
    assert failed["left_span_id"]
    assert _rows(
        db_path, "SELECT * FROM spans WHERE span_id=?", (failed["left_span_id"],)
    )[0]["command_name"] is None
    assert _rows(db_path, "SELECT * FROM distillation_runs")[0]["exec_diverged"] == 1


def test_the_compare_span_reports_the_alignment_it_stored(
    observed_wec, monkeypatch
):
    """§8: one `fw.distill.compare` span per level, run-level (pass NULL), and
    its counts are the ones the rows carry — not a second tally."""
    ctx, sink, db_path = observed_wec
    turn_key, _result = _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(
                cs, ["list_tasks", "complete_task"], cache_hit=False
            ),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )

    compares = _rows(
        db_path,
        "SELECT * FROM spans WHERE trace_id=? AND name=? ORDER BY start_ns",
        (turn_key, tracing.SPAN_DISTILL_COMPARE),
    )
    assert [json.loads(s["attributes"])["level"] for s in compares] == [
        "plan",
        "action",
    ]
    assert all(s["distillation_pass"] is None for s in compares)

    action = json.loads(compares[1]["attributes"])
    assert (action["left_steps"], action["right_steps"]) == (2, 1)
    assert action["algorithm"] == alignment.ALGORITHM
    stored = _divergences(db_path)
    counts: dict = {}
    for row in stored:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    assert action["divergence_counts"] == counts


def test_a_run_without_a_sink_still_detects_divergence(wec, monkeypatch):
    """§7.6 replaces the prose summary's SOURCE; it does not make distillation
    require observability. With no sink there is nothing to align and nowhere
    to store it, so the legacy action-log comparison still runs."""
    summaries: list[str] = []
    queued = [
        lambda cs: _PassAgent(cs, ["list_tasks", "complete_task"], cache_hit=False),
        lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
    ]
    _script_llm_boundaries(
        monkeypatch, wec, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )
    monkeypatch.setattr(
        DistillationSession,
        "extract_insights",
        lambda self, t, s, divergence_summary, q: summaries.append(
            divergence_summary
        ) or [],
    )

    wec._begin_turn("list my tasks")
    result = distill_message(wec, "list my tasks")

    assert result.command_output.command_response.response == "done"
    assert len(summaries) == 1
    assert "complete_task" in summaries[0]


def test_a_read_barrier_that_cannot_be_taken_makes_the_run_evidence_incomplete(
    observed_wec, monkeypatch
):
    """[DR49]'s barrier is a gate, not a courtesy.

    If `flush()` does not come back, the spans a divergence record would cite
    may not be in the table — so the run is `evidence-incomplete`, exactly as
    it is when the writer dropped a span. Silence is never agreement, and here
    it would not even be silence: an unwritten span reads downstream as a
    `missing-in-student` divergence.
    """
    ctx, sink, _db_path = observed_wec
    ds = DistillationSession(ctx)
    ds._passes = {
        "teacher": {"seq": 0, "entry_fingerprint": "abc"},
        "student": {"seq": 1, "entry_fingerprint": "abc"},
    }
    # Equal entry fingerprints and no dropped spans: comparable, until the
    # barrier says otherwise.
    assert ds.comparability_fields()["comparable"] == 1

    monkeypatch.setattr(
        "fastworkflow.distillation._ALIGN_FLUSH_TIMEOUT_S", 0.05
    )
    # The real sink, with nothing draining its queue — the condition a wedged
    # or saturated writer thread puts it in.
    sink._stop.set()
    sink._writer.join(2.0)

    assert ds.read_barrier() is False
    verdict = ds.comparability_fields()
    assert (verdict["comparable"], verdict["comparable_reason"]) == (
        0,
        "evidence-incomplete",
    )


def test_the_span_collector_leaves_the_sink_exactly_as_it_found_it(
    observed_wec, monkeypatch
):
    """[DR49]'s capture shadows `emit_span` on the live sink for the length of
    the run. A shadow that outlived the run would keep a dead run's collector
    in front of every later turn's spans, so the restore is asserted on the
    clean path and on the raising one."""
    ctx, sink, _db_path = observed_wec
    sink_class, emit = type(sink), type(sink).emit_span

    _run_distillation(
        ctx,
        sink,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )

    assert "emit_span" not in vars(sink)
    assert type(sink) is sink_class and type(sink).emit_span is emit
    assert ctx.trace_sink is sink


def test_the_span_collector_is_removed_when_the_teacher_raises(
    observed_wec, monkeypatch
):
    """The half-run path (§18, fix-sb8.2) exits through the same finally."""
    ctx, sink, _db_path = observed_wec
    queued = [lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False)]
    _script_llm_boundaries(
        monkeypatch, ctx, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )

    def raising_get_lm(model_env_var, *args, **kwargs):
        raise ValueError(f"DSPy Language Model not provided. Set {model_env_var}.")

    monkeypatch.setattr("fastworkflow.utils.dspy_utils.get_lm", raising_get_lm)

    ctx._begin_turn("list my tasks")
    with pytest.raises(ValueError):
        distill_message(ctx, "list my tasks")

    assert "emit_span" not in vars(sink)
    assert ctx.trace_sink is sink


# ---------------------------------------------------------------------------
# The insight ledger — fix-sb8.5 (design §8, §13, §15)
#
# The extractor call — the step that actually decides what the rule is — was
# the one step of a distillation run with no span at all, and an emitted
# insight was a numbered line in a markdown file with no back-reference: given
# rule #7 there was no way to find the turn it came from. These cover the
# `fw.distill.extract` span and its exact §8 attribute list, `[DR31]`'s stable
# ids and their markdown marker, `[DR32]`'s provenance chain in BOTH
# directions, `[DR33]`'s three negative outcomes as rows, and the §15 recipes
# executed — per `[DR54]`, "it parses" is retired as a standard of evidence —
# against a hand-checkable fixture.
#
# Real WEC over `tests/todo_list_workflow`, real `SQLiteTraceSink`, real
# `distill_message` / `extract_insights` / `append_insights` / writer thread.
# Only the extractor's LLM boundary is scripted.
# ---------------------------------------------------------------------------

import hashlib

import dspy as dspy_module

from fastworkflow.distillation import (
    EMPTY_REASON_EXTRACTOR,
    EMPTY_REASON_PARSE,
    INSIGHT_KIND_EXECUTION,
    INSIGHT_KIND_PLANNING,
    insight_id,
    insight_text_hash,
    normalize_insight_text,
)
from fastworkflow.utils.insights_loader import (
    format_insight_marker,
    load_workflow_insights,
    marked_insight_ids,
    strip_insight_markers,
)

# The §15 / §13.2 recipes, verbatim. `test_the_documented_recipes_are_the_ones
# _the_design_ships` pins each one against the design document, so what runs
# below is the shipped SQL rather than a paraphrase of it.
_SQL_FORWARD = """\
SELECT i.insight_id, i.text, i.kind, i.extractor_span_id,
       d.divergence_id, d.kind AS divergence_kind, d.material,
       d.command_name, d.left_span_id, d.right_span_id,
       r.turn_key, r.comparable, d.left_pass, d.right_pass
FROM distillation_insights i
JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
JOIN distillation_runs r ON r.run_id = i.run_id
WHERE i.insight_id = :insight_id;
"""

_SQL_REVERSE = """\
SELECT DISTINCT i.insight_id, i.text, i.kind, r.turn_key
FROM distillation_divergences d
JOIN distillation_insight_citations c ON c.divergence_id = d.divergence_id
JOIN distillation_insights i ON i.insight_id = c.insight_id
JOIN distillation_runs r ON r.run_id = i.run_id
WHERE d.left_span_id = :span_id OR d.right_span_id = :span_id;
"""

_SQL_SUPPORT = """\
WITH cited AS (
  SELECT DISTINCT d.command_key, d.kind
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  WHERE i.insight_id = :insight_id
)
SELECT r.run_id, r.turn_key, r.started_at,
       d.divergence_id, d.kind, d.command_name, d.material
FROM distillation_runs r
JOIN distillation_divergences d ON d.run_id = r.run_id
JOIN cited ON cited.command_key = d.command_key AND cited.kind = d.kind
WHERE r.comparable = 1
  AND r.replay_of IS NULL      -- a replay tests an insight; it is not support for it
ORDER BY r.started_at DESC;
"""

_SQL_CONTRADICTION = """\
WITH cited AS (
  SELECT DISTINCT d.command_key, d.command_name, d.kind
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  WHERE i.insight_id = :insight_id
)
SELECT r.run_id, r.turn_key, r.started_at, cited.command_name
FROM distillation_runs r
CROSS JOIN cited            -- deliberate: every cited (command, kind) against every run
WHERE cited.command_name IS NOT NULL   -- run-level divergences have no command [DR54]
  AND r.comparable = 1
  AND r.replay_of IS NULL
  AND EXISTS (SELECT 1 FROM spans s
               WHERE s.trace_id = r.turn_key            -- [DR1]: the invariant holds
                 AND s.distillation_pass = 'student'
                 AND s.name = 'fw.command.execute'
                 AND s.command_name = cited.command_name)
  AND NOT EXISTS (SELECT 1 FROM distillation_divergences d2
                   WHERE d2.run_id = r.run_id
                     AND d2.command_key = cited.command_key
                     AND d2.kind = cited.kind)
ORDER BY r.started_at DESC;
"""

_SQL_PROMOTION = """\
SELECT i.insight_id, i.kind, i.text,
       (SELECT v.verdict FROM distillation_verdicts v
         WHERE v.insight_id = i.insight_id AND v.superseded = 0
         ORDER BY v.created_at DESC LIMIT 1)                     AS verdict,
       (SELECT COUNT(DISTINCT sup.run_id)
          FROM distillation_insight_citations c
          JOIN distillation_divergences cit ON cit.divergence_id = c.divergence_id
          JOIN distillation_divergences sup  ON sup.command_key = cit.command_key
                                            AND sup.kind        = cit.kind
          JOIN distillation_runs r           ON r.run_id = sup.run_id
         WHERE c.insight_id = i.insight_id
           AND cit.command_key IS NOT NULL   -- run-level citations key on nothing
           AND r.comparable = 1
           AND r.replay_of IS NULL           -- a replay is a test, not support [DR54]
           AND r.isolation_verified = 1      -- promotion is a causal claim [DR48]
           AND r.evidence_pruned = 0)                            AS support_runs,
       (SELECT COUNT(DISTINCT sup.run_id)
          FROM distillation_insight_citations c
          JOIN distillation_divergences cit ON cit.divergence_id = c.divergence_id
          JOIN distillation_divergences sup  ON sup.command_key = cit.command_key
                                            AND sup.kind        = cit.kind
          JOIN distillation_runs r           ON r.run_id = sup.run_id
         WHERE c.insight_id = i.insight_id
           AND cit.command_key IS NOT NULL AND sup.material = 1
           AND r.comparable = 1 AND r.replay_of IS NULL
           AND r.isolation_verified = 1 AND r.evidence_pruned = 0) AS material_support_runs
FROM distillation_insights i
ORDER BY support_runs DESC;
"""

_SQL_REVERSE_INDEX = """\
SELECT i.insight_id, r.run_id, r.turn_key, r.started_at
FROM distillation_insights i JOIN distillation_runs r ON r.run_id = i.run_id
WHERE i.text_hash = :text_hash ORDER BY r.started_at;
"""

_DESIGN_DOC = (
    Path(__file__).parent.parent / "docs" / "distillation_observability_design.md"
)


class _PlanningPassAgent(_PassAgent):
    """A pass agent that also emits its planner span, so the PLAN level aligns.

    `plan_steps` reads `fw.planner.plan` spans and their full plan string
    (§7.1); a double that emits none makes every pass planless, and the
    planning extractor never fires.
    """

    def __init__(self, chat_session, command_names, *, cache_hit, plan):
        super().__init__(chat_session, command_names, cache_hit=cache_hit)
        self._plan = plan

    def __call__(self, **kwargs):
        span = tracing.start_span(
            self._chat_session,
            tracing.SPAN_PLANNER_PLAN,
            attributes={"plan": self._plan},
        )
        tracing.end_span(self._chat_session, span)
        return super().__call__(**kwargs)


def _script_extractors(
    monkeypatch, *, planning_raw="EMPTY", execution_raw="EMPTY", calls=None
):
    """Script the two extractor LLM calls at the dspy boundary.

    The extractors themselves stay real — the span, the parse, the ids, the
    file append and the ledger writes are all the shipped code path — so only
    the model's answer is supplied.
    """
    from fastworkflow.distillation import (
        InsightExtractionSignature,
        PlanningInsightExtractionSignature,
    )

    class _ScriptedChainOfThought:
        def __init__(self, signature):
            self._signature = signature

        def __call__(self, **kwargs):
            raw = (
                planning_raw
                if self._signature is PlanningInsightExtractionSignature
                else execution_raw
            )
            if calls is not None:
                calls.append((self._signature.__name__, kwargs))
            return type("Prediction", (), {"insights": raw})()

    assert InsightExtractionSignature is not PlanningInsightExtractionSignature
    monkeypatch.setattr(dspy_module, "ChainOfThought", _ScriptedChainOfThought)


def _run_distillation_with_extraction(
    ctx,
    sink,
    monkeypatch,
    tmp_path,
    agents,
    *,
    planning_raw="EMPTY",
    execution_raw="EMPTY",
    message="list my tasks",
    calls=None,
):
    """One real two-pass run with the REAL extractors and the REAL ledger writes."""
    from fastworkflow.distillation import DistillationSession

    real_execution = DistillationSession.extract_insights
    real_planning = DistillationSession.extract_planning_insights
    queued = list(agents)
    _script_llm_boundaries(
        monkeypatch, ctx, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )
    # `_script_llm_boundaries` stubs both extractors out; this child is about
    # what they do, so put the real ones back and script the LLM boundary
    # underneath them instead.
    monkeypatch.setattr(DistillationSession, "extract_insights", real_execution)
    monkeypatch.setattr(
        DistillationSession, "extract_planning_insights", real_planning
    )
    _script_extractors(
        monkeypatch,
        planning_raw=planning_raw,
        execution_raw=execution_raw,
        calls=calls,
    )
    # The real append path runs, against a directory the TEST owns: the
    # workflow under `tests/` is checked in and an insights file written into
    # it would outlive the run.
    insights_dir = tmp_path / "Insights" / "todo_list_workflow"
    insights_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        DistillationSession, "_insights_dir", lambda self: insights_dir
    )
    ctx._begin_turn(message)
    turn_key = ctx.current_turn_key
    result = distill_message(ctx, message)
    sink.close()
    return turn_key, result, insights_dir


def _diverging_agents():
    """Teacher runs one command the student skips — one action divergence."""
    return [
        lambda cs: _PassAgent(
            cs, ["list_tasks", "complete_task"], cache_hit=False,
            mutate=_teacher_wrote,
        ),
        lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
    ]


def _extract_spans(db_path: str) -> list[dict]:
    return _rows(
        db_path,
        "SELECT * FROM spans WHERE name = ? ORDER BY start_ns",
        (tracing.SPAN_DISTILL_EXTRACT,),
    )


# ---------------------------------------------------------------------------
# (a) The extractor span — §8's exact attribute list
# ---------------------------------------------------------------------------

def test_the_extractor_call_gets_its_own_span_carrying_sections_8_attributes(
    observed_wec, monkeypatch, tmp_path
):
    """§8: `fw.distill.extract`, under the run span, labelled 'extractor'.

    The model invocation that decides what the rule is now has a span to hang
    from — which is the whole of what "the extractor call must be observable"
    reduces to, because a `fw.llm.call` parents onto whatever is on the stack.
    """
    ctx, sink, db_path = observed_wec
    text = "Never complete a task before listing tasks"
    turn_key, result, _dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path, _diverging_agents(),
        execution_raw=f"- {text}",
    )

    spans = _extract_spans(db_path)
    assert len(spans) == 1, "one extractor call, one span"
    span = spans[0]
    assert span["trace_id"] == turn_key
    assert span["distillation_pass"] == "extractor"
    assert span["status"] == tracing.STATUS_OK

    run_span = _rows(
        db_path, "SELECT * FROM spans WHERE name = ?", (tracing.SPAN_DISTILL_RUN,)
    )[0]
    # Under the run, not the turn: an fw.llm.call emitted inside the extractor
    # therefore lands under the extract span rather than loose in the turn.
    assert span["parent_span_id"] == run_span["span_id"]

    attributes = json.loads(span["attributes"])
    assert attributes["run_id"] == result.run_id
    assert attributes["kind"] == INSIGHT_KIND_EXECUTION
    assert attributes["divergence_summary"]
    assert "complete_task" in attributes["divergence_summary"]
    assert attributes["raw_output"] == f"- {text}"
    assert attributes["parsed_count"] == 1
    assert attributes["empty_reason"] is None
    assert attributes["insight_ids"] == [
        insight_id(result.run_id, INSIGHT_KIND_EXECUTION, text)
    ]
    # `span_contract_version` is stamped on EVERY span by `_emit` (arch §12.0
    # delta 5), so it is the emitter's, not this site's. Excluding it keeps
    # this assertion about what §8 says the extract span carries; the
    # emitter-wide key is covered by test_span_contract_versioning.py.
    assert set(attributes) - {tracing.ATTR_SPAN_CONTRACT_VERSION} == {
        "run_id", "kind", "extractor_model", "divergence_summary",
        "existing_insights_bytes", "existing_insights_sha256", "raw_output",
        "parsed_count", "empty_reason", "insight_ids",
    }


def test_the_extract_span_stores_the_existing_corpus_by_length_and_hash(
    observed_wec, monkeypatch, tmp_path
):
    """§8: `existing_insights` is NOT stored — the corpus is pasted whole into
    the prompt and grows without bound, so the span records its identity."""
    ctx, sink, db_path = observed_wec
    workflow = ctx.get_active_workflow()
    _turn_key, _result, _dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path, _diverging_agents(),
        execution_raw="- Do not skip a step the teacher took",
    )

    corpus = (
        load_workflow_insights(workflow.folderpath, "execution_agent") or ""
    ).encode("utf-8")
    attributes = json.loads(_extract_spans(db_path)[0]["attributes"])
    assert attributes["existing_insights_bytes"] == len(corpus)
    assert attributes["existing_insights_sha256"] == hashlib.sha256(corpus).hexdigest()
    # The body itself never rides along, under any key.
    assert "existing_insights" not in attributes


# ---------------------------------------------------------------------------
# (b) Stable insight ids — §13.1 [DR31]
# ---------------------------------------------------------------------------

def test_the_insight_id_formula_is_the_documented_one():
    """§13.1, exactly: normalize, hash the text for the reverse index, and fold
    `run_id` and `kind` into the id itself."""
    text = "  Never call UPDATE_TASK   before\nverifying the task exists.  "
    normalized = "never call update_task before verifying the task exists"
    assert normalize_insight_text(text) == normalized
    assert insight_text_hash(text) == hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]
    assert insight_id("run-abc", INSIGHT_KIND_EXECUTION, text) == "ins-" + hashlib.sha256(
        f"run-abc|execution|{normalized}".encode("utf-8")
    ).hexdigest()[:12]
    assert len(insight_id("run-abc", INSIGHT_KIND_EXECUTION, text)) == len("ins-") + 12


def test_run_id_is_inside_the_id_and_the_file_entry_number_is_not():
    """§13.1's two halves. Two runs saying the same thing are two pieces of
    evidence, not one row — and an id built on the entry number would be
    orphaned by the next renumber, which is why only `text_hash` is shared."""
    text = "Verify the task exists before updating it"
    first = insight_id("run-111111111111", INSIGHT_KIND_EXECUTION, text)
    second = insight_id("run-222222222222", INSIGHT_KIND_EXECUTION, text)
    assert first != second
    assert insight_text_hash(text) == insight_text_hash(text.upper() + " ")
    # Kind is in the id too: the same sentence as a planning rule is a
    # different row from the same sentence as an anti-pattern.
    assert first != insight_id("run-111111111111", INSIGHT_KIND_PLANNING, text)
    # Nothing about the file reaches the id: it is a pure function of run,
    # kind and normalized text.
    assert first == insight_id("run-111111111111", INSIGHT_KIND_EXECUTION, f"  {text}. ")


# ---------------------------------------------------------------------------
# (c) The markdown marker — §13.1, [DR56], §21 objection 5
# ---------------------------------------------------------------------------

def test_the_markdown_marker_round_trips_without_polluting_prompt_content(tmp_path):
    """The marker resolves a file line back to its ledger row, and NO prompt
    consumer ever sees it — there are three of them, not one `[DR56]`."""
    workflow_dir = tmp_path / "todo_list_workflow"
    insights_dir = workflow_dir / "Insights" / "todo_list_workflow"
    insights_dir.mkdir(parents=True)
    marked_file = insights_dir / "execution_agent_anti_patterns.md"
    header = "# Execution Agent Anti-Patterns\n\n"
    pattern = r"^(\d+)\.\s"
    fmt = "{num}. {insight}\n"
    texts = [
        "Never call update_task before verifying the task exists",
        "Do not guess a task id",
    ]
    ids = ["ins-9f3c1a7b2e04", "ins-0123456789ab"]

    numbers = DistillationSession._append_numbered_insights(
        texts, marked_file, header, pattern, fmt, insight_ids=ids
    )
    assert numbers == [1, 2]
    on_disk = marked_file.read_text(encoding="utf-8")
    assert f"1. {texts[0]}  <!-- {ids[0]} -->" in on_disk

    # The marker does not disturb the numbering regex: a third entry continues
    # from 2, which is the property an id built on the number could not have.
    assert DistillationSession._append_numbered_insights(
        ["Read before you write"], marked_file, header, pattern, fmt,
        insight_ids=["ins-ffffffffffff"],
    ) == [3]

    # File -> ledger, in file order, and it survives a hand edit of the text.
    full = marked_file.read_text(encoding="utf-8")
    assert marked_insight_ids(full) == ids + ["ins-ffffffffffff"]
    edited = full.replace("Do not guess a task id", "Do not ever guess a task id")
    assert marked_insight_ids(edited) == ids + ["ins-ffffffffffff"]

    # What every prompt consumer sees is byte-for-byte the unmarked file.
    plain_dir = tmp_path / "plain" / "Insights" / "plain"
    plain_dir.mkdir(parents=True)
    plain_file = plain_dir / "execution_agent_anti_patterns.md"
    DistillationSession._append_numbered_insights(
        texts + ["Read before you write"], plain_file, header, pattern, fmt
    )
    loaded = load_workflow_insights(str(workflow_dir), "execution_agent")
    assert loaded == plain_file.read_text(encoding="utf-8")
    assert "<!--" not in loaded and "ins-" not in loaded
    assert loaded == strip_insight_markers(full)


def test_a_planning_insight_line_also_round_trips(tmp_path):
    """The planning corpus numbers with `^## (\\d+)\\.`; the marker must clear
    that regex too, and be stripped out of the planner's prompt."""
    workflow_dir = tmp_path / "todo_list_workflow"
    insights_dir = workflow_dir / "Insights" / "todo_list_workflow"
    insights_dir.mkdir(parents=True)
    planning_file = insights_dir / "planning_agent_insights.md"
    numbers = DistillationSession._append_numbered_insights(
        ["List before completing"], planning_file,
        "# Planning Agent Insights\n\n", r"^## (\d+)\.", "## {num}. {insight}\n\n",
        insight_ids=["ins-abcdefabcdef"],
    )
    assert numbers == [1]
    assert DistillationSession._append_numbered_insights(
        ["Confirm the id"], planning_file,
        "# Planning Agent Insights\n\n", r"^## (\d+)\.", "## {num}. {insight}\n\n",
        insight_ids=["ins-bbbbbbbbbbbb"],
    ) == [2]
    raw = planning_file.read_text(encoding="utf-8")
    assert marked_insight_ids(raw) == ["ins-abcdefabcdef", "ins-bbbbbbbbbbbb"]
    loaded = load_workflow_insights(str(workflow_dir), "planning_agent")
    assert "ins-" not in loaded
    assert "## 2. Confirm the id\n" in loaded


def test_format_insight_marker_is_what_strip_removes():
    """Producer and consumer share one definition of the marker."""
    line = f"7. Never guess{format_insight_marker('ins-9f3c1a7b2e04')}"
    assert line == "7. Never guess  <!-- ins-9f3c1a7b2e04 -->"
    assert strip_insight_markers(line) == "7. Never guess"
    assert marked_insight_ids(line) == ["ins-9f3c1a7b2e04"]


def test_an_emitted_insight_is_written_to_the_file_with_its_marker(
    observed_wec, monkeypatch, tmp_path
):
    """The end of the chain: the id in the ledger row is the id on the line."""
    ctx, sink, db_path = observed_wec
    text = "Never complete a task before listing tasks"
    _turn_key, result, insights_dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path, _diverging_agents(),
        execution_raw=f"- {text}",
    )
    written = (insights_dir / "execution_agent_anti_patterns.md").read_text(
        encoding="utf-8"
    )
    stored = _rows(db_path, "SELECT * FROM distillation_insights")
    assert len(stored) == 1
    assert stored[0]["text"] == text
    assert stored[0]["run_id"] == result.run_id
    assert stored[0]["kind"] == INSIGHT_KIND_EXECUTION
    assert stored[0]["text_hash"] == insight_text_hash(text)
    # DISPLAY ONLY (§13.1) — stored, but never the identifier.
    assert stored[0]["file_entry_number"] == 1
    assert stored[0]["insight_file"] == str(
        insights_dir / "execution_agent_anti_patterns.md"
    )
    assert marked_insight_ids(written) == [stored[0]["insight_id"]]
    assert stored[0]["extractor_span_id"] == _extract_spans(db_path)[0]["span_id"]


# ---------------------------------------------------------------------------
# (d) The provenance chain, both directions — §13.2 [DR32]
# ---------------------------------------------------------------------------

def test_an_insight_resolves_to_its_divergence_and_span_pair_in_one_query(
    observed_wec, monkeypatch, tmp_path
):
    """Acceptance criterion 2, run as §13.2's own SQL: insight -> divergence ->
    the teacher/student span pair, in ONE query."""
    ctx, sink, db_path = observed_wec
    text = "Never complete a task before listing tasks"
    turn_key, result, _dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path, _diverging_agents(),
        execution_raw=f"- {text}",
    )
    stored = _rows(db_path, "SELECT * FROM distillation_insights")[0]

    rows = _rows(db_path, _SQL_FORWARD, {"insight_id": stored["insight_id"]})
    # The extractor's summary described exactly the non-identical records, so
    # the citation set is those and not the whole alignment.
    assert [r["divergence_kind"] for r in rows] == ["missing-in-student"]
    row = rows[0]
    assert row["text"] == text
    assert row["turn_key"] == turn_key
    assert row["command_name"] == "complete_task"
    assert (row["left_pass"], row["right_pass"]) == ("teacher", "student")
    assert row["material"] == 1

    spans = {s["span_id"]: s for s in _rows(db_path, "SELECT * FROM spans")}
    assert row["left_span_id"] in spans
    assert spans[row["left_span_id"]]["name"] == tracing.SPAN_COMMAND_EXECUTE
    assert spans[row["left_span_id"]]["distillation_pass"] == "teacher"
    # The teacher-only step has no student span, which is what the divergence
    # says; the pair is still resolvable in the one query.
    assert row["right_span_id"] is None
    assert row["extractor_span_id"] == _extract_spans(db_path)[0]["span_id"]
    assert row["comparable"] == 1
    assert row["insight_id"].startswith("ins-")
    assert row["divergence_id"].startswith("div-")
    del result


def test_a_span_resolves_back_to_the_insights_drawn_from_it(
    observed_wec, monkeypatch, tmp_path
):
    """The other direction, from the span a developer is looking at (§13.2).
    Acceptance criterion 7 needs both, and both are one query."""
    ctx, sink, db_path = observed_wec
    text = "Never complete a task before listing tasks"
    turn_key, _result, _dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path, _diverging_agents(),
        execution_raw=f"- {text}\n- Confirm the task list first",
    )
    divergence = _rows(
        db_path,
        "SELECT * FROM distillation_divergences WHERE kind = 'missing-in-student'",
    )[0]

    rows = _rows(db_path, _SQL_REVERSE, {"span_id": divergence["left_span_id"]})
    assert sorted(r["text"] for r in rows) == [
        "Confirm the task list first", text,
    ]
    assert {r["turn_key"] for r in rows} == {turn_key}
    assert {r["kind"] for r in rows} == {INSIGHT_KIND_EXECUTION}

    # A span nothing cites resolves to nothing, rather than to everything.
    identical = _rows(
        db_path,
        "SELECT * FROM distillation_divergences WHERE kind = 'identical'",
    )[0]
    assert _rows(db_path, _SQL_REVERSE, {"span_id": identical["left_span_id"]}) == []


def test_every_emitted_insight_cites_the_divergences_its_summary_described(
    observed_wec, monkeypatch, tmp_path
):
    """v1 is ONE run per insight (§18): the citations are this run's, because
    this run's summary is all the extractor was given."""
    ctx, sink, db_path = observed_wec
    _turn_key, result, _dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path, _diverging_agents(),
        execution_raw="- One\n- Two",
    )
    citations = _rows(
        db_path,
        "SELECT c.* FROM distillation_insight_citations c "
        "JOIN distillation_insights i ON i.insight_id = c.insight_id "
        "ORDER BY i.file_entry_number",
    )
    diverged = _rows(
        db_path,
        "SELECT divergence_id FROM distillation_divergences "
        "WHERE run_id = ? AND kind != 'identical'",
        (result.run_id,),
    )
    assert len(citations) == 2
    assert {c["divergence_id"] for c in citations} == {
        d["divergence_id"] for d in diverged
    }
    # `identical` records are stored but are never cited: they are the
    # denominator of the rates, not evidence a rule was drawn from.
    assert _rows(
        db_path,
        "SELECT 1 FROM distillation_insight_citations c "
        "JOIN distillation_divergences d ON d.divergence_id = c.divergence_id "
        "WHERE d.kind = 'identical'",
    ) == []


def test_the_text_hash_reverse_index_finds_every_run_that_said_it(tmp_path):
    """§15's last recipe: from a markdown line — where only the text survives —
    back to the runs that produced it. Two runs, same sentence, two rows, one
    hash; cross-run consolidation is this view, not a stored relation (§18)."""
    from fastworkflow.observability_store import SQLiteTraceSink

    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    try:
        text = "Never complete a task before listing tasks"
        for index, run_id in enumerate(("run-aaaaaaaaaaaa", "run-bbbbbbbbbbbb")):
            _seed_run(sink, run_id, turn_key=f"turn-{index}",
                      started_at=f"2026-08-0{index + 1}T00:00:00+00:00")
            sink.emit_distillation_record("insight", {
                "insight_id": insight_id(run_id, INSIGHT_KIND_EXECUTION, text),
                "run_id": run_id,
                "kind": INSIGHT_KIND_EXECUTION,
                "text": text,
                "text_hash": insight_text_hash(text),
                "created_at": "2026-08-01T00:00:00+00:00",
            })
    finally:
        sink.close()

    rows = _rows(db_path, _SQL_REVERSE_INDEX, {"text_hash": insight_text_hash(text)})
    assert [r["run_id"] for r in rows] == ["run-aaaaaaaaaaaa", "run-bbbbbbbbbbbb"]
    assert len({r["insight_id"] for r in rows}) == 2


# ---------------------------------------------------------------------------
# (e) Negative outcomes are rows, not absences — §13.3 [DR33]
# ---------------------------------------------------------------------------

def test_a_diverged_run_that_extracted_nothing_writes_extractor_empty(
    observed_wec, monkeypatch, tmp_path
):
    """§13.3 case 1. The divergence happened and the extractor declined it;
    both facts are now on the row instead of one silence."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result, insights_dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path, _diverging_agents(),
        execution_raw="EMPTY",
    )
    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["exec_diverged"] == 1
    assert run["extractor_empty"] == 1
    assert run["execution_insights"] == 0
    assert run["planning_insights"] == 0
    assert run["material_divergences"] == 1
    # The divergence records are still there — the run is evidence even though
    # no rule came out of it.
    assert len(_rows(db_path, "SELECT * FROM distillation_divergences")) == 2
    assert _rows(db_path, "SELECT * FROM distillation_insights") == []
    assert not (insights_dir / "execution_agent_anti_patterns.md").exists()

    attributes = json.loads(_extract_spans(db_path)[0]["attributes"])
    assert attributes["parsed_count"] == 0
    assert attributes["empty_reason"] == EMPTY_REASON_EXTRACTOR
    assert attributes["insight_ids"] == []


def test_a_run_with_no_divergence_at_all_still_writes_its_row(
    observed_wec, monkeypatch, tmp_path
):
    """§13.3 case 2: the "student was already correct" set. It is the
    contradiction pool for every candidate rule (§15), so its absence would
    silently empty half of the promotion decision."""
    ctx, sink, db_path = observed_wec
    _turn_key, result, _dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
            lambda cs: _PassAgent(cs, ["list_tasks"], cache_hit=False),
        ],
    )
    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert run["run_id"] == result.run_id
    assert (run["planning_diverged"], run["exec_diverged"]) == (0, 0)
    assert run["material_divergences"] == 0
    assert (run["planning_insights"], run["execution_insights"]) == (0, 0)
    # Nothing diverged, so no extractor ran: `extractor_empty` means the
    # extractor declined, not that there was nothing to decline.
    assert run["extractor_empty"] == 0
    assert run["completed_at"] is not None
    assert _extract_spans(db_path) == []
    assert [r["kind"] for r in _divergences(db_path)] == ["identical"]


def test_extractor_returned_empty_and_parse_yielded_nothing_are_distinguishable(
    observed_wec, monkeypatch, tmp_path
):
    """§13.3's distinction, which today's silence conflates.

    The planning parser keeps only lines starting with a digit followed by `.`
    within the first three characters, so a bullet-formatted answer yields `[]`
    — indistinguishable from a genuine EMPTY without `empty_reason`. One says
    the extractor is too conservative; the other says the parser is too strict.
    """
    ctx, sink, db_path = observed_wec
    _turn_key, _result, _dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path,
        [
            lambda cs: _PlanningPassAgent(
                cs, ["list_tasks", "complete_task"], cache_hit=False,
                plan="1. list the tasks 2. complete the first one",
            ),
            lambda cs: _PlanningPassAgent(
                cs, ["list_tasks"], cache_hit=False,
                plan="1. list the tasks and stop",
            ),
        ],
        planning_raw="- List the tasks before completing one",
        execution_raw="EMPTY",
    )
    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert (run["planning_diverged"], run["exec_diverged"]) == (1, 1)
    assert run["extractor_empty"] == 1
    assert (run["planning_insights"], run["execution_insights"]) == (0, 0)

    by_kind = {
        json.loads(span["attributes"])["kind"]: json.loads(span["attributes"])
        for span in _extract_spans(db_path)
    }
    assert set(by_kind) == {INSIGHT_KIND_PLANNING, INSIGHT_KIND_EXECUTION}
    # The model answered; the parser kept none of it.
    assert by_kind[INSIGHT_KIND_PLANNING]["empty_reason"] == EMPTY_REASON_PARSE
    assert by_kind[INSIGHT_KIND_PLANNING]["raw_output"] == (
        "- List the tasks before completing one"
    )
    # The model declined.
    assert by_kind[INSIGHT_KIND_EXECUTION]["empty_reason"] == EMPTY_REASON_EXTRACTOR
    assert _rows(db_path, "SELECT * FROM distillation_insights") == []


def test_one_extractor_delivering_does_not_hide_the_other_returning_nothing(
    observed_wec, monkeypatch, tmp_path
):
    """A run-wide insight total reads a half-empty run as a success; the flag
    is set from the per-extraction outcomes instead (§13.3)."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result, _dir = _run_distillation_with_extraction(
        ctx, sink, monkeypatch, tmp_path,
        [
            lambda cs: _PlanningPassAgent(
                cs, ["list_tasks", "complete_task"], cache_hit=False,
                plan="1. list the tasks 2. complete the first one",
            ),
            lambda cs: _PlanningPassAgent(
                cs, ["list_tasks"], cache_hit=False,
                plan="1. list the tasks and stop",
            ),
        ],
        planning_raw="1. Always list the tasks before completing one",
        execution_raw="EMPTY",
    )
    run = _rows(db_path, "SELECT * FROM distillation_runs")[0]
    assert (run["planning_insights"], run["execution_insights"]) == (1, 0)
    assert run["extractor_empty"] == 1
    assert [r["kind"] for r in _rows(
        db_path, "SELECT kind FROM distillation_insights"
    )] == [INSIGHT_KIND_PLANNING]


# ---------------------------------------------------------------------------
# (f) The §15 recipes, executed against a hand-checkable fixture — [DR54]
#
# "They parse and run" is retired as a standard of evidence: revision 1's
# promotion query returned `support_runs = 3` where the answer was 1, because a
# `LEFT JOIN … AND r.comparable = 1` nulls `r` but leaves the row in the COUNT.
# The fixture below is the one [DR54] requires — a non-comparable run, a replay
# run, a run-level NULL-`command_name` divergence — and every count is checked
# by hand in the test's own assertions.
# ---------------------------------------------------------------------------

def _seed_run(sink, run_id, **fields):
    """Insert one `distillation_runs` row through the real write path."""
    row = {
        "run_id": run_id,
        "turn_key": fields.pop("turn_key", f"turn-{run_id}"),
        "user_message": fields.pop("user_message", "complete my first task"),
        "workflow_name": fields.pop("workflow_name", "todo_list_workflow"),
        "entry_context": fields.pop("entry_context", "TodoList"),
        "comparable": fields.pop("comparable", 1),
        "started_at": fields.pop("started_at", "2026-08-01T00:00:00+00:00"),
        "run_json": "{}",
    }
    row.update(fields)
    sink.emit_distillation_record("run", row)
    return row


def _seed_divergence(sink, divergence_id, run_id, **fields):
    row = {
        "divergence_id": divergence_id,
        "run_id": run_id,
        "level": fields.pop("level", "action"),
        "left_pass": "teacher",
        "right_pass": "student",
        "align_index": fields.pop("align_index", 0),
        "kind": fields.pop("kind", "missing-in-student"),
        "material": fields.pop("material", 1),
        "command_key": fields.pop("command_key", "complete_task|{}"),
        "command_name": fields.pop("command_name", "complete_task"),
        "detail_json": "{}",
    }
    row.update(fields)
    sink.emit_distillation_record("divergence", row)
    return row


def _seed_insight(sink, iid, run_id, text, *, cites=()):
    sink.emit_distillation_record("insight", {
        "insight_id": iid,
        "run_id": run_id,
        "kind": INSIGHT_KIND_EXECUTION,
        "text": text,
        "text_hash": insight_text_hash(text),
        "created_at": "2026-08-01T00:00:00+00:00",
    })
    for divergence_id in cites:
        sink.emit_distillation_record(
            "citation", {"insight_id": iid, "divergence_id": divergence_id}
        )


@pytest.fixture
def recipe_corpus(tmp_path):
    """[DR54]'s fixture, built through the real sink and hand-countable.

    runA  comparable, not a replay, isolation-verified, cites a divergence
    runB  NON-COMPARABLE, same command and kind
    runC  a REPLAY of runA, same command and kind
    runD  comparable, NO divergence, with a student span for the cited command

    `isolation_verified = 1` is written here directly because the promotion
    view filters on it; the live writer refuses to set it `[DR48]`, so a
    corpus that exercises the query at all has to be seeded.
    """
    from fastworkflow.observability_store import SQLiteTraceSink

    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    try:
        _seed_run(sink, "runA", turn_key="turnA", isolation_verified=1,
                  started_at="2026-08-04T00:00:00+00:00")
        _seed_run(sink, "runB", turn_key="turnB", comparable=0,
                  comparable_reason="fingerprint-differs", isolation_verified=1,
                  started_at="2026-08-03T00:00:00+00:00")
        _seed_run(sink, "runC", turn_key="turnA~replay.1", replay_of="runA",
                  isolation_verified=1, started_at="2026-08-02T00:00:00+00:00")
        _seed_run(sink, "runD", turn_key="turnD", isolation_verified=1,
                  started_at="2026-08-01T00:00:00+00:00")

        _seed_divergence(sink, "div-A", "runA")
        _seed_divergence(sink, "div-B", "runB")
        _seed_divergence(sink, "div-C", "runC")
        # A run-level record: no command to key on, exactly the shape that
        # made the contradiction recipe return zero rows without an error.
        _seed_divergence(sink, "div-A-run", "runA", level="run", align_index=1,
                         kind="different-answer-same-actions",
                         command_key=None, command_name=None, material=1)

        _seed_insight(sink, "ins-1", "runA",
                      "Never complete a task before listing tasks",
                      cites=("div-A",))
        _seed_insight(sink, "ins-2", "runA",
                      "Answer with the task list, not a summary",
                      cites=("div-A-run",))

        # runD ran the cited command in its STUDENT pass and diverged on
        # nothing: it is the contradiction, and it needs a real span to be one.
        sink.emit_span(tracing.Span(
            span_id="span-runD-student",
            trace_id="turnD",
            name=tracing.SPAN_COMMAND_EXECUTE,
            kind=tracing.KIND_TOOL,
            command_name="complete_task",
            start_ns=1,
            end_ns=2,
            status=tracing.STATUS_OK,
            distillation_pass="student",
        ))
    finally:
        sink.close()
    return db_path


def test_the_documented_recipes_are_the_ones_the_design_ships():
    """Every recipe executed below appears verbatim in the design document, so
    a drift between the shipped SQL and the tested SQL fails here."""
    doc = _DESIGN_DOC.read_text(encoding="utf-8")
    for name, sql in (
        ("forward provenance", _SQL_FORWARD),
        ("reverse provenance", _SQL_REVERSE),
        ("support", _SQL_SUPPORT),
        ("contradiction", _SQL_CONTRADICTION),
        ("promotion", _SQL_PROMOTION),
        ("text_hash reverse index", _SQL_REVERSE_INDEX),
    ):
        assert sql in doc, f"the {name} recipe is not the document's"


def test_the_promotion_query_counts_one_supporter_not_three(recipe_corpus):
    """[DR54]'s regression, re-verified rather than assumed.

    runB is non-comparable and runC is a replay of the very run the insight
    came from; only runA is independent support. Revision 1 returned 3.
    """
    rows = {r["insight_id"]: r for r in _rows(recipe_corpus, _SQL_PROMOTION)}
    assert rows["ins-1"]["support_runs"] == 1
    assert rows["ins-1"]["material_support_runs"] == 1
    # The run-level insight keys on nothing, so it corroborates nothing — and
    # it is STILL LISTED with a zero rather than dropped, which is what the
    # correlated-subquery shape buys.
    assert rows["ins-2"]["support_runs"] == 0
    assert rows["ins-2"]["material_support_runs"] == 0
    assert set(rows) == {"ins-1", "ins-2"}
    assert rows["ins-1"]["verdict"] is None


def test_the_promotion_query_is_blocked_while_isolation_is_unverified(tmp_path):
    """`[DR48]`: sb8 never writes `isolation_verified = 1`, so on real data the
    promotion view lists every insight at zero — which fix-sb8.10 must render
    as "promotion is blocked", not as "no support found"."""
    from fastworkflow.observability_store import SQLiteTraceSink

    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    try:
        _seed_run(sink, "runA", turn_key="turnA")
        _seed_divergence(sink, "div-A", "runA")
        _seed_insight(sink, "ins-1", "runA", "Never guess", cites=("div-A",))
    finally:
        sink.close()

    rows = _rows(db_path, _SQL_PROMOTION)
    assert [r["insight_id"] for r in rows] == ["ins-1"]
    assert rows[0]["support_runs"] == 0


def test_the_support_recipe_returns_the_comparable_non_replay_runs(recipe_corpus):
    """§15's support query: runA alone, by hand — runB fails `comparable = 1`
    and runC fails `replay_of IS NULL`."""
    rows = _rows(recipe_corpus, _SQL_SUPPORT, {"insight_id": "ins-1"})
    assert [r["run_id"] for r in rows] == ["runA"]
    assert rows[0]["command_name"] == "complete_task"
    assert rows[0]["divergence_id"] == "div-A"


def test_the_contradiction_recipe_finds_the_run_the_rule_would_misfire_on(
    recipe_corpus,
):
    """§15's contradiction query: runD ran the cited command in its student
    pass and produced no divergence of the cited kind."""
    rows = _rows(recipe_corpus, _SQL_CONTRADICTION, {"insight_id": "ins-1"})
    assert [r["run_id"] for r in rows] == ["runD"]
    assert rows[0]["command_name"] == "complete_task"
    # The run-level insight has no command to key on. Without [DR54]'s
    # `cited.command_name IS NOT NULL` guard this returns zero rows for every
    # kind, with no error — which reads as "no contradictions found".
    assert _rows(recipe_corpus, _SQL_CONTRADICTION, {"insight_id": "ins-2"}) == []


def test_an_extractor_that_raises_still_closes_its_span_as_an_error(
    observed_wec, monkeypatch, tmp_path
):
    """The span is opened around the model call, so a failing extractor leaves
    a closed error span rather than an open one — and writes no ledger row for
    an insight that was never produced."""
    from fastworkflow.distillation import DistillationSession

    ctx, sink, db_path = observed_wec
    real_execution = DistillationSession.extract_insights
    real_planning = DistillationSession.extract_planning_insights
    queued = list(_diverging_agents())
    _script_llm_boundaries(
        monkeypatch, ctx, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )
    monkeypatch.setattr(DistillationSession, "extract_insights", real_execution)
    monkeypatch.setattr(
        DistillationSession, "extract_planning_insights", real_planning
    )

    class _RaisingChainOfThought:
        def __init__(self, signature):
            pass

        def __call__(self, **_kwargs):
            raise RuntimeError("extractor unavailable")

    monkeypatch.setattr(dspy_module, "ChainOfThought", _RaisingChainOfThought)

    ctx._begin_turn("list my tasks")
    with pytest.raises(RuntimeError):
        distill_message(ctx, "list my tasks")
    sink.close()

    spans = _extract_spans(db_path)
    assert len(spans) == 1
    assert spans[0]["status"] == tracing.STATUS_ERROR
    assert spans[0]["end_ns"] is not None
    assert json.loads(spans[0]["attributes"])["error_type"] == "RuntimeError"
    assert _rows(db_path, "SELECT * FROM distillation_insights") == []
    assert _rows(db_path, "SELECT * FROM distillation_insight_citations") == []
