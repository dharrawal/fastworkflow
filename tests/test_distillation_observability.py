"""The consolidated acceptance suite for distillation observability — fix-sb8.14.

Five children built the P1 backbone (fix-sb8.1 through fix-sb8.5). This file is
the evidence that the *epic* holds, not that each child's file does. It is
organized in three parts:

1. **The nine acceptance criteria**, one section each, named
   `test_ac<N>_…` so the mapping from criterion to test is readable from the
   `-v` output alone.
2. **The five blocking findings the Phase-0 gate caught** (§22). Each was a real
   defect in revision 1 of the design that would have shipped green, so each
   gets a test that fails if it comes back.
3. **The silent-failure sites §19 names**, none of which raises: the
   `emit_span` field copy, the `list_turns(command_name=…)` regression, the
   pre-distillation-DB degradation `[DR29]`, `[DR11]`'s byte-identity default,
   `[DR46]`'s write-path failure containment, and the cost of the feature when
   observability is off.

Per `.cursor/rules/testing_rules.mdc` and design §19: no mocks. A real
`WorkflowExecutionContext` over `tests/todo_list_workflow`, a real
`SQLiteTraceSink` writing a real SQLite file in `tmp_path`, the real
`distill_message` / `_run_agent_pass` / `align_and_record` / extractors /
writer thread. Only the LLM boundaries are scripted, with scripted-agent
doubles driving the real span emitters — the path §18's `fix-sb8.14` note
prescribes, because none of `LLM_TEACHER_AGENT` / `LLM_STUDENT_AGENT` /
`LLM_DISTILLATION` exists in the env template and `dspy_utils.get_lm` raises
when they are unset.

Part 4 holds three tests that were marked `xfail(strict=True)` when this file
was written, each asserting a normative design statement the shipped code did
not implement — `[DR35]`'s `replayable` carve-out, §10.3's pin-at-write-time,
and `[DR51]`'s `fw.ask_user` reparenting. All three producers have since landed
(the audit's findings 1/2/3, 6/8/10 and 7/9), so the strict xfails turned into
XPASSes and the markers were removed. The assertions are untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import dspy as dspy_module
import pytest

import fastworkflow
from fastworkflow import tracing
from fastworkflow import distillation_alignment as alignment
from fastworkflow.distillation import (
    DistillationSession,
    distill_message,
    insight_id,
    insight_text_hash,
    state_fingerprint,
)
from fastworkflow.observability_store import (
    COUNT_LIVE_SPANS,
    FEATURE_DISTILLATION_V1,
    ObservabilityStore,
    OrphanedCitation,
    Redactor,
    ReadOnlyObservabilityStore,
    SQLiteTraceSink,
)
from fastworkflow.utils.insights_loader import strip_insight_markers
from fastworkflow.workflow_agent import _append_action_record
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

_DESIGN_DOC = (
    Path(__file__).parent.parent / "docs" / "distillation_observability_design.md"
)
_TODO_WORKFLOW = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())

# The six tables [DR44]'s erasure obligation names, in the order the store
# deletes them.
_DISTILL_TABLES = (
    "distillation_verdicts",
    "distillation_insight_citations",
    "distillation_insights",
    "distillation_divergences",
    "distillation_passes",
    "distillation_runs",
)


# ---------------------------------------------------------------------------
# Harness — real components, scripted LLM boundaries only
# ---------------------------------------------------------------------------

@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


def _bound_wec() -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=True)
    workflow = fastworkflow.Workflow.create(
        _TODO_WORKFLOW, workflow_id_str=f"sb8-14-{uuid.uuid4().hex}"
    )
    ctx.bind_app_workflow(workflow)
    ctx.push_active_workflow(workflow)
    return ctx


@pytest.fixture
def observed_wec(initialized_fastworkflow, tmp_path):
    """A real WEC writing to a real observability DB. (ctx, sink, db_path)."""
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    ctx = _bound_wec()
    ctx.set_trace_sink(sink)
    ctx.bind_observability_identity(channel_id="sb8-14-channel")
    try:
        yield ctx, sink, db_path
    finally:
        ctx.clear_workflow_stack()
        ctx.close()
        sink.close()


@pytest.fixture
def bare_wec(initialized_fastworkflow):
    """A real WEC with NO sink at all — the observability-off configuration."""
    ctx = _bound_wec()
    try:
        yield ctx
    finally:
        ctx.clear_workflow_stack()
        ctx.close()


def _rows(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _one(db_path: str, sql: str, params: tuple = ()) -> dict:
    found = _rows(db_path, sql, params)
    assert len(found) == 1, f"expected exactly one row, got {len(found)}"
    return found[0]


def _action(command_name: str, **params) -> dict:
    return {
        "command": command_name,
        "command_name": command_name,
        "parameters": params,
        "response": f"ran {command_name}",
    }


class _PassAgent:
    """Stands where DSPy's ReAct call stands, emitting its pass's spans for real.

    The commands it "runs" emit real `fw.command.execute` spans in
    `CommandExecutor.invoke_command`'s shape — `raw_command` at open, then
    `command_name` plus `parameters`/`response_text`/`success` at close — as
    well as the action record, because `[DR17]` aligns over the spans and a
    double that emitted only the action record makes every pass look
    actionless. The `fw.llm.call` span carries `dspy_logger`'s attribute shape
    (`usage` as a JSON *string*), which is what §6.3's rollup queries.

    `failed` names commands that error: they close with `command_name=None`,
    exactly as a command that never resolved to a name does, which is the
    `[DR50]` case.
    """

    def __init__(
        self,
        chat_session,
        command_names,
        *,
        cache_hit=False,
        params=None,
        failed=(),
        mutate=None,
        answer="done",
    ):
        self._chat_session = chat_session
        self._command_names = command_names
        self._cache_hit = cache_hit
        self._params = params or {}
        self._failed = list(failed)
        self._mutate = mutate
        self._answer = answer
        self.current_trajectory: dict = {}

    def _command_span(self, raw_command, command_name, parameters, success):
        span = tracing.start_span(
            self._chat_session,
            tracing.SPAN_COMMAND_EXECUTE,
            kind=tracing.KIND_TOOL,
            attributes={"raw_command": raw_command},
        )
        tracing.end_span(
            self._chat_session,
            span,
            command_name=command_name,
            attributes={
                "parameters": parameters,
                "response_text": f"ran {raw_command}",
                "success": success,
            },
        )
        return span

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
            self._command_span(name, name, self._params.get(name, {}), True)
            _append_action_record(
                self._chat_session, _action(name, **self._params.get(name, {}))
            )
        for raw in self._failed:
            # A failed command never resolves to a command_name: the span
            # closes with the column NULL and only `raw_command` to key on.
            self._command_span(raw, None, {}, False)
        if self._mutate is not None:
            self._mutate(self._chat_session)
        self.current_trajectory = {"thought_0": "scripted"}
        return type("AgentResult", (), {"final_answer": self._answer})()


class _PlanningPassAgent(_PassAgent):
    """A pass agent that also emits its `fw.planner.plan` span, so the PLAN
    level has steps to align (§7.1 reads the full plan string off that span)."""

    def __init__(self, chat_session, command_names, *, plan, **kwargs):
        super().__init__(chat_session, command_names, **kwargs)
        self._plan = plan

    def __call__(self, **kwargs):
        span = tracing.start_span(
            self._chat_session,
            tracing.SPAN_PLANNER_PLAN,
            attributes={"plan": self._plan},
        )
        tracing.end_span(self._chat_session, span)
        return super().__call__(**kwargs)


def _teacher_wrote(chat_session):
    """A teacher-only state change, so the two passes EXIT differently."""
    chat_session.get_active_workflow().context["teacher_wrote"] = "this"


def _script_llm_boundaries(monkeypatch, ctx, agent_factory):
    """Script only the LLM-touching boundaries of a distillation run."""
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


def _script_extractors(monkeypatch, *, planning_raw="EMPTY", execution_raw="EMPTY"):
    """Script the two extractor LLM calls at the dspy boundary only.

    The extractors themselves stay real — the `fw.distill.extract` span, the
    parse, the ids, the file append and the ledger writes are all shipped code.
    """
    from fastworkflow.distillation import PlanningInsightExtractionSignature

    class _ScriptedChainOfThought:
        def __init__(self, signature):
            self._signature = signature

        def __call__(self, **kwargs):
            raw = (
                planning_raw
                if self._signature is PlanningInsightExtractionSignature
                else execution_raw
            )
            return type("Prediction", (), {"insights": raw})()

    monkeypatch.setattr(dspy_module, "ChainOfThought", _ScriptedChainOfThought)


def _run(
    ctx,
    monkeypatch,
    agents,
    *,
    message="list my tasks",
    planning_raw=None,
    execution_raw=None,
    insights_dir=None,
):
    """One real two-pass distillation. Does NOT close the sink, so a test can
    run several turns on one context before reading the DB.

    `planning_raw` / `execution_raw` left None stubs the extractors out
    entirely (the run still records its divergences); supplying either puts the
    REAL extractors back and scripts the model's answer underneath them.
    """
    queued = list(agents)
    _script_llm_boundaries(
        monkeypatch, ctx, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )
    if planning_raw is None and execution_raw is None:
        monkeypatch.setattr(DistillationSession, "extract_insights", lambda *a, **k: [])
        monkeypatch.setattr(
            DistillationSession, "extract_planning_insights", lambda *a, **k: []
        )
    else:
        _script_extractors(
            monkeypatch,
            planning_raw=planning_raw or "EMPTY",
            execution_raw=execution_raw or "EMPTY",
        )
        assert insights_dir is not None, "a real extraction writes a real file"
        insights_dir.mkdir(parents=True, exist_ok=True)
        # The real append path runs, against a directory the TEST owns: the
        # workflow under `tests/` is checked in and an insights file written
        # into it would outlive the run.
        monkeypatch.setattr(
            DistillationSession, "_insights_dir", lambda self: insights_dir
        )
    ctx._begin_turn(message)
    turn_key = ctx.current_turn_key
    result = distill_message(ctx, message)
    return turn_key, result


def _diverging_agents(**kwargs):
    """Teacher runs a command the student skips — one action divergence."""
    return [
        lambda cs: _PassAgent(
            cs, ["list_tasks", "complete_task"], mutate=_teacher_wrote, **kwargs
        ),
        lambda cs: _PassAgent(cs, ["list_tasks"], **kwargs),
    ]


def _agreeing_agents(commands=("list_tasks", "complete_task"), **kwargs):
    """Both passes take the same actions — the no-divergence class (§13.3)."""
    return [
        lambda cs: _PassAgent(cs, list(commands), **kwargs),
        lambda cs: _PassAgent(cs, list(commands), **kwargs),
    ]


def _pass_spans(db_path: str, turn_key: str, label: str) -> list[dict]:
    """What `[DR6]`'s `get_spans(trace_id, distillation_pass=…)` will select.

    The read API is fix-sb8.6 and is not built, so the filter is expressed as
    the SQL it compiles to. `[DR23]` is the point: this is a real column with a
    partial index, so the filter is documented SQL an agent can write.
    """
    return _rows(
        db_path,
        "SELECT * FROM spans WHERE trace_id=? AND distillation_pass=? "
        "ORDER BY start_ns",
        (turn_key, label),
    )


def _ancestors(spans_by_id: dict, span_id: str) -> list[str]:
    """The parent chain of *span_id*, nearest first, stopping at the root."""
    chain: list[str] = []
    seen: set[str] = set()
    current = spans_by_id.get(span_id, {}).get("parent_span_id")
    while current and current not in seen:
        seen.add(current)
        chain.append(current)
        current = spans_by_id.get(current, {}).get("parent_span_id")
    return chain


# ===========================================================================
# ACCEPTANCE CRITERION 1
# A --generate_insights turn yields two independently viewable traces
# (teacher, student) plus one grouping run record; neither waterfall
# interleaves the other.  §3.6 [DR6][DR7][DR8], §8, §9
# ===========================================================================

def test_ac1_a_distilled_turn_yields_two_pass_traces_and_one_grouping_run_record(
    observed_wec, monkeypatch
):
    """`[DR8]`'s separation assertion, in the language the repo can test in.

    The two pass span sets are non-empty and disjoint, no student span hangs
    off the `fw.turn` root, and one `distillation_runs` row groups them.
    """
    ctx, sink, db_path = observed_wec
    turn_key, result = _run(ctx, monkeypatch, _diverging_agents())
    sink.close()

    teacher = {s["span_id"] for s in _pass_spans(db_path, turn_key, "teacher")}
    student = {s["span_id"] for s in _pass_spans(db_path, turn_key, "student")}
    assert teacher and student
    assert not (teacher & student)

    # This half of [DR8] holds for `fw.ask_user` too, since [DR51] landed —
    # see `test_an_ask_user_inside_a_pass_parents_onto_its_own_pass_span`.
    root = tracing.root_span_id(turn_key)
    for span in _pass_spans(db_path, turn_key, "student"):
        assert span["parent_span_id"] != root, (
            "a pass span parented on the turn root is a span the pass filter "
            "cannot separate structurally"
        )

    # Exactly one grouping record, and it is the run the call reported.
    run = _one(db_path, "SELECT * FROM distillation_runs WHERE turn_key=?", (turn_key,))
    assert run["run_id"] == result.run_id
    assert run["user_message"] == "list my tasks"
    assert run["completed_at"] is not None

    # Both passes are recorded against it, in execution order, on the ONE
    # trace id the identity ruling keeps ([DR1]: trace_id == turn_key).
    passes = _rows(
        db_path,
        "SELECT * FROM distillation_passes WHERE run_id=? ORDER BY seq",
        (run["run_id"],),
    )
    assert [(p["pass_label"], p["role"], p["seq"]) for p in passes] == [
        ("teacher", "teacher", 0),
        ("student", "student", 1),
    ]
    assert {p["trace_id"] for p in passes} == {turn_key}


def test_ac1_every_pass_span_descends_from_its_own_pass_span_and_never_the_others(
    observed_wec, monkeypatch
):
    """§3.6's parenting half: the hierarchy, not only the column, separates.

    §8's structure is `turn -> fw.distill.run -> fw.distill.pass -> the pass's
    work`. Deriving "which pass is this span in" from the column is the index;
    the hierarchy is what makes the unfiltered waterfall render the run rather
    than an interleave (§18, fix-kw7.11).
    """
    ctx, sink, db_path = observed_wec
    turn_key, _result = _run(ctx, monkeypatch, _diverging_agents())
    sink.close()

    spans = {s["span_id"]: s for s in _rows(db_path, "SELECT * FROM spans")}
    wrappers = {
        s["distillation_pass"]: s["span_id"]
        for s in spans.values()
        if s["name"] == tracing.SPAN_DISTILL_PASS
    }
    assert set(wrappers) == {"teacher", "student"}
    # [DR51]: the pass span id is computable, not random — that is what lets
    # the ask_user close site parent onto it.
    for label, span_id in wrappers.items():
        assert span_id == tracing.distill_pass_span_id(turn_key, label)

    run_span = _one(
        db_path, "SELECT * FROM spans WHERE name=?", (tracing.SPAN_DISTILL_RUN,)
    )
    assert run_span["parent_span_id"] == tracing.root_span_id(turn_key)
    # Run-level spans carry NULL, per [DR7]'s pass-label table.
    assert run_span["distillation_pass"] is None
    for label, span_id in wrappers.items():
        assert spans[span_id]["parent_span_id"] == run_span["span_id"]

    for label, other in (("teacher", "student"), ("student", "teacher")):
        own = wrappers[label]
        foreign = wrappers[other]
        work = [s for s in _pass_spans(db_path, turn_key, label) if s["span_id"] != own]
        assert work, f"{label} pass emitted no work spans"
        for span in work:
            chain = _ancestors(spans, span["span_id"])
            assert own in chain, f"{span['name']} does not descend from its own pass"
            assert foreign not in chain


def test_ac1_neither_pass_waterfall_interleaves_the_other(observed_wec, monkeypatch):
    """The origin symptom of fix-kw7.11, asserted as a fact about the data.

    Two passes that interleave in time cannot be laid out as two waterfalls no
    matter what the SPA does; two passes whose windows are disjoint can be,
    and `[DR7]`'s per-pass time window is then real rather than nominal.
    """
    ctx, sink, db_path = observed_wec
    turn_key, _result = _run(ctx, monkeypatch, _diverging_agents())
    sink.close()

    windows = {}
    for label in ("teacher", "student"):
        rows = _pass_spans(db_path, turn_key, label)
        assert rows
        assert all(r["end_ns"] for r in rows), "an unclosed span has no extent"
        windows[label] = (
            min(r["start_ns"] for r in rows),
            max(r["end_ns"] for r in rows),
        )
    teacher_start, teacher_end = windows["teacher"]
    student_start, student_end = windows["student"]
    assert teacher_start < teacher_end
    assert student_start < student_end
    assert teacher_end <= student_start, (
        f"pass windows overlap: teacher {windows['teacher']} vs student "
        f"{windows['student']} — the two waterfalls interleave"
    )


# ===========================================================================
# ACCEPTANCE CRITERION 2
# Every emitted insight resolves, in ONE query, to the divergence record and
# the teacher/student span pair that justified it, AND back again.
# §13.2 [DR32]
# ===========================================================================

# The §13.2 and §15 recipes, verbatim. `test_the_sql_this_suite_runs_is_the_
# sql_the_design_ships` pins each string against the design document, so what
# runs below is the shipped SQL and not a paraphrase of it — which is the
# standard of evidence [DR54] installed after "it parses" let two silently
# wrong queries through.
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


def test_the_sql_this_suite_runs_is_the_sql_the_design_ships():
    """[DR54]: a recipe is not verified until it is executed, and it is not
    the shipped recipe unless it is byte-for-byte the document's."""
    doc = _DESIGN_DOC.read_text(encoding="utf-8")
    for name, sql in (
        ("forward provenance §13.2", _SQL_FORWARD),
        ("reverse provenance §13.2", _SQL_REVERSE),
        ("support §15", _SQL_SUPPORT),
        ("contradiction §15", _SQL_CONTRADICTION),
    ):
        assert sql.strip() in doc, f"{name} has drifted from the design"


def _extracted_insight(db_path: str) -> dict:
    return _one(db_path, "SELECT * FROM distillation_insights")


def test_ac2_an_insight_resolves_to_its_divergence_and_span_pair_in_one_query(
    observed_wec, monkeypatch, tmp_path
):
    """§13.2 forward: insight -> citation -> divergence -> the span pair.

    One query, no scraping: the answer to "why does rule 7 exist" is a row set.
    """
    ctx, sink, db_path = observed_wec
    turn_key, _result = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    sink.close()

    insight = _extracted_insight(db_path)
    found = _rows(db_path, _SQL_FORWARD, {"insight_id": insight["insight_id"]})
    assert found, "the insight resolves to nothing — provenance is broken"

    # It resolves to the divergence the extractor was actually shown: the
    # teacher ran a command the student skipped.
    kinds = {row["divergence_kind"] for row in found}
    assert alignment.KIND_MISSING_IN_STUDENT in kinds
    # `identical` records are stored (§7.3) but are not evidence for anything,
    # so they are never cited.
    assert alignment.KIND_IDENTICAL not in kinds

    row = next(r for r in found if r["divergence_kind"] == alignment.KIND_MISSING_IN_STUDENT)
    assert row["turn_key"] == turn_key
    assert row["comparable"] == 1
    assert (row["left_pass"], row["right_pass"]) == ("teacher", "student")
    assert row["command_name"] == "complete_task"
    assert row["extractor_span_id"] is not None

    # The span pair is the point: left_span_id is a real fw.command.execute
    # row in the teacher pass. (The student has none — that IS the divergence.)
    spans = {s["span_id"]: s for s in _rows(db_path, "SELECT * FROM spans")}
    assert row["left_span_id"] in spans
    assert spans[row["left_span_id"]]["name"] == tracing.SPAN_COMMAND_EXECUTE
    assert spans[row["left_span_id"]]["distillation_pass"] == "teacher"
    assert row["right_span_id"] is None
    # The extractor's own call is addressable from the same row.
    assert spans[row["extractor_span_id"]]["name"] == tracing.SPAN_DISTILL_EXTRACT


def test_ac2_a_span_resolves_back_to_the_insights_drawn_from_it(
    observed_wec, monkeypatch, tmp_path
):
    """§13.2 reverse: the developer is looking at a span and asks what rule
    came out of it. Same chain, walked the other way."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    sink.close()

    insight = _extracted_insight(db_path)
    cited = _one(
        db_path,
        "SELECT d.* FROM distillation_divergences d "
        "JOIN distillation_insight_citations c ON c.divergence_id=d.divergence_id "
        "WHERE c.insight_id=?",
        (insight["insight_id"],),
    )
    back = _rows(db_path, _SQL_REVERSE, {"span_id": cited["left_span_id"]})
    assert [r["insight_id"] for r in back] == [insight["insight_id"]]
    assert back[0]["text"] == insight["text"]

    # A span that justified nothing resolves to nothing — the reverse index
    # answers "no rule came from this", not "some rule may have".
    uncited = _one(
        db_path,
        "SELECT * FROM spans WHERE name=? AND distillation_pass=? AND command_name=?",
        (tracing.SPAN_COMMAND_EXECUTE, "student", "list_tasks"),
    )
    assert _rows(db_path, _SQL_REVERSE, {"span_id": uncited["span_id"]}) == []


def test_ac2_the_markdown_line_the_rule_lives_on_carries_the_id_back(
    observed_wec, monkeypatch, tmp_path
):
    """The other half of round-tripping: `[DR31]`'s marker on the written line.

    `text_hash` stops resolving the moment a human edits the line, which is
    exactly when you most want the provenance; the marker survives the edit.
    And `[DR56]` keeps it out of every prompt.
    """
    ctx, sink, db_path = observed_wec
    insights_dir = tmp_path / "Insights" / "todo_list_workflow"
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        execution_raw="- Always complete the task the user asked about",
        insights_dir=insights_dir,
    )
    sink.close()

    insight = _extracted_insight(db_path)
    written = Path(insight["insight_file"]).read_text(encoding="utf-8")
    assert insight["insight_id"] in written
    assert insight["text"] in written
    # [DR56]: stripped on load, so no prompt consumer ever sees the marker.
    assert insight["insight_id"] not in strip_insight_markers(written)
    # `file_entry_number` is DISPLAY ONLY (§13.1): the id is a pure function
    # of (run_id, kind, text), so renumbering the file cannot orphan it.
    assert insight["file_entry_number"] == 1
    assert insight["insight_id"] == insight_id(
        insight["run_id"], insight["kind"], insight["text"]
    )
    assert insight["text_hash"] == insight_text_hash(insight["text"])


# ===========================================================================
# ACCEPTANCE CRITERION 3
# Divergences are stored as structured aligned pairs with a kind and a
# materiality flag, not a prose string.  §7 [DR17][DR18][DR19][DR20]
# ===========================================================================

def _param_diverging_agents():
    """Both passes run the same two commands; one parameter value differs."""
    return [
        lambda cs: _PassAgent(
            cs, ["list_tasks", "complete_task"], params={"complete_task": {"id": "1"}}
        ),
        lambda cs: _PassAgent(
            cs, ["list_tasks", "complete_task"], params={"complete_task": {"id": "2"}}
        ),
    ]


def test_ac3_a_divergence_is_an_aligned_pair_with_a_kind_and_a_param_diff(
    observed_wec, monkeypatch
):
    """The row is the evidence. Before this epic the only artifact of a
    comparison was a formatted string handed to the extractor and thrown away;
    a set difference over formatted actions reported one changed parameter as
    two unrelated orphans."""
    ctx, sink, db_path = observed_wec
    _turn_key, result = _run(ctx, monkeypatch, _param_diverging_agents())
    sink.close()

    records = _rows(
        db_path,
        "SELECT * FROM distillation_divergences ORDER BY level, align_index",
    )
    assert [r["kind"] for r in records] == [
        alignment.KIND_IDENTICAL,
        alignment.KIND_PARAM_VALUE_ONLY,
    ]
    for row in records:
        assert row["kind"] in alignment.KINDS
        assert row["run_id"] == result.run_id
        assert row["level"] in (alignment.LEVEL_PLAN, alignment.LEVEL_ACTION,
                                alignment.LEVEL_RUN)
        assert (row["left_pass"], row["right_pass"]) == ("teacher", "student")
        assert isinstance(row["align_index"], int)
        assert row["command_key"], "an aligned pair keys on the command"
        assert row["material"] in (0, 1)
        assert row["detail_json"], "detail_json is NOT NULL by DDL"

    changed = records[1]
    assert changed["command_name"] == "complete_task"
    # Both sides of the pair resolve to real spans — this is one step compared,
    # not two orphans.
    assert changed["left_step_key"] and changed["right_step_key"]
    assert changed["left_step_key"] != changed["right_step_key"]
    assert changed["left_span_id"] and changed["right_span_id"]
    assert changed["left_span_id"] != changed["right_span_id"]

    # The structured per-key diff the UI renders, one level deep, all three
    # keys always present so the shape is stable.
    diff = json.loads(changed["param_diff_json"])
    assert diff == {
        "changed": {"id": {"left": "1", "right": "2"}},
        "left_only": {},
        "right_only": {},
    }
    # An `identical` record has nothing to highlight, so the column is NULL —
    # absent, rather than an empty-shaped lie.
    assert records[0]["param_diff_json"] is None


def test_ac3_the_prose_is_rendered_from_the_stored_rows_not_stored_as_prose(
    observed_wec, monkeypatch
):
    """§7.6: the summary the extractor is shown is a *rendering* of the rows.

    Rebuild the records from the stored columns alone and re-render: byte-equal
    to what the run recorded. If the rows were a lossy shadow of a prose blob,
    this could not hold.
    """
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(ctx, monkeypatch, _param_diverging_agents())
    sink.close()

    rows = _rows(
        db_path,
        "SELECT * FROM distillation_divergences WHERE level=? ORDER BY align_index",
        (alignment.LEVEL_ACTION,),
    )
    rebuilt = [
        alignment.DivergenceRecord(
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
            param_diff=json.loads(row["param_diff_json"])
            if row["param_diff_json"]
            else None,
            detail=json.loads(row["detail_json"]),
        )
        for row in rows
    ]
    run = _one(db_path, "SELECT * FROM distillation_runs")
    recorded = json.loads(run["run_json"])["execution_summary"]
    assert recorded
    assert alignment.render_divergence_summary(rebuilt) == recorded

    # And no column is the prose: the summary lives on the run row's own
    # json blob, derived; the divergence rows carry structure.
    for row in rows:
        assert recorded not in json.dumps(dict(row))


def test_ac3_the_plan_level_is_aligned_and_recorded_beside_the_action_level(
    observed_wec, monkeypatch
):
    """§7.1 aligns two levels, each with its own `fw.distill.compare` span
    (§8) — so "the plans differed" and "the actions differed" are separate
    stored facts, feeding the two separate extractors."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        [
            lambda cs: _PlanningPassAgent(
                cs, ["list_tasks"], plan="1. list the tasks\n2. complete the one named"
            ),
            lambda cs: _PlanningPassAgent(cs, ["list_tasks"], plan="1. list the tasks"),
        ],
    )
    sink.close()

    compares = _rows(
        db_path,
        "SELECT * FROM spans WHERE name=? ORDER BY start_ns",
        (tracing.SPAN_DISTILL_COMPARE,),
    )
    levels = [json.loads(row["attributes"])["level"] for row in compares]
    assert levels == [alignment.LEVEL_PLAN, alignment.LEVEL_ACTION]
    # Run-level work, so the compare wrappers carry no pass label ([DR7]).
    assert all(row["distillation_pass"] is None for row in compares)

    plan_records = _rows(
        db_path,
        "SELECT * FROM distillation_divergences WHERE level=? ORDER BY align_index",
        (alignment.LEVEL_PLAN,),
    )
    assert plan_records, "the plan level recorded nothing"
    assert {r["command_name"] for r in plan_records} == {alignment.PLAN_COMMAND}
    assert any(r["kind"] != alignment.KIND_IDENTICAL for r in plan_records)

    run = _one(db_path, "SELECT * FROM distillation_runs")
    assert run["planning_diverged"] == 1
    # The actions agreed, so the two levels really are separate verdicts.
    assert run["exec_diverged"] == 0

# ===========================================================================
# ACCEPTANCE CRITERION 4
# Both pass-entry state fingerprints are recorded and compared; a run whose
# passes did not start identical is flagged non-comparable.  §6 [DR14][DR15][DR47]
# ===========================================================================

def test_ac4_both_pass_entry_fingerprints_are_recorded_and_compared(
    observed_wec, monkeypatch
):
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(ctx, monkeypatch, _diverging_agents())
    sink.close()

    passes = _rows(db_path, "SELECT * FROM distillation_passes ORDER BY seq")
    assert [p["pass_label"] for p in passes] == ["teacher", "student"]
    for row in passes:
        assert row["entry_fingerprint"], f"{row['pass_label']} has no entry fingerprint"
        assert row["exit_fingerprint"], f"{row['pass_label']} has no exit fingerprint"
    assert passes[0]["entry_fingerprint"] == passes[1]["entry_fingerprint"]

    run = _one(db_path, "SELECT * FROM distillation_runs")
    # The comparison, denormalized onto the run row and reduced to a verdict.
    assert run["fingerprint_teacher"] == passes[0]["entry_fingerprint"]
    assert run["fingerprint_student"] == passes[1]["entry_fingerprint"]
    assert (run["comparable"], run["comparable_reason"]) == (1, None)
    # The teacher mutated state, so the two passes EXITED differently — which
    # is what makes the unmatched step material.
    assert passes[0]["exit_fingerprint"] != passes[1]["exit_fingerprint"]


def test_ac4_a_run_whose_passes_did_not_start_identical_is_flagged_non_comparable(
    observed_wec, monkeypatch
):
    """End to end, through `distill_message`, with no production code patched.

    The teacher navigates to a command context. `restore_workflow_state`
    restores the two context dicts and `_is_complete` and nothing else, so the
    student enters somewhere the teacher did not — exactly the class of
    unrestored state the flag exists to catch. Its divergences are unusable by
    contract, so `material` is NULL on every one of them ([DR20]), and they are
    still STORED (§6.2 obligation 5: recorded, never deleted).
    """
    from tests.todo_list_workflow.application.todo_list import TodoList

    def navigate(chat_session):
        chat_session.get_active_workflow().current_command_context = TodoList(
            1, "teacher went here"
        )

    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks", "complete_task"], mutate=navigate),
            lambda cs: _PassAgent(cs, ["list_tasks"]),
        ],
    )
    sink.close()

    passes = _rows(db_path, "SELECT * FROM distillation_passes ORDER BY seq")
    assert passes[0]["entry_fingerprint"] != passes[1]["entry_fingerprint"]

    run = _one(db_path, "SELECT * FROM distillation_runs")
    assert run["comparable"] == 0
    assert run["comparable_reason"] == "fingerprint-differs"
    assert run["fingerprint_teacher"] != run["fingerprint_student"]

    records = _rows(db_path, "SELECT * FROM distillation_divergences")
    assert records, "a non-comparable run still records what it saw"
    assert all(r["material"] is None for r in records)
    assert run["material_divergences"] == 0


# ===========================================================================
# ACCEPTANCE CRITERION 5
# Negative outcomes are recorded: divergence-with-EMPTY-extraction, and
# no-divergence turns.  §13.3 [DR33]
# ===========================================================================

def test_ac5_a_divergence_that_extracted_nothing_is_a_row_with_a_reason(
    observed_wec, monkeypatch, tmp_path
):
    """"Divergence found — 0 insights extracted" used to be a console line and
    nothing else. It is now a row, and `empty_reason` separates "the extractor
    judged it context-justified" from "the parser discarded the answer"."""
    ctx, sink, db_path = observed_wec
    insights_dir = tmp_path / "Insights" / "todo_list_workflow"
    _turn_key, result = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        execution_raw="EMPTY",
        insights_dir=insights_dir,
    )
    sink.close()

    run = _one(db_path, "SELECT * FROM distillation_runs")
    assert run["exec_diverged"] == 1
    assert run["execution_insights"] == 0
    assert run["extractor_empty"] == 1
    assert result.insights_extracted == 0

    # The evidence survives the empty extraction: the divergence is still on
    # record for a later rule to be tested against.
    assert _rows(db_path, "SELECT * FROM distillation_divergences")
    assert _rows(db_path, "SELECT * FROM distillation_insights") == []
    # Nothing was appended to the corpus, so a later run's prompt is unchanged.
    assert list(insights_dir.glob("*.md")) == []

    extract = _one(
        db_path,
        "SELECT * FROM spans WHERE name=? AND json_extract(attributes,'$.kind')=?",
        (tracing.SPAN_DISTILL_EXTRACT, "execution"),
    )
    attributes = json.loads(extract["attributes"])
    assert attributes["empty_reason"] == "extractor-returned-empty"
    assert attributes["parsed_count"] == 0


def test_ac5_a_turn_with_no_divergence_at_all_is_a_row(observed_wec, monkeypatch):
    """The "student was already correct" set — the contradiction pool for every
    candidate rule (§15) — is a row, not an absence. It is also why `identical`
    divergence records are stored."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(ctx, monkeypatch, _agreeing_agents())
    sink.close()

    run = _one(db_path, "SELECT * FROM distillation_runs")
    assert (run["planning_diverged"], run["exec_diverged"]) == (0, 0)
    assert run["material_divergences"] == 0
    # Nothing diverged, so nothing was extracted and nothing was empty.
    assert run["extractor_empty"] == 0
    assert run["comparable"] == 1
    assert run["completed_at"] is not None

    records = _rows(db_path, "SELECT * FROM distillation_divergences")
    assert records, "the aligned pairs are stored even when they all agree"
    assert {r["kind"] for r in records} == {alignment.KIND_IDENTICAL}
    assert _rows(db_path, "SELECT * FROM distillation_insights") == []
    # No extractor ran at all, so there is no extract span to misread as one.
    assert _rows(
        db_path, "SELECT * FROM spans WHERE name=?", (tracing.SPAN_DISTILL_EXTRACT,)
    ) == []


# ===========================================================================
# ACCEPTANCE CRITERION 6
# The extractor LLM call is a span with its prompt inputs and its raw output.
# §8, §13 [DR21]
# ===========================================================================

def test_ac6_the_extractor_call_is_a_span_with_its_prompt_inputs_and_raw_output(
    observed_wec, monkeypatch, tmp_path
):
    """The step that actually decides what the rule is was the one step of a
    distillation run with no span at all."""
    ctx, sink, db_path = observed_wec
    raw = "- Always complete the task the user asked about"
    turn_key, _result = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        execution_raw=raw,
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    sink.close()

    extract = _one(
        db_path,
        "SELECT * FROM spans WHERE name=? AND json_extract(attributes,'$.kind')=?",
        (tracing.SPAN_DISTILL_EXTRACT, "execution"),
    )
    assert extract["trace_id"] == turn_key
    assert extract["kind"] == tracing.KIND_INTERNAL
    assert extract["status"] == tracing.STATUS_OK
    # §3.6/[DR7]: the extractor is its own pass label, under the run wrapper.
    assert extract["distillation_pass"] == "extractor"
    run_span = _one(
        db_path, "SELECT * FROM spans WHERE name=?", (tracing.SPAN_DISTILL_RUN,)
    )
    assert extract["parent_span_id"] == run_span["span_id"]

    attributes = json.loads(extract["attributes"])
    # Its prompt inputs...
    assert attributes["divergence_summary"], "the prompt's evidence is on the span"
    assert "complete_task" in attributes["divergence_summary"]
    assert attributes["existing_insights_bytes"] is not None
    assert attributes["existing_insights_sha256"] is not None
    # ...and its raw output, before the parser touched it.
    assert attributes["raw_output"] == raw
    assert attributes["parsed_count"] == 1
    assert attributes["empty_reason"] is None
    # Which ties it to the rows it produced.
    insight = _extracted_insight(db_path)
    assert attributes["insight_ids"] == [insight["insight_id"]]
    assert insight["extractor_span_id"] == extract["span_id"]

    # §8's cap rule: the corpus itself is NOT on the span (it grows without
    # bound), only its length and hash.
    assert "existing_insights" not in attributes


# ===========================================================================
# ACCEPTANCE CRITERION 7
# An agent can answer "which turns support insight X, and which contradict
# it" from documented SQL (§15) without scraping the UI.  §15 [DR36][DR54]
# ===========================================================================

def test_ac7_support_and_contradiction_are_answerable_from_the_documented_sql(
    observed_wec, monkeypatch, tmp_path
):
    """Two REAL runs, then the shipped §15 recipes, executed.

    Run 1: the teacher completes the task, the student does not — a
    `missing-in-student` divergence, and the insight extracted from it.
    Run 2: BOTH passes complete the task — the same command, in the student
    pass, with no divergence of that kind. That is the run where the rule would
    have fired wrongly, and it is what "contradicts" means.

    [DR54] is why this is two real runs rather than a paragraph: revision 1's
    promotion query counted 3 supporters where the answer was 1, and
    parse-checking is retired as a standard of evidence.
    """
    ctx, sink, db_path = observed_wec
    supporting_key, supporting = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    contradicting_key, contradicting = _run(
        ctx, monkeypatch, _agreeing_agents(), message="finish my dishes task"
    )
    sink.close()

    assert supporting.run_id != contradicting.run_id
    insight = _extracted_insight(db_path)
    assert insight["run_id"] == supporting.run_id

    support = _rows(db_path, _SQL_SUPPORT, {"insight_id": insight["insight_id"]})
    assert [r["run_id"] for r in support] == [supporting.run_id]
    assert support[0]["turn_key"] == supporting_key
    assert support[0]["kind"] == alignment.KIND_MISSING_IN_STUDENT
    assert support[0]["command_name"] == "complete_task"
    assert support[0]["material"] == 1

    contradiction = _rows(
        db_path, _SQL_CONTRADICTION, {"insight_id": insight["insight_id"]}
    )
    assert [r["run_id"] for r in contradiction] == [contradicting.run_id]
    assert contradiction[0]["turn_key"] == contradicting_key
    assert contradiction[0]["command_name"] == "complete_task"

    # The two sets are disjoint and neither is the whole corpus: the run the
    # rule came from does not contradict itself, and the run that agreed does
    # not support it.
    assert {r["run_id"] for r in support}.isdisjoint(
        {r["run_id"] for r in contradiction}
    )
    # And the join that makes it all work is the identity ruling's: the
    # contradiction recipe reaches the spans through `s.trace_id = r.turn_key`.
    student_span = _one(
        db_path,
        "SELECT * FROM spans WHERE trace_id=? AND distillation_pass='student' "
        "AND name=? AND command_name=?",
        (contradicting_key, tracing.SPAN_COMMAND_EXECUTE, "complete_task"),
    )
    assert student_span["trace_id"] == contradicting_key


def test_ac7_a_run_level_divergence_is_stored_with_no_command_to_key_on(
    observed_wec, monkeypatch
):
    """§7.3 step 6 and [DR54]'s three-valued-logic trap, at the storage end.

    Two passes that took equivalent actions and returned different answers
    produce a `level='run'` record with NO command. That NULL is why the §15
    action contradiction recipe carries `cited.command_name IS NOT NULL`:
    without it `s.command_name = cited.command_name` is NULL under
    three-valued logic, `EXISTS` is false for every run, and the query returns
    zero rows with no error — indistinguishable from "no contradictions
    found". It is also why §15 ships an eighth, outcome-keyed recipe.

    Note the shipped semantics this pins: the record is STORED but does not
    set `exec_diverged`, because that column has always meant "the two passes
    executed different actions" and letting model wording flip it would change
    how often extraction fires.
    """
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], answer="the laundry is done"),
            lambda cs: _PassAgent(cs, ["list_tasks"], answer="I could not tell"),
        ],
    )
    sink.close()

    run_level = _one(
        db_path, "SELECT * FROM distillation_divergences WHERE level=?", ("run",)
    )
    assert run_level["kind"] == alignment.KIND_DIFFERENT_ANSWER_SAME_ACTIONS
    assert run_level["command_name"] is None
    assert run_level["command_key"] is None
    assert run_level["left_step_key"] is None and run_level["right_step_key"] is None
    detail = json.loads(run_level["detail_json"])
    assert detail["left_answer"] == "the laundry is done"
    assert detail["right_answer"] == "I could not tell"
    # Both passes ended in the same state, so the different wording is not
    # material — which is exactly the judgement the fingerprint buys.
    assert run_level["material"] == 0

    run = _one(db_path, "SELECT * FROM distillation_runs")
    assert run["exec_diverged"] == 0
    assert run["extractor_empty"] == 0
    assert _rows(db_path, "SELECT * FROM distillation_insights") == []

    # The action-level records it coexists with are all equivalent — that is
    # the precondition for the run-level record existing at all.
    action = _rows(
        db_path, "SELECT * FROM distillation_divergences WHERE level=?", ("action",)
    )
    assert action and {r["kind"] for r in action} == {alignment.KIND_IDENTICAL}


# ===========================================================================
# ACCEPTANCE CRITERION 8
# (Design-level only — fix-sb8.11 is not built.) Everything §14 says replay
# needs is actually STORED: the user message, the entry fingerprint, and the
# insight set live at the time.  §14 [DR34][DR41][DR45]
# ===========================================================================

def test_ac8_everything_section_14_says_replay_needs_is_stored(
    observed_wec, monkeypatch, tmp_path
):
    """One assertion per row of §14's table. No replay is attempted: the
    producer is fix-sb8.11 and does not exist. What is checkable today is
    whether the evidence it would need survived the turn that produced it."""
    ctx, sink, db_path = observed_wec
    message = "finish my laundry task"
    # The corpus the WEC holds at agent init, which is what the run's prompts
    # actually saw ([DR34]). `_initialize_agent_functionality` loads it once
    # and never reloads it; this harness scripts the agent, so the same
    # attribute is set directly on the real context.
    corpus = "1. Complete the task the user named\n"
    ctx._execution_insights = corpus
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message=message,
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    sink.close()

    run = _one(db_path, "SELECT * FROM distillation_runs")
    passes = {p["pass_label"]: p for p in _rows(db_path, "SELECT * FROM distillation_passes")}

    # Row 1 — the user message: the replay's input.
    assert run["user_message"] == message

    # Row 2 — the entry fingerprint: the gate. A replay whose reconstructed
    # entry fingerprint differs from the original student pass's is
    # non-comparable and returns no verdict.
    assert passes["student"]["entry_fingerprint"]
    assert passes["teacher"]["entry_fingerprint"]
    # [DR47] splits it: the prompt fingerprint and its bound are the second
    # half of the world gate.
    assert passes["student"]["entry_prompt_fingerprint"]
    assert passes["student"]["history_bound"] is not None

    # Row 3 — the pass's PROMPT INPUTS, and [DR45]'s decisive negative: this
    # column is not restorable state, and it says so about itself.
    entry_inputs = json.loads(passes["student"]["entry_inputs_json"])
    assert entry_inputs["raw_user_message"] == message
    assert "history_tail" in entry_inputs
    assert "insight_set" in entry_inputs
    diagnostic = entry_inputs["context_snapshot"]
    assert diagnostic["diagnostic_only"] is True, (
        "a column that reads as restorable state and is not is worse than no "
        "column ([DR45])"
    )

    # Row 4 — the insight set live at the time. The corpora are loaded once at
    # agent init and never reloaded, so the file's current contents are NOT
    # what the run used.
    insight_set = json.loads(run["insight_set_json"])
    assert set(insight_set) == {"planning", "execution"}
    assert insight_set["execution"] == {
        "bytes": len(corpus.encode()),
        "sha256": hashlib.sha256(corpus.encode()).hexdigest(),
    }
    # No planning corpus was loaded, and that NULL is itself the record: a
    # replay that injected one would be testing a different prompt.
    assert insight_set["planning"] is None
    # The body is never stored — §8 makes the same call for the extract span,
    # and the corpus is reconstructible from the ledger.
    assert corpus not in run["insight_set_json"]

    # Row 5 — the cited divergences: the assertion is "did THESE divergences
    # disappear", not "did anything change".
    insight = _extracted_insight(db_path)
    cited = _rows(
        db_path,
        "SELECT d.* FROM distillation_divergences d "
        "JOIN distillation_insight_citations c ON c.divergence_id=d.divergence_id "
        "WHERE c.insight_id=?",
        (insight["insight_id"],),
    )
    assert cited
    # A replay's assertion keys on (command_key, kind), so both must be there.
    assert all(r["kind"] for r in cited)
    assert any(r["command_key"] for r in cited)

    # And the four gates §14 lists have honest columns to read: this run is
    # comparable, and `isolation_verified` is NULL rather than 1 ([DR48]), so
    # a replay refuses rather than returning a verdict it cannot support.
    assert run["comparable"] == 1
    assert run["isolation_verified"] is None
    assert run["replay_of"] is None and run["replay_trace_id"] is None


def test_ac8_the_replay_namespace_is_addressable_and_has_no_turns_row(
    observed_wec, monkeypatch
):
    """§18's required case (e), as far as the tree allows.

    `[DR41]`'s producer is fix-sb8.11 and is not built — nothing in the tree
    sets `current_replay_trace_id`, so no `~replay` span is emitted by
    production code today. What IS assertable, and is the half that constrains
    every later reader, is the storage contract: a `<turn_key>~replay.<n>`
    trace is a real addressable trace with NO `turns` row, `~` cannot occur in
    a minted turn key, and the replay spans go with their base turn when the
    turn is erased (§3.5 row 4).
    """
    ctx, sink, db_path = observed_wec
    turn_key, _result = _run(ctx, monkeypatch, _diverging_agents())
    replay_trace = f"{turn_key}~replay.1"
    # Written through the real sink, in the shape [DR41]'s producer will write.
    sink.emit_span(
        tracing.Span(
            span_id=uuid.uuid4().hex,
            trace_id=replay_trace,
            name=tracing.SPAN_DISTILL_REPLAY,
            kind=tracing.KIND_INTERNAL,
            channel_id=ctx.observability_channel_id,
            start_ns=1,
            end_ns=2,
            status=tracing.STATUS_OK,
            attributes={"replay_of": _result.run_id, "replay_trace_id": replay_trace},
            distillation_pass="student-replay",
        )
    )
    sink.close()

    assert _rows(db_path, "SELECT * FROM turns WHERE turn_key LIKE '%~%'") == []
    assert "~" not in turn_key
    replay_spans = _rows(
        db_path, "SELECT * FROM spans WHERE trace_id=?", (replay_trace,)
    )
    assert len(replay_spans) == 1
    assert replay_spans[0]["distillation_pass"] == "student-replay"
    # The base turn's own trace is untouched by the replay's arrival — [DR4]:
    # a replay written into the original trace would mutate cited evidence as
    # a SUCCESSFUL upsert.
    assert _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_key,))


# ===========================================================================
# ACCEPTANCE CRITERION 9
# Retention cannot prune a trace that an accepted insight cites.
# §10.3 [DR25][DR26][DR43][DR52]
# ===========================================================================

def _accept(db_path: str, insight_id_value: str) -> None:
    """Record a `supported` verdict, on the direct writable handle [DR46]
    exempts for the verdict route (an HTTP handler, never a turn thread).
    fix-sb8.9 owns the route; the row shape is §9's."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO distillation_verdicts "
            "(verdict_id, insight_id, verdict, note, actor, superseded, created_at) "
            "VALUES (?,?,?,?,?,0,?)",
            (
                uuid.uuid4().hex,
                insight_id_value,
                "supported",
                "the rule held on review",
                "human",
                "2026-08-28T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_ac9_retention_cannot_prune_a_trace_an_accepted_insight_cites(
    observed_wec, monkeypatch, tmp_path
):
    """The pinned run's whole evidence chain survives a prune that deletes
    everything else in the DB, and the chain still resolves afterwards.

    `[DR43]` restates the criterion honestly — "retention **in builds carrying
    the pin predicate**" — so this is the assertion that the predicate is
    carried and that it rides INSIDE the victim subquery.
    """
    ctx, sink, db_path = observed_wec
    cited_key, cited_run = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    other_key, other_run = _run(
        ctx, monkeypatch, _param_diverging_agents(), message="finish my dishes task"
    )
    sink.close()

    insight = _extracted_insight(db_path)
    assert insight["run_id"] == cited_run.run_id
    _accept(db_path, insight["insight_id"])

    store = ObservabilityStore(db_path)
    assert store.pin_distillation_run(cited_run.run_id) is True
    before = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (cited_run.run_id,)
    )
    assert before["pinned"] == 1
    assert before["pinned_at"] and before["pinned_span_count"] > 0

    # Everything is past a zero-day horizon, so retention wants all of it.
    store.prune(retention_days=0, max_bytes=10**9)

    # The pinned trace is whole.
    survivors = _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (cited_key,))
    assert survivors
    # And the unpinned neighbour is gone, so the prune genuinely ran.
    assert _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (other_key,)) == []

    # The chain still resolves end to end, which is the actual criterion.
    forward = _rows(db_path, _SQL_FORWARD, {"insight_id": insight["insight_id"]})
    assert forward
    span_ids = {row["span_id"] for row in survivors}
    for row in forward:
        assert row["left_span_id"] in span_ids or row["right_span_id"] in span_ids

    after = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (cited_run.run_id,)
    )
    assert after["evidence_pruned"] == 0
    assert store.distillation_evidence_shortfall(cited_run.run_id)["incomplete"] is False

    # The unpinned run keeps its CONCLUSIONS, marked, and loses the bulk —
    # so the UI can say "the trace behind this is gone" rather than render an
    # empty diff ([DR52]).
    stripped = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (other_run.run_id,)
    )
    assert stripped["evidence_pruned"] == 1
    assert _rows(
        db_path, "SELECT * FROM distillation_divergences WHERE run_id=?",
        (other_run.run_id,),
    ) == []
    assert all(
        p["entry_inputs_json"] is None
        for p in _rows(
            db_path, "SELECT * FROM distillation_passes WHERE run_id=?",
            (other_run.run_id,),
        )
    )
    # The pinned run's own passes kept theirs.
    assert all(
        p["entry_inputs_json"] is not None
        for p in _rows(
            db_path, "SELECT * FROM distillation_passes WHERE run_id=?",
            (cited_run.run_id,),
        )
    )


# ===========================================================================
# PART 2 — the five blocking findings the Phase-0 gate caught (§22)
#
# Each of these was a real defect in revision 1 of the design that would have
# shipped green, so each gets a test that fails if it comes back.
# ===========================================================================

# --- blocking finding 2: the fingerprint hashed a heap address -------------

_FINGERPRINT_PROBE = '''\
import sys
import fastworkflow
from fastworkflow.distillation import state_fingerprint
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

fastworkflow.init({})
ctx = WorkflowExecutionContext(run_as_agent=True)
workflow = fastworkflow.Workflow.create(sys.argv[1], workflow_id_str=sys.argv[2])
ctx.bind_app_workflow(workflow)
ctx.push_active_workflow(workflow)
if len(sys.argv) > 3:
    workflow.context["mutated"] = sys.argv[3]
print(state_fingerprint(ctx))
'''


def _fingerprint_in_a_fresh_process(tmp_path, state_root, *extra) -> str:
    probe = tmp_path / "fingerprint_probe.py"
    probe.write_text(_FINGERPRINT_PROBE, encoding="utf-8")
    env = dict(os.environ)
    env["FASTWORKFLOW_STATE_ROOT"] = str(state_root)
    completed = subprocess.run(
        [sys.executable, str(probe), _TODO_WORKFLOW, "sb8-14-cross-process", *extra],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout.strip().splitlines()[-1]


def test_blocking_finding_2_the_fingerprint_is_byte_equal_across_two_processes(
    tmp_path
):
    """§19's mandated `[DR47]` case, and the reason the `gate-fails` verdict
    was correct.

    Revision 1's `_digest` passed `default=str`, so a live object in a context
    dict was hashed as `<todo_list.TodoList object at 0x7f...>` — a heap
    address. The fingerprint was therefore different on every run of the same
    state, no replay could ever pass its own entry gate, and nothing would
    have raised. Two SEPARATE processes over the same reconstructed state is
    the only shape of this test that catches it.
    """
    first = _fingerprint_in_a_fresh_process(tmp_path, tmp_path / "state-a")
    second = _fingerprint_in_a_fresh_process(tmp_path, tmp_path / "state-b")
    assert first == second, "the fingerprint is not a function of the state alone"
    assert len(first) == 32

    # And it still bites: a state that genuinely differs hashes differently.
    mutated = _fingerprint_in_a_fresh_process(
        tmp_path, tmp_path / "state-c", "a different world"
    )
    assert mutated != first


def test_blocking_finding_2_two_passes_from_the_same_state_agree(
    observed_wec, monkeypatch
):
    """The in-process consequence, end to end: identical entry states produce
    equal entry fingerprints and a `comparable = 1` run."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(ctx, monkeypatch, _agreeing_agents())
    sink.close()

    passes = _rows(db_path, "SELECT * FROM distillation_passes ORDER BY seq")
    assert passes[0]["entry_fingerprint"] == passes[1]["entry_fingerprint"]
    assert passes[0]["exit_fingerprint"] == passes[1]["exit_fingerprint"]
    run = _one(db_path, "SELECT * FROM distillation_runs")
    assert (run["comparable"], run["comparable_reason"]) == (1, None)
    # `restore_ok_pre_student` is the same claim measured at the restore site:
    # the rollback actually landed, rather than aliasing the live dict.
    assert run["restore_ok_pre_student"] == 1


def test_blocking_finding_2_the_fingerprint_survives_a_live_object_in_context(
    observed_wec, monkeypatch
):
    """A live application object in a context dict — the exact case `default=
    str` rendered as an address — must not disturb the comparison."""
    from tests.todo_list_workflow.application.todo_list import TodoList

    ctx, sink, db_path = observed_wec
    ctx.get_active_workflow().context["live_handle"] = TodoList(7, "a live object")
    before = state_fingerprint(ctx)
    # A second, equal-but-distinct object at a different address.
    ctx.get_active_workflow().context["live_handle"] = TodoList(7, "a live object")
    assert state_fingerprint(ctx) == before

    _turn_key, _result = _run(ctx, monkeypatch, _agreeing_agents())
    sink.close()
    run = _one(db_path, "SELECT * FROM distillation_runs")
    assert run["comparable"] == 1


# --- blocking finding 3: erasure did not reach the six new tables ----------

def _distill_row_counts(db_path: str) -> dict[str, int]:
    return {
        table: _rows(db_path, f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
        for table in _DISTILL_TABLES
    }


def _seed_a_full_distillation_corpus(ctx, sink, monkeypatch, tmp_path):
    """One real run that populates all six tables, plus a replay trace."""
    turn_key, result = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    sink.emit_span(
        tracing.Span(
            span_id=uuid.uuid4().hex,
            trace_id=f"{turn_key}~replay.1",
            name=tracing.SPAN_DISTILL_REPLAY,
            kind=tracing.KIND_INTERNAL,
            channel_id=ctx.observability_channel_id,
            start_ns=1,
            end_ns=2,
            status=tracing.STATUS_OK,
            attributes={"replay_of": result.run_id},
            distillation_pass="student-replay",
        )
    )
    sink.close()
    return turn_key, result


def test_blocking_finding_3_forget_channel_reaches_all_six_new_tables(
    observed_wec, monkeypatch, tmp_path
):
    """[DR44]. Revision 1 certified `[R21]` erasure as "No change, verified"
    on the strength of the `spans` `channel_id` arm alone; `forget_channel`
    and `clear_conversations` are hardcoded table lists, and the new tables
    hold verbatim user content — the user message, the context dicts in
    `entry_inputs_json`, user-supplied parameter values in `param_diff_json`,
    the insight text. §19 is explicit that this asserts row COUNTS, not the
    absence of an exception.
    """
    ctx, sink, db_path = observed_wec
    turn_key, result = _seed_a_full_distillation_corpus(ctx, sink, monkeypatch, tmp_path)
    _accept(db_path, _extracted_insight(db_path)["insight_id"])

    before = _distill_row_counts(db_path)
    assert all(count > 0 for count in before.values()), before
    # The user's own words really are in there — that is why erasure must
    # reach these tables and not only `spans`.
    assert "list my tasks" in _one(
        db_path, "SELECT * FROM distillation_runs"
    )["user_message"]

    store = ObservabilityStore(db_path)
    store.forget_channel("sb8-14-channel")

    after = _distill_row_counts(db_path)
    assert after == {table: 0 for table in _DISTILL_TABLES}, after
    assert _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_key,)) == []
    assert _rows(
        db_path, "SELECT * FROM spans WHERE trace_id LIKE ?", (f"{turn_key}~%",)
    ) == [], "replay spans survived a channel erasure"
    assert result.run_id


def test_blocking_finding_3_clear_conversations_reaches_all_six_new_tables(
    observed_wec, monkeypatch, tmp_path
):
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _seed_a_full_distillation_corpus(
        ctx, sink, monkeypatch, tmp_path
    )
    _accept(db_path, _extracted_insight(db_path)["insight_id"])
    assert all(count > 0 for count in _distill_row_counts(db_path).values())

    ObservabilityStore(db_path).clear_conversations()

    assert _distill_row_counts(db_path) == {table: 0 for table in _DISTILL_TABLES}
    assert _rows(db_path, "SELECT * FROM spans") == []


def test_blocking_finding_3_erasure_leaves_no_orphan_citation_behind(
    observed_wec, monkeypatch, tmp_path
):
    """[DR44]'s ordering clause. There are no foreign keys, so nothing
    cascades; a wrong delete order leaves citations pointing at divergences
    that are gone, which §15's contradiction recipe would then read as real
    evidence."""
    ctx, sink, db_path = observed_wec
    _seed_a_full_distillation_corpus(ctx, sink, monkeypatch, tmp_path)
    ObservabilityStore(db_path).forget_channel("sb8-14-channel")
    orphans = _rows(
        db_path,
        "SELECT c.* FROM distillation_insight_citations c "
        "LEFT JOIN distillation_divergences d ON d.divergence_id=c.divergence_id "
        "WHERE d.divergence_id IS NULL",
    )
    assert orphans == []


# --- blocking finding 7: materiality could never be 0 ----------------------

def test_blocking_finding_7_two_passes_taking_identical_actions_are_not_material(
    observed_wec, monkeypatch
):
    """§18's required case (b).

    Revision 1's `[DR20]` read a fingerprint that included the conversation
    history, and every pass appends its own LLM-generated summary before it
    exits — so the two exit fingerprints could never be equal and `material`
    could never be 0. `[DR47]` splits the fingerprint; materiality reads the
    history-free projection.
    """
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(ctx, monkeypatch, _agreeing_agents())
    sink.close()

    passes = _rows(db_path, "SELECT * FROM distillation_passes ORDER BY seq")
    assert passes[0]["exit_fingerprint"] == passes[1]["exit_fingerprint"]
    records = _rows(db_path, "SELECT * FROM distillation_divergences")
    assert records
    assert all(r["material"] == 0 for r in records)
    assert _one(db_path, "SELECT * FROM distillation_runs")["material_divergences"] == 0


def test_blocking_finding_7_materiality_reads_the_exit_state_not_the_kind(
    observed_wec, monkeypatch
):
    """The stronger form, which the `identical` shortcut would hide: a real
    divergence whose two passes ended in the same state is material = 0, and
    the same divergence with a teacher-only state change is material = 1."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(ctx, monkeypatch, _param_diverging_agents())
    sink.close()

    changed = _one(
        db_path,
        "SELECT * FROM distillation_divergences WHERE kind=?",
        (alignment.KIND_PARAM_VALUE_ONLY,),
    )
    assert changed["material"] == 0, (
        "the same end state reached by a different path is not a mistake"
    )
    passes = _rows(db_path, "SELECT * FROM distillation_passes ORDER BY seq")
    assert passes[0]["exit_fingerprint"] == passes[1]["exit_fingerprint"]


def test_blocking_finding_7_a_divergence_that_changed_the_world_is_material(
    observed_wec, monkeypatch
):
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(ctx, monkeypatch, _diverging_agents())
    sink.close()

    passes = _rows(db_path, "SELECT * FROM distillation_passes ORDER BY seq")
    assert passes[0]["exit_fingerprint"] != passes[1]["exit_fingerprint"]
    missing = _one(
        db_path,
        "SELECT * FROM distillation_divergences WHERE kind=?",
        (alignment.KIND_MISSING_IN_STUDENT,),
    )
    assert missing["material"] == 1
    assert _one(db_path, "SELECT * FROM distillation_runs")["material_divergences"] == 1


# --- blocking finding 8: failed commands collapsed into one key ------------

def test_blocking_finding_8_two_different_failed_commands_get_two_command_keys(
    observed_wec, monkeypatch
):
    """§18's required case (c), and the exact case §7.1 markets as the win.

    Every failed command leaves `spans.command_name` NULL (the early returns
    in `command_executor._invoke_command_impl` never set it), so under
    revision 1 every failure in a pass hashed to the same `command_key` and
    two unrelated failures compared `identical`. `[DR50]` falls back to the
    normalized `raw_command`.
    """
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        [
            lambda cs: _PassAgent(
                cs, ["list_tasks"], failed=["delete task 9", "archive task 3"]
            ),
            lambda cs: _PassAgent(cs, ["list_tasks"]),
        ],
    )
    sink.close()

    # The two failures are two separate unmatched steps, not one.
    failures = _rows(
        db_path,
        "SELECT * FROM distillation_divergences WHERE kind=? ORDER BY align_index",
        (alignment.KIND_MISSING_IN_STUDENT,),
    )
    assert len(failures) == 2, "two different failed commands collapsed into one step"
    assert failures[0]["command_key"] != failures[1]["command_key"]
    assert failures[0]["left_step_key"] != failures[1]["left_step_key"]
    # The displayed name is the `raw:` form, because there is no real name to
    # show — and the spans themselves still carry NULL, which is the condition
    # that made the keys degenerate in the first place.
    assert sorted(r["command_name"] for r in failures) == [
        "raw:archive task 3",
        "raw:delete task 9",
    ]

    # The keys are the documented `raw:` fallback, so they are reproducible.
    spans = {s["span_id"]: s for s in _rows(db_path, "SELECT * FROM spans")}
    for row in failures:
        span = spans[row["left_span_id"]]
        assert span["command_name"] is None
        raw = json.loads(span["attributes"])["raw_command"]
        assert row["command_key"] == alignment.command_key(
            alignment.make_command_step(
                span["span_id"], command_name=None, context=span["context"],
                raw_command=raw,
            )
        )


def test_blocking_finding_8_an_unrelated_failure_in_each_pass_is_not_identical(
    observed_wec, monkeypatch
):
    """The consequence that mattered: a teacher failure and an unrelated
    student failure must read as two orphans, never as a matched pair."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], failed=["delete task 9"]),
            lambda cs: _PassAgent(cs, ["list_tasks"], failed=["archive task 3"]),
        ],
    )
    sink.close()

    kinds = [
        r["kind"]
        for r in _rows(
            db_path,
            "SELECT * FROM distillation_divergences WHERE level=? ORDER BY align_index",
            (alignment.LEVEL_ACTION,),
        )
    ]
    assert alignment.KIND_MISSING_IN_STUDENT in kinds
    assert alignment.KIND_EXTRA_IN_STUDENT in kinds
    # One `identical` for the command that really did match, and no more.
    assert kinds.count(alignment.KIND_IDENTICAL) == 1
    assert _one(db_path, "SELECT * FROM distillation_runs")["exec_diverged"] == 1


# --- blocking finding 6/[DR52]: the all-pinned eviction batch --------------

def _emit_trace(sink, trace_id, channel_id, count, first_ns):
    for index in range(count):
        sink.emit_span(
            tracing.Span(
                span_id=uuid.uuid4().hex,
                trace_id=trace_id,
                name=tracing.SPAN_COMMAND_EXECUTE,
                kind=tracing.KIND_TOOL,
                channel_id=channel_id,
                start_ns=first_ns + index,
                end_ns=first_ns + index + 1,
                status=tracing.STATUS_OK,
                attributes={"raw_command": f"x{index}"},
            )
        )


def test_blocking_finding_6_an_all_pinned_batch_does_not_halt_eviction(
    tmp_path, monkeypatch
):
    """[DR52]: the pin predicate rides INSIDE the victim subquery.

    On the outer DELETE the first batch whose oldest spans are all pinned
    deletes nothing, `rowcount` is 0, and the `break` abandons eviction with
    the DB over its cap and evictable spans still present — no error, no
    marker, no log line. `observability_store.py`'s size-cap arm is where that
    trap lives, so this drives that arm specifically: the retention horizon is
    set a century back so arm 1 does nothing and the size cap does all the
    work.
    """
    from fastworkflow import observability_store as store_module

    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    # The PINNED trace holds the four OLDEST spans, so it is what an
    # oldest-first eviction reaches for first.
    _emit_trace(sink, "pinned-trace", "c", 4, first_ns=1_000)
    _emit_trace(sink, "unpinned-trace", "c", 6, first_ns=9_000)
    sink.emit_distillation_record(
        "run",
        {
            "run_id": "run-pinned",
            "turn_key": "pinned-trace",
            "user_message": "keep me",
            "comparable": 1,
            "run_json": "{}",
        },
    )
    sink.emit_distillation_record(
        "pass",
        {
            "run_id": "run-pinned",
            "pass_label": "teacher",
            "role": "teacher",
            "seq": 0,
            "trace_id": "pinned-trace",
        },
    )
    sink.close()

    store = ObservabilityStore(db_path)
    assert store.pin_distillation_run("run-pinned") is True

    # A batch smaller than the pinned run, so the first batch is all pinned.
    monkeypatch.setattr(store_module, "_PRUNE_BATCH_ROWS", 2)
    # Over the cap, but not so far over that the pinned set breaches its own
    # ceiling: `[DR52]`'s cap-wins-over-the-pin arm is a different mechanism,
    # and letting it fire here would evict the pinned trace for the right
    # reason and prove nothing about the batch loop.
    store.prune(retention_days=36_500, max_bytes=100_000)

    remaining = _rows(db_path, "SELECT trace_id, COUNT(*) AS n FROM spans GROUP BY trace_id")
    by_trace = {row["trace_id"]: row["n"] for row in remaining}
    assert by_trace.get("unpinned-trace", 0) == 0, (
        "eviction halted on an all-pinned batch and left evictable spans behind"
    )
    assert by_trace.get("pinned-trace") == 4, "the pin did not hold"


# ===========================================================================
# PART 3 — the silent-failure sites (§19)
#
# None of these raises. Each is a place where the whole feature becomes a
# no-op, or a shipped query quietly stops returning the right rows, with no
# exception, no log line and no dropped-span counter.
# ===========================================================================

def test_the_emit_span_field_copy_carries_the_pass_label_to_the_db(
    observed_wec, monkeypatch
):
    """§9 producer item 9 — "the single edit most likely to be missed".

    `SQLiteTraceSink.emit_span` rebuilds the `Span` field by field before
    queueing it. Omitting `distillation_pass` there makes the entire feature a
    silent no-op: no exception, no log line, no dropped-span counter, and a
    waterfall that looks exactly as it does today.
    """
    ctx, sink, db_path = observed_wec
    turn_key, _result = _run(ctx, monkeypatch, _diverging_agents())
    sink.close()

    labelled = _rows(
        db_path,
        "SELECT * FROM spans WHERE trace_id=? AND distillation_pass IS NOT NULL",
        (turn_key,),
    )
    assert labelled, "no span reached the DB with a pass label"
    # No extractor ran on this configuration, so the labels are exactly the
    # two passes.
    assert {row["distillation_pass"] for row in labelled} == {"teacher", "student"}
    # The column is a real column, outside the attributes blob — which is what
    # makes `WHERE distillation_pass = ?` documented SQL and keeps the label
    # outside the 16 KiB attribute cap and the Redactor's substring pass
    # ([DR23]).
    for row in labelled:
        assert "distillation_pass" not in json.loads(row["attributes"])


def test_a_span_close_cannot_relabel_a_span(observed_wec):
    """§9 producer item 8: `distillation_pass` is deliberately absent from the
    `ON CONFLICT DO UPDATE` set. Write-once at open is the correct semantics
    and it keeps the close path from being able to relabel a span."""
    ctx, sink, db_path = observed_wec
    span_id = uuid.uuid4().hex
    for label, status in (("teacher", tracing.STATUS_OPEN), ("student", tracing.STATUS_OK)):
        sink.emit_span(
            tracing.Span(
                span_id=span_id,
                trace_id="relabel-trace",
                name=tracing.SPAN_COMMAND_EXECUTE,
                kind=tracing.KIND_TOOL,
                start_ns=1,
                end_ns=None if status == tracing.STATUS_OPEN else 2,
                status=status,
                distillation_pass=label,
            )
        )
    sink.close()

    row = _one(db_path, "SELECT * FROM spans WHERE span_id=?", (span_id,))
    assert row["distillation_pass"] == "teacher"
    # The close's own fields DID land, so this is write-once on one column
    # rather than a dropped write.
    assert row["status"] == tracing.STATUS_OK
    assert row["end_ns"] == 2


def test_list_turns_filtered_by_command_still_finds_a_distilled_turn(
    observed_wec, monkeypatch
):
    """The `list_turns(command_name=…)` regression §19 names.

    The filter is `turn_key IN (SELECT trace_id FROM spans WHERE
    command_name=?)`. It only works because `[DR1]` keeps `trace_id ==
    turn_key` for both passes; a derived per-pass trace id would have dropped
    every distilled turn out of the command-filtered rail, silently.
    """
    from fastworkflow.turn import TurnStatus

    ctx, sink, db_path = observed_wec
    turn_key, _result = _run(ctx, monkeypatch, _diverging_agents())
    # The turn record the WEC writes at finalize, emitted the same way.
    response = fastworkflow.CommandResponse(response="done")
    output = fastworkflow.CommandOutput(command_name="x", command_response=response)
    turn_output = fastworkflow.TurnOutput(
        turn_key=turn_key,
        status=TurnStatus.COMPLETED,
        answer="done",
        command_outputs=[output],
    )
    sink.emit_turn_record(
        fastworkflow.TurnResult(
            turn_output=turn_output,
            channel_id=ctx.observability_channel_id,
            user_message="list my tasks",
        )
    )
    sink.close()

    store = ObservabilityStore(db_path)
    # A command only the TEACHER pass ran still finds the turn.
    found = store.list_turns(command_name="complete_task")
    assert [row["turn_key"] for row in found] == [turn_key]
    # And so does one both passes ran.
    assert [row["turn_key"] for row in store.list_turns(command_name="list_tasks")] == [
        turn_key
    ]
    assert store.list_turns(command_name="never_ran") == []


def test_a_pre_distillation_db_degrades_instead_of_failing(tmp_path):
    """[DR29]: a DB written before this feature has none of the six tables and
    no `distillation_pass` column. Opening it must report the feature absent,
    not raise — the viewer has to be able to open a post-mortem snapshot it
    does not own."""
    old_db = str(tmp_path / "old.sqlite3")
    conn = sqlite3.connect(old_db)
    try:
        # The pre-distillation shape: the tables a v1 build created, minus
        # everything this design adds.
        conn.execute(
            "CREATE TABLE spans (span_id TEXT PRIMARY KEY, trace_id TEXT, "
            "name TEXT, kind TEXT, parent_span_id TEXT, channel_id TEXT, "
            "command_name TEXT, context TEXT, start_ns INTEGER, end_ns INTEGER, "
            "status TEXT, attributes TEXT)"
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()

    reader = ReadOnlyObservabilityStore(old_db)
    assert reader.has_feature(FEATURE_DISTILLATION_V1) is False
    # It never writes, so the file is still the old shape afterwards.
    columns = {
        row["name"]
        for row in _rows(old_db, "SELECT name FROM pragma_table_info('spans')")
    }
    assert "distillation_pass" not in columns

    # A writable store upgrades it in place, and then the feature is present.
    writable = ObservabilityStore(old_db)
    assert writable.has_feature(FEATURE_DISTILLATION_V1) is True
    for table in _DISTILL_TABLES:
        assert _rows(old_db, f"SELECT COUNT(*) AS n FROM {table}")[0]["n"] == 0


def test_deterministic_span_id_is_byte_identical_without_a_pass_label():
    """[DR11]'s prerequisite. Folding `pass_label` into the digest
    unconditionally would silently move every existing span id — including the
    `fw.turn` root and both `fw.ask_user` ids — and the six existing
    assertions in `tests/test_tracing_phase1.py` are what would have caught it
    only if the default stayed byte-identical."""
    key = "20260828T120000.000000Z-abcdef012345"
    assert tracing.deterministic_span_id(key, tracing.SPAN_ASK_USER, 0) == (
        tracing.deterministic_span_id(key, tracing.SPAN_ASK_USER, 0, pass_label=None)
    )
    # And a label genuinely separates them, which is the collision [DR10]
    # would otherwise arm.
    with_label = tracing.deterministic_span_id(
        key, tracing.SPAN_ASK_USER, 0, pass_label="teacher"
    )
    assert with_label != tracing.deterministic_span_id(key, tracing.SPAN_ASK_USER, 0)
    assert with_label != tracing.deterministic_span_id(
        key, tracing.SPAN_ASK_USER, 0, pass_label="student"
    )
    # The pass wrapper's id is a pure function of the turn key and the label,
    # which is what makes it computable at the ask_user close site [DR51].
    assert tracing.distill_pass_span_id(key, "teacher") == (
        tracing.distill_pass_span_id(key, "teacher")
    )
    assert tracing.distill_pass_span_id(key, "teacher") != (
        tracing.distill_pass_span_id(key, "student")
    )


def test_a_store_write_failure_during_distillation_does_not_reach_the_turn(
    observed_wec, monkeypatch
):
    """[DR46]'s whole point, and §19's write-path case.

    Live-run writes go through the sink's record queue, so they inherit the
    writer thread, the busy-retry and the breaker. A direct
    `store._connect(timeout=30.0)` from `distill_message` would have raised a
    lock contention inside `_execute_message` — the same uncaught shape §3.3
    documents as fatal to the turn.
    """
    ctx, sink, db_path = observed_wec

    def explode(self, conn, kind, payload, redactor):
        raise RuntimeError("the distillation table is wedged")

    monkeypatch.setattr(ObservabilityStore, "upsert_distillation_row", explode)
    turn_key, result = _run(ctx, monkeypatch, _diverging_agents())
    # The turn itself completed and returned the teacher's answer.
    assert result.command_output is not None
    assert result.run_id
    sink.close()

    # Not one distillation row landed...
    assert _distill_row_counts(db_path) == {table: 0 for table in _DISTILL_TABLES}
    # ...but the spans did, so the failure was contained to the record queue
    # and is visible where failures are supposed to be visible.
    assert _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_key,))
    health = ObservabilityStore(db_path).writer_health()
    assert health and health.get("write_errors", 0) > 0


def test_distillation_with_no_sink_emits_nothing_and_pays_nothing(
    bare_wec, monkeypatch, tmp_path
):
    """The cost of the feature when observability is off, following
    `TestDisabledObservabilityDspyCost`'s precedent.

    With `NoOpTraceSink` there are no spans to align and no table to write, so
    distillation must keep detecting divergence exactly as it did before this
    epic — otherwise turning observability off silently disables insight
    extraction for everyone.
    """
    ctx = bare_wec
    assert tracing.get_sink(ctx) is None
    _turn_key, result = _run(ctx, monkeypatch, _diverging_agents())

    # The run still happened and still has an id...
    assert result.run_id
    assert result.command_output is not None
    # ...and no DB anywhere in tmp_path was created by it.
    assert list(tmp_path.glob("**/*.sqlite3")) == []

    # The no-sink fallback still detects divergence: `compare_trajectories` /
    # `compare_planning_traces` are kept for exactly this configuration.
    session = DistillationSession.__new__(DistillationSession)
    diverged, summary = session.compare_trajectories(
        [_action("list_tasks"), _action("complete_task")], [_action("list_tasks")]
    )
    assert diverged is True and "complete_task" in summary


# ===========================================================================
# PART 4 — three normative statements that were once unimplemented
#
# `[DR35]`'s `replayable` carve-out, §10.3's pin-at-write-time and `[DR51]`'s
# `fw.ask_user` reparenting each shipped as a strict xfail naming the defect.
# All three producers have since landed, so the strict xfails became XPASSes
# and are now ordinary passing tests: the assertions are unchanged, only the
# markers are gone. They stay in this part of the file because what they guard
# is a normative statement of the design rather than a criterion of the epic.
# ===========================================================================

def test_a_run_level_answer_divergence_is_marked_not_replayable(
    observed_wec, monkeypatch
):
    """[DR35]: `replayable` is a pre-filter that saves the cost of starting a
    replay the observation gate would certainly fail. A
    `different-answer-same-actions` record is response-content evidence, which
    the uncaptured application state cannot support, so it is flagged 0."""
    ctx, sink, db_path = observed_wec
    _turn_key, _result = _run(
        ctx,
        monkeypatch,
        [
            lambda cs: _PassAgent(cs, ["list_tasks"], answer="the laundry is done"),
            lambda cs: _PassAgent(cs, ["list_tasks"], answer="I could not tell"),
        ],
    )
    sink.close()

    run_level = _one(
        db_path, "SELECT * FROM distillation_divergences WHERE level=?", ("run",)
    )
    assert run_level["kind"] == alignment.KIND_DIFFERENT_ANSWER_SAME_ACTIONS
    assert run_level["replayable"] == 0
    # Every other kind stays replayable, so this is a carve-out and not a
    # blanket demotion.
    action = _rows(
        db_path, "SELECT * FROM distillation_divergences WHERE level=?", ("action",)
    )
    assert action and all(r["replayable"] == 1 for r in action)


def test_a_run_that_produced_an_insight_is_pinned_at_write_time(
    observed_wec, monkeypatch, tmp_path
):
    """§10.3, rows 2 and 4 of the pin-class table.

    Without this, `[DR43]`'s restatement — "retention in builds carrying the
    pin predicate cannot prune a trace an accepted insight cites" — holds only
    for runs somebody remembered to pin.
    """
    ctx, sink, db_path = observed_wec
    _turn_key, extracted = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    _turn_key2, agreed = _run(
        ctx, monkeypatch, _agreeing_agents(), message="finish my dishes task"
    )
    sink.close()

    assert _rows(db_path, "SELECT * FROM distillation_insights")
    produced = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (extracted.run_id,)
    )
    assert produced["pinned"] == 1, "a run with an unadjudicated insight is pinned"

    # The no-divergence class is the contradiction pool for every future rule,
    # pinned for FW_OBS_DISTILL_NEGATIVE_PIN_DAYS and then released by the
    # sweep — which is already built and would have nothing to release.
    no_divergence = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (agreed.run_id,)
    )
    assert (no_divergence["planning_diverged"], no_divergence["exec_diverged"]) == (0, 0)
    assert no_divergence["pinned"] == 1


# --- [DR51] / §9 producer items 6 and 12: the ask_user parenting case ------

def _asking_agents():
    """Both passes ask the user one question, so the turn contains the span
    §3.6 says `[DR8]`'s assertion is only satisfiable for because of [DR51]."""

    def ask(chat_session):
        chat_session.append_ask_user_entry("which task did you mean?")
        chat_session.complete_ask_user_entry("the laundry one")

    return [
        lambda cs: _PassAgent(cs, ["list_tasks"], mutate=ask),
        lambda cs: _PassAgent(cs, ["list_tasks"], mutate=ask),
    ]


def test_an_ask_user_inside_a_pass_is_labelled_with_that_pass(
    observed_wec, monkeypatch
):
    """§9 producer item 6, the column half: `fw.ask_user` opens through
    `start_span`, so it picks up the ambient pass label for free."""
    ctx, sink, db_path = observed_wec
    turn_key, _result = _run(ctx, monkeypatch, _asking_agents())
    sink.close()

    asked = _rows(
        db_path,
        "SELECT * FROM spans WHERE trace_id=? AND name=? ORDER BY start_ns",
        (turn_key, tracing.SPAN_ASK_USER),
    )
    assert [row["distillation_pass"] for row in asked] == ["teacher", "student"]
    # Distinct ids, so the two passes' questions are two spans and not one
    # upsert overwriting the other ([DR11]'s collision).
    assert asked[0]["span_id"] != asked[1]["span_id"]
    assert all(row["status"] == tracing.STATUS_OK for row in asked)


def test_an_ask_user_inside_a_pass_parents_onto_its_own_pass_span(
    observed_wec, monkeypatch
):
    """[DR51], and the reason `fw.distill.pass` has a deterministic id at all.

    `_close_ask_user_span` rebuilds its `Span` from pure functions of the turn
    key rather than holding it, so it can only parent onto something it can
    *compute* — which is exactly what `distill_pass_span_id` is for.
    """
    ctx, sink, db_path = observed_wec
    turn_key, _result = _run(ctx, monkeypatch, _asking_agents())
    sink.close()

    asked = _rows(
        db_path,
        "SELECT * FROM spans WHERE trace_id=? AND name=? ORDER BY start_ns",
        (turn_key, tracing.SPAN_ASK_USER),
    )
    assert len(asked) == 2
    root = tracing.root_span_id(turn_key)
    for row in asked:
        expected = tracing.distill_pass_span_id(turn_key, row["distillation_pass"])
        assert row["parent_span_id"] != root
        assert row["parent_span_id"] == expected


# ===========================================================================
# PART 5 — the producer-side findings of the post-backbone audit
#
# Four adversarial auditors read the landed backbone against this design. The
# tests below are the regression pins for the producer-side findings they
# raised: §10.3's missing pin writer (AC9), the read barrier that could not see
# a loss, the record-loss counters nothing consulted, `[DR31]`'s colliding
# insight ids, and a pass that raised being stored as agreement.
# ===========================================================================

def _raising_student():
    """A student pass whose agent call raises, as an unset or refusing LM does."""

    class _Raiser:
        def __init__(self, _chat_session):
            self.current_trajectory: dict = {}

        def __call__(self, **_kwargs):
            raise RuntimeError("the student agent's LM refused")

    return [
        lambda cs: _PassAgent(cs, ["list_tasks", "complete_task"]),
        _Raiser,
    ]


# --- §10.3: the pin has a producer -----------------------------------------

def test_a_run_that_produced_an_insight_survives_the_startup_prune(
    observed_wec, monkeypatch, tmp_path
):
    """AC9 without an operator in the loop.

    `prune()` is not something somebody runs: `SQLiteTraceSink.__init__` calls
    it opportunistically at every startup, on a 30-day default. So the pin has
    to be written by the producer or the evidence behind every extracted rule
    is deleted at the horizon — the insight row and the markdown line survive,
    pointing at nothing. Nothing calls `pin_distillation_run` here.
    """
    ctx, sink, db_path = observed_wec
    cited_key, cited_run = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    # A neighbour that diverged and produced nothing: in none of §10.3's five
    # classes, so it is not pinned and the prune must take it.
    other_key, other_run = _run(
        ctx, monkeypatch, _param_diverging_agents(), message="finish my dishes task"
    )
    sink.close()

    pinned = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (cited_run.run_id,)
    )
    assert pinned["pinned"] == 1
    assert pinned["pinned_at"]
    # [DR43]: the count the shortfall API measures a later loss against.
    assert pinned["pinned_span_count"] > 0

    insight = _extracted_insight(db_path)
    assert insight["run_id"] == cited_run.run_id
    cited_divergences = {
        row["divergence_id"]
        for row in _rows(
            db_path,
            "SELECT * FROM distillation_insight_citations WHERE insight_id=?",
            (insight["insight_id"],),
        )
    }
    assert cited_divergences

    store = ObservabilityStore(db_path)
    store.prune(retention_days=0, max_bytes=10**9)

    # The whole chain the insight rests on is intact: both passes' spans, the
    # divergence rows, and the citations that join them.
    assert _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (cited_key,))
    for label in ("teacher", "student"):
        assert _pass_spans(db_path, cited_key, label)
    survived = {
        row["divergence_id"]
        for row in _rows(
            db_path,
            "SELECT * FROM distillation_divergences WHERE run_id=?",
            (cited_run.run_id,),
        )
    }
    assert cited_divergences <= survived
    assert _rows(db_path, _SQL_FORWARD, {"insight_id": insight["insight_id"]})
    after = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (cited_run.run_id,)
    )
    assert after["evidence_pruned"] == 0
    assert store.distillation_evidence_shortfall(cited_run.run_id)["incomplete"] is False

    # ...and the prune genuinely ran: the unpinned neighbour is gone.
    assert _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (other_key,)) == []
    assert (
        _rows(
            db_path,
            "SELECT * FROM distillation_divergences WHERE run_id=?",
            (other_run.run_id,),
        )
        == []
    )
    stripped = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (other_run.run_id,)
    )
    assert (stripped["pinned"], stripped["evidence_pruned"]) == (0, 1)


def test_the_no_divergence_pin_arrives_in_the_shape_the_release_sweep_expects(
    observed_wec, monkeypatch
):
    """§10.3's fourth class: pinned for 90 days, then released and prunable.

    `_release_distillation_pins` selects `pinned=1 AND planning_diverged=0 AND
    exec_diverged=0` ordered on `COALESCE(pinned_at, started_at)`, so a pin
    written without `pinned_at` would either never be released or be released
    by accident. The sweep runs before both prune arms ([DR52]), so a run
    released this pass is prunable this pass — asserted here in one call.
    """
    ctx, sink, db_path = observed_wec
    turn_key, agreed = _run(ctx, monkeypatch, _agreeing_agents())
    sink.close()

    run = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (agreed.run_id,)
    )
    assert (run["planning_diverged"], run["exec_diverged"]) == (0, 0)
    assert run["pinned"] == 1 and run["pinned_at"]
    assert _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_key,))

    store = ObservabilityStore(db_path)
    # negative_pin_days=0 is "the 90 days have passed".
    store.prune(retention_days=0, max_bytes=10**9, negative_pin_days=0)

    released = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (agreed.run_id,)
    )
    assert released["pinned"] == 0, "the contradiction pool would grow forever"
    assert _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_key,)) == []


# --- [DR49]: the barrier and the counters that prove it ---------------------

@pytest.mark.parametrize(
    "counter", ["spans_dropped", "records_dropped", "write_errors"]
)
def test_evidence_the_writer_lost_after_the_passes_is_never_read_as_agreement(
    observed_wec, counter
):
    """The hole the per-pass `spans_dropped_delta` cannot see.

    The delta is sampled at each pass's own exit, so a batch the writer
    discards while the closing barrier drains moves no per-pass counter — and
    `records_dropped` / `write_errors` moved no counter anyone read at all. A
    run whose divergence rows cite discarded spans, or whose divergence rows
    were themselves discarded, must not be recorded `comparable = 1`: every
    §15 recipe filters on that column, so it would be first-class evidence.
    """
    ctx, sink, _db_path = observed_wec
    ds = DistillationSession(ctx)
    ds._passes = {
        "teacher": {"seq": 0, "entry_fingerprint": "abc", "spans_dropped_delta": 0},
        "student": {"seq": 1, "entry_fingerprint": "abc", "spans_dropped_delta": 0},
    }
    ds.snapshot_writer_counters()
    assert ds.comparability_fields()["comparable"] == 1

    # Exactly what the sink itself does on `queue.Full`, on a rolled-back
    # batch, and on a record write it swallowed.
    sink._count(counter)

    assert ds.check_writer_loss() == {counter: 1}
    verdict = ds.comparability_fields()
    assert (verdict["comparable"], verdict["comparable_reason"]) == (
        0,
        "evidence-incomplete",
    )


def test_a_distillation_record_the_writer_dropped_makes_the_run_incomplete(
    observed_wec, monkeypatch
):
    """Finding: distillation record loss was entirely unmonitored.

    `_apply_distillation` counts a malformed record in `write_errors` and keeps
    the rest of its batch — deliberately, so one bad record cannot roll back a
    good batch — and nothing in the distillation path ever read that counter.
    The run then claimed `comparable = 1` while `material_divergences` counted
    rows that are not in the table and its citations pointed at nothing.
    """
    ctx, sink, db_path = observed_wec
    real_upsert = ObservabilityStore.upsert_distillation_row

    def lose_the_divergences(self, conn, kind, payload, redactor):
        if kind == "divergence":
            # The shape `_apply_distillation` swallows: an IntegrityError from
            # a NOT NULL column is counted, never raised ([DR46]).
            raise sqlite3.IntegrityError("NOT NULL constraint failed")
        return real_upsert(self, conn, kind, payload, redactor)

    monkeypatch.setattr(
        ObservabilityStore, "upsert_distillation_row", lose_the_divergences
    )
    _turn_key, result = _run(ctx, monkeypatch, _diverging_agents())
    sink.close()

    assert _rows(db_path, "SELECT * FROM distillation_divergences") == []
    run = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (result.run_id,)
    )
    assert (run["comparable"], run["comparable_reason"]) == (0, "evidence-incomplete")
    # Which counter moved is recorded too, so the verdict can be diagnosed.
    assert json.loads(run["run_json"])["writer_loss"]["write_errors"] > 0
    # And a run that cannot show its evidence is not pinned as if it could.
    assert run["pinned"] == 0


# --- [DR31]: one insight id, one row, one line ------------------------------

def test_two_insights_that_normalize_alike_are_one_row_and_one_line(
    observed_wec, monkeypatch, tmp_path
):
    """`insight_id` is seeded with the NORMALIZED text (§13.1), so two bullets
    of one extractor call that differ only in case or a trailing period mint
    the same id. Writing both put two lines in the markdown behind one ledger
    row: the `[DR31]` marker then resolved to one row from two lines, and the
    surviving row's `file_entry_number` named the later line only.

    The fix cannot be a richer seed — §13.1 keeps the file entry number out of
    the id on purpose, "a hand edit or a renumber would orphan every
    reference" — so it is deduplication.
    """
    ctx, sink, db_path = observed_wec
    insights_dir = tmp_path / "Insights" / "todo_list_workflow"
    _turn_key, result = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        execution_raw=(
            "- Never call update_task before verifying the task exists\n"
            "- never call update_task before verifying the task exists."
        ),
        insights_dir=insights_dir,
    )
    sink.close()

    # The two bullets are one insight, so they are one row...
    rows = _rows(db_path, "SELECT * FROM distillation_insights")
    assert len(rows) == 1
    insight = rows[0]
    assert insight["file_entry_number"] == 1
    assert result.execution_insights_extracted == 1

    # ...and one line, carrying one marker that resolves back to that row.
    written = Path(insight["insight_file"]).read_text(encoding="utf-8")
    assert written.count(insight["insight_id"]) == 1
    entries = [
        line for line in written.splitlines() if re.match(r"^\d+\.\s", line)
    ]
    assert len(entries) == 1
    assert insight["insight_id"] in entries[0]
    assert insight["insight_id"] == insight_id(
        insight["run_id"], insight["kind"], insight["text"]
    )


# --- A pass that raised is not a pass that agreed ---------------------------

_SQL_AGREEMENT_RATE = """\
SELECT COUNT(*) AS runs,
       SUM(CASE WHEN r.exec_diverged = 1 THEN 1 ELSE 0 END) AS diverged
FROM distillation_runs r
WHERE r.comparable = 1 AND r.replay_of IS NULL
"""


def test_a_student_pass_that_raised_is_not_stored_as_agreement(
    observed_wec, monkeypatch
):
    """§15's aggregates read `comparable = 1, exec_diverged = 0` as "the
    student matched the teacher". On the student-failure path no comparison
    ever runs — `align_and_record` is never reached, so there are no divergence
    rows and both NOT NULL diverged columns keep their DDL default of 0 — and
    the run was still written `comparable = 1` with `completed_at` set. A crash
    counted as agreement is a corpus statistic that lies by commission.

    §18's `fix-sb8.2` note sets the precedent with `comparable_reason =
    'teacher-raised'`; the student-raised case gets its own recorded state.
    """
    ctx, sink, db_path = observed_wec
    _turn_key, result = _run(ctx, monkeypatch, _raising_student())
    sink.close()

    run = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (result.run_id,)
    )
    assert (run["comparable"], run["comparable_reason"]) == (0, "student-raised")
    # The failure is queryable rather than buried in run_json...
    assert json.loads(run["run_json"])["status"] == "student-failed"
    assert json.loads(run["run_json"])["error_type"] == "RuntimeError"
    # ...and nothing about it reads as a comparison that happened.
    assert _rows(db_path, "SELECT * FROM distillation_divergences") == []
    assert run["pinned"] == 0

    # The §15 shape now excludes it instead of counting it as a clean run.
    rate = _one(db_path, _SQL_AGREEMENT_RATE)
    assert rate["runs"] == 0

    # The turn still returned the teacher's answer: this is a recording fix.
    assert result.command_output is not None


def test_an_extractor_that_raised_does_not_leave_the_run_looking_completed(
    observed_wec, monkeypatch
):
    """The same shape by a different route: `extract_planning_insights` /
    `extract_insights` are called with no guard, so a rate-limited or timed-out
    extractor propagates past the `completion.update(...)` that sets
    `exec_diverged` and `completed_at`. The divergence rows were already
    written, so the run row said `exec_diverged = 0`, `completed_at IS NULL`
    and `run_json.status = 'completed'` — a run that never finished comparing,
    described as a finished comparison.
    """
    ctx, sink, db_path = observed_wec
    queued = list(_diverging_agents())
    _script_llm_boundaries(
        monkeypatch, ctx, lambda chat_session, **_kw: queued.pop(0)(chat_session)
    )

    def raise_from_the_extractor(*_args, **_kwargs):
        raise RuntimeError("the extractor LM timed out")

    monkeypatch.setattr(
        DistillationSession, "extract_insights", raise_from_the_extractor
    )
    monkeypatch.setattr(
        DistillationSession, "extract_planning_insights", lambda *a, **k: []
    )
    ctx._begin_turn("list my tasks")
    with pytest.raises(RuntimeError):
        distill_message(ctx, "list my tasks")
    sink.close()

    run = _one(db_path, "SELECT * FROM distillation_runs")
    assert json.loads(run["run_json"])["status"] == "raised"
    assert json.loads(run["run_json"])["error_type"] == "RuntimeError"
    assert run["completed_at"] is None
    assert (run["comparable"], run["comparable_reason"]) == (0, "evidence-incomplete")
    assert run["pinned"] == 0
    # The divergence rows it did write are kept and quarantined, never deleted
    # (§6.2 obligation 5).
    assert _rows(
        db_path, "SELECT * FROM distillation_divergences WHERE run_id=?",
        (run["run_id"],),
    )


# --- fix-sb8.18: the pin costs the turn thread nothing ----------------------

def test_pin_fields_reads_no_database_on_the_turn_thread(observed_wec, monkeypatch):
    """`fix-sb8.18`: `pinned_span_count` is resolved by the WRITER.

    The producer used to open its own read-write sqlite3 connection and run
    `SELECT COUNT(*) FROM spans` on the turn thread at the completion of every
    pinned run — bounded and contained, but new user-visible latency stacked on
    top of the [DR49] barrier waits. The writer thread is already inside a
    transaction on an open connection when it writes the row, so the count
    belongs there.
    """
    ctx, _sink, _db_path = observed_wec
    ds = DistillationSession(ctx)
    ds.planning_insights_extracted = 1

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "pin_fields opened a database connection on the turn thread"
        )

    monkeypatch.setattr(ObservabilityStore, "_connect", forbidden)
    fields = ds.pin_fields(
        {"comparable": 1, "planning_diverged": 1, "exec_diverged": 0}
    )

    assert fields["pinned"] == 1
    assert fields["pinned_span_count"] == COUNT_LIVE_SPANS


def test_the_writer_resolves_the_live_span_count_sentinel(tmp_path):
    """The other half: the sentinel has to become a real count, over the same
    trace set `pin_distillation_run` uses, or [DR43]'s shortfall check measures
    a later loss against nothing."""
    db_path = str(tmp_path / "observability.sqlite3")
    store = ObservabilityStore(db_path)
    redactor = Redactor()
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_distillation_row(
            conn,
            "run",
            {
                "run_id": "r1",
                "turn_key": "t1",
                "channel_id": "c",
                "user_message": "add a todo",
                "comparable": 1,
                "run_json": json.dumps({"run_id": "r1"}),
            },
            redactor,
        )
        store.upsert_distillation_row(
            conn,
            "pass",
            {
                "run_id": "r1",
                "pass_label": "teacher",
                "role": "teacher",
                "seq": 0,
                "trace_id": "t1",
            },
            redactor,
        )
        store.upsert_span_rows(
            conn,
            [
                tracing.Span(
                    span_id=f"s{i}",
                    trace_id="t1",
                    name="fw.turn",
                    start_ns=i,
                    status="ok",
                )
                for i in range(4)
            ]
            # A neighbouring trace, to prove the count is scoped to the run.
            + [
                tracing.Span(
                    span_id="other",
                    trace_id="t-other",
                    name="fw.turn",
                    start_ns=9,
                    status="ok",
                )
            ],
            redactor,
        )
        store.upsert_distillation_row(
            conn,
            "run",
            {
                "run_id": "r1",
                "turn_key": "t1",
                "channel_id": "c",
                "user_message": "add a todo",
                "run_json": json.dumps({"run_id": "r1"}),
                "comparable": 1,
                "pinned": 1,
                "pinned_span_count": COUNT_LIVE_SPANS,
            },
            redactor,
        )
        conn.commit()

    with store._connect() as conn:
        row = conn.execute(
            "SELECT pinned, pinned_span_count FROM distillation_runs WHERE run_id='r1'"
        ).fetchone()
    assert (row["pinned"], row["pinned_span_count"]) == (1, 4)


# --- fix-sb8.16: write ordering, not just loss detection --------------------

def _seed_insight_row(store, conn, redactor, run_id="r1", turn_key="t1"):
    store.upsert_distillation_row(
        conn,
        "run",
        {
            "run_id": run_id,
            "turn_key": turn_key,
            "channel_id": "c",
            "user_message": "add a todo",
            "run_json": json.dumps({"run_id": run_id}),
            "comparable": 1,
        },
        redactor,
    )
    store.upsert_distillation_row(
        conn,
        "insight",
        {
            "insight_id": "ins-1",
            "run_id": run_id,
            "kind": "execution",
            "text": "prefer the id form",
            "text_hash": "deadbeef",
            "created_at": "2026-08-29T00:00:00+00:00",
        },
        redactor,
    )


def _seed_divergence_row(store, conn, redactor, divergence_id, run_id="r1"):
    store.upsert_distillation_row(
        conn,
        "divergence",
        {
            "divergence_id": divergence_id,
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


def test_a_citation_whose_divergence_never_landed_is_refused(tmp_path):
    """`fix-sb8.16`: the ordering, not only the detection.

    The audit round taught the run to NOTICE a dropped divergence
    (`comparable = 0` / `writer_loss`), but citations were still written
    without confirming their divergence rows had landed. §13.2's provenance
    chain resolves insight -> citation -> divergence -> span; an orphaned
    citation is a chain that §15's recipes join through and read as real
    evidence, and it survives the run whose `comparable = 0` flagged it.
    """
    db_path = str(tmp_path / "observability.sqlite3")
    store = ObservabilityStore(db_path)
    redactor = Redactor()
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _seed_insight_row(store, conn, redactor)
        with pytest.raises(OrphanedCitation):
            store.upsert_distillation_row(
                conn,
                "citation",
                {"insight_id": "ins-1", "divergence_id": "never-written"},
                redactor,
            )
        conn.commit()

    assert _rows(db_path, "SELECT * FROM distillation_insight_citations") == []


def test_a_citation_lands_once_its_divergence_row_is_there(tmp_path):
    """The ordinary path is untouched, and a re-emission is not a loss."""
    db_path = str(tmp_path / "observability.sqlite3")
    store = ObservabilityStore(db_path)
    redactor = Redactor()
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _seed_insight_row(store, conn, redactor)
        _seed_divergence_row(store, conn, redactor, "d0")
        for _ in range(2):
            # The second write is the idempotent retry a requeued record
            # produces: a no-op, and specifically NOT an orphan report.
            store.upsert_distillation_row(
                conn,
                "citation",
                {"insight_id": "ins-1", "divergence_id": "d0"},
                redactor,
            )
        conn.commit()

    assert len(_rows(db_path, "SELECT * FROM distillation_insight_citations")) == 1


def test_the_writer_charges_a_suppressed_citation_to_write_errors(tmp_path):
    """The suppression must be visible: it moves a counter `check_writer_loss`
    reads, so the run is completed `evidence-incomplete` rather than quietly
    missing one edge of its provenance."""
    db_path = str(tmp_path / "observability.sqlite3")
    sink = SQLiteTraceSink(db_path)
    try:
        with sink.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _seed_insight_row(sink.store, conn, Redactor())
            conn.commit()
        sink.emit_distillation_record(
            "citation", {"insight_id": "ins-1", "divergence_id": "never-written"}
        )
        assert sink.flush(timeout=10.0) is True
        health = sink.store.writer_health() or {}
    finally:
        sink.close()

    assert int(health.get("write_errors") or 0) >= 1
    assert "citation" in (health.get("last_error") or "")
    assert _rows(db_path, "SELECT * FROM distillation_insight_citations") == []


def test_pass_rows_are_rewritten_even_when_the_rollup_is_empty(
    observed_wec, monkeypatch
):
    """`fix-sb8.16`: `finalize_pass_metrics` returned before re-emitting when
    the rollup came back empty, so a run row could end up with no pass rows at
    all — no models, no fingerprints, no entry inputs, and every §15 recipe
    joins through that table. An empty rollup is precisely when the pass-exit
    write is most likely to be the one that was lost, so it is the case that
    most needs the second attempt."""
    ctx, _sink, _db_path = observed_wec
    ctx._turn_key = "t-finalize"
    ds = DistillationSession(ctx)
    ds._pass_rows = {
        "teacher": {"seq": 0, "role": "teacher"},
        "student": {"seq": 1, "role": "student"},
    }
    monkeypatch.setattr(DistillationSession, "_llm_rollup", lambda self: {})
    emitted = []
    monkeypatch.setattr(
        tracing,
        "emit_distillation_record",
        lambda chat, kind, payload: emitted.append((kind, payload["pass_label"])),
    )

    assert ds.finalize_pass_metrics() is None

    assert sorted(emitted) == [("pass", "student"), ("pass", "teacher")]


# ===========================================================================
# ACCEPTANCE CRITERION 8 — `fix-sb8.11`
# A stored run can be replayed student-only against the stored teacher trace
# to test whether a candidate insight removes the divergence.  §14 [DR34]
# [DR35] [DR41] [DR45]
# ===========================================================================

def _isolation_verified(db_path: str, run_id: str) -> None:
    """Stand in for `fix-35m.3`.

    `[DR48]` makes replay a causal claim, so it declines while
    `isolation_verified` is not 1 — and nothing writes 1 until EXP-013 lands.
    A test that could not set it could only ever assert the decline.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE distillation_runs SET isolation_verified=1 WHERE run_id=?",
            (run_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _restore_student_entry(ctx):
    """Put the world back where the stored student pass entered it.

    Not a fixture convenience: `distill_message` restores the TEACHER's exit
    state on any divergence (§6.2), so straight after a diverging run the live
    world is by construction NOT the one the student entered — and
    `[DR45]`'s entry gate correctly refuses. A real replay runs later, against
    a world that has come back round to that state; this is the test's way of
    being that later.
    """
    ctx.get_active_workflow().context.pop("teacher_wrote", None)
    ctx._turn_key = None


def _replay(ctx, monkeypatch, agent, **kwargs):
    """Drive one replay with a scripted student pass."""
    from fastworkflow import distillation_replay

    _script_llm_boundaries(
        monkeypatch, ctx, lambda chat_session, **_kw: agent(chat_session)
    )
    return distillation_replay.replay_run(ctx, **kwargs)


def test_ac8_a_replay_that_removes_the_cited_divergence_validates_the_insight(
    observed_wec, monkeypatch, tmp_path
):
    """The bridge from insight to VERIFIER.

    The origin run's student skipped `complete_task`, which is the divergence
    the insight cites. The replay's student — the same small model, with the
    candidate rule injected — takes the teacher's actions instead. The stored
    teacher trace is the fixture; "the cited divergence is gone" is the
    assertion; and the answer becomes a `distillation_verdicts` row written by
    `actor = 'replay'`, which is what a promotion decision reads.
    """
    ctx, sink, db_path = observed_wec
    turn_key, origin = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    # The barrier the aligner takes is per-trace; the replay reads the table.
    assert sink.flush(timeout=10.0)
    _isolation_verified(db_path, origin.run_id)

    _restore_student_entry(ctx)
    outcome = _replay(
        ctx,
        monkeypatch,
        lambda cs: _PassAgent(cs, ["list_tasks", "complete_task"]),
        origin_run_id=origin.run_id,
        execution_insights="- Always complete the task the user asked about",
    )
    sink.flush(timeout=10.0)

    assert outcome.status == "verdict", outcome.reason
    assert outcome.divergence_removed is True
    assert outcome.verdict == "supported"

    # [DR41]: a replay writes into a DERIVED trace and never into the turn's.
    assert outcome.trace_id == f"{turn_key}~replay.1"
    replay_spans = _rows(
        db_path, "SELECT * FROM spans WHERE trace_id=?", (outcome.trace_id,)
    )
    assert replay_spans, "the replay wrote no spans"
    assert _rows(db_path, "SELECT * FROM turns WHERE instr(turn_key,'~')>0") == []
    # And the original evidence is untouched — the pin exists to protect it.
    origin_spans = _rows(
        db_path, "SELECT span_id FROM spans WHERE trace_id=?", (turn_key,)
    )
    assert not any("~" in row["span_id"] for row in origin_spans)

    verdicts = _rows(
        db_path,
        "SELECT * FROM distillation_verdicts WHERE actor='replay' ORDER BY created_at",
    )
    assert verdicts, "the replay recorded no verdict"
    assert verdicts[-1]["verdict"] == "supported"
    assert verdicts[-1]["replay_run_id"] == outcome.run_id

    run_row = _one(
        db_path, "SELECT * FROM distillation_runs WHERE run_id=?", (outcome.run_id,)
    )
    assert run_row["replay_of"] == origin.run_id
    assert run_row["replay_trace_id"] == outcome.trace_id


def test_ac8_a_replay_that_reproduces_the_divergence_says_so(
    observed_wec, monkeypatch, tmp_path
):
    """The other answer, which is the one that matters: an insight that does
    not remove the divergence is decoration, and the corpus has to be able to
    say that rather than only being able to confirm."""
    ctx, sink, db_path = observed_wec
    _turn_key, origin = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    assert sink.flush(timeout=10.0)
    _isolation_verified(db_path, origin.run_id)
    _restore_student_entry(ctx)

    outcome = _replay(
        ctx,
        monkeypatch,
        # Unchanged behaviour: the student still skips the command.
        lambda cs: _PassAgent(cs, ["list_tasks"]),
        origin_run_id=origin.run_id,
        execution_insights="- A rule that changes nothing",
    )
    sink.flush(timeout=10.0)

    assert outcome.status == "verdict", outcome.reason
    assert outcome.divergence_removed is False
    assert outcome.verdict == "not-supported-by-cited-evidence"
    assert outcome.remaining, "the cited divergence should still be listed"

    live = _rows(
        db_path,
        "SELECT verdict, actor FROM distillation_verdicts WHERE superseded=0",
    )
    assert live and live[0]["verdict"] == "not-supported-by-cited-evidence"


def test_ac8_replay_declines_while_isolation_is_unverified(
    observed_wec, monkeypatch, tmp_path
):
    """`[DR48]`: promotion is a causal claim, so replay refuses until
    `fix-35m.3` can certify that the passes did not share application state.
    Declining is the honest answer; a verdict here would be a claim the state
    capture does not support."""
    ctx, sink, db_path = observed_wec
    _turn_key, origin = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    assert sink.flush(timeout=10.0)
    _restore_student_entry(ctx)

    outcome = _replay(
        ctx,
        monkeypatch,
        lambda cs: _PassAgent(cs, ["list_tasks", "complete_task"]),
        origin_run_id=origin.run_id,
    )

    assert outcome.status == "not-replayable"
    assert "isolation" in outcome.reason
    # No verdict, and no replay run row: a declined replay costs nothing.
    assert _rows(db_path, "SELECT * FROM distillation_verdicts") == []
    assert _rows(db_path, "SELECT * FROM distillation_runs WHERE replay_of IS NOT NULL") == []


def test_ac8_replay_declines_when_the_world_drifted_at_entry(
    observed_wec, monkeypatch, tmp_path
):
    """`[DR45]`: replay is world-GATED, not world-reconstructing.

    `entry_inputs_json` holds prompt inputs, explicitly not restorable state,
    so the only honest precondition is that the live world still matches the
    one the stored student pass entered. Here the teacher's own write is left
    in place, so it does not.
    """
    ctx, sink, db_path = observed_wec
    _turn_key, origin = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    assert sink.flush(timeout=10.0)
    _isolation_verified(db_path, origin.run_id)
    _restore_student_entry(ctx)
    ctx.get_active_workflow().context["someone_else_wrote"] = "since"

    outcome = _replay(
        ctx,
        monkeypatch,
        lambda cs: _PassAgent(cs, ["list_tasks", "complete_task"]),
        origin_run_id=origin.run_id,
    )

    assert outcome.status == "not-replayable"
    assert outcome.reason == "world drifted at entry"


def test_ac8_replay_declines_on_a_non_comparable_origin(
    observed_wec, monkeypatch, tmp_path
):
    """§6.2's contract: a non-comparable run's divergences are unusable by
    definition, so there is nothing to test the disappearance of."""
    ctx, sink, db_path = observed_wec
    _turn_key, origin = _run(
        ctx,
        monkeypatch,
        _diverging_agents(),
        message="finish my laundry task",
        execution_raw="- Always complete the task the user asked about",
        insights_dir=tmp_path / "Insights" / "todo_list_workflow",
    )
    assert sink.flush(timeout=10.0)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE distillation_runs SET comparable=0, "
            "comparable_reason='fingerprint-differs', isolation_verified=1 "
            "WHERE run_id=?",
            (origin.run_id,),
        )
        conn.commit()
    finally:
        conn.close()
    _restore_student_entry(ctx)

    outcome = _replay(
        ctx,
        monkeypatch,
        lambda cs: _PassAgent(cs, ["list_tasks", "complete_task"]),
        origin_run_id=origin.run_id,
    )

    assert outcome.status == "not-replayable"
    assert "not comparable" in outcome.reason


def test_the_per_step_observation_gate_only_compares_re_run_steps():
    """`[DR45]`'s gate has to catch a world that moved WITHOUT calling every
    behaviour change a drift — the behaviour change is the thing under test."""
    from fastworkflow.distillation_replay import observation_drift

    stored = [("list_tasks", {}, "one task"), ("complete_task", {"id": 1}, "ok")]

    # Identical re-run: no drift.
    assert observation_drift(stored, list(stored)) is None

    # The agent chose differently at step 1 — that is the insight working, and
    # the gate must stay silent about it.
    assert (
        observation_drift(
            stored, [("list_tasks", {}, "one task"), ("add_todo", {}, "added")]
        )
        is None
    )

    # The SAME call returned something else: the world moved underneath.
    assert (
        observation_drift(
            stored, [("list_tasks", {}, "four tasks"), ("complete_task", {"id": 1}, "ok")]
        )
        == 0
    )
