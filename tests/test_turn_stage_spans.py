"""The turn's stage and part spans (bd fix-qjne.34, steps 2 to 7 of the span-tree revision).

A plan-path turn records its stages as children of the turn root, in order: ``fw.split``, ``fw.resolve`` (with one
``fw.resolve.part`` per part), ``fw.review`` when the turn is reviewed (the review's own records under it), ``fw.run``
(one ``fw.run.part`` per part: its phases, steps, plan and commands under it) and ``fw.answer`` (the summary's model
call). Each has a deterministic id, so a turn that waits for the user (here: the review) keeps ONE span per stage,
reopened when the reply resumes it, in this process or in another one built from the saved session (``segments`` 2).
The direct ("/") path has one part and no split or review: its decision is written under ``fw.resolve >
fw.resolve.part`` and its command under ``fw.run > fw.run.part``.

Real library copy with its trained heads and the real agent with a scripted model (the ``gated`` fixture of the review
gate tests); hello_world's trained workflow for the "/" path. No test calls a live model (a ``DummyLM`` answers the
planner and the summary where their spans are the subject).
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.observability import tracing
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests import span_tree
from tests.test_approval_binding import (  # noqa: F401
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, FIND, JOKE, _split_into, gated, remote_decides_emma_and_joke, no_draft_extraction  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from fastworkflow.workflow_agent import build_query_with_next_steps as _REAL_PLANNER   # the fixtures stub it

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _stages(sink) -> list[str]:
    """The names of the turn root's stage children, in start order."""
    (turn,) = span_tree.tree(sink)
    return [child.name for child in turn.children if child.name in tracing.STAGE_SPAN_NAMES]


def test_a_reviewed_two_part_turn_has_split_resolve_review_run_in_order(gated, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    (review,) = span_tree.named(gated.sink, tracing.SPAN_REVIEW)
    assert review.status == "awaiting_user" and review.attributes["outcome"] == "open"
    gated.script.steps = [("finish", {})] * 2
    out = gated.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED
    sink = gated.sink
    assert _stages(sink) == [tracing.SPAN_SPLIT, tracing.SPAN_RESOLVE, tracing.SPAN_REVIEW, tracing.SPAN_RUN,
                             tracing.SPAN_ANSWER], span_tree.render(span_tree.tree(sink))
    # one span per stage across the wait: the review was opened, waited on, and reopened by the reply
    (review,) = span_tree.named(sink, tracing.SPAN_REVIEW)
    assert review.attributes["segments"] == 2 and review.attributes["resumed"] is True
    assert review.attributes["outcome"] == "reviewed" and review.status == tracing.STATUS_OK
    (resolve,) = span_tree.named(sink, tracing.SPAN_RESOLVE)
    assert resolve.attributes["parts"] == 2 and resolve.attributes["segments"] == 1
    parts = span_tree.children_of(resolve, sink, tracing.SPAN_RESOLVE_PART)
    assert [p.attributes["part"] for p in parts] == [1, 2]
    assert [p.attributes["request"] for p in parts] == [EMMA, JOKE]
    for span in span_tree.spans(sink):
        if span.name in tracing.STAGE_SPAN_NAMES:
            assert span.parent_span_id == tracing.root_span_id(span.trace_id)
            assert span.span_id == tracing.stage_span_id(span.trace_id, span.name)
    # the question the review waited on is a child of the review stage
    (question,) = span_tree.named(sink, tracing.SPAN_QUESTION)
    assert question.attributes["kind"] == "review"
    assert span_tree.path_of(question, sink) == [tracing.SPAN_TURN, tracing.SPAN_REVIEW, tracing.SPAN_QUESTION]
    # step 5: the review is the review's own records, never a planning phase run by an agent: the record's
    # transitions, the question, and how the reply read (not a command decision of the review router)
    assert [child.name for child in span_tree.children_of(review, sink)] == [
        tracing.SPAN_APPROVAL, tracing.SPAN_QUESTION, tracing.SPAN_REVIEW_REPLY, tracing.SPAN_APPROVAL]
    (reply,) = span_tree.named(sink, tracing.SPAN_REVIEW_REPLY)
    assert reply.attributes["read_as"] == "accept" and reply.attributes["by"] == "code"
    under_review = [span for span in span_tree.spans(sink) if tracing.SPAN_REVIEW in span_tree.path_of(span, sink)]
    assert not [span.name for span in under_review if span.name in tracing.WORK_SPAN_NAMES]
    assert review.attributes["rounds"] == 1 and review.attributes["items"] == 2
    # step 6: one run part per part, with what it is and what became of it
    (run,) = span_tree.named(sink, tracing.SPAN_RUN)
    parts = span_tree.children_of(run, sink, tracing.SPAN_RUN_PART)
    assert [p.attributes["part"] for p in parts] == [1, 2]
    assert [p.attributes["means"] for p in parts] == ["planning_command", "ad_hoc"]
    # part 1's planning command was refused at run time (its title is not in the reviewed values) and the agent
    # answered it: the part stays the planning command's, refused, answered by the agent
    assert [p.attributes["outcome"] for p in parts] == ["refused", "ran"]
    assert parts[0].attributes["answered_by"] == "agent" and parts[0].attributes["reason"].startswith("refused")
    assert run.attributes["parts"] == 2 and run.attributes["ran"] == 1 and run.attributes["refused"] == 1
    _the_tree_is_the_new_one(sink)


def _the_tree_is_the_new_one(sink) -> None:
    """No span is named for who did the work (agent, skill, planner, nlu), no retired name is written, and no span's
    parent has its own name."""
    names = {span.name for span in span_tree.spans(sink)}
    assert not [name for name in names if name.startswith(("fw.agent.", "fw.skill.", "fw.planner.", "fw.nlu."))
                or name not in tracing.SPAN_CONTRACTS], names
    assert span_tree.same_name_parents(sink) == []


def test_a_turn_resumed_in_another_context_reopens_the_same_stage_span(gated, monkeypatch, remote_decides_emma_and_joke):
    """The saved session carries the stage the turn waits in and how often each span was opened, so a context built
    from it reopens the SAME review span and closes it as its second segment."""
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    state = gated.ctx.serialize_state(channel_id="stages")
    assert state["turn"]["stages"]["waiting_in"] == tracing.SPAN_REVIEW
    (first,) = span_tree.named(gated.sink, tracing.SPAN_REVIEW)

    later = WorkflowExecutionContext(run_as_agent=True, trace_sink=gated.sink)
    later.bind_app_workflow(gated.wf)
    later.apply_serialized_state(state)
    with tracing.host_scope(later):
        tracing.resume_stage(later)
        tracing.close_stages(later, tracing.STATUS_OK)
    (review,) = span_tree.named(gated.sink, tracing.SPAN_REVIEW)
    assert review.span_id == first.span_id and review.attributes["segments"] == 2
    assert review.status == tracing.STATUS_OK


def test_a_fast_path_turn_has_no_review(gated, monkeypatch):
    _split_into(monkeypatch, [FIND])
    gated.script.steps = _finish()
    out = gated.ctx.process_turn(FIND)
    assert out.status is TurnStatus.COMPLETED
    assert _stages(gated.sink) == [tracing.SPAN_SPLIT, tracing.SPAN_RESOLVE, tracing.SPAN_RUN, tracing.SPAN_ANSWER]
    (part,) = span_tree.named(gated.sink, tracing.SPAN_RESOLVE_PART)
    assert part.attributes["fast_path"] is True and part.attributes["outcome"] == "bound"
    assert part.attributes["means"] == "command" and part.command_name == part.attributes["command"]
    (intent,) = [s for s in span_tree.named(gated.sink, tracing.SPAN_INTENT) if s.attributes.get("part") == 1]
    assert span_tree.path_of(intent, gated.sink) == [
        tracing.SPAN_TURN, tracing.SPAN_RESOLVE, tracing.SPAN_RESOLVE_PART, tracing.SPAN_INTENT]
    assert intent.attributes["decided_by"] == "local"


def _run_part_path(*names: str) -> list[str]:
    return [tracing.SPAN_TURN, tracing.SPAN_RUN, tracing.SPAN_RUN_PART, *names]


def _summarized_by(ctx, monkeypatch, summary: str = "the summary") -> None:
    """The turn's real conversation summary (the fixture stubs it), its model a ``DummyLM``."""
    from dspy.utils import DummyLM

    from fastworkflow import workflow_execution_context

    monkeypatch.setattr(ctx, "_extract_conversation_summary",
                        WorkflowExecutionContext._extract_conversation_summary.__get__(ctx))
    monkeypatch.setattr(workflow_execution_context.dspy_utils, "get_lm",
                        lambda *args, **kwargs: DummyLM([{"reasoning": "", "conversation_summary": summary}] * 4))


def test_an_ad_hoc_part_has_its_plan_under_it_and_its_steps_carry_the_plan(gated, monkeypatch, remote_decides_emma_and_joke):
    """A part no command fits, with an agent model (reviewed, then accepted): the text planner's ``fw.plan`` is the
    part's child, and the steps the agent then takes are its siblings, each carrying the plan's id."""
    from dspy.utils import DummyLM

    from fastworkflow import workflow_agent

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", _REAL_PLANNER)
    gated.ctx._current_planner_lm = DummyLM([{"reasoning": "", "next_steps": "1. tell a joke"}] * 4)
    _split_into(monkeypatch, [JOKE])
    gated.script.steps = _finish()
    assert gated.ctx.process_turn(JOKE).status is TurnStatus.AWAITING_USER       # the review of an ad hoc part
    assert gated.ctx.process_turn("yes").status is TurnStatus.COMPLETED
    sink = gated.sink
    (part,) = span_tree.named(sink, tracing.SPAN_RUN_PART)
    assert part.attributes["means"] == "ad_hoc" and part.attributes["outcome"] == "ran", part.attributes
    assert part.attributes["answered_by"] == "agent" and part.attributes["attempts"] == 1
    (plan,) = span_tree.named(sink, tracing.SPAN_PLAN)
    assert span_tree.path_of(plan, sink) == _run_part_path(tracing.SPAN_PLAN), span_tree.render(span_tree.tree(sink))
    assert plan.attributes["trigger"] == "initial"
    steps = span_tree.children_of(part, sink, tracing.SPAN_STEP)
    assert steps and all(step.attributes["plan_span_id"] == plan.span_id for step in steps)
    assert all(step.attributes["run_by"] == "agent" for step in steps)
    assert workflow_agent.build_query_with_next_steps is _REAL_PLANNER
    _the_tree_is_the_new_one(sink)


HELLO_WORLD = str(Path(fastworkflow.__file__).resolve().parent / "examples" / "hello_world")


class _Sink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        return True

    def record_conversation_label(self, *args) -> None:
        pass


def test_the_direct_path_decides_its_one_part_under_resolve(no_planner_key):
    """A person's typed command with no agent: ``fw.turn > fw.resolve > fw.resolve.part`` (the part it bound), and
    ``fw.run > fw.run.part > fw.command`` its run; no split and no review. Its decision (an exact name) and its
    extraction (typed values read in code, all valid) decided nothing, so neither is written (bd fix-qjne.34 decision
    b, on every path): their facts are on ``fw.command``."""
    if not Path(HELLO_WORLD, "___command_info", "global", "threshold.json").is_file():
        pytest.skip("hello_world is not trained on this machine")
    sink = _Sink()
    workflow = fastworkflow.Workflow.create(HELLO_WORLD, workflow_id_str=f"stages-{uuid.uuid4().hex}")
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    out = ctx.process_turn("add_two_numbers first_num=5 second_num=3")
    assert out.status is TurnStatus.COMPLETED
    assert _stages(sink) == [tracing.SPAN_RESOLVE, tracing.SPAN_RUN]
    assert span_tree.named(sink, tracing.SPAN_INTENT) == [] and span_tree.named(sink, tracing.SPAN_PARAMETERS) == []
    (part,) = span_tree.named(sink, tracing.SPAN_RESOLVE_PART)
    assert part.attributes["part"] == 1 and part.attributes["outcome"] == "bound"
    assert part.attributes["command"] == "add_two_numbers" and part.attributes["means"] == "command"
    # the bound command runs under fw.run > fw.run.part 1, its own span the dispatch's record (no tool-call span)
    (command,) = span_tree.named(sink, tracing.SPAN_COMMAND)
    assert span_tree.path_of(command, sink) == _run_part_path(tracing.SPAN_COMMAND)
    assert command.attributes["resolved_by"] == "exact_name" and command.attributes["success"] is True
    assert command.attributes["parameters_read_by"] == "typed" and command.attributes["parameters_valid"] is True
    (run_part,) = span_tree.named(sink, tracing.SPAN_RUN_PART)
    assert run_part.attributes["means"] == "command" and run_part.attributes["outcome"] == "ran"
    assert run_part.attributes["command"] == "add_two_numbers"
    _the_tree_is_the_new_one(sink)
