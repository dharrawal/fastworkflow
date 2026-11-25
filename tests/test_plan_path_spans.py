"""The plan path writes the spans the "/" path writes (span contract v58): one ``fw.intent`` span per decision of a
part (never one per read of the turn's memo) and one ``fw.parameters`` span for the extraction of the part's
own command, both carrying the part's number, written by the same functions the CME's stages call
(``intent_detection.traced_decision``, ``parameter_extraction.traced_extraction``).

Where they sit (bd fix-qjne.34): each is a child of its part's ``fw.resolve.part`` span, under the turn's ``fw.resolve``
stage span (``tests/span_tree.py``), one span per part however many decisions and drafts it has.

Library example, no agent model, the matcher and the extractor scripted (``tests/test_parameter_loop.py``)."""
from __future__ import annotations

import inspect

import pytest

from fastworkflow import tracing
from fastworkflow.utils.signatures import InputForParamExtraction
from tests import span_tree
from tests.conftest import script_matcher
from tests.test_parameter_loop import _reply, HOST, no_agent, _offline, BOOK_PLAN, ABOUT_EMMA, ABOUT_DUNE  # noqa: F401
from tests.test_round1_gate import JOKE, gated  # noqa: F401
from tests.test_approval_binding import annotated_library, grant_root, todo, no_live_model, no_model_call_for_parameters  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from fastworkflow.workflow_agent import build_query_with_next_steps as _REAL_PLANNER   # the fixtures stub it

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

FIND_BOOK, AUTHOR = "Library/find_book", "Library/author_books"


def _spans(env, name: str, since: int = 0) -> list:
    return [span for span in env.sink.spans[since:] if span.name == name]


def test_one_intent_span_per_decision_with_the_part_number_and_the_chain_facts(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="Emma"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA, ABOUT_DUNE]
    no_agent.ctx.process_turn(f"{ABOUT_EMMA}. {ABOUT_DUNE}", approval_surface=HOST)
    spans = _spans(no_agent, tracing.SPAN_INTENT)
    by_part = {span.attributes["part"]: span for span in spans}
    assert sorted(by_part) == [1, 2] and len(spans) == 2, [s.attributes.get("utterance") for s in spans]   # not one per memo read
    intent_ids = {span.span_id for span in spans}
    assert not any(span.parent_span_id in intent_ids for span in spans)   # parts decided side by side are siblings
    for number, text in ((1, ABOUT_EMMA), (2, ABOUT_DUNE)):
        attributes = by_part[number].attributes
        assert attributes["utterance"] == text and attributes["reads_reply"] is False
        assert attributes["decided_by"] == "remote" and attributes["engine_chain"]   # the chain's facts, as the CME's span has them
        assert "resolved" in attributes and "command_name" in attributes
        assert attributes[tracing.ATTR_SPAN_CONTRACT_VERSION] == 14 if tracing.ATTR_SPAN_CONTRACT_VERSION in attributes else True


def test_an_ambiguity_is_traced_with_its_candidates(no_agent, monkeypatch):
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.45, AUTHOR: 0.3, BOOK_PLAN: 0.25, "none": 0.0})
    no_agent.parts = [JOKE]
    no_agent.ctx.process_turn(JOKE, approval_surface=HOST)
    (span,) = _spans(no_agent, tracing.SPAN_INTENT)
    assert span.attributes["part"] == 1 and span.attributes["ambiguous"] is True
    assert span.attributes["decided_by"] == "question" and set(span.attributes["candidates"]) >= {FIND_BOOK, AUTHOR}
    # the which-command question it leads to waits under the same part
    (question,) = span_tree.named(no_agent.sink, tracing.SPAN_QUESTION)
    assert question.attributes["kind"] == "which_command" and question.attributes["part"] == 1
    assert span_tree.path_of(question, no_agent.sink) == [
        tracing.SPAN_TURN, tracing.SPAN_RESOLVE, tracing.SPAN_RESOLVE_PART, tracing.SPAN_QUESTION]


def test_the_extraction_of_the_part_s_own_command_is_traced_with_the_part_number(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="Emma"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA]
    no_agent.ctx.process_turn(ABOUT_EMMA, approval_surface=HOST)
    spans = [s for s in _spans(no_agent, tracing.SPAN_PARAMETERS) if s.attributes.get("part") == 1]
    assert spans, "the part's own extraction wrote no span"
    attributes = spans[0].attributes
    assert attributes["command_name"] == BOOK_PLAN and attributes["parameters_valid"] is True
    assert attributes["extraction_method"] == "llm"


def test_a_refused_value_is_traced_as_the_cme_traces_it(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="it"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    text = "who wrote it and how many copies"
    no_agent.parts = [text]
    no_agent.ctx.process_turn(text, approval_surface=HOST)
    span = next(s for s in _spans(no_agent, tracing.SPAN_PARAMETERS) if s.attributes.get("part") == 1)
    assert span.attributes["parameters_valid"] is False and span.attributes["missing_fields"] == ["title"]


def test_the_not_in_request_advisory_is_on_the_plan_path_s_extraction_span(no_agent, monkeypatch):
    """The CME's span carries ``argument_check_not_in_request`` for an unmarked field (``start``) whose value the request
    does not write; the plan path's span for an execution command's own extraction carries it from the same guard."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(query="windmills", start=7))
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.95, "none": 0.0})
    text = "find a book about windmills"
    no_agent.parts = [text]
    no_agent.ctx.process_turn(text, approval_surface=HOST)
    span = next(s for s in _spans(no_agent, tracing.SPAN_PARAMETERS) if s.attributes.get("part") == 1)
    assert span.attributes.get("argument_check_not_in_request") == ["start"], span.attributes


def test_both_paths_call_the_one_writer():
    from fastworkflow._workflows.command_metadata_extraction import intent_detection, parameter_extraction
    from fastworkflow._workflows.plan_metadata_extraction import resolution
    from fastworkflow import review_proposals

    assert "traced_decision" in inspect.getsource(intent_detection.CommandNamePrediction.predict)
    assert "traced_decision" in inspect.getsource(resolution._decided)
    assert "traced_extraction" in inspect.getsource(parameter_extraction.ParameterExtraction.extract)
    assert "traced_extraction" in inspect.getsource(review_proposals.draft_inputs)


PART_PATH = [tracing.SPAN_TURN, tracing.SPAN_RESOLVE, tracing.SPAN_RESOLVE_PART]


def test_each_part_s_decision_and_extraction_sit_under_that_part_in_the_resolve_stage(no_agent, monkeypatch):
    """Two parts decided and drafted side by side (the pool): every ``fw.intent`` and ``fw.parameters`` span with
    ``part=N`` has the path turn > resolve > resolve.part, and that part span is part N's; ONE part span per part (by
    id), closed with what the resolution bound; no span's parent has its own name; the split stage came first."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="Emma"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA, ABOUT_DUNE]
    no_agent.ctx.process_turn(f"{ABOUT_EMMA}. {ABOUT_DUNE}", approval_surface=HOST)
    sink = no_agent.sink
    index = span_tree.by_id(sink)
    decided = [s for s in span_tree.spans(sink)
               if s.name in (tracing.SPAN_INTENT, tracing.SPAN_PARAMETERS) and s.attributes.get("part")]
    assert {s.name for s in decided} == {tracing.SPAN_INTENT, tracing.SPAN_PARAMETERS}
    for span in decided:
        assert span_tree.path_of(span, sink) == [*PART_PATH, span.name], span_tree.render(span_tree.tree(sink))
        assert index[span.parent_span_id].attributes["part"] == span.attributes["part"]
    parts = {p.attributes["part"]: p for p in span_tree.named(sink, tracing.SPAN_RESOLVE_PART)}
    assert sorted(parts) == [1, 2]
    for number, text in ((1, ABOUT_EMMA), (2, ABOUT_DUNE)):
        part = parts[number]
        assert part.attributes["command"] == BOOK_PLAN and part.command_name == BOOK_PLAN
        assert part.attributes["means"] == "planning_command" and part.attributes["request"] == text
    # part 2's title ("Emma", as scripted) is not in its request, so the turn waits on part 2's values question
    assert parts[1].attributes["outcome"] == "bound" and parts[2].attributes["outcome"] == "asked"
    assert span_tree.same_name_parents(sink) == []
    (turn,) = span_tree.tree(sink)
    stages = [child.name for child in turn.children if child.name in tracing.STAGE_SPAN_NAMES]
    assert stages[:2] == [tracing.SPAN_SPLIT, tracing.SPAN_RESOLVE], stages
    assert len(span_tree.named(sink, tracing.SPAN_RESOLVE)) == 1


def test_a_part_waiting_on_its_question_keeps_one_span_across_the_wait(no_agent, monkeypatch):
    """The which-command question suspends the turn in its resolve stage: the stage and the part close with the
    turn's status while it waits, and the reply reopens the SAME spans (``segments`` 2), closed once more when the
    resolution ends."""
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.45, AUTHOR: 0.3, BOOK_PLAN: 0.25, "none": 0.0})
    no_agent.parts = [JOKE]
    first = no_agent.ctx.process_turn(JOKE, approval_surface=HOST)
    (part,) = span_tree.named(no_agent.sink, tracing.SPAN_RESOLVE_PART)
    assert part.status == "awaiting_user" and part.attributes["outcome"] == "asked", part.attributes
    no_agent.ctx.process_turn("cancel", approval_surface=HOST)
    (part,) = span_tree.named(no_agent.sink, tracing.SPAN_RESOLVE_PART)
    (stage,) = span_tree.named(no_agent.sink, tracing.SPAN_RESOLVE)
    assert part.attributes["segments"] == 2 and part.attributes["resumed"] is True
    assert part.attributes["outcome"] == "dropped" and part.status == tracing.STATUS_OK
    assert stage.attributes["segments"] == 2 and stage.status == tracing.STATUS_OK
    assert first is not None
    # bd fix-qjne.34 decision g: the part dropped at resolve is still a part of the run, not run, with why.
    (run_part,) = span_tree.named(no_agent.sink, tracing.SPAN_RUN_PART)
    assert span_tree.path_of(run_part, no_agent.sink) == [tracing.SPAN_TURN, tracing.SPAN_RUN, tracing.SPAN_RUN_PART]
    assert run_part.attributes["part"] == 1 and run_part.attributes["outcome"] == "not_run", run_part.attributes
    assert run_part.attributes["reason"].startswith("dropped_at_resolve: "), run_part.attributes
    (run,) = span_tree.named(no_agent.sink, tracing.SPAN_RUN)
    assert run.attributes["not_run"] == 1 and run.attributes["ran"] == 0


def test_a_part_dropped_at_resolve_keeps_its_number_under_the_run(no_agent, monkeypatch):
    """Two parts; part 2's values question is cancelled, so part 2 is dropped and part 1 runs: the run has BOTH parts,
    numbered as the resolve numbered them, part 1 ran and part 2 was not run (dropped at resolve)."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="Emma"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA, ABOUT_DUNE]
    no_agent.ctx.process_turn(f"{ABOUT_EMMA}. {ABOUT_DUNE}", approval_surface=HOST)
    no_agent.ctx.process_turn("cancel", approval_surface=HOST)
    if span_tree.named(no_agent.sink, tracing.SPAN_REVIEW):   # part 1 alone is reviewed: accept it
        _reply(no_agent, "yes")
    sink = no_agent.sink
    resolved = {p.attributes["part"]: p for p in span_tree.named(sink, tracing.SPAN_RESOLVE_PART)}
    assert resolved[2].attributes["outcome"] == "dropped", resolved[2].attributes
    ran = {p.attributes["part"]: p for p in span_tree.named(sink, tracing.SPAN_RUN_PART)}
    assert sorted(ran) == [1, 2], span_tree.render(span_tree.tree(sink))
    assert ran[1].attributes["command"] == BOOK_PLAN, ran[1].attributes
    assert not str(ran[1].attributes.get("reason") or "").startswith("dropped_at_resolve"), ran[1].attributes
    assert ran[2].attributes["outcome"] == "not_run", ran[2].attributes
    assert ran[2].attributes["reason"].startswith("dropped_at_resolve: "), ran[2].attributes
    assert ran[2].attributes["command"] == BOOK_PLAN


RUN_PART_PATH = [tracing.SPAN_TURN, tracing.SPAN_RUN, tracing.SPAN_RUN_PART]


def test_a_planning_command_part_with_no_agent_model_runs_its_steps_by_code(no_agent, monkeypatch):
    """No agent model (bd fix-qjne.34 step 6): the part's planning command runs its steps by code, and the tree says so
    in the same names the agent path uses: ``fw.run.part`` (means planning_command) > ``fw.phase`` > ``fw.step`` (run
    by code) > ``fw.command``. No span is named for who did the work."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title="Emma"))
    script_matcher(monkeypatch, lambda t: {BOOK_PLAN: 0.95, "none": 0.0})
    no_agent.parts = [ABOUT_EMMA]
    out = no_agent.ctx.process_turn(ABOUT_EMMA, approval_surface=HOST)
    assert out.status.value == "awaiting_user"        # the turn's review: accept it
    _reply(no_agent, "yes")
    sink = no_agent.sink
    rendered = span_tree.render(span_tree.tree(sink))
    (part,) = span_tree.named(sink, tracing.SPAN_RUN_PART)
    assert part.attributes["means"] == "planning_command" and part.attributes["command"] == BOOK_PLAN, rendered
    phases = span_tree.named(sink, tracing.SPAN_PHASE)
    assert phases and all(span_tree.path_of(p, sink) == [*RUN_PART_PATH, tracing.SPAN_PHASE] for p in phases), rendered
    steps = [s for s in span_tree.named(sink, tracing.SPAN_STEP) if s.attributes.get("kind") == "command"]
    assert steps and all(s.attributes["run_by"] == "code" for s in steps), rendered
    commands = [c for c in span_tree.named(sink, tracing.SPAN_COMMAND)
                if span_tree.path_of(c, sink)[:4] == [*RUN_PART_PATH, tracing.SPAN_PHASE]]
    assert commands and all(span_tree.path_of(c, sink) == [*RUN_PART_PATH, tracing.SPAN_PHASE, tracing.SPAN_STEP,
                                                            tracing.SPAN_COMMAND] for c in commands), rendered
    assert all(c.attributes["resolved_by"] for c in commands)
    names = {s.name for s in span_tree.spans(sink)}
    assert not [n for n in names if n.startswith(("fw.agent.", "fw.skill.", "fw.planner.", "fw.nlu."))], names
    assert span_tree.same_name_parents(sink) == []


def test_an_ad_hoc_part_not_run_without_an_agent_model_says_why_and_keeps_its_plan(no_agent, monkeypatch):
    """No agent model, a part no command fits: its ``fw.run.part`` is ``means`` ad_hoc, ``outcome`` not_run with the
    reason, and the planner's suggested plan is its ``fw.plan`` child."""
    from dspy.utils import DummyLM

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", _REAL_PLANNER)
    no_agent.ctx._current_planner_lm = DummyLM([{"reasoning": "", "next_steps": "1. find a joke\n2. tell it"}] * 4)
    script_matcher(monkeypatch, lambda t: {"none": 0.99})
    no_agent.parts = [JOKE]
    out = no_agent.ctx.process_turn(JOKE, approval_surface=HOST)
    assert out.status.value == "awaiting_user"        # the which-plan question: none of them
    no_agent.ctx.process_turn("none", approval_surface=HOST)
    sink = no_agent.sink
    rendered = span_tree.render(span_tree.tree(sink))
    parts = span_tree.named(sink, tracing.SPAN_RUN_PART)
    assert parts, rendered
    (part,) = parts
    assert part.attributes["means"] == "ad_hoc" and part.attributes["outcome"] == "not_run", part.attributes
    assert part.attributes["reason"]
    (run,) = span_tree.named(sink, tracing.SPAN_RUN)
    assert run.attributes["not_run"] == 1 and run.attributes["ran"] == 0
    plans = span_tree.children_of(part, sink, tracing.SPAN_PLAN)
    assert plans, rendered


def test_an_execution_command_part_is_a_step_with_its_command_and_the_answer_holds_the_summary(no_agent, monkeypatch):
    """One execution command, no agent model: ``fw.run.part`` (means command) > ``fw.step`` (run by code) >
    ``fw.command``. The command's own decision and values decided nothing (its exact name, typed values), so they are
    on the command's span (``resolved_by``, ``parameters_read_by``) and not written under it. The conversation summary's
    model call is the answer's, and says what it is for."""
    from tests.test_turn_stage_spans import _summarized_by

    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(query="windmills"))
    script_matcher(monkeypatch, lambda t: {FIND_BOOK: 0.95, "none": 0.0})
    _summarized_by(no_agent.ctx, monkeypatch)
    text = "find a book about windmills"
    no_agent.parts = [text]
    no_agent.ctx.process_turn(text, approval_surface=HOST)
    sink = no_agent.sink
    rendered = span_tree.render(span_tree.tree(sink))
    (part,) = span_tree.named(sink, tracing.SPAN_RUN_PART)
    assert part.attributes["means"] == "command" and part.attributes["outcome"] == "ran", part.attributes
    assert part.attributes["answered_by"] == "code"
    (command,) = span_tree.named(sink, tracing.SPAN_COMMAND)
    assert span_tree.path_of(command, sink) == [*RUN_PART_PATH, tracing.SPAN_STEP, tracing.SPAN_COMMAND], rendered
    step = span_tree.by_id(sink)[command.parent_span_id]
    assert step.attributes["run_by"] == "code" and step.attributes["kind"] == "command"
    assert step.attributes["raw_command"] and step.attributes["success"] is True
    assert command.attributes["resolved_by"] == "exact_name" and command.attributes["parameters_read_by"] == "typed"
    assert command.attributes["parameters_valid"] is True
    assert not span_tree.children_of(command, sink, tracing.SPAN_INTENT)
    assert not span_tree.children_of(command, sink, tracing.SPAN_PARAMETERS)
    (answer,) = span_tree.named(sink, tracing.SPAN_ANSWER)
    assert answer.attributes["written_by"] == "code" and answer.attributes["summary_failed"] is False
    (summary,) = span_tree.children_of(answer, sink, tracing.SPAN_LLM_CALL)
    assert summary.attributes["purpose"] == "conversation_summary"
    assert [s for s in span_tree.named(sink, tracing.SPAN_LLM_CALL) if s.attributes.get("purpose")] == [summary]
    assert span_tree.same_name_parents(sink) == []
