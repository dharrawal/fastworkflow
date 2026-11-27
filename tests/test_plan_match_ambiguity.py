"""A sub-task that matches several planning commands (bd fix-1ma8.1), and what is left of it since bd fix-o1lg S4/S5.

Since S4/S5 an ambiguous part is ASKED about before the review (``resolution.step``, tests/test_resolve_before_review.py),
so a live review never carries other plans: the whole-review tests below answer that question first
(``_ambiguous_review``). The review's own plan alternatives (the ``alternatives`` field, the line under the step, the
``switch_plan`` reply form, the select on the chat page and the reader of ``use B`` among them) were removed (bd fix-gwr7);
what stays is the typed switch ``step N: use NAME`` (tests/test_plan_switch_clauses.py).

Real components throughout, no mocks. The seams scripted are the ones the neighbouring review tests script: Jev's
remote decision model's answer as intent detection asks it (``scripted_matcher``, tests/conftest.py), the parameter
extractor and the planner. The head's answers to a reply are its recorded ones
(``recorded_review_router``, tests/recorded_router.py).

Part 1 is the rule (``resolution.draft_choice``: a detection binds a match; an ambiguity is left to its question). Part 2
pins that an unambiguous review is byte-identical to what it was before the feature. Part 4 is the reply policy on a
hand-built proposal. Part 5 runs replies through a real review. Part 6 is the cases that must not change: an accept, a
decline, an unattended turn, a multi-part request.
"""
from __future__ import annotations

import json

import pytest

import fastworkflow
from fastworkflow.review import review_engine as re_
from fastworkflow.intent import intent_detector
from fastworkflow import run_listing
from fastworkflow.review import review_proposals
from fastworkflow.observability import tracing
from fastworkflow._workflows.plan_metadata_extraction import resolution
from fastworkflow.review.review_engine import (
    Change, ProposalReplyPolicy, Verdict, prompt_for, reply_forms)
from fastworkflow.review.review_proposals import (
    MAX_PLAN_ALTERNATIVES, Means, SubTask, SubTaskProposal)

from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_plan_reply_policy import _wire_plan, planner, script_param_extraction  # noqa: F401
from tests.test_round1_gate import (  # noqa: F401 - the library review gate; ``no_draft_extraction`` is autouse
    EMMA, JOKE, _marker, _open_review, _split_into, gated, no_draft_extraction)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import no_local_model, script_matcher
from fastworkflow.core.turn import TurnStatus

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

# ----------------------------------------------------------------------
# Part 1: the rule
# ----------------------------------------------------------------------


def _proposal(goals: list[str], choices: dict) -> SubTaskProposal:
    """A drafted proposal: ``choices[n]`` is the plan bound to item n; an item with no entry is ``from_scratch``."""
    items = []
    for number, goal in enumerate(goals, 1):
        if number not in choices:
            items.append(SubTask(goal=goal))
            continue
        items.append(SubTask(goal=goal, means=Means(kind="planning", name=choices[number])))
    return SubTaskProposal(items=items)


def _intent(outcome, *candidates, command=None, kind=None):
    ranked = tuple(resolution.Ranked(name, kind_, score) for name, kind_, score in candidates)
    names = tuple(name for name, _, _ in candidates)
    return resolution.PartIntent(outcome, command, kind, ranked,
                                 "question" if outcome == resolution.AMBIGUOUS else "classifier",
                                 ambiguous=names if outcome == resolution.AMBIGUOUS else ())


P, X = resolution.KIND_PLANNING, resolution.KIND_EXECUTION


def test_the_constants_are_the_ones_the_shortlist_and_the_detection_use():
    assert run_listing.UNSURE_CANDIDATE_FLOOR == 0.2 == intent_detector.MATCHER_UNSURE_THRESHOLD
    assert MAX_PLAN_ALTERNATIVES == 3 == resolution.MAX_ALTERNATIVES == intent_detector.MATCHER_MAX_CANDIDATES


def test_a_draft_binds_only_a_match_and_an_ambiguity_is_left_to_its_question():
    """Rewritten for bd fix-o1lg S4/S5: an ambiguity is asked about before the review, never bound to its best guess
    with the others as alternatives."""
    match = _intent(resolution.MATCH, ("A", P, 0.55), ("B", P, 0.45), command="A", kind=P)
    assert resolution.draft_choice(match) == ("A", P)
    ambiguous = _intent(resolution.AMBIGUOUS, ("A", P, 0.45), ("B", P, 0.4))
    assert resolution.draft_choice(ambiguous) == (None, None)
    assert resolution.draft_choice(_intent(resolution.NONE, ("A", P, 0.1))) == (None, None)


# ----------------------------------------------------------------------
# Part 2: an unambiguous review is what it was
# ----------------------------------------------------------------------

#: What a one-item unambiguous review serialised to before this feature: no ``alternatives`` key.
BEFORE_ITEM = {"goal": "Goal one", "outcome": "", "inputs": [], "means": {"kind": "planning", "name": "A"},
               "check": "", "drafted_means": "", "parts": [], "parts_fingerprint": ""}


def _clear_leader() -> SubTaskProposal:
    return _proposal(["Goal one"], {1: "A"})


def test_an_unambiguous_review_serialises_to_exactly_what_it_did_before():
    plan = _clear_leader()
    assert plan.items[0].model_dump(mode="json") == BEFORE_ITEM
    assert plan.items[0].model_dump() == BEFORE_ITEM
    assert json.loads(plan.items[0].model_dump_json()) == BEFORE_ITEM
    assert plan.binding() == {"spec": "review_subtasks", "kind": "sub_tasks", "items": [BEFORE_ITEM]}
    old_shape = SubTaskProposal(items=[SubTask(goal="Goal one", means=Means(kind="planning", name="A"))])
    assert plan.fingerprint() == old_shape.fingerprint()


def test_an_unambiguous_review_shows_the_same_text_and_the_same_reply_forms():
    plan = _clear_leader()
    assert plan.user_rendering() == "Sub-tasks:\n1. Goal one"
    shown = prompt_for(plan)
    assert shown == "Sub-tasks:\n1. Goal one\n" + re_.SUB_TASK_PROMPT_HINT
    assert reply_forms(shown, plan) == [
        {"kind": "accept", "reply": "yes"}, {"kind": "edit", "free_text": True, "per_item": False}]
    structured = prompt_for(plan, structured=True)
    old_shape = SubTaskProposal(items=[SubTask(goal="Goal one", means=Means(kind="planning", name="A"))])
    assert reply_forms(structured, plan, structured=True) == reply_forms(structured, old_shape, structured=True)
    assert "switch_plan" not in {form["kind"] for form in reply_forms(structured, plan, structured=True)}


def test_a_review_with_no_ambiguous_step_never_reads_a_reply_as_a_plan_switch(monkeypatch):
    plan = _clear_leader()
    for reply in ("use B", "B instead"):
        seen: list[str] = []
        reader = lambda text, proposal: seen.append(text) or []                    # noqa: E731
        decision = ProposalReplyPolicy(reader).decide(plan, reply)
        assert seen == [reply] and decision.verdict is Verdict.REPROMPT            # the existing reader read it, as before


LOANS = "Library/book_loans"
AUTHOR = "Library/author_books"
BY_TITLE = "Library/book_by_title"


# ----------------------------------------------------------------------
# Part 4: the reply reader, policy level (no model)
# ----------------------------------------------------------------------


def _single() -> SubTaskProposal:
    """One step bound to a planning command with a real name."""
    return _proposal(["Who wrote Emma and how many copies"], {1: BY_TITLE})


def test_a_typed_means_edit_still_works_and_is_not_read_as_a_switch():
    decision = ProposalReplyPolicy().decide(_single(), f"change 1: means=planning:{LOANS}")
    assert decision.verdict is Verdict.EDIT
    assert decision.edits == (Change(index=1, field="means", value=f"planning:{LOANS}"),)


# ----------------------------------------------------------------------
# Part 6a: the accept and decline paths are untouched
# ----------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["yes", "accept", "looks good", "go ahead", "Sounds good.", "Yes."])
def test_a_bare_accept_is_decided_before_any_reader_is_asked(reply):
    def reader_that_must_not_run(text, proposal):
        raise AssertionError("the reader ran on an accept")

    decision = ProposalReplyPolicy(reader_that_must_not_run).decide(_single(), reply)
    assert decision.verdict is Verdict.CONFIRM


@pytest.mark.parametrize("reply", ["no", "cancel", "nope", "no thanks", "scratch that plan"])
def test_a_decline_is_decided_before_any_reader_is_asked(reply):
    def reader_that_must_not_run(text, proposal):
        raise AssertionError("the reader ran on a decline")

    assert ProposalReplyPolicy(reader_that_must_not_run).decide(_single(), reply).verdict is Verdict.DECLINE


# ----------------------------------------------------------------------
# Part 6a2: an accept word followed by anything else is the accept it was
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    ["yes use Library/count_books", "yes, run that", "yes, please", "yes, go ahead", "ok, looks good", "sure, run it", "yes, bo",
     "yes, step 2: use author_books", "yes use it"],
)
def test_an_accept_word_followed_by_anything_else_is_the_accept_it_was(reply):
    """Including a plan name: ``yes use X`` accepts, as it did before the plan alternatives were offered."""
    assert ProposalReplyPolicy().decide(_single(), reply).verdict is Verdict.CONFIRM


# ----------------------------------------------------------------------
# Part 5: through a real review (the library copy, the real agent; Jev's match and the models are the scripted seams)
# ----------------------------------------------------------------------


@pytest.fixture
def no_heads(monkeypatch):
    """No trained local model answers (also the state of a workflow that has not been trained), so the decision is the
    remote decision model's scripted answer alone: the numbers below are then the ones the detection sees."""
    no_local_model(monkeypatch)


def _jev(monkeypatch, by_text: dict[str, dict[str, float]]):
    """The remote decision model's answer, scripted for each sub-task text (a text not listed: the call fails)."""
    script_matcher(monkeypatch, by_text)


#: No plan reaches the match bar (0.5), three clear the unsure floor (0.2): an ambiguity, best first.
EMMA_PLANS = {BY_TITLE: 0.45, AUTHOR: 0.4, LOANS: 0.25, "none": 0.0}
JOKE_NONE = {"none": 0.95}


def _ambiguous_review(gated, monkeypatch):
    """A turn whose first part is ambiguous: the part is asked about first (bd fix-o1lg S4/S5), the user picks
    book_by_title, and the review of the whole turn opens with that plan bound."""
    _jev(monkeypatch, {EMMA: EMMA_PLANS, JOKE: JOKE_NONE})
    _split_into(monkeypatch, [EMMA, JOKE])
    asked = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert asked.status is TurnStatus.AWAITING_USER
    assert _artifacts(asked)["question_kind"] == "plan_choice", _artifacts(asked)
    out = gated.ctx.process_turn(BY_TITLE)
    assert out.status is TurnStatus.AWAITING_USER
    return out, _open_review(gated)


def _items(record):
    return re_.proposal_of(record).items


def _artifacts(out) -> dict:
    return out.command_outputs[-1].command_response.artifacts


def test_a_valid_plan_that_is_not_an_alternative_switches_the_step_and_the_reader_is_not_asked(gated, no_heads, planner, monkeypatch):
    """Was "falls through to the existing reader": a bare name that is a planning command of the workflow but not one of
    the step's alternatives is now a switch (bd fix-1ma8.1 follow-up), decided here with no model call."""
    _, record = _ambiguous_review(gated, monkeypatch)
    lm = planner(gated, {"edits": []})
    gated.ctx.process_turn("step 1: use Library/count_books")
    assert record.state == re_.REVISED and _items(record)[0].means.name == "Library/count_books"
    assert len(lm.history) == 0


def test_a_bare_name_that_resolves_to_nothing_still_falls_through_to_the_existing_reader(gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    before = [item.model_dump() for item in _items(record)]
    lm = planner(gated, {"edits": []})
    out = gated.ctx.process_turn("use the library")
    assert record.state == re_.AWAITING and [item.model_dump() for item in _items(record)] == before
    assert len(lm.history) == 1                                  # the existing free-text reader was asked, as before
    assert "Which plan" not in out.answer and "Which step" not in out.answer


@pytest.mark.usefixtures("recorded_review_router")
def test_a_valid_plan_that_is_not_an_alternative_switches_through_the_plan_workflow_too(gated, no_heads, planner, monkeypatch):
    """With the plan metadata extraction workflow reading, the same reply is decided before the head is asked."""
    _, record = _ambiguous_review(gated, monkeypatch)
    _wire_plan(gated)
    planner(gated, {"edits": []})
    gated.ctx.process_turn("step 1: use Library/count_books")
    assert record.state == re_.REVISED and _items(record)[0].means.name == "Library/count_books"


@pytest.mark.usefixtures("recorded_review_router")
def test_a_bare_name_that_resolves_to_nothing_still_falls_through_to_the_plan_head(gated, no_heads, planner, monkeypatch):
    """The plan head's recorded answer for a reply that names no plan: nothing is switched, nothing is asked
    about plans."""
    _, record = _ambiguous_review(gated, monkeypatch)
    _wire_plan(gated)
    before = [item.model_dump() for item in _items(record)]
    planner(gated, {"edits": []})
    out = gated.ctx.process_turn("use book_by_title for the second part")
    assert [item.model_dump() for item in _items(record)] == before
    assert "Which plan" not in out.answer and "Which step do you mean?" not in out.answer


def test_the_switch_also_reads_through_the_plan_workflow_reader(gated, no_heads, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    _wire_plan(gated)
    gated.ctx.process_turn("step 1: use author_books")
    assert record.state == re_.REVISED and _items(record)[0].means.name == AUTHOR


# ----------------------------------------------------------------------
# Part 6b: nobody to answer, and the accept path through a real review
# ----------------------------------------------------------------------


def test_an_unattended_turn_binds_only_a_match_so_an_ambiguous_part_goes_to_the_agent(gated, no_heads, monkeypatch):
    """Nobody reviews an unattended draft, so it binds only what the decision matched (bd fix-o1lg; it bound the
    best-ranked plan before): the ambiguous part is left to the agent, and no ambiguity is recorded."""
    _jev(monkeypatch, {EMMA: EMMA_PLANS, JOKE: JOKE_NONE})
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True)
    assert out.status is TurnStatus.COMPLETED
    assert [record for record in gated.store.records if record.command == "review_subtasks"] == []
    assert _marker(gated) == "not_reviewed"
    plan = gated.ctx._turn_subtask_plan
    assert plan.items[0].means == Means()
    span = [each for each in gated.sink.spans if each.name == tracing.SPAN_RESOLVE][-1]
    assert span.attributes["plan_ambiguity"] is None     # on the resolve stage since span contract 63


def test_an_unambiguous_request_opens_the_same_review_and_records_nothing(gated, no_heads, monkeypatch):
    _jev(monkeypatch, {EMMA: {BY_TITLE: 0.95, AUTHOR: 0.3, "none": 0.0}, JOKE: JOKE_NONE})          # a match
    _split_into(monkeypatch, [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    assert "other plans" not in record.rendering and "other plans" not in out.answer
    review = _artifacts(out)["review"]
    assert all("alternatives" not in item for item in review["items"])
    assert [form["kind"] for form in review["reply_forms"]] == ["accept", "edit"]
    assert all("alternatives" not in item for item in record.binding["items"])
    span = [each for each in gated.sink.spans if each.name == tracing.SPAN_RESOLVE][-1]
    assert span.attributes["plan_ambiguity"] is None     # on the resolve stage since span contract 63


@pytest.mark.parametrize("reply", ["yes", "accept", "looks good"])
def test_an_accept_of_a_review_after_an_ambiguous_part_runs_the_bound_plan(reply, gated, no_heads, monkeypatch):
    # An agent model is set (the agent is scripted, no call reaches it): the explicit-yes rule for an ad hoc part that
    # may change data applies only then. Before, a developer's .env that litellm loaded into the process set it.
    monkeypatch.setitem(fastworkflow._env_vars, "LLM_AGENT", "test/agent-model")
    _, record = _ambiguous_review(gated, monkeypatch)
    gated.script.steps = [("finish", {})] * 2
    out = gated.ctx.process_turn(reply)
    if reply == "looks good":
        # The ad hoc part (JOKE) may change data (the library has a command that writes, and an agent will plan the
        # part), so only an explicit yes runs the turn (bd fix-ftdl.8.4): the accept is asked again, then yes runs it.
        assert out.status is TurnStatus.AWAITING_USER and re_.MAY_CHANGE_DATA_SENTENCE in out.answer
        out = gated.ctx.process_turn("yes")
    assert out.status is TurnStatus.COMPLETED and record.state == re_.CONFIRMED
    assert review_proposals.accepted_subtask_proposal(gated.wf).items[0].means.name == BY_TITLE


def test_a_decline_of_an_ambiguous_review_runs_nothing(gated, no_heads, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    out = gated.ctx.process_turn("no")
    assert record.state == re_.DECLINED and out.status is TurnStatus.COMPLETED
