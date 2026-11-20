"""A sub-task that matches several planning commands (bd fix-1ma8.1), and what is left of it since bd fix-o1lg S4/S5.

Since S4/S5 an ambiguous part is ASKED about before the review (``resolution.step``, tests/test_resolve_before_review.py),
so a live review never carries alternatives: the whole-review tests below answer that question first
(``_ambiguous_review``), and the tests of a review that showed alternatives and switched among them were removed. The
policy-level tests of the switch reader on hand-built proposals with alternatives remain until that reader is deleted
(S10).

Real components throughout, no mocks. The seams scripted are the ones the neighbouring review tests script: Jev's
remote decision model's answer as intent detection asks it (``scripted_matcher``, tests/conftest.py), the parameter
extractor and the planner. The head's answers to a reply are its recorded ones
(``recorded_review_router``, tests/recorded_router.py).

Part 1 is the rule (``resolution.draft_choice``: an ambiguous detection binds its best candidate and names the other
plans, at most three, in the detection's order; a match names none; bd fix-o1lg replaced the margin rule). Part 2 pins that an unambiguous
review is byte-identical to what it was before the feature. Part 3 is what an ambiguous review shows on every surface
(text, payload, reply forms, the turn span). Part 4 is the reply reader, policy level. Part 5 runs the replies through
a real review: the means switches, the inputs are re-drafted, the plan it replaced becomes an alternative. Part 6 is
the cases that must not change: an accept, a decline, an unattended turn, a multi-part request.
"""
from __future__ import annotations

import json

import pytest

import fastworkflow
from fastworkflow import review_engine as re_
from fastworkflow import intent_detector, review_plan_switch, review_proposals, run_listing, tracing
from fastworkflow._workflows.plan_metadata_extraction import resolution
from fastworkflow.review_engine import (
    Change, ProposalReplyPolicy, Verdict, apply_edits, parse_edits, prompt_for, reply_forms)
from fastworkflow.review_proposals import (
    MAX_PLAN_ALTERNATIVES, Means, PlanAlternative, SubTask, SubTaskProposal)

from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_plan_reply_policy import _wire_plan, planner, script_param_extraction  # noqa: F401
from tests.test_round1_gate import (  # noqa: F401 - the library review gate; ``no_draft_extraction`` is autouse
    EMMA, JOKE, _marker, _open_review, _split_into, gated, no_draft_extraction)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import no_local_model, script_matcher
from fastworkflow.turn import TurnStatus

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

# ----------------------------------------------------------------------
# Part 1: the rule
# ----------------------------------------------------------------------


def _proposal(goals: list[str], choices: dict) -> SubTaskProposal:
    """A drafted proposal: ``choices[n] = (chosen plan, [other plan or (name, description), ...])`` for item n; an item
    with no entry is ``from_scratch``. The shape ``draft_round1_proposal`` gives an ambiguous detection."""
    items = []
    for number, goal in enumerate(goals, 1):
        if number not in choices:
            items.append(SubTask(goal=goal))
            continue
        chosen, others = choices[number]
        alternatives = [PlanAlternative(name=each, description="") if isinstance(each, str)
                        else PlanAlternative(name=each[0], description=each[1]) for each in others]
        items.append(SubTask(goal=goal, means=Means(kind="planning", name=chosen), alternatives=alternatives))
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

#: What a one-item unambiguous review serialised to before this feature: no ``alternatives`` key at all.
BEFORE_ITEM = {"goal": "Goal one", "outcome": "", "inputs": [], "means": {"kind": "planning", "name": "A"},
               "check": "", "drafted_means": "", "parts": [], "parts_fingerprint": ""}


def _clear_leader() -> SubTaskProposal:
    return _proposal(["Goal one"], {1: ("A", [])})


def test_an_unambiguous_review_serialises_to_exactly_what_it_did_before():
    plan = _clear_leader()
    assert plan.items[0].alternatives == []
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
    assert all(not item.alternatives for item in plan.items)


def test_a_review_with_no_ambiguous_step_never_reads_a_reply_as_a_plan_switch(monkeypatch):
    plan = _clear_leader()
    for reply in ("use B", "step 1: use B", "B instead", "switch step 1 to B"):
        assert review_plan_switch.read_plan_switch(plan, reply) is None
    seen: list[str] = []
    reader = lambda text, proposal: seen.append(text) or []                        # noqa: E731
    decision = ProposalReplyPolicy(reader).decide(plan, "use B")
    assert seen == ["use B"] and decision.verdict is Verdict.REPROMPT              # the existing reader read it, as before


# ----------------------------------------------------------------------
# Part 3: an ambiguous review shows the other plans on every surface
# ----------------------------------------------------------------------

LOANS = "Library/book_loans"
AUTHOR = "Library/author_books"
BY_TITLE = "Library/book_by_title"


def _ambiguous() -> SubTaskProposal:
    """Step 1 is ambiguous (A, then B and C); step 2 is not."""
    return _proposal(["Goal one", "Goal two"], {1: ("A", [("B", "does b"), ("C", "does c")]), 2: ("D", [])})


def test_the_review_text_names_the_other_plans_in_one_line_under_the_step():
    plan = _ambiguous()
    assert plan.user_rendering() == (
        "Sub-tasks:\n1. Goal one\n   other plans that fit: B, C; reply \"step 1: use B\" to switch\n2. Goal two")
    assert "other plans" not in plan.model_rendering()                       # the agent's own rendering is unchanged
    assert prompt_for(plan).endswith(re_.SUB_TASK_PROMPT_HINT)


def test_the_payload_item_carries_the_alternatives_with_a_description_and_no_score():
    plan = _ambiguous()
    first, second = plan.binding()["items"]
    assert first["alternatives"] == [{"name": "B", "description": "does b"}, {"name": "C", "description": "does c"}]
    assert "alternatives" not in second
    restored = SubTaskProposal.model_validate({"items": plan.binding()["items"]})
    assert restored.items[0].alternatives == plan.items[0].alternatives         # the record's binding round-trips
    assert restored.fingerprint() == plan.fingerprint()
    assert plan.fingerprint() != _clear_leader().fingerprint()


def test_the_reply_forms_end_with_a_switch_plan_form_naming_the_alternatives_of_each_step():
    plan = _ambiguous()
    forms = reply_forms(prompt_for(plan), plan)
    assert forms[:2] == [{"kind": "accept", "reply": "yes"}, {"kind": "edit", "free_text": True, "per_item": False}]
    assert forms[-1] == {"kind": "switch_plan", "pattern": "step N: use NAME", "per_item": True, "separator": ";",
                         "alternatives": {"1": ["B", "C"]}}
    structured = reply_forms(prompt_for(plan, structured=True), plan, structured=True)
    assert structured[-1]["kind"] == "switch_plan" and structured[-1]["alternatives"] == {"1": ["B", "C"]}


def test_the_span_record_is_names_only():
    plan = _ambiguous()
    # The span's record is built by ``resolution.finish`` from the questions asked; the draft itself names the others.
    assert [alternative.name for alternative in plan.items[0].alternatives] == ["B", "C"]
    assert "subtask_plan_ambiguity" in tracing.SPAN_CONTRACTS[tracing.SPAN_TURN].attributes
    assert tracing.SPAN_CONTRACTS[tracing.SPAN_TURN].version == 5


def test_a_hidden_field_cannot_be_edited_by_hand():
    plan = _ambiguous()
    with pytest.raises(re_.EditRefused):
        apply_edits(plan, parse_edits("change 1: alternatives=B"))


# ----------------------------------------------------------------------
# Part 4: the reply reader, policy level (no model)
# ----------------------------------------------------------------------


def _single() -> SubTaskProposal:
    """One ambiguous step, with real planning-command names: A is book_by_title, B author_books, C book_loans."""
    return _proposal(["Who wrote Emma and how many copies"], {1: (BY_TITLE, [
        (AUTHOR, "Name each book by an author"), (LOANS, "Say who has a book on loan")])})


def _two() -> SubTaskProposal:
    """Two ambiguous steps."""
    return _proposal(["Goal one", "Goal two"], {1: ("A", ["B", "C"]), 2: ("D", ["E", "F"])})


def _edit(decision, step=1, name=AUTHOR):
    assert decision.verdict is Verdict.EDIT, (decision.verdict, decision.message)
    assert decision.edits == (Change(index=step, field="means", value=f"planning:{name}"),)


@pytest.mark.parametrize(
    "reply",
    [
        "use Library/author_books", "use author_books", "Use AUTHOR_BOOKS", "use author books",
        "step 1: use author_books", "Step 1 - use author_books", "step 1, use author books", "item 1: use author_books",
        "use author_books for step 1", "use author_books in step 1", "switch step 1 to author_books",
        "Switch step 1 to Library/author_books.", "switch to author_books", "author_books instead",
        "Library/author_books instead", "use the author_books plan", "use 'author_books'", "use `author_books`",
        "please use author_books", "no, use author_books",
        "use author_book",                        # fuzzy: the intent clarification's rule, a name of 4+ characters
    ],
)
def test_each_reply_form_switches_the_one_ambiguous_step(reply):
    _edit(ProposalReplyPolicy().decide(_single(), reply))


def test_a_switch_is_validated_as_any_edit_is_and_applies_to_a_copy():
    plan = _single()
    decision = ProposalReplyPolicy().decide(plan, "step 1: use author_books")
    edited = apply_edits(plan, list(decision.edits))
    assert edited.items[0].means == Means(kind="planning", name=AUTHOR)
    assert plan.items[0].means == Means(kind="planning", name=BY_TITLE)


def test_a_name_that_is_not_among_the_steps_alternatives_is_not_read_here():
    plan = _single()
    seen: list[str] = []

    def reader(text, proposal):
        seen.append(text)
        return []

    # A bare form (no step number) whose name resolves to nothing is not a switch clause: the existing reader reads it.
    # (The numbered forms, "step 1: use something_else", are decided here and never reach it: see Part 7.)
    for reply in ("use Library/count_books", "use the library", "use bo", "use it", "Library/count_books instead"):
        decision = ProposalReplyPolicy(reader).decide(plan, reply)
        assert decision.verdict is Verdict.REPROMPT and "which plan" not in decision.message.lower(), reply
    assert len(seen) == 5                                                     # every one went on to the existing reader


def test_the_name_of_the_plan_already_shown_is_not_an_alternative():
    seen: list[str] = []
    ProposalReplyPolicy(lambda text, proposal: seen.append(text) or []).decide(_single(), "use book_by_title")
    assert seen == ["use book_by_title"]


def test_a_fuzzy_name_that_fits_two_alternatives_asks_which():
    plan = _proposal(["Goal"], {1: ("Library/count_books", ["Library/book_loans", "Library/book_by_title"])})
    decision = ProposalReplyPolicy().decide(plan, "use book")
    assert decision.verdict is Verdict.REPROMPT
    assert "Which plan for step 1?" in decision.message
    assert "Library/book_by_title" in decision.message and "Library/book_loans" in decision.message


def test_two_ambiguous_steps_and_a_reply_that_does_not_say_which_asks_again_naming_both():
    decision = ProposalReplyPolicy().decide(_two(), "use B")
    assert decision.verdict is Verdict.REPROMPT
    assert "Which step do you mean?" in decision.message
    assert "Step 1: B, C." in decision.message and "Step 2: E, F." in decision.message
    assert 'Reply "step N: use NAME"' in decision.message
    assert ProposalReplyPolicy().decide(_two(), "E instead").verdict is Verdict.REPROMPT


def test_two_ambiguous_steps_with_the_step_named_switch_each_in_one_reply():
    policy = ProposalReplyPolicy()
    assert policy.decide(_two(), "step 2: use F").edits == (Change(index=2, field="means", value="planning:F"),)
    both = policy.decide(_two(), "step 1: use C; step 2: use E")
    assert both.verdict is Verdict.EDIT
    assert both.edits == (Change(index=1, field="means", value="planning:C"),
                          Change(index=2, field="means", value="planning:E"))
    assert policy.decide(_two(), "step 1: use B\nstep 2: use F").verdict is Verdict.EDIT


def test_a_name_that_belongs_to_the_other_ambiguous_step_is_not_read_here():
    seen: list[str] = []
    decision = ProposalReplyPolicy(lambda text, proposal: seen.append(text) or []).decide(_two(), "step 1: use F")
    assert seen == [] and decision.verdict is Verdict.REPROMPT          # a numbered clause is decided here (Part 7): asks again
    assert "Step 1" in decision.message or "step 1" in decision.message
    seen.clear()
    ProposalReplyPolicy(lambda text, proposal: seen.append(text) or []).decide(_two(), "use Z")      # fits neither step
    assert seen == ["use Z"]


def test_a_piece_that_is_not_a_switch_makes_the_whole_reply_not_ours():
    seen: list[str] = []
    reader = lambda text, proposal: seen.append(text) or []                        # noqa: E731
    reply = "step 1: use author_books; and also drop step 2"
    assert ProposalReplyPolicy(reader).decide(_single(), reply).verdict is Verdict.REPROMPT
    assert seen == ["and also drop step 2"]          # the switch clause is decided here; only the other clause is read freely


def test_two_different_plans_for_one_step_in_one_reply_ask_again():
    decision = ProposalReplyPolicy().decide(_single(), "step 1: use author_books; step 1: use book_loans")
    assert decision.verdict is Verdict.REPROMPT and "two different plans" in decision.message


def test_a_typed_means_edit_still_works_and_is_not_read_as_a_switch():
    decision = ProposalReplyPolicy().decide(_single(), f"change 1: means=planning:{LOANS}")
    assert decision.verdict is Verdict.EDIT
    assert decision.edits == (Change(index=1, field="means", value=f"planning:{LOANS}"),)


# ----------------------------------------------------------------------
# Part 6a: the accept and decline paths are untouched
# ----------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["yes", "accept", "looks good", "go ahead", "Sounds good.", "Yes."])
def test_a_bare_accept_is_decided_before_a_plan_switch_is_read(reply, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(review_plan_switch, "read_plan_switch", lambda proposal, text, **kw: calls.append(text))

    def reader_that_must_not_run(text, proposal):
        raise AssertionError("the reader ran on an accept")

    decision = ProposalReplyPolicy(reader_that_must_not_run).decide(_single(), reply)
    assert decision.verdict is Verdict.CONFIRM
    assert calls == []


@pytest.mark.parametrize("reply", ["no", "cancel", "nope", "no thanks", "scratch that plan"])
def test_a_decline_is_decided_before_a_plan_switch_is_read(reply, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(review_plan_switch, "read_plan_switch", lambda proposal, text, **kw: calls.append(text))
    decision = ProposalReplyPolicy().decide(_single(), reply)
    assert decision.verdict is Verdict.DECLINE
    assert calls == []




# ----------------------------------------------------------------------
# Part 6a2: an accept word followed by a plan name is a switch, not an accept
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    ["yes use author_books", "ok use author_books", "Yes, use Library/author_books", "yes, author_books", "ok, author_books",
     "yes, but use author_books for step 1", "yes but switch step 1 to author_books", "sure, switch to author_books",
     "sure, author_books instead", "no use author_books", "no, author_books instead", "no, use author_books"],
)
def test_an_accept_word_followed_by_an_alternative_is_the_switch_and_a_no_followed_by_one_is_not_a_decline(reply):
    _edit(ProposalReplyPolicy().decide(_single(), reply))


@pytest.mark.parametrize(
    "reply",
    ["yes use Library/count_books", "yes, run that", "yes, please", "yes, go ahead", "ok, looks good", "sure, run it", "yes, bo",
     "yes, step 2: use author_books", "yes use it"],
)
def test_an_accept_word_followed_by_anything_else_is_the_accept_it_was(reply):
    """Including a name that is not an alternative of any step: ``yes use X`` accepts, as it did before this feature."""
    assert ProposalReplyPolicy().decide(_single(), reply).verdict is Verdict.CONFIRM


def test_an_accept_followed_by_a_name_on_two_ambiguous_steps_asks_which_or_takes_the_numbered_one():
    policy = ProposalReplyPolicy()
    asked = policy.decide(_two(), "yes use B")
    assert asked.verdict is Verdict.REPROMPT and "Which step do you mean?" in asked.message
    assert policy.decide(_two(), "yes, but use F for step 2").edits == (Change(index=2, field="means", value="planning:F"),)
    assert policy.decide(_two(), "yes use Z").verdict is Verdict.CONFIRM


def test_a_review_with_no_alternatives_never_reaches_the_accept_with_switch_check(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(review_plan_switch, "read_accept_with_switch", lambda proposal, text: calls.append(text))
    monkeypatch.setattr(review_plan_switch, "read_plan_switch", lambda proposal, text, **kw: calls.append(text))
    for reply in ("yes use B", "ok use B", "yes, B", "yes", "sure, switch to B"):
        assert ProposalReplyPolicy().decide(_clear_leader(), reply).verdict is Verdict.CONFIRM
    assert calls == []


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
    book_by_title, and the review of the whole turn opens with that plan bound (and no alternatives)."""
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
    assert plan.items[0].means == Means() and plan.items[0].alternatives == []
    span = [each for each in gated.sink.spans if each.name == tracing.SPAN_TURN][-1]
    assert span.attributes["subtask_plan_ambiguity"] is None


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
    span = [each for each in gated.sink.spans if each.name == tracing.SPAN_TURN][-1]
    assert span.attributes["subtask_plan_ambiguity"] is None


@pytest.mark.parametrize("reply", ["yes", "accept", "looks good"])
def test_an_accept_of_an_ambiguous_review_runs_the_winner_and_reads_no_switch(reply, gated, no_heads, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(review_plan_switch, "read_plan_switch", lambda proposal, text: calls.append(text))
    gated.script.steps = [("finish", {})] * 2
    out = gated.ctx.process_turn(reply)
    assert out.status is TurnStatus.COMPLETED and record.state == re_.CONFIRMED and calls == []
    assert review_proposals.accepted_subtask_proposal(gated.wf).items[0].means.name == BY_TITLE


def test_a_decline_of_an_ambiguous_review_runs_nothing(gated, no_heads, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    out = gated.ctx.process_turn("no")
    assert record.state == re_.DECLINED and out.status is TurnStatus.COMPLETED
