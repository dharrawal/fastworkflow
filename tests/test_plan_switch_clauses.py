"""A plan-switch clause in a review reply is decided deterministically, whatever the plan is called (bd fix-1ma8.1).

Real components, no mocks; the seams scripted are the neighbouring tests' (Jev's match, the planner LM, the plan
head's recorded answers). Before this change the clause ``step N: use NAME`` was decided only when NAME was one of the
step's listed alternatives (the review no longer lists any: bd fix-gwr7); any other reply fell to the free-text reader, which folded "use NAME" into the goal of a
neighbouring step (a switch lost and another step damaged), or guessed a plan for "switch step 7 to the plan that shows
...". Part 1 is the policy (no workflow, then the library workflow's real command names), part 2 the guard on the
reader's output, part 3 the whole review in both reader wirings.
"""
from __future__ import annotations

import pytest

from fastworkflow.review import review_engine as re_
from fastworkflow.review.review_engine import Change, ProposalReplyPolicy, Verdict
from fastworkflow.review.review_proposals import (
    EditRefused, Means, ProposedEdit, SubTask, SubTaskProposal, _proposed_edits_to_engine)

from tests.test_approval_binding import annotated_library, no_live_model, todo  # noqa: F401
from tests.test_plan_reply_policy import (  # noqa: F401
    _wire_plan, planner, script_param_extraction)
from tests.test_plan_match_ambiguity import (  # noqa: F401
    AUTHOR, BY_TITLE, LOANS, _ambiguous_review, _items, _proposal, _single, no_heads)
from tests.test_round1_gate import EMMA, JOKE, gated, no_draft_extraction  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from fastworkflow.turn import TurnStatus

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _never_read(seen: list):
    def reader(text, proposal):
        seen.append(text)
        return []
    return reader


def _mixed() -> SubTaskProposal:
    """Step 1 is bound to a plan, steps 2 and 3 are plain goals."""
    return _proposal(["Who wrote Emma", "tell a joke", "count the books"], {1: BY_TITLE})


def _two() -> SubTaskProposal:
    return _proposal(["Goal one", "Goal two"], {1: "A", 2: "D"})


# ----------------------------------------------------------------------
# Part 1: the policy
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    ["step 1: use something_else", "use something_else for step 1", "switch step 1 to something_else",
     "Step 1 - use No Such Plan", "step 1: use Library/count_books"],     # no workflow here: no name resolves
)
def test_a_name_that_resolves_to_nothing_asks_again_and_names_the_step(reply):
    seen: list = []
    plan = _single()
    decision = ProposalReplyPolicy(_never_read(seen)).decide(plan, reply)
    assert decision.verdict is Verdict.REPROMPT and seen == []
    assert "step 1" in decision.message.lower()
    assert 'step 1: use <plan name>' in decision.message and "say what step 1 should do" in decision.message
    assert plan.items[0].means.name == BY_TITLE                                    # the plan is unchanged


def test_a_name_that_resolves_to_nothing_is_asked_about_without_a_list():
    seen: list = []
    decision = ProposalReplyPolicy(_never_read(seen)).decide(_mixed(), "step 2: use something_else")
    assert decision.verdict is Verdict.REPROMPT and seen == []
    assert "step 2" in decision.message and "use <plan name>" in decision.message and "say what step 2 should do" in decision.message
    assert BY_TITLE not in decision.message


def test_a_step_that_does_not_exist_asks_again_naming_the_valid_steps():
    seen: list = []
    decision = ProposalReplyPolicy(_never_read(seen)).decide(_mixed(), "step 9: use author_books")
    assert decision.verdict is Verdict.REPROMPT and seen == []
    assert "no step 9" in decision.message and "steps 1 to 3" in decision.message


@pytest.mark.parametrize(
    "reply",
    ["switch step 1 to the plan that shows who wrote it", "use a different plan for step 1", "wrong plan for step 1",
     "step 1 uses the wrong plan", "change the plan for step 1", "Please use another approach for step 1",
     "switch step 1 to a plan that lists the loans"],
)
def test_a_request_to_change_the_plan_without_naming_one_is_not_guessed(reply):
    seen: list = []
    plan = _single()
    decision = ProposalReplyPolicy(_never_read(seen)).decide(plan, reply)
    assert decision.verdict is Verdict.REPROMPT and seen == []                 # the reader never guessed a plan
    assert "say what step 1 should do" in decision.message
    assert plan.items[0].means.name == BY_TITLE


def test_an_unnamed_request_lists_nothing_and_asks_what_it_should_do():
    seen: list = []
    decision = ProposalReplyPolicy(_never_read(seen)).decide(_mixed(), "use a different plan for step 2")
    assert decision.verdict is Verdict.REPROMPT and seen == []
    assert "step 2" in decision.message and "say what step 2 should do" in decision.message
    assert BY_TITLE not in decision.message and AUTHOR not in decision.message


@pytest.mark.parametrize(
    "reply",
    ["change step 3 to show the nesting too", "make step 1 show the nesting too", "switch step 1 to show the loans as well",
     "use the 2024 data for step 1", "step 1 should also say who is nested under whom", "change 1: goal=show the plan hierarchy"],
)
def test_an_ordinary_goal_edit_still_reaches_the_reader(reply):
    seen: list = []
    ProposalReplyPolicy(_never_read(seen)).decide(_mixed(), reply)
    assert seen == [reply] or seen == []                                       # a typed edit never needed the reader
    if not reply.startswith("change 1:"):
        assert seen == [reply]


def test_a_mixed_reply_applies_each_clause_by_its_own_reader(gated):
    seen: list = []
    reply = "change 2: goal=tell a short joke; step 1: use Library/author_books; drop 3"
    decision = _policy(gated, _never_read(seen)).decide(_mixed(), reply)
    assert decision.verdict is Verdict.EDIT and seen == []
    by_field = {(type(each).__name__, getattr(each, "index", None), getattr(each, "field", None)): each for each in decision.edits}
    assert by_field[("Change", 1, "means")].value == f"planning:{AUTHOR}"
    assert by_field[("Change", 2, "goal")].value == "tell a short joke"
    assert any(type(each).__name__ == "Drop" and each.index == 3 for each in decision.edits)


def test_a_switch_clause_is_never_merged_into_the_goal_of_a_neighbouring_step():
    reply = "change 2: goal=tell a short joke; step 3: use something_else"
    seen: list = []
    decision = ProposalReplyPolicy(_never_read(seen)).decide(_mixed(), reply)
    assert decision.verdict is Verdict.REPROMPT and seen == []                 # asked about step 3; nothing applied
    assert "step 3" in decision.message


def test_a_free_text_clause_with_a_switch_clause_passes_only_the_remainder_to_the_reader(gated):
    seen: list = []
    reader = lambda text, proposal: (seen.append(text) or [Change(index=2, field="goal", value="tell a short joke")])   # noqa: E731
    reply = "make step 2 shorter; step 1: use Library/author_books"
    decision = _policy(gated, reader).decide(_mixed(), reply)
    assert seen == ["make step 2 shorter"]
    assert decision.verdict is Verdict.EDIT
    assert [(each.index, each.field, each.value) for each in decision.edits] == [
        (1, "means", f"planning:{AUTHOR}"), (2, "goal", "tell a short joke")]


def test_when_the_other_clauses_are_not_read_the_switch_is_not_applied_either(gated):
    seen: list = []
    decision = _policy(gated, _never_read(seen)).decide(_mixed(), "make step 2 shorter; step 1: use Library/author_books")
    assert decision.verdict is Verdict.REPROMPT and seen == ["make step 2 shorter"]


def test_a_typed_goal_that_runs_on_into_a_switch_clause_is_refused():
    decision = ProposalReplyPolicy().decide(_two(), "change 1: goal=show the nesting, step 2: use F")
    assert decision.verdict is Verdict.REFUSE and "step 2: use F" in decision.message


def test_the_accept_path_never_reaches_the_clause_reader(monkeypatch):
    from fastworkflow.review import review_plan_switch

    def boom(*args, **kwargs):
        raise AssertionError("read a switch clause")

    monkeypatch.setattr(review_plan_switch, "split_switch_clauses", boom)
    for plan in (_single(), _mixed(), SubTaskProposal(items=[SubTask(goal="a", means=Means())])):
        for reply in ("yes", "accept", "looks good", "no", "decline"):
            assert ProposalReplyPolicy().decide(plan, reply).verdict in (Verdict.CONFIRM, Verdict.DECLINE)


def test_a_reply_with_no_switch_clause_is_read_as_before(monkeypatch):
    from fastworkflow.review import review_plan_switch

    plan = SubTaskProposal(items=[SubTask(goal="a", means=Means()), SubTask(goal="b", means=Means())])
    assert review_plan_switch.split_switch_clauses(plan, "drop 2") is None
    assert review_plan_switch.split_switch_clauses(plan, "make step 2 shorter") is None
    assert ProposalReplyPolicy().decide(plan, "drop 2").verdict is Verdict.EDIT


# -- with the workflow's own command names ------------------------------


def _policy(gated, reader=None):
    return ProposalReplyPolicy(reader, app_workflow=gated.wf)


@pytest.mark.parametrize(
    "reply",
    ["step 1: use Library/count_books", "step 1: use count_books", "Step 1: use COUNT BOOKS", "use count_books for step 1",
     "switch step 1 to count_books"],
)
def test_each_numbered_switch_form_switches_the_step_to_a_planning_command_of_the_workflow(reply, gated):
    seen: list = []
    decision = _policy(gated, _never_read(seen)).decide(_single(), reply)
    assert decision.verdict is Verdict.EDIT and seen == []
    assert decision.edits == (Change(index=1, field="means", value="planning:Library/count_books"),)


def test_a_planning_command_switches_a_step_that_is_a_plain_goal(gated):
    seen: list = []
    decision = _policy(gated, _never_read(seen)).decide(_mixed(), "step 2: use Library/count_books")
    assert decision.verdict is Verdict.EDIT and seen == []
    assert decision.edits == (Change(index=2, field="means", value="planning:Library/count_books"),)


def test_an_execution_command_is_a_valid_plain_means(gated):
    decision = _policy(gated).decide(_mixed(), "step 3: use Library/find_book")
    assert decision.verdict is Verdict.EDIT
    assert decision.edits == (Change(index=3, field="means", value="plain:Library/find_book"),)


def test_a_name_that_is_no_command_of_the_workflow_asks_again(gated):
    seen: list = []
    decision = _policy(gated, _never_read(seen)).decide(_mixed(), "step 2: use not_a_library_plan")
    assert decision.verdict is Verdict.REPROMPT and seen == []
    assert "not_a_library_plan" in decision.message


def test_a_word_is_not_fuzzy_matched_to_a_command_of_the_workflow(gated):
    decision = _policy(gated).decide(_mixed(), "step 2: use library")           # would lead into library_hours by its prefix
    assert decision.verdict is Verdict.REPROMPT


@pytest.mark.parametrize("reply", ["use the library", "use count_books", "count_books instead", "switch to count_books"])
def test_a_form_with_no_step_number_is_left_to_the_reader(reply, gated):
    seen: list = []
    _policy(gated, _never_read(seen)).decide(_single(), reply)
    assert seen == [reply]


def test_a_name_that_fits_several_commands_asks_which(gated):
    decision = _policy(gated).decide(_mixed(), "step 2: use book")                # a prefix is not a name
    assert decision.verdict is Verdict.REPROMPT


# ----------------------------------------------------------------------
# Part 2: the guard on the reader's output
# ----------------------------------------------------------------------


def _read(plan, reply, edits):
    return _proposed_edits_to_engine(plan, [ProposedEdit(**each) for each in edits], reply_text=reply)


@pytest.mark.parametrize(
    ("reply", "goal"),
    [
        ("make step 1 show the nesting, step 2: use F", "Goal one, show the nesting, step 2: use F"),
        ("step 2: use F", "step 2: use F"),
        ("make step 1 show the nesting; use something_else", "Goal one; use something_else"),
        ("also this\\nuse something_else", "Goal one and use something_else"),
    ],
)
def test_a_reader_goal_that_is_or_ends_with_a_switch_clause_of_the_reply_is_refused(reply, goal):
    reply = reply.replace("\\n", "\n")
    with pytest.raises(EditRefused) as refused:
        _read(_two(), reply, [{"op": "change", "index": 1, "field": "goal", "value": goal}])
    assert "plan" in str(refused.value) and "step N: use <plan name>" in str(refused.value)


def test_an_added_step_that_is_a_switch_clause_is_refused():
    with pytest.raises(EditRefused):
        _read(_two(), "add after 1: x; step 2: use F", [{"op": "add", "after": 1, "value": "step 2: use F"}])


def test_a_goal_that_merely_mentions_use_is_not_refused():
    edits = _read(_two(), "make step 1 say how to use the loans", [
        {"op": "change", "index": 1, "field": "goal", "value": "Goal one and say how to use the loans"}])
    assert len(edits) == 1


# ----------------------------------------------------------------------
# Part 3: through a real review, in both reader wirings
# ----------------------------------------------------------------------


@pytest.fixture(params=["session reader", "plan metadata extraction workflow"])
def wiring(request, gated):
    if request.param == "plan metadata extraction workflow":
        _wire_plan(gated)
    return request.param


@pytest.mark.usefixtures("recorded_review_router")
def test_failure_a_a_valid_plan_in_a_mixed_reply_switches_and_damages_no_step(
        wiring, gated, no_heads, planner, monkeypatch):
    """Observed: ``change N: goal=...; step M: use NAME`` with a valid NAME the step did not list. The goal of step N is
    what the typed edit says, step M switches, and the planner is never asked."""
    _, record = _ambiguous_review(gated, monkeypatch)
    lm = planner(gated, {"edits": []})
    gated.ctx.process_turn("change 2: goal=tell me a short joke; step 1: use Library/count_books")
    first, second = _items(record)
    assert first.means == Means(kind="planning", name="Library/count_books") and first.goal == EMMA
    assert second.goal == "tell me a short joke" and "use" not in second.goal
    assert len(lm.history) == 0


@pytest.mark.usefixtures("recorded_review_router")
def test_a_valid_plan_on_a_plain_goal_step_switches_it(wiring, gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    lm = planner(gated, {"edits": []})
    gated.ctx.process_turn("step 2: use Library/count_books")
    first, second = _items(record)
    assert second.means == Means(kind="planning", name="Library/count_books") and second.goal == JOKE
    assert first.means.name == BY_TITLE and len(lm.history) == 0


@pytest.mark.usefixtures("recorded_review_router")
def test_an_unknown_name_asks_again_and_changes_nothing(wiring, gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    before = [item.model_dump() for item in _items(record)]
    lm = planner(gated, {"edits": []})
    out = gated.ctx.process_turn("step 2: use no_such_plan")
    assert out.status is TurnStatus.AWAITING_USER and record.state == re_.AWAITING
    assert [item.model_dump() for item in _items(record)] == before and len(lm.history) == 0
    assert "step 2" in out.answer and "no_such_plan" in out.answer


@pytest.mark.usefixtures("recorded_review_router")
def test_failure_b_an_unnamed_switch_request_is_asked_about_and_no_plan_is_guessed(wiring, gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    before = [item.model_dump() for item in _items(record)]
    lm = planner(gated, {"edits": [{"op": "change", "index": 1, "field": "means", "value": f"planning:{LOANS}"}]})
    out = gated.ctx.process_turn("switch step 1 to the plan that shows who wrote the book")
    assert record.state == re_.AWAITING and [item.model_dump() for item in _items(record)] == before
    assert len(lm.history) == 0                                                  # the reader was not asked to guess
    # Since bd fix-o1lg S4/S5 a live review carries no alternatives (the ambiguity was asked before it), so the question
    # names none; it still guesses no plan.
    assert "Which plan should step 1 use?" in out.answer and "step 1" in out.answer


@pytest.mark.usefixtures("recorded_review_router")
def test_an_ordinary_goal_edit_still_reaches_the_reader_in_a_review(wiring, gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    lm = planner(gated, {"edits": [{"op": "change", "index": 2, "field": "goal", "value": "tell me a short joke"}]})
    gated.ctx.process_turn("change step 2 to a shorter joke")
    assert len(lm.history) == 1
    assert _items(record)[1].goal == "tell me a short joke" and _items(record)[0].means.name == BY_TITLE


@pytest.mark.usefixtures("recorded_review_router")
def test_a_reader_that_copies_a_switch_clause_into_a_goal_is_refused_and_asks_again(wiring, gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    before = [item.model_dump() for item in _items(record)]
    planner(gated, {"edits": [{"op": "change", "index": 2, "field": "goal", "value": f"{JOKE}, use the library"}]})
    out = gated.ctx.process_turn("make step 2 shorter; use the library")
    assert record.state == re_.AWAITING and [item.model_dump() for item in _items(record)] == before
    assert "switch a plan" in out.answer


@pytest.mark.usefixtures("recorded_review_router")
def test_a_free_text_clause_and_a_switch_clause_in_one_reply(wiring, gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    lm = planner(gated, {"edits": [{"op": "change", "index": 2, "field": "goal", "value": "tell me a short joke"}]})
    gated.ctx.process_turn("make the joke shorter; step 1: use Library/count_books")
    first, second = _items(record)
    assert first.means.name == "Library/count_books" and second.goal == "tell me a short joke"
    assert len(lm.history) == 1
    assert "step 1: use" not in lm.history[0]["messages"][-1]["content"]          # only the free-text remainder was read


def test_a_goal_that_ends_in_a_bare_instead_phrase_is_not_a_copied_switch_clause():
    """``make it about trains instead`` is a goal edit the reader may write back whole (the item-slot tests read it)."""
    edits = _read(_two(), "make it about trains instead", [
        {"op": "change", "index": 1, "field": "goal", "value": "make it about trains instead"}])
    assert len(edits) == 1


# ----------------------------------------------------------------------
# Part 4: a plan name followed by a sentence in the same clause
# ----------------------------------------------------------------------

PROSE = "I want its members and what it is nested under."


@pytest.mark.parametrize(
    "reply",
    [f"step 1: use {AUTHOR}. {PROSE}", f"step 1: use author_books. {PROSE}", f"step 1: use author_books, {PROSE}",
     f"Step 1: use author_books - {PROSE}", f"step 1: use author_books because {PROSE}", f"step 1: use author_books! {PROSE}",
     f"use author_books for step 1. {PROSE}", f"switch step 1 to author_books. {PROSE}", f"step 1: use author_books so {PROSE}"],
)
def test_an_explanation_after_the_plan_name_is_ignored_and_the_switch_applies(reply, gated):
    seen: list = []
    decision = _policy(gated, _never_read(seen)).decide(_single(), reply)
    assert decision.verdict is Verdict.EDIT and seen == []
    assert decision.edits == (Change(index=1, field="means", value=f"planning:{AUTHOR}"),)


def test_the_name_must_resolve_for_the_tail_to_be_peeled_off():
    seen: list = []
    decision = ProposalReplyPolicy(_never_read(seen)).decide(_single(), f"step 1: use something_else. {PROSE}")
    assert decision.verdict is Verdict.REPROMPT and seen == []
    assert "something_else" in decision.message


def test_a_leading_word_is_not_fuzzy_matched_when_a_sentence_follows(gated):
    seen: list = []
    decision = _policy(gated, _never_read(seen)).decide(_single(), f"step 1: use author_book. {PROSE}")
    assert decision.verdict is Verdict.REPROMPT and seen == []


def test_a_trailing_edit_is_read_as_a_clause_of_its_own_and_merged(gated):
    seen: list = []
    reader = lambda text, proposal: (seen.append(text) or [Change(index=3, field="goal", value="count every book")])   # noqa: E731
    decision = _policy(gated, reader).decide(_mixed(), f"step 2: use {AUTHOR}. Also change step 3 to count every book")
    assert seen == ["Also change step 3 to count every book"]
    assert decision.verdict is Verdict.EDIT
    assert [(each.index, each.field, each.value) for each in decision.edits] == [
        (2, "means", f"planning:{AUTHOR}"), (3, "goal", "count every book")]


def test_a_trailing_typed_edit_is_applied_with_the_switch(gated):
    decision = _policy(gated).decide(_mixed(), f"step 2: use {AUTHOR}. drop 3")
    assert decision.verdict is Verdict.EDIT
    assert [type(each).__name__ for each in decision.edits] == ["Change", "Drop"]


def test_a_trailing_edit_that_is_not_read_asks_and_applies_nothing_and_folds_no_text_into_a_goal(gated):
    seen: list = []
    decision = _policy(gated, _never_read(seen)).decide(_mixed(), f"step 2: use {AUTHOR}. Also drop step 3")
    assert decision.verdict is Verdict.REPROMPT and seen == ["Also drop step 3"]


def test_an_explanation_after_a_name_of_the_workflow_switches_a_plain_goal_step(gated):
    seen: list = []
    decision = _policy(gated, _never_read(seen)).decide(_mixed(), f"step 3: use Library/count_books. {PROSE}")
    assert decision.edits == (Change(index=3, field="means", value="planning:Library/count_books"),) and seen == []


@pytest.mark.usefixtures("recorded_review_router")
def test_an_explanation_after_the_name_through_a_review_switches_and_changes_nothing_else(wiring, gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    lm = planner(gated, {"edits": []})
    gated.ctx.process_turn(f"step 1: use {AUTHOR}. {PROSE}")
    first, second = _items(record)
    assert first.means.name == AUTHOR and first.goal == EMMA and second.goal == JOKE and second.means.kind == "from_scratch"
    assert len(lm.history) == 0


@pytest.mark.usefixtures("recorded_review_router")
def test_a_trailing_edit_through_a_review_never_folds_text_into_a_goal(wiring, gated, no_heads, planner, monkeypatch):
    _, record = _ambiguous_review(gated, monkeypatch)
    planner(gated, {"edits": [{"op": "change", "index": 2, "field": "goal", "value": "tell me a short joke"}]})
    gated.ctx.process_turn(f"step 1: use {AUTHOR}. Also change step 2 to a short joke")
    first, second = _items(record)
    goals = [first.goal, second.goal]
    assert all("use" not in goal.lower().split() for goal in goals)
    assert (first.means.name == AUTHOR and second.goal == "tell me a short joke") or (
        first.means.name == BY_TITLE and second.goal == JOKE)               # both applied, or nothing applied
