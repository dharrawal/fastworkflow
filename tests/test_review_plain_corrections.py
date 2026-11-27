"""Plain-English corrections of a plan review (bd fix-1e5c).

In a large benchmark most replies to a plan review were accepted as shown, and the ones that corrected the plan in
plain English were often misapplied: ``drop steps 2, 3 and 4 and keep only step 1`` removed step 1, a reply naming two
steps was refused as 'named by more than one edit', and a name holding the word 'and' was cut short or copied into the
wrong field. This file holds one test per class of misapplication.

Part 1 runs the reader's real code on the edits a reader model returned for a reply: the proposal shown, the reply word
for word, the ``ProposedEdit`` list the reader returned (the span the reader's output is recorded as in
``tests/test_review_reader_guards.py``), through the real ``_proposed_edits_to_engine`` and ``apply_edits``. No model is
called and nothing is mocked. The names and goals are generic stand-ins; the shapes of the replies and of the reader's
mistakes are the observed ones. Part 2 sends replies through a real review (the plan metadata extraction workflow's reader, the
head's recorded answers, a scripted planner and parameter extractor).
"""
from __future__ import annotations

import pytest

from fastworkflow._workflows.plan_metadata_extraction import _item_validation
from fastworkflow.review.review_engine import EditRefused, apply_edits
from fastworkflow.review.review_proposals import (
    Means, ProposedEdit, SubTask, SubTaskInput, SubTaskProposal, _proposed_edits_to_engine, reconcile_dropped_items)
from fastworkflow.review.review_step_refs import bare_step_numbers, removal_reading, step_numbers


pytestmark = pytest.mark.usefixtures("review_nouns_workflow")


def _plan(*steps: tuple[str, str]) -> SubTaskProposal:
    """A proposal from ``(goal, "name=value, kind=value")`` pairs."""
    items = []
    for goal, inputs in steps:
        pieces = [SubTaskInput.from_text(text) for text in inputs.split(", ") if text] if inputs else []
        items.append(SubTask(goal=goal, inputs=pieces, means=Means(kind="from_scratch")))
    return SubTaskProposal(items=items)


def _four_step_plan() -> SubTaskProposal:
    return _plan(
        ("Recertify the Atlas application: owner, logins and risk", "application=Atlas"),
        ("Open the first published right of Atlas on its sensitivity", "name=Atlas, kind=right"),
        ("Drill down the Atlas_Admins group and its members", "group=Atlas_Admins"),
        ("Retrieve the risk of the Atlas application", "name=Atlas, kind=application"),
    )


def _read(proposal: SubTaskProposal, reply: str, edits: list[dict]) -> SubTaskProposal:
    """What the plan is after the reader's *edits* for *reply* go through the real guards and the engine."""
    engine = _proposed_edits_to_engine(proposal, [ProposedEdit(**each) for each in edits], reply_text=reply)
    return apply_edits(proposal, engine)


def _goals(proposal: SubTaskProposal) -> list[str]:
    return [item.goal.split(":")[0].split(" ")[1] for item in proposal.items]


# ----------------------------------------------------------------------
# Class 1: a list of steps is read whole (steps 2, 3 and 4 are three steps, not two)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "count", "expected"),
    [
        ("drop steps 2, 3 and 4", 5, [2, 3, 4]),
        ("remove steps 2, 3, 4 and 6", 8, [2, 3, 4, 6]),
        ("steps 2 and 3 and step 5 are wrong", 6, [2, 3, 5]),
        ("steps 1, 3 and 5 to 7", 8, [1, 3, 5, 6, 7]),
        ("Steps 5 to 9 are wrong", 9, [5, 6, 7, 8, 9]),
        ("step 2, step 3 and step 4", 5, [2, 3, 4]),
        ("steps 2 and 9", 5, [2]),
        ("the 3 logins", 5, []),
    ],
)
def test_a_list_of_steps_names_every_step_in_it(text, count, expected):
    assert step_numbers(text, count) == expected
    assert _item_validation.extract_step_numbers(text, count) == expected


def test_a_whole_list_is_read_but_a_reference_among_words_is_not_a_list():
    assert bare_step_numbers("2, 3 and 4", 5) == [2, 3, 4]
    assert bare_step_numbers("steps 2 to 4", 5) == [2, 3, 4]
    assert bare_step_numbers("step 1 - not step 2", 5) == []


def test_a_routed_drop_entry_that_holds_a_list_drops_every_step_in_it():
    from fastworkflow._workflows.plan_metadata_extraction._commands.ProposalReview.drop import Signature

    class _Workflow:
        context = {"plan_proposal": _four_step_plan()}

    parsed = Signature.Input.model_validate({"items": ["2, 3 and 4"]}, context={"workflow": _Workflow()})
    assert parsed.items == ["2", "3", "4"]


# ----------------------------------------------------------------------
# Class 2: 'keep only step 1' is not a removal of step 1
# ----------------------------------------------------------------------

KEEP_ONLY_REPLY = (
    "This is one recertification request. Step 1 already covers it all, so please drop steps 2, 3 and 4 and keep "
    "only step 1.")


def test_the_reading_of_a_removal_that_keeps_one_step():
    reading = removal_reading(KEEP_ONLY_REPLY, 4)
    assert reading.remove == (2, 3, 4)
    assert 1 in reading.keep and reading.keep_only and not reading.contradiction


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param([{"op": "drop", "index": 1}], id="reader dropped the step the reply keeps"),
        pytest.param([{"op": "drop", "index": 2}, {"op": "drop", "index": 4}], id="reader read the list short"),
        pytest.param([{"op": "drop", "indexes": [2, 4, 1]}], id="reader included the kept step"),
        pytest.param([{"op": "drop", "index": 1}, {"op": "drop", "index": 2}], id="one right, one wrong"),
    ],
)
def test_drop_steps_2_3_and_4_keep_only_step_1_keeps_step_1_and_drops_the_rest(returned):
    plan = _four_step_plan()
    applied = _read(plan, KEEP_ONLY_REPLY, returned)
    assert [item.goal for item in applied.items] == [plan.items[0].goal]


def test_a_removal_that_is_negated_is_a_keep():
    plan = _four_step_plan()
    reply = "Do not remove step 2, but drop step 4."
    applied = _read(plan, reply, [{"op": "drop", "index": 2}, {"op": "drop", "index": 4}])
    assert [item.goal for item in applied.items] == [plan.items[i].goal for i in (0, 1, 2)]


def test_a_reply_that_removes_and_keeps_one_step_is_asked_again_with_the_removal_read():
    plan = _four_step_plan()
    reply = "Drop steps 1, 2 and 3, but keep step 2."
    with pytest.raises(EditRefused) as asked:
        _read(plan, reply, [{"op": "drop", "index": 2}])
    assert "step 2" in str(asked.value) and "steps 1, 2 and 3" in str(asked.value)


def test_a_drop_the_reply_asks_for_and_keeps_is_refused_not_applied_to_another_step():
    plan = _four_step_plan()
    with pytest.raises(EditRefused) as asked:
        _read(plan, "Step 1 is fine, and so are steps 2 and 3.", [{"op": "drop", "index": 1}])
    assert "keep" in str(asked.value)


# ----------------------------------------------------------------------
# Class 3: every step a reply names is removed
# ----------------------------------------------------------------------


def test_a_reply_that_removes_two_steps_removes_both_when_the_reader_returned_one():
    plan = _four_step_plan()
    reply = "Steps 3 and 4 repeat steps 1 and 2. Please remove steps 3 and 4; everything else is fine."
    applied = _read(plan, reply, [{"op": "drop", "index": 3}])
    assert [item.goal for item in applied.items] == [plan.items[0].goal, plan.items[1].goal]


def test_a_reply_that_removes_a_list_of_steps_removes_the_whole_list():
    plan = _four_step_plan()
    reply = "Please drop steps 2, 3 and 4; they only repeat step 1."
    applied = _read(plan, reply, [{"op": "drop", "index": 2}, {"op": "drop", "index": 4}])
    assert [item.goal for item in applied.items] == [plan.items[0].goal]


def test_a_removal_aimed_at_a_setting_of_a_step_is_not_a_removal_of_the_step():
    plan = _plan(
        ("Audit the Atlas right across systems", "right=Atlas_Admin, include_risk=true"),
        ("Walk the Birch group", "group=Birch"),
    )
    reply = "In step 1 I did not ask for risk, so drop the risk setting."
    applied = _read(plan, reply, [{"op": "drop", "index": 1}])
    assert len(applied.items) == 2 and "include_risk" not in {each.name for each in applied.items[0].inputs}


def test_the_routed_drop_is_held_to_the_numbers_of_the_reply():
    plan = _four_step_plan()
    assert reconcile_dropped_items(plan, ["2", "4"], KEEP_ONLY_REPLY) == [2, 3, 4]
    assert reconcile_dropped_items(plan, ["1"], "drop step 1") == [1]
    assert reconcile_dropped_items(plan, ["4"], "drop the last step") == [4]
    with pytest.raises(EditRefused):
        reconcile_dropped_items(plan, ["1"], "Step 1 is fine, so are steps 2 and 3.")
    # a step the extractor added that the reply never names is not removed
    assert reconcile_dropped_items(plan, ["2", "3"], "please drop step 2") == [2]


# ----------------------------------------------------------------------
# Class 4: a reply that names two things for one step is not 'named by more than one edit'
# ----------------------------------------------------------------------


def _split_name_plan() -> SubTaskProposal:
    return _plan(
        ("Which loans does Kline have", "name=Kline, kind=person"),
        ("Which loans does Sons Ltd have", "name=Sons Ltd, kind=person"),
    )


def test_make_it_one_step_and_drop_the_second_is_one_merge_not_two_edits_on_step_2():
    plan = _split_name_plan()
    reply = "No, that is one name. Make it a single step for 'Kline and Sons Ltd' and drop the second step."
    returned = [
        {"op": "merge", "indexes": [1, 2], "value": "Which loans does Kline and Sons Ltd have"},
        {"op": "drop", "index": 2},
    ]
    applied = _read(plan, reply, returned)
    assert len(applied.items) == 1
    assert "Kline and Sons Ltd" in applied.items[0].goal


def test_a_removal_of_a_step_with_a_clear_of_one_of_its_inputs_is_one_removal():
    plan = _plan(
        ("Who is Lee", "person=Lee"),
        ("Tag Lee's record and show the tags", "person=Lee, tag=sensitive"),
    )
    reply = "Remove step 2, it is only a heading and not a real task. Step 1 is fine."
    returned = [{"op": "drop", "index": 2}, {"op": "change", "index": 2, "field": "inputs", "value": "tag=none"}]
    applied = _read(plan, reply, returned)
    assert [item.goal for item in applied.items] == ["Who is Lee"]


def test_an_input_the_reader_both_clears_and_sets_is_set():
    plan = _plan(("Tag Lee's record", "person=Lee"), ("Walk the Birch group", "group=Birch"))
    reply = "In step 1 the person is 'Lee Park', not Lee."
    returned = [
        {"op": "change", "index": 1, "field": "inputs", "value": "person=none"},
        {"op": "change", "index": 1, "field": "inputs", "value": "person=Lee Park"},
    ]
    applied = _read(plan, reply, returned)
    assert {each.name: each.value for each in applied.items[0].inputs} == {"person": "Lee Park"}


# ----------------------------------------------------------------------
# Class 5: a name that holds 'and' stays one name
# ----------------------------------------------------------------------


def test_completing_a_cut_short_name_in_a_goal_does_not_replace_the_goal_by_the_name():
    plan = _split_name_plan()
    reply = "No, that is one name. Make it one step: which loans does 'Kline and Sons Ltd' have?"
    returned = [
        {"op": "merge", "indexes": [1, 2], "value": "Which loans does Kline and Sons have"},
    ]
    applied = _read(plan, reply, returned)
    assert len(applied.items) == 1
    assert applied.items[0].goal == "Which loans does Kline and Sons Ltd have"


def test_a_piece_of_a_name_is_not_set_as_the_whole_name_and_the_user_is_asked_for_it():
    plan = _split_name_plan()
    reply = "Remove step 2; 'Sons Ltd' is part of the name in step 1, not a separate entity."
    with pytest.raises(EditRefused) as asked:
        _read(plan, reply, [{"op": "drop", "index": 2}])
    text = str(asked.value)
    assert "Sons Ltd" in text and "whole name" in text and "drop 2" in text


def test_a_quoted_name_with_and_is_one_value():
    plan = _plan(("Which loans does Kline have", "name=Kline, kind=person"), ("Walk the Birch group", "group=Birch"))
    reply = "In step 1 the name is 'Kline and Sons Ltd'."
    applied = _read(plan, reply, [{"op": "change", "index": 1, "field": "inputs", "value": "name=Kline"}])
    assert {each.name: each.value for each in applied.items[0].inputs}["name"] == "Kline and Sons Ltd"


def test_a_quoted_name_that_is_already_the_value_of_another_input_is_not_set_as_the_name():
    plan = _plan(("Retrieve the risk of Payroll for Mara Lind", "name=Payroll, person=Mara Lind, kind=unit"))
    reply = "Make the kind application for 'Mara Lind' in step 1."
    applied = _read(plan, reply, [{"op": "change", "index": 1, "field": "inputs", "value": "kind=application"}])
    values = {each.name: each.value for each in applied.items[0].inputs}
    assert values["name"] == "Payroll"
    assert values["kind"] == "application"


def test_a_field_less_change_that_repeats_a_change_naming_its_field_is_dropped_not_an_error():
    plan = _four_step_plan()
    returned = [
        {"op": "change", "index": 2, "field": "kind", "value": "application"},
        {"op": "change", "index": 2, "field": "", "value": "application"},
    ]
    applied = _read(plan, "Make step 2 an application.", returned)
    assert {each.name: each.value for each in applied.items[1].inputs}["kind"] == "application"


def test_a_field_less_change_with_no_twin_still_raises():
    from fastworkflow.review.review_engine import EditParseError
    plan = _four_step_plan()
    with pytest.raises(EditParseError, match="names no field"):
        _read(plan, "Make step 2 an application.", [{"op": "change", "index": 2, "field": "", "value": "application"}])
    with pytest.raises(EditParseError, match="names no field"):
        _read(plan, "Make step 2 an application.", [
            {"op": "change", "index": 2, "field": "kind", "value": "application"},
            {"op": "change", "index": 2, "field": "", "value": "other"}])


# ----------------------------------------------------------------------
# Class 6: a removal verb that is not a removal of the step (review round 1)
# ----------------------------------------------------------------------


def _filter_plan() -> SubTaskProposal:
    return _plan(
        ("List the open findings", "scope=open"),
        ("List the logins of Atlas with a status filter", "application=Atlas, status_filter=active"),
        ("Show the controls of Atlas for the date filter", "application=Atlas, date_filter=this_year"),
        ("Summarise the risk of Atlas", "application=Atlas"),
    )


def _intact(applied: SubTaskProposal, plan: SubTaskProposal) -> bool:
    return [item.goal for item in applied.items] == [item.goal for item in plan.items]


#: (reply, step the reply is about, what a reader that got it right returns)
NOT_A_REMOVAL = [
    ("Please remove step 2's status filter.", 2, {"op": "change", "index": 2, "field": "inputs", "value": "status_filter=none"}),
    ("drop step 3\u2019s limit", 3, {"op": "change", "index": 3, "field": "goal", "value": "Show the controls of Atlas"}),
    ("cut step 3 down to five results", 3, {"op": "change", "index": 3, "field": "goal", "value": "Show five controls of Atlas"}),
    ("cut step 2 to the last 30 days", 2, {"op": "change", "index": 2, "field": "goal", "value": "List the logins of Atlas in the last 30 days"}),
    ("skip step 2 if there are no results", 2, {"op": "change", "index": 2, "field": "goal", "value": "List the logins of Atlas, skipped when empty"}),
    ("remove step 4 only if step 3 finds nothing", 4, {"op": "change", "index": 4, "field": "goal", "value": "Summarise the risk of Atlas if step 3 finds nothing"}),
    ("remove step 2 from the merge", 2, {"op": "change", "index": 2, "field": "goal", "value": "List the logins of Atlas outside the merge"}),
    ("take step 4 out of the loop", 4, {"op": "change", "index": 4, "field": "goal", "value": "Summarise the risk of Atlas once"}),
]


@pytest.mark.parametrize(("reply", "step", "right"), NOT_A_REMOVAL, ids=[each[0] for each in NOT_A_REMOVAL])
def test_a_verb_that_is_not_a_removal_of_the_step_does_not_make_a_certain_removal(reply, step, right):
    reading = removal_reading(reply, 4)
    assert reading.remove == () and step in reading.uncertain


@pytest.mark.parametrize(("reply", "step", "right"), NOT_A_REMOVAL, ids=[each[0] for each in NOT_A_REMOVAL])
def test_a_reader_that_read_it_right_is_left_alone(reply, step, right):
    plan = _filter_plan()
    applied = _read(plan, reply, [right])
    assert len(applied.items) == 4


@pytest.mark.parametrize(("reply", "step", "right"), NOT_A_REMOVAL, ids=[each[0] for each in NOT_A_REMOVAL])
def test_a_reader_that_dropped_the_step_is_asked_about_it_not_obeyed(reply, step, right):
    plan = _filter_plan()
    with pytest.raises(EditRefused) as asked:
        _read(plan, reply, [{"op": "drop", "index": step}])
    assert f"step {step}" in str(asked.value) and f"drop {step}" in str(asked.value)


def test_drop_step_3_into_step_2_is_a_merge_not_a_removal():
    plan = _filter_plan()
    reply = "drop step 3 into step 2"
    assert removal_reading(reply, 4).remove == ()
    applied = _read(plan, reply, [{"op": "merge", "indexes": [2, 3], "value": "Logins and controls of Atlas"}])
    assert len(applied.items) == 3
    with pytest.raises(EditRefused):
        _read(plan, reply, [{"op": "drop", "index": 3}])


def test_removing_the_date_filter_in_a_step_is_not_a_removal_of_the_step():
    plan = _filter_plan()
    reply = "remove the date filter in step 3"
    reading = removal_reading(reply, 4)
    assert reading.remove == ()
    right = _read(plan, reply, [{"op": "change", "index": 3, "field": "inputs", "value": "date_filter=none"}])
    assert len(right.items) == 4 and "date_filter" not in {each.name for each in right.items[2].inputs}
    # a reader that dropped the step: the setting is cleared instead (the reply names it), or the user is asked
    try:
        wrong = _read(plan, reply, [{"op": "drop", "index": 3}])
    except EditRefused:
        pass
    else:
        assert len(wrong.items) == 4 and "date_filter" not in {each.name for each in wrong.items[2].inputs}


def test_the_certain_removals_beside_an_uncertain_one_are_still_read():
    plan = _filter_plan()
    reply = "Drop step 4, and remove step 2's status filter."
    reading = removal_reading(reply, 4)
    assert reading.remove == (4,) and reading.uncertain == frozenset({2})
    applied = _read(plan, reply, [{"op": "drop", "index": 4}])
    assert [item.goal for item in applied.items] == [item.goal for item in plan.items[:3]]


@pytest.mark.parametrize(
    "reply",
    ["remove step 2 entirely", "remove step 2 from the plan", "drop step 2, it repeats step 1", "drop step 2 too",
     "remove step 2 because it repeats step 1", "drop step 2 and keep step 1", "Please remove step 2 only; the rest is fine.", "Please drop step 2 - it is a heading"],
)
def test_a_list_that_is_the_whole_object_of_the_verb_is_still_a_certain_removal(reply):
    plan = _filter_plan()
    reading = removal_reading(reply, 4)
    assert reading.remove == (2,) and not reading.uncertain
    applied = _read(plan, reply, [{"op": "change", "index": 1, "field": "goal", "value": "List the open findings"}])
    assert [item.goal for item in applied.items] == [plan.items[i].goal for i in (0, 2, 3)]


# ---- a negated keep is a removal -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "removed"),
    [("do not keep step 2", (2,)), ("Please don't keep step 2.", (2,)), ("no need to keep steps 3 and 4", (3, 4))],
)
def test_a_negated_keep_is_a_removal(reply, removed):
    plan = _filter_plan()
    reading = removal_reading(reply, 4)
    assert reading.remove == removed and not reading.keep
    applied = _read(plan, reply, [{"op": "drop", "index": removed[0]}])
    assert [item.goal for item in applied.items] == [
        plan.items[i].goal for i in range(4) if i + 1 not in removed]


# ---- 'fine' and 'leave' ------------------------------------------------------------------------------------------


def test_step_2_is_fine_but_drop_step_3_keeps_2_and_drops_3():
    plan = _filter_plan()
    reply = "step 2 is fine but drop step 3"
    reading = removal_reading(reply, 4)
    assert reading.keep == frozenset({2}) and reading.remove == (3,)
    applied = _read(plan, reply, [{"op": "drop", "index": 2}])
    assert [item.goal for item in applied.items] == [plan.items[i].goal for i in (0, 1, 3)]


@pytest.mark.parametrize("reply", ["steps 2 and 3 are not right", "steps 2 and 3 are not correct", "steps 2 and 3 aren't ok",
                                   "steps 2 and 3 are not fine"])
def test_steps_that_are_not_right_are_neither_kept_nor_removed(reply):
    plan = _filter_plan()
    reading = removal_reading(reply, 4)
    assert not reading.keep and not reading.remove and not reading.uncertain
    changes = [{"op": "change", "index": number, "field": "goal", "value": f"Redo step {number}"} for number in (2, 3)]
    applied = _read(plan, reply, changes)
    assert [item.goal for item in applied.items][1:3] == ["Redo step 2", "Redo step 3"]


def test_leave_step_3_out_is_a_removal_and_leave_it_as_is_is_a_keep_and_bare_leave_is_neither():
    plan = _filter_plan()
    assert removal_reading("leave step 3 out", 4).remove == (3,)
    assert removal_reading("leave out step 3", 4).remove == (3,)
    assert removal_reading("take step 3 out", 4).remove == (3,)
    for phrase in ("leave step 3 as is", "leave step 3 alone", "leave step 3 untouched"):
        reading = removal_reading(phrase, 4)
        assert reading.keep == frozenset({3}) and not reading.remove
    bare = removal_reading("leave step 3", 4)
    assert not bare.keep and not bare.remove and not bare.uncertain
    assert len(_read(plan, "leave step 3 out", [{"op": "drop", "index": 3}]).items) == 3
    with pytest.raises(EditRefused):
        _read(plan, "leave step 3 as is", [{"op": "drop", "index": 3}])


# ----------------------------------------------------------------------
# A first-reply accept never reaches the removal reading
# ----------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["yes", "accept", "looks good", "go ahead", "yes, run that", "Sounds good."])
def test_an_accept_is_decided_before_any_step_reference_is_read(reply, monkeypatch):
    """``ProposalReplyPolicy.decide`` returns CONFIRM for an accept before the typed-edit parser, the reader and every
    guard. A real reader that fails the test if it is called, and a counter over the reading, show nothing ran."""
    from fastworkflow.review import review_engine, review_proposals, review_step_refs

    calls: list[str] = []
    real = review_step_refs.removal_reading

    def counting(text, count):
        calls.append(text)
        return real(text, count)

    monkeypatch.setattr(review_proposals, "removal_reading", counting)
    monkeypatch.setattr(review_step_refs, "removal_reading", counting)

    def reader_that_must_not_run(text, proposal):
        raise AssertionError("the reader ran on an accept")

    policy = review_engine.ProposalReplyPolicy(reader_that_must_not_run)
    decision = policy.decide(_four_step_plan(), reply)
    assert decision.verdict is review_engine.Verdict.CONFIRM
    assert calls == []


# ----------------------------------------------------------------------
# Part 2: the same replies through a real review (the plan metadata extraction workflow reads them)
# ----------------------------------------------------------------------
#
# The head's answer for each reply is its recorded one (``recorded_review_router``, tests/recorded_router.py); the
# planner and the parameter extractor are scripted with what they returned in the benchmark (the planner read the
# reply as a drop of step 1; the extractor took step 2 only), which is the seam the other reader tests use.

from fastworkflow.review import review_engine as re_  # noqa: E402
from fastworkflow.review.review_engine import REVISED, open_review  # noqa: E402

from tests.test_approval_binding import annotated_library, no_live_model, todo  # noqa: E402,F401
from tests.test_plan_reply_policy import (  # noqa: E402,F401
    _PlannerLM, _wire_plan, no_draft_extraction, planner, script_param_extraction)
from tests.test_round1_gate import gated  # noqa: E402,F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: E402,F401


@pytest.fixture(params=["session reader", "plan fallback reader"])
def wiring(request, gated):
    """How the free-text reader is wired: the session's (``store.edit_reader``, as ``_give_review_reader`` sets it for a
    review the gate opened) or the plan ``edit`` command's own, used when the store has none."""
    if request.param == "session reader":
        from fastworkflow.review.review_proposals import model_edit_reader

        gated.store.edit_reader = model_edit_reader(planner_lm=lambda: getattr(gated.ctx, "_current_planner_lm", None))
    return request.param


def _open(gated, proposal: SubTaskProposal) -> re_.ReviewRecord:
    record = open_review(gated.ctx, gated.store, proposal, surface=gated.store.surface)
    _wire_plan(gated)
    return record


def _shown_goals(record: re_.ReviewRecord) -> list[str]:
    return [item.goal for item in re_.proposal_of(record).items]


@pytest.mark.usefixtures("initialized_fastworkflow", "no_live_model", "no_draft_extraction", "recorded_review_router")
class TestThroughTheReview:
    def test_keep_only_step_1_keeps_step_1(self, gated, planner, wiring):
        plan = _four_step_plan()
        record = _open(gated, plan)
        planner(gated, {"edits": [{"op": "drop", "index": 1}]})
        gated.ctx.process_turn(KEEP_ONLY_REPLY)
        assert record.state == REVISED
        assert _shown_goals(record) == [plan.items[0].goal]

    def test_removing_two_named_steps_removes_both(self, gated, planner, wiring):
        plan = _four_step_plan()
        record = _open(gated, plan)
        planner(gated, {"edits": [{"op": "drop", "index": 3}]})
        gated.ctx.process_turn("Steps 3 and 4 repeat steps 1 and 2. Please remove steps 3 and 4; everything else is fine.")
        assert record.state == REVISED
        assert _shown_goals(record) == [plan.items[0].goal, plan.items[1].goal]

    def test_one_step_and_drop_the_second_is_applied(self, gated, planner, wiring):
        record = _open(gated, _split_name_plan())
        planner(gated, {"edits": [
            {"op": "merge", "indexes": [1, 2], "value": "Which loans does Kline and Sons Ltd have"},
            {"op": "drop", "index": 2}]})
        gated.ctx.process_turn(
            "No, that is one name. Make it a single step for 'Kline and Sons Ltd' and drop the second step.")
        assert record.state == REVISED
        assert _shown_goals(record) == ["Which loans does Kline and Sons Ltd have"]

    def test_a_piece_of_a_name_is_asked_about_and_the_plan_is_unchanged(self, gated, planner, wiring):
        plan = _split_name_plan()
        record = _open(gated, plan)
        planner(gated, {"edits": [{"op": "drop", "index": 2}]})
        out = gated.ctx.process_turn(
            "Remove step 2; 'Sons Ltd' is part of the name in step 1, not a separate entity.")
        assert record.state == re_.AWAITING
        assert _shown_goals(record) == [item.goal for item in plan.items]
        assert "whole name" in (out.answer or "")


def test_a_reply_that_names_the_steps_needs_no_question_about_which():
    from fastworkflow.review.review_proposals import certain_removals

    plan = _four_step_plan()
    assert certain_removals(plan, "Please just remove step 2, the one called Drill down.") == [2]
    assert certain_removals(plan, "drop steps 2, 3 and 4 and keep only step 1") == [2, 3, 4]
    assert certain_removals(plan, "drop the last step") == []
    assert certain_removals(plan, "Drop steps 1 and 2, but keep step 2.") == []


@pytest.mark.usefixtures("initialized_fastworkflow", "no_live_model", "no_draft_extraction", "recorded_review_router")
class TestNotARemovalThroughTheReview:
    @pytest.mark.parametrize(
        ("reply", "step"),
        [("Please remove step 2's status filter.", 2), ("cut step 3 down to five results", 3),
         ("skip step 2 if there are no results", 2)],
    )
    def test_a_drop_of_a_step_the_reply_only_touches_is_asked_about_and_the_plan_is_unchanged(
            self, reply, step, gated, planner, wiring):
        plan = _filter_plan()
        record = _open(gated, plan)
        planner(gated, {"edits": [{"op": "drop", "index": step}]})
        out = gated.ctx.process_turn(reply)
        assert record.state == re_.AWAITING
        assert _shown_goals(record) == [item.goal for item in plan.items]
        # asked again: by the guard (it names the step and the way to drop it) or by the drop's own question
        assert f"drop {step}" in (out.answer or "") or "Which of the 4 steps do you mean?" in (out.answer or "")

    def test_a_possessive_the_head_reads_as_a_drop_is_asked_about_and_the_plan_is_unchanged(
            self, gated, planner, wiring):
        """"Please remove step 2's status filter." is a change to step 2, not the removal of step 2.

        Before the framework's review head moved to the embedding layout it answered this reply with a list of
        candidates, the reader fell through to the text planner and applied the planner's (correct) change. The
        embedding head commits to ``drop`` (recorded answer), so the reader's code rule is what protects the user:
        nothing is removed and the user is asked again.
        The planner is scripted with the correct change to prove it is the question, not a wrong edit, that stops it."""
        plan = _filter_plan()
        record = _open(gated, plan)
        planner(gated, {"edits": [{"op": "change", "index": 2, "field": "inputs", "value": "status_filter=none"}]})
        out = gated.ctx.process_turn("Please remove step 2's status filter.")
        assert record.state == re_.AWAITING
        assert _shown_goals(record) == [item.goal for item in plan.items]
        assert "status_filter" in {each.name for each in re_.proposal_of(record).items[1].inputs}
        # asked again: by the guard (it offers 'drop 2' or a change to the step) or by the drop's own question
        assert "drop 2" in (out.answer or "") or "Which of the 4 steps do you mean?" in (out.answer or "")


@pytest.mark.usefixtures("initialized_fastworkflow", "no_live_model", "no_draft_extraction")
@pytest.mark.parametrize("reply", ["yes", "accept", "looks good", "go ahead"])
def test_a_first_reply_accept_through_the_review_is_confirmed_without_reading_steps(reply, gated, monkeypatch):
    from fastworkflow.review import review_proposals, review_step_refs

    calls: list[str] = []
    real = review_step_refs.removal_reading
    monkeypatch.setattr(review_proposals, "removal_reading", lambda text, count: calls.append(text) or real(text, count))
    # A workflow with no command that changes data: the plan's ad hoc steps then need no explicit yes (bd fix-ftdl.8.4),
    # so every accept here is read as an accept.
    monkeypatch.setattr("fastworkflow.core.command_writes.write_commands", lambda path: frozenset())
    record = _open(gated, _four_step_plan())
    gated.ctx.process_turn(reply)
    assert record.state == re_.CONFIRMED
    assert calls == []
