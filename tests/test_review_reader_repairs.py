"""The review's free-text reply reader asks less when the reply is clear (bd fix-ftdl.8.28).

1. A reply that opens with an accept and whose rest the edit reader turns into no edit, or only into edits that change
   nothing, is an accept (``yes, but only list them`` on a plan that only lists): decided from what the reader returns,
   not from a list of words, and never read back. A turn that changes data still takes only an explicit ``yes``.
2. Edits the reader gives that fail the review's checks (a field that names no input, an input the step does not have, a
   value outside its choices, a step number out of range) are read once more with the reason, before the user is asked;
   a whole-input edit is held to the step's input names as a one-input edit is; and the one step a reply says it is
   about overrides the model's step number.
3. An edit that copies one input's value into another input the reply never names is left out, and the read-back says so.
4. ``key=`` removes the input, and the review shows a field the duplicate rule leaves empty removed beside its note: the
   review shows what runs.

Integration: a temporary copy of ``planning_tutorial_workflow`` with one planning command added, the real review engine
and reader, and the planner's model scripted (``DummyLM``); the plan path runs on the head's recorded answers.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow.planning import planning_commands
from fastworkflow.review import review_engine as re_
from fastworkflow.review import review_proposals as rp
from fastworkflow.review.review_engine import CONFIRMED, REVISED, open_review, proposal_of, reply_to_proposal
from fastworkflow.review.review_proposals import Means, SubTask, SubTaskInput, SubTaskProposal

from tests.test_approval_binding import annotated_library, no_live_model, todo  # noqa: F401
from tests.test_plan_reply_policy import (  # noqa: F401
    _no_writes, _open_review_two_step, no_draft_extraction, planner, script_param_extraction)
from tests.test_round1_gate import gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"

#: A planning command with two marked names (the second optional) and a kind with declared choices.
LINKED = '''
from typing import Literal

from pydantic import BaseModel, Field


class Signature:
    """Show the records linked to one record."""

    class Input(BaseModel):
        name: str = Field(description="the record", json_schema_extra={"copied_from_command_text": True})
        kind: Literal["book", "member", "branch"] = Field(default="book", description="what kind of record it is")
        linked_name: str = Field(default="", description="one linked record to narrow to",
                                 json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def steps(self, input):
        return [f"show the records linked to {input.name}"]
'''

GOAL = "show the records linked to Dune and to Ann Lee"


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    folder = tmp_path / "linked_wf"
    shutil.copytree(TUTORIAL, folder, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (folder / "_commands" / "linked_records.py").write_text(LINKED, encoding="utf-8")
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "state")})
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")       # an edit of inputs is rechecked, never drafted again
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(str(folder), load_cached=False)
    yield fastworkflow.Workflow.create(str(folder), workflow_id_str=f"linked_{uuid.uuid4().hex[:8]}")
    fastworkflow.RoutingRegistry.clear_registry()


def _step(goal: str = GOAL, **inputs: str) -> SubTask:
    return SubTask(goal=goal, means=Means(kind="planning", name="linked_records"),
                   inputs=[SubTaskInput(name=name, value=value) for name, value in inputs.items()])


class _Reader(DummyLM):
    """The edit model, scripted: one ``{"edits": [...]}`` answer per call, in order; the prompts it was given are kept."""

    def __init__(self, *answers: list):
        super().__init__([{"edits": edits} for edits in answers])

    @property
    def prompts(self) -> list[str]:
        return ["\n".join(str(m.get("content", "")) for m in (entry.get("messages") or [])) for entry in self.history]


def _review(workflow, *steps: SubTask, reader: _Reader | None = None):
    """A sub-task review of *steps* on *workflow*, its free text read by *reader* (no plan workflow: the reader is asked
    directly, as when the review has no plan CME)."""
    host = SimpleNamespace(_app_workflow=workflow, _current_planner_lm=None)
    store = re_.ReviewStore()
    record = open_review(host, store, SubTaskProposal(items=list(steps)), surface=store.surface)
    if reader is not None:
        names = tuple(sorted(planning_commands.resolve_field_sources(workflow.folderpath).planning_commands))
        store.edit_reader = rp.model_edit_reader(planner_lm=reader, planning_names=names)
    return host, store, record


def _inputs(record, number: int = 1) -> dict:
    return {each.name: each.value for each in proposal_of(record).items[number - 1].inputs}


# ---- 1. an accept with a rider that changes nothing ----------------------------------------------------------------

@pytest.mark.parametrize("reply", ["yes, but only list them, do not change anything",
                                   "Yes, but I also need the records behind it, not only the count.",
                                   "looks good. Then show them to me"])
def test_an_accept_whose_rest_the_reader_reads_no_edit_in_is_an_accept(workflow, reply):
    reader = _Reader([])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, reply)
    assert review.done and record.state == CONFIRMED
    assert len(reader.prompts) == 1


def test_an_accept_whose_edits_change_nothing_is_an_accept_without_a_read_back(workflow):
    reader = _Reader([{"op": "change", "index": 1, "field": "kind", "value": "book"}])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, "Yes, and keep it to books.")
    assert review.done and record.state == CONFIRMED
    assert "I read your reply as these edits" not in review.text


@pytest.mark.parametrize("reply", ["go back and look at it again", "Mostly right, but it is off"])
def test_a_reply_that_does_not_open_with_an_accept_and_reads_as_no_edit_asks_again(workflow, reply):
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=_Reader([]))
    review = reply_to_proposal(host, store, record, reply)
    assert not review.done and record.state == re_.AWAITING
    assert review.text == re_.SUB_TASK_UNREADABLE_REPLY


def test_a_rider_accept_on_a_turn_that_changes_data_still_needs_the_explicit_yes(workflow, monkeypatch):
    monkeypatch.setattr("fastworkflow.command_writes.write_commands", lambda path: frozenset({"linked_records"}))
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=_Reader([]))
    review = reply_to_proposal(host, store, record, "yes, but only list them, do not change anything")
    assert not review.done and record.state == re_.AWAITING
    assert review.text == re_.CHANGES_DATA_NEEDS_YES
    review = reply_to_proposal(host, store, record, "yes")
    assert review.done and record.state == CONFIRMED


reader_on_head = pytest.mark.usefixtures("initialized_fastworkflow", "no_live_model", "no_draft_extraction",
                                         "recorded_review_router")


@reader_on_head
def test_on_the_plan_path_a_rider_the_reader_finds_no_edit_in_is_an_accept(gated, monkeypatch, planner):
    _no_writes(monkeypatch)
    planner(gated, {"edits": []})
    record = _open_review_two_step(gated, monkeypatch)
    review = reply_to_proposal(gated.ctx, gated.store, record, "Yes, but also list which findings are driving her risk")
    assert review.done and record.state == CONFIRMED


@reader_on_head
def test_on_the_plan_path_a_rider_the_head_reads_as_a_decline_is_an_accept(gated, monkeypatch, planner,
                                                                          recorded_review_router):
    # The head's recorded decline ("hell no") stands in for its read of a rider it took for a decline: the polarity guard
    # does not act on it, and the reader finds no edit in the reply.
    reply = "yes, go ahead, but do not apply any fix"
    recorded_review_router.answers[reply] = recorded_review_router.answers["hell no"]
    _no_writes(monkeypatch)
    planner(gated, {"edits": []})
    record = _open_review_two_step(gated, monkeypatch)
    review = reply_to_proposal(gated.ctx, gated.store, record, reply)
    assert review.done and record.state == CONFIRMED


@reader_on_head
def test_on_the_plan_path_a_rider_on_a_writing_turn_asks_for_the_yes(gated, monkeypatch, planner):
    planner(gated, {"edits": []})
    record = _open_review_two_step(gated, monkeypatch)
    names = {item.means.name for item in proposal_of(record).items if item.means.name}
    monkeypatch.setattr("fastworkflow.command_writes.write_commands", lambda path: frozenset(names))
    review = reply_to_proposal(gated.ctx, gated.store, record, "Yes, but also list which findings are driving her risk")
    assert not review.done and record.state != CONFIRMED
    assert re_.CHANGES_DATA_SENTENCE in review.text


# ---- 2. one more read when the reader's edits cannot be applied --------------------------------------------------

@pytest.mark.parametrize("first, why", [
    ({"op": "change", "index": 1, "field": "", "value": "member"}, "names no field"),
    ({"op": "change", "index": 1, "field": "inputs", "value": "people=Dune"}, "has no input 'people'"),
    ({"op": "change", "index": 1, "field": "inputs", "value": "member"}, "names no input"),
    ({"op": "change", "index": 1, "field": "kind", "value": "group, member"}, "kind must be one of"),
    ({"op": "change", "index": 3, "field": "kind", "value": "member"}, "There are only 1 step"),
])
def test_edits_that_cannot_be_applied_are_read_once_more_with_the_reason(workflow, first, why):
    reader = _Reader([first], [{"op": "change", "index": 1, "field": "kind", "value": "member"}])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, "Dune is a member, not a book")
    assert record.state == REVISED and review.text.startswith("I read your reply as these edits")
    assert _inputs(record) == {"name": "Dune", "kind": "member"}
    assert len(reader.prompts) == 2
    assert "cannot be applied" in reader.prompts[1] and why in reader.prompts[1]


def test_a_second_reading_that_cannot_be_applied_either_asks_the_user(workflow):
    bad = {"op": "change", "index": 1, "field": "inputs", "value": "people=Dune"}
    reader = _Reader([bad], [bad])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, "Dune is a member, not a book")
    assert not review.done and record.state == re_.AWAITING
    assert _inputs(record) == {"name": "Dune", "kind": "book"}
    assert len(reader.prompts) == 2


def test_edits_that_apply_are_read_once(workflow):
    reader = _Reader([{"op": "change", "index": 1, "field": "kind", "value": "member"}])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    reply_to_proposal(host, store, record, "Dune is a member, not a book")
    assert record.state == REVISED and _inputs(record)["kind"] == "member" and len(reader.prompts) == 1


def test_the_one_step_a_reply_names_overrides_the_models_step_number(workflow):
    reader = _Reader([{"op": "change", "index": 1, "field": "name", "value": "Emma"}])
    host, store, record = _review(workflow, _step(goal="show the records linked to Dune", name="Dune"),
                                  _step(goal="show the records linked to Emma", name="Persuasion"), reader=reader)
    reply_to_proposal(host, store, record, "Step 2's name should be Emma.")
    assert _inputs(record, 1)["name"] == "Dune" and _inputs(record, 2)["name"] == "Emma"


@pytest.mark.parametrize("reply", ["Step 1 is right. The second one should be for Emma",
                                   "use the name from step 1's search on the second one",
                                   "Step 1 should list them and step 2 should be for Emma"])
def test_a_reply_that_names_no_one_step_to_change_keeps_the_models_step_number(reply):
    proposal = SubTaskProposal(items=[SubTask(goal="list the books"), SubTask(goal="show one member")])
    edits = [rp.ProposedEdit(op="change", index=2, field="goal", value="show Emma")]
    assert [edit.index for edit in rp._explicit_step_of_reply(proposal, edits, reply)] == [2]


# ---- 3. a value moved into an input the reply never names ---------------------------------------------------------

def test_a_value_copied_into_an_input_the_reply_does_not_name_is_left_out_and_said(workflow):
    reader = _Reader([{"op": "change", "index": 1, "field": "name", "value": "Ann Lee"},
                      {"op": "change", "index": 1, "field": "kind", "value": "branch"}])
    host, store, record = _review(workflow, _step(name="Dune", kind="book", linked_name="Ann Lee"), reader=reader)
    review = reply_to_proposal(host, store, record, "Dune is a branch, not a book")
    assert record.state == REVISED
    assert _inputs(record) == {"name": "Dune", "kind": "branch", "linked_name": "Ann Lee"}
    first, note = review.text.split("\n")[:2]
    assert first.startswith("I read your reply as these edits")
    assert note == "I left name of step 1 as it was: your reply does not name it, and 'Ann Lee' is the linked_name."


def test_a_value_set_on_an_input_the_reply_names_stands(workflow):
    reader = _Reader([{"op": "change", "index": 1, "field": "linked_name", "value": "Dune"},
                      {"op": "change", "index": 1, "field": "name", "value": "Ann Lee"}])
    host, store, record = _review(workflow, _step(name="Dune", linked_name="Ann Lee"), reader=reader)
    review = reply_to_proposal(host, store, record, "Swap them: the name is Ann Lee and the linked name is Dune")
    assert _inputs(record) == {"name": "Ann Lee", "linked_name": "Dune"}
    assert "I left" not in review.text


def test_a_reply_whose_only_edit_is_a_moved_value_is_asked_about(workflow):
    reader = _Reader([{"op": "change", "index": 1, "field": "name", "value": "Ann Lee"}])
    host, store, record = _review(workflow, _step(name="Dune", linked_name="Ann Lee"), reader=reader)
    review = reply_to_proposal(host, store, record, "that is not the record I meant")
    assert not review.done and record.state == re_.AWAITING
    assert review.text.startswith("I left name of step 1 as it was")
    assert _inputs(record) == {"name": "Dune", "linked_name": "Ann Lee"}


# ---- 4. the review shows what runs ---------------------------------------------------------------------------------

@pytest.mark.parametrize("reply", ["change 1: inputs=linked_name=", "change 1: linked_name="])
def test_an_empty_value_removes_the_input_and_the_review_shows_it_removed(workflow, reply):
    host, store, record = _review(workflow, _step(name="Dune", kind="book", linked_name="Ann Lee"))
    review = reply_to_proposal(host, store, record, reply)
    assert record.state == REVISED
    assert _inputs(record) == {"name": "Dune", "kind": "book"}
    assert "linked_name" not in review.text


def test_clearing_a_refused_value_the_step_does_not_show_is_read_back_without_the_refusal(workflow):
    """The draft's refused value is not shown; a reply that clears it changes the steps' text in nothing, but it is not
    a no-op: the refusal goes, and the reply is read back as an edit."""
    goal = "show the records linked to Payroll"
    step = _step(goal=goal, name="Payroll").model_copy(update={
        "check": "linked_name='Payroll' is also the value of name", "check_refused": ["linked_name"]})
    reader = _Reader([{"op": "change", "index": 1, "field": "inputs", "value": "linked_name="}])
    host, store, record = _review(workflow, step, reader=reader)
    review = reply_to_proposal(host, store, record, "there is no separate linked name, leave that blank")
    assert record.state == REVISED and review.text.startswith("I read your reply as these edits")
    assert "Cannot run" not in review.text and _inputs(record) == {"name": "Payroll"}


def test_a_field_the_duplicate_rule_leaves_empty_is_shown_removed_beside_its_note(workflow):
    goal = "show the records linked to Payroll"
    item = rp.rechecked(workflow, _step(goal=goal, name="Payroll", linked_name="Payroll"))
    note = "linked_name left empty: 'Payroll' is written once in your request and is the name."
    assert {each.name: each.value for each in item.inputs} == {"name": "Payroll"}
    assert item.check_notes == {"linked_name": note}
    rendering = SubTaskProposal(items=[item]).user_rendering(workflow.folderpath)
    assert "(name: Payroll)" in rendering and f"   {note}" in rendering
    # what runs: the run's check on the values the review showed before gives the same empty field
    command = planning_commands.planning_command(workflow.folderpath, "linked_records")
    ran, _ = planning_commands.check_arguments(command, {"name": "Payroll", "linked_name": "Payroll"}, goal)
    assert ran["linked_name"] == "" and ran["name"] == "Payroll"


def test_an_edit_that_makes_a_duplicate_shows_the_field_removed_with_the_note(workflow):
    goal = "show the records linked to Payroll"
    host, store, record = _review(workflow, _step(goal=goal, name="Payroll"))
    review = reply_to_proposal(host, store, record, "change 1: linked_name=Payroll")
    assert record.state == REVISED and _inputs(record) == {"name": "Payroll"}
    assert "linked_name: Payroll" not in review.text
    assert "linked_name left empty: 'Payroll' is written once in your request and is the name." in review.text


def test_the_accept_lead_reads_only_an_accept_that_ends_before_the_rest():
    assert re_._accept_lead("yes, but do not apply any fix")
    assert re_._accept_lead("Yes. Then list the members")
    assert re_._accept_lead("ok and show the count")
    assert re_._accept_lead("yes")
    assert not re_._accept_lead("go back and change step 2")
    assert not re_._accept_lead("fine print the overdue ones")
    assert not re_._accept_lead("Mostly right, but step 1 must also count them")
    assert not re_._accept_lead("no, change the second one")

