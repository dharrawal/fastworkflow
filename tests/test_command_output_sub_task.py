"""A recorded command output says which request part it was produced for (fix-2nsr).

``CommandOutput.sub_task`` is the 1-based number of the sub-task the framework executed the command under, the same
number the turn's answer gives that part in its ``### N.`` heading. It is a semantic fact about the execution, set from
the turn's execution state when the output is recorded (never parsed from text); it is not a display instruction.
Real library copy, real agent with a scripted model (the fixtures of tests/test_round1_gate.py and
tests/test_round1_handoff.py); the planner's split is the one seam faked, as in the other sub-task tests. No test calls
a model.

Execution is sequential: ``_run_subtasks_from`` runs one entry at a time and records its outputs before the next
starts. The per-sub-task model calls that run side by side (``subtask_pool``) are the referent, the match and the
argument draft, which run before any sub-task executes and record no output.
"""
from __future__ import annotations

import json
import re
import threading

import pytest

import fastworkflow
from fastworkflow.utils import subtask_pool
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import (  # noqa: F401 - the gated context and the gate's helpers
    EMMA, FIND, JOKE, _open_review, _split_into, gated, no_draft_extraction)
from tests.test_round1_handoff import FIND_TEXT, _planner_stub, _scripted_context  # noqa: F401
from tests.conftest import script_matcher
from tests.test_round1_gate import REMOTE_ANSWERS
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow", "recorded_review_router")

HOURS = ("execute_workflow_query", {"command": "library_hours"})
COUNT = ("execute_workflow_query", {"command": "count_books <query>Dune</query>"})
COUNT_NO_QUERY = ("execute_workflow_query", {"command": "count_books"})
ASK = ("ask_user", {"clarification_request": "Which one?"})
CATS, OWLS, DOGS = "tell Ann a joke about cats", "tell Lee a joke about owls", "tell Raj a joke about dogs"


def _remote_decides(monkeypatch):
    """The remote decision model's answers for the reviewed turns (``REMOTE_ANSWERS``; DOGS, a joke too, as JOKE),
    scripted with no model call. These tests used to get them from a live call through a developer's .env."""
    script_matcher(monkeypatch, {**REMOTE_ANSWERS, DOGS: REMOTE_ANSWERS[JOKE]})
THREE = [CATS, OWLS, DOGS]


@pytest.fixture(autouse=True)
def extraction_without_a_model(no_model_call_for_parameters, monkeypatch):
    """The seam of tests/test_command_output_framework_marker.py: the commands here take no argument (or carry their
    value in the command text), so the extraction hop returns an empty Input and no model is called (the guard it
    replaces would raise)."""
    from fastworkflow.parameters.signatures import InputForParamExtraction

    monkeypatch.setattr(InputForParamExtraction, "extract_parameters", lambda _self, cls, *a, **k: cls.model_construct())


def _ran(out):
    return [o for o in out.command_outputs if not o.is_ask_user]


def _by_part(out):
    return [(o.command_name, o.sub_task) for o in _ran(out)]


def _headings(answer):
    return {int(n): title for n, title in re.findall(r"^### (\d+)\. (.*)$", answer or "", re.M)}


def _nested_depths(monkeypatch):
    """Spy (the real method still runs): the frame-stack depth at which each output was recorded."""
    depths: list = []
    real = WorkflowExecutionContext.append_turn_output

    def watch(self, output):
        depths.append((output.command_name, len(self._frame_stack())))
        return real(self, output)

    monkeypatch.setattr(WorkflowExecutionContext, "append_turn_output", watch)
    return depths


def _unreviewed(gated, monkeypatch):
    """The gate off (these turns are about execution); an attended turn then."""
    monkeypatch.setattr(gated.ctx, "_subtask_gate", lambda subtasks: None)


# ----------------------------------------------------------------------
# The field
# ----------------------------------------------------------------------


def test_a_hand_built_output_has_no_part_and_an_old_record_still_validates():
    built = fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(response="x"), command_name="a")
    assert built.sub_task is None and built.model_dump(mode="json")["sub_task"] is None
    legacy = fastworkflow.CommandOutput.model_validate(
        {"command_response": {"response": "x", "success": True}, "command_name": "find_book"})
    assert legacy.sub_task is None


# ----------------------------------------------------------------------
# A split turn: every output carries its part, and the parts agree with the answer's headings
# ----------------------------------------------------------------------


def test_a_three_part_turn_numbers_every_output_by_its_part_and_the_headings_agree(gated, monkeypatch):
    _split_into(monkeypatch, THREE)
    depths = _nested_depths(monkeypatch)
    gated.script.steps = [HOURS, *_finish(), COUNT, *_finish(), HOURS, *_finish()]
    out = gated.ctx.process_turn("do all three", unattended=True)
    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []
    assert _by_part(out) == [("library_hours", 1), ("Library/find_book", 2), ("Library/count_books", 2),
                             ("library_hours", 3)]
    # find_book ran inside count_books' nested frame: it carries the part that called the planning command
    assert dict(depths)["Library/find_book"] > 1 and dict(depths)["Library/count_books"] == 1
    # both ways: each heading has outputs, each numbered output has a heading
    headings = _headings(out.answer)
    assert set(headings) == {1, 2, 3}
    assert {o.sub_task for o in _ran(out)} == set(headings)
    # execution is sequential: the parts come in order
    numbers = [o.sub_task for o in _ran(out)]
    assert numbers == sorted(numbers)


def test_a_part_with_a_failing_command_keeps_its_number_and_the_others_are_unaffected(gated, monkeypatch):
    _split_into(monkeypatch, THREE)
    gated.script.steps = [HOURS, *_finish(), COUNT_NO_QUERY, *_finish(), HOURS, *_finish()]
    out = gated.ctx.process_turn("do all three", unattended=True)
    ran = _ran(out)
    failed = [o for o in ran if not o.success]
    assert failed and {o.sub_task for o in failed} == {2}                # the part's own outputs, failure included
    assert [o.sub_task for o in ran] == [1, 2, 2, 3] and set(_headings(out.answer)) == {1, 2, 3}


def test_the_numbering_is_per_turn_a_later_single_command_turn_has_none(gated, monkeypatch):
    _split_into(monkeypatch, THREE)
    gated.script.steps = [HOURS, *_finish(), HOURS, *_finish(), HOURS, *_finish()]
    first = gated.ctx.process_turn("do all three", unattended=True)
    assert {o.sub_task for o in _ran(first)} == {1, 2, 3}
    _split_into(monkeypatch, [CATS])
    gated.script.steps = [HOURS, *_finish()]
    second = gated.ctx.process_turn("one more", unattended=True)
    assert _by_part(second) == [("library_hours", None)]


def test_a_value_already_set_is_kept(gated, monkeypatch):
    ctx = gated.ctx
    turn = ctx._turn_frame()
    turn.subtasks, turn.origins, turn.subtask_index = ["a", "b"], [0, 1], 1
    decided = fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(response="x"),
                                         command_name="find_book", sub_task=7)
    fresh = fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(response="x"),
                                       command_name="find_book")
    ctx.append_turn_output(decided)
    ctx.append_turn_output(fresh)
    assert (decided.sub_task, fresh.sub_task) == (7, 2)


# ----------------------------------------------------------------------
# Turns that are not split, and outputs that belong to no part
# ----------------------------------------------------------------------


def test_a_one_sub_task_turn_has_no_part(gated, monkeypatch):
    _split_into(monkeypatch, [CATS])
    gated.script.steps = [HOURS, *_finish()]
    out = gated.ctx.process_turn(CATS, unattended=True)
    assert _by_part(out) == [("library_hours", None)]
    assert _headings(out.answer) == {}


def test_a_slash_turn_has_no_part(gated, monkeypatch):
    out = gated.ctx.process_turn("/library_hours")
    assert _ran(out) and all(o.sub_task is None for o in out.command_outputs)


def test_the_review_commands_have_no_part_and_the_reply_turn_numbers_the_parts(gated, monkeypatch):
    queries: list = []
    _planner_stub(monkeypatch, queries)
    _split_into(monkeypatch, [FIND, JOKE])
    _remote_decides(monkeypatch)
    opened = gated.ctx.process_turn(f"{FIND} and {JOKE}")
    assert opened.status is TurnStatus.AWAITING_USER
    _open_review(gated)
    gated.ctx.process_turn("change 1: means=plain:Library/find_member")
    gated.ctx.process_turn("change 1: inputs=query=Ann Lee")
    gated.script.steps = [("execute_workflow_query", {"command": FIND_TEXT}), *_finish(), HOURS, *_finish()]
    out = gated.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []
    by_name = dict(_by_part(out))
    assert by_name["Library/find_member"] == 1 and by_name["library_hours"] == 2
    framework = [o for o in out.command_outputs if o.framework]
    assert framework and all(o.sub_task is None for o in framework)
    headings = _headings(out.answer)
    assert set(headings) == {1, 2} and {o.sub_task for o in _ran(out) if not o.framework} == {1, 2}


def test_dropping_an_item_in_the_review_renumbers_the_parts_as_the_answer_does(gated, monkeypatch):
    queries: list = []
    _planner_stub(monkeypatch, queries)
    _split_into(monkeypatch, [FIND, JOKE, DOGS])
    _remote_decides(monkeypatch)
    gated.ctx.process_turn(f"{FIND}, {JOKE} and {DOGS}")
    _open_review(gated)
    gated.ctx.process_turn("drop 2")
    gated.ctx.process_turn("change 1: means=plain:Library/find_member")
    gated.ctx.process_turn("change 1: inputs=query=Ann Lee")
    gated.script.steps = [("execute_workflow_query", {"command": FIND_TEXT}), *_finish(), HOURS, *_finish()]
    out = gated.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []
    headings = _headings(out.answer)
    assert list(headings) == [1, 2] and "Raj" in headings[2]       # the dropped joke is gone; the dogs part is now 2
    by_name = dict(_by_part(out))
    assert by_name["Library/find_member"] == 1 and by_name["library_hours"] == 2


def test_a_review_saved_and_restored_before_the_accept_numbers_the_parts(annotated_library, monkeypatch):
    queries: list = []
    _planner_stub(monkeypatch, queries)
    _split_into(monkeypatch, [EMMA, JOKE])
    _remote_decides(monkeypatch)
    first = _scripted_context(annotated_library, monkeypatch)
    first.ctx.process_turn(f"{EMMA} and {JOKE}")
    first.ctx.process_turn("change 2: inputs=Ann Lee")
    state = json.loads(json.dumps(first.ctx.serialize_state(channel_id="sub-task-restore")))

    second = _scripted_context(annotated_library, monkeypatch)
    second.ctx.apply_serialized_state(state)
    second.script.steps = [("finish", {})] * 2 + [HOURS, *_finish()]    # the review frame, the first goal, the second
    out = second.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED and second.script.steps == []
    by_name = dict(_by_part(out))
    assert by_name["library_hours"] == 2 and by_name["review_subtasks"] is None
    assert all(o.sub_task is None for o in out.command_outputs if o.framework)
    assert set(_headings(out.answer)) == {1, 2}


def test_a_turn_resumed_after_an_ask_user_inside_a_part_keeps_numbering_the_parts(gated, monkeypatch):
    _unreviewed(gated, monkeypatch)
    _split_into(monkeypatch, THREE)
    gated.script.steps = [HOURS, *_finish(), ASK]
    first = gated.ctx.process_turn("do all three")
    assert first.status is TurnStatus.AWAITING_USER
    state = json.loads(json.dumps(gated.ctx.serialize_state(channel_id="sub-task-ask")))
    assert state["subtask_index"] == 1

    gated.script.steps = [HOURS, *_finish(), HOURS, *_finish()]
    second = gated.ctx.process_turn("the blue one")
    assert second.status is TurnStatus.COMPLETED and gated.script.steps == []
    by_name = _by_part(second)
    assert by_name == [("library_hours", 1), ("library_hours", 2), ("library_hours", 3)]
    assert [(o.command_name, o.sub_task) for o in first.command_outputs] == [("library_hours", 1), ("ask_user", 2)]
    asked = [o for o in second.command_outputs if o.is_ask_user]
    assert asked and all(o.sub_task == 2 for o in asked)
    assert set(_headings(second.answer)) == {1, 2, 3}


# ----------------------------------------------------------------------
# Concurrency: attribution reads the turn's execution state, which the pooled draft calls never touch
# ----------------------------------------------------------------------


def test_the_pooled_model_calls_of_the_draft_phase_cannot_leak_a_part(gated, monkeypatch):
    """Before any sub-task runs the turn has no origins, so an output recorded from any worker thread has no part; the
    pool's workers (FW_SUBTASK_WORKERS=4) only read and record, they set nothing."""
    monkeypatch.setenv("FW_SUBTASK_WORKERS", "4")
    assert subtask_pool.is_parallel(4)
    seen: list = []
    gate = threading.Barrier(4, timeout=30)

    def worker(i):
        def call():
            gate.wait()                                  # all four are in flight together
            output = fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(response=str(i)), command_name=f"draft_{i}")
            gated.ctx.append_turn_output(output)
            seen.append(output.sub_task)
            return i
        return call

    assert subtask_pool.map_in_order([worker(i) for i in range(4)]) == [0, 1, 2, 3]
    assert seen == [None] * 4
