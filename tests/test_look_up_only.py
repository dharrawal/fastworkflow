"""A part the user asked, in the turn's review, to only look up never runs a command that writes (bd fix-ftdl.4.3).

"look up step N" re-picks that part's command from the commands that do not write. Whatever it ends on (a read-only
command, an ad hoc part, or the same command when no re-draft runs), the part carries a read-only mark on its plan item,
and the one dispatch of every command refuses a command that ``command_writes.write_commands`` lists while that part
runs: the agent's own call, a step code runs, and the steps of a planning command it calls.

Real library copy with its trained heads (unannotated: no approval gate in the way); the agent is the real agent with a
scripted model (the approval tests' ``todo`` fixture); the review reads replies with the head's recorded answers. The
planner's split is the one seam faked, as the other gate tests fake it. No test calls a model.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from fastworkflow.review import review_proposals
from fastworkflow.review.review_engine import CONFIRMED, REVISED, proposal_of
from fastworkflow.review.review_proposals import Means, SubTask, SubTaskProposal
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent with a scripted model, and the guards
    _COPY_IGNORE, LIBRARY, _finish, no_live_model, todo)
from tests.test_command_writes import writes_env  # noqa: F401 - the library's planning commands, real dispatch
from tests.test_round1_gate import _open_review, _split_into
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow", "recorded_review_router", "no_planner_key")

ADD_NOTE = "Record/add_note"
NOTE = "add_note text='torn cover'"           # an exact command name: the part is bound to add_note, which writes
SHOW = "show_book"                            # an exact command name that only reads


@pytest.fixture
def annotated_library(tmp_path) -> Path:
    """A plain copy of the library (overrides the approval tests' annotated one): add_note asks no approval here."""
    folder = tmp_path / "library_plain"
    shutil.copytree(LIBRARY, folder, ignore=_COPY_IGNORE, symlinks=True)
    os.symlink(LIBRARY / "___command_info" / "versions", folder / "___command_info" / "versions")
    return folder


@pytest.fixture(autouse=True)
def no_model_call_for_parameters(monkeypatch):
    """Overrides the approval tests' guard: the review's own reply command (look_up) takes its step number from the
    parameter extractor, scripted here (the model boundary); the agent's calls bind their values in code."""
    import dspy
    from dspy.utils import DummyLM

    param_lm = DummyLM([{"reasoning": "n", "item": "1"}] * 12, adapter=dspy.JSONAdapter())

    def get_lm(role, *a, **k):
        if role == "LLM_PARAM_EXTRACTION":
            return param_lm
        raise AssertionError(f"unscripted model role {role!r}")

    monkeypatch.setattr("fastworkflow.utils.dspy_utils.get_lm", get_lm)


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    """The draft's argument extraction is a model call: off, as in the other gate tests."""
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


@pytest.fixture
def gated(todo):
    """The review gate restored, on book b1 (add_note is a command of every record)."""
    vars(todo.ctx).pop("_subtask_gate", None)
    todo.wf.record = todo.chores
    return todo


def _refused(out, name: str = "add_note"):
    return [o for o in out.command_outputs
            if (o.command_name or "").split("/")[-1] == name and not o.command_response.success]


def _review_then_look_up(gated, monkeypatch, parts):
    _split_into(monkeypatch, parts)
    out = gated.ctx.process_turn(" and ".join(parts))
    assert out.status is TurnStatus.AWAITING_USER
    record = _open_review(gated)
    assert proposal_of(record).items[0].means.name == ADD_NOTE
    out = gated.ctx.process_turn("look up step 1")
    assert out.status is TurnStatus.AWAITING_USER and record.state == REVISED
    return record


def test_the_agent_of_a_look_up_part_cannot_run_a_write(gated, monkeypatch):
    record = _review_then_look_up(gated, monkeypatch, [NOTE, SHOW])
    item = proposal_of(record).items[0]
    assert item.read_only is True
    saved = proposal_of(record).model_dump(mode="json")    # saved with the turn's plan, and read back
    assert saved["items"][0]["read_only"] is True and "read_only" not in saved["items"][1]
    assert SubTaskProposal.model_validate(saved).items[0].read_only is True
    assert proposal_of(record).items[1].read_only is False
    gated.script.steps = [("execute_workflow_query", {"command": "add_note <text>torn cover</text>"}), *_finish(),
                          *_finish()]
    out = gated.ctx.process_turn("yes")
    assert record.state == CONFIRMED
    assert out.status is TurnStatus.COMPLETED
    assert gated.chores.notes == gated.base               # nothing was written
    [refused] = _refused(out)
    text = refused.command_response.response
    assert "look" in text and "add_note" in text
    assert refused.command_response.artifacts.get("write_refused") is True


def test_a_look_up_part_bound_to_a_writing_planning_command_runs_none_of_its_writes(writes_env, monkeypatch):
    # With drafting off "look up" re-picks nothing, so a part can stay bound to note_member, whose steps reach add_note.
    # The plan is the one the review accepted (test above): its one item, marked read-only. Code runs note_member's steps
    # (find_member, open_member, add_note) through the executor's own dispatch, in the planning command's frame; the
    # write is refused there, so the member's record is unchanged.
    item = SubTask(goal="note on Raj Patel's record that he paid his fine",
                   means=Means(kind="planning", name="Library/note_member"),
                   inputs=["member=Raj Patel", "text=he paid his fine"], read_only=True)

    def accepted(self, subtasks):
        self._turn_subtask_plan = SubTaskProposal(items=[item])
        self._turn_subtask_review = "reviewed"
        return None

    monkeypatch.setattr(type(writes_env.ctx), "_subtask_gate", accepted)
    library = writes_env.wf.root
    before = list(library.members["m2"].notes)
    writes_env.model.finish()
    writes_env.model.finish()

    result = writes_env.ctx.process_turn(item.goal)

    assert result.status == TurnStatus.COMPLETED
    assert library.members["m2"].notes == before                # nothing was written
    refused = [o for o in writes_env.ctx._turn_outputs if (o.command_name or "").endswith("add_note")]
    assert refused and all(not o.success for o in refused)
    assert all("looked up" in o.command_response.response for o in refused)
    # open_member ran: the steps before the write are the part's, and they read
    opened = [o for o in writes_env.ctx._turn_outputs if (o.command_name or "").endswith("open_member")]
    assert opened and opened[0].success


def test_the_agent_of_a_look_up_part_cannot_call_a_planning_command_that_writes(gated, monkeypatch):
    # A planning command whose steps reach a write is itself a write: refused at its own dispatch, so none of its steps
    # (a nested frame) starts.
    record = _review_then_look_up(gated, monkeypatch, [NOTE, SHOW])
    assert proposal_of(record).items[0].read_only is True
    gated.script.steps = [
        ("execute_workflow_query", {"command": "note_member <member>Raj Patel</member> <text>paid</text>"}),
        *_finish(), *_finish()]
    out = gated.ctx.process_turn("yes")
    assert out.status is TurnStatus.COMPLETED
    assert gated.library.members["m2"].notes == ["prefers email reminders"]
    [refused] = _refused(out, "note_member")
    assert "looked up" in refused.command_response.response


def test_a_part_the_user_did_not_ask_to_look_up_still_writes(gated, monkeypatch):
    # The mark is on the part the reply named only: an accepted write part with no look-up runs its write.
    _split_into(monkeypatch, [NOTE, SHOW])
    gated.ctx.process_turn(f"{NOTE} and {SHOW}")
    record = _open_review(gated)
    assert not any(item.read_only for item in proposal_of(record).items)
    gated.script.steps = [("execute_workflow_query", {"command": "add_note <text>torn cover</text>"}), *_finish(),
                          *_finish()]
    out = gated.ctx.process_turn("yes")
    assert out.status is TurnStatus.COMPLETED
    assert gated.chores.notes == [*gated.base, "torn cover"]


def test_a_look_up_part_that_becomes_ad_hoc_stays_read_only(gated, monkeypatch):
    # The finding (e-writes, item 6): with drafting on, "look up" re-picks the part's command from the commands that
    # do not write; when none fits, the part becomes ad hoc and the agent runs it with every command listed.
    _split_into(monkeypatch, [NOTE, SHOW])
    gated.ctx.process_turn(f"{NOTE} and {SHOW}")
    record = _open_review(gated)
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "on")
    gated.ctx.process_turn("look up step 1")
    item = proposal_of(record).items[0]
    assert item.means.kind == "from_scratch" and item.read_only is True
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    gated.script.steps = [("execute_workflow_query", {"command": "add_note <text>torn cover</text>"}), *_finish(),
                          *_finish()]
    out = gated.ctx.process_turn("yes")
    assert out.status is TurnStatus.COMPLETED
    assert gated.chores.notes == gated.base
    [refused] = _refused(out)
    assert "looked up" in refused.command_response.response


def test_with_no_agent_model_a_look_up_part_ends_as_not_run_naming_the_write(no_agent_model, gated, monkeypatch):
    # No agent model: code runs the part's command as its one step; the refusal ends the part as not run, its reason
    # carrying the refusal, and the other part runs.
    from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT, WorkflowExecutionContext

    def never(self):
        raise AssertionError("the agent LM was built in no-agent mode")

    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context", never)
    _split_into(monkeypatch, [NOTE, SHOW])
    gated.ctx.process_turn(f"{NOTE} and {SHOW}")
    gated.ctx.process_turn("torn cover")            # with no agent model the part's missing text is asked first
    record = _open_review(gated)
    gated.ctx.process_turn("look up step 1")
    assert proposal_of(record).items[0].read_only is True
    out = gated.ctx.process_turn("yes")
    assert out.status is TurnStatus.COMPLETED
    assert gated.chores.notes == gated.base
    [entry] = [entry for o in out.command_outputs
               for entry in (o.command_response.artifacts.get(NOT_RUN_ARTIFACT) or [])]
    assert entry["command"] == ADD_NOTE and entry["sub_task"] == 1
    assert "you asked for this part only to be looked up" in entry["reason"]
    assert [o.command_name for o in out.command_outputs if o.command_response.success
            and o.command_name == "Book/show_book"], "the other part ran"


# ----------------------------------------------------------------------
# What the review says about a look-up-only part (one definition: SubTaskProposal.changes_data / badge_of)
# ----------------------------------------------------------------------

NOTE_TWO = "add_note text='second note'"      # another part that writes


def _review_payload(out):
    return out.command_outputs[-1].command_response.artifacts["review"]


@pytest.mark.parametrize("drafting", ["off", "on"])
def test_after_look_up_the_part_is_shown_look_up_only_and_the_turn_needs_no_explicit_yes(gated, monkeypatch, drafting):
    from fastworkflow.review.review_engine import CHANGES_DATA_SENTENCE

    _split_into(monkeypatch, [NOTE, SHOW])
    gated.ctx.process_turn(f"{NOTE} and {SHOW}")
    record = _open_review(gated)
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", drafting)
    out = gated.ctx.process_turn("look up step 1")
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    assert proposal_of(record).items[0].read_only is True
    lines = out.answer.splitlines()
    assert lines[1].endswith(" - " + review_proposals.LOOK_UP_ONLY_BADGE), lines
    assert "changes data" not in out.answer                # the turn is the read-only turn's question
    assert CHANGES_DATA_SENTENCE not in out.answer
    payload = _review_payload(out)
    assert payload["changes_data"] == [False, False]
    assert payload["badges"] == [review_proposals.LOOK_UP_ONLY_BADGE, ""]
    # an accept that is not a plain yes accepts a turn that changes no data
    gated.script.steps = [*_finish(), *_finish()]
    out = gated.ctx.process_turn("lgtm")
    assert record.state == CONFIRMED and out.status is TurnStatus.COMPLETED
    assert gated.chores.notes == gated.base


def test_another_part_that_writes_keeps_the_explicit_yes(gated, monkeypatch):
    from fastworkflow.review.review_engine import CHANGES_DATA_SENTENCE

    _split_into(monkeypatch, [NOTE, NOTE_TWO])
    gated.ctx.process_turn(f"{NOTE} and {NOTE_TWO}")
    record = _open_review(gated)
    out = gated.ctx.process_turn("look up step 1")
    lines = out.answer.splitlines()
    assert lines[1].endswith(" - " + review_proposals.LOOK_UP_ONLY_BADGE)
    assert lines[2].endswith(" - " + review_proposals.CHANGES_DATA_BADGE)
    assert CHANGES_DATA_SENTENCE in out.answer
    assert _review_payload(out)["changes_data"] == [False, True]
    gated.ctx.process_turn("lgtm")
    assert record.state not in (CONFIRMED,)              # asked again: only an explicit yes runs a write


def _subtask(goal, read_only=False):
    return SubTask(goal=goal, means=Means(kind="plain", name=ADD_NOTE), read_only=read_only)


def test_a_merge_with_a_read_only_item_is_read_only_whichever_comes_first():
    from fastworkflow.review.review_engine import Merge, apply_edits

    for marks in ((False, True), (True, False)):
        proposal = SubTaskProposal(items=[_subtask("a", marks[0]), _subtask("b", marks[1]), _subtask("c")])
        edited = apply_edits(proposal, [Merge(indexes=[1, 2])])
        assert [item.read_only for item in edited.items] == [True, False]


def test_a_split_of_a_read_only_item_gives_the_mark_to_every_part():
    from fastworkflow.review.review_engine import Split, apply_edits

    proposal = SubTaskProposal(items=[_subtask("a", True), _subtask("b")])
    edited = apply_edits(proposal, [Split(index=1, parts=[{"goal": "a1"}, {"goal": "a2"}])])
    assert [item.read_only for item in edited.items] == [True, True, False]
