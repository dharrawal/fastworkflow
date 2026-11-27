"""Where the plan path and the "/" path say the same thing (bd fix-o1lg alignment): a refusal's own message, the
reason a free-text value was not accepted, and the words of a reply that the which-command question reads.

Real ``WorkflowExecutionContext`` on the library example (the no-agent fixture of ``tests/test_parameter_loop.py``),
the extractor and the matcher scripted as there."""
from __future__ import annotations

import pytest

from fastworkflow.parameters.signatures import InputForParamExtraction
from tests.conftest import script_matcher
from tests.test_parameter_loop import HOST, no_agent, _offline  # noqa: F401
from tests.test_round1_gate import gated  # noqa: F401
from tests.test_approval_binding import annotated_library, grant_root, todo, no_live_model, no_model_call_for_parameters  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

OPEN_BOOK = "Library/open_book"


def test_a_refused_command_keeps_its_own_message_in_the_not_run_entry(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(book_uid="b99"))
    script_matcher(monkeypatch, lambda t: {OPEN_BOOK: 0.95, "none": 0.0})
    no_agent.parts = ["open the book b99"]
    out = no_agent.ctx.process_turn("open the book b99", approval_surface=HOST)
    artifacts = out.command_outputs[-1].command_response.artifacts
    entry = artifacts["fw_not_run"][0]
    assert "no book has the uid b99" in entry["reason"] and "no book has the uid b99" in out.answer
    assert entry["command"] == OPEN_BOOK and entry["kind"] == "not_run_here"


# ---- the reason a free-text value was not accepted ---------------------------------------------------------------------

BOOK_BY_TITLE = "Library/book_by_title"


@pytest.mark.parametrize("value, why", [("it", "is a pronoun"), ("book", "is a kind word of this workflow")])
def test_the_plan_question_says_why_a_free_text_value_was_not_taken(no_agent, monkeypatch, value, why):
    """The `/` question words the reason ("title 'it' is a pronoun, not a value"); the plan path's question says the same
    for the same value, through the same ``Gap`` record and the one question builder."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(title=value))
    script_matcher(monkeypatch, lambda t: {BOOK_BY_TITLE: 0.95, "none": 0.0})
    text = "who wrote it and how many copies" if value == "it" else "who wrote the book"
    no_agent.parts = [text]
    out = no_agent.ctx.process_turn(text, approval_surface=HOST)
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") == "planning_parameter", out.answer
    assert artifacts["parameter_request"]["missing"] == ["title"]          # the data is unchanged: still open, still missing
    assert f"title {value!r} {why}" in out.answer, out.answer
    assert "Which title should I use?" in out.answer


def test_a_missing_value_with_nothing_refused_has_no_reason_text(no_agent, monkeypatch):
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct())
    script_matcher(monkeypatch, lambda t: {BOOK_BY_TITLE: 0.95, "none": 0.0})
    no_agent.parts = ["who wrote the thing"]
    out = no_agent.ctx.process_turn("who wrote the thing", approval_surface=HOST)
    assert "your request does not say which title to use. Which title" in out.answer, out.answer


# ---- the approval re-ask on "/" ----------------------------------------------------------------------------------------

def test_a_reask_of_a_slash_approval_carries_the_shape_of_the_first_ask(todo, annotated_library, grant_root):

    ctx = todo.ctx
    first = ctx.process_turn("/add_note text=Ann")
    asked = first.command_outputs[-1].command_response.artifacts
    assert asked["question_kind"] == "approval" and asked["approval"]["command"] and asked["awaiting_user"] is True

    again = ctx.process_turn("maybe later")
    kept = again.command_outputs[-1].command_response.artifacts
    assert kept["question_kind"] == "approval" and kept["approval"] == asked["approval"]
    assert kept["awaiting_user"] is True and "Reply yes to approve" in again.answer


# ---- a refusal that names no field ---------------------------------------------------------------------------------------

NOTE_MEMBER = "Library/note_member"
NOTE_THAT_NAMES_THE_MEMBER = "note on Zed Quill: Zed Quill called"
VALIDATOR_WORDS = "holds the member's name"


def test_a_refusal_that_names_no_field_is_not_asked_on_the_plan_path_and_ends_with_the_validators_message(no_agent, monkeypatch):
    """The documented rule (a refusal that names no input is not asked): the `/` path ends the command with the
    validator's message; the plan path ends the part as not run carrying the same message, with no question."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(member="Zed Quill", text="Zed Quill called"))
    script_matcher(monkeypatch, lambda t: {NOTE_MEMBER: 0.9, "none": 0.0})
    no_agent.parts = [NOTE_THAT_NAMES_THE_MEMBER]
    out = no_agent.ctx.process_turn(NOTE_THAT_NAMES_THE_MEMBER, approval_surface=HOST)
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") is None and "parameter_request" not in artifacts, out.answer
    assert no_agent.ctx._waiting is None
    entry = artifacts["fw_not_run"][0]
    assert entry["command"] == NOTE_MEMBER and VALIDATOR_WORDS in entry["reason"], entry
    assert VALIDATOR_WORDS in out.answer and "does not give" not in out.answer


# ---- the record an Input's validators read ---------------------------------------------------------------------------

def _records_the_validators_read(monkeypatch, command: str) -> list[str]:
    """The class of ``workflow.record`` each time the one validation driver runs *command*'s validators (the driver is
    called through, unchanged)."""
    from fastworkflow.parameters import validate as one_driver

    seen: list[str] = []
    real = one_driver.validate

    def validate(input_model, values, *, command_text, workflow, command_name):
        if command_name == command and workflow is not None:      # the load's own check has no workflow
            seen.append(type(workflow.record).__name__)
        return real(input_model, values, command_text=command_text, workflow=workflow, command_name=command_name)

    monkeypatch.setattr(one_driver, "validate", validate)
    return seen


def test_a_part_of_an_enclosing_command_is_validated_on_the_enclosing_record_on_the_plan_path(no_agent, monkeypatch):
    """Standing on a Member, a part bound to a Library command has its arguments checked (the draft, the gate's values
    question and its reply, the run's check) with ``workflow.record`` the Library, the record the command will run on:
    the "/" path binds the owner's record before extraction, and the plan path now does too (B1's one climb)."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(member="Zed Quill"))
    script_matcher(monkeypatch, lambda t: {NOTE_MEMBER: 0.9, "none": 0.0})
    seen = _records_the_validators_read(monkeypatch, NOTE_MEMBER)
    no_agent.wf.record = no_agent.library.members["m1"]
    no_agent.parts = ["note on Zed Quill"]

    # The plan is pinned (``use_plan``) so that the local model, which stands on the Member, does not take the part.
    asked = no_agent.ctx.process_turn("note on Zed Quill", approval_surface=HOST, use_plan=NOTE_MEMBER)
    assert asked.command_outputs[-1].command_response.artifacts.get("question_kind") == "planning_parameter", asked.answer
    no_agent.ctx.process_turn("text: rehung the banner", approval_surface=HOST)

    assert seen and set(seen) == {"Library"}, seen
    assert no_agent.wf._current_command_context is no_agent.library.members["m1"]     # the position did not move


def test_the_same_part_sent_with_a_slash_is_validated_on_the_enclosing_record_too(no_agent, monkeypatch):
    """The "/" path, for comparison: the CME binds the owner's record before it extracts."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(member="Zed Quill", text="rehung the banner"))
    seen = _records_the_validators_read(monkeypatch, NOTE_MEMBER)
    no_agent.wf.record = no_agent.library.members["m1"]

    no_agent.ctx.process_turn('/note_member member="Zed Quill" text="rehung the banner"', approval_surface=HOST)

    assert seen and set(seen) == {"Library"}, seen


def test_a_review_edit_re_drafts_an_enclosing_command_on_the_enclosing_record(no_agent, monkeypatch):
    """The review engine's own re-draft after an edit (a step switched to another command, ``SubTaskProposal.revised``)
    checks the new command's arguments on the record the command will run on, as the gate's draft does: standing on a
    Member, switching a step to a Library command re-drafts it with ``workflow.record`` the Library."""
    from fastworkflow.review import review_engine

    from tests.test_parameter_loop import ABOUT_DUNE, ABOUT_EMMA, MEMBER, NOTE, _reply

    second = f"{ABOUT_DUNE}; note on {MEMBER}: {NOTE}"
    script_matcher(monkeypatch, lambda t: {"Library/book_by_title": 0.95, "none": 0.0})
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(member=MEMBER, text=NOTE)
                        if "member" in cls.model_fields else
                        cls.model_construct(title="Emma" if "Emma" in self.command else "Dune"))
    no_agent.parts = [ABOUT_EMMA, second]
    review = no_agent.ctx.process_turn(f"{ABOUT_EMMA} and {second}", approval_surface=HOST)
    record = no_agent.store.active()
    assert record is not None and record.state == review_engine.AWAITING, review.answer

    seen = _records_the_validators_read(monkeypatch, NOTE_MEMBER)
    no_agent.wf.record = no_agent.library.members["m1"]
    edited = _reply(no_agent, f"step 2: use {NOTE_MEMBER}")

    assert review_engine.proposal_of(record).items[1].means.name == NOTE_MEMBER, edited.answer
    assert seen and set(seen) == {"Library"}, seen
    assert no_agent.wf._current_command_context is no_agent.library.members["m1"]     # the position did not move
