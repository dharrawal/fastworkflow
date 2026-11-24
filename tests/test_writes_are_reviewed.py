"""A command that changes data runs only when the user named it, or after the user says yes to it (bd fix-ftdl.8.3,
fix-ftdl.8.4, fix-ftdl.8.5, fix-ftdl.8.8; the two marks of fix-qjne.24).

"Named": the user typed its exact name, picked it at the which-command question, or a client pinned or called it
directly. A close spelling of its name is the closest command, not a name: it is confirmed like a model's choice (bd
fix-ftdl.9). "Yes": the turn's review (the explicit-yes rule), the "Did you mean the command 'X'?" question on "/", or the
command's own approval question when the gate will ask it (so a command with both marks is asked once). A turn nobody
can answer has no review, as before.

Real library copy (``Record/add_note`` carries ``@writes``; its approval mark is set per test), its trained heads, the
real agent with a scripted model (the ``todo`` fixture of the approval tests), the planner's split scripted. No model is
called: a test that relies on the remote model NOT answering uses ``no_planner_key``.
"""
from __future__ import annotations

import os
import shutil

import pytest

from fastworkflow import approval_binding as ab
from fastworkflow import command_writes, review_engine, review_proposals
from fastworkflow._workflows.plan_metadata_extraction import resolution
from fastworkflow.approval_annotations import REQUIRED_ALWAYS, REQUIRED_UNTIL_DELEGATED
from fastworkflow.approval_binding import Origin, Outcome
from fastworkflow.review_engine import AWAITING, proposal_of
from fastworkflow.turn import TurnStatus
from fastworkflow.utils.signatures import InputForParamExtraction
from tests.test_approval_binding import (  # noqa: F401 - the real agent on a library copy, and the guards
    _COPY_IGNORE, LIBRARY, _annotate, grant_root, no_live_model, no_model_call_for_parameters, todo)
from tests.test_parameter_loop import _offline, no_agent  # noqa: F401
from tests.test_round1_gate import JOKE, _marker, _split_into, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow", "no_planner_key")

WRITE = "Record/add_note"
NOTE = "the binding is loose"
SURE_WRITE = f"jot down that {NOTE}"          # the head: add_note, confident, in a book's context
UNSURE_WRITE = "note it"                       # the head: add_note or show_notes ("ad_note" is a close spelling)
CLOSE_NAME = "add notte"                       # a close spelling of add_note
RUN_NOTE = [("execute_workflow_query", {"command": f"add_note text={NOTE}"}), ("finish", {})]


@pytest.fixture
def annotated_library(tmp_path, request) -> str:
    """The library copy, with ``add_note``'s approval mark (a ``UserApproval`` member name) given by indirect
    parametrization (none by default); ``<command file>:<member>`` marks that command instead."""
    folder = tmp_path / "library_writes"
    shutil.copytree(LIBRARY, folder, ignore=_COPY_IGNORE, symlinks=True)
    os.symlink(LIBRARY / "___command_info" / "versions", folder / "___command_info" / "versions")
    if (member := getattr(request, "param", None)) is not None:
        command_file, _, member = member.rpartition(":")
        _annotate(folder, command_file or "Record/add_note.py", member)
    return folder


@pytest.fixture
def book(gated):
    """The gated agent context on book b1, where ``add_note`` is reachable."""
    gated.wf.record = gated.library.books["b1"]
    gated.book = gated.library.books["b1"]
    gated.before = list(gated.book.notes)
    return gated


def _added(env) -> list:
    return env.book.notes[len(env.before):]


def _reviews(env):
    return [record for record in env.store.records if record.command == "review_subtasks"]


def _open_review(env):
    (record,) = [record for record in _reviews(env) if record.state == AWAITING]
    return record


def _approvals(env):
    return [record for record in env.store.records if record.command == WRITE]


# ======================================================================================================================
# The rule, unit level
# ======================================================================================================================

def test_a_named_write_is_a_fast_path_and_a_write_the_model_or_a_close_spelling_chose_is_not(book):
    named = resolution.part_intent(book.wf, "add_note")
    close = resolution.part_intent(book.wf, CLOSE_NAME)
    guessed = resolution.part_intent(book.wf, SURE_WRITE)
    assert (named.decided_by, close.decided_by, guessed.decided_by) == (
        resolution.BY_EXACT_NAME, resolution.BY_FUZZY_NAME, resolution.BY_CLASSIFIER)
    assert named.command == close.command == guessed.command == WRITE
    assert resolution.is_fast_path(named, book.wf)
    # bd fix-ftdl.9: a close spelling is not the user naming the command, so a write found that way is reviewed
    assert not resolution.is_fast_path(close, book.wf)
    assert not resolution.is_fast_path(guessed, book.wf)
    # a prefix of the name is not a close spelling of it at all: the name layers do not decide it
    for prefix in ("a", "ad", "add"):
        assert resolution.part_intent(book.wf, prefix).decided_by not in (
            resolution.BY_EXACT_NAME, resolution.BY_FUZZY_NAME), prefix
    # a command that changes nothing keeps the fast path when a close spelling of its name chose it
    shown = resolution.part_intent(book.wf, "show_notse")
    assert shown.decided_by == resolution.BY_FUZZY_NAME and resolution.is_fast_path(shown, book.wf)
    # a command that changes nothing keeps the fast path when the local model alone chose it
    book.wf.record = book.library
    find = resolution.part_intent(book.wf, "find member Ann Lee")
    assert find.decided_by == resolution.BY_CLASSIFIER and resolution.is_fast_path(find, book.wf)
    assert find.command not in command_writes.write_commands(book.wf.folderpath)


@pytest.mark.parametrize("annotation", [REQUIRED_ALWAYS, REQUIRED_UNTIL_DELEGATED])
@pytest.mark.parametrize("granted", [False, True])
@pytest.mark.parametrize("unattended", [False, True])
@pytest.mark.parametrize("origin", [Origin.USER, Origin.AGENT])
def test_would_ask_is_the_gates_own_decision(book, grant_root, monkeypatch, annotation, granted, unattended, origin):
    from fastworkflow import approval_grants, state_paths

    monkeypatch.setattr("fastworkflow.approval_binding.command_approval", lambda folder, name: annotation)
    store = ab.store_for(book.wf)
    store.unattended = unattended
    identity = approval_grants.identity_key(store.surface, store.channel)
    if granted:
        approval_grants.record(identity, state_paths.workflow_id(book.wf.folderpath), WRITE, REQUIRED_UNTIL_DELEGATED)
    expected = ab.decide(annotation, origin=origin, surface=store.surface, unattended=unattended,
                         awaiting_on_channel=False, auto_granted=granted).outcome is Outcome.ASK
    assert ab.would_ask(book.wf, WRITE, origin) is expected


def test_would_ask_is_false_for_a_command_with_no_approval_mark(book):
    assert ab.would_ask(book.wf, WRITE, Origin.AGENT) is False


# ======================================================================================================================
# The plan path (finding 1)
# ======================================================================================================================

def test_an_exactly_named_write_runs_with_no_review(book, monkeypatch):
    text = f"add_note text={NOTE}"
    _split_into(monkeypatch, [text])
    book.script.steps = list(RUN_NOTE)
    out = book.ctx.process_turn(text)
    assert out.status is TurnStatus.COMPLETED, out.answer
    assert _reviews(book) == [] and _added(book) == [NOTE]
    assert _marker(book) == "fast_path"


def test_a_write_found_by_a_close_spelling_is_reviewed_and_runs_on_a_yes(book, monkeypatch):
    # bd fix-ftdl.9: this ran with no review while a close spelling counted as naming the command
    _split_into(monkeypatch, [CLOSE_NAME])
    out = book.ctx.process_turn(CLOSE_NAME)
    assert out.status is TurnStatus.AWAITING_USER, out.answer
    assert [item.means.name for item in proposal_of(_open_review(book)).items] == [WRITE]
    assert _added(book) == []                                      # nothing ran before the yes
    book.script.steps = list(RUN_NOTE)
    ran = book.ctx.process_turn("yes")
    assert ran.status is TurnStatus.COMPLETED, ran.answer
    assert _added(book) == [NOTE] and _marker(book) == "reviewed"


def test_a_write_picked_at_the_which_command_question_runs_with_no_review(book, monkeypatch):
    _split_into(monkeypatch, [UNSURE_WRITE])
    asked = book.ctx.process_turn(UNSURE_WRITE)
    assert asked.status is TurnStatus.AWAITING_USER and "add_note" in asked.answer
    book.script.steps = list(RUN_NOTE)
    out = book.ctx.process_turn("add_note")
    assert out.status is TurnStatus.COMPLETED, out.answer
    assert _reviews(book) == [] and _added(book) == [NOTE]
    assert _marker(book) == "fast_path"


def test_a_write_the_local_model_alone_chose_is_reviewed_and_runs_on_a_yes(book, monkeypatch):
    _split_into(monkeypatch, [SURE_WRITE])
    out = book.ctx.process_turn(SURE_WRITE)
    assert out.status is TurnStatus.AWAITING_USER, out.answer
    record = _open_review(book)
    proposal = proposal_of(record)
    assert [item.means.name for item in proposal.items] == [WRITE]
    assert proposal.badges(book.wf.folderpath) == [review_proposals.CHANGES_DATA_BADGE]
    assert review_engine.CHANGES_DATA_SENTENCE in out.answer
    assert _added(book) == []                                      # nothing ran before the yes
    book.script.steps = list(RUN_NOTE)
    ran = book.ctx.process_turn("yes")
    assert ran.status is TurnStatus.COMPLETED, ran.answer
    assert _added(book) == [NOTE] and _marker(book) == "reviewed"


def test_an_unattended_write_the_local_model_chose_runs_as_drafted_unreviewed(book, monkeypatch):
    _split_into(monkeypatch, [SURE_WRITE])
    book.script.steps = list(RUN_NOTE)
    out = book.ctx.process_turn(SURE_WRITE, unattended=True)
    assert out.status is TurnStatus.COMPLETED, out.answer
    assert _reviews(book) == [] and _added(book) == [NOTE]
    assert _marker(book) == "not_reviewed"


def test_with_no_agent_model_a_guessed_write_asks_its_value_then_the_review_then_runs_on_a_yes(no_agent, monkeypatch):
    env = no_agent
    env.wf.record = env.library.books["b1"]
    env.book, env.before = env.library.books["b1"], list(env.library.books["b1"].notes)
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct())
    env.parts = [SURE_WRITE]
    asked = env.ctx.process_turn(SURE_WRITE)
    artifacts = asked.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") == "planning_parameter", asked.answer
    assert artifacts["parameter_request"]["command"] == WRITE
    reviewed = env.ctx.process_turn(f"text: {NOTE}")
    assert reviewed.status is TurnStatus.AWAITING_USER, reviewed.answer
    assert review_engine.CHANGES_DATA_SENTENCE in reviewed.answer and _added(env) == []
    ran = env.ctx.process_turn("yes")
    assert ran.status is TurnStatus.COMPLETED, ran.answer
    assert _added(env) == [NOTE]


# ---- both marks: the approval question is the yes, asked once -----------------------------------------------------------

@pytest.mark.parametrize("annotated_library", ["REQUIRED_ALWAYS", "REQUIRED_UNTIL_DELEGATED"], indirect=True)
def test_a_guessed_write_that_will_be_asked_for_approval_is_asked_once(book, monkeypatch):
    _split_into(monkeypatch, [SURE_WRITE])
    book.script.steps = [RUN_NOTE[0]]
    out = book.ctx.process_turn(SURE_WRITE)
    assert out.status is TurnStatus.AWAITING_USER, out.answer
    assert _reviews(book) == []                                    # no review: the approval asks
    (approval,) = _approvals(book)
    assert approval.state == AWAITING and _added(book) == []
    book.script.steps = [RUN_NOTE[1]]
    ran = book.ctx.process_turn("yes")
    assert ran.status is TurnStatus.COMPLETED, ran.answer
    assert _added(book) == [NOTE] and len(_approvals(book)) == 1
    assert _marker(book) == "fast_path"


@pytest.mark.parametrize("annotated_library", ["REQUIRED_UNTIL_DELEGATED"], indirect=True)
def test_a_standing_grant_is_not_consent_to_the_models_choice_so_the_review_asks(book, grant_root, monkeypatch):
    from fastworkflow import approval_grants, state_paths

    store = ab.store_for(book.wf)
    approval_grants.record(approval_grants.identity_key(store.surface, store.channel),
                           state_paths.workflow_id(book.wf.folderpath), WRITE, REQUIRED_UNTIL_DELEGATED)
    _split_into(monkeypatch, [SURE_WRITE])
    out = book.ctx.process_turn(SURE_WRITE)
    assert out.status is TurnStatus.AWAITING_USER, out.answer
    _open_review(book)
    book.script.steps = list(RUN_NOTE)
    ran = book.ctx.process_turn("yes")
    assert ran.status is TurnStatus.COMPLETED, ran.answer
    assert _added(book) == [NOTE] and _approvals(book) == []       # the grant ran it: no approval question


# ======================================================================================================================
# Ad hoc parts may change data (finding 2)
# ======================================================================================================================

def test_an_ad_hoc_part_may_change_data_and_needs_an_explicit_yes(gated, monkeypatch, recorded_review_router):
    monkeypatch.setenv("LLM_AGENT", "scripted/agent")
    _split_into(monkeypatch, [JOKE])
    asked = gated.ctx.process_turn(JOKE)                          # the head offers two commands; none of them fits
    assert asked.status is TurnStatus.AWAITING_USER and "Which one is meant?" in asked.answer, asked.answer
    out = gated.ctx.process_turn("none")
    assert out.status is TurnStatus.AWAITING_USER, out.answer
    record = _open_review(gated)
    proposal = proposal_of(record)
    assert proposal.items[0].means.kind == review_proposals.MEANS_FROM_SCRATCH
    assert proposal.badges(gated.wf.folderpath) == [review_proposals.MAY_CHANGE_DATA_BADGE]
    assert f"- {review_proposals.MAY_CHANGE_DATA_BADGE}" in out.answer
    assert review_engine.MAY_CHANGE_DATA_SENTENCE in out.answer
    again = gated.ctx.process_turn("just do it")
    assert again.status is TurnStatus.AWAITING_USER and record.state == AWAITING
    assert review_engine.MAY_CHANGE_DATA_SENTENCE in again.answer
    gated.script.steps = [("finish", {})]
    ran = gated.ctx.process_turn("yes")
    assert ran.status is TurnStatus.COMPLETED and _marker(gated) == "reviewed"


def test_an_ad_hoc_part_carries_no_badge_where_nothing_changes_data(gated, monkeypatch):
    monkeypatch.setenv("LLM_AGENT", "scripted/agent")
    monkeypatch.setattr("fastworkflow.command_writes.write_commands", lambda folder: frozenset())
    proposal = review_proposals.SubTaskProposal(items=[review_proposals.SubTask(goal=JOKE)])
    assert proposal.badges(gated.wf.folderpath) == [""] and not proposal.changes_data(gated.wf.folderpath)


def test_with_no_agent_model_an_ad_hoc_part_carries_no_badge(gated, no_agent_model):
    proposal = review_proposals.SubTaskProposal(items=[review_proposals.SubTask(goal=JOKE)])
    assert proposal.badges(gated.wf.folderpath) == [""]
    assert "not run here" in proposal.user_rendering(gated.wf.folderpath, no_agent=True)


@pytest.mark.parametrize("replies", [["none"], ["blorp", "blorp"]], ids=["none", "two_unreadable"])
def test_rejecting_the_candidates_of_a_one_part_turn_does_not_run_it_unreviewed(book, monkeypatch, replies):
    monkeypatch.setenv("LLM_AGENT", "scripted/agent")
    _split_into(monkeypatch, [UNSURE_WRITE])
    asked = book.ctx.process_turn(UNSURE_WRITE)
    assert asked.status is TurnStatus.AWAITING_USER
    for reply in replies:
        out = book.ctx.process_turn(reply)
    assert out.status is TurnStatus.AWAITING_USER, out.answer
    record = _open_review(book)
    assert proposal_of(record).badges(book.wf.folderpath) == [review_proposals.MAY_CHANGE_DATA_BADGE]
    assert _added(book) == []


# ======================================================================================================================
# The "/" path (finding 1)
# ======================================================================================================================

def _slash(env, text, monkeypatch, extracted=None):
    """One deterministic ("/") turn of *text*, the extractor returning *extracted*."""
    monkeypatch.setattr(InputForParamExtraction, "extract_parameters",
                        lambda self, cls, name, folder: cls.model_construct(**(extracted or {})))
    return env.ctx.process_turn(f"/{text}")


def test_on_the_slash_path_a_guessed_write_asks_did_you_mean_and_yes_runs_it(book, monkeypatch):
    asked = _slash(book, SURE_WRITE, monkeypatch, {"text": NOTE})
    assert "Did you mean the command 'add_note'?" in asked.answer, asked.answer
    assert _added(book) == []
    ran = _slash(book, "yes", monkeypatch, {"text": NOTE})
    assert _added(book) == [NOTE], ran.answer


def test_on_the_slash_path_cancel_at_did_you_mean_writes_nothing(book, monkeypatch):
    _slash(book, SURE_WRITE, monkeypatch, {"text": NOTE})
    _slash(book, "cancel", monkeypatch, {"text": NOTE})
    assert _added(book) == []


def test_on_the_slash_path_a_close_spelling_of_a_write_asks_did_you_mean_and_yes_runs_it(book, monkeypatch):
    asked = _slash(book, CLOSE_NAME, monkeypatch, {"text": NOTE})
    assert "Did you mean the command 'add_note'?" in asked.answer, asked.answer
    assert _added(book) == []
    ran = _slash(book, "yes", monkeypatch, {"text": NOTE})
    assert _added(book) == [NOTE], ran.answer


def test_on_the_slash_path_a_named_write_runs_at_once(book, monkeypatch):
    ran = _slash(book, f"add_note text={NOTE}", monkeypatch, {"text": NOTE})
    assert "Did you mean" not in ran.answer and _added(book) == [NOTE], ran.answer


@pytest.mark.parametrize("annotated_library", ["REQUIRED_ALWAYS"], indirect=True)
def test_on_the_slash_path_a_guessed_write_that_asks_for_approval_is_asked_once(book, monkeypatch):
    asked = _slash(book, SURE_WRITE, monkeypatch, {"text": NOTE})
    assert "Did you mean" not in asked.answer and "Reply yes to approve" in asked.answer, asked.answer
    assert _added(book) == []
    ran = book.ctx.process_turn("yes")
    assert _added(book) == [NOTE], ran.answer


# ======================================================================================================================
# A command with only the approval mark: a model's pick of it is confirmed like a write's (one rule, both paths)
# ======================================================================================================================

SHOW = "Record/show_notes"
SHOW_UNTIL_DELEGATED = "Record/show_notes.py:REQUIRED_UNTIL_DELEGATED"
SHOW_ALWAYS = "Record/show_notes.py:REQUIRED_ALWAYS"
CLOSE_SHOW = "show_notse"                      # a close spelling of show_notes


@pytest.mark.parametrize("annotated_library", [SHOW_UNTIL_DELEGATED], indirect=True)
def test_on_the_slash_path_an_approval_only_command_a_close_spelling_chose_asks_did_you_mean(book, monkeypatch):
    """The user's own "/" message runs a REQUIRED_UNTIL_DELEGATED command unasked (its approval question is not asked),
    so when a model or a close spelling chose it, nothing would ask the user: "Did you mean" asks, and a yes runs it."""
    assert SHOW not in command_writes.write_commands(book.wf.folderpath)
    book.book.notes.append(NOTE)
    asked = _slash(book, CLOSE_SHOW, monkeypatch)
    assert "Did you mean the command 'show_notes'?" in asked.answer, asked.answer
    assert NOTE not in asked.answer
    ran = _slash(book, "yes", monkeypatch)
    assert NOTE in ran.answer and "Did you mean" not in ran.answer, ran.answer
    assert [record for record in book.store.records if record.command == SHOW] == []   # no approval question


@pytest.mark.parametrize("annotated_library", [SHOW_UNTIL_DELEGATED], indirect=True)
def test_on_the_slash_path_a_named_approval_only_command_runs_at_once(book, monkeypatch):
    book.book.notes.append(NOTE)
    ran = _slash(book, "show_notes", monkeypatch)
    assert "Did you mean" not in ran.answer and NOTE in ran.answer, ran.answer


@pytest.mark.parametrize("annotated_library", [SHOW_ALWAYS], indirect=True)
def test_on_the_slash_path_a_guessed_approval_only_command_that_asks_for_approval_is_asked_once(book, monkeypatch):
    asked = _slash(book, CLOSE_SHOW, monkeypatch)
    assert "Did you mean" not in asked.answer and "Reply yes to approve" in asked.answer, asked.answer


@pytest.mark.parametrize("annotated_library", [SHOW_UNTIL_DELEGATED], indirect=True)
def test_on_the_plan_path_an_approval_only_command_is_reviewed_only_when_its_approval_will_not_ask(
        book, grant_root, monkeypatch):
    """The plan path runs the part as the agent's call: with no standing grant its approval question asks (that is its
    yes, so the part keeps the fast path); with a grant nothing would ask, so the model's pick is reviewed."""
    from fastworkflow import approval_grants, state_paths

    close = resolution.part_intent(book.wf, CLOSE_SHOW)
    assert (close.decided_by, close.command) == (resolution.BY_FUZZY_NAME, SHOW)
    assert resolution.is_fast_path(close, book.wf)
    store = ab.store_for(book.wf)
    approval_grants.record(approval_grants.identity_key(store.surface, store.channel),
                           state_paths.workflow_id(book.wf.folderpath), SHOW, REQUIRED_UNTIL_DELEGATED)
    assert not resolution.is_fast_path(close, book.wf)
    assert resolution.is_fast_path(resolution.part_intent(book.wf, "show_notes"), book.wf)   # named: no review


# ======================================================================================================================
# An approved write inside a phase is counted (finding 3)
# ======================================================================================================================

def test_an_approved_write_in_a_phase_counts_against_its_step(tmp_path, initialized_fastworkflow, monkeypatch):
    """note_member's step names add_note once. add_note also asks for approval: code runs the step, the user approves,
    it runs. The phase agent then calls add_note again: the step's count is spent, so it is refused, not asked again."""
    from tests.test_command_writes import NOTE_MEMBER, WRITES_FILES
    from tests.test_step_helpers import _make_library_runner_env, _real_planning_command

    text = (LIBRARY / "_commands" / "Record" / "add_note.py").read_text()
    marked = text.replace("class ResponseGenerator:\n", "@fastworkflow.user_approval_for_execution("
                          "fastworkflow.UserApproval.REQUIRED_ALWAYS)\nclass ResponseGenerator:\n", 1)
    env = _make_library_runner_env(tmp_path, initialized_fastworkflow, monkeypatch, extra_commands=WRITES_FILES,
                                   real_dispatch=True, command_texts={"_commands/Record/add_note.py": marked})
    monkeypatch.setattr(env.ctx, "_subtask_gate", lambda subtasks: None)
    try:
        env.planning_command = _real_planning_command(NOTE_MEMBER, "_commands/Library/note_member.py").command
        env.args = {"member": "Raj Patel", "text": "he paid his fine"}
        env.model.run('add_note text="he paid his fine again"')
        env.model.finish()
        env.model.finish()
        asked = env.ctx.process_turn("note on Raj Patel's record that he paid his fine")
        assert asked.status is TurnStatus.AWAITING_USER and "Reply yes to approve" in asked.answer, asked.answer
        result = env.ctx.process_turn("yes")
        assert result.status is TurnStatus.COMPLETED, result.answer
        assert env.wf.root.members["m2"].notes == ["prefers email reminders", "he paid his fine"]
        refused = [o for o in env.ctx._turn_outputs if (o.command_name or "").endswith("add_note")
                   and not o.command_response.success]
        assert refused and "already ran" in refused[-1].command_response.response
        store = ab.store_for(env.wf)
        assert len([record for record in store.records if record.command.endswith("add_note")]) == 1
    finally:
        env.ctx.pop_active_workflow()
