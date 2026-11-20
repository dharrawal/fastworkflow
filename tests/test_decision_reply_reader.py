"""The decision step's reply is read by the ONE reply reader (bd fix-fpyb.8, step 8 of ``docs/pending_state_design.md``).

A planning command paused at a decision step (no-agent mode: a conditional step is answered yes or no, a command step
missing one plain value takes that value) reads the reply with ``parameters.reply.read_reply`` and the decision's own
answer rule (``reply.read_decision``), so it understands what a value question understands: a request for a moment (the
question stays, nothing is counted), "you misunderstood", a different request, cancel and abort. It is a step of a
running plan, not a part to resolve: "you misunderstood" and a different request END the command at that step as a stop
that says why (the plan is not asked which command again, and the reply is never run as a new request).

The asking rule is unchanged: a reply that gives nothing usable is asked once more, then the command stops.

Real components on the library copies the phase tests use; the remote decision model's read is scripted at the
``dspy.Predict`` seam (``tests/test_one_reply_reader.py``). No live model calls.
"""
from __future__ import annotations

import pytest

from fastworkflow import no_agent_mode
from fastworkflow.parameters import read, reply as replies
from fastworkflow.turn import TurnStatus
from tests.test_no_agent_decisions import (  # noqa: F401 - fixtures and the phases the decision tests use
    Conditional, MissingValue, _pending, _stops)
from tests.test_no_agent_mode import _plan, _sent  # noqa: F401 - helpers (the autouse no-agent fixture comes with the import)
from tests.test_no_agent_mode import _no_agent  # noqa: F401 - autouse
from tests.test_one_reply_reader import _Remote
from tests.test_step_helpers import _build_planning_command, library_env  # noqa: F401 - fixtures
from tests.test_turn_result_capture import initialized_fastworkflow, no_review_gate  # noqa: F401 - shared fixtures


def _asked(env, monkeypatch, phase):
    env.planning_command = _build_planning_command(phase)
    _plan(monkeypatch, env)
    first = env.ctx.process_turn("look up the book Dune")
    assert first.status is TurnStatus.AWAITING_USER
    return first


def _decision(env) -> dict:
    return env.ctx._waiting.payload["decision"]


# ======================================================================================================================
# The reader: the decision's own answer rule is plugged into the one reader
# ======================================================================================================================

CONDITIONAL = {"kind": no_agent_mode.STEP_CONDITIONAL, "condition": "c", "inner": "i", "question": "Does it hold?"}
MISSING = {"kind": no_agent_mode.STEP_MISSING_VALUE, "field": "text", "type": "str", "question": "Which text?"}


def _said(word: str, calls: list):
    def remote(question, reply):
        calls.append((question, reply))
        return word
    return remote


@pytest.mark.parametrize("reply, answer", [("yes", "yes"), ("Y", "yes"), ("no", "no"), ("false.", "no")])
def test_a_conditional_steps_yes_or_no_is_its_answer_and_the_remote_model_is_not_asked(reply, answer):
    calls: list = []
    reading = replies.read_decision(CONDITIONAL, reply, phrases=replies.phrase_of, remote=_said("wait", calls))
    assert (reading.kind, reading.values) == (replies.VALUES, {"answer": answer}) and calls == []


def test_the_own_answer_rule_is_read_before_the_fixed_phrases_and_the_remote_read():
    calls: list = []
    reading = replies.read_reply("maybe", [], [], answers=lambda text: replies.Reading(replies.VALUES, {"x": "1"}),
                                 phrases=lambda text: "abort", remote=_said("wait", calls))
    assert reading.kind == replies.VALUES and calls == []


@pytest.mark.parametrize("reply", ["cancel", "abort", "stop", "never mind"])
def test_cancel_and_its_synonyms_decline_a_decision(reply):
    for decision in (CONDITIONAL, MISSING):
        assert replies.read_decision(decision, reply, remote=_said("answer", [])).kind == replies.DECLINE


def test_a_missing_value_decision_reads_the_whole_reply_or_a_named_line_as_the_value():
    assert replies.read_decision(MISSING, "the cover is torn", remote=None).values == {"text": "the cover is torn"}
    assert replies.read_decision(MISSING, "text: torn", remote=None).values == {"text": "torn"}


@pytest.mark.parametrize("word, kind", [("wait", replies.WAIT), ("misunderstood", replies.MISUNDERSTOOD),
                                        ("different_command", replies.DIFFERENT), ("abort", replies.DECLINE)])
def test_the_remote_read_applies_to_both_kinds_of_decision_once(word, kind):
    for decision, reply in ((CONDITIONAL, "let me think about it"), (MISSING, "let me think about it")):
        calls: list = []
        reading = replies.read_decision(decision, reply, remote=_said(word, calls))
        assert reading.kind == kind and calls == [(decision["question"], reply)]


# ======================================================================================================================
# A conditional step
# ======================================================================================================================

def test_conditional_you_misunderstood_stops_the_command_declined_and_runs_nothing(library_env, monkeypatch):
    # Old: an unreadable reply, asked once more. New: the plan stops at the step, declined, and says why.
    _asked(library_env, monkeypatch, Conditional)
    done = library_env.ctx.process_turn("you misunderstood")
    assert done.status is TurnStatus.COMPLETED
    (stop,) = _stops(done)
    assert stop["open_step"]["reason"].startswith("declined") and "not what you meant" in stop["open_step"]["reason"]
    assert stop["open_step"]["kind"] == "conditional" and library_env.ctx.waiting is None
    assert [command for command in _sent(library_env) if command.startswith("find_book")] == ["find_book query=Dune"]


def test_conditional_a_request_for_a_moment_keeps_the_question_and_counts_nothing(library_env, monkeypatch):
    first = _asked(library_env, monkeypatch, Conditional)
    remote = _Remote(monkeypatch, "wait")
    waited = library_env.ctx.process_turn("hold on, let me check")
    assert waited.status is TurnStatus.AWAITING_USER and len(remote.calls) == 1
    assert waited.answer.startswith(replies.WAIT_TEXT) and first.answer in waited.answer
    assert _decision(library_env)["asked"] == 1
    # The same question is still out and a later plain reply answers it; the wait did not stack a second prefix.
    # A second wait in a row counts as a reply that gave nothing usable (fix-b3w2): the question is asked once more.
    again = library_env.ctx.process_turn("hold on, let me check")
    assert again.status is TurnStatus.AWAITING_USER and _decision(library_env)["asked"] == 2
    done = library_env.ctx.process_turn("no")
    assert done.status is TurnStatus.COMPLETED and _stops(done) is None


def test_conditional_a_third_wait_in_a_row_stops_the_command_and_a_reply_between_resets_the_count(library_env,
                                                                                                   monkeypatch):
    _asked(library_env, monkeypatch, Conditional)
    _Remote(monkeypatch, "wait")
    library_env.ctx.process_turn("hold on")
    library_env.ctx.process_turn("hold on")
    done = library_env.ctx.process_turn("hold on")
    assert done.status is TurnStatus.COMPLETED and library_env.ctx.waiting is None and _stops(done)


def test_conditional_a_different_request_stops_the_command_and_is_not_run_as_a_request(library_env, monkeypatch):
    _asked(library_env, monkeypatch, Conditional)
    remote = _Remote(monkeypatch, "different_command")
    done = library_env.ctx.process_turn("show me the users instead")
    assert done.status is TurnStatus.COMPLETED and len(remote.calls) == 1
    (stop,) = _stops(done)
    assert stop["open_step"]["reason"].startswith("declined") and "not run as a new request" in stop["open_step"]["reason"]
    assert [command for command in _sent(library_env) if command.startswith("find_book")] == ["find_book query=Dune"]


def test_conditional_the_remote_models_abort_stops_the_command(library_env, monkeypatch):
    _asked(library_env, monkeypatch, Conditional)
    remote = _Remote(monkeypatch, "abort")
    done = library_env.ctx.process_turn("forget it, drop this one")
    (stop,) = _stops(done)
    assert len(remote.calls) == 1 and stop["open_step"]["reason"].startswith("declined")


def test_conditional_yes_and_no_ask_no_model_even_with_a_remote_model_set(library_env, monkeypatch):
    _asked(library_env, monkeypatch, Conditional)
    remote = _Remote(monkeypatch, "wait")
    done = library_env.ctx.process_turn("yes")
    assert done.status is TurnStatus.COMPLETED and remote.calls == []


def test_conditional_the_asking_rule_is_unchanged_asked_once_more_then_the_stop(library_env, monkeypatch):
    _asked(library_env, monkeypatch, Conditional)
    again = library_env.ctx.process_turn("perhaps")
    assert again.status is TurnStatus.AWAITING_USER and again.answer.startswith("I could not read that reply as yes or no. ")
    assert _decision(library_env)["asked"] == 2
    ended = library_env.ctx.process_turn("perhaps")
    (stop,) = _stops(ended)
    assert "twice" in stop["open_step"]["reason"]


# ======================================================================================================================
# A command step missing one plain value
# ======================================================================================================================

def test_missing_value_you_misunderstood_stops_the_command_and_is_not_taken_as_the_value(library_env, monkeypatch):
    # Old: the phrase itself became the note. New: the plan stops at the step, declined, and says why.
    _asked(library_env, monkeypatch, MissingValue)
    done = library_env.ctx.process_turn("you misunderstood")
    (stop,) = _stops(done)
    assert stop["open_step"]["kind"] == "missing_value" and "not what you meant" in stop["open_step"]["reason"]
    assert not any(command.startswith("add_note") for command in _sent(library_env))


def test_missing_value_a_request_for_a_moment_keeps_the_question_and_counts_nothing(library_env, monkeypatch):
    first = _asked(library_env, monkeypatch, MissingValue)
    remote = _Remote(monkeypatch, "wait")
    waited = library_env.ctx.process_turn("one second please")
    assert waited.status is TurnStatus.AWAITING_USER and len(remote.calls) == 1
    assert waited.answer.startswith(replies.WAIT_TEXT) and first.answer in waited.answer
    assert _decision(library_env)["asked"] == 1
    assert not any(command.startswith("add_note") for command in _sent(library_env))


def test_missing_value_a_different_request_stops_the_command_and_is_not_run_as_a_request(library_env, monkeypatch):
    _asked(library_env, monkeypatch, MissingValue)
    remote = _Remote(monkeypatch, "different_command")
    done = library_env.ctx.process_turn("list all the members")
    (stop,) = _stops(done)
    assert len(remote.calls) == 1 and "not run as a new request" in stop["open_step"]["reason"]
    assert not any(command.startswith("add_note") for command in _sent(library_env))


def test_missing_value_a_plain_value_is_asked_of_the_remote_model_once_and_kept(library_env, monkeypatch):
    _asked(library_env, monkeypatch, MissingValue)
    remote = _Remote(monkeypatch, "answer")
    done = library_env.ctx.process_turn("the cover is torn")
    assert done.status is TurnStatus.COMPLETED and len(remote.calls) == 1
    assert "add_note text=the cover is torn" in _sent(library_env)


def test_missing_value_a_named_line_is_the_value_and_the_remote_model_is_not_asked(library_env, monkeypatch):
    _asked(library_env, monkeypatch, MissingValue)
    remote = _Remote(monkeypatch, "wait")
    done = library_env.ctx.process_turn("text: the cover is torn")
    assert done.status is TurnStatus.COMPLETED and remote.calls == []
    assert "add_note text=the cover is torn" in _sent(library_env)


def test_missing_value_the_asking_rule_is_unchanged_asked_once_more_then_the_stop(library_env, monkeypatch):
    _asked(library_env, monkeypatch, MissingValue)
    again = library_env.ctx.process_turn("   ")
    assert again.status is TurnStatus.AWAITING_USER and _decision(library_env)["asked"] == 2
    ended = library_env.ctx.process_turn("   ")
    (stop,) = _stops(ended)
    assert "twice" in stop["open_step"]["reason"]


def test_a_decision_reply_with_no_key_asks_no_model(library_env, monkeypatch):
    _asked(library_env, monkeypatch, MissingValue)
    remote = _Remote(monkeypatch, None)
    done = library_env.ctx.process_turn("the cover is torn")
    assert remote.calls == [] and done.status is TurnStatus.COMPLETED
