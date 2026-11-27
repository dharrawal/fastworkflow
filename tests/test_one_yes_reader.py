"""ONE reader for every yes (owner decision 2026-11-23: "yes|true|ok|agreed|accept|go - count these as 'yes'").

The WHOLE reply, case and a trailing ``.`` or ``!`` ignored, is a yes when it is one of ``yes``, ``true``, ``ok``,
``agreed``, ``accept``, ``go`` (``no_agent_mode.is_yes``). Every place a yes is read as consent or confirmation reads it:
the approval question (``review_engine.ActionReplyPolicy``), the turn's review (its plain accept and the explicit yes a
turn that changes data needs), a pinned plan's parameter question (``parameters.read`` with ``confirm``), the
one-candidate "Did you mean the command 'X'?" question (``intent_loop.read``) and a conditional decision step
(``parameters.reply.read_decision``). ``y`` and ``run it`` are not in the set and are no yes anywhere.

No model is called: each reader is the real one, given the reply directly.
"""
from __future__ import annotations

import pytest

from fastworkflow.intent import intent_loop
from fastworkflow.core import no_agent_mode
from fastworkflow.parameters import read, reply as replies
from fastworkflow.review.review_engine import ActionReplyPolicy, Verdict

YES = ["yes", "true", "ok", "agreed", "accept", "go", "Yes", "YES", "yes.", "OK!", " agreed ", "Go.", "TRUE"]
NOT_YES = ["y", "Y", "run it", "okay", "run", "confirm", "sure", "yes please", "go ahead", "yes, go", "",
           "no", "always", "accept it"]
CONDITIONAL = {"kind": no_agent_mode.STEP_CONDITIONAL, "condition": "c", "inner": "i", "question": "Does it hold?"}


@pytest.mark.parametrize("reply", YES)
def test_the_owners_words_are_a_yes(reply):
    assert no_agent_mode.is_yes(reply) is True
    assert no_agent_mode.read_yes_no(reply) is True


@pytest.mark.parametrize("reply", NOT_YES)
def test_anything_else_is_no_yes(reply):
    assert no_agent_mode.is_yes(reply) is False
    assert no_agent_mode.read_yes_no(reply) is not True


def test_the_set_is_the_owners():
    assert no_agent_mode.YES_WORDS == frozenset({"yes", "true", "ok", "agreed", "accept", "go"})


@pytest.mark.parametrize("reply", YES)
def test_the_approval_question_reads_the_one_yes(reply):
    assert ActionReplyPolicy().classify(reply) is Verdict.CONFIRM


@pytest.mark.parametrize("reply", ["y", "run it", "okay", "yes please", "sure"])
def test_the_approval_question_asks_again_on_anything_else(reply):
    assert ActionReplyPolicy().classify(reply) is Verdict.REPROMPT


@pytest.mark.parametrize("reply", YES)
def test_a_pinned_plans_parameter_question_reads_the_one_yes(reply):
    assert read.read(reply, ["title", "author"], [], confirm=True).kind == read.ACCEPT


@pytest.mark.parametrize("reply", ["y", "run it", "okay", "run", "confirm"])
def test_a_pinned_plans_parameter_question_no_longer_takes_the_old_extra_words(reply):
    assert read.read(reply, ["title", "author"], [], confirm=True).kind != read.ACCEPT


@pytest.mark.parametrize("reply", YES)
def test_did_you_mean_reads_the_one_yes(reply):
    assert intent_loop.read(reply, ["Library/find_book"]).name == "Library/find_book"


@pytest.mark.parametrize("reply", ["y", "run it", "okay"])
def test_did_you_mean_does_not_take_y_or_run_it(reply):
    assert intent_loop.read(reply, ["Library/find_book"]).name is None


@pytest.mark.parametrize("reply", YES)
def test_a_conditional_decision_step_reads_the_one_yes(reply):
    reading = replies.read_decision(CONDITIONAL, reply, phrases=None, remote=None)
    assert (reading.kind, reading.values) == (replies.VALUES, {"answer": "yes"})


@pytest.mark.parametrize("reply", ["y", "run it"])
def test_a_conditional_decision_step_does_not_take_y_or_run_it(reply):
    assert replies.read_decision(CONDITIONAL, reply, phrases=None, remote=None).values != {"answer": "yes"}


@pytest.mark.parametrize("reply, needs", [("ok", False), ("agreed", False), ("Go!", False), ("yes", False),
                                          ("y", True), ("run it", True), ("go ahead", True)])
def test_the_explicit_yes_of_a_turn_that_changes_data_is_the_one_yes(reply, needs):
    from fastworkflow.review import review_engine
    from fastworkflow.review.review_proposals import SubTaskProposal

    class Writing(SubTaskProposal):
        def __init__(self):
            pass

        def changes_data(self, folder):
            return True

    assert review_engine._needs_a_yes(Writing(), reply, None) is needs


@pytest.mark.parametrize("reply, verdict", [("No", Verdict.DECLINE), ("no.", Verdict.DECLINE), (" NO! ", Verdict.DECLINE),
                                            ("Always", Verdict.ALWAYS), ("always.", Verdict.ALWAYS),
                                            ("no thanks", Verdict.REPROMPT), ("nope", Verdict.REPROMPT),
                                            ("always please", Verdict.REPROMPT)])
def test_the_approval_questions_no_and_always_are_read_as_a_yes_is(reply, verdict):
    """The whole reply, case and a trailing ``.`` or ``!`` ignored; the word sets are not widened."""
    assert ActionReplyPolicy().classify(reply) is verdict
