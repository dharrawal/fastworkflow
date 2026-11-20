"""bd fix-blui: plain accept and decline cues, and the polarity guard on the classifier's confirm or decline.

The cue matcher is a pure function: no model, no head. The reader tests that follow use replies the cues decide
before the classifier (no router needed) and replies the recorded router already holds.
"""
from __future__ import annotations

import pytest

from fastworkflow import review_engine as re_
from fastworkflow.review_engine import CONFIRMED, DECLINED
from fastworkflow.review_engine import _affirmative_accept_cue, _plain_decline_cue, _polarity_conflict

from tests.test_approval_binding import annotated_library, no_live_model, todo  # noqa: F401
from tests.test_plan_reply_policy import (  # noqa: F401
    _open_review_two_step, no_draft_extraction, planner, script_param_extraction)
from tests.test_round1_gate import gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401


def _cue(text: str) -> str:
    accept, decline = _affirmative_accept_cue(text), _plain_decline_cue(text) or text.strip().lower() in ("no", "decline")
    assert not (accept and decline), f"{text!r} is both"
    return "accept" if accept else "decline" if decline else "none"


@pytest.mark.parametrize("text", [
    "do it", "Run it", "ship it", "go", "go for it", "let's go", "let’s do it", "proceed", "please proceed",
    "continue", "carry on", "make it so", "lgtm", "LGTM!", "all good", "all good, proceed", "all good proceed",
    "that works", "works for me", "fine", "k", "kk", "k thanks", "nothing to change", "no changes",
    "no changes, go ahead", "no changes needed", "no change", "no edits", "no objections", "no problem",
    "as is", "run as is", "run the plan as is", "\U0001F44D", "\U0001F44D\U0001F44D", "\U0001F44D\U0001F3FD",
    "lgtm \U0001F44D", "ok \U0001F44D", "don't change anything, go ahead", "do it now", "go ahead",
])
def test_plain_accepts(text):
    assert _cue(text) == "accept"


@pytest.mark.parametrize("text", [
    "no", "No.", "nope", "nah", "nay", "no thanks", "no, thanks", "cancel", "stop", "abort", "don't", "do not",
    "don't run this", "don't run it", "do not run this", "do not run it", "please don't run this", "I'll pass",
    "pass", "not now", "never mind", "nevermind", "scratch that", "scratch that plan", "forget it", "reject",
    "decline", "discard", "no, cancel", "cancel it", "nope!",
])
def test_plain_declines(text):
    assert _cue(text) == "decline"


@pytest.mark.parametrize("text", [
    # anchoring traps: the cue word opens or sits inside a sentence about an edit, so it is no cue
    "go back and change step 2", "go back to step 1", "pass the id to step 3", "stop after step 2",
    "fine, but drop the last one", "fine but make it monthly", "ok but make it monthly", "yes but skip the last step",
    "sure, change the first one", "cancel the third one", "no, change the second one to weekly",
    "no changes to step 1, but add a step", "do it later", "do it once the test passes", "continue with step 2 only",
    "don't change the second one", "don't do it", "k, but drop step 2",
    # no polarity
    "not sure", "maybe", "hmm", "", "   ", "what do you think?", "asdf", "proced", "okk",
    # a typed edit clause keeps its own path
    "change 1: goal=run it as is", "drop 2",
])
def test_no_cue(text):
    assert _cue(text) == "none"


def test_a_comma_after_no_makes_change_an_edit_not_an_accept():
    assert _cue("no change") == "accept"
    assert _cue("no, change") == "none"
    assert _cue("no - change the goal") == "none"


@pytest.mark.parametrize("text, routed", [
    ("do it later", "decline"), ("do it later", "confirm"), ("hmm", "confirm"), ("hmm", "decline"),
    ("maybe", "confirm"), ("not sure", "confirm"), ("nah, it's fine, go ahead", "confirm"),
    ("nah, it's fine, go ahead", "decline"), ("just do it", "decline"), ("great, proceed", "decline"),
    ("hell no", "confirm"), ("no way", "confirm"), ("not interested, cancel", "confirm"),
    ("cancel the third one", "decline"), ("stop after step 2", "decline"), ("hold on", "confirm"),
])
def test_polarity_guard_refuses_to_act(text, routed):
    assert _polarity_conflict(text, routed)


@pytest.mark.parametrize("text, routed", [
    # no polarity signal either way: the route is acted on as before
    ("I'm happy with that", "confirm"), ("that's right", "confirm"), ("exactly", "confirm"),
    ("good", "confirm"), ("okk", "confirm"), ("unclear one", "decline"), ("make it better somehow", "decline"),
    # the reply agrees with the route
    ("never mind", "decline"), ("great, proceed", "confirm"), ("perfect", "confirm"),
    ("no changes, go ahead", "confirm"), ("nope", "decline"), ("don't run it", "decline"),
])
def test_polarity_guard_lets_an_agreeing_or_silent_reply_through(text, routed):
    assert _polarity_conflict(text, routed) == ""


def test_cues_never_match_a_phrase_inside_a_longer_reply():
    for word in ("go", "pass", "fine", "stop", "k", "no", "cancel", "continue"):
        assert _cue(f"{word} the id to step 3") == "none"
        assert _cue(f"please {word} after the second step") == "none"


def test_the_decision_notes_the_guard_without_changing_the_reason():
    decision = re_.ReplyDecision(re_.Verdict.REPROMPT, "not read as edits")
    assert decision.note == ""


# ---- through the reader -------------------------------------------------------------------------------------
# The three wrong-direction replies of the 2026-11-13 measurement are decided by a cue before the classifier, so these
# need no recorded head answer. The guard tests use replies the recorded router holds (tests/recorded_router.py).

reader = pytest.mark.usefixtures("initialized_fastworkflow", "no_live_model", "no_draft_extraction",
                                 "recorded_review_router")


def _reply(gated, monkeypatch, text):
    from fastworkflow.review_engine import reply_to_proposal

    record = _open_review_two_step(gated, monkeypatch)
    return record, reply_to_proposal(gated.ctx, gated.store, record, text)


@reader
@pytest.mark.parametrize("text", ["do it", "no changes, go ahead", "k", "lgtm \U0001F44D", "nothing to change"])
def test_a_plain_accept_confirms_the_plan(gated, monkeypatch, text):
    record, review = _reply(gated, monkeypatch, text)
    assert record.state == CONFIRMED and review.done


@reader
@pytest.mark.parametrize("text", ["nah", "nope", "I'll pass", "scratch that plan", "no thanks"])
def test_a_plain_decline_declines_the_plan(gated, monkeypatch, text):
    record, review = _reply(gated, monkeypatch, text)
    assert record.state == DECLINED and review.done


@reader
@pytest.mark.parametrize("text", [
    "fine, but drop the last one", "no, change the second one to weekly", "go back and change step 2",
    "stop after step 2", "cancel the third one",
])
def test_a_reply_that_asks_for_an_edit_is_not_taken_for_a_whole_plan_answer(gated, monkeypatch, text, planner):
    planner(gated, *({"edits": []},) * 6)
    record, review = _reply(gated, monkeypatch, text)
    assert record.state not in (CONFIRMED, DECLINED)


#: Replies the head routes to a decline or a confirm that the reader must not act on (recorded answers).
#: "just do it" (a confident decline) and "hmm" (a confident confirm) used to be here, from the per-context head.
#: The framework's embedding head answers "just do it" with a confident confirm (right: see
#: test_a_plain_accept_the_head_routes_to_confirm_is_acted_on) and "hmm" with the not-here class (no command at all:
#: see test_a_hedge_the_head_calls_not_here_asks_again). The polarity guard itself is still pinned for both strings
#: by `_polarity_conflict`'s own table above, whatever a head says.
GUARDED = [
    "nah, it's fine, go ahead",        # head: confirm, an accept and a decline word together
]


@reader
@pytest.mark.parametrize("text", GUARDED)
def test_the_reader_asks_again_when_the_routed_answer_contradicts_the_reply(gated, monkeypatch, text, planner):
    planner(gated, *({"edits": []},) * 6)
    record, review = _reply(gated, monkeypatch, text)
    assert record.state not in (CONFIRMED, DECLINED) and not review.done


@reader
@pytest.mark.parametrize("text", GUARDED)
def test_the_decision_records_why_it_did_not_act(gated, monkeypatch, text, planner):
    from fastworkflow.review_engine import Verdict, proposal_of, reply_policy

    planner(gated, *({"edits": []},) * 6)
    record = _open_review_two_step(gated, monkeypatch)
    decision = reply_policy(
        record.kind, gated.store.edit_reader,
        plan_cme=gated.store.plan_cme,
        plan_workflow=gated.store.plan_workflow,
        app_workflow=gated.wf,
    ).decide(proposal_of(record), text)
    assert decision.verdict is Verdict.REPROMPT and decision.reason == "not read as edits"
    assert decision.note.startswith("polarity guard: routed ")


@reader
def test_a_plain_accept_the_head_routes_to_confirm_is_acted_on(gated, monkeypatch):
    record, review = _reply(gated, monkeypatch, "just do it")     # head: confirm (confident), the reply is an accept
    assert record.state == CONFIRMED and review.done


@reader
def test_a_hedge_the_head_calls_not_here_asks_again(gated, monkeypatch, planner):
    planner(gated, *({"edits": []},) * 6)
    record, review = _reply(gated, monkeypatch, "hmm")            # head: the not-here class (wildcard), no command
    assert record.state not in (CONFIRMED, DECLINED) and not review.done


@reader
def test_an_agreeing_route_is_still_acted_on(gated, monkeypatch):
    record, review = _reply(gated, monkeypatch, "hell no")        # head: decline, the reply is negative
    assert record.state == DECLINED and review.done


@reader
def test_a_reply_with_no_polarity_word_is_acted_on_as_before(gated, monkeypatch):
    record, review = _reply(gated, monkeypatch, "I'm happy with that")   # head: confirm; no cue, no guard signal
    assert record.state == CONFIRMED and review.done


# ---- bd fix-yu3l.6: an accept that goes on to ask for something more is not a plain accept -------------------------
# Pure functions and the policy with a scripted reader: no head, no recorded router answer is needed.

ACCEPT_THEN_REQUEST = [
    "yes, and show me the overdue books", "Yes. Then list the members", "sure, afterwards count the books",
    "yes and show me the overdue books", "Yes, then list the members.", "ok, after that list the members",
    "yes please, and then count the books", "go ahead, then show me the overdue books",
    "sounds good, plus list the members", "Yes, go ahead, and show me the overdue books",
    "looks good. Afterwards, count the books", "do it, and then list the members",
    "that works, and after that count the books", "okay and next list the members",
    # an accept that then contradicts the plan (a contrasting connector) is not a plain accept
    "Yes, but Atlas is a group, not an application. Please treat it as a group.", "yes but make it a group",
    "ok, however use the other branch", "yes, except step 2", "yes, although use the second branch",
]

STILL_PLAIN_ACCEPTS = [
    "yes", "Yes.", "yes please", "ok go ahead", "yes, run it", "sure, that works", "ok go ahead and run it",
    "yes and go ahead", "Yes, and please go ahead", "yes and thanks", "ok then", "ok then, run it",
    "sure, then go ahead", "all good, proceed", "no changes, go ahead", "do it now", "sounds good to me",
    "looks good", "go ahead", "Yes, that is right, run it as shown",
    # an accept that restates the plan, or qualifies it with no effect, is still an accept
    "Yes, go ahead and remove the audit-scope tag from the HR feed",
    "Yes, list every right Brent Randolph holds for recertification.",
    "Yes—finding summary first, then Catherine Taylor offboarding with corrections shown but not applied.",
    "yes, run that, and apply no corrections", "Yes, go ahead. Just do not apply any corrections.",
    "yes, but nothing else", "yes but apply no corrections", "yes, go ahead", "yes, just go ahead", "yes, but do not apply any corrections",
]


@pytest.mark.parametrize("text", ACCEPT_THEN_REQUEST)
def test_an_accept_followed_by_a_request_is_not_a_plain_accept(text):
    assert re_._accept_then_request(text)
    assert not _affirmative_accept_cue(text)
    assert not re_._affirmative_unrouted_confirm(text)
    assert not re_._slot_whole_plan_confirm(text)


@pytest.mark.parametrize("text", STILL_PLAIN_ACCEPTS)
def test_accept_cues_stay_plain_accepts(text):
    assert not re_._accept_then_request(text)
    assert _affirmative_accept_cue(text)


def _two_step_proposal():
    from fastworkflow.review_proposals import SubTask, SubTaskProposal

    return SubTaskProposal(items=[SubTask(goal="List the books"), SubTask(goal="Count the members")])


@pytest.mark.parametrize("text", ACCEPT_THEN_REQUEST)
def test_an_accept_followed_by_a_request_goes_to_the_edit_reader(text):
    """The path of a reply with a corrective word: the reader is asked, and what it reads is the decision."""
    from fastworkflow.review_engine import Add, Verdict

    seen = []

    def reader(reply, proposal):
        seen.append(reply)
        return [Add(after=2, values={"goal": "the extra request"})]

    decision = re_.ProposalReplyPolicy(reader).decide(_two_step_proposal(), text)
    assert seen == [text]
    assert decision.verdict is Verdict.EDIT and decision.edits == (Add(after=2, values={"goal": "the extra request"}),)


@pytest.mark.parametrize("text", ACCEPT_THEN_REQUEST[:3])
def test_an_accept_followed_by_a_request_the_reader_cannot_read_asks_again(text):
    """A reader that reads no change does not turn the reply back into an accept: the person is asked again."""
    from fastworkflow.review_engine import Change, Verdict

    def reader(reply, proposal):
        return [Change(index=1, field="goal", value="List the books")]       # the plan as it stands

    decision = re_.ProposalReplyPolicy(reader).decide(_two_step_proposal(), text)
    assert decision.verdict is Verdict.REPROMPT


@pytest.mark.parametrize("text", STILL_PLAIN_ACCEPTS)
def test_a_plain_accept_never_reaches_the_edit_reader(text):
    from fastworkflow.review_engine import Verdict

    def reader(reply, proposal):
        raise AssertionError(f"the edit reader was asked about {reply!r}")

    assert re_.ProposalReplyPolicy(reader).decide(_two_step_proposal(), text).verdict is Verdict.CONFIRM
