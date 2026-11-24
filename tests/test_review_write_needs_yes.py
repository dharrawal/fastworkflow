"""bd fix-i8im: a review whose turn has a part that changes data is accepted only by an explicit yes.

``dont change anything`` and ``do not change anything`` mean "no edits to the plan" to some readers and "make no change
to my data" to others, so on a turn with a writing part they, and every other accept that is not a plain yes, ask again.
A turn with no writing part keeps its accept cues. No model: the cues and the recorded review head decide.
"""
from __future__ import annotations

import pytest

from fastworkflow.review_engine import CONFIRMED, DECLINED, reply_to_proposal, proposal_of

from tests.test_approval_binding import annotated_library, no_live_model, todo  # noqa: F401
from tests.test_plan_reply_policy import (  # noqa: F401
    _open_review_two_step, no_draft_extraction, planner, script_param_extraction)
from tests.test_round1_gate import gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

reader = pytest.mark.usefixtures("initialized_fastworkflow", "no_live_model", "no_draft_extraction",
                                 "recorded_review_router")

NOT_YES = ["dont change anything", "don't change anything", "do not change anything", "no changes", "do it",
           "go ahead", "sounds good", "lgtm"]


def _open(gated, monkeypatch, *, writes: bool):
    if not writes:
        # the workflow has no command that changes data, from the review's first ask on (an ad hoc part may change data
        # when the workflow has a command that does, bd fix-ftdl.8.4)
        monkeypatch.setattr("fastworkflow.command_writes.write_commands", lambda path: frozenset())
    record = _open_review_two_step(gated, monkeypatch)
    if writes:
        names = {item.means.name for item in proposal_of(record).items if item.means.name}
        # every part of the turn names a command that changes data
        monkeypatch.setattr("fastworkflow.command_writes.write_commands", lambda path: frozenset(names))
    return record


def test_the_whole_accept_list_does_not_hold_dont_change_anything():
    from fastworkflow import review_engine as re_

    assert "dont change anything" not in re_._ACCEPT_PHRASES
    assert not re_._affirmative_accept_cue("dont change anything")


@reader
@pytest.mark.parametrize("text", ["dont change anything", "don't change anything", "do not change anything"])
def test_a_writing_turn_asks_again_on_dont_change_anything(gated, monkeypatch, text, planner, recorded_review_router):
    # The head's recorded answer for "just do it" (a confident confirm) stands in for its read of each spelling: the
    # head is never retrained or re-recorded here, and the rule must hold whatever it says.
    recorded_review_router.answers[text] = recorded_review_router.answers["just do it"]
    planner(gated, *({"edits": []},) * 6)
    record = _open(gated, monkeypatch, writes=True)
    review = reply_to_proposal(gated.ctx, gated.store, record, text)
    assert record.state not in (CONFIRMED, DECLINED) and not review.done
    assert "changes data" in review.text and "yes" in review.text


@reader
@pytest.mark.parametrize("text", ["yes", "Yes.", "accept", "OK", "go", "agreed", "true!"])   # the one yes reader
def test_a_writing_turn_is_accepted_by_a_yes(gated, monkeypatch, text):
    record = _open(gated, monkeypatch, writes=True)
    review = reply_to_proposal(gated.ctx, gated.store, record, text)
    assert record.state == CONFIRMED and review.done


@reader
@pytest.mark.parametrize("text", ["no", "nope"])
def test_a_writing_turn_still_declines(gated, monkeypatch, text):
    record = _open(gated, monkeypatch, writes=True)
    review = reply_to_proposal(gated.ctx, gated.store, record, text)
    assert record.state == DECLINED and review.done


@reader
@pytest.mark.parametrize("text", ["do it", "no changes, go ahead", "sounds good to me", "lgtm", "run it", "okay"])
def test_a_writing_turn_asks_again_on_a_plain_accept_that_is_not_a_yes(gated, monkeypatch, text, planner):
    planner(gated, *({"edits": []},) * 6)
    record = _open(gated, monkeypatch, writes=True)
    review = reply_to_proposal(gated.ctx, gated.store, record, text)
    assert record.state != CONFIRMED and not review.done


@reader
@pytest.mark.parametrize("text", ["do it", "no changes, go ahead", "sounds good to me", "lgtm", "yes"])
def test_a_read_only_turn_accepts_as_before(gated, monkeypatch, text):
    record = _open(gated, monkeypatch, writes=False)
    review = reply_to_proposal(gated.ctx, gated.store, record, text)
    assert record.state == CONFIRMED and review.done


# ---- the first ask says it, and a refused accept ends visibly ---------------------------------------------------

@reader
def test_the_first_ask_of_a_writing_turn_says_a_yes_is_needed(gated, monkeypatch):
    from fastworkflow import review_engine as re_

    record = _open(gated, monkeypatch, writes=True)
    text = re_.prompt_for(proposal_of(record), gated.wf)      # the question as a review opened now would show it
    assert text.endswith(f"{re_.SUB_TASK_PROMPT_HINT} {re_.CHANGES_DATA_SENTENCE}")
    assert re_.CHANGES_DATA_SENTENCE in re_.prompt_for(proposal_of(record), gated.wf, structured=True)
    forms = re_.reply_forms(text, proposal_of(record))
    assert {"kind": "accept", "reply": "yes"} in [{k: f[k] for k in ("kind", "reply") if k in f} for f in forms]


@reader
def test_the_first_ask_of_a_read_only_turn_keeps_todays_hint(gated, monkeypatch):
    from fastworkflow import review_engine as re_

    record = _open(gated, monkeypatch, writes=False)
    assert record.rendering.endswith(re_.SUB_TASK_PROMPT_HINT)
    assert re_.CHANGES_DATA_SENTENCE not in record.rendering


@reader
def test_non_yes_accepts_in_a_row_are_asked_again_then_the_review_expires_unrun(gated, monkeypatch, planner, recorded_review_router):
    """Every non-yes accept is asked again (never run); at the reply cap the review expires as not run, with a plain
    message, and no accept ever confirmed it."""
    from fastworkflow import approval_binding as ab
    from fastworkflow import review_engine as re_

    recorded_review_router.answers["dont change anything"] = recorded_review_router.answers["just do it"]
    planner(gated, *({"edits": []},) * 40)
    record = _open(gated, monkeypatch, writes=True)
    ended = None
    for _ in range(re_.PROPOSAL_TTL_TURNS + 3):
        resolution = ab.resolve_reply(gated.ctx, gated.wf, "dont change anything")
        assert record.state != CONFIRMED
        if resolution.done:
            ended = resolution
            break
        assert "yes" in resolution.text
    assert ended is not None and record.state == re_.EXPIRED
    assert "expired" in ended.text and "not run" in ended.text
