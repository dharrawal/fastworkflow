"""Cancelling a pending sub-task review leaves no review state behind (bd fix-fpyb.2).

Real components: the library copy and the scripted model steps of the round-1 gate tests.
"""
from __future__ import annotations

import json

import pytest

from fastworkflow.review import approval_binding as ab
from fastworkflow.core.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - fixtures
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, FIND, JOKE, _open_review, _split_into, gated, remote_decides_emma_and_joke  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _cancel_via_cancel_pending(ctx):
    return ctx.cancel_pending()


def _cancel_via_approvals(ctx):
    return ctx.cancel_pending_approvals()


def _cancel_via_reset(ctx):
    ctx._reset_agent_suspension()


@pytest.mark.parametrize("route", [_cancel_via_cancel_pending, _cancel_via_approvals, _cancel_via_reset])
def test_cancelling_an_open_review_leaves_nothing_pending(gated, monkeypatch, route, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    asked = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert asked.status is TurnStatus.AWAITING_USER
    _open_review(gated)
    assert gated.ctx._waiting is not None and gated.ctx._waiting.kind == "review"

    route(gated.ctx)

    assert gated.ctx._waiting is None
    assert not gated.ctx.awaiting_user
    if route is not _cancel_via_reset:      # the private reset drops the suspension, not the approval records
        assert ab.active_reviews(gated.wf) == []
        assert gated.ctx.has_pending_state() is False
    state = json.loads(json.dumps(gated.ctx.serialize_state(channel_id="cancel-review")))
    assert state["waiting"] is None

    # The next ordinary message is a new request, not the review's answer.
    if route is _cancel_via_reset:
        return
    _split_into(monkeypatch, [FIND])
    gated.script.steps = _finish()
    out = gated.ctx.process_turn(FIND)
    assert out.status is TurnStatus.COMPLETED
