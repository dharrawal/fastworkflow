"""The round-1 sub-task review gate (fix-3rsq.plan-review.2.2, .2.3, .2.6, .2.8).

A turn's sub-tasks, after split and merge, either run now (the fast path: one sub-task with an exact command name or a
confident head match) or are reviewed first: the draft proposal, built from real candidates, opens review_subtasks. Real
library copy with its trained heads; the agent is the real agent with a scripted model (the ``todo`` fixture from the
approval tests). The planner's split is the one seam faked here, as the other agent tests fake it. No test calls a model.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from fastworkflow._workflows.plan_metadata_extraction import resolution

import fastworkflow
from tests import span_tree
from fastworkflow.review import review_proposals
from fastworkflow import workflow_agent
from fastworkflow.observability import tracing
from fastworkflow.review.approval_binding import Surface
from fastworkflow.planning.planning_commands import planning_command
from fastworkflow.review.review_engine import AWAITING, CONFIRMED, DECLINED, EXPIRED, PROPOSAL_TTL_SECONDS, proposal_of
from fastworkflow.review.review_proposals import SubTaskProposal
from fastworkflow.run_listing import check_planned_arguments
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.conftest import script_matcher
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    """The draft's argument extraction is a model call (fix-48np): off here, so these tests keep asserting that the
    review, the gate and the hand-off make no parameter-extraction model call (the guard fixture raises if one is
    made). Drafting with extraction on is covered by tests/test_drafted_subtask_inputs.py."""
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"
EMMA = "who wrote Emma and how many copies do we have"    # heads: a planning command in the top-k (book_by_title)
JOKE = "tell Ann a joke about cats"                        # heads: unsure, no command fits
FIND = "find member Ann Lee"                               # heads: one confident label (find_member)


#: The remote decision model's answer for the two parts the local model is unsure of (it gives each a list of
#: candidates): EMMA is the planning command ``book_by_title``, and no command fits JOKE, so JOKE is an ad hoc part.
#: Scripted (``script_matcher``), with no model call: these tests used to get it from a live call, because litellm
#: loaded a developer's .env with a ``JEV_API_KEY`` into the process. Without an answer the turn asks which command
#: is meant (the ambiguity question) instead of opening the review.
REMOTE_ANSWERS = {EMMA: {"Library/book_by_title": 0.95, "none": 0.05}, JOKE: {"none": 0.95}}


@pytest.fixture
def remote_decides_emma_and_joke(monkeypatch):
    """The remote decision model answers EMMA and JOKE as ``REMOTE_ANSWERS`` says; any other text gets no answer, so
    the local model's decision stands for it."""
    return script_matcher(monkeypatch, REMOTE_ANSWERS)


@pytest.fixture
def gated(todo):
    """The approval tests' ``todo`` context with the review gate restored (``todo`` bypasses it for those tests), at the
    library's root context, where the sub-tasks are run."""
    vars(todo.ctx).pop("_subtask_gate", None)
    todo.wf.record = todo.library
    return todo


def _split_into(monkeypatch, subtasks):
    """The planner's split (the one model-backed seam before the gate): the sub-tasks it returns."""
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", lambda user_query, **kwargs: list(subtasks))


def _marker(todo) -> str | None:
    """What the last recorded turn's tree says about its review (span contract 63: the review's outcome is on its
    ``fw.review`` span, no longer on the turn's): the review's ``outcome`` (open, reviewed, declined, expired); with no
    review, ``fast_path`` when a part took the gate's fast path, else ``not_reviewed`` for a turn that went through the
    stages; None for a turn that recorded none."""
    turns = [span for span in todo.sink.spans if span.name == tracing.SPAN_TURN]
    if not turns:
        return None
    spans = [span for span in span_tree.spans(todo.sink) if span.trace_id == turns[-1].trace_id]
    reviews = [span for span in spans if span.name == tracing.SPAN_REVIEW]
    if reviews:
        return reviews[-1].attributes.get("outcome")
    if any(span.name == tracing.SPAN_RESOLVE_PART and span.attributes.get("fast_path") for span in spans):
        return "fast_path"
    return "not_reviewed" if any(span.name == tracing.SPAN_RESOLVE for span in spans) else None


def _open_review(todo):
    record = todo.store.active()
    assert record is not None and record.command == "review_subtasks" and record.kind == "sub_tasks"
    assert record.state == AWAITING
    return record


def _ran(out):
    """The turn's command outputs, without the user's own reply to a question (an ask_user entry)."""
    return [output for output in out.command_outputs if not getattr(output, "ask_user_entry", False)]


def _confirmed(todo):
    return [record for record in todo.store.records if record.state == CONFIRMED]


# ----------------------------------------------------------------------
# The fast path (2.2): no review
# ----------------------------------------------------------------------


def test_a_single_confident_sub_task_runs_directly_with_no_review(gated, monkeypatch):
    _split_into(monkeypatch, [FIND])
    intent = resolution.part_intent(gated.wf, FIND)
    assert resolution.is_fast_path(intent, gated.wf) and intent.decided_by == resolution.BY_CLASSIFIER
    gated.script.steps = _finish()
    out = gated.ctx.process_turn(FIND)
    assert out.status is TurnStatus.COMPLETED
    assert gated.store.records == []                       # no review was ever opened
    assert _marker(gated) == "fast_path"


def test_an_exact_command_name_runs_directly_with_no_review(gated, monkeypatch):
    _split_into(monkeypatch, ["count_books"])
    fast = resolution.part_intent(gated.wf, "count_books")
    assert resolution.is_fast_path(fast, gated.wf) and fast.decided_by == resolution.BY_EXACT_NAME
    assert fast.command == "Library/count_books" and fast.kind == resolution.KIND_PLANNING
    gated.script.steps = _finish()
    out = gated.ctx.process_turn("count_books")
    assert out.status is TurnStatus.COMPLETED
    assert gated.store.records == []
    assert _marker(gated) == "fast_path"


def test_an_unsure_single_sub_task_is_not_on_the_fast_path(gated, monkeypatch):
    _split_into(monkeypatch, [JOKE])
    assert not resolution.is_fast_path(resolution.part_intent(gated.wf, JOKE), gated.wf)


# ----------------------------------------------------------------------
# The candidate builder (2.3) and the review (2.6)
# ----------------------------------------------------------------------


def test_a_multi_sub_task_request_opens_the_review_with_the_proposed_means(gated, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert out.status is TurnStatus.AWAITING_USER
    record = _open_review(gated)
    items = record.binding["items"]
    # A planning command in the sub-task's top-k is preferred; a sub-task no command fits is built from scratch, and
    # the framework's own commands (go_up, reset_context, ...) are never offered as means.
    assert items[0]["means"] == {"kind": "planning", "name": "Library/book_by_title"}
    assert "means:" not in record.rendering
    assert "1. who wrote Emma and how many copies do we have" in record.rendering
    assert "means: planning:Library/book_by_title" in proposal_of(record).model_rendering()
    assert _marker(gated) == "open"


def test_an_unsure_single_sub_task_opens_the_review(gated, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [JOKE])
    out = gated.ctx.process_turn(JOKE)
    assert out.status is TurnStatus.AWAITING_USER
    record = _open_review(gated)
    assert len(record.binding["items"]) == 1 and record.binding["items"][0]["goal"] == JOKE


def test_a_refused_planning_argument_is_shown_on_its_line(gated, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    refusal = check_planned_arguments(planning_command(gated.wf.folderpath, "Library/book_by_title"), {}, EMMA).refusal
    assert refusal, "the planning command refuses an empty argument set, so the check has something to show"
    assert proposal_of(record).items[0].check == refusal
    model_line = next(text for text in proposal_of(record).model_rendering().splitlines() if text.startswith("1. "))
    assert f"[check: {refusal}]" in model_line


def test_the_draft_round_trips_through_the_review_as_json(gated, monkeypatch, remote_decides_emma_and_joke):
    """The draft is passed to review_subtasks as JSON, read back as the same items, and stored as the record's binding."""
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    rebuilt = SubTaskProposal.model_validate({"items": record.binding["items"]})
    assert rebuilt.items[0].goal == EMMA and rebuilt.items[1].goal == JOKE
    assert SubTaskProposal.from_text(rebuilt.model_dump_json()) == rebuilt


# ----------------------------------------------------------------------
# Replies: decline and expiry run nothing; accept stores the plan (the 2.9 hook)
# ----------------------------------------------------------------------


def test_a_declined_review_runs_nothing(gated, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    out = gated.ctx.process_turn("no")
    assert record.state == DECLINED
    assert out.status is TurnStatus.COMPLETED
    assert "declined" in (out.answer or "").lower() and "nothing was run" in (out.answer or "")
    assert _ran(out) == []                                  # no sub-task ran
    assert _confirmed(gated) == []
    assert _marker(gated) == "declined"


def test_an_expired_review_runs_nothing(gated, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    record.created_at = time.time() - PROPOSAL_TTL_SECONDS - 1
    out = gated.ctx.process_turn("accept")
    assert record.state == EXPIRED
    assert "expired" in (out.answer or "") and "nothing was run" in (out.answer or "")
    assert _ran(out) == []
    assert _confirmed(gated) == []
    assert _marker(gated) == "expired"


def test_an_accepted_review_is_stored_for_the_hand_off_and_the_turn_runs_its_goals(gated, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    gated.ctx.process_turn("change 2: inputs=Ann Lee")
    # The review frame used to take its own finish. Its result-driven last phase now takes
    # none, so only the two sub-tasks need one.
    gated.script.steps = [("finish", {})] * 2
    out = gated.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED
    assert gated.script.steps == []
    accepted = review_proposals.accepted_subtask_proposal(gated.wf)
    assert accepted is not None and accepted.items[1].inputs[0].value == "Ann Lee"
    assert gated.ctx._turn_subtask_plan == accepted        # the hand-off reads this (fix-3rsq.plan-review.2.9)
    assert _marker(gated) == "reviewed"


# ----------------------------------------------------------------------
# No answerer (2.8): unattended and host turns run the draft, recorded as not reviewed
# ----------------------------------------------------------------------


def _assert_ran_unreviewed(gated, out):
    assert out.status is TurnStatus.COMPLETED
    assert [r for r in gated.store.records if r.command == "review_subtasks"] == []   # the review never opened
    assert gated.script.steps == []
    plan = gated.ctx._turn_subtask_plan
    assert isinstance(plan, SubTaskProposal) and [item.goal for item in plan.items] == [EMMA, JOKE]
    assert _marker(gated) == "not_reviewed"


def test_an_unattended_turn_runs_the_draft_unreviewed(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True)
    _assert_ran_unreviewed(gated, out)


def test_a_host_surface_turn_runs_the_draft_unreviewed(gated, monkeypatch, remote_decides_emma_and_joke):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.script.steps = _finish() + _finish()
    surface = Surface(kind="host", verified=True, uid="user-1")
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", approval_surface=surface)
    _assert_ran_unreviewed(gated, out)


# ----------------------------------------------------------------------
# Jev (2.3): optional; with no key it is never called
# ----------------------------------------------------------------------


def test_with_no_jev_key_jev_is_never_called_and_the_draft_is_still_built(gated, monkeypatch, no_planner_key):
    calls: list = []

    def refuse_jev(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Jev was called without JEV_API_KEY")

    monkeypatch.setattr(workflow_agent, "TypeSafe", refuse_jev)
    assert workflow_agent._jev_lm() is None
    draft = review_proposals.draft_round1_proposal(gated.wf, [EMMA, JOKE])
    assert calls == []
    # The local model is confident of a PLANNING command here; with no key that is an ambiguity to ask about before
    # the review (bd fix-o1lg.2), so the draft does not bind it. The decision still names it first.
    assert draft.items[0].means.kind == "from_scratch"
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    intent = resolution.part_intent(gated.wf, EMMA)
    assert intent.outcome == resolution.AMBIGUOUS and intent.ambiguous[0] == "Library/book_by_title"


# ----------------------------------------------------------------------
# The split sees the new request only (fix-bydz)
# ----------------------------------------------------------------------


conversations: list[str] = []


def _recording_split(monkeypatch, gated):
    """The split seam, recording the text it is given (its conversation input in ``conversations``) and returning
    it whole (one job). The real
    ``_refine_user_query`` is put back (the ``todo`` fixture makes it return the message as is), so the turn's
    request carries the conversation summary as it does in production."""
    vars(gated.ctx).pop("_refine_user_query", None)
    conversations.clear()
    seen: list[str] = []

    def split(user_query, **kwargs):
        seen.append(user_query)
        conversations.append(kwargs.get("conversation", ""))
        return [user_query]

    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", split)
    return seen


def test_the_split_gets_the_summary_as_its_own_input_not_in_the_request(gated, monkeypatch):
    seen = _recording_split(monkeypatch, gated)
    gated.ctx.append_conversation_turn("startup -> The library is open.")
    gated.script.steps = _finish()
    out = gated.ctx.process_turn(FIND)
    assert out.status is TurnStatus.COMPLETED
    assert seen == [FIND]
    assert conversations == ["conversation summary: startup -> The library is open."]
    # The summary is still there for the run of an unsplit request: the turn's refined text carries it.
    refined = gated.ctx._turn_refined_message
    assert "The library is open." in refined and refined.endswith(f"new_user_query: {FIND}")


def test_a_first_turn_gives_the_split_no_conversation(gated, monkeypatch):
    seen = _recording_split(monkeypatch, gated)
    gated.script.steps = _finish()
    gated.ctx.process_turn(FIND)
    assert seen == [FIND] and conversations == [""]


def test_a_drafted_goal_never_holds_the_conversation_summary(gated, monkeypatch):
    seen = _recording_split(monkeypatch, gated)
    gated.ctx.append_conversation_turn("startup -> The library is open.")
    gated.script.steps = _finish()
    out = gated.ctx.process_turn(JOKE, unattended=True)
    assert out.status is TurnStatus.COMPLETED
    assert seen == [JOKE] and conversations == ["conversation summary: startup -> The library is open."]
    assert [item.goal for item in gated.ctx._turn_subtask_plan.items] == [JOKE]


# ----------------------------------------------------------------------
# fix-jj3k: the fast path is bound in agent mode too
# ----------------------------------------------------------------------


def test_a_fast_path_command_is_bound_to_its_part_in_agent_mode(gated, monkeypatch):
    """A single execution command the local model is confident of is drafted as a one-item plan and run bound, as in
    no-agent mode (``_run_plain_means``: the listing is narrowed to the command), not left to the agent. The review is
    still skipped."""
    _split_into(monkeypatch, [FIND])
    gated.script.steps = _finish()
    out = gated.ctx.process_turn(FIND)
    assert out.status is TurnStatus.COMPLETED
    assert gated.store.records == []
    assert _marker(gated) == "fast_path"
    (item,) = gated.ctx._turn_subtask_plan.items
    assert (item.means.kind, item.means.name) == ("plain", "Library/find_member")
