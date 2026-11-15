"""The sub-task review in no-agent mode (fix-rn9p, piece P7).

On a host surface (REST, MCP) the review after split and merge is skipped today, since no one can answer it. With no
agent model the host's client is the agent that drives the work, so the review opens and the client sees it: each ad hoc
item (means ``from_scratch``) is marked as not run here and carries the planner's text plan; on acceptance the
runnable items run and the ad hoc item comes back as a stop. An unattended turn stays unreviewed, and with an agent
model nothing changes (tests/test_round1_gate.py).

Driven as tests/test_round1_gate.py is: the real agent context on the annotated library copy with its trained heads and
the real review; the planner's text plan is the one scripted seam.
"""
from __future__ import annotations

import pytest

from fastworkflow import approval_binding
from fastworkflow.approval_binding import Surface
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, FIND, JOKE, _marker, _open_review, _split_into, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

PLAN_TEXT = "1. Find a joke about cats.\n2. Tell it to Ann."
HOST = Surface(kind="host", verified=True, uid="user-1")


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


@pytest.fixture(autouse=True)
def _no_agent(initialized_fastworkflow, no_agent_model, no_planner_key, gated, monkeypatch):
    """No agent model: its LM is never built, and the planner's text plan is scripted."""
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    def never(self):
        raise AssertionError("the agent LM was built in no-agent mode")

    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context", never)

    def plan(user_query, session, with_agent_inputs_and_trajectory=False, planner_lm=None, planning_command=None,
             **kwargs):
        session._initial_plan_text = PLAN_TEXT if "joke" in user_query else ""
        session._phased_planning_command_call = None
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", plan)


@pytest.fixture(autouse=True)
def _jev_is_confident_about_emma(monkeypatch):
    """Jev's planning-command match, scripted (no key, no model): EMMA is matched to book_by_title confidently, and
    nothing else is matched. The heads alone are only unsure about EMMA (a top-k pick), which no-agent mode does not
    run (``review_proposals.confident_means``)."""
    from fastworkflow import review_proposals
    from fastworkflow.workflow_agent import load_planning_command_map

    def match(workflow, text, exclude=None):
        if text == EMMA:
            command = load_planning_command_map(workflow.folderpath)["Library/book_by_title"]
            return "", command, {"Library/book_by_title": 0.9}
        return "", None, {}

    monkeypatch.setattr(review_proposals, "planning_match", match)


def _reply(gated, message, token):
    """A reply from the host, with the approval token the host was given once when the review opened."""
    return gated.ctx.process_turn(message, approval_surface=HOST, approval_token=token)


def _token(gated):
    delivery = approval_binding.take_delivery(gated.wf)
    assert delivery is not None
    return delivery["token"]


def _artifacts(out):
    return out.command_outputs[-1].command_response.artifacts


def test_a_host_surface_turn_opens_the_review_with_a_plan_preview_on_the_ad_hoc_item(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", approval_surface=HOST)

    # The review opened on a host surface, where it is skipped with an agent model.
    assert out.status is TurnStatus.AWAITING_USER
    record = _open_review(gated)
    assert record.state == "awaiting"
    assert _marker(gated) == "open"
    artifacts = _artifacts(out)
    assert artifacts["question_kind"] == "review"
    items = artifacts["review"]["items"]
    assert [item["goal"] for item in items] == [EMMA, JOKE]
    assert items[0]["means"]["kind"] == "planning" and items[0]["not_run_here"] is False
    assert "plan_preview" not in items[0]
    assert items[1]["not_run_here"] is (items[1]["means"]["kind"] == "from_scratch")

    # An edit makes the second item ad hoc: it is marked, and carries the planner's text plan.
    revised = _reply(gated, "change 2: means=from_scratch", _token(gated))
    assert revised.status is TurnStatus.AWAITING_USER
    items = _artifacts(revised)["review"]["items"]
    assert items[0]["not_run_here"] is False and "plan_preview" not in items[0]
    assert items[1]["means"]["kind"] == "from_scratch" and items[1]["not_run_here"] is True
    assert items[1]["plan_preview"] == PLAN_TEXT


def test_accepting_runs_the_runnable_item_and_returns_the_ad_hoc_item_as_a_stop(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}", approval_surface=HOST)
    token = _token(gated)
    _reply(gated, "change 1: inputs=title=Emma", token)
    _reply(gated, "change 2: means=from_scratch", token)
    gated.script.steps = []

    out = _reply(gated, "accept", token)

    assert out.status is TurnStatus.COMPLETED
    assert _marker(gated) == "reviewed"
    # The runnable item ran by code (book_by_title, a planning command): its output is on the turn.
    assert any(o.command_name == "Library/book_by_title" for o in out.command_outputs), (
        [o.command_name for o in out.command_outputs], out.answer, _artifacts(out).get(NOT_RUN_ARTIFACT))
    stops = _artifacts(out)[NOT_RUN_ARTIFACT]
    assert len(stops) == 1, [(s["sub_task"], s["command"], s["means"]) for s in stops]
    (stop,) = stops
    assert stop["sub_task"] == 2 and stop["command"] is None and stop["means"] == "from_scratch"
    assert stop["suggested_plan"] == PLAN_TEXT
    assert gated.script.steps == []


def test_an_unattended_turn_stays_unreviewed_in_no_agent_mode(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", unattended=True, approval_surface=HOST)

    assert out.status is TurnStatus.COMPLETED
    assert [r for r in gated.store.records if r.command == "review_subtasks"] == []
    assert _marker(gated) == "not_reviewed"
    assert any(stop["sub_task"] == 2 for stop in _artifacts(out)[NOT_RUN_ARTIFACT])


UNVERIFIED_HOST = Surface(kind="host", verified=False, uid=None)
MARS = "plan a trip to Mars"
HAIKU = "write me a haiku about the weather in Paris"
HOURS = "library_hours"   # the caller names the command (an exact name), which needs no input


def _reviews(gated):
    return [r for r in gated.store.records if r.command == "review_subtasks"]


def test_an_unverified_host_identity_opens_no_review_and_the_runnable_part_runs(gated, monkeypatch):
    # D1: the host approval path takes a reply only from a verified end user, so a review opened for an unverified one
    # could never be accepted. No review is opened; the runnable part runs and the ad hoc part comes back as a stop.
    _split_into(monkeypatch, [HOURS, JOKE])
    out = gated.ctx.process_turn(f"{HOURS} and {JOKE}", approval_surface=UNVERIFIED_HOST)

    assert out.status is TurnStatus.COMPLETED
    assert _reviews(gated) == [] and _marker(gated) == "not_reviewed"
    assert any(o.command_name == "library_hours" for o in out.command_outputs)
    (stop,) = _artifacts(out)[NOT_RUN_ARTIFACT]
    assert stop["sub_task"] == 2 and stop["command"] is None and stop["suggested_plan"] == PLAN_TEXT


def test_a_verified_host_identity_opens_the_review_and_yes_with_the_token_accepts_it(gated, monkeypatch):
    _split_into(monkeypatch, [HOURS, JOKE])
    out = gated.ctx.process_turn(f"{HOURS} and {JOKE}", approval_surface=HOST)
    assert out.status is TurnStatus.AWAITING_USER and _marker(gated) == "open"

    accepted = _reply(gated, "yes", _token(gated))

    assert accepted.status is TurnStatus.COMPLETED and _marker(gated) == "reviewed"
    assert any(o.command_name == "library_hours" for o in accepted.command_outputs)
    assert [s["sub_task"] for s in _artifacts(accepted)[NOT_RUN_ARTIFACT]] == [2]


@pytest.mark.parametrize("surface", [HOST, UNVERIFIED_HOST])
def test_a_turn_where_nothing_would_run_never_opens_a_review(gated, monkeypatch, surface):
    _split_into(monkeypatch, [JOKE, MARS])
    out = gated.ctx.process_turn(f"{JOKE} and {MARS}", approval_surface=surface)

    assert out.status is TurnStatus.COMPLETED
    assert _reviews(gated) == [] and _marker(gated) == "not_reviewed"
    assert [s["sub_task"] for s in _artifacts(out)[NOT_RUN_ARTIFACT]] == [1, 2]
    assert not any(o.command_name.startswith("Library/") for o in out.command_outputs)


# ---------------------------------------------------------------- D2: a part runs bound only on a confident match


def _draft(gated, sub_task, *, confident_only=True):
    from fastworkflow import review_proposals

    return review_proposals.draft_round1_proposal(gated.wf, [sub_task], confident_only=confident_only).items[0]


def test_an_unconfident_match_is_not_run_here_with_the_planners_suggestion(gated, monkeypatch):
    from fastworkflow import review_proposals

    # The end-to-end run bound this poem request to a find-by-title command: the heads only rank it in their top-k and Jev
    # picks no planning command, so the unfiltered draft binds it and no-agent mode does not run it.
    monkeypatch.setattr(review_proposals, "planning_match", lambda workflow, text, exclude=None: ("", None, {}))
    assert _draft(gated, HAIKU, confident_only=False).means.kind != "from_scratch"
    assert _draft(gated, HAIKU).means.kind == "from_scratch"

    _split_into(monkeypatch, [HAIKU, JOKE])
    out = gated.ctx.process_turn(f"{HAIKU} and {JOKE}", approval_surface=UNVERIFIED_HOST)

    assert out.status is TurnStatus.COMPLETED
    assert not any(o.command_name.startswith(("Record/", "Library/")) for o in out.command_outputs)
    stops = _artifacts(out)[NOT_RUN_ARTIFACT]
    assert [(s["sub_task"], s["command"], s["means"]) for s in stops] == [(1, None, "from_scratch"),
                                                                          (2, None, "from_scratch")]
    assert stops[1]["suggested_plan"] == PLAN_TEXT


def test_a_confident_match_runs_bound(gated, monkeypatch):
    from fastworkflow import review_proposals

    # Jev's confident pick (scripted by the fixture above) and a head's confident label both bind.
    assert _draft(gated, EMMA).means.name == "Library/book_by_title"
    head = _draft(gated, FIND)
    assert (head.means.kind, head.means.name) == ("plain", "Library/find_member")
    assert review_proposals.confident_means(gated.wf, FIND, head.means)


def test_a_write_command_never_runs_from_an_unconfident_match(gated):
    from fastworkflow import review_proposals
    from fastworkflow.command_writes import write_commands

    assert "Record/add_note" in write_commands(gated.wf.folderpath)
    reminder = "draft a reminder email to each member with an overdue loan"
    # The unfiltered draft binds a note-writing planning command to it (the end-to-end defect); no-agent mode does not.
    unfiltered = _draft(gated, reminder, confident_only=False).means
    assert unfiltered.name in write_commands(gated.wf.folderpath)
    means = review_proposals.Means(kind=review_proposals.MEANS_PLAIN, name="Record/add_note")
    assert not review_proposals.confident_means(gated.wf, reminder, means)
    assert _draft(gated, reminder).means.kind == "from_scratch"


def test_a_part_that_ran_bound_names_its_command_in_the_answer(gated, monkeypatch):
    _split_into(monkeypatch, [HOURS, JOKE])
    out = gated.ctx.process_turn(f"{HOURS} and {JOKE}", approval_surface=UNVERIFIED_HOST)

    assert "(Ran library_hours.)" in out.answer




# ---------------------------------------------------------------- D3: the deterministic path says why it did not plan

UNROUTABLE = "calculate 17 times 23 and then convert the result to Roman numerals"


def _deterministic_fallback(gated):
    gated.ctx._run_as_agent = False     # the CLI's fallback: no agent model, so the session answers deterministically
    gated.wf.context.pop("run_as_agent", None)


def test_an_unroutable_request_that_offers_candidates_is_answered_as_in_deterministic_mode(gated):
    # D3 (narrowed): the reply lists candidate commands, so the user just picks one: no "not planned" note.
    from fastworkflow import no_agent_mode

    _deterministic_fallback(gated)
    out = gated.ctx.process_turn(UNROUTABLE)

    assert _artifacts(out).get("intent_choices"), out.answer
    assert out.answer.startswith("Ambiguous intent error"), out.answer
    assert no_agent_mode.NOT_PLANNED_NOTE not in out.answer and "no agent model" not in out.answer, out.answer


def test_the_note_is_added_when_a_reply_offers_no_candidate_command(gated):
    from fastworkflow import no_agent_mode
    from fastworkflow.command_executor import _no_agent_note

    class Route:
        handled = routed = False
        command_name = parameter_request = intent_choices = None

    _deterministic_fallback(gated)
    cme = gated.ctx._cme_workflow
    assert _no_agent_note(Route(), cme) == f"\n{no_agent_mode.NOT_PLANNED_NOTE}"
    with_candidates = Route()
    with_candidates.intent_choices = ["Library/find_book", "Library/open_book", "Record/find_record"]
    assert _no_agent_note(with_candidates, cme) == ""
    empty = Route()
    empty.intent_choices = []
    assert _no_agent_note(empty, cme) == f"\n{no_agent_mode.NOT_PLANNED_NOTE}"


def test_the_same_request_is_answered_as_before_when_an_agent_model_is_set(gated):
    gated.ctx._run_as_agent = False
    gated.wf.context.pop("run_as_agent", None)
    gated.ctx._cme_workflow.context.pop("no_agent_mode")   # bound only when no agent model is specified
    out = gated.ctx.process_turn(UNROUTABLE)

    assert "agent model" not in out.answer, out.answer
    assert out.answer



# ---------------------------------------------------------------- D2: Jev's confident "none" against a head's planning pick

SUMMARY = "run the library summary"      # heads: one confident label, the planning command Library/overdue_summary


def _jev(monkeypatch, name=None, distribution=None):
    from fastworkflow import review_proposals
    from fastworkflow.workflow_agent import load_planning_command_map

    def match(workflow, text, exclude=None):
        command = load_planning_command_map(workflow.folderpath)[name] if name else None
        return "", command, dict(distribution or {})

    monkeypatch.setattr(review_proposals, "planning_match", match)


def test_a_confident_jev_none_stops_a_head_confident_planning_command_from_running(gated, monkeypatch):
    _jev(monkeypatch, None, {"none": 0.95})
    out = gated.ctx.process_turn(SUMMARY, approval_surface=UNVERIFIED_HOST)

    assert out.status is TurnStatus.COMPLETED and _marker(gated) != "fast_path"
    assert not any(o.command_name == "Library/overdue_summary" for o in out.command_outputs)
    (stop,) = _artifacts(out)[NOT_RUN_ARTIFACT]
    assert stop["command"] is None and stop["means"] == "from_scratch"


def test_a_jev_pick_of_the_same_planning_command_runs_it(gated, monkeypatch):
    from fastworkflow import review_proposals

    _jev(monkeypatch, "Library/overdue_summary", {"Library/overdue_summary": 0.9, "none": 0.05})
    assert review_proposals.confident_pick(gated.wf, SUMMARY).name == "Library/overdue_summary"
    gated.ctx.process_turn(SUMMARY, approval_surface=UNVERIFIED_HOST)
    assert _marker(gated) == "fast_path"


@pytest.mark.parametrize("distribution", [{}, {"none": 0.3}], ids=["jev_silent", "none_not_confident"])
def test_without_a_confident_jev_none_the_head_confident_planning_command_is_as_before(gated, monkeypatch, distribution):
    from fastworkflow import review_proposals

    # No verdict (no key, an error, a timeout) or a "none" below the match threshold: the head's pick stands, as before.
    _jev(monkeypatch, None, distribution)
    assert review_proposals.confident_pick(gated.wf, SUMMARY).name == "Library/overdue_summary"
    gated.ctx.process_turn(SUMMARY, approval_surface=UNVERIFIED_HOST)
    assert _marker(gated) == "fast_path"


def test_a_confident_jev_none_does_not_touch_a_head_confident_plain_command(gated, monkeypatch):
    from fastworkflow import review_proposals

    _jev(monkeypatch, None, {"none": 0.95})
    means = review_proposals.confident_pick(gated.wf, FIND)
    assert (means.kind, means.name) == ("plain", "Library/find_member")



# ---------------------------------------------------------------- the rendered review text marks an item that will not run

SUFFIX = " - not run here (no agent model); a suggested plan is returned"


def _lines(text):
    return [line for line in text.splitlines() if line[:2] in ("1.", "2.", "3.")]


def test_the_review_text_marks_an_ad_hoc_item_on_first_render_and_after_an_edit(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}", approval_surface=HOST)
    record = _open_review(gated)
    first = _lines(record.rendering)
    assert not first[0].endswith(SUFFIX) and first[1].endswith(SUFFIX), first
    assert out.answer.count(SUFFIX) == 1
    token = _token(gated)

    # An edit that leaves item 2 ad hoc and one that makes item 1 ad hoc: the marker follows the means, in the
    # re-render, and the hash is that of the text shown.
    edited = _reply(gated, "change 2: outcome=a short joke", token)
    assert edited.status is TurnStatus.AWAITING_USER, edited.answer
    assert _lines(record.rendering)[1].endswith(SUFFIX)
    from fastworkflow.review_engine import rendering_hash
    assert record.rendering_hash == rendering_hash(record.rendering)
    revised = _reply(gated, "change 1: means=from_scratch", token)
    assert revised.status is TurnStatus.AWAITING_USER
    assert all(line.endswith(SUFFIX) for line in _lines(record.rendering)), record.rendering

    # The reply reader is unaffected: an edit that drops the marked item still parses, and accept still runs.
    dropped = _reply(gated, "drop 2", token)
    assert dropped.status is TurnStatus.AWAITING_USER, dropped.answer
    assert SUFFIX not in "\n".join(_lines(record.rendering)[1:])
    assert _reply(gated, "accept", token).status is TurnStatus.COMPLETED


def test_with_an_agent_model_the_review_text_is_what_it_always_was(gated):
    from fastworkflow import review_engine, review_proposals

    proposal = review_proposals.build_round1_proposal([EMMA, JOKE], {})
    assert any(item.means.kind == "from_scratch" for item in proposal.items)
    plain = review_engine.prompt_for(proposal, gated.wf)
    assert SUFFIX not in plain and plain == review_engine.prompt_for(proposal, gated.wf, no_agent=False)
    marked = review_engine.prompt_for(proposal, gated.wf, no_agent=True)
    assert marked.replace(SUFFIX, "") == plain and SUFFIX in marked
