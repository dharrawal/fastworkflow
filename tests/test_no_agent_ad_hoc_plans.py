"""No-agent mode: an ad hoc part says no matching plan was found and lists the planning commands available here,
the matcher's closest ones first (fix-1ma8.5).

Driven as tests/test_no_agent_review.py is: the real agent context on the annotated library copy with its trained heads;
the scripted seams are the planner's split, its text plan and Jev's planning-command match (its distribution).

The turns here are unattended: a turn somebody can answer asks which plan is meant before it ends a part as not run
(fix-yu3l.5, tests/test_plan_choice_question.py), and these tests are about the entry and the text of a part that IS not
run."""
from __future__ import annotations

import pytest

from fastworkflow.core import no_agent_mode
from fastworkflow.agent import run_listing
from fastworkflow.review import review_proposals
from fastworkflow.review.approval_binding import Surface
from fastworkflow.core.turn import TurnStatus
from fastworkflow.engine.workflow_execution_context import NOT_RUN_ARTIFACT, WorkflowExecutionContext
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the annotated library copy, and the guards
    annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, JOKE, _split_into, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import script_matcher

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

PLAN_TEXT = "1. Find a joke about cats.\n2. Tell it to Ann."
UNVERIFIED_HOST = Surface(kind="host", verified=False, uid=None)
HAIKU = "write me a haiku about the weather in Paris"


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


@pytest.fixture(autouse=True)
def _no_agent(initialized_fastworkflow, no_agent_model, no_planner_key, gated, monkeypatch):
    def never(self):
        raise AssertionError("the agent LM was built in no-agent mode")

    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context", never)

    def plan(user_query, session, with_agent_inputs_and_trajectory=False, planner_lm=None, planning_command=None,
             **kwargs):
        session._initial_plan_text = PLAN_TEXT if "joke" in user_query else ""
        session._phased_planning_command_call = None
        return user_query

    monkeypatch.setattr("fastworkflow.agent.workflow_agent.build_query_with_next_steps", plan)


def _jev(monkeypatch, distributions: dict[str, dict[str, float]] | None = None):
    """The remote decision model, scripted: the distribution of a part's text is its entry (a text not listed gets no
    answer). Every scripted score is below the match bar (0.5), so no command fits any part and each is ad hoc."""
    script_matcher(monkeypatch, distributions or {})


def _stops(out):
    return out.command_outputs[-1].command_response.artifacts[NOT_RUN_ARTIFACT]


def _context_plans(gated) -> list[str]:
    from fastworkflow.agent import workflow_agent

    return sorted(workflow_agent.planning_names_on_path(gated.wf))


def test_the_text_leads_with_the_no_match_statement_and_lists_the_closest_first_flagged(gated, monkeypatch):
    _jev(monkeypatch, {JOKE: {"Library/book_by_title": 0.45, "Library/overdue_summary": 0.25, "none": 0.2}})
    out = gated.ctx.process_turn(JOKE, approval_surface=UNVERIFIED_HOST, unattended=True)

    assert out.status is TurnStatus.COMPLETED
    (stop,) = _stops(out)
    assert out.answer == no_agent_mode.explanation(stop)
    assert out.answer.startswith("No matching plan was found for this part. Free planning is not available on this "
                                 "server, because no agent model is set (LLM_AGENT).")
    plans = stop["available_plans"]
    # The closest first, most probable first, flagged; the rest after, by name, not flagged.
    assert [p["name"] for p in plans[:2]] == ["Library/book_by_title", "Library/overdue_summary"]
    assert [p["closest"] for p in plans] == [True, True] + [False] * (len(plans) - 2)
    rest = [p["name"] for p in plans[2:]]
    assert rest == sorted(rest) and sorted(p["name"] for p in plans) == _context_plans(gated)
    assert all(p["description"] for p in plans)
    closest_at = out.answer.index("The closest planning commands available here:")
    others_at = out.answer.index("Other planning commands available here:")
    assert closest_at < out.answer.index("`Library/book_by_title`") < out.answer.index("`Library/overdue_summary`") < others_at
    for plan in plans:
        assert f"- `{plan['name']}`: {plan['description']}" in out.answer
    # The text lists them in the data's order, and keeps the pointer to the suggested plan, last.
    assert [out.answer.index(f"`{p['name']}`") for p in plans] == sorted(out.answer.index(f"`{p['name']}`") for p in plans)
    assert out.answer.endswith("A suggested plan is in the data (fw_not_run).")
    # Additive: the keys an ad hoc entry always had are unchanged.
    assert set(stop) == {"kind", "command", "phase", "open_step", "done", "facts", "remaining", "suggested_plan",
                         "means", "sub_task", "reason", "available_plans", "request"}
    assert (stop["command"], stop["open_step"], stop["means"], stop["suggested_plan"]) == (
        None, None, "from_scratch", PLAN_TEXT)


def test_a_match_below_the_floor_or_none_is_not_closest(gated, monkeypatch):
    floor = run_listing.UNSURE_CANDIDATE_FLOOR
    _jev(monkeypatch, {JOKE: {"Library/book_by_title": floor - 0.05, "none": 0.9}})
    out = gated.ctx.process_turn(JOKE, approval_surface=UNVERIFIED_HOST, unattended=True)

    (stop,) = _stops(out)
    assert not any(p["closest"] for p in stop["available_plans"])
    assert "The closest planning commands" not in out.answer
    assert "\n\nPlanning commands available here:\n- `Library/author_books`" in out.answer


def test_no_matcher_result_lists_the_context_plans_with_the_local_models_closest_first(gated, monkeypatch):
    """No remote answer: the local model's unsure ranking is the decision's, so it orders the closest plans (bd
    fix-o1lg: one ranked candidate list; before, only the remote model's distribution could flag a plan closest)."""
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    _jev(monkeypatch)
    out = gated.ctx.process_turn(JOKE, approval_surface=UNVERIFIED_HOST, unattended=True)

    (stop,) = _stops(out)
    assert sorted(p["name"] for p in stop["available_plans"]) == _context_plans(gated)
    scores = resolution.part_intent(gated.wf, JOKE).planning_scores()
    assert stop["available_plans"] == no_agent_mode.available_plans(gated.wf, scores)
    closest = [p["name"] for p in stop["available_plans"] if p["closest"]]
    assert all(scores[name] >= run_listing.UNSURE_CANDIDATE_FLOOR for name in closest)


def test_at_most_the_closest_three_are_flagged(gated, monkeypatch):
    names = _context_plans(gated)
    assert len(names) > review_proposals.MAX_PLAN_ALTERNATIVES + 1
    _jev(monkeypatch, {JOKE: {name: 0.45 - 0.01 * i for i, name in enumerate(names)}})
    out = gated.ctx.process_turn(JOKE, approval_surface=UNVERIFIED_HOST, unattended=True)

    (stop,) = _stops(out)
    flagged = [p["name"] for p in stop["available_plans"] if p["closest"]]
    assert flagged == names[:review_proposals.MAX_PLAN_ALTERNATIVES]


def test_a_split_turn_lists_the_plans_in_full_once_and_each_parts_closest_per_part(gated, monkeypatch):
    _jev(monkeypatch, {JOKE: {"Library/book_by_title": 0.45},
                       HAIKU: {"Library/note_member": 0.45, "Library/count_books": 0.3}})
    _split_into(monkeypatch, [JOKE, HAIKU])
    out = gated.ctx.process_turn(f"{JOKE} and {HAIKU}", approval_surface=UNVERIFIED_HOST, unattended=True)

    first, second = _stops(out)
    assert (first["sub_task"], second["sub_task"]) == (1, 2)
    assert [p["name"] for p in first["available_plans"] if p["closest"]] == ["Library/book_by_title"]
    assert [p["name"] for p in second["available_plans"] if p["closest"]] == ["Library/note_member", "Library/count_books"]
    # Each entry keeps the full list as data; the combined text prints the others once, under part 1.
    assert sorted(p["name"] for p in first["available_plans"]) == sorted(p["name"] for p in second["available_plans"])
    part_one, part_two = out.answer.split("### 2. ")
    part_two = part_two.split("\n", 1)[1]
    assert part_one.startswith("### 1. ") and "Other planning commands available here:" in part_one
    assert "Other planning commands available here:" not in part_two
    assert "The other planning commands available here are listed under part 1." in part_two
    assert part_two.index("`Library/note_member`") < part_two.index("`Library/count_books`")
    assert "`Library/author_books`" not in part_two
    assert part_two.startswith("No matching plan was found for this part.")
    assert part_two.rstrip().endswith("No plan could be suggested.")


def test_the_text_lists_at_most_twelve_and_says_how_many_more():
    plans = [{"name": f"Ctx/plan_{i:02d}", "description": f"does thing {i}", "closest": i < 2} for i in range(20)]
    entry = no_agent_mode.stop_entry(command=None, phase=None, open_step=None, done=[], facts={}, remaining=[],
                                     suggested_plan="", means="from_scratch", available_plans=plans)
    text = no_agent_mode.explanation(entry)

    assert text.count("\n- `") == 12
    assert "`Ctx/plan_11`" in text and "`Ctx/plan_12`" not in text
    assert "and 8 more; `what can i do` lists everything." in text
    assert len(entry["available_plans"]) == 20


def test_a_context_with_no_planning_commands_says_so_in_one_line():
    entry = no_agent_mode.stop_entry(command=None, phase=None, open_step=None, done=[], facts={}, remaining=[],
                                     suggested_plan=PLAN_TEXT, means="from_scratch", available_plans=[])
    assert no_agent_mode.explanation(entry) == (
        "No matching plan was found for this part. Free planning is not available on this server, because no agent "
        "model is set (LLM_AGENT).\n\nNo planning commands are available here.\n\n"
        "A suggested plan is in the data (fw_not_run).")


def test_a_stopped_planning_command_and_a_bound_command_that_could_not_start_keep_their_text_and_keys():
    stopped = no_agent_mode.stop_entry(
        command="Library/book_lookup", phase="book",
        open_step={"number": 1, "text": "x", "kind": "conditional", "reason": "a conditional step"},
        done=["find: 1. find_book"], facts={}, remaining=[], suggested_plan="1. x", means="planning")
    assert "available_plans" not in stopped
    assert no_agent_mode.explanation(stopped) == (
        "**What ran**\n- find: 1. find_book\n\n"
        "Library/book_lookup ran as far as code can run it without an agent model: 1 step ran, in phase 'book', then it "
        "stopped at step 1 (a conditional step). The remaining steps are listed in the data (fw_not_run); the wording of "
        "the final answer is left to the client.")
    bound = no_agent_mode.stop_entry(
        command="Library/book_lookup", phase=None, open_step=None, done=[], facts={}, remaining=[],
        suggested_plan="", means="planning", reason="the agent model that would carry out it is not set (LLM_AGENT)")
    assert "available_plans" not in bound
    assert no_agent_mode.explanation(bound) == (
        "Library/book_lookup was not run here: the agent model that would carry out it is not set (LLM_AGENT). "
        "No plan could be suggested.")
