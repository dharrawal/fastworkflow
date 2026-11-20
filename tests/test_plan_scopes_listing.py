"""The plans listed for an unmatched part, and the ones ``use_plan`` accepts, are the set the matcher chooses from
(fix-1ma8.5, fix-1ma8.6): the planning commands on the lookup path (``workflow_agent.planning_scopes``: the active
context, its bases, each live ancestor's context and its bases, then the root's), not only the active context's own.

Reproduced on the annotated library copy with a root-level planning command added to it (no context folder) and the
active context moved to a member object: that context has no planning commands of its own, so the narrow enumeration
(``RoutingDefinition.get_planning_command_names``) listed none while the matcher still offered the library's and the
root's. Real components; the scripted seams are the matcher's distribution and the extractor's model."""
from __future__ import annotations

import pytest

from fastworkflow import approval_binding, no_agent_mode, workflow_agent
from fastworkflow.approval_binding import Surface
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT, UsePlanError, WorkflowExecutionContext
from tests.scripted_extractor import script_extractor
from tests.test_approval_binding import annotated_library, todo  # noqa: F401
from tests.test_round1_gate import gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import script_matcher
import fastworkflow

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

ROOT_PLAN = "count_everything"
LIBRARY_PLAN = "Library/book_by_title"
UNVERIFIED = Surface(kind="host", verified=False, uid=None)
REQUEST = "how many books do we hold about Dune"

ROOT_PLAN_SOURCE = '''"""count_everything: how many books match the words the request gives, from the root."""
from pydantic import BaseModel, Field

from fastworkflow.planning import count, run

from .Library.find_book import Signature as FindBook


class Signature:
    """Count the books whose title or author holds some words (a root-level planning command)."""

    plain_utterances = ["how many books hold Dune"]

    class Input(BaseModel):
        query: str = Field(description="words of the title or the author's name",
                           json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.query)]

    def __result__(self, input, find):
        return f"{input.query} matches {count(find.total, 'book')}."
'''


@pytest.fixture(autouse=True)
def _root_plan_added(annotated_library):  # noqa: F811
    """A root-level planning command in the library copy, before the workflow is built (this runs before ``gated``)."""
    (annotated_library / "_commands" / f"{ROOT_PLAN}.py").write_text(ROOT_PLAN_SOURCE, encoding="utf-8")


@pytest.fixture(autouse=True)
def _no_agent(_root_plan_added, gated, no_agent_model, no_planner_key, monkeypatch):
    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context",
                        lambda self: pytest.fail("the agent LM was built in no-agent mode"))
    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps",
                        lambda user_query, session, **kwargs: setattr(session, "_initial_plan_text", "") or user_query)
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


def _in_a_member_context(gated, monkeypatch):
    """The active context is a member object: a context with no planning commands of its own."""
    member = next(iter(gated.wf.root.members.values()))
    monkeypatch.setattr(gated.wf, "_current_command_context", member)
    assert gated.wf._current_command_context_name != "Library"


def test_the_narrow_enumeration_finds_none_while_the_matcher_path_has_the_library_and_the_root(gated, monkeypatch):
    _in_a_member_context(gated, monkeypatch)
    context = gated.wf._current_command_context_name
    narrow = fastworkflow.RoutingRegistry.get_definition(gated.wf.folderpath).get_planning_command_names(context)
    scopes = workflow_agent.planning_scopes(gated.wf)

    assert ROOT_PLAN not in narrow and LIBRARY_PLAN not in narrow
    assert scopes[0] == context and "Library" in scopes and scopes[-1] == ""
    on_path = workflow_agent.planning_names_on_path(gated.wf)
    assert ROOT_PLAN in on_path and LIBRARY_PLAN in on_path
    # nearest scope first (the library's, then the root's), each by name
    library_names = [n for n in on_path if n.startswith("Library/")]
    assert on_path == library_names + [ROOT_PLAN] and library_names == sorted(library_names)


def test_the_listing_shows_the_plans_of_every_scope_on_the_path_with_the_closest_flagged_across_scopes(gated, monkeypatch):
    _in_a_member_context(gated, monkeypatch)
    script_matcher(monkeypatch, {REQUEST: {ROOT_PLAN: 0.45, LIBRARY_PLAN: 0.3, "none": 0.1}})     # no match: ad hoc
    # Unattended: the listing of a part that is not run is the subject (an attended turn asks which plan is meant
    # first, fix-yu3l.5).
    out = gated.ctx.process_turn(REQUEST, approval_surface=UNVERIFIED, unattended=True)

    (stop,) = out.command_outputs[-1].command_response.artifacts[NOT_RUN_ARTIFACT]
    plans = stop["available_plans"]
    assert [p["name"] for p in plans[:2]] == [ROOT_PLAN, LIBRARY_PLAN]               # closest first, across scopes
    assert [p["closest"] for p in plans[:2]] == [True, True] and not any(p["closest"] for p in plans[2:])
    assert [p["name"] for p in plans] == [ROOT_PLAN, LIBRARY_PLAN] + [
        n for n in workflow_agent.planning_names_on_path(gated.wf) if n not in (ROOT_PLAN, LIBRARY_PLAN)]
    assert "No planning commands are available here." not in out.answer
    assert f"- `{ROOT_PLAN}`:" in out.answer and "The closest planning commands available here:" in out.answer


def test_use_plan_accepts_a_root_scope_plan_from_a_member_context_binds_it_and_runs_it(gated, monkeypatch):
    _in_a_member_context(gated, monkeypatch)
    script_matcher(monkeypatch, lambda text: pytest.fail("a chosen plan must not be matched"))
    script_extractor(monkeypatch, *[{"reasoning": "the words", "query": "Dune"}] * 12)
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "on")
    asked = gated.ctx.process_turn(REQUEST, use_plan=ROOT_PLAN, approval_surface=UNVERIFIED)
    # The chosen plan asks for its parameter values (no review), from a scope that is not the active context's.
    assert asked.status is TurnStatus.AWAITING_USER
    assert asked.command_outputs[-1].command_response.artifacts["parameter_request"]["command"] == ROOT_PLAN
    out = gated.ctx.process_turn("yes", approval_surface=UNVERIFIED)

    assert out.status is TurnStatus.COMPLETED
    assert not out.command_outputs[-1].command_response.artifacts.get(NOT_RUN_ARTIFACT)
    assert any(o.command_name == ROOT_PLAN for o in out.command_outputs), [o.command_name for o in out.command_outputs]


def test_use_plan_accepts_an_ancestor_scope_plan_and_rejects_a_plan_that_is_not_on_the_path(gated, monkeypatch):
    _in_a_member_context(gated, monkeypatch)
    assert no_agent_mode.plan_names(gated.wf) == workflow_agent.planning_names_on_path(gated.wf)
    gated.ctx.check_use_plan(LIBRARY_PLAN)
    gated.ctx.check_use_plan(ROOT_PLAN)
    for name in ("Member/not_a_plan", "find_member", "review_subtasks"):
        with pytest.raises(UsePlanError, match="not a planning command available here"):
            gated.ctx.check_use_plan(name)


def test_a_plan_of_a_scope_off_the_path_is_refused(gated, monkeypatch):
    """A planning command of a context that is neither the active one, a base, an ancestor nor the root."""
    on_path = set(workflow_agent.planning_names_on_path(gated.wf))
    monkeypatch.setattr(workflow_agent, "planning_scopes", lambda workflow: ("Library",))      # the root left off
    assert ROOT_PLAN in on_path and ROOT_PLAN not in workflow_agent.planning_names_on_path(gated.wf)
    with pytest.raises(UsePlanError):
        gated.ctx.check_use_plan(ROOT_PLAN)
    gated.ctx.check_use_plan(LIBRARY_PLAN)


def test_an_empty_path_says_so(gated, monkeypatch):
    monkeypatch.setattr(workflow_agent, "planning_scopes", lambda workflow: ("Nowhere",))
    assert no_agent_mode.available_plans(gated.wf) == []
    entry = no_agent_mode.stop_entry(command=None, phase=None, open_step=None, done=[], facts={}, remaining=[],
                                     suggested_plan="", means="from_scratch", available_plans=[])
    assert "No planning commands are available here." in no_agent_mode.explanation(entry)
