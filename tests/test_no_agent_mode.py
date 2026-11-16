"""No-agent mode (fix-rn9p): the framework run without the ReAct agent model (``LLM_AGENT`` not specified).

Real components throughout, on the library workflow copies the phase tests use: the real phase runner, executor and
execution context. The only seams are the ones those suites already use (the planner's split and plan, the extraction
hop), and the agent LM itself, which is made to fail here: a test in this file that reaches a model fails.
With the agent model set nothing changes, and the rest of the suite is the proof of that.
"""
from __future__ import annotations

import pytest

import fastworkflow
from fastworkflow import command_preview, no_agent_mode
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.planning import run, when
from fastworkflow.planning_commands import PlanningCommandCallPlan, call_phases
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import (
    NOT_RUN_ARTIFACT, TURN_PARTS_ARTIFACT, WorkflowExecutionContext)
from tests.test_assistant_mode_planning import (  # noqa: F401 - the module fixture and the assistant-mode helpers
    LIBRARY, _assistant_ctx, initialized_fastworkflow as _assistant_init)
from tests.test_nested_planning_commands import (  # noqa: F401 - fixtures
    BOOK_BY_TITLE, BOOK_LOANS, _BOOK_BY_TITLE_SOURCE, _nested_state)
from tests.test_planning_command_code_steps import Model
from tests.test_step_helpers import (  # noqa: F401 - fixtures and helpers
    FoundBooks, OpenedBook, _build_planning_command, library_env, script_every_agent)
from tests.test_turn_result_capture import initialized_fastworkflow, no_review_gate  # noqa: F401 - shared fixtures

PLAN_TEXT = "1. Look up the trip.\n2. Book it."

# A planning command that needs the agent: its second step is conditional, a decision code does not make.
OPEN_IF_MATCHED = '''"""open_member_if_matched: find the members the request names, and open the one that matched."""
from pydantic import BaseModel, Field

from fastworkflow.planning import run, when

from .find_member import Signature as FindMember


class Signature:
    """Find the members whose name holds some words, and open the one that matched."""

    plain_utterances = ["open the member called Lee if there is one"]

    class Input(BaseModel):
        query: str = Field(description="words of the member's name",
                           json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input) -> FindMember.Output:
        return [run("find_member", query=input.query),
                when("a member matched", run("open_member", named=input.query))]

    def __result__(self, input, find):
        return f"{input.query} matches {find.total} members."
'''


@pytest.fixture(autouse=True)
def _no_agent(initialized_fastworkflow, no_agent_model, no_planner_key, monkeypatch):
    """Every case runs with no agent model; Jev is not needed here (the matches are scripted). The agent's LM must
    never be built: a case that reaches it fails."""
    def never(self):
        raise AssertionError("the agent LM was built in no-agent mode")

    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context", never)


def _stops(result):
    """The turn's ``fw_not_run`` list, from the command output it travels on (the last one)."""
    assert result.command_outputs, "the turn has no command output to carry its stops"
    return result.command_outputs[-1].command_response.artifacts.get(NOT_RUN_ARTIFACT)


def _plan(monkeypatch, env, matches: dict | None = None, ad_hoc_plan: str = ""):
    """Script the planner: a sub-task whose text holds a key of *matches* matches that planning command (arguments
    ``env.args``), any other is ad hoc and gets the text plan *ad_hoc_plan* (the planner's, no agent)."""
    matches = matches if matches is not None else {"book": env.planning_command}

    def command_for(text, folder_path, **_):
        return "", next((command for key, command in matches.items() if key in text), None)

    def plan(user_query, session, with_agent_inputs_and_trajectory=False, planner_lm=None, planning_command=None,
             **kwargs):
        session._initial_plan_text = ""
        if planning_command is None:
            session._initial_plan_text = ad_hoc_plan
            return user_query
        call = PlanningCommandCallPlan(plan="", args=dict(env.args), notes=[], prediction=None,
                                       phases=call_phases(planning_command, dict(env.args)),
                                       steps_after=[], steps_before=[])
        session._phased_planning_command_call = (planning_command, call)
        return user_query

    import fastworkflow.workflow_agent as workflow_agent
    monkeypatch.setattr(workflow_agent, "planner_planning_command_for", command_for)
    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps", plan)


class CodeOnly:
    """A planning command whose steps code runs and whose ``__result__`` writes the answer."""

    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title)]

    def __result__(self, input, find):
        return f"{input.title}: {len(find.book_uids)} found"


class OpenStep:
    """A planning command whose second phase holds a conditional step: a decision, which code does not make."""

    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title)]

    def book(self, input, find) -> OpenedBook:
        return [when("a book matched", run("open_book", named=input.title))]

    def __result__(self, input, find, book):
        return f"{input.title} opened"


class NoResult:
    """Code runs its step, but no ``__result__`` writes the answer, so a model would."""

    def find(self, input) -> FoundBooks:
        return [run("find_book", query=input.title)]


def _sent(env) -> list[str]:
    return list(env.runner.sent)


# ---------------------------------------------------------------- a code-only planning command runs

def test_a_code_only_planning_command_with_a_result_runs_to_completion_and_builds_no_agent_lm(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(CodeOnly)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == no_agent_mode.ran_note("Dune: 2 found", "Library/book_lookup")
    assert library_env.model.asked == []
    assert not any(NOT_RUN_ARTIFACT in o.command_response.artifacts for o in result.command_outputs)
    assert library_env.ctx.no_agent_mode is True


# ---------------------------------------------------------------- a planning command that stops

def test_a_planning_command_with_an_open_step_returns_a_completed_turn_with_one_stop(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(OpenStep)
    _plan(monkeypatch, library_env)
    # Unattended: nobody can be asked, so the conditional step ends the command with the stop. (Attended, the step is a
    # decision the client answers: tests/test_no_agent_decisions.py.)
    result = library_env.ctx.process_turn("look up the book Dune", unattended=True)

    assert result.status == TurnStatus.COMPLETED
    stops = _stops(result)
    assert len(stops) == 1
    stop = stops[0]
    assert stop["kind"] == "not_run_here"
    assert stop["command"] == "Library/book_lookup"
    assert stop["phase"] == "book"
    assert stop["means"] == "planning"
    assert stop["sub_task"] is None
    step = stop["open_step"]
    assert step["number"] == 1 and step["kind"] == "conditional"
    assert "only when a book matched" in step["text"] and "open_book" in step["text"]
    assert step["reason"].startswith("a conditional step")
    # What ran: the find phase's step, by code; its facts are the typed facts the phase filled.
    assert len(stop["done"]) == 1 and stop["done"][0].startswith("find: 1.") and "find_book" in stop["done"][0]
    assert list(stop["facts"]) == ["find"]
    assert len(stop["facts"]["find"]["book_uids"]) == 2
    assert stop["remaining"] == []
    assert stop["suggested_plan"].startswith("1. only when a book matched")
    # The command has no later phase: the continuation ends with the answer to write.
    assert stop["suggested_plan"].splitlines()[-1] == f"2. {no_agent_mode.ANSWER_LINE}" and "later_phases" not in stop
    # The answer is short, plain and written in code; its wording is the client's.
    assert result.answer == no_agent_mode.explanation(stop)
    assert "Library/book_lookup ran as far as code can run it" in result.answer
    assert "fw_not_run" in result.answer
    # Nothing ran beyond the step that ran, and no model was asked.
    assert all("open_book" not in command for command in _sent(library_env))
    assert library_env.model.asked == []


def test_a_planning_command_whose_answer_needs_a_model_ends_with_the_stop_and_its_facts(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(NoResult)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["command"] == "Library/book_lookup"
    assert stop["open_step"]["kind"] == "answer" and stop["open_step"]["number"] is None
    assert "needs a model" in stop["open_step"]["reason"]
    assert len(stop["done"]) == 1 and "find_book" in stop["done"][0]
    assert len(stop["facts"]["find"]["book_uids"]) == 2
    assert stop["remaining"] == []
    assert stop["suggested_plan"] == "1. Write the answer to the request from the facts and the steps that ran."
    assert "its answer needs a model to write it" in result.answer
    assert library_env.model.asked == []


# ---------------------------------------------------------------- an ad hoc part

def test_an_ad_hoc_request_is_not_run_and_returns_the_planners_text_plan(library_env, monkeypatch):
    library_env.planning_command = None
    _plan(monkeypatch, library_env, matches={}, ad_hoc_plan=PLAN_TEXT)
    result = library_env.ctx.process_turn("plan a trip to Mars")

    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["kind"] == "not_run_here"
    assert (stop["command"], stop["phase"], stop["open_step"], stop["means"]) == (None, None, None, "from_scratch")
    assert stop["done"] == [] and stop["remaining"] == [] and stop["facts"] == {}
    assert stop["suggested_plan"] == PLAN_TEXT
    assert result.answer == no_agent_mode.explanation(stop)
    assert result.answer.startswith("No matching plan was found for this part. Free planning is not available")
    assert "A suggested plan is in the data (fw_not_run)." in result.answer
    assert _sent(library_env) == []
    assert library_env.model.asked == []


def test_an_ad_hoc_request_with_no_planner_text_has_no_suggestion(library_env, monkeypatch):
    library_env.planning_command = None
    _plan(monkeypatch, library_env, matches={}, ad_hoc_plan="")
    result = library_env.ctx.process_turn("plan a trip to Mars")

    (stop,) = _stops(result)
    assert stop["suggested_plan"] == ""
    assert "No plan could be suggested." in result.answer
    assert result.answer.startswith("No matching plan was found for this part.")
    # The turn ran no command, so the stop travels on an output of its own, marked as plumbing for the page.
    (output,) = result.command_outputs
    assert output.framework is True


def test_a_stop_on_a_real_last_output_leaves_that_output_unmarked(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(OpenStep)
    _plan(monkeypatch, library_env)
    # Unattended: nobody can be asked, so the conditional step ends the command with the stop. (Attended, the step is a
    # decision the client answers: tests/test_no_agent_decisions.py.)
    result = library_env.ctx.process_turn("look up the book Dune", unattended=True)

    assert NOT_RUN_ARTIFACT in result.command_outputs[-1].command_response.artifacts
    assert result.command_outputs[-1].framework is not True


# ---------------------------------------------------------------- a multi-part turn

def test_a_multi_part_turn_runs_the_runnable_part_and_marks_the_ad_hoc_part(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(CodeOnly)
    _plan(monkeypatch, library_env, matches={"book": library_env.planning_command}, ad_hoc_plan=PLAN_TEXT)
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda *args, **kwargs: ["look up the book Dune", "plan a trip to Mars"])
    result = library_env.ctx.process_turn("look up the book Dune and plan a trip to Mars")

    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["sub_task"] == 2 and stop["command"] is None and stop["means"] == "from_scratch"
    assert stop["suggested_plan"] == PLAN_TEXT
    assert result.answer == (
        "### 1. look up the book Dune\nDune: 2 found\n\n(Ran Library/book_lookup.)\n\n"
        f"### 2. plan a trip to Mars\n{no_agent_mode.explanation(stop)}")
    parts = result.command_outputs[-1].command_response.artifacts[TURN_PARTS_ARTIFACT]
    assert [(p["number"], p["answered"]) for p in parts] == [(1, True), (2, True)]
    # ``answered`` means "has text under its heading"; ``not_run`` marks the part whose text is a stop's explanation.
    assert [p.get("not_run", False) for p in parts] == [False, True]
    assert "not_run" not in parts[0]
    assert result.answer[parts[1]["start"]:parts[1]["end"]] == no_agent_mode.explanation(stop)
    assert library_env.model.asked == []


# ---------------------------------------------------------------- a nested stop bubbles up

_CONDITIONAL_BOOK = _BOOK_BY_TITLE_SOURCE.replace(
    "from fastworkflow.planning import run\n", "from fastworkflow.planning import run, when\n").replace(
    'return [run("open_book", named=input.title)]',
    'return [when("a book matched", run("open_book", named=input.title))]')


def test_a_nested_stop_bubbles_up_as_the_parents_stop(tmp_path, initialized_fastworkflow, monkeypatch):
    from tests.test_nested_planning_commands import BOOK_BY_TITLE_FILE
    from tests.test_step_helpers import _real_planning_command

    assert "when(" in _CONDITIONAL_BOOK
    state = _nested_state(tmp_path, initialized_fastworkflow, monkeypatch,
                          command_texts={BOOK_BY_TITLE_FILE: _CONDITIONAL_BOOK})
    try:
        state.planning_command = _real_planning_command(BOOK_LOANS, "_commands/Library/book_loans.py").command
        state.args = {"title": "Dune"}
        # Unattended: nobody can be asked, so the nested conditional step stops the parent (attended, it is a question).
        result = state.ctx.process_turn("who has Dune on loan", unattended=True)
    finally:
        state.ctx.pop_active_workflow()

    assert result.status == TurnStatus.COMPLETED
    (stop,) = _stops(result)
    assert stop["command"] == BOOK_LOANS and stop["phase"] == "book"
    assert stop["open_step"]["kind"] == "nested" and stop["open_step"]["number"] == 1
    assert BOOK_BY_TITLE in stop["open_step"]["reason"]
    inner = stop["nested"]
    assert inner["command"] == BOOK_BY_TITLE and inner["phase"] == "book"
    assert inner["open_step"]["kind"] == "conditional"
    assert list(inner["facts"]) == ["find"]
    # The parent's suggestion carries the child's remaining steps first, then its own later phases are named.
    assert stop["suggested_plan"].startswith("1. only when a book matched")
    assert stop["later_phases"] == ["loans"]
    # ... and the continuation names the parent's later phase's steps, then the answer to write (once).
    lines = stop["suggested_plan"].splitlines()
    assert any("(phase loans)" in line for line in lines)
    assert lines[-1].endswith(no_agent_mode.ANSWER_LINE) and stop["suggested_plan"].count(no_agent_mode.ANSWER_LINE) == 1
    assert state.model.asked == [] and state.nested.asked == []
    assert len(state.ctx._frame_stack()) == 1


# ---------------------------------------------------------------- a plain single command answers from its response

def test_a_plain_command_in_an_agent_turn_answers_from_its_own_response(library_env, monkeypatch):
    from fastworkflow import review_proposals
    from fastworkflow.review_proposals import Means, SubTask, SubTaskInput, SubTaskProposal

    item = SubTask(goal="find the book Dune", means=Means(kind="plain", name="Library/find_book"),
                   inputs=[SubTaskInput(name="query", value="Dune")])
    monkeypatch.setattr(WorkflowExecutionContext, "_subtask_gate",
                        lambda self, subtasks: setattr(self, "_turn_subtask_plan", SubTaskProposal(items=[item])))
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", lambda *a, **k: ["find the book Dune"])
    library_env.planning_command = None
    result = library_env.ctx.process_turn("find the book Dune")

    assert result.status == TurnStatus.COMPLETED
    ran = [o for o in result.command_outputs if o.command_name == "Library/find_book"]
    assert len(ran) == 1 and ran[0].success
    assert result.answer == no_agent_mode.ran_note(ran[0].command_response.response, "Library/find_book")
    assert _stops(result) is None
    assert library_env.model.asked == []
    assert review_proposals.drafting_on()


# ---------------------------------------------------------------- the deterministic path

def test_the_deterministic_path_runs_a_needs_agent_planning_command_to_its_first_open_step(tmp_path, monkeypatch):
    ctx, _folder = _assistant_ctx(tmp_path, monkeypatch, "Library/open_member_if_matched", OPEN_IF_MATCHED)
    script_every_agent(monkeypatch, Model(None))

    out = CommandExecutor.invoke_command(ctx, "open_member_if_matched query=Lee")

    # Not refused up front: it ran its code step and stopped at the prose step, as a completed result.
    assert out.success is True, out.command_response.response
    assert NOT_RUN_ARTIFACT in out.command_response.artifacts, (out.command_response.response, out.command_response.artifacts)
    (stop,) = out.command_response.artifacts[NOT_RUN_ARTIFACT]
    assert stop["command"] == "Library/open_member_if_matched" and stop["phase"] == "find"
    assert stop["open_step"]["number"] == 2 and stop["open_step"]["kind"] == "conditional"
    assert "only when a member matched" in stop["open_step"]["text"]
    assert len(stop["done"]) == 1 and "find_member" in stop["done"][0]
    assert out.command_response.response == no_agent_mode.explanation(stop)


def test_with_an_agent_model_set_the_deterministic_refusal_stays(tmp_path, monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "LLM_AGENT", "test/model")
    ctx, _folder = _assistant_ctx(tmp_path, monkeypatch, "Library/open_member_if_matched", OPEN_IF_MATCHED)

    out = CommandExecutor.invoke_command(ctx, "open_member_if_matched query=Lee")

    assert ctx.no_agent_mode is False
    assert out.success is False
    assert "it runs in agent mode" in out.command_response.response


# ---------------------------------------------------------------- the mode and its startup

def test_the_mode_is_decided_in_one_place_from_llm_agent_alone(monkeypatch):
    assert no_agent_mode.agent_model_specified() is False
    monkeypatch.setitem(fastworkflow._env_vars, "LLM_AGENT", "   ")
    assert no_agent_mode.agent_model_specified() is False
    monkeypatch.setitem(fastworkflow._env_vars, "LLM_AGENT", "test/model")
    assert no_agent_mode.agent_model_specified() is True


def test_no_agent_model_and_no_jev_key_is_a_startup_problem_naming_both_settings_and_the_two_ways_out():
    problem = no_agent_mode.startup_problem()

    assert problem is not None
    assert "LLM_AGENT" in problem and "JEV_API_KEY" in problem
    assert "Set LLM_AGENT" in problem and "or set JEV_API_KEY" in problem
    assert no_agent_mode.runtime_problems() == [problem]


def test_the_startup_problem_reaches_a_serving_process_and_never_training():
    problem = no_agent_mode.startup_problem()

    assert problem in command_preview.load_problems(LIBRARY, runtime=True)
    # `fastworkflow train` calls load_problems without it, and stops on any problem: training does not need the agent.
    assert problem not in command_preview.load_problems(LIBRARY)


def test_a_jev_key_alone_is_no_problem_and_the_mode_is_named_once(monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "JEV_API_KEY", "key")

    assert no_agent_mode.startup_problem() is None
    assert no_agent_mode.runtime_problems() == []
    assert "no-agent mode" in no_agent_mode.startup_line()
    assert command_preview.load_problems(LIBRARY, runtime=True) == command_preview.load_problems(LIBRARY)


def test_an_agent_model_alone_is_no_problem_and_names_no_mode(monkeypatch):
    monkeypatch.setitem(fastworkflow._env_vars, "LLM_AGENT", "test/model")

    assert no_agent_mode.startup_problem() is None
    assert no_agent_mode.startup_line() is None


def test_run_with_no_agent_model_falls_back_to_deterministic_mode_and_says_so(monkeypatch):
    run_as_agent, banner = no_agent_mode.run_mode(assistant_requested=False)

    assert run_as_agent is False
    assert banner == no_agent_mode.MODE_BANNER
    for phrase in ("agent model", "deterministic mode", "Ad hoc requests are not planned here",
                   "planning commands run to their first open step"):
        assert phrase.lower() in banner.lower()
    # An explicit --assistant needs no banner; an agent model keeps the agent.
    assert no_agent_mode.run_mode(assistant_requested=True) == (False, None)
    monkeypatch.setitem(fastworkflow._env_vars, "LLM_AGENT", "test/model")
    assert no_agent_mode.run_mode(assistant_requested=False) == (True, None)

