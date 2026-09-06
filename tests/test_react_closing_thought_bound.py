"""Contract tests for the ReAct closing-thought bound (ido-mn1.6.4).

EXP-028 Gate 4 v4 evidence: 131 ReAct LLM calls ended with
``finish_reason=length``. 62 of them were ``self.react`` steps — 46 that kept
writing the report AFTER the ``[[ ## completed ## ]]`` marker (text the adapter
discards) and 16 whose ``next_thought`` itself ran away. These tests pin the
instructions that forbid that, the deterministic cap that enforces it without a
retry, and the boundary the cap must NOT cross: the extraction step's
``final_answer`` is the composition step for the flat arm and the leaf answer
arms B/C concatenate, so it stays unbounded.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import dspy
import pytest

from fastworkflow.turn_budget import LogicalTurnBudget
from fastworkflow.utils.react import (
    MAX_EXTRACT_REASONING_CHARS,
    MAX_NEXT_THOUGHT_CHARS,
    REACT_STOP_SEQUENCES,
    REASONING_TRUNCATION_NOTICE,
    THOUGHT_TRUNCATION_NOTICE,
    fastWorkflowReAct,
)
from fastworkflow.workflow_agent import (
    FINAL_ANSWER_DESC,
    USER_QUERY_DESC,
    WorkflowAgentSignature,
)


def _tool() -> str:
    """A tool."""
    return "x"


@pytest.fixture(scope="module")
def agent():
    return fastWorkflowReAct(WorkflowAgentSignature, tools=[_tool])


def _desc(field) -> str:
    return (field.json_schema_extra or {}).get("desc") or ""


# ---------------------------------------------------------------------------
# 1. The instructions carry the bound and the citation rule
# ---------------------------------------------------------------------------

def test_react_instructions_state_the_bound_and_the_citation_rule(agent):
    instructions = agent.react.signature.instructions
    assert str(MAX_NEXT_THOUGHT_CHARS) in instructions
    assert "observation indices" in instructions
    assert "Never render the report, tables, or full listings inside next_thought" in instructions
    # The closing step specifically.
    assert "When you select `finish`, next_thought must be" in instructions
    assert "Do NOT write the final answer" in instructions
    # And the marker rule that ends the 46-call post-`completed` overrun class.
    assert "Your reply ends at the `[[ ## completed ## ]]` marker." in instructions


def test_next_thought_field_description_carries_the_bound(agent):
    desc = _desc(agent.react.signature.output_fields["next_thought"])
    assert f"at most {MAX_NEXT_THOUGHT_CHARS} characters" in desc
    assert "never render the report" in desc


def test_extraction_reasoning_is_bounded_and_cites(agent):
    reasoning = agent.extract.predict.signature.output_fields["reasoning"]
    desc = _desc(reasoning)
    assert f"at most {MAX_EXTRACT_REASONING_CHARS} characters" in desc
    assert "naming which observations you are drawing on" in desc
    assert "do not draft the answer here" in desc


def test_agent_signature_instructions_forbid_reports_in_reasoning_fields():
    instructions = WorkflowAgentSignature.instructions
    assert "never render the report, tables, or full listings inside" in instructions
    assert "written exactly once, in `final_answer`" in instructions


# ---------------------------------------------------------------------------
# 2. The composition step is not weakened
# ---------------------------------------------------------------------------

def test_final_answer_stays_comprehensive_and_unbounded(agent):
    """`final_answer` IS the composed deliverable — capping it would truncate it.

    Arm A returns it directly; arms B/C concatenate it per leaf in
    ``WorkflowExecutionContext._compose_plan_answer``, which makes no model
    call. So no character bound may appear on this field.
    """
    desc = _desc(agent.extract.predict.signature.output_fields["final_answer"])
    assert desc == FINAL_ANSWER_DESC
    assert "Comprehensive final answer with supporting evidence" in desc
    assert "at most" not in desc
    assert "characters" not in desc


def test_extraction_still_receives_trajectory_and_user_query(agent):
    fields = agent.extract.predict.signature.input_fields
    assert "trajectory" in fields
    assert "user_query" in fields


# ---------------------------------------------------------------------------
# 3. Arm invariance: the insight-enhanced signature carries the same contract
# ---------------------------------------------------------------------------

def test_execution_insight_signature_matches_the_module_level_one():
    """The `execution_insights` clone must not drift: only some runs supply
    insights, and a drifting clone would make the bound arm-dependent."""
    import fastworkflow.workflow_agent as wa

    enhanced = f"{WorkflowAgentSignature.__doc__}\n\nCRITICAL ANTI-PATTERNS TO AVOID:\nx"

    class AgentSignature(dspy.Signature):
        __doc__ = enhanced
        user_query = dspy.InputField(desc=wa.USER_QUERY_DESC)
        final_answer = dspy.OutputField(desc=wa.FINAL_ANSWER_DESC)

    assert wa.USER_QUERY_DESC == USER_QUERY_DESC
    assert wa.FINAL_ANSWER_DESC == FINAL_ANSWER_DESC
    assert _desc(AgentSignature.output_fields["final_answer"]) == FINAL_ANSWER_DESC
    assert "never render the report" in AgentSignature.instructions


# ---------------------------------------------------------------------------
# 4. The tool step is untouched
# ---------------------------------------------------------------------------

def test_tool_fields_are_byte_identical_to_the_dspy_defaults(agent):
    fields = agent.react.signature.output_fields
    assert _desc(fields["next_tool_name"]) == "${next_tool_name}"
    assert _desc(fields["next_tool_args"]) == "${next_tool_args}"


def test_react_stops_at_the_completed_marker_and_extract_does_not(agent):
    assert agent.react.config.get("stop") == list(REACT_STOP_SEQUENCES)
    assert REACT_STOP_SEQUENCES == ("[[ ## completed ## ]]",)
    # `completed` is not an output field, so ChatAdapter.parse never reads it,
    # and it is emitted strictly after next_tool_args.
    assert "completed" not in agent.react.signature.output_fields
    assert "stop" not in agent.extract.predict.config


# ---------------------------------------------------------------------------
# 5. The bound is enforced deterministically, without a retry
# ---------------------------------------------------------------------------

def _bare_agent(**tools):
    a = fastWorkflowReAct.__new__(fastWorkflowReAct)
    a._budget = LogicalTurnBudget(iteration_limit=5)
    a._step_seals = {}
    a.max_iters = 5
    a.inputs = {}
    a.current_trajectory = {}
    a._suspended = None
    a._safety_envelope = None
    a.tools = tools
    return a


OVERLONG = "R" * (MAX_NEXT_THOUGHT_CHARS * 4)


def test_overlong_closing_thought_is_truncated_once_with_no_retry():
    agent = _bare_agent(finish=lambda: "done")
    calls: list[int] = []

    def react(trajectory, **input_args):
        calls.append(1)
        return SimpleNamespace(
            next_thought=OVERLONG,
            next_tool_name="finish",
            next_tool_args={"a": 1},
        )

    agent.react = react
    trajectory: dict = {}
    result = agent._run_loop(
        trajectory, 0, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5), 0
    )

    assert result is None
    # Deterministic: exactly one model call, no retry storm.
    assert len(calls) == 1
    thought = trajectory["thought_0"]
    assert len(thought) <= MAX_NEXT_THOUGHT_CHARS
    assert thought.endswith(THOUGHT_TRUNCATION_NOTICE)
    # The tool step is untouched.
    assert trajectory["tool_name_0"] == "finish"
    assert trajectory["tool_args_0"] == {"a": 1}
    assert agent.current_trajectory["thought_0"] == thought


def test_thought_within_the_bound_is_passed_through_unchanged():
    agent = _bare_agent(finish=lambda: "done")
    short = "Done. Results are in observation_0 and observation_1 (uid abc123)."
    agent.react = lambda trajectory, **input_args: SimpleNamespace(
        next_thought=short, next_tool_name="finish", next_tool_args={}
    )
    trajectory: dict = {}
    agent._run_loop(
        trajectory, 0, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5), 0
    )
    assert trajectory["thought_0"] == short


def test_bound_applies_to_every_step_not_only_the_closing_one():
    agent = _bare_agent(_tool=lambda: "obs", finish=lambda: "done")
    preds = iter([
        SimpleNamespace(next_thought=OVERLONG, next_tool_name="_tool", next_tool_args={}),
        SimpleNamespace(next_thought="done", next_tool_name="finish", next_tool_args={}),
    ])
    agent.react = lambda trajectory, **input_args: next(preds)
    trajectory: dict = {}
    agent._run_loop(
        trajectory, 0, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5), 0
    )
    assert len(trajectory["thought_0"]) <= MAX_NEXT_THOUGHT_CHARS
    assert trajectory["tool_name_0"] == "_tool"
    assert trajectory["observation_0"] == "obs"


def test_async_path_bounds_the_thought_too():
    """Arm invariance: the async loop must not be a hole in the bound."""
    agent = _bare_agent(finish=lambda: "done")
    agent._safety_envelope = None

    class _Tool:
        async def acall(self, **kwargs):
            return "done"

    agent.tools = {"finish": _Tool()}
    agent.react = object()
    agent.extract = object()

    preds = iter([
        SimpleNamespace(next_thought=OVERLONG, next_tool_name="finish", next_tool_args={}),
    ])

    async def call(module, trajectory, **input_args):
        try:
            return next(preds)
        except StopIteration:
            return {"final_answer": "ok", "reasoning": "r"}

    agent._async_call_with_potential_trajectory_truncation = call

    result = asyncio.run(
        agent.aforward(user_query="q", budget=LogicalTurnBudget(iteration_limit=5))
    )
    assert len(result.trajectory["thought_0"]) <= MAX_NEXT_THOUGHT_CHARS
    assert result.trajectory["tool_name_0"] == "finish"


def test_extraction_reasoning_is_capped_but_final_answer_is_not():
    agent = _bare_agent(finish=lambda: "done")
    agent._exhausted_last_run = False
    agent.extract = object()
    long_answer = "A" * (MAX_EXTRACT_REASONING_CHARS * 10)
    calls: list[int] = []

    def call(module, trajectory, **input_args):
        calls.append(1)
        return dspy.Prediction(
            reasoning="B" * (MAX_EXTRACT_REASONING_CHARS * 3),
            final_answer=long_answer,
        )

    agent._call_with_potential_trajectory_truncation = call
    result = agent._finish(
        {"thought_0": "t"}, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )

    assert len(calls) == 1  # no retry
    assert len(result.reasoning) <= MAX_EXTRACT_REASONING_CHARS
    assert result.reasoning.endswith(REASONING_TRUNCATION_NOTICE)
    assert result.final_answer == long_answer  # deliverable untouched


def test_extraction_reasoning_bound_also_applies_to_mapping_results():
    """Some call paths hand back a plain mapping; the bound must not have a hole."""
    agent = _bare_agent(finish=lambda: "done")
    agent._exhausted_last_run = False
    agent.extract = object()
    agent._call_with_potential_trajectory_truncation = (
        lambda module, trajectory, **kwargs: {
            "reasoning": "B" * (MAX_EXTRACT_REASONING_CHARS * 3),
            "final_answer": "A" * (MAX_EXTRACT_REASONING_CHARS * 5),
        }
    )
    result = agent._finish(
        {"thought_0": "t"}, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )
    assert len(result.reasoning) <= MAX_EXTRACT_REASONING_CHARS
    assert len(result.final_answer) == MAX_EXTRACT_REASONING_CHARS * 5


def test_bound_text_is_a_pure_deterministic_truncation():
    f = fastWorkflowReAct._bound_text
    assert f(None, 10, "!") is None
    assert f(123, 10, "!") == 123
    short = "abc"
    assert f(short, 10, "!") is short
    out = f("x" * 100, 10, "!!")
    assert out == "xxxxxxxx!!"
    assert len(out) == 10
    # Same input, same output.
    assert f("x" * 100, 10, "!!") == out
