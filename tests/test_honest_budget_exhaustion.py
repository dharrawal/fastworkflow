"""EXP-027: a truncated walk must not be reported as a whole one.

The defect these cover is not that the agent lies. It is that the runtime knows
the loop was cut short and never tells the model, so the model is handed four
completed identity walks and a request for a final answer and writes a report
covering eight. Measured on `exp025a-g2c`: fabricating attempts averaged 24.5
agent steps against a budget of 25, clean ones 12.9, and all 15 fabrications
were on population-walk cases.
"""

import pytest

from fastworkflow.turn_budget import LogicalTurnBudget, TurnPartial
from fastworkflow.utils.react import fastWorkflowReAct


def react():
    """An instance without __init__ — these are pure functions of their input."""
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent._exhausted_last_run = False
    return agent


TRAJECTORY = {
    "thought_0": "open the first identity",
    "tool_name_0": "execute_workflow_query",
    "observation_0": "Entered Identity context.",
    "thought_1": "list accounts",
    "tool_name_1": "execute_workflow_query",
    "observation_1": "1 account.",
}


class ThePartialComesFromTheRuntime:
    """The clause the whole change rests on."""


class TestPartial:

    def test_no_partial_when_the_turn_did_not_exhaust(self):
        agent = react()
        budget = LogicalTurnBudget(iteration_limit=25, iterations_consumed=12)
        assert agent._turn_partial(TRAJECTORY, budget) is None

    def test_the_counts_come_from_the_budget_and_the_trajectory(self):
        """Not from the model. A completed-count the model supplied would
        reintroduce, one level up, the trust problem this exists to close."""
        agent = react()
        agent._exhausted_last_run = True
        budget = LogicalTurnBudget(iteration_limit=25, iterations_consumed=25)
        partial = agent._turn_partial(TRAJECTORY, budget)
        assert partial.reason == "budget-exhausted"
        assert partial.iterations_consumed == 25
        assert partial.iteration_limit == 25
        # Two tool_name_ entries, both written by the LOOP because a tool ran.
        assert len(partial.commands_executed) == 2

    def test_the_partial_names_no_domain_items(self):
        """The runtime does not know what an item is; only a population cursor
        does, and that is EXP-026's. Naming what was executed is what this
        layer can say truthfully."""
        partial = TurnPartial("budget-exhausted", 25, 25, ("a", "b"))
        assert not hasattr(partial, "items_completed")
        assert "command call(s)" in partial.summary


class TestNotice:

    def test_the_model_is_told_it_did_not_finish(self):
        partial = TurnPartial("budget-exhausted", 25, 25, ("a", "b"))
        notice = fastWorkflowReAct._exhaustion_notice(partial)
        assert "did NOT finish" in notice
        assert "do not present a partial walk as a complete one" in notice

    def test_the_notice_instructs_rather_than_only_reports(self):
        """"You were cut short" and "do not report what you did not retrieve"
        are different instructions, and only the second changes the answer."""
        notice = fastWorkflowReAct._exhaustion_notice(
            TurnPartial("budget-exhausted", 25, 25, ()))
        assert "Report only what you actually retrieved" in notice

    def test_the_notice_carries_the_real_counts(self):
        notice = fastWorkflowReAct._exhaustion_notice(
            TurnPartial("budget-exhausted", 25, 25, ("a", "b", "c")))
        assert "25 of 25" in notice
        assert "3 command call(s)" in notice


class TestSpanAttributes:
    """Exit criterion: the outcome is distinguishable from success AND failure."""

    def test_the_partial_reaches_the_execute_span(self):
        import dspy
        from fastworkflow.workflow_execution_context import (
            _agent_result_attributes)
        partial = TurnPartial("budget-exhausted", 25, 25, ("a", "b"))
        result = dspy.Prediction(final_answer="partial report",
                                 exhausted=True, turn_partial=partial)
        attributes = _agent_result_attributes(result, 1)
        assert attributes["partial_reason"] == "budget-exhausted"
        assert attributes["partial_iterations_consumed"] == 25
        assert attributes["partial_commands_executed"] == 2

    def test_a_completed_turn_carries_no_partial_keys(self):
        """The falsifiable half: a turn that did not exhaust is unchanged."""
        import dspy
        from fastworkflow.workflow_execution_context import (
            _agent_result_attributes)
        result = dspy.Prediction(final_answer="full report", exhausted=False)
        attributes = _agent_result_attributes(result, 1)
        assert not [k for k in attributes if k.startswith("partial_")]


class TestPolicyInteraction:
    """EXP-027 x EXP-025a. Deferring work you have no budget for is honest."""

    def test_the_finish_policy_does_not_fire_on_an_exhausted_turn(self):
        from fastworkflow.policy_decision import (ContractFacts,
                                                  PolicyDecisionPoint,
                                                  PolicyMode)
        from ido_workflow.application.ask_policy import TABLE
        agent = react()
        agent.decision_point = PolicyDecisionPoint(TABLE, PolicyMode.ENFORCE)
        agent.contract_facts = ContractFacts(effect_kind="read_only",
                                             read_only_surface=True)
        deferring = {"final_answer": "Next Steps: Run `who_has_access_to` for "
                                     "the remaining identities."}
        args = {"user_query": "review the department"}
        # Not exhausted: the row is meant to fire, and does.
        assert agent._consult_finish_policy(deferring, TRAJECTORY, args)
        # Exhausted: the same answer is an honest partial, and is left alone.
        agent._exhausted_last_run = True
        assert agent._consult_finish_policy(deferring, TRAJECTORY, args) is None
