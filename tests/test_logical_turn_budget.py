"""FW-REQ-001: the ReAct budget belongs to one logical turn (EXP-010).

The defect these tests hold closed (requirements §5, GAP-01): the iteration
counter lived on the agent object for the life of the process, so a long first
turn starved every later turn in the session, while `ask_user` reset the counter
to -1 and handed the turn a whole fresh budget on every clarification.

The acceptance criteria are named in each test's docstring. The three the
defect made impossible are the first three.
"""

from __future__ import annotations

import uuid
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow.runtime_config import (
    DEFAULT_REACT_MAX_ITERATIONS,
    DERIVED_REACT_MAX_ITERATIONS,
    ITEMS_PER_TURN_TARGET,
    MEASURED_COMMANDS_PER_ITEM_P90,
    MEASURED_WALK_OVERHEAD_COMMANDS,
    REACT_MAX_ITERATIONS_ENV_VAR,
    RuntimeConfig,
    clear_runtime_config,
    derive_react_max_iterations,
    get_runtime_config,
    items_within_react_budget,
    register_runtime_config,
)
from fastworkflow.runtime_manifest import (
    ManifestConformanceError,
    RuntimeManifest,
    merge_and_gate,
    resolve_runtime_config,
)
from fastworkflow.session_state_store import (
    READABLE_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    IncompatibleSessionState,
)
from fastworkflow.turn_budget import (
    BudgetExhausted,
    LegacyTurnBudget,
    LogicalTurnBudget,
    budget_from_state,
)
from fastworkflow.utils.react import MissingTurnBudgetError, fastWorkflowReAct
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()
    clear_runtime_config()


def _bare_react_agent(**tools):
    """A fastWorkflowReAct without Module.__init__ (no dspy Tool wiring)."""
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent.max_iters = 5
    agent._budget = None
    agent.inputs = {}
    agent.current_trajectory = {}
    agent._suspended = None
    agent._exhausted_last_run = False
    agent._step_seals = {}
    agent.tools = tools
    agent.react = object()
    agent.extract = object()
    return agent


def _tool_then_finish(num_tool_steps: int):
    """A truncation-shim that runs `num_tool_steps` tool calls, then finishes."""
    preds = iter(
        [
            SimpleNamespace(
                next_thought=f"act{i}", next_tool_name="do_it", next_tool_args={}
            )
            for i in range(num_tool_steps)
        ]
        + [
            SimpleNamespace(
                next_thought="stop", next_tool_name="finish", next_tool_args={}
            )
        ]
    )

    def call(module, trajectory, **input_args):
        try:
            return next(preds)
        except StopIteration:
            return {"final_answer": "ok"}

    return call


def _never_finishes(agent):
    """A shim whose agent keeps calling a tool: only the budget stops it."""

    def call(module, trajectory, **input_args):
        if module is agent.extract:
            return {"final_answer": "ran out"}
        return SimpleNamespace(
            next_thought="again", next_tool_name="do_it", next_tool_args={}
        )

    return call


# ----------------------------------------------------------------------
# The acceptance criteria
# ----------------------------------------------------------------------


def test_two_consecutive_turns_each_consume_a_full_budget():
    """AC1: two consecutive test turns can each consume the configured full budget.

    Under the defect the second turn got zero iterations, because the counter
    the first turn left at the limit was never reset.
    """
    agent = _bare_react_agent(do_it=lambda: "did it", finish=lambda: "done")

    first = LogicalTurnBudget(iteration_limit=4)
    agent._call_with_potential_trajectory_truncation = _never_finishes(agent)
    result = agent.forward(query="first", budget=first)
    assert first.iterations_consumed == 4
    assert first.exhausted
    assert result.exhausted is True

    second = LogicalTurnBudget(iteration_limit=4)
    agent._call_with_potential_trajectory_truncation = _never_finishes(agent)
    result = agent.forward(query="second", budget=second)
    assert second.iterations_consumed == 4, "the second turn got its own full budget"
    assert result.exhausted is True


def test_a_new_turn_after_an_exhausted_turn_starts_clean():
    """AC3: exhausting a turn does not spend the next turn's budget."""
    agent = _bare_react_agent(do_it=lambda: "did it", finish=lambda: "done")

    exhausted = LogicalTurnBudget(iteration_limit=2)
    agent._call_with_potential_trajectory_truncation = _never_finishes(agent)
    agent.forward(query="burn it", budget=exhausted)
    assert exhausted.exhausted

    fresh = LogicalTurnBudget(iteration_limit=2)
    agent._call_with_potential_trajectory_truncation = _tool_then_finish(1)
    result = agent.forward(query="next", budget=fresh)
    assert fresh.iterations_consumed == 1
    assert result.exhausted is False


def test_resume_continues_on_the_suspended_turns_budget():
    """AC2: a suspended and resumed turn does not receive a fresh budget.

    Under the defect `ask_user` set the counter to -1, so a turn could ask a
    question and buy itself the whole budget again — repeatedly.
    """
    budget = LogicalTurnBudget(iteration_limit=5, iterations_consumed=3)
    agent = _bare_react_agent(finish=lambda: "done")
    agent._budget = budget
    agent._suspended = {
        "trajectory": {"thought_0": "ask", "tool_name_0": "ask_user", "tool_args_0": {}},
        "idx": 0,
        "input_args": {"query": "hello"},
        "max_iters": 5,
        "clarification": "Which one?",
    }
    agent._call_with_potential_trajectory_truncation = _tool_then_finish(0)

    agent.resume("the user's answer")

    # 3 already spent + 1 for the suspending step the resume completes. Nothing
    # replenished, and the finish step does not buy an extra iteration.
    assert budget.iterations_consumed == 4
    assert budget.iterations_remaining == 1


def test_exhaustion_is_reported_only_for_the_turn_that_spent_it():
    """AC4: exhaustion belongs to the turn whose own budget ran out."""
    agent = _bare_react_agent(do_it=lambda: "did it", finish=lambda: "done")

    spender = LogicalTurnBudget(iteration_limit=2)
    agent._call_with_potential_trajectory_truncation = _never_finishes(agent)
    assert agent.forward(query="spend", budget=spender).exhausted is True

    # A different turn, with its own budget, running one cheap step.
    innocent = LogicalTurnBudget(iteration_limit=2)
    agent._call_with_potential_trajectory_truncation = _tool_then_finish(0)
    assert agent.forward(query="cheap", budget=innocent).exhausted is False
    assert innocent.iterations_consumed == 0


def test_an_invalid_tool_selection_consumes_an_iteration():
    """Arch §6.4: invalid model/tool selections consume an iteration.

    They previously cost nothing, so an agent that could not name a tool could
    burn the turn's wall clock while the counter stood still.
    """
    agent = _bare_react_agent(finish=lambda: "done")

    def bad_pick(module, trajectory, **input_args):
        raise ValueError("no such tool")

    agent._call_with_potential_trajectory_truncation = bad_pick
    budget = LogicalTurnBudget(iteration_limit=10)
    agent._run_loop({}, 0, {"query": "x"}, budget, 0)

    # Three failures end the loop (the pre-existing exception_count rule); each
    # one is now charged.
    assert budget.iterations_consumed == 3


def test_a_budget_exhausted_by_invalid_selections_ends_the_turn():
    """The invalid-selection path must respect the limit it now spends."""
    agent = _bare_react_agent(finish=lambda: "done")

    def bad_pick(module, trajectory, **input_args):
        raise ValueError("no such tool")

    agent._call_with_potential_trajectory_truncation = bad_pick
    budget = LogicalTurnBudget(iteration_limit=2)
    agent._run_loop({}, 0, {"query": "x"}, budget, 0)

    assert budget.exhausted
    assert agent._exhausted_last_run is True


def test_forward_requires_a_budget():
    """Arch §6.4: forward() requires the budget and never creates one."""
    agent = _bare_react_agent(finish=lambda: "done")
    with pytest.raises(MissingTurnBudgetError):
        agent.forward(query="no budget here")


def test_iteration_counter_is_a_read_only_view():
    """AC-adjacent (clause 3): the budget is not a writable origin channel.

    The two things done to the old counter were reading `<= 0` as "the user
    typed this" and writing -1 to say so. Both are gone; what remains is a
    number a reader can look at.
    """
    agent = _bare_react_agent(finish=lambda: "done")
    assert agent.iteration_counter == 0

    agent._budget = LogicalTurnBudget(iteration_limit=5, iterations_consumed=2)
    assert agent.iteration_counter == 2

    with pytest.raises(AttributeError):
        agent.iteration_counter = 0


def test_invocation_origin_is_stated_not_inferred():
    """FW-REQ-001 clause 3: origin is a declared value, not a budget reading."""
    from fastworkflow.workflow_agent import (
        CONTEXT_KEY_INVOCATION_ORIGIN,
        InvocationOrigin,
    )

    assert InvocationOrigin.USER.value == "user"
    assert InvocationOrigin.AGENT.value == "agent"
    assert CONTEXT_KEY_INVOCATION_ORIGIN == "invocation_origin"

    # The retired flag has no writer left anywhere in the package.
    import subprocess

    package = Path(fastworkflow.__file__).parent
    hits = subprocess.run(
        ["grep", "-rn", "--include=*.py", 'context\\["is_user_command"\\]', str(package)],
        capture_output=True,
        text=True,
    )
    assert hits.stdout == "", f"is_user_command is still written: {hits.stdout}"


# ----------------------------------------------------------------------
# AC5 — the configuration surface, and its restrictive precedence
# ----------------------------------------------------------------------


def test_deployment_default_is_the_compatible_value():
    """AC5: a configurable limit, defaulting to 3.1.2's hard-coded 25."""
    config, problems = RuntimeConfig.from_env({})
    assert problems == []
    assert config.react_max_iterations == DEFAULT_REACT_MAX_ITERATIONS == 25


def test_the_default_covers_the_floor_the_measurement_derives():
    """`ido-24b.4`: the default is checked against measured cost, not inherited.

    This is the test the derivation exists for. It fails when a re-measured
    command surface pushes the floor above the deployment default — which is
    the moment a reader has to choose between raising the default and accepting
    that the target item count no longer fits in one turn. Either is a decision;
    silently walking fewer items than the corpus assumes is not.
    """
    assert DERIVED_REACT_MAX_ITERATIONS == derive_react_max_iterations()
    # 23 = 4 walk overhead + 3 items x 6.33 commands per item, p90.
    assert DERIVED_REACT_MAX_ITERATIONS == ceil(
        MEASURED_WALK_OVERHEAD_COMMANDS
        + ITEMS_PER_TURN_TARGET * MEASURED_COMMANDS_PER_ITEM_P90
    ) == 23
    assert DEFAULT_REACT_MAX_ITERATIONS >= DERIVED_REACT_MAX_ITERATIONS


def test_re_deriving_tracks_the_command_surface_in_both_directions():
    """A dearer command surface derives a larger floor and a cheaper one a
    smaller floor — the property that makes re-derivation worth doing at all
    rather than a decoration on the constant it produced."""
    assert derive_react_max_iterations(2.20) == 11   # cheapest contract measured
    assert derive_react_max_iterations(6.33) == 23   # dearest contract measured
    # A command split in two roughly doubles per-item cost.
    assert derive_react_max_iterations(2 * 6.33) > DEFAULT_REACT_MAX_ITERATIONS


def test_the_budget_reports_how_many_items_it_can_actually_walk():
    """The inverse: at 25 the dearest contract reaches 3 items and the cheapest
    9, which is why a 10-identity recertification is a multi-turn task and not a
    budget to be raised (`ido-24b.4`, EXP-027)."""
    assert items_within_react_budget(DEFAULT_REACT_MAX_ITERATIONS) == 3
    assert items_within_react_budget(DEFAULT_REACT_MAX_ITERATIONS, 2.20) == 9
    # The derived floor is the smallest limit that reaches the target, so one
    # iteration less does not.
    floor = derive_react_max_iterations()
    assert items_within_react_budget(floor) >= ITEMS_PER_TURN_TARGET
    assert items_within_react_budget(floor - 1) < ITEMS_PER_TURN_TARGET


def test_every_layer_may_lower_the_limit_and_none_may_raise_it():
    """Arch §6.0: effective = min(deployment, manifest, contract, host)."""
    config = RuntimeConfig(react_max_iterations=10)

    assert config.effective_react_max_iterations() == 10
    assert config.effective_react_max_iterations(manifest_limit=4) == 4
    assert config.effective_react_max_iterations(contract_limit=6, host_limit=3) == 3
    # Above the deployment maximum: not the minimum, so it changes nothing.
    assert config.effective_react_max_iterations(manifest_limit=99, host_limit=50) == 10
    # A layer that declares nothing does not participate and is not zero.
    assert config.effective_react_max_iterations(contract_limit=None) == 10


def test_a_bad_deployment_value_fails_startup_conformance():
    """An unusable limit stops startup rather than being silently replaced."""
    config, problems = RuntimeConfig.from_env({REACT_MAX_ITERATIONS_ENV_VAR: "0"})
    assert problems
    assert config.react_max_iterations == DEFAULT_REACT_MAX_ITERATIONS

    with pytest.raises(ManifestConformanceError):
        resolve_runtime_config({REACT_MAX_ITERATIONS_ENV_VAR: "not-a-number"})

    assert resolve_runtime_config(
        {REACT_MAX_ITERATIONS_ENV_VAR: "7"}
    ).react_max_iterations == 7


def test_a_workflow_manifest_may_declare_its_own_ceiling():
    """The manifest layer of the precedence, merged stricter-wins."""
    manifest = RuntimeManifest(
        schema_version=1, manifest_version="1.0", react_max_iterations=6
    )
    metadata = merge_and_gate(manifest, deployment_features={})
    assert metadata.react_max_iterations == 6

    # No manifest, or a manifest that declares nothing: None, not zero.
    assert merge_and_gate(None, deployment_features={}).react_max_iterations is None
    silent = RuntimeManifest(schema_version=1, manifest_version="1.0")
    assert merge_and_gate(silent, deployment_features={}).react_max_iterations is None


def test_a_manifest_ceiling_of_zero_is_rejected_by_the_schema():
    with pytest.raises(Exception):
        RuntimeManifest(
            schema_version=1, manifest_version="1.0", react_max_iterations=0
        )


def test_wec_resolves_the_turn_limit_through_the_configuration(
    initialized_fastworkflow, todo_workflow_path
):
    """The turn's budget is built from the resolved limit, not a caller default."""
    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"budget-{uuid.uuid4().hex}"
    )
    ctx.bind_app_workflow(wf)

    register_runtime_config(RuntimeConfig(react_max_iterations=9))
    ctx._begin_turn("a message")
    assert ctx.turn_budget.iteration_limit == 9

    # The host layer lowers it from the next fresh turn.
    ctx.bind_turn_iteration_limits(host_limit=3)
    ctx._begin_turn("another message")
    assert ctx.turn_budget.iteration_limit == 3

    # And cannot raise it.
    ctx.bind_turn_iteration_limits(host_limit=100)
    ctx._begin_turn("a third message")
    assert ctx.turn_budget.iteration_limit == 9
    ctx.close()


def test_a_running_turn_keeps_the_limit_it_started_with(
    initialized_fastworkflow, todo_workflow_path
):
    """Resume never recomputes the limit from changed configuration (§6.0)."""
    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"budget-{uuid.uuid4().hex}"
    )
    ctx.bind_app_workflow(wf)

    register_runtime_config(RuntimeConfig(react_max_iterations=8))
    ctx._begin_turn("start")
    budget = ctx.turn_budget

    register_runtime_config(RuntimeConfig(react_max_iterations=2))
    assert budget.iteration_limit == 8, "the running turn's limit did not move"
    assert ctx.turn_budget is budget
    ctx.close()


# ----------------------------------------------------------------------
# AC7 — the paths this must not disturb
# ----------------------------------------------------------------------


def test_the_deterministic_path_spends_no_budget(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    """AC7: existing deterministic and assistant paths are unaffected.

    A deterministic turn gets a budget like any other logical turn — WEC does
    not know yet whether the agent will run — and spends none of it.
    """
    ctx = WorkflowExecutionContext(run_as_agent=False)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"det-{uuid.uuid4().hex}"
    )
    ctx.bind_app_workflow(wf)

    from fastworkflow.command_executor import CommandExecutor

    monkeypatch.setattr(
        CommandExecutor,
        "invoke_command",
        classmethod(
            lambda cls, session, command: fastworkflow.CommandOutput(
                command_name="noop",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        ),
    )

    ctx._execute_message("add_todo milk")

    assert ctx.turn_budget is not None
    assert ctx.turn_budget.iterations_consumed == 0
    assert ctx.turn_budget.model_calls_consumed == 0
    # And the origin was stated, for the record the agent path also writes.
    from fastworkflow.workflow_agent import (
        CONTEXT_KEY_INVOCATION_ORIGIN,
        InvocationOrigin,
    )

    assert wf.context[CONTEXT_KEY_INVOCATION_ORIGIN] == InvocationOrigin.USER.value
    ctx.close()


# ----------------------------------------------------------------------
# AC8/AC9 — the schema 3 → 4 migration (arch §9.2)
# ----------------------------------------------------------------------


def test_schema_four_round_trips_the_budget():
    agent = _bare_react_agent(finish=lambda: "done")
    agent._budget = LogicalTurnBudget(iteration_limit=7, iterations_consumed=4)
    agent._suspended = {
        "trajectory": {"thought_0": "ask"},
        "idx": 0,
        "input_args": {"query": "hello"},
        "max_iters": 7,
        "clarification": "Which one?",
    }

    blob = agent.export_suspended()
    assert blob["budget"]["iterations_consumed"] == 4

    restored = _bare_react_agent(finish=lambda: "done")
    restored.import_suspended(blob)
    assert restored.budget.iteration_limit == 7
    assert restored.budget.iterations_consumed == 4
    assert restored.budget.is_legacy is False


def test_a_schema_three_blob_restores_into_an_explicit_legacy_budget():
    """AC8: the counter is kept; a per-turn reconstruction is not claimed."""
    v3_blob = {
        "trajectory": {"thought_0": "ask"},
        "idx": 0,
        "input_args": {"query": "hello"},
        "max_iters": 25,
        "clarification": "Which one?",
        "iteration_counter": 11,
    }

    agent = _bare_react_agent(finish=lambda: "done")
    agent.import_suspended(v3_blob)

    assert agent.budget.is_legacy is True
    assert agent.budget.legacy_counter == 11
    assert agent.budget.iterations_consumed == 11
    assert agent.budget.iteration_limit == 25


def test_a_schema_three_clarification_sentinel_is_preserved_not_laundered():
    """-1 was a sentinel, not a count. Clamping it silently would hide that."""
    legacy = LegacyTurnBudget.from_counter(-1, 25)
    assert legacy.legacy_counter == -1
    assert legacy.iterations_consumed == 0
    assert legacy.is_legacy is True
    # And it survives a schema-4 write of that same suspended turn.
    assert budget_from_state(legacy.to_state()).legacy_counter == -1


def test_a_malformed_schema_three_counter_is_rejected():
    """AC9: malformed schema-3 state is rejected explicitly (§9.2)."""
    with pytest.raises(ValueError):
        LegacyTurnBudget.from_counter("lots", 25)

    agent = _bare_react_agent(finish=lambda: "done")
    with pytest.raises(ValueError):
        agent.import_suspended(
            {
                "trajectory": {},
                "idx": 0,
                "input_args": {},
                "max_iters": 25,
                "iteration_counter": None,
            }
        )


def test_schema_three_is_readable_and_earlier_versions_are_not():
    assert SCHEMA_VERSION == 4
    assert READABLE_SCHEMA_VERSIONS == frozenset({3, 4})


def test_a_forward_version_blob_is_preserved_rather_than_discarded(
    initialized_fastworkflow, todo_workflow_path
):
    """AC9: fail closed on forward-version state, and keep it (§9.2)."""
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key="fwd")
    wf = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"fwd-{uuid.uuid4().hex}"
    )
    ctx.bind_app_workflow(wf)

    with pytest.raises(IncompatibleSessionState) as excinfo:
        ctx.apply_serialized_state(
            {"schema_version": SCHEMA_VERSION + 1, "awaiting_user": True}
        )
    assert excinfo.value.preserve is True
    assert not ctx.awaiting_user
    ctx.close()


def test_a_malformed_suspended_blob_fails_closed(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    """A blob whose agent state cannot be rebuilt is unreadable, not partly applied."""
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key="mal")
    wf = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"mal-{uuid.uuid4().hex}"
    )
    ctx.bind_app_workflow(wf)
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    ctx._workflow_tool_agent = _bare_react_agent(finish=lambda: "done")

    with pytest.raises(IncompatibleSessionState):
        ctx.apply_serialized_state(
            {
                "schema_version": SCHEMA_VERSION,
                "awaiting_user": True,
                "react": {
                    "trajectory": {},
                    "idx": 0,
                    "input_args": {},
                    "max_iters": 25,
                    "budget": {"kind": "logical", "iteration_limit": "unbounded"},
                },
            }
        )
    ctx.close()


# ----------------------------------------------------------------------
# The budget model itself
# ----------------------------------------------------------------------


def test_the_serialized_budget_is_json_safe():
    """`state_serialization.validate_state` rejects datetime and Decimal."""
    from datetime import datetime, timezone
    from decimal import Decimal

    from fastworkflow.state_serialization import validate_state

    budget = LogicalTurnBudget(
        iteration_limit=5,
        deadline_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        cost_limit=Decimal("1.50"),
        cost_consumed=Decimal("0.25"),
    )
    state = budget.to_state()
    validate_state({"react": {"budget": state}})

    restored = budget_from_state(state)
    assert restored.deadline_at == budget.deadline_at
    assert restored.cost_limit == Decimal("1.50")


def test_declared_limits_are_enforced_and_absent_ones_only_record():
    """Arch §6.0: an absent limit records spend, it does not forbid it."""
    unbounded = LogicalTurnBudget(iteration_limit=5)
    for _ in range(50):
        unbounded.consume_model_call()
        unbounded.consume_command_call()
    assert unbounded.model_calls_consumed == 50
    assert unbounded.command_calls_consumed == 50

    bounded = LogicalTurnBudget(iteration_limit=5, model_call_limit=1)
    bounded.consume_model_call()
    with pytest.raises(BudgetExhausted) as excinfo:
        bounded.consume_model_call()
    assert excinfo.value.resource == "model_call"
