"""The parameter-extraction round ordinal, against its real co-writers (fix-8ko2, fix-o1lg S7).

`retry_round_ordinal` counts attempts within one parameter-extraction error state. The count is the number of questions
the pending parameter loop (`parameter_clarification.ParameterLoop.asks`) has put, and the loop is the ONE piece of state
of that error state (fix-o1lg S7 replaced `stored_parameters`, `stored_parameter_round` and `missing_value_question`):

- `Workflow.end_command_processing` deletes the loop, so no counter is left behind pointing at a command that already
  finished.
- `WorkflowExecutionContext.serialize_state` / `apply_serialized_state` carry the loop, so a restored session's third
  attempt is still the third (this closes fix-7gp9: the round used to be unknown after a restore).

These tests hold the rule to the real objects rather than to a description of them -- a real `Workflow`, the real
serializer, the real restore.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.parameters import parameter_clarification
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import round_ordinal
from fastworkflow.workflow_execution_context import WorkflowExecutionContext




@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield tmp_path
    RoutingRegistry.clear_registry()


def _make_ctx(workflow_path: str, channel_id: str) -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key=channel_id)
    workflow = fastworkflow.Workflow.create(workflow_path, workflow_id_str=channel_id)
    ctx.bind_app_workflow(workflow)
    return ctx


def _enter_parameter_extraction(ctx: WorkflowExecutionContext) -> str:
    """The CME parked at a command's parameter stage, with no question put yet."""
    command_name = "TodoListManager/create_todo_list"
    cme = ctx._cme_workflow.context
    cme["command"] = "create a todo list"
    cme["command_name"] = command_name
    return command_name


def _store_a_failed_attempt(ctx: WorkflowExecutionContext, command_name: str) -> None:
    """One round of "extraction ran and came up short", through the real writer (``record_round``): the first creates
    the loop, a later one puts another question on it."""
    context = ctx._cme_workflow.context
    loop = parameter_clarification.pending(context) or parameter_clarification.ParameterLoop(
        command=command_name, kind=parameter_clarification.KIND_EXECUTION)
    parameter_clarification.record_round(loop, parameter_clarification.Extraction(missing=("description",)), "which?")
    parameter_clarification.store(context, loop)


def _round(ctx: WorkflowExecutionContext):
    return round_ordinal(parameter_clarification.pending(ctx._cme_workflow.context))


# ---------------------------------------------------------------------------
# The ordinary sequence, so the regressions below are read against a baseline
# that is known to be right.
# ---------------------------------------------------------------------------

def test_a_clean_sequence_numbers_its_attempts_from_zero(
    initialized_fastworkflow, todo_workflow_path
):
    ctx = _make_ctx(todo_workflow_path, f"cme_{uuid.uuid4().hex[:8]}")
    command_name = _enter_parameter_extraction(ctx)
    try:
        assert _round(ctx) == 0, "the first attempt is attempt 0, not a retry"

        _store_a_failed_attempt(ctx, command_name)
        assert _round(ctx) == 1, "the attempt resuming from one question is #1"

        _store_a_failed_attempt(ctx, command_name)
        assert _round(ctx) == 2

        # Success ends the error state, and the count of its attempts ends with it: they are one piece of state.
        parameter_clarification.clear(ctx._cme_workflow.context)
        assert _round(ctx) == 0, "the next command starts over"
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# Co-writer 1: Workflow.end_command_processing.
# ---------------------------------------------------------------------------

def test_end_command_processing_leaves_no_counter_behind(
    initialized_fastworkflow, todo_workflow_path
):
    """A real `end_command_processing`, not a simulation of one.

    The counter used to be a key of its own that `end_command_processing` did not delete, so the next command's FIRST
    extraction could report itself as retry #3 of a command that already finished. It is the loop's own count now, and
    the loop goes with the command.
    """
    ctx = _make_ctx(todo_workflow_path, f"cme_{uuid.uuid4().hex[:8]}")
    command_name = _enter_parameter_extraction(ctx)
    try:
        for _ in range(3):
            _store_a_failed_attempt(ctx, command_name)
        assert _round(ctx) == 3

        ctx._cme_workflow.end_command_processing()

        assert parameter_clarification.pending(ctx._cme_workflow.context) is None
        assert _round(ctx) == 0

        # Nor does anything seed the next sequence: the new command's first question is round 1, not 4.
        _store_a_failed_attempt(ctx, command_name)
        assert _round(ctx) == 1
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# Co-writer 2: the real suspend/rehydrate path.
# ---------------------------------------------------------------------------

def test_the_serializer_carries_the_count(
    initialized_fastworkflow, todo_workflow_path
):
    """The loop is saved whole, so the round survives a suspend (fix-7gp9: it used to be lost)."""
    channel_id = f"cme_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    command_name = _enter_parameter_extraction(ctx)
    _store_a_failed_attempt(ctx, command_name)
    _store_a_failed_attempt(ctx, command_name)
    assert _round(ctx) == 2

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["waiting"] is not None and blob["waiting"]["kind"] == "parameters"
    assert blob["waiting"]["payload"]["loop"]["asks"] == 2
    assert blob["waiting"]["payload"]["loop"]["open"] == ["description"]
    assert "cme" not in blob and "plan_cme" not in blob


def test_a_round_restored_from_a_continuation_is_the_round_it_was(
    initialized_fastworkflow, todo_workflow_path
):
    """The restored session really is mid-retry, and says which retry."""
    channel_id = f"cme_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    command_name = _enter_parameter_extraction(ctx)
    _store_a_failed_attempt(ctx, command_name)
    _store_a_failed_attempt(ctx, command_name)

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)
    try:
        loop = parameter_clarification.pending(restored._cme_workflow.context)
        assert loop is not None and loop.command == command_name and loop.open == ["description"]
        assert _round(restored) == 2

        # A further failed attempt counts up from the number that was saved.
        _store_a_failed_attempt(restored, command_name)
        assert _round(restored) == 3

        # And the error state ending starts the count over.
        restored._cme_workflow.end_command_processing()
        assert _round(restored) == 0
        _store_a_failed_attempt(restored, command_name)
        assert _round(restored) == 1
    finally:
        restored.close()


# ---------------------------------------------------------------------------
# Corruption is unknown too, never a number that happens to parse.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad", [True, False, 2.0, "3", -1, 0, None, [], {"round": 1}]
)
def test_an_unreadable_count_in_the_loop_is_unknown(
    initialized_fastworkflow, todo_workflow_path, bad
):
    """`True` is not round 1 and `"3"` is not round 3.

    0 is in this list on purpose: a loop is only saved after a question, so a 0 beside a pending loop did not come from
    the writer and saying "first attempt" on the strength of it would be a guess.
    """
    ctx = _make_ctx(todo_workflow_path, f"cme_{uuid.uuid4().hex[:8]}")
    command_name = _enter_parameter_extraction(ctx)
    try:
        _store_a_failed_attempt(ctx, command_name)
        state = parameter_clarification.pending(ctx._cme_workflow.context).to_state()
        state["asks"] = bad
        parameter_clarification.store(ctx._cme_workflow.context, parameter_clarification.ParameterLoop(**state))

        assert _round(ctx) is None
    finally:
        ctx.close()
