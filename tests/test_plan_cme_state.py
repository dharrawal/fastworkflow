"""WEC-owned plan CME: second instance, its own which-command state, and session state (fix-kk46.6)."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import intent_loop, waiting
from fastworkflow import approval_binding, parameter_clarification
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import model_of, park
from fastworkflow.review_engine import open_review
from fastworkflow.review_proposals import SubTask, SubTaskProposal
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

PLAN_DROP = "ProposalReview/drop"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


def _make_ctx(workflow_path: str, channel_id: str) -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key=channel_id)
    workflow = fastworkflow.Workflow.create(workflow_path, workflow_id_str=channel_id)
    ctx.bind_app_workflow(workflow)
    return ctx


def _drop_params_class(ctx: WorkflowExecutionContext):
    routing = fastworkflow.RoutingRegistry.get_definition(
        ctx._plan_workflow.folderpath
    )
    return routing.get_command_class(
        PLAN_DROP, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
    )


def _review_out(ctx: WorkflowExecutionContext, record_id: str = "review-1", *, gate: bool = True) -> None:
    """The turn's review record (the slot question of the plan CME is nested in it)."""
    ctx._waiting = waiting.Waiting(waiting.KIND_REVIEW, payload={"record_id": record_id, "slot": None, "gate": gate})


def _enter_plan_extraction(
    ctx: WorkflowExecutionContext,
    *,
    missing_value_question: str | None = None,
) -> None:
    params_class = _drop_params_class(ctx)
    assert params_class is not None
    cme = ctx._plan_cme.context
    cme["command"] = "drop step 2"
    cme["command_name"] = PLAN_DROP
    park(ctx._plan_cme, PLAN_DROP, params_class.model_construct(items=[]),
         question=missing_value_question or "", request_text="drop step 2")


def _tasks(*goals: str) -> SubTaskProposal:
    return SubTaskProposal(items=[SubTask(goal=g) for g in goals])


def test_wec_owns_plan_workflow_and_cme(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    assert ctx._plan_workflow is not None
    assert ctx._plan_cme is not None
    assert ctx._plan_workflow._current_command_context_name == "ProposalReview"
    assert ctx._plan_cme.context["app_workflow"] is ctx._plan_workflow
    assert "run_as_agent" not in ctx._plan_workflow.context
    assert "run_as_agent" not in ctx._plan_cme.context


def test_the_two_cmes_keep_their_own_pending_which_command_question(todo_workflow_path):
    """The plan CME and the application's CME are two instances. The pending which-command question was a row in
    a file under ``___convo_info`` that the plan CME had to be redirected away from (``convo_path``); it is now
    plain data in each CME's own context (``intent_loop``), so the two cannot collide and nothing is redirected."""
    from fastworkflow import intent_loop

    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    assert "convo_path" not in ctx._plan_cme.context
    record = intent_loop.open_question("x", [{"name": "A/b", "description": "", "closest": True}])
    _review_out(ctx)                    # the plan CME's question is nested in the review record
    intent_loop.store(ctx._plan_cme.context, record)
    assert intent_loop.pending_record(ctx._cme_workflow.context) is None and ctx.waiting.kind == "review"
    assert intent_loop.pending_record(ctx._plan_cme.context) is record
    ctx._clear_cme_continuation(ctx._plan_cme)
    assert intent_loop.pending_record(ctx._plan_cme.context) is None


def test_plan_cme_round_trips_while_subtask_review_open(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    _review_out(ctx)
    _enter_plan_extraction(
        ctx, missing_value_question="Which of the 2 steps do you mean?",
    )

    blob = ctx.serialize_state(channel_id=channel)
    assert "plan_cme" not in blob and "review_slot" not in blob and blob["waiting"]["kind"] == "review"
    slot = blob["waiting"]["payload"]["slot"]
    assert slot is not None and slot["payload"]["command_name"] == PLAN_DROP
    assert slot["payload"]["after"] == "edit_slot"
    assert slot["payload"]["loop"]["question"] == (
        "Which of the 2 steps do you mean?"
    )

    ctx.close()
    restored = _make_ctx(todo_workflow_path, channel)
    restored.apply_serialized_state(blob)

    cme = restored._plan_cme.context
    assert restored._waiting.kind == "review" and restored._waiting.payload["slot"] is waiting.port_of(cme).get()
    assert cme["command_name"] == PLAN_DROP
    loop = parameter_clarification.pending(cme)
    assert loop.question == "Which of the 2 steps do you mean?"
    assert intent_loop.waiting_for(cme) == intent_loop.WAITING_PARAMETERS
    params = model_of(loop, _drop_params_class(restored))
    assert type(params) is _drop_params_class(restored)
    assert loop.open == ["items"] and params.items == []


def test_plan_cme_slot_cannot_exist_without_a_review(todo_workflow_path):
    """The slot question is nested in the review record (step 8): with no review out there is nowhere to hold it, so a
    save and restore never carries one (it used to be saved beside the review and dropped on restore)."""
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    with pytest.raises(AssertionError, match="no review out"):
        _enter_plan_extraction(ctx)
    ctx._plan_cme.context.pop("command", None)
    ctx._plan_cme.context.pop("command_name", None)

    blob = ctx.serialize_state(channel_id=channel)
    assert blob["waiting"] is None and "review_slot" not in blob

    restored = _make_ctx(todo_workflow_path, channel)
    restored.apply_serialized_state(blob)
    cme = restored._plan_cme.context
    assert cme.get("command_name") is None
    assert parameter_clarification.pending(cme) is None
    assert intent_loop.waiting_for(cme) is None


def test_active_subtask_record_restores_plan_without_frame_flag(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    store = approval_binding.store_for(ctx.app_workflow)
    record = open_review(ctx.app_workflow, store, _tasks("alpha", "beta"), surface=store.surface)
    _review_out(ctx, record.id, gate=False)
    _enter_plan_extraction(ctx)

    blob = ctx.serialize_state(channel_id=channel)
    restored = _make_ctx(todo_workflow_path, channel)
    restored.apply_serialized_state(blob)

    cme = restored._plan_cme.context
    assert cme["command_name"] == PLAN_DROP
    store = approval_binding.store_for(restored.app_workflow)
    assert store.plan_cme is restored._plan_cme
    assert store.plan_workflow is restored._plan_workflow


def test_clear_live_pending_clears_plan_cme(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    _review_out(ctx)
    _enter_plan_extraction(ctx)
    ctx._clear_live_pending()
    cme = ctx._plan_cme.context
    assert cme.get("command_name") is None
    assert intent_loop.waiting_for(cme) is None


def test_plan_review_record_is_the_workflow_root_across_contexts_on_one_channel(todo_workflow_path):
    """fix-1hs8: the wildcard walk goes to the parent until it reaches the root. The review record is that root,
    and a second context built on the same channel (a restart, a restore) reuses the one workflow and its root:
    no second root assignment, and the record is still the root so the walk stops there instead of asking the
    record's context class for a parent it does not have."""
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    first = _make_ctx(todo_workflow_path, channel)
    second = _make_ctx(todo_workflow_path, channel)
    assert second._plan_workflow is first._plan_workflow
    workflow = second._plan_workflow
    assert workflow.root is not None
    assert workflow._current_command_context is workflow.root
    assert workflow._current_command_context_name == "ProposalReview"
