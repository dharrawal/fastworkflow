"""Round-trip for the logical-turn accumulator and the CME continuation keys.

Before schema 2 the pending snapshot carried neither, so a restored session
started a fresh logical turn and re-extracted parameters from the answer text
alone. These tests build the state with real objects -- a real Workflow, the
real command Input class, real CommandOutput instances -- rather than driving a
live LLM, because the intent classifier's trained artifacts are not in git and
a test that needs a provider is a latency flake waiting to happen (fix-wi3).
"""

from __future__ import annotations

import copy
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from dspy.utils.exceptions import LMTimeoutError

import fastworkflow
from fastworkflow import result_handles
from fastworkflow.observability_store import SQLiteTraceSink
from fastworkflow.plan import (
    Binding,
    CompositeGroup,
    CompositePackingMetrics,
    PlanEdge,
    PlanExecutionMetadata,
    PlanNode,
    PlanRecord,
)
from fastworkflow.session_state_store import (
    SCHEMA_VERSION,
    IncompatibleSessionState,
)
from fastworkflow.plan_execution import (
    PlanExecutionArm,
    PlanExecutionBinding,
    PlanExecutionScope,
)
from fastworkflow.typed_failure import (
    CODE_EXTRACTION_TRUNCATED,
    extraction_truncated_failure,
)
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
    workflow = fastworkflow.Workflow.create(
        workflow_path, workflow_id_str=channel_id
    )
    ctx.bind_app_workflow(workflow)
    return ctx


def _params_class(ctx: WorkflowExecutionContext, command_name: str):
    routing = fastworkflow.RoutingRegistry.get_definition(
        ctx.app_workflow.folderpath
    )
    return routing.get_command_class(
        command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
    )


def _enter_parameter_extraction(ctx: WorkflowExecutionContext) -> str:
    """Put the CME workflow in the state a failed extraction leaves behind.

    parameter_extraction stores an instance built with model_construct whose
    missing fields hold the NOT_FOUND sentinel, which is exactly the shape
    ordinary validation refuses -- so a restore that validates would reject the
    very state it exists to carry.
    """
    command_name = "TodoListManager/create_todo_list"
    params_class = _params_class(ctx, command_name)
    assert params_class is not None, "test fixture needs a real Input class"

    cme = ctx._cme_workflow.context
    cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command"] = "create a todo list"
    cme["command_name"] = command_name
    cme["stored_parameters"] = params_class.model_construct(description="NOT_FOUND")
    return command_name


def _open_a_turn(ctx: WorkflowExecutionContext) -> None:
    ctx._begin_turn("create a todo list")
    ctx.append_turn_output(
        fastworkflow.CommandOutput(
            command_name="TodoListManager/create_todo_list",
            command_parameters="create a todo list",
            command_response=
                fastworkflow.CommandResponse(response="need a description", success=False),
            started_at=datetime.now(timezone.utc),
        )
    )


def _progress_plan() -> PlanRecord:
    task_key = "review::subject=Casey::<none>"
    return PlanRecord(
        plan_id="progress-plan",
        mode="enforce",
        nodes=(
            PlanNode(
                goal_id="g1",
                level="task",
                goal_text="Casey is reviewed.",
                visibility="public",
                task_key=task_key,
            ),
            PlanNode(
                goal_id="g1.1",
                parent_goal_id="g1",
                level="commands",
                goal_text="Review Casey.",
                executable_goal_text="Review Casey.",
                executable=True,
                status="done",
                command_call_ids=("call-review-casey",),
            ),
        ),
        requested_public_task_keys=(task_key,),
        compiled_public_task_keys=(task_key,),
    )


def test_cme_continuation_survives_a_round_trip(
    initialized_fastworkflow, todo_workflow_path
):
    channel_id = f"cme_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    command_name = _enter_parameter_extraction(ctx)

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["cme"] is not None
    assert blob["cme"]["command_name"] == command_name
    assert blob["cme"]["stored_parameters"] == {"description": "NOT_FOUND"}

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    cme = restored._cme_workflow.context
    assert cme["command_name"] == command_name
    assert cme["command"] == "create a todo list"
    assert (
        cme["NLU_Pipeline_Stage"]
        == fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    )

    params = cme["stored_parameters"]
    # Rebuilt as the real Input class, not a dict: parameter_extraction merges
    # into it with getattr and type(...).model_fields, both of which a dict fails.
    assert type(params) is _params_class(restored, command_name)
    assert params.description == "NOT_FOUND"
    restored.close()


def test_sentinel_in_a_typed_field_survives_restore(
    initialized_fastworkflow, todo_workflow_path
):
    """The sentinel goes into the field whatever the field's declared type is.

    parameter_extraction writes NOT_FOUND into every missing field, so an int
    field holds the string "NOT_FOUND". That is unvalidatable by construction,
    which is exactly why restore rebuilds with model_construct: model_validate
    would raise on the very state the snapshot exists to carry, and the session
    would be stranded mid-extraction with no way back.
    """
    channel_id = f"typed_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)

    command_name = "TodoListManager/get_todo_list"
    params_class = _params_class(ctx, command_name)
    assert "id" in params_class.model_fields

    cme = ctx._cme_workflow.context
    cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command"] = "get the todo list"
    cme["command_name"] = command_name
    cme["stored_parameters"] = params_class.model_construct(id="NOT_FOUND")

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["cme"]["stored_parameters"] == {"id": "NOT_FOUND"}

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    params = restored._cme_workflow.context["stored_parameters"]
    assert type(params) is params_class
    assert params.id == "NOT_FOUND"
    restored.close()


def test_turn_accumulator_survives_a_round_trip(
    initialized_fastworkflow, todo_workflow_path
):
    channel_id = f"turn_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    _open_a_turn(ctx)
    original_key = ctx._turn_key
    assert original_key is not None

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    # The same logical turn, not a new one: resume deliberately skips
    # _begin_turn, so a lost key silently splits one turn's telemetry in two.
    assert restored._turn_key == original_key
    assert restored._turn_user_message == "create a todo list"
    assert restored._turn_entry_context == ctx._turn_entry_context

    assert len(restored._turn_outputs) == 1
    output = restored._turn_outputs[0]
    assert isinstance(output, fastworkflow.CommandOutput)
    assert output.command_name == "TodoListManager/create_todo_list"
    assert output.command_response.response == "need a description"
    assert output.command_response.success is False
    assert output.started_at is not None
    restored.close()


def test_turn_output_with_typed_params_survives_cold_rehydrate(
    initialized_fastworkflow, todo_workflow_path
):
    """CommandExecutor assigns a Pydantic params instance to command_parameters.

    The field was long declared ``str`` while the write path stored a model;
    ``model_dump(mode="json")`` emitted a dict and ``model_validate`` on
    cold-rehydrate rejected it (fix-fjh / A10 honesty). Typed in-memory,
    dict-on-wire, and restore must all agree.
    """
    channel_id = f"params_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    command_name = "TodoListManager/create_todo_list"
    params_class = _params_class(ctx, command_name)
    assert params_class is not None

    ctx._begin_turn("create a todo list called groceries")
    output = fastworkflow.CommandOutput(
        command_name=command_name,
        command_response=fastworkflow.CommandResponse(
            response="created", success=True
        ),
        started_at=datetime.now(timezone.utc),
    )
    # Mirror CommandExecutor.invoke: assign the typed instance (no re-validate).
    output.command_parameters = params_class(description="groceries")
    ctx.append_turn_output(output)

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["turn"]["outputs"][0]["command_parameters"] == {
        "description": "groceries"
    }

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    restored_params = restored._turn_outputs[0].command_parameters
    assert restored_params == {"description": "groceries"}
    restored.close()


def test_ask_user_entry_is_still_completable_after_restore(
    initialized_fastworkflow, todo_workflow_path
):
    """The restored accumulator has to be live state, not an inert record.

    complete_ask_user_entry scans backwards for an unanswered ask_user and
    fills it in. If restore produced dicts, or dropped success=False, the
    user's answer would land nowhere and the entry would stay unanswered.
    """
    channel_id = f"ask_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("add something")
    ctx.append_ask_user_entry("Which list?")

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)
    restored.complete_ask_user_entry("the groceries one")

    entry = restored._turn_outputs[-1]
    assert entry.command_name == "ask_user"
    assert entry.command_response.response == "the groceries one"
    assert entry.command_response.success is True
    restored.close()


def test_no_open_turn_or_command_serializes_as_absent(
    initialized_fastworkflow, todo_workflow_path
):
    """Absent must stay absent, or every idle session looks mid-command."""
    channel_id = f"idle_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)

    assert ctx.has_open_command() is False
    blob = ctx.serialize_state(channel_id=channel_id)
    assert blob["turn"] is None
    assert blob["cme"] is None

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)
    assert restored._turn_key is None
    assert restored._turn_outputs == []
    assert "stored_parameters" not in restored._cme_workflow.context
    ctx.close()
    restored.close()


def test_has_open_command_tracks_the_cme_keys(
    initialized_fastworkflow, todo_workflow_path
):
    """This predicate decides whether the blob is written at all."""
    channel_id = f"open_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    assert ctx.has_open_command() is False

    _enter_parameter_extraction(ctx)
    assert ctx.has_open_command() is True

    # end_command_processing is what a completed command runs.
    ctx._cme_workflow.end_command_processing()
    assert ctx.has_open_command() is False

    # It clears command and stored_parameters but leaves command_name behind,
    # which is why the predicate must not key off command_name: a session that
    # merely ran one command would otherwise look mid-extraction forever.
    assert ctx._cme_workflow.context.get("command_name") is not None
    assert ctx.serialize_state(channel_id=channel_id)["cme"] is None
    ctx.close()


def test_unresolvable_parameter_class_resets_rather_than_resuming(
    initialized_fastworkflow, todo_workflow_path
):
    """A command that no longer exists must not leave the session mid-extraction.

    Restoring PARAMETER_EXTRACTION without a resolvable class would strand the
    session: wildcard.py reads context["command_name"] unconditionally at that
    stage, and the next message would merge into a command that can never run.
    """
    channel_id = f"gone_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    _enter_parameter_extraction(ctx)
    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    blob["cme"]["command_name"] = "TodoListManager/command_deleted_in_a_later_release"

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    cme = restored._cme_workflow.context
    assert (
        cme["NLU_Pipeline_Stage"] == fastworkflow.NLUPipelineStage.INTENT_DETECTION
    )
    assert "command_name" not in cme
    assert "stored_parameters" not in cme
    restored.close()


def test_v1_blob_is_refused_rather_than_partly_restored(
    initialized_fastworkflow, todo_workflow_path
):
    """Schema 1 lacked exactly the fields whose absence made restore wrong."""
    channel_id = f"v1_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    _enter_parameter_extraction(ctx)
    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    legacy = {k: v for k, v in blob.items() if k not in ("turn", "cme")}
    legacy["schema_version"] = 1

    restored = _make_ctx(todo_workflow_path, channel_id)
    with pytest.raises(IncompatibleSessionState):
        restored.apply_serialized_state(legacy)
    assert "stored_parameters" not in restored._cme_workflow.context
    restored.close()


def test_schema_version_is_current(initialized_fastworkflow, todo_workflow_path):
    """Adding fields without bumping would let an old reader half-apply a new blob."""
    assert SCHEMA_VERSION == 8
    channel_id = f"ver_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    assert ctx.serialize_state(channel_id=channel_id)["schema_version"] == SCHEMA_VERSION
    ctx.close()


def test_schema_eight_restores_presentation_and_typed_truncation_state(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"schema8-complete-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("review Casey")
    plan = _progress_plan()
    truncation = extraction_truncated_failure(max_tokens=4096)
    plan.execution = PlanExecutionMetadata(
        arm="b",
        packing_applied=False,
        schedule_sha256="sha256:schedule",
        executed_leaf_goal_ids=("g1.1",),
        public_task_keys=plan.compiled_public_task_keys,
        extraction_truncated_goal_ids=("g1.1",),
        extraction_truncation_failures={"g1.1": truncation},
    )
    ctx._turn_plan = plan
    ctx._turn_plan_answers = [
        {"goal_id": "g1.1", "answer": "partial Casey evidence"}
    ]
    ctx._turn_presented_results = [
        {
            "leaf_goal_id": "g1.1",
            "handles": [{"handle_id": "h" * 32, "trimmed": True}],
            "unresolved": [],
            "omitted_handles": [],
            "trimmed": True,
            "truncation_classification": (
                result_handles.PRESENTATION_TRUNCATION_CLASSIFICATION
            ),
        }
    ]
    ctx._turn_agent_result = SimpleNamespace(
        exhausted=False,
        extraction_truncated=True,
        extraction_truncated_reason=CODE_EXTRACTION_TRUNCATED,
        extraction_truncated_goal_ids=("g1.1",),
        extraction_failure=truncation,
        plan_outcome="completed",
    )
    handle_id = "a" * 32
    ctx.result_handles.put(
        result_handles.StoredResult(
            handle_id=handle_id,
            command_name="show_holders",
            kind="holders",
            summary="two holders",
            ordering="backend order",
            total=2,
            page_size=1,
            filters={"department": "finance"},
            producer_filter_applied=True,
            classification="user-text",
            items=["first", "second"],
        )
    )
    result_handles.fetch_page(
        handle_id,
        contains="first",
        host=ctx,
    )

    blob = ctx.serialize_state(channel_id=channel_id)
    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(json.loads(json.dumps(blob)))

    assert restored._turn_presented_results == ctx._turn_presented_results
    assert restored._turn_agent_result.extraction_truncated_goal_ids == (
        "g1.1",
    )
    assert restored._turn_agent_result.extraction_failure.code == (
        CODE_EXTRACTION_TRUNCATED
    )
    assert restored._turn_plan.execution.extraction_truncated_goal_ids == (
        "g1.1",
    )
    assert restored._turn_plan.execution.extraction_truncation_failures[
        "g1.1"
    ].code == CODE_EXTRACTION_TRUNCATED
    restored_handle = restored.result_handles.get(handle_id)
    assert restored_handle.producer_filter_applied is True
    assert restored_handle.views[0].item_indices == (0,)
    assert restored_handle.views[0].matched == 1

    output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(
            response="partial Casey evidence"
        )
    )
    turn = restored._build_turn_result(output)
    assert turn.turn_output.status is fastworkflow.TurnStatus.CENSORED
    assert turn.turn_output.failure_reason == CODE_EXTRACTION_TRUNCATED
    assert turn.metadata["runtime_failure"]["code"] == CODE_EXTRACTION_TRUNCATED
    ctx.close()
    restored.close()


def test_schema_eight_rejects_missing_truncation_type_before_mutation(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"schema8-malformed-{uuid.uuid4().hex[:8]}"
    source = _make_ctx(todo_workflow_path, channel_id)
    source._begin_turn("review Casey")
    source._turn_plan = _progress_plan()
    source._turn_agent_result = SimpleNamespace(
        exhausted=False,
        extraction_truncated=True,
        extraction_truncated_reason=CODE_EXTRACTION_TRUNCATED,
        extraction_truncated_goal_ids=("g1.1",),
        extraction_failure=extraction_truncated_failure(),
    )
    blob = source.serialize_state(channel_id=channel_id)
    del blob["turn"]["agent_result"]["extraction_failure"]

    restored = _make_ctx(todo_workflow_path, channel_id)
    with pytest.raises(IncompatibleSessionState, match="malformed turn accumulator"):
        restored.apply_serialized_state(blob)
    assert restored._turn_key is None
    assert restored._turn_plan is None
    assert restored._turn_presented_results == []
    source.close()
    restored.close()


def test_schema_eight_restores_active_leaf_binding_scope(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"schema8-scope-{uuid.uuid4().hex[:8]}"
    source = _make_ctx(todo_workflow_path, channel_id)
    source._begin_turn("review Casey")
    plan = _progress_plan()
    leaf = plan.node("g1.1")
    leaf.status = "needs-user"
    plan.execution = PlanExecutionMetadata(
        arm="b",
        packing_applied=False,
        schedule_sha256="sha256:suspended",
    )
    source._turn_plan = plan
    source._turn_active_leaf = leaf.goal_id
    source._turn_leaf_scope = PlanExecutionScope(
        arm=PlanExecutionArm.B,
        task_goal_id="g1",
        task_bindings=(
            (
                "subject",
                PlanExecutionBinding(
                    value="Casey",
                    source="utterance",
                    kind="exact_text",
                    resolver="control-alias@1",
                    source_spans=((7, 12),),
                ),
            ),
        ),
        producer_call_ids=("call-list-controls",),
        navigation_context="ControlsMonitor",
    )

    blob = source.serialize_state(channel_id=channel_id)
    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(json.loads(json.dumps(blob)))

    assert restored._turn_leaf_scope == source._turn_leaf_scope
    assert restored._turn_leaf_scope.navigation_context == "ControlsMonitor"
    binding = dict(restored._turn_leaf_scope.task_bindings)["subject"]
    assert binding.resolver == "control-alias@1"
    assert binding.source_spans == ((7, 12),)
    source.close()
    restored.close()


def test_plan_leaf_answers_survive_turn_restore(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"answers-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("do a composed task")
    ctx._turn_plan = _progress_plan()
    second_task_key = "review::subject=Riley::<none>"
    ctx._turn_plan.nodes = (
        *ctx._turn_plan.nodes,
        PlanNode(
            goal_id="g2",
            level="task",
            goal_text="Riley is reviewed.",
            visibility="public",
            task_key=second_task_key,
            status="done",
        ),
        PlanNode(
            goal_id="g2.1",
            parent_goal_id="g2",
            level="commands",
            goal_text="Review Riley.",
            executable_goal_text="Review Riley.",
            executable=True,
            status="done",
            command_call_ids=("call-review-riley",),
        ),
    )
    ctx._turn_plan.requested_public_task_keys = (
        *ctx._turn_plan.requested_public_task_keys,
        second_task_key,
    )
    ctx._turn_plan.compiled_public_task_keys = (
        *ctx._turn_plan.compiled_public_task_keys,
        second_task_key,
    )
    ctx._turn_plan_answers = [
        {"goal_id": "g1.1", "answer": "first result"},
        {"goal_id": "g2.1", "answer": "second result"},
    ]

    blob = ctx.serialize_state(channel_id=channel_id)
    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    assert restored._turn_plan_answers == ctx._turn_plan_answers
    assert "first result" in restored._compose_plan_answer("2 of 3 done")
    assert "2 of 3 done" in restored._compose_plan_answer("2 of 3 done")
    assert "not independent contract verification" in (
        restored._compose_plan_answer("2 of 3 done")
    )
    ctx.close()
    restored.close()


def test_plan_progress_checkpoint_is_a_durable_preterminal_turn(
    initialized_fastworkflow,
    todo_workflow_path,
    tmp_path,
):
    path = tmp_path / "plan-progress.sqlite3"
    sink = SQLiteTraceSink(str(path))
    channel_id = f"checkpoint-{uuid.uuid4().hex[:8]}"
    ctx = WorkflowExecutionContext(
        run_as_agent=False,
        session_key=channel_id,
        trace_sink=sink,
    )
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=channel_id,
    )
    ctx.bind_app_workflow(workflow)
    ctx.bind_observability_identity(channel_id, 1)
    ctx._begin_turn("review Casey")
    plan = _progress_plan()
    ctx._turn_plan = plan
    ctx._turn_plan_answers = [
        {"goal_id": "g1.1", "answer": "Casey review evidence"}
    ]
    ctx.append_turn_output(
        fastworkflow.CommandOutput(
            command_name="review",
            command_call_id="call-review-casey",
            command_response=fastworkflow.CommandResponse(
                response="reviewed",
            ),
        )
    )

    ctx._checkpoint_plan_progress(plan, plan.leaves[0], "done")
    assert sink.flush()
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status, record_json FROM turns WHERE turn_key=?",
            (ctx.current_turn_key,),
        ).fetchone()

    assert row is not None
    assert row[0] == fastworkflow.TurnStatus.IN_PROGRESS.value
    record = json.loads(row[1])
    assert record["plan"]["nodes"][1]["status"] == "done"
    assert record["metadata"]["plan_checkpoint"]["sequence"] == 1
    assert "Casey review evidence" in record["turn_output"]["answer"]
    assert ctx._turn_plan_last_checkpoint_stored is True

    timeout_output = ctx._provider_timeout_output(
        LMTimeoutError("offline timeout", model="offline-test")
    )
    terminal = ctx._build_turn_result(timeout_output)
    assert sink.flush()
    with sqlite3.connect(path) as connection:
        terminal_row = connection.execute(
            "SELECT status, record_json FROM turns WHERE turn_key=?",
            (ctx.current_turn_key,),
        ).fetchone()
    assert terminal_row is not None
    assert terminal_row[0] == fastworkflow.TurnStatus.PROVIDER_TIMEOUT.value
    terminal_record = json.loads(terminal_row[1])
    assert terminal_record["plan"]["nodes"][1]["status"] == "done"
    assert "Casey review evidence" in terminal_record["turn_output"]["answer"]
    assert (
        terminal.turn_output.status
        is fastworkflow.TurnStatus.PROVIDER_TIMEOUT
    )
    ctx.close()
    sink.close()


def test_plan_checkpoint_certification_survives_restore(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"checkpoint-restore-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("review Casey")
    ctx._turn_plan = _progress_plan()
    ctx._turn_plan_answers = [
        {"goal_id": "g1.1", "answer": "Casey review evidence"}
    ]
    ctx._turn_plan_outcome = "partial"
    ctx._turn_plan_checkpoint_count = 3
    ctx._turn_plan_last_checkpoint_stored = True
    ctx._turn_plan_last_checkpoint_leaf = "g1.1"

    blob = ctx.serialize_state(channel_id=channel_id)
    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    assert restored._turn_plan_outcome == "partial"
    assert restored._turn_plan_checkpoint_count == 3
    assert restored._turn_plan_last_checkpoint_stored is True
    assert restored._turn_plan_last_checkpoint_leaf == "g1.1"
    assert restored._turn_plan.leaves[0].status == "done"
    assert restored._turn_plan_answers == ctx._turn_plan_answers
    ctx.close()
    restored.close()


@pytest.mark.parametrize("schema_version", (5, 6, 7))
def test_plan_checkpoint_state_restores_by_declared_schema(
    initialized_fastworkflow,
    todo_workflow_path,
    schema_version,
):
    channel_id = f"schema-{schema_version}-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("review Casey")
    ctx._turn_plan = _progress_plan()
    ctx._turn_plan_answers = [
        {"goal_id": "g1.1", "answer": "Casey review evidence"}
    ]
    ctx._turn_plan_outcome = "partial"
    ctx._turn_plan_checkpoint_count = 1
    ctx._turn_plan_last_checkpoint_stored = True
    ctx._turn_plan_last_checkpoint_leaf = "g1.1"
    blob = copy.deepcopy(ctx.serialize_state(channel_id=channel_id))
    blob["schema_version"] = schema_version
    if schema_version < 7:
        for field in (
            "plan_outcome",
            "plan_checkpoint_count",
            "plan_last_checkpoint_stored",
            "plan_last_checkpoint_leaf",
        ):
            blob["turn"].pop(field)
    if schema_version < 6:
        blob["turn"].pop("plan_answers")

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    assert restored._turn_plan.leaves[0].status == "done"
    if schema_version >= 6:
        assert restored._turn_plan_answers == ctx._turn_plan_answers
    else:
        assert restored._turn_plan_answers == []
    if schema_version >= 7:
        assert restored._turn_plan_outcome == "partial"
        assert restored._turn_plan_checkpoint_count == 1
        assert restored._turn_plan_last_checkpoint_stored is True
    else:
        assert restored._turn_plan_outcome is None
        assert restored._turn_plan_checkpoint_count == 0
        assert restored._turn_plan_last_checkpoint_stored is None
    ctx.close()
    restored.close()


def test_malformed_schema_seven_plan_state_applies_nothing(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"malformed-plan-{uuid.uuid4().hex[:8]}"
    source = _make_ctx(todo_workflow_path, channel_id)
    source._begin_turn("review Casey")
    source._turn_plan = _progress_plan()
    source._turn_plan_answers = [
        {"goal_id": "missing-leaf", "answer": "unsupported claim"}
    ]
    blob = source.serialize_state(channel_id=channel_id)

    restored = _make_ctx(todo_workflow_path, channel_id)
    with pytest.raises(
        Exception,
        match="matching leaf evidence",
    ):
        restored.apply_serialized_state(blob)

    assert restored._turn_key is None
    assert restored._turn_plan is None
    assert restored._turn_plan_answers == []
    source.close()
    restored.close()


def test_delayed_bindings_use_only_their_successful_declared_producer(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"capture-provenance-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("inspect two account lists")
    plan = PlanRecord(
        plan_id="capture-provenance",
        mode="enforce",
        nodes=(
            PlanNode(
                goal_id="task-a",
                level="task",
                goal_text="Inspect A.",
                visibility="public",
            ),
            PlanNode(
                goal_id="producer-a",
                parent_goal_id="task-a",
                level="commands",
                goal_text="List A accounts.",
                executable_goal_text="List A accounts.",
                executable=True,
                status="done",
                command_call_ids=("call-a",),
            ),
            PlanNode(
                goal_id="consumer-a",
                parent_goal_id="task-a",
                level="commands",
                goal_text="Inspect {account_uid}.",
                status="needs-user",
                prerequisites=("producer-a",),
                bindings={
                    "account_uid": Binding(
                        value=None,
                        source="needs-user",
                    )
                },
            ),
            PlanNode(
                goal_id="task-b",
                level="task",
                goal_text="Inspect B.",
                visibility="public",
            ),
            PlanNode(
                goal_id="producer-b",
                parent_goal_id="task-b",
                level="commands",
                goal_text="List B accounts.",
                executable_goal_text="List B accounts.",
                executable=True,
            ),
            PlanNode(
                goal_id="consumer-b",
                parent_goal_id="task-b",
                level="commands",
                goal_text="Inspect {account_uid}.",
                status="needs-user",
                prerequisites=("producer-b",),
                bindings={
                    "account_uid": Binding(
                        value=None,
                        source="needs-user",
                    )
                },
            ),
        ),
    )
    plan._catalog = SimpleNamespace(get=lambda _name: None)
    ctx._turn_plan = plan
    ctx._turn_outputs = [
        fastworkflow.CommandOutput(
            command_name="failed-list",
            command_call_id="call-a",
            command_response=fastworkflow.CommandResponse(
                response="failed",
                success=False,
                artifacts={"account_uids": ["failed-account"]},
            ),
        ),
        fastworkflow.CommandOutput(
            command_name="list-a",
            command_call_id="call-a",
            command_response=fastworkflow.CommandResponse(
                response="A",
                artifacts={"account_uids": ["account-a"]},
            ),
        ),
        fastworkflow.CommandOutput(
            command_name="unrelated",
            command_call_id="call-unrelated",
            command_response=fastworkflow.CommandResponse(
                response="unrelated",
                artifacts={"account_uids": ["wrong-account"]},
            ),
        ),
    ]

    assert ctx._resolve_delayed_plan_bindings(plan) is True
    assert plan.node("consumer-a").bindings["account_uid"].value == "account-a"
    assert plan.node("consumer-b").bindings["account_uid"].value is None

    producer_b = plan.node("producer-b")
    producer_b.command_call_ids = ("call-b",)
    producer_b.status = "done"
    ctx._turn_outputs.append(
        fastworkflow.CommandOutput(
            command_name="list-b",
            command_call_id="call-b",
            command_response=fastworkflow.CommandResponse(
                response="B",
                artifacts={"account_uids": ["account-b"]},
            ),
        )
    )
    assert ctx._resolve_delayed_plan_bindings(plan) is True
    assert plan.node("consumer-b").bindings["account_uid"].value == "account-b"
    ctx.close()


def test_provider_timeout_after_progress_keeps_answers_and_distinct_status(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"timeout-progress-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("review Casey")
    ctx._turn_plan = _progress_plan()
    ctx._turn_plan_answers = [
        {"goal_id": "g1.1", "answer": "Casey review evidence"}
    ]
    ctx._turn_plan_checkpoint_count = 1
    ctx._turn_plan_last_checkpoint_stored = True
    ctx._turn_plan_last_checkpoint_leaf = "g1.1"

    output = ctx._provider_timeout_output(
        LMTimeoutError("offline timeout", model="offline-test")
    )
    result = ctx._build_turn_result(output)

    assert (
        result.turn_output.status
        is fastworkflow.TurnStatus.PROVIDER_TIMEOUT
    )
    assert result.turn_output.failure_reason == "provider-timeout"
    assert "Casey review evidence" in result.turn_output.answer
    assert result.metadata["plan_runtime"]["done_leaf_count"] == 1
    assert result.metadata["plan_runtime"]["last_checkpoint_stored"] is True
    assert result.metadata["runtime_failure"]["code"] == "provider-timeout"
    ctx.close()


def test_provider_timeout_before_first_progress_has_no_false_durable_work(
    initialized_fastworkflow,
    todo_workflow_path,
):
    channel_id = f"timeout-empty-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("start work")

    output = ctx._provider_timeout_output(
        LMTimeoutError("offline timeout", model="offline-test")
    )
    result = ctx._build_turn_result(output)

    assert (
        result.turn_output.status
        is fastworkflow.TurnStatus.PROVIDER_TIMEOUT
    )
    assert result.plan is None
    assert result.metadata["runtime_failure"]["code"] == "provider-timeout"
    assert "Prior plan progress remains recorded" not in result.turn_output.answer
    ctx.close()


def test_non_stress_off_mode_keeps_default_budget_and_disables_censor(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    monkeypatch.delenv("FW_PLAN_STRESS_MODE", raising=False)
    channel_id = f"off-parity-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("ordinary turn")

    assert ctx.turn_budget is not None
    assert ctx.turn_budget.enforce_iteration_limit is True
    assert ctx._turn_safety_envelope.enabled is False

    ctx._turn_agent_result = SimpleNamespace(exhausted=True)
    output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response="partial")
    )
    result = ctx._build_turn_result(output)
    assert result.turn_output.status is fastworkflow.TurnStatus.FAILED
    assert result.turn_output.failure_reason == "max_iters_exhausted"
    ctx.close()


def test_non_stress_turn_ignores_stale_invalid_safety_limit(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    monkeypatch.setenv("FW_PLAN_STRESS_MODE", "0")
    monkeypatch.setenv("FW_PLAN_WALL_TIME_LIMIT_SECONDS", "not-an-integer")
    channel_id = f"off-stale-safety-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)

    ctx._begin_turn("ordinary turn")

    assert ctx.turn_budget.enforce_iteration_limit is True
    assert ctx._turn_safety_envelope.enabled is False
    ctx.close()


def test_stress_censor_is_metadata_not_turn_failure(
    initialized_fastworkflow,
    todo_workflow_path,
    monkeypatch,
):
    monkeypatch.setenv("FW_PLAN_STRESS_MODE", "1")
    channel_id = f"censor-{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("do a long task")
    ctx._turn_safety_envelope.censor("wall-time-cutoff")
    ctx._turn_agent_result = SimpleNamespace(
        exhausted=False,
        censored=True,
        censored_reason="wall-time-cutoff",
    )
    output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(
            response="censored",
        )
    )

    result = ctx._build_turn_result(output)

    assert result.turn_output.status is fastworkflow.TurnStatus.CENSORED
    assert result.turn_output.failure_reason == "wall-time-cutoff"
    assert result.metadata["exp028_stress"]["censored"] is True
    assert (
        result.metadata["exp028_stress"]["censored_reason"]
        == "wall-time-cutoff"
    )
    ctx.close()


def test_restored_agent_result_carries_exhaustion(
    initialized_fastworkflow, todo_workflow_path
):
    """The finalize path reads .exhausted off the agent result to set FAILED."""
    channel_id = f"exh_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("do something long")

    class _Exhausted:
        exhausted = True

    ctx._turn_agent_result = _Exhausted()
    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["turn"]["agent_result"] == {"exhausted": True}

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)
    assert restored._turn_agent_result is not None
    assert restored._turn_agent_result.exhausted is True
    restored.close()


def test_composite_packing_and_execution_metadata_survive_turn_restore(
    initialized_fastworkflow, todo_workflow_path
):
    channel_id = f"plan_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("review Casey and Riley")
    task_keys = (
        "review::subject=Casey::<none>",
        "review::subject=Riley::<none>",
    )
    group = CompositeGroup(
        group_id="pack-1",
        composite_skill="review-packet",
        member_goal_ids=("g1", "g2"),
        member_task_keys=task_keys,
        shared_bindings={"subjects": ("Casey", "Riley")},
        orchestration_edges=(
            PlanEdge(
                from_goal_id="g1",
                to_goal_id="g2",
                provenance="composite-pack",
            ),
        ),
        signature_sha256="sha256:group",
    )
    ctx._turn_plan = PlanRecord(
        plan_id="restore-plan",
        mode="enforce",
        nodes=(
            PlanNode(
                goal_id="g1",
                level="task",
                skill="review",
                goal_text="Casey is reviewed.",
                visibility="public",
                task_key=task_keys[0],
            ),
            PlanNode(
                goal_id="g2",
                level="task",
                skill="review",
                goal_text="Riley is reviewed.",
                visibility="public",
                task_key=task_keys[1],
            ),
        ),
        requested_public_task_keys=task_keys,
        compiled_public_task_keys=task_keys,
        composite_groups=(group,),
        packing=CompositePackingMetrics(
            candidate_count=1,
            selected_root_group_count=1,
            packed_task_count=2,
            orchestration_edge_count=1,
            shared_binding_count=1,
            packing_sha256="sha256:packing",
        ),
        execution=PlanExecutionMetadata(
            arm="c",
            packing_applied=True,
            schedule_sha256="sha256:schedule",
            composite_group_ids=("pack-1",),
            composite_groups_applied=1,
            grouped_task_count=2,
            public_task_keys=task_keys,
        ),
    )
    ctx._turn_plan_frontier = []
    ctx._turn_active_leaf = None
    original_plan = ctx._turn_plan

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()
    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    assert restored._turn_plan == original_plan
    assert restored._turn_plan.composite_groups[0].shared_bindings == {
        "subjects": ("Casey", "Riley")
    }
    assert restored._turn_plan.execution is not None
    assert restored._turn_plan.execution.packing_applied is True
    assert restored._turn_plan_frontier == []
    assert restored._turn_active_leaf is None
    restored.close()


def test_missing_cme_workflow_reads_as_nothing_in_flight(
    initialized_fastworkflow, todo_workflow_path
):
    """has_open_command() now runs after every turn, not only suspensions.

    close() treats a missing cme_workflow as reachable, so raising here would
    turn a tolerated state into a crash in the post-turn persist path.
    """
    channel_id = f"nocme_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx.close()
    ctx._cme_workflow = None

    assert ctx.has_open_command() is False
    assert ctx._is_extracting_parameters() is False
