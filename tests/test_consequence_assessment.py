"""fw.command.execute and fw.agent.tool_call spans carry consequence attributes.

Architecture §6.6.1: every executed command records a ``ConsequenceAssessment``
on the span that covers it. ``unknown`` must not silently read as ``read_only`` —
an undeclared workflow grades every command at high consequence.

Fixtures use the real ``todo_list_workflow`` for the direct-action path (which
opens only ``fw.agent.tool_call``) and the trained ``hello_world`` workflow for
``fw.command.execute`` via ``invoke_command``. No mocks at the execution boundary.
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.runtime_manifest import (
    CommandDeclaration,
    EffectContract,
    RuntimeManifest,
    clear_runtime_metadata,
    merge_and_gate,
    register_runtime_metadata,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

HELLO_WORLD = str(
    Path(__file__).parent.parent / "fastworkflow" / "examples" / "hello_world"
)
LIST_COMMAND = "TodoListManager/list_todo_lists"
CREATE_COMMAND = "TodoListManager/create_todo_list"


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


@pytest.fixture(scope="module")
def hello_initialized():
    if not Path(HELLO_WORLD, "___command_info").is_dir():
        pytest.skip("hello_world is not trained on this machine")
    env = dotenv_values("fastworkflow/examples/fastworkflow.env")
    fastworkflow.init(dict(env))
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        return True

    def record_conversation_label(self, *args) -> None:
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def todo_ctx(initialized_fastworkflow, todo_workflow_path, tmp_path):
    sink = RecordingTraceSink()
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"conseq-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    yield ctx, sink
    with suppress(Exception):
        ctx.close()


def _action(command_name: str, **parameters) -> fastworkflow.Action:
    return fastworkflow.Action(
        command_name=command_name, command="do it", parameters=parameters
    )


def _consequence(span: tracing.Span) -> dict:
    assert tracing.ATTR_CONSEQUENCE in span.attributes
    consequence = span.attributes[tracing.ATTR_CONSEQUENCE]
    assert isinstance(consequence, dict)
    return consequence


def test_direct_action_tool_call_span_carries_consequence(todo_ctx):
    ctx, sink = todo_ctx
    ctx.process_action_turn(_action(LIST_COMMAND))

    consequence = _consequence(sink.named(tracing.SPAN_AGENT_TOOL_CALL)[-1])
    assert consequence["effect_kind"] == "unknown"
    assert consequence["consequence_class"] == "high"
    assert consequence["assessor_version"] == "default/1"
    assert consequence["reversibility"] == "unknown"
    assert consequence["blast_radius"] == "unknown"


def test_declared_read_only_reaches_the_tool_call_span(
    todo_ctx, todo_workflow_path
):
    ctx, sink = todo_ctx
    manifest = RuntimeManifest(
        schema_version=1,
        manifest_version="1.0.0",
        commands={
            LIST_COMMAND: CommandDeclaration(
                effect=EffectContract(kind="read_only")
            )
        },
    )
    register_runtime_metadata(
        todo_workflow_path, merge_and_gate(manifest, deployment_features={}, env={})
    )
    try:
        ctx.process_action_turn(_action(LIST_COMMAND))
        consequence = _consequence(sink.named(tracing.SPAN_AGENT_TOOL_CALL)[-1])
        assert consequence["effect_kind"] == "read_only"
    finally:
        clear_runtime_metadata()


def test_command_execute_span_carries_consequence(hello_initialized):
    sink = RecordingTraceSink()
    wf = fastworkflow.Workflow.create(
        HELLO_WORLD,
        workflow_id_str=f"conseq-exec-{uuid.uuid4().hex}",
        workflow_context={"run_as_agent": True},
    )
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    ctx.bind_app_workflow(wf)
    ctx._begin_turn("consequence on execute span")
    ctx.push_active_workflow(wf)

    CommandExecutor.invoke_command(
        ctx,
        "add_two_numbers <first_num>2</first_num><second_num>2</second_num>",
    )

    execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[-1]
    consequence = _consequence(execute)
    assert consequence["effect_kind"] == "unknown"
    assert consequence["consequence_class"] == "high"

    ctx.pop_active_workflow()


def test_prose_path_tool_call_and_execute_share_consequence_shape(
    initialized_fastworkflow, todo_workflow_path, tmp_path, monkeypatch
):
    """Both spans on the invoke_command path carry the same assessment fields."""
    sink = RecordingTraceSink()
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"conseq-prose-{uuid.uuid4().hex}"
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))

    real_perform_action = CommandExecutor.perform_action

    def cme_hop(cls, wf, action):
        command_output = real_perform_action(
            workflow, _action(LIST_COMMAND)
        )
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = LIST_COMMAND
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(cme_hop))

    try:
        ctx.process_turn("list my todo lists")

        tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[-1]
        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[-1]
        tool_consequence = _consequence(tool_call)
        execute_consequence = _consequence(execute)

        assert set(tool_consequence) == set(execute_consequence)
        assert tool_consequence["effect_kind"] == execute_consequence["effect_kind"]
    finally:
        with suppress(Exception):
            ctx.close()


def test_navigation_command_consequence_still_unknown_without_declaration(todo_ctx):
    """create_todo_list moves context but carries no manifest entry here."""
    ctx, sink = todo_ctx
    ctx.process_action_turn(_action(CREATE_COMMAND, description="groceries"))

    consequence = _consequence(sink.named(tracing.SPAN_AGENT_TOOL_CALL)[-1])
    assert consequence["effect_kind"] == "unknown"
    assert consequence["consequence_class"] == "high"
