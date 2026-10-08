"""Plain-text planner: signature choice and span attributes."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from dotenv import dotenv_values
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import tracing
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.workflow_agent import build_query_with_next_steps, initialize_workflow_tool_agent
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

TODO_WORKFLOW = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())
COMMANDS_TITLE = "Available execute_workflow_query tool commands"
STRUCTURED_MARK = "Return the plan as structured steps"
TEXT_MARK = "[[ ## next_steps ## ]]"


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> None:
        pass

    def record_conversation_label(self, *args) -> None:
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [s for s in self.spans if s.name == name]


@pytest.fixture
def workflow_env(tmp_path, monkeypatch):
    """Load settings from a real fastworkflow.env / passwords file pair, process env cleared."""
    RoutingRegistry.clear_registry()

    def load(**settings: str) -> None:
        env_file = tmp_path / "fastworkflow.env"
        passwords_file = tmp_path / "fastworkflow.passwords.env"
        env_file.write_text("".join(f"{name}={value}\n" for name, value in settings.items()))
        passwords_file.write_text("")
        fastworkflow.init(env_vars={**dotenv_values(env_file), **dotenv_values(passwords_file)})

    yield load
    RoutingRegistry.clear_registry()


def _session(sink):
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    wf = fastworkflow.Workflow.create(TODO_WORKFLOW, workflow_id_str=f"planner-{uuid.uuid4().hex}")
    ctx.bind_app_workflow(wf)
    ctx._workflow_tool_agent = initialize_workflow_tool_agent(ctx)
    return ctx, wf


def _plan(ctx, wf, lm, **kwargs):
    ctx.push_active_workflow(wf)
    try:
        return build_query_with_next_steps("show my todo items", ctx, planner_lm=lm, **kwargs)
    finally:
        ctx.pop_active_workflow()


def _prompt(lm, index: int) -> str:
    return "\n".join(str(m.get("content", "")) for m in lm.history[index]["messages"])


def _text_answer():
    return {"reasoning": "r", "next_steps": "1. Show all todo items"}


def test_the_text_planner_runs_and_appends_next_steps(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer()])
    result = _plan(ctx, wf, lm)

    assert len(lm.history) == 1
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    assert COMMANDS_TITLE in _prompt(lm, 0)
    assert "Show all todo items" in result
    assert "Execute these next steps:" in result
    plan, = sink.named(tracing.SPAN_PLANNER_PLAN)
    assert (plan.attributes["plan_source"], plan.attributes["subjects"]) == ("text", [])


def test_a_replan_uses_the_text_planner(workflow_env):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)

    ctx._begin_turn("show my todo items")
    lm = DummyLM([_text_answer(), _text_answer()])
    _plan(ctx, wf, lm, with_agent_inputs_and_trajectory=True, trace_trigger="ask_user_response")

    assert len(lm.history) == 1
    assert TEXT_MARK in _prompt(lm, 0) and STRUCTURED_MARK not in _prompt(lm, 0)
    replan, = sink.named(tracing.SPAN_PLANNER_REPLAN)
    assert replan.attributes["plan_source"] == "text"
    assert not sink.named(tracing.SPAN_PLANNER_PLAN)
