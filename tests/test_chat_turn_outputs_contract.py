"""What a real turn puts in ``command_outputs`` and the schema listing, pinned for the chat client (fix-weih.19).

``tests/fixtures/chat_turn_outputs.json`` records, for real "/" turns through ``WorkflowExecutionContext.process_turn``
on the todo workflow (a leaf name, ``get_properties``, owned by two contexts) and on a copy of the library example
(a global command, planning commands, a parent-context walk), each output's ``command_name``, ``context``, output
keys and the command named by ``parameter_request``, together with the schema listing's keys. This test asserts the
fixture equals what the code produces now; ``tests/chatbot_turn_outputs_dom.cjs`` loads the same fixture and asserts
the client resolves every one of those outputs to its schema. Regenerate with
``python -m tests.test_chat_turn_outputs_contract`` (it prints the JSON; nothing writes at test time).

The parameter-extraction model is the one seam: it binds nothing, so a "/" command without bound values fails
validation and the wildcard answers with a ``parameter_request``.
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.mcp_server import command_schema_listing
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

TESTS = Path(__file__).parent
FIXTURE = TESTS / "fixtures" / "chat_turn_outputs.json"
_IGNORE = shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__", "___command_info")


def _properties(schema) -> list[str]:
    return sorted((schema or {}).get("properties", {}))


def _facts(out) -> list[dict]:
    rows = []
    for o in out.command_outputs:
        output = o.command_response.output
        artifacts = o.command_response.artifacts or {}
        row = {
            "command_name": o.command_name,
            "context": o.context,
            "output_keys": sorted(output.model_dump()) if output is not None else [],
        }
        if (o.ran_in_context_name is not None):
            row["ran_in_context_name"] = o.ran_in_context_name
        if "parameter_request" in artifacts:
            row["parameter_request_command"] = artifacts["parameter_request"]["command"]
        if "approval" in artifacts:
            row["approval_command"] = artifacts["approval"]["command"]
        rows.append(row)
    return rows


def _make(tmp_path, source: Path, name: str):
    folder = tmp_path / name
    shutil.copytree(source, folder, ignore=_IGNORE)
    RoutingRegistry.get_definition(str(folder), load_cached=False)
    ctx = WorkflowExecutionContext(run_as_agent=False)
    wf = fastworkflow.Workflow.create(str(folder), workflow_id_str=f"co_{uuid.uuid4().hex[:8]}")
    ctx.bind_app_workflow(wf)
    return folder, ctx, wf


def _schemas(folder) -> dict:
    return {key: {"kind": entry["kind"], "output_properties": _properties(entry["output"])}
            for key, entry in sorted(command_schema_listing(str(folder)).items())}


def produce(tmp_path) -> dict:
    """The fixture's content, from real turns."""
    original = InputForParamExtraction.extract_parameters
    InputForParamExtraction.extract_parameters = lambda _self, cls, *a, **k: cls.model_construct()
    try:
        cases = []

        def run(label, ctx, message):
            cases.append({"case": label, "message": message, "outputs": _facts(ctx.process_turn(message))})

        folder, ctx, wf = _make(tmp_path, TESTS / "todo_list_workflow", "todo")
        todo_schemas = _schemas(folder)
        run("todo: startup is a global command", ctx, "/startup")
        run("todo: a command of the current root context", ctx, "/create_todo_list description=Chores")
        run("todo: a leaf owned by TodoList, TodoList current", ctx, "/get_properties")
        run("todo: a parent context's command run from TodoList", ctx, "/list_todo_lists")
        run("todo: failed validation names the qualified command", ctx, "/add_child_todoitem")
        item = wf.record.add_child_todoitem("milk", "Bob")
        wf.record = item
        run("todo: the same leaf owned by TodoItem, TodoItem current", ctx, "/get_properties")
        run("todo: a grandparent's command run from TodoItem", ctx, "/list_todo_lists")

        folder, ctx, wf = _make(tmp_path, TESTS.parent / "fastworkflow" / "examples" / "library", "library")
        library_schemas = _schemas(folder)
        run("library: startup", ctx, "/startup")
        run("library: a global command with a root context current", ctx, "/library_hours")
        run("library: an execution command of the root context", ctx, "/count_overdue")
        run("library: a planning command, parameters missing", ctx, "/count_books")
        run("library: an execution command failing validation", ctx, "/find_book")
        return {"workflows": {"todo": {"schemas": todo_schemas}, "library": {"schemas": library_schemas}},
                "cases": cases}
    finally:
        InputForParamExtraction.extract_parameters = original


@pytest.fixture(autouse=True)
def _framework(initialized_fastworkflow, tmp_path):  # noqa: F811
    yield


def test_the_checked_in_fixture_is_what_the_real_turns_produce(tmp_path):
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == produce(tmp_path)


def test_every_recorded_command_is_a_listed_schema_key_or_a_framework_entry():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    framework = {"wildcard", "ask_user"}
    for case in data["cases"]:
        keys = data["workflows"][case["case"].split(":")[0]]["schemas"]
        for row in case["outputs"]:
            assert row["command_name"] in keys or row["command_name"] in framework, row
            if "parameter_request_command" in row:
                assert row["parameter_request_command"] in keys, row


if __name__ == "__main__":  # regenerate: python -m tests.test_chat_turn_outputs_contract
    import tempfile
    fastworkflow.init({})
    RoutingRegistry.clear_registry()
    WorkflowExecutionContext._agent_dspy_context = lambda self: (None, None)
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(produce(Path(d)), indent=2, sort_keys=False))
