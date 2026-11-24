"""A workflow folder's ``startup_action.json`` is the default startup of every entry point (bd fix-qjne.10).

One reader (``fastworkflow.workflow_startup_action``); ``fastworkflow run`` and the FastAPI server (which the chat
page's launcher starts) use it when no process flag is given; a flag always wins; a per-session field of
``/initialize`` keeps its precedence over the process-level value.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fastworkflow

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "fastworkflow" / "examples" / "library"
TODO = ROOT / "tests" / "todo_list_workflow"


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "___convo_info", "___workflow_contexts"))
    return dst


def test_the_helper_reads_the_workflow_folders_file(tmp_path):
    (tmp_path / "startup_action.json").write_text(json.dumps({"command_name": "A/b", "parameters": {"x": 1}}))
    action = fastworkflow.workflow_startup_action(str(tmp_path))
    assert action.command_name == "A/b" and action.parameters == {"x": 1}


def test_the_helper_gives_none_without_the_file(tmp_path):
    assert fastworkflow.workflow_startup_action(str(tmp_path)) is None


@pytest.mark.parametrize("text", ["{not json", '{"parameters": {}}', "[1]"])
def test_an_invalid_file_is_an_error_naming_the_file(tmp_path, text):
    (tmp_path / "startup_action.json").write_text(text)
    with pytest.raises(ValueError) as excinfo:
        fastworkflow.workflow_startup_action(str(tmp_path))
    assert str(tmp_path / "startup_action.json") in str(excinfo.value)


def test_the_shipped_library_example_has_a_startup_action():
    assert fastworkflow.workflow_startup_action(str(LIBRARY)).command_name == "startup"


def test_resolve_explicit_flags_win_and_the_file_is_the_default(tmp_path):
    (tmp_path / "startup_action.json").write_text(json.dumps({"command_name": "A/b"}))
    folder = str(tmp_path)
    assert fastworkflow.resolve_startup_action(folder, "", "").command_name == "A/b"
    assert fastworkflow.resolve_startup_action(folder, None, None).command_name == "A/b"
    assert fastworkflow.resolve_startup_action(folder, "do it", "") is None
    assert fastworkflow.resolve_startup_action(folder, "", '{"command_name": "C/d"}').command_name == "C/d"
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"command_name": "E/f"}))
    assert fastworkflow.resolve_startup_action(folder, "", str(other)).command_name == "E/f"


def test_a_workflow_with_no_file_has_no_default(tmp_path):
    assert fastworkflow.resolve_startup_action(str(tmp_path), "", "") is None


def test_the_code_run_reads_the_file_through_the_same_helper():
    import inspect
    from fastworkflow import planning_code_run
    assert "workflow_startup_action" in inspect.getsource(planning_code_run._open_session)


# ---- the server (the chat page's launcher starts this same server) ----

@pytest.fixture
def server(env_files, tmp_path, request):
    """The FastAPI app on a temp copy of the workflow named by ``request.param``, no startup flag."""
    env_file, passwords_file = env_files
    workflow = _copy(request.param, tmp_path)
    sys.argv = ["pytest", "--workflow_path", str(workflow), "--env_file_path", env_file,
                "--passwords_file_path", passwords_file]
    import fastworkflow.run_fastapi_mcp.__main__ as main
    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env
    previous = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts")
    try:
        yield main
    finally:
        restore_fastapi_env(previous)


@pytest.mark.parametrize("server", [TODO], indirect=True)
def test_a_session_with_no_startup_field_runs_the_workflows_startup_action(server):
    channel = f"default_{uuid.uuid4().hex[:8]}"
    with TestClient(server.app) as client:
        resp = client.post("/initialize", json={"channel_id": channel})
        assert resp.status_code in (200, 202)
        assert resp.json().get("startup_turn_key"), resp.json()


@pytest.mark.parametrize("server", [LIBRARY], indirect=True)
def test_the_library_example_session_has_a_root_object_without_a_flag(server):
    """The chat page's server is started with no startup flag; its /initialize gives the library a root object."""
    channel = f"library_{uuid.uuid4().hex[:8]}"
    with TestClient(server.app) as client:
        resp = client.post("/initialize", json={"channel_id": channel})
        assert resp.status_code in (200, 202)
        assert resp.json().get("startup_turn_key"), resp.json()


def test_the_chat_launchers_server_command_names_no_startup_and_the_server_still_finds_the_file(monkeypatch):
    """The launcher builds the server's command line with no startup flag; the server's own default supplies it."""
    from fastworkflow.run_chatbot import launcher

    plan = launcher.plan_server_spawn(
        workflow_path=str(LIBRARY), env_file_path="e.env", passwords_file_path="p.env", server_port=8123,
        chatbot_origin="http://127.0.0.1:39131",
        allow_unsigned_jwt=True, missing_packages=[])
    assert plan.ok
    assert not any(part.startswith("--startup") for part in plan.cmd)
    monkeypatch.setattr(sys, "argv", ["server", *plan.cmd[3:]])
    import fastworkflow.run_fastapi_mcp.__main__ as main
    args = importlib.reload(main).ARGS
    assert not args.startup_action and not args.startup_command
    action = fastworkflow.resolve_startup_action(args.workflow_path, args.startup_command, args.startup_action)
    assert action is not None and action.command_name == "startup"


@pytest.mark.parametrize("server", [TODO], indirect=True)
def test_a_per_session_startup_command_beats_the_workflows_default_file(server, tmp_path):
    """The file is only the default: a request that carries its own startup_command runs that, not the file."""
    import time
    marker = "/startup"
    channel = f"cmdwins_{uuid.uuid4().hex[:8]}"
    ran: list[str] = []
    real = server._build_startup_work_fn

    def spy(ctx, command, action):
        ran.append(f"action={action.command_name}" if action is not None else f"command={command}")
        return real(ctx, command, action)

    server._build_startup_work_fn = spy
    try:
        with TestClient(server.app) as client:
            resp = client.post("/initialize", json={"channel_id": channel, "user_id": "u", "startup_command": marker})
            assert resp.status_code in (200, 202), resp.text
            time.sleep(0.5)
    finally:
        server._build_startup_work_fn = real
    assert ran == ["command=/startup"], ran
