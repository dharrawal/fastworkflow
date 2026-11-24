"""Startup stops when neither the agent model nor Jev is set (fix-rn9p): ``fastworkflow run`` exits non-zero and the
server refuses to start. The entry points only; contexts that tests build never ask for the check."""
from __future__ import annotations

import argparse
import asyncio

import pytest

import fastworkflow
from fastworkflow import no_agent_mode
from tests.test_manager_shutdown_matrix import (  # noqa: F401 - the server fixtures
    app_module, hello_world_workflow_path, isolated_env_file)


@pytest.fixture
def bare_env(tmp_path, monkeypatch):
    """Env files that set neither LLM_AGENT nor JEV_API_KEY, and none in the process."""
    for name in ("LLM_AGENT", "JEV_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / "bare.env"
    env.write_text(f"FASTWORKFLOW_STATE_ROOT={tmp_path / 'state'}\n")
    passwords = tmp_path / "bare.passwords.env"
    passwords.write_text("")
    return str(env), str(passwords)


def test_the_server_refuses_to_start_with_no_agent_model_and_no_jev_key(app_module, bare_env, caplog):
    app_module.ARGS.env_file_path, app_module.ARGS.passwords_file_path = bare_env

    async def start():
        async with app_module.lifespan(app_module.app):
            pass

    with pytest.raises(RuntimeError, match="LLM_AGENT.*JEV_API_KEY"):
        asyncio.run(start())
    assert no_agent_mode.STARTUP_PROBLEM in caplog.text


def test_run_exits_non_zero_after_printing_the_problem(hello_world_workflow_path, bare_env, capsys):
    from fastworkflow.run.__main__ import run_main

    env, passwords = bare_env
    args = argparse.Namespace(
        workflow_path=hello_world_workflow_path, env_file_path=env, passwords_file_path=passwords,
        startup_command=None, startup_action=None, context_file_path=None, keep_alive=False,
        project_folderpath=None, assistant=False, generate_insights=False)
    with pytest.raises(SystemExit) as stopped:
        run_main(args)

    assert stopped.value.code == 1
    assert "Startup problem" in capsys.readouterr().out
    fastworkflow.init({})
