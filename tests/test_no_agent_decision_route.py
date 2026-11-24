"""A message that begins with "/" while a no-agent decision question is pending, through the real HTTP routes (fix-rn9p;
changed by fix-fpyb.7: a "/" message is the reply on BOTH routes now).

``POST /invoke_agent`` (and the MCP ``invoke_agent`` tool) strips every leading "/" before the context sees the message
(its documented rule since 2.17: the agent endpoint is never the deterministic one), so while a ``phase_step`` question
is pending EVERY message on it is the reply, a "/" one included. ``POST /invoke_assistant`` (the MCP ``invoke_assistant``
tool) keeps the "/", and the context removes it itself: the message is the reply there too (it was a separate
deterministic command before fix-fpyb.7).

The routes are the real ones, driven by ``TestClient`` on a real server. The one seam: the server channel's
``process_turn`` hands the message the route passed to the library test context that holds the pending question (a real
phase runner paused at a real decision step), so the decision reader is the real one.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.turn import TurnStatus
from tests.test_fastapi_turn_output_contract import (  # noqa: F401 - fixtures and helper
    _channel, action_app_module)
from tests.test_no_agent_decisions import (  # noqa: F401 - fixtures and helpers
    BOOK_BY_TITLE, BOOK_LOANS, Conditional, _BOOK_BY_TITLE_SOURCE, _build_planning_command, _find_runs,
    _nested_state, _no_agent, _pending, _plan, library_env)
from tests.test_turn_result_capture import initialized_fastworkflow, no_review_gate  # noqa: F401 - shared fixtures


@pytest.fixture
def server(action_app_module):
    return action_app_module


def _no_agent_again(monkeypatch):
    """No agent model and no remote decision model again: the server's startup (inside ``TestClient``) loads the env files,
    which set ``LLM_AGENT`` and the Jev key (a reply to a decision step is read by the remote model when there is one:
    ``no_planner_key`` is this test's premise, as in ``tests/test_no_agent_mode.py``)."""
    from fastworkflow import workflow_agent

    for name in ("LLM_AGENT", "JEV_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fastworkflow, "_env_vars",
                        {name: value for name, value in fastworkflow._env_vars.items()
                         if name not in ("LLM_AGENT", "JEV_API_KEY")})
    getattr(workflow_agent._jev_lm, "cache_clear", lambda: None)()


def _pause_and_route(app_module, client, library_env, monkeypatch):
    """A decision question pending on *library_env*'s context, and a server channel whose turns go to that context.
    Returns ``(headers, seen)``: the auth headers and the list of messages the routes passed to ``process_turn``."""
    _no_agent_again(monkeypatch)
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    first = library_env.ctx.process_turn("look up the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    assert _pending(first)[1]["question_kind"] == "phase_step"

    channel_id = _channel("decision")
    init = client.post("/initialize", json={"channel_id": channel_id})
    assert init.status_code == 200
    headers = {"Authorization": f"Bearer {init.json()['access_token']}"}
    runtime = asyncio.run(app_module.session_manager.get_session(channel_id))
    seen: list[str] = []

    def to_the_paused_context(message, **kwargs):
        seen.append(message)
        return library_env.ctx.process_turn(message, **kwargs)

    runtime.execution_context.process_turn = to_the_paused_context
    return headers, seen


def _deterministic_command_handler(monkeypatch, handled: list[str]):
    def deterministic(cls, session, command):
        handled.append(command)
        session.end_round()   # as the real invoke_command ends
        return fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(response="done"))

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(deterministic))


def test_on_the_assistant_route_a_slash_message_is_the_reply_too(library_env, server, monkeypatch):
    """Changed (bd fix-fpyb.7, owner decision 2026-11-19): ``/invoke_assistant`` keeps the "/", and the context now reads
    a "/" message sent while a question is out as the reply with the "/" removed, as ``/invoke_agent`` always did."""
    with TestClient(server.app) as client:
        headers, seen = _pause_and_route(server, client, library_env, monkeypatch)
        handled: list[str] = []
        _deterministic_command_handler(monkeypatch, handled)
        response = client.post("/invoke_assistant", headers=headers,
                               json={"user_query": "/find_book query=Messiah", "timeout_seconds": 30})

    assert response.status_code == 200, response.text
    # The route kept the "/"; the context read it as the reply (unreadable as yes or no) and ran no command.
    assert seen == ["/find_book query=Messiah"] and handled == []
    assert response.json()["answer"].startswith("I could not read that reply as yes or no. ")
    assert library_env.ctx._awaiting_user is True
    assert library_env.ctx._waiting.payload["decision"]["asked"] == 2
    assert _find_runs(library_env) == ["find_book query=Dune"]


def test_on_the_agent_route_a_slash_message_is_the_reply_and_the_second_one_ends_the_command(
        library_env, server, monkeypatch):
    with TestClient(server.app) as client:
        headers, seen = _pause_and_route(server, client, library_env, monkeypatch)
        handled: list[str] = []
        _deterministic_command_handler(monkeypatch, handled)
        first = client.post("/invoke_agent", headers=headers,
                            json={"user_query": "/find_book query=Messiah", "timeout_seconds": 30})
        assert first.status_code == 200, first.text
        # The route stripped the "/": the context read the message as the reply, and it could not read it as yes or no.
        assert seen == ["find_book query=Messiah"] and handled == []
        body = first.json()
        assert body["status"] == "awaiting_user"
        assert body["answer"].startswith("I could not read that reply as yes or no. ")
        assert library_env.ctx._waiting.payload["decision"]["asked"] == 2

        second = client.post("/invoke_agent", headers=headers,
                             json={"user_query": "/list_branches", "timeout_seconds": 30})
        assert second.status_code == 200, second.text

    assert seen[-1] == "list_branches" and handled == []
    ended = second.json()
    assert ended["status"] == "completed"
    stops = ended["command_outputs"][-1]["command_response"]["artifacts"]["fw_not_run"]
    assert stops[0]["open_step"]["kind"] == "conditional" and "twice" in stops[0]["open_step"]["reason"]
    assert _find_runs(library_env) == ["find_book query=Dune"]


def test_the_question_says_a_slash_message_is_the_reply(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(Conditional)
    _plan(monkeypatch, library_env)
    text = _pending(library_env.ctx.process_turn("look up the book Dune"))[0]
    assert "every message, including one that begins with '/', is read as the reply" in text
    assert "separate command" not in text and "invoke_assistant" not in text
