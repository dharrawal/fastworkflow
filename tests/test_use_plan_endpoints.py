"""``use_plan`` on the HTTP and MCP surfaces (fix-1ma8.6): /invoke_agent, /invoke_agent_stream and the MCP invoke_agent
tool accept it, the other turn tools do not, and a name the session cannot honour is refused before any turn begins
(HTTP 400, MCP isError). Driven as tests/test_approval_endpoint_surfaces.py is: the channel's ``process_turn`` is the one
seam (it records what the endpoint passed); the validation is the context's own ``check_use_plan``, except where a case
needs a plan to exist (the todo workflow has no planning command), which names the one plan the check accepts."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fastworkflow.run_fastapi_mcp import mcp_specific
from tests.test_approval_endpoint_surfaces import (  # noqa: F401  (fixtures and helpers)
    _configure, _session, app_module, fastworkflow_action, no_live_model)
from tests.test_mcp_long_turns import McpClient, mcp_client
from tests.test_mcp_structured_results import text_of

PLAN = "Todo/some_plan"


def _seam(ctx, seen: list):
    def seam(message, *, unattended=False, approval_surface=None, approval_token=None, **extra):
        seen.append({"message": message, **extra})
        return ctx.process_action_turn(fastworkflow_action(), approval_surface=approval_surface)

    ctx.process_turn = seam


def _accept_plan(ctx, monkeypatch):
    real = ctx.check_use_plan
    monkeypatch.setattr(ctx, "check_use_plan", lambda name: None if name == PLAN else real(name))


@pytest.fixture
def http(app_module, monkeypatch):
    uid = _configure(monkeypatch, "unsigned", "channel")
    with TestClient(app_module.app) as client:
        _channel, headers, _wf, _chores, ctx = _session(app_module, client, uid)
        yield client, headers, ctx


def _frames(response):
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


@pytest.mark.parametrize("path", ["/invoke_agent", "/invoke_agent_stream"])
def test_an_unknown_plan_is_a_400_with_a_plain_message_and_no_turn_begins(http, path):
    client, headers, ctx = http
    seen: list = []
    _seam(ctx, seen)
    r = client.post(path, headers=headers, json={"user_query": "tell Ann a joke", "use_plan": "Nowhere/nothing",
                                                 "timeout_seconds": 20})

    assert r.status_code == 400, r.text
    assert "not a planning command available here" in r.json()["detail"]
    assert r.json()["reason"] == "use_plan_refused"
    assert seen == []


def test_invoke_agent_passes_a_chosen_plan_to_the_turn_and_nothing_when_absent(http, monkeypatch):
    client, headers, ctx = http
    seen: list = []
    _seam(ctx, seen)
    _accept_plan(ctx, monkeypatch)

    r = client.post("/invoke_agent", headers=headers, json={"user_query": "tell Ann a joke", "use_plan": PLAN,
                                                            "timeout_seconds": 20})
    assert r.status_code == 200, r.text
    r = client.post("/invoke_agent", headers=headers, json={"user_query": "and another", "timeout_seconds": 20})
    assert r.status_code == 200, r.text

    assert seen == [{"message": "tell Ann a joke", "use_plan": PLAN}, {"message": "and another"}]


def test_the_stream_passes_a_chosen_plan_to_the_turn(http, monkeypatch):
    client, headers, ctx = http
    seen: list = []
    _seam(ctx, seen)
    _accept_plan(ctx, monkeypatch)

    frames = _frames(client.post("/invoke_agent_stream", headers=headers, json={
        "user_query": "tell Ann a joke", "use_plan": PLAN, "timeout_seconds": 20}))

    assert [f["type"] for f in frames][-1] == "output"
    assert seen == [{"message": "tell Ann a joke", "use_plan": PLAN}]


def test_only_the_agent_request_model_has_use_plan():
    from fastworkflow.run_fastapi_mcp.utils import AgentInvokeRequest, InvokeRequest

    assert "use_plan" in AgentInvokeRequest.model_fields and "use_plan" not in InvokeRequest.model_fields


def test_the_mcp_invoke_agent_tool_declares_and_accepts_use_plan_and_the_other_turn_tools_do_not(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        tools = mcp.tools()
        schema = tools["invoke_agent"]["inputSchema"]["properties"]
        assert schema["use_plan"]["description"] == mcp_specific.ARGUMENT_DESCRIPTIONS["use_plan"]
        assert "available_plans" in tools["invoke_agent"]["description"]
        assert "use_plan" not in tools["invoke_assistant"]["inputSchema"]["properties"]

        refused = mcp.call("invoke_assistant", {"user_query": "hello", "use_plan": PLAN})
        assert refused["isError"] is True and "Unknown argument for invoke_assistant: use_plan." in text_of(refused)

        unknown = mcp.call("invoke_agent", {"user_query": "hello", "use_plan": "Nowhere/nothing", "timeout_seconds": 20})
        assert unknown["isError"] is True
        assert "not a planning command available here" in text_of(unknown)

        seen: list = []
        _seam(ctx, seen)
        _accept_plan(ctx, monkeypatch)
        ok = mcp.call("invoke_agent", {"user_query": "tell Ann a joke", "use_plan": PLAN, "timeout_seconds": 20})
        assert not ok.get("isError"), ok
        assert seen == [{"message": "tell Ann a joke", "use_plan": PLAN}]
