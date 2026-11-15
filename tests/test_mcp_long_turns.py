"""Long turns through the MCP mount (fix-1ma8).

The mount is fastapi-mcp over the app in process. These tests call the tools the way an MCP client does, over
the streamable-HTTP route the mount serves at ``/mcp`` (initialize, then tools/call), with no model: a turn is
made slow by wrapping the channel's ``process_turn``, and the real approval gate then runs on a real action.
"""
from __future__ import annotations

import contextlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from tests.test_approval_endpoint_surfaces import (  # noqa: F401  (fixtures and helpers)
    _configure, _session, app_module, fastworkflow_action, no_live_model)

SLOW_SECONDS = 11.5   # just past the 10 s the mount's default client was built with


class McpClient:
    """A minimal MCP client over the /mcp streamable-HTTP route (json_response, stateful session)."""

    def __init__(self, client: TestClient, bearer: str):
        self.client = client
        self.headers = {"Authorization": bearer, "Accept": "application/json, text/event-stream",
                        "Content-Type": "application/json"}
        self._id = 0
        init = self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                        "clientInfo": {"name": "pytest", "version": "0"}})
        assert "result" in init, init
        sid = self.last_headers.get("mcp-session-id")
        if sid:
            self.headers["mcp-session-id"] = sid
        client.post("/mcp", headers=self.headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _rpc(self, method: str, params: dict) -> dict:
        self._id += 1
        r = self.client.post("/mcp", headers=self.headers,
                             json={"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        self.last_headers = r.headers
        assert r.status_code == 200, r.text
        return r.json()

    def tools(self) -> dict:
        return {t["name"]: t for t in self._rpc("tools/list", {})["result"]["tools"]}

    def call(self, name: str, arguments: dict) -> dict:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})["result"]


@contextlib.contextmanager
def mcp_client(app_module, monkeypatch):
    uid = _configure(monkeypatch, "unsigned", "channel")
    with TestClient(app_module.app) as client:     # the lifespan starts the MCP session manager
        channel, headers, _wf, _chores, ctx = _session(app_module, client, uid)
        yield channel, McpClient(client, headers["Authorization"]), ctx


def slow_process_turn(ctx, seconds: float):
    def slow(message, *, unattended=False, approval_surface=None, approval_token=None):
        time.sleep(seconds)
        return ctx.process_action_turn(fastworkflow_action(), approval_surface=approval_surface)
    ctx.process_turn = slow


def body_of(result: dict) -> dict:
    """The JSON body of a one-object tool result."""
    assert not result.get("isError"), result
    return json.loads(result["content"][0]["text"])


def frames_of(result: dict) -> list[dict]:
    """The NDJSON frames of an invoke_agent result (the mount returns the whole stream as one text block)."""
    assert not result.get("isError"), result
    return [json.loads(line) for line in result["content"][0]["text"].splitlines() if line.strip()]


def wait_until_done(mcp: McpClient, turn_key: str, within: float = 20.0) -> dict:
    """get_turn until exec_state is done (a bounded wait on a deliberately slow command)."""
    deadline = time.monotonic() + within
    while True:
        body = body_of(mcp.call("get_turn", {"turn_key": turn_key}))
        if body["exec_state"] != "running" or time.monotonic() > deadline:
            return body
        time.sleep(0.5)


def test_a_turn_longer_than_ten_seconds_completes_through_the_mount_on_both_tools(app_module, monkeypatch):
    """The mount's default HTTP client is built with timeout=10.0. Observed: that does NOT cut a turn through
    the in-process ASGI transport (an 11.5 s turn completes in full on both tools, same result as a fast one),
    so it was not a defect. The explicit client of setup_mcp keeps it that way whatever the library does.
    The two tools run at the same time on two conversations, so the test takes about 12 s, not 24."""
    import concurrent.futures as cf
    uid = _configure(monkeypatch, "unsigned", "channel")
    with TestClient(app_module.app) as client:       # the lifespan starts the MCP session manager
        calls = {}
        for tool in ("invoke_assistant", "invoke_agent"):
            _channel, headers, _wf, _chores, ctx = _session(app_module, client, uid)
            slow_process_turn(ctx, SLOW_SECONDS)
            calls[tool] = McpClient(client, headers["Authorization"])

        def run(tool):
            started = time.monotonic()
            result = calls[tool].call(tool, {"user_query": "change the assignee", "timeout_seconds": 60})
            return result, time.monotonic() - started

        with cf.ThreadPoolExecutor(2) as pool:
            futures = {tool: pool.submit(run, tool) for tool in calls}
            outcomes = {tool: future.result() for tool, future in futures.items()}

    result, took = outcomes["invoke_assistant"]
    assert took >= SLOW_SECONDS
    body = body_of(result)
    assert body["exec_state"] == "done" and body["turn_key"] and body["answer"]
    result, took = outcomes["invoke_agent"]
    assert took >= SLOW_SECONDS
    frames = frames_of(result)
    assert frames[-1]["type"] == "output" and frames[-1]["data"]["answer"]
    assert not [f for f in frames if f["type"] == "error"]


def test_the_mount_client_has_no_read_timeout():
    from fastworkflow.run_fastapi_mcp.mcp_specific import MOUNT_HTTP_TIMEOUT
    assert MOUNT_HTTP_TIMEOUT.read is None and MOUNT_HTTP_TIMEOUT.connect is not None


def test_the_recovery_tools_are_listed_and_the_time_out_is_described(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, _ctx):
        tools = mcp.tools()
    assert {"get_turn", "get_latest_turn", "invoke_assistant", "invoke_agent"} <= set(tools)
    assert "get_turn_trace" not in tools
    for name in ("invoke_assistant", "invoke_agent"):
        text = tools[name]["description"]
        assert "timeout_seconds" in text and "get_turn" in text and "second time" in text
    assert "timed out" in tools["get_turn"]["description"]
    assert tools["get_turn"]["inputSchema"]["required"] == ["turn_key"]


def test_an_assistant_turn_past_its_timeout_returns_a_turn_id_and_is_read_back_with_get_turn(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        slow_process_turn(ctx, 3)
        first = body_of(mcp.call("invoke_assistant", {"user_query": "change the assignee", "timeout_seconds": 1}))
        assert first["exec_state"] == "running" and first["turn_key"]
        running = body_of(mcp.call("get_turn", {"turn_key": first["turn_key"]}))
        assert running["exec_state"] == "running" and running["turn_key"] == first["turn_key"]
        latest = body_of(mcp.call("get_latest_turn", {}))
        assert latest["exec_state"] == "running" and latest["turn_key"] == first["turn_key"]
        done = wait_until_done(mcp, first["turn_key"])
        assert done["exec_state"] == "done" and done["answer"]
        again = body_of(mcp.call("get_latest_turn", {}))
        assert again["exec_state"] == "done" and again["answer"] == done["answer"]


def test_an_agent_turn_past_its_timeout_carries_its_turn_id_in_every_frame_and_is_read_back(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        slow_process_turn(ctx, 3)
        frames = frames_of(mcp.call("invoke_agent", {"user_query": "change the assignee", "timeout_seconds": 1}))
        kinds = [f["type"] for f in frames]
        assert "timeout" in kinds and kinds[-1] == "output"
        assert frames[kinds.index("timeout")]["data"]["still_running"] is True
        keys = {f["turn_key"] for f in frames}
        assert len(keys) == 1 and None not in keys
        # The call returned at the end of the turn; its execution key is retired, the stored turn is read by it.
        done = wait_until_done(mcp, keys.pop())
        assert done["exec_state"] == "done" and done["answer"] == frames[-1]["data"]["answer"]


def test_the_lookup_tools_are_scoped_to_the_callers_own_channel(app_module, monkeypatch):
    uid = _configure(monkeypatch, "unsigned", "channel")
    with TestClient(app_module.app) as client:
        _c1, headers1, _wf1, _ch1, ctx1 = _session(app_module, client, uid)
        _c2, headers2, _wf2, _ch2, _ctx2 = _session(app_module, client, uid)
        owner = McpClient(client, headers1["Authorization"])
        other = McpClient(client, headers2["Authorization"])
        slow_process_turn(ctx1, 2)
        first = body_of(owner.call("invoke_assistant", {"user_query": "change the assignee", "timeout_seconds": 1}))
        key = first["turn_key"]
        # Running: the other channel cannot read it, and cannot tell it from an unknown key.
        refused = other.call("get_turn", {"turn_key": key})
        unknown = other.call("get_turn", {"turn_key": "20000101T000000.000000Z-000000000000"})
        assert refused["isError"] and unknown["isError"]
        assert refused["content"][0]["text"].replace(key, "K") == unknown["content"][0]["text"].replace(
            "20000101T000000.000000Z-000000000000", "K")
        # The other channel's "latest" is its own (nothing yet), not the owner's running turn.
        assert other.call("get_latest_turn", {})["isError"]
        done = wait_until_done(owner, key)
        assert done["exec_state"] == "done"
        # Finished and stored: still the owner's alone.
        assert other.call("get_turn", {"turn_key": done["logical_turn_key"]})["isError"]
        assert other.call("get_latest_turn", {})["isError"]
        assert body_of(owner.call("get_latest_turn", {}))["exec_state"] == "done"
