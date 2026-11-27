"""Progress of long turns over MCP (fix-1ma8, audit gap 5).

The installed fastapi-mcp mount cannot deliver ``notifications/progress`` before a tool result (its
streamable-HTTP transport answers in JSON mode, which discards notifications of a request, and buffers the whole
response even in SSE mode; see the README section on MCP). So progress over MCP is the polling path: a running turn
read with get_turn / get_latest_turn carries ``progress``, from memory. These tests go through the real ``/mcp``
route and the real turn registry, no model: a turn is made slow, and made to show stages, by wrapping the channel's
``process_turn`` and putting the real ``ProgressEvent`` on the channel's trace queue, as the engine does.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import time

import httpx
import jsonschema

import fastworkflow
from fastworkflow.run_fastapi_mcp.mcp_specific import TURN_RESULT_SCHEMA, _mcp_transport
from fastworkflow.run_fastapi_mcp.turns import TurnExecution, TurnStreamChannel, note_stream_frame, running_progress
from fastworkflow.core.turn_progress import ProgressEvent
from tests.test_approval_endpoint_surfaces import (  # noqa: F401  (fixtures and helpers)
    _configure, _session, app_module, no_live_model)
from tests.test_fastapi_end_session import FREE_ACTION
from tests.test_mcp_long_turns import McpClient, body_of, mcp_client

STAGES = ("Splitting the request", "Matching sub-task 1 of 2", "Drafting sub-task 1 of 2")


def staged_turn(ctx, hold: float):
    """A process_turn that announces three stages, ``hold`` seconds apart, then finishes (real TurnOutput)."""
    def process(message, *, unattended=False, approval_surface=None, approval_token=None):
        for index, label in enumerate(STAGES):
            ctx.command_trace_queue.put(ProgressEvent("split", label, int(time.time() * 1000)))
            time.sleep(hold)
        return ctx.process_action_turn(fastworkflow.Action(**FREE_ACTION), approval_surface=approval_surface)
    return process


def poll_until_done(mcp: McpClient, tool: str, arguments: dict, seen: list, within: float = 30.0) -> dict:
    deadline = time.monotonic() + within
    while True:
        body = body_of(mcp.call(tool, arguments))
        seen.append(body)
        if body["exec_state"] != "running" or time.monotonic() > deadline:
            return body
        time.sleep(0.2)


def test_the_mount_discards_progress_notifications_so_none_arrives_and_the_result_is_unchanged(app_module, monkeypatch):
    """A call that carries a progress token gets one JSON body, the result, and nothing before it.

    The token is only a ``_meta`` field the SDK hands the tool handler; the transport (json_response True) drops
    every notification of the request and answers once. Same result with and without the token."""
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        assert _mcp_transport(mcp_of(app_module)).json_response is True
        results = {}
        for label, meta in (("plain", {}), ("token", {"_meta": {"progressToken": "p-1"}})):
            ctx.process_turn = staged_turn(ctx, 0.3)
            mcp._id += 1
            started = time.monotonic()
            response = mcp.client.post("/mcp", headers=mcp.headers, json={
                "jsonrpc": "2.0", "id": mcp._id, "method": "tools/call",
                "params": {"name": "invoke_agent", "arguments": {"user_query": f"go {label}"}, **meta}})
            results[label] = (response, time.monotonic() - started)
    for label, (response, took) in results.items():
        assert response.status_code == 200 and response.headers["content-type"].startswith("application/json"), label
        message = response.json()                       # ONE json-rpc message: the response
        assert message["id"] and "result" in message and message.get("method") is None
        assert "notifications/progress" not in response.text
        assert took >= 0.9                              # the call really stayed open for the whole turn
    plain, token = (results[k][0].json()["result"] for k in ("plain", "token"))
    for key in ("answer", "status", "success", "failure_reason", "exec_state"):
        assert plain["structuredContent"][key] == token["structuredContent"][key]
    assert [o["command_response"] for o in plain["structuredContent"]["command_outputs"]] == \
        [o["command_response"] for o in token["structuredContent"]["command_outputs"]]
    assert plain.get("isError", False) == token.get("isError", False) is False
    frames = [json.loads(line) for line in token["content"][0]["text"].splitlines() if line.strip()]
    assert [f["data"]["label"] for f in frames if f["type"] == "progress"] == list(STAGES)


def mcp_of(app_module):
    return app_module.mcp_mount


def test_a_running_agent_turn_reports_its_latest_step_to_a_second_request(app_module, monkeypatch):
    """The recommended pattern: invoke_agent is open (blocked), a second request reads get_latest_turn / get_turn."""
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        ctx.process_turn = staged_turn(ctx, 0.8)
        watcher = McpClient(mcp.client, mcp.headers["Authorization"])
        with cf.ThreadPoolExecutor(1) as pool:
            call = pool.submit(mcp.call, "invoke_agent", {"user_query": "plan it", "timeout_seconds": 60})
            latest: list[dict] = []
            by_key_running: list[dict] = []
            deadline = time.monotonic() + 20
            while not call.done() and time.monotonic() < deadline:
                try:
                    body = body_of(watcher.call("get_latest_turn", {}))
                except AssertionError:                   # 404 before the turn is registered
                    time.sleep(0.05)
                    continue
                if body["exec_state"] == "running":
                    latest.append(body)
                    by_key = watcher.call("get_turn", {"turn_key": body["turn_key"]})
                    if not by_key.get("isError") and json.loads(by_key["content"][0]["text"])["exec_state"] == "running":
                        by_key_running.append(json.loads(by_key["content"][0]["text"]))
                time.sleep(0.2)
            result = call.result(timeout=20)
    assert latest, "no running read happened during the turn"
    for body in latest:
        jsonschema.validate(body, TURN_RESULT_SCHEMA)
    labels = [b["progress"]["last_frame"]["label"] for b in latest if "last_frame" in b["progress"]]
    assert set(labels) <= set(STAGES) and labels, labels
    assert labels == sorted(labels, key=STAGES.index), "the latest step only moves forward"
    frames = [b["progress"]["frames"] for b in latest]
    assert frames == sorted(frames) and frames[-1] >= 2
    elapsed = [b["progress"]["elapsed_seconds"] for b in latest]
    assert elapsed == sorted(elapsed) and elapsed[-1] > 0.5
    assert all(b["progress"]["last_frame"]["seconds_ago"] >= 0 for b in latest if "last_frame" in b["progress"])
    assert all(b["progress"]["last_frame"]["type"] == "progress" for b in latest if "last_frame" in b["progress"])
    # The finished result is the same as always: no progress key, the terminal frame as the turn object.
    assert "progress" not in result["structuredContent"] and result["structuredContent"]["answer"]
    assert not result.get("isError")
    assert by_key_running and all("progress" in b for b in by_key_running)
    for body in by_key_running:
        jsonschema.validate(body, TURN_RESULT_SCHEMA)


def test_a_running_assistant_turn_reports_elapsed_time_and_no_frames(app_module, monkeypatch):
    """invoke_assistant has no stream, so there is nothing to report but time: it returns past its deadline."""
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        def slow(message, *, unattended=False, approval_surface=None, approval_token=None):
            time.sleep(2.5)
            return ctx.process_action_turn(fastworkflow.Action(**FREE_ACTION), approval_surface=approval_surface)
        ctx.process_turn = slow
        first = body_of(mcp.call("invoke_assistant", {"user_query": "hello", "timeout_seconds": 1}))
        assert first["exec_state"] == "running"
        time.sleep(0.6)
        polled = body_of(mcp.call("get_turn", {"turn_key": first["turn_key"]}))
        seen: list[dict] = []
        done = poll_until_done(mcp, "get_turn", {"turn_key": first["turn_key"]}, seen)
    for body in (first, polled):
        jsonschema.validate(body, TURN_RESULT_SCHEMA)
        assert body["progress"]["frames"] == 0 and "last_frame" not in body["progress"]
    assert polled["progress"]["elapsed_seconds"] > first["progress"]["elapsed_seconds"]
    assert done["exec_state"] == "done" and "progress" not in done


def test_a_client_that_disconnects_mid_turn_leaves_the_turn_and_the_session_healthy(app_module, monkeypatch):
    """Cancel the in-flight MCP request (what a closed connection does to the handler); the turn runs on."""
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        ctx.process_turn = staged_turn(ctx, 0.7)

        async def call_and_give_up():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                                         base_url="http://apiserver", timeout=None) as http:
                request = http.post("/mcp", headers=mcp.headers, json={
                    "jsonrpc": "2.0", "id": 900, "method": "tools/call",
                    "params": {"name": "invoke_agent", "arguments": {"user_query": "abandoned"},
                               "_meta": {"progressToken": "gone"}}})
                try:
                    await asyncio.wait_for(request, 0.8)
                except asyncio.TimeoutError:
                    return True
                return False

        assert mcp.client.portal.call(call_and_give_up), "the call finished before the client gave up"
        latest = body_of(mcp.call("get_latest_turn", {}))
        assert latest["exec_state"] == "running" and "progress" in latest
        seen: list[dict] = []
        done = poll_until_done(mcp, "get_latest_turn", {}, seen)
        assert done["exec_state"] == "done" and done["status"] == "completed" and done["answer"]
        ctx.process_turn = staged_turn(ctx, 0.0)
        again = mcp.call("invoke_agent", {"user_query": "after the disconnect"})
        assert not again.get("isError") and again["structuredContent"]["answer"]
        ended = body_of(mcp.call("end_session", {}))
        assert ended["ended"] is True


def test_the_frame_observer_never_fails_the_stream_and_a_terminal_frame_is_not_remembered():
    stream = TurnStreamChannel()
    stream.bind("exec-1")

    def explode(frame):
        raise RuntimeError("observer broke")
    stream.on_frame = explode
    frame = stream.emit_nowait("progress", {"label": "Splitting the request"})
    assert frame is not None and frame["seq"] == 0          # the frame was queued despite the observer failing

    execn = TurnExecution(turn_key="exec-2", channel_id="c", kind="invoke_agent_stream", idempotency_key="k")
    for seq, (kind, data) in enumerate([("trace", {"direction": "agent_to_workflow", "command_name": "find_book",
                                                   "response_text": "SECRET ANSWER TEXT"}),
                                        ("progress", {"label": "Drafting sub-task 1 of 2"})]):
        note_stream_frame(execn, {"type": kind, "seq": seq, "data": data})
    note_stream_frame(execn, {"type": "output", "seq": 2, "data": {"answer": "x"}})
    progress = running_progress(execn)
    assert progress["frames"] == 2 and progress["last_frame"]["label"] == "Drafting sub-task 1 of 2"
    assert progress["last_frame"]["seq"] == 1 and "SECRET" not in json.dumps(progress)
    note_stream_frame(execn, {"type": "trace", "seq": 3, "data": {"direction": "agent_to_workflow", "command_name": "find_book"}})
    assert running_progress(execn)["last_frame"]["label"] == "agent_to_workflow: find_book"


def test_the_tool_text_names_the_polling_pattern(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, _ctx):
        tools = mcp.tools()
    assert "progress" in tools["get_turn"]["description"]
    assert "No MCP progress notifications" in tools["invoke_agent"]["description"]
    assert "progress" in tools["get_latest_turn"]["outputSchema"]["properties"]
