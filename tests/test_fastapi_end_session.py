"""POST /end_session and the MCP tool of the same name (fix-1ma8).

The server holds one live session per channel and, until this operation, nothing ended one: a session whose
state cannot be checkpointed (a context class with no ``get_state`` hook) is pinned and never evicted, so a
client that opens many short-lived channels grew the process without bound. These tests drive the real app
through FastAPI's TestClient and the real MCP mount, with no model: a turn is made slow, or finished, by
wrapping the channel's ``process_turn`` around a real model-free action, as the other server tests do.

Pinned and checkpointable sessions both use the todo test workflow: its root ``TodoListManager`` implements
``get_state``; a bare ``TodoList`` as the current context has none, which is what pins.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import time
import uuid

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.review import approval_binding as ab
from tests.test_approval_endpoint_surfaces import (  # noqa: F401  (fixtures and helpers)
    ACTION, _configure, app_module, no_live_model)
from tests.test_approval_fastapi import _auth
from tests.test_mcp_long_turns import McpClient, body_of, slow_process_turn
from tests.todo_list_workflow.application.todo_list import TodoList
from tests.todo_list_workflow.application.todo_manager import TodoListManager

FREE_ACTION = {"command_name": "TodoList/get_properties", "command": "get_properties", "parameters": {}}


def _run(coro):
    return asyncio.run(coro)


def _live(app_module) -> int:
    return len(app_module.session_manager._sessions)


def _runtime(app_module, channel_id):
    return _run(app_module.session_manager.get_session(channel_id))


def _pin(app_module, channel_id) -> None:
    """Make the channel's session non-checkpointable: a bare TodoList (no get_state hook) is the context."""
    wf = _runtime(app_module, channel_id).execution_context.app_workflow
    wf.record = TodoList(id=7, description="chores")


def _give_state(app_module, channel_id) -> TodoListManager:
    """Make it checkpointable: the root context is a TodoListManager, which implements get_state."""
    wf = _runtime(app_module, channel_id).execution_context.app_workflow
    manager = TodoListManager(str(Path(wf.folderpath).parent / f"todo_{uuid.uuid4().hex}.json"))
    manager.lists[7] = TodoList(id=7, description="chores")
    manager.lists[7].parent = manager
    wf.root = manager
    return manager


def _finish_a_turn(client, headers, ctx) -> dict:
    """One finished turn on the channel, through the real endpoint, around a model-free free action."""
    def free_turn(message, *, unattended=False, approval_surface=None, approval_token=None):
        return ctx.process_action_turn(fastworkflow.Action(**FREE_ACTION), approval_surface=approval_surface)

    ctx.process_turn = free_turn
    response = client.post("/invoke_assistant", headers=headers,
                           json={"user_query": "show the list", "timeout_seconds": 30})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["exec_state"] == "done"
    return body


def _new_channel(app_module, client, uid, *, pinned: bool):
    channel_id = f"end_{uuid.uuid4().hex[:8]}"
    init = client.post("/initialize", json={"channel_id": channel_id, "user_id": uid})
    assert init.status_code == 200, init.text
    (_pin if pinned else _give_state)(app_module, channel_id)
    ctx = _runtime(app_module, channel_id).execution_context
    return channel_id, _auth(init.json()), ctx.app_workflow, ctx


@pytest.fixture
def client(app_module, monkeypatch):
    uid = _configure(monkeypatch, "unsigned", "channel")
    with TestClient(app_module.app) as test_client:
        test_client.uid = uid
        yield test_client


def test_ending_a_pinned_session_releases_it_and_says_so(app_module, client):
    channel_id, headers, _wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)
    assert _live(app_module) == 1

    response = client.post("/end_session", headers=headers)

    assert response.status_code == 200, response.text
    assert response.json() == {"channel_id": channel_id, "ended": True, "pinned": True,
                               "pending_discarded": False}
    assert _runtime(app_module, channel_id) is None
    assert _live(app_module) == 0


def test_ending_a_checkpointable_session_saves_it_and_the_channel_restores(app_module, client):
    channel_id, headers, wf, _ctx = _new_channel(app_module, client, client.uid, pinned=False)
    assert type(wf.root).__name__ == "TodoListManager"
    wf.root.lists[7].description = "changed before the end"
    from fastworkflow.run_fastapi_mcp import checkpoint
    assert checkpoint.assess(_runtime(app_module, channel_id), app_module.session_manager.session_state_store).evictable

    body = client.post("/end_session", headers=headers).json()

    assert body == {"channel_id": channel_id, "ended": True, "pinned": False, "pending_discarded": False}
    assert _runtime(app_module, channel_id) is None
    manager = app_module.session_manager
    from fastworkflow.run_fastapi_mcp import checkpoint
    assert manager.checkpoint_store.load_for_adoption(
        deployment_id=checkpoint.deployment_id(),
        workflow_fingerprint=checkpoint.workflow_fingerprint(app_module.ARGS.workflow_path),
        channel_id=channel_id,
    ) is not None, "the checkpoint an eviction would have written was not written"

    # The next call on the channel: what an eviction followed by a restore does. The state comes back.
    again = client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid})
    assert again.status_code == 200, again.text
    restored = _runtime(app_module, channel_id).execution_context.app_workflow
    assert type(restored.root).__name__ == "TodoListManager"
    assert restored.root.lists[7].description == "changed before the end"


def test_a_pinned_channel_starts_fresh_when_it_is_used_again(app_module, client):
    channel_id, headers, wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)
    assert wf._current_command_context is not None
    client.post("/end_session", headers=headers)

    # The same token, no initialize: the next ordinary call creates a clean session.
    response = client.post("/invoke_assistant", headers=headers,
                           json={"user_query": "/get_properties", "timeout_seconds": 5})
    assert response.status_code in (200, 202), response.text
    fresh = _runtime(app_module, channel_id).execution_context.app_workflow
    assert fresh is not wf
    assert fresh.root is None and fresh._current_command_context is None, "the dropped state came back"
    assert _live(app_module) == 1

    # And a new initialize after another end does the same.
    client.post("/end_session", headers=headers)
    assert client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid}).status_code == 200
    assert _runtime(app_module, channel_id).execution_context.app_workflow._current_command_context is None


def test_end_is_refused_while_a_turn_runs_and_works_once_it_finished(app_module, client):
    channel_id, headers, _wf, ctx = _new_channel(app_module, client, client.uid, pinned=True)
    slow_process_turn(ctx, 3)
    first = client.post("/invoke_assistant", headers=headers,
                        json={"user_query": "change the assignee", "timeout_seconds": 1})
    assert first.status_code == 202, first.text

    refused = client.post("/end_session", headers=headers)

    assert refused.status_code == 409
    assert "turn is already in progress" in refused.json()["detail"]
    assert _runtime(app_module, channel_id) is not None, "a running turn's session was released"

    turn_key = first.json()["turn_key"]
    deadline = time.monotonic() + 30
    while client.get(f"/turns/{turn_key}", headers=headers).json().get("exec_state") == "running":
        assert time.monotonic() < deadline
        time.sleep(0.3)
    # A gated action is pending once that turn is done; clear it so only the turn state is in question.
    _runtime(app_module, channel_id).execution_context.discard_live_pending()
    assert client.post("/end_session", headers=headers).status_code == 200


def test_end_is_refused_while_another_request_holds_the_channel(app_module, client):
    channel_id, headers, _wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)

    async def while_leased():
        async with app_module.session_manager.leased_session(channel_id):
            return await app_module.session_manager.end_session(channel_id)

    from fastworkflow.run_fastapi_mcp.utils import SessionEndRefused
    with pytest.raises(SessionEndRefused) as refusal:
        _run(while_leased())
    assert refusal.value.reason == "busy"
    assert _runtime(app_module, channel_id) is not None
    assert client.post("/end_session", headers=headers).json()["ended"] is True


def test_a_pending_approval_is_not_lost_by_accident(app_module, client, monkeypatch):
    # An approval is only asked for a verified identity: a signed token, as in the approval tests.
    _configure(monkeypatch, "signed_uid", "host")
    channel_id, headers, wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)
    posted = client.post("/perform_action", headers=headers, json={"action": ACTION, "timeout_seconds": 20})
    assert posted.status_code == 200, posted.text
    assert ab.store_for(wf).active() is not None, posted.text

    refused = client.post("/end_session", headers=headers)

    assert refused.status_code == 409
    assert "approval" in refused.json()["detail"] and "discard_pending" in refused.json()["detail"]
    assert _runtime(app_module, channel_id) is not None
    assert ab.store_for(wf).active() is not None, "a refusal must change nothing"

    ended = client.post("/end_session", headers=headers, json={"discard_pending": True})

    assert ended.status_code == 200, ended.text
    assert ended.json()["ended"] is True and ended.json()["pending_discarded"] is True
    assert not app_module.session_manager.session_state_store.exists(channel_id), \
        "the discarded question would have come back with the next session"
    assert client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid}).status_code == 200
    assert not _runtime(app_module, channel_id).execution_context.has_pending_state()


def test_a_pending_agent_question_is_refused_then_discarded_on_request(app_module, client):
    channel_id, headers, _wf, ctx = _new_channel(app_module, client, client.uid, pinned=True)
    ctx._ask_agent_question("Pick one?", "original")

    refused = client.post("/end_session", headers=headers, json={"discard_pending": False})

    assert refused.status_code == 409
    assert "agent question" in refused.json()["detail"]
    assert _runtime(app_module, channel_id) is not None

    ended = client.post("/end_session", headers=headers, json={"discard_pending": True}).json()
    assert ended["ended"] is True and ended["pending_discarded"] is True and ended["pinned"] is True
    assert not app_module.session_manager.session_state_store.exists(channel_id)
    assert client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid}).status_code == 200
    assert _runtime(app_module, channel_id).execution_context.awaiting_user is False


def test_ending_is_idempotent(app_module, client):
    channel_id, headers, _wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)
    assert client.post("/end_session", headers=headers).json()["ended"] is True

    again = client.post("/end_session", headers=headers)

    assert again.status_code == 200
    assert again.json() == {"channel_id": channel_id, "ended": False, "pinned": False,
                            "pending_discarded": False}
    # A channel whose session is gone for another reason (dropped by the process) is the same case.
    other = "gone_" + uuid.uuid4().hex[:6]
    other_headers = _auth(client.post("/initialize", json={"channel_id": other, "user_id": client.uid}).json())
    _run(app_module.session_manager.remove_session(other))
    assert client.post("/end_session", headers=other_headers).json()["ended"] is False


def test_a_finished_turn_is_still_readable_after_the_session_ended(app_module, client):
    channel_id, headers, _wf, ctx = _new_channel(app_module, client, client.uid, pinned=True)
    done = _finish_a_turn(client, headers, ctx)
    logical = done.get("logical_turn_key") or done["turn_key"]

    assert client.post("/end_session", headers=headers).json()["ended"] is True

    read = client.get(f"/turns/{logical}", headers=headers)
    assert read.status_code == 200, read.text
    assert read.json()["answer"] == done["answer"]
    # The channel's latest turn is its last finished one: the store answers, no session is needed.
    latest = client.get("/turns/latest", headers=headers)
    assert latest.status_code == 200, latest.text
    assert latest.json()["answer"] == done["answer"]
    assert _runtime(app_module, channel_id) is None, "reading a turn must not create a session"
    # The channel's turn pointer is gone, so nothing reports a turn running.
    assert app_module.turn_registry.active_turn_key(channel_id) is None


def test_a_token_can_only_end_its_own_channel(app_module, client):
    first, headers1, _w1, _c1 = _new_channel(app_module, client, client.uid, pinned=True)
    second, headers2, _w2, _c2 = _new_channel(app_module, client, client.uid, pinned=True)

    body = client.post("/end_session", headers=headers2).json()

    assert body["channel_id"] == second and body["ended"] is True
    assert _runtime(app_module, first) is not None, "another channel's session was ended"
    # No channel parameter exists to name the other one.
    named = client.post("/end_session", headers=headers2, json={"channel_id": first, "discard_pending": False})
    assert named.json()["channel_id"] == second and _runtime(app_module, first) is not None
    assert client.post("/end_session").status_code in (401, 403)


def test_the_tool_is_listed_over_mcp_and_works(app_module, client):
    channel_id, headers, _wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)
    mcp = McpClient(client, headers["Authorization"])

    tools = mcp.tools()

    assert "end_session" in tools
    text = tools["end_session"]["description"]
    for phrase in ("finished with a channel", "get_turn", "pinned", "discard_pending", "ended: false"):
        assert phrase in text, phrase
    assert "channel_id" not in (tools["end_session"]["inputSchema"].get("properties") or {})
    for name in ("invoke_agent", "invoke_assistant"):
        assert "end_session" in tools[name]["description"]

    result = mcp.call("end_session", {})
    assert body_of(result) == {"channel_id": channel_id, "ended": True, "pinned": True,
                               "pending_discarded": False}
    assert _runtime(app_module, channel_id) is None


def test_sixty_channels_opened_and_ended_in_a_row_leave_nothing_behind(app_module, client):
    """The benchmark's pattern, on a workflow whose sessions are pinned: one channel per job."""
    peak = 0
    for _ in range(60):
        channel_id, headers, _wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)
        peak = max(peak, _live(app_module))
        assert client.post("/end_session", headers=headers).json()["pinned"] is True
    assert peak <= 1
    assert _live(app_module) == 0


def test_the_probe_reports_live_and_pinned_sessions(app_module, client):
    _c1, _h1, _w1, _x1 = _new_channel(app_module, client, client.uid, pinned=True)
    _c2, _h2, _w2, _x2 = _new_channel(app_module, client, client.uid, pinned=False)

    plain = client.get("/probes/readyz").json()["sessions"]
    assert plain["live"] == 2 and plain["max_live"] == app_module.session_manager.max_live_sessions
    assert plain["over_target"] == 0 and "pinned" not in plain

    detailed = client.get("/probes/readyz", params={"sessions": "true"}).json()["sessions"]
    assert detailed["live"] == 2 and detailed["pinned"] == 1 and detailed["busy_unassessed"] == 0


def test_the_workflow_without_a_state_hook_is_named_at_startup(app_module, monkeypatch):
    from fastworkflow.run_fastapi_mcp import checkpoint
    missing = checkpoint.contexts_without_state_hook(app_module.ARGS.workflow_path)
    assert "TodoList" in missing and "TodoListManager" not in missing

    logger = logging.getLogger("fastWorkflow")
    seen: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record)

    handler = Capture(level=logging.DEBUG)
    logger.addHandler(handler)
    try:
        _configure(monkeypatch, "unsigned", "channel")
        with TestClient(app_module.app):
            pass
    finally:
        logger.removeHandler(handler)
    warnings = [r.getMessage() for r in seen if r.levelno == logging.WARNING and "get_state hook" in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert "TodoList" in warnings[0] and "never be evicted" in warnings[0] and "end_session" in warnings[0]


def test_a_workflow_whose_contexts_all_have_the_hook_gets_no_warning():
    import os
    from fastworkflow.run_fastapi_mcp import checkpoint
    template = os.path.join(fastworkflow.get_fastworkflow_package_path(), "examples", "simple_workflow_template")
    assert checkpoint.contexts_without_state_hook(template) == []
    assert checkpoint.startup_pinning_warning(template) is None
