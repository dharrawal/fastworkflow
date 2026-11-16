"""The hard live-session ceiling, idle expiry of pinned sessions, and the dropped-session notice.

A session is pinned (never evicted) while its workflow's context class has no ``get_state`` hook, so the soft
target MAX_LIVE_SESSIONS alone bounds nothing for such a workflow. These tests drive the real server app and real
runtimes (no model) with sessions pinned the way tests/test_fastapi_end_session.py pins them, and check the
bound: live sessions never pass MAX_LIVE_SESSIONS_HARD, a request that cannot be made room for is refused with
503 + Retry-After, idle pinned sessions are expired only above the soft target, and a channel whose context was
dropped is told so exactly once.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from fastworkflow.run_fastapi_mcp import jwt_manager, session_notices
from fastworkflow.run_fastapi_mcp.utils import (
    BoundedTraceQueue,
    ChannelSessionManager,
    SessionCapacityExceeded,
    resolve_max_live_sessions_hard,
    resolve_pinned_idle_ttl,
    resolve_protected_idle_ttl,
)
from tests.test_approval_endpoint_surfaces import (  # noqa: F401  (fixtures and helpers)
    _configure, app_module, no_live_model)
from tests.test_approval_fastapi import _auth
from tests.test_fastapi_end_session import (  # noqa: F401
    FREE_ACTION, _finish_a_turn, _live, _new_channel, _pin, _run, _runtime, client)
from tests.test_mcp_long_turns import McpClient, body_of


def _limits(app_module, *, soft, hard, ttl=1800, protected_ttl=86400):
    manager = app_module.session_manager
    manager._max_live_sessions = soft
    manager._max_live_hard = hard
    manager._pinned_idle_ttl = ttl
    manager._protected_idle_ttl = protected_ttl
    return manager


def _age(app_module, channel_id, seconds):
    """Make the channel look idle for ``seconds`` (the session's own idle clock, not a stand-in for it)."""
    runtime = _runtime(app_module, channel_id)
    runtime.last_active -= seconds


def _token_for(channel_id: str) -> dict:
    return {"Authorization": f"Bearer {jwt_manager.create_access_token(channel_id, 'alice')}"}


def _fill(app_module, client, count):
    return [_new_channel(app_module, client, client.uid, pinned=True) for _ in range(count)]


# -- the ceiling ---------------------------------------------------------------------------------------------------

def test_a_load_of_pinned_channels_never_passes_the_ceiling(app_module, client):
    manager = _limits(app_module, soft=2, hard=5)
    admitted, refused, peak = [], 0, 0
    for _ in range(40):
        channel_id = f"load_{uuid.uuid4().hex[:8]}"
        response = client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid})
        peak = max(peak, _live(app_module))
        if response.status_code == 200:
            _pin(app_module, channel_id)
            admitted.append(channel_id)
        else:
            refused += 1
            assert response.status_code == 503, response.text
            assert response.headers["retry-after"] == "30"
            assert "ceiling" in response.json()["detail"]
    assert len(admitted) == 5 and refused == 35
    assert peak <= 5 and _live(app_module) == 5
    assert manager.refused_total == 35 and not manager._creating, "a refused request left a reservation behind"

    probe = client.get("/probes/readyz", params={"sessions": "true"})
    assert probe.status_code == 200, "being at the ceiling must not make the pod unready"
    sessions = probe.json()["sessions"]
    assert sessions["hard_ceiling"] == 5 and sessions["at_ceiling"] is True and sessions["refused_total"] == 35
    assert sessions["expired_total"] == 0 and sessions["pinned"] == 5


def test_an_ordinary_call_for_an_unknown_channel_is_refused_at_the_ceiling(app_module, client):
    _limits(app_module, soft=1, hard=2)
    _fill(app_module, client, 2)
    response = client.post("/invoke_assistant", headers=_token_for("someone_new"),
                           json={"user_query": "/get_properties", "timeout_seconds": 5})
    assert response.status_code == 503 and response.headers["retry-after"] == "30"
    assert _live(app_module) == 2


def test_the_mcp_tool_result_carries_the_same_refusal(app_module, client):
    _limits(app_module, soft=1, hard=2)
    _fill(app_module, client, 2)
    mcp = McpClient(client, _token_for("someone_new")["Authorization"])
    for name in ("invoke_assistant", "invoke_agent"):
        description = mcp.tools()[name]["description"]
        assert "live-session ceiling" in description and "session_notice" in description
    result = mcp.call("invoke_assistant", {"user_query": "/get_properties", "timeout_seconds": 5})
    assert result.get("isError") is True
    assert "ceiling" in str(result), result


def test_a_burst_of_concurrent_creators_never_passes_the_ceiling(app_module, client):
    manager = _limits(app_module, soft=2, hard=6)
    fills = _fill(app_module, client, 6)
    assert _live(app_module) == 6
    peak = 0

    async def creator(channel_id):
        try:
            await app_module.ensure_user_runtime_exists(
                channel_id=channel_id, session_manager=manager,
                workflow_path=app_module.ARGS.workflow_path, run_startup=False)
            return True
        except app_module.HTTPException as refusal:
            assert refusal.status_code == 503
            return False

    async def body():
        nonlocal peak
        done = asyncio.Event()

        async def sampler():
            nonlocal peak
            while not done.is_set():
                peak = max(peak, len(manager._sessions))
                await asyncio.sleep(0)

        watcher = asyncio.create_task(sampler())
        results = await asyncio.gather(*(creator(f"burst_{i}_{uuid.uuid4().hex[:6]}") for i in range(20)))
        done.set()
        await watcher
        return results

    results = _run(body())
    assert not any(results), "every slot was held by a pinned session, so nobody could be admitted"
    assert peak <= 6 and _live(app_module) == 6 and manager.refused_total == 20
    assert not manager._creating and len(fills) == 6


def test_a_checkpointable_workflow_is_never_refused_at_the_ceiling(app_module, client):
    from tests.test_fastapi_end_session import _give_state
    manager = _limits(app_module, soft=1, hard=2)
    for _ in range(6):
        channel_id = f"hook_{uuid.uuid4().hex[:8]}"
        assert client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid}).status_code == 200
        _give_state(app_module, channel_id)
    assert manager.refused_total == 0 and _live(app_module) <= 2


# -- expiry ---------------------------------------------------------------------------------------------------------

def test_with_defaults_nothing_is_expired_at_or_below_the_soft_target(app_module, client):
    manager = _limits(app_module, soft=3, hard=10, ttl=1, protected_ttl=1)
    channels = [c for c, *_ in _fill(app_module, client, 3)]
    for channel_id in channels:
        _age(app_module, channel_id, 10**6)
    assert _run(manager.expire_idle_under_pressure()) == 0
    # Even a request that is admitted never expires anything below the ceiling.
    assert client.post("/initialize", json={"channel_id": "fourth", "user_id": client.uid}).status_code == 200
    assert manager.expired_total == 0 and _live(app_module) == 4


def test_idle_pinned_sessions_are_expired_only_above_the_soft_target(app_module, client):
    manager = _limits(app_module, soft=2, hard=4, ttl=60)
    channels = [c for c, *_ in _fill(app_module, client, 4)]
    for channel_id in channels[:3]:
        _age(app_module, channel_id, 120)
    # At the ceiling a newcomer needs room: the longest-idle pinned session goes, and the newcomer comes in.
    _age(app_module, channels[0], 500)
    response = client.post("/initialize", json={"channel_id": "newcomer", "user_id": client.uid})
    assert response.status_code == 200, response.text
    assert _runtime(app_module, channels[0]) is None and manager.expired_total == 1
    assert _live(app_module) == 4
    # The reaper's pass brings the count back toward the soft target and stops there.
    assert _run(manager.expire_idle_under_pressure()) == 2
    assert _live(app_module) == 2 and manager.expired_total == 3
    assert _run(manager.expire_idle_under_pressure()) == 0


def test_a_zero_ttl_never_expires(app_module, client):
    manager = _limits(app_module, soft=1, hard=2, ttl=0, protected_ttl=0)
    channels = [c for c, *_ in _fill(app_module, client, 2)]
    for channel_id in channels:
        _age(app_module, channel_id, 10**7)
    assert _run(manager.expire_idle_under_pressure()) == 0
    assert client.post("/initialize", json={"channel_id": "late", "user_id": client.uid}).status_code == 503


def test_busy_and_leased_sessions_are_never_expired(app_module, client):
    manager = _limits(app_module, soft=1, hard=2, ttl=1)
    first, second = [c for c, *_ in _fill(app_module, client, 2)]
    for channel_id in (first, second):
        _age(app_module, channel_id, 1000)

    async def body():
        async with manager.leased_session(first):
            return await manager.expire_idle_under_pressure()

    assert _run(body()) == 1
    assert _runtime(app_module, first) is not None and _runtime(app_module, second) is None

    runtime = _runtime(app_module, first)
    _age(app_module, first, 1000)

    async def locked():
        async with runtime.lock:  # a running turn holds the runtime lock
            return await manager.expire_idle_under_pressure()

    assert _run(locked()) == 0 and _runtime(app_module, first) is not None


def test_a_pending_question_is_protected_until_the_longer_ttl_and_then_discarded(app_module, client):
    manager = _limits(app_module, soft=1, hard=2, ttl=60, protected_ttl=5000)
    first, second = [c for c, *_ in _fill(app_module, client, 2)]
    from fastworkflow.run_fastapi_mcp import utils
    _make_pending(app_module, first)
    assert utils.pending_kind(_runtime(app_module, first).execution_context) is not None
    _age(app_module, first, 1000)
    _age(app_module, second, 1000)
    assert _run(manager.expire_idle_under_pressure()) == 1
    assert _runtime(app_module, second) is None and _runtime(app_module, first) is not None, \
        "the pending-question session outlived the normal TTL"

    # Once its own, longer TTL has passed the abandoned question stops being a reason to keep the session, but
    # still only under pressure: at the soft target nothing expires.
    _age(app_module, first, 10_000)
    assert _run(manager.expire_idle_under_pressure()) == 0, "below the soft target nothing expires"
    assert client.post("/initialize", json={"channel_id": "x1", "user_id": client.uid}).status_code == 200
    assert _run(manager.expire_idle_under_pressure()) == 1
    assert _runtime(app_module, first) is None
    assert manager.session_state_store.exists(first) is False, "the abandoned question was not discarded"
    notice = manager.notice_store.take(first)
    assert notice["reason"] == session_notices.REASON_EXPIRED_PENDING


def _make_pending(app_module, channel_id):
    """Park a real suspended question on the channel's session and save it, as a finished turn would."""
    from fastworkflow.run_fastapi_mcp.utils import persist_pending_after_turn
    runtime = _runtime(app_module, channel_id)
    ctx = runtime.execution_context
    ctx._awaiting_user = True
    persist_pending_after_turn(app_module.session_manager, runtime, None)


def test_registered_experiment_channels_are_exempt_and_counted(app_module, client):
    manager = _limits(app_module, soft=1, hard=2, ttl=1)
    first, second = [c for c, *_ in _fill(app_module, client, 2)]
    ctx = _runtime(app_module, first).execution_context
    ctx._experiment_id = "exp"  # what bind_experiment_claim sets: the session belongs to a registered attempt
    for channel_id in (first, second):
        _age(app_module, channel_id, 10**6)
    assert _run(manager.expire_idle_under_pressure()) == 1
    assert _runtime(app_module, first) is not None and _runtime(app_module, second) is None
    sessions = manager.session_counts(assess_pinned=True)
    assert sessions["experiment_exempt"] == 1


# -- the notice -----------------------------------------------------------------------------------------------------

def test_an_expired_channel_is_told_once_on_its_next_call(app_module, client):
    manager = _limits(app_module, soft=1, hard=2, ttl=60)
    first, second = _fill(app_module, client, 2)
    _age(app_module, first[0], 1000)
    assert _run(manager.expire_idle_under_pressure()) == 1
    # Another channel takes the slot, so the old channel's call has to make room again.
    _finish_next = client.post("/invoke_assistant", headers=first[1],
                               json={"user_query": "/get_properties", "timeout_seconds": 30})
    assert _finish_next.status_code in (200, 202), _finish_next.text
    body = _finish_next.json()
    if _finish_next.status_code == 202:
        body = client.get(f"/turns/{body['turn_key']}", headers=first[1]).json()
    notice = body.get("session_notice")
    assert notice and notice["code"] == "context_dropped" and notice["reason"] == session_notices.REASON_EXPIRED
    assert "launch configuration" in notice["message"]

    again = client.post("/invoke_assistant", headers=first[1],
                        json={"user_query": "/get_properties 2", "timeout_seconds": 30})
    assert "session_notice" not in again.json(), "the notice must be shown once"


def test_end_session_of_a_pinned_channel_leaves_a_notice_and_initialize_delivers_it(app_module, client):
    channel_id, headers, _wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)
    assert client.post("/end_session", headers=headers).json()["pinned"] is True
    init = client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid}).json()
    assert init["session_notice"]["reason"] == session_notices.REASON_ENDED
    again = client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid}).json()
    assert again.get("session_notice") is None


def test_a_checkpointed_end_leaves_no_notice(app_module, client):
    from tests.test_fastapi_end_session import _give_state
    channel_id = f"ck_{uuid.uuid4().hex[:8]}"
    init = client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid}).json()
    _give_state(app_module, channel_id)
    assert client.post("/end_session", headers=_auth(init)).json()["pinned"] is False
    again = client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid}).json()
    assert again.get("session_notice") is None


def test_an_unreadable_notice_store_does_not_fail_the_request(app_module, client, tmp_path):
    manager = _limits(app_module, soft=1, hard=2, ttl=60)
    channel_id, headers, _wf, _ctx = _new_channel(app_module, client, client.uid, pinned=True)
    client.post("/end_session", headers=headers)
    store = manager.notice_store
    path = store._path(channel_id)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{ not json")
    response = client.post("/initialize", json={"channel_id": channel_id, "user_id": client.uid})
    assert response.status_code == 200 and response.json().get("session_notice") is None


def test_notice_records_are_reaped_by_age(app_module):
    manager = app_module.session_manager
    manager.notice_store.record("old_channel", session_notices.REASON_EXPIRED)
    assert manager.notice_store.reap(max_age_seconds=3600) == 0
    assert manager.notice_store.reap(max_age_seconds=-1) == 1
    assert manager.notice_store.take("old_channel") is None


# -- configuration --------------------------------------------------------------------------------------------------

def test_resolvers_default_validate_and_prefer_the_process_environment(monkeypatch):
    for name in ("MAX_LIVE_SESSIONS_HARD", "PINNED_SESSION_IDLE_TTL_SECONDS", "PROTECTED_SESSION_IDLE_TTL_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    assert resolve_max_live_sessions_hard(50)[0] == 200
    assert resolve_pinned_idle_ttl()[0] == 1800 and resolve_protected_idle_ttl()[0] == 86400
    monkeypatch.setenv("MAX_LIVE_SESSIONS_HARD", "75")
    assert resolve_max_live_sessions_hard(50) == (75, "process environment")
    monkeypatch.setenv("MAX_LIVE_SESSIONS_HARD", "10")
    with pytest.raises(ValueError, match="must not be below"):
        resolve_max_live_sessions_hard(50)
    monkeypatch.setenv("PINNED_SESSION_IDLE_TTL_SECONDS", "0")
    assert resolve_pinned_idle_ttl()[0] == 0
    monkeypatch.setenv("PINNED_SESSION_IDLE_TTL_SECONDS", "-5")
    with pytest.raises(ValueError):
        resolve_pinned_idle_ttl()
    monkeypatch.setenv("PROTECTED_SESSION_IDLE_TTL_SECONDS", "soon")
    with pytest.raises(ValueError, match="not an integer"):
        resolve_protected_idle_ttl()


def test_the_startup_line_names_every_limit_and_its_source(monkeypatch):
    for name in ("MAX_LIVE_SESSIONS", "MAX_LIVE_SESSIONS_HARD", "PINNED_SESSION_IDLE_TTL_SECONDS",
                 "PROTECTED_SESSION_IDLE_TTL_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_LIVE_SESSIONS_HARD", "120")
    manager = ChannelSessionManager()
    manager.configure_max_live_sessions()
    line = manager.limits_summary()
    assert "max_live_sessions=50 (source=" in line
    assert "max_live_sessions_hard=120 (source=process environment)" in line
    assert "pinned_session_idle_ttl_seconds=1800 (source=default)" in line
    assert "protected_session_idle_ttl_seconds=86400 (source=default)" in line


# -- bounded retained data ------------------------------------------------------------------------------------------

def test_the_trace_queue_drops_the_oldest_and_counts(app_module):
    q = BoundedTraceQueue(maxlen=3)
    for i in range(10):
        q.put(i)
    assert [q.get_nowait() for _ in range(3)] == [7, 8, 9] and q.dropped == 7


def test_a_failed_turn_leaves_nothing_in_the_trace_queue(app_module, client):
    channel_id, headers, _wf, ctx = _new_channel(app_module, client, client.uid, pinned=True)

    def failing(message, *, unattended=False, approval_surface=None, approval_token=None):
        ctx.command_trace_queue.put("left behind by a failed turn")
        raise RuntimeError("boom")

    ctx.process_turn = failing
    client.post("/invoke_assistant", headers=headers, json={"user_query": "x", "timeout_seconds": 10})
    assert ctx.command_trace_queue.qsize() == 0


def test_the_session_exposes_its_trace_queue_drop_count(app_module, client):
    _limits(app_module, soft=5, hard=10)
    channel_id, *_ = _new_channel(app_module, client, client.uid, pinned=True)
    queue_ = _runtime(app_module, channel_id).execution_context.command_trace_queue
    assert isinstance(queue_, BoundedTraceQueue)
    sessions = client.get("/probes/readyz", params={"sessions": "true"}).json()["sessions"]
    assert sessions["trace_events_dropped"] == 0


def test_a_deferred_window_trim_is_capped(app_module, client):
    from fastworkflow.run_fastapi_mcp.utils import MAX_CONVERSATION_TURNS_IN_MEMORY, trim_conversation_window
    import logging
    channel_id, headers, _wf, ctx = _new_channel(app_module, client, client.uid, pinned=True)
    runtime = _runtime(app_module, channel_id)
    for i in range(2 * MAX_CONVERSATION_TURNS_IN_MEMORY + 5):
        ctx.append_conversation_turn(f"turn {i}", None)
    ctx._last_turn_record_stored = False  # the last record is queued, not durable: the trim defers
    trimmed = trim_conversation_window(runtime, logging.getLogger("t"))
    assert trimmed > 0, "past twice the window the trim must go ahead even though the record is not durable"
    assert len(ctx.conversation_history.messages) <= 2 * MAX_CONVERSATION_TURNS_IN_MEMORY * 2 + 2
