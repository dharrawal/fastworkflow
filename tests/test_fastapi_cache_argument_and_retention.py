"""A request's ``use_dspy_cache`` cannot change process-wide state or server health, and finished turns
do not accumulate in the turn registry.

Real app, real turn registry, no model: a turn is made model-free by wrapping the channel's ``process_turn``
around a free action, as the other server tests do (tests/test_fastapi_end_session.py).
"""
from __future__ import annotations

import dspy
import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.run_fastapi_mcp import server_memory, utils
from fastworkflow.run_fastapi_mcp.turns import MAX_RETAINED_STARTUP_TURNS
from tests.test_approval_endpoint_surfaces import (  # noqa: F401  (fixtures and helpers)
    _configure, app_module, no_live_model)
from tests.test_fastapi_end_session import (  # noqa: F401
    FREE_ACTION, _new_channel, _finish_a_turn, client)


def _cache_state():
    cache = dspy.cache
    return (id(cache), cache.enable_disk_cache, cache.enable_memory_cache,
            getattr(cache.memory_cache, "maxsize", None))


@pytest.fixture
def env():
    saved = dict(fastworkflow._env_vars)
    yield fastworkflow._env_vars
    fastworkflow._env_vars.clear()
    fastworkflow._env_vars.update(saved)


@pytest.fixture
def policy_in_force(monkeypatch):
    """The readiness probe's cache half armed, as under the server entrypoint (the DSPy settings half is
    left out: installing it would claim DSPy's config owner for the whole test process)."""
    status = server_memory.PolicyStatus(
        dspy_version="test", memory_cache_entries=server_memory._memory_cache_maxsize())
    monkeypatch.setattr(server_memory, "_status", status)
    monkeypatch.setattr(server_memory, "POLICY_SETTINGS", {})
    assert server_memory.check_policy_in_force() is None


@pytest.mark.parametrize("sent", [False, True])
def test_the_argument_changes_nothing_process_wide_and_the_probe_stays_ready(
        app_module, client, policy_in_force, env, sent):
    env["FW_LM_CACHE"] = "1"
    _a, headers_a, _wf, ctx_a = _new_channel(app_module, client, client.uid, pinned=True)
    _b, headers_b, _wf, ctx_b = _new_channel(app_module, client, client.uid, pinned=True)
    before = _cache_state()

    for ctx, headers, extra in ((ctx_a, headers_a, {"use_dspy_cache": sent}), (ctx_b, headers_b, {})):
        ctx.process_turn = lambda message, ctx=ctx, **kw: ctx.process_action_turn(
            fastworkflow.Action(**FREE_ACTION), approval_surface=kw.get("approval_surface"))
        response = client.post("/invoke_assistant", headers=headers,
                               json={"user_query": "show the list", "timeout_seconds": 30, **extra})
        assert response.status_code == 200, response.text
        assert response.json()["exec_state"] == "done"
        # Only a value that disagrees with the (on) server is reported.
        assert ("X-FW-DSPy-Cache" in response.headers) is (extra == {"use_dspy_cache": False})
        assert _cache_state() == before
        ready = client.get("/probes/readyz")
        assert ready.status_code == 200, ready.text
        assert "drifted" not in str(ready.json()["checks"].get("dspy_memory_policy", ""))
    assert server_memory.check_policy_in_force() is None


def test_the_stream_endpoint_ignores_a_disagreeing_value_and_says_so(app_module, client, policy_in_force, env):
    env["FW_LM_CACHE"] = "1"
    _c, headers, _wf, ctx = _new_channel(app_module, client, client.uid, pinned=True)
    ctx.process_turn = lambda message, **kw: ctx.process_action_turn(
        fastworkflow.Action(**FREE_ACTION), approval_surface=kw.get("approval_surface"))
    before = _cache_state()
    response = client.post("/invoke_agent_stream", headers=headers,
                           json={"user_query": "show the list", "timeout_seconds": 30, "use_dspy_cache": False})
    assert response.status_code == 200, response.text
    assert "ignored" in response.headers["X-FW-DSPy-Cache"]
    assert "FW_LM_CACHE=0" in response.headers["X-FW-DSPy-Cache"]
    assert _cache_state() == before
    assert client.get("/probes/readyz").status_code == 200


@pytest.fixture
def toggle_allowed(monkeypatch):
    utils.set_allow_request_cache_toggle(True)
    yield
    utils.set_allow_request_cache_toggle(False)
    server_memory._cache_off_by_request = False
    dspy.configure_cache(enable_memory_cache=True, memory_max_entries=server_memory.SERVER_DSPY_MEMORY_CACHE_ENTRIES,
                         enable_disk_cache=True)


@pytest.mark.parametrize("sent", [False, True, False])
def test_with_the_opt_in_the_argument_toggles_the_cache_as_before_and_readiness_stays_ready(
        app_module, client, policy_in_force, toggle_allowed, sent):
    _c, headers, _wf, ctx = _new_channel(app_module, client, client.uid, pinned=True)
    ctx.process_turn = lambda message, **kw: ctx.process_action_turn(
        fastworkflow.Action(**FREE_ACTION), approval_surface=kw.get("approval_surface"))
    response = client.post("/invoke_assistant", headers=headers,
                           json={"user_query": "show the list", "timeout_seconds": 30, "use_dspy_cache": sent})
    assert response.status_code == 200, response.text
    assert "X-FW-DSPy-Cache" not in response.headers
    assert dspy.cache.enable_disk_cache is sent and dspy.cache.enable_memory_cache is sent
    ready = client.get("/probes/readyz")
    assert ready.status_code == 200, ready.text
    checks = ready.json()["checks"]
    assert "drifted" not in str(checks.get("dspy_memory_policy", ""))
    assert ("dspy_response_cache" in checks) is (not sent)
    assert server_memory.check_policy_in_force() is None
    if sent:  # back on: rebuilt with the same bound the server installs
        assert server_memory._memory_cache_maxsize() == server_memory.SERVER_DSPY_MEMORY_CACHE_ENTRIES


def test_the_server_wide_setting_is_the_only_thing_the_notice_is_measured_against(env):
    env["FW_LM_CACHE"] = "0"
    assert utils.apply_dspy_cache_choice(False) is None      # agrees with a cache-off server
    assert utils.apply_dspy_cache_choice(None) is None
    notice = utils.apply_dspy_cache_choice(True)             # disagrees: ignored, and says how to change it
    assert notice and "off" in notice and "FW_LM_CACHE=1" in notice
    env["FW_LM_CACHE"] = "1"
    assert utils.apply_dspy_cache_choice(True) is None
    assert "FW_LM_CACHE=0" in utils.apply_dspy_cache_choice(False)


def test_a_turn_per_channel_with_end_session_leaves_no_more_than_the_bound_in_memory(app_module, client):
    channels = MAX_RETAINED_STARTUP_TURNS + 10
    for _ in range(channels):
        _c, headers, _wf, ctx = _new_channel(app_module, client, client.uid, pinned=True)
        _finish_a_turn(client, headers, ctx)
        assert client.post("/end_session", headers=headers).status_code == 200
        assert len(app_module.turn_registry._by_key) <= MAX_RETAINED_STARTUP_TURNS
    retained = client.get("/probes/readyz?memory=true").json()["memory"]["retained_turns"]
    assert retained <= MAX_RETAINED_STARTUP_TURNS
    # Chat turns are dropped the moment they finish (the store holds them): nothing grows with the count.
    assert retained == 0
