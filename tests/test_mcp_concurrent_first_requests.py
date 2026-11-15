"""Concurrent first requests through the MCP mount (a load run found HTTP 500 on a fresh server).

fastapi-mcp starts its StreamableHTTP session manager lazily on the first request, and marks it started before its
task group exists, so concurrent first requests raced with that start ("Task group is not initialized"). The app
lifespan now starts the manager before the server is ready. These tests start the real app on a fresh module each
time and fire many first requests at once, with no model.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging

import pytest
from fastapi.testclient import TestClient

from tests.test_approval_endpoint_surfaces import (  # noqa: F401  (fixtures and helpers)
    _configure, _session, app_module, no_live_model)

CLIENTS = 16
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "pytest", "version": "0"}}}


def _first_request(client: TestClient, bearer: str):
    headers = {"Authorization": bearer, "Accept": "application/json, text/event-stream",
               "Content-Type": "application/json"}
    r = client.post("/mcp", headers=headers, json=INIT)
    return r.status_code, r.json() if r.status_code == 200 else r.text


@pytest.mark.parametrize("round_", range(3))
def test_concurrent_first_mcp_requests_all_succeed_on_a_fresh_app(app_module, monkeypatch, caplog, round_):
    uid = _configure(monkeypatch, "unsigned", "channel")
    with caplog.at_level(logging.ERROR):
        with TestClient(app_module.app) as client:
            _channel, headers, _wf, _chores, _ctx = _session(app_module, client, uid)
            bearer = headers["Authorization"]
            with cf.ThreadPoolExecutor(CLIENTS) as pool:
                outcomes = list(pool.map(lambda _i: _first_request(client, bearer), range(CLIENTS)))
    bad = [o for o in outcomes if o[0] != 200]
    assert not bad, bad[:3]
    assert all("result" in body and body["result"].get("serverInfo") for _s, body in outcomes), outcomes[:2]
    noisy = [r.getMessage() for r in caplog.records
             if "Task group" in r.getMessage() or "Error in StreamableHTTP" in r.getMessage()
             or "cancel scope" in r.getMessage().lower()]
    assert not noisy, noisy


@pytest.mark.parametrize("round_", range(3))
def test_first_requests_queued_on_one_event_loop_all_succeed(app_module, monkeypatch, caplog, round_):
    """The deterministic form of the race: N first requests become ready on ONE loop at once (what a loaded
    server sees), so every one runs before the library's own start task gets its first step. Lifespan entered
    by hand on that loop, requests over the real ASGI app."""
    import asyncio
    import httpx
    _configure(monkeypatch, "unsigned", "channel")
    with TestClient(app_module.app) as setup_client:
        _c, headers, *_rest = _session(app_module, setup_client, "alice")
    bearer = headers["Authorization"]
    request_headers = {"Authorization": bearer, "Accept": "application/json, text/event-stream",
                       "Content-Type": "application/json"}

    async def run():
        async with app_module.app.router.lifespan_context(app_module.app):
            transport = httpx.ASGITransport(app=app_module.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await asyncio.gather(*[client.post("/mcp", headers=request_headers, json=INIT)
                                              for _ in range(CLIENTS)])

    with caplog.at_level(logging.ERROR):
        responses = asyncio.run(run())
    assert [r.status_code for r in responses] == [200] * CLIENTS, [r.status_code for r in responses]
    assert all(r.json()["result"]["serverInfo"] for r in responses)
    noisy = [r.getMessage() for r in caplog.records
             if "Task group" in r.getMessage() or "Error in StreamableHTTP" in r.getMessage()
             or "cancel scope" in r.getMessage().lower()]
    assert not noisy, noisy


def test_the_session_manager_is_running_before_any_request_and_stops_with_the_lifespan(app_module, monkeypatch):
    from fastworkflow.run_fastapi_mcp import mcp_specific
    _configure(monkeypatch, "unsigned", "channel")
    transport = app_module.mcp_mount._http_transport
    assert transport._session_manager is None
    with TestClient(app_module.app):
        assert transport._session_manager._task_group is not None      # started by the lifespan, not by a request
        assert transport._manager_task is not None and not transport._manager_task.done()
    assert transport._manager_started is False and transport._manager_task.done()
    assert mcp_specific.MCP_TRANSPORT_START_SECONDS > 0


def test_a_library_without_the_transport_hooks_fails_loudly_at_startup():
    import asyncio
    from fastworkflow.run_fastapi_mcp.mcp_specific import start_mcp_transport

    class FutureLibrary:      # a FastApiMCP whose transport hooks were renamed
        _http_transport = object()

    with pytest.raises(RuntimeError, match="no longer has the HTTP transport hooks"):
        asyncio.run(start_mcp_transport(FutureLibrary()))


def test_a_second_lifespan_of_the_same_app_serves_mcp(app_module, monkeypatch):
    """Shutdown resets the manager, so entering the lifespan again restarts it (as tests do in one process)."""
    _configure(monkeypatch, "unsigned", "channel")
    with TestClient(app_module.app) as client:
        _c, headers, *_rest = _session(app_module, client, "alice")
        bearer = headers["Authorization"]
        assert _first_request(client, bearer)[0] == 200
    with TestClient(app_module.app) as client:
        status, body = _first_request(client, bearer)
        assert status == 200 and body["result"]["serverInfo"], (status, body)
