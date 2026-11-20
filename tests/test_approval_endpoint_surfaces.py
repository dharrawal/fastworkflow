"""The approval surface of each HTTP turn endpoint (fix-rkr1, steps 1 and 2).

/invoke_agent and /perform_action are always a host surface. /invoke_agent_stream (also the MCP
invoke_agent tool) and /invoke_assistant follow ``--chat_approval_surface``. Since step 2 the default is
``host`` (the same rule as /invoke_agent); ``channel`` is the lenient behaviour before the option existed
and is what the local run_chatbot server pins explicitly.

The truth table here is the one in the fix-rkr1 investigation. Every ``channel`` row asserts today's
behaviour exactly, and every ``host`` row of the stream and assistant endpoints must equal the
/invoke_agent row.

No model is called. /perform_action and /invoke_assistant ("/"-prefixed assistant command) ask for the
approval for real. /invoke_agent and /invoke_agent_stream ask through the agent, which needs a model; for
them one seam is used: the channel's ``process_turn`` records the surface and token the endpoint passed
and then runs the same gate on a real, model-free direct action with exactly that surface. So the endpoint
wiring (what it passes) is observed, and the decision is the real ``decide``.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fastworkflow import approval_binding as ab
from tests.test_approval_binding import _annotate
from tests.test_approval_fastapi import (
    MANDATORY, TODO, _COPY_IGNORE, _auth, _setup_todo_context)

ACTION = {"command_name": MANDATORY, "command": "set_properties assign_to=Ann",
          "parameters": {"assign_to": "Ann", "is_complete": None}}
ENDPOINTS = ("invoke_agent", "perform_action", "invoke_agent_stream", "invoke_assistant")
JWT_MODES = ("unsigned", "signed_uid", "signed_no_uid")
OPTIONS = ("channel", "host")


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    import litellm

    def _refuse(*args, **kwargs):
        raise AssertionError("a live model call was attempted from an approval test")

    monkeypatch.setattr(litellm, "completion", _refuse)
    monkeypatch.setattr(litellm, "acompletion", _refuse)


@pytest.fixture
def app_module(tmp_path, monkeypatch, env_files):
    env_file, passwords_file = env_files
    from fastworkflow.run_fastapi_mcp import jwt_manager, utils
    monkeypatch.setattr(jwt_manager, "EXPECT_ENCRYPTED_JWT", True)
    # Start every test from the server's real default (restored afterwards), not from a forced value.
    monkeypatch.setattr(utils, "_chat_approval_surface", utils.DEFAULT_CHAT_APPROVAL_SURFACE)
    folder = tmp_path / "todo_annotated"
    shutil.copytree(TODO, folder, ignore=_COPY_IGNORE, symlinks=True)
    _annotate(folder, "TodoList/set_properties.py", "REQUIRED_ALWAYS")
    sys.argv = ["pytest", "--workflow_path", str(folder), "--env_file_path", env_file,
                "--passwords_file_path", passwords_file]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts")
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def _configure(monkeypatch, jwt_mode: str, option: str) -> str | None:
    """Set the server's JWT mode and option the way startup does; the uid the token will carry."""
    from fastworkflow.run_fastapi_mcp import jwt_manager, utils
    monkeypatch.setattr(jwt_manager, "EXPECT_ENCRYPTED_JWT", jwt_mode != "unsigned")
    utils.set_chat_approval_surface(option)
    return None if jwt_mode == "signed_no_uid" else "alice"


def _session(app_module, client, uid):
    channel_id = f"surf_{uuid.uuid4().hex[:8]}"
    payload = {"channel_id": channel_id}
    if uid is not None:
        payload["user_id"] = uid
    init = client.post("/initialize", json=payload).json()
    wf, chores = _setup_todo_context(app_module, channel_id)
    runtime = asyncio.run(app_module.session_manager.get_session(channel_id))
    return channel_id, _auth(init), wf, chores, runtime.execution_context


def _install_seam(ctx) -> list:
    """Record what an agent-driven endpoint passes to process_turn, then run the gate for real."""
    seen: list = []

    def seam(message, *, unattended=False, approval_surface=None, approval_token=None):
        seen.append({"message": message, "surface": approval_surface, "token": approval_token,
                     "unattended": unattended})
        return ctx.process_action_turn(fastworkflow_action(), approval_surface=approval_surface)

    ctx.process_turn = seam
    return seen


def fastworkflow_action():
    import fastworkflow
    return fastworkflow.Action(**ACTION)


def _post(client, headers, endpoint: str, ctx, **body) -> dict:
    """One request; the response body as a dict (the output frame's data for the stream)."""
    body = {"timeout_seconds": 20, **body}
    if endpoint == "perform_action":
        r = client.post("/perform_action", headers=headers, json={"action": ACTION, **body})
    elif endpoint == "invoke_assistant":
        body.setdefault("user_query", "set_properties assign_to=Ann")
        r = client.post("/invoke_assistant", headers=headers, json=body)
    elif endpoint == "invoke_agent":
        body.setdefault("user_query", "change the assignee")
        r = client.post("/invoke_agent", headers=headers, json=body)
    else:
        body.setdefault("user_query", "change the assignee")
        r = client.post("/invoke_agent_stream", headers=headers, json=body)
        assert r.status_code == 200, r.text
        frames = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        outputs = [f for f in frames if f["type"] == "output"]
        assert len(outputs) == 1, frames
        return outputs[0]["data"]
    assert r.status_code == 200, r.text
    return r.json()


def _ask(client, headers, endpoint: str, ctx) -> dict:
    if endpoint in ("invoke_agent", "invoke_agent_stream"):
        _install_seam(ctx)
    return _post(client, headers, endpoint, ctx)


def _decision(data: dict) -> str:
    if "approval" in data:
        return "ASK+token"
    if ab.MSG_IDENTITY[:40] in data["answer"]:
        return "REFUSE"
    assert "run" in data["answer"] and "cannot be undone" in data["answer"], data["answer"]
    return "ASK"


def _expected(endpoint: str, jwt_mode: str, option: str) -> str:
    """The fix-rkr1 truth table: today's cells, with the option applied to the two chat endpoints only."""
    host = endpoint in ("invoke_agent", "perform_action") or option == "host"
    if not host:
        return "ASK"            # channel: asked in every JWT mode, a plain yes approves, no token
    return "ASK+token" if jwt_mode == "signed_uid" else "REFUSE"


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("jwt_mode", JWT_MODES)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_truth_table(app_module, monkeypatch, endpoint, jwt_mode, option):
    uid = _configure(monkeypatch, jwt_mode, option)
    client = TestClient(app_module.app)
    _channel, headers, _wf, chores, ctx = _session(app_module, client, uid)

    data = _ask(client, headers, endpoint, ctx)

    assert _decision(data) == _expected(endpoint, jwt_mode, option)
    assert chores.assign_to == ""
    if "approval" in data:
        assert data["approval"]["token"] not in data["answer"]


@pytest.mark.parametrize("jwt_mode", JWT_MODES)
@pytest.mark.parametrize("endpoint", ("invoke_agent_stream", "invoke_assistant"))
def test_host_rows_of_the_chat_endpoints_equal_the_invoke_agent_row(app_module, monkeypatch, endpoint, jwt_mode):
    uid = _configure(monkeypatch, jwt_mode, "host")
    client = TestClient(app_module.app)
    _c, headers, _wf, _chores, ctx = _session(app_module, client, uid)
    chat = _decision(_ask(client, headers, endpoint, ctx))
    _c, headers2, _wf2, _chores2, ctx2 = _session(app_module, client, uid)
    assert chat == _decision(_ask(client, headers2, "invoke_agent", ctx2))


@pytest.mark.parametrize("jwt_mode", JWT_MODES)
@pytest.mark.parametrize("option", OPTIONS)
def test_the_option_never_changes_the_two_direct_endpoints(app_module, monkeypatch, jwt_mode, option):
    """The helper's "direct" scope is the expression the two endpoints built inline before: a host
    surface, verified exactly when signatures are checked, carrying the token's uid."""
    from fastworkflow.run_fastapi_mcp import jwt_manager
    from fastworkflow.run_fastapi_mcp.utils import approval_surface_for
    uid = _configure(monkeypatch, jwt_mode, option)
    expected = ab.Surface(kind="host", verified=bool(jwt_manager.EXPECT_ENCRYPTED_JWT), uid=uid)
    assert approval_surface_for(uid) == expected
    assert approval_surface_for(uid, scope="direct") == expected
    chat = approval_surface_for(uid, scope="chat")
    assert chat == (expected if option == "host" else ab.Surface(kind="channel", origin="chat"))


def test_the_default_option_is_host_and_the_stream_passes_the_host_surface_and_the_token(app_module, monkeypatch):
    """Step 2: under the default the stream endpoint passes the same host surface /invoke_agent builds."""
    from fastworkflow.run_fastapi_mcp import utils
    assert utils.DEFAULT_CHAT_APPROVAL_SURFACE == "host"
    assert app_module.ARGS.chat_approval_surface == "host"      # argparse default, no flag given
    assert utils.chat_approval_surface() == "host"               # module default, main() not run
    uid = _configure_default(monkeypatch, "signed_uid")
    assert utils.chat_approval_surface() == "host"
    client = TestClient(app_module.app)
    _c, headers, _wf, _chores, ctx = _session(app_module, client, uid)
    seen = _install_seam(ctx)
    _post(client, headers, "invoke_agent_stream", ctx, approval_token="tok-1")
    assert seen == [{"message": "change the assignee", "token": "tok-1", "unattended": False,
                     "surface": ab.Surface(kind="host", verified=True, uid="alice")}]


def _configure_default(monkeypatch, jwt_mode: str) -> str | None:
    """Like _configure, but leaves the approval option at the server's default (nothing sets it)."""
    from fastworkflow.run_fastapi_mcp import jwt_manager
    monkeypatch.setattr(jwt_manager, "EXPECT_ENCRYPTED_JWT", jwt_mode != "unsigned")
    return None if jwt_mode == "signed_no_uid" else "alice"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("jwt_mode", JWT_MODES)
def test_a_default_server_applies_one_rule_on_every_endpoint(app_module, monkeypatch, endpoint, jwt_mode):
    """No option is set anywhere: all four endpoints give the /invoke_agent answer. Signed with a uid asks
    and delivers a token; anything else is refused (and nothing runs)."""
    uid = _configure_default(monkeypatch, jwt_mode)
    client = TestClient(app_module.app)
    _c, headers, _wf, chores, ctx = _session(app_module, client, uid)
    data = _ask(client, headers, endpoint, ctx)
    assert _decision(data) == ("ASK+token" if jwt_mode == "signed_uid" else "REFUSE")
    assert chores.assign_to == ""


@pytest.mark.parametrize("endpoint", ("invoke_agent_stream", "invoke_assistant"))
def test_channel_option_keeps_the_lenient_behaviour_on_an_unsigned_server(app_module, monkeypatch, endpoint):
    """--chat_approval_surface channel: asked, no token, a plain yes approves (the pre-step-2 behaviour)."""
    uid = _configure(monkeypatch, "unsigned", "channel")
    client = TestClient(app_module.app)
    _c, headers, _wf, chores, ctx = _session(app_module, client, uid)
    data = _ask(client, headers, endpoint, ctx)
    assert _decision(data) == "ASK" and "approval" not in data
    if endpoint == "invoke_agent_stream":
        ctx.__dict__.pop("process_turn", None)
    done = _post(client, headers, endpoint, ctx, user_query="yes")
    assert chores.assign_to == "Ann", done


def test_channel_option_passes_the_chat_origin_channel_surface_and_no_token(app_module, monkeypatch):
    """The channel surface of the chat server carries the chat origin (fix-d6no); the token never travels with it."""
    uid = _configure(monkeypatch, "signed_uid", "channel")
    client = TestClient(app_module.app)
    _c, headers, _wf, _chores, ctx = _session(app_module, client, uid)
    seen = _install_seam(ctx)
    _post(client, headers, "invoke_agent_stream", ctx, approval_token="ignored-under-channel")
    assert seen == [{"message": "change the assignee", "surface": ab.Surface(kind="channel", origin="chat"),
                     "token": None, "unattended": False}]


@pytest.mark.parametrize("endpoint", ("invoke_agent_stream", "invoke_assistant"))
def test_unattended_on_an_unsigned_default_server_is_refused_not_queued(app_module, monkeypatch, endpoint):
    """Owner-accepted consequence: an unattended HTTP run on an unsigned server gets the identity refusal,
    where the channel surface used to queue a visible record. Nothing is queued and nothing runs."""
    uid = _configure_default(monkeypatch, "unsigned")
    client = TestClient(app_module.app)
    _c, headers, wf, chores, ctx = _session(app_module, client, uid)
    if endpoint == "invoke_agent_stream":
        seen = _install_seam(ctx)
    data = _post(client, headers, endpoint, ctx, unattended=True)
    assert _decision(data) == "REFUSE"
    if endpoint == "invoke_agent_stream":
        assert seen[0]["unattended"] is True
    assert chores.assign_to == ""
    assert ab.take_host_delivery(wf) == {}
    # Contrast: the same unattended request under channel is not a refusal.
    uid = _configure(monkeypatch, "unsigned", "channel")
    _c2, headers2, _wf2, chores2, ctx2 = _session(app_module, client, uid)
    if endpoint == "invoke_agent_stream":
        _install_seam(ctx2)
    lenient = _post(client, headers2, endpoint, ctx2, unattended=True)
    assert ab.MSG_IDENTITY[:40] not in lenient["answer"] and chores2.assign_to == ""


def _mcp_call(app_module, tool: str, arguments: dict, headers: dict):
    """One MCP tool call through fastapi-mcp's own executor, over the app in process (the transport the
    library uses: httpx ASGITransport, only the Authorization header forwarded). No MCP session needed."""
    import httpx
    from fastapi_mcp import FastApiMCP
    from fastapi_mcp.types import HTTPRequestInfo

    async def run():
        mcp = FastApiMCP(app_module.app)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://apiserver")
        info = HTTPRequestInfo(method="POST", path="/mcp", headers={"authorization": headers["Authorization"]},
                               cookies={}, query_params={}, body=None)
        try:
            return await mcp._execute_api_tool(client, tool, arguments, mcp.operation_map, info)
        finally:
            await client.aclose()

    return json.loads(asyncio.run(run())[0].text)


@pytest.mark.parametrize("jwt_mode", ("unsigned", "signed_uid"))
def test_the_mcp_invoke_assistant_tool_follows_the_default_host_rule(app_module, monkeypatch, jwt_mode):
    """The MCP tool is the same route: on a default server a tool call is refused without signed JWTs, and
    with them is asked with a token that a plain yes does not replace. (invoke_agent, the stream tool, has
    the identical wiring but needs a model to plan, and its body is a stream the executor returns as text;
    it is covered through the endpoint above, not through MCP.)"""
    uid = _configure_default(monkeypatch, jwt_mode)
    client = TestClient(app_module.app)
    _c, headers, _wf, chores, ctx = _session(app_module, client, uid)
    args = {"user_query": "set_properties assign_to=Ann", "timeout_seconds": 20}
    data = _mcp_call(app_module, "invoke_assistant", args, headers)
    if jwt_mode == "unsigned":
        assert _decision(data) == "REFUSE" and chores.assign_to == ""
        return
    token = data["approval"]["token"]
    assert chores.assign_to == ""
    _mcp_call(app_module, "invoke_assistant", {"user_query": "yes", "approval_token": token, "timeout_seconds": 20}, headers)
    assert chores.assign_to == "Ann"


def test_a_duplicate_request_rejoining_a_running_assistant_turn_still_gets_the_token(app_module, monkeypatch):
    """Under host the token is taken inside the turn and handed to the response. A request that times out
    (202) and is repeated rejoins the same execution; the token must still reach a response.
    The turn is held open by a process_turn seam that waits on an event, then runs the real gate."""
    import threading
    import time
    uid = _configure_default(monkeypatch, "signed_uid")
    client = TestClient(app_module.app)
    _c, headers, wf, chores, ctx = _session(app_module, client, uid)
    release = threading.Event()

    def slow_turn(message, *, unattended=False, approval_surface=None, approval_token=None):
        release.wait(30)
        return ctx.process_action_turn(fastworkflow_action(), approval_surface=approval_surface)

    ctx.process_turn = slow_turn
    body = {"user_query": "set_properties assign_to=Ann", "timeout_seconds": 25}
    results: dict = {}

    def post(name, delay):
        time.sleep(delay)
        results[name] = shared.post("/invoke_assistant", headers=headers, json=body)

    threads = [threading.Thread(target=post, args=("first", 0)), threading.Thread(target=post, args=("dup", 1.0))]
    with TestClient(app_module.app) as shared:       # one client: both requests share one event loop
        _configure_default(monkeypatch, "signed_uid")  # entering the client ran startup, which resets the JWT mode
        for t in threads:
            t.start()
        threading.Timer(3.0, release.set).start()   # the duplicate arrives while the turn is still held open
        for t in threads:
            t.join(60)
    first, dup = results["first"].json(), results["dup"].json()
    assert results["first"].status_code == results["dup"].status_code == 200
    assert first["turn_key"] == dup["turn_key"], "the duplicate did not rejoin the running turn"
    # One execution, one delivery: the token reaches exactly one of the two responses, and the rejoining
    # request must not replace the delivery the executing request fills.
    carriers = [r for r in (first, dup) if r.get("approval", {}).get("token")]
    assert len(carriers) == 1, (first, dup)
    assert chores.assign_to == ""


def _signed_host(app_module, monkeypatch):
    uid = _configure(monkeypatch, "signed_uid", "host")
    client = TestClient(app_module.app)
    return client, _session(app_module, client, uid)


@pytest.mark.parametrize("endpoint", ("invoke_agent_stream", "invoke_assistant"))
def test_host_signed_uid_delivers_the_token_and_a_reply_carrying_it_approves(app_module, monkeypatch, endpoint):
    client, (_channel, headers, wf, chores, ctx) = _signed_host(app_module, monkeypatch)
    data = _ask(client, headers, endpoint, ctx)
    token = data["approval"]["token"]
    assert token and chores.assign_to == ""
    if endpoint == "invoke_agent_stream":
        ctx.__dict__.pop("process_turn", None)      # the reply path needs no model and no seam
    done = _post(client, headers, endpoint, ctx, user_query="yes", approval_token=token)
    assert chores.assign_to == "Ann", done
    assert "approval" not in done


@pytest.mark.parametrize("endpoint", ("invoke_agent_stream", "invoke_assistant"))
def test_host_a_reply_without_the_token_is_handled_as_invoke_agent_handles_it(app_module, monkeypatch, endpoint):
    client, (_c, headers, _wf, chores, ctx) = _signed_host(app_module, monkeypatch)
    _ask(client, headers, endpoint, ctx)
    ctx.__dict__.pop("process_turn", None)
    chat = _post(client, headers, endpoint, ctx, user_query="yes")
    assert chores.assign_to == ""

    _c, headers2, _wf2, chores2, ctx2 = _session(app_module, client, "alice")
    _ask(client, headers2, "perform_action", ctx2)
    agent = _post(client, headers2, "invoke_agent", ctx2, user_query="yes")
    assert chores2.assign_to == ""
    assert chat["answer"] == agent["answer"]


@pytest.mark.parametrize("endpoint", ("invoke_agent_stream", "invoke_assistant"))
def test_the_token_is_in_no_stored_record_turn_output_or_observability_row(app_module, monkeypatch, endpoint, tmp_path):
    client, (channel_id, headers, wf, chores, ctx) = _signed_host(app_module, monkeypatch)
    data = _ask(client, headers, endpoint, ctx)
    token = data["approval"]["token"]
    turn_key = data.get("logical_turn_key") or data["turn_key"]

    assert token not in json.dumps({k: v for k, v in data.items() if k != "approval"})
    stored = app_module.session_manager.session_state_store.load(channel_id)
    assert token not in json.dumps(stored, default=str)
    polled = client.get(f"/turns/{turn_key}", headers=headers)
    assert token not in polled.text
    for db in tmp_path.rglob("observability.sqlite3*"):
        if db.suffix in (".sqlite3",):
            conn = sqlite3.connect(db)
            try:
                dump = "\n".join(conn.iterdump())
            finally:
                conn.close()
            assert token not in dump, db
    # Delivered once: nothing is left to hand out on the next turn.
    assert ab.take_host_delivery(wf) == {}


def test_fix_rkr1_e1_is_closed_by_the_binding_a_channel_asked_approval_is_not_approved_from_invoke_agent(
        app_module, monkeypatch):
    """The investigation suspected (E1) that a record asked on a channel-surface endpoint could be approved
    by a plain yes on /invoke_agent of an unsigned server, which refuses to ASK. It cannot: the approval's
    binding fingerprint includes the surface, so the reply from another surface finds a changed binding and
    the record is invalidated, the action not run. Pinned as the behaviour today (fix-rkr1 step 2 decides
    whether the endpoints should also share one identity rule, which this does not need)."""
    uid = _configure(monkeypatch, "unsigned", "channel")
    client = TestClient(app_module.app)
    _c, headers, _wf, chores, ctx = _session(app_module, client, uid)
    assert _decision(_ask(client, headers, "invoke_agent_stream", ctx)) == "ASK"
    ctx.__dict__.pop("process_turn", None)
    reply = _post(client, headers, "invoke_agent", ctx, user_query="yes")
    assert ab.MSG_INVALIDATED.split("{command}")[0] in reply["answer"]
    assert chores.assign_to == ""


@pytest.mark.parametrize("given,expected", [([], "host"), (["--chat_approval_surface", "host"], "host"),
                                            (["--chat_approval_surface", "channel"], "channel")])
def test_the_fastworkflow_run_fastapi_mcp_wrapper_forwards_the_option_to_the_server(
        tmp_path, monkeypatch, given, expected):
    """`fastworkflow run_fastapi_mcp ... [--chat_approval_surface X]` builds the server command line; the
    server's own argument parser reads the option from it. Only the subprocess launch is stood in for."""
    import subprocess
    from fastworkflow import cli
    env_file = tmp_path / "e.env"
    pw_file = tmp_path / "p.env"
    env_file.write_text("")
    pw_file.write_text("")
    seen: list = []
    monkeypatch.setattr(cli, "_require_server_extra", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(cmd) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(sys, "argv", ["fastworkflow", "run_fastapi_mcp", str(TODO), str(env_file), str(pw_file), *given])
    try:
        cli.main()
    except SystemExit:
        pass
    assert len(seen) == 1
    server_args = seen[0][3:]            # after: python -m fastworkflow.run_fastapi_mcp
    assert server_args[-2:] == ["--chat_approval_surface", expected]
    monkeypatch.setattr(sys, "argv", ["pytest", *server_args])
    import fastworkflow.run_fastapi_mcp.__main__ as main
    assert importlib.reload(main).ARGS.chat_approval_surface == expected
