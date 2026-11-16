"""Structured results, the error flag, strict arguments and tool text of the MCP mount (fix-1ma8).

Real app, real MCP route (initialize, then tools/call over ``/mcp``), no model: a turn's outcome is chosen by
wrapping the channel's ``process_turn`` around the real ``TurnOutput`` model, as the other server tests do.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import time

import httpx
import jsonschema
import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.run_fastapi_mcp import mcp_specific
from fastworkflow.run_fastapi_mcp.mcp_specific import TURN_RESULT_SCHEMA, TURN_TOOLS, stream_turn, turn_failure
from fastworkflow.turn import TurnStatus
from tests.test_approval_endpoint_surfaces import (  # noqa: F401  (fixtures and helpers)
    _configure, _session, app_module, no_live_model)
from tests.test_fastapi_end_session import FREE_ACTION, _new_channel, client  # noqa: F401
from tests.test_mcp_long_turns import McpClient, mcp_client
from tests.test_session_hard_ceiling import _fill, _limits, _token_for


def turn_with(status: TurnStatus, *, answer: str = "done", failure_reason: str | None = None, delay: float = 0.0):
    """A process_turn that finishes with the given outcome (the real TurnOutput model, no command run)."""
    def process(message, *, unattended=False, approval_surface=None, approval_token=None):
        time.sleep(delay)
        return fastworkflow.TurnOutput(turn_key="logical-turn-1", status=status, answer=answer,
                                       failure_reason=failure_reason, command_outputs=[])
    return process


def text_of(result: dict) -> str:
    assert len(result["content"]) == 1 and result["content"][0]["type"] == "text", result
    return result["content"][0]["text"]


def frames_of(result: dict) -> list[dict]:
    """The frames in the text block. A stream of exactly one frame parses as JSON, so the library pretty-prints it."""
    text = text_of(result)
    try:
        return [json.loads(text)]
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def real_turn(ctx, delay: float = 0.0):
    """A process_turn that runs a real model-free action (so the turn is recorded and readable afterwards)."""
    def process(message, *, unattended=False, approval_surface=None, approval_token=None):
        time.sleep(delay)
        return ctx.process_action_turn(fastworkflow.Action(**FREE_ACTION), approval_surface=approval_surface)
    return process


OUTCOMES = [
    ("completed", TurnStatus.COMPLETED, None, False),
    ("awaiting_user", TurnStatus.AWAITING_USER, None, False),
    ("failed", TurnStatus.FAILED, "agent_error", True),
]


# -- structuredContent is the text block's object ---------------------------------------------------------------

@pytest.mark.parametrize("label,status,reason,is_error", OUTCOMES, ids=[o[0] for o in OUTCOMES])
def test_assistant_structured_content_is_the_text_object_and_the_flag_follows_the_outcome(
        app_module, monkeypatch, label, status, reason, is_error):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        ctx.process_turn = turn_with(status, failure_reason=reason)
        result = mcp.call("invoke_assistant", {"user_query": "hello", "timeout_seconds": 20})
    body = json.loads(text_of(result))
    assert result["structuredContent"] == body
    assert body["exec_state"] == "done" and body["status"] == status.value
    assert result.get("isError", False) is is_error
    jsonschema.validate(result["structuredContent"], TURN_RESULT_SCHEMA)
    assert turn_failure(body) == (reason if is_error else None)


@pytest.mark.parametrize("label,status,reason,is_error", OUTCOMES, ids=[o[0] for o in OUTCOMES])
def test_agent_structured_content_is_the_terminal_frame_as_a_turn_and_the_text_stays_the_stream(
        app_module, monkeypatch, label, status, reason, is_error):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        ctx.process_turn = turn_with(status, failure_reason=reason)
        result = mcp.call("invoke_agent", {"user_query": "hello", "timeout_seconds": 20})
    frames = frames_of(result)                       # the text block is still the NDJSON stream
    assert frames[-1]["type"] == "output"
    turn = result["structuredContent"]
    assert turn["exec_state"] == "done" and turn["status"] == status.value
    assert turn["answer"] == frames[-1]["data"]["answer"] and turn["success"] is True
    assert turn["turn_key"] == frames[-1]["turn_key"]            # the execution key get_turn takes
    assert turn["logical_turn_key"] == "logical-turn-1"          # the stored turn's key
    assert result.get("isError", False) is is_error
    jsonschema.validate(turn, TURN_RESULT_SCHEMA)


def test_the_four_turn_tools_return_one_shape(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        ctx.process_turn = real_turn(ctx)
        assistant = mcp.call("invoke_assistant", {"user_query": "hello", "timeout_seconds": 20})
        key = assistant["structuredContent"]["turn_key"]
        by_key = mcp.call("get_turn", {"turn_key": key})
        latest = mcp.call("get_latest_turn", {})
    for result in (assistant, by_key, latest):
        assert result.get("isError", False) is False
        turn = result["structuredContent"]
        assert turn == json.loads(text_of(result))
        assert {"turn_key", "exec_state", "status", "answer", "success", "command_outputs"} <= set(turn)
        jsonschema.validate(turn, TURN_RESULT_SCHEMA)
        assert turn["status"] == "completed" and turn["answer"]


def test_a_list_body_is_wrapped_and_other_tools_carry_their_body(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, _ctx):
        conversations = mcp.call("get_all_conversations", {})
        alive = mcp.call("liveness_probe", {})
    assert conversations["structuredContent"] == {"result": json.loads(text_of(conversations))}
    assert alive["structuredContent"] == json.loads(text_of(alive)) and alive["structuredContent"]["status"] == "alive"
    assert not conversations.get("isError") and not alive.get("isError")


# -- running and waiting are not errors --------------------------------------------------------------------------

def test_a_turn_still_running_past_the_deadline_is_not_an_error(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        ctx.process_turn = real_turn(ctx, delay=2.5)
        early = mcp.call("invoke_assistant", {"user_query": "slow", "timeout_seconds": 1})
        assert early["structuredContent"]["exec_state"] == "running" and not early.get("isError")
        jsonschema.validate(early["structuredContent"], TURN_RESULT_SCHEMA)
        running = mcp.call("get_turn", {"turn_key": early["structuredContent"]["turn_key"]})
        assert not running.get("isError") and running["structuredContent"]["exec_state"] in ("running", "done")
        deadline = time.monotonic() + 15
        while running["structuredContent"]["exec_state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.3)
            running = mcp.call("get_turn", {"turn_key": early["structuredContent"]["turn_key"]})
    assert running["structuredContent"]["exec_state"] == "done" and not running.get("isError")


def test_a_duplicate_agent_submission_is_running_not_an_error(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, _ctx):
        _ctx.process_turn = turn_with(TurnStatus.COMPLETED, delay=2.5)
        with cf.ThreadPoolExecutor(1) as pool:
            first = pool.submit(mcp.call, "invoke_agent", {"user_query": "same", "timeout_seconds": 20})
            time.sleep(1.0)
            second = mcp.call("invoke_agent", {"user_query": "same", "timeout_seconds": 20})
            finished = first.result()
    assert second["structuredContent"]["exec_state"] == "running"
    assert second["structuredContent"]["reason"] == "duplicate_submission" and not second.get("isError")
    assert finished["structuredContent"]["status"] == "completed" and not finished.get("isError")


# -- HTTP failures are errors, with a structured body ----------------------------------------------------------

def test_http_failures_are_errors_with_the_status_in_the_structured_body(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        not_found = mcp.call("get_turn", {"turn_key": "no-such-turn"})
        assert not_found["isError"] is True and not_found["structuredContent"]["http_status"] == 404
        assert "Status code: 404" in text_of(not_found)          # the text block is the library's, unchanged

        unprocessable = mcp.call("post_feedback", {
            "turn_key": "x", "target_label": "Turn", "comment": "c",
            "category": "observations_analysis", "subcategory": "what_to_do"})   # a category and an unpaired subcategory
        assert unprocessable["isError"] is True and unprocessable["structuredContent"]["http_status"] == 422

        ctx.process_turn = turn_with(TurnStatus.COMPLETED, delay=2.5)
        mcp.call("invoke_assistant", {"user_query": "slow", "timeout_seconds": 1})
        conflict = mcp.call("end_session", {})                      # a turn is running on the channel
        assert conflict["isError"] is True and conflict["structuredContent"]["http_status"] == 409
        assert "detail" in conflict["structuredContent"]
        time.sleep(2.0)


def test_a_bad_token_is_an_error(app_module, monkeypatch):
    uid = _configure(monkeypatch, "unsigned", "channel")
    with TestClient(app_module.app) as test_client:
        _session(app_module, test_client, uid)
        mcp = McpClient(test_client, "Bearer not-a-token")
        result = mcp.call("get_latest_turn", {})
    assert result["isError"] is True and result["structuredContent"]["http_status"] == 401


def test_the_session_ceiling_is_an_error_with_retry_after(app_module, client):
    _limits(app_module, soft=1, hard=2)
    _fill(app_module, client, 2)
    mcp = McpClient(client, _token_for("someone_new")["Authorization"])
    result = mcp.call("invoke_assistant", {"user_query": "/get_properties", "timeout_seconds": 5})
    assert result["isError"] is True
    assert result["structuredContent"]["http_status"] == 503
    assert result["structuredContent"]["retry_after_seconds"] == 30
    assert "ceiling" in result["structuredContent"]["detail"]


def test_an_execution_error_is_an_error(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        def explode(message, **kwargs):
            raise RuntimeError("the workflow broke")
        ctx.process_turn = explode
        result = mcp.call("invoke_assistant", {"user_query": "hello", "timeout_seconds": 20})
    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured.get("http_status") == 500 or structured.get("error")


# -- the pure parts: stream endings, the failure definition ------------------------------------------------------

def _stream(text: str, content_type: str, **headers) -> tuple[httpx.Response, str]:
    return httpx.Response(200, text=text, headers={"content-type": content_type, **headers}), text


def test_a_stream_that_ends_in_an_error_frame_or_with_no_terminal_frame_is_a_failed_turn():
    error_frame = json.dumps({"type": "error", "seq": 3, "turn_key": "exec-1", "data": {"detail": "no output"}})
    turn = stream_turn(*_stream(error_frame + "\n", "application/x-ndjson"))
    assert turn == {"turn_key": "exec-1", "exec_state": "done", "error": "no output"}
    assert turn_failure(turn) == "no output"

    trace_only = json.dumps({"type": "trace", "seq": 0, "turn_key": "exec-2", "data": {}}) + "\n"
    turn = stream_turn(*_stream(trace_only, "application/x-ndjson"))
    assert turn_failure(turn) == "the stream ended without a terminal frame"


def test_an_sse_stream_reads_its_terminal_event_and_takes_the_key_from_the_header():
    data = {"turn_key": "logical-9", "status": "awaiting_user", "failure_reason": None, "answer": "which one?",
            "command_outputs": [], "success": False}
    text = ("id: 0\nevent: trace\ndata: {}\n\n"
            f"id: 1\nevent: output\ndata: {json.dumps(data)}\n\n")
    turn = stream_turn(*_stream(text, "text/event-stream", **{"x-fw-turn-key": "exec-9"}))
    assert turn["turn_key"] == "exec-9" and turn["logical_turn_key"] == "logical-9"
    assert turn["status"] == "awaiting_user" and turn_failure(turn) is None


def test_the_failure_definition_over_every_status_and_state():
    assert turn_failure({"exec_state": "running"}) is None
    assert turn_failure({"exec_state": "queued"}) is None
    assert turn_failure({"exec_state": "done", "status": "awaiting_user", "success": False}) is None
    assert turn_failure({"exec_state": "done", "status": "completed", "success": False}) is None
    assert turn_failure({"exec_state": "done", "status": "cancelled"}) is None
    assert turn_failure({"exec_state": "done", "status": "abandoned"}) is None
    assert turn_failure({"exec_state": "done", "status": "failed", "failure_reason": "max_iters_exhausted"}) \
        == "max_iters_exhausted"
    assert turn_failure({"exec_state": "done", "status": "failed"}) == "the turn failed"
    assert turn_failure({"exec_state": "done", "error": "boom"}) == "boom"
    assert turn_failure({"exec_state": "lost"})


# -- arguments --------------------------------------------------------------------------------------------------

def test_unknown_arguments_are_rejected_naming_them_and_listing_the_accepted_ones(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        ran = []
        ctx.process_turn = lambda *a, **k: ran.append(1)
        result = mcp.call("invoke_assistant", {"user_query": "hello", "timeout": 5, "channel_id": "other"})
        assert result["isError"] is True
        message = text_of(result)
        assert "Unknown arguments for invoke_assistant: channel_id, timeout." in message
        for accepted in ("user_query", "timeout_seconds", "use_dspy_cache", "unattended", "approval_token"):
            assert accepted in message
        assert result["structuredContent"]["unknown_arguments"] == ["channel_id", "timeout"]
        assert not ran, "a rejected call must not start a turn"

        no_arguments = mcp.call("get_latest_turn", {"turn_key": "x"})
        assert no_arguments["isError"] is True and "takes no arguments" in text_of(no_arguments)


def test_every_listed_argument_of_every_tool_is_accepted(app_module, monkeypatch):
    """The strict check reads the tool's own schema, so nothing a client can legitimately send is refused."""
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, ctx):
        ctx.process_turn = turn_with(TurnStatus.COMPLETED)
        full = mcp.call("invoke_assistant", {"user_query": "hello", "timeout_seconds": 20, "unattended": False,
                                             "use_dspy_cache": None, "approval_token": None})
        assert not full.get("isError"), full        # an explicit null for an optional field is accepted
        tools = mcp.tools()
        assert set(tools["end_session"]["inputSchema"]["properties"]) == {"discard_pending"}
        ended = mcp.call("end_session", {"discard_pending": True})
        assert not ended.get("isError"), ended
        assert ended["structuredContent"]["ended"] is True


# -- tool definitions -------------------------------------------------------------------------------------------

def test_the_turn_tools_declare_one_output_schema_and_the_others_none(app_module, monkeypatch):
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, _ctx):
        tools = mcp.tools()
    for name in TURN_TOOLS:
        assert tools[name]["outputSchema"] == TURN_RESULT_SCHEMA
    assert not [n for n, t in tools.items() if n not in TURN_TOOLS and t.get("outputSchema")]
    jsonschema.Draft202012Validator.check_schema(TURN_RESULT_SCHEMA)


def test_descriptions_state_the_contract_an_agent_needs(app_module, monkeypatch):
    """Pins only what a client acts on: the three states, how to answer a question, recovery, the limits."""
    with mcp_client(app_module, monkeypatch) as (_channel, mcp, _ctx):
        tools = mcp.tools()
    for name in ("invoke_assistant", "invoke_agent"):
        text = tools[name]["description"]
        for needed in ("awaiting_user", "'running'", "isError", "user_query", "approval_token", "cancel_pending",
                       "get_turn", "end_session", "live-session ceiling", "retry_after_seconds"):
            assert needed in text, (name, needed)
    for name, tool in tools.items():
        assert not any(marker in tool["description"] for marker in ("fix-", "[R9]", "[A39]", "Kubernetes")), name
        properties = tool["inputSchema"].get("properties") or {}
        undocumented = [p for p, schema in properties.items() if not schema.get("description")]
        assert not undocumented, (name, undocumented)


# -- cost -------------------------------------------------------------------------------------------------------

def test_the_wrapper_adds_no_second_pass_over_a_stream():
    """A 20,000-frame stream is read through its last frame only (the cost does not grow with the traces)."""
    frames = "".join(json.dumps({"type": "trace", "seq": i, "turn_key": "e", "data": {"x": "y" * 50}}) + "\n"
                     for i in range(20000))
    last = json.dumps({"type": "output", "seq": 20000, "turn_key": "e", "logical_turn_key": "l",
                       "data": {"turn_key": "l", "status": "completed", "answer": "a", "command_outputs": [],
                                "success": True, "failure_reason": None}})
    response, text = _stream(frames + last + "\n", "application/x-ndjson")
    started = time.perf_counter()
    turn = stream_turn(response, text)
    assert time.perf_counter() - started < 0.2
    assert turn["status"] == "completed" and turn["turn_key"] == "e"
    assert mcp_specific.turn_failure(turn) is None


# -- a review turn with plan alternatives (bd fix-1ma8.1) validates against the output schema ---------------------

def test_a_review_turn_with_plan_alternatives_validates_against_the_output_schema():
    """The review artifact of an ambiguous sub-task review, built by the real review code (items with
    ``alternatives``, the ``switch_plan`` reply form) and carried in a real TurnOutput, is a valid turn result."""
    from fastworkflow import review_engine
    from fastworkflow.review_proposals import Means, PlanAlternative, SubTask, SubTaskProposal

    proposal = SubTaskProposal(items=[
        SubTask(goal="Find the books", means=Means(kind="planning", name="Library/book_by_title"),
                alternatives=[PlanAlternative(name="Library/author_books", description="Name each book by an author.")]),
        SubTask(goal="Tell a joke"),
    ])
    shown = review_engine.prompt_for(proposal)
    review = {
        "kind": "sub_tasks", "title": proposal.title, "revision": 1,
        "items": [item.model_dump(mode="json") for item in proposal.items],
        "changes_data": [True, False],
        "reply_forms": review_engine.reply_forms(shown, proposal),
    }
    assert review["items"][0]["alternatives"][0]["name"] == "Library/author_books"
    assert "alternatives" not in review["items"][1]
    assert review["reply_forms"][-1]["kind"] == "switch_plan"
    output = fastworkflow.TurnOutput(turn_key="logical-turn-1", status=TurnStatus.AWAITING_USER, answer=shown,
                                     command_outputs=[])
    turn = {**output.model_dump(mode="json"), "turn_key": "exec-1", "exec_state": "done",
            "command_outputs": [{"command_responses": [{"response": shown, "artifacts": {
                "question_kind": "review", "review": review}}], "success": True}]}
    jsonschema.validate(turn, TURN_RESULT_SCHEMA)
    assert mcp_specific.turn_failure(turn) is None
