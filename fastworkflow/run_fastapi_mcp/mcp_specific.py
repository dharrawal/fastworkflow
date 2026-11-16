# pyright: reportUnusedFunction=false

import asyncio
import contextvars
import json
import time
from typing import Any, Optional

import httpx
import mcp.types as mcp_types
from fastapi_mcp import FastApiMCP

from fastworkflow.run_fastapi_mcp.turns import ExecState
from fastworkflow.turn import TurnStatus

# The mount calls the app in process, over the same ASGI transport and base URL fastapi-mcp uses by default.
# What differs is the time-out policy, which is stated here instead of inherited from the library's default
# (httpx.AsyncClient(..., timeout=10.0)). A turn is allowed to take minutes, and the turn's own limit is the
# request's ``timeout_seconds`` (a delivery deadline: past it the turn keeps running and the caller is given
# its turn id), so the in-process hop sets NO read time-out: one more limit here could only cut a turn the
# server itself is still running. (httpx's ASGI transport does not enforce read time-outs today, so the
# library default did not cut an 11 s turn either; tests/test_mcp_long_turns.py pins that. This makes the
# behaviour independent of that library detail.) Connect, write and pool stay bounded.
MOUNT_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
MOUNT_BASE_URL = "http://apiserver"

# The tool descriptions an MCP client (an agent) reads. They REPLACE the text generated from the route's
# docstring, which is written for HTTP integrators (internal ticket ids, "Content-Type: application/json" for a
# stream, Kubernetes probe prose). The HTTP OpenAPI text, which the chat page does not read, is unchanged.
# Keyed by operation id; a tool with no entry keeps its generated description.
_TURN_RESULT_HELP = (
    "RESULT: structuredContent is the turn object {turn_key, exec_state, status, failure_reason, success, answer, "
    "command_outputs} (the same shape from every turn tool; the text block holds the same JSON, except invoke_agent "
    "whose text is the frame stream, see below). exec_state 'running': the turn is still going, NOT an error, read "
    "it with get_turn(turn_key); its progress says how long it has run and its latest step. exec_state 'done' with status: 'completed' (read answer); 'awaiting_user' (NOT an "
    "error: the server is asking YOU something, the question is in answer, the turn is suspended until you reply); "
    "'failed' (isError is true, failure_reason says why). On a completed turn success false means a command inside "
    "it reported failure (read command_outputs); it is not isError. "
    "ANSWERING A QUESTION: call the same tool again with your reply as user_query (an approval reply carries "
    "approval_token when the question delivered result.approval.token); to drop the question call cancel_pending. "
    "A failed HTTP call (401 bad token, 404, 409 another turn is running, 422, 503 session limit) is isError true "
    "with structuredContent {http_status, detail, retry_after_seconds?}."
)
_TURN_WAIT_HELP = (
    "SESSION LIMITS: if the server is at its live-session ceiling a call that needs a new session is refused with "
    "503 (retry_after_seconds in the result): wait that long and retry, and call end_session on channels you are "
    "finished with. If the server dropped this channel's workflow context, the first result afterwards carries "
    "session_notice {code: context_dropped, message}: re-establish any context you relied on."
)
TOOL_DESCRIPTIONS = {
    "invoke_assistant": (
        "Run one command of the workflow deterministically: no agent, no planning. user_query is the command line "
        "(for example 'find_book title=Dune'); a leading '/' is optional. Use it when you already know the command "
        "and its values, or to reply to a pending question. "
        + _TURN_RESULT_HELP + " "
        "TIME-OUT AND LONG TURNS: timeout_seconds (default 60) is how long this call waits, not a limit on the "
        "turn. If the turn is still running then, the call returns early with {turn_key, exec_state: 'running'} "
        "and the turn KEEPS RUNNING; read its result with get_turn(turn_key). Raise timeout_seconds to wait "
        "longer in one call. Do not send the same query again to find out: while it runs that joins the same "
        "turn, and once it has finished it runs the request a second time. When you are finished with the "
        "channel, call end_session. " + _TURN_WAIT_HELP
    ),
    "invoke_agent": (
        "Give the workflow's agent a request in natural language (or reply to its pending question). The agent "
        "chooses and runs commands, plans multi-step work, and may ask you questions. Use it for open requests; "
        "use invoke_assistant for a known command. "
        "STREAM: the text block is the whole stream, one frame per line ({type, seq, turn_key, logical_turn_key, "
        "data}; or SSE events if the channel was initialised for SSE): 'trace' and 'progress' frames, a "
        "non-terminal 'timeout' frame, then exactly one terminal 'output' frame (the turn) or 'error' frame. "
        "structuredContent is that terminal frame as the turn object (turn_key is the execution key get_turn "
        "takes; logical_turn_key is the stored turn's key). "
        + _TURN_RESULT_HELP + " "
        "TIME-OUT AND LONG TURNS: this call stays open until the turn ends, however long that is. timeout_seconds "
        "(default 60) only adds a non-terminal 'timeout' frame (still_running true); the turn continues and its "
        "'output' frame still follows. If your own client gives up first, the turn keeps running: call "
        "get_latest_turn, or get_turn(turn_key), to read its status or result. No MCP progress notifications are sent: "
        "to watch a long turn, call get_latest_turn from a second request while this one is open (its progress "
        "gives elapsed_seconds and the latest step). Do not send the same query again: "
        "while it runs you get {exec_state: 'running', reason: 'duplicate_submission'}, and once it has finished "
        "it runs the request a second time. When you are finished with the channel, call end_session. "
        "Without an agent model on the server a turn may carry the artifact fw_not_run on its last command output: "
        "the parts not run here, with a suggested plan for you to carry out; an ad hoc part's entry also lists the "
        "planning commands available here (available_plans) and its own request text (request): to run "
        "that part with one of them, call invoke_agent again with user_query = request and use_plan = that name. " + _TURN_WAIT_HELP
    ),
    "get_turn": (
        "Fetch the status or result of a turn you started, by its turn_key (execution key or logical key). Use it "
        "when a tool call timed out or returned exec_state 'running'. exec_state 'running' means ask again later; "
        "'done' carries the result (status, answer, command_outputs). Only your own conversation's turns are "
        "readable; any other key is a 404. A tool call that timed out is recovered here, not resubmitted. "
        "Same turn object as the turn tools; a finished turn read from the store also carries record (the full "
        "turn record). A running turn carries progress {elapsed_seconds, frames, last_frame?} (the latest step of an "
        "invoke_agent turn), read from memory. isError is true for a failed turn and for a 404."
    ),
    "get_latest_turn": (
        "Your conversation's most recent turn: the turn running now, else the last one that finished. Use it when "
        "your client timed out before any result or turn_key arrived. Same turn object as get_turn. Do not "
        "resubmit the request. 404 when the conversation has no turn."
    ),
    "end_session": (
        "Call this when you are finished with a channel, so the server does not hold its session in memory for "
        "ever. It acts only on your own channel. Kept: your finished turns (get_turn and get_latest_turn still "
        "read them), the conversation history and all records. Dropped: the live session; if the workflow's "
        "state cannot be saved (pinned: true in the result) its in-memory state is lost and a later call on the "
        "channel starts fresh, otherwise it is restored. Refused (409, isError) while a turn is running and while "
        "a question is waiting for your answer, unless you pass discard_pending: true. Safe to repeat: a channel "
        "with no live session returns ended: false. Result {channel_id, ended, pinned, pending_discarded}."
    ),
    "cancel_pending": (
        "Abandon the question the workflow is waiting on (an agent question, a review, an approval) and clear its "
        "saved state, so your next message starts a new request. Use it instead of answering when you do not want "
        "to continue. Result {status: 'ok', cleared}."
    ),
    "new_conversation": (
        "Close the current conversation (the server saves it with a topic and a summary, which costs one model "
        "call) and start a fresh one on this channel. Refused with 409 while a turn is running. A question that "
        "is waiting is parked with the old conversation and returns if you activate it again. Result {status: 'ok'}."
    ),
    "get_all_conversations": (
        "List this channel's saved conversations, most recently updated first: structuredContent.result is a list "
        "of {conversation_id, topic, summary, created_at, updated_at}. limit (default 20) caps the list. Use a "
        "conversation_id with activate_conversation."
    ),
    "activate_conversation": (
        "Switch this channel to a saved conversation by conversation_id (from get_all_conversations); its history "
        "becomes the current one. Refused with 409 while a turn is running. Result {status: 'ok'}."
    ),
    "get_command_schemas": (
        "List the workflow's commands with their JSON Schemas: {qualified_command_name: {kind: 'execution' | "
        "'planning', input, output}}. input is what the command takes (use it to build an invoke_assistant "
        "command line); output is the shape of that command's command_response.output in a turn result. Runs "
        "nothing."
    ),
    "post_feedback": (
        "Record one review note against a turn of yours (turn_key), or a part of it: a category and subcategory "
        "from get_feedback_taxonomy, and a free-text comment. Notes are append-only and nothing feeds them back "
        "into a model's prompt. Returns the turn's notes, newest last."
    ),
    "get_feedback": "List the review notes recorded on one of your turns (turn_key), oldest first.",
    "get_feedback_taxonomy": (
        "The categories and subcategories post_feedback accepts, with their labels and prompts."
    ),
    "liveness_probe": "Is the server process answering. Result {status: 'alive'}.",
    "readiness_probe": (
        "Is the server ready for traffic: {status: 'ready' | 'not_ready', checks, sessions: {live, max_live, "
        "over_target, hard_ceiling, at_ceiling, ...}}. A not_ready server answers 503 (isError). Flags memory, "
        "sessions, observability and runtime add detail; runtime reports the effective runtime configuration."
    ),
}

# What an agent reads next to each argument (MCP only; the HTTP schema's own text is unchanged).
ARGUMENT_DESCRIPTIONS = {
    "user_query": (
        "What to say to the workflow. invoke_agent: a request in natural language. invoke_assistant: a command "
        "line such as 'find_book title=Dune'. While a question is pending, your reply to it."
    ),
    "timeout_seconds": (
        "How long THIS CALL waits for the turn, in seconds (default 60). It is not a limit on the turn: see the "
        "tool description for what happens past it."
    ),
    "unattended": (
        "true when nobody can answer a question this turn: the agent is not given the ask_user tool and a step "
        "that needs an approval is queued or refused instead of asked. This turn only. Default false."
    ),
    "approval_token": (
        "Only when replying to an approval question that delivered a token (result.approval.token): send it back "
        "unchanged. Otherwise omit."
    ),
    "use_plan": (
        "invoke_agent only. A planning command available here, by the name an fw_not_run entry's "
        "available_plans lists (send that entry's `request` as user_query): the request is run as ONE part bound to "
        "that planning command; the plan is settled, so no review opens: the turn is awaiting_user on a "
        "planning_parameter question listing every parameter of the plan with its drafted value (reply yes to run with them, "
        "one 'name: value' line per value to set or change, or cancel), and again while a required value is still missing. An unknown name or one not available here is an isError result. "
        "Omit for an ordinary request."
    ),
    "use_dspy_cache": (
        "Leave unset. Honoured only by a server started with --allow_request_cache_toggle; ignored elsewhere."
    ),
    "turn_key": "The turn_key from a turn result or a 'running' answer (execution key or logical turn key).",
    "discard_pending": (
        "true to end the session although a question is waiting for an answer (the question is lost). Default "
        "false: ending is refused while one is pending."
    ),
    "limit": "The most conversations to return (default 20).",
    "memory": "true adds retention metrics (model-response cache and conversation memory).",
    "observability": "true adds the observability configuration in effect.",
    "runtime": "true adds the effective runtime snapshot (feature vector, trained model version, command count).",
    "sessions": "true adds the pinned-session counts (this runs each session's get_state hook).",
    "target_label": "A short label for what the note is about, for example 'Turn'.",
    "comment": "The note itself, free text.",
    "category": "One of the categories from get_feedback_taxonomy; the subcategory must belong to it.",
    "subcategory": "One of that category's subcategories from get_feedback_taxonomy.",
    "target_kind": "What the note is about: the whole turn (default), or a phase, step, span or task of it.",
    "span_ids": "The span ids the note is about; required for target_kind phase, step and span.",
    "provenance": "Who wrote the note: human, coding_agent (default) or distillation_agent.",
    "experiment_id": "Optional scope: the experiment the turn belongs to (checked against the record).",
    "task_id": "Optional scope: the task of that experiment (checked against the record).",
    "attempt": "Optional scope: the attempt number (checked against the record).",
    "pass_id": "Optional scope: the pass within the turn (checked against the recorded passes).",
    "paired": "A second execution the note compares against, same fields as the primary target.",
    "conversation_id": "The conversation_id of a conversation listed by get_all_conversations.",
}

TURN_TOOLS = ("invoke_assistant", "invoke_agent", "get_turn", "get_latest_turn")

# The one structuredContent shape of the four turn tools. Success results validate against it (an MCP client
# that honours outputSchema does so only for results that are not isError). exec_state and turn_key are the two
# fields every turn result has: a running turn, a finished one, a stored one and an execution error all carry
# them. Everything else is present when it applies.
TURN_RESULT_SCHEMA = {
    "type": "object",
    "title": "TurnResult",
    "description": (
        "A turn, whichever tool returned it. exec_state is where the work is (running: not finished, ask again "
        "with get_turn); status is how the turn came out (completed, awaiting_user: the server asks you "
        "something, failed). A failed turn and every HTTP error are isError results."
    ),
    "properties": {
        "turn_key": {"type": "string", "description": "The execution key; get_turn takes it."},
        "logical_turn_key": {"type": ["string", "null"], "description": "The stored turn's key, once known."},
        "exec_state": {"type": "string", "enum": [state.value for state in ExecState]},
        "status": {"type": ["string", "null"], "enum": [*(s.value for s in TurnStatus), None]},
        "failure_reason": {"type": ["string", "null"]},
        "success": {"type": "boolean", "description": "Every command in the turn reported success."},
        "answer": {"type": "string", "description": "The answer text, or the question while awaiting_user."},
        "command_outputs": {"type": "array", "items": {"type": "object"}},
        "error": {"type": "string", "description": "The execution failed before any turn output existed."},
        "reason": {"type": "string", "description": "duplicate_submission: this query is already running."},
        "approval": {"type": "object", "description": "{record_id, token} of a pending approval (host surface)."},
        "session_notice": {"type": "object", "description": "{code, message}: the channel's context was dropped."},
        "traces": {"type": "array", "items": {"type": "object"}},
        "progress": {
            "type": "object",
            "description": (
                "Only while exec_state is running: {elapsed_seconds, frames, last_frame?: {type, seq, label?, "
                "seconds_ago}}. How long the turn has run and the latest stream frame it produced (invoke_agent "
                "turns; an invoke_assistant turn has frames 0). Read from memory: polling it costs nothing."
            ),
        },
        "record": {"type": "object", "description": "get_turn on a stored turn: the full turn record."},
    },
    "required": ["turn_key", "exec_state"],
    "additionalProperties": True,
}

# Endpoints that are not MCP tools (see the notes in setup_mcp). compose_answer_card and
# compose_question_form are optional helpers for a rendering client, not tools for a model.
EXCLUDED_OPERATIONS = [
    "root",
    "dump_all_conversations",
    "generate_mcp_token",
    "rest_initialize",
    "perform_action",
    "rest_invoke_agent",
    "refresh_token",
    "get_turn_trace",
    "compose_answer_card",
    "compose_question_form",
]

MCP_TRANSPORT_START_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Tool results: structuredContent and isError
# ---------------------------------------------------------------------------
#
# fastapi-mcp 0.4.0 answers every call with one TextContent holding the route's JSON (or, for a streaming route,
# the raw stream text), and the MCP SDK marks a result isError only when the library raises, which it does for
# exactly one reason: an HTTP status of 400 to 599. FastWorkflowMCP overrides ``_execute_api_tool`` (the one seam
# the library's call handler goes through; a handler may return a ready ``CallToolResult``, which the SDK passes
# through) to keep that text block byte for byte and add two things:
#
# * ``structuredContent`` on every result. One object per tool: the parsed body for a JSON route; for a list
#   body ``{"result": [...]}`` (MCP requires an object); for ``invoke_agent`` the terminal frame of its stream as
#   the same turn object the other turn tools return (the text block stays the whole stream).
# * ``isError: true`` for a turn that ended in failure, in addition to the HTTP failures the library already
#   flagged. See ``turn_failure`` for the exact definition.
#
# The wrapper does no model, database or session work: it reads the one response the call already made.

_RESPONSE: contextvars.ContextVar[Optional[httpx.Response]] = contextvars.ContextVar("fw_mcp_response", default=None)

_TERMINAL_FRAMES = ("output", "error")


def turn_failure(turn: dict[str, Any]) -> Optional[str]:
    """Why a turn object is a FAILED turn (isError), or None.

    A turn ended in failure exactly when its execution is finished and one of these holds:

    * ``error`` is present: the execution raised before any turn output existed (an ``error`` frame of a stream
      or the ``error`` body of ``render_turn_response``).
    * ``status`` is ``failed``: the turn raised (``failure_reason`` is the exception) or the agent ran out of
      iterations (``max_iters_exhausted``).
    * ``exec_state`` is ``lost``: the process restarted and the execution's record is gone.

    Not failures, by design: ``exec_state`` running or queued (the turn keeps going; the call only stopped
    waiting), ``status`` awaiting_user (a question, review or approval is waiting for the caller), ``status``
    completed with ``success`` false (one command inside the turn reported failure, which an agent often
    recovers from; ``success`` stays the signal for it, as in the HTTP API), and ``status`` cancelled or
    abandoned (a caller's own decision; the engine does not produce them today).
    """
    if turn.get("error"):
        return str(turn["error"])
    if turn.get("exec_state") == ExecState.LOST.value:
        return "the turn's record was lost (the server restarted)"
    if turn.get("status") == TurnStatus.FAILED.value:
        return turn.get("failure_reason") or "the turn failed"
    return None


def _last_frame(text: str, content_type: str) -> Optional[dict[str, Any]]:
    """The last frame of an NDJSON or SSE stream body, as {type, seq?, turn_key?, logical_turn_key?, data}."""
    try:
        if "event-stream" in content_type:
            block = text.rstrip().rsplit("\n\n", 1)[-1]
            frame: dict[str, Any] = {}
            for line in block.splitlines():
                field, _, value = line.partition(":")
                value = value.lstrip()
                if field == "event":
                    frame["type"] = value
                elif field == "id":
                    frame["seq"] = int(value) if value.isdigit() else value
                elif field == "data":
                    frame["data"] = json.loads(value)
            return frame or None
        last = text.rstrip().rsplit("\n", 1)[-1]
        frame = json.loads(last) if last else None
        return frame if isinstance(frame, dict) else None
    except (ValueError, TypeError):
        return None


def stream_turn(response: httpx.Response, text: str) -> dict[str, Any]:
    """The turn object of a finished ``invoke_agent`` stream: its terminal frame, in the shape of the other turn tools.

    Reads only the last frame, so the cost does not grow with the number of trace frames. ``turn_key`` is the
    execution key (the envelope's, or the X-FW-Turn-Key header for SSE, whose data carries no identity) and
    ``logical_turn_key`` the stored turn's key (the TurnOutput's own ``turn_key``).
    """
    frame = _last_frame(text, response.headers.get("content-type", ""))
    execution_key = (frame or {}).get("turn_key") or response.headers.get("x-fw-turn-key")
    if frame and frame.get("type") == "output" and isinstance(frame.get("data"), dict):
        data = frame["data"]
        turn = {**data, "turn_key": execution_key or data.get("turn_key"), "exec_state": ExecState.DONE.value,
                "logical_turn_key": frame.get("logical_turn_key") or data.get("turn_key")}
        return turn
    if frame and frame.get("type") == "error" and isinstance(frame.get("data"), dict):
        detail = frame["data"].get("detail") or "the stream ended with an error frame"
    else:
        detail = "the stream ended without a terminal frame"
    return {"turn_key": execution_key or "", "exec_state": ExecState.DONE.value, "error": str(detail)}


def structured_body(tool_name: str, response: httpx.Response, text: str) -> Optional[dict[str, Any]]:
    """The structuredContent of a successful call, or None when the body is not JSON."""
    content_type = response.headers.get("content-type", "")
    if "ndjson" in content_type or "event-stream" in content_type:
        # The raw body, not the text block: the library pretty-prints a stream that happens to be one frame.
        return stream_turn(response, response.text)
    try:
        body = json.loads(text)
    except ValueError:
        return None
    return body if isinstance(body, dict) else {"result": body}


def http_failure_body(response: httpx.Response) -> dict[str, Any]:
    """structuredContent of a call whose route answered 4xx or 5xx: the error body plus http_status."""
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}
    failure: dict[str, Any] = {"http_status": response.status_code}
    failure.update(body if isinstance(body, dict) else {"detail": body})
    retry_after = response.headers.get("retry-after", "")
    if retry_after.isdigit():
        failure["retry_after_seconds"] = int(retry_after)
    return failure


def _error_result(text: str, structured: dict[str, Any]) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)], structuredContent=structured, isError=True)


class FastWorkflowMCP(FastApiMCP):
    """fastapi-mcp's mount with structured results, an error flag and strict arguments (see the block above)."""

    def accepted_arguments(self, tool_name: str) -> Optional[list[str]]:
        """The argument names of a listed tool (None for a name that is not one: the library reports that)."""
        for tool in self.tools:
            if tool.name == tool_name:
                return list((tool.inputSchema or {}).get("properties") or {})
        return None

    async def _request(self, client, method, path, query, headers, body):
        response = await super()._request(client, method, path, query, headers, body)
        _RESPONSE.set(response)
        return response

    async def _execute_api_tool(self, client, tool_name, arguments, operation_map, http_request_info=None):
        accepted = self.accepted_arguments(tool_name)
        unknown = sorted(set(arguments or {}) - set(accepted)) if accepted is not None else []
        if unknown:
            listing = ", ".join(accepted) if accepted else "none (this tool takes no arguments)"
            message = (f"Unknown argument{'s' if len(unknown) > 1 else ''} for {tool_name}: "
                       f"{', '.join(unknown)}. Accepted arguments: {listing}.")
            return _error_result(message, {"detail": message, "unknown_arguments": unknown,
                                           "accepted_arguments": accepted})
        _RESPONSE.set(None)
        try:
            content = await super()._execute_api_tool(
                client, tool_name, arguments, operation_map, http_request_info)
        except Exception as error:
            response = _RESPONSE.get()
            if response is None or not 400 <= response.status_code < 600:
                raise
            return _error_result(str(error), http_failure_body(response))
        response = _RESPONSE.get()
        if response is None or not content or not isinstance(content[0], mcp_types.TextContent):
            return content
        structured = structured_body(tool_name, response, content[0].text)
        if structured is None:
            return content
        failure = turn_failure(structured) if tool_name in TURN_TOOLS else None
        return mcp_types.CallToolResult(content=list(content), structuredContent=structured, isError=failure is not None)



def _mcp_transport(mcp):
    """The library's HTTP transport object, or a loud failure naming what changed.

    fastapi-mcp 0.4.0 has no public start hook: it starts its StreamableHTTP session manager lazily on the first
    request (``FastApiHttpSessionManager._ensure_session_manager_started``) and marks it started BEFORE the
    manager's task group exists, so a second concurrent first request skips the start lock, reaches
    ``StreamableHTTPSessionManager.handle_request`` and gets "Task group is not initialized" (HTTP 500). We
    therefore start it ourselves in the app lifespan, before readiness, through the same method and check the
    result. This relies on ``FastApiMCP._http_transport``, ``_ensure_session_manager_started``, ``shutdown`` and
    ``_session_manager._task_group``; if a library version renames one, startup fails here instead of
    silently bringing the race back.
    """
    transport = getattr(mcp, "_http_transport", None)
    if transport is None or not callable(getattr(transport, "_ensure_session_manager_started", None)) \
            or not callable(getattr(transport, "shutdown", None)):
        raise RuntimeError(
            "the installed fastapi-mcp no longer has the HTTP transport hooks this server starts the MCP "
            "session manager through (FastApiMCP._http_transport, _ensure_session_manager_started, shutdown); "
            "update fastworkflow.run_fastapi_mcp.mcp_specific for the new version"
        )
    return transport


async def start_mcp_transport(mcp, timeout: float = MCP_TRANSPORT_START_SECONDS) -> None:
    """Start the MCP session manager and return only when it can take requests (lifespan startup)."""
    transport = _mcp_transport(mcp)
    await transport._ensure_session_manager_started()
    deadline = time.monotonic() + timeout
    while True:
        manager = getattr(transport, "_session_manager", None)
        if manager is not None and getattr(manager, "_task_group", None) is not None:
            return
        task = getattr(transport, "_manager_task", None)
        if task is not None and task.done():
            raise RuntimeError("the MCP session manager stopped while starting; see the log above")
        if time.monotonic() > deadline:
            raise RuntimeError(f"the MCP session manager was not ready within {timeout:.0f}s")
        await asyncio.sleep(0.01)


async def stop_mcp_transport(mcp) -> None:
    """Stop the session manager started by start_mcp_transport (lifespan shutdown)."""
    await _mcp_transport(mcp).shutdown()


def setup_mcp(
    app,
    session_manager,
):
    """Mount MCP to automatically convert FastAPI endpoints to MCP tools.

    FastAPI endpoints are automatically exposed as MCP tools, except those in the exclude list.
    
    Key exposed tools:
    - invoke_agent: Streaming agent invocation with NDJSON/SSE support (from /invoke_agent_stream)
    - invoke_assistant: Assistant mode (deterministic execution)
    - new_conversation, get_all_conversations, post_feedback, activate_conversation
    - end_session: release the caller's channel when finished
    
    MCP Client Setup:
    - MCP clients use pre-configured long-lived access tokens (generated via /admin/generate_mcp_token)
    - Tokens are added to the MCP client configuration, not obtained via tool calls
    - No need for initialize or refresh_token tools in MCP context
    
    Note: Prompt registration (format-command, clarify-params) is commented out
    as fastapi-mcp 0.4.0 does not support custom prompts.

    Results: every tool result keeps its text block and adds ``structuredContent``; ``isError`` is true for an
    HTTP failure and for a turn that ended in failure (``turn_failure``). The tools that return a turn declare
    ``TURN_RESULT_SCHEMA`` as their output schema. Unknown arguments are rejected. All of it lives in
    ``FastWorkflowMCP``; the HTTP routes and the chat page are unchanged.
    """

    # =========================================================================
    # Mount MCP (FastApiMCP will scan and find all FastAPI endpoints)
    # =========================================================================
    
    # Exclude endpoints that should not be exposed as MCP tools:
    # - root: HTML homepage endpoint
    # - dump_all_conversations: Admin-only endpoint for dumping all user conversations
    # - generate_mcp_token: Admin-only endpoint for generating long-lived MCP tokens
    # - rest_initialize: Regular initialization (MCP clients use pre-configured tokens, don't need to initialize)
    # - perform_action: Low-level action execution (use invoke_agent/invoke_assistant instead)
    # - rest_invoke_agent: Non-streaming version (use "invoke_agent" streaming endpoint instead)
    # - refresh_token: JWT token refresh (not needed for MCP since MCP uses long-lived tokens)
    # - get_turn_trace: the replay surface of HTTP clients (spans), not a recovery path.
    # - get_turn and get_latest_turn ARE tools (fix-1ma8): a turn that outlasts the MCP client's own
    #   time-out keeps running and could not be read back. Both are scoped to the caller's channel by
    #   the same JWT check as over HTTP: another channel's key is a 404, as an unknown one.
    #
    # Exposed MCP tools:
    # - invoke_agent (operation_id) → /invoke_agent_stream endpoint (streaming with NDJSON/SSE support)
    # - invoke_assistant: Assistant mode (deterministic execution)
    # - new_conversation, get_all_conversations, post_feedback, activate_conversation
    #
    # Note: MCP clients are configured with long-lived access tokens generated via /admin/generate_mcp_token
    mcp = FastWorkflowMCP(
        app,
        exclude_operations=EXCLUDED_OPERATIONS,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url=MOUNT_BASE_URL,
            timeout=MOUNT_HTTP_TIMEOUT,
        ),
    )
    for tool in mcp.tools:
        if tool.name == "end_session":
            # The route takes an OPTIONAL body, which the library does not turn into arguments, so the tool
            # advertised no discard_pending although its description and the route both have it.
            tool.inputSchema.setdefault("properties", {}).setdefault(
                "discard_pending", {"type": "boolean", "default": False, "title": "discard_pending"})
        if description := TOOL_DESCRIPTIONS.get(tool.name):
            tool.description = description
        for argument, schema in ((tool.inputSchema or {}).get("properties") or {}).items():
            if argument in ARGUMENT_DESCRIPTIONS:
                schema["description"] = ARGUMENT_DESCRIPTIONS[argument]
            if "anyOf" in schema:
                # The generator leaves the first branch's "type" beside anyOf, so an explicit null (which the
                # route accepts for an optional field) failed the SDK's schema check before reaching the route.
                schema.pop("type", None)
        if tool.name in TURN_TOOLS:
            tool.outputSchema = TURN_RESULT_SCHEMA
    mcp.mount_http()
    return mcp

    # Note: Prompt registration is not supported in fastapi-mcp 0.4.0
    # The library automatically converts FastAPI endpoints to MCP tools,
    # but does not provide a way to register custom prompts.
    # Prompts may be added in a future version or via manual MCP server implementation.
    
    # TODO: Re-enable when fastapi-mcp supports prompts or implement custom prompt handler
    # # Prompts
    # mcp.add_prompt(
    #     name="format-command",
    #     description="Given command metadata and a user intent, format a single executable command with XML-tagged parameters.",
    #     arguments=[{"name": "intent", "required": True}, {"name": "metadata", "required": True}],
    #     handler=lambda intent, metadata: [
    #         {
    #             "role": "user",
    #             "content": {
    #                 "type": "text",
    #                 "text": (
    #                     f"Intent: {intent}\n\nMetadata:\n{metadata}\n\n"
    #                     "Format a single command: command_name <param>value</param> ..."
    #                 ),
    #             },
    #         }
    #     ],
    # )
    #
    # mcp.add_prompt(
    #     name="clarify-params",
    #     description="Compose a concise clarification question for missing parameters using the provided metadata.",
    #     arguments=[{"name": "error_message", "required": True}, {"name": "metadata", "required": True}],
    #     handler=lambda error_message, metadata: [
    #         {
    #             "role": "user",
    #             "content": {
    #                 "type": "text",
    #                 "text": (
    #                     f"{error_message}\n\nMetadata:\n{metadata}\n\n"
    #                     "Ask one short question to request the missing parameters."
    #                 ),
    #             },
    #         }
    #     ],
    # )


