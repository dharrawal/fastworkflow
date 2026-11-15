# FastWorkflow FastAPI + MCP Service

HTTP + MCP interface for FastWorkflow workflows with synchronous and streaming execution.

## Overview

This service exposes FastWorkflow workflows as REST endpoints and as MCP tools, enabling clients to:
- Initialize workflow sessions per channel
- Submit natural language queries (agent mode)
- Execute deterministic commands (assistant mode)
- Perform explicit actions
- Manage conversations with persistent history
- Collect feedback on interactions

## Architecture

- **Session Management**: In-memory `ChannelSessionManager` with per-channel `ChatSession` instances
- **Persistence**: Rdict-backed conversation storage (one DB file per channel)
- **Execution**: Synchronous turn-based processing with queue-based communication
- **Tracing**: Traces are collected by default and included in synchronous responses or emitted incrementally during streaming
- **Streaming (REST)**: `/invoke_agent_stream` supports Streamable HTTP via NDJSON by default and SSE when requested in REST initialize
- **Streaming (MCP)**: MCP transport mounted at `/mcp`. A tool call returns once, when the turn ends: `invoke_agent`'s text block is the whole frame stream and `structuredContent` its terminal frame (see "MCP tool results"); progress frames do not reach the client earlier (see "Progress of a long turn over MCP": poll `get_turn`/`get_latest_turn`)

See [`docs/fastworkflow_fastapi_spec.md`](../../docs/fastworkflow_fastapi_spec.md) and [`docs/fastworkflow_fastapi_architecture.md`](../../docs/fastworkflow_fastapi_architecture.md) for complete specification and design.

## Running the Service

### Start Server (REST + MCP)

```bash
uvicorn services.run_fastapi.main:app --host 0.0.0.0 --port 8000

# MCP (auto-mounted via fastapi_mcp) will be available at `/mcp`.
```

### Access Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

## Environment Variables

Configure in your environment (loaded at process startup via CLI args or env load):

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `FASTWORKFLOW_STATE_ROOT` | Absolute root for all persistent state (conversations, suspended sessions, checkpoints) | No | `~/.local/state/fastworkflow` |
| `FASTWORKFLOW_WORKFLOW_ID` | Overrides the per-workflow state namespace (defaults to the workflow folder name) | No | *workflow folder name* |
| `LLM_CONVERSATION_STORE_TIMEOUT_SECONDS` | Per-attempt client-side deadline for the topic/summary call, in seconds. At most 2 attempts, so the wall-clock worst case is about twice this plus backoff — kept under the 30s shutdown drain on purpose | No | `12` |
| `--expect_encrypted_jwt` | Enable full JWT signature verification (pass flag to require signed tokens) | No | False (no verification by default) |
| `--cors_loopback_only` | Pin CORS to loopback origins only (any port); takes precedence over `--cors_origin` | No | False (all origins) |
| `--enable_admin_endpoints` | Serve `/admin/dump_all_conversations` and `/admin/generate_mcp_token` (both then need a Bearer token) | No | False (both routes 404) |
| `--chat_approval_surface` | `host` or `channel`: the approval surface of `/invoke_agent_stream` (MCP `invoke_agent`) and `/invoke_assistant`; see "Approval surface of the chat endpoints" below. `/invoke_agent` and `/perform_action` are always `host` | No | `host` (strict; was `channel` before this change) |

The flags (`--expect_encrypted_jwt`, `--cors_loopback_only`, `--enable_admin_endpoints`, `--chat_approval_surface`) are accepted by both `fastworkflow run_fastapi_mcp` and `python -m fastworkflow.run_fastapi_mcp`; none is read from an env file.

Set the variables in the first three rows in the workflow's `fastworkflow.env`, not as shell exports: values are read from the env files loaded at startup.

### Approval surface of the chat endpoints (`--chat_approval_surface`)

An approval-gated command (`@user_approval_for_execution`) is decided against the request's approval surface.
`POST /invoke_agent` and `POST /perform_action` are always the **host** surface: they need a signature-checked
JWT that carries a `user_id` (`--expect_encrypted_jwt`), the approval token is delivered in the response, and the
reply must send it back as `approval_token`; otherwise the command is refused. `POST /invoke_agent_stream` (the MCP
`invoke_agent` tool) and `POST /invoke_assistant` are set by one option:

| `--chat_approval_surface` | stream and assistant endpoints |
|---|---|
| `host` (default) | The same rule as `/invoke_agent`. Signed JWT with a `user_id`: the command is asked, the token is delivered in the stream's `output` frame (`data.approval = {record_id, token}`) or the `/invoke_assistant` body, and the reply must carry it as `approval_token`. Unsigned JWT, or no `user_id`: refused (an unattended run on an unsigned server gets this refusal, not a queued record). |
| `channel` | The command is asked in every JWT mode and a plain `yes` approves it. No token is delivered; `approval_token` is ignored. For a local development tool only. An `always` reply records a standing approval under the chat origin (`channel:<os user>:chat`, set by the server, never by a request): it applies on these endpoints only, not in the CLI, and an `unattended` request never uses it (it queues). `/approvals` and `/revoke_approval <command>` list and revoke that origin's grants. A verified `host` surface's per-end-user grants are unchanged. |

Both spellings accept it: `fastworkflow run_fastapi_mcp <workflow_path> --chat_approval_surface host` and
`python -m fastworkflow.run_fastapi_mcp --workflow_path <workflow_path> ... --chat_approval_surface host`.

**BREAKING (default changed):** `host` is the default, so the MCP `invoke_agent` and `invoke_assistant` tools follow the same strict rule as `/invoke_agent`. On a server without `--expect_encrypted_jwt`, an approval-gated command on these endpoints is refused instead of asked. To keep the old behaviour pass `--chat_approval_surface channel`; `fastworkflow run_chatbot` already does for the loopback server it spawns, so the chat is unaffected. The option never loosens `/invoke_agent` or `/perform_action`. The token is
live-only: it is in the response that asks, never in a `TurnOutput`, a stored turn record, an observability row or
`GET /turns/{key}`. On `host` an MCP client that calls `invoke_agent` receives it in the tool's output text, as a
REST client does.

Notes:
- Conversations are stored in `FASTWORKFLOW_STATE_ROOT/workflows/<workflow-id>/observability.sqlite3`.
- `/conversations` now accepts a `limit` query parameter (default `20`).
- Shutdown waits up to 30 seconds for active turns (hard-coded).
- Shutdown does **not** generate conversation topics or summaries. It makes any
  unsaved turns durable and leaves the placeholder blank topic. A conversation is
  labeled when it is used as a chat instead: on its first completed chat turn, on
  `POST /activate_conversation` if still blank, and on `POST /new_conversation`.
  Labels refresh as a thread grows, at turn counts 1, 4, 16, 64, … A blank `topic`
  in `GET /conversations` means "no successful title yet" — a conversation that
  only ever ran `/initialize` or `/perform_action` keeps it, by design.

## Session limits

The server keeps one live session per channel in process memory. A session is *pinned* (never evicted) when its state cannot be checkpointed, most often because a workflow context class has no `get_state` hook. Four settings bound the live count for every workflow, hooks or not (process environment first, then the workflow env file; the startup log names each value and its source):

| Setting | Default | Meaning |
|---|---|---|
| `MAX_LIVE_SESSIONS` | 50 | Soft target. Above it, idle sessions that can be checkpointed are saved and released. |
| `MAX_LIVE_SESSIONS_HARD` | 4 x the soft target (200) | The most live sessions, counting sessions being created. Below `MAX_LIVE_SESSIONS` is a startup error. |
| `PINNED_SESSION_IDLE_TTL_SECONDS` | 1800 | A pinned session idle this long may be expired. 0 never expires. |
| `PROTECTED_SESSION_IDLE_TTL_SECONDS` | 86400 | The same for a session with a question waiting for its user. 0 never expires. |

- **The 503.** A request that needs a new session (`/initialize`, or any call on a channel with no live session) at the hard ceiling first makes room: it saves and releases an idle checkpointable session, then expires an idle pinned one. If neither is possible it is refused with `503`, a `Retry-After: 30` header and a message naming the ceiling; the MCP tool result carries the same text. The refusal happens before any model or database work. Requests for channels that already have a live session are never refused.
- **Expiry only above the soft target.** An idle pinned session is expired only while the live count is above `MAX_LIVE_SESSIONS`, when a new session needs room and in the 5-minute maintenance pass. A deployment that stays at or below the soft target loses nothing however long a session sits idle. A session with a running turn, or one a request is holding, is never expired. A session with a pending question waits for the longer TTL; its question is then discarded as `end_session` with `discard_pending` does.
- **Experiment channels.** A channel bound to a registered experiment attempt is exempt from expiry (the attempt claim lives on the live session, and a pinned session has no checkpoint to carry it). It counts toward the ceiling, so enough of them can cause 503s; `GET /probes/readyz?sessions=true` reports them as `experiment_exempt`.
- **The notice.** When the server expires a pinned session, or `end_session` drops one whose state was not saved, the channel's next session carries `session_notice: {code: "context_dropped", reason, message}` once: in the first turn result of every turn endpoint (the `output` frame of `invoke_agent_stream`) and in the `/initialize` response. The workflow context starts from launch configuration; conversation history and finished turns are kept.
- **Readiness.** `GET /probes/readyz` `sessions` reports `live`, `max_live`, `over_target`, `hard_ceiling`, `at_ceiling`, `refused_total`, `expired_total` and both TTLs; `?sessions=true` adds `pinned`, `busy_unassessed`, `pending_protected`, `experiment_exempt` and `trace_events_dropped`. Being at the ceiling does not make the pod unready.
- **Sizing.** Worst-case memory is the hard ceiling times (one session's workflow state + its conversation window). The window is the newest 20 turns, about 9 MB at a 450 KB turn, so 200 sessions is about 1.8 GB for conversation history alone, plus the workflow state. Size the ceiling to the pod. Each session's trace queue holds at most 10,000 events (oldest dropped and counted), and the conversation-window trim, which waits for the newest turn record to be durable, is applied anyway past twice the window.
- **Implementing `get_state`** is the only way to keep a session's state across eviction; with it the session is checkpointed and evicted by the soft target, and the ceiling is never what decides.

## Auth Modes

### Trusted Mode (default): No Signature Verification
When `--expect_encrypted_jwt` is NOT set (trusted environments), the service still creates and returns JWT tokens from `/initialize`, but signature verification is disabled. Clients must include `Authorization: Bearer <access_token>` on subsequent requests.

```bash
uvicorn services.run_fastapi.main:app --workflow_path /path/to/workflow
```

Notes (trusted mode):
- `/initialize` returns access/refresh tokens and, if a startup command/action is provided, also returns the startup turn's `TurnOutput`.
- Subsequent endpoints require `Authorization: Bearer <access_token>`.
- Traces include `user_id` when available (from JWT `uid` claim).

### Secure Mode: Signed JWTs with Verification
For production deployments requiring full RSA signature verification:

```bash
uvicorn services.run_fastapi.main:app --workflow_path /path/to/workflow --expect_encrypted_jwt
```

**Secure mode** (with `--expect_encrypted_jwt` flag):
- `/initialize` issues access/refresh tokens. Subsequent endpoints require `Authorization: Bearer <token>`.
- JWT claims include `sub` (channel_id) and `uid` (user_id when provided).
- Tokens are verified (signature, expiry, audience/issuer). Invalid or expired tokens are rejected.
- Recommended for production deployments in untrusted environments

### Token Access in Workflow Context

In secure mode, JWT tokens are passed to workflows via `workflow_context['http_bearer_token']` to support authenticated upstream calls. In trusted mode, tokens are not created/returned and `http_bearer_token` is absent.

**Important notes:**
- The token is **only available to authenticated endpoints** (those using `get_session_and_ensure_runtime` dependency)
- The token is stored in the workflow context dictionary under the key `http_bearer_token`
- Token is **automatically updated** on every authenticated request, ensuring workflows always have the current valid token
- Token expiration is **automatically verified** by `verify_token()` in both secure mode (`--expect_encrypted_jwt` flag) and trusted network mode
- In secure mode: Full cryptographic signature verification + expiration checking
- In trusted network mode: Expiration checking is performed (signature verification disabled)
- Tokens should be treated as sensitive data and handled securely in workflows
- The `/initialize` endpoint is unauthenticated and does NOT provide a token to the workflow context; tokens are only available after calling `/initialize` and using the returned token in subsequent requests

**Example usage in workflow:**

```python
# In workflow code
workflow_context = self._context
bearer_token = workflow_context.get('http_bearer_token')

# Use token for API calls
headers = {"Authorization": f"Bearer {bearer_token}"}
response = requests.get("https://api.example.com/data", headers=headers)
```

## API Endpoints (REST)

### `POST /initialize`
Initialize a session for a channel. Workflow configuration is loaded at server startup from CLI args/env.

**Request:**
```json
{
  "channel_id": "channel-123",
  "user_id": "user-9",
  "stream_format": "ndjson",
  "startup_command": "load_workflow ...",
  "startup_action": {
    "command_name": "find_orders",
    "parameters": {"channel_id": 42}
  }
}
```

**Rules:**
- `channel_id` is required.
- Exactly one of `startup_command` or `startup_action` may be provided (or neither).
- If startup is provided, `user_id` is required and recorded in the initial trace.
- `stream_format` controls REST streaming format for `/invoke_agent_stream` (NDJSON default, SSE optional).

**Response:**
```json
{
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 3600,
  "startup_output": { /* TurnOutput, present only if the startup turn finished in-window */ },
  "startup_turn_key": "20260807T193000.123456Z-a1b2c3d4e5f6",
  "startup_exec_state": "done"
}
```

### `POST /invoke_agent`
Submit a natural language query for agentic processing.

**Headers:**
- `Authorization: Bearer <access_token>` (JWT contains `sub` for channel_id and optional `uid` for user_id)

**Request:**
```json
{
  "user_query": "find orders for channel 42",
  "timeout_seconds": 60
}
```

**Response:**
```json
{
  "turn_key": "20260807T193000.123456Z-a1b2c3d4e5f6",
  "exec_state": "done",
  "status": "completed",
  "failure_reason": null,
  "success": true,
  "answer": "Found 3 orders for channel 42",
  "command_responses": [
    {
      "response": "Found 3 orders for channel 42",
      "success": true,
      "artifacts": {},
      "next_actions": [],
      "recommendations": []
    }
  ],
  "command_outputs": [
    {
      "command_responses": [ /* ... */ ],
      "workflow_name": "default_workflow",
      "context": "Order management context",
      "command_name": "find_orders",
      "command_parameters": "channel_id=42"
    }
  ],
  "traces": [...]
}
```

Every turn endpoint answers with this shape — the turn's public `TurnOutput`
projection (`turn_key`, `status`, `failure_reason`, `answer`, `command_outputs`,
`success`), plus the transport's `exec_state`, a backward-compatible
`command_responses`, and `traces` when collected. The per-command
`workflow_name`/`context`/`command_name`/`command_parameters` that used to sit at
the top level are now inside each `command_outputs` entry.

`exec_state` is where the *work* is, `status` is how the *turn* came out. A turn
that fails, or that suspends to ask the user something, is still HTTP 200: read
the outcome from `status`/`failure_reason`/`success`, never from the status code.
A turn still running past the wait window returns `202 {turn_key, exec_state:
"running"}`; retrying the same request rejoins the same execution.

See `docs/fastworkflow_fastapi_spec.md` §6a for the full contract.

### `POST /invoke_assistant`
Submit a query for deterministic/assistant execution (no planning).

Same request/response format as `/invoke_agent`.

### `POST /perform_action`
Execute a specific workflow action directly (bypasses parameter extraction).

**Headers:**
- `Authorization: Bearer <access_token>` (JWT contains `sub` for channel_id and optional `uid` for user_id)

**Request:**
```json
{
  "action": {
    "command_name": "find_orders",
    "parameters": {"channel_id": 42}
  },
  "timeout_seconds": 60
}
```

**Response:**
Same format as `/invoke_agent`. Each direct action is its own logical turn.

### `POST /invoke_agent_stream`
Stream trace events and the turn's final `TurnOutput` via Streamable HTTP:
- NDJSON (default; `Content-Type: application/x-ndjson`)
- SSE (when REST `stream_format` is set to `sse`; `Content-Type: text/event-stream`)

**Headers:** Same as `/invoke_agent`.

The terminal `output` event carries the bare `TurnOutput` — no `exec_state` or
`command_responses`, since a stream has no deferral state to report. A failed or
`awaiting_user` turn arrives as an `output` event; the `error` event is reserved
for transport failures.

## Conversation Management (REST)

### `POST /new_conversation`
Persist current conversation and start a new one.

**Request:**
```json
{}
```

**Response:**
```json
{
  "status": "ok"
}
```

### `GET /conversations?channel_id={channel_id}&limit=20`
List conversations for a channel (most recent first). `limit` is optional; defaults to `20`.

**Response:**
```json
[
  {
    "conversation_id": 1,
    "topic": "Order Management",
    "summary": "...",
    "created_at": 1234567890000,
    "updated_at": 1234567890000
  }
]
```

### `POST /activate_conversation`
Switch to a different conversation by ID.

**Request:**
```json
{
  "conversation_id": 1
}
```

### `POST /post_feedback`
Record one review note against a recorded turn, or a component of it. Returns
`201` and the turn's notes, newest last.

Notes are append-only: posting again adds a second note rather than replacing
the first. Nothing feeds them back into a model's prompt.

**Request:**
```json
{
  "turn_key": "chan-1:7",
  "target_kind": "turn",
  "target_label": "Turn",
  "comment": "The request names three people; the plan covers only two.",
  "category": "observations_analysis",
  "subcategory": "observation",
  "provenance": "coding_agent"
}
```

`category`, `subcategory`, `target_kind` and `provenance` are enumerations in
the generated OpenAPI schema, so a client reading the schema sees the allowed
values. Category and subcategory must also PAIR; an unpaired combination is a
`422`. The six confirmed meanings:

| category | subcategories |
| --- | --- |
| `observations_analysis` | `observation`, `analysis` |
| `conclusions` | `what_went_right`, `what_went_wrong` |
| `recommendations` | `what_to_do`, `what_not_to_do` |

`comment` is free text — structure it however you like; no heading in it is
parsed, and the category is never inferred from the text. `provenance` is
`human`, `coding_agent` or `distillation_agent` and defaults to
`coding_agent`. `target_kind` defaults to `turn`; `phase`, `step` and `span`
require `span_ids` that belong to the turn. `experiment_id`, `task_id`,
`attempt` and `pass_id` are optional scope, and are checked against what the
turn actually records rather than believed. `paired` names a second execution
for a comparison note, in the same store.

This replaced a different endpoint of the same name. The old one posted
`{binary_or_numeric_score?, nl_feedback?}`, wrote one mutable row per turn to
an agent-memory `feedback` table, and that row was replayed into the agent's
`dspy.History` on its next turn. The table, the route and the injection were
removed together (`fix-9eg.16`); there is no compatibility shim, and the old
body is rejected.

### `GET /feedback?turn_key=...`
List the review notes recorded against one turn, oldest first. Each row
carries its category, subcategory, display labels, provenance, evidence
anchors and timestamp. Notes written before the taxonomy read back with
`category` and `subcategory` of `null` and `classified: false`.

### `GET /feedback_taxonomy`
The categories, subcategories, UI labels and composer prompts, from the same
source the browser UI uses.

## Admin Endpoints (REST-only; not exposed via MCP)

`POST /admin/dump_all_conversations` and `POST /admin/generate_mcp_token` are
**served only when the server is started with `--enable_admin_endpoints`**, and
require a Bearer token when they are. Without the flag both return 404 and are
absent from the OpenAPI schema. The dump reads every channel on the server, and
the token mint issues year-long credentials for any `channel_id`, so neither is
something a loopback binding should be the only thing standing in front of.

### `POST /admin/dump_all_conversations`
Export all conversations to JSONL.

**Request:**
```json
{
  "output_folder": "/path/to/export"
}
```

**Response:**
```json
{
  "file_path": "/path/to/export/all_conversations_1234567890.jsonl"
}
```

## Usage Examples

### REST initialize
```python
import requests

resp = requests.post("http://localhost:8000/initialize", json={
    "channel_id": "alice",
    "stream_format": "ndjson"
})
print(resp.json())  # {"access_token": "...", "refresh_token": "...", "token_type": "bearer", "expires_in": 3600}
```

### REST sync invoke
```python
# First get a token from /initialize
init_resp = requests.post("http://localhost:8000/initialize", json={
    "channel_id": "alice",
    "stream_format": "ndjson"
})
token_data = init_resp.json()
access_token = token_data["access_token"]

# Then use the token for authenticated requests
resp = requests.post("http://localhost:8000/invoke_agent", 
    headers={"Authorization": f"Bearer {access_token}"},
    json={
        "channel_query": "list all channels",
        "timeout_seconds": 30
    }
)
result = resp.json()
print(result["command_responses"])
print(result.get("traces"))
```

### MCP Quickstart
- The mount is the MCP streamable-HTTP endpoint at `POST /mcp` (JSON-RPC: `initialize`, `tools/list`, `tools/call`), built by fastapi-mcp over the app's routes; it is not a REST path per tool.
- Send `Authorization: Bearer <token>` on the MCP connection; the JWT `sub` claim is the `channel_id`. The tools take no token argument. In secure mode use a long-lived token from `/admin/generate_mcp_token`.
- No initialize or refresh_token tool exists in MCP: those routes are excluded.
- No MCP prompts are registered: fastapi-mcp 0.4.0 has no prompt support, so the server does not offer `prompts/list` at all. (An earlier version of this file claimed `format-command` and `clarify-params`; neither was ever served.)

```bash
curl -s -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -H 'Authorization: Bearer <your-mcp-token>' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"invoke_assistant","arguments":{"user_query":"list_loans","timeout_seconds":60}}}'
```
(after the `initialize` and `notifications/initialized` messages of a normal MCP session.)

### MCP tool results

Every tool result keeps the text block the library has always produced (the route's JSON, or the stream text) and adds
`structuredContent`. All of it is in `mcp_specific.FastWorkflowMCP`; the HTTP routes and the chat page are unchanged, and
the wrapper adds no model, database or session work (it reads the one response the call already made).

**One shape for a turn.** `invoke_assistant`, `invoke_agent`, `get_turn` and `get_latest_turn` return the same object in
`structuredContent` and declare it as their `outputSchema` (`mcp_specific.TURN_RESULT_SCHEMA`):
`{turn_key, exec_state, status, failure_reason, success, answer, command_outputs, ...}`, plus when they apply
`logical_turn_key`, `error`, `reason`, `approval`, `session_notice`, `traces` and `record`.

| tool | text block | structuredContent |
|---|---|---|
| `invoke_assistant`, `get_turn`, `get_latest_turn` | the route's JSON body (unchanged) | the same object, parsed |
| `invoke_agent` | the stream, one frame per line (NDJSON; SSE events if the channel was initialised for SSE). A stream of exactly one frame is a JSON object, which the library pretty-prints | the terminal frame as the turn object: `turn_key` is the execution key (what `get_turn` takes), `logical_turn_key` the stored turn's key, `exec_state` is `done` |
| `invoke_agent`, already running | `{turn_key, exec_state: "running", reason: "duplicate_submission"}` | the same |
| other tools (conversations, feedback, probes, `end_session`, ...) | the route's JSON | the same object; a list body is wrapped as `{"result": [...]}` (MCP requires an object) |

The two body shapes of the HTTP API (a turn body from the plain routes, an NDJSON/SSE stream from `invoke_agent_stream`)
are NOT unified on the wire: that would change what the chat page and HTTP clients read. They are unified in
`structuredContent` instead, which is new. A client reads `structuredContent` and ignores the text; the text stays for
clients that read only text.

**`isError`.** True for:
- an HTTP error status of the route: 401 and 403 (token), 404 (unknown turn, no session), 409 (another turn is running, a
  question is pending for `end_session`), 422, 500, and 503 `session_capacity`. The text is the library's
  `Error calling <tool>. Status code: N. Response: ...`; `structuredContent` is `{http_status, detail, ...the body,
  retry_after_seconds?}` (`retry_after_seconds` from `Retry-After`).
- an unknown argument (below), and the MCP SDK's own input validation.
- a **turn that ended in failure**, exactly: `exec_state` is `done` and the turn object has an `error` (the execution
  raised before any output, or a stream ended with an `error` frame or no terminal frame), or `status` is `failed` (the
  turn raised: `failure_reason` is the exception; or the agent ran out of iterations: `max_iters_exhausted`), or
  `exec_state` is `lost` (the server restarted and the record is gone). The result still carries the turn object.

NOT `isError`: `exec_state` `running` (the call returned early past `timeout_seconds`, or a duplicate submission rejoined a
running turn); `status` `awaiting_user` (a question, plan review or approval is waiting for the client's reply; the
question is in `answer`); `status` `completed` with `success: false` (a command inside the turn reported failure, which an
agent usually recovers from: `success` stays the signal for it, as over HTTP); and `cancelled` / `abandoned` (the engine
does not produce them today). Failure is defined on existing turn fields only; nothing new was added to the turn.

`outputSchema` validation applies, in MCP clients, only to results that are not `isError`, so an error result need not
match it.

**Strict arguments.** A call with an argument the tool does not list is refused before anything runs, with
`Unknown argument(s) for <tool>: x. Accepted arguments: a, b.` (`structuredContent.unknown_arguments`,
`accepted_arguments`). The accepted set is the tool's own `inputSchema`, so nothing a client may send is refused. Two
fixes ride along: `end_session` now lists its `discard_pending` argument (the route and the description always had it,
the generated schema did not), and an explicit `null` for an optional argument (`use_dspy_cache`, `approval_token`) is
accepted.

**Tool and argument text** is written for an agent reader (`mcp_specific.TOOL_DESCRIPTIONS`, `ARGUMENT_DESCRIPTIONS`): what
the tool does, when to call it, the three turn states, how to answer a pending question (call the same tool again with the
reply as `user_query`; `approval_token` when the question delivered one; `cancel_pending` to drop it), how to recover a timed-out turn
(`get_turn`, `get_latest_turn`; never resubmit) and the session limits (503 and `retry_after_seconds`, `end_session`). The
HTTP OpenAPI text is unchanged.

### Progress of a long turn over MCP

**No `notifications/progress` is sent, and none can be delivered before the result with the installed mount** (mcp 1.30,
fastapi-mcp 0.4.0). A tool handler can read a call's `progressToken` (`_meta`) and call
`session.send_progress_notification(..., related_request_id=...)`, and the token does reach the mount's handler. But
the mount's transport (`fastapi_mcp/transport/http.py`) builds `StreamableHTTPSessionManager(json_response=True,
stateless=False)` and `FastApiMCP.mount_http` takes no argument to change it. In JSON mode the SDK answers the request
once and drops every notification of that request (`mcp/server/streamable_http.py`, the `is_json_response_enabled`
branch: "For notifications and request, keep waiting"). Even in SSE mode, which we checked by forcing it on a scratch
app, `handle_fastapi_request` collects the ASGI body into `response_body` and returns one `Response` after the request
ends, so the progress event arrives in the same chunk as the result, at the end. Delivering earlier would mean replacing the
library's transport, which this mount does not do. A call that carries a token is answered exactly as one without
(`tests/test_mcp_progress.py`).

**What a client does instead: poll.** A running turn read with `get_turn(turn_key)` or `get_latest_turn` carries
`progress` (additive; in `TURN_RESULT_SCHEMA`; absent once the turn is done):

```json
{"turn_key": "...", "exec_state": "running",
 "progress": {"elapsed_seconds": 41.3, "frames": 7,
              "last_frame": {"type": "progress", "seq": 6, "label": "Matching sub-task 2 of 3", "seconds_ago": 1.2}}}
```

`elapsed_seconds` is the time since the execution was admitted. `frames` and `last_frame` come from the stream of an
`invoke_agent` turn (its `progress` and `trace` frames; the label is the frame's own stage label, or
`<direction>: <command_name>` for a trace, never answer text, so a caller sees nothing it does not already get in the
stream; `timeout` is labelled `still working`). An `invoke_assistant` turn has no stream, so it reports `elapsed_seconds`
and `frames: 0`; that is all there is to say about a deterministic command. It is read from memory (a few arithmetic
operations), with no store, session or model work; the only cost is on the turn itself, one dictionary assignment per
frame, inside the stream's own emit and guarded so that a failure never reaches the turn.

Recommended client pattern: `invoke_assistant` returns early past its `timeout_seconds` with `{turn_key, exec_state:
"running", progress}`; poll `get_turn(turn_key)` every few seconds until `exec_state` is `done`. `invoke_agent` stays open
until the turn ends; a client that can issue a second request while the first is open (the server handles them
concurrently) polls `get_latest_turn` for `progress` (tested: `tests/test_mcp_progress.py`); one that cannot waits, and after its own
time-out reads the turn with `get_latest_turn`.
A client that disconnects mid-turn leaves the turn running to its end and the session usable (`get_latest_turn`,
`get_turn`, further calls); never resubmit the request.

## Testing

See [`tests/`](../../tests/) for unit and integration tests.

**Key test scenarios:**
- SessionManager concurrency and lifecycle
- Env loading from files only
- CLI argument validation (startup_command vs startup_action)
- Timeout behavior (504 on no output)
- Conversation persistence and listing
- Feedback attachment
 - Initialize with startup command/action returns `startup_output` and records conversation
 - Both trusted mode return tokens; secure mode tokens are encrypted
 - Traces include `user_id` and `raw_command`

## Future Enhancements

- Session TTL and eviction policy
- Richer observability and structured logging
- Security: workflow path allow-list, authn/z
- Conversation history restoration on session resume

