### FastWorkflow FastAPI Service — Architecture & Design

Based on: [fastworkflow_fastapi_spec.md](mdc:docs/fastworkflow_fastapi_spec.md)

#### 1. Scope and Goals
- **Purpose**: Provide a minimal, modular FastAPI service that exposes FastWorkflow workflows over HTTP with behavior matching the CLI runner, replacing its interactive loop with synchronous endpoints.
- **Strict adherence**: This document mirrors the referenced spec and does not introduce functionality beyond what is explicitly stated.

#### 2. Non‑Goals
- **UI**: No user interface. Only REST endpoints with OpenAPI/Swagger at `/docs`.
- **Streaming**: Streamable HTTP at `/invoke_agent_stream` (NDJSON by default; SSE also supported per session preference). MCP Streamable HTTP is also available when MCP is mounted and maps to the same streaming implementation.

#### 3. High‑Level Architecture
- **Process‑wide app**: A single FastAPI app hosts an in‑memory `ChannelSessionManager` keyed by `channel_id`.
- **Per‑channel runtime**:
  - Active `ChatSession` (always agent mode) bound to a current conversation (internal id only).
  - Persistent `ConversationStore` per channel backed by Rdict (one DB file per channel).
- **Synchronous turn execution**: For each request, enqueue channel input and synchronously await a `CommandOutput` from the `command_output_queue` with timeout and single‑turn serialization.
- **Traces**: Trace events are collected by default and included in the synchronous response, or emitted incrementally for the streaming endpoint.
- **Auth modes**:
  - JWT access/refresh tokens are created at `/initialize` and required on authenticated endpoints. Tokens include `sub` (channel_id) and `uid` (user_id, if provided).
  - Trusted mode (default): tokens are unencrypted.
  - Secure mode: tokens are encrypted

#### 4. Minimal Module Decomposition
- **API Layer (FastAPI routes)**
  - Endpoint handlers for all specified routes (Section 6 below).
  - Request validation via Pydantic models mirroring FastWorkflow types (prefer importing canonical FastWorkflow models where available).

- **Session Layer (`ChannelSessionManager`)**
  - In‑memory registry: `{ channel_id: ChannelRuntime }`.
  - `ChannelRuntime` fields: `active_conversation_id`, `chat_session`, `lock`, `stream_format`.

- **Workflow Adapter**
  - Bridges HTTP inputs to `ChatSession` by placing channel messages on the `user_message_queue` (or executing an `Action`) and reading `CommandOutput` from `command_output_queue`.
  - When enabled, drains trace events from `command_trace_queue` into the response `traces` field (REST). Streaming is handled by NDJSON or SSE for REST; MCP tools mirror the same server-side streaming.

- **Persistence Adapter (`ConversationStore`)**
  - Rdict database per channel under `SPEEDDICT_FOLDERNAME/channel_conversations`.
  - Schema and functional rules described in Section 8.

- **Startup Configuration Loader**
  - Parse CLI args at process startup (e.g., `--workflow_path`, `--env_file_path`, `--passwords_file_path`, `--context`, `--startup_command`, `--startup_action`).
  - Load env strictly from provided files and call `fastworkflow.init(env_vars=...)`.
  - Store configuration in process-wide variables for use by `/initialize`.

#### 5. Request Lifecycle (Per Endpoint)

- **POST `/initialize`**
  - Create `ChatSession` with forced agent mode. Restore prior conversation when available; otherwise start a new one.
  - Accept optional `startup_command` or `startup_action` (mutually exclusive). If provided, immediately execute and capture a `CommandOutput` which is returned in the response and persisted to `ConversationStore` as the first turn.
  - Accept optional `user_id` (required if startup is provided) for attribution and trace enrichment.
  - Create and return JWT access/refresh tokens in both trusted and secure modes. In trusted mode, the tokens are unencrypted
  - Store `ChannelRuntime` in `ChannelSessionManager` and return response including tokens and optional `startup_output`.

- **POST `/invoke_agent`**
  - Require existing session; enforce single in‑flight turn per channel (acquire channel lock).
  - If the query begins with `/`, strip all leading slashes before processing.
  - Enqueue the channel message and wait for `CommandOutput` up to `timeout_seconds`.
  - Include collected traces in `CommandOutput.traces`.
  - `user_id` is extracted from the Authorization JWT (`uid` claim) in secure mode and trusted modes.

- **POST `/invoke_agent_stream`**
  - Streamable HTTP (NDJSON) and SSE: emits `{ "type": "trace" }` records and a final `{ "type": "output" }` record.

- **POST `/invoke_assistant`**
  - Same path as agent but deterministic/assistant execution (no planning).
  - Returns `CommandOutput`.
  - `user_id` is extracted from the Authorization JWT (`uid` claim) in secure mode and trusted modes.

- **POST `/perform_action`**
  - Validate session; accept an explicit `Action` and execute it directly (bypass parameter extraction).
  - Serialize per channel; return `CommandOutput` or timeout.
  - `user_id` is extracted from the Authorization JWT (`uid` claim) in secure mode and trusted modes.

- **POST `/new_conversation`**
  - Persist current conversation to Rdict with generated `topic` (unique per channel) and `summary`, both produced synchronously via `dspy.ChainOfThought()`.
  - On success, rotate to a new internal conversation id and clear runtime history; on failure, log critical, return 500, do not rotate.

- **GET `/conversations`**
  - Return up to `limit` (default `20`) latest conversations for the channel by `updated_at` desc, each including `{ conversation_id, topic, summary }`.

- **POST `/post_feedback`**
  - Record ONE free-form comment about recorded evidence, anchored to an
    explicit turn (`?turn_key=`) or to a component within it. It used to attach
    `{binary_or_numeric_score, nl_feedback}` to the latest turn of the active
    conversation; that agent-memory table was removed (fix-9eg.16) and the
    score is not reintroduced.
  - Validate the `category`/`subcategory` pair against the taxonomy
    (`observations_analysis` → `observation` | `analysis`; `conclusions` →
    `what_went_right` | `what_went_wrong`; `recommendations` → `what_to_do` |
    `what_not_to_do`) and the anchor against the evidence it names.
  - Append-only, in the workflow's live evidence database. A database this
    process cannot write refuses the comment rather than diverting it.
  - Reads are separate GETs: `/api/feedback-notes`, `/api/task-feedback`,
    `/api/feedback-taxonomy`.

- **POST `/admin/dump_all_conversations`**
  - Iterate all channels and conversations in Rdict; write a JSONL file to the provided folder and return the file path.

- **POST `/activate_conversation`**
  - Body: `{ channel_id, conversation_id }`. Locate the conversation by ID for the channel and set it as active. 404 if not found.

- **POST `/admin/generate_mcp_token`**
  - Generate long-lived access tokens for MCP client configuration. Default expiration: 365 days. Returns TokenResponse with access_token (no refresh_token).

- **GET `/`**
  - Simple HTML page linking to `/docs`; also serves as a health check.

#### 6. Data Models (Alignment with FastWorkflow)
- **Import first**: When FastWorkflow provides canonical Pydantic/dataclass models for these types, import and use them to avoid divergence.
- **Otherwise mirror the spec** (sketches only):
  - `InitializationRequest`, `InitializationResponse`
  - `InvokeRequest`, `PerformActionRequest`, `PostFeedbackRequest`
  - `Action`, `CommandResponse`, `CommandOutput`
- **Fields and semantics**:
  - `CommandOutput.traces` is included by default (if any traces were produced during the turn).
  - `Action` equals the runtime execution object consumed by `CommandExecutor`.
  - Traces are enriched with `user_id` (when provided) and `raw_command`.

- **`answer` is text, the rest is data**

The turn's `answer` is the default plain-text rendering. Every part of it that the core writes in code is also available as data, and a client may present from the data and ignore the text. Where a key is listed, it is in `command_outputs[*].command_response.artifacts` of the entry named; the answer text is the same with or without it.

| Text the core writes | Data | Where, and shape |
|---|---|---|
| Parts of a multi-part answer (`### N. title` headings, "Part i of n:" labels of a step-cap hand-off) | `fw_sub_tasks` | the turn's last command output; one entry per sub-task: `number`, `text`, `title`, `answered` (the part has text under its heading), `not_run` (`true` only in no-agent mode, for a part with an `fw_not_run` entry: its text is the stop's explanation, not a command's answer; absent otherwise), `start`, `end` (offsets of the part's answer in `answer`), and `runs` only for a part that was continued: `[{index, of, start, end}]`, the offsets of each run's answer after its label |
| A review or plan proposal and the replies it accepts | `review` with `question_kind: "review"` | the waiting entry; `kind`, `title`, `revision`, `items`, `changes_data`, `reply_forms`: `[{kind: accept/decline/edit/add/drop/switch_plan, reply or pattern or free_text, per_item, separator?, alternatives?}]`, the replies the shown hint promises (the reply reader's own cues are wider); an item whose planning-command match was ambiguous also carries `alternatives: [{name, description}]` (the other plans that fit, at most 3; absent otherwise), and `reply_forms` then ends with a `switch_plan` form `{pattern: "step N: use NAME", alternatives: {"<step>": [names]}}` |
| An approval question | `approval` with `question_kind: "approval"` | the waiting entry; `command`, `parameters`, `origin`, `irreversible`, `can_always` |
| A planning command paused at a step the client decides (no-agent mode) | `phase_step` with `question_kind: "phase_step"` | the waiting entry; `command`, `phase`, `step` (`{number, text, kind}`, kind `conditional` or `missing_value`), `decision` (`{kind: "conditional", condition, run_if_yes, replies: ["yes", "no"]}` or `{kind: "missing_value", field, type, description, reply}`), `decline` (`"cancel"`), `asked` (1, or 2 when the first reply could not be read), `done`, `facts` (finished phases), `remaining` (the rest of this phase's steps; `[]` when the work left is in later phases), `later_phases` (the names of the phases after this one). The turn is `awaiting_user` and its answer is the question; the reply is `yes`/`no` (or the value); `cancel` ends the command with its `fw_not_run` stop (`reason` "declined"). While it is pending every message to `/invoke_agent` (the MCP `invoke_agent` tool) is the reply, a `/`-prefixed one too (the endpoint strips the `/`, as for `ask_user`); a message to `/invoke_assistant` (the MCP `invoke_assistant` tool) is a separate command and the question stays. No verified identity is needed |
| A refused planning call or a failed "/" command's missing values | `parameter_request` | `command`, `missing`, `invalid`, `values` |
| The capability answer to "what can you do?" (written in code) | `capabilities` | one `framework` command output of its own in `command_outputs` (not drawn as a result), equal to the data on the final answer's output; planning commands with their example, the current context's commands, the other-context lines |
| A planning command that stopped, or a part not run, without an agent model (`LLM_AGENT` unset: no-agent mode) | `fw_not_run` | the turn's last command output (a turn that ran no command carries it on a command output of its own); a list, one entry per stopped part: `kind` (`not_run_here`), `command` (`null` for an ad hoc part), `phase`, `open_step` (`{number, text, kind, reason}`, `null` for an ad hoc part), `done`, `facts`, `remaining`, `suggested_plan` (a numbered continuation, no model call: the open step, the rest of its phase, each later phase's steps as the template has them, then the line asking the client to write the answer), `means`, `sub_task` (the part's number in a split turn, else `null`), and when they apply `later_phases`, `nested` (the stop of a planning command this one called), `reason`, `request` (an ad hoc part only: that part's own request text), and `available_plans` (an ad hoc part only: the planning commands available here (those the matcher can choose from: the active context, its bases, its live ancestors' contexts and the root's), `{name, description, closest}` each, the matcher's closest ones to the part first, flagged `closest`, then the rest by name; the answer text leads with "No matching plan was found for this part", then lists them as markdown, at most 12 and a count of the rest, and says "No planning commands are available here." when there are none; in a split turn the part with the full list prints it once and a later ad hoc part with the same plans lists only its own closest ones and points to it). To run such a part with one of those plans, send the optional `use_plan` of `/invoke_agent`, `/invoke_agent_stream` or the MCP `invoke_agent` tool with `user_query` = `request` and `use_plan` = a name from `available_plans`: one part, no split, no match, values drafted from the request, then the plan's parameter question instead of a review (the intent is settled, so no goal or steps are shown: one form with the plan's name and description and every parameter, the drafted values prefilled and the missing required ones marked; Run or `yes` confirms, `name: value` lines change values, `cancel` stops; one still missing is asked for again while replies make progress); an unknown or out-of-context name is HTTP 400 / MCP isError. The turn is COMPLETED and its answer is a short explanation written in code; wording the final answer is the client's |
| An ambiguous "/" command | `intent_choices` | qualified command names, in order |
| A capability answer written in code (agent mode, a workflow with planning commands) | `capabilities` | the answer's own `CommandResponse` (the `CommandOutput` that `process_message` returns and the CLI shows); it is not in `command_outputs`, which has no entry for that turn. `{planning_commands: [{name, title, example}], commands: [{name, title}], other_contexts: [{context, title, commands, more}]}` |
| The `what_can_i_do` listing | `capabilities` and `command_response.output` | `{workflow: {name, contexts, context_inheritance}, context, commands: [{name, qualified_name, context, planning, approval_marker, description, parameters, outputs, sample_utterances}]}` |
| `abort` ("command aborted") | `aborted` and `output` | what it ended: `{stage, command?, command_name?}`; `output` is `{command, command_name}` |
| `you_misunderstood` | `command_choices` and `output.valid_command_names` | the command names it lists, in its order |
| A command's own result | `command_response.output` | the typed `Signature.Output` |

#### 7. Error Handling
- **Status codes**
  - 404 Not Found: Missing `channel_id` / session not found.
  - 409 Conflict: Concurrent turn already in progress for the same `channel_id`.
  - 422 Unprocessable Entity: Validation failures, invalid paths, invalid action schema, XOR violations (e.g., both `startup_command` and `startup_action` provided). `/post_feedback` reports an invalid category/subcategory pair or a bad anchor as 400.
  - 500 Internal Server Error: Unexpected errors (log with stack trace; avoid broad except without logging).
  - 504 Gateway Timeout: No `CommandOutput` before `timeout_seconds`.
- **Error body**: `{ "detail": "<human‑readable message>" }`.

#### 8. Persistence Model (Rdict)
- **Database layout** (one file per channel under `USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME`):
  - Key: `meta` → `{ last_conversation_id: int }`
  - Key: `conv:<id>` →
    - `topic: str` (unique per channel; enforce uniqueness via case‑insensitive and whitespace‑insensitive comparison; on collision append incrementing integer)
    - `summary: str`
    - `created_at: int` (epoch ms)
    - `updated_at: int`
    - `turns: [ { "conversation summary": str, "conversation_traces": str (JSON) } ]` — no `feedback` key; review comments live in the observability store, not in conversation memory.
- **Constraints**
  - Single active conversation per channel to avoid write concurrency.
  - Histories persist across restarts; default resume to the last conversation.

#### 9. Concurrency & Timeouts
- **Per‑channel serialization**: Exactly one in‑flight turn per `channel_id` via an asyncio lock stored in `ChannelRuntime`.
- **Request timeout**: Default `timeout_seconds=60` (per request, configurable per call). It is a delivery deadline, not a limit on the turn: `/invoke_assistant` answers 202 `{turn_key, exec_state: "running"}` and the turn keeps running; `/invoke_agent_stream` keeps the body open, adds a non-terminal `timeout` frame, and the `output` frame follows.
- **Long turns over MCP** (fix-1ma8): the mount (`mcp_specific.setup_mcp`) is given its own in-process `httpx.AsyncClient` (same ASGI transport and base URL as fastapi-mcp's default) with no read time-out, so the library's 10 s default can never apply. It did not cut turns before either (httpx's ASGI transport does not enforce read time-outs; `tests/test_mcp_long_turns.py` runs an 11.5 s turn through `/mcp` on both tools), so this pins the policy rather than fixing a defect. What an MCP client's own time-out leaves behind is a running turn: `get_turn` (by `turn_key`; in-flight from the registry, finished from the observability store, a retired execution key through the 300 s alias) and `get_latest_turn` (`GET /turns/latest`: the channel's running turn, else its last stored turn) are tools, both scoped to the JWT's channel (a foreign or unknown key is the same 404). `get_turn_trace` stays excluded. Resubmitting a query is not a recovery: a duplicate while it runs joins it (202 `duplicate_submission` on the stream endpoint), and after it finishes it runs again; the request has no idempotency key a client can set.
- **Ending a session** (fix-1ma8): `ChannelSessionManager.end_session` (`POST /end_session`, MCP tool `end_session`) is the explicit counterpart of session creation; before it nothing ended a session (`remove_session` and `evict_live_session` have no production callers). Under the manager lock it refuses (409) when the channel has work in flight, a lease (another request) or a pending question (`pending_kind`: approval, review, agent_question, parameter_request, question) unless `discard_pending`; discarding drops the live pending state (`WorkflowExecutionContext.discard_live_pending`) and rewrites the stored pending blob, so parked state of other conversations survives and the discarded question does not return. A checkpointable session is then published as a retirement does (a failed write refuses the end and leaves it live: 503); a pinned one drops its in-memory state. The context is closed, the app workflow is evicted from the process registry (so the channel's next session cannot find and reuse it), the channel's turn-registry pointer is released and nothing in the turn, conversation or observability stores is touched. It acts only on the token's channel and is idempotent (`ended: false`). `remove_session` stays the unconditional low-level drop (cancels approvals, no checkpoint); `evict_live_session` no longer cancels pending approvals (its docstring always said it did not). The readiness probe reports `sessions` (`live`, `max_live`, `over_target`; `?sessions=true` adds `pinned` and `busy_unassessed`), and startup logs one warning for context classes without `get_state` (`checkpoint.startup_pinning_warning`). Eviction policy is unchanged: pinned sessions are never evicted and the cap never refuses a channel.

#### 10. CORS & Security
- **CORS**: Allow configured origins; default `*` in development only.
- **Workflow path restrictions**: Enforce that `workflow_path` is valid. If a path allow‑list is configured, restrict resolution accordingly.
- **Logging**: Log each call with `channel_id`, `user_id` (when available), action/command, outcome, and timing. Keep FastWorkflow file logging to `action.jsonl` unchanged.

#### 11. Configuration & Environment Loading
- **Startup**: Configuration is provided via CLI args and environment on process start; do not accept these in `/initialize`.
- **Operational environment variables**:
  - `SPEEDDICT_FOLDERNAME`: base folder for per‑channel Rdict files
  - `LLM_CONVERSATION_STORE`: LiteLLM model string for conversation topic/summary generation
  - `LITELLM_API_KEY_CONVERSATION_STORE`: API key for the `LLM_CONVERSATION_STORE` model
  - `--expect_encrypted_jwt`: when set, secure mode is enabled; otherwise trusted mode (no encryption) is used.
  - `--chat_approval_surface {channel,host}` (default `host`, strict): the approval surface of `/invoke_agent_stream` (and so the MCP `invoke_agent` tool) and `/invoke_assistant`. `host` applies the `/invoke_agent` rule there (without signed JWTs an approval-gated command is refused) and delivers the approval token in the stream's `output` frame or the response body, to come back as `InvokeRequest.approval_token`. `/invoke_agent` and `/perform_action` are always `host`. `channel` (a plain `yes` approves, no token) is for local development tools such as the `run_chatbot` loopback server, which passes it explicitly.

#### 12. Deployment
- **Uvicorn**: Run with `uvicorn services.run_fastapi.main:app --host 0.0.0.0 --port 8000`.
- **Lifespan**: Use default unless startup/shutdown hooks are added; only enable `lifespan="on"` if explicitly needed.

#### 13. Minimal File/Type Layout
- **Code**: `services/run_fastapi/main.py` contains:
  - FastAPI app and route handlers
  - `ChannelSessionManager` and `ChannelRuntime` (in‑memory)
  - Wiring to `ChatSession` queues and optional trace collection (REST)
  - OpenAPI security for JWT Bearer (all endpoints except `/initialize`, `/refresh_token`)
- **Code**: `services/run_fastapi_mcp/mcp_specific.py` contains:
  - MCP mounting using `fastapi_mcp` that exposes FastAPI endpoints as MCP tools
  - Excludes admin-only and non-applicable endpoints
  - Maps the REST streaming endpoint `/invoke_agent_stream` (operation_id `invoke_agent`) for MCP streaming
- **Code**: `services/conversation_store.py` contains:
  - Rdict adapters for `ConversationStore`

#### 14. Testing Guidance
- Parse NDJSON on `/invoke_agent_stream`: verify trace records arrive before the final `command_output` record; validate structure.
- Verify `/invoke_agent` returns `traces` array populated when produced.
- Concurrency: ensure 409 when a turn is in progress.
- Timeouts: ensure 504 when no `CommandOutput` arrives within `timeout_seconds`.
- Initialize with startup action/command returns `startup_output` and persists the first turn.
- Trusted mode returns unencrypted tokens; secure mode returns encrypted tokens. In both cases, the tokens contain `sub` (channel_id) and `uid` (user_id when provided).


- Response cache: a server-wide setting chosen at start (`FW_LM_CACHE=0`, read by `dspy_utils.lm_cache_enabled` when each model is built). The per-request `use_dspy_cache` on the turn endpoints changes it only on a server started with `--allow_request_cache_toggle` (the chatbot launcher's local server): `server_memory.apply_cache_toggle` rebuilds the cache with the same bound the policy installs, so readiness stays ready, and a cache turned off is reported as `checks.dspy_response_cache` rather than as drift. Otherwise it never changes it (it used to rebuild the process-wide DSPy cache, which affected every channel and made `/probes/readyz` report a drifted memory policy); a disagreeing value is ignored and named in the `X-FW-DSPy-Cache` response header. The compose endpoints still honour it per call on their own model copy.
- Retained turns: only `initialize_startup` records are retained after they finish (at most 20, 300 s), swept on every such completion; every other turn is dropped when it finishes and is read back through the observability store (`GET /turns/{key}`, the MCP `get_turn`). `retained_turns` on `/probes/readyz?memory=true` therefore does not grow with the number of finished turns.
