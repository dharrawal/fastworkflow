# fastWorkflow Chat Integration Reference

Detailed contracts for the FastAPI service (`fastworkflow.run_fastapi_mcp`) and the workflow file
formats. Read this when wiring the chat UI to the backend or fixing generated command files.

## Hosting the service

```bash
python -m fastworkflow.run_fastapi_mcp \
  --workflow_path <workflow_dir> \
  --env_file_path <workflow_dir>/fastworkflow.env \
  --passwords_file_path <workflow_dir>/fastworkflow.passwords.env \
  --port 8000
```

Other CLI flags: `--host` (default `0.0.0.0`), `--context` (JSON string), `--startup_command`,
`--startup_action` (JSON file path, or an inline JSON object starting with `{`), `--expect_encrypted_jwt` (enable JWT signature verification;
off by default for trusted networks), `--chat_approval_surface {channel,host}` (the approval surface of the stream and
assistant endpoints and the MCP tools; default `host`, strict: signed JWT with a user_id, token delivered and returned; `channel` is the lenient plain-`yes` form for local development tools). Requires `pip install "fastworkflow[server]"`.

Interactive docs are served at `/docs` (Swagger) and `/redoc`. CORS is open (`*`) by default —
tighten `allow_origins` for production.

## Authentication flow

1. `POST /initialize` → returns `access_token` + `refresh_token`.
2. Send `Authorization: Bearer <access_token>` on all authenticated endpoints.
3. On 401, call `POST /refresh_token` with the **refresh** token in the `Authorization` header.

Tokens are scoped to a `channel_id` (a session/user key you choose) and optional `user_id`.

## Endpoints

### POST /initialize  (public)
Create or resume a session and obtain tokens.
```json
{
  "channel_id": "user-123",
  "user_id": "user-123",
  "stream_format": "ndjson",        // "ndjson" (default) or "sse"
  "startup_command": null,          // optional; mutually exclusive with startup_action
  "startup_action": null            // optional dict; mutually exclusive with startup_command
}
```
Response:
```json
{
  "access_token": "…", "refresh_token": "…", "token_type": "bearer",
  "expires_in": 3600,
  "startup_output": null            // CommandOutput if a startup command/action ran
}
```
Notes: if `startup_command`/`startup_action` is provided, `user_id` is required. Set
`stream_format` here — it controls the framing of `/invoke_agent_stream` for the session.

### POST /refresh_token  (public)
Header: `Authorization: Bearer <refresh_token>`. Returns a new `TokenResponse`
(`access_token`, `refresh_token`, `token_type`, `expires_in`).

### POST /invoke_agent_stream  (auth) — primary chat endpoint
Body:
```json
{ "user_query": "cancel my most recent order", "timeout_seconds": 60 }
```
Streams the internal workflow↔assistant conversation as it happens, then the final output.

Response headers (read these first; both are CORS-exposed, so a browser client on another origin
can see them):

- `X-FW-Turn-Key` — the execution key of this turn, sent before the first frame. It is the handle
  for recovery (below).
- `X-FW-Stream-Format` — `ndjson` or `sse`, i.e. which framing this body actually uses. It follows
  the session's `stream_format` (fixed at `/initialize`); read the header rather than assuming.

**Event types.** `trace` — one public agent↔workflow interaction. `timeout` — NON-terminal: this
turn passed the `timeout_seconds` you asked for, but the deadline governs *delivery*, not
ownership: the turn is still running, its remaining traces and its `output` still arrive, and
resubmitting would start a second turn. `output` — TERMINAL, the turn's `TurnOutput`; a failed or
`awaiting_user` turn arrives here too. `error` — TERMINAL, the turn produced no output at all.
**Exactly one terminal frame ends the body and nothing follows it.**

- **NDJSON** (`application/x-ndjson`), one JSON object per line. The whole envelope is on the line:
  ```json
  {"type":"trace","seq":0,"turn_key":"…exec…","logical_turn_key":"…","data": { /* trace event */ }}
  {"type":"trace","seq":1,"turn_key":"…exec…","logical_turn_key":"…","data": { /* … */ }}
  {"type":"timeout","seq":2,"turn_key":"…exec…","logical_turn_key":"…",
   "data":{"detail":"Command execution timed out after 60 seconds","timeout_seconds":60,"still_running":true}}
  {"type":"output","seq":3,"turn_key":"…exec…","logical_turn_key":"…","data": { /* TurnOutput */ }}
  ```
- **SSE** (`text/event-stream`) — same events, same payloads; `seq` is the event id and `data` is
  the payload alone, so identity comes from the response header:
  ```
  id: 0
  event: trace
  data: { /* trace event */ }

  id: 1
  event: output
  data: { /* TurnOutput */ }
  ```

**Order and deduplication.** `seq` starts at 0 and is gapless and unique per stream; drop a `seq`
you have already rendered. A chunk can split a frame, so buffer until the delimiter (`\n` for
NDJSON, a blank line for SSE).

**Recovery — read, never resubmit.** If the body dies mid-turn, the turn keeps running server-side:
poll `GET /turns/{key}` and, for the interactions you missed, `GET /turns/{key}/trace` (non
destructive and repeatable). Resubmitting the query would be a second turn. *Which key:* start from
`X-FW-Turn-Key`, but keep the `logical_turn_key` that the NDJSON frames and every `/turns` answer
carry — the execution key only resolves while the turn is live (a chat execution is not retained
after it retires), whereas the store is keyed by the logical one. Where a partially read stream and
the stored record disagree, the record wins.

**HTTP 202 — the same query is already in flight.** A retried or duplicated submission is deduped
onto the running execution and answered `202 {"turn_key":…, "exec_state":"running",
"logical_turn_key":…, "reason":"duplicate_submission"}` with no body to read: the frames belong to
the consumer already reading them. Poll that key. A *different* query on a busy channel is
`409 {"detail":…, "reason":"channel_busy", "turn_key":…}`.

**Long turns over MCP.** `timeout_seconds` (default 60) is a delivery deadline: `invoke_assistant` returns 202
`{turn_key, exec_state:"running"}` and the turn keeps running; `invoke_agent` stays open and adds a `timeout`
frame. The mount has no read time-out of its own (the in-process client of `mcp_specific.setup_mcp`), so only the
MCP client's time-out can end a wait. After that, call the `get_turn` tool with the `turn_key` (or `get_latest_turn`
when no key arrived); both read only the caller's own channel. Never resubmit the query: it joins the running
turn, and after it has finished it runs again.

**Ending a channel.** When a client is finished with a channel, call `POST /end_session` (the MCP tool `end_session`;
no arguments, or `{"discard_pending": true}`). It frees the server's live session for that channel (a workflow whose
state cannot be checkpointed otherwise stays in server memory for the life of the process). Finished turns and history
stay readable; a pinned workflow's in-memory state is dropped. 409 while a turn runs or a question is pending (unless
`discard_pending`); repeating it is safe (`ended: false`). The chat page does not need it (it reuses one channel).

**Artifacts.** The terminal `output` carries `command_outputs[*].command_response.artifacts`
alongside the answer. A value may be inline, or an offloaded reference
`{"__fw_artifact_ref__":"<id>","size":…,"content_type":"…"}` to be fetched from the store that
produced it. Render them beside the final answer, not only in a debug view.

UI guidance: render every `trace` event live (this reproduces the `fastWorkflow run` CLI streaming
UX), show a `timeout` as "still working" rather than a failure, then render the `output` event as
the final human-readable answer.

### POST /invoke_agent  (auth)
Non-streaming agent turn. Returns a `CommandOutput` JSON with an extra `traces` array. Use only if
streaming is not feasible.

### POST /invoke_assistant  (auth)
Deterministic / non-agentic turn (no planner). Same body as `/invoke_agent`. The service prefixes
the query with `/` automatically when needed.

### POST /perform_action  (auth)
Execute a specific command directly, bypassing intent + parameter extraction.
```json
{ "action": { "command_context": "Order", "command_name": "cancel_order",
              "command_parameters": { "order_id": "W123" } },
  "timeout_seconds": 60 }
```

### POST /compose_answer_card and POST /compose_question_form  (auth) — optional UI helpers
Two calls a chat client may make after a turn to show a model-composed card or form. Same bearer token as
`GET /command_schemas`; they take no session lease, read no session state, and are not MCP tools. The server
reads the model from `LLM_RESPONSE_GEN` (key `LITELLM_API_KEY_RESPONSE_GEN`); there is no code default, but the shipped
`fastworkflow.env` template sets it to `mistral/mistral-small-latest`.

```json
POST /compose_answer_card
{ "user_message": "where is my order", "answer": "Your order is delivered.", "status": "completed",
  "use_dspy_cache": false,
  "results": [ { "id": "r1", "command": "get_order_details", "context": "", "parameters": {},
                 "output": {"status": "delivered", "total": 186.32}, "titles": {"total": "Total"} } ] }
```
```json
POST /compose_question_form
{ "question": "Which payment method should I refund to: paypal_5521 or the original?",
  "user_message": "refund my order", "results": [ /* same shape */ ], "use_dspy_cache": false }
```
Both always answer `200` with the verified value, or `null` and a `reason`:
```json
{ "card": { "lead": "...", "blocks": [ { "kind": "stats", "stats": [ { "label": "Status", "ref": "r1.status", "tone": "neutral" } ] } ] } }
{ "form": { "fields": [ { "name": "", "label": "Payment method", "kind": "choice",
                          "options": [ { "value": "paypal_5521", "label": "paypal_5521" } ],
                          "required": false, "help": "" } ] } }
{ "card": null, "reason": "no_model" }        // or "failed", "timeout" (the call took over 30 seconds), or "not_needed"
```
- `parts` (optional) is the list of parts of a split turn, as the core recorded it: the `fw_sub_tasks` artifact of the turn's last
  command output, forwarded as it is: `[{"number": 1, "text": "<the sub-task's request>", "title": "<its heading>", "answered": true,
  "start": 25, "end": 45}, ...]` (`start` / `end` are character offsets of that part's answer in `answer`, `null` for a part
  with no answer). Send it only when the turn has it. The server uses it, when it fits `answer`, in place of reading the
  answer's `### N.` headings; a list that is malformed or does not fit is ignored (never a `422`), and the headings are read as before.
- `id` is yours (`r1`, `r2`, ...). A `ref` is `<id>.<key>[.<key>...]` into that result's `output`, and the
  client reads the value from its own copy: the server returns no values. A JSON string held in an `output` field,
  and a one-key wrapper around a record, are decoded up to 5 levels before refs are made; do the same
  (`run_chatbot.composer.normalize_output_dict`), or refs will not resolve on your side.
- The server drops what it cannot ground: a lead or block title or stat label naming a number or ID that is not in
  the turn's data (the lead becomes `""`; a decimal must be a grounded number, rounded), a block whose `ref` does not
  resolve (a record or table whose ref is one value becomes a stat; a ref with a list index over a list of records becomes a
  table over that list), a quote that is not in the answer, a form option that is not a value or dict key in `results` or in the
  question (an option label naming a different ID than its value is replaced by the value).
- `not_needed` means the composer answered and chose the plain view: for a form, no field was left or the form gives the user
  nothing a text box does not (it needs a pick of two or more options, a yes/no or a date); for a card, only a lead that restates
  the answer was left. It is not an error and is not logged as one: treat it like any `null` (no notice, ask again next turn).
  Only `no_model` is worth telling the user, once.
- A card's table block over a whole result is built only from a *page*: an integer `total` (not a boolean) and every list
  field of one length, with at least two non-empty scalar lists (`run_chatbot.composer._is_page_shape`, the same definition as
  `observation_offloading.facts.output_page`; the shared fixture `tests/fixtures/page_shape_cases.json` pins it).
  A table over a list of records, or a table block naming fewer than two columns, shows every column.
- `use_dspy_cache: false` applies to that one call only. On `/invoke_agent`, `/invoke_agent_stream` and `/invoke_assistant` the argument changes the process-wide cache
  only on a server started with `--allow_request_cache_toggle` (the chatbot's local server); elsewhere it is ignored, and a value that
  disagrees with the server is named in the `X-FW-DSPy-Cache` response header. The turn cache is a server-wide
  setting chosen at start (`FW_LM_CACHE=0`).
- `422` when `results` has more than 60 entries, `answer` is over 150000 characters or `question` is over 20000 characters. Anything the
  model does wrong is a `200` with `null`, never a 5xx; keep your deterministic view when you get one.
- Everything in the request, including command outputs, is sent to the `LLM_RESPONSE_GEN` model. Offer a switch.

### Conversation management (auth)
- `POST /new_conversation` — persist the current conversation (generates topic/summary) and start fresh. Use for the **New chat** button.
- `GET /conversations?limit=20` — list past conversations (`ConversationSummary[]`, newest first). Use to render the **history list**.
- `POST /activate_conversation` `{ "conversation_id": 7 }` — restore a past conversation into the active session. Use for **continue previous chat**.
- `POST /cancel_pending` — abandon a suspended `ask_user` clarification turn.

### Feedback (auth)
`POST /post_feedback` records ONE free-form comment about recorded evidence. It
is not a thumbs up/down and carries no score: the old
`{binary_or_numeric_score, nl_feedback}` body and the agent-memory table behind
it were removed, and a request in that shape is rejected rather than scored.

The comment is anchored explicitly — a turn, or a component within it — and
classified with a category and one of its own subcategories:

| `category` | `subcategory` |
| --- | --- |
| `observations_analysis` | `observation`, `analysis` |
| `conclusions` | `what_went_right`, `what_went_wrong` |
| `recommendations` | `what_to_do`, `what_not_to_do` |

```json
POST /post_feedback?turn_key=<turn_key>
{ "target_kind": "turn", "span_ids": [], "target_label": "Turn",
  "provenance": "coding_agent",
  "category": "recommendations", "subcategory": "what_to_do",
  "comment": "Plan all three items before answering." }
```

`comment` is arbitrary text and is stored as written. A comparison comment adds
a `paired` object naming the other execution (`store_id`, `turn_keys`,
`experiment_id`, `task_id`, `attempt`, plus the same target fields), and the one
row is then visible from both tasks.

Reads are separate GETs, never a POST: `GET /api/feedback-notes?turn_key=…` for
one turn, `GET /api/task-feedback?experiment=…&task=…` for a whole task, and
`GET /api/feedback-taxonomy` for the categories above. Evidence that cannot be
written to — an archived experiment, or an older store — still accepts a comment: it
is recorded beside the evidence and the evidence file is not modified.

### Health probes (public)
- `GET /probes/healthz` → `{"status":"alive"}` (liveness).
- `GET /probes/readyz` → `200 {"status":"ready", …}` or `503 {"status":"not_ready", …}` (readiness).

### Admin (off by default; `--enable_admin_endpoints` + auth)
Both routes 404 unless the server was started with `--enable_admin_endpoints`,
and require a Bearer token when it was. Neither is exposed as an MCP tool.
- `POST /admin/dump_all_conversations` `{ "output_folder": "…" }` → dumps all conversations to JSONL.
- `POST /admin/generate_mcp_token` `{ "channel_id":"…", "user_id":"…", "expires_days":365 }` → long-lived token for MCP clients.

## CommandOutput shape (inside `command_outputs[*]` on a turn response)

```json
{
  "command_response":
    { "response": "Your order W123 was cancelled.",
      "success": true, "artifacts": {}, "next_actions": [], "recommendations": [] },
  "workflow_name": "", "context": "", "command_name": "", "command_parameters": ""
}
```
Prefer the turn's top-level `answer` for the assistant's final message. Per-command
text and artifacts live at `command_outputs[*].command_response`. When `command_name`
is `"ask_user"`, roles invert: `command_parameters` holds the agent's question and
the response holds the user's answer; `success=false` means the question is still open.

## Results and waiting turns (what a client renders)

### The result of a command
A command that declares `Signature.Output` returns it, and `command_outputs[*].command_response.output` holds that
model as a plain dict; `response` is the same data as JSON text. A command with no `Signature.Output` has `output`
null. Render `output` rather than parsing `response`. A value can be a JSON string (a workflow may wrap
`json.dumps(...)` in a one-field `Output`); decode it. `GET /command_schemas` (auth) lists, by qualified command name, `{kind, input, output}`: the `Input` and
`Output` JSON Schemas, with each field's `title`, `description`, `examples`, `enum`, bounds, the optional `ui` hint
(`hidden`, `hero`, `format`, `table`, `tones`, `currency` on an `Output` field, `widget` on an `Input` field), `classifier_confirms.meanings`, and
`available_from` (the commands whose output feeds the field). `kind` is `"execution"` or `"planning"`.
Cache the listing per connection.

### When a turn is waiting for the user
A turn waits in one of two shapes. Test for both:
1. `status` is `"awaiting_user"`: the agent, a planning command, an approval or a review asked. The open question is the
   last entry of `command_outputs`, with `command_name` `"ask_user"`, the question in `command_parameters`, and
   `command_response.success` false.
2. `status` is `"completed"` and the last entry's `command_response.artifacts.awaiting_user` is `true`: the user typed
   a `/` command that needs approval (nothing ran). The turn is not suspended, but the next message answers it.

### Prompt keys (on the last entry's `command_response.artifacts`)
A waiting turn's last entry carries `question_kind` and, for five kinds, a payload. These keys are UI protocol, not
user artifacts: do not list them as artifacts, and they are not on the answered entry's final answer. The server adds
them to a waiting turn only; once the prompt is answered, `question_kind` stays on its entry but the payload
keys are left off the turn's final answer.

| `question_kind` | Meaning | Payload key | The next message must be |
|---|---|---|---|
| `agent` | the agent asked an open question | none | free text, the answer |
| `planning_parameter` | an agent's planning command call was refused for an input | `parameter_request` | free text; lines of `<field name>: <value>` (the reply is added to the original request and read by the parameter-extraction model; no code parses it). One question names every argument the call lacks (`parameter_request.missing`, `invalid`), and `values` holds the arguments already obtained. With no agent model (`LLM_AGENT` unset) the question is asked again while each reply makes progress (at least one argument asked about is no longer missing); a reply that makes none is asked once more, a second one ends the part as not run with a reason naming exactly the arguments still missing; `cancel` declines |
| `approval` | a command that needs approval is about to run | `approval` | exactly `yes`, `always` or `no` (lower case; `Yes` and `yes please` are not accepted) |
| `review` | a sub-task or plan review | `review` | `yes`, `no`, or edits (`change 2: goal=...`, `drop 3`; grammar in `review_engine.parse_edits`) |
| `phase_step` | no-agent mode only (`LLM_AGENT` unset): a planning command paused at a conditional step or at a command step missing one plain value | `phase_step` | for a condition exactly `yes` or `no` (`y`/`true`/`n`/`false` also read; anything else is asked once more); for a missing value the value alone; `cancel` declines and ends the command with its `fw_not_run` stop (`reason` "declined"). The data also carries `later_phases` (the phases after this one; `remaining` is only this phase's rest). While it is pending, every message is the reply, a `/`-prefixed one too (`invoke_agent` strips the `/`, as for `ask_user`; `invoke_assistant` keeps it and the context removes it); no command runs beside the question. No verified identity needed; the chat page shows it as a plain question |
| `plan_choice` | no-agent mode only (`LLM_AGENT` unset): a part of the turn matched no planning command, and the turn asks which one is meant | `plan_choice` | a plan's name from `choices` (with or without its context, or written with spaces) or the number the question lists it under; `none` when none fits and `cancel` to decline (either ends the part with its `fw_not_run` entry); anything else is asked once more. The part resumes in the same turn: do not send `use_plan` with the reply.
| `unknown` | the server could not classify it | none | free text; show the entry's text |

```json
{ "question_kind": "approval",
  "approval": { "command": "Record/add_note", "parameters": {"text": "Ann"},
                "origin": "agent", "irreversible": true, "can_always": true } }
```
`origin` is `"agent"` or `"user"` (a `/` command); `can_always` is false when the server has no identity to record a
grant against, so hide "Always allow" and never send `always`.

Under `--chat_approval_surface host` (the default) the response that asks (the stream's `output` frame, `data.approval`, or the
`/invoke_assistant` body) also carries `{"approval": {"record_id": ..., "token": ...}}` beside the turn fields. It is
live-only (never in a stored turn or `GET /turns`). Keep it out of the UI and of any log, and send it once as
`approval_token` on the request that answers the prompt. Under `host` (the default) an unsigned server refuses the command. Under `channel` no token is delivered and
`approval_token` is ignored. The chat page does exactly this.
```json
{ "question_kind": "planning_parameter",
  "parameter_request": { "command": "Library/book_by_title", "missing": ["title"], "invalid": [], "values": {} } }
{ "question_kind": "review",
  "review": { "kind": "sub_tasks", "title": "Sub-tasks", "revision": 0,
              "items": [ { "goal": "...", "outcome": "...", "inputs": [ { "name": "member", "value": "Emma", "from_subtask": null, "drafted": false } ],
                           "means": { "kind": "from_scratch", "name": "" }, "check": "" } ],
              "changes_data": [false] } }
```
`review.kind` is `"sub_tasks"` or `"plan"`; a plan's `items` are `{goal, steps: [text]}`. `changes_data[i]` is true when
item i runs a command that writes data. `revision` rises by one each time an edit re-drafts the review, and a review edit
re-prompts without a new `ask_user` entry, so re-read the keys on every awaiting turn.

### A failed `/` command (assistant mode)
A `/` command whose parameters could not be filled, or whose name is ambiguous, answers with the error text and no
`question_kind` (the turn is not `awaiting_user`). Its last entry's `artifacts` may hold:
- `parameter_request`: `{"command": "record_values", "missing": ["ratio", "ids"], "invalid": [], "values": {"name": "Bob", "count": 3}}`.
  `values` are the fields the user did give (no not-given placeholders). Absent when the refusal names no field.
- `plan_choice` (with `question_kind: "plan_choice"`, the same question and shape the plan path asks): the qualified command
  names the message could mean, in order, without duplicates, as `choices`:
  `{"request": "...", "sub_task": null, "choices": [{"name": "Library/find_book", "description": "...", "closest": true}], "none_reply": "none", "decline": "cancel", "ambiguous": true, "misunderstood": false}`.

Neither is ever sent in agent mode. To answer a `parameter_request`, send `/<command name> name=value ...` (the leaf
name, without a context prefix); to answer `plan_choice`, send the chosen command's name (or its number, or `none`), optionally followed by its parameter text; `abort` or `cancel` ends the round.
In assistant mode `name=value` is bound in code (`/add_two_numbers first_num=2 second_num=3`, no model call); XML and
positional forms remain agent-only, and a leading `/` is dropped before the command name is matched.

### `answer` is the default rendering; the data is there too

The turn's `answer` is the default plain-text rendering. Every part of it that the core writes in code is also available as data, and a client may present from the data and ignore the text. Where a key is listed, it is in `command_outputs[*].command_response.artifacts` of the entry named; the answer text is the same with or without it.

| Text the core writes | Data | Where, and shape |
|---|---|---|
| Parts of a multi-part answer (`### N. title` headings, "Part i of n:" labels of a step-cap hand-off) | `fw_sub_tasks` | the turn's last command output; one entry per sub-task: `number`, `text`, `title`, `answered` (the part has text under its heading), `not_run` (`true` only in no-agent mode, for a part with an `fw_not_run` entry: its text is the stop's explanation, not a command's answer; absent otherwise), `start`, `end` (offsets of the part's answer in `answer`), and `runs` only for a part that was continued: `[{index, of, start, end}]`, the offsets of each run's answer after its label |
| A review or plan proposal and the replies it accepts | `review` with `question_kind: "review"` | the waiting entry; `kind`, `title`, `revision`, `items`, `changes_data`, `reply_forms`: `[{kind: accept/decline/edit/add/drop, reply or pattern or free_text, per_item, separator?}]`, the replies the shown hint promises (the reply reader's own cues are wider); an ambiguous part is asked the `plan_choice` question before the review, so a review lists no other plans; the typed reply `step N: use NAME` switches a step to any planning or execution command of the workflow (it has no entry in `reply_forms`) |
| An approval question | `approval` with `question_kind: "approval"` | the waiting entry; `command`, `parameters`, `origin`, `irreversible`, `can_always` |
| A which-command question (one shape on every path) | `plan_choice` with `question_kind: "plan_choice"` | `request` (the text asked about), `sub_task` (its number in a split turn, else `null`), `choices` (`{name, description, closest}` each), `none_reply` (`"none"`), `decline` (`"cancel"`), `ambiguous` (the choices are the commands one request fits) and `misunderstood` (nothing fitted and the choices are every command). Plan path: a part no planning command matched or more than one command fits, the waiting entry of an `awaiting_user` turn. Deterministic ("/") path: an ambiguous or unmatched request, the last output of a failed COMPLETED turn (human session only). Reply with a name or the number the question lists it under, `none`, or `cancel`; a name may be followed by its parameter text on the deterministic path |
| A refused planning call or a failed "/" command's missing values | `parameter_request` | `command`, `missing`, `invalid`, `values` |
| The capability answer to "what can you do?" (written in code) | `capabilities` | one `framework` command output of its own in `command_outputs` (not drawn as a result), equal to the data on the final answer's output; planning commands with their example, the current context's commands, the other-context lines |
| A planning command that stopped, or a part not run, without an agent model (`LLM_AGENT` unset: no-agent mode) | `fw_not_run` | the turn's last command output (a turn that ran no command carries it on a command output of its own); a list, one entry per stopped part: `kind` (`not_run_here`), `command` (`null` for an ad hoc part), `phase`, `open_step` (`{number, text, kind, reason}`, `null` for an ad hoc part), `done`, `facts`, `remaining`, `suggested_plan` (a numbered continuation written with no model call: the open step, the rest of its phase, each later phase's steps as the template has them, then the line asking the client to write the answer), `means`, `sub_task` (the part's number in a split turn, else `null`), and when they apply `later_phases`, `nested` (the stop of a planning command this one called), `reason`, `request` (an ad hoc part only: that part's own request text), and `available_plans` (an ad hoc part only: the planning commands available here (those the matcher can choose from: the active context, its bases, its live ancestors' contexts and the root's), `{name, description, closest}` each, the matcher's closest ones to the part first, flagged `closest`, then the rest by name; the answer text leads with "No matching plan was found for this part", then lists them as markdown, at most 12 and a count of the rest, and says "No planning commands are available here." when there are none; in a split turn the part with the full list prints it once and a later ad hoc part with the same plans lists only its own closest ones and points to it). On a turn that can ask (attended, no blocking queue) such a part is first asked about (`question_kind` `plan_choice`, above) and gets this entry only after `none`, `cancel` or two unreadable replies. To run such a part with one of those plans in a new turn, send the optional `use_plan` of `/invoke_agent`, `/invoke_agent_stream` or the MCP `invoke_agent` tool with `user_query` = `request` and `use_plan` = a name from `available_plans`: one part, no split, no match, values drafted from the request, then the plan's parameter question instead of a review (the intent is settled, so no goal or steps are shown: one form with the plan's name and description and every parameter, the drafted values prefilled and the missing required ones marked; Run or `yes` confirms, `name: value` lines change values, `cancel` stops; one still missing is asked for again while replies make progress); an unknown or out-of-context name is HTTP 400 / MCP isError. `open_step.number` is the step's own number in its phase's plan (as `done` and `remaining` number theirs); `suggested_plan` is renumbered from 1 and `open_step.plan_number` is the open step's place in it; `facts` holds finished phases only. The turn is COMPLETED and its answer is a short explanation written in code; wording the final answer is the client's |
| A capability answer written in code (agent mode, a workflow with planning commands) | `capabilities` | the answer's own `CommandResponse` (the `CommandOutput` that `process_message` returns and the CLI shows); it is not in `command_outputs`, which has no entry for that turn. `{planning_commands: [{name, title, example}], commands: [{name, title}], other_contexts: [{context, title, commands, more}]}` |
| The `what_can_i_do` listing | `capabilities` and `command_response.output` | `{workflow: {name, contexts, context_inheritance}, context, commands: [{name, qualified_name, context, planning, approval_marker, description, parameters, outputs, sample_utterances}]}` |
| `abort` ("command aborted") | `aborted` and `output` | what it ended: `{stage, command?, command_name?}`; `output` is `{command, command_name}` |
| `you_misunderstood` | `command_choices` and `output.valid_command_names` | the command names it lists, in its order |
| A command's own result | `command_response.output` | the typed `Signature.Output` |

## Workflow file formats

### Command file (single-file pattern, preferred) — `_commands/<command_name>.py`
```python
import fastworkflow
from pydantic import BaseModel, Field
from ..application.<module> import <callable_or_class>

class Signature:
    plain_utterances: list[str] = ["cancel order W123", "please cancel my latest order"]

    class Input(BaseModel):
        order_id: str = Field(description="…", examples=["W123"])   # no default: required, so a missing value is asked for

    class Output(BaseModel):
        status: str

class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: "Signature.Input") -> "Signature.Output":
        # call the app's real business logic here; return the Output (a str also works)
        result = <callable_or_class>(...)
        return Signature.Output(status=str(result))
```
- A parameter with no default is required: a missing value is asked for, not guessed. `Optional[str] = None` makes it optional.
- Use `Field` `description` + `examples` (and `pattern`/`min_length` where helpful) to drive accurate parameter extraction.
- Optional rules: `@model_validator(mode="after")` methods on `Input`. A validator corrects a value by assigning
  to `self.<field>` and returning `self`, refuses by raising `fastworkflow.Refused(message, field=...)`, and reads
  the context from `info.context` (`command_text`, `workflow`, `command`). Value lookups
  (`DatabaseValidator.fuzzy_match`) go in a validator too. The `validate_extracted_parameters` hook is removed (a load
  error names the validator form). `db_lookup` is retired: a `json_schema_extra={"db_lookup": True}` key or a
  `Signature.db_lookup` method is a load error.
- For LLM-generated responses, use `fastworkflow.utils.dspy_utils.dspySignature(Signature.Input, Signature.Output)` with `dspy.Predict`.

### Context model — `_commands/context_inheritance_model.json`
Each context entry has at most two keys:
- `"/"` — list of command names available in that context.
- `"base"` — list of parent context names whose commands are inherited.

To add a command: add its name under the relevant context's `"/"`, then create the matching
`_commands/<command_name>.py`. Every declared command must have an implementation file or routing
validation fails.

## Environment variables (set in fastworkflow.env)

LLM model strings (the templates default them to `mistral/mistral-small-latest`):
`LLM_SYNDATA_GEN`, `LLM_PARAM_EXTRACTION`, `LLM_PLANNER`, `LLM_AGENT`, `LLM_CONVERSATION_STORE`. Matching keys live in
`fastworkflow.passwords.env` as `LITELLM_API_KEY_<ROLE>`.

`LLM_RESPONSE_GEN` (key `LITELLM_API_KEY_RESPONSE_GEN`) is used only by `POST /compose_answer_card` and
`POST /compose_question_form`. The shipped `fastworkflow.env` template sets it to `mistral/mistral-small-latest` (a Bedrock model needs no key entry: AWS credential chain); unset, those
endpoints answer `{"card": null, "reason": "no_model"}` / `{"form": null, ...}`.

LiteLLM Proxy: prefix model names with `litellm_proxy/`, set `LITELLM_PROXY_API_BASE` in the env
file and `LITELLM_PROXY_API_KEY` in the passwords file (per-role keys are then ignored).

## Troubleshooting

- **PARAMETER EXTRACTION ERROR** — the command's `Field` descriptions/examples are too weak, or the user query lacks a required value. Improve the signature or ask the user.
- **Crash on run** — a corrupted `___workflow_contexts` folder; delete it and rerun.
- **Command not recognized** — import/syntax error in the command file; it failed to load. Check logs.
- **Missing API keys** — keys absent from `fastworkflow.passwords.env`.
