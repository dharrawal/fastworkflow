# fastWorkflow Chat Integration Reference

Contracts of the FastAPI service (`fastworkflow.run_fastapi_mcp`) and the workflow file formats. Read this when wiring
the chat UI to the backend or fixing command files. The service's full operator guide (session limits, auth modes,
MCP tool results) is `fastworkflow/run_fastapi_mcp/README.md`; Swagger is served at `/docs`.

## Hosting the service

```bash
python -m fastworkflow.run_fastapi_mcp \
  --workflow_path <workflow_dir> \
  --env_file_path <workflow_dir>/fastworkflow.env \
  --passwords_file_path <workflow_dir>/fastworkflow.passwords.env \
  --port 8000
```

Requires `pip install "fastworkflow[server]"`. Other flags: `--host` (default `0.0.0.0`), `--context` (JSON string),
`--startup_command`, `--startup_action` (JSON file path, or an inline JSON object starting with `{`; with neither given, the
workflow folder's `startup_action.json` runs as every session's first turn, and a flag replaces it),
`--expect_encrypted_jwt` (verify JWT signatures; off by default for trusted networks), `--cors_origin <origin>` (CORS
is open to `*` without it), `--enable_admin_endpoints`, and `--chat_approval_surface {host,channel}` (default `host`:
an approval-gated command needs a signed JWT with a `user_id`, and the reply carries a token; `channel` accepts a
plain `yes`, for local development tools only).

The MCP streamable-HTTP endpoint is mounted at `POST /mcp` on the same server (tools `invoke_agent`,
`invoke_assistant`, `get_turn`, `get_latest_turn`, `cancel_pending`, `end_session`, the conversation tools,
`get_command_schemas` and the feedback tools; an MCP client sends the same Bearer token).

## Authentication

1. `POST /initialize` returns `access_token` and `refresh_token`.
2. Send `Authorization: Bearer <access_token>` on every authenticated endpoint.
3. On 401, call `POST /refresh_token` with the **refresh** token in the `Authorization` header.

Tokens are scoped to a `channel_id` (a session key you choose, one per user chat) and an optional `user_id`.

## One turn per channel, and the session ceiling

- A channel runs one turn at a time. The same request sent again while it runs joins that turn (`202`, poll its key);
  a different request is `409 {"detail", "reason": "channel_busy", "turn_key"}`. Never resubmit a request to find out
  what happened: once it has finished it runs again.
- `timeout_seconds` (default 60) is how long the call waits, not a limit on the turn. Past it `/invoke_agent`,
  `/invoke_assistant` and `/perform_action` answer `202 {turn_key, exec_state: "running", progress}` and the turn keeps
  running; the stream sends a non-terminal `timeout` frame and stays open.
- Read a turn back with `GET /turns/{turn_key}` (`{exec_state: "running", progress}` while it runs, the turn object
  when done), `GET /turns/latest` (the channel's running turn, else its last one) and
  `GET /turns/{turn_key}/trace` (its stored trace, repeatable). Only the caller's own channel is readable; any other key
  is `404`.
- The server holds at most `MAX_LIVE_SESSIONS_HARD` live sessions (default 4 x `MAX_LIVE_SESSIONS`, which defaults to
  50). A call that needs a new session at that ceiling, and cannot make room by releasing an idle one, is refused with
  `503` and `Retry-After: 30`: wait and retry. A session whose workflow context class has no `get_state` hook cannot be
  checkpointed and stays in memory until it expires or ends, so implement `get_state` / `from_state`
  ([design-context-models](../design-context-models/SKILL.md)) and call `POST /end_session` when a chat is finished.
  If the server dropped a channel's state, the first turn result (and `/initialize`) carries
  `session_notice: {code: "context_dropped", reason, message}` once.

## Endpoints

### POST /initialize  (public)
Create or resume a session and obtain tokens.
```json
{
  "channel_id": "user-123",
  "user_id": "user-123",
  "stream_format": "ndjson",
  "startup_command": null,
  "startup_action": null,
  "context": null,
  "timeout_seconds": 60
}
```
`stream_format` is `"ndjson"` (default) or `"sse"` and fixes the framing of `/invoke_agent_stream` for the session.
`startup_command` and `startup_action` are mutually exclusive (400 if both) and need `user_id`. Precedence, top wins: the request's `startup_action` or `startup_command`; then the process flags (a process `--startup_action` still wins over a request `startup_command`); then the folder's `startup_action.json`, which applies only when none of those is given. `context` overrides
the server's `--context` for a session this call creates. Response: `access_token`, `refresh_token`, `token_type`,
`expires_in`, and, when a startup ran, `startup_output` (its turn object) or, if it is still running (`202`),
`startup_turn_key` to poll. `503` with `Retry-After` at the session ceiling.

### POST /refresh_token  (public)
Header `Authorization: Bearer <refresh_token>`. Returns `access_token`, `refresh_token`, `token_type`, `expires_in`.

### POST /invoke_agent_stream  (auth): the primary chat endpoint
```json
{ "user_query": "find the book Dune", "timeout_seconds": 60 }
```
Optional: `unattended` (true: nobody can answer, so the turn asks no questions), `approval_token` (see Approvals),
`use_plan` (see No-agent mode) and `use_dspy_cache`. A leading `/` is stripped: this endpoint always runs the agent
path; send deterministic commands to `/invoke_assistant`.

Response headers, sent before the first frame (both CORS-exposed): `X-FW-Turn-Key` (the execution key, your recovery
handle) and `X-FW-Stream-Format` (`ndjson` or `sse`: read it rather than assuming).

**Frames.** `trace`: one agent and workflow interaction. `progress`: the stage the turn is in (split, match, draft,
review); ignore it if you do not show it. `timeout`: NOT terminal; the turn passed `timeout_seconds` and is still
running, so show "still working". `output`: TERMINAL, the turn object; a failed or `awaiting_user` turn arrives here
too. `error`: TERMINAL, the turn produced no output. Exactly one terminal frame ends the body.

- **NDJSON** (`application/x-ndjson`), one object per line:
  ```json
  {"type":"trace","seq":0,"turn_key":"<exec key>","logical_turn_key":"<key>","data":{}}
  {"type":"timeout","seq":1,"turn_key":"<exec key>","logical_turn_key":"<key>",
   "data":{"detail":"...","timeout_seconds":60,"still_running":true}}
  {"type":"output","seq":2,"turn_key":"<exec key>","logical_turn_key":"<key>","data":{}}
  ```
- **SSE** (`text/event-stream`): the same events, `id:` is `seq`, `event:` the type and `data:` the payload alone.

`seq` starts at 0, is gapless and unique per stream: drop a `seq` you have rendered. A network chunk can split a frame,
so buffer to the delimiter (`\n` for NDJSON, a blank line for SSE).

**Recovery: read, never resubmit.** If the body dies, the turn keeps running. Poll `GET /turns/{key}` and fetch missed
interactions from `GET /turns/{key}/trace`. Start from `X-FW-Turn-Key`, but keep the `logical_turn_key` the frames
carry: the execution key resolves only while the turn is live, the stored turn by the logical key. The same query
sent again while it runs is `202 {turn_key, exec_state: "running", logical_turn_key, reason: "duplicate_submission"}`
with no body; a different one is `409`.

### POST /invoke_agent  (auth)
The same request and turn, not streamed: the body is the turn object plus a `traces` array. Use it only when streaming
is not feasible.

### POST /invoke_assistant  (auth)
A deterministic turn (no agent, no planning) for a known command: `user_query` is the command line, for example
`find_book query=Dune` (`name=value` is bound in code; a leading `/` is added when missing). Same request fields as
`/invoke_agent` except `use_plan`.

### POST /perform_action  (auth)
Run one command directly, with no intent detection or parameter extraction.
```json
{ "action": { "command_name": "add_two_numbers", "parameters": { "first_num": 2, "second_num": 3 } }, "timeout_seconds": 60 }
```
`422` for a malformed action. While a question is pending it runs nothing and answers with the refusal text (see
Waiting turns).

### POST /cancel_pending  (auth)
Drops whatever question the channel is waiting on, of any kind (the agent's own question included), and its saved
state. Returns `{"status": "ok", "cleared": <bool>}`.

### POST /end_session  (auth)
Body optional: `{"discard_pending": true}`. Releases the channel's live session (a checkpointable one is saved and
restored by the next call). Returns `{channel_id, ended, pinned, pending_discarded}`. `409` while a turn runs, or
while a question is pending unless `discard_pending`; `503` when the state could not be saved. Repeating it is safe
(`ended: false`).

### Conversations  (auth)
- `POST /new_conversation`: save the current conversation (one model call for its topic and summary) and start a
  fresh one. Use it for **New chat**. A pending question is parked with the old conversation.
- `GET /conversations?limit=20`: `[{conversation_id, topic, summary, created_at, updated_at}]`, newest first. A blank
  `topic` means none was generated yet. Use it for the **history list**.
- `POST /activate_conversation` `{"conversation_id": 7}`: make a saved conversation current. Use it for **continue a
  previous chat**.

Both mutating calls are `409` while a turn runs.

### GET /command_schemas  (auth)
`{<qualified command name>: {kind: "execution" | "planning", input: <JSON Schema>, output: <JSON Schema>}}` for every
command of the workflow: each field's `title`, `description`, `examples`, enum and bounds, the field options the author
declared in `json_schema_extra` (such as the `ui` hints), and `available_from` (the commands whose output feeds the
field). Cache it per connection.

### POST /compose_answer_card and POST /compose_question_form  (auth): optional
A chat client may call these after a turn for a model-composed answer card or question form. They read no session
state, are not MCP tools, and use the `LLM_RESPONSE_GEN` model (key `LITELLM_API_KEY_RESPONSE_GEN`). Everything in the
request, command outputs included, is sent to that model, so offer users a switch.

```json
POST /compose_answer_card
{ "user_message": "find the book Dune", "answer": "<the turn's answer>", "status": "completed",
  "results": [ { "id": "r1", "command": "Library/find_book", "output": {"book_uids": ["b1"], "titles": ["Dune"], "total": 1},
                 "titles": {"total": "Books found"} } ],
  "parts": null }

POST /compose_question_form
{ "question": "<the pending question>", "user_message": "find the book Dune", "results": [] }
```
- `results` are the turn's command outputs (`command_response.output`), at most 60; `id` is yours. Decode JSON strings
  and one-key wrappers in each `output` as the server does (`fastworkflow.run_chatbot.composer.normalize_output_dict`),
  or the refs will not resolve on your side.
- `parts` (answer card, optional): the `fw_sub_tasks` artifact of a split turn, forwarded as it is. Malformed or not
  matching the answer, it is ignored and the answer's `### N.` headings are read instead.
- Always `200`: `{"card": {v, lead, sections: [{heading, verdict, blocks}], blocks}}` or `{"form": {fields: [{name,
  label, kind, options: [{value, label}], required, help}]}}`, else `{"card": null, "reason": ...}` (or `"form"`) with
  `no_model` (no `LLM_RESPONSE_GEN`: tell the user once), `failed`, `timeout` (over 40 seconds) or `not_needed` (the
  plain view is as good; not an error). On any null keep your plain view.
- A block names data by `ref` (`r1.total`, `r1.titles`): read the value from your own copy; the server returns no
  values. It drops what it cannot ground in the turn's data (a number or ID not in the results, a ref that does not
  resolve, a quote not in the answer, a form option that is not a value in the results or the question).
- `422` for more than 60 results, an answer over 150,000 characters or a question over 20,000.

### Feedback  (auth)
`POST /post_feedback` records one free-form review note on a turn or a component of it:
```json
{ "turn_key": "<turn_key>", "target_kind": "turn", "target_label": "Turn", "span_ids": [],
  "provenance": "coding_agent", "category": "recommendations", "subcategory": "what_to_do",
  "comment": "Plan all three items before answering." }
```
`category` / `subcategory` pairs: `observations_analysis` / `observation`, `analysis`; `conclusions` /
`what_went_right`, `what_went_wrong`; `recommendations` / `what_to_do`, `what_not_to_do` (`GET /feedback_taxonomy`
lists them). `GET /feedback?turn_key=...` reads a turn's notes. Notes never reach a model's prompt.

### Probes (public) and admin
`GET /probes/healthz` (liveness) and `GET /probes/readyz` (`200` ready, `503` not ready). The admin routes
`POST /admin/dump_all_conversations` and `POST /admin/generate_mcp_token` (a long-lived token for an MCP client) are
`404` unless the server runs with `--enable_admin_endpoints`, and then need a Bearer token.

## The turn object

Every turn endpoint, the stream's `output` frame and `GET /turns/{key}` return:
```json
{ "turn_key": "<exec key>", "exec_state": "done", "logical_turn_key": "<key>",
  "status": "completed", "failure_reason": null, "success": true,
  "answer": "<the final message>", "command_outputs": [] }
```
`status` is `completed`, `awaiting_user`, `failed`, `cancelled` or `abandoned`. A failed turn is still HTTP 200: read
`status`. Show `answer` as the assistant's message. Each `command_outputs[*]` is:
```json
{ "command_response": { "response": "...", "success": true, "artifacts": {}, "next_actions": [],
                        "recommendations": [], "output": {} },
  "workflow_name": "", "context": "", "command_name": "", "command_parameters": null }
```
`output` is the command's `Signature.Output` as a dict (null when it declares none): render it rather than parsing
`response`. An artifact value may be inline or an offloaded reference `{"__fw_artifact_ref__": "<id>", "size", "content_type"}`.

## Waiting turns: questions a client must render

A turn waits for the user when `status` is `awaiting_user`, or when it is `completed` and the last command output's
`artifacts.awaiting_user` is `true` (a `/` command the user typed needs approval). The question is the turn's `answer`
(and the last entry of `command_outputs`, an `ask_user` entry). That entry's `artifacts.question_kind` says which
question it is, and a payload key carries its data. These keys are UI protocol, not results: do not list them as
artifacts. Re-read them on every waiting turn (a review edit re-asks with a new `revision`).

The user's next message is the reply. It answers the question and nothing else: the which-command, parameter and
decision-step questions end with "Only your answer is read; anything else in the reply is ignored." While a question
is pending, a `/` message is also the reply (the `/` is dropped), and `/perform_action` or a `use_plan` request is
refused with a text that says how to drop the question. To drop any question, call `POST /cancel_pending`, or reply
`cancel` (`no` for a review or an approval). `cancel` typed to the agent's own question is just an answer to it.

| `question_kind` | Asked when | Payload key | Reply |
|---|---|---|---|
| `agent` | the agent asked an open question | none | free text |
| `plan_choice` | a part of the request fits more than one command, or (no agent model) no planning command | `plan_choice` | a name from `choices` (with or without its context, or with spaces), the number it is listed under, `none` (none fits) or `cancel`; a reply that cannot be read is asked once more, then the part is dropped |
| `planning_parameter` | a planning command lacks or refused a required value, or a `use_plan` plan shows its parameters | `parameter_request` | `name: value` lines; the value alone when one is open; `yes` runs when `confirm` is true; `cancel` drops the part |
| `review` | the turn's one review of its parts, before anything runs | `review` | `yes`, `no`, or edits as `reply_forms` lists them |
| `approval` | a command that needs approval is about to run | `approval` | exactly `yes`, `always` (only when `can_always`) or `no` (lower case) |
| `phase_step` | no agent model: a planning command paused at a yes/no step or a step missing one value | `phase_step` | `yes` / `no`, or the value (or `<field>: <value>`); `cancel` ends the command |
| `unknown` | the server could not classify it | none | free text |

**`plan_choice`**: `{request, sub_task, choices: [{name, description, closest}], none_reply: "none", decline: "cancel",
ambiguous, misunderstood}`. `ambiguous`: the choices are the commands the request fits; `misunderstood`: nothing fit,
so every command is listed. One button per choice sending its `name`, plus None and Cancel, is a complete UI. The
same shape comes on a deterministic (`/`) turn that was ambiguous: a `completed` turn with `question_kind`
`plan_choice` on its last output; there a name may be followed by that command's parameter text.

**`parameter_request`**: `{command, missing: [names], invalid: [names], values: {name: value}}`. One question names
every missing or refused value. It is asked again while each reply fills or fixes one of them; the second reply in a
row that does not ends it (with an agent model the agent takes over; without one the part ends as not run, naming
what is missing). A `use_plan` form adds `confirm: true`, `description` and `parameters: [{name, required, value,
state, description}]` (`state` `missing`, `invalid` or null): prefill the values, mark the missing ones, and send
`yes` (Run) or `name: value` lines. A deterministic (`/`) command that cannot fill its parameters answers with a
`completed` turn whose last output carries the same `parameter_request` and `question_kind` `planning_parameter`; reply the same way, or
send the command again in full; `abort` ends it.

**`review`**: `{kind: "sub_tasks", title, revision, items, changes_data, reply_forms}`. Each item is `{goal, outcome,
inputs: [{name, value, from_subtask, drafted}], means: {kind, name}, check}`; `changes_data[i]` is true when item i
writes data. `reply_forms` lists the replies the shown hint promises (`{kind: accept|decline|edit|add|drop, reply |
pattern | free_text, per_item, separator?}`, for example `change N: goal=text` and `drop N`, several joined with `;`).
The typed reply `step N: use NAME` switches a step to another command. In no-agent mode an item that will not run here
carries `not_run_here: true` and `plan_preview`.

**`approval`**: `{command, parameters, origin: "agent" | "user", irreversible, can_always}`. Hide "Always allow" and
never send `always` when `can_always` is false (it is always false for a `REQUIRED_ALWAYS` command, which asks
every time; a sent `always` is not read as `yes` and the question comes back). Under `--chat_approval_surface host` (the default) the response that
asks also carries `{"approval": {"record_id", "token"}}` beside the turn fields (the stream's `output` frame, or the
body). It is live only (never in a stored turn). Keep it out of the UI and logs, and send it once as `approval_token`
on the request that answers.

**`phase_step`**: `{command, phase, step: {number, text, kind}, decision, decline: "cancel", asked, done, facts,
remaining, later_phases}`, where `decision` is `{kind: "conditional", condition, run_if_yes, replies: ["yes", "no"]}`
or `{kind: "missing_value", field, type, description, reply}`. `cancel`, `abort`, "you misunderstood" or a different
request end the command at that step as a `fw_not_run` stop whose `reason` starts with `declined:`; a reply that gives
nothing usable is asked once more.

## Data beside the answer

The `answer` is plain text written for a person; what the core writes in it is also data in
`command_outputs[*].command_response.artifacts`:

| Key | On | Shape |
|---|---|---|
| `fw_sub_tasks` | the last output of a split turn | one entry per part: `number`, `text`, `title`, `answered`, `start`, `end` (offsets of its answer in `answer`), `not_run` (no-agent mode), `runs` (a continued part) |
| `capabilities` | the answer to "what can you do?" | the planning commands with an example, the current context's commands, the other contexts |
| `fw_not_run` | the last output, no-agent mode | see No-agent mode |
| `aborted` | the reply to `abort` | what it ended: `{waiting_for?, command?, command_name?}` |
| `command_choices` | the reply to "you misunderstood" | the command names it lists |

## No-agent mode: a client that is itself an agent

With `LLM_AGENT` unset or empty the server runs no ReAct agent, for a client (an MCP or HTTP agent) that does its own
reasoning. `JEV_API_KEY` is then required: with neither set the server refuses to start. Routing, parameter
extraction, the planner and splitter (`LLM_PLANNER`) and Jev still run. A single command and a planning command run;
a planning command runs its code steps and pauses at a decision step (`phase_step`). Where code cannot go on, the turn
is `completed` and its last output carries `fw_not_run`, a list with one entry per stopped part:

`kind` (`not_run_here`), `command` (null for an ad hoc part), `phase`, `open_step` (`{number, text, kind, reason}`),
`done` (the steps run), `facts`, `remaining`, `suggested_plan` (numbered text for the client to carry out), `means`,
`sub_task`, and when they apply `later_phases`, `nested`, `reason`, `results` (each `done` step's result text),
`partial_read` (`{pages, rows_read, total, rows}` of a read cut at its page cap), and for an ad hoc part `request` and
`available_plans` (`[{name, description, closest}]`).

The answer text is written in code: a stop with steps already run opens with what ran, each step's result and the facts
recorded (a page-cap stop also lists the rows read and says counts are partial); an ad hoc part says no matching plan
was found and lists the planning commands available. Wording the final answer is the client's job.

An ad hoc part (no planning command matched) of a turn that can ask is first asked about (`plan_choice`). To run one
that came back not run, send a new request with `user_query` = the entry's `request` and `use_plan` = a name from
`available_plans` (`/invoke_agent`, `/invoke_agent_stream`, or the MCP `invoke_agent` tool). It runs as one part with
that plan and asks its parameter form (`planning_parameter` with `confirm`) instead of a review. An unknown name, one
not available here, or a question still pending is `400 {"detail", "reason": "use_plan_refused"}` (MCP `isError`).

## Workflow file formats

### Command file: `_commands/<command_name>.py`, or `_commands/<Context>/<command_name>.py`
```python
import fastworkflow
from pydantic import BaseModel, Field
from ..application.<module> import <callable_or_class>   # one more dot inside a context folder

class Signature:
    """Cancel an order."""   # the first sentence describes the command to the intent model

    class Input(BaseModel):
        order_id: str = Field(description="...", examples=["W123"])   # no default: required, asked for when missing

    class Output(BaseModel):
        status: str

    plain_utterances = ["cancel order W123", "please cancel my latest order"]

class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        result = <callable_or_class>(...)   # the app's real business logic; workflow.record is the context's object
        return Signature.Output(status=str(result))
```
- A field with no default is required: a missing value is asked for, not guessed. `Optional[str] = None` is optional.
- `Field` `description` and `examples` (and `pattern`, bounds) drive parameter extraction.
- Rules go in `@model_validator(mode="after")` methods on `Input`: correct a value by assigning `self.<field>` and
  returning `self`, refuse with `raise fastworkflow.Refused(message, field=...)`, and read `info.context`
  (`command_text`, `workflow`, `command`). Value lookups (`DatabaseValidator.fuzzy_match`) go there too.
- A command may also return a `str`, or raise `fastworkflow.Refused(message)` to fail with that message.

### Contexts: folders, and `_commands/context_inheritance_model.json`
A context is a folder `_commands/<Context>/` holding its commands and `_<Context>.py` (its `Context` class:
`get_parent`, `get_displayname`, and `get_state` / `from_state` so the server can checkpoint a session). Commands at
the top of `_commands/` are in the root context. The JSON file only declares inheritance: each entry is
`{"<Context>": {"base": ["<Parent>", ...]}}` (an empty `{}` is valid; a `"/"` key is a load error). See
[design-context-models](../design-context-models/SKILL.md).

## Environment variables

In `fastworkflow.env`: `LLM_SYNDATA_GEN`, `LLM_PARAM_EXTRACTION`, `LLM_PLANNER`, `LLM_AGENT`, `LLM_CONVERSATION_STORE`,
`LLM_OBSERVATION_SEARCH` (the template sets each to `mistral/mistral-small-latest`). The matching keys are
`LITELLM_API_KEY_<ROLE>` in `fastworkflow.passwords.env`; `JEV_API_KEY` there is optional with an agent model and
required without one. `LLM_RESPONSE_GEN` (key `LITELLM_API_KEY_RESPONSE_GEN`) is used only by the two compose
endpoints; unset, they answer `reason: "no_model"`.

LiteLLM Proxy: prefix model names with `litellm_proxy/`, set `LITELLM_PROXY_API_BASE` in the env file and
`LITELLM_PROXY_API_KEY` in the passwords file (the per-role keys are then ignored).

## Troubleshooting

- **A command file does not load**: `fastworkflow run`, `fastworkflow train` and the server print the problem at
  start. Fix the import or the error it names.
- **PARAMETER EXTRACTION ERROR** or a repeated parameter question: the `Field` descriptions and examples are too weak,
  or the request lacks a required value.
- **Missing API keys**: keys absent from `fastworkflow.passwords.env`; with no `LLM_AGENT` and no `JEV_API_KEY` the
  server does not start.
- **503 from `/initialize`**: the session ceiling; retry after `Retry-After` and end finished sessions.
