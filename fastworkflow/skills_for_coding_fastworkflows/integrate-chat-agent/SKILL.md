---
name: integrate-fastworkflow-chat-agent
description: >-
  Integrate a fastWorkflow-based agentic chat agent into an existing application so users can
  execute the app's business logic and ask questions in natural language, with a popup chat UI
  that streams internal workflow/assistant conversations before the final answer. Use when a
  developer asks to add a chat agent, conversational UI, natural-language interface, or
  fastWorkflow backend to their app, or mentions "AI-enabling" their application.
---

# Integrate a fastWorkflow Chat Agent

## Improve the integrated workflow

Use `fastworkflow run_chatbot` as the developer workbench for the application workflow. In an
existing session, inspect **benchmark → experiment → conversation** (or ad-hoc conversations by
UTC date) on the left, and turns/components on the right. The right breadcrumb retains the full
path. Add component-level **Human feedback** to record what should improve.

For repeatable verification, use [create-workflow-benchmarks](../create-workflow-benchmarks/SKILL.md)
to author benchmarks and task prompts and create experiment identities. The UI does not execute
those tasks: connect an application runner using the experiment and task IDs. Use
[debug-workflow-conversations](../debug-workflow-conversations/SKILL.md) for diagnosis and
[optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md) to verify changes
against recorded outcomes. Human annotations do not automatically update the agent or train models.
When a family of requests needs the same several commands in a known order, write a skill
(`_commands/<name>.py`) with [author-planning-commands](../author-planning-commands/SKILL.md).

## Mission

> Your mission is to integrate agentic chat UI within the application's UI that allows natural
> language conversations to execute the business logic functionality and provide human readable
> answers to questions. All information that is exposed via the UI should also be available via
> the chat interface. Use fastWorkflow (`uv add fastworkflow`) to implement the backend for this
> chat functionality.
>
> This chat window should pop up when an overlaid chat icon is clicked in the web UI. Users should
> be able to start new chats and also continue previous chats. The chat window should stream
> internal fastWorkflow conversations between workflow and assistant (same UX as fastWorkflow cli
> invoked via `fastWorkflow run`) before giving the final answer.

fastWorkflow AI-enables an existing **Python** application by wrapping its classes and methods with
an intent-detection + parameter-extraction + execution pipeline. The chat backend is hosted as the
fastWorkflow **FastAPI service**, which exposes streaming HTTP endpoints the chat UI consumes.

## Track all work in beads

Use **beads (`bd`)** for issue tracking, not markdown TODOs. Install it if missing
(`npm install -g beads` or see beads docs). Create one epic for this integration and tasks under it
that mirror the workflow phases below. Mark tasks `in_progress` / `closed` as you go.

```bash
bd create "Integrate fastWorkflow chat agent" -t epic -p 1 --json
bd create "Build & train fastWorkflow from app business logic" -t task -p 1 --deps discovered-from:<epic-id> --json
bd create "Host fastWorkflow FastAPI streaming service" -t task -p 1 --deps discovered-from:<epic-id> --json
bd create "Build popup chat UI with streaming + conversation history" -t task -p 1 --deps discovered-from:<epic-id> --json
```

## Workflow

Copy this checklist and track progress:

```
- [ ] Step 1: Discover the application's business logic to expose
- [ ] Step 2: Install fastWorkflow and scaffold the workflow directory
- [ ] Step 3: Write command files for each app capability
- [ ] Step 4: Set up env files, then PAUSE for the user to add API keys
- [ ] Step 5: Train the workflow's intent models
- [ ] Step 6: Smoke-test the agent from the CLI
- [ ] Step 7: Host the fastWorkflow FastAPI streaming service
- [ ] Step 8: Build the popup chat UI (new/continue chats + live trace streaming)
- [ ] Step 9: End-to-end verification
```

### Step 1: Discover the business logic to expose

Enumerate every capability the app's UI exposes (and any read-only "questions" users may ask). The
hard requirement: **everything available in the UI must also be reachable through chat.** Map each
UI action / data view to a Python class + method or function in the app's codebase. Group related
operations into classes (contexts) that hold state. This mapping becomes the command set.

If the app's backend is **not** Python, fastWorkflow still runs as a separate Python sidecar
service; the commands call into the app via its existing API/SDK/DB client instead of in-process
imports.

### Step 2: Install fastWorkflow and scaffold

```bash
uv add "fastworkflow[server,training]"    # or: pip install "fastworkflow[server,training]"
fastworkflow examples fetch hello_world   # provides fastworkflow.env + fastworkflow.passwords.env templates
```

Create the workflow directory next to the app code:

```
<app>/chat_workflow/
├── application/        # symlink or thin wrappers calling the app's real business logic
├── _commands/          # command files; a context is a folder _commands/<Context>/
├── fastworkflow.env
└── fastworkflow.passwords.env
```

Add `___command_info` and `___convo_info` to `.gitignore`. Session state lives outside the workflow folder, under
`FASTWORKFLOW_STATE_ROOT` (default `~/.local/state/fastworkflow`).

### Step 3: Write the command files

For each capability mapped in Step 1, create a command file that invokes the app's business logic:
`_commands/<command_name>.py` for the root context, `_commands/<Context>/<command_name>.py` for a
context (a class that holds state). A context's `class Context` in `_commands/<Context>/_<Context>.py`
declares `inherits = [...]` (the contexts whose commands it also has) and, beside its `get_parent`,
`parents = [...]` (the contexts whose records `get_parent` returns). Use Pydantic `Input`/`Output` signatures with strong `Field`
descriptions and examples. A required field (no default) is what makes a missing parameter get asked
for rather than hallucinated; `Optional[...] = None` makes it optional. See [reference.md](reference.md)
for the command-file structure and context format.

Strong field descriptions are the floor, not the finish. Four colocated skills cover the features
that decide whether the agent can actually complete a multi-step task, and are worth reading before
writing the first command rather than after the first bad demo:

| Read | For |
|---|---|
| [declare-parameter-producers](../declare-parameter-producers/SKILL.md) | `from_command_output` producer edges on `Input` fields. Any parameter that is an id, uid or code the user cannot invent needs one, or the planner cannot work out what to call first. This is the single highest-value thing in this step. |
| [validate-command-parameters](../validate-command-parameters/SKILL.md) | The `Input`'s `@model_validator`s: cross-field rules, context preconditions, and normalising a value in place so a near-miss does not cost a round trip, and resolving a typed name against live keys (a typo corrected, or a "did you mean"). |
| [design-context-models](../design-context-models/SKILL.md) | How to group commands into contexts. Read it once the tree is more than a handful of commands, or as soon as intent routing starts confusing two of them. |

The first three form one pipeline per parameter: the `from_command_output` edge says where a **missing** value
comes from, the fuzzy lookup in the validation hook fixes one that arrived **nearly right**, and
the same hook normalises or refuses (`fastworkflow.Refused`) one that arrived **unusable**. A parameter
worth taking is usually worth declaring in at least one of the three.

### Step 4: Set up env files, then PAUSE for API keys

Copy `fastworkflow.env` and `fastworkflow.passwords.env` into the workflow directory. The service
needs LLM API keys to run. **Stop here, tell the user the absolute path of
`fastworkflow.passwords.env`, and ask them to add the keys.** Do not attempt to invent or commit
keys. The keys (Mistral small / OpenRouter free tiers work):

```
LITELLM_API_KEY_SYNDATA_GEN=...
LITELLM_API_KEY_PARAM_EXTRACTION=...
LITELLM_API_KEY_PLANNER=...
LITELLM_API_KEY_AGENT=...
LITELLM_API_KEY_CONVERSATION_STORE=...   # conversation topic/summary
LITELLM_API_KEY_OBSERVATION_SEARCH=...
LITELLM_API_KEY_RESPONSE_GEN=...         # optional: composed answer cards and question forms
# JEV_API_KEY=...                        # optional; required if LLM_AGENT is left unset
```

### Step 5: Train

```bash
fastworkflow train <app>/chat_workflow <app>/chat_workflow/fastworkflow.env <app>/chat_workflow/fastworkflow.passwords.env
```

Training generates synthetic utterances with the `LLM_SYNDATA_GEN` model and fits the local intent
model into `___command_info/` (the default layout fits in seconds once the utterances exist). The first
run also downloads the sentence encoder from Hugging Face. Retrain after changing commands.

### Step 6: Smoke-test from the CLI

```bash
fastworkflow run <app>/chat_workflow <app>/chat_workflow/fastworkflow.env <app>/chat_workflow/fastworkflow.passwords.env
```

Ask "what can you do?" and exercise a few real commands. Prefix a command with `/` to force
deterministic (non-agentic) execution. Confirm the agent reaches every UI capability from Step 1.

### Step 7: Host the FastAPI streaming service

The chat backend is the bundled FastAPI-MCP service (requires the `server` extra). It exposes JWT
auth, conversation persistence, and the streaming endpoint that powers the live trace UX.

```bash
python -m fastworkflow.run_fastapi_mcp \
  --workflow_path <app>/chat_workflow \
  --env_file_path <app>/chat_workflow/fastworkflow.env \
  --passwords_file_path <app>/chat_workflow/fastworkflow.passwords.env \
  --port 8000
```

Run this alongside the app's backend (e.g. as a sidecar/separate process) and proxy it from the
app's server, or call it directly from the frontend. Full endpoint contracts (request/response
shapes, auth flow, streaming format) are in [reference.md](reference.md). The same server
serves MCP at `POST /mcp`; for a client that is itself an agent, leave `LLM_AGENT` unset and set
`JEV_API_KEY` (reference.md, "No-agent mode").

### Step 8: Build the popup chat UI

Required UX (matches the `fastWorkflow run` CLI experience):

1. **Overlaid chat icon** in the app UI that toggles a chat window.
2. **New chat** button → `POST /new_conversation`.
3. **Continue previous chats** → `GET /conversations` lists past conversations;
   `POST /activate_conversation` restores one into the window.
4. **Stream internal conversation** → send each user message to `POST /invoke_agent_stream` and
   render the `trace` events live (the back-and-forth between workflow and assistant) *before*
   rendering the final `output` event. This is the core differentiating UX: do not collapse it
   into a single final answer. A `timeout` frame means "still working", not a failure.
5. **Answer the workflow's questions.** A turn with `status` `awaiting_user` is asking the user
   something (which command, a missing value, a review, an approval, the agent's own question);
   `artifacts.question_kind` on its last command output says which, with data for a form or
   buttons. The user's next message is the reply. Render each kind as
   [reference.md](reference.md) "Waiting turns" describes.

Auth flow: call `POST /initialize` once per user/session to get a JWT, send it as
`Authorization: Bearer <token>` on subsequent calls, and refresh via `POST /refresh_token`.

A channel runs one turn at a time: disable sending while a turn runs (a second request is `409`),
and if the stream breaks, read the turn back with `GET /turns/{key}` instead of resending (resending
runs it again). A `503` with `Retry-After` means the server is at its session ceiling.

### Step 9: End-to-end verification

- Click the chat icon → window opens.
- Send a natural-language command → trace events stream, then a human-readable final answer.
- Send a request missing a required value → the question appears and the reply completes it.
- Verify every UI capability from Step 1 is reachable through chat.
- Start a new chat, send messages, then reopen a previous conversation and confirm history restores.
- Close any open `bd` tasks with a reason.

## Key references

- HTTP API contract, auth, streaming format, command-file & context-model structure: [reference.md](reference.md)
- fastWorkflow concepts, CLI, env vars: project `README.md`
