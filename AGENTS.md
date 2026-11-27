# Agent Instructions

This file provides guidance to AI coding agents when working with code in this repository.

Activate the local `.venv` Python environment before running tests or scripts.

**On a CUDA machine, `poetry install` alone gives you CPU torch.** The default is
pinned to CPU-only wheels, because Poetry cannot lock torch from the CPU and CUDA
indexes at once. GPU users need both steps:

```bash
poetry install --with gpu && make install-gpu-torch
```

There is no warning if you skip it — training silently runs on CPU, which is much
slower and easy to mistake for the suite or the model being slow. `make
install-gpu-torch` verifies CUDA is usable before reporting success, so run
`python -c "import torch; print(torch.cuda.is_available())"` if you are unsure
which you have.

## Project Overview

`fastWorkflow` is a Python framework for building NLP-driven workflows and AI agents with deterministic or LLM-powered business logic. It enables "AI-enabling" existing Python applications by wrapping their classes and methods with an intent-detection and parameter-extraction pipeline built on DSPy, PyTorch/Transformers, scikit-learn utilities, and Pydantic.

## Testing Philosophy

(from `.cursor/rules/testing_rules.mdc`):
- Don't use Mock fixtures — all tests are integration tests against real components
- Use the real test workflows in `tests/example_workflow/`, `tests/hello_world_workflow/`, and `tests/todo_list_workflow/`
- Do NOT remove pytest tests without explicit user approval
- A test that asserts behaviour without the optional planner (`JEV_API_KEY`) must use the `no_planner_key` fixture
  from `tests/conftest.py` (`@pytest.mark.usefixtures("no_planner_key")` or a fixture argument). `monkeypatch.delenv`
  alone is not enough: `get_env_var` falls back to the loaded env files when the process does not set a variable, and
  the planner LM and matcher are `lru_cache`d. When the planner becomes mandatory, the tests that use this fixture are
  retired. Tests never read a developer's `.env`: `import fastworkflow` stops litellm loading one.

### Never run two full suites at once

The full suite loads BERT-family models, and two concurrent runs exceed this box's
31 GB. The second run dies to the OOM killer partway through — observed twice at
~69%, reported as `PYTEST_EXIT=137`, which reads like a crash in whatever test was
running rather than a resource problem. If you are working alongside another agent,
**serialise the suite runs**; a run measured against a tree another agent is still
editing is worthless anyway, so waiting costs nothing.

The suite takes ~2 hours 7 minutes — ~5,200 tests, measured 2026-10-03 (4896
passed, 11 failed, 304 skipped in 2:07:22). It said ~50 minutes before that
(1860 tests, measured 2026-08-20 at v3.1.1), and ~24 minutes before that, from
back when the suite was ~495 tests; if you budget against an old number you will
conclude a healthy run has hung. Budget for it rather than backgrounding it and
hoping.

### On a 24 GB box, run the suite in chunks of files

Measured 2026-11-24: the suite is about 11,550 tests in 541 files, and ONE pytest process over all of it grew to
about 20 GB (13.6 GB resident plus 6.9 GB of swap), filled swap on a 24 GB box and stalled at 77% after an hour. Run it
as serial chunks of about 30 files, each a fresh process, one at a time (19 chunks, about 65 minutes in all). A suite
started detached (`nohup ... &` from a shell without job control) inherits SIGINT ignored; `tests/test_cli_turn_control.py`
installs Python's handler itself for that reason.

### Browser checks skip themselves unless you ask for them

Every DOM test begins by reading `TEST_JSDOM_ROOT` and skipping when it is
unset, so `pytest` on a machine without jsdom silently executes no browser code
at all. Claiming "the browser checks passed" after a plain run is therefore
claiming nothing. When a change touches the chatbot UI, run the gate, which
discovers the DOM tests, runs them, and fails when a required one skipped:

```bash
source .venv/bin/activate
TEST_JSDOM_ROOT=/path/to/dir-with-node_modules-jsdom python -m tests.browser_validation
```

`TEST_JSDOM_ROOT` is a directory containing `node_modules/jsdom` (the harness
scripts require it from there by path). On the owner's box that is
`~/.cache/fw-jsdom`; the location is configuration, so nothing in the repo
assumes it. Pass test files to narrow the run, `--list` to see what would run.

### The plan head: one marker, one switch, one recording

The built-in plan metadata extraction workflow's `ProposalReview` intent head is local and gitignored, so a clean checkout and CI
have none. It is an embedding-layout head (`fastworkflow train <plan path>` writes it in seconds from the
utterance cache, with no model call and a determinate result: the same rows give the same head). Tests that assert the head's own quality (the held-out corpus score, the "a trained head asks again on a
vague reply" class, the drift test) carry `@pytest.mark.requires_plan_head` (`tests/conftest.py`): they skip
with a reason naming the missing head and how to get it (`fastworkflow train` on the plan metadata extraction workflow path), and
`FW_REQUIRE_HEADS=1` (the process environment or an env file, `fastworkflow.raw_env_var`) turns that skip into a failure.
The session header names the head's current version id. Tests of the review's reply READER do not need the head: they
run on the head's recorded answer for each reply (`tests/recorded_router.py`, fixture `recorded_review_router`,
recordings in `tests/fixtures/review_reply_router_answers.json`), placed in `CommandRouter._instances_cache` where
production looks the router up; a reply with no recording fails loudly. After a retrain, run
`test_recorded_router_matches_the_head` (it lists drifted replies), then re-record with
`pytest --record-review-router` over EVERY test file that uses `recorded_review_router` (a reply is recorded only when
a test asks it, so a file left out keeps the previous head's stale answer; the list is in `tests/recorded_router.py`)
and review the fixture diff reply by reply: a reply that became a confident `confirm`, `decline` or `drop`, and a
confident reply that became a list, are the ones to read. Other example heads (hello_world, library) still skip ad hoc.

## fastworkflow CLI

Run `fastworkflow --help` for the full command list (`examples`, `train`, `run_cli`, `run_chatbot`, `run_fastapi_mcp`, `build`, `refine`). Non-obvious behavior:
- `run_cli` defaults to agent mode; `--assistant` runs deterministic mode.
- Prefix a command with `/` at the interactive prompt to force deterministic (non-agentic) execution. While a question is pending (of any kind: an agent question, a plan's question, a review, an approval, a decision step, the deterministic path's own question) a `/` message is the reply to it with the leading `/` removed, never a command beside it (`WorkflowExecutionContext._route_message`; `/approvals`, `/revoke_approval` and `/cancel_pending` keep their meaning). `process_action` and `use_plan` sent while a question is out are refused with one text (`AWAITING_ANSWER_REFUSAL_MESSAGE`, which also says how to drop the question: reply `cancel`, `no` for a review or an approval, `//cancel` at the CLI prompt, or call `cancel_pending` over HTTP/MCP, which silently drops every kind of pending question, the agent's own included).
- `run_cli` prompt: `//cancel` cancels whatever the session is waiting on: a question, review, approval, plan which-command or value question, or paused decision step (`ChatSession.cancel_pending_prompt`: the CANCEL event of `waiting.step` on the turn's one waiting record, `fastworkflow/core/waiting.py`); Ctrl-C or `--timeout_seconds N` (default off) stops waiting on a turn, which finishes in the background with its output discarded (`ChatSession.abandon_turn`, `turn_gate.py`, `run_cli/turn_wait.py`).
- `train` validates seed coverage and duplicate capabilities, reuses generated-data caches,
  and automatically chooses selective or full retraining. `--regenerate-utterances` is the
  only training-policy override.

## Architecture: Three Phases

```
Build-Time → Train-Time → Run-Time
```

**Build-time** (`fastworkflow/build/`): AST-introspects your Python application and generates `_commands/*.py` files. It writes no `context_inheritance_model.json` (a removed spelling) and never edits a `_<Context>.py` that exists: it prints each context's derived `inherits = [...]` line, the file it belongs in and whether that file declares it or something else, and writes the line only into a `_<Context>.py` it creates (`build/context_model_generator.py`); it lists every context whose `get_parent` has no `parents` beside it (parents cannot be derived from code), and names a leftover `context_inheritance_model.json` without reading it.

**Train-time** (`fastworkflow/train/`): Generates synthetic utterances (via LLM + HuggingFace `datasets`) and fits the workflow's local intent model. Outputs go to `___command_info/` inside the workflow directory; publication is atomic and retains only current plus one previous recovery point. The default layout (`INTENT_MODEL_LAYOUT=embedding`) embeds the utterances with a frozen sentence encoder and fits one linear head for the whole workflow (seconds, no model is trained); `workflow` fine-tunes one large classifier for the whole workflow and `per_context` a classifier pair per context, with PyTorch/Transformers (see Environment Variables). The framework's own workflows (plan, command_metadata_extraction) are always in the `embedding` layout (`FRAMEWORK_LAYOUT`), with the default encoder and targets whatever the application's env says.

**Run-time**: A three-stage pipeline for every user turn:
1. **Intent Detection** – one chain: the trained local intent model identifies the target command when it is confident; otherwise the remote decision model (Jev) is asked when `JEV_API_KEY` is set; otherwise the user or the agent is asked
2. **Parameter Extraction** – DSPy + Pydantic validates and extracts inputs
3. **Command Execution** – runs your business logic and generates a response

**Topology B** (current): `WorkflowExecutionContext` is synchronous and transport-free. FastAPI embeds it per-request by calling `process_message` directly. `ask_user` suspends trajectory via `CommandCancelledError` and resumes on next message. `ChatSession` adds optional queues for the CLI `keep_alive` loop. **One success rule, both paths:** a turn that ran to its end is COMPLETED, and whether a part did what was asked is said by that part's own entry (its command output's `success`, the `fw_not_run` entry); FAILED is only for a turn that could not be completed (agent exhaustion, an error of the framework itself). So an execution command whose own code raises on the "/" path is a COMPLETED turn whose command output is failed (`WorkflowExecutionContext._failed_output_of_raising_command`: the message `Execution error in <command>: <error repr>`, artifacts `error_type`, `error_message` and `traceback` (cut at 4000 characters, never in the message) as a failing `__result__` carries them, and the traceback also in the log and on the dispatch span), as on the plan path. What a turn is waiting on is ONE record, `WorkflowExecutionContext._waiting` (`fastworkflow/core/waiting.py`; kinds `agent`, `intent`, `parameters`, `review`, `approval`, `decision`; a review and an approval point at the approval store's record, which the store keeps and whose end ends the turn's record); `_awaiting_user` is read from it (nothing sets it), and `waiting.step(record, event, facts)` is the one decision of what a message, a cancel and a restore do to it. The saved blob holds it under `waiting` (`SCHEMA_VERSION` 8) and a parked conversation parks that one record.

## Environment Variables

Two env files per workflow (see `fastworkflow/examples/fastworkflow.env` for a template):

- `fastworkflow.env` — model strings (`LLM_AGENT`, `LLM_PARAM_EXTRACTION`, etc.), logging, intent model IDs
- `fastworkflow.passwords.env` — API keys (`LITELLM_API_KEY_AGENT`, etc.)

**One precedence rule for every setting** (`fastworkflow.get_env_var` / `raw_env_var`): the process environment (a shell
export, a container or CI variable) wins over the env files `fastworkflow.init` loaded, and the env files win over the
code default; an empty process value counts as set. litellm, when imported, would copy the
first `.env` it finds searching upward from its own installed file into the process environment, where it would beat
the workflow's env files; `import fastworkflow` turns that off by setting `LITELLM_MODE=PRODUCTION` (its only effect in
litellm) unless `LITELLM_MODE` is set (`fastworkflow._stop_litellm_loading_a_dotenv_file`), and warns when litellm was
imported first. The
import-time `LOG_LEVEL` can see only the process environment
(`init` applies it again from the files); it is read in any case and a bad value warns (`utils/logging.parse_log_level`).
`fastworkflow build` and `refine` load the env files as `train` does (`utils/env.init_from_env_files`;
`--env-file-path`, `--passwords-file-path`, default the workflow folder's own files when present).

**One required model** (`dspy_utils.resolve_role`, the rule in `fastworkflow/utils/model_roles.py`, bd fix-ygxo): the
base model is `LLM_AGENT`, else `LLM_PLANNER`; `LLM_PLANNER` falls back to `LLM_AGENT`; `LLM_SYNDATA_GEN`,
`LLM_PARAM_EXTRACTION`, `LLM_CONVERSATION_STORE`, `LLM_OBSERVATION_SEARCH`, `LLM_POLICY_INTENT` and
`LLM_COMMAND_METADATA_GEN` fall back to the base; `LLM_AGENT` never inherits, and `LLM_RESPONSE_GEN` and the distillation
roles do not fall back. The key follows the model: the role's own `LITELLM_API_KEY_*`, else the key of the role its model
came from, else the base role's key only when the provider prefix matches; never a cross-provider key, no global key
(`litellm.api_key` is not set); a `litellm_proxy/` model uses `LITELLM_PROXY_API_KEY`, a `bedrock/` model needs none. A
fallback is said once per role per process at INFO, and provenance records the resolved model and `model_from`. No base
model (neither `LLM_AGENT` nor `LLM_PLANNER`) is a startup problem of `run` and the servers
(`no_agent_mode.BASE_MODEL_PROBLEM`); `fastworkflow train` stops at start when `LLM_SYNDATA_GEN` resolves to no model.
The generator model is in neither the training signature nor the utterance- or parameter-example-cache keys, so changing
it never regenerates (`scripts/rekey_generation_caches.py` re-keyed the caches written before). The three
`SYNTHETIC_UTTERANCE_GEN_*` counts default to 4, 5 and 1.

`LLM_AGENT` unset (or empty) is **no-agent mode** (`fastworkflow/core/no_agent_mode.py`; the one decision is `agent_model_specified()`), for a server whose client is itself an agent. The routing classifier, parameter extraction, `LLM_PLANNER` (splitter and text planner) and Jev stay in use, so **`JEV_API_KEY` is then mandatory** (and `LLM_PLANNER`, then the base model): neither set is a startup problem (`command_preview.load_problems(folder, runtime=True)`, which `fastworkflow run_cli` and the server ask for and `fastworkflow train` does not; `fastworkflow run_cli` prints it and exits 1, and the server raises a startup error and its process exits non-zero; `train` is unaffected). **Every turn resolves its parts before it asks anything else, in both modes** (`fastworkflow/_workflows/plan_metadata_extraction/resolution.py`, bd fix-o1lg.2): (1) every part in order: which command or plan (`part_intent`; asked only when the decision is ambiguous, the candidates, execution and planning commands, being the choices, or, with no agent model, when nothing serves the part and the turn can ask; an ambiguity with exactly ONE candidate is not asked: `resolution._bind_only_candidate` binds it and leaves the part unchosen, so the turn's one review is its confirmation, and a turn that cannot ask keeps today's behaviour; the CME's one-candidate question reads "Did you mean the command 'X'?", answered by its name, `yes` or cancel, because the deterministic path has no review; one of the three remaining differences below), then the values a bound planning command still lacks (asked only when it lacks one, once with an agent model and only where its inputs were drafted; a value an earlier part produces is never asked for), and, with no agent model, the values a part bound to an EXECUTION command lacks, through the same loop (`resolution._values_command`, `planning_commands.execution_command`: its `Input` stated as a command with parameters; the command named by the user, the one the local model is confident of (the gate's fast path asks first and then runs with no review, `part["fast_path"]`), the one picked at the which-command question and the one bound as the only candidate are all asked the same way); with an agent model the agent finds an execution command's missing value, as before; a refusal that names no argument (a validator's `Refused(message)` with no field, `SubTask.refused_without_field`) is NOT asked about: with no agent model the part ends as not run carrying the validator's own message, with an agent model the run hands it to the text planner or the agent; each question can be cancelled, which drops that part and the later parts go on; (2) then ONE review of the whole turn with nothing missing in it (none for a single part that names a command, that the local model is confident of for an execution command that does not change data, or whose command the user chose by answering (a pick: `none`, cancel or two unreadable replies leave the part ad hoc, and so reviewed); a command that changes data (`@writes`, or a planning command whose steps reach one) or carries an approval mark runs only when the user named it (typed its exact name, picked it at the which-command question, or a client pinned or called it directly) or after a yes, so one the local or remote model or a close spelling of its name chose is reviewed, unless its approval question will be asked anyway (`approval_binding.would_ask`: that question is its yes, so a command with both marks is asked once; a standing grant is not consent to the model's choice), and on the "/" path it is asked as "Did you mean the command 'X'?" (`intent_loop.unconfirmed_pick`, one rule for both paths; a person's own "/" message runs a `REQUIRED_UNTIL_DELEGATED` command with no approval question, so a model's pick of one is asked this way); an ad hoc part carries the badge `may change data` and needs the explicit yes below when the workflow has a command that changes data and an agent model is set; a pinned plan's parameter question is that turn's review; none when nobody can answer, or when no agent model is set and nothing would run; a reply that opens with an accept (`yes, but only list them`) in whose rest the free-text edit reader finds no edit, or only edits that change nothing, is an accept, and edits that change nothing are never read back (`review_engine._accept_lead`, bd fix-ftdl.8.28); on a turn with a part that changes (or may change) data the review accepts only an explicit yes (`no_agent_mode.is_yes`, the one reader of every yes: the approval question, the review, a pinned plan's parameter question, "Did you mean", a conditional decision step; `yes`, `true`, `ok`, `agreed`, `accept` or `go`, the whole reply, case and a trailing `.` or `!` ignored): any other reply that would have accepted, a code-read "go ahead" or "dont change anything" or a read of the review head included, asks again with "This request changes data, so I need an explicit yes to run it.", up to the review's cap of 8 unresolved replies, after which it expires and nothing runs); (3) then the run, which asks no more intent or parameter questions except a decision step inside a plan; a value the run refuses is not asked about again (a single part that names a command, planning or execution, has its missing values asked at the gate before it runs, as a picked one does, bd fix-ftdl.7.8); a command that fails or refuses (`fastworkflow.Refused`) when run as a single part's step with no agent model ends as a not-run entry whose reason carries the command's own message, as the `/` path shows it. Plan alternatives are not offered inside the review any more. The resolution's progress (which part it is at, what was bound) is one JSON-safe record, `resolution.PlanResolution`, saved with the session under the key `plan_resolution`; the question that is out, the which-command question (kind `intent`) or a values question (kind `parameters`, its `after` is `resolve_next` at the gate and `resume_part` while a part runs), is NOT in it: it is the turn's one waiting record (`WorkflowExecutionContext._waiting`, the same record the deterministic path's question is, saved under the key `waiting`; `PlanResolution` holds no question); the client-visible shapes (`question_kind` `plan_choice` and `planning_parameter`, `artifacts.plan_choice`, `parameter_request`) are as below. `part_intent` is a thin wrapper (the memo for the turn, `pinned`, `exclude`) around `intent_loop.resolve`, THE intent decision the CME makes too (see "One intent loop" below). A single command and a predefined planning command run; a planning command runs its code steps. A DECISION step pauses it like ask_user and resumes on the reply (`_suspend_for_decision`, `_read_decision_reply` in `workflow_execution_context.py`; the state is the turn's waiting record of kind `decision` (`WorkflowExecutionContext._waiting`, payload `{"decision": ...}`, saved under the key `waiting`; `_phase_run` no longer holds it) plus a minimal agent stash, serialised like any suspension; the turn is `awaiting_user`, `question_kind` `phase_step`, data in `artifacts.phase_step`): a conditional step whose inner step code can run (yes runs it through `CodeSteps`, no skips it) or a command step missing one plain required value (the reply is the value, validated as a code-written step); the reply is read by the one reply reader with the decision's own answer rule (`parameters.reply.read_decision`: yes or no for a conditional step, the value or a `<field>: <value>` line for a missing value), so it understands what a value question understands: a request for a moment (the same question comes back, nothing is counted the first time in a row, then each counts as no progress), and, by the framework's phrases or the remote model's read (one call at most, never for a yes or no or a named line, never with no key), "you misunderstood", a different request, `abort`; a reply that gives nothing usable is asked once more, then the stop; `cancel`, `abort`, "you misunderstood" and a different request all end the command at that step as a stop with reason "declined: ..." naming why (a step of a running plan has no which-command question to return to, and a reply is never run as a new request); only on an attended turn of the agent path. At any other open step (a prose step, a value that must be found, a failed step, no path, the page cap) or where its answer would need a model (no `__result__` text), it ends as a COMPLETED turn carrying the stop as data, the artifact `fw_not_run` (a list, one entry per stopped part: `kind` "not_run_here", `command`, `phase`, `open_step`, `done`, `facts`, `remaining`, `suggested_plan`, `means`, `sub_task`); the answer text of any stop with at least one step run by code (a non-empty `done`: an open step of any kind, a stop at a paged read's page cap, or the `answer` kind, where the command has no `__result__` text after every step ran, or its `__result__` returned `fastworkflow.planning.ANSWER_BY_MODEL`; a `__result__` that raises, returns `None`, an empty string or an unsupported value is NOT that stop: the call fails like an execution command's (a failed output, `fastworkflow.Refused` a refusal, no model call, bd fix-qjne.12), see `planning_phases.run_call_result`), opens with a code-written statement (`no_agent_mode.ran_statement`: what ran, each step's own result text under its step line (the command's response in full up to `RESULT_TEXT_CAP` = 4,000 characters a step and `RESULTS_TOTAL_CAP` = 16,000 for all steps together, a longer one cut at a line with a note of how much was not shown and that the full text is in `fw_not_run` `results`; the additive stop keys `results`, a `done` line to its text, and `partial_read` = `{pages, rows_read, total, rows}`, filled only for a page-cap stop while code runs the plan's steps), and the facts recorded, markdown, no model call) and the page-cap one lists the rows read so far with the total the command reported and says any count is of a partial read; an ad hoc part (no planning command matched) of a turn that can ask (attended, nothing ran in the part: the conditions of the parameter question) asks which planning command is meant (`resolution._intent_question`: `question_kind` `plan_choice`, data in `artifacts.plan_choice` = `{request, sub_task, choices: [{name, description, closest}], none_reply, decline}`, at most 12 numbered in the text with a count of the rest; the reply is read in code by the intent loop's one reader (`intent_loop.read` / `answer`, implemented in `intent_clarification.read`): a name, its leaf, the name with spaces, a listed number, a close misspelling (two that match equally well are unclear, never the first), `none` and its plain variants, cancel; a picked plan is written on that part's plan item (`resolution._bind`) and the resolution goes on to the part's parameter question when a required value is missing, with no task text, and then to the next part; `none` or `cancel` drops the part (it comes back as a not-run entry), a reply that cannot be read is asked once more (`no_agent_mode.MAX_NO_PROGRESS`) and then drops it; a split turn asks about each ad hoc part when it reaches it; the question is the turn's waiting record, saved under the key `waiting`, and the progress under `plan_resolution`); on a turn that cannot ask, and after `none`, `cancel` or two unreadable replies, it is not run and comes back as the same entry with the planner's text plan and `available_plans` (additive: `{name, description, closest}` for the planning commands available here (the lookup path the matcher chooses from, `workflow_agent.planning_names_on_path`), the matcher's closest ones first, then the rest; `no_agent_mode.available_plans`), an ad hoc entry's own `request` text, and `use_plan` (optional on `/invoke_agent`, `/invoke_agent_stream` and the MCP `invoke_agent`; `WorkflowExecutionContext.process_turn(use_plan=)`, checked by `check_use_plan`, `UsePlanError`) that runs one request as one unsplit part with that planning command pinned (no matcher call; the chat page's plan buttons under a part that was not run send it, while its buttons on a `plan_choice` question send the plan's name as the reply), through the round-1 draft for that plan (`draft_round1_proposal(pinned_plan=)`) and then NOT the sub-task review: the intent is settled, so it asks the plan's parameter question (`resolution._values_question` with `pinned`: `question_kind` `planning_parameter`, `parameter_request` with `confirm`, `description` and every parameter with its drafted value and state; a plan with no parameters, or an unattended turn, runs as drafted; replies read by the one reply reader, `parameters.reply.read_reply` (`parameter_clarification.advance(live=True)` calls it; it adds the fixed phrases of `abort` and `you misunderstood` and, for a reply that is not `name: value` lines, ONE remote-model read (none with no key; `remote_intent`, the CME's `classify_missing_value_reply`: wait, abort, misunderstood, different command) to the code rule of `parameters.read`; the first wait in a row leaves the question out and counts nothing, each further one counts as a reply with no progress (the count is `waited` in the waiting record's payload, never in the loop state), `you misunderstood` puts the part back to the which-command question (the misunderstanding question, every command the plan can run; a running part's question does the same, from a one-part resolution built with no model call, `resolution.ask_misunderstood_running_part`, bd fix-fpyb.12), a different request drops the part as not run and is NOT run as a new request, `abort` is `cancel`), and `parameters.read` is: cancel, yes only when every parameter was shown, `name: value` lines of the listed fields, the whole reply as the value when one is open, comma parts in order when several are open and the count matches) while the matched route keeps its review; a planning command's missing value is asked for through ONE parameter clarification loop (`fastworkflow/parameters/parameter_clarification.py`: `Extraction`, the one result contract of every extractor (the values of a command given some text and each parameter's state: given, missing, invalid with the reason, from an earlier part; produced by `planning_commands.argument_gaps` for a plan (the guards run again with each gap's parameter left out, no model call; the gaps a refusal carries are `Gap` records, never read back from its text) and by `parameter_extraction.extraction_of` for an execution command, and read by everything downstream, including the `parameter_request` artifact, `Extraction.request`), then `ParameterLoop` state, `read_reply`, `planning_question`, `advance`/`reopen`, and `record_round` for the CME's parameter stage; entered from `resolution._values_question` for a matched, picked or named plan and a pinned plan, and from `WorkflowExecutionContext._open_parameter_loop` for a typed call refused while its part runs on a turn that did not pass the gate), and the reply goes onto the part's plan item (`resolution._take_values`; for a refused typed call it resumes that part in place, `_resume_parameter_loop_unlistened`, after the part's text); no second split, review or matcher call, and earlier parts are not run twice; the loop is in the waiting record (`parameters`, owner `plan`), and the loop of a running part's question is read by the reply's re-run from `WorkflowExecutionContext._rerun_loop`, a value that lives only inside the one call that took the reply (never saved: a re-run that asks again puts a record that carries the whole loop; old saved sessions are not supported: the keys `plan_choice` and `parameter_loop` and the keys a blob written before them carried are ignored); a value an earlier part produces (`from sub-task N`) is never asked for; ONE asking rule for a missing or refused value, for a planning command and an execution command, in every session (`parameter_clarification`: `made_progress`, `stopped`, `no_agent_mode.MAX_NO_PROGRESS`; there are no policies): one question names every missing or refused argument (`planning_commands.argument_gaps`, `parameter_request.missing`/`invalid`/`values`), the draft keeps the values the checks stand behind, it keeps asking while each reply makes progress (fills or fixes a value the question listed), a reply with none is asked once more, the second one in a row stops it (`ParameterLoop.stalled`), and `cancel` always ends it (the part is not run and the later parts still run; an execution command ends); what "stop" hands over to is the only thing that depends on the session: with an agent model the refusal goes to the text planner or the agent (never asked a third time), otherwise the part ends as not run, or the command as a failed output, naming exactly what is missing; every reply is read by the shared extractor; a plan step missing one plain value (the decision step) reads its reply with the same reader; and its code-written answer text leads with that no matching plan was found, then lists them (markdown, at most 12 and a count of the rest; a later ad hoc part of a split turn with the same plans lists only its own closest and points to the part that lists them all). `fastworkflow run_cli` with no agent model starts in deterministic (assistant) mode and says so. The contexts that serve users ask for the mode (`detect_no_agent_mode`: `ChatSession`, the FastAPI session); one built without asking never enters it. Tests use the `no_agent_model` fixture (`tests/conftest.py`). With `LLM_AGENT` set nothing changes.

The CME parameter stage of an execution command runs on the same loop (bd fix-o1lg S6 and S7): when an extraction leaves a value missing or refused, the turn's ONE waiting record (`fastworkflow/core/waiting.py`, `WorkflowExecutionContext._waiting`, kind `parameters`, payload `loop` = `ParameterLoop.to_state()` of kind `execution`, and `after` = `run_command`) holds the values kept, the open names, the question as the user saw it and its `parameter_request`; `parameter_extraction.model_of` rebuilds the command's `Input` from it (open fields hold their not-given values) and `park` builds it from such an instance (the plan review's slot question). `Workflow.end_command_processing` clears it through the port, the session blob saves the record whole (`waiting`, with the CME's scratch keys `command` and `command_name` in its payload), and its `asks` is the extraction round (`retry_round_ordinal`, now known after a restore). The keys `stored_parameters`, `stored_parameter_round` and `missing_value_question` no longer exist (old saved sessions are not supported). What the CME waits for is read from its two pending records (`intent_loop.waiting_for`, see "One intent loop"); it holds ONE of them at a time (the built-in context holds no pending question of its own: it holds a port, `context["waiting"]`, which `intent_loop` and `parameter_clarification` read and write; `put` replaces the record, so opening one closes the other; a context built without an execution context gets a fallback `waiting.Holder`; the plan review's slot question is nested in the review record (kind `review`, payload `slot`, through the plan CME's `waiting.SlotPort`; a review is the turn's waiting record too, pointing at the approval store's record, `payload.gate` saying the sub-task gate opened it, and the span's `subtask_review` says `open` while it is out); the "/" round flag is `WorkflowExecutionContext.assistant_round` and the record's `assistant`, ended in one place, `end_round`; a handled CME output keeps the round while an intent or parameters record is out, so the next message, with or without a leading "/", is the reply to it in an agent or no-agent session too, and the round ends when the question does: answered, cancelled, two unreadable replies, `cancel_pending`). The reply to a parameter question is read OUTSIDE `_commands/` (`command_metadata_extraction/parameter_reply.read_reply` returns a `Reply`: an action of the frozen vocabulary `run`, `wait`, `different`, `replace`, `merge`, `stop` and its arguments) and `wildcard.py` only dispatches it (`_dispatch_reply`). What still differs from a plan part's loop: the question carries what only an execution command knows (the validators' own messages and suggestions, what was extracted so far, how to leave it) and opens with the sentence a plan's question opens with (`parameters.ask`); the reply is read by the same reader as a plan part's (`parameters.reply.read_reply`; `parameter_reply.read_reply` is only the adapter from its result to the six actions: `cancel` and `abort` are both `abort`, `you misunderstood` runs that command, the pending command sent again in full is `replace` (this path's own rule), wait / different command come from the one remote read, and a different request (only when the one remote read is made; it is skipped when the app context has `run_as_agent`, a person's "/" round in an agent or no-agent session included: `parameter_reply._missing_value_reply_intent`, pinned by `tests/test_jev_missing_value_reply.py`, the plan path makes it) is NOT run, as on the plan path (bd fix-ftdl.8.10): it counts as a reply with no progress, so the question comes back once under a line saying the reply was read as a new request and not run and how to run it (cancel, then send it), and the second in a row ends the command (`parameter_reply.read_reply`; the router's `different` action is left in the frozen vocabulary because `wildcard.py` imports and handles it, and no reader returns it)); and the one asking rule above ends it after two replies in a row with no progress: the command ends as a failed output naming what is missing, with no question pending (`ParameterExtraction._extract_impl`, in a human or an agent session alike; `abort` still ends it at once). The CME's parameter question carries `question_kind` `planning_parameter` (`waiting.TABLE`, the same value as the plan path's) beside its `parameter_request` on the completed `/` output, which the chat page's `/` form still draws (`tmRenderParameterForm` skips a tagged prompt only on a turn that is `awaiting_user`); the re-ask of an approval asked on the `/` form after an unreadable reply is an output of the turn like the first ask (`approval`, `awaiting_user` and `question_kind` artifacts, `_resume_approval_direct`) with the turn awaiting the answer.

**One parameter pipeline (bd fix-qoyz).** A planning command's arguments and an execution command's parameters go through the same stages, in `fastworkflow/parameters/`: `rules` (shared text helpers), `validate` (the one driver of an `Input`'s field checks and model validators), `check` (every guard rule once: `check_plan`, `check_execution`, `Gap` records), `fill` (the one model fill and the four forms an agent's own text is read in), `pipeline` (`extract`: fill, guard, validate, ONE second fill), `ask` (the one question builder, `word`) and `read` (the one reply reader). There is no per-kind switch; what still differs is which stages a caller runs and the audience (an agent reads the unchanged agent text, a person one plain sentence). The rules: the extractor's value of a `str` or `int` STANDS unless the field is marked `copied_from_command_text` (or `record_named_in_command_text`), which is the one explicit strict check, ONE guard on both kinds (`check.check_plan`; an execution command's marked fields reach it through `planning_commands.option_command` in `validate_parameters`, on every value given even when another is missing) (a `str` must be written in the request at word boundaries, "dun" is not written in "dune", and takes its spelling; an id cut short inside one identifier match is widened to it; an `int` must be a number the request writes, with or without any number in it; a marked value given and not written is refused, required or optional, and one not given is not checked; two marked `str` fields may hold one value only when the user's request writes it as often, the one duplicate rule: otherwise an optional one beyond that count is left not given WITH a note (`ask.left_empty_line`: on `argument_notes`, the span's `argument_left_empty`, the review item's `check_notes`, where the item shows the field removed, as the run leaves it, and before a question; not a refusal, no second fill), two required ones are refused (when the `Input`'s validators, run on the same values, refuse too, THEIR refusal is the one reported, on both kinds: `check.validators_first`, bd fix-qjne.38; also in the question built from a planning command's gaps and a review item's check line, `planning_commands.argument_gaps`, `review_proposals.rechecked`, but only when that refusal is the only gap: with any other gap the validators are not run and the framework's wording stands, and a validator refusal naming no field leaves nothing to ask, as a call-level refusal there does), the count reads the turn's own message on the plan and agent path (`check.user_written`), and nothing is reset silently; `Field(pattern=...)` neither marks nor exempts a field, and means one thing on every path: the whole value must match, `rules.pattern_refusal`, checked after a marked id is widened; the remote "was this value given" check, `signatures.fields_not_given`, asks about unmarked filled fields only, and never about a value the request writes at word boundaries (`rules.written_at`, the strict check's rule); `converted_from_command_text` does not exist; the unmarked fields whose value the request does not write are recorded as the span attribute `argument_check_not_in_request`, names only); a field's `examples` are NOT shown to the extraction model (they feed the generated few-shot examples and the hint in a question) and there is no example-copy check anywhere (not the check that was deleted, not the extractor's reward step, not the readers of an agent's own typed text: a value equal to an example is an ordinary value), while what the extraction model is told about a field comes from its declarations, the same on both kinds, from one builder (`planning_commands.option_sentences`, read by the extractor and the planner's typed fill): "Copy it exactly" only for a marked field, the `bool` type's "True only when the request asks for it", and a declared pattern as `Must match the pattern <regex>` (`rules.pattern_sentence`); the keys `enum`, `used_by` and `pattern` of `json_schema_extra` are removed and fail the load of either kind of command, and so does any other key that is not an option of its position, on every `Input` and `Output` field of either kind (one rule, bd fix-ftdl.8.22: `planning_commands.check_field_keys`, `check_output_field_keys`; an execution `Input` and any `Output` field also take the JSON Schema annotations such as `examples` (an execution command's parameter-example generator reads `description` and `examples` there; nothing reads them on a planning `Input`, which takes its options only), an `Output` field takes `ui` and no `Input` option) (allowed values are the `Literal` type; a `One of: a, b` description is prose; a text `Literal` is read as text by the extractor's adapter, so one value outside it refuses that field only, by name with its allowed values, and a member in another case or spacing takes its declared spelling, one rule `rules.allowed_member` for an execution field and a planning argument)); a marked `int` the request does not write is refused, never replaced, and an absent one is not checked; guards read only values the extractor gave; on every `str` field a value that is entirely one template token is refused, one function on every path (`rules.placeholder_like`, read through `signatures.classify_bracketed_value`: a variable name in braces or angle brackets judged by its shape alone, `{name}`, `<name>` or `<ID>`; `<Dune>` is the value Dune, `{"a": 1}` and `{Ann Lee}` are values as written (braces are never unwrapped), and an id such as `SKU_42` is a value), so is a value of only punctuation, and an unwritten value that only echoes the field's own name (`uid` or `your_item_uid` for `item_uid`) (there is no list of stand-in words, and a description's "One of: a, b" exempts no word from the pronoun and vocabulary screen); ONE second extraction pass, only when a guard or a validator refused a value the model produced and named the field (or every answer failed to parse), never for a value that is simply missing and never for a refusal that names no field (bd fix-ftdl.9: its second answer used to be taken whole, so a call could run with values the model rewrote that nobody was asked about), and it answers that refusal and nothing else: only the fields the first verdict left open (refused, reset or missing, `pipeline.Verdict.open`) are taken from it and every other field keeps the first fill's value (`pipeline.answered`; a first answer that did not parse takes it whole), and when the second pass raises the first answer stands and its refusal is handled as any refusal; a guard's refusal of one argument keeps its corrections of the others (a sure verdict, a widened id, the request's spelling, a member's declared spelling: `planning_commands.checked_around_gaps`); a refusal that names no input (a validator's `Refused(message)` with no field, a call-level guard refusal) is not asked: an execution command ends as a failed output with the message and no pending loop, a plan call goes to the text planner or the agent; validators see the request plus the replies as `command_text` on both kinds, and never the answer text of the turn's earlier parts (the guards of a later part read it, so a value an earlier part found counts as written; `planning_commands.checked_arguments(user_text=)`, bd fix-ftdl.10); a validator's rewrites are kept when it passes and dropped when it refuses, and a later check of the values never reads a marked value a validator wrote (the draft's inputs hold the value the checks accepted, `ExtractedArguments.unvalidated`, and the run's validators write theirs again); a default written as text ("0", "false", how a review input holds it) is not a given value (`check.is_given_value`); a review item's check (the "Cannot run ..." line) is worked out again from its inputs whenever they change, by an edit or a reply to the values question, with no model call (`review_proposals.rechecked`: the guards, then the validators), and a refused argument's line says how to change it with `change N: <field>=...`, also one the item does not show; a review edit whose field is a path (`sub_tasks[1].inputs.kind`) names the input of its last piece, and one that names no input of the item's command is asked again with the valid names (`review_proposals._edited_input_name`, which also holds each piece of a whole-input edit, `field="inputs"`); the free-text reader's edits that fail these checks, an item number, an input's declared choices or a means (an `add`'s or a `change`'s) that names no command a step can be bound to (`review_proposals._known_means`, the explicit switch's resolver `review_plan_switch.known_plan_names`; then the user is told the name is not a command available here, with up to 12 of the names, and nothing of the reply is applied: never an ad hoc step; a typed `change N: means=...` is held to the same rule, `review_proposals.edited_means`, and refused at once with no model call, bd fix-ftdl.11) are read ONCE more with the reason (`review_proposals.RepairSubTaskEdits`: one more planner-role model call, only on such a failure) before the user is asked, the one step a reply says it is about (`step 3's ...`, `change 3: ...`) overrides the model's step number, an edit that copies one input's value into another input the reply never names is left out and the read-back says so, and `key=` (an empty value) removes the input as `key=none` does; who is replying is decided by the value of `run_as_agent` AND the round (`ask.audience_of(context, assistant=<the "/" round flag>)`, the one definition: a person's "/" message in an agent session is assistant mode, so its question is worded for the person, carries `parameter_request`, and its reply gets the one remote read; the agent's own tool call gets none of these); two tests of the key's PRESENCE are left, in the CME's `wildcard.py` (the which-command ambiguity question's `for_agent` wording, which a "/" round in an agent session still words for the agent: no "only your answer is read" note) and in `what_can_i_do.py` (with `command_executor._attach_capability_listing`: the session's mode picks the listing's detail); editing those files forces a CME retrain. A reply (`parameters.read`) answers its question and nothing else: `name: value` lines, one open value takes the whole reply, several take comma parts in order only when the count and each part's type fit, `no`/`none`/`neither` are the value of a yes/no field or an allowed value and leave an optional field empty, `you misunderstood` only by its own phrases, a reply never fills a field the question did not list; known limit: prose with a matching comma count sent to several plain-text fields still fills them. A free-text value that is only a pronoun or only a word of the workflow's vocabulary reads as missing on both paths and its question says why (`Extraction.screened`, from `Gap.screened` for a plan's own check and from `signatures.word_screen_reasons` for the draft the plan path asks from, kept on the sub-task as the hidden `check_screened`): "the title 'it' is a pronoun, not a value". Design note: `docs/parameter_pipeline_design.md`.

`LLM_RESPONSE_GEN` (key `LITELLM_API_KEY_RESPONSE_GEN`) is the model of `fastworkflow/run_chatbot/composer.py`: the `run_chatbot` composed answer card and `ask_user` question form (`POST /compose_answer_card`, `POST /compose_question_form`). The shipped `fastworkflow.env` template sets it to `mistral/mistral-small-latest` (key in `fastworkflow.passwords.env`; a Bedrock model needs no `LITELLM_API_KEY_*` entry, it authenticates from the AWS credential chain), and the code has no fallback. Unset, so everything deterministic still works and the composed card and form are absent. The card request takes up to 60 results and an answer of up to 150,000 characters (the question form: 20,000); the model reads the whole answer up to 20,000 characters and above that an excerpt of about 12,000 (every `### N.` heading with the opening of its part), while every check reads the full text; results may carry the recorded request part (`sub_task`), used for the log diagnostics only (the union of it and the parts); the request may carry `parts`, the parts of a split turn as the core recorded them (the `fw_sub_tasks` artifact of the turn's last command output: number, request text, title, answered, offsets of each part's answer in the answer text), and what the model reads and the section caps come from the answered parts when they are sent and fit the answer, else from the `### N.` headings, with the same result; the model writes `lead` and `sections` (at least 5, one per part of a multi-part request up to a safety ceiling of 24, with 4 blocks each; the output budget grows with the section cap, 1800 tokens up to 5 sections and 250 more per section after); the page adds the 8-row table cap and a footer line for every primary result group no block shows. See the README section "Chat: drive the workflow from the browser".

LiteLLM Proxy: prefix model names with `litellm_proxy/` and set `LITELLM_PROXY_API_BASE`.

`INTENT_MODEL_LAYOUT` (workflow env file; `fastworkflow/train/intent_model_layout.py`, where `DEFAULT_LAYOUT` is the
one constant that says what an unset setting means: `embedding`) is read by `fastworkflow train` only. The framework's
own workflows (`fastworkflow/_workflows/*`) are always `embedding` (`FRAMEWORK_LAYOUT`, `is_framework_workflow`; since 2026-11-17, bd fix-o5wi.3): a per-context retrain, which any application's `fastworkflow train` can trigger, did not reproduce the previous review head (17 of 43 recorded replies changed their answer, "hold on" from decline to confirm), the embedding head is deterministic, trains in about 3 s and is 15 KB where the pair was 288 MB, and on the held-out review corpus it scored 31/36 against 28/36 for the per-context head of 13 November and of 17 November with no confident wrong answer. Their embedding settings are pinned (`embedding_settings(workflow_folderpath)`), never the application's: the pinned encoder `FRAMEWORK_ENCODER` is `BAAI/bge-small-en-v1.5` (CLS pooling, no prefix), NOT the default encoder below, because on a temp copy of the plan metadata extraction workflow the default encoder failed the review reply gate (30/36 held-out top answers with `look_up` at 2/6, under the test bar's 0.50 per command, and "hmm" answered `confirm` outright; with bge-small 31/36, `look_up` 3/6, every vague reply the wildcard class; bd fix-o5wi.9). The other rules below apply to them when they are retrained; the heads trained in place before 2026-11-17 are the earlier rule (schema 1, with a margin) and are read as they were until the next retrain. Both of them get the workflow-wide `wildcard` class from the built-in vague replies (the CME's own heads are not read by application workflows at run time: the application's heads carry the core commands, trained from the CME's utterance cache), and a context's mask is its own commands (the plan metadata extraction workflow has no core commands: `CommandDirectory._register_core_commands` skips it by its real path, so a scratch COPY of it trains with the CME's core labels in the mask; train a copy only with `fastworkflow.get_internal_workflow_path` pointed at it, as `tests/test_framework_embedding_layout.py` does). `per_context` and `workflow` stay trainable and readable by name for an application. Three layouts:
- `embedding` (default; `train/embedding_head.py`, `EmbeddingLayoutTrainer`, called from
  `model_pipeline_training.train`): the rows the `workflow` layout assembles are extended (`training_rows`) with, for
  each command, a row of its humanised name and a row of the first sentence of its `Signature` docstring, and every
  seed utterance (the command's `plain_utterances`) counts three times (`sample_weight`), as in the experiment's
  "description rows + seeds x3" variant (`docs/local_intent_model_levers.md`, Part 2). They are embedded with a frozen
  sentence encoder that is never trained (`INTENT_EMBEDDING_MODEL`, default `intfloat/e5-base-v2`, one shared download
  of about 418 MB, nothing stored per workflow, loaded with `transformers.AutoModel`; mean pooling and the prefix
  `query: ` before every text for the e5 family, CLS and no prefix for `BAAI/bge-*`, `INTENT_EMBEDDING_POOLING=cls|mean`
  to override pooling; L2-normalised, 128 tokens; one loaded encoder per (model id, pooling, prefix) per process,
  `SentenceEncoder`), and ONE `LogisticRegression(C=100, max_iter=3000)` is fitted over every label of the workflow.
  Only `coef_`, `intercept_` and the label order are stored; inference is numpy.
  **The local model never passes a request up on its own, and its threshold is a constant.** One command is the answer
  when its probability is at or above `INTENT_COMMIT_THRESHOLD` (default `DEFAULT_COMMIT_THRESHOLD` = 0.65: the
  experiment's 1%-wrong point on the examples, 0.50 on a large workflow, so 0.65 is the safe side; the trainer's cross-fit
  no longer sets it because asking it for 0.5% wrong gave 5 to 8% wrong on real requests;
  `intent_model_layout.calibrated_commit_threshold` is the one place a value calibrated on recorded real requests will be
  read, bd fix-o5wi.6, and is None today). Decision in a context (`embedding_head.decide`, probabilities NOT
  renormalised inside the mask): best label overall `parameter_value` is a bare value, the workflow `wildcard` (the
  built-in vague replies' class) is "not here"; else with `p_in` the largest probability inside the mask and `p_out` the
  largest outside it (reserved labels not counted), one command when `p_in >= threshold` and `p_in >= p_out`, else the
  head is NOT confident and hands the request on with the context's top three, and when a command outside the mask is
  ahead (`p_out > p_in`) it also reports the best ten such commands (`details["outside_labels"]`, `outside_scores`).
  There is no pass-up margin. `predict_with_details` returns the keys the `workflow` layout returns, with `model_tier`
  "embedding" and those two. The five-fold cross-fit (folds grouped by row text) is kept for the training report only
  (`cross_fit_report`): cross-fit accuracy and the most confused pairs of commands (rows of one that the head gives to
  the other, counts and shares, both directions; the top ten and how many exceed 10%), printed in the training report
  with one line on what to do (add seeds that separate the pair, or merge the commands); it sets nothing.
  `scripts/experiments/embedding_commit_rule.py pairs` computes the same pairs.
- `workflow` (`train/workflow_model.py`, `LayoutTrainer`): ONE fine-tuned large classifier over every command, read
  through the same per-context masks.
- `per_context`: a fine-tuned small and large classifier for every context.
The two one-model layouts write `___command_info/versions/<id>/workflow/` (`embedding`: `head.npz` and `scopes.json`
with the encoder id, pooling and prefix, each context's mask, the commit threshold, the row counts and the cross-fit
facts; `workflow`: `largemodel.pth/`, `label_encoder.pkl`, `scopes.json`); each context folder then holds only
`threshold.json` (kept so every "is it trained" check works; its value 2.0 is not a tuned number) and
`workflow_scope.json`, which points to the model and whose `layout` field says which kind it is. The runtime never reads
the setting: `CommandRouter` sees the pointer file and answers the context through the one model. **How the router
tells the rule of an embedding head:** `scopes.json` `schema_version` 2 (a `decision` block, no `margin`) never passes up
alone; a head without it (schema 1, with a `margin`, the encoder's `prefix` absent meaning none) is read as it always
was, with its own threshold and margin, until the workflow is retrained. The layout is in the version manifest (and, for
`embedding`, the cross-fit facts, times and bytes under `intent_embedding_head`; also in each context's held-out report),
and the layout, the encoder id, its pooling, its prefix, the commit threshold and the digest of `embedding_head.py` are
global inputs of the training signature, so changing any of them is a full retrain; in a one-model layout any change to
any command refits the one model (the plan is all or nothing; the embedding head takes seconds). `INTENT_WRONG_COMMIT_TARGET`
and `INTENT_FALSE_ESCALATION_TARGET` were removed: a value left in an env file is ignored with one warning
(`warn_if_removed_settings`). An invalid value of a setting that remains logs a warning and means the default. Tests
about the per-context or `workflow` trainers pin their layout by name.

Measured through the product trainer and router (`scripts/experiments/embedding_head.py product`, 2026-11-17) on temp
copies of `library` and `retail_workflow`, default encoder and rules, threshold 0.65, a request asked of the local
model alone (every command of every other scope counted as reachable for the candidates):
development benchmarks 1 to 3 pooled (735 routing and 120 escalation requests): the one label it answers is right on
59.6% of requests, wrong on 1.8% (of which 0.8 points are requests whose best label is the vague-reply class, which
keeps its meaning, and 0.4 a bare value), handed on with the right command among the candidates 35.6%, handed on without
it 3.0%; for escalation requests the expected ancestor command is among the candidates on 117 of 120 and the head
answered one label on none. Confirmation, fourth benchmark (264 routing and 46 escalation requests, scored once, not
tuned on; the previously shipped head, bge-small with a calibrated threshold and margin, had 60.2% right outright and
10.2% wrong confident on it): 45.8% right outright, 3.0% wrong confident, handed on with the right command 45.8% and
without it 5.3%, ancestor among the candidates 45 of 46, none answered with one label. So the head answers less often
outright (the conservative threshold) and is wrong confidently about a third as often; the rest goes to the next link.
Operational, same runs, CPU: the encoder loads in 0.6 s with the weights in the OS cache (+368 MB resident, 869 MB for
the process); fitting a workflow takes 9 to 10 s for `library` (777 rows, 8.0 s of it embedding them) and about 9 s for
`retail_workflow` (513 rows), 14 s and 11 s through the whole trainer; the artifact is 97,524 and 68,689 bytes; a
request takes a median 25 to 32 ms and 39 to 51 ms at the 95th percentile with 10 torch threads, and 52 to 64 ms and
78 to 103 ms with one (the shipped bge-small head: 10 ms and 20 ms). The layout has not been measured on a large
workflow. The encoder's weights are not in the artifact: a machine that runs an `embedding`-layout workflow loads them
from its Hugging Face cache and downloads them on first use. A process keeps a loaded router per context folder, so
after a retrain a running server must be restarted, as for any retrain.

Intent detection is ONE chain, with no setting to choose it (`fastworkflow/intent/intent_detector.py`; `engine_choice`
names the links a context has, `engine_for` builds them):
1. A context with a trained local model: it decides when it is confident (one label: a command or a bare value), with
   no model call (`ClassifierEngine`, `head_is_confident`). It never passes a request up on its own: when its best
   command lies outside the context it is not confident, and hands the request on with that command among the
   candidates. The detector contract carries this with one parameter: `ClassifierEngine(router, reachable=...)` (given
   by `engine_for`, with or without the remote link) adds up to two of the head's `outside_labels` that are in
   `reachable_labels(workflow_folderpath, scope_path)` (the commands of the scopes on the walk's lookup path, the global
   scope included when it is asked first) to the context's own candidates, best first, and names them in
   `details["handed_on"]`; `ScopeAnswer`, `ScopeWalk` and `Detection` are unchanged (`Detection.candidates` carries them).
   A command chosen at the which-command question is run where it lives (`intent_loop.choose`: the global scope when it is
   asked first and has the command, else the nearest scope of the lookup path that has it), on that scope's record, as a
   climb would (span attribute `chosen_command_elsewhere`); so is a command reached by name with no lookup
   (`CommandExecutor.perform_action`: an MCP tool call, a direct action, run-code, the approval re-run), filed under
   that record. The climb is one generator, `context_navigation.live_chain`.
2. Otherwise (a list of candidates, the vague-reply "not here", or no local model for the context) and `JEV_API_KEY`
   set: the remote decision model (`MatcherEngine`, Jev, one model call per detection over the scope path) is asked and
   decides when it is confident: a command at or above `INTENT_REMOTE_ACCEPT_THRESHOLD` (default 0.9,
   `intent_detector.DEFAULT_REMOTE_ACCEPT_THRESHOLD`) is the answer; "none", or a command of another scope on the path,
   at or above `INTENT_REMOTE_DECLINE_THRESHOLD` (default 0.9) declines the scope; below the accept bar the remote answer
   is not an answer and the candidates (those above the 0.2 `MATCHER_UNSURE_THRESHOLD`) go to the question. Both are
   read in one place (`intent_detector.remote_accept_threshold`, `remote_decline_threshold`; an invalid value warns once
   and means the default) and were 0.5 until 2026-11-18: on a large private workflow 0.5 gave 13.4% confident wrong
   answers against 6.5% at 0.9, for 9.0% against 38.3% asked (`docs/remote_link_worth.md`). There is no separate planning-command
   match: the split's over-cap reduction labels its parts through `resolution.part_intent` (`workflow_agent.plan_label`), the
   turn's memoised decision, and the planner's plan template is read from it too (bd fix-zvme);
   `workflow_agent.PLANNING_COMMAND_MATCH_THRESHOLD` (0.5) is only the bar of the run listing's gathering
   (`run_listing.planning_candidates`).
   `ChainEngine` is the two links (`engine_for` builds them for a scope). When the matcher's call fails, the local model's answer stands (or the question, for a local
   plan pick, below) and the outage is visible: `remote_outage.report` (every caller of the remote model: the command
   match, the reply-intent classifier, and so `_second_opinion`) writes one WARNING for the
   first failure and then at most one per 5 minutes with the count since the last notice, naming the error class and the
   HTTP status code, never the message (which can carry a key), and the `fw.intent` span carries
   `remote_failed` = `{error_class, status_code}` (since span contract v53) beside `matcher.error`. There is no per-turn
   warnings artifact to put it in, so the span and the log are the marker.
   **A confident local answer that names a planning command is never acted on unreviewed, on either path.** Resolution
   does it in `intent_loop._second_opinion`, ONE function for both paths (assistant mode and `/` messages included): the
   remote model confirms the plan or the candidates are asked (the plan first), and with no remote model, or a failed call,
   they are asked. Exact and fuzzy NAME matches get no second opinion (a fuzzy match of a command that changes data is
   confirmed all the same: layer (3) of "One intent loop").
3. Otherwise the candidates (the context's own and the reachable outside ones) go to the user (the ambiguity question) or
   to the agent (the candidate list), and a walk every scope declined ends in the "did not understand" question.

**One intent loop (bd fix-o1lg S7).** The plan path (`plan_metadata_extraction/resolution.py`) and the deterministic path (the CME:
`CommandNamePrediction` and the `wildcard` router, assistant mode and `/` messages) decide a request with ONE function and
read the reply to "which of these is meant?" with ONE reader (`fastworkflow/intent/intent_loop.py`; it imports no model stack at
import time). `intent_loop.resolve(workflow, text, core=False)` is the whole-walk decision, in layers, each living once: (1)
an exact command name on the lookup path (`context_navigation.exact_name_scopes`, nearest scope first, the global scope
last; the framework's core commands first when `core`); (2) the foreign-name guard: a known command name that no scope of the
path owns asks no model and returns the routing hint and the owning contexts (`Intent.hint`, `owners`); (3) a fuzzy name (the
WHOLE request against each scope's WHOLE names, leaf or qualified, case, spaces and underscores ignored,
`intent_clarification.misspelt`, bound `intent_clarification.FUZZY_MAX_DISTANCE` = 0.3: only a misspelt name matches,
never a request that begins like one, so "a" is not `add_note`; two names that match equally well are not a match: the
chain decides; a match is the closest command, NOT the user naming it, so it runs directly as a confident local answer
does, and a command that changes data found this way is confirmed: reviewed on the plan path, "Did you mean the command
'X'?" on "/", bd fix-ftdl.9); (4) the local model over the WHOLE walk first (a confident answer anywhere
decides with no model call); (5) the chain with the remote model (`intent_detector.detect` with `engine_for`, the global
scope asked last when the walk did not ask it and everything declined); (6) a confident local answer that names a planning
command needs remote confirmation or becomes an ambiguity (`_second_opinion`); (7) a bare value is reported as such
(`Intent.bare_value`). A command that only enters a context is not offered to the REMOTE model on either path. The result is
ONE structure, `intent_loop.Intent` (outcome match / ambiguous / none, command, kind execution / planning / core, ranked
candidates, `decided_by`, the `context` and `record` the command lives in, evidence for learning, hint, bare-value flag);
`resolution.PartIntent` is that class. The pending question is ONE JSON-safe `dict` (`open_question`, `ask`, `answer`,
`payload`): both paths keep it in the turn's waiting record, kind `intent` (the plan path's `owner` is `plan`; saved with the
session in the blob's `waiting` section like the parameter loop; `Workflow.end_command_processing` clears it). The client shape
is the same on both paths: `question_kind` `plan_choice` with `artifacts.plan_choice` = `{request, sub_task, choices: [{name,
description, closest}], none_reply, decline, ambiguous, misunderstood}` (the last two say what the choices are: the
candidates of a request more than one command fits, or every command because nothing fitted); the CME's completed
turn carries it on its command output in a human ("/") session only (every which-command question is a failed
output, `success` False, as on the plan path: a question that is out is no success; the nothing-matched question and the
listing after `you misunderstood` also carry the artifact `command_handled`: `command_executor._wildcard_command_output`), the older `intent_choices` artifact is gone and the
chat page draws both with one renderer. The question text is one function (`intent_clarification.plan_choice_question`).
The reader (`intent_loop.read`) takes a name, its leaf, the name with spaces, a listed number, `none`, cancel/abort, a first
word that names a candidate (the pick only: the rest of the reply is not the command's parameter text on either path, the
request that was asked about is), a misspelling of a whole name (`misspelt`; a tie is unclear) and the plain utterances of `abort` and `you_misunderstood` as aliases, matched as the WHOLE reply only (case and a trailing `.` or `!` ignored, no misspelling)
(abort, none of these: `intent_loop.core_aliases`, on both paths, so "you misunderstood" drops a plan part as `none` does), and,
on the CME only, those of `what_can_i_do` (list the commands: `core_aliases(listing=True)`);
a reply that cannot be read is asked once more (`no_agent_mode.MAX_NO_PROGRESS`). A reply that names a real command owned
off the path is answered with the routing hint. `intent_loop.record` is the one learning hook (the CME's `_settle` and
`_record_choice`, the plan path's `resolution.record_part` call it). The `fw.intent` span (v14 of span contract 63) is one decision, not one
prediction per scope, and BOTH paths write it with one function (`intent_detection.traced_decision`: the CME's stage, and the plan
path's `resolution.part_intent` on a memo miss, with the part's number as the attribute `part`; a pinned plan is not detected and
writes none); the `fw.parameters` span (v16) likewise (`parameter_extraction.traced_extraction`: the CME's stage, and the
plan path's draft of a part's own command, with `part`, including `argument_check_not_in_request`). A part's two spans sit under
`fw.turn > fw.resolve > fw.resolve.part` (`tracing.part_span`; the "/" path's one part through `tracing.direct_part_id`); a command's own decision or extraction that decided nothing (its exact name, a typed call read in code, no model call, nothing missing or refused) is not written on any path, and the command's `fw.command` span carries `resolved_by`, `parameters_read_by` and `parameters_valid` instead (`tracing.is_command_decision`). A store written before span contract 63 is not read by the framework's tools (no old-name map in the framework). The `NLUPipelineStage` enum and the context key `NLU_Pipeline_Stage` are gone (S10): what the CME waits for is read from its
two pending records only (`intent_loop.waiting_for(context)`: `WAITING_INTENT` when the record is of kind `intent`,
`WAITING_PARAMETERS` when it is of kind `parameters`, None when the next message is a request). Its readers are
`workflow_agent` (aborts a pending question after an agent's turn), `command_executor._pending_before_cme_reply` (the
`aborted` artifact now carries `waiting_for` instead of `stage`), the CME router and the "/" routing flag; the session blob
no longer has `nlu_stage` or `plan_nlu_stage`. The differences that remain are
the only ones: (a) `core` commands are answerable on the CME only; (b) with ONE candidate the plan path binds it and the
turn's review confirms, and the CME asks "Did you mean the command 'X'?"; (c) after two unreadable replies the plan path
drops the part (it comes back as a not-run entry) and the CME ends the question with the `what_can_i_do` listing. The
`you_misunderstood` listing text is still the command's own (its file is an input of the CME's training signature).
No local model and no key is a startup problem; with the key a workflow with untrained contexts starts on the matcher.
`FW_REQUIRE_HEADS=1` (process environment or env file) makes a missing local model a failure even with the key (tests
rely on it). Startup says what the chain is in one line (`intent_detector.startup_report`, `chain_line`; called by
`fastworkflow run_cli`, the MCP server, the FastAPI server and the chat page's workflow listing). `INTENT_DETECTION_ENGINE`
(`auto|classifier|matcher|hybrid`) was removed: a value left in an env file is ignored with one warning. The older Jev
tie-break among the classifier's candidates (`FW_JEV_INTENT_TIEBREAK`, `choose_ambiguous_command`) was removed with
it: the chain's second link replaces it. The `fw.intent` span records `decided_by` (who decided: `local`,
`remote`, or `question` when no link was confident; `name`, `fuzzy` for a name layer) and `engine_chain` (the links asked, in order); `engine_mode` and
`jev_tiebreak` are no longer written. Gathering (`run_listing.head_answers`) asks trained local models only. Tests
script the matcher's model call with the `matcher_model` fixture (`tests/test_intent_matcher_engine.py`); a test that
asserts what happens when the local model is NOT confident and no remote model answers must use `no_planner_key`,
or the matcher is called live. `scripts/experiments/score_matcher_engine.py` scores the matcher and the chain on an
intent benchmark.

**A reply answers its question and nothing else (bd fix-rpio).** A reply to a pending question (which command, a parameter, a plan's decision step, the turn's review) is read by that question's own reader for its answer, and anything else in the reply is ignored; the which-command, parameter and decision-step questions say so ("Only your answer is read; anything else in the reply is ignored."). There is no code that turns part of a reply into a new request (the earlier `reply_request.split_reply`, `reply_parts`, `fw_unhandled_request` and the `from_reply` mark were removed because they invented parts from ordinary replies, such as a value that begins with a verb); how a value is read from a reply belongs to the parameter extraction (bd fix-qoyz).


Learning from use (bd fix-o5wi.6; `fastworkflow/intent/intent_learning.py`, `fastworkflow/train/head_refit.py`):
- **Recording** (`INTENT_LEARNING_RECORD`, default on; off writes nothing): one row per settled part in
  `<workflow>/___convo_info/intent_learning.sqlite3` (SQLite, WAL, a connection per write): time, the request as detection
  saw it, context, the resolved qualified command, `how` (`local`, `remote`, `name`, `user_choice`, `agent_choice`,
  `review_accept`, `review_switch`), `confirmed` (a pick at a question, a pinned plan, a review that showed the command for
  the part and was accepted, a plan switched in review), the head's facts at decision time (`head_version`, its top
  in-context label, `p_in`, `p_out`, confident, threshold and why) and the remote model's top label and score, and
  `rejected_label` (the answer a switch or a you_misunderstood replaced). Written through ONE hook,
  `intent_loop.record` (-> `intent_learning.record`), called by the CME's `wildcard._settle` and `_record_choice` (where a
  message moves on to parameter extraction) and by the resolution path (`WorkflowExecutionContext._subtask_gate` for the
  fast path, `_record_planned_parts` where a plan is run). It never fails a turn (it logs once). A typed call with parameters is not
  a request and is not recorded (plan steps and agent tool calls are that).
- **What teaches** (`teaching_rows`): confirmed rows only, never `name`, a bare value or a built-in vague reply; a
  (normalised text, context) counts at `INTENT_LEARNING_MIN_CONFIRMATIONS` (default 1) confirmations of one command; a text
  with conflicting confirmed commands is excluded unless one leads by 2. `fastworkflow learned list|remove|stats <workflow>`
  reviews and removes rows.
- **Row store**: the train writes `rows.npz` beside `head.npz` (the vectors it computed, float16, the labels, weights, row
  kinds, texts; 1,028,980 bytes for the library example, 658,080 for retail). A head without it (every head trained before) cannot
  be refitted until its next train. The file is part of `embedding_head.py`, so the training signature digest of every
  embedding-layout workflow, the built-in ones included, changed with it.
- **Refit** (`fastworkflow learn <workflow> [env]`, no model call): stored base vectors plus the learned rows (embedded
  once, cached in the store, weighted like a seed), the same `LogisticRegression`, labels and masks. Published only when
  both gates pass. (a) The base rows, cut into the training report's folds with the learned rows in every training side,
  lose at most `GATE_BASE_TOLERANCE` = 1.5 points of cross-fit accuracy (synthetic rows weigh less than confirmed real
  requests). (b) The learned rows, 5 folds grouped by normalised text, held out of a fit on the base rows plus the
  others, are right outright no less often than held out of a fit on the base rows alone (the replaced head is not the
  baseline: after an earlier refit it has seen some of the rows) and wrong confident no more often, except that wrong
  confident may be higher by `GATE_WRONG_TOLERANCE_ROWS` = 1 row when right outright is higher by at least
  `GATE_WRONG_TOLERANCE_NEEDS_RIGHT_ROWS` = 3 rows. The gate report keeps its fields and adds `rule` to each gate ("no
  loss" or "base tolerance"; "strict" or "wrong confident tolerance"; `fastworkflow learn` names a tolerance when it was
  used) and `baseline_cached`: the base rows' cross-fit WITHOUT the learned rows depends only on the stored rows, so its
  totals are kept in the learning store (meta key `base_crossfit`, keyed by a digest of the base rows) and the next refit
  of the same lineage computes only the refit side. The earlier gate (0.5 points, no tolerance) rejected 21 of 26 refit
  attempts on a large workflow, nearly all of which raised right outright on the held-out learned rows. Neither
  `head_refit.py` nor `selective_training.py` is a training signature input (the signature digests `embedding_head.py`,
  `workflow_model.py`, `generate_synthetic.py` and `model_pipeline_training.py`), so gate changes retrain nothing. At
  least `MIN_TEACHING_ROWS` (5) distinct texts. Published as a new version (`refit_of`, `lineage`,
  `learned_rows`, `refit_gate` in the manifest) through the usual atomic publication; retention keeps the current, the
  previous and the full train the lineage descends from (`retain_current_and_previous(also_keep=)`). The same rows again
  are "nothing new". A tolerance lets a coherent set of wrong confirmations through when it costs the base rows under 1.5
  points (a test with 32 to 104 seed phrasings plus one word, confirmed as the wrong command, now publishes); seed
  phrasings confirmed as another command outright still fail the base rows condition by over 20 points.
- **Calibration** (`calibrate`, read by `intent_model_layout.calibrated_commit_threshold`): only LOWERS the threshold,
  because truth is known only for requests that were asked or reviewed. For t in 0.60, 0.55, 0.50: confirmed rows of the
  head lineage with recorded `p_in` in [t, trained threshold) and `p_out <= p_in` would have been answered; one is wrong
  when the head's top label differs from the confirmed command. Lowered to the smallest t with at least 200 confirmed rows
  in all, 100 in the band and at most 2% of the band wrong (each t judged on its own whole band, which contains the
  bands of the higher candidates). Stored in the learning store; read at
  router load (`threshold_in_force`: `INTENT_COMMIT_THRESHOLD` set in the env always wins, else the calibrated value, else
  the trained one; `details["threshold_source"]`, `ambiguous_threshold` on the span, and a startup line when it is not the
  plain default). A full train bakes a calibrated value into the new head (a changed signature input).
- **Reload without a restart** (`intent_detector.refresh_stale_head`, at most every 3 s per workflow): when a loaded
  embedding head is no longer the `current` version, the next detection builds the new one; a turn in flight keeps its
  router. The `workflow` and `per_context` layouts are unchanged (a retrain still needs a restart).
- **Background refit** (`INTENT_LEARNING_REFIT=background|off`, default `background` since the live path was tested in
  `tests/test_learning_live_refit.py`; set `INTENT_LEARNING_REFIT=off` to turn it off; `INTENT_LEARNING_REFIT_EVERY`, default 25):
  after that many new teaching rows one daemon thread per workflow per process runs the same refit; a file lock
  (`___convo_info/intent_learning.lock`) keeps two apart; never on a turn's path.

Optional settings for planning commands (`_commands/<name>.py` files; see
[author-planning-commands](fastworkflow/skills_for_coding_fastworkflows/author-planning-commands/SKILL.md)):

- `JEV_API_KEY` (optional). Without it, the remote model never confirms a planning command and the typed planning
  call runs with no classifier check (`check_lm=None`: the `classifier_confirms` and `classifier_decides` fields keep the
  extractor's value or their default); intent detection still matches planning commands (by name and by the local model),
  and the text planner plans and names the planning commands in its listing. With it, Jev
  (`workflow_agent._jev_lm`) matches a sub-task to a planning command (`_planning_command_matcher`),
  and the typed planning call fills that command's arguments and runs its classifier checks (`classifier_confirms`). Without the
  key, a workflow that has planning commands logs one INFO line (`workflow_agent._warn_missing_jev_key`).
- `FW_TURN_MAX_STEPS` (optional, default 400). The most model steps the nested planning-command calls of
  one turn may take together; it resets each turn. A value that is not a positive integer logs a
  warning and the default applies (`plan_frame.turn_max_steps`).
- `FW_CODE_MAX_PAGES` (optional, default 20). The most pages code reads of one full-list step of a planning
  command (`planning_phases.CODE_MAX_PAGES` is the default); past it the step is open with a partial read. A value that
  is not a positive integer logs a warning and the default applies (`planning_phases.code_max_pages`, also read by
  `command preview`).
- `FW_SUBTASK_WORKERS` (optional, default 4). After a request is split, the per-sub-task model calls that do
  not read each other (the referent, the argument draft) wait on the model together,
  at most this many at once, and the results keep sub-task order. 1 runs them one after another; a single
  sub-task never uses a pool. A value that is not a positive integer logs a warning and the default applies
  (`subtask_pool.worker_limit`). Set 1 for a model that cannot take parallel calls or a test that scripts one model
  in call order.
- `FW_SPLIT_GATE_THRESHOLD` (optional, default 0.85; needs `JEV_API_KEY`). Before a request is split, the remote
  decision model is asked whether it asks for one job or several, given the request and the splitter's job list
  (`workflow_agent._one_job_gate`); a "one" at or above this probability runs the request as one part and the splitter
  is not called. `off` turns the gate off (no call). A value that is not a number above 0 and at most 1 warns once and
  the default applies (`workflow_agent.split_gate_threshold`). A failed call runs the splitter. The verdict is on
  `fw.split.gate` span, a child of the `fw.split` stage span. Tests run with it `off` (`tests/conftest.py` `split_gate_off`). The default was
  chosen on one workflow's benchmark it was then tested on; a workflow that sees a multi-part request kept whole
  should raise it.
- `LLM_TIMEOUT_SECONDS` (optional, default 120), `LLM_TIMEOUT_SECONDS_<ROLE>` and `LLM_NUM_RETRIES_<ROLE>`
  (optional; `<ROLE>` is the name after `LLM_`, as in `LITELLM_API_KEY_<ROLE>`). The model timeout is per attempt,
  and DSPy makes up to retries + 1 attempts (a failed one waits 1 s, then 2 s). Parameter extraction has
  built-in limits of 20 s and 2 retries (longest wait 63 s); every other role takes `LLM_TIMEOUT_SECONDS` and
  DSPy's 3 retries unless its own setting is given (`dspy_utils.role_limits`). A value that is not a positive
  number (a non-negative integer for retries) logs a warning and the default applies. An extraction that times
  out leaves its parameters not given; the `fw.llm.call` span records `role` (always: the caller names its own
  model setting where it calls, `dspy_utils.role_context`, else the tag `get_lm` put on the LM), `timeout_s`,
  `num_retries`, `outcome` and `timed_out`, not the attempts made (DSPy does not expose them).

Planning command previews: `fastworkflow command preview <workflow_folder>` writes one markdown preview
per planning command to `<workflow_folder>/../docs/planning_command_previews/`, and the workflow's object model (contexts, `inherits`, `parents`, the commands whose `@descends_to` enters each, as a containment tree and a table; `object_model_view.render_object_model`) to `_object_models/<workflow>.md` there. With `--check` it writes
nothing and exits 1 when a preview is stale, missing or extra, or when a command has a load problem (a `from_command_output` edge that does not resolve is one, a bare command name with no `.<field>` included: `command_preview.load_problems` lists it).
`command preview` replaced `fastworkflow skill preview`.

Running one call with no model: `fastworkflow command run-code <workflow_folder> <command> [--input '<json>']
[--start-with '<step>' ...]` runs a planning command phase by phase (its steps as the runtime writes them, each
step code can run, the facts, the answer), stopping at the first open step with exit 0, or an execution command
and prints its `CommandOutput`. It runs the workflow's real commands, so use a development instance. Exit 1 when
the command does not load, a step fails or a hook raises. `fastworkflow.planning.run_code` is the same run for
integration tests (`tests/test_run_code.py`). Command files that do not load are reported by `fastworkflow run_cli`
(before its first prompt), `fastworkflow train` (before training) and the FastAPI server (at startup), all through
`command_preview.load_problems`. One rule for an author's mistake (bd fix-ftdl.2): a mistake inside one command file
(it does not parse or import, defines both generator classes, or any check of its classes, fields, phases, steps or
declarations fails, a call cycle included) fails THAT command only (`CommandDirectory.load_errors` for a file that
cannot be read, `FieldSourceResolution.errors_for` for every other check; a class fetch raises `CommandLoadError`; an
execution command that does not load is in no scope's command set, `RoutingDefinition.not_loading`, and a trained head's
answer naming it is "not here") and is one `<command>: <file>: <message>` line; a mistake in something the workflow
shares (a `_<Context>.py`, its `inherits` and `parents` declarations included, the removed `_commands/context_inheritance_model.json`, `startup_action.json`, `_workflow.py`, and any other
underscore `.py` file or folder of `_commands/` or a context folder: a workflow's helper modules live in ONE package,
`_commands/_shared/`, which nothing reads as commands, contexts, settings or training inputs, imported as
`from ._shared.x import Y` from a root command and `from .._shared.x import Y` from a context folder;
`command_directory.helper_file_problems`, with a did-you-mean for a misspelt `_workflow.py` or `_<Context>.py`; bd
fix-qjne.33) is one line naming the file that stops the workflow (`LoadProblem.workflow_wide`: `fastworkflow run_cli` exits 1, the server refuses to start).
Nothing is only logged (the load-check sweep, bd fix-ftdl.8.22, moved the last silent cases under it: a generator is a
class defined in the command file, a file of `_commands/` with neither `Signature` nor generator is refused, a public member of
`class Context` must be a hook, `command_context_model.CONTEXT_HOOKS`, a public member of a command's `Signature`, own or
inherited, must be one of `planning_commands.SIGNATURE_MEMBERS` (plus `template` on a planning command, `SlotInput` in a
framework workflow); every context is a folder of `_commands/` (`command_context_model.context_folders`); each
`class Context` declares, as list literals read without importing (`command_context_model.read_declarations`),
`inherits = [...]` (contexts whose commands it also has, `"*"` the global commands) and `parents = [...]` (every
context a record of it can sit in, itself included when records nest; none means top level), exactly when it has
`get_parent` and never on an `occupiable = False` context (bd fix-qjne.36; a name that is no context, with a
did-you-mean, a name given twice, self-inheritance, an inheritance cycle, and a folder with no command file that
inherits none, stop the workflow; a context command's `@descends_to` target must list the command's context, or a
context it inherits or that inherits it, among its parents, that command's problem; `fastworkflow migrate-contexts
<workflow>` (`migrate_context_declarations.py`) writes `inherits` from an old file, deletes it and lists the contexts
with a `get_parent`, whose `parents` the author writes by hand), while a folder with no command file that inherits one loads with its `_<Context>.py`; the list is in
CHANGELOG). At run time
`command_executor.call_response_generator` puts `workflow.record` back when a command raises or refuses (not on an
`ask_user` suspension), and `_check_move` fails a command that moved the session (assigned `workflow.record`; setting
`workflow.root` is no move) onto a record whose class is no context, onto a context its `@descends_to` (read from the
generator class that ran) does not name or inherit from (its `inherits`), or, with a root, onto a
non-root record whose context has no `get_parent`; the record is put back. The core commands (go_up, reset_context)
are the framework's navigation and are not checked against a declaration. `Workflow.get_parent`, one rule keyed on
the root: with no root, no `_<Context>.py`, no `get_parent` or `return None` all mean the global commands; with a root,
a non-root record whose context has no file or no `get_parent` raises `MissingGetParentError` (naming the context, the
file and the method) on the climb, and `return None` is the explicit "go to the global commands"; a record
`get_parent` returns whose context is neither a declared parent nor inherits one raises it too, naming the context,
its parents and the record's context.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:970c3bf2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

## bd: closes are invisible to git until you export with `-o`

Outside the managed block above, because it is repo-specific and must survive that
block being regenerated.

`bd` writes to a local Dolt database; `.beads/issues.jsonl` is a passive export and
is the file git tracks. `bd close` updates the database, **not** the JSONL. Since bd
1.1.2, plain `bd export` writes to stdout — the file needs `-o`:

```bash
bd export -o .beads/issues.jsonl
```

Omitting it does not error. `bd close` reports success, `git status` shows nothing,
and the JSONL still says `open` — which looks exactly like bd silently losing writes,
and was filed twice as that bug (`fix-46c`, `fix-fyw`) before the cause was found.

**A second, separate failure mode, which this habit does not catch.** Eight issues
were found finished but open in one session, including an epic sized at a month of
work. Their `closed_at` timestamps show nobody had ever run `bd close` on them — so
the export bug is not what hid them. The work simply shipped under neighbouring
issues and nobody went back for the parent. The export bug explains why closes stay
invisible *once someone tries*; it does not explain issues nobody closed.

So the habit is two things, not one: export after writing, **and** close the parent
when the last piece of its work lands. Verifying before scheduling is the only thing
that catches the second — one of those eight was the largest item on the board.

Verify by reading the file, not by trusting the command's output. Full detail in
`.cursor/skills/beads-workflow/SKILL.md`.

## Do not use grep for searching
- **Go with mgrep if you are exploring an unfamiliar architecture or monorepo**. Its retrieval quality is much stronger at pulling together conceptual overviews, handling fuzzy documentation queries, and synthesizing disparate text assets (like a mix of YAML configs, Markdown files, and codebase PDFs).
- **Go with ColGrep if you need absolute precision inside raw code lines**. Its capability to maintain token name exactness while performing semantic abstractions makes it the superior choice for tracking down bugs, handling heavy refactors, and pinpointing exact functional lines.

## Repowise: semantic search via colgrep

Repowise's own embedder is `mock` (semantic search off), so its `search_codebase` and the semantic half of `get_answer` only match keywords. For meaning-based code search, use colgrep (`colgrep`) instead of Repowise's search. Use Repowise's MCP tools (`get_overview`, `get_context`, `get_risk`, `get_why`) for architecture, ownership, and change-risk questions.
