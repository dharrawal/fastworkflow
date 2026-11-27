---
name: debug-workflow-conversations
description: >-
  Diagnose fastWorkflow failures from run_chatbot traces and human comments, locate the
  correct evidence store, and trace wrong routing, extraction, planning, execution or answers
  to a concrete workflow fix. Use when a conversation behaves incorrectly or review feedback
  needs a trace-supported diagnosis. Read recorded evidence without mutating it.
---

# Debugging workflows from conversation logs

Use recorded turns and their span trees to identify the first wrong decision and its
consequences. Capture depends on redaction, limits and writer health;
not every run contains every input or response.

## 1. Locate the conversation and its evidence store

In an existing `run_chatbot` session, navigate **benchmark → experiment → conversation**, or
**ad-hoc conversations → UTC date → conversation**. The left tree stops at conversations.
Select turns and drill into phases, steps and calls on the right; the right breadcrumb shows
the complete path. Read the **Human feedback** history at the relevant component as well as
its raw evidence. Use [optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md)
when the task is to improve and compare the workflow, rather than diagnose one failure.

For ordinary runs, resolve the database through the framework:

```python
from fastworkflow import state_paths
db_path = state_paths.observability_db("<workflow_folder>")
```

Registered experiments are recorded in the same database.
`fastworkflow.benchmark.setup.load_experiment(workflow_folderpath, experiment_id)["store"]` is null
until a run binds the registration, then `{"store_id": ...}` (the database's `store_identity()`).
Retain `store_id` with every turn/span reference. Conversation IDs are local to channels/stores.

A missing default database does not prove the workflow never ran: check the selected state root
and experiment registration first. If reproduction is needed, prepare it under the project's
execution permissions. The supported picker command is `fastworkflow run_chatbot`;
selecting a workflow can start a server, so it is not merely an offline file viewer.

## 2. Read it: read-only, always

```python
from fastworkflow.observability.store import ReadOnlyObservabilityStore
store = ReadOnlyObservabilityStore(db_path)
```

**Never instantiate `ObservabilityStore` to inspect a database.** That class is
the writer: constructing it can create the file and write-probe it. There is no migration: both
classes refuse a database written at another schema version (older or newer); read such evidence
with the framework version that wrote it. `ReadOnlyObservabilityStore` opens `mode=ro` connections
and cannot mutate anything. Raw SQL is equally fine (the schema is documented
in [reference.md](reference.md)):

```bash
sqlite3 "file:$DB?mode=ro" "SELECT turn_key, status, success, user_message FROM turns ORDER BY turn_key DESC LIMIT 20"
```

## 3. Find the failing turn

```python
store.list_turns(status="failed")                 # turn-level failures
store.list_turns(success=False)                   # any command in the turn failed
store.list_turns(command_name="cancel_pending_order")  # turns that executed a command
store.list_turns(context="TodoList")              # substring match on entry context
turn  = store.get_turn(turn_key)                  # full row incl. record_json
spans = store.get_spans(turn_key)                 # the trace, ordered by start_ns
comments = store.list_human_feedback(turn_key)    # all human annotation anchors for this turn
```

Two orthogonal outcome fields, both worth reading:

- `status` is the turn lifecycle: `completed` / `failed` (agent ran out of
  iterations: `failure_reason: max_iters_exhausted`) / `awaiting_user` (still
  suspended) / `cancelled` / `abandoned`.
- `success` means every command in the turn succeeded. **`completed` with
  `success = 0` is the case to hunt**: a command failed and the agent wrote a
  confident answer over it.

## 4. Walk the trace to the failing stage

A turn's spans nest like this (`parent_span_id` links them; names and full
attribute catalogs in [reference.md](reference.md)):

```
fw.turn                          the whole logical turn (path plan|direct, mode agent|no_agent,
│                                context_mutations, subtask_bindings)
├── fw.split                     how the request was split (raw, merged, rules)
│   └── fw.split.gate | .splitter | .merge | .referent   each with its fw.llm.call
├── fw.resolve                   which parts were asked which command (plan_ambiguity)
│   └── fw.resolve.part          one part (part, outcome bound|ad_hoc|dropped|asked)
│       ├── fw.intent            the one intent decision (decided_by, engine_chain)
│       ├── fw.parameters        extraction + validation, structured (> fw.llm.call)
│       └── fw.question          which command / its values (kind)
├── fw.review                    the review (outcome, rounds, items)
│   └── fw.question | fw.review.reply | fw.review.edit | fw.review.revision | fw.approval
├── fw.run
│   └── fw.run.part              one part (means, outcome ran|not_run|refused, reason)
│       ├── fw.plan              the text plan of an ad hoc part (trigger) > fw.llm.call
│       ├── fw.phase             one phase run of a planning command (steps by code,
│       │   └── fw.step          code_stop, facts) > fw.step (run_by code|agent)
│       ├── fw.step              one step: thought, tool choice, observation, the
│       │   ├── fw.llm.call      dispatched command's facts; its agent call (role ...)
│       │   └── fw.command       resolution + execution (resolved_by, parameters_read_by)
│       └── fw.question          an agent's question, a decision step (kind)
└── fw.answer                    who wrote the answer > fw.llm.call (conversation summary)

Deterministic "/"-mode turns have fw.resolve > fw.resolve.part and fw.run > fw.run.part
> fw.command, no split and no review. A command's own fw.intent / fw.parameters are
written only when they decided something. A store written before span contract 63
holds other names and is not read by the framework's tools.
```

## 5. The triage tree

Inspect the recorded causal order. The checks below help locate the first wrong decision;
a bad plan can precede routing, and a provider or backend failure may be environmental.
Treat a human comment as a lead to verify, not proof of the cause.


**A. Did routing pick the right command?** Every request is decided by ONE intent loop
(`fastworkflow/intent/intent_loop.py`, `resolve`): exact name, foreign-name guard, fuzzy name, the
local model over the whole context walk, then the remote model (`JEV_API_KEY`), then a
which-command question. The deterministic path records that decision as ONE `fw.intent`
span (`stage: INTENT_DETECTION`), and each reply to the question as another
(`matcher_layer: question_reply`, `decided_by: reply`). Compare `fw.command`'s `raw_command`
(what was asked) with its `command_name` (what ran), then read the span:

| Observation | Diagnosis | Fix with |
|---|---|---|
| `resolved: false`, no `candidates` | The request routes nowhere: vocabulary gap, or the command is not reachable from this context | Seed utterances (`plain_utterances`, ~8 varied phrasings) · `design-context-models` |
| `matcher_layer: known_name_foreign_context` | A command name that no scope on the path owns; the hint names its owners | `design-context-models` (navigation to that context) |
| `decided_by: question` with `candidates` | No link was confident: `classifier.confidence` below `classifier.ambiguous_threshold` and, when asked, `matcher.confidence` below `matcher.threshold`; near-misses mean starved or colliding seeds | `detect-duplicate-capabilities` · seeds · `design-context-models` |
| `classifier.handed_on` present | The local model ranked a command of another scope ahead and added it to the candidates | `design-context-models` (context surfaces / `base` inheritance) |
| Wrong `command_name`, `decided_by: local` | A confident local mis-route: the wrong command's training rows claim this phrasing | `detect-duplicate-capabilities` · `evaluate-intent-routing` (measure before/after) · seeds |
| Wrong `command_name`, `decided_by: remote` | The remote model was confident and wrong | Command docstrings and seeds · `evaluate-intent-routing` |
| Wrong `command_name`, `matcher_layer: fuzzy_prematch` | The request resembles another command's name | Rename the colliding command |
| `remote_failed` present | The remote call failed (`error_class`, `status_code`); the local answer or the question stood in | Environment (key, network), not workflow design |

The turn's resolution, which decides each part of a request before the review, uses the
same loop and writes the same span per decision (span contract v58, one writer for both paths), with the part's
number as the attribute `part` (a memo read writes none; a pinned plan is not detected, so it writes none): filter
`fw.intent` by `part` (under its `fw.resolve.part`) to see which part's decision needs attention. Also read `fw.split` (was the
request split correctly?) and `fw.resolve`'s `plan_ambiguity` (parts whose command was asked). Every settled part is also a row of
the intent-learning store (`fastworkflow learned list <workflow>`): `how` it was settled, the
head's facts, and `rejected_label` when a reply corrected it. A local model that turned
confidently wrong after a refit (`fastworkflow learn`, or the background refit) may have learned
wrong confirmations: remove them with `fastworkflow learned remove` (see
[reference.md](reference.md#intent-learning-store)).

**B. Were the parameters extracted correctly?** Execution command parameters and planning
command arguments go through ONE parameter pipeline (`fastworkflow/parameters/`: fill, guard,
validate, one second fill) and ONE asking rule: a question names every missing or refused value,
and two replies in a row that fill nothing stop it. An execution command's extraction is
`fw.parameters` (on the plan path too, for a part's own command, with the attribute `part`); a planning command's arguments are on `fw.plan` (`command_args`,
`command_notes`, `command_fallback`, `command_refusal_asked`).

| Observation | Diagnosis | Fix with |
|---|---|---|
| `missing_fields` non-empty | Extraction could not find the value in the request | Sharpen `Field(description=…, examples=…)` on `Signature.Input`; if the value is an opaque handle the user cannot know, declare its producer with `from_command_output` (`declare-parameter-producers`) |
| `validation_hook` with `refused_fields` naming the field (a `Refused` with `field=`) | The typed value missed the live key set and the hook offered suggestions | `validate-command-parameters` ("Resolve a value against live keys"): check the key set and thresholds; a rejection of a *valid* value usually means the wrong candidate list |
| `validation_hook` with `changed_fields` naming the field, but wrong result downstream | The fuzzy matcher rewrote the value incorrectly (auto-apply too loose, or label/uid mixup) | `validate-command-parameters` ("Resolve a value against live keys"): `auto_apply_threshold`, and return the value the *field* holds, not the label matched on |
| `validation_hook.is_valid: false` | The command's `Input` model validators refused the call: `message` says why, `refused_fields` names the fields reset to missing; `raised` means a validator crashed with an exception that is not a `Refused` or a `ValueError` | `validate-command-parameters` |
| `argument_check_refusals` / `argument_check_refusals_final` | A guard refused a value the model produced (a placeholder, or on a field marked `copied_from_command_text` a value not written in the request); one second fill ran, and `_final` is what stayed refused | Field description; unmark a field whose value is legitimately converted from the request |
| `argument_check_not_in_request` (a list of field names; on `fw.parameters`, and on `fw.plan` where it is null when empty) | Advisory, nothing was refused: these UNMARKED `str` / `int` fields hold a value the request does not write | Read it for fields that may need `copied_from_command_text` (a name, a title, an id the user must write); ignore it for a converted value |
| `argument_refusal` | A refusal of the whole call that names no input: not asked, the command ended as a failed output | `validate-command-parameters` (name the field in `Refused`) |
| `retry_round: true` on successive turns (`retry_round_ordinal` counts them) | The user is answering the parameter question; two replies in a row with no progress end it as a failed output | `validate-command-parameters` (error-message quality) + field descriptions |
| `extraction_method: llm` and the nested `fw.llm.call` shows `cache_hit: true` with a wrong completion | Cached output; the cache hit alone does not prove staleness | Verify request/model/cache provenance; use an isolated fresh cache for an authorized comparison |
| A nested `fw.llm.call` with `status: error` and an `exception` (auth, timeout) | Environment problem (keys, env files), not workflow design | Fix the env/passwords files; nothing to change in the workflow |

**C. Did the agent plan a workable sequence?** `fw.plan` spans carry
their `trigger` (`initial` for the first plan): repeated `parameter_extraction_error` replans mean the agent
cannot discover where a handle comes from → `declare-parameter-producers`.
Repeated `ask_user_response` replans point at ambiguous command surfaces or
missing context navigation → `design-context-models`.

**D. Did the conversation stall on questions?** A turn waits on ONE record at a time
(`fastworkflow/core/waiting.py`). The question shows in `record_json` on the last command output's
artifacts: `question_kind` (`agent`, `plan_choice`, `planning_parameter`, `phase_step`, `review`,
`approval`) with its payload while the turn is `awaiting_user`; in a "/" (deterministic) session,
`parameter_request` or `plan_choice` on the failed output. Multiple
`fw.question` spans in one turn (each records `kind`, `asked` and `user_responses`): review whether
each question was needed. A reply answers its question and nothing else, so a request typed as a
reply is not run. Separate repeated asks after a valid answer from necessary clarification or
retries after invalid answers; inspect state retention as well as B.

**E. Was state stored and used?** The `fw.turn` close carries
`context_mutations` (`added` / `changed` / `removed` keys with brief values).
A command that should have stored a handle but shows no mutation, or a later
turn that re-asks for stored information, is a storing-information-in-context
bug in the command's own code.

**F. A part not run?** With no agent model (`LLM_AGENT` unset), a planning command that stops at
a step code cannot run, or an ad hoc part, ends as a COMPLETED turn whose command output carries
`fw_not_run` (one entry per part: `command`, `phase`, `open_step` with its `kind` and `reason`,
`done`, `facts`, `remaining`, `available_plans`). The `open_step` says what to fix in the
planning command (`author-planning-commands`).

**G. Everything above clean?** Then the failure is the command implementation:
`fw.command` has `status: error` or `success: false` with the
`response_text`; the full `CommandResponse` (and, when
`FW_OBS_CAPTURE_TRACEBACKS=1` was set, the traceback artifact) is in the
turn's `record_json`. Read the command's `_commands/<name>.py` source next.

## 6. Recommending fixes

State the diagnosis with its evidence (span attribute values, not paraphrase),
name the feature, and load the companion skill before writing the change:

| Feature | Companion skill |
|---|---|
| Seed utterances, context layout, `base` inheritance | `design-context-models` |
| Near-duplicate commands | `detect-duplicate-capabilities` |
| Measuring whether a routing change helped | `evaluate-intent-routing` |
| `from_command_output` producer edges (shown in the runtime as `available_from`) | `declare-parameter-producers` |
| Value correction or "did you mean" in the validation hook (`changed_fields`, `refused_fields`) | `validate-command-parameters` ("Resolve a value against live keys") |
| a model validator on `Input` (its facts are the `validation_hook` span) | `validate-command-parameters` |
| Planning commands (`_commands/<name>.py`): phases, facts, step wording (`fw.phase` failures, `fw_not_run`) | [author-planning-commands](../author-planning-commands/SKILL.md) |
| Training utterance realism | `supply-training-personas` |
| Retraining mechanics after any of the above | `train-and-publish-models` |
| Publishing a regression task and creating a comparison experiment | [create-workflow-benchmarks](../create-workflow-benchmarks/SKILL.md) |
| Designing multi-turn regression content | [build-task-benchmarks](../build-task-benchmarks/SKILL.md) |
| Using human feedback to verify an improvement | [optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md) |

## Human feedback and output-quality failures

Read [reference.md](reference.md#human-feedback) for annotation anchors and read APIs.
`human_feedback` comments are distinct from agent-memory `feedback` and formal review ratings.
Do not write an annotation merely to inspect a trace or present an agent judgment as a human one.

If routing and parameters are correct, compare the final answer to command responses and the
stated expected outcome. Inspect `fw.llm.call` request limits (`call_kwargs.max_tokens`), output,
usage and errors for omitted or cut answers. Separate time spent waiting for a user from model
and command time. A cached call is not inherently stale; missing usage is not zero cost.

## Honesty notes

- Everything persisted is **redacted** (credential shapes and secret env
  values) and **size-capped**: an oversized attribute becomes a
  `{"truncated": true, "original_length": …, "sha256": …, "value": prefix}`
  envelope; oversized artifacts become `__fw_artifact_ref__` envelopes whose
  content lives in the `artifacts` table.
- Spans are best-effort: under load they can be dropped (counted in the
  `writer_health/<incarnation>` diagnostics rows; check them before concluding "no spans means
  nothing ran"). Turn records are near-lossless.
- A turn resumed in a different process finalizes without `context_mutations`
  (the baseline is not serialized), and `--generate_insights` CLI runs record
  teacher *and* student passes in one trace, each under its own
  `fw.distillation.pass` span (`fw.pass`).
- Rows with `conversation_id NULL` are real turns from conversation-less
  embedders; query by `turn_key`/`channel_id` instead of the conversation
  drill-down.
