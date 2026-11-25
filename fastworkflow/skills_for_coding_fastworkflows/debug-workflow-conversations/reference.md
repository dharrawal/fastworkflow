# observability.sqlite3: the read contract

Reference for `debug-workflow-conversations` (see SKILL.md for the triage
method). The current source uses `SCHEMA_VERSION = 10` in `observability/store.py` and span
contract 56 (`tracing.SPAN_CONTRACT_VERSION`; every span carries its `span_contract_version`, and
`tracing.SPAN_CONTRACTS` lists each span's keys). A database at another schema version is refused,
never migrated: read it with the framework version that wrote it. Inspect `PRAGMA user_version`
read-only when diagnosing a mismatch, rather than forcing a version number or adding columns.

## Location and safe access

```python
from fastworkflow import state_paths
db_path = state_paths.observability_db("<workflow_folder>")

from fastworkflow.observability.store import ReadOnlyObservabilityStore
store = ReadOnlyObservabilityStore(db_path)   # mode=ro; cannot create/migrate/write
```

Raw SQL: open read-only (`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`
or `sqlite3 "file:...?mode=ro"`). The database is WAL-mode and safe to read
while the workflow runs. **Never** open it writable to inspect it:
`ObservabilityStore` (no `ReadOnly` prefix) is the writer and creates/probes
the file on construction.

## Relevant tables and joins

Use the installed `observability/store.py` for the exact schema; this is a navigation map, not
DDL to recreate or upgrade an evidence database.

| Table | Read purpose and identity |
|---|---|
| `conversations` | `(channel_id, conversation_id)`, topic, timestamps, experiment/task/attempt labels |
| `turns` | `turn_key`, channel/conversation, ordinal, lifecycle status, command success, failure reason, answer, `record_json`, experiment/task/attempt labels |
| `spans` | `span_id`, `trace_id = turn_key`, `parent_span_id`, name/kind, command/context, start/end, status, JSON `attributes` |
| `artifacts` | `artifact_id`, turn/span anchor, content type, byte size/digest, `inline_value`, capture error |
| `experiments` | Experiment identity, description, notes, benchmark pin, capture regime, status |
| `experiment_attempts` | `(experiment_id, task_id, attempt)`, channel, outcome/lifecycle evidence, `runtime_snapshot_json` |
| `human_feedback` | Append-only timestamped review notes anchored to a turn or component spans, with `category`/`subcategory`, `feedback_uid`, frozen `anchors_json` (v7; an older store is refused, not read) |
| `train_runs` | Training metadata and `metrics_json` |
| `diagnostics` | Writer health and store/capture markers |

Use `(store_id, turn_key)` when combining sources, and retain channel identity when looking up a
conversation. Parse `record_json` and span `attributes` as JSON. `end_ns IS NULL` means a span
has not closed, not zero duration. Turns reference immutable catalog tasks by their experiment
labels; the catalog corpus itself lives under the workflow's `benchmarks/` directory.

## Span catalog

`trace_id = turn_key` links every span to its turn; `parent_span_id` builds the
tree. Attribute values over the cap (`tracing.MAX_ATTR_BYTES`, 16 KiB)
are replaced by `{"truncated": true, "original_length", "sha256", "value"}`.

### Span contract 63: the stage tree, and the old names

A turn's spans follow its stages (span contract 63, bd fix-qjne.34). A store written
before contract 63 holds other names (fw.planner.*, fw.skill.*, fw.agent.*,
fw.command.execute, fw.ask_user, fw.nlu.*) and is not read by the framework's
tools; the CHANGELOG entry for contract 63 lists every rename. The tree:

```
fw.turn (path plan|direct, mode agent|no_agent)
  fw.split > fw.split.gate | fw.split.splitter | fw.split.merge | fw.split.referent
  fw.resolve (plan_ambiguity) > fw.resolve.part (part, outcome) > fw.intent | fw.parameters | fw.question
  fw.review (outcome) > fw.question kind=review | fw.review.reply | fw.review.edit | fw.review.revision | fw.approval
  fw.run > fw.run.part (part, means, outcome ran|not_run|refused, reason)
             > fw.plan | fw.phase > fw.step > fw.command | fw.step > fw.command | fw.question
  fw.answer (written_by) > fw.llm.call purpose=conversation_summary
```

A command's own `fw.intent` / `fw.parameters` that decided nothing (exact name,
typed call, no model call, nothing refused) is not written; read `fw.command`'s
`resolved_by` / `parameters_read_by` / `parameters_valid`. Find a span by name
and by its nearest ancestor of a kind, not by its exact parent. Stage and part
spans have deterministic ids: a turn that waits on a person keeps one span per
stage and part (`resumed`, `segments`). A part dropped at resolve is a
`fw.run.part` with `outcome` not_run and `reason` `dropped_at_resolve: ...`.

### `fw.turn` (root; kind `internal`)
One per logical turn; stays open across ask_user suspensions
(`status: awaiting_user`, `end_ns` NULL) and closes at terminal finalize.
Attributes: `turn_key`, `channel_id`, `conversation_id`, `user_message`,
`unattended` (the turn ran with no ask_user tool), `status`, `success`, `failure_reason`,
`suspended_ms`, `path` (`plan` or `direct`), `mode` (`agent` or `no_agent`), and:

- **`context_mutations`**: a shallow diff of the app workflow's context across
  the turn: `{"added": {key: value_repr}, "changed": {key: {"from", "to"}},
  "removed": [keys]}`, or `null` when nothing changed (also `null` after a
  cross-process resume; the baseline is not serialized).
- **`subtask_bindings`**: per bound part, the values the review showed and the arguments it
  ran with (`shown`, `ran`, `match`, `deviation`).

### The stages: `fw.split`, `fw.resolve`, `fw.review`, `fw.run`, `fw.answer`
- **`fw.split`**: `request`, `raw` (the model's parts), `merged` (the parts the turn ran), `rules` (the code rules
  that changed the list), `failed` (`cut_off`, `error`), `attempts`, `prompt_digest`, `prompt_source` (`built_in` or
  `built_in+workflow`), `parts`. Children: `fw.split.gate` (`score`, `label`, `threshold`, `skipped`),
  `fw.split.splitter` (one per model attempt: `attempt`, `raw`, `failed`), `fw.split.merge` (`merged`, `rules`,
  `doubts`), `fw.split.referent` (`part`, `subject`, `description`), each with its model call.
- **`fw.resolve`**: `parts`, `asked`, `plan_ambiguity` (one `{subtask, means, alternatives}` per part whose command
  was asked). **`fw.resolve.part`**: `part`, `request`, `outcome` (`bound`, `ad_hoc`, `dropped`, `asked`), `means`
  (`planning_command`, `command`, `ad_hoc`), `command`, `fast_path`.
- **`fw.review`**: `outcome` (`open`, `reviewed`, `declined`, `expired`), `rounds`, `items`; a turn with no review
  has no `fw.review` (a fast path says so on `fw.resolve.part`). Children: `fw.question` (kind review),
  `fw.review.reply` (`read_as`, `by` code/head/model, `reason`, `explicit_yes_required`), `fw.review.edit`
  (`edits_read`, `edits_applied`, `repaired`, `refused_names`), `fw.review.revision` (`round`, `items_changed`,
  `rechecked`), `fw.approval`.
- **`fw.run`**: `parts`, `ran`, `not_run`, `refused`. **`fw.run.part`**: `part`, `means`, `command`, `outcome`
  (`ran`, `not_run`, `refused`), `answered_by` (code/agent), `reason`, `open_step`, `decided_by` (caller, for a
  direct action), and when an agent ran it `agent_input`, `model`, `attempts`, `final_answer`, `exhausted`,
  `error_type`.
- **`fw.answer`**: `written_by` (code, agent), `chars`, `summary_failed`; the conversation summary's model call is its
  child (`purpose` `conversation_summary`).

### `fw.plan` (kind `llm`)
A plan of the text planner, a child of the run part it plans for. Attributes: `model`, `plan` (capped) and
`trigger` (`initial`, `refused`, `handoff`, `parameter_extraction_error`, `ask_user_response`, or a progress
check's trigger). `plan_source`: `skill_call` (a planning command's typed call filled its arguments and code
expanded its steps), `text` (the text planner) or `none`. For a planning command: `command` (its name),
`command_args`, `command_notes` (what the argument guards changed), `command_fallback` (why the typed call fell
back to the text planner) and `command_refusal_asked` (its missing or refused arguments were asked of the user).
The steps that follow carry its id (`plan_span_id`).

### `fw.phase` (kind `internal`): one phase run of a planning command
A planning command runs each phase as its own run, so a part has one `fw.phase` span per phase run, its steps
its children. Opened with `command`, `phase`, `phase_index` (0-based) of `phase_count`, `step_budget`,
`first_step` and `resumed`; closed with `steps` (the phase's numbered plan), `steps_used`, `exhausted` (reached its
budget), `suspended`, `record` (the `"<Context> <label>"` its output was filled on), `facts` (the filled output,
`null` for a phase with none), and the agent loop's `attempts` and `clarification`. A failed phase has status
`error` and `facts_error`: its steps could not be built, or a required fact had no source run; no later phase of
that call ran. Code runs a phase's mechanical steps before any model: `steps_by_code` counts them and `code_stop`
says why code stopped (`done`, `open step N`, `step N failed`, `no path for step N`, `stopped`, `off`); each such
`fw.step` has `run_by: code`. Also `answer_only` / `answer_by_code` (an answer-only phase, written by code when
true), `skipped` (its `when` was false), and for a fan-out item `item`, `item_index`, `item_count`, `items_total`,
`record_issues`.

A `fw.step` with `kind` answer (`run_by` model) is the one model call that writes a planning command's answer when
its `__result__` gives none.

### `fw.step` (kind `internal`): one step, whoever ran it
A child of its phase, or of its run part for a command or ad hoc part; the step's model call (when the agent ran
it) and its `fw.command` nest under it. Attributes: `step_index`, `run_by` (`code`, `agent`, `model`), `kind`
(`command`, `ask`, `finish`, `tool`, `answer`), `plan_span_id`, `thought`, `tool_name`, `tool_args`,
`observation`; the dispatch's `raw_command` (the exact command text sent, after any folding), `response_text`,
`success`, `command_call_id`, `context_before`, `context_after`; on failures `error_type`/`tool_error` and
`recovered`; on suspension `clarification` (status `awaiting_user`); `folded_command` when the model passed
parameters as tool arguments beside `command` and the tool folded them into the command text (`tool_args` keeps
what was sent; the observation ends with a "(Ran as: ...)" note).
On a step a policy verifier rule rewrote (the workflow's `_policy/table.py`; see the
`distill-policy-verifiers` skill), these are added:
- `policy_source`: the rule's id.
- `policy_outcome` and `policy_mode`.
- `policy_table_version`.
- `policy_intent` and `policy_requests`: what the intent reader had recorded.
- `policy_judgement`: on an `ask_user` step, the question judge's reason.
- `answer`: on a `finish` step, the answer the rule sent back.

The `observation` is the rule's text, and the tool did not run. A command the
rule blocked still has its `fw.command` span (the dispatch happened),
with `success` false and the rule's text as `response_text`. Nothing executed,
and distillation leaves it out of the agent's actions.

### `fw.command` (kind `tool`)
Wraps command resolution + execution. Attributes: `raw_command` (what was
asked), `parameters` (the extracted dict), `response_text`, `success`, `error_type`,
`resolved_by` (`exact_name`, `fuzzy`, `model`, `caller`, ...), `parameters_read_by` (`typed`, `none`, `model`),
`parameters_valid`,
`command_call_id` (the same id is on the command output in `record_json`),
`context_before` / `context_after` (context type names); columns
`command_name` and `context` hold what actually ran and where. Comparing
`raw_command` to `command_name` is the first routing check.

### `fw.intent` (kind `internal`): one intent decision
One span per request: `intent_loop.resolve` decides the whole context walk at once. A reply to
the which-command question is a span of its own. The turn's resolution of a request writes one per part's
decision too, from the same writer (span contract v58), carrying `part`, the part's 1-based number (absent on the
`/` path).

| Attribute | Meaning |
|---|---|
| `context`, `reads_reply`, `question_kind`, `utterance` | `context` is the scope the decision settled in (the current one when nothing was decided). `reads_reply` is true for the reading of a reply to the which-command question, with `question_kind` `ambiguity` or `misunderstanding` |
| `matcher_layer` | Which layer decided: `exact_prefix`, `known_name_foreign_context` (with `known_name_owner_contexts` and `known_name_foreign_context_hint`), `fuzzy_prematch` (`fuzzy_distance`, `fuzzy_threshold`), `classifier`, `matcher`, or `question_reply` for a reply. Older spans may carry other values; show them as recorded |
| `decided_by`, `engine_chain` | Who decided: `name`, `fuzzy`, `foreign_name`, `local` (the local model), `remote` (the remote model), `question` (no link was confident) or `reply`; the links asked, in order |
| `classifier` | The local model's facts: `{model_tier (embedding, large or tiny), confidence, ambiguous_threshold, threshold_source, confident, top_label, topk_labels, topk_scores, p_in, p_out, outside_labels, outside_scores, handed_on, head_version}` (keys vary by layout) |
| `matcher` | The remote model's facts when it was asked: `{confident, top_label, confidence, topk_labels, topk_scores, none_score, threshold (accept), decline_threshold, unsure_threshold, scope_path, model, latency_ms, calls, error}` |
| `remote_failed` | `{error_class, status_code}`: the remote call failed and the local answer or the question stood in |
| `part` | The 1-based part of the turn this decision is for (plan path only; span contract v58) |
| `ambiguous` + `candidates` | The candidates put to the user or the agent |
| `chosen_command_elsewhere` | On a reply: the picked command lives in another scope of the path and runs there |
| `escalation_labels_discarded` | Escalation labels ranked among the candidates but not shown |
| `fuzzy_prematch_tie` | Names that matched equally well (not a match; the chain decided) |
| `command_name`, `resolved`, `is_cme_command` | The outcome; `resolved: false` means no command (a misunderstanding, or a routing hint) |

### `fw.parameters` (kind `internal`)
Wraps parameter extraction + validation for the resolved command. The plan path writes it too, for the extraction
of a part's own command (the draft), from the same writer, with `part`, the part's 1-based number (span contract v58;
absent on the `/` path and on a nested step command's span).

| Attribute | Meaning |
|---|---|
| `command_name`, `extraction_method` | `xml_regex` (read in code from the command text), `llm` (the model fill), `stored_merge` (a reply to the parameter question merged into the values kept) |
| `deterministic_form` | With `xml_regex`: `key_value` (`name=value` pairs) or `positional` (`command value`) when one of those code parsers bound the parameters; absent for XML tags |
| `unbound_value` | A lone value (`command value`) that bound to no parameter and was refused instead of running with defaults |
| `retry_round`, `retry_round_ordinal`, `reads_reply` | true when this extraction reads a reply to the parameter question; which round it is |
| `parameters_valid` | The overall verdict (span `status` stays `ok`; only exceptions mark `error`) |
| `missing_fields` / `invalid_fields` | Structured field lists (no prose parsing needed) |
| `argument_check_refusals`, `argument_check_retry`, `argument_check_refusals_final`, `argument_check_respelled` | The guards: values the model produced that a guard refused (a placeholder; on a field marked `copied_from_command_text`, a value the request does not write), the one second fill, what stayed refused (treated as not given), and values respelled to the request's or the declared spelling |
| `argument_check_not_in_request` | Advisory (on `fw.parameters` and `fw.plan`): the field names, sorted, of UNMARKED `str` / `int` fields whose extracted value the request does not write; never the values; nothing was refused. Absent on an extraction span when empty, null on a planner span |
| `part` | The 1-based part of the turn this extraction is for (plan path only; span contract v58) |
| `argument_refusal` | A refusal of the call as a whole that names no input; it is not asked, the command ends as a failed output |
| `fields_reset_not_given`, `placeholder_values`, `word_screen_values`, `command_name_values`, `out_of_range_values` | Values the screens treated as not given or refused |
| `validation_hook` | `{ran, is_valid, message, raised?, changed_fields, refused_fields}` from the command's `Input` model validators (always emitted; `ran: false` when the Input declares none). `changed_fields` lists the fields the hook corrected in place; `refused_fields` lists the fields a `fastworkflow.Refused(..., field=...)` reset to missing. `raised` is set only when the hook crashed with a non-`Refused` exception |

### `fw.question` (kind `human_wait`)
One per wait for a person (deterministic per-attempt span ids), under the part, review or run part it asks for.
Attributes: `kind` (`which_command`, `parameters`, `review`, `approval`, `decision`, `agent`), `part`,
`asked` (the question), `user_response` (the final reply), `user_responses` (every reply, in
order), `attempt`, `human_wait_ms`; the wall-clock wait also equals `end_ns - start_ns`.

### `fw.llm.call` (kind `llm`): one per DSPy LM invocation
Emitted by a DSPy callback, so it appears under whichever stage made the call
(planner, LLM parameter extraction, summarization…).

| Attribute | Meaning |
|---|---|
| `module`, `module_chain`, `module_input` | Which DSPy module ran and with what inputs |
| `model`, `messages`, `prompt`, `call_kwargs` | The exact request sent to the LM. `messages` over 16 KB is a cut envelope (`truncated`, `original_length`, `sha256`, `value`); when the span also has `prompt_slots_ref` the envelope's `value` is empty and the whole prompt is rebuilt from `prompt_slots` by `ObservabilityStore.prompt_as_sent(turn_key, span_id)` (`/api/prompt/<turn_key>/<span_id>`) |
| `output`, `provider_response`, `reasoning` | What came back (incl. provider-native reasoning when present) |
| `usage`, `cost`, `cache_hit`, `response_model`, `history_uuid`, `capture_source` | Cost accounting; **`cache_hit: true` means the completion came from the DSPy cache**: the classic "stale/frozen LLM output" tell; a hit records `cost` 0 and empty `usage` (nothing was billed) |
| `usage_capture` | Present when usage was unavailable: the LM's result never reached the capture hook and DSPy history was off (spans recorded before fix-sgsp all carry it) |
| `role`, `timeout_s`, `num_retries`, `purpose` | The model role (`LLM_PARAM_EXTRACTION`, `LLM_AGENT`, `JEV` for the remote decision model, ...; always set), the per-attempt timeout and the retries configured (attempts made are not recorded); `purpose` `conversation_summary` for the summary |
| `outcome`, `timed_out`, `exception`, `exception_type` | `ok`, `timeout` or `error` (span `status: error`): auth errors, timeouts |

### Other spans
`fw.approval` (one per approval transition: state before/after, outcome; hashes, never the
token or raw parameters), `fw.compose` (one per `run_chatbot` answer card or question form
compose call: counts, outcome, model; never user text), `fw.distillation.pass` (a teacher or
student pass; its subtree carries `fw.pass`). Reserved, not yet emitted: `fw.train.*`.

## record_json (turns.record_json)

The full internal `TurnResult`, post-redaction:

```
{ "turn_output": {
    "turn_key", "status", "failure_reason", "answer",
    "command_outputs": [                  // every command the turn executed, in order
      { "command_name", "context",
        "command_parameters": {...},      // typed params dumped to a dict
        "command_response": { "response", "success", "artifacts": {...} },
        "started_at", "duration_ms" } ],
    "success" },
  "channel_id", "conversation_id", "ordinal",
  "user_message", "refined_user_message",
  "entry_workflow_name", "entry_context",
  "started_at", "completed_at", "suspended_ms", "continuation_of" }
```

- **What the turn waits on**: while `status` is `awaiting_user`, the last command output's
  `artifacts` carry `question_kind` (`agent`, `plan_choice`, `planning_parameter`,
  `phase_step`, `review`, `approval`) and that question's payload (`plan_choice`,
  `parameter_request`, `phase_step`, `review`, `approval`). The question itself is the turn's
  one waiting record (`fastworkflow/waiting.py`), saved with the session, not in this store.
- **Parts not run** (no agent model): `artifacts.fw_not_run` lists one stop per part
  (`kind`, `command`, `phase`, `open_step` `{number, text, kind, reason}`, `done`, `facts`,
  `remaining`, `suggested_plan`, `means`, `sub_task`, and when present `results`,
  `partial_read`, `available_plans`, `request`). `fw_sub_tasks` lists a split turn's parts and
  where each one's answer sits in the answer text.
- **ask_user entries invert roles**: when `command_name == "ask_user"`,
  `command_parameters` is the agent's QUESTION and the response is the user's
  ANSWER (`success: false` = still unanswered).
- Artifact values over `FW_OBS_INLINE_ARTIFACT_BYTES` (256 KiB) are replaced by
  `{"__fw_artifact_ref__": <artifact_id>, "size", "content_type",
  "content_encoding", "error"}`: fetch the content from the `artifacts` table
  by id.
- The sketch above shows the diagnosis-relevant fields; rows may carry further
  additive fields (e.g. `workflow_name`, `next_actions`, `recommendations` on
  responses): treat unknown keys as informational.
- Non-JSON values become `{"__fw_unserializable__": <type>, "repr": ...}`.
- Tracebacks are persisted only when the run had `FW_OBS_CAPTURE_TRACEBACKS=1`;
  otherwise the artifact holds a suppression notice.

## Read API (`ReadOnlyObservabilityStore`)

| Method | Returns |
|---|---|
| `list_turns(channel_id=, conversation_id=, status=, success=, command_name=, context=, experiment_id=, task_id=, attempt=, limit=, offset=)` | Turn rows newest-first, without `record_json` (`context` is a substring match; `command_name` matches via spans) |
| `get_turn(turn_key)` | The full row incl. `record_json` (parse it yourself) |
| `get_spans(...)` | Span rows for one turn, ordered by `start_ns` (`attributes` is a JSON string). Pass the turn key POSITIONALLY: the parameter is named `trace_id` |
| `list_conversations(channel_id=, limit=, offset=)` / `list_channels()` | Navigation |
| `get_artifact(artifact_id)` | Offloaded artifact row (`inline_value` is bytes) |
| `list_train_runs(limit=)` | Training-run metrics rows, newest first (`metrics_json`) |
| `list_human_feedback(turn_key)` | All component and turn notes, oldest first; decoded `span_ids`, `category`/`subcategory`, decoded `anchors` |
| `list_task_feedback(experiment_id=, task_id=)` | Every note about one task, across attempts and turns, plus notes whose frozen pair anchor names it |
| `get_experiment(experiment_id)` / `experiment_attempt_rows(experiment_id, task_id=)` | Pin/configuration and attempt records; attempt `runtime_snapshot` is decoded or null |
| `store_identity()` | Evidence source identity |
| `writer_health(incarnation_id=None)` | A writer's drop/error counters: read this before trusting span completeness. Each writer process has its own row; with no id, every row folded together |
| `db_size_bytes()` | File + WAL size |

Conversation-memory reads also exist (`get_memory_window`,
`count_usable_turns`, `conversation_summaries`, `list_conversation_summaries`,
`dump_all_conversations`); they are not needed for failure diagnosis. Note `ReadOnlyObservabilityStore` inherits the
writer's method NAMES too; any accidental write raises on the `mode=ro`
connection rather than mutating anything.

## Query recipes

```sql
-- Confidently wrong routing: what was asked vs what ran
SELECT s.trace_id, json_extract(s.attributes,'$.raw_command') AS asked,
       s.command_name AS ran
FROM spans s WHERE s.name='fw.command'
ORDER BY s.start_ns DESC LIMIT 20;

-- Requests no link was confident about, per context, with the local model's numbers
SELECT s.context,
       json_extract(s.attributes,'$.classifier.confidence')  AS conf,
       json_extract(s.attributes,'$.classifier.ambiguous_threshold') AS thr,
       json_extract(s.attributes,'$.engine_chain') AS chain,
       json_extract(s.attributes,'$.candidates') AS candidates
FROM spans s
WHERE s.name='fw.intent' AND json_extract(s.attributes,'$.decided_by')='question';

-- How each split turn's request was split
SELECT trace_id, json_extract(attributes,'$.raw'), json_extract(attributes,'$.merged') FROM spans
WHERE name='fw.split';

-- What happened to each part of a turn
SELECT trace_id, json_extract(attributes,'$.part') AS part, command_name,
       json_extract(attributes,'$.outcome') AS outcome, json_extract(attributes,'$.reason') AS reason
FROM spans WHERE name='fw.run.part' ORDER BY trace_id, part;

-- Parameter-correction loops (users stuck re-entering values)
SELECT trace_id, COUNT(*) AS rounds FROM spans
WHERE name='fw.parameters'
  AND json_extract(attributes,'$.retry_round')
GROUP BY trace_id HAVING rounds > 1;

-- validation-hook refusals with the fields it reset and the message it gave (suggestions are in the message)
SELECT trace_id, json_extract(attributes,'$.validation_hook.refused_fields') AS refused,
       json_extract(attributes,'$.validation_hook.message') AS message
FROM spans
WHERE spans.name='fw.parameters'
  AND json_extract(attributes,'$.validation_hook.refused_fields') <> '[]';

-- values the validation hook corrected in place
SELECT trace_id, json_extract(attributes,'$.validation_hook.changed_fields') AS changed
FROM spans
WHERE spans.name='fw.parameters'
  AND json_extract(attributes,'$.validation_hook.changed_fields') <> '[]';

-- Turns that "completed" over a failed command (the quiet failures)
SELECT turn_key, user_message, answer FROM turns
WHERE status='completed' AND success=0 ORDER BY turn_key DESC;

-- What a turn stored into workflow context
SELECT json_extract(attributes,'$.context_mutations') FROM spans
WHERE name='fw.turn' AND trace_id=:turn_key AND end_ns IS NOT NULL;
```

## Trust notes

- The credential scrub, size limits and offload can replace or shorten inputs, outputs or
  attributes; missing or shortened text is not proof the runtime lacked that data.
- `call_kwargs` is flat: the completion cap is `call_kwargs.max_tokens`. Some mapping attributes
  can be JSON strings; decode them before reading fields. Compare usage, output and request limits
  together; a cap match alone is not a task-failure verdict.

- All persisted text passed the redaction pass (credential shapes + loaded
  secret env values become `[REDACTED]`).
- Turn records are near-lossless; spans are best-effort under load: check
  `writer_health()` (`spans_dropped`, `records_dropped`, `write_errors`)
  before reading absence as evidence.
- `--generate_insights` CLI turns contain teacher AND student passes in one
  trace, each under its own `fw.distillation.pass` span.

## Intent-learning store

Separate from the observability database: `<workflow>/___convo_info/intent_learning.sqlite3`
(`fastworkflow/intent_learning.py`), one row per settled part: the request, context, the
resolved command, `how` (`local`, `remote`, `name`, `user_choice`, `agent_choice`,
`review_accept`, `review_switch`), `confirmed`, the head's facts at decision time
(`head_version`, top label, `p_in`, `p_out`, threshold), the remote model's top label and
score, and `rejected_label` (the answer a reply replaced). Only confirmed rows teach a refit
(`fastworkflow learn`, or the background refit every `INTENT_LEARNING_REFIT_EVERY` new rows).
Read it with `fastworkflow learned list <workflow> [--confirmed]` and `learned stats`; delete
wrong confirmations with `fastworkflow learned remove <workflow> <ids>` before the next refit.
A refit head's version manifest names `refit_of`, `lineage` and `learned_rows`.

## Human feedback

In the working `run_chatbot` session, **Save feedback** appends a comment; it does not edit the
turn, train a model or alter scores. The UI anchors a component using its recorded span IDs.
Supported `target_kind` values are `turn`, `phase`, `step`, `span`:

- A turn anchor has `span_ids=[]`.
- Other anchors require at least one span belonging to that turn. For grouping nodes the UI
  gathers the component's recorded spans. Do not invent IDs or use labels as unique anchors.
- If there is no recorded span, annotate the turn instead.
- Each row includes `feedback_id`, `turn_key`, `target_kind`, decoded `span_ids`, `target_label`,
  `comment` and `created_at`. Comments are credential-scrubbed and retained chronologically.

```python
import json
from fastworkflow.observability.store import ReadOnlyObservabilityStore

store = ReadOnlyObservabilityStore(db_path)
turn = store.get_turn(turn_key)
spans = {row["span_id"]: row for row in store.get_spans(turn_key)}
for comment in store.list_human_feedback(turn_key):
    anchors = [spans[span_id] for span_id in comment["span_ids"] if span_id in spans]
    # Inspect anchors and json.loads(anchor["attributes"]) alongside the original comment.
```

For authorized annotation automation, the write endpoint is
`POST /post_feedback?turn_key=<encoded-key>` with exactly
`{"target_kind":"turn","span_ids":[],"target_label":"Turn","comment":"...",`
`"provenance":"coding_agent","category":"conclusions","subcategory":"what_went_wrong"}`.
Category and subcategory are enums and must pair; `GET /api/feedback-taxonomy` lists all six
pairs with the watermark text the composer shows. Reads are separate routes:
`GET /api/feedback-notes?turn_key=<encoded-key>` for one turn, and
`GET /api/task-feedback?experiment=<id>&task=<id>` for every note about a task across attempts,
turns and components (optional `category`, `subcategory`, `provenance`, `target_kind`,
`component`, `attempt`, `limit`, `offset`; no filter is applied by default).

Preserve the UI's authentication: reads and writes go to the workflow's one live database.

A POST on a live database file this process cannot write is refused (409). Otherwise the server
uses the narrow `ObservabilityStore.open_for_annotation` path; that is not a reason to open a
writer during analysis.

The agent-memory `feedback` table and the `/api/feedback` read routes it backed were removed with
their `dspy.History` injection (fix-9eg.16); those paths now 404.
