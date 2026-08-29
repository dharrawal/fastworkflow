# observability.sqlite3 — the read contract

Reference for `debug-workflow-conversations` (see SKILL.md for the triage
method). Everything here is the shipped contract for READING a workflow's
conversation logs; schema version is `PRAGMA user_version = 1` and readers
must refuse a database with a higher version.

## Location and safe access

```python
from fastworkflow import state_paths
db_path = state_paths.observability_db("<workflow_folder>")

from fastworkflow.observability_store import ReadOnlyObservabilityStore
store = ReadOnlyObservabilityStore(db_path)   # mode=ro; cannot create/migrate/write
```

Raw SQL: open read-only (`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`
or `sqlite3 "file:...?mode=ro"`). The database is WAL-mode and safe to read
while the workflow runs. **Never** open it writable to inspect it —
`ObservabilityStore` (no `ReadOnly` prefix) is the writer and creates/probes
the file on construction.

## Schema (v1)

```sql
CREATE TABLE conversations (
  channel_id TEXT NOT NULL, conversation_id INTEGER NOT NULL,
  topic TEXT, summary TEXT, status TEXT, next_ordinal INTEGER,
  started_at TEXT, last_turn_at TEXT, updated_at TEXT,
  PRIMARY KEY (channel_id, conversation_id));

CREATE TABLE conversation_counters (          -- id minting; never read for debugging
  channel_id TEXT PRIMARY KEY, next_id INTEGER NOT NULL);

CREATE TABLE turns (
  turn_key TEXT PRIMARY KEY,                  -- logical turn key = spans.trace_id
  channel_id TEXT NOT NULL, conversation_id INTEGER, ordinal INTEGER,
  user_message TEXT NOT NULL, refined_user_message TEXT,
  entry_workflow_name TEXT, entry_context TEXT,
  status TEXT NOT NULL,                       -- completed|failed|awaiting_user|cancelled|abandoned
  success INTEGER NOT NULL,                   -- 1 = every command in the turn succeeded
  failure_reason TEXT, answer TEXT,
  conversation_summary TEXT, conversation_traces TEXT,
  started_at TEXT, completed_at TEXT, suspended_ms INTEGER,
  continuation_of TEXT, record_version INTEGER NOT NULL,
  record_json TEXT NOT NULL);                 -- full TurnResult (see below)

CREATE TABLE feedback (
  turn_key TEXT PRIMARY KEY, feedback_json TEXT NOT NULL, updated_at TEXT NOT NULL);

CREATE TABLE spans (
  span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,   -- trace_id = turn_key,
                                              --   or <turn_key>~replay.<n>
                                              --   for a stored-trace replay
  parent_span_id TEXT, name TEXT NOT NULL,
  kind TEXT NOT NULL,                         -- internal|llm|human_wait|tool
  channel_id TEXT, command_name TEXT, context TEXT,
  start_ns INTEGER NOT NULL, end_ns INTEGER,  -- epoch ns; end_ns NULL = still open
  status TEXT NOT NULL,                       -- open|ok|error|cancelled|awaiting_user
  distillation_pass TEXT,                     -- teacher|student|extract, else NULL
  attributes TEXT NOT NULL);                  -- JSON object

CREATE TABLE artifacts (
  artifact_id TEXT PRIMARY KEY, turn_key TEXT NOT NULL, channel_id TEXT,
  span_id TEXT, key TEXT NOT NULL, content_type TEXT,
  size_bytes INTEGER, sha256 TEXT, inline_value BLOB, error TEXT);

CREATE TABLE train_runs (
  run_id TEXT PRIMARY KEY, workflow_fingerprint TEXT, started_at TEXT,
  completed_at TEXT, metrics_json TEXT NOT NULL);

CREATE TABLE diagnostics (                    -- writer health, schema markers
  key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
```

Indexes exist on `spans(trace_id)`, `spans(command_name)` (partial — only
rows with a command_name),
`turns(channel_id, conversation_id, ordinal)`, `turns(status)`,
`artifacts(turn_key)`, and — where the distillation tables are present —
`spans(trace_id, distillation_pass)` (partial). `turn_key` is
`YYYYMMDDTHHMMSS.ffffffZ-<12hex>` — lexicographic order is chronological
order, so `ORDER BY turn_key` sorts by time.

**`trace_id` is the turn key with exactly one exception.** A counterfactual
replay (see "Distillation" below) writes into `<turn_key>~replay.<n>`, which
is a real addressable trace with **no `turns` row**. Nothing else ever mints a
`~` id. So `spans.trace_id = turns.turn_key` is the right join for everything a
user did, and `instr(trace_id, '~') > 0` is how you find — or exclude — the
replays. Generalize a turn-key filter to `trace_id = :turn_key OR trace_id LIKE
:turn_key || '~%'` only when you actually want the replays too.

## Span catalog

`trace_id = turn_key` links every span to its turn; `parent_span_id` builds the
tree. Attribute values over the cap (`FW_OBS_MAX_ATTR_BYTES`, default 16 KiB)
are replaced by `{"truncated": true, "original_length", "sha256", "value"}`.

### `fw.turn` (root; kind `internal`)
One per logical turn; stays open across ask_user suspensions
(`status: awaiting_user`, `end_ns` NULL) and closes at terminal finalize.
Attributes: `turn_key`, `channel_id`, `conversation_id`, `user_message`,
`status`, `success`, `failure_reason`, `suspended_ms`, and
**`context_mutations`** — a shallow diff of the app workflow's context across
the turn: `{"added": {key: value_repr}, "changed": {key: {"from", "to"}},
"removed": [keys]}`, or `null` when nothing changed (also `null` after a
cross-process resume — the baseline is not serialized).

### `fw.planner.plan` / `fw.planner.replan` (kind `llm`)
Around the agent's task-planner calls. Attributes: `model`, the plan text
(capped) and, on replans, `replan_trigger`: `parameter_extraction_error` or
`ask_user_response`.

### `fw.agent.execute` (kind `internal`) — the ReAct loop as a phase
Sibling of `fw.planner.plan` under the turn; NOT `fw.command.execute` (that is
one command inside a tool call — this is the whole loop). Attributes include
`agent_input`, `resumed`, `attempts`, `suspended`, `final_answer`.

### `fw.agent.step` (kind `internal`) — one reasoning step
Child of `fw.agent.execute`; the step's reasoning `fw.llm.call` and its
`fw.agent.tool_call` nest under it. Attributes: `step_index`, `thought`,
`tool_name`, `tool_args`, `observation`; on failures `error_type`/`tool_error`
and `recovered`; on suspension `clarification` (status `awaiting_user`).

### `fw.agent.tool_call` (kind `tool`)
One per agent → workflow invocation; `raw_command` is the exact command text
the agent sent, with `response_text`/`success` (and `error_type` on failure)
added at close.

### `fw.command.execute` (kind `tool`)
Wraps command resolution + execution. Attributes: `raw_command` (what was
asked), `parameters` (the extracted dict), `response_text`, `success`; columns
`command_name` and `context` hold what actually ran and where. Comparing
`raw_command` to `command_name` is the first routing check.

### `fw.nlu.intent` (kind `internal`) — one per prediction attempt
The wildcard pipeline may predict several times per command (walking up the
context chain), so read ALL of a trace's intent spans in `start_ns` order.

| Attribute | Meaning |
|---|---|
| `context`, `stage`, `utterance` | Where/when the attempt ran (`stage` ∈ INTENT_DETECTION, INTENT_AMBIGUITY_CLARIFICATION, INTENT_MISUNDERSTANDING_CLARIFICATION) |
| `matcher_layer` | Which layer decided: `exact_prefix`, `fuzzy_prematch`, `embedding_cache`, `classifier`, `clarification_default` |
| `classifier` | Present when the classifier ran: `{model_tier: tiny\|large, confidence, ambiguous_threshold, confident, top_label, topk_labels}` |
| `ambiguous` + `candidates` | Low-confidence prediction: the candidate list shown to the user/agent |
| `escalation_labels_discarded` | Escalation labels ranked in top-k but suppressed from the prompt |
| `fuzzy_prematch_tie` | Commands that tied at the fuzzy layer (deferred to the classifier) |
| `command_name`, `resolved`, `is_cme_command` | The outcome; `resolved: false` means no local prediction — the caller walks to the parent context |
| `cache_similarity_threshold` | Present on `embedding_cache` hits (0.85) |

### `fw.nlu.param_extraction` (kind `internal`)
Wraps parameter extraction + validation for the resolved command.

| Attribute | Meaning |
|---|---|
| `command_name`, `extraction_method` | `xml_regex` (agent format), `llm` (DSPy), `stored_merge` (a NOT_FOUND retry round merging user corrections) |
| `retry_round` | true when this turn continues a parameter-correction loop |
| `parameters_valid` | The overall verdict (span `status` stays `ok`; only exceptions mark `error`) |
| `missing_fields` / `invalid_fields` | Structured field lists (no prose parsing needed) |
| `db_lookup` | List of per-field events: `{field, input_value, outcome: applied\|rejected\|declined, corrected_value, corrected, suggestions}` — the hook's three-state contract, recorded |
| `validation_hook` | `{ran, is_valid, message, raised?}` from the command's `validate_extracted_parameters` |

### `fw.ask_user` (kind `human_wait`)
One per clarifying question (deterministic per-attempt span ids). Attributes:
`agent_query`, `user_response`, `attempt`, `human_wait_ms`; the wall-clock
wait also equals `end_ns - start_ns`.

### `fw.llm.call` (kind `llm`) — one per DSPy LM invocation
Emitted by a DSPy callback, so it appears under whichever stage made the call
(planner, LLM parameter extraction, summarization…).

| Attribute | Meaning |
|---|---|
| `module`, `module_chain`, `module_input` | Which DSPy module ran and with what inputs |
| `model`, `messages`, `prompt`, `call_kwargs` | The exact request sent to the LM |
| `output`, `provider_response`, `reasoning` | What came back (incl. provider-native reasoning when present) |
| `usage`, `cost`, `cache_hit`, `response_model`, `history_uuid`, `capture_source` | Cost accounting; **`cache_hit: true` means the completion came from the DSPy cache** — the classic "stale/frozen LLM output" tell |
| `usage_capture` | Present when usage was unavailable (DSPy history disabled in that process) |
| `exception` | The LM call failed (span `status: error`) — auth errors, timeouts |

Reserved, not yet emitted: `fw.train.*`.

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

- **ask_user entries invert roles**: when `command_name == "ask_user"`,
  `command_parameters` is the agent's QUESTION and the response is the user's
  ANSWER (`success: false` = still unanswered).
- Artifact values over `FW_OBS_INLINE_ARTIFACT_BYTES` (256 KiB) are replaced by
  `{"__fw_artifact_ref__": <artifact_id>, "size", "content_type",
  "content_encoding", "error"}` — fetch the content from the `artifacts` table
  by id.
- The sketch above shows the diagnosis-relevant fields; rows may carry further
  additive fields (e.g. `workflow_name`, `next_actions`, `recommendations` on
  responses) — treat unknown keys as informational.
- Non-JSON values become `{"__fw_unserializable__": <type>, "repr": ...}`.
- Tracebacks are persisted only when the run had `FW_OBS_CAPTURE_TRACEBACKS=1`;
  otherwise the artifact holds a suppression notice.

## Read API (`ReadOnlyObservabilityStore`)

| Method | Returns |
|---|---|
| `list_turns(channel_id=, conversation_id=, status=, success=, command_name=, context=, limit=, offset=)` | Turn rows newest-first, without `record_json` (`context` is a substring match; `command_name` matches via spans) |
| `get_turn(turn_key)` | The full row incl. `record_json` (parse it yourself) |
| `get_spans(...)` | Span rows for one turn, ordered by `start_ns` (`attributes` is a JSON string). Pass the turn key POSITIONALLY — the parameter is named `trace_id` |
| `list_conversations(channel_id=, limit=, offset=)` / `list_channels()` | Navigation |
| `get_artifact(artifact_id)` | Offloaded artifact row (`inline_value` is bytes) |
| `list_train_runs(limit=)` | Training-run metrics rows, newest first (`metrics_json`) |
| `writer_health()` | The writer's drop/error counters — read this before trusting span completeness |
| `db_size_bytes()` | File + WAL size |

Additional conversation-memory reads exist (`get_memory_window`,
`count_usable_turns`, `conversation_summaries`, `list_conversation_summaries`,
`dump_all_conversations`) — Phase-7 consolidation surface, usable but not
needed for failure diagnosis. Note `ReadOnlyObservabilityStore` inherits the
writer's method NAMES too; any accidental write raises on the `mode=ro`
connection rather than mutating anything.

## Query recipes

```sql
-- Confidently wrong routing: what was asked vs what ran
SELECT s.trace_id, json_extract(s.attributes,'$.raw_command') AS asked,
       s.command_name AS ran
FROM spans s WHERE s.name='fw.command.execute'
ORDER BY s.start_ns DESC LIMIT 20;

-- Ambiguity hot spots per context, with the classifier's numbers
SELECT s.context,
       json_extract(s.attributes,'$.classifier.confidence')  AS conf,
       json_extract(s.attributes,'$.classifier.ambiguous_threshold') AS thr,
       json_extract(s.attributes,'$.candidates') AS candidates
FROM spans s
WHERE s.name='fw.nlu.intent' AND json_extract(s.attributes,'$.ambiguous');

-- Parameter-correction loops (users stuck re-entering values)
SELECT trace_id, COUNT(*) AS rounds FROM spans
WHERE name='fw.nlu.param_extraction'
  AND json_extract(attributes,'$.retry_round')
GROUP BY trace_id HAVING rounds > 1;

-- db_lookup rejections with what was offered instead
SELECT trace_id, json_extract(value,'$.field') AS field,
       json_extract(value,'$.input_value') AS typed,
       json_extract(value,'$.suggestions') AS offered
FROM spans, json_each(json_extract(spans.attributes,'$.db_lookup'))
WHERE spans.name='fw.nlu.param_extraction'
  AND json_extract(value,'$.outcome')='rejected';

-- Turns that "completed" over a failed command (the quiet failures)
SELECT turn_key, user_message, answer FROM turns
WHERE status='completed' AND success=0 ORDER BY turn_key DESC;

-- What a turn stored into workflow context
SELECT json_extract(attributes,'$.context_mutations') FROM spans
WHERE name='fw.turn' AND trace_id=:turn_key AND end_ns IS NOT NULL;
```

## Distillation (`fastworkflow run --generate_insights`)

Present only when `diagnostics.schema_features` names `distillation_v1` (older
DBs have neither the tables nor `spans.distillation_pass`; check before
projecting either, or you get `no such column`). A distilled turn runs the
SAME user message twice — a teacher pass and a student pass — diffs them, and
appends rules to `Insights/<workflow>/planning_agent_insights.md` and
`execution_agent_anti_patterns.md`. These tables are what makes those rules
checkable against the evidence that produced them.

```sql
CREATE TABLE distillation_runs (              -- one per compared message
  run_id TEXT PRIMARY KEY,
  turn_key TEXT NOT NULL,                     -- == spans.trace_id == turns.turn_key
  channel_id TEXT, conversation_id INTEGER,
  user_message TEXT NOT NULL,
  workflow_name TEXT, entry_context TEXT,
  comparable INTEGER NOT NULL,                -- 0 => divergences UNUSABLE as evidence
  comparable_reason TEXT,                     -- fingerprint-differs | evidence-incomplete
                                              --   | teacher-raised | student-raised
  isolation_verified INTEGER,                 -- NULL until EXP-013; promotion needs 1
  fingerprint_teacher TEXT, fingerprint_student TEXT,
  restore_ok_pre_student INTEGER, restore_ok_post_compare INTEGER,
  cache_asymmetric INTEGER NOT NULL DEFAULT 0,-- a hit in one pass, a miss in the other:
                                              --   a COST confound, not a comparability one
  left_steps INTEGER, right_steps INTEGER,
  planning_diverged INTEGER NOT NULL DEFAULT 0,
  exec_diverged INTEGER NOT NULL DEFAULT 0,
  material_divergences INTEGER NOT NULL DEFAULT 0,
  planning_insights INTEGER NOT NULL DEFAULT 0,
  execution_insights INTEGER NOT NULL DEFAULT 0,
  extractor_empty INTEGER NOT NULL DEFAULT 0, -- diverged but extracted nothing
  extractor_model TEXT, insight_set_json TEXT,
  replay_of TEXT, replay_trace_id TEXT,       -- set only on a counterfactual replay
  pinned INTEGER NOT NULL DEFAULT 0,          -- evidence held against retention
  pinned_at TEXT, pinned_span_count INTEGER,
  turn_fields_from TEXT,
  evidence_pruned INTEGER NOT NULL DEFAULT 0, -- the trace behind this run is gone
  started_at TEXT, completed_at TEXT, run_json TEXT NOT NULL);

CREATE TABLE distillation_passes (            -- one row per pass; N-pass ready
  run_id TEXT NOT NULL, pass_label TEXT NOT NULL,  -- joins spans.distillation_pass
  role TEXT NOT NULL,                         -- teacher|student|extractor|student-replay
  seq INTEGER NOT NULL, trace_id TEXT NOT NULL,
  agent_model TEXT, planner_model TEXT, model_params_json TEXT,
  entry_fingerprint TEXT, exit_fingerprint TEXT,
  first_span_id TEXT, last_span_id TEXT,
  wall_ms INTEGER, tokens INTEGER, cost_usd REAL,
  cache_hits INTEGER, cache_misses INTEGER,
  entry_prompt_fingerprint TEXT, exit_prompt_fingerprint TEXT,
  history_bound INTEGER, summary_hash TEXT, spans_dropped_delta INTEGER,
  entry_inputs_json TEXT,                     -- PROMPT INPUTS, not restorable state
  PRIMARY KEY (run_id, pass_label));

CREATE TABLE distillation_divergences (       -- the aligned diff, structured
  divergence_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
  level TEXT NOT NULL,                        -- plan|action|run
  left_pass TEXT NOT NULL, right_pass TEXT NOT NULL,
  align_index INTEGER NOT NULL,
  kind TEXT NOT NULL,                         -- identical | same-command-different-params
                                              -- | param-value-only | extra-in-student
                                              -- | missing-in-student | reordered
                                              -- | different-answer-same-actions
  material INTEGER,                           -- 1|0|NULL(run was not comparable)
  replayable INTEGER NOT NULL DEFAULT 1,
  command_key TEXT, command_name TEXT, context TEXT,
  left_step_key TEXT, right_step_key TEXT,
  left_span_id TEXT, right_span_id TEXT,      -- -> spans.span_id
  param_diff_json TEXT, detail_json TEXT NOT NULL);

CREATE TABLE distillation_insights (
  insight_id TEXT PRIMARY KEY,                -- stable; survives renumbering the .md
  run_id TEXT NOT NULL, kind TEXT NOT NULL,   -- planning|execution
  text TEXT NOT NULL, text_hash TEXT NOT NULL,-- reverse index from a markdown line
  extractor_span_id TEXT, insight_file TEXT,
  file_entry_number INTEGER,                  -- DISPLAY ONLY — never an identifier
  created_at TEXT NOT NULL);

CREATE TABLE distillation_insight_citations ( -- insight <-> divergence, both ways
  insight_id TEXT NOT NULL, divergence_id TEXT NOT NULL,
  PRIMARY KEY (insight_id, divergence_id));

CREATE TABLE distillation_verdicts (          -- append-only, with supersede
  verdict_id TEXT PRIMARY KEY, insight_id TEXT NOT NULL,
  verdict TEXT NOT NULL,                      -- supported | not-supported-by-cited-evidence
                                              -- | overfit-to-single-turn
                                              -- | duplicate-of-existing
                                              -- | contradicted-by-other-turns
  note TEXT, actor TEXT NOT NULL,             -- 'human' | 'agent:<name>' | 'replay'
  replay_run_id TEXT, superseded INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL);
```

**The joins, in one line each.** `distillation_runs.turn_key = turns.turn_key`
(the user-visible turn) and `= spans.trace_id` (its evidence);
`distillation_passes.run_id` + `.pass_label = spans.distillation_pass` (one
pass's spans); `distillation_divergences.left_span_id` / `.right_span_id` →
`spans.span_id` (the two sides of one aligned step);
`distillation_insight_citations` binds an insight to the divergences it was
drawn from, in both directions; `distillation_verdicts.insight_id` carries the
adjudication history, newest non-superseded row first.

**Three things to filter on, every time.** `r.comparable = 1` — a run whose two
passes did not start from the same state proves nothing, and the rows are kept
only so the quarantine is visible. `r.replay_of IS NULL` — a replay is a TEST
of an insight, so counting it as support for that insight double-counts a
result in its own favour. And for the promotion decision only,
`r.isolation_verified = 1` — promotion is a causal claim, and until EXP-013
lands nothing sets that column, so the promotion query legitimately returns
nothing. Read that as "not yet checkable", never as "no support".

### Distillation recipes

Executed against a fixture holding one comparable run, one non-comparable run,
one replay, one no-divergence run, a run-level (NULL-`command_name`) divergence
and a NULL-`command_name` failed span — the shapes that silently break these
queries. "It parses" is not verification here; three-valued logic on nullable
`command_name` turns a wrong query into zero rows and no error.

```sql
-- Which turns SUPPORT insight X (the run it came from included)
WITH cited AS (
  SELECT DISTINCT d.command_key, d.kind
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  WHERE i.insight_id = :insight_id
)
SELECT r.run_id, r.turn_key, r.started_at,
       d.divergence_id, d.kind, d.command_name, d.material
FROM distillation_runs r
JOIN distillation_divergences d ON d.run_id = r.run_id
JOIN cited ON cited.command_key = d.command_key AND cited.kind = d.kind
WHERE r.comparable = 1 AND r.replay_of IS NULL
ORDER BY r.started_at DESC;

-- Which turns CONTRADICT insight X: the cited command ran in the STUDENT pass
-- and produced no divergence of the cited kind — i.e. the rule would have
-- fired wrongly. The `cited.command_name IS NOT NULL` guard is load-bearing:
-- run-level divergences carry no command, and without it the whole query
-- returns zero rows with no error.
WITH cited AS (
  SELECT DISTINCT d.command_key, d.command_name, d.kind
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  WHERE i.insight_id = :insight_id
)
SELECT r.run_id, r.turn_key, r.started_at, cited.command_name
FROM distillation_runs r
CROSS JOIN cited
WHERE cited.command_name IS NOT NULL
  AND r.comparable = 1 AND r.replay_of IS NULL
  AND EXISTS (SELECT 1 FROM spans s
               WHERE s.trace_id = r.turn_key
                 AND s.distillation_pass = 'student'
                 AND s.name = 'fw.command.execute'
                 AND s.command_name = cited.command_name)
  AND NOT EXISTS (SELECT 1 FROM distillation_divergences d2
                   WHERE d2.run_id = r.run_id
                     AND d2.command_key = cited.command_key
                     AND d2.kind = cited.kind)
ORDER BY r.started_at DESC;

-- Which turns CONTRADICT a RUN-LEVEL insight. A run-level divergence has no
-- command to key on, so the contradiction is defined over outcomes: comparable
-- runs in the same context whose two passes agreed on the final answer. Note
-- `IS` and not `=` on the join — both columns are nullable.
WITH cited AS (
  SELECT DISTINCT r0.entry_context, r0.workflow_name
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  JOIN distillation_runs r0 ON r0.run_id = d.run_id
  WHERE i.insight_id = :insight_id AND d.level = 'run'
)
SELECT r.run_id, r.turn_key, r.started_at, r.user_message
FROM distillation_runs r
JOIN cited ON cited.entry_context IS r.entry_context
          AND cited.workflow_name IS r.workflow_name
WHERE r.comparable = 1 AND r.replay_of IS NULL
  AND NOT EXISTS (SELECT 1 FROM distillation_divergences d2
                   WHERE d2.run_id = r.run_id AND d2.level = 'run')
ORDER BY r.started_at DESC;

-- The promotion view: support and contradiction counts per insight, plus the
-- live verdict. Correlated subqueries rather than a join chain ON PURPOSE — a
-- join loses the insight row whenever the filter matches nothing, so a
-- freshly extracted rule with no corroboration would VANISH from the list a
-- human uses to decide, instead of appearing with a zero.
SELECT i.insight_id, i.kind, i.text,
       (SELECT v.verdict FROM distillation_verdicts v
         WHERE v.insight_id = i.insight_id AND v.superseded = 0
         ORDER BY v.created_at DESC LIMIT 1)                     AS verdict,
       (SELECT COUNT(DISTINCT sup.run_id)
          FROM distillation_insight_citations c
          JOIN distillation_divergences cit ON cit.divergence_id = c.divergence_id
          JOIN distillation_divergences sup  ON sup.command_key = cit.command_key
                                            AND sup.kind        = cit.kind
          JOIN distillation_runs r           ON r.run_id = sup.run_id
         WHERE c.insight_id = i.insight_id
           AND cit.command_key IS NOT NULL
           AND r.comparable = 1 AND r.replay_of IS NULL
           AND r.isolation_verified = 1 AND r.evidence_pruned = 0) AS support_runs
FROM distillation_insights i
ORDER BY support_runs DESC;

-- Divergence rate by week: is the student improving as insights accumulate?
SELECT strftime('%Y-W%W', r.started_at) AS week, COUNT(*) AS runs,
       SUM(CASE WHEN r.exec_diverged = 1 THEN 1 ELSE 0 END) AS diverged,
       SUM(r.material_divergences) AS material,
       ROUND(1.0 * SUM(CASE WHEN r.exec_diverged = 1 THEN 1 ELSE 0 END)
                 / COUNT(*), 3) AS rate
FROM distillation_runs r
WHERE r.comparable = 1 AND r.replay_of IS NULL
GROUP BY week ORDER BY week;

-- All material missing-in-student divergences by command, with the run ids
SELECT d.command_name, COUNT(*) AS n, GROUP_CONCAT(DISTINCT r.run_id) AS run_ids
FROM distillation_divergences d
JOIN distillation_runs r ON r.run_id = d.run_id
WHERE d.kind = 'missing-in-student' AND d.material = 1
  AND r.comparable = 1 AND r.replay_of IS NULL
GROUP BY d.command_name ORDER BY n DESC;

-- Divergences by taxonomy kind: which WAY does the student go wrong, rather
-- than on which command. `identical` rows are stored on purpose, so this is a
-- denominator as well as a set of counts.
SELECT d.level, d.kind, COUNT(*) AS n,
       SUM(CASE WHEN d.material = 1 THEN 1 ELSE 0 END) AS material,
       COUNT(DISTINCT d.run_id) AS runs
FROM distillation_divergences d
JOIN distillation_runs r ON r.run_id = d.run_id
WHERE r.comparable = 1 AND r.replay_of IS NULL
GROUP BY d.level, d.kind ORDER BY n DESC;

-- Teacher vs student cost and latency (cache-asymmetric runs excluded: that
-- is exactly the confound that makes the cost columns incomparable)
SELECT p.role, COUNT(*) AS passes, SUM(p.tokens) AS tokens,
       ROUND(SUM(p.cost_usd), 4) AS cost_usd, ROUND(AVG(p.wall_ms)) AS avg_ms,
       SUM(p.cache_hits) AS cache_hits, SUM(p.cache_misses) AS cache_misses
FROM distillation_passes p
JOIN distillation_runs r ON r.run_id = p.run_id
WHERE r.comparable = 1 AND r.cache_asymmetric = 0
  AND p.role IN ('teacher','student')
GROUP BY p.role;

-- From a SPAN back to the rules drawn from it. The other direction of the
-- provenance chain, and the one that answers "this tool call looks wrong —
-- did we already write a rule about it?"
SELECT i.insight_id, i.kind, i.text, d.divergence_id, d.kind AS divergence_kind,
       d.material,
       (SELECT v.verdict FROM distillation_verdicts v
         WHERE v.insight_id = i.insight_id AND v.superseded = 0
         ORDER BY v.created_at DESC LIMIT 1) AS verdict
FROM distillation_divergences d
JOIN distillation_insight_citations c ON c.divergence_id = d.divergence_id
JOIN distillation_insights i ON i.insight_id = c.insight_id
WHERE d.left_span_id = :span_id OR d.right_span_id = :span_id;

-- From a line in the markdown file back to the turns behind it
SELECT i.insight_id, r.run_id, r.turn_key, r.started_at
FROM distillation_insights i JOIN distillation_runs r ON r.run_id = i.run_id
WHERE i.text_hash = :text_hash ORDER BY r.started_at;

-- One pass's trajectory, unmerged
SELECT s.start_ns, s.name, s.command_name,
       json_extract(s.attributes,'$.parameters') AS params
FROM spans s
WHERE s.trace_id = :turn_key AND s.distillation_pass = 'student'
ORDER BY s.start_ns;

-- Negative outcomes, which are evidence too: diverged-but-extracted-nothing,
-- and the runs where the two passes simply agreed
SELECT run_id, turn_key, user_message,
       CASE WHEN extractor_empty = 1 THEN 'diverged, extractor returned EMPTY'
            WHEN planning_diverged = 0 AND exec_diverged = 0 THEN 'no divergence'
            ELSE 'diverged' END AS outcome
FROM distillation_runs
WHERE comparable = 1 AND replay_of IS NULL
ORDER BY started_at DESC;
```

### Distillation read API and HTTP surface

`ReadOnlyObservabilityStore` adds `list_distillation_runs(...)`,
`get_distillation_run(run_id)`, `list_distillation_divergences(run_id, ...)`,
`distillation_insights(run_id=|insight_id=|text_hash=)`,
`distillation_turn_markers(turn_keys)`, `distillation_corpus(channel_id=)`,
`distillation_retention_status(run_id)`, `distillation_evidence_shortfall(run_id)`
and `export_distillation_run(run_id)`; `get_spans` takes an optional
`distillation_pass=` (`'none'` selects the spans belonging to no pass).

The chatbot serves the same shapes over HTTP — `GET /api/distillation/runs`,
`/run/<run_id>`, `/divergences/<run_id>`, `/insights?run=|insight=|text_hash=`,
`/corpus`, `/export/<run_id>`, and `GET /api/spans/<turn_key>?pass=<label>`.
`POST /api/distillation/verdict` (`{insight_id, verdict, note?, actor}`) is the
one write, an append to `distillation_verdicts` and nothing else.

**Prefer `export_distillation_run` / `/api/distillation/export/<run_id>` over
assembling a run yourself.** It returns the run, its passes, the alignment, the
insights, their citations and verdicts, the retention explanation and both
passes' spans as one JSON document — built entirely out of stored rows, so it
inherits the redaction that ran at the sink boundary. Assembling the same
thing from live objects would route around that and put credentials in a file
whose whole point is being handed to another agent.

## Trust notes

- All persisted text passed the redaction pass (credential shapes + loaded
  secret env values become `[REDACTED]`).
- Turn records are near-lossless; spans are best-effort under load — check
  `writer_health()` (`spans_dropped`, `records_dropped`, `write_errors`)
  before reading absence as evidence.
- `--generate_insights` CLI turns contain teacher AND student passes in one
  trace, told apart by `spans.distillation_pass` — so duplicate-looking tool
  calls are expected there, and `AND distillation_pass = 'student'` is what
  turns the merged trace back into one trajectory. A run whose
  `distillation_runs.comparable = 0` is **quarantined evidence**: the rows are
  there, and every recipe below filters them out on purpose.
