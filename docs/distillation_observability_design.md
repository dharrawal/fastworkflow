# fastWorkflow Distillation Observability — Design Doc (Phase 0)

Status: **DECIDED — Phase-0 gate for epic `fix-sb8`, revision 2 (post
adversarial review).** Implementation of `fix-sb8.2`…`fix-sb8.14` and
`fix-kw7.11` may begin against the rulings in §20. §21 records the objections
that were argued and rejected; §22 is the review record.
Author: Claude (with Dhar Rawal), 2026-08-28.
Baseline verified against: v3.2.0 (`observability` @ 9904df5). Every file:line
citation in this document was read against that tree.
Parent platform design: `docs/fastworkflow_observability_studio_design.md`
(rulings `[R1]`–`[R28]`); Phase-B slice: `docs/observability_phase7_consolidation_design.md`.
Gate precedent: `fix-kw7.1` did this for the parent platform; this document is
the same gate for the distillation surface. Revision 2 incorporates four
adversarial reviews (§22): eight blocking findings, six of which were fixed by
changing a decision rather than softening a sentence.

**Ruling namespace.** This document uses a fresh `[DR#]` namespace, not a
continuation of `[R1]`–`[R28]`. The parent's `[R]` numbers are *review-round-1
findings* against a specific document, and each ruling is the owner's answer to
a numbered finding. Continuing the sequence would assert that `[R29]` answers a
finding in that review, which it does not. `[DR#]` rulings are peer to `[R#]`
rulings and are binding on the same terms. Where a `[DR]` ruling supersedes an
`[R]` ruling, it says so explicitly (only `[DR14]` does, reversing `[R24]`).

Where this document conflicts with `fastworkflow_observability_studio_design.md`,
that document's decisions stand unless a `[DR]` ruling here explicitly
supersedes them with a stated reason.

---

## 0. Origin: the problem

`fastworkflow run --generate_insights` runs the **same user message twice** — a
teacher pass and a student pass — diffs them, and appends two markdown files:
`planning_agent_insights.md` ("what TO DO") and
`execution_agent_anti_patterns.md` ("what NOT to do"). Those two files are the
product. Everything that justifies them is discarded.

Verified against the tree, this is what is thrown away:

| What is lost | Where it dies |
|---|---|
| Pass boundaries in the trace | `distillation.py` never imports `tracing` (`grep -c tracing fastworkflow/distillation.py` → 0); every span takes `trace_id = get_turn_key(host)` (`tracing.py:329,342`), so both passes land in one trace under one `root_span_id(turn_key)` (`tracing.py:337,211`) with nothing but the `model` attribute distinguishing them |
| The comparison inputs | `teacher_traj`/`student_traj`/`teacher_actions`/`student_actions`/`teacher_plans`/`student_plans` are locals in `distill_message` (`distillation.py:840-863`) and die with the call |
| The structure of the diff | `compare_trajectories` (`distillation.py:315-407`) falls back to `set(teacher_sigs) - set(student_sigs)` over `(command_name, json.dumps(params, sort_keys=True))` and returns a **prose string**. Set semantics lose repeats; a single wrong parameter renders as two unrelated entries; failed commands (no `command_name`), `ask_user` records, and `ErrorCorrection/*` are filtered out entirely (`:333-340`) |
| Plan-level structure | `PlanningStep.generated_plan` is `prediction.next_steps.split()` (`workflow_agent.py:686`) — a list of **words**, so `compare_planning_traces` (`distillation.py:413-443`) is a word-for-word string comparison of two planner outputs |
| The extractor's reasoning | `extract_insights` (`:592`) and `extract_planning_insights` (`:719`) run `dspy.ChainOfThought` inside `dspy.context(lm=...)`; their `fw.llm.call` spans DO land in the DB (via `@dspy_logger.observe_dspy_calls` on `_execute_message`, `workflow_execution_context.py:1069`) but with **no marker** separating them from the agent's own calls, and no record of the parsed result |
| Provenance | `_append_numbered_insights` (`:665-693`) writes a bare running integer. No id, no `turn_key`, no `span_id`, no timestamp, no model, no divergence reference |
| Negative outcomes | "diverged but the extractor returned EMPTY" (`:635-639`, `:767-769`) and "no divergence at all" (`:903-915`) are both unrecorded, and both are evidence |
| Comparability | `snapshot_workflow_state` (`:197-207`) calls `Workflow._to_dict()`, which returns `self._context` **by reference** (`workflow.py:450`), so the "snapshot" aliases the live dict and `restore_workflow_state` (`:209-234`) undoes nothing done in place. Application object state (`_root_command_context`, `_current_command_context`, `workflow.py:166-168`) is never captured at all |

So the artifacts cannot be verified against the evidence that produced them, by
a human or by an agent.

**Scope: record and review, not re-tune.** This epic does not change extraction
prompts or insight-quality heuristics. It makes the chain

```
teacher trace → student trace → aligned diff → divergence record
              → extractor call → emitted insight
```

inspectable and falsifiable end to end, and exposes it in the chatbot UI and to
agents.

**The bar is set by the downstream use.** These recommendations are raw material
for extracting **skills** and **verifiers** for the planner. A rule is only
promotable to a skill if you can see how many turns support it, which turns
contradict it, and whether applying it actually removes the divergence.
Counterfactual replay (`fix-sb8.11`) is the mechanism that turns a stored
`(turn, teacher-trace)` pair into an executable verifier: replay the student
with the candidate insight injected and diff against the **stored** teacher
trace — one small-model run, no teacher spend. Every decision below is scored
against whether it makes that mechanism work.

---

## 1. Decisions

- **D1 — One identity rule, not two.** A trace is a unit of execution
  attributable to one turn key. `spans.trace_id` is the turn key for executions
  that belong to a recorded turn, and `<turn_key>~<suffix>` for executions that
  **re-run** a recorded turn without being one (replay). A new indexed column
  `spans.distillation_pass` says which pass within a trace. §3 argues this
  against the three candidates.
- **D2 — The pass is a first-class span, not only a column.** `fw.distill.run`
  and `fw.distill.pass` are emitted and parent the pass's work, so the pass is a
  structural fact in the waterfall. The column is the *index* on that structure
  (§9, `[DR16]`).
- **D3 — Comparability outranks the UI.** A divergence is evidence of a student
  mistake only if both passes started identical. Entry fingerprints, the restore
  assertion, per-role model identity, and per-pass cache hit/miss (§6) ship
  before `fix-sb8.7`/`.8`.
- **D4 — The alignment is computed and persisted server-side; the SPA paints a
  stored record.** There is no JavaScript test harness anywhere in the repo (no
  `package.json`; every UI "test" is a Python byte-substring assertion over
  `index.html`, `tests/test_run_chatbot_server.py:376-389`). Sequence alignment
  written in the browser would be untestable by construction.
- **D5 — Additive schema, no `SCHEMA_VERSION` bump, plus a feature marker.**
  The fail-closed `user_version` gate is too coarse to express "same version,
  more tables"; §11 replaces it with runtime feature detection.
- **D6 — Recording is best-effort; the *conclusion* is not.** Span and run
  writes follow `[R14]` (a write failure never fails a turn). But a run whose
  comparability could not be established is marked NON-COMPARABLE and is
  excluded from every aggregate — silence is never read as agreement.
- **D7 — Process.** This is the `fix-sb8.1` gate. Schema or identity changes
  after this document re-enter review.

---

## 2. Baseline reality (v3.2.0, verified)

- `SCHEMA_VERSION = 1` (`observability_store.py:52`), fail-closed on
  newer (`:387-390` for the writer, `:1249-1255` for `ReadOnlyObservabilityStore`).
- `spans` is `(span_id PK, trace_id, parent_span_id, name, kind, channel_id,
  command_name, context, start_ns, end_ns, status, attributes)`
  (`observability_store.py:322-329`), indexed `idx_spans_trace` on `trace_id`
  and `idx_spans_command` on `command_name` **partial**
  (`WHERE command_name IS NOT NULL`, `:341-342`).
- `_ensure_schema` already performs an unversioned additive column migration in
  a `PRAGMA table_info`-guarded block (`:390-402`, `conversations.updated_at`).
- `upsert_span_rows`' `ON CONFLICT(span_id) DO UPDATE` set is exactly
  `end_ns, status, attributes, command_name, context` (`:589-601`). **`trace_id`
  and `parent_span_id` are NOT updatable** — a mismatched close cannot move a
  span between traces; it is silently dropped instead. (This corrects a claim
  made during option review; the mechanics matter for §3.5.)
- `SQLiteTraceSink.emit_span` rebuilds the `Span` **field by field**
  (`:1333-1345`).
- `prune()` deletes **spans and artifacts only**, on a 30-day horizon
  (`_DEFAULT_RETENTION_DAYS = 30`, `:58`; `:1122-1141`) plus 1 GiB oldest-first
  eviction (`_DEFAULT_DB_MAX_BYTES = 1_073_741_824`, `:57`; `:1167-1180`), and
  it runs at **every sink startup** (`SQLiteTraceSink.__init__`, `:1321-1325`).
  Conversations and turn records are exempt (`:1100-1102`, `[R16]`).
- `run_clear_conversations` (`server.py:1205-1209`) constructs a **writable**
  `ObservabilityStore(db_path)`, entirely separate from the per-request
  `open_store()` → `ReadOnlyObservabilityStore` (`server.py:466-479`). This is
  the precedent §12 follows.
- The chatbot never percent-decodes a path segment: `_handle_get` slices the raw
  `urlsplit(self.path).path` (`server.py:826-828`, `:1059-1060`) and
  `grep -c unquote fastworkflow/run_chatbot/server.py` → **0**.
- `selectTurn` (`index.html:822-837`) issues `/api/turn/<k>` **and**
  `/api/spans/<k>` on one string inside a `Promise.all`, and `api()` throws on
  any non-2xx (`:529-535`). A 404 on either kills the whole render.
- `buildTurnTree` computes `extent = rootSpan ? spanExtent([rootSpan]) :
  mergeExtents(children.map(...))` (`index.html:1420-1422`), and
  `renderWaterfall` lays every row against one `[t0,t1]` (`:1658-1665`,
  `:1681-1684`).
- `fw.llm.call` attributes already carry `model` (`utils/dspy_logger.py:374`),
  `usage` (`:425`), `cost` (`:426`), and `cache_hit` (`:429-430`).
  (Revision 2: the earlier citation of `:395` for `model` was wrong — `:395` is
  `innermost.spans.append(...)`. Corrected.) **Verified: §6's
  per-call cache hit/miss is a query over existing data, not new
  instrumentation.**
- `TraceSink` is a three-method Protocol — `emit_span`, `emit_turn_record`,
  `record_conversation_label` (`tracing.py:159-178`) — and `NoOpTraceSink`
  implements exactly those (`:180-195`). Neither `WorkflowExecutionContext` nor
  `ChatSession` holds an `ObservabilityStore`; only `SQLiteTraceSink` does
  (`observability_store.py:1285`). **There is no write path for a non-span,
  non-turn row today.** §9's `[DR46]` builds one.
- `forget_channel` (`observability_store.py:1185-1216`) and
  `clear_conversations` (`:1218-1233`) are **hardcoded five-table lists**
  (`feedback`, `spans`, `artifacts`, `turns`, `conversations`). Nothing in them
  generalizes. `[DR44]` extends both.
- The size-cap eviction loop breaks on `if cur.rowcount == 0` (`:1178-1179`),
  so any predicate placed on the *outer* `DELETE` rather than inside the victim
  subquery turns "this batch was all pinned" into "stop evicting". `[DR52]`.
- Distillation is unreachable from Topology B: the guard is
  `self._generate_insights and self.user_message_queue is not None`
  (`workflow_execution_context.py:1643`), and only `ChatSession` injects queues
  (`chat_session.py:126-133`).

---

## 3. IDENTITY (requirement 1) — the load-bearing decision

### 3.1 The decision `[DR1]`

**Adopt option (b) — `trace_id` stays `== turn_key`, with a new indexed
nullable column `spans.distillation_pass TEXT` — extended by a strictly bounded
derived-trace-id namespace for executions that have no turn.** Formally:

> `base_turn_key(trace_id) == turns.turn_key` for every span, where
> `base_turn_key(x) = substr(x, 1, instr(x,'~')-1)` when `x` contains `~`, else
> `x`. A `~` suffix appears **only** for a counterfactual replay
> (`fix-sb8.11`), which is not a turn and must never mint one.

This is a strict generalization of today's invariant: for every span in every
existing DB, and for every span a live turn will ever write — distilled or not —
`base_turn_key(trace_id) == trace_id == turn_key`, unchanged.

The `~` separator is not cosmetic. `#`, `%`, `/` and space are all escaped by
`encodeURIComponent`, and the chatbot never `unquote`s a path segment
(`server.py:826-828`; verified zero `unquote` occurrences), so any of those would
arrive percent-encoded and silently return an empty span list. `~` is RFC-3986
unreserved, is not a SQL `LIKE` wildcard, cannot occur in a minted turn key
(`turn.py:91-107`: `%Y%m%dT%H%M%S.%f` + `Z-` + 12 hex), and at `0x7E` sorts after
every character a turn key can contain, so a replay trace sorts adjacent to the
turn it replays. `[DR2]`

**Pass labels** are free text, not an enum: `'teacher'`, `'student'`,
`'extractor'`, and for N-pass growth `'student.a'`, `'student.b'`, …. `NULL`
means turn-level or non-distillation. The column is named `distillation_pass`,
never `pass` — `pass` is a Python keyword and a `Span` dataclass field named
`pass` is a syntax error. `[DR3]`

### 3.2 Why (a) — derived per-pass trace ids — loses

Option (a) is the option this design borrows from, and it is worth being precise
about which half is good. Its *replay* argument is correct and is adopted
verbatim in §3.1. Its *live-turn* argument does not survive the code.

1. **Its headline claim is false against the file it cites.** (a)'s strongest
   argument is "zero SPA logic changes — `fix-sb8.7` reduces to swapping the
   string in the fetch at `index.html:829`." But `selectTurn` (`:822-837`) passes
   **one** string to `/api/turn/<k>` and `/api/spans/<k>` inside a `Promise.all`;
   a pass trace id has no `turns` row, so `store.get_turn` returns `None`,
   `server.py:1048-1053` returns 404, `api()` throws (`:529-535`), and the
   catch at `:831` paints *"Failed to load turn: API /api/turn/… -> 404"*. The
   pass selector under (a) is not a string swap; it is a signature change to the
   SPA's only turn-loading path — exactly the untested-JavaScript cost (a)
   charges to (b).
2. **It breaks the one query that joins spans back to turns, silently.**
   `list_turns(command_name=?)` filters
   `turn_key IN (SELECT trace_id FROM spans WHERE command_name=?)`
   (`observability_store.py:1034-1037`). Under (a) a distilled turn's
   `fw.command.execute` spans live under `<turn_key>~teacher`/`~student`, so
   filtering the debug rail by command **omits distillation turns with no
   error**. Under the adopted design those spans keep `trace_id = turn_key`, so
   the filter is correct unchanged — and a replay trace simply fails to match,
   which is right, because a replay is not a turn the rail should list.
3. **It makes shipped, agent-facing documentation false.**
   `skills_for_coding_fastworkflows/debug-workflow-conversations/reference.md:37`
   (`-- logical turn key = spans.trace_id`), `:53` (`-- trace_id = turn_key`),
   `:83` (`trace_id = turn_key links every span to its turn`), and the recipe
   block at `:228-245` that *emits* `trace_id` as output; plus
   `.claude/skills/fastworkflow-diagnostics-and-tooling/scripts/trace_turn.py:16`
   and its `SELECT * FROM spans WHERE trace_id=?` at `:243-246`. An agent
   following any of these against a distilled DB gets a near-empty result and
   **no error**. A plausible wrong answer is worse than a loud failure, and
   requirement 8 of the gate is "answer from documented SQL". Under the adopted
   design every one of those lines stays true, and the delta an agent must learn
   is one column name.
4. **It leaves the live turn's trace near-empty, which is a new confusing
   state.** `buildTurnTree` still finds the `fw.turn` root (`:1375`) and paints a
   turn card with no work in it — and unless the extractor is *also* moved to a
   derived id, the only spans left in the live trace are the two extractors'
   `fw.llm.call` rows (the `@observe_dspy_calls` decorator sits on
   `_execute_message`, `workflow_execution_context.py:1069`, outside any pass
   scope). A turn whose waterfall is entirely the debugging machinery is worse
   than today's undifferentiated mess.
5. **It cannot express the merged view, which is genuinely wanted.**
   Teacher-vs-student cost and latency rollups (`fix-sb8.10`) and "what did this
   whole message cost" are one-trace questions. Under (a) they are a join across
   two synthetic ids; under the adopted design they are `GROUP BY
   distillation_pass` on one indexed trace.

What (a) is right about, and what is taken: separation you can assert in Python
rather than trust in JavaScript. §3.6 recovers that at the fetch boundary and in
the test contract, without minting a second identity for the live turn.

### 3.3 Why (c) — real per-pass logical turn keys — loses

(c) is the most honest-looking option and the most expensive. It is rejected on
four independent grounds, any one of which is sufficient.

1. **It corrupts conversation memory, silently.**
   `_USABLE_TURN_FILTER = "status IN ('completed','failed') AND
   conversation_summary IS NOT NULL"` (`observability_store.py:776-778`) gates
   **six** reads: `count_usable_turns` (`:780`), `get_memory_window` (`:789`),
   `conversation_summaries` (`:820`), `conversation_label_state` (`:834`),
   `get_last_completed_turn_key` (`:859`), and `list_conversation_summaries`
   (`:875`). `_run_agent_pass` calls `summarize_and_record_turn`
   (`distillation.py:567`), which grows the history, so `_turn_memory_entry`
   (`workflow_execution_context.py:1164-1189`) would stamp
   `conversation_summary` on every pass row and admit all of them. That is
   memory corruption and feedback mis-keying, not a display bug, and nothing in
   the code prevents it — it requires a hand-written forced-NULL and a dedicated
   test.
2. **There is no clean answer for `ordinal`, only three compromises.**
   `_assign_ordinal` (`:729-752`) fires for any first insert with a
   `conversation_id` and no ordinal. Give pass rows `conversation_id=NULL` and
   they surface in the SPA's conversation-less group (`index.html:710-717`) and
   become independently deletable by `prune(include_conversationless_turns=True)`
   (`:1144-1164`); let them take ordinals and turn #4 becomes #6, corrupting the
   sequence `idx_turns_conv`, the rail's nesting, and `continuation_of` depend
   on; pre-reserve the parent's ordinal and three rows all read "#4". And the
   damage reaches the **user-facing chat transcript**, not only the debug rail:
   `index.html:2490-2510` fetches `/api/turns`, filters on
   `conversation_id`, and replays it — a user reopening the chatbot would see
   three exchanges per message they sent.
3. **It permanently inflates the one table retention cannot reach.** `prune`
   exempts turn records by design (`:1100-1102`), and `record_json` holds the
   full serialized `TurnResult` including every `command_outputs` (`:212`,
   `:290`). `fix-sb8.11` exists to *sweep the corpus*, so under (c) each replay
   is another permanent, unprunable, user-visible row — while the size cap keeps
   evicting the spans the insights actually cite.
4. **It makes `_begin_turn` re-entrant, and the failure path is live on a fresh
   install.** `_begin_turn` writes 14 fields and clears `_trace_span_stack`
   (`workflow_execution_context.py:337-388`). None of `LLM_TEACHER_AGENT`,
   `LLM_STUDENT_AGENT`, `LLM_DISTILLATION` exist in
   `fastworkflow/examples/fastworkflow.env`; `dspy_utils.get_lm` raises
   (`utils/dspy_utils.py:42-46`); and the teacher-pass raise is **uncaught** (the
   `try/except` at `distillation.py:864` covers only the student). A leaked
   override leaves the WEC holding a pass key, and
   `finalize_turn_for_observability` (`:1315-1338`) then writes the parent's
   record under it — corrupting both rows with no error, on turn one.

(c) does fix three real bugs as a side effect (§3.7); those are fixed directly
instead.

### 3.4 Why the hybrid, and why it is one rule rather than two

Two things need separating, and they are not the same thing.

- A **pass of a live turn** is part of a real user message. It has a `turns`
  row, an `answer` the user saw, a conversation ordinal, and artifacts keyed to
  it. Splitting its spans off the turn is what breaks §3.2's readers.
- A **replay** is not a turn. Nobody sent it, it has no answer, it must not
  consume an ordinal, and it must not write into the original trace.

Option (b) alone has nowhere to put the second. Its own advocate's escape —
"a replay runs as a normal turn and mints its own `turn_key`" — is option (c)
through the back door and strictly worse than (c) at it, because (b) has no
`parent_turn_key` column with which to hide those rows from the rail. And
writing replay spans into the original trace is not merely untidy: the
deterministic ids `root_span_id(turn_key)` and
`deterministic_span_id(turn_key, 'fw.ask_user', attempt)` are pure functions of
the turn key (`tracing.py:206-213`), so a re-run regenerates the *same* span
ids, and `ON CONFLICT(span_id) DO UPDATE` rewrites `end_ns`, `status` and
`attributes` from `excluded` whenever `excluded.end_ns IS NOT NULL`
(`observability_store.py:589-601`) — **mutating, as a successful write, the very
evidence the pin exists to protect.** `[DR4]`

So the derived-id namespace is not a second identity scheme bolted on; it is the
answer to "what is the trace id when there is no turn key to be". Its domain is
exactly one case, stated normatively:

> `[DR5]` A `~` suffix may be minted **only** by the counterfactual-replay path,
> only as `<original_turn_key>~replay.<n>` where `<n>` is
> `1 + (SELECT COUNT(*) FROM distillation_runs WHERE replay_of = :run_id)`
> computed inside the same `BEGIN IMMEDIATE` that inserts the replay's run row,
> and only for a run whose `comparable = 1`. No other code path constructs a
> `~` trace id. A pass of a live turn never gets one.

The suffix is minted transactionally because observability writes are
best-effort (`[R14]`; the record queue can drop, `observability_store.py:63-66`),
so a suffix collision would otherwise surface as a silently missing pass row.

**How a `~` trace id is actually produced. `[DR41]`** Revision 1 stated the
namespace and never said what writes into it — and the only mechanism the code
offers is overriding `current_turn_key`, which is precisely the corruption §3.3
item 4 uses to reject option (c). Verified: `start_span` has no `trace_id`
parameter (`tracing.py:302-320`), hardcodes `turn_key = get_turn_key(host)`
(`:329`) and `trace_id = turn_key` (`:342`); `get_turn_key` resolves
`current_turn_key` (`:270-271`) → `self._turn_key`
(`workflow_execution_context.py:305-307`), written only by `_begin_turn`
(`:337`). Normatively:

> `[DR41]` The host gains a **second, independent** attribute
> `current_replay_trace_id` (`WorkflowExecutionContext._replay_trace_id`,
> default `None`) and `tracing.get_replay_trace_id(host)` beside
> `get_turn_key`. In `start_span`, `_close_ask_user_span` and every direct
> `Span(...)` construction, the trace id is
> `get_replay_trace_id(host) or get_turn_key(host)`, the decline guard becomes
> "neither is set", and the default root parent is `root_span_id(<that same
> id>)`. `current_turn_key` is **never** overridden — the WEC's `_turn_key`
> stays exactly what `_begin_turn` put there.
>
> The replay driver runs on a WEC with `_turn_key is None`, sets
> `_replay_trace_id` through a context manager whose `finally` clears it (the
> failure path included), and **never calls `_begin_turn` or
> `finalize_turn_for_observability`**. That is safe by construction:
> `finalize_turn_for_observability` short-circuits on
> `self._turn_key is None` (`workflow_execution_context.py:1327`), so a leaked
> replay id cannot write a `turns` row under a `~` key even if the context
> manager is bypassed.
>
> `fix-sb8.14` asserts both halves:
> `SELECT COUNT(*) FROM turns WHERE turn_key LIKE '%~%'` is 0, and every span
> of a replay run has `instr(trace_id,'~') > 0` while every span of a live
> distilled turn has `instr(trace_id,'~') = 0`.

This is added to §9's producer-side change list as items 10 and 11. Without it
`fix-sb8.11` has no way to write a span at all, and the reviewer who found this
was right that revision 1 shipped a namespace with no door into it.

### 3.5 Every reader of the `trace_id == turn_key` invariant, and its fate

| # | Reader | Site | Fate |
|---|---|---|---|
| 1 | Span identity contract | `tracing.py:141` (`trace_id: str  # = turn_key`) | **Amend the comment** to `# = turn_key, or <turn_key>~replay.<n> for a stored-trace replay [DR1]` |
| 2 | Orphan parenting | `tracing.py:337-338` (`parent_span_id = root_span_id(turn_key)`) | **No change.** Live turns keep one root. A replay's own root is minted from its derived id by the replay driver, which passes `span_id`/`parent_span_id` explicitly |
| 3 | Turn-list command filter | `observability_store.py:1034-1037` | **No change, and this is load-bearing.** A distilled turn's `fw.command.execute` spans keep `trace_id = turn_key`, so the filter matches. A replay trace does not match, which is correct — a replay is not a turn to list. Regression test required (`fix-sb8.14`) |
| 4 | Conversation-less prune | `observability_store.py:1158` (`DELETE FROM spans WHERE trace_id=?`) | **BREAKS for replays** — leaves them as unreachable rows nothing will ever delete except size-cap eviction. **Fix:** `DELETE FROM spans WHERE trace_id=? OR trace_id LIKE ? ESCAPE '\'` with `(key, key + '~%')` |
| 5 | Erasure `[R21]` | `observability_store.py:1185-1216` (`forget_channel`), `:1218-1233` (`clear_conversations`) | **CHANGES REQUIRED — revision 1 was wrong here.** The *span* reasoning stands: the `channel_id=?` arm covers replay spans, because `start_span` stamps `channel_id=get_channel_id(host)` (`tracing.py:344`) and CLI mode always binds one (`chat_session.py:126-128`). But both functions are **hardcoded five-table lists**, so the six new tables — which hold `user_message`, `entry_inputs_json`, `param_diff_json`, `detail_json` and insight text — survive a channel erasure and survive the SPA's "clear all conversations". `[DR44]` fixes both; §19 asserts it |
| 6 | Retention arms | `observability_store.py:1122-1129`, `:1167-1180` | **Both need the §10 pin predicate.** Required under every option, not just this one |
| 7 | Chatbot span fetch | `server.py:1059-1067`, `index.html:822-837` | **No change for turns** — `selectTurn` keeps issuing both fetches on one string. Replay traces are reached only via `/api/distillation/*` + `/api/spans/<replay_trace_id>`, never through `selectTurn` |
| 8 | FastAPI trace replay | `run_fastapi_mcp/__main__.py:1148-1211` | **Behaviourally unaffected** — distillation is unreachable from Topology B (§2) and replay is CLI-only. The docstring at `:1155-1157` asserts the invariant as fact and gains one sentence naming the CLI-only exception (`fix-sb8.12`) |
| 9 | Shipped agent SQL | `reference.md:37,53,74,83,210,228-245,248,261` | **Stay true.** Append one section documenting `distillation_pass`, the `base_turn_key` generalization, and the recipes in §15 (`fix-sb8.12`). Revision 2 extends the citation list: `:74` (index note), `:210` (`get_spans` parameter named `trace_id`, turn key passed positionally), `:248` and `:261` (`WHERE name='fw.turn' AND trace_id=:turn_key`) all assert the invariant too |
| 10 | Diagnostics script | `.claude/skills/fastworkflow-diagnostics-and-tooling/scripts/trace_turn.py:16,243-246` | **Keeps working.** On a distilled turn it returns *both* passes, which is correct for a whole-turn tool. Optionally print the pass column (`fix-sb8.12`) |
| 10a | Diagnostics skill prose | `.claude/skills/fastworkflow-diagnostics-and-tooling/SKILL.md:132` (`spans` (`trace_id` == `turn_key`)) | **Amend the parenthetical** to name the replay exception, per row 9's wording. Missed in revision 1 even though row 10 cites a script in the same skill directory (`fix-sb8.12`) |
| 11 | `ReadOnlyObservabilityStore` | `observability_store.py:1237-1267` | **NEW HAZARD.** It never runs `_SCHEMA_STATEMENTS` or the ALTER block (`:390-402`) and connects `mode=ro`, so on a DB the new writer has not opened, any read projecting `distillation_pass` raises `no such column`. §11 mandates feature detection |
| 12 | `deterministic_span_id` | `tracing.py:204-213` | **Needs an optional `pass_label`** (§3.7 item 3) |
| 13 | `SQLiteTraceSink.emit_span` | `observability_store.py:1333-1345` | **Field-by-field copy.** Omitting the new field makes the whole feature a silent no-op with no exception, no log line, and no dropped-span counter. Dedicated test required |
| 14 | "`trace_id` is 1:1 with turns" (implicit index assumption) | `idx_spans_trace`, `:341` | **Now 1:N** (a turn plus its replays). Immaterial at this scale, but recorded so future index tuning does not assume otherwise |

### 3.6 Buying back structural separation without a second identity

(a)'s one unanswerable argument is that separation-by-convention fails silently:
any reader that forgets `AND distillation_pass = ?` gets the merged 2× trace
with no error. Three mechanisms answer it, and they are normative.

**(i) Separation at the fetch boundary, as a GET query parameter. `[DR6]`**
`GET /api/spans/<turn_key>?pass=<label>` and
`ObservabilityStore.get_spans(trace_id, distillation_pass=None)`. A query
parameter, not a path grammar, so `selectTurn` keeps calling `/api/turn/<k>` and
`/api/spans/<k>` with the same string and never hits (a)'s 404. `fix-sb8.7`'s
pass selector becomes `selectTurn(turnKey, passLabel)` with one extra param.
This is a GET, so the four-path POST allowlist (`server.py:868-875`) and
`tests/test_run_chatbot_server.py:1005` are untouched.

**(ii) A pass-filtered response omits the `fw.turn` root *and the distillation
wrapper spans*. `[DR7]`** This is the answer to (b)'s real remaining weakness —
that after the SPA edits, both passes still lay out against one shared
`[t0,t1]` window (`index.html:1658-1665`) and pass 2 occupies the right half of
the chart. With the root omitted, `buildTurnTree` falls through to
`mergeExtents(children.map(c => c.extent))` (`index.html:1420-1422`) and **each
pass gets its own time window**. Verified against the file.

**Revision 2 extends this, because the omission list was incomplete and the
omission mattered more than revision 1 realized.** Under §8's hierarchy
(`fw.distill.run` parents `fw.distill.pass` parents the pass's work), a
response that keeps the wrappers gives `buildTurnTree` a `top` array of exactly
one span — every other span has a recorded non-root parent (`index.html:1377-1387`) —
so the phase builders, which run over `top` alone (`:1390-1417`), never fire,
and `buildOtherNode` hands back that single child unwrapped (`:1352-1357`).
Planning/Execution grouping would vanish for exactly the turns this epic
exists to study. Verified against the file. Normatively:

> `[DR7]` `GET /api/spans/<turn_key>?pass=<label>` returns the spans whose
> `distillation_pass = <label>`, **minus** the `fw.turn` root, the
> `fw.distill.run` span, and the `fw.distill.pass` span for that label. The
> pass's own `fw.planner.*` / `fw.agent.*` / `fw.command.execute` /
> `fw.ask_user` spans therefore have an unresolvable parent, land in `top`, and
> the existing phase builders run unchanged.
>
> The unfiltered response (`/api/spans/<turn_key>` with no `?pass=`) is
> **not** phase-grouped for a distilled turn, and that is the intended
> behaviour: it renders `turn → fw.distill.run → {fw.distill.pass ×N,
> fw.distill.compare, fw.distill.extract ×2}`, i.e. the structure of the run,
> with each pass's phases appearing one drill-down level in. §18's `fix-kw7.11`
> note is corrected accordingly.
>
> **Pass labels on the wrapper spans**, because the two answers give opposite
> results and revision 1 named neither: `fw.distill.run` and
> `fw.distill.compare` carry `distillation_pass = NULL` (they are run-level);
> `fw.distill.pass` carries its own label; `fw.distill.extract` and its child
> `fw.llm.call` spans carry `'extractor'`.

**(iii) The separation assertion is a Python test, not a code reading. `[DR8]`**
`fix-sb8.14` asserts, on a real distilled turn:
```python
t = {s["span_id"] for s in store.get_spans(tk, distillation_pass="teacher")}
s = {s["span_id"] for s in store.get_spans(tk, distillation_pass="student")}
assert t and s and not (t & s)
assert all(sp["parent_span_id"] != root_span_id(tk) for sp in
           store.get_spans(tk, distillation_pass="student"))
```
plus a parenting assertion that every pass span descends from its own
`fw.distill.pass` span (§9). That is (a)'s claimed advantage, recovered in the
language the repo can actually test in.

**This assertion is only true because of `[DR51]`.** As written in revision 1 it
would have failed on any distilled turn containing an `ask_user`: both the open
(`workflow_execution_context.py:419-428`) and the close
(`_close_ask_user_span`, `:461-486`) hardcode
`parent_span_id=tracing.root_span_id(self._turn_key)` and bypass the ambient
stack, and the close *rebuilds* the `Span` from pure functions of the turn key,
so it cannot see a randomly minted pass span id. Verified. `[DR51]` gives
`fw.distill.pass` a **deterministic** id — `deterministic_span_id(turn_key,
'fw.distill.pass', seq, pass_label=<label>)` — so both ask_user sites can
*compute* the parent: `parent_span_id = distill_pass_span_id(turn_key, label)`
when a pass label is active, falling back to `root_span_id(turn_key)` when it is
NULL (today's behaviour, byte-identical). This matters at open, not close:
`parent_span_id` is **not** in the `ON CONFLICT DO UPDATE` set
(`observability_store.py:589-601`), so a wrong parent at open is unfixable.

### 3.7 What this design must fix alongside, because it does not fix itself

Adopting (b) leaves three live bugs that (c) would have dissolved. They are in
scope for this epic and are assigned, because leaving them makes the new UI
*look* worse: the SPA would show student answer text on the turn card directly
above a correctly pass-partitioned waterfall.

1. **`turns.answer` holds the wrong text.** The distillation branch returns at
   `workflow_execution_context.py:1650` **before** `self._turn_agent_result =
   agent_result` at `:1652`, so `_build_turn_result` falls into
   `elif self._turn_outputs:` (`:1118-1127`) and overwrites the teacher's answer
   — the text the user actually saw — with the **student pass's last raw command
   response**. `[DR9]` `distill_message` sets `_turn_agent_result` to the
   teacher's result before returning. (Note in passing: the call-site comment at
   `:1639-1641` says distillation "returns the student's CommandOutput"; it
   returns `teacher_output` on both paths, `distillation.py:867,917-921`. Fix the
   comment.)
2. **`command_outputs` is a fiction.** `_turn_outputs` is reset only in
   `_begin_turn` (`:334`), so the record concatenates both passes' executions as
   one trajectory (`:1132`). `[DR10]` `_run_agent_pass` snapshots and restores
   `_turn_outputs` around each pass, and the turn record keeps the **teacher's**
   outputs — the pass-level detail lives in the pass's spans, which is where the
   UI reads it.
3. **The `fw.ask_user` span-id collision is a tripwire armed by `[DR10]`.**
   `attempt = sum(1 for o in self._turn_outputs if o.command_name == "ask_user")`
   (`:405-407`) is unique across passes today only *because* `_turn_outputs` is
   never reset — teacher gets 0, student gets 1. The moment `[DR10]` resets it,
   both passes mint `attempt=0`, the ids collide, and the student's ask_user span
   upserts over the teacher's, destroying one pass's human-wait evidence with no
   error. `[DR11]` `deterministic_span_id(turn_key, span_name, attempt,
   pass_label=None)` folds `pass_label` into the digest **only when non-None**,
   so `f"{turn_key}|{span_name}|{attempt}"` stays byte-identical for every
   existing caller and `root_span_id` is unchanged
   (`tests/test_tracing_phase1.py:185-192,232,391-394,421-422,466,654` stay
   green). `[DR11]` is a **hard prerequisite** for `[DR10]`; they ship in one
   change or not at all.

4. **The turn record mixes provenances, and the half revision 1 missed is the
   half that feeds memory.** `[DR9]`/`[DR10]` make `answer` and
   `command_outputs` the teacher's. But `conversation_summary` /
   `conversation_traces` come from `_turn_memory_entry`
   (`workflow_execution_context.py:1163-1186`), which reads `messages[-1]` — and
   by then the teacher's history entry has been truncated away by the
   pre-student `restore_workflow_state(initial_snapshot)` (`distillation.py:852`,
   truncation at `:226-229`) and the **student's** `summarize_and_record_turn`
   (`:567`) is the newest entry; the post-divergence restore at `:900` truncates
   to the same length and is a no-op. That column gates `_USABLE_TURN_FILTER`
   (`observability_store.py:776-778`) — the same six consumers §3.3 item 1 uses
   to kill option (c). So adopting (b) without this fix feeds conversation
   memory a trajectory whose answer the user never saw. Separately,
   `_turn_refined_message` is assigned only in `_run_agent` (`:1551`);
   `_run_agent_pass` calls `_refine_user_query` directly
   (`distillation.py:505-507`) without storing it, so
   `turns.refined_user_message` is NULL on every distilled turn.
   `[DR42]` `distill_message` stamps the **teacher's** memory entry and the
   teacher pass's refined message onto the turn result before finalize: the
   teacher's `summarize_and_record_turn` output is captured as a value at the
   end of the teacher pass (before the pre-student restore discards it) and
   re-appended to the conversation history when the teacher's state is restored,
   and `_run_agent_pass` sets `_turn_refined_message` from its own
   `_refine_user_query` result. On the no-divergence branch — where the
   student's state is deliberately kept (`distillation.py:899-901`) — the
   student's summary stays, which is correct: the two passes agreed.
   `distillation_passes` records each pass's own summary hash, so the run row
   can always say which pass each turn-level field came from, and `fix-sb8.14`
   asserts that a distilled turn's `conversation_summary` matches the pass the
   run row names.

Two further sites construct `tracing.Span(...)` directly, bypassing
`start_span`: `_close_ask_user_span` (`workflow_execution_context.py:461-491`,
`trace_id` at `:476`) and `_finalize_turn_trace` (`:1256-1270`). Both keep
`trace_id = self._turn_key` — correct under this design — and
`_close_ask_user_span` must additionally set `distillation_pass` and use the
same `pass_label` in its id as the open, or the close silently fails to match
(it cannot corrupt the row: `trace_id` is not in the `DO UPDATE` set, §2). The
turn-root close keeps `distillation_pass` NULL, which is correct: the root is
turn-level.

---

## 4. N-pass capability (requirement 2) `[DR12]`

**The schema is N-pass-capable; we ship 2.**

`spans.distillation_pass` is free `TEXT` with no `CHECK` constraint, so growing
from two passes to N is entirely a matter of what strings the producer writes:
`'teacher'`, `'student.a'`, `'student.b'`, … plus the orthogonal `'extractor'`
and `NULL`. No DDL change, no index change, no route change, no SPA data-model
change; the pass selector enumerates
`SELECT pass_label, role, seq FROM distillation_passes WHERE run_id=? ORDER BY seq`
and renders whatever comes back through the same filter.

The two places N-pass capability has to be *designed in* rather than fall out:

1. **`distillation_passes` is a child table, one row per pass, PK
   `(run_id, pass_label)`.** Adding a student adds a row, never a column. Every
   per-pass fact — role, model identity, fingerprints, timings, token/cost
   rollups, cache hits — lives there.
2. **Divergence records are pairwise and carry both sides' pass labels.**
   `divergences(run_id, left_pass, right_pass, …)` rather than an implicit
   teacher-vs-student pair. N passes yield N−1 comparisons (or N(N−1)/2 if a
   future sweep wants all pairs) as *rows*, not as a schema break. Shipping 2
   means `left_pass='teacher'`, `right_pass='student'` on every row today; the
   columns exist anyway, because retrofitting them later would require
   rewriting every stored record.

Two honest limits at large N, both fixable without schema change and both
recorded so nobody is surprised: `/api/spans/<turn_key>` has no `LIMIT`
(`observability_store.py:978-982`), so an 8-student sweep needs `[DR6]`'s
`?pass=` filter to be *built*, not assumed; and `buildTurnTree` will produce N
top-level phase groups per turn, which is correct but wants the pass selector
rather than one long scroll.

---

## 5. Boundary with `fix-35m.3` / EXP-013 (requirement 3) `[DR13]`

`fix-35m.3` ("Distillation isolation") and `fix-sb8` both use the phrase
"evidence streams". They are different things and must not be built twice.

| Concern | Owner | Deliverable |
|---|---|---|
| Read-only surface check before the teacher pass (refuse to start if an enabled command is not proven `read_only` unless an isolated fake backend is configured) | **35m.3** | A precondition that raises before any pass runs |
| Distinct distillation invocation path (distillation never reuses the live agent's invocation entry point) | **35m.3** | Code structure |
| Separate accumulators — **action log and conversation history** | **35m.3** | Runtime containment |
| Separate accumulator — **`_turn_outputs`** | **sb8**, per `[DR10]` + `[DR11]`, which ship together | Runtime containment *and* span identity |
| Separate budgets and deadlines; extraction has its own deadline phase and cannot kill the shared worker | **35m.3** | Deadline/budget plumbing |
| The *guarantee* that the two passes were isolated | **35m.3** | Behaviour |
| The *observable evidence* that they were — fingerprints, restore assertion, comparability flag | **sb8.3** | Recorded data |
| Every table, span, route, and UI in this document | **sb8** | Recording and review |

**The pass-entry state fingerprint is the seam, and it is computed once.**
`fix-sb8.3`'s fingerprint is the *observable counterpart* of 35m.3's isolation
guarantee: 35m.3 makes isolation true, sb8.3 makes it checkable. Normatively:

> `[DR13]` `fastworkflow/distillation.py` exports exactly one implementation,
> `state_fingerprint(chat_session) -> str` (§6.1). It is called **once per pass
> boundary** — at each pass entry and each pass exit — by the sb8.3 recording
> path, and the value is written to `distillation_passes.entry_fingerprint` /
> `exit_fingerprint`. `fix-35m.3` **consumes nothing from sb8.3 for its
> read-only surface check** — that check is a property of the enabled command
> surface, not of workflow state, and no state hash can answer it (revision 1
> said otherwise; `read_only` does not exist in the codebase at all —
> `grep -rn read_only --include=*.py fastworkflow/` returns nothing, so 35m.3
> is defining that property from scratch). What 35m.3 **may** use
> `state_fingerprint` for is the *post*-condition: comparing the value at
> teacher entry and teacher exit to assert the teacher mutated nothing. The
> only binding constraint is that it must not define a second implementation.
> The function name and the return contract in §6.1 are fixed by this document
> either way, so the two children can land in either order.

**Normative, and it is a data-loss guard rather than a tidiness rule:**
`fix-35m.3` must **not** reset or re-scope `_turn_outputs`. That accumulator
drives the `fw.ask_user` deterministic-id `attempt` counter
(`workflow_execution_context.py:405-407`), which is unique across passes today
only *because* it is never reset. Resetting it without `[DR11]`'s `pass_label`
in the digest collides the two passes' ask_user span ids and destroys one pass's
human-wait evidence as a **successful upsert**, with no exception and no
counter. `_turn_outputs` containment therefore belongs to sb8, bound to
`[DR11]`, and 35m.3 owns the other two accumulators.

Concretely: 35m.3 owns *whether* the student started clean; sb8 owns *the row
that says so*, the banner that shows it, and the rule that a run whose
fingerprints differ is excluded from every aggregate.

One prerequisite that belongs to **sb8.3, not 35m.3**, because it is a
measurement bug rather than an isolation bug: `snapshot_workflow_state`
(`distillation.py:197-207`) must take a `copy.deepcopy` of both context dicts.
`Workflow._to_dict()` returns `self._context` **by reference**
(`workflow.py:450`), so today the "snapshot" aliases the live dict and any
fingerprint computed from it would be comparing an object against itself and
reporting agreement by construction. Without the deep copy, `entry_fingerprint`
is not merely imprecise — it is guaranteed to be equal and therefore worthless.

---

## 6. The comparability record (requirement: comparability proof) — `fix-sb8.3`

### 6.1 The fingerprints `[DR14]` `[DR47]`

**Revision 2 splits this into two projections.** Revision 1 defined one hash
covering context *and* a conversation-history tail and used it for four
different jobs. Two independent reviewers showed that the history component
makes two of those jobs degenerate, and a third showed the hash was not a
function of state at all. Both are fixed below rather than annotated.

```python
def state_fingerprint(chat_session) -> str:
    """Stable hash of pass-entry (or pass-exit) WORLD state. [DR14][DR47].

    Deliberately excludes conversation history: every pass appends its own
    LLM-generated summary before it exits (distillation.py:567), so a
    history-bearing hash can never compare equal across passes.
    """
    workflow = chat_session.get_active_workflow()
    cme = chat_session.cme_workflow
    payload = {
        "v": 2,
        "workflow_context": _canonical(workflow._context),
        "cme_context": _canonical(cme._context),
        "command_context": workflow.current_command_context_name,
        "is_complete": bool(workflow._is_complete),
    }
    return _digest(payload)


def prompt_fingerprint(chat_session, *, history_bound: int) -> str:
    """Hash of the inputs a pass's prompts actually see. [DR47].

    `history_bound` is the length of conversation_history.messages captured at
    the ENTRY of the pass being measured; entry and exit are always computed at
    the same bound, so a pass's own appended summary is never inside its own
    exit hash.
    """
    msgs = chat_session.conversation_history.messages[:history_bound]
    payload = {
        "v": 2,
        "history_tail": [_canonical(m) for m in msgs[-4:]],
        "refined_user_message": chat_session_refined_message(chat_session),
    }
    return _digest(payload)


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]
```

**`_digest` has no `default=`, and that is the point. `[DR47]`** Revision 1
finished with `json.dumps(..., default=str)`. Verified: `cme._context` always
contains a live `Workflow` object — `workflow_execution_context.py:1008` sets
`self._cme_workflow.context["app_workflow"] = workflow`, and
`_workflows/command_metadata_extraction/intent_detection.py:34` reads it back as
an object. `fastworkflow/workflow.py` defines no `__repr__`/`__str__`, so
`default=str` renders it as `<fastworkflow.workflow.Workflow object at 0x…>` —
**a heap address**. The hash was therefore not a function of state, was unstable
across processes, and would have made `fix-sb8.11`'s replay gate reject every
replay, 100% of the time, for a reason nobody would have guessed from the
symptom. `_canonical` closes it:

`_canonical` sorts keys, renders numbers as decimal strings, renders datetimes
as RFC-3339 UTC, preserves list order, **replaces any value that is not a JSON
scalar / list / dict with the structural token `"<type:" + type(v).__name__ +
">"`** (never with its `repr`, never with `str`), and **drops** four classes of
key, each for a stated reason:

- `app_workflow` (and any other live-object handle the CME context carries) —
  an object identity, not state; its contents are already hashed directly as
  `workflow_context`. Named explicitly so the structural-token rule is not the
  only thing standing between the fingerprint and a heap address.
- `raw_user_message` and `is_user_command` — written by the pass itself at entry
  (`distillation.py:481`, `workflow_agent.py:545-547`), so including them makes
  every pass differ from every other by construction.
- Any key matching `_ts$|_at$|^started_|^updated_` — wall clock.
- `stored_parameters` and `NLU_Pipeline_Stage` — mid-pipeline scratch that is
  empty at a pass boundary and would only ever add noise.

`history_tail` stays bounded at the last **4** messages, for the reason revision
1 gave (the tail that reaches the planner and agent prompts is bounded), but it
now lives only in `prompt_fingerprint`, and only under an explicit
`history_bound`.

**Which fingerprint gates what, normatively `[DR47]`:**

| Consumer | Fingerprint | Why |
|---|---|---|
| `comparable` (`[DR15]`) | `state_fingerprint` at entry | "did both passes start in the same world" |
| `restore_ok_pre_student` / `restore_ok_post_compare` (§6.2) | `state_fingerprint` | the restore restores context dicts; it provably cannot restore history (`distillation.py:226-229` only truncates) |
| materiality (`[DR20]`) | `state_fingerprint` at exit | "did the two paths reach the same end state" |
| replay entry gate (§14) | `state_fingerprint` **and** `prompt_fingerprint` | a replay must reproduce both the world and the prompt inputs |

`fix-sb8.14` asserts, as a normative test, that `state_fingerprint` over the
same reconstructed state is **byte-equal across two processes**. Without that
test the whole comparability and replay chain is untestable, and revision 1's
address bug would have shipped green.

### 6.2 What is recorded and what it gates

Per pass, on `distillation_passes`: `entry_fingerprint`, `exit_fingerprint`,
`agent_model`, `planner_model`, `model_params_json` (the temperature /
`max_tokens` / `top_p` resolved by `dspy_utils.get_lm`), `wall_ms`, `tokens`,
`cost_usd`, `cache_hits`, `cache_misses`.

Per run, on `distillation_runs`:

- `comparable INTEGER NOT NULL` — 1 iff every pass's `entry_fingerprint`
  (the `state_fingerprint`, `[DR47]`) is equal. `[DR15]`

  > **What `comparable = 1` does and does not attest, normatively.** It attests
  > that the **context dicts, command context and completion flag** were equal
  > at pass entry. It does **not** attest that the *application world* was
  > equal, and it cannot: `Workflow._to_dict()` returns only `workflow_id`,
  > `workflow_folderpath`, `parent_workflow_id`, `is_complete`,
  > `workflow_context` (`workflow.py:443-451`), so
  > `_root_command_context` / `_current_command_context` /
  > `_command_context_for_response_generation` (`:166-168`) are never captured
  > and never restored — the student runs against entities the teacher may have
  > created, completed or deleted. `current_command_context_name` returns a
  > class name only (`workflow.py:194-196`), not object identity or field
  > values. Revision 1 stated this limit for *replay* (`[DR35]`) and not for the
  > flag that gates every aggregate. That was dishonest by omission and is
  > corrected here. The UI label is **"comparable inputs; application state not
  > verified"**, never a bare "comparable".
- `isolation_verified INTEGER` — `[DR48]`. NULL until `fix-35m.3`'s read-only
  surface check exists; 1 only when that check passed for this run; 0 when it
  ran and failed. sb8 **never** writes 1.
- `restore_ok_pre_student INTEGER` — `[DR15]`. 1 iff
  `restore_workflow_state(initial_snapshot)` (`distillation.py:852`) returned
  without exception **and** the post-restore `state_fingerprint` equals the
  pre-teacher entry `state_fingerprint`.
- `restore_ok_post_compare INTEGER` — 1 iff the divergence-path restore
  (`distillation.py:866` and `:899-900`) returned without exception **and** the
  post-restore `state_fingerprint` equals the **teacher's exit**
  `state_fingerprint`. Revision 1 defined a single `restore_ok` against the
  pre-teacher baseline, which is the wrong baseline for two of the three call
  sites and would have reported 0 on every divergent run. NULL at either column
  when that site did not execute.
- `fingerprint_teacher`, `fingerprint_student` — the entry `state_fingerprint`s,
  denormalized onto the run row so the loud banner and the list-level warning
  need no join.

**The downstream contract, normative.** A run with `comparable = 0`:
1. renders a **loud non-comparable banner** at run level, and a warning marker
   at *list* level in the turn rail (`fix-sb8.7`), stating which fingerprint
   pair differed;
2. has every divergence record written with `material = NULL` (materiality is
   unknowable when the starting states differed), never `0`;
3. is **excluded** from every aggregate in `fix-sb8.10` — every documented
   recipe in §15 carries `AND r.comparable = 1`, and (per `[DR54]`) every
   recipe that counts *independent support* additionally carries
   `AND r.replay_of IS NULL` and joins runs with an **inner** join;
4. is **refused** by counterfactual replay (`fix-sb8.11`), which returns
   `not-replayable: non-comparable origin` rather than a verdict;
5. still produces insights if the extractor ran — the run is recorded, its
   conclusions are quarantined. Deleting the evidence would hide the confound.

**And the ordering constraint honesty requires `[DR48]`.** `fix-sb8.3` may ship
before `fix-35m.3`; it then writes `isolation_verified = NULL` on every run and
the banner says so. What it may **not** do is let a NULL be read as a pass:

- `fix-sb8.10`'s **promotion** view (the support/contradiction counts of §15,
  the query a human or agent uses to decide a rule is real) requires
  `r.isolation_verified = 1`. Until 35m.3 lands, that view is empty and says
  *"application-object isolation is not yet verified by any run; promotion is
  blocked"* — which is true, and is a far better state than a populated table of
  causal claims resting on a flag that cannot see the dominant confound.
- `fix-sb8.11` returns `not-replayable: isolation unverified` when
  `isolation_verified IS NOT 1`.
- Every other surface — the run list, the per-pass waterfall, the aligned diff,
  the divergence records, the raw counts — works on `comparable = 1` alone, so
  sb8 is shippable and useful before 35m.3 exists.

This is deliberately *not* the reviewers' proposal of forcing `comparable = 0`
until 35m.3 lands (§21, objection 3): that would empty every surface, not just
the causal one, and would misuse a flag that has a precise meaning to signal a
different fact.

### 6.3 Per-LLM-call cache hit/miss — a query, not new instrumentation

**Verified against `utils/dspy_logger.py`:** `fw.llm.call` spans already carry
`model` (`:374` — revision 1 cited `:395`, which is
`innermost.spans.append(...)`; corrected), `usage` (`:425`), `cost` (`:426`),
and `cache_hit` (`:429-430`).
No new emitter is needed; the per-pass rollup is a query over existing data,
computed once at run completion and written to `distillation_passes` so the UI
and aggregates never rescan spans:

```sql
SELECT distillation_pass AS pass_label,
       SUM(CASE WHEN json_extract(attributes,'$.cache_hit') THEN 1 ELSE 0 END) AS cache_hits,
       SUM(CASE WHEN json_extract(attributes,'$.cache_hit') THEN 0 ELSE 1 END) AS cache_misses,
       SUM(COALESCE(json_extract(attributes,'$.cost'), 0.0))                   AS cost_usd,
       SUM(COALESCE(json_extract(json_extract(attributes,'$.usage'),
                                 '$.total_tokens'), 0))                        AS tokens
FROM spans
WHERE trace_id = :turn_key AND name = 'fw.llm.call'
  AND distillation_pass IS NOT NULL
GROUP BY distillation_pass;
```

Note the **nested** `json_extract` on `usage`: `dspy_logger` stores it through
`_json_text(...)` (`:425`), so it is a JSON *string* inside the attributes JSON.
Writing the single-level form returns NULL silently, which is exactly the class
of error §15's "verified by execution" rule exists to catch.

A cache hit in one pass and a miss in the other is a silent confound that looks
like a behavioural difference. `[DR16]` a run where
`cache_hits > 0` in one pass and `= 0` in the other is flagged
`cache_asymmetric = 1` on the run row and surfaced in the run header. It does
**not** force `comparable = 0` — a cache hit returns the same completion, so the
trajectory is still comparable; it is a *cost* confound, and the cost columns
are what must not be compared across it.

---

## 7. Divergence taxonomy, alignment, and materiality — `fix-sb8.4`

### 7.1 What is aligned `[DR17]`

**Align over `fw.command.execute` spans, not over `chat_session.action_log`.**
Three reasons, all decisive:

1. Spans are the persisted record; the action log dies with the call.
2. Every span has a `span_id`, which is the global `spans` PRIMARY KEY
   (`observability_store.py:322`) — so a divergence record gets its
   `teacher_span_id` / `student_span_id` provenance link *for free*, with no
   second correlation step.
3. The current filter `is_valid_action` (`distillation.py:333-340`) drops every
   record with a falsy `command_name` — which is **every failed command**
   (`workflow_agent.py:299-305`) — plus every `ask_user` record and every
   `ErrorCorrection/*` record. "The student had to ask the user for something the
   teacher inferred" is currently invisible, and it is one of the most
   informative divergences available. Aligning over spans inherits none of that.

The comparable-unit sequence for a pass is therefore, in `start_ns` order:
every `fw.command.execute` span, plus every `fw.ask_user` span (as a step of its
own kind), under that pass's `fw.distill.pass` span — which for `fw.ask_user`
is true only because `[DR51]` gives that span a deterministic id and reparents
both the ask_user open and `_close_ask_user_span` onto it. Without `[DR51]` the
ask_user spans hang off the turn root and the sentence above is simply false.

**Plan level.** One step per `fw.planner.plan` / `fw.planner.replan` span, and
the step's payload is the **whitespace-normalized full plan string**, not
`prediction.next_steps.split()`. `PlanningStep.generated_plan` is a whitespace
split into individual *words* (`workflow_agent.py:686`), so aligning over it
produces word-level noise, and `compare_planning_traces` today is effectively an
exact string comparison that reports divergence on any wording difference. This
is a change to *what is recorded and compared*, not to any extraction prompt, so
it is in scope.

### 7.2 The canonical step key `[DR18]`

Two keys per step, because alignment must match on command identity while
*detecting* parameter differences:

```
command_key = sha256(f"{command_name}\x1f{context}").hexdigest()[:16]
step_key    = sha256(canonical_json({
                  "command": command_name,
                  "context": context,
                  "params":  canonical(parameters),
              })).hexdigest()[:16]
```

`canonical(parameters)`: keys sorted; numbers as decimal strings; `Decimal` and
`datetime` normalized (RFC-3339 UTC); `None` as JSON null; nested dicts
recursively; **list order preserved**. This fixes a live confound in
`_action_signature`, which uses `json.dumps(..., default=str)`
(`distillation.py:301-306`): today `1` and `"1"` are different signatures while
`datetime(...)` and its `str()` are identical.

For an `fw.ask_user` step, `command_name = "ask_user"` and `params =
{"agent_query": <query>}`.

**The NULL-`command_name` fallback, which revision 1 omitted and which breaks
exactly the case §7.1 sells. `[DR50]`** Two reviewers independently found this
and it is verified: `CommandExecutor._invoke_command_impl` returns early on
`command_output.command_handled` and on `not command_output.success`
(`command_executor.py:116-122`) **before** assigning `command_name` / `context`
(`:158-160`), and `CommandOutput` defaults both to `""`
(`fastworkflow/__init__.py:90,92`); the span close then passes
`command_name=command_output.command_name or None` (`:93`), writing NULL. The
`BaseException` arm (`:57-69`) passes no `command_name`, no `context` and
`attributes={"error_type": …}` only — no `parameters` at all — and
`upsert_span_rows` uses `COALESCE(excluded.command_name, spans.command_name)`
(`observability_store.py:600-601`), so the column stays NULL. Under revision 1
every failed command in a pass collapsed to
`command_key = sha256("None\x1fNone")` and one identical `step_key`, so the LCS
would match *any* teacher failure to *any* student failure and classify the pair
`identical` — the precise opposite of §7.1's claim to rescue failures.

> `[DR50]` When `spans.command_name` IS NULL, the alignment keys are derived
> from `attributes.raw_command`, which **is** recorded at span open
> (`command_executor.py:41-48`, `attributes={"raw_command": command}`) and
> survives the close because `end_span` does `span.attributes.update(...)`
> (`tracing.py:379`) rather than replacing. Normalize it — strip, collapse
> internal whitespace, lowercase the leading verb token — and compute
> `command_key = sha256(f"raw:{normalized_raw}\x1f{context or ''}")`,
> `step_key` over `{"command": f"raw:{normalized_raw}", "context": context,
> "params": canonical(parameters)}`. A span with neither `command_name` nor
> `raw_command` gets `command_key = sha256(f"span:{span_id}")`, i.e. it matches
> nothing — an unmatched step is a visible gap, a falsely matched one is a
> fabricated `identical`.
>
> `fix-sb8.14` asserts that two *different* failed commands in one pass produce
> two different `command_key`s, and that a teacher failure and an unrelated
> student failure do not align as `identical`.

### 7.3 The alignment algorithm `[DR19]`

Sequences are short (bounded by the agent's max iterations, in practice ≤ ~50),
so the plain O(nm) dynamic-programming LCS is correct and fast enough; no
Hunt–Szymanski, no heuristic windowing.

1. **LCS over `command_key`.** Matching on command identity — not on the full
   step key — is what lets "same command, one wrong parameter" come out as *one*
   matched pair instead of two unmatched entries, which is the single worst
   behaviour of today's set difference.
2. **Convert the LCS to an edit script**: matched pairs at their aligned
   positions, plus teacher-only and student-only runs in between.
3. **Classify each matched pair:**
   - `step_key` equal → **`identical`**
   - `step_key` differs, parameter **key sets equal**, values differ →
     **`param-value-only`**
   - `step_key` differs, parameter key sets differ →
     **`same-command-different-params`**
4. **Reordering post-pass** — LCS alone cannot express it. For every
   teacher-only step at index *i* and student-only step at index *j* with an
   **exactly equal `step_key`**, rewrite the pair as one **`reordered`** record
   and remove both unmatched entries. Match greedily by smallest `|i − j|`;
   ties break toward the earlier student index, so the result is deterministic.
5. **Remaining unmatched:** teacher-only → **`missing-in-student`**;
   student-only → **`extra-in-student`**.
6. **Run-level:** if every action-level record is `identical` or `reordered` but
   the two passes' final answers differ, emit one
   **`different-answer-same-actions`** record at `level='run'`, with the two
   `fw.agent.execute` span ids as its span pair.

The full taxonomy, exactly as stored in `divergences.kind`:
`identical` | `same-command-different-params` | `param-value-only` |
`extra-in-student` | `missing-in-student` | `reordered` |
`different-answer-same-actions`.

`identical` records **are stored.** They are what makes the aligned diff
renderable without recomputation, and they are the denominator of every rate in
`fix-sb8.10`. Storage cost is bounded (§10).

### 7.4 Materiality `[DR20]`

> `material = 0` when the run's teacher and student exit **`state_fingerprint`**
> are equal (`[DR47]`) — the same end state reached by a different path is
> **not a mistake** — or when the record's kind is `identical`. `material = 1`
> otherwise. `material = NULL` when the run is non-comparable, because
> materiality is unknowable if the passes did not start from the same place.

**Revision 2 changed which hash this reads, and the change is load-bearing.**
Revision 1 compared the single revision-1 fingerprint, which included
`history_tail`. Every pass calls `summarize_and_record_turn`
(`distillation.py:567`) **inside** the pass, before it returns — an LLM call
(`workflow_execution_context.py:987-1003`, `:1931-1943`) that appends its own
generated summary to `_conversation_history.messages` (`:956-970`). Teacher and
student run different models, so the last history message differs on every run
even when the action sequences are byte-identical. `material` would have been 1
on every non-`identical` record and the entire "same end state, different path"
branch would have been dead code that nobody could have noticed was dead.
`state_fingerprint` excludes history by construction, so the branch is live.
`fix-sb8.14` asserts it directly: two passes taking identical actions produce
equal exit `state_fingerprint`s and therefore `material = 0`.

**Honest limit, stated rather than hidden.** Materiality is a **run-level**
judgement projected onto records. There is no per-step state fingerprint and
this design does not add one: capturing state after every command would require
a deep copy of both context dicts per step, which changes the passes' own cost
profile and would itself become a confound in the cost columns §6.2 records.
The consequence is precise and must be documented in the UI (`fix-sb8.8`): a
material divergence is a divergence *in a run whose end state differed*, not a
divergence proven to have caused it. Non-material divergences are visually
demoted; material ones are not thereby proven causal. Per-step materiality is a
future refinement, not a gap this design pretends to close.

### 7.5 Where the aligner reads, and the barrier it needs `[DR49]`

`[DR17]` rules "align over spans, because spans are the persisted record".
Revision 1 never said *when* those spans are read, and spans reach the table
through an asynchronous, **lossy** path: `SQLiteTraceSink.emit_span`
(`observability_store.py:1330-1351`) ends in
`self._span_queue.put_nowait(...)` / `except queue.Full: self._count("spans_dropped")`
against a bounded 10,000-slot queue (`_DEFAULT_QUEUE_MAX`, `:59`) drained by one
background thread (`:1317`) that takes records first and spans second
(`_next_item`, `:1673-1682`). A teacher span that is merely **late** — never
mind dropped — becomes a fabricated `missing-in-student` divergence, stored as
structured evidence and, under §10.3, pinned. `[DR40]` says silence is never
read as agreement; here silence would be read as a *divergence*, which is worse.

> `[DR49]` The alignment runs over the **in-process `Span` objects** the pass
> emitted, held by the pass and recording their `span_id`s, so it never races
> the writer. Before any divergence row is written, the recording path calls
> `sink.flush()` (`observability_store.py:1530-1538`, which exists and blocks
> until everything enqueued so far is written) as a barrier, and records the
> `writer_health["spans_dropped"]` counter at each pass entry and exit onto
> `distillation_passes.spans_dropped_delta`. **If the counter moved during a
> pass, the run is written `comparable = 0`** with the reason
> `evidence-incomplete`, and its divergences are quarantined by the §6.2
> contract like any other non-comparable run.
>
> `distillation_runs.left_steps` / `right_steps` record the aligner's own
> expected step counts (already listed as `fw.distill.compare` attributes in
> §8), so a later reader can detect a truncated stored sequence by comparing
> them against the rows that survived retention. The DB read of `spans` is then
> purely a **rendering** concern, and a mismatch between the two is visible
> rather than silently absorbed.

### 7.6 The prose summary is rendered from the record

The extractor prompt keeps receiving a prose divergence summary — this epic does
not re-tune extraction — but that string is **rendered from the stored
records**, not computed alongside them. One source of truth for the extractor,
the UI, and the aggregate queries. Better extractor input is a side effect, not
the goal.

---

## 8. Span namespace (requirement 8) `[DR21]`

**Reserved prefix: `fw.distill.*`.** Registered in
`fastworkflow/tracing.py` alongside `V1_SPAN_NAMES`, `AGENT_LOOP_SPAN_NAMES`,
and `RESERVED_V2_SPAN_NAMES` as its own frozenset.

**Not `fw.train.*`.** Two reasons: `fw.train.*` is reserved for *train-time*
metrics (`docs/…studio_design.md` §3.1, `train_runs` table), and distillation is
a run-time surface — filing it under `fw.train.` would put run-time spans in a
namespace whose only documented meaning is "the training pipeline". And
`SPAN_TRAIN_PREFIX = "fw.train."` is stored in `RESERVED_V2_SPAN_NAMES` as a
**prefix, not a name** (`tracing.py:113,117`), so any set-membership test against
that frozenset is already ambiguous for it; adding real emitted names under it
would make that ambiguity load-bearing.

```python
# fastworkflow/tracing.py — beside RESERVED_V2_SPAN_NAMES
SPAN_DISTILL_RUN     = "fw.distill.run"
SPAN_DISTILL_PASS    = "fw.distill.pass"
SPAN_DISTILL_COMPARE = "fw.distill.compare"
SPAN_DISTILL_EXTRACT = "fw.distill.extract"
SPAN_DISTILL_REPLAY  = "fw.distill.replay"

DISTILL_SPAN_NAMES = frozenset({
    SPAN_DISTILL_RUN, SPAN_DISTILL_PASS, SPAN_DISTILL_COMPARE,
    SPAN_DISTILL_EXTRACT, SPAN_DISTILL_REPLAY,
})
```

Every row below also states its `distillation_pass` value, because `[DR7]`'s
pass-filtered response and every `GROUP BY distillation_pass` rollup give
opposite answers depending on it, and revision 1 named none of them.

| Span | Kind | Parent | Stack | `distillation_pass` | Span id | Attributes |
|---|---|---|---|---|---|---|
| `fw.distill.run` | `internal` | `fw.turn` root | yes | `NULL` (run-level) | random | `run_id`, `user_message`, `teacher_agent_model`, `teacher_planner_model`, `student_agent_model`, `student_planner_model`, `comparable`, `comparable_reason`, `isolation_verified`, `restore_ok_pre_student`, `restore_ok_post_compare`, `cache_asymmetric`, `planning_diverged`, `exec_diverged`, `planning_insights`, `execution_insights`, `divergence_counts` (object, kind→count), `material_count` |
| `fw.distill.pass` | `internal` | `fw.distill.run` | yes | its own `pass_label` | **deterministic**, `[DR51]` | `run_id`, `pass_label`, `role`, `seq`, `agent_model`, `planner_model`, `model_params`, `entry_fingerprint`, `exit_fingerprint`, `wall_ms`, `tokens`, `cost_usd`, `cache_hits`, `cache_misses` |
| `fw.distill.compare` | `internal` | `fw.distill.run` | yes | `NULL` (run-level) | random | `run_id`, `level` (`plan` \| `action`), `left_pass`, `right_pass`, `left_steps`, `right_steps`, `matched_pairs`, `divergence_counts`, `material_count`, `algorithm` (`lcs-v1`) |
| `fw.distill.extract` | `internal` | `fw.distill.run` | yes | `'extractor'` | random | `run_id`, `kind` (`planning` \| `execution`), `extractor_model`, `divergence_summary`, `existing_insights_bytes`, `existing_insights_sha256`, `raw_output`, `parsed_count`, `empty_reason` (`extractor-returned-empty` \| `parse-yielded-nothing` \| `null`), `insight_ids` |
| `fw.distill.replay` | `internal` | its own root, in the replay trace | yes | `'student-replay'` | random | `run_id`, `replay_of`, `replay_trace_id`, `injected_insight_ids`, `cited_divergence_ids`, `divergence_removed`, `verdict` |

**`fw.distill.pass` has a deterministic span id. `[DR51]`**
`distill_pass_span_id(turn_key, pass_label) = deterministic_span_id(turn_key,
"fw.distill.pass", seq, pass_label=pass_label)`. It is not an aesthetic choice:
`_close_ask_user_span` (`workflow_execution_context.py:461-486`) deliberately
rebuilds its `Span` from pure functions of the turn key rather than holding it
(docstring at `:434-441`), so it can only ever parent onto something it can
*compute*. A random `uuid4().hex` (`tracing.py:338`) is not computable, and
`parent_span_id` is not in the upsert's `DO UPDATE` set
(`observability_store.py:589-601`), so the open must be right the first time.
The deterministic id is what makes `[DR8]`'s parenting assertion satisfiable at
all.

All five are `KIND_INTERNAL`: they are structure, not LM calls. The actual model
invocations remain `fw.llm.call` spans, which now land as **children** of the
enclosing `fw.distill.pass` or `fw.distill.extract` — which is precisely the
association that does not exist today, and it is what `fix-sb8.5`'s "the
extractor call must be observable" reduces to.

**Why both the hierarchy and the column (D2).** They are not redundant: the
column is the *index* on the hierarchy. Deriving "which pass is this span in"
from parenting requires a recursive CTE in SQL and a tree walk in JavaScript;
deriving it from the column is `WHERE distillation_pass = ?` against a partial
index. And deriving the hierarchy from the column is impossible — the column
cannot say which of a pass's spans is the planner and which is the agent loop.
This is denormalization with a stated justification. The producer writes the
column from the same ambient value that opens the span, so the two cannot
disagree.

**Attribute caps.** `raw_output` and `divergence_summary` on
`fw.distill.extract` ride the standard `FW_OBS_MAX_ATTR_BYTES` = 16 KiB cap with
lossy-and-counted truncation (`tracing.py:133`, `[R10]`). `existing_insights` is
**not** stored on the span — the whole markdown corpus is pasted into the
extractor prompt (`insights_loader.py:55-63`) and grows without bound; the span
stores its byte length and SHA-256 instead, which is what "what did it dedupe
against" actually needs, and the corpus itself is reconstructible from the
insight ledger.

---

## 9. Full DDL

Appended to `_SCHEMA_STATEMENTS` (`observability_store.py:300-345`), matching
the file's style exactly: `CREATE TABLE IF NOT EXISTS`, explicit secondary
indexes as separate statements, no inline `REFERENCES` (the file declares none
and does not enable `PRAGMA foreign_keys`; joins are by convention and are
documented in §15). `[DR22]`

```sql
CREATE TABLE IF NOT EXISTS distillation_runs (
    run_id TEXT PRIMARY KEY,
    turn_key TEXT NOT NULL,               -- == spans.trace_id == turns.turn_key
    channel_id TEXT, conversation_id INTEGER,
    user_message TEXT NOT NULL,
    workflow_name TEXT, entry_context TEXT,
    comparable INTEGER NOT NULL,          -- 0 => NON-COMPARABLE, divergences unusable
    comparable_reason TEXT,               -- 'fingerprint-differs' | 'evidence-incomplete'
                                          --   | 'teacher-raised' | NULL when comparable [DR49]
    isolation_verified INTEGER,           -- NULL until fix-35m.3 exists; sb8 never writes 1 [DR48]
    fingerprint_teacher TEXT, fingerprint_student TEXT,
    restore_ok_pre_student INTEGER,       -- vs the pre-teacher entry state_fingerprint
    restore_ok_post_compare INTEGER,      -- vs the teacher EXIT state_fingerprint [DR15]
    cache_asymmetric INTEGER NOT NULL DEFAULT 0,
    left_steps INTEGER, right_steps INTEGER,   -- aligner's own step counts [DR49]
    planning_diverged INTEGER NOT NULL DEFAULT 0,
    exec_diverged INTEGER NOT NULL DEFAULT 0,
    material_divergences INTEGER NOT NULL DEFAULT 0,
    planning_insights INTEGER NOT NULL DEFAULT 0,
    execution_insights INTEGER NOT NULL DEFAULT 0,
    extractor_empty INTEGER NOT NULL DEFAULT 0,   -- diverged but extracted nothing
    extractor_model TEXT,
    insight_set_json TEXT,                -- insight ids/hashes live at pass entry
    replay_of TEXT,                       -- run_id this run replays, else NULL
    replay_trace_id TEXT,                 -- '<turn_key>~replay.<n>' when a replay
    pinned INTEGER NOT NULL DEFAULT 0,    -- provenance-aware retention [DR26]
    pinned_at TEXT,                       -- when the pin was taken [DR43]
    pinned_span_count INTEGER,            -- live span count at pin time; a later
                                          --   shortfall means an older build pruned it [DR43]
    turn_fields_from TEXT,                -- which pass supplied answer/summary [DR42]
    evidence_pruned INTEGER NOT NULL DEFAULT 0,  -- trace deleted by retention [DR52]
    started_at TEXT, completed_at TEXT,
    run_json TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS distillation_passes (
    run_id TEXT NOT NULL,
    pass_label TEXT NOT NULL,             -- joins spans.distillation_pass verbatim
    role TEXT NOT NULL,                   -- teacher | student | extractor | student-replay
    seq INTEGER NOT NULL,                 -- execution order within the run
    trace_id TEXT NOT NULL,               -- turn_key, or the replay trace id
    agent_model TEXT, planner_model TEXT, model_params_json TEXT,
    entry_fingerprint TEXT, exit_fingerprint TEXT,
    first_span_id TEXT, last_span_id TEXT,
    wall_ms INTEGER, tokens INTEGER, cost_usd REAL,
    cache_hits INTEGER, cache_misses INTEGER,
    entry_prompt_fingerprint TEXT, exit_prompt_fingerprint TEXT,  -- [DR47]
    history_bound INTEGER,                -- len(messages) at pass entry [DR47]
    summary_hash TEXT,                    -- hash of this pass's own turn summary [DR42]
    spans_dropped_delta INTEGER,          -- writer_health delta across the pass [DR49]
    entry_inputs_json TEXT,               -- PROMPT INPUTS, not restorable state [DR45]
    PRIMARY KEY (run_id, pass_label));

CREATE TABLE IF NOT EXISTS distillation_divergences (
    divergence_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    level TEXT NOT NULL,                  -- plan | action | run
    left_pass TEXT NOT NULL,              -- N-pass ready [DR12]
    right_pass TEXT NOT NULL,
    align_index INTEGER NOT NULL,         -- position in the alignment
    kind TEXT NOT NULL,                   -- the [DR19] taxonomy
    material INTEGER,                     -- 1 | 0 | NULL(non-comparable) [DR20]
    replayable INTEGER NOT NULL DEFAULT 1,
    command_key TEXT, command_name TEXT, context TEXT,
    left_step_key TEXT, right_step_key TEXT,
    left_span_id TEXT, right_span_id TEXT,   -- -> spans.span_id (global PK)
    param_diff_json TEXT,                 -- per-key before/after for the UI
    detail_json TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS distillation_insights (
    insight_id TEXT PRIMARY KEY,          -- 'ins-<12 hex>' [DR31]
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,                   -- planning | execution
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,              -- reverse index from a markdown line
    extractor_span_id TEXT,               -- the fw.distill.extract span
    insight_file TEXT,                    -- absolute path appended to
    file_entry_number INTEGER,            -- DISPLAY ONLY — never an identifier
    created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS distillation_insight_citations (
    insight_id TEXT NOT NULL,
    divergence_id TEXT NOT NULL,
    PRIMARY KEY (insight_id, divergence_id));

CREATE TABLE IF NOT EXISTS distillation_verdicts (
    verdict_id TEXT PRIMARY KEY,
    insight_id TEXT NOT NULL,
    verdict TEXT NOT NULL,                -- supported | not-supported-by-cited-evidence
                                          -- | overfit-to-single-turn | duplicate-of-existing
                                          -- | contradicted-by-other-turns
    note TEXT,
    actor TEXT NOT NULL,                  -- 'human' | 'agent:<name>' | 'replay'
    replay_run_id TEXT,                   -- set when a replay produced the verdict
    superseded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL);

CREATE INDEX IF NOT EXISTS idx_spans_trace_pass
    ON spans(trace_id, distillation_pass) WHERE distillation_pass IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_distill_runs_turn
    ON distillation_runs(turn_key);
CREATE INDEX IF NOT EXISTS idx_distill_runs_channel
    ON distillation_runs(channel_id, started_at);
CREATE INDEX IF NOT EXISTS idx_distill_runs_replay
    ON distillation_runs(replay_of) WHERE replay_of IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_distill_runs_pinned
    ON distillation_runs(run_id) WHERE pinned = 1;
CREATE INDEX IF NOT EXISTS idx_distill_passes_run
    ON distillation_passes(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_distill_passes_trace
    ON distillation_passes(trace_id);
CREATE INDEX IF NOT EXISTS idx_distill_div_run
    ON distillation_divergences(run_id);
CREATE INDEX IF NOT EXISTS idx_distill_div_kind
    ON distillation_divergences(kind, command_name);
CREATE INDEX IF NOT EXISTS idx_distill_insights_run
    ON distillation_insights(run_id);
CREATE INDEX IF NOT EXISTS idx_distill_insights_hash
    ON distillation_insights(text_hash);
CREATE INDEX IF NOT EXISTS idx_distill_citations_div
    ON distillation_insight_citations(divergence_id);
CREATE INDEX IF NOT EXISTS idx_distill_verdicts_insight
    ON distillation_verdicts(insight_id, created_at);
```

The one column on an existing table, added in the same `PRAGMA table_info`
guarded block that already handles `conversations.updated_at`
(`observability_store.py:390-402`):

```python
span_cols = {row[1] for row in conn.execute("PRAGMA table_info(spans)").fetchall()}
if "distillation_pass" not in span_cols:
    conn.execute("ALTER TABLE spans ADD COLUMN distillation_pass TEXT")
```

and in the fresh-DB `spans` DDL (`:322-329`), `distillation_pass TEXT` as the
final column. The partial index follows the existing house precedent for a
sparse column (`idx_spans_command … WHERE command_name IS NOT NULL`, `:342`), so
distillation costs nothing on the 99.9% of rows that are not distillation.

**Column, not attribute `[DR23]`.** Four verified reasons a real column beats riding in
the `attributes` JSON: it is indexable without an expression index; it sits
outside the 16 KiB attribute cap (`tracing.py:133`), so a marker on a span with a
large `messages` payload can never be truncated into an envelope; it sits
outside the `Redactor`, which does blind substring replacement of loaded secret
values over the serialized attributes (`observability_store.py:160-168,585-587`);
and `WHERE distillation_pass = 'student'` is documented SQL an agent can write,
whereas `json_extract` on a redacted blob is not.

**Producer-side changes required, each of which fails silently if missed:**

1. `tracing.py:137-152` — `Span` gains `distillation_pass: Optional[str] = None`.
2. `tracing.py:274-278` — `get_distillation_pass(host)` beside `get_channel_id`,
   resolving `current_distillation_pass` on the WEC.
3. `tracing.py:346` — one line in the `Span(...)` literal inside `start_span`.
   This covers **every** span emitted through `start_span`, at all 13 call sites,
   with zero emitter edits.
4. `tracing.py:206-213` — `deterministic_span_id(..., pass_label=None)` per
   `[DR11]`.
5. `workflow_execution_context.py:153-155`, `:305-308` — the field and the
   `current_distillation_pass` property, mirroring `current_turn_key`.
6. `workflow_execution_context.py:405-407,423-426,472-480` — the ask_user open
   and close both take `pass_label`; the close also sets `distillation_pass`.
7. `distillation.py:460-586` — `_run_agent_pass` wraps its body in a context
   manager that sets and, **in the existing `finally` at `:580-586`**, restores
   the label. The restore must cover the student-failure `except` at `:864-867`
   too; a leaked label files every subsequent turn's spans under a stale pass,
   which is worse than the problem being fixed.
8. `observability_store.py:589-616` — `upsert_span_rows` INSERT column list and
   values tuple. **Do not** add it to the `ON CONFLICT DO UPDATE` set:
   write-once at open is the correct semantics, and it keeps the close path from
   being able to relabel a span.
9. `observability_store.py:1333-1345` — `SQLiteTraceSink.emit_span`'s
   field-by-field `Span` snapshot. Omitting the field here makes the entire
   feature a silent no-op: no exception, no log line, no dropped-span counter,
   and a waterfall that looks exactly as it does today. This is the single edit
   most likely to be missed and it gets its own test in `fix-sb8.14`.
10. `tracing.py:270-278, 302-345` — `get_replay_trace_id(host)` and the
    `trace_id = get_replay_trace_id(host) or get_turn_key(host)` substitution
    inside `start_span`, plus the same substitution in the two direct
    `Span(...)` construction sites (`workflow_execution_context.py:476`,
    `:1256-1270`). `[DR41]`. Without this the `~replay` namespace has no
    producer at all and `fix-sb8.11` cannot write a single span.
11. `workflow_execution_context.py:153-155, 305-308` — `_replay_trace_id`, the
    `current_replay_trace_id` property, and the context manager that sets and
    restores it on **every** exit path. `[DR41]`.
12. `workflow_execution_context.py:419-428, 461-486` — both ask_user sites take
    `parent_span_id = distill_pass_span_id(turn_key, label)` when a pass label
    is active, `root_span_id(turn_key)` when it is NULL. `[DR51]`. This is
    separate from item 6 (which covers the id and the column) and is the item
    that makes `[DR8]`'s parenting assertion pass.

**The write path for the six new tables `[DR46]`.** Revision 1 specified no way
to write them, and none exists: `TraceSink` declares exactly `emit_span`,
`emit_turn_record`, `record_conversation_label` (`tracing.py:159-178`),
`NoOpTraceSink` implements only those (`:180-195`), and neither
`WorkflowExecutionContext` nor `ChatSession` holds an `ObservabilityStore` —
only `SQLiteTraceSink` does (`observability_store.py:1285`). D6's promise that
"run writes follow `[R14]`" had no mechanism behind it. Normatively:

> `[DR46]` **Live-run writes go through the sink.** `TraceSink` gains
> `emit_distillation_record(kind: str, payload: dict) -> None` with
> `kind ∈ {run, pass, divergence, insight, citation}`; `NoOpTraceSink`
> implements it as `pass`. `SQLiteTraceSink` enqueues it on the **record**
> queue, so it inherits the existing writer thread, busy-retry, breaker and
> `writer_health` counters, and a write failure can never surface on the turn
> thread — which is what `[R14]` actually requires and what a direct
> `store._connect(timeout=30.0)` call from `distill_message` would have
> violated (a lock contention or `OperationalError` would raise inside
> `_execute_message`, the same uncaught shape §3.3 item 4 documents as fatal).
>
> **Two writes are exempt, deliberately, and both are off the turn thread.**
> (a) `[DR5]`'s replay-suffix mint needs a real `BEGIN IMMEDIATE`, which a
> queue cannot express; and (b) `fix-sb8.9`'s verdict route is an HTTP handler.
> Both are user-initiated, CLI- or viewer-side, and not on a live turn's
> critical path, so both use a direct writable handle wrapped in a blanket
> `try/except` that logs and increments a `writer_health` counter. A failure
> there aborts the *replay* or returns 503 on the *verdict* — neither of which
> is a turn. `[DR53]` further constrains the verdict handle.
>
> `fix-sb8.14` asserts that a store-write failure during distillation does not
> propagate out of `_execute_message`.

**Erasure must reach the six new tables `[DR44]`.** Revision 1 certified
`[R21]` erasure as "No change, verified" on the strength of the `spans`
`channel_id` arm alone. Two reviewers found the same defect and both are right:
`forget_channel` (`observability_store.py:1185-1216`) and `clear_conversations`
(`:1218-1233`) are hardcoded to `feedback`, `spans`, `artifacts`, `turns`,
`conversations` and nothing generalizes them. The new tables hold verbatim user
content — `distillation_runs.user_message TEXT NOT NULL`,
`distillation_passes.entry_inputs_json` (context dicts and the history tail),
`distillation_divergences.param_diff_json` / `detail_json` (user-supplied
parameter values), `distillation_insights.text` — so under revision 1 a channel
erasure and the SPA's "clear all conversations" both left all of it behind. The
parent design's wording is normatively "across **all** tables"
(`docs/fastworkflow_observability_studio_design.md:274-277`), so this was a
breach of `[R21]`, not an extension of it.

> `[DR44]` `forget_channel(channel_id)` additionally deletes:
> `distillation_runs` where `channel_id = ?` **or**
> `turn_key IN (SELECT turn_key FROM turns WHERE channel_id = ?)`; and, by the
> `run_id`s so selected, `distillation_passes`, `distillation_divergences`,
> `distillation_insights`, `distillation_insight_citations` and
> `distillation_verdicts` (the last by `insight_id`). It also deletes replay
> spans by `trace_id LIKE key || '~%'` for each erased turn key, mirroring
> §3.5 row 4. `clear_conversations` extends its table tuple with all six.
>
> Order matters: collect the `run_id` / `insight_id` sets **before** deleting
> the parent rows, inside the existing `BEGIN IMMEDIATE`. There are no
> foreign keys (`PRAGMA foreign_keys` is not enabled, `[DR22]`), so nothing
> cascades on its own, and a wrong order leaves orphans that §15's contradiction
> recipe would then read as real evidence.
>
> `fix-sb8.14` acceptance criterion: `forget_channel` on a channel that has a
> distillation run leaves **zero** rows in all six tables and zero replay spans.
> The same for `clear_conversations`.

---

## 10. Storage budget and retention (requirement 5) `[DR24]`

### 10.1 Measured, not estimated

Taken from a real observability DB on this machine
(`~/.local/state/fastworkflow/workflows/hello_world/observability.sqlite3`,
4 agent turns, 42 spans):

| Measure | Value |
|---|---|
| Spans per **recorded turn** (mixed sample, n=4; only **one** of the four is an agent turn) | 10.5 (per-trace 6 / 11 / 11 / 14) |
| Spans in the single agent trace (`fw.agent.execute` present) | 14 |
| `fw.llm.call` share of span **count** | 19/42 = 45% |
| `fw.llm.call` share of attribute **bytes** | 179,521/184,528 = **97.3%** |
| Mean `fw.llm.call` attribute bytes | 9,448 (max 13,368, against the 16,384 cap) |
| Attribute + row bytes per trace | ~48 KB (per-trace 43.7 / 49.1 / 47.4 / 44.4) |

So span *volume* is a proxy; the real currency is `fw.llm.call` attribute bytes.

**Honesty about the span-count half of this table.** Revision 1 labelled 10.5
"spans per agent turn". Verified: `fw.agent.execute` appears in exactly one of
the four traces, so the agent-turn span count is n=1 at 14 spans, and 10.5 is a
mixed-sample average. The §10.2 budget is re-derived below from the agent trace.
The **byte** figures are unaffected — all four traces sit at 43.7–49.1 KB — so
the ~48 KB baseline and everything downstream of it stand.

### 10.2 The per-run cost, derived from the schema

A `--generate_insights` run is: one `fw.turn` root, two agent passes, two
extractor invocations, plus the new structure spans and rows.

| Component | Spans | Bytes |
|---|---|---|
| `fw.turn` root | 1 | ~0.3 KB |
| Teacher pass (≈ the measured agent trace, less the root) | 13 | ~48 KB |
| Student pass | 13 | ~48 KB |
| `fw.distill.run` / `.pass` ×2 / `.compare` ×2 | 5 | ~2 KB |
| `fw.distill.extract` ×2 | 2 | ~4 KB (raw_output + summary, capped) |
| Extractor `fw.llm.call` ×2 — near-cap: both formatted trajectories plus the whole existing-insights corpus | 2 | ~32 KB |
| **Span subtotal** | **~36** | **~134 KB** |
| `distillation_runs` row (incl. `run_json`, `insight_set_json`) | — | ~2 KB |
| `distillation_passes` ×3 (incl. `entry_inputs_json`) | — | ~6 KB |
| `distillation_divergences` — every aligned pair, ~10 rows × ~600 B | — | ~6 KB |
| `distillation_insights` + citations, 0–4 | — | ~2 KB |
| **New-table subtotal** | — | **~16 KB** |
| **Total per run** | **~36 spans** | **~150 KB** |

**Against a 48 KB / 14-span ordinary agent turn: ~3.1× the bytes
(150 / 48) and ~2.6× the span count (36 / 14).** Revision 1 quoted "2.8× the
bytes and 2.8× the span count", which was the *span subtotal* (134 KB) against
the total baseline — an apples-to-oranges ratio that also happened to omit
exactly the 16 KB of new-table cost the next sentence then reported. The epic's
"roughly triples" is right on bytes. The new tables are ~11% of a run's cost —
real, but not the driver.

At 1 GiB (`_DEFAULT_DB_MAX_BYTES`, `:57`) that is roughly **7,100 distillation
runs**, versus ~22,000 ordinary turns. **That figure is an upper bound, and
knowingly so:** the budget counts row and attribute bytes, while
`db_size_bytes()` (`observability_store.py:1082-1091`) sums the DB file **and**
its `-wal`, i.e. file pages including all index pages plus free pages that
`PRAGMA auto_vacuum=INCREMENTAL` (`:373`) does not reclaim until
`PRAGMA incremental_vacuum` runs (`:1180`) — and §9 adds thirteen new indexes
plus `idx_spans_trace_pass`. A corpus large enough to extract skills from is a
few hundred to a few thousand runs, so the size cap is still **not** the binding
constraint, but "7,100" should be read as "thousands, not tens of thousands".

### 10.3 The 30-day horizon is the binding constraint, and it is silent

`prune()` runs at **every sink startup** (`observability_store.py:1321-1325`) and
deletes spans with `start_ns < horizon` unconditionally
(`:1122-1129`), horizon = 30 days by default (`:58`). Turn records are exempt
(`:1100-1102`); spans are not. So the *default* behaviour today is: **the
evidence behind an insight expires 30 days after the run, at the next CLI
launch, with no warning and no record that it happened** — while the insight
itself sits in a markdown file forever. That is the failure `fix-sb8.13` exists
to prevent, and it is the default, not an edge case.

**Decision `[DR24]`: do not change the global defaults. Pin instead.**
Raising `FW_OBS_RETENTION_DAYS` globally would retain every ordinary turn's
spans too, buying 3% of the value for 100% of the cost, and would be a silent
behaviour change for every non-distillation user. `FW_OBS_DB_MAX_BYTES` stays at
1 GiB for the same reason.

**Decision `[DR25]` — the pin, and what it does not cover.** Both prune arms gain
one predicate:

```sql
AND trace_id NOT IN (
      SELECT trace_id FROM distillation_passes p
      JOIN distillation_runs r ON r.run_id = p.run_id
      WHERE r.pinned = 1)
```

applied to the horizon arm (`:1126-1129`) **and** the size-cap eviction arm
(`:1171-1175`), plus the equivalent guard on the artifact deletes (`:1131-1136`).
The predicate rides `idx_spans_trace`.

**Where the predicate goes, precisely — this is not a detail. `[DR52]`** It goes
**inside the victim-selection subquery**, never on the outer `DELETE`:

```sql
DELETE FROM spans WHERE span_id IN (
  SELECT span_id FROM spans
   WHERE trace_id NOT IN (SELECT trace_id FROM distillation_passes p
                          JOIN distillation_runs r ON r.run_id = p.run_id
                          WHERE r.pinned = 1)
   ORDER BY start_ns LIMIT ?)
```

Verified: the size-cap loop exits on `if cur.rowcount == 0: break`
(`observability_store.py:1178-1179`). With the predicate on the outer `DELETE`,
the first batch whose oldest 5,000 spans happen to all be pinned deletes nothing,
`rowcount` is 0, and the loop **breaks** — leaving the DB over its cap with
evictable spans still present, no error and no marker. Inside the subquery, the
batch selects the oldest *unpinned* spans and the loop makes progress.

`pinned` lives on `distillation_runs`, **not** on `distillation_passes`. This is
deliberate and answers the strongest objection raised against per-pass
pinning: with a per-pass pin, a partial pin can retain the student trace of an
accepted insight while deleting the teacher trace it cites, producing a dangling
divergence record with one side gone — which looks like data, not a bug. Pinning
at run granularity makes the pin **atomic by construction**: a run is pinned
entirely or not at all.

The trade-off `fix-sb8.13` asks to be decided rather than defaulted:

| Run class | Pinned? | Reason |
|---|---|---|
| Produced an **accepted** (`supported`) insight | **yes** | the cited evidence must outlive the rule |
| Produced an **unadjudicated** insight | **yes** | adjudication has not happened; pruning would pre-empt it |
| Produced a **rejected** insight and nothing else | **no** | the conclusion is already recorded on the verdict row; the trace is the bulk |
| **No divergence at all** | **yes for 90 days, then no** | these are the contradiction set for every future rule (§15), and they are also the bulk of the volume. A flat "keep forever" makes the corpus unbounded; a flat "prune immediately" destroys the only counter-evidence pool. 90 days is the compromise: long enough to adjudicate a rule extracted from a run in the same quarter, short enough to bound growth. Implemented as `pinned = 1` at write time with the pin cleared by a bounded sweep in `prune()` for no-divergence runs older than `FW_OBS_DISTILL_NEGATIVE_PIN_DAYS` (default 90) |
| **Non-comparable** | **no** | its divergences are unusable by contract (§6.2); retaining the trace retains a confound, not evidence |

**How a rejected run actually gets unpinned, and the pin ceiling `[DR52]`.**
Revision 1's table required "rejected insight → not pinned" and provided no
mechanism: `pinned` lives on `distillation_runs`, which §12 rule 1 forbade the
verdict route from touching, and the only prune-time sweep specified was the
90-day negative-pin release. Both gaps are closed here:

1. **Rejected-run unpin joins the same bounded `prune()` sweep** that handles
   `FW_OBS_DISTILL_NEGATIVE_PIN_DAYS`: a run whose every insight's newest
   non-superseded verdict is a rejection, and which has no unadjudicated
   insight, gets `pinned = 0`. The verdict route still writes only
   `distillation_verdicts` (§12 rule 1, as reworded); the unpin is a
   *consequence* computed by the pruner, not a write the viewer performs.
2. **The pinned set has a ceiling.** Two of the five classes pin indefinitely,
   so at ~150 KB/run the pinned set alone reaches 1 GiB at ~7,100 runs and the
   size cap would silently stop holding. Normatively: when the size-cap loop
   finishes with `db_size_bytes() > max_bytes` and every remaining candidate is
   pinned, `prune()` (a) writes a `diagnostics` marker `distill_pin_over_cap`
   carrying the pinned byte total and the over-cap delta, (b) logs a warning,
   and (c) if the pinned set alone exceeds `FW_OBS_DISTILL_PIN_MAX_FRACTION`
   (default `0.5`) of `FW_OBS_DB_MAX_BYTES`, evicts **pinned** traces
   oldest-first, trace-atomically, writing the same `[DR27]` eviction marker.
   The cap wins over the pin, loudly — the alternative is a DB that grows past
   its configured bound with no signal, which is the exact class of silence
   `[DR40]` forbids.
3. **The new tables are themselves subject to retention.** Revision 1 left all
   six exempt, so `entry_inputs_json` would accumulate forever against a cap
   that can only be met by deleting spans. At the 30-day horizon an **unpinned**
   run's `entry_inputs_json` is set to NULL and its `_divergences` rows are
   deleted with its spans; the `distillation_runs` and `distillation_insights`
   rows survive (they are the conclusions, and they are small), with the run row
   marked `evidence_pruned = 1` so the UI can say *"the trace behind this is
   gone"* rather than render an empty diff.

**"Why is this trace still here" `[DR26]`** — `fix-sb8.13`'s visibility
requirement, answered by one query and surfaced in the run header:

```sql
SELECT r.run_id, r.turn_key,
       CASE WHEN r.exec_diverged = 0 AND r.planning_diverged = 0
            THEN 'no-divergence contradiction set'
            ELSE 'cited by insight' END AS reason,
       i.insight_id, i.text,
       (SELECT v.verdict FROM distillation_verdicts v
         WHERE v.insight_id = i.insight_id AND v.superseded = 0
         ORDER BY v.created_at DESC LIMIT 1) AS verdict
FROM distillation_runs r
LEFT JOIN distillation_insights i ON i.run_id = r.run_id
WHERE r.pinned = 1 AND r.turn_key = :turn_key;
```

**A hazard neither prune arm handles today, recorded because it will bite.** The
size-cap arm deletes 5,000 spans at a time by `ORDER BY start_ns` with **no
trace awareness** (`:1167-1175`), so over the cap the normal outcome is a
*half-deleted trace* that `buildTurnTree` renders as a waterfall with silently
missing rows and no indication anything was removed. The pin protects pinned
traces and does nothing about this. `[DR27]`: `fix-sb8.13` makes size-cap
eviction **trace-atomic** — select whole victim `trace_id`s oldest-first and
delete their spans together — and writes an eviction marker into `diagnostics`
naming the evicted trace count, so a missing trace reads as evicted rather than
as empty. This is a correctness fix to shipped behaviour, not distillation
scope, but distillation is what makes it consequential.

**Configuration additions:**

| Variable | Default | Meaning |
|---|---|---|
| `FW_OBS_DISTILL_NEGATIVE_PIN_DAYS` | `90` | how long a no-divergence run stays pinned as contradiction evidence |
| `FW_OBS_DISTILL_PIN_MAX_FRACTION` | `0.5` | fraction of `FW_OBS_DB_MAX_BYTES` the pinned set may occupy before the size cap starts evicting pinned traces oldest-first, loudly `[DR52]` |

---

## 11. Schema versioning (requirement 6) `[DR28]`

**Do not bump `SCHEMA_VERSION`. Add a feature marker in `diagnostics` and
feature-detect at read time.**

The gate is fail-closed and, decisively, **coarse**: a reader raises
`IncompatibleObservabilityDB` on a newer `user_version`
(`observability_store.py:387-390`, `:1249-1255`), and in the chatbot
`open_store` catches it only to **re-raise** (`server.py:474-478`) rather than
degrading to the empty views a missing DB gets (`:1001-1011`). So a bump to 2
does not degrade an older build — it makes every v3.2.0 binary **refuse the
entire DB**, including all of its non-distillation turns, over tables it would
never have queried. Post-mortem inspection of a DB you do not own is the
viewer's stated purpose (`server.py:466-470`), and a bump takes that away for
zero benefit: an older build cannot read `distillation_runs` whether or not the
version says 2.

The additive change is provably inert to an older reader in the other
direction, verified in the tree: `get_spans` is `SELECT *` → `dict(r)`
(`:978-982`); `list_turns` uses an explicit projection (`:1039-1044`);
`run_chatbot/server.py:1059-1067` and `run_fastapi_mcp/__main__.py:1037-1047`
are `dict(row)` passthroughs; the schema test is a **subset** assertion
(`{...} <= tables`, `tests/test_observability_store.py:110-138`) so new tables
pass; and its `user_version == 1` assertion at `:117` survives only because we
do not bump.

**The writer-side consequence, which revision 1 never weighed. `[DR43]`**
§11 analysed older builds only as *readers*. As **writers** they are the reason
epic acceptance criterion 9 ("retention cannot prune a trace that an accepted
insight cites") is not achievable unconditionally: `prune()` runs unconditionally
at every sink startup (`observability_store.py:1321-1325`) and a v3.2.0
`prune()` has no pin predicate (`:1122-1140`, `:1165-1180`), so one launch of an
older CLI against the same workflow deletes pinned evidence after the 30-day
horizon. Two reviewers independently observed that bumping `SCHEMA_VERSION`
would *fix* this by accident: `_ensure_schema` would raise
`IncompatibleObservabilityDB` (`:387-390`), `get_or_create_sink` catches it and
returns `None` (`:1840-1842`), so the old build would get no sink and never
prune.

**That counterfactual is real, and the bump is still refused.** Fail-closing an
older build protects the pin by disabling *all* of its observability, and — the
decisive asymmetry — by making the chatbot viewer refuse the whole DB, because
`open_store` re-raises rather than degrading (`server.py:474-478`). Trading
"pinned spans may be pruned by a stale binary" for "every older build loses
reading and recording of every turn, distillation or not" is a bad trade at a
1:1000 row ratio. What is *not* acceptable is leaving the limit implicit, so:

> `[DR43]` The pin binds only builds that carry the pin predicate. A
> mixed-version install defeats it, and epic acceptance criterion 9 is restated
> as: **"retention in builds carrying the pin predicate cannot prune a trace an
> accepted insight cites, and any loss caused by another build is detected and
> displayed."** The detection is concrete, not aspirational: at pin time the run
> row records `pinned_at` and `pinned_span_count` (the live
> `COUNT(*) FROM spans WHERE trace_id = …` at that moment); `fix-sb8.13`
> compares that against the live count whenever a run is opened, and when the
> live count is lower the run header shows **"evidence incomplete — N of M
> spans are gone; a build without the retention pin may have pruned them"** and
> the run is excluded from the promotion view. `evidence_pruned` is set on the
> row so the exclusion survives without a recount.
>
> The escalation path, documented so it is a decision and not a discovery: if
> operators report real losses, the fix is to copy pinned spans into a
> distillation-owned table an older `prune()` does not know about. That roughly
> doubles the bytes of a pinned run and is deliberately **not** shipped in v1.

**But the honest part, which must be signed off rather than discovered.** The
in-file `ALTER TABLE` precedent (`:390-402`) is justified in a comment as
*"schema v1 was never shipped"*. **That premise expired at v3.2.0** (commit
9904df5). Reusing the idiom means two released builds disagree about the `spans`
shape at the same `user_version = 1`. `[DR28]` accepts that deliberately, and
replaces the guarantee `user_version` can no longer provide with one that
actually fits the shape of the change:

> **Feature markers.** `_ensure_schema` writes
> `diagnostics['schema_features'] = ["distillation_v1"]` (merged, not
> overwritten) alongside the existing `schema_opened` probe (`:405-411`).
> `ObservabilityStore` gains `has_feature(name) -> bool`, cached at
> construction, implemented as a `diagnostics` lookup with a
> `PRAGMA table_info(spans)` fallback for a DB written before the marker
> existed.

**This closes the `ReadOnlyObservabilityStore` hole, which is the real hazard
here and is worse than the version question.** `ReadOnlyObservabilityStore`
never runs `_SCHEMA_STATEMENTS` or the ALTER block and connects `mode=ro`
(`:1237-1267`), so it *cannot* migrate. On any DB the new writer has not opened —
a post-mortem snapshot from a 3.2.0 run being the concrete case — a query
projecting `distillation_pass` raises `sqlite3.OperationalError: no such column`,
and `do_GET`'s blanket handler turns that into a bare
`500 internal error: OperationalError` with no message (`server.py:815-824`).
Normatively, `[DR29]`:

1. `get_spans(trace_id, distillation_pass=None)` adds the predicate **only**
   when `has_feature("distillation_v1")`; otherwise it runs the unfiltered query.
2. `/api/meta` grows `distillation_available: bool` so the SPA hides the
   distillation affordances on an old snapshot instead of erroring.
3. Every `/api/distillation/*` route returns an explicit
   `404 {"error": "this database predates distillation recording"}` when the
   feature is absent, never a 500.
4. This is a hard acceptance criterion in `fix-sb8.14`: open a DB created by a
   pre-distillation writer and assert every new route degrades rather than
   raises.

---

## 12. The read-only viewer and adjudication verdicts (requirement 7) `[DR30]`

`fix-sb8.9` must persist verdicts. `[R12]` makes the chatbot viewer read-only
behind a four-path POST allowlist (`server.py:865-875`), guarded by
`tests/test_run_chatbot_server.py:1005` (`test_other_posts_are_still_405`) and
the whole `TestReadOnly` class.

**Decision: add a fifth route, `POST /api/distillation/verdict`, as an argued
exception under the `run_clear_conversations` precedent.** Verdicts stay on the
viewer.

**Why the exception is in keeping with `[R12]` rather than a breach of it.**
The invariant `[R12]` actually protects is stated verbatim in the parent design:
*"recorded observability data stays read-only over HTTP."* All four existing
writes are **control-plane** (`select_workflow`, `configure_env`, `train`) or
**erasure** (`clear_conversations`); none of them *edits* a recorded trace. A
verdict edits nothing either: it is an append to a new annotation table and
cannot alter, delete, or rewrite any span, turn, artifact, run, divergence, or
insight row. The schema already has a precedent for exactly this shape of
object — the `feedback` table, which the parent design calls *"mutable by
design `[R3]`"* — and `fix-sb8.9` explicitly says to follow it rather than
invent a second pattern.

**The precedent is structural, not just rhetorical.** `run_clear_conversations`
(`server.py:1205-1209`) constructs its own **writable** `ObservabilityStore`,
entirely separate from the per-request `open_store()` →
`ReadOnlyObservabilityStore` (`:466-479`). So the mechanism for "the viewer's
read path is read-only, and one enumerated route writes through its own handle"
already exists and is already tested. The verdict route uses it line for line.
The allowlist is a *mechanism* — a deliberate, enumerated, token-and-origin-gated
list — not a quota of four.

**Why moving verdicts off the viewer loses.** Adjudication is, definitionally,
reading the insight beside the divergence records it cites, beside the
teacher/student trace excerpt at that point, beside its support and
contradiction counts (`fix-sb8.9`). Every off-viewer path — a CLI subcommand, a
`bd`-style file, direct SQL — forces the adjudicator to leave the only surface
that shows that evidence, form a judgement, and re-enter it somewhere else. In
practice verdicts would be entered from memory, which defeats the entire point
of an epic whose thesis is that conclusions must be checkable against evidence.
An agent adjudicating through the JSON export (`fix-sb8.12`) has the same
problem in reverse and is served by the same route.

**Normative constraints on the exception:**

1. The route touches **only** `distillation_verdicts`. It never writes
   `spans`, `turns`, `artifacts`, `distillation_runs`, `_passes`,
   `_divergences`, `_insights` or `_insight_citations`. (Revision 1 said "a
   single parameterized `INSERT`. No `UPDATE`, no `DELETE`, ever" two lines
   above rule 2, which mandates an `UPDATE`. The rule is about *which table*, and
   is now stated that way.)
2. **Append-only with supersede.** Changing a verdict inserts a new row **and
   sets `superseded = 1` on the prior non-superseded row** for that `insight_id`,
   in the same transaction. That `UPDATE` is the one write beyond the INSERT and
   it stays within `distillation_verdicts`. The history of judgements is itself
   evidence — an insight that was accepted, then rejected after a replay, is
   exactly the signal `fix-sb8.11` exists to produce. The consequential unpin of
   a rejected run is computed by `prune()` (`[DR52]`), not by this route.
3. Same bearer-token and Host/Origin gates as every other route. **No**
   confirmation phrase — that guard exists for destruction, and applying it to a
   non-destructive annotation would train users to type past confirmations.
4. Body: `{insight_id, verdict, note?, actor}` with `verdict` validated against
   the closed enum in §9 and `actor` validated as `human` or `agent:<name>`;
   `note` capped at 4 KiB. A `replay` actor is written only by the replay path
   (in-process, not over HTTP).
5. `tests/test_run_chatbot_server.py:1005` gains exactly one path, and
   `TestReadOnly` gains a case asserting that the verdict route cannot reach
   `spans`, `turns`, or `artifacts` — the read-only property is re-asserted, not
   relaxed.
6. `[DR30]` **supersedes nothing in `[R12]`**; it extends the enumerated
   allowlist by one and restates the protected object as "recorded observability
   data", which is what `[R12]` always said.
7. **The route must not migrate the DB. `[DR53]`** Revision 1's argument that
   "a verdict edits nothing" is true of the INSERT and false of the *handle*:
   `ObservabilityStore.__init__` calls `_ensure_schema` (`:358-360`), which
   executes every `_SCHEMA_STATEMENTS` entry, the `ALTER TABLE` block,
   `PRAGMA user_version = 1`, an `INSERT INTO diagnostics` write probe, then
   `os.chmod(db_path, 0o600)` and `os.chmod(parent, 0o700)` (`:377-414`).
   Verified. So under revision 1 the *first* verdict POST would mutate the
   post-mortem snapshot the viewer's read-only contract exists to protect
   (`observability_store.py:1237-1244`, `server.py:466-470`) — and would
   silently create the six distillation tables in a pre-distillation DB, the
   exact state `[DR29]` promises to degrade on. Normatively:
   (a) the route evaluates `has_feature("distillation_v1")` through the
   **per-request read-only handle** and returns `404 {"error": "this database
   predates distillation recording"}` *before* any writable handle is
   constructed; (b) the write uses `ObservabilityStore(db_path, migrate=False)`
   (a new constructor flag; a module-level `insert_verdict(db_path, …)` helper
   is an equally acceptable shape) which opens read-write, asserts the feature
   marker, performs the single parameterized INSERT plus the supersede UPDATE,
   and **never** runs `_ensure_schema`, never stamps `diagnostics`, never
   chmods. `run_clear_conversations` keeps its current behaviour; it is the
   precedent for *a writable handle beside the read-only one*, not for
   migrating on read.
8. **The residual truth, stated rather than argued away.** A verdict still
   writes to a file the operator may not own. The claim this design makes is
   narrower than "the viewer edits nothing": it is that the viewer cannot alter
   recorded evidence, and that the one thing it can write is an append-only
   annotation in a table created by the recorder, not by the viewer.

### 12.1 The `/api/distillation/*` route inventory `[DR55]`

Revision 1 named only `/api/distillation/runs`, the wildcard
`/api/distillation/*`, and `POST /api/distillation/verdict`, and marked
`fix-sb8.6` "Fixed except pagination". That was not enough for `fix-sb8.8` to
consume: with no divergence-records route specified, sb8.6 and sb8.8 would each
invent one. The full inventory, fixed here. All are GETs on `_handle_get`'s elif
chain (`server.py:1012-1077`) except the last; all are subject to `[DR29]`'s
explicit 404 on a pre-distillation DB; all use parameterized SQL.

| Route | Query params | Response key | Columns projected |
|---|---|---|---|
| `GET /api/distillation/runs` | `channel`, `conversation`, `comparable` (`0`/`1`), `diverged` (`0`/`1`), `include_replays` (default `0`), `limit` (100), `offset` (0) | `{"runs": [...]}` | `run_id, turn_key, channel_id, conversation_id, user_message, started_at, completed_at, comparable, comparable_reason, isolation_verified, cache_asymmetric, planning_diverged, exec_diverged, material_divergences, planning_insights, execution_insights, extractor_empty, replay_of, pinned, evidence_pruned` |
| `GET /api/distillation/run/<run_id>` | — | `{"run": {...}, "passes": [...]}` | the whole `distillation_runs` row (`run_json` parsed into `record`, mirroring `/api/turn/<k>`'s `record_json` handling at `server.py:1048-1058`) plus every `distillation_passes` row for it, ordered by `seq` — including both pass `trace_id`s, model identity, both fingerprint pairs, timings and cost |
| `GET /api/distillation/divergences/<run_id>` | `kind`, `material` (`0`/`1`/`null`), `level` | `{"divergences": [...]}` | every `distillation_divergences` column, `align_index` ascending, with `param_diff_json` and `detail_json` parsed to objects server-side (the `/api/spans` `attributes` precedent, `server.py:1062-1066`) |
| `GET /api/distillation/insights` | exactly one of `run=<run_id>`, `insight=<insight_id>`, `text_hash=<hash>` | `{"insights": [...], "citations": [...], "verdicts": [...]}` | the §13.2 forward-provenance closure; `insight=` additionally returns the reverse `runs-for-insight` list under `"runs"`, computed by the §15 support recipe |
| `GET /api/spans/<turn_key>?pass=<label>` | `pass` | `{"spans": [...]}` | existing route, `[DR6]` + `[DR7]`; `pass=none` selects the spans belonging to no pass |
| `GET /api/distillation/corpus` | `channel` | `{"weekly", "by_command", "by_kind", "promotion", "promotion_blocked", "cost"}` | §15's aggregates, executed verbatim — `fix-sb8.10`. `promotion_blocked` is true while no run carries `isolation_verified = 1`, so the empty promotion table reads as "not yet checkable" rather than "no support" `[DR48]` |
| `GET /api/distillation/export/<run_id>` | — | one self-contained JSON document | `fix-sb8.12`'s offline export: run, passes, alignment, insights, citations, verdicts, retention status and both passes' spans. Assembled from STORED ROWS only, so it inherits the `[R20]` redaction that ran at the sink boundary |
| `POST /api/distillation/verdict` | — | `{"verdict": {...}}` | §12, `[DR30]` + `[DR53]` |

Two routes were added during implementation and are recorded here rather than
left to drift out of the inventory: `/corpus`, because `fix-sb8.10`'s
aggregates are useless behind a UI-only surface (the `list_train_runs()`
precedent — a store method with no route and no UI, and therefore unreachable
metrics — is the thing to avoid), and `/export/<run_id>`, because
`fix-sb8.12` requires a stable offline document and assembling one client-side
from four routes would be a fifth, undocumented format.

Also implemented beside the inventory: `GET /api/turns` now carries a
`distillation` object per turn where one exists (`distillation_turn_markers`),
which is what `fix-sb8.7`'s list-level marker renders. One statement per page
rather than one per row: the turn list defaults to 100 rows.

**Channel scoping**, which the bead asks for and which the debug rail does not
actually have: `/api/distillation/runs` accepts `channel` and filters on
`distillation_runs.channel_id` via `idx_distill_runs_channel`. Nothing else in
the chatbot read layer scopes by channel (`turnQuery()` never emits a `channel=`
param and `state.channel` is dead code, `index.html:563,780-793`), so this route
is establishing the convention rather than following one. It must not introduce
`/api/channels` or the ids `tabTurns`/`tabConvs`/`channelSel`, all of which are
byte-banned from `index.html` (`tests/test_run_chatbot_server.py:376-389`).

**Replays are excluded by default.** `include_replays` defaults to `0` on the
run list for the same reason §15's recipes carry `r.replay_of IS NULL`: a replay
is a test of an insight, not independent evidence for it.

---

## 13. Insight ledger and bidirectional provenance — `fix-sb8.5`

### 13.1 Stable insight ids `[DR31]`

```
normalized = re.sub(r"\s+", " ", text.strip().lower()).rstrip(".;,")
text_hash  = sha256(normalized).hexdigest()[:16]
insight_id = "ins-" + sha256(f"{run_id}|{kind}|{normalized}").hexdigest()[:12]
```

`run_id` is **in** the id on purpose: two runs producing identical text are two
independent pieces of evidence, not one row, and support counting needs both.
Deduplication is a separate view, grouped on `text_hash` — which is also the
reverse index from a line in a markdown file (where only the text survives) back
to the runs that produced it.

The id is **not** the file entry number. `_append_numbered_insights` continues
numbering by regex over the file (`distillation.py:665-693`), so a hand edit or a
renumber would orphan every reference. `file_entry_number` is stored for display
only and is documented as such in the DDL comment.

**Round-tripping from the file.** The id is appended to the written line as a
trailing HTML comment, which renders invisibly in markdown and does not disturb
the `^(\d+)\.\s` / `^## (\d+)\.` numbering regexes:

```
7. Never call update_task before verifying the task exists  <!-- ins-9f3c1a7b2e04 -->
```

**The marker must be stripped on load, and revision 1's cost accounting was
wrong. `[DR56]`** Revision 1 accounted for one consumer — the extractor prompt —
and there are three. Verified: `_initialize_agent_functionality` loads both
corpora once at agent init (`workflow_execution_context.py:1425-1439`) and hands
them to `initialize_workflow_tool_agent(self, execution_insights=…)`, which
splices them into the **live agent's** signature docstring
(`workflow_agent.py:503-509`, `f"{WorkflowAgentSignature.__doc__}\n\nCRITICAL
ANTI-PATTERNS TO AVOID:\n{execution_insights}"`), and to the **planner**
(`:618-619`). So an unstripped marker would ride into every production agent and
planner prompt on every turn of any workflow with an `Insights/` directory,
distillation or not — which contradicts non-goal 1's promise to change no
prompts.

> `[DR56]` `load_workflow_insights` strips `<!--\s*ins-[0-9a-f]{12}\s*-->`
> (and trailing whitespace it leaves behind) before returning. Every consumer —
> extractor, agent, planner — therefore sees byte-for-byte what it sees today.
> The marker is read only by the ledger tooling that maps a file line back to
> `distillation_insights`.

The marker is **kept** rather than dropped in favour of `text_hash` alone (§21,
objection 5): `text_hash` is a hash of the *text*, so it stops resolving the
moment a human edits the line — and a human editing the line is precisely when
you most want to know which runs produced it. The marker survives the edit; the
hash does not. Cost after stripping: ~24 bytes per line on disk and **zero**
bytes in any prompt.

### 13.2 The provenance chain, both directions `[DR32]`

```
insight ──(distillation_insight_citations)──▶ divergence
        ──(divergences.left_span_id/right_span_id)──▶ spans (global PK)
        ──(spans.trace_id + spans.distillation_pass)──▶ pass
        ──(distillation_passes)──▶ run ──▶ turn
```

Forward, one query (`fix-sb8.5`'s "insight → divergence → span pair"):

```sql
SELECT i.insight_id, i.text, i.kind, i.extractor_span_id,
       d.divergence_id, d.kind AS divergence_kind, d.material,
       d.command_name, d.left_span_id, d.right_span_id,
       r.turn_key, r.comparable, d.left_pass, d.right_pass
FROM distillation_insights i
JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
JOIN distillation_runs r ON r.run_id = i.run_id
WHERE i.insight_id = :insight_id;
```

Reverse, from a span the developer is looking at:

```sql
SELECT DISTINCT i.insight_id, i.text, i.kind, r.turn_key
FROM distillation_divergences d
JOIN distillation_insight_citations c ON c.divergence_id = d.divergence_id
JOIN distillation_insights i ON i.insight_id = c.insight_id
JOIN distillation_runs r ON r.run_id = i.run_id
WHERE d.left_span_id = :span_id OR d.right_span_id = :span_id;
```

**One UI click, both ways.** The SPA has exactly one deep link, `#turn=<key>`,
read once at boot (`index.html:2800-2812`), and it has **never written**
`location.hash` — there is no `pushState`, no `replaceState`, no assignment
anywhere in the file. Round-trip provenance therefore needs the SPA's
first-ever URL write plus a second breadcrumb stack, because `descend()` /
`state.path` (`:1494-1546`) holds exactly one current node. That cost is real,
belongs to `fix-sb8.8`, and is recorded here so it is not scored as free:
`#distill=<run_id>` and `#insight=<insight_id>` join the boot branch, and
selecting a divergence writes the hash.

### 13.3 Negative outcomes are rows, not absences `[DR33]`

Three states are currently invisible and all three are evidence:

1. **Diverged, extractor returned EMPTY.** `distillation_runs.extractor_empty =
   1`, and `fw.distill.extract.empty_reason` distinguishes
   `extractor-returned-empty` (the model said EMPTY, i.e. it judged the
   difference context-justified or a duplicate) from `parse-yielded-nothing`
   (the model answered but the parser kept none of it). That distinction is not
   cosmetic: the planning parser keeps only lines starting with a digit followed
   by `.` within the first three characters (`distillation.py:770-777`), so a
   bullet-formatted answer yields `[]` and would otherwise be indistinguishable
   from a genuine EMPTY. One is "the extractor is too conservative"; the other is
   "the parser is too strict". Today both are the same silence.
2. **No divergence at all.** A `distillation_runs` row with
   `planning_diverged = 0 AND exec_diverged = 0`. This is the "student was
   already correct" set and it is the contradiction pool for every candidate
   rule (§15). It is also why `identical` divergence records are stored (§7.3).
3. **Diverged but restore or comparability failed.** `comparable = 0` /
   `restore_ok_pre_student = 0` or `restore_ok_post_compare = 0`, per §6.2.

Also fixed here, because it currently mislabels the negative case: `distill_message`
sets `any_divergence = True` **before** the extractor runs
(`distillation.py:883`, `:896`), so a divergence that yielded zero insights still
prints "divergence found — 0 insight(s) extracted" and still triggers the
teacher-state restore. The restore behaviour is correct and stays; the CLI
summary and the run row must distinguish *diverged* from *diverged and
extracted*.

---

## 14. Counterfactual replay: what must be stored — `fix-sb8.11` `[DR34]`

A replay re-runs **only** the student, from the recorded entry state, with a
candidate insight set injected, and diffs against the **stored** teacher trace.
Per §3.4 it gets `<turn_key>~replay.<n>`, a real addressable trace with **no
`turns` row**, and a `distillation_runs` row with `replay_of` set.

To reconstruct pass entry the store must hold, and this design mandates:

| What | Where | Why it is not optional |
|---|---|---|
| The user message | `distillation_runs.user_message` | the replay's input |
| The entry fingerprint | `distillation_passes.entry_fingerprint` | the gate: a replay whose reconstructed entry fingerprint differs from the original student pass's is marked non-comparable and returns no verdict |
| **The pass's prompt inputs** | `distillation_passes.entry_inputs_json` | a fingerprint proves comparability; it cannot *reconstruct*. See `[DR45]` immediately below for what this column is and — decisively — what it is not |
| The insight set live at the time | `distillation_runs.insight_set_json` — the ids and `text_hash`es of the planning and execution corpora as loaded at agent init | `_planning_insights` / `_execution_insights` are loaded once at agent init (`workflow_execution_context.py:1430-1440`) and never reloaded, so the corpus a run actually used is *not* the file's current contents. Without this, a replay silently tests a different prompt than the original run used |
| The cited divergences | `distillation_insight_citations` | the assertion is "did *these* divergences disappear", not "did anything change" |

**`entry_inputs_json` is prompt inputs, not restorable state. `[DR45]`**
Revision 1 called this column `entry_state_json` and defined it as the canonical
JSON of the same three components the fingerprint hashes — i.e. as exactly the
lossy hash-input projection that §14 itself argues is insufficient. A reviewer
took that apart and was right on every count: `_canonical` renders numbers as
decimal strings (restoring installs `"1"` where `1` was — the confound §7.2
flags), the history tail is bounded at `messages[-4:]`, `[R20]` redaction is
blind substring replacement over serialized text
(`observability_store.py:160-168`) so any value matching a loaded secret returns
as `[REDACTED]`, and the cme context carries the live `app_workflow` object that
`intent_detection.py:34` reads back. A column that reads as restorable state and
is not is worse than no column.

> `[DR45]` `entry_inputs_json` holds the **prompt inputs** of the pass: the raw
> user message, the refined user message, the plan handed to the executor, the
> history tail as the prompts saw it, and the insight-set ids. It also holds the
> redacted, canonicalized context dicts, explicitly labelled
> `"diagnostic_only": true` — they are there to *explain* a divergence to a
> reader, never to be loaded back into a `Workflow`. Nothing in the system reads
> this column to reconstruct state.
>
> **Replay is therefore world-gated, not world-reconstructing.** A replay is
> valid only when the live workflow's current `state_fingerprint` equals the
> stored student pass's `entry_fingerprint` **and** the reconstructed
> `prompt_fingerprint` equals the stored one. A replay against a drifted world
> returns `not-replayable: world drifted at entry` and no verdict.
>
> **Plus a per-step observation gate.** A ReAct agent's next action is a
> function of the previous tool's `response_text`, so an entry-only gate is not
> enough: the world can move *during* the replay. At each aligned step the
> replay compares its `fw.command.execute` span's `response_text` attribute
> against the stored student pass's; on the first mismatch it returns
> `not-replayable: world drifted at step <i>` instead of a verdict. This is the
> honest generalization of `[DR35]`, whose carve-out was drawn on the wrong
> axis (see the restatement below).

It is written once per pass and is the largest of the new-table costs (§10.2
budgets ~6 KB for all three pass rows together).

**The assertion.** For a replay `R` of run `O` citing divergences `D`:

```
replay_divergences = align(teacher_spans(O), student_spans(R))
divergence_removed = not any(d.command_key == c.command_key and d.kind == c.kind
                             for c in D for d in replay_divergences)
```

Written to `fw.distill.replay.divergence_removed` and to a
`distillation_verdicts` row with `actor = 'replay'` and
`replay_run_id = R.run_id`.

**Scope limit, normative `[DR35]`.** Application object state
(`_root_command_context`, `_current_command_context`, `workflow.py:166-168`, and
the application's own objects) is **not** captured and cannot be by this design.
A replay therefore reconstructs the *context dicts* and the *prompt inputs*, not
the application world. Consequently:

> **A replay verdict is valid only while the reconstructed observations match
> the stored ones.** Concretely, all four gates must hold: (i) the origin run is
> `comparable = 1`; (ii) `isolation_verified = 1` (`[DR48]`); (iii) the entry
> `state_fingerprint` and `prompt_fingerprint` match (`[DR45]`); (iv) every
> aligned step's `response_text` matches the stored student pass's
> (`[DR45]`'s per-step gate). Any failure returns a `not-replayable: <reason>`,
> never a verdict.
>
> `divergences.replayable = 0` for `kind = 'different-answer-same-actions'` and
> `= 1` otherwise. **Revision 2 keeps that flag but demotes it from the whole
> argument to a cheap pre-filter.** Revision 1 drew the carve-out on the wrong
> axis: it excluded response-content evidence and admitted action-level
> evidence, but actions are *chosen from* observations that the uncaptured
> application state produces, so an action-level divergence in a replay is
> confounded by exactly the same drift. The per-step observation gate is the
> check that actually constrains the axis the state capture bounds; the
> `replayable` flag now only saves the cost of starting a replay that gate would
> certainly fail.

This is the honest boundary. Selling replay as a general regression harness
would be a claim the state capture does not support, and it would be discovered
the first time a replay "validated" an insight about a wrong customer id by
re-running against a world the teacher had already mutated.

---

## 15. Documented SQL recipes — `fix-sb8.10` / `fix-sb8.12` `[DR36]`

These ship in
`fastworkflow/skills_for_coding_fastworkflows/debug-workflow-conversations/reference.md`
alongside the existing schema, per the `fix-kw7.15` precedent: **verified by
execution, not written from memory.** Every one carries `r.comparable = 1`
(§6.2 contract).

*Verification status of this document:* the §9 DDL and every SQL block were
executed against a SQLite database built from the live `_SCHEMA_STATEMENTS`
(`observability_store.py:300-345`) plus the §9 additions. They parse and run.

**That was not enough, and revision 1 said so without acting on it. `[DR54]`**
"They parse and run" is a syntax guarantee. A reviewer built a three-row fixture
— one comparable run, one **non-comparable** run, one **replay** — and executed
the flagship promotion query against it: it returned `support_runs = 3` where
the correct answer is 1. I reproduced that result independently before accepting
the finding. Two defects, both silent:

1. `LEFT JOIN distillation_runs r ON r.run_id = sup.run_id AND r.comparable = 1`
   nulls `r` but **leaves the `sup` row in `COUNT(DISTINCT sup.run_id)`**, so
   the comparability filter is inert — directly violating §6.2 obligation 3.
2. The support recipe and the counts query lacked `AND r.replay_of IS NULL`
   (which the contradiction and weekly-rate recipes both have), so a replay run
   — whose alignment result lands in `distillation_divergences`, that being the
   only table with `kind`/`command_key` — counts as independent support for the
   very insight it was run to test. A replay that *fails* to remove the cited
   divergence would have **increased** that insight's support count.

Both are corrected below.

> `[DR54]` A §15 recipe is not "verified" until it has been executed against a
> seeded fixture containing **at least one non-comparable run, one replay run,
> one run-level (NULL-`command_name`) divergence and one NULL-`command_name`
> failed-command span**, with the expected row set asserted. `fix-sb8.12` ships
> that fixture as a test, not as a paragraph. Parse-checking is what let the
> above through and it is retired as a standard of evidence here.

*Revision-2 verification, run before this document was re-issued:* all eight
recipes below were executed against exactly that fixture — runA (comparable,
isolation-verified, cites an action divergence), runB (**non-comparable**, same
command/kind), runC (**replay of runA**, same command/kind), runD (comparable,
**no divergence**, with a student-pass `fw.command.execute` span for the cited
command), plus a run-level `different-answer-same-actions` divergence with NULL
`command_key`/`command_name` cited by a second insight. Results: support → runA
alone; counts → `ins-1` 1/1 and `ins-2` 0/0 (**still listed**, not silently
dropped); `missing-in-student` by command → runA alone; action contradiction →
runD; run-level contradiction → runD. Under revision 1's text the first of those
returned 3 supporters and the last returned nothing at all.

**Which turns SUPPORT insight X** — comparable runs exhibiting the same
`(command, kind)` pattern the insight cites, including the run it came from:

```sql
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
WHERE r.comparable = 1
  AND r.replay_of IS NULL      -- a replay tests an insight; it is not support for it
ORDER BY r.started_at DESC;
```

**Which turns CONTRADICT insight X** — comparable runs where the cited command
ran **in the student pass** and produced no divergence of the cited kind, i.e.
runs where the rule would have fired wrongly. Note the `cited.command_name IS
NOT NULL` guard: `distillation_divergences.command_name` is nullable and §7.3
step 6 creates `different-answer-same-actions` records at `level='run'` with no
command, so without the guard `s.command_name = cited.command_name` evaluates to
NULL under three-valued logic, `EXISTS(...)` is false for every run, and the
query returns **zero rows with no error** for that entire divergence kind —
leaving gate acceptance criterion 7 unmet and looking like "no contradictions
found". Verified by execution.

```sql
WITH cited AS (
  SELECT DISTINCT d.command_key, d.command_name, d.kind
  FROM distillation_insights i
  JOIN distillation_insight_citations c ON c.insight_id = i.insight_id
  JOIN distillation_divergences d ON d.divergence_id = c.divergence_id
  WHERE i.insight_id = :insight_id
)
SELECT r.run_id, r.turn_key, r.started_at, cited.command_name
FROM distillation_runs r
CROSS JOIN cited            -- deliberate: every cited (command, kind) against every run
WHERE cited.command_name IS NOT NULL   -- run-level divergences have no command [DR54]
  AND r.comparable = 1
  AND r.replay_of IS NULL
  AND EXISTS (SELECT 1 FROM spans s
               WHERE s.trace_id = r.turn_key            -- [DR1]: the invariant holds
                 AND s.distillation_pass = 'student'
                 AND s.name = 'fw.command.execute'
                 AND s.command_name = cited.command_name)
  AND NOT EXISTS (SELECT 1 FROM distillation_divergences d2
                   WHERE d2.run_id = r.run_id
                     AND d2.command_key = cited.command_key
                     AND d2.kind = cited.kind)
ORDER BY r.started_at DESC;
```

Note `s.trace_id = r.turn_key` — the join the identity decision buys. Under a
derived-per-pass-id scheme this becomes a join through a synthetic id whose
grammar the agent must first learn, against shipped documentation that
contradicts it.

**Support and contradiction counts for every insight** (the promotion decision):

```sql
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
           AND cit.command_key IS NOT NULL   -- run-level citations key on nothing
           AND r.comparable = 1
           AND r.replay_of IS NULL           -- a replay is a test, not support [DR54]
           AND r.isolation_verified = 1      -- promotion is a causal claim [DR48]
           AND r.evidence_pruned = 0)                            AS support_runs,
       (SELECT COUNT(DISTINCT sup.run_id)
          FROM distillation_insight_citations c
          JOIN distillation_divergences cit ON cit.divergence_id = c.divergence_id
          JOIN distillation_divergences sup  ON sup.command_key = cit.command_key
                                            AND sup.kind        = cit.kind
          JOIN distillation_runs r           ON r.run_id = sup.run_id
         WHERE c.insight_id = i.insight_id
           AND cit.command_key IS NOT NULL AND sup.material = 1
           AND r.comparable = 1 AND r.replay_of IS NULL
           AND r.isolation_verified = 1 AND r.evidence_pruned = 0) AS material_support_runs
FROM distillation_insights i
ORDER BY support_runs DESC;
```

Two shape choices worth stating, both forced by revision 2's fixture:

- **Correlated subqueries, not a join chain.** The join form loses the insight
  row entirely whenever the filter matches nothing — so a run-level insight
  (whose `command_key` is NULL) and a freshly extracted insight with no
  corroboration would both vanish from the promotion list rather than appear
  with a zero. An insight that silently disappears from the view a human uses to
  decide is exactly the failure mode this epic exists to remove.
- **`material_support_runs` counts runs, not divergence rows.** Revision 1's
  `SUM(CASE WHEN sup.material = 1 …)` counted rows, so one run contributing three
  material divergences read as three supporters.

This is **the promotion view**, so it is the one query that carries
`isolation_verified = 1`. Until `fix-35m.3` lands it returns no rows, and
`fix-sb8.10` must render that as *"promotion is blocked: application-object
isolation is not yet verified for any run"* rather than as an empty table.
Every other recipe on this page works on `comparable = 1` alone.

**Divergence rate by week** (is the student improving as insights accumulate?):

```sql
SELECT strftime('%Y-W%W', r.started_at) AS week,
       COUNT(*)                                                       AS runs,
       SUM(CASE WHEN r.exec_diverged = 1 THEN 1 ELSE 0 END)           AS diverged,
       SUM(r.material_divergences)                                    AS material,
       ROUND(1.0 * SUM(CASE WHEN r.exec_diverged = 1 THEN 1 ELSE 0 END)
                 / COUNT(*), 3)                                       AS rate
FROM distillation_runs r
WHERE r.comparable = 1 AND r.replay_of IS NULL
GROUP BY week ORDER BY week;
```

**All `missing-in-student` divergences by command, with supporting run ids:**

```sql
SELECT d.command_name, COUNT(*) AS n,
       GROUP_CONCAT(DISTINCT r.run_id) AS run_ids
FROM distillation_divergences d
JOIN distillation_runs r ON r.run_id = d.run_id
WHERE d.kind = 'missing-in-student' AND d.material = 1
  AND r.comparable = 1 AND r.replay_of IS NULL   -- [DR54]
GROUP BY d.command_name ORDER BY n DESC;
```

**Teacher vs student cost and latency** (the economic case for the student):

```sql
SELECT p.role,
       COUNT(*) AS passes, SUM(p.tokens) AS tokens,
       ROUND(SUM(p.cost_usd), 4) AS cost_usd,
       ROUND(AVG(p.wall_ms)) AS avg_ms,
       SUM(p.cache_hits) AS cache_hits, SUM(p.cache_misses) AS cache_misses
FROM distillation_passes p
JOIN distillation_runs r ON r.run_id = p.run_id
WHERE r.comparable = 1 AND r.cache_asymmetric = 0 AND p.role IN ('teacher','student')
GROUP BY p.role;
```

**Which turns CONTRADICT a run-level insight** — the eighth recipe, added in
revision 2 so the `different-answer-same-actions` kind is not silently
unanswerable. A run-level divergence has no command to key on, so the
contradiction is defined over outcomes: comparable runs whose entry context
matches the cited run's and whose two passes **agreed** on the final answer.

```sql
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
```

Note `IS` rather than `=` on the join predicates: both columns are nullable and
`NULL = NULL` is NULL, which is the same three-valued trap the action-level
recipe fell into.

**From a markdown line back to its runs** (the reverse index):

```sql
SELECT i.insight_id, r.run_id, r.turn_key, r.started_at
FROM distillation_insights i JOIN distillation_runs r ON r.run_id = i.run_id
WHERE i.text_hash = :text_hash ORDER BY r.started_at;
```

---

## 16. The `[R24]` reversal (requirement 4) `[DR37]`

**The reversal.** Parent ruling `[R24]` reads, in
`docs/fastworkflow_observability_studio_design.md` §3.4 "Test mode" and in the
§9 decision-log table: *"insights distillation is explicitly out of Studio
scope."* **`[DR37]` reverses that.** Distillation review is now a first-class
chatbot surface: run list markers, per-pass waterfalls, the aligned diff, the
insight ledger, and adjudication.

**The reason, stated as it will be read a year from now.** `[R24]` was decided
on 2026-08-25, when distillation was a CLI feature whose product was two
markdown files and Studio was scoped to trace debugging plus manual testing.
Scoping it out was correct *then*: there was nothing to view — the comparison
inputs were locals, the diff was a prose string, and no artifact carried a
back-reference. What changed is not the boundary of Studio but the purpose of
distillation: these insights are now the raw material for extracting planner
**skills and verifiers**, and a rule is only promotable if you can see how many
turns support it, which contradict it, and whether applying it removes the
divergence. That is a *review* activity over recorded evidence — which is
exactly what Studio is, and the only surface in the system that already renders
a trace. Keeping distillation out would mean building a second viewer, a second
span renderer, and a second read layer beside the one that exists, or leaving
the verification to unaided reading of two markdown files. `[R24]`'s other
half — the additive `/initialize` startup fields and the CLI-parity table — is
untouched and stands.

**Exact amendment text to be added to
`docs/fastworkflow_observability_studio_design.md`.** A later step applies this;
this document does not edit that file.

*(1) In §3.4, in the "Test mode" paragraph, immediately after the sentence
ending "…so per-session variation is testable; insights distillation is
explicitly out of Studio scope.", append:*

> *Amendment (owner ruling, 2026-08-28, `fix-sb8.1`) — `[R24]`'s
> distillation exclusion is REVERSED.* Distillation review is now in scope and
> is a debug-mode surface, designed in
> `docs/distillation_observability_design.md` (rulings `[DR1]`–`[DR56]`). The
> reason `[R24]` was right in August and is wrong now is that distillation's
> purpose changed: its insights are the raw material for extracting planner
> skills and verifiers, and promotion requires seeing an insight beside the
> divergence records it cites, the trace excerpts behind them, and its support
> and contradiction counts. That is review over recorded evidence — Studio's
> definition — and the alternative is a second viewer beside the one that
> exists. Distillation remains CLI-only to *produce* (`--generate_insights` is
> guarded on `user_message_queue is not None`, so Topology B can never run it);
> only the *review* surface is in Studio. The rest of `[R24]` — the additive
> `/initialize` `startup_command`/`startup_action`/`context` fields and the
> CLI-parity table — is unchanged.

*(2) In §3.2, in the schema block, after the `spans` `CREATE TABLE` statement,
append the comment line:*

> ```
> -- + distillation_pass TEXT (additive, no user_version bump) [DR1][DR28];
> --   distillation_runs / _passes / _divergences / _insights /
> --   _insight_citations / _verdicts per docs/distillation_observability_design.md §9
> ```

*(3) In the §9 decision-log table, replace the `R24` row's ruling cell with:*

> `Additive /initialize startup fields; parity table. Distillation-out-of-scope REVERSED 2026-08-28 by [DR37] — see docs/distillation_observability_design.md §16`

*(4) In §5 Configuration, append two rows:*

> | `FW_OBS_DISTILL_NEGATIVE_PIN_DAYS` | `90` | how long a no-divergence distillation run stays pinned as contradiction evidence `[DR24]` |
> | `FW_OBS_DISTILL_PIN_MAX_FRACTION` | `0.5` | fraction of `FW_OBS_DB_MAX_BYTES` the pinned set may occupy before size-cap eviction starts evicting pinned traces, loudly `[DR52]` |

---

## 17. Non-goals `[DR38]`

Stated so they are refused rather than re-litigated:

1. **Extractor prompts and insight wording are not changed.**
   `InsightExtractionSignature` (`distillation.py:141-180`) and
   `PlanningInsightExtractionSignature` (`:71-138`) are untouched. The divergence
   summary the extractor receives is *rendered from* the new structured record
   (§7.6) so there is one source of truth — better extractor input is a side
   effect, not the goal. The `<!-- ins-… -->` provenance marker `[DR31]` writes
   into the markdown files is **stripped by `load_workflow_insights`** before
   any prompt sees it (`[DR56]`), so the extractor's, the live agent's
   (`workflow_agent.py:503-509`) and the planner's (`:618-619`) prompts are
   byte-identical to today. Revision 1 accounted only for the extractor and
   would have silently changed two production prompts. Insight quality heuristics, the prompt-level dedup
   against the pasted corpus, and the EMPTY policy are out of scope; this epic
   *records* what they did, including when they did nothing.
2. **Distillation for Topology B / FastAPI is out of scope, by construction.**
   `_process_agent_message` guards on
   `self._generate_insights and self.user_message_queue is not None`
   (`workflow_execution_context.py:1643`), and only `ChatSession` injects queues
   (`chat_session.py:126-133`). No FastAPI path can produce a distillation run.
   The *review* surface is served by the chatbot, which reads the same DB; the
   FastAPI `GET /turns/{turn_key}/spans` endpoint gains a docstring correction
   (§3.5 row 8) and nothing else.
3. **Multi-student (>2-way) comparison is not shipped.** The schema must be able
   to grow to N passes without a migration, and §4 specifies exactly how — free
   text pass labels, a row-per-pass child table, pairwise divergence records
   carrying both pass labels. We ship 2. No N-way UI, no N-way alignment, no
   N-way aggregate.
4. **Per-step materiality is not delivered.** §7.4 states the limit and why.
5. **Full application-state capture for replay is not delivered.** §14's
   `[DR35]` states the limit and constrains the replay verdict accordingly.

---

## 18. Per-child implementation notes

**`fix-sb8.2` — `distillation_runs` grouping record.** *Fixed:* the table shape,
its keys, and every column is specified in §9, including the identity question
it was blocked on — `turn_key` is the trace id, so the row needs no
`teacher_trace_id`/`student_trace_id` pair and no synthetic-id grammar for
readers to learn; per-pass facts live in `distillation_passes` keyed
`(run_id, pass_label)`, which is also the N-pass shape (§4). `DistillationResult`
gains `run_id` so the CLI summary (`run/__main__.py:252-257`) can point at the
record. *Left open:* what the run row shows for a run whose teacher pass raised
before the student ran — the `get_lm` ValueError path (§3.3 item 4) is uncaught
today, so sb8.2 must decide whether a half-run writes a row at all. Recommend:
yes, with `comparable = 0`, `comparable_reason = 'teacher-raised'` and
`completed_at` NULL, because a run that could not start is itself a fact worth
seeing in the list. *Added in revision 2:* `[DR46]` supplies the write path this
child was silently assuming — `TraceSink.emit_distillation_record` through the
existing writer thread, with the two named off-turn-thread exemptions. Write the
run row through that method, never through a direct
`ObservabilityStore._connect()` on the turn thread.

**`fix-sb8.3` — comparability evidence.** *Fixed:* §6 gives the exact
fingerprint function, its exclusion rules and their reasons, the `history_tail`
bound and why, the two `restore_ok_*` definitions with their distinct baselines, the full downstream NON-COMPARABLE
contract (five numbered obligations), and the verified finding that per-call
cache hit/miss is a **query over existing `fw.llm.call` attributes**, not new
instrumentation. §5 fixes the ownership seam with `fix-35m.3`: one function, one
call per pass boundary, two consumers. *Revision 2 changed three things here and they are not cosmetic:* the single
fingerprint became **two** (`state_fingerprint` / `prompt_fingerprint`,
`[DR47]`) because a history-bearing hash makes `material` and `restore_ok`
constants; `_digest` lost its `default=str` because `cme._context` carries a
live `Workflow` object and the hash was therefore of a **heap address**
(`workflow_execution_context.py:1008`); and `restore_ok` became two columns with
two distinct baselines, because `distillation.py:852` restores toward the
pre-teacher state while `:866`/`:900` restore toward the teacher's exit state.
`comparable` also gained an explicit published limit (§6.2) and a companion
`isolation_verified` column that sb8.3 may only ever write NULL or 0 into
(`[DR48]`). *Left open:* nothing structural, but one
prerequisite is assigned here rather than to 35m.3 —
`snapshot_workflow_state` must `copy.deepcopy` both context dicts, because
`Workflow._to_dict()` returns `self._context` by reference (`workflow.py:450`)
and a fingerprint over an alias reports agreement by construction. Ship the deep
copy and the fingerprint together or neither means anything.

**`fix-sb8.4` — structured divergence record.** *Fixed:* §7 settles what is
aligned (`fw.command.execute` and `fw.ask_user` **spans**, not the action log —
which also fixes the dropped-failures bug for free and hands provenance its span
ids), the two-key scheme and why matching on `command_key` is what produces one
`same-command-different-params` pair instead of two orphans, the O(nm) LCS plus
the deterministic reordering post-pass, the complete seven-value taxonomy, the
plan-level correction from word-split to plan-string, and the materiality rule
including that `identical` records are stored. *Left open:* the `param_diff_json`
shape for `fix-sb8.8`'s parameter-level highlighting is specified only as
"per-key before/after"; sb8.4 fixes the exact JSON and sb8.8 consumes it. Also
open: how deeply nested parameter values are diffed — recommend one level, with
deeper values compared whole, since the alternative is a general structural diff
this epic does not need. *Added in revision 2, and both are correctness fixes
rather than refinements:* `[DR50]`'s `raw_command` fallback, without which every
failed command in a pass shares one `command_key` and any teacher failure aligns
as `identical` with any student failure — the exact case §7.1 claims to rescue;
and `[DR49]`'s read barrier, without which a merely *late* span becomes a
fabricated `missing-in-student` divergence that is then pinned forever.

**`fix-sb8.5` — insight ledger.** *Fixed:* §8 gives the extractor its own span
(`fw.distill.extract`) with the exact attribute list, including the decision to
store `existing_insights` as length + SHA-256 rather than body; §13 gives the
stable id formula and argues why `run_id` is inside it and the file entry number
is not, the `text_hash` reverse index, the HTML-comment marker that makes a
markdown line round-trip, and the three negative outcomes as rows — including
the `extractor-returned-empty` vs `parse-yielded-nothing` distinction that
today's silence conflates and that tells you whether the extractor or the parser
is the problem. *Left open:* whether an insight may cite divergences from more
than one run. The citations table permits it structurally; recommend no for v1
(one run per insight, since that is what the extractor is actually given), and
let cross-run consolidation be a view over `text_hash` rather than a stored
relation.

**`fix-sb8.6` — read API.** *Fixed:* the routes are all GETs added to
`_handle_get`'s elif chain (`server.py:1012-1077`), following the existing
parameterized-SQL convention, so the four-path POST allowlist is untouched by
this child. `[DR6]` adds `?pass=` to `/api/spans/<turn_key>` as a query
parameter rather than a path grammar, which is what keeps `selectTurn`'s
single-string contract intact. `[DR7]` requires a pass-filtered response to omit
the `fw.turn` root so each pass gets its own time window. `[DR29]` is a hard
acceptance criterion: every new route must return an explicit 404 on a
pre-distillation DB, never fall through to `do_GET`'s bare
`500 internal error: OperationalError` (`server.py:815-824`). *Added in revision 2:* §12.1 is the full route inventory — paths, query
parameters, response keys and the §9 columns each projects — because revision 1
named only two route strings, and `fix-sb8.8` cannot paint a stored alignment
without a divergence-records route, so sb8.6 and sb8.8 would each have invented
one. §12.1 also settles the channel-scoping convention the bead asks for, and
records that the debug rail does not actually have one to follow. *Left open:*
pagination only; follow the `limit`/`offset` convention already used by
`/api/turns` (`server.py:1024-1047`).

**`fix-sb8.7` — mark distillation turns; per-pass waterfall.** *Fixed:* the
pass selector is `selectTurn(turnKey, passLabel)` plus one query parameter, not
a new fetch contract; `[DR7]` solves the shared-`[t0,t1]` problem server-side
(verified against `index.html:1420-1422`) rather than in untestable browser
code; §8's span hierarchy makes each pass a real node. The run header's contents
are fully specified by §6.2 and §9. *Left open, and it is a real gap:* the turn
rail's projection deliberately omits `record_json`
(`observability_store.py:1039-1044`), so list-level marking cannot read anything
out of the turn record. Either `/api/turns` grows a `distillation` flag column
(recommended — one `EXISTS` subquery against `distillation_runs`, which
`idx_distill_runs_turn` serves) or the SPA joins a separate
`/api/distillation/runs` call client-side. Decide in sb8.7; both work, the
former is one less fetch.

**`fix-sb8.8` — side-by-side aligned diff.** *Fixed:* D4 and §7 mean the browser
paints a stored alignment and computes nothing. The survey's conclusion stands
and is adopted: `renderWaterfall` (`index.html:1654-1710`) is **not** an
extension point — it takes one `node`, lays rows out by wall clock against one
shared window, and hardwires every row's click to `descend(child)` on the single
global `state.path` stack. The aligned diff is a new renderer; the reusable
surface is `el()`, `clear()`, `fmtNs`/`fmtCost`/`fmtTokens`, `spanCategory`, and
the `.cat-*` tokens. Three whole-file byte constraints bind every line of it:
the `innerHTML` ban (`tests/test_run_chatbot_server.py:378,678`;
`test_run_chatbot_test_mode.py:276`), the non-loopback URL ban including inside
comments (`:670-679`), and the forbidden ids `tabTurns`/`tabConvs`/`channelSel`
and the string `/api/channels` (`:376-389`). *Left open:* the two-pane layout
under `.wfRow`'s fixed grid (`index.html:272-275`, already floored at a 56px bar
track) — halving the pane starves it further, so sb8.8 needs its own row CSS
rather than reusing `.wfRow`; and `renderWaterfall` appends its own legend on
every call (`:1696-1709`), so a naive two-pane reuse yields two legends.

**`fix-sb8.9` — insight adjudication.** *Fixed:* §12 decides the `[R12]`
question — verdicts stay on the viewer behind a fifth POST route, argued from
the `run_clear_conversations` precedent (`server.py:1205-1209`, a writable store
handle beside the read-only per-request one) and constrained by six normative
rules including append-only-with-supersede. §9 gives the verdict table with the
closed enum sb8.9 specified. *Left open:* the promotion decision itself — this
epic records verdicts and their evidence; whether a `supported` insight becomes
a planner skill is a separate decision with a separate surface, and this design
deliberately stops at making that decision *answerable*.

**`fix-sb8.10` — corpus view.** *Fixed:* §15 gives eight executable recipes
covering every question sb8.10 lists, all carrying `comparable = 1` per §6.2;
the cost/latency rollup additionally excludes `cache_asymmetric = 1` runs
(§6.3), because a cache hit in one pass and a miss in the other is a cost
confound that would otherwise flatter the student. Storing `identical`
divergence records (§7.3) is what makes every *rate* computable rather than only
every count. *Left open:* whether the corpus view is a chatbot tab or lives only
in SQL. Recommend a tab that runs exactly the §15 recipes, so the UI and the
agent surface cannot drift.

**`fix-sb8.11` — counterfactual replay.** *Fixed:* §3.4 and `[DR5]` give a
replay a place to live — `<turn_key>~replay.<n>`, minted transactionally, with
no `turns` row and no conversation ordinal consumed — and §3.4 documents the
trap that made this necessary: writing replay spans into the original trace
regenerates the same deterministic span ids and the `ON CONFLICT DO UPDATE`
clause **mutates the cited evidence as a successful write**. §14 enumerates
exactly what must be stored to reconstruct pass entry, including the two things
easiest to forget: the entry state itself (not just its fingerprint) and the
*insight set as loaded at agent init*, which is not the file's current contents
because `_planning_insights`/`_execution_insights` are loaded once and never
reloaded (`workflow_execution_context.py:1430-1440`). *Revision 2 is a material narrowing of this child and the owner should read it
as such.* Three things changed. (1) `[DR41]` supplies the producer path the
`~replay` namespace lacked — without it this child could not have written a
single span. (2) `entry_state_json` is renamed `entry_inputs_json` and is
**explicitly not restorable state** (`[DR45]`): prompt inputs plus a diagnostic,
redacted context snapshot. (3) Consequently replay is **world-gated, not
world-reconstructing** — it requires the live workflow's current
`state_fingerprint` to equal the stored student entry fingerprint, and it aborts
at the first step whose `response_text` differs from the stored one. That puts a
cold corpus sweep across a mutated world out of reach, which is the honest
answer given that application objects are never captured
(`workflow.py:443-451`), and makes same-world replay of a recent run
trustworthy. `[DR48]` additionally refuses a verdict while `isolation_verified`
is not 1. *Left open:* the corpus-sweep driver — how many replays run
concurrently and against what budget (that is `fix-35m.3`'s budget machinery,
§5) — and, now, **whether a world-gated replay finds enough eligible runs to be
worth building**. That is a measurement sb8.11 must take before it writes the
driver, not an assumption this document can make for it.

**`fix-sb8.12` — agent surface.** *Fixed:* §15's recipes, the `base_turn_key`
generalization to document, and the exact list of documentation sites that stay
true versus need amending (§3.5 rows 8, 9, 10). The `[R24]` amendment text is
written verbatim in §16 for this child to apply. *Left open:* the JSON export's
schema. Recommend one file per run containing the run row, all pass rows, both
pass span lists, all divergence records, insights, citations, and current
verdicts — i.e. exactly the closure of the joins in §13.2 — versioned with an
`export_version` integer so an extraction agent can pin it.

**`fix-sb8.13` — provenance-aware retention.** *Fixed:* §10 gives the measured
budget (2.8× bytes, 2.8× spans, ~150 KB/run, ~7,100 runs to the 1 GiB cap), the
finding that the **30-day horizon running at every sink startup** is the binding
constraint rather than the size cap, the pin predicate for both prune arms, the
decision to pin at *run* granularity so a pin can never be partial, the
per-run-class retention table including the 90-day compromise for the
no-divergence contradiction set, and the "why is this trace still here" query.
*Revision 2 additions:* `[DR52]` puts the pin predicate **inside** the
victim-selection subquery (on the outer `DELETE`, the `rowcount == 0` break makes
one all-pinned batch stop eviction entirely), gives the pinned set a ceiling with
a `distill_pin_over_cap` diagnostic, assigns the rejected-run unpin to the same
bounded sweep as the 90-day negative-pin release, and puts the six new tables
under retention. `[DR43]` records `pinned_at` / `pinned_span_count` so a loss
caused by an older build without the predicate is **detected and displayed**
rather than silent — which is the whole of what acceptance criterion 9 can
honestly promise on a non-version-bumped DB. *Left open, and it is a correctness
fix beyond distillation:* `[DR27]`'s
trace-atomic size-cap eviction. Today the size-cap arm deletes 5,000 spans at a
time by `ORDER BY start_ns` with no trace awareness (`:1167-1175`), so a
half-deleted trace renders as a waterfall with silently missing rows. sb8.13
owns the fix and the `diagnostics` eviction marker.

**`fix-sb8.14` — tests.** *Fixed:* the assertions §3.6 makes checkable in Python
(disjoint pass span sets; every pass span descending from its own
`fw.distill.pass`), and three specific silent-failure sites that each need a
dedicated test because none of them raises: the `emit_span` field-by-field copy
(`observability_store.py:1333-1345`), the `list_turns(command_name=…)` regression
(`:1034-1037`), and `[DR29]`'s pre-distillation-DB degradation. Add the
`deterministic_span_id` prerequisite test for `[DR11]` — that the default digest
is byte-identical when `pass_label` is None, which is what keeps the six
existing assertions in `tests/test_tracing_phase1.py` green. *Revision 2 adds six required cases, every one of which corresponds to a defect
that shipped in revision 1 and would have shipped green:* (a) `state_fingerprint`
is **byte-equal across two processes** for the same reconstructed state
(`[DR47]`); (b) two passes taking identical actions produce `material = 0`
(`[DR20]`); (c) two *different* failed commands in one pass produce two different
`command_key`s (`[DR50]`); (d) `forget_channel` on a channel with a distillation
run leaves zero rows in all six tables and zero replay spans (`[DR44]`);
(e) `SELECT COUNT(*) FROM turns WHERE turn_key LIKE '%~%'` is 0 while every
replay span's `trace_id` contains `~` (`[DR41]`); (f) a store-write failure
during distillation does not propagate out of `_execute_message` (`[DR46]`).
Add also the `fw.ask_user` parenting case (`[DR51]`). *Left open:*
producing a real two-pass distillation run in a test at all. None of
`LLM_TEACHER_AGENT`, `LLM_STUDENT_AGENT`, `LLM_DISTILLATION` exists in
`fastworkflow/examples/fastworkflow.env` and `dspy_utils.get_lm` raises when
unset, so sb8.14 must either add them to the template or drive
`DistillationSession` with a scripted-agent double — the latter matching the
"scripted-agent doubles drive real `invoke_command`" precedent in the parent
design §6, which is the no-mocks-compliant path.

**`fix-kw7.11` — pass pollution of the live trace.** *Fixed:* this is the
origin symptom and it is fully answered — `spans.distillation_pass` plus §8's
span hierarchy give pass boundaries, `[DR6]`'s `?pass=` filter gives separate
viewing, `[DR7]` gives each pass its own time window, and `[DR11]` closes the
`fw.ask_user` id collision the fix would otherwise arm. One correction to the
bead's framing, from the survey: `buildTurnTree`'s run-length grouping
(`index.html:1390-1396`) already splits *alternating* phases, so a two-pass agent
turn today renders as "Planning 1 / Execution 1 / Planning 2 / Execution 2" —
mislabelled as replans rather than flatly interleaved. The sharp hazard is
narrower: `last.phase === phase` **merges adjacent same-phase runs**, so a
teacher-execution run immediately followed by a student-execution run collapses
into one node. **Revision 2 retracts the SPA edit revision 1 prescribed.** Revision 1 said the
fix was the run-key comparison at `index.html:1394` becoming
`last.phase === phase && last.pass === pass`. Under §8's hierarchy that line can
never fire: `buildTurnTree` puts a span in `top` only when its parent is the
`fw.turn` root or unrecorded (`:1377-1387`), so with `fw.distill.run` parenting
`fw.distill.pass` parenting the pass's work, `top` holds **one** span, the phase
builders (`:1390-1417`) never run, and `buildOtherNode` returns that one child
unwrapped (`:1352-1357`). Verified against the file.

What actually fixes kw7.11 is structural, not a comparison:

- **Unfiltered view** — the hierarchy itself marks the boundaries. A distilled
  turn renders `turn → fw.distill.run → {fw.distill.pass ×N,
  fw.distill.compare, fw.distill.extract ×2}`, each pass's phases one drill-down
  in. No merge, no interleave, no mislabelled "Planning 2".
- **Per-pass view** — `[DR7]`, extended in revision 2 to omit the
  `fw.distill.run` and `fw.distill.pass` wrappers as well as the `fw.turn` root,
  so the pass's own spans reach `top`, the **existing** phase builders run
  unchanged, and each pass lays out against its own time window.

The remaining SPA work is a `passOf()` accessor beside `phaseOf` (`:1060`)
reading the top-level property `span.distillation_pass` (`/api/spans` returns
the column directly), a `pass:` field in the `makeNode` opts literal
(`:926-943`), and a pass badge on span rows — labelling, not grouping. If a
future change ever lets pass spans reach `top` unwrapped, the `:1394` composite
key becomes necessary again; it is recorded as a contingency, not a deliverable.
Any per-pass wrapper must still handle `buildExecutionNode` returning an
**array** (`:1238-1242`, `:1274-1278`) and `buildOtherNode` returning an
unwrapped child (`:1354-1356`), both absorbed today by `children.concat(built)`
at `:1417`. *Left open:* nothing; kw7.11 closes when sb8.7 lands.

---

## 19. Test strategy

Per `.cursor/rules/testing_rules.mdc` and the parent design §6: no mocks;
scripted-agent doubles drive real `invoke_command`; store contract tests against
real SQLite in `tmp_path`. New in `tests/test_distillation_observability.py`,
beyond `fix-sb8.14`'s own list:

- **Silent-failure sites get explicit tests**, because none of them raises: the
  `emit_span` field copy, the `list_turns(command_name=…)` regression, the
  pre-distillation-DB degradation (`[DR29]`), and the `deterministic_span_id`
  byte-identity default (`[DR11]`).
- **Cost of the feature when off**, following the `TestDisabledObservabilityDspyCost`
  precedent from `fix-kw7`: with no sink, distillation must emit nothing and pay
  nothing.
- **Prune interaction**: a pinned run survives a prune that removes an unpinned
  one; a replay trace is deleted with its base turn by
  `prune(include_conversationless_turns=True)` (§3.5 row 4).
- **Erasure `[DR44]`**: `forget_channel` removes replay spans via the
  `channel_id` arm **and** must leave zero rows in `distillation_runs`,
  `_passes`, `_divergences`, `_insights`, `_insight_citations` and `_verdicts`;
  `clear_conversations` likewise. Revision 1 asserted `[R21]` stayed whole
  without change; it did not — both functions are hardcoded five-table lists
  (`observability_store.py:1185-1233`) and the new tables hold verbatim user
  content. Assert the row counts, not the absence of an exception.
- **Write-path failure `[DR46]`**: a distillation record write that raises
  inside the store must not propagate out of `_execute_message`.
- **Fingerprint stability `[DR47]`**: `state_fingerprint` over the same
  reconstructed state is byte-equal in two separate Python processes.

---

## 20. Decision Log

| Ruling | Decision |
|---|---|
| `[DR1]` | Identity: `trace_id == turn_key` with an indexed `spans.distillation_pass` column; `base_turn_key(trace_id) == turns.turn_key` generalizes the invariant; `~` suffix only for replays |
| `[DR2]` | Separator is `~` (RFC-3986 unreserved, unescaped by `encodeURIComponent`, absent from minted turn keys); never `#`/`%`/`/`/space, which the chatbot never `unquote`s |
| `[DR3]` | Column and field named `distillation_pass`, free text, never `pass` (Python keyword) |
| `[DR4]` | A replay must never write into the original trace: deterministic span ids collide and the upsert mutates cited evidence as a successful write |
| `[DR5]` | `~` suffixes are minted only by the replay path, transactionally, only for `comparable = 1` runs |
| `[DR6]` | `GET /api/spans/<turn_key>?pass=<label>` — a query parameter, not a path grammar; preserves `selectTurn`'s single-string contract |
| `[DR7]` | A pass-filtered span response omits the `fw.turn` root, so each pass gets its own waterfall time window |
| `[DR8]` | Pass separation is asserted in Python (disjoint span sets, parenting), not trusted from a JavaScript reading |
| `[DR9]` | `distill_message` sets `_turn_agent_result` to the teacher's result; `turns.answer` stops holding the student's last raw tool response |
| `[DR10]` | `_run_agent_pass` snapshots and restores `_turn_outputs`; the turn record keeps the teacher's outputs |
| `[DR11]` | `deterministic_span_id` gains `pass_label`, folded in only when non-None. **Hard prerequisite for `[DR10]`** |
| `[DR12]` | The schema is N-pass-capable (free-text labels, row-per-pass child table, pairwise divergences with both pass labels); we ship 2 |
| `[DR13]` | One `state_fingerprint()` implementation, called once per pass boundary by sb8.3; `fix-35m.3` reads it and defines no second one |
| `[DR14]` | The fingerprint formula and its exclusion classes; superseded in part by `[DR47]`, which splits it into `state_fingerprint` and `prompt_fingerprint` |
| `[DR15]` | `comparable = 1` iff every pass's entry `state_fingerprint` is equal — attesting **context and history comparability only, never application-object isolation**; five numbered obligations on every consumer of a non-comparable run; `restore_ok` split into `restore_ok_pre_student` / `restore_ok_post_compare` with distinct baselines |
| `[DR16]` | Cache asymmetry is flagged separately and confounds **cost**, not comparability |
| `[DR17]` | Align over `fw.command.execute` and `fw.ask_user` **spans**, not the action log; plan steps are plan strings, not word splits |
| `[DR18]` | Two canonical keys per step: `command_key` for matching, `step_key` for detecting |
| `[DR19]` | O(nm) LCS over `command_key`, a deterministic reordering post-pass, and the seven-value taxonomy; `identical` records are stored |
| `[DR20]` | Materiality is a run-level judgement from exit **`state_fingerprint`**s (a history-bearing hash would make `material` a constant 1), projected onto records; `NULL` when non-comparable; per-step materiality is out of scope with the reason stated |
| `[DR21]` | Reserved span prefix `fw.distill.*`, registered as `DISTILL_SPAN_NAMES`; never `fw.train.*` |
| `[DR22]` | Full DDL per §9; additive; house style (`IF NOT EXISTS`, explicit indexes, no inline FKs) |
| `[DR23]` | The pass marker is a real indexed column, not an `attributes` key: indexable without an expression index, outside the 16 KiB attribute cap, outside the `Redactor`'s substring pass, and writable as documented SQL by an agent |
| `[DR24]` | Storage: ~150 KB and ~36 spans per run, ~3.1× the bytes of an ordinary agent turn (~2.6× the spans); the ~7,100-run cap figure is an upper bound because the cap measures file+WAL including index pages. Global retention defaults unchanged; the 30-day horizon at every sink startup is the binding constraint; pin instead |
| `[DR25]` | The pin predicate on both prune arms plus artifacts; `pinned` on `distillation_runs`, so a pin is atomic |
| `[DR26]` | "Why is this trace still here" is one query and appears in the run header |
| `[DR27]` | Size-cap eviction becomes trace-atomic with a `diagnostics` eviction marker — half-deleted traces are a shipped correctness bug |
| `[DR28]` | Do **not** bump `SCHEMA_VERSION`; the fail-closed gate is too coarse and would make v3.2.0 refuse whole DBs. The phase-7 "pre-release" justification has expired and this is a new, deliberate ruling |
| `[DR29]` | Feature markers in `diagnostics` + `has_feature()`; every new read degrades explicitly on a pre-distillation DB, never 500s |
| `[DR30]` | A fifth POST route, `/api/distillation/verdict`, append-only with supersede, following the `run_clear_conversations` precedent; `[R12]`'s protected object is the recorded trace, which a verdict cannot touch |
| `[DR31]` | Stable insight ids `ins-<12 hex>` over `(run_id, kind, normalized_text)`; `text_hash` as the reverse index; the file entry number is display-only; an HTML-comment marker round-trips the markdown line |
| `[DR32]` | Bidirectional provenance via `distillation_insight_citations` and `divergences.left/right_span_id`; the SPA's first hash writes belong to sb8.8 and are not free |
| `[DR33]` | Negative outcomes are rows: EMPTY (distinguishing extractor-empty from parse-empty), no-divergence, and failed comparability |
| `[DR34]` | Replay stores user message, entry fingerprints, **prompt inputs** (`entry_inputs_json`, not restorable state — `[DR45]`), and the insight set as loaded at agent init |
| `[DR35]` | `different-answer-same-actions` is `replayable = 0`. Revision 2 demotes this from the whole argument to a pre-filter: the real constraint is `[DR45]`'s world gate, because actions are chosen from observations the uncaptured application state produces |
| `[DR36]` | The **eight** documented SQL recipes of §15, all filtering `comparable = 1`, verified by execution against the `[DR54]` adversarial fixture |
| `[DR37]` | `[R24]`'s "insights distillation is explicitly out of Studio scope" is **REVERSED**; amendment text in §16 |
| `[DR38]` | Non-goals: extractor prompts/wording; Topology B distillation; >2-way comparison; per-step materiality; full application-state capture |
| `[DR39]` | The alignment is computed and persisted server-side; the SPA paints a stored record (there is no JS test harness in the repo) |
| `[DR40]` | Recording is best-effort per `[R14]`; a run whose comparability could not be established is marked NON-COMPARABLE and excluded from aggregates — silence is never read as agreement |
| `[DR41]` | The `~replay` namespace gets a producer: a separate `current_replay_trace_id` host attribute preferred over `current_turn_key` for `trace_id` only; `current_turn_key` is never overridden and the replay driver never calls `_begin_turn`/`finalize_turn_for_observability` |
| `[DR42]` | A distilled turn's `conversation_summary` and `refined_user_message` come from the pass the run row names (`turn_fields_from`), not from whichever pass happened to run last |
| `[DR43]` | The pin binds only builds carrying the pin predicate; acceptance criterion 9 is restated as "cannot prune, and any loss by another build is detected and displayed", implemented via `pinned_span_count` / `evidence_pruned` |
| `[DR44]` | `forget_channel` and `clear_conversations` extend to all six new tables and to replay spans; `[R21]` was breached by revision 1, not preserved |
| `[DR45]` | `entry_inputs_json` is prompt inputs plus a diagnostic context snapshot, **never restorable state**; replay is world-gated at entry and at every step's `response_text`, not world-reconstructing |
| `[DR46]` | The write path for the six tables: `TraceSink.emit_distillation_record` through the existing writer thread; two named off-turn-thread exemptions (replay-suffix mint, verdict route) |
| `[DR47]` | Two fingerprint projections — `state_fingerprint` (no history, no `default=str`, `app_workflow` excluded) and `prompt_fingerprint` (history at an explicit entry bound) — with a table of which gates what |
| `[DR48]` | `isolation_verified` is written only by `fix-35m.3`; sb8 never writes 1; the promotion view and replay refuse while it is not 1, every other surface works on `comparable` alone |
| `[DR49]` | The aligner reads in-process `Span` objects, flushes the sink as a barrier before writing divergences, and marks the run `comparable = 0` / `evidence-incomplete` if `spans_dropped` moved |
| `[DR50]` | A NULL-`command_name` span keys on normalized `attributes.raw_command`; without it every failed command in a pass shares one key |
| `[DR51]` | `fw.distill.pass` gets a deterministic span id so both `fw.ask_user` sites can compute it as their parent; a wrong parent at open is unfixable at close |
| `[DR52]` | Pin predicate **inside** the victim subquery; pinned-set ceiling with a `distill_pin_over_cap` diagnostic; rejected-run unpin in `prune()`; the six new tables are themselves under retention |
| `[DR53]` | The verdict route feature-checks on the read-only handle and writes through a non-migrating connection — constructing an `ObservabilityStore` runs `_ensure_schema`, which migrates and chmods |
| `[DR54]` | A §15 recipe is verified only against a fixture containing a non-comparable run, a replay run, a run-level NULL-key divergence and a NULL-`command_name` span; parse-checking is retired |
| `[DR55]` | The full `/api/distillation/*` route inventory (§12.1), including the divergence-records route `fix-sb8.8` consumes and the channel-scoping convention |
| `[DR56]` | `load_workflow_insights` strips the `<!-- ins-… -->` marker, so the live agent's and planner's prompts are byte-identical to today |

---

## 21. Rejected objections

Recorded so the next reader does not re-litigate them. Each was raised in the
revision-1 adversarial review, each is substantive, and each is rejected here
with an argued counter-case rather than by silence. Where the *finding* was
accepted but the reviewer's proposed *remedy* was not, that is stated.

**1. "Bump `SCHEMA_VERSION` to 2, so older builds fail-closed and cannot prune
the pinned spans."** *Finding accepted, remedy rejected.* The mechanism is real
and I verified it: a bump makes `_ensure_schema` raise
`IncompatibleObservabilityDB` (`observability_store.py:387-390`),
`get_or_create_sink` catches it and returns `None` (`:1840-1842`), so an older
build gets no sink and never calls `prune()`. But that "protection" is a side
effect of disabling the older build's observability entirely, and — decisively —
of making the chatbot **viewer** refuse the whole DB, because `open_store`
re-raises rather than degrading (`server.py:474-478`) and post-mortem inspection
of a DB you do not own is the viewer's stated purpose (`:466-470`). The trade is
"pinned distillation spans may be pruned by a stale binary" against "every older
build loses read *and* write access to every turn, distillation or not", at a
row ratio of roughly 1:1000. `[DR43]` takes the finding and answers it with
detection (`pinned_span_count`, `evidence_pruned`, a run-header warning, and
exclusion from the promotion view) plus a documented escalation path, rather
than with a gate that costs three orders of magnitude more than it protects.

**2. "Drop `entry_state_json` from the DDL rather than shipping a column that
reads as restorable state and is not."** *Finding accepted in full, remedy
partially rejected.* Every one of the reviewer's four losses is real —
canonicalization retypes values, the history tail is truncated, `[R20]`
redaction is destructive, and the cme context carries a live object. But
deleting the column deletes the only record of *what the pass was actually
prompted with*, which is the thing a human reading a divergence most wants and
which no other table holds. `[DR45]` renames it `entry_inputs_json`, defines it
as prompt inputs plus an explicitly `"diagnostic_only": true` context snapshot,
and — the part that matters — removes every claim that anything reads it back.
Replay is re-specified as world-*gated* rather than world-reconstructing, which
is what the state capture can actually support.

**3. "Until `fix-35m.3`'s read-only surface check is in force, record
`comparable = 0` (or NULL) rather than 1."** *Rejected.* `comparable` has a
precise published meaning — the entry `state_fingerprint`s were equal — and
`comparable = 0` triggers five specific downstream obligations (§6.2) including
a loud "the passes did not start from the same place" banner, which would be a
false statement about a run whose fingerprints matched perfectly. Overloading
one flag with a second, unrelated fact would empty every surface (the run list,
the waterfall, the diff, the raw counts), not just the causal one, and would
make sb8 unshippable ahead of 35m.3 for no gain in honesty. `[DR48]` instead
adds a second column that says the second thing, publishes the limit of the
first in §6.2 and in the UI label, and blocks exactly the surface that makes a
causal claim — the promotion view and replay.

**4. "Make `fix-35m.3` a hard prerequisite of `fix-sb8.3` in §18."** *Rejected,
for the same reason.* sb8.3 records evidence; 35m.3 creates a guarantee. Serializing
them means no distillation evidence exists at all until an unrelated epic lands,
and the first thing 35m.3 will want, while building its isolation work, is the
recording that shows whether isolation is happening. The dependency is real but
it is a dependency of *conclusions*, not of *recording*, and `[DR48]` places it
exactly there.

**5. "Drop the `<!-- ins-… -->` HTML-comment marker and rely on `text_hash` for
the file→ledger direction."** *Finding accepted, remedy rejected.* The reviewer
is right that revision 1 accounted for one prompt consumer and there are three
(`workflow_agent.py:503-509` for the live agent, `:618-619` for the planner),
and that shipping it unstripped would have violated non-goal 1. But `text_hash`
is a hash of the text, so it stops resolving the instant a human edits the line
— and a human editing a distilled rule is precisely the moment provenance
matters most. `[DR56]` keeps the marker and strips it in
`load_workflow_insights`, so every prompt is byte-identical to today and the
round-trip survives edits.

**6. "Narrow `[DR8]`'s separation assertion to exclude `fw.ask_user`, and drop
'under that pass's `fw.distill.pass` span' from §7.1 for ask_user."** *Rejected
in favour of the reviewer's own alternative (b).* Weakening the assertion would
leave "the student had to ask the user something the teacher inferred" — which
§7.1 calls one of the most informative divergences available — outside the
structure the whole design is built on, and would leave `[DR7]`'s pass-filtered
view rendering an ask_user as a sibling of the pass rather than inside it.
`[DR51]` gives `fw.distill.pass` a deterministic id instead, which costs one
helper function and makes both the open and the by-hand-rebuilt close able to
compute the parent.

**7. "`fix-sb8.6`'s route inventory is missing" — with the implication that the
whole child was mis-scoped as 'Fixed'.** *Finding accepted; the "Fixed" label
was wrong and §12.1 now supplies the inventory.* Recorded here only because the
reviewer's framing ("sb8.6 and sb8.8 will each invent it") understates the
consequence: the two children would have invented *different* ones, and the
divergence-records shape is the contract between them.

---

## 22. Review record

This section is the evidence that the Phase-0 gate was exercised, mirroring the
role `fix-kw7.1` played for the parent platform. It is deliberately unflattering
where the draft was wrong.

### What was attacked

Four independent adversarial reviews were run against revision 1, each with a
different brief: **identity-skeptic** (is the `trace_id == turn_key` ruling
survivable?), **replay-skeptic** (can the stored evidence actually deliver
`fix-sb8.11` and `fix-sb8.13`?), **ops-skeptic** (does this work in a real
install — erasure, versions, write paths, retention?), and **scope-skeptic**
(does each child's stated deliverable follow from the text?). Between them they
filed **8 blocking** and **21 major/minor** findings. Reviewers verified
citations against the tree rather than against the document, and two of them
executed SQL against seeded fixtures.

### Verdicts

| Reviewer | Verdict |
|---|---|
| identity-skeptic | gate-passes-with-revisions |
| replay-skeptic | **gate-fails** (revision 1) |
| ops-skeptic | gate-passes-with-revisions |
| scope-skeptic | gate-passes-with-revisions |

The `gate-fails` is the important one and it was correct: revision 1's
fingerprint hashed a heap address, its erasure certification was false, and its
replay-reconstruction column could not reconstruct. Those are not presentation
defects.

### What survived a direct attack

Recorded because it is the load-bearing part of the document and all four
reviewers tried to break it independently:

- **The identity ruling `[DR1]`–`[DR5]`.** Every claim was re-verified against
  the tree by at least two reviewers and all held: `selectTurn` really does pass
  one string to both `/api/turn/<k>` and `/api/spans/<k>` inside a `Promise.all`
  (`index.html:822-837`) with `api()` throwing on non-2xx, so option (a)'s
  "zero SPA changes" claim is false; `list_turns(command_name=?)` really does
  filter `turn_key IN (SELECT trace_id FROM spans WHERE command_name=?)`
  (`observability_store.py:1034-1037`), so option (a) would silently drop
  distilled turns from the command-filtered rail; the shipped agent SQL asserts
  the invariant in seven places; `~` cannot occur in a minted turn key
  (`turn.py:91-107`) and `grep -c unquote fastworkflow/run_chatbot/server.py`
  is 0. One reviewer called `[DR4]` "the strongest thing in the document" after
  confirming that `ON CONFLICT(span_id) DO UPDATE` rewrites `end_ns`, `status`
  and `attributes` from `excluded` while `trace_id`/`parent_span_id` are absent
  from the set — so a replay written into the original trace would mutate cited
  evidence as a *successful* write.
- **`[DR7]`'s root omission.** Two reviewers set out to prove it was hand-waving
  and could not: `buildTurnTree`'s `extent = rootSpan ? spanExtent([rootSpan]) :
  mergeExtents(...)` (`index.html:1420-1422`) does fall through, and
  `mergeExtents` handles the empty list safely (`:885-894`). Revision 2 extends
  the omission list rather than retracting the ruling.
- **`[DR11]` as a hard prerequisite for `[DR10]`.** Recognising that fixing the
  `command_outputs` bug *arms* an `fw.ask_user` span-id collision was called
  "the finding I would have expected a review to catch, not the author".
- **`[DR23]`'s column-not-attribute argument**, confirmed against `Redactor.redact`'s
  blind substring pass (`observability_store.py:160-168`).
- **§10.1's storage measurement.** A reviewer opened the cited DB and reproduced
  42 spans / 4 turns and 179,521 of 184,528 attribute bytes in `fw.llm.call`.
  The byte conclusions stand; the *span-count* label did not (see below).
- **The additive-read-inertness claim in §11**, in the reader direction: a grep
  for fixed-arity row unpacking across the package found only
  `ConversationSummary(**row)` (`run_fastapi_mcp/__main__.py:2016`), over a
  table this design does not touch.

### The eight blocking findings, and what happened to each

| # | Finding | Disposition |
|---|---|---|
| 1 | The `~replay` namespace had no producer path; the only available mechanism was the `current_turn_key` override §3.3 uses to reject option (c) | **Fixed** — `[DR41]`: a separate `current_replay_trace_id`, plus §9 producer items 10-11 and two `fix-sb8.14` assertions |
| 2 | `state_fingerprint` hashed a live object's memory address via `default=str`, so no replay could ever pass its own gate | **Fixed** — `[DR47]`: `_digest` has no `default=`, `_canonical` emits structural tokens for non-JSON values, `app_workflow` is named in the exclusion list, and a cross-process byte-equality test is mandated |
| 3 | Erasure: `forget_channel` and `clear_conversations` are hardcoded five-table lists, so the six new tables survive a channel erasure; §3.5 row 5 and §19 asserted the opposite | **Fixed** — `[DR44]`, §3.5 row 5 corrected from "No change, verified" to "CHANGES REQUIRED", §19 asserts row counts |
| 4 | `entry_state_json` was defined as the fingerprint's own lossy hash-input projection, i.e. it could not reconstruct | **Fixed, with the remedy narrowed** — `[DR45]`; see §21 objection 2 |
| 5 | No write path exists for the six new tables; `TraceSink` has three methods and no component holds a store | **Fixed** — `[DR46]` |
| 6 | `[DR28]`'s no-bump lets an older build's startup `prune()` delete pinned evidence, so acceptance criterion 9 is unachievable | **Finding accepted, bump refused** — `[DR43]` restates the criterion and makes the loss detectable; see §21 objection 1 |
| 7 | `[DR20]` materiality can never be 0, because every pass appends its own LLM summary before the exit fingerprint is taken | **Fixed** — `[DR47]` splits the fingerprint; `[DR20]` reads the history-free projection; `fix-sb8.14` asserts `material = 0` on identical action sequences |
| 8 | `command_key`/`step_key` degenerate for every failed command — the exact case §7.1 markets as the win | **Fixed** — `[DR50]`'s `raw_command` fallback plus a `fix-sb8.14` case |

Blocking finding 9 in scope-skeptic's numbering — `[DR10]` assigning `_turn_outputs`
containment to sb8 while §5's table assigned it to 35m.3 — is also **fixed**, by
splitting the §5 table row and adding the normative "35m.3 must not reset
`_turn_outputs`" sentence.

### What else changed as a result

Beyond the blocking set: `restore_ok` split into two columns with distinct
baselines; `comparable` gained a published limit and the `isolation_verified`
companion; the `fw.ask_user` parent was reparented onto a newly deterministic
`fw.distill.pass` id; an alignment read barrier was added; the pin predicate
moved inside the victim subquery and gained a ceiling and an over-cap
diagnostic; the new tables came under retention; the verdict route stopped
migrating the DB; the `/api/distillation/*` route inventory was written; the
insight marker is stripped before any prompt sees it; §15 gained a seventh
recipe and lost two silently-wrong queries; the §10.2 ratio was corrected from
2.8× (spans-only) to ~3.1× (total); §10.1's span-count row was relabelled after
a reviewer showed the sample contains only one agent turn; and `dspy_logger`'s
`model` citation was corrected from `:395` to `:374`.

### What the gate did not settle

Stated plainly, because a gate that claims to have closed everything has not
been exercised:

1. **Whether world-gated replay finds enough eligible runs to be worth
   building.** `[DR45]` made `fix-sb8.11` honest and thereby made it narrower.
   Nobody has measured how many stored runs still match their world when
   somebody wants to replay them. sb8.11 must measure before it builds.
2. **Whether `isolation_verified` will ever be 1**, i.e. whether `fix-35m.3`
   lands. Until it does, the promotion view is empty by design. That is correct
   but it means the epic's headline capability — deciding a rule is real — is
   gated on another epic.
3. **The mixed-version pin gap.** `[DR43]` detects it; nothing prevents it short
   of the escalation path (copying pinned spans into a distillation-owned
   table), which is documented and not shipped.
4. **Per-step materiality** (`[DR20]`) and **full application-state capture**
   (`[DR35]`, `[DR45]`) remain out of scope with their limits published.
