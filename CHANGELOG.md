# Changelog

Releases before 3.4.0 were announced in their merge-commit subjects
(`feat: v3.2.0 — observability store, chatbot debug UI, …`) and are recoverable
with `git tag` and `git log --first-parent main`. This file starts at 3.4.0; it
does not backfill them.

## 3.4.0 — observation offloading and search

**Observation offloading and answer-time rehydration become the framework's
behaviour for every workflow.** They are no longer modes a deployment opts into:
there is no `FW_OBSERVATION_OFFLOADING` and no `FW_ANSWER_REHYDRATION`. A
fastWorkflow tool agent compacts its trajectory, keeps every observation
reachable, and answers over the evidence behind its labels rather than over the
labels themselves.

### Added

- **Observation offloading** (`fastworkflow.observation_offloading`): canonical
  `O{n}` aliases on every execute observation, an eager SQLite archive, a packed
  trajectory target with offload labels, `search_memory` over one stored
  observation, and segmented continuation with forced replans.
- **Context-instance line** (`fastworkflow.context_identity`): every execute
  observation names the context instance the command ran in, so a listing
  produced inside a context is still attributable to that instance by a reader
  that cannot use the order of the commands.
- **Answer-time rehydration** (`fastworkflow.answer_rehydration`): the extract
  call is given its own copy of the trajectory with the evidence behind offload
  labels put back, under a byte budget, so the answer is written over evidence
  rather than over pointers.
- **Finish reminder**: a `finish` action that never opened a named item of the
  request goes back to the loop once, and only while iterations remain.
- **Context budgets** (`fastworkflow.context_budget`): one input — the model's
  context window — and every byte budget derived from it as a fixed fraction.
  `budget_provenance()` returns the input, its source and every budget.

### Changed

- **Known-name guard**: a known command name is never answered by a context
  that does not own it; the declining prediction carries a hint naming where the
  command lives. When no context on the chain owns it, that hint is the whole
  response and the next message goes through ordinary intent detection, rather
  than `you_misunderstood` and its clarification stage, which only matches the
  current context's commands and so could not route the command the hint names.
  The guard also covers a reply to `you_misunderstood`, which is matched against
  the same full command set; the ambiguity stage, which matches a short
  suggestion list, is still excluded. With several owners, the hint names each
  entering command beside the context it enters.
- **Threshold separation**: `write_ambiguity_thresholds` is the single writer
  for both ambiguity files and establishes a non-empty ambiguity band where the
  artifacts are produced, with `TIER_AMBIGUITY_MIN_SEPARATION` and
  `SINGLE_LABEL_RESOLUTION_FLOOR`.
- **Workflow fingerprint scope rule v2**: a root `benchmarks/` tree and the
  runtime observability store leave `workflow_content_entries`.
  `WORKFLOW_SCOPE_RULE_VERSION` 1 → 2, so a v1 declaration reads as
  `incomparable`, not stale.
- **Synthetic utterance generation** sends `temperature` only; `top_p` is gone,
  because current Bedrock Claude models reject both together and every
  generation call was a hard `BadRequestError`.
- `fw.nlu.intent` span contract v2 → v3.
- `fw.command.execute` span contract v2 → v3: the four auto-navigation
  attributes are gone with the two-step dispatch that wrote them.
- **Intent `signal_version`** no longer carries a threshold-semantics segment.
  It now reads `intent-classifier/<artifact version>/...`, so a version string
  identifies the artifact behind a signal and nothing else.
- **Offloading evidence lives in the observability store.** Archived
  observations, their subjects and the offloading runtime's diagnostic events
  are rows of `offload_evidence`, `offload_subjects` and `offload_events` in the
  workflow's `observability.sqlite3` (feature markers `offload_evidence_v1`,
  `offload_events_v1`; no schema-version bump). They are keyed by turn, and
  `forget_channel`, Clear conversations and retention pruning delete them in
  the same transactions that delete the turn record. Events are read back with
  `ObservabilityStore.offload_events(...)`.
- **Evidence is redacted when it is written.** With
  `FW_OFFLOAD_EVIDENCE_REDACTION=on` (the default) responses and event text are
  stored as the trace sink's credential scrub and capture policy leave them;
  `off` stores them verbatim for development. The process running a turn keeps
  the raw text of its redacted observations in memory until the turn is over,
  so the agent's own reads stay exact; a turn resumed in another process reads
  the redacted text.
- **Observability recording is always on**, for fastWorkflow's entry points and
  for programs that embed the library alike. The database is owner-only (0600
  file, 0700 directory) and pruned by `FW_OBS_RETENTION_DAYS` and
  `FW_OBS_DB_MAX_BYTES`. `get_observability_sink()` no longer takes
  `entry_point`, and returns `None` only when the store cannot be opened.
- The `search_memory` input bound has no tuning override; it is derived from the
  search model's context window only.

### Fixed

- A malformed model reply that arrives after a tool has already run now fails
  the turn instead of silently re-running the whole trajectory. The re-run could
  repeat the side effects of the commands already executed and leave the
  archived evidence out of step with the answer.
- A known command name followed by a newline or a tab is now recognised by the
  known-name guard and by the owning context's exact match, not only when the
  name is followed by a space.
- A command whose name has capital letters is matched by its own context's
  exact match whatever case it is typed in, and is never refused by the
  known-name guard as belonging to another context. The guard compared a
  lowercased name against the context's command names as spelled, so it named
  the current context as the foreign owner.
- A failure while releasing the previous turn's process-local evidence no
  longer aborts the turn that is starting.
- A reply to `you_misunderstood` that matches none of the current context's
  commands now lists what can be done there. It used to raise
  `KeyError: 'what can i do?'`, because the fallback it substitutes was
  registered only for the ambiguity clarification stage.

### Removed

`FW_OBSERVATION_OFFLOADING`, `FW_ANSWER_REHYDRATION`, `FW_OFFLOAD_HANDLE_ARCHIVE`,
`FW_EAGER_ARTIFACT_VALIDATION`, `FW_MAX_FORCED_REPLANS`, `FW_OBS_MAX_ATTR_BYTES`,
`FW_OBSERVABILITY`, `FW_OFFLOAD_EVENTS`, `FW_OFFLOAD_EVENT_BUFFER_MAX`,
`FW_SEARCH_OBSERVATION_MAX_BYTES`, `FW_OFFLOAD_EVIDENCE_PRESERVATION`,
`FW_OFFLOAD_SEAL_GRACE_SECONDS`. Setting any of them has no effect.

### Migration

- `LLM_OBSERVATION_SEARCH` is the recommended setting for the model that answers
  `search_memory`. When it is unset, search runs on `LLM_AGENT`, which also
  fixes the budget the evidence is cut to, since that budget is sized from the
  search model's own context window.
- `LITELLM_API_KEY_OBSERVATION_SEARCH` is the recommended credential for that
  role. When it is unset, search uses the credential configured for `LLM_AGENT`.
- Offloading evidence now lives in `observability.sqlite3`. An evidence file
  left by an earlier build beside it (`observability.sqlite3.offload-handles.sqlite3`,
  its write-ahead-log files and any `.preserve` marker) is deleted the first time
  the store opens; its contents are not migrated.
- Experiment evidence is no longer preserved: Clear conversations, and
  forgetting an experiment run's channel, erase its offloading evidence like any
  other turn's.
- Recording can no longer be turned off. A program that embeds the library gets
  the same owner-only, pruned record as the entry points; to keep it elsewhere,
  set `FASTWORKFLOW_STATE_ROOT`. A program that builds its own execution context
  should open a sink with `get_observability_sink(workflow_path)` so the prune
  that bounds the file runs.
- Readers of the old `FW_OFFLOAD_EVENTS` JSONL file should read
  `ObservabilityStore.offload_events(turn_key=..., channel_id=..., kind=...)`.
- To correct the `search_memory` input bound, set `FW_MODEL_CONTEXT_TOKENS` or
  point `LLM_OBSERVATION_SEARCH` at the intended model.

### Known limits

Offloading and search ship with documented limits rather than silent ones: what
a turn resumed in another process reads, which part of the evidence is stored
unredacted, when pruning runs and what an embedding program must do for it to
run, the worst-case agent work one turn can cost, and several places where a disclosure
line can push a bounded input a few hundred bytes past the budget it reports.
They are listed in
[Retention, redaction and known limits](docs/observation_search.md#retention-redaction-and-known-limits);
read that section before sizing a deployment or writing a retention policy.
