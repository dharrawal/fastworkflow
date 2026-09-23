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
  command lives.
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

### Fixed

- A malformed model reply that arrives after a tool has already run now fails
  the turn instead of silently re-running the whole trajectory. The re-run could
  repeat the side effects of the commands already executed and leave the
  archived evidence out of step with the answer.
- A known command name followed by a newline or a tab is now recognised by the
  known-name guard and by the owning context's exact match, not only when the
  name is followed by a space.
- A failure while sealing or releasing the previous turn's evidence no longer
  aborts the turn that is starting.

### Removed

`FW_OBSERVATION_OFFLOADING`, `FW_ANSWER_REHYDRATION`, `FW_OFFLOAD_HANDLE_ARCHIVE`,
`FW_EAGER_ARTIFACT_VALIDATION`, `FW_MAX_FORCED_REPLANS`, `FW_OBS_MAX_ATTR_BYTES`.

### Migration

- `LLM_OBSERVATION_SEARCH` is the recommended setting for the model that answers
  `search_memory`. When it is unset, search runs on `LLM_AGENT`, which also
  fixes the budget the evidence is cut to, since that budget is sized from the
  search model's own context window.
- `LITELLM_API_KEY_OBSERVATION_SEARCH` is the recommended credential for that
  role. When it is unset, search uses the credential configured for `LLM_AGENT`.

### Known limits

Offloading and search ship with documented limits rather than silent ones: where
the observation sidecar lives and when its rows are sealed, which pruning a
program that embeds the library with observability off has to do itself, the
worst-case agent work one turn can cost, and several places where a disclosure
line can push a bounded input a few hundred bytes past the budget it reports.
They are listed in
[Retention, redaction and known limits](docs/observation_search.md#retention-redaction-and-known-limits);
read that section before sizing a deployment or writing a retention policy.
