# Changelog

Releases before 3.4.0 were announced in their merge-commit subjects
(`feat: v3.2.0 — observability store, chatbot debug UI, …`) and are recoverable
with `git tag` and `git log --first-parent main`. This file starts at 3.4.0; it
does not backfill them.

## 3.4.0 — result search

**Observation offloading and the answer-time behaviours become the framework's
behaviour for every workflow.** They are no longer modes a deployment opts into:
there is no `FW_OBSERVATION_OFFLOADING`, no `FW_AUTO_NAVIGATION`, no
`FW_ANSWER_REHYDRATION`, no `FW_ANSWER_COVERAGE`. A fastWorkflow tool agent
compacts its trajectory, keeps every observation reachable, navigates on a
declaration, and answers over evidence rather than pointers.

Full notes, migration and the experiment lineage:
[`docs/releases/3.4.0-result-search.md`](docs/releases/3.4.0-result-search.md).

### Added

- **Observation offloading** (`fastworkflow.observation_offloading`): canonical
  `O{n}` aliases on every execute observation, an eager SQLite archive, a packed
  trajectory target with offload labels, `search_memory` over one stored
  observation, and segmented continuation with forced replans.
- **Result handles** (`fastworkflow.result_handles`): one alias per listing,
  immutable stored pages, serialisable source descriptors, short cursor tokens,
  bounded page observations, literal filters and page references in artifacts.
- **Auto-navigation** (`fastworkflow.auto_navigation`): deterministic two-step
  dispatch on a known command name a foreign context owns, driven by an
  `enter_command` declaration on the context callback class, with a blocking
  clarification when the entry command needs a parameter. `fastworkflow train`
  reports the entry-contract check.
- **Context-instance line** (`fastworkflow.context_identity`): every execute
  observation names the context instance the command ran in.
- **Answer-time rehydration** (`fastworkflow.answer_rehydration`): the extract
  call reads the evidence behind labels, bounded listings and pages.
- **Coverage statement, roster nudge and evidence sentence**
  (`fastworkflow.answer_coverage`): the extractor is told what the run never
  retrieved; a `finish` action that never opened a named person of the request
  goes back to the loop once; and the block states, per subject, which other
  named items of the request that subject's own observations contain.
- **Attribution check** (`fastworkflow.answer_attribution`): a deterministic
  offline instrument for scoring an answer against the evidence per subject.
- **Context budgets** (`fastworkflow.context_budget`): one input — the model's
  context window — and every byte budget derived from it as a fixed fraction.
  `budget_provenance()` returns the input, its source and every budget.

### Changed

- **R1 known-name guard**: a known command name is never answered by a context
  that does not own it; the declining prediction carries a hint naming where the
  command lives.
- **R3 threshold separation**: `write_ambiguity_thresholds` is the single writer
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

### Fixed

- **`ido-8ps.29`**: a result-handle page fetched under a different context was
  stamped with that context's subject clause. The clause now follows the
  handle's declaring subject, and both the evidence sentence and the attribution
  check inherit it.
- **`ido-8ps.30`**: `react.py` never passed `report.unavailable` to
  `answer_coverage.post_check`, so the attempted-vs-never-attempted split
  recorded `unavailable_total = 0` in every run.

### Removed

`FW_OBSERVATION_OFFLOADING`, `FW_AUTO_NAVIGATION`,
`FW_AUTO_NAVIGATION_CANDIDATE_STEPS`, `FW_AUTO_NAVIGATION_CANDIDATE_MAX`,
`FW_ANSWER_REHYDRATION`, `FW_ANSWER_COVERAGE`, `FW_ROSTER_NUDGE`,
`FW_ANSWER_EVIDENCE`, `FW_EAGER_ARTIFACT_VALIDATION`,
`FW_OFFLOAD_HANDLE_ARCHIVE`, `FW_MAX_FORCED_REPLANS`, `FW_OBS_MAX_ATTR_BYTES`.
See the release notes for what replaces each one.
