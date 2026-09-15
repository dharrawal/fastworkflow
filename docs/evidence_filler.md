# Evidence filler (`FW_EVIDENCE_FILLER`)

The final answer of a turn is produced by the ReAct **extract** step, which reads
the trajectory and nothing else: it cannot call `search_memory`, it cannot call
`fetch_result_page`, and it never sees the archive. Once observations are bounded
(C1) and offloaded, it holds handles where it used to hold values — and it fills
the gap from memory. Four consecutive result-search experiments ended with the
same finding: retrieval was healthy and the report was wrong, in the same four
shapes.

* **Cite instead of report** — "Alan Cooper's rights: see Observation O53".
* **"Not retrieved"** over data the same turn had retrieved.
* **A premise restated** against the run's own table.
* **A placeholder identifier**, sometimes labelled as one.

The filler answers the request's items *from the evidence* before the extract
step runs, validates every answer against the stored text, and hands the
extractor a worksheet it is asked to report from.

It is off by default. `FW_EVIDENCE_FILLER=1` turns it on; with it off nothing in
this document happens and the extract prompt is byte for byte the one the
previous revision produced — the input field does not exist.

## What runs, and when

| | |
|---|---|
| Trigger | the step on which the agent selects `finish`, and every forced replan |
| Route | `LLM_OBSERVATION_SEARCH` (the search model), temperature 0 |
| Reads | the turn's archived observations and stored result pages |
| Never reads | the agent's thoughts, the backend, another turn, another scope |
| Joined | immediately before the extract step |

Two model steps:

1. **Decomposition** (one call). The user's request → short item labels. The
   planner's appended step list is cut off first (`user_request`): the plan is
   how the agent chose to work, and decomposing it would fill the worksheet with
   tool steps. No task-specific schema exists anywhere in the framework — the
   items come from the request.
2. **Fill** (one call per group of items). For each item, a value copied
   verbatim out of the evidence with the `O` handle it came from, or
   `unresolved` with a reason.

Items whose selected evidence is *identical* are answered in one call; groups
are independent and run in parallel (`FW_EVIDENCE_FILLER_WORKERS`, default 4).
Items are never merged into a call that would widen the per-item budget.

### The evidence an item is given

Deterministic and free — no model chooses what to read, so no model can choose
to read nothing and answer anyway. Every archived observation and every stored
result page of the turn is cut into pages of `DEFAULT_PAGE_BYTES` (4 KB) on row
boundaries, and the best `SEARCH_MEMORY_MAX_PAGES` (3) are fed: **the same
budget `search_memory` works under**, at most 12 KB per item. Pages are ranked by
how many of the item's own words they carry, ties broken by the newest
observation first.

Stored result pages matter here more than anywhere else: the rows a bounded
listing did *not* print live only in `result_handle_pages`, and that is exactly
the evidence the extract step lost when listings became bounded. A page-sourced
value cites its listing's `O` handle plus a page token (`O12#p200`), which — like
a bounded search answer's `O12#a1` — names a record and can never be passed back
as a searchable handle.

## The validation rule

Deterministic, mandatory, and in code rather than in a prompt. A filled value
survives only when

1. **the alias it cites is in this turn's printed `O` namespace** — the execute
   ordinals A1 prints on the observations themselves. Archived records that are
   not execute ordinals (`O12#a1`) are not in it. Otherwise the reason is
   `alias_not_printed`.
2. **the value occurs literally in the archived text of that observation**, or in
   a stored page filed under it, after NFKC, space-like and zero-width repair,
   whitespace collapse and casefold — the repair `normalize_literal` performs and
   the case-insensitive comparison `_filter_records` uses, with the constants
   imported from `result_handles` so the two cannot drift. Otherwise the reason
   is `value_not_in_cited_observation`.

A value that fails either check is **dropped**: the item is reported to the
extractor as `unresolved` with the reason, and the discarded text appears only in
the event log. Nothing downgraded ever reaches the extractor.

Two deliberate details:

* The wildcard stripping `normalize_literal` does is **not** applied. `%`, `_`
  and `*` are removed from a *filter* because the portal reads them as LIKE
  wildcards with no escape. A value is not a pattern, and
  `Active Directory_Cloud Administrator` is an identifier in this tenant:
  removing its underscore would let a wrong value match a right one.
* Matching is against the **whole** cited observation, not against the pages the
  item happened to be fed. The rule is about the evidence, not about the prompt.

This is `ido-8ps.5` generalised from handles to values, and it is
what separates this from the `ido-986.6.14` partial-answer arm, which fabricated
identifiers in 3 of 3 attempts because nothing checked its output.

## What the extractor gets

One additional input field on the extract signature, appended last:

```
verified_evidence:
the Cloud Administrator permission uid: 3e3d35f0… (Observation O2)
Alan Cooper's account uid: e8a0c3a1… (Observation O1, page O1#p200)
the collection that confers it: unresolved - no collection row was retrieved
```

The instruction ("report these values, report an unresolved item as unresolved")
lives in the **field description**, not in the signature docstring: `utils/react.py`
builds the ReAct predictor's prompt and the extract prompt from the same
`signature.instructions`, so a docstring change would alter the agent's own
prompt too. This experiment changes the extract step and nothing else.

Nothing else about the extractor changes, and the predictor the agent is built
with is never touched — the worksheet variant is a second predictor, built the
first time it is used.

## Failure, timeout and fallback

| situation | what happens |
|---|---|
| decomposition fails | no worksheet; `filler_failed`, then `filler_fallback`, and the extractor runs exactly as today |
| one fill call fails | only that group's items are `unresolved`, naming the exception |
| the deadline passes | remaining items are `unresolved`; what was filled still stands |
| the join times out | `filler_fallback` with reason `join_timeout`; the extractor runs as today |

"Runs exactly as today" is literal: the fallback path calls the predictor built
from the unmodified signature, so the prompt has no `verified_evidence` field.

## Observability

Events (`FW_OFFLOAD_EVENTS`):

* `filler_started` — trigger, request bytes.
* `filler_finished` — items total / validated / unresolved / downgraded,
  downgrades by reason, calls, prompt and completion tokens, cost, evidence
  bytes, worksheet bytes, latency, model, whether the deadline was passed.
* `filler_joined` — `join_wait_ms` (what the turn actually paid at the finish
  step) and `background_window_ms` (what the early start bought).
* `filler_failed`, `filler_fallback`, `filler_superseded` (a replan-time run
  still in flight when the finish-time one starts).
* `answer_used_worksheet` — for **each** validated value, whether the final
  answer contains it (the same normalised substring test), plus the rate and
  which of the cited aliases the answer names. "The extractor ignored the
  worksheet" is therefore a measurement, not an argument.

`agent_installed` carries `evidence_filler: true|false`, so a run measured with
the filler off is distinguishable from one measured before the filler existed
(where the key is absent, not false).

## Knobs

| variable | default | meaning |
|---|---|---|
| `FW_EVIDENCE_FILLER` | `0` | off / on |
| `FW_EVIDENCE_FILLER_TIMEOUT_S` | `60` | wall clock for one run, decomposition included |
| `FW_EVIDENCE_FILLER_WORKERS` | `4` | fill calls in flight |
| `FW_EVIDENCE_FILLER_MAX_ITEMS` | `40` | most items one decomposition may produce |
| `FW_EVIDENCE_FILLER_MAX_CALLS` | `16` | most fill calls per run; items past it are `unresolved`, named |

Read from the fastworkflow env file first and the process environment second,
the same order `auto_navigation` reads its flag.

## Known limits

* **Decomposition quality is unmeasured by the framework.** The items are
  whatever the model reads out of the request; nothing here knows what a good
  item is for a given card.
* **A replan-time worksheet is recorded, never read.** The finish-time run reads
  the same archive plus everything later segments added, so it supersedes it.
* **Strictly whole values.** A composed answer ("29 entitlements, two of them
  AD") is not a literal substring of anything and is downgraded. That is the
  intended bias: an unresolved row costs a deliverable, a fabricated one costs
  trust.
