# Answer-time rehydration (`ido-8ps.18`)

The ReAct loop and the extract step read the same `trajectory` for two different
jobs, and only one of them benefits from compaction.

The **loop** is a reader that can ask for more. When compaction swaps a 3 KB
listing for its 250 B offload label, the agent can still call `search_memory` on
that alias; when a listing command shows one bounded page of a result handle, the
agent can still call `fetch_result_page`. That is what holds the peak prompt at
~37k tokens against the control's ~80k, and `ido-8ps.17` measured it as a clear
win on trajectory correctness: 0 silent misroutes in 512 execute spans, 44 of 45
searches answered on valid handles, 479 of 479 observations archived.

The **extract step** is a reader that cannot ask for anything. It has no tools,
it runs once, and `trajectory` is its only evidence input
(`fastworkflow/utils/react.py`: `self.extract = ChainOfThought(fallback_signature)`).
At answer time the compaction that helped the loop is what leaves the writer
holding pointers, and `ido-8ps.17` measured that too: 35 of 260 deliverable slots
answered with "see Observation O23" about a listing sitting in that attempt's own
archive, and two attempts in ten that wrote up work which never ran.

So immediately before the extract call — and nowhere else — the extractor is
given its **own copy** of the trajectory with the evidence put back.

## What it does

| | Observation in the loop's trajectory | What the extractor's copy holds |
|---|---|---|
| **(a)** | an offload label: `Use search_memory tool to search inside Observation O18 returned by list_permissions…` | the raw archived observation for `O18`, re-printed with its alias line and the context clause recorded for it (`ido-986.14.9`, `ido-8ps.13`) |
| **(b)** | a bounded listing observation that declared a result handle | that observation, unchanged, plus every stored row behind its handle — whole rows, exactly as the producer rendered them |
| **(c)** | a `fetch_result_page` observation | that observation, unchanged, plus the stored rows behind the listing it paged |

Rows are appended per traversal: the base (unfiltered) walk first, then each
filtered traversal under the opaque query scope its cursor carried. Within a
traversal the rows are the distinct uids in first-seen order — exactly what
`_walk_records` serves the agent — so the answer's copy of the evidence cannot
disagree with the copy the pager showed.

## What it does not do

* **Nothing is chosen by a model.** There is no ranking, no selection, no
  summarisation, no second LM call. Every byte added was produced by a command in
  this turn and stored under a digest.
* **Nothing is invented.** An alias with no archived text is left as its label and
  reported `unresolved`; a context clause that was never recorded prints the plain
  A1 line rather than a guess; a filter literal is never reconstructed from the
  query-scope digest.
* **The ReAct loop's own trajectory is never touched.** `rehydrate` returns a
  copy. The loop keeps the trajectory it ran on, the turn record is unchanged, and
  the 28 KB packed target, offloading, paging, routing and auto-navigation all
  behave exactly as they did.
* **The archive and the handle store are read-only here.** No page is fetched, no
  backend is called, no row is written.

## The budget

The extraction budget bounds the UTF-8 bytes of the whole extractor
trajectory. It is a fraction of the model's context window — 15625/32768 of it,
which is **250,000 bytes** at the 131,072-token window every accepted run used,
the size of the control's ~80k-token answer-time prompt. See
[`docs/context_budget.md`](context_budget.md) for the one input and the whole
table; `FW_ANSWER_REHYDRATION_MAX_BYTES` remains as a tuning override. The walk
goes **most recent first** and **stops at the first
replacement that would not fit**; everything older stays as it is.

Every alias left that way is named in one deterministic line appended to the copy
under the key `answer_rehydration_note`:

```
Not rehydrated for the answer (evidence exists under these observations): O5, O9, O14
```

Ascending by execute ordinal, always the same line for the same run. It exists so
the extractor can report those slots as **unresolved** rather than guessing at
them — the opposite of the pointer answer, which claims the evidence was seen.

A handle whose rows a more recent observation already carries is not rehydrated
twice: the newest carrier holds the block and the older observation is left alone,
because repeating the rows would spend the budget on bytes the extractor already
has.

If the extract call still overflows the model's context window, the existing
truncation fallback runs unchanged — on the copy, never on the loop's trajectory —
and the run records `rehydration_overflow`.

## There is no flag

`ido-pyw.1` removed `FW_ANSWER_REHYDRATION`. Rehydration is what the extract
step does, for every workflow: the loop keeps its compacted trajectory and the
writer gets the evidence behind it. A run with nothing offloaded and nothing
paged rehydrates nothing and its extract call is the call it always was — the
rule is the trajectory's content, not a setting.

| Variable | Default | Meaning |
|---|---|---|
| `FW_ANSWER_REHYDRATION_MAX_BYTES` | derived (**250,000** at a 131,072-token window) | Tuning override on the extraction byte budget. Below 4,096 or unparseable, the derived budget stands and a warning is logged. |

It is read **env file first, then the process environment**:
`fastworkflow.get_env_var` short-circuits on its default before consulting
`os.environ`, so a variable exported into the process but absent from the
workflow env file would otherwise read as the default — and, unlike the
pre-`ido-pyw.1` readers, one written into the workflow's own `fastworkflow.env`
now takes effect.

## Events

Recorded through `observation_offloading.state.record_event`, so they land in the
same `FW_OFFLOAD_EVENTS` file every other measure does.

| Event | Carries |
|---|---|
| `rehydration_started` | `budget_bytes`, `bytes_before`, `scope_id` |
| `rehydration_finished` | `bytes_before`, `bytes_after`, `bytes_added`, `rehydrated_labels` / `rehydrated_listings` / `rehydrated_pages`, per-alias `{alias, kind, added_bytes}`, `dropped_aliases`, `unresolved_aliases`, `stopped_on`, `extract_prompt_tokens`, `extract_duration_ms`, `rehydration_overflow` |
| `rehydration_overflow` | how many times the fallback truncated, the budget, `bytes_after` |
| `rehydration_failed` | the exception type and detail; the extract call then runs on the plain trajectory |

`extract_prompt_tokens` is read from the LM history when it is available. History
can be disabled or carry no usage block; the measure is then simply absent, never
estimated.

## Where it is wired

`fastWorkflowReAct._extract_prediction` (and `_async_extract_prediction`) is the
single place the copy is built. Every extract call site goes through it:

* `fastworkflow/utils/react.py` — `forward` (agent-selected finish and the
  iteration ceiling), `resume` (an `ask_user` continuation), `aforward`;
* `fastworkflow/observation_offloading/continuation.py` —
  `StructuredContinuationReAct._finish_prediction`, which is the one extract call
  of a segmented turn and therefore the site the measured configuration uses.

A failure anywhere in the copy — an unreadable archive, a broken handle store —
falls back to the plain call on the trajectory object it was handed, and records
`rehydration_failed`. An answer over pointers is worse than one over evidence
and far better than no answer.
`tests/test_answer_rehydration.py::ExtractHook::test_a_broken_store_costs_the_evidence_and_not_the_answer`
asserts exactly that.

## Related

* `docs/observation_search.md` — the archive, the `O` namespace, `search_memory`.
* `docs/result_handles.md` — declarations, stored pages, cursors, filtered
  traversals.
