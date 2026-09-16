# Answer coverage statement (`ido-8ps.22`, `ido-8ps.23`)

Two measured failures share one shape: the writer says more about the world than
the run earned.

* **`ido-8ps.22`** — D1 (`exp-ido-gqv-2-20260915T225309`) closed slots with
  *"no finding description containing Christopher Hubbard was available"* after
  searching finding **labels** and never running `show_affected_entities`. An
  assertion of absence is a claim about the platform; the run only ever earned a
  claim about itself.
* **`ido-8ps.23`** — neutral-v2 (`exp-ido-8ps-20-20260915T221729`): the two
  attempts that ended at the iteration limit completed their per-person tables
  from queries never issued; the three that finished normally fabricated nothing.
  A writer who is not told the run was cut short writes as though it was not.

`answer_rehydration` (`ido-8ps.18`) gave the extractor the evidence. This tells
it what the evidence does **not** contain.

## The block

One paragraph, prepended to the extract input — **first**, because
`_format_trajectory` renders the trajectory keys in order and a coverage rule
read after 60 KB of evidence is a rule read too late.

```
Coverage of this run: the loop ended normally. These named items from the
request appear in no retrieved observation: Christopher Hubbard. For each of
them report "not retrieved" and nothing else - no value, no unavailability, no
absence. These named items of the request DO appear in this run's observations:
Alan Cooper; Brandon Miller. Every other named item of the request WAS
retrieved: it appears in this run's observations and must be reported from them.
Do not write "not retrieved", "not available", "no data", or any other statement
of absence about an item that is not named in the unobserved list above. For
items that appear, report only what the observations show.
```

**Sentences four and five are `ido-8ps.24`.** The first version of the block
named the unobserved set and then said only "For items that appear, report only
what the observations show". In the D3+D4 cell
(`exp-ido-gqv-5-20260916T035444`) one attempt read that as licence to write
"not retrieved" against three identity uids that were in its own holder pages:
naming a set is not the same claim as saying what the REST of the set is. The
block now says what the rest of the set is, names it where the names fit
(`OBSERVED_LIST_MAX_BYTES`, 1,024 bytes, after which the rule stands alone), and
forbids the phrasing explicitly. Only kinds that are ever INSTRUCTED as "not
retrieved" are listed back, so the two lists partition one set — a quoted
request phrase is measured and never instructed, in either direction.

Exhausted instead:

```
Coverage of this run: the loop ended at the iteration limit after 96 steps. …
```

Nothing to name:

```
… appear in no retrieved observation: none. …
```

It is stored under the key `coverage_statement` — deliberately not an
`observation_` key: it is a statement **about** the run, not a tool result.

## Where the two halves come from

### The request

The agent's `user_query` is the refined request with a generated plan appended
under a fixed marker (`build_query_with_next_steps`). Everything before
`"\n\nExecute these next steps:\n"` is the request; the plan is a model's
paraphrase, and a name a planner invents is not a named item of the request.

Named items, by regex, in the order they are written:

| kind | rule |
|---|---|
| `name` | maximal runs of **two or more** consecutive capitalised tokens. A lowercase token breaks the run; so does the punctuation that ends a token, so `Alan Cooper, Anna Garcia` is two names. A run that starts a sentence loses its first token **when two or more remain** — `Two Active Directory rights` is about *Active Directory*; `Christopher Hubbard is one of…` keeps both, because a bare surname is a worse handle than the name. |
| `quoted` | `'…'`, `"…"`, `“…”`, `‘…’`. A single quote opens only after whitespace or an opener, so the apostrophe of `this quarter's` never opens a quotation. |
| `uid` | an unbroken hex run of 16–64. |
| `email` | an address. |

Deduplicated on the normalised form, first spelling kept, at most 64 items,
3–120 characters each. No model, no dictionary, no workflow lookup: the same
request always yields the same list.

### What the run retrieved

Two sources, both stores, both read-only:

1. the **archived text** of every execute observation of the turn, plus the
   context clause recorded for its alias (`ido-8ps.13`). The archive keeps the
   command *response*, never the command, so a name the agent merely **typed
   into a query** can never make that name look retrieved;
2. every **stored row** behind every result handle the turn declared — the rows
   `answer_rehydration` puts in front of the extractor, whole.

Search answers (`O5#a1`) are excluded from both: they are a model's summary and
they repeat the agent's own question.

Presence is a substring test on one comparison form: **NFKC, casefolded,
whitespace collapsed**. That is all it is.

The haystack is deliberately a **superset** of what the extract prompt can hold:
when the rehydration budget leaves an alias as a pointer, its evidence still
counts as retrieved. The error that matters is telling a writer that something
was never retrieved when it was, and this cannot make it.

## The two guards

**Quoted phrases are measured, never instructed.** `INSTRUCTED_KINDS` is
`{name, uid, email}`. The pinned card quotes its control as
`'Active contractor identities whom manager left'` while the catalogue's label is
`Contractor whom manager left`; the exact phrase appears in no observation of 61
of the 65 replayable attempts, **including the 22 in which that branch was judged
correct**. A quoted phrase in a request is narrative framing; a name, a uid and an
address are handles the workflow itself prints. Unmatched phrases are reported in
the event as `phrases_unmatched` and never appear in the block.

**The list requires a complete archive.** Every alias on the extractor's
trajectory must be in the archive. Persist-before-label (`ido-986.14.8`) archives
every execute observation, so this holds by construction on the accepted stack;
where it does not — an unwritable archive, a pre-A2 configuration — the run still
gets the exhaustion sentence, the block names nothing, and the event records
`complete: false` with the reason. Without this guard the offline replay would
have exposed 13 judged-**correct** slots; with it, zero.

## What it does not do

* **No model chooses anything.** No ranking, no selection, no second LM call.
* **Nothing is invented.** An item instructed as "not retrieved" is named
  verbatim from the request, and nothing is said about what it would have been.
* **Rehydration is unchanged.** Coverage runs immediately after it, on the copy
  it returned, and adds exactly one key.
* **The ReAct loop's trajectory is never touched**, and the archive and handle
  store are read-only. Routing, paging, offloading, auto-navigation and the 28 KB
  packed target all behave exactly as they did.
* **The post-check is measurement, never a gate.**

## The post-check

After the extract call returns, the finished answer is scanned for
unavailability/absence phrasing within `CLAIM_WINDOW_CHARS` (200) of each item
the block named, and the counts are recorded. It never edits an answer, retries a
call or fails a turn: `ido-8ps.22` asked for a number, not a guard.

In the offline replay, 59 such claims sat beside an item that appears in no
observation, every one of the 45 named items was mentioned somewhere in its
answer, and only 3 were already marked "not retrieved". That is the number to
move.

## The flag

| Variable | Default | Meaning |
|---|---|---|
| `FW_ANSWER_COVERAGE` | `0` | `1`/`true`/`yes`/`on` turns the statement on. Off, the extract call receives byte-for-byte what it received at `4832b3c` (the `ido-8ps.18` stack) — the same module, the same trajectory *object*, the same truncation fallback. |

Read **env file first, then the process environment**, the rule
`auto_navigation` and `answer_rehydration` use
(`answer_rehydration.env_value`): `fastworkflow.get_env_var` short-circuits on
its default before consulting `os.environ`, so a variable exported into the
process but absent from the workflow env file would otherwise read as the
default.

`tests/test_answer_coverage.py::ExtractHook::test_flag_off_is_byte_identical`
asserts the default.

## Events

Recorded through `observation_offloading.state.record_event`, so they land in the
same `FW_OFFLOAD_EVENTS` file every other measure does.

| Event | Carries |
|---|---|
| `coverage_statement` | `exhausted`, `steps`, `entities_total` / `entities_observed` / `entities_unobserved`, `observed`, `unobserved`, `entity_kinds`, `phrases_total`, `phrases_unmatched`, `complete`, `incomplete_reason`, `archived_observations`, `aliased_executes`, `statement_bytes`, `haystack_bytes`, `request_bytes` |
| `coverage_post_check` | `answer_bytes`, `unobserved_total`, `unobserved_mentioned`, **`unavailability_claim_on_unobserved`**, `not_retrieved_on_unobserved`, `silent_on_unobserved`, `per_item`, and the `ido-8ps.24` direction: `observed_total`, `observed_mentioned`, `unavailability_claim_on_observed`, `not_retrieved_on_observed`, **`misuse_on_observed`**, `per_observed_item` |
| `coverage_failed` | the exception type and detail; the extract call then runs on the trajectory it was handed |

## Where it is wired

`fastWorkflowReAct._cover_for_extract` is the single place the flag is read and
the copy is built, called from `_extract_prediction` /
`_async_extract_prediction` immediately after `_rehydrate_for_extract`. Every
extract call site therefore goes through it, including
`StructuredContinuationReAct._finish_prediction` — the one extract call of a
segmented turn, and the site the measured configuration uses.

`truncate_trajectory` skips `coverage_statement` when it drops "the oldest tool
call information": the coverage block is the one key whose whole job is to be
read. With the flag off the key never exists, so that line selects exactly the
keys it always did.

## Related

* `docs/answer_rehydration.md` — the evidence this statement is about.
* `docs/observation_search.md` — the archive, the `O` namespace, `search_memory`.
* `evaluation/artifacts/result-search/honesty-replay.md` (IDO) — the offline
  replay over 88 stored attempts that set both guards.
