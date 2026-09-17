# Answer coverage statement, roster nudge and evidence sentence (`ido-8ps.22`, `ido-8ps.23`, `ido-8ps.27`, `ido-8ps.28`)

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

**The last sentence before the closer is `ido-8ps.28`.** After the
observed-items rule the block carries one further sentence:

```
… not named in the unobserved list above. Evidence by subject - for each named
item of the request that this run made the subject of a command, the OTHER named
items of the request that appear in that subject's own observations: Alan
Cooper: Right A, Right B; Alisha Ochoa: Right B. For items that appear, report
only what the observations show.
```

It answers a failure that is neither retrieval nor coverage: the run retrieves
the right rows for a subject and the answer still credits that subject with a
property its own rows do not carry, copied off the request's premise. The
attribution check (`fastworkflow/answer_attribution.py`) measures that after the
fact and is blind to every answer that asserts the property without naming it;
asked **forward**, of the evidence alone, the same question has no such blind
spot.

Three rules it does not bend:

* **Positive half only.** What a subject's evidence LACKS is never stated. A
  bounded portrait or an unpaged listing would make "not in the evidence" a
  claim about the run dressed as a claim about the world — the `ido-8ps.22`
  false absence `ido-8ps.24` repaired. A subject whose evidence contains none of
  the other items is therefore **not listed at all**, never listed empty.
* **No conclusion is instructed** (the `ido-8ps.24` pattern). It does not say to
  drop a claim, to prefer the evidence, or to check anything. It says what each
  subject's observations contain, and stops.
* **Bounded and whole-subject.** `EVIDENCE_LIST_MAX_BYTES` (1,024, its own
  constant) caps the list; subjects are taken in order until it is reached and
  the remainder is COUNTED (`; and N more subjects`), because a half-written
  subject would read as a short list for that subject.

A subject is one the run stamped a command against — the same per-observation
context clause the roster nudge reads, but per alias rather than concatenated,
through `answer_attribution.observations` / `subject_evidence`. There is no
third reader.

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
| `name` | maximal runs of **two or more** consecutive capitalised tokens. A lowercase token breaks the run; so does the punctuation that ends a token, so `Alan Cooper, Anna Garcia` is two names. A run that starts a sentence loses its first token **when two or more remain** — `Two Active Directory rights` is about *Active Directory*; `Christopher Hubbard is one of…` keeps both, because a bare surname is a worse handle than the name. `ido-jf6`: a possessive marker (`'s`, `’s`, or a bare apostrophe after an `s`) is stripped before the capitalisation test and **ends** the run, so `Alan Cooper's Active Directory rights` is *Alan Cooper* and *Active Directory*, not one item; `/`, `—` and `–` separate two items; and a run that starts a sentence with an imperative verb from a small explicit list (`list`, `show`, `compare`, `audit`, …, none of which is also an ordinary given name) loses that verb however short the run is, so `List Identities` and `Compare Alan` name nobody. |
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

## The roster nudge (`ido-8ps.27` (a))

The statement tells the *writer* what the run never retrieved. The nudge tells
the *loop*, once, while it can still do something about it.

`ido-8ps.27` measured runs that called `finish` with named people of the request
they had never made the subject of any command — not retrieved and not refused,
simply not reached — and then wrote a table about them. So at the one place a
`finish` action is recognised, before the answer is extracted,
`fastWorkflowReAct._roster_nudge` replaces that step's `"Completed."`
observation with a bounded note and lets the loop continue:

```
Harness check before this turn ends: these named items of the request were never
the subject of a command in this run: Brandon Miller; Barbara Sanchez. Open each
one and report what its own data shows, then finish.
```

The rules, all deterministic, all in `answer_coverage.build_nudge`:

* **At most one per turn.** `_roster_nudges_fired` is checked before anything is
  read, so a second `finish` ends the turn as it always did.
* **Never on exhaustion.** The note refuses below `NUDGE_MIN_ITERS_LEFT`
  iterations left, counted as what the agent would still have *after* spending
  this step on the note. A turn with no room is a turn the note cannot help.
* **Never an `ask_user` round.** It is an observation on the agent's own step;
  nothing reaches the user, and the turn is not suspended.
* **Bounded** at `NUDGE_MAX_BYTES`, with what it cut counted in the event.
* **Subjects, not mentions.** "Made the subject of a command" is the recorded
  context clause (`ido-8ps.13`), not a substring of a listing row: appearing in
  someone else's holder page is not being opened.
* **A failure leaves the loop alone.** The exception is recorded as
  `roster_nudge_failed` and the `finish` ends the turn.

It fires whether or not the note is produced, so `roster_nudge` is on the event
log either way, with `fired`, the `reason` when it did not, `subjects_total`,
`subjects_missing`, `subjects_named`, `iterations_left` and the byte counts.

## The evidence sentence (`ido-8ps.28`)

The sentence shown in **The block** above, stated per subject, answers a failure
that is neither retrieval nor coverage: the run retrieves the right rows for a
person and the answer credits that person with a property their own rows do not
carry, copied off the request's premise.

* **Positive half only.** What a subject's evidence *lacks* is never stated. A
  bounded portrait is not a complete one, so "X's observations do not contain Y"
  would be exactly the false absence `ido-8ps.22` exists to prevent. A subject
  with nothing to state is not printed at all — which is why a run whose only
  subject has nothing to state renders the block byte for byte as the
  `ido-8ps.27` stack did.
* **No conclusion is instructed.** The sentence says what the evidence contains
  and stops; it never tells the writer what to do with it.
* **Subjects come from the recorded context clause**, through
  `answer_attribution.observations`, so `ido-8ps.29`'s repair to the clause on a
  re-paged result handle reaches this sentence with no change of its own.
* **Bounded** at `EVIDENCE_LIST_MAX_BYTES`: whole subjects only, and the rest are
  counted rather than half-printed.
* **Not gated on `complete`.** The completeness refusal exists because "appears
  in no retrieved observation" would otherwise be a claim about the archive; a
  sentence that only ever says what IS in an observation cannot make that
  mistake, and on a partial archive it simply says less.

## There are no flags

`ido-pyw.1` removed `FW_ANSWER_COVERAGE`, `FW_ROSTER_NUDGE` and
`FW_ANSWER_EVIDENCE`. The coverage statement, the roster nudge and the evidence
sentence are what an answer-time extract call and a `finish` action get, for
every workflow. Each one is silent when it has nothing to say — a request with
no named items produces no unobserved list and no nudge, and a subject with no
other named item in its own observations is never printed — so what used to be
"the flag is off" is now the ordinary behaviour of a run with nothing to report.

`tests/test_answer_coverage.py::EvidenceInTheBlock::test_a_lone_subject_with_nothing_to_state_is_90a1565`
pins that statement, byte for byte, against the one the `ido-8ps.27` stack
produced.

## Events

Recorded through `observation_offloading.state.record_event`, so they land in the
same `FW_OFFLOAD_EVENTS` file every other measure does.

| Event | Carries |
|---|---|
| `coverage_statement` | `exhausted`, `steps`, `entities_total` / `entities_observed` / `entities_unobserved`, `observed`, `unobserved`, the `ido-8ps.27` (b) split `unavailable` / `never_attempted`, `entity_kinds`, `phrases_total`, `phrases_unmatched`, `complete`, `incomplete_reason`, `archived_observations`, `aliased_executes`, `statement_bytes`, `haystack_bytes`, `request_bytes`, and the `ido-8ps.28` sentence: `evidence` (every stamped subject with the items its own observations contain, including the empty ones the sentence may not print), `evidence_named`, `evidence_subjects`, `evidence_items`, `evidence_bytes` |
| `coverage_post_check` | `answer_bytes`, `unobserved_total`, `unobserved_mentioned`, **`unavailability_claim_on_unobserved`**, `not_retrieved_on_unobserved`, `silent_on_unobserved`, `per_item`, the `ido-8ps.27` (b) half `unavailable_total`, `unavailable_mentioned`, `unavailability_claim_on_unavailable`, `not_retrieved_on_unavailable` (wired in `ido-8ps.30`; they read 0 in every run recorded before it), and the `ido-8ps.24` direction: `observed_total`, `observed_mentioned`, `unavailability_claim_on_observed`, `not_retrieved_on_observed`, **`misuse_on_observed`**, `per_observed_item`, and the `ido-8ps.28` measure: `evidence_subjects_total`, `evidence_subjects_mentioned`, **`evidence_claims_listed`** / **`evidence_claims_unlisted`** (per subject, the request's named items written within `CLAIM_WINDOW_CHARS` of it, split by whether the sentence listed them for that subject), `per_evidence_subject` |
| `coverage_failed` | the exception type and detail; the extract call then runs on the trajectory it was handed |
| `roster_nudge` | `fired`, `reason`, `entities_total`, `subjects_total`, `subjects_missing`, `subjects_named`, `iterations_left`, `text_bytes`, `clause_bytes` |
| `roster_nudge_failed` | the exception type and detail; the `finish` action ends the turn |

## Where it is wired

`fastWorkflowReAct._cover_for_extract` is the single place the copy is built,
called from `_extract_prediction` /
`_async_extract_prediction` immediately after `_rehydrate_for_extract`. Every
extract call site therefore goes through it, including
`StructuredContinuationReAct._finish_prediction` — the one extract call of a
segmented turn, and the site the measured configuration uses.

The roster nudge is wired at the one place a `finish` action is recognised in
`fastWorkflowReAct._run_loop`, after the `"Completed."` observation is written
and before the answer is extracted.

`truncate_trajectory` skips `coverage_statement` when it drops "the oldest tool
call information": the coverage block is the one key whose whole job is to be
read.

## Related

* `docs/answer_rehydration.md` — the evidence this statement is about.
* `docs/observation_search.md` — the archive, the `O` namespace, `search_memory`.
* `evaluation/artifacts/result-search/honesty-replay.md` (IDO) — the offline
  replay over 88 stored attempts that set both guards.
