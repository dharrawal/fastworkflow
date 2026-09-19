# Result handles

A listing command answers with every row it materialised and the ReAct
trajectory then carries all of it. [Observation search](observation_search.md)
moves that text out of the prompt after the fact; `fastworkflow.result_handles`
stores the listing itself, so the agent can ask for one page of it, or for the
rows that contain a name, without re-running the command or re-reading the whole
response.

Two calls: `declare` from the command that produced the listing, and
`fetch_page` from a command the workflow registers for fetching. fastWorkflow
registers no core fetch command — a core command joins every workflow's command
surface, its `what_can_i_do` output and its trained intent model, which would
make "a workflow that did not opt in is untouched" false for every workflow that
never opts in.

## Identity: the handle is the step's canonical `O` alias

A handle is the `O` alias of the execute step that declared it — the alias
[A1](observation_search.md#canonical-observation-handles) already prints on that
step's observation. There is no second agent-visible namespace: the identifier
the agent passes to `fetch_result_page` is the one it reads on the observation
and passes to `search_memory`.

`current_execute_alias()` resolves it from the live agent: ReAct writes
`tool_name_{idx}` before calling a tool and `observation_{idx}` after it
returns, so during a command the in-flight step is the last one with no
observation, and its ordinal is the number of `execute_workflow_query` steps in
`current_trajectory` (which truncation never removes). That is exactly the
ordinal `annotate_execute_observations` prints when the step completes.

Outside an agent step — a direct user command, offloading disabled — there is no
agent-visible namespace at all, and the declaration is filed under a `D{n}` key.
It is a store key, never an `O`, and no agent ever sees it.

A fetch call is itself an execute step with its own `O` alias. Its page
observation is immutable and archived under that alias like any other execute
observation, and the store records the link from the page alias to the listing
it paged: `parent_handle("O43") == "O42"`. Passing a page alias to `fetch_page`
resolves to the listing, so an agent that searched its way to a page can keep
paging from there. A page never redefines itself as the whole population — its
header reports the listing's counts, and its `parent_alias` says whose rows they
are.

## API

```python
from fastworkflow import result_handles

result_handles.declare(spec, *, source=None, scope=None, selected_store=None,
                       alias=None, parent_alias="", query_scope="",
                       cursor_position=0) -> dict

result_handles.fetch_page(handle, cursor=None, contains=None, *, scope=None,
                          selected_store=None, budget_bytes=None) -> ResultPage

result_handles.register_resolver(kind_or_workflow, resolver) -> None
result_handles.unregister_resolver(name) -> None
result_handles.registered_resolvers() -> tuple[str, ...]
result_handles.resolver_for(name) -> Callable        # raises ResultHandleError

result_handles.handle_declaration(handle, *, scope=None, selected_store=None) -> dict
result_handles.parent_handle(handle, *, scope=None, selected_store=None) -> str | None
result_handles.normalize_literal(text) -> Literal
result_handles.reset_result_handle_state() -> None

# Page tokens (ido-986.14.11). encode_cursor and decode_cursor keep their names
# and decode_cursor keeps its payload; both gained keyword arguments.
result_handles.encode_cursor(*, alias, query_scope, position, descriptor_sha256,
                             scope=None, selected_store=None) -> str
result_handles.decode_cursor(cursor, *, alias=None, scope=None,
                             selected_store=None) -> dict
result_handles.cursor_token(alias, tag, page) -> str          # "O7/f1p2"
result_handles.cursor_placeholder(alias, tag="", *, pages_at_most=0) -> str
```

`ResultHandleError` is raised for anything the calling command can act on: an
unknown handle (the message names the handles this turn does store), a page
token from another handle or another filter, a page token that is not a page
token or that this turn never issued (the message names the tokens it did), a
descriptor naming a resolver this process has not registered, a redeclaration of
a different query under a handle that already exists.

`ResultHandleSpec` is what the producing command declares:

| field | meaning |
| --- | --- |
| `kind` | what the rows are (`holder`, `member`, …) |
| `summary` | the producer's own first line |
| `items` | the rendered `uid  label` rows, exactly as the response carried them |
| `ordering` | the producer's ordering label |
| `total` | what the backend reported the whole population is |
| `source_complete` | whether `items` cover `total` |
| `page_size` | the producer's own page size, used as the packer's row budget floor. The size of a *backend* read is `SourceDescriptor.batch_size`, which is the adapter's business |
| `classification` | prompt classification of the row text (`user-text`) |
| `presentation` | whether the rows are deliverable output |
| `filters` | the producer's own filter arguments, for the record |

`items` are rendered rows and not structured fields on purpose: a literal filter
has to be able to find "Alan Cooper" in the row the agent actually read, so the
label must be *in* the row rather than beside it.

`declare` returns the payload a command puts in
`CommandResponse.artifacts`: `result_handle`, `kind`, `summary`, `ordering`,
`total`, `materialized`, `source_complete`, `page_size`, `classification`,
`presentation`, `filters`, `descriptor`, `descriptor_sha256`, `parent_alias`,
`query_scope`, `cursor_position`, `scope_id`, `declared`. A storage failure
never fails a command that produced real output: it records a
`result_handle_declare_refused` event and returns `declared: false`.

## The source descriptor

`SourceDescriptor` is the serialisable description of the query that produced
the rows. Nothing callable is stored: `resolver` names a resolver registered in
this process, and a stored descriptor that names an unregistered one is refused
by name — the stored rows stay readable, only continuation stops.

```python
SourceDescriptor(
    resolver="ido.relation-view",      # registered name, not a callable
    uid_field="identity__id",
    label_fields=("identity_displayname",),
    batch_size=25,
    filter_columns=("identity_displayname", "identity_surname"),
    state={                            # opaque: carried, never inspected
        "view": "ido_permissiondetail_identity",
        "params": {"scope": "85cde168…"},
        "start_offset": 0,
        "materialized": 25,
    },
)
```

Six fields, and the framework reads every one of them. (F1, `fix-iq53.2.5`)
There used to be fourteen: `view`, `params`, `role`, `extra`, `ordering`,
`timeslot`, `start_offset`, `materialized` and `count_only` are gone, because
a framework type that names a SQL view, an offset origin and a snapshot pin is
one workflow's query object rather than an adapter boundary. Those values did
not stop existing — they are in `state` now, under whatever keys the adapter
chooses, and nothing in `fastworkflow.result_handles` looks inside.

The B0 probe (`evaluation/artifacts/result-search/b0-probe.md`, bead
`ido-gqv.6`) still shapes what the descriptor **cannot** say:

* **There is no `sort` field, and no field to put one in.** On
  `ido_groupDetail_identity` an explicit sort combined with offset paging
  silently drops members while returning exactly `total` rows — 20 of 540 at
  page size 40. The descriptor used to carry `ordering` with a constructor
  rule refusing any value but `unsorted-offset`; the field and the rule are
  both gone and the guarantee is stronger for it, because the framework
  cannot send an ordering it has no way to name.
* **There is no `timeslot` field either.** `IDO.timeslot` is `None` and the
  client drops the field rather than sending a null, so there is no pinned
  session timeslot to reproduce. The descriptor used to record the absence
  explicitly (`timeslot=None`, refusing anything else, bead `ido-986.14.1`).
  Recording that a read had no pin is the adapter's evidence to keep now, in
  its own `state`.

`filter_columns` stays a list of **names** and is not collapsed to a
`filterable` boolean (`fix-iq53.2.3`). The names are agent-visible: the page
header prints `filter_columns=` from them, and a complete zero names the
fields it searched rather than saying "the rendered rows" — on exactly the
page whose job is to stop a zero being over-read. A boolean cannot
reconstruct either, and recovering the names out of `state` would mean
inspecting the one field that has to stay opaque. Column names are query
vocabulary, not one backend's policy.

They must already be verified for the source. A filter sent without columns is
silently ignored by the portal and returns the whole scope, so the pair
"filter, no columns" must be impossible to emit; an empty `filter_columns`
means literal filtering is unsupported for the handle and is reported as such.

## Storage and retention

Five tables in the same SQLite file the observation archive uses, so a page and
the observation that showed it survive together. `observation_offload_handles`
is untouched.

`result_handle_declarations`, keyed `(scope_id, alias)`: `scope_json`, `kind`,
`summary`, `ordering`, `total`, `materialized`, `source_complete`, `page_size`,
`classification`, `presentation`, `filters_json`, `descriptor_json`,
`descriptor_sha256`, `columns_json` (verified column names and types, from the
first page), `sample_row_json` (one sample row from the first page),
`parent_alias`, `query_scope`, `cursor_position`, `declared_at`.

`result_handle_batches`, keyed `(scope_id, alias, query_scope, batch_index)`:
`limit_requested`, `source` (`producer` or `resolver`), `row_count`,
`backend_total`, `continuation_json`, `record_json`, `record_sha256`,
`fetched_at`.

`result_handle_walk_terminals`, keyed `(scope_id, alias, query_scope)`:
`terminal_batch_index`, `complete` (nullable), `distinct_uids`, `stop_reason`,
`recorded_at` — where a traversal ended and what the adapter decided about it, so
the end of a walk costs the source nothing twice.

`batch_index` is a framework-allocated monotonic ordinal counting from 0, and 0
is the producer's own page. `continuation_json` is the adapter's resume point for
that batch, stored verbatim; SQL NULL is how a rebuild reads "this walk cannot be
continued past here".

Both tables were renamed by F5a (`fix-iq53.2.9`), from `result_handle_pages`
keyed on `start_offset` and `result_handle_walks` carrying `terminal_offset` and
`count_only`. The old key made one backend's pagination scheme part of the
storage contract: an adapter that walks by cursor or keyset has no offset to key
on, and the framework had to invent one to have somewhere to put the rows. The
dropped `count_only` held the framework's own independent count, which it needed
while it did the coverage comparison itself; the comparison moved to the adapter,
so what is stored is the decision and not the working.

**A rename and not an `ALTER`, in both cases.** A store written before the change
keeps its old tables and still parses, and nothing writes to them again — which is
also why the walk table had to change its NAME to change its COLUMNS: `CREATE
TABLE IF NOT EXISTS` is a no-op against a file that already holds the old table,
so the old column set would survive invisibly and the first insert with the new
columns would fail on a file written yesterday. New table names are the *second*
line of defence for a historical store. The first is never opening one
read-write at all: `ResultHandleStore.open_readonly` (F5b, `fix-iq53.2.10`) skips
the create block and connects through `mode=ro`, because merely CONSTRUCTING the
ordinary store over a frozen artifact rewrites it — measured, a 135,168-byte
store became 192,512 bytes with five tables added and a different sha256.
"Additive" describes the schema delta, not the file delta.

`result_handle_cursor_tags`, keyed `(scope_id, alias, query_scope)`: `tag`,
`created_at` — the short tag that stands for a query scope on a handle.

`result_handle_cursors`, keyed `(scope_id, alias, tag, page)`, unique on
`(scope_id, alias, tag, position, descriptor_sha256)`: `query_scope`,
`position`, `descriptor_sha256`, `issued_at` — everything the agent-visible
token does not carry. Issuing is idempotent per position, so a resumption point
always prints the token it printed the first time, and the row is what lets a
token survive a hot-cache eviction or a restart.

Batches are **append-only and digest-verifiable**. The insert cannot overwrite and
the value returned is the read-back, so re-reading an ordinal that already
exists returns the stored batch and never writes a second row: a retry is
idempotent by construction, not by convention. A stored record whose payload no
longer matches its digest is refused rather than served.

**Retention.** This module deletes nothing. Declarations and pages live exactly
as long as the archive file that holds the turn's observations — that is what
makes a page reconstructable for evaluation after the live turn ended. The hot
cache (`FW_RESULT_HANDLE_HOT_MAX_BYTES`, 256 KB at a 131,072-token window,
oldest walk first) is a
*residency* bound and not a retention bound: every evicted row came from a
stored page and is rebuilt from SQLite on the next read.

**Scope.** Rows are keyed by the same `RuntimeHandleScope` the offload archive
uses (store identity, channel, experiment, task, attempt, turn). A handle
declared in one turn is not visible in another, and the same alias in two scopes
is two different listings. `state.scope_for_host()` is now the single
implementation of that scope, reached both from the ReAct loop and from a
command's own frame — a scope computed two ways would be two scopes the moment
either changed.

## Continuation: the batch walk

When a handle carries a descriptor, `fetch_page` continues the producing query
past the rows the command materialised. The framework allocates the batch
ordinals and carries the adapter's resume point; it does not compute a position in
the source, and since F3 (`fix-iq53.2.7`) it has no arithmetic that could. There
are **two callback shapes**, and an adapter tells them apart by type (F2,
`fix-iq53.2.6`):

```python
SourceRequest(descriptor, continuation=None, limit=0, contains=None)
TerminalRequest(descriptor, continuation=None, contains=None, distinct_uids=0)
```

`SourceRequest` asks for one batch and is charged against the per-fetch purse.
`TerminalRequest` fires once when the walk reaches its end, is never charged,
and is the only callback that may decide completeness. There is no
`count_only` mode flag: a request that turned itself into a reconciliation by
setting a boolean was one type doing two jobs, and an adapter had to branch on
it before it knew what it had been asked.

`continuation` is the adapter's **own** resume point, carried verbatim. The
framework stores it in `result_handle_batches.continuation_json`, hands it back on
the next batch, and never reads inside it — an offset means nothing to an adapter
that walks by cursor, keyset or one-item lookahead. `None` on the first batch of a
walk means "start wherever you start": which row that is, and whether it accounts
for the rows the producer already rendered, is the adapter's answer to give out of
`state`. The same is true of the origin rule — a filtered walk begins at the first
match rather than partway into the relation, because a backend applies `contains`
before it counts, and that is a fact about one backend's query composition rather
than about paging.

`TerminalRequest.distinct_uids` is the framework reporting its own dedup count,
because a completeness rule of the form "independent count == distinct uids" has
always consumed that number. There is deliberately **no** `batches_read`: the
framework knows it, no consumer on the adapter side was ever named for it, and the
number reaches the observability store on the `result_handle_reconciled` event
without crossing the boundary to get there.

A batch returns a mapping (or any object with the same attributes):
`{"rows": [...], "continuation": {...} | None, "total": int | None,
"columns": {name: type} | None}`.
`rows` are the source's own rows; the store renders each one as the producer
would — `uid  label` from `uid_field` and the first non-empty `label_fields`
entry — so a stored listing and its continuation are one sequence of rows to the
agent. The first page's column names and types and one sample row are written
into the declaration, once.

**A batch response may not claim completeness.** `complete` and
`incomplete_reason` on a batch response are ignored; the terminal callback
answers them or nobody does. A terminal returns `complete` (optional bool;
absent is incomplete) and `incomplete_reason` (optional; the adapter's own
typed word, passed through verbatim).

**An empty batch terminates the walk, regardless of any continuation
offered** (`fix-iq53.2.4`). This is the framework's one independent stop
condition and it outranks the adapter's offer. An adapter that answers "no
rows, but there is more" describes a walk with no end: the framework would ask
again, get nothing again, and spend up to `MAX_RESOLVER_CALLS_PER_FETCH` doing
it on every fetch for the life of the handle, with every page looking like
legitimate progress. Within one fetch the purse bounds it; across fetches
nothing does. The offer is therefore discarded at the boundary rather than
trusted and guarded against later, so that recognising the terminal by an
absent continuation and recognising it by an empty batch agree by
construction.

**`rows == total` is not a stop condition and is not a completeness proof.**
B0 measured a sorted offset walk on `ido_groupDetail_identity` returning
exactly `total` rows while 20 of 540 members were never shown.

**A batch that returns rows and no continuation also ends the walk, after those
rows** (F3, `fix-iq53.2.7`). An adapter that answers and offers no way to resume
has said this was its last batch; asking again would either restart a finished
traversal or invent a position in it. The rows it carried are kept — a terminal is
not a discard — and an adapter that can recognise its own last batch never has to
be asked for an empty one, which is one backend read saved per traversal. The two
stop conditions cannot disagree, because an empty batch's continuation is forced
to `None` at the boundary.

**Coverage is decided by the adapter, on the terminal callback, and recorded
rather than audited** (F4, `fix-iq53.2.8`). An empty batch ends the walk; it does
not prove the walk saw everything. When the walk ends the framework issues one
terminal callback, hands the adapter its own distinct-uid tally, and records what
comes back:

* `complete: true` — `source_complete`, or `matched_complete` for a filtered
  query, becomes true.
* `complete: false` — incomplete, with the adapter's own `incomplete_reason`
  passed through verbatim. IDO emits `countonly_mismatch`,
  `countonly_unavailable`, `countonly_error` and `offset_origin_not_zero`, and the
  page prints them exactly as it printed them when the framework owned them.
* **`complete` absent — nothing is settled.** The end is known and unjudged, so
  the callback is owed again and re-fires on the next fetch off the stored
  terminal, costing one callback and no batch read. The framework's own word for
  this is `completeness_not_claimed`, the single word F4 adds. A walk whose
  coverage nobody could decide keeps asking rather than freezing an answer nobody
  gave.

The framework never upgrades an unclaimed walk to complete, and it has no rule
that could: the comparison that used to live here — an independent `countOnly`
against the distinct uids — moved across the boundary with the vocabulary that
described its outcomes. A `count` reported alongside a decision is kept for the
page's prose and the `result_handle_reconciled` event and is compared against
nothing.

A walk that ends without a completeness claim keeps its stop reason **outside**
the `walk_can_continue` allowlist, so it serves every stored row and then stops:
intermediate pages of it still carry a cursor, because the remaining rows are
already stored and paging them costs the source nothing, and only the final page
offers none and prints `continuation=source-incomplete has_more=false`. The
traversal is never silently restarted.

Duplicates are dropped from the traversal sequence in first-seen order and the
raw page that carried them is stored whole, so a repeated row can never displace
one that has not been shown.

One `fetch_page` call reads at most `MAX_RESOLVER_CALLS_PER_FETCH` (8) backend
batches. Reaching that bound is `resolver_call_limit`: the page warns, shows what
it has, and the cursor resumes at the same batch, from the same resume point. It
is a bound on one call, never a cap on enumeration. A capped walk has not reached
its end, so it makes **no** terminal callback; the terminal is reached only
through a terminal batch and never through purse exhaustion. A resolver that raises is `resolver_error` — the
rows already stored are still served and the cursor still advances, because a
refusal is usually transient. A descriptor whose resolver this process has not
registered is `resolver_unavailable`: stored rows are served and no cursor is
offered, because it would not move.

## Filtering

`contains` is a literal, never a question. It is normalised (below), then:

1. if the base traversal is **complete** — declared complete, or walked and
   reconciled — the literal is matched locally over the rendered rows. A
   complete local set *is* the whole relation, so this is a whole-relation
   search, and it matches uid and label alike;
2. otherwise, if the descriptor has verified `filter_columns`, the literal is
   mapped **server-side**: `filter` and `filter_columns` travel together in one
   call, and the filtered walk has its own query scope, its own stored batches,
   its own batch ordinals, its own cursor and its own terminal callback;
3. otherwise the page is **unsupported** and says so. A partial local filter is
   never presented as a whole-relation search.

The literal is never tokenised and tokens are never intersected: "Cooper Alan"
is sent as "Cooper Alan" and a complete zero for it is a complete zero, not an
invitation to try the words separately. A filtered fetch never mutates the base
traversal or the base total.

## Query scopes and page tokens

A cursor is a **page token**: the handle, an optional traversal tag, then the
page ordinal.

| token | what it continues |
| --- | --- |
| `O7/p2` | page 2 of the base (unfiltered) traversal of handle O7 |
| `O7/p3` | page 3 of the same traversal |
| `O7/f1p2` | page 2 of the *first* filtered traversal declared on O7 |
| `O42/f2p11` | page 11 of the second filtered traversal on O42 |

Ten characters at most in practice, no base64, no padding, nothing to decode.
C1 (`exp-ido-gqv-8`) measured the previous 150-byte opaque cursor being re-typed
by hand and corrupted in **4 of 15** fetch calls — and the corruption decoded to
a *different valid handle*, refused only because the handle travelled inside the
cursor. Four of that attempt's ten `search_memory` calls were spent reading the
cursor back out of the previous observation. A token the agent can hold in one
glance removes both costs.

The token carries **no payload**. The query scope, the position and the
descriptor digest live in `result_handle_cursors` under the token, and
`decode_cursor` resolves a token to exactly the payload the old base64 cursor
spelled out (`v`, `h`, `q`, `p`, `d`), so every check downstream is unchanged.
That is what makes a mistyped token safe rather than merely detectable:

| what the agent typed | what happens |
| --- | --- |
| a token for another handle (`O6/p2` on `O7`) | refused, naming both handles — the handle is literal in the token |
| a token this turn never issued (`O7/p9`) | refused, and the message lists the tokens that *were* issued for that handle |
| a filtered token on the base traversal, or the reverse | refused as a different query on that handle (unchanged 14.2 scoping) |
| a token written for a different descriptor | refused, restart at page 1 |
| `O7/p1`, `O7/p0`, `xyz`, an old base64 cursor | refused as not a page token; page 1 is the call with no cursor |
| quoting or case (`` `O7/p2` ``, `"O7/p2"`, `o7/P2`) | accepted — none of that changes which handle or traversal is named |
| any other single-character edit | refused, or a page **of the same handle in the same traversal** — never another listing's rows |

Tokens are stable for the turn and durable: they are rows, so a token printed
before a hot-cache eviction or a process restart still resolves. Page 1 has no
token because page 1 is the call that passes no cursor at all.

A filtered fetch never mutates the base traversal or the base total. Filtered
pages are stored under their own `query_scope`, and `total` on a filtered page is
still the whole relation while `matched` is the filtered population. The base
traversal is the one an agent pages most, so it is the one that stays untagged.

`cursor_placeholder` exists for callers that measure a header before they know
the offset: a page is packed against the widest token its traversal could print,
and a width probe must not issue a real token for a position that may never be
served.

## The literal

`normalize_literal` applies NFKC, maps U+00A0/U+2007/U+202F/U+2009 to a space and
U+2011 to a hyphen, strips zero-width characters, and collapses whitespace runs.
The portal normalises nothing and the model emits exactly those characters in
these very names, so an unnormalised filter returns a confident, silent zero.

`%`, `_` and `*` are LIKE wildcards on the portal, backslash escaping does not
work, and the parameter is described to the agent as literal. They are therefore
**removed**, and the page reports which literal was actually used. (B0 §b left
the choice between stripping and refusing; stripping keeps the lookup moving and
the note keeps it honest.)

## The page observation

`ResultPage.as_observation()` is a listing observation and gets the listing
budget: `RESULT_PAGE_MAX_BYTES` = 3,072 UTF-8 bytes, header line included
(`FW_RESULT_PAGE_MAX_BYTES`, minimum 512). The header carries the handle, the
page position, the counts and the continuation state:

```
result_handle=O42 page 2 rows 26-50 of 477 matched=477 materialized=50 total=477 source_complete=false matched_complete=true continuation=cursor outcome=rows has_more=true next_cursor=O42/p3
```

The header names the outcome class as well as the counts, so a page with no rows
can never be read as a zero when it is an unsupported query or a refusal. The
`next_cursor` value is printed verbatim and is the whole cursor: it is what the
next call passes back, with nothing to reconstruct.

Rows are packed **whole**. The packer stops at the last row that fits and the
next cursor starts at the first one that did not, so a row is never cut, never
skipped and never shown twice. A row wider than the whole budget is emitted
whole with a warning rather than cut — cutting invents a row that was never
returned, and dropping loses evidence. Over-budget warns and continues; it never
refuses.

`ResultPage` exposes `as_observation()`, `matched`, `total`, `materialized`,
`source_complete`, `matched_complete`, `continuation`, `incomplete_reason`,
`next_cursor`, plus `handle`, `page_alias`, `parent_alias`, `rows`, `outcome`,
`position`, `page_index`, `literal`, `filter_columns`, `warnings`, `notes`.

Pages are filled to the observation budget rather than to one backend batch, so
a small `batch_size` does not produce a three-line page. Rows fetched but not
shown are not discarded: they are stored, and the next cursor returns them
without another backend read.

After the third page of one handle in a turn, the observation carries a note
suggesting `contains=<name>` for a named lookup — one filtered call finds a row
at any page position. It suggests; it never refuses, never caps and never
narrows anything itself.

`continuation` describes the query that actually ran: `cursor` (more rows, or a
walk that can continue), `complete` (every row of this query was shown and the
query is proven complete) or `source-incomplete` (no more rows can be shown and
the query could not be proven complete). A filter the backend applied to the
whole relation is `complete` even when the base listing this handle materialised
is not — the header reports both numbers.

`incomplete_reason` is typed. The first group is the **framework's** — it decides
these and chooses the words; the second is the **adapter's**, reported on a
terminal callback and passed through verbatim. Only the owner changed, not the
words or the prose, and the page prints both groups identically.

| reason | owner | meaning |
| --- | --- | --- |
| `producer_materialized_subset` | framework | the command materialised part of the relation and this handle has no descriptor to continue with |
| `no_verified_filter_columns` | framework | the handle cannot map a literal to verified columns for this view |
| `resolver_error` | framework | the source refused a batch; stored rows still served, cursor still advances |
| `resolver_unavailable` | framework | no resolver of that name is registered in this process |
| `resolver_call_limit` | framework | this call reached its backend-batch bound; ask again to continue |
| `store_unavailable` | framework | the batch store could not be written, so this walk stopped where it was |
| `completeness_not_claimed` | framework | the walk reached its end and the adapter decided nothing; the question is asked again next fetch |
| `countonly_error` | framework | the terminal callback raised; stored rows still served |
| `countonly_mismatch` | adapter | the walk and the source's own count disagree |
| `countonly_unavailable` | adapter | the source offers no independent count to prove coverage |
| `offset_origin_not_zero` | adapter, *framework for now* | the handle starts partway into the relation, so a count cannot prove its coverage |

`countonly_error` sits in the framework's group because it is what the framework
says when the terminal callback *raised* rather than answered — the same guard, in
the same place, saying the same thing about the same call as before F4. An adapter
that answers with that word for its own reasons is reported with it too, like any
other.

`offset_origin_not_zero` is designated the adapter's and is **still emitted by the
framework**, from a short-circuit in `_issue_terminal` that reads `start_offset`
out of `state` before any callback is made. That short-circuit is the one place
this package looks inside `state`, and it is a deliberate backstop rather than a
leftover: until the adapter answers the origin question itself (IDO's
`ido-0rk.2.1` I4), removing it would leave a partway-origin handle's completeness
to whatever the adapter happens to say, and the failure that guards against — a
handle covering part of a relation reported as covering all of it — is silent.
When I4 lands, the short-circuit and `_origin_offset` go together, at no change to
the page.

Only the framework's words are continuable, and only two of them
(`resolver_call_limit` and `resolver_error`). Everything else serves its stored
rows and offers no cursor on the final page.

## Outcome classes

Every page states which of these it is, and they are not interchangeable:

| outcome | meaning |
| --- | --- |
| `rows` | rows matched and are shown |
| `complete-zero` | the query ran completely and matched nothing |
| `unsupported` | this handle cannot answer this query at all |
| `error` | the source refused or failed and there are no rows to show |
| `partial` | rows are shown but coverage is not proven |

A page that shows rows it cannot prove to be all of them is `partial`, and says
which of the reasons above made it one. A complete zero is phrased as a fact
about the query — "No rows matched the
literal … in these fields … it is not evidence that the person or object does not
exist" — never as the absence of an entity. A filter over a handle that holds
only part of its relation is reported as **unsupported**, not run locally and
presented as a whole-relation answer.

## Configuration

Both budgets are derived from the model's context window
(see [`docs/context_budget.md`](context_budget.md)); the names below are
**tuning overrides**, not the interface.

```
# fastworkflow.env
FW_RESULT_PAGE_MAX_BYTES=3072            # page observation budget (min 512)
FW_RESULT_HANDLE_HOT_MAX_BYTES=262144    # hot rows per process
```

Those are the values a 131,072-token window produces — `cerebras/gpt-oss-120b`,
the accepted stack's main agent model. A value that is not a valid integer, or
is below the minimum, logs a warning and falls back to the derived budget rather
than aborting a turn.
