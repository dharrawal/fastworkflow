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
```

`ResultHandleError` is raised for anything the calling command can act on: an
unknown handle (the message names the handles this turn does store), a cursor
from another handle or another filter, a cursor this build cannot read, a
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
| `page_size` | rows per backend page for continuation |
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
    view="ido_permissiondetail_identity",
    params={"scope": "85cde168…"},
    filter_columns=("identity_displayname", "identity_surname"),
    uid_field="identity__id",
    label_fields=("identity_displayname",),
    page_size=25,
    ordering="unsorted-offset",        # the only value there is
    start_offset=0,
    materialized=25,
    timeslot=None,                     # explicit: no pin exists
    role=None,
    count_only=True,
    extra={},
)
```

Two fields are shaped by the B0 probe
(`evaluation/artifacts/result-search/b0-probe.md`, bead `ido-gqv.6`):

* **There is no `sort` field.** On `ido_groupDetail_identity` an explicit sort
  combined with offset paging silently drops members while returning exactly
  `total` rows — 20 of 540 at page size 40. The descriptor cannot express a
  sorted walk, so no caller can ask for one.
* **`timeslot` may only be `None`.** `IDO.timeslot` is `None` and the client
  drops the field rather than sending a null, so there is no pinned session
  timeslot to reproduce. The descriptor records the absence explicitly instead
  of leaving the question open; anything else raises.

`filter_columns` must already be verified for that view. A filter sent without
columns is silently ignored by the portal and returns the whole scope, so the
pair "filter, no columns" must be impossible to emit; an empty
`filter_columns` means literal filtering is unsupported for the handle and is
reported as such.

## Storage and retention

Two new tables in the same SQLite file the observation archive uses, so a page
and the observation that showed it survive together. `observation_offload_handles`
is untouched.

`result_handle_declarations`, keyed `(scope_id, alias)`: `scope_json`, `kind`,
`summary`, `ordering`, `total`, `materialized`, `source_complete`, `page_size`,
`classification`, `presentation`, `filters_json`, `descriptor_json`,
`descriptor_sha256`, `columns_json` (verified column names and types, from the
first page), `sample_row_json` (one sample row from the first page),
`parent_alias`, `query_scope`, `cursor_position`, `declared_at`.

`result_handle_pages`, keyed `(scope_id, alias, query_scope, start_offset)`:
`limit_requested`, `source` (`producer` or `resolver`), `row_count`,
`backend_total`, `record_json`, `record_sha256`, `fetched_at`.

Pages are **append-only and digest-verifiable**. The insert cannot overwrite and
the value returned is the read-back, so re-fetching an offset that already
exists returns the stored page and never writes a second row: a retry is
idempotent by construction, not by convention. A stored record whose payload no
longer matches its digest is refused rather than served.

**Retention.** This module deletes nothing. Declarations and pages live exactly
as long as the archive file that holds the turn's observations — that is what
makes a page reconstructable for evaluation after the live turn ended. The hot
cache (`FW_RESULT_HANDLE_HOT_MAX_BYTES`, default 256 KB, oldest walk first) is a
*residency* bound and not a retention bound: every evicted row came from a
stored page and is rebuilt from SQLite on the next read.

**Scope.** Rows are keyed by the same `RuntimeHandleScope` the offload archive
uses (store identity, channel, experiment, task, attempt, turn). A handle
declared in one turn is not visible in another, and the same alias in two scopes
is two different listings. `state.scope_for_host()` is now the single
implementation of that scope, reached both from the ReAct loop and from a
command's own frame — a scope computed two ways would be two scopes the moment
either changed.

## Query scopes and cursors

A cursor is opaque (base64url of a small JSON object) and carries its **query
scope**: the handle, the normalised literal's digest (empty for the unfiltered
listing), the position, and the descriptor digest. A cursor from another filter,
another handle or another descriptor is refused by name, with a message that
says which query it belongs to and that omitting the cursor restarts the query
at its first page.

A filtered fetch never mutates the base traversal or the base total. Filtered
pages are stored under their own `query_scope`, and `total` on a filtered page is
still the whole relation while `matched` is the filtered population.

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
result_handle=O42 page 2 rows 26-50 of 477 matched=477 materialized=50 total=477 source_complete=false matched_complete=true continuation=cursor has_more=true next_cursor=eyJk…
```

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

`continuation` is one of `cursor` (more rows in this query scope),
`complete` (every row of this query was shown and the query is proven complete)
or `source-incomplete` (no more rows can be shown and the source could not be
proven complete). `incomplete_reason` is typed:
`producer_materialized_subset` — the producing command did not materialise every
backend row and this handle has no descriptor to continue with.

## Outcome classes

Every page states which of these it is, and they are not interchangeable:

| outcome | meaning |
| --- | --- |
| `rows` | rows matched and are shown |
| `complete-zero` | the query ran completely and matched nothing |
| `unsupported` | this handle cannot answer this query at all |
| `error` | the source refused or failed |
| `partial` | rows are shown but coverage is not proven |

A complete zero is phrased as a fact about the query — "No rows matched the
literal … in these fields … it is not evidence that the person or object does not
exist" — never as the absence of an entity. A filter over a handle that holds
only part of its relation is reported as **unsupported**, not run locally and
presented as a whole-relation answer.

## Configuration

```
# fastworkflow.env
FW_RESULT_PAGE_MAX_BYTES=3072            # page observation budget (min 512)
FW_RESULT_HANDLE_HOT_MAX_BYTES=262144    # hot rows per process
```

Both follow the `env_int` pattern: a value that is not a valid integer (or is
below the minimum) logs a warning and falls back to the default rather than
aborting a turn.
