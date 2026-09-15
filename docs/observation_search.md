# Observation search

Large, older command results can be saved outside the ReAct prompt. A replacement
is emitted only when it is shorter than the original in both characters and
UTF-8 bytes, and only when the swap frees at least 1 KB (see
[Offload eligibility](#offload-eligibility)). Recent-observation protection and
the trajectory budget still apply. Replan copies follow the same savings rule and
persist any newly labelled command observation before returning a pointer.
Non-command observations in a replan copy stay inline. If this irreducible
evidence exceeds the byte target, the runtime records the overage and continues.

The replacement format is:

> Use search_memory tool to search inside Observation O8 returned by show_holders. It was offloaded to memory and contains identity_uid: identities holding this permission; label: their display names.

The command is the original `execute_workflow_query` command argument. Output
field descriptions come from the resolved command's authored `Signature.Output`
metadata. When those descriptions are unavailable, the label explicitly describes
the beginning of the command output. Hashes remain in the archive and tracing
manifest rather than taking space in the prompt label.

## Offload eligibility

An execute observation is eligible for offloading when replacing it with **its
own label** frees at least `MIN_OFFLOAD_SAVING_BYTES` (1,024) UTF-8 bytes:

```
utf8_bytes(command response, alias line stripped)
    - utf8_bytes(that step's actual offload label)  >=  1024
```

The label is built before the question is asked, because eligibility is a
property of the swap and not of the observation: the same 1.5 KB of output is
worth replacing under a 170 B label and is not under a 500 B one, and the label
carries the step's own alias, the full command argument and the command's
authored `Output` descriptions. `FW_OFFLOAD_MIN_SAVING_BYTES` overrides the
minimum (`0` means "offload whenever the label is smaller"); a value that is not
a non-negative integer logs a warning and falls back to the default.

Nothing else about compaction changes: oldest-first selection, the five most
recent execute observations protected, the 28,000 B packed target and
`replacement_saves_space` are as they were. A protected observation is never
priced, so no label is built for it. `replan_trajectory_skeleton` applies the
same rule — a label merely shorter than its observation is no longer enough
there either.

Decision records report `reason: below_min_saving` with the computed
`offload_saving_bytes` and `label_size`; `response_size` still carries
`characters`, `utf8_bytes` and `estimated_tokens`, and the `offload` event still
reports `estimated_tokens` beside the new `offload_saving_bytes`.

**This replaces a 1,000-estimated-token floor** (`ELIGIBILITY_THRESHOLD_TOKENS`,
~4 KB of ASCII), which asked how big an observation was rather than how much
residency replacing it would buy. Every listing page between ~1.3 KB and 4 KB
stayed resident for a whole turn although its label costs a few hundred bytes.
Replayed over the eleven saved result-search attempts (606 execute observations,
`ido-986.14.6`), the new rule makes 80 more observations eligible — 1,272 to
3,555 B of listing pages and portraits — makes **none** ineligible, and would
have changed 73 actual offload decisions, always by offloading something that had
stayed resident. End-of-turn packed bytes fall by 1.7–26.8 KB per attempt, and no
recorded replan skeleton grows. Runs recorded before this change used the token
floor; their artifacts are unchanged and their numbers are not comparable
observation-by-observation.

## Canonical observation handles

Every `execute_workflow_query` observation is printed with its canonical handle
on the first line, inline results included:

> Observation O42 (execute_workflow_query)
> 477 holder(s).
> ...

`O{n}` is the execute ordinal — the n-th `execute_workflow_query` step of the
turn — **never** the ReAct step number. `compact.execute_ordinals` assigns it,
so the printed handle carries `ordinal_offset` (the count of execute steps the
context-window fallback has truncated away) and matches the alias an offload
label or the archive uses for the same observation. Non-execute tool outputs
(`search_memory`, `ask_user`, `what_can_i_do`, `intent_misunderstood`) get no
handle: there is nothing to search inside them.

### The context instance the command ran in (`ido-8ps.13`)

When the command ran inside a non-root command context, the line also names that
context and, where the workflow declares one, that context's instance identity:

> Observation O22 (execute_workflow_query, in Account e8a0c3a1-… Alan Cooper)
> permission_uid  label
> 85cde168  Active Directory_Cloud Administrator
> ...

**Why.** A listing produced by navigating into a context carries no identifier of
the instance it belongs to: those permission rows do not repeat the account uid,
and the only thing tying them to Alan Cooper is that the previous step entered
his account. The link lives in the ORDER of the commands, so a reader that is not
allowed to use history — the evidence filler, a `search_memory` answer, the
extract step, a human scrolling a store — cannot recover it. 12 of the 14
unresolved rows in `ido-8ps.10` had exactly this cause.

**The context is the one the command RAN IN, not the one it entered.** It is
captured at dispatch (`CommandExecutor._remember_execute_context`), before the
command can move the context, and filed against the execute step's own `O` alias
in a turn-scoped table; `annotate_execute_observations` reads it rather than
recomputing it, because by the time the line is printed the current context has
already moved. So `open_account_by_uid` reads as the `DirectoryExplorer` command
it is, and the `list_permissions` that follows it is the one that belongs to the
account. An observation is evidence about the context it was produced in.

**The instance identity is declared, never derived.** fastWorkflow has no notion
of a context instance's identity — the current context is an arbitrary
application object — so `fastworkflow/context_identity.py` reads a declaration
off the same context callback class that already declares `get_parent` and
`enter_command`: a classmethod `instance_label(command_context_object) -> str`,
or an `instance_label_attr = "uid"` naming an attribute to read. A context that
declares neither prints its NAME alone, and an object that carries no identity
yields no identity: nothing is invented to fill the gap, for the reason
`tracing.context_handle` gives for refusing to mint an `instance_key` — a guess
that looks concrete is worse than an honest absence. The root context prints no
clause at all, so a root-context line is byte-for-byte the A1 line above.

`context_clause` is the one place the clause is made printable. It removes
parentheses and newlines and caps the name at 60 and the label at 80 characters,
which is what lets `ALIAS_LINE_RE` treat the closing `)` as unambiguous and match
lines printed before the clause existed. Each printed line emits a `context_line`
event carrying the alias, the clause, whether an instance was named and the bytes
the clause cost — the line is presentation and reaches neither the step span
(closed with the raw tool return) nor the archive, so the event is the only place
it can be measured.

This is not behind a flag. It is an extension of A1, which is not behind a flag
either, and gating presentation would mean two shapes of printed observation to
reason about for a change whose whole cost is ~40 bytes per observation.

`annotate_execute_observations` writes the line during the ReAct
`on_step_complete` hook, before compaction measures the packed target, so the
byte budget is checked against the trajectory the agent actually receives. The
line is written once and never rewritten: a surviving step's ordinal cannot
change, so a disagreement between a printed handle and the recomputed one is a
defect, recorded as an `alias_conflict` event, with the printed text left as it
stands. There is no step-number fallback anywhere — an `O` the run never printed
stays an explicit `no matching offloaded handle` miss.

**Archived text excludes the handle line**, the context clause included. The
line is presentation only:
`strip_alias_line` recovers the exact command response, and that response — not
the printed text — is what the archive stores, what its `text_sha256` covers,
what the offload label describes, and what the authored-output lookup matches
against the action log. Digests of observation text therefore stay comparable
with observations recorded before handles were printed, and `search_memory`
answers from the unmodified command output. Offload eligibility, the minimum
saving and the savings rule are likewise evaluated on the response alone, so
printing a handle can never be what makes an offload look profitable — a
response that saves 1,023 B stays inline even though the printed text is ~34 B
longer.

**Handles a command can page.** The same `O` alias identifies a stored, pageable
copy of a listing when the producing command declares one: see
[Result handles](result_handles.md). Search answers questions inside one
observation's text; a result handle returns the listing's own rows, a page at a
time or filtered by a literal, and can continue the query against the backend
beyond the rows the command materialised. Both are reached with the alias
printed on the observation.

## Every execute observation is archived

Persistence no longer waits for an offload decision. When a step completes, the
same `on_step_complete` hook that prints the handle calls
`archive_execute_observations`, which writes **every** `execute_workflow_query`
observation into the scoped SQLite archive under its canonical `O` alias.
Offloading is then purely a residency decision — whether the text stays in the
prompt — and never a decision about whether the text can be found again.

Before this, only an observation that compaction chose to replace was persisted,
so an alias the run had just printed on an inline result resolved to
`no matching offloaded handle`: the evidence was visible in the prompt and
unreachable through `search_memory` at the same time.

What is stored is the raw command response, with the presentation line removed by
`strip_alias_line`, and `text_sha256` is the digest of exactly those bytes — the
same convention the offload path uses, so the later offload of an alias finds the
identical row rather than writing a second one. Writes are insert-or-nothing
(`ON CONFLICT DO NOTHING` plus a digest check), and a digest already written in
this process for that alias is skipped, so revisiting a step across the many
compaction passes of a turn costs nothing and can never produce a duplicate.
The archive key is the alias actually **printed** on the observation when there
is one, so the handle the agent can see is always the key its text is stored
under — including in the `alias_conflict` case, where the printed alias stands
and the recomputed one is not used.

Each first write is recorded as an `observation_archived` event (alias, step,
digest, bytes, hot-cache evictions) and puts a copy in the bounded hot cache, so
the existing cap and oldest-first eviction still apply — inline observations now
compete for that cache too, and an evicted alias simply resolves from SQLite at
the `sqlite` tier. The eager archive is independent of the offload decision, so
changing the eligibility rule moves only residency: an observation the rule keeps
inline is archived and searchable exactly like one it offloads.

Persistence is an availability optimisation on the hot path of every agent step,
so a failure must never cost evidence. A failed write records an
`archive_refused` event (`reason: persistence_failed_original_retained`) and
leaves the observation inline and unchanged; nothing is raised into the agent
loop. Rewriting a completed observation's text under a live alias is refused the
same way: the stored evidence stands.

## Inline and offloaded searches, and what a miss means

`search_memory` resolves any alias printed in the current scope, whether its text
is still inline or already replaced by a label, and answers from the same
archived bytes either way. The search event records the answering tier
(`hot`/`sqlite`) as before, plus `still_inline`:

| `still_inline` | meaning |
| --- | --- |
| `true` | the observation was still in the prompt when the agent searched it |
| `false` | it had been offloaded to a label |
| `null` | this process has no record of that alias being printed in this scope |

So a miss with `still_inline: null` is a wrong-handle selection — an invented or
mis-remembered `O` — while a miss on a handle the run did print would be a
retrieval failure. The flag is process-local bookkeeping, so a turn resumed in
another process reports `null` until it prints handles again; `status` still
says whether the search was answered. There is still no step-number fallback and no nearest-handle
guess: an alias that was never printed is an explicit miss, recorded with
`status: missing`, and no model is called.

## Search output residency

An execute observation that grows large is offloaded and replaced by a label. A
`search_memory` answer is not: it is a non-execute observation, so
`compact_trajectory` never selects it, and `replan_trajectory_skeleton` labels
execute observations only, so the answer is carried into every later segment of
the turn in full. Whatever a search answer costs, it costs for the rest of the
turn — and its size is model output, capped only by the 2,048-token completion
limit (roughly 8 KB).

So a search observation is held to the same 3 KB budget a listing observation
has. The budget covers the **whole observation**, header and marking included,
not just the answer body:

```dotenv
FW_SEARCH_ANSWER_MAX_BYTES=3072   # default; values below 1024 fall back to it
```

An answer that fits is presented exactly as before, byte for byte:

```
Observation O34 (tier=hot):
Christopher Hubbard (identity_uid=c062...) holds it; 3 of the 22 remediation ...
```

An answer that does not fit is archived whole, then cut at a line boundary by
`text_page` — the same rule paging uses, so an identifier the answer offers as
evidence is never split mid-token and a row is never halved into a shorter,
plausible-looking one — and the observation says what happened:

```
Observation O34 (tier=hot, bounded):
00000000000000000000000000000000 Person 0 account_uid=account-00000
... 38 whole rows ...
[search_memory BOUNDED ANSWER: shown 2,612 of 27,889 UTF-8 bytes of the answer
for O34; 25,277 bytes are NOT shown. This is not the complete answer, and nothing
missing from it is thereby absent from O34. Full answer archived as O34#a1
(sha256 9f3c1a2b4d5e). To get the rest, call search_memory on O34 again with a
narrower question naming the entity or predicate you still need.]
```

A bounded answer is never presented as a complete one. The marking states the
omission in bytes, denies the absence inference an incomplete answer would
otherwise invite, and gives an action the agent can actually take: the same
observation, a narrower question — which is evidence-grounded, where re-reading
a truncated answer is not.

`O34#a1` is a **record key, not a handle**. The agent-visible `O` namespace is
execute ordinals only, and `search_memory` validates its `alias` against
`O[1-9]\d*`, so this key can never be passed back as an observation: an answer
record is not a searchable observation. The prefix files the answer under the
observation that produced it and the suffix separates repeated searches of the
same observation within one scope. Operators and evaluation tooling read the
complete text with `archived_search_answer("O34#a1", scope=...)`, digest-verified
by the archive and with no second model call; the search event carries the whole
answer regardless, so a bound never loses the evidence.

Archiving happens **before** the cut, and evidence outranks the byte budget: if
that write fails, the answer is not bounded at all. The complete text is returned
inline, over budget, and `search_answer_archive_refused` records
`persistence_failed_complete_answer_retained` — the same choice a failed offload
makes when it keeps its observation inline.

The answered search event gains `answer_bounded`, `answer_utf8_bytes`,
`observation_utf8_bytes` and, when bounded, `answer_shown_utf8_bytes`,
`answer_omitted_utf8_bytes`, `answer_archive_key` and `answer_sha256`.

The bound is a tail guard, not a saving. Measured over the saved
`h1-control` (n=3), A1+A2 smoke and `ido-5uv` stores, the largest answer in 27
was 1,855 B — 60% of the budget — peak completion usage was 790 of 2,048 tokens,
and the `completion_limit` branch has never been taken. Search answers held
2.1–7.4% of end-of-turn packed bytes and 0.0–6.8% at peak, behind execute
observations, thoughts and arguments, and `what_can_i_do` output in every run.
Nothing in the code prevented an 8 KB answer; the runs simply had not produced
one. Search observations were deliberately **not** made eligible for oldest-first
compaction: compaction offloads execute observations only, an offload label is
about the size of a typical answer (median 355 B) so neither
`replacement_saves_space` nor the 1 KB minimum saving would admit the swap, and
reading a label back would cost a paid model call to recover a few hundred bytes.
(That measurement was taken while eligibility was still the 1,000-token floor;
the 1 KB minimum saving refuses these answers for the same reason, only more
directly.)

## Configuration

Set the search model independently of the main agent:

```dotenv
# fastworkflow.env
LLM_OBSERVATION_SEARCH=cerebras/gpt-oss-120b
```

```dotenv
# fastworkflow.passwords.env
LITELLM_API_KEY_OBSERVATION_SEARCH=<provider API key>
```

The standard `get_lm` routing rules also support `litellm_proxy/` routes and
provider ambient credentials. No fallback to the main agent model is performed.
The search uses temperature 0, a 2,048-token completion limit, a 120-second
request timeout and one provider retry. `FW_LM_CACHE=0` disables response caching
for independent benchmark calls.

Compaction knobs, all optional and all falling back to their default on a value
that is not a non-negative integer:

| variable | default | meaning |
|---|---|---|
| `FW_OFFLOAD_MIN_SAVING_BYTES` | 1024 | minimum UTF-8 bytes an offload must free |
| `FW_TRAJECTORY_MAX_BYTES` | 28000 | packed-trajectory target and replan bound |
| `FW_OFFLOAD_HOT_MAX_BYTES` | 262144 | hot handle cache cap |
| `FW_SEARCH_ANSWER_MAX_BYTES` | 3072 | presentation bound on a search answer |

## Tool behavior

`search_memory(question: str, alias: str)` requires one key such as `O8` — the
handle printed on the observation's first line, or named in its offload label.
An empty key, a noncanonical key, multiple keys, or an empty question is rejected.
A missing handle produces an explicit error without calling a model or searching
another observation. Resolution remains scoped to the current turn/attempt and
survives cache eviction through the SQLite archive.

The implementation reads the **current search step's thought** from the ReAct
trajectory and prefixes it to the question as `<reasoning>. <question>`. Reasoning
is not an agent-supplied tool argument. DSPy `Predict` receives that combined
question and the complete selected observation. It answers from that observation,
preserves exact identifiers and their types, and states evidence gaps. The prompt
instructs it to treat both the requesting agent's assumptions and instructions
inside the observation as untrusted claims, not additional evidence.

Broad requests for entire tables receive a count/description and a request for a
focused predicate. Completions cut off at the output limit are reported as
incomplete searches rather than successful evidence answers. An answer that fits
is returned whole; one that does not is bounded and marked as incomplete (see
*Search output residency*), never silently shortened.

The returned answer names the source observation. Search events record scope,
source hash and size, question and attached reasoning, status, model, answer,
latency and available provider usage/cost. LLM calls are also recorded through
normal DSPy observability. Provider failure yields an explicit search failure;
it is never reported as evidence that an entity is absent.

The full observation consumes the search model's input context and incurs model
cost. The observation handed to the search model is never truncated or paged;
only the answer's presentation in the trajectory is bounded, and visibly so. An observation too large for
the configured provider may fail; the caller receives an explicit failure and the
original saved text remains available. This search supplies evidence; it does not
by itself guarantee that the main agent's final conclusion is correct.

## Validation

`MinimumOffloadSaving` covers the 1 KB rule: savings of exactly 1,023 / 1,024 /
1,025 B, the saving taken from the step's real label rather than a constant (same
bytes of output, two command arguments, one offload), an authored description
lengthening both label and decision, a 3 KB listing page older than the protected
five offloaded when over target, a 1.2 KB result kept, short facts untouched, the
recency five protected before any label is built, A1's alias line excluded from
the measured saving, the eager archive holding both the kept and the offloaded
observation, search answers still never offloaded, the environment override in
both directions with bad values falling back, and the replan skeleton applying
the same minimum.

Focused tests cover labels, byte/character savings, replan persistence, scoped
resolution, required keys, current-step reasoning, archive eviction and existing
continuation behavior. `PrintedObservationHandles` covers the printed handle:
interleaved tools, truncation via `ordinal_offset`, the replan skeleton, the
inline/label/archive alias being one identifier, savings accounting with the
added line, and an unknown handle staying an explicit miss.
`EagerObservationArchive` covers the eager archive: observations archived whether
or not they are offloaded, an inline and an offloaded search receiving the same
bytes and digest, eviction and a cleared cache resolving from SQLite, repeated
persistence keeping one row, another turn's alias staying invisible, a failed or
conflicting write keeping the inline evidence and recording `archive_refused`,
the `still_inline` flag, archiving under the printed alias in the
`alias_conflict` case, and the replan skeleton persisting without a second row.
`BoundedSearchAnswers` covers the search output bound: every recorded answer size
presented unchanged, a long answer bounded, marked and archived whole, the cut
landing on a line boundary with identifiers intact, a newline-free answer cut on
a character boundary, the observation fitting the budget at every admissible
bound, a bad or too-small bound falling back to the default, the full answer
retrievable by the key the marking names and invisible to another scope, repeated
searches keeping one record each, the record key rejected by `search_memory` and
never offered as a handle, a failed answer archive keeping the complete answer
inline, a bounded observation getting no `O` alias and not being archived as one,
and the packed and replan cost of a search staying inside the budget.
Provider tests are opt-in:

```bash
FW_TEST_OBSERVATION_SEARCH_LIVE=1 python -m pytest \
  tests/test_observation_offloading.py tests/test_observation_search.py
```

Configure the search model and credentials before enabling provider tests.
Without the opt-in, deterministic integration checks run and paid cases skip.
The IDO epic `ido-5uv` records a fixed-evidence and single-task benchmark comparison.
