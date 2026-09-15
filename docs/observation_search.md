# Observation search

Large, older command results can be saved outside the ReAct prompt. A replacement
is emitted only when it is shorter than the original in both characters and
UTF-8 bytes. The existing size threshold, recent-observation protection and
trajectory budget still apply. Replan copies follow the same savings rule and
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

`annotate_execute_observations` writes the line during the ReAct
`on_step_complete` hook, before compaction measures the packed target, so the
byte budget is checked against the trajectory the agent actually receives. The
line is written once and never rewritten: a surviving step's ordinal cannot
change, so a disagreement between a printed handle and the recomputed one is a
defect, recorded as an `alias_conflict` event, with the printed text left as it
stands. There is no step-number fallback anywhere — an `O` the run never printed
stays an explicit `no matching offloaded handle` miss.

**Archived text excludes the handle line.** The line is presentation only:
`strip_alias_line` recovers the exact command response, and that response — not
the printed text — is what the archive stores, what its `text_sha256` covers,
what the offload label describes, and what the authored-output lookup matches
against the action log. Digests of observation text therefore stay comparable
with observations recorded before handles were printed, and `search_memory`
answers from the unmodified command output. Offload eligibility, the size
thresholds and the savings rule are likewise evaluated on the response alone, so
printing a handle can never be what makes an offload look profitable.

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
the `sqlite` tier. Compaction policy is untouched: eligibility, the recency
protection of the newest execute observations, the savings rule and the packed
target all behave exactly as before.

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
compaction: the eligibility threshold is 1,000 estimated tokens, which no
recorded answer approaches, an offload label is about the size of a typical
answer so `replacement_saves_space` would usually refuse the swap, and reading a
label back would cost a paid model call to recover a few hundred bytes.

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
