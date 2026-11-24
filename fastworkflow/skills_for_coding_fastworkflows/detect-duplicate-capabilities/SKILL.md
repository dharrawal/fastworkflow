---
name: detect-duplicate-capabilities
description: >-
  Read and act on fastWorkflow's near-duplicate capability diagnostics (the pre-flight lexical
  separability scan, the post-training router-confusion scan and the training report's cross-fit
  confused pairs) to decide whether two commands should be merged, re-seeded, or recorded as a
  deliberate near-duplicate. Covers what
  separability actually measures, the reporting bands and their thresholds, the
  `accepted_duplicates.json` accept-list, and the blind spots of each instrument. Use when
  `fastworkflow train` reports DUPLICATE CAPABILITIES, OVERLAPPING or MODEL CONFUSION, when a
  workflow keeps routing one command's phrasings to another, or when reviewing a command set for
  redundancy.
---

# Detecting duplicate capabilities

Some workflows expose the same capability twice: `Monitor/list_flags` and
`Catalog/search_flags` answer the same question. Others hold legitimate neighbours
or opposites whose seed lists do not yet express the distinction. **Both shapes present
identically**, as routing failures, and the fix for one is the opposite of the fix for the
other. These diagnostics exist to tell them apart.

None of them changes what is trained, which labels exist, or what any model predicts. None
blocks a run. They report.

## The instruments

`fastworkflow train` runs three. The first two share one report and one accept-list; the third
is part of the training report of the default `embedding` layout.

| | Pre-flight lexical scan | Post-training router scan | Cross-fit confused pairs |
|---|---|---|---|
| Runs | every `train`, before any model call | after a run that trains something, once every context's model is in the new version | when the `embedding` head is fitted |
| Input | seed utterances (plain and template) | the same seeds, through each context's trained router (the head's decision inside that context) | each distinct command row of the head (seeds and generated), predicted by a fit that did not see it (5 folds, grouped by row text) |
| Asks | "does the training data separate these?" | "does the model separate them, on rows it trained on?" | "does the model separate them on rows it did not see?" |
| Blind to | a duplicate pair sharing no distinctive vocabulary | pairs that are never in one context together | pairs below the top ten (it prints only those, plus the count over 10%) |
| Output | printed scan + `___command_info/duplicate_capabilities.json` | printed `MODEL CONFUSION` warning, added to the same JSON | `confused pairs` in the `LOCAL INTENT MODEL` section of the training report, printed at the end of the run and in `___command_info/training_report.txt` |

They are complements, not a cheap version and a good version. Read all three.

The pre-flight scan prints only when it has something to show (a finding, an accepted pair, an
accept-list problem or a stale entry); the JSON is written on every run. A run that retrains
nothing skips the router scan, so its JSON holds the lexical scan alone.

## What separability measures

"Near-duplicate" is defined as a property of the **training data**, not of the two commands'
meanings, because the training data is the only thing the model ever sees:

> Two commands are near-duplicates when a classifier restricted to that pair, trained on their
> own utterances, cannot tell them apart.

Made concrete as leave-one-out, balanced, nearest-centroid accuracy over the pair's utterances in
TF-IDF space. Three choices in that sentence carry the whole result:

- **Leave-one-out**, because an utterance sitting inside its own centroid classifies itself.
- **Balanced** (the mean of the two per-command recalls), so chance is 0.5 regardless of how many
  utterances each side has.
- **Document frequency computed across every command in the workflow**, so shared vocabulary
  ("order", "my", "please", the workflow's own subject matter) is discounted automatically and
  only distinguishing terms carry weight. This is what stops a workflow full of
  `modify_pending_order_address` / `_items` / `_payment` from producing a wall of false
  positives: those share their boilerplate and differ in exactly the terms IDF promotes.

Ties go to the *other* command. A tie means the classifier has no information, and scoring it as
a success would inflate separability precisely on the pairs the scan exists to find.

## The bands

| Band | Condition | What it means |
|---|---|---|
| `DUPLICATE CAPABILITIES` | separability **<= 0.50** | at or below the coin-flip line; the seeds do not tell the pair apart |
| `OVERLAPPING` | separability <= 0.65 **and** centroid similarity >= 0.50 | separable, but only just; usually not a capability defect |
| `MODEL CONFUSION` | symmetric misroute rate **>= 0.40** | a context's router sends these commands' own seeds to each other (top answer only) |
| skipped | either side has fewer than 3 utterances | counted and reported in the notes, never silently scored |
| not examined | centroid similarity < 0.10 | cost pre-filter only; such pairs share almost no vocabulary |

The 0.10 pre-filter is a performance measure, not part of the definition: a 160-command workflow
has 12,720 pairs and the separability computation is quadratic in utterances.

`MODEL CONFUSION` is measured on seeds the model **trained on** (the `embedding` head weighs each
seed three times), so it is a strong signal: the model failed to separate the pair with the
answer in its training set. The cross-fit pairs are the realistic view: rows held out of the fit,
the head's best label over the whole workflow, the ten pairs with the most rows given to each
other (both directions, counts and shares) and how many pairs exceed 10% of either command's rows.

## Reading one finding

Four fields of a lexical finding decide your response. Read them together:

- **`separability`**: how badly they collide. At or below 0.50 the seeds carry no information
  about which command is meant.
- **`shared_contexts`**: the contexts that hold **both** commands. A pair that co-occurs is a
  live conflict inside one context. A pair in disjoint contexts is a design ambiguity that
  surfaces when a request is handed on to another context, so the responses differ and the report
  does not merge them into one verdict. Empty does **not** mean safe: the router scan never sees
  such a pair, and the cross-fit pairs do.
- **`shared_terms`**: what the seeds must stop relying on.
- **`terms_only_in_a` / `terms_only_in_b`** (printed as "only in first / second"): what the
  seeds must lean on. Empty on one side is the diagnosis by itself: that side's seeds say nothing
  the other side's do not.

## The three responses

**Merge.** The pair is one capability. Delete the redundant command and accept the longer
navigation path: the extra turn is paid once per task, while the confusion is paid on every
request that could mean either. Keep both only when the survivor genuinely cannot serve the
case, for example when a listing must prove parent-to-child membership by paging a
parent-scoped view, which a global search by id cannot do.

**Re-seed.** The commands mean different things and the seeds do not say so. Decide which token
class carries the decision (the entity noun for "same question asked of different things", the
verb for "different action on the same thing") and make the seeds carry it. The
`design-context-models` skill covers which side to anchor and why the fix is usually
one-directional. If you cannot write seeds that separate the pair, it is one capability: merge.

**Accept.** The pair is a deliberate near-duplicate. Record it, or the same warning appears on
every run forever, and a permanent warning is one nobody reads.

## The accept-list

`<workflow>/accepted_duplicates.json`:

```json
{
  "schema_version": 1,
  "accepted": [
    {"commands": ["Order/list_items", "Customer/list_purchases"],
     "reason": "Different scopes; the customer view is required for the returns flow."}
  ]
}
```

- It lives at the **workflow root**, not in `___command_info`. It is your input, like
  `intent_benchmark.json`; `___command_info` is generated output that the trainer prunes and
  that you may delete to force a rebuild.
- Command names must match **exactly** as printed, fully qualified. A bare leaf name matches
  nothing.
- Order within a pair does not matter. A bare `["a", "b"]` pair is accepted, but write the
  object form: it is the only one with somewhere to record **why**.
- One entry answers the pair in the lexical scan and the router scan. The training report's
  cross-fit pairs do not read it.
- Accepted pairs are still scanned and still listed, under `ACCEPTED`, with your reason.
- **Nothing rewrites this file.** It is yours.

Two report sections keep it honest:

- `ACCEPT-LIST PROBLEMS`: a malformed entry never raises (a typo must not stop a run), so it
  suppresses nothing, and it is reported on its own because a suppression you believe is in place
  is not.
- `STALE ACCEPT-LIST ENTRIES`: pairs accepted but no longer reported, usually because the seeds
  have since been made distinctive. Delete them.

## Two blind spots worth stating

1. **The lexical scan is a lexical instrument.** Synonymous *verbs* do not defeat it: a control
   pair saying "list/pull/give" against "search/find/look/get" still scores 0.00, because the
   weight sits on the shared domain nouns. What defeats it is a duplicate pair whose vocabulary
   is disjoint end to end. Nothing lexical can see that; the router scan and the cross-fit pairs
   cover it.
2. **Leave-one-out is biased downward on small utterance sets.** Removing an utterance from its
   own centroid shrinks self-similarity while the comparison centroid keeps all its mass, so the
   0.5 cut is more permissive than a true chance line. On the shipped retail workflow the worst
   pair the scan examines scores 0.600 (reported as `OVERLAPPING`).

Neither blind spot is a reason to distrust a finding. Both are reasons not to read a **clean**
report as proof there are no duplicates.

## Running the lexical scan outside a training run

```python
import fastworkflow
from fastworkflow.train import duplicate_detection

fastworkflow.init({})
report = duplicate_detection.scan_workflow(workflow_folderpath)   # seeds + accept-list
print(duplicate_detection.format_report(report))
```

It needs no LLM key and no network, so run it on every seed edit rather than waiting for a
training run.

## Triage checklist

```
- [ ] Read separability first: <= 0.50 means the seeds do not tell the pair apart
- [ ] Check shared_contexts: a live conflict in one context, or a design ambiguity across contexts
- [ ] Check whether one side's terms_only_in_* list is empty: that is the diagnosis
- [ ] Choose merge / re-seed / accept for every band, not just DUPLICATE CAPABILITIES
- [ ] Record accepted pairs with a reason, at the workflow root
- [ ] Delete stale accept-list entries the report names
- [ ] Do not read a clean scan as proof of no duplicates: read MODEL CONFUSION and the confused pairs too
- [ ] After re-seeding, retrain and compare cases, not percentages
```

## Related

- Which token should discriminate a colliding pair, and which side to anchor: the
  `design-context-models` skill.
- Confirming a re-seed actually helped, without being fooled by training variance: the
  `evaluate-intent-routing` skill.
- Where in a training run each scan happens: the `train-and-publish-models` skill.
