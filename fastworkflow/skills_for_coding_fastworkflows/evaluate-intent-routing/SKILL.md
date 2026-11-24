---
name: evaluate-intent-routing
description: >-
  Read fastWorkflow's held-out routing and escalation evaluation honestly, and decide from it
  whether a change to a workflow's commands, seeds or context models actually helped. Covers why
  the training F1 measures memorisation, the whole-persona holdout and what it does and does not
  remove, which of the reported numbers is comparable across runs, keeping `intent_benchmark.json`
  disjoint from the seeds, leak calibration, and the paired-build discipline a claim requires. Use
  after `fastworkflow train` prints its evaluation table, when comparing two training runs, when
  writing or fixing benchmark cases, or when a reported accuracy looks too good.
---

# Evaluating intent routing

## Relate routing results to application outcomes

`intent_benchmark.json` measures intent routing; workflow-local benchmark versions (`benchmark_versions` in the live DB) measure
application tasks through a driver. Keep both when both matter: a routing improvement does not
prove better task completion. Use [create-workflow-benchmarks](../create-workflow-benchmarks/SKILL.md)
for the versioned task catalog and experiment UI, and
[optimize-workflow-with-feedback](../optimize-workflow-with-feedback/SKILL.md) to compare actual
conversations, trace failures and human feedback before/after a routing change.

## The number that has misled everyone

`in_distribution_f1` is scored on the **same** synthetic utterances the model trained on. Under
the default `embedding` layout it is the cross-fit accuracy of the context's own training rows
(five folds, grouped by row text); under the fine-tuned layouts it is a random split of those
rows. Either way the rows come from a handful of personas expanding one seed list, so it
measures **memorisation**. The measured gap on a 160-command workflow: ~0.94 reported F1 against
46.2% held-out top-1. Judge models on top-1, in-list and the cross-fit confused pairs, never on
`in_distribution_f1`.

## How the local model answers, and what the report can see

Intent detection is one chain (`intent_detector`). The local head answers alone only when it
is confident: in the `embedding` layout, its best command is inside the context and has
probability at least the commit threshold (`INTENT_COMMIT_THRESHOLD`, default 0.65). Otherwise
it **hands the request on** with the context's top three candidates, plus up to two commands of
other reachable scopes when one of those is ahead. It never passes a request up on its own. The
next link is the remote decision model when `JEV_API_KEY` is set (a command at 0.9 or more is
the answer; `none` or another scope's command at 0.9 or more declines the scope and the walk
moves on), and otherwise the which-command question.

The training report scores `CommandRouter.predict` for one context, the head alone, so:

**Routing**: did the head name the right command?

- **top-1**: the expected label came back as the **single, confident** answer. The only
  outcome that is a correct route with no question and no remote call.
- **in-list**: the expected label is among the returned candidates. At runtime that goes to
  the remote model or the question. A real outcome worth tracking, and not a win.
- A single **wrong** label is the costly outcome: the turn runs the wrong command unasked.
  Read `per_command` and the failures for it.

**Escalation**: a case whose command lives in an ancestor scope. The report counts it correct
only when the head returns a lone `wildcard`. A current (`embedding`, schema 2) head returns a
lone `wildcard` only when the request reads as a vague reply, never because a command elsewhere
is ahead, so **benchmark escalation recall near 0 is expected and is not a regression**. What
matters on these cases is that the head did not commit: in `escalation.failures`, a `predicted`
list is the head handing on (correct behaviour); a single local command is a confident wrong
answer. Whether the ancestor command reaches the candidates, and what the remote model then
does, is outside the training report (see "Scoring outside a training run").

Routing and escalation trade against each other, so never blend them into one score.

## The cross-fit report (embedding layout)

The training report (`___command_info/training_report.txt`) has a `LOCAL INTENT MODEL` section:
cross-fit accuracy over the distinct command rows, and the ten most confused command pairs,
with how many pairs give more than 10% of either command's rows to the other. It sets nothing
(the threshold is a constant). It is in-distribution like `in_distribution_f1`, but a confused
pair is a concrete design signal: add seed utterances that tell the two commands apart (a seed
weighs three times a generated row), or merge the two commands when users cannot tell them
apart. See the `detect-duplicate-capabilities` skill.

## Two populations, and only one is comparable across runs

| Population | Source | Comparable across runs? |
|---|---|---|
| persona holdout (`routing`, `holdout_escalation`) | whole personas reserved from the generated utterances | **No.** The split is re-drawn every run, so two runs score different cases. |
| benchmark (`benchmark_routing`, `escalation`) | `<workflow>/intent_benchmark.json` | **Yes.** The file is fixed and its cases pair by construction. |

This is the single most common misreading of the report. A persona-holdout top-1 that moved three
points between runs may mean nothing at all: a context is never scored on the same phrasings
twice. **Compare on the benchmark number.** If a workflow has no benchmark file, it has no
cross-run measurement, only a snapshot.

A ratio with a zero denominator is `null`, never `0.0`. `"escalation_recall": null` means no
escalation cases were scored, which is a different fact from a measured 0%.

## What the persona holdout does and does not remove

Whole personas are reserved, never a random sample of rows. Holding out 25% of the rows of a
persona whose other rows are in training is the defect, not the fix. Consequences:

- Hand-written seed utterances (persona `__seed__`) always train and are never held out.
- An utterance attributed to several personas is held out only when **every** contributing persona
  is held out.
- A label that the split would leave below the trainer's floor of 2 training rows has its held-out
  rows **returned to training wholesale**, rather than being left partially trained to buy a
  metric. Those labels then have **no held-out coverage at all**, and a note in the report names
  them. A command with few utterances can therefore be invisible in this report rather than
  scoring badly in it, so check the note before reading a clean result as coverage.

What it does not remove is the **wording**. Two personas asked the same question in one prompt
often produce near-identical sentences, so removing the author leaves the phrasing behind. That is
what `leak_calibration` quantifies: the median maximum token overlap between held-out rows and
their own label's training rows, and the fraction overlapping at 0.8 or more. High overlap means
the routing number is closer to recall than to generalisation.

`exact_duplicates` should be **0**: the split drops exact leaks. A non-zero value is a bug
signal, not a data observation.

Two more limits worth stating before quoting any number:

- The evaluated model **is** the shipped model, with held-out personas removed from training, so
  the numbers are **lower bounds**.
- The holdout is in-generator: held-out personas came from the same generator, so it measures
  generalisation across personas, not across real users.

## The benchmark file is the instrument; keep it clean

`<workflow>/intent_benchmark.json` is a hand-written held-out test set. Full schema, the two case
kinds, label-qualification rules and a worked example are in `docs/intent_benchmark_format.md` in
the fastWorkflow repository. What matters here is the discipline around it:

**Never paste a failing benchmark case into a seed list to "fix" it.** That is the mistake the
whole design defends against, and it is caught: benchmark utterances are enforced disjoint from
the seed table, and an overlap raises `BenchmarkLeakError` and **fails the run**. Comparison is on
a normalised form (NFKC, smart quotes folded, whitespace collapsed, surrounding punctuation
stripped, casefolded), so dropping the trailing period does not evade it. Merely *similar*
utterances are reported as warnings and do not fail.

The check runs in preflight, **before** any paid generation, so a leak costs you nothing but the
time to fix it.

**`Benchmark defect:` warnings are defects in the benchmark, not in the model.** A routing case
whose expected label is absent from the tested context's label space fails forever. The
authoritative label list is the `contexts` map in `___command_info/routing_definition.json`.
Framework commands keep their `IntentDetection/` prefix in every context including `"*"`;
`wildcard` and `parameter_value` are reserved labels and are never qualified.

Two independent phrasings per command is a reasonable floor. The reference workflow used 446
routing cases and 37 escalation cases.

## Reading the report

Printed at the end of training and written to `___command_info/heldout_evaluation.json`
(schema 2), with a `metric_notes` block restating what each metric does and does not mean.

Per context: `in_distribution_f1`, `routing`, `holdout_escalation`, `leak_calibration`,
`benchmark_routing`, `escalation`, plus the `seed` and the `heldout_personas` list.
`routing.per_command` is where a single bad command hides inside a healthy context average; read
it before concluding a context is fine.

Under the default `embedding` layout one head serves every context, so any retrain refits it and
every context is measured. Only the `per_context` layout retrains selectively: carried-forward
contexts are re-inserted from the previous report and tagged `carried_forward` with the version
they came from. A diff shows them unchanged because nothing re-measured them, not because nothing
moved.

## When the local model hands on: the remote model reads the docstring and the seeds

When the head hands a request on and `JEV_API_KEY` is set, the remote decision model
(`intent_detector.MatcherEngine`) is asked once about every command on the lookup path. For each
command it sees the `Signature` docstring on one line and up to nine of its `plain_utterances`. So
write the docstring's first sentence as the command's plain purpose, in the words a user would
use, and put conditions and usage notes after it: a first sentence that names an input the request
does not give ("... by uid") can keep the command below the 0.9 bar for every request that omits
it. Run without `JEV_API_KEY` to compare routing with the local model alone.

## Deciding whether a change helped

The noise floor depends entirely on whether generation was cached:

| Condition | Measured floor | What one paired build proves |
|---|---|---|
| both caches warm (no seed change, no `--regenerate-utterances`) | two full retrains produced **zero** verdict changes across 446 routing and 37 escalation cases, with byte-identical models | a one-case difference is real |
| generation re-ran | **20.6%** of routing verdicts (92 of 446) changed between two otherwise-identical runs | nothing; a reliability claim needs at least five paired builds, all reported |

So the first question about any comparison is not "did the number move" but "did the caches stay
warm". The `embedding` head is a deterministic fit of its rows with a frozen encoder, so the same
rows give the same head. A change that forces regeneration (a persona change, a model change, a
framework upgrade) puts you in the second row whether you wanted it or not. See the `train-and-publish-models` skill
for what forces regeneration.

Then read cases, not percentages:

1. Compare **benchmark** verdicts, case by case, before and after.
2. Keep only the cases the change can actually explain. A change touching one command's seeds
   cannot move an unrelated framework command; that movement is churn.
3. Report the causal set as a **trade** ("N cases gained here against M lost there"), because a
   well-targeted fix usually does cost something at the boundary it moved.
4. Check the escalation cases separately: a routing gain bought by the head now committing to a
   local command on an escalation case is not a gain.

## Scoring outside a training run

`fastworkflow.train.heldout_evaluation` never loads a model. `score_routing` and
`score_escalation` take a caller-supplied `predict_fn(utterance) -> list[str]` returning ranked
candidates, top-1 first, which is what `CommandRouter.predict` returns, so they score any model you
can wrap in that signature.

The chain beyond the head is measured by two scripts in the fastWorkflow repository, both on a
copy of the workflow: `scripts/experiments/embedding_head.py product` (no model call: right
outright, wrong confident, handed on with or without the right command, and for escalation cases
whether the ancestor command is among the candidates) and
`scripts/experiments/score_matcher_engine.py --mode chain` (real remote calls: the head, the remote
model and the chain on the same benchmark, paired against a baseline with an exact McNemar test).

## Checklist

```
- [ ] in_distribution_f1 is not being quoted as accuracy
- [ ] The comparison is on benchmark_routing, not on the re-drawn persona holdout
- [ ] Caches were warm on both runs, or five paired builds exist
- [ ] Routing and escalation are reported separately; embedding-layout escalation recall near 0 was
      not read as a regression, and its failures were checked for confident local answers
- [ ] top-1 and in-list are not conflated
- [ ] leak_calibration was read; exact_duplicates is 0
- [ ] No benchmark utterance has migrated into a seed list
- [ ] `Benchmark defect:` warnings were fixed as benchmark bugs, not chased as model bugs
- [ ] Confused pairs in the cross-fit report were read
- [ ] (per_context layout) Carried-forward contexts are excluded from the diff, or noted as unmeasured
- [ ] The claim is a case-level trade, not a percentage delta
```

## Related

- Benchmark file schema, case kinds and label qualification: `docs/intent_benchmark_format.md` in
  the fastWorkflow repository.
- What forces regeneration, and therefore which noise floor applies: the
  `train-and-publish-models` skill.
- Turning a routing failure into a design fix: the `design-context-models` and
  `detect-duplicate-capabilities` skills.
