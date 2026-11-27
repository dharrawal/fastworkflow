# Pre-registration, second round: one intent model for the workflow

Follows `docs/single_intent_model_preregistration.md`, whose arms all failed at least one condition.
Written 2026-11-16 after those results and before any run described here.

## What the first round left open

1. The large model alone under the mask (M1) beat the heads on strict top-1 by 16 points and on
   escalation by 29 cases of 32, at a fifth of the disk and memory. It missed the in-list limit by
   one case and the seed-agreement limit, which had no baseline.
2. On the one-scope workflow every single-model arm lost 13 points of in-list. Cause unknown.
3. How much the shipped heads agree with themselves across seeds was never measured.

## Stage A: diagnosis (exploratory, on the first benchmark, which is now a development set)

Nothing in stage A is evidence for adoption. Its only outputs are a fixed recipe for stage B and
two measurements.

* **A1, the heads against themselves.** Per-context heads rebuilt by the shipped recipe (same rows,
  including their "not here" seed rows; same hyperparameters and thresholds) at seeds 42, 43, 44.
  Check: the seed-42 rebuild reproduces the shipped heads' verdicts on at least 90% of cases;
  if not, say why before going on. Output: H's seed agreement on verdicts, and H's range of top-1
  and in-list over seeds.
* **A2, the one-scope loss.** On retail, large model alone: (a) the heads' own label set, no
  workflow `wildcard` class; (b) with the class at weight 4 (the first round); (c) with the class
  at weight 1. Output: which of them recovers the in-list of H.
* **A3, the recipe.** From A1 and A2 choose, on the development set only: whether the workflow
  `wildcard` class is kept and at what weight; the ambiguity threshold rule for the large model
  under the mask (the shipped rule, or the shipped rule computed on the masked calibration rows of
  the scope's own commands only); the escalation margin rule (unchanged unless false escalation is
  above 3%). No change to the base checkpoint, epochs, learning rate or batch size.

The chosen recipe is written into this file under "Recipe for stage B" before stage B starts.

## Stage B: confirmation (on a benchmark nobody has scored)

* **Benchmark**: `tests/fixtures/intent_benchmarks/library_v2.json` and `retail_workflow_v2.json`,
  written by hand for this round: new phrasings, disjoint from the seeds and from the first
  benchmark, at least six routing cases per command and scope and at least eight escalation cases
  per nested scope. Not scored by anything until stage B.
* **Arms**: **H3**, the heads rebuilt at seeds 42, 43, 44 (A1's artifacts), and **S**, the single
  large model under the mask with the stage A recipe, at the same seeds. The shipped heads are
  scored too, as a check on H3.
* **Comparison**: seed by seed (S at seed n against H3 at seed n) and by majority verdict.

Pass conditions for adopting S (all of them):

1. Strict top-1, pooled, majority verdict: S not below H3, and not below by McNemar at p < 0.05
   in any single seed.
2. In-list, pooled, majority verdict: S not below H3 by more than 3 points, and McNemar does not
   show H3 better at p < 0.05.
3. Escalation recall: S above H3 on the majority verdict; false escalation on routing cases at
   most 3% in every seed.
4. No scope with at least 30 routing cases loses more than 10 points of in-list or of top-1.
5. Seed agreement of S on case verdicts not more than 5 points below H3's own.
6. Disk per version at most 40% of H's on the five-context workflow; latency p95 at most 120% of
   H's, both measured on an otherwise idle machine.

Predictions: S passes 1, 3, 5 and 6. Condition 2 is the uncertain one for retail and depends on
what A2 finds; condition 4 is at risk for retail only.

If S passes, it becomes the classifier engine behind the detector contract for newly trained
workflows, the per-context layout stays readable, and the default is revisited on a large
workflow. If it fails, the heads stay and the mask's escalation reading is proposed on its own as
an improvement to the heads' "not here" class.

## Recipe for stage B

Written 2026-11-16 after stage A and before any score on the second benchmark.

Stage A findings that fix it:

* The heads rebuilt at seed 42 reproduce the shipped heads on 229 of 239 verdicts (the ten
  differences are a newer ambiguity-threshold rule in the trainer). Across seeds 42, 43, 44 the
  rebuilt heads agree with each other on 54% of verdicts; strict top-1 ranges 27% to 38% pooled and
  in-list 85% to 90%. The shipped heads sit at or above the best of the three: a favourable seed.
* The one-scope in-list loss of round one is inside that range (rebuilt heads 77% by majority, the
  single model 77%). Ranking is as good in the single model; what differs is how often it commits
  to one label. The workflow `wildcard` class is not the cause (weights 0, 4 and 1 are
  indistinguishable).

**S**: one model pair per workflow trained as in round one (every command once, `parameter_value`
with its seven literals, one workflow `wildcard` class of the built-in vague replies at weight 4;
shipped hyperparameters; one stratified split of the workflow's rows, seed 42). Every decision is
answered by the **large model alone**. Ambiguity threshold per scope: the shipped rule on the
masked, renormalised calibration rows whose label is in the scope's mask. Escalation margin:
unchanged (it is 0 in every run). Artifacts: round one's, read large-only; nothing is retrained.

**H3** for the one-scope workflow uses the shipped label set (no built-in vague class), which is
what the shipped head was trained on.

Readings fixed here: every condition is pooled over the two workflows unless it names a scope;
"in every seed" in condition 3 is the pooled rate of each seed. Known risk going in: on the
development set S's false escalation is 2.4% to 2.9% pooled, one case under the limit.

## Results

Stage B, 2026-11-16, second benchmark (264 routing cases, 44 escalation cases), scored once.

| Pooled, majority of three seeds | Strict top-1 | In-list | One label | Escalation recall | False escalation |
|---|---|---|---|---|---|
| H3 (heads rebuilt) | 23.1% | 86.4% | 27.7% | 3/44 | 0.0% |
| Shipped heads | 31.4% | 87.5% | 35.2% | 7/44 | 0.4% |
| S (single large model, masked) | 53.0% | 83.3% | 66.7% | 44/44 | 2.7% |

Per seed, S: top-1 135 / 152 / 140 of 264, in-list 223 / 213 / 222, false escalation 10 / 11 / 6
(3.8%, 4.2%, 2.3%), escalation 44/44 each, seeds agree on 80% of verdicts. H3: top-1 78 / 64 / 68,
in-list 232 / 226 / 215, escalation 4 / 4 / 7, seeds agree on 62%.

| Condition | S |
|---|---|
| 1 strict top-1 | pass (30 points above H3) |
| 2 in-list within 3 points | **fail** (3.1 points below) |
| 3 escalation recall above, false escalation at most 3% in every seed | **fail** (recall 44/44; false escalation 3.8% and 4.2% in two seeds) |
| 4 no scope loses 10 points | pass |
| 5 seed agreement | pass (80% against 62%) |
| 6 disk and latency | pass |

**Decision: S is not adopted as it stands.** Both failures are the same fault: S commits too
readily. It answers two cases in three with one label where the heads answer one in three, and its
lone labels are right 79% of the time against 83% to 89% for the heads; its escalation margin is 0.
The thresholds and the margin are computed on rows the model was trained beside (the
in-distribution split), where it is almost never wrong, so they come out too low. The ranking
itself is not worse (stage A).

Next: `docs/single_intent_model_preregistration_v3.md` keeps the model and changes only where the
thresholds and the margin are computed.
