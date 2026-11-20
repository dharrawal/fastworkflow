# Pre-registration: one intent model for the workflow

Experiment of `fix-yu3l.8` (task `fix-yu3l.8.1`). Written before any training run of the new model.
The design it tests is section 6 of `docs/unified_command_pipeline_design.md`.

## Question

Can one classifier for the whole workflow, its output masked to the commands of the scope being
asked, replace the per-context heads without losing routing quality or escalation recall?

## Arms

| Arm | Model | Context enters by | Tiers |
|---|---|---|---|
| **H** (baseline) | One pair per context, as shipped | A separate model per context | tiny, then large |
| **M2** | One pair for the workflow | The output mask only | tiny, then large |
| **M1** | One model for the workflow | The output mask only | large alone |
| **C2** | One pair for the workflow | The mask, and the context name as the first segment of a paired input | tiny, then large |

M2 is the candidate. M1 answers "is the tiny tier worth keeping"; C2 answers "does the context as
input add anything to the mask".

## Data

* **Training rows**: the rows the shipped heads were trained on, rebuilt from the utterance caches
  with no model call (a cache miss stops the run). For the single model every command appears once,
  with the union of its rows; `parameter_value` keeps its seven literals; one workflow-wide
  `wildcard` label holds the built-in vague replies. The per-context "not here" seed rows of the
  heads are not used: under the mask, "not here" is "the best label is outside the mask".
* **C2 only**: each row is repeated for every scope whose mask holds its label, with that scope's
  name as the first segment.
* **Calibration split**: the same stratified in-distribution split the heads use (seed 42), taken
  per scope after masking, for the thresholds.
* **Benchmark**: `tests/fixtures/intent_benchmarks/library.json` and `retail_workflow.json`,
  hand-written, disjoint from every seed, written before this experiment and not changed during it.
  Never used for training or calibration.
* **Hyperparameters**: the shipped ones, unchanged (base checkpoints, 12 and 5 epochs, learning
  rates 1e-4 and 5e-5, batch 10, length 128); the built-in vague replies weighted 4 times as shipped.
* **Seeds**: model seeds 42, 43, 44 for every single-model arm; the baseline is the shipped
  artifacts (deterministic given the cached data).

## Decision rule in a scope (fixed before the runs)

1. Softmax over all labels. `p_in` is the largest probability inside the scope's mask, `p_out` the
   largest outside it, not counting `parameter_value`.
2. `parameter_value` or the workflow `wildcard` as the overall best label: `bare_value` /
   `not_mine`, as today.
3. `not_mine` when `p_out > p_in + margin`. The margin is calibrated per workflow on the
   calibration split as the smallest value that keeps in-scope rows from escalating more than 2% of
   the time; a foreign command with the same simple name as a local one never causes escalation.
4. Otherwise renormalise inside the mask; one label when its probability is above the scope's
   ambiguity threshold (computed by the shipped rule for the tier that answered), else the top 3.
5. Two tiers: the tiny model answers when its masked confidence is at or above the scope's tier
   threshold (the shipped sweep), else the large model.

## Metrics

On the benchmark, per workflow and pooled: routing top-1, routing in-list, the share of routing
cases answered with one label, escalation recall (a lone `not_mine`), false escalation rate on
routing cases. On the calibration split: in-distribution F1. Operational: model bytes on disk per
version, resident memory after loading every scope, training wall time (same machine, CPU),
latency per decision (mean, p95). Paired comparisons against H by exact McNemar on case verdicts,
pooled over seeds by majority verdict per case; 95% Wilson intervals on every rate.

## Predictions (M2 against H)

Written before the baseline was scored on the benchmark and before any single-model run, so every
prediction is relative to H; the baseline numbers are in the results section.

| Metric | Prediction |
|---|---|
| Routing top-1, library | Within 3 points of H, either direction. Commands shared by several contexts gain (one set of rows instead of four diluted copies); nothing else moves much |
| Routing in-list, library | Within 2 points of H |
| Routing top-1 and in-list, retail | Within 2 points of H (one scope, so the model is the same shape; only the reserved classes differ) |
| Escalation recall, library | At or above H. This is the uncertain one: H learns a "not here" class from one seed per foreign command, the mask reads it off the label the model already knows |
| False escalation on routing cases | At most 3% |
| In-distribution F1 | Lower than H's mean by up to 3 points (more labels in one softmax) |
| Disk per version, library | About 290 MB against about 1.4 GB (five contexts) |
| Training time, library | Under half of H's |
| Latency | Same as H within noise (one forward pass either way) |
| M1 against M2 | Top-1 within 1 point; latency higher (every decision pays the large model) |
| C2 against M2 | No gain above noise on top-1; escalation recall no better |

## Pass conditions for swap-in (all of them, on M2; M1 or C2 may be adopted instead only if it
passes all of them and beats M2 on a metric that matters)

1. Routing top-1, pooled: not worse than H by more than 3 points, and the McNemar test does not
   show H better at p < 0.05.
2. Routing in-list, pooled: not worse than H by more than 2 points.
3. **Escalation recall** (a condition of its own): not worse than H by more than one case in
   twenty, and the false escalation rate on routing cases at most 3%.
4. No scope loses more than 10 points of top-1 on at least 20 cases.
5. Disk per version at most 40% of H's on the five-context workflow.
6. Latency p95 not above H's by more than 20%.
7. The three seeds agree with each other on at least 90% of case verdicts.

If any condition fails, the per-context heads stay, the reason is recorded on the issue, and the
single model remains available only as an experiment arm.

## What is not tested here

The external 160-command workflow the earlier studies used is not in this repository; the result
holds for the bundled examples and is re-checked on a large workflow before any default changes
there. Selective retraining is given up by design (one model, retrained whole); its cost is the
training time above.

## Results

(Filled in after the runs; nothing above this line changes once the first run starts.)

Run 2026-11-16 on CPU, seeds 42, 43, 44; single-model arms are the majority verdict of the three
seeds. Benchmark: 207 routing and 32 escalation cases (library 145 + 32, retail 62). Top-1 is the
strict one (a lone correct label).

| Pooled | Top-1 | In-list | One label | Escalation recall | False escalation |
|---|---|---|---|---|---|
| H | 42.0% | 89.9% | 46.4% | 3/32 | 0.0% |
| M2 | 37.2% | 87.9% | 47.3% | 32/32 | 1.9% |
| M1 | 58.0% | 87.4% | 69.6% | 32/32 | 2.4% |
| C2 | 49.3% | 87.9% | 56.5% | 29/32 | 0.0% |

Against H by exact McNemar, top-1: M2 42 lost / 32 won (p = 0.30), M1 17 / 50 (p = 7e-5),
C2 27 / 42 (p = 0.09). Escalation: every arm wins 26 to 29 cases and loses none.
Disk for the five-context workflow: 1442 MB (H), 288 MB (M2, C2), 269 MB (M1). Resident memory
1019 MB against about 220 MB. Training 266 s (H, from its manifest) against about 125 s. Latency
p95 22.5 ms (H), 20.5 (M2), 19.0 (M1).

| Condition | M2 | M1 | C2 |
|---|---|---|---|
| 1 top-1 | fail (4.8 points lower) | pass (16.0 higher) | pass |
| 2 in-list | pass (1.9 lower) | fail (2.4 lower) | pass |
| 3 escalation | pass | pass | pass |
| 4 no scope loses 10 points | fail (Library, retail) | pass | fail (retail) |
| 5 disk | pass | pass | pass |
| 6 latency | pass | pass | pass |
| 7 seeds agree on 90% of verdicts | fail (61%) | fail (77%) | fail (60%) |

**Decision: no arm passes; the per-context heads stay for now.**

What the runs showed, against the predictions:

* Escalation was predicted "at or above H" and marked uncertain. It is the clearest result: the
  shipped "not here" class is returned alone 3 times in 32; reading "not here" off the mask gets
  all 32, with 2% false escalations.
* The tiny tier was predicted to be worth about a point. It costs 21 points of top-1 under the
  mask: the shipped threshold sweep, run on masked probabilities, lets the tiny model keep most
  decisions at confidences below its ambiguity floor, so they become lists. The large model alone
  is the better single model and is no slower.
* The context as input (C2) adds nothing reliable to the mask, as predicted.
* Not predicted: on retail, one scope where the mask changes nothing, every single-model arm loses
  13 points of in-list. The differences from H there are the workflow-wide `wildcard` class and the
  training draw; which one is the cause is not known from these runs.
* Condition 7 cannot be judged: the agreement of H with itself across seeds was never measured, and
  many cases sit at the ambiguity threshold.

Readings chosen where the text above was not exact (recorded so a rerun can match): one stratified
split of the workflow's rows (seed 42) with each scope calibrating on its masked part; the persona
holdout of the shipped heads reproduced; the escalation margin came out 0 in all runs.

Follow-up: `docs/single_intent_model_preregistration_v2.md` (a new registration with a new
benchmark, since this one has now been looked at).
