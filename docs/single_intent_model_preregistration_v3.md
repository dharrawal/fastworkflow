# Pre-registration, third round: thresholds from rows the model never saw

Follows `docs/single_intent_model_preregistration_v2.md`. Written 2026-11-16 after its results and
before any run described here.

## Change under test

The single model of round two (arm S: one large model for the workflow, output masked to the scope)
is kept as trained. Only its decision numbers change: they are computed on the **persona holdout**
(the utterances of the personas the trainer already holds out, which no model was trained on)
instead of the in-distribution split.

* **Ambiguity threshold, per scope**: the lowest threshold at which a lone label is right at least
  90% of the time on the scope's held-out rows (masked, renormalised). With fewer than 20 held-out
  rows in a scope, the workflow-wide value is used. Floor 0.5, ceiling 0.99.
* **Escalation margin, per workflow**: the smallest margin at which at most 2% of in-scope held-out
  rows escalate, pooled over scopes.
* Nothing else changes: same weights (round one's artifacts), same mask rule, same seeds.

This arm is **S2**. The rule is fixed here; it is not tuned on any benchmark. The first benchmark is
used once, before stage B, as a sanity check that the rule behaves (reported, not acted on).

## Confirmation

A third benchmark nobody has scored: `tests/fixtures/intent_benchmarks/library_v3.json` and
`retail_workflow_v3.json` (same structure and size as the second). Arms: H3 (the rebuilt heads of
round two, seeds 42, 43, 44), the shipped heads, S (round two's numbers, for reference) and S2.

Pass conditions for adopting S2 (the conditions of round two, unchanged, pooled over both
workflows unless a scope is named):

1. Strict top-1 by majority not below H3, and not below it by McNemar at p < 0.05 in any seed.
2. In-list by majority not below H3 by more than 3 points, and McNemar does not show H3 better at
   p < 0.05.
3. Escalation recall by majority above H3; false escalation on routing cases at most 3% in every
   seed.
4. No scope with at least 30 routing cases loses more than 10 points of in-list or of top-1.
5. Seed agreement not more than 5 points below H3's own.
6. Disk at most 40% of the heads' on the five-context workflow; latency p95 at most 120%.

Predictions: S2 gives one label less often than S (about half of cases), keeps most of the top-1
gain over H3 (at least 15 points), brings in-list to within 3 points of H3, and false escalation
under 3% at the cost of a few escalation cases (recall still above 80%). The risk is condition 3:
a margin that removes the false escalations of short, context-dependent requests may cost more
recall than predicted.

An added measurement, not a condition: the share of lone labels that are wrong (the errors a
hybrid setting would inherit), for the shipped heads, H3, S and S2.

If S2 passes, the single model becomes a layout the trainer can produce and the runtime can read
behind the detector contract, opt-in per workflow, with the per-context layout unchanged and still
the default until the result is repeated on a large workflow. If it fails, the heads stay, and the
mask's escalation reading is proposed separately.

## Results

**S2 was withdrawn before stage B; the third benchmark has not been scored by anything.**

The sanity check on the first benchmark showed that the rule does not do what it was written to
do, so running the confirmation would have spent the benchmark on a known answer:

* Every ambiguity threshold landed on the 0.5 floor. The held-out personas' rows come from the same
  generator and seeds as the training rows and are almost as easy (the top label is right on 146
  of 151 library rows and 87 of 89 retail rows), so "right 90% of the time" is met at the floor.
  S2 therefore gives one label **more** often than S (83% of cases against 70%), not less; its
  share of wrong lone labels is no better (13% against 12%).
* The library escalation margin came out at 0.6 to 0.9, set by five utterances of one held-out
  persona written in another language. False escalation fell to under 1%, and escalation recall
  fell with it to 11 / 25 / 13 of 32 by seed (predicted: above 80%).

What the three rounds establish, and what they do not:

1. One model for the workflow, read through the scope mask, **ranks** as well as the per-context
   heads and **escalates** far better (44 of 44 against 3 to 7 of 44 on the second benchmark), at a
   fifth of the disk and memory, with more stable seeds (80% agreement against 62%).
2. It **commits to one label** about twice as often as the heads. Most of the extra commitments
   are right (strict top-1 53% against 23% to 31%), some are wrong (about 9% of requests against
   about 4%), and the in-list rate ends 3 points lower.
3. Neither the in-distribution split nor the persona holdout can calibrate that commitment: both
   are too easy. The per-context heads have the same weakness (their lone labels are wrong 10% to
   15% of the time across seeds); it was hidden by one favourable seed.
4. Not established: any of this on a large workflow, and any calibration method that works.

Product decision (separate from the pass conditions, which were not met): the single model is
made available as an **opt-in layout** with the round-two recipe S exactly as measured, the
per-context layout stays the default, and the documentation states the trade in the numbers above.
Reason: for a workflow that runs the hybrid setting or an agent, more right answers and working
escalation outweigh a 3-point in-list loss, and five times less storage is the stated goal; for a
workflow where a wrong confident route is costly, the heads remain the safer choice. A calibration
method that can be trusted is open work; the third benchmark is kept unscored for it.
