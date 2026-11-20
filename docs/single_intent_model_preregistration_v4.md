# Pre-registration, fourth round: a frozen sentence encoder with a linear head

Follows `docs/single_intent_model_preregistration_v3.md`. Written 2026-11-17 after the development
runs below and before any score on the third benchmark.

## What changed since round three

The fine-tuned single classifier (arm S) ranks well but commits too readily, and its thresholds could
not be set from rows it was trained beside. This round replaces the fine-tuned model:

* **Encoder**: a sentence encoder pre-trained on similarity pairs, used frozen (never trained here).
  Requests are embedded once (the encoder's own pooling, L2-normalised, no instruction prefix).
* **Head**: multinomial logistic regression (C = 100) over every label of the workflow, fitted on the
  rows arm S was trained on (all of them; there is no calibration split). One head per workflow,
  read through the per-scope mask of arm S.
* **Decision in a scope**: `parameter_value` or the workflow `wildcard` as the best label is a bare
  value or "not here", as before. Otherwise, with `p_in` the largest probability inside the mask and
  `p_out` the largest outside it: "not here" when `p_out - p_in >= margin`; one command when
  `p_in >= threshold` and `p_in >= p_out`; else the scope's top three. Probabilities are NOT
  renormalised inside the mask.
* **Threshold and margin, per head**: from five-fold cross-fitted probabilities of the training rows
  (folds grouped by text; each row counted in every scope whose mask holds its label). The margin is
  the smallest at which at most 0.5% of those rows are sent up; the threshold is the lowest at which
  wrong lone answers are at most 1% of them. No benchmark is used.

Development runs (first and second benchmark, both already looked at): the head ranks the right
command first on 80% to 87% of requests against 72% to 76% for arm S; a frozen DistilBERT with the
same head gets 57% to 59%. An MLP head and two encoders concatenated were no better. Cross-fitted
rows are easy (96% to 98% right), which is why the targets above are strict. A zero-shot decision
model (435M parameters) was dropped on CPU speed: 0.4 s at 10 options, 4.8 s at 160.

## Arms

* **E-small**: `BAAI/bge-small-en-v1.5` (33M parameters) with the head.
* **E-base**: `BAAI/bge-base-en-v1.5` (109M parameters) with the head.
* **H3**: the per-context classifiers rebuilt at seeds 42, 43, 44 (round two's artifacts), by majority.
* **S**: round two's single classifier, by majority, for reference.

The heads are deterministic, so E-small and E-base are one run each.

## Confirmation

`tests/fixtures/intent_benchmarks/library_v3.json` and `retail_workflow_v3.json`, scored once, pooled.
An arm **exceeds the per-context classifiers** when, against H3:

1. strict top-1 is higher;
2. in-list is not lower;
3. wrong lone answers (a lone label that is not the expected one, "not here" included) are not higher;
4. escalation recall is higher;
5. false escalation on routing cases is at most 2%.

Operational, reported: request latency on CPU, bytes per workflow, training time.

Predictions: E-base passes all five. E-small passes 1, 2 and 4, and is at risk on 3 and 5 (on the
second benchmark its wrong lone answers were 6.1% against 4.6% for H3).

If an arm passes, it becomes the local model of the detection chain (local model, then the remote
decision model when its key is set, then the user or the agent) and the default layout for training.
The choice between the two encoders is then made on a large workflow.

## Results

Run 2026-11-17, third benchmark (264 routing cases, 44 escalation cases), scored once. Thresholds from
cross-fitting: E-small 0.53 with margin 0.36, E-base 0.52 with margin 0.40.

| Pooled | Strict top-1 | In-list | One label | Wrong lone answer | False escalation | Escalation recall |
|---|---|---|---|---|---|---|
| H3 (per-context, rebuilt, majority) | 16.7% | 80.3% | 21.6% | 3.4% | 0.0% | 3/44 |
| S (single fine-tuned, majority) | 41.7% | 77.3% | 60.2% | 15.9% | 8.0% | 42/44 |
| E-small | 63.6% | 86.7% | 72.3% | 8.7% | 2.7% | 43/44 |
| E-base | 63.3% | 87.1% | 69.7% | 6.4% | 2.3% | 42/44 |

| Condition | E-small | E-base |
|---|---|---|
| 1 strict top-1 higher | pass | pass |
| 2 in-list not lower | pass | pass |
| 3 wrong lone answers not higher | **fail** (8.7% against 3.4%) | **fail** (6.4% against 3.4%) |
| 4 escalation recall higher | pass | pass |
| 5 false escalation at most 2% | **fail** (2.7%) | **fail** (2.3%) |

**Neither arm passes as registered.** The prediction that E-base would pass all five was wrong.

What the failure is: the heads answer with one label on 70% of requests where H3 does on 22%, so their
wrong lone answers are a larger share of all requests even though a lone answer is right more often
(E-base 91%, E-small 88%, H3 77%, S 69%). Cross-fitted rows are easier than real requests, so a 1%
target on them came out as 6% to 9% on the benchmark; the two targets are the dial, and this benchmark
is now spent for setting them.

**Product decision (owner's delegate, 2026-11-17): adopt the head as the local model anyway**, with
the two targets as settings, because it is better than both alternatives on what the chain needs: it
names the right command four times as often as H3, holds the right command in its answer more often,
passes a request up when it should (42 or 43 of 44 against 3), needs no model training (the head fits
in seconds, a few hundred kilobytes per workflow, the same every run), and where it is unsure the
chain's next link decides. The cost is stated: about 3 to 5 more wrong confident routes per hundred
requests than the per-context classifiers on this benchmark. The next gate is a large workflow.

## Addendum: reproduction from the repository, and what shipped (2026-11-17)

`scripts/experiments/embedding_head.py` reproduces this document. `pooled` is the arithmetic above
(the rows of both workflows cross-fitted and calibrated together, no artifact); `product` trains a
copy of each workflow with the product trainer and scores it through the product router.

| Third benchmark, pooled | Threshold, margin | Strict top-1 | In-list | One label | Wrong lone answer | False escalation | Escalation recall |
|---|---|---|---|---|---|---|---|
| E-small, this document | 0.53, 0.36 | 63.6% | 86.7% | 72.3% | 8.7% | 2.7% | 43/44 |
| E-small, `pooled` | 0.529, 0.360 | 63.6% | 86.7% | 72.3% | 8.7% | 2.7% | 43/44 |
| E-small, `product` | library 0.529, 0.443; retail 0.500, 0 | 64.4% | 86.7% | 73.1% | 8.7% | 2.7% | 42/44 |
| E-base, this document | 0.52, 0.40 | 63.3% | 87.1% | 69.7% | 6.4% | 2.3% | 42/44 |
| E-base, `pooled` | 0.518, 0.395 | 63.3% | 87.1% | 69.7% | 6.4% | 2.3% | 42/44 |
| E-base, `product` | library 0.533, 0.459; retail 0.500, 0 | 63.3% | 87.1% | 69.7% | 6.4% | 2.3% | 41/44 |

The `pooled` rows are the document's rows. The head is fitted in float64, as the registered run
fitted it; fitted on the encoder's float32 vectors the optimiser stops at a slightly different head
and one answer of 264 changes (a request whose best label is then the workflow `wildcard` at
0.2313 against a command at 0.2289).

Two things differ in what shipped, both in calibration, and the `product` rows show their effect:

1. **One threshold and one margin per workflow head.** "Per head" in this document was run as one
   pair of numbers from both workflows' rows together (1,594 cross-fitted rows). A workflow is
   trained alone, so its head is calibrated on its own rows: 1,180 for the library, 414 for retail.
   The library's margin is then 0.443 instead of 0.360 with E-small, and one fewer of its
   escalation cases is passed up.
2. **The threshold is never below 0.5.** Retail alone (one context, cross-fitted rows 98.8% right)
   calibrates to 0.208 with E-small: five wrong rows of 414, four allowed by the 1% target. At
   0.208 the head answers 22.8% of the retail benchmark requests with one wrong command; at 0.5
   (the trainer's existing floor for a single label) it is 8.7%, what the pooled threshold gave.
   Two retail requests between 0.5 and 0.529 are answered alone, and rightly, which is the higher
   strict top-1 of the E-small `product` row.

Operational, E-small through the product router on CPU (10 threads; the workflows as above): the
head takes 2.4 to 3.0 s per workflow, of which 2.1 to 2.5 s is embedding the rows and about half a
second the fit and the calibration; the artifact is 49,857 bytes for the library and 35,463 for
retail; a request takes a median 9.9 to 11.6 ms and 17.4 to 19.7 ms at the 95th percentile (13.4 to
17.6 ms and 22.2 to 28.0 ms on one thread). E-base: about 8 s, 94,385 and 67,713 bytes, a median
21.5 to 30.4 ms and 36.5 to 49.3 ms at the 95th percentile.
