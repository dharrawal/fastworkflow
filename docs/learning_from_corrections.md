# Learning from corrections: what the local head gains

Measured 2026-11-17 for beads task fix-o5wi.6. No product code changed and no model was called: the encoder is the
frozen `BAAI/bge-small-en-v1.5` run locally, the head fit and cross-fit are the product's functions
(`fastworkflow/train/embedding_head.py`); its calibration and decision are the rule as it was until bd fix-o5wi.9 removed
it from the product (kept for these experiments in `scripts/experiments/head_rule_as_experimented.py`). Reproduce with `scripts/experiments/learn_from_corrections.py`
(the command is in its docstring; the large workflow's inputs are passed as paths). "The large workflow" is private:
only numbers appear here.

## Summary

**How the numbers were made.** The "seen in use" requests are taken in a shuffled order (three shuffles; mean and
min-max shown). Each one the current head does not answer with the right command outright (wrong, unsure, passes up
wrongly; for an escalation request, does not pass up) becomes a learned row (text, right command), ADDED to the
training rows. A background worker refits the head with its calibration every 25 corrections and the stream goes on
with the new head. Scores are on a test set the head never learned from. Examples: learn from v1 or v1+v2, test on v3.
Large workflow: its development half split once in two by SHA-256 of (context, utterance), first half "seen", second
"test". Disjointness: training rows are generated utterances, the benchmarks were written apart from them, seen and
test come from different benchmark files (examples) or a hash split (large); test requests whose text equals a
training row or a seen request are dropped (0 were). That is exact-text disjointness only: near-duplicates remain
(share of test requests with a seen request at cosine 0.85 or more: library 34% / 50% for v1 / v1+v2, retail 10% / 22%,
large 11%; with a training row: 40%, 14%, 22%), which if anything flatters every mechanism on the "unseen" side.

**Generalisation (unseen phrasings), refit alone, right command named outright (baseline -> all corrections):**

| workflow, learned from | corrections | right outright | wrong confident | right ranked first | top 3 |
|---|---|---|---|---|---|
| library v1 | 33 | 65.1 -> 71.1 | 8.7 -> 6.2 | 82.6 -> 89.1 | 97.1 -> 97.7 |
| library v1+v2 | 78 | 65.1 -> 75.2 | 8.7 -> 3.7 | 82.6 -> 90.7 | 97.1 -> 98.1 |
| retail v1 | 16 | 63.0 -> 65.2 | 8.7 -> 6.5 | 76.1 -> 82.6 | 85.9 -> 93.5 |
| retail v1+v2 | 42 | 63.0 -> 75.7 | 8.7 -> 3.6 | 76.1 -> 85.5 | 85.9 -> 96.4 |
| large workflow | 110 | 28.6 -> 39.0 | 8.4 -> 5.2 | 68.2 -> 71.4 | 86.4 -> 89.4 |

Expect roughly **+10 points of "named right outright" and -3 to -5 points of wrong confident answers per 80 to 110
corrections** on phrasings the head has not seen (about +13 per hundred on the examples, +9.5 per hundred on the large
workflow; escalation recall does not improve: 95.5 -> 93.9 to 97.7 on library, 61 -> 50 on the large workflow, from
44 and 18 cases, so noise-sized and not a gain). Threshold-free ranking gains are smaller and level off sooner
(large workflow right-ranked-first +3.2, top 3 +3.0; the threshold-free wrong@70 falls 26.0 -> 19.3). Where it levels
off: the large workflow's right-outright curve is 28.6, 32.5, 35.3, 39.2 at 0, 25, 50, 100 corrections and 39.0 at 110, so
the gain flattens near 100 corrections; the example curves (library v1+v2 71.5 at 50, 75.2 at 78; retail 72.8 at 25,
75.7 at 42) were still rising when the corrections ran out, so no ceiling is shown for them. Each correction moves the
head; the first 25 give the most (about +4 points per 25 on the large workflow, +10 on retail v1+v2).

**Recurrence (same requests sent again after being corrected).** The head refit on the corrections answers a
corrected request verbatim right outright 96 to 100% of the time (100% for the corrected ones themselves, 0-8% before),
and 80 to 95% on the same wording with values swapped (wrong confident 0 to 0.7%) and 86 to 93% with light surface variation.
Memorising is therefore close to complete for verbatim repeats without any cache, from the refit alone.

**What comes from recurrence and what from generalisation.** A corrected request that comes back is fixed
by either mechanism (right outright from 0-8% to 100%, about +95 points on those requests). A new phrasing gains about
+10 points (refit) or nothing (cache at any level: +0.6 to +1.8 right outright and +0.4 to +2.5 wrong confident at 0.85,
about zero at 0.90 and above, because it almost never fires). So for a request stream where a share r is a repeat of an
earlier corrected request, the benefit is roughly 95 r + 10 (1 - r) points of right outright: **recurrence pays more
than generalisation as soon as r exceeds about 10%, and it is still +1 point at r = 1%.** The refit gives both; the
cache gives only the recurrence half and costs something (below). Putting the refit's own result in numbers: on a stream
with no repeats the head improves about 10 points per 80-110 corrections; on one where one request in five is a repeat
of a corrected one, about 27.

**The similarity cache, as it is today (cosine 0.85) versus alternatives.** At 0.85 a stored request captures
a request that belongs to a DIFFERENT command on 1.6 to 2.5% of unseen requests (1.7, 1.6, 2.5, 2.4 on library v1,
library v1+v2, retail v1+v2 and large with corrected requests stored); of the unseen requests it answers, 53, 21, 58 and
52% are wrong. With every seen request stored (not just corrected ones) it answers 24 to 40% of unseen library requests
and captures 4.7%. At 0.90 captures fall to 0 to 1.7%; at 0.95 to 0 in every run. Verbatim, the cache is right
92 to 100% at 0.85 and 95 to 100% from 0.90 up; on values-swapped requests at 0.85 it answers 10 to 57% of them and is
wrong on 0 to 13% (versus 0 to 0.7% for the refit alone); on light surface variation it answers 24 to 66% at 0.85, 17
to 57% at 0.95, but exact normalised text only 11 to 35%. Added to the refit, the cache on unseen requests changes
right outright by -2.5 to +0.6 and wrong confident by +1.5 to +2.6 (it overrides a head that is right with a neighbour's
label); on verbatim repeats at 0.85 it lowers right outright by 2 to 3 points against the refit alone. Its one clear
advantage is time: a correction works at once, before the next refit.

**Poisoning.** 5% of the learned rows given a wrong label (another command of the same context) cost
about half of the generalisation gain on library v1+v2 (75.2 -> 67.8) and almost nothing on retail (75.7 -> 74.6) or the
large workflow (39.0 -> 39.6); 10% cost 5.4 (library v1+v2), 5.4 (retail v1+v2), 0.7 (large) points. Wrong confident
answers do not rise (the head becomes less sure, so the loss shows up as requests handed on, +5 to +7 points at 10% on
library v1+v2 and retail v1+v2). A poisoned cache is worse: with 10% wrong labels the cache adds 2.4 to 2.6 points of
wrong confident answers.

**Variants at all corrections** (right outright; full rows below): adding EVERY seen request (not only the corrected
ones) is better than corrections alone: library v1+v2 82.6 vs 75.2, retail v1+v2 77.2 vs 75.7, large 41.6 vs 39.0, and
wrong@70 falls further (library 0.6 vs 1.0), at the price of several times more rows (library: 393 seen requests against 78 corrections). Weighting a
learned row x3 or x5 changes right outright by +0.6 to +2.3 in library and the large workflow and by -2.5 to -3.6 in retail: not worth a knob. Learned rows
concentrate on the labels the head confuses: three labels take 38 to 52% of them in library and retail (29 and 21
labels) and 19% in the large workflow (87 labels, 51 to 53 of them receive one).

**Time of one refit plus recalibration** (vectors cached, one thread, median of 3): library 0.6 s (799 rows), retail
0.6 s (514 rows), large workflow 4 to 8 s (2,472 rows: fit 0.9 to 1.6 s, five-fold calibration 3.3 to 6.4 s; the
spread is other work on this machine). A background worker can do it every few corrections without the user noticing.

## What a product implementation should and should not do

1. Record every correction as a learned row with provenance (request, command, context, source, time) in its own file.
   It is cheap and every number here depends on it; recording needs no refit to be worth having.
2. Refit in the background in batches (every 10 to 25 corrections, or on a timer) on seeds + learned rows, with the
   existing cross-fitted calibration; a refit takes under 1 s on the examples and under 10 s on the large workflow. The gain
   is about +10 points right outright per 80 to 110 corrections, flattening around 100 corrections on the one
   workflow big enough to show it. Expect no help for escalation (pass-up) recall.
3. The refit already memorises recurring requests (96 to 100% on a corrected request sent again, 80 to 95% on value-swapped
   and 86 to 93% on lightly varied ones). Keep the cache only for the minutes before a refit, and raise its threshold from 0.85 to
   0.90 at least: at 0.85 it costs 1.6 to 2.5 points of unseen requests answered with a neighbour's command for a recurrence benefit the refit
   already gives. Exact-normalised text match is risk-free but recovers only a third of light variations.
4. Guard the rows: show them for review, allow removal, never let a learned row replace a seed. Wrong rows at 5% cost
   little in two of three workflows and half the gain in the third; at 10% half the gain in two and little in the third. Count a row only after
   a second agreeing correction, or when it follows the user's explicit choice (a pick at a question), not a silent
   redirect.
5. Add the confirmed-right requests too, when they are known to be right (a pick the head already had right, an
   approved plan): adding every seen request beat corrections alone in all three workflows.
6. Do not weight learned rows and do not fine-tune the encoder on this evidence; the weighting results are noise-sized.
7. Check the refit on held-out cross-fitted folds before swapping it in (as the design says); the calibration already
   moves the threshold with the rows.

Caveats: test sets are small (library 172 routing requests, retail 92, large 154; min-max over three shuffles is shown but
the test set is the same, so differences of 2 to 3 points are noise); the "corrections" are benchmark requests replayed in
order, not real use (a real user's corrections may repeat, drift or be wrong in other ways than the sibling-label poison here);
the large workflow's seen and test halves share one benchmark file, so near-paraphrases between them can flatter every
mechanism; the 100+ range is shown by one workflow only. The value-swap sets are small (library 48 / 100, retail 44 / 105, large 25 requests).

## Product implementation (bd fix-o5wi.6)

Built as the sections above recommend: `fastworkflow/intent/intent_learning.py` (the store, what teaches, the calibration),
`fastworkflow/train/head_refit.py` (the refit and its gate) and the `fastworkflow learn` and `fastworkflow learned`
commands. The settings, the store and the rules are in AGENTS.md ("Learning from use") and the README. In short: every
resolved request is recorded; only confirmed rows (a pick at a question, a review that showed the command and was
accepted, a plan switched in review, a choice after you_misunderstood) teach, never a name match; the train stores the
vectors it computed (`rows.npz`, 1,028,980 bytes for the library example and 658,080 for retail), so a refit embeds only the
learned texts; a refit is published as a new version only when it passes the gate; the commit threshold can only be
lowered; a running process picks up the published head within 3 seconds.

Where the product differs from the experiment above: there is no similarity cache (it was removed, bd fix-o5wi.8);
the refit is gated, which the experiment did not do; and the gate's baseline for the learned rows is the same rows held
out of a fit on the base rows alone, not the head being replaced, because after an earlier refit that head has seen some
of them.

### Measurement through the product code

`scripts/experiments/learning_from_use_product.py` trains a temporary copy of each example with `train_workflow` (so the
copy has real versions), plays the development benchmarks 1 to 3 (routing requests, shuffled, library 489 and retail 246) as
a stream through the product router with truth as the confirmation, and writes every row with `intent_learning.record`: a
confident right answer is an unconfirmed `local` row; an answer that is not confident is a which-command question the user
answers with the truth (`user_choice`); a confident wrong answer is a review switched to the truth (`review_switch`, the
wrong answer kept as `rejected_label`). After every 25 new teaching rows the product refit and calibration run
(`head_refit.learn`), as `INTENT_LEARNING_REFIT=background` would, and the next request is answered by the new head
(`intent_detector.refresh_stale_head`). The fourth benchmark (library 172 and retail 92 routing requests, never played) is
scored before and after, with 3 shuffles of the stream. No model is called.

| workflow | teaching rows | refits published (of attempts) | v4 right outright | v4 wrong confident | v4 handed on |
|---|---|---|---|---|---|
| library | 139, 135, 142 | 4, 4, 2 (10 of 18 attempts) | 50.0 -> 66.9, 69.2, 67.4 (mean 67.8) | 3.5 -> 2.9, 4.1, 5.2 (mean 4.1) | 46.5 -> 30.2, 26.7, 27.3 |
| retail | 87, 100, 82 | 3, 2, 2 (7 of 13 attempts) | 38.0 -> 65.2, 64.1, 62.0 (mean 63.8) | 2.2 -> 0.0, 0.0, 0.0 | 59.8 -> 34.8, 35.9, 38.0 |

Pooled over the six runs the head's right-outright share on the fourth benchmark rose by 17.8 points (library) and 25.8
points (retail), for 135 to 142 and 82 to 100 teaching rows. Library escalation requests (46) keep the ancestor command among
the candidates 45 of 46 times before and after, and the head answers none of them with one label. Of the rows, 4 to 7 per
run (library) and 1 to 3 (retail) were confident wrong answers (corrections); the rest were questions.

Cost: a published refit took 4.5 to 5.3 s for library (embedding 50 to 100 new texts, the encoder on this machine's CPU)
and 2.8 to 3.1 s for retail, the gates included. The head's folder was 1,126,537 bytes before and 1,127,295 after (library),
726,801 and 727,597 (retail): the learned rows are not stored in `rows.npz`, which is rebuilt from the store each refit.

Reading. The direction reproduces the experiment above (right outright up, handed-on down, wrong confident not up in
retail), but the size does not match its rule of thumb of +6 to +10 points per 80 to 110 corrections: here it is +18 and
+26 points per 90 to 140 rows. Three reasons not to read that as a better product: the fourth benchmark's baseline is lower
than the experiment's test sets (50 and 38 against 65 and 63 percent), so there is more to gain; every unconfident request is
confirmed with the truth by an ideal user; and the stream and the fourth benchmark come from the same authors, so their
phrasings are close. Library wrong confident rose from 3.5 to 4.1 percent on average (2.9 to 5.2 across shuffles, 6 to 9
requests of 172) where the experiment saw it fall by 3 to 5 points; retail fell to zero. The gate does not prevent that: it
tests the learned rows by cross-fit, not the fourth benchmark, and it is literal, so on 60 to 140 rows one row decides it.
In the dry runs on library the gate failed 2 of 7 times on a single more confident wrong answer while right outright rose
(for example 71.7 to 73.3 percent with wrong confident 2.9 to 4.3 percent), and in the stream 8 of 18 refit attempts
on library and 6 of 13 on retail were rejected (every reason is a gate failure; they are in the run's results). The numbers above and below are for the first gate (0.5 points on the base rows, no tolerance on the
learned rows); the gate was then loosened, see "The gate with tolerances" below.

### The calibration decision on that stream

The stream gives no calibration: the threshold stays 0.65 in all six runs. The rule needs at least 200 confirmed rows and the
runs have 82 to 142. The candidates, for the three library shuffles (rows in the band [t, 0.65) with the head's top label
right or wrong): t = 0.50 has 60, 50 and 42 rows of which 4, 6 and 5 wrong; 0.55 has 38, 38 and 29 of which 2, 4 and 3; 0.60 has
24, 22 and 12 of which 0, 2 and 1. Even with 200 rows the wrong shares would be 2 to 4 percent of all confirmed rows for 0.50
and 1 to 3 for 0.55, over the 1 percent bar. Retail (t = 0.50: 32, 31 and 25 rows of which 2, 3 and 3 wrong; 0.55: 16, 18, 18
of which 0, 1, 1; 0.60: 7, 11, 10 of which 0, 0, 1) is closer but also short of 200. So on these examples a lowered
threshold would need both more evidence and a cleaner band; the rule says so instead of guessing.

### A large private workflow, through the product code

Measured 2026-11-18 on a temporary copy of a large private workflow (87 labels, 16 contexts, 2,538 rows, encoder
e5-base-v2, threshold 0.65, schema 2) trained by the product trainer, with its `rows.npz` (3,411,085 bytes). The remote
decision model and the synthetic-data model were not called (the planner key was made ineffective after the product
imports, and the check `matcher_available()` was false). The stream was the 294 routing requests of the development
half of the second benchmark (343 cases, 49 escalation cases not played), shuffled 3 ways, with truth as the
confirmation, refit every 25 new teaching rows through `head_refit.learn`. The scored half has 321 cases (282 routing,
39 escalation), asked of each case's own context with every command of the workflow reachable, before and after.

| | teaching rows | refits published (of attempts) | right outright | wrong confident | handed on with the right command | handed on without it |
|---|---|---|---|---|---|---|
| before (all shuffles) | 0 | 0 | 64 (22.7%) | 11 (3.9%) | 184 (65.2%) | 23 (8.2%) |
| shuffle 0, after | 196 | 2 of 8 | 88 | 7 | 162 | 25 |
| shuffle 1, after | 201 | 1 of 9 | 80 | 8 | 162 | 32 |
| shuffle 2, after | 202 | 2 of 9 | 79 | 8 | 169 | 26 |
| mean, after | 199.7 | 1.7 of 8.7 | 82.3 (29.2%) | 7.7 (2.7%) | 164.3 (58.3%) | 27.7 (9.8%) |

Wrong confident, split by what the head answered (before, then the after mean): the wrong command 3 then 4.3, the vague
reply class 4 then 2.7, a bare value 4 then 0.7. Escalation requests (39): the head answered none of them with one label
before or after, and the ancestor command was among the candidates 37 of 39 before and 37, 38, 38 after.

Learning curve, shuffle 0 (the product gate as shipped), right outright by requests played: 64 at 0, 70 at 35, 70 at 68, 88
at 110 and 88 from there to 294. The head published at 25 and at 75 teaching rows and then none of the six later attempts
passed, so the curve is flat from 75 rows although right outright of the learned rows was higher in every one of those
attempts but one.

Cost: a refit, gates included, took 32 to 36 s on this machine's CPU (published or not), against 3 to 5 s on the
examples. The gate cross-fits the 2,538 base rows in five folds each time.

Why attempts were rejected (21 rejected attempts in the three shuffles, the product gate as shipped). The condition that
rejected nearly all of them is the first one, the base rows' cross-fit accuracy: 19 of 21 failed it, the cross-fit
accuracy of the base rows fell from 92.8 percent to between 91.7 and 92.3 against the 0.5 point tolerance. The
learned rows' condition failed 18 of 21 (wrong confident higher in 17, right outright lower in 1); 16 failed both.

What the strict "wrong confident not higher" condition cost. In 20 of the 21 rejected attempts the learned rows were right
outright more often with the refit. Allowing wrong confident to rise by at most one learned row would have passed the
learned-rows condition in 8 of those 20 (3, 3 and 2 per shuffle), but in all but one of them the base-rows condition
failed too, so a tolerance on that single condition would have published 1 more refit in the three shuffles (shuffle 0, at
150 rows: learned right outright 3.3 to 15.3 percent, wrong confident 1 to 2 rows). Simulated, with that tolerance
patched in a scratch script only: shuffle 0 ends at 114 right outright, 3 wrong confident, 138 handed on with the right
command, 27 without (three refits published); shuffles 1 and 2 are unchanged (80 and 79, 8 wrong confident). Mean after
the tolerance, 91.0 right outright (32.3%), 6.3 wrong confident (2.2%), 156.3 handed on with and 28.3 without, escalation
37.3 of 39 with the ancestor among the candidates and none answered with one label. The base-rows condition (a
cross-fit accuracy drop of 0.5 to 1.1 points) is the larger cost on a workflow of this size, not the one-row noise of the
second condition.

Calibration. The threshold stays 0.65 in two shuffles and is lowered to 0.50 in one (shuffle 2: 202 confirmed rows,
32 in the band [0.50, 0.65) of which 2 wrong, a wrong share of 0.99 percent of the confirmed rows). Shuffle 0 has 196 confirmed
rows (below the 200 needed; band rows 33, 22 and 13 for 0.50, 0.55 and 0.60 with 3, 1 and 0 wrong); shuffle 1 has 201
confirmed rows but only 20, 14 and 4 band rows (1, 0 and 0 wrong), below the 30 needed. The decision therefore sits on
the edge of the evidence bars: a lowered threshold appears only in the run that happens to land 202 rows with a 0.99
percent wrong share.

### The gate with tolerances (2026-11-18)

The first gate rejected 21 of 26 refit attempts on a large private workflow although the refit was right outright more often
on the held-out learned rows in 20 of the 21, and learning stopped after about 75 teaching rows. Confirmed real requests
outweigh synthetic rows, and the priority stays few confident wrong answers, so the gate has two named tolerances
(`head_refit.py`): the base rows' cross-fit accuracy may fall by 1.5 points (was 0.5), and on the held-out learned rows wrong
confident may be higher by one row when right outright is higher by at least three rows (otherwise not higher, and right
outright never lower). The gate report says which rule let a refit through. The base rows' cross-fit without the learned
rows depends only on the stored rows, so its totals are kept in the learning store and reused by later refits of the
same lineage; only the refit side is computed again. Same streams, shuffles and scoring as above, no model called.

| workflow | teaching rows | refits published (of attempts) | v4 right outright | v4 wrong confident | v4 handed on |
|---|---|---|---|---|---|
| library | 136, 135, 137 | 6, 4, 5 (15 of 18) | 50.0 -> 73.3, 69.2, 69.8 (mean 70.7; first gate 67.8) | 3.5 -> 3.5, 4.1, 4.1 (mean 3.9; first gate 4.1) | 46.5 -> 23.3, 26.7, 26.2 |
| retail | 87, 91, 82 | 3, 3, 3 (9 of 12) | 38.0 -> 65.2, 70.7, 70.7 (mean 68.8; first gate 63.8) | 2.2 -> 0.0, 1.1, 2.2 (mean 1.1; first gate 0.0) | 59.8 -> 34.8, 28.3, 27.2 |

Retail wrong confident is higher than under the first gate (1.1 against 0.0 percent, one request of 92 on average, both
below the 2.2 percent of the base head); library and the large workflow are lower. In the library run of shuffle 0 one of the
46 escalation requests was answered with one label after learning (none before and under the first gate); the ancestor
command stays among the candidates 45 of 46 times. Every rejected attempt (3 library, 3 retail) failed the learned-rows
condition on one or more extra wrong confident rows with right outright up; none failed the base rows. Refits took 2.1 to
5.0 s.

A large private workflow, same streams as the table above (294 requests, 3 shuffles, refit every 25 teaching rows, the
scored half 282 routing and 39 escalation cases):

| | teaching rows | refits published (of attempts) | right outright | wrong confident | handed on with the right command | handed on without it |
|---|---|---|---|---|---|---|
| before | 0 | 0 | 64 (22.7%) | 11 (3.9%) | 184 (65.2%) | 23 (8.2%) |
| first gate, mean after | 199.7 | 1.7 of 8.7 | 82.3 (29.2%) | 7.7 (2.7%) | 164.3 (58.3%) | 27.7 (9.8%) |
| shuffle 0, new gate | 194 | 7 of 8 | 110 | 4 | 142 | 26 |
| shuffle 1, new gate | 188 | 6 of 8 | 113 | 1 | 143 | 25 |
| shuffle 2, new gate | 190 | 4 of 8 | 112 | 3 | 138 | 29 |
| mean, new gate | 190.7 | 5.7 of 8 | 111.7 (39.6%) | 2.7 (0.9%) | 141.0 (50.0%) | 26.7 (9.5%) |

Wrong confident after, split by what the head answered: the wrong command 2.7 (3 before), the vague reply class 0 (4
before), a bare value 0 (4 before). Escalation requests: none answered with one label, the ancestor command among the
candidates 38 of 39 in every shuffle (37 before). Learning curve, shuffle 0, right outright by requests played: 64 at 0, 69
at 35, 78 at 68, 86 at 110, 96 at 151, 104 at 194, 112 at 238, 110 at 269 and 294 (the last attempt was rejected); no
flat stretch, against a curve flat from 75 rows under the first gate. Of the 7 rejected attempts, all failed on wrong
confident higher on the learned rows without the tolerance's conditions met; none failed the base rows (the base rows'
accuracy fell by 0.0 to 1.1 points, inside the 1.5). A refit attempt took 25.7 to 28.4 s once the base rows'
baseline was cached, 34.7 to 35.3 s for the first attempt of each run (which computes it), against 32 to 36 s every time
under the first gate. The calibration lowered the threshold in none of the three shuffles this time (188 to 194
confirmed rows, below the 200 needed).

## Value-swap examples (bundled examples, deterministic)

- library: "Look up the title Persuasion and tell me its author and number of copies" -> "... the title The Citadel ..."
- library: "Tell me which members currently have Emma checked out" -> "... currently have Dune checked out"
- retail: "I don't have my email handy - can you find me by name? It's Daniel Kim, zip 98101" -> "... It's Amanda White, zip 60614"

(The script prints three per workflow; the replacement value is taken from other requests of the same command, chosen by a
hash of the request.) Surface variation per request: lower- or upper-cased first letter, punctuation stripped (or a
question mark added), "please" dropped (or added), one adjacent-letter typo in a word of five letters or more.

## Full output of the run

Columns: `first` = right command ranked first inside the context; `top3`; `wrong@50/70` = wrong lone answers as a share of
all routing requests when the 50% / 70% most confident are committed (threshold-free); `right outright` / `wrong confident`
at the head's own recalibrated threshold and margin; `handed on` = neither (the top three go to the next link);
`esc recall` = escalation requests passed up; `cache answers` = share of requests the similarity cache answered.
Cells are mean (min-max) over three shuffles, in percent.


## library v1->v3

{'seen': 177, 'test': 216, 'test_routing': 172, 'test_escalation': 44, 'dropped_duplicates': 0, 'test_with_seen_near_dup': 0.3425925925925926, 'test_with_training_row_near_dup': 0.4027777777777778}

value-swap examples (before -> after):
  'Look up the title Persuasion and tell me its author and number of copies'
    -> 'Look up the title The Citadel and tell me its author and number of copies'
  'Tell me which members currently have Emma checked out'
    -> 'Tell me which members currently have Dune checked out'
  'Is Dune Messiah lent out to anybody at the moment?'
    -> 'Is Persuasion lent out to anybody at the moment?'
  shuffle 0: 33 corrections, first 0.826 -> 0.890
  shuffle 1: 34 corrections, first 0.826 -> 0.895
  shuffle 2: 31 corrections, first 0.826 -> 0.890

### library v1->v3: learning curve on the test set (mean, min-max over shuffles; % of requests)

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 (n=3) | 82.6 (83-83) | 97.1 (97-97) | 2.9 (3-3) | 5.8 (6-6) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | 95.5 (95-95) |
| 25 (n=3) | 87.0 (87-87) | 97.7 (97-98) | 0.0 (0-0) | 4.3 (3-5) | 69.2 (67-73) | 8.1 (7-9) | 22.7 (19-25) | 97.7 (98-98) |
| all (n=3, mean 33) | 89.1 (89-90) | 97.7 (98-98) | 0.0 (0-0) | 2.3 (2-2) | 71.1 (69-74) | 6.2 (5-8) | 22.7 (19-26) | 97.7 (98-98) |

### library v1->v3: variants at all corrections

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 corrections (baseline) | 82.6 (83-83) | 97.1 (97-97) | 2.9 (3-3) | 5.8 (6-6) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | 95.5 (95-95) |
| refit x1 (main) | 89.1 (89-90) | 97.7 (98-98) | 0.0 (0-0) | 2.3 (2-2) | 71.1 (69-74) | 6.2 (5-8) | 22.7 (19-26) | 97.7 (98-98) |
| (a) every seen request added | 90.7 (91-91) | 97.7 (98-98) | 0.0 (0-0) | 0.6 (1-1) | 79.7 (80-80) | 6.4 (6-6) | 14.0 (14-14) | 97.7 (98-98) |
| (b) learned rows x3 | 89.1 (89-90) | 97.7 (98-98) | 0.0 (0-0) | 2.9 (3-3) | 73.3 (70-76) | 5.8 (5-6) | 20.9 (19-24) | 97.7 (98-98) |
| (b) learned rows x5 | 89.1 (89-90) | 97.7 (98-98) | 0.0 (0-0) | 3.1 (3-3) | 73.4 (70-75) | 6.2 (6-6) | 20.3 (19-23) | 97.7 (98-98) |
| (d) cache alone (no refit) | 82.6 (83-83) | 97.1 (97-97) | 2.9 (3-3) | 5.8 (6-6) | 64.0 (64-64) | 9.9 (10-10) | 26.2 (26-26) | 95.5 (95-95) |
| (d) cache + refit | 89.1 (89-90) | 97.7 (98-98) | 0.0 (0-0) | 2.3 (2-2) | 70.0 (67-73) | 7.8 (7-9) | 22.3 (19-26) | 97.7 (98-98) |
| (e) 5% wrong labels | 88.6 (88-89) | 97.7 (97-98) | 0.2 (0-1) | 2.9 (3-3) | 68.8 (66-72) | 6.0 (5-7) | 25.2 (22-28) | 95.5 (91-98) |
| (e) 10% wrong labels | 89.0 (88-90) | 97.7 (98-98) | 0.0 (0-0) | 2.9 (3-3) | 70.7 (67-73) | 6.0 (5-7) | 23.3 (20-27) | 97.7 (98-98) |
| (e) 10% wrong + cache | 89.0 (88-90) | 97.7 (98-98) | 0.0 (0-0) | 2.9 (3-3) | 69.4 (66-71) | 7.9 (7-9) | 22.7 (20-27) | 97.7 (98-98) |

recurring sets: {'verbatim': 145, 'values swapped': 48, 'surface variation': 564}

### library v1->v3: unseen phrasings

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 82.6 (83-83) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | - |
| refit alone | 89.1 (89-90) | 71.1 (69-74) | 6.2 (5-8) | 22.7 (19-26) | - |
| cache alone, cosine 0.85 | 82.6 (83-83) | 64.0 (64-64) | 9.9 (10-10) | 26.2 (26-26) | 3.3 (3-3) |
| cache alone, cosine 0.90 | 82.6 (83-83) | 64.5 (65-65) | 9.3 (9-9) | 26.2 (26-26) | 1.6 (1-2) |
| cache alone, cosine 0.95 | 82.6 (83-83) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | 0.4 (0-1) |
| cache alone, exact text | 82.6 (83-83) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 89.1 (89-90) | 70.0 (67-73) | 7.8 (7-9) | 22.3 (19-26) | 3.3 (3-3) |
| refit + cache, cosine 0.90 | 89.1 (89-90) | 70.5 (68-73) | 6.8 (6-8) | 22.7 (19-26) | 1.6 (1-2) |
| refit + cache, cosine 0.95 | 89.1 (89-90) | 71.1 (69-74) | 6.2 (5-8) | 22.7 (19-26) | 0.4 (0-1) |
| refit + cache, exact text | 89.1 (89-90) | 71.1 (69-74) | 6.2 (5-8) | 22.7 (19-26) | 0.0 (0-0) |

### library v1->v3: verbatim

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 91.0 (91-91) | 77.2 (77-77) | 3.4 (3-3) | 19.3 (19-19) | - |
| refit alone | 99.3 (99-99) | 97.2 (96-99) | 0.2 (0-1) | 2.5 (1-4) | - |
| cache alone, cosine 0.85 | 91.0 (91-91) | 96.1 (94-97) | 3.2 (3-3) | 0.7 (0-2) | 25.3 (24-26) |
| cache alone, cosine 0.90 | 91.0 (91-91) | 99.3 (98-100) | 0.0 (0-0) | 0.7 (0-2) | 22.1 (21-23) |
| cache alone, cosine 0.95 | 91.0 (91-91) | 99.3 (98-100) | 0.0 (0-0) | 0.7 (0-2) | 22.1 (21-23) |
| cache alone, exact text | 91.0 (91-91) | 99.3 (98-100) | 0.0 (0-0) | 0.7 (0-2) | 22.1 (21-23) |
| refit + cache, cosine 0.85 | 99.3 (99-99) | 94.7 (93-96) | 3.4 (3-4) | 1.8 (0-3) | 25.3 (24-26) |
| refit + cache, cosine 0.90 | 99.3 (99-99) | 97.2 (96-99) | 0.2 (0-1) | 2.5 (1-4) | 22.1 (21-23) |
| refit + cache, cosine 0.95 | 99.3 (99-99) | 97.2 (96-99) | 0.2 (0-1) | 2.5 (1-4) | 22.1 (21-23) |
| refit + cache, exact text | 99.3 (99-99) | 97.2 (96-99) | 0.2 (0-1) | 2.5 (1-4) | 22.1 (21-23) |

### library v1->v3: values swapped

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 87.5 (88-88) | 79.2 (79-79) | 6.2 (6-6) | 14.6 (15-15) | - |
| refit alone | 99.3 (98-100) | 92.4 (90-94) | 0.0 (0-0) | 7.6 (6-10) | - |
| cache alone, cosine 0.85 | 87.5 (88-88) | 80.6 (79-81) | 4.9 (4-6) | 14.6 (15-15) | 9.7 (8-10) |
| cache alone, cosine 0.90 | 87.5 (88-88) | 78.5 (77-79) | 6.9 (6-8) | 14.6 (15-15) | 5.6 (4-6) |
| cache alone, cosine 0.95 | 87.5 (88-88) | 79.2 (79-79) | 6.2 (6-6) | 14.6 (15-15) | 0.0 (0-0) |
| cache alone, exact text | 87.5 (88-88) | 79.2 (79-79) | 6.2 (6-6) | 14.6 (15-15) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 99.3 (98-100) | 91.0 (90-92) | 2.1 (2-2) | 6.9 (6-8) | 9.7 (8-10) |
| refit + cache, cosine 0.90 | 99.3 (98-100) | 91.0 (90-92) | 2.1 (2-2) | 6.9 (6-8) | 5.6 (4-6) |
| refit + cache, cosine 0.95 | 99.3 (98-100) | 92.4 (90-94) | 0.0 (0-0) | 7.6 (6-10) | 0.0 (0-0) |
| refit + cache, exact text | 99.3 (98-100) | 92.4 (90-94) | 0.0 (0-0) | 7.6 (6-10) | 0.0 (0-0) |

### library v1->v3: surface variation

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 86.7 (87-87) | 71.8 (72-72) | 6.0 (6-6) | 22.2 (22-22) | - |
| refit alone | 95.7 (96-96) | 88.7 (87-90) | 3.3 (3-4) | 8.1 (7-10) | - |
| cache alone, cosine 0.85 | 86.7 (87-87) | 88.2 (87-89) | 4.7 (4-5) | 7.1 (7-8) | 23.8 (23-24) |
| cache alone, cosine 0.90 | 86.7 (87-87) | 89.5 (88-90) | 2.3 (2-2) | 8.2 (8-9) | 19.4 (18-20) |
| cache alone, cosine 0.95 | 86.7 (87-87) | 87.6 (87-88) | 3.2 (3-3) | 9.2 (9-10) | 17.2 (16-18) |
| cache alone, exact text | 86.7 (87-87) | 82.7 (82-83) | 4.4 (4-4) | 12.9 (13-13) | 11.3 (11-12) |
| refit + cache, cosine 0.85 | 95.7 (96-96) | 88.4 (87-89) | 5.3 (5-6) | 6.3 (5-8) | 23.8 (23-24) |
| refit + cache, cosine 0.90 | 95.7 (96-96) | 89.7 (88-91) | 2.9 (2-3) | 7.4 (6-9) | 19.4 (18-20) |
| refit + cache, cosine 0.95 | 95.7 (96-96) | 88.8 (88-90) | 3.1 (3-3) | 8.1 (7-10) | 17.2 (16-18) |
| refit + cache, exact text | 95.7 (96-96) | 88.8 (88-90) | 3.3 (3-4) | 7.9 (6-10) | 11.3 (11-12) |

### library v1->v3: verbatim, corrected requests only

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 59.3 (57-61) | 0.0 (0-0) | 15.7 (15-17) | 84.3 (83-85) | - |
| refit alone | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | - |
| cache alone, cosine 0.85 | 59.3 (57-61) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.90 | 59.3 (57-61) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.95 | 59.3 (57-61) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, exact text | 59.3 (57-61) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, exact text | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |

### library v1->v3: near-miss capture by the cache (share of unseen routing requests)

| stored | hit | captured by a different command | wrong share of hits | exposed to a different command's request |
|---|---|---|---|---|
| corrected @ 0.85 | 3.3 (3-3) | 1.7 (2-2) | 53.3 (50-60) | 1.7 (2-2) |
| corrected @ 0.90 | 1.6 (1-2) | 0.6 (1-1) | 38.9 (33-50) | 0.6 (1-1) |
| corrected @ 0.95 | 0.4 (0-1) | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0-0) |
| every seen @ 0.85 | 24.4 (24-24) | 4.7 (5-5) | 19.0 (19-19) | 5.2 (5-5) |
| every seen @ 0.90 | 4.1 (4-4) | 1.2 (1-1) | 28.6 (29-29) | 1.2 (1-1) |
| every seen @ 0.95 | 0.6 (1-1) | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0-0) |

cache: hit rate 3.3 (3-3), hit precision 46.7 (40-50)
spread of learned rows over labels: [{"learned": 33, "labels_hit": 15, "labels": 29, "top3_share": 0.5151515151515151, "top1_share": 0.24242424242424243, "top3_rows_in_training": [25, 30, 34]}, {"learned": 34, "labels_hit": 15, "labels": 29, "top3_share": 0.5, "top1_share": 0.23529411764705882, "top3_rows_in_training": [25, 30, 34]}, {"learned": 31, "labels_hit": 15, "labels": 29, "top3_share": 0.45161290322580644, "top1_share": 0.1935483870967742, "top3_rows_in_training": [25, 30, 34]}]
refit+recalibration (single thread, vectors cached): {'rows': 756, 'fit_s': 0.2221925969934091, 'calibration_s': 0.8112531970109558, 'total_s': 1.0334457940043649}

## library v1+v2->v3

{'seen': 393, 'test': 216, 'test_routing': 172, 'test_escalation': 44, 'dropped_duplicates': 0, 'test_with_seen_near_dup': 0.5046296296296297, 'test_with_training_row_near_dup': 0.4027777777777778}

value-swap examples (before -> after):
  'Look up the title Persuasion and tell me its author and number of copies'
    -> 'Look up the title Dune and tell me its author and number of copies'
  'Tell me which members currently have Emma checked out'
    -> 'Tell me which members currently have Dune checked out'
  'Is Dune Messiah lent out to anybody at the moment?'
    -> 'Is Persuasion lent out to anybody at the moment?'
  shuffle 0: 76 corrections, first 0.826 -> 0.907
  shuffle 1: 84 corrections, first 0.826 -> 0.907
  shuffle 2: 74 corrections, first 0.826 -> 0.907

### library v1+v2->v3: learning curve on the test set (mean, min-max over shuffles; % of requests)

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 (n=3) | 82.6 (83-83) | 97.1 (97-97) | 2.9 (3-3) | 5.8 (6-6) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | 95.5 (95-95) |
| 25 (n=3) | 87.2 (85-90) | 98.1 (98-98) | 0.8 (1-1) | 4.7 (5-5) | 66.7 (65-69) | 6.4 (6-7) | 26.9 (26-29) | 95.5 (93-98) |
| 50 (n=3) | 89.7 (89-91) | 97.9 (97-98) | 0.0 (0-0) | 1.9 (2-2) | 71.5 (67-76) | 4.5 (4-5) | 24.0 (19-29) | 93.9 (91-95) |
| all (n=3, mean 78) | 90.7 (91-91) | 98.1 (98-98) | 0.0 (0-0) | 1.0 (1-1) | 75.2 (74-76) | 3.7 (3-4) | 21.1 (20-22) | 93.9 (91-98) |

### library v1+v2->v3: variants at all corrections

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 corrections (baseline) | 82.6 (83-83) | 97.1 (97-97) | 2.9 (3-3) | 5.8 (6-6) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | 95.5 (95-95) |
| refit x1 (main) | 90.7 (91-91) | 98.1 (98-98) | 0.0 (0-0) | 1.0 (1-1) | 75.2 (74-76) | 3.7 (3-4) | 21.1 (20-22) | 93.9 (91-98) |
| (a) every seen request added | 93.0 (93-93) | 98.3 (98-98) | 0.0 (0-0) | 0.6 (1-1) | 82.6 (83-83) | 3.5 (3-3) | 14.0 (14-14) | 97.7 (98-98) |
| (b) learned rows x3 | 91.9 (91-92) | 98.6 (98-99) | 0.0 (0-0) | 1.2 (1-1) | 77.5 (76-79) | 4.3 (3-5) | 18.2 (17-19) | 91.7 (91-93) |
| (b) learned rows x5 | 91.9 (91-93) | 99.2 (99-99) | 0.0 (0-0) | 1.2 (1-1) | 76.0 (74-78) | 3.9 (3-5) | 20.2 (18-22) | 89.4 (86-93) |
| (d) cache alone (no refit) | 82.6 (83-83) | 97.1 (97-97) | 2.9 (3-3) | 5.8 (6-6) | 66.9 (66-67) | 9.1 (9-9) | 24.0 (24-24) | 87.9 (86-89) |
| (d) cache + refit | 90.7 (91-91) | 98.1 (98-98) | 0.0 (0-0) | 1.0 (1-1) | 74.6 (74-75) | 5.2 (5-6) | 20.2 (19-21) | 86.4 (84-91) |
| (e) 5% wrong labels | 90.5 (90-91) | 97.9 (97-98) | 0.0 (0-0) | 1.2 (1-2) | 67.8 (65-73) | 2.9 (2-3) | 29.3 (23-33) | 90.9 (84-95) |
| (e) 10% wrong labels | 88.2 (87-91) | 98.4 (98-99) | 0.6 (0-1) | 1.6 (1-3) | 69.8 (67-74) | 3.9 (2-5) | 26.4 (22-30) | 87.1 (84-91) |
| (e) 10% wrong + cache | 88.2 (87-91) | 98.4 (98-99) | 0.6 (0-1) | 1.6 (1-3) | 69.0 (67-73) | 6.4 (5-7) | 24.6 (20-27) | 80.3 (77-84) |

recurring sets: {'verbatim': 317, 'values swapped': 100, 'surface variation': 1233}

### library v1+v2->v3: unseen phrasings

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 82.6 (83-83) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | - |
| refit alone | 90.7 (91-91) | 75.2 (74-76) | 3.7 (3-4) | 21.1 (20-22) | - |
| cache alone, cosine 0.85 | 82.6 (83-83) | 66.9 (66-67) | 9.1 (9-9) | 24.0 (24-24) | 7.4 (6-8) |
| cache alone, cosine 0.90 | 82.6 (83-83) | 65.5 (65-66) | 8.7 (9-9) | 25.8 (26-26) | 3.1 (2-4) |
| cache alone, cosine 0.95 | 82.6 (83-83) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | 0.2 (0-1) |
| cache alone, exact text | 82.6 (83-83) | 65.1 (65-65) | 8.7 (9-9) | 26.2 (26-26) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 90.7 (91-91) | 74.6 (74-75) | 5.2 (5-6) | 20.2 (19-21) | 7.4 (6-8) |
| refit + cache, cosine 0.90 | 90.7 (91-91) | 74.6 (74-76) | 4.3 (4-5) | 21.1 (20-22) | 3.1 (2-4) |
| refit + cache, cosine 0.95 | 90.7 (91-91) | 75.2 (74-76) | 3.7 (3-4) | 21.1 (20-22) | 0.2 (0-1) |
| refit + cache, exact text | 90.7 (91-91) | 75.2 (74-76) | 3.7 (3-4) | 21.1 (20-22) | 0.0 (0-0) |

### library v1+v2->v3: verbatim

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 89.6 (90-90) | 73.5 (74-74) | 4.4 (4-4) | 22.1 (22-22) | - |
| refit alone | 99.5 (99-100) | 97.4 (97-98) | 0.1 (0-0) | 2.5 (2-3) | - |
| cache alone, cosine 0.85 | 89.6 (90-90) | 91.5 (91-92) | 5.5 (5-6) | 3.0 (3-4) | 33.3 (32-34) |
| cache alone, cosine 0.90 | 89.6 (90-90) | 95.3 (94-96) | 0.9 (0-2) | 3.8 (3-4) | 25.0 (25-26) |
| cache alone, cosine 0.95 | 89.6 (90-90) | 95.4 (94-96) | 0.5 (0-1) | 4.1 (4-5) | 23.8 (23-25) |
| cache alone, exact text | 89.6 (90-90) | 95.4 (94-96) | 0.5 (0-1) | 4.1 (4-5) | 23.8 (23-25) |
| refit + cache, cosine 0.85 | 99.5 (99-100) | 92.6 (92-93) | 5.2 (5-5) | 2.2 (2-2) | 33.3 (32-34) |
| refit + cache, cosine 0.90 | 99.5 (99-100) | 97.0 (97-97) | 0.5 (0-1) | 2.5 (2-3) | 25.0 (25-26) |
| refit + cache, cosine 0.95 | 99.5 (99-100) | 97.4 (97-98) | 0.1 (0-0) | 2.5 (2-3) | 23.8 (23-25) |
| refit + cache, exact text | 99.5 (99-100) | 97.4 (97-98) | 0.1 (0-0) | 2.5 (2-3) | 23.8 (23-25) |

### library v1+v2->v3: values swapped

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 93.0 (93-93) | 85.0 (85-85) | 4.0 (4-4) | 11.0 (11-11) | - |
| refit alone | 98.3 (98-99) | 91.7 (91-92) | 0.7 (0-1) | 7.7 (7-9) | - |
| cache alone, cosine 0.85 | 93.0 (93-93) | 80.0 (79-81) | 13.0 (12-14) | 7.0 (6-8) | 26.3 (25-28) |
| cache alone, cosine 0.90 | 93.0 (93-93) | 84.7 (84-85) | 6.7 (6-7) | 8.7 (8-9) | 12.7 (12-13) |
| cache alone, cosine 0.95 | 93.0 (93-93) | 86.7 (86-87) | 3.3 (3-4) | 10.0 (10-10) | 3.0 (2-4) |
| cache alone, exact text | 93.0 (93-93) | 85.0 (85-85) | 4.0 (4-4) | 11.0 (11-11) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 98.3 (98-99) | 83.0 (82-84) | 12.7 (12-13) | 4.3 (3-6) | 26.3 (25-28) |
| refit + cache, cosine 0.90 | 98.3 (98-99) | 88.3 (87-89) | 5.3 (5-6) | 6.3 (5-8) | 12.7 (12-13) |
| refit + cache, cosine 0.95 | 98.3 (98-99) | 91.3 (90-92) | 1.0 (1-1) | 7.7 (7-9) | 3.0 (2-4) |
| refit + cache, exact text | 98.3 (98-99) | 91.7 (91-92) | 0.7 (0-1) | 7.7 (7-9) | 0.0 (0-0) |

### library v1+v2->v3: surface variation

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 85.9 (86-86) | 69.7 (70-70) | 5.8 (6-6) | 24.4 (24-24) | - |
| refit alone | 96.6 (96-97) | 90.6 (90-91) | 1.1 (1-1) | 8.2 (8-9) | - |
| cache alone, cosine 0.85 | 85.9 (86-86) | 86.6 (86-87) | 5.7 (6-6) | 7.7 (7-8) | 30.7 (30-32) |
| cache alone, cosine 0.90 | 85.9 (86-86) | 88.4 (88-89) | 2.1 (2-2) | 9.4 (9-10) | 22.1 (22-23) |
| cache alone, cosine 0.95 | 85.9 (86-86) | 86.2 (86-87) | 2.6 (2-3) | 11.2 (11-12) | 18.8 (18-20) |
| cache alone, exact text | 85.9 (86-86) | 80.5 (80-81) | 3.9 (4-4) | 15.6 (15-16) | 12.1 (12-13) |
| refit + cache, cosine 0.85 | 96.6 (96-97) | 88.7 (88-89) | 5.2 (5-6) | 6.1 (6-6) | 30.7 (30-32) |
| refit + cache, cosine 0.90 | 96.6 (96-97) | 91.4 (91-92) | 1.2 (1-1) | 7.4 (7-8) | 22.1 (22-23) |
| refit + cache, cosine 0.95 | 96.6 (96-97) | 90.9 (90-92) | 1.1 (1-1) | 7.9 (7-8) | 18.8 (18-20) |
| refit + cache, exact text | 96.6 (96-97) | 90.9 (90-91) | 1.1 (1-1) | 8.0 (8-8) | 12.1 (12-13) |

### library v1+v2->v3: verbatim, corrected requests only

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 59.7 (59-60) | 8.0 (7-10) | 16.3 (15-18) | 75.7 (74-77) | - |
| refit alone | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | - |
| cache alone, cosine 0.85 | 59.7 (59-60) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.90 | 59.7 (59-60) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.95 | 59.7 (59-60) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, exact text | 59.7 (59-60) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, exact text | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |

### library v1+v2->v3: near-miss capture by the cache (share of unseen routing requests)

| stored | hit | captured by a different command | wrong share of hits | exposed to a different command's request |
|---|---|---|---|---|
| corrected @ 0.85 | 7.4 (6-8) | 1.6 (1-2) | 21.0 (14-29) | 1.6 (1-2) |
| corrected @ 0.90 | 3.1 (2-4) | 0.6 (1-1) | 19.8 (14-25) | 0.6 (1-1) |
| corrected @ 0.95 | 0.2 (0-1) | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0-0) |
| every seen @ 0.85 | 39.5 (40-40) | 4.7 (5-5) | 11.8 (12-12) | 8.1 (8-8) |
| every seen @ 0.90 | 8.7 (9-9) | 1.7 (2-2) | 20.0 (20-20) | 2.9 (3-3) |
| every seen @ 0.95 | 0.6 (1-1) | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0-0) |

cache: hit rate 7.4 (6-8), hit precision 79.0 (71-86)
spread of learned rows over labels: [{"learned": 76, "labels_hit": 21, "labels": 29, "top3_share": 0.3815789473684211, "top1_share": 0.17105263157894737, "top3_rows_in_training": [25, 34, 30]}, {"learned": 84, "labels_hit": 22, "labels": 29, "top3_share": 0.35714285714285715, "top1_share": 0.14285714285714285, "top3_rows_in_training": [25, 34, 30]}, {"learned": 74, "labels_hit": 22, "labels": 29, "top3_share": 0.35135135135135137, "top1_share": 0.14864864864864866, "top3_rows_in_training": [25, 34, 30]}]
refit+recalibration (single thread, vectors cached): {'rows': 799, 'fit_s': 0.2012065289891325, 'calibration_s': 0.9690598620072706, 'total_s': 1.170266390996403}

## retail v1->v3

{'seen': 62, 'test': 92, 'test_routing': 92, 'test_escalation': 0, 'dropped_duplicates': 0, 'test_with_seen_near_dup': 0.09782608695652174, 'test_with_training_row_near_dup': 0.14130434782608695}

value-swap examples (before -> after):
  "I'd like order #W7730219 cancelled before it goes out"
    -> "I'd like order #W0000001 cancelled before it goes out"
  "I don't have my email handy - can you find me by name? It's Daniel Kim, zip 98101"
    -> "I don't have my email handy - can you find me by name? It's Amanda White, zip 60614"
  'lookup: Sofia Alvarez, 33139'
    -> 'lookup: John Lee, 90210'
  shuffle 0: 16 corrections, first 0.761 -> 0.826
  shuffle 1: 16 corrections, first 0.761 -> 0.826
  shuffle 2: 16 corrections, first 0.761 -> 0.826

### retail v1->v3: learning curve on the test set (mean, min-max over shuffles; % of requests)

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 (n=3) | 76.1 (76-76) | 85.9 (86-86) | 1.1 (1-1) | 7.6 (8-8) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | - |
| all (n=3, mean 16) | 82.6 (83-83) | 93.5 (93-93) | 1.1 (1-1) | 5.4 (5-5) | 65.2 (65-65) | 6.5 (7-7) | 28.3 (28-28) | - |

### retail v1->v3: variants at all corrections

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 corrections (baseline) | 76.1 (76-76) | 85.9 (86-86) | 1.1 (1-1) | 7.6 (8-8) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | - |
| refit x1 (main) | 82.6 (83-83) | 93.5 (93-93) | 1.1 (1-1) | 5.4 (5-5) | 65.2 (65-65) | 6.5 (7-7) | 28.3 (28-28) | - |
| (a) every seen request added | 79.3 (79-79) | 93.5 (93-93) | 0.0 (0-0) | 2.2 (2-2) | 70.7 (71-71) | 7.6 (8-8) | 21.7 (22-22) | - |
| (b) learned rows x3 | 82.6 (83-83) | 94.6 (95-95) | 0.0 (0-0) | 6.5 (7-7) | 66.3 (66-66) | 7.6 (8-8) | 26.1 (26-26) | - |
| (b) learned rows x5 | 81.5 (82-82) | 94.6 (95-95) | 1.1 (1-1) | 5.4 (5-5) | 65.2 (65-65) | 5.4 (5-5) | 29.3 (29-29) | - |
| (d) cache alone (no refit) | 76.1 (76-76) | 85.9 (86-86) | 1.1 (1-1) | 7.6 (8-8) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | - |
| (d) cache + refit | 82.6 (83-83) | 93.5 (93-93) | 1.1 (1-1) | 5.4 (5-5) | 65.2 (65-65) | 6.5 (7-7) | 28.3 (28-28) | - |
| (e) 5% wrong labels | 83.0 (83-84) | 93.8 (93-95) | 0.7 (0-1) | 5.1 (4-5) | 64.1 (63-65) | 6.2 (5-7) | 29.7 (29-30) | - |
| (e) 10% wrong labels | 80.4 (78-82) | 94.2 (93-95) | 0.4 (0-1) | 5.8 (4-8) | 63.0 (57-68) | 6.5 (4-8) | 30.4 (24-36) | - |
| (e) 10% wrong + cache | 80.4 (78-82) | 94.2 (93-95) | 0.4 (0-1) | 5.8 (4-8) | 63.0 (57-68) | 6.9 (4-9) | 30.1 (24-35) | - |

recurring sets: {'verbatim': 62, 'values swapped': 44, 'surface variation': 243}

### retail v1->v3: unseen phrasings

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | - |
| refit alone | 82.6 (83-83) | 65.2 (65-65) | 6.5 (7-7) | 28.3 (28-28) | - |
| cache alone, cosine 0.85 | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | 1.1 (1-1) |
| cache alone, cosine 0.90 | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | 0.0 (0-0) |
| cache alone, cosine 0.95 | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | 0.0 (0-0) |
| cache alone, exact text | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 82.6 (83-83) | 65.2 (65-65) | 6.5 (7-7) | 28.3 (28-28) | 1.1 (1-1) |
| refit + cache, cosine 0.90 | 82.6 (83-83) | 65.2 (65-65) | 6.5 (7-7) | 28.3 (28-28) | 0.0 (0-0) |
| refit + cache, cosine 0.95 | 82.6 (83-83) | 65.2 (65-65) | 6.5 (7-7) | 28.3 (28-28) | 0.0 (0-0) |
| refit + cache, exact text | 82.6 (83-83) | 65.2 (65-65) | 6.5 (7-7) | 28.3 (28-28) | 0.0 (0-0) |

### retail v1->v3: verbatim

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 83.9 (84-84) | 74.2 (74-74) | 3.2 (3-3) | 22.6 (23-23) | - |
| refit alone | 100.0 (100-100) | 98.4 (98-98) | 0.0 (0-0) | 1.6 (2-2) | - |
| cache alone, cosine 0.85 | 83.9 (84-84) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 25.8 (26-26) |
| cache alone, cosine 0.90 | 83.9 (84-84) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 25.8 (26-26) |
| cache alone, cosine 0.95 | 83.9 (84-84) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 25.8 (26-26) |
| cache alone, exact text | 83.9 (84-84) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 25.8 (26-26) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 98.4 (98-98) | 0.0 (0-0) | 1.6 (2-2) | 25.8 (26-26) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 98.4 (98-98) | 0.0 (0-0) | 1.6 (2-2) | 25.8 (26-26) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 98.4 (98-98) | 0.0 (0-0) | 1.6 (2-2) | 25.8 (26-26) |
| refit + cache, exact text | 100.0 (100-100) | 98.4 (98-98) | 0.0 (0-0) | 1.6 (2-2) | 25.8 (26-26) |

### retail v1->v3: values swapped

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 86.4 (86-86) | 72.7 (73-73) | 0.0 (0-0) | 27.3 (27-27) | - |
| refit alone | 100.0 (100-100) | 95.5 (95-95) | 0.0 (0-0) | 4.5 (5-5) | - |
| cache alone, cosine 0.85 | 86.4 (86-86) | 90.9 (91-91) | 0.0 (0-0) | 9.1 (9-9) | 18.2 (18-18) |
| cache alone, cosine 0.90 | 86.4 (86-86) | 88.6 (89-89) | 0.0 (0-0) | 11.4 (11-11) | 15.9 (16-16) |
| cache alone, cosine 0.95 | 86.4 (86-86) | 81.8 (82-82) | 0.0 (0-0) | 18.2 (18-18) | 9.1 (9-9) |
| cache alone, exact text | 86.4 (86-86) | 72.7 (73-73) | 0.0 (0-0) | 27.3 (27-27) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 95.5 (95-95) | 0.0 (0-0) | 4.5 (5-5) | 18.2 (18-18) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 95.5 (95-95) | 0.0 (0-0) | 4.5 (5-5) | 15.9 (16-16) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 95.5 (95-95) | 0.0 (0-0) | 4.5 (5-5) | 9.1 (9-9) |
| refit + cache, exact text | 100.0 (100-100) | 95.5 (95-95) | 0.0 (0-0) | 4.5 (5-5) | 0.0 (0-0) |

### retail v1->v3: surface variation

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 81.5 (81-81) | 70.0 (70-70) | 6.2 (6-6) | 23.9 (24-24) | - |
| refit alone | 97.5 (98-98) | 92.6 (93-93) | 0.8 (1-1) | 6.6 (7-7) | - |
| cache alone, cosine 0.85 | 81.5 (81-81) | 94.2 (94-94) | 2.5 (2-2) | 3.3 (3-3) | 25.5 (26-26) |
| cache alone, cosine 0.90 | 81.5 (81-81) | 93.8 (94-94) | 2.5 (2-2) | 3.7 (4-4) | 24.3 (24-24) |
| cache alone, cosine 0.95 | 81.5 (81-81) | 92.2 (92-92) | 2.9 (3-3) | 4.9 (5-5) | 22.6 (23-23) |
| cache alone, exact text | 81.5 (81-81) | 82.7 (83-83) | 4.5 (5-5) | 12.8 (13-13) | 12.8 (13-13) |
| refit + cache, cosine 0.85 | 97.5 (98-98) | 93.0 (93-93) | 1.2 (1-1) | 5.8 (6-6) | 25.5 (26-26) |
| refit + cache, cosine 0.90 | 97.5 (98-98) | 92.6 (93-93) | 0.8 (1-1) | 6.6 (7-7) | 24.3 (24-24) |
| refit + cache, cosine 0.95 | 97.5 (98-98) | 92.6 (93-93) | 0.8 (1-1) | 6.6 (7-7) | 22.6 (23-23) |
| refit + cache, exact text | 97.5 (98-98) | 92.6 (93-93) | 0.8 (1-1) | 6.6 (7-7) | 12.8 (13-13) |

### retail v1->v3: verbatim, corrected requests only

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 37.5 (38-38) | 0.0 (0-0) | 12.5 (12-12) | 87.5 (88-88) | - |
| refit alone | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | - |
| cache alone, cosine 0.85 | 37.5 (38-38) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.90 | 37.5 (38-38) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.95 | 37.5 (38-38) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, exact text | 37.5 (38-38) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, exact text | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |

### retail v1->v3: near-miss capture by the cache (share of unseen routing requests)

| stored | hit | captured by a different command | wrong share of hits | exposed to a different command's request |
|---|---|---|---|---|
| corrected @ 0.85 | 1.1 (1-1) | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0-0) |
| corrected @ 0.90 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |
| corrected @ 0.95 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |
| every seen @ 0.85 | 9.8 (10-10) | 0.0 (0-0) | 0.0 (0-0) | 1.1 (1-1) |
| every seen @ 0.90 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |
| every seen @ 0.95 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |

cache: hit rate 1.1 (1-1), hit precision 100.0 (100-100)
spread of learned rows over labels: [{"learned": 16, "labels_hit": 11, "labels": 21, "top3_share": 0.5, "top1_share": 0.25, "top3_rows_in_training": [21, 27, 21]}, {"learned": 16, "labels_hit": 11, "labels": 21, "top3_share": 0.5, "top1_share": 0.25, "top3_rows_in_training": [21, 27, 25]}, {"learned": 16, "labels_hit": 11, "labels": 21, "top3_share": 0.5, "top1_share": 0.25, "top3_rows_in_training": [21, 27, 16]}]
refit+recalibration (single thread, vectors cached): {'rows': 491, 'fit_s': 0.09691572398878634, 'calibration_s': 0.3726218430092558, 'total_s': 0.45520912000210956}

## retail v1+v2->v3

{'seen': 154, 'test': 92, 'test_routing': 92, 'test_escalation': 0, 'dropped_duplicates': 0, 'test_with_seen_near_dup': 0.21739130434782608, 'test_with_training_row_near_dup': 0.14130434782608695}

value-swap examples (before -> after):
  "I'd like order #W7730219 cancelled before it goes out"
    -> "I'd like order #W0000001 cancelled before it goes out"
  "I don't have my email handy - can you find me by name? It's Daniel Kim, zip 98101"
    -> "I don't have my email handy - can you find me by name? It's Laura Bennett, zip 77001,"
  'lookup: Sofia Alvarez, 33139'
    -> 'lookup: Emily Carter, 12345'
  shuffle 0: 39 corrections, first 0.761 -> 0.870
  shuffle 1: 43 corrections, first 0.761 -> 0.837
  shuffle 2: 43 corrections, first 0.761 -> 0.859

### retail v1+v2->v3: learning curve on the test set (mean, min-max over shuffles; % of requests)

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 (n=3) | 76.1 (76-76) | 85.9 (86-86) | 1.1 (1-1) | 7.6 (8-8) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | - |
| 25 (n=3) | 84.4 (83-86) | 94.6 (93-96) | 0.0 (0-0) | 3.3 (2-4) | 72.8 (70-76) | 6.9 (5-9) | 20.3 (15-25) | - |
| all (n=3, mean 42) | 85.5 (84-87) | 96.4 (96-97) | 0.0 (0-0) | 1.1 (1-1) | 75.7 (75-77) | 3.6 (3-4) | 20.7 (18-22) | - |

### retail v1+v2->v3: variants at all corrections

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 corrections (baseline) | 76.1 (76-76) | 85.9 (86-86) | 1.1 (1-1) | 7.6 (8-8) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | - |
| refit x1 (main) | 85.5 (84-87) | 96.4 (96-97) | 0.0 (0-0) | 1.1 (1-1) | 75.7 (75-77) | 3.6 (3-4) | 20.7 (18-22) | - |
| (a) every seen request added | 88.0 (88-88) | 97.8 (98-98) | 0.0 (0-0) | 1.1 (1-1) | 77.2 (77-77) | 2.2 (2-2) | 20.7 (21-21) | - |
| (b) learned rows x3 | 84.8 (84-86) | 96.7 (96-98) | 0.4 (0-1) | 2.2 (2-2) | 73.2 (68-76) | 3.6 (3-4) | 23.2 (21-28) | - |
| (b) learned rows x5 | 84.4 (84-86) | 97.1 (97-98) | 0.7 (0-1) | 1.8 (1-2) | 72.1 (71-75) | 3.3 (3-3) | 24.6 (22-26) | - |
| (d) cache alone (no refit) | 76.1 (76-76) | 85.9 (86-86) | 1.1 (1-1) | 7.6 (8-8) | 60.5 (60-61) | 11.2 (11-12) | 28.3 (28-28) | - |
| (d) cache + refit | 85.5 (84-87) | 96.4 (96-97) | 0.0 (0-0) | 1.1 (1-1) | 73.2 (72-75) | 6.2 (5-7) | 20.7 (18-22) | - |
| (e) 5% wrong labels | 84.4 (84-85) | 95.3 (93-97) | 0.0 (0-0) | 1.1 (0-2) | 74.6 (74-76) | 3.6 (2-4) | 21.7 (20-24) | - |
| (e) 10% wrong labels | 85.1 (85-86) | 94.9 (95-96) | 0.4 (0-1) | 1.1 (0-2) | 70.3 (63-74) | 2.5 (1-3) | 27.2 (23-34) | - |
| (e) 10% wrong + cache | 85.1 (85-86) | 94.9 (95-96) | 0.4 (0-1) | 1.1 (0-2) | 67.8 (60-72) | 5.1 (3-7) | 27.2 (23-34) | - |

recurring sets: {'verbatim': 154, 'values swapped': 105, 'surface variation': 606}

### retail v1+v2->v3: unseen phrasings

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | - |
| refit alone | 85.5 (84-87) | 75.7 (75-77) | 3.6 (3-4) | 20.7 (18-22) | - |
| cache alone, cosine 0.85 | 76.1 (76-76) | 60.5 (60-61) | 11.2 (11-12) | 28.3 (28-28) | 4.3 (4-4) |
| cache alone, cosine 0.90 | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | 0.0 (0-0) |
| cache alone, cosine 0.95 | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | 0.0 (0-0) |
| cache alone, exact text | 76.1 (76-76) | 63.0 (63-63) | 8.7 (9-9) | 28.3 (28-28) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 85.5 (84-87) | 73.2 (72-75) | 6.2 (5-7) | 20.7 (18-22) | 4.3 (4-4) |
| refit + cache, cosine 0.90 | 85.5 (84-87) | 75.7 (75-77) | 3.6 (3-4) | 20.7 (18-22) | 0.0 (0-0) |
| refit + cache, cosine 0.95 | 85.5 (84-87) | 75.7 (75-77) | 3.6 (3-4) | 20.7 (18-22) | 0.0 (0-0) |
| refit + cache, exact text | 85.5 (84-87) | 75.7 (75-77) | 3.6 (3-4) | 20.7 (18-22) | 0.0 (0-0) |

### retail v1+v2->v3: verbatim

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 84.4 (84-84) | 70.1 (70-70) | 3.9 (4-4) | 26.0 (26-26) | - |
| refit alone | 99.1 (98-100) | 96.8 (95-99) | 0.6 (0-1) | 2.6 (1-4) | - |
| cache alone, cosine 0.85 | 84.4 (84-84) | 94.4 (92-96) | 2.2 (2-3) | 3.5 (1-6) | 30.1 (28-31) |
| cache alone, cosine 0.90 | 84.4 (84-84) | 96.3 (94-98) | 0.2 (0-1) | 3.5 (1-6) | 27.5 (25-29) |
| cache alone, cosine 0.95 | 84.4 (84-84) | 95.9 (94-97) | 0.2 (0-1) | 3.9 (2-6) | 27.1 (25-28) |
| cache alone, exact text | 84.4 (84-84) | 95.9 (94-97) | 0.2 (0-1) | 3.9 (2-6) | 27.1 (25-28) |
| refit + cache, cosine 0.85 | 99.1 (98-100) | 94.8 (93-97) | 2.6 (2-3) | 2.6 (1-4) | 30.1 (28-31) |
| refit + cache, cosine 0.90 | 99.1 (98-100) | 96.8 (95-99) | 0.6 (0-1) | 2.6 (1-4) | 27.5 (25-29) |
| refit + cache, cosine 0.95 | 99.1 (98-100) | 96.8 (95-99) | 0.6 (0-1) | 2.6 (1-4) | 27.1 (25-28) |
| refit + cache, exact text | 99.1 (98-100) | 96.8 (95-99) | 0.6 (0-1) | 2.6 (1-4) | 27.1 (25-28) |

### retail v1+v2->v3: values swapped

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 81.9 (82-82) | 67.6 (68-68) | 6.7 (7-7) | 25.7 (26-26) | - |
| refit alone | 97.8 (97-98) | 95.2 (94-96) | 0.0 (0-0) | 4.8 (4-6) | - |
| cache alone, cosine 0.85 | 81.9 (82-82) | 88.6 (86-90) | 4.4 (4-6) | 7.0 (6-9) | 29.8 (27-31) |
| cache alone, cosine 0.90 | 81.9 (82-82) | 82.5 (82-83) | 5.1 (5-6) | 12.4 (12-12) | 17.8 (17-18) |
| cache alone, cosine 0.95 | 81.9 (82-82) | 75.2 (75-75) | 6.7 (7-7) | 18.1 (18-18) | 7.9 (8-9) |
| cache alone, exact text | 81.9 (82-82) | 67.6 (68-68) | 6.7 (7-7) | 25.7 (26-26) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 97.8 (97-98) | 94.3 (93-95) | 1.9 (2-2) | 3.8 (3-5) | 29.8 (27-31) |
| refit + cache, cosine 0.90 | 97.8 (97-98) | 96.2 (95-97) | 0.0 (0-0) | 3.8 (3-5) | 17.8 (17-18) |
| refit + cache, cosine 0.95 | 97.8 (97-98) | 95.2 (94-96) | 0.0 (0-0) | 4.8 (4-6) | 7.9 (8-9) |
| refit + cache, exact text | 97.8 (97-98) | 95.2 (94-96) | 0.0 (0-0) | 4.8 (4-6) | 0.0 (0-0) |

### retail v1+v2->v3: surface variation

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 82.7 (83-83) | 66.8 (67-67) | 6.4 (6-6) | 26.7 (27-27) | - |
| refit alone | 98.1 (98-99) | 93.0 (92-95) | 0.6 (0-1) | 6.4 (4-8) | - |
| cache alone, cosine 0.85 | 82.7 (83-83) | 90.6 (88-92) | 3.4 (3-4) | 6.1 (4-9) | 29.3 (27-31) |
| cache alone, cosine 0.90 | 82.7 (83-83) | 91.3 (89-93) | 2.0 (2-2) | 6.7 (5-9) | 26.0 (24-27) |
| cache alone, cosine 0.95 | 82.7 (83-83) | 89.3 (87-91) | 2.7 (2-3) | 8.0 (6-10) | 23.9 (22-25) |
| cache alone, exact text | 82.7 (83-83) | 79.6 (79-80) | 4.3 (4-4) | 16.1 (15-17) | 13.4 (13-14) |
| refit + cache, cosine 0.85 | 98.1 (98-99) | 91.9 (91-94) | 2.4 (2-3) | 5.8 (4-7) | 29.3 (27-31) |
| refit + cache, cosine 0.90 | 98.1 (98-99) | 93.2 (92-95) | 0.6 (0-1) | 6.2 (4-7) | 26.0 (24-27) |
| refit + cache, cosine 0.95 | 98.1 (98-99) | 93.0 (92-95) | 0.6 (0-1) | 6.4 (4-8) | 23.9 (22-25) |
| refit + cache, exact text | 98.1 (98-99) | 93.0 (92-95) | 0.6 (0-1) | 6.4 (4-8) | 13.4 (13-14) |

### retail v1+v2->v3: verbatim, corrected requests only

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 45.5 (44-47) | 4.8 (2-7) | 13.7 (12-15) | 81.5 (79-86) | - |
| refit alone | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | - |
| cache alone, cosine 0.85 | 45.5 (44-47) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.90 | 45.5 (44-47) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.95 | 45.5 (44-47) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, exact text | 45.5 (44-47) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, exact text | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |

### retail v1+v2->v3: near-miss capture by the cache (share of unseen routing requests)

| stored | hit | captured by a different command | wrong share of hits | exposed to a different command's request |
|---|---|---|---|---|
| corrected @ 0.85 | 4.3 (4-4) | 2.5 (2-3) | 58.3 (50-75) | 2.5 (2-3) |
| corrected @ 0.90 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |
| corrected @ 0.95 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |
| every seen @ 0.85 | 21.7 (22-22) | 3.3 (3-3) | 15.0 (15-15) | 5.4 (5-5) |
| every seen @ 0.90 | 2.2 (2-2) | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0-0) |
| every seen @ 0.95 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |

cache: hit rate 4.3 (4-4), hit precision 41.7 (25-50)
spread of learned rows over labels: [{"learned": 39, "labels_hit": 14, "labels": 21, "top3_share": 0.5128205128205128, "top1_share": 0.23076923076923078, "top3_rows_in_training": [21, 27, 16]}, {"learned": 43, "labels_hit": 15, "labels": 21, "top3_share": 0.4883720930232558, "top1_share": 0.20930232558139536, "top3_rows_in_training": [21, 27, 23]}, {"learned": 43, "labels_hit": 13, "labels": 21, "top3_share": 0.46511627906976744, "top1_share": 0.18604651162790697, "top3_rows_in_training": [21, 27, 23]}]
refit+recalibration (single thread, vectors cached): {'rows': 514, 'fit_s': 0.11697515100240707, 'calibration_s': 0.4181220600003144, 'total_s': 0.5587862129905261}

## large

{'seen': 171, 'test': 172, 'test_routing': 154, 'test_escalation': 18, 'dropped_duplicates': 0, 'test_with_seen_near_dup': 0.11046511627906977, 'test_with_training_row_near_dup': 0.21511627906976744}
  shuffle 0: 104 corrections, first 0.682 -> 0.708
  shuffle 1: 116 corrections, first 0.682 -> 0.721
  shuffle 2: 110 corrections, first 0.682 -> 0.714

### large: learning curve on the test set (mean, min-max over shuffles; % of requests)

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 (n=3) | 68.2 (68-68) | 86.4 (86-86) | 13.6 (14-14) | 26.0 (26-26) | 28.6 (29-29) | 8.4 (8-8) | 63.0 (63-63) | 61.1 (61-61) |
| 25 (n=3) | 69.3 (67-71) | 86.6 (86-88) | 13.0 (13-13) | 25.8 (25-27) | 32.5 (31-35) | 7.8 (6-10) | 59.7 (58-64) | 53.7 (50-61) |
| 50 (n=3) | 70.1 (68-71) | 87.9 (86-89) | 11.5 (11-12) | 22.7 (22-23) | 35.3 (34-37) | 6.5 (5-9) | 58.2 (56-62) | 50.0 (39-67) |
| 100 (n=3) | 72.1 (71-73) | 89.6 (89-90) | 10.2 (10-10) | 19.5 (19-20) | 39.2 (39-40) | 5.2 (5-5) | 55.6 (55-56) | 55.6 (56-56) |
| all (n=3, mean 110) | 71.4 (71-72) | 89.4 (89-90) | 10.6 (10-11) | 19.3 (18-20) | 39.0 (39-39) | 5.2 (4-6) | 55.8 (55-57) | 50.0 (44-56) |

### large: variants at all corrections

| step | first | top3 | wrong@50 | wrong@70 | right outright | wrong confident | handed on | esc recall |
|---|---|---|---|---|---|---|---|---|
| 0 corrections (baseline) | 68.2 (68-68) | 86.4 (86-86) | 13.6 (14-14) | 26.0 (26-26) | 28.6 (29-29) | 8.4 (8-8) | 63.0 (63-63) | 61.1 (61-61) |
| refit x1 (main) | 71.4 (71-72) | 89.4 (89-90) | 10.6 (10-11) | 19.3 (18-20) | 39.0 (39-39) | 5.2 (4-6) | 55.8 (55-57) | 50.0 (44-56) |
| (a) every seen request added | 72.7 (73-73) | 89.6 (90-90) | 8.4 (8-8) | 17.5 (18-18) | 41.6 (42-42) | 5.8 (6-6) | 52.6 (53-53) | 55.6 (56-56) |
| (b) learned rows x3 | 69.9 (69-71) | 90.0 (89-91) | 10.2 (10-10) | 18.8 (18-21) | 39.6 (39-40) | 5.4 (5-6) | 55.0 (55-55) | 53.7 (50-56) |
| (b) learned rows x5 | 69.5 (69-71) | 90.3 (90-91) | 10.0 (9-10) | 19.3 (19-19) | 39.6 (39-41) | 5.4 (5-6) | 55.0 (54-56) | 53.7 (50-56) |
| (d) cache alone (no refit) | 68.2 (68-68) | 86.4 (86-86) | 13.6 (14-14) | 26.0 (26-26) | 29.2 (29-29) | 9.5 (9-10) | 61.3 (61-62) | 66.7 (67-67) |
| (d) cache + refit | 71.4 (71-72) | 89.4 (89-90) | 10.6 (10-11) | 19.3 (18-20) | 39.6 (40-40) | 6.9 (6-8) | 53.5 (53-55) | 66.7 (61-72) |
| (e) 5% wrong labels | 71.2 (71-72) | 88.7 (88-90) | 10.2 (10-10) | 19.7 (19-21) | 39.6 (39-40) | 5.2 (4-7) | 55.2 (53-57) | 50.0 (44-56) |
| (e) 10% wrong labels | 70.3 (69-71) | 88.3 (88-89) | 11.3 (11-12) | 21.2 (21-21) | 38.3 (36-40) | 5.2 (5-6) | 56.5 (55-58) | 48.1 (44-50) |
| (e) 10% wrong + cache | 70.3 (69-71) | 88.3 (88-89) | 11.3 (11-12) | 21.2 (21-21) | 39.0 (36-41) | 7.6 (6-8) | 53.5 (51-55) | 64.8 (61-67) |

recurring sets: {'verbatim': 140, 'values swapped': 25, 'surface variation': 554}

### large: unseen phrasings

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 68.2 (68-68) | 28.6 (29-29) | 8.4 (8-8) | 63.0 (63-63) | - |
| refit alone | 71.4 (71-72) | 39.0 (39-39) | 5.2 (4-6) | 55.8 (55-57) | - |
| cache alone, cosine 0.85 | 68.2 (68-68) | 29.2 (29-29) | 9.5 (9-10) | 61.3 (61-62) | 4.5 (5-5) |
| cache alone, cosine 0.90 | 68.2 (68-68) | 29.2 (29-29) | 8.9 (8-9) | 61.9 (62-62) | 1.1 (1-1) |
| cache alone, cosine 0.95 | 68.2 (68-68) | 28.6 (29-29) | 8.4 (8-8) | 63.0 (63-63) | 0.0 (0-0) |
| cache alone, exact text | 68.2 (68-68) | 28.6 (29-29) | 8.4 (8-8) | 63.0 (63-63) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 71.4 (71-72) | 39.6 (40-40) | 6.9 (6-8) | 53.5 (53-55) | 4.5 (5-5) |
| refit + cache, cosine 0.90 | 71.4 (71-72) | 39.0 (39-39) | 5.2 (4-6) | 55.8 (55-57) | 1.1 (1-1) |
| refit + cache, cosine 0.95 | 71.4 (71-72) | 39.0 (39-39) | 5.2 (4-6) | 55.8 (55-57) | 0.0 (0-0) |
| refit + cache, exact text | 71.4 (71-72) | 39.0 (39-39) | 5.2 (4-6) | 55.8 (55-57) | 0.0 (0-0) |

### large: verbatim

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 67.9 (68-68) | 27.1 (27-27) | 7.1 (7-7) | 65.7 (66-66) | - |
| refit alone | 100.0 (100-100) | 96.2 (94-98) | 0.0 (0-0) | 3.8 (2-6) | - |
| cache alone, cosine 0.85 | 67.9 (68-68) | 94.8 (94-96) | 0.0 (0-0) | 5.2 (4-6) | 69.0 (67-71) |
| cache alone, cosine 0.90 | 67.9 (68-68) | 94.8 (94-96) | 0.0 (0-0) | 5.2 (4-6) | 69.0 (67-71) |
| cache alone, cosine 0.95 | 67.9 (68-68) | 94.8 (94-96) | 0.0 (0-0) | 5.2 (4-6) | 69.0 (67-71) |
| cache alone, exact text | 67.9 (68-68) | 94.8 (94-96) | 0.0 (0-0) | 5.2 (4-6) | 69.0 (67-71) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 96.4 (94-98) | 0.0 (0-0) | 3.6 (2-6) | 69.0 (67-71) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 96.4 (94-98) | 0.0 (0-0) | 3.6 (2-6) | 69.0 (67-71) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 96.4 (94-98) | 0.0 (0-0) | 3.6 (2-6) | 69.0 (67-71) |
| refit + cache, exact text | 100.0 (100-100) | 96.4 (94-98) | 0.0 (0-0) | 3.6 (2-6) | 69.0 (67-71) |

### large: values swapped

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 64.0 (64-64) | 20.0 (20-20) | 4.0 (4-4) | 76.0 (76-76) | - |
| refit alone | 100.0 (100-100) | 80.0 (76-88) | 0.0 (0-0) | 20.0 (12-24) | - |
| cache alone, cosine 0.85 | 64.0 (64-64) | 69.3 (68-72) | 12.0 (12-12) | 18.7 (16-20) | 57.3 (56-60) |
| cache alone, cosine 0.90 | 64.0 (64-64) | 52.0 (52-52) | 4.0 (4-4) | 44.0 (44-44) | 32.0 (32-32) |
| cache alone, cosine 0.95 | 64.0 (64-64) | 32.0 (32-32) | 4.0 (4-4) | 64.0 (64-64) | 12.0 (12-12) |
| cache alone, exact text | 64.0 (64-64) | 20.0 (20-20) | 4.0 (4-4) | 76.0 (76-76) | 0.0 (0-0) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 82.7 (80-88) | 8.0 (8-8) | 9.3 (4-12) | 57.3 (56-60) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 85.3 (80-92) | 0.0 (0-0) | 14.7 (8-20) | 32.0 (32-32) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 80.0 (76-88) | 0.0 (0-0) | 20.0 (12-24) | 12.0 (12-12) |
| refit + cache, exact text | 100.0 (100-100) | 80.0 (76-88) | 0.0 (0-0) | 20.0 (12-24) | 0.0 (0-0) |

### large: surface variation

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 65.5 (66-66) | 24.9 (25-25) | 6.3 (6-6) | 68.8 (69-69) | - |
| refit alone | 98.0 (98-98) | 85.6 (84-88) | 0.2 (0-0) | 14.3 (12-16) | - |
| cache alone, cosine 0.85 | 65.5 (66-66) | 88.7 (87-90) | 0.4 (0-0) | 11.0 (9-12) | 65.9 (64-68) |
| cache alone, cosine 0.90 | 65.5 (66-66) | 86.2 (85-88) | 0.7 (1-1) | 13.1 (11-15) | 63.2 (61-66) |
| cache alone, cosine 0.95 | 65.5 (66-66) | 80.0 (79-81) | 1.4 (1-1) | 18.5 (18-20) | 56.5 (55-58) |
| cache alone, exact text | 65.5 (66-66) | 58.8 (58-59) | 3.1 (3-3) | 38.1 (38-39) | 34.9 (34-36) |
| refit + cache, cosine 0.85 | 98.0 (98-98) | 91.8 (91-92) | 0.2 (0-0) | 8.1 (7-9) | 65.9 (64-68) |
| refit + cache, cosine 0.90 | 98.0 (98-98) | 90.6 (89-91) | 0.2 (0-0) | 9.3 (8-11) | 63.2 (61-66) |
| refit + cache, cosine 0.95 | 98.0 (98-98) | 88.3 (87-90) | 0.2 (0-0) | 11.5 (10-13) | 56.5 (55-58) |
| refit + cache, exact text | 98.0 (98-98) | 86.5 (85-89) | 0.2 (0-0) | 13.3 (11-15) | 34.9 (34-36) |

### large: verbatim, corrected requests only

| mechanism | first | right outright | wrong confident | handed on | cache answers |
|---|---|---|---|---|---|
| no learning | 54.1 (52-57) | 2.0 (1-4) | 10.4 (10-11) | 87.6 (86-89) | - |
| refit alone | 100.0 (100-100) | 99.6 (99-100) | 0.0 (0-0) | 0.4 (0-1) | - |
| cache alone, cosine 0.85 | 54.1 (52-57) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.90 | 54.1 (52-57) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, cosine 0.95 | 54.1 (52-57) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| cache alone, exact text | 54.1 (52-57) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.85 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.90 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, cosine 0.95 | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |
| refit + cache, exact text | 100.0 (100-100) | 100.0 (100-100) | 0.0 (0-0) | 0.0 (0-0) | 100.0 (100-100) |

### large: near-miss capture by the cache (share of unseen routing requests)

| stored | hit | captured by a different command | wrong share of hits | exposed to a different command's request |
|---|---|---|---|---|
| corrected @ 0.85 | 4.5 (5-5) | 2.4 (2-3) | 52.4 (43-57) | 2.4 (2-3) |
| corrected @ 0.90 | 1.1 (1-1) | 0.4 (0-1) | 33.3 (0-50) | 0.4 (0-1) |
| corrected @ 0.95 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |
| every seen @ 0.85 | 5.8 (6-6) | 2.6 (3-3) | 44.4 (44-44) | 2.6 (3-3) |
| every seen @ 0.90 | 1.3 (1-1) | 0.6 (1-1) | 50.0 (50-50) | 0.6 (1-1) |
| every seen @ 0.95 | 0.0 (0-0) | 0.0 (0-0) | - | 0.0 (0-0) |

cache: hit rate 4.5 (5-5), hit precision 47.6 (43-57)
spread of learned rows over labels: [{"learned": 104, "labels_hit": 52, "labels": 87, "top3_share": 0.19230769230769232, "top1_share": 0.0673076923076923, "top3_rows_in_training": [33, 32, 33]}, {"learned": 116, "labels_hit": 53, "labels": 87, "top3_share": 0.1896551724137931, "top1_share": 0.06896551724137931, "top3_rows_in_training": [32, 33, 33]}, {"learned": 110, "labels_hit": 51, "labels": 87, "top3_share": 0.19090909090909092, "top1_share": 0.08181818181818182, "top3_rows_in_training": [33, 32, 33]}]
refit+recalibration (single thread, vectors cached): {'rows': 2472, 'fit_s': 1.5841285879869247, 'calibration_s': 6.378895520014339, 'total_s': 7.974603529000888}
