# Pre-registration, fifth round: fewer wrong confident answers from the embedding head

Follows `docs/single_intent_model_preregistration_v4.md`. Written 2026-11-17 after the development runs
below and before any score on a confirmation set. Script: `scripts/experiments/embedding_commit_rule.py` (its calibration and decision are the head's rule as it was until bd fix-o5wi.9, kept in `scripts/experiments/head_rule_as_experimented.py`)
(no model call anywhere; the remote decision link is not called, this is the local head's operating
point). Tracked as bead fix-o5wi.4.

## The question

The head names one wrong command on 8.7% of the third example benchmark (2.7 points of it are requests
passed up to a parent context that should not be), against 3.4% for the old per-context classifiers,
which commit on only 22% of requests against the head's 70%. Cross-fitted training rows are easy (about
97% right) and real requests are not (about 83%), so a 1% target on the rows lets 6% to 9% through.
Can a rule computed from the same embeddings cut the wrong lone answers to a level near the old
classifiers without giving up most of the head's commitments? A request the head no longer answers alone
goes to the next link: the remote decision model when its key is set, else the user.

## What is fixed

* Encoder `BAAI/bge-small-en-v1.5` frozen, one logistic-regression head per workflow (C = 100) over all
  labels, read through each context's mask, the decision rule of `embedding_head.decide`, threshold floor 0.5.
  Each workflow's head is calibrated on its own rows (as the product does); "examples" are the library and
  retail workflows scored together, "large" is one large private workflow (only numbers are reported).
* Rows are rebuilt from the utterance caches (`workflow_model.workflow_rows`), no model call. BASE through
  this script equals the product's calibration (threshold and margin cross-checked at every run).
* Development sets (already looked at): the library and retail benchmarks v1, v2, v3 (concatenated:
  735 routing and 120 escalation cases), and the development half of the large
  workflow's benchmark. Confirmation sets, unscored until the Results section: the library and retail
  `_v4` benchmarks (310 cases), and the confirmation half of the large benchmark. The large benchmark's 664
  cases were split once, before anything was scored, by the low bit of the first byte of SHA-256 of
  `context + newline + utterance` (`split` step): 343 development cases (294 routing, 49 escalation) and 321
  confirmation cases (282 routing, 39 escalation). No case was dropped when the rows were rebuilt.

## The variants

* **BASE**: the head as shipped (targets 0.01 wrong commits, 0.005 false escalation).
* **V1** stricter targets, to be used when the remote link exists: wrong-commit target in {0.01, 0.005,
  0.0025, 0.001}, false-escalation target in {0.005, 0.0025, 0.001}.
* **V2** cross-fit folds grouped by seed instead of by text. A generated row's seed utterance is not
  recorded anywhere (the generator is given all of a command's seeds at once), so the grouping falls back to
  the producing persona: a generated row is folded with every row of its persona (and rows sharing a text
  stay together); a seed, a bare value or a vague reply stands alone by its text. Cross-fitted rows are
  then 94.5% right on the examples (96.7% grouped by text) and 91.9% on the large workflow (94.1%): closer
  to real requests, still easier.
* **V3** agreement gate: commit to one command only when the head's top label inside the mask and a
  nearest-neighbour vote (cosine, k in {5, 10}, over the distinct training rows whose label is in the mask;
  plurality, ties by summed similarity) name the same label; otherwise the top three. The threshold is
  calibrated with the gate in place (wrong commits counted only among agreeing rows; the fold's own
  training rows are the neighbours).
* **V4** confirmed escalation: a lone "not here" (margin test, or the head's best label being the workflow
  `wildcard`) only when the same kind of vote over ALL labels lands outside the mask; otherwise the top three.
  The margin is calibrated with the confirmation in place.

## Development results (examples pooled; large development half; thresholds each variant computes itself)

BASE thresholds and margins: library 0.529 / 0.443, retail 0.500 / 0, large 0.591 / 0.622.
Counts: examples 735 routing + 120 escalation cases, large 294 + 49.

| Variant | Group | Strict top-1 | In-list | One label | Wrong lone | False esc. | Esc. recall | Lone precision |
|---|---|---|---|---|---|---|---|---|
| BASE | examples | 69.5% | 90.5% | 75.4% | 5.9% | 1.9% | 115/120 | 92.2% |
| BASE | large | 27.9% | 79.9% | 35.7% | 7.8% | 5.8% | 30/49 | 78.1% |
| V1 best targets 0.005/0.005 | examples | 66.0% | 90.9% | 71.0% | 5.0% | 1.9% | 115/120 | 92.9% |
| V1 best targets 0.005/0.005 | large | 20.4% | 80.3% | 27.9% | 7.5% | 5.8% | 30/49 | 73.2% |
| V2 persona folds | examples | 66.7% | 91.6% | 71.4% | 4.8% | 1.4% | 105/120 | 93.3% |
| V2 persona folds | large | 30.3% | 79.9% | 38.1% | 7.8% | 5.1% | 24/49 | 79.5% |
| V3 gate k=5 | examples | 63.0% | 91.3% | 68.0% | 5.0% | 1.9% | 115/120 | 92.6% |
| V3 gate k=5 | large | 25.5% | 80.3% | 33.0% | 7.5% | 5.8% | 30/49 | 77.3% |
| V3 gate k=10 | examples | 61.8% | 91.2% | 66.9% | 5.2% | 1.9% | 115/120 | 92.3% |
| V3 gate k=10 | large | 26.5% | 80.3% | 34.0% | 7.5% | 5.8% | 30/49 | 78.0% |
| V4 confirm k=5 | examples | 69.5% | 91.3% | 74.6% | 5.0% | 1.1% | 118/120 | 93.2% |
| V4 confirm k=5 | large | 27.9% | 80.6% | 34.0% | 6.1% | 4.1% | 30/49 | 82.0% |
| V4 confirm k=10 | examples | 69.5% | 91.6% | 74.1% | 4.6% | 0.7% | 114/120 | 93.8% |
| V4 confirm k=10 | large | 27.9% | 81.3% | 33.3% | 5.4% | 3.4% | 30/49 | 83.7% |
| V2+V3 k=10 | examples | 61.9% | 91.7% | 66.7% | 4.8% | 1.4% | 105/120 | 92.9% |
| V2+V3 k=10 | large | 28.9% | 80.6% | 36.1% | 7.1% | 5.1% | 24/49 | 80.2% |
| V2+V4 k=10 | examples | 66.7% | 92.5% | 70.3% | 3.7% | 0.3% | 102/120 | 94.8% |
| V2+V4 k=10 | large | 30.3% | 80.3% | 36.7% | 6.5% | 3.7% | 30/49 | 82.4% |
| V3+V4 k=10 | examples | 61.8% | 92.2% | 65.7% | 3.9% | 0.7% | 114/120 | 94.0% |
| V3+V4 k=10 | large | 26.5% | 81.6% | 31.6% | 5.1% | 3.4% | 30/49 | 83.9% |
| V2+V3+V4 k=10 | examples | 61.9% | 92.7% | 65.6% | 3.7% | 0.3% | 102/120 | 94.4% |
| V2+V3+V4 k=10 | large | 28.9% | 81.0% | 34.7% | 5.8% | 3.7% | 30/49 | 83.3% |
| **Chosen: gate k=5 + confirm k=10** | examples | 63.0% | 92.4% | 66.8% | 3.8% | 0.7% | 114/120 | 94.3% |
| **Chosen: gate k=5 + confirm k=10** | large | 25.5% | 81.6% | 30.6% | 5.1% | 3.4% | 30/49 | 83.3% |

The other 200 configurations (every combination of the 12 target pairs, both fold groupings, gate and
confirmation each off, k=5 or k=10) are in `develop.json` in the scratch folder; none met every limit below.

Trade-off: wrong lone answers (share of all routing cases) when the threshold alone is moved until about
40, 50, 60, 70, 80% of routing cases are answered with one label (the committed share reached is in
brackets; n/a = the rule cannot commit that much). Margin as calibrated, development sets.

| Rule | Group | 40% | 50% | 60% | 70% | 80% |
|---|---|---|---|---|---|---|
| BASE | examples | 2.9% (40%) | 3.1% (50%) | 4.1% (60%) | 5.0% (70%) | 6.8% (80%) |
| BASE | large | 8.5% (40%) | 10.9% (51%) | 14.6% (61%) | 20.7% (70%) | n/a |
| Chosen | examples | 1.6% (40%) | 2.3% (50%) | 2.9% (60%) | 4.4% (70%) | n/a |
| Chosen | large | 6.8% (40%) | 9.5% (51%) | n/a | n/a | n/a |

What the development runs say, plainly: the dominant wrong lone answers on the large workflow are requests
passed up to a parent context that should not be (5.8 of BASE's 7.8 points). Only the confirmed
escalation reduces them (to 3.4); the stricter targets, the harder folds and the gate barely touch them.
Stricter targets (V1) buy 1 point of wrong lone answers on the examples for 3.5 points of strict top-1 and
buy nothing on the large workflow. The agreement gate costs 6 to 8 points of strict top-1 on the examples
for under 1 point of wrong lone answers. Persona folds cost 10 escalation cases of 120 on the examples and
6 of 49 on the large workflow.

## The recipe registered for confirmation

One rule for "remote link present" and for "no remote link" (the stricter-target grid, the only part meant
for the remote case, was searched and not chosen).

**Recipe: targets 0.01/0.005, gate k=5, confirm k=10.**

* Targets: wrong-commit 0.01, false-escalation 0.005 (BASE's). Folds grouped by row text. Threshold floor 0.5.
* Agreement gate, k = 5: a command is answered alone only when the head's top label inside the mask and the
  plurality of the 5 nearest distinct training rows (cosine, only rows whose label is in the mask) are the
  same label. Otherwise the top three.
* Confirmed escalation, k = 10: "not here" alone only when the plurality of the 10 nearest distinct training
  rows over all labels (the bare-value and `wildcard` rows included) has a label outside the mask. Otherwise
  the top three.
* Threshold and margin are recalibrated with both parts in place, from the cross-fitted rows (the neighbours
  of a row are the training side of its fold).

Selection rule, written into the script before it ran: among the 216 configurations, those that meet every
limit (wrong lone answers at most 3.5% on the examples and at most 45% of BASE's on the large workflow;
strict top-1 at most 8 points below BASE; in-list not below BASE; escalation recall at most 2 points
(examples) or 8% relatively (large) below BASE; false escalation at most 1.2%; all on the development sets),
by highest mean strict top-1 and then fewer parts; if none meets every limit, the smallest total excess over
the limits. None met every limit at once (the strict-target configurations reach 2.0% wrong lone answers and 1.0% false
escalation on the large workflow, but only with confirmation and a false-escalation target of 0.001, which cuts the large
workflow's escalation recall from 30 to 10 of 49), so the smallest total excess chose the recipe above (total excess 4.09 points, mostly the
large workflow's wrong lone answers at 5.1% against a limit of 3.5% and its false escalation at 3.4% against 1.2%).

## Predictions

On the confirmation sets, against BASE on the same cases:

1. Wrong lone answers: examples about 4% (between 3% and 5%), so the 4.0% limit is a coin flip; large about
   two thirds of BASE's, so **condition 1 fails on the large workflow**.
2. Strict top-1: 6 to 7 points below BASE on the examples, 2 to 3 on the large workflow. Passes.
3. In-list: higher than BASE by about 1 to 2 points. Passes.
4. Escalation recall: within 2 cases of BASE on the examples (BASE catches nearly all of them); on the
   large workflow equal to BASE. Passes.
5. False escalation: about 0.5% to 1% on the examples (passes); about 3% on the large workflow, so
   **condition 5 fails on the large workflow**.

## Pass conditions (examples = the `_v4` benchmarks pooled; large = confirmation half; each judged separately against BASE)

1. Wrong lone answers at most 4.0% of routing cases on the examples, and at most half of BASE's on the
   large workflow.
2. Strict top-1 not more than 10 points below BASE (with a remote link the lost commitments go to the
   remote model; without one, to the user).
3. Right command in the answer not lower than BASE.
4. Escalation recall not more than 2 cases below BASE on the examples, and not lower on the large workflow
   by more than 10% (relative).
5. False escalation at most 1.5%.

The confirmation is run once for BASE and the recipe (`confirm` step; it refuses to run unless this document
names the recipe). Results follow.

## Results

Run once, 2026-11-17, `embedding_commit_rule.py confirm`. Examples: the library and retail `_v4`
benchmarks pooled (264 routing, 46 escalation cases). Large: the confirmation half (282 routing, 39
escalation cases). The one recipe was used for both "remote link present" and "no remote link" (identical
rows).

| Arm | Group | Strict top-1 | In-list | One label | Wrong lone | False esc. | Esc. recall | Lone precision |
|---|---|---|---|---|---|---|---|---|
| BASE | examples | 60.2% | 86.7% | 70.5% | 10.2% | 3.8% | 43/46 | 85.5% |
| Recipe | examples | 50.4% | 90.9% | 56.1% | 5.7% | 2.3% | 42/46 | 89.9% |
| BASE | large | 22.7% | 80.5% | 31.2% | 8.5% | 5.7% | 27/39 | 72.7% |
| Recipe | large | 22.3% | 81.9% | 29.4% | 7.1% | 4.6% | 26/39 | 75.9% |

| Condition | Examples | Large workflow |
|---|---|---|
| 1 wrong lone answers (at most 4.0% / at most half of BASE's) | **fail** (5.7%) | **fail** (7.1% against a limit of 4.25%) |
| 2 strict top-1 not more than 10 points below BASE | pass (9.8 points below, a hair inside the limit) | pass (0.4 below) |
| 3 in-list not lower than BASE | pass (+4.2 points) | pass (+1.4) |
| 4 escalation recall (examples: at most 2 cases below; large: at most 10% below) | pass (1 case below) | pass (1 case, 3.7%, below) |
| 5 false escalation at most 1.5% | **fail** (2.3%) | **fail** (4.6%) |

**The recipe fails conditions 1 and 5 on both groups; it passes 2, 3 and 4.** The registered prediction
was right about the large workflow (conditions 1 and 5 fail) and wrong about the examples: wrong lone
answers were predicted at about 4% and came out at 5.7% (BASE itself is worse on this benchmark, 10.2%,
than on the development sets, 5.9%), false escalation was predicted at 0.5% to 1% and came out at 2.3%, and
strict top-1 fell 9.8 points where 6 to 7 were predicted.

What it did do: it removed 4.5 points of wrong lone answers on the examples (10.2% to 5.7%, 44% fewer) and 1.4
on the large workflow (17% fewer), raised lone-answer precision (85.5% to 89.9%; 72.7% to 75.9%) and the
right command in the answer by 4.2 and 1.4 points, and cost 10 points of strict top-1 on the examples and
none on the large workflow, at one escalation case on each. It does not bring the head near the old
classifiers' 3.4%. The large workflow's wrong lone answers are mostly requests passed up when they should not
be (4.6 of 7.1 points); the confirmation cut their share from 5.7% to 4.6% of routing cases only.

## Confusable pairs (idea 5, data only)

From BASE's five-fold cross-fitted probabilities (folds grouped by text, each distinct row once), a pair is
two commands one of which the head gives rows of the other to; the share is of the command's own rows.
Files: `confused_pairs_<workflow>.json` in the scratch folder `commit_rule/pairs/` (`pairs` step).
Library: 27 confused pairs, 2 with more than 10% of a command's rows given to the other. Retail: only 4
confused pairs exist, none above 10%. Large workflow: 105 confused pairs, 9 above 10%.

Library, ten most confused (A to B rows / A's rows; both directions):

| Pair | A to B | B to A |
|---|---|---|
| Book/list_loans, Member/list_member_loans | 2/20 (10.0%) | 2/25 (8.0%) |
| Library/open_member, Member/show_member | 3/19 (15.8%) | 0/21 |
| Book/list_loans, Library/book_loans | 0/20 | 2/25 (8.0%) |
| Book/loan_report, Book/shelf_report | 1/20 (5.0%) | 1/30 (3.3%) |
| Book/shelf_report, Book/show_book | 1/30 (3.3%) | 1/28 (3.6%) |
| Book/show_book, Library/book_by_title | 1/28 (3.6%) | 1/19 (5.3%) |
| Book/show_book, Member/show_member | 1/28 (3.6%) | 1/21 (4.8%) |
| Library/book_by_title, Library/find_book | 2/19 (10.5%) | 0/30 |
| Library/book_by_title, Library/find_record | 0/19 | 2/20 (10.0%) |
| Library/find_book, Library/find_record | 0/30 | 2/20 (10.0%) |

Retail (all four): go_up and what_is_current_context (1/23, 0/19); exchange_delivered_order_items and
modify_pending_order_items (1/21, 0/20); find_user_id_by_email and find_user_id_by_name_zip (0/16, 1/21);
modify_pending_order_address and modify_user_address (0/16, 1/31).

## Exploratory follow-up (2026-11-17)

**EXPLORATORY. Not pre-registered.** Run after the confirmation sets above were scored once, on those same
confirmation cases (examples: library and retail `_v4`, 264 routing + 46 escalation; large: the confirmation
half, 282 + 39), so nothing here may be read as a confirmation and no choice should be tuned on it. Script:
`scripts/experiments/embedding_chain_followup.py` (steps `heads`, `old`, `chain`, `report`); same metrics as
the Results section. "V4" is confirmed escalation alone (k = 10, targets 0.01 / 0.005, no agreement gate), which
the registered recipe bundled with the gate. BASE through this script reproduces the Results rows exactly.

### Group 1: examples (library and retail pooled, 310 cases)

| Row | Strict top-1 | In-list | One label | Wrong lone | False esc. | Esc. recall | Lone precision |
|---|---|---|---|---|---|---|---|
| old per-context pair, majority of seeds 42/43/44 | 16.3% | 77.7% | 20.1% | 2.7% | 0.0% | 3/46 | 81.1% |
| (old pair, seeds 42 / 43 / 44 separately: wrong lone) | 4.2% / 7.2% / 5.3% | | | | | | |
| head BASE | 60.2% | 86.7% | 70.5% | 10.2% | 3.8% | 43/46 | 85.5% |
| head V4 | 60.2% | 88.3% | 68.9% | 8.7% | 2.3% | 42/46 | 87.4% |
| chain with BASE | 90.9% | 90.9% | 100.0% | 9.1% | 1.9% | 45/46 | 90.9% |
| chain with V4 | 90.9% | 90.9% | 100.0% | 9.1% | 1.9% | 45/46 | 90.9% |

### Group 2: the large workflow (321 cases)

| Row | Strict top-1 | In-list | One label | Wrong lone | False esc. | Esc. recall | Lone precision |
|---|---|---|---|---|---|---|---|
| old per-context pair (previous trained version, one build) | 26.2% | 64.5% | 44.0% | 17.7% | 13.8% | 21/39 | 59.7% |
| head BASE | 22.7% | 80.5% | 31.2% | 8.5% | 5.7% | 27/39 | 72.7% |
| head V4 | 22.7% | 81.6% | 30.1% | 7.4% | 4.6% | 26/39 | 75.3% |
| chain with BASE | 85.8% | 87.9% | 97.9% | 12.1% | 7.4% | 39/39 | 87.7% |
| chain with V4 | 85.8% | 87.9% | 97.9% | 12.1% | 7.4% | 39/39 | 87.7% |

Chain facts (the remote decision model is asked for every local answer that is a list or a lone "not here"):

| | Asked | Final wrong lone answers | Head confidently wrong | Remote model wrong | Mean added latency per asked case | Ends at a question |
|---|---|---|---|---|---|---|
| examples | 133 of 310 (42.9%) | 24 (9.1% of routing) | 17 | 7 | 0.20 s | 2 of 310 (0.6%) |
| large | 249 of 321 (77.6%) | 34 (12.1% of routing) | 8 | 26 | 0.26 s | 17 of 321 (5.3%) |

What it shows (exploratory): with the remote model behind it the head's strict top-1 goes from 60% to 91%
(examples) and from 23% to 86% (large), and the large workflow's escalation recall to 39/39, but the wrong lone
answers do not fall: 9.1% and 12.1% against the head alone's 10.2% and 8.5%, because the remote model
turns the head's unsure answers into lone answers that are sometimes wrong (7 of the 24 on the examples, 26 of
the 34 on the large workflow, where it also wrongly passes up 7.4% of routing requests). The old per-context
pair still has the fewest wrong lone answers on the examples (2.7%), by committing on 20% of requests and
escalating almost nothing, and on the large workflow the previous version is worse than the head on every
error rate. V4 alone is a small gain for the head (wrong lone 10.2% to 8.7% and 8.5% to 7.4%, false
escalation to 2.3% and 4.6%, one escalation case lost on each group) and is exactly nothing in the chain: V4
only turns a lone "not here" into a list, and both go to the remote model, so the chain rows are identical
case by case.

Not comparable, or comparable only with care:
* The old pair on the large workflow is a previous trained version (an earlier training, possibly other rows
  and data; one build, no seed majority), not a rebuild on today's rows; its row says little about the
  recipe. On the examples the pairs were rebuilt at three seeds on the current rows.
* The chain rows use the live walk's scope path (the context, its enclosing records, then the global scope);
  `score_matcher_engine.py`'s library paths omit the global scope. The remote model's answers were taken once
  per case and recorded; the remote model is not deterministic, so a rerun with a fresh recording can differ.
  Remote answers for the examples are recorded in a file given with `--recording`; no remote call failed.
* "Mean added latency" is the recorded remote call alone (the model cache was off), not the head's time. Chain
  "strict top-1" equals "in-list" on the examples because the remote model's answers there are one label or
  "not here", with 2 questions.
* One run on one confirmation half each: small counts (46 and 39 escalation cases), no intervals computed.

### Second follow-up: does a tuned commit threshold for the remote link help? (exploratory, 2026-11-17)

**EXPLORATORY.** The confirmation cases were scored before, so every confirmation number here is exploratory;
the thresholds are chosen on the development sets only. Script: `scripts/experiments/embedding_chain_followup.py`
(steps `heads --set dev`, `chain --grid`, `grid`, `errors`, `tuned`). `MatcherEngine.decide` reads one constant,
`MATCHER_MATCH_THRESHOLD` (0.5), twice: a scope's best command at or above it is a match, and "none" or a
command of another scope at or above it declines. The sweep changes exactly those two reads (a subclass of
`MatcherEngine` whose `decide` is the product's with the two values apart; it is asserted equal to the product's
class at 0.5 / 0.5 on every development case, and the 0.2 "unsure" threshold and the three-candidate cap are
untouched). The head is BASE; V4 would change nothing in the chain.

#### Types of the remote model's wrong answers at 0.5 / 0.5 (confirmation recordings, no new calls)

| Type | Examples (133 decided by the remote model) | Large (249) |
|---|---|---|
| wrong command of the same scope | 2 | 3 |
| a command of another scope on the path (wrong pass-up) | 2 | 13 |
| "none" when a command was right | 3 | 4 |
| declined with no command above 0.2 (wrong pass-up) | 0 | 4 |
| a command when it should have passed up | 0 | 0 |
| a question with one wrong candidate (a lone answer by the metric) | 0 | 2 |
| wrong lone answers of the remote model | 7 | 26 |

Confidence of the remote model's lone answers (the command's score for a match, the larger of "none" and the
best command of another scope for a decline; the four declines with no confident option sit in the first bin):

| Confidence | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|
| examples, wrong (7) | 1 | 1 | 3 | 1 | 1 |
| examples, right (124) | 1 | 5 | 4 | 9 | 105 |
| large, wrong (24; the 2 single-candidate questions are not binned) | 14 | 5 | 1 | 2 | 2 |
| large, right (208) | 16 | 23 | 20 | 34 | 115 |

On the large workflow 13 of the 24 wrong lone answers are passes-up to a command of another scope, and 14 of
the 24 sit in the 0.5-0.6 bin, where only 16 right answers do; on the examples the 7 errors are spread out.

#### Development recordings and sweep

Chain with head BASE on the development sets (examples v1+v2+v3 pooled, 735 routing + 120 escalation;
large development half, 294 + 49), new recordings: 315 live remote calls for the examples (247 library, 68
retail) and 255 for the large workflow, none failed. Wrong lone = share of routing cases (head + remote
counts in brackets); "questions" = share of all requests that end at a question to the user. The sweep is a
5 x 5 grid (match threshold x decline threshold, each in 0.5 to 0.9); selected cells, full grid in the script
output (scratch `grid.txt`):

| (match, decline) | Examples: top-1 / wrong lone / false esc / questions | Large: top-1 / wrong lone / false esc / questions |
|---|---|---|
| 0.5, 0.5 (now) | 92.4% / 7.5% (29+26) / 3.3% / 0.1% | 83.7% / 15.0% (6+38) / 9.9% / 5.5% |
| 0.6, 0.5 | 92.2% / 7.5% (29+26) / 3.3% / 0.5% | 83.7% / 14.6% (6+37) / 9.9% / 9.3% |
| 0.9, 0.5 | 92.1% / 7.3% (29+25) / 3.3% / 2.9% | 83.0% / 14.6% (6+37) / 9.9% / 27.4% |
| 0.5, 0.7 | 93.2% / 6.7% (29+20) / 2.0% / 1.2% | 86.4% / 12.2% (6+30) / 7.1% / 7.9% |
| 0.5, 0.8 | 93.2% / 6.7% (29+20) / 2.0% / 1.2% | 87.8% / 10.9% (6+26) / 5.8% / 9.0% |
| 0.6, 0.8 | 93.1% / 6.7% (29+20) / 2.0% / 1.5% | 87.8% / 10.5% (6+25) / 5.8% / 12.8% |
| 0.7, 0.8 | 92.9% / 6.5% (29+19) / 2.0% / 1.9% | 87.1% / 10.5% (6+25) / 5.8% / 18.4% |

Escalation recall stays 120/120 and 49/49 in every cell. A decline threshold of 0.9 gives exactly the 0.8
numbers. Raising the match threshold only moves answers to questions (it removes 1 to 2 remote errors and
costs 4 to 22 points of questions on the large workflow); raising the decline threshold is what removes errors.

#### The selection rule (fixed here, before any confirmation number was computed for it)

Choose the (match, decline) pair with the lowest mean, over the two groups, of wrong lone answers (share of
routing cases), among the pairs whose questions to the user are at most 15% of all requests in BOTH groups;
ties go to the higher mean strict top-1, and a complete tie to the lower pair. On the development data it
chooses **match 0.6, decline 0.8** (mean wrong lone 8.6%; the pairs (0.7, 0.8) and above have more than 15%
questions on the large workflow; (0.6, 0.9) ties exactly and is not chosen).

#### Applied to the confirmation recordings (exploratory; no new calls)

| Group | Thresholds (match / decline) | Strict top-1 | Wrong lone (head + remote) | False esc. | Esc. recall | Ends at a question | Remote asked |
|---|---|---|---|---|---|---|---|
| examples (310) | 0.5 / 0.5 (now) | 90.9% | 9.1% (17+7) | 1.9% | 45/46 | 0.6% (2) | 133 |
| examples (310) | 0.6 / 0.8 (chosen) | 91.7% | 8.3% (17+5) | 1.1% | 45/46 | 1.6% (5) | 133 |
| large (321) | 0.5 / 0.5 (now) | 85.8% | 12.1% (8+26) | 7.4% | 39/39 | 5.3% (17) | 249 |
| large (321) | 0.6 / 0.8 (chosen) | 87.6% | 9.2% (8+18) | 5.0% | 39/39 | 11.8% (38) | 249 |

What it shows (exploratory): a higher decline threshold helps, but not enough to fix the remote link. On the
large workflow the chosen pair removes 8 of the remote model's 26 wrong lone answers (12.1% to 9.2%), cuts false
escalation from 7.4% to 5.0% and raises top-1 by 1.8 points, paid for with 21 more requests ending at a question
(5.3% to 11.8%); on the examples it removes 2 of 7 (9.1% to 8.3%) at 3 more questions. The remote wrong answers
are mostly wrong passes-up and "none" at low confidence, but a good part is confident (14 of the 24 large errors
are under 0.6, the rest are not), so no threshold separates them; the large workflow's 9.2% remains above the
head alone's 8.5%, and the head's own 8 confident errors are untouched by any remote threshold. One tuned pair,
one development / confirmation split, small counts; not a registered result.
