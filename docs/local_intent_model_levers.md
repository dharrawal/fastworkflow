# Local intent model: what the cheap levers buy (no remote model)

Tracked as bead fix-o5wi.5. Script: `scripts/experiments/local_head_levers.py` (no model call of any kind; rows come from the
utterance caches, held-out cases are the development benchmarks only). Follows `docs/single_intent_model_preregistration_v4.md`
and `_v5.md`.

## What was measured, and how far to trust it

* The local model is the `embedding` head: a frozen sentence encoder (`BAAI/bge-small-en-v1.5`, CLS pooling) and one
  logistic-regression head (C = 100) per workflow over every label, read through each context's mask.
* **Examples** = the library and retail workflows scored together on the first three benchmarks (735 routing and 120
  escalation requests). **Large** = one large private workflow (144 context-command pairs, 87 labels, about 30 training rows per
  label), development half only (294 routing and 49 escalation requests). Nothing here was scored on the fourth example
  benchmarks or on the large workflow's confirmation half. The large workflow is described by numbers and kinds of confusion only.
* These sets have been used before. Differences under about **2 points on the examples and 4 points on the large workflow
  are noise** (one large-workflow request is 0.34 points; one example request 0.14). A gain counts here only if it shows on BOTH groups.
* Everything is **in-sample on development data** wherever a threshold or margin is chosen on the same cases it is scored on
  (marked below). Real rates will be somewhat worse. Where thresholds come from the training rows' cross-fit instead, they are
  out of sample, and the finding is that they do not hit their target (Part 3).
* Rebuilding the product's calibration from my own cross-fit gives threshold/margin 0.528/0.516 (library), 0.500/0.000 (retail),
  0.586/0.620 (large) against the product's 0.529/0.443, 0.500/0.000, 0.591/0.622: the margin of the small workflow differs
  (tie-sensitive), the rest agrees. The "shipped operating point" rows use the product's own values.
* Ranking rule (owner's request): variants are ordered by **wrong confident answers at a fixed share of requests answered right
  outright** (50% and 60% on the examples, 20% and 30% on the large workflow; the lowest wrong share reachable by any
  threshold and pass-up margin, in-sample), not by best-label-first accuracy alone. A wrong pass-up counts as a wrong confident answer.

Definitions used in the tables (all threshold-free unless stated):

* *Best label first*: the best label inside the context's mask is the expected one. *Right in top 3*: it is among the best three inside the mask (ties count against the variant).
* *Escalations: best label is outside*: for escalation requests the best label overall is not inside the mask; *expected ancestor*: it is exactly the expected parent command.
* *Wrong confident when 50% / 70% are answered*: requests ranked by the head's own confidence (the score of its best label overall); the answer of the top 50% (70%) is that label (so a request passed up is a wrong answer); share of all routing requests.
* Operating points use the product's decision rule (`embedding_head.decide`): bare value or wildcard best label answers alone; else a pass-up when the best outside-the-mask score exceeds the best inside by at least the margin; else one command when its probability is at least the threshold; else the top three. "Not answered" requests go to the next link (remote model, or the user or agent is asked).

## Part 1: where the shipped head goes wrong (threshold-free)

| | Examples | Large |
|---|---|---|
| Best label first | 85.3% | 67.7% |
| Right one in top 3 / 5 / 10 | 95.2 / 97.7 / 99.3% | 84.4 / 89.5 / 94.6% |
| Escalation requests whose best label is outside the mask | 119 of 120 | 49 of 49 |
| ... and is exactly the expected ancestor command | 117 of 120 | 37 of 49 |
| First-ranked misses | 108 | 95 |
| ... best label overall is outside the mask | 23 | 54 |

Misses by kind (each miss in one kind, first match in this order):

| Kind | Examples (108) | Large (95) |
|---|---|---|
| (d) navigation or context-entry command involved (name opens, goes, resets, or is a built-in "what can I do") | 23 (22 with a plain command, 1 between two) | **45 (35 with a plain command, 10 between two)** |
| (a) sibling command of the same context | **61** | 33 |
| (c) planning command against a plain command (17: 10 against one it is built from, 7 against another) | 17 | 0 |
| (b) same capability under another context's label (same command name, other owner) | 0 | 0 |
| (e) other | 7 | 17 |

* **Dominant kinds**: on the large workflow, confusion with context-entry / navigation commands (47% of misses) and between sibling commands of one context (35%). On the examples, sibling commands (56%), then navigation (21%) and planning-against-plain (16%). Same-capability duplicates under other labels cause no first-ranked miss at all (the large workflow has 4 command names exposed under more than one label; the examples none), so merging them (lever 4 iii) has nothing to fix.
* The cross-fit's ten most confused pairs explain almost none of the misses on real requests: 21 of 108 (examples), 4 of 95 (large). The confusions requests expose are not the ones the training rows expose.
* By training rows of the expected label (best label first): examples 16-25 rows 85%, 26-30 86%, 31 or more 81%; large 16-25 rows 75%, 26-30 77%, **31 or more 63%** (the labels with the most rows are the crowded families, not the data-poor ones; the 4 requests whose label has 15 rows or fewer are too few to say anything).
* Requests carrying a concrete name or value against those that do not: examples 84% against 87%, large 68% against 67%. No difference; a value is not what confuses the head.
* Where the head is wrong, the right command is usually still close: 84% in the top 3 on the large workflow, so ranking is the gap on a large workflow (a 17-point miss rate on the best label, a 5-point miss rate by rank 10).
* A floor under "wrong confident": the head's best label overall is the bare-value label on 5 of 735 example requests and 3 of 294 large ones, and the product answers a bare value alone. That alone is 0.7% and 1.0% of requests wrong and confident at any threshold, which is why targets of 0.5% and 1% are not reachable below.


## Part 2: lever sweep

Each lever alone against the shipped head, then the best few combined. Full tables (one row per variant, both groups) are
produced by `local_head_levers.py tables`; the rows below are every variant that was run. Columns are explained above; all
numbers are percentages of the group's routing requests (escalation columns: of its escalation requests).
Ranking within each table is by the first "wrong confident" column, lowest first (ties are noise: one large-workflow request is 0.34 points).


**examples**

| Variant | Wrong confident at 50% right outright | at 60% | Best label first | Right in top 3 | Escalations: best label is outside | ... and is the expected ancestor | Wrong confident when 50% of requests are answered | when 70% |
|---|---|---|---|---|---|---|---|---|
| bge-small, LR C=100 (shipped) | 1.6 | 2.6 | 85.3 | 95.2 | 99.2 | 97.5 | 1.0 | 4.4 |
| encoder bge-base | 1.6 | 2.2 | 86.7 | 95.9 | 99.2 | 98.3 | 1.2 | 3.5 |
| encoder all-mpnet-base-v2 | 0.4 | 0.8 | 88.0 | 98.6 | 99.2 | 94.2 | 0.5 | 2.2 |
| encoder all-MiniLM-L12-v2 | 1.5 | 2.0 | 85.4 | 97.3 | 99.2 | 95.0 | 2.0 | 5.2 |
| encoder e5-base-v2 | 1.4 | 2.0 | 86.0 | 96.1 | 99.2 | 92.5 | 1.0 | 3.7 |
| encoder gte-base | 1.9 | 2.0 | 86.8 | 96.3 | 99.2 | 95.8 | 1.4 | 3.7 |
| encoder e5-small-v2 | 2.9 | 3.7 | 85.2 | 95.8 | 99.2 | 95.8 | 1.8 | 4.8 |
| encoder gte-small | 1.8 | 2.7 | 84.2 | 96.5 | 99.2 | 93.3 | 1.2 | 4.1 |
| LR C=10 | 1.4 | 2.7 | 85.0 | 95.6 | 99.2 | 95.8 | 1.2 | 4.5 |
| LR C=1000 | 1.5 | 2.3 | 85.6 | 94.4 | 99.2 | 95.8 | 1.0 | 4.4 |
| kNN k=10 alone | 4.1 | 8.2 | 76.5 | 92.9 | 98.3 | 85.0 | 3.5 | 9.7 |
| LR + kNN10 average | 1.8 | 2.9 | 82.6 | 96.6 | 99.2 | 95.8 | 1.4 | 4.4 |
| prototypes alone | 3.7 | 7.1 | 81.0 | 94.8 | 99.2 | 92.5 | 2.9 | 9.5 |
| LR + prototypes average | 1.6 | 2.6 | 85.6 | 95.9 | 99.2 | 95.0 | 1.1 | 4.2 |
| + description and name rows | 1.5 | 2.2 | 85.7 | 95.4 | 99.2 | 96.7 | 1.0 | 4.2 |
| seeds x3 | 1.4 | 2.7 | 85.4 | 95.0 | 99.2 | 97.5 | 0.7 | 3.9 |
| description rows + seeds x3 | 1.0 | 2.0 | 86.0 | 95.4 | 99.2 | 97.5 | 0.7 | 3.7 |
| value masking | 2.2 | 2.9 | 83.1 | 95.6 | 99.2 | 96.7 | 1.9 | 5.2 |
| description + seeds x3 + masking | 1.6 | 2.0 | 83.8 | 95.6 | 99.2 | 96.7 | 1.5 | 4.6 |
| head per context (mask only) | 1.2 | 2.3 | 85.7 | 95.4 | 11.7 | 0.0 | 0.5 | 2.4 |
| head per context + foreign seeds as wildcard | 1.4 | 2.6 | 85.3 | 95.2 | 72.5 | 0.0 | 1.4 | 3.7 |
| context name prepended | 1.9 | 2.6 | 87.1 | 96.3 | 95.8 | 91.7 | 1.0 | 2.4 |
| merge labels sharing a command name | 1.6 | 2.6 | 85.3 | 95.2 | 99.2 | 97.5 | 1.0 | 4.4 |
| e5-base + description + seeds x3 | 0.7 | 1.1 | 86.8 | 96.7 | 99.2 | 94.2 | 0.8 | 2.4 |
| e5-base + LR/kNN10 | 1.9 | 3.7 | 81.9 | 96.9 | 98.3 | 90.0 | 1.6 | 5.2 |
| mpnet + description + seeds x3 | 0.4 | 0.7 | 88.4 | 98.5 | 99.2 | 94.2 | 0.5 | 2.0 |
| mpnet + LR/kNN10 | 0.5 | 1.4 | 86.9 | 97.8 | 99.2 | 95.0 | 0.5 | 3.0 |
| fine-tune bge-small (all layers) | 1.6 | 1.8 | 89.0 | 96.2 | 99.2 | 99.2 | 0.4 | 1.9 |
| fine-tune bge-small last 2 layers | 1.8 | 2.4 | 86.7 | 95.4 | 99.2 | 96.7 | 1.1 | 3.9 |
| fine-tune bge-small last layer | 1.5 | 2.4 | 86.0 | 95.0 | 99.2 | 97.5 | 1.1 | 4.1 |
| bge-small + trained projection | 1.5 | 2.0 | 86.9 | 94.3 | 99.2 | 95.0 | 0.8 | 3.7 |
| fine-tune bge-base (all layers) | 0.7 | 1.2 | 90.6 | 97.1 | 99.2 | 98.3 | 0.3 | 1.4 |
| fine-tune bge-base last 2 layers | 1.8 | 2.0 | 88.3 | 96.5 | 99.2 | 97.5 | 1.1 | 2.3 |
| bge-base + trained projection | 1.8 | 1.8 | 87.1 | 95.8 | 99.2 | 98.3 | 1.0 | 2.7 |

**large**

| Variant | Wrong confident at 20% right outright | at 30% | Best label first | Right in top 3 | Escalations: best label is outside | ... and is the expected ancestor | Wrong confident when 50% of requests are answered | when 70% |
|---|---|---|---|---|---|---|---|---|
| bge-small, LR C=100 (shipped) | 1.7 | 2.7 | 67.7 | 84.4 | 100.0 | 75.5 | 14.6 | 26.9 |
| encoder bge-base | 1.0 | 1.4 | 72.8 | 89.5 | 100.0 | 79.6 | 11.9 | 22.4 |
| encoder all-mpnet-base-v2 | 0.7 | 2.7 | 69.0 | 87.4 | 95.9 | 75.5 | 15.6 | 29.3 |
| encoder all-MiniLM-L12-v2 | 0.7 | 1.7 | 67.0 | 83.3 | 98.0 | 65.3 | 15.3 | 27.9 |
| encoder e5-base-v2 | 1.0 | 1.0 | 75.2 | 88.8 | 100.0 | 75.5 | 12.2 | 20.4 |
| encoder gte-base | 1.0 | 1.4 | 71.8 | 86.7 | 100.0 | 83.7 | 11.9 | 22.8 |
| encoder e5-small-v2 | 1.7 | 2.0 | 70.1 | 85.0 | 100.0 | 81.6 | 12.9 | 23.1 |
| encoder gte-small | 2.4 | 3.7 | 70.4 | 84.0 | 100.0 | 71.4 | 14.3 | 28.6 |
| LR C=10 | 1.4 | 1.7 | 70.1 | 87.1 | 100.0 | 73.5 | 11.2 | 23.5 |
| LR C=1000 | 3.1 | 5.1 | 68.7 | 84.7 | 100.0 | 77.6 | 17.0 | 29.6 |
| kNN k=10 alone | 1.4 | 3.4 | 66.3 | 75.9 | 98.0 | 63.3 | 16.7 | 29.3 |
| LR + kNN10 average | 0.7 | 1.4 | 70.7 | 84.7 | 98.0 | 75.5 | 11.6 | 25.2 |
| prototypes alone | 3.7 | 5.4 | 65.6 | 85.7 | 98.0 | 77.6 | 14.3 | 24.8 |
| LR + prototypes average | 2.4 | 3.1 | 69.0 | 86.7 | 100.0 | 79.6 | 13.6 | 26.5 |
| + description and name rows | 1.7 | 2.7 | 69.0 | 84.7 | 100.0 | 73.5 | 13.6 | 25.9 |
| seeds x3 | 1.7 | 2.7 | 68.0 | 85.7 | 100.0 | 81.6 | 15.6 | 26.5 |
| description rows + seeds x3 | 1.4 | 3.1 | 68.7 | 84.7 | 100.0 | 79.6 | 14.3 | 26.9 |
| value masking | 3.4 | 4.1 | 69.0 | 86.4 | 100.0 | 75.5 | 15.6 | 28.2 |
| description + seeds x3 + masking | 2.4 | 3.4 | 69.4 | 85.4 | 100.0 | 75.5 | 14.6 | 27.6 |
| head per context (mask only) | 1.4 | 2.0 | 70.7 | 86.4 | 4.1 | 0.0 | 3.1 | 11.2 |
| head per context + foreign seeds as wildcard | 0.3 | 0.7 | 71.1 | 86.1 | 93.9 | 0.0 | 16.0 | 26.5 |
| context name prepended | 4.1 | 5.1 | 68.7 | 85.4 | 57.1 | 32.7 | 4.4 | 14.3 |
| merge labels sharing a command name | 2.0 | 2.7 | 68.7 | 83.7 | 100.0 | 73.5 | 18.7 | 31.3 |
| e5-base + description + seeds x3 | 0.7 | 0.7 | 73.8 | 88.4 | 100.0 | 73.5 | 11.6 | 22.4 |
| e5-base + LR/kNN10 | 0.3 | 0.3 | 75.2 | 88.1 | 95.9 | 73.5 | 9.2 | 18.4 |
| mpnet + description + seeds x3 | 0.7 | 2.4 | 66.3 | 86.1 | 98.0 | 79.6 | 17.0 | 29.6 |
| mpnet + LR/kNN10 | 0.7 | 1.4 | 67.7 | 87.4 | 98.0 | 77.6 | 15.6 | 29.3 |
| fine-tune bge-small (all layers) | 1.7 | 2.0 | 71.8 | 86.7 | 98.0 | 85.7 | 10.5 | 21.1 |
| fine-tune bge-small last 2 layers | 2.4 | 3.4 | 68.0 | 84.0 | 100.0 | 79.6 | 14.6 | 26.5 |
| fine-tune bge-small last layer | 2.0 | 3.1 | 67.7 | 83.7 | 100.0 | 79.6 | 14.6 | 29.3 |
| bge-small + trained projection | 2.4 | 3.1 | 65.0 | 82.3 | 98.0 | 77.6 | 12.9 | 26.9 |
| fine-tune bge-base (all layers) | 1.4 | 2.0 | 74.5 | 90.5 | 100.0 | 83.7 | 9.9 | 18.4 |
| fine-tune bge-base last 2 layers | 1.4 | 1.7 | 73.1 | 90.1 | 100.0 | 81.6 | 9.9 | 19.7 |
| bge-base + trained projection | 0.7 | 1.7 | 65.6 | 86.1 | 100.0 | 79.6 | 13.9 | 27.2 |

### Encoders (what was used; all eight downloaded)

| Encoder | Pooling | Prefix | Median request time, 10 threads / 1 thread | Weights (fp32) |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` (shipped) | CLS | none | 7.5 / 13.9 ms | 133 MB |
| `BAAI/bge-base-en-v1.5` | CLS | none | 18.4 / 46.6 ms | 438 MB |
| `sentence-transformers/all-MiniLM-L12-v2` | mean | none | 8.2 / 13.7 ms | 133 MB |
| `sentence-transformers/all-mpnet-base-v2` | mean | none | 17.3 / 45.3 ms | 438 MB |
| `thenlper/gte-small` | mean | none | 13.5 / 33.0 ms (noisy run) | 133 MB |
| `thenlper/gte-base` | mean | none | 38.7 / 110.8 ms (noisy run) | 438 MB |
| `intfloat/e5-small-v2` | mean | `query: ` on every text | 8.5 / 14.7 ms | 133 MB |
| `intfloat/e5-base-v2` | mean | `query: ` on every text | 20.2 / 50.5 ms | 438 MB |

Request time is the encoder for one request on this machine's CPU (batch of one, median of 90 requests), not the whole router; the head adds
0.03 ms (linear head) to 0.1 ms (nearest neighbours). Texts are L2-normalised, 128 tokens. bge uses the shipped no-prefix setting
(its retrieval "query instruction" is not meant for symmetric matching).

### Findings, lever by lever

1. **Encoder.** Base-size encoders help the large workflow clearly (best label first: bge-base 72.8, gte-base 71.8, e5-base 75.2 against 67.7) and the examples only a little (86.7, 86.8, 86.0 against 85.3, inside the noise), so the gain holds in direction on both, in size only on the large workflow. At the same time their wrong confident answers fall on both (e5-base: 1.4 / 2.0 on the examples and 1.0 / 1.0 on the large workflow against 1.6 / 2.6 and 1.7 / 2.7). `all-mpnet-base-v2` is the best encoder on the examples (88.0 first, 0.4 / 0.8 wrong confident, 98.6 in top 3) and a modest one on the large workflow (69.0, 0.7 / 2.7); it loses some escalation accuracy (ancestor 94.2 against 97.5). Small-size encoders (MiniLM, gte-small, e5-small) are not better than the shipped one.
2. **Head.** C = 10 and C = 1000 are within noise (1000 is worse on the large workflow's wrong answers). Nearest neighbours alone and prototypes alone are clearly worse (examples first 76.5 and 81.0). Averaging the logistic head with a 10-neighbour vote helps the large workflow's wrong answers (0.7 / 1.4 against 1.7 / 2.7) but costs the examples 2.7 points of best-label accuracy, so it does not hold on both; it also stores every training row (3.6 MB against 134 KB on the large workflow). The nearest-neighbour vote cannot rank beyond its k neighbours (recall at 10 is capped).
3. **Data shape, no new generation.** Adding each command's description and humanised name as rows (+0.4 / +1.3 first) and upweighting the seeds x3 (+0.1 / +0.3) are inside the noise alone; together 86.0 / 68.7 first and 1.0 / 2.0 and 1.4 / 3.1 wrong confident: the examples improve slightly, the large workflow does not. **Value masking hurts** (examples -2.2 first, large wrong confident 3.4 / 4.1): the values carry some of the signal, and the masked placeholder is a single token the encoder learned nothing about.
4. **Scope.** (i) A head per context over that context's mask only: best label first the same (85.7 / 70.7), wrong confident a little lower, but it can no longer pass up: on escalation requests its best label is outside the mask for only 11.7% (examples) and 4.1% (large), and never the ancestor. Adding the product's sparse "not here" seeds (one seed of every foreign command as wildcard) brings passing up back (72.5% / 93.9% of escalations) but as a bare "not here", and at the product's thresholds it passes 30% of the large workflow's routing requests up wrongly (see Part 3). Storage is 2 to 3 times larger (87,780 and 403,480 bytes against 44,660 and 133,980 for library and the large workflow). (ii) Prepending the context name to the text (training rows repeated once per context that can see the label): best label first +1.8 / +1.0, but escalation falls apart (ancestor 91.7 / 32.7, large wrong confident 4.1 / 5.1): the context name makes everything look in-context. (iii) Merging labels that share a command name: the label structure allows it (labels are `Owner/command`; a map from label to group and back through the mask), 4 names in the large workflow and none in the examples; no effect.
5. **Light fine-tuning, below.**


### Light fine-tuning (the one lever that trains a model)

Recipe, identical for every row below: a cosine classifier over every label, initialised at the class prototypes, trained with
cross-entropy (scale 20) on the workflow's training rows with AdamW (encoder 3e-5, classifier 1e-3), batch 32, 4 epochs,
seed 0, deterministic algorithms; afterwards the same logistic head is fitted on the fine-tuned embeddings. One fine-tuned
encoder per workflow. This virtualenv's torch has no CUDA build, so **all training ran on the CPU** (20 threads); the machine's GPU was not used.

| Variant | Best label first (examples / large) | Wrong confident at 50% (examples) / 20% (large) right outright | Training time on CPU (library / retail / large) | **Stored per workflow and per trained version** | Request time, 10 / 1 threads |
|---|---|---|---|---|---|
| frozen bge-small (shipped) | 85.3 / 67.7 | 1.6 / 1.7 | none | nothing (shared 133 MB download) | 7.5 / 13.9 ms |
| fine-tune bge-small, all layers | 89.0 / 71.8 | 1.6 / 1.7 | 41 / 44 / 163 s | **133 MB** | 7.5 / 13.9 ms |
| fine-tune bge-small, last 2 layers | 86.7 / 68.0 | 1.8 / 2.4 | 22 / 23 / 100 s | 14.2 MB (the rest stays the shared download) | 7.5 / 13.9 ms |
| fine-tune bge-small, last layer | 86.0 / 67.7 | 1.5 / 2.0 | 19 / 20 / 81 s | 7.1 MB | 7.5 / 13.9 ms |
| bge-small + trained 384 x 384 projection | 86.9 / 65.0 | 1.5 / 2.4 | under 2 s | 0.59 MB (projection only) | 7.5 / 13.9 ms |
| frozen bge-base | 86.7 / 72.8 | 1.6 / 1.0 | none | nothing (shared 438 MB download) | 18.4 / 46.6 ms |
| **fine-tune bge-base, all layers** | **90.6 / 74.5** | **0.7 / 1.4** | 125 / 125 / 490 s | **438 MB** | 18.4 / 46.6 ms |
| fine-tune bge-base, last 2 layers | 88.3 / 73.1 | 1.8 / 1.4 | 58 / 58 / 281 s | 56.7 MB | 18.4 / 46.6 ms |
| bge-base + trained 768 x 768 projection | 87.1 / 65.6 | 1.8 / 0.7 | 1 / 1 / 4 s | 2.4 MB (projection only) | 18.4 / 46.6 ms |

* **Fine-tuning turns a shared 130 MB (bge-small) or 440 MB (bge-base) download into a per-workflow, per-version artifact of that size.** The encoder's weights then belong to the workflow and to each trained version (the product keeps the current and one previous version, so twice that on disk), and every machine that serves the workflow needs them. Request time does not change (the architecture is the same; add the head's 0.03 ms).
* **Reproducibility**: two runs at the same seed gave identical scores (largest difference 0.0, same top label on every request) for bge-small and for bge-base on all three workflows, on the CPU with deterministic algorithms. Different hardware or thread counts were not tested.
* **Gain holds on both groups** for full fine-tuning: +3.7 / +4.1 (bge-small) and **+5.3 / +6.8 (bge-base)** best label first; the base model also has the lowest wrong confident answers of any variant at the same right-outright share on both groups (0.7 and 1.2 on the examples at 50% and 60% right outright; 1.4 and 2.0 on the large workflow at 20% and 30%).
* **A cheaper form that keeps most of the gain exists for the base encoder only**: training just its last two layers keeps +3.0 / +5.4 of the +5.3 / +6.8 (57% and 79%) at 13% of the storage (56.7 MB) and 57% of the training time. For bge-small the last one or two layers keep almost none of the gain (+0.7 to +1.4 / 0.0 to +0.3, noise).
* **A trained projection on top of the frozen encoder does not work**: +1.6 / -2.7 (bge-small) and +0.4 / -7.2 (bge-base). It overfits the training rows; its wrong-confident figures do not improve consistently either.
* Caveat for thresholds: a fine-tuned encoder has seen every training row, so a cross-fit on those rows is optimistic. The operating-point rows of fine-tuned variants below are therefore chosen on the development cases (in-sample) only.


## Part 3: operating points the owner can choose from

The product escalates (local model, then a remote model, then the user or agent is asked), so the error to minimise is the
**confident wrong answer**, and being asked more often is the acceptable cost. Each table row is a policy; read across.
"Wrong confident" counts every request answered with one command that is not the right one, **including requests wrongly passed up
to a parent context** (also shown on its own). "Not answered" requests go to the next link; they are split by whether the right
command is among the three the head would show.

How a row is reached. Two numbers decide the local model's behaviour: the *threshold* (the probability the best command inside the
context must reach to be answered alone) and the *pass-up margin* (how far the best score outside the context must lead for the
head to pass the request up to the parent alone). "never" means the local model never passes up alone: such a request is shown the
top three and goes to the next link, which is one natural way to remove that error.

* **In-sample rows** pick the threshold and margin, on the development cases themselves, that answer the most requests right
  outright while wrong confident answers stay within the target (all routing requests as the base). For rows that keep pass-ups, the
  margin is also chosen to hand on as many escalation requests as possible while keeping at least 90% (else 75%, else 50%; the level
  is in the row) of the right answers of the never-pass-up row with the same target. **These are in-sample operating points: real
  rates will be somewhat worse.** A row is "not reachable" when no setting stays within the budget (bare-value answers alone put a
  floor of 0.7% and 1.0% under the examples and the large workflow, and a pass-up is itself a wrong answer more often than not).
* **Cross-fit rows** use the product's own rule on the training rows (five folds, folds by row text; threshold = lowest at which
  wrong lone answers are at most the target of the rows, never below 0.5; margin = smallest at which at most 0.5% of rows are sent up),
  then score the development cases (out of sample). Three families: one pair per workflow (what the product does), one pair per
  context (each context's own rows, falling back to the workflow-wide pair under 30 distinct rows; none fell back: the large
  workflow has 16 contexts with their own pair, library 5, retail 1), and per-context thresholds with pass-up never committed alone.
* The shipped operating point is the product's calibrated pair: examples library 0.529 / 0.443 and retail 0.500 / 0.000, large 0.591 / 0.622.

### What the tables say

* **The shipped operating point commits too many wrong answers**: examples 6.0% wrong (2.0 of it wrongly passed up) for 69.4% right outright; large 8.5% wrong (6.1 of it wrongly passed up) for 29.3% right.
* **Never passing up alone** removes the wrongly-passed-up part and costs the escalation hand-on (96.7% and 61.2% of escalation requests, which then go to the next link): at 3% wrong the examples answer 64.8% right outright (shipped: 69.4% at 6.0% wrong) and the large workflow 34.4% (shipped: 29.3% at 8.5% wrong) with 2.7% wrong.
* **Cross-fit thresholds do not hit their targets.** The training rows are easy (cross-fit about 97% and 94% right), real requests are not: a 0.5% target gives 5.0% wrong on the examples and 7.8% on the large workflow, and from a 2% target up every context lands at or within 0.04 of the 0.5 floor, so the targets stop mattering. Choosing the operating point therefore needs a held-out error estimate (the in-sample rows above, or a confirmation set), not the cross-fit.
* **Per-context calibration does not move the trade-off.** On the one workflow head, a pair per context against a pair per workflow at the 1% target: examples 69.3 right / 6.1 wrong against 69.5 / 5.9; large 26.2 / 9.2 against 29.3 / 8.5. With the head-per-context variant: examples 72.9 / 5.0 against 72.7 / 5.4; large 47.6 / 5.8 against 51.4 / 7.1. Both move along the same curve (a little less right, a little less wrong), inside the noise; at targets of 2% and over all contexts sit at the floor. Contexts' thresholds range 0.50 to 0.56 (examples) and 0.50 to 0.74 (large) at the 1% target.
* **The wrong pass-ups are spread, not concentrated in one context.** Large workflow at the shipped operating point: 18 wrong pass-ups in 9 of the 16 contexts; 3 contexts hold half of them (10 of 18) and 6 hold 80%. Examples: 15 in 4 of 6 contexts, 2 contexts hold half. A per-context pair at the 1% target does not change the picture (19 in 8 contexts; 2 contexts hold half). A fix aimed at "the one bad context" would not remove them.
* **Per-context calibration with a head per context** (cross-fit rows below) is as good as the workflow head on wrong-confident answers only because that head cannot pass up (escalations passed up: 8.3% of the examples' and 0.0% of the large workflow's); with the sparse "not here" seeds it passes up readily and wrongly (30.3% of the large workflow's routing requests wrongly passed up at the product's rule).
* **Storage per workflow** (arrays only; the product's file adds a small JSON of masks): shipped single head 44,660 bytes (library), 32,340 (retail), 133,980 (large); with a threshold and a margin per context 44,700, 32,348 and 134,108 (8 bytes per context); a head per context on the shared vectors 87,780, 32,340 and 403,480.


### Operating-point tables

**Shipped head, examples: points chosen on the development cases themselves (in-sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| the shipped operating point | 69.4 | 6.0 | 2.0 | 21.1 | 3.5 | 96.7 | 0.50 / 0.00 |
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept (keeps 50% of the right answers) | 33.3 | 1.9 | 1.1 | 60.5 | 4.2 | 84.2 | 0.88 / 0.70 |
| target 3% wrong, pass-ups kept (keeps 75% of the right answers) | 50.7 | 3.0 | 1.4 | 42.3 | 3.9 | 88.3 | 0.75 / 0.58 |
| target 5% wrong, pass-ups kept (keeps 90% of the right answers) | 66.7 | 4.9 | 1.4 | 24.9 | 3.5 | 88.3 | 0.56 / 0.58 |
| never pass up alone, target 1% | 40.4 | 1.0 | 0.0 | 54.1 | 4.5 | 0.0 | 0.83 / never |
| never pass up alone, target 3% | 64.8 | 3.0 | 0.0 | 28.2 | 4.1 | 0.0 | 0.60 / never |

**Shipped head, examples: thresholds from the cross-fit rule on the training rows, scored on the development cases (out of sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Cross-fit thresholds |
|---|---|---|---|---|---|---|---|
| one pair per workflow, target 0.5% | 66.0 | 5.0 | 1.9 | 25.0 | 3.9 | 91.7 | 0.56/0.52; 0.61/0.00 |
| one pair per context, target 0.5% | 64.9 | 5.4 | 2.2 | 25.7 | 3.9 | 91.7 | thr 0.50-0.69, 5 own, 0 fell back; thr 0.61-0.61, 1 own, 0 fell back |
| one pair per workflow, target 1% | 69.5 | 5.9 | 1.9 | 21.1 | 3.5 | 91.7 | 0.53/0.52; 0.50/0.00 |
| one pair per context, target 1% | 69.3 | 6.1 | 2.2 | 21.1 | 3.5 | 91.7 | thr 0.50-0.56, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| per-context thresholds, never pass up alone, target 1% | 69.3 | 3.9 | 0.0 | 23.0 | 3.8 | 0.0 | thr 0.50-0.56, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 2% | 70.9 | 6.1 | 1.9 | 19.6 | 3.4 | 91.7 | 0.50/0.52; 0.50/0.00 |
| one pair per context, target 2% | 70.3 | 6.1 | 2.2 | 20.0 | 3.5 | 91.7 | thr 0.50-0.52, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 3% | 70.9 | 6.1 | 1.9 | 19.6 | 3.4 | 91.7 | 0.50/0.52; 0.50/0.00 |
| one pair per context, target 3% | 70.9 | 6.4 | 2.2 | 19.3 | 3.4 | 91.7 | thr 0.50-0.50, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| per-context thresholds, never pass up alone, target 3% | 70.9 | 4.2 | 0.0 | 21.2 | 3.7 | 0.0 | thr 0.50-0.50, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 5% | 70.9 | 6.1 | 1.9 | 19.6 | 3.4 | 91.7 | 0.50/0.52; 0.50/0.00 |
| one pair per context, target 5% | 70.9 | 6.4 | 2.2 | 19.3 | 3.4 | 91.7 | thr 0.50-0.50, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |

**Shipped head, large: points chosen on the development cases themselves (in-sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| the shipped operating point | 29.3 | 8.5 | 6.1 | 49.3 | 12.9 | 61.2 | 0.59 / 0.62 |
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept | not reachable | | | | | | |
| target 3% wrong, pass-ups kept | not reachable | | | | | | |
| target 5% wrong, pass-ups kept (keeps 50% of the right answers) | 24.1 | 4.8 | 3.1 | 57.5 | 13.6 | 26.5 | 0.66 / 0.90 |
| never pass up alone, target 3% | 34.4 | 2.7 | 0.0 | 48.3 | 14.6 | 0.0 | 0.47 / never |

**Shipped head, large: thresholds from the cross-fit rule on the training rows, scored on the development cases (out of sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Cross-fit thresholds |
|---|---|---|---|---|---|---|---|
| one pair per workflow, target 0.5% | 19.0 | 7.8 | 6.1 | 59.9 | 13.3 | 61.2 | 0.77/0.62 |
| one pair per context, target 0.5% | 22.1 | 8.8 | 6.5 | 55.8 | 13.3 | 63.3 | thr 0.50-0.82, 16 own, 0 fell back |
| one pair per workflow, target 1% | 29.3 | 8.5 | 6.1 | 49.3 | 12.9 | 61.2 | 0.59/0.62 |
| one pair per context, target 1% | 26.2 | 9.2 | 6.5 | 51.7 | 12.9 | 63.3 | thr 0.50-0.74, 16 own, 0 fell back |
| per-context thresholds, never pass up alone, target 1% | 26.2 | 2.7 | 0.0 | 56.5 | 14.6 | 0.0 | thr 0.50-0.74, 16 own, 0 fell back |
| one pair per workflow, target 2% | 31.3 | 8.8 | 6.1 | 46.9 | 12.9 | 61.2 | 0.50/0.62 |
| one pair per context, target 2% | 29.9 | 9.2 | 6.5 | 48.0 | 12.9 | 63.3 | thr 0.50-0.54, 16 own, 0 fell back |
| one pair per workflow, target 3% | 31.3 | 8.8 | 6.1 | 46.9 | 12.9 | 61.2 | 0.50/0.62 |
| one pair per context, target 3% | 31.3 | 9.2 | 6.5 | 46.6 | 12.9 | 63.3 | thr 0.50-0.50, 16 own, 0 fell back |
| per-context thresholds, never pass up alone, target 3% | 31.3 | 2.7 | 0.0 | 51.4 | 14.6 | 0.0 | thr 0.50-0.50, 16 own, 0 fell back |
| one pair per workflow, target 5% | 31.3 | 8.8 | 6.1 | 46.9 | 12.9 | 61.2 | 0.50/0.62 |
| one pair per context, target 5% | 31.3 | 9.2 | 6.5 | 46.6 | 12.9 | 63.3 | thr 0.50-0.50, 16 own, 0 fell back |

**e5-base + description rows + seeds x3, examples: points chosen on the development cases themselves (in-sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept (keeps 90% of the right answers) | 63.0 | 1.9 | 0.7 | 32.1 | 3.0 | 63.3 | 0.60 / 0.89 |
| target 3% wrong, pass-ups kept (keeps 90% of the right answers) | 65.0 | 3.0 | 1.5 | 29.0 | 3.0 | 96.7 | 0.57 / 0.44 |
| target 5% wrong, pass-ups kept (keeps 90% of the right answers) | 72.0 | 4.9 | 1.6 | 20.3 | 2.9 | 99.2 | 0.47 / 0.34 |
| never pass up alone, target 1% | 59.6 | 1.0 | 0.0 | 36.3 | 3.1 | 0.0 | 0.65 / never |
| never pass up alone, target 3% | 71.8 | 3.0 | 0.0 | 22.2 | 3.0 | 0.0 | 0.47 / never |

**e5-base + description rows + seeds x3, large: points chosen on the development cases themselves (in-sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept | not reachable | | | | | | |
| target 3% wrong, pass-ups kept | not reachable | | | | | | |
| target 5% wrong, pass-ups kept (keeps 90% of the right answers) | 48.3 | 4.8 | 2.4 | 37.4 | 9.5 | 24.5 | 0.30 / 0.88 |
| never pass up alone, target 1% | 36.1 | 0.7 | 0.0 | 51.7 | 11.6 | 0.0 | 0.50 / never |
| never pass up alone, target 3% | 49.3 | 2.7 | 0.0 | 36.7 | 11.2 | 0.0 | 0.28 / never |

**bge-base fine-tuned (all layers), examples: points chosen on the development cases themselves (in-sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept (keeps 90% of the right answers) | 56.6 | 1.0 | 0.3 | 40.0 | 2.4 | 73.3 | 0.82 / 0.91 |
| target 2% wrong, pass-ups kept (keeps 90% of the right answers) | 67.6 | 1.9 | 0.7 | 28.0 | 2.4 | 95.8 | 0.71 / 0.61 |
| target 3% wrong, pass-ups kept (keeps 90% of the right answers) | 72.4 | 3.0 | 1.1 | 22.3 | 2.3 | 97.5 | 0.65 / 0.42 |
| target 5% wrong, pass-ups kept (keeps 90% of the right answers) | 78.8 | 4.9 | 1.6 | 14.1 | 2.2 | 98.3 | 0.55 / 0.34 |
| never pass up alone, target 1% | 57.8 | 1.0 | 0.0 | 38.6 | 2.6 | 0.0 | 0.81 / never |
| never pass up alone, target 3% | 77.7 | 3.0 | 0.0 | 16.9 | 2.4 | 0.0 | 0.55 / never |

**bge-base fine-tuned (all layers), large: points chosen on the development cases themselves (in-sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept (keeps 90% of the right answers) | 5.8 | 0.7 | 0.0 | 84.7 | 8.8 | 18.4 | 0.97 / 0.97 |
| target 2% wrong, pass-ups kept (keeps 90% of the right answers) | 29.9 | 1.7 | 0.0 | 59.5 | 8.8 | 18.4 | 0.76 / 0.97 |
| target 3% wrong, pass-ups kept (keeps 90% of the right answers) | 43.2 | 2.7 | 0.7 | 45.2 | 8.8 | 32.7 | 0.57 / 0.95 |
| target 5% wrong, pass-ups kept (keeps 90% of the right answers) | 50.7 | 4.8 | 1.7 | 35.7 | 8.8 | 53.1 | 0.44 / 0.87 |
| never pass up alone, target 1% | 5.8 | 0.7 | 0.0 | 84.7 | 8.8 | 0.0 | 0.97 / never |
| never pass up alone, target 3% | 48.0 | 2.7 | 0.0 | 40.5 | 8.8 | 0.0 | 0.50 / never |

**bge-base fine-tuned, last 2 layers, examples: points chosen on the development cases themselves (in-sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept (keeps 50% of the right answers) | 35.6 | 1.9 | 0.5 | 59.5 | 3.0 | 54.2 | 0.92 / 0.96 |
| target 3% wrong, pass-ups kept (keeps 90% of the right answers) | 66.3 | 3.0 | 1.0 | 27.8 | 3.0 | 90.0 | 0.64 / 0.71 |
| target 5% wrong, pass-ups kept (keeps 90% of the right answers) | 69.8 | 4.9 | 2.3 | 22.4 | 2.9 | 98.3 | 0.60 / 0.31 |
| never pass up alone, target 3% | 71.8 | 3.0 | 0.0 | 22.0 | 3.1 | 0.0 | 0.58 / never |

**bge-base fine-tuned, last 2 layers, large: points chosen on the development cases themselves (in-sample)**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept (keeps 50% of the right answers) | 25.2 | 1.7 | 0.3 | 64.6 | 8.5 | 8.2 | 0.77 / 0.98 |
| target 3% wrong, pass-ups kept (keeps 90% of the right answers) | 44.9 | 2.7 | 0.7 | 43.9 | 8.5 | 32.7 | 0.49 / 0.95 |
| target 5% wrong, pass-ups kept (keeps 90% of the right answers) | 48.6 | 4.8 | 1.7 | 38.1 | 8.5 | 61.2 | 0.42 / 0.82 |
| never pass up alone, target 3% | 48.6 | 2.7 | 0.0 | 39.8 | 8.8 | 0.0 | 0.43 / never |

### Head per context (mask only) and head per context with the sparse not-here seeds

**scope_percontext, examples: cross-fit thresholds, out of sample**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Cross-fit thresholds |
|---|---|---|---|---|---|---|---|
| one pair per workflow, target 0.5% | 68.3 | 4.1 | 1.0 | 23.7 | 3.9 | 8.3 | 0.63/0.00; 0.61/0.00 |
| one pair per context, target 0.5% | 66.4 | 4.1 | 1.0 | 25.7 | 3.8 | 8.3 | thr 0.53-0.84, 5 own, 0 fell back; thr 0.61-0.61, 1 own, 0 fell back |
| one pair per workflow, target 1% | 72.7 | 5.4 | 1.0 | 18.5 | 3.4 | 8.3 | 0.57/0.00; 0.50/0.00 |
| one pair per context, target 1% | 72.9 | 5.0 | 1.0 | 18.6 | 3.4 | 8.3 | thr 0.50-0.75, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| per-context thresholds, never pass up alone, target 1% | 72.9 | 4.1 | 0.0 | 19.5 | 3.5 | 0.0 | thr 0.50-0.75, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 2% | 75.9 | 6.3 | 1.0 | 14.6 | 3.3 | 8.3 | 0.50/0.00; 0.50/0.00 |
| one pair per context, target 2% | 75.0 | 6.1 | 1.0 | 15.6 | 3.3 | 8.3 | thr 0.50-0.58, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 3% | 75.9 | 6.3 | 1.0 | 14.6 | 3.3 | 8.3 | 0.50/0.00; 0.50/0.00 |
| one pair per context, target 3% | 75.4 | 6.1 | 1.0 | 15.2 | 3.3 | 8.3 | thr 0.50-0.56, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| per-context thresholds, never pass up alone, target 3% | 75.4 | 5.2 | 0.0 | 16.1 | 3.4 | 0.0 | thr 0.50-0.56, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 5% | 75.9 | 6.3 | 1.0 | 14.6 | 3.3 | 8.3 | 0.50/0.00; 0.50/0.00 |
| one pair per context, target 5% | 75.9 | 6.3 | 1.0 | 14.6 | 3.3 | 8.3 | thr 0.50-0.50, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |

**scope_percontext, examples: in-sample**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept (keeps 75% of the right answers) | 45.0 | 1.9 | 1.0 | 48.8 | 4.2 | 8.3 | 0.84 / 0.00 |
| target 3% wrong, pass-ups kept (keeps 75% of the right answers) | 59.2 | 3.0 | 1.0 | 33.7 | 4.1 | 8.3 | 0.73 / 0.00 |
| target 5% wrong, pass-ups kept (keeps 90% of the right answers) | 69.9 | 4.9 | 1.0 | 21.6 | 3.5 | 8.3 | 0.59 / 0.00 |
| never pass up alone, target 1% | 45.0 | 1.0 | 0.0 | 49.7 | 4.4 | 0.0 | 0.84 / never |
| never pass up alone, target 3% | 67.6 | 3.0 | 0.0 | 25.3 | 4.1 | 0.0 | 0.63 / never |

**scope_percontext, large: cross-fit thresholds, out of sample**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Cross-fit thresholds |
|---|---|---|---|---|---|---|---|
| one pair per workflow, target 0.5% | 40.8 | 4.8 | 1.7 | 42.9 | 11.6 | 0.0 | 0.72/0.00 |
| one pair per context, target 0.5% | 40.8 | 5.1 | 1.7 | 42.5 | 11.6 | 0.0 | thr 0.50-0.88, 16 own, 0 fell back |
| one pair per workflow, target 1% | 51.4 | 7.1 | 1.7 | 31.0 | 10.5 | 0.0 | 0.60/0.00 |
| one pair per context, target 1% | 47.6 | 5.8 | 1.7 | 35.4 | 11.2 | 0.0 | thr 0.50-0.80, 16 own, 0 fell back |
| per-context thresholds, never pass up alone, target 1% | 47.6 | 4.1 | 0.0 | 36.4 | 11.9 | 0.0 | thr 0.50-0.80, 16 own, 0 fell back |
| one pair per workflow, target 2% | 54.8 | 10.5 | 1.7 | 25.2 | 9.5 | 0.0 | 0.50/0.00 |
| one pair per context, target 2% | 53.7 | 9.5 | 1.7 | 26.5 | 10.2 | 0.0 | thr 0.50-0.66, 16 own, 0 fell back |
| one pair per workflow, target 3% | 54.8 | 10.5 | 1.7 | 25.2 | 9.5 | 0.0 | 0.50/0.00 |
| one pair per context, target 3% | 54.8 | 10.5 | 1.7 | 25.2 | 9.5 | 0.0 | thr 0.50-0.50, 16 own, 0 fell back |
| per-context thresholds, never pass up alone, target 3% | 54.8 | 8.8 | 0.0 | 26.2 | 10.2 | 0.0 | thr 0.50-0.50, 16 own, 0 fell back |
| one pair per workflow, target 5% | 54.8 | 10.5 | 1.7 | 25.2 | 9.5 | 0.0 | 0.50/0.00 |
| one pair per context, target 5% | 54.8 | 10.5 | 1.7 | 25.2 | 9.5 | 0.0 | thr 0.50-0.50, 16 own, 0 fell back |

**scope_percontext, large: in-sample**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept | not reachable | | | | | | |
| target 3% wrong, pass-ups kept | not reachable | | | | | | |
| target 5% wrong, pass-ups kept (keeps 75% of the right answers) | 42.9 | 4.8 | 1.7 | 40.8 | 11.6 | 0.0 | 0.70 / 0.00 |
| never pass up alone, target 3% | 39.5 | 2.7 | 0.0 | 45.2 | 12.6 | 0.0 | 0.76 / never |

**scope_percontext_fs, examples: cross-fit thresholds, out of sample**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Cross-fit thresholds |
|---|---|---|---|---|---|---|---|
| one pair per workflow, target 0.5% | 66.7 | 6.3 | 3.1 | 23.3 | 3.8 | 72.5 | 0.63/0.00; 0.61/0.00 |
| one pair per context, target 0.5% | 65.2 | 6.3 | 3.1 | 24.9 | 3.7 | 72.5 | thr 0.53-0.84, 5 own, 0 fell back; thr 0.61-0.61, 1 own, 0 fell back |
| one pair per workflow, target 1% | 71.8 | 7.3 | 3.1 | 17.6 | 3.3 | 72.5 | 0.57/0.00; 0.50/0.00 |
| one pair per context, target 1% | 70.1 | 7.3 | 3.1 | 19.3 | 3.3 | 72.5 | thr 0.50-0.75, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| per-context thresholds, never pass up alone, target 1% | 70.1 | 4.2 | 0.0 | 22.0 | 3.7 | 0.0 | thr 0.50-0.75, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 2% | 73.9 | 8.4 | 3.1 | 14.4 | 3.3 | 72.5 | 0.50/0.00; 0.50/0.00 |
| one pair per context, target 2% | 73.1 | 8.0 | 3.1 | 15.6 | 3.3 | 72.5 | thr 0.50-0.58, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 3% | 73.9 | 8.4 | 3.1 | 14.4 | 3.3 | 72.5 | 0.50/0.00; 0.50/0.00 |
| one pair per context, target 3% | 73.5 | 8.0 | 3.1 | 15.2 | 3.3 | 72.5 | thr 0.50-0.56, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| per-context thresholds, never pass up alone, target 3% | 73.5 | 4.9 | 0.0 | 18.0 | 3.7 | 0.0 | thr 0.50-0.56, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |
| one pair per workflow, target 5% | 73.9 | 8.4 | 3.1 | 14.4 | 3.3 | 72.5 | 0.50/0.00; 0.50/0.00 |
| one pair per context, target 5% | 73.9 | 8.4 | 3.1 | 14.4 | 3.3 | 72.5 | thr 0.50-0.50, 5 own, 0 fell back; thr 0.50-0.50, 1 own, 0 fell back |

**scope_percontext_fs, examples: in-sample**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept | not reachable | | | | | | |
| target 3% wrong, pass-ups kept | not reachable | | | | | | |
| target 5% wrong, pass-ups kept (keeps 75% of the right answers) | 56.5 | 4.9 | 3.1 | 34.7 | 3.9 | 72.5 | 0.74 / 0.00 |
| never pass up alone, target 1% | 43.5 | 1.0 | 0.0 | 51.0 | 4.5 | 0.0 | 0.84 / never |
| never pass up alone, target 3% | 63.5 | 3.0 | 0.0 | 29.3 | 4.2 | 0.0 | 0.66 / never |

**scope_percontext_fs, large: cross-fit thresholds, out of sample**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Cross-fit thresholds |
|---|---|---|---|---|---|---|---|
| one pair per workflow, target 0.5% | 32.0 | 31.3 | 30.3 | 31.3 | 5.4 | 93.9 | 0.72/0.00 |
| one pair per context, target 0.5% | 31.0 | 31.6 | 30.3 | 32.0 | 5.4 | 93.9 | thr 0.50-0.88, 16 own, 0 fell back |
| one pair per workflow, target 1% | 38.8 | 32.0 | 30.3 | 23.8 | 5.4 | 93.9 | 0.60/0.00 |
| one pair per context, target 1% | 38.1 | 32.0 | 30.3 | 24.5 | 5.4 | 93.9 | thr 0.50-0.80, 16 own, 0 fell back |
| per-context thresholds, never pass up alone, target 1% | 38.1 | 1.7 | 0.0 | 46.6 | 13.6 | 0.0 | thr 0.50-0.80, 16 own, 0 fell back |
| one pair per workflow, target 2% | 45.6 | 34.4 | 30.3 | 15.6 | 4.4 | 93.9 | 0.50/0.00 |
| one pair per context, target 2% | 43.2 | 33.3 | 30.3 | 18.7 | 4.8 | 93.9 | thr 0.50-0.66, 16 own, 0 fell back |
| one pair per workflow, target 3% | 45.6 | 34.4 | 30.3 | 15.6 | 4.4 | 93.9 | 0.50/0.00 |
| one pair per context, target 3% | 45.6 | 34.4 | 30.3 | 15.6 | 4.4 | 93.9 | thr 0.50-0.50, 16 own, 0 fell back |
| per-context thresholds, never pass up alone, target 3% | 45.6 | 4.1 | 0.0 | 37.8 | 12.6 | 0.0 | thr 0.50-0.50, 16 own, 0 fell back |
| one pair per workflow, target 5% | 45.6 | 34.4 | 30.3 | 15.6 | 4.4 | 93.9 | 0.50/0.00 |
| one pair per context, target 5% | 45.6 | 34.4 | 30.3 | 15.6 | 4.4 | 93.9 | thr 0.50-0.50, 16 own, 0 fell back |

**scope_percontext_fs, large: in-sample**

| Point | Right outright | Wrong confident | of which wrongly passed up | Not answered, right one in top 3 | Not answered, not in top 3 | Escalations passed up | Threshold / margin |
|---|---|---|---|---|---|---|---|
| target 0.5% wrong, pass-ups kept | not reachable | | | | | | |
| target 1% wrong, pass-ups kept | not reachable | | | | | | |
| target 2% wrong, pass-ups kept | not reachable | | | | | | |
| target 3% wrong, pass-ups kept | not reachable | | | | | | |
| target 5% wrong, pass-ups kept | not reachable | | | | | | |
| never pass up alone, target 1% | 31.6 | 0.7 | 0.0 | 53.7 | 13.9 | 0.0 | 0.73 / never |
| never pass up alone, target 3% | 40.8 | 2.7 | 0.0 | 43.5 | 12.9 | 0.0 | 0.58 / never |

## What to take to a confirmation run

Register both before looking at any confirmation score, with the shipped head as the control:

1. **A frozen encoder, no per-workflow weights: `intfloat/e5-base-v2` (mean pooling, `query: ` prefix) + the same logistic head + description/name rows + seeds x3.**
   Development numbers: best label first 86.8 / 73.8 (examples / large; shipped 85.3 / 67.7), wrong confident at the fixed right-outright shares 0.7 / 1.1 and 0.7 / 0.7 (shipped 1.6 / 2.6 and 1.7 / 2.7). At a 3% wrong budget with pass-ups handed on, it answers 71.8% / 49.3% right outright against the shipped head's 64.8% / 34.4%. Cost: a 438 MB shared encoder instead of 133 MB, and about 20 ms / 50 ms per request on CPU (10 / 1 threads) instead of 8 / 14 ms. The same combination with mpnet instead of e5 is better on the examples (88.4 first) and weaker on the large workflow (66.3).
2. **bge-base fine-tuned on the last two layers (or all layers if per-workflow size is not a concern).** Last two layers: best label first 88.3 / 73.1, wrong confident 1.8 / 2.0 and 1.4 / 1.7, per workflow and per version 56.7 MB, about 1 to 5 minutes training on the CPU, reproducible at a fixed seed. All layers: 90.6 / 74.5, 0.7 / 1.2 and 1.4 / 2.0, 438 MB, 2 to 8 minutes; at a 3% wrong budget it answers 77.7% / 48.0% right outright. Fine-tuning e5-base was not run; it is the obvious next arm.
3. **The policy, whichever head: the local model never passes up alone** (a request it thinks is "not here" is shown the top three and handed to the next link), and the threshold is chosen on held-out requests, not the cross-fit. This one change removes 2.0 (examples) and 6.1 (large) points of wrong confident answers at the shipped threshold, for no loss of right answers, and costs only the hand-on that the next link then does.

Expected on a confirmation set (an estimate, not a promise): the encoder gains hold in direction and shrink by about a third to a half because the sets are reused, so roughly +2 (examples) and +4 to +6 (large) points of best label first, and wrong confident answers at a fixed right-outright share about 30% to 50% lower than shipped; the in-sample operating rows above will read optimistic by a few points of wrong answers.

Not worth confirming: value masking, a trained projection, the nearest-neighbour or prototype heads, prepending the context name, a head per context (it loses escalation), per-context thresholds (no movement), merging same-name labels (nothing to fix), C other than 100.

## What none of the levers fixes

* **The large workflow's ceiling.** Even the best variants answer the right command first on at most 75% of its requests and have it in the top three on at most 90%. The misses are the context-entry and navigation commands (47% of misses) and the sibling commands of one context (35%): near-synonyms by design. Larger encoders and fine-tuning shave a few points; none removes them. Only more and sharper training rows (new generation, deliberately excluded here), a description written to separate the siblings, or the remote model fix them.
* **The wrong-confident floor.** Bare-value answers (a request whose best label is the bare-value label) make about 0.7% to 1.0% of requests wrong and confident at any threshold; refusing to commit a bare value alone would remove it.
* **Cross-fit calibration is optimistic.** On training rows the head looks 94% to 97% right; on real requests it is 68% (large) to 85% (examples) right first. No per-workflow or per-context calibration from training rows corrects that; only held-out requests do.
* **Escalation.** The head's best label for an escalation request is outside the mask nearly always (119 of 120, 49 of 49) but is exactly the expected ancestor on 37 of 49 large-workflow requests; a head per context cannot pass up at all unless it is given "not here" seeds, and then it over-passes.
* **Fine-tuning's price** is a per-workflow, per-version artifact (7 MB to 438 MB) and a training step on every retrain; the frozen variants stay a shared download.

## Reproduce

```
python scripts/experiments/embedding_commit_rule.py dump ...   # rows and cases, as in its docstring
python scripts/experiments/local_head_levers.py meta --workflow COPY --rows ROWS.json --out DIR/meta.json
python scripts/experiments/local_head_levers.py sweep --wf NAME=DIR ... --group examples=a,b --group large=c \
    --out OUT --variants all --shipped-op library=0.529,0.443 --shipped-op retail_workflow=0.5,0.0 --shipped-op large=0.591,0.622
python scripts/experiments/local_head_levers.py diagnose ... --pairs NAME=FILE     # Part 1
python scripts/experiments/local_head_levers.py latency ...                         # encoder request time
python scripts/experiments/local_head_levers.py tables ... --ops shipped,combo_e5base_data,finetune_base
```

Combinations are listed in a JSON file passed with `--combos` (`combo_*` and `finetune_e5base` style entries are `{name: spec}`).
The sweep takes about two hours on this machine's CPU (about 80 minutes of it is the bge-base fine-tuning and its repeat).
