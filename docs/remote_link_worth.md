# The remote decision model behind the local model: what it is worth, and where its bars are

Measured 2026-11-17 and 2026-11-18. The remote link is the chain's second link (`MatcherEngine`, asked only when the
local head is not confident). This note records why its two bars, accept and decline, are 0.9 each
(`INTENT_REMOTE_ACCEPT_THRESHOLD`, `INTENT_REMOTE_DECLINE_THRESHOLD`; they were one 0.5 before).

## Method

* 1,102 requests, one remote call each, stored (`scripts/experiments/remote_link_worth.py ask`); no call is made again.
  Two families: the bundled examples (`library`, `retail_workflow`; 855 development and 310 confirmation requests,
  routing and escalation) and a large private workflow (343 development, 321 confirmation). Only the requests the local
  model does not answer with one command are asked; the rest are decided locally.
* Scoring is a simulated walk over the stored answers (`remote_link_worth.py analyse`, free): right (R), wrong and acted
  on (W, a bare value counts), asked with the right command among the candidates (Q+) or without it (Q-). The
  simulation is checked against the product (`verify`): `MatcherEngine` and `ChainEngine` at the shipped 0.9/0.9 give the
  same answer as the offline decision on all 1,102 stored answers (0 mismatches).
* The confirmation sets were held out from the choice of the pair, but they had been used before, in earlier
  experiments on the local model, so they are a weaker check than a set scored once.

## The comparison, pooled routing and escalation, confirmation sets (right / wrong / asked, %)

| Configuration | Examples | Large private workflow |
|---|---|---|
| local model alone | 42.6 / 1.6 / 55.8 | 24.0 / 5.9 / 70.1 |
| chain, bars 0.5 / 0.5 (the earlier default) | 94.2 / 3.2 / 2.6 | 77.6 / 13.4 / 9.0 |
| chain, bars 0.9 / 0.9 (now) | 85.2 / 1.9 / 12.9 | 55.1 / 6.5 / 38.3 |

On the large workflow, 0.5 turns 5.9% wrong answers into 13.4%: the remote model's wrong answers that were acted on
were 24 of 24 with the right command among the local candidates, so a question would have been right. At 0.9 the remote
model's own wrong answers fall from 24 to 2 on the confirmation set (37 to 5 on development), at the price of asking on
38.3% of the requests instead of 9.0%. On the examples the same move costs 9 points of right answers for 1.3 points
fewer wrong ones. The development grid is flat in the decline bar on the large workflow. The fixed rule picks 0.9/0.9 on the pooled
development sets (the examples alone would have picked 0.5/0.9; the large workflow alone 0.9/0.9), so one pair is shipped.

## The real walk at the product's thresholds

`scripts/experiments/real_walk_check.py` runs the product's `resolution._decide` on every request, with the remote
model's stored answer replayed for the call (no model is asked). Graded by `outcome` and `command` (a question with the
expected command among its candidates is a question; a bare value counts as the expected command only when it is
named):

| | Examples dev | Examples conf | Large dev | Large conf |
|---|---|---|---|---|
| local alone, right / wrong / asked | 39.1 / 0.2 / 60.7 | 28.4 / 0.6 / 71.0 | 22.2 / 0.0 / 77.8 | 18.7 / 0.9 / 80.4 |
| product chain at 0.5 / 0.5 | 78.2 / 1.8 / 20.0 | 80.0 / 2.3 / 17.7 | 73.5 / 12.5 / 14.0 | 76.9 / 9.3 / 13.7 |
| product chain at 0.9 / 0.9 | 73.8 / 0.5 / 25.7 | 71.0 / 1.0 / 28.1 | 55.4 / 1.7 / 42.9 | 52.6 / 1.6 / 45.8 |

The real walk does not reproduce the simulation's right rate on the examples (73 to 74% against 85 to 90%) because the
simulation does not apply the product's review of a local plan pick: a confident local answer that names a planning
command is never acted on unreviewed, so it is asked when the remote model does not confirm it at 0.9 (see below). On
the large workflow the right rates agree (52.6 against 55.1% on confirmation) and the product's wrong rate is lower than
the simulation's (1.6 against 6.5%), for the same reason: the plan picks that the simulation counts as wrong answers are
questions in the product.

## Why the deterministic path reviews a plan pick too

On the large workflow, every wrong confident local answer given by the global scope (asked first) was a planning
command. The plan resolution already refuses to act on such an answer without the remote model's confirmation
(`resolution._second_opinion`). The deterministic path (`CommandNamePrediction`, which since fix-o1lg S7 calls the same `intent_loop.resolve`: assistant mode, and messages that start
with `/`) did not: a confident local plan pick was run, mechanical planning commands in assistant mode and any planning
command to its first open step with no agent model. It now uses the same rule, through one predicate
(`intent_loop._second_opinion`): the remote model confirms the plan or the candidates are asked, the plan
first. An exact or fuzzy command NAME is unchanged and still runs.

## A remote outage

When the remote call fails (credits exhausted, a rejected key, a timeout), every caller keeps its fallback, as before:
the local answer, or the question. It is now visible: `remote_outage.report` logs one WARNING for the first failure
and then at most one per 5 minutes with the count since the last notice (error class and HTTP status code only), and the
`fw.nlu.intent` span carries `remote_failed` (`{error_class, status_code}`, span contract v53).

## Limits

The bars were measured on two families, one of them private and large; neither bar was tuned on the confirmation sets,
but the sets are not fresh. A run with a different remote model, or a different local head, moves the numbers.
