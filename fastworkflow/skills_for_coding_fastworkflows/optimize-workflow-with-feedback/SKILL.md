---
name: optimize-workflow-with-feedback
description: >-
  Improve a fastWorkflow application using recorded traces, component-level human feedback,
  regression benchmarks and paired experiments. Use when asked to improve task completion,
  answer quality, latency, cost or repeated clarification, or to turn review comments into
  verified workflow changes. Diagnose a single failure with debug-workflow-conversations.
---

# Optimize workflows using evidence and feedback

Start with a concrete user problem and the recorded behavior that demonstrates it. Deliver a
trace-supported diagnosis, a scoped change, regression coverage and a comparison of observed
outcomes. If new execution is not authorized or available, deliver the diagnosis, change and
prepared comparison inputs, and clearly identify which claims remain unmeasured.

## Find the evidence in run_chatbot

Select the application workflow and use **benchmark → version → experiment → conversation →
turn** on the left. For ordinary chats use **ad-hoc conversations → UTC date → conversation →
turn**. The left tree ends at the turn; inspect its planning, execution steps and individual
calls on the right, whose breadcrumb keeps the whole drill-down path.

Record the evidence store identity/path, benchmark version/digest, experiment ID, task ID,
attempt, channel/conversation, turn key and relevant span IDs. Conversation IDs alone are not
unique across channels or recordings. A sealed experiment's turns are read from its archive file
(`archives/` beside the observability database), not from the live database.

Use [debug-workflow-conversations](../debug-workflow-conversations/SKILL.md) for trace reads and
stage-specific diagnosis. Before interpreting a missing call, inspect writer health and
redaction: redacted, truncated, dropped and absent evidence are different conditions.
Do not rerun a workflow merely to obtain an answer already present in its recording.

## Use human feedback precisely

The **Feedback** box on a turn, phase, step or span (or a whole task, as a task summary)
appends a timestamped comment, categorized with one of three categories and its two subcategories: **Observations / Analysis**
(`observation`, `analysis`), **Conclusions** (`what_went_right`, `what_went_wrong`) and
**Recommendations** (`what_to_do`, `what_not_to_do`). The comment itself is free text and no
heading in it is parsed. A coding agent records the same row with `POST /post_feedback` (taxonomy
at `GET /feedback_taxonomy`). Comment on the narrowest component with recorded spans that explains the
problem; if the component has no span anchor, comment on the turn. Useful comments state:

- What the user expected and what was observed.
- The concrete example or evidence supporting the difference.
- The desired behavior, including cases where the current behavior is already right.

For example: “The order lookup returned two candidates, but this step selected one without asking.
Ask which order the customer means before requesting a refund.” This identifies a decision and
an outcome to test. Treat any proposed fix as a hypothesis to verify against the trace.

Read the full comment history, including disagreements and later corrections. The UI shows
comments matching the selected component; `list_human_feedback(turn_key)` returns all anchors for
that turn with `comment`, `category`/`subcategory` and the frozen `anchors`, and
`list_task_feedback(experiment_id=, task_id=)` returns everything said about a whole task.
Comments written before the taxonomy read back with no category and are shown as unclassified;
do not guess which of the six they would have been. Preserve original comments; record an agent's
interpretation in analysis rather than inventing or overwriting a human judgment.

Comments are annotations on recorded evidence only: they never retrain anything, are never
replayed into a prompt and never change a score. They are recorded in the workflow's live
database; never modify an archived experiment to insert feedback.

## Diagnose and choose the smallest useful change

Walk from the earliest wrong decision to its effects. Compare the intended task, plan, actual
command, extracted parameters, command response and final answer. Typical evidence-to-change paths:

| Evidence | Candidate change |
|---|---|
| Wrong command, or the same which-command question asked again | The training report's `LOCAL INTENT MODEL` section lists the most confused command pairs: add seed utterances that tell a pair apart, or merge commands users cannot tell apart ([detect-duplicate-capabilities](../detect-duplicate-capabilities/SKILL.md)); check where the commands live ([design-context-models](../design-context-models/SKILL.md)); measure with [evaluate-intent-routing](../evaluate-intent-routing/SKILL.md) and retrain with [train-and-publish-models](../train-and-publish-models/SKILL.md) |
| Users keep confirming the same command for the same wording | Learning from use: confirmed picks and accepted reviews are recorded (`INTENT_LEARNING_RECORD`, on by default). `fastworkflow learn <workflow>` (`--dry-run` to only see the gate) refits the local head with them behind a cross-fit gate and can lower its commit threshold once enough confirmed rows exist; by default the refit also runs in the background every 25 new confirmed rows. Review or delete wrong confirmations with `fastworkflow learned list --confirmed`, `learned remove <workflow> <ids>` and `learned stats` |
| Missing handle or unnecessary request for an ID | Producer edges (`from_command_output`, [declare-parameter-producers](../declare-parameter-producers/SKILL.md)) and lookup visibility |
| Valid entity rejected or wrong fuzzy correction | `DatabaseValidator.fuzzy_match` candidates and thresholds in a model validator ([validate-command-parameters](../validate-command-parameters/SKILL.md), "Resolve a value against live keys") |
| Unusable values reach business logic, or a vague parameter question | A model validator on the command's `Input` that refuses with `fastworkflow.Refused(message, field=...)` ([validate-command-parameters](../validate-command-parameters/SKILL.md)); the parameter question then names that field with your message |
| The agent plans the same multi-command job differently or wrongly | A planning command whose steps code runs ([author-planning-commands](../author-planning-commands/SKILL.md)) |
| Correct tools but missing/incorrect final information | Response content, presentation requirements, output limits |
| Long latency/high cost | Identify costly LLM calls, repeated tools or waits before changing models or budgets |
| A rule is answered but asked again | Context/state retention and use of prior answers; distinguish invalid prior answers from forgotten valid ones |

A successful command is not proof of a satisfied task; check the actual application state or
other designated outcome evidence. A model's assertion of completion is not independent evidence.
Likewise an honest clarification can be correct behavior. Distinguish necessary clarification,
redundant confirmation, repeated questions after valid answers, and retries after invalid answers
(a parameter question stops after two replies in a row that fill or fix nothing).

## Preserve the regression and comparison

Use [create-workflow-benchmarks](../create-workflow-benchmarks/SKILL.md) to add the failing case
and nearby successful cases. A single prompt or short conversation is enough when it captures
the failure; use [build-task-benchmarks](../build-task-benchmarks/SKILL.md) for failures that require
longer sequences. Store fixture assumptions, expected outcomes and their evidence requirements
using the application's driver conventions. Remove secrets and replace transient customer data
with reproducible test fixtures. Verify fixture claims; do not invent backend facts.

Freeze the test input before comparing the existing and changed workflows. Keep benchmark pin,
fixtures, responder policy, repetitions, model routes, cache policy and budgets matched except
for the variable under study. If the corpus must change, create a new version and compare both
implementations on that version; do not call a task-set change an implementation improvement.
Keep tuning cases separate from held-out checks. Do not copy evaluation wording into training
seeds, and do not let it in through learning from use: the experiment harness forces `INTENT_LEARNING_REFIT=off` for both
arms of a measured comparison (outside it, set it by hand; a background refit publishes a new head that a running server
loads within seconds), and remove benchmark-run rows with `fastworkflow learned remove` before a
`fastworkflow learn`.

Create a new experiment identity for the candidate. Record the hypothesis/control/change and
baseline identity; use the project's pre-run setup review when it is part of the workflow. The
[runner reference](../create-workflow-benchmarks/runner-reference.md) covers registration,
optional review receipts and execution handoff. Registration and review do not launch anything.
Use existing execution authorization; when approval is required, present the exact run command,
model routes, estimated cost and limits after preparing the runnable comparison.

For interactive runs define a bounded responder/timeout policy and preserve every question and
answer. Incomplete attempts remain visible. Do not silently retry until success, omit failures,
or mix an unreported model/budget change into a workflow fix. Use focused local validation for
the changed feature before spending on measured execution.

## Compare and record what changed

Check evidence validity and the attempt's recorded runtime configuration
before reading score differences. An invalid experiment carries no defensible score. Missing
runtime snapshots limit claims about configuration; today's readiness response does not prove
what ran yesterday. Respect comparison refusals instead of bypassing them to produce a ranking.

Pair results by task ID and attempt where appropriate. Report the counts and denominators,
including missing/incomplete attempts. Keep these dimensions separate as relevant:

- Verified task outcomes and answer quality, including human judgments and unresolved disagreement.
- Command/routing failures and the first consequential error.
- Necessary and repeated clarification; the operator policy used.
- Latency (separating user wait), LLM calls/tokens, reported cost and output truncation.

For checklist grading of benchmark runs, read
[grade-benchmark-runs](../grade-benchmark-runs/SKILL.md). For keep-or-revert decisions over
repeated LLM runs, read [judge-changes-against-noise](../judge-changes-against-noise/SKILL.md).

Compare regressions as well as improvements. Runtime `completed` and command `success` are
not business-success grades. Missing cost/usage is unknown, not zero. Total LLM usage and cost with
`fastworkflow.observability.comparison.usage_rollup(spans)`. It folds re-emitted and wrapper
`fw.llm.call` records, so do not sum the rows or the parent roll-ups yourself. A completion equal to `max_tokens`
is a cutoff clue, not by itself proof that the answer is incomplete.

Write per-run conclusions in **experiment notes** (free text; they compute no score), a remark
about a pair of executions as feedback written from the comparison row (anchored to both), and a
whole-task conclusion as a task summary comment. Include evidence identifiers, the change, observed differences, counterexamples
and limitations. Recommend adoption, revision or another investigation based on those observations;
retain the original recordings.
The handoff should distinguish locally tested changes from measured improvements and identify any
pending execution or human rating explicitly.
