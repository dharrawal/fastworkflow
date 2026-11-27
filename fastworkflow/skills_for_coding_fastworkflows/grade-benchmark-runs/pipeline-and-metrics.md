# Grading every stage of the pipeline, answer quality, and metrics

Companion to [SKILL.md](SKILL.md). A checklist grades the final answer. A lost line is only useful
once it is pinned to the **first stage that went wrong**, because that is the stage a fix has to
change. This page covers each stage of a turn (what to read, the cause tag for a loss there), how
to judge answer quality apart from task completion, and the metrics to report.

Span and attribute names come from `fastworkflow/observability/tracing.py` (`SPAN_CONTRACTS`; span contract 56
at the time of writing). Contracts are versioned, so check them against the version that wrote the
store. Every attribute, the turn record (`turns.record_json`) and its artifacts are catalogued in
the [debug-workflow-conversations reference](../debug-workflow-conversations/reference.md#span-catalog).
This page names spans as span contract 62 and earlier wrote them (the benchmark stores graded so far).
Since contract 63 (bd fix-qjne.34) a turn's spans follow its stages (`fw.split`, `fw.resolve` >
`fw.resolve.part`, `fw.review`, `fw.run` > `fw.run.part` with `outcome`, `fw.answer`) and the names are
neutral: `fw.plan` (was `fw.planner.*`), `fw.phase` (was `fw.skill.phase`, `skill` is `command`), `fw.step`
(was `fw.agent.step` with its `fw.agent.tool_call`, and `fw.skill.answer`; `run_by` code/agent replaces
`by_code`), `fw.command` (was `fw.command.execute`), `fw.question` (was `fw.ask_user`), `fw.intent`,
`fw.parameters` (were `fw.nlu.*`); `fw.agent.execute` is gone (its facts are on `fw.run.part`). A dump of a
new store must keep model calls whose parent is a stage or split/review span, not only `fw.turn` or a
planner span. The framework's tools do not read a store written before contract 63; only the benchmark's own
dump tool maps the old names.

## Contents

- [The order to walk a lost line](#the-order-to-walk-a-lost-line)
- [Stage by stage](#stage-by-stage)
- [Answer quality is not task completion](#answer-quality-is-not-task-completion)
- [Metrics](#metrics)

## The order to walk a lost line

A turn first resolves every part of the request, then holds ONE review of the whole turn, then
runs:

```
split → intent (which command each part means) → parameters (its values) → review
   → plan (typed call or text plan) → phases → executor → finish check → answer
```

Start from the part that should have produced the line. Walk it left to right, and stop at the
first stage whose **output** was wrong given **its input**. That stage is the cause. Everything
after it is a consequence: mention it in the comment, do not tag it.

- **Judge each stage against what it was given, not against what the user wanted.** A planner that
  faithfully planned a wrongly split part did its job: the cause is `split`.
- **Recovery cancels a cause.** A stage that went wrong but was repaired later in the turn, with
  the line still ✅, is at most a waste note. The finish check is the exception: record its effect
  either way (`finish-note:*`).

## Stage by stage

The map below says where to look; the sections after it give the cause tags. Causes and their
stages are listed once, in the [vocabulary](reference.md#cause-and-tag-vocabulary).

| Stage | Read in the trace |
|---|---|
| Split | `subtask_split` on `fw.turn` |
| Intent | the part's bound command (below); a `plan_choice` question |
| Parameters | `subtask_bindings`; `skill_args`, `skill_notes`, `skill_fallback`; `fw.nlu.param_extraction`; a `planning_parameter` question |
| Review | `subtask_review` on `fw.turn`; a `review` question |
| Plan | `fw.planner.plan`: `plan_source`, `plan` |
| Phases | `fw.skill.phase`: `facts` / `facts_error`, `skipped`, `item_count` vs `items_total` |
| Executor | `fw.agent.step` (`by_code`), `fw.command.execute`, `code_stop`, `exhausted`, `fw_not_run` |
| Finish check | the `finish` step's `observation` |
| Answer | `final_answer`, `answer_by_code`, `uses_result`, `turns.answer`, `fw_sub_tasks` |

### Questions and who answers them

A question the turn puts out suspends it (`turns.status` `awaiting_user`); the reply continues
the **same** turn (same `turn_key`). Each question is an `ask_user` entry in the record's
`command_outputs` (the question in `command_parameters`, the reply in its response) whose
`artifacts.question_kind` says which: `plan_choice` (which command a part means), `planning_parameter`
(a missing value), `review`, `approval`, `phase_step` (a decision step of a running plan, no agent
model) or `agent` (the agent's own `ask_user`); and a `fw.ask_user` span (`agent_query`,
`user_responses`).

- **Attended** turns may ask. In an experiment the task's next scripted message is the reply (or
  the user simulator's, `experiments.user_simulator`). An attempt that ends on an unanswered
  question measured nothing (the derived outcome is `incomplete`): fix the script and rerun it.
- **Unattended** turns (`unattended` on `fw.turn`) put no question: the agent has no `ask_user`
  tool, no review is opened (`subtask_review` `not_reviewed`), and an ambiguous part goes to the
  agent with its candidates (with no agent model it comes back in `fw_not_run`).
- On the deterministic path (assistant mode, `/` messages) a question ends its turn instead, and
  the reply is the next turn.
- A reply answers its question and nothing else: extra requests in a reply are ignored by design.
- A question the prompt already answered is `ask`; name its `question_kind` in the comment. A
  wrong scripted reply is a fault of the run setup, not of the system: rerun.

### 1. Split

`subtask_split` on `fw.turn` (null when the turn was not split) is `{request, raw, merged,
changed_by, attempts}`: what the splitter saw, the model's parts, the parts the turn ran after the
code rules (`fastworkflow/split_merge.py`), and which rules changed the list. Judge the split from
`merged`; `raw` and `changed_by` tell you whether the model or a code rule made the mistake.

| Loss | Tag |
|---|---|
| A job asked for is in no part, or a qualifier was dropped ("first", "every", "exactly", "do not apply"; name it) | `[cause:split] [split:dropped]` |
| Two independent jobs joined, so one was half done | `[split:joined]` |
| A part asks for something the user never wrote | `[split:invented]` |
| A later part's "that one" left unresolved, or written in as the wrong record | `[split:referent]` |
| A summary or "header" sentence made into its own job | `[split:header]` |

An over-split that still got everything right is `[split:harmless]`, a note, not a cause. The
row's `split` field records the overall verdict.

### 2. Intent

One decision serves execution and planning commands alike (`fastworkflow/intent/intent_loop.py`): an
exact or fuzzy command name, then the local model, then the remote model when `JEV_API_KEY` is
set, else the which-command question. On the plan path it has no span of its own. Read what it
bound from `subtask_bindings[].means`, from `skill` on the part's `fw.planner.plan` (always null on
`fw.planner.replan`), or from the command code ran first for a bound execution command (`by_code`
step). A `plan_choice` question shows the candidates offered and the reply that picked one.

- **Wrong command or planning command for a faithful part:** `bad-match`; name both in the comment.
- **None chosen where one fits** (an ad hoc part, then a text plan that failed, or with no agent
  model a part in `fw_not_run` with `available_plans`): `bad-match`, noting "none".
- **None chosen where none was needed:** not a cause.

When a new planning command ships, tag every row that used it (`[skill:<name>]`), so you see both
its own tasks and where it was picked by mistake.

### 3. Parameters

Planning and execution commands share one parameter pipeline (`fastworkflow/parameters/`). Values
are drafted before the review; `subtask_bindings` records per bound part the values the review
`shown`, the arguments it `ran` with, and a `deviation`. A planning command's typed call carries
`skill_args`, `skill_notes` (arguments a guard or verify verdict replaced) and `skill_fallback`
(why it fell back to the text planner). An execution command's extraction is `fw.nlu.param_extraction`
(`missing_fields`, `invalid_fields`, `argument_refusal`).

| Loss | Tag |
|---|---|
| A planning command argument wrong (record kind, a flag, a name cut short), drafted or read from a reply | `[cause:other:skill-arg:<param>]` |
| A guard or verify verdict replaced a correct argument | `other:skill-arg:<param>`, citing the `skill_notes` entry |
| Right command; wrong record, uid or option (a look-alike name, a miscopied uid, an extraction that cut a value) | `wrong/argument`, with the `extraction` field filled when the extractor did it |

### 4. Review

`subtask_review` on `fw.turn`: `fast_path` (none needed: a single part that names its command,
that the local model is sure is an execution command, or whose command was chosen by answering),
`reviewed`, `declined`, `expired`, `not_reviewed` (no one could answer, or a plan pinned with
`use_plan`; the draft ran), `open` or `not_applicable`. The review shows what earlier stages bound; it is not a
cause itself. A wrong binding accepted in review is the cause of the stage that bound it. A
declined review runs nothing, so its lines are ❌ and the cause is the reply (rerun, as above) or
the stage whose wrong binding prompted the decline.

### 5. Plan

`fw.planner.plan` carries `plan_source`: `skill_call` (a planning command's typed call; code
expands its steps), `text` (a model-written plan) or `none`. A bound execution command has no
planner call: its plan is its one step, run by code. The planner can add steps around a planning
command (`steps_before` run in a phase of their own first, `code_stop` `off`; `steps_after` go
after the last phase's steps, or into a phase of their own before an answer-only last phase).

| Loss | Tag |
|---|---|
| The planning command's expansion lacks a step the request needs | `[cause:wrong/plan] [other:skill-gap]` |
| A text plan missing or misordering a step | `wrong/plan` |
| A typed call fell back to the text planner, which then failed | `[cause:wrong/plan] [other:skill-fallback]`, citing `skill_fallback` |

### 6. Phases

Each phase run is one `fw.skill.phase`: `phase`, `phase_index` of `phase_count`, `steps`,
`steps_used`, `step_budget` (`PHASE_MAX_STEPS`, 15), `exhausted`, `record`, `facts` or
`facts_error`, `skipped` (its `when` was false); for a fan-out item `item`, `item_index` of
`item_count`, `items_total` (more than `item_count` means the cap cut items: `max_items`, else
`FANOUT_MAX_ITEMS`, 10) and `stop_condition_met`; on the last phase `answer_only`,
`answer_by_code`, `uses_result`.

| Loss | Tag |
|---|---|
| A phase failed and stopped the job (`facts_error`): name the phase and the root cause | `[cause:other:phase-failed]` |
| Facts filled from the wrong row or record, or left empty | `[other:phase-facts]` |
| A needed phase skipped: its `when` misread the input | `other:phase-when` (or `other:skill-arg:<p>` when an argument drove it) |
| Fan-out cut by the cap | `[other:fan-out-cap]`, not budget |

A loss inside a phase whose own commands were wrong goes to the executor's cause, with the phase
named in `phase_cause`.

### 7. Executor

Each `fw.agent.step` has `step_index`, `thought`, `tool_name`, `tool_args`, `observation` and
`by_code` (code ran it, no model). Each `fw.command.execute` has `raw_command`, `parameters`,
`success`, `response_text`, and the `command_name` / `context` columns (what ran, where). A phase's
`steps_by_code` and `code_stop` say how far code got: `done`, `open step N`, `step N failed`,
`no path for step N`, `stopped` or `off`. Code pages a full-list step up to `FW_CODE_MAX_PAGES`
(default 20) pages, one step each; past that it stops (`open step N`) and the agent reads on.

With no agent model (`LLM_AGENT` unset) nothing takes over where code stops: the part comes back
as a stop in the artifact `fw_not_run` of the turn's last command output (`open_step`, `done`,
`facts`, `remaining`, `suggested_plan`, and `partial_read` `{pages, rows_read, total, rows}` for a
page-cap stop), and its answer text is a code-written statement of what ran.

| Loss | Tag |
|---|---|
| Plan right; the agent did something else, stopped early, skipped a step, or stopped paging where "every" was asked | `wrong/agent` |
| The right command errored and nothing recovered | `wrong/command-failed` |
| Out of steps doing real work (`exhausted`, or `failure_reason = max_iters_exhausted`) | `budget/real-work` |
| Out of steps through wandering (repeats, back-and-forth) | the cause of the waste, usually `wrong/agent` |
| Asked instead of acting (see [Questions](#questions-and-who-answers-them)), or an answer ending in a question | `ask` |
| No agent model: the part stopped at an open step code cannot run (`fw_not_run`, `open_step.kind`) | `other:not-run` |
| Code ran a step exactly as planned where an earlier agent improvised a fix | `[other:plan-followed]` |
| A loss inside a flat (single-phase) planning command run | `[other:flat-phase]`, saying whether code or the agent ran the failing step |

**Budgets.** Unphased: a part that reaches the executor's step cap is continued by a fresh
`fw.agent.execute`, at most `MAX_HANDOFFS` (2) times. Phased: no hand-off; a phase that reaches
its budget ends. Before tagging `budget/real-work`, classify where the steps went: navigation,
failed commands, repeats, productive reads. A bigger cap is a hypothesis, not a fix: in one
programme raising it used no extra steps and recovered none of the missing rows.

**What the agent saw is in the step spans.** Each step's `observation` (with any offloading
pointer) is the ground truth; prompt envelopes are capped. Before blaming the agent for missing a
value, check that its observation contained it.

### 8. Finish check

When the agent picks finish, code checks that every command step of the run's plan ran
(`fastworkflow/observation_offloading/finish_check.py`). If one did not, the `observation` of the
`fw.agent.step` whose `tool_name` is `finish` is a note beginning "Finish check: not every plan step
is done yet.", and more steps follow. One finish per agent run is reviewed
(`react.FINISH_REVIEWS_PER_RUN`).

Tag every row `[finish-note:none|helped|ignored|harmful]`: no note; the agent ran what the note
named and that fixed a line; it finished again without doing it; the note sent it into a wrong
context, a duplicate write or the step cap. A harmful note that is the first mistake is
`[cause:other:finish-note]`.

**"Executed" is not "satisfied."** The check confirms each planned command ran, not that its
output held what was asked or that the answer reported it. In one review, 24 of 33 turns it
certified were judged incomplete. It also exempts conditional, negated, answer and navigation-only
steps, so grade those lines from the trace yourself. False notes happen too (an equivalent
command counted as "not run"). Measure the check's detection offline (precision and recall of its
notes against graded outcomes) apart from its live effect (`helped` / `ignored` / `harmful`).

### 9. Answer

| Kind | How to tell | Typical loss |
|---|---|---|
| Model-written | the agent's `final_answer`; an answer-only phase with `answer_by_code` false; a `fw.skill.answer` span | fabrication, copying errors, a count given for a list, a value from the wrong field |
| Code-written | `answer_by_code` true; a planning command's `__result__` text (`uses_result`); a `fw_not_run` stop statement | literal and possibly long: grade content, not style. A loss means the step text lacked the fact: `[other:answer-by-code]` |
| Combined | a split turn's `turns.answer`, one `### N.` part per part; `fw_sub_tasks` lists each part's `text`, `answered` and offsets | anything dropped or invented when joining is its own loss |

A code-written note that a paged read stopped short ("N rows read of M") is honest: a count with
it is not a fabrication; a total stated without it over a partial read is. A loss here is a cause
only when every earlier stage was right and the answer lost or bent the information
(`[other:answer-phase]` for a poorly written answer-only phase).

### Evaluating a stage offline

At runtime no stage has a reference output; offline it can. Replay the cheap stages over stored
inputs before a live stage:

- **Splitter:** score each prompt's split by category (over-split, under-split, dropped, invented,
  name changed or cut, dependent part split off, background sentence made a job). Check names with
  word boundaries. Score the production path, code rules included: `subtask_split.merged`.
- **Intent:** score each faithfully split part's decision; see
  [evaluate-intent-routing](../evaluate-intent-routing/SKILL.md). A planning command's description
  edit moves both intent and splitting.
- **Planner:** score plans in code against reference plans: needed steps kept, "when asked" steps
  only when asked, request values used, inapplicable branches dropped.
- **Finish check / answer checks:** precision and recall against graded outcomes, with the bar
  declared in advance.

All of these are LLM-noisy at temperature 0; see
[judge-changes-against-noise](../judge-changes-against-noise/SKILL.md#6-offline-component-replays-useful-and-easy-to-fool-yourself-with).
A template written after reading the reference plans contaminates the comparison: rerun the
baseline under the same conditions.

## Answer quality is not task completion

A run can complete every step and still answer wrongly. Grade both, line by line:

| Dimension | Question | Rule |
|---|---|---|
| **Accuracy** | Is every claim in some output? | Every name, id, count and value traced to a span. Not found: 🚩. Misread: ⚠️ |
| **Coverage** | Does the answer name everything the line asks for? | "Every" lines need the full set or an honest "K of N". A first page passed off as all is ◐ |
| **Count vs list** | Asked for a list, it gave a count (or the reverse)? | ◐ |
| **Honest gaps** | Says plainly what is unavailable? | ✅ when an output backs the absence |

Coverage without accuracy is the trap. In one re-score, a "names present" metric reproduced at 57
of 59, but accuracy on the same answers was 0 of 59, and the ranking of the arms flipped at the
top. Also record:

- **Retrieval loss vs writer loss.** Was the missing value in the run's own outputs? If so, the
  answer step lost it (a count for the list, "see the output above", "unavailable" for a query
  that never ran); if not, an earlier stage did.
- **Repeated premise.** Repeating a false claim from the prompt is deference, not fabrication:
  `other:premise-repeated`. Flagging it is better than asked: credit it in `judgment_calls`.
  Inventing a value is 🚩.
- **Claims bounded by what ran.** "All 40 members checked" with one member-opening command in the
  trace is ⚠️ and `fabricated: yes`.
- **Cut by the harness.** An answer cut at the model's `max_tokens` is censored: exclude it from
  quality ratings and count it separately.

`turns.status`, `turns.success` and the runner's derived outcome measure execution. Report them,
never as quality.

## Metrics

Compute everything from the validated grades JSON and the store, never from graders' prose.

| Metric | How |
|---|---|
| **Lines per tier per run** | Sum of credit (✅ = 1, ◐ = ½) over the tier's rows, against the frozen line total |
| **Mean over runs** | Mean of the per-run totals, with the per-run values beside it (the spread is your noise) |
| **Pass count / pass@1** | Rows with `pass: true` ÷ all run rows of the tier. `experiment_scores()` computes pass@1 and pass^k from recorded attempt outcomes over the declared denominator, and gives a headline only for a `complete` experiment; when each run is its own experiment, compute from the grades |
| **pass^k** | With k = all runs: tasks whose every run passed ÷ tasks. For smaller k, the unbiased estimator below. Report it beside pass@1; the gap is unreliability |
| **Consistency** | Per task, same pass/fail in every run or not; list the tasks that flip (the summary row's `consistency` says why) |
| **Made-up count** | Rows with `fabricated: yes` |
| **Lines lost per cause / stage / plan source** | 1 (½ for ◐) per non-✅ line to the cause of its item, grouped by cause (with the distinct tasks hit: 8 tasks beat 1 task × 8), by stage, and by the turn's `plan_source`. The [losses script](reference.md#losses-by-cause-and-stage) prints all three |
| **Target-command rate** (single-command tiers) | Rows with `target_ran: yes` ÷ rows, kept apart from the score so a relaxed rule does not hide routing drift |
| **Planning command share** | Per new planning command: rows that used it, its tasks' score, and where it was picked by mistake |
| **Questions and reviews** | `ask_user` entries per job by `question_kind`; turns by `subtask_review` |
| **LLM calls / tokens / cost per job** | `fastworkflow.observability.comparison.usage_rollup(store.get_spans(turn_key))`: `calls`, `tokens`, `cost`, `cache`. It folds re-emitted records and wrapper calls, so never count `fw.llm.call` rows yourself. Missing usage is unknown (None), not zero |
| **Steps used / by code** | Sum `steps_used` and `steps_by_code` over the turn's `fw.skill.phase` spans (model steps = used − by code); unphased, count `fw.agent.step` spans and their `by_code` |
| **Where code stopped** | Distribution of `code_stop` values |
| **Time per job** | `completed_at − started_at − suspended_ms` from `turns`. Say whether stalled first jobs are in; concurrent lanes make times comparable only within one stage design |

Rules for these numbers:

- **Put intervals on pass rates** (for example a bootstrap over tasks), and do not rank
  configurations whose intervals overlap heavily.
- **pass^k unbiased:** for a task with n runs of which c passed, C(c, k) / C(n, k), averaged over
  tasks. The naive (c/n)^k overstates it (n = 5, c = 3, k = 2: 0.36 vs 0.30). Compare the decay
  ratio pass^k / pass@1 between stages.
- **Unscored is null, never 0 or false.** A metric with a zero denominator is null.
- **Two scorecards.** Process (routing, steps, code share, paging, exhaustion, questions, calls,
  cost, time) and outcome (lines, passes, made-up, premise handling), reported separately.
- **Never blend axes that trade off** (abstention vs fabrication, coverage vs accuracy). Enforcing
  "never ask" once moved every traded failure into fabrication and budget exhaustion. Fix floors
  for each axis before the treatment runs.
- **Measure the mechanism as well as the outcome.** For an intervention (a check, a note, a guard),
  report how often it fired, was heeded, and fired where it should. In one stage the finish check
  fired on 21 of 60 turns and was ignored on 11.
- **Validate an automatic stage classifier** on correct rows before relying on it.
- **Audit whoever answers questions like a judge.** Scripted replies or a user simulator are part
  of the environment. One simulator never supplied a correct novel fact, and its refusals licensed
  the agent to improvise. Count what its replies authorised. Prefer constant, scoped replies, or
  unattended runs.
- **Consistency is descriptive**, not correctness. With one run per task, report "insufficient
  repeats".

Report per tier per run, the mean, and the comparison with the previous stage, re-scored under the
current rules first; see [judge-changes-against-noise](../judge-changes-against-noise/SKILL.md).
