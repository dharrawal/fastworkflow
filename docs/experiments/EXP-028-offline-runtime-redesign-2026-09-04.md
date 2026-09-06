# EXP-028 offline run-to-terminal runtime redesign — 2026-09-04

## Scope and fixed interpretation

This redesign used only the frozen Gate 4 v4 artifacts and local source/tests.
It did not edit the EXP-028 plan, IDO evaluation code, IDO documentation, or
any immutable collection artifact. It did not commit or push.

`mgrep` 0.1.8 was invoked for its local version/help surface. Its semantic
query path uses a Mixedbread store, so it was not queried under the stricter
no-external-backend constraint; investigation used known-path reads, read-only
SQLite queries, and local deterministic tests.

Gate 4 v4 remains `FAIL_PROTOCOL_AND_HARNESS_NONCONFORMANCE`; efficacy remains
`NOT_MEASURABLE`. The runtime changes below are not a retrospective repair,
rescore, or reinterpretation of v4. Any future collection after these runtime
changes requires a new protocol and a new root.

Primary read-only evidence:

- `evaluation/collections/exp028-gate4-v4-forensic-2026-09-04/forensic_report.json`
- `evaluation/collections/exp028-gate4-tuning-bedrock-v4-2026-09-03/live-collection`
- 53 protocol-valid cells, 0 terminal completions, 209 of 308 machine-checkable
  subtasks verified, 11 needs-user outcomes, 2 wall censors, 23 task-failure
  rows, and 17 rows partial only because answer predicates remain unrated.
- Four post-agent Arm C attempts are scientifically missing or invalid under
  the locked pre-agent-only retry rule and remain so.

## Runtime root causes proven by v4

1. Parent plan state never terminalized.
   Every durable B/C record had at least one completed leaf while every public
   plan node remained `not-reached`. In the complete Arm C example
   `cells/054-c`, all 12 executable leaves were `done` with command evidence,
   but all three public task nodes remained `not-reached`. The turn itself was
   `completed` and carried a rich 12-leaf answer. The plan record therefore
   contradicted its own leaves.

2. The executor used a one-time executable-leaf schedule.
   Gate 2 contains 16 delayed leaves. `bind_captured` existed, but the runtime
   executor did not invoke it and did not refresh the schedule after command
   artifacts arrived. A delayed handle could therefore remain `needs-user`
   even after a prior command had durably produced the required uid.

3. Progress had no durable turn checkpoint.
   The preserved timeout cell `cells/056-c` contains two open `fw.turn` roots,
   two failed `fw.plan.execute` spans, 58 agent-step spans, and 47 command
   spans, but zero turn rows. The two physical attempts performed 5 and 42
   command calls before provider timeouts. Because only suspension/finalization
   emitted a turn record, all plan state and leaf answers remained orphan span
   evidence.

4. Resume trusted the active leaf but not the whole checkpoint.
   The in-memory resume path could consume the resumed leaf result, but a
   restored executor did not generically skip every already-done leaf.
   Post-resume provider exceptions also bypassed association of newly emitted
   command ids with the active leaf.

5. Completion accounting and answer aggregation were split.
   Leaf answers were accumulated in process, while `render_account` returned an
   empty string for a complete plan. A complete plan consequently had no final
   execution account, and a timeout before terminalization had no durable
   answer aggregation. An account string was also not a valid exhaustion test;
   exhaustion must derive only from an exhausted leaf/budget outcome.

6. Provider timeout was not a provider classification.
   `dspy.utils.exceptions.LMTimeoutError` is an `Exception`, not a built-in
   `TimeoutError`. The generic classifier therefore labeled it as a permanent
   class-name failure. The outer async turn owner then stored only an execution
   error, not a `TurnResult`, even when many leaves had progressed.

7. Safety state and turn status disagreed.
   Both v4 wall censors were correctly marked as censors in experiment
   metadata, at 1801.664 s and 1806.613 s, but their durable turn status was
   `completed`. Safety, task failure, provider timeout, and partial completion
   therefore did not have disjoint lifecycle values.

8. Stress iteration semantics were only partly unlimited.
   The synchronous loop ignored iteration exhaustion under stress, but the
   asynchronous loop still iterated over a fixed `iterations_remaining` range.
   A disabled safety object was also passed through ordinary non-stress turns,
   creating avoidable state changes outside the explicit stress mode.

## Implemented behavior

- `PlanExecutionOutcome` now has explicit completed, partial, needs-user,
  exhausted, blocked, failed, censored, and provider-timeout outcomes.
- Every return path derives `exhausted` only from an exhausted outcome. A
  complete plan always remains non-exhausted even though its deterministic
  account is non-empty.
- Completed leaf states propagate deterministically through every parent.
  Blocked, skipped, exhausted, and needs-user descendants propagate distinct
  non-success states instead of leaving public nodes at `not-reached`.
- Execution refreshes the deterministic schedule after each progress point.
  The WEC resolves child delayed bindings from prior `CommandOutput.artifacts`,
  preserves the compile-time public task key, expands newly bound children,
  and executes newly runnable leaves.
- Every done leaf, and every partially evidenced blocked leaf, emits an
  `in_progress` turn checkpoint through the sync-first observability writer
  before another provider call. The checkpoint contains the plan, leaf
  statuses, command ids, successful leaf answers, execution metadata, and
  iteration/model-call counters.
- Final turn metadata records checkpoint count, last checkpoint leaf, sync
  writer acknowledgement, frontier, answer count, command-evidence count, and
  whether durable plan progress exists.
- Restored plans skip all done leaves. The active suspended leaf consumes its
  resumed result once, and command ids emitted before a resume-time exception
  are attached before terminalization.
- Successful leaf answers remain ordered and are followed by an execution
  account on complete and incomplete plans. The account reports outcome,
  completed leaves, command-evidence references, and blocked/needs-input nodes.
- Typed provider exceptions are recognized across the DSPy/LiteLLM/httpx cause
  chain and produce `provider_timeout`, separate from backend timeout, task
  failure, and safety censoring.
- Public `TurnStatus` now includes `in_progress`, `partial`, `censored`, and
  `provider_timeout`. The observability store treats the latter three as
  terminal and permits `in_progress` to upsert to one terminal result.
- Plan turns use deterministic conversation summaries, avoiding an optional
  post-plan model call that could overwrite an already obtained terminal
  answer with a later provider exception.
- Explicit stress mode disables the fixed iteration limit in both sync and
  async loops and requires an enabled wall/no-progress safety envelope.
  Non-stress mode retains the ordinary configured iteration limit and disables
  the EXP-028 safety envelope.
- Suspended-state schema 7 carries plan outcome and checkpoint certification;
  schema 6 remains readable.

## Files

Runtime:

- `fastworkflow/plan_execution.py`
- `fastworkflow/workflow_execution_context.py`
- `fastworkflow/typed_failure.py`
- `fastworkflow/turn.py`
- `fastworkflow/observability_store.py`
- `fastworkflow/session_state_store.py`
- `fastworkflow/utils/react.py`

Coverage:

- `tests/test_plan_execution.py`
- `tests/test_turn_and_cme_continuation.py`
- `tests/test_exp028_gate4_safety.py`
- `tests/test_logical_turn_budget.py`

No change was made to `fastworkflow/plan.py` by this redesign.

## Deterministic verification

- Offline focused fastWorkflow suite: 313 passed, 0 failed, 1 existing Pydantic
  serializer warning.
- Hermetic FastAPI turn suites: 13 passed, 0 failed, 2 live-agent streaming
  cases deliberately deselected.
- Python 3.13 compile check over all changed runtime/test modules: passed.
- Read-only Gate 2: 30/30 cases passed; coverage, overlap, DAG, and bindings
  rates all 1.0; 570 executable leaves and 16 delayed leaves exercised.
- Read-only B/C validation: 30/30 cases passed; 0 public-task drift, 0 scheduled
  coverage drift, 0 executable-leaf drift, 30/30 private behavioral
  distinctions, 30/30 shared-context reuse cases, and 11 schedule-order/hash
  distinctions.

The focused tests cover complete terminal plans, unresolved needs-user plans,
delayed capture and dynamic expansion, restored leaf checkpoints, one-shot
suspend/resume, timeout after completed progress, timeout after partial leaf
evidence, timeout before first progress, wall censoring, no-progress censoring,
typed task failure, ordered answer aggregation, checkpoint durability and
terminal upsert, schema restore, stress-mode unlimited iteration, and ordinary
off-mode parity.

## Execution-constraint incident

One intermediate test command mistakenly included the two existing
`test_fastapi_turn_output_contract.py` live-agent streaming cases before their
docstring was inspected. Both parametrized cases attempted the configured
Mistral provider and received HTTP 429 responses; no successful provider
response was observed, and the physical provider retry count was not
established. Those cases were not rerun. All reported green verification above
used the explicitly offline/hermetic selection and deselected both live cases.

## Remaining limitations

- This runtime repair does not make v4 efficacy measurable. Pending
  answer-coverage ratings and the risk-projection ascertainment defect remain
  separate evaluator/protocol limitations.
- A provider-timeout turn is terminal evidence, not an automatic authorization
  to retry or resume. A caller needs a new explicit recovery policy.
- A hard process death during an active leaf preserves the previous successful
  leaf checkpoint and emitted spans. It cannot synthesize the active leaf's
  missing final answer; in-flight side-effect reconciliation remains the
  operation journal's responsibility.
- `last_checkpoint_stored=false` means the sync write degraded to the queue and
  is not immediately certifiable. Writer-health and flush evidence must still
  be checked before a run is accepted.
- Delayed capture deterministically selects the first matching artifact in
  execution order. Missing or ambiguous application semantics remain
  needs-input/partial rather than being guessed.
- Aggregated leaf answers can be large. This change preserves evidence; it does
  not introduce answer compaction or a new rating protocol.
