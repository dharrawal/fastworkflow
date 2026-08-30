"""Run a task set as one measured object.

An **experiment** is a labelled set of tasks, each run one or more times, whose
turns can be found, scored and compared as a unit. Before `fix-bn1` no such
object existed: `evidence_run()` minted a `run_id` that lived only inside a
bundle JSON, and no column joined a turn back to the run it belonged to.

This module is the producer. `ObservabilityStore` owns the records (schema and
write-once/terminal enforcement); this owns *running* a task set into them.

Design: `docs/experiment_container_design.md`, rulings `[XR1]`–`[XR20]`.

**What a task set is.** Data. A list of `ExperimentTask(task_id, messages)`, or
anything that yields them. Nothing here knows what tau2 is, what a corpus is, or
what "passed" means — a grader decides that and its verdict is recorded with the
name of who decided (`[XR13]`). The 15-task hard set at k=3 is the motivating
case, not a special case.

**Four things this gets right that are easy to get wrong:**

1. **Pre-registration.** The experiment row, with its hypothesis, is written
   before the first task runs. A prediction recorded afterwards is not a
   prediction, and `hypothesis` is write-once at the store so it cannot become
   one later.
2. **Independence.** One channel, one `Workflow`, one `WorkflowExecutionContext`
   per attempt (`[XR18]`). A shared channel would serialise the run and share
   one `ask_user` pending slot; a shared `workflow_id_str` would hand every
   attempt the same live `Workflow` object out of the registry.
3. **Determinism defeat.** Two caches would otherwise make repeated attempts
   identical and `pass^k` meaningless (`[XR16]`): DSPy's response cache and the
   NLU utterance cache. Both are turned off here, and what was done is recorded
   in the evidence segment.
4. **Never silently partial.** The store computes `complete`; this can only ask
   for it. An attempt that crashed halfway leaves `finished_at IS NULL`, which
   both blocks completion and is what `resume()` selects on.
"""

from __future__ import annotations

import contextlib
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import fastworkflow
from fastworkflow import evidence_run as evidence_run_module
from fastworkflow import observability_store, state_paths
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

# The env knobs `[XR16]` needs in the process before any attempt starts. Both
# are process-wide by nature: `fastworkflow.init()` installs one global env dict
# (`__init__.py:244-286`), so an arm that needs different values needs a
# different process.
LM_CACHE_VAR = "FW_LM_CACHE"
UTTERANCE_CACHE_SCOPE_VAR = "FW_UTTERANCE_CACHE_SCOPE"


class ExperimentAborted(RuntimeError):
    """The run could not be completed and the experiment was left `running`.

    Deliberately not the same thing as an invalid experiment. `running` means
    "resumable"; `invalid` means "do not score this, ever". A crash must not
    silently choose the second.
    """


@dataclass(frozen=True)
class ExperimentTask:
    """One task in a task set.

    `messages` are fed in order through `process_turn`. `startup_action` runs
    first when present, as its own logical turn — the shape
    `_create_user_runtime` uses.

    `task_id` must be stable across runs and across arms: it is the join key
    every score groups by, and it is never derived from the conversation topic
    (LLM-generated and channel-uniquified, so stable is the one thing it is not).
    """

    task_id: str
    messages: Sequence[str] = ()
    startup_action: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptRun:
    """What one attempt produced, handed to the grader."""

    task: ExperimentTask
    attempt: int
    channel_id: str
    conversation_id: Optional[int]
    turn_outputs: list = field(default_factory=list)
    error: Optional[BaseException] = None

    @property
    def turn_keys(self) -> list[str]:
        return [out.turn_key for out in self.turn_outputs if out is not None]

    @property
    def awaiting_user(self) -> bool:
        """Whether the attempt ended suspended on an unanswered question.

        Such an attempt is `incomplete`, not a fail: nobody answered it, so it
        never measured anything. `cancel_pending()` cannot be the remedy — it
        clears in-memory suspension and emits no turn record at all, so the
        stored row would stay `awaiting_user` forever.
        """
        if not self.turn_outputs:
            return False
        last = self.turn_outputs[-1]
        return getattr(getattr(last, "status", None), "value", None) == "awaiting_user"


# A grader receives a finished AttemptRun and returns
# (outcome, outcome_source, reward, detail). Returning None defers to the
# built-in fallback below.
Grader = Callable[[AttemptRun], Optional[tuple]]


def derived_outcome(run: AttemptRun) -> tuple:
    """The fallback verdict, named for what it actually measures (`[XR13]`).

    **This is not a pass/fail judgement and must never be presented as one.**
    It reports whether any command in the attempt returned a failure code.

    `TurnOutput.success` is `all(command_output.success ...)` over per-command
    flags that default to True, and `all([])` is True. For a workflow whose
    commands never set `success=False`, this is true whenever the agent did not
    exhaust its iterations — regardless of whether the task was accomplished.
    Worse, the one thing that *does* flip it is a command that failed and the
    agent then recovered from, so an ordinarily successful trajectory that
    probed a wrong id and recovered reads as a failure.

    It is here because a harness with no grader needs *something*, and because
    recording `outcome_source='derived'` makes that choice visible in the
    record instead of letting a fallback masquerade as a measurement. Supply a
    real grader for any number you intend to report.
    """
    if run.error is not None:
        return ("error", "derived", None, {"error": repr(run.error)})
    if run.awaiting_user:
        return ("incomplete", "derived", None, {"reason": "awaiting_user"})
    if not run.turn_outputs:
        return ("incomplete", "derived", None, {"reason": "no turns"})
    clean = all(
        getattr(getattr(out, "status", None), "value", None) == "completed"
        and out.success
        for out in run.turn_outputs
    )
    return (
        "pass" if clean else "fail",
        "derived",
        None,
        {"predicate": "no_command_reported_failure"},
    )


def channel_for(experiment_id: str, task_id: str, attempt: int) -> str:
    """`[XR18]`: one channel per attempt, deterministic so a resume finds it."""
    return f"exp:{experiment_id}:{task_id}:{attempt}"


class ExperimentHarness:
    """Runs a task set as one experiment.

    Usage::

        harness = ExperimentHarness(workflow_folderpath, label="tau2 hard set",
                                    hypothesis="insight #7 lifts pass^3")
        result = harness.run(tasks, attempts=3, grader=my_grader)

    One harness, one experiment. Reuse across experiments is not supported and
    is not wanted: the pre-registration is per-run.
    """

    def __init__(
        self,
        workflow_folderpath: str,
        *,
        label: str,
        hypothesis: Optional[str] = None,
        arm: Optional[str] = None,
        baseline_experiment_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        run_as_agent: bool = True,
        max_workers: int = 4,
        archive_dir: Optional[str] = None,
        defeat_caches: bool = True,
        install_memory_policy: bool = False,
    ) -> None:
        self.workflow_folderpath = workflow_folderpath
        self.label = label
        self.hypothesis = hypothesis
        self.arm = arm
        self.baseline_experiment_id = baseline_experiment_id
        self.experiment_id = experiment_id or f"exp-{uuid.uuid4().hex}"
        self.run_as_agent = run_as_agent
        self.max_workers = max(1, int(max_workers))
        self.archive_dir = archive_dir
        self.defeat_caches = defeat_caches
        self.install_memory_policy = install_memory_policy
        self._db_path = state_paths.observability_db(workflow_folderpath)
        self._store = observability_store.ObservabilityStore(self._db_path)
        self._lock = threading.Lock()
        self._sink: Optional[observability_store.SQLiteTraceSink] = None

    # -- the two things that must happen on the main thread ---------------

    def _prepare_process(self) -> dict[str, Any]:
        """Everything that must be done before any attempt thread is released.

        Returns what was done, for the evidence segment: a run whose cache
        posture is not recorded cannot be told apart later from one that forgot.
        """
        posture: dict[str, Any] = {"defeat_caches": self.defeat_caches}
        # One process-wide sink, taken BEFORE `evidence_run()` opens, so the
        # gate's `in_process` verdict rests on this process's live counters
        # rather than on a persisted row it had to wait for. Every attempt
        # shares it: one writer thread per DB is the [R7] contract, and N sinks
        # on one file would be N writer threads racing.
        self._sink = observability_store.get_observability_sink(
            self.workflow_folderpath
        )
        posture["sink"] = "in-process" if self._sink is not None else "none"
        if self.defeat_caches:
            # `[XR16]`, both contaminants. Set through `fastworkflow`'s env dict
            # rather than os.environ so `get_env_var` sees them; the process is
            # the experiment, so process-wide is the right scope.
            #
            # MERGED, never `fastworkflow.init({...})`. `init` REPLACES the
            # process env dict wholesale (`__init__.py:253`), so calling it with
            # two keys would delete `LLM_AGENT`, every `LITELLM_API_KEY_*`, and
            # every threshold the run needs -- and the failure would surface as
            # "DSPy Language Model not provided" from somewhere unrelated, long
            # after the harness had already been blamed and cleared.
            fastworkflow._env_vars[LM_CACHE_VAR] = "0"
            fastworkflow._env_vars[UTTERANCE_CACHE_SCOPE_VAR] = "workflow"
            posture[LM_CACHE_VAR] = "0"
            posture[UTTERANCE_CACHE_SCOPE_VAR] = "workflow"

        # The DSPy memory policy. A long agentic sweep in one process fills
        # DSPy's retention structures unbounded, and unlike the
        # `fastworkflow.chat_session` global this is a place where a bare-WEC
        # harness does NOT match FastAPI: `install_policy` is called only from
        # the server entrypoint.
        #
        # OPT-IN, not default, and this is deliberate. `install_policy` claims
        # DSPy's config-owner THREAD process-wide and has no uninstall, so a
        # library that called it on construction would poison every process that
        # ever built a harness -- including a test process, where the next
        # FastAPI app's lifespan then fails its own `claim_async_owner()`.
        # Observed exactly that way. An entry point running a real experiment
        # should pass `install_memory_policy=True` (or call `install_policy()`
        # itself before constructing the harness); an embedded or test caller
        # should not.
        #
        # Either way the answer is RECORDED, so a sweep that ran without the
        # policy is visible in its own evidence rather than silently different
        # from one that did.
        if self.install_memory_policy:
            try:
                from fastworkflow.run_fastapi_mcp import server_memory

                server_memory.install_policy()
                posture["dspy_memory_policy"] = "installed"
            except Exception as exc:  # pragma: no cover - optional extra
                posture["dspy_memory_policy"] = f"unavailable: {type(exc).__name__}"
                logger.warning(
                    f"Could not install the DSPy memory policy ({exc!r}); a long "
                    "experiment may accumulate DSPy history in this process"
                )
        else:
            posture["dspy_memory_policy"] = "not installed (caller opted out)"
        return posture

    def _warm_routers(self) -> None:
        """Build the NLU pipelines once, before N threads each build a copy.

        `CommandRouter._instances_cache` and `ModelPipeline._instances_cache`
        are unlocked dicts whose `__new__` publishes an UNinitialised instance
        and whose `__init__` sets `_initialised` last, so N cold attempts
        released together each run the full constructor: N concurrent
        TinyBERT+DistilBERT loads onto one shared object. An RSS spike, not
        corruption — but on a box that OOM-kills at 31 GB the distinction stops
        mattering.

        Constructing `CommandRouter` per `___command_info` subdirectory is what
        actually fills those caches; `RoutingRegistry.get_definition` fills a
        different one (`_definitions`) and touches neither. The loop mirrors
        `chat_session.py:217-241`, which is the only other place this is done.
        Also warms the CME workflow's routing definition, which the turn path
        builds and would otherwise have N threads racing to write.
        """
        try:
            from pathlib import Path

            from fastworkflow.command_routing import RoutingRegistry
            from fastworkflow.model_pipeline_training import CommandRouter

            RoutingRegistry.get_definition(self.workflow_folderpath)
            with contextlib.suppress(Exception):
                RoutingRegistry.get_definition(
                    fastworkflow.get_internal_workflow_path(
                        "command_metadata_extraction"
                    )
                )
            command_info_root = Path(self.workflow_folderpath) / "___command_info"
            if command_info_root.is_dir():
                for subdir in command_info_root.iterdir():
                    if subdir.is_dir():
                        with contextlib.suppress(Exception):
                            CommandRouter(str(subdir))
                # The global-context artefacts live in a pseudo-folder named '*'
                # in some workflows.
                with contextlib.suppress(Exception):
                    CommandRouter(str(command_info_root / "*"))
        except Exception as exc:
            logger.debug(f"Router warm-up skipped: {exc!r}")

    # -- running ----------------------------------------------------------

    def run(
        self,
        tasks: Iterable[ExperimentTask],
        *,
        attempts: int = 1,
        grader: Optional[Grader] = None,
    ) -> dict[str, Any]:
        """Run every task `attempts` times as one experiment.

        The experiment row is created FIRST, with its hypothesis, before any
        task executes. Everything else runs inside `evidence_run()`, so the run
        gets zero-drop assertion, prune suppression, archival and provenance —
        and an invalid verdict from it makes the experiment invalid.
        """
        task_list = list(tasks)
        if not task_list:
            raise ValueError("a task set with no tasks is not an experiment")
        attempts = int(attempts)
        if attempts <= 0:
            raise ValueError("attempts must be positive")
        seen = {t.task_id for t in task_list}
        if len(seen) != len(task_list):
            raise ValueError(
                "task_ids must be unique within a task set: they are the join "
                "key every score groups by"
            )

        self._store.create_experiment(
            self.experiment_id,
            self.label,
            declared_tasks=len(task_list),
            declared_attempts=attempts,
            hypothesis=self.hypothesis,
            arm=self.arm,
            baseline_experiment_id=self.baseline_experiment_id,
            workflow_name=os.path.basename(
                self.workflow_folderpath.rstrip("/\\")
            ),
        )
        pairs = [(task, n) for task in task_list for n in range(1, attempts + 1)]
        return self._execute(pairs, grader, seq=1)

    def resume(
        self,
        tasks: Iterable[ExperimentTask],
        *,
        grader: Optional[Grader] = None,
    ) -> dict[str, Any]:
        """Re-run the attempts of a crashed run that never finished.

        Selects on `finished_at IS NULL` — the completion marker — not on "has
        no terminal turn". An attempt that crashed after turn 3 of 10 has turns
        and would be skipped by the second test while remaining unfinished.

        Each selected attempt is cleared by `restart_attempt`, which deletes its
        conversations and turns in one transaction and bumps `restarts`. That
        deletion is deliberate: `idx_conv_experiment_attempt` is UNIQUE, so a
        second conversation under the same three labels is refused outright, and
        an abandoned partial trajectory is evidence of nothing.
        """
        experiment = self._store.get_experiment(self.experiment_id)
        if experiment is None:
            raise observability_store.ExperimentNotFound(self.experiment_id)
        if experiment["status"] != "running":
            raise ExperimentAborted(
                f"experiment {self.experiment_id} is {experiment['status']!r}; "
                "only a running experiment is resumable"
            )
        by_id = {t.task_id: t for t in tasks}
        pending = [
            row
            for row in self._store.experiment_attempt_rows(self.experiment_id)
            if row["finished_at"] is None
        ]
        declared = int(experiment["declared_attempts"])
        started = {
            (row["task_id"], row["attempt"])
            for row in self._store.experiment_attempt_rows(self.experiment_id)
        }
        pairs: list[tuple] = []
        for row in pending:
            task = by_id.get(row["task_id"])
            if task is None:
                logger.warning(
                    f"Attempt {row['task_id']}#{row['attempt']} is unfinished but "
                    "its task is not in the supplied task set; it cannot be "
                    "resumed and the experiment cannot become complete"
                )
                continue
            self._store.restart_attempt(
                self.experiment_id, row["task_id"], row["attempt"]
            )
            pairs.append((task, int(row["attempt"])))
        # Attempts that crashed before start_attempt ever ran leave no row at
        # all; they are found by absence, not by an open marker.
        for task in by_id.values():
            for n in range(1, declared + 1):
                if (task.task_id, n) not in started:
                    pairs.append((task, n))
        seq = len(experiment.get("evidence_runs") or []) + 1
        return self._execute(pairs, grader, seq=seq)

    def _execute(
        self, pairs: Sequence[tuple], grader: Optional[Grader], seq: int
    ) -> dict[str, Any]:
        posture = self._prepare_process()
        self._warm_routers()
        grade = grader or derived_outcome
        runs: list[AttemptRun] = []
        body_error: Optional[BaseException] = None

        with evidence_run_module.evidence_run(
            self.workflow_folderpath, archive_dir=self.archive_dir
        ) as evidence:
            try:
                if pairs:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                        # submit + as_completed, NOT pool.map: map's iterator
                        # raises on the first failed future, so an attempt that
                        # raised out of `_run_attempt` would discard every other
                        # attempt's result and leave `runs` empty -- and the
                        # evidence segment would then record `attempts_run: 0`
                        # for a run in which attempts demonstrably ran and wrote
                        # rows. A record whose whole purpose is honesty must not
                        # under-report itself on the crash path.
                        futures = [
                            pool.submit(self._run_attempt, task, n, grade)
                            for task, n in pairs
                        ]
                        for future in as_completed(futures):
                            try:
                                runs.append(future.result())
                            except BaseException as exc:  # noqa: BLE001
                                if body_error is None:
                                    body_error = exc
            except BaseException as exc:  # noqa: BLE001 - held, not re-raised here
                # Held rather than re-raised through the context manager, so the
                # evidence segment below is still recorded. `evidence_run` itself
                # archives on a raising body for the same reason: a run that
                # crashed is exactly when its evidence is most worth keeping, and
                # an experiment whose crash left no segment cannot be told apart
                # later from one that was never started.
                body_error = exc

        record = evidence.as_record()
        record["experiment"] = {
            "experiment_id": self.experiment_id,
            "cache_posture": posture,
            "attempts_run": len(runs),
            "aborted": None if body_error is None else repr(body_error),
        }
        self._store.record_evidence_segment(
            self.experiment_id, seq, evidence.run_id, record
        )
        if body_error is not None:
            # Left `running`, not `invalid`: the run may be resumable, and only
            # a completed check may declare a verdict.
            raise ExperimentAborted(
                f"experiment {self.experiment_id} aborted: {body_error!r}"
            ) from body_error

        status = self._store.complete_experiment(self.experiment_id)
        return {
            "experiment_id": self.experiment_id,
            "status": status,
            "evidence_run_id": evidence.run_id,
            "evidence_valid": evidence.valid,
            "evidence_problems": list(evidence.problems()),
            "attempts": len(runs),
            "scores": self._store.experiment_scores(self.experiment_id),
        }

    # -- one attempt ------------------------------------------------------

    def _run_attempt(
        self, task: ExperimentTask, attempt: int, grade: Grader
    ) -> AttemptRun:
        """Drive one attempt on its own channel, workflow and context.

        Every failure mode here is caught and recorded as an attempt outcome
        rather than raised: one task blowing up must not abort the other 44, and
        an attempt that errored is a data point, not a missing row.
        """
        channel_id = channel_for(self.experiment_id, task.task_id, attempt)
        run = AttemptRun(
            task=task, attempt=attempt, channel_id=channel_id, conversation_id=None
        )
        self._store.start_attempt(
            self.experiment_id, task.task_id, attempt, channel_id
        )
        ctx: Optional[WorkflowExecutionContext] = None
        try:
            ctx = WorkflowExecutionContext(
                run_as_agent=self.run_as_agent,
                session_key=channel_id,
                trace_sink=self._sink,
            )
            ctx.bind_observability_identity(
                channel_id=channel_id,
                experiment_id=self.experiment_id,
                task_id=task.task_id,
                attempt=attempt,
            )
            # A unique workflow_id_str per attempt is not optional: a colliding
            # id returns the registry's existing live Workflow object and
            # overwrites its context, so every attempt would share one
            # application state.
            workflow = fastworkflow.Workflow.create(
                self.workflow_folderpath, workflow_id_str=channel_id
            )
            ctx.bind_app_workflow(workflow)

            if task.startup_action is not None:
                run.turn_outputs.append(ctx.process_action_turn(task.startup_action))
            for message in task.messages:
                run.turn_outputs.append(ctx.process_turn(message))
            run.conversation_id = ctx.observability_conversation_id
        except BaseException as exc:  # noqa: BLE001 - one attempt, not the run
            run.error = exc
            logger.warning(
                f"Attempt {task.task_id}#{attempt} raised {exc!r}; recorded as "
                "an errored attempt"
            )
        finally:
            if ctx is not None:
                if run.conversation_id is None:
                    run.conversation_id = ctx.observability_conversation_id
                with contextlib.suppress(Exception):
                    ctx.close()

        # A grader is caller code and may raise. One task's grader blowing up
        # must not abort the other 44, for the same reason the attempt body's
        # exceptions are caught: an attempt that could not be judged is a data
        # point, not a missing row. The failure is recorded as its own outcome
        # source so it is never mistaken for a real verdict.
        try:
            verdict = grade(run) or derived_outcome(run)
            outcome, source, reward, detail = verdict
        except BaseException as exc:  # noqa: BLE001
            logger.warning(
                f"Grader raised on {task.task_id}#{attempt}: {exc!r}; the attempt "
                "is recorded as incomplete, which blocks a headline score"
            )
            # `incomplete`, not `error`. `error` means the ATTEMPT failed,
            # which is a measurement of the agent and scores as a non-pass. A
            # grader that raised measured nothing about the agent at all, so
            # scoring it as a non-pass would silently attribute the judge's bug
            # to the thing being judged. `incomplete` blocks completion, which
            # forces the grader to be fixed and the attempt re-run.
            outcome, source, reward = "incomplete", "grader_error", None
            detail = {"grader_error": repr(exc)}
        with self._lock:
            self._store.finish_attempt(
                self.experiment_id,
                task.task_id,
                attempt,
                outcome=outcome,
                outcome_source=source,
                reward=reward,
                detail=detail,
                conversation_id=run.conversation_id,
            )
        return run
