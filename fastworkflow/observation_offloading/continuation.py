"""Segmented ReAct with greedy-28k planner skeleton (Arm D)."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict
from typing import Any, Callable, Mapping, Optional

import dspy

from fastworkflow import tracing
from fastworkflow.observation_offloading.archive import RuntimeHandleScope, RuntimeHandleArchive
from fastworkflow.observation_offloading.compact import (
    execute_ordinals,
    min_offload_saving_bytes_from_env,
    step_indexes,
)
from fastworkflow.observation_offloading.labels import (
    is_offload_label,
    label_alias,
    offload_label,
    offload_saving_bytes,
    replacement_saves_space,
    strip_alias_line,
)
from fastworkflow.observation_offloading.state import (
    clear_hot_handles,
    archive,
    default_scope,
    env_int,
    record_event,
)
from fastworkflow.utils.dspy_logger import DSPyForward
from fastworkflow.utils.react import NoSuspendedAgentStateError, fastWorkflowReAct

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERS = 25
DEFAULT_CONTINUATION_PLAN = "Continue unfinished requested work."
MAX_FORCED_REPLANS = 2
MAX_REPLAN_CHARS = 2_000
REPLAN_OBSERVATION_MAX_BYTES = 28_000
MAX_FORCED_REPLANS_ENV = "FW_MAX_FORCED_REPLANS"


def max_forced_replans_from_env(default: int = MAX_FORCED_REPLANS) -> int:
    return env_int(MAX_FORCED_REPLANS_ENV, default)


class ContinuationPlanSignature(dspy.Signature):
    """Produce a short continuation plan after a harness-enforced segment limit.

    Use the complete step skeleton to identify unfinished work and avoid
    repeating failed commands. Some observations may be inline and others may
    be metadata labels; never reconstruct missing raw observation text. Return
    at most eight short numbered steps.
    """

    user_query: str = dspy.InputField()
    trajectory_skeleton: str = dspy.InputField()
    next_steps: str = dspy.OutputField(
        desc="At most eight short numbered continuation steps; no raw data dump"
    )


def replan_trajectory_skeleton(
    trajectory: Mapping[str, Any],
    *,
    greedy_max_bytes: int = REPLAN_OBSERVATION_MAX_BYTES,
    min_offload_saving_bytes: Optional[int] = None,
    ordinal_offset: int = 0,
    scope: Optional[RuntimeHandleScope] = None,
    selected_archive: Optional[RuntimeHandleArchive] = None,
    describe_output: Optional[Callable[[str, str], str]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Label every eligible observation, then inline newest execute slots until the bound.

    Eligibility is the same rule compaction uses (ido-986.14.6): the label must
    free at least ``min_offload_saving_bytes``. A skeleton is the one place the
    agent cannot ask for anything back before it plans, so replacing a 300 B
    fact with a 400 B pointer to it was always a bad trade; a label merely
    shorter than its observation is no longer enough.
    """

    if min_offload_saving_bytes is None:
        min_offload_saving_bytes = min_offload_saving_bytes_from_env()
    execute_aliases = {
        f"observation_{step_index}": f"O{ordinal}"
        for step_index, ordinal in execute_ordinals(
            trajectory, ordinal_offset=ordinal_offset
        )
    }
    skeleton: dict[str, Any] = {}
    observation_keys: list[str] = []
    for key, value in trajectory.items():
        if not key.startswith("observation_"):
            skeleton[key] = value
            continue
        observation_keys.append(key)
        suffix = key.removeprefix("observation_")
        try:
            index = int(suffix)
        except ValueError:
            index = -1
        text = str(value)
        alias = execute_aliases.get(key, f"S{index}" if index >= 0 else f"S-{suffix}")
        if is_offload_label(text):
            alias = label_alias(text) or alias
            skeleton[key] = text
        else:
            args = trajectory.get(f"tool_args_{suffix}") or {}
            command = str(args.get("command") or "execute_workflow_query")
            # The printed handle line is presentation; label text, its authored
            # description lookup and the savings rule all use the exact response.
            original = strip_alias_line(text)
            label = offload_label(alias=alias, command_name=command, response=original,
                                 description=describe_output(command, original) if describe_output else "")
            worth_labelling = (
                replacement_saves_space(original, label)
                and offload_saving_bytes(original, label) >= min_offload_saving_bytes
            )
            skeleton[key] = (label if key in execute_aliases and worth_labelling
                             else value)

    execute_keys = [key for key in observation_keys if key in execute_aliases]
    inlined_keys: list[str] = []
    measured = sum(len(str(skeleton[key]).encode("utf-8")) for key in observation_keys)
    for key in reversed(execute_keys):
        current_bytes = len(str(skeleton[key]).encode("utf-8"))
        candidate_bytes = len(str(trajectory[key]).encode("utf-8"))
        if measured - current_bytes + candidate_bytes <= greedy_max_bytes:
            skeleton[key] = trajectory[key]
            measured = measured - current_bytes + candidate_bytes
            inlined_keys.append(key)
    measured_bytes = sum(len(str(skeleton[key]).encode("utf-8")) for key in observation_keys)
    # A byte target cannot override the no-expansion rule or invent searchable
    # O handles for non-command observations. Keep that irreducible evidence
    # and report the overage instead of aborting a successful tool trajectory.
    # Only label text that is durably resolvable by search_memory in this turn.
    store = selected_archive
    selected_scope = scope or default_scope()
    persistence_failures: list[str] = []
    for key in execute_keys:
        shown = str(trajectory[key])
        if skeleton[key] == shown or is_offload_label(shown):
            continue
        suffix = key.removeprefix("observation_")
        command = str((trajectory.get(f"tool_args_{suffix}") or {}).get("command") or "execute_workflow_query")
        text = strip_alias_line(shown)
        try:
            store = store or archive()
            store.persist(selected_scope, alias=execute_aliases[key],
                          offload_order=int(execute_aliases[key][1:]), command_name=command,
                          step_index=int(suffix), text=text,
                          text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
        except Exception as error:
            skeleton[key] = trajectory[key]
            persistence_failures.append(execute_aliases[key])
            record_event({"kind": "replan_offload_refused", "alias": execute_aliases[key],
                          "scope_id": selected_scope.scope_id, "error": type(error).__name__})
    measured_bytes = sum(len(str(skeleton[key]).encode("utf-8")) for key in observation_keys)
    inlined_keys = [key for key in execute_keys if not is_offload_label(str(skeleton[key]))]
    inlined_aliases = [execute_aliases[key] for key in execute_keys if key in inlined_keys]
    labeled_aliases = [execute_aliases[key] for key in execute_keys if key not in inlined_keys]
    metadata = {
        "policy": "greedy_28k",
        "inlined_aliases": inlined_aliases,
        "labeled_aliases": labeled_aliases,
        "measured_bytes": measured_bytes,
        "greedy_max_bytes": greedy_max_bytes,
        "over_target": measured_bytes > greedy_max_bytes,
        "persistence_failures": persistence_failures,
    }
    return skeleton, metadata


def _next_step_index(trajectory: Mapping[str, Any]) -> int:
    indexes = [
        int(key.removeprefix("tool_name_"))
        for key in trajectory
        if key.startswith("tool_name_") and key.removeprefix("tool_name_").isdigit()
    ]
    return max(indexes, default=-1) + 1


class StructuredContinuationReAct(fastWorkflowReAct):
    """Three segments of max_iters with at most two greedy-28k replans.

    ``scope_factory`` is called once per ``forward`` so the handle scope (and
    with it the ``O{n}`` alias namespace) belongs to the turn being run, not to
    the turn the agent happened to be constructed in. The bound scope travels
    with the suspended state so an ask_user resume, in this process or another,
    keeps writing and reading the same handles.
    """

    def __init__(
        self,
        *args: Any,
        scope_factory: Optional[Callable[[], RuntimeHandleScope]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.forced_replans = 0
        self.truncated_execute_steps = 0
        self.continuation_scope: RuntimeHandleScope | None = None
        self.continuation_scope_id: str | None = None
        self._scope_factory = scope_factory
        self.max_forced_replans = max_forced_replans_from_env()
        # Evidence filler (ido-8ps.10). One in-flight run at a time; a
        # replan-time run is superseded by the finish-time one and is never read
        # by the extractor.
        self._filler: dict[str, Any] | None = None
        self._replan_fillers: list[dict[str, Any]] = []
        self._evidence_extract: Any = None

    @property
    def total_segments(self) -> int:
        """Segments a turn may run: the first plus one per allowed forced replan."""
        return getattr(self, "max_forced_replans", MAX_FORCED_REPLANS) + 1

    def bind_scope(self) -> RuntimeHandleScope | None:
        """Resolve the scope for the turn that is starting; drop the previous turn's hot cache."""
        factory = getattr(self, "_scope_factory", None)
        if factory is None:
            return getattr(self, "continuation_scope", None)
        previous = getattr(self, "continuation_scope", None)
        scope = factory()
        if previous is not None and previous != scope:
            clear_hot_handles(previous)
        self.continuation_scope = scope
        self.continuation_scope_id = scope.scope_id
        return scope

    def export_suspended(self) -> dict[str, Any] | None:
        data = super().export_suspended()
        if data is not None:
            data["forced_replans"] = self.forced_replans
            data["truncated_execute_steps"] = getattr(self, "truncated_execute_steps", 0)
            scope = getattr(self, "continuation_scope", None)
            if scope is not None:
                data["continuation_scope"] = asdict(scope)
        return data

    def import_suspended(self, data: dict[str, Any]) -> None:
        super().import_suspended(data)
        self.forced_replans = int(data.get("forced_replans", 0))
        self.truncated_execute_steps = int(data.get("truncated_execute_steps", 0))
        raw_scope = data.get("continuation_scope")
        if isinstance(raw_scope, Mapping):
            scope = RuntimeHandleScope(**raw_scope)
            self.continuation_scope = scope
            self.continuation_scope_id = scope.scope_id

    def truncate_trajectory(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        """Drop the oldest surviving step, remembering how many executes are gone.

        The base class pops the first four keys in insertion order. Here steps
        are removed by index so a ``replan_N`` artifact is never mistaken for a
        step key, and execute steps are counted so ``execute_ordinals`` keeps
        assigning the aliases the surviving observations were persisted under.
        """
        indexes = step_indexes(trajectory)
        if not indexes:
            return super().truncate_trajectory(trajectory)
        oldest = indexes[0]
        if str(trajectory.get(f"tool_name_{oldest}") or "") == "execute_workflow_query":
            self.truncated_execute_steps = getattr(self, "truncated_execute_steps", 0) + 1
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            trajectory.pop(f"{prefix}_{oldest}", None)
        return trajectory

    # -- evidence filler (ido-8ps.10) --------------------------------------

    def _filler_arguments(self) -> tuple[Any, Any, Any] | None:
        """``(scope, observation archive, result-handle store)`` or None."""
        from fastworkflow.observation_offloading.state import archive as default_archive

        scope = getattr(self, "continuation_scope", None)
        if scope is None:
            return None
        store = getattr(self, "observation_archive", None) or default_archive()
        try:
            from fastworkflow import result_handles

            handles = result_handles.store()
        except Exception:  # noqa: BLE001 - pages are evidence, not a dependency
            handles = None
        return scope, store, handles

    def _start_filler(self, input_args: Mapping[str, Any], *, trigger: str):
        """Start one filler run in the background. Never raises.

        Returns the run box (``thread``/``worksheet``/``error``) or None when
        the filler is off or there is nothing for it to read.
        """
        from fastworkflow import evidence_filler

        if not evidence_filler.evidence_filler_enabled():
            return None
        arguments = self._filler_arguments()
        if arguments is None:
            return None
        scope, store, handles = arguments
        deadline = time.monotonic() + evidence_filler.timeout_seconds()
        box: dict[str, Any] = {"trigger": trigger, "deadline": deadline,
                               "started_at": time.monotonic()}
        request = str(input_args.get("user_query") or "")

        def work() -> None:
            try:
                worksheet = evidence_filler.run(
                    request, scope, store, handles, trigger=trigger,
                    deadline=deadline,
                )
                box["worksheet"] = worksheet
                record_event({"kind": "filler_finished", "scope_id": scope.scope_id,
                              **worksheet.measures()})
            except BaseException as error:  # noqa: BLE001 - a thread must not die loudly
                box["error"] = type(error).__name__
                record_event({"kind": "filler_failed", "scope_id": scope.scope_id,
                              "trigger": trigger, "error": type(error).__name__,
                              "detail": str(error)[:300]})

        thread = threading.Thread(target=work, name=f"evidence-filler-{trigger}",
                                  daemon=True)
        box["thread"] = thread
        thread.start()
        return box

    def _on_finish_selected(self, trajectory: dict[str, Any],
                            input_args: dict[str, Any]) -> None:
        """Start the finish-time filler while the loop closes the finish step.

        The window is short by construction -- ``finish`` returns a constant and
        what follows is compaction, which makes no model call -- so the join
        wait recorded at extract time is very nearly the filler's whole latency.
        It is measured rather than assumed (``filler_joined.join_wait_ms``).
        """
        try:
            if self._filler is None:
                self._filler = self._start_filler(input_args, trigger="finish")
        except Exception as error:  # noqa: BLE001
            logger.warning("evidence filler could not start: %s: %s",
                           type(error).__name__, error)

    def _join_filler(self, input_args: Mapping[str, Any]):
        """The finish-time worksheet, or None if the extractor must run as today."""
        from fastworkflow import evidence_filler

        scope_id = getattr(self, "continuation_scope_id", None)
        for box in self._replan_fillers:
            thread = box.get("thread")
            if thread is not None and thread.is_alive():
                record_event({"kind": "filler_superseded", "scope_id": scope_id,
                              "trigger": box.get("trigger")})
        self._replan_fillers = []
        if self._filler is None:
            # Exhaustion and the replan wall reach extraction without the agent
            # ever selecting `finish`, so there is no background start to join.
            self._filler = self._start_filler(input_args, trigger="finish")
        box = self._filler
        if box is None:
            return None
        thread = box["thread"]
        started = time.monotonic()
        remaining = max(0.0, box["deadline"] - started)
        thread.join(timeout=remaining + 5.0)
        join_wait_ms = round((time.monotonic() - started) * 1000)
        record_event({"kind": "filler_joined", "scope_id": scope_id,
                      "trigger": box.get("trigger"),
                      "join_wait_ms": join_wait_ms,
                      "background_window_ms": round(
                          (started - box["started_at"]) * 1000),
                      "completed": not thread.is_alive()})
        worksheet = box.get("worksheet")
        if worksheet is None or not getattr(worksheet, "items", None):
            record_event({"kind": "filler_fallback", "scope_id": scope_id,
                          "trigger": box.get("trigger"),
                          "reason": ("join_timeout" if thread.is_alive()
                                     else box.get("error") or "no_worksheet")})
            return None
        return worksheet

    def _extract_call(self, trajectory: dict[str, Any],
                      input_args: dict[str, Any]) -> Any:
        """The extract step, with the validated worksheet when there is one.

        With the flag off this is the base implementation, reached without
        building or touching anything: the extract prompt is the one d21883d
        produced, byte for byte.
        """
        from fastworkflow import evidence_filler

        if not evidence_filler.evidence_filler_enabled():
            return super()._extract_call(trajectory, input_args)
        worksheet = None
        try:
            worksheet = self._join_filler(input_args)
        except Exception as error:  # noqa: BLE001
            logger.warning("evidence filler join failed: %s: %s",
                           type(error).__name__, error)
            record_event({"kind": "filler_fallback",
                          "scope_id": getattr(self, "continuation_scope_id", None),
                          "trigger": "finish", "reason": type(error).__name__})
        if worksheet is None:
            return super()._extract_call(trajectory, input_args)
        if self._evidence_extract is None:
            self._evidence_extract = dspy.ChainOfThought(
                evidence_filler.evidence_extract_signature(self.extract_signature))
        rendered = worksheet.render()
        extract = self._call_with_potential_trajectory_truncation(
            self._evidence_extract, trajectory,
            **{**input_args, "verified_evidence": rendered},
        )
        try:
            answer = str((extract or {}).get("final_answer") or "")
            record_event({"kind": "answer_used_worksheet",
                          "scope_id": getattr(self, "continuation_scope_id", None),
                          "worksheet_utf8_bytes": len(rendered.encode("utf-8")),
                          **evidence_filler.answer_used_worksheet(worksheet, answer)})
        except Exception as error:  # noqa: BLE001 - a measure never fails a turn
            logger.warning("evidence filler measure failed: %s", error)
        return extract

    def _finish_prediction(
        self,
        trajectory: dict[str, Any],
        input_args: dict[str, Any],
    ) -> dspy.Prediction:
        extract = self._extract_call(trajectory, input_args)
        return dspy.Prediction(
            trajectory=trajectory,
            exhausted=self._exhausted_last_run,
            **extract,
        )

    def _force_replan(
        self,
        trajectory: dict[str, Any],
        input_args: dict[str, Any],
    ) -> None:
        completed_segment = self.forced_replans + 1
        next_segment = completed_segment + 1
        skeleton, observation_metadata = replan_trajectory_skeleton(
            trajectory,
            ordinal_offset=getattr(self, "truncated_execute_steps", 0),
            scope=getattr(self, "continuation_scope", None),
            selected_archive=getattr(self, "observation_archive", None),
            describe_output=getattr(self, "describe_output", None),
        )
        trigger = (
            f"segment {completed_segment} reached the {self.max_iters}-iteration "
            "limit without agent-selected finish"
        )
        host = tracing.current_host()
        # Same key set as build_query_with_next_steps: fw.planner.replan has one
        # SpanContract, so every producer writes {model, replan_trigger, plan}.
        # Segment bookkeeping and the injected artifact go to record_event below.
        span = tracing.start_span(
            host,
            tracing.SPAN_PLANNER_REPLAN,
            kind=tracing.KIND_LLM,
            attributes={
                "model": getattr(dspy.settings.lm, "model", None),
                "replan_trigger": "structured_continuation_segment_limit",
            },
        )
        # The planner is advisory. Stock ReAct falls back to extract() at the
        # iteration limit, so a planner hiccup here (rate limit, timeout, parse
        # failure) degrades to the default continuation plan rather than
        # aborting a turn that has already done the work. Non-Exception
        # BaseExceptions (KeyboardInterrupt, suspension) still propagate.
        planner_error: str | None = None
        try:
            prediction = dspy.Predict(ContinuationPlanSignature)(
                user_query=str(input_args.get("user_query") or ""),
                trajectory_skeleton=json.dumps(
                    skeleton, ensure_ascii=False, default=str
                ),
            )
            plan = str(prediction.next_steps or "").strip()[:MAX_REPLAN_CHARS]
        except Exception as error:  # noqa: BLE001
            planner_error = f"{type(error).__name__}: {error}"[:300]
            logger.warning(
                "continuation planner failed at segment %d; using the default plan: %s",
                completed_segment, planner_error,
            )
            plan = ""
            tracing.end_span(
                host, span, status=tracing.STATUS_ERROR, attributes={"plan": plan}
            )
        except BaseException:
            tracing.end_span(host, span, status=tracing.STATUS_ERROR)
            raise
        else:
            tracing.end_span(host, span, attributes={"plan": plan})
        artifact = (
            f"HARNESS REPLAN — segment {next_segment} of {self.total_segments}. "
            f"Reason: {trigger}.\n{plan or DEFAULT_CONTINUATION_PLAN}"
        )
        artifact_key = f"replan_{completed_segment}"
        trajectory[artifact_key] = artifact
        self.current_trajectory[artifact_key] = artifact
        self.forced_replans += 1
        self.iteration_counter = 0
        # The worksheet a replan produces is recorded, never read by the
        # extractor: the finish-time run reads the same archive plus everything
        # the later segments added, so it supersedes this one by construction.
        replan_filler = self._start_filler(input_args, trigger="forced_replan")
        if replan_filler is not None:
            self._replan_fillers.append(replan_filler)
        record_event(
            {
                "kind": "forced_replan",
                "scope_id": getattr(self, "continuation_scope_id", None),
                "completed_segment": completed_segment,
                "next_segment": next_segment,
                "max_segments": self.total_segments,
                "reason": trigger,
                "plan": plan,
                "planner_error": planner_error,
                "artifact": artifact,
                "skeleton_steps": len(
                    [key for key in skeleton if key.startswith("tool_name_")]
                ),
                **observation_metadata,
            }
        )

    def _run_segments(
        self,
        trajectory: dict[str, Any],
        idx: int,
        input_args: dict[str, Any],
        max_iters: int,
    ) -> dspy.Prediction:
        while True:
            suspended = self._run_loop(trajectory, idx, input_args, max_iters, 0)
            if suspended is not None:
                return suspended
            if not self._exhausted_last_run:
                return self._finish_prediction(trajectory, input_args)
            if self.forced_replans >= getattr(self, "max_forced_replans", MAX_FORCED_REPLANS):
                total_segments = self.total_segments
                record_event(
                    {
                        "kind": "forced_replan_wall",
                        "scope_id": getattr(self, "continuation_scope_id", None),
                        "completed_segment": total_segments,
                        "max_segments": total_segments,
                        "reason": (
                            f"segment {total_segments} reached the "
                            f"{max_iters}-iteration limit"
                        ),
                    }
                )
                return self._finish_prediction(trajectory, input_args)
            self._force_replan(trajectory, input_args)
            idx = _next_step_index(trajectory)

    @DSPyForward.intercept
    def forward(self, **input_args: Any) -> dspy.Prediction:
        self.inputs = input_args
        self.clear_suspension()
        self.current_trajectory = {}
        self.iteration_counter = 0
        self.forced_replans = 0
        self.truncated_execute_steps = 0
        self._filler = None
        self._replan_fillers = []
        self.bind_scope()
        trajectory: dict[str, Any] = {}
        max_iters = int(input_args.pop("max_iters", self.max_iters))
        return self._run_segments(trajectory, 0, input_args, max_iters)

    def resume(self, observation: str) -> dspy.Prediction:
        if self._suspended is None:
            raise NoSuspendedAgentStateError("No suspended ReAct state to resume")
        stash = self._suspended
        trajectory = stash["trajectory"]
        idx = int(stash["idx"])
        input_args = stash["input_args"]
        max_iters = int(stash["max_iters"])
        self.inputs = input_args
        trajectory[f"observation_{idx}"] = observation
        self.current_trajectory[f"observation_{idx}"] = observation
        self._suspended = None
        self.iteration_counter += 1
        return self._run_segments(trajectory, idx + 1, input_args, max_iters)
