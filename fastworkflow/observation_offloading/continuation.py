"""Segmented ReAct with greedy-28k planner skeleton (Arm D)."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

import dspy

from fastworkflow import tracing
from fastworkflow.observation_offloading.compact import execute_ordinals
from fastworkflow.observation_offloading.labels import is_offload_label
from fastworkflow.observation_offloading.state import record_event
from fastworkflow.utils.dspy_logger import DSPyForward
from fastworkflow.utils.react import NoSuspendedAgentStateError, fastWorkflowReAct

DEFAULT_MAX_ITERS = 25
MAX_FORCED_REPLANS = 2
TOTAL_SEGMENTS = MAX_FORCED_REPLANS + 1
MAX_REPLAN_CHARS = 2_000
REPLAN_OBSERVATION_MAX_BYTES = 28_000
MAX_FORCED_REPLANS_ENV = "FW_MAX_FORCED_REPLANS"


def max_forced_replans_from_env(default: int = MAX_FORCED_REPLANS) -> int:
    raw = os.environ.get(MAX_FORCED_REPLANS_ENV, "").strip()
    return default if not raw else int(raw)


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


def _observation_label(
    trajectory: Mapping[str, Any],
    key: str,
    value: Any,
    alias: str,
) -> str:
    suffix = key.removeprefix("observation_")
    text = str(value)
    command = str(trajectory.get(f"tool_name_{suffix}") or "unknown")
    return (
        f"{alias} — {command}; observation label only "
        f"({len(text)} chars, {len(text.encode('utf-8'))} UTF-8 bytes; "
        f"sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()})"
    )


def replan_trajectory_skeleton(
    trajectory: Mapping[str, Any],
    *,
    greedy_max_bytes: int = REPLAN_OBSERVATION_MAX_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Label every observation, then inline newest execute slots until the bound."""

    execute_aliases = {
        f"observation_{step_index}": f"O{ordinal}"
        for step_index, ordinal in execute_ordinals(trajectory)
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
            first_token = text.split(maxsplit=1)[0]
            if first_token.startswith("O") and first_token[1:].isdigit():
                alias = first_token
        skeleton[key] = _observation_label(trajectory, key, value, alias)

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
    if measured_bytes > greedy_max_bytes:
        raise ValueError("observation labels alone exceed the greedy replan observation bound")
    inlined_aliases = [execute_aliases[key] for key in execute_keys if key in inlined_keys]
    labeled_aliases = [execute_aliases[key] for key in execute_keys if key not in inlined_keys]
    metadata = {
        "policy": "greedy_28k",
        "inlined_aliases": inlined_aliases,
        "labeled_aliases": labeled_aliases,
        "measured_bytes": measured_bytes,
        "greedy_max_bytes": greedy_max_bytes,
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
    """Three segments of max_iters with at most two greedy-28k replans."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.forced_replans = 0
        self.continuation_scope_id: str | None = None
        self.max_forced_replans = max_forced_replans_from_env()

    def export_suspended(self) -> dict[str, Any] | None:
        data = super().export_suspended()
        if data is not None:
            data["forced_replans"] = self.forced_replans
        return data

    def import_suspended(self, data: dict[str, Any]) -> None:
        super().import_suspended(data)
        self.forced_replans = int(data.get("forced_replans", 0))

    def _finish_prediction(
        self,
        trajectory: dict[str, Any],
        input_args: dict[str, Any],
    ) -> dspy.Prediction:
        extract = self._call_with_potential_trajectory_truncation(
            self.extract, trajectory, **input_args
        )
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
        skeleton, observation_metadata = replan_trajectory_skeleton(trajectory)
        trigger = (
            f"segment {completed_segment} reached the {self.max_iters}-iteration "
            "limit without agent-selected finish"
        )
        host = tracing.current_host()
        span = tracing.start_span(
            host,
            tracing.SPAN_PLANNER_REPLAN,
            kind=tracing.KIND_LLM,
            attributes={
                "replan_trigger": "structured_continuation_segment_limit",
                "completed_segment": completed_segment,
                "next_segment": next_segment,
                "max_segments": TOTAL_SEGMENTS,
            },
        )
        try:
            prediction = dspy.Predict(ContinuationPlanSignature)(
                user_query=str(input_args.get("user_query") or ""),
                trajectory_skeleton=json.dumps(
                    skeleton, ensure_ascii=False, default=str
                ),
            )
            plan = str(prediction.next_steps or "").strip()[:MAX_REPLAN_CHARS]
        except BaseException:
            tracing.end_span(host, span, status=tracing.STATUS_ERROR)
            raise
        artifact = (
            f"HARNESS REPLAN — segment {next_segment} of {TOTAL_SEGMENTS}. "
            f"Reason: {trigger}.\n{plan or 'Continue unfinished requested work.'}"
        )
        tracing.end_span(host, span, attributes={"plan": plan, "artifact": artifact})
        artifact_key = f"replan_{completed_segment}"
        trajectory[artifact_key] = artifact
        self.current_trajectory[artifact_key] = artifact
        self.forced_replans += 1
        self.iteration_counter = 0
        record_event(
            {
                "kind": "forced_replan",
                "scope_id": getattr(self, "continuation_scope_id", None),
                "completed_segment": completed_segment,
                "next_segment": next_segment,
                "max_segments": TOTAL_SEGMENTS,
                "reason": trigger,
                "plan": plan,
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
                record_event(
                    {
                        "kind": "forced_replan_wall",
                        "scope_id": getattr(self, "continuation_scope_id", None),
                        "completed_segment": TOTAL_SEGMENTS,
                        "max_segments": TOTAL_SEGMENTS,
                        "reason": (
                            f"segment {TOTAL_SEGMENTS} reached the "
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
