"""
Transport-free, synchronous workflow execution core.

Embedders (e.g. FastAPI) should use one WorkflowExecutionContext per session:
bind_app_workflow once, call process_turn per request in a worker thread or
asyncio task (ContextVar isolates active workflow per thread/task), and close()
on session end.

Topology B (no user_message_queue): ask_user is non-blocking — it suspends the
ReAct trajectory in memory and the turn returns an awaiting_user
CommandOutput; the next process_turn(answer) resumes it. A suspended turn
never hangs, so there is no timeout; embedders abandon an unanswered
clarification with cancel_pending() per their own session lifecycle.

ChatSession composes this core for CLI/REPL (queues, ChatWorker, keep_alive).
"""


from __future__ import annotations

import contextlib
import json
import os
import time
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from types import SimpleNamespace
from typing import Any, Optional

import dspy

import fastworkflow
import fastworkflow.turn
from fastworkflow import active_workflow, external_operations, metrics, tracing
from fastworkflow.result_handles import (
    PRESENTATION_TRUNCATION_CLASSIFICATION,
    ResultHandleStore,
)
from fastworkflow.runtime_config import get_runtime_config
from fastworkflow.runtime_manifest import (
    get_runtime_metadata,
    load_manifest,
    merge_and_gate,
)
from fastworkflow.session_state_store import (
    READABLE_SCHEMA_VERSIONS,
    SCHEMA_VERSION,
    IncompatibleSessionState,
)
from fastworkflow.turn_budget import LogicalTurnBudget
from fastworkflow.typed_failure import (
    CODE_EXTRACTION_TRUNCATED,
    CODE_PROVIDER_TIMEOUT,
    TypedFailure,
    extraction_truncated_failure,
    is_provider_timeout,
    provider_timeout_failure,
)
from fastworkflow.plan import (
    PlanConfigurationError,
    PlanMode,
    PlanRecord,
    bind_captured,
    expand,
    render_account,
)
from fastworkflow.plan_execution import (
    PlanExecutionArm,
    PlanExecutionOutcome,
    PlanExecutionScope,
    SafetyEnvelopeState,
    execute_plan,
    plan_execution_arm_from_env,
    plan_stress_mode_from_env,
    reconcile_plan_statuses,
    render_leaf_instruction,
)
from fastworkflow.state_serialization import validate_state
from fastworkflow.execution_recorder import ExecutionRecorder, record_execution
from fastworkflow.skill_catalog import load_skill_catalog
from fastworkflow.turn import TurnResult, TurnStatus, mint_turn_key
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_logger, dspy_utils
from fastworkflow.utils.react import NoSuspendedAgentStateError


def _agent_result_attributes(result: Any, attempts: int) -> dict[str, Any]:
    """Close-out attributes for fw.agent.execute — what the executor returned.

    Read defensively: a suspended run returns a Prediction with no
    ``final_answer``, and distillation passes their own result shapes through
    the same choke point.
    """
    attributes = {
        "attempts": attempts,
        "final_answer": getattr(result, "final_answer", None),
        "suspended": bool(getattr(result, "suspended", False)),
        "clarification": getattr(result, "clarification", None),
        "exhausted": bool(getattr(result, "exhausted", False)),
        "censored": bool(getattr(result, "censored", False)),
        "censored_reason": getattr(result, "censored_reason", None),
        "provider_timeout": bool(
            getattr(result, "provider_timeout", False)
        ),
        "plan_outcome": getattr(
            getattr(result, "plan_outcome", None),
            "value",
            getattr(result, "plan_outcome", None),
        ),
    }
    # EXP-025a: the BEFORE_FINISH decision, when one was taken. It happens after
    # the tool loop has ended, so it has no step span to hang off, and without it
    # a corrected answer is indistinguishable from an answer that never deferred.
    # Absent on every run where the policy said nothing, which is most of them.
    # EXP-027: a turn that stopped short says so, with counts taken from the
    # runtime's own record. `exhausted` alone is a boolean nobody can act on.
    partial = getattr(result, "turn_partial", None)
    if partial is not None:
        # Written out key by key rather than through a helper: the span-contract
        # test reads emission sites STATICALLY, and a `partial.as_attributes()`
        # call is opaque to it — it refused this file until the keys were
        # visible here. That refusal is the guard working, and the keys being
        # readable at the point they are emitted is worth more than the tidier
        # call it replaced.
        attributes["partial_reason"] = partial.reason
        attributes["partial_iterations_consumed"] = partial.iterations_consumed
        attributes["partial_iteration_limit"] = partial.iteration_limit
        attributes["partial_commands_executed"] = len(partial.commands_executed)

    decision = getattr(result, "finish_policy", None)
    if decision is not None:
        attributes["finish_policy_outcome"] = decision.outcome.value
        attributes["finish_policy_source"] = decision.source_policy
        attributes["finish_policy_table_version"] = decision.table_version

    # ido-mn1.6.6: which result handles the extraction call was allowed to
    # present, and what the byte cap did to them. Written key by key for the
    # reason the partial block above gives: the span-contract scan reads
    # emission sites statically, and a key it cannot see is a key nothing checks.
    # Absent when the agent cited nothing and no skill marked a presentation
    # output, which is every turn that produced no handle at all.
    presented = getattr(result, "presented_results", None)
    if presented:
        attributes["presented_result_handles"] = presented.get("handles")
        attributes["presented_result_bytes"] = presented.get("bytes")
        attributes["presented_result_field_bytes"] = presented.get("field_bytes")
        attributes["presented_result_trimmed"] = presented.get("trimmed")
        attributes["presented_result_omitted_handles"] = presented.get(
            "omitted_handles"
        )
        attributes["presented_result_truncation_classification"] = presented.get(
            "truncation_classification"
        )

    # ido-mn1.6.10: the completion limit this turn's composition call was given,
    # and what it was derived from. Written key by key for the reason the two
    # blocks above give — the span-contract scan reads emission sites
    # statically, so a key it cannot see is a key nothing checks.
    #
    # `fw.llm.call` already records `call_kwargs`, which will show the
    # `max_tokens` and `timeout` that reached the provider. That says WHAT was
    # asked for; these say WHY, and the difference is the whole point: a reader
    # looking at a 4096-token answer cannot tell a floor from a derivation that
    # happened to land there without the inputs, and 69 v4 calls are cut in
    # exactly that undiagnosable way.
    bound = getattr(result, "extraction_bound", None)
    if bound:
        attributes["extraction_max_tokens"] = bound.get("max_tokens")
        attributes["extraction_timeout_s"] = bound.get("timeout_s")
        attributes["extraction_field_bytes"] = bound.get("field_bytes")
        attributes["extraction_thought_bytes"] = bound.get("thought_bytes")
        # v9 (2026-09-05): the trajectory term, without which a flat turn's
        # bound reads as a floor applied to a few hundred bytes.
        attributes["extraction_trajectory_bytes"] = bound.get("trajectory_bytes")
        # v10 (2026-09-05): the provider's completion ceiling for the agent
        # route and whether it bound this call.
        attributes["extraction_provider_max_output_tokens"] = bound.get(
            "provider_max_output_tokens"
        )
        attributes["extraction_provider_cap_applied"] = bound.get(
            "provider_cap_applied"
        )
        attributes["extraction_render_factor"] = bound.get("render_factor")
        attributes["extraction_prose_allowance_tokens"] = bound.get(
            "prose_allowance_tokens"
        )
        attributes["extraction_derived_tokens"] = bound.get("derived_tokens")
        attributes["extraction_floor_applied"] = bound.get("floor_applied")
        attributes["extraction_ceiling_applied"] = bound.get("ceiling_applied")
        attributes["extraction_timeout_clamped"] = bound.get("timeout_clamped")
        # ido-mn1.6.33: what the call ASKED for, what the turn had left, and
        # whether what was left could buy an attempt at all. `timeout_s` alone
        # cannot distinguish a bound that was shortened by the turn deadline
        # from one that was never long, which is the reading the whole
        # late-extraction finding turns on.
        attributes["extraction_derived_timeout_s"] = bound.get(
            "derived_timeout_s"
        )
        attributes["extraction_deadline_remaining_s"] = bound.get(
            "deadline_remaining_s"
        )
        attributes["extraction_deadline_insufficient"] = bound.get(
            "deadline_insufficient"
        )
    if getattr(result, "extraction_truncated", False):
        attributes["extraction_truncated"] = True
        attributes["extraction_truncated_goal_ids"] = list(
            getattr(result, "extraction_truncated_goal_ids", ()) or ()
        )
        # Which mechanism produced the marker: an answer cut at `max_tokens`
        # and an answer that was never composed are both infrastructure and
        # both keep their evidence, but they are not the same finding.
        if cause := getattr(result, "extraction_truncated_cause", None):
            attributes["extraction_truncated_cause"] = cause
    return attributes


class CommandCancelledError(BaseException):
    """
    Raised when a command cannot continue (e.g. the nested intent-clarification
    ask_user is reached with no user_message_queue).

    Subclasses BaseException so fastWorkflowReAct's ``except Exception`` does not
    swallow it; _execute_message converts it to a failed CommandOutput.
    """


@dataclass(frozen=True)
class _RestoredAgentResult:
    """Stand-in for a restored agent result.

    The finalize path reads exactly one attribute off _turn_agent_result
    (``exhausted``) plus whether it is None at all, so a restore carries those
    two facts rather than a dspy Prediction it could not encode anyway.
    """

    exhausted: bool = False
    censored: bool = False
    censored_reason: Optional[str] = None
    provider_timeout: bool = False
    # ido-mn1.6.10: restored with the rest because a resumed turn that was
    # truncated before suspension is still a truncated turn, and a restore that
    # dropped it would silently upgrade the answer to complete.
    extraction_truncated: bool = False
    extraction_truncated_reason: Optional[str] = None
    extraction_truncated_goal_ids: tuple[str, ...] = ()
    extraction_failure: Optional[TypedFailure] = None
    plan_outcome: Optional[str] = None
    failure: Optional[TypedFailure] = None


def _parse_isoformat(value: Optional[str]) -> Optional[datetime]:
    """Parse a serialized timestamp, tolerating a missing or malformed one.

    A timestamp only feeds duration reporting, so a bad one degrades a metric.
    Raising here would fail a restore that is otherwise complete.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        logger.warning(f"Unparseable timestamp in restored turn state: {value!r}")
        return None


class WorkflowExecutionContext:
    """
    Owns NLU (cme_workflow), the bound app workflow, and message execution.

    No queues, threads, or session lifecycle — inject optional queues from
    ChatSession for trace/output/ask_user when running in CLI mode.
    """

    def __init__(
        self,
        run_as_agent: bool = False,
        session_key: Optional[str] = None,
        mirror_action_log_to_file: bool = False,
        generate_insights: bool = False,
        trace_sink: Optional[tracing.TraceSink] = None,
    ):
        """
        Args:
            session_key: Stable id (e.g. channel_id) for cme/app workflow persistence.
                         When omitted, cme uses an ephemeral uuid (CLI one-off sessions).
            mirror_action_log_to_file: DEPRECATED no-op (Phase 7 [R25]). The cwd
                         action.jsonl debug mirror was retired; use the in-process
                         ``action_log`` property (live) or the observability DB
                         (post-mortem) instead. Kept one release for external
                         callers, then removed.
            generate_insights: If True, enable teacher/student distillation on each
                         agent turn (Topology A / CLI only).
            trace_sink: Observability sink for boundary spans and turn records
                         (observability design §3.1). Defaults to a no-op sink;
                         reached via this context, never the transport queues [R28].
        """
        self._session_key = session_key
        self._run_as_agent = run_as_agent
        self._app_workflow: Optional[fastworkflow.Workflow] = None
        self._keep_alive = False
        # Deprecated no-op, retained one release for ctor compatibility [R25].
        self._mirror_action_log_to_file = mirror_action_log_to_file

        self._user_message_queue: Optional[Queue] = None
        self._command_output_queue: Optional[Queue] = None
        self._command_trace_queue: Optional[Queue] = None

        self._conversation_history: dspy.History = dspy.History(messages=[])
        self._action_log: list[dict[str, Any]] = []
        # Full payloads of commands that opted into observation compaction
        # (`result_handles`, ido-mn1.6.1). Session-scoped rather than
        # turn-scoped: a handle the agent was given in one turn is one it may
        # page through in the next, and a turn boundary is not a reason for a
        # citation to stop resolving.
        self._result_handles = ResultHandleStore()

        from fastworkflow.command_executor import CommandExecutor
        self._CommandExecutor = CommandExecutor

        self._workflow_tool_agent = None
        self._intent_clarification_agent = None
        self._context_change_listener = None

        self._awaiting_user = False
        self._suspended_user_message: Optional[str] = None
        self._pending_clarification_request: Optional[str] = None

        # Insights-distillation (teacher/student) state — CLI/Topology-A only.
        self._generate_insights = generate_insights
        self._distillation_insights_count = 0
        self._planning_insights: Optional[str] = None
        self._execution_insights: Optional[str] = None
        # The pass currently executing, stamped onto every span emitted inside
        # it ([DR3]); None between passes and on every non-distilled turn.
        self._distillation_pass: Optional[str] = None
        self._distillation_run_ids: list[str] = []

        # Observability (design §3.1): sink + identity + span bookkeeping.
        # The sink is a per-context attribute, not transport state [R28].
        self._trace_sink: tracing.TraceSink = trace_sink or tracing.NoOpTraceSink()
        self._metrics_sink: metrics.MetricsSink = metrics.NoOpMetricsSink()
        self._channel_id: Optional[str] = None
        self._conversation_id: Optional[int] = None
        self._embedder_owns_conversations: bool = False
        # The experiment container's labels, bound beside channel/conversation
        # identity and stamped onto every TurnResult this context produces
        # (`fix-bn1`, `[XR17]`). None on every ordinary turn.
        self._experiment_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._attempt: Optional[int] = None
        self._trace_span_stack: list[tracing.Span] = []
        self._turn_root_span: Optional[tracing.Span] = None

        # Turn accumulator state (one logical turn = one key, across suspensions)
        self._turn_outputs: list = []
        # Bound here and not only in _begin_turn, because _build_turn_result is
        # reachable on a context that never began a turn in THIS process: resume
        # continues the same logical turn and deliberately skips _begin_turn
        # (see _serialize_turn_accumulator). The read at finalize guards on
        # `is not None`, so an attribute that does not exist made the guard
        # itself raise AttributeError — a missing binding wearing the costume of
        # a null check. fix-ajv.20.
        self._execution_recorder: Optional[ExecutionRecorder] = None
        self._turn_key: Optional[str] = None
        # ido-mn1.6.3: what_can_i_do listings the agent tool has already shown
        # this turn. Reset lazily by `command_listing_memo` on turn-key change.
        self._command_listing_memo: dict = {}
        self._command_listing_memo_turn: Optional[str] = None
        # The logical turn's budget (arch §6.4). Created at _begin_turn, handed
        # to the planner and to ReAct, serialized at suspension, restored
        # unchanged on resume. None between turns and on every deterministic
        # turn, which spends no agent iterations at all.
        self._turn_budget: Optional[LogicalTurnBudget] = None
        self._turn_failure: Optional[TypedFailure] = None
        # The host/request layer of the restrictive budget precedence (arch
        # §6.0). None means the host declares no limit; a value may only lower
        # the deployment maximum, never raise it.
        self._host_react_max_iterations: Optional[int] = None
        self._contract_react_max_iterations: Optional[int] = None
        # [DR41]: a SECOND, independent trace id for the counterfactual-replay
        # path. `_turn_key` is never overridden — overriding it is exactly the
        # corruption §3.3 rejects option (c) for — so a replay writes into
        # `<turn_key>~replay.<n>` while the turn machinery keeps seeing None.
        self._replay_trace_id: Optional[str] = None
        self._turn_started_at: Optional[datetime] = None
        self._turn_user_message: str = ""
        self._turn_refined_message: Optional[str] = None
        self._turn_suspended_ms: int = 0
        self._suspend_began_at: Optional[datetime] = None
        self._turn_entry_workflow_name: str = ""
        self._turn_entry_context: str = ""
        self._turn_agent_result: Any = None
        self._turn_plan: Any = None
        self._turn_plan_answers: list[dict[str, str]] = []
        # One entry per fw.agent.execute: which result handles that call's
        # extraction step was given, and what the byte cap did to them. Lives on
        # the turn rather than only on the spans because the turn record is where
        # `final_answer` provenance is read (ido-mn1.6.6).
        self._turn_presented_results: list[dict[str, Any]] = []
        # The authoritative binding/navigation envelope for the plan leaf
        # currently driving the shared workflow tool. It is process-local and
        # reconstructed from the persisted plan before a resumed leaf runs.
        self._turn_leaf_scope: Optional[PlanExecutionScope] = None
        self._turn_plan_frontier: list[str] = []
        self._turn_active_leaf: Optional[str] = None
        self._turn_plan_outcome: Optional[str] = None
        self._turn_plan_checkpoint_count: int = 0
        self._turn_plan_last_checkpoint_stored: Optional[bool] = None
        self._turn_plan_last_checkpoint_leaf: Optional[str] = None
        self._turn_safety_envelope: SafetyEnvelopeState = SafetyEnvelopeState(
            enabled=False
        )
        self._turn_history_baseline: int = 0
        # turn_key of the newest turn that both completed and contributed a
        # conversation-history entry — the row feedback attaches to (ruling
        # I3). Serialized with session state so a cross-process resume keys
        # feedback off a real turn instead of inferring one from SQL.
        self._last_completed_turn_key: Optional[str] = None
        # Ack from the last turn-record emission: True stored, False queued and
        # not yet durable, None no sink at all. Read by embedders that trim
        # conversation history (ruling I1/I2).
        self._last_turn_record_stored: Optional[bool] = None
        # Whether the last finalize contributed a conversation-history entry —
        # i.e. whether it grew the durable memory the label schedule counts.
        self._last_turn_added_memory: bool = False

        cme_id = (
            f"cme_{session_key}"
            if session_key
            else f"cme_{uuid.uuid4().hex}"
        )
        self._cme_workflow = fastworkflow.Workflow.create(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            workflow_id_str=cme_id,
            workflow_context={
                "NLU_Pipeline_Stage": fastworkflow.NLUPipelineStage.INTENT_DETECTION,
            },
        )

        self.clear_conversation_history()

    @property
    def session_key(self) -> Optional[str]:
        return self._session_key

    # ------------------------------------------------------------------
    # Observability: sink wiring + identity plumbing (design §3.1 [R1][R28])
    # ------------------------------------------------------------------

    @property
    def trace_sink(self) -> tracing.TraceSink:
        return self._trace_sink

    def set_trace_sink(self, sink: Optional[tracing.TraceSink]) -> None:
        """Wire an observability sink (None restores the no-op default)."""
        self._trace_sink = sink or tracing.NoOpTraceSink()

    @property
    def metrics_sink(self) -> metrics.MetricsSink:
        return self._metrics_sink

    def set_metrics_sink(self, sink: Optional[metrics.MetricsSink]) -> None:
        """Wire a metrics sink (None restores the no-op default)."""
        self._metrics_sink = sink or metrics.NoOpMetricsSink()

    def bind_observability_identity(
        self,
        channel_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
        embedder_owns_conversations: Optional[bool] = None,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> None:
        """Bind channel/conversation identity BEFORE the turn [R1].

        The embedder owns identity: FastAPI binds its channel_id, the CLI a
        synthetic ``cli:<session-start>`` channel [R17]. Stamped onto every
        span and TurnResult this context produces. A None argument leaves
        the corresponding binding unchanged (conversation ids rotate without
        re-binding the channel).

        ``embedder_owns_conversations=True`` (additive) disables the WEC's
        own conversation self-minting for this context. FastAPI passes it:
        its minting chokepoint carries the legacy-store floor and syncs it
        back (ruling C2), so a WEC self-mint on its degraded path would mint
        a floor-less id that can alias a legacy conversation and split the
        session across two ids once the chokepoint's own mint succeeds.
        """
        # Validate FIRST, mutate second: a rejected experiment triple must not
        # leave the context half-bound with a new channel_id and the old labels.
        if experiment_id is not None or task_id is not None or attempt is not None:
            self._bind_experiment_labels(experiment_id, task_id, attempt)
        if channel_id is not None:
            self._channel_id = channel_id
        if conversation_id is not None:
            self._conversation_id = conversation_id
        if embedder_owns_conversations is not None:
            self._embedder_owns_conversations = embedder_owns_conversations

    def _bind_experiment_labels(
        self,
        experiment_id: Optional[str],
        task_id: Optional[str],
        attempt: Optional[int],
    ) -> None:
        """Validate and bind the experiment triple (`fix-bn1` `[XR17]`).

        All three or none. A turn labelled with an experiment but no task
        belongs to an experiment and to no task: it contributes to a numerator
        and to no denominator, and every GROUP BY in the scoring layer is
        silently wrong. Refusing here is the only cheap place to catch it.

        The `isinstance(attempt, int)` check is load-bearing and not decorative.
        SQLite's INTEGER is a type AFFINITY, not a constraint -- a string bound
        to it that cannot be losslessly converted is stored as TEXT -- so the
        column's declared type protects nothing on its own. This is what makes
        `attempt` safe to leave unpoliced (`[XR7]`).
        """
        resolved = self._validate_experiment_labels(
            experiment_id if experiment_id is not None else self._experiment_id,
            task_id if task_id is not None else self._task_id,
            attempt if attempt is not None else self._attempt,
        )
        self._experiment_id, self._task_id, self._attempt = resolved

    @staticmethod
    def _validate_experiment_labels(
        experiment_id: Optional[str],
        task_id: Optional[str],
        attempt: Optional[int],
    ) -> tuple[str, str, int]:
        """Check the triple and return it. Pure: assigns nothing."""
        if not experiment_id or not task_id:
            raise ValueError(
                "an experiment binding needs both experiment_id and task_id; "
                f"got experiment_id={experiment_id!r}, task_id={task_id!r}"
            )
        if isinstance(attempt, bool) or not isinstance(attempt, int):
            raise ValueError(f"attempt must be an int, got {type(attempt).__name__}")
        if attempt <= 0:
            raise ValueError(f"attempt must be positive, got {attempt}")
        return experiment_id, task_id, attempt

    def _ensure_observability_conversation(self) -> None:
        """Mint a conversation id when no embedder bound one.

        FastAPI and the CLI both bind one before the first turn. Code that
        embeds this context directly has no such layer, so its turns would be
        filed outside any conversation and grouped nowhere. Minting here keeps
        identity ownership with the embedder wherever one exists — a bound id
        is never replaced — while giving bare embedders the same grouping.

        Never fails a turn: an unmintable id (wedged or corrupt DB) leaves the
        turn conversation-less, exactly as before.

        Scope guard (ruling C2): an embedder that declared
        ``embedder_owns_conversations`` (FastAPI) mints through its own
        chokepoint, which carries the legacy-store floor and syncs it back;
        self-minting on its degraded path would mint a floor-less id that can
        alias a pre-existing legacy conversation and split the session across
        two ids once the chokepoint's own mint succeeds. For such embedders a
        failed mint leaves the turn conversation-less by design.
        """
        if self._conversation_id is not None or getattr(
            self, "_embedder_owns_conversations", False
        ):
            return
        store = getattr(tracing.get_sink(self), "store", None)
        if store is None:
            return
        try:
            # Mint against the channel the sink files this turn's row under.
            self._conversation_id = store.mint_conversation_id(
                self._channel_id or "",
                experiment_id=self._experiment_id,
                task_id=self._task_id,
                attempt=self._attempt,
            )
        except sqlite3.IntegrityError:
            # NOT swallowed. `idx_conv_experiment_attempt` is UNIQUE precisely so
            # that a second conversation under one (experiment, task, attempt)
            # is refused, and degrading that refusal to a conversation-less turn
            # would defeat the invariant silently: the attempt would keep running
            # and its turns would land outside any conversation, which is exactly
            # the unreconstructable state the index exists to prevent. An
            # ordinary turn cannot reach this arm -- the index is partial on
            # experiment_id IS NOT NULL.
            raise
        except Exception as exc:
            logger.warning(
                f"Could not mint a conversation id ({type(exc).__name__}: {exc}); "
                "this turn is recorded without a conversation"
            )

    @property
    def observability_channel_id(self) -> Optional[str]:
        return self._channel_id

    @property
    def observability_conversation_id(self) -> Optional[int]:
        return self._conversation_id

    @property
    def current_turn_key(self) -> Optional[str]:
        """The open logical turn's key, or None between turns."""
        return self._turn_key

    @property
    def current_replay_trace_id(self) -> Optional[str]:
        """The replay trace spans are being written into, or None ([DR41]).

        Set only by `replay_trace_scope`, only by the counterfactual-replay
        driver, and only to `<original_turn_key>~replay.<n>` ([DR5]).
        """
        return self._replay_trace_id

    @contextlib.contextmanager
    def replay_trace_scope(self, trace_id: str):
        """Write spans into a derived replay trace for the enclosed block.

        The `finally` clears it on the failure path too, and that is not the
        only guard: a leaked id still cannot write a `turns` row, because
        `finalize_turn_for_observability` short-circuits on
        `self._turn_key is None` and the replay driver never calls
        `_begin_turn`. Belt and braces, because the thing being protected is
        the evidence a shipped rule cites.
        """
        previous = self._replay_trace_id
        self._replay_trace_id = trace_id
        try:
            yield
        finally:
            self._replay_trace_id = previous

    @property
    def trace_span_stack(self) -> list[tracing.Span]:
        """Open-span stack for parenting nested spans (single-turn: I5)."""
        return self._trace_span_stack

    @property
    def current_refined_message(self) -> Optional[str]:
        """The refined user query this turn's prompts saw, or None.

        Read by `distillation.prompt_fingerprint` ([DR47]), which hashes the
        inputs a pass's prompts actually see rather than the world it runs
        against. It is None until `_process_agent_message` refines the message,
        and it stays None through a distillation pass: the distillation branch
        refines per pass without stamping the turn, and *which* pass supplies a
        turn's fields is [DR42]'s question, not this property's.
        """
        return self._turn_refined_message

    @property
    def current_distillation_pass(self) -> Optional[str]:
        """The distillation pass whose work is running, or None ([DR3])."""
        return self._distillation_pass

    @contextlib.contextmanager
    def distillation_pass_scope(self, pass_label: Optional[str]):
        """Label every span emitted inside the block with *pass_label*.

        Restored on **every** exit path, the student-failure raise included: a
        leaked label files each following turn's spans under a stale pass,
        which is worse than the interleaving this exists to fix.
        """
        previous = self._distillation_pass
        self._distillation_pass = pass_label
        try:
            yield
        finally:
            self._distillation_pass = previous

    def command_listing_memo(self) -> dict[Any, Any]:
        """Listings the agent-facing `what_can_i_do` TOOL already showed this turn.

        ido-mn1.6.3. The entries are keyed and read by
        `fastworkflow.workflow_agent._what_can_i_do_tool_observation`; the only
        thing owned here is the lifetime — the dict is emptied whenever the
        logical turn key changes, so nothing is remembered across turns, while a
        suspended turn (which deliberately keeps its key) resumes with what it
        had. Not serialized into the turn accumulator on purpose: losing the memo
        costs one full listing, whereas a stale memo would cost a wrong reference.
        """
        turn_key = self._turn_key
        if self._command_listing_memo_turn != turn_key:
            self._command_listing_memo_turn = turn_key
            self._command_listing_memo = {}
        return self._command_listing_memo

    def clear_action_log(self) -> None:
        """Clear in-memory action log for a new agent turn."""
        self._action_log.clear()

    def append_action_log(self, record: dict[str, Any]) -> None:
        """Append one agent/workflow interaction record (in-memory only)."""
        self._action_log.append(record)

    @property
    def action_log(self) -> list[dict[str, Any]]:
        return self._action_log

    @property
    def result_handles(self) -> ResultHandleStore:
        """Stored full payloads for commands that opted into compaction.

        Reached by `result_handles.store_for()` through the trace host, which is
        what lets a command's `ResponseGenerator` — which holds a `Workflow` and
        no session — fetch a page without a new parameter on every signature.
        """
        return self._result_handles

    # ------------------------------------------------------------------
    # Turn accumulator (v2.21: capture + TurnResult return type only)
    # ------------------------------------------------------------------

    def bind_turn_iteration_limits(
        self,
        *,
        host_limit: Optional[int] = None,
        contract_limit: Optional[int] = None,
    ) -> None:
        """Declare host/request and task-contract ceilings on the turn budget.

        Both participate in the restrictive minimum of arch §6.0 and neither can
        raise the deployment maximum. Takes effect from the next fresh turn: a
        running turn keeps the budget it was created with, because a limit that
        changed underneath a turn would make its exhaustion unattributable.
        """
        self._host_react_max_iterations = host_limit
        self._contract_react_max_iterations = contract_limit

    def _effective_react_max_iterations(self) -> int:
        """Resolve this turn's iteration limit (arch §6.0 restrictive minimum).

        Deployment maximum, workflow manifest ceiling, task-contract limit and
        host/request limit, minimum wins. Resolved once per fresh turn and
        persisted in the budget; resume never recomputes it, so a configuration
        change mid-suspension cannot retroactively shrink or extend a turn that
        is already running.
        """
        manifest_limit = None
        if self._app_workflow is not None:
            metadata = get_runtime_metadata(self._app_workflow.folderpath)
            if metadata is not None:
                manifest_limit = metadata.react_max_iterations
        return get_runtime_config().effective_react_max_iterations(
            manifest_limit=manifest_limit,
            contract_limit=self._contract_react_max_iterations,
            host_limit=self._host_react_max_iterations,
        )

    def _require_turn_budget(self) -> LogicalTurnBudget:
        """This turn's budget, or a hard failure.

        Agent work outside a begun turn has no budget it can be charged to, and
        minting one here would restore exactly the accounting nobody could
        attribute — the defect FW-REQ-001 closes.
        """
        if self._turn_budget is None:
            raise RuntimeError(
                "no logical turn budget: agent work reached the executor without "
                "_begin_turn having started a turn (arch §6.4)"
            )
        return self._turn_budget

    @property
    def turn_budget(self) -> Optional[LogicalTurnBudget]:
        """The active logical turn's budget, or None between turns."""
        return self._turn_budget

    @property
    def turn_failure(self) -> Optional[TypedFailure]:
        """The classified failure this turn ended with, or None."""
        return self._turn_failure

    def _begin_turn(self, user_message: str) -> None:
        """Atomic turn start [A30]: reset accumulator, mint key, stamp started_at.

        Never called while awaiting_user — a message during suspension is the
        resume answer and continues the same logical turn [A30.2].
        """
        self._ensure_observability_conversation()
        self._turn_outputs = []
        self._turn_key = mint_turn_key()
        self._turn_started_at = datetime.now(timezone.utc)
        self._turn_user_message = user_message
        self._turn_refined_message = None
        self._turn_suspended_ms = 0
        self._suspend_began_at = None
        self._turn_agent_result = None
        self._turn_plan = None
        self._turn_plan_answers = []
        self._turn_presented_results = []
        self._turn_leaf_scope = None
        self._turn_plan_frontier = []
        self._turn_active_leaf = None
        self._turn_plan_outcome = None
        self._turn_plan_checkpoint_count = 0
        self._turn_plan_last_checkpoint_stored = None
        self._turn_plan_last_checkpoint_leaf = None
        stress_mode = plan_stress_mode_from_env()
        if stress_mode:
            raw_wall_limit = os.environ.get(
                "FW_PLAN_WALL_TIME_LIMIT_SECONDS",
                "1800",
            )
            try:
                wall_time_limit_s = int(raw_wall_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "FW_PLAN_WALL_TIME_LIMIT_SECONDS must be a positive integer"
                ) from exc
            if wall_time_limit_s <= 0:
                raise ValueError(
                    "FW_PLAN_WALL_TIME_LIMIT_SECONDS must be a positive integer"
                )
            self._turn_safety_envelope = SafetyEnvelopeState(
                enabled=True,
                wall_time_limit_s=wall_time_limit_s,
            )
        else:
            # Ordinary turns never parse or depend on EXP-028-only safety
            # configuration. A stale experimental variable cannot break the
            # compatible bounded path when stress mode is off.
            self._turn_safety_envelope = SafetyEnvelopeState(enabled=False)
        # The classified failure this turn ended with, when it ended with one
        # (FW-REQ-008 clause 7). None on every turn that did not fail, and
        # cleared per turn like the rest of the accumulator.
        self._turn_failure = None
        self._turn_history_baseline = len(self.conversation_history.messages)

        # The logical turn's budget (arch §6.4, FW-REQ-001 clause 1). Created
        # here and nowhere else: this is the only place a *fresh* turn starts,
        # and resume deliberately does not pass through it, which is what makes
        # "a clarification answer does not replenish the budget" structural
        # rather than a rule somebody has to remember.
        self._turn_budget = LogicalTurnBudget(
            iteration_limit=self._effective_react_max_iterations(),
            enforce_iteration_limit=not stress_mode,
        )

        self._turn_entry_workflow_name = ""
        self._turn_entry_context = ""
        with contextlib.suppress(Exception):
            if self._app_workflow is not None:
                self._turn_entry_workflow_name = (
                    self._app_workflow.folderpath.split("/")[-1]
                )
                self._turn_entry_context = (
                    self._app_workflow.current_command_context_name or ""
                )

        # Context-mutation baseline (D3 as amended): a shallow snapshot of the
        # app workflow's context, diffed at finalize so the root span records
        # what the turn's commands STORED into context — the "storing
        # information in context" feature is otherwise invisible in logs.
        # Sink-gated (zero cost with observability off); not serialized, so a
        # cross-process resume finalizes without a mutation record.
        self._turn_context_snapshot = None
        with contextlib.suppress(Exception):
            if self._app_workflow is not None and tracing.get_sink(self) is not None:
                self._turn_context_snapshot = dict(self._app_workflow.context)

        # Turn-scoped execution ledger (arch §12.1). Sink-gated like the other
        # capture projections: with observability off this must cost nothing.
        self._execution_recorder = (
            ExecutionRecorder() if tracing.get_sink(self) is not None else None
        )

        # Open the fw.turn root span (deterministic id [R6]; emitted at open so
        # a suspended turn is visible before — and closable after — a process
        # boundary). Off the stack: children parent to it via the deterministic
        # root id, which survives suspension where the stack does not.
        self._trace_span_stack.clear()
        # Belt-and-braces beside the stack clear: the pass scope restores on
        # every exit path, but a label that somehow survived a turn would
        # silently file this turn's spans under a pass that never ran.
        self._distillation_pass = None
        self._turn_root_span = tracing.start_span(
            self,
            tracing.SPAN_TURN,
            span_id=tracing.root_span_id(self._turn_key),
            attributes={
                "turn_key": self._turn_key,
                "channel_id": self._channel_id,
                "conversation_id": self._conversation_id,
                "user_message": user_message,
            },
            context=self._turn_entry_context or None,
            use_stack=False,
            emit_open=True,
        )

    def append_turn_output(self, command_output: fastworkflow.CommandOutput) -> None:
        """Append one command execution to the current turn's accumulator."""
        self._turn_outputs.append(command_output)
        fastworkflow.turn.warn_on_unserializable_artifacts(command_output)

    def append_ask_user_entry(self, question: str) -> fastworkflow.CommandOutput:
        """Append an unanswered ask_user exchange entry [A7] and return it.

        Role inversion: command_parameters holds the agent's question; the
        response holds the user's answer ("" + success=False while unanswered).

        Also opens the fw.ask_user human-wait span (deterministic id per
        attempt [R6]; emitted at open so the wait is visible while the turn
        is suspended). Both topologies funnel through here: Topology A via
        _ask_user_tool, Topology B via _note_agent_suspension.
        """
        # `is_ask_user`, not the name: this count feeds a DETERMINISTIC span id,
        # so a failed command called `ask_user` would not just miscount, it
        # would make two real ask_user spans collide on one id. fix-ajv.17.
        attempt = sum(1 for output in self._turn_outputs if output.is_ask_user)
        entry = fastworkflow.CommandOutput(
            command_name="ask_user",
            ask_user_entry=True,
            command_parameters=question,
            command_response=
                fastworkflow.CommandResponse(response="", success=False),
            started_at=datetime.now(timezone.utc),
        )
        self.append_turn_output(entry)

        if self._turn_key:
            # [DR51] / §9 producer items 6 and 12. Inside a distillation pass
            # the question belongs to THAT pass: it takes the pass label into
            # its deterministic id ([DR11], so the two passes' questions are
            # two spans rather than one upsert over the other) and parents onto
            # the pass wrapper instead of the turn root, which is what makes
            # §7.1's own sentence and [DR8]'s parenting assertion true. Outside
            # a pass the label is None and both are byte-identical to before.
            pass_label = tracing.get_distillation_pass(self)
            tracing.start_span(
                self,
                tracing.SPAN_ASK_USER,
                kind=tracing.KIND_HUMAN_WAIT,
                span_id=tracing.deterministic_span_id(
                    self._turn_key,
                    tracing.SPAN_ASK_USER,
                    attempt,
                    pass_label=pass_label,
                ),
                parent_span_id=self._ask_user_parent_span_id(pass_label),
                command_name="ask_user",
                attributes={"agent_query": question, "attempt": attempt},
                use_stack=False,
                emit_open=True,
            )
        return entry

    def complete_ask_user_entry(self, answer: str) -> None:
        """Fill the last unanswered ask_user entry with the user's answer.

        duration_ms is the user's think time [A38]. No-op when there is no
        unanswered ask_user entry.

        Closes the matching fw.ask_user span. The span is rebuilt from the
        entry rather than held in memory, so the close is an idempotent upsert
        that also works when the answer arrives in a different process than
        the question ([R6]).
        """
        for index in range(len(self._turn_outputs) - 1, -1, -1):
            entry = self._turn_outputs[index]
            # `is_ask_user`, not the bare name: a failed command that happens
            # to be called `ask_user` also matches name+unsuccessful, and this
            # loop would overwrite its error with the user's answer and mark it
            # successful. fix-ajv.17.
            if entry.is_ask_user and entry.command_response.success is False:
                entry.command_response.response = answer
                entry.command_response.success = True
                if entry.started_at is not None:
                    entry.duration_ms = int(
                        (datetime.now(timezone.utc) - entry.started_at).total_seconds()
                        * 1000
                    )
                self._close_ask_user_span(index, entry, answer)
                return

    def _ask_user_parent_span_id(self, pass_label: Optional[str]) -> str:
        """The `fw.ask_user` parent: its own pass wrapper, or the turn root.

        §9 producer item 12 / `[DR51]`. `distill_pass_span_id` is deterministic
        precisely so the close site — which holds only the turn key and the
        label — can recompute what the open wrote.
        """
        if pass_label:
            return tracing.distill_pass_span_id(self._turn_key, pass_label)
        return tracing.root_span_id(self._turn_key)

    def _close_ask_user_span(
        self, entry_index: int, entry: fastworkflow.CommandOutput, answer: str
    ) -> None:
        """Emit the closed fw.ask_user span for a just-answered entry [R6]."""
        if not self._turn_key or tracing.get_sink(self) is None:
            return
        attempt = sum(
            1
            for output in self._turn_outputs[:entry_index]
            if output.is_ask_user
        )
        # The close rebuilds the Span from pure functions of the turn key, so
        # it must agree with the open on BOTH the id and the parent: a
        # disagreeing id writes a second row, and `parent_span_id` is
        # deliberately absent from the span upsert's DO UPDATE set (§9 item 8),
        # so a wrong parent at open can never be repaired here.
        pass_label = tracing.get_distillation_pass(self)
        span = tracing.Span(
            span_id=tracing.deterministic_span_id(
                self._turn_key,
                tracing.SPAN_ASK_USER,
                attempt,
                pass_label=pass_label,
            ),
            trace_id=self._turn_key,
            name=tracing.SPAN_ASK_USER,
            kind=tracing.KIND_HUMAN_WAIT,
            parent_span_id=self._ask_user_parent_span_id(pass_label),
            channel_id=self._channel_id,
            command_name="ask_user",
            distillation_pass=pass_label,
            start_ns=tracing.datetime_to_ns(entry.started_at) or 0,
        )
        tracing.end_span(
            self,
            span,
            attributes={
                "agent_query": entry.command_parameters,
                "attempt": attempt,
                "user_response": answer,
                "human_wait_ms": entry.duration_ms,
            },
        )

    def _note_agent_suspension(self, clarification: str) -> None:
        """Bookkeeping when the agent suspends on ask_user (Topology B).

        Appends the unanswered ask_user entry unless the last entry is already
        the same unanswered question (Topology-A's blocking path appends via
        workflow_agent), and stamps the suspension start for suspended_ms.
        """
        last = self._turn_outputs[-1] if self._turn_outputs else None
        already_appended = (
            last is not None
            and last.is_ask_user
            and last.command_response.success is False
            and last.command_parameters == clarification
        )
        if not already_appended:
            self.append_ask_user_entry(clarification)
        self._suspend_began_at = datetime.now(timezone.utc)

    def _note_agent_resume(self) -> None:
        """Fold the elapsed suspension into suspended_ms on resume entry."""
        if self._suspend_began_at is not None:
            self._turn_suspended_ms += int(
                (datetime.now(timezone.utc) - self._suspend_began_at).total_seconds()
                * 1000
            )
            self._suspend_began_at = None

    # ------------------------------------------------------------------
    # Queue injection (CLI driver only)
    # ------------------------------------------------------------------

    def set_transport_queues(
        self,
        user_message_queue: Optional[Queue] = None,
        command_output_queue: Optional[Queue] = None,
        command_trace_queue: Optional[Queue] = None,
        keep_alive: bool = False,
    ) -> None:
        """Wire ChatSession queues and keep_alive flag for REPL transport."""
        self._user_message_queue = user_message_queue
        self._command_output_queue = command_output_queue
        self._command_trace_queue = command_trace_queue
        self._keep_alive = keep_alive

    @property
    def user_message_queue(self) -> Optional[Queue]:
        return self._user_message_queue

    @property
    def command_output_queue(self) -> Optional[Queue]:
        return self._command_output_queue

    @property
    def command_trace_queue(self) -> Optional[Queue]:
        return self._command_trace_queue

    @property
    def keep_alive(self) -> bool:
        return self._keep_alive

    @keep_alive.setter
    def keep_alive(self, value: bool) -> None:
        self._keep_alive = value

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def cme_workflow(self) -> fastworkflow.Workflow:
        return self._cme_workflow

    @property
    def run_as_agent(self) -> bool:
        return self._run_as_agent

    @property
    def app_workflow(self) -> Optional[fastworkflow.Workflow]:
        return self._app_workflow

    @property
    def workflow_tool_agent(self):
        return self._workflow_tool_agent

    @property
    def intent_clarification_agent(self):
        return self._intent_clarification_agent

    @property
    def conversation_history(self) -> dspy.History:
        return self._conversation_history

    @property
    def awaiting_user(self) -> bool:
        """True when the agent suspended on ask_user and awaits the next process_turn."""
        return self._awaiting_user

    @property
    def last_completed_turn_key(self) -> Optional[str]:
        """turn_key of the newest completed turn that produced a memory entry.

        Feedback keys off this rather than off "the newest row for the
        conversation": a max-ordinal query attaches feedback to whatever
        happened to be written last, which on a suspended or cancelled turn is
        not the turn the user was looking at (ruling I3/C4).
        """
        return self._last_completed_turn_key

    @property
    def last_turn_added_memory(self) -> bool:
        """Whether the last finalize grew the durable conversation memory.

        The label-refresh schedule counts usable turns, so it needs to know
        what THIS turn contributed; a cancelled turn or an abandoned suspension
        wrote a row but added nothing to summarize (ruling I10).
        """
        return self._last_turn_added_memory

    @property
    def last_turn_record_stored(self) -> Optional[bool]:
        """Whether the last turn record reached durable storage (ruling I1).

        None when no sink is installed — there is no durable record to wait
        for, so a caller gating a history trim on durability must treat it as
        "nothing to defer for" rather than as a failure.
        """
        return self._last_turn_record_stored

    def _serialize_turn_accumulator(self) -> Optional[dict[str, Any]]:
        """Project the logical-turn accumulator, or None when no turn is open.

        Resume continues the same logical turn rather than starting a new one
        (_begin_turn is deliberately skipped), so without this the resumed turn
        takes a fresh key and reports only its post-resume commands.

        _turn_agent_result is distilled to the one fact the finalize path reads
        (`exhausted`) instead of being serialized whole: it is a dspy Prediction
        whose other fields nothing downstream consults, and storing an opaque
        object would fail the strict encoder for no gain.
        """
        if self._turn_key is None:
            return None

        agent_result = None
        if self._turn_agent_result is not None:
            agent_result = {
                "exhausted": bool(
                    getattr(self._turn_agent_result, "exhausted", False)
                )
            }
            if getattr(self._turn_agent_result, "censored", False):
                agent_result["censored"] = True
            if censored_reason := getattr(
                self._turn_agent_result,
                "censored_reason",
                None,
            ):
                agent_result["censored_reason"] = censored_reason
            if getattr(self._turn_agent_result, "provider_timeout", False):
                agent_result["provider_timeout"] = True
            if getattr(
                self._turn_agent_result, "extraction_truncated", False
            ):
                agent_result["extraction_truncated"] = True
                agent_result["extraction_truncated_reason"] = getattr(
                    self._turn_agent_result,
                    "extraction_truncated_reason",
                    None,
                ) or CODE_EXTRACTION_TRUNCATED
                agent_result["extraction_truncated_goal_ids"] = list(
                    getattr(
                        self._turn_agent_result,
                        "extraction_truncated_goal_ids",
                        (),
                    )
                    or ()
                )
                extraction_failure = getattr(
                    self._turn_agent_result,
                    "extraction_failure",
                    None,
                )
                if not isinstance(extraction_failure, TypedFailure):
                    extraction_failure = extraction_truncated_failure()
                agent_result["extraction_failure"] = (
                    extraction_failure.to_state()
                )
            if plan_outcome := getattr(
                self._turn_agent_result,
                "plan_outcome",
                None,
            ):
                agent_result["plan_outcome"] = getattr(
                    plan_outcome,
                    "value",
                    plan_outcome,
                )
            failure = getattr(self._turn_agent_result, "failure", None)
            if isinstance(failure, TypedFailure):
                agent_result["failure"] = failure.to_state()

        return {
            "key": self._turn_key,
            "outputs": [o.model_dump(mode="json") for o in self._turn_outputs],
            "started_at": (
                self._turn_started_at.isoformat() if self._turn_started_at else None
            ),
            "user_message": self._turn_user_message,
            "refined_message": self._turn_refined_message,
            "suspended_ms": self._turn_suspended_ms,
            "suspend_began_at": (
                self._suspend_began_at.isoformat() if self._suspend_began_at else None
            ),
            "entry_workflow_name": self._turn_entry_workflow_name,
            "entry_context": self._turn_entry_context,
            "agent_result": agent_result,
            "plan": (
                self._turn_plan.model_dump(mode="json")
                if self._turn_plan is not None
                else None
            ),
            "plan_answers": list(self._turn_plan_answers),
            "presented_results": list(self._turn_presented_results),
            "plan_frontier": list(self._turn_plan_frontier),
            "active_leaf": self._turn_active_leaf,
            "active_leaf_scope": (
                self._turn_leaf_scope.to_state()
                if self._turn_leaf_scope is not None
                else None
            ),
            "plan_outcome": self._turn_plan_outcome,
            "plan_checkpoint_count": self._turn_plan_checkpoint_count,
            "plan_last_checkpoint_stored": (
                self._turn_plan_last_checkpoint_stored
            ),
            "plan_last_checkpoint_leaf": self._turn_plan_last_checkpoint_leaf,
            "safety_envelope": self._turn_safety_envelope.to_state(),
        }

    def _serialize_cme_continuation(self) -> Optional[dict[str, Any]]:
        """Project the in-flight CME command, or None when none is in flight.

        Restoring nlu_stage alone is not enough and is actively unsafe: at
        PARAMETER_EXTRACTION, wildcard.py reads context["command_name"]
        unconditionally, and parameter_extraction.py merges the user's answer
        into stored_parameters. Without these three keys a restored session
        either raises KeyError or silently discards every parameter collected
        so far and re-extracts from the answer text alone.
        """
        if self._cme_workflow is None:
            # close() treats this as reachable, and has_open_command() runs
            # after every turn rather than only suspensions, so a missing CME
            # workflow must read as "nothing in flight" instead of raising in
            # the post-turn persist path.
            return None

        ctx = self._cme_workflow.context
        stored = ctx.get("stored_parameters")

        # command_name is deliberately NOT part of this test. Unlike command and
        # stored_parameters, end_command_processing() leaves it behind, so a
        # session that merely ran a command once would look mid-extraction
        # forever and its state would be written on every completed turn.
        if stored is None and not self._is_extracting_parameters():
            return None

        command_name = ctx.get("command_name")
        command = ctx.get("command")

        stored_dump = None
        if stored is not None:
            # model_construct built this without validation (missing fields hold
            # NOT_FOUND sentinels), so dump without validating on the way out.
            stored_dump = stored.model_dump(mode="json")

        return {
            "command": command,
            "command_name": command_name,
            "stored_parameters": stored_dump,
        }

    def _is_extracting_parameters(self) -> bool:
        """True when the NLU pipeline is parked at parameter extraction.

        The stage survives serialization as a raw value, so compare on value
        rather than on enum identity.
        """
        if self._cme_workflow is None:
            return False
        stage = self._cme_workflow.context.get("NLU_Pipeline_Stage")
        target = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
        return stage == target or getattr(stage, "value", stage) == target.value

    def has_open_command(self) -> bool:
        """True when a CME command is mid-extraction.

        Such a session is not awaiting_user, so nothing else marks it as holding
        state that must survive eviction.
        """
        return self._serialize_cme_continuation() is not None

    def serialize_state(self, *, channel_id: str) -> dict[str, Any]:
        """
        Export durable Topology-B state for cross-process resume.

        Requires session_key and bound app_workflow when persisting.
        """
        react_blob = None
        if self._workflow_tool_agent is not None:
            react_blob = self._workflow_tool_agent.export_suspended()

        nlu_stage = self._cme_workflow.context.get("NLU_Pipeline_Stage")
        if hasattr(nlu_stage, "value"):
            nlu_stage = nlu_stage.value
        elif nlu_stage is not None:
            nlu_stage = str(nlu_stage)

        current_context_name = None
        if self._app_workflow and self._app_workflow.current_command_context is not None:
            current_context_name = self._app_workflow.current_command_context_name

        from fastworkflow.conversation_history_io import extract_turns_from_history

        payload = {
            "schema_version": SCHEMA_VERSION,
            "channel_id": channel_id,
            "session_key": self._session_key,
            "app_workflow_id_str": self._session_key or channel_id,
            "cme_workflow_id_str": (
                f"cme_{self._session_key}" if self._session_key else None
            ),
            "workflow_folderpath": (
                self._app_workflow.folderpath if self._app_workflow else None
            ),
            "awaiting_user": self._awaiting_user,
            "suspended_user_message": self._suspended_user_message,
            "pending_clarification_request": self._pending_clarification_request,
            "react": react_blob,
            "nlu_stage": nlu_stage,
            "turn": self._serialize_turn_accumulator(),
            "cme": self._serialize_cme_continuation(),
            "current_command_context_name": current_context_name,
            "action_log": list(self._action_log),
            # The handle store rides the suspension blob because a resumed turn
            # continues the SAME logical turn: the agent's trajectory still
            # carries the compact observations it was given before the
            # suspension, each naming a handle, and a store that did not survive
            # would leave every one of those citations unresolvable in the half
            # of the turn that runs in the other process.
            "result_handles": self._result_handles.to_state(),
            "conversation_history_turns": extract_turns_from_history(
                self.conversation_history
            ),
            "last_completed_turn_key": self._last_completed_turn_key,
            # The experiment labels ride the suspension blob for the same
            # reason channel_id does: a turn suspended on ask_user and resumed
            # in another process must land in the same attempt, and the labels
            # are the only thing that says which one. Absent on every state
            # written before fix-bn1, which `apply_serialized_state` reads as
            # "not part of an experiment".
            "experiment_id": self._experiment_id,
            "task_id": self._task_id,
            "attempt": self._attempt,
        }
        # No default=str round-trip. This is the first serializer, so coercing
        # here is what made every downstream strictness check vacuous: an
        # unsupported value would already be a string by the time anything
        # looked. Raising instead lets the caller keep the runtime live.
        validate_state(payload)
        return payload

    def apply_serialized_state(self, state: dict[str, Any]) -> None:
        """Restore fields from serialize_state() onto this context.

        Raises IncompatibleSessionState if the blob was written at a schema
        version this build does not read, having applied nothing.

        Schema 3 is read (arch §9.2): its suspended ReAct blob carries an
        `iteration_counter` instead of a budget, which is restored into an
        explicit `LegacyTurnBudget` and pinned to that one turn. A
        forward-version blob is refused with `preserve` set, so the caller
        leaves it on disk for the engine that wrote it instead of deleting a
        turn it cannot read.
        """
        found = state.get("schema_version", 0)
        if found not in READABLE_SCHEMA_VERSIONS:
            if isinstance(found, int) and found > SCHEMA_VERSION:
                raise IncompatibleSessionState.forward_version(found)
            raise IncompatibleSessionState(found)
        if int(found) >= 8:
            if "result_handles" not in state or not isinstance(
                state.get("result_handles"),
                list,
            ):
                raise IncompatibleSessionState(
                    f"{found} (missing or malformed result_handles)",
                    expected=SCHEMA_VERSION,
                )
            react_state = state.get("react")
            if react_state is not None:
                presentation_commands = react_state.get(
                    "presentation_commands"
                ) if isinstance(react_state, dict) else None
                if (
                    not isinstance(presentation_commands, list)
                    or any(
                        not isinstance(command, str) or not command
                        for command in presentation_commands
                    )
                ):
                    raise IncompatibleSessionState(
                        f"{found} (missing or malformed presentation_commands)",
                        expected=SCHEMA_VERSION,
                    )
        # Validated up here with the version check, not applied halfway down:
        # this method's contract is "raises having applied nothing", and a
        # malformed experiment triple must not be the one exception that leaves
        # a half-restored context behind.
        if state.get("experiment_id") is not None:
            self._validate_experiment_labels(
                state.get("experiment_id"), state.get("task_id"), state.get("attempt")
            )
        try:
            self._validate_turn_accumulator_state(
                state.get("turn"),
                schema_version=int(found),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IncompatibleSessionState(
                f"{found} (malformed turn accumulator: {exc})",
                expected=SCHEMA_VERSION,
            ) from exc

        self._awaiting_user = bool(state.get("awaiting_user"))
        self._suspended_user_message = state.get("suspended_user_message")
        self._pending_clarification_request = state.get(
            "pending_clarification_request"
        )

        self._action_log = list(state.get("action_log") or [])
        # Absent from a schema-7 blob, which restores as an empty store: the
        # agent then gets `UnknownResultHandle` and re-runs the command, which
        # is a degradation rather than a wrong answer.
        self._result_handles.apply_state(state.get("result_handles"))
        # Absent from blobs written before ruling I3 landed; a missing key just
        # means feedback has no turn to attach to until the next turn completes.
        self._last_completed_turn_key = state.get("last_completed_turn_key")
        # Absent from blobs written before fix-bn1, which reads as "not part of
        # an experiment". Restored through the same validating chokepoint the
        # live binding uses, so a hand-edited blob cannot smuggle a partial
        # triple past `[XR17]`.
        if state.get("experiment_id") is not None:
            self._bind_experiment_labels(
                state.get("experiment_id"),
                state.get("task_id"),
                state.get("attempt"),
            )

        if turns := state.get("conversation_history_turns") or []:
            from fastworkflow.conversation_history_io import restore_history_from_turns

            self._conversation_history = restore_history_from_turns(turns)

        nlu_stage = state.get("nlu_stage")
        if nlu_stage is not None:
            try:
                self._cme_workflow.context["NLU_Pipeline_Stage"] = (
                    fastworkflow.NLUPipelineStage(nlu_stage)
                )
            except (ValueError, TypeError):
                self._cme_workflow.context["NLU_Pipeline_Stage"] = nlu_stage

        self._apply_cme_continuation(state.get("cme"))
        self._apply_turn_accumulator(state.get("turn"))

        react_blob = state.get("react")
        if react_blob and self._awaiting_user:
            self._ensure_agent_initialized()
            if self._workflow_tool_agent is not None:
                try:
                    self._workflow_tool_agent.import_suspended(react_blob)
                except (KeyError, TypeError, ValueError) as e:
                    # Fail closed on malformed suspended state (arch §9.2)
                    # rather than resuming a turn on a budget nobody set. The
                    # blob is reported as unreadable, which is what it is.
                    raise IncompatibleSessionState(
                        f"{found} (malformed suspended agent state: {e})",
                        expected=SCHEMA_VERSION,
                    ) from e
                # The restored budget IS the turn's budget: this process never
                # ran _begin_turn for it.
                self._turn_budget = self._workflow_tool_agent.budget

        saved_context_name = state.get("current_command_context_name")
        if (
            saved_context_name
            and self._app_workflow
            and self._app_workflow.current_command_context is not None
            and self._app_workflow.current_command_context_name != saved_context_name
        ):
            logger.debug(
                "Command context name after rehydrate (%s) differs from saved (%s); "
                "navigation depth may not match until workflow-specific restore is added",
                self._app_workflow.current_command_context_name,
                saved_context_name,
            )

    @staticmethod
    def _validate_turn_accumulator_state(
        turn: Optional[dict[str, Any]],
        *,
        schema_version: int,
    ) -> None:
        """Validate versioned plan state before mutating a live context."""
        if turn is None:
            return
        if not isinstance(turn, dict):
            raise TypeError("turn must be an object or null")

        required_by_version = {
            5: {
                "plan",
                "plan_frontier",
                "active_leaf",
                "safety_envelope",
            },
            6: {"plan_answers"},
            7: {
                "plan_outcome",
                "plan_checkpoint_count",
                "plan_last_checkpoint_stored",
                "plan_last_checkpoint_leaf",
            },
            8: {"presented_results", "active_leaf_scope"},
        }
        required = {
            field
            for version, fields in required_by_version.items()
            if schema_version >= version
            for field in fields
        }
        missing = sorted(required - set(turn))
        if missing:
            raise ValueError(
                "schema "
                f"{schema_version} turn is missing required fields: "
                + ", ".join(missing)
            )

        for output in turn.get("outputs") or ():
            fastworkflow.CommandOutput.model_validate(output)

        agent_result = turn.get("agent_result")
        extraction_truncated_goal_ids: list[str] = []
        if agent_result is not None:
            if not isinstance(agent_result, dict):
                raise TypeError("agent_result must be an object or null")
            if agent_result.get("failure"):
                TypedFailure.model_validate(agent_result["failure"])
            if agent_result.get("provider_timeout") and not agent_result.get(
                "failure"
            ):
                raise ValueError(
                    "a provider-timeout agent_result requires its typed failure"
                )
            if (
                schema_version >= 8
                and agent_result.get("extraction_truncated")
            ):
                if agent_result.get("extraction_truncated_reason") != (
                    CODE_EXTRACTION_TRUNCATED
                ):
                    raise ValueError(
                        "an extraction-truncated result requires the typed reason"
                    )
                goal_ids = agent_result.get("extraction_truncated_goal_ids")
                if not isinstance(goal_ids, list) or any(
                    not isinstance(goal_id, str) or not goal_id
                    for goal_id in goal_ids
                ):
                    raise TypeError(
                        "extraction_truncated_goal_ids must be a list of ids"
                    )
                if len(goal_ids) != len(set(goal_ids)):
                    raise ValueError(
                        "extraction_truncated_goal_ids repeats a goal_id"
                    )
                extraction_truncated_goal_ids = goal_ids
                extraction_failure = TypedFailure.model_validate(
                    agent_result.get("extraction_failure")
                )
                if extraction_failure.code != CODE_EXTRACTION_TRUNCATED:
                    raise ValueError(
                        "extraction_failure must classify extraction truncation"
                    )

        plan_blob = turn.get("plan")
        plan = (
            PlanRecord.model_validate(plan_blob)
            if plan_blob is not None
            else None
        )
        if plan is not None:
            goal_ids = [node.goal_id for node in plan.nodes]
            if len(goal_ids) != len(set(goal_ids)):
                raise ValueError("restored plan repeats a goal_id")
            known_goal_ids = set(goal_ids)
            for node in plan.nodes:
                if (
                    node.parent_goal_id is not None
                    and node.parent_goal_id not in known_goal_ids
                ):
                    raise ValueError(
                        f"plan node {node.goal_id!r} has an unknown parent"
                    )
                unknown_prerequisites = (
                    set(node.prerequisites) - known_goal_ids
                )
                if unknown_prerequisites:
                    raise ValueError(
                        f"plan node {node.goal_id!r} has unknown prerequisites"
                    )
            unknown_truncated_goals = (
                set(extraction_truncated_goal_ids) - known_goal_ids
            )
            if unknown_truncated_goals:
                raise ValueError(
                    "extraction truncation references unknown goals: "
                    + ", ".join(sorted(unknown_truncated_goals))
                )
            if (
                schema_version >= 8
                and
                agent_result is not None
                and plan.execution is not None
                and set(plan.execution.extraction_truncated_goal_ids)
                != set(extraction_truncated_goal_ids)
            ):
                raise ValueError(
                    "agent and aggregate plan truncation goal ids disagree"
                )

        presented_results = turn.get("presented_results", [])
        if not isinstance(presented_results, list):
            raise TypeError("presented_results must be a list")
        for item in presented_results:
            if not isinstance(item, dict):
                raise TypeError("each presented result must be an object")
            if not isinstance(item.get("handles", []), list):
                raise TypeError("presented result handles must be a list")
            if not isinstance(item.get("unresolved", []), list):
                raise TypeError("presented result unresolved must be a list")
            if not isinstance(item.get("omitted_handles", []), list):
                raise TypeError("presented result omitted_handles must be a list")
            if "trimmed" in item and not isinstance(item["trimmed"], bool):
                raise TypeError("presented result trimmed must be boolean")
            if item.get("trimmed") and item.get(
                "truncation_classification"
            ) != PRESENTATION_TRUNCATION_CLASSIFICATION:
                raise ValueError(
                    "a trimmed presented result requires its infrastructure "
                    "truncation classification"
                )
            leaf_goal_id = item.get("leaf_goal_id")
            if leaf_goal_id is not None:
                leaf = plan.node(leaf_goal_id) if plan is not None else None
                if leaf is None or not leaf.is_leaf:
                    raise ValueError(
                        "presented result leaf_goal_id must name a plan leaf"
                    )

        answers = turn.get("plan_answers", [])
        if not isinstance(answers, list):
            raise TypeError("plan_answers must be a list")
        answer_ids: list[str] = []
        for item in answers:
            if not isinstance(item, dict):
                raise TypeError("each plan answer must be an object")
            goal_id = item.get("goal_id")
            answer = item.get("answer")
            if not isinstance(goal_id, str) or not goal_id:
                raise ValueError("each plan answer requires a goal_id")
            if not isinstance(answer, str) or not answer:
                raise ValueError("each plan answer requires non-empty evidence")
            answer_ids.append(goal_id)
            node = plan.node(goal_id) if plan is not None else None
            if (
                node is None
                or not node.is_leaf
                or (
                    node.status != "done"
                    and not node.command_call_ids
                )
            ):
                raise ValueError(
                    f"plan answer {goal_id!r} has no matching leaf evidence"
                )
        if len(answer_ids) != len(set(answer_ids)):
            raise ValueError("plan_answers repeats a goal_id")

        frontier = turn.get("plan_frontier", [])
        if not isinstance(frontier, list):
            raise TypeError("plan_frontier must be a list")
        if len(frontier) != len(set(frontier)):
            raise ValueError("plan_frontier repeats a goal_id")
        for goal_id in frontier:
            node = plan.node(goal_id) if plan is not None else None
            if node is None or not node.is_leaf or node.status != "not-reached":
                raise ValueError(
                    f"frontier goal {goal_id!r} is not an unreached leaf"
                )

        active_leaf_id = turn.get("active_leaf")
        if active_leaf_id is not None:
            active_leaf = (
                plan.node(active_leaf_id) if plan is not None else None
            )
            if (
                active_leaf is None
                or not active_leaf.is_leaf
                or active_leaf.status != "needs-user"
            ):
                raise ValueError(
                    "active_leaf must name the suspended needs-user leaf"
                )
        active_leaf_scope = turn.get("active_leaf_scope")
        if schema_version >= 8 and active_leaf_id is not None and active_leaf_scope is None:
            raise ValueError(
                "schema 8 active_leaf requires its binding/navigation scope"
            )
        if active_leaf_scope is not None:
            restored_scope = PlanExecutionScope.from_state(active_leaf_scope)
            if active_leaf_id is None:
                raise ValueError(
                    "active_leaf_scope requires a suspended active_leaf"
                )
            if (
                plan is not None
                and plan.execution is not None
                and restored_scope.arm.value != plan.execution.arm
            ):
                raise ValueError(
                    "active_leaf_scope arm must match plan execution metadata"
                )

        plan_outcome = turn.get("plan_outcome")
        if plan_outcome is not None:
            PlanExecutionOutcome(plan_outcome)

        checkpoint_count = turn.get("plan_checkpoint_count", 0)
        if (
            isinstance(checkpoint_count, bool)
            or not isinstance(checkpoint_count, int)
            or checkpoint_count < 0
        ):
            raise ValueError(
                "plan_checkpoint_count must be a non-negative integer"
            )
        checkpoint_stored = turn.get("plan_last_checkpoint_stored")
        if checkpoint_stored is not None and not isinstance(
            checkpoint_stored,
            bool,
        ):
            raise TypeError(
                "plan_last_checkpoint_stored must be boolean or null"
            )
        checkpoint_leaf_id = turn.get("plan_last_checkpoint_leaf")
        if checkpoint_count == 0:
            if checkpoint_leaf_id is not None or checkpoint_stored is not None:
                raise ValueError(
                    "zero checkpoints cannot name or certify a checkpoint"
                )
        else:
            checkpoint_leaf = (
                plan.node(checkpoint_leaf_id)
                if plan is not None and checkpoint_leaf_id is not None
                else None
            )
            if (
                checkpoint_leaf is None
                or not checkpoint_leaf.is_leaf
                or (
                    checkpoint_leaf.status != "done"
                    and not checkpoint_leaf.command_call_ids
                )
            ):
                raise ValueError(
                    "last checkpoint must name a leaf with durable evidence"
                )
            if not isinstance(checkpoint_stored, bool):
                raise ValueError(
                    "a checkpoint must record its synchronous-store result"
                )

        SafetyEnvelopeState.from_state(turn.get("safety_envelope"))

    def _apply_turn_accumulator(self, turn: Optional[dict[str, Any]]) -> None:
        """Restore the logical turn so resume continues it instead of starting one."""
        if not turn:
            return

        # Memory-stamp baseline (ruling I5). apply_serialized_state restores the
        # conversation history BEFORE this, so the restored length is the right
        # baseline: a resumed turn stamps only an entry it appends after resume.
        # Reconstructing rather than serializing keeps a cross-process resume
        # from stamping the PREVIOUS turn's summary onto this write-once row.
        self._turn_history_baseline = len(self.conversation_history.messages)

        self._turn_key = turn.get("key")
        # The ledger is per-process and not serialized: the pre-suspension
        # process kept its own, and those records went durable with its spans.
        # What this rebuilds is the accumulator for the commands the RESUMED
        # turn is about to run, so their outcomes can still be joined to their
        # execution records. Sink-gated exactly as _begin_turn gates it, so
        # observability-off costs nothing here either. fix-ajv.20.
        self._execution_recorder = (
            ExecutionRecorder() if tracing.get_sink(self) is not None else None
        )
        self._turn_outputs = [
            fastworkflow.CommandOutput.model_validate(o)
            for o in (turn.get("outputs") or [])
        ]
        self._turn_started_at = _parse_isoformat(turn.get("started_at"))
        self._turn_user_message = turn.get("user_message") or ""
        self._turn_refined_message = turn.get("refined_message")
        self._turn_suspended_ms = int(turn.get("suspended_ms") or 0)
        self._suspend_began_at = _parse_isoformat(turn.get("suspend_began_at"))
        self._turn_entry_workflow_name = turn.get("entry_workflow_name") or ""
        self._turn_entry_context = turn.get("entry_context") or ""

        if agent_result := turn.get("agent_result"):
            self._turn_agent_result = _RestoredAgentResult(
                exhausted=bool(agent_result.get("exhausted")),
                censored=bool(agent_result.get("censored")),
                censored_reason=agent_result.get("censored_reason"),
                provider_timeout=bool(
                    agent_result.get("provider_timeout")
                ),
                extraction_truncated=bool(
                    agent_result.get("extraction_truncated")
                ),
                extraction_truncated_reason=agent_result.get(
                    "extraction_truncated_reason"
                ),
                extraction_truncated_goal_ids=tuple(
                    agent_result.get("extraction_truncated_goal_ids") or ()
                ),
                extraction_failure=(
                    TypedFailure.model_validate(
                        agent_result["extraction_failure"]
                    )
                    if agent_result.get("extraction_failure")
                    else None
                ),
                plan_outcome=agent_result.get("plan_outcome"),
                failure=(
                    TypedFailure.model_validate(agent_result["failure"])
                    if agent_result.get("failure")
                    else None
                ),
            )
        plan_blob = turn.get("plan")
        if plan_blob:
            self._turn_plan = PlanRecord.model_validate(plan_blob)
        self._turn_plan_answers = [
            {
                "goal_id": str(item["goal_id"]),
                "answer": str(item["answer"]),
            }
            for item in (turn.get("plan_answers") or [])
            if isinstance(item, dict)
            and item.get("goal_id")
            and item.get("answer")
        ]
        self._turn_presented_results = [
            dict(item)
            for item in (turn.get("presented_results") or [])
            if isinstance(item, dict)
        ]
        self._turn_plan_frontier = list(turn.get("plan_frontier") or [])
        self._turn_active_leaf = turn.get("active_leaf")
        self._turn_leaf_scope = (
            PlanExecutionScope.from_state(turn["active_leaf_scope"])
            if turn.get("active_leaf_scope") is not None
            else None
        )
        self._turn_plan_outcome = turn.get("plan_outcome")
        self._turn_plan_checkpoint_count = int(
            turn.get("plan_checkpoint_count") or 0
        )
        self._turn_plan_last_checkpoint_stored = turn.get(
            "plan_last_checkpoint_stored"
        )
        self._turn_plan_last_checkpoint_leaf = turn.get(
            "plan_last_checkpoint_leaf"
        )
        self._turn_safety_envelope = SafetyEnvelopeState.from_state(
            turn.get("safety_envelope")
        )

    def _apply_cme_continuation(self, cme: Optional[dict[str, Any]]) -> None:
        """Restore the in-flight CME command so the next message continues it.

        stored_parameters is rebuilt through model_construct rather than
        model_validate: the saved instance was itself built that way and holds
        NOT_FOUND sentinels in the missing fields, which is precisely the state
        validation exists to reject.
        """
        if not cme:
            return

        context = self._cme_workflow.context
        command_name = cme.get("command_name")

        if cme.get("command") is not None:
            context["command"] = cme["command"]
        if command_name is not None:
            context["command_name"] = command_name

        stored = cme.get("stored_parameters")
        if stored is None or command_name is None:
            return

        params_class = self._command_parameters_class(command_name)
        if params_class is None:
            # Losing the partial parameters is bad, but resuming into a command
            # whose parameter class no longer exists is worse. Reset to intent
            # detection so the next message is routed rather than merged into a
            # command that cannot be completed.
            logger.warning(
                f"Cannot restore stored_parameters for '{command_name}': its "
                f"parameter class is gone. Resetting to intent detection."
            )
            context.pop("command", None)
            context.pop("command_name", None)
            context["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.INTENT_DETECTION
            return

        context["stored_parameters"] = params_class.model_construct(**stored)

    def _command_parameters_class(self, command_name: str):
        """The Input class for a command name, or None if it cannot be resolved."""
        if self._app_workflow is None:
            return None
        try:
            routing = fastworkflow.RoutingRegistry.get_definition(
                self._app_workflow.folderpath
            )
            return routing.get_command_class(
                command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
            )
        except Exception:
            return None

    def cancel_pending(self) -> bool:
        """
        Abort a pending ask_user clarification (Topology B).

        ask_user is non-blocking in Topology B (the clarification is returned as a
        CommandOutput), so a suspended trajectory never hangs — it simply waits in
        memory. Embedders call this to abandon it per their own session lifecycle
        (e.g. request timeout, user navigated away).

        Returns True if a pending clarification was cleared, False otherwise.
        """
        if not self._awaiting_user:
            return False
        self._reset_agent_suspension()
        self._suspend_began_at = None
        self._turn_suspended_ms = 0
        return True

    def clear_conversation_history(self) -> None:
        self._conversation_history = dspy.History(messages=[])
        # No history means no turn for feedback to attach to. Leaving the key
        # behind would let feedback given after a rotate land on a turn of the
        # conversation that was just archived (ruling I3).
        self._last_completed_turn_key = None

    def bind_last_completed_turn_key(self, turn_key: Optional[str]) -> None:
        """Point feedback at a turn this context did not run.

        Activating a stored conversation replaces the in-memory history, so the
        turn feedback belongs to is that conversation's newest usable turn, not
        whatever this process happened to run last. The embedder reads the key
        from the store (``get_last_completed_turn_key``) and installs it here.
        """
        self._last_completed_turn_key = turn_key

    def append_conversation_turn(
        self,
        conversation_summary: str,
        conversation_traces: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> None:
        """Append one turn to conversation history in the canonical 3-key shape."""

        self._conversation_history.messages.append(
            {
                "conversation summary": conversation_summary,
                "conversation_traces": conversation_traces,
                "feedback": feedback,
            }
        )

    def trim_conversation_history(self, max_turns: int) -> int:
        """Drop all but the newest ``max_turns`` turns; return how many were dropped.

        Turns are request-sized, so an unbounded history grows with every request
        on a hot channel. Only the newest few are ever read (see
        _refine_user_query). Callers that persist history must record a turn
        durably BEFORE trimming it out of memory.
        """
        messages = self._conversation_history.messages
        excess = len(messages) - max_turns
        if excess <= 0:
            return 0
        del messages[:excess]
        return excess

    def summarize_and_record_turn(
        self, message: str, actions: list, result_text: str
    ) -> tuple[str, Optional[str]]:
        """Summarize a completed agent turn and append it to conversation history.

        When there are executed actions, run LLM summarization; otherwise fall back
        to the raw message. Appends the turn via append_conversation_turn and returns
        (summary, traces) so callers can reuse them (e.g. to set an artifact).
        """
        conversation_summary = message
        conversation_traces = None
        if actions:
            conversation_summary, conversation_traces = self._extract_conversation_summary(
                message, actions, result_text
            )
        self.append_conversation_turn(conversation_summary, conversation_traces)
        return conversation_summary, conversation_traces

    def bind_app_workflow(self, workflow: fastworkflow.Workflow) -> None:
        """Bind the app workflow for NLU (Path 1) and execution (Path 2)."""
        self._app_workflow = workflow
        self._cme_workflow.context["app_workflow"] = workflow

    def _on_app_context_change(self) -> None:
        """Context-change observer: refresh the ReAct agent's available_commands."""
        from fastworkflow.workflow_agent import _refresh_agent_available_commands
        _refresh_agent_available_commands(self)

    def close(self) -> bool:
        """
        Release the cme_workflow speedict session store.

        Call when an embedder session ends; does not close the app workflow
        (caller owns that lifecycle).
        """
        listener = getattr(self, "_context_change_listener", None)
        if listener is not None and self._app_workflow is not None:
            self._app_workflow.remove_context_change_listener(listener)
            self._context_change_listener = None

        if self._cme_workflow is None:
            return True
        try:
            return self._cme_workflow.close()
        except ValueError:
            # Child cme workflows should not occur; ignore if mis-invoked.
            logger.debug("WorkflowExecutionContext.close: cme_workflow is not a root session")
            return False

    # ------------------------------------------------------------------
    # Active workflow stack (contextvar)
    # ------------------------------------------------------------------

    def get_active_workflow(self) -> Optional[fastworkflow.Workflow]:
        return active_workflow.get_active_workflow()

    def push_active_workflow(self, workflow: fastworkflow.Workflow) -> None:
        active_workflow.push_active_workflow(workflow)

    def pop_active_workflow(self) -> Optional[fastworkflow.Workflow]:
        return active_workflow.pop_active_workflow()

    def clear_workflow_stack(self) -> None:
        active_workflow.clear_workflow_stack()

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def process_turn(self, message: str) -> "fastworkflow.TurnOutput":
        """
        Execute one user message synchronously and return the public TurnOutput.

        Shares dispatch with _execute_message(); additionally captures every
        command execution of the logical turn (including ask_user exchanges) [A22]. The
        full internal TurnResult is built and projected onto the slim public
        TurnOutput (see docs/turn_result_design_final.md section 1a).
        """
        command_output = self._execute_message(message)
        turn_result = self._build_turn_result(command_output)
        return turn_result.turn_output

    @dspy_logger.observe_dspy_calls
    def _execute_message(self, message: str) -> fastworkflow.CommandOutput:
        """Shared message dispatch for _execute_message()/process_turn()."""
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before executing a message"
            )

        if not self._awaiting_user:
            # A message during suspension is the resume answer — never a reset.
            self._begin_turn(message)

        self.push_active_workflow(self._app_workflow)
        try:
            self._prepare_message_routing(message)
            if self._should_run_agent_for_message(message):
                if self._awaiting_user:
                    return self._resume_agent_message(message)
                return self._process_agent_message(message)
            return self._process_message(message)
        except CommandCancelledError as exc:
            self._reset_agent_suspension()
            return self._command_cancelled_output(str(exc))
        except Exception as exc:
            if is_provider_timeout(exc):
                return self._provider_timeout_output(exc)
            raise
        finally:
            self.pop_active_workflow()
            if self._app_workflow:
                self._app_workflow.flush()

    def _build_turn_result(
        self, command_output: fastworkflow.CommandOutput
    ) -> TurnResult:
        """Assemble the TurnResult (and its public turn_output) for the message.

        The turn's ``answer`` is plain text — the agent's final answer (or the
        deterministic command's response text). Per-command structured results
        (success/artifacts) live on ``command_outputs``.
        """
        answer = command_output.command_response.response if command_output else ""

        failure_reason: Optional[str] = None
        if self._awaiting_user:
            status = TurnStatus.AWAITING_USER
            completed_at: Optional[datetime] = None
        else:
            status = TurnStatus.COMPLETED
            completed_at = datetime.now(timezone.utc)
            if self._turn_agent_result is not None:
                plan_outcome = getattr(
                    self._turn_agent_result,
                    "plan_outcome",
                    None,
                )
                plan_outcome = getattr(plan_outcome, "value", plan_outcome)
                # A classified failure from the agent's finish phase (EXP-011,
                # arch §8.4): the extraction could not produce a final answer,
                # and the turn returns that as a typed failure carrying the
                # sealed steps rather than re-running the loop. Checked before
                # exhaustion because a turn can be both, and the specific
                # classification is the more useful of the two.
                agent_failure = getattr(self._turn_agent_result, "failure", None)
                if (
                    getattr(
                        self._turn_agent_result,
                        "provider_timeout",
                        False,
                    )
                    or plan_outcome
                    == PlanExecutionOutcome.PROVIDER_TIMEOUT.value
                ):
                    status = TurnStatus.PROVIDER_TIMEOUT
                    failure_reason = CODE_PROVIDER_TIMEOUT
                    if isinstance(agent_failure, TypedFailure):
                        self._turn_failure = agent_failure
                elif (
                    getattr(self._turn_agent_result, "censored", False)
                    or plan_outcome == PlanExecutionOutcome.CENSORED.value
                ):
                    status = TurnStatus.CENSORED
                    failure_reason = getattr(
                        self._turn_agent_result,
                        "censored_reason",
                        None,
                    )
                # ido-mn1.6.10. The harness stopped the deliverable at its
                # completion limit. `CENSORED` and not `FAILED`, because nothing
                # about the task went wrong and the answer above is real work —
                # it is the same class of outcome as the safety envelope and the
                # provider timeout, an infrastructure cutoff a run must never
                # read as the task failing. Ranked BELOW those two: a turn that
                # was censored by the envelope or lost its provider has a more
                # specific thing to say about why it stopped. Ranked ABOVE the
                # generic `agent_failure` branch, which would otherwise record a
                # turn that produced an answer as failed.
                elif getattr(
                    self._turn_agent_result, "extraction_truncated", False
                ):
                    status = TurnStatus.CENSORED
                    failure_reason = getattr(
                        self._turn_agent_result,
                        "extraction_truncated_reason",
                        None,
                    ) or CODE_EXTRACTION_TRUNCATED
                    truncation_failure = getattr(
                        self._turn_agent_result, "extraction_failure", None
                    )
                    if isinstance(truncation_failure, TypedFailure):
                        self._turn_failure = truncation_failure
                elif agent_failure is not None:
                    status = TurnStatus.FAILED
                    failure_reason = agent_failure.code
                    self._turn_failure = agent_failure
                elif plan_outcome == PlanExecutionOutcome.FAILED.value:
                    status = TurnStatus.FAILED
                    failure_reason = "plan-failed"
                elif plan_outcome in {
                    PlanExecutionOutcome.PARTIAL.value,
                    PlanExecutionOutcome.NEEDS_USER.value,
                    PlanExecutionOutcome.EXHAUSTED.value,
                    PlanExecutionOutcome.BLOCKED.value,
                }:
                    status = TurnStatus.PARTIAL
                    failure_reason = f"plan-{plan_outcome}"
                elif getattr(self._turn_agent_result, "exhausted", False):
                    # The turn failed to complete (agent ran out of iterations).
                    # status carries the failure; failure_reason elaborates it.
                    # Orthogonal to TurnOutput.success (command success codes).
                    # `max_iters_exhausted` is kept verbatim: it is a recorded
                    # shape G2A traces already carry, and renaming it would make
                    # the paired comparison compare two different labels.
                    status = TurnStatus.FAILED
                    failure_reason = "max_iters_exhausted"
            elif self._turn_outputs:
                # Deterministic/assistant path: answer text is the last captured
                # output's first response text [A33]. A command-level failure is
                # surfaced by TurnOutput.success (all command_outputs succeeded),
                # not by status/failure_reason.
                answer = self._turn_outputs[-1].command_response.response

        turn_output = fastworkflow.TurnOutput(
            turn_key=self._turn_key or mint_turn_key(),
            status=status,
            failure_reason=failure_reason,
            answer=answer,
            command_outputs=list(self._turn_outputs),
        )

        # An awaiting_user emission leaves the memory columns NULL and the
        # terminal upsert fills them (§2.1). At suspension the newest history
        # entry is not yet the turn's contribution — the resumed half of the
        # exchange has not happened — so stamping here would record a partial
        # exchange as this turn's memory.
        conversation_summary, conversation_traces = (
            (None, None) if self._awaiting_user else self._turn_memory_entry()
        )

        turn_metadata: dict[str, Any] = {}
        if self._app_workflow is not None:
            turn_metadata["workflow_folderpath"] = self._app_workflow.folderpath
        stress_mode = plan_stress_mode_from_env()
        if stress_mode:
            budget = self._turn_budget
            censored = bool(
                self._turn_safety_envelope.censored
                or (
                    self._turn_agent_result is not None
                    and getattr(self._turn_agent_result, "censored", False)
                )
            )
            censored_reason = (
                self._turn_safety_envelope.censored_reason
                or (
                    getattr(
                        self._turn_agent_result,
                        "censored_reason",
                        None,
                    )
                    if self._turn_agent_result is not None
                    else None
                )
            )
            turn_metadata["exp028_stress"] = {
                "stress_mode": True,
                "iterations": (
                    budget.iterations_consumed if budget is not None else None
                ),
                "model_calls": (
                    budget.model_calls_consumed if budget is not None else None
                ),
                "command_outputs": len(
                    [
                        output
                        for output in self._turn_outputs
                        if not output.is_ask_user
                    ]
                ),
                "safety_wall_time_limit_s": (
                    self._turn_safety_envelope.wall_time_limit_s
                ),
                "censored": censored,
                "censored_reason": censored_reason,
            }
        if self._turn_plan is not None:
            leaves = self._turn_plan.leaves
            turn_metadata["plan_runtime"] = {
                "outcome": self._turn_plan_outcome,
                "done_leaf_count": sum(
                    leaf.status == "done" for leaf in leaves
                ),
                "leaf_count": len(leaves),
                "command_evidence_count": sum(
                    len(leaf.command_call_ids) for leaf in leaves
                ),
                "answer_count": len(self._turn_plan_answers),
                "active_leaf": self._turn_active_leaf,
                "frontier": list(self._turn_plan_frontier),
                "checkpoint_count": self._turn_plan_checkpoint_count,
                "last_checkpoint_stored": (
                    self._turn_plan_last_checkpoint_stored
                ),
                "last_checkpoint_leaf": (
                    self._turn_plan_last_checkpoint_leaf
                ),
                "durable_progress": any(
                    leaf.status == "done" or bool(leaf.command_call_ids)
                    for leaf in leaves
                ),
                "extraction_truncated_goal_ids": list(
                    getattr(
                        self._turn_agent_result,
                        "extraction_truncated_goal_ids",
                        (),
                    )
                    if self._turn_agent_result is not None
                    else ()
                ),
                "extraction_truncation_failures": (
                    {
                        goal_id: failure.to_state()
                        for goal_id, failure in (
                            self._turn_plan.execution
                            .extraction_truncation_failures.items()
                        )
                    }
                    if self._turn_plan.execution is not None
                    else {}
                ),
            }
        if self._turn_presented_results:
            # The provenance of what the answer was ABLE to present. An evaluator
            # scoring a listing against the population needs to tell "the agent
            # never fetched those rows" from "the runtime trimmed them", and
            # `final_answer` alone cannot say which.
            turn_metadata["presented_results"] = list(self._turn_presented_results)
        if self._turn_failure is not None:
            turn_metadata["runtime_failure"] = self._turn_failure.to_state()

        turn_result = TurnResult(
            turn_output=turn_output,
            channel_id=self._channel_id,
            conversation_id=self._conversation_id,
            experiment_id=self._experiment_id,
            task_id=self._task_id,
            attempt=self._attempt,
            user_message=self._turn_user_message,
            refined_user_message=self._turn_refined_message,
            entry_workflow_name=self._turn_entry_workflow_name,
            entry_context=self._turn_entry_context,
            started_at=self._turn_started_at,
            completed_at=completed_at,
            suspended_ms=self._turn_suspended_ms,
            conversation_summary=conversation_summary,
            conversation_traces=conversation_traces,
            metadata=turn_metadata,
            execution_records=(
                self._execution_recorder.records()
                if self._execution_recorder is not None
                else ()
            ),
            plan=self._turn_plan,
        )

        self._finalize_turn_trace(turn_result)
        return turn_result

    def _turn_memory_entry(self) -> tuple[Optional[str], Optional[str]]:
        """The conversation-history entry THIS turn appended, or (None, None).

        The turns table carries a row for every logical turn, but only some of
        those turns correspond to a conversation-history entry: a cancelled
        turn, an abandoned suspension, or an agent turn whose history never
        grew has nothing to contribute to memory. Stamping the newest entry
        unconditionally would attribute the previous turn's summary to this
        row, and the row is write-once — so the growth guard is what keeps the
        memory columns honest (§2.1, ruling I5).

        The history is read through the property because distillation replaces
        the object wholesale (and truncates it back to a pre-pass length, which
        is why the comparison is `>` rather than `!=`).
        """
        messages = self.conversation_history.messages
        if len(messages) <= self._turn_history_baseline:
            return None, None
        newest = messages[-1]
        if not isinstance(newest, dict):
            return None, None
        return (
            newest.get("conversation summary"),
            newest.get("conversation_traces"),
        )

    def _compute_context_mutations(self) -> Optional[dict]:
        """Shallow diff of the app workflow's context against the _begin_turn
        snapshot: {added, removed, changed} with repr-capped values, or None
        when nothing changed / no snapshot exists. Never raises — this feeds a
        span attribute and must not affect the turn."""
        snapshot = getattr(self, "_turn_context_snapshot", None)
        if snapshot is None or self._app_workflow is None:
            return None
        try:
            return self._context_mutations_diff(snapshot)
        except Exception:
            # App-authored context can hold anything (uncomparable keys,
            # exploding __eq__) — a diagnostic diff must never fail the turn.
            return None

    def _context_mutations_diff(self, snapshot: dict) -> Optional[dict]:
        current = dict(self._app_workflow.context)

        def brief(value: Any) -> str:
            try:
                return repr(value)[:200]
            except Exception:
                return f"<{type(value).__name__}>"

        mutations: dict = {}
        if added := {
            key: brief(value) for key, value in current.items() if key not in snapshot
        }:
            mutations["added"] = added
        # key=repr: context keys are app-authored and need not be mutually
        # comparable (a str key beside an int key would make plain sorted()
        # raise TypeError).
        if removed := sorted(
            (key for key in snapshot if key not in current), key=repr
        ):
            mutations["removed"] = removed
        changed = {}
        for key, old_value in snapshot.items():
            if key not in current:
                continue
            new_value = current[key]
            try:
                differs = new_value is not old_value and new_value != old_value
            except Exception:
                differs = True  # incomparable values: report, don't hide
            if differs:
                changed[key] = {"from": brief(old_value), "to": brief(new_value)}
        if changed:
            mutations["changed"] = changed
        return mutations or None

    def _finalize_turn_trace(self, turn_result: TurnResult) -> None:
        """Emit the fw.turn root span update/close, the turn record, and turn
        metrics at the finalize chokepoint. Never raises (tracing helpers and
        safe_* wrappers swallow sink failures).

        On AWAITING_USER the root span is updated in place (still open) and
        the record is emitted so the suspended turn is visible ([R2]); the
        terminal finalize closes the same deterministic span id ([R6]) —
        including after a cross-process resume, where the in-memory span
        object is rebuilt from the restored accumulator.
        """
        turn_output = turn_result.turn_output
        status = turn_output.status

        root = self._turn_root_span
        if root is None and self._turn_key and tracing.get_sink(self) is not None:
            root = tracing.Span(
                span_id=tracing.root_span_id(self._turn_key),
                trace_id=self._turn_key,
                name=tracing.SPAN_TURN,
                channel_id=self._channel_id,
                context=self._turn_entry_context or None,
                start_ns=tracing.datetime_to_ns(self._turn_started_at) or 0,
                attributes={
                    "turn_key": self._turn_key,
                    "channel_id": self._channel_id,
                    "conversation_id": self._conversation_id,
                    "user_message": tracing.cap_attr_value(self._turn_user_message),
                },
            )
            self._turn_root_span = root

        awaiting = status == TurnStatus.AWAITING_USER
        tracing.end_span(
            self,
            root,
            status=status.value,
            close=not awaiting,
            attributes={
                "status": status.value,
                "success": turn_output.success,
                "failure_reason": turn_output.failure_reason,
                "suspended_ms": turn_result.suspended_ms,
                "context_mutations": self._compute_context_mutations(),
            },
        )
        if not awaiting:
            self._turn_root_span = None
            self._turn_context_snapshot = None

        self._last_turn_record_stored = tracing.emit_turn_record(self, turn_result)
        self._last_turn_added_memory = (
            not awaiting and turn_result.conversation_summary is not None
        )
        if self._last_turn_added_memory:
            # Only a turn that contributed a memory entry can carry feedback:
            # the memory window filters on exactly that, so keying feedback to
            # any other row would file it where no reader joins it (I3/I4).
            self._last_completed_turn_key = turn_result.turn_output.turn_key

        if turn_result.completed_at is not None:
            metrics.safe_increment(
                self._metrics_sink, "fw_turns_total", status=status.value
            )
            if turn_result.started_at is not None:
                metrics.safe_observe(
                    self._metrics_sink,
                    "fw_turn_duration_seconds",
                    (
                        turn_result.completed_at - turn_result.started_at
                    ).total_seconds(),
                    status=status.value,
                )

    def finalize_turn_for_observability(
        self, command_output: Optional[fastworkflow.CommandOutput]
    ) -> None:
        """Run the finalize chokepoint for a turn driven outside process_turn.

        The CLI chassis (ChatSession loop) dispatches via _execute_message /
        process_action directly — its transport is the queues — so nothing
        else builds the TurnResult for its turns. This emits the root-span
        close, turn record, and metrics; the TurnResult itself is discarded.
        No-op when no logical turn is open, or mid-suspension (Topology A
        blocks through ask_user, so a completed _execute_message is a
        completed turn).
        """
        if self._turn_key is None or self._awaiting_user:
            return
        if tracing.get_sink(self) is None and isinstance(
            self._metrics_sink, metrics.NoOpMetricsSink
        ):
            # Observability fully off: nothing consumes the TurnResult, so
            # skip building it — this path runs after EVERY CLI turn and must
            # cost ~nothing when FW_OBSERVABILITY=0.
            return
        self._build_turn_result(command_output)

    def process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before process_action()"
            )

        # Each direct action is its own logical turn [A30].
        self._begin_turn(action.command_name or "")

        self.push_active_workflow(self._app_workflow)
        try:
            return self._process_action(action)
        finally:
            self.pop_active_workflow()
            if self._app_workflow:
                self._app_workflow.flush()

    def process_action_turn(
        self, action: fastworkflow.Action
    ) -> "fastworkflow.TurnOutput":
        """
        Execute one direct action synchronously and return the public TurnOutput.

        Mirror of process_turn() for the direct-action path: same dispatch as
        process_action() (each direct action is its own logical turn [A30]),
        additionally building the full internal TurnResult and projecting it onto
        the slim public TurnOutput. This lets callers (e.g. the run_fastapi_mcp
        turn registry) store exactly one result type across both the message and
        action paths.
        """
        command_output = self.process_action(action)
        turn_result = self._build_turn_result(command_output)
        return turn_result.turn_output

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _prepare_message_routing(self, message: str) -> None:
        if (
            (
                "NLU_Pipeline_Stage" not in self._cme_workflow.context
                or self._cme_workflow.context["NLU_Pipeline_Stage"]
                == fastworkflow.NLUPipelineStage.INTENT_DETECTION
            )
            and message.startswith("/")
        ):
            self._cme_workflow.context["is_assistant_mode_command"] = True

    def _should_run_agent_for_message(self, message: str) -> bool:
        """Agent path unless assistant-mode '/' command flag is set."""
        return (
            self._run_as_agent
            and "is_assistant_mode_command" not in self._cme_workflow.context
        )

    def _command_cancelled_output(self, reason: str) -> fastworkflow.CommandOutput:
        # sourcery skip: class-extract-method
        command_response = fastworkflow.CommandResponse(
            response=f"Command cancelled: {reason}",
            success=False,
        )
        command_output = fastworkflow.CommandOutput(
            command_response=command_response
        )
        if self._app_workflow:
            command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]
        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()
        return command_output

    def _provider_timeout_output(
        self,
        exc: BaseException,
    ) -> fastworkflow.CommandOutput:
        """Terminalize a provider timeout while preserving prior plan evidence."""
        completed_work = ()
        if self._turn_plan is not None:
            active_leaf = (
                self._turn_plan.node(self._turn_active_leaf)
                if self._turn_active_leaf is not None
                else None
            )
            if (
                active_leaf is not None
                and active_leaf.status != "done"
            ):
                active_leaf.status = "blocked"
                active_leaf.failure_reason = CODE_PROVIDER_TIMEOUT
                reconcile_plan_statuses(self._turn_plan)
                self._checkpoint_plan_progress(
                    self._turn_plan,
                    active_leaf,
                    CODE_PROVIDER_TIMEOUT,
                )
            else:
                reconcile_plan_statuses(self._turn_plan)
            completed_work = tuple(
                {
                    "goal_id": leaf.goal_id,
                    "status": leaf.status,
                    "command_call_ids": leaf.command_call_ids,
                }
                for leaf in self._turn_plan.leaves
                if leaf.status == "done" or leaf.command_call_ids
            )
            self._turn_plan_outcome = (
                PlanExecutionOutcome.PROVIDER_TIMEOUT.value
            )
        failure = provider_timeout_failure(
            exc,
            completed_work=completed_work,
        )
        preserved = (
            self._compose_plan_answer(render_account(self._turn_plan))
            if self._turn_plan is not None
            else ""
        )
        timeout_text = (
            "The model provider timed out. This is an infrastructure outcome, "
            "not a task failure or safety censor."
        )
        if completed_work:
            timeout_text += " Prior plan progress remains recorded."
        response = (
            f"{preserved}\n\n{timeout_text}"
            if preserved
            else timeout_text
        )
        self._turn_failure = failure
        self._turn_agent_result = SimpleNamespace(
            final_answer=response,
            exhausted=False,
            censored=False,
            provider_timeout=True,
            plan_outcome=self._turn_plan_outcome,
            failure=failure,
        )
        command_output = fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(
                response=response,
                success=False,
            )
        )
        if self._app_workflow:
            command_output.workflow_name = (
                self._app_workflow.folderpath.split("/")[-1]
            )
        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()
        return command_output

    def _maybe_enqueue_output(self, command_output: fastworkflow.CommandOutput) -> None:
        if (
            (not command_output.success or self._keep_alive)
            and self._command_output_queue is not None
        ):
            self._command_output_queue.put(command_output)

    def _maybe_enqueue_trace_sentinel(self) -> None:
        if self._command_trace_queue is not None:
            self._command_trace_queue.put(None)

    # ------------------------------------------------------------------
    # Agent mode
    # ------------------------------------------------------------------

    def _initialize_agent_functionality(self) -> None:
        self._cme_workflow.context["run_as_agent"] = True
        if self._app_workflow:
            self._app_workflow.context["run_as_agent"] = True

        # Load workflow-specific insights for insights distillation (if present).
        # These enhance the agent + planner signatures; absent files -> None (no-op).
        from fastworkflow.utils.insights_loader import load_workflow_insights
        if self._app_workflow:
            self._planning_insights = load_workflow_insights(
                self._app_workflow.folderpath, "planning_agent"
            )
            self._execution_insights = load_workflow_insights(
                self._app_workflow.folderpath, "execution_agent"
            )

        from fastworkflow.workflow_agent import initialize_workflow_tool_agent
        self._workflow_tool_agent = initialize_workflow_tool_agent(
            self, execution_insights=self._execution_insights
        )

        # Re-scope the active ReAct agent's available_commands whenever the context changes,
        # driven by the workflow's context-change observer (the single switch chokepoint)
        # Registered once per WEC (agent init runs once). The listener reads the *active* agent
        # dynamically. No-ops when no agent is running. Bound method + remove on close()
        # (and WeakMethod storage on Workflow) avoid listener leaks.
        if self._app_workflow is not None:
            self._app_workflow.add_context_change_listener(self._on_app_context_change)
            self._context_change_listener = self._on_app_context_change

        from fastworkflow.intent_clarification_agent import initialize_intent_clarification_agent
        self._intent_clarification_agent = initialize_intent_clarification_agent(self)

    def _ensure_agent_initialized(self) -> None:
        if self._workflow_tool_agent is None:
            self._initialize_agent_functionality()

    def _reset_agent_suspension(self) -> None:
        """Clear Topology-B ask_user suspend state (abort, finalize, or cancel_pending)."""
        self._awaiting_user = False
        self._suspended_user_message = None
        self._pending_clarification_request = None
        self._turn_leaf_scope = None
        if self._workflow_tool_agent is not None and hasattr(
            self._workflow_tool_agent, "clear_suspension"
        ):
            self._workflow_tool_agent.clear_suspension()

    def _agent_dspy_context(self):
        """Return (lm, adapter) for agent-mode dspy.context blocks."""
        lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
        from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter

        return lm, CommandsSystemPreludeAdapter()

    def _remaining_turn_deadline_seconds(self) -> float:
        """What is left of this logical turn's wall deadline, in seconds.

        The deadline is `FW_TURN_DEADLINE_SECONDS`, resolved by
        `external_operations` so the in-turn clamp and the server's watchdog
        cannot hold two different numbers (ido-mn1.6.33).

        The anchor is the safety envelope's own start, because that is the one
        timestamp taken when the logical turn began; without it a planned turn
        would hand every leaf a fresh full deadline and the bound would apply
        to each leaf rather than to the turn. When no envelope is running --
        ordinary, non-stress deployments -- there is nothing to anchor to and
        the full deadline is used, which is still a bound where there was none.

        Never returns zero or less: an already-overrun turn gets a floor rather
        than a deadline in the past, so the failure it produces comes from the
        call that has no time rather than from a context that refuses before
        anything is attempted.
        """
        deadline = external_operations.resolve_turn_deadline_seconds()
        envelope = getattr(self, "_turn_safety_envelope", None)
        started = float(getattr(envelope, "started_at_epoch_s", 0.0) or 0.0)
        if envelope is not None and getattr(envelope, "enabled", False) and started > 0:
            deadline -= max(0.0, time.time() - started)
        return max(1.0, deadline)

    def _call_agent(self, agent_call, lm=None, *, trace_input=None,
                    resumed=False, presentation_commands=None):
        """Run agent_call once, under an agent dspy.context.

        lm: optional LM override (e.g. distillation's teacher/student model). When
        omitted, the default agent context (LLM_AGENT) is used.

        This is the one choke point both the fresh forward and the resume pass
        through, so it is where the executor phase is recorded: fw.agent.execute
        wraps the call, and ``host_scope`` binds this context so ReAct's
        per-iteration fw.agent.step spans — several frames down, with no
        reference to the WEC — reach the same sink ([R28]).

        **No retry here (EXP-011, arch §8.4).** This used to re-invoke the whole
        agent on an ``AdapterParseError``, which re-executed every tool call the
        failed attempt had already completed — the whole-agent replay FW-REQ-008B
        clause 3 forbids. The recovery it was buying has not been dropped; it
        moved to the phase that owns it, where nothing has executed yet:
        ``fastWorkflowReAct._decide`` retries the decision parse, and
        ``_finish`` retries the extraction against a sealed snapshot.

        ``attempts`` stays on the span, now always 1, because the attribute is
        part of a recorded shape and a reader comparing G2A traces to later ones
        needs the field to exist in both.

        ``presentation_commands`` is the executing skill's optional ``presents:``
        OVERRIDE (ido-mn1.6.6), applied HERE rather than at each call site so the
        flat turn, the shadow turn and every plan leaf reach the extraction-time
        resolver by one road. Empty on the flat paths is not a gap: the DEFAULT
        rule is the producing command's own ``presentation`` flag, which the
        resolver reads off the stored handle in every arm. That is where it has
        to live — with ``FW_PLAN_DECOMPOSITION=off`` the loader never opens
        ``_skills/``, so a skill-only rule would have given the control arm less
        mechanism than the treatment arms and the endpoint would have scored the
        difference as an effect of decomposition.

        ``None`` means LEAVE IT ALONE, which is what the resume path wants: a
        resumed run continues the same leaf of the same logical turn, and
        clearing the override on the way back in would drop the second half of
        that leaf's answer from the presentation rule its first half had.
        """
        default_lm, agent_adapter = self._agent_dspy_context()
        if lm is None:
            lm = default_lm
        turn_seconds = self._remaining_turn_deadline_seconds()
        span = tracing.start_span(
            self,
            tracing.SPAN_AGENT_EXECUTE,
            attributes={
                "agent_input": trace_input,
                "resumed": resumed,
                "model": getattr(lm, "model", None),
            },
        )
        attempts = 1
        agent = self._workflow_tool_agent
        if presentation_commands is not None and agent is not None:
            agent.presentation_commands = frozenset(presentation_commands)
        try:
            with tracing.host_scope(self):
                # ido-mn1.6.33. THE turn-level deadline, opened once around the
                # whole agent run. Until this existed there was no operation in
                # force on this path at all, so `extraction_bound()`'s
                # `clamp_timeout()` had nothing to clamp against and
                # `timeout_clamped` was decorative: the extraction could derive
                # an 802 s bound, take it three times across parse attempts,
                # and outlive the watchdog that was supposed to bound the turn.
                #
                # `operation` nests to the INNER deadline, so the per-call
                # classes underneath (`model.agent` 300 s for a react step,
                # `backend.read` 60 s for a tool) are unchanged; what changes is
                # that none of them, and no sum of them, can now outlive the
                # turn. Opened per agent call rather than per turn because a
                # planned turn calls the agent once per leaf, and the anchor is
                # the turn's own start either way.
                with external_operations.operation(
                    "model.agent", seconds=turn_seconds
                ):
                    with dspy.context(lm=lm, adapter=agent_adapter):
                        result = agent_call()
                self._remember_presented_results(result)
                tracing.end_span(
                    self, span, attributes=_agent_result_attributes(result, attempts)
                )
                return result
        except BaseException as exc:
            # CommandCancelledError/AskUserSuspend are control signals, not
            # failures; either way the span must close rather than leak onto
            # the parenting stack for the rest of the turn.
            tracing.end_span(
                self,
                span,
                status=tracing.status_for_dispatch_exception(exc),
                attributes={"attempts": attempts, "error_type": type(exc).__name__},
            )
            raise

    def _presentation_commands_for(self, node: Any) -> frozenset[str]:
        """The `presents:` OVERRIDE governing one leaf, or none.

        Empty is not "present nothing": it hands the decision back to the
        producing command's own `presentation` flag, which is the arm-invariant
        default and the only rule arm A can read at all. A skill declares
        `presents:` only to narrow that default or to add a command the producer
        did not flag.

        Walks the leaf UP its parent chain and unions what each skill declares,
        because the leaf that actually runs is often a command sequence with no
        skill of its own (`PlanNode.skill` is None for one) while the skill whose
        answer presents the result is its parent. Taking only the leaf's own
        declaration would make `presents:` work on skills whose bodies happen to
        have a leading skill step and silently not on the rest.

        Never raises and returns the empty set for a missing plan or node: this
        selects what MAY be resolved, and a selection failure must cost an answer
        some rows, not a turn.
        """
        plan = self._turn_plan
        if plan is None or node is None:
            return frozenset()
        commands: set[str] = set()
        seen: set[str] = set()
        current = node
        while current is not None and current.goal_id not in seen:
            seen.add(current.goal_id)
            skill = self._presents_source(plan, current.skill)
            if skill is not None:
                commands.update(skill.presents)
            parent_id = current.parent_goal_id
            current = plan.node(parent_id) if parent_id else None
        return frozenset(commands)

    def _presents_source(self, plan: Any, skill_name: Optional[str]):
        """The catalogue entry for `skill_name`, or None when it cannot be read."""
        if not skill_name:
            return None
        try:
            return self._catalog_for_plan_execution(plan).get(skill_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"could not read presents: for skill {skill_name!r}: {exc!r}"
            )
            return None

    def _remember_presented_results(self, result: Any) -> None:
        """Accumulate one agent call's presentation resolution for the turn record.

        One entry per `fw.agent.execute`, so an Arm B/C turn records what each
        leaf presented rather than one collapsed total: the leaves are composed
        by concatenation, and a reader asking why one section of the answer is
        short needs the leaf that produced it, not the turn.
        """
        presented = getattr(result, "presented_results", None)
        if isinstance(presented, Mapping) and (
            presented.get("handles")
            or presented.get("unresolved")
            or presented.get("omitted_handles")
            or presented.get("trimmed")
        ):
            self._turn_presented_results.append(
                {"leaf_goal_id": self._turn_active_leaf, **dict(presented)}
            )

    def _run_agent(self, message: str):
        """Fresh agent turn setup and ReAct forward call."""
        self.clear_action_log()

        if self._app_workflow:
            self._app_workflow.context["raw_user_message"] = message

        budget = self._require_turn_budget()

        with external_operations.operation("model.summarization"):
            refined_user_query = self._refine_user_query(
                message, self.conversation_history
            )
        self._turn_refined_message = refined_user_query

        # Query refinement above is deterministic string assembly. Model-call
        # accounting starts at the planner/selector calls below, so Arm A and
        # Arms B/C do not each receive one phantom call before their real one.

        from fastworkflow.workflow_agent import (
            _plan_decomposition_point,
            _what_can_i_do,
            build_query_with_next_steps,
            select_skills,
        )

        mode, catalog = _plan_decomposition_point(self)
        has_history = bool(self.conversation_history.messages)

        if mode is PlanMode.OFF:
            with external_operations.operation("model.planner"):
                command_info_and_refined_message_with_todolist = build_query_with_next_steps(
                    refined_user_query,
                    self,
                    with_agent_inputs_and_trajectory=has_history,
                    planning_insights=self._planning_insights,
                    planner_lm=getattr(self, "_current_planner_lm", None),
                )
            available_commands = _what_can_i_do(self)
            budget.consume_model_call()

            return self._call_agent(
                lambda: self._workflow_tool_agent(
                    user_query=command_info_and_refined_message_with_todolist,
                    available_commands=available_commands,
                    budget=self._require_turn_budget(),
                    safety_envelope=self._turn_safety_envelope,
                ),
                trace_input=command_info_and_refined_message_with_todolist,
                # Arm A: `off` never opens `_skills/`, so there is no skill to
                # override with. The resolver still runs the SAME two rules —
                # citation, and the producing command's `presentation` flag —
                # which is what keeps the control arm's mechanism equal to the
                # treatment arms' (ido-mn1.6.6).
                presentation_commands=self._presentation_commands_for(None),
            )

        with external_operations.operation("model.planner"):
            selected = select_skills(
                refined_user_query,
                catalog,
                getattr(self, "_current_planner_lm", None),
            )
        selector_provider_calls = int(
            getattr(selected, "provider_call_count", 1) or 1
        )
        selector_provider_responses = int(
            getattr(selected, "provider_response_count", 0) or 0
        )
        budget.consume_model_call(selector_provider_calls)
        compile_span = tracing.start_span(
            self,
            tracing.SPAN_PLAN_COMPILE,
            attributes={
                "mode": mode.value,
                "selector_application_attempts": int(
                    getattr(selected, "application_attempt_count", 0) or 0
                ),
                "selector_provider_calls": selector_provider_calls,
                "selector_provider_responses": selector_provider_responses,
                "selector_adapter_identity": getattr(
                    selected,
                    "adapter_identity",
                    None,
                ),
            },
        )
        try:
            plan = expand(
                catalog,
                selected,
                refined_user_query,
                mode=mode,
                selection_model=getattr(selected, "selection_model", None),
            )
            tracing.end_span(
                self,
                compile_span,
                attributes={
                    "requested_public_task_keys": list(plan.requested_public_task_keys),
                    "compiled_public_task_keys": list(plan.compiled_public_task_keys),
                    "public_node_count": len(plan.public_nodes),
                    "leaf_count": len(plan.leaves),
                    "executable_leaf_count": len(
                        [leaf for leaf in plan.leaves if leaf.executable]
                    ),
                    "edge_count": len(plan.edges),
                    "packing_candidate_count": plan.packing.candidate_count,
                    "composite_group_count": (
                        plan.packing.selected_root_group_count
                    ),
                    "recursive_composite_group_count": (
                        plan.packing.selected_recursive_group_count
                    ),
                    "packed_task_count": plan.packing.packed_task_count,
                    "packing_edge_count": (
                        plan.packing.orchestration_edge_count
                    ),
                    "packing_sha256": plan.packing.packing_sha256,
                },
            )
        except BaseException as exc:
            tracing.end_span(
                self,
                compile_span,
                status=tracing.status_for_dispatch_exception(exc),
                attributes={"error_type": type(exc).__name__},
            )
            raise
        self._turn_plan = plan
        self._turn_plan_frontier = [
            node.goal_id for node in plan.nodes if node.is_leaf and node.status == "not-reached"
        ]

        if mode is PlanMode.SHADOW:
            with external_operations.operation("model.planner"):
                command_info_and_refined_message_with_todolist = build_query_with_next_steps(
                    refined_user_query,
                    self,
                    with_agent_inputs_and_trajectory=has_history,
                    planning_insights=self._planning_insights,
                    planner_lm=getattr(self, "_current_planner_lm", None),
                )
            available_commands = _what_can_i_do(self)
            budget.consume_model_call()
            return self._call_agent(
                lambda: self._workflow_tool_agent(
                    user_query=command_info_and_refined_message_with_todolist,
                    available_commands=available_commands,
                    budget=self._require_turn_budget(),
                    safety_envelope=self._turn_safety_envelope,
                ),
                trace_input=command_info_and_refined_message_with_todolist,
                # Shadow compiles a plan and then runs FLAT, so it must reach the
                # resolver exactly as `off` does or it stops being `off`'s
                # control. `None` here is that equality, not an oversight.
                presentation_commands=self._presentation_commands_for(None),
            )

        arm = plan_execution_arm_from_env()
        if arm is PlanExecutionArm.A:
            raise PlanConfigurationError(
                "FW_PLAN_EXECUTION_ARM=a requires FW_PLAN_DECOMPOSITION=off; "
                "Arm A is the flat planner and cannot execute a compiled plan"
            )

        return self._execute_compiled_plan(plan, arm)

    def _remember_plan_answer(self, goal_id: str, result: Any) -> None:
        """Keep one final leaf answer so suspension cannot erase prior work."""
        if (
            getattr(result, "failure", None) is not None
            or getattr(result, "censored", False)
            or getattr(result, "provider_timeout", False)
        ):
            return
        answer = (
            getattr(result, "final_answer", None)
            or getattr(result, "answer", None)
        )
        if not answer or getattr(result, "suspended", False):
            return
        node = (
            self._turn_plan.node(goal_id)
            if self._turn_plan is not None
            else None
        )
        if node is None or not node.is_leaf or not node.command_call_ids:
            # A model answer without command evidence is not leaf evidence.
            # execute_plan will classify that leaf as blocked; retaining the
            # prose here would let aggregation imply unsupported progress.
            return
        replacement = {"goal_id": goal_id, "answer": str(answer)}
        for index, item in enumerate(self._turn_plan_answers):
            if item["goal_id"] == goal_id:
                self._turn_plan_answers[index] = replacement
                return
        self._turn_plan_answers.append(replacement)

    def _compose_plan_answer(self, account: str) -> str:
        """Compose leaf evidence without promoting it to contract success."""
        sections = []
        if self._turn_plan_answers and self._turn_plan is None:
            raise PlanConfigurationError(
                "plan answers cannot be rendered without their plan"
            )
        leaf_positions = {
            leaf.goal_id: index
            for index, leaf in enumerate(
                self._turn_plan.leaves if self._turn_plan is not None else (),
                start=1,
            )
        }
        for item in self._turn_plan_answers:
            node = self._turn_plan.node(item["goal_id"])
            if node is None or not node.is_leaf:
                raise PlanConfigurationError(
                    "plan answer references a missing or non-leaf goal: "
                    f"{item['goal_id']!r}"
                )
            position = leaf_positions[node.goal_id]
            sections.append(
                f"Leaf {position} execution evidence "
                f"(runtime status: {node.status}; not independent contract "
                f"verification) — {node.goal_text}:\n{item['answer']}"
            )
        if account:
            sections.append(
                "Runtime execution account "
                "(does not certify contract success):\n"
                f"{account}"
            )
        return "\n\n".join(sections)

    def _catalog_for_plan_execution(self, plan: Any):
        """Restore the exact catalogue needed for delayed deterministic binding."""
        catalog = getattr(plan, "_catalog", None)
        if catalog is None:
            if self._app_workflow is None:
                raise PlanConfigurationError(
                    "cannot restore delayed plan bindings without a bound workflow"
                )
            manifest = merge_and_gate(
                load_manifest(self._app_workflow.folderpath)
            )
            catalog = load_skill_catalog(
                self._app_workflow.folderpath,
                manifest,
            )
            if getattr(catalog, "fingerprint", None) != plan.skills_fingerprint:
                raise PlanConfigurationError(
                    "restored plan catalogue fingerprint differs from the "
                    "catalogue that compiled it"
                )
            plan._catalog = catalog
        return catalog

    @staticmethod
    def _node_descends_from(plan: Any, node: Any, ancestor_id: str) -> bool:
        current = node
        while current is not None:
            if current.goal_id == ancestor_id:
                return True
            current = (
                plan.node(current.parent_goal_id)
                if current.parent_goal_id is not None
                else None
            )
        return False

    def _capture_outputs_for_node(self, plan: Any, node: Any) -> tuple[Any, ...]:
        """Successful outputs from this node's completed producers, in order."""
        producer_call_ids: set[str] = set()
        for prerequisite_id in node.prerequisites:
            predecessor = plan.node(prerequisite_id)
            if predecessor is None or predecessor.status != "done":
                continue
            for candidate in plan.nodes:
                if self._node_descends_from(
                    plan,
                    candidate,
                    prerequisite_id,
                ):
                    producer_call_ids.update(candidate.command_call_ids)
        if not producer_call_ids:
            return ()
        return tuple(
            output
            for output in self._turn_outputs
            if output.success
            and output.command_call_id in producer_call_ids
        )

    def _resolve_delayed_plan_bindings(self, plan: Any) -> bool:
        """Bind captured handles from completed command evidence, if available."""
        pending = tuple(
            node
            for node in plan.nodes
            if node.parent_goal_id is not None
            and node.status == "needs-user"
            and node.unbound_slots()
        )
        if not pending:
            return False
        catalog = self._catalog_for_plan_execution(plan)
        changed = False
        while True:
            pass_changed = False
            for node in tuple(plan.nodes):
                if (
                    node.parent_goal_id is None
                    or node.status != "needs-user"
                ):
                    continue
                for slot_name in node.unbound_slots():
                    producer_outputs = self._capture_outputs_for_node(
                        plan,
                        node,
                    )
                    if not producer_outputs:
                        continue
                    task_key = node.task_key
                    binding = bind_captured(
                        plan,
                        node,
                        slot_name,
                        producer_outputs,
                        slot=slot_name,
                        catalog=catalog,
                    )
                    if binding is None:
                        continue
                    # Runtime-captured values complete execution inputs; they do
                    # not rewrite the public task identity frozen at compile.
                    node.task_key = task_key
                    pass_changed = True
                    changed = True
            if not pass_changed:
                break
        return changed

    def _checkpoint_plan_progress(
        self,
        plan: Any,
        leaf: Any,
        reason: str,
    ) -> None:
        """Persist progress before another provider call can obscure it."""
        if reason in {
            "blocked",
            "censored",
            "failed",
            "needs-user",
            "provider-timeout",
        }:
            self._turn_plan_answers = [
                item
                for item in self._turn_plan_answers
                if item["goal_id"] != leaf.goal_id
            ]
        if not (
            leaf.status == "done"
            or bool(leaf.command_call_ids)
        ):
            return
        self._turn_plan = plan
        self._turn_plan_frontier = [
            node.goal_id
            for node in plan.leaves
            if node.status == "not-reached"
        ]
        self._turn_plan_checkpoint_count += 1
        self._turn_plan_last_checkpoint_leaf = leaf.goal_id
        budget = self._turn_budget
        checkpoint = TurnResult(
            turn_output=fastworkflow.TurnOutput(
                turn_key=self._turn_key or mint_turn_key(),
                status=TurnStatus.IN_PROGRESS,
                answer=self._compose_plan_answer(render_account(plan)),
                command_outputs=list(self._turn_outputs),
            ),
            channel_id=self._channel_id,
            conversation_id=self._conversation_id,
            experiment_id=self._experiment_id,
            task_id=self._task_id,
            attempt=self._attempt,
            user_message=self._turn_user_message,
            refined_user_message=self._turn_refined_message,
            entry_workflow_name=self._turn_entry_workflow_name,
            entry_context=self._turn_entry_context,
            started_at=self._turn_started_at,
            completed_at=None,
            suspended_ms=self._turn_suspended_ms,
            metadata={
                "plan_checkpoint": {
                    "sequence": self._turn_plan_checkpoint_count,
                    "leaf_goal_id": leaf.goal_id,
                    "leaf_status": leaf.status,
                    "reason": reason,
                    "done_leaf_count": sum(
                        node.status == "done" for node in plan.leaves
                    ),
                    "leaf_count": len(plan.leaves),
                    "command_evidence_count": sum(
                        len(node.command_call_ids) for node in plan.leaves
                    ),
                    "answer_count": len(self._turn_plan_answers),
                    "iterations": (
                        budget.iterations_consumed
                        if budget is not None
                        else None
                    ),
                    "model_calls": (
                        budget.model_calls_consumed
                        if budget is not None
                        else None
                    ),
                }
            },
            execution_records=(
                self._execution_recorder.records()
                if self._execution_recorder is not None
                else ()
            ),
            plan=plan,
        )
        self._turn_plan_last_checkpoint_stored = tracing.emit_turn_record(
            self,
            checkpoint,
        )

    def _execute_compiled_plan(
        self,
        plan: Any,
        arm: PlanExecutionArm,
        *,
        resumed_leaf_result: Any = None,
    ) -> Any:
        """Execute or resume a compiled B/C plan through one shared chokepoint."""
        from fastworkflow.workflow_agent import _what_can_i_do

        # Restored plans do not serialize their catalogue object. Reload it
        # before scope construction so versioned slot resolvers and typed task
        # bindings reach the same leaf envelope as a fresh plan.
        self._catalog_for_plan_execution(plan)
        resumed_goal_id = self._turn_active_leaf if resumed_leaf_result is not None else None

        def _execute_leaf(node, scope: PlanExecutionScope):
            self._turn_active_leaf = node.goal_id
            available_commands = _what_can_i_do(self)
            agent_input = render_leaf_instruction(node, scope)
            outputs_before = len(self._turn_outputs)
            leaf_result = None
            previous_scope = getattr(self, "_turn_leaf_scope", None)
            self._turn_leaf_scope = scope
            try:
                leaf_result = self._call_agent(
                    lambda: self._workflow_tool_agent(
                        user_query=agent_input,
                        available_commands=available_commands,
                        budget=self._require_turn_budget(),
                        safety_envelope=self._turn_safety_envelope,
                    ),
                    trace_input=agent_input,
                    # Arms B and C: one leaf, one executing skill chain, one
                    # `presents:` union — an OVERRIDE of the command flag the two
                    # flat sites leave standing. The same parameter on the same
                    # road, so the resolver is arm-invariant by construction
                    # rather than by three agreeing copies.
                    presentation_commands=self._presentation_commands_for(node),
                )
            finally:
                self._turn_leaf_scope = (
                    scope
                    if leaf_result is not None
                    and getattr(leaf_result, "suspended", False)
                    else previous_scope
                )
                new_call_ids = tuple(
                    output.command_call_id
                    for output in self._turn_outputs[outputs_before:]
                    if output.command_call_id
                )
                node.command_call_ids = tuple(
                    dict.fromkeys((*node.command_call_ids, *new_call_ids))
                )
            if leaf_result is None:
                raise RuntimeError("plan leaf execution returned no result")
            self._remember_plan_answer(node.goal_id, leaf_result)
            return leaf_result

        frontier_before = len(
            [
                node
                for node in plan.nodes
                if node.is_leaf and node.status == "not-reached"
            ]
        )
        execute_span = tracing.start_span(
            self,
            tracing.SPAN_PLAN_EXECUTE,
            attributes={
                "arm": arm.value,
                "leaf_count": len(plan.leaves),
                "executable_leaf_count": len(
                    [leaf for leaf in plan.leaves if leaf.executable]
                ),
                "frontier_count_before": frontier_before,
                "packing_candidate_count": plan.packing.candidate_count,
                "composite_group_count": plan.packing.selected_root_group_count,
                "recursive_composite_group_count": (
                    plan.packing.selected_recursive_group_count
                ),
                "packed_task_count": plan.packing.packed_task_count,
                "packing_edge_count": plan.packing.orchestration_edge_count,
                "resumed": resumed_leaf_result is not None,
            },
        )
        try:
            result = execute_plan(
                plan,
                execute_leaf=_execute_leaf,
                arm=arm,
                safety=self._turn_safety_envelope,
                resumed_from_goal_id=resumed_goal_id,
                resumed_leaf_result=resumed_leaf_result,
                resolve_delayed_bindings=self._resolve_delayed_plan_bindings,
                current_navigation_context=self._plan_navigation_context,
                on_progress=self._checkpoint_plan_progress,
            )
            self._turn_plan_outcome = result.outcome.value
            self._turn_plan_frontier = [
                node.goal_id
                for node in plan.nodes
                if node.is_leaf and node.status == "not-reached"
            ]
            tracing.end_span(
                self,
                execute_span,
                attributes={
                    "frontier_count_after": len(self._turn_plan_frontier),
                    "budget_consumed": plan.budget_consumed,
                    "budget_limit": plan.budget_limit,
                    "censored": bool(result.censored),
                    "censored_reason": result.censored_reason,
                    "packing_applied": bool(
                        result.metadata and result.metadata.packing_applied
                    ),
                    "schedule_sha256": (
                        result.metadata.schedule_sha256
                        if result.metadata is not None
                        else None
                    ),
                    "composite_groups_applied": (
                        result.metadata.composite_groups_applied
                        if result.metadata is not None
                        else 0
                    ),
                    "grouped_task_count": (
                        result.metadata.grouped_task_count
                        if result.metadata is not None
                        else 0
                    ),
                    "grouped_leaf_count": (
                        result.metadata.grouped_leaf_count
                        if result.metadata is not None
                        else 0
                    ),
                    "shared_binding_count": (
                        result.metadata.shared_binding_count
                        if result.metadata is not None
                        else 0
                    ),
                    "context_reuse_count": (
                        result.metadata.context_reuse_count
                        if result.metadata is not None
                        else 0
                    ),
                },
            )
        except BaseException as exc:
            tracing.end_span(
                self,
                execute_span,
                status=tracing.status_for_dispatch_exception(exc),
                attributes={"error_type": type(exc).__name__},
            )
            raise

        if result.outcome is PlanExecutionOutcome.COMPLETED:
            self._turn_active_leaf = None
        return SimpleNamespace(
            final_answer=self._compose_plan_answer(result.answer),
            plan_outcome=result.outcome,
            exhausted=result.exhausted,
            suspended=result.suspended,
            needs_user=result.needs_user,
            clarification=result.clarification,
            censored=result.censored,
            censored_reason=result.censored_reason,
            provider_timeout=result.provider_timeout,
            # ido-mn1.6.10. Arms B and C compose by concatenating one leaf
            # answer per leaf with no model call, so a leaf whose own extraction
            # was cut is a hole in the composed answer that nothing downstream
            # could otherwise see.
            extraction_truncated=result.extraction_truncated,
            extraction_truncated_goal_ids=result.extraction_truncated_goal_ids,
            extraction_failure=result.extraction_failure,
            failure=result.failure,
            successful_leaf_goal_ids=result.successful_leaf_goal_ids,
        )

    def _plan_navigation_context(self) -> Optional[str]:
        """Return the concrete context an isolated plan leaf starts from."""
        workflow = self._app_workflow
        if workflow is None or workflow.current_command_context is None:
            return "*"
        return workflow.current_command_context_name

    def _call_agent_resume(self, observation: str):
        return self._call_agent(
            lambda: self._workflow_tool_agent.resume(
                observation,
                safety_envelope=self._turn_safety_envelope,
            ),
            trace_input=observation,
            resumed=True,
        )

    def _awaiting_user_output(self, clarification: str) -> fastworkflow.CommandOutput:
        command_response = fastworkflow.CommandResponse(response=clarification)
        command_response.artifacts["awaiting_user"] = True
        command_output = fastworkflow.CommandOutput(
            command_response=command_response
        )
        if self._app_workflow:
            command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]
        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()
        return command_output

    def _finalize_agent_output(
        self, original_message: str, agent_result
    ) -> fastworkflow.CommandOutput:
        result_text = (
            agent_result.final_answer
            if hasattr(agent_result, "final_answer")
            else str(agent_result)
        )

        command_response = fastworkflow.CommandResponse(response=result_text)

        if (
            getattr(agent_result, "censored", False)
            or getattr(agent_result, "provider_timeout", False)
            or self._turn_plan is not None
        ):
            # The safety envelope has already expired. Conversation memory is
            # still updated, but deterministically: making another planner call
            # after censoring would extend the attempt past its own cutoff and
            # charge an unreported provider call to a result already frozen.
            if getattr(agent_result, "censored", False):
                conversation_summary = (
                    f"Censored attempt: "
                    f"{getattr(agent_result, 'censored_reason', None) or 'unknown'}"
                )
            elif getattr(agent_result, "provider_timeout", False):
                conversation_summary = (
                    "Provider timeout after "
                    f"{self._turn_plan_checkpoint_count} durable plan "
                    "checkpoint(s)."
                )
            else:
                done_leaf_count = sum(
                    leaf.status == "done"
                    for leaf in self._turn_plan.leaves
                )
                conversation_summary = (
                    f"Plan {self._turn_plan_outcome or 'partial'}: "
                    f"{done_leaf_count} of "
                    f"{len(self._turn_plan.leaves)} leaves done."
                )
            self.append_conversation_turn(
                conversation_summary,
                json.dumps(
                    {
                        "user_query": original_message,
                        "agent_workflow_interactions": self._action_log,
                        "final_agent_response": result_text,
                    }
                ),
            )
        else:
            conversation_summary, _ = self.summarize_and_record_turn(
                original_message, self._action_log, result_text
            )
        if self._action_log:
            command_response.artifacts["conversation_summary"] = conversation_summary

        # Topic 5: the synthesized agent answer carries only its own artifacts (e.g.
        # conversation_summary), so structured outputs from tool calls during the turn
        # would be dropped on the user-facing path. Merge every artifact-bearing turn
        # response into this single CommandResponse.artifacts dict; on key collision,
        # suffix the incoming key with "_<increment>" (1, 2, ...). The framework does
        # not interpret artifact keys — clients read whatever they need.
        if artifact_responses := fastworkflow.turn.collect_artifact_responses(
            self._turn_outputs
        ):
            fastworkflow.turn.merge_artifact_responses_into(
                command_response, artifact_responses
            )

        command_output = fastworkflow.CommandOutput(
            command_response=command_response
        )
        if self._app_workflow:
            command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]

        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()

        return command_output

    def _process_agent_message(self, message: str) -> fastworkflow.CommandOutput:
        self._ensure_agent_initialized()

        # Insights-distillation mode (CLI / Topology A only): run the teacher/student
        # comparison, which drives its own agent passes and returns the student's
        # CommandOutput. Guarded on user_message_queue so it can never run over a
        # Topology-B suspended trajectory.
        if self._generate_insights and self.user_message_queue is not None:
            from fastworkflow.distillation import distill_message
            result = distill_message(self, message)
            self._distillation_insights_count += result.insights_extracted
            if result.run_id:
                self._distillation_run_ids.append(result.run_id)
            self._maybe_enqueue_output(result.command_output)
            self._maybe_enqueue_trace_sentinel()
            return result.command_output

        agent_result = self._run_agent(message)
        self._turn_agent_result = agent_result
        if getattr(agent_result, "suspended", None) is True:
            self._awaiting_user = True
            self._suspended_user_message = message
            self._pending_clarification_request = agent_result.clarification
            self._note_agent_suspension(agent_result.clarification)
            return self._awaiting_user_output(agent_result.clarification)
        self._reset_agent_suspension()
        return self._finalize_agent_output(message, agent_result)

    def _resume_agent_message(self, user_answer: str) -> fastworkflow.CommandOutput:
        self._ensure_agent_initialized()
        # Catch awaiting_user/_suspended desync before any LLM work: resume()
        # consumes _suspended at entry, so a second request (or a restore that
        # lost the react blob) must not become a bare 500.
        agent = self._workflow_tool_agent
        if agent is None or agent.export_suspended() is None:
            raise NoSuspendedAgentStateError(
                "No suspended ReAct state to resume"
            )
        self._note_agent_resume()

        from fastworkflow.workflow_agent import _post_ask_user_response

        # No budget reset. The agent restored the suspended turn's budget with
        # its trajectory; re-bind it here so a cross-process resume finalizes
        # against the same object rather than a WEC field that this process
        # never created (FW-REQ-001 clause 2).
        if agent.budget is not None:
            self._turn_budget = agent.budget

        observation = _post_ask_user_response(
            self._pending_clarification_request,
            user_answer,
            self,
        )
        outputs_before_resume = len(self._turn_outputs)
        agent_result = None
        try:
            agent_result = self._call_agent_resume(observation)
        finally:
            if (
                self._turn_plan is not None
                and self._turn_active_leaf is not None
            ):
                active_leaf = self._turn_plan.node(
                    self._turn_active_leaf
                )
                if active_leaf is not None:
                    resumed_call_ids = tuple(
                        output.command_call_id
                        for output in self._turn_outputs[outputs_before_resume:]
                        if output.command_call_id
                    )
                    active_leaf.command_call_ids = tuple(
                        dict.fromkeys(
                            (
                                *active_leaf.command_call_ids,
                                *resumed_call_ids,
                            )
                        )
                    )
        if agent_result is None:
            raise RuntimeError("resumed agent execution returned no result")
        self._turn_agent_result = agent_result
        if getattr(agent_result, "suspended", None) is True:
            self._pending_clarification_request = agent_result.clarification
            self._note_agent_suspension(agent_result.clarification)
            return self._awaiting_user_output(agent_result.clarification)

        if (
            self._turn_plan is not None
            and self._turn_plan.mode == PlanMode.ENFORCE.value
            and self._turn_active_leaf is not None
        ):
            active_goal_id = self._turn_active_leaf
            active_leaf = self._turn_plan.node(active_goal_id)
            if active_leaf is None:
                raise PlanConfigurationError(
                    f"resumed plan has no active leaf {active_goal_id!r}"
                )
            self._remember_plan_answer(active_goal_id, agent_result)
            # Alias feedback needed the restored scope while the suspended
            # agent ran. The plan executor now reconstructs authoritative scope
            # for each remaining leaf, so do not leave the resumed leaf's
            # envelope as the previous scope of its siblings.
            self._turn_leaf_scope = None
            agent_result = self._execute_compiled_plan(
                self._turn_plan,
                plan_execution_arm_from_env(),
                resumed_leaf_result=agent_result,
            )
            self._turn_agent_result = agent_result
            if getattr(agent_result, "suspended", None) is True:
                self._pending_clarification_request = agent_result.clarification
                self._note_agent_suspension(agent_result.clarification)
                return self._awaiting_user_output(agent_result.clarification)

        original_message = self._suspended_user_message
        self._reset_agent_suspension()
        return self._finalize_agent_output(original_message, agent_result)

    # ------------------------------------------------------------------
    # Shared capture for the fw.agent.tool_call emission sites
    # ------------------------------------------------------------------
    #
    # _process_message and _process_action both open fw.agent.tool_call and both
    # owe §12.1.1's shared capture, so the projection lives here once rather than
    # being written twice and drifting. Everything below is additive recording:
    # no fastWorkflow control flow reads a context handle or a consequence class,
    # which is EXP-003's exit criterion and arch §17.3's stop condition.
    #
    # Amendment (fix-ajv.8): "here" is now `tracing`, because workflow_agent.py
    # opens the same span from a third site and owes the same record. These two
    # methods stay as the WEC-shaped entry points — they supply the app-workflow
    # fallback that the free functions cannot know about — but the projection
    # itself is written once, for all three sites.

    def _context_before(
        self, span, workflow: Optional[fastworkflow.Workflow] = None
    ) -> Optional[dict]:
        """The active context handle before a command runs, or None.

        Gated on a span having actually opened, matching the existing
        attribute-prep rule at this seam: with observability off this must cost
        nothing.
        """
        return tracing.context_before(span, workflow or self._app_workflow)

    def _capture_attributes(
        self,
        span,
        command_output: fastworkflow.CommandOutput,
        context_before: Optional[dict],
        workflow: Optional[fastworkflow.Workflow] = None,
        command_name: Optional[str] = None,
    ) -> dict:
        """Call-id, context-before/after and consequence for one command.

        The call id is read off the CommandOutput rather than minted here: the
        dispatcher that ran the command already stamped it, and minting a second
        one would produce two ids for one execution and join neither.

        ``command_name`` overrides what the CommandOutput reports, because on the
        direct-action path it reports nothing: ``CommandExecutor.perform_action``
        stamps ``workflow_name`` and ``context`` on its result but never
        ``command_name``, so a direct action's outcome carries "" and the
        consequence lookup would find no declaration for any command. The Action
        names what was dispatched, and that is the authoritative identity for
        that path. Passed in rather than fixed on the CommandOutput because
        writing it there would change a public shape, which this slice may not
        do — the empty ``command_name`` on direct-action outcomes is a separate
        defect.
        """
        return tracing.capture_attributes(
            span,
            command_output,
            context_before,
            workflow or self._app_workflow,
            command_name=command_name,
        )

    # ------------------------------------------------------------------
    # Deterministic / assistant mode
    # ------------------------------------------------------------------

    def _process_message(self, message: str) -> fastworkflow.CommandOutput:
        # The deterministic / assistant path: a human's message goes straight to
        # the command executor, so the origin is user and is stated rather than
        # inferred from a counter (FW-REQ-001 clause 3).
        from fastworkflow.workflow_agent import (
            CONTEXT_KEY_INVOCATION_ORIGIN,
            InvocationOrigin,
        )

        if self._app_workflow is not None:
            self._app_workflow.context[CONTEXT_KEY_INVOCATION_ORIGIN] = (
                InvocationOrigin.USER.value
            )

        if self._command_trace_queue is not None:
            self._command_trace_queue.put(
                fastworkflow.CommandTraceEvent(
                    direction=fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW,
                    raw_command=message,
                    command_name=None,
                    parameters=None,
                    response_text=None,
                    success=None,
                    timestamp_ms=int(time.time() * 1000),
                    turn_key=self._turn_key,
                )
            )

        # Span emission sits OUTSIDE the trace-queue guard: the sink is reached
        # via this context, not the transport-queue contract [R28].
        span = tracing.start_span(
            self,
            tracing.SPAN_AGENT_TOOL_CALL,
            kind=tracing.KIND_TOOL,
            attributes={"raw_command": message},
        )
        context_before = self._context_before(span)

        invoke_started_at = datetime.now(timezone.utc)
        try:
            command_output = self._CommandExecutor.invoke_command(self, message)
        except BaseException as exc:
            # One arm, not two: a separate `except CommandCancelledError` left
            # AskUserSuspend — the other control signal — falling through to the
            # BaseException arm below and closing as STATUS_ERROR. fix-ajv.19.
            #
            # ido-cex.7: the SUCCESS end_span below names the command and the
            # context; this one did not, so the tool_call span for a failed
            # dispatch was the one row in the taxonomy that could not say what
            # it covered — the same gap fix-ajv.16's FW-3 closed one level down
            # on fw.command.execute. The dispatch seams stamp the routed
            # identity onto the exception, so it is readable here.
            from fastworkflow.command_executor import _annotation

            tracing.end_span(
                self,
                span,
                status=tracing.status_for_dispatch_exception(exc),
                command_name=_annotation(exc, "_fw_command_name"),
                context=_annotation(exc, "_fw_context"),
                attributes={"error_type": type(exc).__name__},
            )
            raise
        command_output.started_at = invoke_started_at
        command_output.duration_ms = int(
            (datetime.now(timezone.utc) - invoke_started_at).total_seconds() * 1000
        )
        self.append_turn_output(command_output)

        response_text = command_output.command_response.response or ""

        params = command_output.command_parameters or {}
        if hasattr(params, "model_dump"):
            params_dict = params.model_dump()
        elif hasattr(params, "dict"):
            params_dict = params.dict()
        else:
            params_dict = params

        tracing.end_span(
            self,
            span,
            status=(
                tracing.STATUS_OK if command_output.success else tracing.STATUS_ERROR
            ),
            command_name=command_output.command_name or None,
            context=command_output.context or None,
            attributes={
                "response_text": response_text,
                "success": bool(command_output.success),
                **self._capture_attributes(span, command_output, context_before),
            },
        )

        if self._command_trace_queue is not None:
            self._command_trace_queue.put(
                fastworkflow.CommandTraceEvent(
                    direction=fastworkflow.CommandTraceEventDirection.WORKFLOW_TO_AGENT,
                    raw_command=None,
                    command_name=command_output.command_name or "",
                    parameters=params_dict,
                    response_text=response_text,
                    success=bool(command_output.success),
                    timestamp_ms=int(time.time() * 1000),
                    turn_key=self._turn_key,
                )
            )

        record = {
            "command": message,
            "command_name": command_output.command_name or "",
            "parameters": params_dict,
            "response": response_text,
        }

        # "conversation summary" has to identify the turn on its own: it is the
        # only field generate_topic_and_summary and get_conversation_summaries
        # read, and _refine_user_query feeds the last 5 of them to the LLM that
        # refines the next query. The user-visible message leads, matching the
        # agent path, so history reads the same whichever mode produced a turn;
        # on the '/'-prefixed path it is also '/<command_name> <args>', so it
        # names the command and keeps the arguments the resolved command_name
        # would drop. Each part is sliced before joining and newlines are
        # collapsed (the refine prompt is one "key: value" line per field), so
        # the field stays under ~400 chars and never materializes a large
        # command or response. parameters are deliberately absent -- they carry
        # the request payload, which stays in conversation_traces below.
        self.append_conversation_turn(
            " ".join(f"{message[:200]} -> {response_text[:200]}".split()),
            json.dumps(record),
        )

        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()

        return command_output

    def _process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        workflow = self.get_active_workflow() or self._app_workflow

        params = action.parameters or {}
        if hasattr(params, "model_dump"):
            params_dict = params.model_dump()
        elif hasattr(params, "dict"):
            params_dict = params.dict()
        else:
            params_dict = params

        raw_command = f"{action.command_name} {json.dumps(params_dict)}"
        if self._command_trace_queue is not None:
            self._command_trace_queue.put(
                fastworkflow.CommandTraceEvent(
                    direction=fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW,
                    raw_command=raw_command,
                    command_name=None,
                    parameters=None,
                    response_text=None,
                    success=None,
                    timestamp_ms=int(time.time() * 1000),
                    turn_key=self._turn_key,
                )
            )

        # Outside the trace-queue guard [R28]; see _process_message.
        span = tracing.start_span(
            self,
            tracing.SPAN_AGENT_TOOL_CALL,
            kind=tracing.KIND_TOOL,
            attributes={"raw_command": raw_command},
        )
        # The direct-action path's only span: CommandExecutor.perform_action
        # opens none of its own, so this is where §12.1.1's shared capture has
        # to land for this row of the matrix.
        context_before = self._context_before(span, workflow)

        action_started_at = datetime.now(timezone.utc)
        try:
            command_output = self._CommandExecutor.perform_action(workflow, action)
        except BaseException as exc:
            # One arm, not two: a separate `except CommandCancelledError` left
            # AskUserSuspend — the other control signal — falling through to the
            # BaseException arm below and closing as STATUS_ERROR. fix-ajv.19.
            #
            # ido-cex.7: the SUCCESS end_span below names the command and the
            # context; this one did not, so the tool_call span for a failed
            # dispatch was the one row in the taxonomy that could not say what
            # it covered — the same gap fix-ajv.16's FW-3 closed one level down
            # on fw.command.execute. The dispatch seams stamp the routed
            # identity onto the exception, so it is readable here.
            from fastworkflow.command_executor import _annotation

            tracing.end_span(
                self,
                span,
                status=tracing.status_for_dispatch_exception(exc),
                command_name=_annotation(exc, "_fw_command_name"),
                context=_annotation(exc, "_fw_context"),
                attributes={"error_type": type(exc).__name__},
            )
            raise
        command_output.started_at = action_started_at
        command_output.duration_ms = int(
            (datetime.now(timezone.utc) - action_started_at).total_seconds() * 1000
        )
        self.append_turn_output(command_output)

        response_text = command_output.command_response.response or ""

        tracing.end_span(
            self,
            span,
            status=(
                tracing.STATUS_OK if command_output.success else tracing.STATUS_ERROR
            ),
            command_name=command_output.command_name or None,
            context=command_output.context or None,
            attributes={
                "response_text": response_text,
                "success": bool(command_output.success),
                **self._capture_attributes(
                    span,
                    command_output,
                    context_before,
                    workflow,
                    command_name=action.command_name,
                ),
            },
        )
        record_execution(
            self._execution_recorder,
            command_call_id=command_output.command_call_id,
            parent_call_id=None,
            span_id=span.span_id if span is not None else None,
        )

        if self._command_trace_queue is not None:
            self._command_trace_queue.put(
                fastworkflow.CommandTraceEvent(
                    direction=fastworkflow.CommandTraceEventDirection.WORKFLOW_TO_AGENT,
                    raw_command=None,
                    command_name=command_output.command_name,
                    parameters=params_dict,
                    response_text=response_text,
                    success=bool(command_output.success),
                    timestamp_ms=int(time.time() * 1000),
                    turn_key=self._turn_key,
                )
            )

        record = {
            "command": "process_action",
            "command_name": action.command_name,
            "parameters": params_dict,
            "response": response_text,
        }

        # Same bound and one-line normalization as the deterministic path in
        # _process_message, for the same consumers. A direct action carries no
        # user text, so the command name is the only identity available; it is
        # sliced too, so the bound holds without relying on the caller sending a
        # short name. parameters stay out for the same reason as above.
        self.append_conversation_turn(
            " ".join(
                f"{action.command_name[:200]} -> {response_text[:200]}".split()
            ),
            json.dumps(record),
        )

        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()

        return command_output

    def _refine_user_query(
        self, user_query: str, conversation_history: dspy.History
    ) -> str:
        if not conversation_history.messages:
            return user_query
        messages = []
        for conv_dict in conversation_history.messages[-5:]:
            messages.extend(
                f"{k}: {v}"
                for k, v in conv_dict.items()
                if k != "conversation_traces"
            )
        messages.append(f"new_user_query: {user_query}")
        return "\n".join(messages)

    def _extract_conversation_summary(
        self,
        user_query: str,
        workflow_actions: list[dict[str, str]],
        final_agent_response: str,
    ) -> tuple[str, str]:
        conversation_traces = {
            "user_query": user_query,
            "agent_workflow_interactions": workflow_actions,
            "final_agent_response": final_agent_response,
        }

        class ConversationSummarySignature(dspy.Signature):
            """
            A summary of conversation
            Omit descriptions of action sequences
            Capture relevant facts and parameter values from user query, workflow actions and agent response
            """

            user_query: str = dspy.InputField()
            workflow_actions: list[dict[str, str]] = dspy.InputField()
            final_agent_response: str = dspy.InputField()
            conversation_summary: str = dspy.OutputField(
                desc="A multiline paragraph summary"
            )

        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
        with dspy.context(lm=planner_lm):
            cs_func = dspy.ChainOfThought(ConversationSummarySignature)
            prediction = cs_func(
                user_query=user_query,
                workflow_actions=workflow_actions,
                final_agent_response=final_agent_response,
            )
            return prediction.conversation_summary, json.dumps(conversation_traces)
