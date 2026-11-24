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
import weakref
import os
import re
import time
import sqlite3
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Mapping, Optional

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy

import fastworkflow
import fastworkflow.turn
from fastworkflow import active_workflow, metrics, planning_commands, tracing
from fastworkflow import approval_binding, intent_loop, no_agent_mode, turn_cache
from fastworkflow import waiting as waiting_module
from fastworkflow.parameters import check as one_guard, reply as replies
from fastworkflow.observation_offloading.state import current_execute_alias
from fastworkflow.session_state_store import SCHEMA_VERSION, IncompatibleSessionState
from fastworkflow.state_serialization import validate_state
from fastworkflow.plan_frame import PlanFrame, runs_outside_ranges, turn_max_steps
from fastworkflow.observability import store as observability_store
from fastworkflow.observability.execution_recorder import ExecutionRecorder, record_execution
from fastworkflow.turn import TurnResult, TurnStatus, mint_turn_key
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_logger, dspy_utils
from fastworkflow.utils.input_validation import plain_facts, stored_value, tagged_facts
from fastworkflow.utils.react import AskUserSuspend, NoSuspendedAgentStateError

#: A sub-task that hits the ReAct step cap is continued by a fresh executor at most this many times.
MAX_HANDOFFS = 2

#: The refusal for a planning command called once the turn-wide step budget is spent.
NESTED_BUDGET_SPENT_MESSAGE = (
    "The step budget for nested commands is spent; continue with the remaining steps directly."
)

#: The refusal for a direct action run while a question waits for the user's answer.
AWAITING_ANSWER_REFUSAL_MESSAGE = (
    "A question is waiting for the user's answer; reply to it before sending a request, an action or a plan. "
    "To drop it instead, reply \"cancel\" (\"no\" for a review or an approval) or, at the CLI prompt, type //cancel; "
    "over HTTP or MCP call cancel_pending, which drops any question. A reply \"cancel\" to the agent's own "
    "question is only an answer to it: to drop that one, call cancel_pending."
)


def _planning_args(params: Any) -> dict:
    """The arguments of a planning call as a dict: an ``Input`` instance or a dict."""
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return dict(params)


def _opened_part(trajectory: dict[str, Any]) -> str:
    """The context instances a run's observations entered, one line each; "" when none.

    Read from the observation headers fastworkflow prints, so the ids are exact
    where a summary of the run would drop them.
    """
    from fastworkflow.observation_offloading.labels import printed_instances

    opened: dict[tuple[str, str, str], None] = {}
    for key, observation in trajectory.items():
        if key.startswith("observation_") and isinstance(observation, str):
            opened.update(dict.fromkeys(printed_instances(observation)))
    if not opened:
        return ""
    lines = [f"- {name} {instance_id}" + (f" ({label})" if label else "")
             for name, instance_id, label in opened]
    return "Opened in this sub-task:\n" + "\n".join(lines)


#: The artifact key under which the final answer of a split turn lists its parts (``_answer_parts``).
TURN_PARTS_ARTIFACT = "fw_sub_tasks"
#: The artifact key under which the turn lists its stopped or refused parts (no-agent mode; ``fastworkflow.no_agent_mode``).
NOT_RUN_ARTIFACT = no_agent_mode.NOT_RUN_ARTIFACT


class UsePlanError(ValueError):
    """A ``use_plan`` that names no planning command available here, or is sent while a question waits."""
#: The artifact key under which a pending prompt of a paused planning command's decision step carries its data.
PHASE_STEP_ARTIFACT = no_agent_mode.PHASE_STEP_ARTIFACT
#: The artifact that carries the data of a plan-choice question (``question_kind`` "plan_choice").
PLAN_CHOICE_ARTIFACT = "plan_choice"

_CLAUSE_BREAK = re.compile(r"[.:;](?=\s|$)|\n")


def _heading_title(text: str, limit: int = 80) -> str:
    """A short heading for a sub-task's text: its first clause, cut at a word when still long.

    The clause ends at the first ". ", ": ", "; " or newline that leaves at least 3 words; a text with
    no such break and at most ``limit`` characters is returned as it is (whitespace collapsed).
    Never empty: a blank text comes back trimmed.
    """
    stripped = (text or "").strip()
    for match in _CLAUSE_BREAK.finditer(stripped):
        if len(stripped[:match.start()].split()) >= 3:
            stripped = stripped[:match.start()]
            break
    title = " ".join(stripped.split())
    if len(title) <= limit:
        return title or (text or "").strip()
    cut = title[:limit - 1]
    if title[limit - 1] != " " and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "\u2026"


def _join_parts(parts: list[str]) -> str:
    """One sub-task's answers, one per run in hand-off order; a lone answer is unlabelled.

    A run that repeats the previous run's answer word for word adds nothing, so it is dropped.
    """
    parts = [part for i, part in enumerate(parts) if i == 0 or part != parts[i - 1]]
    if len(parts) == 1:
        return parts[0]
    return "\n\n".join(f"Part {i} of {len(parts)}:\n{part}" for i, part in enumerate(parts, 1))


def _part_runs(parts: list[str]) -> list[dict[str, int]]:
    """The runs ``_join_parts`` labels, as data: ``{index, of, start, end}`` for each run's answer (the text after its
    ``Part i of n:`` label line), as character offsets into ``_join_parts(parts)``. ``[]`` when the part was not
    continued (one run after the repeats are dropped, so the text carries no label)."""
    parts = [part for i, part in enumerate(parts) if i == 0 or part != parts[i - 1]]
    if len(parts) == 1:
        return []
    runs, position = [], 0
    for index, part in enumerate(parts, 1):
        position += len(f"Part {index} of {len(parts)}:\n")
        runs.append({"index": index, "of": len(parts), "start": position, "end": position + len(part)})
        position += len(part) + 2
    return runs


def _agent_result_attributes(result: Any, attempts: int) -> dict[str, Any]:
    """Close-out attributes for fw.agent.execute — what the executor returned.

    Read defensively: a suspended run returns a Prediction with no
    ``final_answer``, and distillation passes their own result shapes through
    the same choke point.
    """
    return {
        "attempts": attempts,
        "final_answer": getattr(result, "final_answer", None),
        "suspended": bool(getattr(result, "suspended", False)),
        "clarification": getattr(result, "clarification", None),
        "exhausted": bool(getattr(result, "exhausted", False)),
    }


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


def _frame_property(field_name: str) -> property:
    """A property over one field of the top PlanFrame, ``self._frames[-1]``."""

    def fget(self):
        return getattr(self._top_frame(), field_name)

    def fset(self, value):
        setattr(self._top_frame(), field_name, value)

    return property(fget, fset)


#: The turn's answer when the sub-task review is declined or expires: nothing runs (fix-3rsq.plan-review.2.6).
SUBTASK_REVIEW_DECLINED = ("You declined the sub-task review, so nothing was run. "
                           "Send the request again to review it again.")
#: The answer of a turn whose every part was cancelled while it was being resolved (before its review).
EVERY_PART_CANCELLED = "Nothing was run: every part of the request was cancelled."
SUBTASK_REVIEW_EXPIRED = ("The sub-task review expired before you answered, so nothing was run, and your message "
                          "was not taken as a reply to it. Send the request again to start a new review.")

#: Owner of pending state saved before pending state was tagged with its conversation (fix-newconv-pending). Such
#: state is parked, never applied to a conversation automatically.
UNOWNED_PENDING = "unowned"
#: "No owner set": the live pending state belongs to the conversation the context is bound to.
_OWNER_UNSET = object()
#: The serialized keys of a parked pending snapshot: the suspended turn and what it waits on, without the history,
#: the approvals (cancelled, never parked) or the experiment labels (per context, not per conversation).
_PARKED_PENDING_KEYS = (
    "plan_resolution", "turn_part_facts", "use_plan",
    "subtasks", "subtask_origins", "subtask_index", "subtask_answers",
    "subtask_outputs", "subtask_exhausted", "subtask_plan", "next_step",
    "skill_phase_run", "react", "frames", "frame_react", "turn_child_step_ranges", "turn_steps_used",
    "turn", "waiting",
)


def _owner_key(owner) -> str:
    """The key a parked snapshot is filed under: the conversation id, "none" or the unowned marker."""
    return "none" if owner is None else str(owner)


def _owner_encoded(owner):
    """The owner as the serialized blob carries it: a conversation id, None, or the unowned marker."""
    return owner if owner is None or owner == UNOWNED_PENDING else int(owner)


def _owner_from_state(state: dict[str, Any]):
    """The conversation a serialized pending state belongs to. A blob with no conversation_id key is unowned."""
    if "conversation_id" not in state:
        return UNOWNED_PENDING
    value = state["conversation_id"]
    if value is None or value == UNOWNED_PENDING:
        return value
    return int(value)


class _FreshDefault:
    """A non-data descriptor: an instance attribute whose first read, on an object built without ``__init__``
    (state restore, a test), makes a fresh value from ``factory`` and stores it on the instance. Once the instance
    has the attribute (``__init__`` or an assignment), the instance's own value is read and this is not consulted."""

    def __init__(self, factory):
        self._factory = factory
        self._name = ""

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self._factory()
        obj.__dict__[self._name] = value
        return value


class WorkflowExecutionContext:
    """
    Owns NLU (cme_workflow), the bound app workflow, and message execution.

    No queues, threads, or session lifecycle — inject optional queues from
    ChatSession for trace/output/ask_user when running in CLI mode.
    """

    #: Whether this context reads the agent model's absence as no-agent mode (``no_agent_mode``). Off for a context
    #: built without asking (an embedder or a test that scripts its own agent LM); the entry points that serve
    #: users (``fastworkflow run``, the FastAPI server) turn it on.
    detect_no_agent_mode = False

    # Defaults for an object built without ``__init__`` (state restore, tests): every attribute ``__init__`` or
    # ``_begin_turn`` sets that code reads elsewhere. Immutable ones sit on the class; mutable ones are made fresh
    # per instance on first read.
    @property
    def _plan_resolution(self) -> Optional[Any]:
        """The turn's pending resolution (``resolution.PlanResolution``): it belongs to the plan metadata extraction
        workflow, so it lives in that workflow's context under ``plan_resolution`` (saved with the session under the same
        key). It holds the resolution's progress only: a question that is out (which command, or a bound plan's values) is
        the turn's waiting record (``_waiting``), as the command metadata extraction workflow's is. A context built without the plan workflow (a test double) keeps it on the instance."""
        workflow = self.__dict__.get("_plan_workflow")
        if workflow is None:
            return self.__dict__.get("_plan_resolution_unbound")
        return workflow.context.get("plan_resolution")

    @_plan_resolution.setter
    def _plan_resolution(self, value: Optional[Any]) -> None:
        workflow = self.__dict__.get("_plan_workflow")
        if workflow is None:
            self.__dict__["_plan_resolution_unbound"] = value
        elif value is None:
            workflow.context.pop("plan_resolution", None)
        else:
            workflow.context["plan_resolution"] = value

    _planning_command_refusal: Optional[str] = None
    _refusal_stop_reason: str = ""
    _part_outputs_start: int = 0
    _turn_parts_of_answer: Optional[tuple[str, list[dict[str, Any]]]] = None
    _last_nested_stop: Optional[dict[str, Any]] = None
    _turn_not_run = _FreshDefault(list)
    _turn_plan_previews = _FreshDefault(dict)
    _ask_user_rounds = _FreshDefault(list)

    def __init__(
        self,
        run_as_agent: bool = False,
        session_key: Optional[str] = None,
        mirror_action_log_to_file: bool = False,
        generate_insights: bool = False,
        trace_sink: Optional[tracing.TraceSink] = None,
        detect_no_agent_mode: bool = False,
    ):
        """
        Args:
            detect_no_agent_mode: If True, an unset agent model (``LLM_AGENT``) puts the context in no-agent mode
                         (``fastworkflow.no_agent_mode``): its turns take no agent model step.
            session_key: Stable id (e.g. channel_id) for cme/app workflow persistence.
                         When omitted, cme uses an ephemeral uuid (CLI one-off sessions).
            mirror_action_log_to_file: DEPRECATED no-op. The cwd
                         action.jsonl debug mirror was retired; use the in-process
                         ``action_log`` property (live) or the observability DB
                         (post-mortem) instead. Kept one release for external
                         callers, then removed.
            generate_insights: If True, enable teacher/student distillation on each
                         agent turn (Topology A / CLI only).
            trace_sink: Observability sink for boundary spans and turn records
                         (observability design §3.1), reached via this context,
                         never the transport queues. When omitted, the context
                         opens the app workflow's own sink when a workflow is
                         bound (``bind_app_workflow``). A sink passed here --
                         including an explicit ``tracing.NoOpTraceSink()``,
                         which records no spans or turn records -- is always
                         kept. Observation offloading still writes its
                         evidence and events to the workflow's
                         observability database whatever the sink.
        """
        self._session_key = session_key
        self._run_as_agent = run_as_agent
        if detect_no_agent_mode:
            self.detect_no_agent_mode = True
        self._app_workflow: Optional[fastworkflow.Workflow] = None
        self._keep_alive = False
        # Deprecated no-op, retained one release for ctor compatibility [R25].
        self._mirror_action_log_to_file = mirror_action_log_to_file

        self._user_message_queue: Optional[Queue] = None
        self._command_output_queue: Optional[Queue] = None
        self._command_trace_queue: Optional[Queue] = None

        self._conversation_history: dspy.History = dspy.History(messages=[])
        self._action_log: list[dict[str, Any]] = []

        from fastworkflow.command_executor import CommandExecutor
        self._CommandExecutor = CommandExecutor

        # One PlanFrame per planning-command call; the turn frame is at index 0. The per-call
        # planning attributes (_subtasks, _phase_run, ...) are properties over the top frame.
        self._frames: list[PlanFrame] = [PlanFrame.turn()]
        self._context_change_listener = None

        # The turn's pending resolution (``resolution.PlanResolution``, bd fix-o1lg S4/S5): every part is resolved
        # before the turn's one review (which command, asked only when ambiguous or, with no agent model, when nothing
        # serves it; then a bound plan's missing values), and the question out is its. Also the record of the one
        # parameter question a single unreviewed part's refused typed call puts while it runs. Saved with the session
        # under ``plan_resolution``, so the reply resumes where the resolution stood.
        # A refused typed planning call's question, left by the planner (workflow_agent) for _call_agent_for_query,
        # which suspends the part on it. Only this hand-off: the planner's function has no other way to say it.
        self._planning_command_refusal: Optional[str] = None
        # Where the part that is running began in the turn's outputs: a question is asked only before the part ran
        # anything, so that the reply can run it from its start with nothing run twice.
        self._part_outputs_start = 0
        self._next_step = 0
        # Model steps taken by nested planning-command frames this turn (design 3.5).
        self._turn_steps_used = 0

        # Insights-distillation (teacher/student) state — CLI/Topology-A only.
        self._generate_insights = generate_insights
        self._distillation_insights_count = 0

        # Observability (design §3.1): sink + identity + span bookkeeping.
        # The sink is a per-context attribute, not transport state [R28].
        self._trace_sink: tracing.TraceSink = trace_sink or tracing.NoOpTraceSink()
        #: Whether the caller chose the sink (here or via ``set_trace_sink``).
        #: A chosen sink is never replaced; only an automatic one follows the
        #: bound app workflow. Tracked explicitly because a caller may choose
        #: the no-op sink, which must stay chosen.
        self._trace_sink_supplied: bool = trace_sink is not None
        #: The workflow folder the automatic sink records into, when there is one.
        self._auto_sink_folder: Optional[str] = None
        self._metrics_sink: metrics.MetricsSink = metrics.NoOpMetricsSink()
        self._channel_id: Optional[str] = None
        self._conversation_id: Optional[int] = None
        # fix-newconv-pending: the conversation the live pending state belongs to, when it is not the bound one (set
        # by apply_serialized_state from the saved blob), and the pending state parked per conversation.
        self._pending_owner_override = _OWNER_UNSET
        self._parked_pending: dict[str, list[dict[str, Any]]] = {}
        self._embedder_owns_conversations: bool = False
        # The experiment container's labels, bound beside channel/conversation
        # identity and stamped onto every TurnResult this context produces
        # (`fix-bn1`, `[XR17]`). None on every ordinary turn.
        self._experiment_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._attempt: Optional[int] = None
        self._claim_epoch: Optional[int] = None
        self._server_incarnation: Optional[str] = None
        self._experiment_claim_store: Any = None
        self._trace_span_stack: list[tracing.Span] = []
        self._turn_root_span: Optional[tracing.Span] = None

        # Turn accumulator state (one logical turn = one key, across suspensions)
        self._turn_outputs: list = []
        # The combined answer of a split turn and its parts as data (``_answer_parts``); None until one is built.
        self._turn_parts_of_answer: Optional[tuple[str, list[dict[str, Any]]]] = None
        # The stops of this turn's parts that were not run here (no-agent mode), as ``fw_not_run`` lists them.
        self._turn_not_run: list[dict[str, Any]] = []
        # The stop a nested planning command ended with, for the parent step that ran it (no-agent mode).
        self._last_nested_stop: Optional[dict[str, Any]] = None
        # The planner's text plan of each ad hoc item of the open sub-task review, by goal (no-agent mode).
        self._turn_plan_previews: dict[str, str] = {}
        # Bound here and not only in _begin_turn, because _build_turn_result is
        # reachable on a context that never began a turn in THIS process: resume
        # continues the same logical turn and deliberately skips _begin_turn
        # (see _serialize_turn_accumulator). The read at finalize guards on
        # `is not None`, so an attribute that does not exist made the guard
        # itself raise AttributeError — a missing binding wearing the costume of
        # a null check. fix-ajv.20.
        self._execution_recorder: Optional[ExecutionRecorder] = None
        self._turn_key: Optional[str] = None
        self._turn_started_at: Optional[datetime] = None
        self._turn_user_message: str = ""
        self._turn_refined_message: Optional[str] = None
        self._turn_suspended_ms: int = 0
        self._suspend_began_at: Optional[datetime] = None
        # fix-4da9: the replies a review took before it settled, as {"reply"} rounds of the open ask_user entry
        # (the final reply is not here: it closes the span). Reset when the entry opens.
        self._ask_user_rounds: list[dict[str, str]] = []
        self._turn_entry_workflow_name: str = ""
        self._turn_entry_context: str = ""
        # Unattended turn: nobody is there to answer, so the agent has no
        # ask_user for the whole logical turn. Set when the turn begins.
        self._turn_unattended: bool = False
        # The planning command the client chose for this turn's one part (``process_turn(use_plan=...)``), and the one
        # named for the turn that begins next (read by ``_begin_turn``).
        self._turn_use_plan: Optional[str] = None
        # No agent model: why the last refusal ended the part (read when no question could be asked about it).
        self._refusal_stop_reason: str = ""
        self._next_use_plan: Optional[str] = None
        # The sub-task review gate's state for the turn (fix-3rsq.plan-review.2.2/.2.6/.2.8): the marker the turn span
        # carries, and the plan the gate built (the accepted one after a review; the draft when no one could answer).
        self._turn_subtask_review: str = "not_applicable"
        # `intent_learning`: what each part's settling knew, by part text, until the plan is run (`_record_planned_parts`).
        self._turn_part_facts: Optional[dict] = None
        self._turn_subtask_plan = None
        # fix-48np.1: one shown == run record per bound sub-task the turn ran with drafting on (review_proposals.
        # binding_record); the turn span carries them as ``subtask_bindings``.
        self._turn_subtask_bindings: list = []
        # fix-1ma8.1: the draft's sub-tasks whose planning-command match was ambiguous (names only); the turn span
        # carries them as ``subtask_plan_ambiguity`` whether or not the review was opened.
        self._turn_plan_ambiguity: list = []
        # fix-hfqv: the split the model was asked for this turn (request, raw, merged, changed_by, attempts); the turn
        # span carries it as ``subtask_split``. Recording only.
        self._turn_split: dict = {}
        self._turn_agent_result: Any = None
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

        # Phase 9 step 4: the CME's heads serve every session, so an old CME is refused here, for
        # every runtime path (chat, MCP, FastAPI). Training does not construct this class.
        from fastworkflow.train.artifact_versioning import require_construction_record
        require_construction_record(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"))
        require_construction_record(
            fastworkflow.get_internal_workflow_path("plan_metadata_extraction"))

        cme_id = (
            f"cme_{session_key}"
            if session_key
            else f"cme_{uuid.uuid4().hex}"
        )
        self._cme_workflow = fastworkflow.Workflow.create(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            workflow_id_str=cme_id,
            workflow_context={},    # a fresh context: a workflow created again under one id starts with nothing pending
        )

        plan_workflow_id = (
            f"plan_{session_key}"
            if session_key
            else f"plan_{uuid.uuid4().hex}"
        )
        plan_path = fastworkflow.get_internal_workflow_path("plan_metadata_extraction")
        self._plan_workflow = fastworkflow.Workflow.create(
            plan_path,
            workflow_id_str=plan_workflow_id,
        )
        from fastworkflow.utils import python_utils

        proposal_review_cb = python_utils.get_module(
            os.path.join(
                plan_path,
                "_commands",
                "ProposalReview",
                "_ProposalReview.py",
            ),
            plan_path,
        )
        # The review record is the ROOT of this workflow: it has no parent. A wildcard answer
        # from the head (bd fix-1hs8) names no command, so the CME walks to the parent, and a
        # non-root record with no parent crashed there (AttributeError: no get_parent).
        # A workflow created again under the same id (a restart or restore on one channel) is the
        # same object and already has its root: the root can be set once, so that record is reused.
        self._plan_resolution = None     # the pending resolution lives in the plan workflow's context: start clean
        existing_root = self._plan_workflow.root
        if existing_root is None:
            self._plan_workflow.root = proposal_review_cb.ProposalReview()
        else:
            self._plan_workflow.record = existing_root

        plan_cme_id = (
            f"plan_cme_{session_key}"
            if session_key
            else f"plan_cme_{uuid.uuid4().hex}"
        )
        self._plan_cme = fastworkflow.Workflow.create(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            workflow_id_str=plan_cme_id,
            workflow_context={"app_workflow": self._plan_workflow},
        )
        self._install_waiting_ports()

        self.clear_conversation_history()

    def _frame_stack(self) -> list[PlanFrame]:
        frames = getattr(self, "_frames", None)
        if not frames:
            # A context built with __new__ (no __init__) starts with the turn frame.
            frames = self._frames = [PlanFrame.turn()]
        return frames

    def _top_frame(self) -> PlanFrame:
        return self._frame_stack()[-1]

    def _turn_frame(self) -> PlanFrame:
        return self._frame_stack()[0]

    # Sub-task split of one turn (see _run_agent / _run_subtasks_from).
    _subtasks = _frame_property("subtasks")
    # Original sub-task number of each entry in _subtasks; a hand-off continuation shares its origin.
    _origins = _frame_property("origins")
    _subtask_index = _frame_property("subtask_index")
    _subtask_answers = _frame_property("subtask_answers")
    _subtask_outputs = _frame_property("subtask_outputs")
    # Set when a sub-task hit the step cap with no hand-off left: the turn is then exhausted.
    _subtask_exhausted = _frame_property("subtask_exhausted")
    # Finish check (fix-3xts): the first step each sub-task origin's outputs count from,
    # and the run's initial plan as the planner wrote it.
    _evidence_floors = _frame_property("evidence_floors")
    _initial_plan_text = _frame_property("initial_plan_text")
    # A phased planning command call in progress (planning_phases; see _run_planning_command_phases): JSON
    # state that rides the suspension blob, and the live (command, phases) for it.
    _phase_run = _frame_property("phase_run")
    _phase_live = _frame_property("phase_live")
    _phased_planning_command_call = _frame_property("phased_planning_command_call")
    _phase_call = _frame_property("phase_call")
    # The ReAct agent of the call in progress: the top frame's agent (the turn frame's, for now).
    _workflow_tool_agent = _frame_property("agent")

    @property
    def _turn_subtask_plan(self):
        """The turn's sub-task plan (review_proposals.SubTaskProposal): the accepted proposal, or the draft when the
        turn was not reviewed. It is on the turn frame, so a suspended turn keeps it (fix-3rsq.plan-review.2.9)."""
        return self._turn_frame().subtask_plan

    @_turn_subtask_plan.setter
    def _turn_subtask_plan(self, plan) -> None:
        self._turn_frame().subtask_plan = plan

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
        """Wire an observability sink chosen by the caller; it is never replaced.

        ``None`` hands the choice back to the context: the sink becomes the
        automatic one again, opened for the bound app workflow now or at the
        next ``bind_app_workflow``. To record no spans or turn records, pass
        ``tracing.NoOpTraceSink()``; observation offloading still writes its
        evidence to the workflow's observability database.
        """
        if sink is None:
            self._trace_sink = tracing.NoOpTraceSink()
            self._trace_sink_supplied = False
            self._auto_sink_folder = None
            if self._app_workflow is not None:
                self._open_auto_sink(self._app_workflow)
            return
        self._trace_sink = sink
        self._trace_sink_supplied = True
        self._auto_sink_folder = None

    def _open_auto_sink(self, workflow: fastworkflow.Workflow) -> None:
        """Open the app workflow's observability sink for a context given none.

        Recording is always on, so a context an embedder builds without a sink
        records into the bound workflow's own DB, owner-only and pruned when
        the sink opens, exactly as fastWorkflow's entry points do. Rebinding to
        another workflow moves the automatic sink to that workflow's DB. The
        internal command-metadata workflow is never a recording target.
        ``get_observability_sink`` never raises: a store that cannot be opened
        leaves the context on the no-op sink.
        """
        folder = str(getattr(workflow, "folderpath", "") or "")
        if not folder:
            return
        resolved = os.path.realpath(folder)
        internal_root = os.path.dirname(os.path.realpath(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction")
        ))
        if resolved == internal_root or resolved.startswith(internal_root + os.sep):
            return
        if self._auto_sink_folder == resolved and tracing.get_sink(self) is not None:
            return
        sink = observability_store.get_observability_sink(folder)
        self._trace_sink = sink or tracing.NoOpTraceSink()
        self._auto_sink_folder = resolved if sink is not None else None

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
        """Bind channel/conversation identity BEFORE the turn.

        The embedder owns identity: FastAPI binds its channel_id, the CLI a
        synthetic ``cli:<session-start>`` channel. Stamped onto every
        span and TurnResult this context produces. A None argument leaves
        the corresponding binding unchanged (conversation ids rotate without
        re-binding the channel).

        ``embedder_owns_conversations=True`` (additive) disables the WEC's
        own conversation self-minting for this context. FastAPI passes it:
        its minting chokepoint carries the legacy-store floor and syncs it
        back, so a WEC self-mint on its degraded path would mint
        a floor-less id that can alias a legacy conversation and split the
        session across two ids once the chokepoint's own mint succeeds.
        """
        # Validate FIRST, mutate second: a rejected experiment triple must not
        # leave the context half-bound with a new channel_id and the old labels.
        if experiment_id is not None or task_id is not None or attempt is not None:
            self._bind_experiment_labels(experiment_id, task_id, attempt)
        if channel_id is not None:
            self._channel_id = channel_id
        if conversation_id is not None and conversation_id != self._conversation_id:
            # fix-newconv-pending: the live pending state settles before the conversation changes, and the new
            # conversation's parked pending state (if any) comes back.
            self._settle_pending_for_conversation(conversation_id)
            self._conversation_id = conversation_id
            self._restore_parked_pending(conversation_id)
        if embedder_owns_conversations is not None:
            self._embedder_owns_conversations = embedder_owns_conversations

    def bind_experiment_claim(self, claim: dict[str, Any], store: Any) -> None:
        """Bind only a store-issued claim, never caller-supplied raw labels."""
        store.validate_attempt_claim(claim)
        self._validate_experiment_labels(
            claim.get("experiment_id"), claim.get("task_id"), claim.get("attempt")
        )
        if not claim.get("epoch") or not claim.get("server_incarnation"):
            raise ValueError("an experiment claim needs epoch and server_incarnation")
        self._experiment_id = claim["experiment_id"]
        self._task_id = claim["task_id"]
        self._attempt = int(claim["attempt"])
        self._claim_epoch = int(claim["epoch"])
        self._server_incarnation = claim["server_incarnation"]
        self._experiment_claim_store = store
        self._channel_id = claim["channel_id"]
        self._conversation_id = int(claim["conversation_id"])

    @property
    def observability_experiment_claim(self) -> dict[str, Any]:
        if self._experiment_id is None:
            return {}
        return {
            "experiment_id": self._experiment_id,
            "task_id": self._task_id,
            "attempt": self._attempt,
            "epoch": self._claim_epoch,
            "server_incarnation": self._server_incarnation,
        }

    def assert_experiment_claim_current(self) -> None:
        """Fence turn admission and each registered command dispatch."""
        if self._experiment_id is None:
            return
        # The in-process ExperimentHarness predates registered external
        # channels and remains compatible. A registered binding always carries
        # both fields and can never degrade to this legacy/internal path.
        if self._claim_epoch is None and self._server_incarnation is None:
            return
        if self._claim_epoch is None or self._server_incarnation is None:
            raise RuntimeError("experiment claim fencing metadata is incomplete")
        if self._experiment_claim_store is not None:
            self._experiment_claim_store.validate_attempt_claim(
                self.observability_experiment_claim
            )

    def _bind_experiment_labels(
        self,
        experiment_id: Optional[str],
        task_id: Optional[str],
        attempt: Optional[int],
    ) -> None:
        """Validate and bind the experiment triple.

        All three or none. A turn labelled with an experiment but no task
        belongs to an experiment and to no task: it contributes to a numerator
        and to no denominator, and every GROUP BY in the scoring layer is
        silently wrong. Refusing here is the only cheap place to catch it.

        The `isinstance(attempt, int)` check is load-bearing and not decorative.
        SQLite's INTEGER is a type AFFINITY, not a constraint -- a string bound
        to it that cannot be losslessly converted is stored as TEXT -- so the
        column's declared type protects nothing on its own. This is what makes
        `attempt` safe to leave otherwise unpoliced.
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
    def current_frame_alias(self) -> str:
        """The alias of the planning frame on top, "" for the turn frame (design 3.5).
        Tracing reads it when a span opens, so the span records the frame it ran under."""
        return self._top_frame().alias

    @property
    def trace_span_stack(self) -> list[tracing.Span]:
        """Open-span stack for parenting nested spans (single-turn: I5)."""
        return self._trace_span_stack

    def clear_action_log(self) -> None:
        """Clear in-memory action log for a new agent turn."""
        self._action_log.clear()

    def append_action_log(self, record: dict[str, Any]) -> None:
        """Append one agent/workflow interaction record (in-memory only)."""
        self._action_log.append(record)

    @property
    def action_log(self) -> list[dict[str, Any]]:
        return self._action_log

    # ------------------------------------------------------------------
    # Turn accumulator (v2.21: capture + TurnResult return type only)
    # ------------------------------------------------------------------

    def _begin_turn(self, user_message: str, unattended: bool = False) -> None:
        """Atomic turn start: reset accumulator, mint key, stamp started_at.

        Never called while awaiting_user — a message during suspension is the
        resume answer and continues the same logical turn.
        """
        self.assert_experiment_claim_current()
        self._ensure_observability_conversation()
        # Drops nested frames only. The turn frame's own state (its sub-tasks and its agent)
        # is left as it was: _clear_subtask_state clears the sub-task state when a run starts,
        # and a message that resumes a suspended turn still finds its sub-tasks.
        del self._frame_stack()[1:]
        self._turn_steps_used = 0
        self._turn_outputs = []
        self._turn_parts_of_answer = None
        self._turn_not_run = []
        self._turn_plan_previews = {}
        self._turn_key = mint_turn_key()
        self._turn_started_at = datetime.now(timezone.utc)
        self._turn_user_message = user_message
        self._turn_unattended = bool(unattended)
        self._turn_use_plan, self._next_use_plan = self._next_use_plan, None
        self._plan_resolution, self._refusal_stop_reason, self._part_outputs_start = None, "", 0
        self._turn_part_facts = None
        self._turn_subtask_review = "not_applicable"
        self._turn_subtask_plan = None
        self._turn_subtask_bindings = []
        self._turn_plan_ambiguity = []
        self._turn_split = {}
        turn_cache.begin_turn()
        self._turn_refined_message = None
        self._turn_suspended_ms = 0
        self._suspend_began_at = None
        self._ask_user_rounds = []
        self._turn_agent_result = None
        self._turn_history_baseline = len(self.conversation_history.messages)
        approval_binding.on_new_turn(self, self._app_workflow, unattended=self._turn_unattended)

        self._turn_entry_workflow_name = ""
        self._turn_entry_context = ""
        with contextlib.suppress(Exception):
            if self._app_workflow is not None:
                self._turn_entry_workflow_name = (
                    self._app_workflow.folderpath.split("/")[-1]
                )
                self._turn_entry_context = (
                    self._app_workflow._current_command_context_name or ""
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
        self._turn_root_span = tracing.start_span(
            self,
            tracing.SPAN_TURN,
            span_id=tracing.root_span_id(self._turn_key),
            attributes={
                "turn_key": self._turn_key,
                "channel_id": self._channel_id,
                "conversation_id": self._conversation_id,
                "user_message": user_message,
                "unattended": self._turn_unattended,
            },
            context=self._turn_entry_context or None,
            use_stack=False,
            emit_open=True,
        )

    def append_turn_output(self, command_output: fastworkflow.CommandOutput) -> None:
        """Append one command execution to the current turn's accumulator."""
        if command_output.framework is None:
            command_output.framework = self._is_framework_command(command_output.command_name)
        if command_output.sub_task is None:
            command_output.sub_task = self._executing_sub_task()
        self._turn_outputs.append(command_output)
        fastworkflow.turn.warn_on_unserializable_artifacts(command_output)

    def _executing_sub_task(self) -> Optional[int]:
        """The 1-based number of the sub-task the turn is executing (the answer's ``### N.`` number), read from the
        turn frame, which a nested planning-command frame leaves alone; None unless the turn runs several sub-tasks."""
        turn = self._turn_frame()
        index = turn.subtask_index
        if len(set(turn.origins)) < 2 or not 0 <= index < len(turn.origins):
            return None
        return turn.origins[index] + 1

    def _is_framework_command(self, command_name: str) -> bool:
        """Whether *command_name* is one of the framework's own commands in the app workflow's command
        directory (``CommandMetadata.framework``). Framework commands are global and registered under
        their bare name, so a workflow's own command of the same leaf (qualified by its context) is not
        one. False when the directory cannot be read: the output is then simply not marked."""
        if self._app_workflow is None or not command_name:
            return False
        try:
            routing = fastworkflow.RoutingRegistry.get_definition(self._app_workflow.folderpath)
            meta = routing.command_directory.map_command_2_metadata.get(command_name)
        except Exception:  # noqa: BLE001 - an unreadable directory marks nothing
            return False
        return bool(meta is not None and meta.framework)

    def append_ask_user_entry(self, question: str) -> fastworkflow.CommandOutput:
        """Append an unanswered ask_user exchange entry and return it.

        Role inversion: command_parameters holds the agent's question; the
        response holds the user's answer ("" + success=False while unanswered).

        Also opens the fw.ask_user human-wait span (deterministic id per
        attempt; emitted at open so the wait is visible while the turn
        is suspended). Both topologies funnel through here: Topology A via
        _ask_user_tool, Topology B via _note_agent_suspension.
        """
        # `is_ask_user`, not the name: this count feeds a DETERMINISTIC span id,
        # so a failed command called `ask_user` would not just miscount, it
        # would make two real ask_user spans collide on one id. fix-ajv.17.
        attempt = sum(1 for output in self._turn_outputs if output.is_ask_user)
        self._ask_user_rounds = []
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
            tracing.start_span(
                self,
                tracing.SPAN_ASK_USER,
                kind=tracing.KIND_HUMAN_WAIT,
                span_id=tracing.deterministic_span_id(
                    self._turn_key, tracing.SPAN_ASK_USER, attempt
                ),
                parent_span_id=tracing.root_span_id(self._turn_key),
                command_name="ask_user",
                attributes={"agent_query": question, "attempt": attempt},
                use_stack=False,
                emit_open=True,
            )
        return entry

    def complete_ask_user_entry(self, answer: str) -> None:
        """Fill the last unanswered ask_user entry with the user's answer.

        duration_ms is the user's think time. No-op when there is no
        unanswered ask_user entry.

        Closes the matching fw.ask_user span. The span is rebuilt from the
        entry rather than held in memory, so the close is an idempotent upsert
        that also works when the answer arrives in a different process than
        the question.
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

    def _close_ask_user_span(
        self, entry_index: int, entry: fastworkflow.CommandOutput, answer: str
    ) -> None:
        """Emit the closed fw.ask_user span for a just-answered entry."""
        if not self._turn_key or tracing.get_sink(self) is None:
            return
        attempt = sum(
            1
            for output in self._turn_outputs[:entry_index]
            if output.is_ask_user
        )
        span = tracing.Span(
            span_id=tracing.deterministic_span_id(
                self._turn_key, tracing.SPAN_ASK_USER, attempt
            ),
            trace_id=self._turn_key,
            name=tracing.SPAN_ASK_USER,
            kind=tracing.KIND_HUMAN_WAIT,
            parent_span_id=tracing.root_span_id(self._turn_key),
            channel_id=self._channel_id,
            command_name="ask_user",
            start_ns=tracing.datetime_to_ns(entry.started_at) or 0,
            experiment_id=self._experiment_id,
            task_id=self._task_id,
            attempt=self._attempt,
            claim_epoch=self._claim_epoch,
            server_incarnation=self._server_incarnation,
        )
        # fix-4da9: every reply of the question in order, the final one last (`user_response` keeps holding it).
        replies = [r["reply"] for r in self._ask_user_rounds] + [answer]
        self._ask_user_rounds = []
        tracing.end_span(
            self,
            span,
            attributes={
                "agent_query": entry.command_parameters,
                "attempt": attempt,
                "user_response": answer,
                "user_responses": replies,
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

    # The deterministic path's pending question (kinds ``intent`` and ``parameters``) is the turn's one waiting record
    # (``fastworkflow.waiting``, ``docs/pending_state_design.md``). The built-in commands reach it through the port in
    # their context. The sub-task review and a paused decision step are the same record (kinds ``review`` and
    # ``decision``); the review's own slot question is nested in its payload (``waiting.SLOT_KEY``, the plan CME's
    # ``SlotPort``). ``_assistant_round``: the round began with a "/" message.
    _waiting: Optional[waiting_module.Waiting] = None
    _assistant_round: bool = False
    #: The loop of the question whose reply is re-running its part, for that re-run only (``_resume_parameter_loop_unlistened``
    #: sets and clears it, in one call, inside one ``process_turn``): the record has left the slot, and ``_open_parameter_loop``
    #: measures the part's next refusal against the names the last question listed and the count of replies with no
    #: progress. When the re-run asks again the new record carries the whole loop, so nothing of it is ever saved.
    _rerun_loop: Optional[Any] = None

    @property
    def _awaiting_user(self) -> bool:
        """The turn is suspended on a question (the next message continues it), read from the waiting record: the agent's
        own question and every other question that rides on the agent's suspension (``Waiting.rides_suspension``). Not a
        stored flag; nothing sets it."""
        held = self._waiting
        return held is not None and held.rides_suspension

    def _put_record(self, record: waiting_module.Waiting) -> None:
        """The one place a record becomes the turn's (it replaces any other, so two are never out); it keeps the "/" round."""
        record.assistant = self._assistant_round
        self._waiting = record

    def _ask_agent_question(self, question: Any, message: Optional[str] = None) -> None:
        """The agent suspended on its own question (``ask_user``): the turn's ``agent`` record. A question that rides on the
        same suspension (a review, a decision step, a plan question, an approval) is already the record and stays; the
        CME's open command, left by a refused call of an agent session, gives way to it."""
        held = self._waiting
        if held is None or held.kind == waiting_module.KIND_AGENT or held.owner == waiting_module.OWNER_CME:
            payload = {} if message is None else {"message": message}
            self._put_record(waiting_module.Waiting(waiting_module.KIND_AGENT, question=str(question or ""), payload=payload))

    def _suspended_message(self) -> str:
        """The user message of the suspended turn (the agent record's, else the turn's)."""
        held = self._waiting
        message = held.payload.get("message") if held is not None and held.kind == waiting_module.KIND_AGENT else None
        return message or self._turn_user_message or ""

    def _put_approval_record(self, record) -> None:
        """The approval store's record of an action awaiting its answer is the turn's question (``approval_binding.gate``
        calls this when it asks). The store keeps the record; this points at it."""
        self._put_record(waiting_module.Waiting(
            waiting_module.KIND_APPROVAL, question=str(record.rendering or ""),
            payload={"record_id": record.id, "suspended_agent": bool(record.suspended_agent)}))

    def _store_record_ended(self, record) -> None:
        """The approval store ended *record* (accepted, declined, expired, invalidated; ``review_engine.transition``): the
        turn's question that points at it ends with it. The one place a store-backed record is dropped."""
        held = self._waiting
        if held is not None and held.store_backed and held.payload.get("record_id") == record.id:
            self._drop_store_record()

    def _turn_port(self) -> waiting_module.Port:
        """The port of the turn's waiting record (the plan path's questions are put and taken through it)."""
        return waiting_module.OwnerPort(self)

    def _install_waiting_ports(self) -> None:
        """Give the two built-in contexts their port (a workflow created again under one id keeps its context, so this
        runs after every ``Workflow.create`` of them)."""
        self._cme_workflow.context[waiting_module.PORT_KEY] = waiting_module.OwnerPort(self)
        plan_cme = getattr(self, "_plan_cme", None)
        if plan_cme is not None:
            plan_cme.context[waiting_module.PORT_KEY] = waiting_module.SlotPort(self)

    @property
    def assistant_round(self) -> bool:
        """The round was started by a "/" message: its bare replies stay deterministic in an agent session."""
        return self._assistant_round

    def start_round(self) -> None:
        self._assistant_round = True

    def end_round(self) -> None:
        """The one place the "/" round ends (the record out, if any, loses the mark with it)."""
        self._assistant_round = False
        if self._waiting is not None:
            self._waiting.assistant = False

    @property
    def run_as_agent(self) -> bool:
        return self._run_as_agent

    @property
    def no_agent_mode(self) -> bool:
        """True when this context detects the mode and the agent model (``LLM_AGENT``) is not specified."""
        return bool(self.detect_no_agent_mode) and not no_agent_mode.agent_model_specified()

    @property
    def app_workflow(self) -> Optional[fastworkflow.Workflow]:
        return self._app_workflow

    @property
    def workflow_tool_agent(self):
        return self._workflow_tool_agent

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
            "ask_user_rounds": list(self._ask_user_rounds),
            "agent_result": agent_result,
            # The turn's marks that later steps read: the review's state, the stops of parts not run here, and
            # whether nobody can answer. Absent from older blobs: the defaults (not applicable, none, attended).
            "subtask_review": self._turn_subtask_review,
            "not_run": json.loads(json.dumps(self._turn_not_run, default=str)),
            "unattended": self._turn_unattended,
        }

    def _waiting_state(self, held: Optional[waiting_module.Waiting], cme_workflow,
                       plan_cme=None) -> Optional[dict[str, Any]]:
        """The saved form of the turn's waiting record *held*, or None when none is out. A review record carries its
        slot question nested in its payload; that slot is a record of the plan CME (*plan_cme*), saved like the CME's own.

        The CME's scratch keys ``command`` (the text being processed) and ``command_name`` (the command whose parameters
        are asked) ride in the payload: wildcard.py reads ``command_name`` unconditionally while a parameter question
        is pending, and the extractor merges the user's answer into the loop. Without them a restored session either raises
        KeyError or silently discards every parameter collected so far. ``end_command_processing`` leaves ``command_name``
        behind, so the record, not that key, is what says a command is in flight.
        """
        if held is None or cme_workflow is None:
            # close() treats this as reachable, and has_open_command() runs after every turn rather than only
            # suspensions, so a missing record or CME workflow must read as "nothing in flight".
            return None
        state = held.to_state()
        if held.owner == waiting_module.OWNER_CME:
            state["payload"] = {**state["payload"], "command": cme_workflow.context.get("command"),
                                "command_name": cme_workflow.context.get("command_name")}
        if held.kind == waiting_module.KIND_REVIEW and state["payload"].get(waiting_module.SLOT_KEY) and plan_cme is not None:
            slot = state["payload"][waiting_module.SLOT_KEY]
            slot["payload"] = {**slot["payload"], "command": plan_cme.context.get("command"),
                               "command_name": plan_cme.context.get("command_name")}
        return state

    def has_open_command(self) -> bool:
        """True when a CME command is mid-extraction (the turn's waiting record is the deterministic path's question).

        Such a session is not awaiting_user, so nothing else marks it as holding
        state that must survive eviction.
        """
        held = self._waiting
        return held is not None and held.owner == waiting_module.OWNER_CME and held.kind in waiting_module.CME_KINDS

    # ------------------------------------------------------------------
    # Pending state and conversations (fix-newconv-pending)
    # ------------------------------------------------------------------
    #
    # Live pending state belongs to the conversation that created it: a paused ask_user question, a planning
    # refusal question, an open review, a CME continuation (parked) and a pending approval (cancelled). When the
    # context moves to another conversation, the live state is settled before anything else runs: approvals are
    # cancelled (they would run side effects on a bare "yes"), the rest is parked under its owner and comes back
    # when that conversation is bound again. stop_asking's tool removal is scoped the same way.

    def _live_pending_owner(self):
        """The conversation the live pending state belongs to (see _OWNER_UNSET)."""
        if self._pending_owner_override is not _OWNER_UNSET:
            return self._pending_owner_override
        return self._conversation_id

    def _has_parkable_pending(self) -> bool:
        """Live pending state that is parked on a conversation change: a question, a refusal, a review or a CME run."""
        return self._waiting is not None

    def _has_live_pending(self) -> bool:
        """True when anything pending is live: a parkable question or review, or a pending approval."""
        return self._has_parkable_pending() or bool(approval_binding.active_approvals(self._app_workflow))

    def has_pending_state(self) -> bool:
        """Public form of _has_live_pending, for the embedder's New conversation."""
        return self._has_live_pending()

    def has_parked_pending(self) -> bool:
        """True when some conversation's pending state is parked (it must be saved with the rest of the state)."""
        return bool(self._parked_pending)

    def _cancel_live_approvals(self, reason: str) -> int:
        """Cancel the pending approvals (an approval runs its side effect on a bare reply, so it never outlives its
        conversation). A suspended approval also abandons the agent's suspended call."""
        if self._app_workflow is None:
            return 0
        held, suspended = self._waiting, self._awaiting_user
        cancelled = approval_binding.cancel_conversation_approvals(self, self._app_workflow, reason=reason)
        if cancelled and held is not None and held.kind == waiting_module.KIND_APPROVAL and held.suspended_agent and suspended:
            self._reset_agent_suspension()
        return cancelled

    def _clear_cme_continuation(
        self, cme_workflow: Optional[fastworkflow.Workflow] = None,
    ) -> None:
        """Drop the CME command in flight, as end_command_processing does (its command_name is kept there, not here)."""
        if cme_workflow is None:
            cme_workflow = self._cme_workflow
        if cme_workflow is None:
            return
        context = cme_workflow.context
        context.pop("command", None)
        context.pop("command_name", None)
        waiting_module.port_of(context).clear()

    def _clear_live_pending(self) -> None:
        """Clear the live pending state (the suspended turn, its review, its CME run) without touching history."""
        self._reset_agent_suspension()
        self._turn_frame().subtask_plan = None
        self._plan_resolution = None
        self._suspend_began_at = None
        self._turn_suspended_ms = 0
        if self.has_open_command():
            self._clear_cme_continuation()
        plan_cme = getattr(self, "_plan_cme", None)
        if plan_cme is not None:
            self._clear_cme_continuation(plan_cme)

    def discard_live_pending(self) -> bool:
        """Drop the live pending state for good (ending a session): approvals, the suspended turn, its review and
        a CME run. Pending state parked for other conversations is left alone. True when anything was live."""
        had = self._has_live_pending()
        self.cancel_pending()
        self._clear_live_pending()
        return had

    def _pending_snapshot(self, owner) -> dict[str, Any]:
        """The live pending state as a parked snapshot, tagged with its owner."""
        payload = self.serialize_state(channel_id=str(self._channel_id or ""))
        snapshot: dict[str, Any] = {key: payload.get(key) for key in _PARKED_PENDING_KEYS}
        held = self._waiting
        if held is not None and held.store_backed and snapshot.get("waiting"):
            # One thing is parked: the turn's record carries the approval store's record it points at, which leaves the
            # store (a parked review must not gate the other conversation) and comes back with the record.
            snapshot["waiting"]["payload"][waiting_module.STORE_RECORD_KEY] = approval_binding.take_record(
                self._app_workflow, held.payload.get("record_id"))
        snapshot["conversation_id"] = owner
        snapshot["schema_version"] = SCHEMA_VERSION
        return snapshot

    def _park_and_clear_live_pending(self, owner) -> None:
        """Park the questions and reviews under their owner (never cancelled), then clear them from the live context.

        A snapshot that cannot be encoded is dropped with an error rather than applied to another conversation.
        """
        if self._has_parkable_pending():
            try:
                snapshot = self._pending_snapshot(owner)
            except Exception:  # noqa: BLE001 - losing one parked question must not route it to another conversation
                logger.error("pending state of conversation %s could not be parked; it is dropped", owner,
                             exc_info=True)
                snapshot = None
            if snapshot is not None:
                self._parked_pending.setdefault(_owner_key(owner), []).append(snapshot)
                logger.info("parked pending state of conversation %s", owner)
        self._clear_live_pending()

    def _settle_pending_for_conversation(self, new_conversation_id) -> None:
        """Settle the live pending state for a context that is moving to *new_conversation_id*.

        Nothing happens when the state already belongs to that conversation. Otherwise approvals are cancelled,
        questions and reviews are parked under their owner, and stop_asking's tool removal ends with its conversation.
        """
        owner = self._live_pending_owner()
        # Unowned state is parked whatever the conversation is. Any other state is adopted, not parked, by a context
        # that has no conversation bound yet: a bare embedder's turn mints its id after this check, as it did before
        # pending state was tagged.
        if owner != UNOWNED_PENDING and (owner == new_conversation_id or new_conversation_id is None):
            self._pending_owner_override = _OWNER_UNSET
            return
        self._pending_owner_override = _OWNER_UNSET
        if self._app_workflow is not None:
            self._cancel_live_approvals(reason="conversation changed")
        if self._has_parkable_pending():
            self._park_and_clear_live_pending(owner)  # a "/" round's flag is parked with it (fix-icce)
        # The flag stays with its round: it means nothing in the other conversation (the agent must be reachable there).
        self.end_round()
        self._enable_all_tools()

    def park_pending_for_new_conversation(self) -> None:
        """New conversation (the button): settle the live pending state at once, not at the next message.

        Approvals are cancelled and questions and reviews are parked under the current conversation. The conversation
        id itself is changed by the embedder (reserve_conversation_id), which parks the state through bind.
        """
        owner = self._live_pending_owner()
        self._pending_owner_override = _OWNER_UNSET
        if self._app_workflow is not None:
            self._cancel_live_approvals(reason="new conversation")
        self._park_and_clear_live_pending(owner)
        self.end_round()  # as in _settle_pending_for_conversation
        self._enable_all_tools()

    def _restore_parked_pending(self, conversation_id) -> None:
        """Bring back the pending state parked for *conversation_id*, the oldest one first.

        Unowned state is never restored here. A snapshot that cannot be applied stays parked.
        """
        key = _owner_key(conversation_id)
        items = self._parked_pending.get(key)
        if not items:
            return
        snapshot = items.pop(0)
        if not items:
            del self._parked_pending[key]
        if snapshot.get("schema_version") != SCHEMA_VERSION:
            self._parked_pending.setdefault(key, []).insert(0, snapshot)
            logger.error("parked pending state of conversation %s is from another schema; it stays parked", key)
            return
        if isinstance(parked := snapshot.get("waiting"), dict):
            approval_binding.restore_record(self._app_workflow, (parked.get("payload") or {}).pop(
                waiting_module.STORE_RECORD_KEY, None))
        state = {
            **snapshot,
            "approval": None,
            "conversation_history_turns": [],
            "action_log": list(self._action_log),
            "last_completed_turn_key": self._last_completed_turn_key,
        }
        self.apply_serialized_state(state)
        self._pending_owner_override = _OWNER_UNSET
        logger.info("restored parked pending state of conversation %s", key)

    def bound_conversation_id(self):
        """The conversation the context is bound to (None when it has none)."""
        return self._conversation_id

    def parked_conversation_ids(self) -> list[int]:
        """The conversation ids that have parked state. "none" and the unowned marker are not conversation ids."""
        return [int(key) for key in self._parked_pending if key.isdigit()]

    def drop_parked_pending(self, conversation_id) -> int:
        """Forget the pending state parked for *conversation_id* (fix-ayez): the conversation was deleted or archived.

        Only the parked entries under that id go. The live state is left alone, because it belongs to whichever
        conversation the context is bound to. Returns the number of snapshots dropped.
        """
        items = self._parked_pending.pop(_owner_key(conversation_id), None) or []
        if items:
            logger.info("dropped %d parked pending state(s) of conversation %s", len(items), conversation_id)
        return len(items)

    def serialize_state(self, *, channel_id: str) -> dict[str, Any]:
        """
        Export durable Topology-B state for cross-process resume.

        Requires session_key and bound app_workflow when persisting.
        """
        turn = self._turn_frame()
        react_blob = None
        if turn.agent is not None:
            react_blob = turn.agent.export_suspended()

        plan_cme = getattr(self, "_plan_cme", None)
        current_context_name = None
        if self._app_workflow and self._app_workflow._current_command_context is not None:
            current_context_name = self._app_workflow._current_command_context_name

        from fastworkflow.conversation_history_io import extract_turns_from_history

        nested = self._frame_stack()[1:]
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
            # Additive (ADR-EXP-025 section 7): the session's approval records and turn clock.
            "approval": self._approval_state_payload(),
            # The turn's pending resolution (which command each part gets, then its missing values; bd fix-o1lg).
            "plan_resolution": None if self._plan_resolution is None else self._plan_resolution.to_state(),
            # What each part's settling knew, kept for the learning rows written where the reviewed plan is run.
            "turn_part_facts": json.loads(json.dumps(self._turn_part_facts, default=str)) if self._turn_part_facts else None,
            "use_plan": self._turn_use_plan,
            "subtasks": list(turn.subtasks),
            "subtask_origins": list(turn.origins),
            "subtask_index": turn.subtask_index,
            "subtask_answers": list(turn.subtask_answers),
            "subtask_outputs": list(turn.subtask_outputs),
            "subtask_exhausted": turn.subtask_exhausted,
            # The turn's sub-task plan, so an accept after a restart (the review is the ``waiting`` record) runs the
            # accepted goals.
            "subtask_plan": None if turn.subtask_plan is None else turn.subtask_plan.model_dump(mode="json"),
            "next_step": self._next_step,
            "skill_phase_run": turn.phase_run,
            "react": react_blob,
            "frames": [frame.to_state() for frame in nested],
            "frame_react": [
                None if frame.agent is None else frame.agent.export_suspended()
                for frame in nested
            ],
            "turn_child_step_ranges": [list(step_range) for step_range in turn.child_step_ranges],
            "turn_steps_used": self._turn_steps_used,
            "turn": self._serialize_turn_accumulator(),
            "waiting": self._waiting_state(self._waiting, self._cme_workflow, plan_cme),
            "current_command_context_name": current_context_name,
            "action_log": list(self._action_log),
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
            "claim_epoch": self._claim_epoch,
            "server_incarnation": self._server_incarnation,
            # Additive (fix-newconv-pending): the conversation the live pending state belongs to, and the pending
            # state parked for the other conversations. Absent from older blobs (read as unowned, see apply).
            "conversation_id": _owner_encoded(self._live_pending_owner()),
            "parked_pending": {key: list(items) for key, items in self._parked_pending.items()},
        }
        # The policy verifier's memory of the conversation, only when a policy is
        # enabled: a session resumed in another process keeps what was read and
        # asked. Absent otherwise, so a state written without a policy is unchanged.
        from fastworkflow.policy_runtime import policy_of

        if policy := policy_of(self._workflow_tool_agent):
            payload["policy_state"] = policy.export_state()
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
        """
        found = state.get("schema_version", 0)
        if found != SCHEMA_VERSION:
            raise IncompatibleSessionState(found)
        # Validated up here with the version check, not applied halfway down:
        # this method's contract is "raises having applied nothing", and a
        # malformed experiment triple must not be the one exception that leaves
        # a half-restored context behind.
        if state.get("experiment_id") is not None:
            self._validate_experiment_labels(
                state.get("experiment_id"), state.get("task_id"), state.get("attempt")
            )
            if bool(state.get("claim_epoch")) != bool(state.get("server_incarnation")):
                raise IncompatibleSessionState(found)

        # Additive (fix-newconv-pending): what was parked, kept whole until its conversation is bound again.
        if "parked_pending" in state:
            self._parked_pending = {
                str(key): [dict(item) for item in items or []]
                for key, items in (state.get("parked_pending") or {}).items()
            }
        # Absent from blobs written before approval binding: no records.
        if state.get("approval") and self._app_workflow is not None:
            approval_binding.store_for(self._app_workflow).load_state(state["approval"])
        # The turn's pending resolution. Old saved sessions are not supported (owner ruling, bd fix-o1lg): the keys an
        # earlier build wrote (parameter_loop, plan_choice, skill_refusal_pending, ...) are ignored, nothing converted.
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        self._plan_resolution = resolution.PlanResolution.from_state(state.get("plan_resolution"))
        self._turn_part_facts = state.get("turn_part_facts") or None
        self._turn_use_plan = state.get("use_plan") or None
        # Absent from blobs written before sub-task splitting; the defaults mean "not split".
        turn = self._turn_frame()
        turn.subtasks = list(state.get("subtasks") or [])
        # Absent before hand-off continuations: each entry is then its own origin.
        turn.origins = list(state.get("subtask_origins") or range(len(turn.subtasks)))
        turn.subtask_index = int(state.get("subtask_index") or 0)
        turn.subtask_answers = list(state.get("subtask_answers") or [])
        turn.subtask_outputs = list(state.get("subtask_outputs") or [])
        turn.subtask_exhausted = bool(state.get("subtask_exhausted"))
        plan = state.get("subtask_plan")
        if plan:
            from fastworkflow.review_proposals import SubTaskProposal

            turn.subtask_plan = SubTaskProposal.model_validate(plan)
        else:
            turn.subtask_plan = None
        self._next_step = int(state.get("next_step") or 0)
        # Absent before planning command phases: no phased planning command call in progress.
        turn.phase_run = state.get("skill_phase_run") or None
        turn.phase_live = None
        # Absent from blobs written before nested frames: no child ranges, zero steps used.
        turn.child_step_ranges = [
            list(step_range) for step_range in state.get("turn_child_step_ranges") or []
        ]
        self._turn_steps_used = int(state.get("turn_steps_used") or 0)

        self._action_log = list(state.get("action_log") or [])
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
            if state.get("claim_epoch") is not None:
                self._claim_epoch = int(state["claim_epoch"])
                self._server_incarnation = state["server_incarnation"]

        if turns := state.get("conversation_history_turns") or []:
            from fastworkflow.conversation_history_io import restore_history_from_turns

            self._conversation_history = restore_history_from_turns(turns)

        # The turn's one waiting record comes back with its "/" round (``Waiting.assistant``), then the live pending state
        # belongs to the conversation the blob names (unowned when it names none). The binding that follows settles it
        # against the conversation the context is bound to (fix-newconv-pending).
        move = waiting_module.step(None, waiting_module.Event(waiting_module.RESTORE, state=state.get("waiting")))
        if move.record is not None and move.record.assistant:
            self.start_round()
        self._waiting = None
        plan_cme = getattr(self, "_plan_cme", None)
        if plan_cme is not None:
            self._clear_cme_continuation(plan_cme)
        self._apply_waiting(move.record)
        self._pending_owner_override = (
            _owner_from_state(state) if self._has_live_pending() else _OWNER_UNSET
        )
        if self._waiting is not None and self._waiting.kind == waiting_module.KIND_REVIEW:
            # The restored review keeps its free-text reader (set by the gate of the turn that opened it).
            self._give_review_reader()
        self._apply_turn_accumulator(state.get("turn"))

        # Restore the turn agent with the turn frame on top: _workflow_tool_agent reads the top frame.
        del self._frame_stack()[1:]
        if state.get("policy_state") is not None:
            from fastworkflow.policy_runtime import policy_of

            self._ensure_agent_initialized()
            if policy := policy_of(self._workflow_tool_agent):
                policy.import_state(state["policy_state"])
        react_blob = state.get("react")
        if react_blob and self._awaiting_user:
            self._ensure_agent_initialized()
            if self._workflow_tool_agent is not None:
                self._workflow_tool_agent.import_suspended(react_blob)

        # Absent from blobs written before nested frames: no nested frames.
        nested = [PlanFrame.from_state(frame_state) for frame_state in state.get("frames") or []]
        frame_react = state.get("frame_react") or [None] * len(nested)
        for frame, blob in zip(nested, frame_react):
            if blob and self._awaiting_user:
                self._attach_nested_agent(frame)
                frame.agent.import_suspended(blob)
        self._frames.extend(nested)

        saved_context_name = state.get("current_command_context_name")
        if (
            saved_context_name
            and self._app_workflow
            and self._app_workflow._current_command_context is not None
            and self._app_workflow._current_command_context_name != saved_context_name
        ):
            logger.debug(
                "Command context name after rehydrate (%s) differs from saved (%s); "
                "navigation depth may not match until workflow-specific restore is added",
                self._app_workflow._current_command_context_name,
                saved_context_name,
            )

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
        self._ask_user_rounds = list(turn.get("ask_user_rounds") or [])
        self._turn_subtask_review = turn.get("subtask_review") or "not_applicable"
        self._turn_not_run = list(turn.get("not_run") or [])
        self._turn_unattended = bool(turn.get("unattended"))

        if agent_result := turn.get("agent_result"):
            self._turn_agent_result = _RestoredAgentResult(
                exhausted=bool(agent_result.get("exhausted"))
            )

    def _apply_waiting(self, held: Optional[waiting_module.Waiting]) -> None:
        """Restore the turn's waiting record so the next message continues it. A review, a decision step, an approval and
        the agent's own question come back as they were (a review's slot question is restored into it as the plan CME's
        own record, below); a question of the which-command or values kind goes through ``_apply_question``."""
        if held is None:
            return
        if held.kind not in waiting_module.CME_KINDS:
            slot = held.payload.pop(waiting_module.SLOT_KEY, None)
            if held.kind == waiting_module.KIND_REVIEW:
                held.payload[waiting_module.SLOT_KEY] = None
            self._waiting = held
            if slot is not None:
                self._apply_question(slot, cme_workflow=getattr(self, "_plan_cme", None),
                                     subject_workflow=self._plan_workflow)
            return
        self._apply_question(held)

    def _apply_question(
        self,
        held: waiting_module.Waiting,
        *,
        cme_workflow: Optional[fastworkflow.Workflow] = None,
        subject_workflow: Optional[fastworkflow.Workflow] = None,
    ) -> None:
        """Restore a which-command or values question (the CME's, the plan path's, or a review's slot).

        The record is plain data and comes back as it was saved; the CME's scratch keys ``command`` and ``command_name``
        are installed from its payload. A parameter loop's extractor rebuilds the command's ``Input`` from the loop
        through model_construct, because the open fields hold NOT_FOUND sentinels, which is precisely the state validation
        exists to reject.
        """
        if held.kind not in waiting_module.CME_KINDS:
            return
        if held.owner == waiting_module.OWNER_PLAN:
            self._turn_port().put(held)      # a plan question: no CME scratch to install
            return

        if cme_workflow is None:
            cme_workflow = self._cme_workflow
        if cme_workflow is None:
            return

        context = cme_workflow.context
        payload = dict(held.payload)
        command, command_name = payload.pop("command", None), payload.pop("command_name", None)
        held.payload = payload

        if command is not None:
            context["command"] = command
        if command_name is not None:
            context["command_name"] = command_name
        if held.kind == waiting_module.KIND_PARAMETERS:
            if command_name is None:
                return
            params_class = self._command_parameters_class(
                command_name, subject_workflow=subject_workflow,
            )
            if params_class is None:
                # Losing the partial parameters is bad, but resuming into a command
                # whose parameter class no longer exists is worse. Reset to intent
                # detection so the next message is routed rather than merged into a
                # command that cannot be completed.
                logger.warning(
                    f"Cannot restore the parameter loop for '{command_name}': its "
                    f"parameter class is gone. Resetting to intent detection."
                )
                context.pop("command", None)
                context.pop("command_name", None)
                return
        waiting_module.port_of(context).put(held)

    def _command_parameters_class(
        self,
        command_name: str,
        *,
        subject_workflow: Optional[fastworkflow.Workflow] = None,
    ):
        """The Input class for a command name, or None if it cannot be resolved."""
        if subject_workflow is None:
            subject_workflow = self._app_workflow
        if subject_workflow is None:
            return None
        from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import parameters_class_of

        try:
            return parameters_class_of(subject_workflow.folderpath, command_name, pending=True)
        except Exception:
            return None

    def cancel_pending(self) -> bool:
        """
        Abort a pending ask_user clarification (Topology B).

        ask_user is non-blocking in Topology B (the clarification is returned as a
        CommandOutput), so a suspended trajectory never hangs — it simply waits in
        memory. Embedders call this to abandon it per their own session lifecycle
        (e.g. request timeout, user navigated away).

        Drops every kind of pending question silently: the agent's own, an approval or review, and the one
        the deterministic path (the CME) holds, which a reply would otherwise answer.

        Returns True if a pending clarification was cleared, False otherwise.
        """
        cancelled_approvals = self.cancel_pending_approvals()
        if not self._awaiting_user:
            if self._waiting is None:
                return cancelled_approvals > 0
            if self.has_open_command():
                self._clear_cme_continuation()
            plan_cme = getattr(self, "_plan_cme", None)
            if plan_cme is not None:
                self._clear_cme_continuation(plan_cme)
            self._waiting = None
            self.end_round()
            return True
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
    ) -> None:
        """Append one turn to conversation history in the canonical 2-key shape.

        Two keys, the shape `conversation_history_io` restores. The third was
        `feedback`, and fix-9eg.16 removed the agent-memory table that was its
        only source, so it could only ever be None here -- while
        `_refine_user_query` renders every key of every remembered turn into the
        refiner's prompt, which put a literal `feedback: None` line in front of
        the model on in-session turns and nowhere on restored ones (fix-24da).
        """

        self._conversation_history.messages.append(
            {
                "conversation summary": conversation_summary,
                "conversation_traces": conversation_traces,
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
        """Bind the app workflow for NLU (Path 1) and execution (Path 2).

        A context whose caller chose no sink opens the workflow's own sink here
        (see ``_open_auto_sink``).
        """
        self._app_workflow = workflow
        self._cme_workflow.context["app_workflow"] = workflow
        if (store := approval_binding.store_for(workflow)) is not None:
            store.owner_ref = weakref.ref(self)     # the approval store's records end this turn's record with them
        if self.no_agent_mode:
            # Read by the CME's wildcard to say why a request the deterministic path cannot route was not planned.
            self._cme_workflow.context["no_agent_mode"] = True
        # getattr: a context built through ``__new__`` without ``__init__`` has
        # no flag, and is treated as having chosen its sink.
        if not getattr(self, "_trace_sink_supplied", True):
            self._open_auto_sink(workflow)

    def _on_app_context_change(self) -> None:
        """Context-change observer: refresh the ReAct agent's available_commands.

        A plain goal bound to its command keeps its narrowed listing through a context switch (fix-3rsq.plan-review.2.9):
        the listing is the bound command's, re-read from the workflow now. Only the turn's own agent (the turn frame).
        """
        from fastworkflow.workflow_agent import _refresh_agent_available_commands

        bound = getattr(self, "_narrow_to", None)
        if bound is not None and self._app_workflow is not None and len(self._frame_stack()) == 1:
            from fastworkflow import run_listing

            inputs = getattr(self._workflow_tool_agent, "inputs", None)
            if isinstance(inputs, dict) and "available_commands" in inputs:
                inputs["available_commands"] = run_listing.bound_command_listing(self._app_workflow, bound)
                return
        _refresh_agent_available_commands(self)

    def _close_workflow_safely(self, workflow: Optional[fastworkflow.Workflow]) -> bool:
        if workflow is None:
            return True
        try:
            return workflow.close()
        except ValueError:
            # Child workflows should not occur; ignore if mis-invoked.
            logger.debug(
                "WorkflowExecutionContext.close: workflow %s is not a root session",
                getattr(workflow, "id", workflow),
            )
            return False

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

        ok = True
        ok = self._close_workflow_safely(self._cme_workflow) and ok
        ok = self._close_workflow_safely(
            getattr(self, "_plan_cme", None),
        ) and ok
        ok = self._close_workflow_safely(
            getattr(self, "_plan_workflow", None),
        ) and ok
        return ok

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

    def process_turn(
        self, message: str, *, unattended: bool = False,
        approval_surface: "approval_binding.Surface | None" = None,
        approval_token: Optional[str] = None,
        use_plan: Optional[str] = None,
    ) -> "fastworkflow.TurnOutput":
        """
        Execute one user message synchronously and return the public TurnOutput.

        ``use_plan`` names a planning command available here (as ``fw_not_run``'s ``available_plans`` lists
        them): the message is then ONE part, not split, and its planning-command match is that command, so the turn
        goes on as a turn Jev matched it would (arguments drafted from the message, a missing required one asked
        for, the command's steps). Raises ``UsePlanError`` for a name that is not a planning command of the current
        context, and when the message answers a waiting question (a reply starts no turn).

        ``unattended=True`` runs the turn with no one to answer: the agent's
        ask_user tool is removed for the whole logical turn. It applies to the
        turn this message begins; a message answering a suspended turn
        continues that turn as it began.

        Shares dispatch with _execute_message(); additionally captures every
        command execution of the logical turn (including ask_user exchanges). The
        full internal TurnResult is built and projected onto the slim public
        TurnOutput (see docs/turn_result_design_final.md section 1a).
        """
        self.check_use_plan(use_plan)
        self._next_use_plan = use_plan or None
        self._note_approval_inbound(approval_surface, approval_token, unattended)
        try:
            command_output = self._execute_message(message, unattended=unattended)
        except Exception as exc:
            approval_binding.clear_inbound(self._app_workflow)
            # A turn that raised is still a turn: record it as failed, with its
            # spans closed under it, before the error propagates. Without this
            # the turn had no record at all and read as lost.
            first_line = (str(exc).strip().splitlines() or [""])[0]
            try:
                self._build_turn_result(
                    None, failure_reason=f"{type(exc).__name__}: {first_line}"[:500])
            except Exception:  # noqa: BLE001 - never mask the turn's own error
                logger.warning("could not record the failed turn", exc_info=True)
            raise
        approval_binding.clear_inbound(self._app_workflow)
        turn_result = self._build_turn_result(command_output)
        return turn_result.turn_output

    def check_use_plan(self, use_plan: Optional[str]) -> None:
        """Refuse a ``use_plan`` that cannot be honoured, before the turn begins (``UsePlanError``)."""
        if not use_plan:
            return
        if self._app_workflow is not None and self._question_is_out():
            raise UsePlanError(AWAITING_ANSWER_REFUSAL_MESSAGE)
        if self._app_workflow is None or use_plan not in no_agent_mode.plan_names(self._app_workflow):
            raise UsePlanError(f"use_plan {use_plan!r} is not a planning command available here")

    @dspy_logger.observe_dspy_calls
    def _execute_message(
        self, message: str, unattended: bool = False
    ) -> fastworkflow.CommandOutput:
        """Shared message dispatch for _execute_message()/process_turn()."""
        self.assert_experiment_claim_current()
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before executing a message"
            )

        # fix-newconv-pending: a message never answers pending state that belongs to another conversation. The live
        # state is parked (or its approvals cancelled) first, so this message begins a new turn.
        self._settle_pending_for_conversation(self._conversation_id)

        # ONE decision of what the message is and what happens to it (``waiting.step``): a reply to the question that is
        # out (its leading "/" removed, nothing else reads that "/") or a new request, and which stage takes it.
        move = self._route(message)
        message = move.text
        if move.begin:
            # A message during suspension is the resume answer — never a reset.
            self._begin_turn(message, unattended=unattended)

        # The trace host of the turn's own decisions (the plan path's ``fw.nlu.intent`` and ``fw.nlu.param_extraction``
        # spans are written several frames down, with no reference to this context, as the "/" path's are).
        with tracing.host_scope(self):
            self.push_active_workflow(self._app_workflow)
            try:
                if move.action == waiting_module.ACT_MANAGE and (
                        managed := approval_binding.management_reply(self, self._app_workflow, move.text)) is not None:
                    return self._approval_text_output(managed)
                if move.action == waiting_module.ACT_APPROVAL:
                    return self._resume_approval_direct(message)
                if move.action in (waiting_module.ACT_RESUME_AGENT, waiting_module.ACT_START_AGENT):
                    # The plan and agent path check a split part's text or an agent's own text; the duplicate rule
                    # counts a value no more often than the user's own request writes it (``check.written_times``).
                    # The "/" path checks the user's own message and binds nothing.
                    with one_guard.user_written(self._turn_user_message):
                        if move.action == waiting_module.ACT_RESUME_AGENT:
                            return self._resume_agent_message(message)
                        return self._process_agent_message(message)
                return self._process_message(message)
            except CommandCancelledError as exc:
                self._reset_agent_suspension()
                return self._command_cancelled_output(str(exc))
            finally:
                self.pop_active_workflow()
                if self._app_workflow:
                    self._app_workflow.flush()

    def _build_turn_result(
        self, command_output: Optional[fastworkflow.CommandOutput],
        failure_reason: Optional[str] = None,
    ) -> TurnResult:
        """Assemble the TurnResult (and its public turn_output) for the message.

        The turn's ``answer`` is plain text — the agent's final answer (or the
        deterministic command's response text). Per-command structured results
        (success/artifacts) live on ``command_outputs``.

        ``failure_reason`` set means the turn raised instead of producing an
        output: it is recorded as FAILED with that reason.
        """
        answer = command_output.command_response.response if command_output else ""

        if failure_reason is not None:
            status = TurnStatus.FAILED
            completed_at: Optional[datetime] = datetime.now(timezone.utc)
        elif self._awaiting_user:
            status = TurnStatus.AWAITING_USER
            completed_at: Optional[datetime] = None
        else:
            status = TurnStatus.COMPLETED
            completed_at = datetime.now(timezone.utc)
            if self._turn_agent_result is not None:
                if getattr(self._turn_agent_result, "exhausted", False):
                    # The turn failed to complete (agent ran out of iterations).
                    # status carries the failure; failure_reason elaborates it.
                    # Orthogonal to TurnOutput.success (command success codes).
                    status = TurnStatus.FAILED
                    failure_reason = "max_iters_exhausted"
            elif self._turn_outputs:
                # Deterministic/assistant path: answer text is the last captured
                # output's first response text [A33]. A command-level failure is
                # surfaced by TurnOutput.success (all command_outputs succeeded),
                # not by status/failure_reason.
                answer = self._turn_outputs[-1].command_response.response

        # The approval token is never in an output: it reaches the host through
        # approval_binding.take_delivery, once, on the live response (ADR section 6.4).
        if status == TurnStatus.AWAITING_USER or self._last_output_awaits_user():
            self._describe_pending_prompt()
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

        turn_result = TurnResult(
            turn_output=turn_output,
            channel_id=self._channel_id,
            conversation_id=self._conversation_id,
            experiment_id=self._experiment_id,
            task_id=self._task_id,
            attempt=self._attempt,
            claim_epoch=self._claim_epoch,
            server_incarnation=self._server_incarnation,
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
        )

        self._finalize_turn_trace(turn_result)
        return turn_result

    #: The artifact keys _describe_pending_prompt owns.
    _PROMPT_DESCRIPTION_KEYS = ("question_kind", "approval", "review", "parameter_request", PHASE_STEP_ARTIFACT,
                                PLAN_CHOICE_ARTIFACT)

    @classmethod
    def _without_prompt_description(cls, output):
        """*output*, or for an ask_user entry a copy whose response lacks the prompt-description keys."""
        if not output.is_ask_user or not any(
                key in output.command_response.artifacts for key in cls._PROMPT_DESCRIPTION_KEYS):
            return output
        kept = {k: v for k, v in output.command_response.artifacts.items()
                if k not in cls._PROMPT_DESCRIPTION_KEYS}
        return output.model_copy(update={
            "command_response": output.command_response.model_copy(update={"artifacts": kept})})

    def _last_output_awaits_user(self) -> bool:
        """A user-originated "/" approval returns ``awaiting_output`` without suspending: its turn is
        COMPLETED, and only the output's ``awaiting_user`` artifact says it waits."""
        if not self._turn_outputs:
            return False
        return self._turn_outputs[-1].command_response.artifacts.get("awaiting_user") is True

    def _describe_pending_prompt(self) -> None:
        """Tag the pending prompt's entry (``artifacts``) with ``question_kind`` and its payload.

        Additive and best effort: it never raises into the turn (a failure tags "unknown"). The keys live
        only in the entry's artifacts, which the agent never reads (it sees the user's answer text), and
        are refreshed on every awaiting turn because a review edit re-prompts without a new entry.
        """
        try:
            target = self._turn_outputs[-1] if self._turn_outputs else None
            if target is None:
                return
            artifacts = target.command_response.artifacts
            if not (
                (target.is_ask_user and target.command_response.success is False)
                or artifacts.get("awaiting_user") is True
            ):
                return
        except Exception:  # noqa: BLE001 - never raises into the turn
            return
        payload_keys = ("approval", "review", "parameter_request", PHASE_STEP_ARTIFACT, PLAN_CHOICE_ARTIFACT)
        kind = "unknown"
        payload: dict[str, Any] = {}
        try:
            kind, payload = self._pending_prompt_kind(target)
        except Exception as exc:  # noqa: BLE001 - a lookup failure tags "unknown", with no payload
            logger.debug(f"pending prompt not described: {type(exc).__name__}: {exc}")
            kind, payload = "unknown", {}
        try:
            for key in payload_keys:
                artifacts.pop(key, None)
            artifacts.update(payload)
            artifacts["question_kind"] = kind
        except Exception:  # noqa: BLE001 - never raises into the turn
            pass

    @property
    def waiting(self):
        """The one question this turn is waiting on (``fastworkflow.waiting.Waiting``) or None: the turn's record itself."""
        return self._waiting

    def _pending_prompt_kind(self, target) -> tuple[str, dict[str, Any]]:
        """``(question_kind, payload)`` of the prompt *target* is; raises on any lookup failure. The kind is the record's
        (``waiting.TABLE``) and its payload is built by ``prompt_payloads``; the agent's own question, and the CME's open
        command beside an ``ask_user`` entry, are ``agent`` with none."""
        from fastworkflow import prompt_payloads

        held = self._waiting
        if held is not None and held.owner != waiting_module.OWNER_CME:
            if (found := prompt_payloads.payload_of(self, held))[0]:
                return found
        return ("agent", {}) if target.is_ask_user else ("unknown", {})

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
        the record is emitted so the suspended turn is visible; the
        terminal finalize closes the same deterministic span id —
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
                    # The root span belongs to the turn frame, whose alias is "" (design 3.5).
                    # frame_alias is an envelope attribute: tracing.end_span stamps it from the
                    # frame on top, which is the turn frame here, so no emitter writes it.
                },
                experiment_id=self._experiment_id,
                task_id=self._task_id,
                attempt=self._attempt,
                claim_epoch=self._claim_epoch,
                server_incarnation=self._server_incarnation,
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
                "subtask_review": self._subtask_review_mark(),
                "subtask_bindings": list(getattr(self, "_turn_subtask_bindings", None) or []) or None,
                "subtask_plan_ambiguity": list(getattr(self, "_turn_plan_ambiguity", None) or []) or None,
                "subtask_split": tracing.split_record_attribute(getattr(self, "_turn_split", None)),
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
            # No sink and no metrics sink (a store that could not be opened,
            # or a host built without either): nothing consumes the
            # TurnResult, so skip building it — this path runs after EVERY
            # turn and must cost ~nothing when there is nowhere to send it.
            return
        self._build_turn_result(command_output)

    def process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        self.assert_experiment_claim_current()
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before process_action()"
            )
        if self._question_is_out():
            # The suspended turn is still waiting for its answer: it is neither ended nor
            # begun over, and its frames stay as they are.
            return fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=AWAITING_ANSWER_REFUSAL_MESSAGE, success=False),
                command_name=action.command_name or "",
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
        self, action: fastworkflow.Action, *,
        approval_surface: "approval_binding.Surface | None" = None,
    ) -> "fastworkflow.TurnOutput":
        """
        Execute one direct action synchronously and return the public TurnOutput.

        Mirror of process_turn() for the direct-action path: same dispatch as
        process_action() (each direct action is its own logical turn),
        additionally building the full internal TurnResult and projecting it onto
        the slim public TurnOutput. This lets callers (e.g. the run_fastapi_mcp
        turn registry) store exactly one result type across both the message and
        action paths.
        """
        self._note_approval_inbound(approval_surface, None, False)
        try:
            command_output = self.process_action(action)
        finally:
            approval_binding.clear_inbound(self._app_workflow)
        if self._question_is_out():
            # The refusal: the suspended turn's record was emitted when it suspended and
            # stands; building the result again would emit it once more with the refusal.
            return fastworkflow.TurnOutput(
                turn_key=self._turn_key or mint_turn_key(),
                status=TurnStatus.AWAITING_USER,
                answer=command_output.command_response.response,
                command_outputs=list(self._turn_outputs),
            )
        turn_result = self._build_turn_result(command_output)
        return turn_result.turn_output

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _question_is_out(self) -> bool:
        """A question of any kind is waiting for its answer: the turn's waiting record."""
        return self._waiting is not None

    def _route(self, message: str) -> waiting_module.Move:
        """The one decision of what an incoming message is and which stage takes it (``waiting.step``): the move says whether
        a turn begins, the text the stages read (a reply has its leading "/" removed, the management messages
        ``/approvals``, ``/revoke_approval`` and ``/cancel_pending`` keep theirs) and, when nothing is out, that a "/"
        message opens the assistant round (applied here)."""
        text = message.strip()
        move = waiting_module.step(self._waiting, waiting_module.Event(waiting_module.MESSAGE, message), waiting_module.Facts(
            agent_path=self._run_as_agent, assistant_round=self._assistant_round,
            manage=text.startswith((approval_binding.MANAGE_LIST, approval_binding.MANAGE_REVOKE)),
            cancel_pending=text == approval_binding.CANCEL_COMMAND))
        if move.assistant:
            self.start_round()
        return move

    def _route_message(self, message: str) -> str:
        """The text the stages below read for *message* (``_route``); a "/" message with nothing out opens the round."""
        return self._route(message).text

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

    def _build_react_agent(self):
        """A workflow-tool ReAct agent for one run; its steps number from this context's next step."""
        from fastworkflow.workflow_agent import initialize_workflow_tool_agent
        agent = initialize_workflow_tool_agent(self)
        agent.step_floor = lambda: self._next_step
        return agent

    def _initialize_agent_functionality(self) -> None:
        self._cme_workflow.context["run_as_agent"] = True
        if self._app_workflow:
            self._app_workflow.context["run_as_agent"] = True

        self._workflow_tool_agent = self._build_react_agent()

        # Re-scope the active ReAct agent's available_commands whenever the context changes,
        # driven by the workflow's context-change observer (the single switch chokepoint)
        # Registered once per WEC (agent init runs once). The listener reads the *active* agent
        # dynamically. No-ops when no agent is running. Bound method + remove on close()
        # (and WeakMethod storage on Workflow) avoid listener leaks.
        if self._app_workflow is not None:
            self._app_workflow.add_context_change_listener(self._on_app_context_change)
            self._context_change_listener = self._on_app_context_change

    def _ensure_agent_initialized(self) -> None:
        if self._workflow_tool_agent is None:
            self._initialize_agent_functionality()

    def _reset_agent_suspension(self) -> None:
        """Clear Topology-B ask_user suspend state (abort, finalize, or cancel_pending)."""
        # The resolution stays (as it did); the next turn begins without it. The question is no longer out: the agent's own question, the turn's open review, a
        # paused decision step and the plan's questions go with the suspension, whichever route ended it. An approval
        # ends with the approval store's record, and the CME's open command with its command.
        if self._waiting is not None and self._waiting.kind != waiting_module.KIND_APPROVAL and (
                self._waiting.owner != waiting_module.OWNER_CME):
            self._waiting = None
        # A suspended nested frame (run_planning_command) is dropped with the suspension;
        # what is cleared below belongs to the turn frame, the only frame left.
        del self._frame_stack()[1:]
        turn = self._turn_frame()
        self._clear_subtask_state()
        turn_agent = turn.agent
        if turn_agent is not None and hasattr(turn_agent, "clear_suspension"):
            turn_agent.clear_suspension()

    def _clear_subtask_state(self) -> None:
        turn = self._turn_frame()
        turn.subtasks = []
        turn.origins = []
        turn.subtask_index = 0
        turn.subtask_answers = []
        turn.subtask_outputs = []
        turn.subtask_exhausted = False
        self._next_step = 0
        turn.evidence_floors = {}
        turn.phase_run = None
        turn.phase_live = None
        # Step numbers restart at 0 with _next_step, so ranges from an earlier run no longer apply.
        turn.child_step_ranges = []
        self._enable_all_tools()

    def _enable_all_tools(self) -> None:
        """Lift any tool restriction a reply left on the agent (see _apply_reply_intent).

        An unattended turn keeps ask_user off: its baseline has no one to ask.
        """
        if self._workflow_tool_agent is not None:
            self._workflow_tool_agent.disabled_tools = (
                frozenset({"ask_user"}) if self._turn_unattended else frozenset()
            )

    def _agent_dspy_context(self):
        """Return (lm, adapter) for agent-mode dspy.context blocks."""
        lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
        from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter

        return lm, CommandsSystemPreludeAdapter()

    def _call_agent_with_retry(self, agent_call, lm=None, *, trace_input=None,
                               resumed=False):
        """Run agent_call under an agent dspy.context, retrying on AdapterParseError.

        lm: optional LM override (e.g. distillation's teacher/student model). When
        omitted, the default agent context (LLM_AGENT) is used.

        This is the one choke point both the fresh forward and the resume pass
        through, so it is where the executor phase is recorded: fw.agent.execute
        wraps the whole loop (retries included), and ``host_scope`` binds this
        context so ReAct's per-iteration fw.agent.step spans — several frames
        down, with no reference to the WEC — reach the same sink.
        """
        from dspy.utils.exceptions import AdapterParseError

        # No-agent mode: no agent LM is built (an unset LLM_AGENT would raise), the run takes no model step
        # (``no_model``), and code alone runs what it can. A run that is given an LM still gets its own.
        agent_adapter = None
        # (getattr: stand-in hosts of this method in tests carry only the attributes it reached for before.)
        no_model = lm is None and getattr(self, "no_agent_mode", False) is True
        if no_model:
            if (agent := getattr(self, "_workflow_tool_agent", None)) is not None:
                agent.no_model = True
        else:
            default_lm, agent_adapter = self._agent_dspy_context()
            if lm is None:
                lm = default_lm
        max_retries = 2
        span = tracing.start_span(
            self,
            tracing.SPAN_AGENT_EXECUTE,
            attributes={
                "agent_input": trace_input,
                "resumed": resumed,
                "model": getattr(lm, "model", None),
            },
        )
        attempts = 0
        try:
            with tracing.host_scope(self):
                for attempt in range(max_retries):
                    attempts = attempt + 1
                    try:
                        with (contextlib.nullcontext() if no_model
                              else dspy.context(lm=lm, adapter=agent_adapter)):
                            result = agent_call()
                    except AdapterParseError:
                        if attempt == max_retries - 1:
                            raise
                        # A tool already ran: its side effects and its archived
                        # evidence rows exist under this turn's keys, which a
                        # retry would reuse. A resumed turn always qualifies,
                        # because resume writes the observation key first.
                        trajectory = getattr(
                            self._workflow_tool_agent, "trajectory", None) or {}
                        if any(str(key).startswith("observation_") for key in trajectory):
                            raise
                        continue
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
        # Every retry raised AdapterParseError but the last re-raise was
        # swallowed by the loop shape: close the span rather than leak it.
        tracing.end_span(self, span, status=tracing.STATUS_ERROR,
                         attributes={"attempts": attempts})

    def _run_agent(self, message: str):
        """Fresh agent turn setup and ReAct forward call, with the turn's stage progress listened for (fix-kq49)."""
        from fastworkflow import turn_progress

        with turn_progress.bound(self):
            return self._run_agent_unlistened(message)

    def _run_agent_unlistened(self, message: str):
        """Fresh agent turn setup and ReAct forward call."""
        self.clear_action_log()

        if self._app_workflow:
            self._app_workflow.context["raw_user_message"] = message
        from fastworkflow.policy_runtime import policy_of

        if policy := policy_of(self._workflow_tool_agent):
            policy.observe_turn(message)

        refined_user_query = self._refine_user_query(message, self.conversation_history)
        self._turn_refined_message = refined_user_query
        self._clear_subtask_state()

        # A workflow with planning commands answers a plain "what can you do?" in code, with no model call.
        from fastworkflow.capability_answer import capability_turn

        if self._app_workflow and not self._turn_use_plan and (capable := capability_turn(
                message, self._app_workflow, self._cme_workflow.folderpath)) is not None:
            # The listing as data rides on the prediction; _finalize_agent_output puts it in the answer's artifacts.
            return dspy.Prediction(final_answer=capable[0], exhausted=False, capabilities=capable[1])

        from fastworkflow import turn_progress
        from fastworkflow import workflow_agent

        turn_progress.emit(turn_progress.STAGE_SPLIT)
        # The split's request is the user's new message; the earlier turns go in as the splitter's separate
        # conversation input, never prepended to the request: the merge returns the request itself as the one job,
        # so a prepended summary became the sub-task's goal, and its names changed how the request merged
        # (fix-bydz). An unsplit request still runs on refined_user_query (_run_turn_subtasks).
        # A message sent with ``use_plan`` is one part: the client named the part and its plan, so it is not split.
        if self._turn_use_plan:
            subtasks = [message]
        else:
            with workflow_agent.recording_split() as split_record:
                subtasks = workflow_agent.split_into_subtasks(
                    message,
                    conversation=self._conversation_context(self.conversation_history),
                    planner_lm=getattr(self, "_current_planner_lm", None),
                    workflow_folderpath=self._app_workflow.folderpath if self._app_workflow else None,
                    workflow=self._app_workflow,
                )
            self._turn_split = dict(split_record)
        if self._app_workflow is not None:
            # The review gate (fix-3rsq.plan-review.2.6; bd fix-o1lg S4/S5): None when the sub-tasks run now (the fast
            # path), else a suspended question or review, or the turn's result.
            asked = self._subtask_gate(subtasks)
            if asked is not None:
                return asked
        return self._run_turn_subtasks(subtasks, refined_user_query, message)

    def _run_turn_subtasks(self, subtasks: list[str], refined_user_query: str, message: str):
        """Run the turn's sub-tasks: several, one after another; or one, with the step-cap hand-off. A turn with a
        sub-task plan (accepted, or the draft) runs through _run_subtasks_from, which binds each goal to its means."""
        plan = self._turn_subtask_plan
        planned = plan is not None and len(plan.items) == len(subtasks)
        if planned:
            self._record_planned_parts(plan, reviewed=self._turn_subtask_review == "reviewed")
        if planned and any(item.parts for item in plan.items):
            # A joined item (a split over the cap) that the review left as drafted runs its parts as jobs of their own.
            from fastworkflow import review_proposals
            subtasks, plan = review_proposals.expand_joined_parts(
                self._app_workflow, subtasks, plan, planner_lm=getattr(self, "_current_planner_lm", None))
            self._turn_subtask_plan = plan
        if len(subtasks) > 1 or planned:
            self._subtasks = subtasks
            self._origins = list(range(len(subtasks)))
            return self._run_subtasks_from(0)

        # An unsplit request is one sub-task, so a run that hits the step cap is handed off the same way.
        self._subtasks = [refined_user_query]
        self._origins = [0]
        self._evidence_floors = {0: 0}
        result = self._call_agent_for_query(refined_user_query, message, evidence_floor=0)
        if getattr(result, "exhausted", None) is not True:
            return result
        self._record_subtask_answer(result)
        self._hand_off(0)
        return self._run_subtasks_from(1)

    # ------------------------------------------------------------------
    # The sub-task review gate (fix-3rsq.plan-review.2.2, .2.3, .2.6, .2.8)
    # ------------------------------------------------------------------

    def _subtask_gate(self, subtasks: list[str]):
        """Decide how the turn's sub-tasks run, after split and merge, before any of them runs.

        Which command or plan each part gets is ONE decision (``resolution.part_intent``, bd fix-o1lg). Fast path (one
        sub-task that names a command, or an execution command the local model is confident of): no review, today's
        run; returns None. Otherwise every part is resolved first, in order (``resolution.step``: which command when
        the decision is ambiguous, or with no agent model when nothing serves it; then a bound plan's missing values),
        each question suspending the turn; then ONE review of the whole turn, or the run when there is none
        (``_after_resolution``). Returns the suspended prediction of a question or of the review, or the turn's result.
        """
        from fastworkflow import review_proposals
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        workflow = self._app_workflow
        pinned = self._turn_use_plan
        # A pinned plan (``use_plan``) is not a fast path: the client chose the plan, not its arguments, so the part is
        # drafted for that plan and its parameter question is that turn's review.
        intent = (resolution.part_intent(workflow, subtasks[0], part=1)
                  if not pinned and len(subtasks) == 1 else None)
        if intent is not None and resolution.is_fast_path(intent, workflow):
            self._turn_subtask_review = "fast_path"
            # A single settled command, execution or planning, is run bound, in both modes (``_run_plain_means``,
            # ``_run_planning_means``: the agent cannot run a different one), so it is drafted as a one-item plan, not
            # left to the agent, as a command the user picked is (bd fix-ftdl.7.8).
            self._turn_subtask_plan = review_proposals.draft_round1_proposal(
                workflow, subtasks, planner_lm=getattr(self, "_current_planner_lm", None))
            # A value it lacks is asked for at the gate, by the part's values question, as a picked command's is (one
            # place asks a missing value): the part is resolved first and runs, with no review, when the reply
            # completes it. The learning row is then written where the plan runs (``_record_planned_parts``).
            if (question := self._fast_path_values_question(subtasks)) is not None:
                return dspy.Prediction(suspended=True, clarification=question, exhausted=False)
            resolution.record_part(workflow, subtasks[0], intent.command, resolution.how_settled(intent.decided_by),
                                   intent.evidence)
            return None
        # The draft binds only what the decision matched; an ambiguous part is asked about below, before the review.
        self._turn_subtask_plan = review_proposals.draft_round1_proposal(
            workflow, subtasks, planner_lm=getattr(self, "_current_planner_lm", None), pinned_plan=pinned)
        self._turn_plan_ambiguity = []
        self._plan_resolution = resolution.start(workflow, subtasks, pinned=pinned)
        self._turn_subtask_plan, question = resolution.step(
            self._plan_resolution, self._turn_subtask_plan, self._resolution_turn())
        if question is not None:
            return dspy.Prediction(suspended=True, clarification=question, exhausted=False)
        return self._after_resolution()

    def _fast_path_values_question(self, subtasks: list[str]) -> Optional[str]:
        """The values question of the gate's fast-path part (one part that names a command, or an execution command the
        local model is sure of), or None when it lacks nothing or nobody asks. The part is resolved at its values stage, marked
        ``fast_path`` so that, once answered, it runs with no review like a part whose command the user picked."""
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        record = resolution.start(self._app_workflow, subtasks)
        record.stage = resolution.STAGE_VALUES
        record.parts[0]["fast_path"] = True
        plan, question = resolution.step(record, self._turn_subtask_plan, self._resolution_turn())
        if question is None:
            return None
        self._plan_resolution, self._turn_subtask_plan, self._turn_plan_ambiguity = record, plan, []
        return question

    @staticmethod
    def _learning_how(record, part) -> Optional[str]:
        """How a resolved part's command was settled, for `intent_learning`: a pick at a question or a pinned plan is
        the user's choice, else the link that decided it."""
        from fastworkflow import intent_learning
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        if part.get("chosen") or record.pinned:
            return intent_learning.HOW_USER_CHOICE
        return resolution.how_settled(part.get("decided_by") or "")

    def _record_planned_parts(self, plan, *, reviewed: bool) -> None:
        """The resolution path's one writer for a planned turn (the fast path of a single part writes its own row in
        ``_subtask_gate``): called where the turn's plan is run, with the plan as it stands, once per turn. A part the
        user chose (a pick at a question, a pinned plan) is that choice; a part of a REVIEWED turn is the review's
        confirmation of the command it showed for it (accepted), or the corrected command (switched, the drafted one
        kept as ``rejected_label``); any other part is the link's own answer, unconfirmed. A part with no command, or
        whose text the review changed, is not recorded. Never raises."""
        from fastworkflow import intent_learning as learning
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        facts = getattr(self, "_turn_part_facts", None)
        self._turn_part_facts = None
        try:
            for item in plan.items if facts else ():
                fact, command = facts.get(item.goal), item.means.name
                if fact is None or not command:
                    continue
                how, rejected = fact["how"], None
                if how != learning.HOW_USER_CHOICE and reviewed:
                    switched = fact["drafted"] != command
                    how = learning.HOW_REVIEW_SWITCH if switched else learning.HOW_REVIEW_ACCEPT
                    rejected = fact["drafted"] if switched else None
                if how:
                    resolution.record_part(self._app_workflow, item.goal, command, how, fact["evidence"], rejected)
        except Exception as error:  # noqa: BLE001 - recording never fails a turn
            logger.debug(f"intent learning: the turn's parts were not recorded ({error})")

    def _resolution_turn(self):
        """What the resolution's stage logic needs from this turn (``resolution.Turn``)."""
        from fastworkflow import workflow_agent
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        return resolution.Turn(workflow=self._app_workflow, no_agent=self.no_agent_mode,
                               can_ask=waiting_module.can_ask(self),
                               planner_lm=getattr(self, "_current_planner_lm", None), port=self._turn_port())

    @property
    def _resolution_pending(self) -> bool:
        """Whether a plan-path question is out (which command, a bound plan's values, or a running part's values), so
        that the next message is its reply."""
        held = self._waiting
        return held is not None and held.owner == waiting_module.OWNER_PLAN and held.kind in waiting_module.CME_KINDS

    def _resume_plan_resolution(self, reply: str) -> fastworkflow.CommandOutput:
        """The reply to the plan path's question. Before the review (``after`` ``resolve_next``) it is read in code
        (``resolution.take_reply``) and the resolution carries on: the next question, or the review, or the run. A
        question a running part put (a refused typed call, ``after`` ``resume_part``) resumes that part
        (``_resume_parameter_loop_unlistened``)."""
        from fastworkflow import parameter_clarification, turn_progress
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        with turn_progress.bound(self):
            if self._waiting.payload.get("after") == parameter_clarification.AFTER_RESUME_PART:
                return self._resume_parameter_loop_unlistened(reply)
            message = self._suspended_message()
            self._note_agent_resume()
            self.complete_ask_user_entry(reply)
            self._turn_subtask_plan, question = resolution.take_reply(
                self._plan_resolution, reply, self._turn_subtask_plan, self._resolution_turn())
            if question is not None:
                return self._settle_agent_result(
                    message, dspy.Prediction(suspended=True, clarification=question, exhausted=False))
            return self._settle_agent_result(message, self._after_resolution())

    def _after_resolution(self):
        """Every part is resolved: the dropped ones are left out, then ONE review of the whole turn, or the run.

        No review when the client pinned the plan (its parameter question was the review), when the turn is one part
        whose command the user chose by answering which one, when nothing would run (no agent model, every part ad
        hoc) and when nobody can answer. Returns the suspended review or the turn's result."""
        from fastworkflow import review_proposals, turn_progress
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        record = self._plan_resolution
        subtasks = [item.goal for item in self._turn_subtask_plan.items]    # the parts' texts, as drafted
        self._note_dropped_parts(record)
        kept_parts = [part for part in record.parts if part["state"] != resolution.PART_DROPPED]
        draft, subtasks, self._turn_plan_ambiguity = resolution.finish(record, self._turn_subtask_plan, subtasks)
        # What each kept part's settling knew, by its text, with the command the draft shows for it (the review may
        # accept or switch it): `_record_planned_parts` writes the rows where the plan is run.
        self._turn_part_facts = {
            part["text"]: {"how": (self._learning_how(record, part)), "evidence": part.get("evidence") or {},
                           "drafted": item.means.name or None}
            for part, item in zip(kept_parts, draft.items)}
        chosen = resolution.single_part_chosen(record)
        self._plan_resolution = None
        self._turn_subtask_plan = draft
        if not draft.items:
            self._turn_subtask_review = "declined"
            self._turn_subtask_plan = None
            # No agent model: each cancelled part is a stop the client reads (``fw_not_run``), and the answer says why.
            answer = "\n\n".join(no_agent_mode.explanation(entry) for entry in self._turn_not_run)
            return dspy.Prediction(final_answer=answer or EVERY_PART_CANCELLED, exhausted=False)
        refined = subtasks[0] if len(subtasks) == 1 and len(record.parts) > 1 else self._turn_refined_message

        def run():
            return self._run_turn_subtasks(subtasks, refined or subtasks[0], self._turn_user_message or subtasks[0])

        if record.pinned:
            self._turn_subtask_review = "not_reviewed"
            return run()
        if chosen:
            self._turn_subtask_review = "fast_path"
            return run()
        turn_progress.emit(turn_progress.STAGE_REVIEW)
        nothing_runs = self.no_agent_mode and all(
            item.means.kind == review_proposals.MEANS_FROM_SCRATCH for item in draft.items)
        if nothing_runs or not waiting_module.can_review(self):
            # Owner decision 2: no one can answer (or, in no-agent mode, nothing would run, so there is nothing to
            # approve), so the draft is recorded as not reviewed and the turn carries on as drafted.
            self._turn_subtask_review = "not_reviewed"
            logger.info("sub-task review not opened (no one to answer): %d sub-task(s) run as drafted, not reviewed",
                        len(draft.items))
            return run()
        return self._open_review(draft)

    def _note_dropped_parts(self, record) -> None:
        """No agent model: each part cancelled while the turn was resolved is not run, and a client that drives the work
        reads it among the turn's stops (``fw_not_run``)."""
        from fastworkflow import review_proposals
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        if not self.no_agent_mode:
            return
        for part in resolution.dropped(record, self._turn_subtask_plan):
            self._turn_not_run.append(no_agent_mode.stop_entry(
                command=part["command"], phase=None, open_step=None, done=[], facts={}, remaining=[],
                suggested_plan="", means=part["means"], sub_task=None, reason=part["reason"],
                request=part["text"],
                # An ad hoc part keeps the plans a client may pin it to (``use_plan``), the closest first.
                available_plans=(no_agent_mode.available_plans(self._app_workflow, part["planning_scores"])
                                 if part["means"] == review_proposals.MEANS_FROM_SCRATCH else None)))

    def _open_review(self, draft):
        """Open the review of *draft* (the whole turn): the suspended review."""
        from fastworkflow import review_proposals

        if self.no_agent_mode:
            self._turn_plan_previews = self._ad_hoc_plan_previews(draft)
        self._give_review_reader()
        with review_proposals.draft_scope(draft):
            asked = self.run_framework_planning_command(
                "review_subtasks", {"subtasks": review_proposals.DRAFT_ITEMS}, "review the sub-tasks",
                within_turn=True)
        if getattr(asked, "suspended", None) is not True:
            raise AssertionError("review_subtasks must suspend the turn on its review")
        return asked

    def _gate_review_out(self) -> bool:
        """Whether the sub-task gate's review is the turn's waiting record (not a stand-alone framework call's)."""
        held = self._waiting
        return held is not None and held.kind == waiting_module.KIND_REVIEW and bool(held.payload.get("gate"))

    def _subtask_review_mark(self) -> str:
        """The turn span's ``subtask_review``: ``open`` while the sub-task gate's review is out (the review record says
        so), else the outcome the turn recorded."""
        return "open" if self._gate_review_out() else self._turn_subtask_review

    def _drop_store_record(self) -> None:
        """The approval store ended the record the turn's question points at (accepted, declined, expired, cancelled): the
        question, and the slot question nested in a review, go."""
        plan_cme = getattr(self, "_plan_cme", None)
        if plan_cme is not None:
            self._clear_cme_continuation(plan_cme)
        self._waiting = None

    def _ad_hoc_plan_previews(self, draft) -> dict[str, str]:
        """``{goal: the planner's text plan}`` for each ad hoc item of *draft* (no-agent mode)."""
        return {item.goal: self._plan_preview(item.goal) for item in draft.items if item.means.kind == "from_scratch"}

    def _plan_preview(self, goal: str) -> str:
        """The planner's text plan for an ad hoc *goal* (no-agent mode): what the client would carry out, since the item
        is not run here. One planner call (``LLM_PLANNER``, never the agent); "" when the planner gave no plan or
        failed."""
        from fastworkflow.workflow_agent import build_query_with_next_steps

        saved = (getattr(self, "_initial_plan_text", ""), getattr(self, "_phased_planning_command_call", None))
        try:
            build_query_with_next_steps(goal, self, planner_lm=getattr(self, "_current_planner_lm", None),
                                        planner_user_query=goal)
            return str(getattr(self, "_initial_plan_text", "") or "").strip()
        except Exception as error:  # noqa: BLE001 - a missing preview never stops the review
            logger.warning("plan preview of %r not made: %s: %s", goal, type(error).__name__, error)
            return ""
        finally:
            self._initial_plan_text, self._phased_planning_command_call = saved

    def _give_review_reader(self) -> None:
        """The free-text reader of this turn's sub-task review (fix-3rsq.plan-review.2.7): one model call on the
        planner's role, with the planner's LM when the turn has one. Set on the session's review store."""
        from fastworkflow import planning_commands, review_engine, review_proposals

        store = review_engine.store_for(self._app_workflow)
        if store is None:
            return
        names = tuple(sorted(planning_commands.resolve_field_sources(self._app_workflow.folderpath).planning_commands))
        # The planner's LM is read when a reply comes, not when the review opens: the turn's planner LM is the one in place then.
        store.edit_reader = review_proposals.model_edit_reader(
            planner_lm=lambda: getattr(self, "_current_planner_lm", None), planning_names=names)
        store.plan_cme = getattr(self, "_plan_cme", None)
        store.plan_workflow = getattr(self, "_plan_workflow", None)

    def _accept_subtask_review(self, agent_result):
        """The gate's review was accepted: its frame finishes with the accepted proposal, which is stored for the
        hand-off (fix-3rsq.plan-review.2.9 reads ``_turn_subtask_plan``), and the turn runs the accepted goals as today.
        Its inputs and means are not used until 2.9."""
        from fastworkflow import review_proposals

        self._finish_framework_call(agent_result)
        accepted = review_proposals.accepted_subtask_proposal(self._app_workflow)
        if accepted is None:
            raise RuntimeError("the accepted sub-task review is not on the session")
        self._turn_subtask_plan = accepted
        self._turn_subtask_review = "reviewed"
        subtasks = [item.goal for item in accepted.items]
        return self._run_turn_subtasks(subtasks, subtasks[0] if len(subtasks) == 1 else self._turn_refined_message,
                                       self._turn_user_message)

    def _end_subtask_review_unrun(self, state: str):
        """The gate's review was declined or expired: its frame is dropped, nothing runs, and the turn says so."""
        from fastworkflow import review_engine

        self._pop_frame(self._top_frame())
        declined = state == review_engine.DECLINED
        self._turn_subtask_review = "declined" if declined else "expired"
        self._turn_subtask_plan = None
        text = SUBTASK_REVIEW_DECLINED if declined else SUBTASK_REVIEW_EXPIRED
        return self._settle_agent_result(self._turn_user_message, dspy.Prediction(final_answer=text, exhausted=False))

    def _call_agent_for_query(self, query: str, planner_user_query: str, *,
                              first_step: int | None = None,
                              template_text: str | None = None,
                              evidence_floor: int | None = None,
                              bound_command: str | None = None,
                              planning_name: str | None = None):
        """Plan one query with the planner, then run the ReAct agent on it.

        ``template_text`` picks the planning command template; it defaults to the query itself.
        ``evidence_floor`` is the first step index whose archived outputs count as
        this run's for the finish check (a hand-off continuation counts the runs
        of its sub-task before it); it defaults to ``first_step``.
        ``planning_name`` names the planning command of a part whose command is already known (a part that
        resumes after its parameter question): the matcher is not asked again.
        """
        from fastworkflow.workflow_agent import build_query_with_next_steps, sub_task_commands_text

        # One planning command match gives both the template and its parsed form; a hand-off continuation
        # (query differs from its original text) plans from the template only, since the planning command
        # call would plan the whole job again.
        # Which plan, if any: the part's one decision (``resolution.part_intent``), memoised for the turn, so the gate's
        # draft and this run read the same answer and the remote model is not asked again. Its planning candidates'
        # scores feed the listings' gathering (run_listing) and order an ad hoc part's plans.
        planning_distribution: dict[str, float] = {}
        plan_template, planning_command = "", None
        known = self._turn_use_plan or planning_name
        if self._app_workflow is not None and known:
            # The client's choice (``use_plan``), or the command a part was already asked about, is the match: the
            # matcher is not called. A bound call (a refusal's fall-back) names its own command below.
            from fastworkflow.workflow_agent import _template_text, load_planning_command_map

            planning_command = (load_planning_command_map(self._app_workflow.folderpath).get(known)
                                if bound_command is None else None)
            if planning_command is not None:
                plan_template = _template_text(planning_command.template)
                planning_distribution = {known: 1.0}
        elif self._app_workflow is not None:
            from fastworkflow._workflows.plan_metadata_extraction import resolution
            from fastworkflow.workflow_agent import _template_text, load_planning_command_map

            intent = resolution.part_intent(self._app_workflow, template_text or query)
            if intent.outcome == resolution.MATCH and intent.kind == resolution.KIND_PLANNING:
                planning_command = load_planning_command_map(self._app_workflow.folderpath).get(intent.command)
                if planning_command is not None:
                    plan_template = _template_text(planning_command.template)
                    planning_distribution = {intent.command: 1.0}
            elif intent.outcome != resolution.MATCH:
                planning_distribution = intent.planning_scores()
        if template_text is not None and template_text != query:
            planning_command = None
        if bound_command is not None:
            # The bound means of a plain goal: the plan names its command, so the finish check holds the goal to it.
            plan_template = f"This sub-task is done with the command {bound_command}."
            planning_command = None
        # Why a refused call was not asked about is read below; it is this call's, never an earlier part's.
        self._refusal_stop_reason = ""
        # Initial turn planning uses every prior turn summary, not the previous
        # turn's full agent trajectory (mid-turn replans still use trajectory).
        command_info_and_refined_message_with_todolist = build_query_with_next_steps(
            query,
            self,
            with_agent_inputs_and_trajectory=False,
            planner_lm=getattr(self, "_current_planner_lm", None),
            planner_user_query=planner_user_query,
            plan_template=plan_template or None,
            planning_command=planning_command,
            planning_probabilities=planning_distribution or None,
        )
        refusal, self._planning_command_refusal = self._planning_command_refusal, None
        if refusal is not None and self._part_question_out:
            # A refused planning command call the user is asked about: the part suspends on the question as an
            # ask_user does (no plan, no run); the reply resumes this part (_resume_parameter_loop_unlistened).
            return self._suspend_on_parameter_loop()
        phased, self._phased_planning_command_call = getattr(self, "_phased_planning_command_call", None), None
        if phased is not None:
            # A planning command call: its phases run as sub-plans, with no hand-off.
            return self._start_planning_command_phases(query, *phased, first_step=first_step or 0)
        if self.no_agent_mode:
            # No agent model: nothing here may run the free agent. The planner's text plan (a planner model call,
            # made above) is the suggestion; none when the planner made none.
            if planning_command is not None and bound_command is None and self._refusal_stop_reason:
                # A planning command matched this part but its call was refused and cannot be asked about: it is that
                # command that was not run, for the arguments that are missing.
                return self._not_run_part(
                    command=planning_command.name, means="planning",
                    reason=f"it cannot run as planned ({self._refusal_stop_reason}), and the agent model that would "
                           "take over is not set (LLM_AGENT)")
            # An ad hoc part was asked about before the review (``resolution.step``) when somebody could answer; here it
            # ends as not run, listing the plans available here.
            origin = self._origins[self._subtask_index] if getattr(self, "_origins", None) else 0
            item = self._bound_item(origin)
            means = item.means.kind if item is not None else "from_scratch"
            return self._not_run_part(
                command=bound_command, means=means,
                # An ad hoc part lists the plans available here, the matcher's closest ones first.
                plans_for=planning_distribution if bound_command is None else None,
                reason=("no planning command or command fits it, so it needs free planning, which is not run "
                        "without an agent model (LLM_AGENT)" if bound_command is None else
                        f"the agent model that would carry out {bound_command} from its inputs is not set (LLM_AGENT)"),
                suggested_plan=str(getattr(self, "_initial_plan_text", "") or "").strip())
        if bound_command is not None:
            from fastworkflow import run_listing

            available_commands = run_listing.bound_command_listing(self._app_workflow, bound_command)
        else:
            available_commands = sub_task_commands_text(self, query, planning_probabilities=planning_distribution or None)
        step_kwargs = {} if first_step is None else {"first_step": first_step}
        self._attach_finish_check(
            evidence_floor if evidence_floor is not None else (first_step or 0),
            final=self._last_subtask_running())

        return self._call_agent_with_retry(
            lambda: self._workflow_tool_agent(
                user_query=command_info_and_refined_message_with_todolist,
                available_commands=available_commands,
                **step_kwargs,
            ),
            trace_input=command_info_and_refined_message_with_todolist,
        )

    def _last_subtask_running(self) -> bool:
        """Whether the sub-task running now is the turn's last (always, for an unsplit turn)."""
        subtasks = getattr(self, "_subtasks", None) or []
        return getattr(self, "_subtask_index", 0) >= len(subtasks) - 1

    def _attach_finish_check(self, floor: int, *, values: bool = False,
                             item_binding: Optional[tuple[Any, list, list]] = None,
                             final: bool = True) -> None:
        """Give the agent this run's finish check, or none when the run has no plan.

        The plan is the one ``build_query_with_next_steps`` just made for this
        run. Set before every run, so a check never outlives the run it was made for.
        *values*: a step's concrete argument values must match (a planning command phase's run).
        *item_binding* ``(item, other items, the phase's planned command names)``: the
        run is one item of a fan-out phase,
        bound to its record; the check also flags a run on another item's record and
        an item never opened (``planning_phases.item_record_issues``).
        """
        agent = self._workflow_tool_agent
        if agent is None:
            return
        agent.finish_reviewer = None
        # *final*: this run ends the turn (the last sub-task, the last phase). The
        # policy's finish check reads the answer against the whole request, so it
        # only runs there (fastworkflow.policy_runtime).
        agent.policy_final_run = final
        from fastworkflow.observation_offloading import finish_check

        plan_text = str(getattr(self, "_initial_plan_text", "") or "")
        if not plan_text.strip() or self._app_workflow is None:
            return
        folderpath = self._app_workflow.folderpath

        def review(trajectory, iters_left: int):
            from fastworkflow.workflow_agent import core_command_names

            # The archive keeps the command text as sent; the action log says
            # which command each text resolved to (an alias, a misspelling).
            note = finish_check.review_finish(
                plan_text=plan_text, floor=floor, iters_left=iters_left,
                commands=self._workflow_command_names(),
                core_commands=core_command_names(),
                resolved=finish_check.resolved_command_names(
                    getattr(self, "_action_log", None) or ()),
                values=values)
            if item_binding is None or iters_left < finish_check.MIN_ITERS_LEFT:
                return note
            from fastworkflow import planning_phases

            try:
                runs = self._archived_command_runs(from_step=int(floor or 0))
                item, others, planned = item_binding
                binding = planning_phases.item_record_note(item, planning_phases.item_record_issues(
                    runs, item, others, self._current_record()[0], planned))
            except Exception as error:  # noqa: BLE001 - the check never fails a turn
                logger.warning("item record check skipped: %s: %s", type(error).__name__, error)
                return note
            return "\n".join(part for part in (note, binding) if part) or None

        agent.finish_reviewer = review

    # ------------------------------------------------------------------
    # Phased planning command calls (fastworkflow.planning_phases)
    # ------------------------------------------------------------------

    def _start_planning_command_phases(self, query: str, command, call, *, first_step: int,
                                       reextracted: bool = False):
        """Run a phased planning command call's phases, in order; the sub-task's result.

        ``call`` is the planning command call (``planning_commands.PlanningCommandCallPlan``) whose ``phases``
        run. ``reextracted``: this call is the restart after a failed first phase (``_restart_after_empty_first_phase``),
        which a call restarts at most once. See ``fastworkflow.planning_phases`` for what the runner does and decides.
        Extra command steps after the planning command's own that would land in an answer-only last
        phase run in a phase of their own just before it (``planning_phases.extra_steps_phase``).
        The extra steps before the planning command's own run first, in a phase of their own
        (``planning_phases.steps_before_phase``).
        """
        from fastworkflow import planning_phases
        from fastworkflow.observation_offloading.facts import fact_sources

        names = self._phase_command_names()
        names["commands"] = names["commands"] + [
            source.command for phase in call.phases if phase.output is not None
            for source in fact_sources(phase.output)]
        extra, steps_after = planning_phases.extra_steps_phase(call.phases, call.steps_after, **names)
        phases = planning_phases.with_phase_before(call.phases, extra)
        before = planning_phases.steps_before_phase(phases, call.steps_before)
        phases = planning_phases.with_phase_before(phases, before, 0)
        self._phase_live = (command, phases)
        self._phase_call = (command, call)
        self._phase_run = {
            "skill": command.name,
            # Set when a phase's lookup found nothing (the run's MissingFactError), read by the restart rule.
            "missing_fact": False,
            # True for the call planned again after a failed first phase: it is not planned again a second time.
            "reextracted": bool(reextracted),
            "args": json.loads(json.dumps(call.args, default=str)),
            "query": query,
            "phases": [phase.name for phase in phases],
            "steps_before": list(call.steps_before),
            "steps_after": steps_after,
            # The phase the extra command steps run in (None when there is none), its steps
            # (to rebuild it after a restore) and its answer (added to the job's answer).
            "extra_phase": extra.name if extra is not None else None,
            "extra_steps": extra.steps_fn() if extra is not None else [],
            "extra_answer": "",
            # The phase the extra steps before the planning command's own run in (None when there
            # are none; its steps are steps_before) and its answer (added to the job's).
            "before_phase": before.name if before is not None else None,
            "before_answer": "",
            "index": 0,
            "outputs": {},
            "answers": [],
            "phase_answers": [],
            # A fan-out phase in progress: its items (capped), how many there were, and
            # the item now running. None outside a fan-out phase.
            "items": None,
            "items_total": 0,
            "item_index": 0,
            # A fan-out phase's `until` condition was met: its remaining items do not run.
            "until_met": False,
            "notes": [],
            "call_first_step": first_step,
            "first_step": first_step,
            "plan": "",
            "uses_result": bool(getattr(command, "uses_result", False)),
            # A flat call: one phase named after the planning command (call_phases).
            "flat": len(call.phases) == 1 and call.phases[0].name == command.name,
        }
        return self._run_planning_command_phases(first_step)

    def _live_planning_command_phases(self):
        """``(command, phases)`` of the call in progress; rebuilt from the planning command after a restore."""
        if self._phase_live is not None:
            return self._phase_live
        from fastworkflow.planning_commands import call_phases
        from fastworkflow.planning_phases import PhaseError
        from fastworkflow.workflow_agent import load_planning_command_map

        state = self._phase_run
        folderpath = self._app_workflow.folderpath
        command = load_planning_command_map(folderpath).get(state["skill"])
        if command is None:
            # A planning command's call (run_planning_command) is not a planning command file.
            command = planning_commands.planning_command(folderpath, state["skill"])
        if command is None:
            raise PhaseError(f"planning command {state['skill']!r} is no longer defined")
        # A flat call (no plan(), or no phases for these arguments) is one phase.
        phases = call_phases(command, state["args"])
        if state.get("extra_phase"):
            from fastworkflow.planning_phases import Phase, with_phase_before

            moved = list(state.get("extra_steps") or [])
            phases = with_phase_before(phases, Phase(state["extra_phase"], lambda: list(moved)))
        if state.get("before_phase"):
            from fastworkflow.planning_phases import Phase, with_phase_before

            first = list(state.get("steps_before") or [])
            phases = with_phase_before(phases, Phase(state["before_phase"], lambda: list(first)), 0)
        if [phase.name for phase in phases] != state["phases"]:
            raise PhaseError(f"planning command {state['skill']!r} now plans other phases")
        self._phase_live = (command, phases)
        return self._phase_live

    def _phase_outputs(self, phases, index: int) -> dict[str, Any]:
        """The typed outputs of the phases before *index* that filled one (a fan-out
        phase's: the list of its per-item outputs)."""
        state = self._phase_run
        outputs: dict[str, Any] = {}
        for p in phases[:index]:
            if p.output is None or p.name not in state["outputs"]:
                continue
            value = state["outputs"][p.name]
            outputs[p.name] = ([stored_value(p.output, v) for v in value] if p.fans_out
                               else stored_value(p.output, value))
        return outputs

    def _run_planning_command_phases(self, first_step: Optional[int] = None, *,
                          resume_observation: Optional[str] = None,
                          resume_completes_step: bool = False):
        """Run the phases from the current one; a suspension, or the sub-task's result.

        With ``resume_observation`` the current phase's executor run, suspended on
        ask_user (or on a nested command, ``resume_completes_step``), is resumed first. The
        result never reports ``exhausted``: a phased
        planning command is not handed off. A fan-out phase runs one sub-plan per item, until its
        items are done or its ``until`` condition is met. A phase with a callable
        ``when`` is decided when it is reached (``planning_phases.runs_now``); a skipped one
        runs nothing and keeps the answers of the phase before it.
        """
        from fastworkflow import planning_phases

        state = self._phase_run
        try:
            command, phases = self._live_planning_command_phases()
        except Exception as error:  # noqa: BLE001 - the call stops, saying why
            return self._planning_command_phases_failed(state["phases"][state["index"]], f"{error}")
        phase_names = frozenset(state["phases"])
        while state["index"] < len(phases):
            if resume_observation is None and self._nested_budget_spent():
                # A nested call's later phases do not start once the turn-wide budget is
                # spent (run_planning_command); the call ends here, as exhausted. A
                # resumed phase is not stopped here: it is already running.
                state["exhausted"] = True
                break
            phase = phases[state["index"]]
            starting = resume_observation is None and (
                not phase.fans_out or state.get("items") is None)
            if starting:
                # What an optional phase that fails gives back (_phase_failure).
                state["phase_start"] = {"answers": len(state["answers"]),
                                        "phase_answers": list(state.get("phase_answers") or []),
                                        "notes": len(state.get("notes") or [])}
            if starting and phase.has_condition:
                try:
                    runs, why = planning_phases.runs_now(
                        phase, self._phase_outputs(phases, state["index"]),
                        phase_names=phase_names), None
                except planning_phases.PhaseError as error:
                    runs, why = False, str(error)
                if not runs:
                    # Skipped: counts as not run (no output; a later parameter named after it
                    # binds None, ``planning_phases.bound_arguments`` with ``phase_names``).
                    # The phase's span says so; no executor runs.
                    outcome, value = self._run_planning_command_phase(command, phases, first_step,
                                                           skip=True, skip_error=why)
                    if outcome == "failed":
                        failed = self._phase_failure(phase, value)
                        if failed is not None:
                            return failed
                        continue
                    state["index"] += 1
                    continue
            if starting:
                state["phase_answers"] = []
            if phase.fans_out and state.get("items") is None:
                try:
                    items, total = planning_phases.fan_out_items(
                        phase, self._phase_outputs(phases, state["index"]))
                except planning_phases.PhaseError as error:
                    failed = self._phase_failure(phase, str(error))
                    if failed is not None:
                        return failed
                    continue
                state.update(items=json.loads(json.dumps(items, default=str)),
                             items_total=total, item_index=0, until_met=False)
                state["outputs"][phase.name] = []
                if total > len(items) and phase.until is None:
                    note = planning_phases.cap_note(phase.name, len(items), total, phase.noun)
                    # Two fan-outs over the same items with one noun say it once.
                    if note not in state["notes"]:
                        state["notes"].append(note)
            if not phase.fans_out or (state["item_index"] < len(state["items"])
                                      and not state.get("until_met")):
                outcome, value = self._run_planning_command_phase(
                    command, phases,
                    state["first_step"] if resume_observation is not None else first_step,
                    resume_observation=resume_observation,
                    resume_completes_step=resume_completes_step)
                resume_observation = None
                resume_completes_step = False
                if outcome == "suspended":
                    return value
                if outcome == "stopped":
                    return self._planning_command_phases_stopped(value)
                if outcome == "failed":
                    first_step = self._next_step
                    # Disabled after benchmark pass 4 (fix-zezw never restarted a call in 840 runs):
                    # restarted = self._restart_after_empty_first_phase(phase, value)
                    # if restarted is not None:
                    #     return restarted
                    failed = self._phase_failure(phase, value)
                    if failed is not None:
                        return failed
                    continue
                first_step = self._next_step
                if phase.fans_out:
                    state["item_index"] += 1
                    continue
            if phase.fans_out and phase.until is not None:
                # Said at the end: stopped on its condition, or capped.
                note = planning_phases.fan_out_note(phase.name, state["item_index"],
                                                 state["items_total"], bool(state.get("until_met")),
                                                 phase.noun)
                if note and note not in state["notes"]:
                    state["notes"].append(note)
            state.update(items=None, items_total=0, item_index=0, until_met=False)
            state["index"] += 1
        uses_result = state.get("uses_result")
        command_output = None
        response = None
        result_failed = False
        if uses_result and command.plan_cls is not None:
            from fastworkflow.planning_commands import _phase_names_for_command, input_instance

            typed = planning_phases.typed_phase_outputs(state["outputs"], phases)
            input_obj = input_instance(command, state["args"])
            # __result__ may name any phase of the command: one this call left out is None (3.16).
            response = planning_phases.run_call_result(
                command.plan_cls, input_obj, typed,
                phase_names=phase_names | frozenset(_phase_names_for_command(command)),
                output_model=command.output_model,
                command_name=command.name,
            )
            result_failed = response is not None and response.success is False
            if result_failed:
                # A failing __result__ fails the call; no model writes the answer in its place (fix-qjne.12).
                self._record_result_failure(command, phases, response)
        if response is not None:
            answer = response.response or ""
            notes = list(state.get("notes") or [])
            if notes:
                answer = "\n\n".join(part for part in (answer, " ".join(notes)) if part)
            command_output = fastworkflow.CommandOutput(
                command_response=response,
                command_name=command.name,
                command_parameters=input_obj,
            )
            if self._app_workflow:
                command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]
            if len(self._frame_stack()) == 1:
                # The turn's own call appends its output here. A nested call's output is
                # returned instead: run_planning_command hands it to the step that ran it,
                # which appends it (workflow_agent._execute_workflow_query).
                self.append_turn_output(command_output)
        elif self.no_agent_mode and self._answer_needs_model(state):
            # No __result__ text and no code-written answer: a model would write it, and none is set.
            return self._planning_command_phases_stopped(self._no_agent_answer_stop(phases[-1], state["plan"], {
                **self._phase_command_names()}, finished=True))
        else:
            # No __result__, or it returned None (no code-written answer, 3.16): the answer is the
            # one the phases give, as with no __result__. A deferred answer is written now.
            settled: dict = {}
            answers = [self._settled_phase_answer(a, phases, settled) for a in state["answers"]]
            phase_answers = [self._settled_phase_answer(a, phases, settled)
                             for a in state.get("phase_answers") or []]
            # The last phase writes the answer, from the facts its steps carry (a fan-out
            # last phase: its items' answers, in item order).
            last = [a for a in phase_answers if a]
            answer = "\n\n".join(last) if last else "\n\n".join(a for a in answers if a)
            extra = self._settled_phase_answer(state.get("extra_answer"), phases, settled)
            if extra and last:
                # The extra command steps' own report (their phase is not the last).
                answer = f"{answer}\n\n{extra}"
            before = self._settled_phase_answer(state.get("before_answer"), phases, settled)
            if before and last:
                # The report of the extra steps before the planning command's own (their phase is first).
                answer = f"{before}\n\n{answer}"
            notes = list(state.get("notes") or [])
            if notes:
                answer = "\n\n".join(part for part in (answer, " ".join(notes)) if part)
        self._phase_run = self._phase_live = None
        # exhausted stays False: a phased planning command is not handed off. phase_exhausted says a
        # phase ran out of steps, and command_output is the call's own output when its
        # __result__ built one (uses_result), which a nested call returns to its step.
        if self.no_agent_mode and len(self._frame_stack()) == 1 and command is not None and not result_failed:
            answer = no_agent_mode.ran_note(answer, command.name)   # the part ran bound: say which command
        return dspy.Prediction(final_answer=answer, exhausted=False,
                               phase_exhausted=bool(state.get("exhausted")),
                               command_output=command_output)

    def _record_result_failure(self, command, phases, response) -> None:
        """A span for a ``__result__`` that failed (fix-qjne.12): ``fw.skill.answer`` with status error and
        ``error_type`` (the exception class; none for a ``fastworkflow.Refused``, which is a decision), the
        span that records the answer of a call, though no model call is made here. A refusal ends it with status ok."""
        span = tracing.start_span(self, tracing.SPAN_PLANNING_COMMAND_ANSWER, attributes={
            "skill": command.name, "phase": phases[-1].name,
            "phase_index": len(phases) - 1, "phase_count": len(phases)})
        error_type = response.artifacts.get("error_type")
        if error_type:
            tracing.end_span(self, span, status=tracing.STATUS_ERROR, attributes={"error_type": error_type})
        else:
            tracing.end_span(self, span)

    def _defer_phase_answer(self, index: int, query: str, plan: str) -> dict:
        """Hold an answer-only phase's model call until the answer is settled (3.16): it runs
        only when ``__result__`` returned None. Returns the marker the phase records."""
        deferred = self._phase_run.setdefault("deferred", {})
        key = str(len(deferred))
        deferred[key] = {"index": index, "query": query, "plan": plan}
        return {"deferred": key}

    def _settled_phase_answer(self, value, phases, settled: dict) -> str:
        """A recorded phase answer as text. A deferred one (a ``{"deferred": key}`` marker) is
        written now, once, by the model from the facts its phase had (``answer_only_answer``)."""
        if not (isinstance(value, dict) and "deferred" in value):
            return value or ""
        key = value["deferred"]
        if key not in settled:
            import dspy
            from fastworkflow import planning_phases

            deferred = self._phase_run["deferred"][key]
            phase = phases[deferred["index"]]
            # The facts of the phase itself and of the phases before it (a judgement phase's own
            # output included), given to the model with the request.
            outputs = self._phase_outputs(phases, deferred["index"] + 1)
            facts = "\n\n".join(planning_phases.found_facts(outputs))
            query = f"{deferred['query']}\n\n{facts}" if facts else deferred["query"]
            span = tracing.start_span(self, tracing.SPAN_PLANNING_COMMAND_ANSWER, attributes={
                "skill": self._phase_run["skill"], "phase": phase.name,
                "phase_index": deferred["index"], "phase_count": len(phases)})
            try:
                result = self._call_agent_with_retry(
                    lambda: dspy.Prediction(
                        final_answer=planning_phases.answer_only_answer(
                            query, deferred["plan"], outputs),
                        suspended=False, exhausted=False),
                    trace_input=query,
                )
            except BaseException as exc:
                tracing.end_span(self, span, status=tracing.status_for_dispatch_exception(exc),
                                 attributes={"error_type": type(exc).__name__})
                raise
            settled[key] = planning_phases.clean_answer(getattr(result, "final_answer", "") or "")
            tracing.end_span(self, span)
        return settled[key]

    def _record_phase_answer(self, phase, answer) -> None:
        """Record a phase's answer (text, or a deferred marker) for the call's answer."""
        state = self._phase_run
        if phase.name == state.get("extra_phase"):
            state["extra_answer"] = answer
        if phase.name == state.get("before_phase"):
            state["before_answer"] = answer
        state["answers"].append(answer)
        state.setdefault("phase_answers", []).append(answer)

    def _restart_after_empty_first_phase(self, phase, observation: str):
        """The call planned again once, when its first phase's lookup found nothing (fix-variance).

        Applies when the first phase (its first item, if it fans out) fails on a required fact no run holds
        (``MissingFactError``), the phase is not optional, and the call was not restarted before. No earlier
        phase exists to have run a command. The failed phase's own steps may have run before its facts failed,
        so the call is not restarted when those steps name a command that writes (``command_writes``): a restart
        would write twice. The arguments are extracted once more with the phase's observation after the request
        (``planning_commands.reextracted_arguments``), and the call runs its phases again from the start. None
        when it does not apply, or the arguments come out the same or are refused: the failure then stops the
        job as before.
        """
        state = self._phase_run
        if (state is None or state.get("index") != 0 or state.get("item_index") or state.get("reextracted")
                or not state.get("missing_fact") or getattr(phase, "optional", False)
                or self._phase_call is None):
            return None
        from fastworkflow import command_writes, planning_commands

        # Read before _phase_run is cleared: the steps of the phase that failed, as its runs were planned.
        planned = command_writes.phase_planned_commands(self)
        if planned is None:
            logger.info("planning command %s: phase %s found nothing, and its planned commands are unknown; "
                        "the call is not planned again", state["skill"], phase.name)
            return None
        writes = {command_writes._plain(name) for name in
                  command_writes.write_commands(self._app_workflow.folderpath)}
        writing = sorted(planned & writes)
        if writing:
            logger.info("planning command %s: phase %s found nothing, but its steps name a command that writes "
                        "(%s); the call is not planned again, so nothing is written twice",
                        state["skill"], phase.name, ", ".join(writing))
            return None

        command, call = self._phase_call
        query = state["query"]
        try:
            args = planning_commands.reextracted_arguments(
                command, query, observation, call.args, workflow=self._app_workflow, verdicts=call.verdicts)
        except Exception as error:  # noqa: BLE001 - the job stops as it would have
            logger.warning("planning command %s: arguments not extracted again after phase %s found nothing: %s: %s",
                           state["skill"], phase.name, type(error).__name__, error)
            return None
        if args is None:
            logger.info("planning command %s: phase %s found nothing; the arguments extracted again are the same "
                        "or refused, so the call is not planned again", state["skill"], phase.name)
            return None
        logger.info("planning command %s: phase %s found nothing; planning the call again with arguments %s",
                    state["skill"], phase.name, args)
        new_call = planning_commands.replanned_call(command, call, args, notes=list(call.notes or []))
        self._phase_run = self._phase_live = None
        return self._start_planning_command_phases(query, command, new_call, first_step=self._next_step or 0,
                                                   reextracted=True)

    def _phase_failure(self, phase, error: str):
        """Phase *phase* failed: the sub-task's result (``_planning_command_phases_failed``), or None
        when the phase is ``optional``. An optional phase then counts as not run: its
        output, the answers its runs wrote and the notes it added are dropped, and the
        runner goes on with the next phase."""
        if not getattr(phase, "optional", False):
            return self._planning_command_phases_failed(phase.name, error)
        state = self._phase_run
        start = state.get("phase_start") or {}
        del state["answers"][start.get("answers", len(state["answers"])):]
        state["phase_answers"] = list(start.get("phase_answers") or [])
        del state["notes"][start.get("notes", len(state["notes"])):]
        state["outputs"].pop(phase.name, None)
        state.update(items=None, items_total=0, item_index=0, until_met=False)
        state["index"] += 1
        logger.info("planning command %s: optional phase %s failed and is left out: %s",
                    state["skill"], phase.name, error)
        return None

    def _planning_command_phases_failed(self, name: str, error: str):
        """The sub-task's result when phase ``name`` failed: the facts found, and what failed.

        Only facts filled in code are shown (``planning_phases.found_facts``), never text a
        phase's run wrote: the failed phase's own answer may report what it never read."""
        from fastworkflow.planning_phases import failure_text, found_facts

        state = self._phase_run
        names = state["phases"]
        not_run = names[names.index(name) + 1:] if name in names else []
        logger.info("planning command %s: phase %s failed: %s", state["skill"], name, error)
        self._phase_run = self._phase_live = None
        return dspy.Prediction(
            final_answer=failure_text(found_facts(plain_facts(state.get("outputs") or {})), name, error,
                                      not_run, notes=state.get("notes") or ()),
            exhausted=False, phase_failed=name)

    # ------------------------------------------------------------------
    # No-agent mode: where a planning command stops (fastworkflow.no_agent_mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_lines(plan_text: str, names: dict) -> list:
        """The steps of *plan_text* as the finish check reads them (``PlanStep``: number, text, kind)."""
        from fastworkflow.observation_offloading import finish_check

        return finish_check.parse_plan(plan_text, names.get("commands") or [], names.get("core_commands") or [])

    def _phase_step_lines(self, phase, state: dict[str, Any], plan_text: str, names: dict, *,
                          ran_only: bool = False, below: Optional[int] = None) -> list[str]:
        """The command steps of *plan_text* as plan lines: ``"<n>. <text>"``, prefixed ``"<phase>: "`` when the call
        has several phases (step numbers restart in each). *below*: only the steps numbered under it."""
        prefix = f"{phase.name}: " if len(state["phases"]) > 1 else ""
        return [f"{prefix}{step.number}. {step.text}" for step in self._plan_lines(plan_text, names)
                if (step.kind == "command" or not ran_only) and (below is None or step.number < below)]

    def _step_results(self, phase, state: dict[str, Any], plan_text: str, names: dict, code, *,
                      below: Optional[int] = None) -> dict[str, str]:
        """The response text of each command step code ran in this phase run, keyed by the step's plan line (as
        ``done`` writes it): what the commands themselves said, for the stop's code-written answer."""
        results = getattr(code, "results", None) or {}
        prefix = f"{phase.name}: " if len(state["phases"]) > 1 else ""
        return {f"{prefix}{step.number}. {step.text}": "\n".join(results[step.number])
                for step in self._plan_lines(plan_text, names)
                if step.number in results and step.kind == "command" and (below is None or step.number < below)}

    def _no_agent_code_stop(self, phase, plan_text: str, code, names: dict) -> Optional[dict[str, Any]]:
        """Where the phase run stopped for want of a model: None when code ran every step (``code.stop`` "done"),
        else the stop's parts (``open_step``, ``done``, ``remaining``, ``suggested_plan``) for the phase's stop entry.

        A run that has no code steps at all (the phase's steps are the planner's) stops at its first step."""
        if code is not None and code.stop == "done":
            return None
        steps = self._plan_lines(plan_text, names)
        stop = str(getattr(code, "stop", "") or "") if code is not None else ""
        found = re.search(r"(\d+)$", stop) or re.search(r"step (\d+)", stop)
        number = int(found.group(1)) if found else (steps[0].number if steps else 1)
        by_number = {step.number: step for step in steps}
        step = by_number.get(number)
        text = step.text if step is not None else ""
        reason_code = next((reason for item, _, reason in (code.items if code is not None else ())
                            if item.number == number), "")
        nested, self._last_nested_stop = self._last_nested_stop, None
        if stop.startswith("step ") and stop.endswith("failed"):
            kind = no_agent_mode.STEP_FAILED
            observed = ""
            trajectory = getattr(self._workflow_tool_agent, "trajectory", None) or {}
            keys = [int(key.rsplit("_", 1)[1]) for key in trajectory
                    if key.startswith("observation_") and key.rsplit("_", 1)[-1].isdigit()]
            if keys:
                observed = " ".join(str(trajectory[f"observation_{max(keys)}"]).split())[:200]
            reason = "its command did not succeed" + (f": {observed}" if observed else "")
        elif stop.startswith("no path"):
            kind = no_agent_mode.STEP_NO_PATH
            reason = "its command is not available where the workflow is, and code found no path to it"
        elif code is not None and getattr(code, "page_hint", ""):
            kind = no_agent_mode.STEP_PAGE_CAP
            reason = "a full list that code read to its page cap, with rows still to read"
        elif reason_code in ("conditional step",):
            kind = no_agent_mode.STEP_CONDITIONAL
            reason = "a conditional step: whether it applies is a decision, which code does not make"
        elif reason_code in ("missing required field", "placeholder"):
            kind = no_agent_mode.STEP_MISSING_VALUE
            reason = "a value the command needs is not given in the step; it has to be found or asked"
        elif step is not None and code is not None and not any(item.number == number for item, _, _ in code.items):
            kind = no_agent_mode.STEP_EXTRA
            reason = "a step the planner wrote beyond the command's own steps; code runs only the command's own steps"
        else:
            kind = no_agent_mode.STEP_PROSE
            reason = ("a step code cannot run as written ({})".format(reason_code)
                      if reason_code and reason_code != "other step" else
                      "a step written in prose, which a model or the client has to carry out")
        if nested is not None:
            kind = no_agent_mode.STEP_NESTED
            inner = (nested.get("open_step") or {}).get("reason") or "it stopped"
            reason = f"the planning command {nested.get('command')} it runs stopped: {inner}"
        state = self._phase_run
        ran = {item.number for item, command, _ in (code.items if code is not None else ()) if command}
        prefix = f"{phase.name}: " if len(state["phases"]) > 1 else ""
        done = [*(state.get("done_steps") or []), *(state.get("carry_done") or []),
                *(f"{prefix}{step.number}. {step.text}" for step in steps
                  if step.number < number and step.number in ran)]
        later = [f"{step.number}. {step.text}" for step in steps if step.number > number]
        # A step that ran a nested command is carried out by that command's own remaining steps.
        own = [f"{number}. {text}"] if step is not None and nested is None else []
        inner_plan = [line for line in str(nested.get("suggested_plan") or "").splitlines()
                      if line.split(". ", 1)[-1] != no_agent_mode.ANSWER_LINE] if nested else []
        later_names, later_lines = self._later_phase_plan(phase)
        suggestion = [*(line.split(". ", 1)[-1] for line in [*inner_plan, *own, *later]), *later_lines,
                      no_agent_mode.ANSWER_LINE]
        return {
            "results": {**(state.get("step_results") or {}),
                        **self._step_results(phase, state, plan_text, names, code, below=number)},
            "partial_read": (code.partial_read(number) if kind == no_agent_mode.STEP_PAGE_CAP else None),
            # ``number`` is the step's own number in the phase's plan (as ``remaining`` and ``done`` number their
            # lines); ``plan_number`` is its place in ``suggested_plan``, which is renumbered from 1 (None when the
            # suggestion does not carry the step: a nested command's stop stands in for it).
            "open_step": {"number": number, "text": text, "kind": kind, "reason": reason,
                          "plan_number": len(inner_plan) + 1 if own else None},
            "done": done, "remaining": later, "nested": nested,
            "suggested_plan": no_agent_mode.numbered(suggestion),
            "phase": phase.name,
            "later_phases": later_names,
        }

    def _no_agent_answer_stop(self, phase, plan_text: str, names: dict, *, finished: bool = False) -> dict[str, Any]:
        """The stop for an answer a model would write (no ``__result__`` text, no code-written answer)."""
        steps = self._plan_lines(plan_text, names)
        done = list(self._phase_run.get("done_steps") or [])
        pending = [] if finished else [f"{step.number}. {step.text}" for step in steps]
        later_names, later_lines = ([], []) if finished else self._later_phase_plan(phase)
        return {
            "open_step": {"number": None, "text": "", "kind": no_agent_mode.STEP_ANSWER,
                          "reason": "the answer needs a model to write it from the facts: the command has no "
                                    "__result__ text, and no agent model is set"},
            "done": done, "remaining": pending, "nested": None,
            "results": dict(self._phase_run.get("step_results") or {}),
            "suggested_plan": no_agent_mode.numbered(
                [line.split(". ", 1)[-1] for line in pending] + later_lines + [no_agent_mode.ANSWER_LINE]),
            "phase": phase.name,
            "later_phases": later_names,
        }

    def _later_phase_plan(self, phase) -> tuple[list[str], list[str]]:
        """``(names, lines)`` of the phases after *phase* in the call in progress: their names, and the plan text of
        their steps as the template has them (no model call; ``planning_phases.placeholder_steps``), so a client
        can carry them out after the stop. A fact an earlier phase has not found yet reads ``<phase.field>``; a phase
        that fans out over an earlier phase's results gives its steps once, marked ``for each <item>``; a phase whose steps cannot be rendered says they are written when it runs."""
        from fastworkflow import planning_phases

        try:
            phases = list(self._live_planning_command_phases()[1])
        except Exception:  # noqa: BLE001 - the continuation is a convenience, never the reason a stop fails
            return [], []
        names = [item.name for item in phases]
        if phase.name not in names:
            return [], []
        later = phases[names.index(phase.name) + 1:]
        lines: list[str] = []
        for item in later:
            mark = f"phase {item.name}" + (f", for each {item.item}" if item.fans_out else "")
            try:
                steps = planning_phases.placeholder_steps(item)
            except Exception:  # noqa: BLE001
                steps = []
            if steps:
                lines.extend(f"({mark}) {line}" for line in steps)
            else:
                lines.append(f"({mark}) steps written when the phase runs")
        return [item.name for item in later], lines

    @staticmethod
    def _answer_needs_model(state: dict[str, Any]) -> bool:
        """Whether a phased call that has no ``__result__`` text has no answer a model did not write: a deferred
        answer (``__result__`` may return None) or no answer text at all."""
        recorded = [*state["answers"], *(state.get("phase_answers") or []),
                    state.get("extra_answer"), state.get("before_answer")]
        if any(isinstance(item, dict) and "deferred" in item for item in recorded):
            return True
        return not any(item for item in recorded if isinstance(item, str))

    def _planning_command_phases_stopped(self, stop: dict[str, Any]):
        """The sub-task's result when the command stopped for want of a model (no-agent mode): the stop as data
        (``not_run_here``) and a short answer text written in code. A command that is the turn's own call records
        the stop for the turn's ``fw_not_run`` list; a nested call hands it to its parent (``_close_planning_call``)."""
        state = self._phase_run
        names = state["phases"]
        index = names.index(stop["phase"]) if stop["phase"] in names else len(names) - 1
        facts = json.loads(json.dumps(plain_facts(state.get("outputs") or {}), default=str))
        entry = no_agent_mode.stop_entry(
            command=state["skill"], phase=stop["phase"], open_step=stop["open_step"], done=stop["done"],
            facts=facts, remaining=stop["remaining"], suggested_plan=stop["suggested_plan"],
            means="planning", sub_task=self._executing_sub_task(),
            later_phases=names[index + 1:], nested=stop.get("nested"),
            results=stop.get("results"), partial_read=stop.get("partial_read"))
        logger.info("planning command %s: stopped without an agent model at %s (phase %s)", state["skill"],
                    stop["open_step"]["kind"], stop["phase"])
        self._phase_run = self._phase_live = None
        if len(self._frame_stack()) == 1:
            # The turn's own call (a nested call's stop is its parent's step to carry: _close_planning_call).
            self._turn_not_run.append(entry)
        return dspy.Prediction(final_answer=no_agent_mode.explanation(entry), exhausted=False, not_run_here=entry)

    # ------------------------------------------------------------------
    # No-agent mode: a decision step pauses the command and resumes on the reply
    # ------------------------------------------------------------------

    def _can_ask_decision(self) -> bool:
        """Whether this turn can put a question to the client (no-agent mode): an attended turn of the agent path
        (the reply comes back through ``_resume_agent_message``), not a "/" command or a framework call."""
        # A nested call a "/" command made has no agent step in flight (``invoking_step`` None): its client is
        # not an agent loop, so the reply could not come back through the suspended step.
        called_by_a_step = len(self._frame_stack()) == 1 or getattr(self._top_frame(), "invoking_step", None) is not None
        in_framework_call = any(self._is_framework_command(frame.command_name or "") for frame in self._frame_stack()[1:])
        return bool(self._run_as_agent and waiting_module.can_ask(self)
                    and not self._assistant_round and called_by_a_step and not in_framework_call)

    def _suspend_for_decision(self, phase, plan_text: str, code, stop: dict[str, Any], own_steps: int):
        """The paused result when open step *stop* is a decision a client can make, else None (free work: the command
        ends with its stop).

        A conditional step (``only when <condition>, <command step>``) is decided yes or no; a command step that leaves
        out one plain required value takes that value. Either way the decision is the turn's waiting record (kind
        ``decision``; the phase's own state stays in ``_phase_run``), the question is the turn's pending prompt
        (``question_kind`` "phase_step"), and the reply is read by ``_read_decision_reply`` when the phase resumes."""
        from fastworkflow import planning_phases

        open_step = stop["open_step"]
        if (open_step["kind"] not in no_agent_mode.DECISION_KINDS or stop.get("nested") or code is None
                or not self._can_ask_decision()):
            return None
        number = open_step["number"]
        names = self._phase_command_names()
        step = next((s for s in self._plan_lines(plan_text, names) if s.number == number), None)
        reason = next((why for item, _, why in code.items if item.number == number), "")
        decision: dict[str, Any]
        if step is None:
            return None
        if open_step["kind"] == no_agent_mode.STEP_CONDITIONAL:
            found = planning_phases.conditional_decision(
                step.text, number, code._commands, code._core_commands, code._parameters, pick=code._pick)
            if found is None:
                return None
            decision = {"kind": no_agent_mode.STEP_CONDITIONAL, "condition": found[0], "inner": found[1]}
        else:
            found = (planning_phases.missing_value_field(step, code._parameters)
                     if reason == "missing required field" else None)
            if found is None:
                return None
            decision = {"kind": no_agent_mode.STEP_MISSING_VALUE, **found}
        state = self._phase_run
        done_before = len(state.get("done_steps") or [])
        decision.update({
            "command": state["skill"], "number": number, "text": step.text, "asked": 1, "own_steps": own_steps,
            "stop": json.loads(json.dumps(stop, default=str)),
            "facts": json.loads(json.dumps(plain_facts(state.get("outputs") or {}), default=str)),
        })
        state["carry_done"] = list(stop["done"][done_before:])
        return self._pause_for_decision(state, decision)

    def _pause_for_decision(self, state: dict[str, Any], decision: dict[str, Any], retry: str = "", *,
                            waited: bool = False):
        """The suspended result that puts decision *decision*'s question (*retry* opens it again; *waited*: the user asked
        for a moment, so the question stays as it was with a line before it) and makes it the turn's waiting record.
        The agent holds a minimal suspended state, so the suspension is kept, restored and resumed as an ask_user
        question is."""
        if waited:
            from fastworkflow import parameter_clarification

            question = parameter_clarification.wait_question(decision["question"])
        else:
            question = decision["question"] = no_agent_mode.decision_question(decision["stop"], decision, retry=retry)
        self._put_record(waiting_module.Waiting(waiting_module.KIND_DECISION, question=question,
                                                payload={"decision": decision}))
        agent = self._workflow_tool_agent
        agent._suspended = {
            "trajectory": dict(getattr(agent, "trajectory", None) or {}), "idx": self._next_step,
            "input_args": dict(getattr(agent, "inputs", None) or {}), "max_iters": 1, "clarification": question,
            "code": None}
        logger.info("planning command %s: paused at step %s for the client's decision (%s)", state["skill"],
                    decision["number"], decision["kind"])
        return dspy.Prediction(suspended=True, clarification=question, exhausted=False)

    def _decision_stop(self, decision: dict[str, Any], reason: str) -> dict[str, Any]:
        """The stop of a decision step that ends the command unanswered: the stop at the step, with *reason*."""
        stop = json.loads(json.dumps(decision["stop"], default=str))
        stop["open_step"] = {**stop["open_step"], "reason": reason}
        return stop

    def _read_decision_reply(self, state: dict[str, Any], decision: dict[str, Any], reply: str, phase,
                             plan_text: str, own_steps: int, names: dict):
        """Read the reply to a decision step's question: ``("go", resume)`` to carry on from the step (``resume`` is
        what ``planning_phases.CodeSteps`` takes), ``("ask", paused result)`` to ask again (the user asked for a moment,
        which counts nothing, or the reply gave nothing usable, once more and then the stop) or ``("stop", stop)`` to
        end the command with its structured stop. The reply is read by the one reader of every value question
        (``parameters.reply.read_decision``): the decision's own answer (yes or no, or the value), cancel, abort,
        "you misunderstood" and a different request. A step of a running plan has no which-command question to return
        to and a reply answers its question and nothing else, so cancel, abort, "you misunderstood" and a different
        request all end the command at this step as declined, each naming why; none is run as a new request."""
        from fastworkflow import planning_phases

        agent = self._workflow_tool_agent
        reply = " ".join(str(reply or "").split())
        reading = replies.read_decision(decision, reply)
        declined = {
            replies.DECLINE: "declined: the client chose not to decide this step; carry it out as its text says",
            replies.MISUNDERSTOOD: "declined: you said this is not what you meant, so the step was not decided; "
                                   "carry it out as its text says",
            replies.DIFFERENT: "declined: the reply asked for something else, and a reply answers its question and "
                               "nothing else, so it was not run as a new request (send it as its own message); the step "
                               "was not decided; carry it out as its text says",
        }
        if reading.kind in declined:
            self._waiting = None
            agent.clear_suspension()
            return "stop", self._decision_stop(decision, declined[reading.kind])
        if reading.kind == replies.WAIT and not decision.get("waited"):
            # The same question stays out and nothing is counted: the first request for a moment in a row. A further
            # one is a reply that gave nothing usable (counted below).
            decision["waited"] = 1
            return "ask", self._pause_for_decision(state, decision, waited=True)
        if reading.kind != replies.WAIT:
            decision.pop("waited", None)
        text: Optional[str] = None
        why = ""
        if decision["kind"] == no_agent_mode.STEP_CONDITIONAL:
            if reading.kind != replies.VALUES:
                why = "I could not read that reply as yes or no. "
            else:
                text = decision["inner"] if reading.values["answer"] == "yes" else ""
        elif value := reading.values.get(decision["field"]):
            reply = value
            candidate = planning_phases.step_with_value(decision["text"], decision["field"], reply)
            code = self._phase_code_steps(plan_text, own_steps, names, answers=False)
            item = next((i for i in (code.items if code is not None else ()) if i[0].number == decision["number"]), None)
            if item is not None and code.replacement_item(item[0], candidate)[1]:
                text = candidate
        if text is None and not why:
            why = (f"{reply!r} is not a valid value for {decision['field']} ({decision['type']}). " if reply
                   else f"I need a value for {decision['field']} ({decision['type']}). ")
        if text is None:
            # A reply that gave nothing usable is asked once more, then the step ends: the conditional's own limit, and
            # for a missing value the parameter loop's no-progress rule (every ask after the first followed such a reply).
            limit = (no_agent_mode.MAX_DECISION_ASKS if decision["kind"] == no_agent_mode.STEP_CONDITIONAL
                     else no_agent_mode.MAX_NO_PROGRESS)
            if decision["asked"] >= limit:
                self._waiting = None
                agent.clear_suspension()
                return "stop", self._decision_stop(decision, "the reply could not be read as an answer, twice")
            decision["asked"] += 1
            return "ask", self._pause_for_decision(state, decision, retry=why)
        self._waiting = None
        agent.clear_suspension()
        return "go", {"number": decision["number"], "text": text}

    # ------------------------------------------------------------------
    # The parameter question of a running part (fix-yu3l.4, fastworkflow/parameter_clarification.py)
    # ------------------------------------------------------------------
    #
    # Every part of a turn that has a review is resolved before it (``resolution.step``); while the parts run nothing
    # more is asked. The one question left here is a single unreviewed part's (the fast path): its refused typed
    # planning call (workflow_agent.planning_command_refusal_to_ask) asks for the missing values before anything of it
    # ran, the turn's waiting record (``parameters``, ``after`` ``resume_part``). The reply resumes the part
    # (_resume_parameter_loop_unlistened) and the turn is not run again.

    @property
    def _part_question_out(self) -> bool:
        """Whether the running part's value question is out (waiting record ``parameters``, ``after`` ``resume_part``)."""
        from fastworkflow import parameter_clarification

        held = self._waiting
        return (self._resolution_pending and held.kind == waiting_module.KIND_PARAMETERS
                and held.payload.get("after") == parameter_clarification.AFTER_RESUME_PART)

    def _put_part_loop(self, loop, waited: int = 0) -> None:
        """Put the running part's value question *loop* out as the turn's one waiting record (*waited*: the requests for
        a moment in a row, see ``parameter_clarification.waited_of``)."""
        from fastworkflow import parameter_clarification

        self._turn_port().put(parameter_clarification.waiting_record(
            loop, parameter_clarification.AFTER_RESUME_PART, loop.origin, waited=waited))

    def _part_ran_something(self) -> bool:
        """Whether the part that is running has run a command of its own (the review's plumbing and a question do not
        count). A question is asked only before it has: the reply then runs the part from its start, nothing twice."""
        return any(not (getattr(output, "framework", False) or output.command_name == "ask_user"
                        or self._is_framework_command(output.command_name))
                   for output in self._turn_outputs[self._part_outputs_start:])

    def _current_part_origin(self) -> Optional[int]:
        """The part of the turn that is running (None before the parts began: the gate)."""
        turn = self._turn_frame()
        return turn.origins[turn.subtask_index] if 0 <= turn.subtask_index < len(turn.origins) else None

    def _suspend_on_parameter_loop(self):
        """The suspended result that puts the running part's question to the user."""
        return dspy.Prediction(suspended=True, clarification=self._waiting.question, exhausted=False)

    def _open_parameter_loop(self, command, refusal, *, request_text: str = "") -> Optional[str]:
        """The question to put to the user about a refused typed call of planning command *command*, or None when it is
        not to be asked (the caller's fallback follows: the text planner or the agent, or with no agent model the part
        ends as not run, with ``_refusal_stop_reason`` naming what is missing). The loop is left on
        the turn's waiting record (``_put_part_loop``); the caller suspends on it (``_suspend_on_parameter_loop``).

        Only on a turn that did not pass the gate (``not_applicable``): every part of a turn that did, the fast path's
        one part included, was resolved there, its missing values asked then, and is not asked again while it runs
        (bd fix-ftdl.7.8). One question names every argument the call lacks. It is asked only when the turn is
        attended, the refusal names an input (a value the user can give) and the part has run nothing yet.

        The asking rule is the one of ``parameter_clarification`` in every session: a question is asked again while each
        reply closes at least one open value, a reply that closes none is asked once more, and the next one ends it. What
        the end hands over to differs: with an agent model the call goes to the text planner or the agent, with none the
        part ends as not run naming what is missing (``_refusal_stop_reason``).
        """
        from fastworkflow import parameter_clarification as loops, planning_commands

        if self._turn_subtask_review != "not_applicable":
            # The turn's parts were resolved at the gate: a value refused while it runs is not asked again (no agent
            # model: the part ends as not run naming it; with one: the agent takes over).
            return None
        error = (refusal if isinstance(refusal, planning_commands.PlanningCommandInputError)
                 else planning_commands.PlanningCommandInputError(str(refusal)))
        gaps = planning_commands.refusal_gaps(command, error)
        origin = self._current_part_origin()
        if self.no_agent_mode:
            self._refusal_stop_reason = gaps.describe() or str(error)
        if (self._part_ran_something() or not waiting_module.can_ask(self)
                or not planning_commands.refusal_names_an_input(command, error)):
            return None
        loop = self._rerun_loop
        if loop is not None and loop.command == command.name and loop.origin == origin:
            # The part ran again with the values the replies gave and its checks still refuse something.
            outcome = loops.reopen(loop, command, gaps, refusal=error)
            if outcome.kind != loops.ASK_AGAIN:
                return None
        else:
            loop, outcome = loops.open_loop(command, gaps, request_text=request_text, origin=origin, refusal=error)
        self._put_part_loop(loop)
        return outcome.question

    def _resume_parameter_loop_unlistened(self, reply: str) -> fastworkflow.CommandOutput:
        from fastworkflow import parameter_clarification as loops, planning_commands
        from fastworkflow._workflows.plan_metadata_extraction.resolution import DIFFERENT_REQUEST_REASON

        loop = loops.ParameterLoop.from_state(self._waiting.payload["loop"])
        waited = loops.waited_of(self._waiting)
        self._turn_port().clear()      # the question is no longer out; the re-run reads the loop for the rule's count
        self._rerun_loop = loop
        try:
            return self._rerun_part_with_reply(loop, reply, waited)
        finally:
            self._rerun_loop = None

    def _rerun_part_with_reply(self, loop, reply: str, waited: int = 0) -> fastworkflow.CommandOutput:
        from fastworkflow import parameter_clarification as loops, planning_commands
        from fastworkflow._workflows.plan_metadata_extraction.resolution import DIFFERENT_REQUEST_REASON

        message = self._suspended_message()
        self._note_agent_resume()
        self.complete_ask_user_entry(reply)
        outputs_before = len(self._turn_outputs)
        command = planning_commands.planning_command(self._app_workflow.folderpath, loop.command)
        if command is None:
            return self._continue_after_loop_part(message, self._not_run_part(
                command=loop.command, means="planning", reason=f"{loop.command} could not be loaded"), outputs_before)
        # Free text is read by the shared extractor, with or without an agent model (the asking rule is one).
        outcome = loops.advance(
            loop, reply, command,
            extract=lambda text: planning_commands.extracted_arguments(command, text, self._app_workflow), live=True,
            waited=waited)
        if outcome.kind == loops.ASK_AGAIN:
            self._put_part_loop(loop, outcome.waited)
            return self._settle_agent_result(message, self._suspend_on_parameter_loop())
        if outcome.kind == loops.WAIT:
            # The user asked for a moment: the same question stays out, and nothing is counted (the first in a row).
            self._put_part_loop(loop, outcome.waited)
            return self._settle_agent_result(message, dspy.Prediction(
                suspended=True, clarification=outcome.question, exhausted=False))
        if outcome.kind == loops.MISUNDERSTOOD:
            # Nothing of the part ran (``_open_parameter_loop`` asks only before it has), so it goes back to the
            # which-command question as at the gate: a one-part resolution of its text, built with no model call. The
            # reply to it goes through ``_resume_plan_resolution``, then the part's values, then the run.
            from fastworkflow._workflows.plan_metadata_extraction import resolution
            self._plan_resolution, self._turn_subtask_plan, question = resolution.ask_misunderstood_running_part(
                self._part_request_text(), self._resolution_turn())
            return self._settle_agent_result(message, dspy.Prediction(
                suspended=True, clarification=question, exhausted=False))
        if outcome.kind == loops.DIFFERENT:
            # A reply answers its question and nothing else: the part ends as not run (the reply is never run as a new
            # request) and the answer says why.
            result = self._not_run_part(command=loop.command, means="planning", reason=DIFFERENT_REQUEST_REASON)
        elif outcome.kind == loops.DECLINED:
            result = self._not_run_part(
                command=loop.command, means="planning",
                reason="it was declined" + (f" while its arguments {', '.join(outcome.missing)} were still missing"
                                            if outcome.missing else ""))
        elif outcome.kind == loops.EXHAUSTED and self.no_agent_mode:
            # The one hand-over: with an agent model the part runs on with what was given (the call refused again goes to
            # the text planner or the agent, never asked a third time: ``reopen``); with none it ends as not run.
            result = self._not_run_part(
                command=loop.command, means="planning",
                reason=f"it cannot run as planned ({outcome.describe}), and the agent model that would take over is "
                       "not set (LLM_AGENT)")
        else:
            result = self._run_loop_part(loop)
        return self._continue_after_loop_part(message, result, outputs_before)

    def _continue_after_loop_part(self, message: str, result, outputs_before: int):
        """Carry the turn on after the loop's part gave *result*: its answer is recorded and the later parts run, as
        after any part that was suspended (``_resume_agent_message``)."""
        if self._subtasks and getattr(result, "suspended", None) is not True:
            self._record_subtask_answer(result, self._typed_output_since(outputs_before))
            if getattr(result, "exhausted", None) is True:
                self._hand_off(self._subtask_index)
            result = self._run_subtasks_from(self._subtask_index + 1)
        return self._settle_agent_result(message, result)

    def _run_loop_part(self, loop):
        """Run the part the loop belongs to, now that its values are given. Nothing of the part ran before its question
        (``_part_ran_something``), so it runs from its start and nothing runs twice: its typed call again, on its own
        text with the reply after it, and the matcher is not asked (the command is known)."""
        turn = self._turn_frame()
        k = turn.subtask_index
        origin = turn.origins[k]
        floors = getattr(self, "_evidence_floors", None)
        if not isinstance(floors, dict):
            floors = self._evidence_floors = {}
        floors.setdefault(origin, self._next_step)
        self._enable_all_tools()
        self._part_outputs_start = len(self._turn_outputs)
        tail = loop.tail()

        def with_reply(text: str) -> str:
            return f"{text}\n{tail}" if tail else text

        if len(turn.subtasks) == 1:
            # An unsplit turn: the call _run_turn_subtasks made, with the reply after the request.
            return self._call_agent_for_query(
                with_reply(turn.subtasks[k]), with_reply(self._turn_user_message or turn.subtasks[k]),
                evidence_floor=0, planning_name=loop.command)
        text = with_reply(turn.subtasks[k])
        return self._call_agent_for_query(text, text, first_step=self._next_step, evidence_floor=floors[origin],
                                          planning_name=loop.command)

    def _part_request_text(self) -> str:
        """The request text of the part that is running: its own text in a split turn, else the turn's message. What
        a not-run entry carries as ``request`` and what the plan-choice question quotes."""
        turn = self._turn_frame()
        split = len(set(turn.origins)) > 1
        origin = turn.origins[turn.subtask_index] if split and 0 <= turn.subtask_index < len(turn.origins) else 0
        return str(self._original_text(origin) if split else self._turn_user_message or "")

    def _not_run_part(self, *, command: Optional[str], reason: str, means: str, suggested_plan: str = "",
                      plans_for: Optional[dict[str, float]] = None):
        """The sub-task's result for a part that is not run here (no-agent mode): an ad hoc part, or a bound
        command that could not start, with the planner's text plan when one was made. Recorded for the turn's
        ``fw_not_run`` list.

        *plans_for* marks an ad hoc part (the matcher's distribution for it, possibly empty): the entry then lists the
        planning commands available here (``available_plans``), and the answer text says so. A later ad hoc part
        of the turn with the same plans lists only its own closest ones and points to the part that lists them all."""
        plans = (no_agent_mode.available_plans(self._app_workflow, plans_for)
                 if plans_for is not None and command is None else None)
        request = self._part_request_text() if plans is not None else None
        entry = no_agent_mode.stop_entry(
            command=command, phase=None, open_step=None, done=[], facts={}, remaining=[],
            suggested_plan=suggested_plan, means=means, sub_task=self._executing_sub_task(), reason=reason,
            available_plans=plans, request=request)
        logger.info("sub-task not run without an agent model (%s): %s", means, reason)
        listed_in = next((earlier.get("sub_task") for earlier in self._turn_not_run
                          if plans and earlier.get("sub_task") is not None
                          and {p["name"] for p in earlier.get("available_plans") or []} == {p["name"] for p in plans}
                          and earlier.get("sub_task") != entry["sub_task"]), None)
        self._turn_not_run.append(entry)
        return dspy.Prediction(final_answer=no_agent_mode.explanation(entry, listed_in), exhausted=False,
                               not_run_here=entry)

    # ------------------------------------------------------------------
    # Planning commands as nested frames (design 3.5)
    # ------------------------------------------------------------------

    def _nested_budget_spent(self) -> bool:
        """True inside a nested frame once the turn-wide step budget is spent.

        Only nested frames stop on it: the turn frame is bounded by its own caps (design 3.5).
        """
        return len(self._frame_stack()) > 1 and self._turn_steps_used >= turn_max_steps()

    def run_framework_planning_command(self, qualified_name: str, params: Any, step_text: str, *,
                                       unattended: bool = False, within_turn: bool = False):
        """The framework's own entry point for a framework planning command (fix-3rsq.plan-review.1).

        Runs *qualified_name* (review_subtasks) as one turn, through the same nested frame
        and executor a planning step uses, without the agent's explicit-name resolution (which refuses
        framework commands). Returns the command's output, or the awaiting output when the review asks the
        user (its replies come as ordinary messages on this session).

        ``within_turn``: the call runs inside a turn already begun (the sub-task gate): that turn's state, its
        unattended flag and its approval surface are kept, and a suspension returns the suspended prediction for the
        caller to settle, not the awaiting output (fix-3rsq.plan-review.2.8). Without it the call begins a turn of its
        own, with *unattended* (attended by default, as the review always was).
        """
        from fastworkflow.utils.react import AskUserSuspend

        from fastworkflow.workflow_agent import framework_call_scope

        self._ensure_agent_initialized()
        if not within_turn:
            self._begin_turn(step_text, unattended=unattended)
            self._note_approval_inbound(None, None, False)
        # The turn frame has no agent step of its own: when the call finishes, its output is the turn's answer.
        self.push_active_workflow(self._app_workflow)
        try:
            with framework_call_scope():
                output = self.run_planning_command(qualified_name, params, step_text)
        except AskUserSuspend as asked:
            if (held := self._waiting) is not None and held.kind == waiting_module.KIND_REVIEW:
                held.payload["gate"] = bool(within_turn)     # opened by ``review_engine.open_review``; the gate's, or not
            else:
                self._ask_agent_question(asked.clarification_request)
            if within_turn:
                return dspy.Prediction(suspended=True, clarification=asked.clarification_request)
            return self._awaiting_user_output(asked.clarification_request)
        finally:
            self.pop_active_workflow()
        self.append_turn_output(output)
        return output

    def _put_review_record(self, record) -> None:
        """A review was opened on the approval store (``review_engine.open_review`` calls this): it is the turn's waiting
        record, pointing at the store's record (the store keeps it). ``gate`` is set by ``run_framework_planning_command``
        when the sub-task gate of the turn opened it (its accept runs the turn's sub-tasks), not a stand-alone call."""
        self._put_record(waiting_module.Waiting(
            waiting_module.KIND_REVIEW, question=str(record.rendering or ""),
            payload={"record_id": record.id, waiting_module.SLOT_KEY: None, "gate": False,
                     "suspended_agent": bool(record.suspended_agent)}))

    def run_planning_command(self, qualified_name: str, params: Any,
                             step_text: str) -> fastworkflow.CommandOutput:
        """Run planning command *qualified_name* as a nested frame, for the step *step_text*.

        *params* are the arguments the executor extracted and checked: an ``Input``
        instance or a dict. Returns the command's output, which the step that called it
        holds as its run. A call that asks the user raises AskUserSuspend and keeps its
        frame on the stack for the resume (sibling task; design 3.5, Ask_user inside a child).
        """
        frames = self._frame_stack()
        if any(frame.command_name == qualified_name for frame in frames):
            return self._nested_refusal(
                qualified_name,
                f"{qualified_name} is already running; call its steps directly.")
        limit = turn_max_steps()
        if self._turn_steps_used >= limit:
            return self._nested_refusal(qualified_name, NESTED_BUDGET_SPENT_MESSAGE)
        folderpath = self._app_workflow.folderpath if self._app_workflow is not None else ""
        command = planning_commands.planning_command(folderpath, qualified_name) if folderpath else None
        if command is None:
            return self._nested_refusal(
                qualified_name, f"{qualified_name} is a planning command that could not be loaded.")
        args = _planning_args(params)
        try:
            phases = planning_commands.call_phases(command, args)
        except planning_commands.PlanningCommandCallError as error:
            return self._nested_refusal(qualified_name, f"{qualified_name} could not run: {error}")

        # The step in flight is the parent's execute step k; the child's steps number after
        # it and after every step already run (step numbers stay unique across frames).
        parent = self._top_frame()
        alias = current_execute_alias(parent.agent) if parent.agent is not None else None
        invoking_step = int(alias[1:]) if alias else None
        first_step = self._next_step
        if invoking_step is not None:
            first_step = max(first_step, invoking_step + 1)
        stem = qualified_name.split("/")[-1]
        count = parent.child_counts.get(stem, 0) + 1
        parent.child_counts[stem] = count
        ran_in_clause, ran_in_context_name = self._current_record()
        run_state = getattr(parent.agent, "run_state", None)
        execute_call = run_state.get("execute_call", {}) if isinstance(run_state, dict) else {}
        frame = PlanFrame(
            kind="planning",
            command_name=qualified_name,
            depth=parent.depth + 1,
            alias=f"{parent.alias}/{stem}#{count}",
            subtasks=[step_text],
            origins=[0],
            subtask_index=0,
            invoking_step=invoking_step,
            first_step=first_step,
            ran_in_clause=ran_in_clause,
            ran_in_context_name=ran_in_context_name,
            call_args=json.loads(json.dumps(args, default=str)),
            command_text=execute_call.get("command", ""),
            folded_note=execute_call.get("folded_note", ""),
        )
        frames.append(frame)
        self._attach_nested_agent(frame)
        # The call's plan is its phases, which carry their own steps: nothing is planned
        # with a model here (the phases of call_phases).
        call = planning_commands.PlanningCommandCallPlan(plan="", args=args, notes=[], prediction=None,
                                            phases=phases)
        try:
            result = self._start_planning_command_phases(step_text, command, call, first_step=first_step)
        except AskUserSuspend:
            raise  # the frame stays for the resume
        except BaseException:
            self._pop_frame(frame)
            raise
        if getattr(result, "suspended", None) is True:
            # The frame stays on the stack; the parent's step is suspended with it.
            raise AskUserSuspend(str(getattr(result, "clarification", "") or ""))

        command_parameters = params if hasattr(params, "model_dump") else (
            planning_commands.input_instance(command, args))
        return self._close_planning_call(frame, result, command_parameters)

    def _attach_nested_agent(self, frame: PlanFrame) -> None:
        """Give a nested frame its agent: each of its model steps counts against the turn-wide
        budget, and ask_user is off for an unattended turn. Also set on a restored frame, which
        is built the same way before its suspended state is imported."""
        frame.agent = self._build_react_agent()
        # One policy verifier per conversation (fastworkflow.policy_runtime): a nested planning-command
        # frame shares the turn agent's, so what was read and asked is remembered across frames.
        turn_policy = getattr(self._turn_frame().agent, "policy", None)
        if turn_policy is not None:
            frame.agent.policy = turn_policy
        frame.agent.disabled_tools = (
            frozenset({"ask_user"}) if getattr(self, "_turn_unattended", False) else frozenset())
        limit = turn_max_steps()

        def count_step() -> bool:
            # Each model step of a nested frame counts against the turn-wide budget.
            self._turn_steps_used += 1
            return self._turn_steps_used < limit

        frame.agent.on_iteration = count_step

    def _close_planning_call(self, frame: PlanFrame, result,
                             command_parameters) -> fastworkflow.CommandOutput:
        """Pop *frame*, a nested call that finished with *result*, and build the output its
        parent's step holds: the call's step range is recorded on the parent."""
        limit = turn_max_steps()
        ran = self._next_step > frame.first_step
        self._pop_frame(frame)
        if ran:
            self._top_frame().child_step_ranges.append([frame.first_step, self._next_step - 1])

        output = getattr(result, "command_output", None)
        if output is None:
            output = fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=str(getattr(result, "final_answer", "") or "")),
            )
        if getattr(result, "phase_exhausted", False) and self._turn_steps_used >= limit:
            # The turn-wide budget stopped this call's steps (design 3.5): its response says so.
            # A call that finished, or a phase that ran out of its own cap, keeps its answer.
            output = fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=NESTED_BUDGET_SPENT_MESSAGE, success=False),
            )
        if getattr(result, "phase_failed", None) or getattr(result, "phase_exhausted", False):
            # Out of steps or failed: the step's observation says so, and the run is not a success.
            output.command_response.success = False
        if (stop := getattr(result, "not_run_here", None)) is not None:
            # No-agent mode: the call stopped where code cannot go on. A call made by the turn itself (the
            # deterministic path) carries its stop as the turn's data; a call made by a parent's step is a
            # failed step of that parent, which stops in turn and carries this stop in its own (nested).
            if len(self._frame_stack()) == 1 and self._phase_run is None:
                output.command_response.artifacts[NOT_RUN_ARTIFACT] = [stop]
            else:
                self._last_nested_stop = stop
                output.command_response.success = False
        output.command_name = frame.command_name
        output.command_parameters = command_parameters
        # The context the call ended on: the one its parent's later steps start from.
        active = self.get_active_workflow()
        if active is not None:
            output.workflow_name = active.folderpath.split("/")[-1]
            output.context = active._current_command_context_displayname
        return output

    def _finish_framework_call(self, result) -> fastworkflow.CommandOutput:
        """The framework's own call (run_framework_planning_command) finished after a suspension: its planning
        command's typed output is the turn's answer, and its frame is closed."""
        frame = self._top_frame()
        command = planning_commands.planning_command(self._app_workflow.folderpath, frame.command_name)
        output = self._close_planning_call(frame, result, planning_commands.input_instance(command, frame.call_args))
        self._pop_frame(frame)
        self.append_turn_output(output)
        return output

    def _finish_nested_frame(self, result) -> str:
        """Close the nested call on top, which finished after a suspension, and return the
        observation of the step it was called from.

        That step is still in flight, so the call's run is held under it, and its turn output
        and action-log record are written, as a call that finishes at once writes them
        (_execute_workflow_query). The observation reads the run as the step's own would have,
        with its handle line. The step's alias, command and folded note are the frame's: a
        restored frame has them, where the step's agent may not have its trajectory yet.
        """
        # inline import avoids circular import: workflow_agent imports this module at load (CommandCancelledError)
        from fastworkflow.workflow_agent import (
            _append_action_record, _append_turn_output, _hold_command_run, action_record,
            command_run_response, run_observation)

        frame = self._top_frame()
        command = planning_commands.planning_command(
            self._app_workflow.folderpath, frame.command_name)
        output = self._close_planning_call(
            frame, result, planning_commands.input_instance(command, frame.call_args))
        _append_turn_output(self, output)
        alias = f"O{frame.invoking_step}" if frame.invoking_step is not None else None
        # The child's last step left the workflow where it ended: the run is held on that record.
        now_clause, now_name = self._current_record()
        if alias is not None:
            _hold_command_run(self, output, now_clause, now_name, alias=alias)
        response = command_run_response(output)
        _append_action_record(self, action_record(frame.command_text, output, response))
        run_state = getattr(self._workflow_tool_agent, "run_state", None)
        if isinstance(run_state, dict):
            # A folded call that suspended never reached the step that reads this.
            run_state.pop("folded_command", None)
        return run_observation(
            alias, frame.ran_in_clause, response, folded_note=frame.folded_note,
            context_changed=now_clause != frame.ran_in_clause, now_clause=now_clause)

    def _nested_refusal(self, qualified_name: str, message: str) -> fastworkflow.CommandOutput:
        """A planning-command call that did not run: a failed output and no frame."""
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response=message, success=False),
            command_name=qualified_name,
        )

    def _pop_frame(self, frame: PlanFrame) -> None:
        """Remove *frame* from the stack when it is on top, which releases its agent and phases."""
        frames = self._frame_stack()
        if frames and frames[-1] is frame:
            frames.pop()

    def _archived_command_runs(self, *, from_step: Optional[int] = None,
                               to_step: Optional[int] = None) -> list[dict[str, Any]]:
        """The command runs archived for this turn, in run order (``offload_command_runs``);
        only steps ``from_step``..``to_step`` (inclusive) when given, filtered in SQL. Runs
        inside the top frame's child step ranges are left out (design 3.5)."""
        from fastworkflow.observation_offloading.state import (
            archive_for_path, observability_db_path, scope_for_host)

        runs = archive_for_path(observability_db_path(self)).command_runs(
            scope_for_host(self), from_step=from_step, to_step=to_step)
        return runs_outside_ranges(runs, self._top_frame().child_step_ranges)

    def _phase_command_names(self) -> dict[str, list[str]]:
        """``commands`` / ``core_commands`` a phase's steps are read with, as the finish check
        reads them. Without the routing definition only the runs' and the facts' commands
        are known (``planning_phases.fill_output`` adds those)."""
        from fastworkflow.workflow_agent import core_command_names

        return {"commands": self._workflow_command_names(),
                "core_commands": list(core_command_names())}

    def _workflow_command_names(self) -> list[str]:
        """The app workflow's command names a step may name: the routed commands and the
        planning commands (which routing does not hold, and which a step may call, 3.5).
        Empty when the names cannot be read: the steps are then read with what is known."""
        if self._app_workflow is None:
            return []
        folderpath = self._app_workflow.folderpath
        names: list[str] = []
        try:
            routing = fastworkflow.RoutingRegistry.get_definition(folderpath)
            names += list(routing.routing_definition_map)
            # Framework execution commands (review_proposal) are not agent-visible, but a plan step may
            # call them by exact name: the framework's own review steps.
            names += [name for name, meta in routing.command_directory.map_command_2_metadata.items()
                      if meta.framework and not meta.is_planning]
        except Exception:  # noqa: BLE001 - read the steps with what is known
            pass
        names += list(self._planning_commands())
        return names

    def _planning_commands(self) -> Mapping[str, Any]:
        """The app workflow's loaded planning commands (qualified name -> ``PlanningCommandCommand``),
        or {} when they cannot be read: a step then reads as before, with the routed commands."""
        if self._app_workflow is None:
            return {}
        try:
            return planning_commands.resolve_field_sources(self._app_workflow.folderpath).planning_commands
        except Exception:  # noqa: BLE001 - read the steps with what is known
            return {}

    def _current_record(self) -> tuple[str, str]:
        """``(clause, context name)`` of the record the workflow is on now (``""`` at the root)."""
        from fastworkflow.context_identity import context_clause_for, context_identity

        active = self.get_active_workflow() or self._app_workflow
        if active is None:
            return "", ""
        return context_clause_for(active), context_identity(active)[0]

    def _run_planning_command_phase(self, command, phases, first_step: int, *,
                         resume_observation: Optional[str] = None,
                         resume_completes_step: bool = False,
                         skip: bool = False, skip_error: Optional[str] = None):
        """Run the current phase (of a fan-out phase: its current item) as one sub-plan
        and fill its output.

        Returns ``("done", None)``, ``("suspended", result)`` or ``("failed", message)``.
        With *skip* the phase's callable ``when`` was false (or, with *skip_error*, could
        not be decided): only its span is written (``skipped``), nothing runs, and the
        result is ``("done", None)`` (``("failed", skip_error)``).
        One ``fw.skill.phase`` span per run (a resumed run opens a new one). An
        answer-only phase (``planning_phases.is_answer_only``) is not an executor run: one
        model call with no tools writes its answer (``planning_phases.write_answer``).
        """
        from fastworkflow import planning_phases
        from fastworkflow.planning_commands import number
        from fastworkflow.observation_offloading.facts import FactsError, MissingFactError, fact_sources
        from fastworkflow.workflow_agent import _what_can_i_do

        state = self._phase_run
        state["missing_fact"] = False
        if not resume_observation:
            self._last_nested_stop = None
        index = state["index"]
        phase = phases[index]
        last = index == len(phases) - 1
        uses_result = state.get("uses_result")
        phase_names = frozenset(state["phases"])
        # A call that runs one phase (call_phases: one phase, whatever its output): its
        # run gets the executor's own step cap, not PHASE_MAX_STEPS.
        # A call that runs one phase has the agent's own cap; a call of several gives each phase the phase cap.
        budget = (getattr(self._workflow_tool_agent, "max_iters", None)
                  if len(phases) == 1 else None) or planning_phases.phase_step_budget()
        resumed = resume_observation is not None
        attributes = {
            "skill": state["skill"],
            "phase": phase.name,
            "phase_index": index,
            "phase_count": len(phases),
            "step_budget": budget,
            "first_step": first_step,
            "resumed": resumed,
            "reextracted": True if state.get("reextracted") else None,
        }
        item = None
        others: list[Any] = []
        if phase.fans_out and not skip:
            items = state["items"]
            k = state["item_index"]
            item, others = items[k], items[:k] + items[k + 1:]
            attributes.update({"item": item, "item_index": k, "item_count": len(items),
                               "items_total": state["items_total"]})
            last_item = k == len(items) - 1
        else:
            last_item = True
        span = tracing.start_span(self, tracing.SPAN_PLANNING_COMMAND_PHASE, attributes=attributes)
        if skip:
            if skip_error is not None:
                tracing.end_span(self, span, status=tracing.STATUS_ERROR,
                                 attributes={"skipped": False, "facts_error": skip_error})
                return "failed", skip_error
            tracing.end_span(self, span, attributes={"skipped": True})
            return "done", None
        names = self._phase_command_names()
        if phase.output is not None:
            names["commands"] = names["commands"] + [
                source.command for source in fact_sources(phase.output)]
        # A paused decision step (no-agent mode, ``_suspend_for_decision``): the reply is read here, and the phase carries
        # on from the decided step as a code run of its own (its plan, first step and facts are the phase's).
        held = self._waiting
        decision = (held.payload["decision"] if resumed and held is not None and held.kind == waiting_module.KIND_DECISION
                    else None)
        resume_spec = None
        own_steps = state.get("own_steps", 0)
        if decision is not None:
            plan_text, own_steps = state["plan"], decision["own_steps"]
            verdict, value = self._read_decision_reply(state, decision, resume_observation, phase, plan_text,
                                                       own_steps, names)
            if verdict != "go":
                tracing.end_span(self, span, status=tracing.STATUS_CANCELLED if verdict == "ask" else tracing.STATUS_OK,
                                 attributes={"steps": plan_text, "steps_used": 0, "exhausted": False,
                                             "suspended": verdict == "ask", "answer_only": False})
                return ("suspended" if verdict == "ask" else "stopped"), value
            resume_spec, resumed, first_step = value, False, self._next_step
        if not resumed and decision is None:
            state["carry_done"] = []
            outputs = self._phase_outputs(phases, index)
            try:
                steps = (planning_phases.phase_steps(phase, outputs, item, phase_names=phase_names)
                         if phase.fans_out
                         else planning_phases.phase_steps(phase, outputs, phase_names=phase_names))
            except planning_phases.PhaseError as error:
                tracing.end_span(self, span, status=tracing.STATUS_ERROR,
                                 attributes={"steps": "", "facts_error": str(error)})
                return "failed", str(error)
            # Code may run the phase's own steps only: never the planner's steps_before
            # (their phase has none of its own) or steps_after (after them).
            own_steps = 0 if phase.name == state.get("before_phase") else len(steps)
            state["own_steps"] = own_steps       # a resumed run (also after a restore) reads it back
            steps = steps + (state["steps_after"] if last and last_item else [])
            state["first_step"] = first_step
            state["plan"] = number(steps)
            state["writes_run"] = {}       # the writes this phase run has made (command_writes)
        plan_text = state["plan"]
        answer_only = not resumed and planning_phases.is_answer_only(plan_text, **names)
        if answer_only and last and planning_phases.answer_by_code(plan_text):
            # 3.16 (b): a last phase of 'say that' lines the load check could not read (its template
            # did not build). Refused as the load check refuses it, as this phase's failure.
            message = planning_phases.say_that_last_message(command.path, phase.name)
            tracing.end_span(self, span, status=tracing.STATUS_ERROR,
                             attributes={"steps": plan_text, "answer_only": True,
                                         "facts_error": message})
            return "failed", message
        planned = sorted(planning_phases.planned_steps(plan_text, **names))
        item_frame = (state["item_index"], len(state["items"]), item) if phase.fans_out else None
        try:
            if answer_only:
                # Steps that name no command: one model call, with no tools, writes the
                # answer from the facts the steps carry (none when every step is verbatim
                # or a guard: code writes it, planning_phases.answer_by_code). Nothing can run a command.
                # A long list the steps do not tell the answer to name is cut short in the
                # prompt (planning_phases.capped_steps); a list to name is given whole.
                earlier_outputs = self._phase_outputs(phases, index)
                prompt_steps = planning_phases.capped_steps(plan_text, earlier_outputs)
                user_query = planning_phases.phase_query(
                    state["query"], prompt_steps, index=index, count=len(phases),
                    name=phase.name, answer_only=True, item=item_frame)
                if not last and planning_phases.answer_by_code(prompt_steps):
                    # 3.16 (a): a phase before the last made only of 'say that' lines writes no
                    # answer; its text reaches the answer only through __result__.
                    tracing.end_span(self, span, attributes={"skipped": True, "answer_only": True})
                    return "done", None
                if uses_result and not planning_phases.answer_by_code(prompt_steps):
                    # __result__ may return ANSWER_BY_MODEL, and then the model writes this answer: it
                    # is asked then, not now (WorkflowExecutionContext._settled_phase_answer).
                    marker = self._defer_phase_answer(index, user_query, prompt_steps)
                    tracing.end_span(self, span, attributes={"skipped": True, "answer_only": True,
                                                             "uses_result": True})
                    self._record_phase_answer(phase, marker)
                    return "done", None
                if self.no_agent_mode and not planning_phases.answer_by_code(prompt_steps):
                    # No agent model writes this answer: the command stops here, with the facts it has.
                    tracing.end_span(self, span, attributes={
                        "steps": plan_text, "steps_used": 0, "exhausted": False, "suspended": False,
                        "answer_only": True, "answer_by_code": False})
                    return "stopped", self._no_agent_answer_stop(phase, plan_text, names)
                result = self._call_agent_with_retry(
                    lambda: dspy.Prediction(
                        final_answer=planning_phases.answer_only_answer(
                            user_query, prompt_steps, earlier_outputs),
                        suspended=False, exhausted=False),
                    trace_input=user_query,
                )
            else:
                # The finish check reads this phase's steps against what this phase ran.
                self._initial_plan_text = plan_text
                self._attach_finish_check(
                    state["first_step"], values=True,
                    item_binding=(item, others, planned) if phase.fans_out and phase.bind
                    else None,
                    final=bool(last and last_item) and self._last_subtask_running())
                if resumed:
                    result = self._call_agent_resume(
                        resume_observation, completes_step=resume_completes_step)
                    # The code that ran the phase's leading steps before the suspension is
                    # still the run's: its count and where it stopped are the phase's.
                    code = getattr(self._workflow_tool_agent, "last_code", None)
                else:
                    # A phased command with __result__ gets its answer from __result__ (or, when that
                    # returns ANSWER_BY_MODEL, from the deferred model call): its last phase's agent only runs steps.
                    result_driven = bool(uses_result and last and not state.get("flat"))
                    # Code runs the leading mechanical steps first (planning_phases.CodeSteps).
                    # A result-driven last phase takes no model step once code has run every step: the
                    # answer is __result__'s (or the deferred answer call's), never the agent's.
                    code = self._phase_code_steps(
                        plan_text, own_steps, names,
                        answers=(last or phase.name == state.get("extra_phase")) and not result_driven,
                        resume=resume_spec)
                    # A phase other than the last sees only its own steps, never the request.
                    user_query = planning_phases.phase_query(
                        state["query"], plan_text, index=index, count=len(phases),
                        name=phase.name, item=item_frame,
                        code_steps=code is not None and code.runs_any,
                        result_driven=result_driven)
                    if uses_result and last and last_item:
                        notes = list(state.get("notes") or [])
                        if notes:
                            user_query = f"{user_query}\n\n{' '.join(notes)}"
                    if code is not None and last:
                        # Code may run every step: the agent then answers from their outputs.
                        code.done_inputs = {"user_query": planning_phases.phase_query(
                            state["query"], plan_text, index=index, count=len(phases),
                            name=phase.name, item=item_frame, code_ran_all=True,
                            result_driven=result_driven)}
                    available_commands = _what_can_i_do(self)

                    def run_phase():
                        self._workflow_tool_agent.code_steps = code
                        return self._workflow_tool_agent(
                            user_query=user_query,
                            available_commands=available_commands,
                            first_step=first_step,
                            max_iters=budget,
                        )

                    result = self._call_agent_with_retry(run_phase, trace_input=user_query)
        except BaseException as exc:
            tracing.end_span(self, span, status=tracing.status_for_dispatch_exception(exc),
                             attributes={"steps": plan_text, "answer_only": answer_only,
                                         "error_type": type(exc).__name__})
            raise
        by_code = {}
        if not answer_only and (not resumed or code is not None):
            by_code = {"steps_by_code": code.count if code is not None else 0,
                       "code_stop": (code.stop if code is not None else None) or "off"}
        if answer_only:
            steps_used = 0
            # Whether code wrote the answer from the steps, with no model call (fix-hia1.30).
            by_code = {"answer_by_code": planning_phases.answer_by_code(prompt_steps)}
        else:
            trajectory = getattr(self._workflow_tool_agent, "trajectory", None) or {}
            indices = [int(key.rsplit("_", 1)[1]) for key in trajectory
                       if key.rsplit("_", 1)[-1].isdigit()]
            steps_used = len({key for key in trajectory if key.startswith("thought_")
                              and int(key.rsplit("_", 1)[1]) >= state["first_step"]})
            if getattr(result, "suspended", None) is True:
                tracing.end_span(self, span, status=tracing.STATUS_CANCELLED,
                                 attributes={"steps": plan_text, "steps_used": steps_used,
                                             "answer_only": False, "suspended": True,
                                             **by_code})
                return "suspended", result
            if indices:
                # Never backwards: a nested frame this phase ran (run_planning_command) may be past it.
                self._next_step = max(self._next_step, max(indices) + 1)
            if self.no_agent_mode and (stop := self._no_agent_code_stop(phase, plan_text, code, names)) is not None:
                if (asked := self._suspend_for_decision(phase, plan_text, code, stop, own_steps)) is not None:
                    # A step the client decides: the command pauses with a question (and resumes on the reply).
                    tracing.end_span(self, span, status=tracing.STATUS_CANCELLED, attributes={
                        "steps": plan_text, "steps_used": steps_used, "exhausted": False, "suspended": True,
                        "answer_only": False, **by_code})
                    return "suspended", asked
                # Code ran every step it can and no model takes over: the command ends here (no-agent mode).
                tracing.end_span(self, span, attributes={
                    "steps": plan_text, "steps_used": steps_used, "exhausted": False, "suspended": False,
                    "answer_only": False, **by_code})
                return "stopped", stop
        if uses_result and last and not state.get("flat") and not answer_only:
            # 3.16: the agent wrote no answer; the answer is __result__'s, or the deferred model call's.
            answer_query = planning_phases.phase_query(
                state["query"], plan_text, index=index, count=len(phases), name=phase.name,
                answer_only=True, item=item_frame)
            answer = self._defer_phase_answer(index, answer_query, plan_text)
        else:
            answer = planning_phases.clean_answer(getattr(result, "final_answer", "") or "")
        self._record_phase_answer(phase, answer)
        exhausted = getattr(result, "exhausted", None) is True
        if exhausted:
            # The call (run_planning_command) reports it as the frame's failure.
            state["exhausted"] = True
        closing = {"steps": plan_text, "steps_used": steps_used, "exhausted": exhausted,
                   "suspended": False, "answer_only": answer_only, **by_code}
        record = None
        facts = None
        issues = None
        need_runs = phase.output is not None or (phase.fans_out and phase.bind)
        try:
            ended_on, record_context = self._current_record() if need_runs else (None, None)
            record = ended_on
            runs = (self._archived_command_runs(from_step=state["first_step"])
                    if need_runs else [])
            earlier = (self._archived_command_runs(to_step=state["first_step"] - 1)
                       if phase.output is not None else [])
            if phase.fans_out and phase.bind:
                # The item is bound to the record that names it, wherever the run ended.
                issues = planning_phases.item_record_issues(runs, item, others, ended_on, planned)
                record = planning_phases.item_record(runs, item, ended_on, planned=planned)
                if record != ended_on:
                    record_context = None
            elif phase.output is not None and not phase.fans_out:
                # The phase did its steps on one record, then moved off it: fill on
                # that record (fix-hia1.10), not on the clause the run ended on.
                rebound = planning_phases.planned_record(
                    runs, plan_text, ended_on=ended_on, **names)
                if rebound is not None:
                    record, record_context = rebound, None
            if phase.output is not None:
                if phase.fans_out and phase.bind and record is None:
                    raise FactsError(f"item {item} was never opened: no command ran on a "
                                     "record that names it")
                # Only runs that do one of the phase's own steps count; an opening step
                # the workflow was already on is done by the run that entered the record.
                filled = planning_phases.fill_output(
                    phase.output, runs, record=record, record_context=record_context,
                    plan_text=plan_text, earlier_runs=earlier, ended_on=ended_on, **names)
                facts = tagged_facts(phase.output, filled)
        except Exception as error:  # noqa: BLE001 - the phase fails, saying why
            from pydantic import ValidationError

            message = (str(error) if isinstance(error, (FactsError, ValidationError)) else
                       f"its command runs could not be read ({type(error).__name__}: {error})")
            state["missing_fact"] = isinstance(error, MissingFactError)
            if phase.fans_out:
                message = f"item {item}: {message}"
            tracing.end_span(self, span, status=tracing.STATUS_ERROR, attributes={
                **closing, "record": record, "record_issues": issues, "facts_error": message})
            return "failed", message
        if phase.output is not None:
            if phase.fans_out:
                state["outputs"][phase.name].append(facts)
            else:
                state["outputs"][phase.name] = facts
        if phase.fans_out and phase.until is not None:
            # The stop condition reads the items' outputs so far (typed) and earlier outputs.
            from fastworkflow.typed_phase_facts import validate_phase_output_value

            so_far = ([validate_phase_output_value(phase.output, v)
                       for v in state["outputs"][phase.name]]
                      if phase.output is not None else [])
            try:
                state["until_met"] = planning_phases.stops_after(
                    phase, so_far, self._phase_outputs(phases, index),
                    phase_names=phase_names)
            except planning_phases.PhaseError as error:
                message = f"item {item}: {error}"
                tracing.end_span(self, span, status=tracing.STATUS_ERROR, attributes={
                    **closing, "record": record, "record_issues": issues, "facts": plain_facts(facts),
                    "facts_error": message})
                return "failed", message
            closing["stop_condition_met"] = state["until_met"]
        tracing.end_span(self, span, attributes={
            **closing, "record": record, "record_issues": issues, "facts": plain_facts(facts)})
        if self.no_agent_mode and not answer_only:
            # The steps this phase ran, kept for the stop of a later phase (``done``).
            state.setdefault("done_steps", []).extend(
                self._phase_step_lines(phase, state, plan_text, names, ran_only=True))
            state.setdefault("step_results", {}).update(self._step_results(phase, state, plan_text, names, code))
        return "done", None

    def _phase_code_steps(self, plan_text: str, own_steps: int, names: dict, *,
                          answers: bool, resume: Optional[dict[str, Any]] = None):
        """The steps code runs at the start of this phase run (``planning_phases.CodeSteps``),
        or None: the run has no steps of its own (planner-written steps only), or nothing
        could be read (never a reason to fail the phase; the agent then runs every step)."""
        from fastworkflow import planning_phases

        if not own_steps or self._app_workflow is None:
            return None
        try:
            state = getattr(self, "_phase_run", None) or {}
            call_first = state.get("call_first_step", state.get("first_step", 0))
            return planning_phases.CodeSteps(
                plan_text, workflow=lambda: self.get_active_workflow() or self._app_workflow,
                runs=lambda index: self._archived_command_runs(from_step=index,
                                                                to_step=index),
                call_runs=lambda: self._archived_command_runs(from_step=call_first),
                own_steps=own_steps, answers=answers, planning=self._planning_commands(),
                resume=resume, **names)
        except Exception as error:  # noqa: BLE001 - the agent runs the steps instead
            logger.warning("code-run steps skipped: %s: %s", type(error).__name__, error)
            return None

    def _run_subtasks_from(self, start: int):
        """Run sub-tasks start.. one after another; the combined answer, or a suspension.

        Each sub-task's ReAct steps continue numbering from the previous one so
        archive aliases stay unique across the turn. A run that hits the step cap
        is continued by a fresh executor, inserted as the next entry (see _hand_off).
        """
        k = start
        # A while loop: a hand-off inserts an entry after the current one, so the end moves.
        while k < len(self._subtasks):
            self._subtask_index = k
            origin = self._origins[k]
            # A hand-off continuation keeps the stop/abort decision of the run before it.
            if k == 0 or self._origins[k - 1] != origin:
                self._enable_all_tools()
            query = self._subtasks[k]
            floors = getattr(self, "_evidence_floors", None)
            if not isinstance(floors, dict):
                floors = self._evidence_floors = {}
            floors.setdefault(origin, self._next_step)
            first = k == 0 or self._origins[k - 1] != origin
            outputs_before = len(self._turn_outputs)
            if first:
                self._part_outputs_start = outputs_before
            result = self._run_entry(k, origin, query, first=first, floor=floors[origin])
            if getattr(result, "suspended", None) is True:
                return result
            self._record_subtask_answer(result, self._typed_output_since(outputs_before))
            # A phased planning command is never handed off: a phase that ran out of steps ends its part as it is.
            if getattr(result, "phase_exhausted", False) or (
                    getattr(result, "exhausted", None) is True and not self._hand_off(k)):
                self._note_partial_read(origin)
            k += 1

        combined = self._combined_answer()
        # Read now: the turn frame is gone by the time the answer is finalized.
        self._turn_parts_of_answer = (combined, self._answer_parts(combined))
        return dspy.Prediction(final_answer=combined, exhausted=self._subtask_exhausted)

    def _bound_item(self, origin: int):
        """The reviewed sub-task (SubTask: goal, inputs, outcome, means) that origin *origin* runs, or None when the turn
        has no plan for it (a turn that was not split under a plan, or a fast-path turn)."""
        plan = self._turn_subtask_plan
        if plan is None or not 0 <= origin < len(plan.items):
            return None
        return plan.items[origin]

    def _running_part_item(self):
        """The plan item (SubTask) of the turn's part that is running now, or None when no part of a sub-task plan runs.
        Read from the TURN frame, so a step of a planning command the part called (a frame above it) finds the part."""
        turn = self._turn_frame()
        if not 0 <= turn.subtask_index < len(turn.origins):
            return None
        return self._bound_item(turn.origins[turn.subtask_index])

    def _answers_by_number(self) -> dict[int, str]:
        """Each sub-task that has run, by its 1-based number, with the answer its run gave (the last of its runs).

        The hand-off's 'from sub-task N' inputs take these: the run's typed Output (JSON) when it produced one, else the
        text its run returned (the command's answer for a planning command, the agent's final answer otherwise)."""
        answers: dict[int, str] = {}
        typed = list(self._subtask_outputs)
        typed += [""] * (len(self._subtask_answers) - len(typed))
        for origin, answer, output in zip(self._origins, self._subtask_answers, typed):
            answers[origin + 1] = output or answer
        return answers

    def _typed_output_since(self, start: int) -> str:
        """The typed Output of the last successful command run since turn output *start*, as JSON; "" when none."""
        for command_output in reversed(self._turn_outputs[start:]):
            response = command_output.command_response
            if response.success and response.output is not None:
                dump = getattr(response.output, "model_dump_json", None)
                return dump() if callable(dump) else json.dumps(response.output, default=str)
        return ""

    def _run_entry(self, k: int, origin: int, query: str, *, first: bool, floor: int):
        """Run entry *k* of the turn: the bound means of its goal when the turn has a plan, else the agent as today.

        A planning means runs its phases with its inputs (through the phase runner), its first entry only. A plain
        means and a from-scratch goal run the agent with the goal's text (its inputs and desired outcome in it); a plain
        means narrows the executor's listing to its command, and the plan of the run names that command, so the
        finish check holds the goal to it. Nothing here marks a goal complete: the finish check does (arch 19.0 inv. 3).
        """
        from fastworkflow import review_proposals

        item = self._bound_item(origin)
        if item is None:
            return self._call_agent_for_query(query, query, first_step=self._next_step,
                                              template_text=self._original_text(origin), evidence_floor=floor)
        resolved = review_proposals.resolve_inputs(item, self._answers_by_number())
        if resolved.missing:
            numbers = ", ".join(str(number) for number in resolved.missing)
            return dspy.Prediction(final_answer=f"This sub-task needs the output of sub-task {numbers}, which has not run.",
                                   exhausted=False)
        if first:
            query = review_proposals.run_text(item, resolved)
        else:
            query = self._continued_text(query, item, resolved)
        if first and item.means.kind == review_proposals.MEANS_PLANNING:
            return self._run_planning_means(item, query, origin, floor=floor)
        if first and item.means.kind == review_proposals.MEANS_PLAIN and (
                review_proposals.drafting_on() or self.no_agent_mode):
            return self._run_plain_means(item, query, origin, floor=floor)
        bound = item.means.name if item.means.kind == review_proposals.MEANS_PLAIN else None
        self._narrow_to = bound
        try:
            return self._call_agent_for_query(
                query, query, first_step=self._next_step, template_text=self._original_text(origin),
                evidence_floor=floor, bound_command=bound)
        finally:
            self._narrow_to = None

    @staticmethod
    def _continued_text(text: str, item, resolved) -> str:
        """A hand-off continuation's text with the sub-task's inputs and outcome in its first line."""
        from fastworkflow import review_proposals

        head, separator, tail = text.partition("\n")
        full = review_proposals.run_text(item, resolved)
        suffix = full[len(item.goal):]
        return f"{head}{suffix}{separator}{tail}" if suffix else text

    def _run_planning_means(self, item, query: str, origin: int, *, floor: int):
        """A sub-task bound to a planning command: its phases run with its inputs (the phase runner, as a planned call's
        phases do; no model plans them). Its arguments are checked as a planned command's are (check_planned_arguments);
        a refusal, or no command, falls back to the agent with the refusal as a note and the bound command's listing.
        With drafting on (fix-48np.1) the reviewed inputs are the arguments (review_proposals.reviewed_arguments), and
        the run records whether they ran as shown; a refusal still falls back to the agent, recorded as a deviation."""
        from fastworkflow import planning_commands, review_proposals, run_listing

        folder = self._app_workflow.folderpath
        name = item.means.name
        command = planning_commands.planning_command(folder, name)
        presets = review_proposals.drafting_on()
        refusal = None
        reviewed = None
        if command is None:
            refusal = f"{name} could not be loaded"
        else:
            resolved = review_proposals.resolve_inputs(item, self._answers_by_number())
            # Its validators read the record it will run on, as at the gate and on the "/" path.
            with intent_loop.on_owner_record(self._app_workflow, name):
                if presets:
                    reviewed, checked = review_proposals.reviewed_arguments(
                        self._app_workflow, command, item, resolved, query,
                        earlier=list(self._turn_subtask_plan.items[:origin]), answers=self._answers_by_number(),
                        earlier_outputs=tuple(self._subtask_answers),
                        planner_lm=getattr(self, "_current_planner_lm", None))
                else:
                    checked = run_listing.check_planned_arguments(command, dict(resolved.named), query,
                                                                  earlier_outputs=list(self._subtask_answers))
            refusal = checked.refusal
            if refusal is None:
                try:
                    phases = planning_commands.call_phases(command, checked.args)
                except planning_commands.PlanningCommandCallError as error:
                    refusal = str(error)
        if presets:
            self._turn_subtask_bindings.append(review_proposals.binding_record(
                origin + 1, item, reviewed, None if refusal is not None else checked.args,
                f"refused, the agent ran it instead: {refusal}" if refusal is not None else None))
        if refusal is not None and self.no_agent_mode:
            # The part was resolved before the review (its missing values asked for then); a refusal now is not asked
            # again: the part ends as not run, naming every argument still missing or refused (the checks stop at the
            # first), from the values the review bound.
            checked_text = "\n\n".join([query, *self._subtask_answers])
            with intent_loop.on_owner_record(self._app_workflow, name):
                gaps = (planning_commands.argument_gaps(
                    command, dict(reviewed.values) if reviewed is not None else dict(resolved.named),
                    checked_text) if command is not None else None)
            return self._not_run_part(
                command=name, means=review_proposals.MEANS_PLANNING,
                reason=f"it cannot run as planned ({gaps.describe() if gaps is not None and gaps.names else refusal}), "
                       "and the agent model that would take over is not set (LLM_AGENT)")
        if refusal is not None:
            return self._call_agent_for_query(
                f"{query}\n\nThe planning command {name} cannot run as planned: {refusal}.",
                query, first_step=self._next_step, template_text=self._original_text(origin),
                evidence_floor=floor, bound_command=name)
        call = planning_commands.PlanningCommandCallPlan(
            plan="", args=checked.args, notes=list(checked.notes), prediction=None, phases=phases)
        return self._start_planning_command_phases(query, command, call, first_step=self._next_step)

    def _run_plain_means(self, item, query: str, origin: int, *, floor: int):
        """F1 (fix-48np.1, drafting on): a sub-task bound to an execution command runs that command with its reviewed
        inputs. Code runs it as the plan's one step (``planning_phases.CodeSteps``: the agent's own command tool, so the
        CME parameter path and its validation, as a planning command's code steps run), going to the command's context
        first when code finds a path; the agent then answers from its output. The agent writes no command text for it.

        When a required input is unbound (review_proposals.reviewed_values), the agent handles the sub-task as before
        (F2). When the CME refuses the values, or code cannot run the step, the agent takes over with the refusal in its
        trajectory (F2). Either is recorded as a deviation (review_proposals.binding_record)."""
        from fastworkflow import ModuleType, review_proposals, run_listing
        from fastworkflow.planning import run as run_step

        name = item.means.name
        params_class = fastworkflow.RoutingRegistry.get_definition(self._app_workflow.folderpath).get_command_class(
            name, ModuleType.COMMAND_PARAMETERS_CLASS)
        required = [field for field, info in (params_class.model_fields.items() if params_class else ())
                    if info.is_required()]
        resolved = review_proposals.resolve_inputs(item, self._answers_by_number())
        with intent_loop.on_owner_record(self._app_workflow, name):   # a run-time draft checks on the record it runs on
            reviewed = review_proposals.reviewed_values(
                self._app_workflow, item, resolved, required, earlier=list(self._turn_subtask_plan.items[:origin]),
                answers=self._answers_by_number(), planner_lm=getattr(self, "_current_planner_lm", None))
        self._narrow_to = name
        try:
            if reviewed.unbound and self.no_agent_mode:
                self._turn_subtask_bindings.append(review_proposals.binding_record(
                    origin + 1, item, reviewed, None,
                    f"unbound {', '.join(reviewed.unbound)}, not run: no agent model is set"))
                return self._not_run_part(
                    command=name, means=review_proposals.MEANS_PLAIN,
                    reason=f"its required value(s) {', '.join(reviewed.unbound)} are not given, and no agent model "
                           "is set to find them (LLM_AGENT)",
                    suggested_plan=no_agent_mode.numbered(
                        [f"Find the value(s) {', '.join(reviewed.unbound)} and run {name} with them."]))
            if reviewed.unbound:
                self._turn_subtask_bindings.append(review_proposals.binding_record(
                    origin + 1, item, reviewed, None,
                    f"unbound {', '.join(reviewed.unbound)}, the agent ran it instead"))
                return self._call_agent_for_query(
                    query, query, first_step=self._next_step, template_text=self._original_text(origin),
                    evidence_floor=floor, bound_command=name)
            plan_text = f"1. {run_step(name.rsplit('/', 1)[-1], **reviewed.values)}"
            outputs_before = len(self._turn_outputs)
            self._initial_plan_text = plan_text
            self._attach_finish_check(floor, values=True)
            code = self._phase_code_steps(plan_text, 1, self._phase_command_names(), answers=True)
            if code is not None:
                code.done_inputs = {"user_query": (
                    f"{query}\n\nThis step has run, with the reviewed inputs; answer from its output:\n{plan_text}")}
            agent = self._workflow_tool_agent
            listing = run_listing.bound_command_listing(self._app_workflow, name)

            def run_bound():
                agent.code_steps = code
                return agent(user_query=f"{query}\n\nExecute these next steps:\n{plan_text}",
                             available_commands=listing, first_step=self._next_step)

            result = self._call_agent_with_retry(run_bound, trace_input=plan_text)
            ran = None
            for output in self._turn_outputs[outputs_before:]:
                if output.command_name == name and output.command_response.success:
                    ran = _planning_args(output.command_parameters)
                    break
            stop = getattr(code, "stop", None) if code is not None else "no code steps"
            if self.no_agent_mode:
                # No agent step writes the answer: the command's own response is the sub-task's answer.
                self._turn_subtask_bindings.append(review_proposals.binding_record(
                    origin + 1, item, reviewed, ran,
                    None if ran is not None else f"code did not run it ({stop}), not run: no agent model is set"))
                if ran is None:
                    # The command's own refusal (``fastworkflow.Refused``) or failure text is the answer to "why": the `/`
                    # path shows it, so the not-run entry carries it too.
                    own = next((output.command_response.response for output in self._turn_outputs[outputs_before:]
                                if output.command_name == name and not output.command_response.success
                                and str(output.command_response.response or "").strip()), None)
                    why = f"{stop}: {str(own).strip()}" if own else stop
                    return self._not_run_part(
                        command=name, means=review_proposals.MEANS_PLAIN,
                        reason=f"code could not run it ({why}), and no agent model is set to take over (LLM_AGENT)",
                        suggested_plan=plan_text)
                text = next(output.command_response.response for output in self._turn_outputs[outputs_before:]
                            if output.command_name == name and output.command_response.success)
                # The part ran bound, so the answer says which command it ran (the client sees only the text).
                return dspy.Prediction(final_answer=no_agent_mode.ran_note(text, name), exhausted=False)
            deviation = None if ran is not None else f"code did not run it ({stop}), the agent ran it instead"
            self._turn_subtask_bindings.append(review_proposals.binding_record(
                origin + 1, item, reviewed, ran, deviation))
            return result
        finally:
            self._narrow_to = None

    def _original_text(self, origin: int) -> str:
        """The sub-task text an origin was split out as (its first entry, before any hand-off)."""
        return self._subtasks[self._origins.index(origin)]

    def _note_partial_read(self, origin: int) -> None:
        """A part that ran out of steps with no continuation left ends its answer with a code-written note for each
        paged read of its commands that stopped with rows unread and a reported total (``no_agent_mode.step_limit_note``),
        so a count or "none found" in the answer is not taken for the whole list. A part that did not run out of
        steps, or a read with no total, adds nothing."""
        from fastworkflow.observation_offloading.facts import PAGING_PARAMETERS
        from fastworkflow.planning_phases import _page_read

        try:
            floor = (getattr(self, "_evidence_floors", None) or {}).get(origin, 0)
            runs = self._archived_command_runs(from_step=floor)
        except Exception as error:  # noqa: BLE001 - the note is additive, never a reason to fail the turn
            logger.debug(f"partial-read note skipped ({error})")
            return
        reads: dict[tuple[str, str], dict[str, Any]] = {}
        for run in runs:
            page = _page_read(run, PAGING_PARAMETERS)
            if page is None:
                continue
            name, start, count, total = page
            rest = {key: value for key, value in (run.get("input") or {}).items() if key != name}
            read = reads.setdefault((run["command"], json.dumps(rest, sort_keys=True, default=str)),
                                    {"pages": {}, "rows": {}})
            read["pages"][start], read["total"] = count, total
            listed = next((v for v in run["output"].values() if isinstance(v, (list, tuple))), [])
            read["rows"][start] = list(listed)
        notes = []
        for (command, _), read in reads.items():
            rows_read = sum(read["pages"].values())
            if rows_read < read["total"]:
                rows = [row for _, page_rows in sorted(read["rows"].items()) for row in page_rows]
                notes.append(no_agent_mode.step_limit_note(command, {
                    "pages": len(read["pages"]), "rows_read": rows_read, "total": read["total"], "rows": rows}))
        if notes and self._subtask_answers:
            self._subtask_answers[-1] = f"{self._subtask_answers[-1]}\n\n" + "\n".join(notes)

    def _hand_off(self, k: int) -> bool:
        """Queue a fresh-executor continuation of sub-task k, which hit the step cap; False when none is left.

        The progress check reads the agent's current trajectory, so it runs before
        the next run replaces that trajectory. At most MAX_HANDOFFS continuations per sub-task.
        """
        turn = self._turn_frame()
        origin = turn.origins[k]
        if turn.origins.count(origin) > MAX_HANDOFFS:
            # No continuation left: the capped answer stands as partial and the turn is exhausted.
            turn.subtask_exhausted = True
            return False
        from fastworkflow.workflow_agent import run_progress_check

        done = run_progress_check(self, "step_limit", done_only=True).strip()
        opened = _opened_part(getattr(turn.agent, "trajectory", None) or {})
        logger.info("sub-task %d hit the step limit; handing off to a fresh executor (hand-off %d)",
                    origin + 1, turn.origins.count(origin))
        turn.subtasks.insert(k + 1, (
            f"{self._original_text(origin)}\n\n"
            "Already done in this sub-task (do not repeat it; observation ids such as O12 can be looked up):\n"
            f"{done}"
            + (f"\n\n{opened}" if opened else "")))
        turn.origins.insert(k + 1, origin)
        # This capped run's answer is kept: _combined_answer joins it with the continuation's.
        return True

    def _combined_answer(self) -> str:
        """Answers grouped under their original sub-task; a single-origin turn is its answers with no heading.

        A sub-task that was handed off has one answer per run, kept in order and labelled as parts.
        """
        answers: dict[int, list[str]] = {}
        for origin, answer in zip(self._origins, self._subtask_answers):
            if answer:
                answers.setdefault(origin, []).append(answer)
        joined = {origin: _join_parts(parts) for origin, parts in answers.items()}
        if len(joined) == 1:
            body = next(iter(joined.values()))
        else:
            body = "\n\n".join(
                f"### {origin + 1}. {_heading_title(self._original_text(origin))}\n{text}"
                for origin, text in joined.items()
            )
        return body

    def _answer_parts(self, answer: str) -> list[dict[str, Any]]:
        """The parts of a split turn as data, beside the combined answer text (which they never change).

        One entry per sub-task in order, the answered ones and the ones that gave no answer (those have no heading):
        ``number`` (1-based, the answer's ``### N.`` number and ``CommandOutput.sub_task``), ``text`` (the sub-task's
        request, uncut), ``title`` (the heading's text), ``answered``, and ``start``/``end``, the offsets (in characters) of the
        part's answer in *answer*, the text after its heading; ``None`` for a part with no answer; and, only for a part
        continued after a step-cap hand-off, ``runs``: ``{index, of, start, end}`` for each run's answer, the text after
        its ``Part i of n:`` label (same offsets). Offsets rather than
        the answers, so a long answer is not stored twice (an artifact over the store's inline limit is offloaded).

        ``[]`` for a turn of fewer than two sub-tasks, and when *answer* is not the combined text these parts were
        read from (it was cleaned or replaced since): the offsets would point at the wrong text."""
        origins = list(dict.fromkeys(self._origins))
        if len(origins) < 2:
            return []
        answers: dict[int, list[str]] = {}
        for origin, part in zip(self._origins, self._subtask_answers):
            if part:
                answers.setdefault(origin, []).append(part)
        joined = {origin: _join_parts(texts) for origin, texts in answers.items()}
        if self._combined_answer() != answer:
            return []
        spans: dict[int, tuple[int, int]] = {}
        if len(joined) == 1:
            origin, text = next(iter(joined.items()))
            spans[origin] = (0, len(text))
        else:
            position = 0
            for origin, text in joined.items():
                position += len(f"### {origin + 1}. {_heading_title(self._original_text(origin))}\n")
                spans[origin] = (position, position + len(text))
                position += len(text) + 2
        not_run_numbers = {stop.get("sub_task") for stop in self._turn_not_run if isinstance(stop, dict)}
        entries = []
        for origin in origins:
            entry = {"number": origin + 1, "text": self._original_text(origin),
                     "title": _heading_title(self._original_text(origin)), "answered": origin in spans,
                     "start": spans[origin][0] if origin in spans else None,
                     "end": spans[origin][1] if origin in spans else None}
            if origin + 1 in not_run_numbers:
                # No-agent mode: the text under this part's heading is the code-written explanation of a stop
                # (``fw_not_run``), not an answer a command produced. ``answered`` still means "has text".
                entry["not_run"] = True
            if origin in spans and (runs := _part_runs(answers[origin])):
                # A part continued after a step-cap hand-off: where each run's answer is, beside the "Part i of n:"
                # labels in its text. Absent for a part that was not continued.
                entry["runs"] = [{**run, "start": run["start"] + spans[origin][0], "end": run["end"] + spans[origin][0]}
                                 for run in runs]
            entries.append(entry)
        return entries

    def _tag_turn_parts(self, answer: str) -> None:
        """Carry the parts of a split turn (``_answer_parts``) to the client: in the artifacts of the turn's last command
        output, the entry ``TurnOutput.command_outputs`` ends with (as ``_describe_pending_prompt`` tags a prompt's
        entry). The final answer's own response is not in ``command_outputs``, and ``TurnOutput`` gains no field. The answer
        text is untouched. Best effort: a turn with no command output carries nothing, and a client reads the headings."""
        try:
            recorded = getattr(self, "_turn_parts_of_answer", None)
            if self._turn_outputs and recorded is not None and recorded[0] == answer and recorded[1]:
                self._turn_outputs[-1].command_response.artifacts[TURN_PARTS_ARTIFACT] = recorded[1]
        except Exception as exc:  # noqa: BLE001 - never raises into the turn
            logger.debug(f"turn parts not recorded: {type(exc).__name__}: {exc}")

    def _tag_not_run(self) -> None:
        """Carry the turn's stops (no-agent mode: a planning command that stopped, a part not run here) to the client:
        the list ``fw_not_run`` in the artifacts of the turn's last command output, as ``_tag_turn_parts`` does for
        the parts. A turn that ran no command has none, so a command output of its own carries them. Best effort."""
        entries = list(self._turn_not_run)
        if not entries:
            return
        try:
            if not self._turn_outputs:
                self.append_turn_output(fastworkflow.CommandOutput(
                    command_response=fastworkflow.CommandResponse(
                        response="\n".join(no_agent_mode.explanation(entry) for entry in entries)),
                    command_name=entries[0].get("command") or "",
                    workflow_name=self._app_workflow.folderpath.split("/")[-1] if self._app_workflow else "",
                    # Plumbing, not a result: the chat page leaves it out of its cards and results panel.
                    framework=True,
                ))
            self._turn_outputs[-1].command_response.artifacts[NOT_RUN_ARTIFACT] = entries
        except Exception as exc:  # noqa: BLE001 - never raises into the turn
            logger.warning(f"turn stops not recorded: {type(exc).__name__}: {exc}")

    def _record_subtask_answer(self, result, typed: str = "") -> None:
        outputs = self._subtask_outputs
        # Padded first: a blob written before the typed outputs were kept has answers and no outputs.
        outputs.extend([""] * (len(self._subtask_answers) - len(outputs)))
        self._subtask_answers.append(result.final_answer)
        outputs.append(typed)
        trajectory = getattr(self._workflow_tool_agent, "trajectory", None) or {}
        steps = [int(key.rsplit("_", 1)[1]) for key in trajectory
                 if key.rsplit("_", 1)[-1].isdigit()]
        if steps:
            # Never backwards: a nested frame's steps (run_planning_command) may be past it.
            self._next_step = max(self._next_step, max(steps) + 1)

    def _call_agent_resume(self, observation: str, *, completes_step: bool = False):
        def resume():
            # A plain answer is resumed as it always was; the flag is passed only for a nested run.
            if completes_step:
                return self._workflow_tool_agent.resume(observation, completes_step=True)
            return self._workflow_tool_agent.resume(observation)

        return self._call_agent_with_retry(resume, trace_input=observation, resumed=True)

    def _resume_top_frame(self, observation: str, *, completes_step: bool = False):
        """Resume the top frame's suspended run with *observation*: its phase run, or its agent.

        *completes_step*: the observation is the run of a nested command the step waited on,
        not a user's answer (``fastWorkflowReAct.resume``)."""
        if self._phase_run is not None:
            # A phase of a phased planning command call was suspended: resume it, then run the rest.
            return self._run_planning_command_phases(resume_observation=observation,
                                          resume_completes_step=completes_step)
        return self._call_agent_resume(observation, completes_step=completes_step)

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
        if "[[" in result_text:
            # A DSPy field marker ("[[ ## completed ## ]]") a reply carried into its text.
            from fastworkflow.planning_phases import clean_answer
            result_text = clean_answer(result_text)

        command_response = fastworkflow.CommandResponse(response=result_text)

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
        # An ask_user entry's prompt-description keys (_describe_pending_prompt) describe its own, now
        # answered, prompt: they are withheld from the answer. Any other artifact still flows.
        if artifact_responses := fastworkflow.turn.collect_artifact_responses(
            [self._without_prompt_description(o) for o in self._turn_outputs]
        ):
            fastworkflow.turn.merge_artifact_responses_into(
                command_response, artifact_responses
            )
        self._tag_turn_parts(result_text)
        self._tag_not_run()
        if (capabilities := getattr(agent_result, "capabilities", None)) is not None:
            # A capability answer written in code (capability_answer): its listing as data, beside the unchanged text.
            command_response.artifacts["capabilities"] = capabilities
            if not self._turn_outputs:
                # The final answer's own output is not in TurnOutput.command_outputs, so an HTTP or MCP client would
                # never see the data: the turn carries it on a framework output of its own (not drawn as a result).
                self.append_turn_output(fastworkflow.CommandOutput(
                    command_response=fastworkflow.CommandResponse(
                        response=result_text, artifacts={"capabilities": capabilities}),
                    workflow_name=self._app_workflow.folderpath.split("/")[-1] if self._app_workflow else "",
                    framework=True))

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
            self._maybe_enqueue_output(result.command_output)
            self._maybe_enqueue_trace_sentinel()
            return result.command_output

        agent_result = self._run_agent(message)
        return self._settle_agent_result(message, agent_result)

    def _settle_agent_result(self, message: str, agent_result) -> fastworkflow.CommandOutput:
        """The turn's output for one agent run: its suspension (the turn awaits the user), or its answer."""
        self._turn_agent_result = agent_result
        if getattr(agent_result, "suspended", None) is True:
            self._ask_agent_question(agent_result.clarification, message)
            self._note_agent_suspension(agent_result.clarification)
            return self._awaiting_user_output(agent_result.clarification)
        self._reset_agent_suspension()
        return self._finalize_agent_output(message, agent_result)

    def _approval_state_payload(self) -> Optional[dict]:
        store = approval_binding.store_for(self._app_workflow)
        if store is None or not store.records and not store.turn_seq:
            return None
        return store.to_state()

    def _note_approval_inbound(self, surface, token: Optional[str], unattended: bool) -> None:
        """Record the surface, token and turn kind the approval gate reads for this message."""
        store = approval_binding.store_for(self._app_workflow)
        if store is None:
            return
        store.surface = surface or approval_binding.CHANNEL_SURFACE
        store.inbound_token = token
        store.channel = str(self._channel_id or "")
        store.task = str(self._session_key or self._channel_id or "")

    def _resume_approval_direct(self, message: str) -> fastworkflow.CommandOutput:
        """The reply to an approval raised outside the agent's loop (an MCP tool call), or a
        /cancel_pending for a suspended one: the record resolves here, and its outcome is this
        turn's output."""
        held = self._waiting
        was_suspended = held is not None and held.suspended_agent
        resolution = approval_binding.resolve_reply(self, self._app_workflow, message)
        if not resolution.done:
            # A reply that was not understood: the question is asked again, and the next message answers it.
            if held is not None and held.store_backed:
                held.payload[waiting_module.ASKED_AGAIN] = True
            asked_again = self._awaiting_user_output(resolution.text)
            if not was_suspended:
                # A question raised outside the agent's loop (a "/" command) has no ask_user entry to carry its description:
                # the re-ask is an output of the turn like the first ask was, so it is tagged like it (``question_kind``,
                # ``approval``).
                self.append_turn_output(asked_again)
            return asked_again
        # A user-typed "/" command that was held for this approval left its assistant-mode flag set (the
        # held branch of invoke_command returns before it clears it). The approval is answered: the next
        # plain message belongs to the agent again.
        self.end_round()
        if was_suspended:
            self._reset_agent_suspension()
        output = resolution.output
        if output is None:
            output = fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(response=resolution.text))
        self.append_turn_output(output)
        return output

    def _approval_text_output(self, text: str) -> fastworkflow.CommandOutput:
        """A plain text output for the user's approval management messages (never agent-visible)."""
        output = fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(response=text))
        if self._app_workflow is not None:
            output.workflow_name = self._app_workflow.folderpath.split("/")[-1]
        self.append_turn_output(output)
        return output

    def cancel_pending_approvals(self) -> int:
        """Invalidate every awaiting approval on this session (session end, eviction, /cancel_pending).

        A suspended approval also abandons the agent's suspended tool call, so a later message is
        never taken as its answer. Returns how many records were cancelled.
        """
        if self._app_workflow is None:
            return 0
        held, suspended = self._waiting, self._awaiting_user
        cancelled = approval_binding.cancel_all(self, self._app_workflow, reason="cancelled")
        if cancelled:
            # A held "/" approval was dropped: its assistant-mode flag goes with it (see _resume_approval_direct).
            self.end_round()
        if held is not None and held.suspended_agent and suspended:
            self._reset_agent_suspension()
        return cancelled

    def _hold_resumed_approval_run(self, output: fastworkflow.CommandOutput) -> None:
        """Hold the run an approval reply executed under the step it suspended (fix-3rsq.approval-finish-check).

        The approved call runs through approval_binding._execute_stored (perform_action), not the
        agent's execute path, so nothing holds its run for the step's archive row. Without that row
        the finish check reads the step as not run and sends the agent back to run it again. The
        step is the suspended agent's in-flight execute step: the nested frame's agent when the
        suspension is inside a planning command, else the turn's agent. Never raises.
        """
        from fastworkflow.workflow_agent import _held_context, _hold_command_run

        try:
            agent = self._top_frame().agent or self._workflow_tool_agent
            alias = current_execute_alias(agent)
            if alias is None:
                return
            clause, name = _held_context(self, output, *self._current_record())
            _hold_command_run(self, output, clause, name, alias=alias)
        except Exception:  # noqa: BLE001 - holding a record must not fail the approved command
            logger.debug("could not hold the approved run for its archive row", exc_info=True)

    def _resume_agent_message(self, user_answer: str) -> fastworkflow.CommandOutput:
        self._ensure_agent_initialized()
        held = self._waiting
        reader = held.reader if held is not None else waiting_module.READER_AGENT      # the kind table (``waiting.step``)
        if reader == waiting_module.READER_RESOLUTION:
            return self._resume_plan_resolution(user_answer)
        # Catch awaiting_user/_suspended desync before any LLM work: resume()
        # consumes _suspended at entry, so a second request (or a restore that
        # lost the react blob) must not become a bare 500.
        agent = self._workflow_tool_agent
        if agent is None or agent.export_suspended() is None:
            raise NoSuspendedAgentStateError(
                "No suspended ReAct state to resume"
            )
        # What the reply is about, read before it changes anything: the turn's message, the question, the store's record
        # (a reply that ends it ends the turn's record with it) and whether it is the sub-task gate's review.
        original_message, question, gate_review = self._suspended_message(), held.question if held else "", self._gate_review_out()
        awaiting = (approval_binding.store_for(self._app_workflow).get(held.payload["record_id"])
                    if reader == waiting_module.READER_APPROVAL else None)
        approval_resumed = awaiting is not None
        if approval_resumed:
            # The suspended tool call is an approval (ADR-EXP-025): the reply resolves it, and
            # the resolution is the observation. Nothing else reads the reply.
            resolution = approval_binding.resolve_reply(self, self._app_workflow, user_answer)
            if not resolution.done:
                self._ask_user_rounds.append({"reply": user_answer})
                return self._awaiting_user_output(resolution.text)
            self._note_agent_resume()
            self.complete_ask_user_entry(user_answer)
            if resolution.output is None and gate_review:
                # Declined or expired: the gate's review ends, and nothing runs (fix-3rsq.plan-review.2.6).
                return self._end_subtask_review_unrun(awaiting.state)
            if resolution.output is not None:
                self.append_turn_output(resolution.output)
                self._hold_resumed_approval_run(resolution.output)
            self._workflow_tool_agent.iteration_counter = -1
            observation = resolution.text
        elif reader == waiting_module.READER_DECISION:
            # A planning command paused at a step the client decides (no-agent mode): the reply itself is the observation,
            # read by the phase (_read_decision_reply). No planner call and no reply-intent call: yes, no, a value or
            # cancel are read in code.
            self._note_agent_resume()
            from fastworkflow.workflow_agent import _append_action_record

            _append_action_record(self, {"agent_query": question, "user_response": user_answer})
            self.complete_ask_user_entry(user_answer)
            self._workflow_tool_agent.iteration_counter = -1
            observation = user_answer
        else:
            self._note_agent_resume()

            from fastworkflow.workflow_agent import _post_ask_user_response

            self._workflow_tool_agent.iteration_counter = -1
            observation = _post_ask_user_response(question, user_answer, self)
        # An approval's observation is the approved command's own run: the step completes as one
        # that returned normally, so its completion hook archives the run held above.
        agent_result = self._resume_top_frame(observation, completes_step=approval_resumed)
        # The suspension may be nested: each call that finishes hands its run to the step it
        # was called from, which resumes next, until the turn frame is on top again.
        while len(self._frame_stack()) > 1 and getattr(agent_result, "suspended", None) is not True:
            if len(self._frame_stack()) == 2 and self._is_framework_command(self._top_frame().command_name or ""):
                if gate_review:
                    # The gate's review was accepted: the turn runs its sub-tasks (fix-3rsq.plan-review.2.6).
                    return self._settle_agent_result(self._turn_user_message, self._accept_subtask_review(agent_result))
                return self._finish_framework_call(agent_result)
            observation = self._finish_nested_frame(agent_result)
            agent_result = self._resume_top_frame(observation, completes_step=True)
        if self._subtasks and getattr(agent_result, "suspended", None) is not True:
            self._record_subtask_answer(agent_result)
            if getattr(agent_result, "exhausted", None) is True:
                self._hand_off(self._subtask_index)
            agent_result = self._run_subtasks_from(self._subtask_index + 1)
        self._turn_agent_result = agent_result
        if getattr(agent_result, "suspended", None) is True:
            self._ask_agent_question(agent_result.clarification, original_message)
            self._note_agent_suspension(agent_result.clarification)
            return self._awaiting_user_output(agent_result.clarification)

        self._reset_agent_suspension()
        return self._finalize_agent_output(original_message, agent_result)

    # ------------------------------------------------------------------
    # Shared capture for the fw.agent.tool_call emission sites
    # ------------------------------------------------------------------
    #
    # _process_message and _process_action both open fw.agent.tool_call and both
    # owe §12.1.1's shared capture, so the projection lives here once rather than
    # being written twice and drifting. Everything below is additive recording:
    # no fastWorkflow control flow reads a recorded context type or a
    # consequence class, which arch §17.3 requires.
    #
    # "Here" is in fact `tracing`, because workflow_agent.py
    # opens the same span from a third site and owes the same record. These two
    # methods stay as the WEC-shaped entry points — they supply the app-workflow
    # fallback that the free functions cannot know about — but the projection
    # itself is written once, for all three sites.

    def _context_before(
        self, span, workflow: Optional[fastworkflow.Workflow] = None
    ) -> Optional[str]:
        """The active context type before a command runs, or None.

        Gated on a span having actually opened, matching the existing
        attribute-prep rule at this seam: with observability off this must cost
        nothing.
        """
        return tracing.context_before(span, workflow or self._app_workflow)

    def _capture_attributes(
        self,
        span,
        command_output: fastworkflow.CommandOutput,
        context_before: Optional[str],
        workflow: Optional[fastworkflow.Workflow] = None,
        command_name: Optional[str] = None,
    ) -> dict:
        """Call-id, context type before/after and consequence for one command.

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

    @staticmethod
    def _failed_output_of_raising_command(exc: BaseException) -> Optional[fastworkflow.CommandOutput]:
        """The failed output of a command whose own code raised on the "/" path, or None when *exc* is not that (a control
        signal, an error of the framework itself). Its message names the command and the error as a planning command's
        failing ``__result__`` does (``planning_phases.run_call_result``), with ``error_type`` and ``error_message``; the
        traceback is logged, never put in the message. The turn is COMPLETED; this part's output says it failed."""
        if not isinstance(exc, Exception) or not getattr(exc, "_fw_raised_by_command", False):
            return None
        command_name = getattr(exc, "_fw_command_name", None) or "the command"
        logger.error("command %r raised %s: %s; the turn goes on with the command's failed output",
                     command_name, type(exc).__name__, exc, exc_info=exc)
        output = fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(
                response=f"Execution error in {command_name}: {exc!r}"[:500], success=False,
                artifacts={"error_type": type(exc).__name__, "error_message": str(exc)[:1000],
                           "traceback": "".join(traceback.format_exception(exc))[:4000]}),
            command_name=command_name)
        output.workflow_name = getattr(exc, "_fw_workflow_name", None) or ""
        output.context = getattr(exc, "_fw_context", None) or ""
        return output

    def _process_message(self, message: str) -> fastworkflow.CommandOutput:
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
            if (failed := self._failed_output_of_raising_command(exc)) is not None:
                # One success rule (both paths): a turn that ran to its end is COMPLETED, and the part's own output says
                # it failed. The traceback went to the log and the dispatch span (``command_executor.invoke_command``).
                command_output = failed
                self.end_round()
            else:
                # One arm, not two: a separate `except CommandCancelledError` left
                # AskUserSuspend — the other control signal — falling through to the
                # BaseException arm below and closing as STATUS_ERROR. fix-ajv.19.
                tracing.end_span(
                    self,
                    span,
                    status=tracing.status_for_dispatch_exception(exc),
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
            tracing.end_span(
                self,
                span,
                status=tracing.status_for_dispatch_exception(exc),
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

    @staticmethod
    def _conversation_context(conversation_history: dspy.History) -> str:
        """The last five turns' summaries, one ``key: value`` line each; "" with no earlier turn."""
        messages = []
        for conv_dict in conversation_history.messages[-5:]:
            messages.extend(
                f"{k}: {v}"
                for k, v in conv_dict.items()
                if k != "conversation_traces"
            )
        return "\n".join(messages)

    def _refine_user_query(
        self, user_query: str, conversation_history: dspy.History
    ) -> str:
        if not conversation_history.messages:
            return user_query
        context = self._conversation_context(conversation_history)
        return f"{context}\nnew_user_query: {user_query}" if context else f"new_user_query: {user_query}"

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

        from fastworkflow.conversation_summary import bounded_summary_inputs

        inputs = bounded_summary_inputs(user_query, workflow_actions, final_agent_response)
        try:
            planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
            with dspy.context(lm=planner_lm):
                prediction = dspy.ChainOfThought(ConversationSummarySignature)(**inputs)
            summary = prediction.conversation_summary
        except Exception as exc:
            # Conversation memory is ancillary: failure must not discard the
            # executor's completed answer. Preserve raw evidence in traces.
            logger.warning("Conversation summary failed; using bounded fallback (%s)",
                           type(exc).__name__)
            conversation_traces["summary_fallback_error_type"] = type(exc).__name__
            summary = ("Conversation summary unavailable. User request excerpt: "
                       + inputs["user_query"] + "\nAgent response excerpt: "
                       + inputs["final_agent_response"])
        return summary, json.dumps(conversation_traces)
