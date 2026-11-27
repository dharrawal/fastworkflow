"""What a turn is waiting for: one record type, one kind table, one transition function (bd fix-fpyb.8,
``docs/pending_state_design.md``).

A session has at most one question out, and it is ONE ``Waiting`` record in ``WorkflowExecutionContext._waiting``: the
agent's own question, the turn's which-command and values questions, the sub-task review and an approval (both point at
the approval store's record, which the store keeps), a paused decision step, and the deterministic path's open command.
``step`` is the one place that decides what a message, a cancel or a restore does to the record that is out (the
table of section 3.4 of the design note as code); the readers of each kind stay where they are.

No model imports here. ``Port`` and ``Holder`` (a record the CME's helpers read and write): the deterministic path's
question (kinds ``intent`` and ``parameters``) lives in ``WorkflowExecutionContext._waiting``, and the built-in commands
reach it through the port they find in their context under ``PORT_KEY``.
"""
from __future__ import annotations

import json
import weakref
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

KIND_INTENT = "intent"
KIND_PARAMETERS = "parameters"
KIND_REVIEW = "review"
KIND_APPROVAL = "approval"
KIND_DECISION = "decision"
KIND_AGENT = "agent"
KINDS = (KIND_INTENT, KIND_PARAMETERS, KIND_REVIEW, KIND_APPROVAL, KIND_DECISION, KIND_AGENT)

#: The kinds the deterministic path (the CME) holds, and so the only ones a CME port may drop.
CME_KINDS = (KIND_INTENT, KIND_PARAMETERS)

#: The key the CME's context holds its port under (``port_of``).
PORT_KEY = "waiting"
#: The flag a context with no execution context keeps in itself ("/" round); see ``Holder``.
ASSISTANT_KEY = "is_assistant_mode_command"

#: The payload key of a review record that holds its own slot question (a nested ``Waiting`` of kind ``parameters``, or None).
SLOT_KEY = "slot"
#: The payload key an approval record gets when a reply to it was not understood and it was asked again.
ASKED_AGAIN = "asked_again"
#: The payload key of a parked review: the approval store's record, parked with the turn's record (one thing).
STORE_RECORD_KEY = "store_record"

#: Who holds the question: the turn's plan resolution, or the CME (the deterministic path).
OWNER_PLAN = "plan"
OWNER_CME = "cme"


#: The reader a reply to a kind goes to (the per-kind work stays where it was; ``step`` only names it).
READER_RESOLUTION = "resolution"     # the plan path's which-command or values question (``resolution.take_reply``)
READER_APPROVAL = "approval"         # ``approval_binding.resolve_reply`` (an approval or a review)
READER_DECISION = "decision"         # a paused decision step (``WorkflowExecutionContext._read_decision_reply``)
READER_AGENT = "agent"               # the agent's own question (``workflow_agent._post_ask_user_response``)
READER_CME = "cme"                   # the deterministic path's open command (the CME's ``wildcard``)


class KindSpec(NamedTuple):
    """What a kind means to a client, to ``//cancel`` and to the dispatch.

    ``question_kind``: the value clients read (unchanged since before the table). ``cancel_text``: the reply that ends
    the question the way typing it does (None: not a reply, see ``cancel_label``); ``cme_cancel_text`` is the reply
    when the CME holds the question (its reader answers ``abort``). ``cancel_label``: what ``ChatSession`` reports
    (``"question"`` is a blocked ``ask_user`` read, ended by the queue's own cancel, not by a reply). ``reader``: where
    a reply goes (``cme_reader`` when the CME holds the question).
    """

    question_kind: str
    cancel_text: Optional[str]
    cancel_label: str
    cme_cancel_text: Optional[str] = None
    reader: str = ""
    cme_reader: Optional[str] = None


TABLE: dict[str, KindSpec] = {
    KIND_INTENT: KindSpec("plan_choice", "cancel", "clarification", "abort", READER_RESOLUTION, READER_CME),
    KIND_PARAMETERS: KindSpec("planning_parameter", "cancel", "clarification", "abort", READER_RESOLUTION, READER_CME),
    KIND_REVIEW: KindSpec("review", "no", "review", None, READER_APPROVAL),
    KIND_APPROVAL: KindSpec("approval", "no", "review", None, READER_APPROVAL),
    KIND_DECISION: KindSpec("phase_step", "cancel", "clarification", None, READER_DECISION),
    KIND_AGENT: KindSpec("agent", None, "question", None, READER_AGENT),
}


@dataclass
class Waiting:
    """The one question a turn is waiting on. ``payload`` points at the record that holds it today (live objects, not
    copies: the view is read-only); ``to_state`` is the plain-JSON form."""

    kind: str
    owner: str = OWNER_PLAN
    question: str = ""
    payload: dict = field(default_factory=dict)
    assistant: bool = False      # the round was started by a "/" message

    @property
    def spec(self) -> KindSpec:
        return TABLE[self.kind]

    @property
    def question_kind(self) -> str:
        return self.spec.question_kind

    @property
    def cancel_text(self) -> Optional[str]:
        """The reply that cancels this question (None when it is not ended by a reply)."""
        spec = self.spec
        return spec.cme_cancel_text if self.owner == OWNER_CME and spec.cme_cancel_text else spec.cancel_text

    @property
    def reader(self) -> str:
        """Where a reply to this question goes (``TABLE``)."""
        spec = self.spec
        return spec.cme_reader if self.owner == OWNER_CME and spec.cme_reader else spec.reader

    @property
    def store_backed(self) -> bool:
        """The question is a record of the approval store (an approval or a review): ``payload["record_id"]``."""
        return self.kind in (KIND_REVIEW, KIND_APPROVAL)

    @property
    def suspended_agent(self) -> bool:
        """The record is one the agent's own tool call suspended on (an approval raised in its loop, a review the framework
        call opened): the store's record says so (``ReviewRecord.suspended_agent``), copied into the payload."""
        return self.store_backed and bool(self.payload.get("suspended_agent"))

    @property
    def rides_suspension(self) -> bool:
        """Whether the turn is suspended on this question (``WorkflowExecutionContext._awaiting_user``): the next message
        continues the turn instead of beginning one. True for every question of the agent path; false for the CME's open
        command (a reply to it is a message of its own) and for an approval or a review opened outside the agent's loop (a
        "/" command or a tool call: the turn that raised it is over) until a reply to it was not understood (``ASKED_AGAIN``)."""
        if self.owner == OWNER_CME:
            return False
        if self.store_backed:
            return bool(self.payload.get("suspended_agent") or self.payload.get(ASKED_AGAIN))
        return True

    def to_state(self) -> dict:
        payload = {key: value.to_state() if isinstance(value, Waiting) else value for key, value in self.payload.items()}
        return json.loads(json.dumps({"kind": self.kind, "owner": self.owner, "question": self.question,
                                      "payload": payload, "assistant": self.assistant}, default=str))

    @classmethod
    def from_state(cls, state: Any) -> Optional["Waiting"]:
        if not isinstance(state, dict) or state.get("kind") not in TABLE:
            return None
        payload = dict(state.get("payload") or {})
        if state["kind"] == KIND_REVIEW:
            payload[SLOT_KEY] = cls.from_state(payload.get(SLOT_KEY))
        return cls(kind=state["kind"], owner=str(state.get("owner") or OWNER_PLAN),
                   question=str(state.get("question") or ""), payload=payload,
                   assistant=bool(state.get("assistant")))


# ---- the port the built-in commands reach the record through -------------------------------------------------------------

class Port:
    """How the CME's helpers read and write the turn's one waiting record (``intent_loop`` and
    ``parameter_clarification`` are the only callers). ``put`` replaces any record, so opening one closes the other;
    ``clear`` drops only the kinds the holder owns (``CME_KINDS``); ``assistant`` says whether the round was started by
    a "/" message and ``end_round`` ends that."""

    def get(self) -> Optional[Waiting]:
        raise NotImplementedError

    def put(self, record: Waiting) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    @property
    def assistant(self) -> bool:
        raise NotImplementedError

    def end_round(self) -> None:
        raise NotImplementedError


class Holder(Port):
    """The fallback port: keeps its record in itself, and the "/" flag in the context it sits in (``ASSISTANT_KEY``).
    ``port_of`` installs one in a CME context built without an execution context (unit tests do that)."""

    def __init__(self, context: Optional[dict] = None) -> None:
        self.record: Optional[Waiting] = None
        self._context = context if context is not None else {}

    def get(self) -> Optional[Waiting]:
        return self.record

    def put(self, record: Waiting) -> None:
        record.assistant = self.assistant
        self.record = record

    def clear(self) -> None:
        if self.record is not None and self.record.kind in CME_KINDS:
            self.record = None

    @property
    def assistant(self) -> bool:
        return ASSISTANT_KEY in self._context

    def end_round(self) -> None:
        self._context.pop(ASSISTANT_KEY, None)
        if self.record is not None:
            self.record.assistant = False


class OwnerPort(Port):
    """The execution context's port: reads and writes the turn's waiting record (``WorkflowExecutionContext._waiting``);
    the "/" flag is the turn's (``assistant_round``, ``end_round``). The owner is held weakly: the port sits in a context
    the owner holds."""

    def __init__(self, owner: Any) -> None:
        self._owner = weakref.ref(owner)

    def get(self) -> Optional[Waiting]:
        owner = self._owner()
        return None if owner is None else getattr(owner, "_waiting", None)

    def put(self, record: Waiting) -> None:
        if (owner := self._owner()) is not None:
            owner._put_record(record)

    def clear(self) -> None:
        if (held := self.get()) is not None and held.kind in CME_KINDS:
            self._owner()._waiting = None

    @property
    def assistant(self) -> bool:
        owner = self._owner()
        return bool(owner is not None and owner.assistant_round)

    def end_round(self) -> None:
        if (owner := self._owner()) is not None:
            owner.end_round()


class SlotPort(Port):
    """The plan review's slot question (the plan CME's port): a nested record inside the payload of the turn's review
    record (``SLOT_KEY``). With no review record out nothing is held (``get`` is None, ``clear`` does nothing), and a
    ``put`` is a defect: the slot question is only ever asked while the review is out."""

    def __init__(self, owner: Any) -> None:
        self._owner = weakref.ref(owner)

    def _review(self) -> Optional[Waiting]:
        owner = self._owner()
        held = None if owner is None else getattr(owner, "_waiting", None)
        return held if held is not None and held.kind == KIND_REVIEW else None

    def get(self) -> Optional[Waiting]:
        review = self._review()
        return None if review is None else review.payload.get(SLOT_KEY)

    def put(self, record: Waiting) -> None:
        if (review := self._review()) is None:
            raise AssertionError("a review's slot question was put with no review out")
        review.payload[SLOT_KEY] = record

    def clear(self) -> None:
        if (held := self.get()) is not None and held.kind in CME_KINDS:
            self._review().payload[SLOT_KEY] = None

    @property
    def assistant(self) -> bool:
        return False

    def end_round(self) -> None:
        return None


def port_of(context: dict) -> Port:
    """The port of a CME context, installing a ``Holder`` when it has none."""
    port = context.get(PORT_KEY)
    if port is None:
        port = context[PORT_KEY] = Holder(context)
    return port


# ---- the transition table (design note section 3.4), as code ---------------------------------------------------------------

MESSAGE = "message"      # a message arrived: the reply to the question that is out, else the start of a turn
CANCEL = "cancel"        # the question is cancelled (``//cancel``, the chat page's Cancel)
RESTORE = "restore"      # a saved record is put back (``Event.state``)

ACT_NOTHING = "nothing"                  # CANCEL: not a reply (no question, or a blocked ``ask_user`` read)
ACT_REPLY = "reply"                      # CANCEL: send ``Move.text`` as a reply, the way typing it does
ACT_RESTORE = "restore"                  # RESTORE: ``Move.record`` is the record that comes back
ACT_MANAGE = "manage"                    # MESSAGE: the user's own approval management message (``/approvals`` ...)
ACT_APPROVAL = "approval"                # MESSAGE: the approval store resolves it (an approval raised outside the agent's loop,
                                         #   or ``/cancel_pending``)
ACT_RESUME_AGENT = "resume_agent"        # MESSAGE: the reply continues the suspended turn; ``Move.reader`` reads it
ACT_START_AGENT = "start_agent"          # MESSAGE: the agent starts on a request
ACT_DETERMINISTIC = "deterministic"      # MESSAGE: the deterministic path (the CME) takes it


class Event(NamedTuple):
    """What happened: a message (``text`` as the user typed it), a cancel, or a restore (the saved ``state``)."""

    kind: str
    text: str = ""
    state: Any = None


class Facts(NamedTuple):
    """What ``step`` needs to know about the turn and nothing else: the session's path and the message's own shape."""

    agent_path: bool = False         # the session runs the agent (a "/" round and a CME reply are not the agent's)
    assistant_round: bool = False    # a "/" round is already open
    manage: bool = False             # the text is an approval management message (``/approvals``, ``/revoke_approval``)
    cancel_pending: bool = False     # the text is ``/cancel_pending``


class Move(NamedTuple):
    """The one decision. ``text``: what the stages below read (a reply has its leading "/" removed). ``begin``: a turn
    begins first (the message is not a reply to a suspended turn). ``assistant``: the message opens a "/" round.
    ``reader``: where a reply goes (``TABLE``). ``label``: what a cancel is called (``KindSpec.cancel_label``).
    ``record``: the restored record."""

    action: str
    text: str = ""
    begin: bool = False
    assistant: bool = False
    reader: str = ""
    label: str = ""
    record: Optional[Waiting] = None


def step(record: Optional[Waiting], event: Event, facts: Facts = Facts()) -> Move:
    """What *event* does to the question that is out (*record*, None when there is none): pure, no state, no model.

    =========  ==========================================================================================================
    event      decision
    =========  ==========================================================================================================
    CANCEL     the kind's cancel reply (``TABLE``), or nothing for no question and for the agent's own (not a reply)
    RESTORE    the record the saved state describes (None for none)
    MESSAGE    with no question out: a turn begins, and a "/" message opens the assistant round; with a question out: the
               message is its reply, with every leading "/" removed (a "/" never starts a command beside a question), and
               goes to the kind's reader. A management message (``/approvals`` ...) and ``/cancel_pending`` keep their
               text and their meaning. Then, in this order: the management message is answered unless an approval or a
               review that suspended the agent is out; an approval or a review (the store's record) resolves the message
               unless it suspended the agent in its loop (then only ``/cancel_pending`` is handled there); on the agent
               path a reply continues the suspended turn and a request starts the agent, unless a "/" round is open (the
               deterministic path); everything else is the deterministic path's.
    =========  ==========================================================================================================
    """
    if event.kind == CANCEL:
        if record is None:
            return Move(ACT_NOTHING)
        spec = record.spec
        text = record.cancel_text
        return Move(ACT_NOTHING if text is None else ACT_REPLY, text or "", label=spec.cancel_label)
    if event.kind == RESTORE:
        return Move(ACT_RESTORE, record=Waiting.from_state(event.state))
    if event.kind != MESSAGE:
        raise ValueError(f"unknown event {event.kind!r}")

    text, out, assistant = event.text, record is not None, False
    if not (facts.manage or facts.cancel_pending):
        if out:
            text = text.lstrip("/")
        elif text.startswith("/"):
            assistant = True
    begin = not (out and record.rides_suspension)
    reader = record.reader if out else ""
    store_out = out and record.store_backed
    store_suspended = out and record.suspended_agent
    if facts.manage and not store_suspended:
        return Move(ACT_MANAGE, event.text, begin, assistant, reader)
    if store_out and (not store_suspended or facts.cancel_pending):
        return Move(ACT_APPROVAL, text, begin, assistant, reader)
    if facts.agent_path and not (facts.assistant_round or assistant):
        return Move(ACT_RESUME_AGENT if not begin else ACT_START_AGENT, text, begin, assistant, reader)
    return Move(ACT_DETERMINISTIC, text, begin, assistant, reader)


def can_ask(ctx: Any) -> bool:
    """Whether a question can be put to somebody on this turn and its reply come back: the turn is attended. The one
    predicate; each call site adds only its own further conditions (nothing ran yet, not asked yet, the agent path, ...).

    Every question it governs suspends the turn on its waiting record, and the reply is the next message on every
    transport, the terminal's included (``ChatSession`` reads it from its queue and routes it through ``step`` like any
    message; ``//cancel`` answers it with its kind's cancel reply). Only the agent's own ``ask_user`` blocks on that
    queue instead of suspending, and it does not ask this predicate (bd fix-ftdl.7.7)."""
    return getattr(ctx, "_turn_unattended", True) is False


def can_review(ctx: Any) -> bool:
    """Whether somebody can answer the turn's one review: not an unattended turn (ask_user is off for it) and not a host
    surface (REST, MCP), except that with no agent model the host's client is the agent that drives the work, so it sees
    the review (with a plan preview on each part that will not be run here) and answers it. That is only where the reply
    can be accepted: the host approval path takes a reply to a review only from a verified end user
    (``Surface.satisfies_end_user``); with an unverified one the review would be a dead end, so none is opened.
    Unattended turns stay unreviewed. An agent-to-agent turn reaches the framework as an unattended turn."""
    from fastworkflow.review import approval_binding

    store = approval_binding.store_for(ctx._app_workflow)
    host = store is not None and store.surface.kind == "host"
    if host and ctx.no_agent_mode and not ctx._turn_unattended and store.surface.satisfies_end_user():
        return True
    return not (bool(ctx._turn_unattended) or host)
