"""Approval binding for approval-annotated commands (ADR-EXP-025, sections 6-12 and 15).

A thin ``action``-kind wrapper over the review engine (``fastworkflow.review.review_engine``). The engine owns
the record, the state machine, the cascade, the audit and the redaction rule. This module owns what is
specific to a gated write:

* the binding and its fingerprint (sha256 over canonical JSON; section 6.2);
* the policy decision point for the gate, which returns a graded outcome: run / ask / not asked / refuse
  (ADR section 19.0 condition of approval; deterministic, no model call);
* the gate itself (``gate``), called from the single execution choke point (``CommandExecutor``);
* the resolution of a reply: a yes (``no_agent_mode.is_yes``), "no" or "always" (read as a yes is), token, identity class, fingerprint, TTL,
  execute-once (sections 8, 10, 11); a reply on a proposal record goes to the review engine;
* the user's management messages (standing approvals) and the origin contextvars.

Switch: ``FW_APPROVAL_BINDING`` = ``off`` | ``shadow`` | ``enforce`` (default ``enforce``), read by the
engine. ``off`` restores the pre-change path; ``shadow`` records and audits every decision but executes as
before.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import string  # noqa: F401 - kept importable for callers that formatted templates here
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import fastworkflow
from fastworkflow.review import approval_grants, review_engine
from fastworkflow.session import state_paths
from fastworkflow.approval_annotations import (APPROVAL_VALUES, REQUIRED_ALWAYS,
                                               REQUIRED_UNTIL_DELEGATED, command_approval)
from fastworkflow.context_identity import context_identity
# The engine's public names, re-exported: the approval surface the gate, the hosts and the tests use.
from fastworkflow.review.review_engine import (  # noqa: F401
    ACTIVE_STATES,
    AWAITING,
    CHANNEL_SURFACE,
    CONFIRMED,
    DECLINED,
    DEFAULT_MODE,
    ENV_SWITCH,
    EXECUTED,
    EXPIRED,
    IDENTITY_CHANNEL,
    IDENTITY_END_USER,
    INVALIDATED,
    KIND_ACTION,
    MAX_HISTORY,
    MODE_ENFORCE,
    MODE_OFF,
    MODE_SHADOW,
    MODES,
    TTL_SECONDS,
    TTL_TURNS,
    ReviewRecord as PendingAction,
    ReviewStore as ApprovalStore,
    Surface,
    active_record,
    audit as _audit,
    canonical_json,
    clear_inbound,
    default_rendering,
    expire_if_due,
    fingerprint,
    new_token,
    propagate,
    proposal_of,
    reply_policy,
    reply_to_proposal,
    render,
    rendering_hash,
    store_for,
    switch_mode,
    take_delivery,
    token_hash,
    token_matches,
    transition as _transition,
)
from fastworkflow.review.review_engine import ActionReplyPolicy, Verdict

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Exact user-facing texts (ADR section 10). Framework-rendered only.
# ----------------------------------------------------------------------

# The question states only what the framework knows: who asks, the command and its values, and the replies. Whether
# the change can be undone is the author's to say, in the command's ``approval_template`` (bd fix-ftdl.7.6).
MSG_ASK = ("The agent wants to run {command} with {rendering}. "
           "Reply yes to approve or no to decline. You are asked every time.")
MSG_ASK_USER = ("You asked to run {command} with {rendering}. "
                "Reply yes to approve or no to decline. You are asked every time.")
MSG_ASK_DELEGATABLE = ("The agent wants to run {command} with {rendering}. "
                       "Reply yes to approve, no to decline, or always to approve this command from now on.")
# A call that needs a yes and cannot be asked now is not run and not kept: one question is out at a time (``waiting``),
# and a record nobody would ever be asked about is not a question (bd fix-ftdl.7.6).
MSG_UNATTENDED = ("{command} was not run: it needs your approval, and no one can give it on this turn. "
                  "Run it again when you can answer.")
MSG_BEHIND = ("{command} was not run: it needs your approval, and another approval is already waiting for your "
              "answer. Answer that one first, then run it again.")
MSG_REPLY = "Reply yes to approve, no to decline, or always to approve this command from now on. The action has not run."
#: The same question for a ``REQUIRED_ALWAYS`` command: no standing approval exists for it, so ``always`` is not offered.
MSG_REPLY_EVERY_TIME = ("Reply yes to approve or no to decline. This command asks every time, so there is no "
                        "always. The action has not run.")
MSG_GRANT_IGNORED = " (ignored: this command asks every time; revoke it to remove it)"
MSG_RECORD_FAILED = "The standing approval for {command} could not be recorded, so the action was not run."
MSG_NO_GRANT = "No standing approval for {command} is recorded for you."
MSG_ALWAYS_HINT = ("Standing approval recorded. List your standing approvals with /approvals and revoke "
                   "one with /revoke_approval <command>.")
MSG_GRANT_REVOKED = "The standing approval for {command} is revoked. It will ask you again."
MSG_DECLINED = "Declined. {command} was not run."
MSG_INVALIDATED = ("The parameters for {command} changed after the approval was requested. The action "
                   "was not run. Review the new request.")
MSG_EXPIRED = "The approval for {command} expired and the action was not run."
MSG_CANCELLED = "The approval for {command} was cancelled and the action was not run."
MSG_NO_LONGER_VALID = "That approval is no longer valid."
MSG_IDENTITY = ("This action requires a verified user identity, which this server is not configured "
                "to check.")
MSG_FAILED = "{command} failed after approval; it was not retried: {error}"
#: A call the gate cannot check for an approval mark is not run (it fails closed, bd fix-ftdl.8.17).
MSG_UNCHECKED = ("{command} was not run: the workflow's commands could not be read to check whether it needs your "
                 "approval.")
MSG_SEVERAL = ("{command} was not run: that name matches several commands ({names}) and one of them needs your "
               "approval. Call it by its full name.")

#: The reply that approves (the card's word; any yes of ``no_agent_mode.is_yes`` approves), and the one that declines
#: (the whole reply, case and a trailing ``.`` or ``!`` ignored, as a yes is).
REPLY_YES = "yes"
REPLY_NO = "no"
#: Approves this one binding and records a standing auto-approval for the command (owner 2026-11-02).
REPLY_ALWAYS = "always"
#: The user's management messages (listing and revoking standing approvals); never the agent's.
MANAGE_LIST = "/approvals"
MANAGE_REVOKE = "/revoke_approval"
CANCEL_COMMAND = "/cancel_pending"
#: The framework's review command (a core execution command in the command metadata extraction workflow).
REVIEW_COMMAND = "review_proposal"


# ----------------------------------------------------------------------
# Origin and approved-execution context (ADR section 5)
# ----------------------------------------------------------------------


class Origin(str, Enum):
    """Who asked for the call. Only agent-originated calls are gated."""

    USER = "user"                  # the user's own instruction: assistant mode, '/', direct actions
    AGENT = "agent"                # the agent's tool loop, including planning-command steps: suspendable
    AGENT_DIRECT = "agent_direct"  # agent-originated MCP tool calls (owner answer 3): not suspendable


_origin_var: contextvars.ContextVar[Origin] = contextvars.ContextVar(
    "fw_approval_origin", default=Origin.USER)
_approved_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "fw_approval_approved_record", default=None)


def current_origin() -> Origin:
    return _origin_var.get()


@contextlib.contextmanager
def origin_scope(value: Origin):
    token = _origin_var.set(value)
    try:
        yield
    finally:
        _origin_var.reset(token)


@contextlib.contextmanager
def approved_execution(record_id: str):
    """Mark the execution of one confirmed record: the gate lets exactly that binding run."""
    token = _approved_var.set(record_id)
    try:
        yield
    finally:
        _approved_var.reset(token)


# ----------------------------------------------------------------------
# Binding (ADR sections 6.1, 6.2)
# ----------------------------------------------------------------------


def make_binding(*, command: str, params: dict, targets: list, principal: dict,
                 task: str, channel: str) -> dict:
    return {"command": command, "params": params, "targets": targets,
            "principal": principal, "task": task, "channel": channel}


# ----------------------------------------------------------------------
# Policy decision point (ADR section 19.0 condition of approval)
# ----------------------------------------------------------------------


class Outcome(str, Enum):
    RUN = "run"
    ASK = "ask"
    #: Needs a yes that cannot be asked for now (no person, or another question is out): not run, nothing kept.
    NOT_ASKED = "not_asked"
    REFUSE = "refuse"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: str


def decide(annotation: Optional[str], *, origin: Origin, surface: Surface, unattended: bool,
           awaiting_on_channel: bool, auto_granted: bool = False, consequence: Any = None,
           uncertainty: Any = None) -> Decision:
    """The gate's policy: a pure, deterministic function from its inputs to a graded outcome.

    ``auto_granted`` is the user's standing approval for this command (approval_grants): an input,
    so a recorded grant yields ``run`` without a prompt for ``REQUIRED_UNTIL_DELEGATED`` only. On a channel
    surface an unattended turn is checked first and is not asked, whatever the grant says; a verified host surface's
    per-end-user grant still runs (ADR-EXP-025 amendment 2026-11-02: the grant is the user's delegation).
    A ``REQUIRED_ALWAYS`` command is never run on a grant (bd fix-qjne.11): a person confirms every run, so
    the grant is ignored and the command asks, or is not asked (and not run) when nobody can answer. ``consequence`` and
    ``uncertainty`` are reserved inputs (fix-ajv.11, fix-47a9 delegation): accepted so the decision
    can grow without a new signature, and they never turn an outcome into a boolean.

    A user's own ``REQUIRED_UNTIL_DELEGATED`` instruction runs (owner 2026-11-02 leaves it unchanged). A user's
    own ``REQUIRED_ALWAYS`` instruction asks, the same as an agent's call (owner amendment 2026-11-02).
    """
    del consequence, uncertainty  # reserved; see docstring
    if annotation not in APPROVAL_VALUES:
        return Decision(Outcome.RUN, "unannotated")
    if origin is Origin.USER and annotation == REQUIRED_UNTIL_DELEGATED:
        return Decision(Outcome.RUN, "delegatable, and the user's own instruction")
    if surface.required_identity == IDENTITY_END_USER and not surface.satisfies_end_user():
        return Decision(Outcome.REFUSE, "end_user identity required and not verifiable")
    if unattended and surface.kind != "host":
        # A channel surface's standing grant is the local user's, made at a prompt on an interactive
        # channel; no human is present now, so it never lets an unattended run execute (fix-d6no).
        return Decision(Outcome.NOT_ASKED, "no human channel")
    if auto_granted and annotation != REQUIRED_ALWAYS:
        return Decision(Outcome.RUN, "standing auto-approval recorded by the user")
    if unattended:
        return Decision(Outcome.NOT_ASKED, "no human channel")
    if awaiting_on_channel:
        return Decision(Outcome.NOT_ASKED, "behind an awaiting approval")
    if annotation == REQUIRED_UNTIL_DELEGATED:
        return Decision(Outcome.ASK, "delegatable asks until delegation exists")
    return Decision(Outcome.ASK, "mandatory")


# ----------------------------------------------------------------------
# Turns
# ----------------------------------------------------------------------


def on_new_turn(host: Any, workflow: Any, *, unattended: bool) -> None:
    """A new user turn starts: advance the turn counter and note whether a human can answer this
    turn (ask_user is off for an unattended turn). An open record ages by the replies it reads."""
    del host  # kept in the signature: the hosts call it with themselves
    store = store_for(workflow)
    if store is None:
        return
    store.turn_seq += 1
    store.unattended = bool(unattended)


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------


def _qualified(workflow, command_name: str) -> tuple[Optional[str], tuple[str, ...]]:
    """``(qualified, candidates)``: the qualified name of *command_name* in the workflow's routing (None when it names
    no command or several), and the commands its leaf names when there are several. Raises when the routing cannot be
    read: the gate then cannot tell whether the call needs approval (bd fix-ftdl.8.17)."""
    routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    directory = routing.command_directory.map_command_2_metadata
    if command_name in directory:
        return command_name, ()
    leaf = sorted(name for name in directory if name.split("/")[-1] == command_name)
    return (leaf[0], ()) if len(leaf) == 1 else (None, tuple(leaf))


def _checked_name(workflow, command_name: str) -> tuple[Optional[str], Optional[str]]:
    """``(qualified, refusal)``: the qualified name of *command_name* (None when it names no command, which has no mark
    to read), or the text of the refusal of a call the gate cannot check for an approval mark, which is therefore not
    run (the gate fails closed, bd fix-ftdl.8.17): the routing or a mark cannot be read, or the name matches several
    commands one of which carries a mark."""
    try:
        qualified, candidates = _qualified(workflow, command_name)
        marked = [name for name in candidates if command_approval(workflow.folderpath, name) is not None]
    except Exception:  # noqa: BLE001 - unread routing or mark: the call is not run
        logger.warning("approval gate: routing unread for %r; the call is not run", command_name, exc_info=True)
        return None, MSG_UNCHECKED.format(command=command_name)
    if marked:
        return None, MSG_SEVERAL.format(command=command_name, names=", ".join(candidates))
    return qualified, None


def _params_of(input_obj: Any) -> dict:
    if input_obj is None:
        return {}
    if hasattr(input_obj, "model_dump"):
        return input_obj.model_dump(mode="json")
    return dict(input_obj) if isinstance(input_obj, dict) else {}


def build_binding(workflow, command: str, params: dict, surface: Surface, store: ApprovalStore) -> dict:
    """The binding that would execute now: the target is the instance in the current context."""
    principal = ({"class": IDENTITY_END_USER, "ref": surface.uid or "none"}
                 if surface.required_identity == IDENTITY_END_USER
                 else {"class": IDENTITY_CHANNEL, "ref": store.channel})
    return make_binding(command=command, params=params, targets=list(context_identity(workflow)),
                        principal=principal, task=store.task, channel=store.channel)


def awaiting_output(workflow, text: str):
    """An awaiting CommandOutput carrying the framework text, and any token just delivered."""
    command_response = fastworkflow.CommandResponse(response=text)
    command_response.artifacts["awaiting_user"] = True
    output = fastworkflow.CommandOutput(command_response=command_response)
    if workflow is not None:
        output.workflow_name = workflow.folderpath.split("/")[-1]
    return output


def _refusal_output(workflow, command: str, text: str):
    output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response=text, success=False),
        command_name=command)
    if workflow is not None:
        output.workflow_name = workflow.folderpath.split("/")[-1]
    return output


def _decision(workflow, store: ApprovalStore, qualified: str, annotation: str, origin: Origin, *,
              awaiting_on_channel: bool) -> tuple[Decision, Optional[dict], Any]:
    """``(decision, grant, identity)``: the gate's decision for *qualified* now, from the user's standing grant and
    whether a person can answer this turn. The one computation ``gate`` and ``would_ask`` read."""
    identity = approval_grants.identity_key(store.surface, store.channel)
    grant = approval_grants.find(identity, state_paths.workflow_id(workflow.folderpath), qualified)
    decision = decide(annotation, origin=origin, surface=store.surface, unattended=store.unattended,
                      awaiting_on_channel=awaiting_on_channel, auto_granted=grant is not None)
    return decision, grant, identity


def would_ask(workflow: Any, qualified: str, origin: Origin) -> bool:
    """Whether running *qualified* now, called by *origin*, would ask the user for approval (the gate enforces, the
    command carries an approval mark, and the decision is ASK: no standing grant lets it run, a person can answer).
    Such a question names the command and its values, so it is the user's yes to it: a command that changes data and
    that the local model alone chose needs no other confirmation (``intent_loop.unconfirmed_pick``). Another approval
    already out is not counted: that one is answered first. Never raises (False)."""
    try:
        store = store_for(workflow)
        if store is None or switch_mode() != MODE_ENFORCE:
            return False
        annotation = command_approval(workflow.folderpath, qualified)
        if annotation is None:
            return False
        decision, _, _ = _decision(workflow, store, qualified, annotation, origin, awaiting_on_channel=False)
    except Exception:  # noqa: BLE001 - unreadable: the approval is not counted on, so the other confirmation asks
        logger.debug("approval decision unread for %r", qualified, exc_info=True)
        return False
    return decision.outcome is Outcome.ASK


def gate(chat_session: Any, workflow: Any, command_name: str, input_obj: Any, raw_command: str):
    """Decide whether a command call may run now.

    Returns None to run; otherwise a CommandOutput the caller returns as the command's result
    (refuse, not asked, or an awaiting output for a non-suspendable caller). For an agent-origin ask
    inside the agent's tool loop, raises ``AskUserSuspend``: the ask_user suspension.
    """
    store = store_for(workflow)
    if store is None:
        return None
    approved = _approved_var.get()
    if approved is not None:
        # The one execution a confirmed record authorised: the stored binding, unchanged.
        record = store.get(approved)
        if record is not None and record.state == CONFIRMED:
            return None
    mode = switch_mode()
    if mode == MODE_OFF:
        return None
    origin = current_origin()
    qualified, refusal = _checked_name(workflow, command_name)
    if refusal is not None:
        return None if mode == MODE_SHADOW else _refusal_output(workflow, command_name, refusal)
    if qualified is None:
        return None
    annotation = command_approval(workflow.folderpath, qualified)
    if annotation is None:
        return None

    params = _params_of(input_obj)
    binding = build_binding(workflow, qualified, params, store.surface, store)
    fp = fingerprint(binding)
    rendering = render(_approval_template(workflow, qualified), params)
    decision, grant, identity = _decision(workflow, store, qualified, annotation, origin,
                                          awaiting_on_channel=store.active() is not None)
    if grant is not None and decision.outcome is Outcome.RUN and mode == MODE_ENFORCE:
        approval_grants.audit(chat_session, grant["id"], command=qualified, annotation=annotation,
                              before=approval_grants.STATE_GRANTED, after="used", reason=decision.reason,
                              identity=identity, channel=store.channel, surface_kind=store.surface.kind,
                              outcome="used")

    def new_record(*, state: str, suspended: bool, text: str) -> PendingAction:
        record = PendingAction(
            id=uuid.uuid4().hex, token_hash="", command=qualified, annotation=annotation,
            raw_command=raw_command, binding=binding, fingerprint=fp, rendering=text,
            rendering_hash=rendering_hash(text), required_identity=store.surface.required_identity,
            created_turn=store.turn_seq, turns_seen=0, ttl_turns=TTL_TURNS,
            ttl_seconds=TTL_SECONDS, created_at=time.time(), state=state, origin=origin.value,
            suspended_agent=suspended, shadow=(mode == MODE_SHADOW))
        store.add(record)
        return record

    def audited_transient(after: str, text: str) -> None:
        """A refusal or a call not asked creates no record (nothing waits for an answer); it is audited on a
        transient record that is never stored."""
        transient = PendingAction(
            id=uuid.uuid4().hex, token_hash="", command=qualified, annotation=annotation,
            raw_command=raw_command, binding=binding, fingerprint=fp, rendering=text,
            rendering_hash=rendering_hash(text), required_identity=store.surface.required_identity,
            created_turn=store.turn_seq, turns_seen=0, ttl_turns=TTL_TURNS, ttl_seconds=TTL_SECONDS,
            created_at=time.time(), state=after, origin=origin.value, shadow=(mode == MODE_SHADOW))
        _audit(chat_session, transient, before="", after=after, reason=decision.reason,
               surface=store.surface, mode=mode, outcome=after)

    if decision.outcome is Outcome.REFUSE:
        audited_transient("refused", "")
        if mode == MODE_SHADOW:
            return None
        return _refusal_output(workflow, qualified, MSG_IDENTITY)

    if mode == MODE_SHADOW:
        # Shadow records the decision and runs as before: the record is born executed, and its
        # one audit row says what enforcement would have done.
        record = new_record(state=EXECUTED, suspended=False, text="")
        record.outcome = "shadow"
        _audit(chat_session, record, before=EXECUTED, after=EXECUTED,
               reason=f"shadow: would {decision.outcome.value} ({decision.reason}); ran",
               surface=store.surface, mode=mode, outcome="shadow")
        return None

    if decision.outcome is Outcome.RUN:
        return None

    if decision.outcome is Outcome.NOT_ASKED:
        text = (MSG_UNATTENDED if store.unattended else MSG_BEHIND).format(command=qualified)
        audited_transient(Outcome.NOT_ASKED.value, text)
        return _refusal_output(workflow, qualified, text)

    # ASK
    if origin is Origin.USER:
        template = MSG_ASK_USER
    else:
        template = MSG_ASK if annotation == REQUIRED_ALWAYS else MSG_ASK_DELEGATABLE
    text = template.format(command=qualified, rendering=rendering)
    suspendable = origin is Origin.AGENT
    record = new_record(state=AWAITING, suspended=suspendable, text=text)
    token = new_token()
    record.token_hash = token_hash(token)
    if store.surface.kind == "host":
        store.delivery = {"record_id": record.id, "token": token}
    _audit(chat_session, record, before="", after=AWAITING, reason=decision.reason,
           surface=store.surface, mode=mode, outcome="asked")
    if (put := getattr(review_engine.question_owner(store, chat_session), "_put_approval_record", None)) is not None:
        put(record)      # the question is the turn's waiting record, pointing at this one (``waiting``)
    if suspendable:
        from fastworkflow.utils.react import AskUserSuspend

        raise AskUserSuspend(text)
    return awaiting_output(workflow, text)


def _approval_template(workflow, qualified: str) -> Optional[str]:
    try:
        routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        cls = routing.get_command_class(qualified, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        value = getattr(cls, "approval_template", None)
        return value if isinstance(value, str) else None
    except Exception:  # noqa: BLE001 - a missing template falls back to the default rendering
        return None


# ----------------------------------------------------------------------
# Resolution of a reply (ADR sections 8, 10, 11)
# ----------------------------------------------------------------------


@dataclass
class Resolution:
    """``done`` False: the record is still open; ``text`` is what the user is told. ``done``
    True: the record left the open states; ``text`` is the observation for the agent, ``output`` is the
    command output to return when no agent is waiting, and ``value`` is what a review accepted (None for
    an action)."""

    done: bool
    text: str
    output: Any = None
    value: Any = None


def resolve_reply(chat_session: Any, workflow: Any, message: str) -> Optional[Resolution]:
    """Resolve the next message on a session that has an open record; None when there is none."""
    store = store_for(workflow)
    record = store.active() if store is not None else None
    if record is None:
        return None
    surface = store.surface
    cmd = record.command

    if message == CANCEL_COMMAND:
        _transition(chat_session, store, record, INVALIDATED, reason="cancelled",
                    surface=surface, outcome="cancelled")
        propagate(chat_session, store, record, surface=surface)
        return Resolution(True, MSG_CANCELLED.format(command=cmd))

    if record.time_expired(time.time()):
        _transition(chat_session, store, record, EXPIRED, reason="ttl", surface=surface, outcome="expired")
        propagate(chat_session, store, record, surface=surface)
        return Resolution(True, MSG_EXPIRED.format(command=cmd))

    if record.kind == KIND_ACTION and record.turns_expired(turn_seq=store.turn_seq):
        _transition(chat_session, store, record, EXPIRED, reason="ttl", surface=surface, outcome="expired")
        propagate(chat_session, store, record, surface=surface)
        return Resolution(True, MSG_EXPIRED.format(command=cmd))

    if record.required_identity == IDENTITY_END_USER:
        if not token_matches(store.inbound_token, record.token_hash):
            return Resolution(False, MSG_NO_LONGER_VALID)
        if not surface.satisfies_end_user():
            return Resolution(False, MSG_IDENTITY)

    if record.kind != KIND_ACTION:
        # Proposals: the reply cap counts unresolved replies only. On the cap turn (turns_seen already at the
        # limit), confirm or decline is still honoured; any other reply expires without applying an edit.
        cap_turn = record.turns_seen >= record.ttl_turns
        review = reply_to_proposal(chat_session, store, record, message, cap_turn=cap_turn)
        if review.done and review.proposal is not None:
            from fastworkflow.review.review_proposals import PROPOSAL_KINDS

            if any(cls.spec == record.command for cls in PROPOSAL_KINDS.values()):
                # A framework round accepted: the review command runs once more under this record, so its typed
                # output is the accepted proposal (the step the planning command is waiting on).
                output = _execute_review(workflow, record)
                return Resolution(True, output.command_response.response or review.text, output, review.proposal)
            # Any other proposal (a caller's own spec): the accepted proposal is the result.
            return Resolution(True, review.text, None, review.proposal)
        if not review.done:
            record.turns_seen += 1
        return Resolution(review.done, review.text, None, review.proposal)

    record.turns_seen += 1
    verdict = ActionReplyPolicy().classify(message)
    if verdict is Verdict.REPROMPT or (verdict is Verdict.ALWAYS and record.annotation == REQUIRED_ALWAYS):
        # ``always`` is not an answer to a REQUIRED_ALWAYS question: asked again, nothing recorded.
        return Resolution(False, MSG_REPLY_EVERY_TIME if record.annotation == REQUIRED_ALWAYS else MSG_REPLY)

    if verdict is Verdict.DECLINE:
        _transition(chat_session, store, record, DECLINED, reason="declined", surface=surface,
                    outcome="declined",
                    resolution=_resolution(record, surface, "declined"))
        return Resolution(True, MSG_DECLINED.format(command=cmd))

    # always: the standing approval needs an identity the grant can belong to.
    identity = approval_grants.identity_key(surface, store.channel)
    if verdict is Verdict.ALWAYS and identity is None:
        return Resolution(False, MSG_IDENTITY)

    # yes / always: the binding that would execute now must still be the one the user saw.
    params = dict(record.binding.get("params") or {})
    current = build_binding(workflow, cmd, params, surface, store)
    if fingerprint(current) != record.fingerprint:
        _transition(chat_session, store, record, INVALIDATED, reason="binding changed",
                    surface=surface, outcome="invalidated")
        return Resolution(True, MSG_INVALIDATED.format(command=cmd))

    if verdict is Verdict.ALWAYS:
        # Recorded only now, once the binding is confirmed: a rejected reply records nothing.
        try:
            grant = approval_grants.record(identity, state_paths.workflow_id(workflow.folderpath), cmd,
                                           record.annotation)
        except (RuntimeError, OSError) as exc:
            logger.warning(f"standing approval not recorded ({type(exc).__name__})")
            return Resolution(False, MSG_RECORD_FAILED.format(command=cmd))
        approval_grants.audit(chat_session, grant["id"], command=cmd, annotation=record.annotation,
                              before=approval_grants.STATE_NONE, after=approval_grants.STATE_GRANTED,
                              reason="recorded by the user (always)", identity=identity,
                              channel=store.channel, surface_kind=surface.kind, outcome="recorded")

    _transition(chat_session, store, record, CONFIRMED, reason="confirmed", surface=surface,
                resolution=_resolution(record, surface, "confirmed"))
    try:
        with approved_execution(record.id):
            output = _execute_stored(workflow, record)
    except Exception as exc:  # noqa: BLE001 - executed once; the failure is the outcome, never retried
        _transition(chat_session, store, record, EXECUTED, reason="execution failed", surface=surface,
                    outcome="failed")
        text = MSG_FAILED.format(command=cmd, error=f"{type(exc).__name__}: {exc}")
        return Resolution(True, text, _refusal_output(workflow, cmd, text))
    _transition(chat_session, store, record, EXECUTED, reason="executed", surface=surface,
                outcome="success" if output.success else "failed")
    if output.success:
        # An approved write inside a planning command's phase counts against its step, as a run that was not asked
        # about does (``CommandExecutor._invoke_command_impl``): a repeat is then refused, not asked again.
        from fastworkflow import command_writes

        command_writes.record_phase_write(chat_session, workflow.folderpath, cmd)
    text = output.command_response.response or ""
    if verdict is Verdict.ALWAYS:
        # Say how to undo it where the user reads the reply (the rendering hash covers the prompt, not this).
        text = f"{text}\n\n{MSG_ALWAYS_HINT}" if text else MSG_ALWAYS_HINT
        output.command_response.response = text
    return Resolution(True, text, output)


def _resolution(record: PendingAction, surface: Surface, verdict: str) -> dict:
    return {"verdict": verdict, "identity_class": record.required_identity,
            "identity_ref": surface.uid if surface.uid else "none"}


def _execute_review(workflow, record: PendingAction):
    """Run the review command once for an accepted proposal record (the approved execution)."""
    from fastworkflow.command_executor import CommandExecutor

    action = fastworkflow.Action(command_name=REVIEW_COMMAND, command=REVIEW_COMMAND,
                                 parameters={"items": record.binding.get("items_text", "")})
    with review_engine.approved_review(record.id):
        return CommandExecutor.perform_action(workflow, action)


def _execute_stored(workflow, record: PendingAction):
    """Run the stored binding once, through the same executor the agent's call uses."""
    from fastworkflow.command_executor import CommandExecutor

    action = fastworkflow.Action(command_name=record.command, command=record.raw_command,
                                 parameters=dict(record.binding.get("params") or {}))
    return CommandExecutor.perform_action(workflow, action)


def cancel_all(chat_session: Any, workflow: Any, *, reason: str = "session ended") -> int:
    """Invalidate every open record: session end, eviction, /cancel_pending (ADR section 11).

    Returns how many records were cancelled. Nothing executes, and a later reply finds no record.
    """
    store = store_for(workflow)
    if store is None:
        return 0
    cancelled = 0
    for record in list(store.records):
        if record.state in ACTIVE_STATES:
            _transition(chat_session, store, record, INVALIDATED, reason=reason,
                        surface=store.surface, outcome="cancelled")
            propagate(chat_session, store, record, surface=store.surface)
            cancelled += 1
    store.delivery = None
    store.inbound_token = None
    return cancelled


def active_approvals(workflow: Any) -> list:
    """The active approval records (action kind) on the session; [] without a workflow."""
    store = store_for(workflow)
    if store is None:
        return []
    return [r for r in store.records if r.state in ACTIVE_STATES and r.kind == KIND_ACTION]


def active_reviews(workflow: Any) -> list:
    """The active review records (sub-task or plan proposals) on the session; [] without a workflow."""
    store = store_for(workflow)
    if store is None:
        return []
    return [r for r in store.records if r.state in ACTIVE_STATES and r.kind != KIND_ACTION]


def cancel_conversation_approvals(chat_session: Any, workflow: Any, *, reason: str) -> int:
    """Cancel the active approvals of a conversation that is no longer the active one (fix-newconv-pending).

    Recorded the way cancel_all records a cancellation: a transition to invalidated with outcome "cancelled". Reviews
    are not touched; they are parked with their conversation. Returns how many approvals were cancelled.
    """
    store = store_for(workflow)
    if store is None:
        return 0
    cancelled = 0
    for record in active_approvals(workflow):
        _transition(chat_session, store, record, INVALIDATED, reason=reason,
                    surface=store.surface, outcome="cancelled")
        propagate(chat_session, store, record, surface=store.surface)
        cancelled += 1
    if cancelled:
        store.delivery = None
        store.inbound_token = None
    return cancelled


def take_record(workflow: Any, record_id: Any) -> Optional[dict]:
    """Remove the record *record_id* from the session's store and return it, to be parked with the turn's waiting record
    that points at it (None when the store holds none such)."""
    store = store_for(workflow)
    record = None if store is None else store.get(record_id)
    if record is None:
        return None
    store.records = [r for r in store.records if r is not record]
    return record.to_dict()


def restore_record(workflow: Any, data: Optional[dict]) -> None:
    """Put a parked record back on the session's store (the conversation it belongs to is active again)."""
    store = store_for(workflow)
    if store is not None and data:
        store.records.append(PendingAction.from_dict(data))


def take_host_delivery(workflow: Any) -> dict:
    """The live-response fields for a host: ``{"approval": {record_id, token}}``, or {} when no token is
    pending. Taken once; the caller attaches it to the response of the request that produced it."""
    delivery = take_delivery(workflow)
    return {"approval": delivery} if delivery is not None else {}


# ----------------------------------------------------------------------
# The user's management messages: list and revoke standing approvals (owner 2026-11-02)
# ----------------------------------------------------------------------


def _origin_note(surface: Surface) -> tuple[str, str]:
    """(where, other): a channel grant belongs to where it was made, and the other place's grants are
    neither listed nor revoked here (fix-d6no). A host surface has one identity: ("", "")."""
    if getattr(surface, "kind", "channel") == "host":
        return "", ""
    return " on this channel", "Approvals made on another channel are not shown or revoked here."


def management_reply(chat_session: Any, workflow: Any, message: str) -> Optional[str]:
    """The reply to ``/approvals`` or ``/revoke_approval <command>``; None for any other message.

    These are the user's own messages on the session, handled before routing, so the agent has no
    tool for them. Recording a standing approval is only ever the ``always`` reply to a prompt.
    """
    text = message.strip()
    if not text.startswith((MANAGE_LIST, MANAGE_REVOKE)):
        return None
    store = store_for(workflow)
    if store is None:
        return None
    identity = approval_grants.identity_key(store.surface, store.channel)
    if identity is None:
        return MSG_IDENTITY
    workflow_id = state_paths.workflow_id(workflow.folderpath)
    if text == MANAGE_LIST:
        grants = approval_grants.list_active(identity, workflow_id)
        where, other = _origin_note(store.surface)
        if not grants:
            return " ".join(part for part in (f"No standing approvals are recorded for you{where}.", other) if part)
        lines = [f"- {grant['command']}"
                 + (MSG_GRANT_IGNORED if command_approval(workflow.folderpath, grant['command']) == REQUIRED_ALWAYS else "")
                 for grant in grants]
        head = f"Standing approvals{where} (revoke one with /revoke_approval <command>)"
        return head + ":\n" + "\n".join(lines) + (f"\n{other}" if other else "")
    if not text.startswith(MANAGE_REVOKE + " ") or not text[len(MANAGE_REVOKE):].strip():
        return f"Use {MANAGE_REVOKE} <command>, or {MANAGE_LIST} to see the standing approvals."
    name = text[len(MANAGE_REVOKE):].strip()
    matches = [g for g in approval_grants.list_active(identity, workflow_id)
               if g["command"] == name or g["command"].split("/")[-1] == name]
    if not matches:
        return MSG_NO_GRANT.format(command=name)
    if len(matches) > 1:
        return f"{name} matches more than one standing approval; name it fully (see {MANAGE_LIST})."
    grant = approval_grants.revoke(identity, workflow_id, matches[0]["command"])
    if grant is None:
        return MSG_NO_GRANT.format(command=name)
    approval_grants.audit(chat_session, grant["id"], command=grant["command"], annotation=grant["annotation"],
                          before=approval_grants.STATE_GRANTED, after="auto_revoked",
                          reason="revoked by the user", identity=identity, channel=store.channel,
                          surface_kind=store.surface.kind, outcome="revoked")
    return MSG_GRANT_REVOKED.format(command=grant["command"])
