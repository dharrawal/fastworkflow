"""The review engine: one typed-ask primitive for every review the framework asks a user for.

Generalised from the approval binding (ADR-EXP-025, fix-3rsq.10.14). Three proposal kinds share
one record, one state machine, one suspend/resume path, one audit and one redaction rule:

* ``action``: one gated write. Its binding is the tuple that will execute (``approval_binding``
  owns the gate, the token and the execution; this module owns the record and the transitions).
* ``sub_tasks`` and ``plan``: a typed :class:`Proposal` (a Pydantic object: a list of typed items)
  rendered from a template and fingerprinted. A reply is classified by a reply policy. ``accept``
  and ``no`` are verdicts; anything else is interpreted into TYPED EDITS (``change N: field=value``,
  ``drop N``), validated against the item schema before it is applied, and the proposal is then
  re-rendered and asked again. Free text goes to an injectable model interpreter (none is registered
  by default, and tests never register one).

Records form a dependency graph: a record may depend on upstream records (``upstream`` maps each
upstream id to the fingerprint it was built on). Changing or invalidating an upstream invalidates
every dependent whose recorded fingerprint no longer matches, transitively (``propagate``).

Redaction and persistence: records persist with their session (additive ``approval`` state key).
The one-time token is never stored (only its hash) and never reaches audit attributes. Audit
attributes carry hashes, state names, counts and field names, never raw proposal values.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import hmac
import json
import logging
import re
import secrets
import string
import time
import uuid
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Any, Callable, ClassVar, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import fastworkflow
from fastworkflow import intent_loop, parameter_clarification, tracing

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Switch (the approval gate's; the review engine reads it for the audit's mode)
# ----------------------------------------------------------------------

ENV_SWITCH = "FW_APPROVAL_BINDING"
MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ENFORCE = "enforce"
MODES = (MODE_OFF, MODE_SHADOW, MODE_ENFORCE)
DEFAULT_MODE = MODE_ENFORCE

#: An approval (action) record's reply cap and wall-clock TTL: two replies, or 15 minutes.
TTL_TURNS = 2
TTL_SECONDS = 900
#: A proposal review (sub_tasks) is a dialog of edits: about 8 replies, or 30 minutes (fix-3rsq.plan-review.2.5).
PROPOSAL_TTL_TURNS = 8
PROPOSAL_TTL_SECONDS = 30 * 60
MAX_HISTORY = 100

# ----------------------------------------------------------------------
# Kinds and states
# ----------------------------------------------------------------------

KIND_ACTION = "action"
KIND_SUB_TASKS = "sub_tasks"
KINDS = (KIND_ACTION, KIND_SUB_TASKS)

AWAITING = "awaiting"
REVISED = "revised"
CONFIRMED = "confirmed"
DECLINED = "declined"
EXPIRED = "expired"
INVALIDATED = "invalidated"
EXECUTED = "executed"
ACTIVE_STATES = frozenset({AWAITING, REVISED})
#: The only legal transitions. A terminal state never returns to an active one.
_LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    AWAITING: frozenset({REVISED, CONFIRMED, DECLINED, EXPIRED, INVALIDATED}),
    REVISED: frozenset({REVISED, CONFIRMED, DECLINED, EXPIRED, INVALIDATED}),
    CONFIRMED: frozenset({EXECUTED, INVALIDATED}),
    DECLINED: frozenset(),
    EXPIRED: frozenset(),
    INVALIDATED: frozenset(),
    EXECUTED: frozenset(),
}

IDENTITY_CHANNEL = "channel"
IDENTITY_END_USER = "end_user"

class IllegalTransition(RuntimeError):
    """A transition the state machine does not allow (for example, out of a terminal state)."""


class ReviewError(RuntimeError):
    """A review cannot be opened or resolved as asked (another awaiting review, a missing upstream)."""


# ----------------------------------------------------------------------
# Identity surface (ADR-EXP-025 section 8)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """The surface a message arrived on.

    ``channel``: the CLI, or an embedding host's bound channel. Satisfies the ``channel`` class.
    ``host``: FastAPI or MCP. Requires the ``end_user`` class: a verified ``uid`` on a request that
    was signature-checked. Without verification the class cannot be satisfied.
    """

    kind: str = "channel"
    verified: bool = False
    uid: Optional[str] = None
    #: Where a channel-surface caller came from, set only by the server's own construction of the surface
    #: (never by a request): "" is the CLI, an embedding host and the experiment runners (the legacy
    #: standing-approval key); "chat" is the run_chatbot server's channel surface (fix-d6no).
    origin: str = ""

    @property
    def required_identity(self) -> str:
        return IDENTITY_END_USER if self.kind == "host" else IDENTITY_CHANNEL

    def satisfies_end_user(self) -> bool:
        return self.kind == "host" and self.verified and bool(self.uid)


CHANNEL_SURFACE = Surface()


# ----------------------------------------------------------------------
# Canonical JSON, fingerprints, tokens, rendering
# ----------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Sorted keys, no whitespace, UTF-8 text, no NaN. The one canonicaliser used here."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str)


def fingerprint(binding: Any) -> str:
    return hashlib.sha256(canonical_json(binding).encode("utf-8")).hexdigest()


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: Optional[str], expected_hash: str) -> bool:
    return bool(token) and hmac.compare_digest(token_hash(token), expected_hash)


def _value_text(value: Any) -> str:
    """Human-readable value text: true reads as yes, false as no, None as none; lists and maps item by item."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value_text(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_value_text(item)}" for key, item in value.items()) + "}"
    return str(value)


def default_rendering(params: dict) -> str:
    shown = [f"{name}={_value_text(value)}" for name, value in params.items()]
    return ", ".join(shown) if shown else "no parameters"


def render(template: Optional[str], params: dict) -> str:
    """The text the user is asked about: ``template`` (a ``str.format`` string over the values) or the
    default rendering. A template that does not fill from the values falls back to the default."""
    if template:
        try:
            return string.Formatter().vformat(template, (), {k: _value_text(v) for k, v in params.items()})
        except (KeyError, IndexError, ValueError, AttributeError):
            logger.warning("review template did not fill from its values; using the default rendering")
    return default_rendering(params)


def rendering_hash(rendering: str) -> str:
    return hashlib.sha256(rendering.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# The record and the per-session store
# ----------------------------------------------------------------------


@dataclass
class ReviewRecord:
    """One review: an action awaiting its answer, or a proposal under review.

    The first fields are the approval record of ADR-EXP-025 section 7 (unchanged, so stored sessions
    still load). The last three are the review engine's: the kind, the revision count, and the
    upstream records this one was built on (id -> the fingerprint it was built on).
    """

    id: str
    token_hash: str
    command: str
    annotation: Optional[str]
    raw_command: str
    binding: dict
    fingerprint: str
    rendering: str
    rendering_hash: str
    required_identity: str
    created_turn: int
    turns_seen: int
    ttl_turns: int
    ttl_seconds: int
    created_at: float
    state: str
    origin: str
    suspended_agent: bool = False
    queued: bool = False
    shadow: bool = False
    resolution: Optional[dict] = None
    outcome: Optional[str] = None
    kind: str = KIND_ACTION
    revision: int = 0
    upstream: dict = field(default_factory=dict)

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    def time_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def turns_expired(self, *, turn_seq: int) -> bool:
        """Queued records age by turns; awaiting records by the replies they have received."""
        if self.queued:
            return turn_seq - self.created_turn >= self.ttl_turns
        return self.turns_seen >= self.ttl_turns

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class ReviewStore:
    """The review state of one session: records, the turn counter, and the inbound context.

    ``delivery`` holds a freshly minted token until the awaiting output that carries it is built; it is
    never serialised and never reaches the agent.
    """

    turn_seq: int = 0
    records: list = field(default_factory=list)
    surface: Surface = CHANNEL_SURFACE
    inbound_token: Optional[str] = None
    unattended: bool = False
    channel: str = ""
    task: str = ""
    delivery: Optional[dict] = None
    #: The model-based reader of free-text replies for this session's proposal review (fix-3rsq.plan-review.2.7).
    #: Set by the sub-task gate for the turn that opens the review; never serialised (a restored review has none
    #: until the gate of its turn sets it again).
    edit_reader: Optional[Callable[[str, "Proposal"], list]] = None
    plan_cme: Any = None
    plan_workflow: Any = None
    #: A weak reference to the turn object whose waiting record points at this store's records (set when the context binds
    #: its app workflow; ``question_owner``). Never serialised.
    owner_ref: Any = field(default=None, repr=False, compare=False)

    def active(self) -> Optional[ReviewRecord]:
        for record in self.records:
            if record.state in ACTIVE_STATES and not record.queued:
                return record
        return None

    def get(self, record_id: str) -> Optional[ReviewRecord]:
        return next((r for r in self.records if r.id == record_id), None)

    def add(self, record: ReviewRecord) -> None:
        self.records.append(record)
        terminal = [r for r in self.records if r.state not in ACTIVE_STATES]
        excess = len(terminal) - MAX_HISTORY
        if excess > 0:
            drop = {id(r) for r in terminal[:excess]}
            self.records = [r for r in self.records if id(r) not in drop]

    def to_state(self) -> dict:
        return {"turn_seq": self.turn_seq, "records": [r.to_dict() for r in self.records]}

    def load_state(self, state: dict) -> None:
        self.turn_seq = int(state.get("turn_seq") or 0)
        self.records = [ReviewRecord.from_dict(item) for item in state.get("records") or []]


def store_for(workflow: Any) -> Optional[ReviewStore]:
    """The session's store on its app workflow, created on first use; None without a workflow."""
    if workflow is None:
        return None
    store = getattr(workflow, "_approval_store", None)
    if store is None:
        store = ReviewStore()
        try:
            workflow._approval_store = store
        except AttributeError:
            return None
    store._workflow = workflow
    return store


def _prompt_workflow(host: Any, store: ReviewStore) -> Any:
    """The app workflow used to render a sub-task review (writes badges); not persisted on the store."""
    return (getattr(host, "_app_workflow", None)
            or (host if hasattr(host, "folderpath") else None)
            or getattr(store, "_workflow", None))


def question_owner(store: ReviewStore, host: Any) -> Any:
    """The turn object whose waiting record points at *store*'s records: the one the store is bound to, else *host* (a store
    no context bound, as in a unit test of the engine)."""
    ref = store.owner_ref
    return (ref() if ref is not None else None) or host


def active_record(workflow: Any) -> Optional[ReviewRecord]:
    store = store_for(workflow)
    return store.active() if store is not None else None


def clear_inbound(workflow: Any) -> None:
    """The inbound token is for one message: drop it when the turn ends (it is never kept)."""
    store = store_for(workflow)
    if store is not None:
        store.inbound_token = None


def take_delivery(workflow: Any) -> Optional[dict]:
    store = store_for(workflow)
    if store is None or store.delivery is None:
        return None
    delivery, store.delivery = store.delivery, None
    return delivery


# ----------------------------------------------------------------------
# Switch reader
# ----------------------------------------------------------------------

_warned_switch: set[str] = set()


def switch_mode() -> str:
    """``FW_APPROVAL_BINDING``: off | shadow | enforce. An unknown value logs once and enforces."""
    raw = str(fastworkflow.get_env_var(ENV_SWITCH, str, default="") or "").strip().lower()
    if raw == "":
        return DEFAULT_MODE
    if raw in MODES:
        return raw
    if raw not in _warned_switch:
        _warned_switch.add(raw)
        logger.warning(f"{ENV_SWITCH}={raw!r} is not one of off, shadow, enforce; using {DEFAULT_MODE}")
    return DEFAULT_MODE


# ----------------------------------------------------------------------
# Audit and redaction (ADR-EXP-025 section 12). One span per transition.
# ----------------------------------------------------------------------


def audit_attributes(record: ReviewRecord, *, before: str, after: str, reason: str, surface: Surface,
                     mode: str, outcome: str = "") -> dict:
    """The span attributes for one transition. Hashes, states, names and counts only: never a raw
    proposal value, a parameter value, or the token."""
    # The attribute set is the fw.approval contract (tracing.SPAN_CONTRACTS, version 31): a proposal's
    # row is told apart by approval_command (its spec name), so no attribute is added here.
    return {
        "approval_record_id": record.id,
        "approval_command": record.command,
        "approval_annotation": record.annotation or "",
        "approval_state_before": before,
        "approval_state_after": after,
        "approval_reason": reason,
        "approval_outcome": outcome,
        "approval_fingerprint": record.fingerprint,
        "approval_rendering_hash": record.rendering_hash,
        "approval_channel": str(record.binding.get("channel", "")) if isinstance(record.binding, dict) else "",
        "approval_identity_class": record.required_identity,
        "approval_identity_ref": surface.uid if surface.uid else "none",
        "approval_surface": surface.kind,
        "approval_origin": record.origin,
        "approval_mode": mode,
        "approval_shadow": record.shadow,
    }


def audit(host: Any, record: ReviewRecord, *, before: str, after: str, reason: str, surface: Surface,
          mode: str, outcome: str = "") -> None:
    try:
        attributes = audit_attributes(record, before=before, after=after, reason=reason, surface=surface,
                                      mode=mode, outcome=outcome)
        attributes["approval_shadow"] = record.shadow
        span = tracing.start_span(host, tracing.SPAN_APPROVAL, kind=tracing.KIND_INTERNAL,
                                  attributes=attributes, command_name=record.command)
        tracing.end_span(host, span, status=tracing.STATUS_OK)
    except Exception:  # noqa: BLE001 - auditing must never fail the decision it records
        logger.debug("review audit skipped", exc_info=True)


def transition(host: Any, store: ReviewStore, record: ReviewRecord, to: str, *, reason: str,
               surface: Surface, outcome: str = "", resolution: Optional[dict] = None) -> None:
    """Move a record to *to* (one audit row). Illegal moves, including out of a terminal state, raise."""
    before = record.state
    if to not in _LEGAL_TRANSITIONS.get(before, frozenset()):
        raise IllegalTransition(f"{record.id}: {before} -> {to} is not allowed")
    record.state = to
    if resolution is not None:
        record.resolution = resolution
    if outcome:
        record.outcome = outcome
    audit(host, record, before=before, after=to, reason=reason, surface=surface, mode=switch_mode(),
          outcome=outcome)
    if to not in ACTIVE_STATES and (ended := getattr(question_owner(store, host), "_store_record_ended", None)):
        ended(record)    # the turn's waiting record that points at it ends with it: the one place (``waiting``)


def expire_if_due(host: Any, store: ReviewStore, record: ReviewRecord, *, surface: Surface,
                  now: Optional[float] = None) -> bool:
    """Expire an active record whose TTL has passed (turns or wall clock). True when it expired."""
    now = time.time() if now is None else now
    if record.state in ACTIVE_STATES and (record.time_expired(now) or record.turns_expired(turn_seq=store.turn_seq)):
        transition(host, store, record, EXPIRED, reason="ttl", surface=surface, outcome="expired")
        return True
    return False


# ----------------------------------------------------------------------
# Dependencies: link, check, propagate (cascading invalidation)
# ----------------------------------------------------------------------


def link(record: ReviewRecord, upstream: "list[ReviewRecord]") -> None:
    """Record that *record* was built on each upstream record's current fingerprint. An upstream must be
    confirmed (or executed): a review cannot depend on a proposal nobody accepted."""
    for up in upstream:
        if up.state not in (CONFIRMED, EXECUTED):
            raise ReviewError(f"{record.command} depends on {up.command}, which is {up.state}, not confirmed")
        record.upstream[up.id] = up.fingerprint


def stale_upstream(store: ReviewStore, record: ReviewRecord) -> Optional[str]:
    """Why *record* may no longer rest on its upstream records, or None. Fails closed: a missing or
    unconfirmed upstream, or one whose fingerprint has moved, makes the record stale."""
    for up_id, built_on in record.upstream.items():
        up = store.get(up_id)
        if up is None:
            return "an upstream review is no longer held"
        if up.state not in (CONFIRMED, EXECUTED):
            return f"an upstream review is {up.state}"
        if up.fingerprint != built_on:
            return "an upstream review changed"
    return None


def propagate(host: Any, store: ReviewStore, source: ReviewRecord, *, surface: Surface) -> list[ReviewRecord]:
    """After *source* changed (revised, invalidated, declined or expired), invalidate every dependent that
    was built on a fingerprint no longer current, and every dependent of those, transitively.

    Returns the invalidated records. A dependent that is already terminal is left as it is.
    """
    invalidated: list[ReviewRecord] = []
    pending = [source]
    seen: set[str] = {source.id}
    while pending:
        upstream = pending.pop()
        for dependent in list(store.records):
            if dependent.id in seen or upstream.id not in dependent.upstream:
                continue
            if dependent.state not in ACTIVE_STATES and dependent.state != CONFIRMED:
                continue
            gone = upstream.state in (INVALIDATED, DECLINED, EXPIRED)
            moved = dependent.upstream[upstream.id] != upstream.fingerprint
            if not (gone or moved):
                continue
            seen.add(dependent.id)
            transition(host, store, dependent, INVALIDATED,
                       reason=f"upstream {upstream.command} {'is ' + upstream.state if gone else 'changed'}",
                       surface=surface, outcome="invalidated")
            invalidated.append(dependent)
            pending.append(dependent)
    return invalidated


# ----------------------------------------------------------------------
# Typed proposals (sub_tasks and plan)
# ----------------------------------------------------------------------


class Proposal(BaseModel):
    """A typed review proposal: a list of typed items, rendered from a template, fingerprinted.

    A subclass names its kind and spec (the registered command name), its item model in ``items``, the
    line each item renders as (``item_template``, a ``str.format`` string over the item's fields and
    ``index``) and the least number of items it may hold.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ClassVar[str] = KIND_SUB_TASKS
    spec: ClassVar[str] = ""
    title: ClassVar[str] = "Proposal"
    item_template: ClassVar[str] = "{index}. {summary}"
    min_items: ClassVar[int] = 1

    items: list[Any] = Field(default_factory=list)

    def item_values(self, item: Any) -> dict[str, Any]:
        """The values one item's line is formatted from: each field as text. A subclass adds or replaces entries."""
        return {name: _value_text(getattr(item, name)) for name in shown_fields(type(item))}

    def rendering(self) -> str:
        lines = [f"{self.title}:"]
        for index, item in enumerate(self.items, 1):
            values = self.item_values(item)
            values["index"] = index
            try:
                lines.append(string.Formatter().vformat(self.item_template, (), _SafeValues(values)))
            except (ValueError, AttributeError):
                lines.append(f"{index}. " + ", ".join(f"{k}={v}" for k, v in values.items() if k != "index"))
        return "\n".join(lines)

    def revised(self, host: Any, before: "Proposal") -> "Proposal":
        """This proposal after an edit of *before* was applied, as stored and shown: a subclass may bring fields that
        derive from others up to date (the sub-task review re-drafts the inputs of a sub-task whose means changed).
        *host* is the session the reply came through. The default changes nothing."""
        return self

    def binding(self) -> dict:
        return {"spec": self.spec, "kind": self.kind,
                "items": [item.model_dump(mode="json") for item in self.items]}

    def fingerprint(self) -> str:
        return fingerprint(self.binding())


class _SafeValues(dict):
    """A values map for ``str.format`` that renders a missing field as the field name, not an error."""

    def __missing__(self, key: str) -> str:
        return f"<{key}?>"


PROPOSAL_SPECS: dict[str, type[Proposal]] = {}


def register_proposal(cls: type[Proposal]) -> type[Proposal]:
    """Register a proposal spec under its command name (``spec``), so a stored record rebuilds it."""
    if not cls.spec:
        raise ValueError(f"{cls.__name__} declares no spec name")
    PROPOSAL_SPECS[cls.spec] = cls
    return cls


_approved_review_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "fw_review_approved_record", default=None)


@contextlib.contextmanager
def approved_review(record_id: str):
    """Mark the execution of one accepted proposal: the review command returns its typed result."""
    token = _approved_review_var.set(record_id)
    try:
        yield
    finally:
        _approved_review_var.reset(token)


def approved_review_proposal(workflow: Any, spec: str) -> Optional[Proposal]:
    """The accepted proposal of *spec* that is being executed now, or None outside that execution."""
    record_id = _approved_review_var.get()
    store = store_for(workflow)
    record = store.get(record_id) if (record_id and store is not None) else None
    if record is None or record.state != CONFIRMED or record.command != spec:
        return None
    return proposal_of(record)


def open_for_workflow(workflow: Any, proposal: Proposal, *, items_text: str = "") -> ReviewRecord:
    """Open a proposal review on a workflow's session, suspending the agent call on the prompt."""
    from fastworkflow import tracing

    store = store_for(workflow)
    if store is None:
        raise ReviewError("the workflow has no session store for a review")
    return open_review(tracing.current_host(), store, proposal, surface=store.surface, origin="agent",
                       suspend=True, items_text=items_text)


def proposal_of(record: ReviewRecord) -> Proposal:
    """The typed proposal a proposal record holds, rebuilt (and so re-validated) from its binding."""
    import fastworkflow.review_proposals  # noqa: F401 - registers the framework's proposal specs

    cls = PROPOSAL_SPECS.get(record.command)
    if cls is None:
        raise ReviewError(f"no proposal spec is registered as {record.command!r}")
    return cls.model_validate({"items": record.binding.get("items") or []})


def ttl_for_kind(kind: str) -> tuple[int, int]:
    """The ``(reply cap, seconds)`` a record of *kind* gets: approvals keep TTL_TURNS, proposal reviews their own."""
    if kind == KIND_ACTION:
        return TTL_TURNS, TTL_SECONDS
    return PROPOSAL_TTL_TURNS, PROPOSAL_TTL_SECONDS


PROMPT_HINT = ("Reply accept to confirm, no to decline, or edit with 'change N: field=value' or 'drop N' "
               "(N is the number shown; separate several edits with ';').")

SUB_TASK_PROMPT_HINT = (
    "Reply yes to run this, or tell me what to change "
    "(put single or double quotes around multi-word names, e.g. 'Henderson Ramirez and Lewis and Sons')."
)

SUB_TASK_UNREADABLE_REPLY = (
    "I couldn't tell what to change. Say which step and what it should be "
    "(put single or double quotes around multi-word names), or reply yes to run it as shown."
)

SUB_TASK_STRUCTURED_PROMPT_HINT = (
    "Reply accept to confirm, no to decline, or edit with "
    "'change N: inputs=key=value', 'change N: goal=text', 'add after N: goal=text', or 'drop N' "
    "(N is the step number, 0 to add before step 1, and key is a field name shown in parentheses; "
    "separate several edits with ';')."
)

SUB_TASK_STRUCTURED_UNREADABLE_REPLY = (
    f"I couldn't tell what to change. {SUB_TASK_STRUCTURED_PROMPT_HINT}"
)

#: The replies each prompt hint promises, as data (fix-szfq). One entry per hint constant above, in the order the hint
#: text names them; a ``reply`` is the literal word the hint says to send, a ``pattern`` the edit form it spells
#: (``N`` is the number shown), ``per_item`` is True for a form that names an item, ``free_text`` for "tell me what to
#: change". These are what the HINT TEXT promises: the reply reader's own cues (``_ACCEPT_PHRASES``, ``_DECLINE_PHRASES``,
#: the edit parsers) are separate and wider, and are not derived from this table.
REPLY_FORMS_OF_HINT: tuple[tuple[str, tuple[dict[str, Any], ...]], ...] = (
    (SUB_TASK_STRUCTURED_PROMPT_HINT, (
        {"kind": "accept", "reply": "accept"},
        {"kind": "decline", "reply": "no"},
        {"kind": "edit", "pattern": "change N: inputs=key=value", "per_item": True},
        {"kind": "edit", "pattern": "change N: goal=text", "per_item": True},
        {"kind": "add", "pattern": "add after N: goal=text", "per_item": True},
        {"kind": "drop", "pattern": "drop N", "per_item": True},
    )),
    (SUB_TASK_PROMPT_HINT, (
        {"kind": "accept", "reply": "yes"},
        {"kind": "edit", "free_text": True, "per_item": False},
    )),
    (PROMPT_HINT, (
        {"kind": "accept", "reply": "accept"},
        {"kind": "decline", "reply": "no"},
        {"kind": "edit", "pattern": "change N: field=value", "per_item": True},
        {"kind": "drop", "pattern": "drop N", "per_item": True},
    )),
)

#: What separates several edits in one reply, where the hint says so (every hint with an edit pattern).
REPLY_EDIT_SEPARATOR = ";"


def reply_forms(rendering: str, proposal: Proposal, *, structured: bool = False) -> list[dict[str, Any]]:
    """The reply forms the hint of the prompt shown (*rendering*) promises, as data, in the hint's order.

    The hint is found in *rendering* (the text the user was shown, which ends with it); a text that carries none
    (a refusal message) falls back to the hint ``prompt_for`` would pick for *proposal* and *structured*. Each form is
    ``{kind, reply | pattern | free_text, per_item}``, and an edit form with a pattern also carries ``separator``."""
    from fastworkflow.review_proposals import SubTaskProposal

    chosen = next((forms for hint, forms in REPLY_FORMS_OF_HINT if hint in (rendering or "")), None)
    if chosen is None:
        hint = ((SUB_TASK_STRUCTURED_PROMPT_HINT if structured else SUB_TASK_PROMPT_HINT)
                if isinstance(proposal, SubTaskProposal) else PROMPT_HINT)
        chosen = next(forms for text, forms in REPLY_FORMS_OF_HINT if text == hint)
    forms = [{**form, **({"separator": REPLY_EDIT_SEPARATOR} if "pattern" in form else {})} for form in chosen]
    return forms


def _step_count_message(count: int) -> str:
    label = "step" if count == 1 else "steps"
    return f"There are only {count} {label}."


def _no_agent_of(host: Any) -> bool:
    """Whether *host* (the session a review is on) is in no-agent mode; False for anything else."""
    return getattr(host, "no_agent_mode", False) is True


def prompt_for(proposal: Proposal, workflow: Any = None, *, structured: bool = False, no_agent: bool = False) -> str:
    from fastworkflow.review_proposals import SubTaskProposal

    folder = getattr(workflow, "folderpath", None) if workflow is not None else None
    if isinstance(proposal, SubTaskProposal):
        hint = SUB_TASK_STRUCTURED_PROMPT_HINT if structured else SUB_TASK_PROMPT_HINT
        return f"{proposal.user_rendering(folder, no_agent=no_agent)}\n{hint}"
    return f"{proposal.rendering()}\n{PROMPT_HINT}"


def _structured_error_prompt(proposal: Proposal, workflow: Any = None, message: str = "", *,
                             no_agent: bool = False) -> str:
    from fastworkflow.review_proposals import SubTaskProposal

    if isinstance(proposal, SubTaskProposal):
        if not message or message in (SUB_TASK_UNREADABLE_REPLY, SUB_TASK_STRUCTURED_UNREADABLE_REPLY):
            header = "I couldn't tell what to change."
        else:
            header = message
        folder = getattr(workflow, "folderpath", None) if workflow is not None else None
        current_steps = proposal.user_rendering(folder, no_agent=no_agent).replace("Sub-tasks:", "Current steps:", 1)
        return f"{header}\n{current_steps}\n{SUB_TASK_STRUCTURED_PROMPT_HINT}"
    return message or PROMPT_HINT


def _input_signature(inputs: Any) -> list[tuple]:
    """What an input list says, for comparing two of them: each input's name, value and source sub-task."""
    return [(each.name, each.value, each.from_subtask) for each in inputs]


def _normalize_parsed_subtask_edits(proposal: Proposal, edits: list[Edit]) -> list[Edit]:
    from fastworkflow.review_proposals import (
        SubTaskProposal, declared_input_names, merge_subtask_input_edit, unreadable_input_value)

    if not isinstance(proposal, SubTaskProposal):
        return edits
    normalized: list[Edit] = []
    for edit in edits:
        if isinstance(edit, Change) and edit.field not in (
                "goal", "outcome", "inputs", "means", "check", "drafted_means"):
            on_item = (
                1 <= edit.index <= len(proposal.items)
                and any(inp.name == edit.field for inp in proposal.items[edit.index - 1].inputs)
            )
            if on_item or edit.field in declared_input_names():
                normalized.append(
                    Change(index=edit.index, field="inputs", value=f"{edit.field}={edit.value}"))
            else:
                normalized.append(edit)
        else:
            normalized.append(edit)
    inputs_positions: dict[int, list[int]] = {}
    for position, edit in enumerate(normalized):
        if isinstance(edit, Change) and edit.field == "inputs":
            inputs_positions.setdefault(edit.index, []).append(position)
    to_remove: set[int] = set()
    for index, positions in inputs_positions.items():
        if len(positions) <= 1:
            continue
        if not 1 <= index <= len(proposal.items):
            continue  # apply_edits refuses it ("There are only N steps"), so nothing is merged for it
        values = [normalized[position].value for position in positions]
        if any(unreadable_input_value(value) for value in values):
            continue  # left as written: apply_edits refuses it with the message (fix-2vyv)
        existing = list(proposal.items[index - 1].inputs)
        merged_inputs = existing
        for value in values:
            merged_inputs = merge_subtask_input_edit(merged_inputs, value)
        joined = ", ".join(item.render() for item in merged_inputs)
        if _input_signature(merge_subtask_input_edit(existing, joined)) != _input_signature(merged_inputs):
            # The merge cannot be written back as one text without losing a part (a goal, outcome, means or check
            # input, a quoted value that would split again, a cleared input): leave the edits as written, one per
            # input. apply_edits applies them in order, each over the one before, and nothing is dropped (fix-2vyv).
            continue
        normalized[positions[0]] = Change(index=index, field="inputs", value=joined)
        to_remove.update(positions[1:])
    if not to_remove:
        return normalized
    return [edit for position, edit in enumerate(normalized) if position not in to_remove]


def _refuse_dropped_input_parts(proposal: Proposal, edits: list[Edit]) -> None:
    """Typed edits only (fix-j6xa): refuse a packed ``inputs`` edit whose part named like a step field would be dropped.

    ``change 1: inputs=member=Z, check=x`` is what a person typed, so losing ``check=x`` without a word is refused with
    the part named. (A model's reading of a plain-language reply is not checked: it echoes the shown step's means and
    outcome into its inputs text, and dropping those is what keeps that reading correct.)
    """
    from fastworkflow.review_proposals import SubTaskProposal, dropped_input_part, merge_subtask_input_edit

    if not isinstance(proposal, SubTaskProposal):
        return
    current = {number: list(item.inputs) for number, item in enumerate(proposal.items, 1)}
    for edit in edits:
        if not (isinstance(edit, Change) and edit.field == "inputs" and edit.index in current):
            continue
        if (lost := dropped_input_part(current[edit.index], edit.value)) is not None:
            raise EditRefused(
                f"Item {edit.index}: '{lost}' inside a list of inputs would be dropped, not read as an input. "
                f"Give it as its own edit: 'change {edit.index}: inputs={lost}' sets an input of that name, "
                f"'change {edit.index}: {lost}' changes the step's field of that name.")
        current[edit.index] = merge_subtask_input_edit(current[edit.index], edit.value)


# ----------------------------------------------------------------------
# Typed edits: parsed deterministically, validated against the schema, then applied
# ----------------------------------------------------------------------


class EditParseError(ValueError):
    """A reply the deterministic reader does not understand as edits (the message says how to write them)."""


class EditRefused(ValueError):
    """Edits that do not validate against the proposal's schema. The message is shown to the user."""


class Change(BaseModel):
    """Set one field of one item (``change 2: field=value``). ``index`` is 1-based."""

    model_config = ConfigDict(extra="forbid")
    index: int
    field: str
    value: str


class Drop(BaseModel):
    """Remove one item (``drop 3``). ``index`` is 1-based."""

    model_config = ConfigDict(extra="forbid")
    index: int


class Add(BaseModel):
    """Insert one new item after item ``after`` (0: before the first). ``values`` are the item's fields as text."""

    model_config = ConfigDict(extra="forbid")
    after: int
    values: dict[str, str] = Field(default_factory=dict)


class Merge(BaseModel):
    """Join items ``indexes`` (at least two, 1-based) into one at the first's place. The first item's fields are
    kept, and ``values`` (text) override them."""

    model_config = ConfigDict(extra="forbid")
    indexes: list[int]
    values: dict[str, str] = Field(default_factory=dict)


class Split(BaseModel):
    """Replace item ``index`` by one item per entry of ``parts``; each part is the item's full field set as text."""

    model_config = ConfigDict(extra="forbid")
    index: int
    parts: list[dict[str, str]]


class LookUp(BaseModel):
    """Re-draft item ``index`` for look_up (goal-preserving; applied in ``SubTaskProposal.revised``). ``index`` is 1-based."""

    model_config = ConfigDict(extra="forbid")
    index: int


Edit = Union[Change, Drop, Add, Merge, Split, LookUp]

_CHANGE = re.compile(r"^change\s+(\d+)\s*:?\s*([A-Za-z_]\w*)\s*=\s*(.*)$", re.IGNORECASE | re.DOTALL)
_DROP = re.compile(r"^drop\s+(\d+)$", re.IGNORECASE)
_ADD = re.compile(
    r"^add\s+(?:after\s+)?(\d+)\s*:?\s*(?:goal\s*=\s*)?(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_BARE_ASSIGN = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.*)$", re.DOTALL)


def parse_edits(text: str) -> list[Edit]:
    """The deterministic reader: ``change N: field=value`` and ``drop N``, separated by ``;`` or new lines.

    Any piece it cannot read fails the whole reply (all or nothing), naming the piece.
    """
    pieces = [piece.strip() for piece in re.split(r"[;\n]", text or "") if piece.strip()]
    if not pieces:
        raise EditParseError("The reply was empty.")
    edits: list[Edit] = []
    last_index: Optional[int] = None
    last_was_add = False
    for piece in pieces:
        if match := _CHANGE.match(piece):
            last_index = int(match.group(1))
            last_was_add = False
            edits.append(Change(index=last_index, field=match.group(2), value=match.group(3).strip()))
        elif match := _DROP.match(piece):
            last_index = None
            last_was_add = False
            edits.append(Drop(index=int(match.group(1))))
        elif match := _ADD.match(piece):
            last_index = None
            last_was_add = True
            goal_value = match.group(2).strip()
            if not goal_value:
                raise EditParseError("An added sub-task needs its goal.")
            edits.append(Add(after=int(match.group(1)), values={"goal": goal_value}))
        elif (match := _BARE_ASSIGN.match(piece)) and (last_index is not None or last_was_add):
            if last_index is not None:
                edits.append(Change(index=last_index, field=match.group(1), value=match.group(2).strip()))
            elif match.group(1).lower() == "goal" and edits and isinstance(edits[-1], Add):
                edits[-1] = Add(after=edits[-1].after, values={"goal": match.group(2).strip()})
        else:
            raise EditParseError(f"I cannot read {piece!r}. Write 'change N: field=value' or 'drop N'.")
    return edits


def _coerce(model: type[BaseModel], name: str, text: str) -> Any:
    """A field's text as the field's value: a list field reads comma-separated items; 'none' is None."""
    annotation = model.model_fields[name].annotation
    if text.strip().lower() == "none":
        return None
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def _item_model(proposal: Proposal):
    """The class of one item of *proposal* (its ``items`` element type), or None when it is not a model."""
    from typing import get_args

    args = get_args(type(proposal).model_fields["items"].annotation)
    model = args[0] if args else None
    return model if isinstance(model, type) and issubclass(model, BaseModel) else None


def shown_fields(model) -> list[str]:
    """The fields of an item model a user sees and edits: every field but one marked
    ``json_schema_extra={"review_hidden": True}`` (framework bookkeeping, fix-48np.1)."""
    return [name for name, info in model.model_fields.items()
            if not (isinstance(info.json_schema_extra, dict) and info.json_schema_extra.get("review_hidden"))]


def _built_item(model, values: dict[str, str], base: Optional[BaseModel] = None):
    """A new item of *model*: *base*'s fields (if any) with *values* (text) set over them. Refused when it does not validate."""
    data = base.model_dump() if base is not None else {}
    for name, text in values.items():
        if name not in shown_fields(model):
            raise EditRefused(f"An item has no field {name!r}. Its fields are: {', '.join(shown_fields(model))}.")
        data[name] = _coerce(model, name, text)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise EditRefused(f"A new item is not valid: {exc.errors()[0]['msg']}.") from exc


def apply_edits(proposal: Proposal, edits: list[Edit]) -> Proposal:
    """The proposal with *edits* applied, validated against the item schema and the proposal's rules.

    Every index refers to the proposal as it is shown (before this reply). An item may be changed, or it may be
    dropped, merged or split, but not two of these in one reply. Adds insert after a shown item (0: before the first).
    Raises :class:`EditRefused` with a message for the user; the original proposal is never modified.
    """
    items = list(proposal.items)
    count = len(items)
    model = _item_model(proposal)

    def number_ok(number: int, low: int = 1) -> None:
        if not low <= number <= count:
            raise EditRefused(_step_count_message(count))

    whole_item: set[int] = set()
    field_change: dict[tuple[int, str], str] = {}

    def claim_whole(number: int) -> None:
        if number in whole_item or any(index == number for index, _ in field_change):
            raise EditRefused(f"Item {number} is named by more than one edit.")

    def claim_field(number: int, field_name: str) -> None:
        if number in whole_item:
            raise EditRefused(f"Item {number} is named by more than one edit.")
        key = (number, field_name)
        if key in field_change:
            raise EditRefused(f"Item {number}'s {field_name!r} is named by more than one edit.")
        field_change[key] = "change"

    def _inputs_claim_keys(number: int, value: str) -> list[str]:
        from fastworkflow.review_proposals import SubTaskInput, _input_edit_parts

        stripped = value.strip()
        if stripped.lower() == "none":
            return ["inputs"]
        parts = _input_edit_parts(stripped)
        names = [SubTaskInput.from_text(part).name for part in parts]
        if names and all(names):
            return [f"inputs:{name}" for name in names]
        return ["inputs"]

    def claim_inputs(number: int, value: str) -> None:
        if number in whole_item:
            raise EditRefused(f"Item {number} is named by more than one edit.")
        from fastworkflow.review_proposals import unreadable_input_value

        if (problem := unreadable_input_value(value)) is not None:
            raise EditRefused(f"Item {number}: {problem}")
        for claim in _inputs_claim_keys(number, value):
            if claim == "inputs":
                for key in field_change:
                    if key[0] == number and (key[1] == "inputs" or key[1].startswith("inputs:")):
                        raise EditRefused(f"Item {number}'s inputs are named by more than one edit.")
                field_change[(number, "inputs")] = "change"
                return
            if (number, "inputs") in field_change:
                raise EditRefused(f"Item {number}'s inputs are named by more than one edit.")
            key = (number, claim)
            if key in field_change:
                label = claim.split(":", 1)[1]
                raise EditRefused(f"Item {number}'s input {label!r} is named by more than one edit.")
            field_change[key] = "change"

    for edit in edits:
        if isinstance(edit, Change):
            number_ok(edit.index)
            if edit.field == "inputs" and model is not None and model.__name__ == "SubTask":
                claim_inputs(edit.index, edit.value)
            else:
                claim_field(edit.index, edit.field)
        elif isinstance(edit, Drop):
            number_ok(edit.index)
            claim_whole(edit.index)
            whole_item.add(edit.index)
        elif isinstance(edit, Merge):
            if len(edit.indexes) < 2 or len(set(edit.indexes)) != len(edit.indexes):
                raise EditRefused("A merge joins at least two different items.")
            for number in edit.indexes:
                number_ok(number)
                claim_whole(number)
            whole_item.update(edit.indexes)
        elif isinstance(edit, Split):
            number_ok(edit.index)
            if not edit.parts:
                raise EditRefused(f"Item {edit.index} cannot be split into no parts.")
            claim_whole(edit.index)
            whole_item.add(edit.index)
        elif isinstance(edit, LookUp):
            number_ok(edit.index)
            claim_field(edit.index, "look_up")
        elif isinstance(edit, Add):
            number_ok(edit.after, low=0)
            if model is None:
                raise EditRefused("This proposal takes no added items.")

    look_up_at = {edit.index for edit in edits if isinstance(edit, LookUp)}
    changed = list(items)
    for edit in edits:
        if not isinstance(edit, Change):
            continue
        position = edit.index - 1
        item_model = type(changed[position])
        if edit.field not in shown_fields(item_model):
            raise EditRefused(f"Item {edit.index} has no field {edit.field!r}. Its fields are: "
                              f"{', '.join(shown_fields(item_model))}.")
        data = changed[position].model_dump()
        if edit.field == "inputs" and item_model.__name__ == "SubTask":
            from fastworkflow.review_proposals import merge_subtask_input_edit

            before_named = {
                each.name: each.value for each in changed[position].inputs if each.name
            }
            data["inputs"] = merge_subtask_input_edit(changed[position].inputs, edit.value)
            after_named = {
                each.name: each.value for each in data["inputs"] if each.name
            }
            old_name = before_named.get("name")
            new_name = after_named.get("name")
            goal_text = data.get("goal", changed[position].goal)
            if (
                old_name
                and new_name
                and old_name != new_name
                and old_name in goal_text
                and new_name not in goal_text
            ):
                data["goal"] = goal_text.replace(old_name, new_name)
        else:
            data[edit.field] = _coerce(item_model, edit.field, edit.value)
        try:
            changed[position] = item_model.model_validate(data)
        except ValidationError as exc:
            detail = exc.errors()[0]["msg"]
            if edit.value.strip().lower() == "none":
                detail += " ('none' means no value, so it cannot be the text of this field)"
            raise EditRefused(f"{edit.field} of item {edit.index} is not valid: {detail}.") from exc

    dropped = {edit.index for edit in edits if isinstance(edit, Drop)}
    merged: dict[int, Any] = {}
    for edit in edits:
        if isinstance(edit, Merge):
            first = min(edit.indexes)
            merged[first] = _built_item(type(changed[first - 1]), edit.values, changed[first - 1])
            dropped.update(number for number in edit.indexes if number != first)
    split: dict[int, list] = {}
    for edit in edits:
        if isinstance(edit, Split):
            split[edit.index] = [_built_item(type(changed[edit.index - 1]), part) for part in edit.parts]
    after: dict[int, list] = {}
    for edit in edits:
        if isinstance(edit, Add):
            after.setdefault(edit.after, []).append(_built_item(model, edit.values))

    result: list[Any] = []
    provenance: list[tuple[Any, Any, bool]] = []
    for new_item in after.get(0, []):
        result.append(new_item)
        provenance.append((new_item, None, False))
    for number in range(1, count + 1):
        if number not in dropped:
            look_up = number in look_up_at
            if number in split:
                for part in split[number]:
                    result.append(part)
                    provenance.append((part, items[number - 1], look_up))
            elif number in merged:
                result.append(merged[number])
                provenance.append((merged[number], items[number - 1], look_up))
            else:
                result.append(changed[number - 1])
                provenance.append((changed[number - 1], items[number - 1], look_up))
        for new_item in after.get(number, []):
            result.append(new_item)
            provenance.append((new_item, None, False))
    if len(result) < type(proposal).min_items:
        raise EditRefused(f"The proposal needs at least {type(proposal).min_items} item(s).")
    try:
        edited = type(proposal).model_validate({"items": result})
    except ValidationError as exc:
        raise EditRefused(f"The edited proposal is not valid: {exc.errors()[0]['msg']}.") from exc
    if model is not None and model.__name__ == "SubTask":
        from fastworkflow.review_proposals import SubTask

        for item, (_, before, look_up) in zip(edited.items, provenance):
            if isinstance(item, SubTask):
                item._before_item = before
                item._look_up = look_up
    return edited


# ----------------------------------------------------------------------
# Reply policy: a graded decision per reply (the policy decision point for replies)
# ----------------------------------------------------------------------


class Verdict(str, Enum):
    CONFIRM = "confirm"      # accept (proposals) / yes (action)
    ALWAYS = "always"        # yes, and remember it (action only)
    DECLINE = "decline"      # no
    EDIT = "edit"            # typed edits to apply, then re-render (proposals)
    REFUSE = "refuse"        # edits that do not validate: nothing changes, the record stays open
    REPROMPT = "reprompt"    # not a reply the framework reads: ask again, nothing changes


@dataclass(frozen=True)
class ReplyDecision:
    verdict: Verdict
    reason: str
    edits: tuple = ()
    message: str = ""
    #: Why the reader did not act on the classifier's answer (the polarity guard, bd fix-blui); empty otherwise.
    note: str = ""


#: A model-based reader of free-text replies: ``(text, proposal) -> list of typed edits``. Not registered by
#: default; tests never register one. It runs only after the deterministic reader has failed, and its
#: edits are validated exactly as the deterministic ones are.
EditInterpreter = Callable[[str, Proposal], list]
_model_interpreter: Optional[EditInterpreter] = None


def set_model_interpreter(interpreter: Optional[EditInterpreter]) -> None:
    """Register (or with None, remove) the model-based edit interpreter. Injection point only."""
    global _model_interpreter
    _model_interpreter = interpreter


class ActionReplyPolicy:
    """The action kind: exact yes, no or always. No model, no edits."""

    def classify(self, message: str) -> Verdict:
        word = message
        if word == "yes":
            return Verdict.CONFIRM
        if word == "no":
            return Verdict.DECLINE
        if word == "always":
            return Verdict.ALWAYS
        return Verdict.REPROMPT


#: The reason of a decision whose edits a model read from free text (shown back to the user as read).
READ_REASON = "edits read from the reply"


def _review_planning_names(app_workflow: Any) -> tuple[str, ...]:
    if app_workflow is None:
        return ()
    folder = getattr(app_workflow, "folderpath", None)
    if not folder:
        return ()
    from fastworkflow import planning_commands

    return tuple(sorted(planning_commands.resolve_field_sources(folder).planning_commands))


def _routed_to_command(w_out: Any) -> bool:
    """True when the wildcard result routes on to a named command.

    An answer flagged routed with no command name (an ambiguous or wildcard head answer that
    the framework still reports as handled) names nothing to run, so it is unrouted: the reader
    asks again instead of acting (bd fix-1hs8; the reply "nope" used to raise ValueError).
    """
    return bool(getattr(w_out, "routed", False)) and bool(getattr(w_out, "command_name", None))


def _slot_question(proposal: Proposal) -> str:
    return f"Which of the {len(proposal.items)} steps do you mean?"


def _clear_plan_cme_continuation(cme: Any) -> None:
    cme.end_command_processing()
    context = cme.context
    for key in ("command", "command_name"):
        context.pop(key, None)
    cme.context = context


def _prepare_plan_workflow(policy: "ProposalReplyPolicy", proposal: Proposal) -> None:
    from fastworkflow._workflows.plan_metadata_extraction._commands.ProposalReview._ProposalReview import ProposalReview

    workflow = policy._plan_workflow
    if workflow._current_command_context_name != "ProposalReview":
        workflow.record = ProposalReview()
    workflow.context["plan_proposal"] = proposal
    workflow.context["plan_model_rendering"] = proposal.model_rendering()
    workflow.context["review_planning_names"] = _review_planning_names(policy._app_workflow)
    reader = policy._interpreter or _model_interpreter
    if reader is not None:
        workflow.context["plan_edit_reader"] = reader
    else:
        workflow.context.pop("plan_edit_reader", None)


def _validate_edits_decision(
    proposal: Proposal,
    edits: tuple,
    *,
    reason: str,
    text: str | None = None,
    workflow: Any = None,
) -> ReplyDecision:
    try:
        edited = apply_edits(proposal, list(edits))
    except EditRefused as refused:
        return ReplyDecision(Verdict.REFUSE, "edits refused", message=str(refused))
    if workflow is not None and reason == READ_REASON:
        # Edits a model read from free text may set an enumerated input outside its choices (fix-j9qk): refused through
        # the same path as any invalid edit, so the user is asked again. Typed and form edits are not checked here.
        from fastworkflow.review_proposals import refused_choice_message

        if (message := refused_choice_message(workflow, edited)) is not None:
            return ReplyDecision(Verdict.REFUSE, "edits refused", message=message)
    if reason == READ_REASON and edited.binding() == proposal.binding():
        if text and _affirmative_accept_cue(text):
            return ReplyDecision(Verdict.CONFIRM, "accepted")
        return ReplyDecision(Verdict.REPROMPT, "not read as edits", message=SUB_TASK_UNREADABLE_REPLY)
    return ReplyDecision(Verdict.EDIT, reason, edits=edits)


_AFFIRMATIVE_ACCEPT_PREFIX = re.compile(
    r"^\s*(?:yes|yep|yeah|sure|ok|okay|accept|approve|go\s+ahead|sounds\s+good|looks\s+good)\b",
    re.I,
)
_AFFIRMATIVE_UNROUTED_PREFIX = re.compile(
    r"^\s*(?:yes|yep|yeah|accept|approve|go\s+ahead|sounds\s+good|looks\s+good)\b",
    re.I,
)
_CONFIRM_CORRECTIVE = re.compile(
    r"\b(?:"
    r"but\s+(?:in\s+step|for\s+step|step\s+\d+|also|instead|change|make|use|set|drop|remove|add|only|"
    r"do\s+not|don'?t|not\s+just|not\s+only|(?:the|that|this)\s+(?:step|plan|list)\s+(?:must|should|needs?|has\s+to))|"
    r"except|however|though|although|instead|also|not\s+just|not\s+only|only\s+want|"
    r"change|drop|"
    r"add\s+(?:a\s+)?(?:step|item|sub-?task|another)|"
    r"remove\s+(?:step|item|sub-?task|the\s+(?:first|second|third|fourth|fifth|last)|#\d+|\d+\b)|"
    r"replace|update|delete|skip|omit|"
    r"in\s+step\s+\d+|for\s+step\s+\d+|step\s+\d+\s+should)\b",
    re.I,
)
_BENIGN_BUT_NOT_APPLIED = re.compile(
    r"\b(?:shown|listed|available)\s+but\s+not\s+applied\b",
    re.I,
)
_SLOT_WHOLE_PLAN_CONFIRM = re.compile(
    r"\b(?:confirm(?:ing)?\s+(?:the\s+)?(?:whole\s+|entire\s+)?plan|"
    r"not\s+changing\s+(?:either|any)|run\s+it\s+as\s+shown|accept\s+(?:the\s+)?(?:whole\s+)?plan)\b",
    re.I,
)
_AFFIRMATIVE_PARAPHRASE = re.compile(
    r"\b(?:run\s+it\s+as\s+shown|keep\s+it\s+as\s+(?:is|shown)|that\s+is\s+right|"
    r"that\s+matches\s+what\s+i\s+want)\b",
    re.I,
)


#: A qualifier on a yes that asks for no edit: the plan's steps only read, and the user says nothing is to be applied
#: (``yes, go ahead, but do not run any of the actions``, ``yes, and apply no corrections``). It names what must not be
#: changed (actions, corrections, anything), never a step, so it does not edit the plan.
_NO_EFFECT_OBJECT = (
    r"(?:any|anything|none|nothing|no)(?:\s+of)?(?:\s+(?:the|these|those))?"
    r"(?:\s+(?:actions?|corrections?|remediations?|fixes|changes|updates|writes|modifications?))?"
    r"|(?:the\s+)?(?:actions?|corrections?|remediations?|fixes|changes|updates)")
_NO_EFFECT_QUALIFIER = re.compile(
    r"[,;.]?\s*(?:(?:but|and|just|only)\s+)*(?:"
    rf"(?:do\s+not|don'?t|never)\s+(?:run|apply|execute|perform|make|record|write|modify|change)\s+(?:{_NO_EFFECT_OBJECT})"
    rf"|(?:run|apply|make)\s+no\s+(?:actions?|corrections?|remediations?|fixes|changes|updates)"
    r")\s*[.!]?\s*$",
    re.I,
)


def _scrub_benign_confirm_phrases(text: str) -> str:
    return _NO_EFFECT_QUALIFIER.sub("", _BENIGN_BUT_NOT_APPLIED.sub("", text or ""))


_EDIT_CLAUSE = re.compile(r"^(?:change\s+\d+\s*:?\s*[A-Za-z_]\w*\s*=|drop\s+\d+$|add\s+after\s+\d+)", re.IGNORECASE)


def _has_edit_clause(text: str) -> bool:
    """True when a ``;`` or new-line clause of *text* opens as a typed edit (``change N: field=``, ``drop N``,
    ``add after N``): the person wrote an edit, so no accept cue in the reply may swallow it (fix-2vyv)."""
    return any(_EDIT_CLAUSE.match(piece.strip()) for piece in re.split(r"[;\n]", text or ""))


def _affirmative_accept_cue(text: str) -> bool:
    """Whether *text* accepts the proposal. A reply with a typed edit clause never does: a cue inside an edit's value
    (``change 1: goal=Run it as shown``) is part of the value, and a cue beside an edit would drop the edit unread, so
    such a reply goes to the edit reader, which applies it or asks again. Send an acceptance on its own."""
    scrubbed = _scrub_benign_confirm_phrases(text)
    if _has_edit_clause(scrubbed):
        return False
    if _accept_then_request(text):
        return False
    if _AFFIRMATIVE_PARAPHRASE.search(scrubbed):
        return True
    if _WHOLE_ACCEPT.match(_lexical(text)) and not _NO_COMMA_CHANGE.match(text or ""):
        return True
    if _CONFIRM_CORRECTIVE.search(scrubbed):
        return False
    if _AFFIRMATIVE_ACCEPT_PREFIX.match(scrubbed):
        return True
    return False


# ---- plain accept and decline cues, and the polarity guard (bd fix-blui) --------------------------------------
# A cue is anchored: the WHOLE reply (after lower-casing, dropping apostrophes and punctuation, and reading a
# thumbs-up emoji as the word ``thumbsup``) is one cue phrase, optionally followed by courtesy words, never a phrase
# found somewhere inside a sentence. ``go`` does not match ``go back and change step 2``, ``pass`` does not match
# ``pass the id to step 3``, ``stop`` does not match ``stop after step 2``, ``fine`` does not match ``fine, but drop
# the last one``: the rest of the reply is not a courtesy word, so the reply is not a cue and goes on to the
# classifier and the edit reader. No fuzzy or typo matching: the reader has no typo mechanism, and a misspelling
# asks again (``proced``).

_THUMBS_UP = re.compile("\U0001F44D[\U0001F3FB-\U0001F3FF]?️?")
_APOSTROPHES = re.compile("['’`´]")


def _lexical(text: str, *, scrub: bool = False) -> str:
    """*text* as space-separated lower-case words: apostrophes dropped, other punctuation a space, a thumbs-up
    emoji the word ``thumbsup``."""
    value = _scrub_benign_confirm_phrases(text) if scrub else (text or "")
    value = _APOSTROPHES.sub("", _THUMBS_UP.sub(" thumbsup ", value.lower()))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _alternation(phrases: Sequence[str]) -> str:
    return "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))


_ACCEPT_PHRASES = (
    "do it", "run it", "ship it", "go", "go for it", "lets go", "lets do it", "proceed", "continue", "carry on",
    "make it so", "lgtm", "all good", "that works", "works for me", "fine", "its fine", "thats fine", "k", "kk",
    "nothing to change", "nothing needs to change", "no changes", "no change", "no edits", "no objections",
    "no problem", "no problems", "as is", "run as is", "run it as is", "run the plan as is",
    "dont change anything", "thumbsup",
)
_ACCEPT_TAIL = (
    "please", "thanks", "thank you", "thx", "ty", "now", "thumbsup", "go ahead", "ahead", "proceed", "continue",
    "with it", "do it", "run it", "needed", "as is", "to me",
)
_WHOLE_ACCEPT = re.compile(
    rf"^(?:please\s+)?(?:{_alternation(_ACCEPT_PHRASES)})(?:\s+(?:{_alternation(_ACCEPT_TAIL)}))*$")
#: ``no, change the second one`` reads as the words ``no change ...``; the comma is the difference, and it is an edit.
_NO_COMMA_CHANGE = re.compile(r"^\W*no\s*[,;:\-–—]\s*change", re.I)


# ---- an accept that goes on to ask for something more (bd fix-yu3l.6) ------------------------------------------
# ``yes, and show me the overdue books``, ``Yes. Then list the members``, ``sure, afterwards count the books``: the
# reply opens as an accept and a connecting word then starts a further request. That is not a plain accept: it takes
# the path a reply with a corrective word takes (the edit reader), so the request is read or asked about, never
# dropped. What follows the connecting word must be more than accept words (``yes and go ahead``, ``ok then, run
# it`` stay accepts), and ``go ahead and <verb>`` is the idiom for restating the plan, not a second request.
_ACCEPT_LEADS = _ACCEPT_PHRASES + (
    "yes", "yep", "yeah", "yup", "sure", "ok", "okay", "accept", "approve", "approved", "go ahead", "sounds good",
    "looks good", "great", "perfect", "agreed", "confirm", "confirmed", "run it as shown", "keep it as is",
    "keep it as shown", "that is right", "thats right", "that matches what i want",
)
_ACCEPT_LEAD = re.compile(rf"(?:{_alternation(_ACCEPT_LEADS)})\b")
_ACCEPT_LEAD_OR_TAIL = re.compile(rf"(?:{_alternation(_ACCEPT_LEADS + _ACCEPT_TAIL)})\b")
_ACCEPT_ONLY = re.compile(rf"^(?:(?:{_alternation(_ACCEPT_LEADS + _ACCEPT_TAIL)})(?:\s+|$))+$")
_REQUEST_CONNECTOR = re.compile(
    r"(?:and\s+then|and\s+also|and\s+afterwards?|and\s+after\s+that|and\s+next|then|afterwards?|after\s+that|"
    r"after\s+this|plus|additionally|and|but|however|except|although|though)\b")
#: What may follow a contrasting connector without being a request: ``yes, but nothing else``, ``yes, just that``.
_NO_REQUEST_AFTER_CONTRAST = re.compile(r"^(?:nothing\s+(?:else|more)|that\s+is\s+all|thats\s+all|that\s+only|as\s+is)$")
_SEPARATORS = re.compile(r"[^a-z0-9]*")


def _accept_then_request(text: str) -> bool:
    """Whether *text* opens as an accept and then, after a connecting word (``and``, ``then``, ``afterwards``,
    ``after that``, ``plus``) or a contrasting one (``but``, ``however``, ``except``, ``although``, ``though``)
    asks for something more or says something that contradicts the plan. Read on the reply with the no-effect qualifiers removed, so
    ``yes, run that, and apply no corrections`` is not one."""
    value = _APOSTROPHES.sub("", _THUMBS_UP.sub(" thumbsup ", _scrub_benign_confirm_phrases(text).lower()))
    position = _SEPARATORS.match(value).end()
    if value.startswith("please ", position):
        position += len("please ")
    lead = _ACCEPT_LEAD.match(value, position)
    if lead is None:
        return False
    last = lead.group(0)
    position = lead.end()
    while True:
        gap = _SEPARATORS.match(value, position)
        more = _ACCEPT_LEAD_OR_TAIL.match(value, gap.end())
        if more is None:
            break
        last, position = more.group(0), more.end()
    gap = _SEPARATORS.match(value, position)
    connector = _REQUEST_CONNECTOR.match(value, gap.end())
    if connector is None:
        return False
    if connector.group(0) == "and" and last == "go ahead" and gap.group(0).isspace():
        return False                                  # ``go ahead and remove the tag``: one request, restated
    rest = " ".join(re.sub(r"[^a-z0-9]+", " ", value[connector.end():]).split())
    if _NO_REQUEST_AFTER_CONTRAST.match(rest):
        return False
    return bool(rest) and not _ACCEPT_ONLY.match(rest)


_DECLINE_PHRASES = (
    "no", "nope", "nah", "nay", "no thanks", "no thank you", "cancel", "stop", "abort", "dont", "do not", "ill pass",
    "pass", "not now", "never mind", "nevermind", "scratch that", "scratch that plan", "forget it", "reject",
    "decline", "discard",
)
_DECLINE_TAIL = (
    "please", "thanks", "thank you", "thx", "it", "this", "that", "plan", "the plan", "this plan", "that plan",
    "now", "for now", "right now", "anyway", "at all", "run it", "run this", "run that", "run the plan",
    "run this plan", "run",
)
_WHOLE_DECLINE = re.compile(
    rf"^(?:please\s+)?(?:{_alternation(_DECLINE_PHRASES)})"
    rf"(?:\s+(?:{_alternation(_DECLINE_PHRASES + _DECLINE_TAIL)}))*$")


def _plain_decline_cue(text: str) -> bool:
    """Whether the whole of *text* is a plain decline (``no``, ``nope``, ``no thanks``, ``cancel``, ``dont run it``,
    ``ill pass``, ``scratch that plan``). ``no changes, go ahead`` and ``no, change the second one`` are not."""
    return bool(_WHOLE_DECLINE.match(_lexical(text)))


# The polarity guard reads words ANYWHERE in the reply, deliberately wider than the cues above: a false hit only
# makes the reader ask again, never act.
_ACCEPT_SIGNAL_PHRASES = (
    "no changes", "no change", "no edits", "no objections", "no problem", "no problems", "nothing to change",
    "nothing needs to change", "dont change anything", "do not change anything", "go ahead", "do it", "run it",
    "ship it", "lets go", "lets do it", "go for it", "make it so", "carry on", "looks good", "sounds good",
    "all good", "that works", "works for me", "as is", "its fine", "thats fine", "yes", "yep", "yeah",
    "yup", "ok", "okay", "sure", "fine", "great", "perfect", "agreed", "correct", "approve", "approved", "accept",
    "accepted", "confirm", "confirmed", "proceed", "continue", "lgtm", "thumbsup", "k", "kk",
)
_ACCEPT_SIGNAL = re.compile(rf"\b(?:{_alternation(_ACCEPT_SIGNAL_PHRASES)})\b")
#: Accept phrases that contain a decline word (``no changes``); removed before the decline words are looked for.
_ACCEPT_HIDES_DECLINE = re.compile(
    r"\b(?:no changes?|no edits|no objections|no problems?|dont change anything|do not change anything)\b")
#: ``dont run it``: the action words there are part of the refusal, not an accept word.
_NEGATED_ACTION = re.compile(r"\b(?:dont|do not)\s+(?:run|do|ship)\s+(?:it|this|that)\b")
_DECLINE_SIGNAL = re.compile(
    r"\b(?:no|nope|nah|nay|cancel|stop|abort|dont|do not|never mind|nevermind|scratch that|forget it|reject|"
    r"decline|discard|ill pass|pass|not now|not yet|no thanks)\b")
#: A reply that hedges or defers is no decision either way.
_HEDGE_SIGNAL = re.compile(
    r"\b(?:hm+|um+|uh+|maybe|perhaps|possibly|not sure|unsure|idk|dunno|i guess|i dont know|dont know|"
    r"later|tomorrow|tonight|wait|hold on|hold off)\b")
#: A reply that names one step or item is about an edit, not about the whole plan.
_STEP_REFERENCE = re.compile(
    r"\b(?:steps?|items?|tasks?)\s*#?\d+\b|\bafter\s+step\b|"
    r"\b(?:first|second|third|fourth|fifth|last|next|previous|\d+(?:st|nd|rd|th))\s+(?:one|step|item|task)\b")


def _reply_polarity(text: str) -> tuple[bool, bool, bool]:
    """(has an accept word, has a decline word, hedges or defers) by a conservative lexical test over *text*."""
    lexical = _lexical(text, scrub=True)
    accept = bool(_ACCEPT_SIGNAL.search(_NEGATED_ACTION.sub(" ", lexical)))
    decline = bool(_DECLINE_SIGNAL.search(_ACCEPT_HIDES_DECLINE.sub(" ", lexical)))
    return accept, decline, bool(_HEDGE_SIGNAL.search(lexical))


def _polarity_conflict(text: str, routed: str) -> str:
    """Why the reader must not act on the classifier's ``confirm`` or ``decline`` for *text* (empty: it may).

    A routed decline on a reply that holds an accept word, a routed confirm on one that holds a decline word, a
    reply holding both, a hedge or a deferral, and a routed decline on a reply about one step: none is acted on; the
    reader asks again, the way it does for an unrouted reply. A reply with no signal is acted on as before."""
    accept, decline, hedge = _reply_polarity(text)
    if hedge:
        return f"routed {routed} on a hedging or deferring reply"
    if accept and decline:
        return f"routed {routed} on a reply with both an accept and a decline word"
    if routed == "decline" and accept:
        return "routed decline on an affirmative reply"
    if routed == "confirm" and decline:
        return "routed confirm on a negative reply"
    if routed == "decline" and _STEP_REFERENCE.search(_lexical(text)):
        return "routed decline on a reply about one step"
    return ""


def _affirmative_unrouted_confirm(text: str) -> bool:
    scrubbed = _scrub_benign_confirm_phrases(text)
    return (bool(_AFFIRMATIVE_UNROUTED_PREFIX.match(scrubbed)) and not _CONFIRM_CORRECTIVE.search(scrubbed)
            and not _accept_then_request(text))


def _slot_whole_plan_confirm(text: str) -> bool:
    scrubbed = _scrub_benign_confirm_phrases(text)
    return bool(_SLOT_WHOLE_PLAN_CONFIRM.search(scrubbed)) or _affirmative_accept_cue(text)


def _unrouted_plan_confirm(text: str) -> bool:
    """Accept on an unrouted turn when the reply opens with yes and has no corrective clause.

    Covers the case where the ProposalReview head ranks ``confirm`` first but does not route
    because it is not confident (domain wording in the rest of the sentence).
    """
    return _affirmative_unrouted_confirm(text)
_DECLINE_CORRECTIVE = re.compile(
    r"\b(?:not\s+just|not\s+only|not\s+quite|instead|i\s+asked|i\s+want|i\s+meant|"
    r"give\s+me|please|should\s+be|is\s+a|use\s+your\s+best\s+judgment|change|look\s+up|report)\b",
    re.I,
)
_LOOK_UP_CUE = re.compile(
    r"\b(?:look\s+(?:it\s+|that\s+|step\s+)?up|lookup|what\s+does\s+step|explain\s+step|"
    r"details\s+on\s+step|do\s+not\s+(?:record|change|write|modify)|don'?t\s+(?:record|change|write|modify)|"
    r"change\s+nothing|record\s+nothing|read\s+it|only\s+check|just\s+check)\b",
    re.I,
)
_DROP_CUE = re.compile(
    r"\b(?:drop|remove|delete|skip|cut|omit|leave\s+out|take\s+out|get\s+rid|without|"
    r"don'?t\s+(?:need|do)|do\s+not\s+(?:need|do))\b",
    re.I,
)
_EDIT_ALONGSIDE_DROP = re.compile(
    r"\b(?:change|make|set|update|replace|rename|modify|merge|combine|is\s+one|is\s+a)\b",
    re.I,
)
_STEP_NUM_IN_REPLY = re.compile(r"\b(?:step|item)\s+#?(\d+)\b", re.I)


def _compound_drop_edit(text: str, routed_items: Optional[Sequence[Any]] = None) -> bool:
    if not _DROP_CUE.search(text):
        return False
    if ";" in text:
        return True
    if _EDIT_ALONGSIDE_DROP.search(text):
        return True
    if routed_items:
        routed_set = {str(item).strip() for item in routed_items}
        mentioned = _STEP_NUM_IN_REPLY.findall(text)
        if any(num not in routed_set for num in mentioned):
            return True
    return False


def switch_clause_in_goal_message(clause: str) -> str:
    return (f'I read "{clause}" as part of a step\'s goal, but it asks to switch a plan. Reply "step N: use <plan name>" '
            "on its own, or say what the step should do.")


def _refuse_switch_clause_in_goal(edits: list, text: str) -> None:
    """Refuse a typed edit whose goal is, or ends with, an explicit switch clause of the reply (``change 6: goal=show it,
    step 7: use NAME``): the clause is a plan switch, not goal text (bd fix-1ma8.1)."""
    from fastworkflow.review_plan_switch import goal_copies_switch_clause

    for edit in edits:
        goal = (edit.value if isinstance(edit, Change) and edit.field == "goal"
                else (edit.values or {}).get("goal", "") if isinstance(edit, Add) else "")
        if goal and (clause := goal_copies_switch_clause(goal, text)):
            raise EditRefused(switch_clause_in_goal_message(clause))


def _switch_clauses_decision(proposal: Proposal, text: str, workflow: Any, read_rest: Callable[[str], "ReplyDecision"],
                             ) -> Optional["ReplyDecision"]:
    """The decision for a reply that holds an explicit plan-switch clause (``step N: use NAME``, ...), or None when it holds
    none (bd fix-1ma8.1). The clauses are read one by one (``review_plan_switch.split_switch_clauses``): a switch clause is
    decided here and never reaches the free-text reader; the other clauses go to *read_rest* (the policy's own reading of
    any reply). A question about a switch (unknown name, no such step, which plan) asks again and applies nothing. When the
    other clauses are not read as edits (a question, a refusal) that decision is returned and the switch is not applied
    either: the plan stays as it was. Read after the accept and decline cues, so an accept never gets here."""
    from fastworkflow.review_plan_switch import split_switch_clauses

    split = split_switch_clauses(proposal, text, workflow)
    if split is None:
        return None
    if split.ask is not None:
        return split.ask
    edits = list(split.edits)
    reason = "plan switch"
    if split.remainder.strip():
        rest = read_rest(split.remainder)
        if rest.verdict is Verdict.EDIT:
            edits += list(rest.edits)
            reason = rest.reason
        elif rest.verdict is not Verdict.CONFIRM:       # a bare accept word among the clauses adds nothing to the switch
            return rest
    try:
        apply_edits(proposal, edits)
    except EditRefused as refused:
        return ReplyDecision(Verdict.REFUSE, "edits refused", message=str(refused))
    return ReplyDecision(Verdict.EDIT, reason, edits=tuple(edits))


class ProposalReplyPolicy:
    """The sub_tasks and plan kinds: accept, no, or typed edits (deterministic first, then the reader).

    The reader is the session's (the review store's, set for the turn that opens a review) or the registered one. A
    reply the reader cannot read as edits asks again (REPROMPT); the record stays open and the reply counts toward its cap.
    """

    def __init__(
        self,
        interpreter: Optional[EditInterpreter] = None,
        *,
        plan_cme: Any = None,
        plan_workflow: Any = None,
        app_workflow: Any = None,
        structured_fallback: bool = False,
    ):
        self._interpreter = interpreter
        self._plan_cme = plan_cme
        self._plan_workflow = plan_workflow
        self._app_workflow = app_workflow
        self._structured_fallback = structured_fallback

    def decide(self, proposal: Proposal, message: str) -> ReplyDecision:
        from fastworkflow.review_proposals import workflow_scope

        with workflow_scope(getattr(self._app_workflow, "folderpath", None)):
            return self._decide(proposal, message)

    def _decide(self, proposal: Proposal, message: str) -> ReplyDecision:
        from fastworkflow.review_proposals import SubTaskProposal

        text = (message or "").strip()
        word = text.lower()
        sub_task_review = isinstance(proposal, SubTaskProposal)
        if (self._plan_cme is not None and self._plan_workflow is not None
                and sub_task_review):
            return self._decide_with_plan(proposal, text, word)
        if word in ("accept", "yes"):
            return ReplyDecision(Verdict.CONFIRM, "accepted")
        if _affirmative_accept_cue(text):
            return ReplyDecision(Verdict.CONFIRM, "accepted")
        if word in ("no", "decline") or _plain_decline_cue(text):
            return ReplyDecision(Verdict.DECLINE, "declined")
        if sub_task_review and (clauses := _switch_clauses_decision(
                proposal, text, self._app_workflow, lambda rest: self.decide(proposal, rest))) is not None:
            return clauses                            # an explicit switch clause is decided here, never by the reader
        try:
            edits = parse_edits(text)
            edits = _normalize_parsed_subtask_edits(proposal, edits)
        except EditParseError as exc:
            reader = self._interpreter or _model_interpreter
            if reader is None:
                if sub_task_review:
                    return ReplyDecision(Verdict.REPROMPT, "not read as edits", message=SUB_TASK_UNREADABLE_REPLY)
                return ReplyDecision(Verdict.REPROMPT, "not read as edits", message=f"{exc} {PROMPT_HINT}")
            try:
                edits = list(reader(text, proposal))
            except EditRefused as refused:
                return ReplyDecision(Verdict.REFUSE, "edits refused", message=str(refused))
            except EditParseError as inner:
                unreadable = SUB_TASK_UNREADABLE_REPLY if sub_task_review else f"{inner} {PROMPT_HINT}"
                return ReplyDecision(Verdict.REPROMPT, "not read as edits", message=unreadable)
            if not edits:
                unreadable = SUB_TASK_UNREADABLE_REPLY if sub_task_review else f"I could not read that as an edit. {PROMPT_HINT}"
                return ReplyDecision(Verdict.REPROMPT, "not read as edits", message=unreadable)
            try:
                apply_edits(proposal, edits)
            except EditRefused as refused:
                return ReplyDecision(Verdict.REFUSE, "edits refused", message=str(refused))
            return _validate_edits_decision(
                proposal, tuple(edits), reason=READ_REASON, text=text, workflow=self._app_workflow)
        try:
            _refuse_dropped_input_parts(proposal, edits)
            _refuse_switch_clause_in_goal(edits, text)
            apply_edits(proposal, edits)           # validate only; the caller applies
        except EditRefused as exc:
            return ReplyDecision(Verdict.REFUSE, "edits refused", message=str(exc))
        return ReplyDecision(Verdict.EDIT, "typed edits", edits=tuple(edits))

    def _in_plan_slot(self) -> bool:
        return intent_loop.waiting_for(self._plan_cme.context) == intent_loop.WAITING_PARAMETERS

    def _decide_with_plan(self, proposal: Proposal, text: str, word: str) -> ReplyDecision:
        if self._structured_fallback and self._plan_cme is not None:
            _clear_plan_cme_continuation(self._plan_cme)
        was_in_slot = self._in_plan_slot()
        if not was_in_slot:
            if word in ("accept", "yes"):
                return ReplyDecision(Verdict.CONFIRM, "accepted")
            if _affirmative_accept_cue(text):
                return ReplyDecision(Verdict.CONFIRM, "accepted")
            if word in ("no", "decline") or _plain_decline_cue(text):
                return ReplyDecision(Verdict.DECLINE, "declined")
            if (clauses := _switch_clauses_decision(
                    proposal, text, self._app_workflow,
                    lambda rest: self._decide_with_plan(proposal, rest, rest.lower()))) is not None:
                return clauses                        # an explicit switch clause is decided here, never by the head or reader
            try:
                edits = parse_edits(text)
                edits = _normalize_parsed_subtask_edits(proposal, edits)
            except EditParseError:
                pass
            else:
                try:
                    _refuse_dropped_input_parts(proposal, edits)
                    _refuse_switch_clause_in_goal(edits, text)
                    apply_edits(proposal, edits)
                except EditRefused as refused:
                    return ReplyDecision(Verdict.REFUSE, "edits refused", message=str(refused))
                return ReplyDecision(Verdict.EDIT, "typed edits", edits=tuple(edits))

        decision = self._plan_turn(proposal, text, was_in_slot=was_in_slot, initial_word=word)
        if (self._structured_fallback
                and decision.reason in ("item slot", "slot wait", "slot retry")):
            if self._plan_cme is not None:
                _clear_plan_cme_continuation(self._plan_cme)
            return ReplyDecision(
                Verdict.REPROMPT, "not read as edits", message=SUB_TASK_STRUCTURED_UNREADABLE_REPLY)
        return decision

    def _wildcard_route_from_cme(self, cme_out: Any) -> Any:
        """The CME wildcard's routing ``Signature.Output`` (fix-3rsq.12.3 framing).

        ``_wildcard_command_output`` copies routing fields into ``artifacts`` and sets
        ``CommandResponse.output`` to the nested command output only (handled paths). Plan
        policy needs the routing record itself.
        """
        from fastworkflow._workflows.command_metadata_extraction._commands.wildcard import (
            Signature as WildcardSignature,
        )

        out = getattr(cme_out.command_response, "output", None)
        if out is not None and hasattr(out, "routed"):
            return out
        arts = getattr(cme_out.command_response, "artifacts", None) or {}
        return WildcardSignature.Output(
            handled=arts.get("command_handled") is True,
            routed=bool(cme_out.success),
            response=getattr(cme_out.command_response, "response", "") or "",
            command=arts.get("command"),
            command_name=arts.get("command_name"),
            cmd_parameters=arts.get("cmd_parameters"),
            ran_on_context_name=arts.get("ran_on_context_name"),
            ran_on_context_clause=arts.get("ran_on_context_clause"),
            reply_intent=arts.get("reply_intent"),
            output=out if out is not None and not hasattr(out, "routed") else None,
        )

    def _plan_turn(
        self,
        proposal: Proposal,
        text: str,
        *,
        was_in_slot: bool,
        initial_word: str,
    ) -> ReplyDecision:
        from fastworkflow.command_executor import CommandExecutor
        from fastworkflow._workflows.plan_metadata_extraction._item_validation import explicit_out_of_range_step
        from fastworkflow.review_proposals import certain_removals

        _prepare_plan_workflow(self, proposal)
        slot_question = _slot_question(proposal)
        cme_out = CommandExecutor.perform_action(
            self._plan_cme,
            fastworkflow.Action(command_name="wildcard", command=text),
        )
        w_out = self._wildcard_route_from_cme(cme_out)
        if w_out is None:
            return self._edit_fallback(proposal, text, slot_question)

        if was_in_slot:
            handled = self._handle_slot_wildcard(proposal, text, w_out, slot_question, initial_word)
            if handled is not None:
                return handled
            was_in_slot = False

        if _routed_to_command(w_out):
            return self._handle_routed_plan(proposal, text, w_out, slot_question)

        pending_name = self._plan_cme.context.get("command_name")
        if (intent_loop.waiting_for(self._plan_cme.context) == intent_loop.WAITING_PARAMETERS
                and pending_name in ("ProposalReview/drop", "ProposalReview/look_up")):
            count = len(proposal.items)
            if explicit_out_of_range_step(text, count):
                _clear_plan_cme_continuation(self._plan_cme)
                return ReplyDecision(Verdict.REFUSE, "edits refused", message=_step_count_message(count))
            if pending_name.endswith("/look_up") and not _LOOK_UP_CUE.search(text):
                return self._edit_fallback(proposal, text, slot_question)
            if pending_name.endswith("/drop") and (
                count == 1 or not _DROP_CUE.search(text) or _compound_drop_edit(text)
            ):
                return self._edit_fallback(proposal, text, slot_question)
            if count == 1:
                _clear_plan_cme_continuation(self._plan_cme)
                edit = Drop(index=1) if pending_name.endswith("/drop") else LookUp(index=1)
                return _validate_edits_decision(proposal, (edit,), reason="typed edits")
            if pending_name.endswith("/drop") and (named := certain_removals(proposal, text)):
                # The reply names the steps to remove by number: no need to ask which.
                _clear_plan_cme_continuation(self._plan_cme)
                return self._drop_decision(proposal, text, named)
            parameter_clarification.requestion(self._plan_cme.context, slot_question)
            return ReplyDecision(Verdict.REPROMPT, "item slot", message=slot_question)

        if (
            not was_in_slot
            and not _routed_to_command(w_out)
            and _unrouted_plan_confirm(text)
        ):
            _clear_plan_cme_continuation(self._plan_cme)
            return ReplyDecision(Verdict.CONFIRM, "accepted")

        return self._edit_fallback(proposal, text, slot_question)

    def _handle_slot_wildcard(
        self,
        proposal: Proposal,
        text: str,
        w_out: Any,
        slot_question: str,
        initial_word: str,
    ) -> Optional[ReplyDecision]:
        if _slot_whole_plan_confirm(text):
            _clear_plan_cme_continuation(self._plan_cme)
            return ReplyDecision(Verdict.CONFIRM, "accepted")
        intent = getattr(w_out, "reply_intent", None)
        if intent == "wait":
            parameter_clarification.requestion(self._plan_cme.context, slot_question)
            return ReplyDecision(Verdict.REPROMPT, "slot wait", message=slot_question)
        if intent == "abort":
            _clear_plan_cme_continuation(self._plan_cme)
            return ReplyDecision(
                Verdict.REPROMPT,
                "slot aborted",
                message=f"OK, nothing changed.\n{prompt_for(proposal, self._app_workflow)}",
            )
        if intent == "misunderstood":
            _clear_plan_cme_continuation(self._plan_cme)
            return self._edit_fallback(proposal, text, slot_question)
        if intent == "answer" and not _routed_to_command(w_out):
            parameter_clarification.requestion(self._plan_cme.context, slot_question)
            return ReplyDecision(Verdict.REPROMPT, "slot retry", message=slot_question)
        if intent == "different_command":
            if _routed_to_command(w_out):
                return self._handle_routed_plan(proposal, text, w_out, slot_question)
            if initial_word in ("accept", "yes"):
                return ReplyDecision(Verdict.CONFIRM, "accepted")
            if initial_word in ("no", "decline"):
                return ReplyDecision(Verdict.DECLINE, "declined")
            return None
        return None

    def _plan_action_parameters(self, cmd_parameters: Any) -> dict:
        if cmd_parameters is None:
            return {}
        if hasattr(cmd_parameters, "model_dump"):
            return cmd_parameters.model_dump()
        if isinstance(cmd_parameters, dict):
            return cmd_parameters
        return {}

    def _park_edit_item_slot(self, text: str, slot_question: str, pending_proposed: list) -> None:
        from fastworkflow.utils.signatures import NOT_FOUND
        from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import park
        from fastworkflow._workflows.plan_metadata_extraction._commands.ProposalReview.edit import Signature as EditSignature

        pending = EditSignature.SlotInput.model_construct(
            item=NOT_FOUND,
            edits=list(pending_proposed),
        )
        ctx = self._plan_cme.context
        ctx["command"] = text
        ctx["command_name"] = "ProposalReview/edit"
        self._plan_cme.context = ctx
        park(self._plan_cme, "ProposalReview/edit", pending, question=slot_question, request_text=text)

    def _reply_from_edit_output(
        self,
        proposal: Proposal,
        text: str,
        routed: Any,
        slot_question: str,
    ) -> ReplyDecision:
        if getattr(routed, "refusal", None):
            _clear_plan_cme_continuation(self._plan_cme)
            return ReplyDecision(Verdict.REFUSE, "edits refused", message=routed.refusal)
        pending = list(getattr(routed, "pending_proposed", None) or [])
        if pending:
            self._park_edit_item_slot(text, slot_question, pending)
            return ReplyDecision(Verdict.REPROMPT, "item slot", message=slot_question)
        edits = tuple(getattr(routed, "edits", None) or [])
        if not edits:
            _clear_plan_cme_continuation(self._plan_cme)
            return ReplyDecision(Verdict.REPROMPT, "not read as edits", message=SUB_TASK_UNREADABLE_REPLY)
        _clear_plan_cme_continuation(self._plan_cme)
        return _validate_edits_decision(
            proposal, edits, reason=READ_REASON, text=text, workflow=self._app_workflow)

    def _perform_plan_command(self, command_name: str, command: str, input_obj: Any) -> Any:
        from fastworkflow.command_executor import call_response_generator
        from fastworkflow.command_routing import ModuleType

        definition = fastworkflow.RoutingRegistry.get_definition(self._plan_workflow.folderpath)
        response_class = definition.get_command_class(command_name, ModuleType.RESPONSE_GENERATION_INFERENCE)
        if not response_class:
            raise ValueError(f"Response generation class not found for command name {command_name!r}")
        return call_response_generator(
            response_class(),
            self._plan_workflow,
            command,
            input_obj,
            command_name=command_name,
        )

    def _handle_routed_plan(
        self,
        proposal: Proposal,
        text: str,
        w_out: Any,
        slot_question: str,
    ) -> ReplyDecision:
        from fastworkflow.command_executor import CommandExecutor

        name = w_out.command_name or ""
        cmd_parameters = w_out.cmd_parameters
        if isinstance(cmd_parameters, BaseModel):
            cmd_out = self._perform_plan_command(name, text, cmd_parameters)
        else:
            cmd_out = CommandExecutor.perform_action(
                self._plan_workflow,
                fastworkflow.Action(
                    command_name=name,
                    command=text,
                    parameters=self._plan_action_parameters(cmd_parameters),
                ),
            )
        routed = getattr(cmd_out.command_response, "output", None)
        if routed is None:
            return self._edit_fallback(proposal, text, slot_question)
        if name.endswith("/confirm"):
            if _CONFIRM_CORRECTIVE.search(_scrub_benign_confirm_phrases(text)) or _accept_then_request(text):
                return self._edit_fallback(proposal, text, slot_question)
            if conflict := _polarity_conflict(text, "confirm"):
                return self._ask_again_on_conflict(proposal, text, slot_question, conflict)
            _clear_plan_cme_continuation(self._plan_cme)
            return ReplyDecision(Verdict.CONFIRM, "accepted")
        if name.endswith("/decline"):
            if _DECLINE_CORRECTIVE.search(text) or _DROP_CUE.search(text):
                return self._edit_fallback(proposal, text, slot_question)
            if conflict := _polarity_conflict(text, "decline"):
                return self._ask_again_on_conflict(proposal, text, slot_question, conflict)
            _clear_plan_cme_continuation(self._plan_cme)
            return ReplyDecision(Verdict.DECLINE, "declined")
        if name.endswith("/drop"):
            drop_items = list(getattr(routed, "items", None) or [])
            if _compound_drop_edit(text, drop_items):
                return self._edit_fallback(
                    proposal, text, slot_question, preserve_drop_items=drop_items or None
                )
            _clear_plan_cme_continuation(self._plan_cme)
            return self._drop_decision(proposal, text, drop_items)
        if name.endswith("/add"):
            _clear_plan_cme_continuation(self._plan_cme)
            after = len(proposal.items) if routed.after is None else int(routed.after)
            edit = Add(after=after, values={"goal": routed.goal})
            return _validate_edits_decision(proposal, (edit,), reason="typed edits")
        if name.endswith("/look_up"):
            _clear_plan_cme_continuation(self._plan_cme)
            edit = LookUp(index=int(routed.item))
            return _validate_edits_decision(proposal, (edit,), reason="typed edits")
        if name.endswith("/edit"):
            return self._reply_from_edit_output(proposal, text, routed, slot_question)
        _clear_plan_cme_continuation(self._plan_cme)
        return self._edit_fallback(proposal, text, slot_question)

    def _drop_decision(self, proposal: Proposal, text: str, items: list) -> ReplyDecision:
        """Drops for the steps a routed drop (or the slot) names, held to the reply's own step numbers: a list the
        extractor read short is completed, a step the reply says to keep is not dropped, an unclear one asks again."""
        from fastworkflow.review_proposals import reconcile_dropped_items

        try:
            numbers = reconcile_dropped_items(proposal, [str(item) for item in items], text)
        except EditRefused as refused:
            return ReplyDecision(Verdict.REFUSE, "edits refused", message=str(refused))
        return _validate_edits_decision(
            proposal, tuple(Drop(index=number) for number in numbers), reason="typed edits")

    def _ask_again_on_conflict(
        self, proposal: Proposal, text: str, slot_question: str, conflict: str
    ) -> ReplyDecision:
        """The classifier's confirm or decline is not acted on (the polarity guard, bd fix-blui): the reply takes
        the path of an unrouted one, and the decision records why."""
        logger.info("review reply %r not acted on: %s", text, conflict)
        decision = self._edit_fallback(proposal, text, slot_question)
        return replace(decision, note=f"polarity guard: {conflict}")

    def _edit_fallback(
        self,
        proposal: Proposal,
        text: str,
        slot_question: str,
        *,
        preserve_drop_items: list[str] | None = None,
    ) -> ReplyDecision:
        from fastworkflow.command_executor import CommandExecutor

        _clear_plan_cme_continuation(self._plan_cme)
        _prepare_plan_workflow(self, proposal)
        cmd_out = CommandExecutor.perform_action(
            self._plan_workflow,
            fastworkflow.Action(command_name="ProposalReview/edit", command=text),
        )
        routed = getattr(cmd_out.command_response, "output", None)
        if routed is None:
            if preserve_drop_items:
                _clear_plan_cme_continuation(self._plan_cme)
                return self._drop_decision(proposal, text, preserve_drop_items)
            return ReplyDecision(Verdict.REPROMPT, "not read as edits", message=SUB_TASK_UNREADABLE_REPLY)
        decision = self._reply_from_edit_output(proposal, text, routed, slot_question)
        if preserve_drop_items and (
            decision.reason == "item slot"
            or (decision.verdict == Verdict.REPROMPT and decision.reason == "not read as edits")
        ):
            _clear_plan_cme_continuation(self._plan_cme)
            return self._drop_decision(proposal, text, preserve_drop_items)
        return decision


def reply_policy(
    kind: str,
    interpreter: Optional[EditInterpreter] = None,
    *,
    plan_cme: Any = None,
    plan_workflow: Any = None,
    app_workflow: Any = None,
    structured_fallback: bool = False,
):
    """The reply policy for a kind (the policy decision point for replies); *interpreter* is the proposal reader."""
    if kind == KIND_ACTION:
        return ActionReplyPolicy()
    if kind == KIND_SUB_TASKS:
        return ProposalReplyPolicy(
            interpreter,
            plan_cme=plan_cme,
            plan_workflow=plan_workflow,
            app_workflow=app_workflow,
            structured_fallback=structured_fallback,
        )
    raise ValueError(f"unknown review kind {kind!r}")


# ----------------------------------------------------------------------
# Open, reply and revise (the suspend/resume primitive for proposals)
# ----------------------------------------------------------------------


@dataclass
class ReviewReply:
    """``done`` False: the record is still open and ``text`` is what the user is shown. ``done`` True: the
    record left the open states; ``proposal`` is the accepted proposal (None when declined or invalidated)."""

    done: bool
    text: str
    proposal: Optional[Proposal] = None


def open_review(host: Any, store: ReviewStore, proposal: Proposal, *, surface: Surface, origin: str = "agent",
                upstream: "list[ReviewRecord] | tuple" = (), suspend: bool = False,
                items_text: str = "") -> ReviewRecord:
    """Open a review of *proposal* on the session: one awaiting record, rendered from the template.

    With *suspend*, the agent's tool call is suspended on the prompt (``AskUserSuspend``, the ask_user
    machinery); the next message resumes it through :func:`reply_to_proposal`. Only one review is open
    per session at a time.
    """
    if store.active() is not None:
        raise ReviewError("another review is awaiting the user on this session")
    # items_text: the request's items as the planning step passed them. The accepted run re-executes with the
    # same text, so the run the phase archives carries the step's own parameters (the finish check matches them).
    binding = {**proposal.binding(), "channel": store.channel, "task": store.task, "items_text": items_text}
    text = prompt_for(proposal, _prompt_workflow(host, store), no_agent=_no_agent_of(host))
    ttl_turns, ttl_seconds = ttl_for_kind(proposal.kind)
    record = ReviewRecord(
        id=uuid.uuid4().hex, token_hash="", command=proposal.spec, annotation=None, raw_command="",
        binding=binding, fingerprint=fingerprint(binding), rendering=text,
        rendering_hash=rendering_hash(text), required_identity=surface.required_identity,
        created_turn=store.turn_seq, turns_seen=0, ttl_turns=ttl_turns, ttl_seconds=ttl_seconds,
        created_at=time.time(), state=AWAITING, origin=origin, suspended_agent=suspend,
        shadow=(switch_mode() == MODE_SHADOW), kind=proposal.kind)
    link(record, list(upstream))
    if surface.kind == "host":
        token = new_token()
        record.token_hash = token_hash(token)
        store.delivery = {"record_id": record.id, "token": token}
    store.add(record)
    audit(host, record, before="", after=AWAITING, reason="opened", surface=surface, mode=switch_mode(),
          outcome="asked")
    if (put := getattr(question_owner(store, host), "_put_review_record", None)) is not None:
        put(record)      # the review is the turn's waiting record, pointing at this one (``waiting``)
    if suspend:
        from fastworkflow.utils.react import AskUserSuspend

        raise AskUserSuspend(text)
    return record


def reply_to_proposal(
    host: Any,
    store: ReviewStore,
    record: ReviewRecord,
    message: str,
    *,
    cap_turn: bool = False,
) -> ReviewReply:
    """Resolve one reply on an open proposal record. Callers have already checked TTL and identity."""
    proposal = proposal_of(record)
    surface = store.surface
    answering_q3_or_later = record.turns_seen >= 2
    producing_q3_or_later = record.turns_seen >= 1
    decision = reply_policy(
        record.kind,
        store.edit_reader,
        plan_cme=getattr(store, "plan_cme", None),
        plan_workflow=getattr(store, "plan_workflow", None),
        app_workflow=_prompt_workflow(host, store),
        structured_fallback=answering_q3_or_later,
    ).decide(proposal, message)
    if cap_turn and decision.verdict not in (Verdict.CONFIRM, Verdict.DECLINE):
        transition(host, store, record, EXPIRED, reason="ttl", surface=surface, outcome="expired")
        propagate(host, store, record, surface=surface)
        from fastworkflow.approval_binding import MSG_EXPIRED

        return ReviewReply(True, MSG_EXPIRED.format(command=record.command))
    if decision.verdict is Verdict.CONFIRM:
        stale = stale_upstream(store, record)
        if stale is not None:
            transition(host, store, record, INVALIDATED, reason=stale, surface=surface, outcome="invalidated")
            return ReviewReply(True, f"The {record.command} review was invalidated: {stale}. Nothing was accepted.")
        transition(host, store, record, CONFIRMED, reason="accepted", surface=surface, outcome="accepted",
                   resolution={"verdict": "confirmed", "identity_class": record.required_identity,
                               "identity_ref": surface.uid if surface.uid else "none"})
        return ReviewReply(True, f"{proposal.title} accepted.", proposal)
    if decision.verdict is Verdict.DECLINE:
        transition(host, store, record, DECLINED, reason="declined", surface=surface, outcome="declined")
        propagate(host, store, record, surface=surface)
        return ReviewReply(True, f"Declined. The {record.command} review was not accepted.")
    if decision.verdict is Verdict.EDIT:
        if producing_q3_or_later and getattr(store, "plan_cme", None) is not None:
            _clear_plan_cme_continuation(store.plan_cme)
        edited = apply_edits(proposal, list(decision.edits)).revised(host, proposal)
        record.binding = {**record.binding, "items": edited.binding()["items"]}
        record.fingerprint = fingerprint(record.binding)   # the record's fingerprint covers its channel and task too
        record.revision += 1
        record.rendering = prompt_for(
            edited, _prompt_workflow(host, store), structured=producing_q3_or_later, no_agent=_no_agent_of(host))
        record.rendering_hash = rendering_hash(record.rendering)
        transition(host, store, record, REVISED, reason=f"revision {record.revision}: {decision.reason}",
                   surface=surface, outcome="revised")
        propagate(host, store, record, surface=surface)
        if decision.reason == READ_REASON:
            # Shown back as read, for confirmation: nothing is confirmed until the user accepts it.
            return ReviewReply(False, "I read your reply as these edits. Accept to confirm them, or reply with more "
                                      f"edits.\n{record.rendering}")
        return ReviewReply(False, record.rendering)
    # REFUSE or REPROMPT: nothing changes, and the user is asked again.
    if producing_q3_or_later:
        if getattr(store, "plan_cme", None) is not None:
            _clear_plan_cme_continuation(store.plan_cme)
        from fastworkflow.review_proposals import SubTaskProposal

        if isinstance(proposal, SubTaskProposal):
            return ReviewReply(
                False,
                _structured_error_prompt(
                    proposal,
                    _prompt_workflow(host, store),
                    decision.message or SUB_TASK_UNREADABLE_REPLY,
                    no_agent=_no_agent_of(host),
                ),
            )
    return ReviewReply(False, decision.message or PROMPT_HINT)
