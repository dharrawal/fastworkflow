import asyncio
import contextlib
import os
import queue
import time
import uuid
import weakref
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Queue
from typing import Annotated, Any, Callable, Literal, Optional

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwt.exceptions import PyJWTError as JWTError
from pydantic import BaseModel, Field, field_validator

import fastworkflow
from fastworkflow.review import approval_binding, approval_grants
from fastworkflow.session.session_state_store import (
    IncompatibleSessionState,
    PendingReapOutcome,
    PendingRetentionPolicy,
    SessionStateStore,
    get_session_state_store,
)
from fastworkflow.session.state_serialization import StateEncodingError
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from fastworkflow.experiment.readiness import runtime_readiness_snapshot
from fastworkflow.utils.logging import logger
from fastworkflow.turn_progress import ProgressEvent
from fastworkflow.experiment.runner import (
    MissingExperimentLifecycleFeature,
    experiment_store_readiness,
)

from fastworkflow.session.checkpoint_store import (
    ChannelCheckpointStore,
    CheckpointIdentity,
    CheckpointStoreError,
    QuarantineReason,
    RetentionPolicy,
)
from fastworkflow.session.conversation_history_io import restore_history_from_turns
from fastworkflow.observability import feedback as observability_feedback
from fastworkflow.observability.comparison import (
    PassSelector,
    discover_pass_selectors,
    # Private by name, shared by intent: `observability.feedback` already
    # imports `_check_scope` and `_SpanTree` from here for exactly this reason.
    # A second implementation of "which pass does this span belong to" is the
    # drift the shared-reference design exists to prevent, and this is the
    # function the comparison projection attributes steps with.
    _pass_id_for,
    _SpanTree,
)
from fastworkflow.observability.store import (
    FEEDBACK_PROVENANCES,
    AttemptClaimError,
    ObservabilityStore,
    SQLiteTraceSink,
    StoreIdentityMismatch,
    get_observability_sink,
)
from fastworkflow.observability.tracing import ATTR_PASS
from . import checkpoint
from . import session_notices
from .session_notices import SessionNoticeStore
from . import jwt_manager
from .jwt_manager import verify_token


# Live sessions kept in the process cache. Lowered from 2000 now that eviction
# writes a checkpoint instead of discarding state.
DEFAULT_MAX_LIVE_SESSIONS = 50

# How often the server reclaims abandoned checkpoints. A private constant, not a
# knob: the store deliberately has no timer of its own, so somebody has to drive
# it, but an operator does not need a second dial to turn.
CHECKPOINT_REAP_INTERVAL_SECONDS = 300.0

# How long shutdown waits for in-flight turns before giving up on them. Also not a
# knob; it is named, and lives here rather than at its one call site in
# __main__.lifespan, so that the startup check comparing the topic-generation
# budget against it cannot drift away from the value actually used.
SHUTDOWN_DRAIN_SECONDS = 30

# One owner id for every registered attempt claimed by this server process.
# Session incarnations are intentionally separate: eviction/recreation may
# replace a live WEC without changing which process owns the external attempt.
SERVER_INCARNATION = uuid.uuid4().hex


def resolve_max_live_sessions() -> tuple[int, str]:
    """Resolve MAX_LIVE_SESSIONS under the one precedence rule (``fastworkflow.get_env_var``: the process
    environment, then the workflow env file, then the default); unset or empty is the default.

    Returns the value and where it came from, because an operator has to be able
    to see whether their override actually took effect.
    """
    raw = fastworkflow.raw_env_var("MAX_LIVE_SESSIONS")
    source = fastworkflow.env_source("MAX_LIVE_SESSIONS")
    if raw is None or raw == "":
        return DEFAULT_MAX_LIVE_SESSIONS, "default"

    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MAX_LIVE_SESSIONS={raw!r} (from {source}) is not an integer"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"MAX_LIVE_SESSIONS={value} (from {source}) must be greater than zero"
        )
    return value, source


# The live-session count has two limits. MAX_LIVE_SESSIONS (above) is the soft target: past it the server evicts
# sessions it can checkpoint. A session it cannot checkpoint (a workflow context class with no get_state hook, a
# declared EPHEMERAL one, a failing hook) is never evicted, so the soft target alone does not bound anything.
# The hard ceiling does: a request that would create a session past it is first made room for (evict, then expire
# idle sessions), and otherwise refused with 503 + Retry-After.
DEFAULT_HARD_CEILING_FACTOR = 4
# A pinned session idle this long may be expired, but only while the live count is above the soft target.
DEFAULT_PINNED_IDLE_TTL_SECONDS = 1800
# A pinned session with a question waiting for its user is protected from the shorter TTL; it expires after this
# one (and its pending question is discarded), again only above the soft target. 0 disables expiry.
DEFAULT_PROTECTED_IDLE_TTL_SECONDS = 86400
SESSION_CAPACITY_RETRY_AFTER_SECONDS = 30
# Most trace events one session's queue holds. The server drains it every turn; the cap only matters for a turn
# that produces events faster than they are drained, or one that failed before its drain.
MAX_TRACE_QUEUE_EVENTS = 10_000


def _resolve_env_int(name: str, default: int, *, minimum: int) -> tuple[int, str]:
    """An integer setting under the one precedence rule (as resolve_max_live_sessions)."""
    raw = fastworkflow.raw_env_var(name)
    source = fastworkflow.env_source(name)
    if raw is None or raw == "":
        return default, "default"
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}={raw!r} (from {source}) is not an integer") from exc
    if value < minimum:
        raise ValueError(f"{name}={value} (from {source}) must be at least {minimum}")
    return value, source


def resolve_max_live_sessions_hard(soft_target: int) -> tuple[int, str]:
    """MAX_LIVE_SESSIONS_HARD: the live-session ceiling. Default is 4 x the soft target; below it is an error."""
    value, source = _resolve_env_int(
        "MAX_LIVE_SESSIONS_HARD", soft_target * DEFAULT_HARD_CEILING_FACTOR, minimum=1)
    if value < soft_target:
        raise ValueError(
            f"MAX_LIVE_SESSIONS_HARD={value} (from {source}) must not be below MAX_LIVE_SESSIONS={soft_target}"
        )
    return value, source


def resolve_pinned_idle_ttl() -> tuple[int, str]:
    """PINNED_SESSION_IDLE_TTL_SECONDS (default 1800; 0 never expires a session)."""
    return _resolve_env_int("PINNED_SESSION_IDLE_TTL_SECONDS", DEFAULT_PINNED_IDLE_TTL_SECONDS, minimum=0)


def resolve_protected_idle_ttl() -> tuple[int, str]:
    """PROTECTED_SESSION_IDLE_TTL_SECONDS (default 86400; 0 never expires a protected session)."""
    return _resolve_env_int("PROTECTED_SESSION_IDLE_TTL_SECONDS", DEFAULT_PROTECTED_IDLE_TTL_SECONDS, minimum=0)


class BoundedTraceQueue(queue.Queue):
    """The per-session trace queue, bounded: past ``maxlen`` the OLDEST event is dropped and counted.

    Unbounded ``Queue`` was fine only while every turn was drained. The server drains it after every turn and
    during streaming, but a turn that fails before its drain leaves its events behind, so the bound is made
    structural instead of depending on that. ``dropped`` is summed into the readiness probe.
    """

    def __init__(self, maxlen: int = MAX_TRACE_QUEUE_EVENTS) -> None:
        super().__init__()
        self._maxlen = maxlen
        self.dropped = 0

    def _put(self, item) -> None:  # called by Queue.put with the queue mutex held
        if len(self.queue) >= self._maxlen:
            self.queue.popleft()
            self.dropped += 1
        super()._put(item)


class SessionCapacityExceeded(Exception):
    """A new session would pass the live-session ceiling and none could be made room for."""

    def __init__(self, live: int, ceiling: int) -> None:
        super().__init__(
            f"The server is at its live-session ceiling ({ceiling}; {live} sessions are held) and no idle "
            f"session could be released. Retry in {SESSION_CAPACITY_RETRY_AFTER_SECONDS} seconds, or end "
            "channels you no longer use with end_session."
        )
        self.live = live
        self.ceiling = ceiling


# Newest conversation turns kept in memory per channel. Turns are request-sized
# (~450 KB in the workload this bound was sized against, so ~9 MB per channel),
# only the newest few are ever read, and the durable record keeps the rest.
MAX_CONVERSATION_TURNS_IN_MEMORY = 20
# While the newest turn record is not yet durable the window trim is deferred, but never past this many windows.
DEFERRED_TRIM_WINDOW_FACTOR = 2


# ============================================================================
# Data Models (aligned with FastWorkflow canonical types)
# ============================================================================

# A channel id as a client supplies it. Empty is the one value the storage layer
# refuses: both durable stores key records by `encode_path_component(channel_id)`,
# which raises on "" and is total for every other string. Rejected here so a
# malformed request is answered as one, rather than surfacing as a 500 carrying a
# storage-layer message the caller cannot act on (fix-3xf). The ValueError stays
# as the backstop; this is what makes it unreachable from a well-formed request.
#
# Deliberately no length cap. The encoder's ceiling is 200 bytes on the *encoded*
# name, which percent-escaping can triple, so any raw-length number derived from
# it would reject ordinary ids to prevent nothing: an oversized id is a supported
# path (prefix plus sha256, with the raw id re-checked on read), not an error.
#
# Applies to values a client sends. `SessionData.channel_id` comes out of a JWT
# this server minted, so closing both minting endpoints closes that route too.
ChannelId = Annotated[str, Field(min_length=1)]


class ExperimentBootstrapRequest(BaseModel):
    """One-use external-attempt credential, without caller-supplied labels."""

    registration_id: str = Field(min_length=1)
    secret: str = Field(min_length=1, repr=False)
    store_id: str = Field(min_length=1)


class InitializationRequest(BaseModel):
    """Request to initialize a FastWorkflow session for a channel"""
    channel_id: ChannelId
    user_id: Optional[str] = None  # Required if startup_command or startup_action provided
    stream_format: Optional[str] = None  # "ndjson" | "sse" (default ndjson)
    startup_command: Optional[str] = None  # Mutually exclusive with startup_action
    startup_action: Optional[dict[str, Any]] = None  # Mutually exclusive with startup_command
    # Per-session workflow context [R24]: overrides the server-level --context
    # CLI arg for THIS session only. Additive — absent means current behavior
    # (the server-level context, or none). Applied at session creation, so it
    # only takes effect on the /initialize call that creates the session.
    context: Optional[dict[str, Any]] = None
    # How long the request blocks for the startup turn before deferring (202).
    # Same shape/default as InvokeRequest/PerformActionRequest.timeout_seconds.
    timeout_seconds: int = 60
    experiment_bootstrap: Optional[ExperimentBootstrapRequest] = None


class TokenResponse(BaseModel):
    """JWT token pair returned from initialization or token refresh"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Access token expiration in seconds


class InitializeResponse(BaseModel):
    """Response from initialization including tokens and optional startup output.

    Startup runs as a turn (wait-or-defer). If it finishes within the wait
    window, ``startup_output`` is present (200). Otherwise it is still running
    and the caller polls via ``startup_turn_key`` (202). The "already exists"
    branch returns the SAME startup execution's three-state status, never a
    silently-empty result (§3.3).

    ``startup_output`` is the startup turn's ``TurnOutput`` — the same public
    projection every other turn endpoint returns. Each command's own
    ``command_response``/``artifacts`` are still reachable under
    ``startup_output.command_outputs``.
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Access token expiration in seconds
    startup_output: Optional[fastworkflow.TurnOutput] = None  # Present if startup finished in-window
    startup_turn_key: Optional[str] = None  # Handle to poll the startup turn
    startup_exec_state: Optional[str] = None  # queued | running | done | lost
    startup_error: Optional[str] = None  # Present if the startup turn failed
    # [R9] the startup turn's LOGICAL key (TurnOutput.turn_key), once known —
    # the key GET /turns and the observability store are ultimately keyed by.
    startup_logical_turn_key: Optional[str] = None
    # Present when this channel's earlier session was dropped by the server (expired under capacity pressure, or
    # ended with state that could not be saved): {code: "context_dropped", reason, message}. Shown once.
    session_notice: Optional[dict[str, Any]] = None


class SessionData(BaseModel):
    """Validated session data extracted from JWT token"""
    channel_id: str
    user_id: Optional[str] = None  # From JWT uid claim
    token_type: str  # "access" or "refresh"
    issued_at: int  # Unix timestamp
    expires_at: int  # Unix timestamp
    jti: str  # JWT ID (unique token identifier)
    http_bearer_token: Optional[str] = None  # The actual JWT token string for workflow context access


class InvokeRequest(BaseModel):
    """
    Request to invoke agent or assistant.
    Requires channel_id to be passed in the Authorization header (via JWT token).
    """
    user_query: str
    timeout_seconds: int = 60
    use_dspy_cache: Optional[bool] = Field(
        default=None,
        description=(
            "Takes effect only on a server started with --allow_request_cache_toggle (the local chat); "
            "otherwise it is ignored and a value that disagrees with the server is named in the "
            "X-FW-DSPy-Cache response header (the response cache is chosen at start with FW_LM_CACHE=0)."
        ),
    )
    # This turn runs with nobody to answer: the agent gets no ask_user tool.
    # Per turn, never sticky; omitted means attended.
    unattended: bool = False
    # The token of a pending approval (ADR-EXP-025), carried back on the reply that answers it.
    approval_token: Optional[str] = None


class AgentInvokeRequest(InvokeRequest):
    """Request to invoke the agent (``/invoke_agent``, ``/invoke_agent_stream``): an ``InvokeRequest`` that may also
    name the plan to use."""
    use_plan: Optional[str] = Field(
        default=None,
        description=(
            "A planning command available here, by the name an fw_not_run entry's available_plans lists. "
            "user_query is then run as ONE part (not split) bound to that planning command, and no "
            "review: the turn asks for the plan's parameter values (every parameter, the drafted values shown; yes runs, "
            "'name: value' lines change them, cancel stops; a required value still missing is asked for). Send it with the entry's own `request` "
            "text to run a part that was not run here. An unknown name, or one not available here, "
            "is refused with 400. Omit for an ordinary request."
        ),
    )


class EndSessionRequest(BaseModel):
    """Optional body of POST /end_session. The channel is the token's; it is never a parameter."""
    discard_pending: bool = Field(
        default=False,
        description=(
            "Set true to end the session even though a question is waiting for an answer (an agent question, "
            "a review, an approval, a parameter request). The question is lost. Left false, ending is "
            "refused while one is pending."
        ),
    )


class EndSessionResponse(BaseModel):
    """What POST /end_session did to the caller's channel."""
    channel_id: str
    ended: bool = Field(description="True when a live session was released; false when the channel had none.")
    pinned: bool = Field(
        description=(
            "True when the session's state could not be checkpointed, so its in-memory workflow state was "
            "dropped and a later call on the channel starts fresh. False: it was checkpointed and a later "
            "call restores it (or there was no session)."
        )
    )
    pending_discarded: bool = Field(description="True when a pending question was discarded (discard_pending).")


DSPY_CACHE_HEADER = "X-FW-DSPy-Cache"

_allow_request_cache_toggle = False
_dspy_cache_enabled: Optional[bool] = None


def set_allow_request_cache_toggle(allowed: bool) -> None:
    """Set at startup from --allow_request_cache_toggle (like set_chat_approval_surface)."""
    global _allow_request_cache_toggle, _dspy_cache_enabled
    _allow_request_cache_toggle = bool(allowed)
    _dspy_cache_enabled = None


def allow_request_cache_toggle() -> bool:
    return _allow_request_cache_toggle


def apply_dspy_cache_choice(use_dspy_cache: Optional[bool]) -> Optional[str]:
    """Act on a request's ``use_dspy_cache`` only on a server the operator started for it.

    DEFAULT (``--allow_request_cache_toggle`` off): the argument never applies. The response cache is a
    SERVER-WIDE setting chosen at start (``FW_LM_CACHE=0`` builds every model without it;
    ``dspy_utils.lm_cache_enabled``); a request that rebuilt the process-wide DSPy cache would change every
    other channel's behaviour and drop the memory bound the readiness probe checks. Returns None when the
    request left it unset or agrees with the server; when it disagrees, it is ignored and the returned notice
    (also the ``X-FW-DSPy-Cache`` header and a log line) says what was asked and what the server does.

    OPT-IN (a single local user, the chat page's "LLM cache" checkbox): the argument toggles the process-wide
    cache, as it always did, through ``server_memory.apply_cache_toggle``, which keeps the memory bound so
    readiness stays green. Nothing is reported.
    """
    global _dspy_cache_enabled
    if use_dspy_cache is None:
        return None
    if _allow_request_cache_toggle:
        if use_dspy_cache != _dspy_cache_enabled:
            from fastworkflow.run_fastapi_mcp import server_memory
            server_memory.apply_cache_toggle(use_dspy_cache)
            _dspy_cache_enabled = use_dspy_cache
        return None
    from fastworkflow.utils.dspy_utils import lm_cache_enabled
    server_on = lm_cache_enabled()
    if use_dspy_cache == server_on:
        return None
    notice = (
        f"ignored: use_dspy_cache={'true' if use_dspy_cache else 'false'} was requested, but the response "
        f"cache is a server-wide setting and is {'on' if server_on else 'off'} (start the server with "
        f"FW_LM_CACHE={'1' if use_dspy_cache else '0'} to change it)"
    )
    logger.info(f"use_dspy_cache {notice}")
    return notice


def with_cache_notice(response, notice: Optional[str]):
    """``response`` with the ignored-argument notice as a header, when there is one."""
    if notice:
        response.headers[DSPY_CACHE_HEADER] = notice
    return response


class PerformActionRequest(BaseModel):
    """
    Request to perform a specific action.
    Requires channel_id to be passed in the Authorization header (via JWT token).
    """
    action: dict[str, Any]  # Will be converted to fastworkflow.Action
    timeout_seconds: int = 60


# The taxonomy as TYPES, so the three categories, the six subcategories, the
# target kinds and the provenances appear as enumerations in the generated
# OpenAPI schema and in the MCP tool description. A coding agent reading the
# schema can then see the allowed values instead of having to be told them in
# prose, and an unknown value is refused by the model rather than by a string
# comparison further in. Derived from the taxonomy rather than restated, so the
# schema cannot drift from what the writer accepts.
FeedbackCategory = Literal[tuple(observability_feedback.FEEDBACK_CATEGORIES)]
FeedbackSubcategory = Literal[
    tuple(
        value
        for values in observability_feedback.FEEDBACK_SUBCATEGORIES.values()
        for value in values
    )
]
FeedbackTargetKind = Literal[tuple(observability_feedback.FEEDBACK_TARGET_KINDS)]
FeedbackProvenance = Literal[tuple(sorted(FEEDBACK_PROVENANCES))]


class FeedbackAnchorRequest(BaseModel):
    """The second execution a comparison comment refers to.

    Same vocabulary as the primary target. `store_id` may be given and is
    checked; it must be this channel's evidence store, because the live
    runtime holds exactly one and resolving another database from a request
    would be reaching for evidence nobody authorized. Cross-store pairs are
    recorded through the Observability server, which knows which experiment
    stores are registered.
    """
    turn_key: str
    target_label: str
    target_kind: FeedbackTargetKind = "turn"
    span_ids: list[str] = Field(default_factory=list)
    store_id: Optional[str] = None
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    pass_id: Optional[str] = None


class PostFeedbackRequest(BaseModel):
    """Request to record one review note against recorded evidence.

    This replaced the agent-memory feedback post in fix-9eg.16. The old body
    (`binary_or_numeric_score` / `nl_feedback`) wrote one mutable row per turn
    that was read straight back into the agent's `dspy.History`; that table,
    that route and that injection are gone. What is recorded now is an
    append-only review note on a turn or a component of it, categorized with
    the owner-confirmed taxonomy, and never fed back into a prompt.

    A coding agent and a person post the SAME body. `provenance` says which,
    and `comment` is free-form for both: structure it however you like, but
    the category and subcategory arrive as enums rather than being inferred
    from the text.
    """
    turn_key: str
    target_label: str
    comment: str
    category: FeedbackCategory
    subcategory: FeedbackSubcategory
    target_kind: FeedbackTargetKind = "turn"
    span_ids: list[str] = Field(default_factory=list)
    provenance: FeedbackProvenance = "coding_agent"
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    pass_id: Optional[str] = None
    paired: Optional[FeedbackAnchorRequest] = None

    @field_validator('subcategory')
    @classmethod
    def validate_taxonomy(cls, v, info):
        """Refuse an unpaired category/subcategory at the door, as a 422."""
        category = info.data.get('category')
        if category is None:
            return v
        observability_feedback.validate_category(category, v)
        return v


class ActivateConversationRequest(BaseModel):
    """
    Request to activate a conversation by ID.
    Requires channel_id to be passed in the Authorization header (via JWT token).
    """
    conversation_id: int


class ConversationSummary(BaseModel):
    """One row of GET /conversations.

    Re-homed here from conversation_store.py when the observability DB became
    the single source of truth (Phase 7 §2.8). The wire shape is unchanged —
    NULL topic/summary project to "" and the timestamps stay ms epoch (ruling
    C7) — so a client cannot tell which store answered.
    """
    conversation_id: int
    topic: str
    summary: str
    created_at: int
    updated_at: int


class DumpConversationsRequest(BaseModel):
    """Admin request to dump all conversations"""
    output_folder: str


class GenerateMCPTokenRequest(BaseModel):
    """Request to generate a long-lived MCP token"""
    channel_id: ChannelId
    user_id: Optional[str] = None
    expires_days: int = 365


class CancelPendingRequest(BaseModel):
    """Optional body for /cancel_pending (channel from JWT)."""
    pass


# class CommandOutputWithTraces(BaseModel):
#     """CommandOutput extended with optional traces for HTTP responses"""
#     command_response: dict[str, Any]
#     workflow_name: str = ""
#     context: str = ""
#     command_name: str = ""
#     command_parameters: str = ""
#     success: bool = True
#     traces: Optional[list[dict[str, Any]]] = None


# ============================================================================
# Helper Functions
# ============================================================================

# Create HTTPBearer security scheme instance
# This integrates with FastAPI's OpenAPI/Swagger UI to provide the "Authorize" button
http_bearer = HTTPBearer(
    scheme_name="BearerAuth",
    description="JWT Bearer token obtained from /initialize or /refresh_token endpoint",
    auto_error=True
)

def get_session_from_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer)
) -> SessionData:
    """
    FastAPI dependency to extract and validate session data from JWT Bearer token.
    
    This dependency integrates with FastAPI's security system and Swagger UI:
    - Shows the "Authorize" button in Swagger UI
    - Automatically handles "Bearer " prefix (no need to type it manually)
    - Validates token format and presence
    
    Args:
        credentials: HTTPAuthorizationCredentials from the Authorization header.
                    FastAPI automatically extracts and validates the Bearer token format.
        
    Returns:
        SessionData: Validated session data extracted from the JWT token
        
    Raises:
        HTTPException: If the Authorization header is missing, malformed, or contains an invalid/expired token
        
    Example:
        Use as a dependency in FastAPI endpoints:
        ```python
        @app.post("/endpoint")
        async def endpoint(session: SessionData = Depends(get_session_from_jwt)):
            # Use session.channel_id, session.token_type, etc.
            pass
        ```
        
    HTTP Request Example:
        ```bash
        curl -X POST "http://localhost:8000/endpoint" \\
             -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..." \\
             -H "Content-Type: application/json" \\
             -d '{"data": "value"}'
        ```
    
    Swagger UI Usage:
        1. Click the "Authorize" button (lock icon)
        2. Enter ONLY your JWT token (without "Bearer " prefix)
        3. Swagger UI automatically adds the "Bearer " prefix
    """
    # Extract token from credentials (already validated by HTTPBearer)
    token = credentials.credentials

    # Verify and decode token
    try:
        payload = verify_token(token, expected_type="access")

        # Extract session data from payload, including the token for workflow context
        return SessionData(
            channel_id=payload["sub"],
            user_id=payload.get("uid"),  # Optional user_id from uid claim
            token_type=payload["type"],
            issued_at=payload["iat"],
            expires_at=payload["exp"],
            jti=payload["jti"],
            http_bearer_token=token  # Store the actual token for workflow access
        )

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token missing required claim: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def _merge_workflow_context(
    context: Optional[dict],
    http_bearer_token: Optional[str],
) -> Optional[dict]:
    """Launch context only. The credential is installed per accepted turn.

    It used to be merged in here and refreshed on every lookup, which put a
    request-scoped secret into shared, checkpointable workflow state — readable
    by whichever turn happened to be running, and durable once state is
    persisted. See turns.installed_credential.
    """
    return context


def _runtime_snapshot_for_binding(workflow_path: str) -> Optional[dict[str, Any]]:
    """The server's runtime snapshot to stamp on an attempt it binds (fix-qe2).

    Taken in-process, at the moment of binding, from the same function the
    ``?runtime=true`` probe answers with -- so what the attempt record says
    about its server is what the driver could have asserted against. A
    failure to describe the runtime records None rather than refusing the
    bind: the attempt still runs, and a null stamp is visible evidence that
    the configuration could not be certified, which is more useful than a
    403 that hides it.
    """
    try:
        return runtime_readiness_snapshot(workflow_path)
    except Exception as exc:  # noqa: BLE001 - never block a bind on a probe
        logger.warning(
            f"runtime snapshot unavailable at attempt binding: {type(exc).__name__}"
        )
        return None


def _claim_registered_attempt(
    workflow_path: str,
    channel_id: str,
    bootstrap: ExperimentBootstrapRequest,
) -> tuple[dict[str, Any], SQLiteTraceSink]:
    """Consume a public bootstrap before any WEC or workflow is constructed."""
    trace_sink = get_observability_sink(workflow_path)
    if trace_sink is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "experiment bootstrap requires the observability store, "
                "which could not be opened"
            ),
        )

    try:
        readiness = experiment_store_readiness(trace_sink.store.db_path)
        if readiness["store_id"] != bootstrap.store_id:
            raise StoreIdentityMismatch(
                f"bootstrap targets store {bootstrap.store_id!r}, but this "
                f"server owns {readiness['store_id']!r}"
            )
        claim = trace_sink.store.claim_attempt(
            {
                "registration_id": bootstrap.registration_id,
                "secret": bootstrap.secret,
            },
            channel_id=channel_id,
            server_incarnation=SERVER_INCARNATION,
            runtime_snapshot=_runtime_snapshot_for_binding(workflow_path),
        )
    except MissingExperimentLifecycleFeature as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="experiment claim feature is unavailable",
        ) from exc
    except (AttemptClaimError, StoreIdentityMismatch) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"experiment bootstrap refused: {exc}",
        ) from exc
    return claim, trace_sink


_REGISTERED_CLAIM_IDENTITY_FIELDS = (
    "experiment_id",
    "task_id",
    "attempt",
    "epoch",
    "server_incarnation",
)


def _assert_restored_claim_matches(
    restored_claim: Any,
    active_claim: Optional[dict[str, Any]],
    *,
    source: str,
) -> None:
    """Refuse registered state unless it belongs to the active store claim."""
    if restored_claim is None and active_claim is None:
        return
    if active_claim is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"{source} belongs to a registered experiment channel; "
                "present a new one-use bootstrap"
            ),
        )
    if not isinstance(restored_claim, dict):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{source} has no matching registered experiment claim",
        )
    restored_identity = tuple(
        restored_claim.get(field) for field in _REGISTERED_CLAIM_IDENTITY_FIELDS
    )
    active_identity = tuple(
        active_claim.get(field) for field in _REGISTERED_CLAIM_IDENTITY_FIELDS
    )
    if restored_identity != active_identity:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{source} experiment claim does not match the active claim",
        )


def _run_startup_sync(
    ctx: WorkflowExecutionContext,
    startup_command: Optional[str],
    startup_action: Optional[fastworkflow.Action],
) -> Optional[fastworkflow.TurnOutput]:
    """Run startup as its own logical turn. Mirrors __main__._build_startup_work_fn."""
    if startup_action:
        if startup_action.workflow_id is None and ctx.app_workflow:
            startup_action.workflow_id = ctx.app_workflow.id
        return ctx.process_action_turn(startup_action)
    if startup_command:
        assistant_command = f"/{startup_command.lstrip('/')}"
        return ctx.process_turn(assistant_command)
    return None


async def ensure_user_runtime_exists(
    channel_id: str,
    session_manager: "ChannelSessionManager",
    workflow_path: str,
    context: Optional[dict] = None,
    startup_command: Optional[str] = None,
    startup_action: Optional["fastworkflow.Action"] = None,
    stream_format: str = "ndjson",
    http_bearer_token: Optional[str] = None,
    *,
    run_startup: bool = True,
    experiment_bootstrap: Optional[ExperimentBootstrapRequest] = None,
) -> None:
    """
    Ensure a Topology-B runtime exists for channel_id (WorkflowExecutionContext, no worker thread).
    """
    existing_runtime = await session_manager.get_session(channel_id)
    if existing_runtime:
        if experiment_bootstrap is not None:
            refuse_registered_token_reissue(existing_runtime)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="an experiment bootstrap cannot bind an existing session",
            )
        logger.debug(f"Session for channel_id {channel_id} already exists, skipping creation")
        return

    if session_manager.admission_closed:
        # Creating one now would produce a session the drain has already scanned
        # past, so it would be neither drained nor closed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down; not creating new sessions",
        )

    # Single-flight session creation (§3.5): serialize per-channel so two
    # concurrent cold requests don't both build a ctx (and double-run startup).
    # Bind the lock to a local: the manager's mapping is weak-valued.
    creation_lock = session_manager.get_creation_lock(channel_id)
    async with creation_lock:
        # Re-check under the creation lock — another request may have created
        # the session while we waited.
        existing_runtime = await session_manager.get_session(channel_id)
        if existing_runtime:
            if experiment_bootstrap is not None:
                refuse_registered_token_reissue(existing_runtime)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="an experiment bootstrap cannot bind an existing session",
                )
            logger.debug(
                f"Session for channel_id {channel_id} created concurrently, skipping creation"
            )
            return

        # The hard ceiling: reserve room before anything is built, or refuse (503 + Retry-After). Released in
        # the finally on every path, so a failed creation does not hold a slot.
        try:
            await session_manager.admit_new_session(channel_id)
        except SessionCapacityExceeded as capacity:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(capacity),
                headers={"Retry-After": str(SESSION_CAPACITY_RETRY_AFTER_SECONDS)},
            ) from capacity
        try:
            # Hold the lease across creation so the overflow sweep inside
            # create_session cannot choose the runtime being created (invariant 20).
            async with session_manager.initialization_lease(channel_id):
                await _create_user_runtime(
                    channel_id=channel_id,
                    session_manager=session_manager,
                    workflow_path=workflow_path,
                    context=context,
                    startup_command=startup_command,
                    startup_action=startup_action,
                    stream_format=stream_format,
                    http_bearer_token=http_bearer_token,
                    run_startup=run_startup,
                    experiment_bootstrap=experiment_bootstrap,
                )
                _attach_dropped_session_notice(session_manager, channel_id)
        finally:
            session_manager.release_admission(channel_id)


def _attach_dropped_session_notice(session_manager: "ChannelSessionManager", channel_id: str) -> None:
    """If this channel's earlier session was dropped by the server, put the notice on the new one (once).

    Fails open: the notice describes a loss that already happened, so a store that cannot be read is logged and
    skipped, never allowed to fail the request that created the session.
    """
    try:
        notice = session_manager.notice_store.take(channel_id)
        runtime = session_manager._sessions.get(channel_id)
        if notice is not None and runtime is not None:
            runtime.session_notice = notice
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Dropped-session notice not attached ({type(exc).__name__}: {exc})")


@dataclass
class _RestoredCheckpoint:
    """What a checkpoint contributed to a freshly built runtime."""

    session_incarnation: str
    runtime_fields: dict = field(default_factory=dict)
    startup: dict = field(default_factory=dict)
    active_claim: Optional[dict[str, Any]] = None
    applied: bool = False


def _restore_from_checkpoint(
    session_manager: "ChannelSessionManager",
    channel_id: str,
    workflow_path: str,
    app_workflow: fastworkflow.Workflow,
    launch_context: Optional[dict],
    active_claim: Optional[dict[str, Any]],
    claim_store: Optional[ObservabilityStore],
    allow_registered_restore: bool,
) -> _RestoredCheckpoint:
    """Apply this channel's checkpoint, or start from launch configuration.

    A record that cannot be applied is quarantined and the channel starts clean.
    Partially applying one would leave a session that looks restored and is not.
    """
    store = session_manager.checkpoint_store
    fresh_incarnation = checkpoint.new_session_incarnation()

    try:
        # Adoption, not continuation: a cold session has no way to know which
        # incarnation the stored record names until it reads it. Every other
        # identity field is still validated.
        record = store.load_for_adoption(
            deployment_id=checkpoint.deployment_id(),
            workflow_fingerprint=checkpoint.workflow_fingerprint(workflow_path),
            channel_id=channel_id,
        )
    except Exception as exc:
        logger.warning(
            f"Checkpoint for channel_id {channel_id} could not be read "
            f"({type(exc).__name__}); starting from launch configuration"
        )
        record = None

    if record is None:
        return _RestoredCheckpoint(
            session_incarnation=fresh_incarnation,
            active_claim=active_claim,
        )

    restored_claim = (record.runtime or {}).get("experiment_claim")
    if active_claim is None and restored_claim is not None:
        if not allow_registered_restore:
            _assert_restored_claim_matches(
                restored_claim,
                None,
                source="checkpoint",
            )
        if claim_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "registered checkpoint restore requires the active "
                    "experiment claim store"
                ),
            )
        if restored_claim.get("server_incarnation") != SERVER_INCARNATION:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "checkpoint experiment claim belongs to a different "
                    "server incarnation"
                ),
            )
        restored_conversation_id = (record.runtime or {}).get(
            "active_conversation_id"
        )
        if not restored_conversation_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="registered checkpoint has no claimed conversation",
            )
        active_claim = {
            **restored_claim,
            "channel_id": channel_id,
            "conversation_id": int(restored_conversation_id),
        }
        try:
            claim_store.validate_attempt_claim(active_claim)
        except AttemptClaimError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="checkpoint experiment claim is no longer current",
            ) from exc
    else:
        # Validate identity before checkpoint.restore can quarantine malformed
        # registered state and silently continue as an ordinary session.
        _assert_restored_claim_matches(
            restored_claim,
            active_claim,
            source="checkpoint",
        )

    try:
        runtime_fields = checkpoint.restore(
            app_workflow,
            record,
            current_launch=dict(launch_context or {}),
            channel_id=channel_id,
        )
    except Exception as exc:
        logger.error(
            f"Checkpoint for channel_id {channel_id} could not be applied "
            f"({type(exc).__name__}: {exc}); quarantining and starting from "
            "launch configuration"
        )
        store.quarantine(record.identity, QuarantineReason.UNREADABLE_RECORD)
        return _RestoredCheckpoint(session_incarnation=fresh_incarnation)

    _assert_restored_claim_matches(
        runtime_fields.get("experiment_claim"),
        active_claim,
        source="checkpoint",
    )
    logger.info(f"Restored checkpoint for channel_id {channel_id}")
    return _RestoredCheckpoint(
        session_incarnation=record.identity.session_incarnation,
        runtime_fields=runtime_fields,
        startup=dict(record.startup or {}),
        active_claim=active_claim,
        applied=True,
    )


def _restore_conversation_memory(
    ctx: WorkflowExecutionContext,
    trace_sink: Optional[SQLiteTraceSink],
    channel_id: str,
) -> Optional[int]:
    """Rebuild in-memory conversation history from the observability DB.

    Gate 1 of the Phase-7 consolidation ([R3], §2.3): the memory window comes
    from the turns table rather than the retired per-channel store. Only the
    window the running session keeps is read, not the whole conversation — a
    turn record carries its full payload, so hydrating a long conversation to
    then discard all but the newest few is the memory spike the bounded window
    exists to prevent.

    Which conversation: the newest for the channel, stepping back one if it has
    no usable turns yet (the reserved-but-empty state a rotate leaves behind).
    Emptiness is zero USABLE turn rows, not an absent row — the store mints a
    conversations row at reserve time, so row-presence would never step back
    (ruling I7). The stepped-back conversation becomes the active one too, so
    the next turn continues it rather than starting a third.

    Returns the conversation id this session should be BOUND to, which is not
    always one it restored history from. A newest-but-empty conversation is
    reused rather than stepped over: it is what a rotate just reserved for the
    next exchange, so minting again would strand it and add a conversations row
    per cold start on an idle channel. None means there is nothing to reuse and
    the caller should mint.
    """
    if trace_sink is None:
        return None
    store = trace_sink.store
    reusable_empty: Optional[int] = None
    for conv_id in store.newest_conversation_ids(channel_id, limit=2):
        window = store.get_memory_window(
            channel_id, conv_id, MAX_CONVERSATION_TURNS_IN_MEMORY
        )
        if not window:
            if reusable_empty is None:
                reusable_empty = conv_id
            continue
        ctx._conversation_history = restore_history_from_turns(window)
        ctx.bind_last_completed_turn_key(
            store.get_last_completed_turn_key(channel_id, conv_id)
        )
        logger.info(f"Restored conversation {conv_id} for user {channel_id}")
        return conv_id
    if reusable_empty is not None:
        logger.info(
            f"Reusing empty conversation {reusable_empty} for user {channel_id}"
        )
    return reusable_empty


async def _create_user_runtime(
    channel_id: str,
    session_manager: "ChannelSessionManager",
    workflow_path: str,
    context: Optional[dict],
    startup_command: Optional[str],
    startup_action: Optional["fastworkflow.Action"],
    stream_format: str,
    http_bearer_token: Optional[str],
    run_startup: bool,
    experiment_bootstrap: Optional[ExperimentBootstrapRequest],
) -> None:
    """Build and register a fresh Topology-B runtime (caller holds creation lock)."""
    context = _merge_workflow_context(context, http_bearer_token)
    # Phase 9 step 4: refuse an app whose artifacts predate the not-here construction, before the
    # experiment claim below reserves anything. The CME is checked by WorkflowExecutionContext.
    from fastworkflow.train.artifact_versioning import require_construction_record
    require_construction_record(workflow_path)
    logger.info(f"Creating new Topology-B session for channel_id: {channel_id}")

    claim: Optional[dict[str, Any]] = None
    trace_sink = get_observability_sink(workflow_path)
    if experiment_bootstrap is not None:
        claim, trace_sink = _claim_registered_attempt(
            workflow_path, channel_id, experiment_bootstrap
        )

    # The claim transaction above reserves the labelled conversation before
    # this constructor can allocate any session work.
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id, detect_no_agent_mode=True)
    # Identity plumbing [R1]: the embedder binds channel identity before any
    # turn so spans/turn records are attributable (conversation ids are minted
    # by the observability store from Phase 2).
    ctx.bind_observability_identity(
        channel_id=channel_id,
        # FastAPI mints conversation ids through reserve_conversation_id (the
        # store's per-channel counter, ruling C2); WEC self-minting must stay
        # off so a degraded eager mint cannot double-mint.
        embedder_owns_conversations=True,
    )
    # Observability sink [R4]: recording is always on. One sink (one writer
    # thread) per workflow DB, shared across channels; it is None only when
    # the store could not be opened.
    if trace_sink is not None:
        ctx.set_trace_sink(trace_sink)
    if claim is not None:
        ctx.bind_experiment_claim(claim, trace_sink.store)
    trace_queue: Queue = BoundedTraceQueue()
    ctx.set_transport_queues(command_trace_queue=trace_queue)

    app_workflow = fastworkflow.Workflow.create(
        workflow_path,
        workflow_id_str=channel_id,
        workflow_context=context,
    )
    ctx.bind_app_workflow(app_workflow)

    restored = _restore_from_checkpoint(
        session_manager,
        channel_id,
        workflow_path,
        app_workflow,
        context,
        claim,
        trace_sink.store if trace_sink is not None else None,
        http_bearer_token is not None,
    )
    claim = restored.active_claim
    if claim is not None:
        ctx.bind_experiment_claim(claim, trace_sink.store)

    conv_id_to_restore = _restore_conversation_memory(
        ctx, trace_sink, channel_id
    )

    loop = asyncio.get_running_loop()
    startup_state = restored.startup.get("state") or checkpoint.STARTUP_NOT_ATTEMPTED
    startup_epoch = int(restored.startup.get("epoch") or 0)
    startup_key = restored.startup.get("idempotency_key")
    startup_ran = bool(restored.runtime_fields.get("startup_ran"))

    # Whether startup already ran is read from the durable record, never from
    # in-process turn retention, so it cannot depend on a retention window.
    already_succeeded = startup_state == checkpoint.STARTUP_SUCCEEDED
    if run_startup and (startup_command or startup_action) and not already_succeeded:
        await loop.run_in_executor(
            None,
            lambda: _run_startup_sync(ctx, startup_command, startup_action),
        )
        startup_ran = True
        startup_state = checkpoint.STARTUP_SUCCEEDED
    elif already_succeeded:
        logger.info(
            f"Skipping startup for channel_id {channel_id}: the durable record "
            f"says it already succeeded at epoch {startup_epoch}"
        )

    if pending := session_manager.session_state_store.load(channel_id):
        # fix-ayez: a restored blob drops parked state of conversations the store no longer has, so a deleted or
        # reused id cannot resurrect it. The blob's own conversation is kept.
        if _prune_parked_blob(
            pending, channel_id,
            trace_sink.store if trace_sink is not None else None,
            keep=pending.get("conversation_id"),
        ):
            session_manager.session_state_store.save(channel_id, pending)
            logger.info(f"Pruned parked pending state of deleted conversations for channel_id {channel_id}")
        pending_claim = (
            {
                "experiment_id": pending.get("experiment_id"),
                "task_id": pending.get("task_id"),
                "attempt": pending.get("attempt"),
                "epoch": pending.get("claim_epoch"),
                "server_incarnation": pending.get("server_incarnation"),
            }
            if pending.get("experiment_id") is not None
            else None
        )
        _assert_restored_claim_matches(
            pending_claim,
            claim,
            source="pending session state",
        )
        try:
            ctx.apply_serialized_state(pending)
        except IncompatibleSessionState as exc:
            # Drop it rather than retrying forever: nothing about a later read
            # makes an unreadable version readable, and leaving it in place
            # would re-raise on every cold create for this channel.
            session_manager.session_state_store.clear(channel_id)
            logger.error(
                f"Discarded unreadable pending state for channel_id {channel_id}: "
                f"{exc}. The suspended turn is lost; the session starts clean."
            )
        else:
            if claim is not None:
                # apply_serialized_state restores fencing values but cannot
                # restore the live store object used by admission checks.
                ctx.bind_experiment_claim(claim, trace_sink.store)
            logger.info(
                f"Restored pending suspended session for channel_id {channel_id}"
            )

    # A checkpointed active conversation wins over the one memory was restored
    # from (ruling I7); otherwise the restore's choice — including its
    # step-back — is the active conversation.
    if claim is not None:
        restored_conversation_id = restored.runtime_fields.get(
            "active_conversation_id"
        )
        if (
            restored.applied
            and restored_conversation_id is not None
            and int(restored_conversation_id) != int(claim["conversation_id"])
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "checkpoint conversation does not match the active "
                    "experiment claim"
                ),
            )
        active_conversation_id = int(claim["conversation_id"])
    else:
        active_conversation_id = (
            restored.runtime_fields.get("active_conversation_id")
            if restored.applied
            else conv_id_to_restore
        )
    # [R1] conversation-id reservation moves ahead of the turn: a fresh session
    # mints its first conversation id NOW, so the very first turn's records are
    # conversation-attributed. Restored sessions keep their restored id.
    if not active_conversation_id and trace_sink is not None:
        active_conversation_id = _mint_conversation_id(trace_sink, channel_id)
    if active_conversation_id:
        ctx.bind_observability_identity(
            conversation_id=int(active_conversation_id)
        )

    # A restored record is authoritative for the fields the framework knows how
    # to restore; the request's stream_format only applies to a fresh session.
    await session_manager.create_session(
        channel_id=channel_id,
        execution_context=ctx,
        active_conversation_id=active_conversation_id,
        stream_format=(
            restored.runtime_fields.get("stream_format") or stream_format
            if restored.applied
            else stream_format
        ),
        workflow_path=workflow_path,
        startup_ran=startup_ran,
        session_incarnation=restored.session_incarnation,
        startup_state=startup_state,
        startup_idempotency_key=startup_key,
        startup_epoch=startup_epoch,
    )
    logger.info(f"Successfully created session for channel_id: {channel_id}")


def get_channel_session_state_dir(workflow_path: str) -> str:
    """Workflow-namespaced folder for suspended Topology-B blobs (created)."""
    from fastworkflow.session import state_paths
    return state_paths.session_state_dir(workflow_path)


def _is_awaiting_user_output(output: Optional[fastworkflow.TurnOutput]) -> bool:
    """Read suspension off the turn's own status, not out of a command's artifacts.

    The artifacts key this used to inspect is a per-command detail that happens
    to correlate with suspension; ``TurnStatus`` is the turn-level statement of
    it, which is what a caller deciding whether to persist a suspended turn is
    actually asking about.
    """
    return (
        output is not None
        and output.status == fastworkflow.TurnStatus.AWAITING_USER
    )


def persist_pending_after_turn(
    session_manager: "ChannelSessionManager",
    runtime: "ChannelRuntime",
    output: Optional[fastworkflow.TurnOutput],
) -> None:
    """Save or clear durable suspended state after a Topology-B turn."""
    ctx = runtime.execution_context
    # has_open_command(): a mid-parameter-extraction session is not
    # awaiting_user but still holds partially extracted parameters (see the
    # twin in turns.persist_pending_after_turn).
    # A pending approval (ADR-EXP-025) is durable suspended state too: it survives eviction and restart.
    pending_approval = approval_binding.active_record(ctx.app_workflow) is not None
    # fix-ayez: parked state of a conversation the store no longer has is dropped before the save decides what to keep.
    prune_parked_pending(runtime)
    # Parked state (fix-newconv-pending) is saved too: it is kept for its conversation until that one is active again.
    if ctx.awaiting_user or _is_awaiting_user_output(output) or ctx.has_open_command() or pending_approval or ctx.has_parked_pending():
        try:
            state = ctx.serialize_state(channel_id=runtime.channel_id)
        except StateEncodingError as exc:
            logger.warning(
                f"Suspended state for channel_id {runtime.channel_id} is not "
                f"losslessly encodable, so it was not persisted: {exc}"
            )
            return
        session_manager.session_state_store.save(runtime.channel_id, state)
    else:
        session_manager.session_state_store.clear(runtime.channel_id)


def _existing_conversation_ids(store, channel_id: str, conversation_ids: list[int]) -> Optional[set[int]]:
    """The ids that still name a live conversation, or None when the store cannot say (then nothing is pruned)."""
    try:
        return store.existing_conversation_ids(channel_id, conversation_ids)
    except Exception as exc:  # noqa: BLE001 - an unreadable store must not fail a save; it prunes nothing
        logger.warning(
            f"Parked pending state of channel_id {channel_id} not pruned: the conversation store could not be read "
            f"({type(exc).__name__}: {exc})"
        )
        return None


def prune_parked_pending(runtime: "ChannelRuntime") -> int:
    """Drop parked pending state whose conversation the observability store no longer has (fix-ayez).

    Archiving and deleting conversations happen in the run_chatbot server, which cannot reach this process's parked
    state, so the parked state is checked against the store whenever it is about to be saved, and again before a
    conversation id is minted (a reused id must not inherit a deleted conversation's state). The bound conversation
    keeps its state, and the "none" and unowned keys are not conversation ids. No store, or an unreadable one, prunes
    nothing. Returns the number of parked snapshots dropped.
    """
    ctx = runtime.execution_context
    if not ctx.has_parked_pending():
        return 0
    store = getattr(runtime, "observability_store", None)
    if store is None:
        return 0
    bound = ctx.bound_conversation_id()
    ids = [cid for cid in ctx.parked_conversation_ids() if cid != bound]
    if not ids:
        return 0
    live = _existing_conversation_ids(store, runtime.channel_id, ids)
    if live is None:
        return 0
    return sum(ctx.drop_parked_pending(cid) for cid in ids if cid not in live)


def _prune_parked_blob(state: dict, channel_id: str, store, keep) -> int:
    """The same prune on a saved blob, before it is restored (fix-ayez). Edits *state* in place; returns the count."""
    parked = dict(state.get("parked_pending") or {})
    ids = [int(key) for key in parked if key.isdigit() and int(key) != keep]
    if not ids or store is None:
        return 0
    live = _existing_conversation_ids(store, channel_id, ids)
    if live is None:
        return 0
    dropped = 0
    for cid in ids:
        if cid not in live:
            dropped += len(parked.pop(str(cid), None) or [])
    if dropped:
        state["parked_pending"] = parked
    return dropped


def drop_parked_pending_for_conversation(
    session_manager: "ChannelSessionManager",
    channel_id: str,
    conversation_id: int,
    runtime: "Optional[ChannelRuntime]" = None,
) -> int:
    """Drop the pending state parked for a deleted or archived conversation (fix-ayez).

    A loaded runtime is changed in memory and saved through persist_pending_after_turn, the helper every other path
    uses. With no loaded runtime the saved blob is edited directly: nothing live can re-save the entry. Returns the
    number of parked snapshots dropped.
    """
    if runtime is not None:
        dropped = runtime.execution_context.drop_parked_pending(conversation_id)
        persist_pending_after_turn(session_manager, runtime, None)
        return dropped
    state = session_manager.session_state_store.load(channel_id)
    if not state:
        return 0
    parked = dict(state.get("parked_pending") or {})
    dropped = parked.pop(str(conversation_id), None) or []
    if not dropped:
        return 0
    state["parked_pending"] = parked
    session_manager.session_state_store.save(channel_id, state)
    return len(dropped)


# ----------------------------------------------------------------------
# The approval surface of a request (fix-rkr1)
# ----------------------------------------------------------------------
# /invoke_agent and /perform_action are always a host surface. --chat_approval_surface picks what the
# streaming endpoint (also the MCP invoke_agent tool) and /invoke_assistant pass; it never changes the
# first two. The default is ``host`` (strict: one rule on every endpoint and every MCP tool); ``channel``
# passes no surface, so the turn falls back to its channel surface (a plain yes approves, no token), which
# is what the local `run_chatbot` server pins explicitly.
CHAT_APPROVAL_SURFACES = ("channel", "host")
DEFAULT_CHAT_APPROVAL_SURFACE = "host"
_chat_approval_surface = DEFAULT_CHAT_APPROVAL_SURFACE


def set_chat_approval_surface(mode: str) -> None:
    """Set at startup from --chat_approval_surface (like jwt_manager.set_jwt_verification_mode)."""
    global _chat_approval_surface
    if mode not in CHAT_APPROVAL_SURFACES:
        raise ValueError(f"chat approval surface must be one of {CHAT_APPROVAL_SURFACES}, got {mode!r}")
    _chat_approval_surface = mode


def chat_approval_surface() -> str:
    return _chat_approval_surface


def approval_surface_for(user_id: Optional[str], *, scope: str = "direct") -> "approval_binding.Surface | None":
    """The approval surface one request passes to the turn.

    scope "direct" (/invoke_agent, /perform_action): always the host surface, verified only when
    JWT signatures are checked. Never loosened by --chat_approval_surface.

    scope "chat" (/invoke_agent_stream, /invoke_assistant): under ``channel`` this returns the channel
    surface with the chat origin (its standing approvals are the chat's own, not the CLI's). Under ``host`` (the
    default) it returns the same surface scope "direct" builds.
    """
    if scope not in ("direct", "chat"):
        raise ValueError(f"unknown approval scope {scope!r}")
    if scope == "chat" and _chat_approval_surface != "host":
        # The channel surface of the chat server: the origin is set here, on the server side, so its standing
        # approvals are the chat's own and never the CLI's (fix-d6no). Nothing a caller sends reaches it.
        return approval_binding.Surface(kind="channel", origin=approval_grants.ORIGIN_CHAT)
    return approval_binding.Surface(
        kind="host", verified=bool(jwt_manager.EXPECT_ENCRYPTED_JWT), uid=user_id)


def turn_approval_kwargs(surface: "approval_binding.Surface | None", token: Optional[str]) -> dict:
    """process_turn keyword arguments for a surface: none at all when there is no surface, so the call
    is the one made before the option existed; the token only travels with a surface."""
    if surface is None:
        return {}
    # A channel surface has no token to check (a plain yes approves), so none travels with it.
    return {"approval_surface": surface, "approval_token": token if surface.kind == "host" else None}


async def run_process_turn(
    runtime: "ChannelRuntime",
    message: str,
    timeout_seconds: int,
    session_manager: "ChannelSessionManager",
    unattended: bool = False,
    approval_surface: "approval_binding.Surface | None" = None,
    approval_token: Optional[str] = None,
    delivered: Optional[dict] = None,
    use_plan: Optional[str] = None,
) -> fastworkflow.TurnOutput:
    """Run one logical turn in a thread pool with timeout (Topology B).

    ``delivered``, when given, receives the host's live approval fields (the token) taken after the
    turn on the same thread; they are never part of the returned TurnOutput.
    """
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context

    def _run() -> fastworkflow.TurnOutput:
        # process_turn() is the public execution API: it returns the TurnOutput
        # the whole transport edge speaks (and captures the full turn).
        output = ctx.process_turn(
            message, unattended=unattended,
            **turn_approval_kwargs(approval_surface, approval_token),
            **({"use_plan": use_plan} if use_plan else {}))
        if delivered is not None:
            delivered.update(approval_binding.take_host_delivery(ctx.app_workflow))
        return output

    try:
        output = await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        logger.error(
            f"Command execution timed out after {timeout_seconds}s "
            f"for channel_id: {runtime.channel_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Command execution timed out after {timeout_seconds} seconds",
        ) from exc

    persist_pending_after_turn(session_manager, runtime, output)
    return output


async def run_process_action(
    runtime: "ChannelRuntime",
    action: fastworkflow.Action,
    timeout_seconds: int,
    session_manager: "ChannelSessionManager",
) -> fastworkflow.TurnOutput:
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context

    def _run() -> fastworkflow.TurnOutput:
        return ctx.process_action_turn(action)

    try:
        output = await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Action execution timed out after {timeout_seconds} seconds",
        ) from exc

    persist_pending_after_turn(session_manager, runtime, output)
    return output


def _format_trace_event(evt: Any, user_id: Optional[str]) -> dict[str, Any]:
    trace = {
        "direction": evt.direction.value if hasattr(evt.direction, "value") else str(evt.direction),
        "raw_command": evt.raw_command,
        "command_name": evt.command_name,
        "parameters": evt.parameters,
        "response_text": evt.response_text,
        "success": evt.success,
        "timestamp_ms": evt.timestamp_ms,
    }
    if user_id is not None:
        trace["user_id"] = user_id
    return trace


async def _maybe_await(result: Any) -> None:
    """Accept either a sync or an async callback."""
    if asyncio.iscoroutine(result):
        await result


async def _emit_trace_callback(
    on_trace: Callable[[dict[str, Any]], Any],
    trace_dict: dict[str, Any],
    _user_id: Optional[str],
) -> None:
    await _maybe_await(on_trace(trace_dict))


async def _deliver_trace_or_progress(
    evt: Any,
    on_trace: Callable[[dict[str, Any]], Any],
    on_progress: Optional[Callable[[dict[str, Any]], Any]],
    user_id: Optional[str],
) -> None:
    """A stage-progress event goes to ``on_progress`` (dropped when there is none or it fails); a trace event to ``on_trace``."""
    if isinstance(evt, ProgressEvent):
        if on_progress is not None:
            try:
                await _maybe_await(on_progress(evt.to_dict()))
            except Exception as exc:  # progress is advisory; it never fails the turn
                logger.debug(f"progress callback failed: {exc!r}")
        return
    await _emit_trace_callback(on_trace, _format_trace_event(evt, user_id), user_id)


async def run_process_message_with_trace_stream(
    runtime: "ChannelRuntime",
    message: str,
    timeout_seconds: int,
    session_manager: "ChannelSessionManager",
    on_trace: Callable[[dict[str, Any]], Any],
    user_id: Optional[str] = None,
    on_timeout: Optional[Callable[[str], Any]] = None,
    unattended: bool = False,
    on_progress: Optional[Callable[[dict[str, Any]], Any]] = None,
    approval_surface: "approval_binding.Surface | None" = None,
    approval_token: Optional[str] = None,
    delivered: Optional[dict] = None,
    use_plan: Optional[str] = None,
) -> fastworkflow.TurnOutput:
    """
    Run one logical turn in an executor while draining command_trace_queue concurrently.

    The deadline governs *delivery*, not ownership. When it passes, ``on_timeout``
    is invoked once so the caller can tell the client, but the executor future is
    still awaited to completion: abandoning it would leave a thread mutating the
    workflow context after the caller released ``runtime.lock``, free for eviction
    or shutdown to snapshot and close that context underneath it.
    """
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context
    trace_queue = ctx.command_trace_queue
    if trace_queue is None:
        return await run_process_turn(
            runtime, message, timeout_seconds, session_manager,
            unattended=unattended, approval_surface=approval_surface,
            approval_token=approval_token, delivered=delivered, use_plan=use_plan,
        )

    def _run() -> fastworkflow.TurnOutput:
        output = ctx.process_turn(
            message, unattended=unattended,
            **turn_approval_kwargs(approval_surface, approval_token),
            **({"use_plan": use_plan} if use_plan else {}))
        if delivered is not None:
            delivered.update(approval_binding.take_host_delivery(ctx.app_workflow))
        return output

    exec_future = loop.run_in_executor(None, _run)
    start = time.time()
    timed_out = False

    while not exec_future.done():
        while True:
            try:
                evt = trace_queue.get_nowait()
            except queue.Empty:
                break
            if evt is None:
                continue
            await _deliver_trace_or_progress(evt, on_trace, on_progress, user_id)
        if not timed_out and time.time() - start >= timeout_seconds:
            timed_out = True
            logger.warning(
                f"Streaming turn for channel_id {runtime.channel_id} passed its "
                f"{timeout_seconds}s delivery deadline; still owning the executor "
                "until it exits"
            )
            if on_timeout is not None:
                await _maybe_await(
                    on_timeout(
                        f"Command execution timed out after {timeout_seconds} seconds"
                    )
                )
        await asyncio.sleep(0.05)

    output = await exec_future

    while True:
        try:
            evt = trace_queue.get_nowait()
        except queue.Empty:
            break
        if evt is None:
            continue
        await _deliver_trace_or_progress(evt, on_trace, on_progress, user_id)
    persist_pending_after_turn(session_manager, runtime, output)
    return output


def collect_trace_events(runtime: "ChannelRuntime", user_id: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Drain and collect all trace events from the queue.
    
    Args:
        runtime: ChannelRuntime containing the trace queue
        user_id: Optional user_id to include in traces
        
    Returns:
        List of trace event dictionaries with optional user_id
    """
    traces = []
    
    trace_queue = runtime.execution_context.command_trace_queue
    if trace_queue is None:
        return traces

    while True:
        try:
            evt = trace_queue.get_nowait()
            if evt is None:
                break
            if isinstance(evt, ProgressEvent):
                continue
            traces.append(_format_trace_event(evt, user_id))
        except queue.Empty:
            break

    return traces


async def collect_trace_events_async(
    trace_queue: queue.Queue,
    user_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Async version: Drain and collect all trace events from a trace queue.
    
    Args:
        trace_queue: The trace queue to drain
        user_id: Optional user_id to include in traces
        
    Returns:
        List of trace event dictionaries with optional user_id
    """
    traces = []
    
    while True:
        try:
            evt = trace_queue.get_nowait()
            if evt is None:
                break
            if isinstance(evt, ProgressEvent):
                continue
            trace = {
                "direction": evt.direction.value if hasattr(evt.direction, 'value') else str(evt.direction),
                "raw_command": evt.raw_command,
                "command_name": evt.command_name,
                "parameters": evt.parameters,
                "response_text": evt.response_text,
                "success": evt.success,
                "timestamp_ms": evt.timestamp_ms
            }
            if user_id is not None:
                trace["user_id"] = user_id
            traces.append(trace)
        except queue.Empty:
            break
    
    return traces


# ============================================================================
# Session Management
# ============================================================================

@dataclass
class ChannelRuntime:
    """Per-channel Topology-B runtime (live WEC + metadata)."""

    channel_id: str
    active_conversation_id: int
    execution_context: WorkflowExecutionContext
    lock: asyncio.Lock
    # Serializes topic/summary generation for this channel. Deliberately NOT
    # ``lock``: that one is a busy signal (busy_channel_ids reads
    # lock.locked()), so holding it across an LLM round trip would put topic
    # generation back inside the shutdown drain this whole change exists to keep
    # it out of.
    #
    # THE GUARANTEE THIS BUYS, STATED HONESTLY (fix-dzs.6, R9): at most one
    # topic generation per *writer*, not one per conversation. This is an
    # asyncio.Lock, so it serializes only coroutines in THIS event loop. Per the
    # ratified single-writer-per-channel invariant (R27 of the TurnResult review,
    # docs/turn_result_design_review.md:1272-1285) a channel is expected to be
    # served by one process at a time; where that expectation is broken - two
    # pods serving one channel - both can generate a label for the same
    # conversation. That is bounded waste, not corruption: each writes through
    # ObservabilityStore.record_conversation_label, whose self-excluding
    # uniqueness check (fix-dzs.2) makes a rewrite idempotent, so the loser's
    # title is simply replaced. Nothing here is a distributed lock and nothing
    # should be built on it as if it were.
    label_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stream_format: str = "ndjson"
    workflow_path: str = ""
    startup_ran: bool = False
    # Startup authority is durable (invariant 18): whether startup already ran
    # is read from the checkpoint, never inferred from in-process turn
    # retention, so it cannot depend on a wall-clock retention window.
    startup_state: str = checkpoint.STARTUP_NOT_ATTEMPTED
    startup_idempotency_key: Optional[str] = None
    startup_epoch: int = 0
    # Binds this live session to the checkpoint it may write. A record carrying
    # a different incarnation is a channel-id reuse, and is quarantined rather
    # than applied.
    session_incarnation: str = field(default_factory=checkpoint.new_session_incarnation)
    # DEAD FIELD, kept for checkpoint compatibility (Phase 7 ruling C5). It was
    # the high-water mark for the legacy store's incremental turn save; under
    # sync-first turn records there is nothing incremental to track, because
    # every turn is recorded by the sink under its own turn_key at finalize.
    # Still written to checkpoints (as 0) and still read without effect, so a
    # checkpoint from before the cutover deserializes.
    durable_turn_count: int = 0
    # turn_key of the startup turn (if any), so the /initialize "already exists"
    # branch can return its three-state status (§3.3).
    startup_turn_key: Optional[str] = None
    # Monotonic time of the channel's last use; the idle age that session expiry reads. Updated by every lookup
    # and when a turn finishes (ChannelSessionManager.note_activity).
    last_active: float = field(default_factory=time.monotonic)
    # A dropped-session notice (session_notices) waiting for this session's first reply; taken exactly once.
    session_notice: Optional[dict[str, Any]] = None

    @property
    def chat_session(self) -> WorkflowExecutionContext:
        """Backward-compatible alias for endpoints that referenced chat_session."""
        return self.execution_context

    @property
    def observability_store(self) -> Optional[ObservabilityStore]:
        """The conversation record for this channel, or None when unavailable.

        Single source of truth since Phase 7: conversations, turns, memory
        rebuild and feedback all live in this one per-workflow DB. None means
        the store could not be opened, in which case the session keeps
        in-memory history only and nothing survives a restart.
        """
        sink = getattr(self.execution_context, "trace_sink", None)
        return sink.store if isinstance(sink, SQLiteTraceSink) else None


def refuse_registered_token_reissue(runtime: ChannelRuntime) -> None:
    """Keep unauthenticated initialize retries compatible only for ordinary channels."""
    if runtime.execution_context.observability_experiment_claim:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "registered experiment channels cannot reissue tokens from "
                "channel knowledge; present a valid credential or a new "
                "one-use bootstrap"
            ),
        )


class ChannelSessionManager:
    """
    Process-local cache of live WorkflowExecutionContext instances.

    Suspended (awaiting_user) state is persisted via SessionStateStore so any
    worker can cold-rehydrate after eviction or restart.
    """

    def __init__(
        self,
        session_state_store: Optional[SessionStateStore] = None,
        max_live_sessions: int = DEFAULT_MAX_LIVE_SESSIONS,
        max_live_sessions_hard: Optional[int] = None,
        pinned_idle_ttl_seconds: int = DEFAULT_PINNED_IDLE_TTL_SECONDS,
        protected_idle_ttl_seconds: int = DEFAULT_PROTECTED_IDLE_TTL_SECONDS,
    ):
        self._sessions: OrderedDict[str, ChannelRuntime] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_live_sessions = max_live_sessions
        self._max_live_sessions_source = "constructor"
        # The hard ceiling (None: DEFAULT_HARD_CEILING_FACTOR x the soft target) and the two idle TTLs that let
        # the server release sessions it can never checkpoint. See admit_new_session.
        self._max_live_hard = max_live_sessions_hard
        self._max_live_hard_source = "constructor" if max_live_sessions_hard is not None else "default"
        self._pinned_idle_ttl = pinned_idle_ttl_seconds
        self._protected_idle_ttl = protected_idle_ttl_seconds
        self._ttl_sources = ("default", "default")
        # Channels admitted by admit_new_session whose session is not registered yet. They count toward the
        # ceiling, or concurrent creators would each see room and overshoot it.
        self._creating: set[str] = set()
        # Bumped whenever a session is added or removed; lets a "nothing can be released" verdict be reused
        # briefly instead of rescanning every session (and running every get_state hook) per refused request.
        self._membership_version = 0
        self._saturated: Optional[tuple[int, float]] = None
        self.refused_total = 0
        self.expired_total = 0
        self._notice_store: Optional[SessionNoticeStore] = None
        # A FastAPI process serves exactly one workflow; this is set at server
        # startup (before any store access) so the lazily-built stores below can
        # namespace their paths under that workflow. See set_workflow_path.
        self.workflow_path: str = ""
        # Built lazily on first access so the state-root read happens after
        # fastworkflow.init() loads the env file, not at module-import time.
        self._session_state_store = session_state_store
        # Per-channel creation guard for single-flight session creation: two
        # concurrent cold requests for the same channel must not both build a
        # ctx (wasted work / double startup). Keyed by channel_id; dict access
        # is atomic in the single event loop.
        #
        # Weak values, because a channel_id seen once would otherwise keep its
        # lock forever. The holder and every queued waiter keep a strong
        # reference for as long as the lock matters, so the entry only
        # disappears once nobody is using it. Never prune it by hand: a waiter
        # can be awake but not yet resumed, and dropping the shared lock in that
        # window lets the next caller create a second one and lose single-flight.
        self._creation_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = (
            weakref.WeakValueDictionary()
        )
        # Optional predicate (channel_id -> bool) wired by the server to the
        # turn registry's active-execution pointer. Eviction must never close a
        # live turn's ctx, so a busy channel is skipped. See §3.6 of the design.
        self.is_channel_busy: Optional[Callable[[str], bool]] = None
        # Per-channel eviction leases (refcounts). A lease is taken under the
        # manager lock by the same lookup that hands out the runtime, which is
        # what makes lookup and admission atomic. A boolean sampled afterwards
        # cannot: get_session() releases the manager lock before the caller has
        # registered a turn or taken runtime.lock, and in that window both halves
        # of the busy predicate are false while the runtime is very much in use.
        # Keyed by channel_id rather than by runtime, so a channel can be leased
        # while it is still being created.
        self._leases: dict[str, int] = {}
        self._checkpoint_store: Optional[ChannelCheckpointStore] = None
        self._admission_closed = False
        # Wired by the server to the turn registry: forget the channel's active-turn pointer on end_session.
        self.release_channel_turns: Optional[Callable[[str], None]] = None
        # The configured initial context, kept so restore can tell an operator's
        # change from an application's (three-way merge, design 11.9).
        self._launch_context: dict = {}

    @asynccontextmanager
    async def leased_session(self, channel_id: str):
        """Look up a runtime and hold an eviction lease for the whole block.

        This is the safe way to obtain a runtime you are going to *use*. The
        lease outlives the manager lock and is released only when the block
        exits, by which time either the turn registry or ``runtime.lock`` owns
        the runtime — so there is no interval where a live runtime looks idle.

        Yields None when the channel has no live session, so callers keep their
        existing not-found handling.
        """
        async with self._lock:
            runtime = self._sessions.get(channel_id)
            if runtime is not None:
                self._touch(channel_id)
                self._acquire_lease(channel_id)
        try:
            yield runtime
        finally:
            if runtime is not None:
                runtime.last_active = time.monotonic()
                self._release_lease(channel_id)

    @asynccontextmanager
    async def initialization_lease(self, channel_id: str):
        """Hold a lease across session creation, before the session exists.

        A runtime under construction is the *only* apparently safe eviction
        candidate exactly when every older session is pinned — it has no registry
        pointer, its lock is free, and its workflow has no context object yet. So
        the manager can evict the runtime it is in the middle of creating, inside
        create_session's own overflow sweep. Leasing the channel_id before
        creation starts closes that window.
        """
        async with self._lock:
            self._acquire_lease(channel_id)
        try:
            yield
        finally:
            self._release_lease(channel_id)

    def _acquire_lease(self, channel_id: str) -> None:
        """Caller holds ``self._lock``."""
        self._leases[channel_id] = self._leases.get(channel_id, 0) + 1

    def _release_lease(self, channel_id: str) -> None:
        remaining = self._leases.get(channel_id, 0) - 1
        if remaining > 0:
            self._leases[channel_id] = remaining
        else:
            self._leases.pop(channel_id, None)

    def close_admission(self) -> None:
        """Stop creating sessions. Closing the registry alone is not enough.

        A channel created concurrently with shutdown is in neither the registry
        nor ``_sessions`` when the drain scans, so it would be neither drained
        nor closed.
        """
        self._admission_closed = True

    @property
    def admission_closed(self) -> bool:
        return self._admission_closed

    def busy_channel_ids(self) -> list[str]:
        """Channels that are leased or have work in flight.

        The shutdown drain needs this rather than ``runtime.lock.locked()``: a
        QUEUED execution whose task has not yet taken the lock reports not-busy,
        and so does a request that has been handed a runtime but not yet
        registered its turn.

        Leases are scanned separately from ``_sessions`` because a channel being
        created holds one before it has a session to be found under.
        """
        candidates = set(self._sessions) | set(self._leases)
        return sorted(
            channel_id
            for channel_id in candidates
            if self._leases.get(channel_id) or self._has_work_in_flight(channel_id)
        )

    def _has_work_in_flight(self, channel_id: str) -> bool:
        """Union predicate: a live registry execution OR a held runtime lock.

        The registry pointer alone is not enough, because /invoke_agent_stream
        runs an entire turn without ever creating a TurnExecution and guards
        itself with the lock instead. Invariant 2 still forbids the lock as the
        409 idempotency source; eviction safety is a different question, and the
        lock answers it correctly where the pointer does not.
        """
        if self.is_channel_busy and self.is_channel_busy(channel_id):
            return True
        runtime = self._sessions.get(channel_id)
        return runtime is not None and runtime.lock.locked()

    def get_creation_lock(self, channel_id: str) -> asyncio.Lock:
        """Return the per-channel creation lock, creating it on first use.

        Callers must hold on to the returned lock for as long as they need it;
        the mapping only keeps a weak reference.
        """
        lock = self._creation_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._creation_locks[channel_id] = lock
        return lock

    @property
    def max_live_sessions(self) -> int:
        return self._max_live_sessions

    @property
    def max_live_sessions_source(self) -> str:
        return self._max_live_sessions_source

    def configure_max_live_sessions(self) -> tuple[int, str]:
        """Apply the resolvers (soft target, hard ceiling, both TTLs). Must run after fastworkflow.init() loads env files.

        Returns the soft target and its source, as before; ``limits_summary`` names the rest.
        """
        value, source = resolve_max_live_sessions()
        self._max_live_sessions = value
        self._max_live_sessions_source = source
        self._max_live_hard, self._max_live_hard_source = resolve_max_live_sessions_hard(value)
        self._pinned_idle_ttl, pinned_source = resolve_pinned_idle_ttl()
        self._protected_idle_ttl, protected_source = resolve_protected_idle_ttl()
        self._ttl_sources = (pinned_source, protected_source)
        return value, source

    @property
    def hard_ceiling(self) -> int:
        if self._max_live_hard is not None:
            return self._max_live_hard
        return self._max_live_sessions * DEFAULT_HARD_CEILING_FACTOR

    def limits_summary(self) -> str:
        """One line naming every session limit in force and where each came from (the startup log)."""
        return (
            f"max_live_sessions={self._max_live_sessions} (source={self._max_live_sessions_source}), "
            f"max_live_sessions_hard={self.hard_ceiling} (source={self._max_live_hard_source}), "
            f"pinned_session_idle_ttl_seconds={self._pinned_idle_ttl} (source={self._ttl_sources[0]}), "
            f"protected_session_idle_ttl_seconds={self._protected_idle_ttl} (source={self._ttl_sources[1]})"
        )

    @property
    def notice_store(self) -> SessionNoticeStore:
        if self._notice_store is None:
            from fastworkflow.session import state_paths
            self._notice_store = SessionNoticeStore(
                os.path.join(state_paths.workflow_state_dir(self.workflow_path), "session_notices")
            )
        return self._notice_store

    def take_session_notice(self, channel_id: str) -> Optional[dict[str, Any]]:
        """The dropped-session notice waiting on the channel's live session, once; None if there is none."""
        runtime = self._sessions.get(channel_id)
        if runtime is None or runtime.session_notice is None:
            return None
        notice, runtime.session_notice = runtime.session_notice, None
        return notice

    def note_activity(self, channel_id: str) -> None:
        """Mark the channel as just used (idle age restarts). A turn that ran for an hour is not 'idle'."""
        runtime = self._sessions.get(channel_id)
        if runtime is not None:
            runtime.last_active = time.monotonic()

    def reap_checkpoints(self, policy: Optional[RetentionPolicy] = None) -> Any:
        """Reclaim abandoned checkpoints, protecting every channel we hold live.

        The store has no timer of its own — a background sweeper inside it would
        be a second writer per channel — so the server drives retention and names
        what it must not touch.
        """
        return self.checkpoint_store.reap(
            policy or RetentionPolicy(),
            protected_channel_ids=set(self._sessions) | set(self._leases),
        )

    def reap_pending_state(
        self, policy: Optional[PendingRetentionPolicy] = None
    ) -> PendingReapOutcome:
        """Reclaim abandoned suspended sessions, protecting every channel we hold.

        A suspended session waits on a user who may never return, and since
        v2.28.0 removed the pin it can be evicted — which means the blob is then
        the only copy of that conversation. Protecting live and leased channels
        is what keeps this a reaper of abandoned state rather than of state
        someone is still using.
        """
        return self.session_state_store.reap(
            policy or PendingRetentionPolicy(),
            protected_channel_ids=set(self._sessions) | set(self._leases),
        )

    @property
    def session_state_store(self) -> SessionStateStore:
        if self._session_state_store is None:
            self._session_state_store = get_session_state_store(
                base_folder=get_channel_session_state_dir(self.workflow_path)
            )
        return self._session_state_store

    @property
    def checkpoint_store(self) -> ChannelCheckpointStore:
        # Built lazily for the same reason as the state store: the state-root
        # read must happen after fastworkflow.init().
        if self._checkpoint_store is None:
            self._checkpoint_store = ChannelCheckpointStore(
                checkpoint.get_checkpoint_dir(self.workflow_path),
                min_readable_protocol_version=checkpoint.fleet_protocol_floor(),
            )
        return self._checkpoint_store

    def set_launch_context(self, launch_context: Optional[dict]) -> None:
        """The configured initial context, needed for three-way reconciliation."""
        self._launch_context = dict(launch_context or {})

    def checkpoint_for_shutdown(self, skip_channel_ids: list[str]) -> int:
        """Publish checkpoints for quiescent runtimes before the process exits.

        Retirement writes on eviction; without this, a clean restart loses
        everything a live channel accumulated since it was last evicted — the
        same loss the deadline rule guards against, reached by writing nothing
        rather than by writing something stale. Busy channels are skipped for
        exactly that reason: their snapshot would predate work still running.
        """
        skip = set(skip_channel_ids)
        written = 0
        for channel_id, runtime in list(self._sessions.items()):
            if channel_id in skip:
                continue
            eligibility = checkpoint.assess(runtime, self.session_state_store)
            if not eligibility.evictable:
                continue
            try:
                checkpoint.publish(
                    self.checkpoint_store, runtime, eligibility, self._launch_context
                )
                written += 1
            except Exception as exc:
                logger.warning(
                    f"Could not checkpoint channel_id {channel_id} at shutdown "
                    f"({type(exc).__name__}: {exc})"
                )
        return written

    def commit_startup_state(self, runtime: ChannelRuntime) -> None:
        """Persist the startup outcome now, not at retirement.

        Invariant 25: a fact whose loss changes restart behaviour is committed
        before it becomes observable, and its durability must not depend on
        whether the application happened to mutate its context.
        """
        eligibility = checkpoint.assess(runtime, self.session_state_store)
        if not eligibility.evictable:
            # Nothing durable to attach it to. The session is pinned anyway, so
            # it will not be evicted and the fact stays in memory where it is
            # still correct for this process.
            return
        checkpoint.publish(
            self.checkpoint_store, runtime, eligibility, self._launch_context
        )

    def _touch(self, channel_id: str) -> None:
        runtime = self._sessions.get(channel_id)
        if runtime is not None:
            self._sessions.move_to_end(channel_id)
            runtime.last_active = time.monotonic()

    async def trim_live_sessions(self) -> None:
        """Bring the cache back to target after a turn frees a channel up.

        Creation is not the only moment the cache can be over target: a channel
        that was skipped because it was busy becomes retirable the moment its
        turn ends, and nothing else would revisit it.
        """
        async with self._lock:
            await self._evict_oldest_if_needed()

    async def _evict_oldest_if_needed(self) -> None:
        while len(self._sessions) > self._max_live_sessions:
            # Never close a live turn's ctx mid-mutation (§3.6) — eviction would
            # race the executor thread — and never evict a session whose state
            # cannot be written back. OrderedDict iterates oldest-first.
            if not await self._retire_one_candidate():
                break

    async def _retire_one_candidate(self, *, log_if_none: bool = True) -> bool:
        """Retire the oldest retirable session. False when none can be.

        Retirement is publish-then-pop. Popping first and writing "best effort"
        afterwards is how a failed write becomes lost application state.
        """
        skipped_pinned = 0
        for channel_id, runtime in list(self._sessions.items()):
            if self._leases.get(channel_id) or self._has_work_in_flight(channel_id):
                continue

            eligibility = checkpoint.assess(runtime, self.session_state_store)
            if not eligibility.evictable:
                skipped_pinned += 1
                checkpoint.warn_pinned_once(
                    channel_id, runtime.workflow_path, eligibility
                )
                continue

            # Eviction is not session end: pending approvals live in the session's stored state and
            # survive it (ADR-EXP-025 answer 4, section 6.4 token hash only), so they are not cleared here.

            try:
                generation = checkpoint.publish(
                    self.checkpoint_store, runtime, eligibility, self._launch_context
                )
            except (StateEncodingError, CheckpointStoreError, OSError) as exc:
                # Leave it live and try the next candidate. The cache staying
                # over target is a visible, metered condition; a dropped runtime
                # whose state never landed is not.
                logger.warning(
                    f"Not evicting channel_id {channel_id}: checkpoint write "
                    f"failed ({type(exc).__name__}: {exc})"
                )
                continue

            self._sessions.pop(channel_id, None)
            self._membership_version += 1
            runtime.execution_context.close()
            logger.debug(
                f"Retired channel_id {channel_id} at generation {generation}"
            )
            return True

        if log_if_none:
            logger.warning(
                "Session cache over capacity but no candidate could be retired "
                f"({skipped_pinned} pinned); staying over target"
            )
        return False

    # ------------------------------------------------------------------
    # The hard ceiling: admission and expiry
    # ------------------------------------------------------------------

    def _occupancy(self) -> int:
        """Live sessions plus admitted creations whose session is not registered yet."""
        return len(self._sessions) + len(self._creating - self._sessions.keys())

    def release_admission(self, channel_id: str) -> None:
        """Forget the creation reservation (always called, success or failure)."""
        self._creating.discard(channel_id)

    async def admit_new_session(self, channel_id: str) -> None:
        """Reserve room for one new session, or raise ``SessionCapacityExceeded``.

        Called by ``ensure_user_runtime_exists`` before anything is built, so a refusal costs one lock hold and
        no model or database work. The reservation counts toward the ceiling until ``release_admission``.
        Below the hard ceiling this is a set insert. At it, in order:

        1. retire an idle session the server can checkpoint (publish, then drop; nothing is lost);
        2. expire an idle pinned session (``_expire_one_idle_pinned``), which is only allowed because the count
           is above the soft target;
        3. refuse. Pinned sessions that are busy, leased, belong to a registered experiment or are not idle
           for long enough are never touched.

        A "nothing can be released" verdict is reused for one second while no session was added or removed, so a
        burst of refused requests does not run every session's ``get_state`` hook once per request under the lock.
        """
        async with self._lock:
            self._creating.add(channel_id)
            try:
                ceiling = self.hard_ceiling
                while self._occupancy() > ceiling:
                    now = time.monotonic()
                    verdict = self._saturated
                    if verdict is not None and verdict[0] == self._membership_version and now - verdict[1] < 1.0:
                        raise SessionCapacityExceeded(len(self._sessions), ceiling)
                    if await self._retire_one_candidate(log_if_none=False):
                        continue
                    if self._expire_one_idle_pinned(now, pressure=True):
                        continue
                    self._saturated = (self._membership_version, now)
                    raise SessionCapacityExceeded(len(self._sessions), ceiling)
            except BaseException as exc:
                self._creating.discard(channel_id)
                if isinstance(exc, SessionCapacityExceeded):
                    self.refused_total += 1
                    logger.warning(
                        f"Refused a new session for channel_id {channel_id}: at the live-session ceiling "
                        f"({ceiling}); {len(self._sessions)} live, none releasable"
                    )
                raise

    async def expire_idle_under_pressure(self, max_expiries: int = 25) -> int:
        """Expire idle pinned sessions while the live count is above the soft target (the reaper's pass).

        Never expires below the soft target: a deployment that stays within it loses nothing, however long its
        sessions sit idle. Returns how many were expired. ``max_expiries`` bounds one pass.
        """
        expired = 0
        async with self._lock:
            while expired < max_expiries and len(self._sessions) > self._max_live_sessions:
                if not self._expire_one_idle_pinned(time.monotonic(), pressure=True):
                    break
                expired += 1
        return expired

    def _expire_one_idle_pinned(self, now: float, *, pressure: bool) -> bool:
        """Expire the longest-idle expirable pinned session. False when there is none. Caller holds ``_lock``.

        Expirable means all of: ``pressure`` (live count above the soft target), no lease and no work in
        flight, not a registered experiment channel (its attempt claim lives on the session), idle at least the
        TTL that applies, and not checkpointable (a checkpointable session is retired, not expired).
        The TTL is PINNED_SESSION_IDLE_TTL_SECONDS, or PROTECTED_SESSION_IDLE_TTL_SECONDS for a session with a
        question waiting for its user (that question is discarded on expiry). A TTL of 0 never expires.
        """
        if not pressure:
            return False
        best: Optional[tuple[float, str, ChannelRuntime, bool]] = None
        for channel_id, runtime in list(self._sessions.items()):
            if self._leases.get(channel_id) or self._has_work_in_flight(channel_id):
                continue
            ctx = runtime.execution_context
            if ctx.observability_experiment_claim:
                continue
            idle = now - runtime.last_active
            pending = pending_kind(ctx) is not None
            ttl = self._protected_idle_ttl if pending else self._pinned_idle_ttl
            if ttl <= 0 or idle < ttl:
                continue
            if best is not None and idle <= best[0]:
                continue
            try:
                evictable = checkpoint.assess(runtime, self.session_state_store).evictable
            except Exception:  # noqa: BLE001 - a hook that raises is a pinned session
                evictable = False
            if evictable:
                continue
            best = (idle, channel_id, runtime, pending)
        if best is None:
            return False
        idle, channel_id, runtime, pending = best
        self._expire_locked(channel_id, runtime, pending, idle)
        return True

    def _expire_locked(self, channel_id: str, runtime: "ChannelRuntime", pending: bool, idle: float) -> None:
        """Drop one pinned session, leave a notice for its next session, and count it. Caller holds ``_lock``."""
        ctx = runtime.execution_context
        if pending:
            # The question's user never came back. Discard it the way end_session(discard_pending) does, so the
            # saved blob does not outlive the context object it refers to.
            ctx.discard_live_pending()
            persist_pending_after_turn(self, runtime, None)
        self._teardown_locked(channel_id, runtime)
        self.expired_total += 1
        self.notice_store.record(
            channel_id,
            session_notices.REASON_EXPIRED_PENDING if pending else session_notices.REASON_EXPIRED,
        )
        logger.info(
            f"Expired pinned session for channel_id {channel_id} after {int(idle)} s idle "
            f"({'a pending question was discarded' if pending else 'no pending question'}); "
            f"{len(self._sessions)} live"
        )

    def _teardown_locked(self, channel_id: str, runtime: "ChannelRuntime") -> None:
        """Remove a session from the cache and close everything it holds. Caller holds ``_lock``."""
        ctx = runtime.execution_context
        self._sessions.pop(channel_id, None)
        self._membership_version += 1
        ctx.close()
        app_workflow = ctx.app_workflow
        if app_workflow is not None:
            # Evict it from the process registry now. Left to garbage collection, a stale workflow can
            # still be found by the channel's next session, which would reuse its state.
            import sys as _sys
            folder = app_workflow.folderpath
            was_on_path = folder in _sys.path
            with contextlib.suppress(Exception):
                app_workflow.close()
            if was_on_path and folder not in _sys.path:
                _sys.path.insert(0, folder)  # close() tidies it away; other channels still import from it
        if self.release_channel_turns is not None:
            self.release_channel_turns(channel_id)

    async def get_session(self, channel_id: str) -> Optional[ChannelRuntime]:
        async with self._lock:
            runtime = self._sessions.get(channel_id)
            if runtime:
                self._touch(channel_id)
            return runtime

    async def create_session(
        self,
        channel_id: str,
        execution_context: WorkflowExecutionContext,
        active_conversation_id: Optional[int] = None,
        stream_format: str = "ndjson",
        workflow_path: str = "",
        startup_ran: bool = False,
        session_incarnation: Optional[str] = None,
        startup_state: str = checkpoint.STARTUP_NOT_ATTEMPTED,
        startup_idempotency_key: Optional[str] = None,
        startup_epoch: int = 0,
    ) -> ChannelRuntime:
        async with self._lock:
            runtime = ChannelRuntime(
                channel_id=channel_id,
                active_conversation_id=active_conversation_id or 0,
                execution_context=execution_context,
                lock=asyncio.Lock(),
                stream_format=stream_format,
                workflow_path=workflow_path,
                startup_ran=startup_ran,
                session_incarnation=(
                    session_incarnation or checkpoint.new_session_incarnation()
                ),
                startup_state=startup_state,
                startup_idempotency_key=startup_idempotency_key,
                startup_epoch=startup_epoch,
            )
            self._sessions[channel_id] = runtime
            self._membership_version += 1
            self._touch(channel_id)
            await self._evict_oldest_if_needed()
            return runtime

    def _clear_pending_approvals(self, channel_id: str, runtime: "ChannelRuntime") -> bool:
        """Cancel the session's pending approvals and drop their stored state. True when any were cancelled."""
        cancelled = runtime.execution_context.cancel_pending_approvals()
        if cancelled:
            self.session_state_store.clear(channel_id)
        return bool(cancelled)

    async def remove_session(self, channel_id: str) -> None:
        """Drop a session without consulting eligibility and without writing a checkpoint.

        The low-level removal (tests and embedders). It does not refuse a busy channel; the HTTP
        operation is ``end_session`` below.
        """
        async with self._lock:
            if runtime := self._sessions.pop(channel_id, None):
                self._membership_version += 1
                # Session end: pending approvals end with the session (owner 2026-11-01).
                self._clear_pending_approvals(channel_id, runtime)
                runtime.execution_context.close()

    async def evict_live_session(self, channel_id: str) -> None:
        """Drop from process cache without clearing durable pending state.

        Unlike ``remove_session`` it does not cancel pending approvals: their stored state survives, as it
        does for an ordinary retirement (the old body delegated to ``remove_session`` and so cleared it).
        """
        async with self._lock:
            if runtime := self._sessions.pop(channel_id, None):
                self._membership_version += 1
                runtime.execution_context.close()

    def session_counts(self, *, assess_pinned: bool = False) -> dict[str, Any]:
        """Live and over-target facts for the readiness probe.

        ``pinned`` is only computed on request: it runs each session's ``get_state`` hook, which a probe
        that fires every few seconds should not do by default. A channel with work in flight is not
        assessed (its state is being mutated); it is counted under ``busy_unassessed``.
        """
        live = len(self._sessions)
        counts: dict[str, Any] = {
            "live": live,
            "max_live": self._max_live_sessions,
            "over_target": max(0, live - self._max_live_sessions),
            "hard_ceiling": self.hard_ceiling,
            "at_ceiling": self._occupancy() >= self.hard_ceiling,
            "refused_total": self.refused_total,
            "expired_total": self.expired_total,
            "pinned_idle_ttl_seconds": self._pinned_idle_ttl,
            "protected_idle_ttl_seconds": self._protected_idle_ttl,
        }
        if assess_pinned:
            pinned = busy = pending_protected = experiment_exempt = 0
            for channel_id, runtime in list(self._sessions.items()):
                if self._leases.get(channel_id) or self._has_work_in_flight(channel_id):
                    busy += 1
                    continue
                try:
                    is_pinned = not checkpoint.assess(runtime, self.session_state_store).evictable
                except Exception:  # noqa: BLE001 - a hook that raises is a pinned session, not a failed probe
                    is_pinned = True
                if not is_pinned:
                    continue
                pinned += 1
                if runtime.execution_context.observability_experiment_claim:
                    experiment_exempt += 1
                elif pending_kind(runtime.execution_context) is not None:
                    pending_protected += 1
            counts["pinned"] = pinned
            counts["busy_unassessed"] = busy
            counts["pending_protected"] = pending_protected
            counts["experiment_exempt"] = experiment_exempt
            counts["trace_events_dropped"] = sum(
                getattr(runtime.execution_context.command_trace_queue, "dropped", 0)
                for runtime in list(self._sessions.values())
            )
        return counts

    async def end_session(self, channel_id: str, *, discard_pending: bool = False) -> "EndSessionOutcome":
        """End a channel's live session: the explicit counterpart of creation (``POST /end_session``).

        Refuses (``SessionEndRefused``) while a turn runs or another request holds the channel, and while a
        question is pending unless ``discard_pending``. Otherwise:

        * checkpointable session: the checkpoint is published first, exactly as an eviction does, so a later
          call on the channel restores it; a failed write refuses the end and leaves the session live
          (state is never dropped by a write that did not land);
        * pinned session (state that cannot be checkpointed): the in-memory workflow state is dropped; a later
          call on the channel starts from launch configuration. Conversation history is not lost either way:
          every finished turn was written to the evidence store when it finished.

        No stored turn, conversation record or observability row is touched. Ending a channel with no live
        session is not an error (``ended`` False). Everything runs under the manager lock, so the checks and
        the removal cannot be separated by a request that arrives between them.
        """
        async with self._lock:
            runtime = self._sessions.get(channel_id)
            if runtime is None:
                discarded = False
                if discard_pending and self.session_state_store.exists(channel_id):
                    self.session_state_store.clear(channel_id)
                    discarded = True
                return EndSessionOutcome(channel_id, ended=False, pinned=False, pending_discarded=discarded)

            if self._has_work_in_flight(channel_id):
                raise SessionEndRefused("busy", f"A turn is already in progress for user: {channel_id}")
            if self._leases.get(channel_id):
                raise SessionEndRefused(
                    "busy", f"Another request is using the channel right now: {channel_id}; try again")

            ctx = runtime.execution_context
            kind = pending_kind(ctx)
            if kind is not None and not discard_pending:
                raise SessionEndRefused(
                    "pending",
                    f"The channel has a pending {kind.replace('_', ' ')}; ending the session would lose it. "
                    "Answer it, or send discard_pending=true to end the session and discard it.",
                    pending_kind=kind,
                )
            discarded = kind is not None
            if discarded:
                ctx.discard_live_pending()
                persist_pending_after_turn(self, runtime, None)

            eligibility = checkpoint.assess(runtime, self.session_state_store)
            if eligibility.evictable:
                try:
                    checkpoint.publish(self.checkpoint_store, runtime, eligibility, self._launch_context)
                except (StateEncodingError, CheckpointStoreError, OSError) as exc:
                    raise SessionEndRefused(
                        "not_saved",
                        f"The session's state could not be saved ({type(exc).__name__}); it was left live.",
                    ) from exc

            self._teardown_locked(channel_id, runtime)
            if not eligibility.evictable:
                # Nothing was saved, so the channel's next session starts from launch configuration: say so once.
                self.notice_store.record(channel_id, session_notices.REASON_ENDED)
            logger.info(
                f"Ended session for channel_id {channel_id} "
                f"({'checkpointed' if eligibility.evictable else 'pinned, in-memory state dropped'})"
            )
            return EndSessionOutcome(
                channel_id, ended=True, pinned=not eligibility.evictable, pending_discarded=discarded
            )


class SessionEndRefused(Exception):
    """``end_session`` declined: ``reason`` is busy, pending or not_saved."""

    def __init__(self, reason: str, message: str, pending_kind: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.pending_kind = pending_kind


@dataclass
class EndSessionOutcome:
    channel_id: str
    ended: bool
    pinned: bool
    pending_discarded: bool


def pending_kind(ctx: WorkflowExecutionContext) -> Optional[str]:
    """What the session is waiting on, or None: approval, review, agent_question, parameter_request, question."""
    workflow = ctx.app_workflow
    if workflow is not None:
        if approval_binding.active_approvals(workflow):
            return "approval"
        if approval_binding.active_reviews(workflow):
            return "review"
    if ctx.awaiting_user:
        return "agent_question"
    if ctx.has_open_command():
        return "parameter_request"
    return "question" if ctx.has_pending_state() else None


# ============================================================================
# Helper Functions
# ============================================================================

def _mint_conversation_id(sink: SQLiteTraceSink, channel_id: str) -> Optional[int]:
    """Mint the next conversation id from the store's per-channel counter.

    Never raises: minting is an observability concern, and a wedged or corrupt
    DB must not fail /initialize or a turn (same posture as
    ``get_observability_sink``). A channel that cannot mint runs with
    active_conversation_id 0 and retries on its next rotate.

    The Phase-A ``legacy_floor`` argument is gone with the legacy store. It
    existed to stop a fresh observability DB re-issuing an id that already
    named one of the channel's per-channel-DB conversations, which mattered
    only while BOTH stores were being written — an aliased id meant new turns
    appended into the user's oldest legacy conversation. Nothing reads those
    files now, so the counter alone is authoritative (design §3: legacy-only
    conversations become invisible, they do not become misattributed).
    """
    try:
        return sink.store.mint_conversation_id(channel_id)
    except Exception as exc:
        logger.warning(
            f"Conversation-id mint failed for channel_id {channel_id} "
            f"({type(exc).__name__}: {exc}); the channel runs unattributed "
            "until its next rotate"
        )
        return None


def reserve_conversation_id(runtime: "ChannelRuntime") -> int:
    """Mint the next conversation id for a channel — one chokepoint [R1].

    The observability DB is the sole minting authority. Returns 0 when there is
    no sink or the mint failed, which callers treat as "no active conversation"
    exactly as they treated a never-reserved id. The freshly minted id is bound
    onto the WEC so subsequent turn records and spans carry it.
    """
    ctx = runtime.execution_context
    sink = getattr(ctx, "trace_sink", None)
    conv_id = None
    if isinstance(sink, SQLiteTraceSink):
        # fix-ayez: prune before the mint, so an id the store reuses after a deletion never restores the deleted
        # conversation's parked state (the bound conversation's own state is kept).
        prune_parked_pending(runtime)
        conv_id = _mint_conversation_id(sink, runtime.channel_id)
    conv_id = int(conv_id or 0)
    if conv_id:
        ctx.bind_observability_identity(conversation_id=conv_id)
    return conv_id


def trim_conversation_window(runtime: ChannelRuntime, logger) -> int:
    """Window the in-memory conversation history, once the record is durable.

    GATED ON THE EMIT ACK (rulings I1/I2). Turn records are best-effort by
    design, so trimming on a timer would silently punch permanent holes in
    conversation memory: the turn leaves memory, its row never lands, and no
    later read can tell that anything is missing. Deferring instead bounds the
    damage to memory growth, which the pending-retry ring bounds in turn.

    ``last_turn_record_stored`` is None when no sink is installed. That is not
    a failed write — it means nothing is recording turns at all, so there is
    nothing to wait for and the window is enforced as before.
    """
    stored = runtime.execution_context.last_turn_record_stored
    if stored is False:
        # The deferral is capped: a sink that never lands its writes must not let the history grow without
        # bound. Past twice the window the OLDEST turns go anyway (the undurable one is the newest, so it stays).
        hard_window = DEFERRED_TRIM_WINDOW_FACTOR * MAX_CONVERSATION_TURNS_IN_MEMORY
        if len(runtime.execution_context.conversation_history.messages) <= hard_window:
            logger.warning(
                f"Deferring the conversation-window trim for channel_id "
                f"{runtime.channel_id}: the last turn record is queued, not yet "
                "durable, and trimming it out of memory now could lose it"
            )
            return 0
        trimmed = runtime.execution_context.trim_conversation_history(hard_window)
        logger.warning(
            f"The conversation-window trim for channel_id {runtime.channel_id} was deferred past twice the "
            f"window; trimmed {trimmed} oldest turn(s) to {hard_window} with the last record still queued"
        )
        return trimmed
    trimmed = runtime.execution_context.trim_conversation_history(
        MAX_CONVERSATION_TURNS_IN_MEMORY
    )
    if trimmed:
        logger.debug(
            f"Trimmed {trimmed} in-memory conversation turn(s) for channel_id "
            f"{runtime.channel_id} (window={MAX_CONVERSATION_TURNS_IN_MEMORY})"
        )
    return trimmed


class FeedbackTurnNotFound(ValueError):
    """A referenced turn is not this channel's to read or write about.

    A ``ValueError`` subclass so the direct callers that already treat an
    unrecorded turn as one keep working; the HTTP layer catches this subclass
    FIRST and answers 404, because an unknown turn and another channel's turn
    have to be indistinguishable ([A39], the rule GET /turns already follows).
    """

    def __init__(self, turn_key: str) -> None:
        super().__init__(f"Turn not found: {turn_key}")
        self.turn_key = turn_key


def assert_channel_owns_turn(
    store: ObservabilityStore, channel_id: str, turn_key: str
) -> dict[str, Any]:
    """Return the turn row, but only when THIS channel recorded it.

    The evidence store is per workflow, not per channel: every session this
    server serves appends to the same database. "The store has this turn" is
    therefore not an authorization, it is the bare-handle read [A39] forbids,
    and a feedback route that settles for it lets one channel read and annotate
    another's turns (fix-bnym).

    Unknown and foreign collapse into one refusal for the same reason they do
    in ``_turn_not_found``: distinguishing them confirms that somebody else's
    turn key exists. A NULL-channel row is foreign too — nothing unattributed
    is served through a channel-scoped route.
    """
    row = store.get_turn(turn_key)
    if row is None or row.get("channel_id") != channel_id:
        raise FeedbackTurnNotFound(turn_key)
    return row


def _anchor_turn_keys(
    target: "observability_feedback.FeedbackTarget",
) -> tuple[str, ...]:
    """Every turn one side of a note claims, anchored or merely referenced.

    ``validate_target`` checks the whole reference against the evidence, so the
    whole reference has to be owned: a scope frozen into the row forever is an
    assertion about turns this caller must have been allowed to see.
    """
    keys = list(target.ref.turn_keys)
    if target.turn_key not in keys:
        keys.append(target.turn_key)
    return tuple(keys)


def _assert_spans_are_in_pass(
    target: "observability_feedback.FeedbackTarget",
    selector: PassSelector,
    spans: list[dict[str, Any]],
) -> None:
    """Refuse a component anchor whose spans belong to a different pass.

    ``feedback.validate_target`` does NOT answer this, and the two questions it
    does answer look enough like it to hide the gap: it checks that the span
    ids are recorded ON THE TURN, and it calls ``PassSelector.resolve_against``,
    which only asks whether the selector matches SOME span of the turn. On a
    turn holding a teacher pass and a student pass, both are satisfied for
    either pass — so ``pass_id="teacher"`` anchored to the student's span was
    accepted and frozen into the row as a claim about the teacher.

    Membership comes from ``comparison._pass_id_for``, the same function the
    projection attributes steps with, so a span is in the pass here exactly
    when the compare view would show it under that pass. It walks ancestry
    rather than comparing stamps, which is what a producer opening one span per
    pass relies on: everything the pass did inherits membership from the pass
    span, and a check written against stamps alone would refuse every nested
    dispatch of a real distillation trace.

    Spans the turn does not record are left to ``validate_target``'s own
    "not recorded on turn" refusal. They are a different mistake and deserve
    their own message; answering both here would restate that rule and let the
    two drift.
    """
    tree = _SpanTree(spans)
    outside = sorted(
        span_id
        for span_id in target.span_ids
        if span_id in tree.by_id
        and _pass_id_for(selector, tree, span_id) != selector.pass_id
    )
    if outside:
        raise observability_feedback.FeedbackError(
            f"span(s) {outside} recorded on turn {target.turn_key!r} are not "
            f"part of pass {selector.pass_id!r}; a comment anchored to a pass "
            "names that pass's own recorded activity"
        )


def _recorded_pass_selector(
    store: ObservabilityStore, target: "observability_feedback.FeedbackTarget"
) -> Optional[PassSelector]:
    """The selector for the pass a reference names, built from recorded spans.

    ``pass_id`` has been on the wire since the anchors landed, and nothing
    supplied the selector that makes it resolvable, so every reference naming a
    real pass was refused as if it named an imaginary one (fix-jxkk). The
    selector is DISCOVERED here rather than accepted from the request — the
    same rule ``selection_api._pass_scope`` follows — so a caller names a pass
    and never describes one, and there is no way to assert membership for
    activity the span tree does not place in the pass.

    Discovery runs over the anchored turn's spans, which is the turn
    ``feedback.validate_target`` resolves the selector against; a pass recorded
    only on some other turn is not this anchor's pass. A ``pass_id`` no span of
    that turn stamps is refused by name, listing what the turn does record,
    rather than being relabelled onto a neighbouring pass or waved through.

    Naming a recorded pass is necessary and not sufficient: the anchor's own
    spans are then checked against it by ``_assert_spans_are_in_pass``, which
    is the part ``validate_target`` leaves open.
    """
    pass_id = target.ref.pass_id
    if pass_id is None:
        return None
    spans = store.get_spans(target.turn_key)
    recorded = {
        selector.pass_id: selector
        for selector in discover_pass_selectors(spans, attribute_key=ATTR_PASS)
    }
    selector = recorded.get(pass_id)
    if selector is None:
        raise observability_feedback.FeedbackError(
            f"turn {target.turn_key!r} records no pass {pass_id!r} under "
            f"attribute {ATTR_PASS!r}; recorded passes here: "
            + (", ".join(sorted(recorded)) or "none")
        )
    _assert_spans_are_in_pass(target, selector, spans)
    return selector


def _feedback_target(
    store: ObservabilityStore, anchor: Any
) -> "observability_feedback.FeedbackTarget":
    """One posted anchor as a validated feedback target in THIS store.

    The store id is filled in from the store actually serving the channel, and
    a request that names a different one is refused rather than resolved: the
    live runtime holds one evidence database and has no authorized way to
    reach another.
    """
    identity = store.store_identity()
    claimed = getattr(anchor, "store_id", None)
    if claimed and claimed != identity:
        raise ValueError(
            f"store {claimed!r} is not this channel's evidence store; record "
            "a cross-store comparison through the observability server"
        )
    return observability_feedback.FeedbackTarget.from_mapping({
        "store_id": identity,
        "turn_key": anchor.turn_key,
        "experiment_id": anchor.experiment_id,
        "task_id": anchor.task_id,
        "attempt": anchor.attempt,
        "pass_id": anchor.pass_id,
        "target_kind": anchor.target_kind,
        "span_ids": anchor.span_ids,
        "target_label": anchor.target_label,
    })


def record_turn_feedback(
    runtime: ChannelRuntime, request: Any, logger
) -> dict[str, Any]:
    """Append one review note to this channel's evidence store.

    `save_last_turn_feedback` stood here until fix-9eg.16. It upserted the
    agent-memory feedback row for whatever turn the WEC had last completed,
    and `get_memory_window` joined it back into `dspy.History` on the next
    turn. Both the table and that injection are gone; a note now names the
    turn it is about explicitly, is appended rather than replaced, and is
    validated against recorded evidence before it is stored.
    """
    store = runtime.observability_store
    if store is None:
        raise ValueError(
            "no observability store is active for this channel; feedback is "
            "recorded against recorded evidence and there is none"
        )
    primary = _feedback_target(store, request)
    paired = _feedback_target(store, request.paired) if request.paired else None
    # Ownership BEFORE evidence. Every later refusal is specific about what the
    # store recorded — which spans a turn has, which passes it stamped — so
    # asking those questions about a foreign turn first would answer them.
    for target in (primary, paired):
        if target is not None:
            for turn_key in _anchor_turn_keys(target):
                assert_channel_owns_turn(store, runtime.channel_id, turn_key)
    stored = observability_feedback.record_feedback(
        store,
        primary=primary,
        comment=request.comment,
        category=request.category,
        subcategory=request.subcategory,
        provenance=request.provenance,
        paired=paired,
        primary_pass_selector=_recorded_pass_selector(store, primary),
        paired_pass_selector=(
            _recorded_pass_selector(store, paired) if paired is not None else None
        ),
    )
    logger.debug(
        f"Recorded {request.category}/{request.subcategory} feedback on turn "
        f"{primary.turn_key} for channel_id {runtime.channel_id}"
    )
    return stored


# ============================================================================
# Conversation topic/summary labeling (lazy fill)
# ============================================================================

# How much a conversation has to grow before its label is regenerated. A
# conversation is labeled the first time it holds a turn and then each time its
# durable turn count reaches the next power of this factor - 1, 4, 16, 64, ... -
# so a long thread costs O(log(turns)) LLM calls rather than one per turn.
#
# Why refresh at all: labeling once on the first chat turn and never again gives
# a 40-turn thread the title of its first exchange, permanently. That is the same
# error _conversation_turns_for_summary was written to avoid ("topic of the last
# N turns"), reached from the other end, and it is worse than the shutdown
# labeling it replaces, which at least saw the whole conversation. Why
# geometrically rather than on every turn, or on a doubling: a fixed geometric
# schedule is derivable from the durable turn count alone, so it needs no new
# field on the stored record and no migration, and each refresh sees
# proportionally more of the thread than the last one. Unbounded rather than a
# fixed set of milestones, because any last milestone is a turn count past which
# the title freezes again.
CONVERSATION_LABEL_GROWTH_FACTOR = 4


def _label_milestones_reached(turn_count: int) -> int:
    """How many refresh milestones a conversation of this many turns has passed.

    Counting rather than membership-testing is what makes the schedule immune to
    a save that appends several turns at once (a restored session, or shutdown
    persisting a backlog): the caller compares the count before and after, so a
    milestone that was stepped over still fires.
    """
    reached = 0
    milestone = 1
    while milestone <= turn_count:
        reached += 1
        milestone *= CONVERSATION_LABEL_GROWTH_FACTOR
    return reached


def _label_is_due(stored_topic: str, durable_turns: int, turns_appended: int) -> bool:
    """Should this conversation's topic and summary be (re)generated now?

    ``turns_appended`` is how many turns the trigger just made durable, and is 0
    for a trigger that appended nothing (activating a conversation).

    A blank topic is "no successful title yet" (fix-dzs.3) and is filled by any
    trigger, which is what makes the label appear promptly on the first chat turn
    of a conversation this process has never labeled - including after a bounce,
    where the durable turn count can sit anywhere.

    THE COST DECISION THAT GOES WITH THAT (fix-dzs.6, R6): because "" also means
    "generated and failed", a provider that keeps failing is retried on every
    chat turn rather than only at milestones. That is deliberate. The retry is
    one attempt per user message, bounded at the LLM client
    (LLM_CONVERSATION_STORE_TIMEOUT_SECONDS x attempts, fix-dzs.4) and off the
    turn's critical path, so it is self-limiting in exactly the way fix-dzs.3
    relies on for every other trigger; and the alternative - waiting for the next
    milestone - leaves the conversation with no title, which is what makes it
    unreachable through /activate_conversation's topic lookup. Distinguishing
    "never generated" from "generation failed" would need a new field on the
    stored record, which is out of scope here and noted as the limitation on
    update_conversation_topic_summary.
    """
    if durable_turns <= 0:
        # Nothing recorded to summarize. Generating here would hand the model an
        # empty list and store whatever it invented.
        return False
    if not stored_topic.strip():
        return True
    if turns_appended <= 0:
        # Already labeled, and this trigger added nothing to label. Refreshing
        # would spend a call to re-describe the same turns.
        return False
    return _label_milestones_reached(durable_turns) > _label_milestones_reached(
        durable_turns - turns_appended
    )


async def ensure_topic_and_summary(
    runtime: ChannelRuntime,
    generate: Callable[[list[dict[str, Any]]], tuple[str, str]],
    *,
    turns: Optional[list[dict[str, Any]]] = None,
    force: bool = False,
    turns_appended: int = 0,
) -> Optional[tuple[str, str]]:
    """Label the runtime's active conversation, if it is due. One chokepoint.

    Returns the generated ``(topic, summary)``, or None when nothing was
    generated - which ``force`` with a non-empty ``turns`` never is, so that
    caller can unpack the result.

    RAISES whatever ``generate`` raises. Whether a missing label is fatal is the
    caller's decision - a rotate treats it as fatal, an opportunistic fill does
    not - and neither can tell the difference if this swallows the error. Callers
    of the second kind use try_ensure_topic_and_summary below.

    ``generate`` is injected rather than imported: the topic-generation entry
    point is the seam every call site's tests substitute, and resolving it from
    the caller's module keeps that seam where the caller is.

    ``turns`` is what to summarize, defaulting to the whole durable conversation.
    The durable record, never the in-memory window, because the window would
    silently produce "topic of the last N turns" (see
    __main__._conversation_turns_for_summary, which is what the rotate path
    passes in so it keeps its in-memory fallback for a conversation that has no
    record yet).

    The generation runs in an executor. DSPy is synchronous, so calling it from
    here would stall the event loop - every other channel's request included -
    for the length of an LLM round trip. There is deliberately no
    asyncio.wait_for around that await: it would bound the wait and not the work,
    leaving a default-executor thread that CPython then joins for up to
    asyncio.constants.THREAD_JOIN_TIMEOUT (300 s) at exit. The bound lives at the
    LLM client, inside generate_topic_and_summary.

    The write goes through ObservabilityStore.record_conversation_label so this
    inherits the blank-topic policy (fix-dzs.3): a blank generated topic is not
    stored as a title, the conversation stays retryable, and the operator gets
    one WARNING naming it.
    """
    conv_id = runtime.active_conversation_id
    store = runtime.observability_store
    if store is None:
        return None

    if not force:
        if conv_id == 0:
            # No conversation to label. A reserved-but-unwritten id is the same
            # case: there is no record, so there is nothing recorded to read.
            return None
        if runtime.label_lock.locked():
            # Decline rather than queue. Whoever holds it is generating for this
            # same channel, so waiting would either duplicate the call or add an
            # LLM round trip to this trigger's latency for a label that is about
            # to be written anyway. Safe to check-then-act: this is a plain read
            # with no await before the acquire below, so nothing can take the
            # lock in between (the same reasoning as TurnRegistry.has_active).
            return None

    async with runtime.label_lock:
        if not force:
            stored_topic, usable_turns = store.conversation_label_state(
                runtime.channel_id, conv_id
            )
            if not _label_is_due(stored_topic, usable_turns, turns_appended):
                return None

        if turns is None:
            turns = (
                store.conversation_summaries(runtime.channel_id, conv_id)
                if conv_id
                else []
            )
        if not turns:
            return None

        loop = asyncio.get_running_loop()
        topic, summary = await loop.run_in_executor(None, generate, turns)

        if conv_id:
            # Written synchronously rather than through the sink's queue: this
            # already runs off the turn's critical path, and a queued label is
            # not visible to the very next _label_is_due read, which would
            # spend another LLM call on a conversation that was just titled.
            # Uniquification happens inside the write's own transaction either
            # way (ruling I9) — this is the same enforcement point.
            #
            # A blank topic passes as None so a failed generation never
            # clobbers a good title; the store reports back what it actually
            # stored, which is what gets logged.
            stored_topic = store.record_conversation_label(
                runtime.channel_id,
                conv_id,
                topic if topic and topic.strip() else None,
                summary,
            )
            if not (topic and topic.strip()):
                logger.warning(
                    f"Blank topic generated for conversation {conv_id} on channel "
                    f"{runtime.channel_id}; leaving its topic as {stored_topic!r} "
                    "so it is retried"
                )
        return topic, summary


async def try_ensure_topic_and_summary(
    runtime: ChannelRuntime,
    generate: Callable[[list[dict[str, Any]]], tuple[str, str]],
    *,
    turns_appended: int = 0,
) -> Optional[tuple[str, str]]:
    """ensure_topic_and_summary for triggers where a missing label is not an error.

    A label nobody asked for must never fail the thing that triggered it: a chat
    turn still succeeded and an activated conversation is still activated. The
    conversation keeps its blank topic, which is the sentinel that makes the next
    eligible trigger try again, so the failure is retried rather than swallowed.

    The exception is logged here rather than at each call site so that policy
    lives in one place. Broad on purpose: a store read that fails must not take
    down a turn either.
    """
    try:
        return await ensure_topic_and_summary(
            runtime, generate, turns_appended=turns_appended
        )
    except Exception as exc:
        logger.warning(
            f"Could not generate a topic/summary for conversation "
            f"{runtime.active_conversation_id} on channel_id {runtime.channel_id} "
            f"({type(exc).__name__}: {exc}); it stays unlabeled and will be "
            "retried on the next eligible trigger"
        )
        return None


