"""Bounds for every call that leaves the process (arch §13.1-13.2, FW-REQ-008).

Deadlines today are per-call-site defaults chosen by whoever wrote the call
site: `timeout=60` on one client, `timeout=30` on a workflow, a 900-second poll
loop under both. None of them knows about the others or about the turn they
belong to, so a turn can exceed any bound its caller believes it has simply by
composing bounded calls — each one inside its own limit, the sum inside
nobody's.

The fix is an **absolute** deadline carried in a ContextVar and clamped into
each adapter's native timeout. Absolute, not a duration, because a duration
restarts at every layer and that is exactly how the composition problem is
built. Native, because an outer `asyncio.wait_for()` returning is not proof
that the blocking thread stopped: it abandons the call and reports a bound that
was never enforced (arch §13.2), which is worse than no bound because it looks
like one.

The ContextVar is **explicitly copied** into executor threads. A ContextVar does
not cross a thread boundary on its own, and a deadline that silently stops
applying inside the executor — which is where every blocking backend call
runs — would be a bound that exists only where nothing needs it.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping, Optional

from fastworkflow.typed_failure import (
    CODE_BACKEND_TIMEOUT,
    TurnFailedError,
    TypedFailure,
)

# Arch §13.2. A closed set: each class has a configuration source, a native
# enforcement point and a retry owner in the §13.2.1 matrix, and a call that
# does not name one of these has no declared owner for any of the three.
OperationKind = Literal[
    "model.planner",
    "model.agent",
    "model.extraction",
    "model.clarification",
    "model.parameter_extraction",
    "model.summarization",
    "distillation.decision",
    "distillation.insight_extraction",
    "backend.read",
    "backend.write",
    "auth.refresh",
    "backend.polling",
    "policy",
    "operation_journal",
    "session_store",
    "checkpoint_store",
    "reconciliation",
    "compensation",
]

# Who may retry a call of this kind (§13.2.1, "Retry owner"). `none` is a real
# answer, not a gap: after a write is dispatched or dispatch authority is
# consumed, nothing may retry without the operation contract saying so.
RetryOwner = Literal[
    "phase_caller",
    "distillation_phase",
    "read_policy",
    "operation_contract",
    "auth_adapter",
    "journal_transaction",
    "store_idempotent",
    "none",
]

_RETRY_OWNERS: Mapping[str, RetryOwner] = {
    "model.planner": "phase_caller",
    "model.agent": "phase_caller",
    "model.extraction": "phase_caller",
    "model.clarification": "phase_caller",
    "model.parameter_extraction": "phase_caller",
    "model.summarization": "phase_caller",
    "distillation.decision": "distillation_phase",
    "distillation.insight_extraction": "distillation_phase",
    "backend.read": "read_policy",
    # Never `read_policy`, and never the phase caller: a write may only be
    # retried where the command contract guarantees deduplication under the same
    # operation ID, or reconciliation proves it safe (FW-REQ-008B clause 1).
    "backend.write": "operation_contract",
    "auth.refresh": "auth_adapter",
    # Dispatch authority is consumed at the call; retrying re-asks a question
    # that has already been answered.
    "policy": "none",
    "operation_journal": "journal_transaction",
    "session_store": "store_idempotent",
    "checkpoint_store": "store_idempotent",
    "reconciliation": "operation_contract",
    "compensation": "operation_contract",
}

# Deployment defaults, in seconds. Deliberately generous: these are a backstop
# for a call that will never return, not a performance target. A contract or a
# host lowers them; nothing raises them (the same restrictive rule as the turn
# budget, arch §6.0).
# The original invariant above refers to runtime precedence. The derived
# extraction deadline may choose a larger class default before the runtime
# clamp, which is why the wording below makes that boundary explicit.
# host lowers them; nothing raises them at runtime (the same restrictive rule as
# the turn budget, arch §6.0).
#
# The three roles that generate to `max_tokens` bound a call at 300 s rather
# than 120 s. This is a deadline on ONE model call, and 120 s was below what one
# of these calls nominally costs: 4096 output tokens at the ~35 tok/s measured
# on bedrock/us.anthropic.claude-sonnet-4-6 is ~117 s of generation before a
# non-streaming Converse response sends its first byte. A deadline shorter than
# the work it bounds does not bound a hang — it cancels healthy calls, and (in
# EXP-028 Gate 4 v4) got them silently re-dispatched by the provider library
# until a turn had spent ~484 s to produce nothing. `dspy_utils.RoleBoundLM`
# spends this budget across attempts; see the note on `_ROLE_TIMEOUTS` there.
DEFAULT_DEADLINES: Mapping[str, float] = {
    "model.planner": 300.0,
    "model.agent": 300.0,
    "model.extraction": 300.0,
    "model.clarification": 120.0,
    "model.parameter_extraction": 120.0,
    "model.summarization": 120.0,
    "distillation.decision": 180.0,
    "distillation.insight_extraction": 180.0,
    "backend.read": 60.0,
    "backend.write": 60.0,
    "auth.refresh": 30.0,
    "backend.polling": 900.0,
    "policy": 15.0,
    "operation_journal": 10.0,
    "session_store": 15.0,
    "checkpoint_store": 15.0,
    "reconciliation": 120.0,
    "compensation": 120.0,
}


# ---------------------------------------------------------------------------
# The turn deadline (ido-mn1.6.33)
#
# The watchdog in `run_fastapi_mcp.turns` has always resolved this variable to
# decide when an execution is stuck. Nothing INSIDE the turn read it, so the
# bound existed only as an observer: `WorkflowExecutionContext` opened no
# operation around the agent run, `extraction_bound()`'s `clamp_timeout()` had
# no deadline to clamp against, and `timeout_clamped` was decorative on the one
# path where the whole turn's remaining time is the binding constraint.
#
# It lives here, and not in the server module, because the deadline is a
# property of the turn rather than of the transport: WEC must be able to ask
# for it without importing the FastAPI server, and the watchdog and the
# in-turn clamp must not be able to disagree about the number.
DEFAULT_TURN_DEADLINE_SECONDS = 900.0
TURN_DEADLINE_ENV_VAR = "FW_TURN_DEADLINE_SECONDS"


def resolve_turn_deadline_seconds() -> float:
    """The configured whole-turn deadline, in seconds.

    Read per call rather than captured at import: a deployment (or a test) sets
    it without rebuilding a session, and a value captured once would describe
    the process that started rather than the turn that is running.
    """
    raw = os.environ.get(TURN_DEADLINE_ENV_VAR)
    if raw in (None, ""):
        return DEFAULT_TURN_DEADLINE_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{TURN_DEADLINE_ENV_VAR}={raw!r} is not a number"
        ) from exc
    if value <= 0:
        raise ValueError(f"{TURN_DEADLINE_ENV_VAR} must be positive")
    return value


def retry_owner(kind: str) -> RetryOwner:
    """Who owns retry for this operation kind. Unknown kinds retry nowhere."""
    return _RETRY_OWNERS.get(kind, "none")


@dataclass(frozen=True)
class ExternalOperationContext:
    """The bound and the correlation for one external call (arch §13.1).

    `deadline_monotonic` is the load-bearing field. Everything else is
    correlation and policy; the deadline is what an adapter clamps to.
    """

    kind: OperationKind
    # Absolute, on the monotonic clock: a wall clock that steps (NTP, a
    # suspend/resume) would move a deadline that has already been promised.
    deadline_monotonic: float
    turn_key: Optional[str] = None
    task_id: Optional[str] = None
    security_scope: Optional[str] = None
    operation_id: Optional[str] = None
    # Present when this operation is a retry, so a record can say which attempt
    # a timeout belonged to.
    attempt: int = 1

    @property
    def owner_of_retry(self) -> RetryOwner:
        return retry_owner(self.kind)

    def remaining(self, now: Optional[float] = None) -> float:
        """Seconds left, never negative."""
        return max(0.0, self.deadline_monotonic - (now if now is not None else time.monotonic()))

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def clamp(self, native_timeout: Optional[float]) -> float:
        """The timeout to hand an adapter: the smaller of its own and ours.

        This is the whole mechanism. A client that would have waited 60 seconds
        waits for what is left of the turn's deadline instead, and a client
        whose own timeout is already shorter keeps it — the clamp only ever
        lowers, so wiring it in cannot lengthen any call.
        """
        remaining = self.remaining()
        if native_timeout is None:
            return remaining
        return min(float(native_timeout), remaining)

    def child(self, kind: OperationKind, *, seconds: Optional[float] = None) -> "ExternalOperationContext":
        """A nested operation, never outliving its parent.

        `seconds` may shorten the child; it can never extend past the parent's
        deadline, because a sub-call that outlives the call containing it is
        how a bounded operation becomes an unbounded one.
        """
        deadline = self.deadline_monotonic
        if seconds is not None:
            deadline = min(deadline, time.monotonic() + float(seconds))
        return replace(self, kind=kind, deadline_monotonic=deadline, attempt=1)

    def failure(self, detail: str = "") -> TypedFailure:
        """The typed failure for this operation running out of time.

        A **read** that times out is `transient`: nothing happened that anyone
        has to reconcile. A **write** that times out is `outcome-unknown`, and
        the difference is the whole of FW-REQ-008B: the request may have landed,
        and calling that a failure is the conversion nothing is allowed to make
        without evidence.
        """
        unknown = self.kind in ("backend.write", "compensation", "reconciliation")
        return TypedFailure(
            disposition="outcome-unknown" if unknown else "transient",
            code=CODE_BACKEND_TIMEOUT,
            detail=detail or f"{self.kind} exceeded its deadline",
        )


_current: contextvars.ContextVar[Optional[ExternalOperationContext]] = contextvars.ContextVar(
    "fastworkflow_external_operation", default=None
)


def current_operation() -> Optional[ExternalOperationContext]:
    """The operation context in force, or None outside one."""
    return _current.get()


def remaining_seconds(default: Optional[float] = None) -> Optional[float]:
    """Seconds left on the active deadline, or `default` when unbounded.

    The one function a generated client needs: it takes no arguments it would
    have to be given, so a template can call it without the workflow having to
    thread a context through every call site.
    """
    operation = _current.get()
    return default if operation is None else operation.remaining()


def clamp_timeout(native_timeout: Optional[float]) -> Optional[float]:
    """Clamp an adapter's own timeout to the active deadline.

    Returns the adapter's timeout unchanged when no operation is in force, so a
    client using this is correct both inside and outside a bounded turn — which
    is what lets the generated clients call it unconditionally.
    """
    operation = _current.get()
    return native_timeout if operation is None else operation.clamp(native_timeout)


@contextlib.contextmanager
def operation(
    kind: OperationKind,
    *,
    seconds: Optional[float] = None,
    turn_key: Optional[str] = None,
    task_id: Optional[str] = None,
    security_scope: Optional[str] = None,
    operation_id: Optional[str] = None,
    deadlines: Mapping[str, float] = DEFAULT_DEADLINES,
):
    """Run a block under a deadline for `kind`.

    Nested inside an existing operation, the deadline is the **inner** of the
    two: a nested call cannot outlive the one containing it, whatever its own
    class default says.
    """
    parent = _current.get()
    budget = seconds if seconds is not None else deadlines.get(kind, 60.0)
    if parent is not None:
        context = parent.child(kind, seconds=budget)
        context = replace(
            context,
            turn_key=turn_key or parent.turn_key,
            task_id=task_id or parent.task_id,
            security_scope=security_scope or parent.security_scope,
            operation_id=operation_id or parent.operation_id,
        )
    else:
        context = ExternalOperationContext(
            kind=kind,
            deadline_monotonic=time.monotonic() + float(budget),
            turn_key=turn_key,
            task_id=task_id,
            security_scope=security_scope,
            operation_id=operation_id,
        )
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def require_time(detail: str = "") -> None:
    """Raise before dispatch when the deadline has already passed.

    Checked *before* a call rather than only after it, because starting a
    backend call with no time left produces an outcome nobody can use and, for
    a write, an outcome nobody can classify.
    """
    context = _current.get()
    if context is not None and context.expired:
        raise TurnFailedError(context.failure(detail))


def copy_into_thread(func: Callable) -> Callable:
    """Wrap a callable so it runs with THIS thread's operation context.

    `ContextVar` values do not cross a thread boundary, and every blocking
    backend call in this runtime runs in an executor thread — so without this
    the deadline would apply exactly where nothing needs it and stop applying
    where everything does. `contextvars.copy_context()` at wrap time captures
    the caller's context; `.run()` re-enters it inside the worker.

    Deliberately not `asyncio.wait_for`: an outer timeout that returns while the
    thread keeps running is not enforcement, and recording it as one is the
    false certification arch §13.2 rules out.
    """
    context = contextvars.copy_context()

    @functools.wraps(func)
    def runner(*args: Any, **kwargs: Any) -> Any:
        return context.run(func, *args, **kwargs)

    return runner
