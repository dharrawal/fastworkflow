"""The failure vocabulary (arch §6.0, FW-REQ-008 clause 7).

FW-REQ-008 requires every failure to be *classified* — transient, permanent,
cancelled, budget-exhausted, or outcome-unknown — and today none of them are.
A model parse error, a dead worker, a backend timeout and a cancelled turn all
arrive at the caller as the same thing: a string in a log, or an exception that
the nearest ``except Exception`` turns into an observation the model then
reasons about as if it were data.

The distinction that costs the most to lose is **outcome-unknown**. A call that
timed out after dispatch is not a failure; it is an absence of knowledge, and
the one thing nothing may do is convert it to succeeded or failed without
evidence (FW-REQ-008B). Making it a value rather than a phrase is what lets the
rest of the system refuse to.

The taxonomies layer rather than compete (arch §6.0):

* ``FailureDisposition`` classifies a turn- or task-level failure;
* ``FailureCode`` adds diagnostic detail *beneath* a disposition and never
  substitutes for one;
* ``FailureClass`` (§6.8) classifies a single side-effect attempt, and
  ``SideEffectOutcome`` states what happened to the external effect — both
  arrive with the operation journal (EXP-014), and neither is a disposition.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

# Arch §6.0. A closed set, because it is written into records a reader has to
# be able to case over exhaustively.
FailureDisposition = Literal[
    # Retrying the same thing could succeed. Whether it *may* be retried is a
    # separate question the contract owns — a transient failure after a write
    # dispatch is still not automatically retryable.
    "transient",
    # Retrying the same thing cannot succeed.
    "permanent",
    # Stopped on request. Never implies compensation (arch §6.0).
    "cancelled",
    # The turn or task spent its declared budget (FW-REQ-001, arch §6.4).
    "budget-exhausted",
    # We do not know what happened. NOT a failure — an absence of knowledge.
    # Nothing may convert this to succeeded or failed without reconciliation
    # evidence (FW-REQ-008B acceptance criteria).
    "outcome-unknown",
]

# Diagnostic detail beneath the disposition. Open by intent — arch §6.0 gives
# examples rather than a closed set, because a code names a mechanism and
# mechanisms are added. The DISPOSITION is what a reader cases over.
FailureCode = str

# The codes this slice introduces, named so they are greppable and so two call
# sites cannot spell the same mechanism differently.
CODE_ADAPTER_PARSE = "adapter-parse"
CODE_EXTRACTION_FAILED = "extraction-failed"
CODE_TOOL_FAILED = "tool-failed"
CODE_WORKER_FAILED = "worker-failed"
CODE_WORKER_STUCK = "worker-stuck"
CODE_WORKER_DEAD = "worker-dead"
CODE_BACKEND_TIMEOUT = "backend-timeout"
CODE_STATE_STORE_UNAVAILABLE = "state-store-unavailable"
CODE_INCOMPATIBLE_STATE = "incompatible-state"
CODE_BUDGET_EXHAUSTED = "budget-exhausted"
CODE_CANCELLED = "cancelled"


class TypedFailure(BaseModel):
    """One classified failure, carrying what the caller needs to act on it.

    ``detail`` is a human-readable elaboration and is never the classification:
    a reader that has to parse the message to learn the disposition is back
    where this module started.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: FailureDisposition
    code: FailureCode
    detail: str = ""
    # Set when the failure happened *after* work had already been completed —
    # the completed execution records, the trajectory step index, whatever the
    # layer has. FW-REQ-008B clause 3: a failure at the end of a turn must not
    # erase what the turn already did, because that is what invites a replay.
    completed_work: tuple[Any, ...] = ()

    @property
    def is_unknown(self) -> bool:
        return self.disposition == "outcome-unknown"

    @property
    def retryable_by_disposition(self) -> bool:
        """Whether the *disposition alone* leaves retry open.

        Deliberately not called ``retryable``. Only a contract can authorize a
        retry of anything with an external effect; this answers the narrower
        question of whether the classification rules retry out. An unknown
        outcome rules it out here and can only be reopened by reconciliation.
        """
        return self.disposition == "transient"

    def as_observation(self) -> str:
        """The text form handed to a model, when one has to see it.

        Prefixed with the disposition so a trajectory carries the
        classification rather than a bare error string the model can read as
        ordinary data — the failure mode architecture §8.4 is about when it
        says control signals "cannot become ordinary observations from which
        the model invents success".
        """
        detail = f": {self.detail}" if self.detail else ""
        return f"[{self.disposition}/{self.code}]{detail}"

    def to_state(self) -> dict[str, Any]:
        """A JSON-safe projection, for spans and durable records."""
        return {
            "disposition": self.disposition,
            "code": self.code,
            "detail": self.detail,
        }


class ControlSignal(BaseException):
    """A signal that halts the turn and must never become an observation.

    Architecture §8.4: ``reconciliation-required`` and write-safety signals are
    caught *before* ReAct's generic ``except Exception``, because a model handed
    "the write may or may not have landed" as an ordinary tool observation will
    reason about it as data and invent a success.

    Derived from ``BaseException`` for the same reason ``AskUserSuspend`` and
    ``CommandCancelledError`` are: an ``except Exception`` somewhere between the
    raise and the handler must not be able to swallow it. The signals themselves
    arrive with the operation journal (EXP-014); this is the base they will
    derive from, declared here so the ReAct loop can already refuse to convert
    one into text.
    """

    def __init__(self, failure: "TypedFailure"):
        self.failure = failure
        super().__init__(failure.as_observation())


class TurnFailedError(Exception):
    """A turn failed with a classified reason.

    Raised where a bounded failure has to propagate rather than become an
    observation — the finish phase, worker supervision, the deadline layers.
    Carries the ``TypedFailure`` so the handler does not have to re-derive the
    classification from an exception type.
    """

    def __init__(self, failure: TypedFailure):
        self.failure = failure
        super().__init__(failure.as_observation())


def classify_exception(exc: BaseException) -> TypedFailure:
    """Best classification for an exception with nothing else to go on.

    Deliberately conservative and deliberately small. Anything with a real
    classification builds its own ``TypedFailure`` at the site that knows; this
    is the fallback for the outermost handler, and it answers ``permanent``
    rather than ``transient`` because a wrong ``transient`` invites a retry
    while a wrong ``permanent`` only reports a turn as failed.

    It never answers ``outcome-unknown``: an exception object does not know
    whether an external effect landed, and only the layer holding the operation
    can say. Guessing here would be the conversion FW-REQ-008B forbids, in the
    other direction.
    """
    import asyncio
    import concurrent.futures

    if isinstance(exc, (asyncio.CancelledError, concurrent.futures.CancelledError)):
        return TypedFailure(
            disposition="cancelled", code=CODE_CANCELLED, detail=str(exc)
        )
    if isinstance(exc, TimeoutError):
        return TypedFailure(
            disposition="transient", code=CODE_BACKEND_TIMEOUT, detail=str(exc)
        )
    return TypedFailure(
        disposition="permanent",
        code=type(exc).__name__,
        detail=f"{type(exc).__name__}: {exc}",
    )


def budget_exhausted(detail: str = "", completed_work: tuple[Any, ...] = ()) -> TypedFailure:
    return TypedFailure(
        disposition="budget-exhausted",
        code=CODE_BUDGET_EXHAUSTED,
        detail=detail,
        completed_work=completed_work,
    )
