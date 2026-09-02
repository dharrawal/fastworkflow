"""The budget a single logical turn is allowed to spend (arch §6.4, FW-REQ-001).

A *logical turn* is one user message and everything the runtime does to answer
it, including any clarification round-trip. It is not one ReAct iteration and
not one process lifetime, and the difference is the defect this module closes:
``fastWorkflowReAct.iteration_counter`` was created once per agent instance and
never reset, so the second turn of a session inherited the first turn's spend
(GAP-01), while every ``ask_user`` round-trip reset it to ``-1`` and handed the
turn a fresh budget for free.

The rules this module exists to make structural (arch §6.4):

* WEC creates the budget at fresh logical-turn start and passes *the same
  object* to the planner and to ReAct;
* ``forward()`` requires that budget and never creates or resets one;
* suspension serializes the same budget and resume restores it unchanged;
* receiving a clarification answer does not replenish it;
* invalid model or tool selections consume an iteration;
* exhaustion produces a typed non-success result rather than a silent stop.

Nothing here reads configuration. Resolving the effective limit from the
deployment, manifest, contract and host is ``runtime_config.RuntimeConfig``'s
job; this model is handed a number.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# The resource a consumption attempt asked for, and the resource a budget ran
# out of. Kept as a closed set because it is written into failure records.
BudgetResource = Literal["iteration", "model_call", "command_call", "cost", "deadline"]


class BudgetExhausted(Exception):
    """A budget could not fund what was asked of it.

    Carries the resource so a caller can report *which* limit stopped the turn
    rather than the generic "max iterations" the runtime used to log for every
    kind of stop.
    """

    def __init__(self, resource: BudgetResource, limit: Any, consumed: Any):
        self.resource = resource
        self.limit = limit
        self.consumed = consumed
        super().__init__(
            f"logical-turn budget exhausted: {resource} limit {limit!r} "
            f"with {consumed!r} consumed"
        )


class LogicalTurnBudget(BaseModel):
    """What one logical turn may spend (arch §6.4).

    Limits other than ``iteration_limit`` default to ``None``, which means "not
    bounded here" and never "zero": an absent cost limit leaves cost recorded
    and unenforced (arch §6.0), it does not forbid spending. The counters are
    always kept, so a limit that arrives later has a history to enforce against.
    """

    model_config = ConfigDict(extra="forbid")

    iteration_limit: int = Field(gt=0)
    iterations_consumed: int = Field(default=0, ge=0)
    model_call_limit: Optional[int] = Field(default=None, gt=0)
    model_calls_consumed: int = Field(default=0, ge=0)
    command_call_limit: Optional[int] = Field(default=None, gt=0)
    command_calls_consumed: int = Field(default=0, ge=0)
    deadline_at: Optional[datetime] = None
    cost_limit: Optional[Decimal] = None
    cost_consumed: Optional[Decimal] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def is_legacy(self) -> bool:
        """False. ``LegacyTurnBudget`` is the one that answers True.

        Callers ask this rather than ``isinstance`` so that a restored schema-3
        turn can be reported as carrying legacy semantics without every call
        site importing the legacy class.
        """
        return False

    # ------------------------------------------------------------------
    # Iterations — the resource FW-REQ-001 is about
    # ------------------------------------------------------------------

    @property
    def iterations_remaining(self) -> int:
        return max(0, self.iteration_limit - self.iterations_consumed)

    @property
    def exhausted(self) -> bool:
        """True when this turn has spent its own iteration budget.

        FW-REQ-001 clause 4: exhaustion is reported only when *the current
        turn* consumed its own budget. With a per-turn object that is what this
        property means by construction — there is no prior turn's spend in it.
        """
        return self.iterations_remaining == 0

    def consume_iteration(self, count: int = 1) -> int:
        """Record ``count`` iterations and return what remains.

        Does not raise on exhaustion: the ReAct loop finishes the step it is in
        and reports exhaustion as a turn outcome, which is the behavior the
        ``max_iters_exhausted`` failure reason has always described. Use
        ``require_iteration`` for a hard pre-check.
        """
        self.iterations_consumed += count
        return self.iterations_remaining

    def require_iteration(self) -> None:
        """Raise ``BudgetExhausted`` unless another iteration can be funded."""
        if self.exhausted:
            raise BudgetExhausted(
                "iteration", self.iteration_limit, self.iterations_consumed
            )

    # ------------------------------------------------------------------
    # The other resources: recorded always, enforced when a limit is declared
    # ------------------------------------------------------------------

    def consume_model_call(self, count: int = 1) -> None:
        self.model_calls_consumed += count
        if (
            self.model_call_limit is not None
            and self.model_calls_consumed > self.model_call_limit
        ):
            raise BudgetExhausted(
                "model_call", self.model_call_limit, self.model_calls_consumed
            )

    def consume_command_call(self, count: int = 1) -> None:
        self.command_calls_consumed += count
        if (
            self.command_call_limit is not None
            and self.command_calls_consumed > self.command_call_limit
        ):
            raise BudgetExhausted(
                "command_call", self.command_call_limit, self.command_calls_consumed
            )

    def consume_cost(self, amount: Decimal) -> None:
        """Add to recorded cost. Absent assurance is ``unknown``, not zero (§6.0)."""
        self.cost_consumed = (self.cost_consumed or Decimal(0)) + amount
        if self.cost_limit is not None and self.cost_consumed > self.cost_limit:
            raise BudgetExhausted("cost", self.cost_limit, self.cost_consumed)

    def check_deadline(self, now: Optional[datetime] = None) -> None:
        """Raise when a declared wall deadline has passed.

        P0 declares no deadline (``deadline_at`` is None everywhere until
        EXP-013 wires the external-operation classes), so this is a no-op today
        and the enforcement point exists rather than being retrofitted later.
        """
        if self.deadline_at is None:
            return
        current = now or datetime.now(timezone.utc)
        if current >= self.deadline_at:
            raise BudgetExhausted("deadline", self.deadline_at, current)

    # ------------------------------------------------------------------
    # Serialization — schema-4 pending state
    # ------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """A JSON-safe dict for the pending blob.

        ``state_serialization.validate_state`` rejects ``datetime`` and
        ``Decimal`` outright, so the two are encoded as strings here rather
        than left for a ``default=str`` coercion — which is exactly the
        coercion that made downstream strictness checks vacuous elsewhere in
        this codebase.
        """
        state: dict[str, Any] = {
            "kind": self.state_kind(),
            "iteration_limit": self.iteration_limit,
            "iterations_consumed": self.iterations_consumed,
            "model_call_limit": self.model_call_limit,
            "model_calls_consumed": self.model_calls_consumed,
            "command_call_limit": self.command_call_limit,
            "command_calls_consumed": self.command_calls_consumed,
            "deadline_at": (
                self.deadline_at.isoformat() if self.deadline_at is not None else None
            ),
            "cost_limit": (
                str(self.cost_limit) if self.cost_limit is not None else None
            ),
            "cost_consumed": (
                str(self.cost_consumed) if self.cost_consumed is not None else None
            ),
        }
        return state

    @staticmethod
    def state_kind() -> str:
        return "logical"

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "LogicalTurnBudget":
        """Rebuild from ``to_state()`` output.

        Raises ``ValueError`` on anything it cannot rebuild exactly. The caller
        turns that into a fail-closed restore (arch §9.2): a budget that parsed
        into *something* would be a turn running on a limit nobody set.
        """
        kind = state.get("kind", cls.state_kind())
        target = _BUDGET_KINDS.get(kind)
        if target is None:
            raise ValueError(f"unknown turn-budget kind {kind!r}")
        if target is not cls and issubclass(target, cls):
            return target.from_state(state)
        try:
            return cls(
                iteration_limit=int(state["iteration_limit"]),
                iterations_consumed=int(state.get("iterations_consumed", 0)),
                model_call_limit=_optional_int(state.get("model_call_limit")),
                model_calls_consumed=int(state.get("model_calls_consumed", 0)),
                command_call_limit=_optional_int(state.get("command_call_limit")),
                command_calls_consumed=int(state.get("command_calls_consumed", 0)),
                deadline_at=_optional_datetime(state.get("deadline_at")),
                cost_limit=_optional_decimal(state.get("cost_limit")),
                cost_consumed=_optional_decimal(state.get("cost_consumed")),
            )
        except (KeyError, TypeError, ValueError, InvalidOperation) as e:
            raise ValueError(f"malformed turn budget in pending state: {e}") from e


class LegacyTurnBudget(LogicalTurnBudget):
    """A schema-3 ``iteration_counter``, restored as what it actually is (§9.2).

    A version-3 suspended turn cannot be reconstructed exactly: its counter may
    contain a prior turn's leakage, a clarification may have reset it to -1,
    invalid decisions left no durable marker, and its trajectory may have been
    truncated. So the migration does not claim a reconstruction. It restores
    the persisted counter into this explicit carrier, keeps *that one suspended
    turn* pinned to it until it completes or is cancelled, and starts the next
    fresh turn on a WEC-owned schema-4 budget.

    The negative counter is why ``legacy_counter`` is kept beside
    ``iterations_consumed``: -1 was a sentinel meaning "the next command came
    from the user", not a count, and clamping it to 0 for the model would erase
    the only evidence that the value is not trustworthy.
    """

    model_config = ConfigDict(extra="forbid")

    legacy_counter: int = 0

    @property
    def is_legacy(self) -> bool:
        return True

    @staticmethod
    def state_kind() -> str:
        return "legacy"

    @classmethod
    def from_counter(cls, counter: Any, iteration_limit: int) -> "LegacyTurnBudget":
        """Build from a schema-3 blob's ``iteration_counter``.

        Raises ``ValueError`` for a counter that is not an integer: a malformed
        schema-3 blob is rejected explicitly (§9.2) rather than restored with a
        guessed spend.
        """
        if isinstance(counter, bool) or not isinstance(counter, int):
            raise ValueError(
                f"schema-3 iteration_counter must be an int, got {counter!r}"
            )
        if iteration_limit <= 0:
            raise ValueError(f"iteration_limit must be positive, got {iteration_limit}")
        return cls(
            iteration_limit=iteration_limit,
            iterations_consumed=max(0, counter),
            legacy_counter=counter,
        )

    def to_state(self) -> dict[str, Any]:
        state = super().to_state()
        state["legacy_counter"] = self.legacy_counter
        return state

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "LegacyTurnBudget":
        base = LogicalTurnBudget.from_state({**state, "kind": "logical"})
        try:
            legacy_counter = int(state.get("legacy_counter", base.iterations_consumed))
        except (TypeError, ValueError) as e:
            raise ValueError(f"malformed legacy turn budget: {e}") from e
        return cls(**base.model_dump(), legacy_counter=legacy_counter)


_BUDGET_KINDS: dict[str, type[LogicalTurnBudget]] = {
    LogicalTurnBudget.state_kind(): LogicalTurnBudget,
    LegacyTurnBudget.state_kind(): LegacyTurnBudget,
}


def budget_from_state(state: Optional[dict[str, Any]]) -> Optional[LogicalTurnBudget]:
    """Rebuild whichever budget kind the pending blob carries, or None."""
    if state is None:
        return None
    if not isinstance(state, dict):
        raise ValueError(f"turn budget must be an object, got {type(state).__name__}")
    return LogicalTurnBudget.from_state(state)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_decimal(value: Any) -> Optional[Decimal]:
    return None if value is None else Decimal(str(value))


def _optional_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
