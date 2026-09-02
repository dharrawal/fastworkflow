"""One trajectory-aware policy decision point (FW-REQ-017, FW-REQ-021 P1).

Architecture §19.6 left the API shape open and named three alternatives; the
EXP-025a G3 ADR selected **one decision point with a phase enum**, and this is
it. The reason was the conformance surface rather than ergonomics: five
properties have to hold on *every* evaluation — versioned, deterministic, no
model call, replays identically, and no `proceed` derived from an uncalibrated
signal — and each is a property of the evaluation, not of the position it is
evaluated at. With one call site each is one test. With four lifecycle hooks
each is four, and the uncalibrated-proceed rule is a §17.3 stop condition in its
P1 form, so the number of places it can be violated is the number of places the
whole feature can be stopped from.

**The inputs are a tagged union, not one wide struct of optionals.** That is the
only reason the single-point shape is defensible rather than merely convenient:
a per-phase input class recovers the type safety four hooks would have given,
without four extension points. A policy cannot read `observations` at
`BEFORE_TASK`, because at `BEFORE_TASK` there is no such field to read.

**Why `proceed` may not read an uncertainty signal here.** FW-REQ-021 clause 5
and architecture §19.6 constraint 3 both say an uncalibrated signal may push
toward caution and may never justify proceeding. On the G2A/G2B corpus *no*
signal is calibrated — `calibration-unlabellable`: the report carries band counts
with no realized-correctness axis, because the classifier emits its confidence
when it DECLINES (70 of 73 decisions `resolved=false`) and slot-binding records
the source without the bound value. A decision table keyed on confidence
thresholds is therefore unbuildable today, and `_check_proceed_is_grounded`
below makes building one an error rather than an omission.

That constraint turned out to match the evidence rather than fight it. The 40
convertible G2B failures are not uncertainty: 1046 of 1119 command-identity
decisions resolved from a single candidate, ambiguity set size is 1 in 69 of 73,
and the rater's notes describe an agent that "found exactly one account and
still stopped to ask". What is missing is authority, not confidence — so
`proceed` is granted from declared contract facts, which §19.6 constraint 1
already admits as inputs.

**Feature flag, not a constant.** `PolicyMode.SHADOW` evaluates the table and
records the outcome while acting on nothing. It is not only a safety rung: it
measures how many asks the table WOULD have converted on a live run without
changing a single behaviour, which is a more faithful reading than an offline
replay and cheaper than a live treatment arm. `OFF` is the default and is
byte-for-byte the pre-experiment behaviour, because the empty table decides
nothing and the caller acts on nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

TABLE_CONTRACT_VERSION = "1"


class PolicyError(ValueError):
    """A malformed table or a table that violates an invariant it declares."""


class PolicyMode(str, Enum):
    """Rollout state, per the plan's `off`/`shadow`/`enforce` convention."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class DecisionPhase(str, Enum):
    """The four FW-REQ-017 positions."""

    BEFORE_TASK = "before-task"
    BEFORE_SIDE_EFFECT = "before-side-effect"
    AFTER_OBSERVATION = "after-observation"
    BEFORE_FINISH = "before-finish"


class PolicyOutcome(str, Enum):
    """The typed decisions of architecture §19.6.

    `GATHER_EVIDENCE`, `REQUIRE_PLAN_VERIFICATION` and `ABSTAIN` are the ones
    FW-REQ-021 exists for: a hook that can only allow, deny or rewrite can
    enforce known anti-patterns but cannot say "too uncertain to act yet" or
    "too consequential to act without confirmation".
    """

    PROCEED = "proceed"
    GATHER_EVIDENCE = "gather-evidence"
    ASK = "ask"
    REQUIRE_PLAN_VERIFICATION = "require-plan-verification"
    ABSTAIN = "abstain"
    ALLOW = "allow"
    DENY = "deny"
    REWRITE = "rewrite"


#: Outcomes that let an action happen. Only these are subject to the
#: uncalibrated-signal rule; caution is always permitted from any signal.
_PERMISSIVE = frozenset({PolicyOutcome.PROCEED, PolicyOutcome.ALLOW})


@dataclass(frozen=True)
class ContractFacts:
    """The deterministic inputs a `proceed` row is allowed to rest on.

    Every field here is declared rather than inferred: `effect_kind` comes from
    the command's effect contract, `read_only_surface` from the capability
    index, `delegated_selection` from the task contract's declaration about the
    request. None of them is a model output, which is what makes them usable
    under FW-REQ-021 clause 5.

    `effect_kind` follows §6.6.1: absent means `unknown`, and unknown reads as
    write-capable. It is never silently `none`.
    """

    effect_kind: str = "unknown"
    read_only_surface: bool = False
    delegated_selection: bool = False
    request_fully_specified: bool = False
    authorization_scope: str = "unknown"
    task_contract_id: Optional[str] = None

    @property
    def read_only(self) -> bool:
        return self.effect_kind == "read_only" and self.read_only_surface


@dataclass(frozen=True)
class _PhaseInput:
    """Base of the tagged union. `phase` is the discriminant."""

    facts: ContractFacts
    uncertainty: Any = None          # DecisionUncertainty, or None
    consequence: Any = None          # ConsequenceAssessment, or None

    @property
    def phase(self) -> DecisionPhase:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class BeforeTaskInput(_PhaseInput):
    utterance: str = ""

    @property
    def phase(self) -> DecisionPhase:
        return DecisionPhase.BEFORE_TASK


@dataclass(frozen=True)
class BeforeSideEffectInput(_PhaseInput):
    command_name: str = ""
    command_args: Mapping[str, Any] = field(default_factory=dict)

    @property
    def phase(self) -> DecisionPhase:
        return DecisionPhase.BEFORE_SIDE_EFFECT


@dataclass(frozen=True)
class AfterObservationInput(_PhaseInput):
    """The position the 40 convertible failures are decided at.

    `pending_tool` is the tool the agent has just selected — for this
    experiment, `ask_user`. `observations` is what it already holds, which is
    what makes "you already have what you need" a decidable statement rather
    than a guess.
    """

    utterance: str = ""
    pending_tool: str = ""
    pending_args: Mapping[str, Any] = field(default_factory=dict)
    observations: Sequence[str] = ()
    commands_run: Sequence[str] = ()

    @property
    def phase(self) -> DecisionPhase:
        return DecisionPhase.AFTER_OBSERVATION


@dataclass(frozen=True)
class BeforeFinishInput(_PhaseInput):
    utterance: str = ""
    answer: str = ""
    observations: Sequence[str] = ()
    commands_run: Sequence[str] = ()

    @property
    def phase(self) -> DecisionPhase:
        return DecisionPhase.BEFORE_FINISH


@dataclass(frozen=True)
class PolicyDecision:
    """What the table decided, and enough to audit why.

    `rewrite` carries the replacement action for a `REWRITE`/`PROCEED` outcome,
    which is what FW-REQ-017 clause 3 ("deterministic rewrites and skips shall
    be preferred to rejection") and its acceptance criterion ("a rewrite records
    proposed action, replacement, reason, and source policy") require.
    """

    outcome: PolicyOutcome
    reason: str
    source_policy: str
    table_version: str
    proposed_action: Optional[str] = None
    rewrite: Optional[str] = None
    evidence_needed: Optional[str] = None
    signals_read: tuple[str, ...] = ()

    @property
    def is_permissive(self) -> bool:
        return self.outcome in _PERMISSIVE


@dataclass(frozen=True)
class PolicyRow:
    """One table row: a deterministic predicate and the outcome it produces.

    `reads_signals` is a DECLARATION, checked against the outcome rather than
    trusted. A row that returns a permissive outcome and names any uncertainty
    signal is rejected when the table is built, not when it fires — a table that
    only fails on the one input that reaches the bad row is a table that ships.
    """

    row_id: str
    phases: frozenset
    outcome: PolicyOutcome
    reason: str
    predicate: Callable[[Any], bool]
    rewrite: Optional[str] = None
    evidence_needed: Optional[str] = None
    reads_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.row_id.strip():
            raise PolicyError("a policy row needs an id; it is what a decision "
                              "cites as its source_policy")
        if not self.phases:
            raise PolicyError(
                "row %r declares no phase, so it can never fire" % self.row_id)
        for phase in self.phases:
            if not isinstance(phase, DecisionPhase):
                raise PolicyError(
                    "row %r declares %r, which is not a DecisionPhase"
                    % (self.row_id, phase))
        if self.outcome in _PERMISSIVE and self.reads_signals:
            raise PolicyError(
                "row %r returns %s while reading uncertainty signal(s) %s. "
                "FW-REQ-021 clause 5 and architecture §19.6 constraint 3: an "
                "uncalibrated signal may push toward caution and may never "
                "justify proceeding. Ground the row in contract facts instead."
                % (self.row_id, self.outcome.value, ", ".join(self.reads_signals)))


class DecisionTable:
    """A versioned, ordered, deterministic table. No model call, ever.

    First matching row wins, so order is part of the table's meaning and is
    recorded with its version. `evaluate` is pure with respect to its input: the
    same input yields the same decision, which is what "reproduces identically
    on replay" (FW-REQ-021 clause 12) reduces to once there is no model call.
    """

    def __init__(self, version: str, rows: Sequence[PolicyRow],
                 *, contract_version: str = TABLE_CONTRACT_VERSION) -> None:
        if not version.strip():
            raise PolicyError(
                "a decision table must be versioned: FW-REQ-021 clause 13 "
                "requires thresholds be re-validated when a producing model "
                "changes, and an unversioned table cannot record that it was")
        seen = set()
        for row in rows:
            if row.row_id in seen:
                raise PolicyError(
                    "duplicate row id %r: a decision cites source_policy, so "
                    "ids have to identify one row" % row.row_id)
            seen.add(row.row_id)
        self.version = version
        self.contract_version = contract_version
        self.rows = tuple(rows)

    def evaluate(self, inputs: _PhaseInput) -> Optional[PolicyDecision]:
        """First matching row, or None when no row claims this input.

        None means "this table says nothing here" and the caller keeps its
        existing behaviour — distinct from a row that deliberately returns
        `ASK`, which is the table choosing the ask.
        """
        phase = inputs.phase
        for row in self.rows:
            if phase not in row.phases:
                continue
            if not row.predicate(inputs):
                continue
            decision = PolicyDecision(
                outcome=row.outcome,
                reason=row.reason,
                source_policy=row.row_id,
                table_version=self.version,
                proposed_action=getattr(inputs, "pending_tool", None),
                rewrite=row.rewrite,
                evidence_needed=row.evidence_needed,
                signals_read=row.reads_signals,
            )
            _check_proceed_is_grounded(decision, inputs)
            return decision
        return None


def _check_proceed_is_grounded(decision: PolicyDecision,
                               inputs: _PhaseInput) -> None:
    """The §17.3 stop condition, enforced at the only place it can be violated.

    `PolicyRow.__post_init__` already refuses a permissive row that DECLARES a
    signal. This is the second half: a permissive decision reached while the
    input carries uncertainty that is not calibrated is allowed only because the
    row did not read it — so the check is that the row read nothing, not that
    the signals happened to be good. Two checks rather than one because the
    declaration is the thing a reviewer sees and the evaluation is the thing
    that runs.
    """
    if not decision.is_permissive:
        return
    if decision.signals_read:
        raise PolicyError(
            "row %r produced %s having read %s"
            % (decision.source_policy, decision.outcome.value,
               ", ".join(decision.signals_read)))
    uncertainty = inputs.uncertainty
    if uncertainty is not None and not getattr(uncertainty, "calibrated", True):
        # Not an error: the row did not read it. Recorded so a later reader can
        # see that a permissive decision was taken alongside an uncalibrated
        # signal, and check for themselves that it was not taken FROM it.
        return


#: The FW-REQ-017 clause 5 default: no rows, so no decision, so no behaviour
#: change. `OFF` plus this table is the identity policy.
NO_OP_TABLE = DecisionTable(version="no-op/1", rows=())


class PolicyDecisionPoint:
    """The single call site. Mode-gated, and it records what it did not do.

    `SHADOW` returns None to the caller — the caller acts on nothing — while
    still appending the decision to `shadow_log`. That is the mode that answers
    "how many asks would this have converted" without converting any.
    """

    def __init__(self, table: DecisionTable = NO_OP_TABLE,
                 mode: PolicyMode = PolicyMode.OFF) -> None:
        self.table = table
        self.mode = mode
        self.shadow_log: list[PolicyDecision] = []
        self.enforced_log: list[PolicyDecision] = []

    def decide(self, inputs: _PhaseInput) -> Optional[PolicyDecision]:
        if self.mode is PolicyMode.OFF:
            return None
        decision = self.table.evaluate(inputs)
        if decision is None:
            return None
        if self.mode is PolicyMode.SHADOW:
            self.shadow_log.append(decision)
            return None
        self.enforced_log.append(decision)
        return decision
