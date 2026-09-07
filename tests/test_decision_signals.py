"""Coverage for the uncertainty and consequence contracts (arch §6.6.1).

These are capture-only records, so there is no behavior to assert — what is worth
testing is the set of ways a plausible-looking record can be wrong. Most of this
file is therefore about rejection:

* an unknown effect contract quietly grading as `none`, which §6.6.1 forbids and
  which produces a clean row rather than an error;
* `True` entering a calibration curve as a confidence of 1.0, which Pydantic's
  lax bool-to-int coercion allowed until a `before` validator stopped it;
* a decision that carries neither a signal nor a reason it has none, which is
  indistinguishable from an uninstrumented one;
* entity content reaching a signal value, which would make the record unsafe to
  retain under the evidence capture profile.

Two tests assert conformance against *other files* rather than against this
module's own beliefs: the slot-binding vocabulary is checked against the literals
`parameter_extraction.py` actually emits, and the leaf-import constraint of arch
§22 is checked by reading the import statements. Both catch drift that a
self-consistent unit test cannot.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

import fastworkflow.decision_signals as decision_signals
from fastworkflow.decision_signals import (
    _CONSEQUENCE_ORDER,
    _EFFECT_BASE,
    _REVERSIBILITY_FLOOR,
    ASSESSOR_VERSION,
    SIGNAL_DOMAINS,
    SLOT_BINDING_SOURCES,
    BlastRadius,
    ConsequenceAssessment,
    ConsequenceClass,
    DecisionKind,
    DecisionUncertainty,
    Reversibility,
    SignalKind,
    UncertaintySignal,
    ambiguity_set_size,
    assess_consequence,
    classifier_confidence,
    classifier_topk_margin,
    fuzzy_distance,
    slot_binding_source,
)
from fastworkflow.runtime_manifest import EffectKind

EFFECT_KINDS = get_args(EffectKind)
REVERSIBILITIES = get_args(Reversibility)
BLAST_RADII = get_args(BlastRadius)


def _every_assessor_input():
    """The assessor's entire input space. Small enough to enumerate exhaustively."""
    for effect_kind in EFFECT_KINDS:
        for reversibility in REVERSIBILITIES:
            for blast_radius in BLAST_RADII:
                for decision_critical in (False, True):
                    yield effect_kind, reversibility, blast_radius, decision_critical


# ----------------------------------------------------------------------
# Enum exhaustiveness: a new member must not land without its metadata
# ----------------------------------------------------------------------


def test_every_signal_kind_declares_a_domain():
    """A kind with no domain has no declared unit and no declared polarity.

    `fuzzy-score` is the reason this matters: its value is a distance, so a
    consumer that assumes "higher is better" from the name inverts the curve.
    Adding a kind without answering that question must fail here.
    """
    assert set(SIGNAL_DOMAINS) == set(get_args(SignalKind))


def test_effect_and_reversibility_tables_are_exhaustive():
    assert set(_EFFECT_BASE) == set(EFFECT_KINDS)
    assert set(_REVERSIBILITY_FLOOR) == set(REVERSIBILITIES)


def test_consequence_order_covers_the_enum():
    assert set(_CONSEQUENCE_ORDER) == set(get_args(ConsequenceClass))


def test_message_intent_is_a_recorded_decision_kind():
    """Requirements §4.14: misclassifying a cancellation as an answer is a
    high-consequence decision and is recorded like any other."""
    kinds = set(get_args(DecisionKind))
    assert "message-intent" in kinds
    assert kinds == {
        "command-identity",
        "target-binding",
        "slot-binding",
        "branch-selection",
        "predicate-evaluation",
        "message-intent",
    }


def test_fuzzy_score_polarity_is_recorded_as_distance():
    """The one contract name that reads backwards, pinned so it stays documented."""
    domain = SIGNAL_DOMAINS["fuzzy-score"]
    assert domain.higher_is_more_confident is False
    assert "levenshtein" in domain.unit
    assert fuzzy_distance(0.28, signal_version="lev/1").kind == "fuzzy-score"


def test_enumerated_kinds_declare_no_polarity():
    """`llm` is not more confident than `stored_merge`; the order is undefined."""
    for kind in ("slot-binding-source", "predicate-evidence"):
        assert SIGNAL_DOMAINS[kind].higher_is_more_confident is None


# ----------------------------------------------------------------------
# Signal values: numeric or enumerated, never free text, never bool
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: UncertaintySignal(
                signal_id="s", signal_version="v", kind="classifier-confidence", value=True
            ),
            id="bool-true-as-confidence",
        ),
        pytest.param(
            lambda: UncertaintySignal(
                signal_id="s", signal_version="v", kind="classifier-confidence", value=False
            ),
            id="bool-false-as-confidence",
        ),
        pytest.param(
            lambda: ambiguity_set_size(True, signal_version="v"), id="bool-as-count"
        ),
    ],
)
def test_bool_is_never_a_signal_value(build):
    """`bool` subclasses `int`, and Pydantic's lax union coerces `True` to `1`.

    Measured: before the `before` validator existed, `value=True` was accepted
    and stored as a confidence of 1.0, so the obvious `after`-validator
    `isinstance(..., bool)` check was dead code that looked correct.
    """
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: classifier_confidence(1.5, signal_version="v"), id="above-1"),
        pytest.param(lambda: classifier_confidence(-0.1, signal_version="v"), id="below-0"),
        pytest.param(
            lambda: classifier_topk_margin(2.0, signal_version="v"), id="margin-above-1"
        ),
        pytest.param(lambda: fuzzy_distance(1.7, signal_version="v"), id="distance-above-1"),
        pytest.param(lambda: ambiguity_set_size(-1, signal_version="v"), id="negative-count"),
    ],
)
def test_numeric_signals_stay_in_their_declared_range(build):
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize(
    "value",
    ["telepathy", "", "sara_doe_496", "user@example.com"],
)
def test_enumerated_signals_reject_anything_outside_the_vocabulary(value):
    """Also the entity-content guard: a uid is not a binding source."""
    with pytest.raises(ValidationError):
        slot_binding_source(value, signal_version="v")


def test_numeric_kinds_reject_free_text():
    with pytest.raises(ValidationError):
        UncertaintySignal(
            signal_id="s",
            signal_version="v",
            kind="classifier-confidence",
            value="sara_doe_496",
        )


def test_enumerated_kinds_reject_numbers():
    with pytest.raises(ValidationError):
        UncertaintySignal(
            signal_id="s", signal_version="v", kind="slot-binding-source", value=0.5
        )


def test_valid_signals_round_trip():
    assert classifier_confidence(0.87, signal_version="tiny/3").value == pytest.approx(0.87)
    assert ambiguity_set_size(4, signal_version="tiny/3").value == 4
    assert slot_binding_source("db_lookup", signal_version="pe/1").value == "db_lookup"


def test_signals_are_frozen_and_reject_unknown_fields():
    signal = classifier_confidence(0.5, signal_version="v")
    with pytest.raises(ValidationError):
        signal.value = 0.9
    with pytest.raises(ValidationError):
        UncertaintySignal(
            signal_id="s",
            signal_version="v",
            kind="classifier-confidence",
            value=0.5,
            threshold=0.7,
        )


def test_calibration_ref_is_reported_not_assumed():
    """FW-REQ-021 clause 5: an uncalibrated signal is recordable, not proof."""
    assert classifier_confidence(0.9, signal_version="v").calibrated is False
    assert (
        classifier_confidence(0.9, signal_version="v", calibration_ref="cal/1").calibrated
        is True
    )
    # A count states no probability, so there is nothing about it to calibrate.
    assert ambiguity_set_size(3, signal_version="v").calibration_ref is None


# ----------------------------------------------------------------------
# Decision records: signals, or a stated reason there are none
# ----------------------------------------------------------------------


def test_a_decision_without_signals_must_say_why():
    """Exit criterion 1. Silence and 'not instrumented' must not look alike."""
    with pytest.raises(ValidationError):
        DecisionUncertainty(decision_kind="command-identity", candidate_count=1)


def test_a_deterministic_resolution_records_no_confidence():
    """An exact match has no uncertainty; reporting 1.0 would be a measurement."""
    decision = DecisionUncertainty(
        decision_kind="command-identity",
        candidate_count=1,
        signals_absent_reason="deterministic-resolution",
    )
    assert decision.signals == ()
    assert decision.reducible is None


def test_signals_and_an_absence_reason_are_mutually_exclusive():
    with pytest.raises(ValidationError):
        DecisionUncertainty(
            decision_kind="command-identity",
            candidate_count=2,
            signals=(classifier_confidence(0.4, signal_version="v"),),
            signals_absent_reason="not-applicable",
        )


def test_decision_calibration_requires_every_signal_to_be_backed():
    partly = DecisionUncertainty(
        decision_kind="message-intent",
        candidate_count=4,
        signals=(
            classifier_confidence(0.42, signal_version="v", calibration_ref="cal/1"),
            classifier_topk_margin(0.03, signal_version="v"),
        ),
    )
    assert partly.calibrated is False


def test_candidate_count_cannot_be_negative():
    with pytest.raises(ValidationError):
        DecisionUncertainty(
            decision_kind="command-identity",
            candidate_count=-1,
            signals_absent_reason="not-applicable",
        )


# ----------------------------------------------------------------------
# Consequence: unknown is not zero, and a read is not automatically cheap
# ----------------------------------------------------------------------


@pytest.mark.parametrize("inputs", list(_every_assessor_input()))
def test_unknown_effect_is_never_graded_below_high(inputs):
    """§6.6.1: an absent effect contract reads as write-capable and high."""
    effect_kind, reversibility, blast_radius, decision_critical = inputs
    assessment = assess_consequence(
        effect_kind=effect_kind,
        reversibility=reversibility,
        blast_radius=blast_radius,
        decision_critical=decision_critical,
    )
    if effect_kind == "unknown":
        assert assessment.consequence_class in ("high", "critical")
        assert assessment.write_capable is True


def test_the_default_assessor_never_claims_zero_consequence():
    """`none` is a claim about the world a declared contract cannot support."""
    produced = {
        assess_consequence(
            effect_kind=effect_kind,
            reversibility=reversibility,
            blast_radius=blast_radius,
            decision_critical=decision_critical,
        ).consequence_class
        for effect_kind, reversibility, blast_radius, decision_critical in _every_assessor_input()
    }
    assert "none" not in produced


def test_a_read_whose_result_authorizes_something_is_not_low():
    """§4.15's stale-attribute case: cost is carried by what acts on the result."""
    plain = assess_consequence(
        effect_kind="read_only", reversibility="reversible", blast_radius="single-entity"
    )
    critical = assess_consequence(
        effect_kind="read_only",
        reversibility="reversible",
        blast_radius="single-entity",
        decision_critical=True,
    )
    assert plain.consequence_class == "low"
    assert _CONSEQUENCE_ORDER.index(critical.consequence_class) > _CONSEQUENCE_ORDER.index(
        plain.consequence_class
    )


def test_unknown_reversibility_is_planned_for_as_irreversible():
    assert _REVERSIBILITY_FLOOR["unknown"] == _REVERSIBILITY_FLOOR["irreversible"]


def test_tenant_wide_blast_radius_is_critical():
    assert (
        assess_consequence(
            effect_kind="write", reversibility="irreversible", blast_radius="tenant-wide"
        ).consequence_class
        == "critical"
    )


def test_read_only_is_the_only_effect_that_is_not_write_capable():
    for effect_kind in EFFECT_KINDS:
        assessment = assess_consequence(effect_kind=effect_kind)
        assert assessment.write_capable is (effect_kind != "read_only")


def test_the_assessor_is_deterministic():
    """FW-REQ-021 clause 12: reproduces identically on replay, with no model call."""
    for inputs in _every_assessor_input():
        effect_kind, reversibility, blast_radius, decision_critical = inputs
        results = {
            assess_consequence(
                effect_kind=effect_kind,
                reversibility=reversibility,
                blast_radius=blast_radius,
                decision_critical=decision_critical,
            ).consequence_class
            for _ in range(3)
        }
        assert len(results) == 1


def test_the_assessor_records_its_version():
    """§6.6.1: so a reassessment is distinguishable from a behavior change."""
    assert assess_consequence(effect_kind="write").assessor_version == ASSESSOR_VERSION


def test_risk_class_is_carried_without_being_reinterpreted():
    """§4.15: `risk_class` is the task-level declaration, not the action grade."""
    assessment = assess_consequence(effect_kind="read_only", risk_class="tier-3")
    assert assessment.risk_class == "tier-3"
    assert assessment.consequence_class != "tier-3"


# ----------------------------------------------------------------------
# The same floors hold for a hand-built record, not only the assessor
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            dict(
                consequence_class="none",
                effect_kind="unknown",
                reversibility="reversible",
                blast_radius="single-entity",
            ),
            id="unknown-effect-as-none",
        ),
        pytest.param(
            dict(
                consequence_class="low",
                effect_kind="unknown",
                reversibility="reversible",
                blast_radius="single-entity",
            ),
            id="unknown-effect-as-low",
        ),
        pytest.param(
            dict(
                consequence_class="low",
                effect_kind="read_only",
                reversibility="unknown",
                blast_radius="single-entity",
            ),
            id="unknown-reversibility-as-low",
        ),
        pytest.param(
            dict(
                consequence_class="low",
                effect_kind="read_only",
                reversibility="reversible",
                blast_radius="unknown",
            ),
            id="unknown-blast-radius-as-low",
        ),
        pytest.param(
            dict(
                consequence_class="none",
                effect_kind="write",
                reversibility="reversible",
                blast_radius="single-entity",
            ),
            id="write-as-none",
        ),
    ],
)
def test_a_hand_built_record_cannot_grade_below_its_floor(kwargs):
    with pytest.raises(ValidationError):
        ConsequenceAssessment(decision_critical=False, **kwargs)


def test_a_workflow_assessor_may_still_assert_none_for_a_pure_read():
    """The member stays usable by an assessor that knows more than the manifest."""
    assessment = ConsequenceAssessment(
        consequence_class="none",
        effect_kind="read_only",
        reversibility="reversible",
        blast_radius="single-entity",
        decision_critical=False,
        assessor_version="workflow/1",
    )
    assert assessment.consequence_class == "none"


# ----------------------------------------------------------------------
# Conformance against other files
# ----------------------------------------------------------------------


def test_slot_binding_vocabulary_matches_what_the_runtime_emits():
    """The enum is checked against `parameter_extraction.py`, not against itself.

    A new `extraction_method` added there without a member here would be dropped
    at capture time, which is invisible in a self-consistent unit test.
    """
    source = (
        Path(inspect.getfile(decision_signals)).parent
        / "_workflows"
        / "command_metadata_extraction"
        / "parameter_extraction.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    emitted = {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "extraction_method"
    }
    assert emitted, "found no extraction_method assignments to compare against"
    assert emitted <= SLOT_BINDING_SOURCES, (
        f"parameter_extraction.py emits {sorted(emitted - SLOT_BINDING_SOURCES)}, "
        "which SLOT_BINDING_SOURCES does not allow"
    )


def test_module_stays_a_leaf():
    """Arch §22: standard library, Pydantic, and `runtime_manifest` only."""
    tree = ast.parse(Path(inspect.getfile(decision_signals)).read_text(encoding="utf-8"))
    fastworkflow_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("fastworkflow")
    }
    assert fastworkflow_imports == {"fastworkflow.runtime_manifest"}


def test_module_defines_no_decision_function():
    """EXP-003 exit criterion 2, at module scope.

    No threshold lives here and nothing returns a permission, so there is nothing
    for control flow to import and branch on. A function answering "should we
    proceed" would be the §17.3 stop condition arriving disguised as a helper.
    """
    tree = ast.parse(Path(inspect.getfile(decision_signals)).read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden = {
        name
        for name in names
        if any(
            token in name.lower()
            for token in ("should", "allow", "permit", "gate", "proceed", "threshold")
        )
    }
    assert not forbidden, f"decision-shaped helpers must not live here: {sorted(forbidden)}"
