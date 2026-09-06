"""Generic contracts for versioned deterministic candidate resolution."""

from __future__ import annotations

import pytest

from fastworkflow.binding_normalizers import (
    CONTROL_ALIAS_RESOLVER_V1,
    AliasCandidate,
    resolve_binding_alias,
)


@pytest.mark.parametrize(
    ("query", "expected_handle"),
    (
        (
            "Enabled vendor identities with expired ending date and owning "
            "active accounts",
            "rule-one",
        ),
        (
            "Enabled external workers whose manager departed",
            "rule-two",
        ),
        (
            "Enabled vendors with expired ending date and no active account",
            "rule-three",
        ),
    ),
)
def test_unique_maximal_semantic_coverage_resolves_without_guessing(
    query,
    expected_handle,
):
    candidates = (
        AliasCandidate(
            handle="rule-one",
            label="Vendor with past end date and active account",
        ),
        AliasCandidate(
            handle="rule-two",
            label="External worker whose manager left",
        ),
        AliasCandidate(
            handle="rule-three",
            label="Vendor still active with past end date and no active account",
        ),
    )

    result = resolve_binding_alias(
        CONTROL_ALIAS_RESOLVER_V1,
        query,
        candidates,
    )

    assert result.status == "resolved"
    assert result.handle == expected_handle
    assert result.matched_by == "unique-maximal-semantic-coverage"


def test_explicit_alias_resolves_before_semantic_coverage():
    result = resolve_binding_alias(
        CONTROL_ALIAS_RESOLVER_V1,
        "Dormant vendor rule",
        (
            AliasCandidate(
                handle="rule-one",
                label="Vendor inactivity",
                aliases=("Dormant vendor rule",),
            ),
        ),
    )

    assert result.status == "resolved"
    assert result.handle == "rule-one"
    assert result.matched_by == "normalized-exact-label-or-alias"


def test_equal_specificity_is_typed_ambiguous_and_never_guessed():
    result = resolve_binding_alias(
        CONTROL_ALIAS_RESOLVER_V1,
        "Enabled vendor with past end date and active account",
        (
            AliasCandidate(
                handle="rule-one",
                label="Vendor with past end date and active account",
            ),
            AliasCandidate(
                handle="rule-two",
                label="Active vendor with past ending date and account",
            ),
        ),
    )

    assert result.status == "ambiguous"
    assert result.handle is None
    assert result.candidate_handles == ("rule-one", "rule-two")


def test_uncovered_alias_is_a_typed_gap_not_a_fabricated_handle():
    result = resolve_binding_alias(
        CONTROL_ALIAS_RESOLVER_V1,
        "Unrelated operator phrase",
        (
            AliasCandidate(
                handle="rule-one",
                label="Vendor with past end date and active account",
            ),
        ),
    )

    assert result.status == "unresolved"
    assert result.handle is None
    assert result.candidate_handles == ()
