"""Arm-invariant runtime safety controls for EXP-028 Gate 4."""

from __future__ import annotations

import asyncio

import pytest
from dspy.utils.exceptions import LMTimeoutError

from fastworkflow.observability_store import (
    ReadOnlyObservabilityStore,
    SQLiteTraceSink,
)
from fastworkflow.run_fastapi_mcp.turns import (
    CONVERSATION_LABELING_ENV_VAR,
    DEFAULT_TURN_DEADLINE_SECONDS,
    TURN_DEADLINE_ENV_VAR,
    _label_conversation_after_turn,
    resolve_turn_deadline_seconds,
)
from fastworkflow.typed_failure import (
    CODE_BACKEND_TIMEOUT,
    CODE_PROVIDER_TIMEOUT,
    classify_exception,
)


def test_turn_watchdog_deadline_can_outlive_gate4_censor(monkeypatch):
    monkeypatch.setenv(TURN_DEADLINE_ENV_VAR, "1980")

    assert resolve_turn_deadline_seconds() == 1980.0


def test_turn_watchdog_deadline_default_is_unchanged(monkeypatch):
    monkeypatch.delenv(TURN_DEADLINE_ENV_VAR, raising=False)

    assert (
        resolve_turn_deadline_seconds()
        == DEFAULT_TURN_DEADLINE_SECONDS
        == 900.0
    )


@pytest.mark.parametrize("value", ("0", "-1", "not-a-number"))
def test_invalid_turn_watchdog_deadline_is_refused(monkeypatch, value):
    monkeypatch.setenv(TURN_DEADLINE_ENV_VAR, value)

    with pytest.raises(ValueError):
        resolve_turn_deadline_seconds()


def test_gate4_can_disable_unjoined_post_turn_model_call(monkeypatch):
    monkeypatch.setenv(CONVERSATION_LABELING_ENV_VAR, "0")

    asyncio.run(
        _label_conversation_after_turn(
            runtime=object(),
            execn=object(),
            turns_appended=1,
        )
    )


def test_writer_health_zero_baseline_exists_before_any_turn(tmp_path):
    path = tmp_path / "observability.sqlite3"
    sink = SQLiteTraceSink(str(path))
    try:
        health = ReadOnlyObservabilityStore(str(path)).writer_health()
    finally:
        sink.close()

    assert health is not None
    assert health["records_dropped"] == 0
    assert health["spans_dropped"] == 0
    assert health["write_errors"] == 0


def test_provider_timeout_has_its_own_failure_classification():
    provider = classify_exception(
        LMTimeoutError("offline timeout", model="offline-test")
    )
    backend = classify_exception(TimeoutError("backend timeout"))

    assert provider.code == CODE_PROVIDER_TIMEOUT
    assert provider.disposition == "transient"
    assert backend.code == CODE_BACKEND_TIMEOUT
    assert provider.code != backend.code
