"""The capture policy where it meets the store (arch §6.6 / §12.0 delta 3).

`tests/test_capture_policy.py` covers the policy in isolation; this file covers
the wiring, against real SQLite in tmp_path per the no-mocks-for-stores rule.

Three properties are worth more than the rest:

1. **The default profile is inert.** EXP-003 is a Phase 0 slice, so a stock
   deployment must persist exactly what 3.2.0 persisted. The oversize-artifact
   case is tested explicitly because that is where an earlier version of the
   policy silently truncated: capture runs before the artifact-offload pass, so a
   value bounded here falls under the offload threshold and never reaches the
   `artifacts` table at all.

2. **Withheld values never reach the artifacts table.** Redacting `record_json`
   while leaving the raw bytes in `artifacts.inline_value` would be worse than
   not redacting, because it would look redacted.

3. **Conversation memory survives the evidence profile.** `get_memory_window`
   reads `conversation_summary` and `conversation_traces`, and
   `_USABLE_TURN_FILTER` requires the summary to be non-NULL. Withholding those
   would not reduce exposure, it would make the agent forget — a behavior change,
   which is outside this slice.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

import fastworkflow
from fastworkflow import TurnStatus
from fastworkflow import observability_store as obs
from fastworkflow.capture_policy import (
    CaptureFieldPolicy,
    CaptureProfileError,
    debug_policy,
    evidence_policy,
    is_capture_envelope,
)

UID = "sara_doe_496"
PHONE = "call me at 555-0100"
EMAIL = "s@example.com"
QUESTION = "Which order did you mean?"
TENANT_STRINGS = (UID, PHONE, EMAIL, QUESTION)


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


@pytest.fixture
def sink(db_path):
    created = obs.SQLiteTraceSink(db_path)
    yield created
    created.close()


def _rows(db_path: str, sql: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _turn_result(artifacts: dict | None = None, *, with_ask_user: bool = True):
    """A turn carrying tenant data in every place the policy has to reach."""
    outputs = [
        fastworkflow.CommandOutput(
            command_name="get_user_details",
            context="Order",
            command_parameters={"user_id": UID, "note": PHONE},
            command_response=fastworkflow.CommandResponse(
                response=f"Sara Doe, {EMAIL}, 4 orders.",
                artifacts=artifacts if artifacts is not None else {"row": {"uid": UID}},
            ),
        )
    ]
    if with_ask_user:
        # Role inversion [A10]: the "parameters" are the agent's question and the
        # "response" is the user's answer.
        outputs.append(
            fastworkflow.CommandOutput(
                command_name="ask_user",
                context="Order",
                command_parameters=QUESTION,
                command_response=fastworkflow.CommandResponse(response="the second one"),
            )
        )
    turn_output = fastworkflow.TurnOutput(
        turn_key=fastworkflow.mint_turn_key(),
        status=TurnStatus.COMPLETED,
        answer="Your order ships Tuesday.",
        command_outputs=outputs,
    )
    return fastworkflow.TurnResult(
        turn_output=turn_output,
        channel_id="c",
        conversation_id=1,
        user_message="where is my order",
        conversation_summary="user asked about order status",
        conversation_traces="get_user_details -> ok",
        entry_workflow_name="retail",
        entry_context="Order",
    )


def _first_command(turn_row: dict) -> dict:
    return json.loads(turn_row["record_json"])["turn_output"]["command_outputs"][0]


# ----------------------------------------------------------------------
# Phase 0: the default profile persists what 3.2.0 persisted
# ----------------------------------------------------------------------


def test_the_default_profile_is_debug():
    """Nothing changes for a deployment that sets no environment variable."""
    assert obs.resolve_capture_policy().profile == "debug"


def test_debug_profile_persists_every_value_verbatim():
    turn_row, _ = obs.serialize_turn_result(_turn_result(), policy=debug_policy())
    command = _first_command(turn_row)
    assert command["command_parameters"] == {"user_id": UID, "note": PHONE}
    assert command["command_response"]["response"] == f"Sara Doe, {EMAIL}, 4 orders."
    assert command["command_response"]["artifacts"] == {"row": {"uid": UID}}
    assert turn_row["user_message"] == "where is my order"
    assert turn_row["answer"] == "Your order ships Tuesday."
    assert obs.capture_policy_module.CAPTURE_ENVELOPE_MARKER not in turn_row["record_json"]


def test_debug_profile_still_offloads_an_oversize_artifact_at_full_fidelity(db_path, sink):
    """The regression that matters most.

    An earlier version bounded this to 16 KB during capture, after which it was
    under the offload threshold — so the artifacts table never saw it and the
    record held a truncated prefix. Under the profile whose job is to change
    nothing.
    """
    big = "y" * 520_000
    sink.emit_turn_record(_turn_result({"big_blob": big}))
    assert sink.flush()

    turn_row = _rows(db_path, "SELECT * FROM turns")[0]
    envelope = _first_command(turn_row)["command_response"]["artifacts"]["big_blob"]
    assert envelope["__fw_artifact_ref__"]
    assert envelope["size"] == len(json.dumps(big, ensure_ascii=False).encode("utf-8"))

    stored = _rows(db_path, "SELECT * FROM artifacts")
    assert len(stored) == 1
    assert stored[0]["size_bytes"] == envelope["size"]
    assert big.encode("utf-8") in bytes(stored[0]["inline_value"])


def test_debug_profile_leaves_small_artifacts_inline(db_path, sink):
    sink.emit_turn_record(_turn_result({"small": "ok"}))
    assert sink.flush()
    turn_row = _rows(db_path, "SELECT * FROM turns")[0]
    assert _first_command(turn_row)["command_response"]["artifacts"]["small"] == "ok"
    assert _rows(db_path, "SELECT * FROM artifacts") == []


# ----------------------------------------------------------------------
# The evidence profile withholds, and leaves a badge
# ----------------------------------------------------------------------


def test_evidence_profile_withholds_parameters_response_and_artifacts():
    turn_row, _ = obs.serialize_turn_result(_turn_result(), policy=evidence_policy())
    command = _first_command(turn_row)
    assert command["command_parameters"]["user_id"]["disposition"] == "omit"
    assert command["command_parameters"]["note"]["disposition"] == "omit"
    assert command["command_response"]["response"]["disposition"] == "omit"
    assert command["command_response"]["artifacts"]["row"]["disposition"] == "omit"


def test_evidence_profile_leaks_no_tenant_string_into_any_column():
    turn_row, artifact_rows = obs.serialize_turn_result(
        _turn_result(), policy=evidence_policy()
    )
    haystack = json.dumps(turn_row, default=str) + json.dumps(
        [{k: str(v) for k, v in row.items()} for row in artifact_rows]
    )
    for secret in TENANT_STRINGS:
        assert secret not in haystack, secret


def test_evidence_profile_withholds_the_ask_user_question():
    """The role inversion means `command_parameters` here is free text, not a
    parameter mapping, and it is the agent's question to a real user."""
    turn_row, _ = obs.serialize_turn_result(_turn_result(), policy=evidence_policy())
    ask_user = json.loads(turn_row["record_json"])["turn_output"]["command_outputs"][1]
    assert ask_user["command_name"] == "ask_user"
    assert is_capture_envelope(ask_user["command_parameters"])
    assert ask_user["command_parameters"]["classification"] == "user-text"


def test_a_withheld_value_carries_its_size_and_digest_not_silence():
    """§12.0 delta 3: a viewer shows a badge, so the UI degrades rather than
    going dark."""
    turn_row, _ = obs.serialize_turn_result(_turn_result(), policy=evidence_policy())
    withheld = _first_command(turn_row)["command_parameters"]["user_id"]
    assert withheld["classification"] is None or withheld["classification"]
    assert withheld["original_bytes"] > 0
    assert withheld["digest"].startswith("sha256:")
    assert withheld["reason"]
    assert withheld["policy_version"]


def test_policed_turn_columns_stay_bindable_text(db_path, sink, monkeypatch):
    """sqlite3 cannot bind a dict, and these are TEXT columns — an envelope has
    to be serialized on its way into one."""
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    evidence_sink = obs.SQLiteTraceSink(db_path)
    try:
        evidence_sink.emit_turn_record(_turn_result())
        assert evidence_sink.flush()
    finally:
        evidence_sink.close()

    turn_row = _rows(db_path, "SELECT * FROM turns")[0]
    for column in ("user_message", "answer"):
        assert isinstance(turn_row[column], str)
        assert is_capture_envelope(json.loads(turn_row[column]))


# ----------------------------------------------------------------------
# Withheld values must not survive anywhere else
# ----------------------------------------------------------------------


def test_a_withheld_oversize_artifact_never_reaches_the_artifacts_table(db_path, monkeypatch):
    """Redacting the record while the raw bytes sit in `inline_value` would be
    worse than not redacting, because it would look redacted."""
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    evidence_sink = obs.SQLiteTraceSink(db_path)
    try:
        evidence_sink.emit_turn_record(_turn_result({"big_blob": UID * 40_000}))
        assert evidence_sink.flush()
    finally:
        evidence_sink.close()

    assert _rows(db_path, "SELECT * FROM artifacts") == []
    turn_row = _rows(db_path, "SELECT * FROM turns")[0]
    assert UID not in turn_row["record_json"]


def test_an_artifact_ref_envelope_is_not_re_policed():
    """A ref is a pointer, not content: digesting it loses the join to the
    artifacts row while protecting nothing."""
    turn_row, artifact_rows = obs.serialize_turn_result(
        _turn_result({"big_blob": "y" * 520_000}), policy=debug_policy()
    )
    envelope = _first_command(turn_row)["command_response"]["artifacts"]["big_blob"]
    assert envelope["__fw_artifact_ref__"] == artifact_rows[0]["artifact_id"]


# ----------------------------------------------------------------------
# Conversation memory is operational state, not evidence
# ----------------------------------------------------------------------


def test_conversation_memory_columns_are_exempt_from_the_policy():
    turn_row, _ = obs.serialize_turn_result(_turn_result(), policy=evidence_policy())
    assert turn_row["conversation_summary"] == "user asked about order status"
    assert turn_row["conversation_traces"] == "get_user_details -> ok"


def test_the_exempt_set_is_exactly_what_the_memory_read_needs():
    """Guards against the exemption drifting from `get_memory_window`'s SELECT."""
    assert obs._POLICY_EXEMPT_TURN_COLUMNS == {
        "conversation_summary",
        "conversation_traces",
    }
    assert "conversation_summary" in obs.ObservabilityStore._USABLE_TURN_FILTER
    policed = {column for column, _ in obs._POLICED_TURN_COLUMNS}
    assert not policed & obs._POLICY_EXEMPT_TURN_COLUMNS


def test_memory_rebuild_still_works_under_the_evidence_profile(db_path, monkeypatch):
    """The end-to-end version: an evidence run's agent must not go amnesiac."""
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    evidence_sink = obs.SQLiteTraceSink(db_path)
    try:
        evidence_sink.emit_turn_record(_turn_result())
        assert evidence_sink.flush()
    finally:
        evidence_sink.close()

    store = obs.ObservabilityStore(db_path)
    assert store.count_usable_turns("c", 1) == 1
    window = store.get_memory_window("c", 1, max_turns=10)
    assert len(window) == 1
    assert window[0]["conversation summary"] == "user asked about order status"
    assert window[0]["conversation_traces"] == "get_user_details -> ok"


# ----------------------------------------------------------------------
# Declared classifications and the resolver hook
# ----------------------------------------------------------------------


def test_a_declared_identifier_is_digested_and_still_joins():
    """The `classify` hook is where `RuntimeMetadata.capture_classification()`
    plugs in once the manifest is threaded through."""

    def classify(command_name: str, field_name: str):
        return "identifier" if field_name == "user_id" else None

    turn_row, _ = obs.serialize_turn_result(
        _turn_result(), policy=evidence_policy(), classify=classify
    )
    captured = _first_command(turn_row)["command_parameters"]["user_id"]
    assert captured["disposition"] == "digest"
    assert captured["classification"] == "identifier"
    assert UID not in json.dumps(captured)


def test_a_broken_classifier_denies_rather_than_losing_the_turn():
    """Default-deny is the conservative direction for a resolver that raises."""

    def classify(command_name: str, field_name: str):
        raise RuntimeError("manifest unavailable")

    turn_row, _ = obs.serialize_turn_result(
        _turn_result(), policy=evidence_policy(), classify=classify
    )
    assert _first_command(turn_row)["command_parameters"]["user_id"]["disposition"] == "omit"


def test_a_declared_field_policy_can_re_admit_one_field():
    """An evidence deployment keeps a reviewed field by naming it."""
    policy = evidence_policy(
        (
            CaptureFieldPolicy(
                field_path="command.get_user_details.parameters.note",
                classification="controlled-vocabulary",
                disposition="bounded-text",
                redact_before_trace=False,
            ),
        )
    )
    turn_row, _ = obs.serialize_turn_result(_turn_result(), policy=policy)
    assert _first_command(turn_row)["command_parameters"]["note"] == PHONE


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def test_an_unknown_profile_name_is_refused(monkeypatch):
    """A typo must not silently mean `debug`, which would capture tenant data
    verbatim while the operator believed otherwise."""
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidnce")
    obs._CAPTURE_POLICY_CACHE.pop("evidnce", None)
    with pytest.raises(CaptureProfileError):
        obs.resolve_capture_policy()


def test_a_misconfigured_profile_fails_when_the_sink_is_built(db_path, monkeypatch):
    """At startup, not per turn — and not after a month of verbatim capture."""
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "nonsense")
    obs._CAPTURE_POLICY_CACHE.pop("nonsense", None)
    with pytest.raises(CaptureProfileError):
        obs.SQLiteTraceSink(db_path)


@pytest.mark.parametrize("profile", ["debug", "evidence"])
def test_a_valid_profile_resolves_and_is_recorded_on_the_sink(db_path, monkeypatch, profile):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, profile)
    created = obs.SQLiteTraceSink(db_path)
    try:
        assert created._capture_policy.profile == profile
    finally:
        created.close()


def test_the_credential_scrub_still_runs_under_both_profiles(db_path, monkeypatch):
    """The policy *extends* the sink-boundary redaction rather than replacing it:
    the policy decides what is captured, the redactor scrubs what survives."""
    monkeypatch.setenv("SOME_SERVICE_API_KEY", "sk-livekey1234567890abcdef")
    for profile in ("debug", "evidence"):
        monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, profile)
        created = obs.SQLiteTraceSink(db_path)
        try:
            turn = _turn_result({"note": "token sk-livekey1234567890abcdef here"})
            created.emit_turn_record(turn)
            assert created.flush()
        finally:
            created.close()
        turn_row = _rows(db_path, "SELECT * FROM turns")[-1]
        assert "sk-livekey1234567890abcdef" not in turn_row["record_json"], profile
