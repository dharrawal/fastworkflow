"""Evidence-grade run mode (arch §12.0 delta 6, requirements §12.4).

Against real SQLite in tmp_path. The properties worth testing are the failure
paths, because the success path is the one that gets exercised by accident:

* a dropped **turn record** must invalidate the run, and name the turns;
* a dropped **span** must NOT invalidate it, and must still name the turns — spans
  are best-effort by design, so treating them as fatal would make every busy run
  unreportable;
* an unreadable health snapshot must read as "unknown", never as "no drops";
* pruning must not run inside the block, including from a child process;
* the archive must be a consistent, verifiable, immutable file — and reading it
  must not change its digest.

Drop paths are driven through the sink's real internal methods
(`_requeue_records`, `_remember_pending`) rather than by racing the queues, which
is deterministic and still exercises the production code: those two are where all
four drop sites funnel.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import TurnStatus, tracing
from fastworkflow import observability_store as obs
from fastworkflow.evidence_run import (
    EvidenceRun,
    EvidenceRunInvalid,
    capture_observability_provenance,
    evidence_run,
)


@pytest.fixture
def workflow_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("FW_OBSERVABILITY", "1")
    fastworkflow.init({})
    path = tmp_path / "wf"
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)
    obs.close_all_sinks()


def _turn(index: int = 0):
    output = fastworkflow.CommandOutput(
        command_name="get_user",
        command_response=fastworkflow.CommandResponse(response="ok"),
    )
    turn_output = fastworkflow.TurnOutput(
        turn_key=fastworkflow.mint_turn_key(),
        status=TurnStatus.COMPLETED,
        answer=f"answer-{index}",
        command_outputs=[output],
    )
    return fastworkflow.TurnResult(
        turn_output=turn_output,
        channel_id="c",
        conversation_id=1,
        user_message=f"message-{index}",
        conversation_summary=f"summary-{index}",
        conversation_traces="t",
        entry_workflow_name="w",
        entry_context="C",
    )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# The clean run
# ----------------------------------------------------------------------


def test_a_clean_run_is_valid_and_reports_nothing(workflow_path, tmp_path):
    with evidence_run(
        workflow_path,
        run_id="run-clean",
        archive_dir=str(tmp_path / "bundle"),
        dspy_history_enabled=True,
    ) as run:
        sink = obs.get_observability_sink(workflow_path)
        for index in range(3):
            sink.emit_turn_record(_turn(index))

    assert run.valid
    assert run.problems() == ()
    assert run.delta.records_dropped == 0
    assert run.delta.incomparable is False
    assert run.completed_at is not None


def test_the_archived_db_holds_the_run(workflow_path, tmp_path):
    with evidence_run(
        workflow_path, run_id="run-arch", archive_dir=str(tmp_path / "bundle")
    ) as run:
        sink = obs.get_observability_sink(workflow_path)
        for index in range(3):
            sink.emit_turn_record(_turn(index))

    archived = obs.ReadOnlyObservabilityStore(run.archive["path"])
    assert len(archived.list_turns(limit=50)) == 3
    assert run.archive["schema_version"] == obs.SCHEMA_VERSION


def test_the_archive_is_a_single_wal_free_file(workflow_path, tmp_path):
    """`VACUUM INTO`, not a copy: copying a WAL-mode DB loses whatever is still
    in the sidecar, which is the end of the run."""
    with evidence_run(
        workflow_path, run_id="run-wal", archive_dir=str(tmp_path / "bundle")
    ) as run:
        obs.get_observability_sink(workflow_path).emit_turn_record(_turn())

    path = run.archive["path"]
    assert not Path(f"{path}-wal").exists()
    assert run.archive["size_bytes"] == os.path.getsize(path)


def test_the_archive_is_immutable_and_its_digest_verifies(workflow_path, tmp_path):
    """Reading an archive must not change it.

    `ObservabilityStore.__init__` runs `_ensure_schema` and switches the DB to
    WAL, so without the read-only mode merely inspecting an archive would rewrite
    its header and break the recorded digest — indistinguishable from tampering.
    """
    with evidence_run(
        workflow_path, run_id="run-immutable", archive_dir=str(tmp_path / "bundle")
    ) as run:
        obs.get_observability_sink(workflow_path).emit_turn_record(_turn())

    path = run.archive["path"]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o444
    assert run.archive["read_only"] is True

    obs.ReadOnlyObservabilityStore(path).list_turns(limit=5)
    assert _sha256(path) == run.archive["sha256"]

    with pytest.raises(Exception):
        obs.ObservabilityStore(path)
    assert _sha256(path) == run.archive["sha256"]


def test_archiving_never_overwrites_a_previous_run(workflow_path, tmp_path):
    """Overwriting an archive destroys the previous run's evidence."""
    with evidence_run(
        workflow_path, run_id="run-once", archive_dir=str(tmp_path / "bundle")
    ) as run:
        obs.get_observability_sink(workflow_path).emit_turn_record(_turn())

    store = obs.ObservabilityStore(obs.state_paths.observability_db(workflow_path))
    with pytest.raises(FileExistsError):
        store.archive_to(run.archive["path"])


def test_a_failed_archive_is_reported_and_does_not_raise(workflow_path, tmp_path):
    """A harness should still get its results when only archival failed."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    with evidence_run(
        workflow_path, run_id="run-badarchive", archive_dir=str(blocker / "sub")
    ) as run:
        obs.get_observability_sink(workflow_path).emit_turn_record(_turn())

    assert run.archive is None
    assert any("archival failed" in problem for problem in run.problems())
    assert run.valid is False


# ----------------------------------------------------------------------
# Dropped turn records invalidate; dropped spans do not
# ----------------------------------------------------------------------


def test_a_dropped_turn_record_invalidates_the_run_and_names_the_turn(workflow_path):
    with evidence_run(workflow_path, run_id="run-lost") as run:
        sink = obs.get_observability_sink(workflow_path)
        turn_row, _ = obs.serialize_turn_result(_turn())
        # The real retry-exhausted path: a terminal row that has burned its
        # SQLITE_BUSY retries is dropped and counted.
        sink._requeue_records([("turn", turn_row, [], obs._RECORD_BUSY_MAX_RETRIES)])

    assert run.delta.records_dropped == 1
    assert run.delta.records_dropped_turn_keys == (turn_row["turn_key"],)
    assert run.delta.lost_turn_records is True
    assert run.valid is False
    assert any("not valid evidence" in problem for problem in run.problems())
    assert any(turn_row["turn_key"] in problem for problem in run.problems())


def test_the_pending_ring_overflow_also_names_the_turn_it_gave_up_on(workflow_path):
    """The other real record-drop path: the bounded retry ring evicting oldest."""
    sink = obs.get_observability_sink(workflow_path)
    first_row, _ = obs.serialize_turn_result(_turn(0))
    with evidence_run(workflow_path, run_id="run-ring") as run:
        sink._remember_pending(first_row, [])
        for index in range(obs._PENDING_RETRY_MAX + 1):
            row, _ = obs.serialize_turn_result(_turn(index + 1))
            sink._remember_pending(row, [])

    assert run.delta.records_dropped >= 1
    assert first_row["turn_key"] in run.delta.records_dropped_turn_keys
    assert run.valid is False


def test_a_dropped_span_reports_its_turn_but_leaves_the_run_valid(workflow_path):
    """Spans are droppable by design; making them fatal would make every busy
    run unreportable, which is not what §12.4 asks for."""
    turn_key = fastworkflow.mint_turn_key()
    span = tracing.Span(
        span_id="s1", trace_id=turn_key, name=tracing.SPAN_COMMAND_EXECUTE
    )
    with evidence_run(workflow_path, run_id="run-span") as run:
        obs.get_observability_sink(workflow_path)._requeue_records([("span", span)])

    assert run.delta.spans_dropped == 1
    assert run.delta.spans_dropped_turn_keys == (turn_key,)
    assert run.delta.lost_turn_records is False
    assert run.valid is True
    assert any("incomplete detail" in problem for problem in run.problems())


def test_drops_from_before_the_run_are_not_attributed_to_it(workflow_path):
    """Counters are cumulative, so the delta has to be a subtraction and the
    turn-key lists a set difference — otherwise every run inherits the DB's
    whole history of drops."""
    sink = obs.get_observability_sink(workflow_path)
    old_span = tracing.Span(
        span_id="old", trace_id="turn-old", name=tracing.SPAN_COMMAND_EXECUTE
    )
    sink._requeue_records([("span", old_span)])

    with evidence_run(workflow_path, run_id="run-after") as run:
        pass

    assert run.delta.spans_dropped == 0
    assert run.delta.spans_dropped_turn_keys == ()
    assert run.valid is True


def test_the_affected_turn_key_list_is_bounded_and_says_when_it_elided(workflow_path):
    """A run that loses everything must not turn a counter into a memory leak,
    and must not pretend the truncated list is complete."""
    sink = obs.get_observability_sink(workflow_path)
    with evidence_run(workflow_path, run_id="run-flood") as run:
        for index in range(obs._DROP_TURN_KEY_MAX + 10):
            span = tracing.Span(
                span_id=f"s{index}",
                trace_id=f"turn-{index}",
                name=tracing.SPAN_COMMAND_EXECUTE,
            )
            sink._requeue_records([("span", span)])

    assert len(run.delta.spans_dropped_turn_keys) == obs._DROP_TURN_KEY_MAX
    assert run.delta.dropped_turn_keys_elided == 10
    assert any("list capped" in problem for problem in run.problems())


# ----------------------------------------------------------------------
# "Could not tell" is not "nothing was dropped"
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "before,after",
    [(None, {}), ({}, None), (None, None)],
)
def test_a_missing_snapshot_is_incomparable_not_clean(before, after):
    delta = obs.health_delta(before, after)
    assert delta.incomparable is True
    assert delta.evidence_valid is False
    assert any("not the same as valid" in problem for problem in delta.problems())


def test_an_unfinished_run_is_not_valid(workflow_path):
    """`delta is None` means the interval never closed; treating that as a pass
    would let a crashed run be reported as clean."""
    unfinished = EvidenceRun(
        run_id="never-finished",
        db_path="/nowhere",
        provenance=capture_observability_provenance(),
        started_at=datetime.now(timezone.utc),
    )
    assert unfinished.valid is False
    assert unfinished.problems() == (
        "the evidence run never completed, so nothing was verified",
    )


# ----------------------------------------------------------------------
# Prune suppression
# ----------------------------------------------------------------------


def test_pruning_is_suppressed_inside_the_block_and_restored_after(workflow_path):
    store = obs.ObservabilityStore(obs.state_paths.observability_db(workflow_path))
    with evidence_run(workflow_path, run_id="run-prune") as run:
        assert obs.pruning_suppressed() is True
        assert store.prune() == {"suppressed": 1}
    assert obs.pruning_suppressed() is False
    assert "suppressed" not in store.prune()

    # This run installs no sink and writes nothing, so no writer health exists
    # on either side and the run cannot certify anything. It used to report
    # `valid` here — which is the false certification fix-ajv.13 is about: the
    # zeros came from counters nobody had written, not from a measured interval.
    # Suppression, this test's actual subject, is asserted above.
    assert run.valid is False
    assert any("no writer health is available" in p for p in run.problems())


def test_suppression_is_distinguishable_from_having_nothing_to_prune(workflow_path):
    """An all-zero result would look identical to a prune that ran and found
    nothing, so a caller could not tell whether retention was withheld."""
    store = obs.ObservabilityStore(obs.state_paths.observability_db(workflow_path))
    ran = store.prune()
    with obs.suppress_pruning():
        withheld = store.prune()
    assert withheld == {"suppressed": 1}
    assert "suppressed" not in ran


def test_nested_suppression_does_not_re_enable_pruning_early():
    """Counted, not a boolean: an inner run's exit must not lift the outer one."""
    with obs.suppress_pruning():
        with obs.suppress_pruning():
            assert obs.pruning_suppressed() is True
        assert obs.pruning_suppressed() is True
    assert obs.pruning_suppressed() is False


def test_suppression_propagates_to_a_child_process(tmp_path):
    """The chatbot spawns a server, so an in-process flag cannot reach the writer
    that actually prunes — the env var is what crosses the boundary."""
    script = (
        "import fastworkflow;"
        "from fastworkflow import observability_store as obs;"
        "print(obs.pruning_suppressed())"
    )
    env = dict(os.environ, FW_OBS_SUPPRESS_PRUNE="1")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.stdout.strip().endswith("True"), result.stderr[-2000:]


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


def test_provenance_records_the_capture_regime(workflow_path):
    provenance = capture_observability_provenance(dspy_history_enabled=True)
    assert provenance.enabled is True
    assert provenance.capture_profile == "debug"
    assert provenance.span_contract_version == tracing.SPAN_CONTRACT_VERSION
    assert provenance.db_schema_version == obs.SCHEMA_VERSION
    assert provenance.capture_policy_version
    assert provenance.evidence_interpretable is True


def test_provenance_records_defaults_nobody_set(workflow_path):
    """A run whose provenance omits FW_OBS_RETENTION_DAYS because it was unset is
    a run nobody can reproduce, so the config is enumerated rather than scanned
    out of os.environ."""
    config = capture_observability_provenance().config
    for name in (
        "FW_OBS_RETENTION_DAYS",
        "FW_OBS_DB_MAX_BYTES",
        "FW_OBS_INLINE_ARTIFACT_BYTES",
        "FW_OBS_QUEUE_MAX",
        "FW_OBS_MAX_ATTR_BYTES",
        obs.CAPTURE_PROFILE_VAR,
    ):
        assert config.get(name), name


def test_provenance_reflects_the_selected_profile(workflow_path, monkeypatch):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    provenance = capture_observability_provenance()
    assert provenance.capture_profile == "evidence"
    assert provenance.default_deny is True


def test_unknown_dspy_history_is_not_interpretable(workflow_path):
    """Defaults to None, and "not False" must not read as yes: with history off
    there is no token or cost evidence, and its absence is indistinguishable from
    a run that cost nothing."""
    assert capture_observability_provenance().dspy_history_enabled is None
    assert capture_observability_provenance().evidence_interpretable is False
    assert (
        capture_observability_provenance(dspy_history_enabled=False).evidence_interpretable
        is False
    )


def test_the_run_record_is_serializable(workflow_path, tmp_path):
    with evidence_run(
        workflow_path,
        run_id="run-record",
        archive_dir=str(tmp_path / "bundle"),
        dspy_history_enabled=True,
    ) as run:
        obs.get_observability_sink(workflow_path).emit_turn_record(_turn())

    record = json.loads(json.dumps(run.as_record()))
    assert record["run_id"] == "run-record"
    assert record["valid"] is True
    assert record["observability"]["span_contract_version"] == tracing.SPAN_CONTRACT_VERSION
    assert record["writer_health_before"] is not None
    assert record["writer_health_after"] is not None
    assert record["archive"]["sha256"]


# ----------------------------------------------------------------------
# Preconditions and raising behavior
# ----------------------------------------------------------------------


def test_disabled_observability_is_reported_as_a_problem(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("FW_OBSERVABILITY", "0")
    fastworkflow.init({})
    path = tmp_path / "wf"
    path.mkdir(parents=True, exist_ok=True)

    with evidence_run(str(path), run_id="run-off") as run:
        pass

    assert run.valid is False
    assert any("observability is disabled" in problem for problem in run.problems())


def test_dspy_history_off_is_reported_as_a_problem(workflow_path):
    with evidence_run(
        workflow_path, run_id="run-nohistory", dspy_history_enabled=False
    ) as run:
        pass
    assert any("no token or cost evidence" in problem for problem in run.problems())
    assert run.valid is False


def test_requiring_the_evidence_profile_is_opt_in(workflow_path):
    with evidence_run(workflow_path, run_id="run-permissive") as permissive:
        pass
    assert all("capture profile" not in problem for problem in permissive.problems())

    with evidence_run(
        workflow_path, run_id="run-strict", require_evidence_profile=True
    ) as strict:
        pass
    assert any("not 'evidence'" in problem for problem in strict.problems())


def test_raise_on_invalid_raises_only_when_asked(workflow_path):
    turn_row, _ = obs.serialize_turn_result(_turn())
    with pytest.raises(EvidenceRunInvalid) as raised:
        with evidence_run(
            workflow_path, run_id="run-strictfail", raise_on_invalid=True
        ):
            obs.get_observability_sink(workflow_path)._requeue_records(
                [("turn", turn_row, [], obs._RECORD_BUSY_MAX_RETRIES)]
            )
    assert raised.value.run_id == "run-strictfail"
    assert raised.value.problems


def test_a_body_exception_is_not_replaced_by_an_evidence_exception(workflow_path):
    """The body's failure is the cause; incomplete evidence is a consequence.
    Raising the consequence would send the reader looking in the wrong place."""
    turn_row, _ = obs.serialize_turn_result(_turn())
    with pytest.raises(ValueError, match="the real failure"):
        with evidence_run(
            workflow_path, run_id="run-bodyfail", raise_on_invalid=True
        ):
            obs.get_observability_sink(workflow_path)._requeue_records(
                [("turn", turn_row, [], obs._RECORD_BUSY_MAX_RETRIES)]
            )
            raise ValueError("the real failure")


def test_a_crashed_run_is_still_verified_and_archived(workflow_path, tmp_path):
    """A run that crashed is exactly when its evidence is most worth keeping."""
    with pytest.raises(ValueError):
        with evidence_run(
            workflow_path, run_id="run-crash", archive_dir=str(tmp_path / "bundle")
        ) as run:
            obs.get_observability_sink(workflow_path).emit_turn_record(_turn())
            raise ValueError("boom")

    assert run.delta is not None
    assert run.archive is not None
    assert Path(run.archive["path"]).exists()


# ----------------------------------------------------------------------
# Cross-process runs must not certify themselves (fix-ajv.13)
# ----------------------------------------------------------------------
#
# The gate used to call get_observability_sink(), which CONSTRUCTS one. A
# harness driving a separate server process therefore got a sink of its own,
# and both snapshots read that sink's counters — which are zero and stay zero,
# because the server is the process doing the dropping. Delta all zeros,
# incomparable=False, evidence_valid=True, for a run that measured nothing.
# Constructing it also started a second writer thread against the server's
# database and ran SQLiteTraceSink.__init__'s opportunistic prune, so the
# evidence gate pruned evidence on its way in.


def test_the_gate_does_not_construct_a_sink(workflow_path):
    """The load-bearing regression: opening the gate must not make this process
    a writer, because that is what silently replaced the server's counters with
    the harness's own — and what ran a prune before suppression was entered."""
    assert obs.existing_observability_sink(workflow_path) is None

    with evidence_run(workflow_path, run_id="run-nosink"):
        assert obs.existing_observability_sink(workflow_path) is None

    assert obs.existing_observability_sink(workflow_path) is None


def test_a_run_with_no_readable_writer_health_is_not_valid(workflow_path):
    """Unknown is not a pass. Zeros out of a counter nobody wrote are not a
    measurement of zero, and this is exactly the shape a cross-process harness
    had: no sink here, nothing published there."""
    with evidence_run(workflow_path, run_id="run-unknown") as run:
        pass

    assert run.valid is False
    assert run.in_process is False
    assert any("no writer health is available" in p for p in run.problems())


def test_a_sink_installed_during_the_run_is_measured_from_zero(workflow_path):
    """The ordinary in-process shape: the gate opens, then the workflow starts
    and installs the sink. Its counters began at zero INSIDE the block, so every
    drop it counted belongs to this run and the baseline is empty — not the
    persisted row, which would belong to a different writer instance."""
    with evidence_run(workflow_path, run_id="run-appears") as run:
        sink = obs.get_observability_sink(workflow_path)
        sink.emit_turn_record(_turn(0))

    assert run.in_process is True
    assert run.health_before == {}
    assert run.delta.incomparable is False
    assert run.valid


def test_the_record_says_which_process_the_verdict_came_from(workflow_path, tmp_path):
    """A reader cannot otherwise tell a measured interval from an assumed one."""
    with evidence_run(
        workflow_path, run_id="run-topology", archive_dir=str(tmp_path / "b")
    ) as run:
        sink = obs.get_observability_sink(workflow_path)
        sink.emit_turn_record(_turn(0))

    record = run.as_record()
    assert record["in_process"] is True
    assert json.dumps(record)
