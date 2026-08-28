"""Evidence-grade capture: a run mode, not a global change.

Architecture §12.0 delta 6 and requirements §12.4. The 3.2.0 observability store
is best-effort by design — spans ride a droppable queue, turn records ride a small
one, and retention prunes at thirty days or a gigabyte. That posture is right for
a developer debugging a workflow and wrong for a measured run, where a silently
dropped turn record means the reported score is computed over a denominator nobody
can reconstruct.

This module does not change that posture. It wraps a run in four checks and leaves
normal operation exactly as it was:

1. **Zero-drop assertion.** Writer-health counters are snapshotted before and
   after. A dropped turn record makes the run invalid, full stop. A dropped span
   does not — spans are best-effort — but it is reported with the turn keys it
   affected, so a reader knows which turns have incomplete detail instead of
   assuming all of them are whole.

2. **Prune suppression.** §12.4: "pruning shall not run mid-evaluation". The
   thirty-day horizon sounds like it could not possibly bite inside one run, but
   the size cap evicts oldest-first regardless of age, so a long or high-volume run
   can delete its own early spans while still recording its later ones.

3. **Archival.** The run's DB is copied into the bundle before retention can ever
   reach it, with a digest, via ``VACUUM INTO`` rather than a file copy — see
   `ObservabilityStore.archive_to` for why copying a WAL-mode database silently
   loses the end of the run.

4. **Provenance.** The `FW_OBS_*` values in effect, the capture profile and its
   policy version, the span-contract version, and the DB schema version. Trace
   evidence without these is uninterpretable rather than merely undocumented: the
   same workflow under the `evidence` and `debug` profiles produces records that
   differ in what they contain, not in what happened.

**What this module cannot promise.** It asserts that the *store* did not drop what
it was given. It cannot assert that everything which happened was emitted — an
emitter that was never called leaves no trace and no drop counter, so a run can be
zero-drop and still incomplete. `fix-ajv.9`'s unredacted write paths and the
un-migrated dispatch paths of §12.1.1 are both examples. Zero-drop is a floor.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastworkflow import capture_policy, observability_store, state_paths, tracing
from fastworkflow.provenance import ObservabilityProvenance
from fastworkflow.utils.logging import logger


class EvidenceRunInvalid(RuntimeError):
    """A run that was required to be evidence-grade was not.

    Carries every problem found rather than the first, so an operator sees the
    whole picture in one pass — same reasoning as
    `runtime_manifest.ManifestConformanceError`.
    """

    def __init__(self, run_id: str, problems: tuple[str, ...]) -> None:
        self.run_id = run_id
        self.problems = problems
        detail = "\n".join(f"  {i}. {p}" for i, p in enumerate(problems, start=1))
        super().__init__(f"evidence run {run_id} is not valid evidence:\n{detail}")


def capture_observability_provenance(
    *,
    dspy_history_enabled: Optional[bool] = None,
    evidence_grade: Optional[bool] = None,
) -> ObservabilityProvenance:
    """Snapshot the capture regime in effect (§12.4).

    Lives here rather than in `provenance` because it reads `observability_store`
    and `tracing`, neither of which is a §22 leaf.
    """
    config = observability_store.observability_config()
    return ObservabilityProvenance(
        enabled=observability_store.observability_enabled(default_on=True),
        capture_profile=config[observability_store.CAPTURE_PROFILE_VAR],
        capture_policy_version=capture_policy.CAPTURE_POLICY_VERSION,
        span_contract_version=tracing.SPAN_CONTRACT_VERSION,
        span_contract_versions=tracing.span_contract_versions(),
        db_schema_version=observability_store.SCHEMA_VERSION,
        config=config,
        dspy_history_enabled=dspy_history_enabled,
        evidence_grade=evidence_grade,
    )


@dataclass
class EvidenceRun:
    """One measured run's evidence record. Built by `evidence_run()`."""

    run_id: str
    db_path: str
    provenance: ObservabilityProvenance
    started_at: datetime
    health_before: Optional[dict[str, Any]] = None
    health_after: Optional[dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    delta: Optional[observability_store.WriterHealthDelta] = None
    archive: Optional[dict[str, Any]] = None
    extra_problems: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether this run may be reported as evidence.

        An unfinished run is not valid: `delta is None` means the interval was
        never closed, and treating that as a pass would let a crashed run be
        reported as clean.
        """
        return (
            self.delta is not None
            and self.delta.evidence_valid
            and not self.extra_problems
        )

    def problems(self) -> tuple[str, ...]:
        if self.delta is None:
            return ("the evidence run never completed, so nothing was verified",)
        return tuple(self.delta.problems()) + tuple(self.extra_problems)

    def as_record(self) -> dict[str, Any]:
        """The serializable summary a harness writes into its bundle."""
        return {
            "run_id": self.run_id,
            "db_path": self.db_path,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "valid": self.valid,
            "problems": list(self.problems()),
            "observability": self.provenance.model_dump(mode="json"),
            "writer_health_before": self.health_before,
            "writer_health_after": self.health_after,
            "writer_health_delta": (
                self.delta.model_dump(mode="json") if self.delta else None
            ),
            "archive": self.archive,
        }


def _health_snapshot(
    db_path: str, sink: Optional[observability_store.SQLiteTraceSink]
) -> Optional[dict[str, Any]]:
    """The live counters if a sink is in this process, else the persisted row.

    Prefers the sink because the `diagnostics` row lags by up to a heartbeat: a
    run that read only the row could snapshot a drop that had already happened as
    though it had not. The row is the cross-process fallback, and `None` from both
    is reported as `incomparable` rather than as zero.
    """
    if sink is not None and not sink._closed:
        return sink.health_snapshot()
    try:
        return observability_store.ObservabilityStore(db_path).writer_health()
    except Exception as exc:
        logger.warning(f"Could not read writer health from {db_path}: {exc!r}")
        return None


@contextlib.contextmanager
def evidence_run(
    workflow_folderpath: str,
    *,
    run_id: Optional[str] = None,
    archive_dir: Optional[str] = None,
    dspy_history_enabled: Optional[bool] = None,
    require_evidence_profile: bool = False,
    raise_on_invalid: bool = False,
):
    """Record a measured run, then verify nothing was silently lost.

    Yields an `EvidenceRun` that is populated on exit — `valid`, `problems()` and
    `archive` are only meaningful after the block ends.

    Pruning is suppressed for the block. On exit the sink is flushed, its counters
    are persisted so the archive carries the same verdict as the in-process delta,
    the health delta is computed, and the DB is archived if `archive_dir` is given.

    `raise_on_invalid` is opt-in rather than the default. A harness usually wants
    to record the verdict alongside its results and decide what to do with a
    partially-valid run itself; raising by default would discard the run's data to
    signal that the run's data is suspect.

    `require_evidence_profile` additionally demands
    `FW_OBS_CAPTURE_PROFILE=evidence`. Off by default because the profile governs
    exposure, not completeness — a `debug`-profile run loses no evidence, it just
    keeps more than a tenant deployment should.

    Archiving and verification run even when the body raises, because a run that
    crashed is exactly when its evidence is most worth keeping.
    """
    run_id = run_id or f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    db_path = state_paths.observability_db(workflow_folderpath)
    sink = observability_store.get_observability_sink(workflow_folderpath)

    provenance = capture_observability_provenance(
        dspy_history_enabled=dspy_history_enabled, evidence_grade=True
    )
    run = EvidenceRun(
        run_id=run_id,
        db_path=db_path,
        provenance=provenance,
        started_at=datetime.now(timezone.utc),
        health_before=_health_snapshot(db_path, sink),
    )
    if not provenance.enabled:
        run.extra_problems.append(
            "observability is disabled (FW_OBSERVABILITY), so this run recorded no "
            "trace evidence at all"
        )
    if require_evidence_profile and provenance.capture_profile != "evidence":
        run.extra_problems.append(
            f"capture profile is '{provenance.capture_profile}', not 'evidence'; "
            "unclassified fields were captured rather than withheld"
        )
    if dspy_history_enabled is False:
        run.extra_problems.append(
            "DSPy history is off, so this run has no token or cost evidence (§12.4)"
        )

    body_failed = False
    try:
        with observability_store.suppress_pruning():
            yield run
    except BaseException:
        # Tracked so the verification below cannot replace the body's exception
        # with its own. A run that crashed has a cause, and "the evidence is
        # incomplete" is a consequence of it — raising the consequence would hide
        # the cause and send the reader looking in the wrong place.
        body_failed = True
        raise
    finally:
        run.completed_at = datetime.now(timezone.utc)
        if sink is not None and not sink._closed:
            # Flush first: an unflushed queue makes the "after" snapshot describe
            # a moment before the last turns were written, so a drop during that
            # final write would land outside the measured interval.
            sink.flush()
            sink.persist_health()
        run.health_after = _health_snapshot(db_path, sink)
        run.delta = observability_store.health_delta(run.health_before, run.health_after)

        if archive_dir is not None:
            try:
                target = Path(archive_dir) / f"{run_id}-observability.sqlite3"
                run.archive = observability_store.ObservabilityStore(db_path).archive_to(
                    str(target)
                )
            except Exception as exc:
                run.extra_problems.append(f"evidence archival failed: {exc!r}")
                logger.warning(f"Evidence archival failed for {run_id}: {exc!r}")

        if run.valid:
            logger.info(f"Evidence run {run_id}: valid, no evidence lost")
        else:
            for problem in run.problems():
                logger.warning(f"Evidence run {run_id}: {problem}")
            if raise_on_invalid and not body_failed:
                raise EvidenceRunInvalid(run_id, run.problems())
