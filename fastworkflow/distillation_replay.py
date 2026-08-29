"""Counterfactual replay: turn a stored (turn, teacher-trace) pair into a test.

`fix-sb8.11`, design §14 `[DR34]` / `[DR35]` / `[DR41]` / `[DR45]`.

An insight is a hypothesis. The stored trace corpus is its test set. Because
the teacher trace is persisted, testing one costs a single small-model run
instead of two: re-run **only** the student from the recorded entry state with
a candidate insight set injected, then diff against the STORED teacher trace
and ask whether the cited divergence disappeared.

That is also exactly the shape a planner *verifier* needs — the stored pair is
the fixture and "the student now matches the teacher on the cited dimension" is
the assertion — which is why this module is the bridge from insight to
verifier rather than another view.

**What this is not.** Application object state is not captured and cannot be by
this design (`[DR35]`), so replay is *world-gated*, never world-reconstructing.
Four gates must all hold or the answer is `not-replayable: <reason>` and no
verdict is written:

1. the origin run is `comparable = 1`;
2. `isolation_verified = 1` (`[DR48]`) — a causal claim needs the isolation
   guarantee `fix-35m.3` owns, and until that lands every replay declines here;
3. the entry `state_fingerprint` AND `prompt_fingerprint` match the stored
   student pass's (`[DR45]`);
4. every step the replay re-ran identically observed what the stored student
   pass observed (`[DR45]`'s per-step gate).

Selling replay as a general regression harness would be a claim the state
capture does not support, and it would be discovered the first time a replay
"validated" an insight about a wrong customer id by re-running against a world
the teacher had already mutated.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from fastworkflow import distillation_alignment as alignment
from fastworkflow import tracing
from fastworkflow.distillation import (
    DistillationSession,
    _attr_text,
    _utc_now,
    prompt_fingerprint,
    state_fingerprint,
)
from fastworkflow.observability_store import ObservabilityStore, Redactor
from fastworkflow.utils.logging import logger

# The replay's own pass label and role. The LABEL is `student`, so a replay
# trace answers the same `distillation_pass = 'student'` filter every other
# reader uses; the ROLE is `student-replay`, which is what §9's role column
# exists to distinguish and what keeps the teacher-vs-student cost aggregate
# from counting replays as student passes.
REPLAY_PASS_LABEL = "student"
REPLAY_ROLE = "student-replay"

NOT_REPLAYABLE = "not-replayable"
VERDICT = "verdict"

# §14's assertion writes one of these two, never a human's vocabulary.
VERDICT_SUPPORTED = "supported"
VERDICT_UNSUPPORTED = "not-supported-by-cited-evidence"


@dataclass
class ReplayOutcome:
    """What a replay concluded, and why.

    `status` is `verdict` or `not-replayable`. A `not-replayable` outcome is a
    first-class answer, not an error: it says the corpus cannot currently
    adjudicate this insight, which is different from — and must never be
    rendered as — "the insight failed".
    """

    origin_run_id: str
    status: str
    reason: Optional[str] = None
    run_id: Optional[str] = None
    trace_id: Optional[str] = None
    divergence_removed: Optional[bool] = None
    verdict: Optional[str] = None
    cited_divergences: list[str] = field(default_factory=list)
    remaining: list[dict[str, Any]] = field(default_factory=list)

    @property
    def replayable(self) -> bool:
        return self.status == VERDICT


def _not_replayable(origin_run_id: str, reason: str, **extra: Any) -> ReplayOutcome:
    logger.info(f"Distillation replay {origin_run_id}: not replayable — {reason}")
    return ReplayOutcome(
        origin_run_id=origin_run_id,
        status=NOT_REPLAYABLE,
        reason=reason,
        **extra,
    )


# ----------------------------------------------------------------------
# Stored spans -> alignment steps
# ----------------------------------------------------------------------

_ACTION_SPAN_NAMES = (tracing.SPAN_COMMAND_EXECUTE, tracing.SPAN_ASK_USER)


def _row_attributes(row: Any) -> dict[str, Any]:
    try:
        return json.loads(row["attributes"]) or {}
    except (ValueError, TypeError, KeyError):
        return {}


def steps_from_stored_spans(rows: list[Any]) -> list:
    """Alignment steps built from STORED span rows.

    The live path builds these from in-memory `Span` objects held by the pass
    span collector; a replay's left-hand side is a teacher trace recorded
    weeks ago, so it has to come out of the table. Same fields, same
    constructors, so the two sides align through identical keys — anything
    else would make a replay's diff incomparable with the diff it is testing.
    """
    steps = []
    for row in sorted(rows, key=lambda r: r["start_ns"] or 0):
        attributes = _row_attributes(row)
        if row["name"] == tracing.SPAN_ASK_USER:
            steps.append(
                alignment.make_ask_user_step(
                    row["span_id"],
                    _attr_text(attributes.get("agent_query")) or "",
                    context=row["context"],
                    start_ns=row["start_ns"],
                )
            )
            continue
        parameters = attributes.get("parameters")
        steps.append(
            alignment.make_command_step(
                row["span_id"],
                command_name=row["command_name"] or None,
                context=row["context"],
                parameters=parameters if isinstance(parameters, dict) else None,
                raw_command=_attr_text(attributes.get("raw_command")),
                start_ns=row["start_ns"],
            )
        )
    return steps


def _observation(step_source: Any) -> Optional[str]:
    """A step's `response_text`, from a stored row or a live `Span`."""
    if isinstance(step_source, dict):
        attributes = step_source
    else:
        attributes = getattr(step_source, "attributes", None) or {}
    return _attr_text(attributes.get("response_text"))


def observation_drift(
    stored: list[tuple[str, Optional[dict], Optional[str]]],
    replayed: list[tuple[str, Optional[dict], Optional[str]]],
) -> Optional[int]:
    """`[DR45]`'s per-step gate: the index where the world moved, or None.

    Each entry is `(command_name, parameters, response_text)`. The comparison
    is deliberately narrow: only positions where the replay re-ran the SAME
    command with the SAME parameters are compared, because those are the only
    positions where a differing observation can mean the world moved rather
    than the agent chose differently. Where the replay took a different action
    — which is the whole point of injecting an insight — there is nothing to
    compare and the gate says nothing.

    An entry-only gate is not enough on its own: a ReAct agent's next action is
    a function of the previous tool's `response_text`, so a world that moves
    mid-replay produces a trajectory the fingerprints already certified as
    comparable.
    """
    for index, (stored_step, replay_step) in enumerate(zip(stored, replayed)):
        if stored_step[0] != replay_step[0]:
            continue
        if stored_step[1] != replay_step[1]:
            continue
        if stored_step[2] != replay_step[2]:
            return index
    return None


# ----------------------------------------------------------------------
# The driver
# ----------------------------------------------------------------------


def _store_for(chat_session, db_path: Optional[str]) -> Optional[ObservabilityStore]:
    if db_path:
        return ObservabilityStore.open_for_annotation(db_path)
    store = getattr(tracing.get_sink(chat_session), "store", None)
    return store


def mint_replay_trace(
    store: ObservabilityStore, origin: dict[str, Any], insight_set_json: Optional[str]
) -> tuple[str, str]:
    """Reserve `<turn_key>~replay.<n>` and the run row, transactionally `[DR5]`.

    The suffix is minted inside the same `BEGIN IMMEDIATE` that inserts the
    row precisely because observability writes are otherwise best-effort
    (`[R14]`): a suffix chosen outside the transaction could collide, and the
    collision would surface as a silently missing pass row rather than an
    error. This is one of `[DR46]`'s two off-turn-thread exemptions — a replay
    is a deliberate offline operation, not a user's turn.
    """
    run_id = f"rpl-{uuid.uuid4().hex[:12]}"
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        taken = conn.execute(
            "SELECT COUNT(*) FROM distillation_runs WHERE replay_of=?",
            (origin["run_id"],),
        ).fetchone()[0]
        trace_id = f"{origin['turn_key']}~replay.{taken + 1}"
        conn.execute(
            "INSERT INTO distillation_runs (run_id, turn_key, channel_id, "
            "conversation_id, user_message, workflow_name, entry_context, "
            "comparable, replay_of, replay_trace_id, insight_set_json, "
            "started_at, run_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                origin["turn_key"],
                origin["channel_id"],
                origin["conversation_id"],
                origin["user_message"],
                origin["workflow_name"],
                origin["entry_context"],
                # A replay opens non-comparable for the same reason a live run
                # does: no pass has reached a boundary yet, so the only honest
                # value is 0 until the gates have all been cleared.
                0,
                origin["run_id"],
                trace_id,
                insight_set_json,
                _utc_now(),
                json.dumps(
                    {"status": "started", "replay_of": origin["run_id"]},
                    separators=(",", ":"),
                ),
            ),
        )
        conn.commit()
    return run_id, trace_id


def replay_run(
    chat_session,
    origin_run_id: str,
    *,
    planning_insights: Optional[str] = None,
    execution_insights: Optional[str] = None,
    db_path: Optional[str] = None,
) -> ReplayOutcome:
    """Re-run the student against a stored teacher trace, and report.

    `planning_insights` / `execution_insights` are the CANDIDATE corpora — the
    hypothesis under test. Passing neither replays the run with whatever the
    session currently carries, which answers the weaker question "does the
    divergence still reproduce".
    """
    store = _store_for(chat_session, db_path)
    if store is None:
        return _not_replayable(origin_run_id, "no observability store")

    with store._connect() as conn:
        origin_row = conn.execute(
            "SELECT * FROM distillation_runs WHERE run_id=?", (origin_run_id,)
        ).fetchone()
        if origin_row is None:
            return _not_replayable(origin_run_id, "origin run not found")
        origin = dict(origin_row)
        student_pass_row = conn.execute(
            "SELECT * FROM distillation_passes WHERE run_id=? AND pass_label='student'",
            (origin_run_id,),
        ).fetchone()
        cited_rows = conn.execute(
            "SELECT d.* FROM distillation_divergences d "
            "JOIN distillation_insight_citations c "
            "  ON c.divergence_id = d.divergence_id "
            "JOIN distillation_insights i ON i.insight_id = c.insight_id "
            "WHERE i.run_id=? ORDER BY d.align_index",
            (origin_run_id,),
        ).fetchall()
        teacher_rows = conn.execute(
            "SELECT * FROM spans WHERE trace_id=? AND distillation_pass='teacher' "
            "AND name IN (?, ?) ORDER BY start_ns",
            (origin["turn_key"], *_ACTION_SPAN_NAMES),
        ).fetchall()
        stored_student_rows = conn.execute(
            "SELECT * FROM spans WHERE trace_id=? AND distillation_pass='student' "
            "AND name IN (?, ?) ORDER BY start_ns",
            (origin["turn_key"], *_ACTION_SPAN_NAMES),
        ).fetchall()

    # --- Gate 1: the origin run's own evidence has to be usable -----------
    if not origin["comparable"]:
        return _not_replayable(
            origin_run_id,
            f"origin run is not comparable ({origin['comparable_reason']})",
        )
    if origin["replay_of"]:
        return _not_replayable(origin_run_id, "origin run is itself a replay")
    if origin["evidence_pruned"]:
        return _not_replayable(origin_run_id, "the origin run's evidence was pruned")

    # --- Gate 2: promotion is a causal claim [DR48] -----------------------
    if origin["isolation_verified"] != 1:
        return _not_replayable(
            origin_run_id,
            "application-object isolation is not verified for the origin run",
        )

    cited = [dict(r) for r in cited_rows]
    if not cited:
        return _not_replayable(origin_run_id, "the origin run cites no divergence")
    unreplayable = [row for row in cited if not row["replayable"]]
    if unreplayable:
        # The cheap pre-filter, not the argument ([DR35] as restated): it saves
        # starting a replay whose per-step observation gate is certain to fail.
        return _not_replayable(
            origin_run_id,
            f"cited divergence kind {unreplayable[0]['kind']!r} is not replayable",
        )
    if student_pass_row is None:
        return _not_replayable(origin_run_id, "the origin run has no student pass row")
    student_pass = dict(student_pass_row)
    if not teacher_rows:
        return _not_replayable(
            origin_run_id, "the stored teacher trace has no action spans"
        )

    # --- Gate 3: the world at entry [DR45] --------------------------------
    history_bound = student_pass.get("history_bound")
    if history_bound is None:
        history_bound = len(chat_session.conversation_history.messages)
    try:
        live_state = state_fingerprint(chat_session)
        live_prompt = prompt_fingerprint(chat_session, history_bound=history_bound)
    except Exception as exc:
        return _not_replayable(origin_run_id, f"entry fingerprint failed: {exc!r}")
    if live_state != student_pass.get("entry_fingerprint"):
        return _not_replayable(origin_run_id, "world drifted at entry")
    if (
        student_pass.get("entry_prompt_fingerprint")
        and live_prompt != student_pass["entry_prompt_fingerprint"]
    ):
        return _not_replayable(origin_run_id, "prompt inputs drifted at entry")

    # --- Run the student, and only the student ----------------------------
    session = DistillationSession(chat_session)
    original = (
        getattr(chat_session, "_planning_insights", None),
        getattr(chat_session, "_execution_insights", None),
    )
    if planning_insights is not None:
        chat_session._planning_insights = planning_insights
    if execution_insights is not None:
        chat_session._execution_insights = execution_insights
    run_id, trace_id = mint_replay_trace(
        store,
        origin,
        json.dumps(session.insight_set(), separators=(",", ":")),
    )
    session.run_id = run_id

    chat_session.clear_action_log()
    # Hold the replay's spans in memory as the sink emits them, exactly as a
    # live pass does: aligning against the table while the writer is still
    # draining would read a merely late span as a real divergence. A sink-less
    # session simply collects nothing and the alignment below sees no steps.
    session.install_span_collector()
    replay_span = None
    replay_status = tracing.STATUS_OK
    try:
        with chat_session.replay_trace_scope(trace_id):
            replay_span = tracing.start_span(
                chat_session,
                tracing.SPAN_DISTILL_REPLAY,
                attributes={
                    "run_id": run_id,
                    "replay_of": origin_run_id,
                    "replay_trace_id": trace_id,
                    "cited_divergences": [row["divergence_id"] for row in cited],
                },
            )
            try:
                session._run_agent_pass(
                    origin["user_message"],
                    "LLM_STUDENT_AGENT",
                    "LITELLM_API_KEY_STUDENT_AGENT",
                    "LLM_STUDENT_PLANNER",
                    "LITELLM_API_KEY_STUDENT_PLANNER",
                    pass_label=REPLAY_PASS_LABEL,
                    role=REPLAY_ROLE,
                    seq=0,
                )
            except Exception as exc:
                replay_status = tracing.STATUS_ERROR
                _finish(
                    store,
                    run_id,
                    {
                        "completed_at": _utc_now(),
                        "run_json": json.dumps(
                            {
                                "status": "replay-failed",
                                "replay_of": origin_run_id,
                                "error_type": type(exc).__name__,
                            },
                            separators=(",", ":"),
                        ),
                    },
                )
                return _not_replayable(
                    origin_run_id,
                    f"the student pass raised {type(exc).__name__}",
                    run_id=run_id,
                    trace_id=trace_id,
                )
            replay_steps = session.action_steps(REPLAY_PASS_LABEL)
    finally:
        session.remove_span_collector()
        chat_session._planning_insights, chat_session._execution_insights = original

    # --- Gate 4: the world during the replay [DR45] -----------------------
    stored_observations = [
        (
            row["command_name"],
            _row_attributes(row).get("parameters"),
            _attr_text(_row_attributes(row).get("response_text")),
        )
        for row in stored_student_rows
    ]
    replay_spans = session._pass_spans(REPLAY_PASS_LABEL, _ACTION_SPAN_NAMES)
    replay_observations = [
        (
            span.command_name,
            (span.attributes or {}).get("parameters"),
            _observation(span),
        )
        for span in replay_spans
    ]
    drift_index = observation_drift(stored_observations, replay_observations)
    if drift_index is not None:
        _end_replay_span(chat_session, replay_span, tracing.STATUS_OK, {
            "replayable": False,
            "reason": f"world drifted at step {drift_index}",
        })
        _finish(
            store,
            run_id,
            {
                "completed_at": _utc_now(),
                "run_json": json.dumps(
                    {
                        "status": "not-replayable",
                        "replay_of": origin_run_id,
                        "reason": f"world drifted at step {drift_index}",
                    },
                    separators=(",", ":"),
                ),
            },
        )
        return _not_replayable(
            origin_run_id,
            f"world drifted at step {drift_index}",
            run_id=run_id,
            trace_id=trace_id,
        )

    # --- The assertion (§14) ----------------------------------------------
    result = alignment.align_passes(
        run_id=run_id,
        left_pass="teacher",
        right_pass=REPLAY_PASS_LABEL,
        left_steps=steps_from_stored_spans(list(teacher_rows)),
        right_steps=replay_steps,
        comparable=True,
        level=alignment.LEVEL_ACTION,
    )
    cited_pairs = {(row["command_key"], row["kind"]) for row in cited}
    remaining = [
        record.to_row()
        for record in result.records
        if (record.command_key, record.kind) in cited_pairs
    ]
    divergence_removed = not remaining

    for record in result.records:
        row = record.to_row()
        row["run_id"] = run_id
        _insert_divergence(store, row)

    _end_replay_span(
        chat_session,
        replay_span,
        replay_status,
        {
            "replayable": True,
            "divergence_removed": divergence_removed,
            "remaining_divergences": len(remaining),
        },
    )
    verdict = VERDICT_SUPPORTED if divergence_removed else VERDICT_UNSUPPORTED
    _finish(
        store,
        run_id,
        {
            "comparable": 1,
            "completed_at": _utc_now(),
            "exec_diverged": 0 if divergence_removed else 1,
            "material_divergences": sum(
                1 for record in result.records if record.material == 1
            ),
            "run_json": json.dumps(
                {
                    "status": "completed",
                    "replay_of": origin_run_id,
                    "divergence_removed": divergence_removed,
                },
                separators=(",", ":"),
            ),
        },
    )
    _record_replay_verdicts(store, origin_run_id, run_id, verdict, divergence_removed)
    return ReplayOutcome(
        origin_run_id=origin_run_id,
        status=VERDICT,
        run_id=run_id,
        trace_id=trace_id,
        divergence_removed=divergence_removed,
        verdict=verdict,
        cited_divergences=[row["divergence_id"] for row in cited],
        remaining=remaining,
    )


def _end_replay_span(
    chat_session, span: Any, status: str, attributes: dict[str, Any]
) -> None:
    """Close the replay wrapper span.

    No `replay_trace_scope` needed: `end_span` emits the `Span` object it is
    handed, whose `trace_id` was fixed when `start_span` minted it inside the
    scope. Re-entering the scope here would only look like it mattered.
    """
    tracing.end_span(chat_session, span, status=status, attributes=attributes)


def _insert_divergence(store: ObservabilityStore, row: dict[str, Any]) -> None:
    """A replay's own divergence rows, written directly.

    [DR46]'s exemption again: there is no turn thread to protect here, and the
    replay must be able to report before the process exits.
    """
    try:
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            store.upsert_distillation_row(conn, "divergence", row, Redactor())
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning(f"Distillation replay: divergence row not written: {exc!r}")


def _finish(store: ObservabilityStore, run_id: str, fields: dict[str, Any]) -> None:
    assignments = ", ".join(f"{name}=?" for name in fields)
    try:
        with store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"UPDATE distillation_runs SET {assignments} WHERE run_id=?",
                (*fields.values(), run_id),
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning(f"Distillation replay: run row not completed: {exc!r}")


def _record_replay_verdicts(
    store: ObservabilityStore,
    origin_run_id: str,
    replay_run_id: str,
    verdict: str,
    divergence_removed: bool,
) -> None:
    """One `actor = 'replay'` verdict per insight the origin run produced.

    §12's supersede rule applies unchanged, which is the point: an insight
    accepted by a human and then contradicted by a replay leaves BOTH rows, and
    the history of judgements is itself the evidence a promotion decision reads.
    """
    note = (
        "replay removed the cited divergence"
        if divergence_removed
        else "replay reproduced the cited divergence"
    )
    with store._connect() as conn:
        insight_ids = [
            r[0]
            for r in conn.execute(
                "SELECT insight_id FROM distillation_insights WHERE run_id=?",
                (origin_run_id,),
            ).fetchall()
        ]
    for insight_id in insight_ids:
        try:
            store.insert_verdict(
                insight_id=insight_id,
                verdict=verdict,
                actor="replay",
                note=f"{note} (replay {replay_run_id})",
                replay_run_id=replay_run_id,
            )
        except (ValueError, sqlite3.Error) as exc:
            logger.warning(
                f"Distillation replay: verdict for {insight_id} not written: {exc!r}"
            )
