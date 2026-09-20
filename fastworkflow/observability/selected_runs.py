"""A bounded, explicitly chosen set of whole runs, summarized once.

`fix-9eg.3.2.1`. A reader looking at one task's repeated attempts asks "what
did THESE five runs do", names them, and gets one answer over exactly those
five. Nothing here looks for runs: the caller names them, the server resolves
them, and a run nobody named is not in the answer -- which is what keeps this
apart from the all-runs scan (`fix-9eg.3.2.2`) that remains deferred.

The whole of the arithmetic is the reducer `command_summary` already applies
to one run. It takes observations, not projections, and each observation
carries the run it came from, so pooling several runs' observations is the
supported use of it rather than a second implementation of it. That is why the
pooled median is a median of the underlying DURATIONS, not a median of the
per-run medians, and why five runs with wildly unequal dispatch counts pool
correctly without weighting anything.

What this module refuses, because each one would be a quiet wrong answer:

- **Part of a run.** Members are whole finished runs. No pass scope, no turn
  subset: a pooled figure over "the teacher pass of run 1 and all of run 2"
  describes nothing.
- **Runs that have not finished.** A run still going has not produced the
  behaviour anybody is aggregating. Named ones are excluded and listed, never
  silently dropped.
- **More than one source, experiment or task.** Every member comes from one
  authorized source's record of one task in one experiment. Two members that
  resolve to different stores are refused rather than merged.
- **Overlapping members.** Two members naming the same turn would count its
  dispatches twice, so that is a refusal and not a deduplication.
- **A winner.** There is no ranking, no representative run and no composite
  score here. Counts by outcome, the recorded duration order statistics with
  their coverage, and the canonical recorded LLM cost. Nothing derived.

A finished run whose evidence cannot be read is a MEMBER, not an exclusion. It
was asked for, it finished, and it contributes no observations -- so it stays
in the population and says so. Dropping it would shrink the denominator of
every figure here to the runs that happened to be readable.

Validation, and what it does not promise
----------------------------------------

Evidence is not frozen and this module does not freeze it. `evidence_digest`
is derived from the ACTUAL projected values -- each dispatch's recorded
outcome and duration, each turn's answer and cost, the canonical LLM calls --
so a span edited in place, with every identifier unchanged, changes it.
`validate_selection` re-projects the named runs through the same reader and
the same code path and says which members no longer match what a caller was
told. That is optimistic validation: it detects a change, it does not prevent
one, and a member can still change between the check and the navigation that
follows it. There is no snapshot, no cache and no stored copy of evidence
anywhere in here.

For a coding agent
------------------

Two GETs, the same ones the page uses::

    GET /api/experiments/exp-7/tasks/task-3/selected-runs?attempt=1&attempt=2&attempt=5

    {"members": [{"attempt": 1, "ref_id": "xr-...", "evidence_digest": "mev-...", ...}],
     "population": {"requested": 3, "included": 3, "unfinished": [], ...},
     "command_summary": {"groups": [...], "totals": {...}, "coverage": {...}},
     "cost": {"calls": 9, "recorded": 7, "unrecorded": 2, "total": 0.0431},
     "evidence_digest": "ev-..."}

    GET /api/experiments/exp-7/tasks/task-3/selected-runs/validation
        ?attempt=5&expect_member=5:mev-...

    {"stale": true, "changed": [5],
     "members": [{"attempt": 5, "state": "changed", "evidence_digest": "mev-..."}]}

Attempt numbers are the only thing a request names. Turn keys, stores and
references are resolved from the evidence, so there is no scope a request can
assert and no store a request can point at.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from fastworkflow.observability import command_summary as command_summary_module
from fastworkflow.observability.comparison import (
    ExecutionRef,
    merge_usage_rollups,
    project_execution,
)

# Request safety, and nothing else. It is NOT a sample: a request naming more
# runs than this is refused so the caller knows its question was not answered,
# because silently summarizing the first twenty would publish a figure over a
# population the caller never chose.
MAX_SELECTED_RUNS = 20

# A separate, much looser bound on how many `attempt` parameters will even be
# parsed, so a malformed client cannot make the parser the denial of service
# the limit above exists to prevent.
MAX_ATTEMPT_PARAMS = 200

METRIC_SCOPE = "selected_runs"
DIGEST_BASIS = "selected_runs_evidence/1"

REASON_UNFINISHED = "unfinished"
REASON_NOT_RECORDED = "not_recorded"


class SelectedRunsError(Exception):
    """Base for refusals this module makes about a selection."""

    status = 400

    def __init__(self, message: str, *, reason: str, **payload: Any) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.payload = payload


class SelectionTooLarge(SelectedRunsError):
    status = 400


class SelectionIncoherent(SelectedRunsError):
    """The named runs cannot be one population: two stores, or overlapping
    evidence. Refused rather than merged, because both would be answered as a
    number nobody could check."""

    status = 409


def bound_attempts(attempts: Sequence[int]) -> tuple[list[int], int]:
    """`(distinct ascending attempts, duplicates collapsed)`, or a refusal.

    Sorted, so the answer does not depend on the order a page happened to
    render its checkboxes in, and deduplicated, so naming a run twice is one
    member rather than a doubled contribution.
    """
    if len(attempts) > MAX_ATTEMPT_PARAMS:
        raise SelectionTooLarge(
            f"this request names {len(attempts)} attempt parameters; at most "
            f"{MAX_ATTEMPT_PARAMS} are parsed",
            reason="too_many_parameters",
            max_parameters=MAX_ATTEMPT_PARAMS,
            requested=len(attempts),
        )
    distinct = sorted({int(attempt) for attempt in attempts})
    if not distinct:
        raise SelectedRunsError(
            "name at least one attempt to summarize, with one or more "
            "?attempt= parameters",
            reason="no_runs_selected",
        )
    if len(distinct) > MAX_SELECTED_RUNS:
        raise SelectionTooLarge(
            f"this request selects {len(distinct)} runs; at most "
            f"{MAX_SELECTED_RUNS} can be summarized in one request. This is a "
            "bound on the request, not a sample: nothing was summarized, "
            "because summarizing part of what was asked for would publish a "
            "figure over a population nobody chose",
            reason="too_many_runs",
            max_runs=MAX_SELECTED_RUNS,
            requested=len(distinct),
        )
    return distinct, len(attempts) - len(distinct)


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _digest(prefix: str, payload: Any) -> str:
    return prefix + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]


def _usage_without_anchors(usage: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """A usage roll-up with the per-call anchor list removed.

    The anchors belong to the turn that recorded them and are published there
    once; repeating them at selection scope would put the same span id on the
    wire twice and invite a reader to tally it twice.
    """
    return {
        key: value for key, value in dict(usage or {}).items() if key != "calls_detail"
    }


def _llm_call_values(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The recorded LLM calls of one run, reduced to what a reader is told.

    Kept for the digest and not for the payload: a cost that changed on an
    otherwise identical call must change the digest, and the aggregate reports
    canonical totals rather than re-listing every call.
    """
    calls: list[dict[str, Any]] = []
    for turn in turns:
        for call in ((turn.get("usage") or {}).get("calls_detail") or []):
            calls.append(
                {
                    "span_id": call.get("span_id"),
                    "turn_key": call.get("turn_key"),
                    "model": call.get("model"),
                    "cost": call.get("cost"),
                    "total_tokens": call.get("total_tokens"),
                    "usage_state": call.get("usage_state"),
                    "cache_state": call.get("cache_state"),
                }
            )
    calls.sort(key=lambda call: (str(call["turn_key"]), str(call["span_id"])))
    return calls


def _turn_values(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """What each turn RECORDED, as the digest sees it.

    The answer text is in here on purpose. A drill-down opens the answers
    first, so an answer that changed while the summary was on screen has to
    move the digest even though no count above it moved.
    """
    return sorted(
        (
            {
                "turn_key": turn.get("turn_key"),
                "turn_index": turn.get("turn_index"),
                "status": turn.get("status"),
                "success": turn.get("success"),
                "failure_reason": turn.get("failure_reason"),
                "answer": turn.get("answer"),
                "turn_answer": turn.get("turn_answer"),
                "cost": turn.get("cost"),
                "tokens": (turn.get("usage") or {}).get("tokens"),
            }
            for turn in turns
        ),
        key=lambda turn: str(turn["turn_key"]),
    )


@dataclass(frozen=True)
class SelectedRun:
    """One resolved member: the run's own row, its reference, and what it
    actually projected."""

    row: Mapping[str, Any]
    ref: Optional[ExecutionRef]
    projection: Any
    observations: tuple[command_summary_module.CommandObservation, ...]
    turns: tuple[Mapping[str, Any], ...]
    cost: Mapping[str, Any]
    usage: Mapping[str, Any]
    unavailable: tuple[str, ...]
    digest: str

    @property
    def attempt(self) -> int:
        return int(self.row["attempt"])

    @property
    def readable_turns(self) -> int:
        return len(self.turns)

    def as_member(self) -> dict[str, Any]:
        row = self.row
        return {
            "attempt": self.attempt,
            "label": row.get("label") or f"attempt {self.attempt}",
            "execution_status": row.get("execution_status"),
            "execution_finished_at": row.get("execution_finished_at"),
            "outcome": row.get("outcome"),
            "outcome_source": row.get("outcome_source"),
            "reward": row.get("reward"),
            "restarts": row.get("restarts"),
            "finished": bool(row.get("finished")),
            "turn_count": int(row.get("turn_count") or 0),
            "evidence_state": row.get("evidence_state"),
            "evidence_label": row.get("evidence_label"),
            "is_best": bool(row.get("is_best")),
            "is_reference": bool(row.get("is_reference")),
            "segment_id": row.get("segment_id"),
            "store_id": None if self.ref is None else self.ref.store_id,
            # An archive has TWO names and they are not interchangeable: the
            # reference carries the evidence identity, and every workspace
            # turn/span route is addressed by the manifest's own name for the
            # same archive. A drill-down that used the identity would not
            # open. Absent outside a workspace, where there is one name.
            "manifest_store_id": row.get("manifest_store_id"),
            "ref_id": None if self.ref is None else self.ref.ref_id(),
            "execution_ref": None if self.ref is None else self.ref.as_dict(),
            # The exact contribution, so a reader can check the pooled figures
            # against the runs rather than trust them.
            "dispatches": len(self.observations),
            "turns_read": self.readable_turns,
            "turns_unreadable": len(self.unavailable),
            "unreadable_turns": list(self.unavailable),
            "cost": dict(self.cost),
            "usage": _usage_without_anchors(self.usage),
            "evidence_digest": self.digest,
        }


def _run_facts(row: Mapping[str, Any]) -> dict[str, Any]:
    """The run facts this summary PUBLISHES about a member.

    In the digest because they are on the screen: the run-level outcome tally
    moves when one of these does, and a member whose outcome was corrected
    from `pass` to `fail` -- with the same turns, the same dispatches and the
    same costs -- must not validate as unchanged.

    `is_best` and `is_reference` are deliberately out. They are the current
    best-run POINTER's labels, they describe a decision rather than the
    evidence, and somebody pinning a different best run does not change what
    any of these runs recorded.
    """
    return {
        "attempt": int(row["attempt"]),
        "execution_status": row.get("execution_status"),
        "execution_finished_at": row.get("execution_finished_at"),
        "finished": bool(row.get("finished")),
        "outcome": row.get("outcome"),
        "outcome_source": row.get("outcome_source"),
        "reward": row.get("reward"),
        "restarts": row.get("restarts"),
        "turn_count": row.get("turn_count"),
        "evidence_state": row.get("evidence_state"),
    }


def _project_member(row: Mapping[str, Any], reader: Any) -> SelectedRun:
    """Project one named run and derive its evidence digest from the values.

    A finished run with no reference is projected as nothing: it is still a
    member, it simply contributes no observation, and the empty projection is
    what makes that state say itself rather than disappear.
    """
    raw_ref = row.get("execution_ref")
    ref = None if not raw_ref else ExecutionRef.from_mapping(raw_ref)
    if ref is None:
        digest = _digest("mev-", {"run": _run_facts(row), "evidence": None})
        return SelectedRun(
            row=row, ref=None, projection=None, observations=(), turns=(),
            cost={"calls": 0, "recorded": 0, "unrecorded": 0, "total": None},
            usage={}, unavailable=(), digest=digest,
        )
    projection = project_execution(ref, reader)
    data = projection.as_dict()
    observations = tuple(
        command_summary_module.observations_from_projection(projection)
    )
    turns = tuple(data.get("turns") or ())
    unavailable = tuple(data.get("unavailable") or ())
    # The digest is built from the SAME values the summary is built from, and
    # from the answers and costs a drill-down opens. An identifier-only digest
    # would call a run unchanged after its recorded outcome, its duration or
    # its cost had been edited in place.
    digest = _digest(
        "mev-",
        {
            "run": _run_facts(row),
            "ref": ref.as_dict(),
            "unavailable": sorted(unavailable),
            "turns": _turn_values(turns),
            "dispatches": sorted(
                (observation.as_ref() for observation in observations),
                key=lambda ref_: (
                    str(ref_.get("turn_key")), str(ref_.get("command_call_id"))
                ),
            ),
            "llm_calls": _llm_call_values(turns),
            "artifacts": data.get("artifacts"),
            "cost": data.get("cost"),
            "usage": _usage_without_anchors(data.get("usage")),
        },
    )
    return SelectedRun(
        row=row,
        ref=ref,
        projection=projection,
        observations=observations,
        turns=turns,
        cost=dict(data.get("cost") or {}),
        usage=dict(data.get("usage") or {}),
        unavailable=unavailable,
        digest=digest,
    )


def _resolve(
    *,
    requested: Sequence[int],
    recorded_attempts: Sequence[int],
    candidate_rows: Sequence[Mapping[str, Any]],
    reader: Any,
) -> tuple[list[SelectedRun], list[dict[str, Any]], list[int], list[int]]:
    """`(members, excluded, unfinished, not_recorded)` for one named selection."""
    by_attempt = {int(row["attempt"]): row for row in candidate_rows}
    recorded = {int(attempt) for attempt in recorded_attempts}
    members: list[SelectedRun] = []
    excluded: list[dict[str, Any]] = []
    unfinished: list[int] = []
    not_recorded: list[int] = []
    for attempt in requested:
        row = by_attempt.get(attempt)
        if row is None:
            not_recorded.append(attempt)
            excluded.append(
                {
                    "attempt": attempt,
                    "reason": REASON_NOT_RECORDED,
                    "detail": "this experiment has recorded no such attempt of "
                              "this task"
                              + (
                                  ""
                                  if not recorded
                                  else "; recorded attempts are "
                                       + ", ".join(str(a) for a in sorted(recorded))
                              ),
                }
            )
            continue
        if not row.get("finished"):
            unfinished.append(attempt)
            excluded.append(
                {
                    "attempt": attempt,
                    "reason": REASON_UNFINISHED,
                    "detail": "this attempt has not finished, so it has not "
                              "produced the behaviour a summary would be about",
                }
            )
            continue
        members.append(_project_member(row, reader))

    stores = {
        member.ref.store_id for member in members if member.ref is not None
    }
    if len(stores) > 1:
        raise SelectionIncoherent(
            "the selected runs resolve to more than one evidence store ("
            + ", ".join(sorted(str(store) for store in stores))
            + "); one summary over several sources would pool evidence whose "
            "identities are not comparable",
            reason="multiple_sources",
            store_ids=sorted(str(store) for store in stores),
        )
    seen: dict[str, int] = {}
    for member in members:
        if member.ref is None:
            continue
        for turn_key in member.ref.turn_keys:
            owner = seen.get(turn_key)
            if owner is not None:
                raise SelectionIncoherent(
                    f"attempts {owner} and {member.attempt} both name recorded "
                    f"turn {turn_key!r}; counting it under both would report "
                    "activity that happened once as though it happened twice",
                    reason="overlapping_runs",
                    turn_key=turn_key,
                    attempts=[owner, member.attempt],
                )
            seen[turn_key] = member.attempt
    return members, excluded, unfinished, not_recorded


def _population(
    *,
    requested: Sequence[int],
    members: Sequence[SelectedRun],
    excluded: Sequence[Mapping[str, Any]],
    unfinished: Sequence[int],
    not_recorded: Sequence[int],
) -> dict[str, Any]:
    missing_evidence = [
        member.attempt for member in members if member.ref is None
    ]
    readable = [member for member in members if member.readable_turns]
    return {
        "requested": len(requested),
        "included": len(members),
        "excluded": len(excluded),
        "with_readable_evidence": len(readable),
        # Asked for, finished, and unreadable: still a member, still in every
        # denominator here, contributing nothing and saying so.
        "missing_evidence": missing_evidence,
        "unfinished": list(unfinished),
        "not_recorded": list(not_recorded),
        "unreadable_turns": sum(len(member.unavailable) for member in members),
        "excluded_runs": [dict(row) for row in excluded],
    }


def _evidence_gaps(members: Sequence[SelectedRun]) -> dict[str, list[dict[str, Any]]]:
    """Which members were read whole, in part, or not at all.

    Three states, not two. A run that named four turns and yielded three is
    not "no evidence" and is not complete either, and both of the simpler
    labels would be a false statement about the counts it is inside.
    """
    partial: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for member in members:
        if member.ref is None or not member.unavailable:
            continue
        named = len(member.ref.turn_keys)
        row = {
            "attempt": member.attempt,
            "turns_named": named,
            "turns_read": member.readable_turns,
            "turns_unreadable": len(member.unavailable),
        }
        (unreadable if not member.readable_turns else partial).append(row)
    return {"partial": partial, "unreadable": unreadable}


def _gap_phrase(row: Mapping[str, Any]) -> str:
    return (
        f"attempt {row['attempt']}: {row['turns_read']} of "
        f"{row['turns_named']} turn(s) read"
    )


def _apply_population_coverage(
    summary: dict[str, Any],
    population: Mapping[str, Any],
    members: Sequence[SelectedRun],
) -> None:
    """Say, on the summary itself, which RUNS it is made of.

    The reducer's coverage answers a question about turns: were the turns
    this scope named readable, and were the dispatches in them fully
    recorded. A member with no reference names no turn, so it goes through
    that machinery invisibly -- and a selection of five runs, two of which
    recorded nothing, would otherwise publish `enumeration_complete: true`
    beside a population that says two runs are missing.

    So the run-level facts are stated here, on the coverage a reader is
    already looking at, and `enumeration_complete` is lowered when a run this
    selection asked about contributed no evidence. No missing turn count is
    invented: nobody knows how many turns a run that recorded none would have
    had, and guessing would be worse than saying it is unknown.
    """
    coverage = summary["coverage"]
    gaps_by_kind = _evidence_gaps(members)
    missing = list(population["missing_evidence"])
    unreadable = gaps_by_kind["unreadable"]
    partial = gaps_by_kind["partial"]
    coverage.update(
        {
            "runs_requested": population["requested"],
            "runs_included": population["included"],
            "runs_excluded": population["excluded"],
            "runs_with_readable_evidence": population["with_readable_evidence"],
            "runs_without_readable_evidence": len(missing) + len(unreadable),
            "runs_missing_evidence": missing,
            "runs_unreadable": [row["attempt"] for row in unreadable],
            # Read in part. Separate from the two above because it is a
            # different fact: some of this run IS in the counts, and calling
            # it "no evidence" would be as wrong as calling it complete.
            "runs_partially_read": [row["attempt"] for row in partial],
            "runs_unfinished_excluded": list(population["unfinished"]),
            "runs_not_recorded": list(population["not_recorded"]),
            # The honest headline: these counts describe the evidence that was
            # observed, and the population above says what that is a part of.
            "population_complete": (
                not missing
                and not unreadable
                and not partial
                and not population["unfinished"]
                and not population["not_recorded"]
            ),
            "totals_cover": "observed evidence of the included runs only",
        }
    )
    if missing or unreadable or partial:
        coverage["enumeration_complete"] = False
        gaps = coverage.setdefault("capture_gaps", [])
        if missing:
            gaps.append(
                f"{len(missing)} selected run(s) finished with no recorded "
                "turns at all ("
                + ", ".join(f"attempt {attempt}" for attempt in missing)
                + "), so they are in this population and in none of these "
                "counts; how much they would have contributed is unknown"
            )
        if unreadable:
            gaps.append(
                f"{len(unreadable)} selected run(s) name turns of which none "
                "could be read ("
                + ", ".join(_gap_phrase(row) for row in unreadable)
                + ")"
            )
        if partial:
            # The known turn counts are kept, because here they ARE known: the
            # run names its turns and some of them were read.
            gaps.append(
                f"{len(partial)} selected run(s) were read only in part ("
                + ", ".join(_gap_phrase(row) for row in partial)
                + "), so these counts are over the turns that were read"
            )
        if coverage.get("capture") == command_summary_module.COVERAGE_COMPLETE:
            coverage["capture"] = command_summary_module.COVERAGE_PARTIAL


def _run_outcomes(members: Sequence[SelectedRun]) -> dict[str, Any]:
    """Run-level tallies, kept apart from the dispatch-level ones.

    A run containing a failed dispatch is not a failed run, and a failed run
    is not a run whose every dispatch failed. Both facts are reported; neither
    is derived from the other.
    """
    by_outcome: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for member in members:
        outcome = member.row.get("outcome")
        by_outcome[str(outcome) if outcome is not None else "unrecorded"] = (
            by_outcome.get(str(outcome) if outcome is not None else "unrecorded", 0) + 1
        )
        status = member.row.get("execution_status")
        by_status[str(status) if status is not None else "unrecorded"] = (
            by_status.get(str(status) if status is not None else "unrecorded", 0) + 1
        )
    return {
        "runs": len(members),
        "by_outcome": by_outcome,
        "by_execution_status": by_status,
        "runs_with_failed_dispatches": sum(
            1
            for member in members
            if any(
                observation.response_success is False
                for observation in member.observations
            )
        ),
        "runs_without_readable_evidence": sum(
            1 for member in members if not member.readable_turns
        ),
    }


def _cost(
    members: Sequence[SelectedRun],
    usage: Mapping[str, Any],
    excluded: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The canonical recorded LLM cost of the selection.

    Summed from the per-run roll-ups by the same merge the per-run figures are
    made with, so nothing is re-derived and nothing is allocated: no share of
    a run's cost is attributed to a command or a turn, because the recorded
    evidence charges a provider response, not a dispatch.
    """
    recorded = dict(usage.get("cost") or {})
    with_cost = sum(1 for member in members if (member.cost.get("recorded") or 0))
    unreadable = sum(1 for member in members if not member.readable_turns)
    gaps = _evidence_gaps(members)
    return {
        "calls": int(recorded.get("calls") or 0),
        "recorded": int(recorded.get("recorded") or 0),
        "unrecorded": int(recorded.get("unrecorded") or 0),
        # None, never 0, when nothing recorded a cost: a zero here would be a
        # claim that these runs spent nothing.
        "total": recorded.get("total"),
        "members_with_recorded_cost": with_cost,
        "members_without_recorded_cost": len(members) - with_cost,
        # Every OBSERVED call having a price does not make this total the
        # selection's spend: a member whose evidence could not be read
        # recorded calls nobody here can see, and a complete-looking cost
        # beside an incomplete population is the specific claim this refuses.
        "members_without_readable_evidence": unreadable,
        # Read in part counts too: a run whose second turn could not be read
        # made calls in it that nothing here can see.
        "members_with_unreadable_turns": len(gaps["partial"]),
        "runs_excluded": len(excluded),
        "covers_observed_calls_only": True,
        # About the whole REQUEST, not just the runs that made it in: a total
        # called complete while two requested runs were excluded is a claim
        # about a population this never summed.
        "population_complete": (
            unreadable == 0 and not gaps["partial"] and not excluded
        ),
        "allocated_to_commands": False,
    }


def _build(
    *,
    experiment_id: str,
    task_id: str,
    source_id: Optional[str],
    store_id: Optional[str],
    requested: Sequence[int],
    duplicate_requests: int,
    members: Sequence[SelectedRun],
    excluded: Sequence[Mapping[str, Any]],
    unfinished: Sequence[int],
    not_recorded: Sequence[int],
    sealed: bool,
) -> dict[str, Any]:
    """The whole answer over runs that are already resolved and projected.

    Separate from resolution so the validation route derives its digest from
    ONE projection of the evidence rather than projecting each run twice.
    """
    observations: list[command_summary_module.CommandObservation] = []
    unreadable: list[str] = []
    turns_examined = 0
    for member in members:
        observations.extend(member.observations)
        unreadable.extend(member.unavailable)
        turns_examined += member.readable_turns

    summary = command_summary_module.summarize_commands(
        observations,
        requested_scope={
            "kind": METRIC_SCOPE,
            "experiment_id": experiment_id,
            "task_id": task_id,
            "store_id": store_id,
            "attempts": list(requested),
        },
        coverage={
            "unreadable_turns": unreadable,
            "turns_examined": turns_examined,
            "steps_outside_pass": 0,
        },
    )
    usage = merge_usage_rollups([member.usage for member in members])
    population = _population(
        requested=requested,
        members=members,
        excluded=excluded,
        unfinished=unfinished,
        not_recorded=not_recorded,
    )
    _apply_population_coverage(summary, population, members)
    member_rows = [member.as_member() for member in members]
    cost = _cost(members, usage, excluded)
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "source_id": source_id,
        "store_id": store_id,
        "sealed": bool(sealed),
        "scope": {
            "kind": METRIC_SCOPE,
            "requested_attempts": list(requested),
            "requested": len(requested),
            "duplicate_requests": int(duplicate_requests),
            "max_runs": MAX_SELECTED_RUNS,
            "whole_runs_only": True,
            "pass_scope": None,
        },
        "population": population,
        "members": member_rows,
        "member_count": len(member_rows),
        "run_outcomes": _run_outcomes(members),
        "command_summary": summary,
        "cost": cost,
        "usage": _usage_without_anchors(usage),
        "digest_basis": DIGEST_BASIS,
        "validation": "optimistic",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    # Membership, the metric values, the coverage and the projected evidence
    # behind them, in one identity. A caller echoes it back to
    # `validate_selection`; a change to any of them answers "stale".
    payload["evidence_digest"] = _digest(
        "ev-",
        {
            # Bound to the scope, so a selection of runs that recorded nothing
            # -- every member digest identical, every count zero -- is still a
            # different identity under a different task, experiment or source.
            "scope": {
                "experiment_id": experiment_id,
                "task_id": task_id,
                "source_id": source_id,
                "store_id": store_id,
                "sealed": bool(sealed),
                "requested_attempts": list(requested),
            },
            "members": [
                {
                    "attempt": row["attempt"],
                    "ref_id": row["ref_id"],
                    "evidence_digest": row["evidence_digest"],
                }
                for row in member_rows
            ],
            "population": population,
            "run_outcomes": payload["run_outcomes"],
            "totals": summary["totals"],
            "coverage": summary["coverage"],
            "groups": [
                {
                    "command_name": group["command_name"],
                    "dispatches": group["dispatches"],
                    "response_success": group["response_success"],
                    "duration": group["duration"],
                }
                for group in summary["groups"]
            ],
            "cost": cost,
        },
    )
    return payload


def aggregate_selected_runs(
    *,
    experiment_id: str,
    task_id: str,
    source_id: Optional[str],
    store_id: Optional[str],
    requested: Sequence[int],
    duplicate_requests: int,
    recorded_attempts: Sequence[int],
    candidate_rows: Sequence[Mapping[str, Any]],
    reader: Any,
    sealed: bool = False,
) -> dict[str, Any]:
    """One summary over exactly the runs a caller named.

    `candidate_rows` are the attempt rows of this task as the CALLER's own
    resolver produced them -- the live control's authorized source, or one
    sealed archive. This function resolves no source of its own and looks for
    no run: it is handed the population and the reader, which is what keeps
    the source boundary in the route that can enforce it.
    """
    members, excluded, unfinished, not_recorded = _resolve(
        requested=requested,
        recorded_attempts=recorded_attempts,
        candidate_rows=candidate_rows,
        reader=reader,
    )
    return _build(
        experiment_id=experiment_id,
        task_id=task_id,
        source_id=source_id,
        store_id=store_id,
        requested=requested,
        duplicate_requests=duplicate_requests,
        members=members,
        excluded=excluded,
        unfinished=unfinished,
        not_recorded=not_recorded,
        sealed=sealed,
    )


def validate_selection(
    *,
    experiment_id: str,
    task_id: str,
    source_id: Optional[str],
    store_id: Optional[str],
    requested: Sequence[int],
    recorded_attempts: Sequence[int],
    candidate_rows: Sequence[Mapping[str, Any]],
    reader: Any,
    expect: Optional[str] = None,
    expect_members: Optional[Mapping[int, str]] = None,
    sealed: bool = False,
) -> dict[str, Any]:
    """Re-project the named runs and say which no longer match.

    The point is the navigation that follows it. A reader who opens a
    contributor, or compares one member with the best run, must not be shown
    evidence that has changed since the totals were computed WITHOUT being
    told -- the drill-down would otherwise quietly stand in for the evidence
    the summary was made of.

    Optimistic, and says so: it detects a change rather than preventing one.
    Re-projection is bounded by the same selection limit, and a single-member
    check names one attempt and re-projects one run.
    """
    expected = {int(key): str(value) for key, value in (expect_members or {}).items()}
    unrelated = sorted(set(expected) - set(int(a) for a in requested))
    if unrelated:
        raise SelectedRunsError(
            "expect_member names attempt(s) "
            + ", ".join(str(attempt) for attempt in unrelated)
            + " that this request did not select; an expectation about a run "
            "outside the selection cannot be checked against it",
            reason="unrelated_expectation",
            attempts=unrelated,
        )
    members, excluded, unfinished, not_recorded = _resolve(
        requested=requested,
        recorded_attempts=recorded_attempts,
        candidate_rows=candidate_rows,
        reader=reader,
    )
    present = {member.attempt for member in members}
    rows: list[dict[str, Any]] = []
    changed: list[int] = []
    missing: list[int] = []
    not_compared: list[int] = []
    # The SAME projection these members were just checked against, summarized
    # once more rather than re-read: the result digest has to be the one the
    # aggregate route would publish for this selection, or "stale" would mean
    # two different things on the two routes. The scope is passed through for
    # the same reason -- a digest computed without it would not be the one.
    digest = _build(
        experiment_id=experiment_id,
        task_id=task_id,
        source_id=source_id,
        store_id=store_id,
        requested=requested,
        duplicate_requests=0,
        members=members,
        excluded=excluded,
        unfinished=unfinished,
        not_recorded=not_recorded,
        sealed=sealed,
    )["evidence_digest"]
    # A matching RESULT digest is a statement about every member, because
    # each member's digest is inside it. A differing one says something
    # changed without saying which member, so members nobody gave a baseline
    # for stay `not_compared` rather than being called unchanged.
    result_matches = expect is not None and digest == expect
    for member in members:
        want = expected.get(member.attempt)
        if want is not None:
            state = "unchanged" if want == member.digest else "changed"
        elif result_matches:
            state = "unchanged"
        else:
            state = "not_compared"
        if state == "changed":
            changed.append(member.attempt)
        elif state == "not_compared":
            not_compared.append(member.attempt)
        rows.append(
            {
                "attempt": member.attempt,
                "state": state,
                "evidence_digest": member.digest,
                "expected_digest": want,
                "ref_id": None if member.ref is None else member.ref.ref_id(),
            }
        )
    for attempt in requested:
        if attempt in present:
            continue
        # Asked about and not a member now: excluded as unfinished, or never
        # recorded at all. Either way it is not evidence anybody can open.
        missing.append(attempt)
        rows.append(
            {
                "attempt": attempt,
                "state": "missing",
                "evidence_digest": None,
                "expected_digest": expected.get(attempt),
                "reason": next(
                    (row["reason"] for row in excluded if row["attempt"] == attempt),
                    None,
                ),
            }
        )
    rows.sort(key=lambda row: row["attempt"])
    compared = bool(expect) or bool(expected)
    # None, never False, when there was no baseline: "nothing has changed" is
    # a claim, and a check with nothing to compare against has not made it.
    stale: Optional[bool] = (
        (bool(changed or missing) or (expect is not None and digest != expect))
        if (compared or missing)
        else None
    )
    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "source_id": source_id,
        "store_id": store_id,
        "attempts": list(requested),
        "expect": expect,
        "evidence_digest": digest,
        "members": rows,
        "changed": changed,
        "missing": missing,
        "not_compared": not_compared,
        "compared": compared,
        "stale": stale,
        "digest_basis": DIGEST_BASIS,
        "validation": "optimistic",
        # Said in the payload, not only in a docstring: a client that treats
        # "unchanged" as a guarantee has misread this answer.
        "guarantee": (
            "this detects a change in the recorded evidence behind the "
            "selection; it does not freeze that evidence, and a run can "
            "change between this answer and what is opened next"
        ),
        "unfinished": list(unfinished),
        "not_recorded": list(not_recorded),
    }


__all__ = [
    "DIGEST_BASIS",
    "MAX_ATTEMPT_PARAMS",
    "MAX_SELECTED_RUNS",
    "METRIC_SCOPE",
    "REASON_NOT_RECORDED",
    "REASON_UNFINISHED",
    "SelectedRun",
    "SelectedRunsError",
    "SelectionIncoherent",
    "SelectionTooLarge",
    "aggregate_selected_runs",
    "bound_attempts",
    "validate_selection",
]
