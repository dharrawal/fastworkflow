#!/usr/bin/env python3
"""Score the decision "which plan, if any, does this part get" on a held-out fixture, before and after
the one ``part_intent`` decision replaced the hand-built combinations (bd fix-o1lg, steps S0 to S3).

The fixture (``tests/planning_test_sets/library_plan_intent_heldout.json``) holds hand-written requests on
the bundled library example: requests a planning command serves, requests a plain execution command
answers (no plan may be given), requests nothing serves, and two-part requests whose parts are scored one
by one, as the split wrote them (the split itself is not scored here).

Two arms, on the same parts:

* ``before``: the rules as they stood in ``review_proposals`` and the turn's gate before the change
  (the fast-path predicate, the candidate list with the larger of the local model's and the planning
  matcher's score per name, the best-ranked pick with a planning command preferred, ``confident_pick``
  and the "none" vetoes). They are kept here as a frozen copy, written from the two inputs those rules
  read and that still exist: the local model's answers (``run_listing.head_candidates``) and the planning
  matcher's answer over the planning commands alone (``resolution.part_intent``, the intent loop's decision, since bd fix-zvme).
* ``after``: the product code, ``resolution.part_intent`` (exact name, then the detection chain over the
  execution and planning commands of the part's scope path) and what the callers do with its outcome.

Each arm is scored in the two modes that differ: ``no_agent`` (no agent model: only what the signals stand
behind is run, anything else is asked about or not run) and ``agent`` (an attended turn with an agent
model: a single confident part runs with no review, anything else is drafted for the review).

Model answers. The remote decision model is asked ONLY when ``--record`` is given, and every answer is
written to ``--answers`` (the planning matcher's distribution per part for ``before``, the chain matcher's
distribution per part for ``after``). Without ``--record`` the file is read and a part with no recorded
answer is scored with the local model alone (the report counts them). So both arms, and any rerun, read
the same answers. Nothing is written into the workflow folder: the workflow is copied (its trained
versions linked) and scored there. Never called from pytest.

Usage
-----
    score_plan_intent.py --answers plan_intent_answers.json --record --out plan_intent_report.json
    score_plan_intent.py --answers plan_intent_answers.json            # rerun on the recorded answers
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW = REPO / "fastworkflow" / "examples" / "library"
DEFAULT_FIXTURE = REPO / "tests" / "planning_test_sets" / "library_plan_intent_heldout.json"

PLAN, EXECUTION, AD_HOC = "plan", "execution", "ad_hoc"
#: The outcome columns, per part.
RIGHT_PLAN = "right plan"
WRONG_PLAN = "wrong plan"
PLAN_NOT_EXPECTED = "plan given, none expected"
NO_PLAN_EXPECTED_ONE = "none given, plan expected"
ASKED = "asked"
RIGHT_NO_PLAN = "right: no plan"
COLUMNS = (RIGHT_PLAN, WRONG_PLAN, PLAN_NOT_EXPECTED, NO_PLAN_EXPECTED_ONE, ASKED, RIGHT_NO_PLAN)


# ---------------------------------------------------------------- the workflow under score

def _copy(workflow: Path, parent: Path) -> Path:
    """A copy of *workflow* with its trained versions linked, not copied (they are large and only read)."""
    target = parent / workflow.name
    shutil.copytree(workflow, target, symlinks=True, ignore=shutil.ignore_patterns(
        "___workflow_contexts", "___convo_info", "__pycache__", "versions"))
    versions = workflow / "___command_info" / "versions"
    if versions.exists():
        os.symlink(versions, target / "___command_info" / "versions")
    return target


def _stand_on(app, context: str):
    """Put the library workflow where a case says it stands: the root record, or the book b1 inside it."""
    from fastworkflow.examples.library.application.library import Library

    if app.root is None:
        app.root = Library()
    app.record = app.root if context == "Library" else app.root.books["b1"]
    import fastworkflow

    assert app._current_command_context_name == context, context


# ---------------------------------------------------------------- recorded model answers

class Answers:
    """The remote decision model's answers, by arm and part: read from the file, and asked live (then
    written) only with *record*."""

    def __init__(self, path: Path, record: bool):
        self.path, self.record = path, record
        self.data = json.loads(path.read_text()) if path.exists() else {}
        self.data.setdefault("planning_match", {})
        self.data.setdefault("matcher", {})
        self.missing: set[tuple[str, str]] = set()
        self.asked = 0

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True, ensure_ascii=False))

    def get(self, arm: str, key: str, ask):
        """The recorded answer of *arm* for *key*; asked with *ask* and recorded under ``--record``; else None."""
        if key in self.data[arm]:
            return self.data[arm][key]
        if not self.record:
            self.missing.add((arm, key))
            return None
        self.asked += 1
        self.data[arm][key] = ask()
        self.save()
        return self.data[arm][key]


def _planning_match(answers: Answers, workflow, context: str, text: str) -> tuple[str | None, dict]:
    """``(the planning command the planning matcher picked, its distribution)``: what
    ``review_proposals.planning_match`` returned before the change. ``(None, {})`` with no answer."""
    from fastworkflow import workflow_agent

    def ask() -> dict:
        # The planning-command match is the intent loop's decision since bd fix-zvme.
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        intent = resolution.part_intent(workflow, text)
        planning = intent.outcome == resolution.MATCH and intent.kind == resolution.KIND_PLANNING
        return {"command": intent.command if planning else None, "distribution": intent.planning_scores()}

    answer = answers.get("planning_match", f"{context}|{text}", ask)
    return (None, {}) if answer is None else (answer["command"], dict(answer["distribution"]))


# ---------------------------------------------------------------- the arm "before" (a frozen copy)

PLAN_AMBIGUITY_CLEAR_MARGIN = 0.5
MAX_PLAN_ALTERNATIVES = 3
MATCH_THRESHOLD = 0.5
NONE_VETO_THRESHOLD = 0.9


def _decision(kind: str, name: str | None = None, *, asked: bool = False, alternatives=(), why: str = "") -> dict:
    return {"kind": kind, "name": name, "asked": asked, "alternatives": list(alternatives), "why": why}


def before(answers: Answers, workflow, context: str, text: str, mode: str) -> dict:
    """What the code before the change gave one part standing alone as a turn's only sub-task (``mode``
    ``no_agent`` or ``agent``), or as one of several (``no_agent_part`` / ``agent_part``: no fast path)."""
    from fastworkflow.planning import planning_commands
    from fastworkflow import run_listing, workflow_agent

    planning = set(planning_commands.resolve_field_sources(workflow.folderpath).planning_commands)
    core = workflow_agent.core_command_names()
    exact = run_listing.exact_command_name(workflow, text)
    heads = run_listing.head_candidates(workflow, text)
    jev, distribution = _planning_match(answers, workflow, context, text)
    none = distribution.get("none", 0.0)
    alone = not mode.endswith("_part")
    fast = (exact or (next(iter(heads.confident_labels)) if len(heads.confident_labels) == 1 else None)) if alone else None

    def bound(name: str, why: str) -> dict:
        return _decision(PLAN if name in planning else EXECUTION, name, why=why)

    if mode.startswith("no_agent"):
        if fast and fast in planning and not exact and jev is None and none >= MATCH_THRESHOLD:
            fast = None                                              # the gate's veto of a confident planning label
        if fast and fast in planning:
            # the fast path left the part to the run, which asked the planning matcher again
            return bound(jev, "fast path, planning matcher") if jev else _decision(AD_HOC, asked=True, why="fast path, no match")
        # confident_pick
        if exact and exact not in core:
            return bound(exact, "exact name")
        if jev and jev not in core:
            return bound(jev, "planning matcher")
        confident = [name for name in heads.confident_labels if name not in core]
        if len(confident) == 1:
            if confident[0] in planning and jev is None and none >= MATCH_THRESHOLD:
                return _decision(AD_HOC, asked=True, why="local model's plan vetoed by the matcher's none")
            return bound(confident[0], "local model")
        return _decision(AD_HOC, asked=True, why="no confident signal")

    if fast:
        # agent mode fast path: the run asks the planning matcher; with no plan the agent takes the part
        if jev:
            return bound(jev, "fast path, planning matcher")
        return (_decision(EXECUTION, fast, why="fast path, agent") if fast not in planning
                else _decision(AD_HOC, why="fast path, agent plans"))
    # the draft: the larger of the two scores per name, best-ranked with a planning command preferred
    scores = {**heads.execution_scores, **heads.planning_scores}
    for name, probability in distribution.items():
        scores[name] = max(scores.get(name, 0.0), probability)
    named = run_listing.gather(workflow, text)
    names = (run_listing.gather(workflow, text, distribution or None) | heads.labels
             | heads.planning_scores.keys() | heads.execution_scores.keys())
    candidates = [(name, scores.get(name, 0.0)) for name in sorted(names)
                  if name not in core and (name in named or scores.get(name, 0.0) >= run_listing.UNSURE_CANDIDATE_FLOOR)]
    ranked = sorted(candidates, key=lambda pair: -pair[1])[:3]
    choice = next((pair for pair in ranked if pair[0] in planning), None) or (ranked[0] if ranked else None)
    if choice is None:
        return bound(jev, "from scratch, planning matcher") if jev else _decision(AD_HOC, why="from scratch")
    alternatives: list[str] = []
    if choice[0] in planning:
        runners = sorted(((name, score) for name, score in candidates
                          if name in planning and name != choice[0] and score >= run_listing.UNSURE_CANDIDATE_FLOOR),
                         key=lambda pair: (-pair[1], pair[0]))
        if runners and choice[1] - runners[0][1] < PLAN_AMBIGUITY_CLEAR_MARGIN:
            alternatives = [name for name, _ in runners[:MAX_PLAN_ALTERNATIVES]]
    decision = bound(choice[0], "draft, best ranked")
    decision["alternatives"] = alternatives
    # what an unreviewed draft (nobody to answer) would have run: the 0.9 veto of a planning pick
    decision["unreviewed_vetoed"] = bool(choice[0] in planning and not exact and jev is None and none >= NONE_VETO_THRESHOLD)
    return decision


# ---------------------------------------------------------------- the arm "after" (the product code)

def after(answers: Answers, workflow, context: str, text: str, mode: str) -> dict:
    """What the product code gives the same part: ``resolution.part_intent`` and its callers' reading."""
    from fastworkflow import intent_detector, turn_cache
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    live = intent_detector._matcher_distribution

    def recorded(workflow_folderpath: str, scope_path: tuple, utterance: str, exclude=frozenset()) -> dict:
        # An answer over a list that leaves commands out is recorded apart from one over the full list.
        key = f"{context}|{utterance}" + (f"|without {','.join(sorted(exclude))}" if exclude else "")
        answer = answers.get("matcher", key, lambda: live(workflow_folderpath, scope_path, utterance, exclude=exclude))
        return answer if answer is not None else {"scores": {}, "error": "no recorded answer", "calls": 0}

    intent_detector._matcher_distribution = recorded
    try:
        turn_cache.begin_turn()
        intent = resolution.part_intent(workflow, text)
    finally:
        intent_detector._matcher_distribution = live
    alone = not mode.endswith("_part")
    why = f"{intent.outcome} ({intent.decided_by})"
    if alone and resolution.is_fast_path(intent):
        return _decision(PLAN if intent.kind == resolution.KIND_PLANNING else EXECUTION, intent.command,
                         why=f"fast path, {why}")
    # The turn resolves every part before its review (bd fix-o1lg, S4/S5): a match is bound, an ambiguity is ASKED in
    # both modes (the review offers no alternatives), and a part nothing serves is asked about with no agent model
    # (the plan-choice question) and left to the agent with one.
    if intent.outcome == resolution.MATCH:
        return _decision(PLAN if intent.kind == resolution.KIND_PLANNING else EXECUTION, intent.command, why=why)
    if intent.outcome == resolution.AMBIGUOUS:
        return _decision(AD_HOC, asked=True, why=why)
    return _decision(AD_HOC, asked=mode.startswith("no_agent"), why=why)


# ---------------------------------------------------------------- scoring

def column(part: dict, decision: dict) -> str:
    """The outcome column of one decision for one part."""
    if part["expect"] == PLAN:
        if decision["kind"] == PLAN:
            return RIGHT_PLAN if decision["name"] == part["expected"] else WRONG_PLAN
        return ASKED if decision["asked"] else NO_PLAN_EXPECTED_ONE
    if decision["kind"] == PLAN:
        return PLAN_NOT_EXPECTED
    return ASKED if decision["asked"] else RIGHT_NO_PLAN


def parts_of(fixture: dict) -> list[dict]:
    """Every scored part: the single requests (a turn of one part) and each part of the two-part requests."""
    parts = [{**case, "alone": True} for case in fixture["cases"]]
    for request in fixture["two_part"]:
        for number, part in enumerate(request["parts"], 1):
            parts.append({**part, "id": f"{request['id']}.{number}", "context": request["context"], "alone": False,
                          "group": "two_part"})
    return parts


def tally(rows: list[dict], arm: str, mode: str) -> dict:
    counts = {name: 0 for name in COLUMNS}
    extra = {"right execution command": 0, "execution parts": 0, "alternatives shown": 0, "n": 0}
    for row in rows:
        decision = row[arm][mode]
        counts[column(row, decision)] += 1
        extra["n"] += 1
        extra["alternatives shown"] += bool(decision["alternatives"])
        if row["expect"] == EXECUTION:
            extra["execution parts"] += 1
            extra["right execution command"] += decision["kind"] == EXECUTION and decision["name"] == row["expected"]
    return {**counts, **extra}


GROUPS = ("plan", "execution", "resembles_plan", "none", "two_part", "all")


def table(report: dict) -> str:
    """One row per group, mode and arm: the outcome columns, then the plain commands named right."""
    head = ["group", "mode", "arm", "n", *COLUMNS, "right execution command", "alternatives shown"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    arms = list(report)
    for group in GROUPS:
        for mode in ("no_agent", "agent"):
            for arm in arms:
                counts = report[arm][mode].get(group)
                if not counts or not counts["n"]:
                    continue
                executions = (f"{counts['right execution command']}/{counts['execution parts']}"
                              if counts["execution parts"] else "-")
                cells = [group, mode, arm, str(counts["n"]), *(str(counts[name]) for name in COLUMNS),
                         executions, str(counts["alternatives shown"])]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workflow", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--env", help="the workflow env file (default: <workflow>/fastworkflow.env)")
    parser.add_argument("--passwords", help="the passwords env file (default: <workflow>/fastworkflow.passwords.env)")
    parser.add_argument("--answers", required=True, help="the file of recorded model answers (read, and written with --record)")
    parser.add_argument("--record", action="store_true", help="ask the remote decision model for parts with no recorded answer")
    parser.add_argument("--arms", default="before,after")
    parser.add_argument("--out", help="write the per-part decisions and the counts here")
    args = parser.parse_args(argv)

    from dotenv import dotenv_values

    import fastworkflow

    source = Path(args.workflow).resolve()
    env = {**dotenv_values(args.env or source / "fastworkflow.env"),
           **dotenv_values(args.passwords or source / "fastworkflow.passwords.env")}
    fixture = json.loads(Path(args.fixture).read_text())
    answers = Answers(Path(args.answers), args.record)
    arms = {"before": before, "after": after}
    chosen = [name for name in args.arms.split(",") if name]
    with tempfile.TemporaryDirectory(prefix="score_plan_intent_") as scratch:
        env["SPEEDDICT_FOLDERNAME"] = os.path.join(scratch, "speeddict")
        if not args.record:
            env.pop("JEV_API_KEY", None)                 # a rerun reads the file and can never call out
        fastworkflow.init(env_vars=env)
        from fastworkflow import intent_detector, turn_cache

        if args.record and not intent_detector.matcher_available():
            print("JEV_API_KEY is not set in the env files given: nothing can be recorded.", file=sys.stderr)
            return 2
        folder = _copy(source, Path(scratch))
        app = fastworkflow.Workflow.create(workflow_folderpath=str(folder), workflow_id_str="score-plan-intent")
        rows = parts_of(fixture)
        for row in rows:
            _stand_on(app, row["context"])
            for arm in chosen:
                modes = ("no_agent", "agent") if row["alone"] else ("no_agent_part", "agent_part")
                row[arm] = {}
                for mode in modes:
                    turn_cache.begin_turn()
                    row[arm][mode.removesuffix("_part")] = arms[arm](answers, app, row["context"], row["utterance"], mode)
    report = {arm: {mode: {group: tally([row for row in rows if group == "all" or row["group"] == group], arm, mode)
                           for group in GROUPS}
                    for mode in ("no_agent", "agent")}
              for arm in chosen}
    print(table(report))
    unanswered = sorted(answers.missing)
    print(f"\nparts: {len(rows)}; model answers asked now: {answers.asked}; "
          f"answers missing (scored with the local model alone): {len(unanswered)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"report": report, "rows": rows, "unanswered": unanswered},
                                             indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
