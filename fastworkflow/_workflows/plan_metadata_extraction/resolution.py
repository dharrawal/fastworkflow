"""Which command or plan a part of a turn gets: ONE decision (bd fix-o1lg, steps S2 and S3).

``part_intent`` decides it for every caller (the turn's gate, the draft the review shows, a review's look_up and add,
the part's run, the no-agent plan-choice question and its list of plans), so no caller combines the local model's
and the remote decision model's answers itself any more. In order:

1. a plan the client pinned (``use_plan``): that plan, nothing is detected;
2. ``intent_loop.resolve``, the loop the deterministic path (the CME) decides with too: an exact command name, the
   foreign-name guard, a fuzzy name, the local model over the whole walk, then the detection chain with the remote
   decision model (see ``intent_loop`` for the layers). The only difference of this path is that a plan does not run
   the framework's core commands (``core`` False).

The answer is memoised for the turn (``turn_cache``): the gate, the draft and the run read one decision, and the
remote model is called at most once per part per turn (its own call is memoised there too).

The callers keep their jobs: ``is_fast_path`` says which part runs with no review (one part, an exact name or a
confident local model, never a data change only a model chose), ``draft_choice`` what a draft binds (a confident match
always; the best candidate of an ambiguity only where somebody reviews the draft), and ``PartIntent.planning_scores``
orders the plans an ad hoc part lists ("closest" first).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Optional

from fastworkflow import intent_loop, turn_cache, waiting
from fastworkflow.intent_loop import (  # noqa: F401 - the names this module's callers read from here
    AMBIGUOUS, BY_CLASSIFIER, BY_EXACT_NAME, BY_FUZZY_NAME, BY_MATCHER, BY_NOTHING, BY_PINNED, BY_QUESTION,
    FAST_PATH_DECIDERS, KIND_EXECUTION, KIND_PLANNING, MATCH, MAX_ALTERNATIVES, NONE, Ranked)

#: The decision for one part is the intent loop's: ONE structure, no second result type.
PartIntent = intent_loop.Intent


def part_intent(workflow, text: str, *, pinned: Optional[str] = None,
                exclude: Iterable[str] = (), part: Optional[int] = None) -> PartIntent:
    """Which command or plan the part *text* gets, from where *workflow* stands (see the module docstring).

    *pinned* is a plan the client chose (``use_plan``): it is the answer and nothing is detected. *exclude* leaves
    commands out (a review's look_up re-matches a write command's part with the write commands left out): a match on
    one of them becomes an ambiguity among the rest, or none. The decision is the intent loop's with no core command (a
    plan does not run them). Memoised for the turn. Never raises for a model that
    cannot be reached: the chain then stands on the local model.

    The decision is written in ONE ``fw.intent`` span, by the writer the CME's intent stage uses
    (``intent_detection.traced_decision``), when it is made (a memo miss), never on a memo read; *part* is the part's
    1-based number in the turn, recorded on the span when the caller knows it, and the span is then a child of that
    part's ``fw.resolve.part`` span while the turn is in its resolve stage (``tracing.part_span``). A pinned plan is not detected, so it
    writes none.
    """
    if pinned:
        planning, _ = intent_loop._kinds(workflow)
        kind = KIND_PLANNING if pinned in planning else KIND_EXECUTION
        return PartIntent(MATCH, pinned, kind, (Ranked(pinned, kind, 1.0),), BY_PINNED)
    key = ("part_intent", workflow.folderpath, workflow._current_command_context_name, text)
    intent = turn_cache.memo(key, lambda: _decided(workflow, text, part))
    excluded = set(exclude)
    if not excluded:
        return intent
    rest = tuple(each for each in intent.candidates if each.name not in excluded)
    if intent.outcome == MATCH and intent.command not in excluded:
        return PartIntent(MATCH, intent.command, intent.kind, rest, intent.decided_by)
    names = tuple(each for each in (intent.ambiguous or tuple(c.name for c in intent.candidates))
                  if each not in excluded)
    if intent.outcome in (MATCH, AMBIGUOUS) and names:
        return PartIntent(AMBIGUOUS, candidates=rest, decided_by=BY_QUESTION, ambiguous=names[:MAX_ALTERNATIVES])
    return PartIntent(NONE, candidates=rest, decided_by=intent.decided_by)


def _decided(workflow, text: str, part: Optional[int]) -> PartIntent:
    """The intent loop's decision for *text* (no core command), in its span: the CME's writer, called from here, under
    the part's span."""
    from fastworkflow import tracing
    from fastworkflow._workflows.command_metadata_extraction import intent_detection

    with tracing.part_span(tracing.current_host(), tracing.SPAN_RESOLVE_PART, part):
        return intent_detection.traced_decision(workflow, text, core=False, part=part)[0]


how_settled = intent_loop.how_settled


def record_part(workflow, text: str, command: str, how: str, evidence: Optional[dict] = None,
                rejected: Optional[str] = None) -> None:
    """Record a part's settled command (`intent_learning.record`; never raises): the intent loop's one writer."""
    intent_loop.record(workflow, text, command, how, evidence, rejected)


def is_fast_path(intent: PartIntent, workflow) -> bool:
    """Whether a turn's ONLY part runs with no review: it names a command, or the local model is confident of an
    execution command that does not change data. A planning command the local model alone chose is never a fast path
    (``_second_opinion``), and neither is a command that changes data the local model alone chose
    or carries an approval mark (``intent_loop.unconfirmed_pick``), unless its approval question will be asked anyway:
    that question is its yes.
    The part runs through the agent's command tool, so the approval is read for an agent's call."""
    from fastworkflow.approval_binding import Origin

    return (intent.outcome == MATCH and intent.decided_by in FAST_PATH_DECIDERS
            and (intent.decided_by == BY_EXACT_NAME or intent.kind != KIND_PLANNING)
            and not intent_loop.unconfirmed_pick(workflow, intent, Origin.AGENT))


def draft_choice(intent: PartIntent) -> tuple[Optional[str], Optional[str]]:
    """``(command, kind)`` a draft binds for a part, or ``(None, None)``: only a match is bound. An ambiguity is asked
    about before the review (``step``), never bound to a guess, and a part nothing serves is ad hoc."""
    if intent.outcome == MATCH:
        return intent.command, intent.kind
    return None, None


# ======================================================================================================================
# The pending resolution of a turn's parts (bd fix-o1lg, steps S4 and S5)
# ======================================================================================================================
#
# Every part is resolved, in order, before the turn's one review: which command or plan (asked only when the decision
# is an ambiguity, or with no agent model when nothing serves the part and the turn can ask), then the values a bound
# planning command still lacks (asked only when it lacks one; a value an earlier part produces is never asked for).
# Each question can be cancelled, which drops that part; the later parts go on. The progress is plain data, saved with
# the session under ONE key (``plan_resolution``). A question that is out is NOT part of it: it is the turn's one waiting
# record (``waiting.Waiting``, kind ``intent`` or ``parameters``, ``docs/pending_state_design.md``), which these functions
# put and take through ``Turn.port``. The execution context keeps the progress and calls ``start`` / ``step`` /
# ``take_reply`` / ``finish``; nothing here suspends a turn or calls the agent.

#: The stages of the part being resolved.
STAGE_INTENT = "intent"
STAGE_VALUES = "values"
#: A part's state.
PART_OPEN = "open"
PART_RESOLVED = "resolved"
PART_DROPPED = "dropped"


@dataclass
class PlanResolution:
    """A turn's resolution progress, as plain data (saved with the session under ``plan_resolution``).

    ``parts``: one dict per part of the turn, in order: ``text``, the decision's ``outcome``, its ``ambiguous``
    candidates, the ``planning_scores`` that order an ad hoc part's plans, ``state`` (open, resolved, dropped) and
    ``chosen`` (the user answered which command it gets). ``index`` and ``stage`` say where the resolution stands.
    ``pinned``: the plan the client chose (``use_plan``); its parameter question confirms every value and is that
    turn's review. ``asked`` records the which-command questions put (for the turn span). The question that is out
    is not here: it is the turn's waiting record (``Turn.port``).
    """

    parts: list[dict] = field(default_factory=list)
    index: int = 0
    stage: str = STAGE_INTENT
    pinned: Optional[str] = None
    asked: list[dict] = field(default_factory=list)

    def to_state(self) -> dict:
        import json

        # A copy: the choices also sit in a not-run entry of the turn's outputs, and a state holds a container once.
        return json.loads(json.dumps({each.name: getattr(self, each.name) for each in fields(self)}))

    @classmethod
    def from_state(cls, state: Any) -> Optional["PlanResolution"]:
        """The resolution a saved state holds, or None. Unknown keys are ignored."""
        if not isinstance(state, dict):
            return None
        known = {each.name for each in fields(cls)}
        return cls(**{key: value for key, value in state.items() if key in known})


def prompt(held) -> tuple[str, dict]:
    """``(question_kind, artifacts)`` of the question *held* (a plan-owned waiting record): ``plan_choice`` (artifact
    ``plan_choice``) or ``planning_parameter`` (artifact ``parameter_request``); ``("", {})`` when it has none."""
    if held.kind == waiting.KIND_INTENT:
        artifacts = intent_loop.prompt_artifacts(held.payload["record"])
        return artifacts.pop("question_kind"), artifacts
    if request := (held.payload.get("loop") or {}).get("request"):
        return "planning_parameter", {"parameter_request": dict(request)}
    return "", {}


@dataclass(frozen=True)
class Turn:
    """What the stage logic needs from the turn: the workflow, whether an agent model is set, whether anybody can
    answer a question now (attended: ``waiting.can_ask``), the planner's LM for drafting a picked command's
    inputs, and the port of the turn's waiting record (where a question is put and taken from)."""

    workflow: Any
    no_agent: bool
    can_ask: bool
    planner_lm: Any = None
    port: Any = None


def start(workflow, subtasks: list[str], *, pinned: Optional[str] = None) -> PlanResolution:
    """A resolution of *subtasks*, each part's decision read from ``part_intent`` (memoised for the turn)."""
    parts = []
    for text in subtasks:
        intent = part_intent(workflow, str(text), pinned=pinned, part=len(parts) + 1)
        parts.append({"text": str(text), "outcome": intent.outcome, "ambiguous": list(intent.ambiguous),
                      "planning_scores": intent.planning_scores(), "state": PART_OPEN, "chosen": False,
                      "decided_by": intent.decided_by, "evidence": dict(intent.evidence)})
    return PlanResolution(parts=parts, pinned=pinned)


def _with_item(plan, index: int, item):
    items = list(plan.items)
    items[index] = item
    return plan.model_copy(update={"items": items})


def _from_earlier_parts(item) -> set[str]:
    """The inputs of *item* an earlier part produces ("from sub-task N"): never asked for."""
    return {each.name for each in item.inputs if each.from_subtask is not None and each.name}


def _put_choice(turn: Turn, resolution: PlanResolution, record: dict) -> None:
    """Make the which-command question *record* the turn's one waiting record."""
    turn.port.put(waiting.Waiting(waiting.KIND_INTENT, question=str(record.get("question") or ""),
                                  payload={"record": record, "part": resolution.index}))


def _ask_choice(resolution: PlanResolution, turn: Turn, record: dict) -> str:
    question = intent_loop.ask(record)
    _put_choice(turn, resolution, record)
    return question


def _intent_question(resolution: PlanResolution, plan, turn: Turn) -> Optional[str]:
    """The which-command question of the part at ``index``, or None when its command is settled or nobody asks.

    An ambiguity is asked in both modes, its candidates (execution and planning commands) as the choices. A part
    nothing serves is asked about only with no agent model, the planning commands available here as the choices
    (``no_agent_mode.available_plans``, the closest first); with an agent model the agent plans it."""
    from fastworkflow import no_agent_mode

    part = resolution.parts[resolution.index]
    item = plan.items[resolution.index]
    if not turn.can_ask or resolution.pinned or item.means.kind != "from_scratch":
        return None
    if part["outcome"] == AMBIGUOUS and part["ambiguous"]:
        choices = intent_loop.choices_of(turn.workflow, part["ambiguous"])
        ambiguous = True
    elif part["outcome"] != MATCH and turn.no_agent:
        choices = no_agent_mode.available_plans(turn.workflow, part["planning_scores"])
        ambiguous = False
    else:
        return None
    if not choices:
        return None
    record = intent_loop.open_question(
        part["text"], choices, sub_task=resolution.index + 1 if len(resolution.parts) > 1 else None,
        ambiguous=ambiguous, origin=resolution.index)
    return _ask_choice(resolution, turn, record)


def _bind_only_candidate(resolution: PlanResolution, plan, turn: Turn):
    """An ambiguity with exactly one candidate (a confident local plan the remote model did not confirm, with nothing
    else to offer) is not a question: the candidate is bound as drafted and the turn's ONE review is its confirmation.
    The part is NOT marked ``chosen``, because nobody chose it, so the review is shown. Where the turn cannot ask (and
    so cannot review) the part stays as it was."""
    part = resolution.parts[resolution.index]
    item = plan.items[resolution.index]
    if (not turn.can_ask or resolution.pinned or item.means.kind != "from_scratch"
            or part["outcome"] != AMBIGUOUS or len(part["ambiguous"]) != 1):
        return plan
    name = part["ambiguous"][0]
    planning, _ = intent_loop._kinds(turn.workflow)
    part["outcome"] = MATCH
    return _bind(plan, resolution.index, name, intent_loop._kind_of(name, planning), turn)


def _values_command(turn: Turn, item):
    """``(command, kind)`` the values of a part bound to *item*'s means are asked through, or None: a planning command,
    or (with no agent model, where nothing else would find a value the user did not give) an execution command stated as
    a command with parameters (``planning_commands.execution_command``). With an agent model the agent finds an
    execution command's missing value."""
    from fastworkflow import parameter_clarification as loops, planning_commands, review_proposals

    if item.means.kind == review_proposals.MEANS_PLANNING:
        return planning_commands.planning_command(turn.workflow.folderpath, item.means.name), loops.KIND_PLANNING
    if item.means.kind == review_proposals.MEANS_PLAIN and turn.no_agent:
        return planning_commands.execution_command(turn.workflow.folderpath, item.means.name), loops.KIND_EXECUTION
    return None, loops.KIND_PLANNING


def _values_question(resolution: PlanResolution, plan, turn: Turn) -> Optional[str]:
    """The parameter question of the part at ``index``, or None when it lacks nothing (or nobody asks).

    A part bound to a planning command is asked about, and only for the arguments it lacks
    (``planning_commands.argument_gaps`` on its drafted values; the ones an earlier part produces are never open). So is a
    part bound to an execution command when no agent model is set (``_values_command``): the same loop, the same
    reader, the same asking rule. A pinned plan is always asked (every parameter shown, ``yes`` runs). With an agent model,
    a part whose inputs were not drafted (``FW_DRAFT_ARGUMENTS`` off) is not asked: the run reads its values."""
    from fastworkflow import parameter_clarification as loops, planning_commands, review_proposals

    item = plan.items[resolution.index]
    if not turn.can_ask:
        return None
    command, kind = _values_command(turn, item)
    if command is None or not command.params:
        return None
    confirm = bool(resolution.pinned)
    if not (confirm or turn.no_agent or item.drafted_means == item.means.name):
        return None
    raw = {p.name: next((each.value for each in item.inputs if each.name == p.name and each.value is not None), None)
           for p in command.params}
    with intent_loop.on_owner_record(turn.workflow, command.name):   # its validators read the record it will run on
        gaps = planning_commands.argument_gaps(command, raw, item.goal, workflow=turn.workflow).from_earlier(
            _from_earlier_parts(item))
    if item.check_screened and not gaps.screened:
        # the draft refused an extracted value as not given (a pronoun, a word of the workflow's vocabulary): the question
        # says why, as the `/` question does
        gaps = dataclasses.replace(gaps, screened=tuple(
            (name, text) for name, text in item.check_screened.items() if name in gaps.missing))
    if item.check_notes and not gaps.left_empty:
        # the draft left an optional value out by the duplicate rule: the question says so, as the review item does
        gaps = dataclasses.replace(gaps, left_empty=tuple(item.check_notes.values()))
    if not gaps.names and not confirm:
        return None
    description = review_proposals.planning_descriptions(turn.workflow.folderpath).get(command.name, "") if confirm else ""
    shown = {name: value for name, value in raw.items() if value is not None and name not in gaps.names}
    loop, outcome = loops.open_loop(
        command, gaps, request_text=item.goal, origin=resolution.index, confirm=confirm, description=description,
        values=shown if confirm else None, kind=kind)
    turn.port.put(loops.waiting_record(loop, loops.AFTER_RESOLVE_NEXT, resolution.index))
    return outcome.question


def _refused_without_field(resolution: PlanResolution, plan, turn: Turn) -> Optional[str]:
    """The refusal of the draft of the part at ``index`` when it names no argument (``SubTask.refused_without_field``), or
    None: a plan the client pinned shows every parameter and is read as it always was."""
    if resolution.pinned or not turn.can_ask:
        return None
    return plan.items[resolution.index].refused_without_field or None


def step(resolution: PlanResolution, plan, turn: Turn):
    """``(plan, question)``: the resolution carried on from where it stands to its next question, or ``(plan, None)``
    when every part is resolved (or dropped). Parts that need no question are settled on the way."""
    while resolution.index < len(resolution.parts):
        part = resolution.parts[resolution.index]
        if part["state"] != PART_OPEN:
            resolution.index, resolution.stage = resolution.index + 1, STAGE_INTENT
            continue
        if resolution.stage == STAGE_INTENT:
            plan = _bind_only_candidate(resolution, plan, turn)
            if (question := _intent_question(resolution, plan, turn)) is not None:
                return plan, question
            resolution.stage = STAGE_VALUES
        if (refusal := _refused_without_field(resolution, plan, turn)) is not None:
            # A refusal that names no argument is not asked (nothing is missing to give). With no agent model the part
            # ends as not run, saying why; with one the run hands it to the text planner or the agent.
            if turn.no_agent:
                plan = _drop(resolution, plan, (f"it cannot run as drafted ({refusal}), and the agent model that would "
                                                "take over is not set (LLM_AGENT)"))
                continue
        elif (question := _values_question(resolution, plan, turn)) is not None:
            return plan, question
        part["state"] = PART_RESOLVED
        resolution.index, resolution.stage = resolution.index + 1, STAGE_INTENT
    return plan, None


def _drop(resolution: PlanResolution, plan, reason: str):
    """Drop the part at ``index`` (*reason* says why): it is not run, and a later part's input that it would have
    produced is no longer "from" it (the value is then the user's to give, and asked for when that part is reached)."""
    number = resolution.index + 1
    resolution.parts[resolution.index]["state"] = PART_DROPPED
    resolution.parts[resolution.index]["dropped_reason"] = reason
    for later in range(resolution.index + 1, len(plan.items)):
        item = plan.items[later]
        kept = [each for each in item.inputs if each.from_subtask != number]
        if len(kept) != len(item.inputs):
            plan = _with_item(plan, later, item.model_copy(update={"inputs": kept}))
    return plan


def _overlay(plan, index: int, given: dict, workflow):
    """The values the user's replies gave put on the part's item, as its inputs: the user's own (not drafted). Its check
    is worked out again from the inputs (``review_proposals.rechecked``), so the review shows what they now earn."""
    from fastworkflow import review_proposals

    if not given:
        return plan
    item = plan.items[index]
    inputs = [*(each for each in item.inputs if each.name not in given),
              *(review_proposals.SubTaskInput(name=name, value=str(value), drafted=False)
                for name, value in given.items())]
    return _with_item(plan, index, review_proposals.rechecked(workflow, item.model_copy(update={"inputs": inputs})))


def _bind(plan, index: int, name: str, kind: str, turn: Turn):
    """The part at *index* bound to the command the user picked, its inputs drafted from its text."""
    from fastworkflow import review_proposals

    item = plan.items[index]
    means = review_proposals.Means(kind=review_proposals.MEANS_PLANNING if kind == KIND_PLANNING
                                   else review_proposals.MEANS_PLAIN, name=name)
    bound = review_proposals.drafted_item(turn.workflow, review_proposals.SubTask(goal=item.goal, means=means),
                                          item.goal, list(plan.items[:index]), planner_lm=turn.planner_lm, part=index + 1)
    if item.parts:
        bound = bound.model_copy(update={"parts": list(item.parts), "parts_fingerprint": ""})
    return _with_item(plan, index, bound)


def _take_choice(resolution: PlanResolution, held, reply: str, plan, turn: Turn):
    """The reply to the which-command question, read by the intent loop's one reader (``intent_loop.answer``). Only the
    pick is read (words after a picked name are not its values, on this path and the CME's): the part's values are drafted
    from its own text and asked for after."""
    from fastworkflow import planning_commands

    record = held.payload["record"]
    names = [each["name"] for each in record["choices"]]
    # The phrases of "you misunderstood" (none of these) and "abort" (cancel) read as on the CME (one source); the
    # listing of commands (``what_can_i_do``) is the CME's alone.
    reply_read = intent_loop.answer(record, reply, aliases=intent_loop.core_aliases(listing=False))
    if reply_read.kind == intent_loop.ASK_AGAIN:
        _put_choice(turn, resolution, record)
        return plan, reply_read.question
    # The second reply that cannot be read ends the question like "none": the part is not bound (it comes back as a
    # not-run entry) and the later parts go on.
    kind = "unclear" if reply_read.kind == intent_loop.EXHAUSTED else reply_read.kind
    name = reply_read.name
    part = resolution.parts[resolution.index]
    resolution.asked.append({"origin": resolution.index, "alternatives": names, "picked": name if kind == "pick" else None,
                             "reply": kind})
    ambiguous = bool(record.get("ambiguous"))
    if kind == intent_loop.ABORT:
        what = "command" if ambiguous else "planning command"
        return _drop(resolution, plan, f"it was declined when asked which {what} should run it"), None
    # Only a pick settles the part's command: after ``none`` or two unreadable replies it stays ad hoc and unchosen, so
    # a one-part turn is reviewed like any ad hoc part (or comes back not run, with no agent model).
    part["chosen"] = kind == intent_loop.PICK
    if kind == intent_loop.PICK:
        planning = set(planning_commands.resolve_field_sources(turn.workflow.folderpath).planning_commands)
        plan = _bind(plan, resolution.index, name, KIND_PLANNING if name in planning else KIND_EXECUTION, turn)
    resolution.stage = STAGE_VALUES
    return plan, None


#: Why a part ended when the reply to its value question asked for something else (it is not run as a new request).
DIFFERENT_REQUEST_REASON = ("the reply to its question asked for something else; a reply answers its question and "
                            "nothing else, so it was not run as a new request: send it as its own message")


def ask_misunderstood(resolution: PlanResolution, plan, turn: Turn):
    """The reply said the command was not the one meant: the part goes back to the which-command question (every command
    of the part's scope the plan can run, the intent loop's misunderstanding question). The part is bound to nothing
    until the user picks; ``none`` leaves it ad hoc and unchosen (the deterministic path aborts instead)."""
    from fastworkflow import review_proposals

    part = resolution.parts[resolution.index]
    item = plan.items[resolution.index]
    plan = _with_item(plan, resolution.index, item.model_copy(update={
        "means": review_proposals.Means(kind=review_proposals.MEANS_FROM_SCRATCH), "inputs": [], "drafted_means": ""}))
    record = intent_loop.open_question(
        part["text"], intent_loop.command_choices(turn.workflow, core=False),
        sub_task=resolution.index + 1 if len(resolution.parts) > 1 else None, ambiguous=False, misunderstood=True,
        origin=resolution.index)
    part["outcome"], part["ambiguous"] = NONE, []
    resolution.stage = STAGE_INTENT
    return plan, _ask_choice(resolution, turn, record)


def ask_misunderstood_running_part(text: str, turn: Turn):
    """``(resolution, plan, question)`` for the one part of an unreviewed turn whose value question was answered with
    "you misunderstood" while the part ran (nothing of it ran): a one-part resolution of *text* and its from-scratch
    plan, built with no model call, put back to the which-command question (``ask_misunderstood``). The reply to it is
    read by ``take_reply`` like any which-command reply, then the part's values, then the run."""
    from fastworkflow import review_proposals

    resolution = PlanResolution(parts=[{
        "text": text, "outcome": NONE, "ambiguous": [], "planning_scores": {}, "state": PART_OPEN, "chosen": False,
        "decided_by": BY_QUESTION, "evidence": {}}])
    plan = review_proposals.SubTaskProposal(items=[review_proposals.SubTask(goal=text)])
    plan, question = ask_misunderstood(resolution, plan, turn)
    return resolution, plan, question


def _take_values(resolution: PlanResolution, held, reply: str, plan, turn: Turn):
    from fastworkflow import parameter_clarification as loops, planning_commands, review_proposals

    loop = loops.ParameterLoop.from_state(held.payload["loop"])
    item = plan.items[resolution.index]
    command = (planning_commands.execution_command(turn.workflow.folderpath, loop.command)
               if loop.kind == loops.KIND_EXECUTION
               else planning_commands.planning_command(turn.workflow.folderpath, loop.command))
    if command is None:
        return plan, None
    extract = (None if loop.confirm else
               (lambda text: planning_commands.extracted_arguments(command, text, turn.workflow)))
    description = (review_proposals.planning_descriptions(turn.workflow.folderpath).get(command.name, "")
                   if loop.confirm else "")
    with intent_loop.on_owner_record(turn.workflow, command.name):   # its validators read the record it will run on
        outcome = loops.advance(loop, reply, command, extract=extract, description=description,
                                not_asked=_from_earlier_parts(item), live=True, waited=loops.waited_of(held))
    if outcome.kind in (loops.ASK_AGAIN, loops.WAIT):
        # The asking rule (``parameter_clarification``): asked again while each reply makes progress, and once more after
        # one that does not; in every session. The first reply in a row that asked for a moment leaves the question out
        # and counts nothing; each further one counts as a reply with no progress (``waited`` is in the record).
        turn.port.put(loops.waiting_record(loop, loops.AFTER_RESOLVE_NEXT, resolution.index, waited=outcome.waited))
        return plan, outcome.question
    if outcome.kind == loops.MISUNDERSTOOD:
        return ask_misunderstood(resolution, plan, turn)
    if outcome.kind == loops.DIFFERENT:
        # A reply answers its question and nothing else: it is not run as a new request, so the part is set aside and
        # the answer says why.
        return _drop(resolution, plan, DIFFERENT_REQUEST_REASON), None
    if outcome.kind == loops.DECLINED:
        missing = f" while its arguments {', '.join(outcome.missing)} were still missing" if outcome.missing else ""
        return _drop(resolution, plan, f"it was declined{missing}"), None
    if outcome.kind == loops.EXHAUSTED and loop.confirm:
        # A pinned plan's question is that turn's review: values it never confirmed are not run.
        names = outcome.missing
        return _drop(resolution, plan, (
            f"it cannot run as planned (missing required argument{'s' if len(names) > 1 else ''} {', '.join(names)}), "
            "and no usable reply was given" if names else "its parameters were not confirmed: no usable reply was given")
        ), None
    if outcome.kind == loops.EXHAUSTED and turn.no_agent:
        # No agent model and no usable reply: the part cannot run, and a review is shown only for what can, so it is
        # dropped (not run, naming exactly what is missing) like a cancelled one.
        return _drop(resolution, plan, f"it cannot run as planned ({outcome.describe}), and no usable reply was given"
                     ), None
    # Run with what was given: the complete loop, or (the one hand-over, with an agent model) the stop after the second
    # reply in a row with no progress, which leaves the rest to the agent as the refused call did.
    resolution.parts[resolution.index]["state"] = PART_RESOLVED
    plan = _overlay(plan, resolution.index, loop.given_values(), turn.workflow)
    resolution.index, resolution.stage = resolution.index + 1, STAGE_INTENT
    return plan, None


def take_reply(resolution: PlanResolution, reply: str, plan, turn: Turn):
    """``(plan, question)`` after the reply to the question out: read in code (``intent_loop.answer`` for
    which command, ``parameter_clarification.advance`` for values), applied to the part, then ``step`` on. The record
    leaves the turn's slot first (the question is no longer out); a question asked again is put back."""
    held = turn.port.get()
    turn.port.clear()
    take = _take_choice if held.kind == waiting.KIND_INTENT else _take_values
    plan, question = take(resolution, held, reply, plan, turn)
    if question is not None:
        return plan, question
    return step(resolution, plan, turn)


def dropped(resolution: PlanResolution, plan) -> list[dict]:
    """``{text, command, means, reason, planning_scores}`` of each part that was cancelled while it was resolved (before ``finish``)."""
    return [{"text": part["text"], "command": plan.items[index].means.name or None,
             "means": plan.items[index].means.kind, "reason": part.get("dropped_reason") or "it was declined",
             "planning_scores": part.get("planning_scores") or {}}
            for index, part in enumerate(resolution.parts) if part["state"] == PART_DROPPED]


def finish(resolution: PlanResolution, plan, subtasks: list):
    """``(plan, subtasks, asked)`` with the dropped parts left out and every ``from sub-task N`` renumbered, and the
    span's record of the which-command questions (``{subtask, means, alternatives}`` by the new numbers)."""
    keep = [index for index, part in enumerate(resolution.parts) if part["state"] != PART_DROPPED]
    number = {old + 1: new + 1 for new, old in enumerate(keep)}
    items = []
    for old in keep:
        item = plan.items[old]
        inputs = [each.model_copy(update={"from_subtask": number[each.from_subtask]})
                  if each.from_subtask is not None and each.from_subtask in number else each
                  for each in item.inputs if each.from_subtask is None or each.from_subtask in number]
        items.append(item.model_copy(update={"inputs": inputs}))
    asked = [{"subtask": number[record["origin"] + 1], "means": items[keep.index(record["origin"])].means.render(),
              "alternatives": [name for name in record["alternatives"] if name != record.get("picked")]}
             for record in resolution.asked if record["origin"] + 1 in number]
    return plan.model_copy(update={"items": items}), [subtasks[old] for old in keep], asked


def single_part_chosen(resolution: PlanResolution) -> bool:
    """Whether the turn is one part whose command the user settled by answering which one it is, or that the gate's fast
    path took (``fast_path``: it names a command, or the local model is sure of it) and whose values were then asked for:
    like a named command, that part runs with no review."""
    kept = [part for part in resolution.parts if part["state"] != PART_DROPPED]
    return len(kept) == 1 and bool(kept[0]["chosen"] or kept[0].get("fast_path"))
