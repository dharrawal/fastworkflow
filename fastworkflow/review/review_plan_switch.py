"""Switching a sub-task's plan from the review, by naming it (bd fix-1ma8.1).

A reply clause in the explicit form ``step N: use NAME`` (also ``use NAME for step N`` and ``switch step N to NAME``) is
read here, deterministically and with no model, and becomes the existing means edit (``change N: means=planning:NAME``,
or ``plain:NAME`` for an execution command), which re-drafts the step's inputs (``SubTaskProposal.revised``). The reply
policy (``review_engine.ProposalReplyPolicy``) reads it after the accept and decline cues, so an accept or a decline
never reaches it, and before the edit reader, so a switch clause never reaches the model.

What it handles and what it leaves alone:

- ``split_switch_clauses`` reads a reply clause by clause (``;`` or a new line). A numbered clause whose name is exactly
  a command of the workflow (case, spaces, ``_`` and the context prefix ignored: ``known_plan_names``) switches; a name
  that resolves to nothing or to several, or a step that does not exist, asks again and applies nothing. The other clauses
  are handed back as the remainder for the reader the policy uses for any reply. A form with no step number is not read
  here.
- ``explicit_switch_clauses``, ``goal_copies_switch_clause`` and the review engine's ``_refuse_switch_clause_in_goal``
  keep a switch clause from being copied into a step's goal as text.
"""
from __future__ import annotations

import re
from typing import Optional

_STEP = r"(?:step|item|sub-?task)\s+#?(\d+)"
_PIECES: tuple[tuple[re.Pattern, tuple[Optional[int], int]], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), groups) for pattern, groups in (
        (rf"^{_STEP}\s*[:,\-]?\s*use\s+(.+)$", (1, 2)),
        (rf"^use\s+(.+?)\s+(?:for|in|on)\s+{_STEP}$", (2, 1)),
        (rf"^switch\s+{_STEP}\s+to\s+(.+)$", (1, 2)),
        (r"^(?:switch\s+to|use)\s+(.+)$", (None, 1)),
        (r"^(.+?)\s+instead$", (None, 1)),
    ))
_FILLER = re.compile(r"^(?:the|a|plan)\s+|\s+(?:plan|please|then)$", re.IGNORECASE)
_QUOTES = "\"'`“”‘’"


def _clean_phrase(text: str) -> str:
    text = text.strip().rstrip(".!?, ").strip(_QUOTES + " ")
    previous = None
    while previous != text:
        previous = text
        text = _FILLER.sub("", text).strip(_QUOTES + " ")
    return text


def _resolve(phrase: str, pool: list) -> list[str]:
    """The names of *pool* that *phrase* names exactly (case, spaces, ``_`` and the context prefix ignored): one name, several
    (a tie) or none. The phrase as written is tried first, then without a leading "the" or a trailing "plan"."""
    from fastworkflow.utils.fuzzy_match import normalize_text

    for text in dict.fromkeys((phrase, _clean_phrase(phrase))):
        wanted = normalize_text(text)
        if wanted and (found := list(dict.fromkeys(
                each.name for each in pool
                if wanted in (normalize_text(each.name), normalize_text(each.name.rsplit("/", 1)[-1]))))):
            return found
    return []


def _ask_again(message: str):
    from fastworkflow.review.review_engine import ReplyDecision, Verdict

    return ReplyDecision(Verdict.REPROMPT, "plan switch unclear", message=message)


# ----------------------------------------------------------------------
# Clause-level reading (bd fix-1ma8.1 follow-up)
#
# A reply clause in an explicit switch form is decided here and never reaches the free-text reader, whatever the name
# says: a name of a planning or execution command of the workflow switches (the ``change N: means=...`` edit, which the
# structured edit accepts for any step); a name that resolves to nothing, or to several, asks again; a step that does not
# exist asks again. A reply of several clauses is
# read clause by clause: the switch clauses here, the rest as the reply policy reads any reply. No model is called.
# ----------------------------------------------------------------------

_TYPED_CLAUSE = re.compile(r"^\s*(?:change\s+#?\d+|drop|add|merge|split|look[\s_]*up)\b", re.IGNORECASE)
_STEP_REF = re.compile(r"\b(?:step|item|sub-?task)\s+#?(\d+)\b", re.IGNORECASE)
_PLAN_NOUN = r"(?:plans?|approach(?:es)?|planning\s+commands?)"
#: A request to change WHICH plan a step uses that names no plan: a plan or approach noun is the object of
#: switch/use/pick/change and a step is referred to. "change step 3 to show the nesting too" has no plan noun, so it is not one.
_UNNAMED_SWITCH = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    rf"\b(?:switch|change|swap|move)\s+(?:step|item|sub-?task)\s+#?\d+\s+(?:over\s+)?to\s+(?:the|a|an|another|that|this)\s+"
    rf"(?:\w+\s+){{0,2}}{_PLAN_NOUN}\b",
    rf"\b(?:use|pick|choose|select|try)\s+(?:a|an|the|another)\s+(?:different|other|another|better|new|right)?\s*"
    rf"{_PLAN_NOUN}\s+(?:for|in|on)\s+(?:step|item|sub-?task)\s+#?\d+",
    rf"\b(?:wrong|different|another|other|right|better)\s+{_PLAN_NOUN}\b.{{0,30}}\b(?:step|item|sub-?task)\s+#?\d+",
    rf"\b(?:step|item|sub-?task)\s+#?\d+\b.{{0,40}}\b(?:wrong|different|another|other)\s+{_PLAN_NOUN}\b",
    rf"\b(?:switch|change|swap)\s+(?:the\s+)?{_PLAN_NOUN}\s+(?:for|of|in|on)\s+(?:step|item|sub-?task)\s+#?\d+",
))
#: The pattern of ``_PIECES`` that is always a switch clause: ``step N: use NAME``, the form the review offers. The other
#: numbered forms (``use NAME for step N``, ``switch step N to NAME``) are one when NAME is a single word or the clause
#: asks for a different plan without naming one; ``use the 2024 data for step 3`` and ``switch step 3 to show the nesting
#: too`` are goal edits and go to the reader.
_ALWAYS_EXPLICIT = (0,)
_NAME_LIKE = re.compile(r"^[\w./\-]+$")                       # one word: a name, not a phrase


def _explicit_piece(piece: str) -> Optional[tuple[Optional[int], str, bool]]:
    """``(step or None, the name as written, whether the form is always a switch)`` of an explicit switch clause, else None."""
    text = piece.strip().rstrip(".!?").strip()
    text = re.sub(r"^(?:please|ok(?:ay)?|no|actually)\s*[,:]?\s+", "", text, flags=re.IGNORECASE)
    for position, (pattern, (step_group, name_group)) in enumerate(_PIECES):
        if match := pattern.match(text):
            phrase = match.group(name_group).strip().rstrip(".!?, ").strip(_QUOTES + " ")
            if phrase:
                return ((int(match.group(step_group)) if step_group else None), phrase, position in _ALWAYS_EXPLICIT)
    return None


def explicit_switch_clauses(text: str) -> list[str]:
    """The clauses of *text* in an explicit switch form with a step number (``step N: use NAME``, ``use NAME for step N``,
    ``switch step N to NAME``): each as written, found as a ``;``/newline piece or at the end of a longer piece. A bare
    ``use NAME`` is one only as a piece of its own."""
    found: list[str] = []
    for piece in re.split(r"[;\n]", text or ""):
        if not piece.strip():
            continue
        read = None if _TYPED_CLAUSE.match(piece) else _explicit_piece(piece)
        # A bare form (no step number) is a switch clause only for a one-word name: "make it about trains instead" is a goal.
        if read is not None and (read[0] is not None or _NAME_LIKE.match(_clean_phrase(read[1]))):
            found.append(piece.strip())
            continue
        for pattern, _ in _PIECES[:3]:
            inner = re.compile(pattern.pattern.lstrip("^"), re.IGNORECASE)
            for match in inner.finditer(piece):
                found.append(match.group(0).strip())
    return found


def _norm_clause(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().rstrip(".!?,; ").lower()


def goal_copies_switch_clause(goal: str, reply_text: str) -> Optional[str]:
    """The explicit switch clause of *reply_text* that *goal* is, or ends with (a clause copied into a goal), else None."""
    written = _norm_clause(goal)
    for clause in explicit_switch_clauses(reply_text):
        wanted = _norm_clause(clause)
        if wanted and written.endswith(wanted):
            return clause
    return None


def known_plan_names(workflow) -> dict[str, str]:
    """``{qualified name: "planning" | "plain"}`` of the commands a sub-task may be bound to in *workflow*: its planning
    commands and its execution commands (the framework's own and the core commands left out). Empty with no workflow or
    when the registry cannot be read. Nothing is trained or called."""
    if workflow is None or not getattr(workflow, "folderpath", None):
        return {}
    try:
        from fastworkflow.planning import planning_commands
        from fastworkflow.agent import run_listing
        from fastworkflow.agent.workflow_agent import core_command_names

        folder = workflow.folderpath
        every = planning_commands.resolve_field_sources(folder).planning_commands
        planning = {name for name, command in every.items() if not planning_commands.is_framework_planning_command(command)}
        crd = run_listing._routing_registry().get_definition(folder)
        core = core_command_names()
        names = {name for context in crd.contexts for name in run_listing._context_names(crd, context)}
        return {name: ("planning" if name in planning else "plain") for name in sorted(names)
                if name not in core and name not in set(every) - planning
                and name.rsplit("/", 1)[-1] not in run_listing.NON_COMMAND_LABELS
                and not crd.command_directory.runs_only_as_action(name)}
    except Exception:  # noqa: BLE001 - a registry that cannot be read means no name beyond the alternatives resolves
        return {}


class _Named:
    def __init__(self, name: str):
        self.name = name


def _resolve_name(phrase: str, universe: dict[str, str]) -> tuple[str, list[str]]:
    """``("one", [name])``, ``("many", names)`` or ``("none", [])``: the phrase named exactly against every command of the
    workflow. No fuzzy match: "use the library" would land on ``library_hours`` by its leading characters."""
    names = _resolve(phrase, [_Named(name) for name in universe])
    return ("one" if len(names) == 1 else "many", names) if names else ("none", [])


_TAIL_SEP = r"(?:[.!,]|:(?=\s)|\s-\s|\s(?:because|so|since)\b)"
_NAME_THEN_TAIL = tuple(re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in (
    rf"^(?P<head>{_STEP}\s*[:,\-]?\s*use)\s+(?P<name>[^\s.!,:;]+)(?P<sep>{_TAIL_SEP})\s*(?P<tail>\S.*)$",
    rf"^(?P<head>switch\s+{_STEP}\s+to)\s+(?P<name>[^\s.!,:;]+)(?P<sep>{_TAIL_SEP})\s*(?P<tail>\S.*)$",
    rf"^(?P<head>(?:switch\s+to|use))\s+(?P<name>[^\s.!,:;]+)(?P<sep>{_TAIL_SEP})\s*(?P<tail>\S.*)$",
    rf"^use\s+(?P<name>[^\s.!,:;]+)\s+(?P<tail_for>(?:for|in|on)\s+{_STEP})(?P<sep>{_TAIL_SEP})\s*(?P<tail>\S.*)$",
))
#: Trailing text after a plan name that asks for something: a step reference, an assignment or an edit verb.
_EDIT_TAIL = re.compile(
    rf"{_STEP}|=|\b(?:drop|remove|delete|add|change|merge|split|rename|set|make|replace|swap|switch|use|update|skip|keep)\b",
    re.IGNORECASE)


def _peel_tail(piece: str) -> Optional[tuple[str, str]]:
    """``(the clause up to the name, the text after it)`` for ``step 7: use NAME. I want ...`` and ``use NAME for step 7. ...``
    (the name is one token followed by sentence punctuation, ``because``, ``so`` or ``since``, and more text), else None."""
    text = re.sub(r"^(?:please|ok(?:ay)?|no|actually)\s*[,:]?\s+", "", piece.strip(), flags=re.IGNORECASE)
    for position, pattern in enumerate(_NAME_THEN_TAIL):
        if match := pattern.match(text):
            name = match.group("name")
            clause = (f"use {name} {match.group('tail_for')}" if position == 3 else f"{match.group('head')} {name}")
            return clause, match.group("tail").strip()
    return None


def _unnamed(piece: str) -> bool:
    return any(pattern.search(piece) for pattern in _UNNAMED_SWITCH)


def _ask_unresolved(step: int, phrase: Optional[str], many: Optional[list[str]] = None):
    if many:
        return _ask_again(f"Which plan for step {step}? {', '.join(many)}. Reply \"step {step}: use NAME\".")
    shown = f"I could not find a plan named {phrase!r} for step {step}. " if phrase else f"Which plan should step {step} use? "
    return _ask_again(f"{shown}Reply \"step {step}: use <plan name>\", or say what step {step} should do. The plan is unchanged.")


class SwitchSplit:
    """A reply read clause by clause: the switch edits it holds, the text of its other clauses, or a question to ask."""

    def __init__(self, edits=(), remainder: str = "", ask=None):
        self.edits = tuple(edits)
        self.remainder = remainder
        self.ask = ask


def split_switch_clauses(proposal, text: str, workflow=None) -> Optional[SwitchSplit]:
    """Read *text* clause by clause (``;`` or a new line, as typed edits are separated) and decide each explicit switch
    clause here. None when no clause was a switch (the reply is read as before). Otherwise a ``SwitchSplit``: ``ask`` when
    a question must be asked (nothing is applied), else ``edits`` (a means ``Change`` per switched step) and ``remainder``
    (the other clauses, in order, joined with ``;``, for the reader the reply policy uses for any other reply). Never calls
    a model."""
    from fastworkflow.review.review_engine import Change
    from fastworkflow.review.review_proposals import SubTaskProposal

    if not isinstance(proposal, SubTaskProposal) or not proposal.items:
        return None
    pieces = [piece for piece in re.split(r"[;\n]", text or "") if piece.strip()]
    count = len(proposal.items)
    universe: Optional[dict[str, str]] = None
    chosen: dict[int, str] = {}
    rest: list[str] = []

    def pool() -> dict[str, str]:
        nonlocal universe
        if universe is None:
            universe = known_plan_names(workflow)
        return universe

    def kind_of(name: str) -> str:
        return pool().get(name, "planning")

    for piece in pieces:
        if _TYPED_CLAUSE.match(piece):
            rest.append(piece)
            continue
        if (peeled := _peel_tail(piece)) is not None:
            # ``step 7: use NAME. I want ...``: the leading token is the name when it resolves (exactly); the sentence after
            # it is an explanation (ignored) or a clause of its own (read like any other remainder clause).
            clause, tail = peeled
            lead = _explicit_piece(clause)
            if lead and lead[0] is not None and 1 <= lead[0] <= count and _resolve_name(lead[1], pool())[0] == "one":
                piece = clause
                if _EDIT_TAIL.search(tail):
                    rest.append(tail)
        read = _explicit_piece(piece)
        if read is None:
            if _unnamed(piece) and (ref := _STEP_REF.search(piece)):
                step = int(ref.group(1))
                if not 1 <= step <= count:
                    return SwitchSplit(ask=_no_such_step(step, count))
                return SwitchSplit(ask=_ask_unresolved(step, None))
            rest.append(piece)
            continue
        step, phrase, always = read
        if step is None:
            rest.append(piece)                       # a form with no step number is not read here
            continue
        if not 1 <= step <= count:
            return SwitchSplit(ask=_no_such_step(step, count))
        state, names = _resolve_name(phrase, pool())
        if state == "none":
            if not (always or _NAME_LIKE.match(_clean_phrase(phrase)) or _unnamed(piece)):
                rest.append(piece)                   # "use the 2024 data for step 3": a goal edit, the reader's
                continue
            return SwitchSplit(ask=_ask_unresolved(step, None if _unnamed(piece) else phrase))
        if state == "many":
            return SwitchSplit(ask=_ask_unresolved(step, phrase, names))
        if chosen.get(step, names[0]) != names[0]:
            return SwitchSplit(ask=_ask_again(f"Step {step} was given two different plans. Reply with one."))
        chosen[step] = names[0]
    if not chosen:
        return None
    return SwitchSplit(
        edits=tuple(Change(index=step, field="means", value=f"{kind_of(name)}:{name}")
                    for step, name in sorted(chosen.items())),
        remainder="; ".join(piece.strip() for piece in rest))


def _no_such_step(step: int, count: int):
    valid = "step 1" if count == 1 else f"steps 1 to {count}"
    return _ask_again(f"There is no step {step}; this plan has {valid}. Reply \"step N: use <plan name>\" with one of them.")
