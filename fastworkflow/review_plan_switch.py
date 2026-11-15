"""Switching a sub-task's plan from the review, by naming it (bd fix-1ma8.1).

When a sub-task's planning-command match was ambiguous (``review_proposals.plan_alternatives``), the review names the
other plans that fit and says ``step N: use NAME`` switches to one. This module reads that reply, deterministically
and with no model: it turns ``use B``, ``step N: use B``, ``use B for step N``, ``switch step N to B``, ``switch to B``
and ``B instead`` into the existing means edit (``change N: means=planning:B``), which re-drafts the step's inputs
(``SubTaskProposal.revised``). It is the reply policy's step between the accept and decline cues and the edit reader
(``review_engine.ProposalReplyPolicy``): an accept or a decline is read first and never reaches it.

What it handles and what it leaves alone:

- Only a review with at least one step that has alternatives. Any other review is read as before (it returns None
  without looking at the reply).
- A reply is one or more pieces separated by ``;`` or a new line; every piece must be a plan switch, or the reply is not
  handled here (all or nothing, like the typed edits).
- A name is matched against the alternatives of that step only: first by exact name (case, spaces, ``_`` and the
  context prefix ignored), then by the same fuzzy rule the intent clarification reads a command name with
  (``utils.fuzzy_match.find_best_matches`` and intent detection's own threshold, on the names without their
  context prefix, a name of at least ``MIN_FUZZY_CHARS`` characters). A name that is not among the step's alternatives is not handled here (None: the
  existing reader reads the reply, which may mean something else by "use").
- ``read_plan_switch`` is the all-switch reader of the alternatives (kept for the accept-with-switch path). A reply that
  holds an explicit switch clause is first read clause by clause by ``split_switch_clauses`` (bd fix-1ma8.1 follow-up,
  below), which the reply policy calls before this reader.
- A piece with no step number is for the one ambiguous step; with two or more ambiguous steps it is asked again, naming
  them. A name that fits several alternatives of the step is asked again too.
"""
from __future__ import annotations

import re
from typing import Optional

#: A fuzzy name shorter than this (after dropping spaces and ``_``) is matched exactly only: the fuzzy rule compares the
#: leading characters of a name, so a word of a few letters would match any name that starts with it.
MIN_FUZZY_CHARS = 4

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


def _read_piece(piece: str, bare: bool = False) -> Optional[tuple[Optional[int], str, bool]]:
    """``(step or None, the name as written)`` of one plan-switch piece, or None when the piece is something else.
    With *bare* (the rest of a reply that opened with an accept word: ``yes, B``) a piece that is only a name is one, read
    by exact name only (the third item is True): ``yes, please`` must not fuzzy-match a plan."""
    piece = piece.strip().rstrip(".!?").strip()
    piece = re.sub(r"^(?:please|ok(?:ay)?|no|actually)\s*[,:]?\s+", "", piece, flags=re.IGNORECASE)
    for pattern, (step_group, name_group) in _PIECES:
        if match := pattern.match(piece):
            phrase = match.group(name_group).strip().rstrip(".!?, ").strip(_QUOTES + " ")
            if phrase:
                return (int(match.group(step_group)) if step_group else None), phrase, False
    if bare and piece and not re.search(r"\bstep\b", piece, re.IGNORECASE):
        return None, piece.strip(_QUOTES + " "), True
    return None


def _names_of(alternative_name: str) -> list[str]:
    """The ways an alternative may be named: its full name and, for ``Context/name``, the name without the context."""
    leaf = alternative_name.rsplit("/", 1)[-1]
    return [alternative_name] if leaf == alternative_name else [alternative_name, leaf]


def _resolve(phrase: str, alternatives: list, exact_only: bool = False) -> tuple[list[str], bool]:
    """``(the alternatives' names *phrase* names, whether it was a fuzzy match)``: one name, several (a tie), or none.
    The phrase as written is tried first, then without a leading "the" or a trailing "plan"."""
    cleaned = _clean_phrase(phrase)
    names, fuzzy = _resolve_one(phrase, alternatives, exact_only)
    if not names and cleaned and cleaned != phrase:
        return _resolve_one(cleaned, alternatives, exact_only)
    return names, fuzzy


def _resolve_one(phrase: str, alternatives: list, exact_only: bool = False) -> tuple[list[str], bool]:
    from fastworkflow.utils.fuzzy_match import find_best_matches, normalize_text

    wanted = normalize_text(phrase)
    if not wanted:
        return [], False
    by_written: dict[str, str] = {}
    for each in alternatives:
        for written in _names_of(each.name):
            by_written.setdefault(written, each.name)
    exact = list(dict.fromkeys(by_written[w] for w in by_written if normalize_text(w) == wanted))
    if exact:
        return exact, False
    if exact_only or len(wanted) < MIN_FUZZY_CHARS or len(_clean_phrase(phrase).split()) > 1:
        # A sentence is never fuzzy-matched to a name: "the plan that shows ..." must not land on a plan by its first letters.
        return [], False
    from fastworkflow._workflows.command_metadata_extraction.intent_detection import _FUZZY_PREMATCH_MAX_DISTANCE

    # Fuzzy on the names without their context prefix: the rule compares the leading characters of a name, and every name
    # of a context starts with the context's.
    by_leaf: dict[str, list[str]] = {}
    for each in alternatives:
        by_leaf.setdefault(each.name.rsplit("/", 1)[-1], []).append(each.name)
    matched, _ = find_best_matches(phrase.replace(" ", "_"), list(by_leaf), threshold=_FUZZY_PREMATCH_MAX_DISTANCE)
    return list(dict.fromkeys(name for leaf in matched for name in by_leaf[leaf])), True


def alternatives_by_step(proposal) -> dict[int, list]:
    """``{1-based step: its alternatives}`` of the steps of a sub-task review that have any."""
    from fastworkflow.review_proposals import SubTaskProposal

    if not isinstance(proposal, SubTaskProposal):
        return {}
    return {number: list(item.alternatives) for number, item in enumerate(proposal.items, 1) if item.alternatives}


def _ask_again(message: str):
    from fastworkflow.review_engine import ReplyDecision, Verdict

    return ReplyDecision(Verdict.REPROMPT, "plan switch unclear", message=message)


def _steps_listing(ambiguous: dict[int, list]) -> str:
    return " ".join(f"Step {number}: {', '.join(each.name for each in alts)}." for number, alts in ambiguous.items())


def read_plan_switch(proposal, text: str, *, bare: bool = False):
    """The reply policy's decision for *text* when it switches a step's plan, else None (not handled here).

    An EDIT decision carries ``Change(index, "means", "planning:NAME")`` per switched step. A REPROMPT decision asks
    again (which step, or which of several names), with its message. Never calls a model."""
    ambiguous = alternatives_by_step(proposal)
    if not ambiguous:
        return None
    pieces = [piece for piece in re.split(r"[;\n]", text or "") if piece.strip()]
    read = [_read_piece(piece, bare) for piece in pieces]
    if not read or any(each is None for each in read):
        return None
    from fastworkflow.review_engine import Change, ReplyDecision, Verdict

    chosen: dict[int, str] = {}
    for step, phrase, exact_only in read:
        if step is None:
            if len(ambiguous) == 1:
                step = next(iter(ambiguous))
            else:
                # The reply does not say which step. A name that fits none of the ambiguous steps is not ours.
                fitting = [number for number, alts in ambiguous.items() if _resolve(phrase, alts, exact_only)[0]]
                if not fitting:
                    return None
                return _ask_again(
                    f"Which step do you mean? These steps have other plans that fit: "
                    f"{_steps_listing(ambiguous)} Reply \"step N: use NAME\".")
        if step not in ambiguous:
            return None
        names, _ = _resolve(phrase, ambiguous[step], exact_only)
        if not names:
            return None
        if len(names) > 1:
            return _ask_again(f"Which plan for step {step}? {', '.join(names)}. Reply \"step {step}: use NAME\".")
        if chosen.get(step, names[0]) != names[0]:
            return _ask_again(f"Step {step} was given two different plans. Reply with one: "
                              f"{', '.join(each.name for each in ambiguous[step])}.")
        chosen[step] = names[0]
    return ReplyDecision(Verdict.EDIT, "plan switch",
                         edits=tuple(Change(index=step, field="means", value=f"planning:{name}")
                                     for step, name in sorted(chosen.items())))


_ACCEPT_OPENING = re.compile(
    r"^\s*(?:yes|yep|yeah|sure|ok|okay|accept|approve|go\s+ahead|sounds\s+good|looks\s+good)\b[\s,;:.!\-\u2013\u2014]*"
    r"(?:but\s+|except\s+|only\s+)?", re.IGNORECASE)


def read_accept_with_switch(proposal, text: str):
    """The decision for a reply that opens with an accept word and then names another plan (``yes use B``, ``yes, but use
    B for step 2``, ``sure, switch to B``, ``ok, B``), or None: an accept stays an accept. Only a review with an ambiguous
    step is ever read (the caller checks), only the text after the accept word is read (a bare ``yes`` has none), and
    only when it names one of that step's alternatives (or is one of the ask-again cases of ``read_plan_switch``)."""
    match = _ACCEPT_OPENING.match(text or "")
    rest = (text or "")[match.end():] if match else ""
    if not rest.strip():
        return None
    return read_plan_switch(proposal, rest, bare=True)


# ----------------------------------------------------------------------
# Clause-level reading (bd fix-1ma8.1 follow-up)
#
# A reply clause in an explicit switch form is decided here and never reaches the free-text reader, whatever the name
# says: a name among the step's alternatives switches; a name of any other planning or execution command of the workflow
# switches too (the same ``change N: means=...`` edit, which the structured edit accepts for any step); a name that
# resolves to nothing, or to several, asks again; a step that does not exist asks again. A reply of several clauses is
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
        from fastworkflow import planning_commands, run_listing
        from fastworkflow.workflow_agent import core_command_names

        folder = workflow.folderpath
        every = planning_commands.resolve_field_sources(folder).planning_commands
        planning = {name for name, command in every.items() if not planning_commands.is_framework_planning_command(command)}
        crd = run_listing._routing_registry().get_definition(folder)
        core = core_command_names()
        names = {name for context in crd.contexts for name in run_listing._context_names(crd, context)}
        return {name: ("planning" if name in planning else "plain") for name in sorted(names)
                if name not in core and name not in set(every) - planning
                and name.rsplit("/", 1)[-1] not in (*run_listing.NON_COMMAND_LABELS, "startup")}
    except Exception:  # noqa: BLE001 - a registry that cannot be read means no name beyond the alternatives resolves
        return {}


class _Named:
    def __init__(self, name: str):
        self.name = name


def _resolve_name(phrase: str, alternatives: list, universe: dict[str, str], fuzzy: bool = True) -> tuple[str, list[str]]:
    """``("one", [name])``, ``("many", names)`` or ``("none", [])``: exact on the alternatives, exact on every command of
    the workflow, then (a one-word phrase only) fuzzy on the alternatives, each only when it names exactly one. No fuzzy
    match over every command of the workflow: "use the library" would land on ``library_hours`` by its leading characters."""
    everything = [_Named(name) for name in universe]
    for pool, exact_only in ((alternatives, True), (everything, True), (alternatives, False)):
        if not pool or (not fuzzy and not exact_only):
            continue
        names, _ = _resolve(phrase, pool, exact_only)
        if len(names) == 1:
            return "one", names
        if len(names) > 1:
            return "many", names
    return "none", []


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


def _ask_unresolved(step: int, phrase: Optional[str], alternatives: list, many: Optional[list[str]] = None):
    if many:
        return _ask_again(f"Which plan for step {step}? {', '.join(many)}. Reply \"step {step}: use NAME\".")
    shown = f"I could not find a plan named {phrase!r} for step {step}. " if phrase else f"Which plan should step {step} use? "
    if alternatives:
        return _ask_again(f"{shown}Plans that fit step {step}: {', '.join(each.name for each in alternatives)}. "
                          f"Reply \"step {step}: use <one of these>\", or say what step {step} should do.")
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
    from fastworkflow.review_engine import Change
    from fastworkflow.review_proposals import SubTaskProposal

    if not isinstance(proposal, SubTaskProposal) or not proposal.items:
        return None
    pieces = [piece for piece in re.split(r"[;\n]", text or "") if piece.strip()]
    count = len(proposal.items)
    ambiguous = alternatives_by_step(proposal)
    universe: Optional[dict[str, str]] = None
    chosen: dict[int, str] = {}
    rest: list[str] = []

    def alternatives_of(step: int) -> list:
        return list(proposal.items[step - 1].alternatives)

    def pool() -> dict[str, str]:
        nonlocal universe
        if universe is None:
            universe = known_plan_names(workflow)
        return universe

    def kind_of(name: str, step: int) -> str:
        # an alternative is a planning command; any other name has the kind the workflow lists it with
        return "planning" if name in {each.name for each in alternatives_of(step)} else pool().get(name, "planning")

    for piece in pieces:
        if _TYPED_CLAUSE.match(piece):
            rest.append(piece)
            continue
        if (peeled := _peel_tail(piece)) is not None:
            # ``step 7: use NAME. I want ...``: the leading token is the name when it resolves (exactly); the sentence after
            # it is an explanation (ignored) or a clause of its own (read like any other remainder clause).
            clause, tail = peeled
            lead = _explicit_piece(clause)
            lead_step = lead[0] if lead and lead[0] is not None else (next(iter(ambiguous)) if len(ambiguous) == 1 else None)
            if lead and lead_step is not None and 1 <= lead_step <= count and _resolve_name(
                    lead[1], alternatives_of(lead_step), pool(), fuzzy=False)[0] == "one":
                piece = clause
                if _EDIT_TAIL.search(tail):
                    rest.append(tail)
        read = _explicit_piece(piece)
        if read is None:
            if _unnamed(piece) and (ref := _STEP_REF.search(piece)):
                step = int(ref.group(1))
                if not 1 <= step <= count:
                    return SwitchSplit(ask=_no_such_step(step, count))
                return SwitchSplit(ask=_ask_unresolved(step, None, alternatives_of(step)))
            rest.append(piece)
            continue
        step, phrase, always = read
        if step is None:
            # A form with no step number: only for the one ambiguous step, and only when the name resolves.
            if len(ambiguous) == 1:
                step = next(iter(ambiguous))
                state, names = _resolve_name(phrase, alternatives_of(step), pool())
            elif len(ambiguous) > 1:
                fitting = [number for number, alts in ambiguous.items() if _resolve(phrase, alts)[0]]
                if not fitting:
                    rest.append(piece)
                    continue
                return SwitchSplit(ask=_ask_again(
                    f"Which step do you mean? These steps have other plans that fit: "
                    f"{_steps_listing(ambiguous)} Reply \"step N: use NAME\"."))
            else:
                rest.append(piece)
                continue
            if state == "none":
                rest.append(piece)
                continue
        else:
            if not 1 <= step <= count:
                return SwitchSplit(ask=_no_such_step(step, count))
            state, names = _resolve_name(phrase, alternatives_of(step), pool())
            if state == "none":
                if not (always or _NAME_LIKE.match(_clean_phrase(phrase)) or _unnamed(piece)):
                    rest.append(piece)               # "use the 2024 data for step 3": a goal edit, the reader's
                    continue
                return SwitchSplit(ask=_ask_unresolved(step, None if _unnamed(piece) else phrase, alternatives_of(step)))
        if state == "many":
            return SwitchSplit(ask=_ask_unresolved(step, phrase, alternatives_of(step), names))
        if chosen.get(step, names[0]) != names[0]:
            return SwitchSplit(ask=_ask_again(f"Step {step} was given two different plans. Reply with one."))
        chosen[step] = names[0]
    if not chosen:
        return None
    return SwitchSplit(
        edits=tuple(Change(index=step, field="means", value=f"{kind_of(name, step)}:{name}")
                    for step, name in sorted(chosen.items())),
        remainder="; ".join(piece.strip() for piece in rest))


def _no_such_step(step: int, count: int):
    valid = "step 1" if count == 1 else f"steps 1 to {count}"
    return _ask_again(f"There is no step {step}; this plan has {valid}. Reply \"step N: use <plan name>\" with one of them.")
