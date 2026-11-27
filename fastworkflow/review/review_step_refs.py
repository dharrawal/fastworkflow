"""Step references in a plan-review reply, read without a model.

A user answering a plan review names steps by the numbers the review showed: ``drop steps 2, 3 and 4 and keep only
step 1``, ``remove steps 4 and 5``, ``steps 5 to 9 are wrong``. Which numbers a sentence names, and whether it asks to
remove or to keep them, is decided here by one grammar, so every reader of a reply (the routed ``drop`` command, the
free-text edit reader's guards, the engine) agrees on it. Numbers are the 1-based numbers of the plan as shown, never
renumbered.

The reading is conservative on purpose: a removal that is negated (``do not remove step 2``) is a keep, a reply that
both removes and keeps the same step is a contradiction (``RemovalReading.contradiction``), and nothing here guesses
a step from a phrase. A caller that finds the reading disagrees with what another reader produced either corrects it
(the reply is certain) or asks the user again.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUM = r"#?\d+"
_RANGE_SEP = r"\s*(?:to|through|thru|until|[-–—])\s*"
_ELEM = rf"{_NUM}(?:{_RANGE_SEP}{_NUM})?"
_NOUN = r"(?:steps?|items?|sub-?tasks?)"
#: Separators of a list of steps: ``2, 3 and 4``, ``2 & 3``, ``2, 3, and 4``, ``2 or 3``.
_LIST_SEP = r"(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+|\s*&\s*)"
#: One reference to one or more steps: the noun, then the list. The noun may be repeated inside the list
#: (``steps 2 and 3 and step 5``).
#: The numbers of a list with no leading noun (``2, 3 and 4``, ``2 to 4``, ``2 and step 5``).
NUMBER_LIST = rf"{_ELEM}(?:{_LIST_SEP}(?:{_NOUN}\s+)?{_ELEM})*"
STEP_LIST = rf"{_NOUN}\s+{_ELEM}(?:{_LIST_SEP}(?:{_NOUN}\s+)?{_ELEM})*"
_STEP_LIST_RE = re.compile(rf"\b{STEP_LIST}", re.I)
_ELEM_RE = re.compile(rf"({_NUM})(?:{_RANGE_SEP}({_NUM}))?")


def _numbers(text: str, count: int) -> list[int]:
    found: list[int] = []
    for match in _ELEM_RE.finditer(text):
        low = int(match.group(1).lstrip("#"))
        high = int(match.group(2).lstrip("#")) if match.group(2) else low
        if low > high:
            low, high = high, low
        for number in range(low, min(high, count) + 1):
            if 1 <= number <= count and number not in found:
                found.append(number)
    return found


def step_numbers(text: str, count: int) -> list[int]:
    """The 1-based step numbers *text* names, within ``1..count``, in the order written, each once.

    Handles ``step 2``, ``steps 2 and 3``, ``steps 2, 3 and 4``, ``steps 2 to 4``, ``steps 1, 3 and 5 to 7``,
    ``step 2 and step 3`` and ``#2``. A bare number with no ``step`` before it is not a reference, and neither is a
    number outside the plan.
    """
    if not text or count <= 0:
        return []
    found: list[int] = []
    for match in _STEP_LIST_RE.finditer(text):
        body = re.sub(rf"^{_NOUN}\s+", "", match.group(0), flags=re.I)
        for number in _numbers(body, count):
            if number not in found:
                found.append(number)
    return found


def bare_step_numbers(text: str, count: int) -> list[int]:
    """The numbers of *text* when the whole text is a list of steps (``2, 3 and 4``, ``steps 2 to 4``), else [].

    A text that only contains a reference among other words is not one: ``step 1 - not step 2`` names two steps and
    is no list.
    """
    stripped = (text or "").strip().rstrip(".!").strip()
    if re.fullmatch(rf"(?:{_NOUN}\s+)?{_ELEM}(?:{_LIST_SEP}(?:{_NOUN}\s+)?{_ELEM})*", stripped, re.I):
        return _numbers(re.sub(rf"{_NOUN}\s+", "", stripped, flags=re.I), count)
    return []


# ----------------------------------------------------------------------
# What the reply does with the numbers it names
# ----------------------------------------------------------------------

_REMOVE_VERBS = r"(?:drop|remove|delete|skip|omit|cut|leave\s+out|take\s+out|get\s+rid\s+of|discard|scrap)"
_DETERMINERS = r"(?:(?:the|these|those|this|that|both|all|extra|duplicate|stray|separate|following|last)\s+)*"
#: ``drop steps 2, 3 and 4`` / ``please remove the extra steps 4 and 5``.
_REMOVE_OBJECT = re.compile(rf"\b{_REMOVE_VERBS}\s+{_DETERMINERS}(?P<list>{STEP_LIST})", re.I)
#: ``remove the date filter in step 2``: the verb's object is something in the step, so the step is touched but not removed.
_REMOVE_PART_OF_STEP = re.compile(
    rf"\b{_REMOVE_VERBS}\s+(?!{_DETERMINERS}{_NOUN}\b)(?!\d)[^.;,]{{1,60}}?\b(?:in|of|from|on|for|at)\s+(?P<list>{STEP_LIST})", re.I)
#: ``drop 2 and 3`` (no noun): taken only when the list is the whole object of the verb, followed by the end of the
#: clause or a courtesy word, so ``remove 2 logins`` is not a removal of step 2.
_REMOVE_BARE = re.compile(
    rf"\b{_REMOVE_VERBS}\s+{_DETERMINERS}(?P<list>{_ELEM}(?:{_LIST_SEP}{_ELEM})*)\s*(?:please|thanks|thank\s+you)?\s*[.!;]?\s*$",
    re.I)
#: ``steps 2 and 3 should go`` / ``step 4 can be dropped`` / ``step 2 should not be there``.
_PASSIVE_REMOVE = re.compile(
    rf"\b(?P<list>{STEP_LIST})\s+(?:(?:should|must|can|could|needs?\s+to|has\s+to|have\s+to)\s+"
    r"(?:not\s+be\s+there|go|be\s+(?:removed|dropped|deleted|omitted|skipped|cut))|"
    r"(?:is|are)\s+(?:not\s+needed|not\s+wanted|unneeded|unnecessary|not\s+there)|(?:isn'?t|aren'?t)\s+needed)",
    re.I)
#: ``keep only step 1`` / ``just keep steps 1 and 2``. ``only`` or ``just`` makes the list the only steps that stay.
#: A negated keep (``do not keep step 2``, ``no need to keep steps 3 and 4``) is a removal.
_KEEP = re.compile(
    rf"\b(?:keep|retain|preserve|stay\s+with)\s+(?P<only>(?:only|just)\s+)?{_DETERMINERS}(?P<list>{STEP_LIST})", re.I)
#: ``leave step 3 out`` / ``take step 3 out`` is a removal; ``leave step 3 as is`` / ``alone`` / ``untouched`` is a keep. A bare ``leave step 3``
#: is neither.
_LEAVE_OUT = re.compile(rf"\b(?:leave|take)\s+{_DETERMINERS}(?P<list>{STEP_LIST})\s+out\b", re.I)
_LEAVE_AS_IS = re.compile(
    rf"\bleave\s+{_DETERMINERS}(?P<list>{STEP_LIST})\s+(?:as\s+(?:it\s+is|they\s+are|is|shown)|alone|untouched|unchanged|be)\b", re.I)
_KEEP_ONLY_AFTER = re.compile(
    rf"\b(?:only|just)\s+(?P<list>{STEP_LIST})\s+(?:stays?|remains?|is\s+(?:needed|enough|wanted)|are\s+(?:needed|enough|wanted)|"
    r"should\s+(?:stay|remain)|can\s+stay)", re.I)
#: ``steps 1 to 4 are fine`` (the step is right as it is: it stays).
_FINE = re.compile(
    rf"\b(?P<list>{STEP_LIST})\s+(?:(?:look|looks|is|are|seem|seems)\s+)?(?:fine|good|ok|okay|right|correct|great)\b", re.I)
_NEGATION_BEFORE = re.compile(
    r"(?:\bnot\b|n't\b|\bnever\b|\bno\s+need\s+to\b|\bwithout\b|\binstead\s+of\b|\brather\s+than\b|\bno\s+longer\b)"
    r"(?:(?!\bbut\b)[^.;!?,\n]){0,24}$", re.I)
_CLAUSE_SPLIT = re.compile(r"(?:[.;!?\n]+|\s+[—–]\s+)")


@dataclass(frozen=True)
class RemovalReading:
    """What a reply says about removing and keeping steps.

    ``remove``: steps the reply clearly asks to remove, in the order written. ``keep``: steps it says stay (``keep only
    step 1``, ``steps 1 to 4 are fine``, ``do not remove step 2``). ``keep_only``: the reply named the steps that are
    the only ones to stay (so every other step is a removal). ``contradiction``: a step is both removed and kept.
    ``uncertain``: steps a removal verb touched without a certain reading (``remove step 2's filter``, ``skip step 2 if
    ...``, ``drop step 3 into step 2``): neither removed nor kept, and no reader may be corrected on them.
    """

    remove: tuple[int, ...] = ()
    keep: frozenset[int] = field(default_factory=frozenset)
    keep_only: bool = False
    contradiction: bool = False
    uncertain: frozenset[int] = field(default_factory=frozenset)

    @property
    def certain(self) -> bool:
        return bool(self.remove) and not self.contradiction


def _negated(clause: str, start: int) -> bool:
    return bool(_NEGATION_BEFORE.search(clause[:start]))


#: What may follow a list of steps for it to be the complete object of the removal verb: the end of the clause, a comma, a
#: courtesy or emphasis word, 'from the plan', or a coordinating or explanatory continuation ('and keep ...', 'but ...',
#: 'because ...'). A possessive, a condition, a restructuring ('into step 2', 'from the merge', 'to five results') or a
#: noun after the list means the verb is about something of the step, not the step.
_CERTAIN_TAIL = re.compile(
    r"^\s*(?:$|[,:;]|[—–-]\s|(?:please|thanks|thank\s+you|too|also|as\s+well|now|entirely|completely|altogether|"
    r"outright|only(?!\s+(?:if|when|unless|once))|of\s+the\s+(?:plan|list)|from\s+the\s+(?:plan|list|review)|out\s+of\s+the\s+(?:plan|list)|"
    r"but|so|because|since|as|and\s+(?:keep|retain|leave|also|then|drop|remove|delete|skip|omit|cut|discard|scrap|step|steps)|"
    r"they|it|those|these|which|that|where|for)\b)", re.I)
_CONDITION_BEFORE = re.compile(r"\b(?:if|when|whenever|unless|in\s+case|once)\b", re.I)


def _complete_object(clause: str, match: "re.Match[str]") -> bool:
    """The list in *match* is the whole object of its verb and the removal is not conditional."""
    if _CONDITION_BEFORE.search(clause[:match.start()]):
        return False
    return bool(_CERTAIN_TAIL.match(clause[match.end():]))


def removal_reading(text: str, count: int) -> RemovalReading:
    """Read which steps *text* asks to remove and which it says to keep (see :class:`RemovalReading`)."""
    if not text or count <= 0:
        return RemovalReading()
    remove: list[int] = []
    keep: set[int] = set()
    keep_only: set[int] | None = None
    uncertain: set[int] = set()

    def removed(numbers: list[int]) -> None:
        remove.extend(number for number in numbers if number not in remove)

    for clause in (piece.strip() for piece in _CLAUSE_SPLIT.split(text)):
        if not clause:
            continue
        for pattern in (_REMOVE_OBJECT, _PASSIVE_REMOVE, _LEAVE_OUT):
            for match in pattern.finditer(clause):
                numbers = step_numbers(match.group("list"), count)
                if not _complete_object(clause, match):
                    uncertain.update(numbers)
                elif _negated(clause, match.start()):
                    keep.update(numbers)
                else:
                    removed(numbers)
        for match in _REMOVE_PART_OF_STEP.finditer(clause):
            uncertain.update(step_numbers(match.group("list"), count))
        for match in _REMOVE_BARE.finditer(clause):
            numbers = bare_step_numbers(match.group("list"), count)
            if _CONDITION_BEFORE.search(clause[:match.start()]):
                uncertain.update(numbers)
            elif _negated(clause, match.start()):
                keep.update(numbers)
            else:
                removed(numbers)
        for match in _KEEP.finditer(clause):
            numbers = step_numbers(match.group("list"), count)
            if not _complete_object(clause, match):
                continue
            if _negated(clause, match.start()):
                removed(numbers)
                continue
            if match.group("only"):
                keep_only = set(numbers) if keep_only is None else keep_only | set(numbers)
            keep.update(numbers)
        for match in _LEAVE_AS_IS.finditer(clause):
            if not _negated(clause, match.start()):
                keep.update(step_numbers(match.group("list"), count))
        for match in _KEEP_ONLY_AFTER.finditer(clause):
            numbers = step_numbers(match.group("list"), count)
            keep_only = set(numbers) if keep_only is None else keep_only | set(numbers)
            keep.update(numbers)
        for match in _FINE.finditer(clause):
            if not _negated(clause, match.start()):
                keep.update(step_numbers(match.group("list"), count))
    if keep_only is not None:
        keep |= keep_only
        remove.extend(number for number in range(1, count + 1) if number not in keep_only and number not in remove)
    contradiction = bool(set(remove) & keep)
    return RemovalReading(
        remove=tuple(remove),
        keep=frozenset(keep),
        keep_only=keep_only is not None,
        contradiction=contradiction,
        uncertain=frozenset(uncertain - set(remove) - keep),
    )
