"""The one reply reader: how a reply to a parameter question reads (bd fix-qoyz steps 4 and 6).

A reply to a parameter question is read in code, the same way whether the command is a planning command or an
execution command. ``read`` is the only reader; ``parameter_clarification.read_reply`` (a plan's loop and a plan's
decision step) and ``ParameterExtraction._extract_and_merge_missing_parameters`` (an execution command's reply) call it
the same way. The rule:

a. ``cancel`` and its synonyms decline. A yes (``no_agent_mode.is_yes``: yes, true, ok, agreed, accept, go) accepts
   only when the question was a confirmation (it showed drafted values and asked to confirm, ``confirm``); otherwise
   it is read by c.
b. ``name: value`` (or ``name=value``) lines set the named fields, for any field the question listed: the open ones
   (``open_names``) and the ones it showed with a value (``names``). A one-line reply whose comma-separated parts are
   ALL such lines (``title: Dune, author: Herbert``) is read as those lines.
c. With exactly one value open, ``yes``, ``no``, ``none`` and the like are that value when the field is a yes/no field
   or the word is one of its allowed values (compared without regard to case; an allowed value is returned spelled as
   the field declares it). When the field is neither, a bare ``no``, ``none``, ``neither``, ``n`` or ``nope`` for an
   OPTIONAL field means "leave it empty" (the value ``None``, the question is closed for that field) and for a REQUIRED
   field is unclear (asked again; the user has ``cancel``). Every other word, ``yes`` included, is an ordinary value.
d. With exactly one value open and a reply that is not a name line, the WHOLE reply is the value, commas included;
   surrounding quotes are removed.
e. With several values open and a reply with no name lines, its comma-separated parts fill the open values in the
   order the question listed them, only when (i) the number of parts equals the number of open values and none is
   empty, (ii) no part has the form ``word: value`` or ``word=value`` (a named part whose name is not a listed field
   means the reply tried the named form with a name the question does not have), and (iii) every part fits its field
   where code can tell: a yes/no field takes a yes/no word, a field with allowed values one of them (any case), an
   int/float field a number. Otherwise the reply is unclear and the question is asked again. A part in quotes may
   contain commas. There is NO check that a plain-text part is "a value and not a sentence": the earlier plan-side
   reading never had one (it was an opt-in flag, off for prose), and no structural test separates "hmm, I am not sure"
   from "Jane Austen, Pride and Prejudice" without a word limit that rejects real titles.
f. A reply never fills a field the question did not ask for: a value without a name fills open fields only, and a
   named line sets only a field the question listed.
g. Anything else in the reply is ignored.

Imports no model stack at import time.
"""
from __future__ import annotations

import re
import typing
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from fastworkflow.core import no_agent_mode

#: How a reply reads (``Read.kind``).
DECLINE = "decline"
ACCEPT = "accept"
VALUES = "values"
FREE = "free"

#: The bare replies that say "no value": a value for a yes/no field or an allowed value, else the field is left empty
#: (optional) or the reply is unclear (required).
NO_WORDS = frozenset({"no", "none", "neither", "n", "nope"})
PARAMETER_LINE = re.compile(r"^\s*([A-Za-z_]\w*)\s*[:=]\s*(.*?)\s*$")
_PART = re.compile(r"""\s*(?:"([^"]*)"|'([^']*)'|([^,]*))\s*(,|\Z)""")


@dataclass(frozen=True)
class Slot:
    """What a reply needs to know about one field to read a bare yes or no for it (rule c): ``boolean`` (a yes/no
    field), ``options`` (the values it allows, as declared), ``required`` and ``number`` (an int or float field)."""

    boolean: bool = False
    options: tuple = ()
    required: bool = True
    number: bool = False


@dataclass(frozen=True)
class Read:
    """How a reply read: ``kind`` (``decline``, ``accept``, ``values`` or ``free``) and the ``values`` it gave, by name.
    A value of ``None`` means the field is left empty (rule c). ``named``: the values came from ``name: value`` lines,
    which only an answer is written in (``parameters.reply`` makes no remote read of such a reply)."""

    kind: str
    values: dict = field(default_factory=dict)
    named: bool = False     # the values came from ``name: value`` lines (rule b), not from the reply taken as a value


def slot_of(annotation: Any, required: bool = True) -> Slot:
    """The ``Slot`` of a field with type *annotation* (``bool``, a ``Literal`` or an ``Enum``, ``Optional`` of one
    included). *required*: whether the field must be given."""
    for each in _members(annotation):
        if each is bool:
            return Slot(boolean=True, required=required)
        if each in (int, float):
            return Slot(number=True, required=required)
        if typing.get_origin(each) is typing.Literal:
            return Slot(options=tuple(str(arg) for arg in typing.get_args(each)), required=required)
        if isinstance(each, type) and hasattr(each, "__members__"):
            return Slot(options=tuple(str(getattr(member, "value", member)) for member in each), required=required)
    return Slot(required=required)


def _members(annotation: Any) -> list:
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _members(typing.get_args(annotation)[0])
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", typing.Union)):
        return [member for arg in typing.get_args(annotation) if arg is not type(None) for member in _members(arg)]
    return [annotation]


def slot_of_text(type_text: str, required: bool = True) -> Slot:
    """The ``Slot`` for a field known only by the text of its type (``bool``, ``Literal['a', 'b']``, ``X | None``)."""
    for part in str(type_text or "").split("|"):
        part = part.strip()
        if part == "bool":
            return Slot(boolean=True, required=required)
        if part in ("int", "float"):
            return Slot(number=True, required=required)
        if match := re.fullmatch(r"Literal\[(.*)\]", part):
            return Slot(options=tuple(re.findall(r"""['"]([^'"]*)['"]""", match.group(1))), required=required)
    return Slot(required=required)


def named_values(reply: str, names: Iterable[str]) -> dict[str, str]:
    """The ``name: value`` (or ``name=value``) lines of *reply* that name one of *names*, one per line; the name is
    matched without regard to case and returned as *names* spells it. A line with no value is skipped."""
    known = {str(name).lower(): str(name) for name in names}
    values: dict[str, str] = {}
    for line in str(reply or "").splitlines():
        match = PARAMETER_LINE.match(line)
        if match and match.group(1).lower() in known and match.group(2):
            values[known[match.group(1).lower()]] = match.group(2)
    return values


def split_values(text: str) -> list[str]:
    """*text* cut at its commas, each part trimmed and, when it is wholly in quotes, unquoted (a quoted part may hold
    commas). ``"a, b"`` is two parts, ``"\\"a, b\\", c"`` is two, a trailing comma gives an empty last part."""
    text = str(text or "").strip()
    parts, position = [], 0
    while True:
        match = _PART.match(text, position)
        parts.append(next(group for group in match.groups()[:3] if group is not None).strip())
        if match.group(4) != ",":
            return parts
        position = match.end()


def _unquoted(text: str) -> str:
    """*text* trimmed, and without one pair of quotes that wraps all of it."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'" and text[0] not in text[1:-1]:
        return text[1:-1].strip()
    return text


def read(reply: str, names: Iterable[str], open_names: Iterable[str], *, confirm: bool = False,
         slots: Optional[Mapping[str, Slot]] = None) -> Read:
    """How *reply* reads, strictly and in code (the module docstring is the rule).

    *names* are the fields the question listed (the open ones and the ones it showed with a value), *open_names* the
    ones it asked for, in the order it listed them. *confirm*: the question showed every drafted value and asked to
    confirm, so ``yes`` runs. *slots* gives a field's ``Slot`` (yes/no, allowed values, required) for rule c; a field
    with none is a required plain value.

    The result is ``decline``, ``accept`` (only when *confirm*), ``values`` (by name; ``None`` leaves the field empty),
    or ``free`` (the reply is unclear: the caller asks again or lets an extractor read it)."""
    text = str(reply or "")
    open_names = [str(name) for name in open_names]
    names = [str(name) for name in names]
    word = no_agent_mode._bare(text)
    if word in no_agent_mode._DECLINE:
        return Read(DECLINE)
    if confirm and no_agent_mode.is_yes(text):
        return Read(ACCEPT)
    listed = list(dict.fromkeys([*names, *open_names]))
    if "\n" not in text.strip() and "," in text:
        # several named values on one line: only when every part is a name line of a listed field
        raw = [part.strip() for part in text.split(",")]
        if all(PARAMETER_LINE.match(part) for part in raw) and (values := named_values("\n".join(raw), listed)):
            return Read(VALUES, values, named=True)
    if values := named_values(text, listed):
        return Read(VALUES, values, named=True)
    if not text.strip():
        return Read(FREE)
    if len(open_names) == 1:
        return _lone(open_names[0], _unquoted(text), (slots or {}).get(open_names[0], Slot()))
    if len(open_names) > 1 and "," in text:
        parts = split_values(text)
        if len(parts) == len(open_names) and all(parts) and not any(PARAMETER_LINE.match(part) for part in parts):
            fitted = [_fit(part, (slots or {}).get(name, Slot())) for name, part in zip(open_names, parts)]
            if all(fit is not None for fit in fitted):
                return Read(VALUES, dict(zip(open_names, fitted)))
    return Read(FREE)


def _lone(name: str, value: str, slot: Slot) -> Read:
    """The reading of a reply that is the value of the one open field (rules c and d)."""
    word = no_agent_mode._bare(value)
    if not value:
        return Read(FREE)
    if slot.boolean:
        return Read(VALUES, {name: value})
    for option in slot.options:
        if word == option.lower():
            return Read(VALUES, {name: option})
    if word in NO_WORDS:
        return Read(VALUES, {name: None}) if not slot.required else Read(FREE)
    return Read(VALUES, {name: value})


def fits_constrained(reply: str, open_names: Iterable[str], slots: Optional[Mapping[str, Slot]]) -> bool:
    """True when *reply* is, in code, a valid value of every open field and each of those fields is constrained (a yes/no,
    allowed values or a number): the one open field takes the whole reply, several take its comma parts in order. A free
    text field never fits (so a reply like "hold on" is still read by the remote model)."""
    names = [str(name) for name in open_names]
    if not names or not slots or any(not (slot := slots.get(name)) or not (slot.boolean or slot.options or slot.number)
                                     for name in names):
        return False
    text = _unquoted(str(reply or ""))
    parts = [text] if len(names) == 1 else split_values(text)
    return len(parts) == len(names) and all(
        part and _fit(part, slots[name]) is not None for name, part in zip(names, parts))


#: The words a yes/no FIELD takes as its value (a value, not consent: a yes that consents is ``no_agent_mode.is_yes``).
YES_NO_VALUES = frozenset({"yes", "y", "true", "no", "n", "false", "nope", "none", "neither"})
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def _fit(part: str, slot: Slot) -> Optional[str]:
    """*part* as the value of the field *slot* describes, or None when it cannot be one (rule e, iii)."""
    word = no_agent_mode._bare(part)
    if slot.boolean:
        return part if word in YES_NO_VALUES else None
    if slot.options:
        return next((option for option in slot.options if word == option.lower()), None)
    if slot.number:
        return part if _NUMBER.fullmatch(part.strip()) else None
    return part
