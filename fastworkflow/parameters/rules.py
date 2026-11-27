"""Pure helpers both argument guards share (bd fix-qoyz step 1: a move, no behaviour change).

``planning_commands.check_arguments`` (planning command side) and
``utils.argument_checks.check_extracted_values`` (execution command side) read the request with these
functions. They were defined in one of the two modules and imported by the other through a cycle
(``argument_checks`` imports ``planning_commands``); they live here so neither depends on the other.
The old modules keep their old names, bound to the same objects.

Every function is a pure function of its arguments. This module imports no model stack and neither
guard module.

Where the two sides read the same thing differently the note in a comment names the difference; step 1
merges none of them (design note section 2, rows 7, 8, 12).
"""
from __future__ import annotations

import re
import typing
from typing import Any, List, Optional, Tuple
from fastworkflow.utils.python_utils import is_union

#: Number words the request may write instead of digits (0 to 20). Read by ``written_numbers`` only (the planning
#: check reaches it through ``written_numbers``).
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}

#: A whole number the request writes: digits (with thousands separators), not part of a word or code.
#: A digit run followed by a dot and a digit is a decimal, version or address piece, not a whole number.
_DIGITS = re.compile(r"(?<![\w.])(-?)((?>\d{1,3}(?:,\d{3})+|\d+))(?!\w|\.\d)")

#: Tens words that join a unit ("twenty five"); ``twenty`` is also in ``NUMBER_WORDS``.
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
#: Words that make a stretch of number words unreadable as one whole number: the scale words and the vague ones.
_UNJOINABLE = ("hundred", "thousand", "million", "billion", "dozen", "half")
_WORDS = "|".join(sorted([*NUMBER_WORDS, *_TENS, *_UNJOINABLE], key=len, reverse=True))
#: A stretch of number words that sit together: separated by a space or a hyphen only, or by "and" right after a
#: scale word ("one hundred and five").
_STRETCH = re.compile(
    rf"\b(?:{_WORDS})(?:(?:[ \-]|(?<=hundred) and |(?<=thousand) and |(?<=million) and |(?<=billion) and )(?:{_WORDS}))*\b",
    re.IGNORECASE)


def _stretch_number(stretch: str) -> Optional[int]:
    """The ONE whole number a stretch of number words writes, or None. The grammar is: one word of ``NUMBER_WORDS``
    or a tens word alone, or a tens word (twenty to ninety) then a unit (zero excluded, 1 to 9), joined by a space or
    a hyphen. Anything else ("one hundred", "two dozen", "two three", "twenty ten") is not read at all."""
    words = re.split(r"[ \-]", stretch.lower())
    if len(words) == 1:
        return NUMBER_WORDS.get(words[0], _TENS.get(words[0]))
    if len(words) == 2 and words[0] in _TENS and words[1] in NUMBER_WORDS and 1 <= NUMBER_WORDS[words[1]] <= 9:
        return _TENS[words[0]] + NUMBER_WORDS[words[1]]
    return None

#: A value that is ENTIRELY one pair of braces, possibly doubled (``{x}``, ``{{x}}``); group 1 is the text inside, judged
#: by the same shape as a bracketed value's (``placeholder_inner``). A value that merely contains a brace is a value, so
#: is ``{"a": 1}``, and an upper-case id with a number (``SKU_42``) is a value (bd fix-ftdl.6.1).
_BRACED = re.compile(r"\{+\s*([^{}]*?)\s*\}+")

#: A value that is ENTIRELY one bracketed token, possibly doubled (``<<x>>``); group 1 is the text inside.
_BRACKETED = re.compile(r"<+\s*([^<>]*?)\s*>+")

#: Letters joined by ``_`` or ``-``, no digit, no space: the shape of a variable name (matched on lower-cased text).
IDENTIFIER_SHAPE = re.compile(r"[a-z]+(?:[_\-][a-z]+)*")

#: Dashes folded to "-" when an argument is looked up in the request (non-breaking hyphen etc.).
_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"})


def clean_str(value) -> str:
    """*value* as text without surrounding whitespace, matching quote pairs and trailing ``" .;,"``."""
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text.rstrip(" .;,")


def fold(text: str) -> str:
    """``text`` with each Unicode dash as "-" (one character for one: positions are kept)."""
    return text.translate(_DASHES)


def raw_position(text: str, folded: str) -> int:
    """Where *text* first occurs in the folded request *folded* as plain characters (dashes and case ignored), even
    inside a longer word; -1 when it does not. Read only to widen an id cut short (``planning_commands._whole_identifier``)."""
    return folded.find(fold(text).lower()) if text else -1


def _bounded(text: str, folded: str):
    """The positions where the folded request *folded* writes *text* at word boundaries (``written_at``)."""
    needle = fold(text).lower() if text else ""
    at = folded.find(needle) if needle else -1
    while at >= 0:
        end = at + len(needle)
        if not (needle[0].isalnum() and at > 0 and folded[at - 1].isalnum()) and \
                not (needle[-1].isalnum() and end < len(folded) and folded[end].isalnum()):
            yield at
        at = folded.find(needle, at + 1)


def written_at(text: str, folded: str) -> int:
    """Where the folded request *folded* writes *text* (dashes and case ignored) at word boundaries, or -1: the value's
    own first and last characters do not continue into a letter or digit of the request, so "dun" is not written in
    "dune" and "Ann" not in "Joanna" (bd fix-ftdl.6.6). A value that begins or ends with another character ("#W123")
    is bounded by that character itself."""
    return next(_bounded(text, folded), -1)


def written_count(text: str, folded: str) -> int:
    """How many times the folded request *folded* writes *text* at word boundaries, the occurrences not overlapping."""
    count, free = 0, 0
    for at in _bounded(text, folded):
        if at >= free:
            count, free = count + 1, at + len(text)
    return count


def written(text: str, folded: str, request: str) -> str | None:
    """*text* as the request writes it at word boundaries (its own spelling, dashes and case ignored), or None."""
    at = written_at(text, folded)
    return request[at:at + len(text)] if at >= 0 else None


def placeholder_inner(inner: str) -> bool:
    """Whether *inner*, the text between the brackets of a wholly bracketed value, names a value instead of being one:
    empty or punctuation only once its edges are stripped of what is not a letter or digit, or a variable name in one
    case (``name``, ``code``, ``repo_uid``, ``ID``, ``UNKNOWN``, the extractor's not-given marker). The shape alone:
    mixed case, a digit or a space makes it a value (``Dune``, ``9a1b``, ``Ann Lee``)."""
    core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", inner)
    if not core or core.lower() == "unknown":
        # ``<UNKNOWN>`` in any case is the extractor's not-given marker (fix-1pfu), never a value
        return True
    return (core.islower() or core.isupper()) and IDENTIFIER_SHAPE.fullmatch(core.lower()) is not None


def bracketed_inner(text: str) -> Optional[str]:
    """The text inside *text* when *text* is wholly one bracketed token (``<x>``, ``<<x>>``), else None."""
    match = _BRACKETED.fullmatch(text.strip())
    return match[1] if match else None


def placeholder_like(text: str) -> bool:
    """THE template-token verdict of a value (bd fix-ftdl.6.1): whether *text* is ENTIRELY one template token, a variable
    name in braces or angle brackets (``{name}``, ``{{repo_uid}}``, ``<name>``, ``<ID>``), the inner text judged by its
    shape alone (``placeholder_inner``). Nothing else is judged: there is no list of stand-in words, the request is not
    read, a value that merely contains a bracket is a value, ``{"a": 1}``, ``{Ann Lee}`` and ``{x1}`` are values left as
    written, and ``<Dune>`` is the value ``Dune`` in brackets (``utils.signatures.classify_bracketed_value`` unwraps angle
    brackets only)."""
    text = text.strip()
    braced = _BRACED.fullmatch(text)
    inner = braced[1] if braced else bracketed_inner(text)
    return inner is not None and placeholder_inner(inner)


def written_numbers(request: str, texts: List[str]) -> set:
    """The whole numbers *request* writes (digits, with thousands separators and a minus sign, or number
    words), outside the text of *texts*: a whole number or nothing. Number words that sit together are ONE number
    (``_stretch_number``) and a decimal or dotted token is none. Shared by the execution check and the planning check
    (``planning_commands._request_numbers``)."""
    lowered = request.lower()
    taken: List[Tuple[int, int]] = []
    for text in texts:
        text = text.lower()
        at = lowered.find(text) if text else -1
        while at >= 0:
            taken.append((at, at + len(text)))
            at = lowered.find(text, at + 1)
    found = set()
    for match in _DIGITS.finditer(request):
        if any(a <= match.start(2) < b for a, b in taken):
            continue
        parts = [match[2].replace(",", "")] + (match[2].split(",") if "," in match[2] else [])
        for part in parts:
            number = int(part)
            found.add(number)
            if match[1]:
                found.add(-number)
    for match in _STRETCH.finditer(request):
        if not any(a <= match.start() < b for a, b in taken) and (number := _stretch_number(match[0])) is not None:
            found.add(number)
    return found


def field_extra(field_info: Any) -> dict:
    """``json_schema_extra`` of a Pydantic field when it is a dict, else ``{}``."""
    extra = getattr(field_info, "json_schema_extra", None)
    return extra if isinstance(extra, dict) else {}


def field_pattern(field_info: Any) -> Optional[str]:
    """The regular expression a field declares with ``Field(pattern=...)``, or None."""
    return next((meta.pattern for meta in getattr(field_info, "metadata", None) or () if getattr(meta, "pattern", None)), None)


def _pattern_text(pattern: Any) -> str:
    """A declared pattern as text (a compiled one by its source)."""
    return str(getattr(pattern, "pattern", pattern))


def pattern_sentence(field_info: Any) -> str:
    """The sentence the extraction prompt carries for a field that declares a pattern (empty without one): the format
    the model is told the value must match. The one wording of both kinds of command."""
    pattern = field_pattern(field_info)
    return f"Must match the pattern {_pattern_text(pattern)}" if pattern else ""


def pattern_refusal(name: str, field_info: Any, value: Any) -> Optional[str]:
    """THE meaning of ``Field(pattern=...)`` on every path (bd fix-ftdl.6.4): a given text value must match the WHOLE
    pattern (``re.fullmatch``; a compiled pattern keeps its flags). The refusal, naming the field, the value, the pattern
    and the field's first example when it has one; None when the value matches, is not text, is empty or the field
    declares no pattern."""
    pattern = field_pattern(field_info)
    if pattern is None or not isinstance(value, str) or not value or re.fullmatch(pattern, value):
        return None
    examples = getattr(field_info, "examples", None) or []
    example = f" (e.g., {examples[0]})" if examples else ""
    return f"{name} {value!r} does not match the pattern {_pattern_text(pattern)}{example}"


def literal_values(annotation: Any) -> Tuple[str, ...]:
    """The text values the type *annotation* allows: the members of a ``Literal`` of text, alone or inside an
    ``Optional`` (the ONE way a field declares its allowed values). Empty for every other type, and for a ``Literal``
    of anything but text (an ``int`` or ``bool`` one is parsed by its type as before)."""
    members = [a for a in typing.get_args(annotation) if a is not type(None)] if is_union(annotation) else [annotation]
    if len(members) != 1 or typing.get_origin(members[0]) is not typing.Literal:
        return ()
    values = typing.get_args(members[0])
    return tuple(values) if values and all(isinstance(v, str) for v in values) else ()


def allowed_member(allowed: Tuple[str, ...], text: str) -> Optional[str]:
    """The allowed-values rule, ONE for both kinds of command: the member of *allowed* that *text* is, ignoring case
    and surrounding or repeated whitespace, in its declared spelling; None when it is none of them (refused)."""
    folded = " ".join(text.lower().split())
    return next((v for v in allowed if " ".join(v.lower().split()) == folded), None)


def not_allowed_why(allowed: Tuple[str, ...]) -> str:
    """The refusal of a value outside *allowed*, on either path."""
    return f"is not one of {', '.join(allowed)}"
