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
from typing import Any, List, Optional, Tuple

#: Number words the request may write instead of digits (0 to 20). Read by ``written_numbers`` here and
#: by the planning side's number handling in ``planning_commands``.
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}

#: ``One of: a, b, c`` (or ``a, b or c``) in a field description: the allowed values, identifier-shaped.
_ONE_OF = re.compile(r"\bone of\s*:\s*(?P<items>[^.;\n]+)", re.IGNORECASE)
_ONE_OF_SEPARATOR = re.compile(r"\s*(?:,|\bor\b)\s*")
_ONE_OF_ITEM = re.compile(r"[A-Za-z0-9][\w-]*")
#: A whole number the request writes: digits (with thousands separators), not part of a word or code.
_DIGITS = re.compile(r"(?<![\w.])(-?)(\d{1,3}(?:,\d{3})+|\d+)(?![\w])")
_NUMBER_WORD = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)

#: Words of an invented example value ("John Doe"): a closed list.
PLACEHOLDER_WORDS = frozenset({
    "john", "jane", "doe", "smith", "foo", "bar", "baz", "example", "placeholder", "unknown",
    "unspecified", "tbd", "todo", "n/a", "none", "null"})
#: A template token written as is: brackets and braces, an upper-case ``NAME_1``, "unspecified_x".
_PLACEHOLDER_SHAPE = re.compile(r"[<>{}]|\b[A-Z][A-Z_]*_\d+\b|(?i:\bunspecified\w*|\bplaceholder\b)")

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


def written(text: str, folded: str, request: str) -> str | None:
    """*text* as the request writes it (its own spelling, dashes and case ignored), or None."""
    at = folded.find(fold(text).lower()) if text else -1
    return request[at:at + len(text)] if at >= 0 else None


def placeholder_like(text: str, folded_request: str) -> bool:
    """A template token or an invented example value. The example words count only when the
    request does not itself write the value (a real person named "John Doe" passes)."""
    if _PLACEHOLDER_SHAPE.search(text):
        return True
    words = re.findall(r"[a-z/]+", text.lower())
    return bool(words) and any(w in PLACEHOLDER_WORDS for w in words) and text.lower() not in folded_request


def written_numbers(request: str, texts: List[str]) -> set:
    """The whole numbers *request* writes (digits, with thousands separators and a minus sign, or number
    words), outside the text of *texts*. Shared by the execution check and the planning check
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
    for match in _NUMBER_WORD.finditer(request):
        if not any(a <= match.start() < b for a, b in taken):
            found.add(NUMBER_WORDS[match[1].lower()])
    return found


def field_extra(field_info: Any) -> dict:
    """``json_schema_extra`` of a Pydantic field when it is a dict, else ``{}``."""
    extra = getattr(field_info, "json_schema_extra", None)
    return extra if isinstance(extra, dict) else {}


def allowed_values(field_info: Any) -> Tuple[str, ...]:
    """The values a ``str`` field allows when its type does not say: ``json_schema_extra["enum"]`` (a
    list of text), else a ``One of: a, b, c`` list in its description (two or more identifier-shaped
    items; anything else there is prose, not a list). Empty when the field declares neither."""
    declared = field_extra(field_info).get("enum")
    if isinstance(declared, (list, tuple)) and declared and all(isinstance(v, str) for v in declared):
        return tuple(declared)
    match = _ONE_OF.search(getattr(field_info, "description", None) or "")
    if not match:
        return ()
    items = [item.strip().strip("'\"`") for item in _ONE_OF_SEPARATOR.split(match["items"].strip())]
    items = [item for item in items if item]
    if len(items) < 2 or not all(_ONE_OF_ITEM.fullmatch(item) for item in items):
        return ()
    return tuple(items)


def allowed_member(allowed: Tuple[str, ...], text: str) -> Optional[str]:
    """The allowed-values rule, shared by both checkers (fix-48np.3): the member of *allowed* that *text* is,
    ignoring case, in its declared spelling; None when *text* is none of them (the value is refused).

    Not shared: a planning ``Literal``/enum argument is compared lower-cased against ``p.values`` and
    stored lower-case (``planning_commands.check_arguments``, the ``p.type == "enum"`` branch), while an
    execution ``Literal`` takes the one case-insensitive match in its declared spelling
    (``signatures._try_coerce_literal``). Both stay where they are in step 1 (design note, row 7)."""
    folded = text.lower()
    return next((v for v in allowed if v.lower() == folded), None)


def not_allowed_why(allowed: Tuple[str, ...]) -> str:
    """The refusal of a value outside *allowed*, on either path."""
    return f"is not one of {', '.join(allowed)}"
