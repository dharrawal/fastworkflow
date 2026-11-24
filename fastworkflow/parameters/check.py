"""The one guard module: every rule that refuses a made-up value, once (bd fix-qoyz step 3).

Two modules used to guard extracted values against the request: ``planning_commands.check_arguments`` for a
planning command and ``utils.argument_checks.check_extracted_values`` for an execution command. Both are now thin
callers of this module:

- ``check_plan`` is the planning command's guard (a ``PlanningCommandCommand``'s params, in order, stopping at the
  first refusal, as ``check_arguments`` always did);
- ``check_execution`` is the execution command's screen of the values the LLM gave (an ``Input`` class, every refusal
  collected): the screens that need no knowledge of the request, the allowed values of a ``Literal``. The strict check
  of a marked field is ``check_plan``'s on both kinds (reached for an execution command through
  ``planning_commands.option_command``): ONE guard per marked field, the same function (bd fix-ftdl.6.3).

They share the rule functions below, and ONE set of rules: there is no switch between the two kinds (bd fix-qoyz
step 6). A value the extractor did not give (a default, ``NOT_FOUND``, the unknown marker) is not guarded. The
extractor's value of a ``str`` or ``int`` STANDS unless its field is marked (``copied_from_command_text`` or
``record_named_in_command_text``, ``is_marked``): only a marked field must be written in the request (a ``str`` as the
request writes it at word boundaries, an ``int`` a number it writes, with or without any number in the request; an int
is never replaced), required or optional; a marked value the extractor did not give is not checked. Two marked fields
may hold one value only when the user writes it as often (the one duplicate rule: an optional one beyond that count is
left not given with a note, a required one is refused). Checked on every ``str`` field:
a value that is entirely one template token, a value of only punctuation. The unmarked fields whose value the request
does not write are only recorded (``not_written_fields``, an advisory on the span).

The rules and where they live:

==========================  ==========================================  ================================
rule                        function                                    notes
==========================  ==========================================  ================================
quotes, brackets, trailing  ``rules.clean_str`` and the screens
punctuation
a whole template token      ``rules.placeholder_like``, through         every str field, every path
                            ``signatures.classify_bracketed_value``
a marked str / int must be  ``is_marked``, ``ground_rule``              only a marked field, at word
written in the request                                                  boundaries; no default
two marked fields hold one  ``check_plan`` (the duplicate rule)         only marked fields
value written once
a value of only punctuation ``_execution_str`` / ``check_plan``         every field
a marked int the request    ``number_rule``                             only a marked field
does not write
an unmarked value the       ``not_written_fields``                      advisory only, refuses nothing
request does not write
which values are checked    ``is_given`` / ``is_given_value``           only a value the extractor gave
==========================  ==========================================  ================================

A refusal that names no input is not asked on either kind of command (the text planner or the agent takes over; an
execution command ends as a failed output): that is the callers' business, no guard decides it.

Plan-only rules (subjects and pronouns, identifier expansion, kind and vocabulary words, classifier options,
roles, ``ignore`` words, list items) stay as helpers of ``planning_commands`` and are called from ``check_plan``;
an execution ``Input`` has no command, vocabulary or request subjects to read them with. A field of an execution
command that carries such an option is checked by ``check_plan`` through ``planning_commands.option_command``.

This module imports no model stack at import time: what it needs of ``utils.signatures`` and
``planning_commands`` is read inside the functions.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, get_origin

from fastworkflow.parameters.rules import (
    allowed_member,
    literal_values,
    clean_str,
    field_extra,
    fold,
    not_allowed_why,
    placeholder_like,
    raw_position,
    written,
    written_at,
    written_count,
    written_numbers,
)

#: The refusal reasons (``why``) of a value. Their texts are read back by the question wording
#: (``planning_commands.refusal_question``) and by the span diagnostics, so they are part of the contract.
NOT_WRITTEN_WHY = "is not written in the request"
BRACKET_PLACEHOLDER_WHY = "is a placeholder, not a value"
NO_VALUE_WHY = "is punctuation, not a value"
#: The refusals of a free-text value that the question asks for without quoting it.
PLACEHOLDER_WHY = "is a placeholder, not a value"
#: The refusal of one of two marked fields that hold one value the request writes fewer times (the duplicate rule).
ALSO_VALUE_OF_WHY = "is also the value of {other}"

#: ``Gap.state``.
MISSING = "missing"
INVALID = "invalid"


@dataclass(frozen=True)
class Gap:
    """One argument a guard refused (or found missing): ``name``, its ``state`` (``MISSING`` or ``INVALID``), a
    ``code`` for the rule that refused it, ``why`` (the reason, as the question and the diagnostics word it),
    the ``value`` refused, ``message`` (the whole refusal text a caller raises or shows) and, for a validator's
    own refusal, its ``suggestions``. ``call`` marks a refusal of the call as a whole (a classifier finds no
    such argument in the request) rather than of one value."""

    name: str
    state: Literal["missing", "invalid"]
    code: str
    why: str
    value: Any = None
    message: str = ""
    suggestions: Tuple[str, ...] = ()
    call: bool = False

    @property
    def screened(self) -> str:
        """For a value the guards refused that reads as not given (a pronoun, a word of the workflow's vocabulary, a
        placeholder): the value with the reason (``'it' is a pronoun, not a value``), so the question can say why it was
        not taken; else ''."""
        if self.state != MISSING or self.code != "refused" or self.value in (None, "") or self.why == PLACEHOLDER_WHY:
            return ""
        return f"{self.value!r} {self.why}"


@dataclass
class Checked:
    """What a guard found. ``values``: the guarded values (a plan's complete arguments; empty for an execution
    command, whose values the caller reads from ``respelled`` and ``gaps``). ``respelled``: field -> ``(given,
    the request's or the declared spelling)``. ``notes``: what a guard changed, in words. ``gaps``: the refusals,
    in the order the guards found them (a plan stops at the first, see ``stop``). ``not_in_request``: the advisory
    of an execution check (``not_written_fields``, the field names only; a plan's is read from its arguments)."""

    values: Dict[str, Any] = field(default_factory=dict)
    respelled: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    gaps: List[Gap] = field(default_factory=list)
    not_in_request: List[str] = field(default_factory=list)

    @property
    def stop(self) -> Optional[Gap]:
        """The refusal that ended a plan's check (None when it passed, or for an execution check)."""
        return self.gaps[0] if self.gaps else None

    @property
    def extraction(self):
        """The gaps as the one result contract (``parameter_clarification.Extraction``): the values kept, the names
        missing, the ones refused with why."""
        from fastworkflow.parameter_clarification import Extraction

        return Extraction(values=dict(self.values),
                          missing=tuple(g.name for g in self.gaps if g.state == MISSING),
                          invalid=tuple((g.name, g.why) for g in self.gaps if g.state == INVALID),
                          screened=tuple((g.name, g.screened) for g in self.gaps if g.screened))


# ---------------------------------------------------------------------------- the rules, one function each

def ground_rule(text: str, folded_request: str, request: str) -> Optional[str]:
    """*text* as the request writes it at word boundaries (its own spelling; case and dashes ignored), or None."""
    return written(text, folded_request, request)


def numbers_why(numbers) -> str:
    if not numbers:
        return "is not a number the request writes"
    return f"is not a number the request writes ({', '.join(map(str, sorted(numbers)))})"


def number_rule(number: int, numbers) -> Tuple[str, Any]:
    """The verdict on a MARKED int against the numbers the request writes: ``("ok", None)``, or ``("refuse", why)``
    for a number the request does not write (never replaced by one it does), whether or not it writes any number."""
    if number in numbers:
        return "ok", None
    return "refuse", numbers_why(numbers)


def state_of_why(why: str) -> str:
    """``MISSING`` for a refusal that reads as the value not being given (a placeholder, a pronoun, a vocabulary
    word), else ``INVALID`` (as ``planning_commands`` words the question)."""
    from fastworkflow import planning_commands as pc
    from fastworkflow.utils.workflow_vocabulary import KIND_WORD_WHY

    return MISSING if why in (PLACEHOLDER_WHY, pc.PRONOUN_WHY, KIND_WORD_WHY) else INVALID


# ---------------------------------------------------------------------------- the field readers (execution side)

def is_marked(field_info: Any) -> bool:
    """Whether a field carries ``copied_from_command_text`` or ``record_named_in_command_text`` (any form of the
    option): the one explicit strict check. Nothing else (a ``Field(pattern=...)``, a description example, the type)
    makes a value be compared with the request."""
    from fastworkflow.field_options import COPIED_FROM_COMMAND_TEXT, RECORD_NAMED_IN_COMMAND_TEXT

    extra = field_extra(field_info)
    return COPIED_FROM_COMMAND_TEXT in extra or RECORD_NAMED_IN_COMMAND_TEXT in extra


def is_str_field(field_info: Any) -> bool:
    from fastworkflow.utils.signatures import is_optional_str_field

    return field_info.annotation is str or is_optional_str_field(field_info)


def is_int_field(field_info: Any) -> bool:
    from fastworkflow.utils.signatures import _numeric_kind

    annotation = field_info.annotation
    if annotation is bool or (isinstance(annotation, type) and issubclass(annotation, Enum)):
        return False
    if get_origin(annotation) is Literal:
        return False
    return _numeric_kind(annotation) is int


def is_given_value(value: Any, default: Any) -> bool:
    """Whether *value* of a planning argument is a value the extractor gave: not its way of saying "not given" (the
    ``NOT_FOUND`` text, the ``<UNKNOWN>`` marker, its "not given" number) and not the argument's default."""
    from fastworkflow.utils.signatures import NOT_FOUND, is_sentinel_number, is_unknown_marker

    if isinstance(value, str) and (is_unknown_marker(value) or value.strip() == NOT_FOUND):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool) and is_sentinel_number(value):
        return False   # the extractor's "not given" number
    if isinstance(value, str) and isinstance(default, (bool, int, float)) and \
            value.strip().lower() == str(default).lower():
        return False   # the default as text (a review input holds every value as text): "0" for 0, "false" for False
    return value != default


def is_given(field_info: Any, value: Any) -> bool:
    """Whether *value* is a value the extractor gave, not its way of saying "not given"."""
    from fastworkflow.utils.signatures import NOT_FOUND, _is_filled, is_unknown_marker

    if isinstance(value, str) and (is_unknown_marker(value) or value.strip() in ("", "None", NOT_FOUND)):
        return False
    return _is_filled(field_info, value)


def as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------- the execution command's guard

def _execution_str(name: str, value: Any, request: str, out: Checked) -> None:
    """Screen one given ``str`` value for what needs no knowledge of the request: a whole template token (the one
    verdict, ``utils.signatures.classify_bracketed_value``, which ``validate_parameters`` and a plan's check also read),
    only punctuation. Nothing is compared with the request here."""
    from fastworkflow import planning_commands as pc
    from fastworkflow.utils.signatures import NOT_FOUND, classify_bracketed_value, unwrap_quoted_value

    def refuse(why: str, code: str) -> None:
        out.gaps.append(Gap(name, INVALID, code, why, value, f"{name}={value!r} {why}"))

    single = pc._single(value)
    if not isinstance(single, str) or isinstance(single, Enum):
        return
    _, text = unwrap_quoted_value(single, request)
    kind, inner = classify_bracketed_value(text)
    if kind == "placeholder":
        refuse(BRACKET_PLACEHOLDER_WHY, "placeholder")
        return
    text = clean_str(inner)
    if not text.strip():
        # only quotes or punctuation (``...``): nothing of the request, and no value
        refuse(NO_VALUE_WHY, "no_value")
        return
    if text.strip() in ("None", NOT_FOUND):
        return


def _execution_allowed(name: str, allowed: Tuple[str, ...], value: Any, out: Checked) -> None:
    """Check one given ``Literal`` text value: a member takes its declared spelling (recorded as a respelling); a value
    outside them refuses THAT field only, naming the allowed values (the other fields of the answer keep theirs)."""
    text = value if isinstance(value, str) else None
    member = allowed_member(allowed, text) if text is not None else None
    if member is None:
        why = not_allowed_why(allowed)
        out.gaps.append(Gap(name, INVALID, "not_allowed", why, value, f"{name}={value!r} {why}"))
    elif member != value:
        out.respelled[name] = (value, member)


def check_execution(model: type, values: Mapping[str, Any], request: Optional[str]) -> Checked:
    """Check the values the LLM extractor gave an execution command (*model*: its ``Input`` class, *values*: field
    -> value) against *request*. Reads, changes nothing; with no request text there is nothing to check against,
    and nothing is refused. ``Checked.gaps`` are the refusals in order, ``Checked.respelled`` the respellings.

    The screens that need no knowledge of the request run on every ``str`` field. The strict check of a MARKED field is
    not made here: its one guard is ``check_plan``, which the validation stage runs on the command's marked fields
    (``planning_commands.option_command``), as it runs on a planning command's."""
    out = Checked()
    if not request or not request.strip():
        return out
    for name, field_info in model.model_fields.items():
        value = values.get(name)
        if not is_given(field_info, value):
            continue
        if allowed := literal_values(field_info.annotation):
            _execution_allowed(name, allowed, value, out)
        elif is_str_field(field_info):
            _execution_str(name, value, request, out)
    refused = {gap.name for gap in out.gaps}
    out.not_in_request = [n for n in not_written_fields(model, values, request) if n not in refused]
    return out


def not_written_fields(model: type, values: Mapping[str, Any], request: Optional[str]) -> List[str]:
    """The advisory: the names (sorted, never the values) of the UNMARKED ``str`` and ``int`` fields of *model* whose
    given value in *values* the request does not write (the test the copy check applies to a marked field: a ``str`` as
    ``ground_rule`` reads it, an ``int`` a number ``written_numbers`` does not find). Refuses nothing and calls no model;
    a field the extractor did not give, a ``Literal`` and a marked field are not looked at. The one function of both
    kinds of command: an execution check reads it on the extracted values, a planning call on its checked arguments."""
    from fastworkflow.utils.signatures import is_sentinel_number

    if not request or not request.strip():
        return []
    folded = fold(request).lower()
    names: List[str] = []
    texts: List[str] = []
    numbers: List[Tuple[str, int]] = []
    for name, field_info in model.model_fields.items():
        value = values.get(name)
        if is_marked(field_info) or literal_values(field_info.annotation) or not is_given(field_info, value):
            continue
        if is_str_field(field_info):
            if isinstance(value, str):
                text = clean_str(value)
                texts.append(text)
                if text and ground_rule(text, folded, request) is None:
                    names.append(name)
        elif is_int_field(field_info):
            number = as_int(value)
            if number is not None and not is_sentinel_number(number):
                numbers.append((name, number))
    if numbers:
        written_here = written_numbers(request, texts)
        names += [name for name, number in numbers if number not in written_here]
    return sorted(names)


# ---------------------------------------------------------------------------- the planning command's guard

class _Stop(Exception):
    """Internal: a plan check stops at its first refusal (carries the ``Gap``)."""

    def __init__(self, gap: Gap):
        super().__init__(gap.message)
        self.gap = gap


def check_plan(command, raw: dict, request: str, verdicts: Optional[dict] = None, vocabulary=None) -> Checked:
    """The guarded, complete arguments of a planning command call (``Checked.values``, with ``Checked.notes``).

    The rules are those documented on ``planning_commands.check_arguments``; a required argument the guards refuse
    is ``Checked.stop`` (a ``Gap``: ``call`` says the check finds no such argument in the request at all)."""
    from fastworkflow import planning_commands as pc
    from fastworkflow.utils.signatures import is_sentinel_number
    from fastworkflow.utils.workflow_vocabulary import KIND_WORD_WHY, is_kind_word, is_pronoun_phrase

    out = Checked()

    def stop(p, why: str, shown=None, *, code: str, message: Optional[str] = None, state: Optional[str] = None,
             call: bool = False):
        text = message if message is not None else f"{p.name}={shown!r} {why}"
        raise _Stop(Gap(p.name, state or (MISSING if code == "missing" else INVALID), code, why, shown, text,
                        call=call))

    try:
        _check_plan_values(command, raw, request, verdicts, vocabulary, out, stop, pc, is_sentinel_number,
                           KIND_WORD_WHY, is_kind_word, is_pronoun_phrase)
    except _Stop as refused:
        out.gaps.append(refused.gap)
    return out


def _check_plan_values(command, raw, request, verdicts, vocabulary, out, stop, pc, is_sentinel_number,
                       KIND_WORD_WHY, is_kind_word, is_pronoun_phrase) -> None:
    notes, args = out.notes, out.values
    NOT_FOUND_MISSING = "missing required argument"
    # verify "alone": no model fills the field; its default stands until a sure verdict replaces it
    raw = {**raw, **{p.name: p.default for p in command.params if p.alone}}
    folded = fold(request).lower()
    subjects = pc._request_subjects(command, request)
    named = {subject.lower() for subject in subjects}
    if vocabulary is None:
        vocabulary = pc._vocabulary_for(command)
    kinds = frozenset(v.lower() for p in command.params if p.type == "enum" for v in p.values)
    single = {**raw, **{p.name: raw.get(p.name) if p.type == "list[str]" else pc._single(raw.get(p.name))
                        for p in command.params}}
    screened, refused = pc._screen(command, single, request)
    #: the ints that took their default because the extractor left them out
    absent: set = set()

    def refuse(p, value, why: str) -> None:
        if p.required:
            stop(p, why, value, code="refused", state=state_of_why(why))
        notes.append(f"{p.name}={value!r} {why}: default")
        args[p.name] = p.default
        if p.type == "int":
            absent.add(p.name)   # an int that took its default is "not given" and is not checked against the request

    def missing(p) -> None:
        stop(p, "", None, code="missing", message=f"{NOT_FOUND_MISSING} {p.name}")

    for p in command.params:
        shown = raw.get(p.name)   # the model's argument, as the notes show it
        value = screened.get(p.name, single[p.name])
        compared = shown          # what a changed spelling is noted against
        sure = pc._confident(verdicts, p.name) if p.verify else None
        if sure is not None and p.type == "enum" and (pc._empty(value) or allowed_member(p.values, clean_str(value)) is None):
            notes.append(f"{p.name}={shown!r} -> {sure[0]!r} (check {sure[1]:.2f})")
            args[p.name] = sure[0]
            continue
        if sure is not None and p.type == "str":
            chosen = None if sure[0] == pc.NO_SUBJECT else sure[0]
            given = "" if pc._empty(value) or p.name in refused else pc.strip_words(clean_str(value), p.ignore)
            if (chosen or "").lower() != given.lower():
                notes.append(f"{p.name}={shown!r} -> {chosen or p.default!r} (check {sure[1]:.2f})")
            if chosen is None:
                if p.required:
                    stop(p, "", shown, code="no_such_argument", state=INVALID, call=True,
                         message=f"{p.name}={shown!r}: the check finds no {p.name} in the request")
                args[p.name] = p.default
                continue
            value = compared = chosen
        elif p.name in refused:
            refuse(p, shown, refused[p.name])
            continue
        if p.type == "list[str]":
            # no check by type (design 2.46): the elements are cleaned; with in_request, each one the
            # request does not write is dropped, and the request's own spelling of the rest is kept
            items = [text for text in (clean_str(item) for item in pc._list_items(value))
                     if text and text != "None"]
            if p.in_request:
                kept = []
                for item in items:
                    exact = ground_rule(item, folded, request)
                    if exact is None:
                        notes.append(f"{p.name}={item!r} {NOT_WRITTEN_WHY}: dropped")
                    else:
                        kept.append(exact)
                if items and not kept and p.required:
                    stop(p, NOT_WRITTEN_WHY, shown, code="not_written")
                items = kept
            if not items and p.required:
                missing(p)
            args[p.name] = items if items else list(p.default)
            continue
        if pc._empty(value) or not is_given_value(shown, p.default):
            if p.required:
                missing(p)
            args[p.name] = p.default
            if p.type == "int":
                absent.add(p.name)
            continue
        if p.type == "bool":
            if isinstance(value, str):
                if value.strip().lower() not in ("true", "false"):
                    stop(p, "is not true or false", shown, code="not_bool")
                value = value.strip().lower() == "true"
            args[p.name] = bool(value)
        elif p.type == "int":
            try:
                number = int(str(value).strip())
            except ValueError:
                stop(p, "is not an integer", shown, code="not_int")
            if is_sentinel_number(number):   # the extractor's "not given" marker or a made-up extreme
                if p.required:
                    missing(p)
                number = p.default
                absent.add(p.name)
            args[p.name] = number
        elif p.type == "float":
            try:
                number = float(str(value).strip())
            except ValueError:
                stop(p, "is not a number", shown, code="not_float")
            if not math.isfinite(number) or is_sentinel_number(number):   # the extractor's "not given" marker
                if p.required:
                    missing(p)
                number = p.default
            args[p.name] = number
        elif p.type == "enum":
            # the rule of an execution command's ``Literal`` (``allowed_member``): the declared spelling, or refused
            member = allowed_member(p.values, clean_str(value))
            if member is None:
                stop(p, not_allowed_why(p.values), shown, code="not_enum")
            args[p.name] = member
        else:
            text = pc.strip_words(clean_str(value), p.ignore)
            if not text.strip():
                refuse(p, shown, NO_VALUE_WHY)
                continue
            if text.strip() == "None":
                # the missing-value marker, not a value: the argument is absent (as the parameter
                # screen treats it), so a required one is missing and an optional one takes its default
                if p.required:
                    missing(p)
                args[p.name] = p.default
                continue
            if placeholder_like(text):
                # the one template-token verdict of both kinds (``_screen`` gave it on the raw value), on the value
                # this check uses: a classifier's choice, or the value with its ``ignore`` words removed
                refuse(p, shown, PLACEHOLDER_WHY)
                continue
            if text and text.lower() not in named:
                # the free-text screens: a pronoun, a word of the workflow's own vocabulary. A value the request
                # names as a record (a subject) always passes them.
                if is_pronoun_phrase(text):
                    if len(subjects) != 1:
                        refuse(p, shown, pc.PRONOUN_WHY)
                        continue
                    notes.append(f"{p.name}={shown!r} -> {subjects[0]!r} (the request's one subject)")
                    text = subjects[0]
                elif is_kind_word(text, p.name, vocabulary):
                    refuse(p, shown, KIND_WORD_WHY)
                    continue
            if not p.in_request:
                # no request check without the option (design 2.46): the value as the model wrote it, after the
                # free-text screens above (a placeholder, a pronoun, a word of the workflow's vocabulary), which
                # refuse whatever the field's options
                args[p.name] = text
                continue
            # an id the model cut short ("4421" of "acct_4421") is widened to the one identifier match that holds it,
            # and a record name cut short is completed by the subject rule below (``_as_subject``, which reads whole
            # words itself); any other value must be written at word boundaries ("dun" is not written in "dune"). A
            # marked value given and not written is refused, required or optional (bd fix-ftdl.6.3).
            raw_at = raw_position(text, folded)
            whole, cut = (pc._whole_identifier(request, request[raw_at:raw_at + len(text)], command.identifiers)
                          if raw_at >= 0 else (None, False))
            if cut and whole is None:
                refuse(p, shown, "is part of more than one identifier the request writes")
                continue
            if whole is None:
                at = raw_at if p.subject else ground_position(text, folded)
                if at < 0:
                    stop(p, NOT_WRITTEN_WHY, shown, code="not_written")
                whole = request[at:at + len(text)]
            exact = whole
            if p.subject:
                subject, why = pc._as_subject(request, exact, subjects, kinds)
                if subject is None:
                    refuse(p, shown, why)
                    continue
                exact = subject
            if exact != str(compared):
                notes.append(f"{p.name}={shown!r} -> {exact!r}")
            args[p.name] = exact
    written_here = None
    for p in command.params:
        if p.type != "int" or not p.in_request:
            continue
        if p.name in absent:   # an int the extractor left out is "not given" and stays so
            continue
        if written_here is None:
            written_here = pc._request_numbers(request, args, command)
        verdict, detail = number_rule(args[p.name], written_here)
        if verdict == "refuse":
            stop(p, detail, args[p.name], code="number", message=f"{p.name}={args[p.name]} {detail}")
    for p in command.params:
        if p.type != "str" or not p.in_request:
            continue
        value = args[p.name]
        if pc._empty(value) or (not p.required and value == p.default):
            continue
        if ground_rule(str(value), folded, request) is None:
            stop(p, NOT_WRITTEN_WHY, raw.get(p.name, value), code="not_written")
    _duplicate_rule(command, args, request, folded, stop, pc, notes)
    for p in command.params:
        sure = pc._confident(verdicts, p.name) if p.verify and p.type != "str" else None
        if sure is not None and args[p.name] != sure[0]:
            notes.append(f"{p.name}={args[p.name]!r} -> {sure[0]!r} (check {sure[1]:.2f})")
            args[p.name] = sure[0]


def _duplicate_rule(command, args: dict, request: str, folded: str, stop, pc, notes: List[str]) -> None:
    """THE duplicate rule, part of the strict check, on both kinds (bd fix-qjne.20): marked ``str`` fields that hold one
    value (case ignored) may all keep it only when the user writes that value as often as the fields hold it ("New York,
    New York" fills a city and a state; ``written_times``). Otherwise the fields beyond that count give it up, one at a
    time: the one chosen is among those whose role the request does not write just before the value
    (``planning_commands._role_fits``; all of them when that tells none apart): the first optional one, else the first.
    An OPTIONAL one chosen is left not given (its default) with a note that says so (``ask.left_empty_line``: on the
    span's notes, the review item and a question), not a refusal; a REQUIRED one chosen, or a value the user does not
    write at all, is refused. Unmarked fields are never compared."""
    from fastworkflow.parameters.ask import left_empty_line

    holders: Dict[str, list] = {}
    for p in command.params:
        value = args.get(p.name)
        if p.type == "str" and p.in_request and not pc._empty(value) and (p.required or value != p.default):
            holders.setdefault(str(value).lower(), []).append(p)
    for value, fields in holders.items():
        times = written_times(value, folded) if len(fields) > 1 else len(fields)
        while len(fields) > times:
            unfit = [p for p in fields if not pc._role_fits(request, str(args[p.name]), p)] or fields
            loser = next((p for p in unfit if not p.required), unfit[0])
            other = next(p for p in fields if p is not loser)
            if loser.required or times == 0:
                stop(loser, ALSO_VALUE_OF_WHY.format(other=other.name), args[loser.name], code="role")
            notes.append(left_empty_line(loser.name, str(args[loser.name]), times, other.name))
            args[loser.name] = loser.default
            fields = [p for p in fields if p is not loser]


#: The text the user wrote for the turn, when the text a check reads is not that text (a split part of it, an agent's
#: own command text): bound by the turn for its plan and agent path (``user_written``), unset on the "/" path, whose
#: checked text is the user's own message.
_USER_WRITTEN: ContextVar[Optional[str]] = ContextVar("fw_user_written_request", default=None)


@contextmanager
def user_written(text: Optional[str]):
    """While the block runs, the duplicate rule counts a value in *text*, the user's own request for the turn (see
    ``written_times``). None or empty binds nothing."""
    token = _USER_WRITTEN.set(text or None)
    try:
        yield
    finally:
        _USER_WRITTEN.reset(token)


def written_times(value: str, folded: str) -> int:
    """How many times the user writes *value* (lowercase) for the duplicate rule: in the checked text *folded* (the
    folded, lowercase request), and no more often than in the user's own request for the turn when one is bound
    (``user_written``) and writes it, so a split part or an agent's text that repeats a name the user wrote once does not
    let two fields hold it. A value the user's request does not write (it came from an earlier step's output) is counted
    in the checked text alone."""
    count = written_count(value, folded)
    user = _USER_WRITTEN.get()
    if user:
        in_user = written_count(value, fold(user).lower())
        if in_user:
            count = min(count, in_user)
    return count


def ground_position(text: str, folded_request: str) -> int:
    """Where *text* starts in the folded request at word boundaries (-1 when the request does not write it)."""
    return written_at(text, folded_request)
