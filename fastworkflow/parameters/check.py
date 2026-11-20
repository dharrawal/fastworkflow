"""The one guard module: every rule that refuses a made-up value, once (bd fix-qoyz step 3).

Two modules used to guard extracted values against the request: ``planning_commands.check_arguments`` for a
planning command and ``utils.argument_checks.check_extracted_values`` for an execution command. Both are now thin
callers of this module:

- ``check_plan`` is the planning command's guard (a ``PlanningCommandCommand``'s params, in order, stopping at the
  first refusal, as ``check_arguments`` always did);
- ``check_execution`` is the execution command's guard (an ``Input`` class and the values the LLM gave, every
  refusal collected).

They share the rule functions below, and ONE set of rules: there is no switch between the two kinds (bd fix-qoyz
step 6). A value the extractor did not give (a default, ``NOT_FOUND``, the unknown marker) is not guarded; a ``str``
or ``int`` must be written in the request unless its field opts out; an int the request does not write is refused
(never replaced); a value of only punctuation, or an unwritten one equal to a description example, is refused.

The rules and where they live:

==========================  ==========================================  ================================
rule                        function                                    notes
==========================  ==========================================  ================================
quotes, brackets, trailing  ``rules.clean_str`` and the screens
punctuation
placeholder token or        ``placeholder_rule``
invented example name
allowed values of a str     ``allowed_rule`` (``rules.allowed_member``) a declared value needs no grounding
grounding of a str / int    ``must_ground``, ``ground_rule``            opt out: converted, or a pattern
in the request
a description example the   ``example_rule``
request does not write
a value of only punctuation ``_execution_str`` / ``check_plan``
an int the request does     ``number_rule``
not write
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, get_origin

from fastworkflow.parameters.rules import (
    allowed_member,
    allowed_values,
    clean_str,
    field_extra,
    fold,
    not_allowed_why,
    placeholder_like,
    written,
    written_numbers,
)

#: The refusal reasons (``why``) of a value. Their texts are read back by the question wording
#: (``planning_commands.refusal_question``) and by the span diagnostics, so they are part of the contract.
NOT_WRITTEN_WHY = "is not written in the request"
EXAMPLE_WHY = "is an example from the field's description, not a value the request writes"
BRACKET_PLACEHOLDER_WHY = "is a placeholder, not a value"
NO_VALUE_WHY = "is punctuation, not a value"
#: The refusals of a free-text value that the question asks for without quoting it.
PLACEHOLDER_WHY = "is a placeholder, not a value"

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


@dataclass
class Checked:
    """What a guard found. ``values``: the guarded values (a plan's complete arguments; empty for an execution
    command, whose values the caller reads from ``respelled`` and ``gaps``). ``respelled``: field -> ``(given,
    the request's or the declared spelling)``. ``notes``: what a guard changed, in words. ``gaps``: the refusals,
    in the order the guards found them (a plan stops at the first, see ``stop``)."""

    values: Dict[str, Any] = field(default_factory=dict)
    respelled: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    gaps: List[Gap] = field(default_factory=list)

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
                          invalid=tuple((g.name, g.why) for g in self.gaps if g.state == INVALID))


# ---------------------------------------------------------------------------- the rules, one function each

def must_ground(*, exempt: bool) -> bool:
    """Whether a ``str`` or ``int`` value must be written in the request: always, unless its field opts out
    (*exempt*: the value is converted from the request, the ``converted_from_command_text`` option, or the field
    declares a ``pattern``). One rule for every command; the copy options (``copied_from_command_text``,
    ``record_named_in_command_text``) add behaviour of their own (a subject must name a record, a list keeps the
    request's items) but no longer decide whether a value is grounded."""
    return not exempt


def placeholder_rule(text: str, folded_request: str) -> Optional[str]:
    """The reason when *text* is a template token or an invented example name, else None."""
    return PLACEHOLDER_WHY if placeholder_like(text, folded_request) else None


def allowed_rule(allowed: Tuple[str, ...], text: str) -> Tuple[Optional[str], str]:
    """``(member, why)``: the declared spelling of the allowed value *text* is, or None with the refusal's reason."""
    member = allowed_member(allowed, text)
    return member, ("" if member is not None else not_allowed_why(allowed))


def ground_rule(text: str, folded_request: str, request: str) -> Optional[str]:
    """*text* as the request writes it (its own spelling; case and dashes ignored), or None."""
    return written(text, folded_request, request)


def example_rule(field_info: Any, text: str) -> bool:
    """Whether an unwritten *text* is one of the field's description ``examples``: refused in its own words
    (``EXAMPLE_WHY``), on both kinds of command."""
    examples = [str(e).lower() for e in (getattr(field_info, "examples", None) or ())]
    return text.lower() in examples


def numbers_why(numbers) -> str:
    return f"is not a number the request writes ({', '.join(map(str, sorted(numbers)))})"


def number_rule(number: int, numbers) -> Tuple[str, Any]:
    """The verdict on an int against the numbers the request writes: ``("ok", None)``, or ``("refuse", why)`` for a
    number the request does not write (never replaced by one it does). With no number written there is nothing to
    check against."""
    if not numbers or number in numbers:
        return "ok", None
    return "refuse", numbers_why(numbers)


def state_of_why(why: str) -> str:
    """``MISSING`` for a refusal that reads as the value not being given (a placeholder, a pronoun, a vocabulary
    word), else ``INVALID`` (as ``planning_commands`` words the question)."""
    from fastworkflow import planning_commands as pc
    from fastworkflow.utils.workflow_vocabulary import KIND_WORD_WHY

    return MISSING if why in (PLACEHOLDER_WHY, pc.PRONOUN_WHY, KIND_WORD_WHY) else INVALID


# ---------------------------------------------------------------------------- the field readers (execution side)

def is_converted(field_info: Any) -> bool:
    """Whether a field carries ``json_schema_extra={"converted_from_command_text": True}``."""
    from fastworkflow.field_options import CONVERTED_FROM_COMMAND_TEXT

    return field_extra(field_info).get(CONVERTED_FROM_COMMAND_TEXT) is True


def is_grounding_exempt(field_info: Any) -> bool:
    """Whether a ``str`` field's value need not be written in the request: it is converted, or declares a ``pattern``."""
    if is_converted(field_info):
        return True
    if "pattern" in field_extra(field_info):
        return True
    return any(getattr(meta, "pattern", None) for meta in getattr(field_info, "metadata", None) or ())


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

def _execution_str(name: str, field_info: Any, value: Any, request: str, folded: str,
                   out: Checked) -> Optional[str]:
    """Check one given ``str`` value; returns the text it stands for when it passes, else None."""
    from fastworkflow import planning_commands as pc
    from fastworkflow.utils.signatures import NOT_FOUND, classify_bracketed_value, unwrap_quoted_value

    def refuse(why: str, code: str) -> None:
        out.gaps.append(Gap(name, INVALID, code, why, value, f"{name}={value!r} {why}"))

    single = pc._single(value)
    if not isinstance(single, str) or isinstance(single, Enum):
        return None
    _, text = unwrap_quoted_value(single, request)
    kind, inner = classify_bracketed_value(text, name)
    if kind == "placeholder":
        refuse(BRACKET_PLACEHOLDER_WHY, "placeholder")
        return None
    text = clean_str(inner)
    if not text.strip():
        # only quotes or punctuation (``...``): nothing of the request, and no value
        refuse(NO_VALUE_WHY, "no_value")
        return None
    if text.strip() in ("None", NOT_FOUND):
        return None

    allowed = allowed_values(field_info)
    if allowed:
        member, why = allowed_rule(allowed, text)
        if member is None:
            refuse(why, "not_allowed")
            return None
        if member != value:
            out.respelled[name] = (value, member)
        return member

    if why := placeholder_rule(text, folded):
        refuse(why, "placeholder")
        return None
    if not must_ground(exempt=is_grounding_exempt(field_info)):
        # A converted value can equal a description example legitimately: "New York" on "May 20"
        # gives origin JFK and date 2024-05-20, the airline search fields' own examples. So a field
        # that is not grounded gets no example check.
        return text
    exact = ground_rule(text, folded, request)
    if exact is None:
        if example_rule(field_info, text):
            refuse(EXAMPLE_WHY, "example")
        else:
            refuse(NOT_WRITTEN_WHY, "not_written")
        return None
    if exact != value:
        out.respelled[name] = (value, exact)
    return exact


def check_execution(model: type, values: Mapping[str, Any], request: Optional[str]) -> Checked:
    """Check the values the LLM extractor gave an execution command (*model*: its ``Input`` class, *values*: field
    -> value) against *request*. Reads, changes nothing; with no request text there is nothing to check against,
    and nothing is refused. ``Checked.gaps`` are the refusals in order, ``Checked.respelled`` the respellings."""
    from fastworkflow.utils.signatures import is_sentinel_number

    out = Checked()
    if not request or not request.strip():
        return out
    folded = fold(request).lower()
    texts: List[str] = []
    numbers: List[Tuple[str, Any, int]] = []
    for name, field_info in model.model_fields.items():
        value = values.get(name)
        given = is_given(field_info, value)
        is_int = (is_int_field(field_info) and not is_str_field(field_info)
                  and must_ground(exempt=is_converted(field_info)))
        if not given:
            continue
        if is_str_field(field_info):
            text = _execution_str(name, field_info, value, request, folded, out)
            if text:
                texts.append(text)
        elif is_int:
            shown = value
            number = as_int(value)
            if number is not None and not is_sentinel_number(number):
                numbers.append((name, shown, number))
    if numbers:
        written_here = written_numbers(request, texts)
        for name, value, number in numbers:
            verdict, detail = number_rule(number, written_here)
            if verdict == "refuse":
                out.gaps.append(Gap(name, INVALID, "number", detail, value, f"{name}={value!r} {detail}"))
    return out


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


def _plan_exempt(command, p) -> bool:
    info = command.input_model.model_fields.get(p.name) if command.input_model is not None else None
    return info is not None and is_grounding_exempt(info)


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
    declared_names: set = set()   # the str arguments that took a value of their own "One of:" list
    exempt = {p.name: _plan_exempt(command, p) for p in command.params}

    def grounded(p) -> bool:
        return must_ground(exempt=exempt[p.name])

    def refuse(p, value, why: str) -> None:
        if p.required:
            stop(p, why, value, code="refused", state=state_of_why(why))
        notes.append(f"{p.name}={value!r} {why}: default")
        args[p.name] = p.default

    def missing(p) -> None:
        stop(p, "", None, code="missing", message=f"{NOT_FOUND_MISSING} {p.name}")

    for p in command.params:
        shown = raw.get(p.name)   # the model's argument, as the notes show it
        value = screened.get(p.name, single[p.name])
        compared = shown          # what a changed spelling is noted against
        sure = pc._confident(verdicts, p.name) if p.verify else None
        if sure is not None and p.type == "enum" and (pc._empty(value) or clean_str(value).lower() not in p.values):
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
            text = clean_str(value).lower()
            if text not in p.values:
                stop(p, f"is not one of {', '.join(p.values)}", shown, code="not_enum")
            args[p.name] = text
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
            allowed = pc._allowed_values(command, p)
            declared = False   # a value of the field's own list needs no grounding (as for an execution command)
            if allowed and text:
                # the allowed-values rule shared with the execution guard: a value outside the field's "One of:"
                # list is refused; one in it takes the declared spelling. The checks below still run (an
                # in_request field keeps the request's spelling).
                member, why = allowed_rule(allowed, text)
                if member is None:
                    refuse(p, shown, why)
                    continue
                if member != text:
                    notes.append(f"{p.name}={shown!r} -> {member!r} (a declared value)")
                    text = compared = member
                declared = True
                declared_names.add(p.name)
            if text and text.lower() not in named:
                # the free-text screens: a template token, a pronoun, a word of the workflow's own
                # vocabulary. A value the request names as a record (a subject) always passes.
                if why := placeholder_rule(text, folded):
                    refuse(p, shown, why)
                    continue
                if is_pronoun_phrase(text):
                    if len(subjects) != 1:
                        refuse(p, shown, pc.PRONOUN_WHY)
                        continue
                    notes.append(f"{p.name}={shown!r} -> {subjects[0]!r} (the request's one subject)")
                    text = subjects[0]
                elif is_kind_word(text, p.name, vocabulary, pc._kind_selector_options(p, vocabulary)):
                    refuse(p, shown, KIND_WORD_WHY)
                    continue
            if not (grounded(p) or p.subject) or (declared and not (p.in_request or p.subject)):
                # no request check without the option (design 2.46): the value as the model wrote it,
                # after the free-text screens above, which refuse a placeholder, a pronoun or a word
                # of the workflow's vocabulary whatever the field's options
                args[p.name] = text
                continue
            at = ground_position(text, folded)
            if at < 0:
                info = command.input_model.model_fields.get(p.name) if command.input_model is not None else None
                if info is not None and example_rule(info, text):
                    refuse(p, shown, EXAMPLE_WHY)
                    continue
                if p.required:
                    stop(p, NOT_WRITTEN_WHY, shown, code="not_written")
                notes.append(f"{p.name}={shown!r} is not in the request: default")
                args[p.name] = p.default
                continue
            exact = request[at:at + len(text)]
            whole, cut = pc._whole_identifier(request, exact, command.identifiers)
            if cut and whole is None:
                refuse(p, shown, "is part of more than one identifier the request writes")
                continue
            exact = whole or exact
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
        if p.type != "int" or not grounded(p):
            continue
        if p.name in absent:   # an int the extractor left out is "not given" and stays so
            continue
        if written_here is None:
            written_here = pc._request_numbers(request, args, command)
        verdict, detail = number_rule(args[p.name], written_here)
        if verdict == "refuse":
            stop(p, detail, args[p.name], code="number", message=f"{p.name}={args[p.name]} {detail}")
    for p in command.params:
        if p.type != "str" or not grounded(p):
            continue
        value = args[p.name]
        if pc._empty(value) or (not p.required and value == p.default) or p.name in declared_names:
            continue
        if ground_rule(str(value), folded, request) is None:
            shown = raw.get(p.name, value)
            if p.required:
                stop(p, NOT_WRITTEN_WHY, shown, code="not_written")
            notes.append(f"{p.name}={shown!r} {NOT_WRITTEN_WHY}: default")
            args[p.name] = p.default
    for p in command.params:
        sure = pc._confident(verdicts, p.name) if p.verify and p.type != "str" else None
        if sure is not None and args[p.name] != sure[0]:
            notes.append(f"{p.name}={args[p.name]!r} -> {sure[0]!r} (check {sure[1]:.2f})")
            args[p.name] = sure[0]
    strs = [p for p in command.params if p.type == "str"]
    for p in strs:
        if p.required or args[p.name] == p.default:
            continue
        if any(o is not p and str(args[o.name]).lower() == str(args[p.name]).lower() for o in strs):
            notes.append(f"{p.name}={args[p.name]!r} repeats another argument: default")
            args[p.name] = p.default


def ground_position(text: str, folded_request: str) -> int:
    """Where *text* starts in the folded request (-1 when the request does not write it)."""
    return folded_request.find(fold(text).lower()) if text else -1
