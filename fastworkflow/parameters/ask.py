"""The one question builder: every question asked for a missing or refused parameter (bd fix-qoyz step 4).

The wording used to live in two places. The plan side (``planning_commands.refusal_question``,
``missing_argument_question``, ``gaps_question``, ``planning_refusal_parameter_request`` and
``parameter_clarification.planning_question``) and the execution side (the message half of
``InputForParamExtraction.validate_parameters`` and the framing in ``ParameterExtraction._extract_impl``) are now thin
callers of ``word``. Step 6 of the staged plan (docs/parameter_pipeline_design.md) gave a person ONE wording for a missing or refused value
on both kinds; the table ``tests/fixtures/parameter_ask_read_equivalence.json`` holds the text of every case.

``word(extraction, *, form, ...)`` says what is asked in one of these forms:

- ``"refusal"``: a planning command's refused call (``refusal``), put to the user in plain words; one question for
  several arguments when the call has several gaps. With ``confirm`` the command is presented whole instead (every
  parameter, its value or what it lacks, how to reply). Both can open with a ``retry`` sentence.
- ``"gaps"``: the same, always worded from the gaps (never from one refusal's text).
- ``"argument"``: the one question about ONE required argument the request does not give (``name``).
- ``"validation"``: an execution command's own message (what is missing or refused and why, the validator's text and
  suggestions, how to give the values), from ``Parts``.
- ``"frame"``: that message with what the extractor kept and what it refused, and how to get out (``audience``).
- ``"lead"``: the line that names the command, before the message (what the CME records as the question it asked).

What still differs between the two kinds is real, not a switch:

- ``audience``: an agent session is taught the comma protocol and told ``abort and use ...`` in a different way, and
  gets one closing line instead of the answer-only note and the abort lines (``validation``, ``frame``). The text an
  agent reads is unchanged.
- ``confirm``: only a plan the client chose is presented whole (a draft is confirmed there).
- the execution forms carry what only an execution command knows (where a missing value can be looked up, the screens'
  notes, the validators' own messages, suggestions, what was extracted so far, how to leave the question); the
  sentence they open with is the one the plan forms use (``_sentence``): one wording for a person on both kinds.
- the answer-only note is said once by whoever puts the question to a person (``with_answer_only_note``): the plan
  loop, or the execution frame.

Imports no model stack at import time; the plan forms import ``planning_commands`` when they run (it imports this
package's neighbours, so the import cannot be at the top).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from fastworkflow import no_agent_mode

HUMAN = "human"
AGENT = "agent"


def audience_of(context: Any, *, assistant: bool = False) -> str:
    """Who reads a question of the session whose workflow *context* this is: an agent when ``run_as_agent`` is truthy
    (the VALUE decides, never the key's presence: a context that holds ``run_as_agent=False`` is a person's).

    ``assistant`` is the turn's "/" round flag (``waiting.port_of(cme_context).assistant``): a person typed the message,
    so the audience is a person and the replier is the person even in an agent session. This is the ONE definition of
    "the replier is the agent"; every caller that has the round at hand passes it."""
    if assistant:
        return HUMAN
    return AGENT if isinstance(context, Mapping) and context.get("run_as_agent") else HUMAN


FORM_REFUSAL = "refusal"
FORM_GAPS = "gaps"
FORM_ARGUMENT = "argument"
FORM_VALIDATION = "validation"
FORM_FRAME = "frame"
FORM_LEAD = "lead"

MISSING_INFORMATION = "Missing parameter values: "
INVALID_INFORMATION = "Invalid parameter values: "

#: The states a parameter of a ``confirm`` presentation has (the same words ``parameter_clarification`` uses).
_MISSING = "missing"
_INVALID = "invalid"


@dataclass(frozen=True)
class Question:
    """A question and, for a client that renders a form, the open parameters as data (``parameter_request``)."""

    text: str
    request: Optional[dict] = None


@dataclass(frozen=True)
class Parts:
    """What an execution command's validation found, as the message is built from it (``form="validation"``).

    ``missing`` and ``invalid`` name the open fields (``invalid`` as ``"name 'value'"``); ``available_from`` maps a
    missing field to the commands that list it; ``notes`` are the lines of the screens (placeholder, command name,
    word), ``refusals`` the validators' own messages, ``range_notes`` the out-of-range lines, ``suggestions`` the
    suggestions by field."""

    missing: Sequence[str] = ()
    invalid: Sequence[str] = ()
    available_from: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    notes: Sequence[str] = ()
    refusals: Sequence[str] = ()
    range_notes: Sequence[str] = ()
    suggestions: Mapping[str, Sequence[str]] = field(default_factory=dict)


def open_names(missing: Sequence[str], invalid: Sequence[str]) -> list[str]:
    """The fields an execution question asks for, in the order it lists them: the missing ones, then the refused
    ones (their names, without the quoted value)."""
    names = list(missing)
    names.extend(each.split(" '")[0].strip() for each in invalid)
    return names


def with_answer_only_note(text: str) -> str:
    """*text*, a question put to a person, with the note that only the answer is read. Said once, by whoever puts the
    question to the person (the loop, or the execution frame): ``word`` words what is asked, not how it is read."""
    return f"{text} {no_agent_mode.ANSWER_ONLY_NOTE}"


def word(extraction: Any = None, *, form: str = FORM_REFUSAL, audience: str = HUMAN, command: Any = None,
         refusal: Any = None, name: str = "", values: Optional[dict] = None, confirm: bool = False,
         description: str = "", retry: str = "", parts: Optional[Parts] = None,
         message: str = "", refusals: Sequence[str] = (), shown: str = "", command_name: str = "") -> Question:
    """The question for *form* (see the module docstring), as a ``Question``.

    *extraction* is the ``parameter_clarification.Extraction`` of the gaps (plan forms). *command* is the planning
    command (its ``name`` and ``params``). *refusal* is the refused call (``PlanningCommandInputError``); when a
    ``refusal`` form is asked with none, one standing for *extraction* is made. *values* are the values kept (the
    ``confirm`` presentation). *parts*: the
    ``validation`` form's findings. *message*, *refusals*, *shown*: the ``frame`` form's validation message, the
    argument checks' refusals and the parameters shown so far; *command_name* names the command in ``lead``."""
    if form == FORM_VALIDATION:
        return Question(_validation_text(parts or Parts(), audience, command_name))
    if form == FORM_FRAME:
        return Question(_frame_text(message, refusals, shown, audience))
    if form == FORM_LEAD:
        return Question(f"PARAMETER EXTRACTION ERROR FOR COMMAND '{command_name}'\n{message}")
    # the duplicate rule's notes of these values (an optional field it left not given), said before the question
    left_empty = tuple(getattr(extraction, "left_empty", ()) or getattr(getattr(refusal, "gaps", None), "left_empty", ())
                       or ())
    if form == FORM_ARGUMENT:
        return Question(with_left_empty(
            _argument_text(command, name, dict(getattr(extraction, "screened", ())).get(name, "")), left_empty))
    if form == FORM_GAPS:
        return Question(with_left_empty(_gaps_text(command, extraction), left_empty))
    if confirm:
        text, request = _presented(command, values or {}, extraction, description, retry)
        return Question(with_left_empty(text, left_empty), request)
    from fastworkflow import planning_commands

    if refusal is None:
        from fastworkflow import parameter_clarification

        refusal = parameter_clarification.refusal_from_gaps(extraction, values or {})
    text = with_left_empty(f"{retry}{_refusal_text(command, refusal)}", left_empty)
    return Question(text, planning_commands.refusal_gaps(command, refusal).request(command.name))


# ---- plan forms -----------------------------------------------------------------------------------------------------

def _label(name: str) -> str:
    return name.replace("_", " ")


def _argument_text(command: Any, name: str, screened: str = "") -> str:
    """*screened*: the value that was given and not taken, with why (``Gap.screened``), said before the question."""
    task = _label(command.name)
    label = _label(name)
    why = f" ({label} {screened})" if screened else ""
    return (f"I can't run {task} yet: your request does not say which {label} to use{why}. "
            f"Which {label} should I use?")


def _joined(names: Sequence[str]) -> str:
    names = [_label(n) for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _sentence(task: str, missing: Sequence[str], invalid: Sequence[tuple[str, str]],
              explanations: Sequence[str] = (), screened: Sequence[tuple[str, str]] = ()) -> str:
    """THE words a person is told for the values a command lacks or refuses, on both kinds of command: what is wrong
    (the request does not give the missing ones; each refused one with why, or the validators' own *explanations* when
    they gave them) and what to answer (one value, or name: value lines for several)."""
    problems = []
    if missing:
        problems.append(f"your request does not give the {_joined(missing)}")
        problems.extend(f"the {_label(name)} {text}" for name, text in screened if name in missing)
    if invalid:
        problems.extend(explanations or [f"the {_label(name)} {why}" for name, why in invalid])
    names = list(dict.fromkeys([*missing, *(name for name, _ in invalid)]))
    if len(names) == 1:
        ask = f"Which {_label(names[0])} should I use?"
    else:
        example = "; ".join(f"{name}: ..." for name in names[:2])
        ask = (f"What should the {_joined(names)} be? Reply with name: value lines ({example}), or with the values in "
               f"order separated by commas.")
    return f"I can't run {task} yet: {'; '.join(problems)}. {ask}"


def _gaps_text(command: Any, gaps: Any) -> str:
    # ``reason`` beside refused values: the validators' own message on a tie (``planning_commands.argument_gaps``)
    reason = getattr(gaps, "reason", "")
    return _sentence(_label(command.name), gaps.missing, gaps.invalid, [reason] if reason and gaps.invalid else (),
                     screened=gaps.screened)


def _refusal_text(command: Any, refusal: Any) -> str:
    """The words for a refused planning call. What it says comes from the refusal's structured parts: ``gaps`` (every
    argument), ``fields`` (a validator's ``Refused(field=...)``), ``gap`` (the one argument a guard refused: name,
    state, why, value) and ``refusal`` (a validator's own message, bare); the message text is read for none of them."""
    from fastworkflow import planning_commands as pc

    gaps = pc.refusal_gaps(command, refusal)
    if len(gaps.names) > 1:
        return _gaps_text(command, gaps)
    task = _label(command.name)
    detail = getattr(refusal, "refusal", None) or str(refusal)
    if refusal.fields:
        # A Refused(field=...) from the hook: the arguments it named are asked for, with its message
        return _sentence(task, (), [(name, "") for name in dict.fromkeys(refusal.fields)], [detail])
    gap = refusal.gap
    param = next((p for p in command.params if gap is not None and gap.name == p.name), None)
    if param is not None:
        if gap.state == "missing":
            return _argument_text(command, param.name, dict(gaps.screened).get(param.name, ""))
        label = _label(param.name)
        return f"I can't run {task} yet: {refused_clause(gap, param.values)}. Which {label} should I use?"
    return f"I can't run {task} yet: {detail}. What should I change?"


def refused_clause(gap: Any, allowed: Sequence[str] = ()) -> str:
    """What is wrong with the value one guard refused (a ``check.Gap``), in the words a person reads, the same on both
    kinds of command: "the title 'Dune' is not in your request", "'Ann Lee' is the member in your request, not the
    branch". *allowed*: the values of a choice, for a value outside them."""
    label, value, why = _label(gap.name), repr(gap.value), gap.why
    if why.startswith("is also the value of "):
        return f"{value} is the {_label(why.removeprefix('is also the value of '))} in your request, not the {label}"
    if why.startswith("is not written in the request"):
        return f"the {label} {value} is not in your request"
    if why.startswith("names no record of the request"):
        return f"{value} does not match anything in your request"
    if why.startswith("is not one of ") and allowed:
        return f"the {label} must be one of: {', '.join(allowed)}"
    if why.startswith("is not an integer"):
        return f"the {label} must be a whole number"
    if why.startswith("is not true or false"):
        return f"the {label} must be yes or no"
    return f"the {label} {value} is not valid"


#: The start of a note the duplicate rule writes (``left_empty_line``): the field's name, then this.
_LEFT_EMPTY = " left empty: "
_LEFT_EMPTY_NOTE = re.compile(r"^([A-Za-z_]\w*)" + re.escape(_LEFT_EMPTY))


def left_empty_line(name: str, value: str, times: int, holder: str) -> str:
    """The note of an optional field the duplicate rule left not given (``check._duplicate_rule``), one plain line on the
    span's notes, the review item and a question: "connected_name left empty: 'X' is written once in your request and is
    the name." Not a refusal: the call runs without it."""
    written = {0: "nowhere", 1: "once", 2: "twice"}.get(times, f"{times} times")
    return f"{name}{_LEFT_EMPTY}{value!r} is written {written} in your request and is the {_label(holder)}."


def left_empty_lines(notes: Sequence[str]) -> dict[str, str]:
    """``{field: line}`` of the duplicate rule's notes among a check's *notes* (``left_empty_line``), in order."""
    return {match.group(1): str(note) for note in notes or () if (match := _LEFT_EMPTY_NOTE.match(str(note)))}


def with_left_empty(text: str, lines: Sequence[str]) -> str:
    """*text*, a question, with the duplicate rule's notes of the same values before it, one line each."""
    return "\n".join([*lines, text]) if lines else text


def _presented(command: Any, values: dict, gaps: Any, description: str, retry: str) -> tuple[str, dict]:
    """A plan the client chose, presented whole: its parameters with the value kept or what each lacks, and how to
    reply; the data names the same."""
    states = {p.name: gaps.state_of(p.name)[0] for p in command.params}
    parameters, lines = [], []
    for p in command.params:
        state = states[p.name] if states[p.name] in (_MISSING, _INVALID) else None
        value = None if state else values.get(p.name)
        parameters.append({"name": p.name, "required": bool(p.required), "value": value, "state": state,
                           "description": p.desc})
        label = f"{p.name} ({'required' if p.required else 'optional'})"
        lines.append(f"- {label}: " + (str(value) if value not in (None, "") else
                                       ("not given" if state != "invalid" else "not usable")))
    text = (f"{retry}You chose the planning command {command.name}" + (f": {description.rstrip('.')}" if description else "")
            + ".\nIts parameters:\n" + "\n".join(lines)
            + "\nReply with one name: value line for each value to set or change, or yes to run with the values shown"
            + (" (the ones not given must be given first)" if gaps.names else "")
            + f"; cancel stops. {no_agent_mode.ANSWER_ONLY_NOTE}")
    request = {"command": command.name, "description": description, "confirm": True, "parameters": parameters,
               "missing": [n for n in states if states[n] == _MISSING],
               "invalid": [n for n in states if states[n] == _INVALID],
               "values": {name: str(value) for name, value in values.items()
                          if value is not None and name not in gaps.names}}
    return text, request


# ---- execution forms ------------------------------------------------------------------------------------------------

def _validation_text(parts: Parts, audience: str, command_name: str = "") -> str:
    if audience != AGENT and open_names(parts.missing, parts.invalid):
        return _person_validation_text(parts, command_name)
    message = ""
    if parts.missing:
        message += MISSING_INFORMATION + ", ".join(parts.missing) + "\n"
        for missing_field in parts.missing:
            is_available_from = parts.available_from.get(missing_field)
            if is_available_from:
                msg_prefix = "abort and " if audience != AGENT else ""
                bare_commands = [
                    str(entry).split("/")[-1].split(".", 1)[0]
                    for entry in is_available_from
                ]
                message += f"{msg_prefix}use the {' or '.join(bare_commands)} command(s) to get {missing_field} information. OR...\n"

    if parts.invalid:
        message += INVALID_INFORMATION + ", ".join(parts.invalid) + "\n"

    for note in parts.notes:
        message += note + "\n"

    for refusal_message in parts.refusals:
        message += refusal_message + "\n"

    for note in parts.range_notes:
        message += note + "\n"

    for field_name, suggestions in parts.suggestions.items():
        if suggestions:
            is_format_instruction = any(("format" in str(s).lower() or "pattern" in str(s).lower()) for s in suggestions)

            if is_format_instruction:
                message += f"\n{field_name}: {', '.join(suggestions)}"
            else:
                message += f"Did you mean one of these {field_name}s? {', '.join(suggestions)}\n"

    combined_fields = open_names(parts.missing, parts.invalid)
    if combined_fields:
        combined_fields_str = ", ".join(combined_fields)
        if audience == AGENT:
            message += f"\nProvide corrected parameter values in the exact order specified below, separated by commas:\n{combined_fields_str}"
        elif len(combined_fields) == 1:
            message += f"\nReply with the value for {combined_fields_str}."
        else:
            message += ("\nReply with name: value lines, or with the values in the order listed below separated by commas "
                        "(put a value that has a comma in quotes):\n" + combined_fields_str)
    return message


def _person_validation_text(parts: Parts, command_name: str) -> str:
    """The message a person reads for an execution command's open values: the one sentence both kinds use
    (``_sentence``), then what only this kind knows, one line each: where a missing value can be looked up, the
    screens' notes, the validators' own messages, the out-of-range lines and the suggestions."""
    invalid: dict[str, str] = {}
    for each in parts.invalid:
        name = each.split(" '")[0].strip()
        invalid.setdefault(name, f"{each[len(name):].strip()} is not valid")
    message = _sentence(_label(command_name.split("/")[-1]), parts.missing, list(invalid.items()), parts.refusals)
    for missing_field in parts.missing:
        if is_available_from := parts.available_from.get(missing_field):
            bare_commands = [str(entry).split("/")[-1].split(".", 1)[0] for entry in is_available_from]
            message += f"\nabort and use the {' or '.join(bare_commands)} command(s) to get {missing_field} information."
    for note in (*parts.notes, *parts.range_notes):
        message += "\n" + note
    for field_name, suggestions in parts.suggestions.items():
        if suggestions:
            if any(("format" in str(s).lower() or "pattern" in str(s).lower()) for s in suggestions):
                message += f"\n{field_name}: {', '.join(suggestions)}"
            else:
                message += f"\nDid you mean one of these {field_name}s? {', '.join(suggestions)}"
    return message


def _frame_text(message: str, refusals: Sequence[str], shown: str, audience: str) -> str:
    if refusals:
        # Why a value the request seemed to give is asked for again (fix-qjf4).
        message = "Values not used: " + "; ".join(refusals) + "\n" + message
    if shown:
        message = f"Extracted parameters so far:\n{shown}\n\n{message}"

    if audience != AGENT:
        message += "\n" + no_agent_mode.ANSWER_ONLY_NOTE
        message += "\nEnter 'abort' to get out of this error state and/or execute a different command."
        message += "\nEnter 'you misunderstood' if the wrong command was executed."
    else:
        message += "\nCheck your command name if the wrong command was executed."
    return message
