"""Argument checks of an execution command's LLM parameter extraction (fix-qjf4).

A planning command's typed call has its arguments checked against the request
(``planning_commands.check_arguments``). An execution command's parameters, when the LLM
extractor fills them, get the same kinds of checks here, built on the same helpers:

- **grounding**: a ``str`` value must be written in the request, verbatim up to case and
  Unicode dashes (any dash matches ``-``), after surrounding quotes, angle brackets and a
  trailing ``.;,`` are removed. The value then takes the request's own spelling.
- **placeholders**: punctuation alone (``...``), a template token or an invented example value (``planning_commands.
  _placeholder_like``: ``{name}``, ``NAME_1``, "unspecified", "John Doe" the request does not
  write), a bracketed placeholder (``<item_uid>``), and one of the field's own description
  ``examples`` the request does not write.
- **allowed values**: a ``str`` field whose allowed values are declared outside its type, in
  ``json_schema_extra={"enum": [...]}`` (the JSON Schema keyword) or in its description as
  ``One of: a, b, c``, must hold one of them (case-insensitive; the declared spelling is taken).
  A ``Literal`` or ``Enum`` field is already checked by its type in ``validate_parameters``. The rule
  (``allowed_member``) is shared: ``planning_commands.check_arguments`` applies it to a planning command's
  ``One of:`` list (fix-48np.3).
- **numbers**: an ``int`` value must be a number the request writes, when the request writes any
  number outside the text of its ``str`` values. Unlike the planning rule, a value the request does
  not write is refused, never replaced by the one number the request writes.

**What "the request" is.** The command text the extractor read (``InputForParamExtraction.command``:
the user's message in assistant mode, the command the agent wrote in agent mode). It is the
extractor's only input, so a value it does not write came from somewhere else: the field's
description, its few-shot examples, or the model. The conversation's earlier user messages are not
counted: the extractor never sees them, and the offline replay (the app) found that adding them
changes no verdict on a ``str`` value.

**What is not checked.**

- A value that is not given: ``None``, empty, ``NOT_FOUND``, the ``<UNKNOWN>`` marker, the
  extractor's "not given" number, the field's not-given value or its declared default (the
  extractor is told to answer the default for a value the request does not give, so a default is
  indistinguishable from "not given"). An absent ``Optional[int] = None`` is never refused.
- Types other than ``str`` and ``int`` (``float``, ``bool``, dates, lists, models): their values are
  converted on purpose, or (lists) are not checked by design, as on the planning side.
- A field exempted from grounding (its value is legitimately converted from the request: a date
  written "next Monday", an amount, a code the request spells another way). The marker is the
  ``converted_from_command_text`` option: ``json_schema_extra={"converted_from_command_text": True}``. A field with a
  declared ``pattern`` is exempt too: it declares a format the value is converted into (``#W123`` of
  "order W123"). An exempt field still gets the placeholder and allowed-values checks, but not the
  description-example check: a converted value can equal an example legitimately (tau-bench airline:
  "New York" on "May 20" gives ``JFK`` and ``2024-05-20``, the search fields' own examples). On an ``int``
  field the same marker turns the number check off; it needs a plain default (``int = 0``), since a
  check option cannot sit on an ``Optional[int]`` (``planning_commands.option_params`` refuses it).
- A planning command's ``Input``: ``check_arguments`` checks it already.

A refused value is treated as a value that was not given: the extractor is asked once more with the
refusal appended to the command (``REFUSED_EXTRACTION_NOTE``, as ``typed_planning_call`` does with
``REFUSED_CALL_NOTE``), the new values are checked against the original command text, and a value
still refused takes the field's not-given value (a required field is then asked for; an optional one
takes its default). ``FW_EXTRACTION_ARGUMENT_CHECKS=off`` turns the checks off.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

import fastworkflow
from fastworkflow.parameters import check as one_guard
from fastworkflow.utils.input_validation import set_field_unvalidated
from fastworkflow.utils.signatures import NOT_FOUND, not_given_value

#: The switch. On unless set to off, 0 or false (case-insensitive), like the other extraction guards.
ARGUMENT_CHECKS_ENV_VAR = "FW_EXTRACTION_ARGUMENT_CHECKS"
_OFF = ("off", "0", "false")

#: Appended to the command text when the extractor is asked once more after a refusal.
REFUSED_EXTRACTION_NOTE = (
    "(A first extraction from this command was refused: {refusal}. Extract the values again from the "
    "command above, correcting that: copy each value exactly as the command writes it, and answer "
    f"{NOT_FOUND} for a value the command does not give.)")

#: Appended to the request when a call's first phase found nothing for its lookup, and the arguments are extracted
#: once more (``planning_commands.reextracted_arguments``): the phase's observation, in the refusal note's style.
PHASE_FAILED_EXTRACTION_NOTE = (
    "(A first run of this command found nothing for its lookup: {observation}. Extract the values again from the "
    "command above, correcting that: copy each value exactly as the command writes it, and answer "
    f"{NOT_FOUND} for a value the command does not give.)")


def checks_enabled() -> bool:
    """False when FW_EXTRACTION_ARGUMENT_CHECKS is off, 0 or false."""
    value = str(fastworkflow.get_env_var(ARGUMENT_CHECKS_ENV_VAR, default="on")).strip().lower()
    return value not in _OFF


@dataclass
class ExtractionCheck:
    """What ``check_extracted_values`` found: ``refused`` maps a field to ``(value, why)``,
    ``respelled`` maps a field to ``(value, the request's or the declared spelling)``."""

    refused: Dict[str, Tuple[Any, str]] = dataclass_field(default_factory=dict)
    respelled: Dict[str, Tuple[Any, Any]] = dataclass_field(default_factory=dict)

    def refusal_texts(self) -> List[str]:
        """``name='value' why`` for each refused field, in field order."""
        return [f"{name}={value!r} {why}" for name, (value, why) in self.refused.items()]

    def note(self) -> str:
        """The refusal, as appended to the command for the second extraction."""
        return REFUSED_EXTRACTION_NOTE.format(refusal="; ".join(self.refusal_texts()))

    def apply(self, params: BaseModel) -> None:
        """Write the respellings onto *params*, and give each refused field its not-given value."""
        fields = type(params).model_fields
        for name, (_, spelled) in self.respelled.items():
            if name not in self.refused:
                set_field_unvalidated(params, name, spelled)
        for name in self.refused:
            set_field_unvalidated(params, name, not_given_value(fields[name]))


def check_extracted_values(params: BaseModel, request: Optional[str]) -> ExtractionCheck:
    """Check the values the LLM extractor gave in *params* against *request* (see the module doc).

    Reads *params*, changes nothing: ``ExtractionCheck.apply`` writes the outcome. With no request
    text there is nothing to check against, and nothing is refused. The rules are
    ``fastworkflow.parameters.check.check_execution``'s (the one set of guard rules).
    """
    model = type(params)
    checked = one_guard.check_execution(
        model, {name: getattr(params, name, None) for name in model.model_fields}, request)
    result = ExtractionCheck()
    for gap in checked.gaps:
        result.refused[gap.name] = (gap.value, gap.why)
    result.respelled = dict(checked.respelled)
    return result
