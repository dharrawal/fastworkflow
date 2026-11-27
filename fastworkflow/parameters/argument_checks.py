"""Argument checks of an execution command's LLM parameter extraction (fix-qjf4).

A planning command's typed call has its arguments checked against the request
(``planning_commands.check_arguments``). An execution command's parameters, when the LLM
extractor fills them, get the same checks here, built on the same helpers:

- **placeholders** (every ``str``): punctuation alone (``...``) and a value that is entirely one template token
  (``rules.placeholder_like``: ``{name}``, ``<item_uid>``), the verdict ``validate_parameters`` and a plan's check also
  read. There is no list of stand-in words, and an id such as ``SKU_42`` is a value (bd fix-ftdl.6.1).
- **copy**: not here. The strict check of a field marked ``copied_from_command_text`` or
  ``record_named_in_command_text`` is the one guard of both kinds, ``parameters.check.check_plan``, which
  ``validate_parameters`` runs on the command's marked fields (bd fix-ftdl.6.3).

**What "the request" is.** The command text the extractor read (``InputForParamExtraction.command``:
the user's message in assistant mode, the command the agent wrote in agent mode). It is the
extractor's only input, so a value it does not write came from somewhere else: the field's
description, its few-shot examples, or the model. The conversation's earlier user messages are not
counted: the extractor never sees them.

**What is not checked.**

- A value that is not given: ``None``, empty, ``NOT_FOUND``, the ``<UNKNOWN>`` marker, the
  extractor's "not given" number, the field's not-given value or its declared default (the
  extractor is told to answer the default for a value the request does not give, so a default is
  indistinguishable from "not given"). An absent ``Optional[int] = None`` is never refused.
- Types other than ``str`` and ``int`` (``float``, ``bool``, dates, lists, models): their values are
  converted on purpose, or (lists) are not checked by design, as on the planning side.
- A field with no option: the extractor's value stands (a date written "next Monday", an amount, a code the request
  spells another way, a name the model normalised). A marked field is checked by the one guard, at validation. A ``Field(pattern=...)`` only validates; it does not check the
  request. The span's ``argument_check_not_in_request`` lists such fields whose value the request does not write, so an
  author can see which ones may need ``copied_from_command_text``.
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
from fastworkflow.parameters.input_validation import set_field_unvalidated
from fastworkflow.parameters.signatures import NOT_FOUND, not_given_value

#: The switch. On unless set to off, 0 or false (case-insensitive), like the other extraction guards.
ARGUMENT_CHECKS_ENV_VAR = "FW_EXTRACTION_ARGUMENT_CHECKS"
_OFF = ("off", "0", "false")

#: Appended to the command text when the extractor is asked once more after a refusal.
REFUSED_EXTRACTION_NOTE = (
    "(A first extraction from this command was refused: {refusal}. Extract the values again from the "
    "command above, correcting that, and answer "
    f"{NOT_FOUND} for a value the command does not give.)")

#: Appended to the request when a call's first phase found nothing for its lookup, and the arguments are extracted
#: once more (``planning_commands.reextracted_arguments``): the phase's observation, in the refusal note's style.
PHASE_FAILED_EXTRACTION_NOTE = (
    "(A first run of this command found nothing for its lookup: {observation}. Extract the values again from the "
    "command above, correcting that, and answer "
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
    #: The advisory (``parameters.check.not_written_fields``): unmarked fields whose value the request does not write.
    not_in_request: List[str] = dataclass_field(default_factory=list)

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
    result.not_in_request = list(checked.not_in_request)
    return result
