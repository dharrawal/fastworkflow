"""The one driver that runs an ``Input`` class's field checks and model validators (bd fix-qoyz step 2).

Both callers used to carry their own copy of this run. The typed planning call
(``planning_commands.check_input``) and the execution command's turn
(``InputForParamExtraction._check_input_rules``, which went through ``check_input``) read the same
outcome from ``validate`` and rebuild their own wording and exceptions from it, so a message or an
attribute changes in one place only on purpose.

``validate`` never raises for a refusal. It raises ``ValidatorReturnError`` for a programming error
of the command author (a model validator that returns something else than the ``Input``), exactly where
the old code raised ``PlanningCommandCallError`` for it.

This module imports no model stack at import time (no dspy, torch or transformers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

#: ``Refusal.kind``: a ``fastworkflow.Refused`` a validator raised, naming fields or none.
REFUSED = "refused"
#: ``Refusal.kind``: a plain ``ValueError`` or ``AssertionError`` a model validator raised (a refusal, no field).
PLAIN = "plain"
#: ``Refusal.kind``: a field's type or field validator broke a rule; ``message`` is the whole sentence.
RULE = "rule"
#: ``Refusal.kind``: any other exception a model validator raised; ``Refusal.raised`` names its type.
RAISED = "raised"


class ValidatorReturnError(ValueError):
    """A model validator of an ``Input`` returned something other than the ``Input``: a bug of the command's author."""


@dataclass(frozen=True)
class Refusal:
    """Why the validators refused the values.

    ``message``: what the validator said, bare, except for kind ``rule``, whose ``message`` is the whole
    sentence ``the arguments break a rule of <command>: <field>: <message>``. ``fields``: the fields a
    ``fastworkflow.Refused`` named. ``raised``: the exception type of a validator that raised (kind ``raised``;
    ``message`` is then the exception's text). ``kind`` tells the callers which wording each side gives it.
    ``detail``: for kind ``rule``, the bare reason without the sentence around it (``<field>: <message>``).
    """
    message: str
    fields: tuple[str, ...] = ()
    raised: Optional[str] = None
    kind: str = REFUSED
    detail: str = ""


@dataclass(frozen=True)
class Validated:
    """The outcome of ``validate``.

    ``values``: every given field as the validators left it (empty on a refusal). ``written``: the fields the
    validators rewrote (a correction, such as a fuzzy match to a known record), ``{name: value}``. A rewrite is KEPT
    when the validation passes and DROPPED when it refuses: ``written`` is empty on a refusal, for both kinds of
    command. ``refusal``: None when the values pass.
    """
    values: dict = field(default_factory=dict)
    written: dict = field(default_factory=dict)
    refusal: Optional[Refusal] = None


def validation_text(error) -> str:
    """A pydantic ValidationError as one line: ``field: message; ...`` (a validator's own
    ValueError text without pydantic's "Value error, " prefix)."""
    parts = []
    for e in error.errors():
        message = str(e.get("msg", "")).removeprefix("Value error, ")
        where = ".".join(str(x) for x in e.get("loc", ()))
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


def validate(input_model: type, values: Mapping[str, Any], *, command_text: str, workflow: Any,
             command_name: str) -> Validated:
    """Run ``input_model``'s field types and validators on ``values``, with the validation context
    ``{"command_text": command_text, "workflow": workflow, "command": command_name}``.

    *values* are passed as given (the caller applies any reading of an absent value first). Raises
    ``ValidatorReturnError`` when a model validator does not return the ``Input``; every refusal is
    in the returned ``Validated``.
    """
    from pydantic import ValidationError

    import fastworkflow
    from fastworkflow.utils.input_validation import ValidationContext, validate_input

    context = ValidationContext({"command_text": command_text, "workflow": workflow, "command": command_name})
    try:
        validated = validate_input(input_model, values, context)
    except ValidationError as exc:
        refused = next((e["ctx"]["error"] for e in exc.errors()
                        if isinstance(e.get("ctx", {}).get("error"), fastworkflow.Refused)), None)
        if refused is not None:
            return Validated(refusal=Refusal(str(refused), tuple(refused.fields), None, REFUSED))
        return Validated(refusal=Refusal(
            f"the arguments break a rule of {command_name}: {validation_text(exc)}", (), None, RULE,
            validation_text(exc)))
    except fastworkflow.Refused as refused:
        return Validated(refusal=Refusal(str(refused), tuple(refused.fields), None, REFUSED))
    except (ValueError, AssertionError) as exc:
        return Validated(refusal=Refusal(str(exc), (), None, PLAIN))
    except Exception as exc:  # noqa: BLE001 - workflow code
        return Validated(refusal=Refusal(str(exc), (), type(exc).__name__, RAISED))
    if not isinstance(validated, input_model):
        raise ValidatorReturnError(f"{command_name}: a validator on Input must return self (the corrected Input) "
                                   f"or raise; it returned {type(validated).__name__}")
    out = {name: getattr(validated, name) for name in input_model.model_fields if name in values}
    return Validated(values=out, written={name: value for name, value in out.items() if value != values[name]})
