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

#: ``Refusal.kind``: a ``fastworkflow.Refused`` a validator raised, naming fields or none. A plain ``ValueError`` or
#: ``AssertionError`` is exactly a ``Refused`` that names no field. A field's type or ``@field_validator`` that fails is
#: exactly a ``Refused`` naming that field (pydantic's error locates it), with the validator's own message.
REFUSED = "refused"
#: ``Refusal.kind``: a rule pydantic reports on no field of the ``Input`` broke; ``message`` is the whole sentence.
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


def _bare_message(error: dict) -> str:
    """A pydantic error's message without pydantic's prefix for a validator's own ``ValueError`` or assertion."""
    return str(error.get("msg", "")).removeprefix("Value error, ").removeprefix("Assertion failed, ")


def _refused_fields(exc, input_model: type) -> Optional[Refusal]:
    """The refusal of the fields pydantic's errors name, when every error is located on a field of *input_model* (a
    field's type or ``@field_validator`` failed): one error is its bare message, several are ``field: message; ...``.
    None when an error is located on no field."""
    errors = exc.errors()
    located = [str(error["loc"][0]) for error in errors if error.get("loc")]
    if not errors or len(located) != len(errors) or not set(located) <= set(input_model.model_fields):
        return None
    message = _bare_message(errors[0]) if len(errors) == 1 else validation_text(exc)
    return Refusal(message, tuple(dict.fromkeys(located)), None, REFUSED)


def validation_text(error) -> str:
    """A pydantic ValidationError as one line: ``field: message; ...`` (a validator's own
    ValueError text without pydantic's "Value error, " prefix)."""
    parts = []
    for e in error.errors():
        message = _bare_message(e)
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

    A field that declares ``Field(pattern=...)`` is checked first, the ONE meaning of a pattern on every path (bd
    fix-ftdl.6.4): a given text value must match the WHOLE pattern (``rules.pattern_refusal``), and a value that does not
    is a refusal naming its field. Every caller runs this after the guards, so a marked id the model cut short has
    already been widened to the id the request writes.
    """
    from pydantic import ValidationError

    import fastworkflow
    from fastworkflow.parameters.rules import pattern_refusal
    from fastworkflow.parameters.input_validation import ValidationContext, validate_input

    mismatched = [(name, refusal) for name, value in values.items() if name in input_model.model_fields
                  and (refusal := pattern_refusal(name, input_model.model_fields[name], value))]
    if mismatched:
        return Validated(refusal=Refusal("; ".join(refusal for _, refusal in mismatched),
                                         tuple(name for name, _ in mismatched), None, REFUSED))

    context = ValidationContext({"command_text": command_text, "workflow": workflow, "command": command_name})
    try:
        validated = validate_input(input_model, values, context)
    except ValidationError as exc:
        error = next((e for e in exc.errors() if isinstance(e.get("ctx", {}).get("error"), fastworkflow.Refused)),
                     None)
        if error is not None:
            refused = error["ctx"]["error"]
            # a bare Refused raised in a @field_validator names the field pydantic located it on, as a ValueError
            # there does
            located = tuple(str(part) for part in error.get("loc", ())[:1] if str(part) in input_model.model_fields)
            return Validated(refusal=Refusal(str(refused), tuple(refused.fields) or located, None, REFUSED))
        if (on_fields := _refused_fields(exc, input_model)) is not None:
            return Validated(refusal=on_fields)
        return Validated(refusal=Refusal(
            f"the arguments break a rule of {command_name}: {validation_text(exc)}", (), None, RULE,
            validation_text(exc)))
    except fastworkflow.Refused as refused:
        return Validated(refusal=Refusal(str(refused), tuple(refused.fields), None, REFUSED))
    except (ValueError, AssertionError) as exc:
        return Validated(refusal=Refusal(str(exc), (), None, REFUSED))
    except Exception as exc:  # noqa: BLE001 - workflow code
        return Validated(refusal=Refusal(str(exc), (), type(exc).__name__, RAISED))
    if not isinstance(validated, input_model):
        raise ValidatorReturnError(f"{command_name}: a validator on Input must return self (the corrected Input) "
                                   f"or raise; it returned {type(validated).__name__}")
    out = {name: getattr(validated, name) for name in input_model.model_fields if name in values}
    return Validated(values=out, written={name: value for name, value in out.items() if value != values[name]})
