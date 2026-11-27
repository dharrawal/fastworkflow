"""The ``ui.tones`` (badge tone map) and ``ui.currency`` (ISO code) options of a ``Signature.Output`` field
(fix-weih.29, fix-weih.30). Same load check as the other ui options, so ``fastworkflow run_cli``, ``train``, the server and
``command preview --check`` report a refusal the same way."""
from __future__ import annotations

import enum
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

from fastworkflow import FieldOptions, UiOptions
from fastworkflow.planning import planning_commands as pc


def _problems(output_cls):
    class Command:
        """Show it."""

    class Input(BaseModel):
        query: str = Field(default="")

    Command.Input = Input
    Command.Output = output_cls
    resolution = pc.FieldSourceResolution(
        workflow_folderpath="", problems=(), skipped_commands=frozenset(), model_to_command={},
        signatures={"Library/show": Command}, command_files={}, import_errors={}, base_contexts={})
    return [p.message for p in pc.resolve_signature_field_sources("Library/show", Command, "show.py", resolution)]


def _field(annotation, default, **ui):
    return annotation, Field(default=default, json_schema_extra=FieldOptions(ui=UiOptions(**ui)))


def _output(annotation, default, **ui):
    from pydantic import create_model
    return create_model("Output", value=_field(annotation, default, **ui))


class Color(str, enum.Enum):
    RED = "red"
    BLUE = "blue"


class Level(enum.Enum):
    LOW = 1
    HIGH = 2


TONES = {"delivered": "good", "cancelled": "bad", "on hold": "warning"}


@pytest.mark.parametrize("annotation", [str, Optional[str], list[str], Optional[list[str]],
                                        Literal["delivered", "cancelled", "on hold"],
                                        list[Literal["delivered", "cancelled", "on hold"]]])
def test_tones_on_a_text_field_are_accepted_and_travel_in_the_schema_unchanged(annotation):
    Output = _output(annotation, None, format="badge", tones=TONES)
    assert _problems(Output) == []
    assert Output.model_json_schema()["properties"]["value"]["ui"] == {"format": "badge", "tones": TONES}


def test_tones_on_an_enum_are_keyed_by_the_member_value_as_text():
    assert _problems(_output(Color, Color.RED, format="badge", tones={"RED": "bad", " blue ": "good"})) == []
    assert _problems(_output(Level, Level.LOW, format="badge", tones={"1": "good", "2": "bad"})) == []
    problems = _problems(_output(Level, Level.LOW, format="badge", tones={"low": "good"}))
    assert len(problems) == 1 and "'low'" in problems[0] and "declared values" in problems[0] and "'1'" in problems[0]


def test_a_tones_key_is_checked_against_a_literal_case_insensitively():
    assert _problems(_output(Literal["Delivered", "Cancelled"], "Delivered", format="badge",
                             tones={"delivered": "good", " CANCELLED": "bad"})) == []
    problems = _problems(_output(Literal["delivered", "cancelled"], "delivered", format="badge",
                                 tones={"delivred": "good"}))
    assert len(problems) == 1 and "'delivred'" in problems[0] and "'delivered'" in problems[0]


@pytest.mark.parametrize("annotation", [bool, int, float, dict[str, str], list[dict[str, str]], Literal[True, False]])
def test_tones_on_a_field_that_is_not_drawn_as_a_text_badge_are_refused(annotation):
    problems = _problems(_output(annotation, None, format="badge", tones=TONES))
    assert len(problems) == 1 and "ui.tones is for a str, Literal or Enum" in problems[0]


@pytest.mark.parametrize("tones, needle", [
    ({}, "non-empty dict"),
    (["good"], "non-empty dict"),
    ("good", "non-empty dict"),
    ({"delivered": "green"}, "one of 'good', 'warning', 'bad', 'neutral'"),
    ({"delivered": "Good"}, "one of 'good', 'warning', 'bad', 'neutral'"),
    ({"delivered": True}, "one of 'good', 'warning', 'bad', 'neutral'"),
    ({"": "good"}, "keys must be non-empty text"),
    ({"  ": "good"}, "keys must be non-empty text"),
    ({1: "good"}, "keys must be non-empty text"),
    ({"Delivered": "good", " delivered ": "bad"}, "are the same value"),
])
def test_a_malformed_tone_map_is_refused_naming_the_fault(tones, needle):
    class Output(BaseModel):
        value: str = Field(default="", json_schema_extra={"ui": {"format": "badge", "tones": tones}})

    problems = _problems(Output)
    assert len(problems) == 1 and needle in problems[0] and "'value'" in problems[0], problems


@pytest.mark.parametrize("fmt", [None, "code", "percent", "currency"])
def test_tones_need_format_badge(fmt):
    ui = {"tones": TONES} | ({"format": fmt} if fmt else {})

    class Output(BaseModel):
        value: str = Field(default="", json_schema_extra={"ui": ui})

    problems = _problems(Output)
    assert len(problems) == 1 and 'ui.tones needs ui.format "badge"' in problems[0]


@pytest.mark.parametrize("annotation", [float, int, Optional[float], list[float], list[int], Optional[list[float]]])
@pytest.mark.parametrize("code", ["USD", "eur", "Jpy", "XXX"])
def test_a_currency_code_on_a_number_is_accepted_and_travels_unchanged(annotation, code):
    Output = _output(annotation, None, format="currency", currency=code)
    assert _problems(Output) == []
    assert Output.model_json_schema()["properties"]["value"]["ui"] == {"format": "currency", "currency": code}


@pytest.mark.parametrize("code", ["", "US", "USDD", "US1", "U D", "€", "us-d", 3, None, True, ["USD"]])
def test_a_currency_that_is_not_three_letters_is_refused(code):
    class Output(BaseModel):
        value: float = Field(default=0.0, json_schema_extra={"ui": {"format": "currency", "currency": code}})

    problems = _problems(Output)
    assert len(problems) == 1 and "three-letter ISO 4217 code" in problems[0] and "'value'" in problems[0]


@pytest.mark.parametrize("annotation", [str, bool, list[str], dict[str, float]])
def test_a_currency_on_a_field_that_is_not_a_number_is_refused(annotation):
    problems = _problems(_output(annotation, None, format="currency", currency="USD"))
    assert len(problems) == 1 and "ui.currency is for an int or float" in problems[0]


@pytest.mark.parametrize("fmt", [None, "badge", "percent", "code"])
def test_a_currency_needs_format_currency(fmt):
    ui = {"currency": "USD"} | ({"format": fmt} if fmt else {})

    class Output(BaseModel):
        value: float = Field(default=0.0, json_schema_extra={"ui": ui})

    problems = _problems(Output)
    assert len(problems) == 1 and 'ui.currency needs ui.format "currency"' in problems[0]


def test_both_options_are_refused_on_an_input_field():
    with pytest.raises(pc.PlanningCommandFormatError, match="ui.tones is for a Signature.Output field"):
        pc.check_ui_option("state", {"ui": {"tones": TONES}}, "Signature.Input", kind="str")
    with pytest.raises(pc.PlanningCommandFormatError, match="ui.currency is for a Signature.Output field"):
        pc.check_ui_option("price", {"ui": {"currency": "USD"}}, "Signature.Input", kind="other")


def test_a_hint_with_neither_option_is_unchanged():
    class Output(BaseModel):
        state: str = Field(default="", json_schema_extra={"ui": {"format": "badge"}})
        total: float = Field(default=0.0, json_schema_extra={"ui": {"format": "currency"}})

    assert _problems(Output) == []
    assert Output.model_json_schema()["properties"]["total"]["ui"] == {"format": "currency"}
