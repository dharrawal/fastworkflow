"""``X | None`` is ``Optional[X]`` everywhere an annotation is unwrapped (bd fix-apjz.10, fix-smgl, fix-ftdl.6.7), and a
check option on an optional str or int loads on an execution command; a planning command keeps its limit and says why.
Real guards, real prompt builder; no model."""
from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel, Field

from fastworkflow import planning_commands as pc
from fastworkflow.parameters import check
from fastworkflow.parameters.fill import unwrap_optional
from fastworkflow.parameters.rules import is_union
from fastworkflow.parameters import signatures

COPIED = {"copied_from_command_text": True}
RECORD = {"record_named_in_command_text": True}


def test_the_two_spellings_of_an_optional_type_are_one_thing():
    assert is_union(Optional[int]) and is_union(int | None) and not is_union(int)
    assert unwrap_optional(int | None) == unwrap_optional(Optional[int]) == (int, None)
    assert unwrap_optional(list[str] | None) == unwrap_optional(Optional[list[str]]) == (list[str], list)


def test_the_extraction_prompt_does_not_call_an_optional_int_required_in_either_spelling():
    class A(BaseModel):
        a: Optional[int] = Field(default=None, description="first")
        b: int | None = Field(default=None, description="second")

    signature = signatures.InputForParamExtraction.create_signature_from_pydantic_model(A)
    descriptions = [f.json_schema_extra["desc"] for name, f in signature.output_fields.items() if name in ("a", "b")]
    assert len(descriptions) == 2 and all("This field is Optional" in d for d in descriptions), descriptions


@pytest.mark.parametrize("option", [COPIED, RECORD])
def test_a_check_option_on_an_optional_str_or_int_loads_on_an_execution_command(option):
    class Marked(BaseModel):
        n: Optional[int] = Field(default=None, json_schema_extra=COPIED)
        m: int | None = Field(default=None, json_schema_extra=COPIED)
        s: str | None = Field(default=None, json_schema_extra=option)
        t: Optional[str] = Field(default=None, json_schema_extra=option)

    params = pc.option_params(Marked, "Library/x")
    assert {p.name: (p.type, p.required) for p in params} == {
        "n": ("int", False), "m": ("int", False), "s": ("str", False), "t": ("str", False)}


def test_an_absent_marked_optional_value_is_not_checked_and_a_given_one_must_be_written():
    class Marked(BaseModel):
        n: int | None = Field(default=None, json_schema_extra=COPIED)
        s: Optional[str] = Field(default=None, json_schema_extra=COPIED)

    # the one guard of a marked execution field (``check_plan`` through ``option_command``, bd fix-ftdl.6.3)
    command = pc.option_command(Marked, "app/show")
    assert check.check_plan(command, {"n": None, "s": None}, "show it").gaps == []
    assert check.check_plan(command, {"n": 5, "s": None}, "show 5").gaps == []
    assert check.check_plan(command, {"n": 7, "s": None}, "show 5").stop.name == "n"
    assert check.check_plan(command, {"n": None, "s": "xyz"}, "show 5").stop.name == "s"   # given, not written


def test_a_marked_optional_int_on_a_planning_command_says_why_it_is_refused():
    class Marked(BaseModel):
        n: int | None = Field(default=None, json_schema_extra=COPIED)

    with pytest.raises(pc.PlanningCommandFormatError) as raised:
        pc.option_params(Marked, "plan.py: weigh", planning=True)
    text = str(raised.value)
    assert "field 'n'" in text and "reach it as text" in text and "default" in text


def test_a_marked_optional_str_loads_on_a_planning_command_in_both_spellings():
    class Marked(BaseModel):
        s: str | None = Field(default=None, json_schema_extra=COPIED)
        t: Optional[str] = Field(default=None, json_schema_extra=COPIED)

    assert [p.name for p in pc.option_params(Marked, "plan.py: weigh", planning=True)] == ["s", "t"]


@pytest.mark.parametrize("name", sorted(pc.RESERVED_NAMES))
def test_a_reserved_planning_field_name_is_refused_with_the_reason_and_the_field_named(name):
    model = type("Reserved", (BaseModel,), {"__annotations__": {name: str},
                                           name: Field(default="", description="d", json_schema_extra=COPIED)})
    with pytest.raises(pc.PlanningCommandFormatError) as raised:
        pc.option_params(model, "plan.py: weigh", planning=True)
    text = str(raised.value)
    assert f"field {name!r} is reserved for the framework's own planner inputs" in text, text
    assert all(other in text for other in pc.RESERVED_NAMES)


def test_a_malformed_planning_field_name_keeps_the_bad_name_message():
    with pytest.raises(pc.PlanningCommandFormatError, match="bad parameter name 'a-b'"):
        pc._parse_param("a-b", {"type": "str"}, "plan.py: weigh")
