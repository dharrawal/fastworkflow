"""One spelling for the allowed values, the producer hint and the pattern of a field (bd fix-qjne.3, part 2).

Removed: ``json_schema_extra={"enum": [...]}`` (write ``Literal[...]``), ``json_schema_extra={"used_by": ...}`` (nothing
read it) and ``json_schema_extra={"pattern": ...}`` (write ``Field(pattern=...)``). Each fails the load of an ``Input``
or ``Output`` field of either kind of command, naming what to write. The ``One of: a, b`` words of a description are
prose now: nothing reads them. Real pydantic models, the real loader and resolver, the real extractor with a scripted
model boundary (the module this one reuses the fixtures of)."""
from __future__ import annotations

import re
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from fastworkflow.planning import planning_commands as pc
from fastworkflow.parameters import check as one_guard, pipeline, rules
from fastworkflow.parameters.validate import validate
from tests.test_extraction_argument_checks import RETAIL, _calls, _framework, extract  # noqa: F401 - fixtures
from tests.test_field_option_keys import _Output

REMOVED = [
    ({"enum": ["book", "member"]}, 'enum: ["book", "member"] is removed: write the field\'s type as Literal["book", "member"]'),
    ({"used_by": ["get_user_details"]}, 'used_by: ["get_user_details"] is removed: delete it (nothing reads it)'),
    ({"pattern": "^#W\\d+$"}, 'pattern: "^#W\\\\d+$" is removed: write Field(pattern="^#W\\\\d+$")'),
]


def _input(extra: dict) -> type:
    return type("Input", (BaseModel,), {"__annotations__": {"kind": str},
                                         "kind": Field(default="", description="a kind", json_schema_extra=extra)})


@pytest.mark.parametrize("planning", [False, True], ids=["execution", "planning"])
@pytest.mark.parametrize("extra, message", REMOVED)
def test_a_removed_key_on_an_input_field_fails_the_load_of_either_kind_of_command(extra, message, planning):
    with pytest.raises(pc.PlanningCommandFormatError) as caught:
        pc.option_params(_input(extra), "app/show", planning=planning)
    assert str(caught.value) == f"app/show: field 'kind': {message}"


def _output_problems(output_cls: type) -> list[str]:
    class Signature:
        """Show one record."""

        class Input(BaseModel):
            query: str = Field(default="", description="the query")

        Output = output_cls

    resolution = pc.FieldSourceResolution(
        workflow_folderpath="", problems=(), skipped_commands=frozenset(), model_to_command={},
        signatures={"Library/show": Signature}, command_files={}, import_errors={}, base_contexts={})
    problems = pc.resolve_signature_field_sources("Library/show", Signature, "show.py", resolution)
    return [p.message for p in problems if p.severity == "error"]


@pytest.mark.parametrize("extra, message", REMOVED)
def test_a_removed_key_on_an_output_field_fails_the_load(extra, message):
    class Output(BaseModel):
        status: str = Field(default="", description="the status", json_schema_extra=extra)

    assert _output_problems(Output) == [f"Signature.Output: field 'status': {message}"]


def test_an_output_field_without_the_removed_keys_loads():
    assert _output_problems(_Output) == []


def test_a_removed_key_on_a_planning_command_file_fails_its_load(tmp_path):
    from tests.test_argument_checks import _load_error, _module, _write

    error = _load_error(_write(tmp_path, {"bad": _module(
        'kind: str = Field(default="", description="k", json_schema_extra={"enum": ["a", "b"]})')}))
    assert 'enum: ["a", "b"] is removed: write the field\'s type as Literal["a", "b"]' in error


def test_a_pattern_is_not_an_option_and_neither_marks_nor_exempts_a_field():
    """A ``Field(pattern=...)`` only validates (bd fix-qjne.21): it is no longer an exemption from a request check, because
    an unmarked field has none; only ``copied_from_command_text`` marks a field."""
    class Input(BaseModel):
        by_extra: str = Field(default="", json_schema_extra={"pattern": "^#W\\d+$"})
        by_field: str = Field(default="", pattern=r"^#W\d+$")
        plain: str = Field(default="")
        marked: str = Field(default="", pattern=r"^#W\d+$", json_schema_extra={"copied_from_command_text": True})

    fields = Input.model_fields
    assert [one_guard.is_marked(fields[n]) for n in ("by_extra", "by_field", "plain", "marked")] == [False, False, False, True]


def test_one_of_words_and_an_enum_key_are_not_allowed_values_any_more():
    class Input(BaseModel):
        kind: str = Field(default="", description="the kind. One of: book, member, branch")

    assert not hasattr(rules, "allowed_values")
    refused = one_guard.check_execution(Input, {"kind": "shelf"}, "find the shelf Ann")
    assert [g.name for g in refused.gaps] == []   # only a written value is asked of the request: "shelf" is written


# ---------------------------------------------------------------- what a Literal field tells the user

def test_a_value_outside_a_literal_is_refused_by_name_with_its_allowed_values():
    """The validation driver's refusal of a value the type does not allow: it names the field and every allowed value."""
    class Input(BaseModel):
        reason: Literal["no longer needed", "ordered by mistake"] = "ordered by mistake"

    refusal = validate(Input, {"reason": "price dropped"}, command_text="x", workflow=None, command_name="cancel").refusal
    assert refusal is not None
    assert refusal.fields == ("reason",)
    assert refusal.message == "Input should be 'no longer needed' or 'ordered by mistake'"


def test_the_retry_note_of_an_unreadable_answer_names_only_the_fields_whose_type_still_fails_the_parse():
    """A ``Literal`` of text no longer fails a whole answer (the guard refuses its value by name), so the note names
    only an ``Enum`` or a non-text ``Literal`` field."""
    import enum

    class Colour(enum.Enum):
        RED = "red"
        BLUE = "blue"

    class Input(BaseModel):
        order_id: str = Field(description="the order")
        reason: Literal["no longer needed", "ordered by mistake"] = "ordered by mistake"
        colour: Colour = Colour.RED
        level: Literal[1, 2] = 1

    assert pipeline.unparsed_refusal(Input) == (
        f"{pipeline.UNPARSED_REFUSAL}: colour must be one of red, blue; level must be one of 1, 2")

    class Plain(BaseModel):
        name: str

    assert pipeline.unparsed_refusal(Plain) == pipeline.UNPARSED_REFUSAL


def test_a_literal_reason_outside_its_values_is_read_once_more_with_the_field_and_values_named(extract):
    """The retail cancel command types its reason as a Literal. The answer outside it is read (every other field
    kept), the guard refuses ``reason`` and the second fill's note names the field and its allowed values."""
    bad = {"reasoning": "x", "order_id": "#W123", "reason": "price dropped elsewhere"}
    good = {"reasoning": "x", "order_id": "#W123", "reason": "no longer needed"}
    # a description example ("no longer needed") is a value like any other: the second fill is one attempt
    output, _, lm = extract(RETAIL, "cancel_pending_order", "cancel order W123, the price dropped elsewhere",
                            bad, good)

    note = "reason='price dropped elsewhere' is not one of no longer needed, ordered by mistake"
    assert _calls(lm) == 2
    assert note not in lm.prompts[0]
    assert all(note in prompt for prompt in lm.prompts[1:])
    assert output.cmd_parameters.reason == "no longer needed"
    assert "Literal['no longer needed', 'ordered by mistake']" in lm.prompts[0]


# ---------------------------------------------------------------- a Literal loses nothing against the removed enum key

def test_one_value_outside_a_literal_refuses_that_field_only_and_keeps_the_others(extract):
    """The answer holds a valid ``order_id`` and a ``reason`` outside the Literal. Only ``reason`` is refused (the one
    second pass names it and its allowed values), and ``order_id`` keeps its value. ``reason`` has a default, so after
    the second pass's refusal it takes it, as any refused optional value does."""
    bad = {"reasoning": "x", "order_id": "#W123", "reason": "price dropped elsewhere"}
    output, diagnostics, lm = extract(RETAIL, "cancel_pending_order", "cancel order #W123, the price dropped elsewhere",
                                      bad, bad)

    assert _calls(lm) == 2
    assert diagnostics["argument_check_refusals"] == [
        "reason='price dropped elsewhere' is not one of no longer needed, ordered by mistake"]
    assert "reason='price dropped elsewhere' is not one of no longer needed, ordered by mistake" in lm.prompts[1]
    assert output.cmd_parameters.order_id == "#W123"
    assert output.cmd_parameters.reason == "ordered by mistake"


def test_a_required_literal_outside_its_values_is_asked_about_by_name_with_its_values_and_nothing_else_is_lost():
    """A required ``Literal`` field and a valid value in the same answer: the guard's one refusal is that field, as an
    invalid value (its reason lists the allowed values); the valid field is not among the gaps."""
    class Input(BaseModel):
        name: str = Field(description="the name")
        kind: Literal["book", "member"] = Field(description="the kind")

    checked = one_guard.check_execution(Input, {"name": "Ann", "kind": "shelf"}, "find the shelf Ann")
    assert [(g.name, g.state, g.why) for g in checked.gaps] == [("kind", "invalid", "is not one of book, member")]
    assert one_guard.check_execution(Input, {"name": "Ann", "kind": "Member "}, "find Ann").respelled == {
        "kind": ("Member ", "member")}


@pytest.mark.parametrize("written", ["No Longer Needed", "  no   longer\tneeded ", "NO LONGER NEEDED"])
def test_a_literal_value_differing_in_case_or_whitespace_takes_the_declared_spelling(extract, written):
    answer = {"reasoning": "x", "order_id": "#W123", "reason": written}
    output, diagnostics, lm = extract(RETAIL, "cancel_pending_order", "cancel order #W123, no longer needed", answer)

    assert _calls(lm) == 1
    assert diagnostics["argument_check_respelled"] == {"reason": "no longer needed"}
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.reason == "no longer needed"
    assert output.cmd_parameters.order_id == "#W123"


def test_the_second_pass_fixes_the_one_refused_literal_field(extract):
    bad = {"reasoning": "x", "order_id": "#W123", "reason": "price dropped elsewhere"}
    good = {"reasoning": "x", "order_id": "#W123", "reason": "ordered by mistake"}
    # the second fill's value is a description example: one attempt, like any other value
    output, _, lm = extract(RETAIL, "cancel_pending_order", "cancel order #W123, I ordered it by mistake",
                            bad, good)

    assert _calls(lm) == 2
    assert "reason='price dropped elsewhere' is not one of no longer needed, ordered by mistake" in lm.prompts[1]
    assert output.parameters_are_valid is True
    assert (output.cmd_parameters.order_id, output.cmd_parameters.reason) == ("#W123", "ordered by mistake")


# ---------------------------------------------------------------- the parse, field by field where a Literal is concerned

def _parse(completion: dict, **annotations):
    import json
    import dspy
    from fastworkflow.parameters.signatures import _TextLiteralsAsText

    signature = dspy.Signature({"statement": (str, dspy.InputField()),
                                **{name: (annotation, dspy.OutputField()) for name, annotation in annotations.items()}})
    return _TextLiteralsAsText().parse(signature, json.dumps(completion))


def test_a_literal_of_text_is_parsed_as_the_text_written_whatever_it_is():
    from typing import Optional

    parsed = _parse({"name": "Ann", "kind": "shelf", "mode": None}, name=str, kind=Literal["book", "member"],
                    mode=Optional[Literal["a", "b"]])
    assert parsed == {"name": "Ann", "kind": "shelf", "mode": None}


def test_a_wrong_type_in_another_field_still_fails_the_whole_answer_as_before():
    """Only a ``Literal`` of text is read as text. A text for an ``int``, a non-text ``Literal`` or an ``Enum`` still fails the
    answer (every field of it is lost), as before this change."""
    import enum

    from dspy.utils.exceptions import AdapterParseError

    class Colour(enum.Enum):
        RED = "red"

    for annotation, value in ((int, "many"), (Literal[1, 2], 3), (Colour, "green")):
        with pytest.raises(AdapterParseError):
            _parse({"name": "Ann", "other": value}, name=str, other=annotation)
