"""The author-facing field option keys of planning command and execution command Inputs (spec sections 2.0 to 2.7).

Each key is read by the real loader (``planning_commands.option_params`` and the resolver) and must give the
internal option its old spelling gave. Each removed spelling fails the load naming its replacement. No mocks:
the Inputs are real pydantic models, and the producer edge is resolved by the real resolver.
"""

from __future__ import annotations

import re
from typing import Literal, Optional, get_origin

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import FieldOptions, UiOptions
from fastworkflow import planning_commands as pc
from fastworkflow.field_options import OPTION_KEYS, ConfirmsOptions, CopiedOptions, RecordNamedOptions, UiOptions
from fastworkflow.parameters import check as one_guard


def _params(model, planning=False):
    return {p.name: p for p in pc.option_params(model, "app/show", planning=planning)}


_DEFAULTS = {str: "", int: 0, bool: False, "list[str]": []}


def _field(annotation, extra, **kwargs):
    """A one-field Input model ``value`` of *annotation* with the Field options *extra* (a default that
    fits the type unless *kwargs* gives one; a Literal has none, so it is required)."""
    key = "list[str]" if annotation == list[str] else annotation
    if "default" not in kwargs and key in _DEFAULTS:
        kwargs["default"] = _DEFAULTS[key]
    return type("Input", (BaseModel,), {"__annotations__": {"value": annotation},
                                         "value": Field(json_schema_extra=extra, **kwargs)})


# ---------------------------------------------------------------- each key is the internal option its old spelling gave

@pytest.mark.parametrize("annotation, extra, expected", [
    # copied_from_command_text (old: in_request True / {"ignore": ...})
    (str, {"copied_from_command_text": True}, dict(in_request=True, subject=False, verify=False, ignore=())),
    (str, {"copied_from_command_text": {"ignore": ["the"]}},
     dict(in_request=True, subject=False, verify=False, ignore=("the",))),
    (int, {"copied_from_command_text": True}, dict(in_request=True, subject=False, verify=False)),
    (list[str], {"copied_from_command_text": True}, dict(in_request=True, subject=False, verify=False)),
    # record_named_in_command_text (old: in_request "subject" / subject + ignore)
    (str, {"record_named_in_command_text": True}, dict(in_request=True, subject=True, verify=False, ignore=())),
    (str, {"record_named_in_command_text": {"ignore": ["group", "unit"]}},
     dict(in_request=True, subject=True, verify=False, ignore=("group", "unit"))),
    # classifier_confirms (old: verify True / a meanings dict on an enum)
    (bool, {"classifier_confirms": True}, dict(verify=True, alone=False, in_request=False)),
    (Literal["a", "b"], {"classifier_confirms": {"meanings": {"a": "the first"}}},
     dict(verify=True, alone=False, means=(("a", "the first"),))),
    (str, {"record_named_in_command_text": True, "classifier_confirms": True},
     dict(in_request=True, subject=True, verify=True, alone=False)),
    # classifier_decides (old: verify "alone" on a bool)
    (bool, {"classifier_decides": True}, dict(verify=True, alone=True)),
])
def test_each_key_gives_the_internal_option_its_old_spelling_gave(annotation, extra, expected):
    model = _field(annotation, extra)
    param = _params(model, planning=True)["value"]
    for attr, value in expected.items():
        assert getattr(param, attr) == value, (attr, param)


def test_a_planning_command_field_with_no_check_option_is_not_a_param():
    model = _field(str, {"from_command_output": "find_book.uid"})
    assert pc.option_params(model, "app/show", planning=True) == ()


# ---------------------------------------------------------------- the same keys through a FieldOptions dict

def test_fieldoptions_is_a_plain_dict_the_loader_reads_the_same_as_a_dict_literal():
    literal = _field(str, {"copied_from_command_text": {"ignore": ["the"]}})
    typed = _field(str, FieldOptions(copied_from_command_text={"ignore": ["the"]}))
    assert type(FieldOptions(copied_from_command_text=True)) is dict
    assert _params(literal)["value"] == _params(typed)["value"]


def test_fieldoptions_names_the_option_keys_and_the_option_typeddicts_are_exported():
    """FieldOptions has one entry per option key (the six author options and ``ui``); every option TypedDict,
    ``UiOptions`` included, is exported identically on ``fastworkflow`` and ``fastworkflow.planning``."""
    assert set(FieldOptions.__annotations__) == set(OPTION_KEYS)
    assert {"ignore"} == set(CopiedOptions.__annotations__) == set(RecordNamedOptions.__annotations__)
    assert set(ConfirmsOptions.__annotations__) == {"meanings", "words"}
    assert set(UiOptions.__annotations__) == {"hidden", "hero", "format", "widget", "table", "tones", "currency"}
    for name in ("FieldOptions", "CopiedOptions", "RecordNamedOptions", "ConfirmsOptions", "UiOptions"):
        assert getattr(fastworkflow, name) is getattr(fastworkflow.planning, name)


# ---------------------------------------------------------------- the producer and positional keys

def test_command_text_after_command_name_is_read_as_the_positional_field_of_an_execution_command():
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction

    class Input(BaseModel):
        note: Optional[str] = Field(default=None, json_schema_extra={"command_text_after_command_name": True})
        other: str = Field(default="", description="a plain field")

    assert ParameterExtraction._is_positional_field(Input.model_fields["note"])
    assert not ParameterExtraction._is_positional_field(Input.model_fields["other"])
    # the old spelling is no longer read as the marker
    old = Input.model_fields["note"]
    assert not ParameterExtraction._is_positional_field(
        type("F", (), {"json_schema_extra": {"positional": True}})())
    assert old.json_schema_extra == {"command_text_after_command_name": True}


def test_two_command_text_after_command_name_fields_in_one_input_fail_the_load():
    class Input(BaseModel):
        first: str = Field(default="", json_schema_extra={"command_text_after_command_name": True})
        second: str = Field(default="", json_schema_extra={"command_text_after_command_name": True})

    with pytest.raises(pc.PlanningCommandFormatError, match="one field per Input takes the command text"):
        pc.option_params(Input, "app/show")


def test_only_a_marked_field_is_the_strict_check_and_an_execution_input_reads_it_through_argument_checks():
    class Input(BaseModel):
        due: str = Field(default="", description="the due date")
        name: str = Field(default="", description="a name", json_schema_extra={"copied_from_command_text": True})

    assert not one_guard.is_marked(Input.model_fields["due"])
    assert one_guard.is_marked(Input.model_fields["name"])
    # an execution command's Input: an unmarked field carries no check option, so it is no param
    assert "due" not in _params(Input) and "name" in _params(Input)


def _resolve(consumer_input_cls, producer_output_cls, *, consumer="Library/open_book"):
    """Resolve the producer edges of *consumer_input_cls* against a producer command ``Library/find_book``."""
    class Producer:
        """Find books."""

        class Input(BaseModel):
            query: str = Field(default="", description="the query")

        Output = producer_output_cls

    class Consumer:
        """Open a book."""

        Input = consumer_input_cls

    resolution = pc.FieldSourceResolution(
        workflow_folderpath="", problems=(), skipped_commands=frozenset(), model_to_command={},
        signatures={"Library/find_book": Producer, consumer: Consumer}, command_files={}, import_errors={},
        base_contexts={})
    problems = pc.resolve_signature_field_sources(consumer, Consumer, "open_book.py", resolution)
    return Consumer, problems


class _Output(BaseModel):
    book_uids: list[str] = Field(description="the uids of the matching books")


def test_from_command_output_writes_the_producer_edge_as_from_the_old_from_did():
    class Input(BaseModel):
        book_uid: str = Field(default="", description="the book", json_schema_extra={"from_command_output": "find_book.book_uids"})

    consumer, problems = _resolve(Input, _Output)
    assert problems == []
    assert list(consumer.Input.model_fields["book_uid"].json_schema_extra["available_from"]) == [
        "Library/find_book.book_uids"]


def test_from_command_output_may_be_a_list_of_sources():
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"from_command_output": ["find_book.book_uids"]})

    consumer, problems = _resolve(Input, _Output)
    assert problems == []
    assert "Library/find_book.book_uids" in list(consumer.Input.model_fields["book_uid"].json_schema_extra["available_from"])


def test_from_command_output_on_an_output_field_is_a_load_problem_naming_the_key():
    class Output(BaseModel):
        author: str = Field(default="", json_schema_extra={"from_command_output": "find_book.book_uids"})

    class Consumer(BaseModel):
        book_uid: str = Field(default="")

    class Input(BaseModel):
        book_uid: str = Field(default="")

    class Command:
        """Open a book."""

    Command.Input = Input
    Command.Output = Output
    resolution = pc.FieldSourceResolution(
        workflow_folderpath="", problems=(), skipped_commands=frozenset(), model_to_command={},
        signatures={"Library/open_book": Command}, command_files={}, import_errors={}, base_contexts={})
    problems = pc.resolve_signature_field_sources("Library/open_book", Command, "open_book.py", resolution)
    assert any("cannot declare from_command_output" in p.message for p in problems), problems


def test_a_hand_written_available_from_is_still_a_load_problem():
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"available_from": ["Library/find_book.book_uids"]})

    _, problems = _resolve(Input, _Output)
    assert any("available_from is written by from_command_output" in p.message for p in problems), problems


def test_from_as_a_key_on_an_input_is_a_load_error_naming_the_replacement():
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"from": "find_book.book_uids"})

    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(
            'from: "find_book.book_uids" is removed: write "from_command_output": "find_book.book_uids"')):
        pc.option_params(Input, "app/show")


# ---------------------------------------------------------------- each removed spelling fails, naming its replacement

@pytest.mark.parametrize("annotation, extra, message", [
    (str, {"in_request": True}, '"copied_from_command_text": True'),
    (str, {"in_request": {"ignore": ["the"]}}, '"copied_from_command_text": {"ignore": ["the"]}'),
    (str, {"in_request": "subject"}, '"record_named_in_command_text": True'),
    (str, {"in_request": {"subject": True, "ignore": ["group"]}}, '"record_named_in_command_text": {"ignore": ["group"]}'),
    (bool, {"verify": True}, '"classifier_confirms": True'),
    (bool, {"verify": "alone"}, '"classifier_decides": True'),
    (Literal["a", "b"], {"verify": {"a": "x"}}, '"classifier_confirms": {"meanings": {"a": "x"}}'),
    (str, {"positional": True}, '"command_text_after_command_name": True'),
    (str, {"subject": True}, '"record_named_in_command_text": True'),
    (Literal["a", "b"], {"means": {"a": "x"}}, '"classifier_confirms": {"meanings": {"a": "x"}}'),
    (str, {"strip": ["group"]}, '"copied_from_command_text": {"ignore": ["group"]}'),
])
def test_a_removed_spelling_fails_the_load_naming_its_replacement(annotation, extra, message):
    model = _field(annotation, extra)
    with pytest.raises(pc.PlanningCommandFormatError) as caught:
        pc.option_params(model, "app/show", planning=True)
    assert message in str(caught.value)
    assert "is removed: write" in str(caught.value)


def test_the_load_error_for_in_request_subject_reads_as_the_spec_example():
    class Input(BaseModel):
        who: str = Field(default="", json_schema_extra={"in_request": "subject"})

    with pytest.raises(pc.PlanningCommandFormatError) as caught:
        pc.option_params(Input, "app/who", planning=True)
    assert str(caught.value) == ('app/who: field \'who\': in_request: "subject" is removed: write '
                                 '"record_named_in_command_text": True')


def test_a_removed_spelling_on_an_execution_command_fails_the_load_too():
    class Input(BaseModel):
        who: str = Field(default="", json_schema_extra={"in_request": True})

    with pytest.raises(pc.PlanningCommandFormatError, match='"copied_from_command_text": True'):
        pc.option_params(Input, "app/show")


# ---------------------------------------------------------------- unknown keys, bad values, conflicts, types

@pytest.mark.parametrize("extra, suggestion", [
    ({"copied_from_command_txt": True}, "did you mean 'copied_from_command_text'?"),
    ({"record_named_in_command_txt": True}, "did you mean 'record_named_in_command_text'?"),
    ({"classifier_confirm": True}, "did you mean 'classifier_confirms'?"),
])
def test_an_unknown_key_suggests_the_closest_option(extra, suggestion):
    class Input(BaseModel):
        name: str = Field(default="", json_schema_extra=extra)

    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(suggestion)):
        pc.option_params(Input, "app/show", planning=True)


def test_an_unknown_key_on_a_planning_command_fails_even_without_a_close_match():
    class Input(BaseModel):
        name: str = Field(default="", json_schema_extra={"colour": "red"})

    with pytest.raises(pc.PlanningCommandFormatError, match="unknown option 'colour'"):
        pc.option_params(Input, "app/show", planning=True)


@pytest.mark.parametrize("extra", [{"in_request": False}, {"verify": False}])
def test_a_retired_key_with_no_replacement_says_leave_it_out(extra):
    class Input(BaseModel):
        name: str = Field(default="", json_schema_extra=extra)

    with pytest.raises(pc.PlanningCommandFormatError, match=r"is removed: leave the key out$"):
        pc.option_params(Input, "app/show", planning=True)


def test_an_execution_command_may_carry_a_json_schema_annotation_but_no_unknown_key():
    # One rule (bd fix-ftdl.8.22, stated in check_field_keys): an unknown key fails an Input field of either kind; an
    # execution command's Input also takes the JSON Schema annotations (its parameter-example generator reads
    # description and examples there), a planning command's takes its options only.
    class Input(BaseModel):
        kind: str = Field(default="", json_schema_extra={"examples": ["book"], "title": "Kind"})

    assert pc.option_params(Input, "app/show") == ()
    with pytest.raises(pc.PlanningCommandFormatError, match="unknown option 'examples'"):
        pc.option_params(Input, "app/show", planning=True)
    class Own(BaseModel):
        kind: str = Field(default="", json_schema_extra={"examples": ["book"], "x_note": "an author's own key"})

    with pytest.raises(pc.PlanningCommandFormatError, match="unknown option 'x_note'; its options are .*examples"):
        pc.option_params(Own, "app/show")
    class Bad(BaseModel):
        kind: str = Field(default="", json_schema_extra={"copied_from_command_txt": True})

    with pytest.raises(pc.PlanningCommandFormatError, match="did you mean 'copied_from_command_text'"):
        pc.option_params(Bad, "app/show")


@pytest.mark.parametrize("extra, message", [
    ({"copied_from_command_text": False}, "copied_from_command_text: False is not a value; leave the key out"),
    ({"record_named_in_command_text": False}, "record_named_in_command_text: False is not a value"),
    ({"classifier_decides": False}, "classifier_decides: False is not a value"),
    ({"copied_from_command_text": "yes"}, "copied_from_command_text must be True or a dict of ignore"),
    ({"copied_from_command_text": {}}, "copied_from_command_text is empty"),
    ({"copied_from_command_text": {"ignor": ["the"]}}, "did you mean 'ignore'?"),
    ({"copied_from_command_text": {"ignore": "the"}}, "copied_from_command_text ignore must be a list of words"),
    ({"classifier_confirms": {"meanings": {}}}, "classifier_confirms meanings must be a dict"),
    ({"classifier_decides": {"x": 1}}, "classifier_decides must be True"),
    ({"from_command_output": 3}, 'from_command_output must be "[Context/]command.field" text or a list of it'),
])
def test_a_false_or_bad_value_is_rejected(extra, message):
    model = _field(bool if "classifier_decides" in extra else str, extra)
    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(message)):
        pc.option_params(model, "app/show", planning=True)


@pytest.mark.parametrize("annotation, extra, message", [
    (str, {"copied_from_command_text": True, "record_named_in_command_text": True},
     "copied_from_command_text and record_named_in_command_text cannot both be on a field"),
    (bool, {"classifier_confirms": True, "classifier_decides": True},
     "classifier_confirms and classifier_decides cannot both be on a field"),
    (str, {"classifier_confirms": True}, "option classifier_confirms does not apply to field 'value' of type str"),
])
def test_conflicting_or_incomplete_pairs_fail_the_load(annotation, extra, message):
    model = _field(annotation, extra)
    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(message)):
        pc.option_params(model, "app/show", planning=True)


@pytest.mark.parametrize("annotation, extra, message, planning", [
    (bool, {"copied_from_command_text": True},
     "option copied_from_command_text does not apply to field 'value' of type bool", True),
    (int, {"record_named_in_command_text": True},
     "option record_named_in_command_text does not apply to field 'value' of type int (it takes a str)", True),
    (str, {"classifier_decides": True},
     "option classifier_decides does not apply to field 'value' of type str (it takes a bool)", True),
    (int, {"command_text_after_command_name": True},
     "option command_text_after_command_name does not apply to field 'value' of type int (it takes a str)", False),
    (Literal["a", "b"], {"copied_from_command_text": True},
     "option copied_from_command_text does not apply to field 'value' of type enum (it takes a str, an int or a list[str])", True),
])
def test_a_key_on_a_type_it_does_not_apply_to_fails_naming_the_types(annotation, extra, message, planning):
    model = _field(annotation, extra)
    with pytest.raises(pc.PlanningCommandFormatError, match=re.escape(message)):
        pc.option_params(model, "app/show", planning=planning)


def test_a_planning_command_field_rejects_the_execution_only_keys():
    class Input(BaseModel):
        note: str = Field(default="", json_schema_extra={"command_text_after_command_name": True})

    with pytest.raises(pc.PlanningCommandFormatError, match="command_text_after_command_name is for an execution command"):
        pc.option_params(Input, "app/show", planning=True)


def test_meanings_keys_must_be_values_of_the_literal():
    model = _field(Literal["a", "b"], {"classifier_confirms": {"meanings": {"c": "x"}}})
    with pytest.raises(pc.PlanningCommandFormatError, match="classifier_confirms meanings: 'c' is not a value of it"):
        pc.option_params(model, "app/show", planning=True)


def test_the_new_keys_read_together_on_one_planning_input():
    class Input(BaseModel):
        who: str = Field(description="the record", json_schema_extra={"record_named_in_command_text": {"ignore": ["group"]}})
        kind: Literal["group", "unit"] = Field(description="what it is",
                                               json_schema_extra={"classifier_confirms": {"meanings": {"group": "a group"}}})
        verified: bool = Field(default=False, description="asked", json_schema_extra={"classifier_decides": True})

    params = _params(Input, planning=True)
    assert (params["who"].subject, params["who"].ignore) == (True, ("group",))
    assert (params["kind"].verify, params["kind"].means) == (True, (("group", "a group"),))
    assert (params["verified"].alone, params["verified"].verify) == (True, True)


# ---------------------------------------------------------------- the ui hint (presentation only)

def test_ui_is_a_registered_option_key_and_not_a_check_option():
    assert "ui" in OPTION_KEYS
    assert "ui" not in pc.CHECK_OPTIONS


@pytest.mark.parametrize("annotation, extra", [
    (str, FieldOptions(ui=UiOptions(widget="textarea"))),
    (Optional[str], {"ui": {"widget": "textarea"}}),
    (Literal["a", "b"], {"ui": {"widget": "radio"}}),
    (Optional[Literal["a", "b"]], {"ui": {"widget": "radio"}}),
])
@pytest.mark.parametrize("planning", [True, False])
def test_a_valid_input_ui_hint_loads_and_makes_no_check_param(annotation, extra, planning):
    default = {} if get_origin(annotation) is Literal else {"default": None if "Optional" in str(annotation) else ""}
    model = _field(annotation, extra, **default)
    assert pc.option_params(model, "app/show", planning=planning) == ()
    assert model.model_json_schema()["properties"]["value"]["ui"] == extra["ui"]


def test_a_ui_hint_beside_a_check_option_keeps_the_check_param_unchanged():
    plain = _params(_field(str, {"copied_from_command_text": True}))["value"]
    hinted = _params(_field(str, {"copied_from_command_text": True, "ui": {"widget": "textarea"}}))["value"]
    assert plain == hinted


@pytest.mark.parametrize("planning", [True, False])
@pytest.mark.parametrize("annotation, extra, needles", [
    (str, {"ui": {"widgt": "textarea"}}, ["did you mean 'widget'?", "'value'"]),
    (str, {"ui": {"zzzzzz": True}}, ["its options are hidden, hero, format, widget, table"]),
    (int, {"ui": {"widget": "textarea"}}, ["textarea", "'value'", "a str"]),
    (str, {"ui": {"widget": "radio"}}, ["radio", "Literal"]),
    (Literal["a", "b"], {"ui": {"widget": "textarea"}}, ["textarea", "a str"]),
    (str, {"ui": {"widget": "dropdown"}}, ['"textarea" or "radio"', '"dropdown"']),
    (str, {"ui": {"hero": True}}, ["ui.hero", "Signature.Output"]),
    (str, {"ui": {"hidden": True}}, ["ui.hidden", "Signature.Output"]),
    (str, {"ui": {"format": "code"}}, ["ui.format", "Signature.Output"]),
    (str, {"ui": {}}, ["ui is empty", "widget"]),
    (str, {"ui": False}, ["ui: False is not a value", "leave the key out to turn it off"]),
    (str, {"ui": "textarea"}, ["ui must be a dict", '"textarea"']),
])
def test_an_invalid_input_ui_hint_fails_the_load_naming_the_field_the_key_and_what_is_accepted(
        annotation, extra, needles, planning):
    with pytest.raises(pc.PlanningCommandFormatError) as error:
        pc.option_params(_field(annotation, extra), "app/show", planning=planning)
    for needle in needles:
        assert needle in str(error.value), (needle, str(error.value))


def _output_problems(output_cls):
    class Command:
        """Show it."""

    class Input(BaseModel):
        query: str = Field(default="")

    Command.Input = Input
    Command.Output = output_cls
    resolution = pc.FieldSourceResolution(
        workflow_folderpath="", problems=(), skipped_commands=frozenset(), model_to_command={},
        signatures={"Library/show": Command}, command_files={}, import_errors={}, base_contexts={})
    return pc.resolve_signature_field_sources("Library/show", Command, "show.py", resolution)


@pytest.mark.parametrize("hint", [{"hidden": True}, {"hero": True, "format": "currency"}, {"format": "badge"},
                                  {"format": "percent"}, {"format": "code"}])
def test_a_valid_output_ui_hint_is_not_a_load_problem_and_is_in_the_output_json_schema(hint):
    class Output(BaseModel):
        total: float = Field(default=0.0, json_schema_extra=FieldOptions(ui=UiOptions(**hint)))

    assert _output_problems(Output) == []
    assert Output.model_json_schema()["properties"]["total"]["ui"] == hint


@pytest.mark.parametrize("hint, needles", [
    ({"widget": "textarea"}, ["ui.widget", "Signature.Input", "'total'"]),
    ({"hidden": False}, ["ui.hidden", "leave the key out to turn it off"]),
    ({"hero": False}, ["ui.hero", "leave the key out to turn it off"]),
    ({"hero": "yes"}, ["ui.hero must be True"]),
    ({"format": "dollars"}, ["ui.format", "'currency'", '"dollars"']),
    ({"formt": "code"}, ["did you mean 'format'?"]),
    ({}, ["ui is empty"]),
])
def test_an_invalid_output_ui_hint_is_a_field_source_problem_naming_the_bad_option(hint, needles):
    class Output(BaseModel):
        total: float = Field(default=0.0, json_schema_extra={"ui": hint})

    problems = _output_problems(Output)
    assert len(problems) == 1 and problems[0].severity == "error" and problems[0].command == "Library/show"
    assert problems[0].file == "show.py"
    for needle in needles:
        assert needle in problems[0].message, (needle, problems[0].message)


def test_an_output_ui_false_is_a_field_source_problem():
    class Output(BaseModel):
        total: float = Field(default=0.0, json_schema_extra={"ui": False})

    assert "leave the key out to turn it off" in _output_problems(Output)[0].message


@pytest.mark.parametrize("extra, needle", [
    ({"colour": "red"}, "unknown option 'colour'; an Output field takes ui"),
    ({"uii": {"format": "code"}}, "unknown option 'uii'; did you mean 'ui'?"),
    ({"verify": True}, "verify: True is removed: write"),
    ({"classifier_confirms": True}, "classifier_confirms is an option of an Input field"),
])
def test_an_output_field_takes_only_ui_and_json_schema_annotations(extra, needle):
    class Output(BaseModel):
        total: float = Field(default=0.0, json_schema_extra=extra)

    problems = _output_problems(Output)
    assert len(problems) == 1 and needle in problems[0].message, problems


def test_a_field_without_a_ui_key_is_unchanged_by_the_ui_check():
    class Output(BaseModel):
        total: float = Field(default=0.0, json_schema_extra={"examples": [1.0]})

    assert _output_problems(Output) == []
    pc.check_ui_option("total", {"examples": [1.0]}, "show.py")   # a no-op without the key


def test_a_workflow_with_input_and_output_ui_hints_loads_without_problems_and_serves_the_hints(tmp_path):
    """The real load path (command_preview.load_problems, the one run/train/the server report) accepts the
    hints on a copy of the hello workflow, and the schema listing behind GET /command_schemas carries them."""
    import shutil
    from pathlib import Path

    from fastworkflow import command_preview
    from fastworkflow.mcp_server import command_schema_listing

    hello = Path(__file__).resolve().parent / "hello_world_workflow"
    folder = tmp_path / "hello_ui"
    shutil.copytree(hello, folder, symlinks=True,
                    ignore=shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__"))
    command_file = folder / "_commands" / "add_two_numbers.py"
    source = command_file.read_text()
    source = source.replace(
        '        second_num: float = Field(description="Second number")\n',
        '        second_num: float = Field(description="Second number")\n'
        '        note: str = Field(default="", description="A note",\n'
        '                          json_schema_extra={"ui": {"widget": "textarea"}})\n', 1)
    source = source.replace(
        'Field(description="The sum of the two provided numbers")',
        'Field(description="The sum of the two provided numbers",\n'
        '                                        json_schema_extra={"ui": {"hero": True, "format": "currency"}})', 1)
    assert source.count('"ui"') == 2
    command_file.write_text(source)

    assert command_preview.load_problems(str(folder)) == []
    entry = command_schema_listing(str(folder))["add_two_numbers"]
    assert entry["input"]["properties"]["note"]["ui"] == {"widget": "textarea"}
    assert entry["output"]["properties"]["sum_of_two_numbers"]["ui"] == {"hero": True, "format": "currency"}


def test_a_workflow_with_a_misspelt_output_ui_hint_is_a_field_source_error_naming_the_fix(tmp_path):
    """``command preview --check`` reports ``resolve_field_sources`` errors, and the command directory logs the
    same problem and skips the command at hydration."""
    import shutil
    from pathlib import Path

    hello = Path(__file__).resolve().parent / "hello_world_workflow"
    folder = tmp_path / "hello_ui_typo"
    shutil.copytree(hello, folder, symlinks=True,
                    ignore=shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__"))
    command_file = folder / "_commands" / "add_two_numbers.py"
    source = command_file.read_text().replace(
        'Field(description="The sum of the two provided numbers")',
        'Field(description="The sum of the two provided numbers", json_schema_extra={"ui": {"formt": "code"}})', 1)
    command_file.write_text(source)

    errors = pc.resolve_field_sources(str(folder)).errors
    assert any("did you mean 'format'?" in error.message and error.command == "add_two_numbers"
               for error in errors), errors
