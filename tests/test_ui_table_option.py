"""The ``ui.table`` option of a ``Signature.Output`` field (fix-weih.23): which list fields are the columns of one table.

The load check is the one the other ui options use (``resolve_signature_field_sources``), so ``fastworkflow run_cli``,
``train``, the server and ``command preview --check`` report a refusal the same way.
"""
from __future__ import annotations

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
    return pc.resolve_signature_field_sources("Library/show", Command, "show.py", resolution)


def _col(annotation, **ui):
    return Field(default_factory=list, json_schema_extra=FieldOptions(ui=UiOptions(**ui)))


def test_two_list_fields_of_one_group_are_accepted_and_the_group_is_in_the_output_json_schema():
    class Output(BaseModel):
        branches: list[str] = _col(list[str], table="rows")
        hours: list[str] = _col(list[str], table="rows")

    assert _problems(Output) == []
    assert Output.model_json_schema()["properties"]["hours"]["ui"] == {"table": "rows"}


@pytest.mark.parametrize("annotation", [
    list[str], list[int], list[float], list[bool], list[Literal["a", "b"]],
    Optional[list[str]], Optional[list[Literal["a", "b"]]]])
def test_each_list_of_scalars_is_accepted_as_a_column(annotation):
    class Output(BaseModel):
        first: annotation = Field(default=None if "None" in str(annotation) else [],
                                  json_schema_extra={"ui": {"table": "t"}})
        second: list[str] = Field(default_factory=list, json_schema_extra={"ui": {"table": "t"}})

    assert _problems(Output) == []


@pytest.mark.parametrize("annotation", [str, int, dict[str, str], list[dict[str, str]], list[list[str]],
                                        list[Optional[str]], Optional[int]])
def test_a_column_that_is_not_a_list_of_scalars_is_refused(annotation):
    class Output(BaseModel):
        first: annotation = Field(default=None, json_schema_extra={"ui": {"table": "t"}})
        second: list[str] = Field(default_factory=list, json_schema_extra={"ui": {"table": "t"}})

    problems = _problems(Output)
    assert len(problems) == 1 and problems[0].severity == "error" and problems[0].command == "Library/show"
    assert "ui.table is for a list of" in problems[0].message and "'first'" in problems[0].message


def test_a_group_with_one_member_is_refused():
    class Output(BaseModel):
        alone: list[str] = _col(list[str], table="rows")
        other: list[str] = Field(default_factory=list)

    problems = _problems(Output)
    assert len(problems) == 1
    assert "group 'rows'" in problems[0].message and "at least two" in problems[0].message


def test_a_group_needs_two_visible_members_but_a_hidden_member_belongs_to_it():
    class Hidden(BaseModel):
        row_id: list[str] = _col(list[str], table="rows", hidden=True)
        name: list[str] = _col(list[str], table="rows")
        hours: list[str] = _col(list[str], table="rows")

    assert _problems(Hidden) == []

    class TooFewVisible(BaseModel):
        row_id: list[str] = _col(list[str], table="rows", hidden=True)
        name: list[str] = _col(list[str], table="rows")

    problems = _problems(TooFewVisible)
    assert len(problems) == 1 and "fewer than two visible" in problems[0].message


@pytest.mark.parametrize("group", ["", "two words", "1abc", "a-b", "a.b"])
def test_a_group_name_that_is_not_a_simple_identifier_is_refused(group):
    class Output(BaseModel):
        first: list[str] = Field(default_factory=list, json_schema_extra={"ui": {"table": group}})
        second: list[str] = Field(default_factory=list, json_schema_extra={"ui": {"table": group}})

    problems = _problems(Output)
    assert len(problems) == 2 and all("ui.table must be a group name" in p.message for p in problems)


@pytest.mark.parametrize("group", [3, True, ["rows"], None])
def test_a_group_name_that_is_not_text_is_refused(group):
    class Output(BaseModel):
        first: list[str] = Field(default_factory=list, json_schema_extra={"ui": {"table": group}})

    problems = _problems(Output)
    assert len(problems) == 1 and "ui.table must be a group name" in problems[0].message


def test_hero_and_table_on_one_field_is_refused():
    class Output(BaseModel):
        first: list[str] = Field(default_factory=list, json_schema_extra={"ui": {"table": "t", "hero": True}})
        second: list[str] = _col(list[str], table="t")

    problems = _problems(Output)
    assert len(problems) == 1 and "ui.hero" in problems[0].message and "ui.table" in problems[0].message


def test_format_and_table_together_are_accepted():
    class Output(BaseModel):
        code: list[str] = _col(list[str], table="t", format="code")
        hours: list[str] = _col(list[str], table="t")

    assert _problems(Output) == []


def test_table_on_an_input_field_is_refused():
    with pytest.raises(pc.PlanningCommandFormatError, match="ui.table is for a Signature.Output field"):
        pc.check_ui_option("rows", {"ui": {"table": "t"}}, "Signature.Input", kind="other")


def test_two_groups_are_checked_apart():
    class Output(BaseModel):
        a1: list[str] = _col(list[str], table="one")
        a2: list[str] = _col(list[str], table="one")
        b1: list[str] = _col(list[str], table="two")

    problems = _problems(Output)
    assert len(problems) == 1 and "group 'two'" in problems[0].message


def test_a_misspelt_option_names_table_among_the_choices():
    class Output(BaseModel):
        a: list[str] = Field(default_factory=list, json_schema_extra={"ui": {"tabel": "t"}})

    problems = _problems(Output)
    assert len(problems) == 1 and "did you mean 'table'?" in problems[0].message
