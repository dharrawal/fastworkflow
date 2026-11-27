"""A marked optional field the request does not give keeps its own default on an execution command.

The guard of a marked field (``planning_commands.check_arguments`` through ``option_command``) states an absent
``Optional[str] = None`` argument as ``""``. The planning path reads that back as None (``planning_commands
._absent_as_none``) before any code sees it; the execution path wrote the ``""`` onto the ``Input``, so a command body
got ``""`` for "not given" and could not tell it from an empty filter. Both paths now apply the one conversion, and an
optional ``int`` and a field whose default is not None come back as their own defaults.

Real guards and the real ``validate_parameters``; no model is called.
"""
from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel, Field

from fastworkflow.planning import planning_commands as pc
from fastworkflow.parameters.signatures import InputForParamExtraction

COPIED = {"copied_from_command_text": True}


class Listing(BaseModel):
    filter: Optional[str] = Field(default=None, description="text to filter on", json_schema_extra=COPIED)
    limit: Optional[int] = Field(default=None, description="how many rows", json_schema_extra=COPIED)
    scope: str = Field(default="all", description="which rows", json_schema_extra=COPIED)
    page_size: int = Field(default=5, description="rows a page", json_schema_extra=COPIED)
    note: Optional[str] = Field(default=None, description="a note")


NOT_GIVEN = {"filter": None, "limit": None, "scope": "all", "page_size": 5, "note": None}


def _validated(values: dict, request: str) -> tuple[bool, BaseModel]:
    """``validate_parameters`` of an execution command (the stage every form of its extraction reaches)."""
    params = Listing.model_construct(**values)
    extractor = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
    valid, _message, _suggestions, _fields = extractor.validate_parameters(None, "app/list_rows", params)
    return valid, params


def test_marked_optional_fields_not_given_keep_their_own_defaults_on_an_execution_command():
    valid, params = _validated(dict(NOT_GIVEN), "list them all")
    assert valid is True
    assert {name: getattr(params, name) for name in NOT_GIVEN} == NOT_GIVEN


def test_a_marked_optional_str_given_in_the_request_keeps_its_value():
    valid, params = _validated({**NOT_GIVEN, "filter": "dune"}, "list the rows matching Dune")
    assert valid is True and params.filter == "Dune"


@pytest.mark.parametrize("field, default", [("filter", None), ("limit", None), ("scope", "all"),
                                            ("page_size", 5)])
def test_the_planning_and_execution_paths_state_an_absent_argument_alike(field, default):
    """The guard's output for an absent argument, read through the one conversion, is the field's own default."""
    command = pc.option_command(Listing, "app/list_rows")
    args, _ = pc.check_arguments(command, {name: None for name in ("filter", "limit", "scope", "page_size")},
                                 "list them all")
    assert pc._absent_as_none(command, args)[field] == default
