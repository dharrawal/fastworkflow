"""Unit tests for the parameter-extraction robustness helpers.

Covers _unwrap_optional (Optional/Union unwrapping) and the XML list-field
extraction (repeated tags collected via findall, robust list parsing). Pure
logic — no LLM/network.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    _unwrap_optional,
    ParameterExtraction,
)


# ---------------------------------------------------------------------------
# _unwrap_optional
# ---------------------------------------------------------------------------

def test_unwrap_optional_plain_type_unchanged():
    inner, origin = _unwrap_optional(str)
    assert inner is str
    assert origin is None


def test_unwrap_optional_unwraps_optional():
    inner, origin = _unwrap_optional(Optional[str])
    assert inner is str


def test_unwrap_optional_unwraps_optional_list_to_list_origin():
    inner, origin = _unwrap_optional(Optional[List[str]])
    # Origin should be list so callers can detect list fields.
    assert origin is list


def test_unwrap_optional_union_with_none():
    inner, _ = _unwrap_optional(Union[int, None])
    assert inner is int


# ---------------------------------------------------------------------------
# _extract_parameters_from_xml — list fields via repeated tags
# ---------------------------------------------------------------------------

class _ListParams(BaseModel):
    order_id: str = Field(default="NOT_FOUND")
    item_ids: List[str] = Field(default_factory=list)


def test_xml_extraction_collects_repeated_list_tags():
    # An agent may repeat a list tag once per item; all must be collected
    # (regression: re.search dropped all but the first).
    command = (
        "<order_id>#W1</order_id> "
        "<item_ids>a</item_ids> <item_ids>b</item_ids> <item_ids>c</item_ids>"
    )
    result = ParameterExtraction._extract_parameters_from_xml(command, _ListParams)
    assert result is not None
    assert result.order_id == "#W1"
    assert result.item_ids == ["a", "b", "c"]


def test_xml_extraction_single_list_item():
    command = "<order_id>#W1</order_id> <item_ids>only</item_ids>"
    result = ParameterExtraction._extract_parameters_from_xml(command, _ListParams)
    assert result is not None
    assert result.item_ids == ["only"]


class _MultiParams(BaseModel):
    order_id: str = Field(default="NOT_FOUND")
    reason: str = Field(default="NOT_FOUND")


def test_xml_extraction_multiple_scalar_fields():
    command = "<order_id>#W123</order_id> <reason>no longer needed</reason>"
    result = ParameterExtraction._extract_parameters_from_xml(command, _MultiParams)
    assert result is not None
    assert result.order_id == "#W123"
    assert result.reason == "no longer needed"


def test_xml_extraction_no_params_returns_empty_model():
    class _NoParams(BaseModel):
        pass

    result = ParameterExtraction._extract_parameters_from_xml("anything", _NoParams)
    assert result is not None
    assert isinstance(result, _NoParams)


# ---------------------------------------------------------------------------
# _extract_parameters_from_xml — an omitted OPTIONAL tag is not a regex failure
# (ido-mn1.6.12)
# ---------------------------------------------------------------------------
#
# `all_fields_extracted = len(extracted_data) == len(field_names)` demanded a tag
# for every declared field, and returning None is what `_extract_impl` reads as
# "regex failed" — so it answered with `extract_parameters`, an LLM round trip
# (`extraction_method="llm"`). A command whose docstring TELLS the agent to omit
# its optional parameters therefore bought a model call on every ordinary call,
# to conclude that the omitted optionals are their defaults.


class _OptionalParams(BaseModel):
    handle: str = Field(description="required")
    cursor: Optional[str] = Field(default=None)
    contains: Optional[str] = Field(default=None)


def test_omitted_optional_tags_are_satisfied_by_their_defaults():
    result = ParameterExtraction._extract_parameters_from_xml(
        "<handle>abc</handle>", _OptionalParams
    )
    assert result is not None, "regex declined; the runtime would call the LLM"
    assert result.handle == "abc"
    assert result.cursor is None
    assert result.contains is None


def test_a_supplied_optional_still_wins_over_its_default():
    result = ParameterExtraction._extract_parameters_from_xml(
        "<handle>abc</handle> <cursor>c2</cursor>", _OptionalParams
    )
    assert result is not None
    assert result.cursor == "c2"
    assert result.contains is None


def test_an_omitted_REQUIRED_tag_still_falls_back_to_the_llm():
    """The other half of the rule. Only the LLM can find a value in prose."""
    assert (
        ParameterExtraction._extract_parameters_from_xml(
            "<cursor>c2</cursor>", _OptionalParams
        )
        is None
    )


class _SentinelParams(BaseModel):
    """fastWorkflow's own convention for a REQUIRED parameter.

    The whole `retail_workflow` is written this way: a default of `NOT_FOUND`
    plus a pattern that accepts the sentinel. Pydantic calls the field optional
    because it has a default, but the default IS the missing marker — so the
    rule cannot be "any field with a default is satisfied", or every command in
    that workflow would stop consulting the LLM for parameters nobody supplied.
    """

    order_id: str = Field(default="NOT_FOUND")
    reason: str = Field(default="NOT_FOUND")


def test_a_field_defaulted_to_the_missing_sentinel_is_not_satisfied():
    assert (
        ParameterExtraction._extract_parameters_from_xml(
            "<order_id>#W1</order_id>", _SentinelParams
        )
        is None
    )


def test_unlabelled_text_still_goes_to_the_llm():
    """An all-optional command called with prose and no tags.

    Nothing was extracted and the caller still supplied text, so that text was
    meant for a field. Answering with defaults would drop it silently; only the
    LLM can place it.
    """

    class _AllOptional(BaseModel):
        note: Optional[str] = Field(default=None)
        other: Optional[str] = Field(default=None)

    assert (
        ParameterExtraction._extract_parameters_from_xml(
            "remind me about the invoice", _AllOptional
        )
        is None
    )


def test_an_all_optional_command_called_bare_extracts_deterministically():
    class _AllOptional(BaseModel):
        note: Optional[str] = Field(default=None)
        other: Optional[str] = Field(default=None)

    result = ParameterExtraction._extract_parameters_from_xml("  ", _AllOptional)
    assert result is not None
    assert result.note is None
    assert result.other is None
