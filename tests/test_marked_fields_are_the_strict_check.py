"""The written-in-the-request check runs only on a field marked ``copied_from_command_text`` (or
``record_named_in_command_text``) (bd fix-qjne.21, owner decision).

An unmarked ``str`` or ``int`` field keeps the extractor's value: no "is not written in the request" refusal, no
description-example refusal, no number rule. What needs no knowledge of the request stays on for every field (a value
that is entirely one template token, a value of only punctuation). A marked field is
checked as it always was, and a marked ``int`` strictly: a number the request does not write is refused, with or
without any number in the request. The unmarked fields whose value the request does not write are recorded as an
advisory (``argument_check_not_in_request``, names only) on both paths. Real guards, real loader, real planner path;
the model boundary is the scripted extractor."""
from __future__ import annotations

import pytest
from dspy.utils import DummyLM
from pydantic import BaseModel, Field

from fastworkflow import planning_commands as pc
from fastworkflow.parameters import check
from tests.scripted_extractor import script_extractor
from tests.test_argument_checks import _command, _module
from tests.test_planning_command_commands import (  # noqa: F401 - workflow_env is a fixture
    REQUEST, _arguments, _command as _lookup_command, _plan_request, _plan_span, _typed_answer)
from tests.test_text_planner_and_split import RecordingTraceSink, _session, workflow_env  # noqa: F401

COPIED = {"copied_from_command_text": True}


class Plain(BaseModel):
    name: str = Field(default="", description="a name", examples=["Ann Lee"])
    count: int = Field(default=0, description="how many")


class Marked(BaseModel):
    name: str = Field(default="", description="a name", examples=["Ann Lee"], json_schema_extra=COPIED)
    count: int = Field(default=0, description="how many", json_schema_extra=COPIED)


# ---------------------------------------------------------------- execution commands: the guard

def test_an_unmarked_str_and_int_keep_the_extractors_value_on_an_execution_command():
    checked = check.check_execution(Plain, {"name": "Bob", "count": 8}, "find Ann, 7 rows")
    assert checked.gaps == [] and checked.respelled == {}
    assert checked.not_in_request == ["count", "name"]          # the advisory: names, sorted


def test_an_unmarked_str_equal_to_a_description_example_the_request_does_not_write_is_accepted():
    """There is no example check: the examples are not shown to the model, so a value equal to one is the model's own."""
    checked = check.check_execution(Plain, {"name": "Ann Lee", "count": 0}, "find the member")
    assert checked.gaps == [] and checked.not_in_request == ["name"]
    assert not hasattr(check, "example_rule") and not hasattr(check, "EXAMPLE_WHY")


@pytest.mark.parametrize("value", ["<name>", "{name}", "{{name}}", "..."])
def test_what_needs_no_knowledge_of_the_request_stays_on_for_an_unmarked_str(value):
    (gap,) = check.check_execution(Plain, {"name": value, "count": 0}, "find the member").gaps
    assert gap.name == "name" and gap.code in ("placeholder", "no_value")


def test_a_written_unmarked_value_is_not_in_the_advisory():
    assert check.check_execution(Plain, {"name": "Ann", "count": 7}, "find Ann, 7 rows").not_in_request == []


def _guard(model: type, values: dict, request: str) -> check.Checked:
    """The one guard of an execution command's marked fields (``check_plan`` through ``option_command``)."""
    return check.check_plan(pc.option_command(model, "app/find"), values, request)


def test_a_marked_str_is_refused_as_before_and_a_copied_example_is_just_unwritten():
    gap = _guard(Marked, {"name": "Bob", "count": 0}, "find Ann").stop
    assert (gap.name, gap.code, gap.why) == ("name", "not_written", check.NOT_WRITTEN_WHY)
    gap = _guard(Marked, {"name": "Ann Lee", "count": 0}, "find the member").stop
    assert (gap.code, gap.why) == ("not_written", check.NOT_WRITTEN_WHY)
    ok = _guard(Marked, {"name": "ann lee", "count": 0}, "find Ann Lee")
    assert ok.gaps == [] and ok.values["name"] == "Ann Lee"
    assert check.check_execution(Marked, {"name": "Bob", "count": 0}, "find Ann").gaps == []   # not the check stage's


@pytest.mark.parametrize("request_text, why", [
    ("find Ann, 7 rows", check.numbers_why({7})),
    ("find Ann", "is not a number the request writes"),         # no number written: still refused
])
def test_a_marked_int_the_request_does_not_write_is_refused(request_text, why):
    gap = _guard(Marked, {"name": "", "count": 8}, request_text).stop
    assert (gap.name, gap.code, gap.why) == ("count", "number", why)


def test_a_marked_int_the_request_writes_passes_and_one_the_extractor_did_not_give_is_not_checked():
    assert _guard(Marked, {"name": "", "count": 7}, "find Ann, 7 rows").gaps == []
    assert _guard(Marked, {"name": "", "count": 0}, "find Ann").gaps == []


def test_a_pattern_no_longer_exempts_and_does_not_check_the_request():
    class Id(BaseModel):
        plain: str = Field(default="", pattern=r"^#W\d+$")
        marked: str = Field(default="", pattern=r"^#W\d+$", json_schema_extra=COPIED)

    unmarked = check.check_execution(Id, {"plain": "#W123", "marked": ""}, "cancel order W123")
    assert unmarked.gaps == [] and unmarked.not_in_request == ["plain"]
    # a marked id field with a pattern is a copy check like any other: the request must write "#W123"
    gap = _guard(Id, {"plain": "", "marked": "#W123"}, "cancel order W123").stop
    assert (gap.name, gap.why) == ("marked", check.NOT_WRITTEN_WHY)
    assert _guard(Id, {"plain": "", "marked": "#W123"}, "cancel order #W123").gaps == []


def test_the_advisory_leaves_out_a_value_that_was_refused_and_one_not_given():
    checked = check.check_execution(Plain, {"name": "<name>", "count": 0}, "find the member")
    assert checked.not_in_request == []


# ---------------------------------------------------------------- planning commands: the guard

def _plan(tmp_path, field: str):
    return _command(tmp_path, _module(field), name="weigh")


def test_an_unmarked_str_keeps_the_extractors_value_on_a_planning_command(tmp_path):
    command = _plan(tmp_path, 'name: str = Field(description="n", examples=["Ann Lee"])')
    args, _ = pc.check_arguments(command, {"name": "Billing"}, "Show the Payroll group")
    assert args["name"] == "Billing"
    args, _ = pc.check_arguments(command, {"name": "Ann Lee"}, "Show the Payroll group")   # an example: no check
    assert args["name"] == "Ann Lee"
    model = command.input_model
    assert check.not_written_fields(model, {"name": "Billing"}, "Show the Payroll group") == ["name"]
    assert check.not_written_fields(model, {"name": "Payroll"}, "Show the Payroll group") == []


@pytest.mark.parametrize("value", ["<name>", "{name}", "..."])
def test_what_needs_no_knowledge_of_the_request_stays_on_for_an_unmarked_planning_str(tmp_path, value):
    command = _plan(tmp_path, 'name: str = Field(description="n")')
    with pytest.raises(pc.PlanningCommandInputError):
        pc.check_arguments(command, {"name": value}, "Show the Payroll group")


def test_a_marked_planning_str_is_refused_and_an_example_in_its_own_words(tmp_path):
    command = _plan(tmp_path, 'name: str = Field(description="n", examples=["Ann Lee"], json_schema_extra={"copied_from_command_text": True})')
    with pytest.raises(pc.PlanningCommandInputError, match="name='Billing' is not written in the request"):
        pc.check_arguments(command, {"name": "Billing"}, "Show the Payroll group")
    with pytest.raises(pc.PlanningCommandInputError, match="name='Ann Lee' is not written in the request"):
        pc.check_arguments(command, {"name": "Ann Lee"}, "Show the Payroll group")


def test_an_unmarked_planning_int_is_not_checked_and_a_marked_one_is_checked_strictly(tmp_path):
    plain = _plan(tmp_path / "a", 'count: int = Field(default=0, description="n")')
    marked = _plan(tmp_path / "b", 'count: int = Field(default=0, description="n", json_schema_extra={"copied_from_command_text": True})')
    for request in ("Open 3 members of Payroll", "Open the Payroll members"):
        args, _ = pc.check_arguments(plain, {"count": 5}, request)
        assert args["count"] == 5
    assert check.not_written_fields(plain.input_model, {"count": 5}, "Open 3 members") == ["count"]
    assert check.not_written_fields(marked.input_model, {"count": 5}, "Open 3 members") == []   # marked: refused, not advised
    with pytest.raises(pc.PlanningCommandInputError, match=r"count=5 is not a number the request writes \(3\)"):
        pc.check_arguments(marked, {"count": 5}, "Open 3 members of Payroll")
    with pytest.raises(pc.PlanningCommandInputError, match=r"count=5 is not a number the request writes$"):
        pc.check_arguments(marked, {"count": 5}, "Open the Payroll members")
    assert pc.check_arguments(marked, {"count": 3}, "Open 3 members of Payroll")[0]["count"] == 3
    # an optional marked int the extractor did not give is not checked
    assert pc.check_arguments(marked, {"count": None}, "Open the Payroll members")[0]["count"] == 0


def test_a_marked_id_field_with_a_pattern_is_a_copy_check_for_the_planning_guard():
    class Input(BaseModel):
        oid: str = Field(description="o", pattern=r"^#W\d+$", json_schema_extra=COPIED)

    command = pc.option_command(Input, "app/cancel")      # the fields that carry an option, as the guard reads them
    with pytest.raises(pc.PlanningCommandInputError, match="is not written in the request"):
        pc.check_arguments(command, {"oid": "#W123"}, "cancel order W123")
    assert pc.check_arguments(command, {"oid": "#W123"}, "cancel order #W123")[0]["oid"] == "#W123"


# ---------------------------------------------------------------- converted_from_command_text is gone

def test_the_option_key_and_its_readers_are_gone():
    from fastworkflow import field_options

    assert "converted_from_command_text" not in field_options.OPTION_KEYS
    assert not hasattr(field_options, "CONVERTED_FROM_COMMAND_TEXT")
    assert not hasattr(check, "is_converted") and not hasattr(check, "is_grounding_exempt")
    assert "converted_from_command_text" not in pc.REMOVED_FIELD_KEYS + pc.RETIRED_OPTIONS


# ---------------------------------------------------------------- the advisory on the spans

def test_the_execution_path_records_the_advisory_as_names_only_and_absent_when_empty():
    from fastworkflow._workflows.command_metadata_extraction import parameter_extraction as px
    from fastworkflow.parameters import pipeline

    diagnostics: dict = {}
    check_stage, settle = px._llm_guard("find Ann, 7 rows", diagnostics)
    verdict = check_stage(Plain.model_construct(name="Bob", count=8), 1)
    settle(Plain.model_construct(name="Bob", count=8), verdict)
    assert verdict.kind == pipeline.KIND_OK
    assert diagnostics["argument_check_not_in_request"] == ["count", "name"]
    assert "Bob" not in str(diagnostics["argument_check_not_in_request"])
    quiet: dict = {}
    check_stage, settle = px._llm_guard("find Ann, 7 rows", quiet)
    settle(Plain.model_construct(name="Ann", count=7), check_stage(Plain.model_construct(name="Ann", count=7), 1))
    assert "argument_check_not_in_request" not in quiet


def test_the_planner_span_records_the_same_attribute(tmp_path, workflow_env, monkeypatch, no_planner_key):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(REQUEST)
    # owner and count are unmarked: "Zed" and 5 are not written in the request
    script_extractor(monkeypatch, _arguments(_typed_answer(include_risk=True, owner="Zed", count=5)))
    lm = DummyLM([_typed_answer(include_risk=True)])
    _plan_request(ctx, wf, lm, plan_template="TEMPLATE", planning_command=_lookup_command(tmp_path))
    span = _plan_span(sink)
    assert span.attributes["plan_source"] == "skill_call"
    assert span.attributes["argument_check_not_in_request"] == ["count", "owner"]
    assert "Zed" not in str(span.attributes["argument_check_not_in_request"])


def test_the_planner_span_attribute_is_null_when_every_unmarked_value_is_written(tmp_path, workflow_env, monkeypatch,
                                                                                   no_planner_key):
    workflow_env()
    sink = RecordingTraceSink()
    ctx, wf = _session(sink)
    ctx._begin_turn(REQUEST)
    script_extractor(monkeypatch, _arguments(_typed_answer(include_risk=True, count=2)))
    lm = DummyLM([_typed_answer(include_risk=True)])
    _plan_request(ctx, wf, lm, plan_template="TEMPLATE", planning_command=_lookup_command(tmp_path))
    assert _plan_span(sink).attributes["argument_check_not_in_request"] is None


# ---------------------------------------------------------------- the extraction prompt: no examples, the pattern

def _descriptions(model: type) -> dict:
    from fastworkflow.utils.signatures import InputForParamExtraction

    signature = InputForParamExtraction.create_signature_from_pydantic_model(model)
    return {name: field.json_schema_extra["desc"] for name, field in signature.output_fields.items()}


class _Prompted(BaseModel):
    name: str = Field(default="", description="a name", examples=["Ann Lee"])
    oid: str = Field(default="", description="an order id", pattern=r"^#W\d+$", examples=["#W0000000"])


def test_the_extraction_prompt_of_an_execution_command_has_no_examples_and_carries_the_pattern():
    desc = _descriptions(_Prompted)
    assert "Ann Lee" not in desc["name"] and "Examples" not in desc["name"]
    assert "#W0000000" not in desc["oid"] and "Examples" not in desc["oid"]
    assert "Must match the pattern ^#W\\d+$" in desc["oid"]
    assert "pattern" not in desc["name"]
    # examples stay on the field for the generated few-shot examples and the hint in a question
    assert _Prompted.model_fields["name"].examples == ["Ann Lee"]


def test_the_extraction_prompt_of_a_planning_commands_arguments_is_the_same_signature(tmp_path):
    command = _plan(tmp_path, 'name: str = Field(description="n", examples=["Ann Lee"])')
    desc = _descriptions(command.input_model)
    assert "Ann Lee" not in desc["name"] and "Examples" not in desc["name"]
    # the planner's typed fill (the fallback) never carried the examples: its field text is the description
    assert "Ann Lee" not in str(pc.typed_signature(command).output_fields["name"].json_schema_extra)


def test_a_pattern_field_of_a_planning_command_reaches_the_prompt_through_the_same_helper():
    from fastworkflow.parameters import rules

    class Input(BaseModel):
        oid: str = Field(description="o", pattern=r"^#W\d+$")
        plain: str = Field(default="")

    assert rules.pattern_sentence(Input.model_fields["oid"]) == "Must match the pattern ^#W\\d+$"
    assert rules.pattern_sentence(Input.model_fields["plain"]) == ""
    assert "Must match the pattern ^#W\\d+$" in _descriptions(Input)["oid"]


def test_a_stand_in_word_is_a_value_and_a_unicode_dash_matches_a_hyphen():
    """There is no list of stand-in words (bd fix-ftdl.6.1): "Smith" and "John Doe" are values; a request that writes
    "Ann Lee–Smith" (an en dash) writes the value "Ann Lee-Smith"."""
    from fastworkflow.parameters import rules

    request = "find the member Ann Lee–Smith"
    folded = rules.fold(request).lower()
    assert not rules.placeholder_like("Ann Lee–Smith") and not rules.placeholder_like("John Doe")
    assert rules.written("Ann Lee-Smith", folded, request) == "Ann Lee–Smith"
