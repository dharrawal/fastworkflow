"""Argument checks on an execution command's LLM parameter extraction (fix-qjf4).

The checks a planning command's typed call gets (``planning_commands.check_arguments``) also run on
the values the LLM extractor gives an execution command (``fastworkflow.parameters.argument_checks``): a
``str`` value must be written in the request and takes the request's spelling, a placeholder is
refused, and an ``int`` must be a
number the request writes. A refused value makes the extractor run once more with the refusal
appended to the command text; a value still refused is treated as not given (asked for when
required, the default when optional).

Integration: real workflows (``tests/planning_tutorial_workflow``, and the bundled retail example
for a ``Literal`` field and a ``pattern`` field), the real routing,
``ParameterExtraction``, the real DSPy extractor and the real validator. The model boundary is a
scripted ``DummyLM`` that records each request, so the retry's note can be read where a live model
would read it. No test calls a model.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import dspy
import pytest
from dspy.utils import DummyLM
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction
from fastworkflow.parameters import check as one_guard, rules
from fastworkflow.parameters import signatures
from fastworkflow.parameters.argument_checks import check_extracted_values
from fastworkflow.parameters.signatures import NOT_FOUND

TUTORIAL = str(Path(__file__).parent.joinpath("planning_tutorial_workflow").resolve())
RETAIL = str(Path(__file__).parent.parent.joinpath("fastworkflow", "examples", "retail_workflow").resolve())
NOTE_START = "(A first extraction from this command was refused:"


class _ScriptedLM(DummyLM):
    """Answers the extractor's calls in order and records the text of each request."""

    def __init__(self, answers):
        super().__init__(answers, adapter=dspy.JSONAdapter())
        self.prompts: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture(autouse=True)
def _framework(tmp_path, monkeypatch):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    # Jev's "is this field given" check is a model call: off, so the scripted LM answers alone.
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "off")
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture
def extract(monkeypatch):
    """Run one extraction round of *command_name* in *folder* (assistant mode: the LLM path) with the
    extractor answering *answers* in order. Returns (output, diagnostics, lm)."""
    def run(folder: str, command_name: str, text: str, *answers: dict):
        fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
        workflow = fastworkflow.Workflow.create(folder, workflow_id_str=f"checks_{uuid.uuid4().hex[:8]}")
        # One answer per extraction: BestOfN stops at an answer that copies no description example.
        lm = _ScriptedLM(list(answers))
        monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
        diagnostics: dict = {}
        cme = SimpleNamespace(context={})
        output = ParameterExtraction(cme, workflow, command_name, text)._extract_impl(diagnostics)
        return output, diagnostics, lm
    return run


def _calls(lm: _ScriptedLM) -> int:
    return len(lm.prompts)


# ---------------------------------------------------------------- grounding (planning_tutorial_workflow)

def test_an_invented_str_value_is_refused_retried_once_and_then_asked_for(extract):
    output, diagnostics, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee",
        {"reasoning": "name", "query": "Bob Jones"},
        {"reasoning": "name again", "query": "Bob Jones"})

    # the marked field's one guard (``check_plan``, at the validation stage; bd fix-ftdl.6.3) refuses it
    assert diagnostics["argument_refusal"] == "query='Bob Jones' is not written in the request"
    assert _calls(lm) == 2
    # the second request carries the refusal, as typed_planning_call's REFUSED_CALL_NOTE does
    assert NOTE_START not in lm.prompts[0]
    assert NOTE_START in lm.prompts[1]
    assert "query='Bob Jones' is not written in the request" in lm.prompts[1]
    # still refused: the field is asked for, the question saying why the value was not taken (the plan path's words)
    assert output.parameters_are_valid is False
    assert output.cmd_parameters.query == NOT_FOUND
    assert "the query 'Bob Jones' is not in your request. Which query should I use?" in output.error_msg


def test_the_retry_fixes_a_refused_value(extract):
    output, diagnostics, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee",
        {"reasoning": "name", "query": "Ann Leigh"},
        {"reasoning": "copied", "query": "Ann Lee"})

    assert "query='Ann Leigh' is not written in the request" in lm.prompts[1]   # the one guard's refusal, read once more
    assert "argument_refusal" not in diagnostics and "argument_check_refusals_final" not in diagnostics
    assert _calls(lm) == 2
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.query == "Ann Lee"


def test_a_grounded_value_takes_the_request_spelling_without_a_retry(extract):
    output, diagnostics, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee",
        {"reasoning": "name", "query": "ann lee."})

    assert _calls(lm) == 1
    assert "argument_check_retry" not in diagnostics and "argument_refusal" not in diagnostics
    assert diagnostics["argument_notes"] == ["query='ann lee.' -> 'Ann Lee'"]   # the one guard's respelling
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.query == "Ann Lee"


def test_a_dash_variant_is_grounded(extract):
    output, _, lm = extract(
        TUTORIAL, "find_member", "find the member Ann Lee–Smith",
        {"reasoning": "name", "query": "Ann Lee-Smith"})

    assert _calls(lm) == 1
    assert output.cmd_parameters.query == "Ann Lee–Smith"


# ---------------------------------------------------------------- numbers (planning_tutorial_workflow)

@pytest.mark.parametrize("start", [None, signatures.INVALID_INT_VALUE])
def test_an_absent_optional_int_is_not_refused_when_the_request_writes_numbers(extract, start):
    """The planning rule refused an absent ``Optional[int] = None`` whenever the request wrote two or
    more numbers (22 of 974 Haiku calls in the replay); an absent value is not checked."""
    output, diagnostics, lm = extract(
        TUTORIAL, "list_loans", "list the loans of book b1 for the 2 or 3 members who asked",
        {"reasoning": "book only", "book_uid": "b1", "start": start})

    assert _calls(lm) == 1
    assert "argument_check_refusals" not in diagnostics
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.book_uid == "b1"
    assert output.cmd_parameters.start is None


def test_an_unmarked_int_the_request_does_not_write_stands_and_is_recorded(extract):
    """``start`` carries no ``copied_from_command_text`` (an ``Optional[int]`` cannot): One number written (10) and
    another extracted (20): the extractor's value stands (bd fix-qjne.21), recorded as an advisory, names only. A MARKED
    int is refused (``test_a_marked_int_is_refused_...`` below and ``test_marked_fields_are_the_strict_check``)."""
    output, diagnostics, lm = extract(
        TUTORIAL, "list_loans", "list the loans of book b1 from row 10",
        {"reasoning": "page", "book_uid": "b1", "start": 20})

    assert "argument_check_refusals" not in diagnostics
    assert diagnostics["argument_check_not_in_request"] == ["start"]
    assert _calls(lm) == 1
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.start == 20


def test_an_int_the_request_writes_passes(extract):
    output, diagnostics, lm = extract(
        TUTORIAL, "list_loans", "list the loans of book b1 from row 10",
        {"reasoning": "page", "book_uid": "b1", "start": 10})

    assert _calls(lm) == 1
    assert "argument_check_refusals" not in diagnostics
    assert output.cmd_parameters.start == 10


# ---------------------------------------------------------------- allowed values (a Literal), exempt field (retail)

def test_an_unmarked_pattern_field_keeps_the_converted_value_and_a_literal_reason_is_taken_as_typed(extract):
    """retail ``cancel_pending_order.reason`` is a ``Literal`` (the one spelling of allowed values; its refusals are in
    ``test_one_spelling_fields``). ``order_id`` declares ``Field(pattern=...)`` and no ``copied_from_command_text``: a
    converted value (``#W123`` of "order W123") is left as the extractor wrote it (an unmarked field is not compared with
    the request; the pattern only validates)."""
    output, diagnostics, lm = extract(
        RETAIL, "cancel_pending_order", "cancel order W123, I ordered it by mistake",
        {"reasoning": "id and reason", "order_id": "#W123", "reason": "ordered by mistake"})

    assert "argument_check_refusals" not in diagnostics
    assert "Literal['no longer needed', 'ordered by mistake']" in lm.prompts[0]
    assert output.cmd_parameters.order_id == "#W123"
    assert output.cmd_parameters.reason == "ordered by mistake"


def test_an_allowed_value_takes_the_declared_spelling(extract):
    """A ``Literal`` member written in another case takes its declared spelling, silently (recorded as a respelling)."""
    output, diagnostics, lm = extract(
        RETAIL, "cancel_pending_order", "cancel order #W123, no longer needed",
        {"reasoning": "id and reason", "order_id": "#W123", "reason": "No Longer Needed"})

    assert _calls(lm) == 1
    assert diagnostics["argument_check_respelled"] == {"reason": "no longer needed"}
    assert output.cmd_parameters.reason == "no longer needed"


def test_the_switch_turns_the_checks_off(extract, monkeypatch):
    """``count_overdue.branch`` carries no option. On, the check stage refuses a whole template token; off, the check
    stage says nothing, and the one template-token verdict, which no option turns off and every path reaches, refuses
    it at validation instead (``screen_parameter_values``; bd fix-ftdl.6.1)."""
    on = extract(TUTORIAL, "count_overdue", "how many loans are overdue at the branch",
                 {"reasoning": "branch", "branch": "{branch}"}, {"reasoning": "branch", "branch": "{branch}"})
    assert on[1]["argument_check_refusals"] == ["branch='{branch}' is a placeholder, not a value"]
    monkeypatch.setitem(fastworkflow._env_vars, "FW_EXTRACTION_ARGUMENT_CHECKS", "off")
    output, diagnostics, lm = extract(
        TUTORIAL, "count_overdue", "how many loans are overdue at the branch",
        {"reasoning": "branch", "branch": "{branch}"}, {"reasoning": "branch", "branch": "{branch}"})

    assert _calls(lm) == 2
    assert "argument_check_refusals" not in diagnostics
    assert diagnostics["placeholder_values"] == {"branch": "{branch}"}
    assert not output.parameters_are_valid


# ---------------------------------------------------------------- the check itself, field by field

COPIED = {"copied_from_command_text": True}


class _Fields(BaseModel):
    # name, kind and count are marked (the strict check); due and code are not (the extractor's value stands)
    name: str = Field(description="a person's name", json_schema_extra=COPIED)
    due: str = Field(description="the due date, YYYY-MM-DD")
    kind: Optional[str] = Field(default=None, description="the kind of record. One of: book, member, branch",
                                json_schema_extra=COPIED)
    code: str = Field(default="", description="a code", examples=["XY-12"])
    count: int = Field(default=5, description="how many", json_schema_extra=COPIED)
    start: Optional[int] = Field(default=None, description="first row")


def _check(request: str, **values):
    defaults = {"name": NOT_FOUND, "due": NOT_FOUND, "kind": None, "code": "", "count": 5, "start": None}
    return check_extracted_values(_Fields.model_construct(**{**defaults, **values}), request)


def _guarded(request: str, **values):
    """The one guard of ``_Fields``' marked fields (``check_plan`` through ``option_command``, bd fix-ftdl.6.3): the
    refused names (``invalid``) and the guarded values (``values``) of every gap, as ``argument_gaps`` works them out."""
    from fastworkflow.planning import planning_commands as pc

    defaults = {"name": None, "kind": None, "count": 5}
    command = pc.option_command(_Fields, "app/find")
    gaps = pc.argument_gaps(command, {**defaults, **{k: v for k, v in values.items() if k in defaults}}, request)
    return {name for name, _ in gaps.invalid}, gaps.values


def test_an_unmarked_field_keeps_a_converted_value_and_records_it():
    check = _check("add Ann Lee, due next Monday", name="Ann Lee", due="2026-11-09")
    assert check.refused == {} and check.not_in_request == ["due"]


def test_an_unmarked_field_still_refuses_a_placeholder():
    check = _check("add Ann Lee, due next Monday", name="Ann Lee", due="{due_date}")
    assert check.refused == {"due": ("{due_date}", "is a placeholder, not a value")}


def test_one_of_in_the_description_is_prose_not_an_allowed_list():
    """The words ``One of: a, b`` of a description are read by nobody (a ``Literal`` is the one spelling): a value
    outside them is only held to the grounding rule, and a written one passes with no respelling."""
    check = _check("find the shelf Ann", name="Ann", kind="shelf")
    assert check.refused == {} and check.respelled == {}
    assert _guarded("find the shelf Ann", name="Ann", kind="shelf") == (set(), {"name": "Ann", "kind": "shelf", "count": 5})
    assert _guarded("find the member Ann", name="Ann", kind="Member")[1]["kind"] == "member"   # the request's spelling


def test_a_value_equal_to_a_description_example_is_not_refused():
    """The examples are not shown to the extraction model, so there is no example check (bd fix-qjne.21)."""
    assert _check("add Ann Lee", name="Ann Lee", code="XY-12").refused == {}
    assert _check("add Ann Lee with code XY-12", name="Ann Lee", code="XY-12").refused == {}


class _Converted(BaseModel):
    origin: str = Field(description="IATA code", examples=["JFK"])
    date: str = Field(description="YYYY-MM-DD", examples=["2024-05-20"], pattern=r"^\d{4}-\d{2}-\d{2}$")


def test_a_converted_value_equal_to_a_description_example_is_not_refused():
    """tau-bench airline: "New York ... May 20" gives origin JFK and date 2024-05-20, both the fields' own
    examples. The fields are unmarked and the examples are not shown to the model: nothing refuses either."""
    params = _Converted.model_construct(origin="JFK", date="2024-05-20")
    assert check_extracted_values(params, "fly from New York to Seattle on May 20").refused == {}


def test_punctuation_alone_is_refused():
    assert _check("find the member Ann", name="...").refused == {"name": ("...", one_guard.NO_VALUE_WHY)}


def test_values_not_given_are_not_checked():
    check = _check("find twenty two books", name="<UNKNOWN>", due="&lt;UNKNOWN&gt;", count=5, start=None)
    assert check.refused == {}


def test_number_words_count_as_written_numbers():
    assert _guarded("show eight rows", count=8)[0] == set()
    assert "count" in _guarded("show eight rows", count=9)[0]


def test_a_compound_number_word_is_one_number_so_25_passes_and_20_is_refused():
    """bd fix-9c0z: "page twenty five" writes 25 and no 20 or 5."""
    assert _guarded("show page twenty five", count=25)[0] == set()
    assert "count" in _guarded("show page twenty five", count=20)[0]
    assert "count" in _guarded("show page twenty five", count=6)[0]


def test_a_decimal_writes_no_whole_number_so_a_marked_int_is_refused():
    """bd fix-9c0z: "last 3.5 days" writes no whole number, so a MARKED int is refused whatever it is (bd fix-qjne.21:
    "no number read, no check" is gone): it is not accepted on the strength of the 3 inside the decimal either."""
    assert rules.written_numbers("show the last 3.5 days", []) == set()
    assert "count" in _guarded("show the last 3.5 days", count=3)[0]
    assert "count" in _guarded("show the last 3.5 days", count=9)[0]
    # with another whole number written, the rule does read it: 7 passes, the 3 of the decimal is refused
    assert _guarded("show the last 3.5 days of 7 rows", count=7)[0] == set()
    assert "count" in _guarded("show the last 3.5 days of 7 rows", count=3)[0]


def test_digits_inside_a_written_str_value_are_not_numbers_the_request_writes():
    """``Doyle and Sons 1``: the 1 belongs to the name, so the request writes no number for count: a marked int is
    refused."""
    assert "count" in _guarded("find Doyle and Sons 1", name="Doyle and Sons 1", count=7)[0]


def test_the_agents_own_command_word_is_not_the_request_the_guard_reads(extract):
    """fix-2juj, through ``ParameterExtraction._extract_impl``: 'member' is spelled only inside the command word
    ``find_member`` the agent's text begins with (the caller's ``find_member:`` prefix is stripped, the word after it is not), so it is not written in the request; the retry's 'Ann Lee' is."""
    output, diagnostics, lm = extract(
        TUTORIAL, "find_member", "find_member: find_member Ann Lee",
        {"reasoning": "the command word", "query": "member"},
        {"reasoning": "the name", "query": "Ann Lee"})

    # 'member' is also a word of the workflow's vocabulary: the word screen clears it before the guard reads it
    assert diagnostics["word_screen_values"] == {"query": "member"}
    assert _calls(lm) == 2
    assert output.parameters_are_valid is True
    assert output.cmd_parameters.query == "Ann Lee"


def test_the_one_guard_does_not_read_the_agents_command_word_as_the_request():
    """The validation stage's guard of a marked field reads the request as the check stage does: without the command
    word an agent's text starts with (``fill.without_own_command_word``), so 'find', spelled only inside
    ``find_member``, is not written in it (whole words: "find" is bounded by the "_" of the command word)."""
    from fastworkflow.planning import planning_commands as pc
    from fastworkflow.parameters.signatures import InputForParamExtraction

    class Query(BaseModel):
        query: str = Field(description="words of the member's name", json_schema_extra=COPIED)

    extractor = InputForParamExtraction(command="find_member Ann Lee", input_for_param_extraction_class=None)
    refusal = extractor._check_option_arguments(pc.option_command(Query, "find_member"),
                                                Query.model_construct(query="find"), None, {})
    assert refusal is not None and str(refusal) == "query='find' is not written in the request"
    assert extractor._check_option_arguments(pc.option_command(Query, "find_member"),
                                             Query.model_construct(query="Ann Lee"), None, {}) is None
