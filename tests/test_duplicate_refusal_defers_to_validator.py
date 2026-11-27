"""Validators first on a tie (bd fix-qjne.38).

The duplicate rule (``parameters.check._duplicate_rule``) refuses a REQUIRED marked ``str`` field whose value another
marked field also holds when the request writes it fewer times. The guards run before the ``Input``'s validators, so a
command's own validator that refuses the same values with a clearer message used to be unreachable. The one ordering
rule (``parameters.check.validators_first``): when the duplicate rule refuses and the validators, run on the same
values, refuse too, the validators' refusal is the one reported, handled as that path handles a validator's refusal;
when they pass, the duplicate rule's refusal stands. Both kinds of command: a planning command's typed call
(``planning_commands.checked_arguments``) and an execution command's validation stage
(``InputForParamExtraction.validate_parameters``).

Real loader, guards and validators; no model is called.
"""
from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel, Field, model_validator

import fastworkflow
from fastworkflow import planning_commands as pc
from fastworkflow.parameters import check
from tests.test_argument_checks import _command

COPIED = {"copied_from_command_text": True}
SAME_MESSAGE = "record_a and record_b are both {value!r}; record_b is the other record the request names"
REQUEST = "compare Payroll with itself"


def _refuse_same(self, field):
    if self.record_a and self.record_b and str(self.record_a).lower() == str(self.record_b).lower():
        raise fastworkflow.Refused(SAME_MESSAGE.format(value=self.record_a), field=field)
    return self


# ---------------------------------------------------------------- execution commands

class Compare(BaseModel):
    record_a: str = Field(description="the first record", json_schema_extra=COPIED)
    record_b: str = Field(description="the second record", json_schema_extra=COPIED)

    @model_validator(mode="after")
    def two_records(self):
        return _refuse_same(self, "record_b")


class CompareNoField(BaseModel):
    record_a: str = Field(description="the first record", json_schema_extra=COPIED)
    record_b: str = Field(description="the second record", json_schema_extra=COPIED)

    @model_validator(mode="after")
    def two_records(self):
        return _refuse_same(self, None)


class CompareOtherRule(BaseModel):
    """A validator that refuses something else only: it passes the duplicated values."""
    record_a: str = Field(description="the first record", json_schema_extra=COPIED)
    record_b: str = Field(description="the second record", json_schema_extra=COPIED)

    @model_validator(mode="after")
    def no_archive(self):
        if self.record_a == "Archive":
            raise fastworkflow.Refused("the archive cannot be compared", field="record_a")
        return self


class Place(BaseModel):
    city: str = Field(description="the city", json_schema_extra=COPIED)
    state: Optional[str] = Field(default=None, description="the state", json_schema_extra=COPIED)

    @model_validator(mode="after")
    def not_the_same(self):
        if self.state and self.city and self.state.lower() == self.city.lower():
            raise fastworkflow.Refused(f"city and state are both {self.city!r}", field="state")
        return self


def _slash_validate(model: type, values: dict, request: str):
    from fastworkflow.parameters.signatures import InputForParamExtraction

    extractor = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
    diagnostics: dict = {}
    valid, message, _, fields = extractor.validate_parameters(None, "app/compare", model.model_construct(**values),
                                                              diagnostics)
    return valid, message, fields, diagnostics


def test_execution_both_refuse_reports_the_validators_message_and_asks_for_its_field():
    valid, message, fields, diagnostics = _slash_validate(
        Compare, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert valid is False
    assert SAME_MESSAGE.format(value="Payroll") in message
    assert "is also the value of" not in message
    assert fields == ["record_b"]
    assert "argument_refusal" not in diagnostics
    assert diagnostics["validation_hook"]["refused_fields"] == ["record_b"]


def test_execution_validator_refusal_naming_no_field_answers_the_call():
    valid, message, fields, _ = _slash_validate(
        CompareNoField, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert (valid, message, fields) == (False, SAME_MESSAGE.format(value="Payroll"), [])


def test_execution_validators_pass_so_the_duplicate_rules_refusal_stands():
    valid, message, fields, diagnostics = _slash_validate(
        CompareOtherRule, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert valid is False
    assert "is also the value of" in diagnostics["argument_refusal"]
    assert fields and set(fields) <= {"record_a", "record_b"}
    assert "the archive" not in message


def test_execution_optional_field_left_empty_is_unchanged_and_runs_no_validator_early():
    """The validator refuses city == state: had it run on the duplicated values it would refuse. The duplicate rule
    leaves the optional state not given with its note, and the validators then see it empty and pass."""
    valid, message, fields, diagnostics = _slash_validate(Place, {"city": "Albany", "state": "Albany"},
                                                          "stores in Albany")
    assert valid is True, message
    assert diagnostics["argument_left_empty"] == ["state"]


def test_execution_a_value_written_twice_is_no_refusal():
    valid, message, _, _ = _slash_validate(Place, {"city": "New York", "state": "New York"},
                                           "stores in New York, New York")
    # the duplicate rule lets both fields hold it; the command's own validator is what refuses it now, as before
    assert valid is False and message and "is also the value of" not in message


def test_execution_a_value_written_twice_passes_with_a_validator_that_allows_it():
    class PlainPlace(BaseModel):
        city: str = Field(description="the city", json_schema_extra=COPIED)
        state: Optional[str] = Field(default=None, description="the state", json_schema_extra=COPIED)

    valid, _, _, diagnostics = _slash_validate(PlainPlace, {"city": "New York", "state": "New York"},
                                               "stores in New York, New York")
    assert valid is True and "argument_refusal" not in diagnostics


# ---------------------------------------------------------------- planning commands

def _plan(tmp_path, validator: str, state_optional: bool = False):
    """A planning command with two marked fields and the model validator *validator* (a body that reads ``self``)."""
    second = ('record_b: Optional[str] = Field(default=None, description="the second record", '
              'json_schema_extra={"copied_from_command_text": True})' if state_optional else
              'record_b: str = Field(description="the second record", '
              'json_schema_extra={"copied_from_command_text": True})')
    module = ('from typing import Optional\n\nimport fastworkflow\nfrom pydantic import BaseModel, Field, '
              'model_validator\n\n\nclass Signature:\n    """Compare two records."""\n\n'
              '    class Input(BaseModel):\n'
              '        record_a: str = Field(description="the first record", '
              'json_schema_extra={"copied_from_command_text": True})\n'
              f'        {second}\n\n'
              '        @model_validator(mode="after")\n'
              '        def check(self):\n'
              f'            {validator}\n'
              '            return self\n\n\n'
              'class PlanGenerator:\n    def steps(self, input):\n        return ["say which record is larger"]\n')
    return _command(tmp_path, module, name="compare")


SAME = ('if self.record_b and self.record_a.lower() == self.record_b.lower(): '
        'raise fastworkflow.Refused(f"record_a and record_b are both {self.record_a!r}; record_b is the other record '
        'the request names"{field})')


def test_planning_both_refuse_reports_the_validators_refusal_with_its_field(tmp_path):
    command = _plan(tmp_path, SAME.replace('{field}', ', field="record_b"'))
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.checked_arguments(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert str(refused.value) == SAME_MESSAGE.format(value="Payroll")
    assert refused.value.fields == ("record_b",)
    assert refused.value.gap is None
    assert pc.refusal_names_an_input(command, refused.value)


def test_planning_validator_refusal_naming_no_field_is_a_refusal_of_the_call(tmp_path):
    command = _plan(tmp_path, SAME.replace('{field}', ""))
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.checked_arguments(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert str(refused.value) == SAME_MESSAGE.format(value="Payroll")
    assert not pc.refusal_names_an_input(command, refused.value)


def test_planning_validators_pass_so_the_duplicate_rules_refusal_stands(tmp_path):
    command = _plan(tmp_path, 'if self.record_a == "Archive": raise fastworkflow.Refused("no archive")')
    with pytest.raises(pc.PlanningCommandInputError, match="is also the value of") as refused:
        pc.checked_arguments(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert check.validators_first(refused.value.gap)


def test_planning_optional_field_left_empty_is_unchanged_and_runs_no_validator_early(tmp_path):
    command = _plan(tmp_path, SAME.replace('{field}', ', field="record_b"'), state_optional=True)
    args, notes = pc.checked_arguments(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert args["record_a"] == "Payroll" and not args["record_b"]
    assert any("left empty" in note for note in notes)


def test_planning_a_value_written_twice_is_no_refusal(tmp_path):
    command = _plan(tmp_path, 'if self.record_a == "Archive": raise fastworkflow.Refused("no archive")')
    args, _ = pc.checked_arguments(command, {"record_a": "New York", "record_b": "New York"},
                                   "compare New York with New York")
    assert (args["record_a"], args["record_b"]) == ("New York", "New York")


def test_the_guards_alone_still_refuse_with_the_duplicate_rule(tmp_path):
    """What is refused does not change: the guards (no validators) still stop at the duplicate rule."""
    command = _plan(tmp_path, SAME.replace('{field}', ', field="record_b"'))
    with pytest.raises(pc.PlanningCommandInputError, match="is also the value of"):
        pc.check_arguments(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)


# ---------------------------------------------------------------- the question and the review line (argument_gaps)
#
# The same rule where the user reads the refusal before anything runs: the question built from a planning command's
# argument gaps, and a review item's check line (``review_proposals.rechecked``). Only on a tie: the duplicate rule's
# refusal is the only gap and every required value is present. With another gap the validators are not run.

#: The calls of the recording validator below (a list the generated command file appends to).
VALIDATOR_CALLS: list = []

RECORD_CALL = "import tests.test_duplicate_refusal_defers_to_validator as t; t.VALIDATOR_CALLS.append(dict(self))"


def _module(validator: str, extra: tuple[str, ...] = ()) -> str:
    """A planning command file: two required marked records, *extra* field lines, the model validator *validator*
    (lines of a body that reads ``self``; the body ends with ``return self``)."""
    fields = "".join(f"        {line}\n" for line in extra)
    body = "".join(f"            {line}\n" for line in validator.splitlines())
    return ('from typing import Optional\n\nimport fastworkflow\nfrom pydantic import BaseModel, Field, '
            'model_validator\n\n\nclass Signature:\n    """Compare two records."""\n\n'
            '    class Input(BaseModel):\n'
            '        record_a: str = Field(description="the first record", '
            'json_schema_extra={"copied_from_command_text": True})\n'
            '        record_b: str = Field(description="the second record", '
            'json_schema_extra={"copied_from_command_text": True})\n'
            f'{fields}\n'
            '        @model_validator(mode="after")\n'
            '        def check(self):\n'
            f'{body}'
            '            return self\n\n\n'
            'class PlanGenerator:\n    def steps(self, input):\n        return ["say which record is larger"]\n')


SAME_WITH_FIELD = SAME.replace('{field}', ', field="record_b"')
NOTE_FIELD = 'note: str = Field(default="", description="a note to add")'
SCOPE_FIELD = 'scope: str = Field(description="where to compare")'


def test_the_question_on_a_tie_carries_the_validators_message_and_field(tmp_path):
    from fastworkflow.parameters import parameter_clarification as loops

    command = _command(tmp_path, _module(SAME_WITH_FIELD), name="compare")
    gaps = pc.argument_gaps(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    message = SAME_MESSAGE.format(value="Payroll")
    assert gaps.invalid == (("record_b", message),) and not gaps.missing
    assert gaps.values == {"record_a": "Payroll"}
    question, request = loops.planning_question(command, gaps.values, gaps)
    assert message in question and "is also the value of" not in question
    assert "Which record b should I use?" in question
    assert request["invalid"] == ["record_b"] and request["values"] == {"record_a": "Payroll"}


def test_the_question_keeps_the_framework_wording_when_the_validators_pass(tmp_path):
    from fastworkflow.parameters import parameter_clarification as loops

    command = _command(tmp_path, _module('if self.record_a == "Archive": raise fastworkflow.Refused("no archive")'),
                       name="compare")
    gaps = pc.argument_gaps(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert [why for _, why in gaps.invalid] == ["is also the value of record_b"] and not gaps.reason
    question, _ = loops.planning_question(command, gaps.values, gaps)
    assert "in your request, not the" in question


def test_with_another_gap_the_validators_are_not_run_and_the_framework_wording_stands(tmp_path):
    from fastworkflow.parameters import parameter_clarification as loops

    command = _command(tmp_path, _module(f"{RECORD_CALL}\n{SAME_WITH_FIELD}", extra=(SCOPE_FIELD,)), name="compare")
    VALIDATOR_CALLS.clear()
    gaps = pc.argument_gaps(command, {"record_a": "Payroll", "record_b": "Payroll", "scope": None}, REQUEST)
    assert VALIDATOR_CALLS == []
    assert gaps.missing == ("scope",) and [why for _, why in gaps.invalid] == ["is also the value of record_b"]
    assert not gaps.reason
    question, _ = loops.planning_question(command, gaps.values, gaps)
    assert SAME_MESSAGE.format(value="Payroll") not in question
    # the same values with the other gap filled are a tie: the recording validator runs, and its refusal is the gap
    gaps = pc.argument_gaps(command, {"record_a": "Payroll", "record_b": "Payroll", "scope": "the west"},
                            f"{REQUEST} in the west")
    assert len(VALIDATOR_CALLS) == 1 and gaps.reason == SAME_MESSAGE.format(value="Payroll")


def test_a_validators_rewrite_is_not_kept_in_the_gaps_values(tmp_path):
    rewrite = f'if self.note == "as asked": self.note = "rewritten"\n{SAME_WITH_FIELD}'
    command = _command(tmp_path, _module(rewrite, extra=(NOTE_FIELD,)), name="compare")
    gaps = pc.argument_gaps(command, {"record_a": "Payroll", "record_b": "Payroll", "note": "as asked"}, REQUEST)
    assert gaps.reason == SAME_MESSAGE.format(value="Payroll")
    assert gaps.values == {"record_a": "Payroll", "note": "as asked"}
    # and a rewrite of validators that pass is not kept either: the duplicate rule's gap stands with the given values
    command = _command(tmp_path / "pass", _module('if self.note == "as asked": self.note = "rewritten"',
                                                  extra=(NOTE_FIELD,)), name="compare")
    gaps = pc.argument_gaps(command, {"record_a": "Payroll", "record_b": "Payroll", "note": "as asked"}, REQUEST)
    assert not gaps.reason and gaps.values == {"record_b": "Payroll", "note": "as asked"}


def test_a_validator_refusal_naming_no_field_leaves_nothing_to_ask_as_a_call_refusal_does(tmp_path):
    command = _command(tmp_path, _module(SAME.replace('{field}', "")), name="compare")
    gaps = pc.argument_gaps(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert not gaps.names and gaps.values == {} and not gaps.reason
    # the run reports it, as a refusal that names no input (``checked_arguments``)
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.checked_arguments(command, {"record_a": "Payroll", "record_b": "Payroll"}, REQUEST)
    assert not pc.refusal_names_an_input(command, refused.value)


@pytest.fixture
def compare_workflow(tmp_path, monkeypatch):
    """A maker of the planning tutorial workflow with a ``compare`` command whose validator is the one given."""
    import shutil
    import uuid
    from pathlib import Path

    def make(validator: str):
        folder = tmp_path / f"compare_wf_{uuid.uuid4().hex[:6]}"
        shutil.copytree(Path(__file__).resolve().parent / "planning_tutorial_workflow", folder,
                        ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
        (folder / "_commands" / "compare.py").write_text(_module(validator), encoding="utf-8")
        fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "state")})
        fastworkflow.RoutingRegistry.clear_registry()
        fastworkflow.RoutingRegistry.get_definition(str(folder), load_cached=False)
        return fastworkflow.Workflow.create(str(folder), workflow_id_str=f"compare_{uuid.uuid4().hex[:8]}")

    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")
    yield make
    fastworkflow.RoutingRegistry.clear_registry()


def _compare_step(goal: str = REQUEST):
    from fastworkflow.review_proposals import Means, SubTask, SubTaskInput

    return SubTask(goal=goal, means=Means(kind="planning", name="compare"),
                   inputs=[SubTaskInput(name="record_a", value="Payroll"),
                           SubTaskInput(name="record_b", value="Payroll")])


def test_the_review_check_line_on_a_tie_shows_the_validators_message(compare_workflow):
    from fastworkflow import review_proposals as rp

    item = rp.rechecked(compare_workflow(SAME_WITH_FIELD), _compare_step())
    assert item.check == SAME_MESSAGE.format(value="Payroll")
    assert item.check_refused == ["record_b"] and not item.refused_without_field


def test_the_review_check_line_naming_no_field_shows_the_validators_message(compare_workflow):
    from fastworkflow import review_proposals as rp

    item = rp.rechecked(compare_workflow(SAME.replace('{field}', "")), _compare_step())
    assert item.check == item.refused_without_field == SAME_MESSAGE.format(value="Payroll")


def test_the_review_check_line_keeps_the_duplicate_rule_when_the_validators_pass(compare_workflow):
    from fastworkflow import review_proposals as rp

    item = rp.rechecked(compare_workflow('if self.record_a == "Archive": raise fastworkflow.Refused("no archive")'),
                        _compare_step())
    assert "is also the value of record_b" in item.check
    assert item.check_refused == ["record_a"]
