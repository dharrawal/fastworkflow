"""The one validation driver (``fastworkflow.parameters.validate``, bd fix-qoyz step 2), and the two old
drivers that word its outcome: ``planning_commands.check_input`` (a typed planning call) and
``InputForParamExtraction.validate_parameters`` (an execution command's turn).

The ``BEFORE_*`` literals were captured by running the drivers BEFORE the one driver existed, on the same
``Input`` classes and values; the drivers must still produce them. Real components, no mocks.
"""
import types

import pytest

from fastworkflow.planning import planning_commands as pc
from fastworkflow.agent import workflow_agent
from fastworkflow.parameters.validate import (
    RAISED, REFUSED, Refusal, Validated, ValidatorReturnError, validate)
from fastworkflow.parameters.signatures import InputForParamExtraction
from tests.test_input_model_validators import PLANNING
from tests.test_planning_command_files import _write

HEAD = "from pydantic import BaseModel, Field, model_validator"
BODIES = {
    "field_rule": '''
        @field_validator("name")
        @classmethod
        def no_digits(cls, v):
            if any(c.isdigit() for c in v):
                raise ValueError("no digits in a name")
            return v
''',
    "field_refused_bare": '''
        @field_validator("name")
        @classmethod
        def no_digits(cls, v):
            if any(c.isdigit() for c in v):
                raise fastworkflow.Refused("no digits in a name")
            return v
''',
    "refused_fields": '''
        @model_validator(mode="after")
        def closed(self, info):
            if self.name == "attic":
                raise fastworkflow.Refused("the attic is closed", field="name")
            return self
''',
    "refused_nofield": '''
        @model_validator(mode="after")
        def nofit(self, info):
            raise fastworkflow.Refused("this command does not fit the request")
''',
    "plain": '''
        @model_validator(mode="after")
        def plain(self, info):
            raise ValueError("name and note are one record")
''',
    "unexpected": '''
        @model_validator(mode="after")
        def boom(self, info):
            raise KeyError("nope")
''',
    "rewrite": '''
        @model_validator(mode="after")
        def upper(self, info):
            self.name = self.name.upper()
            return self
''',
    "rewrite_then_refuse": '''
        @model_validator(mode="after")
        def upper(self, info):
            self.name = self.name.upper()
            raise fastworkflow.Refused("stopped after rewriting", field="note")
''',
    "rewrite_then_plain": '''
        @model_validator(mode="after")
        def upper(self, info):
            self.name = self.name.upper()
            raise AssertionError("assertion after rewriting")
''',
    "wrong_return": '''
        @model_validator(mode="after")
        def wrong(self, info):
            return None
''',
    "clean": '''
        @model_validator(mode="after")
        def fine(self, info):
            return self
''',
    "reads_context": '''
        @model_validator(mode="after")
        def look(self, info):
            self.note = info.context["command_text"] + "|" + info.context["command"]
            return self
''',
}
VALUES = {"field_rule": "x1", "field_refused_bare": "x1", "refused_fields": "attic"}


def _command(tmp_path, case):
    text = PLANNING.format(name=case, body=BODIES[case]).replace(
        HEAD, "from pydantic import BaseModel, Field, field_validator, model_validator")
    return workflow_agent.load_planning_command_map(_write(tmp_path, {case: text}))[case]


def _args(case):
    return {"name": VALUES.get(case, "ann"), "note": ""}


def _plan_driver(command, case):
    """What the typed planning call's driver raises or returns: (class name, text, fields, refusal, written, raised, detail)."""
    try:
        return ["ok", pc.check_input(command, _args(case), "show it", "WF")]
    except Exception as e:  # noqa: BLE001 - the test reads whatever the driver raises
        return [type(e).__name__, str(e), list(e.fields) if hasattr(e, "fields") else None,
                getattr(e, "refusal", "ABSENT"), getattr(e, "written", "ABSENT"),
                getattr(e, "raised", "ABSENT"), getattr(e, "detail", "ABSENT")]


def _exec_driver(command, case):
    """What the execution command's driver returns, its diagnostics and the parameters it leaves."""
    diagnostics: dict = {}
    params = command.input_model.model_construct(**_args(case))
    # the workflow is only read for its context when the refusal's question is worded
    result = InputForParamExtraction(command="show it", input_for_param_extraction_class=None).validate_parameters(
        types.SimpleNamespace(context={}), "Library/" + case, params, diagnostics=diagnostics)
    return [list(result), diagnostics, {n: getattr(params, n) for n in params.__class__.model_fields}]


# ------------------------------------------------------------------ the one driver, directly

def _run(command, case, text="show it", workflow="WF"):
    return validate(command.input_model, pc._absent_as_none(command, _args(case)),
                    command_text=text, workflow=workflow, command_name=command.name)


def test_a_clean_pass_returns_the_values_and_no_write_or_refusal(tmp_path):
    out = _run(_command(tmp_path, "clean"), "clean")
    assert out == Validated(values={"name": "ann", "note": ""}, written={}, refusal=None)


def test_a_field_validator_refusal_is_a_refusal_of_that_field_with_its_own_message(tmp_path):
    # pydantic locates the failing field validator's error on its field: the refusal names it (bd fix-ftdl.7.10)
    out = _run(_command(tmp_path, "field_rule"), "field_rule")
    assert out.values == {} and out.written == {}
    assert out.refusal == Refusal("no digits in a name", ("name",), None, REFUSED)


def test_a_bare_refused_in_a_field_validator_names_that_field_as_a_value_error_there_does(tmp_path):
    # the field comes from pydantic's error location, so the command asks again for it (fix-ftdl.8.20)
    out = _run(_command(tmp_path, "field_refused_bare"), "field_refused_bare")
    assert out.values == {} and out.written == {}
    assert out.refusal == Refusal("no digits in a name", ("name",), None, REFUSED)


def test_a_model_validator_refused_with_fields_keeps_its_fields(tmp_path):
    out = _run(_command(tmp_path, "refused_fields"), "refused_fields")
    assert out.refusal == Refusal("the attic is closed", ("name",), None, REFUSED)


def test_a_refused_with_no_field_names_none(tmp_path):
    out = _run(_command(tmp_path, "refused_nofield"), "refused_nofield")
    assert out.refusal == Refusal("this command does not fit the request", (), None, REFUSED)


def test_a_plain_value_error_is_a_bare_plain_refusal(tmp_path):
    out = _run(_command(tmp_path, "plain"), "plain")
    assert out.refusal == Refusal("name and note are one record", (), None, REFUSED)


def test_an_assertion_error_is_a_plain_refusal_and_drops_what_the_validator_wrote(tmp_path):
    out = _run(_command(tmp_path, "rewrite_then_plain"), "rewrite_then_plain")
    assert out.refusal == Refusal("assertion after rewriting", (), None, REFUSED)
    assert out.written == {}


def test_an_unexpected_exception_is_a_raised_refusal_naming_its_type(tmp_path):
    out = _run(_command(tmp_path, "unexpected"), "unexpected")
    assert out.refusal == Refusal("'nope'", (), "KeyError", RAISED)


def test_a_validator_that_rewrites_a_field_reports_it_as_written(tmp_path):
    out = _run(_command(tmp_path, "rewrite"), "rewrite")
    assert out.refusal is None
    assert out.values == {"name": "ANN", "note": ""} and out.written == {"name": "ANN"}


def test_a_rewrite_before_a_refusal_is_dropped(tmp_path):
    out = _run(_command(tmp_path, "rewrite_then_refuse"), "rewrite_then_refuse")
    assert out.refusal == Refusal("stopped after rewriting", ("note",), None, REFUSED)
    assert out.written == {} and out.values == {}


def test_the_validation_context_is_the_text_the_workflow_and_the_command_name(tmp_path):
    out = _run(_command(tmp_path, "reads_context"), "reads_context", text="a request", workflow="WF")
    assert out.written == {"note": "a request|reads_context"}


def test_a_validator_that_does_not_return_the_input_is_a_programming_error(tmp_path):
    with pytest.raises(ValidatorReturnError) as raised:
        _run(_command(tmp_path, "wrong_return"), "wrong_return")
    assert str(raised.value) == ("wrong_return: a validator on Input must return self (the corrected Input) "
                                 "or raise; it returned NoneType")


# ------------------------------------------------------------------ both drivers, unchanged (captured before the change)

PLAN_BEFORE = {
    # changed on purpose (bd fix-ftdl.7.10): a failing field validator refuses its field, as Refused(field=) does
    "field_rule": ["PlanningCommandInputError", "no digits in a name",
                   ["name"], "no digits in a name", "ABSENT", "ABSENT", "ABSENT"],
    "refused_fields": ["PlanningCommandInputError", "the attic is closed",
                       ["name"], "the attic is closed", "ABSENT", "ABSENT", "ABSENT"],
    "refused_nofield": ["PlanningCommandInputError",
                        "this command does not fit the request",
                        [], "this command does not fit the request", "ABSENT", "ABSENT", "ABSENT"],
    "plain": ["PlanningCommandInputError", "name and note are one record",
              [], "name and note are one record", "ABSENT", "ABSENT", "ABSENT"],
    "unexpected": ["PlanningCommandCallError", "a validator raised KeyError: 'nope'",
                   None, "ABSENT", "ABSENT", "KeyError", "'nope'"],
    "rewrite": ["ok", {"name": "ANN"}],
    "rewrite_then_refuse": ["PlanningCommandInputError", "stopped after rewriting",
                            ["note"], "stopped after rewriting", "ABSENT", "ABSENT", "ABSENT"],
    "rewrite_then_plain": ["PlanningCommandInputError",
                           "assertion after rewriting",
                           [], "assertion after rewriting",
                           "ABSENT", "ABSENT", "ABSENT"],
    "wrong_return": ["PlanningCommandCallError",
                     "wrong_return: a validator on Input must return self (the corrected Input) or raise; "
                     "it returned NoneType", None, "ABSENT", "ABSENT", "ABSENT", "ABSENT"],
    "clean": ["ok", {}],
    "reads_context": ["ok", {"note": "show it|reads_context"}],
}

_QUESTION = "I can't run {command} yet: {message}. Which {field} should I use?"


def _hook(is_valid, message, changed=(), refused=(), raised=None, **more):
    hook = {"changed_fields": list(changed), "is_valid": is_valid, "message": message, "ran": True,
            "refused_fields": list(refused)}
    if raised:
        hook["raised"] = raised
    return {"validation_hook": hook, **more}


OK = [True, "All required parameters are valid.", {}, []]
EXEC_BEFORE = {
    # changed on purpose (bd fix-ftdl.7.10): asked again for the field, as a Refused naming it is
    "field_rule": [[False, _QUESTION.format(command="field rule", field="name", message="no digits in a name"), {},
                    ["name"]],
                   _hook(False, "no digits in a name", refused=["name"], invalid_fields=["name"], missing_fields=[]),
                   {"name": "x1", "note": ""}],
    "refused_fields": [[False, _QUESTION.format(command="refused fields", field="name", message="the attic is closed"), {}, ["name"]],
                       _hook(False, "the attic is closed", refused=["name"], invalid_fields=["name"], missing_fields=[]),
                       {"name": "attic", "note": ""}],
    "refused_nofield": [[False, "this command does not fit the request", {}, []],
                        _hook(False, "this command does not fit the request"), {"name": "ann", "note": ""}],
    "plain": [[False, "name and note are one record", {}, []],
              _hook(False, "name and note are one record"),
              {"name": "ann", "note": ""}],
    "unexpected": [[False, "Exception in Library/unexpected's validators: 'nope'", {}, []],
                   _hook(False, "Exception in Library/unexpected's validators: 'nope'",
                         raised="KeyError"), {"name": "ann", "note": ""}],
    "rewrite": [OK, _hook(True, None, changed=["name"]), {"name": "ANN", "note": ""}],
    "rewrite_then_refuse": [[False, _QUESTION.format(command="rewrite then refuse", field="note", message="stopped after rewriting"), {}, ["note"]],
                            _hook(False, "stopped after rewriting", refused=["note"],
                                  invalid_fields=["note"], missing_fields=[]),
                            {"name": "ann", "note": ""}],
    "rewrite_then_plain": [[False, "assertion after rewriting", {}, []],
                           _hook(False, "assertion after rewriting"),
                           {"name": "ann", "note": ""}],
    "wrong_return": [[False, "Library/wrong_return: a validator on Input must return self (the corrected Input) "
                             "or raise; it returned NoneType", {}, []],
                     _hook(False, "Library/wrong_return: a validator on Input must return self (the corrected Input) "
                                  "or raise; it returned NoneType"), {"name": "ann", "note": ""}],
    "clean": [OK, _hook(True, None), {"name": "ann", "note": ""}],
    "reads_context": [OK, _hook(True, None, changed=["note"]),
                      {"name": "ann", "note": "show it|Library/reads_context"}],
}


@pytest.mark.parametrize("case", sorted(PLAN_BEFORE))
def test_the_typed_planning_call_driver_gives_what_it_gave_before(tmp_path, case):
    assert _plan_driver(_command(tmp_path, case), case) == PLAN_BEFORE[case]


@pytest.mark.parametrize("case", sorted(EXEC_BEFORE))
def test_the_execution_driver_gives_what_it_gave_before(tmp_path, case):
    assert _exec_driver(_command(tmp_path, case), case) == EXEC_BEFORE[case]


def test_the_two_drivers_drop_a_rewrite_before_a_refusal_and_keep_one_when_it_passes(tmp_path):
    """One rule for both kinds (step-6 decision F): a rewrite is kept when the validation passes and dropped when it
    refuses; neither the execution parameters nor the planning error carry it."""
    command = _command(tmp_path, "rewrite_then_refuse")
    assert _exec_driver(command, "rewrite_then_refuse")[2]["name"] == "ann"
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc._validate_arguments(command, _args("rewrite_then_refuse"), "show it", "WF", [])
    assert not hasattr(refused.value, "written")
    (tmp_path / "kept").mkdir()
    kept = _command(tmp_path / "kept", "rewrite")
    assert _exec_driver(kept, "rewrite")[2]["name"] == "ANN"
    assert pc._validate_arguments(kept, _args("rewrite"), "show it", "WF", [])["name"] == "ANN"
