"""The one guard module, ``fastworkflow/parameters/check.py`` (bd fix-qoyz step 3).

1. Direct cases of each rule.
2. The equivalence table: ``tests/fixtures/parameter_check_equivalence.json`` holds inputs and the outputs the two old
   guard entry points gave BEFORE they were merged (``planning_commands.check_arguments`` / ``checked_arguments`` /
   ``argument_gaps`` and ``argument_checks.check_extracted_values``): values, respellings, refusal texts and their
   order, exception type, message and attributes, notes, and the ``Extraction`` of ``argument_gaps``. Both entry
   points must still give exactly those outputs. Steps 4 to 6 of the staged plan change behaviour on purpose, one
   decision at a time: a step that moves an output edits the case it moves here, in its own commit, with its measurement.
3. There is no guard switch: both kinds of command are checked by one set of rules (bd fix-qoyz step 6).

Real components only: the tutorial workflow's planning commands (and the table's own, written into a temporary
copy), the real guard modules. No model is called.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Literal
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.planning import planning_commands as pc
from fastworkflow.parameters import check
from tests import parameter_check_corpus as corpus

FIXTURE = Path(__file__).parent / "fixtures" / "parameter_check_equivalence.json"

@pytest.fixture(autouse=True)
def _framework(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture(scope="module")
def commands(tmp_path_factory) -> dict:
    return corpus.build_commands(tmp_path_factory.mktemp("check_table"))


@pytest.fixture(scope="module")
def table() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. the rules, under both policies

def test_the_module_imports_no_model_stack():
    code = ("import sys; import fastworkflow.parameters.check; "
            "heavy = [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'dspy')]; "
            "print(','.join(sorted(heavy)))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == ""


@pytest.mark.parametrize("extra, expected", [
    ({"copied_from_command_text": True}, True),
    ({"copied_from_command_text": {"ignore": ["the"]}}, True),
    ({"record_named_in_command_text": True}, True),
    ({"classifier_confirms": True}, False),       # another option: not the strict check
    ({}, False),                                  # no option: the extractor's value stands
])
def test_is_marked(extra, expected):
    class Model(BaseModel):
        value: str = Field(default="", json_schema_extra=extra)

    assert check.is_marked(Model.model_fields["value"]) is expected


def test_a_pattern_does_not_mark_a_field():
    class Model(BaseModel):
        value: str = Field(default="", pattern=r"^#W\d+$")

    assert check.is_marked(Model.model_fields["value"]) is False


def test_is_given_value_reads_only_what_the_extractor_gave():
    assert check.is_given_value("Dune", "") is True
    assert check.is_given_value("", "") is False                    # the default
    assert check.is_given_value(0, 0) is False
    assert check.is_given_value("NOT_FOUND", "") is False
    assert check.is_given_value("<UNKNOWN>", "") is False


@pytest.mark.parametrize("number, numbers, expected", [
    (3, {3, 4}, ("ok", None)),                                  # written
    (9, set(), ("refuse", "is not a number the request writes")),   # a marked int: refused when no number is written too
    (9, {3}, ("refuse", check.numbers_why({3}))),               # never replaced by the one number written
    (9, {3, 4}, ("refuse", check.numbers_why({3, 4}))),
])
def test_number_rule(number, numbers, expected):
    assert check.number_rule(number, numbers) == expected


def test_there_is_no_example_rule():
    """The examples are not shown to the model any more, so a value equal to one is not refused (bd fix-qjne.21)."""
    assert not hasattr(check, "example_rule") and not hasattr(check, "EXAMPLE_WHY")


def test_allowed_placeholder_and_ground_rules():
    from fastworkflow.parameters import rules

    # the one template-token verdict lives in ``rules`` (read through ``signatures.classify_bracketed_value``)
    assert not hasattr(check, "placeholder_rule")
    assert rules.placeholder_like("{name}") and not rules.placeholder_like("Dune")
    request = "Find Ann‑Lee now"
    folded = check.fold(request).lower()
    assert check.ground_rule("ann-lee", folded, request) == "Ann‑Lee"
    assert check.ground_rule("Bob", folded, request) is None
    assert check.ground_position("ann-lee", folded) == 5 and check.ground_position("", folded) == -1


def test_state_of_why_reads_a_placeholder_pronoun_or_kind_word_as_missing():
    from fastworkflow.parameters.workflow_vocabulary import KIND_WORD_WHY

    for why in (check.PLACEHOLDER_WHY, pc.PRONOUN_WHY, KIND_WORD_WHY):
        assert check.state_of_why(why) == check.MISSING
    assert check.state_of_why(check.NOT_WRITTEN_WHY) == check.INVALID


class _Row(BaseModel):
    # name and count are marked: the strict check; note is not (its value stands)
    name: str = Field(description="a name", json_schema_extra={"copied_from_command_text": True})
    count: int = Field(default=0, description="how many", json_schema_extra={"copied_from_command_text": True})
    note: str = Field(default="", description="a note")


def test_the_one_guard_of_marked_execution_fields_refuses_an_unwritten_value_and_an_unwritten_number():
    """The strict check of a marked execution field is ``check_plan``'s, through ``option_command`` (bd fix-ftdl.6.3):
    ``check_execution`` (the check stage) no longer makes it."""
    values = {"name": "Bob", "count": 8, "note": ""}
    request = "find Ann Lee, 7 rows"
    assert check.check_execution(_Row, values, request).gaps == []
    gaps = pc.argument_gaps(pc.option_command(_Row, "app/find"), values, request)
    assert gaps.invalid == (("name", check.NOT_WRITTEN_WHY), ("count", check.numbers_why({7})))
    assert check.check_execution(_Row, values, "  ").gaps == []


def test_the_one_guard_of_a_marked_execution_field_carries_the_value_and_message():
    checked = check.check_plan(pc.option_command(_Row, "app/find"), {"name": "Bob", "count": 0, "note": ""}, "find Ann")
    assert checked.stop == check.Gap("name", check.INVALID, "not_written", check.NOT_WRITTEN_WHY, "Bob",
                                     "name='Bob' is not written in the request")


def test_check_plan_grounds_a_marked_value(commands):
    command = commands["book_details"]               # title carries copied_from_command_text: the strict check
    checked = check.check_plan(command, {"title": "Dune Messiah"}, "who has Foundation on loan", None, None)
    assert checked.stop.message == f"title='Dune Messiah' {check.NOT_WRITTEN_WHY}"
    assert checked.stop.state == check.INVALID and checked.stop.call is False


def test_check_plan_stop_kinds(commands):
    command = commands["author_books"]
    missing = check.check_plan(command, {"author": None}, "show every book", None, None).stop
    assert (missing.state, missing.code, missing.message) == (
        check.MISSING, "missing", "missing required argument author")
    placeholder = check.check_plan(command, {"author": "<entity_name>"}, "show every book", None, None).stop
    assert placeholder.state == check.MISSING and placeholder.why == check.PLACEHOLDER_WHY
    verdict = check.check_plan(commands["shape_verify_required"], {"subject": "Thing"}, "scan the thing",
                               {"subject": (pc.NO_SUBJECT, 0.95)}, None).stop
    assert verdict.call is True and verdict.message == "subject='Thing': the check finds no subject in the request"


def test_checked_extraction_reads_the_gaps(commands):
    checked = check.check_plan(commands["transfer_loan"], {"member": "Ann Lee", "branch": None},
                               "Move Ann Lee's loan", None, None)
    extraction = checked.extraction
    assert extraction.missing == ("branch",) and extraction.invalid == ()


def test_a_planning_literal_argument_and_an_execution_literal_field_follow_one_rule(commands):
    """``allowed_member`` is the one rule: a member takes its declared spelling (case and whitespace ignored), a value
    outside refuses THAT argument only, with the allowed values in the reason."""
    from pydantic import BaseModel

    plan = commands["transfer_loan"]
    ok = check.check_plan(plan, {"member": "Ann Lee", "branch": "Main", "entity_type": " ACCOUNT "},
                          "Move Ann Lee's loan to Main", None, None)
    assert ok.values["entity_type"] == "account" and ok.stop is None
    refused = check.check_plan(plan, {"member": "Ann Lee", "branch": "Main", "entity_type": "savings"},
                               "Move Ann Lee's loan to Main", None, None).stop
    assert (refused.name, refused.state, refused.why) == ("entity_type", check.INVALID, "is not one of identity, account")

    class Input(BaseModel):
        member: str
        entity_type: Literal["identity", "account"] = "identity"

    executed = check.check_execution(Input, {"member": "Ann Lee", "entity_type": " ACCOUNT "}, "Move Ann Lee's loan")
    assert executed.respelled == {"entity_type": (" ACCOUNT ", "account")} and executed.gaps == []
    outside = check.check_execution(Input, {"member": "Ann Lee", "entity_type": "savings"}, "Move Ann Lee's loan")
    assert [(g.name, g.why) for g in outside.gaps] == [("entity_type", refused.why)]


# ---------------------------------------------------------------- 2. the equivalence table

def test_the_table_is_broad(table):
    cases = table["cases"]
    assert len(cases) >= 200
    assert {c["kind"] for c in cases} == {"plan", "exec", "exec_input", "option"}
    assert set(table["expected"]) == {c["id"] for c in cases}
    assert not [k for k, v in table["expected"].items() if "CRASH" in v]


def _outcome(case: dict, commands: dict) -> dict:
    return json.loads(json.dumps(corpus.run_case(case, commands), sort_keys=True))


def test_both_entry_points_still_give_exactly_the_captured_outputs(table, commands):
    """The permanent guard for steps 4 to 6: no output of ``check_arguments``, ``checked_arguments``,
    ``argument_gaps`` or ``check_extracted_values`` moves unless a case of this table is edited with it."""
    moved = [case["id"] for case in table["cases"] if _outcome(case, commands) != table["expected"][case["id"]]]
    assert moved == []


def test_the_captured_outputs_hold_every_kind_of_outcome(table):
    expected = table["expected"]
    plan = [v for v in expected.values() if "check_arguments" in v]
    assert sum("ok" in v["check_arguments"] for v in plan) >= 50
    assert sum("error" in v["check_arguments"] for v in plan) >= 50
    assert any(v["check_arguments"].get("notes") for v in plan)
    assert any(len(v["argument_gaps"]["missing"]) + len(v["argument_gaps"]["invalid"]) > 1 for v in plan)
    executed = [v for v in expected.values() if "refused" in v]
    # the check stage of an execution command screens; the strict check of its marked fields is the one guard's
    # (``check_plan``, the plan cases of this table: bd fix-ftdl.6.3), so it refuses fewer values than it did
    # (and respells only a Literal's member; a marked value takes the request's spelling from the one guard, a note)
    assert sum(bool(v["refused"]) for v in executed) >= 15 and sum(bool(v["respelled"]) for v in executed) >= 3
    assert sum(any("->" in n for n in v["check_arguments"].get("notes", [])) for v in plan) >= 10
    types = {v["check_arguments"]["error"]["type"] for v in plan if "error" in v["check_arguments"]}
    assert types == {"PlanningCommandInputError"}   # the call-level error is covered directly, see test_check_plan_stop_kinds


# ---------------------------------------------------------------- 3. the one rule, no switch

def test_no_guard_switch_is_left():
    """Step 6 of bd fix-qoyz decided every switch between the two kinds of command; no ``Policy`` is left and no
    guard reads one."""
    source = Path(check.__file__).read_text(encoding="utf-8")
    assert "parameters.spec" not in source and "policy" not in source.lower()
