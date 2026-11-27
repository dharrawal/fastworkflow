"""The one duplicate rule leaves an OPTIONAL marked field not given, with a visible note (bd fix-qjne.20 follow-up).

Two marked ``str`` fields that hold one value the user writes fewer times than the fields that hold it: the optional one
beyond that count is left not given (its default), and a note says so, on the check's notes (``argument_notes``), on the
extraction span (``argument_left_empty``, names only), on a question when one is asked and on the review item. It is not
a refusal: no second fill, no "Cannot run as shown" line. Two REQUIRED fields, or a value the request does not write,
are still refused. The count reads the user's own request for the turn when the turn binds it (a split part that repeats
the name does not let two fields hold it).

Integration only: a temporary copy of ``planning_tutorial_workflow`` with two commands added, the real loader, the real
guards (``check_arguments``, ``argument_gaps``) and ``validate_parameters``.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.planning import planning_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.parameters import ask
from fastworkflow.parameters import check as one_guard
from fastworkflow.parameters.signatures import InputForParamExtraction

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"

#: A planning command: a required name and an optional connected name, both marked.
CONNECTED = '''
from pydantic import BaseModel, Field


class Signature:
    """Show the records connected to one record."""

    class Input(BaseModel):
        name: str = Field(description="the record", json_schema_extra={"copied_from_command_text": True})
        connected_name: str = Field(default="", description="one connected record to narrow to",
                                    json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def steps(self, input):
        return [f"show the records connected to {input.name}"]
'''

#: An execution command with the same two marked fields and one required plain field.
CONNECT_NOTE = '''
from typing import Optional

from pydantic import BaseModel, Field


class Signature:
    """File a note on the link between two records."""

    class Input(BaseModel):
        name: str = Field(description="the record", json_schema_extra={"copied_from_command_text": True})
        connected_name: Optional[str] = Field(default=None, description="the connected record",
                                              json_schema_extra={"copied_from_command_text": True})
        topic: str = Field(description="what the note is about")

    class Output(BaseModel):
        pass


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return Signature.Output()
'''

REQUEST = "show the records connected to Payroll"
LINE = "connected_name left empty: 'Payroll' is written once in your request and is the name."


@pytest.fixture
def workflow(tmp_path):
    folder = tmp_path / "left_empty_wf"
    shutil.copytree(TUTORIAL, folder, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (folder / "_commands" / "connected.py").write_text(CONNECTED, encoding="utf-8")
    (folder / "_commands" / "connect_note.py").write_text(CONNECT_NOTE, encoding="utf-8")
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(str(folder), load_cached=False)
    app_workflow = fastworkflow.Workflow.create(str(folder), workflow_id_str=f"left_{uuid.uuid4().hex[:8]}")
    yield app_workflow, str(folder)
    fastworkflow.RoutingRegistry.clear_registry()


def _connected(folder: str) -> sc.PlanningCommandCommand:
    return workflow_agent.load_planning_command_map(folder)["connected"]


def test_an_optional_duplicate_of_a_value_written_once_is_left_empty_with_a_note(workflow):
    _, folder = workflow
    args, notes = sc.check_arguments(_connected(folder), {"name": "Payroll", "connected_name": "Payroll"}, REQUEST)

    assert args["name"] == "Payroll" and args["connected_name"] == ""
    assert LINE in notes and ask.left_empty_lines(notes) == {"connected_name": LINE}
    # written twice, both keep it and nothing is noted
    args, notes = sc.check_arguments(_connected(folder), {"name": "Payroll", "connected_name": "Payroll"},
                                     "show the records of Payroll connected to Payroll")
    assert args["connected_name"] == "Payroll" and not ask.left_empty_lines(notes)


def test_the_gaps_carry_the_note_and_a_question_says_it_first(workflow):
    _, folder = workflow
    command = _connected(folder)
    gaps = sc.argument_gaps(command, {"name": "Payroll", "connected_name": "Payroll"}, REQUEST)

    assert not gaps.names and gaps.left_empty == (LINE,)
    question = ask.word(gaps.__class__(missing=("name",), left_empty=gaps.left_empty), form=ask.FORM_GAPS,
                        command=command).text
    assert question.startswith(LINE + "\n") and "Cannot run" not in question


def test_the_count_reads_the_users_request_not_a_split_part_that_repeats_the_name(workflow):
    _, folder = workflow
    part = "show the records of Payroll connected to Payroll"   # the splitter wrote the name twice
    with one_guard.user_written("who owns Payroll, and what is connected to it"):
        args, notes = sc.check_arguments(_connected(folder), {"name": "Payroll", "connected_name": "Payroll"}, part)
    assert args["connected_name"] == "" and LINE in notes
    # a value the user's request does not write (an earlier step's output) is counted in the checked text alone
    with one_guard.user_written("what is connected to the first record"):
        args, _ = sc.check_arguments(_connected(folder), {"name": "Payroll", "connected_name": "Payroll"}, part)
    assert args["connected_name"] == "Payroll"


def _validate(app_workflow, command_name: str, request: str, diagnostics: dict, **values):
    definition = fastworkflow.RoutingRegistry.get_definition(app_workflow.folderpath)
    response_generator = definition.get_command_class(command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
    model = sys.modules[response_generator.__module__].Signature.Input
    extraction = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
    params = model.model_construct(**values)
    return extraction.validate_parameters(app_workflow, command_name, params, diagnostics), params


def test_the_slash_path_runs_without_the_duplicate_and_names_it_on_the_span(workflow):
    app_workflow, _ = workflow
    diagnostics: dict = {}
    (is_valid, message, _, _), params = _validate(
        app_workflow, "connect_note", "note the link of Payroll about billing", diagnostics,
        name="Payroll", connected_name="Payroll", topic="billing")

    assert is_valid, message
    assert not params.connected_name and params.name == "Payroll"   # not given (the guard writes its default)
    assert diagnostics["argument_left_empty"] == ["connected_name"]
    line = "connected_name left empty: 'Payroll' is written once in your request and is the name."
    assert line in diagnostics["argument_notes"]
    assert not diagnostics.get("invalid_fields") and not diagnostics.get("argument_refusal")


def test_a_question_on_the_slash_path_shows_the_note(workflow):
    app_workflow, _ = workflow
    diagnostics: dict = {}
    (is_valid, message, _, fields), _ = _validate(
        app_workflow, "connect_note", "note the link of Payroll", diagnostics,
        name="Payroll", connected_name="Payroll", topic="NOT_FOUND")

    assert not is_valid and fields == ["topic"]
    assert "connected_name left empty: 'Payroll' is written once in your request and is the name." in message
    assert "Cannot run" not in message and diagnostics["argument_left_empty"] == ["connected_name"]
