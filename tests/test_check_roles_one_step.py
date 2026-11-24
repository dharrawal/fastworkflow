"""Two inputs given one value: the one duplicate rule, part of the strict check (bd fix-qjne.20; earlier
fix-3rsq.check-roles-one-step).

Marked ``str`` fields (``copied_from_command_text`` / ``record_named_in_command_text``) that hold one value are refused
only when the request writes that value fewer times than the fields that hold it, by the one guard
(``parameters.check.check_plan``) on both paths and both kinds of command. Unmarked fields are never compared (the
earlier ``_check_roles`` refused any two required ``str`` inputs of a planning command with one value; it is gone).

Integration only: a temporary copy of ``planning_tutorial_workflow`` with three commands added (two
planning commands, one execution command), the real loader, ``check_arguments`` and ``validate_parameters``.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest
import fastworkflow
from fastworkflow import planning_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.utils.signatures import InputForParamExtraction

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"

#: Two name inputs (``in_request``): the same value for both is refused on the turn-frame path.
NAME_PAIR = '''
from pydantic import BaseModel, Field


class Signature:
    """Move one member's loan of a book to a branch."""

    class Input(BaseModel):
        member: str = Field(description="the member whose loan moves", json_schema_extra={"copied_from_command_text": True})
        branch: str = Field(description="the branch the loan moves to", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.member}", f"move the loan to the {input.branch} branch"]
'''

#: One name input and one plain ``str`` input (no ``in_request``): still checked for roles.
NAME_AND_FREE_TEXT = '''
from pydantic import BaseModel, Field


class Signature:
    """File a note about one member."""

    class Input(BaseModel):
        member: str = Field(description="the member the note is about", json_schema_extra={"copied_from_command_text": True})
        title: str = Field(description="a title for the note")


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.member}", f"file the note titled {input.title}"]
'''

#: An execution command with two required ``str`` inputs: never checked for roles.
PAIR_NAMES = '''
from pydantic import BaseModel, Field


class Signature:
    """Pair two names."""

    class Input(BaseModel):
        first: str = Field(description="a name")
        second: str = Field(description="a name")

    class Output(BaseModel):
        pass


class ResponseGenerator:
    def __call__(self, workflow, command, command_parameters):
        return Signature.Output()
'''

REQUEST = "Move the loan of member Ann Lee to branch Ann Lee"
#: The request writes the value once: two marked fields cannot both hold it.
REQUEST_ONCE = "Move the loan of member Ann Lee to its branch"


@pytest.fixture
def workflow(tmp_path):
    """The tutorial copy with the three commands, loaded for ``validate_parameters``."""
    folder = tmp_path / "roles_wf"
    shutil.copytree(TUTORIAL, folder,
                    ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (folder / "_commands" / "name_pair.py").write_text(NAME_PAIR, encoding="utf-8")
    (folder / "_commands" / "name_and_free_text.py").write_text(NAME_AND_FREE_TEXT, encoding="utf-8")
    (folder / "_commands" / "pair_names.py").write_text(PAIR_NAMES, encoding="utf-8")
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(str(folder), load_cached=False)
    app_workflow = fastworkflow.Workflow.create(str(folder), workflow_id_str=f"roles_{uuid.uuid4().hex[:8]}")
    yield app_workflow, str(folder)
    fastworkflow.RoutingRegistry.clear_registry()


def _validate(app_workflow, command_name: str, request: str, **values):
    """``validate_parameters`` for *command_name* with the given values, as the turn-frame path calls it."""
    planning = workflow_agent.load_planning_command_map(app_workflow.folderpath).get(command_name)
    if planning is not None:
        model = planning.input_model
    else:
        definition = fastworkflow.RoutingRegistry.get_definition(app_workflow.folderpath)
        response_generator = definition.get_command_class(
            command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        model = sys.modules[response_generator.__module__].Signature.Input
    extraction = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
    return extraction.validate_parameters(app_workflow, command_name, model(**values))


def test_two_name_inputs_with_one_value_written_once_are_refused_on_the_turn_frame_path(workflow):
    app_workflow, _ = workflow

    is_valid, message, _, fields = _validate(app_workflow, "name_pair", REQUEST_ONCE,
                                             member="Ann Lee", branch="Ann Lee")

    assert not is_valid
    # the refusal names the argument it refused (the one whose role the request does not write before the value)
    assert fields == ["branch"] and "'Ann Lee' is the member in your request, not the branch" in message
    # written twice, the two fields may both hold it
    is_valid, message, _, _ = _validate(app_workflow, "name_pair", REQUEST, member="Ann Lee", branch="Ann Lee")
    assert is_valid, message


def test_the_turn_frame_refusal_matches_the_typed_call(workflow):
    app_workflow, folder = workflow
    command = workflow_agent.load_planning_command_map(folder)["name_pair"]

    with pytest.raises(sc.PlanningCommandInputError) as typed:
        sc.check_arguments(command, {"member": "Ann Lee", "branch": "Ann Lee"}, REQUEST_ONCE)

    is_valid, message, _, fields = _validate(app_workflow, "name_pair", REQUEST_ONCE,
                                             member="Ann Lee", branch="Ann Lee")
    assert not is_valid
    # one refusal, and the person reads the same words on both paths
    assert fields == ["branch"] and sc.refusal_question(command, typed.value) in message


def test_a_plain_str_input_equal_to_a_name_is_not_compared_on_either_path(workflow):
    app_workflow, folder = workflow
    request = "File a note about member Ann Lee"
    command = workflow_agent.load_planning_command_map(folder)["name_and_free_text"]
    args, _ = sc.check_arguments(command, {"member": "Ann Lee", "title": "Ann Lee"}, request)

    assert sc._validate_arguments(command, args, request, None)["title"] == "Ann Lee"
    is_valid, message, _, _ = _validate(app_workflow, "name_and_free_text", request,
                                        member="Ann Lee", title="Ann Lee")
    assert is_valid, message


def test_an_execution_command_with_two_equal_required_strings_is_not_refused(workflow):
    app_workflow, _ = workflow

    is_valid, message, _, _ = _validate(app_workflow, "pair_names", "Pair Ann Lee with Ann Lee",
                                        first="Ann Lee", second="Ann Lee")

    assert is_valid, message
