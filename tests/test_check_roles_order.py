"""Two unmarked inputs given one value are left to the Input's validators on both paths (bd fix-qjne.13, fix-qjne.20).

A validator may correct a value (write it to ``self``), so two inputs given one value can differ once it ran. The
typed planning call (``planning_commands._validate_arguments``) and the turn-frame path (``validate_parameters``) run
the author's validators; the one duplicate rule reads only marked fields (``test_check_roles_one_step``), so two
unmarked fields with one value are not refused by the framework. Reuses the temporary copy of
``planning_tutorial_workflow`` of ``test_check_roles_one_step``.
"""

from __future__ import annotations

import pytest

from fastworkflow.planning import planning_commands as sc
from fastworkflow.agent import workflow_agent
from fastworkflow.parameters.signatures import InputForParamExtraction
from tests.test_check_roles_one_step import REQUEST, workflow  # noqa: F401  (workflow is a fixture)

#: Two required str inputs; a validator corrects ``branch`` when it repeats ``member``, so the two then differ.
CORRECTING_PAIR = '''
from pydantic import BaseModel, Field, model_validator


class Signature:
    """Move one member's loan of a book to a branch."""

    class Input(BaseModel):
        member: str = Field(description="the member whose loan moves")
        branch: str = Field(description="the branch the loan moves to")

        @model_validator(mode="after")
        def correct_branch(self):
            if self.branch == self.member:
                self.branch = self.branch + " Branch"
            return self


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.member}", f"move the loan to the {input.branch} branch"]
'''

#: The same pair with no validator: a real duplicate.
PLAIN_PAIR = CORRECTING_PAIR.replace(
    '''        @model_validator(mode="after")
        def correct_branch(self):
            if self.branch == self.member:
                self.branch = self.branch + " Branch"
            return self
''', "")


def _validate(app_workflow, command_name: str, request: str, **values):
    """``validate_parameters`` on a ``model_construct`` instance, as the LLM extractor hands it over (the Input's
    own validators have not run: ``validate_parameters`` runs them)."""
    model = workflow_agent.load_planning_command_map(app_workflow.folderpath)[command_name].input_model
    extraction = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
    return extraction.validate_parameters(app_workflow, command_name, model.model_construct(**values))


@pytest.fixture
def pair_workflow(workflow):  # noqa: F811
    from fastworkflow import RoutingRegistry
    app_workflow, folder = workflow
    from pathlib import Path
    (Path(folder) / "_commands" / "correcting_pair.py").write_text(CORRECTING_PAIR, encoding="utf-8")
    (Path(folder) / "_commands" / "plain_pair.py").write_text(PLAIN_PAIR, encoding="utf-8")
    RoutingRegistry.clear_registry()
    RoutingRegistry.get_definition(folder, load_cached=False)
    return app_workflow, folder


def test_typed_call_validator_correction_comes_before_the_roles_guard(pair_workflow):
    _, folder = pair_workflow
    command = workflow_agent.load_planning_command_map(folder)["correcting_pair"]
    args, _ = sc.check_arguments(command, {"member": "Ann Lee", "branch": "Ann Lee"}, REQUEST)

    checked = sc._validate_arguments(command, args, REQUEST, None)

    assert checked["branch"] == "Ann Lee Branch"


def test_turn_frame_validator_correction_comes_before_the_roles_guard(pair_workflow):
    app_workflow, _ = pair_workflow

    is_valid, message, _, fields = _validate(app_workflow, "correcting_pair", REQUEST,
                                             member="Ann Lee", branch="Ann Lee")

    assert is_valid, message


def test_an_unmarked_duplicate_is_not_refused_on_either_path(pair_workflow):
    app_workflow, folder = pair_workflow
    command = workflow_agent.load_planning_command_map(folder)["plain_pair"]
    args, _ = sc.check_arguments(command, {"member": "Ann Lee", "branch": "Ann Lee"}, REQUEST)
    assert sc._validate_arguments(command, args, REQUEST, None)["branch"] == "Ann Lee"

    is_valid, message, _, _ = _validate(app_workflow, "plain_pair", REQUEST, member="Ann Lee", branch="Ann Lee")

    assert is_valid, message
