"""An Input's validators read the user's own words, never text the framework wrote (bd fix-ftdl.10).

When a later part of a turn is checked, the guards read the request with the answers of the turn's earlier parts after
it, so a value an earlier part found counts as written. The validators' ``command_text`` is the user's request alone:
a phrase in an earlier part's answer (a not-run listing, a command's description) is not something the user said.
Reuses the temporary copy of ``planning_tutorial_workflow`` of ``test_check_roles_one_step``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastworkflow.planning import planning_commands as sc
from fastworkflow import workflow_agent
from tests.test_check_roles_one_step import workflow  # noqa: F401  (workflow is a fixture)

#: ``quietly`` is switched off when the command text says "loudly"; ``member`` is a marked value.
READS_THE_TEXT = '''
from pydantic import BaseModel, Field, ValidationInfo, model_validator


class Signature:
    """Report one member's loans."""

    class Input(BaseModel):
        member: str = Field(description="the member's name", json_schema_extra={"copied_from_command_text": True})
        quietly: bool = Field(default=True, description="leave the details out")

        @model_validator(mode="after")
        def loudly_when_asked(self, info: ValidationInfo):
            if "loudly" in ((info.context or {}).get("command_text") or ""):
                self.quietly = False
            return self


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.member}"]
'''

REQUEST = "Report the loans of the member the first part found."
EARLIER_ANSWER = "Found the member Ann Lee. Not run: a report that says everything loudly."


@pytest.fixture
def command(workflow):  # noqa: F811
    from fastworkflow import RoutingRegistry
    _, folder = workflow
    (Path(folder) / "_commands" / "reads_the_text.py").write_text(READS_THE_TEXT, encoding="utf-8")
    RoutingRegistry.clear_registry()
    RoutingRegistry.get_definition(folder, load_cached=False)
    return workflow_agent.load_planning_command_map(folder)["reads_the_text"]


def test_a_validator_does_not_read_an_earlier_parts_answer_as_the_users_words(command):
    with_earlier = "\n\n".join([REQUEST, EARLIER_ANSWER])

    args, _ = sc.checked_arguments(command, {"member": "Ann Lee"}, with_earlier, None, user_text=REQUEST)

    # The guard read the earlier answer (Ann Lee is written there, not in the request); the validator did not.
    assert args["member"] == "Ann Lee"
    assert args["quietly"] is True


def test_a_validator_still_reads_what_the_user_wrote(command):
    request = "Report Ann Lee's loans loudly."

    args, _ = sc.checked_arguments(command, {"member": "Ann Lee"}, request, None, user_text=request)

    assert args["quietly"] is False


def test_without_a_separate_user_text_the_validators_read_the_request(command):
    args, _ = sc.checked_arguments(command, {"member": "Ann Lee"}, "Report Ann Lee's loans loudly.")

    assert args["quietly"] is False


def test_a_marked_value_only_an_earlier_part_found_is_refused_without_the_earlier_answer(command):
    with pytest.raises(sc.PlanningCommandInputError):
        sc.checked_arguments(command, {"member": "Ann Lee"}, REQUEST, None, user_text=REQUEST)
