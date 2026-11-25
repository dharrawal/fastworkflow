"""The review reply guards know input names only from the workflow's planning commands (bd fix-exvh).

A small workflow from another domain (stock keeping: ``sku``, ``colour``, ``express``) declares none of the names the
review code once listed (``person``, ``unit``, ``right``, ...). The guards work on the names it declares, and a name of
the old lists that it does not declare is not an input there.
"""
from __future__ import annotations

import pytest

from fastworkflow.review_engine import EditParseError, EditRefused, Verdict, apply_edits, parse_edits
from fastworkflow.review_engine import _normalize_parsed_subtask_edits
from fastworkflow.review_proposals import (
    Means, ProposedEdit, SubTask, SubTaskInput, SubTaskProposal, _proposed_edits_to_engine, declared_input_names,
    entity_input_names, workflow_scope)

PLAN = '''"""{name}: a planning command of a stock-keeping fixture."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Check stock."""

    plain_utterances = ["check the stock"]

    class Input(BaseModel):
        sku: str = Field(description="the item", json_schema_extra={{"copied_from_command_text": True}})
        colour: Literal["red", "blue"] = Field(default="red", description="the colour")
        express: bool = Field(default=False, description="ship fast")


class PlanGenerator:
    def check(self, input):
        return [do_not_run("noop")]

    def __result__(self, input):
        return "Planned."
'''
NOOP = '''"""noop"""
from typing import List

import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Do nothing."""

    class Input(BaseModel):
        sku: str = Field(description="the item")

    class Output(BaseModel):
        summary: str = Field(description="s")

    plain_utterances = ["do nothing", "do nothing at all"]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow, command_name):
        return ["do nothing"]


class ResponseGenerator:
    def __call__(self, workflow, command, input):
        return fastworkflow.CommandResponse(response="ok", output=Signature.Output(summary="ok"))
'''


@pytest.fixture
def stock_workflow(tmp_path):
    commands = tmp_path / "stock" / "_commands"
    commands.mkdir(parents=True)
    (commands / "check_stock.py").write_text(PLAN.format(name="check_stock"))
    (commands / "noop.py").write_text(NOOP)
    folder = str(tmp_path / "stock")
    with workflow_scope(folder):
        yield folder


def _two_steps(*inputs: SubTaskInput) -> SubTaskProposal:
    return SubTaskProposal(items=[
        SubTask(goal="Check the stock", inputs=list(inputs), means=Means(kind="planning", name="check_stock")),
        SubTask(goal="notify"),
    ])


def test_the_declared_names_are_the_workflows_own_and_none_of_the_old_nouns(stock_workflow):
    assert declared_input_names() == {"sku", "colour", "express"}
    assert entity_input_names() == {"sku"}
    assert not {"person", "unit", "right", "control", "tag"} & declared_input_names()


def test_without_a_workflow_nothing_is_declared():
    assert declared_input_names() == frozenset()
    assert entity_input_names() == frozenset()


def test_a_declared_input_named_in_a_change_becomes_an_inputs_edit(stock_workflow):
    proposal = _two_steps(SubTaskInput(name="sku", value="Widget"))
    edits = _normalize_parsed_subtask_edits(proposal, parse_edits("change 1: express=true"))
    assert [(e.field, e.value) for e in edits] == [("inputs", "express=true")]
    # a declared input of the workflow that the step does not carry yet is still an input
    edits = _normalize_parsed_subtask_edits(_two_steps(), parse_edits("change 1: colour=blue"))
    assert [(e.field, e.value) for e in edits] == [("inputs", "colour=blue")]


def test_a_noun_of_the_old_list_the_workflow_does_not_declare_is_not_an_input(stock_workflow):
    proposal = _two_steps(SubTaskInput(name="sku", value="Widget"))
    edits = _normalize_parsed_subtask_edits(proposal, parse_edits("change 1: person=Lee Park"))
    assert [(e.field, e.value) for e in edits] == [("person", "Lee Park")]
    with pytest.raises(EditRefused) as refused:
        apply_edits(proposal, edits)
    assert "no field 'person'" in str(refused.value)


def test_a_quoted_name_completes_the_declared_entity_input_and_the_kind_word_sets_the_declared_choice(stock_workflow):
    proposal = _two_steps(SubTaskInput(name="sku", value="Widget"), SubTaskInput(name="colour", value="blue"))
    reply = 'Drop the second step; "Widget Pro Max" is one entity, red, in step 1.'
    edits = _proposed_edits_to_engine(proposal, [ProposedEdit(op="drop", index=2)], reply_text=reply)
    after = apply_edits(proposal, edits)
    assert len(after.items) == 1
    assert {i.name: i.value for i in after.items[0].inputs} == {"sku": "Widget Pro Max", "colour": "red"}


def test_a_word_the_workflow_does_not_declare_as_a_choice_is_not_inferred(stock_workflow):
    proposal = _two_steps(SubTaskInput(name="sku", value="Widget"), SubTaskInput(name="colour", value="blue"))
    reply = 'Drop the second step; "Widget Pro Max" is one entity, an organization, in step 1.'
    edits = _proposed_edits_to_engine(proposal, [ProposedEdit(op="drop", index=2)], reply_text=reply)
    after = apply_edits(proposal, edits)
    assert {i.name: i.value for i in after.items[0].inputs} == {"sku": "Widget Pro Max", "colour": "blue"}


def test_an_old_entity_noun_is_not_the_step_entity_here(stock_workflow):
    # `person` is not declared, so a step that carries it has no entity input of its own and a quoted name sets nothing
    proposal = SubTaskProposal(items=[
        SubTask(goal="Check the stock", inputs=[SubTaskInput(name="person", value="Lee")],
                means=Means(kind="planning", name="check_stock")),
        SubTask(goal="notify"),
    ])
    reply = 'Drop the second step; "Lee Park" is one entity in step 1.'
    edits = _proposed_edits_to_engine(proposal, [ProposedEdit(op="drop", index=2)], reply_text=reply)
    after = apply_edits(proposal, edits)
    assert {i.name: i.value for i in after.items[0].inputs} == {"person": "Lee"}
