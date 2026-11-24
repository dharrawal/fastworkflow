"""old_style_values: once the deprecated spelling (a required str declared with default="NOT_FOUND", now a load
error), migrated to a plain required field. ``tests/test_plain_required_fields.py`` writes the old spelling back on
a temp copy to show the load error."""

from typing import List

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    """Record one value."""

    class Input(BaseModel):
        name: str = Field(description="a name")

    class Output(BaseModel):
        summary: str = Field(description="the recorded value")

    plain_utterances = ["record the old style value"]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> List[str]:
        return ["record the old style value"]


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> fastworkflow.CommandResponse:
        summary = f"name={input.name}"
        return fastworkflow.CommandResponse(
            response=summary, output=Signature.Output(summary=summary))
