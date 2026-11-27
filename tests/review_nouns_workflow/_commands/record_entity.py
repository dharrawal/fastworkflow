"""record_entity: the execution command the review-vocabulary planning commands never run."""

from typing import List

import fastworkflow
from fastworkflow.core.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    """Record an entity."""

    class Input(BaseModel):
        name: str = Field(description="a name")

    class Output(BaseModel):
        summary: str = Field(description="the recorded entity")

    plain_utterances = ["record the entity", "record this entity"]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> List[str]:
        return ["record the entity"]


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> fastworkflow.CommandResponse:
        return fastworkflow.CommandResponse(response=input.name, output=Signature.Output(summary=input.name))
