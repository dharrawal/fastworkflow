"""person_sweep: a planning command of the review-vocabulary fixture; it declares inputs, nothing runs."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Sweep a person."""

    plain_utterances = ["sweep the person"]

    class Input(BaseModel):
        person: str = Field(description="the person", json_schema_extra={"copied_from_command_text": True})
        right: str = Field(default="x", description="the right", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def check(self, input):
        return [do_not_run("record_entity")]

    def __result__(self, input):
        return "Planned."
