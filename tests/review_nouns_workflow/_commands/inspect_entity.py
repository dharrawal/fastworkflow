"""inspect_entity: a planning command of the review-vocabulary fixture; it declares inputs, nothing runs."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Inspect one entity."""

    plain_utterances = ["inspect the entity"]

    class Input(BaseModel):
        name: str = Field(description="the entity name", json_schema_extra={"copied_from_command_text": True})
        kind: Literal["person", "application", "unit", "right", "group", "login"] = Field(default="person", description="what kind of entity")


class PlanGenerator:
    def check(self, input):
        return [do_not_run("record_entity")]

    def __result__(self, input):
        return "Planned."
