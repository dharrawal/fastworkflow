"""privilege_audit: a planning command of the review-vocabulary fixture; it declares inputs, nothing runs."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Audit a right."""

    plain_utterances = ["audit the right"]

    class Input(BaseModel):
        right: str = Field(description="the right", json_schema_extra={"copied_from_command_text": True})
        include_risk: bool = Field(default=False, description="include the risk")


class PlanGenerator:
    def check(self, input):
        return [do_not_run("record_entity")]

    def __result__(self, input):
        return "Planned."
