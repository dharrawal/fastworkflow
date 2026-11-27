"""control_investigation: a planning command of the review-vocabulary fixture; it declares inputs, nothing runs."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Investigate a control."""

    plain_utterances = ["investigate the control"]

    class Input(BaseModel):
        control: str = Field(description="the control", json_schema_extra={"copied_from_command_text": True})
        family: str = Field(default="", description="the family")
        open_first_person: bool = Field(default=False, description="open the first")
        whole_list: bool = Field(default=False, description="the whole list")


class PlanGenerator:
    def check(self, input):
        return [do_not_run("record_entity")]

    def __result__(self, input):
        return "Planned."
