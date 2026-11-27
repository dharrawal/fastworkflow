"""group_walk: a planning command of the review-vocabulary fixture; it declares inputs, nothing runs."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Walk a group."""

    plain_utterances = ["walk the group"]

    class Input(BaseModel):
        group: str = Field(description="the group", json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def check(self, input):
        return [do_not_run("record_entity")]

    def __result__(self, input):
        return "Planned."
