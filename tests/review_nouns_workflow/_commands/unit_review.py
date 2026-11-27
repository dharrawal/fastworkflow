"""unit_review: a planning command of the review-vocabulary fixture; it declares inputs, nothing runs."""
from typing import Literal

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Review a unit."""

    plain_utterances = ["review the unit"]

    class Input(BaseModel):
        unit: str = Field(description="the unit", json_schema_extra={"copied_from_command_text": True})
        with_login_only: bool = Field(default=False, description="logins only")


class PlanGenerator:
    def check(self, input):
        return [do_not_run("record_entity")]

    def __result__(self, input):
        return "Planned."
