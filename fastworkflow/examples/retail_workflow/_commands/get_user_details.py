import json
from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.core.workflow import Workflow

from ..retail_data import load_data
from ..tools.get_user_details import GetUserDetails


class Signature:
    """Get user details"""
    class Input(BaseModel):
        user_id: str = Field(
            description="The user ID to get details for",
            json_schema_extra={"from_command_output": ["find_user_id_by_email.user_id", "find_user_id_by_name_zip.user_id"]},
            pattern=r"^([a-z]+_[a-z]+_\d+)$",
            examples=["sara_doe_496"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        user_details: str = Field(
            description=(
                "Detailed user information such as "
                "first and last name, address, email, payment methods and "
                "the list of order id's"
            )
        )
        order_ids: List[str] = Field(
            default_factory=list,
            json_schema_extra={"ui": {"hidden": True}},
            description="The user's order ids, such as #W0000000; empty when the user is not found.",
        )
        payment_method_ids: List[str] = Field(
            default_factory=list,
            json_schema_extra={"ui": {"hidden": True}},
            description="The ids of the user's payment methods, such as credit_card_0000000; empty when the user is not found.",
        )

    plain_utterances: List[str] = [
        "Can you pull up my account info?",
        "I want to see all the orders linked to my profile.",
        "What details do you have on my user account?",
        "Can you show me everything tied to my profile?",
        "I'd like to review my account and recent activity.",
        "retrieve user details"
    ]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> fastworkflow.CommandResponse:
        output = self._process_command(workflow, command_parameters)
        # the response text is the user_details field's JSON alone, as it was before the id lists were added
        return fastworkflow.CommandResponse(
            response=output.model_dump_json(include={"user_details"}), output=output)

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        user_details = GetUserDetails.invoke(data=data, user_id=input.user_id)
        user = _json_object(user_details)
        return Signature.Output(
            user_details=user_details,
            order_ids=[str(order_id) for order_id in user.get("orders") or []],
            payment_method_ids=[str(key) for key in user.get("payment_methods") or {}],
        )


def _json_object(text: str) -> dict:
    """The tool's JSON text as a dict; empty for its error text."""
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}