import json
from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict, model_validator
from fastworkflow.workflow import Workflow

# Domain helpers
from ..retail_data import load_data
from ..tools.get_order_details import GetOrderDetails


class Signature:
    """Get order details"""
    class Input(BaseModel):
        order_id: str = Field(
            json_schema_extra={"from_command_output": "get_user_details.order_ids"},
            description=(
                "The order ID to get details for"
            ),
            pattern=r"^(#?[\w\d]+)$",
            examples=["#W0000000"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

        @model_validator(mode="after")
        def order_id_has_hash(self, info):
            if not self.order_id.startswith('#'):
                self.order_id = f'#{self.order_id}'
            return self

    class Output(BaseModel):
        order_details: str = Field(
            description=(
                "Detailed information about the order such as "
                "shipping address, "
                "items ordered with details including name, product and item id, price and options, "
                "fulfillments with details including tracking id and item ids, "
                "order status, and"
                "payment history with details including transaction type, amount and payment method id"
            )
        )
        item_ids: List[str] = Field(
            default_factory=list,
            json_schema_extra={"ui": {"hidden": True}},
            description="The ids of the items in the order; empty when the order is not found.",
        )
        product_ids: List[str] = Field(
            default_factory=list,
            json_schema_extra={"ui": {"hidden": True}},
            description="The ids of the products the order's items belong to; empty when the order is not found.",
        )
        payment_method_ids: List[str] = Field(
            default_factory=list,
            json_schema_extra={"ui": {"hidden": True}},
            description="The ids of the payment methods the order was paid with; empty when the order is not found.",
        )

    plain_utterances: List[str] = [
        "Can you tell me what's going on with my order #W1234567?",
        "I want to check the status of the order I placed yesterday.",
        "What's the current update on my recent purchase?",
        "Can you pull up the details of my last order?",
        "Is there any info on when my package will be delivered?",
    ]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        from fastworkflow.train.generate_synthetic import generate_diverse_utterances

        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)

class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        command_parameters: Signature.Input,
    ) -> fastworkflow.CommandResponse:
        output = self._process_command(workflow, command_parameters)
        # the response text is the order_details field's JSON alone, as it was before the id lists were added
        return fastworkflow.CommandResponse(
            response=output.model_dump_json(include={"order_details"}), output=output)

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        order_details = GetOrderDetails.invoke(data=data, order_id=input.order_id)
        order = _json_object(order_details)
        items = [item for item in order.get("items") or [] if isinstance(item, dict)]
        payments = [p for p in order.get("payment_history") or [] if isinstance(p, dict)]
        return Signature.Output(
            order_details=order_details,
            item_ids=[str(item["item_id"]) for item in items if "item_id" in item],
            product_ids=list(dict.fromkeys(str(item["product_id"]) for item in items if "product_id" in item)),
            payment_method_ids=list(dict.fromkeys(
                str(p["payment_method_id"]) for p in payments if "payment_method_id" in p)),
        )

def _json_object(text: str) -> dict:
    """The tool's JSON text as a dict; empty for its error text."""
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}