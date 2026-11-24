import json
from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow

from ..retail_data import load_data
from ..tools.get_product_details import GetProductDetails


class Signature:
    """Get product details"""
    class Input(BaseModel):
        product_id: str = Field(
            json_schema_extra={"from_command_output": ["get_order_details.product_ids", "list_all_product_types.product_ids"]},
            description="The product ID (numeric string)",
            pattern=r"^(\d{10})$",
            examples=["6086499569"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        product_details: str = Field(
            description=(
                "Detailed product information of all product variants of a given product type"
                "including item id, options, availability and price"
            )
        )
        item_ids: List[str] = Field(
            default_factory=list,
            json_schema_extra={"ui": {"hidden": True}},
            description="The ids of the product's items (variants); empty when the product is not found.",
        )

    plain_utterances: List[str] = [
        "Can you give me more information about this product?",
        "I need the specs and availability for an item I saw earlier.",
        "What are the details of that product with the long ID?",
        "Is this product still in stock and what are its features?",
        "Can you check if this product is available and tell me more about it?",
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
        # the response text is the product_details field's JSON alone, as it was before the id list was added
        return fastworkflow.CommandResponse(
            response=output.model_dump_json(include={"product_details"}), output=output)

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        product_details = GetProductDetails.invoke(data=data, product_id=input.product_id)
        product = _json_object(product_details)
        variants = product.get("variants")
        return Signature.Output(
            product_details=product_details,
            item_ids=[str(v.get("item_id", key)) for key, v in variants.items() if isinstance(v, dict)]
            if isinstance(variants, dict) else [],
        )

def _json_object(text: str) -> dict:
    """The tool's JSON text as a dict; empty for its error text."""
    try:
        value = json.loads(text)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}