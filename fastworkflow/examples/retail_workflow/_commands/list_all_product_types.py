import json
from typing import List

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.workflow import Workflow

# Import business-logic helper
from ..retail_data import load_data
from ..tools.list_all_product_types import ListAllProductTypes


class Signature:
    """List all product types"""
    """Metadata and parameter definitions for `list_all_product_types`."""

    class Input(BaseModel):
        """No parameters expected for this command."""

    class Output(BaseModel):
        status: str = Field(
            description="List of product type and product id tuples, or a JSON string representation of that list."
        )
        product_ids: List[str] = Field(
            default_factory=list,
            json_schema_extra={"ui": {"hidden": True}},
            description="The ids of all the product types, in the order of the listing.",
        )

    # ---------------------------------------------------------------------
    # Utterances
    # ---------------------------------------------------------------------

    plain_utterances: List[str] = [
        "What kind of products do you have in the store?",
        "Can you show me everything you carry?",
        "I'm curious about all the categories you offer.",
        "I'd like to browse your full product range.",
        "What are the different types of items you sell?",
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
        command_parameters: Signature.Input | None,
    ) -> fastworkflow.CommandResponse:
        output = self._process_command(workflow)
        return fastworkflow.CommandResponse(
            response=f"list of product types: {output.status}", output=output)

    def _process_command(self, workflow: Workflow) -> Signature.Output:
        """Run domain logic and wrap into `Signature.Output`."""
        data = load_data()
        result = ListAllProductTypes.invoke(data=data)
        listing = json.loads(result)
        return Signature.Output(status=result, product_ids=[str(pid) for pid in listing.values()])