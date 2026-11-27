"""review_proposal: put a typed proposal to the user and return the accepted one (fix-3rsq.plan-review.1).

A framework execution command (fastworkflow/review/commands; system machinery, not an agent or user tool):
the review loop, called once by the review planning command
(review_subtasks). The first call opens the review and suspends the agent on the prompt
(ask_user): the user's replies edit the proposal, and only confirm or decline ends the review. On accept
the framework runs this command again under the accepted record (review_engine.approved_review), and the
command then returns the accepted proposal as its typed Output. The agent never produces an accepted
proposal itself.
"""
import fastworkflow
from pydantic import BaseModel, Field

from fastworkflow.review import review_engine


class Signature:
    """Put a proposal to the user for review and return what the user accepted."""

    class Input(BaseModel):
        items: str = Field(default="", description="the proposal's items, as the proposal reads them")

    class Output(BaseModel):
        accepted: str = Field(description="the accepted items, as JSON")

    plain_utterances = ["review the sub-tasks with me", "let me approve the plan first"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        from fastworkflow.review.review_proposals import SubTaskProposal as cls

        accepted = review_engine.approved_review_proposal(workflow, cls.spec)
        if accepted is not None:
            return Signature.Output(accepted=accepted.model_dump_json())
        from fastworkflow.review import review_proposals

        proposal = review_proposals.draft_for(command_parameters.items) or cls.from_text(command_parameters.items)
        review_engine.open_for_workflow(workflow, proposal,
                                        items_text=command_parameters.items)   # raises AskUserSuspend
        raise AssertionError("open_for_workflow must suspend")
