"""list_member_loans: the books the open member has on loan."""
from typing import Optional

import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Member, page


class Signature:
    """List the books the open member has on loan, with their due dates."""

    class Input(BaseModel):
        start: Optional[int] = Field(default=None, ge=0, description="the first row of the page")

    class Output(BaseModel):
        titles: list[str] = Field(description="each borrowed book's title, in row order",
                                  examples=[["Dune", "Emma"], ["Persuasion"]],
                                  json_schema_extra={"ui": {"table": "loans"}})
        due_dates: list[str] = Field(description="each loan's due date as an ISO date "
                                                 "(YYYY-MM-DD), in row order",
                                     examples=[["2026-10-20", "2026-11-05"], ["2026-10-18"]],
                                     json_schema_extra={"ui": {"table": "loans"}})
        total: int = Field(description="how many books the member has on loan, on every page")

    plain_utterances = [
        "what has this member borrowed",
        "show the next page of loans for this member",
        "how many books does this member have out",
        "give me the full list of their borrowed titles with due dates",
        "which books are on loan to this patron, and when are they due",
        "list the books this member currently holds",
        "every title this member has checked out, please",
        "what titles does this member have on loan, row by row",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        member: Member = workflow.record
        loans = member.library.loans_of_member(member.uid)
        rows = page(loans, input.start)
        return Signature.Output(titles=[t for t, _ in rows], due_dates=[d for _, d in rows],
                                total=len(loans))
