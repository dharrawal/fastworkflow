"""find_book: the books whose title or author holds a query."""
from typing import Optional

import fastworkflow
from pydantic import BaseModel, Field

from ...application.library import Library, page


class Signature:
    """Find books by title or author."""

    class Input(BaseModel):
        query: str = Field(description="words of the title or the author's name",
                           examples=["Middlemarch", "George Eliot"],
                           json_schema_extra={"copied_from_command_text": True})
        start: Optional[int] = Field(default=None, ge=0, description="the first row of the page")

    class Output(BaseModel):
        book_uids: list[str] = Field(description="the uid of each book found, in row order, such as "
                                                 "b1, b2; open_book takes one",
                                     examples=[["b1", "b2"], ["b3"]],
                                     json_schema_extra={"ui": {"table": "books"}})
        titles: list[str] = Field(description="the title of each book found, in row order",
                                  examples=[["Dune", "Dune Messiah"], ["Emma"]],
                                  json_schema_extra={"ui": {"table": "books"}})
        total: int = Field(description="how many books match, on every page")

    plain_utterances = [
        "find the book Dune",
        "look up books by Jane Austen",
        "search the catalogue for anything with Persuasion in it",
        "which titles mention Herbert",
        "any books with Messiah in the name",
        "run a search for Dune across titles and authors",
        "show the next page of search results for Austen",
        "list the catalogue entries that contain Jane",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        library: Library = workflow.record
        found = library.find_books(input.query)
        rows = page(found, input.start)
        return Signature.Output(book_uids=rows, titles=[library.books[u].title for u in rows],
                                total=len(found))
