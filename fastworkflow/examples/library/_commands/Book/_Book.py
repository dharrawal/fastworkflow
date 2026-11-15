from typing import Any

import fastworkflow

from ...application.library import Book, Library


class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, record: Book) -> Library:
        return record.library

    @classmethod
    def get_displayname(cls, record: Book) -> str:
        return f"Book: {record.title} ({record.uid})"

    @classmethod
    def get_state(cls, record: Book) -> dict[str, Any]:
        """A book is only ever reached through its library, so the snapshot is the library plus the book's uid."""
        return {"library": record.library.to_state_dict(), "uid": record.uid}

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> Book:
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)
        return Library.from_state_dict(state["library"]).books[state["uid"]]
