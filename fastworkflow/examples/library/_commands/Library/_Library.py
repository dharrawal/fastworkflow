from typing import Any, Optional, Union

import fastworkflow

from ...application.library import Book, Library, Member


class Context:
    state_version = 1

    @classmethod
    def get_displayname(cls, record: Library) -> str:
        return "Library"

    @classmethod
    def get_state(cls, record: Library) -> dict[str, Any]:
        """Snapshot the library so a session standing on it can be evicted and restored.

        Without this hook the session is pinned (never evicted). Books and members hold the library
        they belong to, so the snapshot is of the whole library, once; the library rebuilds the links.
        """
        return record.to_state_dict()

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> Library:
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)
        return Library.from_state_dict(state)

    @classmethod
    def get_locator(cls, record: Union[Book, Member]) -> str:
        """The current context can be a book or a member inside the library: point at it by uid."""
        kind = "book" if type(record).__name__ == "Book" else "member"
        return f"{kind}:{record.uid}"

    @classmethod
    def find_by_locator(cls, anchor: Library, locator: str) -> Optional[Union[Book, Member]]:
        kind, _, uid = locator.partition(":")
        return (anchor.books if kind == "book" else anchor.members).get(uid)
