from typing import Any

import fastworkflow

from ...application.library import Library, Member


class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, record: Member) -> Library:
        return record.library

    @classmethod
    def get_displayname(cls, record: Member) -> str:
        return f"Member: {record.name} ({record.uid})"

    @classmethod
    def get_state(cls, record: Member) -> dict[str, Any]:
        """A member is only ever reached through its library, so the snapshot is the library plus the member's uid."""
        return {"library": record.library.to_state_dict(), "uid": record.uid}

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> Member:
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)
        return Library.from_state_dict(state["library"]).members[state["uid"]]
