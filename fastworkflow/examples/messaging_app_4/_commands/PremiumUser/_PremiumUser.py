from typing import Any

import fastworkflow

from ...application.chatroom import ChatRoom
from ...application.user import PremiumUser, User


class Context:
    inherits = ["User"]
    parents = ["ChatRoom"]

    state_version = 1

    @classmethod
    def get_parent(cls, record: PremiumUser) -> ChatRoom:
        return record.chatroom

    @classmethod
    def get_state(cls, record: PremiumUser) -> dict[str, Any]:
        """As for User: the user with its room, so the restored object is a PremiumUser inside a room."""
        return {"room": record.chatroom.to_state_dict(), "name": record.name, "type": type(record).__name__}

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> User:
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)
        room = ChatRoom.from_state_dict(state["room"])
        user = next((u for u in room.users if u.name == state["name"]), None)
        if user is None:
            user = User.from_state_dict({"name": state["name"], "type": state["type"]}, room)
        return user
