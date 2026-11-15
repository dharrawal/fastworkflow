from typing import Any

import fastworkflow

from ...application.chatroom import ChatRoom
from ...application.user import User


class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, record: User) -> ChatRoom:
        return record.chatroom

    @classmethod
    def get_state(cls, record: User) -> dict[str, Any]:
        """Snapshot a user that is the anchor of the session (no room is the root).

        A user is normally reached inside a ChatRoom root, whose own hook snapshots it. As an anchor it is
        snapshotted with its room, so ``get_parent`` still has a room to climb to after a restore.
        """
        return {"room": record.chatroom.to_state_dict(), "name": record.name, "type": type(record).__name__}

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> User:
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)
        room = ChatRoom.from_state_dict(state["room"])
        user = next((u for u in room.users if u.name == state["name"]), None)
        if user is None:  # the user was not in its room's list: rebuild it against the room
            user = User.from_state_dict({"name": state["name"], "type": state["type"]}, room)
        return user
