"""Durable notices that a channel's live session was dropped by the server.

A session whose workflow state cannot be checkpointed (no ``get_state`` hook) is lost when the server expires it
under capacity pressure, or when the client ends it. The next call on the channel would otherwise start a fresh
session without a word, and a client could carry on believing its workflow context was still there. This store
keeps one small record per dropped channel, so the next session created for that channel can say so exactly once
(the ``session_notice`` field of the first turn result and of ``/initialize``).

Failing open is the rule here: a record that cannot be written or read is logged and skipped, never raised,
because the notice is information about a loss that has already happened and must not turn it into a failed
request. The record files are tiny and reaped by age.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from fastworkflow.session.storage_keys import encode_path_component
from fastworkflow.utils.logging import logger

NOTICE_CODE = "context_dropped"
# Records older than this are reclaimed by the periodic reaper: a channel unseen for a month is not coming back
# to read a notice, and the directory must not grow without bound.
NOTICE_RETENTION_SECONDS = 30 * 24 * 3600

REASON_EXPIRED = "expired"
REASON_EXPIRED_PENDING = "expired_pending_question"
REASON_ENDED = "ended"

_MESSAGES = {
    REASON_EXPIRED: (
        "The server dropped this channel's workflow context after it sat idle while the server was at its "
        "session limit. This session starts from the launch configuration. Conversation history was kept."
    ),
    REASON_EXPIRED_PENDING: (
        "The server dropped this channel's workflow context, and a question that was waiting for your answer, "
        "after the channel sat idle for a long time while the server was at its session limit. This session "
        "starts from the launch configuration and nothing is pending. Conversation history was kept."
    ),
    REASON_ENDED: (
        "This channel's session was ended and its workflow context could not be saved, so this session "
        "starts from the launch configuration. Conversation history was kept."
    ),
}


class SessionNoticeStore:
    """One JSON file per channel under a workflow-namespaced folder."""

    def __init__(self, base_folder: str) -> None:
        self._base = base_folder

    def _path(self, channel_id: str) -> str:
        return os.path.join(self._base, encode_path_component(channel_id) + ".json")

    def record(self, channel_id: str, reason: str) -> None:
        """Write the notice for a dropped channel. Never raises."""
        try:
            os.makedirs(self._base, exist_ok=True)
            payload = {"code": NOTICE_CODE, "reason": reason, "at": time.time()}
            path = self._path(channel_id)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 - fail open (see module docstring)
            logger.warning(f"Could not record the dropped-session notice ({type(exc).__name__}: {exc})")

    def take(self, channel_id: str) -> Optional[dict[str, Any]]:
        """Read and delete the channel's notice, or None. Never raises."""
        try:
            path = self._path(channel_id)
            if not os.path.isfile(path):
                return None
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            with_suppress_remove(path)
            reason = str(payload.get("reason") or REASON_EXPIRED)
            return {
                "code": NOTICE_CODE,
                "reason": reason,
                "message": _MESSAGES.get(reason, _MESSAGES[REASON_EXPIRED]),
            }
        except Exception as exc:  # noqa: BLE001 - fail open
            logger.warning(f"Could not read the dropped-session notice ({type(exc).__name__}: {exc})")
            return None

    def reap(self, max_age_seconds: float = NOTICE_RETENTION_SECONDS) -> int:
        """Delete records older than ``max_age_seconds``. Returns how many. Never raises."""
        removed = 0
        try:
            if not os.path.isdir(self._base):
                return 0
            cutoff = time.time() - max_age_seconds
            for name in os.listdir(self._base):
                path = os.path.join(self._base, name)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        removed += 1
                except OSError:
                    continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Dropped-session notice reap failed ({type(exc).__name__}: {exc})")
        return removed


def with_suppress_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
