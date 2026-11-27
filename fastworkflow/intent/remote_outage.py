"""A remote decision-model outage must be visible, and must not change the fallback.

When the remote model's call fails (credits exhausted, a rejected key, a timeout) every caller keeps its
fallback: the local answer stands, or the candidates go to the question. This module is the one place that says so:
``report`` returns the marker callers put on their span or result (``error_class`` and ``status_code``, never the
message, which can carry a key) and writes a WARNING the first time and then at most one per ``NOTICE_INTERVAL_SECONDS``,
with the count of failures since the last notice.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

NOTICE_INTERVAL_SECONDS = 300.0

_lock = threading.Lock()
_state: dict = {"total": 0, "pending": 0, "last_notice": None}


def status_code_of(error: BaseException) -> Optional[int]:
    """The HTTP status carried by *error* or its causes (litellm sets ``status_code``), else None."""
    seen = 0
    current: Optional[BaseException] = error
    while current is not None and seen < 5:
        code = getattr(current, "status_code", None)
        if code is None:
            code = getattr(getattr(current, "response", None), "status_code", None)
        if isinstance(code, int):
            return code
        current = current.__cause__ or current.__context__
        seen += 1
    return None


def report(error: BaseException, where: str) -> dict:
    """Record one failed remote call made for *where*; returns ``{"error_class", "status_code"}``."""
    from fastworkflow.utils.logging import logger

    marker = {"error_class": type(error).__name__, "status_code": status_code_of(error)}
    now = time.monotonic()
    with _lock:
        _state["total"] += 1
        _state["pending"] += 1
        last = _state["last_notice"]
        due = last is None or now - last >= NOTICE_INTERVAL_SECONDS
        count = _state["pending"]
        if due:
            _state["last_notice"] = now
            _state["pending"] = 0
    if due:
        status = f", status {marker['status_code']}" if marker["status_code"] is not None else ""
        logger.warning(
            f"The remote decision model failed ({where}): {marker['error_class']}{status}. "
            f"{count} failure(s) since the last notice; the local answer or a question stands in. "
            f"Further notices at most every {int(NOTICE_INTERVAL_SECONDS)} s.")
    return marker


def reset() -> None:
    """Forget the counts and the last notice (tests)."""
    with _lock:
        _state.update(total=0, pending=0, last_notice=None)
