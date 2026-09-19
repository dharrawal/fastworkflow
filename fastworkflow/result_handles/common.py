"""Low-level values shared by result models, persistence, and paging."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

DEFAULT_PAGE_SIZE = 25
UNSORTED_OFFSET = "unsorted-offset"
#: Page 1 is the call that passes no cursor, so the first token a traversal
#: issues is page 2.
FIRST_CURSOR_PAGE = 2
CURSOR_VERSION = 1
CURSOR_TOKEN_EXAMPLE = "O7/p2"
MAX_CURSOR_PAGE = 999_999
MAX_CURSOR_PAGE_DIGITS = 12
MAX_ALIAS_DIGITS = 9
MAX_TAG_DIGITS = 6
CURSOR_TOKEN_RE = re.compile(
    r"^(?P<alias>[OD][1-9]\d{0,%d})/(?P<tag>[a-z]\d{1,%d})?p(?P<page>[1-9]\d{0,%d})$"
    % (MAX_ALIAS_DIGITS - 1, MAX_TAG_DIGITS, MAX_CURSOR_PAGE_DIGITS - 1),
    re.IGNORECASE,
)
CURSOR_TOKEN_OVERLONG_RE = re.compile(
    r"^[OD][1-9]\d{0,%d}/(?:[a-z]\d{1,%d})?p(?P<page>[1-9]\d{%d,})$"
    % (MAX_ALIAS_DIGITS - 1, MAX_TAG_DIGITS, MAX_CURSOR_PAGE_DIGITS),
    re.IGNORECASE,
)
CURSOR_TOKEN_TRIM = "`'\"<>[](){} \t\r\n,.;:"


class ResultHandleError(RuntimeError):
    """A handle, cursor or filter a caller can act on — never a crash.

    Unknown handle, a cursor written for another query scope, a descriptor that
    names an unregistered resolver: each is a reached decision, declined by
    name, so the calling command can refuse in the agent's own terms.
    """


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        default=str,
    ).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
