"""Cursor issue, validation, and compact agent-visible tokens."""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Mapping, Optional

from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.observation_offloading.state import record_event
from fastworkflow.result_handles.common import (
    CURSOR_TOKEN_EXAMPLE, CURSOR_VERSION, FIRST_CURSOR_PAGE, MAX_CURSOR_PAGE,
    ResultHandleError, CURSOR_TOKEN_OVERLONG_RE as _CURSOR_TOKEN_OVERLONG_RE,
    CURSOR_TOKEN_RE as _CURSOR_TOKEN_RE, CURSOR_TOKEN_TRIM as _CURSOR_TOKEN_TRIM,
)

if TYPE_CHECKING:
    # Annotation only. This component is handed a store; it never looks one up,
    # and importing the class for real would give the low-level cursor module a
    # runtime edge to its own persistence peer, which is the dependency
    # direction this split exists to remove.
    from fastworkflow.result_handles.store import ResultHandleStore

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_cursor_tokens: dict[str, dict[str, Any]] = {}


def reset_cursor_state() -> None:
    with _lock:
        _cursor_tokens.clear()


def release_scope(scope_id: str) -> None:
    prefix = "%s@" % str(scope_id)
    with _lock:
        for key in [key for key in _cursor_tokens if key.startswith(prefix)]:
            _cursor_tokens.pop(key, None)
def _cache_namespace(scope: RuntimeHandleScope, store_: Any) -> str:
    return "%s@%s" % (scope.scope_id, store_.db_path)

def cursor_token(alias: str, tag: str, page: int) -> str:
    """``O7/p2``, or ``O7/f1p2`` for the first filtered traversal of O7."""
    return "%s/%s%s%d" % (alias, tag, "p", int(page))


def cursor_placeholder(alias: str, tag: str = "", *, pages_at_most: int = 0) -> str:
    """The widest token this traversal could print, for measuring a header.

    A page is packed against the header it will finally carry, so the cursor the
    packer measures must never be narrower than the cursor the page prints. The
    real ordinal is not known until the packer has answered, so the placeholder
    is all nines at the widest the ordinal could be: an ordinal is only ever
    allocated for a distinct position, so it cannot exceed the row count plus
    the one page this call is about to add.

    Callers that measure a header before they know the offset (the IDO bounded
    listing helper is one) should use this rather than issuing a real token for
    a position they may never serve.
    """
    digits = max(4, len(str(max(int(pages_at_most), 1))))
    return "%s/%sp%s" % (alias, tag, "9" * digits)


def _cursor_tag(
    scope: RuntimeHandleScope, store_: "ResultHandleStore", alias: str, query_scope: str
) -> str:
    if not query_scope:
        return ""
    try:
        return store_.cursor_tag(scope, alias=alias, query_scope=query_scope)
    except Exception as error:  # noqa: BLE001
        # A tag this process invented still scopes the token correctly for the
        # rest of the turn; the refusal path below is what protects the rows.
        record_event({"kind": "result_handle_cursor_tag_failed",
                      "scope_id": scope.scope_id, "alias": alias,
                      "error": type(error).__name__})
        logger.warning("result handle cursor tag failed: %s", error)
        return "f1"


def _remember_token(namespace: str, token: str, payload: Mapping[str, Any]) -> None:
    with _lock:
        _cursor_tokens["%s|%s" % (namespace, token)] = dict(payload)


def _recall_token(namespace: str, token: str) -> Optional[dict[str, Any]]:
    with _lock:
        payload = _cursor_tokens.get("%s|%s" % (namespace, token))
    return None if payload is None else dict(payload)


def encode_cursor(
    *,
    alias: str,
    query_scope: str,
    position: int,
    descriptor_sha256: str,
    scope: Optional[RuntimeHandleScope] = None,
    selected_store: Optional["ResultHandleStore"] = None,
) -> str:
    """Issue the short token that resumes ``alias`` at ``position``.

    (ido-986.14.11) The returned string is the whole agent-visible cursor and
    carries nothing: the query scope, the offset and the descriptor digest are
    written to the store under the token, which is what makes the token short
    enough to type and impossible to edit into another query. Issuing is
    idempotent per position, so the same resumption point always prints the same
    token.

    ``scope`` and ``selected_store`` are new keyword arguments; both default to
    the ambient scope and store, so existing keyword calls keep working.
    """
    if scope is None or selected_store is None:
        raise RuntimeError("cursor implementation requires explicit scope and store")
    selected_scope = scope
    store_ = selected_store
    tag = _cursor_tag(selected_scope, store_, alias, query_scope)
    digest = str(descriptor_sha256 or "")[:16]
    try:
        page = store_.issue_cursor(
            selected_scope, alias=alias, tag=tag, query_scope=query_scope,
            position=int(position), descriptor_sha256=digest,
        )
    except Exception as error:  # noqa: BLE001
        # The store is the durable copy, not the only one. A page that cannot
        # write its token still serves its rows and still continues inside this
        # process; the event says the durability was lost.
        record_event({"kind": "result_handle_cursor_issue_failed",
                      "scope_id": selected_scope.scope_id, "alias": alias,
                      "error": type(error).__name__})
        logger.warning("result handle cursor could not be stored: %s", error)
        page = _fallback_page(
            _cache_namespace(selected_scope, store_), alias, tag, int(position)
        )
    token = cursor_token(alias, tag, page)
    _remember_token(
        _cache_namespace(selected_scope, store_),
        token,
        {"v": CURSOR_VERSION, "h": alias, "q": query_scope, "p": int(position),
         "d": digest},
    )
    return token


def _fallback_page(namespace: str, alias: str, tag: str, position: int) -> int:
    """An ordinal for a token the store refused to write. Process-local only."""
    prefix = "%s|%s/%sp" % (namespace, alias, tag)
    with _lock:
        for key, payload in _cursor_tokens.items():
            if key.startswith(prefix) and int(payload.get("p") or 0) == position:
                return int(str(key).rsplit("p", 1)[-1])
        issued = [int(str(key).rsplit("p", 1)[-1])
                  for key in _cursor_tokens if key.startswith(prefix)]
    return max(issued or [FIRST_CURSOR_PAGE - 1]) + 1


def _parse_cursor_token(cursor: str) -> tuple[str, str, int]:
    """``"O7/f1p2"`` -> ``("O7", "f1", 2)``, or a refusal a caller can act on.

    Deliberately literal. Whitespace and the quoting a model wraps a copied
    value in are trimmed, and the fixed letters are case-folded, because none of
    that can change which handle or which traversal the token names. Nothing
    else is repaired: a token with a different handle, a different tag or a
    different ordinal is a different token and is refused by name below, never
    guessed at.
    """
    text = str(cursor or "").strip().strip(_CURSOR_TOKEN_TRIM).replace(" ", "")
    match = _CURSOR_TOKEN_RE.match(text)
    if match is None:
        overlong = _CURSOR_TOKEN_OVERLONG_RE.match(text)
        if overlong is not None:
            # (ido-1de, F22) Refused here, as this module's own error, before
            # int() or SQLite ever sees the digits.
            raise ResultHandleError(
                "that page token names a page %d digits long; the largest page a "
                "traversal can name is %d. A page token is printed on the page it "
                "continues as next_cursor=%s - copy it from that page, or omit "
                "cursor to start this query at its first page."
                % (len(overlong.group("page")), MAX_CURSOR_PAGE,
                   CURSOR_TOKEN_EXAMPLE)
            )
        raise ResultHandleError(
            "%r is not a page token. A page token is short and is printed on the "
            "page it continues as next_cursor=%s - the result handle, then the "
            "page. Copy it from that page, or omit cursor to start this query at "
            "its first page." % (str(cursor)[:40], CURSOR_TOKEN_EXAMPLE)
        )
    alias = match.group("alias").upper()
    tag = (match.group("tag") or "").lower()
    page = int(match.group("page"))
    if page > MAX_CURSOR_PAGE:
        # (ido-1de, F22) Same refusal, for an ordinal short enough to parse and
        # still far past any page this traversal could have issued.
        raise ResultHandleError(
            "page token %s names page %d; the largest page a traversal can name "
            "is %d. Copy the token from the page it continues, or omit cursor to "
            "start this query at its first page."
            % (cursor_token(alias, tag, page), page, MAX_CURSOR_PAGE)
        )
    if page < FIRST_CURSOR_PAGE:
        raise ResultHandleError(
            "page token %s names page %d; page 1 is the call that passes no "
            "cursor at all, so omit cursor to read it."
            % (cursor_token(alias, tag, page), page)
        )
    return alias, tag, page


def _issued_tokens(
    scope: RuntimeHandleScope, store_: "ResultHandleStore", alias: str
) -> list[str]:
    try:
        rows = store_.list_cursors(scope, alias=alias)
    except Exception:  # noqa: BLE001
        rows = []
    tokens = [cursor_token(alias, str(row["tag"]), int(row["page"])) for row in rows]
    prefix = "%s|%s/" % (_cache_namespace(scope, store_), alias)
    with _lock:
        tokens.extend(key[len(prefix) - len(alias) - 1:]
                      for key in _cursor_tokens if key.startswith(prefix))
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return seen


def decode_cursor(
    cursor: str,
    *,
    alias: Optional[str] = None,
    scope: Optional[RuntimeHandleScope] = None,
    selected_store: Optional["ResultHandleStore"] = None,
) -> dict[str, Any]:
    """Resolve a page token to the resumption point it was issued for.

    (ido-986.14.11) The payload is unchanged - ``v``, ``h``, ``q``, ``p``, ``d``
    - so every check that read a decoded cursor still reads one; only its source
    moved, from the string the agent typed to the row the store issued. That is
    the whole point: a token the store never issued resolves to nothing at all,
    so a single mistyped character can no longer decode into a valid position on
    some other handle.

    ``alias``, ``scope`` and ``selected_store`` are new keyword arguments;
    ``alias`` is the handle the call is for, checked first so the refusal names
    both handles.
    """
    token_alias, tag, page = _parse_cursor_token(cursor)
    if alias and token_alias != alias:
        raise ResultHandleError(
            "this cursor belongs to result handle %s, not %s. Page tokens carry "
            "their handle, so %s cannot be continued with a token issued for %s; "
            "omit cursor to start %s at its first page"
            % (token_alias, alias, alias, token_alias, alias)
        )
    if scope is None or selected_store is None:
        raise RuntimeError("cursor implementation requires explicit scope and store")
    selected_scope = scope
    store_ = selected_store
    token = cursor_token(token_alias, tag, page)
    payload = _recall_token(_cache_namespace(selected_scope, store_), token)
    if payload is None:
        row = store_.get_cursor(selected_scope, alias=token_alias, tag=tag, page=page)
        if row is not None:
            payload = {"v": CURSOR_VERSION, "h": token_alias,
                       "q": str(row["query_scope"]), "p": int(row["position"]),
                       "d": str(row["descriptor_sha256"])[:16]}
            _remember_token(
                _cache_namespace(selected_scope, store_), token, payload
            )
        else:
            issued = _issued_tokens(selected_scope, store_, token_alias)
            raise ResultHandleError(
                "no page token %s has been issued for %s in this turn (%s). A "
                "page token is only ever printed by the page it continues; it "
                "cannot be composed. Omit cursor to start this query at its "
                "first page."
                % (
                    token,
                    token_alias,
                    ("tokens issued for this handle: " + ", ".join(issued))
                    if issued
                    else "no page of this handle has offered a continuation yet",
                )
            )
    return payload
