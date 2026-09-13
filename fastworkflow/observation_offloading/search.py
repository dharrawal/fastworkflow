"""Bounded reads of offloaded observation handles."""
from __future__ import annotations

from typing import Any, Optional

from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.state import (
    archive,
    default_scope,
    record_event,
    stored_handles,
)

DEFAULT_PAGE_BYTES = 4096
SEARCH_MEMORY_MAX_PAGES = 3


class InvalidPageBoundary(ValueError):
    """A requested page would start inside a UTF-8 or newline record."""


def text_page(text: str, start_byte: int, max_bytes: int) -> dict[str, Any]:
    payload = text.encode("utf-8")
    if start_byte < 0 or start_byte > len(payload):
        raise InvalidPageBoundary("start_byte is outside the stored text")
    if start_byte and payload[start_byte - 1 : start_byte] != b"\n":
        raise InvalidPageBoundary("start_byte must be zero or immediately after a newline")
    if start_byte == len(payload):
        return {
            "start_byte": start_byte,
            "end_byte": start_byte,
            "text": "",
            "has_more": False,
            "total_bytes": len(payload),
        }
    candidate_end = min(len(payload), start_byte + max_bytes)
    if candidate_end < len(payload):
        newline = payload.rfind(b"\n", start_byte, candidate_end + 1)
        end_byte = candidate_end if newline < start_byte else newline + 1
    else:
        end_byte = len(payload)
    return {
        "start_byte": start_byte,
        "end_byte": end_byte,
        "text": payload[start_byte:end_byte].decode("utf-8"),
        "has_more": end_byte < len(payload),
        "total_bytes": len(payload),
    }


def _search_handles(
    handles: list[dict[str, Any]],
    terms: list[str],
    *,
    tier: str,
) -> tuple[list[str], bool]:
    chunks: list[str] = []
    pages_used = 0
    found = False
    for handle in handles:
        text = str(handle["text"])
        start = 0
        total = len(text.encode("utf-8"))
        while pages_used < SEARCH_MEMORY_MAX_PAGES and start < total:
            page = text_page(text, start, DEFAULT_PAGE_BYTES)
            pages_used += 1
            haystack = page["text"].lower()
            if not terms or any(term in haystack for term in terms):
                found = True
                chunks.append(
                    f"{handle['alias']} tier={tier} bytes "
                    f"{page['start_byte']}-{page['end_byte']} "
                    f"has_more={page['has_more']} sha256={handle['text_sha256']}:\n"
                    f"{page['text'][:800]}"
                )
                break
            if not page["has_more"]:
                break
            start = page["end_byte"]
        if pages_used >= SEARCH_MEMORY_MAX_PAGES:
            break
    return chunks, found


def search_memory(
    question: str,
    alias: str = "",
    *,
    scope: Optional[RuntimeHandleScope] = None,
    selected_archive: Optional[RuntimeHandleArchive] = None,
) -> str:
    selected_scope = scope or default_scope()
    store = selected_archive or archive()
    wanted = alias.strip()
    hot_by_alias = stored_handles(selected_scope)
    if wanted:
        hot_handles = [hot_by_alias[wanted]] if wanted in hot_by_alias else []
    else:
        hot_handles = list(hot_by_alias.values())
    archived = store.list(selected_scope, wanted)
    record_event(
        {
            "kind": "search_memory",
            "scope_id": selected_scope.scope_id,
            "alias": wanted or None,
            "question": question[:300],
            "hot_handle_count": len(hot_handles),
            "archive_handle_count": len(archived),
        }
    )
    if not hot_handles and not archived:
        return (
            "search_memory: no matching offloaded handle. "
            "Available handles: (none). "
            "Use find_* workflow commands for live directory lists; "
            "do not ask search_memory to reconstruct a dumped table."
        )
    terms = [token.lower() for token in question.split() if len(token) >= 4][:8]
    chunks, found_hot = _search_handles(hot_handles, terms, tier="hot")
    used_sqlite_fallback = False
    if not found_hot:
        archived_only = [handle for handle in archived if handle["alias"] not in hot_by_alias]
        archive_chunks, found_archive = _search_handles(
            archived_only, terms, tier="sqlite"
        )
        chunks.extend(archive_chunks)
        used_sqlite_fallback = bool(archived_only)
        record_event(
            {
                "kind": "search_memory_sqlite_fallback",
                "scope_id": selected_scope.scope_id,
                "alias": wanted or None,
                "archive_candidates": len(archived_only),
                "found": found_archive,
            }
        )
    if not chunks:
        chunks.append(
            f"{wanted or 'requested handles'}: no page in the first "
            f"{SEARCH_MEMORY_MAX_PAGES} pages per tier matched the question terms."
        )
    return (
        "search_memory compact excerpts from offloaded handles "
        "(not a full table dump; use find_* for live lists; "
        f"sqlite_fallback={used_sqlite_fallback}):\n"
        + "\n---\n".join(chunks)
    )
