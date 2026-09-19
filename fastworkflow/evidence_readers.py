"""Shared, read-only evidence contracts used by answer-time consumers.

This module writes no state and does not decide whether evidence is complete or
supports an answer. It joins observation, subject, and result-page records and
owns the existing mechanical qualified-name matching shared by coverage and
attribution. Callers still provide their established normalization and echo
handling, so moving the reader does not change retrieval or answer policy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from fastworkflow.observation_offloading.labels import is_search_answer_key
from fastworkflow.observation_offloading.state import context_clause_of, stored_handles

logger = logging.getLogger(__name__)

#: The shortest last segment of a qualified name that may stand for the whole
#: name. Short tails ("Officer", "1") are ambiguous in any workflow.
#:
#: This module is the single definition. ``answer_attribution`` imports both of
#: these and re-exports ``MIN_SEGMENT_CHARS`` on its released surface; it does
#: not restate them, because the segmentation rule that actually runs is the one
#: below and a second copy could be changed without changing it.
MIN_SEGMENT_CHARS = 8
_SEGMENT_SEPARATORS = "_/:\\"


@dataclass(frozen=True)
class Observation:
    """One immutable observation and the subject clause recorded for it."""

    alias: str
    clause: str
    text: str


def _segment_tail(entity: Any) -> str:
    """The last segment of *entity* when it is long enough to stand alone."""
    tail = entity.key
    for separator in _SEGMENT_SEPARATORS:
        tail = tail.rsplit(separator, 1)[-1]
    tail = tail.strip()
    if tail and tail != entity.key and len(tail) >= MIN_SEGMENT_CHARS:
        return tail
    return ""


def match_forms(entity: Any, *, allow_segments: bool = True,
                among: Iterable[Any] = ()) -> list[str]:
    """Return the existing full-name and unambiguous-tail match forms."""
    forms = [entity.key]
    if not allow_segments:
        return forms
    tail = _segment_tail(entity)
    if not tail:
        return forms
    if any(other.key != entity.key and tail in other.key for other in among):
        return forms
    return [*forms, tail]


def match_forms_index(entities: Iterable[Any], *, allow_segments: bool = True
                      ) -> dict[str, list[str]]:
    items = list(entities)
    return {
        entity.key: match_forms(entity, allow_segments=allow_segments, among=items)
        for entity in items
    }


def writes(text: str, forms: list[str]) -> bool:
    """Apply the existing qualified-tail containment rule."""
    for index, form in enumerate(forms):
        if not form:
            continue
        at = text.find(form)
        while at >= 0:
            if index == 0 or at == 0 or text[at - 1] not in _SEGMENT_SEPARATORS:
                return True
            at = text.find(form, at + 1)
    return False


def observations(
    *,
    scope: Any = None,
    archive: Any = None,
    handle_store: Any = None,
    strip_alias_line: Callable[[str], str],
    stored_rows_block: Callable[..., str],
    drop_zero_match_echo: Callable[[str, Mapping[str, Any] | None], str],
    normalise: Callable[[str], str],
) -> list[Observation]:
    """Read one evidence value per alias from the existing three stores."""
    from fastworkflow import result_handles
    from fastworkflow.observation_offloading import state as offload_state

    scope = scope or offload_state.default_scope()
    if archive is None:
        archive = offload_state.archive()
    if handle_store is None:
        try:
            handle_store = result_handles.store()
        except Exception:  # noqa: BLE001 - missing result storage omits only rows
            handle_store = None
    try:
        rows = archive.list(scope)
    except Exception:  # noqa: BLE001 - an unreadable source contributes no evidence
        logger.debug("evidence reader could not list the archive", exc_info=True)
        rows = []

    texts: dict[str, list[str]] = {}
    order: list[str] = []

    def take(alias: str, handle: Mapping[str, Any]) -> None:
        if not alias or is_search_answer_key(alias):
            return
        if alias not in texts:
            texts[alias] = []
            order.append(alias)
        texts[alias].append(strip_alias_line(str(handle.get("text") or "")))

    for handle in rows:
        take(str(handle.get("alias") or ""), handle)
    for alias, handle in stored_handles(scope).items():
        take(str(alias), handle or {})

    if handle_store is not None:
        for alias in order:
            try:
                declaration = handle_store.get_declaration(scope, alias)
            except Exception:  # noqa: BLE001
                continue
            if declaration is None:
                continue
            if str(declaration.get("parent_alias") or ""):
                # (ido-3f8) A filtered page that carried no rows retrieved
                # nothing, so its own text may not be the reason its literal
                # looks retrieved here either. Same marker, same removal, one
                # observation at a time.
                texts[alias] = [
                    drop_zero_match_echo(chunk, declaration)
                    for chunk in texts[alias]
                ]
                continue
            try:
                texts[alias].append(
                    stored_rows_block(alias, scope=scope, store=handle_store)
                )
            except Exception:  # noqa: BLE001
                logger.debug("evidence reader could not read rows for %s", alias,
                             exc_info=True)

    return [
        Observation(
            alias=alias,
            clause=normalise(
                context_clause_of(scope, alias, selected_archive=archive) or ""
            ),
            text=normalise("\n".join(texts[alias])),
        )
        for alias in order
    ]


def subject_evidence(
    entities: Iterable[Any],
    observations: Iterable[Observation],
    *,
    normalise: Callable[[str], str],
    allow_segments: bool = True,
) -> list[tuple[Any, list[Any]]]:
    """Return the other requested items present in each subject's observations."""
    items = list(entities)
    evidence = [
        Observation(item.alias, normalise(item.clause), normalise(item.text))
        for item in observations
    ]
    forms = match_forms_index(items, allow_segments=allow_segments)
    by_subject = {
        entity.key: [
            item for item in evidence
            if item.clause and writes(item.clause, forms[entity.key])
        ]
        for entity in items
    }
    return [
        (
            subject,
            [
                other for other in items
                if other.key != subject.key
                and any(writes(item.text, forms[other.key])
                        for item in by_subject[subject.key])
            ],
        )
        for subject in items
        if by_subject[subject.key]
    ]
