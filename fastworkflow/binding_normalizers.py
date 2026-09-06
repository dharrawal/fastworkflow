"""Versioned, deterministic normalizers for typed plan bindings.

Normalizers are deliberately a closed registry. A skill may name one by id,
but loading arbitrary Python from skill metadata would turn deterministic plan
compilation into code execution and make a recorded normalizer id impossible to
replay reliably.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Mapping, Optional

ENTITY_TYPE_NORMALIZER_V1 = "entity-type@1"
CONTROL_ALIAS_RESOLVER_V1 = "control-alias@1"

_ENTITY_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "identity": (
        "identity",
        "identities",
        "person",
        "people",
        "employee",
        "employees",
        "user",
        "users",
        "human",
        "humans",
    ),
    "account": (
        "account",
        "accounts",
        "login",
        "logins",
        "credential",
        "credentials",
    ),
    "organization": (
        "organization",
        "organizations",
        "organisation",
        "organisations",
        "org",
        "orgs",
        "business unit",
        "business units",
        "department",
        "departments",
        "unit",
        "units",
    ),
    "application": (
        "application",
        "applications",
        "system",
        "systems",
        "tool",
        "tools",
        "platform",
        "platforms",
    ),
    "repository": (
        "repository",
        "repositories",
        "connector",
        "connectors",
        "source feed",
        "source feeds",
        "feed",
        "feeds",
    ),
    "permission": (
        "permission",
        "permissions",
        "entitlement",
        "entitlements",
        "right",
        "rights",
        "privilege",
        "privileges",
    ),
    "group": (
        "group",
        "groups",
        "collection",
        "collections",
        "ad group",
        "ad groups",
    ),
}


def _normalized_token(value: str) -> str:
    return " ".join(
        str(value).strip(" \t\r\n'\"`.,:;!?()[]{}").casefold().split()
    )


def _normalize_entity_type_v1(value: str) -> Optional[str]:
    token = _normalized_token(value)
    for canonical, aliases in _ENTITY_TYPE_ALIASES.items():
        if token in aliases:
            return canonical
    return None


_NORMALIZERS: dict[str, Callable[[str], Optional[str]]] = {
    ENTITY_TYPE_NORMALIZER_V1: _normalize_entity_type_v1,
}


@dataclass(frozen=True)
class AliasCandidate:
    """One canonical label and handle offered to an alias resolver."""

    handle: str
    label: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class AliasResolution:
    """Replayable result of resolving one operator-authored alias."""

    resolver_id: str
    status: Literal["resolved", "ambiguous", "unresolved"]
    query: str
    handle: Optional[str] = None
    label: Optional[str] = None
    matched_by: Optional[str] = None
    candidate_handles: tuple[str, ...] = ()


# These are lexical equivalences, not tenant aliases. Tenant-specific labels and
# handles always arrive as candidates at runtime; adding one here would make a
# recorded resolver version silently depend on a customer's catalogue.
_CONTROL_TOKEN_EQUIVALENTS = {
    "accounts": "account",
    "contractors": "contractor",
    "departed": "left",
    "departure": "end",
    "departures": "end",
    "enabled": "active",
    "ended": "end",
    "ending": "end",
    "expired": "past",
    "identities": "identity",
    "managers": "manager",
    "passed": "past",
    "vendors": "vendor",
    "workers": "worker",
}

_CONTROL_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "has",
        "have",
        "is",
        "of",
        "or",
        "own",
        "owning",
        "set",
        "still",
        "the",
        "their",
        "that",
        "to",
        "who",
        "whom",
        "whose",
        "with",
    }
)


def _alias_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _control_signature(value: str) -> frozenset[str]:
    tokens = []
    for raw in re.findall(r"[a-z0-9]+", str(value).casefold()):
        token = _CONTROL_TOKEN_EQUIVALENTS.get(raw, raw)
        if token and token not in _CONTROL_STOP_TOKENS:
            tokens.append(token)
    return frozenset(tokens)


def _coerce_alias_candidates(
    candidates: Iterable[AliasCandidate | Mapping[str, object]],
) -> tuple[AliasCandidate, ...]:
    coerced: list[AliasCandidate] = []
    for item in candidates:
        if isinstance(item, AliasCandidate):
            candidate = item
        else:
            aliases = item.get("aliases") or ()
            if isinstance(aliases, str):
                aliases = (aliases,)
            candidate = AliasCandidate(
                handle=str(item.get("handle") or ""),
                label=str(item.get("label") or ""),
                aliases=tuple(str(alias) for alias in aliases),
            )
        if not candidate.handle or not candidate.label:
            raise ValueError("alias candidates require non-empty handle and label")
        coerced.append(candidate)
    return tuple(coerced)


def _resolved(
    query: str,
    candidate: AliasCandidate,
    *,
    matched_by: str,
) -> AliasResolution:
    return AliasResolution(
        resolver_id=CONTROL_ALIAS_RESOLVER_V1,
        status="resolved",
        query=query,
        handle=candidate.handle,
        label=candidate.label,
        matched_by=matched_by,
        candidate_handles=(candidate.handle,),
    )


def _resolve_control_alias_v1(
    query: str,
    candidates: Iterable[AliasCandidate | Mapping[str, object]],
) -> AliasResolution:
    options = _coerce_alias_candidates(candidates)
    exact_handle = [item for item in options if query == item.handle]
    if len(exact_handle) == 1:
        return _resolved(query, exact_handle[0], matched_by="exact-handle")

    folded_query = _alias_text(query)
    normalized_handles = {
        item.handle: item
        for item in options
        if folded_query == _alias_text(item.handle)
    }
    if len(normalized_handles) == 1:
        return _resolved(
            query,
            next(iter(normalized_handles.values())),
            matched_by="normalized-exact-handle",
        )
    if len(normalized_handles) > 1:
        return AliasResolution(
            resolver_id=CONTROL_ALIAS_RESOLVER_V1,
            status="ambiguous",
            query=query,
            matched_by="normalized-exact-handle",
            candidate_handles=tuple(sorted(normalized_handles)),
        )

    exact_text = [
        item
        for item in options
        if folded_query
        in {
            _alias_text(item.label),
            *(_alias_text(alias) for alias in item.aliases),
        }
    ]
    exact_by_handle = {item.handle: item for item in exact_text}
    if len(exact_by_handle) == 1:
        return _resolved(
            query,
            next(iter(exact_by_handle.values())),
            matched_by="normalized-exact-label-or-alias",
        )
    if len(exact_by_handle) > 1:
        return AliasResolution(
            resolver_id=CONTROL_ALIAS_RESOLVER_V1,
            status="ambiguous",
            query=query,
            matched_by="normalized-exact-label-or-alias",
            candidate_handles=tuple(sorted(exact_by_handle)),
        )

    query_signature = _control_signature(query)
    semantic_matches: list[tuple[int, AliasCandidate]] = []
    for item in options:
        signatures = tuple(
            signature
            for signature in (
                _control_signature(item.label),
                *(_control_signature(alias) for alias in item.aliases),
            )
            if signature
        )
        matching = [
            signature
            for signature in signatures
            if len(signature) >= 3 and signature <= query_signature
        ]
        if matching:
            semantic_matches.append(
                (max(len(signature) for signature in matching), item)
            )

    if semantic_matches:
        maximum_specificity = max(score for score, _item in semantic_matches)
        finalists = {
            item.handle: item
            for score, item in semantic_matches
            if score == maximum_specificity
        }
        if len(finalists) == 1:
            return _resolved(
                query,
                next(iter(finalists.values())),
                matched_by="unique-maximal-semantic-coverage",
            )
        return AliasResolution(
            resolver_id=CONTROL_ALIAS_RESOLVER_V1,
            status="ambiguous",
            query=query,
            matched_by="unique-maximal-semantic-coverage",
            candidate_handles=tuple(sorted(finalists)),
        )

    return AliasResolution(
        resolver_id=CONTROL_ALIAS_RESOLVER_V1,
        status="unresolved",
        query=query,
    )


_ALIAS_RESOLVERS = {
    CONTROL_ALIAS_RESOLVER_V1: _resolve_control_alias_v1,
}

_ALIAS_RESOLVER_HANDLE_FIELDS = {
    CONTROL_ALIAS_RESOLVER_V1: "control_codes",
}


def is_registered_normalizer(normalizer_id: str) -> bool:
    """Whether ``normalizer_id`` names a replayable implementation."""
    return normalizer_id in _NORMALIZERS


def normalize_binding_value(normalizer_id: str, value: str) -> Optional[str]:
    """Return the canonical value, or ``None`` when the input is not accepted."""
    normalizer = _NORMALIZERS.get(normalizer_id)
    if normalizer is None:
        raise ValueError(f"unknown binding normalizer {normalizer_id!r}")
    return normalizer(value)


def aliases_for_normalized_value(
    normalizer_id: str, canonical_value: str
) -> tuple[str, ...]:
    """Return source phrases accepted for one canonical value.

    This is primarily useful to deterministic fixtures that must provide a
    real source span rather than asserting that a canonical enum came from the
    utterance without showing where.
    """
    if normalizer_id != ENTITY_TYPE_NORMALIZER_V1:
        raise ValueError(f"normalizer {normalizer_id!r} does not expose aliases")
    return _ENTITY_TYPE_ALIASES.get(canonical_value, ())


def is_registered_alias_resolver(resolver_id: str) -> bool:
    """Whether ``resolver_id`` names a replayable candidate resolver."""
    return resolver_id in _ALIAS_RESOLVERS


def alias_resolver_handle_field(resolver_id: str) -> str:
    """Return the aligned artifact handle field consumed by a resolver."""
    try:
        return _ALIAS_RESOLVER_HANDLE_FIELDS[resolver_id]
    except KeyError as exc:
        raise ValueError(f"unknown binding alias resolver {resolver_id!r}") from exc


def resolve_binding_alias(
    resolver_id: str,
    query: str,
    candidates: Iterable[AliasCandidate | Mapping[str, object]],
) -> AliasResolution:
    """Resolve one alias deterministically, preserving ambiguity as a result."""
    resolver = _ALIAS_RESOLVERS.get(resolver_id)
    if resolver is None:
        raise ValueError(f"unknown binding alias resolver {resolver_id!r}")
    return resolver(query, candidates)
