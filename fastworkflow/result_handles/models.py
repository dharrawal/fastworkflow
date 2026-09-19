"""Serializable values used by result declaration and paging."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from fastworkflow.result_handles.common import (
    DEFAULT_PAGE_SIZE,
    UNSORTED_OFFSET,
    ResultHandleError,
)
from fastworkflow.result_handles.common import canonical_json, digest

@dataclass(frozen=True)
class SourceDescriptor:
    """Everything needed to re-issue the producing query, and nothing callable.

    ``resolver`` names a resolver registered in this process (see
    ``register_resolver``); the rest is JSON. There is deliberately no ``sort``
    field and no ``timeslot`` value other than ``None``: B0 established that
    the views C1 pages have a stable, complete, repeatable default order with
    no timeslot sent, and that an explicit sort is what breaks offset paging.
    ``timeslot`` is carried explicitly as ``None`` so evidence records that the
    read had no pin rather than leaving the question open.
    """

    resolver: str
    view: str
    params: Mapping[str, Any] = field(default_factory=dict)
    #: Columns the filter may be mapped to, already verified against this view.
    #: Empty means literal filtering is unsupported for this handle — a filter
    #: sent without columns is silently ignored by the portal and returns the
    #: whole scope, so that pair must be impossible to emit.
    filter_columns: Sequence[str] = ()
    uid_field: str = ""
    label_fields: Sequence[str] = ()
    page_size: int = DEFAULT_PAGE_SIZE
    ordering: str = UNSORTED_OFFSET
    #: Backend offset the producer's own first row came from.
    start_offset: int = 0
    #: Rows the producer already materialised from ``start_offset``. The walk
    #: continues at ``start_offset + materialized``.
    materialized: int = 0
    timeslot: None = None
    role: Optional[str] = None
    count_only: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resolver:
            raise ResultHandleError("a source descriptor must name a resolver")
        if self.ordering != UNSORTED_OFFSET:
            raise ResultHandleError(
                "ordering %r is not available: C1 walks offsets in the view's "
                "default order only (ido-gqv.6 B0)" % (self.ordering,)
            )
        if self.timeslot is not None:
            raise ResultHandleError(
                "no timeslot pin exists for these views; the descriptor records "
                "timeslot=None (ido-986.14.1)"
            )
        if int(self.page_size) < 1:
            raise ResultHandleError("page_size must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filter_columns"] = list(self.filter_columns)
        payload["label_fields"] = list(self.label_fields)
        payload["params"] = dict(self.params)
        payload["extra"] = dict(self.extra)
        return payload

    @property
    def digest(self) -> str:
        return digest(canonical_json(self.as_dict()))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceDescriptor":
        known = {key: payload[key] for key in payload if key in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ResultHandleSpec:
    """What a producing command declares about the listing it just rendered.

    The field names are the ones the producing command already uses for its own
    rendering, so a workflow declares what it showed rather than translating it.
    ``items`` are the rendered ``uid  label`` lines exactly as the response
    carried them: a literal filter has to be able to find a name in the row the
    agent read.
    """

    kind: str
    summary: str = ""
    items: Sequence[str] = ()
    ordering: str = UNSORTED_OFFSET
    total: int = 0
    source_complete: bool = True
    page_size: int = DEFAULT_PAGE_SIZE
    classification: str = "user-text"
    presentation: bool = True
    filters: Mapping[str, str] = field(default_factory=dict)
