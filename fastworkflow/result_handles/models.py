"""Serializable values used by result declaration and paging."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from fastworkflow.result_handles.common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PAGE_SIZE,
    UNSORTED_OFFSET,
    ResultHandleError,
)
from fastworkflow.result_handles.common import canonical_json, digest

@dataclass(frozen=True)
class SourceDescriptor:
    """Everything needed to re-issue the producing query, and nothing callable.

    Six fields, and every one of them is something the FRAMEWORK reads. F1
    (fix-iq53.2.5) removed the nine that encoded one backend's findings —
    ``view``, ``params``, ``role``, ``extra``, ``ordering``, ``timeslot``,
    ``start_offset``, ``materialized``, ``count_only`` — because a framework
    type that names a SQL view, an offset origin and a snapshot pin is not a
    generic adapter boundary, it is one workflow's query object wearing one.
    Those values did not stop existing; they moved into ``state``, which the
    adapter owns and this package never interprets.

    ``resolver`` names a resolver registered in this process (see
    ``register_resolver``); the rest is JSON.

    **There is still no ``sort`` field and no snapshot pin, and now there is no
    field to put one in at all.** B0 (ido-gqv.6) established that the views C1
    pages have a stable, complete, repeatable default order with no timeslot
    sent, and that an explicit sort is what breaks offset paging: a sorted
    offset walk returned exactly ``total`` rows while 20 of 540 group members
    were never shown. The old descriptor spent two fields and two constructor
    rules refusing that (``ordering`` had to be ``UNSORTED_OFFSET``, citing
    ido-gqv.6 B0; ``timeslot`` had to be ``None``, citing ido-986.14.1). The
    rules are gone because the fields are gone — the framework cannot send an
    ordering it has no way to name. Naming one, refusing one, and recording
    that a read had no pin rather than leaving the question open are now the
    adapter's, in its own ``state`` and its own evidence.
    """

    resolver: str
    uid_field: str = ""
    label_fields: Sequence[str] = ()
    #: Rows to ask the adapter for in one batch callback.
    batch_size: int = DEFAULT_BATCH_SIZE
    #: Columns the literal filter may be mapped to, already verified against
    #: this source. Empty means literal filtering is unsupported for this
    #: handle — a filter sent without columns is silently ignored by the portal
    #: and returns the whole scope, so that pair must be impossible to emit.
    #:
    #: (fix-iq53.2.3) This stays a list of NAMES and does not collapse to a
    #: ``filterable`` boolean. The names are agent-visible: the page header
    #: prints ``filter_columns=`` from them (``rendering.py``), and a complete
    #: zero names the fields it searched rather than saying "the rendered
    #: rows". A boolean cannot reconstruct either, and recovering the names
    #: from ``state`` would mean inspecting the one field that must stay
    #: opaque. Column names are query vocabulary, not one backend's policy.
    filter_columns: Sequence[str] = ()
    #: The adapter's own query state, carried verbatim and NEVER inspected
    #: here. Everything F1 removed lives in here now, under whatever keys the
    #: adapter chooses: the view and its params, the role, the backend offset
    #: the producer's own first row came from, the rows it already
    #: materialised from that offset, the ordering policy, the snapshot pin,
    #: and whether the source can answer an independent count at all.
    state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resolver:
            raise ResultHandleError("a source descriptor must name a resolver")
        if int(self.batch_size) < 1:
            raise ResultHandleError("batch_size must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filter_columns"] = list(self.filter_columns)
        payload["label_fields"] = list(self.label_fields)
        payload["state"] = dict(self.state)
        return payload

    @property
    def digest(self) -> str:
        return digest(canonical_json(self.as_dict()))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceDescriptor":
        """This dataclass built from a loose mapping, unknown keys dropped.

        (F1) Not a rehydrator, and must not become one. Dropping unknown keys
        is right for a workflow handing ``declare`` a dict it assembled, and
        wrong for a descriptor read back out of a store: a payload written
        before F1 carries eight fields this class no longer has, they are
        dropped here, and ``as_dict`` then digests to something the stored
        ``descriptor_sha256`` will never match. Every verifier in both
        repositories re-digests the STORED JSON instead and is unaffected —
        ``store.py``'s redeclaration refusal, IDO's ``stored_descriptor`` and
        the offline evaluator's ``_restore_descriptor``. Keep it that way.
        """
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
