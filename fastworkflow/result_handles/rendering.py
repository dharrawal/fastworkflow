"""Result-page presentation and byte-bounded rendering."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

@dataclass
class ResultPage:
    """One rendered page of a stored handle, and the truth about its coverage.

    Every count is about the query that was actually run: with a filter,
    ``matched`` is the filtered population and ``total`` is still the whole
    relation, so a page can never present a filtered count as a population or a
    partial walk as a complete one.
    """

    handle: str
    kind: str
    summary: str
    rows: list[str]
    matched: int
    total: int
    materialized: int
    source_complete: bool
    matched_complete: bool
    continuation: str
    incomplete_reason: Optional[str]
    next_cursor: Optional[str]
    outcome: str
    position: int
    page_index: int
    page_alias: Optional[str] = None
    parent_alias: Optional[str] = None
    literal: Optional[str] = None
    filter_columns: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    observation: str = ""

    def as_observation(self) -> str:
        return self.observation

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["filter_columns"] = list(self.filter_columns)
        payload["warnings"] = list(self.warnings)
        payload["notes"] = list(self.notes)
        return payload


def _header(page: "ResultPage") -> str:
    parts = [
        "result_handle=%s" % page.handle,
        "page %d" % page.page_index,
        (
            "rows %d-%d of %d"
            % (page.position + 1, page.position + len(page.rows), page.matched)
            if page.rows
            else "rows 0 of %d" % page.matched
        ),
        "matched=%d" % page.matched,
        "materialized=%d" % page.materialized,
        "total=%d" % page.total,
        "source_complete=%s" % str(page.source_complete).lower(),
        "matched_complete=%s" % str(page.matched_complete).lower(),
        "continuation=%s" % page.continuation,
        "outcome=%s" % page.outcome,
        "has_more=%s" % str(page.next_cursor is not None).lower(),
    ]
    if page.literal:
        parts.insert(1, 'filter="%s"' % page.literal)
        if page.filter_columns:
            parts.insert(2, "filter_columns=%s" % ",".join(page.filter_columns))
    if page.next_cursor:
        parts.append("next_cursor=%s" % page.next_cursor)
    if page.incomplete_reason:
        parts.append("incomplete_reason=%s" % page.incomplete_reason)
    return " ".join(parts)


def _fixed_lines(page: "ResultPage") -> list[str]:
    """Everything above the rows: the header, the producer's summary, the notes."""
    lines = [_header(page)]
    if page.summary:
        lines.append(page.summary)
    lines.extend(page.notes)
    return lines


def _pack(page: "ResultPage", *, budget_bytes: int) -> tuple[list[str], bool]:
    """The whole rows that fit, and whether the page is over its budget.

    Rows are never split and never skipped: the packer stops at the last row
    that fits and the next cursor starts at the first one that did not. A single
    row wider than the whole budget is emitted whole — cutting it would invent a
    row that was never returned, dropping it would lose evidence — and the
    overage is reported instead.

    This is the ONLY place that decides how many rows a page shows. The cursor
    is computed from its answer and the text is assembled from the same list, so
    a header that grows after packing can never silently swallow a row.

    (ido-56z, F26) The overage is measured on the OBSERVATION, not on the rows.
    Gating it on there being rows made a page with none of them unable to report
    an overage it certainly had: a zero-row page carrying a 200 KB filter
    literal in its header rendered 400,382 bytes against a 3,072-byte budget
    with ``warnings=[]``. Everything above the rows costs the prompt exactly
    what a row costs it.
    """
    fixed = _fixed_lines(page)
    overhead = sum(len(line.encode("utf-8")) + 1 for line in fixed)
    overhead += sum(len(warning.encode("utf-8")) + 1 for warning in page.warnings)
    shown: list[str] = []
    used = 0
    for line in page.rows:
        cost = len(line.encode("utf-8")) + 1
        if shown and overhead + used + cost > budget_bytes:
            break
        shown.append(line)
        used += cost
    return shown, overhead + used > budget_bytes


def _assemble(
    page: "ResultPage", shown: Sequence[str], *, over_budget: bool, budget_bytes: int
) -> tuple[str, tuple[str, ...]]:
    warnings = list(page.warnings)
    if over_budget:
        warnings.append(
            "Warning: this page is over its %d-byte observation budget. A row is "
            "never cut or skipped, so it is shown whole and the overage is "
            "reported instead." % budget_bytes
        )
    return "\n".join(_fixed_lines(page) + list(shown) + warnings), tuple(warnings)
