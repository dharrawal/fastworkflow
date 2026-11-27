"""The span tree of recorded turns, for asserting where a span sits (bd fix-qjne.34).

A recording sink receives a span more than once: a span emitted when it opens (``emit_open``) is emitted again when
it closes, and a stage or part span with a deterministic id is opened again when its turn resumes. ``spans(sink)``
reads them as the store does: one span per id, the first start and the last of everything else.

``tree(sink)`` is the nested ``(name, attributes, children)`` of each root, in start order; ``path_of(span, sink)`` is
the names from the root down to *span* (``["fw.turn", "fw.resolve", "fw.resolve.part", "fw.intent"]``);
``same_name_parents(sink)`` lists the spans whose parent has their own name (there should be none).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional

from fastworkflow.observability import tracing


@dataclass
class Node:
    name: str
    attributes: dict[str, Any]
    span: tracing.Span
    children: list["Node"] = field(default_factory=list)


def spans(sink: Any) -> list[tracing.Span]:
    """The sink's spans, one per id (the store's upsert: the first start, the rest from the last emission), in order of
    first emission."""
    merged: dict[str, tracing.Span] = {}
    for span in getattr(sink, "spans", sink):
        first = merged.get(span.span_id)
        if first is None:
            merged[span.span_id] = replace(span, attributes=dict(span.attributes))
        else:
            start = first.start_ns or span.start_ns
            parent = first.parent_span_id
            merged[span.span_id] = replace(span, attributes=dict(span.attributes), start_ns=start,
                                           parent_span_id=parent)
    return list(merged.values())


def by_id(sink: Any) -> dict[str, tracing.Span]:
    return {span.span_id: span for span in spans(sink)}


def tree(sink: Any) -> list[Node]:
    """The roots (spans whose parent is not recorded), each with its children in start order."""
    recorded = spans(sink)
    nodes = {span.span_id: Node(span.name, span.attributes, span) for span in recorded}
    roots: list[Node] = []
    for span in sorted(recorded, key=lambda each: each.start_ns):
        parent = nodes.get(span.parent_span_id) if span.parent_span_id else None
        (parent.children if parent is not None else roots).append(nodes[span.span_id])
    return roots


def path_of(span: tracing.Span, sink: Any) -> list[str]:
    """The names from the root down to *span*, *span*'s own last."""
    index = by_id(sink)
    names: list[str] = []
    current: Optional[tracing.Span] = index.get(span.span_id, span)
    seen: set[str] = set()
    while current is not None and current.span_id not in seen:
        seen.add(current.span_id)
        names.append(current.name)
        current = index.get(current.parent_span_id) if current.parent_span_id else None
    return list(reversed(names))


def named(sink: Any, name: str) -> list[tracing.Span]:
    """The spans named *name*, one per id."""
    return [span for span in spans(sink) if span.name == name]


def children_of(span: tracing.Span, sink: Any, name: Optional[str] = None) -> list[tracing.Span]:
    """*span*'s direct children (named *name* when given), in start order."""
    found = [each for each in spans(sink) if each.parent_span_id == span.span_id
             and (name is None or each.name == name)]
    return sorted(found, key=lambda each: each.start_ns)


def ancestors(span: tracing.Span, sink: Any) -> list[tracing.Span]:
    """*span*'s ancestors, the nearest first."""
    index = by_id(sink)
    out: list[tracing.Span] = []
    current = index.get(span.parent_span_id) if span.parent_span_id else None
    while current is not None and current not in out:
        out.append(current)
        current = index.get(current.parent_span_id) if current.parent_span_id else None
    return out


def same_name_parents(sink: Any) -> list[tuple[str, str]]:
    """``(name, span_id)`` of every span whose parent has the same name."""
    index = by_id(sink)
    return [(span.name, span.span_id) for span in index.values()
            if span.parent_span_id in index and index[span.parent_span_id].name == span.name]


def render(nodes: Iterable[Node], depth: int = 0) -> str:
    """The tree as indented names, for an assertion message."""
    lines = []
    for node in nodes:
        lines.append("  " * depth + node.name)
        if node.children:
            lines.append(render(node.children, depth + 1))
    return "\n".join(lines)
