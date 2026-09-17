"""Answer-time rehydration: the extract step reads the evidence, not the pointers.

``ido-8ps.18``. The ReAct loop's trajectory is a *working* surface: compaction
swaps a 3 KB listing for a 250 B offload label, and a listing command shows one
bounded page of a result handle whose remaining rows live in the store. That is
what keeps the agent's peak prompt at ~37k tokens instead of the control's ~80k,
and ``ido-8ps.17`` measured it as a clear win on trajectory correctness.

The extract step is a different reader with a different job. It has no tools, it
runs once, and ``trajectory`` is its only evidence input -- so at answer time the
same compaction that helped the loop is what leaves the writer holding pointers.
``ido-8ps.17`` measured that too: 35 of 260 deliverable slots answered with "see
Observation O23" about a listing sitting in that attempt's own archive, and two
attempts in ten that wrote up work which never ran.

So immediately before the extract call, and only there, this module builds the
extractor's OWN copy of the trajectory with the evidence put back:

(a) an **offload label** becomes the raw archived observation it names, printed
    with its alias line and the context clause recorded for that alias, exactly
    as the agent first saw it (``ido-986.14.9`` / ``ido-8ps.13``);
(b) a **bounded listing observation** that declared a result handle keeps its own
    text and gains every stored row behind that handle, whole, as the producer
    rendered them (``ido-986.14.1`` / ``14.2``);
(c) a **fetch_result_page observation** is treated the same way through the
    listing it paged: the rows behind the page it served are appended to it.

Rules this module does not bend:

* **Nothing is chosen by a model and nothing is invented.** Every byte added here
  was produced by a command in this turn and stored under a digest.
* **The ReAct loop's own trajectory object is never touched.** ``rehydrate``
  works on a copy; the archive and the handle store are opened read-only.
* **The budget is a hard bound.** Most recent first, stop at the first
  replacement that would not fit, and say which aliases were left as pointers so
  the extractor can name them unresolved instead of guessing.

Unconditional since ``ido-pyw.1``: rehydration is what the extract step does,
for every workflow. The budget is derived from the model's context window
(``fastworkflow.context_budget``); ``FW_ANSWER_REHYDRATION_MAX_BYTES`` remains
as a tuning override.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from fastworkflow import context_budget
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.labels import (
    annotated_observation,
    is_offload_label,
    label_alias,
    printed_alias,
)
from fastworkflow.observation_offloading.state import (
    context_clause_of,
    default_scope,
    stored_handles,
)

logger = logging.getLogger(__name__)

#: The tuning override for the extraction byte budget, in UTF-8 bytes of the
#: whole extractor trajectory. The budget itself is a fraction of the model's
#: context window (``context_budget.ANSWER_REHYDRATION``).
ANSWER_REHYDRATION_MAX_BYTES_ENV = context_budget.ANSWER_REHYDRATION.override_env
#: ~250 KB of UTF-8 at the reference window -- about the 80k-token answer-time
#: prompt the control cells of ``ido-8ps.17`` answered from. It is a ceiling,
#: not a target: a run whose evidence is smaller produces a smaller prompt.
DEFAULT_MAX_BYTES = context_budget.REFERENCE_ANSWER_REHYDRATION_MAX_BYTES
#: Below this a budget could not hold one page of evidence, so it is refused and
#: the derived budget stands rather than silently producing a pointer-only prompt.
MIN_MAX_BYTES = context_budget.ANSWER_REHYDRATION.floor

#: The key the drop line is appended under. Deliberately not an observation key:
#: it is a statement about the trajectory, not a tool result, and the extractor
#: must never read it as evidence about the workflow.
NOT_REHYDRATED_KEY = "answer_rehydration_note"
NOT_REHYDRATED_PREFIX = (
    "Not rehydrated for the answer (evidence exists under these observations): "
)

KIND_LABEL = "label"        # (a)
KIND_LISTING = "listing"    # (b)
KIND_PAGE = "page"          # (c)


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------

def max_bytes_from_env() -> int:
    """The effective extraction byte budget for this run.

    A fraction of the model's context window, unless
    ``FW_ANSWER_REHYDRATION_MAX_BYTES`` overrides it. See
    ``fastworkflow.context_budget``.
    """
    return context_budget.answer_rehydration_max_bytes()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

@dataclass
class RehydrationReport:
    """What the extractor's copy gained, and what it did not."""

    budget_bytes: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    counts: dict[str, int] = field(
        default_factory=lambda: {KIND_LABEL: 0, KIND_LISTING: 0, KIND_PAGE: 0}
    )
    rehydrated: list[dict[str, Any]] = field(default_factory=list)
    dropped_aliases: list[str] = field(default_factory=list)
    unresolved_aliases: list[str] = field(default_factory=list)
    note_line: str = ""
    stopped_on: str = ""

    @property
    def bytes_added(self) -> int:
        return self.bytes_after - self.bytes_before

    def as_event(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "bytes_added": self.bytes_added,
            "rehydrated_total": len(self.rehydrated),
            "rehydrated_labels": self.counts[KIND_LABEL],
            "rehydrated_listings": self.counts[KIND_LISTING],
            "rehydrated_pages": self.counts[KIND_PAGE],
            "dropped_aliases": list(self.dropped_aliases),
            "unresolved_aliases": list(self.unresolved_aliases),
            "stopped_on": self.stopped_on,
            "aliases": [
                {"alias": item["alias"], "kind": item["kind"],
                 "added_bytes": item["added_bytes"]}
                for item in self.rehydrated
            ],
        }


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def trajectory_bytes(trajectory: Mapping[str, Any]) -> int:
    """UTF-8 bytes of every value in the trajectory.

    The same quantity for the copy and the original, which is all a budget needs
    to be comparable. It is deliberately not the formatted prompt: the adapter's
    field framing is a constant per key and would make the budget depend on the
    adapter in force rather than on the evidence.
    """
    return sum(len(str(value).encode("utf-8")) for value in trajectory.values())


def _alias_ordinal(alias: str) -> int:
    try:
        return int(str(alias)[1:])
    except (TypeError, ValueError):
        return 0


def _step_indexes(trajectory: Mapping[str, Any]) -> list[int]:
    indexes: set[int] = set()
    for key in trajectory:
        name = str(key)
        if not name.startswith("observation_"):
            continue
        suffix = name.removeprefix("observation_")
        if suffix.isdigit():
            indexes.add(int(suffix))
    return sorted(indexes)


def _candidates(trajectory: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    """``(step_index, alias, text)`` for every aliased execute observation.

    Most recent first. The alias comes off the observation itself -- the printed
    A1 line, or the label's own alias -- so this never has to recompute execute
    ordinals or know how many steps the loop truncated away. An execute step with
    no alias on it (an error string, a refusal) is not a candidate: there is
    nothing stored to put back.
    """
    found: list[tuple[int, str, str]] = []
    for index in _step_indexes(trajectory):
        if str(trajectory.get(f"tool_name_{index}") or "") != "execute_workflow_query":
            continue
        text = trajectory.get(f"observation_{index}")
        if not isinstance(text, str) or not text:
            continue
        alias = label_alias(text) if is_offload_label(text) else printed_alias(text)
        if not alias:
            continue
        found.append((index, alias, text))
    found.reverse()
    return found


# ---------------------------------------------------------------------------
# (a) The archived observation behind a label
# ---------------------------------------------------------------------------

def archived_observation(
    alias: str,
    *,
    scope: RuntimeHandleScope,
    archive: RuntimeHandleArchive,
) -> Optional[str]:
    """The raw archived text for *alias*, hot cache first, then SQLite.

    The same resolution ``search_memory`` uses, for the same reason: both tiers
    hold the identical digest-verified bytes, and the hot one saves a read.
    """
    handle = stored_handles(scope).get(alias)
    if handle is None:
        try:
            handle = archive.get(scope, alias)
        except Exception:  # noqa: BLE001 - a read failure is a miss, never a turn failure
            logger.debug("answer rehydration could not read %s", alias, exc_info=True)
            return None
    if handle is None:
        return None
    text = handle.get("text")
    return text if isinstance(text, str) else None


def rehydrated_label(
    alias: str,
    *,
    scope: RuntimeHandleScope,
    archive: RuntimeHandleArchive,
) -> Optional[str]:
    """The label's observation, re-printed exactly as the agent first saw it.

    The archive stores the command response WITHOUT the presentation line
    (``ido-986.14.9``), so the line is rebuilt here from the alias and the
    context clause recorded for it at dispatch (``ido-8ps.13``). An alias with no
    recorded clause prints the plain A1 line: the clause is presentation and its
    absence is never guessed at.

    ``annotated_observation`` joins the two, exactly as the compaction hook did
    when the step completed, so a response whose own first line is shaped like
    a handle line is quoted here too and reads back as the same response
    (``ido-cku``).
    """
    text = archived_observation(alias, scope=scope, archive=archive)
    if text is None:
        return None
    return annotated_observation(alias, context_clause_of(scope, alias) or "", text)


# ---------------------------------------------------------------------------
# (b) and (c) The stored rows behind a result handle
# ---------------------------------------------------------------------------

def _page_query_scopes(store: Any, scope: RuntimeHandleScope, alias: str) -> list[str]:
    """Every traversal stored for *alias*: the base one first, then filtered."""
    lister = getattr(store, "list_page_query_scopes", None)
    if lister is None:
        return [""]
    scopes = [str(value) for value in lister(scope, alias=alias)]
    base = [value for value in scopes if not value]
    return base + sorted(value for value in scopes if value)


def _traversal_rows(store: Any, scope: RuntimeHandleScope, alias: str,
                    query_scope: str) -> tuple[list[str], int]:
    """Distinct rows of one traversal, first-seen order, and the pages read.

    Distinct uids in first-seen order is exactly what ``_walk_records`` serves
    the agent, so a duplicate page cannot make the answer's copy of the evidence
    disagree with the copy the pager showed.
    """
    rows: list[str] = []
    seen: set[str] = set()
    pages = store.list_pages(scope, alias=alias, query_scope=query_scope)
    for page in pages:
        for entry in (page.get("record") or {}).get("records") or []:
            uid = str(entry.get("uid") or "")
            line = str(entry.get("line") or "")
            if not line:
                continue
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            rows.append(line)
    return rows, len(pages)


def stored_rows_block(
    listing_alias: str,
    *,
    scope: RuntimeHandleScope,
    store: Any,
    shown_for: str = "",
) -> str:
    """Every stored row behind *listing_alias*, whole, as its producer rendered it.

    One block per traversal: the base (unfiltered) walk first, then each filtered
    traversal under the opaque query scope its cursor carried. The filter literal
    is NOT reconstructed here -- the store keeps a digest of it, not its text, and
    the page observation that ran it already prints ``filter="..."`` in its own
    header, which is in the trajectory beside this block.
    """
    parts: list[str] = []
    for query_scope in _page_query_scopes(store, scope, listing_alias):
        rows, pages = _traversal_rows(store, scope, listing_alias, query_scope)
        if not rows:
            continue
        traversal = (
            "base traversal" if not query_scope
            else "filtered traversal %s" % query_scope
        )
        heading = (
            "[answer-time rehydration] stored rows behind result_handle=%s, %s: "
            "%d rows from %d stored page(s), whole and unabridged%s"
            % (listing_alias, traversal, len(rows), pages,
               "" if not shown_for or shown_for == listing_alias
               else " (shown here under %s, which paged it)" % shown_for)
        )
        parts.append("\n".join([heading, *rows]))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def rehydrate(
    trajectory: Mapping[str, Any],
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    handle_store: Any = None,
    budget: Optional[int] = None,
) -> tuple[dict[str, Any], RehydrationReport]:
    """The extractor's copy of *trajectory*, with the evidence behind it put back.

    Returns ``(trajectory_copy, report)``. The input mapping is never mutated:
    the copy is what the extract call receives, so the ReAct loop keeps the
    trajectory it ran on and the turn record is unchanged.

    Order is most recent first, and the walk STOPS at the first replacement that
    would take the copy over ``budget``. Everything older than that stop stays as
    it was -- a label or a bounded page -- and every alias left that way is named
    in one deterministic line appended to the copy, so the extractor can report
    those slots as unresolved instead of inventing them.
    """
    from fastworkflow import result_handles

    selected_scope = scope or default_scope()
    if archive is None:
        from fastworkflow.observation_offloading import state as offload_state

        archive = offload_state.archive()
    if handle_store is None:
        handle_store = result_handles.store()
    budget_bytes = int(budget if budget is not None else max_bytes_from_env())

    copy: dict[str, Any] = dict(trajectory)
    report = RehydrationReport(budget_bytes=budget_bytes)
    report.bytes_before = trajectory_bytes(trajectory)
    used = report.bytes_before
    seen_blocks: set[str] = set()
    candidates = _candidates(trajectory)
    stopped = False

    for position, (index, alias, text) in enumerate(candidates):
        if stopped:
            report.dropped_aliases.append(alias)
            continue
        kind = ""
        listing_alias = ""
        replacement: Optional[str] = None
        if is_offload_label(text):
            kind = KIND_LABEL
            replacement = rehydrated_label(
                alias, scope=selected_scope, archive=archive
            )
            if replacement is None:
                report.unresolved_aliases.append(alias)
                continue
        else:
            declaration = _declaration(handle_store, selected_scope, alias)
            if declaration is None:
                continue
            parent = str(declaration.get("parent_alias") or "")
            kind = KIND_PAGE if parent else KIND_LISTING
            listing_alias = parent or alias
            if listing_alias in seen_blocks:
                # A later (more recent) observation already carries this
                # handle's rows in full. Repeating them would spend the budget
                # on bytes the extractor is already holding.
                continue
            block = stored_rows_block(
                listing_alias, scope=selected_scope, store=handle_store,
                shown_for=alias,
            )
            if not block:
                continue
            replacement = text + "\n" + block

        added = (len(replacement.encode("utf-8")) - len(text.encode("utf-8")))
        if used + added > budget_bytes:
            stopped = True
            report.stopped_on = alias
            report.dropped_aliases.append(alias)
            continue
        copy[f"observation_{index}"] = replacement
        used += added
        report.counts[kind] += 1
        if listing_alias:
            seen_blocks.add(listing_alias)
        report.rehydrated.append({
            "alias": alias,
            "kind": kind,
            "listing_alias": listing_alias,
            "step_index": index,
            "recency_rank": position,
            "added_bytes": added,
            "utf8_bytes": len(replacement.encode("utf-8")),
        })

    if report.dropped_aliases:
        report.dropped_aliases.sort(key=_alias_ordinal)
        report.note_line = NOT_REHYDRATED_PREFIX + ", ".join(report.dropped_aliases)
        copy[NOT_REHYDRATED_KEY] = report.note_line
    report.bytes_after = trajectory_bytes(copy)
    return copy, report


def _declaration(store: Any, scope: RuntimeHandleScope, alias: str) -> Optional[dict]:
    """The stored result-handle declaration for *alias*, or None.

    A miss is the normal case -- most execute observations are not listings -- so
    it is a return value, not an exception, and a store that cannot be read at
    all leaves every observation exactly as it stands.
    """
    try:
        return store.get_declaration(scope, alias)
    except Exception:  # noqa: BLE001
        logger.debug(
            "answer rehydration could not read the declaration for %s",
            alias, exc_info=True,
        )
        return None


__all__ = [
    "ANSWER_REHYDRATION_MAX_BYTES_ENV",
    "DEFAULT_MAX_BYTES",
    "KIND_LABEL",
    "KIND_LISTING",
    "KIND_PAGE",
    "MIN_MAX_BYTES",
    "NOT_REHYDRATED_KEY",
    "NOT_REHYDRATED_PREFIX",
    "RehydrationReport",
    "archived_observation",
    "max_bytes_from_env",
    "rehydrate",
    "rehydrated_label",
    "stored_rows_block",
    "trajectory_bytes",
]
