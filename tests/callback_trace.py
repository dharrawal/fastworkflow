"""F7 (fix-iq53.2.12): the callback tracer the boundary design was measured with.

Revision 4 §2 of the adapter-boundary design settled root's second demand --
"real callback/backend-call traces for normal, filtered, exhausted and capped
walks, before equivalent latency/work is accepted" -- with a harness that drove
the real ``fetch_page`` through the real ``FakePortal`` fixture and counted what
came out. The review verified all nine rows of §2.1 against its raw JSON and
found every number faithful.

**That harness lived in ``/tmp``.** It is here now, ported to the contract it
helped settle, because a measurement nobody can re-run is an assertion, and the
numbers it produced are the evidence that the terminal callback preserved the
work profile ``_reconcile`` had. ``tests/test_result_handle_callback_budget.py``
turns those numbers into assertions; this module is what produces them, and
stays runnable on its own so a future change can be measured the same way::

    python -m tests.callback_trace            # the whole trace table, as JSON

What this measures, and what it cannot: the framework's callback sequence, its
counts, and where the terminal lands. Those are properties of ``paging.py``'s
control flow and do not depend on what sits behind the resolver. Wall-clock
latency, the portal-side cost of a terminal versus a batch read, and whether any
real view behaves unlike ``FakePortal`` are all out of reach offline and are
IDO's ido-0rk.2.2 (I6) to close with a counting wrapper on the live client.

**Notation.** ``B`` is a charged batch callback and ``T`` is the uncharged
terminal callback. Revision 4's tables write the terminal as ``C``, for the
``countOnly`` the framework used to make there itself; ``BBBC`` in the report and
``BBBT`` here are the same sequence, and the letter changed because the caller
did. Sequences read left to right within one fetch.
"""
from __future__ import annotations

import json
import os
import tempfile

from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.result_handles import paging as result_handles
from fastworkflow.result_handles import (
    ResultHandleSpec,
    ResultHandleStore,
    SourceDescriptor,
    TerminalRequest,
    declare,
    fetch_page,
    reset_result_handle_state,
)
from tests.test_result_handles import FakePortal, offset_of, portal_rows

RESOLVER = "fake-portal"


def scope(turn: str = "turn-1", *, task: str = "task-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id="channel", experiment_id="exp",
        task_id=task, attempt=1, turn_key=turn,
    )


class TracingPortal:
    """``FakePortal`` wrapped in a per-fetch call recorder.

    The wrapper is the whole method: it is registered as the resolver, so what
    it records is what the framework actually asked for, in the order it asked.
    It does not reimplement the walk and it does not simulate one -- a harness
    that modelled the control flow instead of driving it would measure its own
    model.

    (F2, fix-iq53.2.6) The two callbacks are told apart by TYPE. The original
    harness split them on ``request.count_only``, the mode flag that made one
    request type do two jobs; there is no flag to read now, which is the point.
    """

    def __init__(self, portal) -> None:
        self.portal = portal
        self.fetch_index = -1
        #: One entry per resolver invocation, flat, across every fetch.
        self.log: list[dict] = []
        self.errors = 0

    def begin_fetch(self) -> None:
        """Start a new fetch's slice of the log. Called before each fetch_page."""
        self.fetch_index += 1

    def __call__(self, request):
        terminal = isinstance(request, TerminalRequest)
        entry = {
            "fetch": self.fetch_index,
            "seq": len([r for r in self.log if r["fetch"] == self.fetch_index]),
            "kind": "terminal" if terminal else "batch",
            # The adapter's own resume point, unpacked the way the adapter
            # unpacks it. The framework neither sent nor read this number.
            "offset": offset_of(request),
            "limit": None if terminal else request.limit,
            "contains": request.contains,
            "distinct_uids": request.distinct_uids if terminal else None,
            "continuation": dict(request.continuation or {}) or None,
        }
        self.log.append(entry)
        try:
            return self.portal(request)
        except Exception:
            entry["raised"] = True
            self.errors += 1
            raise

    def per_fetch(self) -> list[dict]:
        """One row per fetch: counts, the sequence, and where the terminal landed."""
        out = []
        for index in range(self.fetch_index + 1):
            calls = [r for r in self.log if r["fetch"] == index]
            batches = [r for r in calls if r["kind"] == "batch"]
            terminals = [r for r in calls if r["kind"] == "terminal"]
            positions = [r["seq"] for r in terminals]
            out.append({
                "fetch": index,
                "callbacks": len(calls),
                "batches": len(batches),
                "terminals": len(terminals),
                "terminal_seq": positions,
                # None when no terminal fired, so "never fired" and "fired last"
                # are different answers rather than the same falsy one.
                "terminal_is_last": (positions == [len(calls) - 1]
                                     if positions else None),
                "sequence": "".join("T" if r["kind"] == "terminal" else "B"
                                    for r in calls),
                "offsets": [r["offset"] for r in batches],
            })
        return out


class Harness:
    """One declared handle over one ``FakePortal``, with its calls counted.

    The descriptor is the post-F1 six-field one: every portal fact -- the view,
    its params, the offset the producer's own first row came from, the rows it
    already materialised -- is in opaque ``state``, where the framework never
    reads it and the fixture's own ``offset_of`` does.
    """

    def __init__(self, rows, *, batch_size, total=None, materialized=0,
                 complete=False, count=True, lossy=False, fail_at=None,
                 filter_columns=("identity_displayname", "identity_surname"),
                 alias="O7", portal=None) -> None:
        reset_result_handle_state()
        self.temp = tempfile.TemporaryDirectory()
        self.store = ResultHandleStore(os.path.join(self.temp.name, "h.sqlite3"))
        self.rows = rows
        self.alias = alias
        self.portal = TracingPortal(
            portal if portal is not None
            else FakePortal(rows, count=count, lossy=lossy, fail_at=fail_at))
        result_handles.register_resolver(RESOLVER, self.portal)
        rendered = ["%s  %s" % (row["identity__id"], row["identity_displayname"])
                    for row in rows[:materialized]]
        population = len(rows) if total is None else total
        declare(
            ResultHandleSpec(
                kind="member", summary="%d member(s)." % population,
                items=rendered, total=population,
                source_complete=complete, page_size=batch_size),
            source=SourceDescriptor(
                resolver=RESOLVER,
                uid_field="identity__id",
                label_fields=("identity_displayname",),
                batch_size=batch_size,
                filter_columns=filter_columns,
                state={
                    "view": "ido_groupDetail_identity",
                    "params": {"scope": "f737245119f5ee6347e6f10cb569fe86"},
                    "materialized": materialized,
                },
            ),
            scope=scope(), selected_store=self.store, alias=alias,
        )

    def fetch(self, cursor=None, contains=None, *, budget_bytes=3_072):
        """One fetch, with its callbacks recorded under their own fetch index."""
        self.portal.begin_fetch()
        return fetch_page(self.alias, cursor, contains, scope=scope(),
                          selected_store=self.store, budget_bytes=budget_bytes)

    def walk(self, *, contains=None, budget_bytes=3_072, max_fetches=60):
        """Page to the end, returning every row served and every page served."""
        seen: list[str] = []
        cursor, pages = None, []
        for _ in range(max_fetches):
            page = self.fetch(cursor, contains, budget_bytes=budget_bytes)
            pages.append(page)
            seen.extend(page.rows)
            cursor = page.next_cursor
            if cursor is None:
                break
        return seen, pages

    def cold(self):
        """Drop every in-process walk and hot row, keeping the SAME store file.

        What a fresh process image sees. The stored batches and the stored
        verdict are all that survive, which is the property the ido-1r0 walk
        terminal exists to protect.
        """
        result_handles.unregister_resolver(RESOLVER)
        reset_result_handle_state()
        self.portal = TracingPortal(self.portal.portal)
        result_handles.register_resolver(RESOLVER, self.portal)
        return self.portal

    def close(self) -> None:
        result_handles.unregister_resolver(RESOLVER)
        reset_result_handle_state()
        self.temp.cleanup()


def summarise(name, harness, seen, pages, note="") -> dict:
    per = harness.portal.per_fetch()
    last = pages[-1]
    return {
        "shape": name,
        "note": note,
        "fetches": len(pages),
        "rows_served": len(seen),
        "distinct_rows": len(set(seen)),
        "total_callbacks": len(harness.portal.log),
        "total_batch_reads": sum(row["batches"] for row in per),
        "total_terminals": sum(row["terminals"] for row in per),
        "per_fetch": per,
        "sequences": [row["sequence"] for row in per],
        "final_source_complete": bool(last.source_complete),
        "final_incomplete_reason": last.incomplete_reason,
        "final_outcome": last.outcome,
        "final_continuation": last.continuation,
        "terminal_is_last_in_its_fetch": [
            row["terminal_is_last"] for row in per if row["terminals"]],
        "max_callbacks_per_fetch": max([row["callbacks"] for row in per] or [0]),
        "max_terminals_per_fetch": max([row["terminals"] for row in per] or [0]),
        "max_batches_per_fetch": max([row["batches"] for row in per] or [0]),
    }


# ---------------------------------------------------------------------------
# The shapes. Each builder returns one summary dict and cleans up after itself.
# ---------------------------------------------------------------------------

def shape_normal() -> dict:
    """540 rows, batch size 40, 3 KB budget, unfiltered, walked to completion."""
    harness = Harness(portal_rows(540), batch_size=40, materialized=40)
    try:
        seen, pages = harness.walk()
        return summarise("normal", harness, seen, pages,
                         "540 rows / batch_size 40 / 3KB observation budget, "
                         "unfiltered, walked to completion")
    finally:
        harness.close()


def shape_filtered() -> dict:
    """120 rows, a literal matching all of them, on its own query scope."""
    harness = Harness(portal_rows(120), batch_size=25, materialized=25,
                      total=120, alias="O3")
    try:
        seen, pages = harness.walk(contains="Cooper", budget_bytes=900)
        return summarise("filtered", harness, seen, pages,
                         "120 rows / batch_size 25 / contains='Cooper' matching "
                         "all 120, backend-filter plan on its own query_scope")
    finally:
        harness.close()


def shape_filtered_narrow() -> dict:
    """The same relation, a literal matching exactly one row."""
    harness = Harness(portal_rows(120), batch_size=25, materialized=25,
                      total=120, alias="O3")
    try:
        seen, pages = harness.walk(contains="Alan Cooper 0", budget_bytes=3_072)
        return summarise("filtered-narrow", harness, seen, pages,
                         "120 rows / contains='Alan Cooper 0' matching exactly 1")
    finally:
        harness.close()


def shape_exhausted() -> dict:
    """80 rows at batch size 40: an exact multiple, so the empty probe is owed.

    An offset walker cannot know its last full batch was last, so it offers a
    continuation after it and the framework reaches the empty batch by asking
    once more. This is the shape the retracted ``len(rows) < limit`` rule got
    wrong; F6's todo adapter answers it in two callbacks with lookahead, and
    this one cannot, which is the difference the boundary has to allow for.
    """
    harness = Harness(portal_rows(80), batch_size=40, total=80)
    try:
        seen, pages = harness.walk(budget_bytes=1_000_000)
        return summarise("exhausted", harness, seen, pages,
                         "80 rows / batch_size 40 (exact multiple): two full "
                         "batches, one empty terminal probe, then the terminal")
    finally:
        harness.close()


def shape_exhausted_partial_last() -> dict:
    """70 rows at batch size 40: a short last batch, and still an empty probe."""
    harness = Harness(portal_rows(70), batch_size=40, total=70)
    try:
        seen, pages = harness.walk(budget_bytes=1_000_000)
        return summarise("exhausted-partial-last", harness, seen, pages,
                         "70 rows / batch_size 40 (partial last batch): this "
                         "adapter still probes empty")
    finally:
        harness.close()


def shape_capped_call_limit() -> dict:
    """batch size 1 over 540 rows: the purse bites long before the end."""
    harness = Harness(portal_rows(540), batch_size=1, total=540)
    try:
        page = harness.fetch(budget_bytes=3_072)
        return summarise("capped-call-limit", harness, page.rows, [page],
                         "540 rows / batch_size 1: reaches "
                         "MAX_RESOLVER_CALLS_PER_FETCH=8 before exhaustion; "
                         "single fetch shown")
    finally:
        harness.close()


def shape_capped_byte_budget() -> dict:
    """540 rows and a 3 KB page: the packer stops the fill, not the purse."""
    harness = Harness(portal_rows(540), batch_size=40, materialized=40, total=540)
    try:
        page = harness.fetch(budget_bytes=3_072)
        return summarise("capped-byte-budget", harness, page.rows, [page],
                         "540 rows / batch_size 40 / 3KB budget, first fetch "
                         "only: stops on the packer's byte cap")
    finally:
        harness.close()


def shape_exhausted_mismatch() -> dict:
    """A lossy portal: 520 distinct rows walked while the count says 540."""
    harness = Harness(portal_rows(540), batch_size=40, total=540, lossy=True)
    try:
        seen, pages = harness.walk()
        return summarise("exhausted-mismatch", harness, seen, pages,
                         "540 rows, lossy portal: the walk yields 520 distinct "
                         "and the adapter's own count says 540 -> "
                         "countonly_mismatch")
    finally:
        harness.close()


def shape_exhausted_no_count() -> dict:
    """A portal that cannot answer an independent count at all."""
    harness = Harness(portal_rows(80), batch_size=40, total=80, count=False)
    try:
        seen, pages = harness.walk(budget_bytes=1_000_000)
        return summarise("exhausted-no-count", harness, seen, pages,
                         "80 rows, portal offers no independent count -> the "
                         "adapter answers countonly_unavailable")
    finally:
        harness.close()


#: The four shapes root named, then the five the contract also has to survive.
#: Keyed so a test can name one without knowing its position.
SHAPES = {
    "normal": shape_normal,
    "filtered": shape_filtered,
    "filtered-narrow": shape_filtered_narrow,
    "exhausted": shape_exhausted,
    "exhausted-partial-last": shape_exhausted_partial_last,
    "capped-call-limit": shape_capped_call_limit,
    "capped-byte-budget": shape_capped_byte_budget,
    "exhausted-mismatch": shape_exhausted_mismatch,
    "exhausted-no-count": shape_exhausted_no_count,
}


def per_fetch_maxima() -> dict:
    """Both sides of the 8-charged / 9-total boundary, measured.

    ``MAX_RESOLVER_CALLS_PER_FETCH`` bounds CHARGED batch callbacks. The
    terminal is uncharged, so the arithmetic maximum for one fetch is 8 + 1.
    Reaching it needs a population whose empty terminal probe IS the eighth
    charged call; one row more and the purse bites first and the terminal never
    fires at all.
    """
    out = {
        "MAX_RESOLVER_CALLS_PER_FETCH": result_handles.MAX_RESOLVER_CALLS_PER_FETCH,
        "MAX_FILL_ROUNDS": result_handles.MAX_FILL_ROUNDS,
        "cases": [],
    }
    for population, batch_size, label in [
        (280, 40, "terminal probe IS the 8th charged call -> 8B + 1T = 9"),
        (320, 40, "8 charged calls all return rows -> the cap bites, no terminal"),
        (240, 40, "terminal probe is the 7th charged call -> 7B + 1T"),
    ]:
        harness = Harness(portal_rows(population), batch_size=batch_size,
                          total=population)
        try:
            page = harness.fetch(budget_bytes=1_000_000)
            row = harness.portal.per_fetch()[0]
            out["cases"].append({
                "label": label, "population": population,
                "batch_size": batch_size, "callbacks": row["callbacks"],
                "batches": row["batches"], "terminals": row["terminals"],
                "sequence": row["sequence"],
                "terminal_seq": row["terminal_seq"],
                "terminal_is_last": row["terminal_is_last"],
                "incomplete_reason": page.incomplete_reason,
                "source_complete": bool(page.source_complete),
            })
        finally:
            harness.close()
    return out


class UnjudgedTerminal:
    """A portal that answers the terminal callback WITHOUT deciding.

    This is the unsettled-terminal retry, and it is a deliberate carry-forward
    rather than a defect: the old ``_reconcile`` returned without settling when
    the count was unavailable or raised, so the question was re-asked on every
    later fetch, one callback and no batch read each time. F4 reproduces that
    shape for an adapter that answers and declines to judge.

    ``FakePortal(count=False)`` does NOT reach it: that fixture answers
    ``complete: False`` with ``countonly_unavailable``, which is a DECISION and
    settles the walk for good. The difference is the contract working as
    specified -- absent is unjudged, ``False`` is judged -- and it is the reason
    this class exists instead of reusing the no-count portal.
    """

    def __init__(self, inner, *, reason="countonly_unavailable") -> None:
        self.inner = inner
        self.reason = reason

    def __call__(self, request):
        if isinstance(request, TerminalRequest):
            return {"incomplete_reason": self.reason}
        return self.inner(request)


class RaisingTerminal:
    """A portal whose terminal callback raises: ``countonly_error``."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def __call__(self, request):
        if isinstance(request, TerminalRequest):
            raise RuntimeError("Cannot get view results")
        return self.inner(request)


def edge_cases() -> dict:
    """The four edges root did not ask for and the contract has to survive."""
    results: dict = {}

    harness = Harness(portal_rows(80), batch_size=40,
                      portal=UnjudgedTerminal(FakePortal(portal_rows(80))))
    try:
        pages = []
        for _ in range(4):
            page = harness.fetch(budget_bytes=1_000_000)
            pages.append({"reason": page.incomplete_reason,
                          "complete": bool(page.source_complete),
                          "rows": len(page.rows)})
        results["unjudged_terminal_repeated_fetches"] = {
            "note": "an answer with no `complete` settles nothing, so the "
                    "question is owed again: one callback, zero batch reads",
            "per_fetch": harness.portal.per_fetch(),
            "pages": pages,
        }
    finally:
        harness.close()

    harness = Harness(portal_rows(80), batch_size=40,
                      portal=RaisingTerminal(FakePortal(portal_rows(80))))
    try:
        pages = []
        for _ in range(3):
            page = harness.fetch(budget_bytes=1_000_000)
            pages.append({"reason": page.incomplete_reason,
                          "complete": bool(page.source_complete),
                          "rows": len(page.rows)})
        results["raising_terminal_repeated_fetches"] = {
            "note": "countonly_error: the same unsettled-terminal retry path",
            "per_fetch": harness.portal.per_fetch(),
            "pages": pages,
        }
    finally:
        harness.close()

    harness = Harness(portal_rows(540), batch_size=40, materialized=40)
    try:
        warm_seen, _ = harness.walk()
        warm = harness.portal.per_fetch()
        warm_total = len(harness.portal.log)
        cold_portal = harness.cold()
        cold_seen, cold_pages = harness.walk()
        results["cold_resume"] = {
            "note": "a warm walk to completion, then the hot state dropped and "
                    "the SAME store re-walked in a fresh process image",
            "warm_fetches": len(warm),
            "warm_total_callbacks": warm_total,
            "warm_rows": len(warm_seen),
            "cold_per_fetch": cold_portal.per_fetch(),
            "cold_total_callbacks": len(cold_portal.log),
            "cold_rows": len(cold_seen),
            "cold_final_complete": bool(cold_pages[-1].source_complete),
        }
    finally:
        harness.close()

    harness = Harness(portal_rows(80), batch_size=40, total=80)
    try:
        harness.fetch(budget_bytes=1_000_000)
        first = len(harness.portal.log)
        cold_portal = harness.cold()
        page = harness.fetch(budget_bytes=1_000_000)
        results["verdict_survives_cold_resume"] = {
            "note": "a SETTLED verdict is inherited: the second process makes "
                    "no call at all",
            "first_pass_callbacks": first,
            "second_pass_callbacks": len(cold_portal.log),
            "second_pass_per_fetch": cold_portal.per_fetch(),
            "still_complete": bool(page.source_complete),
            "rows": len(page.rows),
        }
    finally:
        harness.close()

    return results


def trace_everything() -> dict:
    return {
        "shapes": [builder() for builder in SHAPES.values()],
        "maxima": per_fetch_maxima(),
        "edges": edge_cases(),
    }


def main() -> None:
    print(json.dumps(trace_everything(), indent=2))


if __name__ == "__main__":
    main()
