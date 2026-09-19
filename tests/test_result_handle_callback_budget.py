"""F7 (fix-iq53.2.12): the boundary's work profile, pinned as regression tests.

Revision 4 §2 of the adapter-boundary design answered root's demand for real
traces of the normal, filtered, exhausted and capped walks, and the review
verified every row of it against the harness's raw JSON. Then the contract was
implemented (F1-F5a) and the thing that made those numbers trustworthy -- the
harness -- was still in ``/tmp``. Numbers nobody can re-derive stop being
evidence the first time the code moves.

So the harness is in the tree (``tests/callback_trace.py``) and this module
turns its output into assertions. **The four properties the design is actually
bought with**, each asserted over every shape rather than just the one it was
noticed in:

1. **At most 9 callbacks in one fetch** -- eight charged batch reads plus the
   one uncharged terminal, and no arrangement of population and budget reaches
   ten.
2. **At most 1 terminal callback in one fetch.** Never two, in any shape.
3. **The terminal is the LAST callback of its fetch** whenever it fires.
4. **A capped walk makes ZERO terminal callbacks.** The terminal is reached by
   exhausting the population, never by exhausting a budget: a walk the purse
   stopped has not ended, and asking whether its end covers the query would be
   asking about an end that does not exist yet.

Plus the fifth, which is a carry-forward rather than a bound: **the unsettled
terminal re-fires for one callback and zero batch reads.**

**What this module does NOT re-test.** ``TerminalCallbackTests`` in
``tests/test_result_handles.py`` already owns the terminal's semantics at unit
scale -- that a decision is stored and never asked twice, that an adapter's own
reason word is passed through, that a refusal to decide offers no cursor, what
``distinct_uids`` carries, and that a capped walk asks for no verdict. What was
missing, and is here, is the *shape* of the callback sequence across a whole
walk: the counts per fetch, their maximum, and where in each fetch the terminal
lands. One is about what a callback means; this one is about how many there are
and when.

The numbers below are the measured ones and they are identical to the
pre-implementation baseline in Revision 4 §2.1/§2.3/§2.4 -- same fetch counts,
same batch counts, same sequences, same placement. That identity is the
evidence that the terminal callback preserved ``_reconcile``'s work profile
rather than merely promising to.
"""
from __future__ import annotations

import unittest

from fastworkflow.result_handles import paging as result_handles
from fastworkflow.result_handles import reset_result_handle_state
from tests.callback_trace import (
    SHAPES,
    FakePortal,
    Harness,
    RaisingTerminal,
    UnjudgedTerminal,
    per_fetch_maxima,
    portal_rows,
)

#: The arithmetic maximum of one fetch: ``MAX_RESOLVER_CALLS_PER_FETCH``
#: charged batch reads plus the single uncharged terminal.
MAX_CALLBACKS_PER_FETCH = result_handles.MAX_RESOLVER_CALLS_PER_FETCH + 1

#: shape -> (fetches, callbacks, batch reads, terminals, per-fetch sequences).
#: ``B`` charged batch, ``T`` uncharged terminal, ``""`` a fetch served
#: entirely from stored batches and costing the source nothing. Revision 4
#: writes the terminal as ``C``; it is the same call in the same place, made by
#: the adapter now instead of by the framework.
EXPECTED = {
    "normal": (6, 15, 14, 1, ["BBB", "BB", "BBB", "BB", "BBB", "BT"]),
    "filtered": (6, 7, 6, 1, ["BBBB", "", "", "", "B", "BT"]),
    "filtered-narrow": (1, 3, 2, 1, ["BBT"]),
    "exhausted": (1, 4, 3, 1, ["BBBT"]),
    "exhausted-partial-last": (1, 4, 3, 1, ["BBBT"]),
    "capped-call-limit": (1, 8, 8, 0, ["BBBBBBBB"]),
    "capped-byte-budget": (1, 3, 3, 0, ["BBB"]),
    "exhausted-mismatch": (6, 16, 15, 1,
                           ["BBBBBBB", "", "BB", "BB", "BBBBT", ""]),
    "exhausted-no-count": (1, 4, 3, 1, ["BBBT"]),
}

#: shape -> (rows served, source_complete, incomplete_reason, outcome,
#: continuation) on the last page of the walk. Pinned beside the callback
#: counts because a cheaper walk that stopped answering the question would
#: otherwise look like an improvement.
EXPECTED_FINAL = {
    "normal": (540, True, None, "rows", "complete"),
    "filtered": (120, False, None, "rows", "complete"),
    "filtered-narrow": (1, False, None, "rows", "complete"),
    "exhausted": (80, True, None, "rows", "complete"),
    "exhausted-partial-last": (70, True, None, "rows", "complete"),
    "capped-call-limit": (8, False, "resolver_call_limit", "partial", "cursor"),
    "capped-byte-budget": (111, False, None, "partial", "cursor"),
    "exhausted-mismatch": (520, False, "countonly_mismatch", "partial",
                           "source-incomplete"),
    "exhausted-no-count": (80, False, "countonly_unavailable", "partial",
                           "source-incomplete"),
}

#: The shapes root named. The other four in ``SHAPES`` are edges the contract
#: also has to survive, and they are traced under the same invariants.
REQUIRED_SHAPES = ("normal", "filtered", "exhausted", "capped-call-limit")

_TRACES: dict | None = None


def traces() -> dict:
    """Every shape, walked once for the whole module.

    Each builder in ``SHAPES`` opens its own store, registers its own resolver
    and closes both, so running them once and reading the summaries nine times
    is equivalent to running them nine times and costs a ninth as much.
    """
    global _TRACES
    if _TRACES is None:
        reset_result_handle_state()
        _TRACES = {name: builder() for name, builder in SHAPES.items()}
    return _TRACES


class CallbackBudgetInvariantTests(unittest.TestCase):
    """The four properties, over every shape, with no exception allowed."""

    def test_no_fetch_ever_makes_more_than_nine_callbacks(self):
        """Eight charged plus one uncharged, and there is no tenth."""
        for name, trace in traces().items():
            with self.subTest(shape=name):
                for row in trace["per_fetch"]:
                    self.assertLessEqual(
                        row["callbacks"], MAX_CALLBACKS_PER_FETCH,
                        "%s fetch %d made %d callbacks"
                        % (name, row["fetch"], row["callbacks"]))
                    self.assertLessEqual(
                        row["batches"],
                        result_handles.MAX_RESOLVER_CALLS_PER_FETCH,
                        "the purse did not bound the charged calls")

    def test_no_fetch_ever_makes_more_than_one_terminal_callback(self):
        """Never two. Three guards compose to make a second unreachable.

        ``_extend_walk`` short-circuits on ``reconciled`` and on ``complete``
        before it can reach ``_issue_terminal`` again, and ``fetch_page``'s
        fill loop breaks on a complete walk. This is the measurement that
        agrees with that reading of the control flow.
        """
        for name, trace in traces().items():
            with self.subTest(shape=name):
                for row in trace["per_fetch"]:
                    self.assertLessEqual(
                        row["terminals"], 1,
                        "%s fetch %d asked twice" % (name, row["fetch"]))

    def test_a_terminal_is_always_the_last_callback_of_its_fetch(self):
        """It is issued after the fill loop breaks, so nothing follows it."""
        for name, trace in traces().items():
            with self.subTest(shape=name):
                placements = trace["terminal_is_last_in_its_fetch"]
                self.assertNotIn(False, placements,
                                 "%s put a callback after its terminal" % name)
                for row in trace["per_fetch"]:
                    if row["terminals"]:
                        self.assertTrue(row["sequence"].endswith("T"))
                        self.assertEqual(row["sequence"].count("T"), 1)

    def test_a_capped_walk_makes_no_terminal_callback_at_all(self):
        """Purse exhaustion and byte exhaustion are not ends of a walk.

        Both capped shapes stop for a budget reason with the population still
        running, so there is no terminal to judge and none is asked for. This
        is the pair where a per-callback operation allowance would have been
        worst: it would have authorised a coverage call on every callback of a
        walk that legitimately makes none.
        """
        for name in ("capped-call-limit", "capped-byte-budget"):
            with self.subTest(shape=name):
                trace = traces()[name]
                self.assertEqual(trace["total_terminals"], 0)
                self.assertEqual(trace["terminal_is_last_in_its_fetch"], [])
                for row in trace["per_fetch"]:
                    self.assertNotIn("T", row["sequence"])

    def test_every_shape_root_named_was_actually_traced(self):
        """A guard on the evidence rather than on the code.

        Root asked for four shapes by name. A refactor that quietly dropped one
        of them from the harness would leave this module green while measuring
        less than it says it measures.
        """
        for name in REQUIRED_SHAPES:
            self.assertIn(name, traces())
        self.assertEqual(set(traces()), set(EXPECTED))


class WalkShapeTraceTests(unittest.TestCase):
    """The per-shape numbers, pinned exactly as Revision 4 measured them."""

    def test_each_shape_makes_the_callbacks_the_baseline_made(self):
        for name, expected in EXPECTED.items():
            with self.subTest(shape=name):
                trace = traces()[name]
                fetches, callbacks, batches, terminals, sequences = expected
                self.assertEqual(
                    (trace["fetches"], trace["total_callbacks"],
                     trace["total_batch_reads"], trace["total_terminals"]),
                    (fetches, callbacks, batches, terminals))
                self.assertEqual(trace["sequences"], sequences)

    def test_each_shape_still_answers_what_it_answered_before(self):
        """The counts are only evidence if the pages did not get worse.

        A walk that made fewer callbacks by serving fewer rows, or by giving up
        on a completeness question it used to answer, would pass the budget
        assertions above and be a regression. These are the answers.
        """
        for name, expected in EXPECTED_FINAL.items():
            with self.subTest(shape=name):
                trace = traces()[name]
                rows, complete, reason, outcome, continuation = expected
                self.assertEqual(trace["rows_served"], rows)
                self.assertEqual(trace["distinct_rows"], rows,
                                 "a row was served twice")
                self.assertEqual(trace["final_source_complete"], complete)
                self.assertEqual(trace["final_incomplete_reason"], reason)
                self.assertEqual(trace["final_outcome"], outcome)
                self.assertEqual(trace["final_continuation"], continuation)

    def test_the_exact_multiple_still_costs_its_empty_probe(self):
        """``exhausted`` is 80 rows at batch size 40, and it reads three.

        An offset walker cannot know its last full batch was last, so the third
        read is the empty one that proves it. This is deliberately NOT optimised
        away: the retracted ``len(rows) < limit`` rule is what optimising it
        without lookahead would amount to, and it is wrong. F6's todo adapter
        does answer this shape in two callbacks -- with one-item lookahead,
        which is an adapter capability and not a framework rule.
        """
        trace = traces()["exhausted"]
        self.assertEqual(trace["sequences"], ["BBBT"])
        self.assertEqual(trace["per_fetch"][0]["offsets"], [0, 40, 80])
        self.assertTrue(trace["final_source_complete"])

    def test_a_fetch_served_from_stored_batches_costs_the_source_nothing(self):
        """The empty sequences are real: the walk read ahead of the packer.

        Three of ``filtered``'s six fetches make no callback at all, because
        the byte budget could print less than one batch held. Rows fetched and
        not shown are stored, not discarded, and the next cursor returns them
        without another backend read.
        """
        trace = traces()["filtered"]
        free = [row for row in trace["per_fetch"] if row["callbacks"] == 0]
        self.assertEqual(len(free), 3)
        self.assertEqual(trace["total_batch_reads"], 6)
        self.assertEqual(trace["rows_served"], 120)


class PerFetchMaximumTests(unittest.TestCase):
    """Both sides of the 8-charged/9-total boundary, reached on purpose."""

    def setUp(self) -> None:
        reset_result_handle_state()
        self.addCleanup(reset_result_handle_state)
        self.cases = per_fetch_maxima()["cases"]

    def test_the_ninth_callback_is_reachable_and_is_the_terminal(self):
        """280 rows at batch size 40: the empty probe IS the eighth charged call."""
        case = self.cases[0]
        self.assertEqual(case["sequence"], "BBBBBBBBT")
        self.assertEqual((case["callbacks"], case["batches"], case["terminals"]),
                         (MAX_CALLBACKS_PER_FETCH,
                          result_handles.MAX_RESOLVER_CALLS_PER_FETCH, 1))
        self.assertTrue(case["terminal_is_last"])
        self.assertTrue(case["source_complete"])

    def test_when_the_purse_bites_first_the_terminal_never_fires(self):
        """320 rows: eight charged calls all return rows, so there is no end yet."""
        case = self.cases[1]
        self.assertEqual(case["sequence"], "BBBBBBBB")
        self.assertEqual(case["terminals"], 0)
        self.assertEqual(case["incomplete_reason"], "resolver_call_limit")
        self.assertFalse(case["source_complete"])

    def test_a_terminal_below_the_cap_is_still_the_last_call(self):
        case = self.cases[2]
        self.assertEqual(case["sequence"], "BBBBBBBT")
        self.assertEqual(case["callbacks"], 8)
        self.assertTrue(case["terminal_is_last"])
        self.assertTrue(case["source_complete"])

    def test_the_purse_bounds_charged_calls_and_nothing_else(self):
        for case in self.cases:
            with self.subTest(population=case["population"]):
                self.assertLessEqual(
                    case["batches"],
                    result_handles.MAX_RESOLVER_CALLS_PER_FETCH)
                self.assertLessEqual(case["callbacks"], MAX_CALLBACKS_PER_FETCH)


class UnsettledTerminalTraceTests(unittest.TestCase):
    """The retry, in trace form: one callback, zero batch reads, every fetch.

    ``TerminalCallbackTests.test_an_unsettled_terminal_refires_once_per_fetch``
    already asserts the semantics on a four-row fixture. What is added here is
    the per-fetch cost across a real multi-batch walk, which is the number the
    design had to carry forward: the retry is cheap, and it is cheap because it
    reads the stored terminal batch instead of re-walking to it.
    """

    def setUp(self) -> None:
        reset_result_handle_state()
        self.addCleanup(reset_result_handle_state)

    def trace_repeated_fetches(self, portal, rounds: int):
        harness = Harness(portal_rows(80), batch_size=40, portal=portal)
        try:
            pages = [harness.fetch(budget_bytes=1_000_000)
                     for _ in range(rounds)]
            return harness.portal.per_fetch(), pages
        finally:
            harness.close()

    def test_an_answer_with_no_decision_is_owed_again_for_one_callback(self):
        per_fetch, pages = self.trace_repeated_fetches(
            UnjudgedTerminal(FakePortal(portal_rows(80))), 4)

        self.assertEqual([row["sequence"] for row in per_fetch],
                         ["BBBT", "T", "T", "T"])
        for row in per_fetch[1:]:
            self.assertEqual((row["callbacks"], row["batches"]), (1, 0),
                             "the retry read a batch")
        for page in pages:
            self.assertEqual(page.incomplete_reason, "countonly_unavailable")
            self.assertFalse(page.source_complete)
            self.assertEqual(len(page.rows), 80, "the retry lost stored rows")

    def test_a_terminal_that_raises_takes_the_same_retry_path(self):
        per_fetch, pages = self.trace_repeated_fetches(
            RaisingTerminal(FakePortal(portal_rows(80))), 3)

        self.assertEqual([row["sequence"] for row in per_fetch],
                         ["BBBT", "T", "T"])
        for page in pages:
            self.assertEqual(page.incomplete_reason, "countonly_error")
            self.assertEqual(len(page.rows), 80)

    def test_an_adapter_that_decides_no_is_not_asked_again(self):
        """The other side of the retry, and a real change from before F4.

        ``FakePortal(count=False)`` answers ``complete: False`` with
        ``countonly_unavailable`` -- a DECISION, made by the adapter, about a
        source that cannot prove its own coverage. That settles the walk, so
        later fetches cost nothing. The framework's old ``_reconcile`` could
        not tell those apart: an unavailable count returned without settling
        and so retried forever, whether or not anything was ever going to
        change. Absent is unjudged and ``False`` is judged, and which one a
        source deserves is the adapter's call to make.
        """
        per_fetch, pages = self.trace_repeated_fetches(
            FakePortal(portal_rows(80), count=False), 3)

        self.assertEqual([row["sequence"] for row in per_fetch],
                         ["BBBT", "", ""])
        for page in pages:
            self.assertEqual(page.incomplete_reason, "countonly_unavailable")
            self.assertEqual(len(page.rows), 80)


class ColdResumeCostTests(unittest.TestCase):
    """A stored verdict makes a completed walk free to replay."""

    def setUp(self) -> None:
        reset_result_handle_state()
        self.addCleanup(reset_result_handle_state)

    def test_a_completed_walk_costs_nothing_in_a_fresh_process_image(self):
        """Six fetches, 540 rows, and not one callback.

        The hot walk cache is dropped and the same store file re-read. The
        stored batches serve the rows and the stored terminal verdict answers
        the coverage question, so the source is not contacted at all -- which
        is the ido-1r0 property the ``batch_index`` rename had to not break.
        """
        harness = Harness(portal_rows(540), batch_size=40, materialized=40)
        try:
            warm_seen, _ = harness.walk()
            self.assertEqual(len(warm_seen), 540)
            self.assertEqual(len(harness.portal.log), 15)

            cold = harness.cold()
            cold_seen, cold_pages = harness.walk()
            self.assertEqual(len(cold_seen), 540)
            self.assertEqual(len(cold.log), 0, "a completed walk was re-proved")
            self.assertTrue(cold_pages[-1].source_complete)
        finally:
            harness.close()

    def test_a_settled_verdict_is_inherited_rather_than_re_proved(self):
        harness = Harness(portal_rows(80), batch_size=40, total=80)
        try:
            harness.fetch(budget_bytes=1_000_000)
            self.assertEqual(len(harness.portal.log), 4)

            cold = harness.cold()
            page = harness.fetch(budget_bytes=1_000_000)
            self.assertEqual(len(cold.log), 0)
            self.assertTrue(page.source_complete)
            self.assertEqual(len(page.rows), 80)
        finally:
            harness.close()


if __name__ == "__main__":
    unittest.main()
