"""F6 (fix-iq53.2.11): continuation over a workflow that has no database.

The adapter boundary in ``fastworkflow.result_handles`` is not a database
abstraction, and this module is the proof that would cost something to fake: it
walks the real bundled ``tests/todo_list_workflow`` -- in-memory ``TodoList``
objects with integer ids and a description, no SQL, no offsets, no view, no
snapshot pin, not even a query language -- through the same ``fetch_page`` a
portal-backed handle walks, and gets the same pages out.

Three things about this adapter are the point, and each has a test named for
it below.

**One-item lookahead, and NOT ``len(rows) < limit``.** Revision 1 of the
boundary design said a batch is the last one when it comes back shorter than
the limit it was given. That is wrong on an exact multiple: the last full batch
is indistinguishable from a non-final one, so the rule has to spend one more
round trip on an empty batch to find out. It was retracted for that reason and
must not come back. This adapter asks for ``limit + 1``, returns ``limit`` and
lets the extra row answer has-more exactly, which is a thing an in-memory
source can do and an offset-paged view cannot. ``exact multiple`` below pins
two batch callbacks over six items at a batch size of three, and asserts there
is no third.

**A terminal callback that performs ZERO backend operations.** The lookahead
already proved exhaustion by the time the walk ends, so this adapter answers
the coverage question out of what it already knows. IDO's adapter answers the
same question with one ``countOnly`` against its portal. Both are one framework
callback; the work behind it differs by the whole of one backend operation.
That difference is what the dedicated terminal callback buys and what a
per-callback operation allowance could not have expressed -- an allowance would
have had to be large enough for IDO's terminal on every one of this adapter's
batches.

**The non-paginated half needs no adapter at all.** A command whose result is
small and complete declares a ``ResultHandleSpec`` with ``source=None``, takes
the ``plan == "local"`` path, and still gets stored rows, paging, tokens and a
whole-relation literal filter. ``NonPaginatedHandleTests`` is that exemplar: no
resolver is registered, nothing is callable, and the completeness on the page
is the producer's own claim carried through.
"""
from __future__ import annotations

import os
import tempfile
import unittest

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
from tests.todo_list_workflow.application.todo_list import TodoList

#: The name this module's adapter registers under. Nothing is persisted but
#: the name; the callable stays in this process, like every resolver.
RESOLVER = "todo-children"


def scope(turn: str = "turn-1", *, task: str = "task-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store",
        channel_id="channel",
        experiment_id="exp",
        task_id=task,
        attempt=1,
        turn_key=turn,
    )


def todo_list_of(size: int, *, description: str = "Trip") -> TodoList:
    """A real ``TodoList`` with *size* real children, built by the real API.

    ``add_child_todoitem`` allocates the ids, so the ids are the application's
    own and not a fixture's idea of them -- which is the whole reason this
    example uses the bundled workflow rather than a list of dicts.
    """
    todo_list = TodoList(id=1, description=description)
    for index in range(size):
        todo_list.add_child_todoitem("step %d" % (index + 1), assign_to="me")
    return todo_list


class TodoBackend:
    """The 'backend', and the one operation an adapter callback may spend on it.

    ``children_after`` stands where IDO's ``session.ws.query_view`` stands: the
    single call a callback is allowed to make, counted here so a test can
    assert a budget the framework cannot see across the boundary. The framework
    bounds callbacks, never their contents (Revision 4 §3.4), so the only place
    "one operation per callback" can be checked at all is the adapter's side of
    the line -- which is exactly where this counter is.

    There is no query here and nothing to page by: it reads the live object
    graph once and hands back a window of it.
    """

    def __init__(self, todo_list: TodoList) -> None:
        self.todo_list = todo_list
        self.operations = 0

    def children_after(self, after_id, limit: int) -> list:
        """The children after *after_id*, at most *limit* of them. One read."""
        self.operations += 1
        children = list(self.todo_list.get_all_children())
        if after_id is None:
            return children[:limit]
        for index, child in enumerate(children):
            if child.id == after_id:
                return children[index + 1:index + 1 + limit]
        return []


class TodoSource:
    """The adapter: lookahead on the batch, nothing at all on the terminal.

    Both callbacks arrive here and are told apart by TYPE, which is what F2
    (fix-iq53.2.6) bought by splitting the overloaded ``count_only`` mode flag
    into two request types: this adapter branches on what it was handed rather
    than on a boolean it has to read first.
    """

    def __init__(self, backend: TodoBackend, *, claim=True) -> None:
        self.backend = backend
        #: ``True``/``False`` settle the walk; ``None`` answers the terminal
        #: without deciding, which is the "no claim offered" case.
        self.claim = claim
        self.calls: list = []
        #: What was answered to each of them, in the same order, so a test can
        #: read the adapter's own vocabulary rather than replay its calls.
        self.replies: list[dict] = []
        #: Backend operations spent inside each callback, in call order. One
        #: entry per callback, so the terminal's entry is the zero.
        self.operations_per_call: list[int] = []

    def __call__(self, request):
        before = self.backend.operations
        self.calls.append(request)
        reply: dict = {}
        try:
            reply = (self.terminal(request) if isinstance(request, TerminalRequest)
                     else self.batch(request))
            return reply
        finally:
            self.replies.append(reply)
            self.operations_per_call.append(self.backend.operations - before)

    def batch(self, request) -> dict:
        """One batch, by one-item lookahead, in one backend operation.

        The resume point is the adapter's own and is opaque to the framework:
        ``{"after_id": n}`` is a key into this object graph and means nothing
        anywhere else. No offset is computed, offered or stored -- the
        framework allocates the batch ordinals and carries this mapping back
        verbatim on the next call.

        Nothing here claims completeness, and nothing here may: a batch
        response's ``complete`` is ignored (F2). Deciding is the terminal's
        job, and this adapter is in no hurry to do it early -- the lookahead
        that answers has-more is a statement about the NEXT batch, not about
        coverage of the query.
        """
        after = (request.continuation or {}).get("after_id")
        window = self.backend.children_after(after, request.limit + 1)
        batch, has_more = window[:request.limit], len(window) > request.limit
        reply: dict = {
            "rows": [{"id": child.id, "description": child.description}
                     for child in batch],
        }
        if has_more:
            reply["continuation"] = {"after_id": batch[-1].id}
        return reply

    def terminal(self, request) -> dict:
        """The coverage judgment, for ZERO backend operations.

        The lookahead already proved exhaustion: this adapter reached a batch
        whose window held no extra row, so it knows it walked the whole list
        and says so without reading anything again. An adapter that cannot know
        that -- IDO's offset walker, which cannot see past the batch it asked
        for -- spends one ``countOnly`` here instead. Same single framework
        callback, same placement, same count; the work behind it is the
        adapter's business and differs by a whole backend operation.

        ``claim is None`` answers without deciding, which settles nothing and
        leaves the callback owed again on the next fetch.
        """
        if self.claim is None:
            return {}
        return {"complete": self.claim}


class NonDatabaseContinuationTests(unittest.TestCase):
    """A walk over in-memory objects, with the ids the application allocated."""

    BATCH_SIZE = 3

    def setUp(self) -> None:
        reset_result_handle_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(reset_result_handle_state)
        self.addCleanup(result_handles.unregister_resolver, RESOLVER)
        self.store = ResultHandleStore(os.path.join(self.temp.name, "h.sqlite3"))

    def register(self, population: int, *, claim=True) -> TodoSource:
        self.backend = TodoBackend(todo_list_of(population))
        self.source = TodoSource(self.backend, claim=claim)
        result_handles.register_resolver(RESOLVER, self.source)
        return self.source

    def declare_handle(self, *, materialized: int = 0, total=None, **state):
        """What the producing command declares about the list it just showed.

        ``state`` is the adapter's, carried verbatim and never inspected here;
        ``list_id`` is in it because a real adapter would need to find its list
        again, and this one keeps the reference instead. There is no ``view``,
        no ``params``, no ``start_offset`` and no ``timeslot`` to put in it,
        which is the shape F1 (fix-iq53.2.5) was meant to make possible.
        """
        children = self.backend.todo_list.get_all_children()
        rendered = ["%d  %s" % (child.id, child.description)
                    for child in children[:materialized]]
        population = len(children) if total is None else total
        return declare(
            ResultHandleSpec(kind="todo", summary="%d step(s)." % population,
                             items=rendered, total=population,
                             source_complete=False, page_size=self.BATCH_SIZE),
            source=SourceDescriptor(
                resolver=RESOLVER, uid_field="id", label_fields=("description",),
                batch_size=self.BATCH_SIZE,
                state={"list_id": self.backend.todo_list.id, **state}),
            scope=scope(), selected_store=self.store, alias="O1",
        )

    def fetch(self, cursor=None, contains=None, *, budget=100_000):
        return fetch_page("O1", cursor, contains, scope=scope(),
                          selected_store=self.store, budget_bytes=budget)

    def counts(self) -> tuple[int, int]:
        """(batch callbacks, terminal callbacks) over the whole walk so far."""
        batches = [call for call in self.source.calls
                   if not isinstance(call, TerminalRequest)]
        terminals = [call for call in self.source.calls
                     if isinstance(call, TerminalRequest)]
        return len(batches), len(terminals)

    # -- the three population shapes -------------------------------------

    def test_an_empty_population_is_one_batch_and_one_terminal(self):
        """0 items at batch size 3: nothing to show, and it is proven nothing.

        The empty batch is THE stop condition (fix-iq53.2.4), so the walk ends
        on the first callback and the terminal judges an enumeration of zero
        rows as complete. A zero that was PROVEN is not the same page as a zero
        nobody could account for, and this is the one that was proven.
        """
        self.register(0)
        self.declare_handle()
        page = self.fetch()

        self.assertEqual(self.counts(), (1, 1))
        self.assertEqual(page.rows, [])
        self.assertTrue(page.source_complete)
        self.assertEqual(page.continuation, "complete")
        self.assertEqual(page.outcome, "complete-zero")
        self.assertIsNone(page.next_cursor)
        self.assertIsNone(page.incomplete_reason)
        # The batch offered no resume point, so the terminal was handed none.
        terminal = self.source.calls[-1]
        self.assertIsNone(terminal.continuation)
        self.assertEqual(terminal.distinct_uids, 0)

    def test_an_exact_multiple_needs_no_extra_empty_batch(self):
        """6 items at batch size 3: exactly two batch callbacks, and no third.

        This is the test the retracted ``len(rows) < limit`` rule fails. Under
        that rule both batches come back full, neither can be called last, and
        the adapter has to be asked a third time to be told there is nothing
        there. One-item lookahead sees the sixth item while serving the fifth,
        so the second batch returns its three rows and no continuation -- which
        F3 (fix-iq53.2.7) already treats as the end of the walk, after those
        rows rather than instead of them.
        """
        self.register(6)
        self.declare_handle()
        page = self.fetch()

        self.assertEqual(self.counts(), (2, 1),
                         "the exact multiple cost an extra empty batch")
        self.assertEqual(len(page.rows), 6)
        self.assertEqual(page.rows[0], "1  step 1")
        self.assertEqual(page.rows[-1], "6  step 6")
        self.assertTrue(page.source_complete)
        self.assertEqual(page.continuation, "complete")
        # The second batch answered and offered no way to resume, so that is
        # the resume point the terminal is handed: none.
        terminal = self.source.calls[-1]
        self.assertIsNone(terminal.continuation)
        self.assertEqual(terminal.distinct_uids, 6)

    def test_a_partial_last_batch_carries_its_rows_and_ends_the_walk(self):
        """7 items at batch size 3: three batch callbacks, the third with 1 row."""
        self.register(7)
        self.declare_handle()
        page = self.fetch()

        self.assertEqual(self.counts(), (3, 1))
        self.assertEqual([len(reply.get("rows", ())) for reply in
                          self.source.replies[:3]], [3, 3, 1])
        self.assertEqual(len(page.rows), 7)
        self.assertEqual(page.rows[-1], "7  step 7")
        self.assertTrue(page.source_complete)
        self.assertEqual(page.continuation, "complete")
        # Batch three asked to resume after the sixth child and was the last.
        batches = [call for call in self.source.calls
                   if not isinstance(call, TerminalRequest)]
        self.assertEqual([call.continuation for call in batches],
                         [None, {"after_id": 3}, {"after_id": 6}])

    # -- what the split of the two callbacks buys -------------------------

    def test_the_terminal_callback_costs_zero_backend_operations(self):
        """The claim the whole example exists to make, measured.

        Every batch callback spends exactly one operation; the terminal spends
        none. A per-callback allowance sized for IDO's terminal would have
        authorised a second operation on every batch of this walk, which is
        root's 2N objection in the shape it actually bites.
        """
        source = self.register(7)
        self.declare_handle()
        self.fetch()

        batch_count, terminal_count = self.counts()
        self.assertEqual((batch_count, terminal_count), (3, 1))
        # One entry per callback, terminal last, and it is the zero.
        self.assertEqual(source.operations_per_call, [1, 1, 1, 0])
        self.assertEqual(self.backend.operations, 3,
                         "the terminal read the backend")
        self.assertLessEqual(max(source.operations_per_call), 1,
                             "a callback spent more than one operation")

    def test_no_batch_callback_ever_claims_completeness(self):
        """Rule 3, from the adapter's side: it is not that it is ignored.

        A batch response MAY not claim completeness and this one does not try.
        The rule is enforced by nothing in the framework ever reading
        ``complete`` off a batch, so the thing worth asserting on this side is
        that the adapter's own vocabulary respects the split: the only reply
        carrying a decision is the reply to a ``TerminalRequest``.
        """
        source = self.register(7)
        self.declare_handle()
        self.fetch()

        self.assertEqual(len(source.calls), len(source.replies))
        for call, reply in zip(source.calls, source.replies):
            if isinstance(call, TerminalRequest):
                self.assertIn("complete", reply)
            else:
                self.assertNotIn("complete", reply)
                self.assertNotIn("incomplete_reason", reply)

    def test_a_terminal_that_decides_nothing_serves_its_rows_and_stops(self):
        """The no-claim case: ``completeness_not_claimed``, and no cursor.

        Nothing about this depends on the source being a database, which is the
        reason it is asserted here as well as against the portal fixture: an
        adapter that declines to judge gets the same conservative page whatever
        it is made of.
        """
        self.register(6, claim=None)
        self.declare_handle()
        page = self.fetch()

        self.assertEqual(self.counts(), (2, 1))
        self.assertEqual(len(page.rows), 6, "a stalled walk lost stored rows")
        self.assertFalse(page.source_complete)
        self.assertEqual(page.incomplete_reason, "completeness_not_claimed")
        self.assertEqual(page.continuation, "source-incomplete")
        self.assertIsNone(page.next_cursor)
        self.assertIn("has_more=false", page.as_observation())

    def test_the_unjudged_terminal_refires_and_reads_no_backend(self):
        """Owed again next fetch, for one callback and zero operations.

        The retry is the measured behaviour F4 deliberately reproduced rather
        than fixed. Here it is free twice over: one callback, no batch read,
        and -- because this adapter's terminal touches nothing -- no backend
        operation either.
        """
        source = self.register(6, claim=None)
        self.declare_handle()
        self.fetch()
        self.assertEqual(self.counts(), (2, 1))
        operations_after_walk = self.backend.operations

        for expected in (2, 3):
            reset_result_handle_state()
            page = self.fetch()
            batches, terminals = self.counts()
            self.assertEqual(terminals, expected, "the terminal did not re-fire")
            self.assertEqual(batches, 2, "the retry read a batch")
            self.assertEqual(self.backend.operations, operations_after_walk)
            self.assertEqual(page.incomplete_reason, "completeness_not_claimed")
            self.assertEqual(len(page.rows), 6)
        self.assertEqual(source.operations_per_call[-1], 0)

    # -- paging and filtering are generic, not portal-shaped --------------

    def test_the_walk_pages_by_token_and_never_repeats_a_row(self):
        """Generic paging (F9): tokens, ordinals and the packer, over objects.

        Nothing in the page machinery knows what this source is. The rows
        arrive one at a time under a one-row budget, the tokens resolve, and
        the sequence is the application's own child order with no row shown
        twice and none skipped.
        """
        self.register(7)
        self.declare_handle()
        seen: list[str] = []
        cursor, page = None, None
        for _ in range(12):
            page = self.fetch(cursor, budget=self.one_row_budget())
            seen.extend(page.rows)
            cursor = page.next_cursor
            if cursor is None:
                break
        self.assertEqual(seen, ["%d  step %d" % (index, index)
                                for index in range(1, 8)])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(page.continuation, "complete")

    def test_a_complete_walk_answers_a_literal_over_the_whole_relation(self):
        """Once the walk is proven complete the filter is a whole-relation one.

        ``plan == "local-filter"``: the framework matches the literal over its
        own stored rows because a complete local set IS the relation. This
        adapter has no filter columns and could not have run a server-side
        filter at all, which is the honest shape for a source with no query
        language -- and it still answers a named lookup, because completeness
        and not filterability is what makes that sound.
        """
        self.register(7)
        self.declare_handle()
        self.assertTrue(self.fetch().source_complete)

        hit = self.fetch(contains="step 4")
        self.assertEqual(hit.rows, ["4  step 4"])
        self.assertEqual(hit.outcome, "rows")
        self.assertTrue(hit.matched_complete)

        miss = self.fetch(contains="step 99")
        self.assertEqual(miss.rows, [])
        self.assertEqual(miss.outcome, "complete-zero")

    def one_row_budget(self) -> int:
        """A byte budget that admits the header, a token and exactly one row."""
        whole = self.fetch()
        lines = whole.as_observation().splitlines()
        fixed = [line for line in lines if line not in whole.rows]
        return (sum(len(line.encode("utf-8")) + 1 for line in fixed)
                + len(whole.rows[0].encode("utf-8")) + 1 + 32)


class NonPaginatedHandleTests(unittest.TestCase):
    """The ``source=None`` exemplar: a complete listing, and no adapter at all.

    A command whose result is small and already whole declares its rows and
    stops. There is no descriptor, so there is no resolver, nothing callable,
    no walk and no terminal callback -- ``fetch_page`` takes the ``local`` plan
    and serves what is stored. This is the half of the contract that needs no
    new machinery, and it is worth a test precisely because it is the half
    nobody thinks to check after changing the other one.
    """

    ROWS = ["%d  step %d" % (index, index) for index in range(1, 6)]

    def setUp(self) -> None:
        reset_result_handle_state()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(reset_result_handle_state)
        self.store = ResultHandleStore(os.path.join(self.temp.name, "h.sqlite3"))
        declare(
            ResultHandleSpec(kind="todo", summary="5 step(s).", items=self.ROWS,
                             total=5, source_complete=True, page_size=2),
            scope=scope(), selected_store=self.store, alias="O1",
        )

    def fetch(self, cursor=None, contains=None, *, budget=100_000):
        return fetch_page("O1", cursor, contains, scope=scope(),
                          selected_store=self.store, budget_bytes=budget)

    def test_the_handle_stores_no_descriptor_and_names_no_resolver(self):
        stored = self.store.get_declaration(scope(), "O1")
        self.assertEqual(stored["descriptor"], {})
        self.assertNotIn(RESOLVER, result_handles.registered_resolvers())

    def test_the_page_is_complete_on_the_producers_own_word(self):
        """``source_complete`` came from the declaration, not from a walk."""
        page = self.fetch()
        self.assertEqual(page.rows, self.ROWS)
        self.assertTrue(page.source_complete)
        self.assertEqual(page.continuation, "complete")
        self.assertEqual(page.outcome, "rows")
        self.assertIsNone(page.next_cursor)
        self.assertIsNone(page.incomplete_reason)

    def test_the_stored_rows_still_page(self):
        seen: list[str] = []
        cursor, page = None, None
        for _ in range(10):
            page = self.fetch(cursor, budget=self.one_row_budget())
            seen.extend(page.rows)
            cursor = page.next_cursor
            if cursor is None:
                break
        self.assertEqual(seen, self.ROWS)
        self.assertEqual(page.continuation, "complete")

    def test_the_stored_rows_still_filter_as_a_whole_relation(self):
        hit = self.fetch(contains="step 3")
        self.assertEqual(hit.rows, ["3  step 3"])
        self.assertTrue(hit.matched_complete)
        miss = self.fetch(contains="step 42")
        self.assertEqual(miss.outcome, "complete-zero")

    def one_row_budget(self) -> int:
        whole = self.fetch()
        lines = whole.as_observation().splitlines()
        fixed = [line for line in lines if line not in whole.rows]
        return (sum(len(line.encode("utf-8")) + 1 for line in fixed)
                + len(whole.rows[0].encode("utf-8")) + 1 + 32)


if __name__ == "__main__":
    unittest.main()
