"""C1 result handles: the store, its cursors, and bounded page observations."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from fastworkflow import result_handles
from fastworkflow import tracing
from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive
from fastworkflow.observation_offloading.labels import printed_alias, strip_alias_line
from fastworkflow.observation_offloading.state import reset_runtime_state
from fastworkflow.result_handles import (
    ResultHandleError,
    ResultHandleSpec,
    ResultHandleStore,
    SourceDescriptor,
    current_execute_alias,
    declare,
    fetch_page,
    normalize_literal,
    parent_handle,
    reset_result_handle_state,
)


def scope(turn: str = "turn-1", *, task: str = "task-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store",
        channel_id="channel",
        experiment_id="exp",
        task_id=task,
        attempt=1,
        turn_key=turn,
    )


def holders(count: int, *, prefix: str = "uid") -> list[str]:
    """`uid  label` rows exactly as a listing command renders them."""
    names = ["Alan Cooper", "Brandon Cooper", "Jill Cook", "John Martinez",
             "Zoe Bell", "Jennifer Wilson"]
    return [
        "%s%03d  %s %d" % (prefix, index, names[index % len(names)], index)
        for index in range(count)
    ]


#: Room for the continuation the header will carry once the page is not the
#: whole listing. The packer reserves the same thing internally.
CURSOR_ROOM = 128


def one_row_budget(handle, store, *, contains=None, selected_scope=None):
    """A budget that admits the header, its cursor and exactly one row."""
    whole = fetch_page(handle, None, contains, scope=selected_scope or scope(),
                       selected_store=store, budget_bytes=1_000_000)
    lines = whole.as_observation().splitlines()
    fixed = [line for line in lines if line not in whole.rows]
    return (sum(len(line.encode("utf-8")) + 1 for line in fixed)
            + len(whole.rows[0].encode("utf-8")) + 1 + CURSOR_ROOM)


class ResultHandleStoreTests(unittest.TestCase):
    """ido-986.14.1: declarations, immutable pages, scope isolation, retention."""

    def setUp(self) -> None:
        reset_result_handle_state()
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, "handles.sqlite3")
        self.store = ResultHandleStore(self.path)

    def tearDown(self) -> None:
        reset_result_handle_state()
        reset_runtime_state()
        self.temp.cleanup()

    def declare_listing(self, *, alias="O4", rows=6, total=None, complete=True,
                        selected_scope=None, source=None):
        items = holders(rows)
        return declare(
            ResultHandleSpec(
                kind="holder",
                summary="%d holder(s)." % (total or rows),
                items=items,
                ordering=result_handles.UNSORTED_OFFSET,
                total=total or rows,
                source_complete=complete,
                page_size=3,
                classification="user-text",
                presentation=True,
                filters={"permission": "Cloud Administrator"},
            ),
            source=source,
            scope=selected_scope or scope(),
            selected_store=self.store,
            alias=alias,
        )

    def test_declare_round_trips_through_sqlite(self):
        payload = self.declare_listing()
        self.assertTrue(payload["declared"])
        self.assertEqual(payload["result_handle"], "O4")
        # A separate store object on the same file: nothing is held in memory.
        reset_result_handle_state()
        reopened = ResultHandleStore(self.path)
        stored = reopened.get_declaration(scope(), "O4")
        self.assertEqual(stored["kind"], "holder")
        self.assertEqual(stored["total"], 6)
        self.assertEqual(stored["filters"], {"permission": "Cloud Administrator"})
        page = fetch_page("O4", scope=scope(), selected_store=reopened)
        self.assertEqual(page.rows, holders(6))

    def test_pages_are_immutable_and_refetch_is_idempotent(self):
        self.declare_listing()
        first = self.store.get_page(scope(), alias="O4", query_scope="", start_offset=0)
        again = self.store.put_page(
            scope(), alias="O4", query_scope="", start_offset=0,
            limit_requested=6, source="producer",
            record={"rows": [], "records": [{"uid": "other", "line": "other  x",
                                             "row": None}]},
            backend_total=6,
        )
        self.assertEqual(again["record_sha256"], first["record_sha256"])
        self.assertEqual(len(self.store.list_pages(scope(), alias="O4", query_scope="")), 1)
        self.assertEqual(again["record"]["records"][0]["line"], holders(6)[0])

    def test_redeclaring_a_different_query_under_one_alias_is_refused(self):
        self.declare_listing()
        with self.assertRaises(ResultHandleError):
            declare(
                ResultHandleSpec(kind="member", items=holders(2), total=2),
                source=SourceDescriptor(resolver="probe", view="other_view"),
                scope=scope(),
                selected_store=self.store,
                alias="O4",
            )

    def test_scopes_are_isolated(self):
        self.declare_listing(alias="O4", rows=6, selected_scope=scope("turn-1"))
        declare(
            ResultHandleSpec(kind="holder", items=holders(2, prefix="zzz"), total=2),
            scope=scope("turn-2"),
            selected_store=self.store,
            alias="O4",
        )
        first = fetch_page("O4", scope=scope("turn-1"), selected_store=self.store)
        second = fetch_page("O4", scope=scope("turn-2"), selected_store=self.store)
        self.assertEqual(len(first.rows), 6)
        self.assertEqual(len(second.rows), 2)
        with self.assertRaises(ResultHandleError):
            fetch_page("O4", scope=scope("turn-3"), selected_store=self.store)

    def test_eviction_never_loses_a_stored_page(self):
        self.declare_listing(rows=40)
        os.environ["FW_RESULT_HANDLE_HOT_MAX_BYTES"] = "1"
        try:
            for turn in range(4):
                declare(
                    ResultHandleSpec(kind="holder", items=holders(40), total=40),
                    scope=scope("turn-%d" % turn),
                    selected_store=self.store,
                    alias="O9",
                )
                fetch_page("O9", scope=scope("turn-%d" % turn),
                           selected_store=self.store)
            page = fetch_page("O4", scope=scope(), selected_store=self.store,
                              budget_bytes=100_000)
            self.assertEqual(page.rows, holders(40))
        finally:
            os.environ.pop("FW_RESULT_HANDLE_HOT_MAX_BYTES", None)

    def test_unknown_handle_names_what_is_stored(self):
        self.declare_listing()
        with self.assertRaises(ResultHandleError) as caught:
            fetch_page("O77", scope=scope(), selected_store=self.store)
        self.assertIn("O77", str(caught.exception))
        self.assertIn("O4", str(caught.exception))

    def test_descriptor_refuses_a_sorted_walk_and_an_invented_timeslot(self):
        with self.assertRaises(ResultHandleError):
            SourceDescriptor(resolver="probe", view="v", ordering="sorted-offset")
        with self.assertRaises(ResultHandleError):
            SourceDescriptor(resolver="probe", view="v", timeslot="2026-09-14")
        self.assertNotIn(
            "sort", SourceDescriptor(resolver="probe", view="v").as_dict()
        )

    def test_no_callable_is_persisted(self):
        result_handles.register_resolver("probe", lambda request: {"rows": []})
        try:
            self.declare_listing(
                source=SourceDescriptor(resolver="probe", view="ido_permissiondetail_identity",
                                        filter_columns=("identity_displayname",),
                                        uid_field="identity__id", page_size=3)
            )
            stored = self.store.get_declaration(scope(), "O4")
            self.assertEqual(stored["descriptor"]["resolver"], "probe")
            with open(self.path, "rb") as handle:
                blob = handle.read()
            self.assertNotIn(b"lambda", blob)
            self.assertNotIn(b"function", blob)
        finally:
            result_handles.unregister_resolver("probe")

    def test_an_unregistered_resolver_is_refused_by_name(self):
        with self.assertRaises(ResultHandleError) as caught:
            result_handles.resolver_for("not-registered")
        self.assertIn("not-registered", str(caught.exception))


class CursorTests(unittest.TestCase):
    """Query-scoped, opaque, and refused by name when they belong elsewhere."""

    def setUp(self) -> None:
        reset_result_handle_state()
        self.temp = tempfile.TemporaryDirectory()
        self.store = ResultHandleStore(os.path.join(self.temp.name, "h.sqlite3"))
        declare(
            ResultHandleSpec(kind="holder", items=holders(12), total=12,
                             source_complete=True, page_size=4),
            scope=scope(), selected_store=self.store, alias="O2",
        )

    def tearDown(self) -> None:
        reset_result_handle_state()
        self.temp.cleanup()

    def test_paging_never_skips_or_repeats_a_row(self):
        seen: list[str] = []
        cursor = None
        for _ in range(20):
            page = fetch_page("O2", cursor, scope=scope(),
                              selected_store=self.store, budget_bytes=600)
            seen.extend(page.rows)
            cursor = page.next_cursor
            if cursor is None:
                break
        self.assertEqual(seen, holders(12))
        self.assertEqual(page.continuation, "complete")

    def test_a_cursor_from_another_filter_scope_is_refused(self):
        filtered = fetch_page("O2", None, "Cooper", scope=scope(),
                              selected_store=self.store,
                              budget_bytes=one_row_budget("O2", self.store,
                                                          contains="Cooper"))
        self.assertIsNotNone(filtered.next_cursor)
        with self.assertRaises(ResultHandleError) as caught:
            fetch_page("O2", filtered.next_cursor, scope=scope(),
                       selected_store=self.store)
        self.assertIn("different query", str(caught.exception))
        unfiltered = fetch_page("O2", None, scope=scope(),
                                selected_store=self.store,
                                budget_bytes=one_row_budget("O2", self.store))
        with self.assertRaises(ResultHandleError):
            fetch_page("O2", unfiltered.next_cursor, "Cooper", scope=scope(),
                       selected_store=self.store)

    def test_a_cursor_from_another_handle_is_refused(self):
        declare(
            ResultHandleSpec(kind="holder", items=holders(4, prefix="b"), total=4),
            scope=scope(), selected_store=self.store, alias="O3",
        )
        page = fetch_page("O2", None, scope=scope(), selected_store=self.store,
                          budget_bytes=one_row_budget("O2", self.store))
        with self.assertRaises(ResultHandleError) as caught:
            fetch_page("O3", page.next_cursor, scope=scope(),
                       selected_store=self.store)
        self.assertIn("O2", str(caught.exception))

    def test_an_unreadable_cursor_is_refused_not_crashed(self):
        with self.assertRaises(ResultHandleError):
            fetch_page("O2", "!!not-base64!!", scope=scope(),
                       selected_store=self.store)

    def test_a_filtered_fetch_does_not_mutate_the_base_traversal(self):
        fetch_page("O2", None, "Cooper", scope=scope(), selected_store=self.store)
        base = fetch_page("O2", None, scope=scope(), selected_store=self.store,
                          budget_bytes=100_000)
        self.assertEqual(base.rows, holders(12))
        self.assertEqual(base.matched, 12)
        self.assertEqual(base.total, 12)


class ObservationTests(unittest.TestCase):
    """The rendered page: bounded, honest about coverage, never silently short."""

    def setUp(self) -> None:
        reset_result_handle_state()
        self.temp = tempfile.TemporaryDirectory()
        self.store = ResultHandleStore(os.path.join(self.temp.name, "h.sqlite3"))

    def tearDown(self) -> None:
        reset_result_handle_state()
        self.temp.cleanup()

    def declare_rows(self, items, **kwargs):
        spec = ResultHandleSpec(kind="holder", summary="%d holder(s)." % len(items),
                                items=items, total=kwargs.pop("total", len(items)),
                                **kwargs)
        return declare(spec, scope=scope(), selected_store=self.store, alias="O1")

    def test_the_header_carries_counts_the_handle_and_the_continuation(self):
        self.declare_rows(holders(12))
        page = fetch_page("O1", scope=scope(), selected_store=self.store,
                          budget_bytes=one_row_budget("O1", self.store))
        header = page.as_observation().splitlines()[0]
        self.assertIn("result_handle=O1", header)
        self.assertIn("total=12", header)
        self.assertIn("materialized=12", header)
        self.assertIn("continuation=cursor", header)
        self.assertIn("next_cursor=", header)
        self.assertLessEqual(
            len(page.as_observation().encode("utf-8")),
            one_row_budget("O1", self.store),
        )

    def test_an_oversized_row_is_shown_whole_with_a_warning(self):
        self.declare_rows(["uid1  " + "x" * 4000, "uid2  Alan Cooper"])
        page = fetch_page("O1", scope=scope(), selected_store=self.store)
        self.assertEqual(len(page.rows), 1)
        self.assertIn("x" * 4000, page.as_observation())
        self.assertTrue(
            any("over its" in warning and "budget" in warning
                for warning in page.warnings)
        )
        self.assertEqual(page.continuation, "cursor")

    def test_a_complete_zero_is_not_phrased_as_absence(self):
        self.declare_rows(holders(6))
        page = fetch_page("O1", None, "Ochoa", scope=scope(),
                          selected_store=self.store)
        self.assertEqual(page.outcome, "complete-zero")
        self.assertEqual(page.matched, 0)
        self.assertTrue(page.matched_complete)
        text = page.as_observation()
        self.assertIn("complete zero", text)
        self.assertIn("not evidence", text)

    def test_filtering_a_partial_listing_is_unsupported_not_partial(self):
        self.declare_rows(holders(6), total=477, source_complete=False)
        page = fetch_page("O1", None, "Cooper", scope=scope(),
                          selected_store=self.store)
        self.assertEqual(page.outcome, "unsupported")
        self.assertEqual(page.continuation, "source-incomplete")
        self.assertEqual(page.incomplete_reason, "producer_materialized_subset")
        self.assertIn("unsupported", page.as_observation().lower())

    def test_an_incomplete_source_is_reported_on_the_unfiltered_page(self):
        self.declare_rows(holders(6), total=477, source_complete=False)
        page = fetch_page("O1", scope=scope(), selected_store=self.store,
                          budget_bytes=100_000)
        self.assertEqual(page.materialized, 6)
        self.assertEqual(page.total, 477)
        self.assertFalse(page.source_complete)
        self.assertEqual(page.continuation, "source-incomplete")
        self.assertEqual(page.outcome, "partial")

    def test_the_literal_is_normalised_and_wildcards_are_removed(self):
        literal = normalize_literal("  Alan Coo%per​ ")
        self.assertEqual(literal.text, "Alan Cooper")
        self.assertTrue(any("wildcards" in note for note in literal.notes))
        self.declare_rows(holders(6))
        page = fetch_page("O1", None, "cooper", scope=scope(),
                          selected_store=self.store)
        self.assertEqual(page.matched, 2)
        self.assertIn('filter="cooper"', page.as_observation())


class AliasTests(unittest.TestCase):
    """The handle is the execute step's canonical O alias; a page links to it."""

    def setUp(self) -> None:
        reset_result_handle_state()
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self.store = ResultHandleStore(os.path.join(self.temp.name, "h.sqlite3"))

    def tearDown(self) -> None:
        reset_result_handle_state()
        reset_runtime_state()
        self.temp.cleanup()

    @staticmethod
    def host(trajectory, selected_scope=None):
        agent = SimpleNamespace(current_trajectory=trajectory,
                                continuation_scope=selected_scope or scope())
        return SimpleNamespace(workflow_tool_agent=agent)

    def test_the_alias_is_the_in_flight_execute_ordinal(self):
        trajectory = {
            "tool_name_0": "execute_workflow_query", "observation_0": "done",
            "tool_name_1": "search_memory", "observation_1": "answer",
            "tool_name_2": "execute_workflow_query",
        }
        with tracing.host_scope(self.host(trajectory)):
            self.assertEqual(current_execute_alias(), "O2")
        # Once the step has completed, this call is no longer inside it.
        trajectory["observation_2"] = "rows"
        with tracing.host_scope(self.host(trajectory)):
            self.assertIsNone(current_execute_alias())

    def test_a_declaration_with_no_agent_gets_a_non_O_store_key(self):
        payload = declare(
            ResultHandleSpec(kind="holder", items=holders(2), total=2),
            scope=scope(), selected_store=self.store,
        )
        self.assertTrue(payload["result_handle"].startswith("D"))

    def test_a_page_links_back_to_the_listing_it_paged(self):
        trajectory = {"tool_name_0": "execute_workflow_query"}
        with tracing.host_scope(self.host(trajectory)):
            payload = declare(
                ResultHandleSpec(kind="holder", items=holders(8), total=8,
                                 page_size=4),
                selected_store=self.store,
            )
        self.assertEqual(payload["result_handle"], "O1")
        trajectory["observation_0"] = "rows"
        trajectory["tool_name_1"] = "execute_workflow_query"
        with tracing.host_scope(self.host(trajectory)):
            page = fetch_page("O1", selected_store=self.store, budget_bytes=400)
            self.assertEqual(page.page_alias, "O2")
            self.assertEqual(parent_handle("O2", selected_store=self.store), "O1")
            # The page alias resolves to the listing, so the agent can page on
            # either handle and reach the same rows.
            follow = fetch_page("O2", page.next_cursor, selected_store=self.store,
                                budget_bytes=400)
            self.assertEqual(follow.handle, "O1")
            self.assertEqual(follow.rows, holders(8)[len(page.rows):])
            self.assertIsNone(parent_handle("O1", selected_store=self.store))


class OffloadingInterplayTests(unittest.TestCase):
    """A1's alias line, A2's eager archive and 14.6's saving rule are untouched."""

    def setUp(self) -> None:
        reset_result_handle_state()
        reset_runtime_state()
        self.temp = tempfile.TemporaryDirectory()
        self.store = ResultHandleStore(os.path.join(self.temp.name, "h.sqlite3"))
        self.archive = RuntimeHandleArchive(os.path.join(self.temp.name, "h.sqlite3"))

    def tearDown(self) -> None:
        reset_result_handle_state()
        reset_runtime_state()
        self.temp.cleanup()

    def test_the_page_observation_is_archived_without_its_alias_line(self):
        declare(
            ResultHandleSpec(kind="holder", items=holders(30), total=30),
            scope=scope(), selected_store=self.store, alias="O1",
        )
        page = fetch_page("O1", scope=scope(), selected_store=self.store)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": "listing text",
            "tool_name_1": "execute_workflow_query",
            "tool_args_1": {"command": "fetch_result_page handle=O1"},
            "observation_1": page.as_observation(),
        }
        compact_trajectory(trajectory, scope=scope(), selected_archive=self.archive)
        self.assertEqual(printed_alias(trajectory["observation_1"]), "O2")
        stored = self.archive.get(scope(), "O2")
        self.assertEqual(stored["text"], page.as_observation())
        self.assertEqual(strip_alias_line(trajectory["observation_1"]), stored["text"])

    def test_the_offload_archive_table_and_the_handle_tables_share_one_file(self):
        declare(
            ResultHandleSpec(kind="holder", items=holders(4), total=4),
            scope=scope(), selected_store=self.store, alias="O1",
        )
        self.archive.persist(
            scope(), alias="O1", offload_order=1, command_name="show_holders",
            step_index=0, text="listing text",
            text_sha256=__import__("hashlib").sha256(b"listing text").hexdigest(),
        )
        self.assertIsNotNone(self.archive.get(scope(), "O1"))
        self.assertIsNotNone(self.store.get_declaration(scope(), "O1"))


if __name__ == "__main__":
    unittest.main()
