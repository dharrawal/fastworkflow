"""ido-8ps.18: the extract step reads the evidence behind labels and pages.

Offline only. Nothing here starts a server, calls a model or touches a backend:
the archive and the result-handle store are real SQLite files in a temp dir, the
extract module is a recorder, and every assertion is about bytes that are already
on disk before the test begins.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import unittest
from unittest import mock

import dspy
from litellm import ContextWindowExceededError

from fastworkflow import answer_rehydration
from fastworkflow.answer_rehydration import (
    ANSWER_REHYDRATION_MAX_BYTES_ENV,
    DEFAULT_MAX_BYTES,
    KIND_LABEL,
    KIND_LISTING,
    KIND_PAGE,
    MIN_MAX_BYTES,
    NOT_REHYDRATED_KEY,
    NOT_REHYDRATED_PREFIX,
    max_bytes_from_env,
    rehydrate,
    trajectory_bytes,
)
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.labels import (
    alias_line,
    offload_label,
    printed_context,
)
from fastworkflow.observation_offloading.state import (
    record_context_clause,
    reset_runtime_state,
    snapshot_events,
)
from fastworkflow.result_handles import ResultHandleStore
from fastworkflow.utils.react import fastWorkflowReAct


def scope_for(name: str) -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity=name, channel_id="c", experiment_id="exp-ido-8ps-18",
        task_id="task", attempt=1, turn_key="turn-1",
    )


def rows(prefix: str, count: int, width: int = 40) -> list[str]:
    return ["%s-%03d | %s" % (prefix, index, "x" * width) for index in range(count)]


def page_record(lines: list[str]) -> dict:
    return {
        "rows": [],
        "records": [
            {"uid": line.split(" | ", 1)[0].strip(), "line": line, "row": None}
            for line in lines
        ],
    }


def declaration_payload(**overrides) -> dict:
    payload = {
        "kind": "permission-holders", "summary": "holders of the right",
        "ordering": "unsorted-offset", "total": 60, "materialized": 25,
        "source_complete": True, "page_size": 25, "classification": "user-text",
        "presentation": True, "filters": {}, "descriptor": {},
        "descriptor_sha256": hashlib.sha256(b"{}").hexdigest(),
        "parent_alias": "", "query_scope": "", "cursor_position": 0,
    }
    payload.update(overrides)
    return payload


class Fixture:
    """A real archive and store, with one label, one listing and one page."""

    def __init__(self, directory: str, *, listing_rows: int = 60) -> None:
        self.scope = scope_for(directory)
        path = os.path.join(directory, "obs.sqlite3")
        self.archive = RuntimeHandleArchive(path)
        self.store = ResultHandleStore(path)
        # (a) an offloaded observation, archived without its presentation line
        self.o1_text = "identity_uid | rights\n" + "\n".join(rows("ident", 30))
        self.archive.persist(
            self.scope, alias="O1", offload_order=1,
            command_name="list_permissions", step_index=0, text=self.o1_text,
            text_sha256=hashlib.sha256(self.o1_text.encode("utf-8")).hexdigest(),
        )
        record_context_clause(self.scope, "O1", "Identity 28c5aeb5 Alan Cooper")
        # (b) a listing that declared a handle, with three stored pages
        self.store.put_declaration(self.scope, "O2", declaration_payload())
        self.listing_rows = rows("holder", listing_rows)
        for index in range(0, listing_rows, 25):
            self.store.put_page(
                self.scope, alias="O2", query_scope="", start_offset=index,
                limit_requested=25, source="resolver",
                record=page_record(self.listing_rows[index:index + 25]),
                backend_total=listing_rows,
            )
        # a filtered traversal of the same handle
        self.filtered_rows = rows("filtered", 4)
        self.store.put_page(
            self.scope, alias="O2", query_scope="c:abc123", start_offset=0,
            limit_requested=25, source="resolver",
            record=page_record(self.filtered_rows), backend_total=4,
        )
        # (c) a page observation of that listing
        self.store.put_declaration(
            self.scope, "O3",
            declaration_payload(kind="permission-holders-page", parent_alias="O2",
                                materialized=25, cursor_position=25),
        )

    def trajectory(self) -> dict:
        return {
            "thought_0": "look up the holders",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "list_permissions"},
            "observation_0": offload_label(
                alias="O1", command_name="list_permissions",
                response=self.o1_text, description="the entitlement rows",
            ),
            "thought_1": "list the holders",
            "tool_name_1": "execute_workflow_query",
            "tool_args_1": {"command": "show_holders"},
            "observation_1": (
                alias_line("O2", "Permission 6fadcafc Cloud Administrator")
                + "result_handle=O2 page 1 rows 1-25 of 60\n"
                + "\n".join(self.listing_rows[:25])
            ),
            "thought_2": "page it",
            "tool_name_2": "execute_workflow_query",
            "tool_args_2": {"command": "fetch_result_page"},
            "observation_2": (
                alias_line("O3")
                + "result_handle=O2 page 2 rows 26-50 of 60\n"
                + "\n".join(self.listing_rows[25:50])
            ),
        }


class Budget(unittest.TestCase):
    """The budget: derived from the window, overridable, defended.

    ``ido-pyw.1`` removed ``FW_ANSWER_REHYDRATION``; rehydration is what the
    extract step does. What is left here is the byte budget, which is now a
    fraction of the model's context window with the old name as a tuning
    override. ``tests/test_context_budget.py`` owns the derivation; this owns
    the module's view of it.
    """

    def setUp(self) -> None:
        os.environ.pop(ANSWER_REHYDRATION_MAX_BYTES_ENV, None)

    tearDown = setUp

    def test_the_derived_budget_is_the_accepted_value(self) -> None:
        self.assertEqual(max_bytes_from_env(), DEFAULT_MAX_BYTES)
        self.assertEqual(DEFAULT_MAX_BYTES, 250_000)

    def test_the_override_is_read_and_defended(self) -> None:
        os.environ[ANSWER_REHYDRATION_MAX_BYTES_ENV] = "60000"
        self.assertEqual(max_bytes_from_env(), 60_000)
        os.environ[ANSWER_REHYDRATION_MAX_BYTES_ENV] = "not-a-number"
        self.assertEqual(max_bytes_from_env(), DEFAULT_MAX_BYTES)
        os.environ[ANSWER_REHYDRATION_MAX_BYTES_ENV] = str(MIN_MAX_BYTES - 1)
        self.assertEqual(max_bytes_from_env(), DEFAULT_MAX_BYTES)

    def test_the_env_file_is_read_first(self) -> None:
        import fastworkflow

        with mock.patch.dict(fastworkflow._env_vars,
                             {ANSWER_REHYDRATION_MAX_BYTES_ENV: "60000"},
                             clear=False):
            self.assertEqual(max_bytes_from_env(), 60_000)


class RehydratesEachKind(unittest.TestCase):

    def setUp(self) -> None:
        reset_runtime_state()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(reset_runtime_state)
        self.fixture = Fixture(self.directory.name)

    def rehydrate(self, trajectory=None, budget=DEFAULT_MAX_BYTES):
        return rehydrate(
            trajectory if trajectory is not None else self.fixture.trajectory(),
            scope=self.fixture.scope, archive=self.fixture.archive,
            handle_store=self.fixture.store, budget=budget,
        )

    def test_a_label_becomes_the_archived_observation(self) -> None:
        copy, report = self.rehydrate()
        self.assertIn(self.fixture.o1_text, copy["observation_0"])
        self.assertEqual(report.counts[KIND_LABEL], 1)

    def test_the_alias_and_context_lines_are_preserved(self) -> None:
        copy, _ = self.rehydrate()
        self.assertTrue(copy["observation_0"].startswith(
            "Observation O1 (execute_workflow_query, in Identity 28c5aeb5 Alan Cooper)\n"))
        self.assertEqual(printed_context(copy["observation_0"]),
                         "Identity 28c5aeb5 Alan Cooper")

    def test_an_alias_with_no_recorded_context_prints_the_plain_a1_line(self) -> None:
        reset_runtime_state()
        fixture = Fixture(tempfile.mkdtemp())
        copy, _ = rehydrate(
            fixture.trajectory(), scope=fixture.scope, archive=fixture.archive,
            handle_store=fixture.store, budget=DEFAULT_MAX_BYTES,
        )
        # record_context_clause ran inside Fixture; drop it to make the point.
        reset_runtime_state()
        copy, _ = rehydrate(
            fixture.trajectory(), scope=fixture.scope, archive=fixture.archive,
            handle_store=fixture.store, budget=DEFAULT_MAX_BYTES,
        )
        self.assertTrue(copy["observation_0"].startswith(
            "Observation O1 (execute_workflow_query)\n"))

    def test_a_listing_keeps_its_text_and_gains_every_stored_row(self) -> None:
        trajectory = self.fixture.trajectory()
        # Without the page observation above it, the listing is the carrier.
        for key in ("thought_2", "tool_name_2", "tool_args_2", "observation_2"):
            trajectory.pop(key)
        copy, report = self.rehydrate(trajectory)
        text = copy["observation_1"]
        self.assertIn("result_handle=O2 page 1 rows 1-25 of 60", text)
        for line in self.fixture.listing_rows:
            self.assertIn(line, text)
        self.assertEqual(report.counts[KIND_LISTING], 1)

    def test_the_filtered_traversal_is_appended_and_labelled(self) -> None:
        copy, _ = self.rehydrate()
        text = copy["observation_2"]
        self.assertIn("filtered traversal c:abc123", text)
        for line in self.fixture.filtered_rows:
            self.assertIn(line, text)

    def test_a_page_observation_is_rehydrated_through_its_listing(self) -> None:
        trajectory = self.fixture.trajectory()
        # Only the page observation, so the dedupe cannot hide the behaviour.
        for key in ("thought_1", "tool_name_1", "tool_args_1", "observation_1"):
            trajectory.pop(key)
        copy, report = self.rehydrate(trajectory)
        self.assertEqual(report.counts[KIND_PAGE], 1)
        self.assertIn("which paged it", copy["observation_2"])
        for line in self.fixture.listing_rows:
            self.assertIn(line, copy["observation_2"])

    def test_rows_behind_one_handle_are_not_repeated(self) -> None:
        copy, report = self.rehydrate()
        # Most recent first: the page observation carries the block, and the
        # listing below it is left alone rather than paying for it twice.
        self.assertEqual(report.counts[KIND_PAGE], 1)
        self.assertEqual(report.counts[KIND_LISTING], 0)
        self.assertNotIn("stored rows behind result_handle=O2",
                         copy["observation_1"])

    def test_nothing_is_invented_for_an_unarchived_alias(self) -> None:
        trajectory = self.fixture.trajectory()
        trajectory["observation_0"] = offload_label(
            alias="O9", command_name="list_permissions", response="x" * 4000)
        trajectory["tool_name_0"] = "execute_workflow_query"
        copy, report = self.rehydrate(trajectory)
        self.assertEqual(copy["observation_0"], trajectory["observation_0"])
        self.assertIn("O9", report.unresolved_aliases)

    def test_a_plain_observation_is_untouched(self) -> None:
        trajectory = self.fixture.trajectory()
        trajectory["observation_3"] = "Observation O4 (execute_workflow_query)\nfine"
        trajectory["tool_name_3"] = "execute_workflow_query"
        copy, _ = self.rehydrate(trajectory)
        self.assertEqual(copy["observation_3"], trajectory["observation_3"])

    def test_the_input_trajectory_is_never_mutated(self) -> None:
        trajectory = self.fixture.trajectory()
        before = dict(trajectory)
        copy, _ = self.rehydrate(trajectory)
        self.assertEqual(trajectory, before)
        self.assertIsNot(copy, trajectory)

    def test_the_archive_and_store_are_only_read(self) -> None:
        before_archive = self.fixture.archive.list(self.fixture.scope)
        before_pages = self.fixture.store.list_pages(
            self.fixture.scope, alias="O2", query_scope="")
        self.rehydrate()
        self.assertEqual(self.fixture.archive.list(self.fixture.scope),
                         before_archive)
        self.assertEqual(
            self.fixture.store.list_pages(self.fixture.scope, alias="O2",
                                          query_scope=""),
            before_pages)


class BudgetAndOrder(unittest.TestCase):

    def setUp(self) -> None:
        reset_runtime_state()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(reset_runtime_state)
        self.fixture = Fixture(self.directory.name)

    def test_the_budget_is_respected(self) -> None:
        trajectory = self.fixture.trajectory()
        budget = trajectory_bytes(trajectory) + 200
        copy, report = rehydrate(
            trajectory, scope=self.fixture.scope, archive=self.fixture.archive,
            handle_store=self.fixture.store, budget=budget,
        )
        self.assertLessEqual(trajectory_bytes(copy) - len(
            copy.get(NOT_REHYDRATED_KEY, "").encode("utf-8")), budget)
        self.assertTrue(report.dropped_aliases)

    def test_most_recent_first(self) -> None:
        trajectory = self.fixture.trajectory()
        # Room for the page block (the newest candidate) and nothing else.
        page_block = answer_rehydration.stored_rows_block(
            "O2", scope=self.fixture.scope, store=self.fixture.store,
            shown_for="O3")
        budget = trajectory_bytes(trajectory) + len(page_block.encode("utf-8")) + 1
        copy, report = rehydrate(
            trajectory, scope=self.fixture.scope, archive=self.fixture.archive,
            handle_store=self.fixture.store, budget=budget,
        )
        self.assertEqual([item["alias"] for item in report.rehydrated], ["O3"])
        self.assertIn("O1", report.dropped_aliases)
        self.assertEqual(copy["observation_0"], trajectory["observation_0"])

    def test_the_dropped_alias_line(self) -> None:
        trajectory = self.fixture.trajectory()
        copy, report = rehydrate(
            trajectory, scope=self.fixture.scope, archive=self.fixture.archive,
            handle_store=self.fixture.store, budget=trajectory_bytes(trajectory),
        )
        self.assertEqual(copy[NOT_REHYDRATED_KEY], report.note_line)
        self.assertTrue(report.note_line.startswith(NOT_REHYDRATED_PREFIX))
        self.assertEqual(report.note_line,
                         NOT_REHYDRATED_PREFIX + ", ".join(report.dropped_aliases))
        # Deterministic: ascending by execute ordinal, every time.
        self.assertEqual(report.dropped_aliases,
                         sorted(report.dropped_aliases,
                                key=lambda alias: int(alias[1:])))

    def test_no_drop_line_when_everything_fits(self) -> None:
        copy, report = rehydrate(
            self.fixture.trajectory(), scope=self.fixture.scope,
            archive=self.fixture.archive, handle_store=self.fixture.store,
            budget=DEFAULT_MAX_BYTES,
        )
        self.assertNotIn(NOT_REHYDRATED_KEY, copy)
        self.assertEqual(report.dropped_aliases, [])

    def test_the_report_counts_the_bytes(self) -> None:
        trajectory = self.fixture.trajectory()
        copy, report = rehydrate(
            trajectory, scope=self.fixture.scope, archive=self.fixture.archive,
            handle_store=self.fixture.store, budget=DEFAULT_MAX_BYTES,
        )
        self.assertEqual(report.bytes_before, trajectory_bytes(trajectory))
        self.assertEqual(report.bytes_after, trajectory_bytes(copy))
        self.assertGreater(report.bytes_after, report.bytes_before)


class Recorder:
    """Stands in for ``self.extract``: records what it was called with."""

    def __init__(self, *, fail_times: int = 0) -> None:
        self.calls: list[str] = []
        self.fail_times = fail_times

    def __call__(self, **kwargs):
        self.calls.append(kwargs["trajectory"])
        if len(self.calls) <= self.fail_times:
            raise ContextWindowExceededError(
                message="too long", model="m", llm_provider="p")
        return dspy.Prediction(answer="done")

    async def acall(self, **kwargs):
        return self(**kwargs)


def build_agent() -> fastWorkflowReAct:
    def a_tool(value: str) -> str:
        """A tool."""
        return value

    return fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=2)


class ExtractHook(unittest.TestCase):

    def setUp(self) -> None:
        reset_runtime_state()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(reset_runtime_state)
        os.environ.pop(ANSWER_REHYDRATION_MAX_BYTES_ENV, None)
        self.addCleanup(
            lambda: os.environ.pop(ANSWER_REHYDRATION_MAX_BYTES_ENV, None))
        self.fixture = Fixture(self.directory.name)
        self.agent = build_agent()
        self.agent.continuation_scope = self.fixture.scope
        self.agent.observation_archive = self.fixture.archive

    def _patch_store(self):
        return mock.patch("fastworkflow.result_handles.store",
                          return_value=self.fixture.store)

    def test_the_extractor_is_handed_the_evidence(self) -> None:
        """ido-pyw.1: nothing set, and the extract call still gets the rows."""
        trajectory = self.fixture.trajectory()
        plain = self.agent._format_trajectory(trajectory)
        recorder = Recorder()
        self.agent.extract = recorder
        with self._patch_store():
            self.agent._extract_prediction(trajectory, user_query="q")
        self.assertNotEqual(recorder.calls[0], plain)
        self.assertIn(self.fixture.o1_text, recorder.calls[0])
        self.assertIn(self.fixture.listing_rows[-1], recorder.calls[0])

    def test_the_react_trajectory_is_unchanged_after_extraction(self) -> None:
        trajectory = self.fixture.trajectory()
        before = dict(trajectory)
        self.agent.extract = Recorder()
        with self._patch_store():
            self.agent._extract_prediction(trajectory, user_query="q")
        self.assertEqual(trajectory, before)

    def test_the_events_carry_the_measures(self) -> None:
        os.environ[ANSWER_REHYDRATION_MAX_BYTES_ENV] = "80000"
        self.agent.extract = Recorder()
        with self._patch_store():
            self.agent._extract_prediction(self.fixture.trajectory(), user_query="q")
        events = snapshot_events()
        started = [e for e in events if e["kind"] == "rehydration_started"]
        finished = [e for e in events if e["kind"] == "rehydration_finished"]
        self.assertEqual(len(started), 1)
        self.assertEqual(len(finished), 1)
        self.assertEqual(started[0]["budget_bytes"], 80_000)
        self.assertGreater(finished[0]["bytes_after"], finished[0]["bytes_before"])
        self.assertEqual(finished[0]["rehydrated_labels"], 1)
        self.assertEqual(finished[0]["rehydrated_pages"], 1)
        self.assertIn("dropped_aliases", finished[0])
        self.assertFalse(finished[0]["rehydration_overflow"])
        self.assertIn("extract_prompt_tokens", finished[0])
        self.assertIsInstance(finished[0]["extract_duration_ms"], float)

    def test_the_overflow_fallback_still_truncates_and_is_recorded(self) -> None:
        trajectory = self.fixture.trajectory()
        recorder = Recorder(fail_times=1)
        self.agent.extract = recorder
        with self._patch_store():
            self.agent._extract_prediction(trajectory, user_query="q")
        self.assertEqual(len(recorder.calls), 2)
        overflow = [e for e in snapshot_events()
                    if e["kind"] == "rehydration_overflow"]
        self.assertEqual(len(overflow), 1)
        finished = [e for e in snapshot_events()
                    if e["kind"] == "rehydration_finished"][0]
        self.assertTrue(finished["rehydration_overflow"])
        # The fallback truncated the COPY; the loop's trajectory still has step 0.
        self.assertIn("observation_0", trajectory)

    def test_a_broken_store_costs_the_evidence_and_not_the_answer(self) -> None:
        """The extract call still happens, over the pointers it already had.

        It is no longer byte-identical to the un-rehydrated call, because the
        coverage statement (ido-8ps.22, unconditional since ido-pyw.1) is
        prepended to every extract call whether rehydration worked or not. What
        the fallback owes is the answer, and the record of why: no stored row
        reaches the extractor and ``rehydration_failed`` is filed.
        """
        trajectory = self.fixture.trajectory()
        recorder = Recorder()
        self.agent.extract = recorder
        with mock.patch("fastworkflow.result_handles.store",
                        side_effect=RuntimeError("no store")):
            self.agent._extract_prediction(trajectory, user_query="q")
        self.assertEqual(len(recorder.calls), 1)
        self.assertNotIn(self.fixture.o1_text, recorder.calls[0])
        self.assertTrue([e for e in snapshot_events()
                         if e["kind"] == "rehydration_failed"])

    def test_the_async_site_behaves_the_same(self) -> None:
        recorder = Recorder()
        self.agent.extract = recorder
        with self._patch_store():
            asyncio.run(self.agent._async_extract_prediction(
                self.fixture.trajectory(), user_query="q"))
        self.assertIn(self.fixture.o1_text, recorder.calls[0])
        self.assertTrue([e for e in snapshot_events()
                         if e["kind"] == "rehydration_finished"])


class ContinuationSite(unittest.TestCase):
    """The agent that actually runs is the segmented one (ido-986.14.x)."""

    def test_finish_prediction_goes_through_the_hook(self) -> None:
        from fastworkflow.observation_offloading.continuation import (
            StructuredContinuationReAct,
        )

        agent = StructuredContinuationReAct(
            "user_query -> answer", tools=[lambda value: value], max_iters=2)
        with mock.patch.object(
            StructuredContinuationReAct, "_extract_prediction",
            return_value={"answer": "a"},
        ) as hook:
            agent._finish_prediction({"thought_0": "t"}, {"user_query": "q"})
        hook.assert_called_once()


if __name__ == "__main__":
    unittest.main()
