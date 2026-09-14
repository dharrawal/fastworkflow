"""Arm D observation offloading: compact, search_memory, continuation, span manifest."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dspy

from fastworkflow import tracing
from fastworkflow.observation_offloading.agent import (
    ENABLED_ENV,
    build_compacting_step,
    build_tool_agent,
)
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import (
    PACKED_TARGET_BYTES,
    compact_trajectory,
    execute_ordinals,
    packed_target_bytes_from_env,
)
from fastworkflow.observation_offloading.continuation import (
    DEFAULT_CONTINUATION_PLAN,
    MAX_FORCED_REPLANS,
    REPLAN_OBSERVATION_MAX_BYTES,
    StructuredContinuationReAct,
    max_forced_replans_from_env,
    replan_trajectory_skeleton,
)
from fastworkflow.observation_offloading.labels import offload_label
from fastworkflow.observation_offloading.manifest import (
    classify_against_steps,
    install_span_policy,
    uninstall_span_policy,
)
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.observation_offloading.search import (
    DEFAULT_PAGE_BYTES,
    InvalidPageBoundary,
    search_memory,
    text_page,
)
from fastworkflow.observation_offloading.state import (
    HANDLE_ARCHIVE_ENV,
    HOT_HANDLE_MAX_BYTES,
    clear_hot_handles,
    hot_handle_max_bytes_from_env,
    hot_payload_bytes,
    record_event,
    remember_handle,
    reset_runtime_state,
    snapshot_events,
    stored_handles,
)
from fastworkflow.observation_offloading.compact import RECENT_OBSERVATIONS_PROTECTED


class CompactTrajectory(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            store_identity="fixture-store",
            channel_id="fixture-channel",
            experiment_id="fixture-experiment",
            task_id="fixture-task",
            attempt=1,
            turn_key="fixture-turn",
        )

    def compact(self, trajectory, **kwargs):
        return compact_trajectory(
            trajectory,
            scope=self.scope,
            selected_archive=self.archive,
            **kwargs,
        )

    def test_offloads_old_large_execute_observation_and_keeps_recent(self) -> None:
        large = ("holder uid label\n" + ("aaaa " * 400)) * 4
        trajectory = {}
        for index in range(7):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"show_holders_{index}"}
            trajectory[f"observation_{index}"] = large if index == 0 else f"small-{index}"
        decisions = self.compact(
            trajectory,
            packed_target_tokens=10,
            recent_observations_protected=5,
        )
        self.assertEqual(decisions[0]["action"], "offloaded")
        self.assertEqual(decisions[0]["alias"], "O1")
        self.assertIn("Use search_memory tool to search inside Observation", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_6"], "small-6")
        self.assertIn("O1", stored_handles(self.scope))

    @unittest.skipUnless(os.environ.get("FW_TEST_OBSERVATION_SEARCH_LIVE") == "1", "requires configured observation-search provider")
    def test_search_memory_returns_handle_page_not_full_dump(self) -> None:
        large = "477 holder(s).\nAaron Garrison\n" + ("row\n" * 4000)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = f"small-{index}"
        self.compact(trajectory, packed_target_tokens=10)
        answer = search_memory(
            "Is Aaron Garrison on the offloaded holders?",
            alias="O1",
            scope=self.scope,
            selected_archive=self.archive,
        )
        self.assertIn("Aaron Garrison", answer)
        self.assertLess(len(answer), len(large))
        self.assertIn("Observation O1", answer)

    def test_default_packed_target_is_28000_utf8_bytes(self) -> None:
        large = "holder uid label\n" + ("x" * 30_000)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = f"small-{index}"
        decisions = self.compact(trajectory)
        self.assertEqual(decisions[0]["action"], "offloaded")
        self.assertEqual(trajectory["observation_5"], "small-5")

    def test_default_recent_observations_protected_is_5(self) -> None:
        self.assertEqual(RECENT_OBSERVATIONS_PROTECTED, 5)
        large = "holder uid label\n" + ("x" * 30_000)
        trajectory = {}
        for index in range(7):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"show_{index}"}
            trajectory[f"observation_{index}"] = large
        decisions = self.compact(trajectory)
        recency = {item["alias"]: item["recency_protected"] for item in decisions}
        self.assertEqual(
            recency,
            {
                "O1": False,
                "O2": False,
                "O3": True,
                "O4": True,
                "O5": True,
                "O6": True,
                "O7": True,
            },
        )
        self.assertIn("Use search_memory tool to search inside Observation", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_6"], large)

    def test_default_target_measures_multibyte_utf8_not_characters(self) -> None:
        large = "é" * 15_000
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = f"small-{index}"
        decisions = self.compact(trajectory)
        self.assertEqual(len(large.encode("utf-8")), 30_000)
        self.assertEqual(decisions[0]["action"], "offloaded")

    @unittest.skipUnless(os.environ.get("FW_TEST_OBSERVATION_SEARCH_LIVE") == "1", "requires configured observation-search provider")
    def test_hot_cache_cap_oldest_eviction_and_durable_fallback(self) -> None:
        large = "target person\n" + ("row\n" * 4000)
        trajectory = {}
        for index in range(7):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"show_{index}"}
            trajectory[f"observation_{index}"] = large if index < 2 else f"small-{index}"
        self.compact(trajectory, packed_target_tokens=10, hot_handle_max_bytes=8_000)
        self.assertEqual(hot_payload_bytes(self.scope), 0)
        answer = search_memory(
            "target person",
            alias="O1",
            scope=self.scope,
            selected_archive=self.archive,
        )
        self.assertIn("tier=sqlite", answer)
        self.assertIn("target person", answer)

    @unittest.skipUnless(os.environ.get("FW_TEST_OBSERVATION_SEARCH_LIVE") == "1", "requires configured observation-search provider")
    def test_restart_like_empty_hot_cache_finds_sqlite(self) -> None:
        large = "restart answer\n" + ("row\n" * 1_500)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = f"small-{index}"
        self.compact(trajectory, packed_target_tokens=10)
        clear_hot_handles(self.scope)
        answer = search_memory(
            "What exact heading appears at the start of this observation?",
            alias="O1",
            scope=self.scope,
            selected_archive=self.archive,
        )
        self.assertEqual(stored_handles(self.scope), {})
        self.assertIn("tier=sqlite", answer)
        self.assertIn("restart answer", answer)

    def test_broken_persistence_retains_original_without_label(self) -> None:
        class BrokenArchive:
            def persist(self, *args, **kwargs):
                raise OSError("disk unavailable")

        large = "holder uid label\n" + ("x" * 30_000)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": large,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = f"small-{index}"
        compact_trajectory(
            trajectory,
            scope=self.scope,
            selected_archive=BrokenArchive(),  # type: ignore[arg-type]
            packed_target_tokens=10,
        )
        self.assertEqual(trajectory["observation_0"], large)

    def test_scope_prevents_alias_collision(self) -> None:
        other = RuntimeHandleScope(
            store_identity="fixture-store",
            channel_id="fixture-channel",
            experiment_id="fixture-experiment",
            task_id="fixture-task",
            attempt=1,
            turn_key="other-turn",
        )
        self.archive.persist(
            self.scope,
            alias="O1",
            offload_order=1,
            command_name="show",
            step_index=0,
            text="first-turn secret",
            text_sha256=hashlib.sha256(b"first-turn secret").hexdigest(),
        )
        answer = search_memory(
            "first-turn secret",
            alias="O1",
            scope=other,
            selected_archive=self.archive,
        )
        self.assertIn("no matching offloaded handle", answer)
        with self.assertRaises(PersistenceError):
            self.archive.persist(
                self.scope,
                alias="O1",
                offload_order=1,
                command_name="show",
                step_index=0,
                text="different text",
                text_sha256=hashlib.sha256(b"different text").hexdigest(),
            )


class StructuredContinuation(unittest.TestCase):
    def setUp(self):
        reset_runtime_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        from fastworkflow.observation_offloading.state import HANDLE_ARCHIVE_ENV
        env = patch.dict(os.environ, {HANDLE_ARCHIVE_ENV: str(Path(self.tempdir.name) / "replan.sqlite3")})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(reset_runtime_state)

    def test_greedy_28k_inlines_newest_first_and_never_exceeds_bound(self) -> None:
        trajectory = {}
        for index in range(4):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = str(index) * 600
        skeleton, metadata = replan_trajectory_skeleton(trajectory, greedy_max_bytes=2_000)
        self.assertLessEqual(metadata["measured_bytes"], 2_000)
        self.assertEqual(metadata["inlined_aliases"], ["O3", "O4"])
        self.assertEqual(metadata["labeled_aliases"], ["O1", "O2"])
        self.assertEqual(skeleton["observation_3"], "3" * 600)
        self.assertIn("Use search_memory tool to search inside Observation", skeleton["observation_1"])

    def test_greedy_28k_labels_single_oversized_newest_observation(self) -> None:
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": "x" * (REPLAN_OBSERVATION_MAX_BYTES + 1),
        }
        skeleton, metadata = replan_trajectory_skeleton(trajectory)
        self.assertLessEqual(metadata["measured_bytes"], REPLAN_OBSERVATION_MAX_BYTES)
        self.assertEqual(metadata["inlined_aliases"], [])
        self.assertEqual(metadata["labeled_aliases"], ["O1"])
        self.assertIn("Use search_memory tool to search inside Observation", skeleton["observation_0"])

    def test_limit_fires_at_25_twice_then_third_cap_stops(self) -> None:
        class ScriptedAgent(StructuredContinuationReAct):
            def _run_loop(self, trajectory, idx, input_args, max_iters, exception_count):
                self.segment_calls.append(max_iters)
                self.iteration_counter += max_iters
                trajectory[f"tool_name_{idx}"] = "execute_workflow_query"
                trajectory[f"tool_args_{idx}"] = {"command": f"segment-{len(self.segment_calls)}"}
                trajectory[f"observation_{idx}"] = f"result-{len(self.segment_calls)}"
                self._exhausted_last_run = True
                return None

            def _force_replan(self, trajectory, input_args):
                self.replan_counter_snapshots.append(self.iteration_counter)
                self.forced_replans += 1
                self.iteration_counter = 0
                trajectory[f"replan_{self.forced_replans}"] = "short plan"

            def _finish_prediction(self, trajectory, input_args):
                self.finished_trajectory = dict(trajectory)
                return SimpleNamespace(exhausted=self._exhausted_last_run)

        agent = ScriptedAgent.__new__(ScriptedAgent)
        agent.max_iters = 25
        agent.forced_replans = 0
        agent.iteration_counter = 0
        agent.segment_calls = []
        agent.replan_counter_snapshots = []
        result = agent._run_segments({}, 0, {"user_query": "task"}, 25)
        self.assertTrue(result.exhausted)
        self.assertEqual(agent.segment_calls, [25, 25, 25])
        self.assertEqual(agent.replan_counter_snapshots, [25, 25])
        self.assertEqual(agent.forced_replans, MAX_FORCED_REPLANS)

    def test_natural_finish_before_25_does_not_replan(self) -> None:
        class FinishingAgent(StructuredContinuationReAct):
            def _run_loop(self, trajectory, idx, input_args, max_iters, exception_count):
                self.iteration_counter = 12
                self._exhausted_last_run = False
                return None

            def _force_replan(self, trajectory, input_args):
                raise AssertionError("natural finish must not replan")

            def _finish_prediction(self, trajectory, input_args):
                return SimpleNamespace(exhausted=self._exhausted_last_run)

        agent = FinishingAgent.__new__(FinishingAgent)
        agent.max_iters = 25
        agent.forced_replans = 0
        agent.iteration_counter = 0
        result = agent._run_segments({}, 0, {"user_query": "task"}, 25)
        self.assertFalse(result.exhausted)
        self.assertEqual(agent.forced_replans, 0)

    def test_ask_user_resume_does_not_consume_replan_allowance(self) -> None:
        class ResumeAgent(StructuredContinuationReAct):
            def _run_segments(self, trajectory, idx, input_args, max_iters):
                self.resume_state = {"trajectory": dict(trajectory), "idx": idx}
                return SimpleNamespace(suspended=False)

        agent = ResumeAgent.__new__(ResumeAgent)
        agent._suspended = {
            "trajectory": {
                "tool_name_4": "ask_user",
                "tool_args_4": {"clarification_request": "Which scope?"},
            },
            "idx": 4,
            "input_args": {"user_query": "task"},
            "max_iters": 25,
        }
        agent.current_trajectory = {}
        agent.iteration_counter = -1
        agent.forced_replans = 1
        agent.resume("All domains")
        self.assertEqual(agent.forced_replans, 1)
        self.assertEqual(agent.iteration_counter, 0)
        self.assertEqual(agent.resume_state["idx"], 5)
        self.assertEqual(agent.resume_state["trajectory"]["observation_4"], "All domains")

    def test_actual_replan_resets_only_iteration_counter(self) -> None:
        captured = {}

        def fake_predict(_signature):
            def invoke(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(next_steps="1. Continue with focused lookup.")

            return invoke

        agent = StructuredContinuationReAct.__new__(StructuredContinuationReAct)
        agent.max_iters = 25
        agent.forced_replans = 0
        agent.iteration_counter = 25
        agent.current_trajectory = {}
        trajectory = {
            "thought_0": "Need evidence",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": "x" * 30_000,
        }
        with patch(
            "fastworkflow.observation_offloading.continuation.dspy.Predict",
            side_effect=fake_predict,
        ):
            agent._force_replan(trajectory, {"user_query": "audit"})
        self.assertEqual(agent.iteration_counter, 0)
        self.assertEqual(agent.forced_replans, 1)
        self.assertEqual(trajectory["observation_0"], "x" * 30_000)
        self.assertIn("segment 2 of 3", trajectory["replan_1"])
        self.assertNotIn("x" * 1000, captured["trajectory_skeleton"])


class TrajectoryManifest(unittest.TestCase):
    def setUp(self) -> None:
        install_span_policy()
        self.addCleanup(uninstall_span_policy)

    def test_manifest_classifies_resident_labelled_absent(self) -> None:
        resident = "holder uid Alan Cooper\n" + ("row\n" * 40)
        raw_offloaded = "permission portrait\n" + ("field\n" * 80)
        label = offload_label(
            alias="O5",
            command_name="Permission/show_holders",
            response=raw_offloaded,
        )
        user = (
            "[[ ## thought_0 ## ]]\nlook\n"
            f"[[ ## observation_0 ## ]]\n{resident}\n"
            f"[[ ## observation_1 ## ]]\n{label}\n"
            "[[ ## next_thought ## ]]\nanswer"
        )
        system = "Available execute_workflow_query tool commands:\n" + ("- cmd\n" * 4000)
        payload = json.dumps(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            ensure_ascii=False,
        )
        capped = tracing._capped({"messages": payload, "module": "react"})
        manifest = capped["trajectory_manifest"]
        stored = capped["messages"]
        self.assertTrue(stored["truncated"])
        self.assertEqual(stored["value"], "")
        classified = classify_against_steps(
            manifest,
            {
                0: hashlib.sha256(resident.encode("utf-8")).hexdigest(),
                1: hashlib.sha256(raw_offloaded.encode("utf-8")).hexdigest(),
                2: hashlib.sha256(b"later").hexdigest(),
            },
        )
        self.assertEqual(classified["resident"], [0])
        self.assertEqual(classified["labelled"], [1])
        self.assertEqual(classified["absent"], [2])


class PageBoundaries(unittest.TestCase):
    """Review finding: text_page handed back a non-newline end that the next call rejected."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            store_identity="fixture-store",
            channel_id="fixture-channel",
            experiment_id="fixture-experiment",
            task_id="fixture-task",
            attempt=1,
            turn_key="fixture-turn",
        )

    def _persist(self, text: str) -> None:
        self.archive.persist(
            self.scope,
            alias="O1",
            offload_order=1,
            command_name="dump",
            step_index=0,
            text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def test_pages_chain_through_a_text_with_no_newlines(self) -> None:
        text = "a" * 9_000 + " needle " + "b" * 1_000
        pages = []
        start = 0
        while True:
            page = text_page(text, start, DEFAULT_PAGE_BYTES)
            pages.append(page)
            if not page["has_more"]:
                break
            start = page["end_byte"]
        self.assertEqual(len(pages), 3)
        self.assertEqual("".join(page["text"] for page in pages), text)
        self.assertEqual(pages[0]["end_byte"], DEFAULT_PAGE_BYTES)

    def test_multibyte_text_without_newlines_never_splits_a_character(self) -> None:
        text = "é" * 5_000 + " 針 needle"
        page = text_page(text, 0, DEFAULT_PAGE_BYTES)
        self.assertEqual(page["end_byte"] % 2, 0)
        self.assertEqual(page["text"], "é" * (DEFAULT_PAGE_BYTES // 2))
        second = text_page(text, page["end_byte"], DEFAULT_PAGE_BYTES)
        self.assertTrue(second["text"].startswith("é"))

    def test_start_inside_a_multibyte_character_is_rejected(self) -> None:
        with self.assertRaises(InvalidPageBoundary):
            text_page("é" * 10, 1, DEFAULT_PAGE_BYTES)

    @unittest.skipUnless(os.environ.get("FW_TEST_OBSERVATION_SEARCH_LIVE") == "1", "requires configured observation-search provider")
    def test_search_memory_reaches_a_later_page_of_a_single_line_blob(self) -> None:
        text = "x" * 6_000 + " Aaron Garrison " + "y" * 500
        self._persist(text)
        answer = search_memory(
            "Is Aaron Garrison in the blob?",
            alias="O1",
            scope=self.scope,
            selected_archive=self.archive,
        )
        # The answer must include the evidence itself, not only a page range.
        self.assertIn("Aaron Garrison", answer)
        self.assertNotIn("no page in the first", answer)

    @unittest.skipUnless(os.environ.get("FW_TEST_OBSERVATION_SEARCH_LIVE") == "1", "requires configured observation-search provider")
    def test_search_memory_survives_non_ascii_single_line_blob(self) -> None:
        text = "日本語" * 1_000 + " target person"
        self._persist(text)
        answer = search_memory(
            "target person",
            alias="O1",
            scope=self.scope,
            selected_archive=self.archive,
        )
        self.assertIn("target person", answer)


class TruncationTolerance(unittest.TestCase):
    """Review finding: execute_ordinals stopped at the first missing step index."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            store_identity="fixture-store",
            channel_id="fixture-channel",
            experiment_id="fixture-experiment",
            task_id="fixture-task",
            attempt=1,
            turn_key="fixture-turn",
        )

    @staticmethod
    def _trajectory(count: int) -> dict:
        trajectory = {}
        for index in range(count):
            trajectory[f"thought_{index}"] = f"think-{index}"
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"show_{index}"}
            trajectory[f"observation_{index}"] = f"small-{index}"
        return trajectory

    def test_ordinals_survive_a_missing_leading_step(self) -> None:
        trajectory = self._trajectory(4)
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            del trajectory[f"{prefix}_0"]
        self.assertEqual(execute_ordinals(trajectory), [(1, 1), (2, 2), (3, 3)])
        self.assertEqual(
            execute_ordinals(trajectory, ordinal_offset=1),
            [(1, 2), (2, 3), (3, 4)],
        )

    def test_ordinals_skip_non_execute_steps_and_interior_gaps(self) -> None:
        trajectory = self._trajectory(5)
        trajectory["tool_name_1"] = "what_can_i_do"
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            del trajectory[f"{prefix}_2"]
        self.assertEqual(execute_ordinals(trajectory), [(0, 1), (3, 2), (4, 3)])

    def test_compaction_continues_after_truncation_without_alias_collision(self) -> None:
        large_first = "first dump\n" + ("row\n" * 3_000)
        large_second = "second dump\n" + ("col\n" * 3_000)
        # Eight executes: the newest five are recency-protected, so O1..O3 are
        # the offloadable slots before truncation and O2..O3 after it.
        trajectory = self._trajectory(8)
        trajectory["observation_0"] = large_first
        trajectory["observation_1"] = large_second
        first = compact_trajectory(
            trajectory,
            scope=self.scope,
            selected_archive=self.archive,
            packed_target_tokens=10,
            recent_observations_protected=5,
        )
        offloaded = [item["alias"] for item in first if item["action"] == "offloaded"]
        self.assertEqual(offloaded, ["O1", "O2"])

        # The base ReAct context-window fallback drops step 0 entirely.
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            del trajectory[f"{prefix}_0"]
        trajectory["observation_2"] = "third dump\n" + ("val\n" * 3_000)
        second = compact_trajectory(
            trajectory,
            scope=self.scope,
            selected_archive=self.archive,
            packed_target_tokens=10,
            recent_observations_protected=5,
            ordinal_offset=1,
        )
        by_alias = {item["alias"]: item for item in second}
        self.assertEqual(by_alias["O2"]["reason"], "already_label")
        self.assertEqual(by_alias["O3"]["action"], "offloaded")
        stored = {row["alias"]: row["text"] for row in self.archive.list(self.scope)}
        self.assertEqual(stored["O1"], large_first)
        self.assertEqual(stored["O2"], large_second)
        self.assertTrue(stored["O3"].startswith("third dump"))

    def test_agent_truncation_pops_by_step_index_and_counts_executes(self) -> None:
        agent = StructuredContinuationReAct.__new__(StructuredContinuationReAct)
        agent.truncated_execute_steps = 0
        trajectory = {"replan_1": "plan first"}
        trajectory.update(self._trajectory(3))
        trajectory["tool_name_1"] = "what_can_i_do"
        agent.truncate_trajectory(trajectory)
        self.assertEqual(agent.truncated_execute_steps, 1)
        self.assertIn("replan_1", trajectory)
        self.assertNotIn("observation_0", trajectory)
        agent.truncate_trajectory(trajectory)
        self.assertEqual(agent.truncated_execute_steps, 1)
        self.assertNotIn("tool_name_1", trajectory)
        self.assertIn("tool_name_2", trajectory)

    def test_replan_skeleton_keeps_persisted_aliases_after_truncation(self) -> None:
        trajectory = self._trajectory(3)
        for prefix in ("thought", "tool_name", "tool_args", "observation"):
            del trajectory[f"{prefix}_0"]
        _skeleton, metadata = replan_trajectory_skeleton(trajectory, ordinal_offset=1)
        self.assertEqual(metadata["inlined_aliases"], ["O2", "O3"])


class PerTurnScope(unittest.TestCase):
    """Review finding: the handle scope was frozen at agent construction."""

    class Signature(dspy.Signature):
        user_query: str = dspy.InputField()
        answer: str = dspy.OutputField()

    @staticmethod
    def _scope(turn_key: str) -> RuntimeHandleScope:
        return RuntimeHandleScope(
            store_identity="fixture-store",
            channel_id="fixture-channel",
            experiment_id="fixture-experiment",
            task_id="fixture-task",
            attempt=1,
            turn_key=turn_key,
        )

    def setUp(self) -> None:
        reset_runtime_state()

    def _agent(self, turn_keys: list[str]) -> StructuredContinuationReAct:
        keys = iter(turn_keys)

        def noop_tool(command: str) -> str:
            """Return the command unchanged."""
            return command

        return StructuredContinuationReAct(
            self.Signature,
            tools=[noop_tool],
            max_iters=3,
            scope_factory=lambda: self._scope(next(keys)),
        )

    def test_bind_scope_follows_the_turn_and_clears_the_old_hot_cache(self) -> None:
        agent = self._agent(["turn-1", "turn-2"])
        first = agent.bind_scope()
        remember_handle(first, {"alias": "O1", "text": "abc", "text_sha256": "x"})
        self.assertEqual(set(stored_handles(first)), {"O1"})
        second = agent.bind_scope()
        self.assertNotEqual(first.scope_id, second.scope_id)
        self.assertEqual(agent.continuation_scope, second)
        self.assertEqual(agent.continuation_scope_id, second.scope_id)
        self.assertEqual(stored_handles(first), {})

    def test_suspended_state_carries_the_scope_across_processes(self) -> None:
        agent = self._agent(["turn-1"])
        scope = agent.bind_scope()
        agent.truncated_execute_steps = 2
        agent._suspended = {
            "trajectory": {"tool_name_0": "ask_user"},
            "idx": 0,
            "input_args": {"user_query": "q"},
            "max_iters": 3,
            "clarification": "Which?",
        }
        blob = json.loads(json.dumps(agent.export_suspended()))
        self.assertEqual(blob["continuation_scope"]["turn_key"], "turn-1")
        restored = self._agent(["turn-9"])
        restored.import_suspended(blob)
        self.assertEqual(restored.continuation_scope, scope)
        self.assertEqual(restored.continuation_scope_id, scope.scope_id)
        self.assertEqual(restored.truncated_execute_steps, 2)

    def test_two_turns_offload_their_own_O1_into_one_archive(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        archive = RuntimeHandleArchive(str(Path(tempdir.name) / "handles.sqlite3"))
        agent = self._agent(["turn-1", "turn-2"])
        step = build_compacting_step(
            lambda: agent, fallback_scope=self._scope("fallback"), selected_archive=archive
        )
        texts = []
        for turn in range(2):
            agent.bind_scope()
            text = f"turn {turn} dump\n" + (f"row{turn}\n" * 3_000)
            texts.append(text)
            trajectory = {}
            for index in range(7):
                trajectory[f"tool_name_{index}"] = "execute_workflow_query"
                trajectory[f"tool_args_{index}"] = {"command": f"show_{index}"}
                trajectory[f"observation_{index}"] = text if index == 0 else "small"
            trajectory["observation_1"] = "z" * 30_000
            self.assertTrue(step(6, trajectory))
            self.assertIn("Use search_memory tool to search inside Observation", trajectory["observation_0"])
        first_handles = {
            scope_key: archive.get(self._scope(scope_key), "O1")["text"][:11]
            for scope_key in ("turn-1", "turn-2")
        }
        self.assertEqual(first_handles, {"turn-1": "turn 0 dump", "turn-2": "turn 1 dump"})
        self.assertIsNone(archive.get(self._scope("fallback"), "O1"))
        refused = [e for e in snapshot_events() if e["kind"] == "offload_refused"]
        self.assertEqual(refused, [])


class HookIsolation(unittest.TestCase):
    """Review finding: a failure inside the step hook aborted the whole turn."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tempdir.name) / "handles.sqlite3"))
        self.scope = RuntimeHandleScope(
            store_identity="fixture-store",
            channel_id="fixture-channel",
            experiment_id="fixture-experiment",
            task_id="fixture-task",
            attempt=1,
            turn_key="fixture-turn",
        )

    def _set_env(self, name: str, value: str) -> None:
        previous = os.environ.get(name)

        def restore() -> None:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

        self.addCleanup(restore)
        os.environ[name] = value

    def test_compaction_failure_is_recorded_and_the_step_continues(self) -> None:
        seen = []
        broken_agent = SimpleNamespace(
            continuation_scope=self.scope, truncated_execute_steps="not-a-number"
        )
        step = build_compacting_step(
            lambda: broken_agent,
            fallback_scope=self.scope,
            selected_archive=self.archive,
            on_step_complete=lambda idx, trajectory: seen.append(idx) or True,
        )
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show"},
            "observation_0": "x" * 30_000,
        }
        self.assertTrue(step(0, trajectory))
        self.assertEqual(seen, [0])
        self.assertEqual(trajectory["observation_0"], "x" * 30_000)
        failures = [e for e in snapshot_events() if e["kind"] == "compaction_failed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["error"], "ValueError")

    def test_malformed_numeric_env_falls_back_to_defaults(self) -> None:
        self._set_env("FW_TRAJECTORY_MAX_BYTES", "28k")
        self._set_env("FW_OFFLOAD_HOT_MAX_BYTES", "-5")
        self._set_env("FW_MAX_FORCED_REPLANS", "two")
        self.assertEqual(packed_target_bytes_from_env(), PACKED_TARGET_BYTES)
        self.assertEqual(hot_handle_max_bytes_from_env(), HOT_HANDLE_MAX_BYTES)
        self.assertEqual(max_forced_replans_from_env(), MAX_FORCED_REPLANS)
        trajectory = {
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show"},
            "observation_0": "holder\n" + "x" * 30_000,
        }
        for index in range(1, 6):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = f"small-{index}"
        decisions = compact_trajectory(
            trajectory, scope=self.scope, selected_archive=self.archive
        )
        self.assertEqual(decisions[0]["action"], "offloaded")

    def test_unwritable_event_log_does_not_raise(self) -> None:
        blocker = Path(self.tempdir.name) / "not-a-directory"
        blocker.write_text("occupied", encoding="utf-8")
        self._set_env("FW_OFFLOAD_EVENTS", str(blocker / "events.jsonl"))
        record_event({"kind": "probe"})
        record_event({"kind": "probe-again"})
        self.assertEqual(
            [e["kind"] for e in snapshot_events()], ["probe", "probe-again"]
        )


class AgentConstruction(unittest.TestCase):
    """Review nit: the wrapper discarded a fully built ReAct and rebuilt it."""

    class Signature(dspy.Signature):
        user_query: str = dspy.InputField()
        answer: str = dspy.OutputField()

    def setUp(self) -> None:
        reset_runtime_state()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self._set_env(HANDLE_ARCHIVE_ENV, str(Path(self.tempdir.name) / "handles.sqlite3"))

    def _set_env(self, name: str, value: str) -> None:
        previous = os.environ.get(name)

        def restore() -> None:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

        self.addCleanup(restore)
        os.environ[name] = value

    @staticmethod
    def noop_tool(command: str) -> str:
        """Return the command unchanged."""
        return command

    def test_enabled_builds_one_continuation_agent_with_search_memory(self) -> None:
        self._set_env(ENABLED_ENV, "1")
        agent = build_tool_agent(
            SimpleNamespace(), self.Signature, [self.noop_tool], max_iters=3
        )
        self.assertIsInstance(agent, StructuredContinuationReAct)
        self.assertEqual(set(agent.tools), {"noop_tool", "search_memory", "finish"})
        self.assertEqual(agent.max_iters, 3)
        installed = [e for e in snapshot_events() if e["kind"] == "agent_installed"]
        self.assertEqual(len(installed), 1)

    def test_disabled_builds_a_stock_react_without_search_memory(self) -> None:
        self._set_env(ENABLED_ENV, "0")
        agent = build_tool_agent(
            SimpleNamespace(), self.Signature, [self.noop_tool], max_iters=3
        )
        self.assertIsInstance(agent, fastWorkflowReAct)
        self.assertNotIsInstance(agent, StructuredContinuationReAct)
        self.assertEqual(set(agent.tools), {"noop_tool", "finish"})


class PlannerFailure(unittest.TestCase):
    """Review nit: a planner LLM error at the segment limit aborted the turn."""

    def setUp(self) -> None:
        reset_runtime_state()

    def test_planner_error_degrades_to_the_default_plan(self) -> None:
        agent = StructuredContinuationReAct.__new__(StructuredContinuationReAct)
        agent.max_iters = 25
        agent.forced_replans = 0
        agent.iteration_counter = 25
        agent.current_trajectory = {}
        trajectory = {
            "thought_0": "Need evidence",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "show_holders"},
            "observation_0": "x" * 3_000,
        }
        # No LM configured: dspy.Predict raises for real, no stubbing needed.
        with dspy.context(lm=None):
            agent._force_replan(trajectory, {"user_query": "audit"})
        self.assertEqual(agent.forced_replans, 1)
        self.assertEqual(agent.iteration_counter, 0)
        self.assertIn(DEFAULT_CONTINUATION_PLAN, trajectory["replan_1"])
        self.assertIn("segment 2 of 3", trajectory["replan_1"])
        events = [e for e in snapshot_events() if e["kind"] == "forced_replan"]
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["planner_error"])
        self.assertEqual(events[0]["plan"], "")
