"""Arm D observation offloading: compact, search_memory, continuation, span manifest."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastworkflow import tracing
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.continuation import (
    MAX_FORCED_REPLANS,
    REPLAN_OBSERVATION_MAX_BYTES,
    StructuredContinuationReAct,
    replan_trajectory_skeleton,
)
from fastworkflow.observation_offloading.labels import offload_label
from fastworkflow.observation_offloading.manifest import (
    classify_against_steps,
    install_span_policy,
    uninstall_span_policy,
)
from fastworkflow.observation_offloading.search import search_memory
from fastworkflow.observation_offloading.state import (
    clear_hot_handles,
    hot_payload_bytes,
    reset_runtime_state,
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
        self.assertIn("full saved observation offloaded", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_6"], "small-6")
        self.assertIn("O1", stored_handles(self.scope))

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
        self.assertIn("offloaded handles", answer)

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
        self.assertIn("full saved observation offloaded", trajectory["observation_0"])
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
            "restart answer",
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
    def test_greedy_28k_inlines_newest_first_and_never_exceeds_bound(self) -> None:
        trajectory = {}
        for index in range(4):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": f"find_{index}"}
            trajectory[f"observation_{index}"] = str(index) * 600
        skeleton, metadata = replan_trajectory_skeleton(trajectory, greedy_max_bytes=1_600)
        self.assertLessEqual(metadata["measured_bytes"], 1_600)
        self.assertEqual(metadata["inlined_aliases"], ["O3", "O4"])
        self.assertEqual(metadata["labeled_aliases"], ["O1", "O2"])
        self.assertEqual(skeleton["observation_3"], "3" * 600)
        self.assertIn("observation label only", skeleton["observation_1"])

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
        self.assertIn("observation label only", skeleton["observation_0"])

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
        self.assertNotIn("x" * 50, captured["trajectory_skeleton"])


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
