"""ido-pyw.1: one input, every byte budget derived from it.

Offline only. Nothing here calls a model; the one litellm lookup the module can
perform is exercised through a stub and through the real table read for the
calibration model, which is a local dictionary in litellm.

The claim under test is the calibration identity: at the reference window --
131,072 tokens, which is what ``litellm`` reports as ``max_input_tokens`` for
``cerebras/gpt-oss-120b``, the main agent model of every accepted result-search
run -- the derived budgets ARE the values the accepted stack ran with. A change
to a fraction that would move a measured value fails here.
"""
from __future__ import annotations

import os
import unittest
from fractions import Fraction
from unittest import mock

from fastworkflow import context_budget as cb


class Calibration(unittest.TestCase):
    """The accepted stack's pinned values, reproduced from the window."""

    PINNED = {
        "trajectory_max_bytes": 28_000,
        "answer_rehydration_max_bytes": 250_000,
        "result_page_max_bytes": 3_072,
        "search_answer_max_bytes": 3_072,
        "offload_hot_max_bytes": 262_144,
        "result_handle_hot_max_bytes": 262_144,
        "offload_min_saving_bytes": 1_024,
    }

    def test_the_reference_window_is_the_gpt_oss_120b_window(self) -> None:
        self.assertEqual(cb.REFERENCE_WINDOW_TOKENS, 131_072)
        self.assertEqual(cb.BYTES_PER_TOKEN, 4)
        self.assertEqual(cb.REFERENCE_WINDOW_BYTES, 524_288)

    def test_litellm_still_reports_that_window_for_the_model(self) -> None:
        """If the metadata moves, the calibration reference has to be restated
        deliberately rather than drift. A litellm without the model in its table
        is not a failure of this repository, so it skips."""
        try:
            import litellm

            info = litellm.get_model_info("cerebras/gpt-oss-120b") or {}
        except Exception as error:  # noqa: BLE001
            self.skipTest(f"litellm has no entry for the model: {error}")
        self.assertEqual(info.get("max_input_tokens"), cb.REFERENCE_WINDOW_TOKENS)

    def test_every_budget_at_the_reference_window_is_its_pinned_value(self) -> None:
        for spec in cb.BUDGETS:
            with self.subTest(spec.name):
                self.assertEqual(spec.reference_bytes, self.PINNED[spec.name])

    def test_the_pinned_set_is_the_whole_set(self) -> None:
        """No budget is derived without being calibrated, and none is calibrated
        without being derived."""
        self.assertEqual({spec.name for spec in cb.BUDGETS}, set(self.PINNED))

    def test_the_fractions_are_exact_rationals(self) -> None:
        """Floats would make "exactly 28,000" a rounding accident."""
        for spec in cb.BUDGETS:
            with self.subTest(spec.name):
                self.assertIsInstance(spec.fraction, Fraction)
                self.assertEqual(
                    spec.fraction * cb.REFERENCE_WINDOW_BYTES,
                    self.PINNED[spec.name],
                )


class Scaling(unittest.TestCase):
    """A smaller and a larger window move every budget in proportion."""

    def test_half_the_window_is_half_of_every_budget(self) -> None:
        for spec in cb.BUDGETS:
            with self.subTest(spec.name):
                self.assertEqual(
                    spec.bytes_for(cb.REFERENCE_WINDOW_TOKENS // 2),
                    spec.reference_bytes // 2,
                )

    def test_double_the_window_is_double_every_budget(self) -> None:
        for spec in cb.BUDGETS:
            with self.subTest(spec.name):
                self.assertEqual(
                    spec.bytes_for(cb.REFERENCE_WINDOW_TOKENS * 2),
                    spec.reference_bytes * 2,
                )

    def test_a_tiny_window_never_falls_below_a_budget_s_floor(self) -> None:
        """A page budget of 192 bytes is a header and nothing else, so each
        budget keeps the minimum its own module always refused to go under."""
        for spec in cb.BUDGETS:
            with self.subTest(spec.name):
                self.assertGreaterEqual(
                    spec.bytes_for(cb.MIN_WINDOW_TOKENS), spec.floor)


class Resolution(unittest.TestCase):
    """Where the one number comes from, in order."""

    def setUp(self) -> None:
        cb.reset_cache()
        self.addCleanup(cb.reset_cache)
        self.saved = {
            name: os.environ.pop(name, None)
            for name in (cb.MODEL_CONTEXT_TOKENS_ENV, cb.AGENT_MODEL_ENV)
        }
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_the_setting_wins(self) -> None:
        os.environ[cb.MODEL_CONTEXT_TOKENS_ENV] = "200000"
        os.environ[cb.AGENT_MODEL_ENV] = "cerebras/gpt-oss-120b"
        self.assertEqual(cb.context_window_tokens(), (200_000, cb.SOURCE_SETTING))

    def test_the_env_file_is_read_before_the_process(self) -> None:
        import fastworkflow

        with mock.patch.dict(fastworkflow._env_vars,
                             {cb.MODEL_CONTEXT_TOKENS_ENV: "65536"}, clear=False):
            self.assertEqual(
                cb.context_window_tokens(), (65_536, cb.SOURCE_SETTING))

    def test_the_model_metadata_answers_when_nothing_is_set(self) -> None:
        os.environ[cb.AGENT_MODEL_ENV] = "a-made-up/model"
        with mock.patch.object(cb, "_model_window_tokens", return_value=65_536):
            tokens, source = cb.context_window_tokens()
        self.assertEqual(tokens, 65_536)
        self.assertEqual(source, f"{cb.SOURCE_MODEL_METADATA}:a-made-up/model")

    def test_an_unmapped_model_falls_back_and_says_so(self) -> None:
        os.environ[cb.AGENT_MODEL_ENV] = "a-provider/a-model-litellm-never-heard-of"
        self.assertEqual(
            cb.context_window_tokens(),
            (cb.REFERENCE_WINDOW_TOKENS, cb.SOURCE_FALLBACK),
        )

    def test_no_model_at_all_falls_back(self) -> None:
        self.assertEqual(
            cb.context_window_tokens(),
            (cb.REFERENCE_WINDOW_TOKENS, cb.SOURCE_FALLBACK),
        )

    def test_a_nonsense_setting_is_refused_not_obeyed(self) -> None:
        for raw in ("not-a-number", "12"):
            with self.subTest(raw):
                os.environ[cb.MODEL_CONTEXT_TOKENS_ENV] = raw
                self.assertEqual(
                    cb.context_window_tokens(),
                    (cb.REFERENCE_WINDOW_TOKENS, cb.SOURCE_FALLBACK),
                )

    def test_the_metadata_lookup_is_answered_once_per_model(self) -> None:
        os.environ[cb.AGENT_MODEL_ENV] = "cerebras/gpt-oss-120b"
        with mock.patch("litellm.get_model_info",
                        return_value={"max_input_tokens": 131_072}) as lookup:
            cb.context_window_tokens()
            cb.context_window_tokens()
        self.assertEqual(lookup.call_count, 1)


class Overrides(unittest.TestCase):
    """The per-budget names are tuning, not the interface."""

    def setUp(self) -> None:
        cb.reset_cache()
        self.addCleanup(cb.reset_cache)
        self.names = [spec.override_env for spec in cb.BUDGETS] + [
            cb.MODEL_CONTEXT_TOKENS_ENV, cb.AGENT_MODEL_ENV]
        self.saved = {name: os.environ.pop(name, None) for name in self.names}
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_an_override_wins_over_the_derived_budget(self) -> None:
        os.environ[cb.TRAJECTORY.override_env] = "12345"
        self.assertEqual(cb.trajectory_max_bytes(), 12_345)
        self.assertEqual(cb.result_page_max_bytes(), 3_072)  # the rest unmoved

    def test_an_override_below_the_floor_is_refused(self) -> None:
        os.environ[cb.SEARCH_ANSWER.override_env] = "16"
        self.assertEqual(cb.search_answer_max_bytes(), 3_072)

    def test_an_unparseable_override_is_refused(self) -> None:
        os.environ[cb.RESULT_PAGE.override_env] = "3k"
        self.assertEqual(cb.result_page_max_bytes(), 3_072)

    def test_an_override_survives_a_change_of_window(self) -> None:
        os.environ[cb.MODEL_CONTEXT_TOKENS_ENV] = str(cb.REFERENCE_WINDOW_TOKENS * 2)
        os.environ[cb.TRAJECTORY.override_env] = "12345"
        self.assertEqual(cb.trajectory_max_bytes(), 12_345)
        self.assertEqual(cb.answer_rehydration_max_bytes(), 500_000)


class Provenance(unittest.TestCase):
    """The one record a runner files (ido-pyw.2 wires it)."""

    def setUp(self) -> None:
        cb.reset_cache()
        self.addCleanup(cb.reset_cache)
        self.names = [spec.override_env for spec in cb.BUDGETS] + [
            cb.MODEL_CONTEXT_TOKENS_ENV, cb.AGENT_MODEL_ENV]
        self.saved = {name: os.environ.pop(name, None) for name in self.names}
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_it_names_the_input_its_source_and_every_budget(self) -> None:
        record = cb.budget_provenance()
        self.assertEqual(record["context_window_tokens"], cb.REFERENCE_WINDOW_TOKENS)
        self.assertEqual(record["context_window_source"], cb.SOURCE_FALLBACK)
        self.assertEqual(record["bytes_per_token"], 4)
        self.assertEqual(record["context_window_bytes"], 524_288)
        self.assertEqual(record["budgets"], Calibration.PINNED)
        self.assertEqual(record["overrides"], {})

    def test_an_override_is_named_as_one(self) -> None:
        os.environ[cb.TRAJECTORY.override_env] = "12345"
        record = cb.budget_provenance()
        self.assertEqual(record["budgets"]["trajectory_max_bytes"], 12_345)
        self.assertEqual(record["overrides"], {cb.TRAJECTORY.override_env: 12_345})

    def test_the_source_is_recorded_for_a_setting(self) -> None:
        os.environ[cb.MODEL_CONTEXT_TOKENS_ENV] = "262144"
        record = cb.budget_provenance()
        self.assertEqual(record["context_window_source"], cb.SOURCE_SETTING)
        self.assertEqual(record["budgets"]["trajectory_max_bytes"], 56_000)

    def test_it_is_json_serialisable(self) -> None:
        import json

        self.assertIsInstance(json.dumps(cb.budget_provenance()), str)


class ModulesReadTheSameBudgets(unittest.TestCase):
    """The named defaults each module exports are the derived values."""

    def test_every_module_default_is_its_budget(self) -> None:
        from fastworkflow import answer_rehydration, result_handles
        from fastworkflow.observation_offloading import compact, continuation, search

        self.assertEqual(compact.PACKED_TARGET_BYTES, 28_000)
        self.assertEqual(compact.MIN_OFFLOAD_SAVING_BYTES, 1_024)
        self.assertEqual(continuation.REPLAN_OBSERVATION_MAX_BYTES, 28_000)
        self.assertEqual(search.SEARCH_ANSWER_MAX_BYTES, 3_072)
        self.assertEqual(result_handles.RESULT_PAGE_MAX_BYTES, 3_072)
        self.assertEqual(result_handles.HOT_ROWS_MAX_BYTES, 262_144)
        self.assertEqual(answer_rehydration.DEFAULT_MAX_BYTES, 250_000)

    def test_every_module_reader_is_the_budget_function(self) -> None:
        from fastworkflow import answer_rehydration, result_handles
        from fastworkflow.observation_offloading import compact, search, state

        self.assertEqual(compact.packed_target_bytes_from_env(), 28_000)
        self.assertEqual(compact.min_offload_saving_bytes_from_env(), 1_024)
        self.assertEqual(search.search_answer_max_bytes_from_env(), 3_072)
        self.assertEqual(state.hot_handle_max_bytes_from_env(), 262_144)
        self.assertEqual(result_handles.page_max_bytes_from_env(), 3_072)
        self.assertEqual(result_handles.hot_rows_max_bytes_from_env(), 262_144)
        self.assertEqual(answer_rehydration.max_bytes_from_env(), 250_000)


if __name__ == "__main__":
    unittest.main()
