"""ido-8ps.22 / ido-8ps.23: the extractor is told what the run never retrieved.

Offline only. Nothing here starts a server, calls a model or touches a backend:
the archive and the result-handle store are real SQLite files in a temp dir, the
extract module is a recorder, and every assertion is about bytes already on disk
before the test begins.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from unittest import mock

import dspy

from fastworkflow import answer_coverage
from fastworkflow.answer_coverage import (
    ANSWER_COVERAGE_ENV,
    COVERAGE_KEY,
    INSTRUCTED_KINDS,
    aliased_executes,
    answer_coverage_enabled,
    build_statement,
    coverage_block,
    named_entities,
    normalise,
    post_check,
    request_text,
    retrieved_corpus,
    split_by_presence,
)
from fastworkflow.answer_rehydration import ANSWER_REHYDRATION_ENV
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.labels import alias_line, offload_label
from fastworkflow.observation_offloading.state import (
    record_context_clause,
    reset_runtime_state,
    snapshot_events,
)
from fastworkflow.result_handles import ResultHandleStore

from tests.test_answer_rehydration import (  # reuse: one fixture, one meaning
    Recorder,
    declaration_payload,
    page_record,
)

CARD = (
    "Two Active Directory rights keep coming back on this quarter's "
    "privileged-access exceptions: Active Directory_Cloud Administrator and "
    "Active Directory_Compliance Officer. Audit both wherever they appear — "
    "the right, the system that publishes it. Five people on this quarter's "
    "exception list need the same treatment — Alan Cooper, Alisha Ochoa, "
    "Anna Garcia, Barbara Sanchez and Brandon Miller. Separately, the control "
    "'Active contractor identities whom manager left' has an open finding. "
    "Christopher Hubbard is one of the people it names and leaves on Friday."
)


def scope_for(name: str) -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity=name, channel_id="c", experiment_id="exp-ido-8ps-22",
        task_id="task", attempt=1, turn_key="turn-1",
    )


class RequestSplit(unittest.TestCase):
    """The plan the planner appended is not part of the request."""

    def test_the_todo_list_is_cut_off(self) -> None:
        self.assertEqual(
            request_text("Find Alan Cooper.\n\nExecute these next steps:\n1. Zeno Ppp"),
            "Find Alan Cooper.",
        )

    def test_the_user_query_prefix_is_removed(self) -> None:
        self.assertEqual(request_text("User Query:\nFind Alan Cooper."),
                         "Find Alan Cooper.")

    def test_a_query_without_a_plan_is_its_own_request(self) -> None:
        self.assertEqual(request_text("Find Alan Cooper."), "Find Alan Cooper.")

    def test_a_planner_invention_never_becomes_a_named_item(self) -> None:
        names = [e.text for e in named_entities(
            request_text("Find Alan Cooper.\n\nExecute these next steps:\n"
                         "1. open Zeno Ppp then Quentin Rrr")
        )]
        self.assertEqual(names, ["Alan Cooper"])


class EntityExtraction(unittest.TestCase):
    """Deterministic spans: no model, no dictionary, no workflow lookup."""

    def test_the_card_yields_exactly_its_named_items(self) -> None:
        found = [(e.kind, e.text) for e in named_entities(CARD)]
        self.assertEqual(found, [
            ("name", "Active Directory"),
            ("name", "Active Directory_Cloud Administrator"),
            ("name", "Active Directory_Compliance Officer"),
            ("name", "Alan Cooper"),
            ("name", "Alisha Ochoa"),
            ("name", "Anna Garcia"),
            ("name", "Barbara Sanchez"),
            ("name", "Brandon Miller"),
            ("quoted", "Active contractor identities whom manager left"),
            ("name", "Christopher Hubbard"),
        ])

    def test_a_sentence_capital_is_dropped_only_when_two_tokens_remain(self) -> None:
        # "Two Active Directory rights" is about Active Directory.
        self.assertIn("Active Directory",
                      [e.text for e in named_entities("Two Active Directory rights.")])
        # "Christopher Hubbard is one of..." keeps both: a bare surname is worse.
        self.assertEqual([e.text for e in named_entities("Christopher Hubbard is here.")],
                         ["Christopher Hubbard"])

    def test_a_comma_separates_two_names(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities("we saw Alan Cooper, Anna Garcia today")],
            ["Alan Cooper", "Anna Garcia"],
        )

    def test_a_single_capitalised_token_is_not_a_name(self) -> None:
        self.assertEqual(named_entities("Separately, he leaves on Friday."), [])

    def test_an_apostrophe_does_not_open_a_quotation(self) -> None:
        kinds = {e.kind for e in named_entities("this quarter's list and that one's")}
        self.assertNotIn("quoted", kinds)

    def test_uids_and_addresses_are_named_items(self) -> None:
        found = {e.kind: e.text for e in named_entities(
            "open 28c5aeb5b64e4ac6c40c57b0235980e2 and mail a.cooper@example.com"
        )}
        self.assertEqual(found["uid"], "28c5aeb5b64e4ac6c40c57b0235980e2")
        self.assertEqual(found["email"], "a.cooper@example.com")

    def test_duplicates_collapse_on_the_normalised_form(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities("Alan Cooper met ALAN COOPER again.")],
            ["Alan Cooper"],
        )

    def test_only_names_uids_and_addresses_are_ever_instructed(self) -> None:
        self.assertEqual(INSTRUCTED_KINDS, frozenset({"name", "uid", "email"}))


class Presence(unittest.TestCase):
    """Normalisation and the context clause."""

    def test_case_and_whitespace_are_normalised(self) -> None:
        entities = named_entities("Find Alan Cooper.")
        observed, unobserved = split_by_presence(
            entities, normalise("ALAN\n  COOPER  holds the right")
        )
        self.assertEqual([e.text for e in observed], ["Alan Cooper"])
        self.assertEqual(unobserved, [])

    def test_nfkc_folds_a_full_width_spelling(self) -> None:
        entities = named_entities("Find Alan Cooper.")
        observed, _ = split_by_presence(
            entities, normalise("Ａｌａｎ Cooper"))
        self.assertEqual([e.text for e in observed], ["Alan Cooper"])

    def test_a_name_only_in_a_context_clause_counts_as_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reset_runtime_state()
            self.addCleanup(reset_runtime_state)
            scope = scope_for(directory)
            archive = RuntimeHandleArchive(os.path.join(directory, "obs.sqlite3"))
            text = "3 rights\nright-a\nright-b\nright-c"
            archive.persist(
                scope, alias="O1", offload_order=1, command_name="list_permissions",
                step_index=0, text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
            record_context_clause(scope, "O1", "Identity 28c5aeb5 Alan Cooper")
            haystack, aliases = retrieved_corpus(
                scope=scope, archive=archive, handle_store=None)
            self.assertEqual(aliases, ["O1"])
            self.assertIn(normalise("Alan Cooper"), haystack)

    def test_a_search_answer_is_not_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reset_runtime_state()
            self.addCleanup(reset_runtime_state)
            scope = scope_for(directory)
            archive = RuntimeHandleArchive(os.path.join(directory, "obs.sqlite3"))
            answer = "No row for Christopher Hubbard was found in O1."
            archive.persist(
                scope, alias="O1#a1", offload_order=0, command_name="search_memory",
                step_index=1, text=answer,
                text_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            )
            haystack, aliases = retrieved_corpus(
                scope=scope, archive=archive, handle_store=None)
            self.assertEqual(aliases, [])
            self.assertNotIn(normalise("Christopher Hubbard"), haystack)

    def test_stored_rows_behind_a_handle_count_as_retrieved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reset_runtime_state()
            self.addCleanup(reset_runtime_state)
            scope = scope_for(directory)
            path = os.path.join(directory, "obs.sqlite3")
            archive = RuntimeHandleArchive(path)
            store = ResultHandleStore(path)
            shown = "result_handle=O2 page 1 rows 1-1 of 2\nuid-1  Alan Cooper"
            archive.persist(
                scope, alias="O2", offload_order=2, command_name="show_holders",
                step_index=1, text=shown,
                text_sha256=hashlib.sha256(shown.encode("utf-8")).hexdigest(),
            )
            store.put_declaration(scope, "O2", declaration_payload(total=2))
            store.put_page(
                scope, alias="O2", query_scope="", start_offset=0, limit_requested=25,
                source="resolver", backend_total=2,
                record=page_record(["uid-1  Alan Cooper", "uid-2  Anna Garcia"]),
            )
            haystack, _ = retrieved_corpus(
                scope=scope, archive=archive, handle_store=store)
            # Anna Garcia is on page 1 of the store but not in the bounded text.
            self.assertNotIn("Anna Garcia", shown)
            self.assertIn(normalise("Anna Garcia"), haystack)


class Block(unittest.TestCase):
    """The exact wording, and the exhaustion branch."""

    def test_normal_ending_and_a_list(self) -> None:
        self.assertEqual(
            coverage_block(unobserved=["Christopher Hubbard"], exhausted=False, steps=9),
            'Coverage of this run: the loop ended normally. These named items from '
            'the request appear in no retrieved observation: Christopher Hubbard. '
            'For each of them report "not retrieved" and nothing else - no value, '
            'no unavailability, no absence. For items that appear, report only what '
            'the observations show.',
        )

    def test_exhausted_ending_names_the_step_count(self) -> None:
        block = coverage_block(unobserved=[], exhausted=True, steps=96)
        self.assertIn("the loop ended at the iteration limit after 96 steps", block)
        self.assertIn("appear in no retrieved observation: none.", block)

    def test_several_items_are_separated_deterministically(self) -> None:
        block = coverage_block(
            unobserved=["Alisha Ochoa", "Christopher Hubbard"],
            exhausted=False, steps=1,
        )
        self.assertIn("observation: Alisha Ochoa; Christopher Hubbard.", block)

    def test_the_same_input_always_gives_the_same_block(self) -> None:
        args = dict(unobserved=["A B", "C D"], exhausted=True, steps=3)
        self.assertEqual(coverage_block(**args), coverage_block(**args))


class Statement(unittest.TestCase):
    """build_statement: the copy, the ordering, and the completeness guard."""

    def trajectory(self) -> dict:
        return {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "find_identity <search>Christopher Hubbard</search>"},
            "observation_0": alias_line("O1") + "0 identities.",
        }

    def test_the_block_precedes_the_trajectory(self) -> None:
        copy, report = build_statement(
            self.trajectory(), user_query=CARD, exhausted=False,
            haystack=normalise("Alan Cooper Alisha Ochoa Anna Garcia Barbara Sanchez "
                               "Brandon Miller Active Directory_Cloud Administrator "
                               "Active Directory_Compliance Officer"),
        )
        self.assertEqual(list(copy)[0], COVERAGE_KEY)
        self.assertEqual(list(copy)[1:], list(self.trajectory()))
        self.assertEqual(report.unobserved, ["Christopher Hubbard"])

    def test_the_input_trajectory_is_never_mutated(self) -> None:
        trajectory = self.trajectory()
        before = dict(trajectory)
        build_statement(trajectory, user_query=CARD, exhausted=False,
                        haystack=normalise("nothing here"))
        self.assertEqual(trajectory, before)

    def test_a_quoted_phrase_is_measured_and_never_instructed(self) -> None:
        _, report = build_statement(
            self.trajectory(), user_query=CARD, exhausted=False,
            haystack=normalise("Christopher Hubbard Alan Cooper"),
        )
        self.assertNotIn("Active contractor identities whom manager left",
                         report.unobserved)
        self.assertIn("Active contractor identities whom manager left",
                      report.phrases_unmatched)
        self.assertNotIn("Active contractor", report.statement)

    def test_an_incomplete_archive_names_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reset_runtime_state()
            self.addCleanup(reset_runtime_state)
            scope = scope_for(directory)
            archive = RuntimeHandleArchive(os.path.join(directory, "obs.sqlite3"))
            _, report = build_statement(
                self.trajectory(), user_query=CARD, exhausted=False,
                scope=scope, archive=archive, handle_store=None,
            )
            self.assertFalse(report.complete)
            self.assertEqual(report.unobserved, [])
            self.assertIn("no archived observations", report.incomplete_reason)
            self.assertIn("observation: none.", report.statement)

    def test_an_archive_missing_one_alias_names_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reset_runtime_state()
            self.addCleanup(reset_runtime_state)
            scope = scope_for(directory)
            archive = RuntimeHandleArchive(os.path.join(directory, "obs.sqlite3"))
            text = "0 identities."
            archive.persist(
                scope, alias="O1", offload_order=1, command_name="find_identity",
                step_index=0, text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
            trajectory = self.trajectory()
            trajectory.update({
                "thought_1": "again", "tool_name_1": "execute_workflow_query",
                "tool_args_1": {"command": "list_controls"},
                "observation_1": alias_line("O2") + "101 controls",
            })
            _, report = build_statement(
                trajectory, user_query=CARD, exhausted=False,
                scope=scope, archive=archive, handle_store=None,
            )
            self.assertFalse(report.complete)
            self.assertIn("O2", report.incomplete_reason)
            self.assertEqual(report.unobserved, [])

    def test_the_exhaustion_sentence_needs_no_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reset_runtime_state()
            self.addCleanup(reset_runtime_state)
            archive = RuntimeHandleArchive(os.path.join(directory, "obs.sqlite3"))
            _, report = build_statement(
                self.trajectory(), user_query=CARD, exhausted=True,
                scope=scope_for(directory), archive=archive, handle_store=None,
            )
            self.assertTrue(report.exhausted)
            self.assertIn("the loop ended at the iteration limit after 1 steps",
                          report.statement)

    def test_aliased_executes_ignores_non_execute_steps(self) -> None:
        trajectory = {
            "tool_name_0": "search_memory",
            "observation_0": alias_line("O9") + "an answer",
            "tool_name_1": "execute_workflow_query",
            "observation_1": offload_label(
                alias="O4", command_name="show_holders", response="rows",
                description="the holders",
            ),
        }
        self.assertEqual(aliased_executes(trajectory), {"O4"})


class PostCheckCounts(unittest.TestCase):
    """Measurement only: it counts, it never gates."""

    def test_an_unavailability_claim_beside_a_named_item_is_counted(self) -> None:
        check = post_check(
            "Christopher Hubbard: the platform does not provide a finding "
            "description for him.",
            ["Christopher Hubbard"],
        )
        self.assertEqual(check.unavailability_claim_on_unobserved, 1)
        self.assertEqual(check.not_retrieved_on_unobserved, 0)
        self.assertEqual(check.unobserved_mentioned, 1)

    def test_not_retrieved_is_counted_separately(self) -> None:
        check = post_check("Christopher Hubbard: not retrieved.",
                           ["Christopher Hubbard"])
        self.assertEqual(check.not_retrieved_on_unobserved, 1)
        self.assertEqual(check.unavailability_claim_on_unobserved, 0)

    def test_an_item_the_answer_never_mentions_is_counted_silent(self) -> None:
        check = post_check("Alan Cooper holds the right.", ["Christopher Hubbard"])
        self.assertEqual(check.silent_on_unobserved, 1)
        self.assertEqual(check.unobserved_mentioned, 0)

    def test_a_distant_claim_is_not_attributed_to_the_item(self) -> None:
        answer = ("Christopher Hubbard appears in the roster. " + "filler. " * 60
                  + "The feed is not available.")
        self.assertEqual(
            post_check(answer, ["Christopher Hubbard"])
            .unavailability_claim_on_unobserved, 0)

    def test_nothing_is_recorded_for_an_empty_list(self) -> None:
        check = post_check("anything", [])
        self.assertEqual(check.unobserved_total, 0)
        self.assertEqual(check.unavailability_claim_on_unobserved, 0)


class ExtractHook(unittest.TestCase):
    """The flag, the byte-identical default, and the events."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for name in (ANSWER_COVERAGE_ENV, ANSWER_REHYDRATION_ENV):
            os.environ.pop(name, None)
        self.addCleanup(lambda: [os.environ.pop(name, None) for name in
                                 (ANSWER_COVERAGE_ENV, ANSWER_REHYDRATION_ENV)])
        from fastworkflow.utils.react import fastWorkflowReAct

        def a_tool(value: str) -> str:
            """A tool."""
            return value

        self.scope = scope_for(self.directory.name)
        path = os.path.join(self.directory.name, "obs.sqlite3")
        self.archive = RuntimeHandleArchive(path)
        self.store = ResultHandleStore(path)
        text = "1 identity.\nuid-1  Alan Cooper"
        self.archive.persist(
            self.scope, alias="O1", offload_order=1, command_name="find_identity",
            step_index=0, text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        self.agent = fastWorkflowReAct("user_query -> final_answer",
                                       tools=[a_tool], max_iters=2)
        self.agent.continuation_scope = self.scope
        self.agent.observation_archive = self.archive
        self.trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "find_identity"},
            "observation_0": alias_line("O1") + text,
        }

    def _patch_store(self):
        return mock.patch("fastworkflow.result_handles.store",
                          return_value=self.store)

    def test_the_flag_is_read_env_file_first(self) -> None:
        self.assertFalse(answer_coverage_enabled())
        with mock.patch.dict("fastworkflow._env_vars",
                             {ANSWER_COVERAGE_ENV: "1"}, clear=False):
            self.assertTrue(answer_coverage_enabled())
        os.environ[ANSWER_COVERAGE_ENV] = "yes"
        self.assertTrue(answer_coverage_enabled())

    def test_flag_off_is_byte_identical(self) -> None:
        expected = self.agent._format_trajectory(self.trajectory)
        recorder = Recorder()
        self.agent.extract = recorder
        self.agent._extract_prediction(self.trajectory, user_query=CARD)
        self.assertEqual(recorder.calls, [expected])
        self.assertEqual(snapshot_events(), [])

    def test_flag_on_prepends_the_block(self) -> None:
        os.environ[ANSWER_COVERAGE_ENV] = "1"
        recorder = Recorder()
        self.agent.extract = recorder
        with self._patch_store():
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        rendered = recorder.calls[0]
        self.assertIn("Coverage of this run:", rendered)
        self.assertIn("Christopher Hubbard", rendered)
        self.assertLess(rendered.index("Coverage of this run:"),
                        rendered.index("thought_0"))

    def test_the_loop_trajectory_is_unchanged(self) -> None:
        os.environ[ANSWER_COVERAGE_ENV] = "1"
        before = dict(self.trajectory)
        self.agent.extract = Recorder()
        with self._patch_store():
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        self.assertEqual(self.trajectory, before)

    def test_the_events_carry_the_measures(self) -> None:
        os.environ[ANSWER_COVERAGE_ENV] = "1"
        self.agent.extract = mock.Mock(
            return_value=dspy.Prediction(
                final_answer="Christopher Hubbard: no record is available."))
        with self._patch_store():
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        events = snapshot_events()
        statement = [e for e in events if e["kind"] == "coverage_statement"]
        check = [e for e in events if e["kind"] == "coverage_post_check"]
        self.assertEqual(len(statement), 1)
        self.assertEqual(len(check), 1)
        self.assertEqual(statement[0]["entities_total"], 10)
        self.assertIn("Christopher Hubbard", statement[0]["unobserved"])
        self.assertTrue(statement[0]["complete"])
        self.assertEqual(statement[0]["archived_observations"], 1)
        self.assertGreater(statement[0]["statement_bytes"], 0)
        self.assertEqual(check[0]["unavailability_claim_on_unobserved"], 1)

    def test_a_failure_falls_back_to_the_plain_call(self) -> None:
        os.environ[ANSWER_COVERAGE_ENV] = "1"
        expected = self.agent._format_trajectory(self.trajectory)
        recorder = Recorder()
        self.agent.extract = recorder
        with mock.patch.object(answer_coverage, "build_statement",
                               side_effect=RuntimeError("boom")):
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        self.assertEqual(recorder.calls, [expected])
        self.assertEqual([e["kind"] for e in snapshot_events()], ["coverage_failed"])

    def test_truncation_never_drops_the_block(self) -> None:
        os.environ[ANSWER_COVERAGE_ENV] = "1"
        copy, _ = build_statement(
            dict(self.trajectory), user_query=CARD, exhausted=False,
            haystack=normalise("nothing"),
        )
        truncated = self.agent.truncate_trajectory(copy)
        self.assertIn(COVERAGE_KEY, truncated)
        self.assertNotIn("thought_0", truncated)

    def test_it_runs_after_rehydration_on_the_rehydrated_copy(self) -> None:
        os.environ[ANSWER_COVERAGE_ENV] = "1"
        os.environ[ANSWER_REHYDRATION_ENV] = "1"
        label = offload_label(
            alias="O1", command_name="find_identity",
            response="1 identity.\nuid-1  Alan Cooper", description="the identity",
        )
        trajectory = dict(self.trajectory, observation_0=label)
        recorder = Recorder()
        self.agent.extract = recorder
        with self._patch_store():
            self.agent._extract_prediction(trajectory, user_query=CARD)
        rendered = recorder.calls[0]
        self.assertIn("Coverage of this run:", rendered)
        self.assertIn("uid-1  Alan Cooper", rendered)  # rehydration still ran
        kinds = [e["kind"] for e in snapshot_events()]
        self.assertLess(kinds.index("rehydration_started"),
                        kinds.index("coverage_statement"))


if __name__ == "__main__":
    unittest.main()
