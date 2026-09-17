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
    COVERAGE_KEY,
    EVIDENCE_HEAD,
    EVIDENCE_LIST_MAX_BYTES,
    INSTRUCTED_KINDS,
    NUDGE_MAX_BYTES,
    NUDGE_MIN_ITERS_LEFT,
    RETRIEVED_RULE,
    aliased_executes,
    build_nudge,
    build_statement,
    coverage_block,
    evidence_by_subject,
    evidence_sentence,
    issued_commands,
    named_entities,
    normalise,
    nudge_block,
    post_check,
    request_text,
    retrieved_corpus,
    split_by_presence,
    strip_query_echoes,
    subject_corpus,
)
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

    def test_a_possessive_ends_the_name_it_marks(self) -> None:
        # ido-jf6/F29. "Cooper's" ends in a letter, so the run used to carry on
        # into the next capital and name a thing no observation can contain.
        self.assertEqual(
            [e.text for e in named_entities(
                "List Alan Cooper's Active Directory rights and "
                "Brandon Miller's accounts."
            )],
            ["Alan Cooper", "Active Directory", "Brandon Miller"],
        )

    def test_a_curly_possessive_reads_the_same_as_a_straight_one(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities("Audit Barbara Sanchez’s permissions.")],
            ["Barbara Sanchez"],
        )
        self.assertEqual(
            [e.text for e in named_entities("open Alan Cooper’s Active Directory row")],
            ["Alan Cooper", "Active Directory"],
        )

    def test_a_plural_possessive_is_a_bare_apostrophe(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities(
                "Audit the Cooper Brothers' Active Directory rights.")],
            ["Cooper Brothers", "Active Directory"],
        )

    def test_an_apostrophe_inside_a_name_survives(self) -> None:
        # Only the possessive marker comes off; O'Brien is the name itself.
        self.assertEqual(
            [e.text for e in named_entities("Find Sean O'Brien and Anna Garcia.")],
            ["Sean O'Brien", "Anna Garcia"],
        )

    def test_a_sentence_initial_imperative_is_not_part_of_a_name(self) -> None:
        # ido-jf6/F29. "List Identities", "Show Accounts" and "Compare Alan"
        # were named items; the capital is grammar, not a handle.
        self.assertEqual(
            named_entities("List Identities whose manager left. Show Accounts for each."),
            [],
        )
        self.assertEqual(named_entities("Compare Alan and Brandon."), [])
        self.assertEqual(
            [e.text for e in named_entities("Show Alan Cooper's manager.")],
            ["Alan Cooper"],
        )

    def test_an_imperative_word_away_from_the_sentence_start_is_kept(self) -> None:
        # Only the FIRST token of a sentence is tested against the verb list.
        self.assertEqual(
            [e.text for e in named_entities("we saw Alan Cooper, Barbara List today")],
            ["Alan Cooper", "Barbara List"],
        )

    def test_a_sentence_capital_that_is_not_a_verb_is_still_kept(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities("Christopher Hubbard leaves on Friday.")],
            ["Christopher Hubbard"],
        )

    def test_a_slash_or_a_dash_separates_two_named_items(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities(
                "Audit Alan Cooper/Brandon Miller and Anna Garcia—contractor.")],
            ["Alan Cooper", "Brandon Miller", "Anna Garcia"],
        )
        self.assertEqual(
            [e.text for e in named_entities("Anna Garcia – contractor and Alan Cooper.")],
            ["Anna Garcia", "Alan Cooper"],
        )

    def test_an_ordinary_multi_word_name_is_unchanged(self) -> None:
        self.assertEqual(
            [e.text for e in named_entities(
                "we saw Alan Cooper and Active Directory_Cloud Administrator today")],
            ["Alan Cooper", "Active Directory_Cloud Administrator"],
        )

    def test_a_particle_or_an_initial_is_not_part_of_a_name(self) -> None:
        # Documented limit, unchanged by ido-jf6: a lowercase particle and the
        # full stop of an initial both end a run, so these name nobody. The
        # module never invents a handle it cannot spell from the request.
        self.assertEqual(
            named_entities("Check Maria de la Cruz and J. R. Smith and "
                           "Ludwig van der Berg."),
            [],
        )

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


class QueryEchoes(unittest.TestCase):
    """ido-mng: a miss that quotes the query back is not a retrieval.

    The contract is "a name the agent merely typed into a query can never make
    that name look retrieved". A filtered listing that matched nothing still
    prints its filter in the header, and a backend may word its miss with the
    literal in it, so the echo has to go before presence is decided.
    """

    ZERO_PAGE = (
        'result_handle=O2 filter="Christopher Hubbard" '
        "filter_columns=name page 1 rows 0 of 12 matched=0 materialized=12 "
        "total=12 source_complete=true matched_complete=true "
        "continuation=none outcome=complete has_more=false\n"
        'No rows matched the literal "Christopher Hubbard" in these fields: '
        "name. That is a complete zero for this literal in this listing; it is "
        "not evidence that the person or object does not exist."
    )

    def test_a_filtered_miss_does_not_retrieve_its_own_literal(self) -> None:
        entities = named_entities("Find Christopher Hubbard and list his rights.")
        observed, unobserved = split_by_presence(entities, normalise(self.ZERO_PAGE))
        self.assertEqual([e.text for e in observed], [])
        self.assertEqual([e.text for e in unobserved], ["Christopher Hubbard"])

    def test_a_backend_wording_of_the_miss_is_no_different(self) -> None:
        echo = ("Observation O4 (execute_workflow_query, in DirectoryExplorer)\n"
                "No identity matching 'Christopher Hubbard' was found.")
        entities = named_entities("Find Christopher Hubbard and list his rights.")
        observed, unobserved = split_by_presence(entities, normalise(echo))
        self.assertEqual([e.text for e in observed], [])
        self.assertEqual([e.text for e in unobserved], ["Christopher Hubbard"])

    def test_a_name_in_a_retrieved_row_stays_observed(self) -> None:
        page = (
            'result_handle=O2 filter="Alan Cooper" page 1 rows 1-1 of 1 '
            "matched=1\nuid-1  Alan Cooper  active"
        )
        entities = named_entities("Find Alan Cooper and list his rights.")
        observed, unobserved = split_by_presence(entities, normalise(page))
        self.assertEqual([e.text for e in observed], ["Alan Cooper"])
        self.assertEqual(unobserved, [])

    def test_only_the_echo_is_removed_never_the_line_around_it(self) -> None:
        """A row that merely contains a miss keeps every name it retrieved."""
        row = "uid-1  Alan Cooper  no manager found  active"
        entities = named_entities("Find Alan Cooper and list his rights.")
        observed, _ = split_by_presence(entities, normalise(row))
        self.assertEqual([e.text for e in observed], ["Alan Cooper"])
        self.assertEqual(strip_query_echoes(row), row)

    def test_the_block_then_allows_the_only_true_statement(self) -> None:
        query = "Find Christopher Hubbard and list his rights."
        copy, report = build_statement(
            {}, user_query=query, exhausted=False,
            haystack=normalise(self.ZERO_PAGE),
        )
        self.assertEqual(report.unobserved, ["Christopher Hubbard"])
        self.assertEqual(report.observed_named, [])
        block = copy[COVERAGE_KEY]
        self.assertIn(
            "appear in no retrieved observation: Christopher Hubbard.", block)
        self.assertNotIn(
            "DO appear in this run's observations: Christopher Hubbard", block)


class Block(unittest.TestCase):
    """The exact wording, and the exhaustion branch."""

    def test_normal_ending_and_a_list(self) -> None:
        self.assertEqual(
            coverage_block(unobserved=["Christopher Hubbard"], exhausted=False, steps=9),
            'Coverage of this run: the loop ended normally. These named items from '
            'the request appear in no retrieved observation: Christopher Hubbard. '
            'For each of them report "not retrieved" and nothing else - no value, '
            'no unavailability, no absence. '
            'Every other named item of the request WAS retrieved: it appears in '
            "this run's observations and must be reported from them. "
            'Do not write "not retrieved", "not available", "no data", or any '
            'other statement of absence about an item that is not named in the '
            'unobserved list above. '
            'For items that appear, report only what the observations show.',
        )

    # ------------------------------------------------------------- ido-8ps.24
    def test_the_retrieved_rule_is_present_with_or_without_an_observed_list(
        self,
    ) -> None:
        for observed in ([], ["Alan Cooper"]):
            with self.subTest(observed=observed):
                block = coverage_block(unobserved=["Christopher Hubbard"],
                                       exhausted=False, steps=9,
                                       observed=observed)
                self.assertIn("WAS retrieved", block)
                self.assertIn('Do not write "not retrieved", "not available"',
                              block)

    def test_observed_items_are_named_back(self) -> None:
        block = coverage_block(unobserved=["Christopher Hubbard"],
                               exhausted=False, steps=9,
                               observed=["Alan Cooper", "Brandon Miller"])
        self.assertIn(
            "These named items of the request DO appear in this run's "
            "observations: Alan Cooper; Brandon Miller.",
            block,
        )

    def test_an_observed_name_is_never_also_in_the_unobserved_list(self) -> None:
        block = coverage_block(unobserved=["Christopher Hubbard"],
                               exhausted=False, steps=9,
                               observed=["Alan Cooper"])
        head, _, tail = block.partition("DO appear")
        self.assertIn("Christopher Hubbard", head)
        self.assertNotIn("Christopher Hubbard", tail)

    def test_duplicate_observed_names_are_listed_once(self) -> None:
        block = coverage_block(unobserved=[], exhausted=False, steps=1,
                               observed=["Alan Cooper", "Alan Cooper"])
        self.assertEqual(block.count("Alan Cooper"), 1)

    def test_an_oversized_observed_list_falls_back_to_the_rule_alone(self) -> None:
        names = ["Person %04d" % index for index in range(200)]
        block = coverage_block(unobserved=[], exhausted=False, steps=1,
                               observed=names)
        self.assertNotIn("DO appear in this run's observations", block)
        self.assertIn("WAS retrieved", block)
        self.assertLess(len(block.encode("utf-8")), 1024)

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

    def test_a_possessive_request_instructs_no_false_absence(self) -> None:
        # ido-jf6/F29. The run opened Alan Cooper; the block used to name
        # "Alan Cooper's" as an item no observation contains.
        _, report = build_statement(
            self.trajectory(),
            user_query="List Alan Cooper's rights and Brandon Miller's accounts.",
            exhausted=False,
            haystack=normalise("identity 28c5 alan cooper 29 permissions "
                               "identity 9a1 brandon miller 3 accounts"),
        )
        self.assertEqual(report.unobserved, [])
        self.assertEqual(report.observed, ["Alan Cooper", "Brandon Miller"])

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

    # ------------------------------------------------------------- ido-8ps.24
    def test_absence_phrasing_about_a_retrieved_item_is_counted(self) -> None:
        check = post_check(
            "Alan Cooper: not retrieved. " + "filler. " * 60
            + "Brandon Miller: no data is available.",
            [],
            ["Alan Cooper", "Brandon Miller"],
        )
        self.assertEqual(check.observed_total, 2)
        self.assertEqual(check.observed_mentioned, 2)
        self.assertEqual(check.not_retrieved_on_observed, 1)
        self.assertEqual(check.unavailability_claim_on_observed, 1)
        self.assertEqual(check.misuse_on_observed, 2)

    def test_a_clean_report_of_a_retrieved_item_counts_nothing(self) -> None:
        check = post_check("Alan Cooper holds Cloud Administrator.", [],
                           ["Alan Cooper"])
        self.assertEqual(check.observed_mentioned, 1)
        self.assertEqual(check.misuse_on_observed, 0)

    def test_the_observed_direction_is_off_unless_it_is_asked_for(self) -> None:
        check = post_check("Alan Cooper: not retrieved.", [])
        self.assertEqual(check.observed_total, 0)
        self.assertEqual(check.misuse_on_observed, 0)

    def test_both_directions_are_in_the_event(self) -> None:
        event = post_check("Alan Cooper: not available.", ["X Y"],
                           ["Alan Cooper"]).as_event()
        self.assertEqual(event["misuse_on_observed"], 1)
        self.assertEqual(event["observed_total"], 1)
        self.assertIn("per_observed_item", event)


class ExtractHook(unittest.TestCase):
    """What the extract call receives, and the events it leaves behind.

    ``ido-pyw.1`` removed ``FW_ANSWER_COVERAGE``: the statement is what an
    answer-time extract call gets, so the tests that pinned the flag-off call
    against ``4832b3c`` are gone and these assert the behaviour instead.
    """

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
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

    def test_the_block_is_prepended_with_nothing_set(self) -> None:
        """ido-pyw.1: no setting, no override -- the block is simply there."""
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
        before = dict(self.trajectory)
        self.agent.extract = Recorder()
        with self._patch_store():
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        self.assertEqual(self.trajectory, before)

    def test_the_events_carry_the_measures(self) -> None:
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

    def test_the_post_check_receives_the_unavailable_split(self) -> None:
        """ido-8ps.30. `CoverageReport.unavailable` is the subset of the
        unobserved items the run DID name in a command (ido-8ps.27 (b)). It was
        computed on the report and never passed to `post_check`, so every
        recorded `coverage_post_check` event said `unavailable_total: 0` -- and
        the three measures beside it, which only ever count over that subset,
        were zero by construction rather than by observation.

        The trajectory here names Christopher Hubbard in a command and retrieves
        nothing for him, which is exactly the attempted-and-empty case."""
        trajectory = dict(
            self.trajectory,
            tool_args_0={"command": "find_identity <name>Christopher Hubbard</name>"},
        )
        self.agent.extract = mock.Mock(
            return_value=dspy.Prediction(
                final_answer="Christopher Hubbard: no record is available."))
        with self._patch_store():
            self.agent._extract_prediction(trajectory, user_query=CARD)
        statement = [e for e in snapshot_events()
                     if e["kind"] == "coverage_statement"][0]
        check = [e for e in snapshot_events()
                 if e["kind"] == "coverage_post_check"][0]
        self.assertIn("Christopher Hubbard", statement["unavailable"])
        self.assertEqual(check["unavailable_total"], len(statement["unavailable"]))
        self.assertGreater(check["unavailable_total"], 0)
        self.assertEqual(check["unavailable_mentioned"], 1)
        self.assertEqual(check["unavailability_claim_on_unavailable"], 1)

    def test_an_item_the_run_never_asked_for_is_not_in_the_unavailable_count(self) -> None:
        """The other half of ido-8ps.27 (b): never-attempted items stay out of
        the unavailable measure, so the split is a split and not a rename."""
        self.agent.extract = mock.Mock(
            return_value=dspy.Prediction(
                final_answer="Christopher Hubbard: no record is available."))
        with self._patch_store():
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        statement = [e for e in snapshot_events()
                     if e["kind"] == "coverage_statement"][0]
        check = [e for e in snapshot_events()
                 if e["kind"] == "coverage_post_check"][0]
        self.assertEqual(statement["unavailable"], [])
        self.assertEqual(check["unavailable_total"], 0)
        self.assertEqual(check["unavailability_claim_on_unobserved"], 1)

    def test_a_subject_with_nothing_to_state_leaves_the_sentence_out(self) -> None:
        """ido-8ps.28 / ido-pyw.1. The sentence is computed unconditionally, and
        a run whose only subject has nothing to state still renders the block
        the accepted stack rendered at 90a1565 -- the positive-half-only rule,
        not a flag, is what keeps it silent."""
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        recorder = Recorder()
        self.agent.extract = recorder
        with self._patch_store():
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        self.assertNotIn(EVIDENCE_HEAD, recorder.calls[0])
        self.assertIn(STATEMENT_AT_90A1565, recorder.calls[0])

    def test_the_evidence_sentence_reaches_the_extract_call(self) -> None:
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        text = "Active Directory_Compliance Officer"
        self.archive.persist(
            self.scope, alias="O2", offload_order=2,
            command_name="list_entitlements", step_index=1, text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        record_context_clause(self.scope, "O2", "Identity 3f22  Alisha Ochoa")
        self.agent.extract = mock.Mock(
            return_value=dspy.Prediction(
                final_answer="| Alisha Ochoa | Active Directory_Cloud "
                             "Administrator |"))
        with self._patch_store():
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        events = snapshot_events()
        statement = [e for e in events if e["kind"] == "coverage_statement"][0]
        check = [e for e in events if e["kind"] == "coverage_post_check"][0]
        self.assertEqual(statement["evidence_named"], ["Alisha Ochoa"])
        self.assertGreater(statement["evidence_bytes"], 0)
        self.assertEqual(check["evidence_claims_unlisted"], 1)
        self.assertEqual(
            check["per_evidence_subject"][-1]["claimed_unlisted"],
            ["Active Directory_Cloud Administrator"],
        )

    def test_a_failure_costs_the_block_and_not_the_answer(self) -> None:
        """The extract call still happens, with no coverage block in it."""
        recorder = Recorder()
        self.agent.extract = recorder
        with mock.patch.object(answer_coverage, "build_statement",
                               side_effect=RuntimeError("boom")):
            self.agent._extract_prediction(self.trajectory, user_query=CARD)
        self.assertEqual(len(recorder.calls), 1)
        self.assertNotIn("Coverage of this run:", recorder.calls[0])
        self.assertIn("coverage_failed",
                      [e["kind"] for e in snapshot_events()])

    def test_truncation_never_drops_the_block(self) -> None:
        copy, _ = build_statement(
            dict(self.trajectory), user_query=CARD, exhausted=False,
            haystack=normalise("nothing"),
        )
        truncated = self.agent.truncate_trajectory(copy)
        self.assertIn(COVERAGE_KEY, truncated)
        self.assertNotIn("thought_0", truncated)

    def test_it_runs_after_rehydration_on_the_rehydrated_copy(self) -> None:
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


class WhyAnItemIsMissing(unittest.TestCase):
    """ido-8ps.27 (b): never attempted is not the same fact as unavailable."""

    def test_a_name_in_a_command_is_found(self) -> None:
        issued = issued_commands(
            {"tool_args_0": {"command": "find_identity <name>Anna Garcia</name>"},
             "tool_args_1": "fetch_result_page O5"}
        )
        self.assertIn(normalise("Anna Garcia"), issued)
        self.assertIn(normalise("fetch_result_page"), issued)

    def test_an_observation_is_not_a_command(self) -> None:
        issued = issued_commands(
            {"observation_0": "1 identity. uid-1 Anna Garcia",
             "thought_0": "look for Anna Garcia"}
        )
        self.assertEqual(issued, "")

    def test_the_block_is_unchanged_when_nothing_was_attempted(self) -> None:
        self.assertEqual(
            coverage_block(unobserved=["Anna Garcia"], exhausted=False, steps=3),
            coverage_block(unobserved=["Anna Garcia"], exhausted=False, steps=3,
                           unavailable=[]),
        )

    def test_the_two_kinds_are_named_apart(self) -> None:
        block = coverage_block(
            unobserved=["Anna Garcia", "Christopher Hubbard"],
            exhausted=False, steps=9, unavailable=["Christopher Hubbard"],
        )
        self.assertIn("WERE attempted and the attempt returned nothing about "
                      "them: Christopher Hubbard", block)
        self.assertIn("never attempted - no command of this run named them: "
                      "Anna Garcia", block)
        self.assertIn('report "not retrieved" and nothing else', block)

    def test_an_unavailable_item_outside_the_list_is_ignored(self) -> None:
        block = coverage_block(unobserved=["Anna Garcia"], exhausted=False,
                               steps=1, unavailable=["Someone Else"])
        self.assertNotIn("WERE attempted", block)

    def test_every_missing_item_may_be_the_attempted_kind(self) -> None:
        block = coverage_block(unobserved=["Anna Garcia"], exhausted=False,
                               steps=1, unavailable=["Anna Garcia"])
        self.assertIn("The rest were never attempted - no command of this run "
                      "named them: none.", block)

    def test_build_statement_splits_from_the_trajectory(self) -> None:
        trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "find_identity <name>Anna Garcia</name>"},
            "observation_0": "complete-zero: no rows",
        }
        _, report = build_statement(
            trajectory, user_query=CARD, exhausted=False,
            haystack=normalise("nothing was retrieved"),
        )
        self.assertIn("Anna Garcia", report.unavailable)
        self.assertIn("Alan Cooper", report.never_attempted)
        self.assertEqual(
            sorted(report.unavailable + report.never_attempted),
            sorted(report.unobserved),
        )
        self.assertIn("WERE attempted", report.statement)

    def test_the_post_check_counts_the_two_kinds_separately(self) -> None:
        check = post_check(
            "Anna Garcia: not retrieved. Alan Cooper: no record is available.",
            ["Anna Garcia", "Alan Cooper"],
            (),
            ["Anna Garcia"],
        )
        self.assertEqual(check.unavailable_total, 1)
        self.assertEqual(check.never_attempted_total, 1)
        self.assertEqual(check.not_retrieved_on_unavailable, 1)
        self.assertEqual(check.unavailability_claim_on_never_attempted, 1)
        self.assertEqual(
            check.unavailable_total + check.never_attempted_total,
            check.unobserved_total,
        )
        self.assertEqual(
            [detail["kind"] for detail in check.details],
            ["unavailable", "never_attempted"],
        )

    def test_the_kinds_are_in_the_event(self) -> None:
        check = post_check("nothing", ["Anna Garcia"], (), ["Anna Garcia"])
        event = check.as_event()
        self.assertEqual(event["unavailable_total"], 1)
        self.assertEqual(event["never_attempted_total"], 0)


class SubjectOfACommand(unittest.TestCase):
    """ido-8ps.27: the context clause, not the corpus, answers "did you go there"."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scope = scope_for(self.directory.name)
        self.archive = RuntimeHandleArchive(
            os.path.join(self.directory.name, "obs.sqlite3"))

    def _archive(self, alias: str, text: str, clause: str = "") -> None:
        self.archive.persist(
            self.scope, alias=alias, offload_order=int(alias[1:]),
            command_name="find_identity", step_index=int(alias[1:]), text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        record_context_clause(self.scope, alias, clause)

    def test_a_listing_row_is_not_a_subject(self) -> None:
        self._archive("O1", "5 identities.\nuid-2  Anna Garcia", "DirectoryExplorer")
        haystack = subject_corpus(scope=self.scope, archive=self.archive)
        self.assertNotIn(normalise("Anna Garcia"), haystack)

    def test_a_context_instance_is_a_subject(self) -> None:
        self._archive("O1", "3 accounts.", "Identity 3f22  Anna Garcia")
        haystack = subject_corpus(scope=self.scope, archive=self.archive)
        self.assertIn(normalise("Anna Garcia"), haystack)

    def _rights(self) -> None:
        """The two rights and their application, opened as every attempt does."""
        self._archive("O8", "477 holders.",
                      "Permission 85cd  Active Directory_Cloud Administrator")
        self._archive("O9", "513 holders.",
                      "Permission 3e3d  Active Directory_Compliance Officer")

    def test_the_nudge_names_the_people_the_run_never_opened(self) -> None:
        self._archive("O1", "1 identity.\nuid-1  Alan Cooper", "DirectoryExplorer")
        self._archive("O2", "29 permissions.", "Identity 28c5  Alan Cooper")
        self._rights()
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            scope=self.scope, archive=self.archive,
        )
        self.assertTrue(report.fired)
        self.assertNotIn("Alan Cooper", text)
        for name in ("Alisha Ochoa", "Anna Garcia", "Barbara Sanchez",
                     "Brandon Miller", "Christopher Hubbard"):
            self.assertIn(name, text)
        self.assertIn("10 more actions", text)
        self.assertIn("do not ask the user", text)
        self.assertEqual(report.subjects_named, 5)

    def test_a_run_that_reached_everyone_is_not_nudged(self) -> None:
        for index, name in enumerate(
            ["Alan Cooper", "Alisha Ochoa", "Anna Garcia", "Barbara Sanchez",
             "Brandon Miller", "Christopher Hubbard"], start=1
        ):
            self._archive(f"O{index}", "rows", f"Identity uid-{index}  {name}")
        self._rights()
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            scope=self.scope, archive=self.archive,
        )
        self.assertEqual(text, "")
        self.assertFalse(report.fired)
        self.assertEqual(report.reason, "every named item was already a subject")

    def test_a_possessive_spelling_never_nudges_an_opened_subject(self) -> None:
        # ido-jf6/F29. Both people were opened; the nudge used to fire on
        # "Alan Cooper's" and "Brandon Miller's", which no clause can contain.
        text, report = build_nudge(
            user_query="List Alan Cooper's rights and Brandon Miller's accounts.",
            iterations_left=10,
            clauses=normalise("Identity 28c5  Alan Cooper\nIdentity 9a1  Brandon Miller"),
        )
        self.assertEqual(text, "")
        self.assertFalse(report.fired)
        self.assertEqual(report.subjects_missing, [])

    def test_no_recorded_clause_says_nothing(self) -> None:
        text, report = build_nudge(
            user_query=CARD, iterations_left=10,
            scope=self.scope, archive=self.archive,
        )
        self.assertEqual(text, "")
        self.assertEqual(report.reason, "no context clauses recorded")

    def test_a_turn_with_no_room_is_never_nudged(self) -> None:
        text, report = build_nudge(
            user_query=CARD, iterations_left=NUDGE_MIN_ITERS_LEFT - 1,
            clauses=normalise("DirectoryExplorer"),
        )
        self.assertEqual(text, "")
        self.assertEqual(report.reason, "no room to act")

    def test_the_note_is_bounded_and_counts_what_it_cut(self) -> None:
        names = [f"Personname Number{index:03d}" for index in range(120)]
        text, named = nudge_block(names, 5)
        self.assertLessEqual(len(text.encode("utf-8")), NUDGE_MAX_BYTES)
        self.assertLess(named, len(names))
        self.assertIn(f"and {len(names) - named} more", text)

    def test_the_same_state_always_gives_the_same_note(self) -> None:
        clauses = normalise("Identity 28c5  Alan Cooper")
        first, _ = build_nudge(user_query=CARD, iterations_left=7, clauses=clauses)
        second, _ = build_nudge(user_query=CARD, iterations_left=7, clauses=clauses)
        self.assertEqual(first, second)
        self.assertNotEqual("", first)


class Stub:
    """Stands in for ``self.react``: replays a fixed list of actions."""

    def __init__(self, actions) -> None:
        self.actions = list(actions)
        self.calls: list[str] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs["trajectory"])
        name, args = self.actions[min(len(self.calls) - 1, len(self.actions) - 1)]
        return dspy.Prediction(next_thought="t", next_tool_name=name,
                               next_tool_args=args)


class LoopHook(unittest.TestCase):
    """ido-8ps.27 (a): the finish action and one nudge per turn.

    ``ido-pyw.1`` removed ``FW_ROSTER_NUDGE``. The loop check is unconditional,
    so "the loop it was at 9e5e9d9" is no longer a claim these tests can make;
    what replaces it is the run that has nothing to be nudged about.
    """

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        from fastworkflow.utils.react import fastWorkflowReAct

        def a_tool(value: str = "") -> str:
            """A tool."""
            return f"observed {value}"

        self.scope = scope_for(self.directory.name)
        self.archive = RuntimeHandleArchive(
            os.path.join(self.directory.name, "obs.sqlite3"))
        text = "1 identity.\nuid-1  Alan Cooper"
        self.archive.persist(
            self.scope, alias="O1", offload_order=1, command_name="find_identity",
            step_index=0, text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        for alias, clause in (
            ("O8", "Permission 85cd  Active Directory_Cloud Administrator"),
            ("O9", "Permission 3e3d  Active Directory_Compliance Officer"),
        ):
            self.archive.persist(
                self.scope, alias=alias, offload_order=int(alias[1:]),
                command_name="show_holders", step_index=int(alias[1:]),
                text="holders", text_sha256=hashlib.sha256(b"holders").hexdigest(),
            )
            record_context_clause(self.scope, alias, clause)
        self.agent = fastWorkflowReAct("user_query -> final_answer",
                                       tools=[a_tool], max_iters=12)
        self.agent.continuation_scope = self.scope
        self.agent.observation_archive = self.archive

    def _run(self, actions):
        self.agent.react = Stub(actions)
        trajectory: dict = {}
        self.agent._roster_nudges_fired = 0
        self.agent.iteration_counter = 0
        self.agent._run_loop(trajectory, 0, {"user_query": CARD}, 12, 0)
        return trajectory

    def test_a_request_with_no_named_items_ends_at_the_finish_action(self) -> None:
        """ido-pyw.1. The check runs on every finish action and this is what it
        does when there is nothing to say: the trajectory is the one a finish
        action always produced, the nudge is recorded as not fired, and the
        reason names why rather than naming a setting."""
        self.agent.react = Stub([("finish", {}), ("a_tool", {"value": "x"})])
        trajectory: dict = {}
        self.agent._roster_nudges_fired = 0
        self.agent.iteration_counter = 0
        self.agent._run_loop(trajectory, 0, {"user_query": "list everything"}, 12, 0)
        self.assertEqual(
            trajectory,
            {"thought_0": "t", "tool_name_0": "finish", "tool_args_0": {},
             "observation_0": "Completed."},
        )
        event = [e for e in snapshot_events() if e["kind"] == "roster_nudge"][0]
        self.assertFalse(event["fired"])

    def test_the_nudge_returns_control_to_the_loop(self) -> None:
        trajectory = self._run([("finish", {}), ("a_tool", {"value": "more"}),
                                ("finish", {})])
        self.assertIn("Harness check before this turn ends",
                      trajectory["observation_0"])
        self.assertIn("Brandon Miller", trajectory["observation_0"])
        self.assertNotIn("Alan Cooper", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_1"], "observed more")
        self.assertEqual(trajectory["tool_name_2"], "finish")
        self.assertEqual(trajectory["observation_2"], "Completed.")

    def test_at_most_one_nudge_per_turn(self) -> None:
        trajectory = self._run([("finish", {})])
        self.assertIn("Harness check", trajectory["observation_0"])
        self.assertEqual(trajectory["observation_1"], "Completed.")
        self.assertNotIn("observation_2", trajectory)
        fired = [event for event in snapshot_events()
                 if event["kind"] == "roster_nudge" and event["fired"]]
        self.assertEqual(len(fired), 1)

    def test_the_event_carries_what_the_summarizer_counts(self) -> None:
        self._run([("finish", {})])
        event = [e for e in snapshot_events() if e["kind"] == "roster_nudge"][0]
        self.assertTrue(event["fired"])
        self.assertEqual(event["subjects_total"], 9)
        self.assertEqual(event["subjects_named"], 5)
        self.assertIn("Brandon Miller", event["subjects_missing"])
        self.assertGreater(event["text_bytes"], 0)
        self.assertLessEqual(event["text_bytes"], NUDGE_MAX_BYTES)
        self.assertGreater(event["iterations_left"], 0)

    def test_never_on_a_turn_with_no_room(self) -> None:
        self.agent.react = Stub([("finish", {})])
        trajectory: dict = {}
        self.agent._roster_nudges_fired = 0
        self.agent.iteration_counter = 11  # max_iters is 12: nothing left to do
        self.agent._run_loop(trajectory, 0, {"user_query": CARD}, 12, 0)
        self.assertEqual(trajectory["observation_0"], "Completed.")
        self.assertFalse(self.agent._exhausted_last_run)
        event = [e for e in snapshot_events() if e["kind"] == "roster_nudge"][0]
        self.assertFalse(event["fired"])
        self.assertEqual(event["reason"], "no room to act")

    def test_a_failure_leaves_the_loop_alone(self) -> None:
        with mock.patch.object(answer_coverage, "build_nudge",
                               side_effect=RuntimeError("boom")):
            trajectory = self._run([("finish", {})])
        self.assertEqual(trajectory["observation_0"], "Completed.")
        self.assertEqual([e["kind"] for e in snapshot_events()],
                         ["roster_nudge_failed"])


if __name__ == "__main__":
    unittest.main()


#: ido-8ps.28. The statement this fixture produced at 90a1565, byte for byte --
#: which is also the statement it produces now whenever no subject has anything
#: to state. Frozen here, not recomputed, because it is a claim about bytes a
#: model received on a day and a recomputed expectation would move with the code
#: it is meant to pin.
STATEMENT_AT_90A1565 = (
    "Coverage of this run: the loop ended normally. These named items from the "
    "request appear in no retrieved observation: Active Directory; Active "
    "Directory_Cloud Administrator; Active Directory_Compliance Officer; "
    "Alisha Ochoa; Anna Garcia; Barbara Sanchez; Brandon Miller; Christopher "
    'Hubbard. For each of them report "not retrieved" and nothing else - no '
    "value, no unavailability, no absence. These named items of the request DO "
    "appear in this run's observations: Alan Cooper. Every other named item of "
    "the request WAS retrieved: it appears in this run's observations and must "
    'be reported from them. Do not write "not retrieved", "not available", "no '
    'data", or any other statement of absence about an item that is not named '
    "in the unobserved list above. For items that appear, report only what the "
    "observations show."
)


class EvidenceSentenceText(unittest.TestCase):
    """ido-8ps.28: the sentence itself, as bytes. No store, no run."""

    def test_nothing_to_state_is_silence(self) -> None:
        self.assertEqual(evidence_sentence([]), ("", 0, 0))
        self.assertEqual(evidence_sentence([("Anna Garcia", [])]), ("", 0, 0))

    def test_a_subject_with_no_items_is_never_printed_empty(self) -> None:
        """The one rule this sentence cannot bend: no absence, in any spelling."""
        text, subjects, items = evidence_sentence(
            [("Alan Cooper", ["Active Directory_Cloud Administrator"]),
             ("Anna Garcia", []),
             ("Barbara Sanchez", ["Active Directory_Cloud Administrator"])]
        )
        self.assertNotIn("Anna Garcia", text)
        self.assertEqual((subjects, items), (2, 2))
        self.assertNotIn("none", text)
        self.assertNotIn("not", text.removeprefix(EVIDENCE_HEAD))

    def test_the_shape_is_subject_colon_items(self) -> None:
        text, subjects, items = evidence_sentence(
            [("Alan Cooper", ["Right A", "Right B"]),
             ("Alisha Ochoa", ["Right B"])]
        )
        self.assertEqual(
            text,
            EVIDENCE_HEAD + "Alan Cooper: Right A, Right B; Alisha Ochoa: Right B. ",
        )
        self.assertEqual((subjects, items), (2, 3))

    def test_it_is_deterministic(self) -> None:
        pairs = [("A Name", ["Item One"]), ("B Name", ["Item Two"])]
        self.assertEqual(evidence_sentence(pairs), evidence_sentence(list(pairs)))

    def test_whole_subjects_only_and_the_rest_are_counted(self) -> None:
        """A half-written subject would read as a short list, and a short list
        is how a positive statement turns into an absence."""
        pairs = [(f"Subject Number {index}", ["A Long Property Literal Here"] * 3)
                 for index in range(40)]
        text, subjects, _ = evidence_sentence(pairs)
        listed = text.removeprefix(EVIDENCE_HEAD)
        self.assertLessEqual(len(listed.encode("utf-8")), EVIDENCE_LIST_MAX_BYTES + 2)
        self.assertLess(subjects, 40)
        self.assertIn(f"; and {40 - subjects} more subjects", text)
        for entry in listed.split("; and ")[0].split("; "):
            self.assertEqual(entry.count("A Long Property Literal Here"), 3)


class EvidenceBySubject(unittest.TestCase):
    """ido-8ps.28: which named items a subject's OWN observations contain."""

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scope = scope_for(self.directory.name)
        self.archive = RuntimeHandleArchive(
            os.path.join(self.directory.name, "obs.sqlite3"))

    def _archive(self, alias: str, text: str, clause: str = "") -> None:
        self.archive.persist(
            self.scope, alias=alias, offload_order=int(alias[1:]),
            command_name="list_entitlements", step_index=int(alias[1:]), text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        record_context_clause(self.scope, alias, clause)

    def _items(self):
        return [entity for entity in named_entities(CARD)
                if entity.kind in INSTRUCTED_KINDS]

    def _evidence(self):
        return [
            (subject.text, [item.text for item in items])
            for subject, items in evidence_by_subject(
                self._items(), scope=self.scope, archive=self.archive,
                handle_store=None,
            )
        ]

    def test_only_a_stamped_subject_appears_at_all(self) -> None:
        self._archive("O1", "5 identities.\nuid-2  Anna Garcia", "DirectoryExplorer")
        self.assertEqual(self._evidence(), [])

    def test_a_subject_gets_what_its_own_observations_contain(self) -> None:
        self._archive("O1", "Active Directory_Compliance Officer",
                      "Identity 3f22  Alisha Ochoa")
        self._archive("O2", "Active Directory_Cloud Administrator\n"
                            "Active Directory_Compliance Officer",
                      "Identity 28c5  Alan Cooper")
        by_subject = dict(self._evidence())
        self.assertEqual(by_subject["Alisha Ochoa"],
                         ["Active Directory", "Active Directory_Compliance Officer"])
        self.assertEqual(by_subject["Alan Cooper"],
                         ["Active Directory", "Active Directory_Cloud Administrator",
                          "Active Directory_Compliance Officer"])

    def test_another_subjects_rows_are_not_this_subjects_evidence(self) -> None:
        """The premise-copy failure, stated as the evidence question."""
        self._archive("O1", "Active Directory_Cloud Administrator\n"
                            "Active Directory_Compliance Officer",
                      "Identity 28c5  Alan Cooper")
        self._archive("O2", "Active Directory_Compliance Officer",
                      "Identity 3f22  Alisha Ochoa")
        by_subject = dict(self._evidence())
        self.assertNotIn("Active Directory_Cloud Administrator",
                         by_subject["Alisha Ochoa"])

    def test_a_stamped_subject_with_nothing_is_still_reported_to_the_caller(self) -> None:
        """The reader returns it; the SENTENCE is what refuses to print it."""
        self._archive("O1", "0 entitlements.", "Identity 9a11  Barbara Sanchez")
        self.assertEqual(self._evidence(), [("Barbara Sanchez", [])])
        self.assertEqual(evidence_sentence(self._evidence()), ("", 0, 0))

    def test_a_subject_is_never_its_own_item(self) -> None:
        self._archive("O1", "Alan Cooper holds nothing here.",
                      "Identity 28c5  Alan Cooper")
        self.assertEqual(dict(self._evidence())["Alan Cooper"], [])


class EvidenceInTheBlock(unittest.TestCase):
    """ido-8ps.28: where the sentence sits, and when it says nothing.

    ``ido-pyw.1`` removed ``FW_ANSWER_EVIDENCE``; what kept the sentence honest
    was never the flag but the positive-half-only rule, and that is what these
    now test.
    """

    def setUp(self) -> None:
        reset_runtime_state()
        self.addCleanup(reset_runtime_state)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
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
        self.trajectory = {
            "thought_0": "look",
            "tool_name_0": "execute_workflow_query",
            "tool_args_0": {"command": "find_identity"},
            "observation_0": alias_line("O1") + text,
        }

    def _build(self):
        return build_statement(
            self.trajectory, user_query=CARD, exhausted=False,
            scope=self.scope, archive=self.archive, handle_store=self.store,
        )

    def test_a_lone_subject_with_nothing_to_state_is_90a1565(self) -> None:
        """The sentence is computed, the subject is known, and the block is the
        one the accepted stack rendered -- because the subject's own
        observations contain no OTHER named item of the request."""
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        _, report = self._build()
        self.assertEqual(report.statement, STATEMENT_AT_90A1565)
        self.assertEqual(report.statement_bytes, 815)
        self.assertEqual(report.evidence, [("Alan Cooper", [])])
        self.assertEqual(report.evidence_bytes, 0)

    def _ochoa(self) -> None:
        """One further observation, stamped against a second subject, whose text
        carries one of the request's other named items."""
        text = "Active Directory_Compliance Officer"
        self.archive.persist(
            self.scope, alias="O2", offload_order=2,
            command_name="list_entitlements", step_index=1, text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        record_context_clause(self.scope, "O2", "Identity 3f22  Alisha Ochoa")

    def test_the_sentence_sits_after_the_observed_items_rule(self) -> None:
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        self._ochoa()
        _, report = self._build()
        statement = report.statement
        self.assertIn(EVIDENCE_HEAD, statement)
        self.assertLess(statement.index(RETRIEVED_RULE),
                        statement.index(EVIDENCE_HEAD))
        self.assertLess(statement.index(EVIDENCE_HEAD),
                        statement.index("For items that appear, report only"))

    def test_the_sentence_states_the_subjects_own_items(self) -> None:
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        self._ochoa()
        _, report = self._build()
        self.assertIn("Alisha Ochoa: Active Directory, "
                      "Active Directory_Compliance Officer", report.statement)
        self.assertNotIn("Alisha Ochoa: Active Directory, Active "
                         "Directory_Cloud Administrator", report.statement)
        self.assertEqual(report.evidence_named, ["Alisha Ochoa"])
        self.assertGreater(report.evidence_bytes, 0)

    def test_no_subject_has_anything_and_the_sentence_is_absent(self) -> None:
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        _, report = self._build()
        self.assertEqual(report.evidence, [("Alan Cooper", [])])
        self.assertEqual(report.statement, STATEMENT_AT_90A1565)

    def test_no_clause_recorded_says_nothing(self) -> None:
        _, report = self._build()
        self.assertEqual(report.evidence, [])
        self.assertEqual(report.statement, STATEMENT_AT_90A1565)

    def test_a_failure_in_the_reader_costs_the_sentence_and_nothing_else(self) -> None:
        record_context_clause(self.scope, "O1", "Identity 28c5  Alan Cooper")
        from fastworkflow import answer_attribution

        with mock.patch.object(answer_attribution, "observations",
                               side_effect=RuntimeError("boom")):
            _, report = self._build()
        self.assertEqual(report.statement, STATEMENT_AT_90A1565)
        self.assertEqual(report.evidence, [])


class EvidenceMeasure(unittest.TestCase):
    """ido-8ps.28: per subject, claimed items the sentence listed vs not."""

    def test_a_claim_the_sentence_listed_and_one_it_did_not(self) -> None:
        answer = (
            "| Alisha Ochoa | Active Directory_Cloud Administrator, "
            "Active Directory_Compliance Officer |"
        )
        check = post_check(
            answer, [], [],
            evidence=[("Alisha Ochoa", ["Active Directory_Compliance Officer"])],
            items=["Active Directory_Cloud Administrator",
                   "Active Directory_Compliance Officer"],
        )
        event = check.as_event()
        self.assertEqual(event["evidence_subjects_total"], 1)
        self.assertEqual(event["evidence_subjects_mentioned"], 1)
        self.assertEqual(event["evidence_claims_listed"], 1)
        self.assertEqual(event["evidence_claims_unlisted"], 1)
        row = event["per_evidence_subject"][0]
        self.assertEqual(row["claimed_unlisted"],
                         ["Active Directory_Cloud Administrator"])

    def test_a_subject_the_answer_never_names(self) -> None:
        check = post_check("nothing at all", [], [],
                           evidence=[("Anna Garcia", ["Right A"])],
                           items=["Right A"])
        event = check.as_event()
        self.assertEqual(event["evidence_subjects_total"], 1)
        self.assertEqual(event["evidence_subjects_mentioned"], 0)
        self.assertEqual(event["evidence_claims_listed"], 0)
        self.assertFalse(event["per_evidence_subject"][0]["mentioned"])

    def test_it_measures_a_subject_the_sentence_did_not_print(self) -> None:
        """The sentence may not print an empty subject; the measure may count it."""
        check = post_check("Barbara Sanchez - Right A.", [], [],
                           evidence=[("Barbara Sanchez", [])], items=["Right A"])
        event = check.as_event()
        self.assertEqual(event["evidence_claims_unlisted"], 1)
        self.assertEqual(event["evidence_claims_listed"], 0)

    def test_the_measure_is_absent_when_nothing_is_passed(self) -> None:
        event = post_check("anything", ["Anna Garcia"]).as_event()
        self.assertEqual(event["evidence_subjects_total"], 0)
        self.assertEqual(event["evidence_claims_unlisted"], 0)
