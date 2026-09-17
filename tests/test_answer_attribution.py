"""ido-8ps.28: a subject and a property literal, and what the evidence carries.

Offline only. Nothing here starts a server, calls a model or touches a backend:
the evidence is either a list of ``Observation`` values written in the test or a
real SQLite archive in a temp dir, and every assertion is about bytes the test
itself put there.

The check is framework-generic by construction, so the cases are written on a
workflow that does not exist: ``Ada Lovelace``, ``Charles Babbage`` and the two
items ``Analytical Engine_Drive Wheel`` and ``Analytical Engine_Mill Gear``.
Anything that only passes on the IDO card would be a check this bead may not
build.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

from fastworkflow import answer_attribution
from fastworkflow.answer_attribution import (
    MAX_ANSWER_BYTES,
    MAX_FLAGS,
    MAX_MENTIONS_PER_UNIT,
    MAX_UNITS,
    MIN_SEGMENT_CHARS,
    Observation,
    REASON_SUBJECT_NOT_OBSERVED,
    REASON_UNSUPPORTED,
    attribution_report,
    check_attribution,
    match_forms,
    match_forms_index,
    observations,
    subject_evidence,
    units,
    writes,
)
from fastworkflow.answer_coverage import named_entities
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.state import (
    record_context_clause,
    reset_runtime_state,
)
from fastworkflow.result_handles import ResultHandleStore

from tests.test_answer_rehydration import declaration_payload, page_record

REQUEST = (
    "Two engine parts keep coming back on this quarter's exceptions: "
    "Analytical Engine_Drive Wheel and Analytical Engine_Mill Gear. Two people "
    "need the same treatment - Ada Lovelace and Charles Babbage - so I want "
    "each of them down to their parts."
)

ADA = Observation(
    alias="O1",
    clause="Person 1815 Ada Lovelace",
    text="1815 [parts]  Analytical Engine_Mill Gear via device 7\n"
         "1816 [parts]  Difference Engine_Crank",
)
CHARLES = Observation(
    alias="O2",
    clause="Person 1791 Charles Babbage",
    text="1791 [parts]  Analytical Engine_Drive Wheel via device 2\n"
         "1792 [parts]  Analytical Engine_Mill Gear via device 2",
)


def scope_for(name: str) -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity=name, channel_id="c", experiment_id="exp-ido-8ps-28",
        task_id="task", attempt=1, turn_key="turn-1",
    )


def flags_of(report, reason: str = REASON_UNSUPPORTED) -> list[tuple[str, str]]:
    return [(flag.subject, flag.property) for flag in report.flags
            if flag.reason == reason]


class Units(unittest.TestCase):
    """A table row is one claim; a sentence is one claim."""

    def test_a_table_row_stays_whole(self) -> None:
        self.assertEqual(
            units("| Ada Lovelace | Mill Gear | yes |"),
            ["| Ada Lovelace | Mill Gear | yes |"],
        )

    def test_a_line_is_cut_into_sentences_and_clauses(self) -> None:
        self.assertEqual(
            units("Ada holds one. Charles holds two; Ada holds none."),
            ["Ada holds one.", "Charles holds two", "Ada holds none."],
        )

    def test_units_are_capped(self) -> None:
        self.assertEqual(len(units("x.\n" * (MAX_UNITS + 50))), MAX_UNITS)


class Supported(unittest.TestCase):
    """A pairing the evidence carries is not a flag."""

    def test_a_supported_sentence_raises_nothing(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="Charles Babbage holds Analytical Engine_Drive Wheel.",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(report.pairs_supported, 1)
        self.assertEqual(report.pairs_unsupported, 0)
        self.assertEqual(report.flags, [])

    def test_the_evidence_must_be_the_subjects_own(self) -> None:
        # The Drive Wheel is in Charles's observation and in nobody else's.
        report = check_attribution(
            request=REQUEST,
            answer="Ada Lovelace holds Analytical Engine_Drive Wheel.",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(flags_of(report),
                         [("Ada Lovelace", "Analytical Engine_Drive Wheel")])
        self.assertEqual(report.pairs_unsupported, 1)

    def test_a_list_of_items_is_not_a_claim_about_them(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="The people are Ada Lovelace and Charles Babbage.",
            observations=[ADA, CHARLES],
        )
        # One enumeration group, so there is no pair to test at all.
        self.assertEqual(report.pairs_total, 0)
        self.assertEqual(report.flags, [])

    def test_an_aside_does_not_turn_a_list_into_a_claim(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="Holders: Ada Lovelace (observation O1) and Charles Babbage.",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(report.flags, [])

    def test_a_distributive_list_is_checked_member_by_member(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="Ada Lovelace and Charles Babbage both hold "
                   "Analytical Engine_Drive Wheel.",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(flags_of(report),
                         [("Ada Lovelace", "Analytical Engine_Drive Wheel")])

    def test_two_clauses_are_two_claims(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="Ada Lovelace holds Analytical Engine_Mill Gear; "
                   "Charles Babbage holds Analytical Engine_Drive Wheel.",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(report.flags, [])
        self.assertEqual(report.pairs_supported, 2)


class TableRows(unittest.TestCase):
    """The row's first named cell is its subject; the later cells are the claim."""

    ROW = (
        "| **Ada Lovelace** | `1815` - Ada Lovelace | Analytical Engine_Drive "
        "Wheel, Analytical Engine_Mill Gear | 2 parts |"
    )

    def test_a_row_co_attributes_its_later_cells_to_its_first(self) -> None:
        report = check_attribution(
            request=REQUEST, answer=self.ROW, observations=[ADA, CHARLES]
        )
        self.assertEqual(flags_of(report),
                         [("Ada Lovelace", "Analytical Engine_Drive Wheel")])
        # The Mill Gear IS in her evidence, so the same row raises one flag only.
        self.assertEqual(report.pairs_unsupported, 1)
        self.assertGreaterEqual(report.pairs_supported, 1)

    def test_items_in_one_cell_are_not_claims_about_each_other(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="| **Charles Babbage** | Analytical Engine_Drive Wheel, "
                   "Analytical Engine_Mill Gear |",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(report.flags, [])

    def test_the_span_shows_the_row(self) -> None:
        report = check_attribution(
            request=REQUEST, answer=self.ROW, observations=[ADA, CHARLES]
        )
        self.assertIn("ada lovelace", report.flags[0].answer_span)
        self.assertIn("drive wheel", report.flags[0].answer_span)


class Negation(unittest.TestCase):
    """A denial is not an attribution."""

    def test_does_not_hold_is_not_a_claim(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="Ada Lovelace does not hold Analytical Engine_Drive Wheel.",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(report.pairs_negated, 1)
        self.assertEqual(report.flags, [])

    def test_a_denial_after_the_property_is_still_a_denial(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="Ada Lovelace holds two parts, but Analytical Engine_Drive "
                   "Wheel is not present in her list.",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(report.pairs_negated, 1)
        self.assertEqual(report.flags, [])

    def test_a_denial_in_the_next_cell_does_not_excuse_this_one(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="| Ada Lovelace | Analytical Engine_Drive Wheel | no data |",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(flags_of(report),
                         [("Ada Lovelace", "Analytical Engine_Drive Wheel")])

    def test_only_is_not_a_negation(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="Ada Lovelace holds only Analytical Engine_Drive Wheel.",
            observations=[ADA, CHARLES],
        )
        self.assertEqual(report.pairs_negated, 0)
        self.assertEqual(len(flags_of(report)), 1)


class NeverASubject(unittest.TestCase):
    """A subject no command was stamped against is its own kind of finding."""

    def test_a_subject_with_no_clause_is_reported_apart(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="Charles Babbage holds Analytical Engine_Drive Wheel.",
            observations=[ADA],
        )
        self.assertEqual(report.pairs_unsupported, 0)
        self.assertEqual(report.pairs_subject_not_observed, 1)
        self.assertEqual(flags_of(report, REASON_SUBJECT_NOT_OBSERVED),
                         [("Charles Babbage", "Analytical Engine_Drive Wheel")])

    def test_a_turn_with_no_clauses_raises_no_unsupported_flag(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="| Ada Lovelace | Analytical Engine_Drive Wheel |",
            observations=[Observation(alias="O1", clause="", text="anything")],
        )
        self.assertEqual(report.clauses_total, 0)
        self.assertEqual(report.pairs_unsupported, 0)
        self.assertEqual(report.pairs_subject_not_observed, 1)

    def test_no_evidence_at_all_is_the_same_refusal(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="| Ada Lovelace | Analytical Engine_Drive Wheel |",
            observations=[],
        )
        self.assertEqual(report.observations_total, 0)
        self.assertEqual(report.pairs_unsupported, 0)


class Segments(unittest.TestCase):
    """A qualified name written by its last segment."""

    def test_the_last_segment_is_a_form_of_the_item(self) -> None:
        entity = named_entities(REQUEST)[0]
        self.assertEqual(entity.text, "Analytical Engine_Drive Wheel")
        self.assertEqual(match_forms(entity), ["analytical engine_drive wheel",
                                               "drive wheel"])
        self.assertEqual(match_forms(entity, allow_segments=False),
                         ["analytical engine_drive wheel"])

    def test_a_short_segment_is_not_a_handle(self) -> None:
        entity = named_entities("Please audit the Big Engine_Gear now.")[0]
        self.assertEqual(entity.text, "Big Engine_Gear")
        self.assertLess(len("gear"), MIN_SEGMENT_CHARS)
        self.assertEqual(match_forms(entity), ["big engine_gear"])

    def test_segments_are_what_catches_an_abbreviated_row(self) -> None:
        row = "| Ada Lovelace | Drive Wheel, Mill Gear |"
        loose = check_attribution(request=REQUEST, answer=row,
                                  observations=[ADA, CHARLES])
        strict = check_attribution(request=REQUEST, answer=row,
                                   observations=[ADA, CHARLES],
                                   allow_segments=False)
        self.assertEqual(flags_of(loose),
                         [("Ada Lovelace", "Analytical Engine_Drive Wheel")])
        self.assertEqual(strict.flags, [])
        self.assertFalse(strict.segments)


SIBLING_REQUEST = (
    "Audit Analytical Engine_Drive Wheel and Difference Engine_Drive Wheel "
    "for Ada Lovelace and Charles Babbage."
)

#: Ada's OWN rows carry the Difference Engine part and nothing else.
ADA_ONE_PART = Observation(
    alias="O7",
    clause="Person 1815 Ada Lovelace",
    text="1 part.\n1815 [parts]  Difference Engine_Drive Wheel via device 7",
)


class SiblingSegments(unittest.TestCase):
    """ido-rf3: a tail two request items share names neither of them."""

    def test_a_shared_tail_is_not_a_form_of_either_item(self) -> None:
        items = named_entities(SIBLING_REQUEST)
        forms = match_forms_index(items)
        self.assertEqual(forms["analytical engine_drive wheel"],
                         ["analytical engine_drive wheel"])
        self.assertEqual(forms["difference engine_drive wheel"],
                         ["difference engine_drive wheel"])

    def test_an_unshared_tail_still_stands_for_its_item(self) -> None:
        items = named_entities(REQUEST)
        forms = match_forms_index(items)
        self.assertEqual(forms["analytical engine_drive wheel"],
                         ["analytical engine_drive wheel", "drive wheel"])
        self.assertEqual(forms["analytical engine_mill gear"],
                         ["analytical engine_mill gear", "mill gear"])

    def test_a_shared_tail_does_not_make_the_sibling_supported(self) -> None:
        answer = ("| Person | Part |\n"
                  "| Ada Lovelace | Analytical Engine_Drive Wheel |\n")
        report = check_attribution(request=SIBLING_REQUEST, answer=answer,
                                   observations=[ADA_ONE_PART])
        self.assertEqual(report.pairs_supported, 0)
        self.assertEqual(report.pairs_unsupported, 1)
        self.assertEqual(
            flags_of(report),
            [("Ada Lovelace", "Analytical Engine_Drive Wheel")],
        )

    def test_the_part_the_rows_do_carry_is_still_supported(self) -> None:
        answer = ("| Person | Part |\n"
                  "| Ada Lovelace | Difference Engine_Drive Wheel |\n")
        report = check_attribution(request=SIBLING_REQUEST, answer=answer,
                                   observations=[ADA_ONE_PART])
        self.assertEqual(report.pairs_supported, 1)
        self.assertEqual(report.pairs_unsupported, 0)
        self.assertEqual(report.flags, [])

    def test_an_unshared_tail_still_supports_an_abbreviated_row(self) -> None:
        """The convenience survives where the tail is unambiguous."""
        abbreviated = Observation(
            alias="O8",
            clause="Person 1815 Ada Lovelace",
            text="1 part.\n1815 [parts]  Drive Wheel via device 7",
        )
        report = check_attribution(
            request=REQUEST,
            answer="| Ada Lovelace | Analytical Engine_Drive Wheel |",
            observations=[abbreviated],
        )
        self.assertEqual(report.pairs_supported, 1)
        self.assertEqual(report.flags, [])

    def test_a_tail_inside_another_qualified_name_is_that_other_name(self) -> None:
        """The request names ONE such item and the rows carry a different one."""
        request = "Audit Analytical Engine_Drive Wheel for Ada Lovelace."
        other_part = Observation(
            alias="O9",
            clause="Person 1815 Ada Lovelace",
            text="1 part.\n1815 [parts]  Difference Engine_Drive Wheel via 7",
        )
        report = check_attribution(
            request=request,
            answer="| Ada Lovelace | Analytical Engine_Drive Wheel |",
            observations=[other_part],
        )
        self.assertEqual(report.pairs_supported, 0)
        self.assertEqual(report.pairs_unsupported, 1)

    def test_the_containment_test_reads_a_prefix_as_a_different_thing(self) -> None:
        forms = ["analytical engine_drive wheel", "drive wheel"]
        self.assertTrue(writes("1815 [parts]  drive wheel via 7", forms))
        self.assertTrue(writes("1815  analytical engine_drive wheel", forms))
        self.assertFalse(writes("1815  difference engine_drive wheel", forms))
        self.assertTrue(
            writes("difference engine_drive wheel and drive wheel", forms))

    def test_a_clause_stamped_on_another_qualified_name_is_not_this_subject(
        self,
    ) -> None:
        request = "Audit Analytical Engine_Drive Wheel for Ada Lovelace."
        item = [e for e in named_entities(request)
                if e.text == "Analytical Engine_Drive Wheel"]
        elsewhere = Observation(
            alias="O10",
            clause="Part 77aa Difference Engine_Drive Wheel",
            text="3 holders.\np1  Ada Lovelace",
        )
        self.assertEqual(
            answer_attribution.subject_index(item, [elsewhere]),
            {"analytical engine_drive wheel": []},
        )

    def test_the_evidence_sentence_does_not_lend_one_item_the_others_rows(self) -> None:
        items = named_entities(SIBLING_REQUEST)
        evidence = subject_evidence(items, [ADA_ONE_PART])
        self.assertEqual(
            [(subject.text, [other.text for other in others])
             for subject, others in evidence],
            [("Ada Lovelace", ["Difference Engine_Drive Wheel"])],
        )


class Caps(unittest.TestCase):
    """Bounds, so one answer can never become an unbounded report."""

    def test_the_answer_is_truncated_and_says_so(self) -> None:
        filler = "Ada Lovelace holds Analytical Engine_Drive Wheel.\n"
        answer = filler * (MAX_ANSWER_BYTES // len(filler) + 200)
        report = check_attribution(request=REQUEST, answer=answer,
                                   observations=[ADA, CHARLES])
        self.assertTrue(report.truncated)
        self.assertLessEqual(len(report.flags), MAX_FLAGS)

    def test_flags_are_capped_and_the_overflow_is_counted(self) -> None:
        rows = "".join(
            f"| Ada Lovelace | Analytical Engine_Drive Wheel | row {index} |\n"
            for index in range(MAX_FLAGS + 20)
        )
        report = check_attribution(request=REQUEST, answer=rows,
                                   observations=[ADA, CHARLES])
        self.assertEqual(len(report.flags), MAX_FLAGS)
        self.assertEqual(report.flags_dropped, 20)
        self.assertEqual(report.pairs_unsupported, MAX_FLAGS + 20)

    def test_mentions_in_one_unit_are_capped(self) -> None:
        unit = "| Ada Lovelace " + "| Analytical Engine_Drive Wheel " * 200 + "|"
        report = check_attribution(request=REQUEST, answer=unit,
                                   observations=[ADA, CHARLES])
        self.assertLessEqual(report.pairs_total, MAX_MENTIONS_PER_UNIT)

    def test_the_span_is_bounded(self) -> None:
        row = ("| Ada Lovelace | " + "x " * 500
               + "| Analytical Engine_Drive Wheel |")
        report = check_attribution(request=REQUEST, answer=row,
                                   observations=[ADA, CHARLES])
        self.assertLessEqual(len(report.flags[0].answer_span),
                             answer_attribution.SPAN_MAX_CHARS + 6)


class Event(unittest.TestCase):
    """``as_event`` is the shape the judgement tooling reads."""

    def test_the_event_carries_the_counts_and_the_flags(self) -> None:
        report = check_attribution(
            request=REQUEST,
            answer="| Ada Lovelace | Analytical Engine_Drive Wheel |",
            observations=[ADA, CHARLES],
        )
        event = report.as_event()
        self.assertEqual(event["unsupported"], 1)
        self.assertEqual(event["pairs_supported"], 0)
        self.assertEqual(event["clauses_total"], 2)
        self.assertEqual(event["flags"][0]["reason"], REASON_UNSUPPORTED)
        self.assertEqual(event["flags"][0]["subject"], "Ada Lovelace")
        self.assertTrue(event["segments"])

    def test_an_empty_request_names_nothing_and_flags_nothing(self) -> None:
        report = check_attribution(request="", answer="anything at all",
                                   observations=[ADA])
        self.assertEqual(report.entities_total, 0)
        self.assertEqual(report.flags, [])


class Evidence(unittest.TestCase):
    """The reader is the one ``answer_coverage`` and rehydration already use."""

    def test_clause_and_rows_come_off_the_real_stores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reset_runtime_state()
            self.addCleanup(reset_runtime_state)
            scope = scope_for(directory)
            path = os.path.join(directory, "obs.sqlite3")
            archive = RuntimeHandleArchive(path)
            store = ResultHandleStore(path)
            shown = "result_handle=O2 page 1 rows 1-1 of 2\n1815  Mill Gear"
            archive.persist(
                scope, alias="O2", offload_order=2, command_name="list_parts",
                step_index=1, text=shown,
                text_sha256=hashlib.sha256(shown.encode("utf-8")).hexdigest(),
            )
            record_context_clause(scope, "O2", "Person 1815 Ada Lovelace")
            store.put_declaration(scope, "O2", declaration_payload(total=2))
            store.put_page(
                scope, alias="O2", query_scope="", start_offset=0,
                limit_requested=25, source="resolver", backend_total=2,
                record=page_record(["1815  Analytical Engine_Mill Gear",
                                    "1816  Difference Engine_Crank"]),
            )
            found = observations(scope=scope, archive=archive, handle_store=store)
            self.assertEqual([item.alias for item in found], ["O2"])
            self.assertEqual(found[0].clause, "person 1815 ada lovelace")
            # The stored row is evidence even though the bounded text lacks it.
            self.assertNotIn("Analytical Engine_Mill Gear", shown)
            self.assertIn("analytical engine_mill gear", found[0].text)

            report = attribution_report(
                request=REQUEST,
                answer="| Ada Lovelace | Analytical Engine_Mill Gear | "
                       "Analytical Engine_Drive Wheel |",
                scope=scope, archive=archive, handle_store=store,
            )
            self.assertEqual(flags_of(report),
                             [("Ada Lovelace", "Analytical Engine_Drive Wheel")])

    def test_a_search_answer_is_not_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reset_runtime_state()
            self.addCleanup(reset_runtime_state)
            scope = scope_for(directory)
            archive = RuntimeHandleArchive(os.path.join(directory, "obs.sqlite3"))
            answer = "Ada Lovelace holds Analytical Engine_Drive Wheel."
            archive.persist(
                scope, alias="O1#a1", offload_order=0, command_name="search_memory",
                step_index=1, text=answer,
                text_sha256=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            )
            record_context_clause(scope, "O1#a1", "Person 1815 Ada Lovelace")
            self.assertEqual(
                observations(scope=scope, archive=archive, handle_store=None), []
            )


if __name__ == "__main__":
    unittest.main()
