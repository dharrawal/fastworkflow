"""Evidence filler (ido-8ps.10): the validation rule, the fallback, the prompts.

Offline: no model, no server, no backend. The two model steps are driven by a
fake ``dspy.LM`` that answers with adapter-formatted text, the same way the rest
of the offloading suite drives DSPy without a provider.
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dspy

from fastworkflow import evidence_filler as ef
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.continuation import StructuredContinuationReAct
from fastworkflow.observation_offloading.state import reset_runtime_state, snapshot_events


HOLDERS = "\n".join(
    ["477 holder(s)"]
    + [f"{i:032x}  Person {i}" for i in range(60)]
    + ["e8a0c3a1d4b5f6a7b8c9d0e1f2a3b4c5  Alan Cooper",
       "d1c528f3aaaabbbbccccddddeeeeffff  Alisha Ochoa"]
)
PERMISSIONS = ("29 permission(s) for account e8a0c3a1d4b5f6a7b8c9d0e1f2a3b4c5\n"
               "3e3d35f0112233445566778899aabbcc  Active Directory_Cloud Administrator\n"
               "85cde16899887766554433221100ffee  Active Directory_Compliance Officer")


class FakeLM(dspy.LM):
    """Answers whichever of the two signatures the prompt asks for."""

    def __init__(self, items=None, filled=None, fail=False):
        super().__init__(model="fake/evidence-filler")
        self._items = items if items is not None else ["the right", "the holders"]
        self._filled = filled or []
        self._fail = fail
        self.calls = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        text = json.dumps(messages or prompt)
        self.calls.append(text)
        if self._fail:
            raise RuntimeError("provider refused")
        if "## filled ##" in text:
            payload = json.dumps(self._filled)
            return [f"[[ ## filled ## ]]\n{payload}\n\n[[ ## completed ## ]]"]
        payload = json.dumps(self._items)
        return [f"[[ ## requested_items ## ]]\n{payload}\n\n[[ ## completed ## ]]"]


def entry(item, value, observation, status="filled", reason=""):
    return {"item": item, "value": value, "observation": observation,
            "status": status, "reason": reason}


class ValidationRule(unittest.TestCase):
    """A filled value survives only when the evidence really carries it."""

    def setUp(self):
        self.printed = {"O1", "O2"}
        self.haystacks = {"O1": ef.normalise(HOLDERS), "O2": ef.normalise(PERMISSIONS)}

    def validate(self, payload, item="the right"):
        return ef.validate_entry(payload, printed_aliases=self.printed,
                                 haystacks=self.haystacks, item=item)

    def test_a_value_present_in_the_cited_observation_is_kept(self):
        row = self.validate(entry("the right", "Alan Cooper", "O1"))
        self.assertEqual(row.status, ef.FILLED)
        self.assertEqual(row.value, "Alan Cooper")
        self.assertEqual(row.alias, "O1")
        self.assertEqual(row.downgraded_from, "")

    def test_a_value_absent_from_the_cited_observation_is_downgraded(self):
        """The ido-986.6.14 failure: a plausible identifier nothing recorded."""
        row = self.validate(entry("the right", "b6a442b8c1bcaaf46f4a5f35dc39d30", "O2"))
        self.assertEqual(row.status, ef.UNRESOLVED)
        self.assertEqual(row.reason, ef.VALUE_NOT_IN_CITED_OBSERVATION)
        self.assertEqual(row.downgraded_from, "b6a442b8c1bcaaf46f4a5f35dc39d30")
        self.assertNotIn("b6a442b8", row.render())

    def test_a_value_in_another_observation_is_still_a_downgrade(self):
        """Right value, wrong citation: the rule is per alias, not per turn."""
        row = self.validate(entry("the right", "Alan Cooper", "O2"))
        self.assertEqual(row.reason, ef.VALUE_NOT_IN_CITED_OBSERVATION)

    def test_an_alias_the_turn_never_printed_is_downgraded(self):
        """Q1's defect: O56-O62 cited when the namespace ended at O54."""
        for alias in ["O56", "", "S3", "O1#a1", "step 12", "O0"]:
            row = self.validate(entry("the right", "Alan Cooper", alias))
            self.assertEqual(row.status, ef.UNRESOLVED)
            self.assertEqual(row.reason, ef.ALIAS_NOT_PRINTED, alias)

    def test_normalisation_cases_match_the_filters(self):
        """NBSP, zero width, fullwidth and case are repaired, on both sides."""
        for value in ["Alan Cooper", "Alan Cooper​", "ALAN COOPER",
                      "Alan   Cooper", "Ａlan Cooper", " Alan Cooper "]:
            row = self.validate(entry("the right", value, "O1"))
            self.assertEqual(row.status, ef.FILLED, value)
        # A zero width INSIDE a name deletes the space it replaced,
        # exactly as the filter does: "Alan<ZWSP>Cooper" is not "Alan Cooper".
        self.assertEqual(
            self.validate(entry("the right", "Alan​Cooper", "O1")).reason,
            ef.VALUE_NOT_IN_CITED_OBSERVATION)

    def test_an_underscore_is_not_stripped_the_way_a_filter_strips_it(self):
        """A value is not a LIKE pattern: removing `_` would match a wrong uid."""
        row = self.validate(entry("the right",
                                  "Active Directory_Cloud Administrator", "O2"))
        self.assertEqual(row.status, ef.FILLED)
        self.assertEqual(
            self.validate(entry("x", "Active DirectoryCloud Administrator", "O2")).reason,
            ef.VALUE_NOT_IN_CITED_OBSERVATION)

    def test_a_model_unresolved_row_is_not_a_downgrade(self):
        row = self.validate(entry("the right", "", "O1", status="unresolved",
                                  reason="the listing does not name a collection"))
        self.assertEqual(row.status, ef.UNRESOLVED)
        self.assertEqual(row.downgraded_from, "")
        self.assertIn("does not name a collection", row.render())

    def test_a_filled_row_with_no_value_is_unresolved(self):
        row = self.validate(entry("the right", "   ", "O1"))
        self.assertEqual(row.status, ef.UNRESOLVED)
        self.assertEqual(row.downgraded_from, "")


class WorksheetRendering(unittest.TestCase):
    def test_the_rendered_lines_are_the_two_shapes_the_extractor_reads(self):
        sheet = ef.Worksheet(items=[
            ef.WorksheetItem(item="the right", status=ef.FILLED,
                             value="3e3d35f0", alias="O2"),
            ef.WorksheetItem(item="its holders", status=ef.FILLED,
                             value="477 holder(s)", alias="O1",
                             page="O1#p200"),
            ef.WorksheetItem(item="the collection", status=ef.UNRESOLVED,
                             reason="no collection row was retrieved"),
        ])
        self.assertEqual(sheet.render(), "\n".join([
            "the right: 3e3d35f0 (Observation O2)",
            "its holders: 477 holder(s) (Observation O1, page O1#p200)",
            "the collection: unresolved - no collection row was retrieved",
        ]))

    def test_the_measures_count_downgrades_by_reason(self):
        sheet = ef.Worksheet(items=[
            ef.WorksheetItem(item="a", status=ef.FILLED, value="x", alias="O1"),
            ef.WorksheetItem(item="b", reason=ef.ALIAS_NOT_PRINTED,
                             downgraded_from="y"),
            ef.WorksheetItem(item="c", reason=ef.VALUE_NOT_IN_CITED_OBSERVATION,
                             downgraded_from="z"),
            ef.WorksheetItem(item="d", reason="the evidence does not establish it"),
        ])
        measures = sheet.measures()
        self.assertEqual(measures["items_total"], 4)
        self.assertEqual(measures["items_validated"], 1)
        self.assertEqual(measures["items_unresolved"], 3)
        self.assertEqual(measures["items_downgraded"], 2)
        self.assertEqual(measures["downgraded_by_reason"],
                         {ef.ALIAS_NOT_PRINTED: 1, ef.VALUE_NOT_IN_CITED_OBSERVATION: 1})


class AnswerUsedWorksheet(unittest.TestCase):
    def setUp(self):
        self.sheet = ef.Worksheet(items=[
            ef.WorksheetItem(item="a", status=ef.FILLED, value="Alan Cooper", alias="O1"),
            ef.WorksheetItem(item="b", status=ef.FILLED, value="3e3d35f0", alias="O2"),
            ef.WorksheetItem(item="c", reason="unresolved"),
        ])

    def test_a_value_the_answer_carries_counts_and_one_it_drops_does_not(self):
        measure = ef.answer_used_worksheet(
            self.sheet, "The holder is Alan COOPER (Observation O1).")
        self.assertEqual(measure["validated_values"], 2)
        self.assertEqual(measure["values_in_answer"], 1)
        self.assertEqual(measure["rate"], 0.5)
        self.assertEqual(measure["aliases_of_validated_values_cited_in_answer"], ["O1"])

    def test_an_empty_worksheet_has_no_rate_rather_than_a_perfect_one(self):
        measure = ef.answer_used_worksheet(ef.Worksheet(), "anything")
        self.assertIsNone(measure["rate"])
        self.assertEqual(measure["values_in_answer"], 0)


class EvidenceSelection(unittest.TestCase):
    def setUp(self):
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / "a.sqlite3"))
        self.scope = RuntimeHandleScope("store", "channel", "exp", "task", 1, "turn")
        self.persist("O1", HOLDERS)
        self.persist("O2", PERMISSIONS)
        self.archive.persist(self.scope, alias="O2#a1", offload_order=0,
                             command_name="search_memory", step_index=-1,
                             text="a bounded search answer",
                             text_sha256=hashlib.sha256(
                                 b"a bounded search answer").hexdigest())

    def persist(self, alias, text):
        self.archive.persist(self.scope, alias=alias, offload_order=int(alias[1:]),
                             command_name="show_holders", step_index=int(alias[1:]),
                             text=text,
                             text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def test_the_printed_namespace_excludes_a_search_answer_record(self):
        pages, haystacks, printed = ef.collect_evidence(self.scope, self.archive)
        self.assertEqual(printed, {"O1", "O2"})
        self.assertNotIn("O2#a1", haystacks)
        self.assertTrue(pages)

    def test_no_item_is_fed_more_than_the_search_memory_page_budget(self):
        pages, _, _ = ef.collect_evidence(self.scope, self.archive)
        selected = ef.select_evidence("Alan Cooper's account", pages)
        self.assertLessEqual(len(selected), ef.EVIDENCE_MAX_PAGES)
        self.assertLessEqual(
            len(ef.render_evidence(selected).encode("utf-8")),
            ef.EVIDENCE_MAX_BYTES + 200)  # + the block headers

    def test_selection_prefers_the_page_that_carries_the_words(self):
        pages, _, _ = ef.collect_evidence(self.scope, self.archive)
        selected = ef.select_evidence("Cloud Administrator permission uid", pages)
        self.assertEqual(selected[0].alias, "O2")

    def test_a_stored_result_page_is_evidence_and_cites_its_listing(self):
        class Store:
            @staticmethod
            def list_scope_pages(scope):
                return [{"alias": "O1", "start_offset": 200,
                         "record": {"records": [
                             {"line": "aa11  Brandon Miller"}]}}]

        pages, haystacks, printed = ef.collect_evidence(self.scope, self.archive, Store())
        page = next(p for p in pages if p.token)
        self.assertEqual(page.token, "O1#p200")
        self.assertIn("brandon miller", haystacks["O1"])
        row = ef.validate_entry(entry("x", "Brandon Miller", "O1"),
                                printed_aliases=printed, haystacks=haystacks)
        self.assertEqual(row.status, ef.FILLED)

    def test_decomposition_and_fill_over_the_fixture_archive(self):
        lm = FakeLM(
            items=["the Cloud Administrator permission uid", "Alan Cooper's account uid",
                   "the collection that confers it"],
            filled=[
                entry("the Cloud Administrator permission uid",
                      "3e3d35f0112233445566778899aabbcc", "O2"),
                entry("Alan Cooper's account uid",
                      "e8a0c3a1d4b5f6a7b8c9d0e1f2a3b4c5", "O1"),
                entry("the collection that confers it", "", "", status="unresolved",
                      reason="no collection row was retrieved"),
            ])
        with mock.patch.object(ef, "_search_lm", return_value=lm):
            sheet = ef.run("Audit both rights.", self.scope, self.archive)
        self.assertEqual(len(sheet.items), 3)
        self.assertEqual(len(sheet.filled), 2)
        self.assertEqual(sheet.items[2].status, ef.UNRESOLVED)
        self.assertGreaterEqual(sheet.calls, 2)  # decomposition + at least one fill
        self.assertIn("(Observation O2)", sheet.render())
        kinds = [e["kind"] for e in snapshot_events()]
        self.assertIn("filler_started", kinds)

    def test_a_fabricated_value_from_the_model_never_reaches_the_worksheet(self):
        lm = FakeLM(items=["the finding uid"],
                    filled=[entry("the finding uid",
                                  "cdft_0ngAyfr52SnAN0sEddnaxA", "O1")])
        with mock.patch.object(ef, "_search_lm", return_value=lm):
            sheet = ef.run("Clear the finding.", self.scope, self.archive)
        self.assertEqual(sheet.items[0].status, ef.UNRESOLVED)
        self.assertEqual(sheet.items[0].reason, ef.VALUE_NOT_IN_CITED_OBSERVATION)
        self.assertNotIn("cdft_", sheet.render())

    def test_a_decomposition_failure_is_a_filler_failure_not_a_half_worksheet(self):
        with mock.patch.object(ef, "_search_lm", return_value=FakeLM(fail=True)):
            with self.assertRaises(Exception):
                ef.run("Audit both rights.", self.scope, self.archive)

    def test_one_failed_fill_call_leaves_only_its_own_items_unresolved(self):
        calls = {"n": 0}
        good = FakeLM(items=["alpha beta", "gamma delta"],
                      filled=[entry("alpha beta", "Alan Cooper", "O1")])

        def flaky(*args, **kwargs):
            calls["n"] += 1
            return good if calls["n"] <= 2 else FakeLM(fail=True)

        with mock.patch.object(ef, "_search_lm", side_effect=flaky):
            sheet = ef.run("Audit.", self.scope, self.archive)
        self.assertEqual(len(sheet.items), 2)
        self.assertTrue(any(i.status == ef.UNRESOLVED for i in sheet.items))


class RankingAndRounds(unittest.TestCase):
    """The two pieces the pre-flight measurement put in (ido-8ps.10)."""

    def pages(self):
        # The shape that defeats word counting: the person's name is on the
        # listing that found them, and their rows are on a listing that names
        # only their account.
        named = "d1c528f3  Alisha Ochoa account"
        rows = "\n".join(f"perm{i:02d}  d1c528f3  Right {i}" for i in range(27))
        holders = "\n".join([f"uid{i:03d}  Person {i}" for i in range(60)]
                            + ["uid999  Alisha Ochoa"])
        return [ef.EvidencePage(alias="O45", text=named),
                ef.EvidencePage(alias="O46", text=rows),
                ef.EvidencePage(alias="O6", text=holders)]

    def test_rarity_and_frequency_beat_a_page_that_only_carries_the_name(self):
        pages = self.pages()
        first = ef.select_evidence("Alisha Ochoa rights", pages)
        # Nothing yet links her to those rows, so the pages her NAME is on win
        # and the 27 rows that answer the item are not fed at all - which is the
        # failure the second round exists to repair.
        self.assertEqual(first[0].alias, "O45")
        self.assertNotIn("O46", [page.alias for page in first[:1]])
        second = ef.select_evidence("Alisha Ochoa rights", pages, ["d1c528f3"])
        self.assertEqual(second[0].alias, "O46")

    def test_only_a_model_unresolved_row_is_retried(self):
        answered = {
            "a": ef.WorksheetItem(item="a", status=ef.FILLED, value="v", alias="O1"),
            "b": ef.WorksheetItem(item="b", reason="no evidence", retryable=True),
            "c": ef.WorksheetItem(item="c", reason=ef.VALUE_NOT_IN_CITED_OBSERVATION,
                                  downgraded_from="made up"),
            "d": ef.WorksheetItem(item="d", reason="call budget"),
        }
        self.assertEqual([k for k, v in answered.items() if v.retryable], ["b"])

    def test_the_expansion_is_only_validated_values_of_the_same_subject(self):
        answered = {
            "Alisha Ochoa logins": ef.WorksheetItem(
                item="Alisha Ochoa logins", status=ef.FILLED, value="d1c528f3",
                alias="O45"),
            "Anna Garcia logins": ef.WorksheetItem(
                item="Anna Garcia logins", status=ef.FILLED, value="f8feaba2",
                alias="O48"),
            "Alisha Ochoa collections": ef.WorksheetItem(
                item="Alisha Ochoa collections", reason="x", downgraded_from="z"),
        }
        self.assertEqual(ef._expanded_terms("Alisha Ochoa rights", answered),
                         ["d1c528f3"])

    def test_a_long_copy_is_bounded_for_presentation_and_says_so(self):
        value = " ".join(f"row{i:04d}" for i in range(200))
        bounded = ef.bound_value(value)
        self.assertLess(len(bounded.encode("utf-8")), len(value.encode("utf-8")))
        self.assertTrue(value.startswith(bounded.split(" [...")[0]))
        self.assertIn("more bytes in this observation", bounded)
        self.assertEqual(ef.bound_value("short"), "short")

    def test_the_bound_is_presentation_only_and_validation_saw_it_all(self):
        """A bounded line is still a literal prefix of the cited observation."""
        value = " ".join(f"row{i:04d}" for i in range(200))
        haystacks = {"O1": ef.normalise("header " + value + " footer")}
        row = ef.validate_entry(entry("rows", value, "O1"),
                                printed_aliases={"O1"}, haystacks=haystacks)
        self.assertEqual(row.status, ef.FILLED)
        self.assertIn(ef.normalise(row.render().split(" [...")[0].split(": ", 1)[1]),
                      haystacks["O1"])


class ExtractorWiring(unittest.TestCase):
    """The flag decides whether the extract prompt has the field at all."""

    class Signature(dspy.Signature):
        """Carefully review the user request, then execute the next steps."""

        user_query = dspy.InputField(desc="The natural language user query.")
        final_answer = dspy.OutputField(desc="Comprehensive final answer.")

    def setUp(self):
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / "a.sqlite3"))
        self.scope = RuntimeHandleScope("store", "channel", "exp", "task", 1, "turn")
        for name in (ef.ENABLED_ENV,):
            self.addCleanup(os.environ.pop, name, None)
        os.environ.pop(ef.ENABLED_ENV, None)

    def agent(self):
        def tool() -> str:
            """A tool."""
            return "ok"

        agent = StructuredContinuationReAct(self.Signature, tools=[tool], max_iters=3)
        agent.continuation_scope = self.scope
        agent.continuation_scope_id = self.scope.scope_id
        agent.observation_archive = self.archive
        return agent

    @staticmethod
    def rendered(predictor, **inputs):
        adapter = dspy.ChatAdapter()
        signature = getattr(predictor, "signature", None) or predictor.predict.signature
        return json.dumps(adapter.format(signature, [], inputs), sort_keys=True)

    def test_with_the_flag_off_both_prompts_are_the_ones_the_predecessor_built(self):
        """d21883d's construction, rebuilt here, must render the same bytes."""
        off = self.agent()
        os.environ[ef.ENABLED_ENV] = "1"
        on = self.agent()
        predecessor = dspy.ChainOfThought(dspy.Signature(
            {**self.Signature.input_fields, **self.Signature.output_fields},
            self.Signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str))
        inputs = {"user_query": "q", "trajectory": "t"}
        expected = self.rendered(predecessor, **inputs)
        self.assertEqual(self.rendered(off.extract, **inputs), expected)
        # The flag does not change the extract predictor the agent was built
        # with, nor the react predictor: only the extract CALL differs.
        self.assertEqual(self.rendered(on.extract, **inputs), expected)
        self.assertEqual(self.rendered(on.react, user_query="q", trajectory="t"),
                         self.rendered(off.react, user_query="q", trajectory="t"))
        self.assertEqual(on.extract.predict.signature.instructions,
                         off.extract.predict.signature.instructions)
        self.assertNotIn("verified_evidence",
                         on.extract.predict.signature.input_fields)
        self.assertNotIn("verified_evidence", on.extract_signature.input_fields)

    def test_the_evidence_signature_adds_exactly_one_input_field_last(self):
        signature = ef.evidence_extract_signature(self.agent().extract_signature)
        self.assertEqual(list(signature.input_fields)[-1], "verified_evidence")
        self.assertEqual(set(signature.input_fields) -
                         set(self.agent().extract_signature.input_fields),
                         {"verified_evidence"})
        self.assertEqual(signature.instructions,
                         self.agent().extract_signature.instructions)
        self.assertEqual(set(signature.output_fields),
                         set(self.agent().extract_signature.output_fields))

    def test_with_the_flag_off_the_extract_call_never_touches_the_filler(self):
        agent = self.agent()
        with mock.patch.object(ef, "run", side_effect=AssertionError("called")):
            with mock.patch.object(agent, "extract",
                                   return_value=dspy.Prediction(final_answer="a")):
                agent._extract_call({}, {"user_query": "q"})
        self.assertEqual([e for e in snapshot_events()
                          if str(e.get("kind", "")).startswith("filler")], [])

    def test_a_filler_failure_runs_the_extractor_exactly_as_today(self):
        os.environ[ef.ENABLED_ENV] = "1"
        agent = self.agent()
        seen = {}

        def base(**kwargs):
            seen.update(kwargs)
            return dspy.Prediction(final_answer="a")

        with mock.patch.object(ef, "run", side_effect=RuntimeError("provider")):
            with mock.patch.object(agent, "extract", side_effect=base):
                agent._extract_call({}, {"user_query": "q"})
        self.assertNotIn("verified_evidence", seen)
        kinds = [e["kind"] for e in snapshot_events()]
        self.assertIn("filler_fallback", kinds)
        self.assertIn("filler_failed", kinds)

    def test_the_worksheet_reaches_the_extractor_and_use_is_measured(self):
        os.environ[ef.ENABLED_ENV] = "1"
        agent = self.agent()
        sheet = ef.Worksheet(items=[
            ef.WorksheetItem(item="a", status=ef.FILLED, value="Alan Cooper",
                             alias="O1"),
            ef.WorksheetItem(item="b", reason="unresolved"),
        ])
        seen = {}

        def evidence_extract(**kwargs):
            seen.update(kwargs)
            return dspy.Prediction(final_answer="Alan Cooper holds it (Observation O1).")

        agent._evidence_extract = evidence_extract
        with mock.patch.object(ef, "run", return_value=sheet):
            agent._extract_call({}, {"user_query": "q"})
        self.assertEqual(seen["verified_evidence"],
                         "a: Alan Cooper (Observation O1)\nb: unresolved - unresolved")
        used = next(e for e in snapshot_events()
                    if e["kind"] == "answer_used_worksheet")
        self.assertEqual(used["validated_values"], 1)
        self.assertEqual(used["values_in_answer"], 1)
        self.assertEqual(used["rate"], 1.0)
        joined = next(e for e in snapshot_events() if e["kind"] == "filler_joined")
        self.assertTrue(joined["completed"])
        self.assertIn("join_wait_ms", joined)

    def test_the_finish_step_starts_the_filler_before_extraction(self):
        os.environ[ef.ENABLED_ENV] = "1"
        agent = self.agent()
        with mock.patch.object(ef, "run", return_value=ef.Worksheet(
                items=[ef.WorksheetItem(item="a", reason="x")])):
            agent._on_finish_selected({}, {"user_query": "q"})
            self.assertIsNotNone(agent._filler)
            agent._filler["thread"].join(timeout=5)
            self.assertIsNotNone(agent._filler.get("worksheet"))


if __name__ == "__main__":
    unittest.main()
