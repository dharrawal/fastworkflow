"""ido-8ps.15: the readers that produce the answer can see the context line.

``ido-8ps.13`` printed the context instance on every execute observation and
then measured that no reader which produces the final answer could see it: the
archive stores the raw command response (the A1 convention), and the evidence
filler, ``search_memory`` and the extract step all read stored text. This suite
covers the four places the clause now travels -- the archive row's metadata, the
filler's pages and its ranking, the observation handed to the search model, and
the worksheet line the extractor reads -- and the two things that must NOT move:
the stored response and its digest.

Offline: no model, no server, no backend.
"""
import hashlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from fastworkflow import evidence_filler as ef
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.labels import alias_line, context_clause
from fastworkflow.observation_offloading.search import search_memory
from fastworkflow.observation_offloading.state import (
    observation_context,
    record_context_clause,
    reset_runtime_state,
    snapshot_events,
)

ALAN_UID = "e8a0c3a1d4b5f6a7b8c9d0e1f2a3b4c5"
ALISHA_UID = "d1c528f3aaaabbbbccccddddeeeeffff"
ALAN_CLAUSE = context_clause("Account", f"{ALAN_UID} Alan Cooper")
ALISHA_CLAUSE = context_clause("Account", f"{ALISHA_UID} Alisha Ochoa")

#: The listing that FINDS the people: it prints uid and name, and nothing else.
ROSTER = "\n".join(
    ["477 holder(s)"]
    + [f"{i:032x}  Person {i}" for i in range(40)]
    + [f"{ALAN_UID}  Alan Cooper", f"{ALISHA_UID}  Alisha Ochoa"]
)
#: The listings that ANSWER the question. Produced by navigating into an account
#: and calling ``list_permissions`` there, so -- and this is the whole of the
#: ido-8ps.10 finding -- no row names the account, let alone the person.
ALAN_PERMISSIONS = (
    "29 permission(s)\n"
    "3e3d35f0112233445566778899aabbcc  Active Directory_Cloud Administrator\n"
    "85cde16899887766554433221100ffee  Active Directory_Compliance Officer")
ALISHA_PERMISSIONS = (
    "12 permission(s)\n"
    "77aa11bb22cc33dd44ee55ff66007788  Okta_Finance Reader\n"
    "99cc88dd77ee66ff5500112233445566  Okta_Payments Approver")
LATER = "3 finding(s)\nctrl_aa1122  open\nctrl_bb3344  closed"


def scope_for(name: str) -> RuntimeHandleScope:
    return RuntimeHandleScope(store_identity=name, channel_id="c", experiment_id="e",
                              task_id="t", attempt=1, turn_key="turn-1")


def digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ArchiveMetadata(unittest.TestCase):
    """The clause is stored beside the text, and the text does not change."""

    def setUp(self):
        reset_runtime_state()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.archive = RuntimeHandleArchive(os.path.join(self.dir.name, "a.sqlite3"))
        self.scope = scope_for("metadata")

    def persist(self, alias, text, context=None, order=1):
        self.archive.persist(self.scope, alias=alias, offload_order=order,
                             command_name="list_permissions", step_index=order,
                             text=text, text_sha256=digest_of(text), context=context)

    def test_the_clause_round_trips_on_get_and_on_list(self):
        self.persist("O1", ALAN_PERMISSIONS, ALAN_CLAUSE)
        self.assertEqual(self.archive.get(self.scope, "O1")["context"], ALAN_CLAUSE)
        self.assertEqual(self.archive.list(self.scope)[0]["context"], ALAN_CLAUSE)
        self.assertEqual(self.archive.context_clause(self.scope, "O1"), ALAN_CLAUSE)
        self.assertEqual(self.archive.context_clauses(self.scope), {"O1": ALAN_CLAUSE})

    def test_root_and_unrecorded_are_different_facts(self):
        """"" means it ran at the root; None means nothing was captured."""
        self.persist("O1", LATER, "", order=1)
        self.persist("O2", ALAN_PERMISSIONS, None, order=2)
        self.assertEqual(self.archive.context_clause(self.scope, "O1"), "")
        self.assertIsNone(self.archive.context_clause(self.scope, "O2"))
        self.assertEqual(self.archive.get(self.scope, "O1")["context"], "")
        self.assertIsNone(self.archive.get(self.scope, "O2")["context"])

    def test_the_stored_response_and_its_digest_are_untouched(self):
        """A clause is metadata: the evidence record is byte-identical."""
        plain = scope_for("plain")
        self.persist("O1", ALAN_PERMISSIONS, ALAN_CLAUSE)
        self.archive.persist(plain, alias="O1", offload_order=1,
                             command_name="list_permissions", step_index=1,
                             text=ALAN_PERMISSIONS,
                             text_sha256=digest_of(ALAN_PERMISSIONS))
        with_clause = self.archive.get(self.scope, "O1")
        without = self.archive.get(plain, "O1")
        self.assertEqual(with_clause["text"], ALAN_PERMISSIONS)
        self.assertEqual(with_clause["text"], without["text"])
        self.assertEqual(with_clause["text_sha256"], without["text_sha256"])
        self.assertEqual(with_clause["text_sha256"], digest_of(ALAN_PERMISSIONS))
        self.assertNotIn("Alan Cooper", with_clause["text"])

    def test_the_first_clause_recorded_for_an_alias_stands(self):
        self.persist("O1", ALAN_PERMISSIONS, ALAN_CLAUSE)
        self.persist("O1", ALAN_PERMISSIONS, ALISHA_CLAUSE)
        self.assertEqual(self.archive.context_clause(self.scope, "O1"), ALAN_CLAUSE)

    def test_record_context_files_a_clause_for_stored_text(self):
        self.persist("O1", ALAN_PERMISSIONS)
        self.assertIsNone(self.archive.context_clause(self.scope, "O1"))
        self.archive.record_context(self.scope, "O1", ALAN_CLAUSE)
        self.assertEqual(self.archive.get(self.scope, "O1")["context"], ALAN_CLAUSE)

    def test_the_resolver_prefers_this_process_then_the_archive(self):
        self.persist("O1", ALAN_PERMISSIONS, ALAN_CLAUSE)
        self.assertEqual(observation_context(self.scope, "O1", self.archive),
                         ALAN_CLAUSE)
        record_context_clause(self.scope, "O2", ALISHA_CLAUSE)
        self.assertEqual(observation_context(self.scope, "O2", self.archive),
                         ALISHA_CLAUSE)
        self.assertIsNone(observation_context(self.scope, "O9", self.archive))


def pages_of(contexts: bool) -> list[ef.EvidencePage]:
    """The turn: a roster, two context-scoped listings, a later unrelated one."""
    rows = [("O1", ROSTER, ""), ("O2", ALAN_PERMISSIONS, ALAN_CLAUSE),
            ("O3", ALISHA_PERMISSIONS, ALISHA_CLAUSE), ("O4", LATER, "")]
    return [page for alias, text, clause in rows
            for page in ef._paginate(alias, text,
                                     context=clause if contexts else "")]


class PagesAndRanking(unittest.TestCase):
    """Every page says whose it is, and the question about a person finds it."""

    def test_the_page_header_is_the_printed_provenance_clause(self):
        page = ef.EvidencePage(alias="O2", text=ALAN_PERMISSIONS, context=ALAN_CLAUSE)
        self.assertEqual(
            page.header(),
            f"Observation O2 (execute_workflow_query, in {ALAN_CLAUSE}):")
        self.assertIn(page.header(), ef.render_evidence([page]))
        self.assertIn(ALAN_CLAUSE, ef.render_evidence([page]))

    def test_a_page_with_no_clause_keeps_the_plain_header(self):
        self.assertEqual(ef.EvidencePage(alias="O4", text=LATER).header(),
                         "Observation O4 (execute_workflow_query):")

    def test_a_stored_result_page_keeps_its_token_and_gains_the_clause(self):
        page = ef.EvidencePage(alias="O2", text=ALAN_PERMISSIONS,
                               token="O2#p4096", context=ALAN_CLAUSE)
        self.assertEqual(
            page.header(),
            f"Observation O2 (execute_workflow_query, in {ALAN_CLAUSE}, "
            "stored page O2#p4096):")

    def test_every_page_of_a_long_listing_carries_the_clause(self):
        long_text = "row\n" * 4_000
        pages = ef._paginate("O2", long_text, context=ALAN_CLAUSE)
        self.assertGreater(len(pages), 1)
        self.assertTrue(all(page.context == ALAN_CLAUSE for page in pages))
        self.assertEqual("".join(page.text for page in pages), long_text)

    def test_without_the_clause_the_persons_question_misses_their_listing(self):
        """The ido-8ps.10 failure, reproduced: 2 of 36 row-attempts."""
        index = ef.EvidenceIndex(pages_of(contexts=False))
        self.assertEqual(index.score(1, ef.query_terms("Alan Cooper rights"))[0], 0.0)
        self.assertNotEqual(index.select("Alan Cooper rights")[0].alias, "O2")

    def test_with_the_clause_the_scoped_listing_ranks_first(self):
        index = ef.EvidenceIndex(pages_of(contexts=True))
        self.assertEqual(index.select("Alan Cooper rights")[0].alias, "O2")
        self.assertEqual(index.select("Alisha Ochoa collections")[0].alias, "O3")

    def test_the_clause_does_not_select_the_other_persons_listing(self):
        index = ef.EvidenceIndex(pages_of(contexts=True))
        order = [page.alias for page in index.select("Alan Cooper rights")]
        self.assertEqual(order[0], "O2")
        self.assertNotIn("O3", order)
        terms = ef.query_terms("Alan Cooper rights")
        self.assertGreater(index.score(1, terms)[0], index.score(2, terms)[0])


class CollectAndValidate(unittest.TestCase):
    """What the archive hands the filler, and what the rule then accepts."""

    def setUp(self):
        reset_runtime_state()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.archive = RuntimeHandleArchive(os.path.join(self.dir.name, "a.sqlite3"))
        self.scope = scope_for("collect")
        for order, (alias, text, clause) in enumerate(
                [("O1", ROSTER, ""), ("O2", ALAN_PERMISSIONS, ALAN_CLAUSE),
                 ("O3", ALISHA_PERMISSIONS, ALISHA_CLAUSE)], start=1):
            self.archive.persist(self.scope, alias=alias, offload_order=order,
                                 command_name="list_permissions", step_index=order,
                                 text=text, text_sha256=digest_of(text),
                                 context=clause)
        self.pages, self.haystacks, self.printed = ef.collect_evidence(
            self.scope, self.archive)

    def test_collected_pages_carry_the_recorded_clause(self):
        by_alias = {page.alias: page for page in self.pages}
        self.assertEqual(by_alias["O2"].context, ALAN_CLAUSE)
        self.assertEqual(by_alias["O1"].context, "")
        self.assertEqual(self.printed, {"O1", "O2", "O3"})

    def test_the_clause_is_part_of_the_haystack_of_its_own_observation(self):
        """The stated choice: a uid may be cited from the scope line."""
        self.assertIn(ef.normalise(ALAN_UID), self.haystacks["O2"])
        self.assertIn(ef.normalise("Alan Cooper"), self.haystacks["O2"])
        self.assertNotIn(ef.normalise("Alan Cooper"), self.haystacks["O3"])

    def test_a_value_from_the_scope_line_validates_against_that_observation(self):
        row = ef.validate_entry(
            {"item": "Alan Cooper account", "value": ALAN_UID, "observation": "O2",
             "status": "filled"},
            printed_aliases=self.printed, haystacks=self.haystacks,
            contexts={"O2": ALAN_CLAUSE})
        self.assertEqual(row.status, ef.FILLED)
        self.assertEqual(row.context, ALAN_CLAUSE)

    def test_the_clause_does_not_license_a_value_nothing_recorded(self):
        """Widened evidence, unchanged rule."""
        row = ef.validate_entry(
            {"item": "Alan Cooper rights", "value": "Okta_Finance Reader",
             "observation": "O2", "status": "filled"},
            printed_aliases=self.printed, haystacks=self.haystacks,
            contexts={"O2": ALAN_CLAUSE})
        self.assertEqual(row.status, ef.UNRESOLVED)
        self.assertEqual(row.reason, ef.VALUE_NOT_IN_CITED_OBSERVATION)

    def test_another_observations_clause_is_not_this_observations_evidence(self):
        row = ef.validate_entry(
            {"item": "Alan Cooper account", "value": ALAN_UID, "observation": "O3",
             "status": "filled"},
            printed_aliases=self.printed, haystacks=self.haystacks,
            contexts={"O3": ALISHA_CLAUSE})
        self.assertEqual(row.status, ef.UNRESOLVED)
        self.assertEqual(row.reason, ef.VALUE_NOT_IN_CITED_OBSERVATION)

    def test_an_alias_outside_the_namespace_is_still_refused(self):
        row = ef.validate_entry(
            {"item": "Alan Cooper rights", "value": "29 permission(s)",
             "observation": "O9", "status": "filled"},
            printed_aliases=self.printed, haystacks=self.haystacks,
            contexts={"O9": ALAN_CLAUSE})
        self.assertEqual(row.reason, ef.ALIAS_NOT_PRINTED)

    def test_an_archive_written_before_the_metadata_falls_back_to_the_process(self):
        plain = scope_for("older")
        self.archive.persist(plain, alias="O1", offload_order=1,
                             command_name="list_permissions", step_index=1,
                             text=ALAN_PERMISSIONS,
                             text_sha256=digest_of(ALAN_PERMISSIONS))
        record_context_clause(plain, "O1", ALAN_CLAUSE)
        pages, haystacks, _ = ef.collect_evidence(plain, self.archive)
        self.assertEqual(pages[0].context, ALAN_CLAUSE)
        self.assertIn(ef.normalise(ALAN_UID), haystacks["O1"])


class EndToEndThroughCompaction(unittest.TestCase):
    """The production path: dispatch records it, the step archives it."""

    def setUp(self):
        reset_runtime_state()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.archive = RuntimeHandleArchive(os.path.join(self.dir.name, "a.sqlite3"))
        self.scope = scope_for("end-to-end")

    def test_the_clause_the_line_printed_is_the_clause_the_archive_stores(self):
        from fastworkflow.observation_offloading.compact import (
            annotate_execute_observations,
            archive_execute_observations,
        )

        trajectory = {}
        for index, text in enumerate([ROSTER, ALAN_PERMISSIONS, LATER]):
            trajectory[f"tool_name_{index}"] = "execute_workflow_query"
            trajectory[f"tool_args_{index}"] = {"command": "list_permissions"}
            trajectory[f"observation_{index}"] = text
        # What CommandExecutor._remember_execute_context files at dispatch.
        record_context_clause(self.scope, "O1", "")
        record_context_clause(self.scope, "O2", ALAN_CLAUSE)
        annotate_execute_observations(trajectory, scope=self.scope)
        archive_execute_observations(trajectory, scope=self.scope,
                                     selected_archive=self.archive)
        self.assertTrue(trajectory["observation_1"].startswith(
            alias_line("O2", ALAN_CLAUSE)))
        self.assertEqual(self.archive.context_clause(self.scope, "O2"), ALAN_CLAUSE)
        self.assertEqual(self.archive.context_clause(self.scope, "O1"), "")
        # O3 was never dispatched through the hook: nothing is invented for it.
        self.assertIsNone(self.archive.context_clause(self.scope, "O3"))
        self.assertEqual(self.archive.get(self.scope, "O2")["text"], ALAN_PERMISSIONS)
        self.assertEqual(self.archive.get(self.scope, "O2")["text_sha256"],
                         digest_of(ALAN_PERMISSIONS))
        pages, haystacks, printed = ef.collect_evidence(self.scope, self.archive)
        self.assertEqual(
            ef.EvidenceIndex(pages).select("Alan Cooper rights")[0].alias, "O2")


class WorksheetLine(unittest.TestCase):
    """What the extract step reads: the value, its handle, and whose it is."""

    def test_a_filled_line_names_the_context_instance(self):
        item = ef.WorksheetItem(item="Alan Cooper rights", status=ef.FILLED,
                                value="Active Directory_Cloud Administrator",
                                alias="O2", context=ALAN_CLAUSE)
        self.assertEqual(
            item.render(),
            "Alan Cooper rights: Active Directory_Cloud Administrator "
            f"(Observation O2, in {ALAN_CLAUSE})")

    def test_a_page_citation_keeps_both(self):
        item = ef.WorksheetItem(item="Alan Cooper rights", status=ef.FILLED,
                                value="29 permission(s)", alias="O2",
                                page="O2#p4096", context=ALAN_CLAUSE)
        self.assertEqual(
            item.render(),
            "Alan Cooper rights: 29 permission(s) "
            f"(Observation O2, in {ALAN_CLAUSE}, page O2#p4096)")

    def test_a_line_with_no_recorded_clause_is_exactly_what_it_was(self):
        item = ef.WorksheetItem(item="the findings", status=ef.FILLED,
                                value="3 finding(s)", alias="O4")
        self.assertEqual(item.render(), "the findings: 3 finding(s) (Observation O4)")

    def test_an_unresolved_line_is_unchanged(self):
        item = ef.WorksheetItem(item="the feed", reason="no observation links it")
        self.assertEqual(item.render(), "the feed: unresolved - no observation links it")


class SearchMemoryProvenance(unittest.TestCase):
    """The search model is shown the line; the archive keeps the raw text."""

    def setUp(self):
        reset_runtime_state()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.archive = RuntimeHandleArchive(os.path.join(self.dir.name, "a.sqlite3"))
        self.scope = scope_for("search")
        self.archive.persist(self.scope, alias="O2", offload_order=2,
                             command_name="list_permissions", step_index=2,
                             text=ALAN_PERMISSIONS,
                             text_sha256=digest_of(ALAN_PERMISSIONS),
                             context=ALAN_CLAUSE)

    def search(self, alias="O2"):
        seen = {}
        lm = SimpleNamespace(history=[{"usage": {"completion_tokens": 7}, "cost": 0.0}],
                             model="fixture-lm")

        def predict(_signature):
            def call(question, observation):
                seen["observation"] = observation
                return SimpleNamespace(answer="an answer")
            return call

        with mock.patch("fastworkflow.observation_offloading.search.get_lm",
                        return_value=lm), \
                mock.patch("fastworkflow.observation_offloading.search.dspy") as fake:
            fake.Predict.side_effect = predict
            seen["returned"] = search_memory("Whose rights are these?", alias,
                                             scope=self.scope,
                                             selected_archive=self.archive)
        seen["event"] = [e for e in snapshot_events()
                         if e["kind"] == "search_memory"][-1]
        return seen

    def test_the_observation_handed_to_the_model_is_prefixed_with_the_clause(self):
        seen = self.search()
        self.assertEqual(seen["observation"],
                         alias_line("O2", ALAN_CLAUSE) + ALAN_PERMISSIONS)
        self.assertTrue(seen["observation"].startswith(
            f"Observation O2 (execute_workflow_query, in {ALAN_CLAUSE})\n"))

    def test_the_archived_text_stays_raw(self):
        self.search()
        row = self.archive.get(self.scope, "O2")
        self.assertEqual(row["text"], ALAN_PERMISSIONS)
        self.assertEqual(row["text_sha256"], digest_of(ALAN_PERMISSIONS))

    def test_the_event_records_the_clause_and_both_sizes(self):
        event = self.search()["event"]
        self.assertEqual(event["context"], ALAN_CLAUSE)
        self.assertEqual(event["observation_bytes"],
                         len(ALAN_PERMISSIONS.encode("utf-8")))
        self.assertEqual(
            event["presented_observation_bytes"],
            len((alias_line("O2", ALAN_CLAUSE) + ALAN_PERMISSIONS).encode("utf-8")))

    def test_an_observation_with_no_recorded_clause_gets_the_plain_a1_line(self):
        self.archive.persist(self.scope, alias="O4", offload_order=4,
                             command_name="list_findings", step_index=4,
                             text=LATER, text_sha256=digest_of(LATER))
        seen = self.search("O4")
        self.assertEqual(seen["observation"], alias_line("O4") + LATER)
        self.assertEqual(seen["event"]["context"], "")

    def test_the_signature_tells_the_model_what_the_first_line_is(self):
        from fastworkflow.observation_offloading.search import ObservationSearchSignature

        instructions = ObservationSearchSignature.instructions
        self.assertIn("provenance line", instructions)
        self.assertIn("context instance", instructions)


if __name__ == "__main__":
    unittest.main()
