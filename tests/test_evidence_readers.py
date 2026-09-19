"""Shared evidence readers preserve implicit runtime source resolution."""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

from fastworkflow import answer_attribution, result_handles
from fastworkflow.answer_coverage import (
    Entity,
    drop_zero_match_echo,
    evidence_by_subject,
    normalise,
)
from fastworkflow.answer_rehydration import stored_rows_block
from fastworkflow.evidence_readers import observations
from fastworkflow.observation_offloading import state as offload_state
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive
from fastworkflow.observation_offloading.labels import strip_alias_line
from fastworkflow.observation_offloading.state import record_context_clause

from tests.test_answer_rehydration import declaration_payload, page_record


class DefaultEvidenceSources(unittest.TestCase):
    def test_implicit_sources_equal_explicit_real_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saved_archive = offload_state._default_archive
            archive = RuntimeHandleArchive(os.path.join(directory, "evidence.sqlite3"))
            offload_state._default_archive = archive
            result_handles.reset_result_handle_state()
            self.addCleanup(setattr, offload_state, "_default_archive", saved_archive)
            self.addCleanup(result_handles.reset_result_handle_state)
            self.addCleanup(offload_state.reset_runtime_state)

            scope = offload_state.default_scope()
            store = result_handles.store()
            text = "result_handle=O91 page 1 rows 1-1 of 2\n1  bounded row"
            archive.persist(
                scope, alias="O91", offload_order=91, command_name="list_items",
                step_index=1, text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
            record_context_clause(scope, "O91", "Person 1 Ada Lovelace")
            store.put_declaration(scope, "O91", declaration_payload(total=2))
            store.put_page(
                scope, alias="O91", query_scope="", batch_index=0,
                limit_requested=25, source="resolver", backend_total=2,
                record=page_record(["1  bounded row", "2  durable row"]),
            )

            explicit = observations(
                scope=scope, archive=archive, handle_store=store,
                strip_alias_line=strip_alias_line,
                stored_rows_block=stored_rows_block,
                drop_zero_match_echo=drop_zero_match_echo,
                normalise=normalise,
            )
            implicit = answer_attribution.observations()

            self.assertEqual(implicit, explicit)
            self.assertEqual([item.alias for item in implicit], ["O91"])
            self.assertIn("durable row", implicit[0].text)
            self.assertEqual(implicit[0].clause, "person 1 ada lovelace")

            entities = [
                Entity("Ada Lovelace", "name", 0),
                Entity("Durable Row", "name", 13),
            ]
            implicit_coverage = evidence_by_subject(entities)
            explicit_coverage = evidence_by_subject(
                entities, scope=scope, archive=archive, handle_store=store
            )
            self.assertEqual(implicit_coverage, explicit_coverage)
            self.assertEqual(
                [(subject.text, [item.text for item in items])
                 for subject, items in implicit_coverage],
                [("Ada Lovelace", ["Durable Row"])],
            )


if __name__ == "__main__":
    unittest.main()
