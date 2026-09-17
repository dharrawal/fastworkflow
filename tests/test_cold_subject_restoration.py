"""ido-dhw (F3) and ido-kmm (F4): a subject that outlives the process.

Two defects, one fixture, because the second is only observable once the first
has landed.

**F3.** The subject of an observation -- the context clause captured before the
command ran -- lived only in ``state._context_clauses``, and the auto-navigation
entry registry only in ``auto_navigation._entries``. A suspension exported
neither, and the archive deliberately stores the raw command response with the
handle line stripped, so nothing on disk said whose listing a listing was. A
turn resumed in another process therefore rehydrated its observations without
their subject, dropped a page's clause because the declaring handle's clause was
unavailable, could not attribute a name to the subject whose evidence contained
it, and could not resolve an ``O`` handle it had itself printed.

**F4.** ``search_memory`` handed the model ``handle['text']`` alone. That text is
the raw response: a ``list_permissions`` page is a table of permission rows with
nothing in it saying whose permissions they are, and the search model is
instructed to use only its supplied observation.

Everything here is offline: scripted ReAct decisions, a fixture tool, a stub
predictor that captures its input, and one temporary sidecar. No model call and
no backend.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import dspy

from fastworkflow import answer_attribution, auto_navigation, result_handles, tracing
from fastworkflow.answer_rehydration import rehydrated_label
from fastworkflow.observation_offloading.agent import build_compacting_step
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.continuation import (
    StructuredContinuationReAct,
)
from fastworkflow.observation_offloading.search import (
    declaring_subject,
    search_memory,
    subject_is_unknown,
)
from fastworkflow.observation_offloading.state import (
    context_clause_of,
    record_context_clause,
    reclaim_scope,
    reset_runtime_state,
)
from fastworkflow.result_handles import (
    ResultHandleSpec,
    ResultHandleStore,
    current_execute_alias,
    declare,
    fetch_page,
    reset_result_handle_state,
)
from fastworkflow.utils.react import AskUserSuspend

#: The subject of the listing: the identity the agent had open when it ran
#: ``list_permissions``. Exactly the ido-8ps.13 clause shape.
COOPER = "Identity 28c5aeb5b64e4ac6c40c57b0235980e2 Alan Cooper"
#: Where the agent went afterwards. The evidence must not follow it.
ELSEWHERE = "Account ACC-7"
EXPLORER = "DirectoryExplorer"

#: A permission listing exactly as a backend renders one: rows, and not one
#: word about whose permissions they are. This is the F4 evidence case.
PERMISSION_ROWS = [
    "perm-%03d  %s  %s" % (index, verb, target)
    for index, (verb, target) in enumerate(
        [("read", "payroll"), ("write", "payroll"), ("read", "hr-archive"),
         ("approve", "expenses"), ("read", "directory"), ("write", "directory"),
         ("read", "audit-log"), ("admin", "billing")]
    )
]
PERMISSION_TEXT = "\n".join(PERMISSION_ROWS)


def a_scope(turn: str = "turn-1") -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store", channel_id="channel", experiment_id="unbound",
        task_id="task-1", attempt=1, turn_key=turn,
    )


class ColdSubjectRestorationTests(unittest.TestCase):
    """Record a turn, wipe the process, restore from the sidecar alone."""

    class Signature(dspy.Signature):
        user_query: str = dspy.InputField()
        final_answer: str = dspy.OutputField()

    #: The entry contract rule 3 decides against. One required parameter, so a
    #: handle has to identify the instance; that is the interesting case.
    CONTRACT = auto_navigation.EntryContract(
        context="Account",
        declaration="open_account_by_uid <account_uid>ACC-1</account_uid>",
        command_name="open_account_by_uid",
        qualified_command_name="Account/open_account_by_uid",
        owner_contexts=("Account",),
        required_parameters=("account_uid",),
        optional_parameters=("verbose",),
    )

    def setUp(self) -> None:
        self.wipe()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.wipe)
        # One file, as production has it: ``result_handles.store()`` is keyed by
        # the agent's archive path, so the declaration and the observation live
        # side by side and the subject rows land beside both.
        self.sidecar = os.path.join(self.temp.name, "obs.sqlite3.offload-handles")
        self.scope = a_scope()
        self.open_stores()

    @staticmethod
    def wipe() -> None:
        """Everything this process remembers about any turn, gone."""
        reset_result_handle_state()
        auto_navigation.reset_auto_navigation_state()
        reset_runtime_state()

    def open_stores(self) -> None:
        """Fresh objects over the same file: what a restarted process gets."""
        self.archive = RuntimeHandleArchive(self.sidecar)
        self.store = ResultHandleStore(self.sidecar)

    # -- the turn ----------------------------------------------------------

    def _tools(self):
        def execute_workflow_query(command: str) -> str:
            """A listing, a context entry, or a page of the listing."""
            alias = current_execute_alias()
            if command == "list_permissions":
                record_context_clause(self.scope, alias, COOPER)
                declare(
                    ResultHandleSpec(kind="permission", summary="permissions",
                                     items=PERMISSION_ROWS,
                                     total=len(PERMISSION_ROWS), page_size=4,
                                     source_complete=True),
                    scope=self.scope, selected_store=self.store,
                )
                return PERMISSION_TEXT
            if command == "open_account_by_uid":
                record_context_clause(self.scope, alias, EXPLORER)
                auto_navigation.record_context_entry(
                    self.scope.scope_id, scope=self.scope, context="Account",
                    command_name="open_account_by_uid",
                    parameters={"account_uid": "ACC-7", "verbose": "True"},
                    alias=alias, required_parameters=("account_uid",),
                )
                return "opened ACC-7"
            if command.startswith("page "):
                # The agent has MOVED: the dispatch-time stamp here is the
                # account it is standing in, and the page is still Cooper's.
                record_context_clause(self.scope, alias, ELSEWHERE)
                handle, _, cursor = command.removeprefix("page ").partition(" ")
                page = fetch_page(handle, cursor, scope=self.scope,
                                  selected_store=self.store, budget_bytes=400)
                self.pages.append(page)
                return page.as_observation()
            raise AssertionError("unscripted command %r" % command)

        def ask_user(question: str) -> str:
            """Suspend the turn on a local fixture question."""
            raise AskUserSuspend(question)

        return [execute_workflow_query, ask_user]

    def _agent(self) -> StructuredContinuationReAct:
        holder: dict = {}
        agent = StructuredContinuationReAct(
            self.Signature,
            tools=self._tools(),
            max_iters=8,
            on_step_complete=build_compacting_step(
                lambda: holder.get("agent"),
                fallback_scope=self.scope,
                selected_archive=self.archive,
            ),
            scope_factory=lambda: self.scope,
        )
        holder["agent"] = agent
        agent.observation_archive = self.archive
        agent.extract = lambda **kwargs: dspy.Prediction(final_answer="done")
        return agent

    @staticmethod
    def _script(agent, steps) -> None:
        queue = iter(steps)

        def decide(**kwargs):
            name, arguments = next(queue)
            return dspy.Prediction(next_thought="scripted",
                                   next_tool_name=name, next_tool_args=arguments)

        agent.react = decide

    def _run(self, agent, steps, *, resume_with: str = "") -> dspy.Prediction:
        self.pages: list = getattr(self, "pages", [])
        self._script(agent, steps)
        with tracing.host_scope(SimpleNamespace(workflow_tool_agent=agent)):
            if resume_with:
                return agent.resume(resume_with)
            return agent.forward(user_query="who can approve expenses?")

    def warm_turn(self) -> dict:
        """O1 the listing, O2 the account entry, O3 a page of O1, then suspend."""
        self.pages = []
        agent = self._agent()
        prediction = self._run(agent, [
            ("execute_workflow_query", {"command": "list_permissions"}),
            ("execute_workflow_query", {"command": "open_account_by_uid"}),
            ("execute_workflow_query", {"command": "page O1 "}),
            ("ask_user", {"question": "which account?"}),
        ])
        self.assertTrue(prediction.suspended)
        # Exactly what the session state file carries: JSON, nothing else.
        return json.loads(json.dumps(agent.export_suspended()))

    # -- what a reader can say about the turn ------------------------------

    def _rule_three(self, utterance: str) -> auto_navigation.AutoNavigationDecision:
        """Rule 3 against the registry as this process can see it."""
        return auto_navigation.decide(
            command_name="list_entitlements",
            utterance=utterance,
            owner_contexts=("Account",),
            contracts={"Account": self.CONTRACT},
            # Named explicitly because this reader runs outside a dispatch, so
            # there is no tracing host to resolve the turn's archive from.
            entries=auto_navigation.context_entries(
                self.scope.scope_id, selected_archive=self.archive),
        )

    def _subject_of(self, alias: str) -> str:
        """The metadata ``search_memory`` hands the model, via a stub predictor."""
        seen: dict = {}
        lm = SimpleNamespace(history=[{"usage": {"completion_tokens": 3}}],
                             model="fixture-lm")

        def predict(_signature):
            def call(question, subject, observation):
                seen["subject"] = subject
                seen["observation"] = observation
                return SimpleNamespace(answer="see rows")
            return call

        with patch("fastworkflow.observation_offloading.search.get_lm",
                   return_value=lm), \
                patch("fastworkflow.observation_offloading.search.dspy") as fake:
            fake.Predict.side_effect = predict
            search_memory("Whose permissions are these?", alias,
                          scope=self.scope, selected_archive=self.archive)
        self.seen_by_search = seen
        return seen["subject"]

    def readings(self) -> dict:
        """Everything F3 says is lost across a restart, read in one place."""
        label = rehydrated_label("O1", scope=self.scope, archive=self.archive)
        clauses = {
            item.alias: item.clause
            for item in answer_attribution.observations(
                scope=self.scope, archive=self.archive, handle_store=self.store)
        }
        decision = self._rule_three("list_entitlements O2")
        return {
            "label_line": (label or "").splitlines()[0] if label else None,
            "page_clause": context_clause_of(self.scope, "O3",
                                             selected_archive=self.archive),
            "attribution": clauses,
            "navigation": (decision.kind, decision.rule, decision.entered_context,
                           decision.entry_parameters, decision.handle),
            "search_subject": self._subject_of("O1"),
        }

    # -- the tests ---------------------------------------------------------

    def test_a_restored_turn_reads_exactly_as_the_turn_that_recorded_it(self) -> None:
        blob = self.warm_turn()
        warm = self.readings()

        # The warm run really did record all four facts, or the comparison
        # below would be an equality between two identical absences.
        self.assertIn(COOPER, warm["label_line"])
        self.assertEqual(warm["page_clause"], COOPER)
        self.assertEqual(warm["attribution"]["O1"], COOPER.lower())
        self.assertEqual(warm["navigation"][:3],
                         (auto_navigation.DISPATCH,
                          auto_navigation.RULE_EXPLICIT_HANDLE, "Account"))
        self.assertIn(COOPER, warm["search_subject"])

        # A restart: every process-local cache dropped, every store object
        # rebuilt. All that crosses the gap is the suspension payload, and the
        # only thing in it this fix relies on is the scope.
        self.wipe()
        self.open_stores()
        resumed = self._agent()
        resumed.import_suspended(blob)
        self.assertEqual(resumed.continuation_scope, self.scope)

        self.assertEqual(self.readings(), warm)

    def test_a_page_fetched_after_the_restart_still_carries_its_handles_subject(
        self,
    ) -> None:
        blob = self.warm_turn()
        first_page = self.pages[0]
        self.assertTrue(first_page.next_cursor)

        self.wipe()
        self.open_stores()
        resumed = self._agent()
        resumed.import_suspended(blob)
        self._run(resumed, [
            ("execute_workflow_query",
             {"command": "page O1 %s" % first_page.next_cursor}),
            ("finish", {}),
        ], resume_with="the second account")

        # O4 was dispatched from ELSEWHERE and is still Cooper's page: the
        # declaring handle's subject was read back from the sidecar.
        self.assertEqual(
            context_clause_of(self.scope, "O4", selected_archive=self.archive),
            COOPER)
        self.assertEqual(len(self.pages), 2)
        self.assertNotEqual(self.pages[1].rows, self.pages[0].rows)

    def test_search_is_given_the_recorded_subject_of_a_subjectless_listing(
        self,
    ) -> None:
        self.warm_turn()
        # The active workflow has moved to the account; the observation has not.
        self.assertEqual(
            context_clause_of(self.scope, "O2", selected_archive=self.archive),
            EXPLORER)

        subject = self._subject_of("O1")
        observation = self.seen_by_search["observation"]

        # The body really is subjectless -- that is the defect.
        self.assertNotIn("Cooper", observation)
        self.assertIn("perm-000", observation)
        # And the subject reaches the model beside it, naming O1 and the
        # identity the command ran in, not the account the agent moved to.
        self.assertIn(COOPER, subject)
        self.assertIn("O1", subject)
        self.assertNotIn(ELSEWHERE, subject)
        self.assertFalse(subject_is_unknown(subject))

    def test_the_subject_is_paid_for_out_of_the_search_models_budget(self) -> None:
        from fastworkflow.observation_offloading.search import (
            evidence_max_bytes, search_observation_max_bytes,
        )

        budget = search_observation_max_bytes()
        record_context_clause(self.scope, "O9", COOPER,
                              selected_archive=self.archive)
        text = "row  value\n" * 40_000
        self.archive.persist(self.scope, alias="O9", offload_order=9,
                             command_name="show_rows", step_index=8, text=text,
                             text_sha256=hashlib.sha256(text.encode()).hexdigest())
        subject = self._subject_of("O9")
        sent = self.seen_by_search["observation"]
        # The evidence fills the budget the subject leaves, to the last whole
        # row: a line boundary, never a byte more.
        self.assertLessEqual(len(sent.encode("utf-8")),
                             evidence_max_bytes(subject, budget))
        self.assertGreater(len(sent.encode("utf-8")),
                           evidence_max_bytes(subject, budget) - 20)
        self.assertLess(evidence_max_bytes(subject, budget), budget)
        # The whole input -- metadata and evidence together -- stays inside the
        # one bound derived from the search model's window.
        self.assertLessEqual(
            len(subject.encode("utf-8")) + len(sent.encode("utf-8")), budget)

    def test_a_record_written_before_this_existed_reads_as_explicitly_unknown(
        self,
    ) -> None:
        """No subject row, no guess: the oldest sidecar stays readable."""
        text = "\n".join(PERMISSION_ROWS)
        self.archive.persist(self.scope, alias="O7", offload_order=7,
                             command_name="list_permissions", step_index=6,
                             text=text,
                             text_sha256=hashlib.sha256(text.encode()).hexdigest())
        self.assertIsNone(self.archive.get_subject(self.scope, "O7"))
        self.assertIsNone(
            context_clause_of(self.scope, "O7", selected_archive=self.archive))

        subject = self._subject_of("O7")
        self.assertTrue(subject_is_unknown(subject))
        self.assertIn("O7", subject)
        # Not a guess from the alias, the question or the workflow's context.
        self.assertNotIn("Cooper", subject)
        # And the handle line it rehydrates with is the plain one.
        label = rehydrated_label("O7", scope=self.scope, archive=self.archive)
        self.assertEqual(label.splitlines()[0],
                         "Observation O7 (execute_workflow_query)")

    def test_reclaiming_a_scope_drops_residency_and_keeps_the_records(self) -> None:
        self.warm_turn()
        warm = self.readings()

        reclaim_scope(self.scope)

        from fastworkflow.observation_offloading import state as offload_state

        # Residency really went.
        self.assertEqual(offload_state.stored_handles(self.scope), {})
        self.assertNotIn(offload_state.handle_key(self.scope, "O1"),
                         offload_state._context_clauses)
        self.assertIsNone(auto_navigation._entries.get(self.scope.scope_id))
        # Evidence did not: every reading rebuilds from the sidecar.
        self.assertEqual(self.readings(), warm)

    def test_the_declaring_subject_field_keeps_its_three_states_apart(self) -> None:
        recorded = declaring_subject("O1", COOPER, "list_permissions")
        root = declaring_subject("O1", "", "list_permissions")
        unknown = declaring_subject("O1", None, "list_permissions")
        self.assertIn(COOPER, recorded)
        self.assertFalse(subject_is_unknown(recorded))
        self.assertIn("workflow root", root)
        self.assertFalse(subject_is_unknown(root))
        self.assertTrue(subject_is_unknown(unknown))
        self.assertNotEqual(root, unknown)


class ErasureAndPreservationTests(unittest.TestCase):
    """The new rows are evidence, and the sidecar's own policy governs them."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sidecar = os.path.join(self.temp.name, "obs.sqlite3.offload-handles")
        self.archive = RuntimeHandleArchive(self.sidecar)

    def _fill(self, scope: RuntimeHandleScope) -> None:
        self.archive.put_subject(scope, "O1", COOPER)
        self.archive.put_context_entry(
            scope, sequence=1, context="Account",
            command_name="open_account_by_uid",
            parameters={"account_uid": "ACC-7"}, alias="O1",
            required_parameters=("account_uid",),
        )

    def test_both_tables_are_discovered_structurally(self) -> None:
        import sqlite3

        from fastworkflow.observation_offloading import erasure

        with sqlite3.connect(self.sidecar) as conn:
            discovered = erasure.evidence_tables(conn)
        for name in ("observation_subjects", "observation_context_entries"):
            self.assertIn(name, discovered)
            self.assertTrue(discovered[name]["scope_json"], name)
            self.assertEqual(discovered[name]["timestamp"], "recorded_at", name)

    def test_a_chatbot_channel_is_erased_and_an_experiment_is_preserved(self) -> None:
        from fastworkflow.observation_offloading import erasure

        chat = RuntimeHandleScope("store", "chan", "unbound", "task", 1, "turn-a")
        run = RuntimeHandleScope("store", "chan", "exp-9", "task", 1, "turn-b")
        self._fill(chat)
        self._fill(run)

        result = erasure.forget_channel(self.sidecar, "chan")

        self.assertEqual(result["observation_subjects"], 1)
        self.assertEqual(result["observation_context_entries"], 1)
        self.assertIsNone(self.archive.get_subject(chat, "O1"))
        self.assertEqual(self.archive.list_context_entries(chat), [])
        self.assertEqual(self.archive.get_subject(run, "O1"), COOPER)
        self.assertEqual(len(self.archive.list_context_entries(run)), 1)
        with open(self.sidecar, "rb") as handle:
            self.assertNotIn(chat.scope_id.encode("ascii"), handle.read())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
