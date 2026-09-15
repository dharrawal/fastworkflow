"""Scoped archive and label integration, plus opt-in real DSPy provider tests."""
import hashlib
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastworkflow.observation_offloading.agent import current_search_reasoning
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.continuation import replan_trajectory_skeleton
from fastworkflow.observation_offloading.labels import offload_label, label_alias, is_offload_label, alias_line
from fastworkflow.observation_offloading.search import search_memory, completion_was_truncated
from fastworkflow.observation_offloading.state import reset_runtime_state, snapshot_events


class ObservationSearch(unittest.TestCase):
    def setUp(self):
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 1, 'turn')

    def persist(self, alias, text):
        self.archive.persist(self.scope, alias=alias, offload_order=int(alias[1:]),
                             command_name='show_holders', step_index=int(alias[1:])-1,
                             text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def search(self, question, alias, **kwargs):
        return search_memory(question, alias, scope=self.scope, selected_archive=self.archive, **kwargs)

    def test_truncated_provider_response_is_not_an_evidence_answer(self):
        self.assertTrue(completion_was_truncated({'response': {'choices': [{'finish_reason': 'length'}]}}))
        self.assertTrue(completion_was_truncated({'usage': {'completion_tokens': 2048}}))
        self.assertFalse(completion_was_truncated({'response': {'choices': [{'finish_reason': 'stop'}]}, 'usage': {'completion_tokens': 50}}))

    def test_alias_is_required_and_validated_before_model_call(self):
        self.assertIs(inspect.signature(search_memory).parameters['alias'].default, inspect.Parameter.empty)
        for alias in ['', 'O0', 'O1 O2', 'O-1', 'S1', 'O1; O2']:
            with self.assertRaises(ValueError):
                self.search('Who?', alias)

    def test_no_fallback_to_another_handle_or_turn(self):
        self.persist('O1', 'Secret from another observation')
        self.assertIn('no matching offloaded handle O2', self.search('Who?', 'O2'))
        other = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 2, 'another-turn')
        result = search_memory('Who?', 'O1', scope=other, selected_archive=self.archive)
        self.assertIn('no matching offloaded handle O1', result)

    def test_label_uses_command_argument_and_authored_description(self):
        label = offload_label(alias='O12', command_name='show_holders limit=100',
                             response='payload', description='identity UIDs and holder names')
        self.assertEqual(label, 'Use search_memory tool to search inside Observation O12 returned by show_holders limit=100. It was offloaded to memory and contains identity UIDs and holder names.')
        self.assertTrue(is_offload_label(label))
        self.assertEqual(label_alias(label), 'O12')

    def test_small_observations_and_long_command_arguments_never_expand(self):
        for turn, (text, command) in enumerate([("Context is now '*'", 'reset_context'), ('x'*5000, 'query '+'é'*6000)]):
            # One observation per alias per scope: each case is its own turn.
            scope = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 1, f'turn-{turn}')
            trajectory = {'tool_name_0': 'execute_workflow_query', 'tool_args_0': {'command': command}, 'observation_0': text}
            # min_offload_saving_bytes=0 removes the 1 KB floor entirely, so the
            # only thing left to refuse these is the swap itself being a loss.
            decisions = compact_trajectory(trajectory, min_offload_saving_bytes=0,
                recent_observations_protected=0, packed_target_tokens=1,
                scope=scope, selected_archive=self.archive)
            self.assertEqual(trajectory['observation_0'], alias_line('O1') + text)
            self.assertEqual(decisions[0]['reason'], 'below_min_saving')
            self.assertLess(decisions[0]['offload_saving_bytes'], 0)
            skeleton, _ = replan_trajectory_skeleton(trajectory, scope=scope, selected_archive=self.archive)
            # An inline copy in the replan skeleton keeps the same printed handle.
            self.assertEqual(skeleton['observation_0'], alias_line('O1') + text)
            # The observation stays inline AND is searchable: keeping it in the
            # prompt is a residency decision, not an availability one (A2).
            self.assertEqual(self.archive.get(scope, 'O1')['text'], text)

    def test_replan_pointer_is_persisted_and_small_text_stays_inline(self):
        trajectory = {'tool_name_0': 'execute_workflow_query', 'tool_args_0': {'command': 'show_holders'}, 'observation_0': 'holder rows\n'+'x'*9000,
                      'tool_name_1': 'execute_workflow_query', 'tool_args_1': {'command': 'reset_context'}, 'observation_1': "Context is now '*'"}
        skeleton, _ = replan_trajectory_skeleton(trajectory, greedy_max_bytes=1000, scope=self.scope, selected_archive=self.archive)
        self.assertTrue(is_offload_label(skeleton['observation_0']))
        self.assertEqual(self.archive.get(self.scope, 'O1')['text'], trajectory['observation_0'])
        self.assertEqual(skeleton['observation_1'], trajectory['observation_1'])

    def test_replan_archive_failure_preserves_original_evidence(self):
        # A real SQLite failure: the database path names a directory.
        self.archive.db_path = self.tmp.name
        text = 'holder rows\n' + 'x'*9000
        skeleton, metadata = replan_trajectory_skeleton(
            {'tool_name_0': 'execute_workflow_query', 'tool_args_0': {'command': 'show_holders'}, 'observation_0': text},
            greedy_max_bytes=1000, scope=self.scope, selected_archive=self.archive)
        self.assertEqual(skeleton['observation_0'], text)
        self.assertEqual(metadata['persistence_failures'], ['O1'])
        self.assertTrue(metadata['over_target'])

    def test_replan_keeps_irreducible_non_command_evidence_without_aborting(self):
        text = 'available command metadata\n' + 'x'*30000
        skeleton, metadata = replan_trajectory_skeleton(
            {'tool_name_0': 'what_can_i_do', 'observation_0': text},
            scope=self.scope, selected_archive=self.archive)
        self.assertEqual(skeleton['observation_0'], text)
        self.assertTrue(metadata['over_target'])
        self.assertIsNone(self.archive.get(self.scope, 'O1'))

    def test_reasoning_is_current_step_even_after_resume_or_truncation(self):
        trajectory = {'tool_name_8': 'search_memory', 'thought_8': 'stale thought',
                      'tool_name_20': 'search_memory', 'thought_20': 'Need the account UID, not identity UID'}
        agent = SimpleNamespace(current_trajectory=trajectory)
        self.assertEqual(current_search_reasoning(agent), trajectory['thought_20'])
        trajectory['tool_name_21'] = 'execute_workflow_query'
        self.assertEqual(current_search_reasoning(agent), '')

    @unittest.skipUnless(os.environ.get('FW_TEST_OBSERVATION_SEARCH_LIVE') == '1', 'requires configured observation-search provider')
    def test_broad_question_returns_bounded_summary_not_truncated_table(self):
        text = '477 holder(s).\n' + '\n'.join(f'{i:032x} Person {i}' for i in range(477))
        self.persist('O1', text)
        import dspy
        with dspy.context(disable_history=True):
            answer = self.search('Give every identity_uid and label in this result.', 'O1')
        self.assertIn('477', answer)
        self.assertLess(len(answer), 2000)
        self.assertEqual(snapshot_events()[-1]['status'], 'answered')
        self.assertGreater(snapshot_events()[-1]['usage']['completion_tokens'], 0)

    @unittest.skipUnless(os.environ.get('FW_TEST_OBSERVATION_SEARCH_LIVE') == '1', 'requires configured observation-search provider')
    def test_full_observation_reasoning_and_scope_with_real_dspy(self):
        self.persist('O1', 'Directory data\n'+'unrelated row\n'*1500+'\nAlisha Ochoa identity_uid=c062a2718f5148a84d081358a2b082b1 account_uid=account-123\n')
        self.persist('O2', 'Alisha Ochoa account_uid=WRONG-OTHER-OBSERVATION')
        answer = self.search('What is her UID?', 'O1', reasoning='I need Alisha Ochoa account UID, not her identity UID')
        self.assertIn('account-123', answer)
        self.assertNotIn('WRONG-OTHER-OBSERVATION', answer)
        event = snapshot_events()[-1]
        self.assertEqual(event['status'], 'answered')
        self.assertGreater(event['observation_bytes'], 12000)
        self.assertIn('account UID', event['reasoning'])
