"""Scoped archive and label integration, plus opt-in real DSPy provider tests."""
import hashlib
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastworkflow.observation_offloading.agent import current_search_reasoning
from fastworkflow.observation_offloading.archive import RuntimeHandleArchive, RuntimeHandleScope
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.continuation import replan_trajectory_skeleton
from fastworkflow.observation_offloading.labels import offload_label, label_alias, is_offload_label, alias_line
from fastworkflow import context_budget
from fastworkflow.observation_offloading.search import (
    DEFAULT_PAGE_BYTES,
    SEARCH_MEMORY_MAX_PAGES,
    SEARCH_MODEL_ENV,
    SEARCH_OBSERVATION,
    bounded_evidence,
    completion_was_truncated,
    is_bounded_evidence_observation,
    is_context_window_error,
    is_over_window_observation,
    search_answer_max_bytes_from_env,
    search_memory,
    search_observation_max_bytes,
    search_window_tokens,
)
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


class ContextWindowExceededError(Exception):
    """The litellm class, by name only: the detector matches the chain, not the import."""


class SearchInputBound(unittest.TestCase):
    """F12 (ido-3vp): what one search_memory call may hand the search model.

    Every test here is offline. The predictor is a stub that records the
    ``observation`` it was constructed with, so the byte bound is proved by
    measuring the input, not by trusting a provider to refuse it.
    """

    def setUp(self) -> None:
        reset_runtime_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.archive = RuntimeHandleArchive(str(Path(self.tmp.name) / 'archive.sqlite3'))
        self.scope = RuntimeHandleScope('store', 'channel', 'experiment', 'task', 1, 'turn')

    def persist(self, text, alias='O1', command='show_holders'):
        self.archive.persist(self.scope, alias=alias, offload_order=int(alias[1:]),
                             command_name=command, step_index=int(alias[1:]) - 1,
                             text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def run_search(self, *, answer='ok', alias='O1', error=None, question='Which rows mention value?'):
        """search_memory against a stub predictor that captures its input."""
        seen: dict = {}
        lm = SimpleNamespace(history=[{'usage': {'completion_tokens': 7}, 'cost': 0.0}],
                             model='fixture-lm')

        def predict(_signature):
            def call(question, subject, observation):
                seen['observation'] = observation
                seen['subject'] = subject
                seen['subject_bytes'] = len(subject.encode('utf-8'))
                seen['observation_bytes'] = len(observation.encode('utf-8'))
                if error is not None:
                    raise error
                return SimpleNamespace(answer=answer)
            return call

        with patch('fastworkflow.observation_offloading.search.get_lm', return_value=lm), \
                patch('fastworkflow.observation_offloading.search.dspy') as fake_dspy:
            fake_dspy.Predict.side_effect = predict
            seen['result'] = search_memory(question, alias, scope=self.scope,
                                           selected_archive=self.archive)
        seen['event'] = [e for e in snapshot_events() if e['kind'] == 'search_memory'][-1]
        return seen

    # -- the budget ----------------------------------------------------------

    def test_the_declared_page_geometry_is_the_reference_value_of_the_budget(self):
        # The two constants F12 found unused are now the budget's value at the
        # reference window, not a second contract beside it.
        self.assertEqual(SEARCH_OBSERVATION.reference_bytes,
                         DEFAULT_PAGE_BYTES * SEARCH_MEMORY_MAX_PAGES)
        self.assertEqual(SEARCH_OBSERVATION.reference_bytes, 12_288)
        # And it is a fraction of a window, so it moves with the model.
        self.assertEqual(SEARCH_OBSERVATION.bytes_for(2 * context_budget.REFERENCE_WINDOW_TOKENS),
                         2 * SEARCH_OBSERVATION.reference_bytes)
        self.assertEqual(SEARCH_OBSERVATION.floor, DEFAULT_PAGE_BYTES)

    def test_the_bound_comes_from_the_search_models_own_window(self):
        env = {'FW_MODEL_CONTEXT_TOKENS': '', 'FW_SEARCH_OBSERVATION_MAX_BYTES': '',
               SEARCH_MODEL_ENV: 'vendor/wide-search-model'}
        windows = {'vendor/wide-search-model': 4 * context_budget.REFERENCE_WINDOW_TOKENS}
        with patch.dict(os.environ, env), patch.dict('fastworkflow._env_vars', {}, clear=True), \
                patch.object(context_budget, '_model_window_tokens', windows.get):
            tokens, source = search_window_tokens()
            self.assertEqual(tokens, 4 * context_budget.REFERENCE_WINDOW_TOKENS)
            self.assertIn('vendor/wide-search-model', source)
            self.assertEqual(search_observation_max_bytes(),
                             4 * SEARCH_OBSERVATION.reference_bytes)

    def test_the_tuning_override_behaves_like_every_other_budget(self):
        base = {'FW_MODEL_CONTEXT_TOKENS': '', SEARCH_MODEL_ENV: ''}
        for raw, expected in (('8192', 8192), ('10', 12_288), ('not-a-number', 12_288), ('', 12_288)):
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {**base, 'FW_SEARCH_OBSERVATION_MAX_BYTES': raw}), \
                        patch.dict('fastworkflow._env_vars', {}, clear=True):
                    self.assertEqual(search_observation_max_bytes(), expected)

    def test_the_page_is_a_hard_byte_bound_and_a_prefix(self):
        # text_page ends just after the newline that can sit AT the budget;
        # bounded_evidence never reports more bytes than it was given.
        for width in (1, 2, 3, 11):
            with self.subTest(width=width):
                text = (('x' * (width - 1)) + '\n') * (8_192 // width + 10)
                page = bounded_evidence(text, 4_096)
                self.assertLessEqual(page['shown_bytes'], 4_096)
                self.assertEqual(page['shown_bytes'], len(page['text'].encode('utf-8')))
                self.assertTrue(text.startswith(page['text']))
                self.assertTrue(page['bounded'])
                self.assertEqual(page['total_bytes'], len(text.encode('utf-8')))

    def test_a_heading_before_one_long_line_still_spends_the_whole_budget(self):
        # One text_page call ends at the LAST newline in its window, so a
        # 17-byte heading followed by an unbroken 30 KB line would be read as
        # 17 bytes and the search answered from the heading.
        text = 'holder uid label\n' + 'x' * 30_000
        page = bounded_evidence(text, 12_288)
        self.assertEqual(page['shown_bytes'], 12_288)
        self.assertTrue(text.startswith(page['text']))

    # -- the input the model actually gets -----------------------------------

    def test_an_oversized_observation_is_cut_before_the_model_is_called(self):
        # The F12 evidence case, byte for byte: 40,000 short rows, 440,000 bytes.
        text = 'row  value\n' * 40_000
        self.assertEqual(len(text.encode('utf-8')), 440_000)
        self.persist(text)
        budget = search_observation_max_bytes()
        seen = self.run_search()
        self.assertLessEqual(seen['observation_bytes'], budget)
        # Not a token of the budget wasted either: a whole page is still read.
        self.assertGreater(seen['observation_bytes'], budget - DEFAULT_PAGE_BYTES)
        self.assertTrue(text.startswith(seen['observation']))
        event = seen['event']
        self.assertEqual(event['observation_bytes'], 440_000)
        self.assertEqual(event['observation_sent_bytes'], seen['observation_bytes'])
        self.assertTrue(event['observation_bounded'])
        self.assertEqual(event['observation_max_bytes'], budget)
        self.assertEqual(event['status'], 'answered')

    def test_the_answer_states_the_omission(self):
        text = 'row  value\n' * 40_000
        self.persist(text)
        seen = self.run_search(answer='rows 1-3 mention value')
        result, shown = seen['result'], seen['observation_bytes']
        self.assertTrue(is_bounded_evidence_observation(result))
        self.assertTrue(result.startswith('Observation O1 (tier=sqlite, bounded):\n'), result)
        self.assertIn('rows 1-3 mention value', result)
        self.assertIn(f'answered from the first {shown:,} of 440,000 UTF-8 bytes of O1', result)
        self.assertIn(f'{440_000 - shown:,} bytes were NOT read', result)
        # The absence inference a partial read would invite is denied outright.
        self.assertIn('nothing missing from the answer is thereby absent from O1', result)
        # And the action offered is one that can actually reach the other bytes.
        self.assertIn('re-run show_holders', result)
        self.assertNotIn('call search_memory on O1 again with a narrower question', result)

    def test_a_bounded_read_and_a_bounded_answer_share_the_one_budget(self):
        self.persist('row  value\n' * 40_000)
        answer = '\n'.join(f'{index:032x} Person {index}' for index in range(400))
        seen = self.run_search(answer=answer)
        result = seen['result']
        self.assertLessEqual(len(result.encode('utf-8')), search_answer_max_bytes_from_env())
        self.assertIn('BOUNDED ANSWER', result)
        self.assertIn('BOUNDED EVIDENCE', result)
        self.assertTrue(result.rstrip().endswith(']'))

    def test_an_observation_under_the_bound_is_passed_through_unchanged(self):
        text = 'holder rows\n' + 'x' * 4_000
        self.assertLess(len(text.encode('utf-8')), search_observation_max_bytes())
        self.persist(text)
        seen = self.run_search(answer='Cooper holds it.')
        # Byte-identical input: nothing is re-paged, re-joined or re-encoded.
        self.assertEqual(seen['observation'], text)
        # Byte-identical output: exactly what an unbounded search returned.
        self.assertEqual(seen['result'], 'Observation O1 (tier=sqlite):\nCooper holds it.')
        self.assertNotIn('BOUNDED', seen['result'])
        self.assertFalse(seen['event']['observation_bounded'])
        self.assertEqual(seen['event']['observation_sent_bytes'],
                         seen['event']['observation_bytes'])

    # -- the overflow that used to be a bare failure string -------------------

    def test_a_context_window_error_is_a_typed_actionable_outcome(self):
        self.persist('row  value\n' * 40_000)
        seen = self.run_search(error=ContextWindowExceededError('prompt of 90000 tokens'))
        result = seen['result']
        self.assertTrue(is_over_window_observation(result), result)
        # Not the generic failure string, which named the class and nothing else.
        self.assertNotIn('failed (ContextWindowExceededError)', result)
        self.assertIn('do not retry it unchanged', result)
        self.assertIn('Re-run show_holders', result)
        self.assertIn('FW_SEARCH_OBSERVATION_MAX_BYTES', result)
        self.assertIn(SEARCH_MODEL_ENV, result)
        self.assertIn(f'{search_observation_max_bytes():,}-byte bound', result)
        event = seen['event']
        self.assertEqual(event['status'], 'over_window')
        self.assertEqual(event['reason'], 'context_window_exceeded')
        self.assertEqual(event['error'], 'ContextWindowExceededError')

    def test_the_overflow_is_recognised_by_class_chain_or_by_words(self):
        class Subclass(ContextWindowExceededError):
            pass
        self.assertTrue(is_context_window_error(Subclass('x')))
        self.assertTrue(is_context_window_error(
            RuntimeError("This model's maximum context length is 8192 tokens")))
        self.assertTrue(is_context_window_error(ValueError('code: context_length_exceeded')))
        self.assertFalse(is_context_window_error(RuntimeError('connection reset by peer')))

    def test_an_unrelated_provider_failure_is_still_the_generic_failure(self):
        self.persist('holder rows')
        seen = self.run_search(error=RuntimeError('connection reset by peer'))
        self.assertFalse(is_over_window_observation(seen['result']))
        self.assertIn('search of O1 failed (RuntimeError)', seen['result'])
        # The provider message itself is never printed back.
        self.assertNotIn('connection reset', seen['result'])
        self.assertEqual(seen['event']['status'], 'error')
