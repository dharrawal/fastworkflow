"""Feedback uses real stores and HTTP, without model or backend calls."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fastworkflow import observability_store as obs
from tests.test_chatbot_benchmarks import (
    _request, experiment_server, live_server, workflow_dir, workspace_server,
)
from tests.test_observability_workspace import _turn_row


def seed(store):
    with store._connect() as conn:
        for key in ('turn-a', 'turn-b'):
            row = _turn_row(key, 'exp-1', 'task-1', 1)
            assert store.upsert_turn_row(conn, row, [], store._store_redactor())
            conn.execute(
                'INSERT INTO spans(span_id,trace_id,name,kind,start_ns,status,attributes) '
                'VALUES(?,?,?,?,?,?,?)',
                ('span-' + key, key, 'fw.planner.plan', 'internal', 1, 'ok', '{}'),
            )


def payload(**kw):
    return dict(
        target_kind='turn',
        span_ids=[],
        target_label='Turn',
        comment='Needs a clearer answer.',
        provenance='human',
        **kw,
    )


def test_existing_labelled_comments_parse_into_tabs():
    labelled = (
        "What worked: The current plan correctly reflects almost the entire task\n"
        "What went wrong: Step(s) needed to provide a complete picture of Denise's access are missing\n"
        "What should change: Between steps 3 and 4, there need to be a whole series of steps"
    )
    parsed = obs.parse_human_feedback_comment(labelled)
    assert parsed["worked"].startswith("The current plan correctly")
    assert parsed["went_wrong"].startswith("Step(s) needed")
    assert parsed["should_change"].startswith("Between steps 3 and 4")
    synonym = (
        "What worked: The system understood the task\n"
        "What did not work: The harness did not answer the question\n"
        "What should change: The harness should have answered"
    )
    parsed = obs.parse_human_feedback_comment(synonym)
    assert parsed["went_wrong"] == "The harness did not answer the question"
    unlabeled = "Looks good! The agent did everything it was asked correctly"
    assert obs.parse_human_feedback_comment(unlabeled) == {
        "went_wrong": "", "worked": "", "should_change": "",
    }
    composed = obs.compose_human_feedback_comment(
        went_wrong="Missing a person",
        worked="Honest about the gap",
        should_change="Retain all three people",
    )
    assert obs.parse_human_feedback_comment(composed) == {
        "went_wrong": "Missing a person",
        "worked": "Honest about the gap",
        "should_change": "Retain all three people",
    }


def test_feedback_history_and_evidence_unchanged(experiment_server, tmp_path):
    server, store = experiment_server
    seed(store)
    before = store.get_turn('turn-a'), store.get_spans('turn-a')
    path = '/api/human-feedback?turn_key=turn-a'
    assert _request(server, path)[1]['feedback'] == []
    assert _request(server, path, 'POST', payload(), token=None)[0] == 401
    assert _request(server, path, 'POST', payload())[0] == 201
    structured = dict(
        target_kind='turn',
        span_ids=[],
        target_label='Turn',
        provenance='human',
        went_wrong='The request names three people, but this plan covers only two.',
        worked='The plan is honest about the unfinished work.',
        should_change='Retain all three and track completion separately.',
    )
    assert _request(server, path, 'POST', structured)[0] == 201
    component = dict(
        target_kind='phase',
        span_ids=['span-turn-a'],
        target_label='Planning',
        comment='Missing a prerequisite.\nUse this order instead.',
        provenance='coding_agent',
    )
    assert _request(server, path, 'POST', component)[0] == 201
    component['comment'] = 'Follow-up: check the prerequisite first.'
    component['provenance'] = 'distillation_agent'
    assert _request(server, path, 'POST', component)[0] == 201
    rows = _request(server, path)[1]['feedback']
    assert [r['target_kind'] for r in rows] == ['turn', 'turn', 'phase', 'phase']
    assert [r['provenance'] for r in rows] == [
        'human', 'human', 'coding_agent', 'distillation_agent',
    ]
    assert rows[0]['comment'] == 'Needs a clearer answer.'
    assert rows[0]['went_wrong'] == ''
    assert rows[1]['went_wrong'].startswith('The request names three people')
    assert rows[1]['worked'].startswith('The plan is honest')
    assert rows[1]['should_change'].startswith('Retain all three')
    assert rows[2]['comment'].startswith('Missing a prerequisite.\n')
    with store._connect() as conn:
        stored = [r[0] for r in conn.execute(
            'SELECT comment FROM human_feedback WHERE turn_key=? ORDER BY feedback_id',
            ('turn-a',),
        )]
    assert stored[0] == 'Needs a clearer answer.'
    assert 'What went wrong:' in stored[1]
    assert 'What worked:' in stored[1]
    assert 'What should change:' in stored[1]
    assert all(r['created_at'] for r in rows)
    assert before == (store.get_turn('turn-a'), store.get_spans('turn-a'))
    assert store.get_feedback('turn-a') is None
    assert store.list_human_feedback('turn-b') == []
    # A database snapshot carries the comments without another file.
    archive = tmp_path / 'copy.sqlite3'
    store.archive_to(str(archive))
    assert obs.ReadOnlyObservabilityStore(str(archive)).list_human_feedback('turn-a') == rows
    with store._connect() as conn:
        conn.execute('DELETE FROM turns WHERE turn_key=?', ('turn-a',))
    assert store.list_human_feedback('turn-a') == []


@pytest.mark.parametrize('changes', [
    {'target_kind': 'span', 'span_ids': ['span-turn-b']},
    {'target_kind': 'span', 'span_ids': []},
    {'span_ids': ['span-turn-a']},
    {'target_kind': 'invented'},
    {'comment': ''},
    {'comment': 123},
    {'span_ids': 'span-turn-a'},
    {'provenance': 'agent'},
])
def test_bad_feedback_anchors_refused(experiment_server, changes):
    server, store = experiment_server
    seed(store)
    body = payload(); body.update(changes)
    assert _request(server, '/api/human-feedback?turn_key=turn-a', 'POST', body)[0] == 400
    assert store.list_human_feedback('turn-a') == []


def test_unknown_turn_and_malformed_request(experiment_server):
    server, store = experiment_server
    assert _request(server, '/api/human-feedback?turn_key=missing', 'POST', payload())[0] == 404
    assert _request(server, '/api/human-feedback', 'POST', payload())[0] == 400
    assert _request(server, '/api/human-feedback?turn_key=x', 'POST', ['bad'])[0] == 400
    seed(store)
    missing_provenance = payload()
    missing_provenance.pop('provenance')
    assert _request(
        server,
        '/api/human-feedback?turn_key=turn-a',
        'POST',
        missing_provenance,
    )[0] == 400


def test_workspace_feedback_cannot_mutate_archive(workspace_server):
    server, _workflow, _before = workspace_server
    stores = server.workspace.stores()
    sid = stores[0]['store_id']
    descriptor = server.workspace.registry.descriptor(sid)
    before = hashlib.sha256(descriptor.path.read_bytes()).hexdigest()
    path = '/api/human-feedback?turn_key=turn&store_id=' + sid
    status, data = _request(server, path)
    assert status == 200 and data == {"feedback": [], "read_only": True}
    assert _request(server, path, 'POST', payload())[0] == 403
    assert hashlib.sha256(descriptor.path.read_bytes()).hexdigest() == before


def test_experiment_analysis_route_is_gone(experiment_server):
    server, store = experiment_server
    assert _request(server, '/api/experiment/exp-1/analysis', 'PUT', {'analysis': 'x'})[0] == 405
    assert store.get_experiment('exp-1')['notes'] == 'original notes'


@pytest.mark.parametrize('value', ['Free-form observations\nNo JSON required.', '', ['one'], None])
def test_freeform_benchmark_analysis(live_server, value):
    path = '/api/benchmarks/smoke/analysis'
    assert _request(live_server, path, 'PUT', {'analysis': value})[0] == 200
    assert _request(live_server, path)[1]['analysis'] == value


def test_ui_analysis_and_stable_component_anchors():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to execute UI helper tests')
    page = Path(__file__).parents[1].joinpath('fastworkflow/run_chatbot/static/index.html').read_text()
    script = re.search(r'<script[^>]*>(.*?)</script>', page, re.S).group(1)
    subprocess.run([node, '--check'], input=script, text=True, check=True, capture_output=True)
    funcs = '\n'.join(
        re.search(r'function ' + name + r'\(.*?\n\}', script, re.S).group(0)
        for name in ['analysisText', 'feedbackAnchor', 'feedbackProvenanceLabel',
                     'parseHumanFeedbackComment', 'composeHumanFeedbackComment']
    )
    tabs = re.search(r'var FEEDBACK_TABS = \[.*?\n\];', script, re.S).group(0)
    checks = '''
const assert = require('assert');
assert.equal(analysisText(null), ''); assert.equal(analysisText({}), '');
assert.equal(analysisText('free text'), 'free text');
assert.equal(feedbackProvenanceLabel('human'), 'Human');
assert.equal(feedbackProvenanceLabel('coding_agent'), 'Coding Agent');
assert.equal(feedbackProvenanceLabel('distillation_agent'), 'Distillation Agent');
const a = {kind:'span', span:{span_id:'a'}, children:[]};
const b = {kind:'span', span:{span_id:'b'}, children:[]};
assert.deepEqual(feedbackAnchor({kind:'turn',children:[a]}), []);
assert.deepEqual(feedbackAnchor({kind:'phase',children:[b,a,a]}), ['a','b']);
assert.deepEqual(feedbackAnchor({kind:'step',span:{span_id:'step'},children:[a,b]}), ['step']);
assert.deepEqual(parseHumanFeedbackComment('Looks good!'), {went_wrong:'', worked:'', should_change:''});
const parsed = parseHumanFeedbackComment('What worked: Kept the honest stop\\nWhat did not work: Stopped too early\\nWhat should change: Resume with remaining work');
assert.equal(parsed.worked, 'Kept the honest stop');
assert.equal(parsed.went_wrong, 'Stopped too early');
assert.equal(parsed.should_change, 'Resume with remaining work');
assert.equal(composeHumanFeedbackComment({went_wrong:'A', worked:'B', should_change:'C'}),
  'What went wrong: A\\n\\nWhat worked: B\\n\\nWhat should change: C');
'''
    subprocess.run([node, '-e', tabs + '\n' + funcs + checks], check=True, capture_output=True, text=True)
