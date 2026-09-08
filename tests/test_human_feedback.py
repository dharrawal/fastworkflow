"""Human feedback uses real stores and HTTP, without model or backend calls."""
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
    return dict(target_kind='turn', span_ids=[], target_label='Turn', comment='Needs a clearer answer.', **kw)


def test_feedback_history_and_evidence_unchanged(experiment_server, tmp_path):
    server, store = experiment_server
    seed(store)
    before = store.get_turn('turn-a'), store.get_spans('turn-a')
    path = '/api/human-feedback?turn_key=turn-a'
    assert _request(server, path)[1]['feedback'] == []
    assert _request(server, path, 'POST', payload(), token=None)[0] == 401
    assert _request(server, path, 'POST', payload())[0] == 201
    component = dict(target_kind='phase', span_ids=['span-turn-a'], target_label='Planning', comment='Missing a prerequisite.\nUse this order instead.')
    assert _request(server, path, 'POST', component)[0] == 201
    component['comment'] = 'Follow-up: check the prerequisite first.'
    assert _request(server, path, 'POST', component)[0] == 201
    rows = _request(server, path)[1]['feedback']
    assert [r['target_kind'] for r in rows] == ['turn', 'phase', 'phase']
    assert rows[1]['comment'].startswith('Missing a prerequisite.\n')
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


@pytest.mark.parametrize('value', ['Free-form notes\n**Markdown** is fine.', '', ['observations'], False, 12, None])
def test_freeform_experiment_analysis(experiment_server, value):
    server, store = experiment_server
    status, data = _request(server, '/api/experiment/exp-1/analysis', 'PUT', {'analysis': value})
    assert status == 200
    raw = data['experiment']['analysis_json']
    assert (json.loads(raw) if raw is not None else None) == value
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
    funcs = '\n'.join(re.search(r'function ' + name + r'\(.*?\n\}', script, re.S).group(0)
                      for name in ['analysisText', 'feedbackAnchor'])
    checks = '''
const assert = require('assert');
assert.equal(analysisText(null), ''); assert.equal(analysisText({}), '');
assert.equal(analysisText('free text'), 'free text');
const a = {kind:'span', span:{span_id:'a'}, children:[]};
const b = {kind:'span', span:{span_id:'b'}, children:[]};
assert.deepEqual(feedbackAnchor({kind:'turn',children:[a]}), []);
assert.deepEqual(feedbackAnchor({kind:'phase',children:[b,a,a]}), ['a','b']);
assert.deepEqual(feedbackAnchor({kind:'step',span:{span_id:'step'},children:[a,b]}), ['step']);
'''
    subprocess.run([node, '-e', funcs + checks], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('value', [float('nan'), ('coerced',), {1: 'non-string key'}])
def test_analysis_rejects_non_json_native_values(experiment_server, value):
    _server, store = experiment_server
    with pytest.raises(ValueError):
        store.update_experiment_analysis('exp-1', value)
