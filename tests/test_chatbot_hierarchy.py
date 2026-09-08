"""Real SQLite/HTTP navigation contracts and optional DOM click integration."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from fastworkflow import benchmark_setup as setup
from fastworkflow import observability_store as obs
from fastworkflow.experiment import ExperimentController
from fastworkflow.run_chatbot.navigation import build_navigation
from tests.test_chatbot_benchmarks import _request, experiment_server, workflow_dir, workspace_server
from tests.test_observability_workspace import _turn_row


def add_turn(store, key, eid=None, day='2026-09-08', channel='chat', cid=1):
    row = _turn_row(key, eid, 'task' if eid else None, 1 if eid else None)
    row.update(channel_id=channel, conversation_id=cid, started_at=day+'T12:00:00+00:00', user_message=key)
    with store._connect() as conn:
        assert store.upsert_turn_row(conn, row, [], store._store_redactor())


def walk(node):
    yield node
    for child in node['children']:
        yield from walk(child)


@pytest.fixture
def hierarchy_server(experiment_server, tmp_path):
    server, default = experiment_server
    for key, day in [('plain today', '2026-09-08'), ('plain yesterday', '2026-09-07')]:
        add_turn(default, key, day=day)
    add_turn(default, 'outside benchmark', eid='exp-1', channel='unassigned')
    spec = setup.save_benchmark(server.workflow_path, {'title': 'Tuning benchmark', 'description': 'Review this benchmark', 'tasks': [{}]})
    registration = setup.create_experiment(server.workflow_path, spec['benchmark_id'], 'v1')
    store = obs.ObservabilityStore(str(tmp_path / 'registered.sqlite3'))
    controller = ExperimentController(store.db_path, store.store_identity(), external=False, workflow_folderpath=server.workflow_path)
    eid = registration['experiment_id']
    controller.create_experiment(eid, 'Recorded experiment', declared_tasks=1, declared_attempts=1,
        declarations=[(registration['task_ids'][0], 1, 'registered')])
    add_turn(store, 'experiment-turn', eid=eid, channel='registered')
    with store._connect() as conn:
        conn.execute("INSERT INTO spans(span_id,trace_id,name,kind,start_ns,end_ns,status,attributes) VALUES(?,?,?,?,?,?,?,?)",
                     ('planning', 'experiment-turn', 'fw.planner.plan', 'internal', 1, 1000, 'ok', '{}'))
    setup.create_experiment(server.workflow_path, spec['benchmark_id'], 'v1')
    yield server, spec, eid, default, store


def test_hierarchy_separates_benchmarks_experiments_and_dates(hierarchy_server):
    server, spec, eid, _default, _store = hierarchy_server
    assert _request(server, '/api/navigation', token=None)[0] == 401
    status, data = _request(server, '/api/navigation')
    assert status == 200
    root = data['root']
    benchmark = next(n for n in root['children'] if n.get('benchmark_id') == spec['benchmark_id'])
    assert benchmark['label'] == 'Tuning benchmark'
    assert len(benchmark['children']) == 2
    recorded = next(n for n in benchmark['children'] if n['experiment_id'] == eid)
    assert recorded['recorded']
    turn = next(n for n in walk(recorded) if n['kind'] == 'turn')
    assert turn['turn_key'] == 'experiment-turn' and turn['source'] == {'benchmark_experiment': eid}
    adhoc = next(n for n in root['children'] if n['kind'] == 'adhoc')
    assert [n['label'] for n in adhoc['children']] == ['2026-09-08', '2026-09-07']
    assert {n['turn_key'] for n in walk(adhoc) if n['kind'] == 'turn'} == {'plain today', 'plain yesterday'}
    assert any(n['turn_key'] == 'outside benchmark' for n in walk(root) if n['kind'] == 'turn')


def test_more_than_one_page_and_colliding_conversation_ids(tmp_path):
    a = obs.ObservabilityStore(str(tmp_path / 'a.sqlite3'))
    b = obs.ObservabilityStore(str(tmp_path / 'b.sqlite3'))
    for i in range(503):
        add_turn(a, f'a-{i}')
    add_turn(b, 'b')
    root = build_navigation([], [], [{'store': a, 'source': {'store_id': 'a'}}, {'store': b, 'source': {'store_id': 'b'}}])
    assert len([n for n in walk(root) if n['kind'] == 'turn']) == 504
    assert len([n for n in walk(root) if n['kind'] == 'conversation']) == 2


def test_navigation_workspace_is_scoped(workspace_server):
    server, _workflow, _before = workspace_server
    status, data = _request(server, '/api/navigation')
    assert status == 200
    turns = [n for n in walk(data['root']) if n['kind'] == 'turn']
    assert turns and all(n['source'].get('store_id') for n in turns)


def test_incompatible_default_does_not_hide_registered_experiment(hierarchy_server, tmp_path):
    import sqlite3
    server, _spec, eid, _default, _store = hierarchy_server
    old = tmp_path / 'old.sqlite3'
    with sqlite3.connect(old) as conn:
        conn.execute('pragma user_version=1')
    server.db_path = str(old)
    status, data = _request(server, '/api/navigation')
    assert status == 200 and data['root']['info']['warnings']
    assert any(n.get('experiment_id') == eid and n['recorded'] for n in walk(data['root']) if n['kind'] == 'experiment')


def test_hierarchy_dom_clicks(hierarchy_server):
    dependency = os.environ.get('TEST_JSDOM_ROOT')
    if not dependency:
        pytest.skip('Set TEST_JSDOM_ROOT to run DOM integration with jsdom')
    server, _spec, eid, _default, _store = hierarchy_server
    script = Path(__file__).with_name('chatbot_hierarchy_dom.cjs')
    result = subprocess.run(['node', str(script), dependency,
        f'http://127.0.0.1:{server.port}/?token={server.token}', eid],
        capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
