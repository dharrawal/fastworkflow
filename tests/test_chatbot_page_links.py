"""The address bar names the page on screen, and every name it shows opens that page.

Real SQLite stores behind a real chatbot server; the DOM checks run in jsdom
and skip unless TEST_JSDOM_ROOT is set (run them through tests.browser_validation).
"""
import json

from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request, _turn_row, experiment_server, workflow_dir
from tests.test_chatbot_hierarchy import _run_dom, hierarchy_server


def _seed_turn(store, key, message, spans, *, channel='chat', cid=9):
    row = _turn_row(key, None, None, None)
    row.update(channel_id=channel, conversation_id=cid, ordinal=1,
               user_message=message, started_at='2026-09-08T01:00:00+00:00')
    with store._connect() as conn:
        assert store.upsert_turn_row(conn, row, [], store._store_redactor())
        for span_id, parent, name, start, end, attrs in spans:
            conn.execute(
                'INSERT INTO spans(span_id,trace_id,parent_span_id,name,kind,start_ns,end_ns,status,attributes)'
                ' VALUES(?,?,?,?,?,?,?,?,?)',
                (span_id, key, parent, name, 'internal', start, end, 'ok', json.dumps(attrs)))
        conn.commit()


def _recorded_spans(prefix):
    """A turn of span contract 63: its run stage, one run part, the part's plan and two steps."""
    return [
        (prefix + '-run', None, 'fw.run', 1, 9500, {}),
        (prefix + '-part', prefix + '-run', 'fw.run.part', 2, 9400, {'part': 1, 'outcome': 'ran'}),
        (prefix + '-plan', prefix + '-part', 'fw.plan', 3, 1000, {'trigger': 'initial'}),
        (prefix + '-step-1', prefix + '-part', 'fw.step', 2100, 5000,
         {'tool_name': 'first_tool', 'run_by': 'agent'}),
        (prefix + '-step-2', prefix + '-part', 'fw.step', 5100, 8900,
         {'tool_name': 'second_tool', 'run_by': 'agent'}),
    ]


# A second turn, in a conversation of its own (the rail's conversation page).
OTHER_SPANS = [
    ('other-run', None, 'fw.run', 100, 400, {}),
]


def test_page_names_pages_without_adding_history_entries():
    page = run_chatbot_server.load_index_html()
    assert b'id="copyLinkBtn"' in page
    assert b'history.replaceState' in page
    # The record navigator is how a reader moves back; a pushed entry would
    # make Back walk the pages instead of leaving the app.
    assert b'pushState' not in page
    assert b'addEventListener("popstate"' not in page


def test_page_links_round_trip_dom(hierarchy_server):
    server, spec, eid, default, _store = hierarchy_server
    _seed_turn(default, 'crumb-turn', 'climb the crumbs', _recorded_spans('crumb'))
    _seed_turn(default, 'other-turn', 'another conversation', OTHER_SPANS,
               channel='cli:local', cid=10)
    status, data = _request(server, f'/api/experiment/{eid}/tasks')
    assert status == 200
    task_id = data['tasks'][0]['task_id']
    _run_dom('chatbot_page_link_dom.cjs', server, eid, spec['benchmark_id'], task_id,
             timeout=120)
