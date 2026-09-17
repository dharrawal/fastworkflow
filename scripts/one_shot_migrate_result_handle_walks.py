#!/usr/bin/env python
"""ONE-SHOT (ido-1r0): give an existing handle store the result_handle_walks table.

Delete this script once it has been run over the stores that need it. It carries
no ongoing compatibility: nothing in ``fastworkflow`` reads it, and the framework
does not branch on whether a store has been through it.

What it does
------------
``result_handle_walks`` records where a traversal ended and the ``countOnly``
that judged it, so a rebuilt walk does not ask the source to prove its end over
and over. ``ResultHandleStore.__init__`` creates the table with
``CREATE TABLE IF NOT EXISTS``, so a store opened by the new code gets it anyway;
running this first is how a file that is only ever read for scoring gets the new
shape deliberately instead of on the next open.

What it deliberately does NOT do
--------------------------------
It writes no verdicts. A stored empty resolver page proves that the pages ran
out; only the ``countOnly`` reply proves the walk saw everything, and that reply
was never written down before this change. Backfilling ``complete`` from the
pages alone would manufacture a coverage claim nothing measured, so each walk is
judged once, for real, the next time it is read. The report below counts the
walks in that position, and the empty pages past a terminal that the old rebuild
left behind - those are evidence of the defect and are never deleted.

It never writes to a page, a declaration or a cursor row.

Usage
-----
    python scripts/one_shot_migrate_result_handle_walks.py PATH [PATH ...]
    python scripts/one_shot_migrate_result_handle_walks.py --root DIR
    ... --apply        # without it the run is a read-only report

Always take a copy of an irreplaceable store first: an evaluation run's archive
is experiment evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

STORE_SUFFIX = "offload-handles.sqlite3"

CREATE_WALKS = """
CREATE TABLE IF NOT EXISTS result_handle_walks (
    scope_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    query_scope TEXT NOT NULL,
    terminal_offset INTEGER NOT NULL,
    complete INTEGER NOT NULL,
    count_only INTEGER,
    distinct_uids INTEGER NOT NULL,
    stop_reason TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (scope_id, alias, query_scope)
)
"""


def store_paths(args) -> list[str]:
    paths = [os.path.abspath(path) for path in args.path]
    for root in args.root or []:
        for base, _, names in os.walk(root):
            paths.extend(os.path.join(base, name) for name in names
                         if name.endswith(STORE_SUFFIX))
    seen: list[str] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return sorted(seen)


def survey(conn: sqlite3.Connection) -> dict:
    """What this store holds, and what the old rebuild left in it."""
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "result_handle_pages" not in tables:
        return {"handle_store": False}
    walks: dict[tuple, dict] = {}
    rows = conn.execute(
        "SELECT scope_id, alias, query_scope, start_offset, row_count, source "
        "FROM result_handle_pages ORDER BY scope_id, alias, query_scope, "
        "start_offset"
    ).fetchall()
    for scope_id, alias, query_scope, start_offset, row_count, source in rows:
        walk = walks.setdefault(
            (scope_id, alias, query_scope),
            {"pages": 0, "terminal_offset": None, "empty_pages_past_terminal": 0},
        )
        walk["pages"] += 1
        if int(row_count) == 0 and str(source) != "producer":
            if walk["terminal_offset"] is None:
                walk["terminal_offset"] = int(start_offset)
            else:
                walk["empty_pages_past_terminal"] += 1
    ended = [walk for walk in walks.values() if walk["terminal_offset"] is not None]
    return {
        "handle_store": True,
        "has_walks_table": "result_handle_walks" in tables,
        "walk_count": len(walks),
        "walks_with_a_stored_end": len(ended),
        "empty_pages_past_a_terminal": sum(
            walk["empty_pages_past_terminal"] for walk in ended),
        "verdicts_owed_on_next_read": len(ended),
    }


def migrate(path: str, *, apply: bool) -> dict:
    mode = "" if apply else "?mode=ro"
    conn = sqlite3.connect("file:%s%s" % (path, mode), uri=True)
    try:
        report = survey(conn)
        report["path"] = path
        if not report.get("handle_store"):
            report["action"] = "skipped: not a result handle store"
            return report
        if report["has_walks_table"]:
            report["action"] = "already migrated"
            return report
        if not apply:
            report["action"] = "would create result_handle_walks"
            return report
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(CREATE_WALKS)
        conn.commit()
        report["action"] = "created result_handle_walks"
        report["has_walks_table"] = True
        return report
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="*", help="a *.offload-handles.sqlite3 file")
    parser.add_argument("--root", action="append",
                        help="a directory to scan for store files")
    parser.add_argument("--apply", action="store_true",
                        help="write; without it the run only reports")
    args = parser.parse_args()
    paths = store_paths(args)
    if not paths:
        parser.error("no store paths given")
    reports = [migrate(path, apply=args.apply) for path in paths]
    for report in reports:
        print(json.dumps(report, sort_keys=True))
    changed = sum(1 for report in reports
                  if report["action"].startswith(("created", "would create")))
    print("# %d store(s), %d %s"
          % (len(reports), changed,
             "migrated" if args.apply else "to migrate (dry run)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
