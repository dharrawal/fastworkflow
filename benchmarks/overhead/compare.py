#!/usr/bin/env python
"""Side-by-side table for run_overhead.py JSON reports.

    python compare.py results/fork.json results/change4.json results/port.json

Prints one row per report; the first report is the baseline for the delta
columns. Pure stdlib; no fastworkflow import.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional


def _get(d: dict, *path, default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _num(v) -> Optional[float]:
    return v if isinstance(v, (int, float)) else None


def row(report: dict) -> dict[str, Any]:
    m = report.get("metrics", {})
    tables = _get(report, "observability_rows", "tables", default={}) or {}
    return {
        "label": report.get("commit_label", "?"),
        "obs": report.get("observability", "?"),
        "turns": _get(m, "turn_wall_ms", "n"),
        "cmds": _num(_get(m, "commands_per_turn", "median")),
        "turn_p50_ms": _num(_get(m, "turn_wall_ms", "median")),
        "turn_p90_ms": _num(_get(m, "turn_wall_ms", "p90")),
        "cmd_p50_ms": _num(_get(m, "command_gap_ms", "median")),
        "cmd_p90_ms": _num(_get(m, "command_gap_ms", "p90")),
        "cpu_per_turn_s": _num(m.get("process_cpu_per_turn_s")),
        "peak_rss_mb": (_num(m.get("peak_rss_bytes")) or 0) / (1024 * 1024) if _num(m.get("peak_rss_bytes")) else None,
        "db_kb": (_num(_get(report, "observability_db_bytes", "total")) or 0) / 1024
        if _num(_get(report, "observability_db_bytes", "total")) else None,
        "spans": tables.get("spans"),
        "turn_rows": tables.get("turns"),
        "tripwire": report.get("tripwire_trips"),
        "error": bool(report.get("error")),
    }


def fmt(v, nd=2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def delta(base: Optional[float], cur: Optional[float]) -> str:
    if base is None or cur is None or base == 0:
        return ""
    return f" ({(cur - base) / base * 100:+.1f}%)"


def render(reports: list[dict]) -> str:
    rows = [row(r) for r in reports]
    base = rows[0] if rows else None
    cols = [
        ("label", "label"), ("obs", "obs"), ("turns", "turns"), ("cmds", "cmds/turn"),
        ("turn_p50_ms", "turn p50 ms"), ("turn_p90_ms", "turn p90 ms"),
        ("cmd_p50_ms", "cmd p50 ms"), ("cmd_p90_ms", "cmd p90 ms"),
        ("cpu_per_turn_s", "cpu/turn s"), ("peak_rss_mb", "peak RSS MB"),
        ("db_kb", "db KB"), ("spans", "spans"), ("turn_rows", "turn rows"),
        ("tripwire", "tripwire"), ("error", "error"),
    ]
    delta_cols = {"turn_p50_ms", "turn_p90_ms", "cmd_p50_ms", "cmd_p90_ms", "cpu_per_turn_s", "peak_rss_mb", "db_kb"}
    table = [[title for _, title in cols]]
    for r in rows:
        line = []
        for key, _ in cols:
            cell = fmt(r[key])
            if key in delta_cols and base is not None and r is not base:
                cell += delta(base[key], r[key])
            line.append(cell)
        table.append(line)
    widths = [max(len(line[i]) for line in table) for i in range(len(cols))]
    out = []
    for n, line in enumerate(table):
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)))
        if n == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    reports = []
    for path in argv:
        with open(path, "r", encoding="utf-8") as fh:
            reports.append(json.load(fh))
    print(render(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
