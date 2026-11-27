#!/usr/bin/env python3
"""Measurement gate for the parameter pipeline (bd fix-qoyz): values right, missing, made up, questions, model calls.

Every behaviour change to parameter extraction is measured with this tool before and after (docs/parameter_pipeline_design.md,
sections 5 and 7). It runs the product's extraction on hand-written natural-language requests and counts, per field:

* right    the value equals the gold value (after trivial normalisation: case, quotes, whitespace, Unicode dashes, a leading
           ``#``), or the gold says the value is not given and none was produced (the default);
* missing  the request gives the value (gold) and the extraction did not produce it (not given, or refused by a check);
* made up  a value produced where the gold says the request gives none, or a value different from the gold.

and per row whether a question would be asked (any required value missing or refused, or a refused call) and how many
extractor (language model) calls were made.

Subcommands
-----------
    score --split dev|confirm [--variant baseline|retry_missing|plan_no_retry] [--limit N] [--show-errors] [--workers N]
        Runs ``parameter_extraction.extract_arguments(app_workflow, command_name, text, diagnostics)`` for every row of
        ``data/parameter_pipeline_requests.json`` of that split. The entry point matches the product: it is the extractor of
        both kinds of command (``ParameterExtraction._llm_extract`` + ``argument_checks`` for an execution command, with its
        one retry on a guard refusal), then ``InputForParamExtraction.validate_parameters`` (types, patterns, the Input's own
        validators, and for a planning command ``check_arguments``), and for a planning command one more extraction with the
        refusal. It does NOT run the CME's xml / ``name=value`` parsing or the stored-value merge (no CME state), and it does not
        start the Jev checks unless ``JEV_API_KEY`` is set in the env files.
        ``retry_missing`` is EXPERIMENT code in this file only: after the product's answer, when a required value is still
        missing and the product made no second pass of its own, extract once more with a note saying which value was not found
        (copy it if the request writes it, leave it not given otherwise), apply the same guards to that second answer against the
        ORIGINAL text, and let it stand (as the plan side does). The report also prints the delta against the same row's first
        pass (paired, same run).
    (variant ``plan_no_retry``: planning rows are scored on the product's FIRST pass, i.e. without the second extraction
        ``extract_arguments`` makes for a planning command. Nothing is patched out: the script wraps
        ``InputForParamExtraction.validate_parameters`` and keeps the params and diagnostics of the first validation of the
        same call, which is exactly the pre-retry state, so first pass and final answer come from ONE call (paired).
        Execution rows run as baseline. Planning rows of every variant also get the paired report ``plan second pass``.)
    spans --db <observability sqlite> --out <jsonl>
        Read-only. One JSON line per ``fw.parameters`` span. ``--out`` must not be inside the repository.
    spans-summary --in <jsonl>
        Counts by extraction method, share with a missing required field, share with a guard refusal, share retried.

Nothing is written into a workflow folder: the two example workflows are copied to a temp directory (the trained models are
left out; nothing is trained and no routing runs), state goes to a temp ``FASTWORKFLOW_STATE_ROOT``, the language-model cache is
off (``FW_LM_CACHE=0``) so each run is a real call. Passwords are read by ``dotenv_values`` and handed to ``fastworkflow.init``;
they are never printed. Results are single runs of a stochastic model: run a variant twice to see the noise.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "parameter_pipeline_requests.json"
EXAMPLES = REPO / "fastworkflow" / "examples"
MISSING, NOT_GIVEN = "missing", None

RETRY_NOTE = (
    "(A first extraction from this command found no value for the required {names}. If the command above writes "
    "{names}, copy it exactly as the command writes it; if it does not, answer NOT_FOUND for it.)")


# ------------------------------------------------------------------ normalisation and classification

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def norm(value, field: str = "") -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(sorted(norm(v) for v in value))
    if isinstance(value, bool):
        return str(value).lower()
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASHES).strip()
    text = text.strip("\"'‘’“”` ").lstrip("#")
    text = re.sub(r"\s+", " ", text).casefold()
    return re.sub(r"\s+", "", text) if field == "expression" else text


def classify(row: dict, extraction, params_class) -> dict:
    """Per field: ``right`` | ``missing`` | ``made_up_wrong`` | ``made_up_invented`` | ``ok_empty``, plus ``refused``."""
    values = dict(extraction.values)
    open_names = {*extraction.missing, *(n for n, _ in extraction.invalid)}
    refused = {n for n, _ in extraction.invalid}
    out = {}
    for name, gold in row["gold"].items():
        info = params_class.model_fields.get(name)
        default = info.get_default(call_default_factory=True) if info is not None and not info.is_required() else None
        got = None if name in open_names else values.get(name)
        absent = got is None or got in ("", [], "NOT_FOUND") or (isinstance(got, list) and all(x == "NOT_FOUND" for x in got))
        if gold == MISSING:
            kind = "ok_empty" if absent else "made_up_invented"
        elif gold is None:
            same_as_default = (not absent) and norm(got, name) == norm(default, name) and default is not None
            kind = "ok_empty" if absent or same_as_default else "made_up_invented"
        elif absent:
            kind = "missing"
        else:
            if name in row.get("loose", []):
                a, b = norm(gold, name), norm(got, name)
                same = a in b or b in a
            else:
                same = norm(gold, name) == norm(got, name)
            kind = "right" if same else "made_up_wrong"
        out[name] = {"kind": kind, "gold": gold, "got": got, "refused": name in refused}
    return out


# ------------------------------------------------------------------ the product under test

class Counter:
    """Counts extractor calls per thread (the wrapper around ``InputForParamExtraction.extract_parameters``)."""

    def __init__(self):
        self.local = threading.local()

    def reset(self):
        self.local.n = 0

    def bump(self):
        self.local.n = getattr(self.local, "n", 0) + 1

    @property
    def n(self) -> int:
        return getattr(self.local, "n", 0)


def _setup(scratch: Path):
    from dotenv import dotenv_values

    import fastworkflow

    env = {**dotenv_values(EXAMPLES / "fastworkflow.env"), **dotenv_values(EXAMPLES / "fastworkflow.passwords.env")}
    env["SPEEDDICT_FOLDERNAME"] = str(scratch / "speeddict")
    env["FASTWORKFLOW_STATE_ROOT"] = str(scratch / "state")
    env["FW_LM_CACHE"] = "0"
    fastworkflow.init(env_vars=env)
    apps = {}
    for name in ("library", "retail_workflow"):
        target = scratch / name
        shutil.copytree(EXAMPLES / name, target, symlinks=True, ignore=shutil.ignore_patterns(
            "___workflow_contexts", "___convo_info", "__pycache__", "versions"))
        apps[name] = fastworkflow.Workflow.create(workflow_folderpath=str(target), workflow_id_str=f"qoyz-{name}")
    return apps, env.get("LLM_PARAM_EXTRACTION", "?")


_SNAP = threading.local()


def _first_pass(app, row, counter):
    """The product's ``extract_arguments``; for a planning command also the state after its FIRST validation, kept by a
    wrapper of ``validate_parameters`` (``_SNAP.first``: params copy, diagnostics copy, validity, calls so far)."""
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import extract_arguments

    diagnostics: dict = {}
    counter.reset()
    _SNAP.first, _SNAP.counter = None, counter
    result = extract_arguments(app, row["command"], row["text"], diagnostics)
    return result, diagnostics


def _install_snapshot():
    import copy
    from fastworkflow.parameters.signatures import InputForParamExtraction

    original = InputForParamExtraction.validate_parameters

    def wrapped(self, app, command, params, diagnostics, *a, **k):
        out = original(self, app, command, params, diagnostics, *a, **k)
        if getattr(_SNAP, "first", "unset") is None:
            _SNAP.first = {"params": params.model_copy(deep=True), "diag": copy.deepcopy(dict(diagnostics)),
                           "valid": out[0], "message": out[1], "calls": _SNAP.counter.n}
        return out

    InputForParamExtraction.validate_parameters = wrapped
    return original


def _first_as_result(snap):
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ExtractedArguments, extraction_of

    params, diag = snap["params"], snap["diag"]
    if snap["valid"]:
        return ExtractedArguments(params=params, extraction=extraction_of(params, {}))
    missing = list(diag.get("missing_fields") or [])
    reason = (("missing required argument " + ", ".join(missing)) if missing and not diag.get("invalid_fields")
              else (snap["message"] or "").strip()) or "the arguments are refused"
    return ExtractedArguments(params=params, extraction=extraction_of(params, diag, reason))


def _second_pass(app, row, first, counter):
    """EXPERIMENT (variant ``retry_missing``): one more extraction naming the missing required values."""
    import fastworkflow
    from fastworkflow import ModuleType
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
        ExtractedArguments, extraction_of)
    from fastworkflow.parameters import argument_checks
    from fastworkflow.parameters.signatures import InputForParamExtraction
    from fastworkflow.parameters.signatures import _planning_command_of as planning_of

    folder = app.folderpath
    command, text = row["command"], row["text"]
    params_class = fastworkflow.RoutingRegistry.get_definition(folder).get_command_class(
        command, ModuleType.COMMAND_PARAMETERS_CLASS)
    extractor = InputForParamExtraction.create(app, command, text)
    extractor.command = f"{text}\n\n{RETRY_NOTE.format(names=', '.join(first.extraction.missing))}"
    counter.bump()
    try:
        params = extractor.extract_parameters(params_class, command, folder)
    except Exception:  # noqa: BLE001 - the second pass is an extra: the first answer stands
        return first
    finally:
        extractor.command = text
    if argument_checks.checks_enabled() and planning_of(app, command) is None:
        check = argument_checks.check_extracted_values(params, text)      # against the ORIGINAL text
        check.apply(params)
    diagnostics: dict = {}
    valid, message, _, _ = extractor.validate_parameters(app, command, params, diagnostics, None)
    if valid:
        return ExtractedArguments(params=params, extraction=extraction_of(params, {}), retried=True)
    missing = list(diagnostics.get("missing_fields") or [])
    reason = (("missing required argument " + ", ".join(missing)) if missing and not diagnostics.get("invalid_fields")
              else (message or "").strip()) or "the arguments are refused"
    return ExtractedArguments(params=params, extraction=extraction_of(params, diagnostics, reason), retried=True)


def _is_planning(app, command) -> bool:
    from fastworkflow.parameters.signatures import _planning_command_of as planning_of
    return planning_of(app, command) is not None


def _run_row(apps, row, variant, counter):
    import fastworkflow
    from fastworkflow import ModuleType

    app = apps[row["workflow"]]
    params_class = fastworkflow.RoutingRegistry.get_definition(app.folderpath).get_command_class(
        row["command"], ModuleType.COMMAND_PARAMETERS_CLASS)
    record = {"id": row["id"], "kind": row["kind"], "cls": row["cls"], "command": row["command"], "text": row["text"]}
    try:
        first, _ = _first_pass(app, row, counter)
    except Exception as exc:  # noqa: BLE001 - a failing row is a result, not a crash
        record.update(error=f"{type(exc).__name__}: {str(exc)[:160]}")
        return record
    first_calls = counter.n
    record["first"] = {"fields": classify(row, first.extraction, params_class), "asked": not first.valid,
                       "calls": first_calls, "retried": first.retried}
    final, calls = first, first_calls
    snap = getattr(_SNAP, "first", None)
    if _is_planning(app, row["command"]) and snap is not None:
        pre = _first_as_result(snap)
        record["plan"] = {
            "fired": bool(first.retried), "first_calls": snap["calls"], "final_calls": first_calls,
            "first": {"fields": classify(row, pre.extraction, params_class), "asked": not pre.valid},
            "final": {"fields": record["first"]["fields"], "asked": record["first"]["asked"]},
            "same_when_not_fired": (not first.retried) and classify(row, pre.extraction, params_class) == record["first"]["fields"]}
        if variant == "plan_no_retry":
            record["first"] = {"fields": record["plan"]["first"]["fields"], "asked": not pre.valid,
                               "calls": snap["calls"], "retried": False}
            final, calls = pre, snap["calls"]
    if variant == "retry_missing" and first.extraction.missing and not first.retried:
        final = _second_pass(app, row, first, counter)
        calls = counter.n
    record["final"] = {"fields": classify(row, final.extraction, params_class), "asked": not final.valid,
                       "calls": calls, "second_pass": final is not first}
    return record


# ------------------------------------------------------------------ language-model call tallies

class CallStats:
    """Process-wide tallies of language-model calls (thread-safe)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.lm_calls = 0
        self.prompt_chars = []
        self.prompt_messages = collections.Counter()  # messages per model call (few-shot demos add pairs)

    def add(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)


STATS = CallStats()


# ------------------------------------------------------------------ reporting

KINDS = ("execution", "planning")
CLASSES = ("a", "b", "c", "d", "e")
CLASS_NAMES = {"a": "a all given", "b": "b one missing", "c": "c awkward form", "d": "d optional absent", "e": "e decoy word"}


def tally(records, which="final"):
    t = collections.Counter()
    for r in records:
        if "error" in r:
            t["errors"] += 1
            continue
        part = r[which]
        t["rows"] += 1
        t["asked"] += part["asked"]
        t["calls"] += part["calls"]
        for f in part["fields"].values():
            k = f["kind"]
            t["fields"] += 1
            if k == "right":
                t["right"] += 1
            elif k == "ok_empty":
                t["right"] += 1
                t["ok_empty"] += 1
            elif k == "missing":
                t["missing"] += 1
                t["refused"] += f["refused"]
            else:
                t["made_up"] += 1
                t["made_up_wrong" if k == "made_up_wrong" else "made_up_invented"] += 1
    return t


def table(records, which="final", title=""):
    head = f"{'kind':10} {'class':16} {'rows':>4} {'fields':>6} {'right':>5} {'missing':>7} {'made_up':>7} {'asked':>5} {'calls':>5} {'err':>3}"
    lines = [title, head, "-" * len(head)]
    total = collections.Counter()
    for kind in (*KINDS, None):
        for cls in (*CLASSES, None):
            subset = [r for r in records if (kind is None or r["kind"] == kind) and (cls is None or r["cls"] == cls)]
            if not subset:
                continue
            t = tally(subset, which)
            if kind is None and cls is not None:
                continue
            lines.append(f"{kind or 'ALL':10} {CLASS_NAMES.get(cls, 'all') if cls else 'all':16} {t['rows']:>4} {t['fields']:>6} "
                         f"{t['right']:>5} {t['missing']:>7} {t['made_up']:>7} {t['asked']:>5} {t['calls']:>5} {t['errors']:>3}")
        lines.append("")
    t = tally(records, which)
    lines.append(f"totals: right {t['right']} (of which correctly empty {t['ok_empty']}), missing {t['missing']} "
                 f"(refused by a check {t['refused']}), made up {t['made_up']} (invented where none given {t['made_up_invented']}, "
                 f"different from gold {t['made_up_wrong']}); rows asking {t['asked']}/{t['rows']}; extractor calls {t['calls']}; "
                 f"errors {t['errors']}")
    return "\n".join(lines)


def delta(records):
    """Paired: each row's first pass against its final answer (variant retry_missing)."""
    out = collections.defaultdict(collections.Counter)
    for r in records:
        if "error" in r or not r["final"].get("second_pass"):
            continue
        d = out[r["kind"]]
        d["rows_with_second_pass"] += 1
        d["extra_calls"] += r["final"]["calls"] - r["first"]["calls"]
        for name, after in r["final"]["fields"].items():
            before = r["first"]["fields"][name]
            if before["kind"] == "missing" and after["kind"] == "right":
                d["recovered_correctly"] += 1
            if before["kind"] == "missing" and after["kind"].startswith("made_up"):
                d["gold_given_now_wrong"] += 1
            if before["kind"] in ("ok_empty", "missing") and after["kind"] == "made_up_invented" and before["kind"] != "made_up_invented":
                d["newly_made_up_where_none_given"] += 1
            if before["kind"] == "right" and after["kind"] != "right":
                d["right_lost"] += 1
    return out


def plan_report(records) -> str:
    """Planning rows: what the product's own second pass did, paired with the same call's first pass."""
    rows = [r for r in records if "plan" in r]
    worse = {"right": 0, "ok_empty": 0}
    labels = collections.defaultdict(list)
    tot = {"first": collections.Counter(), "final": collections.Counter()}
    for r in rows:
        p = r["plan"]
        for which in ("first", "final"):
            for f in p[which]["fields"].values():
                k = f["kind"]
                tot[which]["right" if k in ("right", "ok_empty") else "missing" if k == "missing" else "made_up"] += 1
            tot[which]["calls"] += p[which + "_calls"]
            tot[which]["asked"] += p[which]["asked"]
        if not p["fired"]:
            continue
        label = "iii changed nothing"
        for name, after in p["final"]["fields"].items():
            before = p["first"]["fields"][name]
            bk, ak = before["kind"], after["kind"]
            good = lambda k: k in ("right", "ok_empty")
            if good(bk) and not good(ak):
                label = "iv made up or lost a right one"
                break
            if bk == "missing" and ak.startswith("made_up"):
                label = "iv made up or lost a right one"
                break
            if bk == "missing" and ak == "right":
                label = "ii fixed a refused value" if before["refused"] else "i recovered a missing value"
        labels[label].append(f"{r['id']}/{r['cls']}")
    fired = sum(r["plan"]["fired"] for r in rows)
    out = [f"== plan second pass (paired, planning rows only) ==",
           f"rows where the second pass fired: {fired}/{len(rows)}"]
    for label in ("i recovered a missing value", "ii fixed a refused value", "iii changed nothing", "iv made up or lost a right one"):
        out.append(f"  {label}: {len(labels[label])} {labels[label]}")
    for which, name in (("first", "without the second pass"), ("final", "with the second pass (product)")):
        t = tot[which]
        out.append(f"{name}: fields right {t['right']}, missing {t['missing']}, made up {t['made_up']}; "
                   f"rows asking {t['asked']}; extractor calls {t['calls']}")
    out.append(f"first pass equals final when the product did not fire: {sum(r['plan']['same_when_not_fired'] for r in rows)}/"
               f"{sum(not r['plan']['fired'] for r in rows)}")
    return "\n".join(out)


def cmd_score(args) -> int:
    rows = [r for r in json.loads(DATA.read_text())["rows"] if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    counter = Counter()
    with tempfile.TemporaryDirectory(prefix="parameter_pipeline_gate_") as scratch:
        apps, model = _setup(Path(scratch))
        from fastworkflow.parameters.signatures import InputForParamExtraction

        original = InputForParamExtraction.extract_parameters

        def counted(self, *a, **k):
            counter.bump()
            return original(self, *a, **k)

        InputForParamExtraction.extract_parameters = counted
        validate_original = _install_snapshot()
        import dspy
        from fastworkflow.parameters import signatures

        STATS.reset()
        lm_call_original = dspy.BaseLM.__call__

        def counted_call(self, *a, **k):
            STATS.add(lm_calls=1)
            with STATS.lock:
                STATS.prompt_messages[len(k.get("messages") or ())] += 1
                STATS.prompt_chars.append(sum(len(str(m.get("content", ""))) for m in (k.get("messages") or ())))
            return lm_call_original(self, *a, **k)

        dspy.BaseLM.__call__ = counted_call
        started = time.monotonic()
        try:
            with concurrent.futures.ThreadPoolExecutor(max(1, args.workers)) as pool:
                records = list(pool.map(lambda r: _run_row(apps, r, args.variant, counter), rows))
        finally:
            wall = time.monotonic() - started
            dspy.BaseLM.__call__ = lm_call_original
            InputForParamExtraction.extract_parameters = original
            InputForParamExtraction.validate_parameters = validate_original
    print(f"split={args.split} variant={args.variant} rows={len(rows)} model(LLM_PARAM_EXTRACTION)={model} (single run, cache off)\n")
    print(table(records, "final", f"== {args.variant} =="))
    print("\n" + plan_report(records))
    if args.variant == "retry_missing":
        print("\n" + table(records, "first", "== first pass of the same rows (the product's own answer, paired) =="))
        print("\ndelta (second pass against the same row's first pass), by kind:")
        for kind, d in sorted(delta(records).items()):
            print(f"  {kind}: " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))
    print(f"\nMODEL CALLS (dspy.BaseLM.__call__, every variant): {STATS.lm_calls}; wall time {wall:.1f}s "
          f"({wall / max(1, len(rows)):.2f}s per row, {max(1, args.workers)} workers)\n"
          f"messages per model call {dict(sorted(STATS.prompt_messages.items()))}; prompt characters per call: "
          f"min {min(STATS.prompt_chars, default=0)}, mean {sum(STATS.prompt_chars) / max(1, len(STATS.prompt_chars)):.0f}, "
          f"max {max(STATS.prompt_chars, default=0)}")
    if args.show_errors:
        for r in records:
            if "error" in r:
                print(f"\nERROR {r['id']} {r['command']}: {r['error']}")
                continue
            for name, f in r["final"]["fields"].items():
                if f["kind"].startswith("made_up") or f["kind"] == "missing":
                    print(f"{f['kind']:17} {r['id']} {r['kind'][:4]}/{r['cls']} {r['command']}.{name}: gold={f['gold']!r} got={f['got']!r}"
                          f"{' (refused)' if f['refused'] else ''}  <- {r['text']!r}")
    return 0


# ------------------------------------------------------------------ spans

def _outside_repo(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise SystemExit(f"refusing to write {resolved}: it is inside the repository (private data stays out of the repo)")
    return resolved


def _command_attributes(db, span_id) -> dict:
    """The attributes of the nearest `fw.command` span at or above *span_id*, or {} when there is none (an extraction
    at resolve is its part's, not a command's). A store written before span contract 63 is not read."""
    hops = 0
    while span_id and hops < 1000:
        row = db.execute("SELECT name, parent_span_id, attributes FROM spans WHERE span_id=?", (span_id,)).fetchone()
        if row is None:
            return {}
        if row["name"] == "fw.command":
            return json.loads(row["attributes"] or "{}")
        span_id, hops = row["parent_span_id"], hops + 1
    return {}


def cmd_spans(args) -> int:
    out = _outside_repo(args.out)
    db = sqlite3.connect(f"file:{Path(args.db).expanduser().resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    n = 0
    with open(out, "w") as fh:
        for s in db.execute("SELECT span_id, trace_id, parent_span_id, command_name, attributes FROM spans "
                            "WHERE name='fw.parameters' ORDER BY start_ns"):
            a = json.loads(s["attributes"] or "{}")
            pa = _command_attributes(db, s["parent_span_id"])
            turn = db.execute("SELECT user_message FROM turns WHERE turn_key=?", (s["trace_id"],)).fetchone()
            llm = [json.loads(x["attributes"]) for x in db.execute(
                "SELECT attributes FROM spans WHERE parent_span_id=? AND name='fw.llm.call'", (s["span_id"],))]
            fh.write(json.dumps({
                "command_name": a.get("command_name") or s["command_name"],
                "command_text": pa.get("raw_command"),          # nearest fw.command ancestor
                "turn_user_message": turn["user_message"] if turn else None,
                "extraction_method": a.get("extraction_method"),
                "recorded_values": pa.get("parameters"),         # the executed command's parameters (after validation)
                "missing_fields": a.get("missing_fields") or [],
                "invalid_fields": a.get("invalid_fields") or [],
                "argument_check_refusals": a.get("argument_check_refusals"),
                "argument_check_retry": a.get("argument_check_retry"),
                "argument_check_refusals_final": a.get("argument_check_refusals_final"),
                "argument_check_respelled": a.get("argument_check_respelled"),
                "retry_round": a.get("retry_round"), "retry_round_ordinal": a.get("retry_round_ordinal"),
                "parameters_valid": a.get("parameters_valid"),
                "llm_calls_under_span": len(llm),
                "llm_output": next((str(x.get("output"))[:1500] for x in llm if x.get("output")), None),
                "attribute_names": sorted(a),
            }) + "\n")
            n += 1
    print(f"{n} fw.parameters spans -> {out}")
    return 0


def cmd_summary(args) -> int:
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    n = len(rows) or 1
    methods = collections.Counter(r["extraction_method"] or "(none)" for r in rows)
    print(f"spans: {len(rows)}")
    for m, c in methods.most_common():
        print(f"  extraction_method {m}: {c}")
    missing = sum(bool(r["missing_fields"]) for r in rows)
    refused = sum(bool(r["argument_check_refusals"]) for r in rows)
    retried = sum(bool(r["argument_check_retry"] or r["retry_round"]) for r in rows)
    print(f"with a missing required field: {missing} ({missing / n:.1%})")
    print(f"with an argument-check refusal: {refused} ({refused / n:.1%})")
    print(f"retried (argument_check_retry or retry_round): {retried} ({retried / n:.1%})")
    print(f"with command text recorded: {sum(bool(r['command_text']) for r in rows)}; "
          f"with values recorded: {sum(r['recorded_values'] is not None for r in rows)}; "
          f"with the extractor's raw output: {sum(bool(r['llm_output']) for r in rows)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--split", choices=("dev", "confirm"), required=True)
    s.add_argument("--variant", choices=("baseline", "retry_missing", "plan_no_retry"), default="baseline")
    s.add_argument("--limit", type=int)
    s.add_argument("--show-errors", action="store_true")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(fn=cmd_score)
    p = sub.add_parser("spans")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_spans)
    q = sub.add_parser("spans-summary")
    q.add_argument("--in", dest="input", required=True)
    q.set_defaults(fn=cmd_summary)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
