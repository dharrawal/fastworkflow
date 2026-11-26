#!/usr/bin/env python3
"""The gates of the package reorganisation (bd fix-bl97), run after each step of ``moves.py``.

    python scripts/reorg/check.py g1 --step 2 --root . --root ../app      # no old path remains
    python scripts/reorg/check.py g2 [--record|--compare BASELINE.json]     # every module imports, --help, wheel
    python scripts/reorg/check.py g3 --record BASELINE.json [--workflow PATH ...]   # load problems, preview --check
    python scripts/reorg/check.py g3 --compare BASELINE.json [--workflow PATH ...]
    python scripts/reorg/check.py g4 --record BASELINE.json [--workflow PATH ...]   # training signatures
    python scripts/reorg/check.py g4 --compare BASELINE.json [--workflow PATH ...]
    python scripts/reorg/check.py patches [--root tests]                   # every patch target resolves

``--repo`` (default: the repo of this script) is the checkout whose ``fastworkflow`` package is checked; every
subprocess runs with it first on ``PYTHONPATH``, so a scratch clone is checked as itself. Nothing here trains,
calls a model or writes inside a workflow: G3 and G4 work on copies of the workflows (without their trained
artefacts) under ``--scratch``. G4 records each workflow's training signature as ``fastworkflow train`` would
compute it now, with the copy's folder and the package folder written as ``<WF>`` and ``<PKG>``, so a
baseline taken before a step compares with the same workflows after it: equal signatures mean the next
``fastworkflow train`` plans exactly what it planned before (nothing retrains because of the step).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moves as table  # noqa: E402  (the mapping table beside this file)

REPO = table.REPO
warnings.filterwarnings("ignore", category=SyntaxWarning)  # ast.parse of files with old escape sequences
DEFAULT_SCRATCH = Path(tempfile.gettempdir()) / "fastworkflow_reorg_check"
NOT_MODULES = ("examples", "_workflows", "skills_for_coding_fastworkflows", "_framework_commands", "___")


def run_python(repo: Path, code: str, timeout: int = 600, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-c", code], cwd=str(cwd or repo), env=env, capture_output=True,
                          text=True, timeout=timeout)


# ---------------------------------------------------------------- G1


def _python_import_hits(path: Path, root: Path, old_modules: set[str]) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return []
    module, is_init = table._module_of(path, root)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if module is None:
                    continue
                pkg = module.split(".") if is_init else module.split(".")[:-1]
                pkg = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                base = ".".join(pkg + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            for name in [base] + [f"{base}.{a.name}" for a in node.names]:
                if name in old_modules or any(name.startswith(o + ".") for o in old_modules):
                    hits.append(f"{path}:{node.lineno}: import of {name}")
    return hits


def g1(args) -> int:
    moves = table.selected(args.step)
    rules = table.text_rules(moves)
    old_modules = {f"{table.PACKAGE}.{m.old}" for m in moves if m.new != m.old}
    problems: list[str] = []
    for m in moves:
        src, dst = args.repo / table.source_path(m), table.target_path(m)
        if m.kind != "to_package" and src.exists():
            problems.append(f"{src}: still exists")
        if dst and not (args.repo / dst).exists():
            problems.append(f"{args.repo / dst}: missing")
    for root in args.root or [args.repo]:
        for path, kind in table.files_under(root, args.changelog):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern, _new in rules:
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    problems.append(f"{path}:{line}: {match.group(0)}")
            if kind == "python":
                problems.extend(_python_import_hits(path, root, old_modules))
    for problem in problems[: args.limit]:
        print(problem)
    print(f"G1: {len(problems)} old path(s) remain" + (" (first shown)" if len(problems) > args.limit else ""))
    return 1 if problems else 0


# ---------------------------------------------------------------- G2


def package_modules(repo: Path) -> list[str]:
    package = repo / table.PACKAGE
    names = []
    for dirpath, dirnames, filenames in os.walk(package):
        rel = Path(dirpath).relative_to(repo)
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__" and not d.startswith(NOT_MODULES)
                             and d not in ("static", "tests"))
        for name in sorted(filenames):
            if name.endswith(".py"):
                parts = list(rel.parts) + ([] if name == "__init__.py" else [name[:-3]])
                names.append(".".join(parts))
    return names


def g2(args) -> int:
    """Every module imports in a fresh process, ``from fastworkflow import *``, ``--help`` and the wheel. With
    --record/--compare the failures are compared by where each module lives once every move is made, because a
    few modules fail on any tree (a script that needs another checkout, an import-order problem of a library)."""
    failures: dict[str, str] = {}
    modules = package_modules(args.repo)
    mapping = table.dotted_map(table.MOVES)

    def one(name: str) -> tuple[str, str] | None:
        # An entry module (__main__) parses its arguments when imported: give it --help, which exits 0.
        argv = "import sys; sys.argv = ['check', '--help']; " if name.endswith("__main__") else ""
        result = run_python(args.repo, f"{argv}import importlib; importlib.import_module({name!r})", timeout=300)
        if result.returncode:
            last = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else str(result.returncode)
            return table.remap(name, mapping) or name, last.replace(str(args.repo), "<REPO>")
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for problem in pool.map(one, modules):
            if problem:
                failures[problem[0]] = problem[1]
    print(f"G2: {len(modules)} modules imported in fresh processes, {len(failures)} failed")
    star = run_python(args.repo, "from fastworkflow import *")
    if star.returncode:
        failures["from fastworkflow import *"] = star.stderr.strip().splitlines()[-1]
    helped = run_python(args.repo, "import sys; from fastworkflow.cli import main; sys.argv = ['fastworkflow', "
                                   "'--help']; main()")
    if helped.returncode or "usage" not in helped.stdout.lower():
        failures["fastworkflow --help"] = f"exit {helped.returncode}: {helped.stderr.strip()[-300:]}"
    else:
        print("G2: fastworkflow --help ok")
    if not args.no_wheel:
        out = args.scratch / "wheel"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        built = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(out),
                                str(args.repo)], capture_output=True, text=True, timeout=900)
        wheels = list(out.glob("*.whl"))
        if built.returncode or not wheels:
            failures["wheel"] = f"exit {built.returncode}: {built.stderr.strip()[-400:]}"
        else:
            print(f"G2: wheel ok ({wheels[0].name})")
    for name, failure in sorted(failures.items()):
        print(f"  {name}: {failure}")
    if args.record or args.compare:
        # Which modules fail is the gate; the message names paths that differ between checkouts.
        return _baseline(args, {name: "fails" for name in failures}, "G2")
    return 1 if failures else 0


# ---------------------------------------------------------------- workflows (G3, G4)


def default_workflows(repo: Path) -> list[Path]:
    found = []
    for parent in (repo / "fastworkflow" / "examples", repo / "tests", repo / "fastworkflow" / "_workflows"):
        for child in sorted(parent.iterdir()) if parent.is_dir() else []:
            if (child / "_commands").is_dir():
                found.append(child)
    return found


def copy_workflow(workflow: Path, scratch: Path, repo: Path) -> Path:
    """A copy without trained artefacts, at a path that depends only on the workflow's place in its repo, so
    copies taken before and after a step, or of a clone, land in the same place."""
    try:
        rel = workflow.resolve().relative_to(repo.resolve())
    except ValueError:
        rel = Path("extra") / workflow.name
    target = scratch / "workflows" / rel
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(workflow, target, ignore=shutil.ignore_patterns(
        "___*", "__pycache__", ".venv", ".git", "node_modules"), symlinks=True)
    docs = workflow.parent / "docs"
    if docs.is_dir():
        shutil.rmtree(target.parent / "docs", ignore_errors=True)
        shutil.copytree(docs, target.parent / "docs", symlinks=True)
    return target


def workflows_of(args) -> list[Path]:
    found = (default_workflows(args.repo) if not args.only else []) + [Path(w).resolve() for w in args.workflow]
    return [w for w in found if not any(w.resolve() == (args.repo / x).resolve() for x in args.exclude)]


# A copy of a framework workflow is read as the framework's own (its pinned layout, its SlotInput) by pointing
# get_internal_workflow_path at the copies. An application's copy keeps the package's own framework workflows:
# their command files may only be imported from inside the package.
PRELUDE = r"""
import os, fastworkflow
_copies = {copies!r}
if {framework!r}:
    fastworkflow.get_internal_workflow_path = lambda name: os.path.join(_copies, "fastworkflow", "_workflows", name)
"""


def _is_framework_copy(copy: Path, args) -> bool:
    return (args.scratch / "workflows" / "fastworkflow" / "_workflows") in copy.parents

G3_CODE = PRELUDE + r"""
import io, json, re, sys, contextlib
from fastworkflow.command_preview import load_problems
from fastworkflow.cli import main
folder = {folder!r}
problems = sorted(str(p) for p in load_problems(folder))
out = io.StringIO()
sys.argv = ["fastworkflow", "command", "preview", folder, "--check"]
code = 0
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
    try:
        main()
    except SystemExit as exit:
        code = exit.code or 0
lines = sorted(line.strip() for line in out.getvalue().splitlines()
               if re.match(r"\s*(missing|stale|extra):", line))
print(json.dumps({{"load_problems": problems, "preview_exit": code, "preview": lines}}))
"""


def _normalise(text: str, args) -> str:
    return text.replace(str(args.scratch / "workflows"), "<COPIES>").replace(str(args.repo), "<REPO>")


def _copy_all(args) -> list[tuple[Path, Path]]:
    pairs = []
    for name in ("command_metadata_extraction", "plan_metadata_extraction"):
        copy_workflow(args.repo / "fastworkflow" / "_workflows" / name, args.scratch, args.repo)
    for workflow in workflows_of(args):
        pairs.append((workflow, copy_workflow(workflow, args.scratch, args.repo)))
    return pairs


def _baseline(args, results: dict, gate: str) -> int:
    if args.record:
        Path(args.record).write_text(json.dumps(results, indent=1, sort_keys=True))
        print(f"{gate}: baseline of {len(results)} workflow(s) written to {args.record}")
        return 0
    baseline = json.loads(Path(args.compare).read_text())
    differences = [key for key in sorted(set(baseline) | set(results)) if baseline.get(key) != results.get(key)]
    for key in differences:
        before, after = baseline.get(key), results.get(key)
        if gate == "G4" and isinstance(before, dict) and isinstance(after, dict) \
                and "signature" in before and "signature" in after:
            b, a = before["signature"], after["signature"]
            fields = sorted(k for k in set(b) | set(a) if k != "contexts" and b.get(k) != a.get(k))
            contexts = sorted(c for c in set(b.get("contexts") or {}) | set(a.get("contexts") or {})
                              if (b.get("contexts") or {}).get(c) != (a.get("contexts") or {}).get(c))
            print(f"  {key}: global fields changed {fields}; contexts changed {contexts}")
        else:
            print(f"  {key}:\n    before {json.dumps(before)[:600]}\n    after  {json.dumps(after)[:600]}")
    print(f"{gate}: {len(differences)} workflow(s) differ from the baseline")
    return 1 if differences else 0


def g3(args) -> int:
    """Load problems and ``preview --check`` per workflow, compared with a baseline: some test workflows
    exist to have load problems, and most have no committed previews, so "unchanged" is the gate."""
    results = {}
    copies = str(args.scratch / "workflows")
    for workflow, copy in _copy_all(args):
        key = str(copy.relative_to(args.scratch / "workflows"))
        done = run_python(args.repo, G3_CODE.format(copies=copies, folder=str(copy),
                                                    framework=_is_framework_copy(copy, args)), timeout=600)
        if done.returncode:
            results[key] = {"error": _normalise(done.stderr.strip()[-400:], args)}
        else:
            results[key] = json.loads(_normalise(done.stdout.strip().splitlines()[-1], args))
        r = results[key]
        print(f"G3 {key}: " + ("ERROR" if "error" in r else
              f"{len(r['load_problems'])} load problem(s), preview --check exit {r['preview_exit']}"
              f" ({len(r['preview'])} missing/stale/extra)"))
    return _baseline(args, results, "G3")


G4_CODE = PRELUDE + r"""
import json
from dotenv import dotenv_values
env = {{k: v for k, v in dotenv_values({env!r}).items() if v is not None}}
fastworkflow.init(env_vars=env)
from fastworkflow.train import determinism, intent_model_layout, selective_training
folder = {folder!r}
signature, unresolved = selective_training.compute_training_signature(
    folder, selective_training.contexts_for_training(folder), seed=determinism.get_model_seed(),
    intent_model_layout_name=intent_model_layout.layout_for_workflow(folder))
text = json.dumps({{"signature": signature.model_dump(), "unresolved": unresolved}}, sort_keys=True, default=str)
print(text.replace(folder, "<WF>").replace(_copies, "<COPIES>").replace(
    fastworkflow.get_fastworkflow_package_path(), "<PKG>"))
"""


def g4(args) -> int:
    # Resolved here: the check runs in the scratch folder, where a relative path names no file.
    env_file = Path(args.env or args.repo / "fastworkflow" / "examples" / "fastworkflow.env").resolve()
    if not env_file.is_file():
        raise SystemExit(f"G4: env file {env_file} does not exist")
    results: dict[str, object] = {}
    copies = str(args.scratch / "workflows")
    for workflow, copy in _copy_all(args):
        key = str(copy.relative_to(args.scratch / "workflows"))
        done = run_python(args.repo, G4_CODE.format(copies=copies, env=str(env_file), folder=str(copy),
                                                   framework=_is_framework_copy(copy, args)),
                          timeout=900, cwd=args.scratch)
        if done.returncode:
            results[key] = {"error": done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "failed"}
        else:
            results[key] = json.loads(done.stdout.strip().splitlines()[-1])
        print(f"G4 {'ok  ' if 'error' not in results[key] else 'ERR '} {key}")
    return _baseline(args, results, "G4")


# ---------------------------------------------------------------- patch targets


def patch_targets(roots: list[Path]) -> list[tuple[str, str]]:
    found = []
    for root in roots:
        for path, kind in table.files_under(root):
            if kind != "python":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, ValueError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                func = ast.unparse(node.func)
                if not (func.endswith("patch") or func.endswith("monkeypatch.setattr")
                        or func.endswith("monkeypatch.delattr") or func.endswith("mocker.patch")):
                    continue
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str) \
                        and first.value.startswith(table.PACKAGE + "."):
                    found.append((f"{path}:{node.lineno}", first.value))
    return found


RESOLVE_CODE = r"""
import importlib, importlib.util, json, sys
targets = json.loads(sys.stdin.read())
bad = []
for where, target in targets:
    parts = target.split(".")
    obj = None
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        except BaseException:  # a module that runs at import (argparse, sys.exit): read its names instead
            obj = "raised"
            name = ".".join(parts[:i - 1])
            spec = importlib.util.find_spec(name) if i > 1 else None
            defined = set()
            if spec and spec.origin and spec.origin.endswith(".py"):
                import ast
                for node in ast.parse(open(spec.origin).read()).body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        defined.add(node.name)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        defined.update((a.asname or a.name).split(".")[0] for a in node.names)
                    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                        for target_node in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                            if isinstance(target_node, ast.Name):
                                defined.add(target_node.id)
            if parts[i - 1] not in defined:
                bad.append(f"{where}: {target}")
            break
        rest = parts[i:]
        break
    if obj == "raised":
        continue
    try:
        if obj is None:
            raise AttributeError(target)
        for name in rest:
            obj = getattr(obj, name)
    except AttributeError:
        bad.append(f"{where}: {target}")
print(json.dumps(bad))
"""


def patches(args) -> int:
    targets = patch_targets(args.root or [args.repo / "tests"])
    env = dict(os.environ, PYTHONPATH=str(args.repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run([sys.executable, "-c", RESOLVE_CODE], input=json.dumps(targets), cwd=str(args.repo),
                          env=env, capture_output=True, text=True, timeout=900)
    if done.returncode:
        print(done.stderr[-800:])
        return 1
    bad = json.loads(done.stdout.strip().splitlines()[-1])
    for line in bad:
        print("  " + line)
    print(f"patches: {len(targets)} dotted patch target(s), {len(bad)} do not resolve")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gate", choices=["g1", "g2", "g3", "g4", "patches"])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--step", type=int, action="append")
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--workflow", action="append", default=[], help="an extra workflow folder (repeatable)")
    parser.add_argument("--only", action="store_true", help="only the --workflow folders, not the repo's own")
    parser.add_argument("--exclude", action="append", default=[],
                        help="a workflow folder, relative to the repo, to leave out (repeatable)")
    parser.add_argument("--changelog", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-wheel", action="store_true")
    parser.add_argument("--env", type=Path, help="g4: the env file (default: the repo's examples/fastworkflow.env)")
    record = parser.add_mutually_exclusive_group()
    record.add_argument("--record", help="g2, g3, g4: write the baseline JSON here")
    record.add_argument("--compare", help="g2, g3, g4: compare with this baseline JSON")
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.gate in ("g3", "g4") and not (args.record or args.compare):
        parser.error(f"{args.gate} needs --record or --compare")
    return {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "patches": patches}[args.gate](args)


if __name__ == "__main__":
    sys.exit(main())
