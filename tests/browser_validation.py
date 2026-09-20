"""Run the browser checks instead of letting them skip themselves.

Every DOM test in this suite starts with the same three lines: read
``TEST_JSDOM_ROOT``, and skip when it is unset. That is the right default for
``pytest`` on a machine with no jsdom — but it also means a run that was
supposed to prove the UI works can report success having executed no browser
code at all, which is how "relevant browser interaction checks passed" ends up
meaning nothing.

This module is the other mode: prerequisites are checked up front and reported
as failures rather than skips, the DOM tests are discovered rather than listed
(so a new one is covered the day it is written), and a run in which a required
check skipped is a failed run. It adds no framework — it selects and reports on
the existing jsdom tests, which ``pytest`` still runs.

Usage, from the repository root with the venv active::

    TEST_JSDOM_ROOT=/path/to/jsdom-install python -m tests.browser_validation

``TEST_JSDOM_ROOT`` is a directory containing ``node_modules/jsdom`` — the
harness scripts require it from there by path, so the location is the caller's
to configure and is never assumed here. Narrow the run by passing test files:

    TEST_JSDOM_ROOT=... python -m tests.browser_validation tests/test_x.py
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

JSDOM_ROOT_VAR = "TEST_JSDOM_ROOT"
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


class BrowserPrerequisiteMissing(RuntimeError):
    """A browser check cannot run, and saying so is the point."""


@dataclass
class ValidationReport:
    required: list[str]
    passed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    pytest_exit_code: Optional[int] = None

    @property
    def ok(self) -> bool:
        """Success needs every required check to have actually executed."""
        return bool(
            self.required
            and self.pytest_exit_code == 0
            and not self.skipped
            and not self.failed
            and not self.missing
        )

    def summary(self) -> str:
        lines = [
            f"required browser checks: {len(self.required)}",
            f"executed and passed:     {len(self.passed)}",
        ]
        for label, nodes in (
            ("skipped", self.skipped),
            ("failed", self.failed),
            ("never reported", self.missing),
        ):
            if nodes:
                lines.append(f"{label}: {len(nodes)}")
                lines.extend(f"  - {node}" for node in nodes)
        if not self.required:
            lines.append(
                "no browser checks were discovered at all, which is itself a "
                "failure: this gate exists to run them"
            )
        return "\n".join(lines)


def _jsdom_functions(tree: ast.Module, source: str) -> set[str]:
    """Names in one module that reach jsdom, directly or through a helper.

    Two of the DOM tests call the harness through a module-local helper, so
    naming or direct-reference matching would miss them; the closure below
    keeps discovery honest as more helpers appear.
    """
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reaching = {
        name
        for name, node in functions.items()
        if JSDOM_ROOT_VAR in (ast.get_source_segment(source, node) or "")
    }
    growing = True
    while growing:
        growing = False
        for name, node in functions.items():
            if name in reaching:
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in reaching
                ):
                    reaching.add(name)
                    growing = True
                    break
    return reaching


def discover_required_dom_tests(
    paths: Optional[Iterable[Path]] = None,
) -> list[str]:
    """Node ids of every test that drives a real DOM, newest ones included."""
    candidates: list[Path] = []
    for path in paths or [TESTS_DIR]:
        path = Path(path)
        if path.is_dir():
            candidates.extend(sorted(path.glob("test_*.py")))
        elif path.is_file():
            candidates.append(path)

    node_ids: list[str] = []
    for candidate in candidates:
        source = candidate.read_text(encoding="utf-8", errors="ignore")
        if JSDOM_ROOT_VAR not in source:
            continue
        tree = ast.parse(source)
        reaching = _jsdom_functions(tree, source)
        try:
            relative = candidate.resolve().relative_to(REPO_ROOT)
        except ValueError:
            relative = candidate
        node_ids.extend(
            f"{relative}::{name}"
            for name in sorted(reaching)
            if name.startswith("test_")
        )
    return node_ids


def check_prerequisites(jsdom_root: Optional[str] = None) -> str:
    """Refuse to pretend. Returns the configured jsdom root."""
    root = jsdom_root if jsdom_root is not None else os.environ.get(JSDOM_ROOT_VAR)
    if not root:
        raise BrowserPrerequisiteMissing(
            f"{JSDOM_ROOT_VAR} is not set. The DOM tests skip themselves "
            "without it, so this validation has nothing to run. Point it at a "
            "directory containing node_modules/jsdom."
        )
    if shutil.which("node") is None:
        raise BrowserPrerequisiteMissing(
            "node is not on PATH; the jsdom harness scripts cannot run."
        )
    module = Path(root).expanduser() / "node_modules" / "jsdom"
    if not module.is_dir():
        raise BrowserPrerequisiteMissing(
            f"{JSDOM_ROOT_VAR}={root!r} has no node_modules/jsdom. The harness "
            "requires jsdom from that path; install it there (offline: copy an "
            "existing install) rather than pointing at a different directory."
        )
    probe = subprocess.run(
        ["node", "-e", f"require({str(module)!r})"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise BrowserPrerequisiteMissing(
            f"jsdom under {JSDOM_ROOT_VAR}={root!r} is not loadable by node:\n"
            f"{probe.stderr.strip()}"
        )
    return str(Path(root).expanduser())


def read_junit_outcomes(report_path: Path) -> dict[str, str]:
    """Outcome per node id from a pytest JUnit report.

    Parametrised cases report one entry each; they are folded back onto the
    function that generated them, and a function counts as executed only when
    every one of its cases did.
    """
    outcomes: dict[str, str] = {}
    for case in ElementTree.parse(report_path).getroot().iter("testcase"):
        module = (case.get("classname") or "").split(".")
        name = (case.get("name") or "").split("[", 1)[0]
        if not module or not name:
            continue
        node = f"{Path(*module).with_suffix('.py')}::{name}"
        if any(child.tag == "skipped" for child in case):
            outcome = "skipped"
        elif any(child.tag in {"failure", "error"} for child in case):
            outcome = "failed"
        else:
            outcome = "passed"
        # Worst outcome wins across a function's parametrisations.
        ranking = {"passed": 0, "skipped": 1, "failed": 2}
        if ranking[outcome] >= ranking.get(outcomes.get(node, "passed"), 0):
            outcomes[node] = outcome
    return outcomes


def outcome_for(node_id: str, outcomes: dict[str, str]) -> Optional[str]:
    """Find a node's outcome despite pytest's rootdir-relative reporting.

    A JUnit `classname` is relative to whatever pytest chose as its rootdir,
    which is not always this repository, so the recorded key can be a shorter
    path than the node id that asked for it. Test file names are unique within
    `tests/`, so matching on the file name and the function is exact here.
    """
    path, _, name = node_id.partition("::")
    for key, outcome in outcomes.items():
        key_path, _, key_name = key.partition("::")
        if key_name == name and (
            key_path == path or Path(key_path).name == Path(path).name
        ):
            return outcome
    return None


def validate(
    paths: Optional[Sequence[Path]] = None,
    *,
    jsdom_root: Optional[str] = None,
    extra_pytest_args: Sequence[str] = (),
) -> ValidationReport:
    """Run the discovered browser checks and report what actually executed."""
    root = check_prerequisites(jsdom_root)
    required = discover_required_dom_tests(paths)
    report = ValidationReport(required=required)
    if not required:
        return report

    environment = dict(os.environ, **{JSDOM_ROOT_VAR: root})
    with tempfile.TemporaryDirectory(prefix="fw-browser-validation-") as scratch:
        junit = Path(scratch) / "browser-validation.xml"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *required,
                "-p",
                "no:randomly",
                "-q",
                f"--junitxml={junit}",
                *extra_pytest_args,
            ],
            cwd=REPO_ROOT,
            env=environment,
        )
        report.pytest_exit_code = completed.returncode
        outcomes = read_junit_outcomes(junit) if junit.exists() else {}

    for node in required:
        outcome = outcome_for(node, outcomes)
        if outcome == "passed":
            report.passed.append(node)
        elif outcome == "skipped":
            report.skipped.append(node)
        elif outcome == "failed":
            report.failed.append(node)
        else:
            report.missing.append(node)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the jsdom browser checks and fail when any required check "
            "did not execute."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="test files or directories to validate (default: tests/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the required browser checks without running them",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        dest="pytest_args",
        help="extra argument forwarded to pytest (repeatable)",
    )
    arguments = parser.parse_args(argv)

    if arguments.list:
        for node in discover_required_dom_tests(arguments.paths or None):
            print(node)
        return 0

    try:
        report = validate(
            arguments.paths or None, extra_pytest_args=arguments.pytest_args
        )
    except BrowserPrerequisiteMissing as missing:
        print(f"browser validation cannot run: {missing}", file=sys.stderr)
        return 2

    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
