"""The gate that stops a browser-free run from reporting browser coverage.

Nothing here is mocked: discovery runs against this suite's real DOM tests,
the prerequisite check runs against the real environment, and the outcome
reader parses a JUnit report written by a real pytest run. The one thing these
tests deliberately never contain is the literal name of the jsdom environment
variable — the gate discovers browser checks by looking for it in the source,
so spelling it here would enrol this file in its own validation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests import browser_validation
from tests.browser_validation import (
    BrowserPrerequisiteMissing,
    JSDOM_ROOT_VAR,
    ValidationReport,
    check_prerequisites,
    discover_required_dom_tests,
    outcome_for,
    read_junit_outcomes,
)


def _fake_dom_test_file(directory: Path, body: str) -> Path:
    """A test file shaped like this suite's DOM tests, outside the repo."""
    path = directory / "test_fake_browser_check.py"
    path.write_text(
        "import os\n"
        "import pytest\n"
        "\n"
        "\n"
        "def _drive_the_dom():\n"
        f"    return os.environ.get({JSDOM_ROOT_VAR!r})\n"
        "\n"
        "\n"
        f"{body}",
        encoding="utf-8",
    )
    return path


def test_the_real_dom_tests_are_all_discovered():
    required = discover_required_dom_tests()

    assert len(required) >= 17
    assert "tests/test_chatbot_hierarchy.py::test_hierarchy_dom_clicks" in required
    assert all("::test_" in node for node in required)
    for node in required:
        source = Path(node.split("::")[0]).read_text(encoding="utf-8")
        assert JSDOM_ROOT_VAR in source


def test_a_browser_check_reached_through_a_helper_is_still_discovered(tmp_path):
    """Two real DOM tests call the harness through a module-local helper.

    Discovery that only looked for the variable in the test body would report
    coverage of a smaller suite than exists, which is the failure this whole
    module is about.
    """
    path = _fake_dom_test_file(
        tmp_path,
        "def test_through_a_helper():\n    assert _drive_the_dom() is not None\n",
    )

    required = discover_required_dom_tests([path])

    assert required == [f"{path}::test_through_a_helper"]


def test_the_helper_itself_is_not_reported_as_a_check(tmp_path):
    path = _fake_dom_test_file(
        tmp_path,
        "def test_through_a_helper():\n    assert _drive_the_dom() is not None\n",
    )

    assert not any(
        "_drive_the_dom" in node for node in discover_required_dom_tests([path])
    )


def test_an_unconfigured_jsdom_location_is_an_error_not_a_skip(monkeypatch):
    monkeypatch.delenv(JSDOM_ROOT_VAR, raising=False)

    with pytest.raises(BrowserPrerequisiteMissing) as failure:
        check_prerequisites()

    assert JSDOM_ROOT_VAR in str(failure.value)


def test_a_configured_location_without_jsdom_is_reported_as_such(tmp_path):
    with pytest.raises(BrowserPrerequisiteMissing) as failure:
        check_prerequisites(str(tmp_path))

    assert "node_modules/jsdom" in str(failure.value)


def test_a_run_whose_checks_all_skipped_is_not_a_successful_run():
    report = ValidationReport(
        required=["tests/test_x.py::test_dom"],
        skipped=["tests/test_x.py::test_dom"],
        pytest_exit_code=0,
    )

    assert report.ok is False
    assert "skipped" in report.summary()


def test_discovering_no_checks_at_all_is_also_a_failure():
    report = ValidationReport(required=[], pytest_exit_code=0)

    assert report.ok is False
    assert "no browser checks were discovered" in report.summary()


def test_outcomes_are_read_back_from_a_real_pytest_report(tmp_path):
    """A skipped browser check reads as skipped, not as absent."""
    path = _fake_dom_test_file(
        tmp_path,
        "def test_that_passes():\n"
        "    assert _drive_the_dom() is None or True\n"
        "\n"
        "\n"
        "def test_that_skips():\n"
        "    _drive_the_dom()\n"
        "    pytest.skip('the dependency is not really here')\n",
    )
    junit = tmp_path / "report.xml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(path),
            "-p",
            "no:randomly",
            "-q",
            f"--junitxml={junit}",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )

    outcomes = read_junit_outcomes(junit)

    assert outcome_for(f"{path}::test_that_passes", outcomes) == "passed"
    assert outcome_for(f"{path}::test_that_skips", outcomes) == "skipped"
    assert outcome_for(f"{path}::test_never_written", outcomes) is None


def test_the_gate_reports_failure_when_a_required_check_skips(tmp_path):
    """End to end, with a real pytest run underneath.

    The prerequisites are the real ones: without a configured jsdom this test
    has nothing to say and says so, which is the skip the ordinary suite is
    allowed to keep.
    """
    try:
        check_prerequisites()
    except BrowserPrerequisiteMissing as missing:
        pytest.skip(str(missing))

    path = _fake_dom_test_file(
        tmp_path,
        "def test_that_skips():\n"
        "    _drive_the_dom()\n"
        "    pytest.skip('the dependency is not really here')\n",
    )

    report = browser_validation.validate([path])

    assert report.ok is False
    assert report.skipped == [f"{path}::test_that_skips"]
    assert report.passed == []


def test_the_gate_does_not_count_itself_as_a_browser_check():
    required = discover_required_dom_tests()

    assert not any("test_browser_validation_gate" in node for node in required)
