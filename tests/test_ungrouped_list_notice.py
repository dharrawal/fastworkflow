"""The notice for an output that would be a table but declares none (fix-weih follow-up).

A ``Signature.Output`` with two or more visible list-of-scalar fields in no ``ui.table`` group gets a notice from
``fastworkflow command preview --check``. It is never an error: not among ``load_problems``, never an exit status.
A field's name decides nothing."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from fastworkflow import cli, command_preview, planning_commands

EXAMPLES = Path(__file__).resolve().parent.parent / "fastworkflow" / "examples"

COMMAND = '''"""{name}: a test command."""
import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Say it."""

    class Input(BaseModel):
        branch: str = Field(default="", description="a branch")

    class Output(BaseModel):
{fields}

    plain_utterances = [
        "show {name} one", "show {name} two", "show {name} three", "show {name} four",
        "show {name} five", "show {name} six", "show {name} seven", "show {name} eight",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: Signature.Input) -> Signature.Output:
        return Signature.Output()
'''

T = '        {n}: list[str] = Field(default_factory=list, json_schema_extra={{"ui": {{{ui}}}}})'
P = '        {n}: list[str] = Field(default_factory=list)'
CASES = {
    "loose_pair": [P.format(n="left"), P.format(n="right")],
    "grouped_pair": [T.format(n="left", ui='"table": "rows"'), T.format(n="right", ui='"table": "rows"')],
    "single_list": [P.format(n="only"), "        count: int = 0"],
    "hidden_pair": [T.format(n="left", ui='"hidden": True'), T.format(n="right", ui='"hidden": True')],
    "one_hidden_one_shown": [T.format(n="left", ui='"hidden": True'), P.format(n="right")],
    "record_lists": ["        rows: list[dict[str, str]] = Field(default_factory=list)",
                     "        more: list[dict[str, str]] = Field(default_factory=list)"],
    "total_named": [P.format(n="total_a"), P.format(n="total_b"), "        total: int = 0"],
}


@pytest.fixture
def library_with_cases(tmp_path):
    folder = tmp_path / "library"
    shutil.copytree(EXAMPLES / "library", folder, ignore=shutil.ignore_patterns("__pycache__", "___*"))
    for name, lines in CASES.items():
        (folder / "_commands" / f"{name}.py").write_text(
            COMMAND.format(name=name, fields="\n".join(lines)), encoding="utf-8")
    return str(folder)


def _run_cli(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["fastworkflow", *argv])
    try:
        cli.main()
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return 0


def test_the_notice_names_the_command_and_the_loose_fields_and_gives_the_table_hint(library_with_cases):
    notices = command_preview.check_ungrouped_list_notices(library_with_cases)

    loose = [n for n in notices if n.startswith("notice: loose_pair ")]
    assert len(loose) == 1
    assert "'left', 'right'" in loose[0]
    assert '{"ui": {"table": "<group name>"}}' in loose[0]
    assert "separate lists" in loose[0]


def test_a_field_name_decides_nothing_a_total_named_pair_is_noticed_like_any_other(library_with_cases):
    notices = command_preview.check_ungrouped_list_notices(library_with_cases)
    assert [n for n in notices if n.startswith("notice: total_named ")] == [
        n for n in notices if "'total_a', 'total_b'" in n]
    assert len([n for n in notices if "'total_a', 'total_b'" in n]) == 1


@pytest.mark.parametrize("name", ["grouped_pair", "single_list", "hidden_pair", "one_hidden_one_shown",
                                  "record_lists"])
def test_no_notice_for_a_declared_group_a_single_list_hidden_lists_or_lists_of_records(library_with_cases, name):
    notices = command_preview.check_ungrouped_list_notices(library_with_cases)
    assert not [n for n in notices if f"notice: {name} " in n]


def test_a_notice_is_not_a_load_problem_and_leaves_check_at_exit_zero(library_with_cases, tmp_path,
                                                                     monkeypatch, capsys):
    assert not [p for p in command_preview.load_problems(library_with_cases) if "loose_pair" in p]
    out = str(tmp_path / "previews")
    assert _run_cli(monkeypatch, "command", "preview", library_with_cases, "--out", out) == 0
    capsys.readouterr()

    assert _run_cli(monkeypatch, "command", "preview", library_with_cases, "--check", "--out", out) == 0
    printed = capsys.readouterr().out
    assert printed.count("notice: loose_pair ") == 1
    assert "notice: grouped_pair" not in printed
    assert "are current" in printed


def test_the_notice_function_never_raises_on_a_folder_with_no_workflow(tmp_path):
    assert command_preview.check_ungrouped_list_notices(str(tmp_path / "missing")) == []
    assert planning_commands.ungrouped_list_notices(type("NoFields", (), {"model_fields": {}}), "x") == []


@pytest.mark.parametrize("example", ["library", "hello_world", "retail_workflow", "messaging_app_4",
                                     "simple_workflow_template", "extended_workflow_example"])
def test_the_bundled_examples_produce_no_notice(example):
    assert command_preview.check_ungrouped_list_notices(str(EXAMPLES / example)) == []


# fix-bh1e.16: a command with more utterances than the matcher shows gets a notice.

def _library_with_utterance_counts(tmp_path, counts):
    folder = tmp_path / "library"
    shutil.copytree(EXAMPLES / "library", folder, ignore=shutil.ignore_patterns("__pycache__", "___*"))
    for name, count in counts.items():
        utterances = ",\n        ".join(f'"show {name} number {i}"' for i in range(count))
        (folder / "_commands" / f"{name}.py").write_text(
            COMMAND.format(name=name, fields="        left: list[str] = Field(default_factory=list, description='l')")
            .split("    plain_utterances")[0]
            + f"    plain_utterances = [\n        {utterances},\n    ]\n\n\nclass ResponseGenerator:\n"
            "    def __call__(self, workflow: fastworkflow.Workflow, command: str,\n"
            "                 input: Signature.Input) -> Signature.Output:\n"
            "        return Signature.Output(left=[])\n",
            encoding="utf-8")
    return str(folder)


def test_a_command_with_more_utterances_than_the_matcher_shows_is_noticed(tmp_path, monkeypatch, capsys):
    from fastworkflow.workflow_agent import MATCHER_EXAMPLES

    folder = _library_with_utterance_counts(
        tmp_path, {"many_phrases": MATCHER_EXAMPLES + 1, "few_phrases": MATCHER_EXAMPLES})

    notices = command_preview.check_matcher_example_notices(folder)

    many = [n for n in notices if n.startswith("notice: many_phrases ")]
    assert len(many) == 1
    assert f"first {MATCHER_EXAMPLES} of {MATCHER_EXAMPLES + 1}" in many[0]
    assert not [n for n in notices if "few_phrases" in n]

    out = str(tmp_path / "previews")
    assert _run_cli(monkeypatch, "command", "preview", folder, "--out", out) == 0
    capsys.readouterr()
    assert _run_cli(monkeypatch, "command", "preview", folder, "--check", "--out", out) == 0
    assert capsys.readouterr().out.count("notice: many_phrases ") == 1
