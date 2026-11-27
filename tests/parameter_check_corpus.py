"""Commands, ``Input`` models and the runner of the argument-guard equivalence table (bd fix-qoyz step 3).

``tests/fixtures/parameter_check_equivalence.json`` holds a table of cases (inputs and the outputs the two guard
entry points gave) captured from the code BEFORE the guards were merged into ``fastworkflow/parameters/check.py``.
This module holds what a case needs to run again: the planning commands the cases name (written into a temporary
copy of ``tests/planning_tutorial_workflow``), the execution ``Input`` classes, and ``run_case``, which calls the two
old public entry points (``planning_commands.check_arguments`` / ``checked_arguments`` / ``argument_gaps`` and
``argument_checks.check_extracted_values``) and reduces every outcome to plain JSON.

It is self-contained on purpose: it does not read other test modules, so a later edit of those cannot move the table.
"""
from __future__ import annotations

import dataclasses
import re
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"
ACCOUNTS = (re.compile(r"\bacct_\d+\b"),)

TRANSFER_LOAN = '''
from typing import Literal
from pydantic import BaseModel, Field


class Signature:
    """Move one member's loan of a book to a branch."""

    class Input(BaseModel):
        member: str = Field(description="the member whose loan moves")
        branch: str = Field(description="the branch the loan moves to")
        entity_type: Literal["identity", "account"] = Field(
            default="identity", description="the kind of record the request names")
        note: str = Field(default="", description="a note on the move")


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.member}", f"move the loan to the {input.branch} branch"]
'''


def _command(name: str, doc: str, fields: str, step: str) -> str:
    return f'''
from typing import Literal

from pydantic import BaseModel, Field


class Signature:
    """{doc}"""

    class Input(BaseModel):
{fields}


class PlanGenerator:
    def steps(self, input):
        return [{step}]
'''


#: Planning commands of the table, written into the temporary copy. Each name is a case's ``command``.
PLAN_COMMANDS = {
    "drift_rows": _command("drift_rows", "List rows.", '''        title: str = Field(default="", description="the book's title", json_schema_extra={"copied_from_command_text": True})
        count: int = Field(default=0, description="how many rows", json_schema_extra={"copied_from_command_text": True})''',
                           'f"list {input.count} rows of {input.title}"'),
    "drift_page": _command("drift_page", "Show a page.", '''        size: int = Field(default=5, description="rows on the page", json_schema_extra={"copied_from_command_text": True})''',
                           'f"show {input.size} loans"'),
    "drift_code": _command("drift_code", "Label a shelf.", '''        name: str = Field(description="a person's name", json_schema_extra={"copied_from_command_text": True})
        code: str = Field(default="", description="a shelf code", examples=["XY-12"], json_schema_extra={"copied_from_command_text": True})''',
                           'f"label {input.name} {input.code}"'),
    "drift_converted": _command("drift_converted", "Set a due date.", '''        name: str = Field(default="", description="a person's name", json_schema_extra={"copied_from_command_text": True})
        due: str = Field(default="", description="the due date, YYYY-MM-DD", examples=["2024-05-20"])''',
                           'f"set the due date of {input.name} to {input.due}"'),
    "drift_shelf": '''
from pydantic import BaseModel, Field


class Signature:
    """Move a loan to a shelf."""

    template = ["move the loan to the shelf"]

    class Input(BaseModel):
        shelf: str = Field(description="the shelf code", examples=["#S1"], pattern=r"^#S[0-9]+$")


class PlanGenerator:
    def steps(self, input):
        return [f"move the loan to shelf {input.shelf}"]
''',
    "drift_choice": _command("drift_choice", "Find a record of one kind.", '''        kind: Literal["book", "member", "branch"] = Field(default="book", description="the kind of record")
        kind_prose: str = Field(default="book", description="the kind of record. One of: book, member, branch")''',
                           'f"find a {input.kind}"'),
    "drift_subject": _command("drift_subject", "Show one record's books.", '''        name: str = Field(description="the record's name", json_schema_extra={"record_named_in_command_text": True})''',
                           'f"show the books of {input.name}"'),
    "drift_verify": _command("drift_verify", "Find a record of the kind meant.", '''        name: str = Field(default="", description="the name to look for", json_schema_extra={"copied_from_command_text": True})
        kind: Literal["book", "member", "branch"] = Field(default="branch", description="the kind of record", json_schema_extra={"classifier_confirms": True})''',
                           'f"find the {input.kind} {input.name}"'),
    # shapes the drift commands lack: every type of a planning Param, with and without the options
    "shape_all": _command("shape_all", "Every shape of argument.", '''        label: str = Field(description="a label", json_schema_extra={"copied_from_command_text": {"ignore": ["the", "a"]}})
        ratio: float = Field(default=1.5, description="a ratio")
        flag: bool = Field(default=False, description="a flag")
        flag_on: bool = Field(default=True, description="a flag that defaults on")
        limit: int = Field(default=10, description="a limit")
        size: Literal["small", "medium", "large"] = Field(default="small", description="a size")
        colour: str = Field(default="", description="a colour. One of: red, Green, blue")
        req: str = Field(description="a required plain text")
        note: str = Field(default="", description="a note")''',
                           'f"do {input.label} {input.req}"'),
    "shape_lists": _command("shape_lists", "Lists of text.", '''        tags: list[str] = Field(default_factory=list, description="tags", json_schema_extra={"copied_from_command_text": True})
        plain_tags: list[str] = Field(default_factory=list, description="more tags")
        need: list[str] = Field(description="a required list", json_schema_extra={"copied_from_command_text": True})
        ratio: float = Field(default=1.5, description="a ratio")''',
                           'f"tag {input.tags} {input.need}"'),
    "shape_required_enum": _command("shape_required_enum", "A required choice and a required number.", '''        size: Literal["small", "large"] = Field(description="a size")
        qty: int = Field(description="a quantity", json_schema_extra={"copied_from_command_text": True})
        amount: float = Field(description="an amount")''',
                           'f"order {input.qty} {input.size}"'),
    "shape_subjects": _command("shape_subjects", "Two subjects and a plain text.", '''        who: str = Field(description="who", json_schema_extra={"record_named_in_command_text": {"ignore": ["the"]}})
        other: str = Field(default="", description="another", json_schema_extra={"record_named_in_command_text": True})
        text: str = Field(default="", description="free text")''',
                           'f"link {input.who} {input.other}"'),
    "shape_default_str": _command("shape_default_str", "A copied text with a real default.", '''        scope: str = Field(default="all", description="the scope", json_schema_extra={"copied_from_command_text": True})
        needed: str = Field(description="a required copied text", json_schema_extra={"copied_from_command_text": True})''',
                           'f"list {input.scope} {input.needed}"'),
    "shape_verify_required": _command("shape_verify_required", "A required record the classifier checks.", '''        subject: str = Field(description="a record", json_schema_extra={"record_named_in_command_text": True, "classifier_confirms": True})
        mode: Literal["a", "b"] = Field(default="a", description="a mode")''',
                           'f"scan {input.subject} {input.mode}"'),
    "shape_verify": _command("shape_verify", "Classifier options of each kind.", '''        subject: str = Field(default="", description="a record", json_schema_extra={"record_named_in_command_text": True, "classifier_confirms": True})
        mode: Literal["a", "b", "c"] = Field(description="a mode", json_schema_extra={"classifier_confirms": True})
        deep: bool = Field(default=False, description="deep", json_schema_extra={"classifier_decides": True})
        on: bool = Field(default=False, description="on", json_schema_extra={"classifier_confirms": True})''',
                           'f"scan {input.subject} {input.mode}"'),
}


class ExecStr(BaseModel):
    """An execution ``Input`` with every ``str`` shape the execution guard reads."""

    name: str = Field(description="a person's name", json_schema_extra={"copied_from_command_text": True})
    nick: Optional[str] = Field(default=None, description="a nickname", json_schema_extra={"copied_from_command_text": True})
    code: str = Field(default="", description="a code", examples=["XY-12"], json_schema_extra={"copied_from_command_text": True})
    conv: str = Field(default="", description="an unmarked value", examples=["2024-05-20"])
    shelf: str = Field(default="", description="a shelf", examples=["#S1"], pattern=r"^#S[0-9]+$")
    enum_s: Literal["book", "Member", "branch"] = Field(default="book", description="a kind")
    one_of: str = Field(default="red", description="a colour. One of: red, Green, blue")
    prose: str = Field(default="", description="a text. One of the many things; not a list")


class ExecNum(BaseModel):
    """An execution ``Input`` with the numbers, and the types the guard leaves alone."""

    count: int = Field(default=0, description="how many", json_schema_extra={"copied_from_command_text": True})
    size: int = Field(default=5, description="a size", json_schema_extra={"copied_from_command_text": True})
    opt_count: Optional[int] = Field(default=None, description="an optional count")
    conv_count: int = Field(default=0, description="an unmarked count")
    title: str = Field(default="", description="a title", json_schema_extra={"copied_from_command_text": True})
    ratio: float = Field(default=1.0, description="a ratio")
    flag: bool = Field(default=False, description="a flag")
    tags: list[str] = Field(default_factory=list, description="tags")
    kind: Literal["book", "member"] = Field(default="book", description="a kind")


class ExecOptions(BaseModel):
    """An execution ``Input`` whose fields carry check options (``validate_parameters`` runs them through
    ``check_arguments``)."""

    copied: str = Field(default="", description="copied", json_schema_extra={"copied_from_command_text": {"ignore": ["the"]}})
    named: str = Field(default="", description="a record", json_schema_extra={"record_named_in_command_text": True})
    qty: int = Field(default=0, description="a quantity", json_schema_extra={"copied_from_command_text": True})
    items: list[str] = Field(default_factory=list, description="items", json_schema_extra={"copied_from_command_text": True})
    plain: str = Field(default="", description="plain")


class ChoiceText(BaseModel):
    kind_text: Literal["book", "member", "branch"] = Field(default="book", description="the kind of record")


EXEC_MODELS = {"ExecStr": ExecStr, "ExecNum": ExecNum, "ExecOptions": ExecOptions, "ChoiceText": ChoiceText}


def build_commands(directory: Path) -> dict:
    """The tutorial's planning commands plus this table's, loaded from a copy under *directory*."""
    from fastworkflow.agent import workflow_agent

    copy = directory / "tutorial"
    shutil.copytree(TUTORIAL, copy, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (copy / "_commands" / "transfer_loan.py").write_text(TRANSFER_LOAN, encoding="utf-8")
    for name, text in PLAN_COMMANDS.items():
        (copy / "_commands" / f"{name}.py").write_text(text, encoding="utf-8")
    loaded = workflow_agent.load_planning_command_map(str(copy))
    return {name: dataclasses.replace(command, identifiers=ACCOUNTS) for name, command in loaded.items()}


# ---------------------------------------------------------------- outcomes as JSON

def norm(value: Any) -> Any:
    """A JSON-safe form that keeps the type distinctions the guards make (1 / True / 1.0 / "1")."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else repr(value)
    if isinstance(value, (list, tuple)):
        return [norm(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((norm(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {str(k): norm(v) for k, v in value.items()}
    return f"<{type(value).__name__}> {value!r}"


def _error(exc: BaseException) -> dict:
    gaps = getattr(exc, "gaps", None)
    return {"type": type(exc).__name__, "message": str(exc), "fields": norm(getattr(exc, "fields", None)),
            "gaps": None if gaps is None else _extraction(gaps)}


def _extraction(extraction) -> dict:
    return {"values": norm(extraction.values), "missing": norm(extraction.missing),
            "invalid": norm(extraction.invalid), "earlier": norm(extraction.earlier), "reason": extraction.reason,
            "names": norm(extraction.names)}


def _verdicts(raw: Optional[dict]) -> Optional[dict]:
    return None if raw is None else {name: (v[0], v[1]) for name, v in raw.items()}


def run_plan(command, case: dict) -> dict:
    """``check_arguments``, ``checked_arguments`` and ``argument_gaps`` of one planning-command case."""
    from fastworkflow.planning import planning_commands as pc

    raw, request, verdicts = dict(case["raw"]), case["request"], _verdicts(case.get("verdicts"))
    out: dict = {}
    for label, call in (
            ("check_arguments", lambda: pc.check_arguments(command, dict(raw), request, verdicts)),
            ("checked_arguments", lambda: pc.checked_arguments(command, dict(raw), request, None, verdicts))):
        try:
            args, notes = call()
            out[label] = {"ok": norm(args), "notes": norm(notes)}
        except (pc.PlanningCommandCallError, ValueError) as refused:
            out[label] = {"error": _error(refused)}
    out["argument_gaps"] = _extraction(pc.argument_gaps(command, dict(raw), request, verdicts))
    return out


def run_exec(case: dict, commands: dict | None = None) -> dict:
    """``check_extracted_values`` and ``ExtractionCheck.apply`` of one execution-command case (on one of
    ``EXEC_MODELS``, or, for kind ``exec_input``, on the ``Input`` of a planning command of the table)."""
    from fastworkflow.parameters import argument_checks

    model = commands[case["command"]].input_model if case["kind"] == "exec_input" else EXEC_MODELS[case["model"]]
    params = model.model_construct(**case["values"])
    check = argument_checks.check_extracted_values(params, case["request"])
    out = {"refused": {n: [repr(v), why] for n, (v, why) in check.refused.items()},
           "respelled": {n: [repr(v), repr(s)] for n, (v, s) in check.respelled.items()},
           "refused_order": list(check.refused), "respelled_order": list(check.respelled),
           "texts": check.refusal_texts(), "note": check.note()}
    check.apply(params)
    out["applied"] = {name: norm(getattr(params, name, None)) for name in model.model_fields}
    return out


def run_option(case: dict) -> dict:
    """``check_arguments`` on the command ``option_command`` builds from an execution ``Input`` (what
    ``validate_parameters`` runs for the fields that carry options)."""
    from fastworkflow.planning import planning_commands as pc

    model = EXEC_MODELS[case["model"]]
    command = pc.option_command(model, "table_command", ACCOUNTS)
    sub = dict(case, raw={p.name: case["raw"].get(p.name) for p in command.params})
    return run_plan(command, sub)


def run_case(case: dict, commands: dict) -> dict:
    kind = case["kind"]
    if kind == "plan":
        return run_plan(commands[case["command"]], case)
    if kind in ("exec", "exec_input"):
        return run_exec(case, commands)
    return run_option(case)
