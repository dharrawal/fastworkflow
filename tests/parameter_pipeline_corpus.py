"""Commands, cases and the runner of the parameter-pipeline equivalence table (bd fix-qoyz step 5).

``tests/fixtures/parameter_pipeline_equivalence.json`` holds a table of cases captured from the code BEFORE the
fill / check / validate / retry sequence was written once (``fastworkflow/parameters/fill.py`` and ``pipeline.py``):
for every call site that ran it (``ParameterExtraction._extract_impl`` and ``_llm_extract``, ``extract_arguments``,
``planning_commands.extracted_arguments`` / ``checked_arguments`` / ``typed_planning_call`` /
``reextracted_arguments``, and the four code forms that read an agent's own command text) the inputs, EVERY TEXT SENT
TO THE (SCRIPTED) MODEL IN ORDER, and the result, diagnostics and exception.

The model is a scripted ``DummyLM`` that answers each call from a per-case list, so the number and the order of
model calls, and the text of each (the note a retry appends, byte for byte), are part of what a case pins. Nothing
here calls a model.

Self-contained on purpose: it reads no other test module, so an edit of those cannot move the table. The old
public entry points are the only things it calls, so it runs against code from before and after the change.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import shutil
import types
import uuid
from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"
ACCOUNTS = (re.compile(r"\bacct_\d+\b"),)
NF = "NOT_FOUND"

# ---------------------------------------------------------------- the commands of the table (written into a temp copy)

_EXEC = '''
"""{name}: an execution command of the pipeline table."""
from typing import List, Literal, Optional

import fastworkflow
from pydantic import BaseModel, Field, model_validator


class Signature:
    """{doc}"""

    class Input(BaseModel):
{fields}
{validators}

    class Output(BaseModel):
        done: bool = Field(description="whether it was done")

    plain_utterances = ["{name} please"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(done=True)
'''

EXEC_COMMANDS = {
    "lend_book": _EXEC.format(name="lend_book", doc="Lend one book to a member.", validators="", fields='''
        member: str = Field(description="the member who borrows", examples=["Jane Roe"], json_schema_extra={"copied_from_command_text": True})
        title: str = Field(description="the book's title", json_schema_extra={"copied_from_command_text": True})
        copies: int = Field(default=1, description="how many copies", json_schema_extra={"copied_from_command_text": True})
        tags: Optional[List[str]] = Field(default=None, description="labels for the loan")
        kind: Literal["loan", "hold"] = Field(default="loan", description="the kind of lending")
        note: Optional[str] = Field(default=None, description="a free note")'''),
    "shelf_code": _EXEC.format(name="shelf_code", doc="Label a shelf with a code.", fields='''
        code: str = Field(description="the shelf code", json_schema_extra={"copied_from_command_text": True})
        floor: int = Field(default=0, description="the floor")''', validators='''
        @model_validator(mode="after")
        def closed(self, info):
            if self.code.startswith("BAD"):
                raise fastworkflow.Refused("that shelf is closed", field="code")
            if self.code == "NOFIELD":
                raise fastworkflow.Refused("this does not fit the request")
            return self'''),
    "all_defaults": _EXEC.format(name="all_defaults", doc="Show the defaults.", validators="", fields='''
        page: int = Field(default=1, description="the page", json_schema_extra={"copied_from_command_text": True})
        label: Optional[str] = Field(default=None, description="a label")'''),
}

_PLAN = '''
"""{name}: a planning command of the pipeline table."""
from typing import List, Literal, Optional

import fastworkflow
from pydantic import BaseModel, Field, model_validator


class Signature:
    """{doc}"""

    class Input(BaseModel):
{fields}
{validators}


class PlanGenerator:
    def steps(self, input):
        return ["say the record"]
'''

PLAN_COMMANDS = {
    "ledger_pc": _PLAN.format(name="ledger_pc", doc="Write one ledger line.", fields='''
        owner: str = Field(description="the owner of the line", json_schema_extra={"copied_from_command_text": True})
        count: int = Field(default=0, description="how many", json_schema_extra={"copied_from_command_text": True})
        code: str = Field(default="", description="a code")''', validators='''
        @model_validator(mode="after")
        def closed(self, info):
            if self.code == "BAD":
                raise fastworkflow.Refused("that code is closed", field="code")
            if self.owner == "Nobody":
                raise fastworkflow.Refused("this does not fit the request")
            return self'''),
    "taglist_pc": _PLAN.format(name="taglist_pc", doc="Tag a file.", validators="", fields='''
        label: str = Field(description="the file", json_schema_extra={"copied_from_command_text": True})
        tags: list[str] = Field(description="the tags", json_schema_extra={"copied_from_command_text": True})'''),
    "empty_pc": _PLAN.format(name="empty_pc", doc="Say hello.", validators="", fields="        pass"),
}

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


def build_folder(directory: Path) -> str:
    """The tutorial workflow plus this table's commands, in a copy under *directory* (never the original)."""
    copy = directory / "pipeline_tutorial"
    shutil.copytree(TUTORIAL, copy, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    commands = copy / "_commands"
    (commands / "transfer_loan.py").write_text(TRANSFER_LOAN, encoding="utf-8")
    for name, text in {**EXEC_COMMANDS, **PLAN_COMMANDS}.items():
        (commands / f"{name}.py").write_text(text, encoding="utf-8")
    return str(copy)


# ---------------------------------------------------------------- Input classes of the four code forms

class FindParams(BaseModel):
    query: Optional[str] = Field(default=None)
    start: Optional[int] = Field(default=None)


class MarkedFindParams(BaseModel):
    query: Optional[str] = Field(default=None, json_schema_extra={"command_text_after_command_name": True})
    start: Optional[int] = Field(default=None)


class OpenParams(BaseModel):
    book_uid: str = Field(description="uid")


class OneStr(BaseModel):
    query: str = Field(description="the words")


class TwoStrings(BaseModel):
    book_uid: Optional[str] = Field(default=None)
    filter: Optional[str] = Field(default=None)


class MarkedTwoStrings(BaseModel):
    book_uid: Optional[str] = Field(default=None, json_schema_extra={"command_text_after_command_name": True})
    filter: Optional[str] = Field(default=None)


class ListParams(BaseModel):
    ids: List[str] = Field(default_factory=list)
    counts: Optional[List[int]] = Field(default=None)
    note: Optional[str] = Field(default=None)


class ExampleParams(BaseModel):
    query: str = Field(default="x", description="words", examples=["Jane Roe"])
    size: int = Field(default=3)


class NoParams(BaseModel):
    pass


class EnumParams(BaseModel):
    kind: Literal["loan", "hold"] = Field(default="loan")
    note: Optional[str] = Field(default=None)


FORM_CLASSES = {c.__name__: c for c in (FindParams, MarkedFindParams, OpenParams, OneStr, TwoStrings,
                                        MarkedTwoStrings, ListParams, ExampleParams, NoParams, EnumParams)}


# ---------------------------------------------------------------- outcomes as JSON

def norm(value: Any) -> Any:
    """A JSON-safe form that keeps the type distinctions the pipeline makes (1 / True / 1.0 / "1")."""
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
    if hasattr(value, "value") and hasattr(type(value), "__members__"):
        return f"<enum> {value.value!r}"
    return f"<{type(value).__name__}> {value!r}"


def dump(model: Any) -> Any:
    """Every field of a (possibly ``model_construct``-ed) model, as ``norm`` forms; None for None."""
    if model is None:
        return None
    return {name: norm(getattr(model, name, None)) for name in type(model).model_fields}


def error(exc: BaseException) -> dict:
    gaps = getattr(exc, "gaps", None)
    return {"type": type(exc).__name__, "message": str(exc), "fields": norm(getattr(exc, "fields", None)),
            "gaps": None if gaps is None else extraction(gaps)}


def extraction(value) -> dict:
    return {"values": norm(value.values), "missing": norm(value.missing), "invalid": norm(value.invalid),
            "earlier": norm(value.earlier), "reason": value.reason, "names": norm(value.names)}


_DATE = re.compile(r"Today's date is \d{4}-\d\d-\d\d")


def _prompt_text(prompt, messages) -> str:
    text = "\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt)
    return _DATE.sub("Today's date is DATE", text)


# ---------------------------------------------------------------- the scripted model

class Environment:
    """The framework state a case runs in and the scripted models: ``begin(case)`` arms them, ``prompts`` is every text
    the scripted extractor was sent in order, ``planner_prompts`` the planner's."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.folder = build_folder(directory)
        self.prompts: list[str] = []
        self.planner_prompts: list[str] = []
        self.lm = None
        self.lm_requests = 0
        self.raise_on: Optional[int] = None
        self.resets: list = []
        self.reset_statements: list[str] = []
        self.records_resets = False
        self._undo: list = []

    def __enter__(self) -> "Environment":
        import dspy
        from dspy.utils import DummyLM

        import fastworkflow
        from fastworkflow.agent import workflow_agent
        from fastworkflow.parameters import signatures

        fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(self.directory / "workflow_contexts")})
        fastworkflow.RoutingRegistry.clear_registry()
        saved_env = {name: fastworkflow._env_vars.get(name) for name in
                     ("JEV_API_KEY", "FW_JEV_FIELD_GIVEN", "FW_EXTRACTION_ARGUMENT_CHECKS")}
        saved_os = os.environ.pop("JEV_API_KEY", None)
        fastworkflow._env_vars.pop("JEV_API_KEY", None)
        fastworkflow._env_vars["FW_JEV_FIELD_GIVEN"] = "off"
        fastworkflow._env_vars["FW_EXTRACTION_ARGUMENT_CHECKS"] = "on"
        getattr(workflow_agent._jev_lm, "cache_clear", lambda: None)()
        saved_get_lm = signatures.dspy_utils.get_lm
        env = self

        class Scripted(DummyLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                env.prompts.append(_prompt_text(prompt, messages))
                return super().forward(prompt=prompt, messages=messages, **kwargs)

        class Planner(DummyLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                env.planner_prompts.append(_prompt_text(prompt, messages))
                return super().forward(prompt=prompt, messages=messages, **kwargs)

        self._Scripted, self._Planner = Scripted, Planner

        def get_lm(*args, **kwargs):
            env.lm_requests += 1
            if env.raise_on == env.lm_requests:
                raise RuntimeError("scripted model outage")
            return env.lm

        saved_fields_not_given = signatures.fields_not_given

        def fields_not_given(statement, model_class, values):
            # stands in for Jev's "is this field given" check (a model call): the case says what it answers, per call
            env.reset_statements.append(_DATE.sub("Today's date is DATE", statement))
            return dict(env.resets.pop(0)) if env.resets else {}

        signatures.dspy_utils.get_lm = get_lm
        signatures.fields_not_given = fields_not_given
        self._undo = [
            lambda: setattr(signatures, "fields_not_given", saved_fields_not_given),
            lambda: setattr(signatures.dspy_utils, "get_lm", saved_get_lm),
            lambda: [fastworkflow._env_vars.__setitem__(n, v) if v is not None else fastworkflow._env_vars.pop(n, None)
                     for n, v in saved_env.items()],
            lambda: os.environ.__setitem__("JEV_API_KEY", saved_os) if saved_os is not None else None,
            lambda: fastworkflow.RoutingRegistry.clear_registry(),
        ]
        self._dspy = dspy
        return self

    def __exit__(self, *exc) -> None:
        for undo in reversed(self._undo):
            undo()

    def begin(self, case: dict) -> None:
        """Arm the scripted models for *case* and clear what was recorded."""
        import dspy

        import fastworkflow

        self.prompts, self.planner_prompts, self.lm_requests = [], [], 0
        self.raise_on = case.get("raise_on")
        self.resets, self.reset_statements = list(case.get("resets", [])), []
        self.records_resets = "resets" in case
        self.lm = self._Scripted(list(case.get("answers", [])), adapter=dspy.JSONAdapter())
        self.planner = self._Planner(list(case.get("planner", [])))
        fastworkflow._env_vars["FW_EXTRACTION_ARGUMENT_CHECKS"] = case.get("checks", "on")

    def workflow(self, case: dict):
        import fastworkflow

        fastworkflow.RoutingRegistry.get_definition(self.folder)
        workflow = fastworkflow.Workflow.create(self.folder, workflow_id_str=f"pipe_{uuid.uuid4().hex[:8]}")
        mode = case.get("mode", "assist")
        if mode in ("agent", "agent_code", "slash"):
            workflow.context["run_as_agent"] = True
        if mode == "agent_code":
            workflow.context["by_code"] = True
        return workflow

    def command(self, name: str):
        from fastworkflow.agent import workflow_agent

        found = workflow_agent.load_planning_command_map(self.folder)[name]
        return dataclasses.replace(found, identifiers=ACCOUNTS)


@contextlib.contextmanager
def _logged():
    """Keep the logger quiet and count nothing: a case's record is its outputs, not its log lines."""
    yield


def _record(env: Environment, out: dict) -> dict:
    out["model_calls"] = len(env.prompts)
    out["prompts"] = list(env.prompts)
    if env.planner_prompts:
        out["planner_prompts"] = list(env.planner_prompts)
    if env.records_resets:
        out["reset_statements"] = list(env.reset_statements)
    return out


# ---------------------------------------------------------------- the runners, one per call site

def _output(output) -> dict:
    return {"valid": output.parameters_are_valid, "params": dump(output.cmd_parameters),
            "error_msg": output.error_msg, "suggestions": norm(output.suggestions),
            "parameter_request": norm(output.parameter_request)}


def _observed_context(context: dict) -> dict:
    """The CME context's keys, and the parameter loop of the turn's waiting record (the pending question) as the loop's
    own state, under ``parameter_loop``."""
    from fastworkflow.parameters import parameter_clarification
    from fastworkflow.core import waiting

    shown = {key: value for key, value in context.items() if key != waiting.PORT_KEY}
    if (loop := parameter_clarification.pending(context)) is not None:
        shown["parameter_loop"] = loop.to_state()
    return shown


def run_impl(env: Environment, case: dict) -> dict:
    """``ParameterExtraction._extract_impl``: the deterministic path of an execution command (the code forms of an
    agent's command text, then the LLM extractor with its checks and its one retry, then validation)."""
    from fastworkflow._workflows.command_metadata_extraction import parameter_extraction as pe

    workflow = env.workflow(case)
    cme = types.SimpleNamespace(context={"is_assistant_mode_command": True} if case.get("mode") == "slash" else {})
    name = case["command"]
    if pending := case.get("pending"):
        params_class = pe.parameters_class_of(workflow.folderpath, name)
        pe.park(cme, name, params_class.model_construct(**pending), question="Q?", request_text=case["text"])
    diagnostics: dict = {}
    out: dict = {}
    try:
        out["output"] = _output(pe.ParameterExtraction(cme, workflow, name, case["text"])._extract_impl(diagnostics))
    except BaseException as exc:  # noqa: BLE001 - an exception is part of the record
        out["exception"] = error(exc)
    out["diagnostics"] = norm(diagnostics)
    out["cme_context"] = norm(_observed_context(cme.context))
    return _record(env, out)


def run_llm(env: Environment, case: dict) -> dict:
    """``ParameterExtraction._llm_extract`` on its own."""
    from fastworkflow._workflows.command_metadata_extraction import parameter_extraction as pe
    from fastworkflow.parameters.signatures import InputForParamExtraction

    workflow = env.workflow(case)
    name, text = case["command"], case["text"]
    params_class = pe.parameters_class_of(workflow.folderpath, name)
    extractor = InputForParamExtraction.create(workflow, name, text)
    diagnostics: dict = {}
    out: dict = {}
    try:
        params = pe.ParameterExtraction(None, workflow, name, text)._llm_extract(
            extractor, params_class, workflow.folderpath, diagnostics)
        out["params"] = dump(params)
    except BaseException as exc:  # noqa: BLE001
        out["exception"] = error(exc)
    out["diagnostics"] = norm(diagnostics)
    out["extractor_command"] = extractor.command
    return _record(env, out)


def run_args(env: Environment, case: dict) -> dict:
    """``parameter_extraction.extract_arguments``: what the plan draft and the arguments of a planning command read."""
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import extract_arguments

    workflow = env.workflow(case)
    diagnostics: dict = {}
    out: dict = {}
    try:
        result = extract_arguments(workflow, case["command"], case["text"], diagnostics)
        out["result"] = {"params": dump(result.params), "extraction": extraction(result.extraction),
                         "retried": result.retried, "valid": result.valid, "refusal": result.refusal,
                         "missing": norm(result.missing)}
    except BaseException as exc:  # noqa: BLE001
        out["exception"] = error(exc)
    out["diagnostics"] = norm(diagnostics)
    return _record(env, out)


def run_typed(env: Environment, case: dict) -> dict:
    """``planning_commands.typed_planning_call`` with the planner scripted (the typed fill and the step proposal)."""
    import dspy

    from fastworkflow.planning import planning_commands as pc

    workflow = env.workflow(case)
    command = env.command(case["command"])
    out: dict = {}
    try:
        with dspy.context(lm=env.planner):
            call = pc.typed_planning_call(command, case["text"], "", None, workflow)
        out["call"] = {"plan": call.plan, "args": norm(call.args), "notes": norm(call.notes),
                       "steps_before": norm(call.steps_before), "steps_after": norm(call.steps_after),
                       "verdicts": norm(call.verdicts), "phases": len(call.phases or [])}
    except BaseException as exc:  # noqa: BLE001
        out["exception"] = error(exc)
    return _record(env, out)


def run_extracted(env: Environment, case: dict) -> dict:
    """``planning_commands.extracted_arguments`` (the shared extractor's arguments, unchecked) and, when the case asks,
    ``checked_arguments`` on them."""
    from fastworkflow.planning import planning_commands as pc

    workflow = env.workflow(case)
    command = env.command(case["command"])
    out: dict = {}
    try:
        raw = pc.extracted_arguments(command, case["text"], workflow)
        out["raw"] = norm(raw)
        if case.get("check"):
            try:
                args, notes = pc.checked_arguments(command, raw, case["text"], workflow)
                out["checked"] = {"args": norm(args), "notes": norm(notes)}
            except pc.PlanningCommandCallError as refused:
                out["checked_error"] = error(refused)
    except BaseException as exc:  # noqa: BLE001
        out["exception"] = error(exc)
    return _record(env, out)


def run_reextract(env: Environment, case: dict) -> dict:
    """``planning_commands.reextracted_arguments``: the arguments read again after a first phase found nothing."""
    from fastworkflow.planning import planning_commands as pc

    workflow = env.workflow(case)
    command = env.command(case["command"])
    out: dict = {}
    try:
        out["args"] = norm(pc.reextracted_arguments(
            command, case["text"], case["observation"], dict(case["args"]), workflow, case.get("verdicts")))
    except BaseException as exc:  # noqa: BLE001
        out["exception"] = error(exc)
    return _record(env, out)


def run_form(env: Environment, case: dict) -> dict:
    """One of the four code forms that read an agent's own command text, called as ``ParameterExtraction`` has always
    exposed it, on a class of the table."""
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction

    cls = FORM_CLASSES[case["cls"]]
    form, text, name = case["form"], case["text"], case.get("command", "Shelf/find_book")
    try:
        if form == "xml":
            result = ParameterExtraction._extract_parameters_from_xml(text, cls)
        elif form == "key_value":
            result = ParameterExtraction._extract_parameters_from_key_values(text, cls, name)
        elif form == "key_value_code":
            result = ParameterExtraction._extract_parameters_from_key_values(text, cls, name, code_written=True)
        elif form == "positional":
            result = ParameterExtraction._extract_parameters_from_positional(text, cls, name)
        else:
            result = ParameterExtraction._extract_parameters_from_bare(text, cls, name)
        out = {"params": dump(result), "fields_set": None if result is None else sorted(result.model_fields_set)}
    except BaseException as exc:  # noqa: BLE001
        out = {"exception": error(exc)}
    return _record(env, out)


RUNNERS = {"impl": run_impl, "llm": run_llm, "args": run_args, "typed": run_typed, "extracted": run_extracted,
           "reextract": run_reextract, "form": run_form}


def run_case(env: Environment, case: dict) -> dict:
    env.begin(case)
    return RUNNERS[case["site"]](env, case)


# ---------------------------------------------------------------- the cases

def A(**values) -> dict:
    """One scripted extractor answer."""
    return {"reasoning": "scripted", **values}


def LEND(member="Ann Lee", title="Dune", copies=1, tags=None, kind="loan", note=None) -> dict:
    return A(member=member, title=title, copies=copies, tags=tags, kind=kind, note=note)


def LEDGER(owner="Ann Lee", count=0, code="") -> dict:
    return A(owner=owner, count=count, code=code)


def TAGS(label="Atlas", tags=("draft", "final")) -> dict:
    return A(label=label, tags=list(tags))


def PLANNER(**fill) -> dict:
    """One scripted planner answer: the typed fill's values (when the fallback asks) and an empty step proposal."""
    return {"reasoning": "scripted", **fill, "steps_before": [], "steps_after": []}


def _cases() -> list[dict]:
    cases: list[dict] = []

    def add(case_id: str, site: str, **fields) -> None:
        cases.append({"id": case_id, "site": site, **fields})

    # ---- ParameterExtraction._extract_impl, assistant mode (the LLM path with its guard and its one retry)
    FM, FMT = "find_member", "find the member Ann Lee"
    add("impl-fm-clean", "impl", command=FM, text=FMT, answers=[A(query="Ann Lee")])
    add("impl-fm-guard-then-clean", "impl", command=FM, text=FMT, answers=[A(query="Bob Jones"), A(query="Ann Lee")])
    add("impl-fm-guard-twice", "impl", command=FM, text=FMT, answers=[A(query="Bob Jones"), A(query="Bob Jones")])
    add("impl-fm-guard-then-other-invented", "impl", command=FM, text=FMT,
        answers=[A(query="Bob Jones"), A(query="Carl Smith")])
    add("impl-fm-guard-then-retry-raises", "impl", command=FM, text=FMT, answers=[A(query="Bob Jones")], raise_on=2)
    add("impl-fm-guard-then-retry-missing", "impl", command=FM, text=FMT, answers=[A(query="Bob Jones"), A(query=NF)])
    add("impl-fm-missing-no-retry", "impl", command=FM, text=FMT, answers=[A(query=NF)])
    add("impl-fm-missing-empty-no-retry", "impl", command=FM, text=FMT, answers=[A(query="")])
    add("impl-fm-placeholder-then-clean", "impl", command=FM, text=FMT, answers=[A(query="<name>"), A(query="Ann Lee")])
    add("impl-fm-punctuation", "impl", command=FM, text=FMT, answers=[A(query="..."), A(query="...")])
    add("impl-fm-respelled", "impl", command=FM, text=FMT, answers=[A(query="ann lee.")])
    add("impl-fm-checks-off-invented", "impl", command=FM, text=FMT, answers=[A(query="Bob Jones")], checks="off")
    add("impl-fm-prefix", "impl", command=FM, text="find_member: find the member Ann Lee", answers=[A(query="Ann Lee")])
    add("impl-fm-key-value-assist", "impl", command=FM, text="find_member query=Ann Lee", answers=[])
    add("impl-fm-key-value-assist-unknown-key", "impl", command=FM, text="find_member name=Ann Lee",
        answers=[A(query="Ann Lee")])
    add("impl-fm-in-text-twice-wrong", "impl", command=FM, text="look up Ann Lee or Bob Jones",
        answers=[A(query="Carl"), A(query="Bob Jones")])
    # lend_book: int, enum, optional, list, example values
    LB, LBT = "lend_book", "lend Dune to Ann Lee, 2 copies, tagged rare and old, as a hold"
    add("impl-lb-clean", "impl", command=LB, text=LBT,
        answers=[LEND(copies=2, tags=["rare", "old"], kind="hold")])
    add("impl-lb-int-not-written-then-clean", "impl", command=LB, text=LBT,
        answers=[LEND(copies=7, tags=["rare", "old"], kind="hold"), LEND(copies=2, tags=["rare", "old"], kind="hold")])
    add("impl-lb-int-not-written-twice", "impl", command=LB, text=LBT,
        answers=[LEND(copies=7, kind="hold"), LEND(copies=9, kind="hold")])
    add("impl-lb-example-member", "impl", command=LB, text=LBT, answers=[LEND(member="Jane Roe", copies=2)] * 4)
    add("impl-lb-missing-title-no-retry", "impl", command=LB, text="lend a book to Ann Lee",
        answers=[LEND(title=NF)])
    add("impl-lb-invented-and-missing", "impl", command=LB, text="lend to Ann Lee",
        answers=[LEND(member="Bob", title=NF), LEND(title=NF)])
    add("impl-lb-second-pass-invents-title", "impl", command=LB, text="lend to Ann Lee",
        answers=[LEND(member="Bob", title=NF), LEND(title="Ulysses")])
    add("impl-lb-second-pass-recovers-both", "impl", command=LB, text="lend Dune to Ann Lee",
        answers=[LEND(member="Bob", title=NF), LEND(title="Dune")])
    add("impl-lb-note-grounded", "impl", command=LB, text="lend Dune to Ann Lee, note: fragile",
        answers=[LEND(note="fragile")])
    add("impl-lb-note-ungrounded-then-clean", "impl", command=LB, text="lend Dune to Ann Lee",
        answers=[LEND(note="extra care"), LEND(note=None)])
    add("impl-lb-tags-empty", "impl", command=LB, text="lend Dune to Ann Lee", answers=[LEND(tags=[])])
    add("impl-lb-tags-items", "impl", command=LB, text="lend Dune to Ann Lee tagged rare",
        answers=[LEND(tags=["rare"])])
    add("impl-lb-kind-hold", "impl", command=LB, text="place a hold on Dune for Ann Lee", answers=[LEND(kind="hold")])
    add("impl-lb-kind-wrong-spelling", "impl", command=LB, text="place a hold on Dune for Ann Lee",
        answers=[LEND(kind="HOLD")])
    add("impl-lb-kind-outside-enum", "impl", command=LB, text="lend Dune to Ann Lee",
        answers=[LEND(kind="borrow"), LEND(kind="loan")])
    add("impl-lb-key-value-assist", "impl", command=LB, text="lend_book member=Ann Lee title=Dune copies=2", answers=[])
    add("impl-lb-key-value-assist-bad-int", "impl", command=LB, text="lend_book member=Ann Lee title=Dune copies=two",
        answers=[LEND()])
    # shelf_code: a model validator refuses (with a field, and without)
    SC = "shelf_code"
    add("impl-sc-clean", "impl", command=SC, text="label the shelf A-12", answers=[A(code="A-12", floor=0)])
    add("impl-sc-validator-field", "impl", command=SC, text="label the shelf BAD-1",
        answers=[A(code="BAD-1", floor=0), A(code="BAD-1", floor=0)])
    add("impl-sc-validator-then-clean", "impl", command=SC, text="label the shelf BAD-1 or A-2",
        answers=[A(code="BAD-1", floor=0), A(code="A-2", floor=0)])
    add("impl-sc-validator-no-field", "impl", command=SC, text="label the shelf NOFIELD",
        answers=[A(code="NOFIELD", floor=0), A(code="NOFIELD", floor=0)])
    add("impl-sc-guard-then-validator", "impl", command=SC, text="label the shelf BAD-1",
        answers=[A(code="Q-9", floor=0), A(code="BAD-1", floor=0)])
    add("impl-sc-missing", "impl", command=SC, text="label the shelf", answers=[A(code=NF, floor=0)])
    # all defaults: nothing required
    AD = "all_defaults"
    add("impl-ad-clean", "impl", command=AD, text="show the page", answers=[A(page=1, label=None)])
    add("impl-ad-int-not-written-then-clean", "impl", command=AD, text="show page 3",
        answers=[A(page=4, label=None), A(page=3, label=None)])
    # a planning command's Input read by the execution path: the guard is left to check_arguments
    add("impl-plan-author-clean", "impl", command="author_books", text="list the books by Frank Herbert",
        answers=[A(author="Frank Herbert")])
    add("impl-plan-author-invented", "impl", command="author_books", text="list the books by Frank Herbert",
        answers=[A(author="Stephen King"), A(author="Stephen King")])
    add("impl-plan-author-missing", "impl", command="author_books", text="list the books", answers=[A(author=NF)])
    # an agent's own command text: xml, name=value, positional, bare, then the LLM
    for mode in ("agent", "agent_code"):
        add(f"impl-{mode}-xml-single", "impl", mode=mode, command=FM, text="find_member <query>Ann Lee</query>", answers=[])
        add(f"impl-{mode}-xml-example", "impl", mode=mode, command=LB,
            text="<member>Jane Roe</member><title>Dune</title><copies>1</copies><tags>a</tags><kind>loan</kind><note>x</note>",
            answers=[LEND()])
        add(f"impl-{mode}-xml-all", "impl", mode=mode, command=LB,
            text="<member>Ann Lee</member><title>Dune</title><copies>2</copies><tags>a</tags><tags>b</tags><kind>hold</kind><note>x</note>",
            answers=[])
        add(f"impl-{mode}-key-value", "impl", mode=mode, command=FM, text="find_member query=Ann Lee", answers=[])
        add(f"impl-{mode}-key-value-quoted-equals", "impl", mode=mode, command=FM, text="find_member query='a=b c'",
            answers=[A(query="a")])
        add(f"impl-{mode}-key-value-note", "impl", mode=mode, command=FM, text="find_member query=Ann Lee (opens it)", answers=[])
        add(f"impl-{mode}-positional-one-token", "impl", mode=mode, command=FM, text="find_member Ann", answers=[])
        add(f"impl-{mode}-positional-multi-token-llm", "impl", mode=mode, command=FM, text="find_member Ann Lee",
            answers=[A(query="Ann Lee")])
        add(f"impl-{mode}-bare", "impl", mode=mode, command=AD, text="all_defaults", answers=[])
        add(f"impl-{mode}-bare-required-llm", "impl", mode=mode, command=FM, text="find_member",
            answers=[A(query=NF)])
        add(f"impl-{mode}-llm-clean", "impl", mode=mode, command=LB, text="lend Dune to Ann Lee please",
            answers=[LEND()])
        add(f"impl-{mode}-llm-guard-then-clean", "impl", mode=mode, command=FM, text="please find Ann Lee",
            answers=[A(query="Bob"), A(query="Ann Lee")])
        add(f"impl-{mode}-llm-guard-twice", "impl", mode=mode, command=FM, text="please find Ann Lee",
            answers=[A(query="Bob"), A(query="Bob")])
        add(f"impl-{mode}-unbound-lone-value", "impl", mode=mode, command=SC, text="shelf_code A12",
            answers=[A(code=NF, floor=0)])
        add(f"impl-{mode}-validator-field", "impl", mode=mode, command=SC, text="shelf_code code=BAD-7", answers=[])
    add("impl-slash-in-agent-session", "impl", mode="slash", command=FM, text="find_member query=Ann Lee", answers=[])
    add("impl-slash-in-agent-session-llm", "impl", mode="slash", command=FM, text="find Ann Lee",
        answers=[A(query="Ann Lee")])
    # a pending question (the stored-merge path: no extractor model)
    add("impl-pending-merge-single", "impl", command=FM, text="Ann Lee", pending={"query": NF}, answers=[])
    add("impl-pending-merge-two", "impl", command=LB, text="Dune, Ann Lee",
        pending={"member": NF, "title": NF, "copies": 1, "tags": None, "kind": "loan", "note": None}, answers=[])
    add("impl-pending-merge-validator", "impl", command=SC, text="BAD-3", pending={"code": NF, "floor": 0}, answers=[])

    # ---- ParameterExtraction._llm_extract on its own
    add("llm-fm-clean", "llm", command=FM, text=FMT, answers=[A(query="Ann Lee")])
    add("llm-fm-guard-then-clean", "llm", command=FM, text=FMT, answers=[A(query="Bob"), A(query="Ann Lee")])
    add("llm-fm-guard-twice", "llm", command=FM, text=FMT, answers=[A(query="Bob"), A(query="Bob")])
    add("llm-fm-retry-raises", "llm", command=FM, text=FMT, answers=[A(query="Bob")], raise_on=2)
    add("llm-fm-first-raises", "llm", command=FM, text=FMT, answers=[], raise_on=1)
    add("llm-fm-checks-off", "llm", command=FM, text=FMT, answers=[A(query="Bob")], checks="off")
    add("llm-plan-author-skips-guard", "llm", command="author_books", text="list the books by Frank Herbert",
        answers=[A(author="Stephen King"), A(author="Stephen King")])
    add("llm-lb-int-respell", "llm", command=LB, text="lend DUNE to ann lee, 2 copies", answers=[LEND(member="ann lee", title="DUNE", copies=2)])

    # ---- extract_arguments: a planning command (retry on any invalid result) and an execution command
    AB, ABT = "author_books", "list the books by Frank Herbert"
    add("args-ab-clean", "args", command=AB, text=ABT, answers=[A(author="Frank Herbert")])
    add("args-ab-invented-then-clean", "args", command=AB, text=ABT, answers=[A(author="Stephen King"), A(author="Frank Herbert")])
    add("args-ab-invented-twice", "args", command=AB, text=ABT, answers=[A(author="Stephen King"), A(author="Stephen King")])
    add("args-ab-missing-then-clean", "args", command=AB, text=ABT, answers=[A(author=NF), A(author="Frank Herbert")])
    add("args-ab-missing-twice", "args", command=AB, text=ABT, answers=[A(author=NF), A(author=NF)])
    add("args-ab-missing-then-invents", "args", command=AB, text=ABT, answers=[A(author=NF), A(author="Stephen King")])
    add("args-ab-retry-raises", "args", command=AB, text=ABT, answers=[A(author="Stephen King")], raise_on=2)
    add("args-ab-first-raises", "args", command=AB, text=ABT, answers=[], raise_on=1)
    add("args-ab-pronoun", "args", command=AB, text="which books did the author of that one write", answers=[A(author="it"), A(author="it")])
    add("args-ab-placeholder-then-clean", "args", command=AB, text=ABT, answers=[A(author="<entity_name>"), A(author="Frank Herbert")])
    add("args-ab-vocabulary", "args", command=AB, text="which books did the author write", answers=[A(author="author"), A(author="author")])
    TL, TLT = "transfer_loan", "Move Ann Lee's loan to the Main branch"
    add("args-tl-clean", "args", command=TL, text=TLT, answers=[A(member="Ann Lee", branch="Main", entity_type="identity", note="")])
    add("args-tl-role-conflict", "args", command=TL, text="Move the loan of member Ann Lee to branch Ann Lee",
        answers=[A(member="Ann Lee", branch="Ann Lee", entity_type="identity", note="")] * 2)
    add("args-tl-role-conflict-then-clean", "args", command=TL, text="Move the loan of member Ann Lee to branch Main",
        answers=[A(member="Ann Lee", branch="Ann Lee", entity_type="identity", note=""),
                 A(member="Ann Lee", branch="Main", entity_type="identity", note="")])
    add("args-tl-missing-branch", "args", command=TL, text="Move Ann Lee's loan",
        answers=[A(member="Ann Lee", branch=NF, entity_type="identity", note=""),
                 A(member="Ann Lee", branch=NF, entity_type="identity", note="")])
    add("args-tl-optional-placeholder", "args", command=TL, text=TLT,
        answers=[A(member="Ann Lee", branch="Main", entity_type="identity", note="<entity_name>")])
    LG = "ledger_pc"
    add("args-lg-clean", "args", command=LG, text="write a line for Ann Lee, 3 items",
        answers=[LEDGER(count=3)])
    add("args-lg-validator-field-then-clean", "args", command=LG, text="write a line for Ann Lee with code BAD",
        answers=[LEDGER(code="BAD"), LEDGER(code="OK")])
    add("args-lg-validator-field-twice", "args", command=LG, text="write a line for Ann Lee with code BAD",
        answers=[LEDGER(code="BAD"), LEDGER(code="BAD")])
    add("args-lg-validator-no-field", "args", command=LG, text="write a line for Nobody",
        answers=[LEDGER(owner="Nobody"), LEDGER(owner="Nobody")])
    add("args-lg-validator-no-field-then-clean", "args", command=LG, text="write a line for Nobody and Ann Lee",
        answers=[LEDGER(owner="Nobody"), LEDGER(owner="Ann Lee")])
    add("args-lg-count-not-written", "args", command=LG, text="write a line for Ann Lee, 3 items",
        answers=[LEDGER(count=8), LEDGER(count=8)])
    TG = "taglist_pc"
    add("args-tg-clean", "args", command=TG, text="tag Atlas with draft and final", answers=[TAGS()])
    add("args-tg-unwritten-item-dropped", "args", command=TG, text="tag Atlas with draft",
        answers=[TAGS(tags=["draft", "zzz"])])
    add("args-tg-missing-then-clean", "args", command=TG, text="tag Atlas with draft and final",
        answers=[TAGS(tags=[]), TAGS()])
    add("args-tg-missing-twice", "args", command=TG, text="tag Atlas", answers=[TAGS(tags=[])] * 2)
    add("args-tg-all-unwritten", "args", command=TG, text="tag Atlas with draft",
        answers=[TAGS(tags=["zzz"]), TAGS(tags=["draft"])])
    add("args-lg-missing-owner-then-clean", "args", command=LG, text="write a line for Ann Lee",
        answers=[LEDGER(owner=NF), LEDGER(owner="Ann Lee")])
    add("args-pc-no-params", "args", command="empty_pc", text="say hello", answers=[A()])
    add("args-pc-no-params-retry-not-needed", "args", command="empty_pc", text="say hello", answers=[])
    add("args-bd-clean", "args", command="book_details", text="give me the details of Dune", answers=[A(title="Dune")])
    add("args-bd-not-given-text", "args", command="book_details", text="give me the details of this book",
        answers=[A(title=NF), A(title=NF)])
    add("args-br-bools", "args", command="book_report", text="summarise the loans and shelf for Dune",
        answers=[A(title="Dune", loans=True, brief=False)])
    add("args-fr-enum-bad", "args", command="find_record", text="is there a member named Lee",
        answers=[A(name="Lee", kind="author"), A(name="Lee", kind="member")])
    # an execution command through extract_arguments: the guard retry inside the fill, no outer retry
    add("args-exec-fm-clean", "args", command=FM, text=FMT, answers=[A(query="Ann Lee")])
    add("args-exec-fm-guard-then-clean", "args", command=FM, text=FMT, answers=[A(query="Bob"), A(query="Ann Lee")])
    add("args-exec-fm-guard-twice", "args", command=FM, text=FMT, answers=[A(query="Bob"), A(query="Bob")])
    add("args-exec-fm-missing-no-retry", "args", command=FM, text=FMT, answers=[A(query=NF)])
    add("args-exec-fm-retry-raises", "args", command=FM, text=FMT, answers=[A(query="Bob")], raise_on=2)
    add("args-exec-lb-int-not-written", "args", command=LB, text=LBT,
        answers=[LEND(copies=7), LEND(copies=7)])
    add("args-exec-sc-validator-retry", "args", command=SC, text="label the shelf BAD-1",
        answers=[A(code="BAD-1", floor=0), A(code="BAD-1", floor=0)])
    add("args-exec-fm-checks-off", "args", command=FM, text=FMT, answers=[A(query="Bob")], checks="off")
    add("args-exec-ad-clean", "args", command=AD, text="show the page", answers=[A(page=1, label=None)])

    # ---- typed_planning_call: the arguments, the guards, one retry, the typed fill fallback, the step proposal
    STEP = PLANNER()
    add("typed-ab-clean", "typed", command=AB, text=ABT, answers=[A(author="Frank Herbert")], planner=[STEP])
    add("typed-ab-refused-then-clean", "typed", command=AB, text=ABT,
        answers=[A(author="Stephen King"), A(author="Frank Herbert")], planner=[STEP])
    add("typed-ab-refused-twice-fallback-fills", "typed", command=AB, text=ABT,
        answers=[A(author="Stephen King"), A(author="Stephen King")],
        planner=[PLANNER(author="Frank Herbert"), STEP])
    add("typed-ab-refused-twice-fallback-refused", "typed", command=AB, text=ABT,
        answers=[A(author="Stephen King"), A(author="Stephen King")],
        planner=[PLANNER(author="Isaac Asimov"), STEP])
    add("typed-ab-refused-twice-fallback-empty", "typed", command=AB, text=ABT,
        answers=[A(author="Stephen King"), A(author="Stephen King")],
        planner=[PLANNER(author=""), STEP])
    add("typed-ab-missing-then-clean", "typed", command=AB, text=ABT,
        answers=[A(author=NF), A(author="Frank Herbert")], planner=[STEP])
    add("typed-ab-missing-twice-fallback-fills", "typed", command=AB, text=ABT,
        answers=[A(author=NF), A(author=NF)], planner=[PLANNER(author="Frank Herbert"), STEP])
    add("typed-ab-missing-twice-fallback-not-given", "typed", command=AB, text=ABT,
        answers=[A(author=NF), A(author=NF)], planner=[PLANNER(author=NF), STEP])
    add("typed-ab-retry-raises", "typed", command=AB, text=ABT, answers=[A(author="Stephen King")], planner=[STEP], raise_on=2)
    add("typed-ab-first-raises", "typed", command=AB, text=ABT, answers=[], planner=[STEP], raise_on=1)
    add("typed-ab-pronoun-twice", "typed", command=AB, text="which books did the author of that one write",
        answers=[A(author="it"), A(author="it")], planner=[PLANNER(author="it"), STEP])
    add("typed-tl-clean", "typed", command=TL, text=TLT,
        answers=[A(member="Ann Lee", branch="Main", entity_type="identity", note="")], planner=[STEP])
    add("typed-tl-role-conflict-twice", "typed", command=TL, text="Move the loan of member Ann Lee to branch Ann Lee",
        answers=[A(member="Ann Lee", branch="Ann Lee", entity_type="identity", note="")] * 2,
        planner=[PLANNER(member="Ann Lee", branch="Ann Lee"), STEP])
    add("typed-tl-keeps-extractor-value", "typed", command=TL, text=TLT,
        answers=[A(member="Ann Lee", branch=NF, entity_type="identity", note="")] * 2,
        planner=[PLANNER(member="Main", branch="Main"), STEP])
    add("typed-tl-enum-outside-then-clean", "typed", command=TL, text=TLT,
        answers=[A(member="Ann Lee", branch="Main", entity_type="savings", note=""),
                 A(member="Ann Lee", branch="Main", entity_type="account", note="")], planner=[STEP])
    add("typed-tl-fallback-adapter-parse-error", "typed", command=TL, text="Move Ann Lee's loan",
        answers=[A(member="Ann Lee", branch=NF, entity_type="identity", note="")] * 2,
        planner=[PLANNER(branch="Main", entity_type="savings"), STEP])
    add("typed-lg-validator-field-twice", "typed", command=LG, text="write a line for Ann Lee with code BAD",
        answers=[LEDGER(code="BAD")] * 2, planner=[STEP])
    add("typed-lg-validator-no-field-twice", "typed", command=LG, text="write a line for Nobody",
        answers=[LEDGER(owner="Nobody")] * 2, planner=[PLANNER(owner="Nobody"), STEP])
    add("typed-lg-validator-then-clean", "typed", command=LG, text="write a line for Ann Lee with code BAD",
        answers=[LEDGER(code="BAD"), LEDGER(code="OK")], planner=[STEP])
    add("typed-tg-clean", "typed", command=TG, text="tag Atlas with draft and final", answers=[TAGS()],
        planner=[STEP])
    add("typed-tg-unwritten-item-dropped", "typed", command=TG, text="tag Atlas with draft",
        answers=[TAGS(tags=["draft", "zzz"])], planner=[STEP])
    add("typed-tg-missing-twice-fallback-fills", "typed", command=TG, text="tag Atlas with draft and final",
        answers=[TAGS(tags=[])] * 2, planner=[PLANNER(tags=["draft", "final"]), STEP])
    add("typed-lg-count-replaced", "typed", command=LG, text="write a line for Ann Lee, 3 items",
        answers=[LEDGER(count=8)], planner=[STEP])
    add("typed-lg-count-refused", "typed", command=LG, text="write a line for Ann Lee, 3 or 4 items",
        answers=[LEDGER(count=8)] * 2, planner=[STEP])
    add("typed-no-params", "typed", command="empty_pc", text="say hello", answers=[], planner=[])
    add("typed-bd-not-given", "typed", command="book_details", text="give me the details of this book",
        answers=[A(title=NF)] * 2, planner=[PLANNER(title=""), STEP])
    add("typed-br-bools", "typed", command="book_report", text="summarise the loans and shelf for Dune",
        answers=[A(title="Dune", loans=True, brief=False)], planner=[STEP])
    add("typed-fr-enum", "typed", command="find_record", text="is there a member named Lee",
        answers=[A(name="Lee", kind="member")], planner=[STEP])

    # ---- extracted_arguments and checked_arguments directly
    add("extracted-ab", "extracted", command=AB, text=ABT, answers=[A(author="Frank Herbert")], check=True)
    add("extracted-ab-not-given", "extracted", command=AB, text=ABT, answers=[A(author=NF)], check=True)
    add("extracted-ab-invented", "extracted", command=AB, text=ABT, answers=[A(author="Stephen King")], check=True)
    add("extracted-tl", "extracted", command=TL, text=TLT,
        answers=[A(member="Ann Lee", branch="Main", entity_type="identity", note="")], check=True)
    add("extracted-lg-unchecked", "extracted", command=LG, text="write a line for Ann Lee",
        answers=[LEDGER(owner="Ann Lee", count=2)])
    add("extracted-raises", "extracted", command=AB, text=ABT, answers=[], raise_on=1)
    add("extracted-no-model-answer", "extracted", command=AB, text=ABT, answers=[])

    # ---- reextracted_arguments: a first phase found nothing
    OBS = "no author matched"
    add("reextract-new-args", "reextract", command=AB, text=ABT, observation=OBS, args={"author": "Frank Herbert"},
        answers=[A(author="Herbert")])
    add("reextract-equal-args", "reextract", command=AB, text=ABT, observation=OBS, args={"author": "Frank Herbert"},
        answers=[A(author="Frank Herbert")])
    add("reextract-refused", "reextract", command=AB, text=ABT, observation=OBS, args={"author": "Frank Herbert"},
        answers=[A(author="Stephen King")])
    add("reextract-missing", "reextract", command=AB, text=ABT, observation=OBS, args={"author": "Frank Herbert"},
        answers=[A(author=NF)])
    add("reextract-raises", "reextract", command=AB, text=ABT, observation=OBS, args={"author": "Frank Herbert"},
        answers=[], raise_on=1)
    add("reextract-validator-refuses", "reextract", command=LG, text="write a line for Ann Lee with code BAD",
        observation=OBS, args={"owner": "Ann Lee", "count": 0, "code": "OK"}, answers=[LEDGER(code="BAD")])
    add("reextract-with-verdicts", "reextract", command=AB, text=ABT, observation=OBS, args={"author": "X"},
        answers=[A(author="Frank Herbert")], verdicts={})
    # ---- the extractor's own "is this field given" check resets a filled field (scripted, per call)
    add("impl-fm-reset-first", "impl", command=FM, text=FMT, answers=[A(query="Ann Lee"), A(query="Ann Lee")],
        resets=[{"query": 0.97}, {}])
    add("impl-fm-guard-then-reset-second", "impl", command=FM, text=FMT,
        answers=[A(query="Bob Jones"), A(query="Ann Lee")], resets=[{}, {"query": 0.91}])
    add("impl-lb-reset-first-guard-retry-no-reset", "impl", command=LB, text="lend Dune to Ann Lee",
        answers=[LEND(member="Bob", copies=1), LEND()], resets=[{"copies": 0.9, "note": 0.95}, {}])
    add("impl-lb-reset-both", "impl", command=LB, text="lend Dune to Ann Lee",
        answers=[LEND(member="Bob", copies=1), LEND()], resets=[{"copies": 0.9}, {"kind": 0.93}])
    add("impl-agent-llm-reset", "impl", mode="agent", command=FM, text="please find Ann Lee",
        answers=[A(query="Ann Lee"), A(query="Ann Lee")], resets=[{"query": 0.99}, {}])
    add("llm-fm-reset-retry-only", "llm", command=FM, text=FMT, answers=[A(query="Bob"), A(query="Ann Lee")],
        resets=[{}, {"query": 0.9}])
    add("llm-fm-reset-first-retry-clean", "llm", command=FM, text=FMT, answers=[A(query="Bob"), A(query="Ann Lee")],
        resets=[{"query": 0.9}, {}])
    add("args-ab-reset-then-clean", "args", command=AB, text=ABT, answers=[A(author="Frank Herbert")] * 2,
        resets=[{"author": 0.97}, {}])
    add("args-ab-reset-both", "args", command=AB, text=ABT, answers=[A(author="Frank Herbert")] * 2,
        resets=[{"author": 0.97}, {"author": 0.97}])
    add("args-exec-fm-reset-first", "args", command=FM, text=FMT, answers=[A(query="Ann Lee"), A(query="Ann Lee")],
        resets=[{"query": 0.97}, {}])
    add("args-exec-fm-guard-then-reset", "args", command=FM, text=FMT, answers=[A(query="Bob"), A(query="Ann Lee")],
        resets=[{}, {"query": 0.9}])
    add("typed-ab-reset-then-clean", "typed", command=AB, text=ABT, answers=[A(author="Frank Herbert")] * 2,
        planner=[STEP], resets=[{"author": 0.97}, {}])
    add("extracted-ab-reset", "extracted", command=AB, text=ABT, answers=[A(author="Frank Herbert")],
        check=True, resets=[{"author": 0.97}])
    add("reextract-reset", "reextract", command=AB, text=ABT, observation=OBS, args={"author": "Frank Herbert"},
        answers=[A(author="Herbert")], resets=[{"author": 0.97}])
    return cases


CASES = _cases()


def _form_cases() -> list[dict]:
    forms: list[dict] = []

    def add(form: str, cls: str, text: str, **extra) -> None:
        forms.append({"id": f"form-{form}-{len(forms):03d}-{cls}", "site": "form", "form": form, "cls": cls, "text": text,
                      **extra})

    # xml
    for cls, text in [
        ("OneStr", "<query>Ann Lee</query>"), ("OneStr", "<x>  spaced out  </x>"), ("OneStr", "no tags here"),
        ("OneStr", "<q>multi\nline</q>"), ("NoParams", "anything"), ("FindParams", "<query>Ann</query><start>2</start>"),
        ("FindParams", "<query>Ann</query>"), ("FindParams", "<start>2</start><query>Ann</query>"),
        ("ListParams", "<ids>a</ids><ids>b</ids><counts>1</counts><note>n</note>"),
        ("ListParams", "<ids>[\"a\", \"b\"]</ids><counts>[1, 2]</counts><note>n</note>"),
        ("ListParams", "<ids>a, b</ids><counts>1 2</counts><note>n</note>"),
        ("ListParams", "<ids>a b</ids><counts>x</counts><note>n</note>"),
        ("ListParams", "<ids>['a', 'b']</ids><counts>[3]</counts><note>n</note>"),
        ("ListParams", "<ids>solo</ids><counts>7</counts><note>n</note>"),
        ("ExampleParams", "<query>Jane Roe</query><size>3</size>"), ("ExampleParams", "<query>Ann</query><size>3</size>"),
        ("EnumParams", "<kind>hold</kind><note>x</note>"), ("TwoStrings", "<book_uid>b1</book_uid><filter>f</filter>"),
        ("TwoStrings", "<book_uid>b1</book_uid>"),
    ]:
        add("xml", cls, text)
    # name=value
    for cls, text, extra in [
        ("FindParams", "Shelf/find_book query=Ann Lee start=2", {}), ("FindParams", "find_book query=Ann Lee", {}),
        ("FindParams", "query=Ann Lee", {}), ("FindParams", "open_book query=Ann", {}),
        ("FindParams", "find_book query=Ann Lee start=two", {}), ("FindParams", "find_book query=None", {}),
        ("FindParams", "find_book query=\"Ann Lee\" start=3", {}), ("FindParams", "find_book query=Ann Lee (opens it)", {}),
        ("FindParams", "find_book query=Ann query=Bob", {}), ("FindParams", "find_book colour=red", {}),
        ("FindParams", "find_book query=<name>", {}), ("FindParams", "find_book query=a <b>x</b>", {}),
        ("FindParams", "find_book query=...", {}), ("FindParams", "find_book query=", {}),
        ("FindParams", "find_book", {}), ("FindParams", "find_book query=Ann, start=4", {}),
        ("FindParams", "find_book query=`Ann Lee`", {}),
        ("ExampleParams", "find_book query=Jane Roe", {}), ("ExampleParams", "find_book query=Ann size=2", {}),
        ("ListParams", "find_book ids=a,b counts=[1,2]", {}), ("ListParams", "find_book ids=[\"a\",\"b\"] note=n", {}),
        ("ListParams", "find_book ids=[bad", {}),
        ("EnumParams", "find_book kind=hold", {}), ("EnumParams", "find_book kind=borrow", {}),
        ("OneStr", "find_book query=a=b", {}), ("NoParams", "find_book x=1", {}),
        ("FindParams", "find_book query='Ann Lee' (note)", {"form": "key_value_code"}),
        ("FindParams", "find_book query='a=b c' start=2", {"form": "key_value_code"}),
        ("FindParams", "find_book query=\"x <y> ...\"", {"form": "key_value_code"}),
        ("FindParams", "find_book query=Ann Lee (note)", {"form": "key_value_code"}),
        ("ExampleParams", "find_book query=Jane Roe", {"form": "key_value_code"}),
        ("FindParams", "find_book query=Ann Lee start='x'", {"form": "key_value_code"}),
    ]:
        add(extra.pop("form", "key_value"), cls, text, **extra)
    # positional
    for cls, text in [
        ("OneStr", "find_book Ann"), ("OneStr", "find_book Ann Lee"), ("OneStr", "find_book query Ann"),
        ("OneStr", "find_book query"), ("OneStr", "find_book 'Ann'"), ("OneStr", "find_book \"\""), ("OneStr", "find_book *"),
        ("OneStr", "find_book a / b"), ("OneStr", "find_book"), ("OneStr", "open_book Ann"),
        ("MarkedFindParams", "find_book Ann Lee"), ("MarkedFindParams", "find_book query: Ann Lee"),
        ("MarkedFindParams", "find_book start 3"), ("MarkedFindParams", "find_book 'Ann Lee'"),
        ("FindParams", "find_book Ann"), ("TwoStrings", "find_book b1"), ("MarkedTwoStrings", "find_book b1 f"),
        ("MarkedTwoStrings", "find_book filter x"), ("OpenParams", "find_book uid 203b"), ("OpenParams", "find_book book_uid 203b"),
        ("OpenParams", "find_book other_uid 2b55"), ("OpenParams", "find_book 203b"), ("ExampleParams", "find_book Jane Roe"),
        ("OneStr", "find_book a=b"), ("OneStr", "find_book <x>y</x>"), ("EnumParams", "find_book hold"),
    ]:
        add("positional", cls, text)
    # bare
    for cls, text in [("AllDefaultsLike", ""), ]:
        pass
    for cls, text in [
        ("FindParams", ""), ("FindParams", "find_book"), ("FindParams", "Shelf/find_book"), ("FindParams", "/find_book"),
        ("FindParams", "find_book x"), ("OneStr", ""), ("NoParams", ""), ("EnumParams", "find_book"),
        ("ExampleParams", "find_book"), ("OpenParams", "find_book"), ("ListParams", ""),
    ]:
        add("bare", cls, text)
    return forms


CASES = CASES + _form_cases()
assert len({case["id"] for case in CASES}) == len(CASES), "duplicate case ids"
