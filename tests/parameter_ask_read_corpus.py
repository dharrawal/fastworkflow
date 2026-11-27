"""The table of the question-wording and reply-reading equivalence test (bd fix-qoyz step 4).

``tests/fixtures/parameter_ask_read_equivalence.json`` holds what the code gave BEFORE the wording was moved into
``fastworkflow/parameters/ask.py`` and the reading into ``fastworkflow/parameters/read.py``: the questions of the
planning side (``planning_commands.refusal_question`` / ``missing_argument_question`` / ``gaps_question`` /
``planning_refusal_parameter_request`` and ``parameter_clarification.planning_question``), the questions of the
execution side (``InputForParamExtraction.validate_parameters``' message and the framing of
``ParameterExtraction._extract_impl``, driven with a scripted extractor), and the reading of replies
(``parameter_clarification.read_reply`` / ``named_values`` and
``ParameterExtraction._extract_and_merge_missing_parameters``). ``build_table`` runs the code as it is and returns the
same table; the test compares the two.

Real components only: the tutorial workflow's planning commands (and ``tests/parameter_check_corpus``'s, written into a
temporary copy: that module may gain commands, a command this table names must not change), real ``Input`` classes and
the real guards. The extractor's model call is replaced by a scripted answer.
"""
from __future__ import annotations

import contextlib
import json
import types
from pathlib import Path

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

import fastworkflow
from fastworkflow.parameters import parameter_clarification as pcl
from fastworkflow import planning_commands as pc
from fastworkflow.parameters.parameter_clarification import Extraction
from fastworkflow.parameters import check as one_guard
from fastworkflow.parameters.read import Slot
from tests import parameter_check_corpus as corpus


def norm(v):
    return corpus.norm(v)


def ext(e):
    return None if e is None else {"values": norm(e.values), "missing": list(e.missing), "invalid": [list(x) for x in e.invalid],
                                   "earlier": list(e.earlier), "reason": e.reason}


def safe(fn):
    try:
        return {"ok": fn()}
    except BaseException as exc:  # noqa: BLE001
        return {"raised": type(exc).__name__, "message": str(exc)}


TABLE: dict = {"wording_plan": {}, "wording_exec": {}, "frame": {}, "read_reply": {}, "merge": {}}


def setup(tmp: Path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp / "ctx")})
    fastworkflow.RoutingRegistry.clear_registry()
    return tmp, corpus.build_commands(tmp)


# ------------------------------------------------------------------------------------------------ plan wording

def plan_wording(commands: dict):
    out = TABLE["wording_plan"]
    # 1. real refusals of real commands
    scenarios = [
        ("member_loans", {"member": ""}, "Which books does that member have on loan?"),
        ("member_loans", {"member": "Ann"}, "Which books does it have?"),
        ("find_record", {"name": "Dune", "kind": "film"}, "Is Dune a film?"),
        ("find_record", {"name": "", "kind": ""}, "find it"),
        ("find_books", {"query": "Dune"}, "which books by Asimov"),
        ("find_books", {"query": ""}, "which books"),
        ("find_book", {"title": ""}, "find the book"),
        ("find_book", {"title": "Zorg"}, "find the book Dune"),
        ("book_by_title", {"title": "it"}, "the book by title"),
        ("drift_rows", {"title": "", "count": 3}, "list rows of Dune"),
        ("drift_rows", {"title": "Dune", "count": 99}, "list 3 rows of Dune and 4 more"),
        ("drift_code", {"name": "", "code": "ZZ-99"}, "label the shelf"),
        ("drift_code", {"name": "Ann", "code": "ZZ-99"}, "label Ann XY-12"),
        ("drift_shelf", {"shelf": ""}, "move it"),
        ("drift_shelf", {"shelf": "S1"}, "move it to S1"),
        ("drift_shelf", {"shelf": "#S1"}, "move to #S1"),
        ("drift_choice", {"kind": "film"}, "find a film"),
        ("drift_subject", {"name": "Nobody"}, "show the books of Ann"),
        ("drift_subject", {"name": ""}, "show the books"),
        ("shape_all", {}, "do it"),
        ("shape_all", {"label": "x", "req": "y", "limit": "ten"}, "do x y limit ten"),
        ("shape_all", {"label": "x", "req": "", "flag": "maybe"}, "do x"),
        ("shape_all", {"label": "the x", "req": "y", "size": "huge"}, "do the x y"),
        ("shape_all", {"label": "x", "req": "y", "colour": "pink"}, "do x y pink"),
        ("shape_lists", {}, "tag them"),
        ("shape_lists", {"need": ["a"], "tags": ["zz"]}, "tag a"),
        ("shape_required_enum", {}, "order it"),
        ("shape_required_enum", {"size": "tiny", "qty": 3, "amount": 1.5}, "order 3 tiny"),
        ("shape_required_enum", {"size": "small"}, "order small"),
        ("shape_default_str", {}, "list"),
        ("shape_subjects", {"who": "Nobody"}, "link Ann and Bob"),
        ("shape_subjects", {"who": ""}, "link them"),
        ("transfer_loan", {"member": "", "branch": ""}, "move it"),
        ("transfer_loan", {"member": "Ann", "branch": "Ann"}, "move the loan of Ann to Ann"),
        ("transfer_loan", {"member": "Ann", "branch": ""}, "move Ann's loan"),
    ]
    n = 0
    for name, raw, request in scenarios:
        command = commands.get(name)
        if command is None:
            out[f"missing_command:{name}"] = "absent"
            continue
        full = {p.name: raw.get(p.name) for p in command.params}
        key = f"{name}|{json.dumps(raw, sort_keys=True)}|{request}"
        record: dict = {}
        try:
            pc.checked_arguments(command, dict(full), request, None, None)
            record["checked"] = "ok"
        except pc.PlanningCommandCallError as refusal:
            record["checked"] = "refused"
            record.update(plan_refusal_outputs(command, refusal))
        gaps = pc.argument_gaps(command, dict(full), request)
        record["gaps"] = ext(gaps)
        record["gaps_question"] = safe(lambda: pc.gaps_question(command, gaps)) if gaps.names else None
        for confirm in (False, True):
            for retry in ("", pcl.RETRY):
                for description in ("", "Does a thing."):
                    q = safe(lambda: pcl.planning_question(command, {k: v for k, v in full.items() if v not in (None, "")}, gaps,
                                                           confirm=confirm, description=description, retry=retry))
                    record[f"planning_question|confirm={confirm}|retry={bool(retry)}|desc={bool(description)}"] = q
        out[key] = record
        n += 1
    # 2. synthetic refusals over every param of a few commands
    for name in ("shape_all", "member_loans", "find_record", "drift_shelf", "shape_required_enum"):
        command = commands[name]
        for p in command.params:
            for text, state, why in (
                    (f"missing required argument {p.name}", "missing", ""),
                    (f"{p.name}='zz' is not written in the request", "invalid", "is not written in the request"),
                    (f"{p.name}='zz' names no record of the request", "invalid", "names no record of the request"),
                    (f"{p.name}='zz' is not one of a, b", "invalid", "is not one of a, b"),
                    (f"{p.name}='zz' is not an integer", "invalid", "is not an integer"),
                    (f"{p.name}='zz' is not true or false", "invalid", "is not true or false"),
                    (f"{p.name}='zz' is also the value of other_name", "invalid", "is also the value of other_name"),
                    (f"{p.name}='zz' {pc.PLACEHOLDER_WHY}", "missing", pc.PLACEHOLDER_WHY),
                    (f"{p.name}='zz' {pc.PRONOUN_WHY}", "missing", pc.PRONOUN_WHY),
                    (f"{p.name}='zz' is odd", "invalid", "is odd"),
                    (f"{p.name} is wrong", None, f"{p.name} is wrong"),
                    ("the arguments break a rule of x: nope", None, "nope")):
                if state is None:       # a validator's own refusal naming no input: its bare message is carried
                    refusal = pc.PlanningCommandInputError(text)
                    refusal.refusal = why
                else:                   # a guard's refusal of one argument, as data
                    refusal = pc.PlanningCommandInputError(
                        text, gap=one_guard.Gap(p.name, state, "missing" if state == "missing" and not why else "refused",
                                                why, None if state == "missing" and not why else "zz", text))
                out[f"synthetic|{name}|{p.name}|{text}"] = plan_refusal_outputs(command, refusal)
            for fields in ((p.name,), (p.name, "other")):
                refusal = pc.PlanningCommandInputError("bad things", fields)
                refusal.refusal = "bad things"
                out[f"fields|{name}|{fields}"] = plan_refusal_outputs(command, refusal)
        out[f"missing_argument_question|{name}"] = {p.name: pc.missing_argument_question(command, p.name) for p in command.params}
        # gaps made by hand, attached to a refusal
        names = [p.name for p in command.params]
        hand = [Extraction(missing=tuple(names[:1])), Extraction(missing=tuple(names[:2]), invalid=((names[-1], "is odd"),)),
                Extraction(invalid=((names[0], "is not one of a, b"), (names[-1], "names no record of the request"))),
                Extraction(missing=(names[0],), earlier=(names[-1],), values={names[-1]: "v"}),
                Extraction(reason="a rule of the command"), Extraction()]
        for i, gaps in enumerate(hand):
            refusal = pc.PlanningCommandInputError("something", ())
            refusal.gaps = gaps
            out[f"hand_gaps|{name}|{i}"] = plan_refusal_outputs(command, refusal)
            out[f"hand_gaps|{name}|{i}|refusal_from_gaps"] = safe(lambda: refusal_from_gaps_record(gaps, {names[0]: "kept"}))
            for confirm in (False, True):
                for retry in ("", pcl.RETRY):
                    out[f"hand_gaps|{name}|{i}|planning_question|{confirm}|{bool(retry)}"] = safe(
                        lambda: pcl.planning_question(command, {names[-1]: "kept", names[0]: None}, gaps, confirm=confirm,
                                                      retry=retry, description="Desc here."))
            out[f"hand_gaps|{name}|{i}|planning_question_refusal"] = safe(
                lambda: pcl.planning_question(command, {}, gaps, refusal=refusal))
            out[f"hand_gaps|{name}|{i}|request"] = {"str": ext(gaps) and gaps.request(command.name),
                                                    "raw": gaps.request(command.name, stringify=False)}
    # 3. a plan with no parameters
    for name, command in commands.items():
        if not command.params:
            refusal = pcl.refusal_from_gaps(Extraction(missing=("x",)))
            out[f"noparams|{name}"] = plan_refusal_outputs(command, refusal)
            out[f"noparams|{name}|confirm"] = safe(lambda: pcl.planning_question(command, {}, Extraction(), confirm=True,
                                                                                  description="Nothing to give"))
            out[f"noparams|{name}|plain"] = safe(lambda: pcl.planning_question(command, {}, Extraction()))
            break
    return n


def refusal_from_gaps_record(gaps, values):
    e = pcl.refusal_from_gaps(gaps, values)
    return {"text": str(e), "gaps": ext(e.gaps), "fields": list(e.fields)}


def plan_refusal_outputs(command, refusal):
    return {
        "refusal": str(refusal),
        "refusal_question": safe(lambda: pc.refusal_question(command, refusal)),
        "refusal_gaps": safe(lambda: ext(pc.refusal_gaps(command, refusal))),
        "names_an_input": safe(lambda: pc.refusal_names_an_input(command, refusal)),
        "request": safe(lambda: pc.planning_refusal_parameter_request(command, refusal)),
        "planning_question": safe(lambda: pcl.planning_question(command, {"a": 1}, pc.refusal_gaps(command, refusal), refusal=refusal)),
        "planning_question_retry": safe(lambda: pcl.planning_question(command, {}, pc.refusal_gaps(command, refusal), refusal=refusal,
                                                                      retry=pcl.RETRY)),
        "planning_question_nogaps": safe(lambda: pcl.planning_question(command, {"a": 1}, pc.refusal_gaps(command, refusal))),
    }


# ------------------------------------------------------------------------------------------------ exec wording

class Req2(BaseModel):
    name: str = Field(description="a person's name")
    dept: str = Field(description="a department")


class WithSource(BaseModel):
    uid: str = Field(description="uid", json_schema_extra={"available_from": ["Library/find_book.py", "Library/list_books.py"]})
    other: str = Field(description="other", json_schema_extra={"available_from": ["x/y.py"]})
    plain: str = Field(description="plain")


class Pattern(BaseModel):
    code: str = Field(description="a code", examples=["XY-12"], pattern=r"^[A-Z]{2}-\d+$")
    tag: str = Field(description="a tag", pattern=r"^#\w+$")


class Typed(BaseModel):
    count: int = Field(description="how many")
    ratio: float = Field(description="ratio")
    flag: bool = Field(description="flag")
    kind: Literal["book", "member"] = Field(description="kind")


class Bounded(BaseModel):
    limit: int = Field(default=5, description="a limit")
    page_size: int = Field(default=5, ge=1, le=50, description="page")
    name: str = Field(default="", description="name")


class Listed(BaseModel):
    items: List[str] = Field(description="items")
    opt: Optional[List[str]] = Field(default=None, description="opt")


class Optional1(BaseModel):
    title: Optional[str] = Field(default=None, description="title")
    name: str = Field(description="a name")


class Rule(BaseModel):
    a: str = Field(description="a")
    b: str = Field(default="x", description="b")

    @model_validator(mode="after")
    def _rule(self):
        if self.a == "bad":
            raise fastworkflow.Refused("a must not be bad", field="a")
        if self.a == "both":
            raise fastworkflow.Refused("both are wrong", field=["a", "b"])
        if self.a == "nofield":
            raise fastworkflow.Refused("the call does not fit")
        if self.a == "boom":
            raise RuntimeError("boom")
        return self


class Fv(BaseModel):
    qty: int = Field(default=1, description="qty")

    @field_validator("qty")
    @classmethod
    def _q(cls, v):
        if v > 10:
            raise fastworkflow.Refused("too many", field="qty")
        return v


NF = "NOT_FOUND"
INT_NF = -9223372036854775807


def exec_wording(tmp: Path, commands: dict):
    from fastworkflow.parameters import signatures
    from fastworkflow.parameters.signatures import InputForParamExtraction

    out = TABLE["wording_exec"]
    folder = str(tmp / "tutorial")
    cases = [
        (Req2, {"name": NF, "dept": NF}, "find a person"),
        (Req2, {"name": "Ann", "dept": NF}, "find Ann"),
        (Req2, {"name": NF, "dept": "Sales"}, "find in Sales"),
        (Req2, {"name": "Ann", "dept": "Sales"}, "find Ann in Sales"),
        (Req2, {"name": "<name>", "dept": NF}, "find"),
        (Req2, {"name": "<9a1b2c3d4e5f60718293a4b5c6d7e8f9>", "dept": "Sales"}, "find Sales"),
        (Req2, {"name": "find_book", "dept": "Sales"}, "find_book Sales"),
        (Req2, {"name": "cmd", "dept": "Ctx/cmd"}, "cmd Ctx/cmd"),
        (Req2, {"name": "Find_Book", "dept": NF}, "Find_Book"),
        (Optional1, {"title": "list_books", "name": "Ann"}, "Ann list_books"),
        (Req2, {"name": "he", "dept": "book"}, "he is in book"),
        (Req2, {"name": "that member", "dept": "Sales"}, "that member Sales"),
        (Req2, {"name": "it", "dept": "Sales"}, "find it in Sales"),
        (Req2, {"name": "name", "dept": "Sales"}, "find name Sales"),
        (WithSource, {"uid": NF, "other": NF, "plain": NF}, "show it"),
        (WithSource, {"uid": NF, "other": "o", "plain": "p"}, "show it"),
        (WithSource, {"uid": "u", "other": "o", "plain": NF}, "show u o"),
        (Pattern, {"code": "bad", "tag": "bad"}, "code bad"),
        (Pattern, {"code": "XY-1", "tag": "#ok"}, "code XY-1 #ok"),
        (Pattern, {"code": NF, "tag": "nope"}, "tag nope"),
        (Typed, {"count": "x", "ratio": "y", "flag": "maybe", "kind": "film"}, "do it"),
        (Typed, {"count": 3, "ratio": 1.5, "flag": True, "kind": "book"}, "do 3 1.5 book"),
        (Typed, {"count": INT_NF, "ratio": "1", "flag": "true", "kind": "member"}, "do"),
        (Typed, {"count": "x", "ratio": 1.5, "flag": True, "kind": "book"}, "x 1.5"),
        (Bounded, {"limit": -3, "page_size": 100, "name": ""}, "limit -3"),
        (Bounded, {"limit": 5, "page_size": 0, "name": "n"}, "page 0"),
        (Bounded, {"limit": 5, "page_size": 5, "name": "n"}, "n"),
        (Listed, {"items": [], "opt": None}, "list"),
        (Listed, {"items": ["a"], "opt": []}, "list a"),
        (Optional1, {"title": NF, "name": NF}, "x"),
        (Optional1, {"title": "t", "name": NF}, "x"),
        (Rule, {"a": "bad"}, "bad"),
        (Rule, {"a": "both"}, "both"),
        (Rule, {"a": "nofield"}, "nofield"),
        (Rule, {"a": "boom"}, "boom"),
        (Rule, {"a": NF}, "x"),
        (Rule, {"a": "fine"}, "fine"),
        (Fv, {"qty": 3}, "3"),
        (corpus.ExecStr, {"name": NF}, "no name"),
        (corpus.ExecStr, {"name": "Ann", "code": "ZZ-99"}, "Ann"),
        (corpus.ExecStr, {"name": "Ann", "shelf": "S1"}, "Ann S1"),
        (corpus.ExecStr, {"name": "Ann", "enum_s": "movie"}, "Ann movie"),
        (corpus.ExecNum, {"count": 7, "title": "T"}, "show 3 things"),
        (corpus.ExecNum, {"kind": "film"}, "x"),
        (corpus.ExecOptions, {"copied": "zzz", "qty": 4}, "copy something 3"),
        (corpus.ExecOptions, {"named": "Nobody", "items": ["q"]}, "named Ann"),
        (corpus.ChoiceText, {"kind_text": "movie"}, "movie"),
    ]
    n = 0
    for model, values, request in cases:
        for audience in ("human", "agent"):
            ctx = {} if audience == "human" else {"run_as_agent": True}
            if audience == "agent":
                ctx["raw_user_message"] = request
            wf = types.SimpleNamespace(context=ctx, folderpath=folder)
            for diag in (False, True):
                diagnostics = {} if diag else None
                key = f"{model.__name__}|{json.dumps(values, sort_keys=True, default=str)}|{request}|{audience}|diag={diag}"

                def run():
                    params = model.model_construct(**{k: v for k, v in values.items()})
                    extraction = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
                    result = extraction.validate_parameters(wf, "Ctx/cmd", params, diagnostics=diagnostics)
                    return {"result": norm(list(result)), "diagnostics": norm(diagnostics)}
                out[key] = safe(run)
                n += 1
    return n


# ------------------------------------------------------------------------------------------------ the extraction frame

@contextlib.contextmanager
def _scripted(extractor_class, definition, fake_llm):
    """The routing registry answers with *definition* and the extractor's model call with *fake_llm*, for the block."""
    registry, extract = fastworkflow.RoutingRegistry, extractor_class.extract_parameters
    fastworkflow.RoutingRegistry = types.SimpleNamespace(get_definition=lambda *args, **kwargs: definition)
    extractor_class.extract_parameters = fake_llm
    try:
        yield
    finally:
        fastworkflow.RoutingRegistry, extractor_class.extract_parameters = registry, extract


def frame_cases(tmp: Path):
    """_extract_impl driven with a fake extractor: the framing of the question around the validation message."""
    from fastworkflow import ModuleType
    from fastworkflow._workflows.command_metadata_extraction import parameter_extraction as pe
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction
    from fastworkflow.parameters.signatures import InputForParamExtraction

    out = TABLE["frame"]
    n = 0
    models = {"Req2": Req2, "WithSource": WithSource, "Optional1": Optional1, "Typed": Typed, "Pattern": Pattern, "Rule": Rule}
    scenarios = [
        # (model, llm values, first text, agent ctx, replies)
        ("Req2", {"name": NF, "dept": NF}, "find a person", "human", ["Ann", "Ann, Sales", "name: Ann\ndept: Sales", "a, b, c", "Smith, John", "x"]),
        ("Req2", {"name": "Ann", "dept": NF}, "find Ann", "human", ["Sales", "Sales, Eng", "", "dept: Eng"]),
        ("Req2", {"name": NF, "dept": NF}, "find a person", "agent", ["Ann, Sales"]),
        ("Req2", {"name": NF, "dept": NF}, "find a person", "agent_false", []),
        ("WithSource", {"uid": NF, "other": "o", "plain": "p"}, "show o p", "human", ["u1"]),
        ("WithSource", {"uid": NF, "other": "o", "plain": "p"}, "show o p", "agent", []),
        ("Optional1", {"title": NF, "name": NF}, "x", "human", ["Ann"]),
        ("Typed", {"count": "x", "ratio": 1.5, "flag": True, "kind": "film"}, "do it", "human", ["3", "3, book"]),
        ("Pattern", {"code": "bad", "tag": "#t"}, "code bad", "human", ["XY-1"]),
        ("Rule", {"a": "bad"}, "bad", "human", ["fine", "bad"]),
        ("Req2", {"name": "Zed Zed", "dept": "Qq"}, "find someone", "human", []),     # values the request does not write
        ("Req2", {"name": "Zed Zed", "dept": "Qq"}, "find someone", "agent", []),
    ]
    state = {"cls": None, "llm": None}

    class Definition:
        command_directory = types.SimpleNamespace(map_command_2_metadata={})

        @staticmethod
        def get_command_class(name, kind):
            return state["cls"] if kind == ModuleType.COMMAND_PARAMETERS_CLASS else None

    def fake_llm(self, cls, command_name, folder):
        return cls.model_construct(**state["llm"])

    with _scripted(InputForParamExtraction, Definition, fake_llm):
        for model_name, llm, text, audience, replies in scenarios:
            cls = models[model_name]
            for reply in [None, *replies]:
                cme = types.SimpleNamespace(context={})
                app_ctx = {}
                if audience == "agent":
                    app_ctx = {"run_as_agent": True}
                elif audience == "agent_false":
                    app_ctx = {"run_as_agent": False}
                app = types.SimpleNamespace(context=app_ctx, folderpath=str(tmp / "tutorial"))
                state["cls"], state["llm"] = cls, llm
                key = f"{model_name}|{json.dumps(llm, sort_keys=True)}|{text}|{audience}|reply={reply!r}"

                def run():
                    steps = []
                    extractor = ParameterExtraction(cme, app, "Ctx/cmd", text)
                    diagnostics: dict = {}
                    first = extractor._extract_impl(diagnostics)
                    steps.append(frame_record(first, cme, diagnostics))
                    if reply is not None:
                        extractor2 = ParameterExtraction(cme, app, "Ctx/cmd", reply)
                        diagnostics2: dict = {}
                        second = extractor2._extract_impl(diagnostics2)
                        steps.append(frame_record(second, cme, diagnostics2))
                    return steps
                out[key] = safe(run)
                n += 1
    return n


def frame_record(output, cme, diagnostics):
    return {"valid": output.parameters_are_valid, "error_msg": output.error_msg, "suggestions": norm(output.suggestions),
            "parameter_request": norm(output.parameter_request), "params": norm(output.cmd_parameters.model_dump())
            if output.cmd_parameters is not None else None,
            "loop": norm(loop.to_state() if (loop := pcl.pending(cme.context)) is not None else None),
            "diag": norm({k: v for k, v in diagnostics.items() if k in (
                "missing_fields", "invalid_fields", "extraction_method", "argument_check_refusals_final", "retry_round",
                "retry_round_ordinal")})}


# ------------------------------------------------------------------------------------------------ reading

REPLIES = [
    "", " ", "   \n  ", "Dune", "  Dune  ", "title: Dune", "Title: Dune", "TITLE:Dune", "title=Dune", "title = Dune",
    "title: Dune\nauthor: Herbert", "author: Herbert\ntitle: Dune", "title: Dune\nauthor:", "title:", "title: Dune\nnotes",
    "title: Dune, author: Herbert", "title: Dune,author: Herbert", "title: Dune, Herbert", "unknown: x", "title: Dune\nunknown: x",
    "unknown: x\nunknown2: y", "Dune, Herbert", "Dune,Herbert", "Dune, Herbert, 1965", "Dune, Herbert, 1965, extra", "Smith, John",
    "Dune,", ",Dune", "Dune,,Herbert", "a, , b", ",", ",,", "\"Dune\"", "'Dune'", "\"Dune, Part One\"", "Dune and also show members",
    "Dune. Then count the overdue loans", "yes", "Yes", "y", "ok", "okay", "run", "run it", "go", "confirm", "true", "yes please",
    "no", "No.", "none", "neither", "n", "cancel", "Cancel", "cancel!", "abort", "stop", "decline", "never mind", "nevermind",
    "Never  Mind", "something else", "I do not know", "hmm, I am not sure", "I think it is Dune, maybe", "the book Dune",
    "Dune\n", "\nDune", "Dune\nHerbert", "Dune\n\nHerbert", "- Dune", "title: Dune; author: Herbert", "title: a: b",
    "title: Dune\nTitle: Other", "12", "3, 4", "1.5", "-1", "  title :  Dune  ", "_x: 1", "1x: 1", "title: \"Dune\"", "name: Ann, dept: Sales",
    "name: Ann,dept: Sales", "Ann, Sales, Eng", "Ann, Sales", "Ann Sales", "dept: Sales", "name: Ann", "Name: Ann\nDEPT: Sales",
    # added by step 6 (the one reading rule)
    "nope", "NONE", "Yes", "Y", "fast", "SLOW", "maybe", "'Dune, Part One'", "\"Dune, Part One\", Herbert",
    "\"Dune, Part One\", \"Herbert, Frank\"", "title: \"Dune, Part One\"", "No, thanks",
    # added by the rule-e guard (parts must fit): prose, unlisted named parts, typed parts, multi-word values in order
    "Jane Austen, Pride and Prejudice", "x: 1, y: 2", "yes, 3", "maybe, 3", "no, three", "FAST, red", "slow, 2.5",
    "Dune: Part Two, Herbert", "I will, I think", "yes, 12",
]
CONFIGS = [
    ("one_open", ["title", "author", "year"], ["title"], {}),
    ("one_open_confirm", ["title", "author", "year"], ["title"], {"confirm": True}),
    ("two_open", ["title", "author", "year"], ["title", "author"], {}),
    ("two_open_positional", ["title", "author", "year"], ["title", "author"], {"positional": True}),
    ("two_open_confirm", ["title", "author", "year"], ["title", "author"], {"confirm": True}),
    ("three_open_positional", ["title", "author", "year"], ["title", "author", "year"], {"positional": True}),
    ("none_open", ["title", "author", "year"], [], {}),
    ("none_open_confirm", ["title", "author", "year"], [], {"confirm": True}),
    ("dept_two", ["name", "dept"], ["name", "dept"], {"positional": True}),
]
#: Field facts for rule c (step 6): (label, names, open names, slots).
SLOT_NAMES = ["flag", "mode", "note", "tag", "count"]
SLOT_CONFIGS = [
    ("bool_open", ["flag"], {"flag": Slot(boolean=True)}),
    ("enum_open", ["mode"], {"mode": Slot(options=("fast", "slow"))}),
    ("optional_open", ["note"], {"note": Slot(required=False)}),
    ("required_open", ["tag"], {"tag": Slot()}),
    ("bool_and_required_open", ["flag", "tag"], {"flag": Slot(boolean=True), "tag": Slot()}),
    ("bool_and_number_open", ["flag", "count"], {"flag": Slot(boolean=True), "count": Slot(number=True)}),
    ("enum_and_required_open", ["mode", "tag"], {"mode": Slot(options=("fast", "slow")), "tag": Slot()}),
    ("bool_open_note_shown", ["flag"], {"flag": Slot(boolean=True), "note": Slot(required=False)}),
]


def read_reply_cases():
    out = TABLE["read_reply"]
    n = 0
    for reply in REPLIES:
        for label, names, open_names, kwargs in CONFIGS:
            key = f"{reply!r}|{label}"
            kwargs = {k: v for k, v in kwargs.items() if k != "positional"}   # the argument is gone (step 6)
            out[key] = safe(lambda: list(pcl.read_reply(reply, names, open_names, **kwargs)))
            n += 1
        for label, open_names, slots in SLOT_CONFIGS:
            key = f"{reply!r}|slots:{label}"
            out[key] = safe(lambda: list(pcl.read_reply(reply, SLOT_NAMES, open_names, slots=slots)))
            n += 1
        out[f"{reply!r}|named_values"] = norm(pcl.named_values(reply, ["title", "author", "Year"]))
        n += 1
    return n


class _M:
    pass


def merge_models():
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction  # noqa: F401

    class One(BaseModel):
        title: str = Field(description="t")
        author: str = Field(default="a", description="a")

    class Two(BaseModel):
        title: str = Field(description="t")
        author: str = Field(description="a")
        year: int = Field(default=0, description="y")

    class Three(BaseModel):
        title: str = Field(description="t")
        author: str = Field(description="a")
        year: str = Field(description="y")

    class OptFirst(BaseModel):
        filter: Optional[str] = Field(default=None, description="f")
        name: str = Field(description="n")
        dept: str = Field(description="d")

    class OnlyOptional(BaseModel):
        filter: Optional[str] = Field(default=None, description="f")
        other: Optional[str] = Field(default=None, description="o")

    class Mixed(BaseModel):
        tags: Optional[List[str]] = Field(default=None)
        start: Optional[int] = Field(default=None)
        count: int = Field(description="c")

    class BoolOne(BaseModel):
        active: bool = Field(description="a")

    class ModeOne(BaseModel):
        mode: Literal["fast", "slow"] = Field(description="m")

    class OptOne(BaseModel):
        tag: Optional[str] = Field(default=None, description="t")

    class ReqOne(BaseModel):
        tag: str = Field(description="t")

    class BoolInt(BaseModel):
        active: bool = Field(description="a")
        count: int = Field(description="c")

    class BoolAndOpt(BaseModel):
        active: bool = Field(description="a")
        tag: Optional[str] = Field(default=None, description="t")

    return [
        ("BoolOne", BoolOne, {"active": NF}),
        ("ModeOne", ModeOne, {"mode": NF}),
        ("OptOne", OptOne, {"tag": NF}),
        ("ReqOne", ReqOne, {"tag": NF}),
        ("BoolInt", BoolInt, {"active": NF, "count": INT_NF}),
        ("BoolAndOpt", BoolAndOpt, {"active": NF, "tag": None}),
        ("One", One, {"title": NF, "author": "keep"}),
        ("Two", Two, {"title": NF, "author": NF, "year": 7}),
        ("Two_year_open", Two, {"title": "keep", "author": NF, "year": INT_NF}),
        ("Three", Three, {"title": NF, "author": NF, "year": NF}),
        ("OptFirst", OptFirst, {"filter": None, "name": NF, "dept": NF}),
        ("OptFirst_one", OptFirst, {"filter": None, "name": "keep", "dept": NF}),
        ("OptFirst_invalid_opt", OptFirst, {"filter": NF, "name": "keep", "dept": "keep"}),
        ("OnlyOptional", OnlyOptional, {"filter": None, "other": None}),
        ("Mixed", Mixed, {"tags": None, "start": None, "count": INT_NF}),
        ("Nothing", Three, {"title": "a", "author": "b", "year": "c"}),
    ]


def merge_cases():
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import ParameterExtraction

    out = TABLE["merge"]
    n = 0
    for label, cls, values in merge_models():
        for reply in REPLIES + ["a, b", "x, y, z", "name: Ann, dept: Sales", "Ann, Sales", "filter: x", "year: 1965, title: Dune", "active: no", "tag: x"]:
            key = f"{label}|{reply!r}"
            out[key] = safe(lambda: norm(ParameterExtraction._extract_and_merge_missing_parameters(
                cls.model_construct(**values), reply).model_dump()))
            n += 1
    return n


def build_table(tmp: Path) -> dict:
    """The table for the code as it is, under *tmp* (a scratch directory), as plain JSON."""
    for part in TABLE.values():
        part.clear()
    tmp_commands = setup(tmp)
    commands = tmp_commands[1]
    counts = {"plan_wording_scenarios": plan_wording(commands), "exec_wording": exec_wording(tmp, commands),
              "frame": frame_cases(tmp), "read_reply": read_reply_cases(), "merge": merge_cases()}
    counts["wording_plan_entries"] = len(TABLE["wording_plan"])
    return json.loads(json.dumps({**TABLE, "counts": counts}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    import os
    import sys
    import tempfile

    os.environ.pop("JEV_API_KEY", None)

    Path(sys.argv[1]).write_text(
        json.dumps(build_table(Path(tempfile.mkdtemp(prefix="ask_read_"))), ensure_ascii=False, default=str, separators=(",", ":")),
        encoding="utf-8")
