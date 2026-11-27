"""``from_command_output`` is written ``"<command>.<field>"`` (bd fix-qjne.4).

A bare command name fails the load of the command that wrote it, and so does any declaration that does not resolve
(no such command, a command off the consumer's lookup path, a field the producer's Output lacks, a field of a type
that does not fit). The failing command is a problem line of ``command_preview.load_problems`` (what ``fastworkflow
run``, ``train`` and the server print) and is not available, while the rest of the workflow loads. A call cycle
between planning commands is a different kind of error and behaves as it did: the commands still load.
"""
from __future__ import annotations

import os

import pytest
from dotenv import dotenv_values
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import ModuleType, workflow_agent
from fastworkflow.planning import command_preview
from fastworkflow.planning import planning_commands as pc
from fastworkflow.core.command_routing import RoutingRegistry
from tests.test_command_preview import AGAIN, PING, PONG
from tests.test_field_sources import ROOT, _build_workflow


# ------------------------------------------------------------------ the resolver (unit)

class _FindBookOutput(BaseModel):
    book_uids: list[str] = Field(description="the uids of the matching books")
    titles: list[str] = Field(description="the titles of the matching books")
    total: int = Field(description="how many match")


def _resolve(consumer_input_cls, producer_output_cls=_FindBookOutput):
    class Producer:
        """Find books."""

        class Input(BaseModel):
            query: str = Field(default="", description="the query")

        Output = producer_output_cls

    class Consumer:
        """Open a book."""

        Input = consumer_input_cls

    resolution = pc.FieldSourceResolution(
        workflow_folderpath="", problems=(), skipped_commands=frozenset(), model_to_command={},
        signatures={"Library/find_book": Producer, "Library/open_book": Consumer}, command_files={},
        import_errors={}, base_contexts={})
    return Consumer, pc.resolve_signature_field_sources("Library/open_book", Consumer, "open_book.py", resolution)


def test_a_bare_command_name_is_an_error_that_names_the_field_and_the_fitting_output_fields():
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"from_command_output": "find_book"})

    consumer, problems = _resolve(Input)
    assert [p.severity for p in problems] == ["error"]
    message = problems[0].message
    assert message.startswith("Input.book_uid (from 'find_book'): ")
    assert 'write "<command>.<field>"' in message
    assert "book_uids" in message and "titles" in message     # the Output fields of a fitting type
    assert "total" not in message                              # an int does not fit a str input
    assert "available_from" not in (consumer.Input.model_fields["book_uid"].json_schema_extra or {})


def test_a_list_with_one_bare_entry_is_an_error_for_that_entry_only():
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"from_command_output": ["find_book.book_uids", "find_book"]})

    _, problems = _resolve(Input)
    assert len(problems) == 1 and "(from 'find_book')" in problems[0].message, problems
    assert 'write "<command>.<field>"' in problems[0].message


def test_a_bare_name_whose_producer_has_no_output_field_of_a_fitting_type_says_so():
    class Output(BaseModel):
        total: int = Field(description="how many match")

    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"from_command_output": "find_book"})

    _, problems = _resolve(Input, Output)
    assert len(problems) == 1
    assert 'write "<command>.<field>"' in problems[0].message
    assert "no Output field of a type that fits" in problems[0].message


@pytest.mark.parametrize("spelled", ["find_book.book_uids", "Library/find_book.book_uids",
                                     ["find_book.book_uids", "Library/find_book.titles"]])
def test_the_command_dot_field_spellings_still_load(spelled):
    class Input(BaseModel):
        book_uid: str = Field(default="", json_schema_extra={"from_command_output": spelled})

    consumer, problems = _resolve(Input)
    assert problems == []
    assert list(consumer.Input.model_fields["book_uid"].json_schema_extra["available_from"])


# ------------------------------------------------------------------ the load (integration)

def _open_book(source: str, annotation: str = "str") -> str:
    return f'''import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Open one book by its uid."""

    class Input(BaseModel):
        book_uid: {annotation} = Field(description="the uid of the book to open",
                                       json_schema_extra={{"from_command_output": {source}}})

    class Output(BaseModel):
        title: str = Field(description="the title")

    plain_utterances = ["open book b1"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        return Signature.Output(title="Dune")
'''


_BAD_DECLARATIONS = {
    "bare": ('"find_book"', "str", 'write "<command>.<field>"'),
    "bare_in_a_list": ('["find_book.book_uids", "find_book"]', "str", 'write "<command>.<field>"'),
    "no_such_command": ('"find_boook.book_uids"', "str", "no command named 'find_boook'"),
    "not_reachable": ('"show_member.member_uid"', "str", "Member/show_member is in another context. Write the qualified name only when"),
    "no_such_field": ('"find_book.book_uid_list"', "str", "has no Output field 'book_uid_list'"),
    "type_does_not_fit": ('"find_book.total"', "str", "does not match"),
}


@pytest.fixture(autouse=True)
def _init():
    fastworkflow.init(env_vars=dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env"))))
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()


@pytest.mark.parametrize("case", sorted(_BAD_DECLARATIONS))
def test_an_execution_command_with_a_bad_producer_declaration_is_a_load_problem_and_not_available(tmp_path, case):
    source, annotation, reason = _BAD_DECLARATIONS[case]
    folder = _build_workflow(tmp_path, f"wf_{case}", extra_commands={"Library/open_book": _open_book(source, annotation)})
    problems = command_preview.load_problems(folder)
    mine = [p for p in problems if p.startswith("Library/open_book: ")]
    assert len(mine) == 1 and len(problems) == 1, problems
    assert "open_book.py" in mine[0] and "Input.book_uid" in mine[0] and reason in mine[0], mine[0]
    with pytest.raises(Exception, match="Input.book_uid"):
        RoutingRegistry.get_definition(folder).get_command_class(
            "Library/open_book", ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    # the rest of the workflow loads
    assert RoutingRegistry.get_definition(folder).get_command_class(
        "Library/find_book", ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS) is not None


def test_a_good_declaration_is_not_a_load_problem(tmp_path):
    folder = _build_workflow(tmp_path, "wf_good", extra_commands={
        "Library/open_book": _open_book('["find_book.book_uids", "Library/find_book.titles"]')})
    assert command_preview.load_problems(folder) == []


_INSPECT = '''from pydantic import BaseModel, Field


class Signature:
    """Inspect a book by uid."""

    class Input(BaseModel):
        book_uid: str = Field(description="the uid", json_schema_extra={"from_command_output": SOURCE})


class PlanGenerator:
    def steps(self, input):
        return ["library_hours", "say when the library is open"]
'''


@pytest.mark.parametrize("source,reason", [
    ('"find_book"', 'write "<command>.<field>"'),
    ('"find_boook.book_uids"', "no command named 'find_boook'"),
    ('"find_book.nope"', "has no Output field 'nope'"),
    ('"find_book.total"', "does not match"),
])
def test_a_planning_command_with_a_bad_producer_declaration_is_a_load_problem_and_not_loaded(tmp_path, source, reason):
    folder = _build_workflow(tmp_path, "wf_plan", extra_planning_commands={
        "Library/inspect_book": _INSPECT.replace("SOURCE", source)})
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].startswith("Library/inspect_book: "), problems
    assert "inspect_book.py" in problems[0] and "Input.book_uid" in problems[0] and reason in problems[0], problems
    assert "Library/inspect_book" not in workflow_agent.load_planning_command_map(folder)
    assert "Library/hours_overview" not in problems[0] and "hours_overview" in workflow_agent.load_planning_command_map(folder)


def test_a_call_cycle_fails_the_load_of_every_command_on_it(tmp_path):
    """One rule for an author's mistake (bd fix-ftdl.2): ping, pong and again are on call cycles, so none of them
    loads and each is listed; the rest of the workflow loads. (They used to load, the cycle only logged.)"""
    folder = _build_workflow(tmp_path, "wf_cycle", extra_planning_commands={"ping": PING, "pong": PONG, "again": AGAIN})
    res = pc.resolve_field_sources(folder)
    assert {"ping", "pong", "again"} <= {e.command for e in res.errors}
    loaded = set(workflow_agent.load_planning_command_map(folder))
    assert not {"ping", "pong", "again"} & loaded and "hours_overview" in loaded
    problems = command_preview.load_problems(folder)
    assert sorted(p.split(": ", 1)[0] for p in problems) == ["again", "ping", "pong"], problems
    assert all("a planning command cannot call itself" in p for p in problems)
    assert "ping -> pong -> ping" in next(p for p in problems if p.startswith("ping: "))
