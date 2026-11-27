"""Two training-only author declarations on an ``Input`` field (bd fix-0iuo.16).

``classifier_confirms`` ``words`` (the words users write for each value of a ``Literal``) and ``examples_by``
(example values of a field per value of a choice field of the same ``Input``) are read by the synthetic utterance
generator only. These tests show that both load on a planning command and an execution command, that each load rule
fails with its message, that misspellings are still load problems, and that NOTHING read at run time changes: the
check ``Param`` the classifier check reads, the extraction prompt, the command listing and the command preview are
byte-identical with and without them. Where they do show: the field's JSON Schema (``GET /command_schemas``).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.planning import planning_commands as pc
from fastworkflow.command_routing import RoutingRegistry

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"
HELLO = Path(__file__).resolve().parent / "hello_world_workflow"
_COPY_IGNORE = shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__")

KIND = Literal["identity", "group"]
MEANINGS = {"identity": "one person", "group": "a set of people"}
WORDS = {"identity": ["person", "employee"], "group": ["team", "role"]}
EXAMPLES_BY = {"kind": {"identity": ["Maria Okafor", "Tom Reyes"], "group": ["Payroll Approvers EMEA"]}}


def _input(kind_extra: dict, name_extra: dict, *, name_annotation=str, name_field: Optional[dict] = None):
    """An ``Input`` with a choice field ``kind`` and a dependent field ``name``."""
    name_kwargs = dict(name_field or {"default": "", "description": "who or which group"})
    return type("Input", (BaseModel,), {
        "__annotations__": {"kind": KIND, "name": name_annotation},
        "kind": Field(default="identity", description="a person or a group", json_schema_extra=kind_extra),
        "name": Field(examples=["Maria Okafor", "Payroll Approvers EMEA"], json_schema_extra=name_extra,
                      **name_kwargs),
    })


def _load_error(model, planning: bool) -> str:
    with pytest.raises(pc.PlanningCommandFormatError) as caught:
        pc.option_params(model, "app/grant", planning=planning)
    return str(caught.value)


# ---------------------------------------------------------------- they load, and the check reads the same Param

@pytest.mark.parametrize("planning", [True, False])
def test_words_and_examples_by_load_and_leave_the_check_param_unchanged(planning):
    plain = _input({"classifier_confirms": {"meanings": MEANINGS}}, {"copied_from_command_text": True})
    declared = _input({"classifier_confirms": {"meanings": MEANINGS, "words": WORDS}},
                      {"copied_from_command_text": True, "examples_by": EXAMPLES_BY})
    assert pc.option_params(declared, "app/grant", planning=planning) == pc.option_params(
        plain, "app/grant", planning=planning)
    [kind, _] = pc.option_params(declared, "app/grant", planning=planning)
    assert (kind.verify, kind.alone, kind.means) == (True, False, tuple(MEANINGS.items()))


@pytest.mark.parametrize("planning", [True, False])
def test_words_without_meanings_is_the_check_classifier_confirms_true_gives(planning):
    true = _input({"classifier_confirms": True}, {})
    words = _input({"classifier_confirms": {"words": WORDS}}, {})
    assert pc.option_params(words, "app/grant", planning=planning) == pc.option_params(
        true, "app/grant", planning=planning)
    assert pc.option_params(words, "app/grant", planning=planning)[0].means == ()


def test_examples_by_alone_makes_no_check_param():
    model = _input({}, {"examples_by": EXAMPLES_BY})
    assert pc.option_params(model, "app/grant", planning=True) == ()
    assert pc.option_params(model, "app/grant") == ()


def test_a_value_may_have_no_words_and_no_examples():
    model = _input({"classifier_confirms": {"words": {"group": ["team"]}}},
                   {"examples_by": {"kind": {"identity": ["Maria Okafor"]}}})
    assert len(pc.option_params(model, "app/grant", planning=True)) == 1


def test_examples_by_may_name_an_optional_choice_field_on_an_execution_command():
    model = type("Input", (BaseModel,), {
        "__annotations__": {"kind": Optional[KIND], "count": int},
        "kind": Field(default=None),
        "count": Field(default=1, ge=1, json_schema_extra={"examples_by": {"kind": {"group": [3, 12]}}}),
    })
    assert pc.option_params(model, "app/grant") == ()


# ---------------------------------------------------------------- the load rules of words, one test each

@pytest.mark.parametrize("planning", [True, False])
@pytest.mark.parametrize("words, message", [
    ({"person": ["someone"]}, "field 'kind': classifier_confirms words: 'person' is not a value of it"),
    ({"identity": []}, "classifier_confirms words for 'identity' must be a non-empty list of words; got []"),
    ({"identity": "person"}, "classifier_confirms words for 'identity' must be a non-empty list of words"),
    ({"identity": ["person", " "]}, "classifier_confirms words for 'identity': \" \" is not a word (non-empty text)"),
    ({"identity": ["person", 3]}, "classifier_confirms words for 'identity': 3 is not a word"),
    ({"identity": ["Person", "person"]},
     "classifier_confirms words for 'identity' list 'Person' and 'person', the same word (compared ignoring case)"),
    ({"identity": ["member"], "group": ["Member"]},
     "classifier_confirms words: 'Member' is listed for both 'identity' and 'group'; a word stands for one value only"),
    ({}, "classifier_confirms words must be a dict from a value to the words users write for it"),
    (["person"], "classifier_confirms words must be a dict from a value to the words users write for it"),
])
def test_each_words_rule_fails_the_load_with_its_message(words, message, planning):
    model = _input({"classifier_confirms": {"words": words}}, {})
    assert message in _load_error(model, planning)


@pytest.mark.parametrize("planning", [True, False])
def test_words_on_a_field_that_is_not_a_literal_fails_the_load(planning):
    model = type("Input", (BaseModel,), {
        "__annotations__": {"name": str},
        "name": Field(default="", json_schema_extra={"record_named_in_command_text": True,
                                                       "classifier_confirms": {"words": {"a": ["b"]}}}),
    })
    assert ("option classifier_confirms words does not apply to field 'name' of type str (it takes a Literal)"
            in _load_error(model, planning))


# ---------------------------------------------------------------- the load rules of examples_by, one test each

@pytest.mark.parametrize("planning", [True, False])
@pytest.mark.parametrize("examples_by, message", [
    ({"role": {"identity": ["Maria"]}},
     "field 'name': examples_by names 'role', which is not a field of this Input; its fields are 'kind', 'name'"),
    ({"name": {"identity": ["Maria"]}},
     "examples_by names this field itself ('name'); name the choice field its examples depend on"),
    ({"kind": {"person": ["Maria"]}}, "examples_by kind: 'person' is not a value of 'kind'"),
    ({"kind": {"identity": []}}, "examples_by kind: the examples for 'identity' must be a non-empty list; got []"),
    ({"kind": {"identity": "Maria"}}, "examples_by kind: the examples for 'identity' must be a non-empty list"),
    ({"kind": {"identity": [7]}},
     "examples_by kind: the example 7 for 'identity' is not a valid value of this field (str)"),
    ({"kind": {"identity": [None]}}, "examples_by kind: an example for 'identity' is None"),
    ({"kind": {}}, "examples_by for 'kind' must be a non-empty dict from a value of 'kind' to a list of examples"),
    ({}, "examples_by must be a dict from the name of a choice (Literal) field of this Input"),
    (["Maria"], "examples_by must be a dict from the name of a choice (Literal) field of this Input"),
])
def test_each_examples_by_rule_fails_the_load_with_its_message(examples_by, message, planning):
    model = _input({}, {"examples_by": examples_by})
    assert message in _load_error(model, planning)


@pytest.mark.parametrize("planning", [True, False])
def test_examples_by_must_name_a_choice_field(planning):
    model = type("Input", (BaseModel,), {
        "__annotations__": {"team": str, "name": str},
        "team": Field(default=""),
        "name": Field(default="", json_schema_extra={"examples_by": {"team": {"x": ["Maria"]}}}),
    })
    assert ("examples_by names 'team', which is not a choice field (its type is str, not a Literal)"
            in _load_error(model, planning))


@pytest.mark.parametrize("planning", [True, False])
def test_examples_by_names_one_choice_field_at_most(planning):
    model = type("Input", (BaseModel,), {
        "__annotations__": {"kind": KIND, "scope": Literal["local", "global"], "name": str},
        "kind": Field(default="identity"),
        "scope": Field(default="local"),
        "name": Field(default="", json_schema_extra={"examples_by": {"kind": {"identity": ["Maria"]},
                                                                      "scope": {"local": ["Maria"]}}}),
    })
    assert ("examples_by names 2 choice fields ('kind', 'scope'); name one: examples that depend on several "
            "choices are not supported" in _load_error(model, planning))


@pytest.mark.parametrize("planning", [True, False])
def test_an_example_must_match_the_fields_pattern(planning):
    model = _input({}, {"examples_by": {"kind": {"group": ["payroll approvers"]}}},
                   name_field={"default": "", "pattern": r"^[A-Z]"})
    assert ("examples_by kind: the example 'payroll approvers' for 'group' is not a valid value of this field (str): "
            "String should match pattern '^[A-Z]'" in _load_error(model, planning))


def test_an_example_must_be_within_the_fields_bounds():
    model = type("Input", (BaseModel,), {
        "__annotations__": {"kind": KIND, "count": int},
        "kind": Field(default="identity"),
        "count": Field(default=1, ge=1, json_schema_extra={"examples_by": {"kind": {"group": [0]}}}),
    })
    assert ("examples_by kind: the example 0 for 'group' is not a valid value of this field (int): Input should be "
            "greater than or equal to 1" in _load_error(model, True))


# ---------------------------------------------------------------- misspellings are still load problems

@pytest.mark.parametrize("planning", [True, False])
def test_a_misspelt_examples_by_is_a_load_problem(planning):
    model = _input({}, {"example_by": EXAMPLES_BY})
    assert "unknown option 'example_by'; did you mean 'examples_by'?" in _load_error(model, planning)


@pytest.mark.parametrize("planning", [True, False])
def test_a_misspelt_words_inside_classifier_confirms_is_a_load_problem(planning):
    model = _input({"classifier_confirms": {"word": WORDS}}, {})
    assert "classifier_confirms has unknown option 'word'; did you mean 'words'?" in _load_error(model, planning)


def test_an_execution_commands_own_examples_key_is_not_read_as_a_misspelt_examples_by():
    model = _input({}, {"examples": ["Maria"]})
    assert pc.option_params(model, "app/grant") == ()
    assert "unknown option 'examples'" in _load_error(model, True)    # a planning command takes option keys only


# ---------------------------------------------------------------- the extraction prompt is byte-identical

def _extraction_prompt(model) -> list[dict]:
    """The messages the parameter extractor sends for *model* (no model call): the signature the extractor builds
    (``InputForParamExtraction.extract_parameters``) formatted by its adapter."""
    from fastworkflow.parameters import signatures

    signature = signatures.InputForParamExtraction.create_signature_from_pydantic_model(
        model, pc.option_sentences(model), pc.classifier_alone_fields(model))
    return signatures._TextLiteralsAsText().format(
        signature, demos=[], inputs={"statement": "give the payroll team access"})


@pytest.mark.parametrize("plain_kind, declared_kind", [
    ({"classifier_confirms": {"meanings": MEANINGS}}, {"classifier_confirms": {"meanings": MEANINGS, "words": WORDS}}),
    ({"classifier_confirms": True}, {"classifier_confirms": {"words": WORDS}}),
])
def test_the_extraction_prompt_is_byte_identical_with_and_without_the_declarations(plain_kind, declared_kind):
    plain = _input(plain_kind, {"copied_from_command_text": True})
    declared = _input(declared_kind, {"copied_from_command_text": True, "examples_by": EXAMPLES_BY})
    prompt = _extraction_prompt(plain)
    assert prompt == _extraction_prompt(declared)
    text = repr(prompt)
    for word in ("employee", "Tom Reyes", "examples_by", "words"):
        assert word not in text


# ---------------------------------------------------------------- real workflows: load, listing, preview, schema

def _library_copy(tmp_path, name: str, *, declared: bool) -> Path:
    """A copy of the library example whose planning command ``find_record`` confirms ``kind`` with meanings, and,
    when *declared*, also carries ``words`` on ``kind`` and ``examples_by`` on ``name``."""
    folder = tmp_path / name
    shutil.copytree(LIBRARY, folder, ignore=_COPY_IGNORE, symlinks=True)
    path = folder / "_commands" / "Library" / "find_record.py"
    source = path.read_text()
    words = ', "words": {"book": ["title", "paperback"], "member": ["patron", "cardholder"]}' if declared else ""
    kind_old = '            default="book", description="whether to look for a book or a member")\n'
    kind_new = ('            default="book", description="whether to look for a book or a member",\n'
                '            json_schema_extra={"classifier_confirms": {"meanings": {"book": "a book in the catalogue", '
                f'"member": "a library member"}}{words}}}}})\n')
    name_old = '                          json_schema_extra={"copied_from_command_text": True})\n'
    examples_by = (', "examples_by": {"kind": {"book": ["Dune Messiah", "Emma"], "member": ["Lee", "Ines Varga"]}}'
                   if declared else "")
    name_new = f'                          json_schema_extra={{"copied_from_command_text": True{examples_by}}})\n'
    assert source.count(kind_old) == 1 and source.count(name_old) == 1
    path.write_text(source.replace(kind_old, kind_new).replace(name_old, name_new))
    return folder


@pytest.fixture
def library_pair(tmp_path):
    fastworkflow.init({})
    yield _library_copy(tmp_path, "plain", declared=False), _library_copy(tmp_path, "declared", declared=True)
    RoutingRegistry.clear_registry()


def test_a_planning_command_with_the_declarations_loads_without_problems(library_pair):
    from fastworkflow.planning import command_preview
    from fastworkflow.workflow_agent import load_planning_command_map

    plain, declared = library_pair
    assert command_preview.load_problems(str(declared)) == command_preview.load_problems(str(plain)) == []
    assert "examples_by" in (declared / "_commands" / "Library" / "find_record.py").read_text()
    # the parsed command the planner, the typed call and the classifier check read is the same
    plain_command = load_planning_command_map(str(plain))["Library/find_record"]
    declared_command = load_planning_command_map(str(declared))["Library/find_record"]
    assert plain_command.params == declared_command.params


def test_the_command_listings_are_byte_identical(library_pair):
    from fastworkflow.command_metadata_api import CommandMetadataAPI

    cme = fastworkflow.get_internal_workflow_path("command_metadata_extraction")

    def listings(folder: Path) -> list[str]:
        RoutingRegistry.clear_registry()   # command classes are cached by name for the process
        return [
            CommandMetadataAPI.get_command_display_text(
                subject_workflow_path=str(folder), cme_workflow_path=cme, active_context_name="Library",
                for_agents=for_agents, with_templates=with_templates)
            for for_agents in (False, True) for with_templates in (False, True)
        ] + [
            CommandMetadataAPI.get_command_display_text_for_command(
                subject_workflow_path=str(folder), cme_workflow_path=cme, active_context_name="Library",
                qualified_command_name="Library/find_record", for_agents=for_agents)
            for for_agents in (False, True)
        ]

    plain, declared = library_pair
    plain_text, declared_text = listings(plain), listings(declared)
    assert plain_text == declared_text
    assert any("find_record" in text for text in plain_text)
    assert not any(w in text for text in declared_text for w in ("paperback", "cardholder", "Ines Varga", "examples_by"))


def test_the_command_preview_is_byte_identical(library_pair):
    from fastworkflow.planning import command_preview

    plain, declared = library_pair
    plain_previews = command_preview.render_previews(str(plain))
    assert plain_previews == command_preview.render_previews(str(declared))
    [find_record] = [text for name, text in plain_previews.items() if name.endswith("find_record.md")]
    assert "paperback" not in find_record and "Ines Varga" not in find_record


def _hello_copy(tmp_path, name: str, label_examples: str) -> Path:
    folder = tmp_path / name
    shutil.copytree(HELLO, folder, symlinks=True, ignore=_COPY_IGNORE)
    path = folder / "_commands" / "add_two_numbers.py"
    source = path.read_text()
    source = source.replace("from typing import Any, Dict, Optional\n",
                            "from typing import Any, Dict, Literal, Optional\n", 1)
    field_old = '        second_num: float = Field(description="Second number")\n'
    field_new = field_old + (
        '        kind: Literal["sum", "total"] = Field(default="sum", description="what to call the result",\n'
        '            json_schema_extra={"classifier_confirms": {"words": {"sum": ["add up"], "total": ["grand total"]}}})\n'
        '        label: str = Field(default="", description="a label for the result",\n'
        f'            json_schema_extra={{"examples_by": {{"kind": {{"total": {label_examples}}}}}}})\n')
    assert source.count(field_old) == 1 and "Literal" in source
    path.write_text(source.replace(field_old, field_new, 1))
    return folder


def test_an_execution_command_with_the_declarations_loads_and_its_schema_carries_them(tmp_path):
    from fastworkflow.planning import command_preview
    from fastworkflow.mcp_server import command_schema_listing

    folder = _hello_copy(tmp_path, "hello_declared", '["Q3 total", "Year to date"]')
    try:
        assert command_preview.load_problems(str(folder)) == []
        properties = command_schema_listing(str(folder))["add_two_numbers"]["input"]["properties"]
        # the JSON Schema (GET /command_schemas) is where the declarations show
        assert properties["kind"]["classifier_confirms"] == {"words": {"sum": ["add up"], "total": ["grand total"]}}
        assert properties["label"]["examples_by"] == {"kind": {"total": ["Q3 total", "Year to date"]}}
    finally:
        RoutingRegistry.clear_registry()


def test_an_execution_command_with_a_bad_example_is_a_load_problem_naming_it(tmp_path):
    from fastworkflow.planning import command_preview

    folder = _hello_copy(tmp_path, "hello_bad_example", "[2026]")
    try:
        problems = [str(p) for p in command_preview.load_problems(str(folder))]
    finally:
        RoutingRegistry.clear_registry()
    assert any("add_two_numbers" in p and re.search(
        r"field 'label': examples_by kind: the example 2026 for 'total' is not a valid value of this field \(str\)", p)
        for p in problems), problems


# ---------------------------------------------------------------- no runtime module reads them

def test_no_runtime_module_reads_the_declarations():
    """The two keys are named in the framework (the built-in workflows included, the bundled examples not) only where
    they are declared and checked at load: no other module looks up ``examples_by`` or a ``"words"`` key."""
    root = Path(fastworkflow.__file__).resolve().parent
    lookup = re.compile(r"""examples_by|EXAMPLES_BY|\[\s*["']words["']\s*\]|\.get\(\s*["']words["']""")
    readers = set()
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("examples/"):
            continue
        if lookup.search(path.read_text(encoding="utf-8")):
            readers.add(relative)
    # workflow_vocabulary's payload.get("words") reads its own cache file, unrelated to classifier_confirms
    assert readers == {"parameters/field_options.py", "planning/planning_commands.py", "parameters/workflow_vocabulary.py"}, readers
