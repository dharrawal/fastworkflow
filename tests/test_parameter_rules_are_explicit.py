"""The parameter pipeline does only what a field's own declarations say (bd fix-ftdl.6.1 to .6.6, fix-qjne.20).

Option B (fix-qjne.21): the extractor's value of a ``str`` or ``int`` stands unless the field is marked
``copied_from_command_text`` (or ``record_named_in_command_text``), the one explicit strict check. A developer reading a
command file can tell from the field's declarations what is checked and what the extraction model is told, and both
kinds of command do the same:

1. A value is refused as a placeholder only when it is ENTIRELY one template token (``<name>``, ``{name}``) or only
   punctuation; there is no list of stand-in words, and a value that merely contains a bracket is a value. An id
   written like ``SKU_42`` is a value. One function decides on every path (section 8). A description's
   "One of: a, b" exempts nothing from the vocabulary-word screen (allowed values are the ``Literal`` type).
2. The remote "was this value given" check is not asked about a marked field.
3. A marked field has ONE guard, the same function on both kinds; a marked optional ``str`` given and not written is
   refused, and one not given is not checked.
4. ``Field(pattern=...)`` is a full match on every path, checked after a marked id has been widened.
5. The sentences the extraction model reads about a field come from its declarations, the same for both kinds.
6. The strict ``str`` check matches at word boundaries ("dun" is not written in "find dune messiah").
7. One duplicate rule, part of the strict check: two marked fields may hold one value only when the request writes
   it as often; nothing is reset silently.

Real guards, loader, validators and extractor; the model boundary is a scripted ``DummyLM``. No model is called.
"""
from __future__ import annotations

import dataclasses

import re
from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from fastworkflow import planning_commands as pc
from fastworkflow.parameters import check, rules
from fastworkflow.parameters import workflow_vocabulary as wv
from tests.test_argument_checks import _command

COPIED = {"copied_from_command_text": True}
ACCOUNTS = (re.compile(r"\bacct_\d+\b"),)


class Plain(BaseModel):
    name: str = Field(default="", description="a name")


def _plan(tmp_path, fields: str):
    """A planning command whose ``Input`` has *fields* (several are separated by a newline and 8 spaces)."""
    module = ('from typing import Optional\n\nfrom pydantic import BaseModel, Field\n\n\nclass Signature:\n'
              '    """Weigh one title."""\n\n'
              f'    class Input(BaseModel):\n        {fields}\n\n\n'
              'class PlanGenerator:\n    def steps(self, input):\n        return ["say the weight"]\n')
    return _command(tmp_path, module, name="weigh")


# ---------------------------------------------------------------- 1. placeholders and the vocabulary exemption

@pytest.mark.parametrize("value", ["Adam Smith", "John Doe", "unknown", "foo", "unspecified", "a<b", "use {name} here",
                                   "Jane Austen", "placeholder text"])
def test_a_value_that_is_not_entirely_a_template_token_is_a_value_on_both_kinds(tmp_path, value):
    request = "find the author of the wealth of nations"
    assert check.check_execution(Plain, {"name": value}, request).gaps == []
    command = _plan(tmp_path, 'name: str = Field(description="n")')
    assert pc.check_arguments(command, {"name": value}, request)[0]["name"] == value


@pytest.mark.parametrize("value", ["<name>", "{name}", "{{name}}", "  {author}  "])
def test_a_value_that_is_entirely_one_template_token_is_refused_on_both_kinds(tmp_path, value):
    (gap,) = check.check_execution(Plain, {"name": value}, "find the member").gaps
    assert gap.code == "placeholder"
    command = _plan(tmp_path, 'name: str = Field(description="n")')
    with pytest.raises(pc.PlanningCommandInputError):
        pc.check_arguments(command, {"name": value}, "find the member")


def test_the_placeholder_word_list_is_gone_and_the_rule_reads_no_request():
    assert not hasattr(rules, "PLACEHOLDER_WORDS")
    assert rules.placeholder_like("<name>") and not rules.placeholder_like("John Doe")


def test_a_description_that_lists_kind_words_exempts_nothing_from_the_vocabulary_screen(tmp_path):
    command = _plan(tmp_path, 'kind: str = Field(description="the kind of record. One of: group, account")')
    vocabulary = wv.WorkflowVocabulary(words=frozenset({"group", "account"}), pronouns=wv.PRONOUNS,
                                       source_fingerprint="x", origin="derived",
                                       kind_words=frozenset({"group", "account"}))
    with pytest.raises(pc.PlanningCommandInputError, match="kind word"):
        pc.check_arguments(command, {"kind": "group"}, "show the group Payroll", None, vocabulary)
    # the vocabulary's kind selectors are not read from a description at all
    assert "field_descriptions" not in wv.WorkflowMetadata.__dataclass_fields__
    assert wv.FORMAT_VERSION == 3


def test_the_pronoun_screen_stays_and_its_question_says_why(tmp_path):
    command = _plan(tmp_path, 'name: str = Field(description="n")')
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.check_arguments(command, {"name": "it"}, "show it")
    assert refused.value.gap.screened == "'it' " + pc.PRONOUN_WHY


# ---------------------------------------------------------------- 2. the remote "was it given" check skips a marked field

def test_the_remote_given_check_is_not_asked_about_a_marked_field(monkeypatch):
    import dspy
    from dspy.utils import DummyLM

    import fastworkflow
    from fastworkflow import workflow_agent
    from fastworkflow.parameters.signatures import fields_not_given
    from tests.test_jev_field_given import _Recorder

    class Input(BaseModel):
        title: str = Field(description="the title", json_schema_extra=COPIED)
        note: Optional[str] = Field(default=None, description="a note")

    recorder = _Recorder({"title": ("not_given", 0.99), "note": ("not_given", 0.99)})
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "on")
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
    monkeypatch.setattr(dspy, "Predict", recorder)
    assert fields_not_given("lend Dune", Input, {"title": "Dune", "note": "rare"}) == {"note": 0.99}
    assert recorder.calls == ["note"]


# ---------------------------------------------------------------- 3. one guard per marked field

class Marked(BaseModel):
    title: str = Field(description="the title", json_schema_extra=COPIED)
    note: Optional[str] = Field(default=None, description="a note", json_schema_extra=COPIED)
    copies: int = Field(default=1, description="how many", json_schema_extra=COPIED)


def test_the_execution_check_stage_leaves_a_marked_field_to_the_one_guard():
    checked = check.check_execution(Marked, {"title": "Ulysses", "note": "xyz", "copies": 7}, "find dune")
    assert checked.gaps == [] and checked.respelled == {} and checked.not_in_request == []
    (gap,) = check.check_execution(Marked, {"title": "<title>", "note": None, "copies": 1}, "find dune").gaps
    assert gap.code == "placeholder"   # what needs no knowledge of the request is screened on every str


@pytest.mark.parametrize("kind", ["execution", "planning"])
def test_a_marked_optional_str_given_and_not_written_is_refused_and_one_not_given_is_not_checked(tmp_path, kind):
    if kind == "execution":
        command = pc.option_command(Marked, "app/find")
    else:
        command = _plan(tmp_path, 'title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})\n'
                                  '        note: Optional[str] = Field(default=None, description="a note", json_schema_extra={"copied_from_command_text": True})')
    with pytest.raises(pc.PlanningCommandInputError, match="note='xyz' is not written in the request") as refused:
        pc.check_arguments(command, {"title": "dune", "note": "xyz"}, "find dune")
    assert (refused.value.gap.name, refused.value.gap.state) == ("note", check.INVALID)
    args, _ = pc.check_arguments(command, {"title": "dune", "note": None}, "find dune")
    assert args["title"] == "dune" and args["note"] in ("", None)


def test_the_slash_path_asks_for_a_marked_optional_str_the_request_does_not_write():
    from fastworkflow.parameters.signatures import InputForParamExtraction

    extractor = InputForParamExtraction(command="find dune", input_for_param_extraction_class=None)
    valid, message, _, fields = extractor.validate_parameters(
        None, "app/find", Marked.model_construct(title="dune", note="xyz", copies=1))
    assert not valid and fields == ["note"] and "the note 'xyz' is not in your request" in message


# ---------------------------------------------------------------- 4. a pattern is a full match, after the id is widened

def test_a_pattern_is_a_full_match_on_the_typed_planning_call():
    class Input(BaseModel):
        acct: str = Field(description="the account", pattern=r"acct_[0-9]+")

    command = pc.PlanningCommandCommand(name="show", description="", params=(), input_model=Input)
    with pytest.raises(pc.PlanningCommandInputError, match="does not match the pattern") as refused:
        pc.check_input(command, {"acct": "xx acct_1 yy"}, "show xx acct_1 yy")
    assert refused.value.fields == ("acct",)
    assert pc.check_input(command, {"acct": "acct_1"}, "show acct_1") == {}


def test_a_compiled_pattern_keeps_its_flags():
    class Input(BaseModel):
        code: str = Field(description="a code", pattern=re.compile(r"ab-\d+", re.I))

    command = pc.PlanningCommandCommand(name="show", description="", params=(), input_model=Input)
    assert pc.check_input(command, {"code": "AB-1"}, "show AB-1") == {}


def test_a_marked_id_with_a_pattern_is_widened_then_matched_on_the_typed_call():
    class Input(BaseModel):
        acct: str = Field(description="the account", pattern=r"^acct_\d+$", json_schema_extra=COPIED)

    command = pc.option_command(Input, "app/show", ACCOUNTS)
    args, _ = pc.checked_arguments(command, {"acct": "4421"}, "show acct_4421")
    assert args["acct"] == "acct_4421"


SHOW_ACCOUNT = '''
"""show_account: an execution command whose marked id field declares a pattern."""
import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Show one account."""

    class Input(BaseModel):
        acct: str = Field(description="the account", pattern=r"^acct_\\d+$",
                          json_schema_extra={"copied_from_command_text": True})

    class Output(BaseModel):
        done: bool = Field(description="whether it was done")

    plain_utterances = ["show an account"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(done=True)
'''


@pytest.fixture
def account_environment(tmp_path):
    from tests import parameter_pipeline_corpus as corpus

    env = corpus.Environment(tmp_path)
    commands = Path(env.folder) / "_commands"
    (commands / "show_account.py").write_text(SHOW_ACCOUNT, encoding="utf-8")
    (commands / "_workflow.py").write_text(
        "import re\n\n\nclass Context:\n    identifier_patterns = [re.compile(r'\\bacct_\\d+\\b')]\n", encoding="utf-8")
    with env:
        yield env


def test_the_slash_path_widens_a_cut_id_before_its_pattern_is_matched(account_environment):
    from tests import parameter_pipeline_corpus as corpus

    case = {"id": "x", "site": "impl", "command": "show_account", "text": "show acct_4421",
            "answers": [corpus.A(acct="4421")]}
    result = corpus.run_case(account_environment, case)
    assert result["output"]["valid"] is True, result["output"]["error_msg"]
    assert result["output"]["params"]["acct"] == "acct_4421"


def test_the_slash_path_refuses_a_value_its_pattern_does_not_match_in_full(account_environment):
    from tests import parameter_pipeline_corpus as corpus

    case = {"id": "x", "site": "impl", "command": "show_account", "text": "show acct_4421 now",
            "answers": [corpus.A(acct="acct_4421 now")] * 2}
    result = corpus.run_case(account_environment, case)
    assert result["output"]["valid"] is False
    assert "does not match the pattern" in result["output"]["error_msg"]


def test_the_policy_fallback_keeps_the_flags_of_the_workflows_id_patterns(tmp_path):
    from fastworkflow.policy.policy_decision import ContractFacts
    from fastworkflow.policy.policy_loader import with_workflow_identifiers

    (tmp_path / "_commands").mkdir()
    (tmp_path / "_commands" / "_workflow.py").write_text(
        "import re\n\n\nclass Context:\n    identifier_patterns = [re.compile(r'acct_\\d+', re.I)]\n", encoding="utf-8")
    facts = with_workflow_identifiers(ContractFacts(), str(tmp_path))
    assert facts.looks_like_id("ACCT_12") and not facts.looks_like_id("x acct_12")


# ---------------------------------------------------------------- 5. what the extraction model is told

def _descriptions(model: type) -> dict:
    from fastworkflow.parameters.signatures import InputForParamExtraction

    signature = InputForParamExtraction.create_signature_from_pydantic_model(model, pc.option_sentences(model))
    return {name: field.json_schema_extra["desc"] for name, field in signature.output_fields.items()}


FIELDS = ('title: str = Field(description="the title", json_schema_extra={"copied_from_command_text": True})\n'
          '        note: str = Field(default="", description="a note")\n'
          '        rare: bool = Field(default=False, description="only rare copies")\n'
          '        listed: bool = Field(default=True, description="list them")\n'
          '        code: Optional[str] = Field(default=None, description="a code", pattern=r"^C\\d+$")')


class Declared(BaseModel):
    title: str = Field(description="the title", json_schema_extra=COPIED)
    note: str = Field(default="", description="a note")
    rare: bool = Field(default=False, description="only rare copies")
    listed: bool = Field(default=True, description="list them")
    code: Optional[str] = Field(default=None, description="a code", pattern=r"^C\d+$")


def test_the_extraction_sentences_come_from_the_declarations_and_are_the_same_on_both_kinds(tmp_path):
    planning = _plan(tmp_path, FIELDS).input_model
    for model in (Declared, planning):
        desc = _descriptions(model)
        assert pc.IN_REQUEST_STR_SENTENCE.rstrip(".") in desc["title"]
        assert "Copy it exactly" not in desc["note"] and "verbatim" not in desc["note"].lower()
        assert pc.BOOL_SENTENCE.rstrip(".") in desc["rare"]
        assert pc.BOOL_TRUE_SENTENCE.rstrip(".") in desc["listed"]
        assert "Must match the pattern ^C\\d+$" in desc["code"]
    assert _descriptions(Declared) == _descriptions(planning)


def test_the_planners_typed_fill_reads_the_same_sentences(tmp_path):
    command = _plan(tmp_path, FIELDS)
    desc = {name: field.json_schema_extra["desc"] for name, field in pc.typed_signature(command).output_fields.items()}
    assert pc.IN_REQUEST_STR_SENTENCE.rstrip(".") in desc["title"]
    assert "verbatim" not in desc["note"].lower() and "Copy it exactly" not in desc["note"]
    assert pc.BOOL_SENTENCE.rstrip(".") in desc["rare"] and pc.BOOL_TRUE_SENTENCE.rstrip(".") in desc["listed"]
    assert "Must match the pattern ^C\\d+$" in desc["code"]


def test_no_sentence_depends_on_the_kind_of_command():
    from fastworkflow.parameters import signatures

    assert not hasattr(signatures, "planning_verbatim_sentences")


# ---------------------------------------------------------------- 6. the strict check reads whole words

@pytest.mark.parametrize("value, request_text", [("dun", "find dune messiah"), ("Ann", "find Joanna"),
                                                 ("4421", "show acct_44210")])
def test_a_marked_str_cut_from_a_written_word_is_not_written(value, request_text):
    command = pc.option_command(Marked, "app/find")
    with pytest.raises(pc.PlanningCommandInputError, match="is not written in the request"):
        pc.check_arguments(command, {"title": value}, request_text)


@pytest.mark.parametrize("value, request_text, kept", [
    ("dune", "find Dune messiah", "Dune"), ("Dune Messiah", "find dune messiah.", "dune messiah"),
    ("#W123", "cancel order #W123", "#W123"), ("Lee-Smith", "find Ann Lee–Smith", "Lee–Smith")])
def test_a_marked_str_written_as_whole_words_passes_in_the_requests_spelling(value, request_text, kept):
    command = pc.option_command(Marked, "app/find")
    assert pc.check_arguments(command, {"title": value}, request_text)[0]["title"] == kept


def test_an_id_cut_short_is_still_widened_to_the_whole_id():
    command = pc.option_command(Marked, "app/find", ACCOUNTS)
    assert pc.check_arguments(command, {"title": "4421"}, "show acct_4421")[0]["title"] == "acct_4421"
    assert rules.written("dun", "find dune", "find dune") is None


# ---------------------------------------------------------------- 7. one duplicate rule, part of the strict check

class Place(BaseModel):
    city: str = Field(description="the city", json_schema_extra=COPIED)
    state: Optional[str] = Field(default=None, description="the state", json_schema_extra=COPIED)


class PlainPlace(BaseModel):
    city: str = Field(description="the city")
    state: Optional[str] = Field(default=None, description="the state")


def test_two_marked_fields_may_hold_one_value_the_request_writes_twice():
    command = pc.option_command(Place, "app/stores")
    args, notes = pc.check_arguments(command, {"city": "New York", "state": "New York"}, "stores in New York, New York")
    assert (args["city"], args["state"]) == ("New York", "New York") and notes == []


def test_two_marked_fields_holding_one_value_the_request_writes_once_refuse_one():
    """The optional one is left not given with a note that says so (not a refusal, not silent); two required ones are
    still refused."""
    command = pc.option_command(Place, "app/stores")
    args, notes = pc.check_arguments(command, {"city": "Albany", "state": "Albany"}, "stores in Albany")
    assert args["city"] == "Albany" and not args["state"]
    assert notes == ["state left empty: 'Albany' is written once in your request and is the city."]
    both_required = pc.option_command(Place, "app/stores")
    both_required = dataclasses.replace(both_required, params=tuple(
        dataclasses.replace(p, required=True) for p in both_required.params))
    with pytest.raises(pc.PlanningCommandInputError, match="is also the value of") as refused:
        pc.check_arguments(both_required, {"city": "Albany", "state": "Albany"}, "stores in Albany")
    assert refused.value.gap.code == "role"


def test_unmarked_fields_that_hold_one_value_are_not_reset(tmp_path):
    command = _plan(tmp_path, 'city: str = Field(description="the city")\n'
                              '        state: Optional[str] = Field(default=None, description="the state")')
    args, notes = pc.check_arguments(command, {"city": "New York", "state": "New York"}, "stores in New York")
    assert (args["city"], args["state"]) == ("New York", "New York")
    assert not any("repeats another argument" in note for note in notes)
    assert not hasattr(pc, "_check_roles")


# ---------------------------------------------------------------- 8. one template-token and bracket verdict, no lists

def _slash_validate(model: type, values: dict, request: str):
    """``validate_parameters``, the stage the "/" reply merge, the agent's typed call and every code-read form of an
    execution command reach (and the LLM fill too, after its check stage)."""
    from fastworkflow.parameters.signatures import InputForParamExtraction

    extractor = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
    valid, message, _, fields = extractor.validate_parameters(None, "app/find", model.model_construct(**values))
    return valid, message, fields


@pytest.mark.parametrize("value", ["SKU_42", "ORDER_123", "NAME_1", "A_1"])
def test_an_upper_case_id_with_a_number_is_a_value_on_every_path(tmp_path, value):
    request = f"restock {value}"
    assert not rules.placeholder_like(value)
    assert check.check_execution(Plain, {"name": value}, request).gaps == []
    assert _slash_validate(Plain, {"name": value}, request)[0] is True
    command = _plan(tmp_path, 'name: str = Field(description="n")')
    assert pc.check_arguments(command, {"name": value}, request)[0]["name"] == value


@pytest.mark.parametrize("value, verdict", [
    ("<name>", "placeholder"), ("<code>", "placeholder"), ("<repo_uid>", "placeholder"), ("<ID>", "placeholder"),
    ("<dune>", "placeholder"), ("<first-order>", "placeholder"), ("<>", "placeholder"), ("<...>", "placeholder"),
    ("<UNKNOWN>", "placeholder"),
    ("<Dune>", "unwrapped"), ("<Ann Lee>", "unwrapped"), ("<9a1b>", "unwrapped"), ("<sku_42>", "unwrapped"),
])
def test_a_bracketed_value_is_judged_by_its_shape_alone(value, verdict):
    from fastworkflow.parameters import signatures

    assert signatures.classify_bracketed_value(value)[0] == verdict
    assert not hasattr(signatures, "_PLACEHOLDER_WORDS")


@pytest.mark.parametrize("value, field, refused", [
    ("uid", "item_uid", True), ("item_uid", "item_uid", True), ("your_item_uid", "item_uid", True),
    ("title", "title", True), ("desired-tag", "tag", True), ("the-tag", "tag", True),
    # a stand-in that does not echo the field's own name is the remote "was it given" check's business now
    ("value", "query", False), ("your_name", "title", False), ("some-value", "tag", False),
    ("your_uid", "item_uid", False), ("new-name", "group_name", False), ("placeholder", "query", False),
])
def test_an_ungrounded_placeholder_is_only_an_echo_of_the_fields_own_name(value, field, refused):
    from fastworkflow.parameters import signatures

    assert signatures.is_ungrounded_placeholder(value, field, "label the item 9a1b2c3d") is refused
    assert not hasattr(signatures, "_PLACEHOLDER_QUALIFIERS")


@pytest.mark.parametrize("value, kept", [
    ("{name}", None), ("{{name}}", None), ("<name>", None), ("<dune>", None), ("<Dune>", "Dune"),
    ("SKU_42", "SKU_42"), ("<SKU_42>", "SKU_42"), ("Ann Lee", "Ann Lee"),
])
def test_one_verdict_on_the_execution_fill_the_slash_stage_and_the_plan_call(tmp_path, value, kept):
    request = "restock SKU_42 for Ann Lee from Dune"
    gaps = check.check_execution(Plain, {"name": value}, request).gaps
    valid, _, fields = _slash_validate(Plain, {"name": value}, request)
    command = _plan(tmp_path, 'name: str = Field(description="n")')
    if kept is None:
        assert [g.code for g in gaps] == ["placeholder"]
        assert valid is False and fields == ["name"]
        with pytest.raises(pc.PlanningCommandInputError, match="is a placeholder, not a value"):
            pc.check_arguments(command, {"name": value}, request)
    else:
        assert gaps == [] and valid is True
        assert pc.check_arguments(command, {"name": value}, request)[0]["name"] == kept


@pytest.mark.parametrize("value, placeholder", [
    ("{name}", True), ("{{name}}", True), ("{repo_uid}", True), ("{ID}", True), ("{}", True),
    ('{"a": 1}', False), ("{Ann Lee}", False), ("{x1}", False), ('{"title": "Dune", "copies": 2}', False),
])
def test_braces_are_judged_by_the_same_inner_shape_as_brackets_and_never_unwrapped(value, placeholder):
    from fastworkflow.parameters import signatures

    assert rules.placeholder_like(value) is placeholder
    assert signatures.classify_bracketed_value(value) == (("placeholder" if placeholder else "unchanged"), value)


@pytest.mark.parametrize("value", ['{"a": 1}', '{"title": "Dune", "copies": 2}', "{Ann Lee}", "{x1}"])
def test_a_json_object_string_is_a_value_left_as_written_on_every_path(tmp_path, value):
    request = f"store {value}"
    assert check.check_execution(Plain, {"name": value}, request).gaps == []          # the execution fill
    params = Plain.model_construct(name=value)
    from fastworkflow.parameters.signatures import InputForParamExtraction

    extractor = InputForParamExtraction(command=request, input_for_param_extraction_class=None)
    valid, _, _, _ = extractor.validate_parameters(None, "app/find", params)          # the agent's typed call, "/"
    assert valid is True and params.name == value
    command = _plan(tmp_path, 'name: str = Field(description="n")')
    assert pc.check_arguments(command, {"name": value}, request)[0]["name"] == value  # the plan call
