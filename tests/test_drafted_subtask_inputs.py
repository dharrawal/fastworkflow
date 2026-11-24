"""A sub-task draft's inputs from the shared parameter extractor (fix-48np, stage 1).

``draft_round1_proposal`` fills each sub-task's inputs with the values the shared extractor gives on the
sub-task's text (``parameter_extraction.extract_arguments``: the execution commands' LLM extractor and
its checks, and for a planning command ``check_arguments``, the checks ``typed_planning_call`` runs). A
value that a declared producer (``from_``) an earlier sub-task runs is ``from sub-task N``, not extracted.
Stage 1 was display only; since fix-48np.1 the drafted inputs are run (``resolve_inputs``), and a planning
command whose required value the extractor leaves missing gets the typed fill fallback in the draft too.

Gate 1a of the issue: the shared extractor runs outside the CME workflow, with no CME workflow object at
all, and leaves no state behind on the app workflow.

Integration: real workflows (``tests/planning_tutorial_workflow``, a copy of the library example with its
trained heads), the real routing, the real DSPy extractor and the real checks. The model boundary is a
scripted ``DummyLM`` that records each request. No test calls a model.
"""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import review_proposals
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import extract_arguments
from fastworkflow.review_proposals import Means, SubTask, SubTaskInput, draft_inputs, resolve_inputs, run_text
from fastworkflow.run_listing import has_trained_heads
from fastworkflow.utils import signatures
from tests.test_approval_binding import annotated_library  # noqa: F401 - the library copy with its trained heads

TUTORIAL = str(Path(__file__).parent.joinpath("planning_tutorial_workflow").resolve())
NOTE_START = "(A first extraction from this command was refused:"
EMMA = "who wrote Emma and how many copies do we have"    # heads: book_by_title in the top-k
JOKE = "tell Ann a joke about cats"                        # heads: no command fits


class _ScriptedLM(DummyLM):
    """Answers the extractor's calls in order and records the text of each request."""

    def __init__(self, answers):
        super().__init__(answers, adapter=dspy.JSONAdapter())
        self.prompts: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


class _PlannerLM(DummyLM):
    """The planner's answers (the typed fill fallback), in the planner adapter's own format; records each request."""

    def __init__(self, answers):
        super().__init__(answers)
        self.prompts: list[str] = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.prompts.append("\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt))
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture(autouse=True)
def _framework(tmp_path, monkeypatch, no_planner_key):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    # Jev's "is this field given" check is a model call: off, so the scripted LM answers alone.
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "off")
    # Drafting is on by default; set explicitly so a developer's environment cannot turn it off here.
    monkeypatch.setitem(fastworkflow._env_vars, "FW_DRAFT_ARGUMENTS", "on")
    # One extractor LM is scripted in CALL order across the sub-tasks: drafts run one after another here (fix-awg8).
    monkeypatch.setenv("FW_SUBTASK_WORKERS", "1")
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture
def scripted(monkeypatch):
    """Make the extractor answer *answers* in order; returns the LM (its ``prompts`` are the requests)."""
    def script(*answers: dict) -> _ScriptedLM:
        lm = _ScriptedLM(list(answers))
        monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
        return lm
    return script


def _workflow(folder: str) -> fastworkflow.Workflow:
    fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    return fastworkflow.Workflow.create(folder, workflow_id_str=f"draft_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def library(annotated_library):  # noqa: F811
    from fastworkflow.examples.library.application.library import Library

    wf = _workflow(str(annotated_library))
    root = Library()
    wf.root = root
    wf.record = root
    return wf


@pytest.fixture
def library_heads(library, monkeypatch):
    """The library copy, for a test that needs its trained heads to pick each sub-task's means.

    A planning command the local model picks is checked by the remote decision model (fix-o1lg.2: with no remote
    answer the pick is only a candidate, and the part is asked about, not bound), so that model's answer is scripted
    here: a request about a book's title gets ``Library/book_by_title``, any other request gets "none"."""
    if not has_trained_heads(library):
        pytest.skip("the library example's trained heads are not on this machine (fastworkflow train it first)")
    from tests.conftest import script_matcher

    script_matcher(monkeypatch, lambda text: ({"Library/book_by_title": 0.9, "none": 0.05}
                                              if ("Emma" in text or "Dune" in text) else {"none": 0.95}))
    return library


# ---------------------------------------------------------------- gate 1a: outside the CME workflow


def test_the_shared_extractor_fills_a_planning_command_with_no_cme_workflow(scripted):
    wf = _workflow(TUTORIAL)
    before = copy.deepcopy(dict(wf.context))
    lm = scripted({"reasoning": "the title", "title": "Dune"})

    result = extract_arguments(wf, "book_by_title", "who wrote Dune")

    assert result.valid and result.refusal is None and not result.retried
    assert result.params.title == "Dune"
    assert len(lm.prompts) == 1 and "who wrote Dune" in lm.prompts[0]
    assert dict(wf.context) == before          # nothing stored on the app workflow (no parameter loop, no stage)


def test_a_planning_value_not_in_the_request_is_refused_by_check_arguments_retried_once_and_refused(scripted):
    wf = _workflow(TUTORIAL)
    lm = scripted({"reasoning": "a guess", "title": "Dune Messiah"},
                  {"reasoning": "a guess again", "title": "Dune Messiah"})

    result = extract_arguments(wf, "book_loans", "who has Foundation on loan")

    assert result.retried and not result.valid
    assert result.refusal == "title='Dune Messiah' is not written in the request"
    assert NOTE_START not in lm.prompts[0]
    # the retry reads the refusal after the text; the checks read the text alone (so the note's own words never pass)
    assert NOTE_START in lm.prompts[1] and "title='Dune Messiah' is not written in the request" in lm.prompts[1]


def test_the_retry_can_fix_a_refused_planning_value(scripted):
    wf = _workflow(TUTORIAL)
    scripted({"reasoning": "a guess", "title": "Dune Messiah"}, {"reasoning": "copied", "title": "Foundation"})

    result = extract_arguments(wf, "book_loans", "who has Foundation on loan")

    assert result.retried and result.valid and result.params.title == "Foundation"


def test_a_missing_required_planning_value_is_named(scripted):
    wf = _workflow(TUTORIAL)
    scripted({"reasoning": "no author", "author": "NOT_FOUND"}, {"reasoning": "still none", "author": "NOT_FOUND"})

    result = extract_arguments(wf, "author_books", "show every book by this author in the library")

    assert not result.valid and result.missing == ["author"]
    assert result.refusal == "missing required argument author"


def test_an_execution_command_gets_the_execution_checks(scripted):
    """The same entry point on an execution command: argument_checks refuses an invented value and retries."""
    wf = _workflow(TUTORIAL)
    lm = scripted({"reasoning": "name", "query": "Bob Jones"}, {"reasoning": "copied", "query": "Ann Lee"})

    result = extract_arguments(wf, "find_member", "find the member Ann Lee")

    assert result.valid and result.params.query == "Ann Lee"
    assert len(lm.prompts) == 2 and NOTE_START in lm.prompts[1]


def test_two_extractions_in_a_row_share_no_state(scripted):
    """No hidden state between calls: the second command's extraction sees only its own text and answer."""
    wf = _workflow(TUTORIAL)
    scripted({"reasoning": "t", "title": "Dune"}, {"reasoning": "a", "author": "Octavia Butler"})

    first = extract_arguments(wf, "book_by_title", "who wrote Dune")
    second = extract_arguments(wf, "author_books", "what titles did Octavia Butler write")

    assert first.params.title == "Dune" and second.params.author == "Octavia Butler"
    assert not hasattr(second.params, "title")


# ---------------------------------------------------------------- the draft (stage 1)


def test_the_draft_shows_the_extracted_planning_arguments(library_heads, scripted):
    lm = scripted({"reasoning": "the title", "title": "Emma"})

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, JOKE])

    first = draft.items[0]
    assert first.means == Means(kind="planning", name="Library/book_by_title")
    assert first.inputs == [SubTaskInput(name="title", value="Emma", drafted=True)]
    assert first.check == ""
    assert "(title: Emma)" in draft.rendering()
    assert "inputs: title=Emma" in draft.model_rendering()
    assert EMMA in lm.prompts[0]                  # each sub-task's means is extracted from its own text


def test_the_draft_shows_a_refusal_and_no_values(library_heads, scripted):
    scripted({"reasoning": "a guess", "title": "Persuasion"}, {"reasoning": "again", "title": "Persuasion"})
    # then the typed fill fallback on the planner (fix-48np.1) gives a title the request does not write either
    planner = _PlannerLM([{"reasoning": "typed fill", "title": "Persuasion", "steps_before": [], "steps_after": []}])

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, JOKE], planner_lm=planner)

    assert draft.items[0].inputs == []
    # the refused title is left out of the inputs, so the item as shown lacks it: its line says the title is missing,
    # not a refusal of a value the item does not hold (the draft is worked out again from its inputs, ``rechecked``)
    assert draft.items[0].check == "missing required argument title"
    assert draft.items[0].check_missing == ["title"] and draft.items[0].check_refused == []
    assert EMMA in planner.prompts[0]                    # the typed fill read the sub-task's text


def test_with_the_switch_off_the_draft_makes_no_extraction(library_heads, scripted, monkeypatch):
    lm = scripted()
    monkeypatch.setitem(fastworkflow._env_vars, review_proposals.DRAFT_ARGUMENTS_ENV_VAR, "off")

    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, JOKE])

    assert lm.prompts == [] and draft.items[0].inputs == []
    assert draft.items[0].check                    # the empty-argument check, as before


def test_drafting_is_on_by_default(library_heads, scripted, monkeypatch):
    """On unless switched off (fix-48np.1, after the round-2 measurement): with the variable unset, the draft extracts."""
    lm = scripted({"reasoning": "the title", "title": "Emma"}, *[{"reasoning": "the member", "member": "Ann"}] * 3)
    monkeypatch.setattr(fastworkflow, "_env_vars",
                        {k: v for k, v in fastworkflow._env_vars.items()
                         if k != review_proposals.DRAFT_ARGUMENTS_ENV_VAR})
    monkeypatch.delenv(review_proposals.DRAFT_ARGUMENTS_ENV_VAR, raising=False)

    assert review_proposals.drafting_on() is True
    draft = review_proposals.draft_round1_proposal(library_heads, [EMMA, JOKE])

    assert lm.prompts and draft.items[0].inputs == [SubTaskInput(name="title", value="Emma", drafted=True)]


@pytest.mark.parametrize("value", ["off", "0", "false", "no", "OFF"])
def test_drafting_is_off_only_when_switched_off(monkeypatch, value):
    monkeypatch.setitem(fastworkflow._env_vars, review_proposals.DRAFT_ARGUMENTS_ENV_VAR, value)
    assert review_proposals.drafting_on() is False


def test_a_value_an_earlier_sub_task_produces_is_marked_from_that_sub_task(library, scripted):
    """open_book's book_uid is declared from find_book: after a find_book sub-task it is 'from sub-task 1', not guessed."""
    scripted({"reasoning": "no uid in the text", "book_uid": "NOT_FOUND"})
    earlier = [SubTask(goal="find the book Dune", means=Means(kind="plain", name="Library/find_book"))]
    item = SubTask(goal="open it", means=Means(kind="plain", name="Library/open_book"))

    drafted = draft_inputs(library, item, "open it", earlier)

    assert drafted.refusal is None
    assert list(drafted.inputs) == [SubTaskInput(name="book_uid", from_subtask=1, drafted=True)]


def test_with_no_producer_before_it_the_value_is_not_marked(library, scripted):
    scripted({"reasoning": "no uid in the text", "book_uid": "NOT_FOUND"})
    item = SubTask(goal="open it", means=Means(kind="plain", name="Library/open_book"))

    drafted = draft_inputs(library, item, "open it", [])

    assert drafted.inputs == () and drafted.refusal == "missing required argument book_uid"


def test_a_missing_value_is_kept_by_name_on_the_item_and_the_review_line_reads_the_names(library, scripted):
    scripted({"reasoning": "no title", "title": "NOT_FOUND"}, {"reasoning": "still none", "title": "NOT_FOUND"})
    item = SubTask(goal="find the book", means=Means(kind="planning", name="Library/book_by_title"))

    drafted = review_proposals.drafted_item(library, item, "find the book", [])

    assert drafted.check_missing == ["title"] and drafted.check == "missing required argument title"
    # the line reads the names, not the text: change the text and the line stays; drop the names and it is the refusal
    reworded = drafted.model_copy(update={"check": "the title is needed"})
    assert "the required value 'title' is missing" in review_proposals._check_line(1, reworded)
    assert 'change 1: title=...' in review_proposals._check_line(1, reworded)
    assert review_proposals._check_line(1, reworded.model_copy(update={"check_missing": []})) == \
        "Cannot run as shown: the title is needed."
    given = reworded.model_copy(update={"inputs": [SubTaskInput(name="title", value="Emma")]})
    assert review_proposals._check_line(1, given) == ""          # given since the draft: the note is stale
    assert "check_missing" not in SubTask(goal="g").model_dump()  # empty: absent from the serialized item


def test_drafted_inputs_are_run_as_the_review_shows_them():
    """Stage 1 showed drafted inputs and did not run them; fix-48np.1 runs them: a drafted value is resolved as a given
    one, and a drafted 'from sub-task N' takes N's output once N has run."""
    item = SubTask(goal="who wrote Emma", means=Means(kind="planning", name="Library/book_by_title"),
                   inputs=[SubTaskInput(name="title", value="Emma", drafted=True),
                           SubTaskInput(name="n", from_subtask=1, drafted=True)])

    before = resolve_inputs(item, {})
    assert before.named == {"title": "Emma"} and before.missing == (1,)

    after = resolve_inputs(item, {1: "3"})
    assert after.named == {"title": "Emma", "n": "3"} and after.missing == ()
    assert run_text(item, after) == "who wrote Emma (inputs: title=Emma, n=3)"


def test_a_drafted_input_round_trips_through_the_review_json():
    item = SubTask(goal="g", inputs=[SubTaskInput(name="title", value="Emma", drafted=True)])
    proposal = review_proposals.SubTaskProposal(items=[item])
    assert review_proposals.SubTaskProposal.from_text(proposal.model_dump_json()) == proposal


# ---------------------------------------------------------------- the copy-it-exactly sentence (fix-48np)


def _signature_text(folder: str, name: str) -> str:
    from fastworkflow import ModuleType, planning_commands
    from fastworkflow.utils.signatures import InputForParamExtraction

    fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    model = fastworkflow.RoutingRegistry.get_definition(folder).get_command_class(
        name, ModuleType.COMMAND_PARAMETERS_CLASS)
    signature = InputForParamExtraction.create_signature_from_pydantic_model(model, planning_commands.option_sentences(model))
    return "\n".join(f"{n}: {f.json_schema_extra.get('desc', '')}" for n, f in signature.output_fields.items())


def test_a_planning_commands_str_field_is_told_to_copy_only_when_it_is_marked(annotated_library):  # noqa: F811
    from fastworkflow.planning_commands import IN_REQUEST_STR_SENTENCE

    text = _signature_text(str(annotated_library), "Library/note_member")
    # member and text both carry copied_from_command_text; a field without it is told nothing about copying, as on
    # an execution command (bd fix-ftdl.6.5: the sentences come from the field's own declarations, on both kinds)
    assert text.count(IN_REQUEST_STR_SENTENCE.rstrip(".")) == 2           # member and text, once each


def test_an_execution_commands_fields_are_told_nothing_new():
    from fastworkflow.planning_commands import IN_REQUEST_STR_SENTENCE

    assert IN_REQUEST_STR_SENTENCE.rstrip(".") not in _signature_text(TUTORIAL, "count_overdue")


def test_a_failed_draft_extraction_is_logged_and_shows_no_inputs(library, monkeypatch, caplog):
    def refuse(self, *args, **kwargs):
        raise AssertionError("the parameter-extraction model was called")

    monkeypatch.setattr(signatures.InputForParamExtraction, "extract_parameters", refuse)
    item = SubTask(goal="who wrote Emma", means=Means(kind="planning", name="Library/book_by_title"))

    with caplog.at_level("WARNING"):
        assert draft_inputs(library, item, "who wrote Emma", []) is None
    assert any("could not draft the inputs of Library/book_by_title (AssertionError" in r.getMessage()
               for r in caplog.records)


# ---------------------------------------------------------------- the check is worked out again when the inputs change

def _refused_name_plan(**check) -> "review_proposals.SubTaskProposal":
    item = SubTask(goal="do you have a book called Emma", means=Means(kind="planning", name="find_record"),
                   inputs=[SubTaskInput(name="kind", value="book", drafted=True)], drafted_means="find_record", **check)
    return review_proposals.SubTaskProposal(items=[item])


def _revised(plan, *changes):
    import types

    from fastworkflow.review_engine import Change, apply_edits

    host = types.SimpleNamespace(_app_workflow=_workflow(TUTORIAL), _current_planner_lm=None)
    return apply_edits(plan, [Change(index=1, field="inputs", value=value) for value in changes]).revised(host, plan)


def test_a_review_edit_that_fixes_a_refused_value_clears_the_line_and_one_that_does_not_keeps_it():
    plan = _refused_name_plan(check="name='Emma_X' is not written in the request", check_refused=["name"])
    # the draft's refusal of a value it left out names the field and how to change it
    assert 'Reply "change 1: name=..." to change it.' in plan.user_rendering()

    fixed = _revised(plan, "name=Emma")
    assert fixed.items[0].check == "" and "Cannot run" not in fixed.user_rendering()

    wrong = _revised(plan, "name=Zed")
    assert wrong.items[0].check == "name='Zed' is not written in the request"
    assert wrong.items[0].check_refused == ["name"]
    assert "Cannot run as shown: name='Zed' is not written in the request." in wrong.user_rendering()


def test_a_refusal_of_a_value_the_item_does_not_show_is_cleared_by_an_edit_that_names_it():
    plan = review_proposals.SubTaskProposal(items=[SubTask(
        goal="do you have a book called Emma", means=Means(kind="planning", name="find_record"),
        inputs=[SubTaskInput(name="name", value="Emma", drafted=True)], drafted_means="find_record",
        check="kind='author' is not one of book, member", check_refused=["kind"])])
    assert 'change 1: kind=...' in plan.user_rendering()
    cleared = _revised(plan, "kind=none")
    assert cleared.items[0].check == "" and "Cannot run" not in cleared.user_rendering()


def test_a_reply_to_the_values_question_rechecks_the_item():
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    plan = _refused_name_plan(check="name='Emma_X' is not written in the request", check_refused=["name"])
    given = resolution._overlay(plan, 0, {"name": "Emma"}, _workflow(TUTORIAL))
    assert given.items[0].check == "" and "Cannot run" not in given.user_rendering()
    still = resolution._overlay(plan, 0, {"name": "Zed"}, _workflow(TUTORIAL))
    assert "name='Zed' is not written in the request" in still.user_rendering()


# ---------------------------------------------------------------- a validator's rewrite is not guarded again

_SINGULAR = '''
"""tag_records: a planning command whose validator rewrites a marked value."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Signature:
    """Tag the records of one group."""

    plain_utterances = ["tag the contractors group"]

    class Input(BaseModel):
        name: str = Field(description="the group", json_schema_extra={"copied_from_command_text": True})
        kind: Literal["group", "person"] = Field(default="group", description="the kind of record")

        @model_validator(mode="after")
        def singular(self, info):
            if self.name.endswith("s"):
                self.name = self.name[:-1]
            return self


class PlanGenerator:
    def steps(self, input):
        return [f"tag the {input.kind} {input.name}"]
'''


@pytest.fixture
def singular_workflow(tmp_path):
    import shutil

    folder = tmp_path / "singular_tutorial"
    shutil.copytree(TUTORIAL, folder, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (folder / "_commands" / "tag_records.py").write_text(_SINGULAR, encoding="utf-8")
    return _workflow(str(folder))


def test_a_validators_rewrite_of_a_marked_value_is_not_refused_by_a_later_check(singular_workflow, scripted):
    """The validator writes the singular, which the request does not write: the draft keeps the value the checks
    stand behind (the one the request writes), so the later checks (the values question's, the run's) pass, and the run's
    own validators write the singular again."""
    from fastworkflow import planning_commands

    scripted({"reasoning": "the group", "name": "contractors", "kind": "group"})
    goal = "tag the contractors group"
    item = SubTask(goal=goal, means=Means(kind="planning", name="tag_records"))
    drafted = review_proposals.drafted_item(singular_workflow, item, goal, [])

    assert drafted.check == "" and {each.name: each.value for each in drafted.inputs}.get("name") == "contractors"
    command = planning_commands.planning_command(singular_workflow.folderpath, "tag_records")
    raw = {each.name: each.value for each in drafted.inputs}
    assert not planning_commands.argument_gaps(command, raw, goal).names
    args, _ = planning_commands.checked_arguments(command, raw, goal, singular_workflow)
    assert args["name"] == "contractor"


_COUNT_LINE = '''
"""count_line: a planning command with a marked int that defaults to 0."""
from pydantic import BaseModel, Field


class Signature:
    """Write one line for an owner."""

    plain_utterances = ["write a line for Ann Lee"]

    class Input(BaseModel):
        owner: str = Field(description="the owner", json_schema_extra={"copied_from_command_text": True})
        count: int = Field(default=0, description="how many, when the request gives a number",
                           json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def steps(self, input):
        return [f"write the line of {input.owner}"]
'''


def test_a_refused_draft_does_not_keep_an_int_at_its_default_for_a_later_check_to_refuse(singular_workflow, scripted):
    """The request gives no number: the draft's refusal of the owner keeps the values the checks stand behind, and the
    count's default is not one of them, so the values question asks for the owner alone (not a whole number)."""
    from fastworkflow import planning_commands

    (Path(singular_workflow.folderpath) / "_commands" / "count_line.py").write_text(_COUNT_LINE, encoding="utf-8")
    workflow = _workflow(singular_workflow.folderpath)
    scripted({"reasoning": "r", "owner": "Bob", "count": 0}, {"reasoning": "r", "owner": "Bob", "count": 0})
    goal = "write a line for Ann Lee"
    item = SubTask(goal=goal, means=Means(kind="planning", name="count_line"))
    planner = _PlannerLM([{"reasoning": "r", "owner": "Bob", "count": 0}])
    drafted = review_proposals.drafted_item(workflow, item, goal, [], planner_lm=planner)

    assert "count" not in {each.name for each in drafted.inputs}
    command = planning_commands.planning_command(workflow.folderpath, "count_line")
    raw = {p.name: next((each.value for each in drafted.inputs if each.name == p.name), None) for p in command.params}
    assert planning_commands.argument_gaps(command, raw, goal).names == ("owner",)


# ---------------------------------------------------------------- the duplicate rule's note, and no stale refusal line

_CONNECTED = '''
"""connected: a planning command with a required and an optional marked name."""
from pydantic import BaseModel, Field


class Signature:
    """Show the records connected to one record."""

    plain_utterances = ["show the records connected to Payroll"]

    class Input(BaseModel):
        name: str = Field(description="the record", json_schema_extra={"copied_from_command_text": True})
        connected_name: str = Field(default="", description="one connected record to narrow to",
                                    json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def steps(self, input):
        return [f"show the records connected to {input.name}"]
'''


@pytest.fixture
def connected_workflow(singular_workflow):
    (Path(singular_workflow.folderpath) / "_commands" / "connected.py").write_text(_CONNECTED, encoding="utf-8")
    fastworkflow.RoutingRegistry.clear_registry()   # the definition loaded before the command was written
    return _workflow(singular_workflow.folderpath)


def test_a_draft_that_puts_one_written_name_in_two_fields_shows_the_note_and_no_refusal(connected_workflow, scripted):
    """The optional field is left empty with a note under the item (not "Cannot run as shown"), and the extractor is
    not asked a second time (nothing was refused)."""
    lm = scripted({"reasoning": "r", "name": "Payroll", "connected_name": "Payroll"})
    goal = "show the records connected to Payroll"
    item = SubTask(goal=goal, means=Means(kind="planning", name="connected"))

    drafted = review_proposals.drafted_item(connected_workflow, item, goal, [])

    line = "connected_name left empty: 'Payroll' is written once in your request and is the name."
    assert drafted.check == "" and {each.name: each.value for each in drafted.inputs} == {"name": "Payroll"}
    assert drafted.check_notes == {"connected_name": line} and len(lm.prompts) == 1
    shown = review_proposals.SubTaskProposal(items=[drafted]).user_rendering()
    assert f"   {line}" in shown and "Cannot run" not in shown
    # the note goes with a reply that leaves the field out, and is worked out again when an edit puts it back
    rechecked = review_proposals.rechecked(connected_workflow, drafted)
    assert rechecked.check_notes == {"connected_name": line}
    edited = review_proposals.rechecked(connected_workflow, drafted.model_copy(update={"inputs": [
        *drafted.inputs, SubTaskInput(name="connected_name", value="Payroll")]}))
    assert edited.check_notes == {"connected_name": line} and edited.check == ""


def test_a_draft_that_leaves_a_refused_value_out_prints_no_refusal_about_it(connected_workflow, scripted):
    """The draft refuses an optional value the request does not write and leaves it out of the inputs: the item as
    shown runs without it, so its review line says nothing about it."""
    scripted({"reasoning": "r", "name": "Payroll", "connected_name": "Billing"},
             {"reasoning": "r", "name": "Payroll", "connected_name": "Billing"})
    goal = "show the records connected to Payroll"
    item = SubTask(goal=goal, means=Means(kind="planning", name="connected"))
    planner = _PlannerLM([{"reasoning": "r", "name": "Payroll", "connected_name": "Billing"}])

    drafted = review_proposals.drafted_item(connected_workflow, item, goal, [], planner_lm=planner)

    assert {each.name: each.value for each in drafted.inputs} == {"name": "Payroll"}
    assert drafted.check == "" and drafted.check_refused == []
    assert "Cannot run" not in review_proposals.SubTaskProposal(items=[drafted]).user_rendering()


def test_the_error_reshow_rechecks_a_refusal_of_a_value_the_item_does_not_hold():
    """An unreadable review reply shows the steps again: a refusal of a value the item does not hold is worked out
    again first (``review_engine._structured_error_prompt``), as a revised review's is."""
    from fastworkflow.review_engine import _structured_error_prompt

    plan = _refused_name_plan(check="kind='author' is not one of book, member", check_refused=["kind"])
    plan = plan.model_copy(update={"items": [plan.items[0].model_copy(update={
        "inputs": [SubTaskInput(name="name", value="Emma", drafted=True)]})]})
    assert "Cannot run as shown" in plan.user_rendering()

    shown = _structured_error_prompt(plan, _workflow(TUTORIAL), "")

    assert "Cannot run as shown" not in shown and shown.startswith("I couldn't tell what to change.")
