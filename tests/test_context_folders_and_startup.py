"""A context folder, the startup command and a command that does not load (bd fix-ftdl.7.3, fix-ftdl.7.5, B6 items 4-6).

* A folder of ``_commands/`` named after a class is a context whatever kind of command it holds: a folder of planning
  commands only is a routing context.
* A command with no ``Signature`` runs only as an action (the startup action, a direct action): no typed name, no
  fuzzy name and no remote candidate reaches it, and no listing names it.
* A workflow-wide load problem says so (``LoadProblem.workflow_wide``); a command's own problem does not.
* An execution command that does not load is not in any scope's command set, and a trained head's answer naming it
  is "not here".

Each case is built on a TEMP COPY of the library example's commands (``tests.test_field_sources._build_workflow``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.planning import command_preview
from fastworkflow import context_navigation, intent_detector, workflow_agent
from fastworkflow.planning import planning_commands as pc
from fastworkflow.command_routing import RoutingRegistry
from tests.test_field_sources import ROOT, _build_workflow
from tests.test_one_rule_for_author_mistakes import _exec


@pytest.fixture(autouse=True)
def _init():
    fastworkflow.init(env_vars=dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env"))))
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()
    workflow_agent.split_settings.cache_clear()
    yield
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()


_SHELF_PLAN = '''from pydantic import BaseModel, Field


class Signature:
    """Report what a shelf holds."""

    class Input(BaseModel):
        topic: str = Field(description="what to report on")

    plain_utterances = ["report on this shelf"]

    @staticmethod
    def generate_utterances(workflow, command_name):
        # written out, so training a copy calls no model
        return ["report on this shelf", "what is on this shelf", "summarise the shelf", "shelf report please"]


class PlanGenerator:
    def steps(self, input):
        return ["say what the shelf holds"]
'''


class Shelf:
    """A record of the context ``Shelf`` (the context is the record's class name)."""


# ------------------------------------------------------------------ item 1: a folder of planning commands only


def test_a_folder_of_planning_commands_only_is_a_context(tmp_path):
    folder = _build_workflow(tmp_path, "wf_shelf", extra_planning_commands={"Shelf/shelf_report": _SHELF_PLAN})
    assert command_preview.load_problems(folder) == []
    definition = RoutingRegistry.get_definition(folder)
    assert "Shelf" in definition.contexts
    core = set(definition.command_directory.core_command_names) - {"wildcard"}
    assert set(definition.get_command_names("Shelf")) == core          # no execution command of its own
    assert definition.get_planning_command_names("Shelf") == ["Shelf/shelf_report"]

    workflow = fastworkflow.Workflow.create(workflow_folderpath=folder, workflow_id_str="b6-shelf")
    workflow.root = Shelf()
    scope, names = context_navigation.command_on_path(workflow, "shelf_report")
    assert (scope.context_name, names) == ("Shelf", ("Shelf/shelf_report",))


@pytest.mark.slow
@pytest.mark.requires_llm_key
def test_training_takes_a_context_with_no_execution_command(tmp_path, no_planner_key):
    """A temp copy of the library example with a planning-only folder trains: the folder gets its trained files, and
    every "is it trained" gate says yes. No model is called (the copy's own caches, ``no_model_calls``)."""
    import importlib.util

    from fastworkflow.model_pipeline_training import is_workflow_trained
    from fastworkflow.train import artifact_versioning, embedding_head, utterance_cache
    from fastworkflow.train import intent_model_layout as layout
    from tests.test_intent_model_layout_training import (
        LIBRARY, _copy_library, _rekey_parameter_examples, _train, _training_env)

    if importlib.util.find_spec("datasets") is None:
        pytest.skip("datasets package not installed; intent-detection training is skipped.")
    cache_root = LIBRARY / "___command_info" / utterance_cache.CACHE_DIRNAME
    if not cache_root.is_dir() or not any(cache_root.glob("*.json")):
        pytest.skip(f"the library example has no generated-utterance cache at {cache_root}")
    try:
        embedding_head.SentenceEncoder.get(layout.DEFAULT_ENCODER)
    except Exception as error:  # noqa: BLE001 - no network and no cached copy of the encoder
        pytest.skip(f"the default sentence encoder cannot be loaded here ({type(error).__name__}: {error})")

    workflow_path = _copy_library(tmp_path / "library")
    shelf = Path(workflow_path) / "_commands" / "Shelf"
    shelf.mkdir()
    (shelf / "shelf_report.py").write_text(_SHELF_PLAN, encoding="utf-8")
    fastworkflow.init(env_vars=_training_env(workflow_path, None))
    _rekey_parameter_examples(workflow_path)
    version = _train(workflow_path, None)

    vdir = artifact_versioning.version_dir(workflow_path, version)
    assert sorted(os.listdir(vdir / "Shelf")) == ["threshold.json", "workflow_scope.json"]
    assert is_workflow_trained(workflow_path) == (True, [])


# ------------------------------------------------------------------ item 5: a command that does not load is not routable

_BROKEN = {"Library/greet": _exec(attributes="    writes = True")}     # a retired attribute: greet does not load


class _Router:
    """A trained head's answer, scripted (the shape ``CommandRouter.predict_with_details`` returns)."""

    def __init__(self, labels):
        self.labels = list(labels)

    def predict_with_details(self, utterance):
        return list(self.labels), {"confident": True, "topk_labels": list(self.labels),
                                   "topk_scores": [0.9] * len(self.labels)}


def test_an_execution_command_that_does_not_load_is_in_no_scopes_command_set(tmp_path):
    folder = _build_workflow(tmp_path, "wf_broken", extra_commands=_BROKEN)
    problems = command_preview.load_problems(folder)
    assert [p.split(": ", 1)[0] for p in problems] == ["Library/greet"], problems
    definition = RoutingRegistry.get_definition(folder)
    assert "Library/greet" in definition.command_directory.map_command_2_metadata    # the file was read
    for context in definition.contexts:
        assert "Library/greet" not in definition.get_command_names(context)
    assert "Library/greet" not in context_navigation.scope_command_names(folder, "Library")
    assert "Library/greet" not in {c.name for c in intent_detector.candidates(folder, "Library")}
    assert "Library/find_book" in definition.get_command_names("Library")           # the rest still routes


def test_a_heads_answer_naming_a_command_that_does_not_load_is_not_here(tmp_path):
    folder = _build_workflow(tmp_path, "wf_broken_head", extra_commands=_BROKEN)
    answer = intent_detector.ClassifierEngine(_Router(["Library/greet"]), workflow_folderpath=folder).decide("hi")
    assert answer.kind == intent_detector.NOT_MINE and answer.confident is False
    answer = intent_detector.ClassifierEngine(
        _Router(["Library/greet", "Library/find_book"]), workflow_folderpath=folder).decide("hi")
    # still not one sure answer: the other candidate stays a candidate, never a match
    assert answer.labels == ["Library/find_book"] and answer.kind == intent_detector.AMBIGUOUS


# ------------------------------------------------------------------ item 4: a workflow-wide problem stops every entry


def _bad_startup_action(tmp_path, name: str) -> str:
    folder = _build_workflow(tmp_path, name, extra_commands=_BROKEN)
    (Path(folder) / "startup_action.json").write_text('{"command_name": "strtup"}', encoding="utf-8")
    return folder


def test_each_load_problem_says_whether_it_stops_the_workflow(tmp_path):
    problems = command_preview.load_problems(_bad_startup_action(tmp_path, "wf_wide"))
    assert {p.split(": ", 1)[0]: p.workflow_wide for p in problems} == {
        "Library/greet": False, "startup_action.json": True}, problems


def _env_files() -> list[str]:
    examples = os.path.join(ROOT, "fastworkflow", "examples")
    return [os.path.join(examples, "fastworkflow.env"), os.path.join(examples, "fastworkflow.passwords.env")]


def test_fastworkflow_run_exits_on_a_workflow_wide_problem_naming_the_file(tmp_path):
    from tests.test_run_code import _cli, _no_llm_env

    folder = _bad_startup_action(tmp_path, "wf_run")
    cli = _cli("run_cli", folder, *_env_files(), env=_no_llm_env(), timeout=300)
    assert cli.returncode == 1, cli.stdout + cli.stderr
    out = " ".join((cli.stdout + cli.stderr).split())
    assert "startup_action.json: command_name 'strtup' is no command of this workflow" in out
    assert "the workflow does not load" in out


def test_the_server_refuses_to_start_on_a_workflow_wide_problem_naming_the_file(tmp_path):
    import subprocess
    import sys

    from tests.test_run_code import ROOT as REPO, _free_port, _no_llm_env

    folder = _bad_startup_action(tmp_path, "wf_server")
    result = subprocess.run(
        [sys.executable, "-m", "fastworkflow.cli", "run_fastapi_mcp", folder, *_env_files(),
         "--port", str(_free_port())],
        cwd=REPO, env={**_no_llm_env(), "PYTHONUNBUFFERED": "1"}, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=300)
    text = result.stdout + result.stderr
    assert "Application startup failed" in text, text      # uvicorn's own line
    assert "startup problem: startup_action.json: command_name 'strtup' is no command of this workflow" in text
    assert "Application ready to accept traffic" not in text
    assert result.returncode != 0, text                    # was 0: a supervisor read the failed start as a clean exit


def test_the_server_process_exits_non_zero_on_a_runtime_startup_problem(tmp_path):
    """The runtime problems (``load_problems(runtime=True)``: here no agent model and no Jev key) stop the server the
    same way, and its process exits non-zero, as ``fastworkflow run_cli`` exits 1."""
    import subprocess
    import sys

    from tests.test_run_code import _free_port, _no_llm_env

    folder = _build_workflow(tmp_path, "wf_runtime")
    env_file, passwords = tmp_path / "bare.env", tmp_path / "bare.passwords.env"
    env_file.write_text(f"FASTWORKFLOW_STATE_ROOT={tmp_path / 'state'}\n", encoding="utf-8")
    passwords.write_text("", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "fastworkflow.cli", "run_fastapi_mcp", folder, str(env_file), str(passwords),
         "--port", str(_free_port())],
        # Both set empty (which means unset): importing litellm loads a .env it finds above its own install folder into
        # the process environment, never over a variable that is already there, and a developer's .env sets them.
        cwd=str(tmp_path), env={**_no_llm_env(), "LLM_AGENT": "", "JEV_API_KEY": "", "PYTHONUNBUFFERED": "1"},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    try:
        out, _ = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, 9)
        out, _ = process.communicate()
        pytest.fail(f"the server did not exit on its startup problem:\n{out[-4000:]}")
    result = subprocess.CompletedProcess(process.args, process.returncode, out, "")
    text = result.stdout + result.stderr
    assert "Application startup failed" in text, text
    assert "LLM_AGENT" in text and "JEV_API_KEY" in text, text
    assert "Application ready to accept traffic" not in text
    assert result.returncode != 0, text


# ------------------------------------------------------------------ item 6: the test example workflow loads


def test_the_example_test_workflow_has_no_load_problem():
    # Its _commands/ is what the build test generates from application/ (no startup command); the tests that use it
    # set its root themselves, so it names no startup action.
    assert command_preview.load_problems(os.path.join(ROOT, "tests", "example_workflow")) == []


# ------------------------------------------------------------------ item 2b/c: a command with no Signature


def _on_the_root(folder: str, name: str):
    from fastworkflow.examples.library.application.library import Library

    workflow = fastworkflow.Workflow.create(workflow_folderpath=folder, workflow_id_str=name)
    workflow.root = Library()
    return workflow


@pytest.mark.usefixtures("no_planner_key")
def test_a_command_with_no_signature_is_reached_by_no_name_and_no_words(tmp_path):
    from fastworkflow import intent_loop, run_listing
    from fastworkflow.review import review_plan_switch

    folder = _build_workflow(tmp_path, "wf_startup_name")
    definition = RoutingRegistry.get_definition(folder)
    assert "startup" in definition.get_command_names("*")              # it is a command: an action can run it
    assert "startup" not in context_navigation.scope_command_names(folder, "*")
    assert "startup" not in {c.name for c in intent_detector.candidates(folder, "*")}
    workflow = _on_the_root(folder, "b6-startup-name")
    assert context_navigation.command_on_path(workflow, "startup") is None
    for text in ("startup", "/startup", "startp"):
        assert intent_loop.resolve(workflow, text, core=True).command != "startup", text
    assert "startup" not in run_listing._global_line(definition, set())
    assert "library_hours" in run_listing._global_line(definition, set())
    assert "startup" not in review_plan_switch.known_plan_names(workflow)


def test_a_command_named_startup_that_has_a_signature_is_reached_by_name(tmp_path):
    # The test is the Signature, not the name: a startup command asked for in words is an ordinary command.
    folder = _build_workflow(tmp_path, "wf_startup_signature")
    (Path(folder) / "startup_action.json").unlink()
    (Path(folder) / "_commands" / "startup.py").write_text(_exec(), encoding="utf-8")
    assert "startup" in context_navigation.scope_command_names(folder, "*")


# ------------------------------------------------------------------ item 2a: build writes the startup action


def test_build_writes_the_startup_action_with_the_startup_command(tmp_path):
    import json

    from fastworkflow.build.__main__ import generate_startup_command

    app = tmp_path / "src" / "application"
    app.mkdir(parents=True)
    (app / "manager.py").write_text("class TestManager:\n    def __init__(self, filepath):\n        pass\n")
    workflow = tmp_path / "wf"
    (workflow / "_commands").mkdir(parents=True)
    assert generate_startup_command(str(workflow), str(tmp_path / "src")) is True
    assert json.loads((workflow / "startup_action.json").read_text()) == {"command_name": "startup"}
    # an author's own action file is kept
    (workflow / "startup_action.json").write_text('{"command_name": "open_shop"}')
    assert generate_startup_command(str(workflow), str(tmp_path / "src")) is True
    assert json.loads((workflow / "startup_action.json").read_text()) == {"command_name": "open_shop"}


def test_fastworkflow_run_stops_when_nothing_runs_the_startup_command(tmp_path):
    from tests.test_run_code import _cli, _no_llm_env

    folder = _build_workflow(tmp_path, "wf_no_action")
    (Path(folder) / "startup_action.json").unlink()
    assert command_preview.load_problems(folder) == []     # a client may still pass its own startup action
    # The copy is untrained, so the run needs the remote decision model's key to start at all; the run stops at the
    # startup check, before any model call.
    env = {**_no_llm_env(), "JEV_API_KEY": "placeholder-never-called"}
    cli = _cli("run_cli", folder, *_env_files(), env=env, timeout=300)
    out = " ".join((cli.stdout + cli.stderr).split())
    assert cli.returncode == 1, out
    assert ("startup has no Signature, so it runs only as the workflow's startup action, and nothing names it: "
            "write startup_action.json") in out, out


# ------------------------------------------------------------------ item 3: field validators, a body's Refused, preview


def test_a_failing_field_validator_refuses_its_field_on_both_kinds_of_command(tmp_path):
    from tests.test_parameter_validate import _command, _exec_driver, _plan_driver, _run
    from fastworkflow.parameters.validate import REFUSED, Refusal

    command = _command(tmp_path, "field_rule")
    assert _run(command, "field_rule").refusal == Refusal("no digits in a name", ("name",), None, REFUSED)
    kind, text, fields, refusal = _plan_driver(command, "field_rule")[:4]
    assert (kind, text, fields, refusal) == ("PlanningCommandInputError", "no digits in a name", ["name"],
                                             "no digits in a name")
    result, diagnostics, _ = _exec_driver(command, "field_rule")
    assert result[0] is False and result[3] == ["name"]               # asked again for that field, not ended
    assert diagnostics["validation_hook"]["refused_fields"] == ["name"]


def test_a_refusal_from_a_command_body_names_its_field_on_the_failed_answer():
    from fastworkflow.command_executor import call_response_generator
    from fastworkflow.planning.planning_phases import run_call_result

    class ResponseGenerator:
        def __call__(self, workflow, command, input):
            raise fastworkflow.Refused("no copy of that title is in stock", field="title")

    output = call_response_generator(ResponseGenerator(), None, "lend Dune", None, command_name="Library/lend")
    response = output.command_response
    assert (response.success, response.response) == (False, "no copy of that title is in stock")
    assert response.artifacts == {"refused_fields": ["title"]}

    class PlanGenerator:
        def __result__(self):
            raise fastworkflow.Refused("that branch is closed today", field=["branch"])

    answer = run_call_result(PlanGenerator, None, {}, phase_names=frozenset(), output_model=None,
                             command_name="hours_overview")
    assert (answer.success, answer.response, answer.artifacts) == (
        False, "that branch is closed today", {"refused_fields": ["branch"]})

    class Quiet:
        def __call__(self, workflow, command, input):
            raise fastworkflow.Refused("not today")

    assert call_response_generator(Quiet(), None, "x", None).command_response.artifacts == {}


_READS_THE_WORKFLOW = '''
        @model_validator(mode="after")
        def on_a_record(self, info):
            if info.context["workflow"].record is None:
                raise fastworkflow.Refused("open a record first")
            return self
'''

_REFUSES_A_VALUE = '''
        @model_validator(mode="after")
        def no_attic(self, info):
            if self.note == "attic":
                raise fastworkflow.Refused("the attic is closed")
            return self
'''


def _preview_of(tmp_path, name: str, body: str) -> str:
    from tests.test_input_model_validators import PLANNING
    from tests.test_planning_command_files import _write

    folder = _write(tmp_path / name, {name: PLANNING.format(name=name, body=body)})
    command = workflow_agent.load_planning_command_map(folder)[name]
    return command_preview.render_preview(name, "Show one record.", (), command, (), f"_commands/{name}.py")


def test_the_preview_skips_a_validator_that_cannot_run_without_a_workflow_and_says_so(tmp_path):
    text = _preview_of(tmp_path, "reads_workflow", _READS_THE_WORKFLOW)
    assert "say the record" in text                                   # rendered, not listed as refused
    assert "Not rendered" not in text and "a validator raised AttributeError" not in text
    assert ("The Input validator did not run here: it raised AttributeError while the preview renders the plan, "
            "where info.context[\"workflow\"] is None") in " ".join(text.split())


def test_the_preview_still_lists_what_a_validator_refuses(tmp_path):
    text = _preview_of(tmp_path, "refuses_value", _REFUSES_A_VALUE)
    assert "did not run here" not in text
