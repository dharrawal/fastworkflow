"""R1 (ido-8ps.8): a known command name is never answered by a foreign context.

The defect (`ido-8ps.6.1`): the exact-command-name matcher in
`CommandNamePrediction._predict_impl` is scoped to the CURRENT context's command
set, so a command the workflow owns *somewhere else* -- typed verbatim, with its
parameters -- is invisible to the exact and fuzzy layers and is adjudicated by
this context's classifier instead. Across 31 stored IDO runs that produced 100
misroutes, 69 of them silent: `fetch_result_page <handle>O9</handle>
<contains>Alan Cooper</contains>` was answered `Directory/find_permission` ->
"No permissions found.", which reads as a true negative.

The guard: after the context-scoped exact-prefix miss and BEFORE fuzzy, cache or
classifier, the first token is tested against the workflow's FULL inventory. A
real command name that this context does not own returns `command_name=None` at
once, which is already the signal that drives the parent-chain walk in
`_commands/wildcard.py:98-106`.

Two levels, both offline and deterministic:

* against the real `ido_workflow` inventory (`tests/fixtures/ido_routing_inventory.json`,
  copied from its `___command_info/routing_definition.json`), table-driven from
  the misroute table in section 4.2 of the investigation. The real `predict` path
  runs; only the command inventory is supplied, and the classifier double raises
  if it is ever consulted;
* against the real `tests/todo_list_workflow` with its real `RoutingDefinition`,
  so the guard is also exercised end to end with no inventory injected at all.

No trained artifact is read or written by any of this, and no model is loaded.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT,
    CommandNamePrediction,
)

INVENTORY_FIXTURE = Path(__file__).parent / "fixtures" / "ido_routing_inventory.json"
#: Where the fixture was copied from. Present on the machine the IDO series runs
#: on and absent everywhere else, so the drift check below skips rather than fails.
LIVE_IDO_ROUTING_DEFINITION = Path(
    "/home/drawal/rl/ido/ido_workflow/___command_info/routing_definition.json"
)

INTENT_DETECTION = fastworkflow.NLUPipelineStage.INTENT_DETECTION


# ---------------------------------------------------------------------------
# The misroute table (investigation section 4.2), as rows
# ---------------------------------------------------------------------------

#: (issuing context, utterance as the agent wrote it, owning contexts).
#: Every row is an observed live misroute except where marked: the utterance is
#: reproduced with its parameters because the exact-prefix matcher splits on
#: space and '(' only -- never on '<' -- which is why the XML tail is part of
#: what the classifier was given.
MISROUTE_TABLE = [
    # The bead's case: four occurrences across four runs, each "No permissions found."
    ("Permission", "fetch_result_page <handle>O9</handle> <contains>Alan Cooper</contains>",
     ["*"]),
    ("DirectoryExplorer", "fetch_result_page <handle>O6</handle> <contains>Alan Cooper</contains>",
     ["*"]),
    # 7x / 2 runs, loud only because fetch_result_page has a required parameter.
    ("DirectoryExplorer", "show_holders <filter>Alan Cooper</filter>", ["Permission"]),
    ("Account", "show_holders <filter>Brandon Miller</filter>", ["Permission"]),
    ("Identity", "show_holders", ["Permission"]),
    # 12x / 4 runs, SILENT: answered with the directory-wide 64-permission catalogue.
    ("Identity", "list_permissions", ["Account"]),
    # 4x / 2 runs, SILENT.
    ("ControlsMonitor", "list_accounts", ["Application", "Identity", "Repository"]),
    ("DirectoryExplorer", "list_accounts", ["Application", "Identity", "Repository"]),
    # 4x / 3 runs, SILENT.
    ("DirectoryExplorer", "list_findings", ["ControlsMonitor"]),
    ("Account", "list_findings", ["ControlsMonitor"]),
    # 28x / 13 runs, SILENT -- that bucket was the embedding cache (R2), but the
    # name is foreign to the issuing context either way and the guard returns
    # before the cache is consulted.
    ("*", "open_identity_by_uid <identity_uid>81b86cf622ed7f1f3be7b964852e0f42</identity_uid>",
     ["DirectoryExplorer"]),
    # 1x, SILENT: answered by who_am_i.
    ("*", "who_has_access_to <resource>Active Directory</resource>",
     ["Directory", "DirectoryExplorer"]),
    # 1x, SILENT: answered by ControlCatalog/list_controls.
    ("Account", "list_entitlements", ["Identity"]),
    # 1x, SILENT: answered by open_reconciliation.
    ("ControlsMonitor", "open_account_by_uid <account_uid>3f2a</account_uid>",
     ["DirectoryExplorer"]),
]

#: First tokens that name no command of this workflow. `list_holders` and
#: `show_portrait` are names the agent invented; `search_memory` is the DSPy
#: agent-side tool, which the workflow has never heard of and which was answered
#: "No permissions found." three times. All three must still reach fuzzy and the
#: classifier -- the guard is about names the workflow owns, not about names it
#: does not.
#: The third element is the label the classifier double returns: the real router
#: returns a label the context owns, and `resolve_fully_qualified_command_name`
#: looks it up in that context's dict, so an invented label would fail for a
#: reason that has nothing to do with this guard.
NOT_A_COMMAND_ANYWHERE = [
    ("DirectoryExplorer", "list_holders <filter>Alan Cooper</filter>", "find_permission"),
    ("Account", "search_memory <question>who holds AD admin rights</question>",
     "show_metadata"),
    ("Permission", "show_portrait", "show_metadata"),
    ("DirectoryExplorer", "which identities hold the compliance officer permission",
     "find_permission"),
]


# ---------------------------------------------------------------------------
# Doubles: the real predict path, an injected inventory, no model
# ---------------------------------------------------------------------------

def _refusing_router(recorder):
    """A `CommandRouter` that records any consultation and refuses to answer.

    The guard's whole claim is that nothing below it runs. A double that returns
    a plausible label could not tell "the guard fired" from "the classifier
    happened to agree", so this one raises instead.
    """
    class RefusingRouter:
        def __init__(self, model_artifact_path):
            self.model_artifact_path = model_artifact_path
            # `cache_match` computes an embedding only when this is not None, so
            # None keeps the embedding cache off without loading a transformer.
            self.modelpipeline = None

        def predict(self, command):
            recorder.append(command)
            raise AssertionError(
                f"the classifier was consulted for {command!r}; the guard must "
                "return before any of fuzzy / cache / classifier runs"
            )

        def predict_with_details(self, command):
            return self.predict(command)

    return RefusingRouter


def _labelling_router(label):
    class LabellingRouter:
        def __init__(self, model_artifact_path):
            self.modelpipeline = None

        def predict(self, command):
            return self.predict_with_details(command)[0]

        def predict_with_details(self, command):
            return [label], {}

    return LabellingRouter


class _InventoryRoutingDefinition:
    """The parts of `RoutingDefinition` the prediction path reads.

    `contexts` is what the new inventory helper reads; `get_command_names` is
    what the per-context candidate set is built from. Supplying one object for
    both is the point: a test that injected a different inventory to the guard
    than to the matcher could pass while the two disagreed in production.
    """

    def __init__(self, contexts: dict[str, list[str]]):
        self.contexts = contexts

    def get_command_names(self, context: str) -> list[str]:
        if context not in self.contexts:
            raise ValueError(f"Context '{context}' not found in the workflow.")
        return [c for c in self.contexts[context] if c != "wildcard"]


@pytest.fixture(scope="module")
def ido_contexts() -> dict[str, list[str]]:
    return json.loads(INVENTORY_FIXTURE.read_text())["contexts"]


@pytest.fixture
def todolist_workflows(tmp_path):
    """A real app workflow plus the real CME workflow bound to it."""
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
        ),
    )
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path),
        workflow_id_str=f"foreign-context-{tmp_path.name}",
    )
    cme_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow},
    )
    return app_workflow, cme_workflow


@pytest.fixture
def ido_predictor(todolist_workflows, ido_contexts, monkeypatch):
    """A predictor whose app workflow reports the real IDO command inventory.

    The CME workflow keeps its real routing definition; only the app workflow's
    is replaced, which is exactly the seam the guard reads.
    """
    _, cme_workflow = todolist_workflows
    predictor = CommandNamePrediction(cme_workflow)
    app_folderpath = predictor.app_workflow_folderpath
    real_get_definition = fastworkflow.RoutingRegistry.get_definition
    ido_definition = _InventoryRoutingDefinition(ido_contexts)

    def get_definition(workflow_folderpath, *args, **kwargs):
        if str(workflow_folderpath) == str(app_folderpath):
            return ido_definition
        return real_get_definition(workflow_folderpath, *args, **kwargs)

    monkeypatch.setattr(
        fastworkflow.RoutingRegistry, "get_definition", staticmethod(get_definition)
    )
    return predictor


# ---------------------------------------------------------------------------
# The fixture is the real inventory
# ---------------------------------------------------------------------------

def test_inventory_fixture_matches_the_live_ido_workflow():
    """The copied inventory must not drift from the workflow it was copied from."""
    if not LIVE_IDO_ROUTING_DEFINITION.exists():
        pytest.skip("the IDO workflow is not on this machine")
    live = json.loads(LIVE_IDO_ROUTING_DEFINITION.read_text())["contexts"]
    fixture = json.loads(INVENTORY_FIXTURE.read_text())["contexts"]
    assert {k: sorted(v) for k, v in live.items()} == fixture


def test_fixture_carries_the_topology_the_defect_needs(ido_contexts):
    """`*` is a peer context that nothing inherits -- the structural exposure."""
    assert "fetch_result_page" in ido_contexts["*"]
    assert not any(
        "fetch_result_page" in commands
        for context, commands in ido_contexts.items() if context != "*"
    )
    assert "Permission/show_holders" in ido_contexts["Permission"]
    assert "Permission/show_holders" not in ido_contexts["DirectoryExplorer"]


# ---------------------------------------------------------------------------
# The table: None from the foreign context, resolved from the owner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "context,utterance,owners", MISROUTE_TABLE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in MISROUTE_TABLE],
)
def test_known_name_is_not_answered_by_a_foreign_context(
    ido_predictor, monkeypatch, setup_test_environment, context, utterance, owners
):
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))

    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        context, utterance, INTENT_DETECTION, nlu_trace)

    assert result.command_name is None
    assert result.error_msg is None
    assert nlu_trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    assert nlu_trace["known_name_foreign_context"] is True
    assert sorted(nlu_trace["known_name_owner_contexts"]) == sorted(owners)
    assert not consulted


@pytest.mark.parametrize(
    "context,utterance,owners", MISROUTE_TABLE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in MISROUTE_TABLE],
)
def test_the_owning_context_still_resolves_the_same_name(
    ido_predictor, monkeypatch, setup_test_environment, context, utterance, owners
):
    """The other half of the walk: where the name lives, it resolves exactly."""
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))
    token = utterance.split(" ", 1)[0]

    for owner in owners:
        nlu_trace: dict = {}
        result = ido_predictor._predict_impl(
            owner, utterance, INTENT_DETECTION, nlu_trace)
        assert result.command_name.split("/")[-1] == token, (owner, token)
        assert nlu_trace["matcher_layer"] == "exact_prefix"
    assert not consulted


def test_the_walk_lands_the_bead_case_on_the_root_context(
    ido_predictor, monkeypatch, setup_test_environment
):
    """`Permission -> DirectoryExplorer -> *`, the chain the bead's call walked.

    This is the loop `_commands/wildcard.py:98-106` runs: keep asking the parent
    while the name is None. Before R1 it stopped at the first context whose
    classifier produced a label -- `Directory/find_permission`, at 0.284.
    """
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router([]))
    utterance = "fetch_result_page <handle>O9</handle> <contains>Alan Cooper</contains>"

    resolved = None
    visited = []
    for context in ("Permission", "DirectoryExplorer", "*"):
        nlu_trace: dict = {}
        output = ido_predictor._predict_impl(
            context, utterance, INTENT_DETECTION, nlu_trace)
        visited.append((context, output.command_name, nlu_trace["matcher_layer"]))
        if output.command_name is not None:
            resolved = output.command_name
            break

    assert resolved == "fetch_result_page"
    assert visited == [
        ("Permission", None, MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT),
        ("DirectoryExplorer", None, MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT),
        ("*", "fetch_result_page", "exact_prefix"),
    ]


# ---------------------------------------------------------------------------
# Negative control: a token that is not a command anywhere is untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "context,utterance,label", NOT_A_COMMAND_ANYWHERE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in NOT_A_COMMAND_ANYWHERE],
)
def test_a_token_that_is_not_a_command_still_reaches_the_lower_layers(
    ido_predictor, monkeypatch, setup_test_environment, context, utterance, label
):
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _labelling_router(label))

    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        context, utterance, INTENT_DETECTION, nlu_trace)

    # Whatever the lower layers decide, they decided it -- the guard did not.
    assert nlu_trace["matcher_layer"] in {"fuzzy_prematch", "classifier"}
    assert "known_name_foreign_context" not in nlu_trace
    assert result.command_name is not None


def test_free_text_still_reaches_the_classifier_and_its_label_wins(
    ido_predictor, monkeypatch, setup_test_environment
):
    monkeypatch.setattr(
        intent_detection, "CommandRouter",
        _labelling_router("find_permission"))

    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        "DirectoryExplorer", "who can read the payroll repository",
        INTENT_DETECTION, nlu_trace)

    assert nlu_trace["matcher_layer"] == "classifier"
    assert result.command_name == "Directory/find_permission"


def test_the_guard_is_scoped_to_intent_detection(
    ido_predictor, monkeypatch, setup_test_environment
):
    """Clarification stages match a constrained suggestion set, not this one.

    There, "the current context does not own this name" is the ordinary case
    rather than a misroute, so the guard must not fire and end the stage.
    """
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _labelling_router("show_metadata"))

    nlu_trace: dict = {}
    ido_predictor._predict_impl(
        "Permission", "fetch_result_page <handle>O9</handle>",
        fastworkflow.NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
        nlu_trace)

    assert nlu_trace["matcher_layer"] != MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    assert "known_name_foreign_context" not in nlu_trace


# ---------------------------------------------------------------------------
# The same guard against a real RoutingDefinition, nothing injected
# ---------------------------------------------------------------------------

def test_real_workflow_foreign_name_returns_none(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """`create_todo_list` belongs to TodoListManager; TodoItem may not answer it."""
    _, cme_workflow = todolist_workflows
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoItem", "create_todo_list my list", INTENT_DETECTION)

    assert result.command_name is None
    assert result.error_msg is None
    assert not consulted


def test_real_workflow_owning_context_still_resolves_by_exact_prefix(
    todolist_workflows, monkeypatch, setup_test_environment
):
    _, cme_workflow = todolist_workflows
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoListManager", "create_todo_list my list", INTENT_DETECTION)

    assert result.command_name == "TodoListManager/create_todo_list"
    assert not consulted


def test_real_workflow_inherited_name_is_not_foreign(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """Inheritance is ownership: TodoList has TodoItem as a base.

    The guard reads the same resolved command lists the candidate set is built
    from, so an inherited name is owned, not foreign.
    """
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router([]))

    predictor = CommandNamePrediction(cme_workflow)
    result = predictor.predict("TodoList", "assign_to bob", INTENT_DETECTION)

    assert result.command_name is not None
    assert result.command_name.split("/")[-1] == "assign_to"
    assert "assign_to" in predictor.command_inventory()


def test_inventory_excludes_reserved_labels(
    todolist_workflows, setup_test_environment
):
    """`wildcard` names no command; treating it as one would swallow the
    escalation utterances the classifier is trained on."""
    _, cme_workflow = todolist_workflows
    inventory = CommandNamePrediction(cme_workflow).command_inventory()
    assert "wildcard" not in inventory
    assert "parameter_value" not in inventory
    assert "create_todo_list" in inventory
