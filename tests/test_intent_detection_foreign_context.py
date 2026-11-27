"""A known command name is never answered by a context that does not own it (the foreign-name guard).

The defect: the exact-command-name matcher was scoped to the CURRENT context's command set, so a command the workflow
owns *somewhere else* -- typed verbatim, with its parameters -- was invisible to the exact and fuzzy layers and was
adjudicated by this context's classifier instead. Measured on a large multi-context workflow, that produced a hundred
misroutes across thirty-one runs, most of them silent: `fetch_result_page <handle>O9</handle> <contains>Alan Cooper</contains>`
was answered `Directory/find_permission` -> "No permissions found.", which reads as a true negative.

The guard is a layer of the intent loop (`intent_loop.resolve`, bd fix-o1lg S7), shared by the plan path and the CME:
after the exact-name miss over the whole lookup path and BEFORE fuzzy, the local model or the chain, the first word is
tested against the workflow's FULL inventory. A real command name that no scope of the path owns returns no command and
the routing hint (where the name lives and how to get there), asking no model. A name a scope of the path owns is found
by the exact layer in the scope it lives in (the old per-scope guard returned None there and the climb found it).

Two levels, both offline and deterministic:

* against a captured multi-context inventory (`tests/fixtures/multi_context_routing_inventory.json`, a copy of a real
  workflow's `___command_info/routing_definition.json`), table-driven from the observed misroutes. The inventory is
  supplied in place of the routing definition, so the guard's decision is `intent_loop.foreign_owners` for the path a
  workflow standing in the issuing context has (that context, then the global scope);
* against the real `tests/todo_list_workflow` with its real `RoutingDefinition`, so the guard is also exercised end to end
  with no inventory injected at all.

No trained artifact is read or written by any of this, and no model is loaded.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.intent import intent_loop
from fastworkflow.train import model_pipeline_training
from fastworkflow._workflows.command_metadata_extraction._commands import (
    wildcard as wildcard_module,
)
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT,
    CommandNamePrediction,
    foreign_context_hint,
)
from tests.todo_list_workflow.application.todo_item import TodoItem
from tests.todo_list_workflow.application.todo_list import TodoList
from tests.todo_list_workflow.application.todo_manager import TodoListManager

INVENTORY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "multi_context_routing_inventory.json"
)

# ---------------------------------------------------------------------------
# The misroute table (investigation section 4.2), as rows
# ---------------------------------------------------------------------------

#: (issuing context, utterance as the agent wrote it, owning contexts).
#: Every row is an observed live misroute except where marked: the utterance is
#: reproduced with its parameters because the exact-prefix matcher splits on
#: space and '(' only -- never on '<' -- which is why the XML tail is part of
#: what the classifier was given.
MISROUTE_TABLE = [
    # The original report: four occurrences across four runs, each "No permissions found."
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
    # 28x / 13 runs, SILENT -- that bucket came from the embedding cache, but the
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
# Doubles: the real decision path, an injected inventory, no model
# ---------------------------------------------------------------------------

def _refusing_router(recorder):
    """A `CommandRouter` that records any consultation and refuses to answer.

    The guard's whole claim is that nothing below it runs. A double that returns a plausible label could not tell "the
    guard fired" from "the classifier happened to agree", so this one raises instead.
    """
    class RefusingRouter:
        def __init__(self, model_artifact_path):
            self.model_artifact_path = model_artifact_path
            # The real router's pipeline; None keeps the double free of a transformer.
            self.modelpipeline = None

        def predict(self, command):
            recorder.append(command)
            raise AssertionError(
                f"the classifier was consulted for {command!r}; the guard must "
                "return before any of fuzzy / classifier runs"
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


def _fake_heads(app_workflow, *contexts):
    """A trained-looking head for each context: the router double answers for it (a scope with no head has no engine)."""
    from fastworkflow.intent import intent_detector

    for context in contexts:
        folder = intent_detector.head_folder(app_workflow.folderpath, context)
        os.makedirs(os.path.join(folder, "tinymodel.pth"), exist_ok=True)
        for name in ("label_encoder.pkl", "threshold.json"):
            with open(os.path.join(folder, name), "w") as marker:
                marker.write("{}")


class _InventoryRoutingDefinition:
    """The parts of `RoutingDefinition` the inventory reads.

    `contexts` is what the inventory helper reads; `get_command_names` is what the per-context candidate set is built
    from. Supplying one object for both is the point: a test that injected a different inventory to the guard than to
    the matcher could pass while the two disagreed in production.
    """

    def __init__(self, contexts: dict[str, list[str]]):
        self.contexts = contexts

    def get_command_names(self, context: str) -> list[str]:
        if context not in self.contexts:
            raise ValueError(f"Context '{context}' not found in the workflow.")
        return [c for c in self.contexts[context] if c != "wildcard"]

    def get_planning_command_names(self, context: str) -> list[str]:
        # The captured inventory holds execution commands only; a workflow with no
        # planning commands has none to add to the candidate set or the inventory.
        return []


@pytest.fixture(scope="module")
def inventory_contexts() -> dict[str, list[str]]:
    return json.loads(INVENTORY_FIXTURE.read_text())["contexts"]


def _declare_descends_to(workflow_path, command_file: str, context_name: str) -> None:
    """Write `@descends_to("<Context>")` on a COPY's command, the one that enters the context.

    Phase 10 (design 3.15): `enter_command` on the context class is a load error, so the entering command declares the
    move itself. The copy is what the fixture built in tmp_path, so the real routing and navigation read a real class
    attribute, and nothing here stubs the lookup being tested. The shared fixture on disk is untouched.
    """
    path = workflow_path / "_commands" / command_file
    source = path.read_text()
    marker = "class ResponseGenerator:\n"
    assert marker in source, path
    path.write_text(source.replace(
        marker, f'@fastworkflow.descends_to("{context_name}")\nclass ResponseGenerator:\n', 1))


def _copy_of_the_todo_workflow(tmp_path, *, declare_entering_command: bool = False):
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
        workflow_path,
        ignore=shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__"),
    )
    # The shipped workflow declares every move it makes (load-check sweep, batch 2); these tests are about one declared
    # entering command, so the copy starts with none and declares only that one.
    for path in (workflow_path / "_commands").rglob("*.py"):
        source = path.read_text()
        if "@fastworkflow.descends_to(" in source:
            path.write_text("".join(line for line in source.splitlines(keepends=True)
                                    if not line.startswith("@fastworkflow.descends_to(")))
    if declare_entering_command:
        _declare_descends_to(workflow_path, "TodoListManager/get_todo_list.py", "TodoList")
    return workflow_path


def _workflows_at(workflow_path, workflow_id, *, root=None, record=None):
    """A real app workflow and the real CME workflow bound to it, standing on *record* below *root*."""
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path), workflow_id_str=workflow_id)
    cme_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow},
    )
    if root is not None:
        app_workflow.root = root
    if record is not None:
        app_workflow.record = record
    return app_workflow, cme_workflow


@pytest.fixture
def todolist_workflows(tmp_path, setup_test_environment):
    """A real app workflow plus the real CME workflow bound to it, standing on an item of a list of a manager."""
    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    todo_list = TodoList(1, "groceries")
    todo_list.parent = manager
    item = TodoItem(1, "milk")
    item.parent = todo_list
    workflow_path = _copy_of_the_todo_workflow(tmp_path)
    return _workflows_at(workflow_path, f"foreign-context-{tmp_path.name}", root=manager, record=item)


@pytest.fixture
def inventory(todolist_workflows, inventory_contexts, monkeypatch):
    """The captured multi-context inventory in place of the app workflow's routing definition (the one seam the guard
    reads). Returns the app workflow's folder, whose inventory it now is."""
    app_workflow, _ = todolist_workflows
    folderpath = app_workflow.folderpath
    real_get_definition = fastworkflow.RoutingRegistry.get_definition
    definition = _InventoryRoutingDefinition(inventory_contexts)

    def get_definition(workflow_folderpath, *args, **kwargs):
        if str(workflow_folderpath) == str(folderpath):
            return definition
        return real_get_definition(workflow_folderpath, *args, **kwargs)

    monkeypatch.setattr(fastworkflow.RoutingRegistry, "get_definition", staticmethod(get_definition))
    return folderpath


def _foreign(folderpath, context, utterance):
    """The contexts that own the first word of *utterance* when a workflow standing in *context* cannot reach them: the
    path of a workflow in *context* is that context, then the global scope (asked last)."""
    on_path = [context] if context == "*" else [context, "*"]
    return intent_loop.foreign_owners(folderpath, intent_loop.first_word(utterance), on_path)


# ---------------------------------------------------------------------------
# The fixture is the real inventory
# ---------------------------------------------------------------------------

def test_fixture_carries_the_topology_the_defect_needs(inventory_contexts):
    """`*` is a peer context that nothing inherits -- the structural exposure."""
    assert "fetch_result_page" in inventory_contexts["*"]
    assert not any(
        "fetch_result_page" in commands
        for context, commands in inventory_contexts.items() if context != "*"
    )
    assert "Permission/show_holders" in inventory_contexts["Permission"]
    assert "Permission/show_holders" not in inventory_contexts["DirectoryExplorer"]


# ---------------------------------------------------------------------------
# The table: foreign to the issuing context unless the global scope owns it; the owner resolves it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "context,utterance,owners", MISROUTE_TABLE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in MISROUTE_TABLE],
)
def test_known_name_is_not_answered_by_a_context_that_cannot_reach_its_owner(
    inventory, context, utterance, owners
):
    # A name the global scope owns is on every lookup path (it is found there by the exact layer, no climb needed); any
    # other owner is foreign to a workflow that stands elsewhere.
    expected = [] if owners == ["*"] else owners
    assert sorted(_foreign(inventory, context, utterance)) == sorted(expected)


@pytest.mark.parametrize(
    "context,utterance,owners", MISROUTE_TABLE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in MISROUTE_TABLE],
)
def test_the_owning_context_still_resolves_the_same_name(inventory, context, utterance, owners):
    """The other half: where the name lives, it is not foreign (the exact layer takes it)."""
    for owner in owners:
        assert _foreign(inventory, owner, utterance) == [], owner


def test_the_bead_case_is_found_in_the_global_scope_without_a_climb(inventory):
    """`Permission -> DirectoryExplorer -> *`, the chain the misrouted call walked: the name is the global scope's, which
    every path ends with. Without the guard it stopped at the first context whose classifier produced a label --
    `Directory/find_permission`, at 0.284."""
    utterance = "fetch_result_page <handle>O9</handle> <contains>Alan Cooper</contains>"
    assert intent_loop.command_inventory(inventory)["fetch_result_page"] == ("*",)
    for context in ("Permission", "DirectoryExplorer"):
        assert _foreign(inventory, context, utterance) == []


# ---------------------------------------------------------------------------
# The separator: any whitespace ends the command name, not a space alone
# ---------------------------------------------------------------------------

#: The separator between a command name and its tail. An agent writes `list_permissions\n<scope>all</scope>` as readily as
#: it writes the same thing with a space, and on a space-only split the whole line becomes the tentative name: it matches
#: nothing, the refusal guard never fires, and a context that cannot run the command adjudicates it anyway -- the silent
#: misroute this guard exists to stop. The last two cases split either way and are here as the controls that say so.
SEPARATORS = [
    ("newline", "\n"),
    ("tab", "\t"),
    ("carriage return", "\r"),
    ("newline after space", " \n"),
    ("two spaces", "  "),
]


@pytest.mark.parametrize("name,separator", SEPARATORS, ids=[n for n, _ in SEPARATORS])
def test_any_whitespace_ends_the_command_name_for_the_guard(inventory, name, separator):
    # `list_permissions` from Identity is row 6 of the misroute table: owned by Account, 12 silent occurrences.
    text = f"list_permissions{separator}<scope>all</scope>"
    assert intent_loop.first_word(text) == "list_permissions"
    assert _foreign(inventory, "Identity", text) == ["Account"]


@pytest.mark.parametrize("name,separator", SEPARATORS, ids=[n for n, _ in SEPARATORS])
def test_the_owner_still_resolves_the_name_across_separators(inventory, name, separator):
    """The other half: the same parse must not break the owning context."""
    text = f"list_permissions{separator}<scope>all</scope>"
    assert _foreign(inventory, "Account", text) == []
    from fastworkflow.core import context_navigation
    assert context_navigation.token_names_command(intent_loop.first_word(text), "Account/list_permissions")


# ---------------------------------------------------------------------------
# Negative control: a token that is not a command anywhere is untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "context,utterance,label", NOT_A_COMMAND_ANYWHERE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in NOT_A_COMMAND_ANYWHERE],
)
def test_a_token_that_is_not_a_command_still_reaches_the_lower_layers(inventory, context, utterance, label):
    """The guard is about names the workflow owns, not names it does not: an invented name is nobody's foreign
    command, so the fuzzy layer, the local model and the chain decide it."""
    assert _foreign(inventory, context, utterance) == []


def test_a_reply_to_the_which_command_question_is_read_by_the_reader_not_refused_as_foreign(
    todolist_workflows, monkeypatch
):
    """The guard is a layer of the decision. A reply to the question names one of its candidates (or not); a first word
    that names a candidate is a pick, with the rest of the reply kept."""
    reading = intent_loop.read("get_properties <filter>x</filter>", ["TodoList/get_properties", "TodoItem/mark_completed"])
    assert (reading.kind, reading.name) == ("pick", "TodoList/get_properties")
    assert reading.rest == "<filter>x</filter>"


# ---------------------------------------------------------------------------
# The same guard against a real RoutingDefinition, nothing injected
# ---------------------------------------------------------------------------

@pytest.fixture
def item_below_a_list_root(tmp_path, setup_test_environment):
    """The workflow stands on an item below a LIST that is the root: the manager is not on the chain, so the manager's
    commands are foreign to it."""
    todo_list = TodoList(1, "groceries")
    item = TodoItem(1, "milk")
    item.parent = todo_list
    workflow_path = _copy_of_the_todo_workflow(tmp_path)
    app_workflow, cme_workflow = _workflows_at(
        workflow_path, f"foreign-real-{tmp_path.name}", root=todo_list, record=item)
    _fake_heads(app_workflow, "TodoItem", "TodoList", "*")
    return app_workflow, cme_workflow


def test_real_workflow_foreign_name_returns_no_command(item_below_a_list_root, monkeypatch, setup_test_environment):
    """`create_todo_list` belongs to TodoListManager, which is not on this chain: it may not be answered here."""
    _, cme_workflow = item_below_a_list_root
    consulted: list[str] = []
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router(consulted))
    predictor = CommandNamePrediction(cme_workflow)

    result = predictor.predict("create_todo_list my list")

    assert result.command is None and result.outcome == "none" and result.decided_by == "foreign_name"
    assert result.owners == ("TodoListManager",) and "TodoListManager context" in result.hint
    assert predictor.trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    assert not consulted


def test_real_workflow_a_name_on_the_chain_is_found_in_the_scope_that_owns_it(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """The manager is the chain's root: from the item, the name is exact in the manager's scope, whose record it runs on
    (what the old walk found by climbing from the item)."""
    app_workflow, cme_workflow = todolist_workflows
    consulted: list[str] = []
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router(consulted))

    result = CommandNamePrediction(cme_workflow).predict("create_todo_list my list")

    assert result.command == "TodoListManager/create_todo_list"
    assert result.context == "TodoListManager" and result.record is app_workflow.root
    assert not consulted


def test_real_workflow_owning_context_still_resolves_by_exact_prefix(
    tmp_path, monkeypatch, setup_test_environment
):
    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    _, cme_workflow = _workflows_at(
        _copy_of_the_todo_workflow(tmp_path), f"foreign-owner-{tmp_path.name}", root=manager, record=manager)
    consulted: list[str] = []
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router(consulted))

    predictor = CommandNamePrediction(cme_workflow)
    result = predictor.predict("create_todo_list my list")

    assert result.command == "TodoListManager/create_todo_list"
    assert predictor.trace["matcher_layer"] == "exact_prefix"
    assert not consulted


def test_real_workflow_inherited_name_is_not_foreign(todolist_workflows, monkeypatch, setup_test_environment):
    """Inheritance is ownership: TodoList has TodoItem as a base.

    The guard reads the same resolved command lists the candidate set is built from, so an inherited name is owned, not
    foreign.
    """
    app_workflow, cme_workflow = todolist_workflows
    monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router([]))
    app_workflow.record = app_workflow.get_parent(app_workflow._current_command_context)   # stand on the list

    result = CommandNamePrediction(cme_workflow).predict("assign_to bob")

    assert result.command is not None
    assert result.command.split("/")[-1] == "assign_to"
    assert "assign_to" in intent_loop.command_inventory(app_workflow.folderpath)


def test_inventory_excludes_reserved_labels(todolist_workflows, setup_test_environment):
    """`wildcard` names no command; treating it as one would swallow the escalation utterances the classifier is trained
    on."""
    app_workflow, _ = todolist_workflows
    inventory = intent_loop.command_inventory(app_workflow.folderpath)
    assert "wildcard" not in inventory
    assert "parameter_value" not in inventory
    assert "create_todo_list" in inventory


# ---------------------------------------------------------------------------
# Mixed-case command names: a context never refuses its own command
# ---------------------------------------------------------------------------

MIXED_CASE_CONTEXTS = {
    "*": [],
    "Child": ["Child/ListPermissions", "Child/show_metadata"],
    "Sibling": ["Sibling/ShowHolders"],
}


@pytest.fixture
def mixed_case_folder(todolist_workflows, monkeypatch):
    """An app workflow that reports mixed-case command names. The inventory is keyed lowercase while each context keeps
    its commands' own spelling; that pair is exactly what disagreed."""
    app_workflow, _ = todolist_workflows
    folderpath = app_workflow.folderpath
    real_get_definition = fastworkflow.RoutingRegistry.get_definition
    definition = _InventoryRoutingDefinition(MIXED_CASE_CONTEXTS)

    def get_definition(workflow_folderpath, *args, **kwargs):
        if str(workflow_folderpath) == str(folderpath):
            return definition
        return real_get_definition(workflow_folderpath, *args, **kwargs)

    monkeypatch.setattr(fastworkflow.RoutingRegistry, "get_definition", staticmethod(get_definition))
    return folderpath


@pytest.mark.parametrize("utterance", [
    "ListPermissions",
    "listpermissions",
    "LISTPERMISSIONS <scope>all</scope>",
])
def test_a_mixed_case_command_resolves_in_its_own_context(mixed_case_folder, utterance):
    """The token is lowercased and the context's own key is not. Before the fix the exact match missed and the guard
    named the current context as the foreign owner, refusing a command that belongs here."""
    from fastworkflow.core import context_navigation

    assert _foreign(mixed_case_folder, "Child", utterance) == []
    assert context_navigation.token_names_command(
        intent_loop.first_word(utterance).lower(), "Child/ListPermissions")


def test_a_mixed_case_command_owned_elsewhere_is_still_foreign(mixed_case_folder):
    assert _foreign(mixed_case_folder, "Child", "ShowHolders") == ["Sibling"]
    hint = intent_loop.routing_hint_for(mixed_case_folder, "showholders", ["Sibling"])
    assert "Sibling context" in hint


def test_the_current_context_is_never_its_own_foreign_owner(inventory):
    """Whatever the candidate set says, the context being asked is dropped from the owners it reports."""
    assert _foreign(inventory, "Permission", "show_holders") == []
    assert _foreign(inventory, "Identity", "show_holders") == ["Permission"]


# ---------------------------------------------------------------------------
# The hint at the end of the walk (owner-approved scope addition, 2026-09-15)
# ---------------------------------------------------------------------------

class TestTheHintTextItself:
    """A pure composer: what it says, and what it refuses to invent."""

    def test_an_entity_context_hint_names_the_context_and_how_to_enter_it(self):
        hint = foreign_context_hint(
            "list_permissions", ["Account"],
            {"Account": ["open_account_by_uid <account_uid>"]})
        assert "'list_permissions'" in hint
        assert "Account context" in hint
        assert "open_account_by_uid <account_uid>" in hint

    def test_a_stateless_workspace_hint_names_the_command_that_enters_it(self):
        """No uid to supply: the workspace is entered by one bare command."""
        hint = foreign_context_hint(
            "list_findings", ["ControlsMonitor"],
            {"ControlsMonitor": ["open_controls_monitor"]})
        assert "ControlsMonitor context" in hint
        assert "'open_controls_monitor'" in hint
        assert "<" not in hint.split("Enter it with:")[1]

    def test_several_owners_are_all_named(self):
        hint = foreign_context_hint(
            "who_has_access_to", ["Directory", "DirectoryExplorer"], {})
        assert "Directory, DirectoryExplorer contexts" in hint

    def test_several_owners_each_get_their_own_entering_command(self):
        """A merged list cannot say which command opens which context."""
        hint = foreign_context_hint(
            "get_properties", ["TodoItem", "TodoList", "Board"],
            {"TodoItem": ["open_item <id>"], "TodoList": ["get_todo_list <id>"],
             "Board": []})
        assert "TodoItem, TodoList, Board contexts" in hint
        assert "Enter TodoItem with: 'open_item <id>'." in hint
        assert "Enter TodoList with: 'get_todo_list <id>'." in hint
        assert "Enter Board with" not in hint
        assert "Enter it with" not in hint

    def test_a_shared_entering_command_is_named_beside_each_context(self):
        hint = foreign_context_hint(
            "show", ["A", "B"], {"A": ["open_ab"], "B": ["open_ab"]})
        assert "Enter A with: 'open_ab'." in hint
        assert "Enter B with: 'open_ab'." in hint

    def test_without_a_declaration_it_names_the_context_and_nothing_else(self):
        """Nothing in the routing definition records which command enters a
        context, and guessing one from a command's NAME would bake one
        workflow's spelling conventions into the framework. Naming the owner
        alone is the honest floor."""
        hint = foreign_context_hint("list_permissions", ["Account"], {})
        assert "Account context" in hint
        assert "Enter it with" not in hint
        assert "what_can_i_do" in hint

    def test_without_a_declaration_it_never_says_then_after_nothing(self):
        """'Then run it there' needs an entry instruction before it; with no
        declaration the hint says where to go instead."""
        hint = foreign_context_hint("list_permissions", ["Account"], {})
        assert "Then run" not in hint
        assert "Move into that context first, then run 'list_permissions' there." in hint

    def test_several_undeclared_owners_are_one_of_those_contexts(self):
        hint = foreign_context_hint("who_has_access_to", ["Directory", "DirectoryExplorer"], {})
        assert "Then run" not in hint
        assert "Move into one of those contexts first" in hint

    def test_a_declared_entry_still_reads_enter_then_run(self):
        hint = foreign_context_hint(
            "list_permissions", ["Account"],
            {"Account": ["open_account_by_uid <account_uid>"]})
        assert ("Enter it with: 'open_account_by_uid <account_uid>'. "
                "Then run 'list_permissions' there.") in hint



class TestTheHintIsComposedAndRecorded:
    def test_the_guard_records_the_hint_on_its_own_span(self, inventory, todolist_workflows, monkeypatch):
        app_workflow, cme_workflow = todolist_workflows
        owners = _foreign(inventory, "Identity", "list_permissions")
        hint = intent_loop.routing_hint_for(inventory, "list_permissions", owners)
        assert owners == ["Account"] and "Account context" in hint

    def test_the_inventory_declares_no_entering_command_so_the_hint_names_the_context(
        self, inventory, setup_test_environment
    ):
        """Recorded rather than asserted as a wish: the captured inventory declares no `enter_command` on any context,
        so its hints name the owning context and stop there."""
        hint = intent_loop.routing_hint_for(inventory, "list_permissions", ["Account"])
        assert "Enter it with" not in hint

    def test_the_guard_declines_and_records_no_flag(self, item_below_a_list_root, monkeypatch, setup_test_environment):
        """There is no auto-navigation setting for the routing event to carry, so the guard's own record says only what
        the guard did: it declined, at the known-name foreign-context layer."""
        _, cme_workflow = item_below_a_list_root
        monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router([]))
        predictor = CommandNamePrediction(cme_workflow)
        output = predictor.predict("create_todo_list")

        assert output.command is None
        assert "auto_navigation_enabled" not in predictor.trace
        assert predictor.trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
        assert predictor.trace["known_name_foreign_context_hint"] == output.hint

    def test_a_resolved_prediction_carries_no_hint(self, todolist_workflows, monkeypatch, setup_test_environment):
        """A command of the path is reached by the exact layer and needs no hint at all."""
        _, cme_workflow = todolist_workflows
        monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router([]))
        predictor = CommandNamePrediction(cme_workflow)
        output = predictor.predict("create_todo_list my list")

        assert output.command == "TodoListManager/create_todo_list"
        assert output.hint is None and output.owners == ()
        assert "known_name_foreign_context_hint" not in predictor.trace
        assert "auto_navigation_enabled" not in predictor.trace


class TestTheDeclaredEnteringCommandIsRead:
    """The entering command is read from the descend edges (`@descends_to` on the command)."""

    @pytest.fixture
    def declared_workflows(self, tmp_path):
        manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
        todo_list = TodoList(1, "groceries")
        todo_list.parent = manager
        item = TodoItem(1, "milk")
        item.parent = todo_list
        workflow_path = _copy_of_the_todo_workflow(tmp_path, declare_entering_command=True)
        # The item is the root: neither the list nor the manager is on its chain.
        item.parent = None
        return _workflows_at(workflow_path, f"enter-command-{tmp_path.name}", root=item, record=item)

    def test_the_hint_names_the_owner_and_its_declared_entering_command(
        self, declared_workflows, monkeypatch, setup_test_environment
    ):
        app_workflow, cme_workflow = declared_workflows
        monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router([]))
        from fastworkflow.core.context_navigation import entering_commands

        assert entering_commands(app_workflow.folderpath, "TodoList") == ["get_todo_list <id>"]
        output = CommandNamePrediction(cme_workflow).predict("get_properties_of_nothing_here_but_mark_completed")
        assert output.command is None or output.hint is None   # an invented name is nobody's foreign command
        hint = intent_loop.routing_hint_for(app_workflow.folderpath, "mark_completed", ["TodoList"])
        assert "TodoList context" in hint
        assert "get_todo_list <id>" in hint

    def test_an_undeclared_context_resolves_to_no_entering_command(self, declared_workflows, setup_test_environment):
        from fastworkflow.core.context_navigation import entering_commands

        app_workflow, _ = declared_workflows
        assert entering_commands(app_workflow.folderpath, "TodoItem") == []
        assert entering_commands(app_workflow.folderpath, "no-such-context") == []


# ---------------------------------------------------------------------------
# The hint reaches the caller: wildcard's end-of-walk path
# ---------------------------------------------------------------------------

class TestTheHintReachesTheFailureMessage:
    """The decision names no command and carries a hint; say where the name lives.

    The CME wildcard command is driven directly with doubles for the two collaborators it calls out to - the predictor
    and the executor - because what is under test is the lines between them: the hint becomes the message the caller
    actually reads, and a decision without one is the misunderstanding.
    `TestTheHintPathOnARealWorkflow` below drives the same path with nothing replaced but the classifier.
    """

    class _Workflow:
        """The CME workflow: a context dict and an app workflow."""

        def __init__(self, app_workflow):
            self._context = {"app_workflow": app_workflow}

        @property
        def context(self):
            return self._context

        @context.setter
        def context(self, value):
            self._context = value

        def end_command_processing(self):
            self._context.pop("command", None)
            intent_loop.clear(self._context)

    class _AppWorkflow:
        def __init__(self):
            self._current_command_context_name = "Identity"
            self.folderpath = "/nowhere"
            self.context = {}

    @staticmethod
    def _predictor_declining(hint, asked):
        class DecliningPredictor:
            def __init__(self, cme_workflow):
                pass

            def predict(self, command):
                asked.append(command)
                return intent_loop.Intent(
                    intent_loop.NONE, decided_by=intent_loop.BY_FOREIGN_NAME if hint else "", hint=hint,
                    owners=("Account",) if hint else ())
        return DecliningPredictor

    def _run(self, monkeypatch, hint):
        from fastworkflow._workflows.command_metadata_extraction._commands import (
            wildcard as wildcard_command,
        )

        asked: list[str] = []
        actions_run: list[str] = []
        monkeypatch.setattr(wildcard_command, "CommandNamePrediction", self._predictor_declining(hint, asked))
        monkeypatch.setattr(wildcard_command.intent_loop, "command_choices", lambda app_workflow: [])

        def perform_action(workflow, action):
            actions_run.append(action.command_name)
            return fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response="I couldn't determine which available command "
                             "matches your request.",
                    success=False))

        monkeypatch.setattr(
            wildcard_command.CommandExecutor, "perform_action",
            staticmethod(perform_action))

        cme_workflow = self._Workflow(self._AppWorkflow())
        output = wildcard_command.ResponseGenerator()(
            cme_workflow, "list_permissions", None)
        return output, asked, actions_run, cme_workflow

    def test_the_message_names_the_owner_and_how_to_get_there(self, monkeypatch, setup_test_environment):
        hint = foreign_context_hint(
            "list_permissions", ["Account"],
            {"Account": ["open_account_by_uid <account_uid>"]})
        output, asked, actions_run, cme_workflow = self._run(monkeypatch, hint)

        response = output.response
        assert response == hint
        assert "Account context" in response
        assert "open_account_by_uid <account_uid>" in response
        assert output.routed is False
        assert output.handled is True
        assert asked == ["list_permissions"]
        # Not a misunderstanding: no you_misunderstood, and no clarification
        # stage waiting to swallow the entering command the hint names.
        assert actions_run == []
        assert intent_loop.waiting_for(cme_workflow.context) is None
        assert "command" not in cme_workflow.context

    def test_a_walk_with_no_hint_is_left_exactly_as_it_was(self, monkeypatch, setup_test_environment):
        """Free text that no context could route still gets the plain message: the hint is for a name the workflow owns,
        and nothing else."""
        output, _, actions_run, cme_workflow = self._run(monkeypatch, None)
        assert output.response == (
            "I couldn't determine which available command matches your request.")
        assert actions_run == ["ErrorCorrection/you_misunderstood"]
        assert intent_loop.waiting_for(cme_workflow.context) == intent_loop.WAITING_INTENT
        assert intent_loop.pending_record(cme_workflow.context)["misunderstood"] is True       # the question is pending


# ---------------------------------------------------------------------------
# The reply to you_misunderstood: nothing matched is not a KeyError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utterance", [
    "gibberish that matches nothing",
    "list_holders <filter>Alan Cooper</filter>",
])
def test_an_unmatched_reply_is_asked_again_and_then_ends_with_what_can_i_do(utterance):
    """The reply to the question that listed every command is read by the one reader: a reply that is not one of them is
    asked once more (`That did not name one of them`), and the second one ends the question with the listing."""
    record = intent_loop.open_question(
        "tell me a joke", [{"name": "IntentDetection/what_can_i_do", "description": "", "closest": False}],
        ambiguous=False, misunderstood=True)
    intent_loop.ask(record)
    first = intent_loop.answer(record, utterance)
    assert first.kind == intent_loop.ASK_AGAIN and first.question.startswith("That did not name one of them.")
    assert intent_loop.answer(record, utterance).kind == intent_loop.EXHAUSTED




# ---------------------------------------------------------------------------
# The end of the walk on a real workflow, then the message after it
# ---------------------------------------------------------------------------

class TestTheHintPathOnARealWorkflow:
    """The real CME wildcard command, executor and `you_misunderstood`, over a copy of `tests/todo_list_workflow` whose
    get_todo_list declares `@descends_to` TodoList. Only the classifier is replaced; no LLM is called and no trained
    model is read.

    From TodoListManager the chain is `TodoListManager -> *`, and `get_properties` is owned by TodoItem and TodoList,
    both off it.
    """

    @pytest.fixture
    def workflows(self, tmp_path):
        workflow_path = _copy_of_the_todo_workflow(tmp_path, declare_entering_command=True)
        manager = TodoListManager()
        app_workflow, cme_workflow = _workflows_at(
            workflow_path, f"hint-path-{tmp_path.name}", root=manager, record=manager)
        _fake_heads(app_workflow, "TodoListManager", "*")
        return app_workflow, cme_workflow

    def test_a_foreign_name_gets_the_hint_and_the_entering_command_then_routes(
        self, workflows, monkeypatch, setup_test_environment
    ):
        app_workflow, cme_workflow = workflows
        monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router([]))

        output = wildcard_module.ResponseGenerator()(cme_workflow, "get_properties", None)

        response = output.response
        assert output.routed is False
        assert output.handled is True
        assert "'get_properties' is a command of the TodoItem, TodoList contexts" in response
        assert "'get_todo_list <id>'" in response
        assert "I couldn't determine" not in response

        # Doing what the hint says: the next message is the entering command, and it is routed by ordinary intent
        # detection in this context.
        assert intent_loop.waiting_for(cme_workflow.context) is None
        assert "command" not in cme_workflow.context
        following = CommandNamePrediction(cme_workflow).predict("get_todo_list 1")
        assert following.command == "TodoListManager/get_todo_list"

    @pytest.mark.usefixtures("no_planner_key")
    def test_free_text_still_ends_at_you_misunderstood_and_a_stray_reply_is_answered(
        self, workflows, monkeypatch, setup_test_environment
    ):
        """No known name, so no hint: every context's classifier escalates, the walk ends at the root, and the real
        `you_misunderstood` runs. A reply that matches nothing in the question is asked again, and the second ends it
        with the listing instead of raising."""
        app_workflow, cme_workflow = workflows
        monkeypatch.setattr(model_pipeline_training, "CommandRouter", _labelling_router("wildcard"))

        output = wildcard_module.ResponseGenerator()(cme_workflow, "tell me a joke", None)

        assert output.response.startswith(
            "I couldn't determine which available command matches your request.")
        assert output.handled is True
        assert intent_loop.pending_record(cme_workflow.context)["misunderstood"] is True
        assert output.plan_choice["misunderstood"] is True and output.plan_choice["choices"]

        again = wildcard_module.ResponseGenerator()(cme_workflow, "zxq vvv", None)
        assert again.routed is False and "That did not name one of them." in again.response
        last = wildcard_module.ResponseGenerator()(cme_workflow, "zxq vvv", None)
        assert last.handled is True and last.response.startswith("Workflow Definition:")
        assert intent_loop.waiting_for(cme_workflow.context) is None

    @pytest.mark.usefixtures("no_planner_key")
    def test_a_foreign_name_as_the_misunderstanding_reply_gets_the_hint(
        self, workflows, monkeypatch, setup_test_environment
    ):
        """Free text puts the workflow in the misunderstanding question; the reply is a real command owned off the
        chain. The reply ends with the hint and the question closed, instead of a list of this context's commands."""
        app_workflow, cme_workflow = workflows
        monkeypatch.setattr(model_pipeline_training, "CommandRouter", _labelling_router("wildcard"))
        wildcard_module.ResponseGenerator()(cme_workflow, "tell me a joke", None)
        assert intent_loop.pending_record(cme_workflow.context)["misunderstood"] is True

        monkeypatch.setattr(model_pipeline_training, "CommandRouter", _refusing_router([]))
        output = wildcard_module.ResponseGenerator()(cme_workflow, "get_properties", None)

        response = output.response
        assert output.routed is False
        assert output.handled is True
        assert "'get_properties' is a command of the TodoItem, TodoList contexts" in response
        assert "Enter TodoList with: 'get_todo_list <id>'." in response
        assert intent_loop.waiting_for(cme_workflow.context) is None
        assert "command" not in cme_workflow.context
