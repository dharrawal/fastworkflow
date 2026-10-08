"""The unavailable-command refusal names the path, and the parameters it still needs.

``go_up`` follows the live parent, not every parent the hierarchy file allows.
``reset_context`` is used when it is shorter than climbing. A descend step says
``needs <field>`` when the command cannot run without it, and ``<field> required
to enter <context>`` when the command runs either way and only navigates when
the field is present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

import fastworkflow
from fastworkflow import context_navigation
from fastworkflow.command_executor import CommandNotFoundError
from fastworkflow.context_navigation import (
    DescendEdge,
    collect_descend_edges,
    live_context_chain,
    plan_navigation,
    render_unavailable_command,
    unavailable_command_message,
)
from fastworkflow.workflow_agent import _explicit_agent_command

IDO_WORKFLOW = Path(__file__).resolve().parents[2] / "ido" / "ido_workflow"


@pytest.mark.skipif(
    not (IDO_WORKFLOW / "workflow_runtime.json").is_file(),
    reason="ido workflow is not checked out beside fastworkflow",
)
def test_deferred_planner_skips_mixin_contexts_and_navigates_to_explorer():
    import fastworkflow
    from fastworkflow.command_metadata_api import CommandMetadataAPI

    class Workflow:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "*"
        current_command_context = None
        root_command_context = None

        def get_parent(self, obj):
            return None

    fastworkflow.init({"NOT_FOUND": "NOT_FOUND"})
    text = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=str(IDO_WORKFLOW),
        cme_workflow_path=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        active_context_name="*",
        navigation_workflow=Workflow(),
    )
    assert "after entering the Directory context" not in text
    assert "after entering the DirectoryExplorer context" in text
    assert (
        "Before executing find_account, navigate to DirectoryExplorer as follows: "
        "open_directory"
    ) in text
    assert "navigate to Directory as follows:" not in text


def test_deferred_planner_command_matches_find_identity_shape():
    import fastworkflow
    from fastworkflow.command_metadata_api import CommandMetadataAPI

    class Workflow:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "*"
        current_command_context = None
        root_command_context = None

        def get_parent(self, obj):
            return None

    block = CommandMetadataAPI.format_deferred_planner_command(
        subject_workflow_path=str(IDO_WORKFLOW),
        cme_workflow_path=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        owning_context_name="DirectoryExplorer",
        qualified_command_name="Directory/find_identity",
        active_context_name="*",
        navigation_workflow=Workflow(),
    )
    assert block.startswith("- find_identity\n")
    assert "Find identities by name, login or email" in block
    assert (
        "Before executing find_identity, navigate to DirectoryExplorer as follows: "
        "open_directory"
    ) in block


def _steps(path):
    return [step.command for step in path]


def test_global_reaches_a_workspace_by_its_descend_command():
    edges = (DescendEdge("*", "open_directory", "DirectoryExplorer"),)
    path = plan_navigation("*", ["DirectoryExplorer"], ("*",), edges)["DirectoryExplorer"]
    assert _steps(path) == ["open_directory"]
    assert path[0].needs == ()
    assert path[0].required_to_enter == ()


def test_go_up_follows_the_live_parent_not_every_declared_one():
    """Account may sit under DirectoryExplorer or Identity. The object in hand
    is under Identity, so the climb is two steps, not the static shortcut."""
    chain = ("Account", "Identity", "DirectoryExplorer", "*")
    path = plan_navigation("Account", ["DirectoryExplorer"], chain, ())["DirectoryExplorer"]
    assert _steps(path) == ["go_up", "go_up"]


def test_one_go_up_beats_reset_when_the_parent_is_the_target():
    chain = ("Identity", "DirectoryExplorer", "*")
    edges = (DescendEdge("*", "open_directory", "DirectoryExplorer"),)
    path = plan_navigation("Identity", ["DirectoryExplorer"], chain, edges)["DirectoryExplorer"]
    assert _steps(path) == ["go_up"]


def test_reset_is_used_when_it_is_shorter_than_climbing():
    chain = ("ControlFinding", "ControlsMonitor", "*")
    edges = (DescendEdge("*", "open_directory", "DirectoryExplorer"),)
    path = plan_navigation(
        "ControlFinding", ["DirectoryExplorer"], chain, edges)["DirectoryExplorer"]
    assert _steps(path) == ["reset_context", "open_directory"]


def test_equal_length_prefers_the_step_that_needs_nothing():
    edges = (
        DescendEdge("DirectoryExplorer", "list_accounts", "Account", required_to_enter=("account_uid",)),
        DescendEdge("DirectoryExplorer", "open_account_free", "Account"),
    )
    path = plan_navigation(
        "DirectoryExplorer", ["Account"], ("DirectoryExplorer", "*"), edges)["Account"]
    assert _steps(path) == ["open_account_free"]


def test_a_shorter_conditional_step_is_kept_and_names_its_parameter():
    edges = (
        DescendEdge("Identity", "list_accounts", "Account", required_to_enter=("account_uid",)),
        DescendEdge("DirectoryExplorer", "open_account_by_uid", "Account", needs=("account_uid",)),
    )
    chain = ("Identity", "DirectoryExplorer", "*")
    path = plan_navigation("Identity", ["Account"], chain, edges)["Account"]
    assert _steps(path) == ["list_accounts"]
    assert path[0].required_to_enter == ("account_uid",)


def test_no_declared_descend_has_no_path():
    path = plan_navigation("*", ["Subscription"], ("*",), ())["Subscription"]
    assert path is None


def test_the_message_for_find_identity_says_open_directory():
    edges = (DescendEdge("*", "open_directory", "DirectoryExplorer"),)
    paths = plan_navigation("*", ["DirectoryExplorer"], ("*",), edges)
    text = render_unavailable_command("find_identity", "*", ["DirectoryExplorer"], paths)
    assert text == (
        "Command 'find_identity' is not available in the current context 'global'. "
        "It is available in: 'DirectoryExplorer'. "
        "From here to 'DirectoryExplorer': open_directory. Then retry 'find_identity'."
    )
    assert "go_up or reset_context" not in text


def test_the_message_lists_both_kinds_of_caveat():
    edges = (
        DescendEdge("*", "open_directory", "DirectoryExplorer"),
        DescendEdge(
            "DirectoryExplorer", "open_account_by_uid", "Account", needs=("account_uid",)),
        DescendEdge(
            "Identity", "list_accounts", "Account", required_to_enter=("account_uid",)),
    )
    opened = plan_navigation("*", ["Account"], ("*",), edges)
    from_global = render_unavailable_command("list_permissions", "*", ["Account"], opened)
    assert (
        "open_directory, then open_account_by_uid <account_uid> (needs account_uid)"
        in from_global
    )

    chain = ("Identity", "DirectoryExplorer", "*")
    listed = plan_navigation("Identity", ["Account"], chain, edges)
    from_identity = render_unavailable_command("list_permissions", "Identity", ["Account"], listed)
    assert (
        "list_accounts <account_uid> (account_uid required to enter Account)"
        in from_identity
    )


def test_each_home_context_gets_its_own_path():
    edges = (
        DescendEdge("*", "open_directory", "DirectoryExplorer"),
        DescendEdge("*", "open_controls_monitor", "ControlsMonitor"),
    )
    paths = plan_navigation("*", ["ControlsMonitor", "DirectoryExplorer"], ("*",), edges)
    text = render_unavailable_command("list_findings", "*", ["ControlsMonitor", "DirectoryExplorer"], paths)
    assert "From here to 'ControlsMonitor': open_controls_monitor." in text
    assert "From here to 'DirectoryExplorer': open_directory." in text


def test_live_chain_stops_at_global_when_the_parent_is_none():
    class Explorer:
        pass

    class Workflow:
        current_command_context_name = "DirectoryExplorer"
        current_command_context = Explorer()
        root_command_context = None

        def get_parent(self, obj):
            return None

    assert live_context_chain(Workflow()) == ("DirectoryExplorer", "*")


class _Model:
    def inherited_base_contexts(self, name):
        if name == "DirectoryExplorer":
            return {"Directory"}
        return set()


class _App:
    contexts = {
        "*": ["open_directory"],
        "Directory": ["Directory/find_identity"],
        "DirectoryExplorer": ["Directory/find_identity"],
    }
    context_model = _Model()

    def get_command_names(self, context):
        return self.contexts[context]


class _Cme:
    def get_command_names(self, context):
        return []


class _Workflow:
    folderpath = "app"
    current_command_context_name = "*"
    current_command_context = None
    root_command_context = None

    def get_parent(self, obj):
        return None


def test_the_agent_guard_reports_the_path(monkeypatch):
    internal = fastworkflow.get_internal_workflow_path("command_metadata_extraction")

    class Registry:
        @staticmethod
        def get_definition(path, load_cached=True):
            if path == internal:
                return _Cme()
            return _App()

    monkeypatch.setattr(fastworkflow, "RoutingRegistry", Registry)
    monkeypatch.setattr(
        context_navigation,
        "collect_descend_edges",
        lambda path: (DescendEdge("*", "open_directory", "DirectoryExplorer"),),
    )
    with pytest.raises(CommandNotFoundError) as caught:
        _explicit_agent_command("find_identity", _Workflow())
    assert "open_directory" in str(caught.value)
    assert "go_up or reset_context" not in str(caught.value)


def test_a_spec_root_prefix_matches_the_unqualified_global_command(tmp_path, monkeypatch):
    """IDO/open_directory in the manifest is open_directory on the global context."""
    manifest = {
        "schema_version": 1,
        "manifest_version": "1.0.0",
        "contexts": {
            "*": {"occupiable": True},
            "DirectoryExplorer": {"occupiable": True},
            "Identity": {"occupiable": True},
            "Account": {"occupiable": True},
            "Directory": {"occupiable": False},
        },
        "commands": {
            "IDO/open_directory": {
                "navigation_effect": {
                    "kind": "descend",
                    "target_context": "DirectoryExplorer",
                    "remains_active": True,
                },
            },
            "DirectoryExplorer/open_account_by_uid": {
                "navigation_effect": {
                    "kind": "descend",
                    "target_context": "Account",
                    "remains_active": True,
                },
            },
            "Identity/list_accounts": {
                "navigation_effect": {
                    "kind": "descend",
                    "target_context": "Account",
                    "remains_active": True,
                    "when_parameter_present": "account_uid",
                },
            },
            "IDO/open_subscription_manager": {
                "navigation_effect": {"kind": "none"},
            },
        },
    }
    (tmp_path / "workflow_runtime.json").write_text(json.dumps(manifest), encoding="utf-8")

    class Input(BaseModel):
        account_uid: str

    class Routing:
        contexts = {
            "*": ["open_directory", "open_subscription_manager"],
            "DirectoryExplorer": ["DirectoryExplorer/open_account_by_uid"],
            "Identity": ["Identity/list_accounts"],
            "Account": [],
            "Directory": ["Directory/find_identity"],
            "SubscriptionManager": [],
        }

        def get_command_class(self, command_name, module_type):
            if command_name == "DirectoryExplorer/open_account_by_uid":
                return Input
            return None

    class Registry:
        @staticmethod
        def get_definition(path, load_cached=True):
            return Routing()

    monkeypatch.setattr(context_navigation, "RoutingRegistry", Registry)
    monkeypatch.setattr(
        context_navigation,
        "declared_entry_commands",
        lambda folder, context: {
            "DirectoryExplorer": ["open_directory"],
            "Account": ["open_account_by_uid <account_uid>"],
            "SubscriptionManager": ["open_subscription_manager"],
        }.get(context, []),
    )

    edges = {(edge.source, edge.command, edge.target): edge
             for edge in collect_descend_edges(str(tmp_path))}
    assert edges["*", "open_directory", "DirectoryExplorer"].needs == ()
    account = edges["DirectoryExplorer", "open_account_by_uid", "Account"]
    assert account.needs == ("account_uid",)
    assert account.required_to_enter == ()
    listed = edges["Identity", "list_accounts", "Account"]
    assert listed.required_to_enter == ("account_uid",)
    assert listed.needs == ()
    assert not any(edge.command == "open_subscription_manager" for edge in edges.values())

    class Here(_Workflow):
        folderpath = str(tmp_path)

    text = unavailable_command_message(Here(), "find_identity", "*", ["DirectoryExplorer"])
    assert "open_directory" in text
    assert "needs" not in text.split("open_directory", 1)[1]


@pytest.mark.skipif(
    not (IDO_WORKFLOW / "workflow_runtime.json").is_file(),
    reason="ido workflow is not checked out beside fastworkflow",
)
def test_ido_global_find_identity_path_is_open_directory():
    class Workflow:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "*"
        current_command_context = None
        root_command_context = None

        def get_parent(self, obj):
            return None

    text = unavailable_command_message(
        Workflow(), "find_identity", "*", ["DirectoryExplorer"])
    assert "From here to 'DirectoryExplorer': open_directory." in text
    assert "go_up or reset_context" not in text

    class DirectoryExplorer:
        pass

    class Identity:
        pass

    class InIdentity:
        folderpath = str(IDO_WORKFLOW)
        current_command_context_name = "Identity"
        current_command_context = Identity()
        root_command_context = None

        def get_parent(self, obj):
            if isinstance(obj, Identity):
                return DirectoryExplorer()
            return None

    from_identity = unavailable_command_message(
        InIdentity(), "list_permissions", "Identity", ["Account"])
    assert "list_accounts <account_uid>" in from_identity
    assert "account_uid required to enter Account" in from_identity

    edges = collect_descend_edges(str(IDO_WORKFLOW))
    opened = plan_navigation("*", ["Account"], ("*",), edges)["Account"]
    assert [step.command for step in opened] == ["open_directory", "open_account_by_uid"]
    assert "account_uid" in opened[1].needs or "account_uid" in opened[1].required_to_enter
