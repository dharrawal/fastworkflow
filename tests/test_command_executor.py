from __future__ import annotations

import os
import uuid
from contextlib import suppress
from pathlib import Path
from dotenv import dotenv_values
import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.chat_session import ChatSession


@pytest.fixture(scope="module")
def retail_workflow_path() -> str:
    """Absolute path to the retail workflow example used by the integration tests."""
    return str(Path(__file__).parent.parent.joinpath("fastworkflow", "examples", "retail_workflow").resolve())


@pytest.fixture(scope="function")
def chat_session(retail_workflow_path: str, request):
    """Spin up an in-memory chat session for each test so state cannot leak."""
    env_vars = {
        **dotenv_values("./env/.env"),
        **dotenv_values("./passwords/.env")
    }
    fastworkflow.init(env_vars)
    
    # Clear ALL caches AFTER initialization
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.CommandContextModel.load(retail_workflow_path)
    # Force rebuild of routing definition to avoid stale persisted JSON
    fastworkflow.RoutingRegistry.get_definition(retail_workflow_path, load_cached=False)

    chat_session = ChatSession()
    chat_session.start_workflow(
        retail_workflow_path,
        workflow_id_str=str(uuid.uuid4()),
        keep_alive=True,
    )

    def _teardown():
        # Nudge exit by marking not keep_alive and stopping
        chat_session._keep_alive = False
        with suppress(Exception):
            chat_session.stop_workflow()
        # Clear caches after test completes
        fastworkflow.RoutingRegistry.clear_registry()

    request.addfinalizer(_teardown)
    return chat_session


class TestCommandExecutor:
    """Basic sanity checks for the refactored CommandExecutor.perform_action."""

    def test_perform_action_simple_command(self, chat_session: ChatSession):
        """Ensure a parameter-free command can be executed successfully."""
        action = fastworkflow.Action(
            command_name="list_all_product_types",
            command="List all the product categories you have.",
            parameters={},
        )

        active_workflow = chat_session.get_active_workflow()
        result = CommandExecutor.perform_action(active_workflow, action)

        assert isinstance(result, fastworkflow.CommandOutput)
        assert result.success is True
        assert "product" in result.command_response.response.lower()


    def test_perform_action_with_parameters(self, chat_session: ChatSession):
        """Execute a command that expects parameters and verify validation passes."""
        action = fastworkflow.Action(
            command_name="find_user_id_by_email",
            command="Find the user id for john.doe@example.com",
            parameters={"email": "john.doe@example.com"},
        )

        active_workflow = chat_session.get_active_workflow()
        result = CommandExecutor.perform_action(active_workflow, action)

        assert isinstance(result, fastworkflow.CommandOutput)
        # Response text should contain a user id (pattern xyz_xyz_\d+)
        assert "user id" in result.command_response.response.lower()


# ---------------------------------------------------------------------------
# ido-8yb / F11: navigation BETWEEN INSTANCES of one context class
# ---------------------------------------------------------------------------

import shutil

from fastworkflow import auto_navigation


def _declare_enter_command(workflow_path, context_name: str, declaration: str) -> None:
    """Write an `enter_command` declaration into a COPY of the test workflow."""
    path = workflow_path / "_commands" / context_name / f"_{context_name}.py"
    source = path.read_text()
    marker = "class Context:\n"
    assert marker in source, path
    path.write_text(source.replace(
        marker, f"class Context:\n    enter_command = {declaration!r}\n", 1))


@pytest.fixture
def inherited_entry_workflow(tmp_path):
    """The real todo_list_workflow, with TodoList entered by an INHERITED command.

    `get_todo_list <id>` is owned by TodoListManager, TodoList's PARENT, so it
    can be run from inside a TodoList and lands in another TodoList: the exact
    shape -- IDO's `open_account_by_uid` from inside an Account -- where the
    context CLASS NAME does not change across a real context entry.
    """
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        Path(__file__).parent / "todo_list_workflow",
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
        ),
    )
    _declare_enter_command(workflow_path, "TodoList", "get_todo_list <id>")
    return workflow_path


class _TodoList:
    """A context instance. Two of these are two different TodoLists."""

    def __init__(self, todo_id):
        self.id = todo_id


class _Manager:
    pass


class _Workflow:
    def __init__(self, folderpath, context):
        self.folderpath = str(folderpath)
        self.current_command_context = context

    @property
    def current_command_context_name(self):
        return type(self.current_command_context).__name__.lstrip("_")


class _Session:
    def __init__(self, workflow):
        self._workflow = workflow

    def get_active_workflow(self):
        return self._workflow


def _command_output(command_name, parameters):
    output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response="ok", success=True))
    output.command_name = command_name
    output.command_parameters = parameters
    return output


class _Params:
    """A parameters model, as the executor receives it: a `model_dump()` with
    every field of the Input model, defaults included."""

    def __init__(self, **values):
        self._values = values

    def model_dump(self):
        return dict(self._values)


def _turn_scope():
    """A scope of the shape a host binds: one per TURN, not one per process.

    ido-bhf (F37). The registry refuses to resolve a handle under the process
    default scope, because that id is constant across turns while the `O`
    aliases restart each trajectory. These tests are about the RECORDER, so
    they run under the scope a real turn would have.
    """
    from fastworkflow.observation_offloading.archive import RuntimeHandleScope

    turn = uuid.uuid4().hex
    return RuntimeHandleScope(
        store_identity=f"tests-{turn}", channel_id=f"tests-{turn}",
        experiment_id="tests", task_id="tests", attempt=0, turn_key=turn)


class TestEntryBetweenInstancesOfOneClass:
    """ido-8yb / F11. The recorder read "did this command enter a context?" off
    the context CLASS NAME, so a move from one TodoList to another -- a real,
    successful, declared entry -- was never recorded, and the second instance's
    printed `O` alias could not resolve once the walk had left the context.

    Offline: a real workflow folder (so the declaration and the contract are the
    real ones) driven through the recorder with a stub session; no model, no
    backend, no trained artifact.
    """

    def setup_method(self):
        auto_navigation.reset_auto_navigation_state()

    def teardown_method(self):
        auto_navigation.reset_auto_navigation_state()

    @pytest.fixture(autouse=True)
    def _alias(self, monkeypatch):
        """The `O` alias of the execute step in flight, and the turn scope the
        entry is filed under -- both read off the offloading runtime."""
        import fastworkflow.result_handles as result_handles

        self.alias = None
        self.scope = _turn_scope()
        self.scope_id = self.scope.scope_id
        monkeypatch.setattr(
            result_handles, "current_execute_alias", lambda: self.alias)
        monkeypatch.setattr(
            result_handles, "current_scope", lambda: self.scope)

    def _enter(self, session, name_before, instance_before, command, alias, **params):
        self.alias = alias
        CommandExecutor._remember_context_entry(
            session, _command_output(command, _Params(**params)),
            name_before, instance_before)

    def _walk(self, workflow_path):
        """Enter TodoList A, then TodoList B directly, then run an ordinary
        command inside B, then leave. Returns (session, A, B, entries)."""
        a, b = _TodoList("aaaa1111"), _TodoList("bbbb2222")
        manager = _Manager()
        workflow = _Workflow(workflow_path, manager)
        session = _Session(workflow)

        workflow.current_command_context = a
        self._enter(session, "Manager", manager, "get_todo_list", "O4",
                    id="aaaa1111", page_size=25, include_completed=True)

        workflow.current_command_context = b
        self._enter(session, "TodoList", a, "get_todo_list", "O7",
                    id="bbbb2222", page_size=25, include_completed=True)

        # An ordinary command inside B: no move, so nothing to record.
        self._enter(session, "TodoList", b, "mark_completed", "O8")

        # An ordinary command that REPLACES the context object without being a
        # declared entry command is not an entry either.
        replacement = _TodoList("bbbb2222")
        workflow.current_command_context = replacement
        self._enter(session, "TodoList", b, "set_properties", "O9",
                    description="renamed")
        workflow.current_command_context = b

        workflow.current_command_context = manager  # the walk leaves TodoList
        return session, a, b, auto_navigation.context_entries(self.scope_id)

    def test_the_second_instance_is_registered_and_its_handle_resolves(
        self, inherited_entry_workflow, setup_test_environment
    ):
        _, _, b, entries = self._walk(inherited_entry_workflow)

        assert [(e.context, e.alias, e.parameters["id"]) for e in entries] == [
            ("TodoList", "O4", "aaaa1111"), ("TodoList", "O7", "bbbb2222")]

        contract = auto_navigation.entry_contract_for(
            str(inherited_entry_workflow), "TodoList")
        assert contract.command_name == "get_todo_list"
        decision = auto_navigation.decide(
            command_name="mark_completed", utterance="mark_completed O7",
            owner_contexts=["TodoList"], contracts={"TodoList": contract},
            entries=entries,
        )
        assert decision.kind == auto_navigation.DISPATCH
        assert decision.rule == auto_navigation.RULE_EXPLICIT_HANDLE
        assert decision.entry_utterance == f"get_todo_list <id>{b.id}</id>"

    def test_an_ordinary_command_adds_no_entry(
        self, inherited_entry_workflow, setup_test_environment
    ):
        """`mark_completed` and `set_properties` are not this context's declared
        entry command, so neither belongs in a table of what a handle denotes --
        not even the one that swapped the context object."""
        _, _, _, entries = self._walk(inherited_entry_workflow)
        assert [e.command_name for e in entries] == [
            "get_todo_list", "get_todo_list"]

    def test_only_the_identifying_parameter_is_a_handle(
        self, inherited_entry_workflow, setup_test_environment
    ):
        """ido-nx6 at the recorder: the dump carries the defaults, the contract
        says which value NAMES the instance."""
        _, _, _, entries = self._walk(inherited_entry_workflow)
        second = entries[1]
        assert second.required_parameters == ("id",)
        assert "page_size" in second.parameters
        assert set(second.handles()) == {"O7", "bbbb2222"}

    def test_a_workflow_that_declares_nothing_records_no_same_class_move(
        self, tmp_path, setup_test_environment
    ):
        """Without a declaration there is no entry command, nothing rule 3 could
        dispatch to, and therefore nothing to record."""
        workflow_path = tmp_path / "todo_list_workflow"
        shutil.copytree(
            Path(__file__).parent / "todo_list_workflow", workflow_path,
            ignore=shutil.ignore_patterns(
                "___command_info", "___workflow_contexts", "___convo_info",
                "__pycache__"))
        a, b = _TodoList("aaaa1111"), _TodoList("bbbb2222")
        workflow = _Workflow(workflow_path, a)
        session = _Session(workflow)
        workflow.current_command_context = b
        self._enter(session, "TodoList", a, "get_todo_list", "O7", id="bbbb2222")
        assert auto_navigation.context_entries(self.scope_id) == ()


# ---------------------------------------------------------------------------
# ido-91o / F14: one auto-navigated step files ONE entry
# ---------------------------------------------------------------------------

class TestAnAutoNavigatedStepFilesOneEntry:
    """`_auto_navigate` runs two commands through `invoke_command` and then
    returns to the OUTER frame, which started outside the context and sees it
    moved. Recording there files a second entry for the same entry, under the
    same `O` alias, carrying the ORIGINAL command's name and parameters -- and
    when the original command carries the entry contract's required parameter
    names with other values, rule 3 then sees two entries behind one handle and
    declines as ambiguous, for a handle this very dispatch produced.

    Offline: the recorder driven directly with a stub session, the marks
    `_auto_navigate` puts on the final output, and a hand-written contract.
    """

    def setup_method(self):
        auto_navigation.reset_auto_navigation_state()

    def teardown_method(self):
        auto_navigation.reset_auto_navigation_state()

    @pytest.fixture(autouse=True)
    def _runtime(self, monkeypatch):
        import fastworkflow.result_handles as result_handles

        self.alias = "O4"
        self.scope = _turn_scope()
        self.scope_id = self.scope.scope_id
        monkeypatch.setattr(
            result_handles, "current_execute_alias", lambda: self.alias)
        monkeypatch.setattr(result_handles, "current_scope", lambda: self.scope)

    CONTRACT = auto_navigation.EntryContract(
        context="TodoList", declaration="get_todo_list <id>",
        command_name="get_todo_list",
        qualified_command_name="TodoListManager/get_todo_list",
        owner_contexts=("TodoList",), required_parameters=("id",))

    @staticmethod
    def _marked(output):
        """The final output as `_auto_navigate` hands it back."""
        output.command_response.artifacts.update({
            auto_navigation.ATTR_AUTO_NAVIGATED: True,
            auto_navigation.ATTR_AUTO_NAVIGATION_RULE: 3,
            auto_navigation.ATTR_ENTERED_CONTEXT: "TodoList",
        })
        return output

    def _dispatch(self, inherited_entry_workflow):
        """One execute step: the entry step, the original step, and the outer
        frame the two of them returned to."""
        manager, a = _Manager(), _TodoList("aaaa1111")
        workflow = _Workflow(inherited_entry_workflow, manager)
        session = _Session(workflow)

        workflow.current_command_context = a  # the entry step entered A
        CommandExecutor._remember_context_entry(
            session, _command_output("get_todo_list", _Params(id="aaaa1111")),
            "Manager", manager)
        # the original command, run inside A: no move, nothing to record
        CommandExecutor._remember_context_entry(
            session, _command_output("mark_completed", _Params(id="bbbb2222")),
            "TodoList", a)
        # ...and the outer frame, which started in Manager
        CommandExecutor._remember_context_entry(
            session,
            self._marked(_command_output(
                "mark_completed", _Params(id="bbbb2222"))),
            "Manager", manager)
        return auto_navigation.context_entries(self.scope_id)

    def test_the_outer_frame_records_nothing(
        self, inherited_entry_workflow, setup_test_environment
    ):
        entries = self._dispatch(inherited_entry_workflow)
        assert [(e.command_name, e.parameters["id"], e.alias) for e in entries] == [
            ("get_todo_list", "aaaa1111", "O4")]

    def test_the_handle_the_dispatch_produced_still_resolves(
        self, inherited_entry_workflow, setup_test_environment
    ):
        """The point of the entry: `O4` denotes the TodoList the step entered,
        and rule 3 rebuilds the entry command from the values that entered it."""
        entries = self._dispatch(inherited_entry_workflow)
        decision = auto_navigation.decide(
            command_name="mark_completed", utterance="mark_completed O4",
            owner_contexts=["TodoList"], contracts={"TodoList": self.CONTRACT},
            entries=entries)
        assert decision.kind == auto_navigation.DISPATCH
        assert decision.rule == auto_navigation.RULE_EXPLICIT_HANDLE
        assert decision.entry_utterance == "get_todo_list <id>aaaa1111</id>"

    def test_an_ordinary_step_that_moves_the_context_is_still_recorded(
        self, inherited_entry_workflow, setup_test_environment
    ):
        """The skip is keyed on the marks, which live on the final output of a
        two-step dispatch and nowhere else. An agent's own entry command, with
        the same shape and no marks, records exactly as it did."""
        manager, a = _Manager(), _TodoList("aaaa1111")
        workflow = _Workflow(inherited_entry_workflow, manager)
        session = _Session(workflow)
        workflow.current_command_context = a
        CommandExecutor._remember_context_entry(
            session, _command_output("get_todo_list", _Params(id="aaaa1111")),
            "Manager", manager)
        entries = auto_navigation.context_entries(self.scope_id)
        assert [e.command_name for e in entries] == ["get_todo_list"]
