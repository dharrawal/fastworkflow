"""Agent-facing `what_can_i_do` compaction (ido-mn1.6.3).

Two independent changes are covered:

1. The RENDERING. `_compact_command_listing` replaces the YAML-like listing for
   the agent's `available_commands` prelude and its `what_can_i_do` tool. It
   drops the `outputs:` and `examples:` blocks — measured as the bulk — and
   keeps what a caller needs to emit a valid `execute_workflow_query`: the
   command name, the PARAMETER NAMES, whether each is required, the docstring
   and the `available_from` hints the ido skills walk.

2. The per-turn MEMO. A `what_can_i_do` tool call that would repeat a listing
   already produced in the same turn returns a short reference instead.

`CommandMetadataAPI` itself is untouched, so the CME `what_can_i_do` command,
the MCP tool descriptions and the skill catalogue still read the full text.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow import workflow_agent
from fastworkflow.workflow_agent import (
    _compact_command_listing,
    _what_can_i_do_tool_observation,
)


def _todo_list_path() -> str:
    return str((Path(__file__).parent / "todo_list_workflow").resolve())


def _cme_workflow_path() -> str:
    return fastworkflow.get_internal_workflow_path("command_metadata_extraction")


@pytest.fixture
def todo_list_env():
    RoutingRegistry.clear_registry()
    fastworkflow.init({"NOT_FOUND": "NOT_FOUND"})
    yield
    RoutingRegistry.clear_registry()


# ----------------------------------------------------------------------
# 1. Rendering
# ----------------------------------------------------------------------

def test_compact_listing_is_smaller_and_keeps_the_calling_contract(todo_list_env):
    subject, cme = _todo_list_path(), _cme_workflow_path()
    full = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="TodoList",
    )
    compact = _compact_command_listing(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="TodoList",
    )

    # A meaningful reduction, not a rounding error.
    assert len(compact) < 0.7 * len(full), (len(full), len(compact))

    # Same header, so the prelude and the tool observation still announce the context.
    assert compact.splitlines()[0] == full.splitlines()[0]

    # Every command still listed, one line each.
    meta = CommandMetadataAPI.get_enhanced_command_info(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="TodoList",
    )
    names = sorted(c["name"] for c in meta["commands"])
    body = compact.splitlines()[1:]
    assert len(body) == len(names)
    for name, line in zip(names, body):
        assert line.startswith(f"- {name}")

    # The bulk is gone.
    assert "outputs:" not in compact
    assert "examples:" not in compact


def test_compact_listing_names_parameters_so_commands_can_be_written(todo_list_env):
    """The old renderer printed a parameter's DESCRIPTION where its name belongs.

    `execute_workflow_query` takes `command_name <param_name>value</param_name>`,
    so the parameter name is the one field the agent cannot do without.
    """
    subject, cme = _todo_list_path(), _cme_workflow_path()
    compact = _compact_command_listing(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="TodoList",
    )
    line = next(
        ln for ln in compact.splitlines() if ln.startswith("- add_child_todoitem")
    )
    meta = CommandMetadataAPI.get_enhanced_command_info(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="TodoList",
    )
    cmd = next(c for c in meta["commands"] if c["name"] == "add_child_todoitem")
    assert cmd["inputs"], "fixture must have inputs for this test to mean anything"
    for inp in cmd["inputs"]:
        assert f"{inp['name']}" in line

    old = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="TodoList",
    )
    # The regression this guards: the full listing never names `description`.
    assert "- description," not in old


def test_optional_parameters_are_marked_and_types_simplified(todo_list_env):
    subject, cme = _todo_list_path(), _cme_workflow_path()
    compact = _compact_command_listing(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="TodoList",
    )
    line = next(ln for ln in compact.splitlines() if ln.startswith("- set_properties"))
    assert "typing.Optional" not in line
    assert "?: str" in line


def test_global_context_header_says_global(todo_list_env):
    compact = _compact_command_listing(
        subject_workflow_path=_todo_list_path(),
        cme_workflow_path=_cme_workflow_path(),
        active_context_name="*",
    )
    assert compact.startswith("Commands available in the current context (global):")


def test_rendering_failure_falls_back_to_the_full_listing(todo_list_env, monkeypatch):
    """A metadata failure degrades to the old listing, not to an empty menu."""
    subject, cme = _todo_list_path(), _cme_workflow_path()
    expected = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=subject,
        cme_workflow_path=cme,
        active_context_name="TodoList",
    )

    real = CommandMetadataAPI.get_enhanced_command_info
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("metadata unavailable")
        return real(**kwargs)

    monkeypatch.setattr(
        CommandMetadataAPI, "get_enhanced_command_info", staticmethod(flaky)
    )
    assert (
        _compact_command_listing(
            subject_workflow_path=subject,
            cme_workflow_path=cme,
            active_context_name="TodoList",
        )
        == expected
    )


# ----------------------------------------------------------------------
# 2. Per-turn memo
# ----------------------------------------------------------------------

class _FakeHost:
    """The duck-typed surface `_what_can_i_do_tool_observation` reads."""

    def __init__(self, context_name="Account", turn_key="turn-1", agent=None):
        self.current_turn_key = turn_key
        self._context_name = context_name
        self._memo: dict = {}
        self._memo_turn = turn_key
        self.workflow_tool_agent = agent

    # mirrors WorkflowExecutionContext.command_listing_memo
    def command_listing_memo(self):
        if self._memo_turn != self.current_turn_key:
            self._memo_turn = self.current_turn_key
            self._memo = {}
        return self._memo

    def get_active_workflow(self):
        return SimpleNamespace(
            folderpath="/nowhere",
            current_command_context_name=self._context_name,
        )


@pytest.fixture
def listing(monkeypatch):
    """Stub the renderer so the memo is tested, not the metadata layer."""
    state = {"text": "Commands available in the current context (Account):\n- show_x"}
    monkeypatch.setattr(
        workflow_agent, "_what_can_i_do", lambda chat_session_obj: state["text"]
    )
    return state


def test_repeat_call_in_one_turn_returns_a_short_reference(listing):
    agent = SimpleNamespace(current_trajectory={"observation_0": "x", "observation_1": "y"})
    host = _FakeHost(agent=agent)

    first = _what_can_i_do_tool_observation(host)
    assert first == listing["text"]

    second = _what_can_i_do_tool_observation(host)
    assert second != listing["text"]
    assert len(second) < len(listing["text"]) + 200
    assert "unchanged since observation 2" in second
    assert "execute_workflow_query" in second
    assert "Account" in second


def test_changed_context_returns_the_full_listing(listing):
    host = _FakeHost()
    assert _what_can_i_do_tool_observation(host) == listing["text"]

    host._context_name = "Identity"
    listing["text"] = "Commands available in the current context (Identity):\n- show_y"
    assert _what_can_i_do_tool_observation(host) == listing["text"]


def test_changed_command_set_in_the_same_context_returns_the_listing(listing):
    """The fingerprint is over the listing, so a surface change re-lists."""
    host = _FakeHost()
    assert _what_can_i_do_tool_observation(host) == listing["text"]
    listing["text"] += "\n- show_z"
    assert _what_can_i_do_tool_observation(host) == listing["text"]


def test_memo_does_not_leak_across_turns(listing):
    host = _FakeHost()
    assert _what_can_i_do_tool_observation(host) == listing["text"]
    assert _what_can_i_do_tool_observation(host) != listing["text"]

    host.current_turn_key = "turn-2"
    assert _what_can_i_do_tool_observation(host) == listing["text"]


def test_no_turn_key_disables_the_memo(listing):
    host = _FakeHost(turn_key=None)
    assert _what_can_i_do_tool_observation(host) == listing["text"]
    assert _what_can_i_do_tool_observation(host) == listing["text"]


def test_host_without_a_memo_always_gets_the_listing(listing):
    host = SimpleNamespace(
        current_turn_key="turn-1",
        get_active_workflow=lambda: SimpleNamespace(
            folderpath="/nowhere", current_command_context_name="Account"
        ),
    )
    assert _what_can_i_do_tool_observation(host) == listing["text"]
    assert _what_can_i_do_tool_observation(host) == listing["text"]


def test_reference_points_at_the_prelude_across_plan_leaves(listing):
    """Leaf 2's ReAct run does not carry leaf 1's trajectory.

    Naming an observation index from the other leaf would point at something the
    agent cannot see, so the reference names the `available_commands` prelude —
    which every leaf is given and which the context-change observer keeps current.
    """
    agent = SimpleNamespace(current_trajectory={"observation_0": "x"})
    host = _FakeHost(agent=agent)
    assert _what_can_i_do_tool_observation(host) == listing["text"]

    # New leaf: react.forward() rebinds current_trajectory to a fresh dict.
    agent.current_trajectory = {}
    second = _what_can_i_do_tool_observation(host)
    assert "available_commands" in second
    assert "observation" not in second.split("(context")[0].replace(
        "available_commands", ""
    )


def test_suspend_resume_keeps_the_memo_and_a_rehydrated_context_re_lists(listing):
    """A suspended turn keeps its key, so the memo survives in-process.

    A resume in a fresh process rebuilds the context with an empty memo, which
    degrades to one extra full listing rather than to a stale reference.
    """
    agent = SimpleNamespace(current_trajectory={"observation_0": "x"})
    host = _FakeHost(agent=agent)
    assert _what_can_i_do_tool_observation(host) == listing["text"]
    # Suspension does not mint a new turn key.
    assert _what_can_i_do_tool_observation(host) != listing["text"]

    rehydrated = _FakeHost(agent=agent, turn_key=host.current_turn_key)
    assert _what_can_i_do_tool_observation(rehydrated) == listing["text"]


def test_intent_misunderstood_is_never_memoised(listing, monkeypatch):
    """It exists because the agent used a wrong name; it must show the names."""
    host = _FakeHost()
    assert _what_can_i_do_tool_observation(host) == listing["text"]
    assert workflow_agent._intent_misunderstood(host) == listing["text"]
    assert workflow_agent._intent_misunderstood(host) == listing["text"]


def test_wec_memo_resets_on_turn_key_change():
    """The one WorkflowExecutionContext hunk: lifetime only."""
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    host = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    host._command_listing_memo = {}
    host._command_listing_memo_turn = None
    host._turn_key = "t1"

    memo = host.command_listing_memo()
    memo["k"] = "v"
    assert host.command_listing_memo() == {"k": "v"}

    host._turn_key = "t2"
    assert host.command_listing_memo() == {}


def test_agent_tool_is_wired_to_the_memo(monkeypatch, todo_list_env):
    """The `what_can_i_do` TOOL, not just the helper, consults the memo."""
    from fastworkflow.chat_session import ChatSession
    from fastworkflow.mcp_server import FastWorkflowMCPServer
    from fastworkflow.workflow_agent import initialize_workflow_tool_agent

    seen: list[object] = []
    monkeypatch.setattr(
        workflow_agent,
        "_what_can_i_do_tool_observation",
        lambda host: seen.append(host) or "memoised",
    )

    chat_session = ChatSession(run_as_agent=True)
    agent = initialize_workflow_tool_agent(FastWorkflowMCPServer(chat_session))
    assert agent.tools["what_can_i_do"]() == "memoised"
    assert len(seen) == 1
