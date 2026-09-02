"""EXP-012: exact identity, span parsing, occupancy (FW-REQ-003/004/005).

Three defects, and the tests that hold each one closed:

* a command the caller named exactly still went through the model — there was
  no exact-identity path at all;
* a command not callable here became a misroute, because nothing could say
  "known, but not from where you are";
* the parser removed **every** occurrence of the command name, so a parameter
  whose value contained it came back mutilated.

The stop condition for this experiment is that classifier predictions do not
move (requirements §12.3), so the tests below are also where the scope of the
change is pinned: qualified identities always resolve, bare simple names only
under the manifest feature gate.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_resolution import (
    FEATURE_ID,
    CommandCapabilityIndex,
    EffectiveCapability,
    ExactResolution,
    is_qualified_identity,
    parse_command,
    simple_name_resolution_enforced,
    strip_command_token,
)


# A workflow shaped like the one the defects were found in: a base context two
# entity contexts inherit, plus a base-of-a-base, plus a collision.
RAW_CONTEXTS = {
    "Identity": {"/": ["Identity/list_accounts"], "base": ["Resource"]},
    "Account": {"/": ["Account/list_permissions"], "base": ["Resource"]},
    "Resource": {"/": ["Resource/show_properties", "Resource/open_portrait"]},
    "Explorer": {"/": ["Explorer/browse"], "base": ["Lookup", "Catalog"]},
    "Lookup": {"/": ["Lookup/find_thing"]},
    "Catalog": {"/": ["Catalog/find_thing", "Catalog/list_controls"]},
    "*": {"/": ["who_am_i"]},
}


@pytest.fixture
def index() -> CommandCapabilityIndex:
    return CommandCapabilityIndex.build(
        RAW_CONTEXTS, core_command_names=("IntentDetection/go_up",)
    )


# ----------------------------------------------------------------------
# FW-REQ-003 clauses 1-3: exact identity resolves, without a model call
# ----------------------------------------------------------------------


def test_an_effective_context_alias_resolves(index):
    resolution = index.resolve_exact("Identity/show_properties", "Identity")
    assert resolution.resolved
    assert resolution.capability.definition.definition_id == "Resource/show_properties"
    assert resolution.capability.effective_context_name == "Identity"
    assert resolution.capability.source == "inherited"


def test_a_canonical_definition_id_resolves_through_inheritance(index):
    """`Resource/show_properties` is callable in Identity even though the alias differs."""
    resolution = index.resolve_exact("Resource/show_properties", "Identity")
    assert resolution.resolved
    assert resolution.capability.display_alias == "Identity/show_properties"


def test_an_unambiguous_simple_name_resolves(index):
    resolution = index.resolve_exact("show_properties", "Identity")
    assert resolution.resolved
    assert resolution.capability.definition.definition_id == "Resource/show_properties"


def test_an_own_definition_overrides_an_inherited_one():
    """Precedence 1: a concrete own definition beats the base it shadows."""
    index = CommandCapabilityIndex.build(
        {
            "Identity": {"/": ["Identity/show_properties"], "base": ["Resource"]},
            "Resource": {"/": ["Resource/show_properties"]},
        }
    )
    resolution = index.resolve_exact("show_properties", "Identity")
    assert resolution.capability.definition.definition_id == "Identity/show_properties"
    assert resolution.capability.source == "own"
    # And the inherited one is still *known*, not erased — which is the
    # information `commands()` throws away.
    assert {
        cap.definition.definition_id
        for cap in index.collisions("Identity", "show_properties")
    } == {"Identity/show_properties", "Resource/show_properties"}


def test_a_core_command_is_available_everywhere_at_lowest_precedence(index):
    resolution = index.resolve_exact("go_up", "Identity")
    assert resolution.resolved
    assert resolution.capability.source == "core"


# ----------------------------------------------------------------------
# FW-REQ-003 clause 4 / FW-REQ-005 clause 1: typed, not arbitrary
# ----------------------------------------------------------------------


def test_an_ambiguous_simple_name_is_structured_ambiguity_not_a_pick(index):
    resolution = index.resolve_exact("find_thing", "Explorer")
    assert resolution.failure == "ambiguous-route"
    assert {cap.definition.definition_id for cap in resolution.candidates} == {
        "Lookup/find_thing",
        "Catalog/find_thing",
    }


def test_an_ambiguous_alias_does_not_resolve_either(index):
    """Arch §10.1 precedence 4: an equal-rank collision has no simple-name alias.

    Both bases project onto the same `Explorer/find_thing` alias, so resolving
    it would be picking the winner the precedence rules refuse to pick.
    """
    assert index.resolve_exact("Explorer/find_thing", "Explorer").failure == (
        "ambiguous-route"
    )
    # Naming the definition is the way through, and it works.
    assert index.resolve_exact("Lookup/find_thing", "Explorer").resolved


def test_a_known_command_that_is_not_callable_here_is_typed(index):
    resolution = index.resolve_exact("list_permissions", "Identity")
    assert resolution.failure == "not-callable-here"
    assert not resolution.is_unknown
    assert "Identity" in resolution.detail


def test_a_qualified_name_for_an_unreachable_definition_is_typed(index):
    resolution = index.resolve_exact("Account/list_permissions", "Identity")
    assert resolution.failure == "not-callable-here"


def test_only_genuinely_unknown_text_continues_to_the_classifier(index):
    """Step 6 of the resolution order, and the only outcome that may fall through."""
    resolution = index.resolve_exact("what accounts does this person have", "Identity")
    assert resolution.failure == "unknown-command"
    assert resolution.is_unknown


# ----------------------------------------------------------------------
# FW-REQ-003 clause 5: the parser removes one token, not every match
# ----------------------------------------------------------------------


def test_a_parameter_containing_the_command_name_survives():
    """The defect, stated as its symptom.

    `command.replace("add_tag", "")` on `add_tag <tag>add_tag</tag>` returned
    `<tag></tag>` — a plausible-looking string with the value deleted.
    """
    raw = "add_tag <tag>add_tag</tag>"
    assert strip_command_token(raw, "Resource/add_tag") == "<tag>add_tag</tag>"
    assert raw.replace("add_tag", "").strip() == "<tag></tag>", "the old behavior"


def test_only_the_leading_token_span_is_removed():
    parsed = parse_command("show_properties uid=show_properties-1")
    assert parsed.command_token == "show_properties"
    assert parsed.argument_text == "uid=show_properties-1"


def test_a_parenthesised_call_form_parses():
    """The token ends at `(`, matching what intent detection has always done.

    `command.split(" ", 1)[0].split("(", 1)[0]` is the rule the runtime used;
    the span parser reproduces it rather than inventing a new one, so a call
    form is not suddenly a different command name.
    """
    parsed = parse_command("who_am_i()")
    assert parsed.command_token == "who_am_i"
    assert parsed.argument_text == "()"


def test_the_assistant_prefix_is_recorded_not_lost():
    parsed = parse_command("@show_properties uid=1")
    assert parsed.explicit_assistant_prefix is True
    assert parsed.command_token == "show_properties"


def test_a_line_that_does_not_start_with_the_command_is_left_alone():
    """Nothing mid-string is removed, which is the whole point."""
    assert (
        strip_command_token("please show_properties for me", "show_properties")
        == "please show_properties for me"
    )


def test_an_empty_line_parses_to_nothing():
    parsed = parse_command("   ")
    assert parsed.command_token == ""
    assert parsed.argument_text == ""


# ----------------------------------------------------------------------
# The slash-command misread, found by the FastAPI startup-command test
# ----------------------------------------------------------------------


def test_a_slash_command_prefix_is_not_a_context_qualifier():
    """`/add_two_numbers` is a chat client's prefix, not `Context/name`.

    Read as a qualified identity, its empty first half became a context, and a
    perfectly callable command was refused with `not-callable-here` in context
    `*`. Both halves have to be non-empty.
    """
    assert is_qualified_identity("/add_two_numbers") is False
    assert is_qualified_identity("Identity/show_properties") is True
    assert is_qualified_identity("Identity/") is False
    assert is_qualified_identity("a//b") is False
    assert is_qualified_identity("add_two_numbers") is False


# ----------------------------------------------------------------------
# FW-REQ-004: occupancy
# ----------------------------------------------------------------------


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


def test_occupancy_without_a_manifest_is_compatibility_mode(
    initialized_fastworkflow, todo_workflow_path
):
    """A workflow that declared nothing keeps current behavior (arch §7.1).

    "Nobody said" is not "not enterable": treating an undeclared context as
    unenterable would break every workflow without a manifest, which is most of
    them.
    """
    from fastworkflow.command_context_model import CommandContextModel

    model = CommandContextModel.load(todo_workflow_path)
    assert model._manifest_occupiable() is None
    assert model.is_occupiable("TodoList") is True
    assert "TodoList" in model.occupiable_contexts()


def test_the_manifest_is_authoritative_when_it_declares_occupancy(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    from fastworkflow.command_context_model import CommandContextModel
    from fastworkflow.runtime_manifest import (
        ContextDeclaration,
        RuntimeMetadata,
        clear_runtime_metadata,
        register_runtime_metadata,
    )

    model = CommandContextModel.load(todo_workflow_path)
    register_runtime_metadata(
        todo_workflow_path,
        RuntimeMetadata(
            contexts={
                "TodoList": ContextDeclaration(occupiable=True),
                "TodoItem": ContextDeclaration(occupiable=False),
            },
            commands={},
            feature_modes={},
            workflow_fingerprint=None,
            has_workflow_manifest=True,
        ),
    )
    try:
        assert model.is_occupiable("TodoList") is True
        assert model.is_occupiable("TodoItem") is False
        assert model.occupiable_contexts() == ("TodoList",)
    finally:
        clear_runtime_metadata()


def test_effective_capabilities_project_inherited_commands_onto_the_concrete_context(
    initialized_fastworkflow, todo_workflow_path
):
    """FW-REQ-003 clause 7: mixin-owned commands are shown on an occupiable context."""
    from fastworkflow.command_context_model import CommandContextModel

    model = CommandContextModel.load(todo_workflow_path)
    capabilities = model.effective_capabilities("TodoList")
    assert capabilities
    assert all(
        cap.display_alias.startswith("TodoList/") for cap in capabilities
    ), "every alias names the context you are actually in"


# ----------------------------------------------------------------------
# The scope gate, and why it exists
# ----------------------------------------------------------------------


def test_bare_simple_name_resolution_is_off_until_a_deployment_enables_it(
    initialized_fastworkflow, todo_workflow_path
):
    """The classifier-parity stop condition, held as a test.

    A leading word that happens to match a command name in another context is
    ordinary natural language until a deployment says otherwise. Answering
    `not-callable-here` for it would move classifier predictions in a
    non-training slice (requirements §12.3), so that half rides the manifest
    feature gate and is off by default.
    """
    assert simple_name_resolution_enforced(todo_workflow_path) is False


def test_the_gate_opens_when_the_feature_is_enforced(
    initialized_fastworkflow, todo_workflow_path
):
    from fastworkflow.runtime_manifest import (
        RuntimeMetadata,
        clear_runtime_metadata,
        register_runtime_metadata,
    )

    register_runtime_metadata(
        todo_workflow_path,
        RuntimeMetadata(
            contexts={},
            commands={},
            feature_modes={FEATURE_ID: "enforce"},
            workflow_fingerprint=None,
            has_workflow_manifest=True,
        ),
    )
    try:
        assert simple_name_resolution_enforced(todo_workflow_path) is True
    finally:
        clear_runtime_metadata()


# ----------------------------------------------------------------------
# The index is built from raw declarations, not from the flattened views
# ----------------------------------------------------------------------


def test_the_index_keeps_collisions_that_the_flattened_view_discards():
    """Arch §10.1: `commands()` has already picked one winner per simple name.

    This is the structural reason the index cannot be built from it: by then
    "`find_thing` is ambiguous between two bases" is no longer expressible.
    """
    index = CommandCapabilityIndex.build(RAW_CONTEXTS)
    collisions = index.collisions("Explorer", "find_thing")
    assert len(collisions) == 2

    from fastworkflow.command_context_model import CommandContextModel

    # The flattened view keeps exactly one, by construction.
    flattened = {}
    for qualified in ("Lookup/find_thing", "Catalog/find_thing"):
        flattened[qualified.split("/")[-1]] = qualified
    assert len(flattened) == 1


def test_effective_capabilities_are_one_winner_per_name(index):
    aliases = [cap.display_alias for cap in index.effective_capabilities("Identity")]
    assert len(aliases) == len(set(aliases))
    assert "Identity/show_properties" in aliases
    assert "Identity/list_accounts" in aliases
