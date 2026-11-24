"""A record is compared by identity, never by truth value or equality (fix-ftdl.7.4).

``Workflow`` tested ``if self._root_command_context:`` and ``record == root``: a record whose ``bool()`` is False
read as "no root / global", the root could be set twice, and a record equal to the root never climbed. Real
``Workflow`` on the library example with record classes that have ``__bool__`` False, ``__len__`` 0 and
``__eq__`` always True.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.context_navigation import exact_name_scopes, live_context_chain

LIBRARY = str(Path(__file__).resolve().parent.parent / "fastworkflow" / "examples" / "library")


class FalsyRoot:
    def __bool__(self):
        return False


class EmptyRoot:
    def __len__(self):
        return 0


class EqualToAnything:
    def __eq__(self, other):
        return True

    __hash__ = object.__hash__


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    fastworkflow.RoutingRegistry.clear_registry()


@pytest.fixture
def workflow():
    wf = fastworkflow.Workflow.create(workflow_folderpath=LIBRARY, workflow_id_str=f"identity-{uuid.uuid4().hex}")
    yield wf
    wf.close()


@pytest.mark.parametrize("root_class", [FalsyRoot, EmptyRoot])
def test_a_root_that_is_falsy_still_counts_as_set(workflow, root_class):
    workflow.root = root_class()
    with pytest.raises(ValueError, match="only be set once"):
        workflow.root = root_class()


@pytest.mark.parametrize("root_class", [FalsyRoot, EmptyRoot])
def test_a_falsy_record_is_named_for_its_class_not_global(workflow, root_class):
    workflow.root = root_class()
    workflow._command_context_for_response_generation = workflow._current_command_context   # as a dispatch sets it
    assert fastworkflow.Workflow.get_command_context_name(workflow.record) == root_class.__name__
    assert workflow._current_command_context_name == root_class.__name__
    assert workflow.is_record_root


@pytest.mark.parametrize("root_class", [FalsyRoot, EmptyRoot])
def test_a_falsy_root_is_the_parent_of_none_and_of_itself(workflow, root_class):
    root = root_class()
    workflow.root = root
    assert workflow.get_parent(None) is root
    assert workflow.get_parent(root) is root


def test_a_record_equal_to_the_root_but_not_the_root_is_not_the_root(workflow):
    workflow.root = EqualToAnything()
    other = EqualToAnything()
    workflow.record = other
    assert not workflow.is_record_root
    assert not workflow._is_current_command_context_root
    assert workflow.get_parent(other) is not workflow.root  # it climbs (no enclosing record found: None)


def test_the_context_walks_handle_a_falsy_root(workflow):
    workflow.root = FalsyRoot()
    assert live_context_chain(workflow)[0] == "FalsyRoot"
    assert exact_name_scopes(workflow)[0].record is workflow.root
