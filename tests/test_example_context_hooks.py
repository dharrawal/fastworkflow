"""Serialization hooks of the bundled library and messaging_app_4 contexts.

Each context class is driven the way the framework drives it: ``project_command_contexts`` on a live workflow,
then ``restore_command_contexts`` into a different ``Workflow``. A last test runs the library example inside the
real server and checks that its session is not pinned and survives eviction and restore with its current context
intact. Nothing needs a trained head: the contexts are built directly, as the other hook tests do.
"""
from __future__ import annotations

import asyncio
import gc
import importlib
import shutil
import sys
import uuid
import weakref
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow import serialization_hooks
from fastworkflow.serialization_hooks import ProjectionOutcome
from fastworkflow.state_serialization import validate_state
from tests.test_workflow_serialization_hooks import (  # noqa: F401  (fixtures and helpers)
    EXAMPLES, initialized_fastworkflow, make_workflow)

LIBRARY = EXAMPLES / "library"
MESSAGING = EXAMPLES / "messaging_app_4"


def _project(workflow):
    projection = serialization_hooks.project_command_contexts(workflow)
    assert projection.outcome is ProjectionOutcome.SERIALIZABLE, projection.reason
    record = projection.as_record()
    validate_state({"context": record})
    return projection, record


def _restore(folder, record):
    restored = make_workflow(folder)
    assert serialization_hooks.restore_command_contexts(restored, record)
    return restored


# -- library ----------------------------------------------------------------------------------------------------

def _library():
    from fastworkflow.examples.library.application.library import Library
    library = Library()
    library.books["b1"].notes.append("added in the session")
    library.members["m2"].notes.append("moved to the Riverside branch")
    library.loans.append(type(library.loans[0])("b2", "m3", "2026-12-01", overdue=False))
    return library


def _library_workflow():
    workflow, library = make_workflow(LIBRARY), _library()
    workflow.root = library
    return workflow, library


def test_library_as_current_context_round_trips(initialized_fastworkflow):
    workflow, library = _library_workflow()
    workflow.record = library
    projection, record = _project(workflow)
    assert projection.anchor_is_root and projection.current_locator == serialization_hooks.ANCHOR_LOCATOR

    restored = _restore(LIBRARY, record)
    root = restored.root
    assert root is not library and restored._current_command_context is root
    assert list(root.books) == ["b1", "b2", "b3", "b4"] and list(root.members) == ["m1", "m2", "m3"]
    assert root.books["b1"].notes == ["one copy is a large-print edition", "added in the session"]
    assert root.members["m2"].notes == ["prefers email reminders", "moved to the Riverside branch"]
    assert [(l.book_uid, l.member_uid) for l in root.loans][-1] == ("b2", "m3")
    assert all(book.library is root for book in root.books.values())
    assert all(member.library is root for member in root.members.values())


@pytest.mark.parametrize("kind, uid, locator", [("books", "b3", "book:b3"), ("members", "m2", "member:m2")])
def test_a_book_or_member_current_inside_the_library_comes_back_as_the_restored_node(
        initialized_fastworkflow, kind, uid, locator):
    workflow, library = _library_workflow()
    workflow.record = getattr(library, kind)[uid]
    projection, record = _project(workflow)
    assert projection.anchor_is_root and projection.current_locator == locator

    restored = _restore(LIBRARY, record)
    node = restored._current_command_context
    assert node is getattr(restored.root, kind)[uid], "the current context must be a node of the restored library"
    assert node.library is restored.root and node.uid == uid


@pytest.mark.parametrize("kind, uid, expected", [("books", "b4", "Book"), ("members", "m3", "Member")])
def test_a_book_or_member_as_the_anchor_round_trips(initialized_fastworkflow, kind, uid, expected):
    """No root context: the book or member alone is the anchor, and carries its library with it."""
    workflow, library = make_workflow(LIBRARY), _library()
    workflow.record = getattr(library, kind)[uid]
    projection, record = _project(workflow)
    assert not projection.anchor_is_root and projection.context_name == expected

    restored = _restore(LIBRARY, record)
    node = restored._current_command_context
    assert type(node).__name__ == expected and node.uid == uid
    assert node.library.books[uid if kind == "books" else "b1"] is not None
    assert getattr(node.library, kind)[uid] is node


def test_record_is_declared_ephemeral_so_no_context_of_the_library_needs_a_hook():
    from fastworkflow.run_fastapi_mcp import checkpoint
    assert checkpoint.contexts_without_state_hook(str(LIBRARY)) == []


# -- messaging_app_4 ------------------------------------------------------------------------------------------

def _room():
    from fastworkflow.examples.messaging_app_4.application.chatroom import ChatRoom
    from fastworkflow.examples.messaging_app_4.application.user import PremiumUser, User
    room = ChatRoom()
    room.add_user(User(room, "Alice"))
    room.add_user(PremiumUser(room, "Bob Smith"))
    return room


@pytest.mark.parametrize("index, expected_type", [(0, "User"), (1, "PremiumUser")])
def test_a_user_as_the_anchor_round_trips_with_its_room(initialized_fastworkflow, index, expected_type):
    """No ChatRoom root: the user is the anchor. Its room is carried so get_parent still has somewhere to climb."""
    workflow = make_workflow(MESSAGING)
    room = _room()
    workflow.record = room.users[index]
    projection, record = _project(workflow)
    assert not projection.anchor_is_root and projection.context_name == expected_type

    restored = _restore(MESSAGING, record)
    user = restored._current_command_context
    assert type(user).__name__ == expected_type and user.name == room.users[index].name
    assert user is not room.users[index]
    assert user.chatroom.list_users() == ["Alice", "Bob Smith"] and user in user.chatroom.users


@pytest.mark.parametrize("index", [0, 1])
def test_a_user_inside_a_chatroom_root_is_still_restored_by_the_room(initialized_fastworkflow, index):
    workflow = make_workflow(MESSAGING)
    room = _room()
    workflow.root = room
    room.current_user = room.users[index]
    workflow.record = room.users[index]
    projection, record = _project(workflow)
    assert projection.anchor_is_root and projection.current_locator == room.users[index].name
    restored = _restore(MESSAGING, record)
    assert restored._current_command_context is restored.root.users[index]
    assert restored.root.current_user is restored.root.users[index]


def test_no_context_of_messaging_app_4_lacks_a_hook():
    from fastworkflow.run_fastapi_mcp import checkpoint
    assert checkpoint.contexts_without_state_hook(str(MESSAGING)) == []


# -- the training decision does not depend on these files ----------------------------------------------------

def test_editing_a_context_class_file_changes_no_command_fingerprint(tmp_path, initialized_fastworkflow):
    """`fastworkflow train` decides what to retrain from per-command source hashes; a context class file
    (_Book.py) is not a command file, so adding or editing a hook cannot make a head retrain."""
    from fastworkflow.train import selective_training as st
    folder = tmp_path / "library"
    shutil.copytree(LIBRARY, folder, ignore=shutil.ignore_patterns(
        "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"), symlinks=True)

    def digests():
        def where(path):
            path = Path(path)
            return path.relative_to(folder).as_posix() if folder in path.parents else path.name
        return {name: (where(fp.source_path), fp.source_sha256, fp.seed_utterances_sha256)
                for name, fp in st.compute_command_fingerprints(str(folder)).items()}

    before = digests()
    assert before and not any(Path(path).name.startswith("_") for path, *_ in before.values()), \
        "a context class file is among the training inputs"
    for context_file in ("Book/_Book.py", "Member/_Member.py", "Library/_Library.py", "Record/_Record.py"):
        path = folder / "_commands" / context_file
        path.write_text(path.read_text() + "\n# an edit that must not retrain anything\n")
    assert digests() == before


# -- the library inside the real server -------------------------------------------------------------------------

@pytest.fixture
def library_app(tmp_path, env_files):
    env_file, passwords_file = env_files
    folder = tmp_path / "library"
    shutil.copytree(LIBRARY, folder, ignore=shutil.ignore_patterns(
        "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"), symlinks=True)
    sys.argv = ["pytest", "--workflow_path", str(folder), "--env_file_path", env_file,
                "--passwords_file_path", passwords_file]
    import fastworkflow.run_fastapi_mcp.__main__ as main
    importlib.reload(main)
    from fastworkflow.run_fastapi_mcp import jwt_manager
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env
    previous_env = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts")
    previous_mode = jwt_manager.EXPECT_ENCRYPTED_JWT
    jwt_manager.EXPECT_ENCRYPTED_JWT = False
    try:
        yield main
    finally:
        jwt_manager.EXPECT_ENCRYPTED_JWT = previous_mode
        restore_fastapi_env(previous_env)


def test_a_library_session_is_not_pinned_and_survives_eviction_and_restore(library_app):
    from fastworkflow.examples.library.application.library import Library
    from fastworkflow.run_fastapi_mcp import checkpoint
    manager = library_app.session_manager

    def runtime(channel):
        return asyncio.run(manager.get_session(channel))

    with TestClient(library_app.app) as client:
        manager._max_live_sessions = 1  # after startup, which resolves the configured value
        first = f"lib_{uuid.uuid4().hex[:8]}"
        assert client.post("/initialize", json={"channel_id": first, "user_id": "alice"}).status_code == 200
        wf = runtime(first).execution_context.app_workflow
        library = Library()
        library.books["b2"].notes.append("kept across the eviction")
        wf.root = library
        wf.record = library.books["b2"]
        assert checkpoint.assess(runtime(first), manager.session_state_store).evictable, "the library pinned the session"
        assert client.get("/probes/readyz", params={"sessions": "true"}).json()["sessions"]["pinned"] == 0

        # A second channel pushes the first one out: it is saved, then released.
        second = f"lib_{uuid.uuid4().hex[:8]}"
        assert client.post("/initialize", json={"channel_id": second, "user_id": "alice"}).status_code == 200
        assert runtime(first) is None, "the idle library session was not evicted"
        # Drop every reference to the evicted session's objects, so the restore below has to rebuild them
        # (a Workflow still referenced is handed back as it was, which would prove nothing).
        old_workflow = weakref.ref(wf)
        del wf, library
        gc.collect()
        assert old_workflow() is None, "the evicted workflow is still alive, so a restore would not rebuild it"

        # Its next call restores it with the current context intact.
        assert client.post("/initialize", json={"channel_id": first, "user_id": "alice"}).status_code == 200
        restored = runtime(first).execution_context.app_workflow
        assert type(restored.root).__name__ == "Library"
        current = restored._current_command_context
        assert current is restored.root.books["b2"], "the current context is not a node of the restored library"
        assert current.notes == ["kept across the eviction"]
        assert current.library is restored.root
