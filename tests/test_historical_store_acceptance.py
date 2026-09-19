"""Revision 4 §5.4 acceptance bracket for historical result-handle stores.

Root's requirement was *"preserve all historical stores and test the chosen
fresh store/schema approach before any real workflow state is opened for
mutation."* Revision 4 §5.4 turns that into seven ordered steps. This module is
steps **1, 3, 4 and 6**, plus the step-7 gate (fix-iq53.2.10, F5b). Steps 2 and
5 assert the post-rename schema and belong to F5a; they are deliberately not
here, and F5a must not start until this module passes.

  1. Checksum manifest, before anything — sha256 and byte size of every
     historical store, recorded OUTSIDE both trees.
  3. Read-only open over a COPY of a real historical store: reads what it
     should, creates no table, creates no ``-shm`` or ``-wal`` sidecar, leaves
     the copy's sha256 unchanged.
  4. Negative test: the WRITING constructor over the same copy DOES change it.
     Without this, step 3 can pass because nothing happened at all, and an
     ``open_readonly`` that silently fell back to read-write would still look
     green. This is the step that stops step 3 rotting.
  6. Re-verify the manifest: every checksum from step 1 unchanged.
  7. Only then may any real workflow state be opened for mutation.

Steps 1 and 6 bracket the whole of F5, not just this module's own tests, so
they run first and last here and are worth running around any other work that
touches the store. The order is the file order; do not reorder the functions.

NOTHING UNDER A HISTORICAL PATH IS OPENED FOR WRITING, and nothing anywhere is
deleted outside pytest's own ``tmp_path``. Every test that wants a realistic
store copies one out and works on the copy. The census is read through
``Path.stat`` and ``Path.read_bytes`` — a historical store is never handed to
``sqlite3.connect`` at all, because a plain read-only open of a WAL-mode
database creates sidecars next to it, which is the very thing step 6 looks for.

The corpus lives in a sibling checkout, so the census steps skip when it is
absent. ``FW_HISTORICAL_STORE_ROOTS`` (os.pathsep-separated) overrides where to
look; ``FW_HISTORICAL_STORE_MANIFEST_DIR`` overrides where the manifest lives.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import pytest

from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.result_handles.store import (
    ReadOnlyResultHandleStore,
    ResultHandleStore,
)

# --------------------------------------------------------------------------
# Where the corpus and the manifest live
# --------------------------------------------------------------------------

#: Trees to census. Revision 4 §5.1 counted 2 787 ``*.sqlite3*`` files in IDO,
#: of which 438 have "handles" in the name and 47 sit under
#: ``evaluation/artifacts/**``; those two classes are what a result-handle
#: reader would plausibly be pointed at, and they are what the manifest covers.
DEFAULT_ROOTS = (Path(__file__).resolve().parents[2] / "ido",)

#: Outside both trees, on purpose: a manifest that lives inside the tree it is
#: checksumming can be rewritten by the same accident it exists to detect.
DEFAULT_MANIFEST_DIR = Path.home() / ".fastworkflow-historical-store-manifest"

MANIFEST_NAME = "historical-store-manifest.json"

#: The store Revision 4 §5.2 measured its probes against — one table,
#: ``observation_offload_handles``, 11 rows, journal mode ``delete``.
REAL_STORE_RELPATH = Path(
    "evaluation/artifacts/ido-986-step10-recurrence-1/offload-handles.sqlite3"
)
REAL_STORE_ROWS = 11


def census_roots() -> list[Path]:
    override = os.environ.get("FW_HISTORICAL_STORE_ROOTS")
    roots = (
        [Path(part) for part in override.split(os.pathsep) if part]
        if override
        else list(DEFAULT_ROOTS)
    )
    return [root for root in roots if root.is_dir()]


def manifest_path() -> Path:
    override = os.environ.get("FW_HISTORICAL_STORE_MANIFEST_DIR")
    directory = Path(override) if override else DEFAULT_MANIFEST_DIR
    return directory / MANIFEST_NAME


def is_historical_store(root: Path, path: Path) -> bool:
    """A file the manifest covers: a handle store, or a frozen artifact store.

    ``*.sqlite3*`` rather than ``*.sqlite3`` so that a ``-wal`` or ``-shm``
    already sitting beside a frozen store is covered too. Eight of those exist
    today and they are evidence like any other byte in the tree.
    """
    relative = path.relative_to(root).parts
    if ".venv" in relative:
        return False
    return "handles" in path.name or relative[:2] == ("evaluation", "artifacts")


def census(roots: Iterable[Path]) -> dict[str, dict[str, object]]:
    """Every historical store under *roots*, keyed by ``root::relative-path``.

    Reads bytes only. Nothing here opens a database.
    """
    found: dict[str, dict[str, object]] = {}
    for root in roots:
        resolved = root.resolve()
        for path in resolved.rglob("*.sqlite3*"):
            if not path.is_file() or not is_historical_store(resolved, path):
                continue
            key = f"{resolved.name}::{path.relative_to(resolved).as_posix()}"
            found[key] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
    return found


def census_or_skip() -> dict[str, dict[str, object]]:
    roots = census_roots()
    if not roots:
        pytest.skip(
            "no historical-store corpus on this machine; set "
            "FW_HISTORICAL_STORE_ROOTS to run the census steps"
        )
    entries = census(roots)
    if not entries:
        pytest.skip(f"no historical stores found under {roots}")
    return entries


def real_store_or_skip() -> Path:
    for root in census_roots():
        candidate = root / REAL_STORE_RELPATH
        if candidate.is_file():
            return candidate
    pytest.skip(f"{REAL_STORE_RELPATH} is not on this machine")


# --------------------------------------------------------------------------
# Observing a store without opening it as a database
# --------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecars_of(path: Path) -> list[str]:
    """Names of the ``-wal`` / ``-shm`` / ``-journal`` files beside *path*."""
    return sorted(
        child.name
        for child in path.parent.iterdir()
        if child.name.startswith(path.name + "-")
    )


def table_names(path: Path) -> list[str]:
    """The tables in a store, read through a throwaway connection.

    Only ever called on a copy under ``tmp_path``.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return sorted(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        )
    finally:
        conn.close()


def convert_to_wal(path: Path) -> None:
    """Put a COPY into WAL mode and clear the sidecars that leaves behind."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()
    finally:
        conn.close()
    for name in sidecars_of(path):
        (path.parent / name).unlink()


@pytest.fixture
def historical_copy(tmp_path: Path) -> Path:
    """A copy of the real historical store, in a directory we own.

    Steps 3 and 4 both run against this same store: step 4 is only a check on
    step 3 if the thing it mutates is the thing step 3 left alone.
    """
    source = real_store_or_skip()
    target = tmp_path / "copy-of-historical-store.sqlite3"
    shutil.copy2(source, target)
    return target


def scope() -> RuntimeHandleScope:
    return RuntimeHandleScope(
        store_identity="store",
        channel_id="channel",
        experiment_id="exp",
        task_id="task-1",
        attempt=1,
        turn_key="turn-1",
    )


def declaration_payload() -> dict[str, object]:
    return {
        "kind": "listing",
        "summary": "three rows",
        "ordering": "uid",
        "total": 3,
        "materialized": 3,
        "source_complete": True,
        "page_size": 10,
        "classification": "listing",
        "presentation": True,
        "filters": {},
        "descriptor": {"query": "all"},
        "descriptor_sha256": "d" * 64,
    }


@pytest.fixture
def written_store(tmp_path: Path) -> Path:
    """A store built by the ordinary constructor, with rows worth reading back.

    The realistic corpus is optional; this is not. Steps 3 and 4 run against it
    too, so the bracket still means something on a machine that has no frozen
    evidence to copy.
    """
    path = tmp_path / "written" / "handles.sqlite3"
    store = ResultHandleStore(str(path))
    handle_scope = scope()
    store.put_declaration(handle_scope, "O1", declaration_payload())
    store.put_page(
        handle_scope,
        alias="O1",
        query_scope="",
        start_offset=0,
        limit_requested=10,
        source="producer",
        record={"records": ["uid000  Alan", "uid001  Bea", "uid002  Cy"]},
        backend_total=3,
    )
    store.put_walk_terminal(
        handle_scope,
        alias="O1",
        query_scope="",
        terminal_offset=3,
        complete=True,
        count_only=3,
        distinct_uids=3,
        stop_reason="exhausted",
    )
    store.issue_cursor(
        handle_scope,
        alias="O1",
        tag="",
        query_scope="",
        position=0,
        descriptor_sha256="d" * 64,
    )
    return path


# ==========================================================================
# Step 1 — the checksum manifest, before anything
# ==========================================================================


def test_step1_checksum_manifest_exists_outside_both_trees() -> None:
    """Record sha256 and byte size of every historical store.

    Written once and then never rewritten by this test. A manifest that
    re-baselines itself on every run cannot detect anything: the first run
    after a store was damaged would simply record the damage as the new truth.
    So an existing manifest is authoritative and is only checked for coverage.
    """
    entries = census_or_skip()
    target = manifest_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        target.write_text(
            json.dumps(
                {
                    "roots": [str(root.resolve()) for root in census_roots()],
                    "entries": entries,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    recorded = json.loads(target.read_text(encoding="utf-8"))["entries"]
    assert recorded, f"{target} records no stores"
    for key, value in recorded.items():
        assert len(str(value["sha256"])) == 64, f"{key} has no usable digest"
        assert int(value["size"]) >= 0

    uncovered = sorted(set(entries) - set(recorded))
    assert not uncovered, (
        f"{len(uncovered)} historical stores appeared since the manifest was "
        f"taken and are not bracketed by it, e.g. {uncovered[:5]}. Re-take the "
        f"manifest deliberately by moving {target} aside."
    )
    print(
        f"\nstep 1: {len(recorded)} historical stores bracketed by {target} "
        f"({sum(1 for k in recorded if 'handles' in k.rsplit('/', 1)[-1])} "
        f"named *handles*)"
    )


# ==========================================================================
# Step 3 — read-only open over a copy of a real historical store
# ==========================================================================


def test_step3_readonly_open_leaves_a_real_historical_copy_untouched(
    historical_copy: Path,
) -> None:
    """Probe C of Revision 4 §5.2, promoted to a regression test."""
    before_sha = sha256_of(historical_copy)
    before_size = historical_copy.stat().st_size
    before_tables = table_names(historical_copy)
    assert before_tables == ["observation_offload_handles"], (
        "the fixture store is not the shape §5.2 measured; re-anchor the test"
    )

    store = ResultHandleStore.open_readonly(str(historical_copy))
    conn = store._connect()
    try:
        rows = conn.execute(
            "SELECT count(*) AS n FROM observation_offload_handles"
        ).fetchone()
    finally:
        conn.close()

    assert rows["n"] == REAL_STORE_ROWS, "the copy did not read back its rows"
    assert table_names(historical_copy) == before_tables, "a table was created"
    assert sidecars_of(historical_copy) == [], "a -wal or -shm sidecar appeared"
    assert historical_copy.stat().st_size == before_size
    assert sha256_of(historical_copy) == before_sha, "the copy was rewritten"


def test_step3_readonly_open_reads_a_store_back_faithfully(
    written_store: Path,
) -> None:
    """"Reads what it should", through the store's own API rather than SQL.

    The frozen artifact predates these tables, so it can only prove that
    nothing was written. This proves the read half: every accessor returns
    through a read-only connection exactly what the writing store put there.
    """
    before_sha = sha256_of(written_store)
    handle_scope = scope()

    reader = ResultHandleStore.open_readonly(str(written_store))
    declaration = reader.get_declaration(handle_scope, "O1")
    page = reader.get_page(
        handle_scope, alias="O1", query_scope="", start_offset=0
    )
    terminal = reader.get_walk_terminal(
        handle_scope, alias="O1", query_scope=""
    )

    assert declaration is not None and declaration["total"] == 3
    assert reader.list_declarations(handle_scope) == [declaration]
    assert page is not None and page["row_count"] == 3
    assert page["record"]["records"][0] == "uid000  Alan"
    assert reader.list_pages(handle_scope, alias="O1", query_scope="") == [page]
    assert reader.list_page_query_scopes(handle_scope, alias="O1") == [""]
    assert terminal is not None and terminal["complete"] is True
    assert reader.list_cursors(handle_scope, alias="O1")[0]["position"] == 0
    assert sidecars_of(written_store) == []
    assert sha256_of(written_store) == before_sha, "reading rewrote the store"


def test_step3_readonly_open_refuses_to_write_through_an_inherited_writer(
    written_store: Path,
) -> None:
    """A writer inherited from the parent fails loudly rather than quietly.

    Not overridden on the read-only class, exactly as
    ``ReadOnlyObservabilityStore`` does not override its parent's writers:
    SQLite refuses the statement itself, and its message is unambiguous.
    """
    before_sha = sha256_of(written_store)
    reader = ResultHandleStore.open_readonly(str(written_store))

    with pytest.raises(sqlite3.OperationalError, match="readonly database"):
        reader.put_page(
            scope(),
            alias="O2",
            query_scope="",
            start_offset=0,
            limit_requested=10,
            source="producer",
            record={"records": ["uid999  Nope"]},
            backend_total=1,
        )

    assert sha256_of(written_store) == before_sha


def test_step3_readonly_open_creates_neither_file_nor_directory(
    tmp_path: Path,
) -> None:
    """The absent-store case, which the writing constructor would materialise.

    ``ResultHandleStore.__init__`` calls ``os.makedirs`` and then five
    ``CREATE TABLE`` statements, so a typo'd path produces a new empty store
    and a new directory. Pointed at a historical tree that is itself a
    mutation, so the read-only path has to raise instead.
    """
    missing = tmp_path / "not-a-directory" / "absent.sqlite3"

    with pytest.raises(sqlite3.OperationalError):
        ResultHandleStore.open_readonly(str(missing))

    assert not missing.exists(), "the read-only open created the store"
    assert not missing.parent.exists(), "the read-only open created a directory"


# ==========================================================================
# Step 4 — the negative test that stops step 3 rotting
# ==========================================================================


def test_step4_writing_constructor_does_change_the_same_copy(
    historical_copy: Path,
) -> None:
    """Probe A of Revision 4 §5.2: the hazard that exists today, before F5a.

    If this ever stops failing the file — if the writing constructor becomes
    harmless — then step 3 is passing for the wrong reason and proves nothing,
    because an ``open_readonly`` that silently fell back to read-write would
    also leave the copy untouched. Equally, if ``open_readonly`` were ever
    wired to the read-write connection, step 3 would start failing loudly here
    instead of quietly passing.
    """
    before_sha = sha256_of(historical_copy)
    before_size = historical_copy.stat().st_size
    before_tables = table_names(historical_copy)

    ResultHandleStore(str(historical_copy))

    after_tables = table_names(historical_copy)
    assert sha256_of(historical_copy) != before_sha, (
        "the writing constructor no longer changes a store it is pointed at; "
        "step 3 can now pass vacuously and must be re-anchored"
    )
    assert historical_copy.stat().st_size > before_size
    assert set(after_tables) - set(before_tables) == {
        "result_handle_declarations",
        "result_handle_pages",
        "result_handle_walks",
        "result_handle_cursor_tags",
        "result_handle_cursors",
    }


def test_step4_writing_constructor_changes_any_store_lacking_the_tables(
    tmp_path: Path,
) -> None:
    """The same negative check without the optional corpus.

    A store the writing constructor has already created may legitimately be
    byte-stable when it is reopened, so the sharp form of the negative test is
    a store that is MISSING the handle tables — which is exactly the shape of
    every frozen artifact. Runs everywhere, so the rot-detection does not
    vanish on a machine that skips the corpus.
    """
    bare = tmp_path / "bare.sqlite3"
    conn = sqlite3.connect(str(bare))
    try:
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()
    finally:
        conn.close()

    before_sha = sha256_of(bare)
    before_size = bare.stat().st_size

    ResultHandleStore(str(bare))

    assert sha256_of(bare) != before_sha, (
        "the writing constructor no longer changes a store that lacks the "
        "handle tables; the negative test has stopped testing anything and "
        "step 3 can now pass vacuously"
    )
    assert bare.stat().st_size > before_size
    assert "result_handle_pages" in table_names(bare)
    assert "unrelated" in table_names(bare), "an existing table was lost"


# ==========================================================================
# The `_connect` seam, and why there is no `immutable` parameter
# ==========================================================================


class SidecarSafeHandleStore(ReadOnlyResultHandleStore):
    """What a caller that knows a file is frozen writes, with no new parameter.

    The shape IDO's evaluation layer already uses against
    ``ReadOnlyObservabilityStore`` at ``evaluation/observability.py``: one
    overridden method, and the policy that method encodes lives with the code
    that knows which paths are frozen evidence. Nothing in ``store.py`` learns
    what ``immutable`` means. Kept at module scope so that what the seam costs
    a caller is legible as a whole.
    """

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro&immutable=1",
            uri=True,
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn


def count_tables(store: ReadOnlyResultHandleStore) -> int:
    """One real read through the store's own connection.

    Any read of a WAL-mode database has to build the WAL index, which is what
    materialises the ``-shm``; this is the smallest query that does it.
    """
    conn = store._connect()
    try:
        return int(
            conn.execute("SELECT count(*) AS n FROM sqlite_master").fetchone()["n"]
        )
    finally:
        conn.close()


@pytest.fixture
def wal_copy(request: pytest.FixtureRequest) -> Path:
    """A WAL-mode store to read: the real historical one where it exists.

    The five WAL-mode stores in the corpus today are ``observability.sqlite3``
    files, so no handle store is in WAL mode right now. The copy is converted
    here rather than hunted for, so the seam stays covered whether or not that
    stays true — and so these two tests still run on a machine with no corpus.
    """
    try:
        path = request.getfixturevalue("historical_copy")
    except pytest.skip.Exception:
        path = request.getfixturevalue("written_store")
    convert_to_wal(path)
    return path


def test_plain_readonly_open_of_a_wal_store_creates_sidecars(
    wal_copy: Path,
) -> None:
    """The incident, reproduced: ``mode=ro`` is not enough on a WAL store.

    17 sidecars once appeared under a frozen evidence root this way. The
    framework's own ``open_readonly`` is plain ``mode=ro`` and does exactly
    this too. That is the point of the next test rather than a defect in this
    one: suppressing the sidecar requires knowing the file is frozen, the
    framework cannot know that, and this is the measurement that says so.
    """
    assert sidecars_of(wal_copy) == []

    store = ResultHandleStore.open_readonly(str(wal_copy))
    assert count_tables(store) > 0
    created = sidecars_of(wal_copy)

    assert created, (
        "a read-only open of a WAL-mode store created no sidecar; if SQLite "
        "has changed this, the rationale for the _connect seam needs rereading"
    )


def test_the_connect_seam_suppresses_the_sidecars(wal_copy: Path) -> None:
    """The seam is load-bearing: one overridden method, no sidecar, same rows."""
    before_sha = sha256_of(wal_copy)
    assert sidecars_of(wal_copy) == []

    # The seam first, while the directory is still clean, because the
    # comparison open below is the one that dirties it.
    safe = SidecarSafeHandleStore(str(wal_copy))
    seen_through_seam = count_tables(safe)
    assert sidecars_of(wal_copy) == [], (
        "the seam did not suppress the sidecar; overriding _connect is the "
        "only lever a caller has and it has stopped working"
    )

    plain = ResultHandleStore.open_readonly(str(wal_copy))
    assert seen_through_seam == count_tables(plain), "the seam changed the read"
    assert sha256_of(wal_copy) == before_sha


def test_immutable_reads_a_truncated_database_when_the_wal_is_live(
    tmp_path: Path,
) -> None:
    """Why the framework must NOT apply ``immutable=1`` itself.

    ``immutable=1`` makes SQLite ignore the write-ahead log, so against a store
    whose WAL has not been checkpointed it returns an older, shorter database
    and says nothing. Whether that is acceptable depends on whether the path is
    frozen evidence or a directory still being written, which is knowledge the
    caller has and the framework does not. That asymmetry, measured here, is
    the whole argument for the seam over a parameter.
    """
    path = tmp_path / "live.sqlite3"
    writer = sqlite3.connect(str(path))
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE t (x INTEGER)")
        writer.executemany(
            "INSERT INTO t (x) VALUES (?)", [(n,) for n in range(10)]
        )
        writer.commit()
        # Checkpoint once, so the main file really does hold a readable table
        # with ten rows. Without this the immutable reader fails outright with
        # "no such table", which is a louder failure than the one that matters:
        # what makes this dangerous is that it answers, and answers wrongly.
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.executemany(
            "INSERT INTO t (x) VALUES (?)", [(n,) for n in range(2000)]
        )
        writer.commit()
        assert (path.parent / (path.name + "-wal")).stat().st_size > 0

        plain = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        immutable = sqlite3.connect(
            f"file:{path}?mode=ro&immutable=1", uri=True
        )
        try:
            seen_plain = plain.execute("SELECT count(*) FROM t").fetchone()[0]
            seen_immutable = immutable.execute(
                "SELECT count(*) FROM t"
            ).fetchone()[0]
        finally:
            plain.close()
            immutable.close()
    finally:
        writer.close()

    assert seen_plain == 2010, "the plain reader did not see the live WAL"
    assert seen_immutable == 10, (
        "immutable=1 no longer truncates an uncheckpointed WAL; if that is "
        "really true the policy argument in ReadOnlyResultHandleStore's "
        "docstring should be revisited rather than quietly left in place"
    )


def test_open_readonly_takes_no_policy_parameter() -> None:
    """The owner's ruling, expressed as a test so it cannot drift back.

    Revision 4 §9.3 inclined towards a framework ``immutable`` flag. That was
    rejected: the framework owns the mechanism, the caller owns the judgment,
    and the caller expresses it by overriding ``_connect``. A parameter
    reappearing on any of these three signatures is that decision being undone.
    """
    assert list(
        inspect.signature(ResultHandleStore.open_readonly).parameters
    ) == ["db_path"]
    assert list(
        inspect.signature(ReadOnlyResultHandleStore.__init__).parameters
    ) == ["self", "db_path"]
    assert list(
        inspect.signature(ReadOnlyResultHandleStore._connect).parameters
    ) == ["self"]
    assert list(
        inspect.signature(ResultHandleStore.__init__).parameters
    ) == ["self", "db_path"]

    assert ReadOnlyResultHandleStore._connect is not ResultHandleStore._connect
    assert issubclass(ReadOnlyResultHandleStore, ResultHandleStore)
    source = inspect.getsource(ReadOnlyResultHandleStore._connect)
    assert "mode=ro" in source and "uri=True" in source
    assert "immutable" not in source, (
        "the framework has taken a position on immutable=1; that judgment "
        "belongs to the caller that knows which paths are frozen"
    )


# ==========================================================================
# Step 6 — re-verify the manifest. Keep this last.
# ==========================================================================


def test_step6_every_checksum_from_step1_is_unchanged() -> None:
    """The bracket closes. Nothing this task did touched a historical store."""
    entries = census_or_skip()
    target = manifest_path()
    if not target.exists():
        pytest.skip(f"no step-1 manifest at {target}; run step 1 first")
    recorded = json.loads(target.read_text(encoding="utf-8"))["entries"]

    missing = sorted(set(recorded) - set(entries))
    changed = sorted(
        key
        for key, value in recorded.items()
        if key in entries
        and (
            entries[key]["sha256"] != value["sha256"]
            or entries[key]["size"] != value["size"]
        )
    )
    appeared = sorted(set(entries) - set(recorded))

    assert not missing, f"{len(missing)} historical stores are gone: {missing[:5]}"
    assert not changed, (
        f"{len(changed)} historical stores changed since the step-1 manifest: "
        f"{changed[:5]}"
    )
    assert not appeared, (
        f"{len(appeared)} files appeared beside the historical stores, which "
        f"is what a stray sidecar looks like: {appeared[:5]}"
    )
    print(f"\nstep 6: {len(recorded)} checksums re-verified unchanged")


if __name__ == "__main__":  # pragma: no cover - convenience for a manual run
    sys.exit(pytest.main([__file__, "-v", "-s"]))
