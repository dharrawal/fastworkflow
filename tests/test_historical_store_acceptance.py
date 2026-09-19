"""Revision 4 §5.4 acceptance bracket for historical result-handle stores.

Root's requirement was *"preserve all historical stores and test the chosen
fresh store/schema approach before any real workflow state is opened for
mutation."* Revision 4 §5.4 turns that into seven ordered steps. This module is **all seven**:
steps 1, 3, 4 and 6 plus the step-7 gate arrived with F5b (fix-iq53.2.10), and
steps 2 and 5, which assert the post-rename schema, arrived with F5a
(fix-iq53.2.9) once there was a renamed schema to assert.

  1. Checksum manifest, before anything — sha256 and byte size of every
     historical store, recorded OUTSIDE both trees.
  2. Fresh-store schema: create a new store in a temporary directory, exercise
     declare → walk → terminal → cursor → cold resume against it, and assert the
     ``result_handle_batches`` shape, the ``continuation_json`` round trip,
     ``batch_index`` allocation from 0, and the absence of ``count_only``.
  3. Read-only open over a COPY of a real historical store: reads what it
     should, creates no table, creates no ``-shm`` or ``-wal`` sidecar, leaves
     the copy's sha256 unchanged.
  4. Negative test: the WRITING constructor over the same copy DOES change it.
     Without this, step 3 can pass because nothing happened at all, and an
     ``open_readonly`` that silently fell back to read-write would still look
     green. This is the step that stops step 3 rotting.
  5. Old-shape read: a store holding ``result_handle_pages`` and no
     ``result_handle_batches`` is a historical record — returned, not continued —
     and is not written to on the way to being recognised as one.
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
from fastworkflow.result_handles import paging
from fastworkflow.result_handles.models import ResultHandleSpec, SourceDescriptor
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

    (fix-iq53.2.9, F5a) The writes are keyed on batch ORDINALS now, and the
    terminal row carries no ``count_only``. This fixture is deliberately spelled
    out against the store's own API rather than driven through ``declare`` and
    ``fetch_page``: it has to keep meaning the same thing when the walk above it
    changes, which is precisely what happened here.
    """
    path = tmp_path / "written" / "handles.sqlite3"
    store = ResultHandleStore(str(path))
    handle_scope = scope()
    store.put_declaration(handle_scope, "O1", declaration_payload())
    store.put_page(
        handle_scope,
        alias="O1",
        query_scope="",
        batch_index=0,
        limit_requested=10,
        source="producer",
        record={"records": ["uid000  Alan", "uid001  Bea", "uid002  Cy"]},
        backend_total=3,
    )
    store.put_walk_terminal(
        handle_scope,
        alias="O1",
        query_scope="",
        terminal_batch_index=1,
        complete=True,
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
# Step 2 — the fresh store's schema, exercised end to end
# ==========================================================================


def walked_store(path: Path) -> tuple[ResultHandleStore, list[object]]:
    """A fresh store with a whole traversal driven through the public API.

    declare → walk → terminal → cursor → cold resume, against a resolver that
    behaves like an offset-paging adapter: it offers a resume point whenever rows
    came back, and the framework reaches the end by asking once more and getting
    an empty batch. Nothing is hand-written into the tables, so what step 2
    asserts is the shape the RUNTIME produces and not the shape a fixture
    imagined.
    """
    rows = [{"uid": "u%03d" % index, "name": "Row %d" % index}
            for index in range(7)]
    calls: list[object] = []

    def resolver(request):
        calls.append(request)
        if isinstance(request, paging.TerminalRequest):
            # The adapter decides, in its own words, and reports the number its
            # own rule ran on. The framework records the decision (fix-iq53.2.8).
            return {"complete": request.distinct_uids == len(rows),
                    "incomplete_reason": "countonly_mismatch",
                    "count": len(rows)}
        start = int((request.continuation or {}).get("offset") or 0)
        served = rows[start:start + max(1, int(request.limit))]
        return {
            "rows": served,
            "total": len(rows),
            "continuation": ({"offset": start + max(1, int(request.limit))}
                             if served else None),
        }

    store = ResultHandleStore(str(path))
    paging.register_resolver("acceptance-step2", resolver)
    try:
        paging.declare(
            ResultHandleSpec(kind="row", summary="7 row(s).", items=[], total=7,
                             source_complete=False, page_size=3),
            source=SourceDescriptor(
                resolver="acceptance-step2", uid_field="uid",
                label_fields=("name",), filter_columns=("name",), batch_size=3,
                state={"view": "rows", "start_offset": 0, "materialized": 0},
            ),
            scope=scope(), selected_store=store, alias="O1",
        )
        cursor = None
        for _ in range(20):
            page = paging.fetch_page("O1", cursor, scope=scope(),
                                    selected_store=store, budget_bytes=200)
            cursor = page.next_cursor
            if cursor is None:
                break
        # The cold resume: a fresh process rebuilds the walk from these tables
        # alone, so a page served after it is a page served out of storage.
        paging.reset_result_handle_state()
        resumed = paging.fetch_page("O1", scope=scope(), selected_store=store,
                                   budget_bytes=100_000)
        assert len(resumed.rows) == 7, "the cold resume did not serve the walk"
        assert resumed.continuation == "complete"
    finally:
        paging.unregister_resolver("acceptance-step2")
        paging.reset_result_handle_state()
    return store, calls


def columns_of(path: Path, table: str) -> list[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
    finally:
        conn.close()


def test_step2_a_fresh_store_has_the_batch_shape_and_nothing_of_the_old_one(
    tmp_path: Path,
) -> None:
    """The post-rename schema, asserted on a store the runtime just built.

    Revision 4 §5.4 step 2, and the reason it comes BEFORE steps 3 to 6 touch
    anything realistic: the new shape is proven on a store created for the
    purpose, in a directory pytest owns, before any question of opening an
    existing file arises.
    """
    path = tmp_path / "fresh" / "handles.sqlite3"
    walked_store(path)

    tables = table_names(path)
    assert "result_handle_batches" in tables
    assert "result_handle_walk_terminals" in tables
    # The rename is a rename: a fresh store carries no trace of the old shape,
    # which is what makes "a store with result_handle_pages and no
    # result_handle_batches" a reliable signature of a historical record (step 5).
    assert "result_handle_pages" not in tables
    assert "result_handle_walks" not in tables

    batch_columns = columns_of(path, "result_handle_batches")
    assert "batch_index" in batch_columns
    assert "continuation_json" in batch_columns
    assert "start_offset" not in batch_columns

    terminal_columns = columns_of(path, "result_handle_walk_terminals")
    assert "terminal_batch_index" in terminal_columns
    # The column the framework's own coverage arithmetic used to need. It is gone
    # because the arithmetic is gone, not because the number stopped existing.
    assert "count_only" not in terminal_columns
    assert "terminal_offset" not in terminal_columns


def test_step2_batch_ordinals_are_allocated_from_zero_and_are_contiguous(
    tmp_path: Path,
) -> None:
    """``batch_index`` counts from 0, monotonically, with no gaps.

    7 rows at batch_size 3 is batches 0, 1, 2 with rows and batch 3 empty — the
    probe an offset walker cannot avoid, because it cannot know its last full
    batch was last. The ordinals are the FRAMEWORK's: they do not encode 0, 3, 6,
    9, which is where an offset-keyed store filed the same four batches.
    """
    path = tmp_path / "fresh" / "handles.sqlite3"
    store, _ = walked_store(path)

    batches = store.list_pages(scope(), alias="O1", query_scope="")
    assert [batch["batch_index"] for batch in batches] == [0, 1, 2, 3]
    assert [batch["row_count"] for batch in batches] == [3, 3, 1, 0]
    assert [batch["source"] for batch in batches] == ["resolver"] * 4

    terminal = store.get_walk_terminal(scope(), alias="O1", query_scope="")
    assert terminal is not None
    assert terminal["terminal_batch_index"] == 3
    assert terminal["complete"] is True
    assert terminal["distinct_uids"] == 7
    assert "count_only" not in terminal


def test_step2_the_continuation_round_trips_through_storage_verbatim(
    tmp_path: Path,
) -> None:
    """What went into ``continuation_json`` is what comes back out.

    This is the value the walk resumes from after an eviction or a restart, so it
    is the one column in the new table that the framework must never interpret
    and must never lose. The adapter's key here is ``offset``; the assertion is
    about the round trip and not about the key, which is the point.
    """
    path = tmp_path / "fresh" / "handles.sqlite3"
    store, _ = walked_store(path)

    batches = store.list_pages(scope(), alias="O1", query_scope="")
    assert [batch["continuation"] for batch in batches] == [
        {"offset": 3}, {"offset": 6}, {"offset": 9}, None,
    ]
    # NULL and only NULL on the batch that ended the walk: an empty batch keeps
    # no resume point however insistently one is offered (fix-iq53.2.4), and NULL
    # is how a rebuild reads "this walk cannot be continued past here".
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        stored = conn.execute(
            "SELECT batch_index, continuation_json FROM result_handle_batches "
            "WHERE query_scope = '' ORDER BY batch_index"
        ).fetchall()
    finally:
        conn.close()
    assert stored == [(0, '{"offset":3}'), (1, '{"offset":6}'),
                      (2, '{"offset":9}'), (3, None)]


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
        handle_scope, alias="O1", query_scope="", batch_index=0
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
            batch_index=0,
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
    # (fix-iq53.2.9, F5a) Still exactly five tables, two of them renamed. The
    # count is what the §5.2 measurement rests on, so it is asserted as a SET and
    # not as a length: a sixth table appearing here is a schema change nobody
    # declared, and that is worth failing on.
    assert set(after_tables) - set(before_tables) == {
        "result_handle_declarations",
        "result_handle_batches",
        "result_handle_walk_terminals",
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
    assert "result_handle_batches" in table_names(bare)
    assert "unrelated" in table_names(bare), "an existing table was lost"


# ==========================================================================
# Step 5 — a store of the old shape is a historical record, not a walk
# ==========================================================================


@pytest.fixture
def old_shape_store(tmp_path: Path) -> Path:
    """A store exactly as the pre-F5a constructor left it, built by hand.

    By hand on purpose: the code that used to create these tables is gone, so the
    only way to have one is to write the DDL out, and writing it out is what
    pins what "the old shape" was. This is the schema at ``6cf4ba8``, verbatim,
    with one producer page and one judged walk in it.
    """
    path = tmp_path / "old-shape" / "handles.sqlite3"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE result_handle_pages (
                scope_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                query_scope TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                limit_requested INTEGER NOT NULL,
                source TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                backend_total INTEGER,
                record_json BLOB NOT NULL,
                record_sha256 TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (scope_id, alias, query_scope, start_offset)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE result_handle_walks (
                scope_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                query_scope TEXT NOT NULL,
                terminal_offset INTEGER NOT NULL,
                complete INTEGER NOT NULL,
                count_only INTEGER,
                distinct_uids INTEGER NOT NULL,
                stop_reason TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (scope_id, alias, query_scope)
            )
            """
        )
        record = b'{"records":["uid000  Alan"]}'
        conn.execute(
            "INSERT INTO result_handle_pages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (scope().scope_id, "O1", "", 12, 10, "producer", 1, 1, record,
             hashlib.sha256(record).hexdigest(), "2026-09-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO result_handle_walks VALUES (?,?,?,?,?,?,?,?,?)",
            (scope().scope_id, "O1", "", 22, 1, 1, 1, "", "2026-09-01T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()
    return path


def test_step5_an_old_shape_store_is_returned_as_evidence_and_never_written(
    old_shape_store: Path,
) -> None:
    """Revision 4 §5.4 step 5: recognised as a historical record.

    "Returned, not continued", and both halves matter.

    RETURNED: the rows are still there and still readable. A store written before
    the rename keeps every byte it had, and a reader that wants them can have them
    — through a read-only connection, in the shape they were written in. That is
    what the new TABLE NAME buys over an ``ALTER``: nothing had to be migrated for
    the old rows to still parse, so nothing had to be rewritten to read them.

    NOT CONTINUED: the walk accessors do not silently answer "no batches" for a
    store that simply keys its batches differently — that would read an archive of
    540 rows as an empty traversal, which is the shape of a silently shorter
    listing. They fail loudly instead, on a file the framework has no business
    walking. Continuing one of these is the offline evaluator's job (IDO's I5),
    against the old table, by ``start_offset``.
    """
    before_sha = sha256_of(old_shape_store)
    before_size = old_shape_store.stat().st_size
    assert table_names(old_shape_store) == [
        "result_handle_pages", "result_handle_walks"
    ]

    reader = ResultHandleStore.open_readonly(str(old_shape_store))

    # RETURNED. The old rows, through the store's own read-only connection.
    conn = reader._connect()
    try:
        page = conn.execute(
            "SELECT start_offset, row_count, record_json FROM result_handle_pages"
        ).fetchone()
        walk = conn.execute(
            "SELECT terminal_offset, count_only, complete FROM result_handle_walks"
        ).fetchone()
    finally:
        conn.close()
    assert page["start_offset"] == 12 and page["row_count"] == 1
    assert json.loads(bytes(page["record_json"]).decode("utf-8")) == {
        "records": ["uid000  Alan"]
    }
    assert (walk["terminal_offset"], walk["count_only"], walk["complete"]) == (22, 1, 1)

    # NOT CONTINUED, and not silently either.
    for call in (
        lambda: reader.get_page(scope(), alias="O1", query_scope="", batch_index=0),
        lambda: reader.list_pages(scope(), alias="O1", query_scope=""),
        lambda: reader.list_page_query_scopes(scope(), alias="O1"),
        lambda: reader.get_walk_terminal(scope(), alias="O1", query_scope=""),
    ):
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            call()

    # NEVER WRITTEN. Not by the open, and not by the four refusals either: the
    # missing table is not created on the way to reporting that it is missing.
    assert table_names(old_shape_store) == [
        "result_handle_pages", "result_handle_walks"
    ], "reading an old-shape store created a table"
    assert sidecars_of(old_shape_store) == []
    assert old_shape_store.stat().st_size == before_size
    assert sha256_of(old_shape_store) == before_sha


def test_step5_the_writing_constructor_adds_the_new_tables_beside_the_old(
    old_shape_store: Path,
) -> None:
    """And if one is ever opened read-write, the old rows still survive it.

    The read-only open is the primary mechanism and this is the secondary one,
    measured rather than asserted from the DDL: a new table name means an
    accidental read-write open of an old-shape store is additive. It still
    REWRITES THE FILE — that is §5.2's finding and the reason the read-only open
    comes first — but it does not drop a row, drop a table, or migrate anything.
    """
    before_sha = sha256_of(old_shape_store)

    ResultHandleStore(str(old_shape_store))

    assert sha256_of(old_shape_store) != before_sha, (
        "an additive CREATE TABLE no longer rewrites the file; §5.2's measurement "
        "and the ordering it justifies both need rereading"
    )
    assert table_names(old_shape_store) == [
        "result_handle_batches",
        "result_handle_cursor_tags",
        "result_handle_cursors",
        "result_handle_declarations",
        "result_handle_pages",
        "result_handle_walk_terminals",
        "result_handle_walks",
    ]
    conn = sqlite3.connect(f"file:{old_shape_store}?mode=ro", uri=True)
    try:
        assert conn.execute(
            "SELECT count(*) FROM result_handle_pages").fetchone()[0] == 1
        assert conn.execute(
            "SELECT start_offset FROM result_handle_pages").fetchone()[0] == 12
        assert conn.execute(
            "SELECT count(*) FROM result_handle_walks").fetchone()[0] == 1
        assert conn.execute(
            "SELECT count(*) FROM result_handle_batches").fetchone()[0] == 0
    finally:
        conn.close()


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
