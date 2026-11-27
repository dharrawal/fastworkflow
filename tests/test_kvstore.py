"""Integration tests for the SQLite KVStore replacing speedict."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from fastworkflow.session.kvstore import KVStore


def test_kvstore_round_trip_and_keyerror(tmp_path: Path):
    path = str(tmp_path / "store.sqlite3")
    with KVStore(path) as db:
        db["meta"] = {"last": 1}
        db["list"] = ["a", "b"]
        assert db["meta"] == {"last": 1}
        assert "list" in db
        assert db.get("missing", 42) == 42
        del db["list"]
        assert "list" not in db
        with pytest.raises(KeyError):
            del db["list"]
        with pytest.raises(KeyError):
            _ = db["list"]


def test_kvstore_coerces_int_keys(tmp_path: Path):
    path = str(tmp_path / "intkeys.sqlite3")
    with KVStore(path) as db:
        db[0] = {"utterance": "hi", "label": "x"}
        assert db.get(0)["utterance"] == "hi"
        assert 0 in db


def test_kvstore_keys_materialised_for_concurrent_mutation(tmp_path: Path):
    path = str(tmp_path / "keys.sqlite3")
    with KVStore(path) as db:
        for i in range(5):
            db[f"k{i}"] = i
        seen = []
        for key in db.keys():
            seen.append(key)
            if key == "k0":
                db["k_new"] = 99
                del db["k4"]
        assert "k0" in seen
        assert "k_new" in db
        assert "k4" not in db


def test_kvstore_values_are_json_not_pickle(tmp_path: Path):
    path = tmp_path / "nopickle.sqlite3"
    with KVStore(str(path)) as db:
        db["obj"] = {"class_name_marker": "DefinitelyNotPickle"}
    raw = path.read_bytes()
    assert b"\x80\x05" not in raw
    assert b"DefinitelyNotPickle" in raw  # plain JSON text


def _mp_writer(path: str, worker_id: int, n: int, result_path: str) -> None:
    try:
        with KVStore(path, timeout=30.0) as db:
            for i in range(n):
                key = f"w{worker_id}:{i}"
                db[key] = {"worker": worker_id, "i": i}
                assert db[key]["worker"] == worker_id
        Path(result_path).write_text("ok", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — report failure to parent via file
        Path(result_path).write_text(f"fail:{exc!r}", encoding="utf-8")
        raise SystemExit(1) from exc


def test_kvstore_four_process_concurrency(tmp_path: Path):
    path = str(tmp_path / "concurrent.sqlite3")
    n_workers = 4
    n_ops = 50
    result_paths = [str(tmp_path / f"result_{i}.txt") for i in range(n_workers)]
    procs = [
        multiprocessing.Process(
            target=_mp_writer, args=(path, i, n_ops, result_paths[i])
        )
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exit {p.exitcode}"

    for rp in result_paths:
        assert Path(rp).read_text(encoding="utf-8") == "ok"

    with KVStore(path) as db:
        keys = list(db.keys())
        assert len(keys) == n_workers * n_ops
        for key in keys:
            worker_s, i_s = key.split(":", 1)
            worker_id = int(worker_s[1:])
            i = int(i_s)
            assert db[key] == {"worker": worker_id, "i": i}
        for worker_id in range(n_workers):
            for i in range(n_ops):
                assert db[f"w{worker_id}:{i}"] == {"worker": worker_id, "i": i}
