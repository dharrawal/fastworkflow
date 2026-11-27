"""Re-key a workflow's generated-data caches after the generator model left their keys (bd fix-ygxo).

One-off. The utterance cache (``___command_info/utterance_cache``) and the parameter-example cache
(``___command_info/param_example_cache``) file each command's generated data under a variant key hashed from its
fingerprint inputs. The generator model was one of those inputs; it no longer is, so every file written before
would miss. This script rewrites each such file under the key its inputs have without ``model``
(``utterance_cache.variant_key_of``, the caches' own derivation), so the next ``fastworkflow train`` reuses it.

No model call. Idempotent: a file whose inputs carry no ``model`` is left alone. Two files of one command that
differed only in the model become one file; their seed entries are merged, and where both hold the same seed the
more recently written file's entry is kept. A file whose stored key does not match its own inputs is reported and
left alone. The re-keyed file keeps the modification time of its source, which the caches' retention reads.

Usage:
    python scripts/rekey_generation_caches.py [--dry-run] <workflow folder> [<workflow folder> ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

from fastworkflow.train.utterance_cache import variant_key_of

CACHE_DIRNAMES = ("utterance_cache", "param_example_cache")
_VARIANT_FILE = re.compile(r"^(?P<stem>.+)\.(?P<key>[0-9a-f]{24})\.json$")


def _read(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write(path: str, payload: dict, mtime: float) -> None:
    """Same-directory temporary + replace, sorted keys and indent 2 as the caches write, then the source's mtime."""
    handle, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=f".{os.path.basename(path)}.rekey-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.utime(path, (mtime, mtime))


def rekey_cache_dir(cache_dir: str, *, dry_run: bool) -> dict:
    """Re-key every variant file of one cache directory. Returns the counts (files, entries, skipped)."""
    counts = {"files": 0, "entries": 0, "already_current": 0, "skipped": [], "merged": 0}
    try:
        names = sorted(os.listdir(cache_dir))
    except OSError:
        return counts
    # Oldest first, so a newer file's entry for the same seed overwrites an older one's in a merge.
    candidates = []
    for name in names:
        match = _VARIANT_FILE.match(name)
        if not match:
            continue
        path = os.path.join(cache_dir, name)
        payload = _read(path)
        if payload is None or not isinstance(payload.get("fingerprint_inputs"), dict):
            counts["skipped"].append(f"{name}: not a cache variant file")
            continue
        candidates.append((os.path.getmtime(path), name, path, match.group("stem"), payload))
    candidates.sort()

    groups: dict[str, list] = {}
    for mtime, name, path, stem, payload in candidates:
        inputs = payload["fingerprint_inputs"]
        if "model" not in inputs:
            counts["already_current"] += 1
            continue
        if payload.get("variant_key") != variant_key_of(inputs) or not name.endswith(f".{payload.get('variant_key')}.json"):
            counts["skipped"].append(f"{name}: its stored key does not match its inputs; left alone")
            continue
        new_inputs = {key: value for key, value in inputs.items() if key != "model"}
        new_key = variant_key_of(new_inputs)
        target = os.path.join(cache_dir, f"{stem}.{new_key}.json")
        groups.setdefault(target, []).append((mtime, path, payload, new_key, new_inputs))
        counts["files"] += 1
        counts["entries"] += len(payload.get("entries") or {})

    for target, sources in groups.items():
        existing = _read(target) if os.path.exists(target) else None
        entries = dict(existing.get("entries") or {}) if existing else {}
        counts["merged"] += len(sources) - 1 + (1 if existing else 0)
        for _mtime, _path, payload, _key, _inputs in sources:  # oldest first: a newer entry wins
            entries.update(payload.get("entries") or {})
        mtime, _path, payload, new_key, new_inputs = sources[-1]
        if dry_run:
            continue
        _write(target, {**payload, "variant_key": new_key, "fingerprint_inputs": new_inputs, "entries": entries},
               max(mtime, os.path.getmtime(target)) if existing else mtime)
        for _mtime, path, *_rest in sources:
            os.remove(path)
    return counts


def rekey_workflow(workflow_folder: str, *, dry_run: bool) -> dict:
    """Re-key both caches of one workflow folder. Returns ``{cache dirname: counts}``."""
    info = os.path.join(workflow_folder, "___command_info")
    return {name: rekey_cache_dir(os.path.join(info, name), dry_run=dry_run) for name in CACHE_DIRNAMES}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workflow_folders", nargs="+", help="Workflow folders whose caches to re-key")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be re-keyed; write nothing")
    args = parser.parse_args(argv)
    status = 0
    for folder in args.workflow_folders:
        if not os.path.isdir(folder):
            print(f"{folder}: not a folder", file=sys.stderr)
            status = 1
            continue
        for cache, counts in rekey_workflow(folder, dry_run=args.dry_run).items():
            verb = "would re-key" if args.dry_run else "re-keyed"
            print(f"{folder} {cache}: {verb} {counts['files']} file(s), {counts['entries']} entr(y/ies)"
                  f" ({counts['merged']} merged into another variant), {counts['already_current']} already current")
            for line in counts["skipped"]:
                print(f"  skipped {line}")
    return status


if __name__ == "__main__":
    sys.exit(main())
