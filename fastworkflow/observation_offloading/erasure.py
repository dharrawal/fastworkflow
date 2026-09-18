"""Scope-aware erasure and retention for the offload evidence sidecar.

**What this module is for (ido-gls, F5).** Every execute response is persisted
in ``<observability.sqlite3>.offload-handles.sqlite3`` -- the *sidecar* -- and
result handles add raw source pages, cursor tokens and walk verdicts to the
same file. The sidecar holds exact response bytes and a ``scope_json`` that
names the channel that produced them. Until this module existed neither the
archive nor the result-handle store offered any deletion path at all, so
``ObservabilityStore.forget_channel`` erased a channel from the main store and
left its complete responses fully recoverable beside it, and the configured
age/size retention never reached the file.

**Evidence-retention and sensitive-data policy for this store.**

1. *The sidecar is subject to the same retention as the store it sits beside.*
   ``ObservabilityStore.prune`` now prunes both files with one horizon and one
   size cap. The unit of sidecar retention is the TURN SCOPE, not the row: a
   scope is dropped whole, oldest first, so retention can never leave a
   declaration whose pages are gone. A scope's age is its EARLIEST evidence
   timestamp, i.e. when the turn began -- the same rule the main store applies
   to artifacts, which it ages by the turn key.
2. *Erasure is complete or it has not happened.* Forgetting a channel deletes
   its rows in every evidence table in the file, drops the process-local caches
   that could still serve them, and filters this run's ``FW_OFFLOAD_EVENTS``
   file, which is the other place a turn's text lands.
3. *Evidence preservation is explicit, and it is ON for experiment runs.* See
   below. Retention and erasure are both refused for a preserved scope.
4. *Sensitive data: redaction is a toggle, and it is ON by default.* The
   sidecar's response bytes now pass through the trace sink's own credential
   scrub and capture policy on their way to disk -- the same two protections,
   called in the same order, by way of
   ``observability.store.protect_offload_observation`` rather than by a second
   implementation that would drift from the first (``ido-zlm``). The toggle is
   ``FW_OFFLOAD_EVIDENCE_REDACTION``: ``on`` (the DEFAULT, and what an
   unconfigured deployment gets) or ``off``, with an unrecognised value warned
   about and treated as the default, exactly as the preservation mode below is.
   The mechanism lives in ``observation_offloading.archive``; the policy is
   written here so it is read together with the retention above.

   Both states are first-class and neither is degraded:

   * DEVELOPERS turn it ``off`` for debugging and optimisation. The archive's
     reason for existing is to reproduce exactly what the agent read, and a
     redacted archive no longer does. Full fidelity is a legitimate posture for
     a development environment and is not a misconfiguration.
   * DEVOPS leave it ``on`` in production, so a credential that appears in a
     command response is not written verbatim to disk where the same text in a
     span would have been scrubbed.
   * The DEFAULT is ``on`` because the two mistakes are not symmetric. A wrong
     default in this direction costs archive fidelity, which re-running
     recovers; the other direction writes a credential to disk, which nothing
     recovers.

   Every archived observation records which mode produced it, in
   ``observation_capture_policy``: the capture-policy contract version, the
   profile consulted, the toggle state, and whether the stored bytes actually
   DIFFER from what the command returned. The last of those is what lets a
   reader tell a redacted row from one that never contained a secret. That
   table is scope-keyed like every other, so it is discovered structurally and
   erased with its channel, and a row written before it existed has no entry
   and reads as UNKNOWN -- never as an assumption of full fidelity.

   Two things the toggle does NOT change. It does not change ERASABILITY: this
   module removes a redacted row and a verbatim one alike, so the toggle
   governs how long a secret is exposed, not whether it can be removed.
   And it does not protect what leaves this process by another route -- the
   agent's own prompt, and the observation-search call, still carry the text the
   command returned.

   Encryption at rest is explicitly OUT of scope and deferred. It is the control
   that protects a secret WITHOUT costing evidence fidelity, so it may later
   reduce how often redaction needs to be on; nothing here is designed around
   it.

**The preservation mode.** ``RuntimeHandleScope.experiment_id`` is the signal.
``scope_for_host`` sets it from the experiment claim bound to the session and
writes the literal ``"unbound"`` when there is no claim, so a chatbot channel
says so in the row itself. A scope is therefore an EXPERIMENT run when its
persisted ``scope_json`` carries an ``experiment_id`` that is neither empty nor
``"unbound"``, and a CHATBOT channel when it carries exactly one of those. The
signal is trustworthy because it is written into the evidence row at the moment
the evidence is written, by the same function that computes the scope the row
is keyed by: classifying a row needs no other file, no live session and no
inference from a path.

Everything here fails SAFE. A row whose ``scope_json`` is missing,
unparseable, or silent about ``experiment_id`` is PRESERVED, as is a scope with
no ``scope_json`` row at all. Deletion happens only for a scope this module can
positively show to be a chatbot channel.

Two further, deliberately blunt preservation controls:

* ``FW_OFFLOAD_EVIDENCE_PRESERVATION`` -- ``experiments`` (the default),
  ``all`` (nothing in the file is ever erased or pruned) or ``none`` (an
  operator honouring a deletion request that covers an experiment channel).
  An unrecognised value warns and falls back to the default, because a typo
  must not quietly change policy in either direction.
* A ``<sidecar>.preserve`` sentinel FILE beside the sidecar preserves that
  whole file unconditionally. It is for a store whose value does not depend on
  the framework having classified it correctly -- an evaluation corpus -- and
  it needs no schema change and no running process to apply.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

#: The sidecar's name is the observability database's name plus this suffix.
SIDECAR_SUFFIX = ".offload-handles.sqlite3"
#: A file with this name beside a sidecar preserves that sidecar entirely.
PRESERVE_SENTINEL_SUFFIX = ".preserve"

PRESERVATION_ENV = "FW_OFFLOAD_EVIDENCE_PRESERVATION"
#: Preserve experiment runs, erase chatbot channels. The owner's default.
PRESERVE_EXPERIMENTS = "experiments"
#: Preserve everything in the file.
PRESERVE_ALL = "all"
#: Preserve nothing the caller names: an operator deleting experiment evidence.
PRESERVE_NONE = "none"
_MODES = (PRESERVE_EXPERIMENTS, PRESERVE_ALL, PRESERVE_NONE)

#: What ``scope_for_host`` writes when the session carries no experiment claim.
UNBOUND_EXPERIMENT_ID = "unbound"

#: How many scopes one size-cap batch drops before the file is re-measured.
_SIZE_BATCH_SCOPES = 25
#: Upper bound on size-cap batches, so a pathological file cannot spin.
_SIZE_MAX_BATCHES = 40
#: How many scope ids go into one ``IN (...)`` clause.
_DELETE_CHUNK = 400

#: Columns that date a row, most specific first.
_TIMESTAMP_COLUMNS = (
    "persisted_at",
    "declared_at",
    "fetched_at",
    "recorded_at",
    "issued_at",
    "created_at",
)

_warned_modes: set[str] = set()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def preservation_mode(override: Optional[str] = None) -> str:
    """The active mode: the argument, else the environment, else the default."""
    raw = override if override is not None else os.environ.get(PRESERVATION_ENV, "")
    value = str(raw or "").strip().lower()
    if not value:
        return PRESERVE_EXPERIMENTS
    if value in _MODES:
        return value
    if value not in _warned_modes:
        _warned_modes.add(value)
        logger.warning(
            "ignoring %s=%s: expected one of %s; using %r",
            PRESERVATION_ENV, raw, ", ".join(_MODES), PRESERVE_EXPERIMENTS,
        )
    return PRESERVE_EXPERIMENTS


def sidecar_path(observability_db_path: str) -> str:
    """The evidence sidecar that belongs to this observability database."""
    return str(observability_db_path) + SIDECAR_SUFFIX


def preserve_sentinel_path(db_path: str) -> str:
    return str(db_path) + PRESERVE_SENTINEL_SUFFIX


def file_is_preserved(db_path: str, mode: Optional[str] = None) -> bool:
    """Whether NOTHING in this sidecar may be erased or pruned."""
    if preservation_mode(mode) == PRESERVE_ALL:
        return True
    return os.path.exists(preserve_sentinel_path(db_path))


def scope_experiment_id(scope_json: Any) -> Optional[str]:
    """The ``experiment_id`` a persisted scope names, or ``None`` if unknowable.

    ``None`` is the fail-safe answer: absent, malformed or silent ``scope_json``
    is a scope this module refuses to classify, and an unclassified scope is
    never deleted.
    """
    if isinstance(scope_json, Mapping):
        payload: Any = scope_json
    else:
        text = "" if scope_json is None else str(scope_json)
        if not text.strip():
            return None
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, Mapping) or "experiment_id" not in payload:
        return None
    value = payload.get("experiment_id")
    return None if value is None else str(value)


def is_experiment_scope(scope_json: Any) -> Optional[bool]:
    """``True`` experiment run, ``False`` chatbot channel, ``None`` cannot tell."""
    experiment_id = scope_experiment_id(scope_json)
    if experiment_id is None:
        return None
    normalised = experiment_id.strip()
    if not normalised or normalised == UNBOUND_EXPERIMENT_ID:
        return False
    return True


def scope_is_erasable(scope_json: Any, mode: Optional[str] = None) -> bool:
    """Whether this module may delete the rows of the scope described here."""
    active = preservation_mode(mode)
    if active == PRESERVE_ALL:
        return False
    verdict = is_experiment_scope(scope_json)
    if verdict is None:
        return False  # fail safe: an unclassifiable scope is preserved
    if active == PRESERVE_NONE:
        return True
    return not verdict


def scope_channel_id(scope_json: Any) -> Optional[str]:
    if isinstance(scope_json, Mapping):
        payload: Any = scope_json
    else:
        try:
            payload = json.loads(str(scope_json or ""))
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("channel_id")
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def evidence_tables(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Every scope-keyed evidence table in this file, discovered not listed.

    The rule is structural: a table in the sidecar that has a ``scope_id``
    column is a table of one turn's evidence. Discovery rather than a constant
    is the point -- ``result_handle_walks`` arrived in ``173e14b`` after the
    review that found this defect was written, and the next such table must not
    need this module edited to be erased with its channel.
    """
    tables: dict[str, dict[str, Any]] = {}
    # Positional row access throughout: a caller may hand this function a
    # connection with any ``row_factory``, and discovery must not depend on it.
    names = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    for name in names:
        columns = [
            str(row[1])
            for row in conn.execute(f'PRAGMA table_info("{name}")').fetchall()
        ]
        if "scope_id" not in columns:
            continue
        stamps = [column for column in _TIMESTAMP_COLUMNS if column in columns]
        stamps += [
            column
            for column in columns
            if column.endswith("_at") and column not in stamps
        ]
        tables[name] = {
            "columns": tuple(columns),
            "scope_json": "scope_json" in columns,
            "timestamp": stamps[0] if stamps else None,
        }
    return tables


def _scope_json_by_id(
    conn: sqlite3.Connection, tables: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    """``scope_id -> scope_json`` for every scope the file can describe."""
    described: dict[str, str] = {}
    for name, info in tables.items():
        if not info.get("scope_json"):
            continue
        for row in conn.execute(
            f'SELECT DISTINCT scope_id, scope_json FROM "{name}"'
        ).fetchall():
            described.setdefault(str(row[0]), str(row[1]))
    return described


def _earliest_by_scope(
    conn: sqlite3.Connection, tables: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    """``scope_id -> the earliest timestamp on any of its rows`` (turn start)."""
    earliest: dict[str, str] = {}
    for name, info in tables.items():
        column = info.get("timestamp")
        if not column:
            continue
        for row in conn.execute(
            f'SELECT scope_id, MIN("{column}") FROM "{name}" GROUP BY scope_id'
        ).fetchall():
            stamp = row[1]
            if stamp is None:
                continue
            scope_id = str(row[0])
            current = earliest.get(scope_id)
            if current is None or str(stamp) < current:
                earliest[scope_id] = str(stamp)
    return earliest


def _delete_scopes(
    conn: sqlite3.Connection,
    tables: Mapping[str, Mapping[str, Any]],
    scope_ids: Iterable[str],
) -> dict[str, int]:
    ordered = list(dict.fromkeys(str(value) for value in scope_ids))
    deleted: dict[str, int] = {}
    if not ordered:
        return deleted
    for name in tables:
        removed = 0
        for start in range(0, len(ordered), _DELETE_CHUNK):
            chunk = ordered[start:start + _DELETE_CHUNK]
            marks = ",".join("?" for _ in chunk)
            removed += conn.execute(
                f'DELETE FROM "{name}" WHERE scope_id IN ({marks})', chunk
            ).rowcount
        if removed:
            deleted[name] = removed
    return deleted


def _reclaim_caches(scope_ids: Iterable[str]) -> None:
    """Drop the process-local caches that could still serve the deleted rows.

    ``reclaim_scope`` is a RESIDENCY entry point and it deletes nothing on
    disk, so it is not an erasure mechanism and is not used as one here: the
    rows are already gone when this runs. It is called because it is the one
    place that knows every cache a scope id reaches -- the hot handles, the
    archive memo, the context clauses, the cursor tokens, the page index, the
    navigation registry and this process's event ring.
    """
    ordered = [str(value) for value in scope_ids]
    if not ordered:
        return
    try:
        from fastworkflow.observation_offloading import state as offload_state
    except Exception:  # pragma: no cover - import guard only
        logger.debug("offload cache reclamation unavailable", exc_info=True)
        return
    for scope_id in ordered:
        try:
            offload_state.reclaim_scope(scope_id)
        except Exception:  # noqa: BLE001 - a cache drop must not fail erasure
            logger.warning(
                "could not drop process caches for erased scope %s", scope_id,
                exc_info=True,
            )


def _forget_events_file(scope_ids: Iterable[str]) -> int:
    """Remove the erased scopes' lines from this run's event log, if any.

    ``FW_OFFLOAD_EVENTS`` is a separate plaintext sink carrying search
    questions, model reasoning and full answers (bead ``ido-gpb``). Erasure of
    a channel is not meaningful while that file keeps the same text in the
    clear, so the file is filtered here whenever the erasing process knows
    where it is. A line whose ``scope_id`` is absent or unrecognised is KEPT,
    by the same fail-safe rule as the tables, and an unreadable or unwritable
    log is reported rather than raised: an operator's deletion of the rows must
    not fail because a diagnostic sink is on a full disk.
    """
    targets = {str(value) for value in scope_ids}
    if not targets:
        return 0
    raw = os.environ.get("FW_OFFLOAD_EVENTS", "").strip()
    if not raw:
        return 0
    path = Path(raw)
    try:
        if not path.is_file():
            return 0
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("could not read %s for erasure", path, exc_info=True)
        return -1
    kept: list[str] = []
    removed = 0
    for line in lines:
        scope_id = None
        try:
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                scope_id = payload.get("scope_id")
        except (TypeError, ValueError):
            scope_id = None
        if scope_id is not None and str(scope_id) in targets:
            removed += 1
            continue
        kept.append(line)
    if not removed:
        return 0
    temporary = path.with_name(path.name + ".erase-tmp")
    try:
        temporary.write_text(
            "".join(line + "\n" for line in kept), encoding="utf-8"
        )
        os.replace(temporary, path)
    except OSError:
        logger.warning("could not rewrite %s for erasure", path, exc_info=True)
        try:
            temporary.unlink()
        except OSError:
            pass
        return -1
    return removed


def _compact(conn: sqlite3.Connection) -> None:
    """Return the deleted pages to the filesystem.

    A plain ``VACUUM``: the sidecar is created without ``auto_vacuum``, so
    ``incremental_vacuum`` is a no-op on it and the erased bytes would stay
    readable in free pages of the file.
    """
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")


# ---------------------------------------------------------------------------
# Erasure
# ---------------------------------------------------------------------------


def forget_channel(
    db_path: str, channel_id: str, *, mode: Optional[str] = None
) -> dict[str, int]:
    """Delete one chatbot channel's evidence from a sidecar, everywhere in it.

    Returns per-table delete counts plus ``scopes`` (how many turn scopes were
    erased), ``preserved_scopes`` (matched the channel and were kept) and
    ``events_removed`` (lines dropped from the event log, ``-1`` when the log
    could not be rewritten). The file is never created: a workflow that has
    never offloaded has nothing to erase.
    """
    result: dict[str, int] = {"scopes": 0, "preserved_scopes": 0}
    path = os.path.abspath(os.path.expanduser(str(db_path)))
    if not os.path.exists(path):
        return {}
    if file_is_preserved(path, mode):
        result["file_preserved"] = 1
        return result
    wanted = str(channel_id)
    with closing(_connect(path)) as conn:
        tables = evidence_tables(conn)
        if not tables:
            return result
        described = _scope_json_by_id(conn, tables)
        erasable: list[str] = []
        for scope_id, scope_json in described.items():
            if scope_channel_id(scope_json) != wanted:
                continue
            if scope_is_erasable(scope_json, mode):
                erasable.append(scope_id)
            else:
                result["preserved_scopes"] += 1
        if not erasable:
            return result
        conn.execute("BEGIN IMMEDIATE")
        result.update(_delete_scopes(conn, tables, erasable))
        conn.commit()
        _compact(conn)
    result["scopes"] = len(erasable)
    _reclaim_caches(erasable)
    result["events_removed"] = _forget_events_file(erasable)
    return result


def forget_all_channels(
    db_path: str, *, mode: Optional[str] = None
) -> dict[str, int]:
    """Erase every erasable scope in a sidecar (the UI's clear-conversations).

    Preservation applies unchanged: an experiment run's evidence survives a
    clear-conversations, which is the whole point of the mode.
    """
    result: dict[str, int] = {"scopes": 0, "preserved_scopes": 0}
    path = os.path.abspath(os.path.expanduser(str(db_path)))
    if not os.path.exists(path):
        return {}
    if file_is_preserved(path, mode):
        result["file_preserved"] = 1
        return result
    with closing(_connect(path)) as conn:
        tables = evidence_tables(conn)
        if not tables:
            return result
        described = _scope_json_by_id(conn, tables)
        erasable = [
            scope_id
            for scope_id, scope_json in described.items()
            if scope_is_erasable(scope_json, mode)
        ]
        result["preserved_scopes"] = len(described) - len(erasable)
        if not erasable:
            return result
        conn.execute("BEGIN IMMEDIATE")
        result.update(_delete_scopes(conn, tables, erasable))
        conn.commit()
        _compact(conn)
    result["scopes"] = len(erasable)
    _reclaim_caches(erasable)
    result["events_removed"] = _forget_events_file(erasable)
    return result


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def prune(
    db_path: str,
    *,
    retention_days: int,
    max_bytes: int,
    mode: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict[str, int]:
    """Age- and size-bounded retention for a sidecar, by whole turn scope.

    A scope is beyond the horizon when its EARLIEST evidence row is -- that is,
    when the turn began -- which matches how the main store ages a turn's
    artifacts by its turn key and guarantees a scope is dropped whole. When the
    file is still over ``max_bytes`` after that, the oldest remaining erasable
    scopes go in batches until it fits or nothing erasable is left.

    Preserved scopes are never counted towards the cap's solution and never
    deleted: a file that is over the cap entirely because of experiment
    evidence stays over the cap, and says so through ``over_cap``.
    """
    result: dict[str, int] = {"scopes": 0, "size_scopes": 0, "preserved_scopes": 0}
    path = os.path.abspath(os.path.expanduser(str(db_path)))
    if not os.path.exists(path):
        return {}
    if file_is_preserved(path, mode):
        result["file_preserved"] = 1
        return result
    moment = now or datetime.now(timezone.utc)
    horizon = (moment - timedelta(days=max(0, int(retention_days)))).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    erased: list[str] = []
    with closing(_connect(path)) as conn:
        tables = evidence_tables(conn)
        if not tables:
            return result
        described = _scope_json_by_id(conn, tables)
        erasable = {
            scope_id
            for scope_id, scope_json in described.items()
            if scope_is_erasable(scope_json, mode)
        }
        result["preserved_scopes"] = len(described) - len(erasable)
        earliest = _earliest_by_scope(conn, tables)
        aged = sorted(
            scope_id
            for scope_id in erasable
            if earliest.get(scope_id) is not None
            and str(earliest[scope_id]) < horizon
        )
        if aged:
            conn.execute("BEGIN IMMEDIATE")
            for table, count in _delete_scopes(conn, tables, aged).items():
                result[table] = result.get(table, 0) + count
            conn.commit()
            _compact(conn)
            erased.extend(aged)
            result["scopes"] = len(aged)

        gone = set(aged)
        remaining = sorted(
            (scope_id for scope_id in erasable if scope_id not in gone),
            key=lambda scope_id: (str(earliest.get(scope_id) or ""), scope_id),
        )
        cap = max(0, int(max_bytes))
        for _ in range(_SIZE_MAX_BATCHES):
            if os.path.getsize(path) <= cap or not remaining:
                break
            batch = remaining[:_SIZE_BATCH_SCOPES]
            del remaining[:_SIZE_BATCH_SCOPES]
            conn.execute("BEGIN IMMEDIATE")
            for table, count in _delete_scopes(conn, tables, batch).items():
                result[table] = result.get(table, 0) + count
            conn.commit()
            _compact(conn)
            erased.extend(batch)
            result["size_scopes"] += len(batch)
        if os.path.getsize(path) > cap:
            result["over_cap"] = 1
    if erased:
        _reclaim_caches(erased)
        result["events_removed"] = _forget_events_file(erased)
    return result


__all__ = [
    "PRESERVATION_ENV",
    "PRESERVE_ALL",
    "PRESERVE_EXPERIMENTS",
    "PRESERVE_NONE",
    "PRESERVE_SENTINEL_SUFFIX",
    "SIDECAR_SUFFIX",
    "UNBOUND_EXPERIMENT_ID",
    "evidence_tables",
    "file_is_preserved",
    "forget_all_channels",
    "forget_channel",
    "is_experiment_scope",
    "preservation_mode",
    "preserve_sentinel_path",
    "prune",
    "scope_channel_id",
    "scope_is_erasable",
    "sidecar_path",
]
