"""Learning from use (bd fix-o5wi.6): what a workflow records about the requests it resolves, which of them teach
the local intent head, and the lower-only calibration of its commit threshold.

ONE store per workflow, ``<workflow>/___convo_info/intent_learning.sqlite3`` (SQLite in WAL mode, a connection per
operation, so the server's concurrent channels in one process and a ``fastworkflow learn`` in another need nothing
more). Three tables: ``rows`` (one per resolved part), ``vectors`` (the encoder's vector of a learned text, so a refit
embeds each text once) and ``meta`` (the stored calibration and the refit marker).

Recording is on by default (``INTENT_LEARNING_RECORD``). It is written from where a part's command is finally settled
(`record`): the resolution path in `WorkflowExecutionContext` and the CME's wildcard command. It never fails a turn.

A row teaches only when a person or an agent confirmed it (`HOWS_THAT_CONFIRM`: a pick at a question, a review that
showed the command for the part and was accepted, a plan switched in review), never an exact or fuzzy NAME match, never
a bare value or a vague reply, and only by the rules of `teaching_rows`. Learned rows are added to the training rows
with the seed weight by `train.head_refit`; they never replace or remove another row.

The calibration (`calibrate`) can only LOWER the commit threshold: the requests whose truth is known were asked or
reviewed, so a confident answer's wrongness is not seen. This module imports nothing heavy (the training and
refit code import it, and the router reads it at load).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

from fastworkflow.utils.logging import logger

RECORD_SETTING = "INTENT_LEARNING_RECORD"
MIN_CONFIRMATIONS_SETTING = "INTENT_LEARNING_MIN_CONFIRMATIONS"
REFIT_SETTING = "INTENT_LEARNING_REFIT"
REFIT_EVERY_SETTING = "INTENT_LEARNING_REFIT_EVERY"
DEFAULT_MIN_CONFIRMATIONS = 1
DEFAULT_REFIT_EVERY = 25
REFIT_OFF = "off"
REFIT_BACKGROUND = "background"
DEFAULT_REFIT_MODE = REFIT_BACKGROUND   # the one place the default is named (tested live: tests/test_learning_live_refit.py)

STORE_FILENAME = "intent_learning.sqlite3"
CONVO_FOLDER = "___convo_info"

#: How a part's command was settled.
HOW_LOCAL = "local"                  # the local head was confident
HOW_REMOTE = "remote"                # the remote decision model was confident
HOW_NAME = "name"                    # an exact or fuzzy command-name match
HOW_USER_CHOICE = "user_choice"      # a pick at a which-command or which-plan question (or a pinned plan)
HOW_AGENT_CHOICE = "agent_choice"    # the agent picked from the candidate list
HOW_REVIEW_ACCEPT = "review_accept"  # the review showed this command for this part and it was accepted
HOW_REVIEW_SWITCH = "review_switch"  # the review's plan was switched: the row is the corrected command
HOWS = (HOW_LOCAL, HOW_REMOTE, HOW_NAME, HOW_USER_CHOICE, HOW_AGENT_CHOICE, HOW_REVIEW_ACCEPT, HOW_REVIEW_SWITCH)
#: The hows that are a human's or an agent's confirmation of the command for this text.
HOWS_THAT_CONFIRM = frozenset({HOW_USER_CHOICE, HOW_AGENT_CHOICE, HOW_REVIEW_ACCEPT, HOW_REVIEW_SWITCH})

#: The lower-only calibration (`calibrate`): candidate thresholds, and the evidence each needs.
CALIBRATION_CANDIDATES = (0.60, 0.55, 0.50)
CALIBRATION_MIN_CONFIRMED = 200
CALIBRATION_MIN_BAND = 100
#: The wrong share is of the BAND (the requests the lower threshold would newly answer), at most 2 wrong of 100.
CALIBRATION_MAX_WRONG_SHARE = 0.02
#: Distinct teaching texts a refit needs before its gate can say anything.
MIN_TEACHING_ROWS = 5

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS rows (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, text TEXT NOT NULL, norm TEXT NOT NULL,
        context TEXT NOT NULL, command TEXT NOT NULL, how TEXT NOT NULL, confirmed INTEGER NOT NULL,
        head_version TEXT, head_top TEXT, p_in REAL, p_out REAL, head_confident INTEGER,
        threshold REAL, threshold_source TEXT, remote_label TEXT, remote_score REAL, rejected_label TEXT)""",
    "CREATE INDEX IF NOT EXISTS rows_norm ON rows (norm, context)",
    "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)",
    "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)",
)
_COLUMNS = ("id", "ts", "text", "norm", "context", "command", "how", "confirmed", "head_version", "head_top", "p_in",
            "p_out", "head_confident", "threshold", "threshold_source", "remote_label", "remote_score",
            "rejected_label")

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message)


# ---------------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------------

def _env(name: str, default: str = "") -> str:
    import fastworkflow

    try:
        return str(fastworkflow.get_env_var(name, default=default) or "").strip()
    except Exception:  # noqa: BLE001 - an env that cannot be read is the default
        return default


def _positive_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value >= 1:
        return value
    _warn_once(f"bad:{name}={raw}", f"{name}={raw!r} is not a positive whole number: using {default}.")
    return default


def recording_enabled() -> bool:
    """``INTENT_LEARNING_RECORD``: on unless set to off, 0, false or no."""
    return _env(RECORD_SETTING, "on").lower() not in ("off", "0", "false", "no")


def min_confirmations() -> int:
    return _positive_int(MIN_CONFIRMATIONS_SETTING, DEFAULT_MIN_CONFIRMATIONS)


def refit_every() -> int:
    return _positive_int(REFIT_EVERY_SETTING, DEFAULT_REFIT_EVERY)


def refit_mode() -> str:
    """``INTENT_LEARNING_REFIT``: ``background`` (default) or ``off``; another value warns once and is off."""
    raw = _env(REFIT_SETTING, DEFAULT_REFIT_MODE).lower() or DEFAULT_REFIT_MODE
    if raw in (REFIT_OFF, REFIT_BACKGROUND):
        return raw
    _warn_once(f"bad:{REFIT_SETTING}={raw}", f"{REFIT_SETTING}={raw!r} is not off or background: using off.")
    return REFIT_OFF


# ---------------------------------------------------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------------------------------------------------

def store_path(workflow_folderpath: str) -> str:
    return os.path.join(workflow_folderpath, CONVO_FOLDER, STORE_FILENAME)


def _connect(workflow_folderpath: str, *, create: bool = True) -> Optional[sqlite3.Connection]:
    """A connection to the workflow's store, or None when there is none and *create* is not asked."""
    from fastworkflow.kvstore import _open_sqlite

    path = store_path(workflow_folderpath)
    if not create and not os.path.isfile(path):
        return None
    conn = _open_sqlite(path, timeout=30.0)
    for statement in _SCHEMA:
        conn.execute(statement)
    conn.commit()
    return conn


def normalise(text: str) -> str:
    """The text two requests are the same by: lower case, words separated by one space, no end punctuation."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).strip(" \t.!?,;:")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------------------------------------------------
# Evidence: what the detection chain knew when it decided
# ---------------------------------------------------------------------------------------------------------------------

def _head_facts(details: dict) -> Optional[dict]:
    labels = details.get("topk_labels") or ()
    top = labels[0] if labels else details.get("top_label")
    if top is None and "confidence" not in details:
        return None
    return {"version": details.get("head_version"), "top": top, "p_in": details.get("p_in"),
            "p_out": details.get("p_out"), "confident": bool(details.get("confident")),
            "threshold": details.get("ambiguous_threshold"), "threshold_source": details.get("threshold_source")}


def _remote_facts(details: dict) -> Optional[dict]:
    if "none_score" not in details:
        return None
    return {"label": details.get("top_label"), "score": details.get("confidence")}


def evidence_of(answers: Iterable[Any]) -> dict:
    """``{"head": {...}, "remote": {...}}`` from the engines' answers of a detection (``ScopeAnswer``: ``engine`` and
    ``details``): the local head's facts at decision time and the remote decision model's top label and score, each
    None when that link did not answer. A chain answer carries the head's facts as ``details["classifier"]``."""
    head = remote = None
    for answer in answers:
        details = getattr(answer, "details", None) or {}
        if getattr(answer, "engine", "") == "matcher":
            remote = _remote_facts(details) or remote
            if isinstance(details.get("classifier"), dict):
                head = _head_facts(details["classifier"]) or head
        else:
            head = _head_facts(details) or head
    return {"head": head, "remote": remote}


# ---------------------------------------------------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------------------------------------------------

def record(workflow_folderpath: str, *, text: str, context: str, command: str, how: str,
           evidence: Optional[dict] = None, rejected_label: Optional[str] = None) -> bool:
    """Write one resolved part. ``confirmed`` follows from *how* (`HOWS_THAT_CONFIRM`). Returns whether a row was
    written. Never raises: a store that cannot be written is logged once and the turn goes on."""
    try:
        if not recording_enabled() or not str(text or "").strip() or not command:
            return False
        if how not in HOWS:
            raise ValueError(f"unknown how {how!r}")
        evidence = evidence or {}
        head, remote = evidence.get("head") or {}, evidence.get("remote") or {}
        conn = _connect(workflow_folderpath)
        try:
            conn.execute(
                f"INSERT INTO rows ({', '.join(_COLUMNS[1:])}) VALUES ({', '.join('?' * (len(_COLUMNS) - 1))})",
                (_now(), str(text), normalise(text), str(context or ""), str(command), how,
                 int(how in HOWS_THAT_CONFIRM), head.get("version"), head.get("top"), head.get("p_in"),
                 head.get("p_out"), None if head.get("confident") is None else int(bool(head["confident"])),
                 head.get("threshold"), head.get("threshold_source"), remote.get("label"), remote.get("score"),
                 rejected_label))
            conn.commit()
        finally:
            conn.close()
    except Exception as error:  # noqa: BLE001 - recording must never fail a turn
        _warn_once("record", f"intent learning: a resolved request was not recorded ({type(error).__name__}: {error}); "
                             "later failures are not logged")
        return False
    _maybe_background_refit(workflow_folderpath)
    return True


def rows(workflow_folderpath: str, *, confirmed_only: bool = False) -> list[dict]:
    """Every recorded row, oldest first, as dicts keyed by the column names."""
    conn = _connect(workflow_folderpath, create=False)
    if conn is None:
        return []
    try:
        where = " WHERE confirmed = 1" if confirmed_only else ""
        found = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM rows{where} ORDER BY id").fetchall()
    finally:
        conn.close()
    return [dict(zip(_COLUMNS, row)) for row in found]


def remove(workflow_folderpath: str, ids: Sequence[int]) -> int:
    """Delete rows by id; returns how many were there."""
    conn = _connect(workflow_folderpath, create=False)
    if conn is None or not ids:
        return 0
    try:
        removed = conn.execute(
            f"DELETE FROM rows WHERE id IN ({', '.join('?' * len(ids))})", [int(i) for i in ids]).rowcount
        conn.commit()
    finally:
        conn.close()
    return removed


# ---------------------------------------------------------------------------------------------------------------------
# Which rows teach
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TeachingRow:
    text: str
    norm: str
    context: str
    command: str
    confirmations: int


def _is_not_a_request(norm: str) -> bool:
    """A bare value (no word in it, or one of the bare-value placeholders) or a built-in vague reply."""
    from fastworkflow.nlu_labels import PARAMETER_VALUE_PLACEHOLDERS

    if not re.search(r"[a-z]", norm):
        return True
    if norm in {normalise(each) for each in PARAMETER_VALUE_PLACEHOLDERS}:
        return True
    try:
        from fastworkflow.model_pipeline_training import BUILT_IN_VAGUE_REPLIES
    except Exception:  # noqa: BLE001 - without the trainer's list only the checks above apply
        return False
    return norm in {normalise(each) for each in BUILT_IN_VAGUE_REPLIES}


def teaching_rows(workflow_folderpath: str, *, min_count: Optional[int] = None,
                  labels: Optional[Iterable[str]] = None) -> list[TeachingRow]:
    """The rows that teach, one per (normalised text, context): confirmed rows only, never a ``name`` row, never a bare
    value or a vague reply. A (text, context) counts once its command has *min_count* confirmations
    (``INTENT_LEARNING_MIN_CONFIRMATIONS``); with conflicting confirmed commands it is excluded unless one leads the
    next by at least 2, and then it is that one. A text the contexts answer with different commands is excluded.
    *labels* (the head's labels) leaves out a command the head does not have."""
    need = min_confirmations() if min_count is None else int(min_count)
    known = None if labels is None else set(labels)
    counts: dict[tuple[str, str], dict[str, int]] = {}
    first_text: dict[tuple[str, str], str] = {}
    for row in rows(workflow_folderpath, confirmed_only=True):
        if row["how"] == HOW_NAME or _is_not_a_request(row["norm"]):
            continue
        key = (row["norm"], row["context"])
        counts.setdefault(key, {}).setdefault(row["command"], 0)
        counts[key][row["command"]] += 1
        first_text.setdefault(key, row["text"])
    chosen: dict[tuple[str, str], tuple[str, int]] = {}
    for key, by_command in counts.items():
        ranked = sorted(by_command.items(), key=lambda pair: (-pair[1], pair[0]))
        leader, count = ranked[0]
        if len(ranked) > 1 and count - ranked[1][1] < 2:
            continue
        if count >= need and (known is None or leader in known):
            chosen[key] = (leader, count)
    commands_of_text: dict[str, set[str]] = {}
    for (norm, _context), (command, _count) in chosen.items():
        commands_of_text.setdefault(norm, set()).add(command)
    return [TeachingRow(first_text[key], key[0], key[1], command, count)
            for key, (command, count) in sorted(chosen.items()) if len(commands_of_text[key[0]]) == 1]


def new_teaching_count(workflow_folderpath: str) -> int:
    """Confirmed, non-name rows recorded since the last refit (the background refit's trigger)."""
    marker = int((get_meta(workflow_folderpath, "refit_marker") or {}).get("row_id", 0))
    return sum(1 for row in rows(workflow_folderpath, confirmed_only=True)
               if row["id"] > marker and row["how"] != HOW_NAME)


# ---------------------------------------------------------------------------------------------------------------------
# Meta and cached vectors
# ---------------------------------------------------------------------------------------------------------------------

def get_meta(workflow_folderpath: str, key: str) -> Optional[dict]:
    conn = _connect(workflow_folderpath, create=False)
    if conn is None:
        return None
    try:
        found = conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    finally:
        conn.close()
    return None if found is None else json.loads(found[0])


def set_meta(workflow_folderpath: str, key: str, value: dict) -> None:
    conn = _connect(workflow_folderpath)
    try:
        conn.execute("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                     (key, json.dumps(value)))
        conn.commit()
    finally:
        conn.close()


def cached_vectors(workflow_folderpath: str, encoder_key: str, norms: Sequence[str]) -> dict:
    """``{norm: float32 vector}`` of the texts whose vector this encoder gave before."""
    import numpy as np

    conn = _connect(workflow_folderpath, create=False)
    if conn is None or not norms:
        return {}
    try:
        found = {}
        for norm in norms:
            row = conn.execute("SELECT vec FROM vectors WHERE key = ?", (f"{encoder_key}|{norm}",)).fetchone()
            if row is not None:
                found[norm] = np.frombuffer(row[0], dtype=np.float16).astype(np.float32)
    finally:
        conn.close()
    return found


def store_vectors(workflow_folderpath: str, encoder_key: str, vectors: dict) -> None:
    import numpy as np

    conn = _connect(workflow_folderpath)
    try:
        for norm, vector in vectors.items():
            data = np.asarray(vector, dtype=np.float16)
            conn.execute("INSERT OR REPLACE INTO vectors (key, dim, vec) VALUES (?, ?, ?)",
                         (f"{encoder_key}|{norm}", int(data.shape[0]), data.tobytes()))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------------------------------------------------
# Head lineage and the lower-only calibration
# ---------------------------------------------------------------------------------------------------------------------

def lineage_of(workflow_folderpath: str, version_id: str) -> list[str]:
    """The versions of *version_id*'s head lineage, oldest first: the full train and every refit of it up to this
    version (a refit's manifest lists them; a trained version is a lineage of one)."""
    from fastworkflow.train import artifact_versioning

    try:
        listed = artifact_versioning.read_manifest(workflow_folderpath, version_id).get("lineage")
    except ValueError:
        listed = None
    return [str(each) for each in listed] if listed else [version_id]


def current_version(workflow_folderpath: str) -> Optional[str]:
    from fastworkflow.train import artifact_versioning

    return artifact_versioning.resolve_current_version(workflow_folderpath)


def calibrate(workflow_folderpath: str, *, base_threshold: float, lineage: Sequence[str]) -> dict:
    """The calibration decision for a head lineage whose trained commit threshold is *base_threshold*.

    The evidence is censored: a request the head answered confidently is rarely reviewed, so truth is known mostly
    for requests that were asked or reviewed. A lower threshold would answer more of those, and their truth is
    known; a higher one would take answers away whose wrongness is not seen. So only LOWERING is considered.

    For each candidate ``t`` in `CALIBRATION_CANDIDATES` below the base: among the confirmed rows of the lineage
    (``head_version`` in *lineage*; a ``name`` row never counts), the band is the rows whose recorded ``p_in`` is in
    ``[t, base)`` with ``p_out <= p_in``. Those the head would have answered; one is wrong when the head's top
    in-context label differs from the confirmed command. The threshold is lowered to the smallest ``t`` with at least
    `CALIBRATION_MIN_CONFIRMED` confirmed rows in all, at least `CALIBRATION_MIN_BAND` in the band, and the wrong
    share OF THE BAND at most `CALIBRATION_MAX_WRONG_SHARE`. The band of a lower ``t`` contains the band of a higher
    one and each ``t`` is judged on its own whole band, so a clean band of 0.60 never carries 0.50. Returns ``{"threshold": value or None,
    "base": ..., "evidence": [one entry per candidate], "confirmed": ...}``."""
    members = set(lineage)
    confirmed = [row for row in rows(workflow_folderpath, confirmed_only=True)
                 if row["how"] != HOW_NAME and row["head_version"] in members]
    entries = []
    chosen = None
    for candidate in sorted(CALIBRATION_CANDIDATES):
        if candidate >= base_threshold:
            continue
        band = [row for row in confirmed
                if row["p_in"] is not None and candidate <= row["p_in"] < base_threshold
                and row["head_top"] and (row["p_out"] or 0.0) <= row["p_in"]]
        wrong = sum(1 for row in band if row["head_top"] != row["command"])
        share = wrong / len(band) if band else 0.0
        enough = (len(confirmed) >= CALIBRATION_MIN_CONFIRMED and len(band) >= CALIBRATION_MIN_BAND
                  and share <= CALIBRATION_MAX_WRONG_SHARE)
        entries.append({"candidate": candidate, "band_rows": len(band), "band_wrong": wrong,
                        "wrong_share_of_band": share, "enough": enough})
        if enough and chosen is None:
            chosen = candidate
    return {"threshold": chosen, "base": base_threshold, "confirmed": len(confirmed), "evidence": entries,
            "needs": _calibration_needs()}


def _calibration_needs() -> dict:
    """The evidence rule a calibration was judged by. It is stored with the decision, and a stored decision made
    under another rule is not read (`calibration`), so a threshold lowered on weaker evidence does not stay in force."""
    return {"confirmed": CALIBRATION_MIN_CONFIRMED, "band": CALIBRATION_MIN_BAND,
            "max_wrong_share": CALIBRATION_MAX_WRONG_SHARE, "share_of": "band"}


def store_calibration(workflow_folderpath: str, decision: dict, lineage: Sequence[str]) -> None:
    set_meta(workflow_folderpath, "calibration", {**decision, "lineage_root": lineage[0], "computed_at": _now()})


def calibration(workflow_folderpath: Optional[str], version_id: Optional[str] = None) -> Optional[dict]:
    """The stored calibration when it was made for the lineage of *version_id* (default: the current version)
    under the rule in force and lowers the threshold; else None."""
    if not workflow_folderpath:
        return None
    try:
        stored = get_meta(workflow_folderpath, "calibration")
        if not stored or stored.get("threshold") is None or stored.get("needs") != _calibration_needs():
            return None
        version = version_id or current_version(workflow_folderpath)
        if version is None or lineage_of(workflow_folderpath, version)[0] != stored.get("lineage_root"):
            return None
        return stored
    except Exception as error:  # noqa: BLE001 - an unreadable store leaves the trained threshold in force
        _warn_once("calibration", f"intent learning: the stored calibration was not read ({error}); the trained "
                                  "commit threshold stands")
        return None


def calibrated_threshold(workflow_folderpath: Optional[str], version_id: Optional[str] = None) -> Optional[float]:
    stored = calibration(workflow_folderpath, version_id)
    return None if stored is None else float(stored["threshold"])


def threshold_line(workflow_folderpath: str) -> str:
    """One line for the startup report: the commit threshold in force and why, or "" for a workflow with no embedding
    head or whose threshold is the plain default (the detection span records the one in force on every decision)."""
    from fastworkflow.train import artifact_versioning, intent_model_layout

    version = artifact_versioning.resolve_current_version(workflow_folderpath)
    if version is None:
        return ""
    model_dir = str(artifact_versioning.version_dir(workflow_folderpath, version) / intent_model_layout.WORKFLOW_MODEL_DIRNAME)
    try:
        with open(os.path.join(model_dir, intent_model_layout.SCOPES_FILENAME), encoding="utf-8") as f:
            scopes = json.load(f)
    except (OSError, ValueError):
        return ""
    if scopes.get("layout") != intent_model_layout.EMBEDDING or int(scopes.get("schema_version", 1)) < 2:
        return ""
    trained = float(scopes["threshold"])
    value, why = intent_model_layout.threshold_in_force(model_dir, trained)
    if why == "trained" and value == intent_model_layout.DEFAULT_COMMIT_THRESHOLD:
        return ""          # the plain default is not worth a line; the span and `learned stats` show it
    if why == "calibrated":
        stored = calibration(workflow_folderpath, version) or {}
        return (f"Local model commit threshold {value:.2f}: calibrated from {stored.get('confirmed', '?')} confirmed "
                f"requests (trained {trained:.2f}).")
    if why == "setting":
        return f"Local model commit threshold {value:.2f}: set by {intent_model_layout.COMMIT_THRESHOLD_SETTING}."
    return f"Local model commit threshold {value:.2f}: the trained value."


# ---------------------------------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------------------------------

def stats(workflow_folderpath: str) -> dict:
    found = rows(workflow_folderpath)
    by_how: dict[str, int] = {}
    for row in found:
        by_how[row["how"]] = by_how.get(row["how"], 0) + 1
    return {"rows": len(found), "confirmed": sum(row["confirmed"] for row in found), "by_how": by_how,
            "corrections": sum(1 for row in found if row["rejected_label"]),
            "teaching_rows": len(teaching_rows(workflow_folderpath)),
            "new_since_refit": new_teaching_count(workflow_folderpath) if found else 0}


# ---------------------------------------------------------------------------------------------------------------------
# The background refit (on by default; INTENT_LEARNING_REFIT=off turns it off)
# ---------------------------------------------------------------------------------------------------------------------

_background: dict[str, threading.Thread] = {}
_background_lock = threading.Lock()


def _maybe_background_refit(workflow_folderpath: str) -> None:
    """In ``background`` mode, once `refit_every` new teaching rows have arrived: one daemon thread per workflow per
    process runs the same refit as ``fastworkflow learn`` (a file lock keeps a second process out). Never on a turn's
    critical path: the thread is started and not waited on. Never raises."""
    try:
        if refit_mode() != REFIT_BACKGROUND:
            return
        key = os.path.realpath(workflow_folderpath)
        with _background_lock:
            running = _background.get(key)
            if running is not None and running.is_alive():
                return
            if new_teaching_count(workflow_folderpath) < refit_every():
                return
            thread = threading.Thread(target=_background_refit, args=(workflow_folderpath,), daemon=True,
                                      name=f"intent-learning-refit-{os.path.basename(key)}")
            _background[key] = thread
            thread.start()
    except Exception as error:  # noqa: BLE001
        _warn_once("background", f"intent learning: the background refit was not started ({error})")


def _background_refit(workflow_folderpath: str) -> None:
    from fastworkflow.train import head_refit

    try:
        with head_refit.refit_lock(workflow_folderpath) as acquired:
            if not acquired:
                return
            report = head_refit.refit(workflow_folderpath)
            logger.info("intent learning (background): " + report.summary_line())
    except Exception as error:  # noqa: BLE001 - a daemon thread must never raise
        logger.warning(f"intent learning: the background refit failed ({type(error).__name__}: {error})")


def wait_for_background(workflow_folderpath: str, timeout: float = 60.0) -> None:
    """Join the workflow's background refit thread, if any (a test's and a clean shutdown's tool)."""
    thread = _background.get(os.path.realpath(workflow_folderpath))
    if thread is not None:
        thread.join(timeout)

