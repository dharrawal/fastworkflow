"""
Pure alignment algorithm for distillation divergence records (`fix-sb8.4`).

Implements §7 of `docs/distillation_observability_design.md`: the two canonical
keys `[DR18]` (including the `raw_command` fallback `[DR50]`), the O(nm) LCS
alignment with its deterministic reordering post-pass `[DR19]`, the seven-value
divergence taxonomy, materiality `[DR20]`, and the prose summary that §7.6
requires to be *rendered from* the structured records rather than computed
alongside them.

**This module is deliberately pure.** Standard library and typing only: it never
imports `fastworkflow.distillation`, `fastworkflow.observability_store`, a DB
handle or an LLM. Plain data in, plain data out. `[DR49]`'s read barrier — the
`sink.flush()` and the `spans_dropped` delta — belongs to the *caller*, which is
what decides `comparable`; this module only consumes the resulting flag.

The step contract
-----------------
The comparable-unit sequence for a pass is, in `start_ns` order, every
`fw.command.execute` span plus every `fw.ask_user` span under that pass's
`fw.distill.pass` span (§7.1). One `StepInput` per span:

| `StepInput` key  | Source                                                      |
|------------------|-------------------------------------------------------------|
| `span_id`        | `spans.span_id` (the global PRIMARY KEY — this is what gives |
|                  | a divergence record its provenance link for free)            |
| `command_name`   | `spans.command_name`; NULL/empty on every failed command     |
| `context`        | `spans.context`                                              |
| `parameters`     | span `attributes["parameters"]` (`{"agent_query": …}` for an |
|                  | `fw.ask_user` step)                                          |
| `raw_command`    | span `attributes["raw_command"]`, written at span open and   |
|                  | surviving the close; the `[DR50]` fallback key material      |
| `start_ns`       | `spans.start_ns`, ordering only (`sort_steps`)               |

`make_command_step` / `make_ask_user_step` / `make_plan_step` build the mapping;
a caller may equally hand in plain dicts.

`param_diff_json`, fixed here for `fix-sb8.8` (§18 leaves the shape to sb8.4)
---------------------------------------------------------------------------
One level deep, deeper values compared whole — a general structural diff is not
needed. The schema is stable: all three keys are always present, even empty.

```json
{"changed":    {"<key>": {"left": <canonical>, "right": <canonical>}},
 "left_only":  {"<key>": <canonical>},
 "right_only": {"<key>": <canonical>}}
```

Values are the *canonical* forms (§7.2): numbers as decimal strings, `datetime`
as RFC-3339 UTC, nested dicts recursive, list order preserved. So the UI diffs
what the alignment actually compared, not a second rendering of it.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional, Sequence, TypedDict

# ----------------------------------------------------------------------
# The step contract
# ----------------------------------------------------------------------


class StepInput(TypedDict, total=False):
    """One comparable unit: an `fw.command.execute` or `fw.ask_user` span.

    Only `span_id` is structurally required; every other field is allowed to be
    absent or NULL exactly as the span row allows it, because a failed command
    writes neither `command_name` nor `context` nor `parameters` (`[DR50]`).
    """
    span_id: str
    command_name: Optional[str]
    context: Optional[str]
    parameters: Optional[dict]
    raw_command: Optional[str]
    start_ns: Optional[int]


ASK_USER_COMMAND = "ask_user"
PLAN_COMMAND = "plan"

# The divergence taxonomy, exactly as stored in `distillation_divergences.kind`.
KIND_IDENTICAL = "identical"
KIND_SAME_COMMAND_DIFFERENT_PARAMS = "same-command-different-params"
KIND_PARAM_VALUE_ONLY = "param-value-only"
KIND_EXTRA_IN_STUDENT = "extra-in-student"
KIND_MISSING_IN_STUDENT = "missing-in-student"
KIND_REORDERED = "reordered"
KIND_DIFFERENT_ANSWER_SAME_ACTIONS = "different-answer-same-actions"

KINDS = frozenset({
    KIND_IDENTICAL,
    KIND_SAME_COMMAND_DIFFERENT_PARAMS,
    KIND_PARAM_VALUE_ONLY,
    KIND_EXTRA_IN_STUDENT,
    KIND_MISSING_IN_STUDENT,
    KIND_REORDERED,
    KIND_DIFFERENT_ANSWER_SAME_ACTIONS,
})

LEVEL_PLAN = "plan"
LEVEL_ACTION = "action"
LEVEL_RUN = "run"

ALGORITHM = "lcs-v1"

# Kinds that do not, on their own, mean the two passes did different work — the
# §7.3 step-6 precondition for the run-level record.
_EQUIVALENT_KINDS = frozenset({KIND_IDENTICAL, KIND_REORDERED})


def make_command_step(
    span_id: str,
    command_name: Optional[str] = None,
    context: Optional[str] = None,
    parameters: Optional[dict] = None,
    raw_command: Optional[str] = None,
    start_ns: Optional[int] = None,
) -> StepInput:
    """Build a step from an `fw.command.execute` span's fields."""
    return {
        "span_id": span_id,
        "command_name": command_name,
        "context": context,
        "parameters": parameters,
        "raw_command": raw_command,
        "start_ns": start_ns,
    }


def make_ask_user_step(
    span_id: str,
    agent_query: str,
    context: Optional[str] = None,
    start_ns: Optional[int] = None,
) -> StepInput:
    """Build a step from an `fw.ask_user` span.

    §7.2: `command_name = "ask_user"` and `params = {"agent_query": <query>}`.
    """
    return {
        "span_id": span_id,
        "command_name": ASK_USER_COMMAND,
        "context": context,
        "parameters": {"agent_query": agent_query},
        "raw_command": None,
        "start_ns": start_ns,
    }


def make_plan_step(
    span_id: str,
    plan_text: str,
    context: Optional[str] = None,
    start_ns: Optional[int] = None,
) -> StepInput:
    """Build a plan-level step from an `fw.planner.plan` / `.replan` span.

    §7.1 fixes the *payload* — the whitespace-normalized full plan string, not
    `PlanningStep.generated_plan`'s word split — and leaves the pseudo-command
    name to the implementation; this module uses `"plan"` so that plan-level
    alignment runs through the same keys and the same LCS as the action level.
    """
    return {
        "span_id": span_id,
        "command_name": PLAN_COMMAND,
        "context": context,
        "parameters": {"plan": normalize_whitespace(plan_text)},
        "raw_command": None,
        "start_ns": start_ns,
    }


def sort_steps(steps: Iterable[StepInput]) -> list[StepInput]:
    """Return the steps in `start_ns` order, stably (§7.1).

    Steps without a `start_ns` keep their given relative order at the end, since
    a missing timestamp is not evidence that the step came first.
    """
    ordered = list(steps)
    return sorted(
        ordered,
        key=lambda s: (s.get("start_ns") is None, s.get("start_ns") or 0),
    )


# ----------------------------------------------------------------------
# Canonicalization and the two keys `[DR18]` `[DR50]`
# ----------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Strip, and collapse every internal whitespace run to a single space."""
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_raw_command(raw: str) -> str:
    """`[DR50]`: strip, collapse internal whitespace, lowercase the leading verb."""
    normalized = normalize_whitespace(raw)
    if not normalized:
        return ""
    head, _, tail = normalized.partition(" ")
    return f"{head.lower()} {tail}" if tail else head.lower()


def _canonical_number(value) -> str:
    """A number as a decimal string, trailing zeros normalized away.

    §7.2 requires numbers to be decimal strings so that the live `_action_signature`
    confound goes away: today `1` and `"1"` are different signatures.
    """
    try:
        dec = value if isinstance(value, Decimal) else Decimal(str(value))
        return format(dec.normalize(), "f")
    except (InvalidOperation, ValueError):
        # NaN / Infinity and anything else Decimal will not normalize.
        return str(value)


def canonical_value(value):
    """One value in canonical form (§7.2).

    Keys sorted; numbers as decimal strings; `Decimal` and `datetime` normalized
    (RFC-3339 UTC); `None` as JSON null; nested dicts recursively; **list order
    preserved**. Booleans stay booleans — a bool is an `int` in Python, and
    collapsing `True` to `"1"` would make it equal to the string `"1"`.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, Decimal)):
        return _canonical_number(value)
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return aware.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): canonical_value(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    return str(value)


def canonical_parameters(parameters: Optional[dict]) -> dict:
    """`canonical(parameters)` from §7.2; a missing parameter map is `{}`."""
    if not isinstance(parameters, dict):
        return {}
    return {str(k): canonical_value(parameters[k]) for k in sorted(parameters, key=str)}


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, compact separators, no ASCII escaping."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _key_command_name(step: StepInput) -> Optional[str]:
    """The `command` value both keys are built from, applying `[DR50]`.

    Returns `spans.command_name` when it is set; otherwise `raw:<normalized>`
    from `attributes.raw_command`; otherwise `None`, meaning the step has no
    identity at all and must match nothing.
    """
    command_name = step.get("command_name")
    if command_name:
        return command_name
    raw = normalize_raw_command(step.get("raw_command") or "")
    return f"raw:{raw}" if raw else None


def command_key(step: StepInput) -> str:
    """The identity key the LCS matches on (§7.2).

    A step with neither `command_name` nor `raw_command` keys off its own
    `span_id`, so it matches nothing: an unmatched step is a visible gap, a
    falsely matched one is a fabricated `identical` (`[DR50]`).
    """
    name = _key_command_name(step)
    context = step.get("context")
    if name is None:
        return _sha16(f"span:{step.get('span_id')}")
    if name.startswith("raw:"):
        return _sha16(f"{name}\x1f{context or ''}")
    return _sha16(f"{name}\x1f{context}")


def step_key(step: StepInput) -> str:
    """The full step key: command identity plus context plus canonical params."""
    name = _key_command_name(step)
    if name is None:
        name = f"span:{step.get('span_id')}"
    return _sha16(canonical_json({
        "command": name,
        "context": step.get("context"),
        "params": canonical_parameters(step.get("parameters")),
    }))


# ----------------------------------------------------------------------
# The divergence record
# ----------------------------------------------------------------------


@dataclass
class DivergenceRecord:
    """One row of `distillation_divergences` (§9), before serialization."""
    divergence_id: str
    run_id: str
    level: str
    left_pass: str
    right_pass: str
    align_index: int
    kind: str
    material: Optional[int]
    # [DR35]: derived from `kind` in __post_init__, never from the caller. A
    # default alone is what let every stored record carry 1.
    replayable: int = 1
    command_key: Optional[str] = None
    command_name: Optional[str] = None
    context: Optional[str] = None
    left_step_key: Optional[str] = None
    right_step_key: Optional[str] = None
    left_span_id: Optional[str] = None
    right_span_id: Optional[str] = None
    param_diff: Optional[dict] = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """`[DR35]`: `replayable = 0` for `different-answer-same-actions`.

        Normative in §14, and derived here rather than passed at the one
        construction site that can produce the kind, so no future site can
        create the record without the flag. Revision 2 demotes the flag from
        the whole argument to a cheap pre-filter — it saves the cost of
        starting a replay whose per-step observation gate is certain to fail,
        because response-content evidence is exactly what the uncaptured
        application state cannot support. A field that is always 1 makes that
        pre-filter admit precisely the records it exists to exclude.
        """
        self.replayable = 0 if self.kind == KIND_DIFFERENT_ANSWER_SAME_ACTIONS else 1

    def to_row(self) -> dict:
        """The record as the column map `distillation_divergences` stores.

        `param_diff_json` is NULL when there is no parameter diff to show;
        `detail_json` is NOT NULL in the DDL and is therefore always written.
        """
        return {
            "divergence_id": self.divergence_id,
            "run_id": self.run_id,
            "level": self.level,
            "left_pass": self.left_pass,
            "right_pass": self.right_pass,
            "align_index": self.align_index,
            "kind": self.kind,
            "material": self.material,
            "replayable": self.replayable,
            "command_key": self.command_key,
            "command_name": self.command_name,
            "context": self.context,
            "left_step_key": self.left_step_key,
            "right_step_key": self.right_step_key,
            "left_span_id": self.left_span_id,
            "right_span_id": self.right_span_id,
            "param_diff_json": (
                canonical_json(self.param_diff) if self.param_diff is not None else None
            ),
            "detail_json": canonical_json(self.detail),
        }


@dataclass
class AlignmentResult:
    """Everything one `fw.distill.compare` span reports (§8), plus the records."""
    records: list[DivergenceRecord]
    left_steps: int
    right_steps: int
    matched_pairs: int
    divergence_counts: dict
    material_count: int
    algorithm: str = ALGORITHM

    def compare_attributes(self, level: str, left_pass: str, right_pass: str) -> dict:
        """The `fw.distill.compare` attribute map for this alignment."""
        return {
            "level": level,
            "left_pass": left_pass,
            "right_pass": right_pass,
            "left_steps": self.left_steps,
            "right_steps": self.right_steps,
            "matched_pairs": self.matched_pairs,
            "divergence_counts": dict(self.divergence_counts),
            "material_count": self.material_count,
            "algorithm": self.algorithm,
        }


def divergence_id(run_id: str, level: str, left_pass: str, right_pass: str, align_index: int) -> str:
    """A deterministic `divergence_id`, in §13.1's `<prefix>-<12 hex>` grammar.

    §9 declares the column a PRIMARY KEY but fixes no format; deriving it from
    the alignment's own coordinates keeps a re-run of the same alignment
    idempotent instead of minting a second set of rows.
    """
    digest = hashlib.sha256(
        f"{run_id}|{level}|{left_pass}|{right_pass}|{align_index}".encode("utf-8")
    ).hexdigest()[:12]
    return f"div-{digest}"


# ----------------------------------------------------------------------
# LCS alignment `[DR19]`
# ----------------------------------------------------------------------

# Edit-script op codes.
_OP_MATCH = "match"
_OP_LEFT = "left"
_OP_RIGHT = "right"


def _lcs_edit_script(left_keys: Sequence[str], right_keys: Sequence[str]) -> list[tuple]:
    """Plain O(nm) DP LCS over `command_key`, returned as an ordered edit script.

    Sequences are bounded by the agent's max iterations (§7.3, ≤ ~50 in
    practice), so the textbook table is correct and fast enough — no
    Hunt–Szymanski, no windowing.

    Each entry is `(_OP_MATCH, i, j)`, `(_OP_LEFT, i, None)` or
    `(_OP_RIGHT, None, j)`, in sequence order.
    """
    n, m = len(left_keys), len(right_keys)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = table[i], table[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if left_keys[i] == right_keys[j] else max(nxt[j], row[j + 1])

    script: list[tuple] = []
    i = j = 0
    while i < n and j < m:
        if left_keys[i] == right_keys[j]:
            script.append((_OP_MATCH, i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            # Ties emit the left-only step first, so the script order is fixed.
            script.append((_OP_LEFT, i, None))
            i += 1
        else:
            script.append((_OP_RIGHT, None, j))
            j += 1
    while i < n:
        script.append((_OP_LEFT, i, None))
        i += 1
    while j < m:
        script.append((_OP_RIGHT, None, j))
        j += 1
    return script


def _apply_reordering(script: list[tuple], left_step_keys: Sequence[str],
                      right_step_keys: Sequence[str]) -> list[tuple]:
    """§7.3 step 4: rewrite equal-`step_key` orphan pairs as one `reordered` op.

    Greedy by smallest `|i − j|`; ties break toward the earlier student (right)
    index, then the earlier teacher index, so the result is deterministic
    whatever order the candidates were generated in.
    """
    left_positions = {i: pos for pos, (op, i, _) in enumerate(script) if op == _OP_LEFT}
    right_positions = {j: pos for pos, (op, _, j) in enumerate(script) if op == _OP_RIGHT}

    candidates = [
        (abs(i - j), j, i)
        for i in left_positions
        for j in right_positions
        if left_step_keys[i] == right_step_keys[j]
    ]
    candidates.sort()

    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _distance, j, i in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        pairs.append((i, j))

    if not pairs:
        return script

    # The reordered record takes the earlier of the two script positions, so the
    # alignment keeps reading in sequence order; the other entry is dropped.
    dropped: set[int] = set()
    rewritten = dict(enumerate(script))
    for i, j in pairs:
        pos_left, pos_right = left_positions[i], right_positions[j]
        keep, drop = min(pos_left, pos_right), max(pos_left, pos_right)
        rewritten[keep] = (KIND_REORDERED, i, j)
        dropped.add(drop)
    return [rewritten[pos] for pos in range(len(script)) if pos not in dropped]


# ----------------------------------------------------------------------
# Classification, materiality and param diffs
# ----------------------------------------------------------------------


def _classify_pair(left: StepInput, right: StepInput) -> str:
    """§7.3 step 3, for a pair the LCS matched on `command_key`."""
    if step_key(left) == step_key(right):
        return KIND_IDENTICAL
    left_params = canonical_parameters(left.get("parameters"))
    right_params = canonical_parameters(right.get("parameters"))
    if set(left_params) == set(right_params):
        return KIND_PARAM_VALUE_ONLY
    return KIND_SAME_COMMAND_DIFFERENT_PARAMS


def build_param_diff(left_parameters: Optional[dict], right_parameters: Optional[dict]) -> dict:
    """Per-key before/after, one level deep, deeper values compared whole."""
    left_params = canonical_parameters(left_parameters)
    right_params = canonical_parameters(right_parameters)
    changed = {
        key: {"left": left_params[key], "right": right_params[key]}
        for key in sorted(set(left_params) & set(right_params))
        if left_params[key] != right_params[key]
    }
    return {
        "changed": changed,
        "left_only": {key: left_params[key] for key in sorted(set(left_params) - set(right_params))},
        "right_only": {key: right_params[key] for key in sorted(set(right_params) - set(left_params))},
    }


def compute_material(
    kind: str,
    comparable: bool,
    left_exit_state_fingerprint: Optional[str],
    right_exit_state_fingerprint: Optional[str],
) -> Optional[int]:
    """`[DR20]` materiality, over the **`state_fingerprint`** — never the prompt one.

    `material = NULL` when the run is non-comparable (materiality is unknowable
    if the passes did not start from the same place); `0` when the kind is
    `identical` or the two exit state fingerprints are equal — the same end
    state reached by a different path is not a mistake; `1` otherwise.

    A fingerprint that is missing on either side is not evidence of agreement,
    so it does not demote the record.
    """
    if not comparable:
        return None
    if kind == KIND_IDENTICAL:
        return 0
    if (
        left_exit_state_fingerprint is not None
        and left_exit_state_fingerprint == right_exit_state_fingerprint
    ):
        return 0
    return 1


# ----------------------------------------------------------------------
# The entry point
# ----------------------------------------------------------------------


def align_passes(
    *,
    run_id: str,
    left_pass: str,
    right_pass: str,
    left_steps: Sequence[StepInput],
    right_steps: Sequence[StepInput],
    comparable: bool = True,
    left_exit_state_fingerprint: Optional[str] = None,
    right_exit_state_fingerprint: Optional[str] = None,
    level: str = LEVEL_ACTION,
    left_answer: Optional[str] = None,
    right_answer: Optional[str] = None,
    left_run_span_id: Optional[str] = None,
    right_run_span_id: Optional[str] = None,
) -> AlignmentResult:
    """Align two passes' step sequences into `distillation_divergences` records.

    `left`/`right` rather than teacher/student because §9 keys the table by pass
    label for N-pass readiness `[DR12]`; the two exit `state_fingerprint`s and
    `comparable` are taken as parameters and never computed here (`[DR47]`,
    `[DR49]`: the caller owns the flush barrier that decides `comparable`).

    Returns every record, `identical` ones included — they are what makes the
    aligned diff renderable without recomputation and are the denominator of
    every rate in `fix-sb8.10` (§7.3).
    """
    left_list = list(left_steps)
    right_list = list(right_steps)

    left_command_keys = [command_key(s) for s in left_list]
    right_command_keys = [command_key(s) for s in right_list]
    left_step_keys = [step_key(s) for s in left_list]
    right_step_keys = [step_key(s) for s in right_list]

    script = _lcs_edit_script(left_command_keys, right_command_keys)
    script = _apply_reordering(script, left_step_keys, right_step_keys)

    records: list[DivergenceRecord] = []
    matched_pairs = 0

    def _record(align_index: int, kind: str, **kwargs) -> DivergenceRecord:
        return DivergenceRecord(
            divergence_id=divergence_id(run_id, level, left_pass, right_pass, align_index),
            run_id=run_id,
            level=level,
            left_pass=left_pass,
            right_pass=right_pass,
            align_index=align_index,
            kind=kind,
            material=compute_material(
                kind, comparable,
                left_exit_state_fingerprint, right_exit_state_fingerprint,
            ),
            **kwargs,
        )

    for align_index, (op, i, j) in enumerate(script):
        if op in (_OP_MATCH, KIND_REORDERED):
            left, right = left_list[i], right_list[j]
            kind = _classify_pair(left, right) if op == _OP_MATCH else KIND_REORDERED
            if op == _OP_MATCH:
                matched_pairs += 1
            param_diff = (
                None if kind in (KIND_IDENTICAL, KIND_REORDERED)
                else build_param_diff(left.get("parameters"), right.get("parameters"))
            )
            records.append(_record(
                align_index, kind,
                command_key=left_command_keys[i],
                command_name=_display_command_name(left) or _display_command_name(right),
                context=left.get("context") if left.get("context") is not None else right.get("context"),
                left_step_key=left_step_keys[i],
                right_step_key=right_step_keys[j],
                left_span_id=left.get("span_id"),
                right_span_id=right.get("span_id"),
                param_diff=param_diff,
                detail={
                    "left_index": i,
                    "right_index": j,
                    "left_parameters": canonical_parameters(left.get("parameters")),
                    "right_parameters": canonical_parameters(right.get("parameters")),
                    "left_raw_command": left.get("raw_command"),
                    "right_raw_command": right.get("raw_command"),
                    "algorithm": ALGORITHM,
                },
            ))
        elif op == _OP_LEFT:
            left = left_list[i]
            records.append(_record(
                align_index, KIND_MISSING_IN_STUDENT,
                command_key=left_command_keys[i],
                command_name=_display_command_name(left),
                context=left.get("context"),
                left_step_key=left_step_keys[i],
                left_span_id=left.get("span_id"),
                detail={
                    "left_index": i,
                    "left_parameters": canonical_parameters(left.get("parameters")),
                    "left_raw_command": left.get("raw_command"),
                    "algorithm": ALGORITHM,
                },
            ))
        else:
            right = right_list[j]
            records.append(_record(
                align_index, KIND_EXTRA_IN_STUDENT,
                command_key=right_command_keys[j],
                command_name=_display_command_name(right),
                context=right.get("context"),
                right_step_key=right_step_keys[j],
                right_span_id=right.get("span_id"),
                detail={
                    "right_index": j,
                    "right_parameters": canonical_parameters(right.get("parameters")),
                    "right_raw_command": right.get("raw_command"),
                    "algorithm": ALGORITHM,
                },
            ))

    # §7.3 step 6: every action-level record equivalent, but the answers differ.
    if _answers_differ(left_answer, right_answer) and all(
        r.kind in _EQUIVALENT_KINDS for r in records
    ):
        run_index = len(records)
        records.append(DivergenceRecord(
            divergence_id=divergence_id(
                run_id, LEVEL_RUN, left_pass, right_pass, run_index),
            run_id=run_id,
            level=LEVEL_RUN,
            left_pass=left_pass,
            right_pass=right_pass,
            align_index=run_index,
            kind=KIND_DIFFERENT_ANSWER_SAME_ACTIONS,
            material=compute_material(
                KIND_DIFFERENT_ANSWER_SAME_ACTIONS, comparable,
                left_exit_state_fingerprint, right_exit_state_fingerprint,
            ),
            left_span_id=left_run_span_id,
            right_span_id=right_run_span_id,
            detail={
                "left_answer": left_answer,
                "right_answer": right_answer,
                "algorithm": ALGORITHM,
            },
        ))

    divergence_counts: dict = {}
    for record in records:
        divergence_counts[record.kind] = divergence_counts.get(record.kind, 0) + 1

    return AlignmentResult(
        records=records,
        left_steps=len(left_list),
        right_steps=len(right_list),
        matched_pairs=matched_pairs,
        divergence_counts=divergence_counts,
        material_count=sum(1 for r in records if r.material == 1),
    )


def _display_command_name(step: StepInput) -> Optional[str]:
    """What `distillation_divergences.command_name` shows for a step.

    The real name when there is one, else the `[DR50]` `raw:` form so a failed
    command is still legible in the UI and in `idx_distill_div_kind` rollups.
    """
    return _key_command_name(step)


def _answers_differ(left_answer: Optional[str], right_answer: Optional[str]) -> bool:
    """Two final answers differ, compared whitespace-normalized."""
    if left_answer is None or right_answer is None:
        return False
    return normalize_whitespace(left_answer) != normalize_whitespace(right_answer)


# ----------------------------------------------------------------------
# The prose summary, rendered from the records (§7.6)
# ----------------------------------------------------------------------


def _format_params(params: dict) -> str:
    return canonical_json(params) if params else ""


def _format_step(command_name: Optional[str], params: dict) -> str:
    name = command_name or "<unknown command>"
    return f"{name}({_format_params(params)})" if params else name


def _materiality_note(record: DivergenceRecord) -> str:
    if record.material == 0 and record.kind != KIND_IDENTICAL:
        return " [non-material: both passes ended in the same state]"
    if record.material is None:
        return " [materiality unknown: run is non-comparable]"
    return ""


def render_record_summary(record: DivergenceRecord) -> str:
    """One human-readable line for one stored record.

    §7.6: the extractor prompt, the UI and the aggregate queries all read the
    same source of truth, so the prose is rendered *from* the record and never
    computed alongside it.
    """
    detail = record.detail or {}
    left_params = detail.get("left_parameters") or {}
    right_params = detail.get("right_parameters") or {}
    context = f" in context {record.context}" if record.context else ""
    note = _materiality_note(record)

    if record.kind == KIND_DIFFERENT_ANSWER_SAME_ACTIONS:
        return (
            f"[run] different-answer-same-actions: {record.left_pass} and {record.right_pass} "
            f"took equivalent actions but returned different answers — "
            f"{record.left_pass}: {detail.get('left_answer')!r}, "
            f"{record.right_pass}: {detail.get('right_answer')!r}{note}"
        )

    prefix = f"[{record.align_index}] {record.kind}: "

    if record.kind == KIND_IDENTICAL:
        return f"{prefix}{_format_step(record.command_name, left_params)}{context}{note}"

    if record.kind == KIND_MISSING_IN_STUDENT:
        return (
            f"{prefix}{record.left_pass} executed "
            f"{_format_step(record.command_name, left_params)}{context}; "
            f"{record.right_pass} did not{note}"
        )

    if record.kind == KIND_EXTRA_IN_STUDENT:
        return (
            f"{prefix}{record.right_pass} executed "
            f"{_format_step(record.command_name, right_params)}{context}; "
            f"{record.left_pass} did not{note}"
        )

    if record.kind == KIND_REORDERED:
        return (
            f"{prefix}{_format_step(record.command_name, left_params)}{context} — "
            f"{record.left_pass} step {detail.get('left_index')}, "
            f"{record.right_pass} step {detail.get('right_index')}{note}"
        )

    # param-value-only / same-command-different-params
    diff = record.param_diff or {}
    parts: list[str] = []
    for key, sides in (diff.get("changed") or {}).items():
        parts.append(f"{key}: {sides['left']!r} -> {sides['right']!r}")
    for key, value in (diff.get("left_only") or {}).items():
        parts.append(f"{key}: {value!r} -> <absent>")
    for key, value in (diff.get("right_only") or {}).items():
        parts.append(f"{key}: <absent> -> {value!r}")
    changes = "; ".join(parts) if parts else "parameters differ"
    return (
        f"{prefix}{record.command_name or '<unknown command>'}{context} — {changes}{note}"
    )


def render_divergence_summary(
    result: AlignmentResult | Sequence[DivergenceRecord],
    include_identical: bool = False,
) -> str:
    """The prose divergence summary the extractor prompt receives (§7.6).

    `identical` records are stored but omitted from the prose by default: they
    are the denominator of the rates, not a difference worth describing to the
    extractor.
    """
    records = result.records if isinstance(result, AlignmentResult) else list(result)
    lines = [
        render_record_summary(record)
        for record in records
        if include_identical or record.kind != KIND_IDENTICAL
    ]
    return "\n".join(lines)
