"""Exhaustive tests for the pure alignment algorithm (`fix-sb8.4`, §7).

`fastworkflow.distillation_alignment` is standard-library-only by design — no
DB, no LLM, no WEC — so every case below is a real end-to-end exercise of the
shipped algorithm with plain data, not a fixture standing in for one.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import fastworkflow.distillation_alignment as alignment_module
from fastworkflow.distillation_alignment import (
    ALGORITHM,
    KIND_DIFFERENT_ANSWER_SAME_ACTIONS,
    KIND_EXTRA_IN_STUDENT,
    KIND_IDENTICAL,
    KIND_MISSING_IN_STUDENT,
    KIND_PARAM_VALUE_ONLY,
    KIND_REORDERED,
    KIND_SAME_COMMAND_DIFFERENT_PARAMS,
    KINDS,
    AlignmentResult,
    DivergenceRecord,
    align_passes,
    build_param_diff,
    canonical_json,
    canonical_parameters,
    command_key,
    compute_material,
    divergence_id,
    make_ask_user_step,
    make_command_step,
    make_plan_step,
    normalize_raw_command,
    render_divergence_summary,
    render_record_summary,
    sort_steps,
    step_key,
)


# ---------------------------------------------------------------------------
# Helpers — plain data, matching the documented step contract
# ---------------------------------------------------------------------------

def step(span_id: str, command: str, context: str = "TodoList", **params):
    return make_command_step(
        span_id,
        command_name=command,
        context=context,
        parameters=params or None,
    )


def failed_step(span_id: str, raw_command: str, context: str | None = None):
    """A failed command: `spans.command_name` is NULL, `raw_command` survives."""
    return make_command_step(span_id, command_name=None, context=context,
                             raw_command=raw_command)


def align(left, right, **kwargs) -> AlignmentResult:
    params = {
        "run_id": "run-1",
        "left_pass": "teacher",
        "right_pass": "student",
        "left_steps": left,
        "right_steps": right,
    }
    params.update(kwargs)
    return align_passes(**params)


def kinds(result: AlignmentResult) -> list[str]:
    return [r.kind for r in result.records]


# ---------------------------------------------------------------------------
# Purity — the property that lets this module be written and tested alone
# ---------------------------------------------------------------------------

def test_module_imports_only_the_standard_library():
    source = Path(alignment_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"hashlib", "json", "re", "dataclasses", "datetime",
                        "decimal", "typing"}, imported
    assert not any(name.startswith("fastworkflow") for name in imported)


# ---------------------------------------------------------------------------
# §7.2 canonicalization and the two keys
# ---------------------------------------------------------------------------

def test_canonical_parameters_sorts_keys_and_preserves_list_order():
    canonical = canonical_parameters({"b": [3, 1, 2], "a": "x"})
    assert list(canonical) == ["a", "b"]
    assert canonical["b"] == ["3", "1", "2"]


def test_canonical_numbers_are_decimal_strings():
    # The live `_action_signature` confound: today `1` and `"1"` differ.
    assert canonical_parameters({"id": 1}) == canonical_parameters({"id": "1"})
    assert canonical_parameters({"n": 1.50}) == {"n": "1.5"}
    assert canonical_parameters({"n": Decimal("2.500")}) == {"n": "2.5"}


def test_canonical_booleans_stay_booleans():
    # bool is an int in Python; collapsing True to "1" would equal the string.
    assert canonical_parameters({"f": True}) != canonical_parameters({"f": 1})
    assert canonical_parameters({"f": True}) == {"f": True}


def test_canonical_datetime_is_rfc3339_utc():
    naive = datetime(2026, 1, 2, 3, 4, 5)
    aware = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert canonical_parameters({"t": naive}) == {"t": "2026-01-02T03:04:05Z"}
    assert canonical_parameters({"t": naive}) == canonical_parameters({"t": aware})


def test_canonical_none_is_json_null():
    assert json.loads(canonical_json(canonical_parameters({"x": None}))) == {"x": None}


def test_command_key_ignores_parameters_but_step_key_does_not():
    a = step("s1", "get_task", task_id="1")
    b = step("s2", "get_task", task_id="2")
    assert command_key(a) == command_key(b)
    assert step_key(a) != step_key(b)


def test_command_key_separates_contexts():
    assert command_key(step("s1", "list", context="TodoList")) != \
        command_key(step("s2", "list", context="TodoItem"))


def test_ask_user_step_keys_off_the_agent_query():
    a = make_ask_user_step("s1", "which task?")
    b = make_ask_user_step("s2", "which project?")
    assert command_key(a) == command_key(b)
    assert step_key(a) != step_key(b)


def test_plan_step_payload_is_the_whitespace_normalized_plan_string():
    a = make_plan_step("s1", "  do   this\nthen that ")
    b = make_plan_step("s2", "do this then that")
    assert a["parameters"] == {"plan": "do this then that"}
    assert step_key(a) == step_key(b)


def test_sort_steps_orders_by_start_ns():
    late = make_command_step("s2", command_name="b", start_ns=20)
    early = make_command_step("s1", command_name="a", start_ns=10)
    assert [s["span_id"] for s in sort_steps([late, early])] == ["s1", "s2"]


# ---------------------------------------------------------------------------
# [DR50] the raw_command fallback
# ---------------------------------------------------------------------------

def test_normalize_raw_command_lowercases_only_the_leading_verb():
    assert normalize_raw_command("  Add   Task  Buy Milk ") == "add Task Buy Milk"
    assert normalize_raw_command("LIST") == "list"


def test_two_different_failed_commands_do_not_share_a_command_key():
    a = failed_step("s1", "add task buy milk")
    b = failed_step("s2", "delete project alpha")
    assert command_key(a) != command_key(b)
    assert step_key(a) != step_key(b)


def test_unrelated_teacher_and_student_failures_do_not_align_as_identical():
    result = align([failed_step("t1", "add task buy milk")],
                   [failed_step("s1", "delete project alpha")])
    assert KIND_IDENTICAL not in kinds(result)
    assert sorted(kinds(result)) == [KIND_EXTRA_IN_STUDENT, KIND_MISSING_IN_STUDENT]


def test_the_same_failed_command_on_both_sides_still_aligns():
    result = align([failed_step("t1", "Add task  buy milk")],
                   [failed_step("s1", "add   task buy milk")])
    assert kinds(result) == [KIND_IDENTICAL]
    assert result.records[0].command_name == "raw:add task buy milk"


def test_a_span_with_no_name_and_no_raw_command_matches_nothing():
    left = [make_command_step("t1")]
    right = [make_command_step("s1")]
    assert command_key(left[0]) != command_key(right[0])
    assert sorted(kinds(align(left, right))) == [KIND_EXTRA_IN_STUDENT,
                                                 KIND_MISSING_IN_STUDENT]


# ---------------------------------------------------------------------------
# §7.3 the LCS alignment and the taxonomy
# ---------------------------------------------------------------------------

def test_identical_sequences_produce_identical_records_only():
    left = [step("t1", "list"), step("t2", "get_task", task_id="1")]
    right = [step("s1", "list"), step("s2", "get_task", task_id="1")]
    result = align(left, right)
    assert kinds(result) == [KIND_IDENTICAL, KIND_IDENTICAL]
    assert result.matched_pairs == 2
    assert result.divergence_counts == {KIND_IDENTICAL: 2}
    # `identical` records ARE stored: they are the rate denominator (§7.3).
    assert [r.left_span_id for r in result.records] == ["t1", "t2"]
    assert [r.right_span_id for r in result.records] == ["s1", "s2"]


def test_one_wrong_parameter_is_one_pair_not_two_orphans():
    result = align([step("t1", "get_task", task_id="1")],
                   [step("s1", "get_task", task_id="2")])
    assert len(result.records) == 1
    record = result.records[0]
    assert record.kind == KIND_PARAM_VALUE_ONLY
    assert record.left_span_id == "t1" and record.right_span_id == "s1"
    assert record.param_diff["changed"] == {"task_id": {"left": "1", "right": "2"}}
    assert record.param_diff["left_only"] == {} and record.param_diff["right_only"] == {}


def test_different_param_key_sets_are_same_command_different_params():
    result = align([step("t1", "get_task", task_id="1")],
                   [step("s1", "get_task", title="milk")])
    assert kinds(result) == [KIND_SAME_COMMAND_DIFFERENT_PARAMS]
    diff = result.records[0].param_diff
    assert diff["left_only"] == {"task_id": "1"}
    assert diff["right_only"] == {"title": "milk"}
    assert diff["changed"] == {}


def test_extra_student_step():
    left = [step("t1", "list")]
    right = [step("s1", "list"), step("s2", "get_task", task_id="1")]
    result = align(left, right)
    assert kinds(result) == [KIND_IDENTICAL, KIND_EXTRA_IN_STUDENT]
    extra = result.records[1]
    assert extra.right_span_id == "s2" and extra.left_span_id is None
    assert extra.left_step_key is None and extra.right_step_key is not None


def test_missing_student_step():
    left = [step("t1", "list"), step("t2", "get_task", task_id="1")]
    right = [step("s1", "list")]
    result = align(left, right)
    assert kinds(result) == [KIND_IDENTICAL, KIND_MISSING_IN_STUDENT]
    missing = result.records[1]
    assert missing.left_span_id == "t2" and missing.right_span_id is None


def test_pure_reordering_is_one_reordered_record():
    left = [step("t1", "list"), step("t2", "get_task", task_id="1")]
    right = [step("s1", "get_task", task_id="1"), step("s2", "list")]
    result = align(left, right)
    assert sorted(kinds(result)) == [KIND_IDENTICAL, KIND_REORDERED]
    reordered = next(r for r in result.records if r.kind == KIND_REORDERED)
    assert reordered.left_span_id == "t1" or reordered.right_span_id is not None
    assert reordered.left_step_key == reordered.right_step_key
    assert KIND_MISSING_IN_STUDENT not in kinds(result)
    assert KIND_EXTRA_IN_STUDENT not in kinds(result)


def test_reordering_requires_an_exactly_equal_step_key():
    # Crossed, same command, different parameters: §7.3 step 4 demands an
    # exactly equal step_key, so this stays two orphans rather than a
    # `reordered` pair that would hide the parameter difference.
    left = [step("t1", "get_task", task_id="1"), step("t2", "close")]
    right = [step("s1", "close"), step("s2", "get_task", task_id="9")]
    result = align(left, right)
    assert KIND_REORDERED not in kinds(result)
    assert sorted(kinds(result)) == [KIND_EXTRA_IN_STUDENT, KIND_IDENTICAL,
                                     KIND_MISSING_IN_STUDENT]


def test_crossed_identical_steps_reorder_around_a_matched_pair():
    # The same crossing, but with equal parameters on both sides: now the
    # post-pass rewrites the two orphans as one `reordered` record, and the
    # LCS-matched pair beside it is classified on its own merits.
    left = [step("t1", "list"), step("t2", "get_task", task_id="1")]
    right = [step("s1", "get_task", task_id="9"), step("s2", "list")]
    result = align(left, right)
    assert sorted(kinds(result)) == [KIND_PARAM_VALUE_ONLY, KIND_REORDERED]


# ---------------------------------------------------------------------------
# §7.3 step 4 — the reordering post-pass must be deterministic
# ---------------------------------------------------------------------------

def _tie_sequences():
    """Left-only `x` at index 3; right-only `x`s at 0 and 6 — both distance 3."""
    left = [step(f"t{n}", c) for n, c in enumerate(["a", "b", "c", "x", "d", "e"])]
    right = [step(f"s{n}", c) for n, c in enumerate(["x", "a", "b", "c", "d", "e", "x"])]
    return left, right


def test_reordering_tie_breaks_toward_the_earlier_student_index():
    left, right = _tie_sequences()
    result = align(left, right)
    reordered = next(r for r in result.records if r.kind == KIND_REORDERED)
    assert reordered.right_span_id == "s0"      # j = 0 wins over j = 6
    assert reordered.left_span_id == "t3"
    leftovers = [r for r in result.records if r.kind == KIND_EXTRA_IN_STUDENT]
    assert [r.right_span_id for r in leftovers] == ["s6"]


def test_reordering_tie_is_deterministic_across_repeated_runs():
    left, right = _tie_sequences()
    first = [r.to_row() for r in align(left, right).records]
    second = [r.to_row() for r in align(left, right).records]
    third = [r.to_row() for r in align(*_tie_sequences()).records]
    assert first == second == third


def test_reordering_tie_is_deterministic_on_mirrored_input():
    left, right = _tie_sequences()
    mirrored_first = [r.to_row() for r in align(right, left).records]
    mirrored_second = [r.to_row() for r in align(right, left).records]
    assert mirrored_first == mirrored_second
    reordered = next(r for r in align(right, left).records if r.kind == KIND_REORDERED)
    # Distance and student index both tie, so the earlier teacher index wins.
    assert reordered.left_span_id == "s0" and reordered.right_span_id == "t3"


# ---------------------------------------------------------------------------
# Empty sequences
# ---------------------------------------------------------------------------

def test_both_sequences_empty():
    result = align([], [])
    assert result.records == []
    assert (result.left_steps, result.right_steps, result.matched_pairs) == (0, 0, 0)
    assert result.divergence_counts == {}


def test_empty_teacher_makes_every_student_step_extra():
    result = align([], [step("s1", "list"), step("s2", "list")])
    assert kinds(result) == [KIND_EXTRA_IN_STUDENT, KIND_EXTRA_IN_STUDENT]
    assert result.left_steps == 0 and result.right_steps == 2


def test_empty_student_makes_every_teacher_step_missing():
    result = align([step("t1", "list")], [])
    assert kinds(result) == [KIND_MISSING_IN_STUDENT]


# ---------------------------------------------------------------------------
# §7.3 step 6 — the run-level record
# ---------------------------------------------------------------------------

def test_different_answer_same_actions_is_emitted_at_run_level():
    left = [step("t1", "list")]
    right = [step("s1", "list")]
    result = align(left, right, left_answer="3 tasks", right_answer="three tasks",
                   left_run_span_id="ta", right_run_span_id="sa")
    run_records = [r for r in result.records if r.level == "run"]
    assert len(run_records) == 1
    record = run_records[0]
    assert record.kind == KIND_DIFFERENT_ANSWER_SAME_ACTIONS
    assert (record.left_span_id, record.right_span_id) == ("ta", "sa")
    assert record.detail["left_answer"] == "3 tasks"


def test_run_level_record_tolerates_reordering():
    left = [step("t1", "list"), step("t2", "get_task", task_id="1")]
    right = [step("s1", "get_task", task_id="1"), step("s2", "list")]
    result = align(left, right, left_answer="a", right_answer="b")
    assert KIND_DIFFERENT_ANSWER_SAME_ACTIONS in kinds(result)


def test_no_run_level_record_when_the_actions_already_diverged():
    result = align([step("t1", "list")], [step("s1", "list"), step("s2", "list")],
                   left_answer="a", right_answer="b")
    assert KIND_DIFFERENT_ANSWER_SAME_ACTIONS not in kinds(result)


def test_no_run_level_record_when_the_answers_match_modulo_whitespace():
    result = align([step("t1", "list")], [step("s1", "list")],
                   left_answer="3 tasks", right_answer="3   tasks\n")
    assert kinds(result) == [KIND_IDENTICAL]


def test_no_run_level_record_when_an_answer_is_missing():
    result = align([step("t1", "list")], [step("s1", "list")], left_answer="a")
    assert kinds(result) == [KIND_IDENTICAL]


# ---------------------------------------------------------------------------
# §7.4 materiality [DR20]
# ---------------------------------------------------------------------------

def test_material_is_zero_when_the_exit_state_fingerprints_are_equal():
    result = align([step("t1", "get_task", task_id="1")],
                   [step("s1", "get_task", task_id="2")],
                   left_exit_state_fingerprint="fp",
                   right_exit_state_fingerprint="fp")
    assert result.records[0].kind == KIND_PARAM_VALUE_ONLY
    assert result.records[0].material == 0
    assert result.material_count == 0


def test_material_is_one_when_the_exit_state_fingerprints_differ():
    result = align([step("t1", "get_task", task_id="1")],
                   [step("s1", "get_task", task_id="2")],
                   left_exit_state_fingerprint="fp-a",
                   right_exit_state_fingerprint="fp-b")
    assert result.records[0].material == 1
    assert result.material_count == 1


def test_identical_records_are_never_material():
    result = align([step("t1", "list")], [step("s1", "list")],
                   left_exit_state_fingerprint="fp-a",
                   right_exit_state_fingerprint="fp-b")
    assert result.records[0].material == 0


def test_material_is_null_when_the_run_is_non_comparable():
    result = align([step("t1", "get_task", task_id="1")],
                   [step("s1", "get_task", task_id="2")],
                   comparable=False,
                   left_exit_state_fingerprint="fp-a",
                   right_exit_state_fingerprint="fp-b")
    assert result.records[0].material is None
    assert result.records[0].to_row()["material"] is None
    assert result.material_count == 0


def test_a_missing_fingerprint_is_not_evidence_of_agreement():
    assert compute_material(KIND_PARAM_VALUE_ONLY, True, None, None) == 1
    assert compute_material(KIND_PARAM_VALUE_ONLY, True, "fp", None) == 1
    assert compute_material(KIND_IDENTICAL, True, None, None) == 0
    assert compute_material(KIND_PARAM_VALUE_ONLY, False, "fp", "fp") is None


# ---------------------------------------------------------------------------
# The stored row shape (§9 DDL)
# ---------------------------------------------------------------------------

DDL_COLUMNS = [
    "divergence_id", "run_id", "level", "left_pass", "right_pass", "align_index",
    "kind", "material", "replayable", "command_key", "command_name", "context",
    "left_step_key", "right_step_key", "left_span_id", "right_span_id",
    "param_diff_json", "detail_json",
]


def test_to_row_maps_onto_the_ddl_columns():
    result = align([step("t1", "get_task", task_id="1")],
                   [step("s1", "get_task", task_id="2")])
    row = result.records[0].to_row()
    assert list(row) == DDL_COLUMNS
    assert row["level"] == "action"
    assert row["replayable"] == 1
    assert row["detail_json"]                       # NOT NULL in the DDL
    assert json.loads(row["param_diff_json"])["changed"]["task_id"]["right"] == "2"
    assert json.loads(row["detail_json"])["algorithm"] == ALGORITHM


def test_param_diff_json_is_null_when_there_is_nothing_to_highlight():
    result = align([step("t1", "list")], [step("s1", "list")])
    assert result.records[0].to_row()["param_diff_json"] is None


def test_param_diff_compares_nested_values_whole():
    diff = build_param_diff({"filter": {"a": 1, "b": 2}}, {"filter": {"a": 1, "b": 3}})
    assert diff["changed"]["filter"]["left"] == {"a": "1", "b": "2"}
    assert diff["changed"]["filter"]["right"] == {"a": "1", "b": "3"}


def test_align_indexes_are_contiguous_and_ids_deterministic():
    left, right = _tie_sequences()
    result = align(left, right)
    assert [r.align_index for r in result.records] == list(range(len(result.records)))
    assert len({r.divergence_id for r in result.records}) == len(result.records)
    assert result.records[0].divergence_id == divergence_id(
        "run-1", "action", "teacher", "student", 0)


def test_every_emitted_kind_is_in_the_taxonomy():
    left, right = _tie_sequences()
    result = align(left, right, left_answer="a", right_answer="b")
    assert {r.kind for r in result.records} <= KINDS


def test_compare_attributes_match_the_span_contract():
    left = [step("t1", "list")]
    right = [step("s1", "list"), step("s2", "get_task", task_id="1")]
    attrs = align(left, right).compare_attributes("action", "teacher", "student")
    assert attrs["left_steps"] == 1 and attrs["right_steps"] == 2
    assert attrs["matched_pairs"] == 1
    assert attrs["algorithm"] == ALGORITHM
    assert attrs["divergence_counts"][KIND_EXTRA_IN_STUDENT] == 1


def test_plan_level_alignment_uses_the_same_machinery():
    result = align([make_plan_step("t1", "list tasks then close")],
                   [make_plan_step("s1", "close then list tasks")],
                   level="plan")
    assert result.records[0].level == "plan"
    assert result.records[0].kind == KIND_PARAM_VALUE_ONLY


# ---------------------------------------------------------------------------
# §7.6 the prose summary is rendered from the record
# ---------------------------------------------------------------------------

def test_summary_omits_identical_records_by_default():
    result = align([step("t1", "list")], [step("s1", "list")])
    assert render_divergence_summary(result) == ""
    assert "identical" in render_divergence_summary(result, include_identical=True)


def test_summary_names_the_changed_parameter():
    result = align([step("t1", "get_task", task_id="1")],
                   [step("s1", "get_task", task_id="2")])
    summary = render_divergence_summary(result)
    assert "param-value-only" in summary
    assert "get_task" in summary
    assert "task_id" in summary and "'1'" in summary and "'2'" in summary


def test_summary_reports_orphans_by_pass_label():
    result = align_passes(run_id="run-1", left_pass="teacher", right_pass="student",
                          left_steps=[step("t1", "list")],
                          right_steps=[step("s1", "list"), step("s2", "close")])
    summary = render_divergence_summary(result)
    assert "extra-in-student" in summary
    assert "student executed close" in summary
    assert "teacher did not" in summary


def test_summary_flags_non_material_and_unknown_materiality():
    same_state = align([step("t1", "get_task", task_id="1")],
                       [step("s1", "get_task", task_id="2")],
                       left_exit_state_fingerprint="fp",
                       right_exit_state_fingerprint="fp")
    assert "non-material" in render_divergence_summary(same_state)
    unknown = align([step("t1", "get_task", task_id="1")],
                    [step("s1", "get_task", task_id="2")], comparable=False)
    assert "materiality unknown" in render_divergence_summary(unknown)


def test_summary_renders_the_run_level_record():
    result = align([step("t1", "list")], [step("s1", "list")],
                   left_answer="3 tasks", right_answer="four tasks")
    summary = render_divergence_summary(result)
    assert "different-answer-same-actions" in summary
    assert "3 tasks" in summary and "four tasks" in summary


def test_summary_is_rendered_from_the_record_alone():
    # §7.6: one source of truth — a record reconstructed from its own row
    # renders the same line, with no access to the original steps.
    result = align([step("t1", "get_task", task_id="1")],
                   [step("s1", "get_task", task_id="2")])
    original = result.records[0]
    row = original.to_row()
    rebuilt = DivergenceRecord(
        divergence_id=row["divergence_id"], run_id=row["run_id"], level=row["level"],
        left_pass=row["left_pass"], right_pass=row["right_pass"],
        align_index=row["align_index"], kind=row["kind"], material=row["material"],
        replayable=row["replayable"], command_key=row["command_key"],
        command_name=row["command_name"], context=row["context"],
        left_step_key=row["left_step_key"], right_step_key=row["right_step_key"],
        left_span_id=row["left_span_id"], right_span_id=row["right_span_id"],
        param_diff=json.loads(row["param_diff_json"]),
        detail=json.loads(row["detail_json"]),
    )
    assert render_record_summary(rebuilt) == render_record_summary(original)


def test_summary_of_a_failed_command_shows_its_raw_form():
    result = align([failed_step("t1", "Add task buy milk")], [])
    assert "raw:add task buy milk" in render_divergence_summary(result)


# ---------------------------------------------------------------------------
# A longer, mixed sequence — the taxonomy end to end
# ---------------------------------------------------------------------------

def test_mixed_sequence_produces_every_action_level_kind():
    left = [
        step("t1", "list"),
        step("t2", "get_task", task_id="1"),
        step("t3", "set_status", task_id="1", status="done"),
        make_ask_user_step("t4", "which project?"),
        step("t5", "close"),
    ]
    right = [
        step("s1", "list"),
        step("s2", "get_task", task_id="2"),
        step("s3", "set_status", task_id="1", state="done"),
        step("s4", "audit"),
        step("s5", "close"),
    ]
    result = align(left, right, left_exit_state_fingerprint="a",
                   right_exit_state_fingerprint="b")
    counts = result.divergence_counts
    assert counts[KIND_IDENTICAL] == 2
    assert counts[KIND_PARAM_VALUE_ONLY] == 1
    assert counts[KIND_SAME_COMMAND_DIFFERENT_PARAMS] == 1
    assert counts[KIND_MISSING_IN_STUDENT] == 1          # the ask_user step
    assert counts[KIND_EXTRA_IN_STUDENT] == 1            # the audit step
    assert result.material_count == 4
    assert len(render_divergence_summary(result).splitlines()) == 4
