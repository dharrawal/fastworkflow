"""The large tier's ambiguity threshold must not fall to the floor because it has no errors.

No mocks (repo rule): the pure functions run over real statistics dicts, the real
`analyze_model_confidence` runs over a real (tiny, untrained-weights-free) torch module and
loader, and the writer writes real files to `tmp_path`. No model is trained.
"""

import json

import pytest
import torch

from fastworkflow.train.model_pipeline_training import (
    CORRECT_CONFIDENCE_LOW_QUANTILE,
    LARGE_FAILED_ROWS_MIN,
    LARGE_RULE_CORRECT_QUANTILE,
    LARGE_RULE_FAILED_MEAN,
    LARGE_RULE_FLOOR_NO_STATISTICS,
    MAX_AMBIGUITY_THRESHOLD,
    SINGLE_LABEL_RESOLUTION_FLOOR,
    analyze_model_confidence,
    at_floor_without_wildcard,
    floored_large_ambiguous_threshold,
    large_ambiguous_threshold_and_rule,
    separated_tiny_ambiguous_threshold,
    write_ambiguity_thresholds,
)
from fastworkflow.train.heldout_evaluation import HeldoutReport


def _stats(failed_mean, failed_count, low_quantile, ok_mean=0.9):
    return {
        'failed': {'min': None, 'max': None, 'mean': failed_mean, 'median': None,
                   'count': failed_count},
        'successful': {'min': None, 'max': None, 'mean': ok_mean, 'median': None,
                       'count': 40, 'low_quantile': low_quantile},
    }


TINY = {
    'failed': {'mean': 0.4}, 'successful': {'mean': 0.8},
}


def test_zero_failures_use_the_correct_side_quantile_not_the_floor():
    value, rule = large_ambiguous_threshold_and_rule(_stats(None, 0, 0.81))
    assert value == pytest.approx(0.81)
    assert rule == LARGE_RULE_CORRECT_QUANTILE
    assert floored_large_ambiguous_threshold(_stats(None, 0, 0.81)) == pytest.approx(0.81)


def test_few_failures_use_the_correct_side_rule_even_when_their_mean_is_high():
    count = LARGE_FAILED_ROWS_MIN - 1
    value, rule = large_ambiguous_threshold_and_rule(_stats(0.95, count, 0.7))
    assert (value, rule) == (pytest.approx(0.7), LARGE_RULE_CORRECT_QUANTILE)
    # Few failures with a low mean used to give the floor; it must not.
    assert floored_large_ambiguous_threshold(_stats(0.3, count, 0.66)) == pytest.approx(0.66)


def test_enough_failures_keep_todays_value_exactly():
    value, rule = large_ambiguous_threshold_and_rule(
        _stats(0.71277, LARGE_FAILED_ROWS_MIN, 0.9)
    )
    assert value == pytest.approx(0.71277)
    assert rule == LARGE_RULE_FAILED_MEAN
    # A failed mean at or below the floor is still floored, as before.
    assert floored_large_ambiguous_threshold(
        _stats(0.41043, 12, 0.9)
    ) == pytest.approx(SINGLE_LABEL_RESOLUTION_FLOOR)


def test_both_rules_are_clamped_to_floor_and_cap():
    assert floored_large_ambiguous_threshold(_stats(None, 0, 0.2)) == pytest.approx(
        SINGLE_LABEL_RESOLUTION_FLOOR
    )
    assert floored_large_ambiguous_threshold(_stats(None, 0, 0.999)) == pytest.approx(
        MAX_AMBIGUITY_THRESHOLD
    )
    assert floored_large_ambiguous_threshold(_stats(0.999, 9, 0.9)) == pytest.approx(
        MAX_AMBIGUITY_THRESHOLD
    )


def test_no_statistics_at_all_is_the_floor():
    stats = _stats(None, 0, None, ok_mean=None)
    assert large_ambiguous_threshold_and_rule(stats) == (
        SINGLE_LABEL_RESOLUTION_FLOOR, LARGE_RULE_FLOOR_NO_STATISTICS
    )


def test_stats_without_a_count_behave_as_before():
    legacy = {'failed': {'mean': 0.7}, 'successful': {'mean': 0.9}}
    assert floored_large_ambiguous_threshold(legacy) == pytest.approx(0.7)
    legacy_none = {'failed': {'mean': None}, 'successful': {'mean': None}}
    assert floored_large_ambiguous_threshold(legacy_none) == pytest.approx(
        SINGLE_LABEL_RESOLUTION_FLOOR
    )


def test_tiny_tier_is_unchanged_by_the_new_stats():
    plain = separated_tiny_ambiguous_threshold(0.5, TINY)
    extra = separated_tiny_ambiguous_threshold(
        0.5,
        {'failed': {'mean': 0.4, 'count': 0}, 'successful': {'mean': 0.8, 'low_quantile': 0.1}},
    )
    assert plain == extra == pytest.approx(0.6)


class _FixedLogits(torch.nn.Module):
    """A real torch module returning fixed logits per row; no training involved."""

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    def forward(self, input_ids, attention_mask=None):
        return self._Out(input_ids.float())


def _loader(logit_rows, labels):
    ids = torch.tensor(logit_rows)
    return [({'input_ids': ids, 'attention_mask': torch.ones_like(ids)},
             torch.tensor(labels), None)]


def test_analyze_reports_failure_count_and_correct_side_quantile():
    rows = [[5.0, 0.0]] * 9 + [[0.1, 0.0]]  # nine confident, one weak; all predict class 0
    stats, *_ = analyze_model_confidence(
        _FixedLogits(), _loader(rows, [0] * 10), "cpu", "t")
    assert stats['failed']['count'] == 0 and stats['failed']['mean'] is None
    assert stats['successful']['count'] == 10
    expected = float(torch.softmax(torch.tensor([5.0, 0.0]), 0)[0])
    assert stats['successful']['low_quantile'] > 0.5
    assert stats['successful']['low_quantile'] <= expected
    assert CORRECT_CONFIDENCE_LOW_QUANTILE == 0.10

    wrong, *_ = analyze_model_confidence(
        _FixedLogits(), _loader(rows[:3], [1, 1, 0]), "cpu", "t")
    assert wrong['failed']['count'] == 2 and wrong['successful']['count'] == 1


def test_writer_writes_the_correct_side_value_for_zero_failures(tmp_path):
    _tiny, large = write_ambiguity_thresholds(
        str(tmp_path / "Ctx"), 0.5, TINY, _stats(None, 0, 0.83))
    on_disk = json.load(open(tmp_path / "Ctx" / "large_ambiguous_threshold.json"))
    assert large == pytest.approx(0.83) == pytest.approx(on_disk['confidence_threshold'])


def test_floor_without_wildcard_warns_and_still_writes_files(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _t, large = write_ambiguity_thresholds(
            str(tmp_path / "Ctx"), 0.5, TINY, _stats(0.3, 20, 0.9),
            has_wildcard_class=False)
    assert large == pytest.approx(SINGLE_LABEL_RESOLUTION_FLOOR)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "Ctx" in warnings[0].getMessage() and "no wildcard" in warnings[0].getMessage()
    assert at_floor_without_wildcard(large, False) is True
    assert (tmp_path / "Ctx" / "large_ambiguous_threshold.json").exists()
    assert (tmp_path / "Ctx" / "tiny_ambiguous_threshold.json").exists()


def test_no_warning_with_a_wildcard_class_or_above_the_floor(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _t, large = write_ambiguity_thresholds(
            str(tmp_path / "A"), 0.5, TINY, _stats(0.3, 20, 0.9), has_wildcard_class=True)
        assert large == pytest.approx(SINGLE_LABEL_RESOLUTION_FLOOR)
        _t, large = write_ambiguity_thresholds(
            str(tmp_path / "B"), 0.5, TINY, _stats(None, 0, 0.8), has_wildcard_class=False)
        assert large == pytest.approx(0.8)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
    assert at_floor_without_wildcard(0.5, True) is False
    assert at_floor_without_wildcard(0.5, None) is False
    assert at_floor_without_wildcard(0.8, False) is False


def test_heldout_report_carries_the_large_ambiguity_record():
    record = {"failed_rows": 0, "rule": LARGE_RULE_CORRECT_QUANTILE,
              "threshold": 0.8, "has_wildcard_class": False,
              "at_floor_without_wildcard": False}
    report = HeldoutReport(context="Ctx", large_ambiguity=record)
    assert HeldoutReport.model_validate(report.model_dump()).large_ambiguity == record
    assert HeldoutReport(context="Old").large_ambiguity is None
