"""Regression tests for fix-ajv.15: the intent decoder must not share module state.

``predict_single_sentence`` used to load the label encoder into a process global and
read it back after ``predict_batch``. The forward pass releases the GIL for tens of
milliseconds, so a prediction for another context could rebind the global inside that
window and the decode would return that other context's command.
"""

import pickle
import threading
import time

from sklearn.preprocessing import LabelEncoder

from fastworkflow.model_pipeline_training import (
    get_label_encoder,
    predict_single_sentence,
)


def _write_encoder(path, classes):
    encoder = LabelEncoder()
    encoder.fit(classes)
    with open(path, 'wb') as f:
        pickle.dump(encoder, f)
    return str(path)


class _FakePipeline:
    """Stands in for ModelPipeline: only ``predict_batch`` is on the decode path.

    ``on_forward`` runs where the real forward pass would, i.e. after the caller has
    resolved its encoder and before it decodes - the exact window the race lived in.
    """

    def __init__(self, top_k, on_forward=None):
        self.top_k = top_k
        self.on_forward = on_forward

    def predict_batch(self, texts, batch_size=32, k_val=None):
        if self.on_forward is not None:
            self.on_forward()
        return {
            "predictions": [self.top_k[0]],
            "confidences": [0.91],
            "used_distil": [False],
            "top_k_predictions": [list(self.top_k)],
            "top_k_scores": [[0.91, 0.06, 0.03]],
        }


def test_decode_uses_own_context_encoder_when_another_load_interleaves(tmp_path):
    """A second context's load landing mid-forward-pass must not steer the decode."""
    path_a = _write_encoder(tmp_path / "a.pkl", ["a_create", "a_delete", "a_list"])
    path_b = _write_encoder(tmp_path / "b.pkl", ["b_pay", "b_refund", "b_status"])

    a_in_forward = threading.Event()
    b_decoded = threading.Event()

    def park_until_b_has_loaded():
        a_in_forward.set()
        assert b_decoded.wait(timeout=10), "context B never completed its prediction"

    result_a = {}

    def run_a():
        result_a["value"] = predict_single_sentence(
            _FakePipeline([1, 0, 2], on_forward=park_until_b_has_loaded),
            "delete the thing",
            path_a,
        )

    thread_a = threading.Thread(target=run_a)
    thread_a.start()
    try:
        assert a_in_forward.wait(timeout=10), "context A never reached its forward pass"
        result_b = predict_single_sentence(
            _FakePipeline([2, 1, 0]), "what is my balance", path_b
        )
    finally:
        b_decoded.set()
        thread_a.join(timeout=10)

    assert not thread_a.is_alive()
    assert result_b["label"] == "b_status"
    assert result_a["value"]["label"] == "a_delete"
    assert list(result_a["value"]["topk_labels"]) == ["a_delete", "a_create", "a_list"]


def test_concurrent_predictions_across_contexts_never_cross_decode(tmp_path):
    """The same isolation under real thread interleaving rather than a forced one."""
    contexts = {
        _write_encoder(tmp_path / "ctx1.pkl", ["c1_add", "c1_drop", "c1_show"]): "c1_",
        _write_encoder(tmp_path / "ctx2.pkl", ["c2_ship", "c2_track", "c2_void"]): "c2_",
    }

    # A bare yield is enough: the race needed only that some other thread runs
    # between the caller resolving its encoder and decoding with it.
    def yield_thread():
        time.sleep(0.001)

    errors = []
    mismatches = []

    def predict_many(path, prefix):
        pipeline = _FakePipeline([1, 0, 2], on_forward=yield_thread)
        for _ in range(40):
            try:
                result = predict_single_sentence(pipeline, "do the thing", path)
            except Exception as exc:  # noqa: BLE001 - the old code raised here too
                errors.append(f"{prefix}: {exc!r}")
                return
            labels = [result["label"], *result["topk_labels"]]
            mismatches.extend(
                label for label in labels if not str(label).startswith(prefix)
            )

    threads = [
        threading.Thread(target=predict_many, args=(path, prefix))
        for path, prefix in contexts.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert not mismatches


def test_encoder_is_unpickled_once_per_artefact_version(tmp_path):
    """Repeat predictions reuse the cached encoder; a retrain invalidates it."""
    path = _write_encoder(tmp_path / "ctx.pkl", ["c_add", "c_drop", "c_show"])

    first = get_label_encoder(path)
    assert get_label_encoder(path) is first

    _write_encoder(tmp_path / "ctx.pkl", ["c_add", "c_drop", "c_show", "c_purge"])
    retrained = get_label_encoder(path)
    assert retrained is not first
    assert list(retrained.classes_) == ["c_add", "c_drop", "c_purge", "c_show"]


def test_prediction_does_not_disturb_the_trainer_global(tmp_path):
    """The trainer fits and pickles the module global; inference must leave it alone."""
    import fastworkflow.model_pipeline_training as mpt

    trainer_encoder = mpt.label_encoder
    path = _write_encoder(tmp_path / "ctx.pkl", ["c_add", "c_drop", "c_show"])

    predict_single_sentence(_FakePipeline([0, 1, 2]), "add one", path)

    assert mpt.label_encoder is trainer_encoder
