"""The ``embedding`` intent model layout's head (bd fix-o5wi.2, fix-o5wi.9): the decision in a context
(it never passes a request up on its own), the training rows (description rows, seeds weighted), the
cross-fit report (accuracy and confused pairs), the fit and its artifacts, and the runtime reader behind
``CommandRouter``.

The decision and report tests are arithmetic on hand-made probabilities. The fit and reader
tests use the real default sentence encoder on a few hand-written rows (no workflow is trained, no
language model is called); they skip, saying why, on a machine that cannot load the encoder. The
end-to-end run (`fastworkflow train` in the layout, then the runtime) is the slow test in
``tests/test_intent_model_layout_training.py``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import fastworkflow
from fastworkflow import intent_detector
from fastworkflow import model_pipeline_training as heads
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, PARAMETER_VALUE_PLACEHOLDERS, WILDCARD_LABEL
from fastworkflow.train import artifact_versioning
from fastworkflow.train import embedding_head as eh
from fastworkflow.train import intent_model_layout as layout
from fastworkflow.train import workflow_model

LABELS = ["Book/list_loans", "Book/show_book", "Library/list_branches", PARAMETER_VALUE_LABEL, WILDCARD_LABEL]
BOOK_MASK = ["Book/list_loans", "Book/show_book"]


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


def _decide(probabilities, threshold=0.5, margin=None, mask=BOOK_MASK):
    inside, outside = eh.mask_indices(LABELS, mask)
    return eh.decide(np.array(probabilities, dtype=float), LABELS, inside, outside, threshold, margin)


# ---------------------------------------------------------------- the decision in a context

def test_the_mask_holds_the_contexts_commands_and_the_reserved_labels_are_on_neither_side():
    inside, outside = eh.mask_indices(LABELS, BOOK_MASK)
    assert [LABELS[i] for i in inside] == BOOK_MASK
    assert [LABELS[i] for i in outside] == ["Library/list_branches"]


@pytest.mark.parametrize("probabilities, decision, labels", [
    # one command: the best label inside the mask is at or above the threshold and nothing outside is ahead
    ([0.70, 0.10, 0.10, 0.05, 0.05], "match", ["Book/list_loans"]),
    ([0.50, 0.20, 0.20, 0.05, 0.05], "match", ["Book/list_loans"]),            # at the threshold
    # below the threshold: the context's top candidates, best first
    ([0.30, 0.45, 0.10, 0.10, 0.05], "ambiguous", ["Book/show_book", "Book/list_loans"]),
    # a command outside the mask is ahead: the head NEVER says "not here" for that, however far ahead
    ([0.10, 0.05, 0.75, 0.05, 0.05], "ambiguous", ["Book/list_loans", "Book/show_book"]),
    ([0.20, 0.05, 0.50, 0.15, 0.10], "ambiguous", ["Book/list_loans", "Book/show_book"]),
    ([0.35, 0.05, 0.45, 0.10, 0.05], "ambiguous", ["Book/list_loans", "Book/show_book"]),
    # ... even when the inside label is above the threshold: it is not the best
    ([0.42, 0.02, 0.52, 0.02, 0.02], "ambiguous", ["Book/list_loans", "Book/show_book"]),
    # the reserved labels are read as the best label overall
    ([0.20, 0.10, 0.10, 0.50, 0.10], "bare_value", [PARAMETER_VALUE_LABEL]),
    ([0.25, 0.10, 0.10, 0.05, 0.50], "not_mine", [WILDCARD_LABEL]),
])
def test_the_decision_in_a_context_never_passes_up_alone(probabilities, decision, labels):
    got, returned, _numbers = _decide(probabilities, threshold=0.5)
    assert (got, returned) == (decision, labels)


def test_a_hand_on_names_the_best_commands_outside_the_mask_best_first():
    """The information a pass-up used to throw away: the next link is told where the head thinks the
    command is."""
    labels = ["Book/a", "Book/b", "Library/x", "Library/y", "Library/z", "*/hours", PARAMETER_VALUE_LABEL, WILDCARD_LABEL]
    inside, outside = eh.mask_indices(labels, ["Book/a", "Book/b"])
    probabilities = np.array([0.05, 0.02, 0.10, 0.05, 0.03, 0.60, 0.10, 0.05])
    decision, returned, numbers = eh.decide(probabilities, labels, inside, outside, 0.65)
    assert decision == "ambiguous" and returned == ["Book/a", "Book/b"]
    assert numbers["outside_labels"] == ["*/hours", "Library/x", "Library/y", "Library/z"]
    assert numbers["outside_scores"] == pytest.approx([0.60, 0.10, 0.05, 0.03])
    # An unconfident answer whose best command is inside the mask carries no outside commands.
    _d, _r, inside_best = eh.decide(
        np.array([0.40, 0.02, 0.10, 0.05, 0.03, 0.30, 0.05, 0.05]), labels, inside, outside, 0.65)
    assert inside_best["outside_labels"] == [] and inside_best["outside_scores"] == []


def test_the_older_rule_with_a_margin_is_still_read_for_a_head_trained_before_the_change():
    assert _decide([0.10, 0.05, 0.75, 0.05, 0.05], margin=0.3)[:2] == ("not_mine", [WILDCARD_LABEL])
    assert _decide([0.20, 0.05, 0.50, 0.15, 0.10], margin=0.3)[:2] == ("not_mine", [WILDCARD_LABEL])   # exactly the margin
    assert _decide([0.35, 0.05, 0.45, 0.10, 0.05], margin=0.3)[:2] == (
        "ambiguous", ["Book/list_loans", "Book/show_book"])
    assert _decide([0.35, 0.05, 0.45, 0.10, 0.05], margin=0.3)[2]["outside_labels"] == []


def test_the_probabilities_are_not_renormalised_inside_the_mask():
    """0.45 of everything is 0.9 of the mask. The rule reads 0.45."""
    decision, labels, numbers = _decide([0.45, 0.05, 0.10, 0.20, 0.20], threshold=0.5)
    assert decision == "ambiguous" and labels == ["Book/list_loans", "Book/show_book"]
    assert numbers["p_in"] == pytest.approx(0.45) and numbers["p_out"] == pytest.approx(0.10)
    assert numbers["topk_scores"] == pytest.approx([0.45, 0.05])        # the probabilities as the head gave them


def test_the_reserved_labels_do_not_count_as_outside_the_mask():
    """A wildcard or bare-value probability below the best label never holds back a command."""
    decision, labels, numbers = _decide([0.50, 0.02, 0.03, 0.05, 0.40], threshold=0.5)
    assert decision == "match" and labels == ["Book/list_loans"] and numbers["p_out"] == pytest.approx(0.03)


def test_a_context_that_holds_every_command_has_nothing_outside():
    decision, labels, numbers = _decide(
        [0.2, 0.2, 0.3, 0.2, 0.1], threshold=0.5, mask=["Book/list_loans", "Book/show_book", "Library/list_branches"])
    assert decision == "ambiguous" and numbers["p_out"] == 0.0 and numbers["outside_labels"] == []
    assert labels == ["Library/list_branches", "Book/list_loans", "Book/show_book"]      # the top three


def test_at_most_three_candidates_are_returned():
    labels = [f"c{i}" for i in range(6)]
    inside, outside = eh.mask_indices(labels, labels[:5])
    _decision, returned, numbers = eh.decide(
        np.array([0.1, 0.25, 0.2, 0.15, 0.05, 0.25]), labels, inside, outside, 0.9)
    assert returned == ["c1", "c2", "c3"] == numbers["topk_labels"]


def test_the_default_threshold_is_a_constant_and_the_calibration_is_gone():
    assert layout.DEFAULT_COMMIT_THRESHOLD == 0.65
    for removed in ("calibrate", "calibration_records", "THRESHOLD_FLOOR"):
        assert not hasattr(eh, removed)


# ---------------------------------------------------------------- the report's cross-fit numbers

def test_the_confused_pairs_list_both_directions_and_skip_the_reserved_labels():
    labels = ["Book/list_loans", "Book/show_book", "Library/list_branches", PARAMETER_VALUE_LABEL, WILDCARD_LABEL]
    rows = ([(f"loans {i}", "Book/list_loans") for i in range(10)]
            + [(f"show {i}", "Book/show_book") for i in range(10)]
            + [(f"branch {i}", "Library/list_branches") for i in range(10)])
    probabilities = np.full((30, 5), 0.01)
    for position in range(30):
        probabilities[position, position // 10] = 0.9
    for position in range(3):                       # 3 of 10 loans rows go to show_book
        probabilities[position, :] = 0.01
        probabilities[position, 1] = 0.9
    probabilities[10, :] = 0.01                     # one show row goes to loans
    probabilities[10, 0] = 0.9
    probabilities[20, :] = 0.01                     # one branch row is called a vague reply: not a pair
    probabilities[20, 4] = 0.9

    report = eh.cross_fit_report(probabilities, rows, labels, {"Book": labels[:2], "Library": labels[2:3]}, 30, {})

    assert report["rows"] == 30 and report["accuracy"] == pytest.approx(25 / 30)
    assert report["confused_pairs"] == 1 and report["pairs_over_limit"] == 1
    pair = report["pairs"][0]
    assert (pair["a"], pair["b"]) == ("Book/list_loans", "Book/show_book")
    assert (pair["a_as_b"], pair["a_rows"], pair["b_as_a"], pair["b_rows"]) == (3, 10, 1, 10)
    assert pair["a_share"] == pytest.approx(0.3) and pair["b_share"] == pytest.approx(0.1)
    assert pair["max_share"] == pytest.approx(0.3) and pair["total"] == 4
    # Per context: the rows of the commands it holds, and how many its best command is right on.
    assert report["scopes"]["Book"] == {"rows": 20, "right": 16, "accuracy": pytest.approx(0.8)}
    # A context holding one command is right on every row of it: the best label INSIDE the mask is read.
    assert report["scopes"]["Library"]["rows"] == 10 and report["scopes"]["Library"]["right"] == 10


def test_only_the_top_ten_pairs_are_kept_and_all_are_counted():
    labels = [f"C/c{i}" for i in range(24)] + [PARAMETER_VALUE_LABEL, WILDCARD_LABEL]
    rows = [(f"row {c} {i}", labels[c]) for c in range(24) for i in range(4)]
    probabilities = np.full((len(rows), 26), 0.001)
    for position, (_text, label) in enumerate(rows):
        probabilities[position, labels.index(label)] = 0.9
    for pair in range(12):                          # 12 disjoint pairs, one row of each pair's first command goes to the second
        probabilities[(2 * pair) * 4] = 0.001
        probabilities[(2 * pair) * 4, 2 * pair + 1] = 0.9
    report = eh.cross_fit_report(probabilities, rows, labels, {"C": labels[:24]}, len(rows), {})
    assert report["confused_pairs"] == 12 and len(report["pairs"]) == eh.PAIRS_REPORTED == 10
    assert report["pairs_over_limit"] == 12         # one of four rows is 25%


def test_cross_fitting_never_puts_a_repeated_text_on_both_sides():
    """The built-in vague replies are repeated; the folds are grouped by text."""
    generator = np.random.default_rng(0)
    texts = [f"text {i}" for i in range(20)] * 3
    targets = np.array([i % 4 for i in range(20)] * 3)
    centres = generator.normal(size=(4, 8))
    vectors = np.tile(centres[targets[:20]] + 0.1 * generator.normal(size=(20, 8)), (3, 1)).astype(np.float32)

    probabilities, facts = eh.cross_fitted_probabilities(vectors, targets, texts, 4)

    assert facts == {"folds": 5, "grouped_by": "row text", "cross_fitted": True}
    assert probabilities.shape == (60, 4) and np.allclose(probabilities.sum(axis=1), 1.0)
    # Each repeat of a text was scored by the same held-out head: grouped folds give it the same numbers.
    assert np.allclose(probabilities[:20], probabilities[20:40], atol=1e-6)
    assert np.allclose(probabilities[:20], probabilities[40:], atol=1e-6)

    # Too few distinct rows to cross-fit: the facts say the numbers are in-sample.
    _p, facts = eh.cross_fitted_probabilities(vectors[:3], np.array([0, 1, 2]), ["same"] * 3, 3)
    assert facts["cross_fitted"] is False and facts["folds"] == 0


# ---------------------------------------------------------------- the fit, the artifacts and the reader

ROWS = {
    "Book/list_loans": [
        "list the loans of this book", "who has borrowed this book", "show me the loans for this title",
        "which members have this book out", "loan history of this book", "who is reading this one now",
        "has anyone borrowed it", "current loans on this book"],
    "Book/show_book": [
        "show this book", "what are the details of this book", "tell me about this book",
        "display the book's information", "book details please", "what is this book",
        "show me its author and year", "describe this title"],
    "Library/list_branches": [
        "list all the branches", "which branches does the library have", "show me the library branches",
        "where are your locations", "what branches are there", "name every branch of the library",
        "all branch locations please", "how many branches do you run"],
    "library_hours": [
        "when is the library open", "what are your opening hours", "are you open on sunday",
        "what time do you close today", "opening times please", "is the library open tomorrow morning",
        "hours of operation", "when do you open on weekdays"],
}
MASKS = {"Book": ["Book/list_loans", "Book/show_book"], "Library": ["Library/list_branches"],
         "*": ["library_hours"]}


def _rows() -> list[tuple[str, str]]:
    scopes = {
        name: workflow_model.ScopeRows(
            name=name, mask=sorted(mask),
            cached_rows={command: list(ROWS[command]) for command in mask},
            trained_rows={command: list(ROWS[command]) for command in mask})
        for name, mask in MASKS.items()
    }
    rows, _findings = workflow_model.assemble_rows(scopes)       # the assembly the layout trainer uses
    return rows


@pytest.fixture(scope="module")
def encoder():
    try:
        return eh.SentenceEncoder.get(layout.DEFAULT_ENCODER)
    except Exception as error:  # noqa: BLE001 - no network and no cached copy: nothing to test with
        pytest.skip(f"the default sentence encoder {layout.DEFAULT_ENCODER} cannot be loaded here "
                    f"({type(error).__name__}: {error})")


@pytest.fixture(scope="module")
def published_head(tmp_path_factory, encoder):
    """A version folder in the embedding layout, fitted on the hand-written rows and published the way
    the trainer publishes: ``workflow/`` beside the context folders, and the usual links."""
    workflow = str(tmp_path_factory.mktemp("embedding_head") / "wf")
    version = "v1"
    vdir = artifact_versioning.version_dir(workflow, version)
    fitted = eh.fit_embedding_head(_rows(), MASKS, str(vdir / layout.WORKFLOW_MODEL_DIRNAME))
    for context_name in MASKS:
        layout.write_context_files(
            str(vdir / artifact_versioning.context_folder_name(context_name)), context_name, layout.EMBEDDING)
    artifact_versioning.write_manifest(workflow, version, seed=42)
    artifact_versioning.publish_version(workflow, version)
    yield workflow, vdir, fitted
    for key in [key for key in heads.CommandRouter._instances_cache if key.startswith(workflow)]:
        heads.CommandRouter._instances_cache.pop(key, None)


def _router(workflow: str, context: str):
    return heads.CommandRouter(intent_detector.head_folder(workflow, context))


def test_the_rows_are_the_workflow_layouts_rows():
    rows = _rows()
    labels = {label for _text, label in rows}
    assert labels == set(ROWS) | {PARAMETER_VALUE_LABEL, WILDCARD_LABEL}
    assert sum(label == PARAMETER_VALUE_LABEL for _t, label in rows) == len(set(PARAMETER_VALUE_PLACEHOLDERS))
    assert sum(label == WILDCARD_LABEL for _t, label in rows) == (
        len(heads.BUILT_IN_VAGUE_REPLIES) * heads.BUILT_IN_VAGUE_WEIGHT)


def test_the_artifact_is_the_head_and_a_scopes_file(published_head):
    _workflow, vdir, fitted = published_head
    model_dir = vdir / layout.WORKFLOW_MODEL_DIRNAME
    assert sorted(os.listdir(model_dir)) == ["head.npz", "rows.npz", "scopes.json"]

    scopes = json.loads((model_dir / "scopes.json").read_text())
    assert scopes == json.loads(json.dumps(fitted.payload))
    assert scopes["schema_version"] == 2
    assert scopes["layout"] == "embedding" and scopes["labels"] == sorted(set(ROWS) | {
        PARAMETER_VALUE_LABEL, WILDCARD_LABEL})
    assert scopes["encoder"] == {"model": "intfloat/e5-base-v2", "pooling": "mean", "prefix": "query: ",
                                 "max_length": 128, "dimensions": 768, "normalized": True}
    assert scopes["head"] == {"kind": "logistic_regression", "C": 100.0, "max_iter": 3000, "file": "head.npz"}
    assert scopes["rows"]["total"] == len(_rows()) and scopes["rows"]["own"] == len(_rows())
    assert {scope: entry["mask"] for scope, entry in scopes["scopes"].items()} == MASKS
    # One fixed commit threshold for the head, repeated on each context; no margin anywhere.
    assert scopes["threshold"] == layout.DEFAULT_COMMIT_THRESHOLD == 0.65
    assert scopes["decision"] == {"rule": "never_passes_up", "commit_threshold": 0.65, "outside_candidates": 10}
    assert all(entry["threshold"] == 0.65 and "margin" not in entry for entry in scopes["scopes"].values())
    assert "margin" not in scopes and "calibration" not in scopes
    cross_fit = scopes["cross_fit"]
    assert cross_fit["rows"] == sum(len(ROWS[c]) for c in ROWS) and cross_fit["cross_fitted"] is True
    assert 0.0 <= cross_fit["accuracy"] <= 1.0 and "pairs" in cross_fit and "pairs_over_limit" in cross_fit
    assert all(scopes[name] >= 0.0 for name in ("embed_seconds", "fit_seconds", "cross_fit_seconds"))

    with np.load(model_dir / "head.npz") as head:
        assert sorted(head.files) == ["coef", "intercept"]
        assert head["coef"].shape == (len(scopes["labels"]), 768) and head["coef"].dtype == np.float32
        assert head["intercept"].shape == (len(scopes["labels"]),)
    # A few tens of kilobytes: the encoder's weights are not in the artifact.
    assert sum(os.path.getsize(model_dir / name) for name in os.listdir(model_dir)) < 150_000

    summary = eh.read_summary(str(model_dir))
    assert summary["encoder"] == "intfloat/e5-base-v2" and summary["labels"] == len(scopes["labels"])
    assert summary["threshold"] == 0.65 and summary["cross_fit"]["rows"] == cross_fit["rows"]


def test_the_default_encoder_is_e5_base_with_its_pooling_and_prefix():
    assert layout.DEFAULT_ENCODER == "intfloat/e5-base-v2"
    assert layout.default_pooling(layout.DEFAULT_ENCODER) == "mean"
    assert layout.default_prefix(layout.DEFAULT_ENCODER) == "query: "
    assert layout.default_prefix("BAAI/bge-small-en-v1.5") == ""


def test_description_rows_and_seed_weights_are_in_the_assembled_rows():
    hello_world = os.path.join("fastworkflow", "examples", "hello_world")
    crd = RoutingRegistry.get_definition(hello_world)
    seed = "add two numbers"
    rows = [(seed, "add_two_numbers"), ("please sum 2 and 3", "add_two_numbers"), ("france", PARAMETER_VALUE_LABEL),
            ("hmm", WILDCARD_LABEL)]

    built = eh.training_rows(rows, crd)

    assert built.rows[:4] == rows and built.base_count == 4
    # Seeds weigh three times a generated row; reserved rows and other texts weigh one.
    assert list(built.weights[:4]) == [3.0, 1.0, 1.0, 1.0] and built.seed_rows == 1
    # One row of the humanised name and one of the first sentence of the description, for each command.
    assert built.rows[4:] == [
        ("add two numbers", "add_two_numbers"),
        ("Takes two floating point numbers and returns their sum", "add_two_numbers")]
    assert built.description_rows == 2 and list(built.weights[4:]) == [1.0, 1.0]
    # Without a routing definition nothing is added or weighted.
    plain = eh.training_rows(rows)
    assert plain.rows == rows and set(plain.weights) == {1.0} and plain.description_rows == 0


def test_a_weighted_fit_differs_from_an_unweighted_one_and_a_uniform_weight_is_the_plain_fit():
    generator = np.random.default_rng(1)
    vectors = generator.normal(size=(40, 6)).astype(np.float32)
    targets = np.array([i % 4 for i in range(40)])
    plain = eh.fit_head(vectors, targets, 4)
    assert all(np.allclose(a, b) for a, b in zip(plain, eh.fit_head(vectors, targets, 4, np.ones(40))))
    heavy = np.where(np.arange(40) < 10, 3.0, 1.0)
    assert not np.allclose(plain[0], eh.fit_head(vectors, targets, 4, heavy)[0], atol=1e-4)


def test_inference_is_numpy_and_gives_what_the_fitted_head_gives(encoder):
    rows = _rows()
    labels = sorted({label for _text, label in rows})
    targets = np.array([labels.index(label) for _text, label in rows])
    vectors = encoder.embed([text for text, _label in rows])
    assert vectors.shape == (len(rows), 768) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)          # L2-normalised

    head = eh._logistic_regression().fit(vectors.astype(np.float64), targets)       # fitted in float64
    coef, intercept = eh.fit_head(vectors, targets, len(labels))
    queries = encoder.embed(["who took this book home", "opening hours on monday", "france", "hmm not sure"])
    assert np.allclose(eh.softmax_probabilities(queries, coef, intercept), head.predict_proba(queries), atol=1e-4)


def test_the_reader_answers_through_the_mask_with_the_documented_details(published_head):
    workflow, _vdir, fitted = published_head
    book = _router(workflow, "Book")
    assert book._workflow_router.__class__ is eh.EmbeddingHeadRouter
    assert book.tiny_path is None and book.large_path is None

    labels, details = book.predict_with_details("list the loans of this book")
    assert labels == ["Book/list_loans"]
    # The keys the fine-tuned workflow model's reader returns, so every consumer reads them unchanged.
    assert set(details) == {"model_tier", "confidence", "ambiguous_threshold", "confident", "top_label",
                            "topk_labels", "topk_scores", "p_in", "p_out", "outside_labels", "outside_scores",
                            "head_version", "threshold_source"}
    assert details["model_tier"] == "embedding" and details["confident"] is True
    assert details["top_label"] == "Book/list_loans" and details["confidence"] == details["p_in"]
    assert details["ambiguous_threshold"] == fitted.payload["threshold"]
    assert details["topk_labels"] == ["Book/list_loans", "Book/show_book"]        # the context's own, best first
    assert details["topk_scores"] == sorted(details["topk_scores"], reverse=True)
    assert details["p_in"] >= details["ambiguous_threshold"] and details["p_in"] >= details["p_out"]
    assert book.predict("list the loans of this book") == ["Book/list_loans"]


def test_another_contexts_command_is_handed_on_never_passed_up_alone(published_head):
    workflow, _vdir, _fitted = published_head
    request = "which branches does the library have"
    labels, details = _router(workflow, "Book").predict_with_details(request)
    # Not the lone "not here" it used to be: the context's own candidates, not confident, and the
    # command the head thinks is meant named beside them.
    assert len(labels) > 1 and WILDCARD_LABEL not in labels and details["confident"] is False
    assert details["p_out"] > details["p_in"]
    assert details["outside_labels"][0] == "Library/list_branches"
    assert details["outside_scores"][0] == pytest.approx(details["p_out"])
    assert _router(workflow, "Library").predict(request) == ["Library/list_branches"]
    assert _router(workflow, "*").predict("what are your opening hours") == ["library_hours"]


def test_a_head_trained_before_the_change_is_read_as_it_always_was(published_head, tmp_path):
    """The router tells the rule from ``scopes.json``: no ``schema_version`` 2 and a ``margin`` means the
    older rule, with the head's own threshold and margin, until the workflow is retrained."""
    _workflow, vdir, _fitted = published_head
    legacy = tmp_path / "workflow"
    legacy.mkdir()
    for name in ("head.npz", "scopes.json"):
        (legacy / name).write_bytes((vdir / layout.WORKFLOW_MODEL_DIRNAME / name).read_bytes())
    scopes = json.loads((legacy / "scopes.json").read_text())
    scopes["schema_version"] = 1
    scopes["margin"] = 0.0
    scopes["threshold"] = 0.5
    for entry in scopes["scopes"].values():
        entry["margin"], entry["threshold"] = 0.0, 0.5
    (legacy / "scopes.json").write_text(json.dumps(scopes))

    router = eh.EmbeddingHeadRouter(str(legacy), "Book")
    assert router.escalation_margin == 0.0
    labels, details = router.predict_with_details("which branches does the library have")
    assert labels == [WILDCARD_LABEL] and details["escalation_margin"] == 0.0
    assert eh.EmbeddingHeadRouter(str(vdir / layout.WORKFLOW_MODEL_DIRNAME), "Book").escalation_margin is None


def test_a_bare_value_and_a_vague_reply_are_the_reserved_labels_in_every_context(published_head):
    workflow, _vdir, _fitted = published_head
    value = sorted(PARAMETER_VALUE_PLACEHOLDERS)[0]
    vague = heads.BUILT_IN_VAGUE_REPLIES[0]
    for context in MASKS:
        router = _router(workflow, context)
        assert router.predict(value) == [PARAMETER_VALUE_LABEL]
        assert router.predict(vague) == [WILDCARD_LABEL]
    with pytest.raises(ValueError):
        _router(workflow, "Book").predict("   ")


def test_the_detector_contract_and_the_gates_read_the_layout_without_change(published_head):
    workflow, vdir, _fitted = published_head
    for context in MASKS:
        assert intent_detector.has_head(intent_detector.head_folder(workflow, context))
    assert not intent_detector.has_head(str(vdir / layout.WORKFLOW_MODEL_DIRNAME))
    engine = intent_detector.engine_for(workflow, "Book", allow_matcher=False)
    assert isinstance(engine, intent_detector.ClassifierEngine)
    # The folder holds only the published head (no _commands/), so the engine's filter of the commands that do not
    # load (RoutingDefinition.not_loading, which reads the workflow's routing) cannot run here: the head's answers are
    # read through the same engine with no workflow to filter against.
    engine = intent_detector.ClassifierEngine(engine.router)

    match = engine.decide("list the loans of this book")
    assert match.kind == intent_detector.MATCH and match.confident and match.labels == ["Book/list_loans"]
    assert intent_detector.head_is_confident(match)
    assert match.scores == dict(zip(match.details["topk_labels"], match.details["topk_scores"]))
    handed = engine.decide("which branches does the library have")
    assert handed.kind == intent_detector.AMBIGUOUS and not intent_detector.head_is_confident(handed)
    vague = engine.decide(heads.BUILT_IN_VAGUE_REPLIES[0])
    assert vague.kind == intent_detector.NOT_MINE and not intent_detector.head_is_confident(vague)
    assert engine.decide(sorted(PARAMETER_VALUE_PLACEHOLDERS)[0]).kind == intent_detector.BARE_VALUE

    assert sorted(heads.warm_intent_heads(workflow)) == ["Book", "Library", "global"]
    from fastworkflow._workflows.command_metadata_extraction import intent_detection

    assert intent_detection._classifier_signal_version(
        f"{workflow}/___command_info/*", match.details["model_tier"]) == "intent-classifier/v1/global/embedding"


def test_one_encoder_is_loaded_per_model_id_and_every_context_shares_the_head(published_head, tmp_path, encoder):
    workflow, vdir, _fitted = published_head
    routers = [_router(workflow, context) for context in MASKS]
    assert len({id(router.modelpipeline) for router in routers}) == 1
    assert routers[0].modelpipeline.encoder is encoder
    assert eh.SentenceEncoder.get(layout.DEFAULT_ENCODER) is encoder

    # Another workflow's head, in another folder: its own head, the same loaded encoder.
    other = eh.fit_embedding_head(_rows(), MASKS, str(tmp_path / "workflow"))
    other_router = eh.EmbeddingHeadRouter(other.model_dir, "Book")
    assert other_router.modelpipeline is not routers[0].modelpipeline
    assert other_router.modelpipeline.encoder is encoder
    # Another pooling or prefix of the same model id reads the same loaded weights.
    assert eh.SentenceEncoder.get(layout.DEFAULT_ENCODER, "cls").model is encoder.model
    assert eh.SentenceEncoder.get(layout.DEFAULT_ENCODER, "mean", "").model is encoder.model


def test_cls_pooling_differs_from_mean_and_the_prefix_is_put_before_every_text(encoder):
    cls = eh.SentenceEncoder.get(layout.DEFAULT_ENCODER, "cls")
    bare = eh.SentenceEncoder.get(layout.DEFAULT_ENCODER, "mean", "")
    texts = ["list the loans of this book", "a much longer request about the opening hours of the library"]
    mean_vectors, cls_vectors = encoder.embed(texts), cls.embed(texts)
    assert encoder.pooling == "mean" and encoder.prefix == "query: "
    assert np.allclose(np.linalg.norm(mean_vectors, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(cls_vectors, mean_vectors, atol=1e-3)
    assert not np.allclose(bare.embed(texts), mean_vectors, atol=1e-3)          # the prefix changes the vector
    # Padding does not change a text's vector: the mean is over the attention mask.
    assert np.allclose(encoder.embed(texts[:1])[0], mean_vectors[0], atol=1e-4)
    with pytest.raises(ValueError):
        eh.SentenceEncoder.get(layout.DEFAULT_ENCODER, "max")


# ---------------------------------------------------------------- the training report's section

def test_the_training_report_reads_the_heads_cross_fit_and_lists_the_confused_pairs(published_head):
    from fastworkflow.train import training_report as tr

    workflow, _vdir, fitted = published_head
    problems: list[str] = []
    model = tr.load_intent_model(workflow, "v1", problems)

    assert problems == [] and model is not None
    cross_fit = fitted.payload["cross_fit"]
    assert model.layout == "embedding" and model.encoder == "intfloat/e5-base-v2"
    assert model.commit_threshold == 0.65 and model.rows == cross_fit["rows"]
    assert model.accuracy == cross_fit["accuracy"] and model.folds == 5
    assert model.pairs_over_limit == cross_fit["pairs_over_limit"] and len(model.pairs) <= 10
    # The current version is read when none is named.
    assert tr.load_intent_model(workflow, None, []) is not None
    assert tr.load_intent_model(str(os.path.join(workflow, "nothing-here")), None, []) is None

    pair = tr.ConfusedPair(a="Book/list_loans", b="Book/show_book", a_as_b=3, a_rows=10, a_share=0.3,
                           b_as_a=1, b_rows=10, b_share=0.1, total=4, max_share=0.3)
    lines = tr._intent_model_lines(model.model_copy(update={
        "accuracy": 0.9, "rows": 30, "confused_pairs": 1, "pairs_over_limit": 1, "pairs": [pair]}))
    text = "\n".join(lines)
    assert "cross-fit accuracy: 90.0% of 30 distinct command rows" in text
    assert "1 in all, 1 with more than 10%" in text
    assert "Book/list_loans -> Book/show_book: 3/10 (30.0%); Book/show_book -> Book/list_loans: 1/10 (10.0%)" in text
    assert "add seed utterances that tell the two commands apart" in text and "merge the two commands" in text
    assert tr._intent_model_lines(None) == []
    assert "none: the head gave no command's rows to another command" in "\n".join(
        tr._intent_model_lines(model.model_copy(update={"pairs": [], "confused_pairs": 0})))
