"""The public framework reads a training-data generator needs (bd fix-0iuo.23).

- ``signatures.extraction_signature`` / ``extraction_messages`` / ``extraction_answer``: the parameter extractor's
  prompt and answer format as data, built by the calls ``extract_parameters`` makes, so pairs written in the
  extractor's format cannot drift from what the runtime sends and parses.
- ``workflow_agent.split_examples``: the splitter's examples as data.
- ``observability.store.REDACTED``: the text the store writes for a scrubbed credential.
- ``ObservabilityStore.spans_for_turns(name=)``: the spans of given turns, of one name.
- ``ObservabilityStore.count_spans_without_turn(name=)``: how many spans (of one name) belong to no turn row.

Real example workflows, real adapters and a real store; no model is called.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import ModuleType
from fastworkflow.planning import planning_commands as pc
from fastworkflow.observability import tracing
from fastworkflow.core.command_routing import RoutingRegistry
from fastworkflow.observability import store as obs
from fastworkflow.parameters import signatures as S

import dspy  # noqa: E402  (after fastworkflow: litellm must be imported before dspy)

ROOT = Path(__file__).resolve().parent.parent
RETAIL = str(ROOT / "fastworkflow" / "examples" / "retail_workflow")
LIBRARY = str(ROOT / "fastworkflow" / "examples" / "library")


def _input(folder: str, command: str):
    return RoutingRegistry.get_definition(folder).get_command_class(
        command, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS).Input


class LoanInput(BaseModel):
    """Unusual shapes: an int, an optional float, a bool, a Literal of text, a non-ASCII default, an alone flag."""
    member: str = Field(description="the member's full name")
    copies: int = Field(default=1, description="how many copies")
    fine: Optional[float] = Field(default=None, description="the fine to charge")
    renew: bool = Field(default=False, description="the request asks to renew the loan")
    branch: Literal["Nord", "Süd", "O'Hare / East, Annex"] = Field(default="Nord", description="the branch")
    note: str = Field(default="ä, \"quoted\"", description="a note")
    quiet: bool = Field(default=False, description="the request asks not to notify",
                        json_schema_extra={"classifier_decides": True})


def _runtime_prompt(model, demos, text):
    """What ``extract_parameters`` composes: ChainOfThought over the signature built from the field declarations,
    ``LabeledFewShot`` with k = the number of demos, and the extractor's adapter."""
    signature = S.InputForParamExtraction.create_signature_from_pydantic_model(
        model, pc.option_sentences(model), pc.classifier_alone_fields(model))

    class Extractor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought(signature)

    predict = dspy.LabeledFewShot(k=len(demos)).compile(student=Extractor(), trainset=list(demos)).predictor.predict
    return predict, S._TextLiteralsAsText().format(predict.signature, list(predict.demos), {"statement": text})


def _demo(text: str, **values) -> "dspy.Example":
    return dspy.Example(statement=text, **values).with_inputs("statement")


# ---------------------------------------------------------------- the extractor's prompt

def test_the_signature_is_the_one_the_runtime_builds():
    for model in (_input(RETAIL, "modify_pending_order_address"), LoanInput):
        built = S.extraction_signature(model)
        expected = S.InputForParamExtraction.create_signature_from_pydantic_model(
            model, pc.option_sentences(model), pc.classifier_alone_fields(model))
        assert list(built.output_fields) == list(expected.output_fields)
        assert [f.json_schema_extra for f in built.output_fields.values()] == \
            [f.json_schema_extra for f in expected.output_fields.values()]
        assert built.instructions == expected.instructions
    assert "quiet" not in S.extraction_signature(LoanInput).output_fields


@pytest.mark.parametrize("folder, command, text", [
    (RETAIL, "cancel_pending_order", "Cancel order #W2219034, I ordered it by mistake."),
    (LIBRARY, "Library/book_by_title", "Do you have « Cent ans de solitude », the 1967 one?"),
])
def test_the_messages_are_the_runtime_prompt_with_and_without_demos(folder, command, text):
    fastworkflow.init({})
    model = _input(folder, command)
    demos = [_demo("first example, with a comma / slash", **{n: "x" for n in S.extraction_signature(model)
                                                                .output_fields})]
    for given in ([], demos):
        _predict, expected = _runtime_prompt(model, given, text)
        assert S.extraction_messages(model, command, folder, text, given) == expected
    messages = S.extraction_messages(model, command, folder, text, demos)
    assert messages[-1]["role"] == "user" and text in messages[-1]["content"]
    assert len(messages) == 2 + 2 * len(demos)          # system, a user/assistant pair per demo, the request
    assert "first example, with a comma / slash" in messages[1]["content"]
    # None reads the command's own labelled examples, as the runtime does
    trainset = S.get_trainset(command, folder)
    assert S.extraction_messages(model, command, folder, text) == _runtime_prompt(model, trainset, text)[1]


def test_the_answer_is_read_back_by_the_extractors_adapter():
    predict, _ = _runtime_prompt(LoanInput, [], "renew Zoë Ångström's loan at O'Hare / East, Annex")
    values = {"member": "Zoë Ångström", "renew": True, "branch": "O'Hare / East, Annex", "copies": 2}
    answer = S.extraction_answer(LoanInput, values)
    parsed = S._TextLiteralsAsText().parse(predict.signature, answer)
    assert parsed["reasoning"] == ""
    assert {k: parsed[k] for k in values} == values
    # an unnamed output holds what the prompt says for an unmentioned argument (the output field's default)
    defaults = {n: f.default for n, f in S.extraction_signature(LoanInput).output_fields.items()}
    for name in ("fine", "note"):
        assert parsed[name] == defaults[name]
    assert parsed["note"] == "ä, \"quoted\""


def test_an_answer_naming_what_the_extractor_is_not_asked_for_raises():
    with pytest.raises(KeyError, match="quiet"):
        S.extraction_answer(LoanInput, {"quiet": True})
    with pytest.raises(KeyError, match="nickname"):
        S.extraction_answer(_input(RETAIL, "cancel_pending_order"), {"nickname": "x"})


def test_an_input_with_no_fields_has_an_answer_of_its_reasoning_only():
    class Empty(BaseModel):
        pass

    assert S.extraction_answer(Empty, {}) == S._TextLiteralsAsText().format_assistant_message_content(
        _runtime_prompt(Empty, [], "x")[0].signature, {"reasoning": ""})


# ---------------------------------------------------------------- the splitter's examples

def _split_file(folder: Path, text: str) -> str:
    (folder / "_commands").mkdir(parents=True, exist_ok=True)
    from tests.workflow_settings_file import write_workflow_file

    write_workflow_file(folder / "_commands", text)
    return str(folder)


def test_the_default_splitter_examples_as_data():
    from fastworkflow.workflow_agent import SplitIntoSubtasks, split_examples

    examples = split_examples(None)
    assert len(examples) == 3
    assert all(e.user_query and e.subtasks for e in examples)
    for example in examples:
        for text in (example.user_query, *example.subtasks):
            assert text in SplitIntoSubtasks.instructions
    assert examples[1].subtasks == (examples[1].user_query,)       # one request is returned as one sub-task
    assert split_examples(None, workflow_only=True) == ()


def test_a_workflows_own_examples_replace_the_defaults(tmp_path):
    from fastworkflow.workflow_agent import SplitExample, split_examples

    folder = _split_file(tmp_path / "shop", '''
split_examples = [
    ('Rückgabe für Bestellung #W1 und Adresse ändern: "Hauptstraße 5, 2/3"',
     ["Rückgabe für Bestellung #W1", 'Adresse ändern: "Hauptstraße 5, 2/3"']),
    ("List every product type.", ["List every product type."]),
]
''')
    assert split_examples(folder) == split_examples(folder, workflow_only=True) == (
        SplitExample('Rückgabe für Bestellung #W1 und Adresse ändern: "Hauptstraße 5, 2/3"',
                     ("Rückgabe für Bestellung #W1", 'Adresse ändern: "Hauptstraße 5, 2/3"')),
        SplitExample("List every product type.", ("List every product type.",)),
    )


def test_a_split_file_without_instructions_has_no_examples_of_its_own(tmp_path):
    from fastworkflow.workflow_agent import split_examples

    folder = _split_file(tmp_path / "library", "import re\nidentifier_patterns = [re.compile(r'\\bM-[0-9]+')]\n")
    assert split_examples(folder, workflow_only=True) == ()
    assert split_examples(folder) == split_examples(None)
    assert split_examples(str(tmp_path / "no_split_file"), workflow_only=True) == ()


# ---------------------------------------------------------------- the store

def test_the_redaction_text_is_public_and_is_what_the_store_writes():
    assert obs.REDACTED == obs._REDACTED and obs.REDACTED
    assert obs.Redactor().redact("key sk-" + "a" * 24 + " end") == f"key {obs.REDACTED} end"


@pytest.fixture
def two_kinds_of_span(tmp_path):
    db = str(tmp_path / "observability.sqlite3")
    sink = obs.SQLiteTraceSink(db)
    keys = [f"20261123T{index:06d}-turn" for index in range(3)]
    for turn, key in enumerate(keys):
        for index, name in enumerate((tracing.SPAN_TURN, tracing.SPAN_COMMAND, "fw.llm.call", tracing.SPAN_COMMAND)):
            sink.emit_span(tracing.Span(span_id=f"{key}-s{index}", trace_id=key, name=name,
                                        start_ns=10_000 + turn * 100 + index, status="ok",
                                        attributes={"seq": index}))
    assert sink.flush()
    sink.close()
    return db, keys


def test_spans_for_turns_reads_one_span_name(two_kinds_of_span):
    db, keys = two_kinds_of_span
    store = obs.ReadOnlyObservabilityStore(db)
    named = store.spans_for_turns([*keys, "never-recorded", keys[0]], name=tracing.SPAN_COMMAND)
    assert list(named) == [*keys, "never-recorded"]
    assert named["never-recorded"] == []
    for key in keys:
        expected = [row for row in store.get_spans(key) if row["name"] == tracing.SPAN_COMMAND]
        assert named[key] == expected and [row["span_id"] for row in expected] == [f"{key}-s1", f"{key}-s3"]
    assert store.spans_for_turns(keys, name="no.such.span") == {key: [] for key in keys}
    # without a name: every span, as before
    assert store.spans_for_turns(keys) == {key: store.get_spans(key) for key in keys}


def test_spans_for_turns_by_name_is_chunked(two_kinds_of_span):
    db, keys = two_kinds_of_span
    store = obs.ReadOnlyObservabilityStore(db)
    padding = [f"absent-{index}" for index in range(1000)]
    answer = store.spans_for_turns(keys + padding, name="fw.llm.call")
    assert len(answer) == 1003 and all(len(answer[key]) == 1 for key in keys)


def _record_turn(db: str, key: str) -> None:
    """A real turn row for *key*, written through the store's own turn-row writer."""
    import json  # noqa: PLC0415
    writer = obs.ObservabilityStore(db)
    row = {"turn_key": key, "channel_id": "chat", "conversation_id": 1, "ordinal": 1,
           "user_message": "a request", "refined_user_message": None, "entry_workflow_name": "wf",
           "entry_context": "*", "status": "completed", "success": 1, "failure_reason": None,
           "answer": "done", "conversation_summary": None, "conversation_traces": None,
           "started_at": "2026-11-23T00:00:00+00:00", "completed_at": "2026-11-23T00:00:01+00:00",
           "suspended_ms": 0, "continuation_of": None, "record_version": 1, "experiment_id": None,
           "task_id": None, "attempt": None, "claim_epoch": None, "server_incarnation": None,
           "record_json": json.dumps({"turn_output": {"turn_key": key, "success": True}})}
    with writer._connect() as conn:
        assert writer.upsert_turn_row(conn, row, [], writer._store_redactor())
        conn.commit()


def test_count_spans_without_turn_counts_only_spans_no_turn_row_names(two_kinds_of_span):
    db, keys = two_kinds_of_span
    store = obs.ReadOnlyObservabilityStore(db)
    # no turn rows yet: every span belongs to no turn
    assert store.count_spans_without_turn() == 12
    assert store.count_spans_without_turn(name=tracing.SPAN_COMMAND) == 6
    _record_turn(db, keys[0])
    assert store.get_turn(keys[0]) is not None
    assert store.count_spans_without_turn() == 8
    assert store.count_spans_without_turn(name=tracing.SPAN_COMMAND) == 4
    assert store.count_spans_without_turn(name="fw.llm.call") == 2
    assert store.count_spans_without_turn(name="no.such.span") == 0
    for key in keys[1:]:
        _record_turn(db, key)
    assert store.count_spans_without_turn() == 0
    assert store.count_spans_without_turn(name=tracing.SPAN_COMMAND) == 0


def test_count_spans_without_turn_on_an_empty_store(tmp_path):
    db = str(tmp_path / "observability.sqlite3")
    obs.ObservabilityStore(db)
    store = obs.ReadOnlyObservabilityStore(db)
    assert store.count_spans_without_turn() == 0
    assert store.count_spans_without_turn(name=tracing.SPAN_COMMAND) == 0
