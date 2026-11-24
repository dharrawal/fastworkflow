"""A value equal to one of a field's ``examples`` is an ordinary value (bd fix-qjne.21 follow-up).

The examples are not shown to the extraction model and the description-example check is gone, so the two leftovers
that still treated such a value specially are gone too: the extractor's reward step (one more attempt for a value equal to
an example) and the readers of an agent's own typed text (``fill.from_xml``, ``fill.from_key_values``,
``fill.from_positional``: the value was "not given", the form fell through to the model). Real DSPy and pydantic objects;
the model boundary is a scripted ``DummyLM``.
"""
from __future__ import annotations

import dspy
import pytest
from dspy.utils import DummyLM
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.parameters import fill
from fastworkflow.utils import signatures


class Note(BaseModel):
    text: str = Field(description="the note to add", examples=["the cover is torn"])


class Marked(BaseModel):
    text: str = Field(description="the note to add", examples=["the cover is torn"],
                      json_schema_extra={"command_text_after_command_name": True})


class Pair(BaseModel):
    text: str = Field(description="the note to add", examples=["the cover is torn"])
    kind: str = Field(default="", description="the kind", examples=["loan"])


class _ScriptedLM(DummyLM):
    def __init__(self, answers):
        super().__init__(answers, adapter=dspy.JSONAdapter())
        self.attempts = []   # shared with the copies BestOfN makes

    @property
    def calls(self):
        return len(self.attempts)

    def copy(self, **kwargs):
        clone = super().copy(**kwargs)
        clone.attempts = self.attempts
        return clone

    def forward(self, prompt=None, messages=None, **kwargs):
        self.attempts.append(1)
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture
def extractor(tmp_path, monkeypatch):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "off")
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")

    def run(request, *answers):
        lm = _ScriptedLM(list(answers))
        monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
        found = signatures.InputForParamExtraction(command=request).extract_parameters(
            Note, "add_note", str(tmp_path))
        return found, lm
    return run


# ---------------------------------------------------------------- the extractor's reward step

def test_an_extracted_value_equal_to_an_example_is_one_attempt_like_any_other(extractor):
    found, lm = extractor("add a note that the cover is torn",
                          {"reasoning": "r", "text": "the cover is torn"},
                          {"reasoning": "r", "text": "OTHER"})
    assert found.text == "the cover is torn"
    assert lm.calls == 1


def test_an_extracted_value_not_equal_to_an_example_is_one_attempt(extractor):
    found, lm = extractor("add a note that the spine is cracked", {"reasoning": "r", "text": "the spine is cracked"})
    assert found.text == "the spine is cracked" and lm.calls == 1


# ---------------------------------------------------------------- an agent's own typed text

def test_from_xml_takes_a_value_equal_to_an_example():
    bound = fill.from_xml("<text>the cover is torn</text>", Note)
    assert bound is not None and bound.text == "the cover is torn"


def test_from_xml_with_two_fields_takes_an_example_in_either():
    bound = fill.from_xml("<text>spine</text><kind>loan</kind>", Pair)
    assert bound is not None and (bound.text, bound.kind) == ("spine", "loan")


def test_from_key_values_takes_a_value_equal_to_an_example():
    bound = fill.from_key_values('add_note text="the cover is torn"', Note, "add_note")
    assert bound is not None and bound.text == "the cover is torn"
    bound = fill.from_key_values("add_note text=the cover is torn", Note, "add_note")
    assert bound is not None and bound.text == "the cover is torn"


def test_from_positional_takes_a_value_equal_to_an_example():
    bound = fill.from_positional("add_note the cover is torn", Marked, "add_note")
    assert bound is not None and bound.text == "the cover is torn"
    bound = fill.from_positional("add_note torn", Note, "add_note")
    assert bound is not None and bound.text == "torn"
    Note.model_fields["text"].examples = ["torn"]
    try:
        bound = fill.from_positional("add_note torn", Note, "add_note")
    finally:
        Note.model_fields["text"].examples = ["the cover is torn"]
    assert bound is not None and bound.text == "torn"
