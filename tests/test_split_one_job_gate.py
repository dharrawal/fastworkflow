"""The one-job gate in front of the request splitter (bd fix-qjne.26).

When the remote decision model is configured, ``workflow_agent.split_into_subtasks`` first asks it one question: is the
request one job or several, given the same job list the splitter gets. A confident "one" (at or above
``FW_SPLIT_GATE_THRESHOLD``, default 0.85) runs the request as one part and the splitter is not called; anything else runs
the splitter as before. ``off`` turns the gate off; with no remote model, or a failed call, nothing changes.

Real components: the remote model's answer is scripted at ``dspy.Predict`` (as the matcher and SameRequest tests script
it), the splitter's model is the scripted ``DummyLM`` the neighbouring split tests use, so the real merge rules run.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow.intent import remote_outage
from fastworkflow.agent import workflow_agent
from fastworkflow.observability import tracing
from fastworkflow.core.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401
    _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_plan_reply_policy import _wire_plan, planner, script_param_extraction  # noqa: F401
from tests.test_round1_gate import gated, no_draft_extraction  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests import span_tree

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

LIBRARY = "fastworkflow/examples/library"
TUTORIAL = "tests/planning_tutorial_workflow"
ONE = "Tell Ann a joke."
SEVERAL = "Find the book Dune. Find the book Emma."
SEVERAL_PARTS = ["Find the book Dune.", "Find the book Emma."]
#: The real split: the ``todo`` fixture the gate tests build on replaces it with "the request whole".
REAL_SPLIT = workflow_agent.split_into_subtasks


class _Splitter(DummyLM):
    """The splitter's scripted model. The splitter calls copies of it (``workflow_agent._split_lm``); ``calls`` is the
    history of every model call made on it and its copies."""

    def copy(self, **kwargs):
        copied = super().copy(**kwargs)
        self.__dict__.setdefault("copies", []).append(copied)
        return copied

    @property
    def calls(self) -> list:
        return [entry for lm in [self, *self.__dict__.get("copies", [])] for entry in lm.history]


def _splitter(items: list[str]) -> _Splitter:
    text = "\n".join(f"{n}. {item}" for n, item in enumerate(items, 1))
    return _Splitter([{"reasoning": "r", "subtasks": text}])


@pytest.fixture
def remote(monkeypatch):
    """Script the remote decision model's one-job answer: ``remote(answer)`` with ``answer`` = ``(label, p(one))`` or an
    exception to raise. Returns the inputs of each gate call. Any other remote question of the process answers nothing
    (SameRequest: no verdict, so the lexical rule; the referent: none), as in ``tests/test_text_planner_and_split.py``.
    The setting is cleared (``tests/conftest.py`` turns the gate off for every other test), and the outage counter
    starts and ends fresh, so the failures scripted here do not silence another test's outage notice."""
    remote_outage.reset()
    monkeypatch.setattr(fastworkflow, "_env_vars", {
        name: value for name, value in fastworkflow._env_vars.items()
        if name != workflow_agent.SPLIT_GATE_THRESHOLD_SETTING})
    monkeypatch.delenv(workflow_agent.SPLIT_GATE_THRESHOLD_SETTING, raising=False)
    monkeypatch.setattr(workflow_agent, "_choose_referent", lambda *args: None)

    def script(answer) -> list[dict]:
        calls: list[dict] = []
        real_predict = workflow_agent.dspy.Predict

        def predict_for(signature):
            if signature is not workflow_agent.OneOrSeveralJobs:
                if signature is workflow_agent.SameRequest:
                    def no_verdict(**inputs):
                        raise RuntimeError("not scripted")
                    return no_verdict
                return real_predict(signature)

            def predict(**inputs):
                calls.append(inputs)
                if isinstance(answer, Exception):
                    raise answer
                label, p_one = answer
                probabilities = {"one": p_one, "several": 1 - p_one}
                return SimpleNamespace(jobs=SimpleNamespace(
                    value=label, confidence=probabilities[label], probabilities=probabilities))
            return predict

        monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: DummyLM([]))
        monkeypatch.setattr(workflow_agent.dspy, "Predict", predict_for)
        return calls

    yield script
    remote_outage.reset()


def _set(monkeypatch, value: str) -> None:
    monkeypatch.setitem(fastworkflow._env_vars, workflow_agent.SPLIT_GATE_THRESHOLD_SETTING, value)


def test_a_confident_one_skips_the_splitter_and_records_the_gate(remote):
    calls = remote(("one", 0.97))
    splitter = _splitter(SEVERAL_PARTS)
    with workflow_agent.recording_split() as record:
        out = workflow_agent.split_into_subtasks(ONE, planner_lm=splitter, workflow_folderpath=LIBRARY)
    assert out == [ONE]
    assert splitter.calls == []  # the splitter's model was never called
    assert len(calls) == 1
    assert record == {"request": ONE, "raw": [], "merged": [ONE], "attempts": 0,
                      "gate": {"score": 0.97, "label": "one", "threshold": 0.85, "skipped": True}}
    # on the split span: no splitter prompt was used, so no prompt values; the gate's verdict is its own span's
    recorded = tracing.split_stage_attributes(record)
    assert "gate" not in recorded and recorded["prompt_digest"] is None and recorded["prompt_source"] is None
    json.dumps(recorded)


def test_the_gate_is_asked_with_the_request_and_the_job_list_the_splitter_gets(remote):
    calls = remote(("one", 0.99))
    workflow_agent.split_into_subtasks(ONE, planner_lm=_splitter([ONE]), workflow_folderpath=TUTORIAL)
    job_types = workflow_agent._job_types_for(TUTORIAL)
    assert job_types != "(none)"
    assert calls == [{"request": ONE, "job_types": job_types}]


def test_a_workflow_with_no_planning_commands_asks_with_what_the_splitter_gets(remote):
    calls = remote(("one", 0.99))
    workflow_agent.split_into_subtasks(ONE, planner_lm=_splitter([ONE]), workflow_folderpath=None)
    assert calls == [{"request": ONE, "job_types": "(none)"}]


def test_one_below_the_bar_runs_the_splitter(remote):
    remote(("one", 0.84))
    splitter = _splitter(SEVERAL_PARTS)
    with workflow_agent.recording_split() as record:
        out = workflow_agent.split_into_subtasks(SEVERAL, planner_lm=splitter, workflow_folderpath=LIBRARY)
    assert out == SEVERAL_PARTS and len(splitter.calls) == 1
    assert record["gate"] == {"score": 0.84, "label": "one", "threshold": 0.85, "skipped": False}
    assert record["attempts"] == 1 and record["raw"] == SEVERAL_PARTS and "prompt_digest" in record


def test_the_default_bar_is_0_85_so_a_one_at_0_87_skips_the_splitter(remote):
    # the shipped default (bd fix-qjne.26 stage 24): requests scoring between 0.85 and 0.9 were all single-part
    assert workflow_agent.DEFAULT_SPLIT_GATE_THRESHOLD == 0.85
    remote(("one", 0.87))
    splitter = _splitter(SEVERAL_PARTS)
    with workflow_agent.recording_split() as record:
        out = workflow_agent.split_into_subtasks(ONE, planner_lm=splitter, workflow_folderpath=LIBRARY)
    assert out == [ONE] and splitter.calls == []
    assert record["gate"] == {"score": 0.87, "label": "one", "threshold": 0.85, "skipped": True}


def test_several_runs_the_splitter(remote):
    remote(("several", 0.02))
    splitter = _splitter(SEVERAL_PARTS)
    with workflow_agent.recording_split() as record:
        out = workflow_agent.split_into_subtasks(SEVERAL, planner_lm=splitter, workflow_folderpath=LIBRARY)
    assert out == SEVERAL_PARTS and len(splitter.calls) == 1
    assert record["gate"] == {"score": 0.02, "label": "several", "threshold": 0.85, "skipped": False}


def test_a_threshold_setting_moves_the_bar(remote, monkeypatch):
    remote(("one", 0.8))
    _set(monkeypatch, "0.75")
    splitter = _splitter(SEVERAL_PARTS)
    assert workflow_agent.split_into_subtasks(ONE, planner_lm=splitter, workflow_folderpath=LIBRARY) == [ONE]
    assert splitter.calls == []


def test_off_makes_no_remote_call(remote, monkeypatch):
    calls = remote(("one", 0.99))
    _set(monkeypatch, " OFF ")
    splitter = _splitter(SEVERAL_PARTS)
    with workflow_agent.recording_split() as record:
        out = workflow_agent.split_into_subtasks(SEVERAL, planner_lm=splitter, workflow_folderpath=LIBRARY)
    assert out == SEVERAL_PARTS and calls == [] and len(splitter.calls) == 1
    assert "gate" not in record and "gate" not in tracing.split_stage_attributes(record)


def test_no_remote_model_makes_no_remote_call(remote, monkeypatch, no_planner_key):
    calls = remote(("one", 0.99))
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: None)
    splitter = _splitter(SEVERAL_PARTS)
    with workflow_agent.recording_split() as record:
        out = workflow_agent.split_into_subtasks(SEVERAL, planner_lm=splitter, workflow_folderpath=LIBRARY)
    assert out == SEVERAL_PARTS and calls == [] and len(splitter.calls) == 1
    assert "gate" not in record


def test_an_invalid_threshold_warns_once_and_means_the_default(remote, monkeypatch, caplog):
    from fastworkflow.train import intent_model_layout

    monkeypatch.setattr(intent_model_layout, "_warned_values", set())
    remote(("one", 0.95))
    _set(monkeypatch, "1.5")
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            with workflow_agent.recording_split() as record:
                workflow_agent.split_into_subtasks(ONE, planner_lm=_splitter([ONE]), workflow_folderpath=LIBRARY)
            assert record["gate"]["threshold"] == 0.85 and record["gate"]["skipped"] is True
    warnings = [line for line in caplog.messages if "FW_SPLIT_GATE_THRESHOLD" in line]
    assert len(warnings) == 1 and "using 0.85" in warnings[0]


def test_a_failed_remote_call_is_reported_and_the_splitter_runs(remote, caplog):
    remote(RuntimeError("the remote model is down"))
    splitter = _splitter(SEVERAL_PARTS)
    with caplog.at_level(logging.WARNING), workflow_agent.recording_split() as record:
        out = workflow_agent.split_into_subtasks(SEVERAL, planner_lm=splitter, workflow_folderpath=LIBRARY)
    assert out == SEVERAL_PARTS and len(splitter.calls) == 1
    assert "gate" not in record and record["attempts"] == 1
    assert any("The remote decision model failed (split gate): RuntimeError" in line for line in caplog.messages)


def test_a_turn_the_gate_kept_whole_runs_one_part_and_records_the_gate(gated, monkeypatch, remote,
                                                                       no_draft_extraction):
    monkeypatch.setattr(workflow_agent, "split_into_subtasks", REAL_SPLIT)
    remote(("one", 0.99))
    splitter = _splitter(SEVERAL_PARTS)
    gated.ctx._current_planner_lm = splitter
    gated.script.steps = _finish() * 2
    out = gated.ctx.process_turn(ONE, unattended=True)
    assert out.status is TurnStatus.COMPLETED
    assert splitter.calls == []
    (split,) = span_tree.named(gated.sink, tracing.SPAN_SPLIT)
    recorded = split.attributes
    assert recorded["request"] == ONE and recorded["merged"] == [ONE] and recorded["raw"] == []
    assert recorded["attempts"] == 0 and recorded["prompt_digest"] is None
    # the gate is the split stage's own step, and the splitter never ran
    (gate,) = span_tree.children_of(split, gated.sink, tracing.SPAN_SPLIT_GATE)
    assert gate.attributes["skipped"] is True and gate.attributes["label"] == "one" and gate.attributes["score"] == 0.99
    assert gate.attributes["threshold"] == 0.85
    assert not span_tree.children_of(split, gated.sink, tracing.SPAN_SPLIT_SPLITTER)
