"""A draft's verify checks run beside its parameter extraction (fix-w07e).

``parameter_extraction.extract_arguments`` (the sub-task draft path) starts ``check_verdicts`` in a worker
thread, in a copy of the caller's context, just before the first extraction when a Jev LM is set and the
command has a ``verify`` parameter, and validation reads the verdicts where it would have called
``check_verdicts``. The checks read only the Jev LM, the static command and the request text, so the
outcome is the sequential one; the retry of a planning command reuses the verdicts (it validates against
the same text).

Integration: a copy of ``tests/planning_tutorial_workflow`` whose ``book_loans`` (a planning command) and
``find_member`` (an execution command) gain a ``classifier_confirms`` bool, the real routing, the real DSPy
extractor and the real checks. The model boundaries are a scripted extractor ``DummyLM`` and the spy that
replaces ``planning_commands._check_one`` (the classifier call), as the existing check tests do.
"""
from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import planning_commands as pc
from fastworkflow import workflow_agent
from fastworkflow._workflows.command_metadata_extraction import parameter_extraction
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import extract_arguments
from fastworkflow.utils import signatures

TUTORIAL = Path(__file__).parent / "planning_tutorial_workflow"
WAIT = 20
FLAG = ('        flagged: bool = Field(default=False, description="the request asks for the loans",\n'
        '                              json_schema_extra={"classifier_confirms": True})\n')
REQUEST = "who has Foundation on loan"


class _GatedLM(DummyLM):
    """The extractor: answers in order; ``hook`` (a class attribute, so the copies the extractor makes share it) runs first."""
    hook = None

    def __init__(self, answers):
        super().__init__(answers, adapter=dspy.JSONAdapter())

    def forward(self, prompt=None, messages=None, **kwargs):
        if _GatedLM.hook is not None:
            _GatedLM.hook()
        return super().forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture
def wf(tmp_path, monkeypatch):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "off")
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    folder = tmp_path / "tutorial"
    shutil.copytree(TUTORIAL, folder, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
    for name in ("book_loans", "find_member"):
        path = folder / "_commands" / f"{name}.py"
        text = path.read_text()
        marker = "    class Input(BaseModel):\n"
        assert marker in text
        path.write_text(text.replace(marker, marker + FLAG, 1))
    fastworkflow.RoutingRegistry.get_definition(str(folder), load_cached=False)
    workflow = fastworkflow.Workflow.create(str(folder), workflow_id_str=f"eager_{uuid.uuid4().hex[:8]}")
    yield workflow
    _GatedLM.hook = None
    fastworkflow.RoutingRegistry.clear_registry()


def _script(monkeypatch, *answers) -> _GatedLM:
    lm = _GatedLM([{"reasoning": "scripted", **a} for a in answers])
    monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: lm)
    return lm


class Spy:
    """Stands for the classifier call: records (param, thread) and answers ``flagged`` true and sure."""

    def __init__(self, monkeypatch, jev=True):
        self.calls: list[tuple[str, threading.Thread]] = []
        self.started = threading.Event()
        self.pause = None
        monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: object() if jev else None)
        monkeypatch.setattr(pc, "_check_one", self.check_one)

    def check_one(self, lm, command, p, request):
        self.calls.append((p.name, threading.current_thread()))
        self.started.set()
        if self.pause is not None:
            self.pause()
        return True, 0.99


def _run(wf, command="book_loans", text=REQUEST):
    diagnostics: dict = {}
    result = extract_arguments(wf, command, text, diagnostics)
    return result, diagnostics


def _sequential(monkeypatch):
    monkeypatch.setattr(parameter_extraction, "_start_argument_checks", lambda *a: (None, None))


def _same(a, b):
    ra, da = a
    rb, db = b
    assert (ra.valid, ra.refusal, ra.missing, ra.retried) == (rb.valid, rb.refusal, rb.missing, rb.retried)
    assert ra.params.model_dump() == rb.params.model_dump()
    assert da.get("argument_notes") == db.get("argument_notes")
    assert da.get("argument_refusal") == db.get("argument_refusal")


def test_the_checks_start_before_the_extraction_finishes_and_the_outcome_is_the_sequential_one(wf, monkeypatch):
    answers = [{"title": "Foundation", "flagged": False}]
    spy = Spy(monkeypatch)
    saw_check_start = []

    def extractor_waits_for_the_check():
        saw_check_start.append(spy.started.wait(WAIT))

    _script(monkeypatch, *answers)
    _GatedLM.hook = extractor_waits_for_the_check
    eager = _run(wf)
    _GatedLM.hook = None

    assert saw_check_start and all(saw_check_start), "the checks had not started when the extractor ran"
    assert [name for name, _ in spy.calls] == ["flagged"]
    assert spy.calls[0][1] is not threading.main_thread() and spy.calls[0][1] is not threading.current_thread()
    assert eager[0].valid and eager[0].params.flagged is True and eager[1]["argument_notes"]

    spy.calls.clear()
    _sequential(monkeypatch)
    _script(monkeypatch, *answers)
    _same(eager, _run(wf))
    assert [name for name, _ in spy.calls] == ["flagged"]       # one call either way


def test_the_checks_run_once_per_draft_and_once_in_all_when_the_planning_command_retries(wf, monkeypatch):
    spy = Spy(monkeypatch)
    _script(monkeypatch, {"title": "Foundation", "flagged": False})
    result, _ = _run(wf)
    assert result.valid and not result.retried and [n for n, _ in spy.calls] == ["flagged"]

    spy.calls.clear()
    _script(monkeypatch, {"title": "Dune Messiah", "flagged": False}, {"title": "Foundation", "flagged": False})
    result, _ = _run(wf)
    assert result.retried and result.valid
    assert [n for n, _ in spy.calls] == ["flagged"]             # the retry reuses the verdicts

    spy.calls.clear()
    _sequential(monkeypatch)
    _script(monkeypatch, {"title": "Dune Messiah", "flagged": False}, {"title": "Foundation", "flagged": False})
    assert _run(wf)[0].retried
    assert [n for n, _ in spy.calls] == ["flagged", "flagged"]  # what it was: one call set per validation


def test_an_execution_command_starts_its_checks_early_too(wf, monkeypatch):
    spy = Spy(monkeypatch)
    _script(monkeypatch, {"query": "Ann Lee", "flagged": False})
    result, _ = _run(wf, "find_member", "find the member Ann Lee")
    assert result.valid and result.params.flagged is True
    assert [n for n, _ in spy.calls] == ["flagged"]


def test_no_jev_lm_or_no_verify_param_starts_nothing(wf, monkeypatch):
    starts = []
    real = pc.check_verdicts

    def counting(lm, command, request):
        starts.append(threading.current_thread())
        return real(lm, command, request)

    monkeypatch.setattr(pc, "check_verdicts", counting)

    spy = Spy(monkeypatch, jev=False)                           # no Jev key
    _script(monkeypatch, {"title": "Foundation", "flagged": False})
    result, _ = _run(wf)
    assert parameter_extraction._start_argument_checks(wf, signatures.InputForParamExtraction.create(
        wf, "book_loans", REQUEST), "book_loans", type(result.params), REQUEST) == (None, None)
    assert result.valid and result.params.flagged is False and spy.calls == []
    assert starts == [threading.current_thread()]               # validation's own call, in this thread

    starts.clear()
    spy = Spy(monkeypatch, jev=True)                            # a Jev LM, a command with no verify param
    _script(monkeypatch, {"author": "Asimov"})
    result, _ = _run(wf, "author_books", "show every book by Asimov")
    assert result.valid and spy.calls == []
    assert starts == [threading.current_thread()]


_THREADS_BEFORE: set = set()


@pytest.fixture(autouse=True)
def _snapshot_threads():
    """Threads alive before the test: an earlier file's pool worker is not this check's to join."""
    _THREADS_BEFORE.clear()
    _THREADS_BEFORE.update(threading.enumerate())
    yield
    _THREADS_BEFORE.clear()


def _threads_started_by_this_test() -> list:
    return [t for t in threading.enumerate()
            if t.name.startswith("ThreadPoolExecutor") and t not in _THREADS_BEFORE]


def test_an_extraction_that_raises_raises_the_same_error_and_the_check_thread_is_joined(wf, monkeypatch):
    spy = Spy(monkeypatch)
    finished = threading.Event()

    def slow_check():
        spy.started.wait(WAIT)
        threading.Event().wait(0.3)
        finished.set()

    spy.pause = slow_check

    def broken(*args, **kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(signatures.dspy_utils, "get_lm", broken)
    with pytest.raises(RuntimeError, match="no model"):
        _run(wf)
    assert finished.is_set(), "extract_arguments returned while the check was still running"
    assert not _threads_started_by_this_test()


def test_a_check_that_raises_surfaces_where_validation_reads_it(wf, monkeypatch):
    Spy(monkeypatch)

    def down(lm, command, request):
        raise RuntimeError("classifier down")

    monkeypatch.setattr(pc, "check_verdicts", down)
    _script(monkeypatch, {"title": "Foundation", "flagged": False})
    with pytest.raises(RuntimeError, match="classifier down"):
        _run(wf)
    assert not _threads_started_by_this_test()


def test_validation_that_stops_before_the_check_stage_discards_the_verdicts(wf, monkeypatch):
    spy = Spy(monkeypatch)
    _script(monkeypatch, {"title": "NOT_FOUND", "flagged": False}, {"title": "NOT_FOUND", "flagged": False})
    result, _ = _run(wf)
    assert not result.valid and result.missing == ["title"]
    assert not _threads_started_by_this_test()
    assert len(spy.calls) <= 1                                  # started early at most once, then dropped


def test_a_command_that_cannot_be_resolved_starts_nothing_and_raises_where_validation_raised(wf, monkeypatch):
    Spy(monkeypatch)

    def broken(*args, **kwargs):
        raise LookupError("no such command")

    extractor = type("Stub", (), {"checked_command": broken})()
    assert parameter_extraction._start_argument_checks(
        wf, extractor, "book_loans", object, REQUEST) == (None, None)

    # through extract_arguments: nothing is started, and the error is validation's own
    monkeypatch.setattr(signatures.InputForParamExtraction, "checked_command", broken)
    _script(monkeypatch, {"title": "Foundation", "flagged": False})
    with pytest.raises(LookupError, match="no such command"):
        _run(wf)
    assert not _threads_started_by_this_test()
