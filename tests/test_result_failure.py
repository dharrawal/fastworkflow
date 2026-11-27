"""A failing ``__result__`` fails the command; ``None`` is the one way to hand the answer to a model (bd fix-qjne.12).

An exception raised inside ``__result__``, an empty or whitespace-only string, or a value the command may not return
used to be swallowed: one warning, and one model call wrote the answer (runtime rule 3.16), so an author's bug read as a
fluent answer. Now the call fails as an execution command's call fails (``command_executor``): a failed output naming
the command and the error, ``fastworkflow.Refused`` a refusal with its message, and no model call. ``None`` is
unchanged. ``__result__`` also follows the one result rule ``command_executor.command_response_of`` applies to
``__call__`` (a command that declares ``Signature.Output`` returns that Output, exactly).

Real components: the library copy through the real phase runner (``library_env``), the nested harness, and ``run-code``.
The only seam is ``write_answer`` (the one model call that writes an answer), recorded so a test can prove it is never
made for a failing ``__result__``.
"""
from __future__ import annotations

import logging
import os
import shutil

import pytest

import fastworkflow
from fastworkflow.observability import tracing
from fastworkflow.planning import run, run_code
from fastworkflow.planning.planning_code_run import RunCodeError
from tests.test_answer_hook import (  # noqa: F401 - fixtures and helpers of the answer hook's tests
    Declared, Other, _command_with_declared_output, _fresh_command_caches, _outputs, _run, answered)
from tests.test_nested_planning_commands import (  # noqa: F401
    BOOK_BY_TITLE_FILE, _BOOK_BY_TITLE_SOURCE, _run_book_loans, nested_env)
from tests.test_step_helpers import FoundBooks, _build_planning_command, library_env  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow, no_review_gate  # noqa: F401
from tests.test_no_agent_mode import _plan
from fastworkflow.engine.workflow_execution_context import WorkflowExecutionContext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")


class _Sub(Declared):
    """A subclass of the declared Output."""

    extra: int = 0


def _raises(input, find):
    raise RuntimeError("boom")


def _empty_text(input, find):
    return ""


def _blank_text(input, find):
    return "  \n "


def _an_int(input, find):
    return 3


def _forgot_the_return(input, find):
    _ = len(find.book_uids)         # falls off the end: None


def _other_output(input, find):
    return Other(found=True)


def _refuses(input, find):
    raise fastworkflow.Refused("no book of that title is on the shelf")


def _text_for_a_declared_output(input, find):
    return "found it"


def _response_without_output(input, find):
    return fastworkflow.CommandResponse(response="found it")


def _subclass_directly(input, find):
    return _Sub(found=True, extra=1)


def _subclass_in_a_response(input, find):
    return fastworkflow.CommandResponse(response="found it", output=_Sub(found=True, extra=1))


def _plan_of(result_fn):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find):
            return result_fn(input, find)

    return Plan


# ----------------------------------------------------------------------
# The agent path: the call fails, no model writes anything
# ----------------------------------------------------------------------

@pytest.mark.parametrize("result_fn, fragment", [
    (_raises, "RuntimeError('boom')"),
    (_empty_text, "empty"),
    (_blank_text, "empty"),
    (_an_int, "int"),
    (_forgot_the_return, "returned None; it must return the answer, or fastworkflow.planning.ANSWER_BY_MODEL"),
], ids=["raises", "empty-str", "blank-str", "int", "none"])
def test_a_failing_result_fails_the_call_and_no_model_writes_the_answer(library_env, answered, caplog,
                                                                         result_fn, fragment):
    command = _build_planning_command(_plan_of(result_fn))
    with caplog.at_level(logging.WARNING):
        result = _run(library_env, command)

    assert answered == []                 # write_answer, the model call that used to hide the bug
    assert library_env.model.asked == []
    assert result.answer != "written"
    out = _outputs(library_env, command)
    assert len(out) == 1
    assert out[0].success is False
    text = out[0].command_response.response
    assert command.name in text and "__result__" in text and fragment in text
    assert result.answer == text
    assert out[0].command_response.artifacts["error_type"]
    assert not any("writing the answer with one model call" in r.getMessage() for r in caplog.records)
    assert [r for r in caplog.records if r.levelno >= logging.ERROR and "__result__" in r.getMessage()]


def test_a_failing_result_shows_on_a_span_as_an_error_with_its_type(library_env, answered):
    command = _build_planning_command(_plan_of(_raises))
    _run(library_env, command)

    # The call's answer step (``fw.step`` kind answer, bd fix-qjne.34; was fw.skill.answer), run by code: no model call.
    (span,) = [s for s in library_env.sink.named(tracing.SPAN_STEP) if s.attributes.get("kind") == "answer"]
    assert span.status == tracing.STATUS_ERROR
    assert span.attributes["error_type"] == "RuntimeError"
    assert span.attributes["command"] == command.name and span.attributes["run_by"] == "code"


def test_refused_in_result_is_a_refusal_with_its_message(library_env, answered):
    command = _build_planning_command(_plan_of(_refuses))
    result = _run(library_env, command)

    assert answered == []
    (out,) = _outputs(library_env, command)
    assert out.success is False
    assert out.command_response.response == "no book of that title is on the shelf"
    assert "error_type" not in out.command_response.artifacts      # a decision, not a crash
    assert result.answer == "no book of that title is on the shelf"


def test_a_declared_output_returned_as_text_from_result_is_a_call_error(library_env, answered):
    command = _command_with_declared_output(_plan_of(_text_for_a_declared_output))
    _run(library_env, command)

    assert answered == []
    (out,) = _outputs(library_env, command)
    assert out.success is False
    assert out.command_response.response == (
        f"Execution error in {command.name}'s __result__: TypeError('{command.name} declares an Output but its "
        "__result__ returned str; return the Output, or CommandResponse(response=..., output=<Output>)')")


def test_answer_by_model_from_a_command_with_a_declared_output_is_a_call_error(library_env, answered):
    """The declared Output is the call's answer: handing the answer to a model would leave it out (fix-ftdl.8.13)."""
    from fastworkflow.planning import ANSWER_BY_MODEL

    command = _command_with_declared_output(_plan_of(lambda input, find: ANSWER_BY_MODEL))
    _run(library_env, command)

    assert answered == []
    (out,) = _outputs(library_env, command)
    assert out.success is False
    assert "declares an Output but its __result__ returned ANSWER_BY_MODEL" in out.command_response.response


@pytest.mark.parametrize("result_fn, returned", [
    (_text_for_a_declared_output, "str"),
    (_response_without_output, "CommandResponse without output"),
    (_other_output, "Other"),
    (_subclass_directly, "_Sub"),
    (_subclass_in_a_response, "CommandResponse(output=_Sub)"),
], ids=["str", "response-without-output", "other-model", "subclass-direct", "subclass-in-response"])
def test_result_of_a_command_with_an_output_must_be_exactly_that_output(library_env, answered, result_fn, returned):
    command = _command_with_declared_output(_plan_of(result_fn))
    _run(library_env, command)

    (out,) = _outputs(library_env, command)
    assert out.success is False
    assert f"its __result__ returned {returned};" in out.command_response.response
    assert answered == []


def test_the_declared_output_directly_and_inside_a_response_are_accepted(library_env, answered):
    for result_fn in (lambda i, f: Declared(found=True),
                      lambda i, f: fastworkflow.CommandResponse(response="yes", output=Declared(found=True))):
        command = _command_with_declared_output(_plan_of(result_fn))
        _run(library_env, command)
        out = _outputs(library_env, command)[-1]
        assert out.success is True and out.command_response.output == Declared(found=True)
    assert answered == []


def test_answer_by_model_from_result_hands_the_answer_to_a_model(library_env, answered):
    from fastworkflow.planning import ANSWER_BY_MODEL

    assert repr(ANSWER_BY_MODEL) == "fastworkflow.planning.ANSWER_BY_MODEL"
    command = _build_planning_command(_plan_of(lambda input, find: ANSWER_BY_MODEL))
    result = _run(library_env, command)

    assert result.answer == "written"
    assert len(answered) == 1
    assert _outputs(library_env, command) == []


# ----------------------------------------------------------------------
# No agent model: the turn ends as a failed command, not a completed answer and not a stop for want of a model
# ----------------------------------------------------------------------

@pytest.fixture
def no_agent(initialized_fastworkflow, no_agent_model, no_planner_key, monkeypatch):
    """No agent model, and the agent's LM must never be built (as ``tests/test_no_agent_mode.py`` asserts)."""
    def never(self):
        raise AssertionError("the agent LM was built in no-agent mode")

    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context", never)


@pytest.mark.parametrize("result_fn", [_raises, _empty_text, _an_int], ids=["raises", "empty-str", "int"])
def test_no_agent_a_failing_result_is_a_failed_output_not_an_answer_stop(no_agent, library_env, monkeypatch, answered, result_fn):
    command = _build_planning_command(_plan_of(result_fn))
    library_env.planning_command = command
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert answered == []
    (out,) = [o for o in result.command_outputs if o.command_name == command.name]
    assert out.command_response.success is False
    assert "__result__" in result.answer and command.name in result.answer
    # Not a normal completed answer: no "(Ran X.)" note, and not the stop that says a model is needed.
    assert "Ran " not in result.answer and "needs a model" not in result.answer
    assert out.command_response.artifacts.get("fw_not_run") is None
    assert result.success is False


def test_no_agent_answer_by_model_from_result_is_the_answer_stop(no_agent, library_env, monkeypatch, answered):
    from fastworkflow.planning import ANSWER_BY_MODEL

    command = _build_planning_command(_plan_of(lambda input, find: ANSWER_BY_MODEL))
    library_env.planning_command = command
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert answered == []
    assert "needs a model" in result.answer
    assert result.command_outputs[-1].command_response.artifacts.get("fw_not_run")


def test_no_agent_refused_in_result_is_its_refusal(no_agent, library_env, monkeypatch, answered):
    command = _build_planning_command(_plan_of(_refuses))
    library_env.planning_command = command
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune")

    assert result.answer == "no book of that title is on the shelf"
    assert result.success is False


# ----------------------------------------------------------------------
# A nested call: the caller's step fails, as any failed step does
# ----------------------------------------------------------------------

_RAISING_RESULT = ("    def __result__(self, input, book):\n"
                   "        raise RuntimeError('the nested answer broke')\n")
_TEXT_RESULT = ("    def __result__(self, input, book):\n"
                "        return f'{input.title} is a book'\n")


def _book_by_title_with(result_method: str) -> str:
    head, _, _ = _BOOK_BY_TITLE_SOURCE.partition("    def __result__")
    return head + result_method


@pytest.mark.parametrize("result_method, fragment", [
    (_RAISING_RESULT, "the nested answer broke"),
    (_TEXT_RESULT, "declares an Output but its __result__ returned str"),
], ids=["raises", "text-for-a-declared-output"])
def test_a_nested_call_with_a_failing_result_fails_the_callers_step(
        tmp_path, initialized_fastworkflow, monkeypatch, answered, result_method, fragment):
    from tests.test_nested_planning_commands import _nested_state

    state = _nested_state(tmp_path, initialized_fastworkflow, monkeypatch,
                          command_texts={BOOK_BY_TITLE_FILE: _book_by_title_with(result_method)})
    try:
        for _ in range(4):      # the agent takes over a failed step; it only reports it
            state.model.finish()
        result = _run_book_loans(state)

        assert answered == []
        child = next(o for o in state.ctx._turn_outputs if o.command_name == "Library/book_by_title")
        assert child.success is False
        assert fragment in child.command_response.response
        # The caller's `book` phase had no Output to read: it failed, and its call says so.
        assert "which failed" in result.answer or fragment in result.answer
        assert "is on loan to" not in result.answer
        assert len(state.ctx._frame_stack()) == 1
    finally:
        state.ctx.pop_active_workflow()


def test_a_text_result_for_a_declared_output_is_the_one_rules_error():
    from fastworkflow.engine.command_executor import command_response_of

    """Before fix-qjne.12 a command that declares an Output and wrote text from ``__result__`` handed its caller a
    response with ``output=None`` (the nested test above showed 'Dune is a book', success True, no Output, with the
    caller's phase left with nothing to read). The one rule refuses it where the text is written."""
    with pytest.raises(TypeError, match="declares an Output but its __result__ returned str"):
        command_response_of("a sentence", command_name="Library/book_by_title", output_model=Declared,
                            returned_by="__result__")


# ----------------------------------------------------------------------
# The one result rule, for both kinds
# ----------------------------------------------------------------------

def test_one_rule_for_what_a_command_returns():
    from fastworkflow.engine.command_executor import command_response_of

    declared = Declared(found=True)
    # the Output, directly or in a CommandResponse: accepted, the response text is the Output as JSON by default
    assert command_response_of(declared, command_name="x", output_model=Declared).output is declared
    kept = fastworkflow.CommandResponse(response="yes", output=declared)
    assert command_response_of(kept, command_name="x", output_model=Declared) is kept
    # a str is the answer only when no Output is declared
    assert command_response_of("text", command_name="x", output_model=None).response == "text"
    for bad in ("text", fastworkflow.CommandResponse(response="t"), Other(found=True), None):
        with pytest.raises(TypeError):
            command_response_of(bad, command_name="x", output_model=Declared)
    # no declared Output: a model instance is no short form
    with pytest.raises(TypeError, match="declares no Signature.Output"):
        command_response_of(declared, command_name="x", output_model=None)


@pytest.mark.parametrize("hand_over", [
    lambda out: _Sub(found=True, extra=1),
    lambda out: fastworkflow.CommandResponse(response="r", output=_Sub(found=True, extra=1)),
], ids=["direct", "in-a-response"])
def test_a_subclass_of_the_declared_output_is_refused_the_same_way_direct_or_in_a_response(hand_over):
    from fastworkflow.engine.command_executor import command_response_of, frame_command_result

    with pytest.raises(TypeError, match="declares an Output but returned"):
        frame_command_result(hand_over(None), command_name="x", output_model=Declared)
    with pytest.raises(TypeError, match="declares an Output but its __result__ returned"):
        command_response_of(hand_over(None), command_name="x", output_model=Declared, returned_by="__result__")


def test_a_success_false_response_is_refused_for_result_as_for_call():
    from fastworkflow.engine.command_executor import command_response_of

    with pytest.raises(TypeError, match="success=False"):
        command_response_of(fastworkflow.CommandResponse(response="r", success=False), command_name="x",
                            output_model=None, returned_by="__result__")


# ----------------------------------------------------------------------
# run-code: a failing __result__ stops the run (exit 1); None and a bare Output are read, not crashed on
# ----------------------------------------------------------------------

@pytest.fixture
def library(tmp_path) -> str:
    folder = str(tmp_path / "library")
    shutil.copytree(LIBRARY, folder, ignore=shutil.ignore_patterns("__pycache__", "___*"))
    return folder


@pytest.fixture
def no_model(monkeypatch):
    import litellm

    def refuse(*args, **kwargs):
        raise AssertionError("run-code called a model")

    monkeypatch.setattr(litellm, "completion", refuse)
    monkeypatch.setattr(litellm, "acompletion", refuse)


def _edit_library(folder: str, command_file: str, old: str, new: str) -> None:
    path = os.path.join(folder, "_commands", "Library", command_file)
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert source.count(old) == 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source.replace(old, new))


@pytest.mark.parametrize("body, fragment", [
    ('        return ""\n', "empty"),
    ("        return 3\n", "int"),
], ids=["empty-str", "int"])
def test_run_code_a_failing_result_is_a_run_error_naming_the_command(library, no_model, body, fragment):
    _edit_library(library, "book_loans.py", "    def __result__(self, input, book, loans):\n",
                  "    def __result__(self, input, book, loans):\n" + body)
    with pytest.raises(RunCodeError) as caught:
        run_code(library, "book_loans", {"title": "Dune"})
    assert "__result__" in str(caught.value) and "Library/book_loans" in str(caught.value)
    assert fragment in str(caught.value)


@pytest.mark.parametrize("body, fragment", [
    ("        return None\n", "returned None"),
    ('        return ""\n', "returned an empty str"),
    ('        return "  \\n "\n', "returned an empty str"),
    ("        return 3\n", "returned builtins.int"),
], ids=["none", "empty-str", "blank-str", "int"])
def test_one_rule_for_an_empty_or_unsupported_result_on_run_code_and_the_preview(library, no_model, body, fragment):
    """``planning_phases.judge_result`` is the one judge of a ``__result__`` return: the runtime (the agent-path test
    above), run-code and the preview all fail None, an empty or whitespace-only str and an unsupported value, with
    the same message (bd fix-ftdl.9.4 item 11)."""
    from fastworkflow.planning import command_preview

    _edit_library(library, "book_loans.py", "    def __result__(self, input, book, loans):\n",
                  "    def __result__(self, input, book, loans):\n" + body)
    with pytest.raises(RunCodeError) as caught:
        run_code(library, "book_loans", {"title": "Dune"})
    assert fragment in str(caught.value)
    preview = next(text for name, text in command_preview.render_previews(library).items() if "book_loans" in name)
    assert "it fails the call at run time" in preview and fragment in preview


def test_run_code_a_text_result_for_a_declared_output_is_a_run_error(library, no_model):
    _edit_library(library, "book_by_title.py", "    def __result__(self, input, book):\n",
                  "    def __result__(self, input, book):\n        return 'a sentence'\n")
    with pytest.raises(RunCodeError, match="declares an Output but its __result__ returned str"):
        run_code(library, "book_by_title", {"title": "Dune"})


def test_run_code_none_from_result_is_a_run_error(library, no_model):
    _edit_library(library, "book_loans.py", "    def __result__(self, input, book, loans):\n",
                  "    def __result__(self, input, book, loans):\n        return None\n")
    with pytest.raises(RunCodeError, match="returned None"):
        run_code(library, "book_loans", {"title": "Dune"})


def test_run_code_answer_by_model_says_a_model_would_write_the_answer(library, no_model):
    _edit_library(library, "book_loans.py", "    def __result__(self, input, book, loans):\n",
                  "    def __result__(self, input, book, loans):\n"
                  "        from fastworkflow.planning import ANSWER_BY_MODEL\n        return ANSWER_BY_MODEL\n")
    result = run_code(library, "book_loans", {"title": "Dune"})
    assert "no answer written by code" in result.text
    assert result.answer in ("", None)


def test_run_code_a_bare_output_from_result_is_the_answer_and_the_output(library, no_model):
    _edit_library(library, "book_by_title.py", "    def __result__(self, input, book):\n",
                  "    def __result__(self, input, book):\n        return Signature.Output(found=False)\n")
    result = run_code(library, "book_by_title", {"title": "Dune"})
    assert result.output is not None and result.output.found is False
    assert "answer written by __result__:" in result.text


# ----------------------------------------------------------------------
# The loader can tell when a return annotation contradicts the declared Output
# ----------------------------------------------------------------------

def test_a_result_annotated_str_for_a_declared_output_does_not_load():
    from fastworkflow.planning.planning_commands import PlanningCommandFormatError

    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find) -> str:
            return "x"

    with pytest.raises(PlanningCommandFormatError,
                       match=r"__result__ is annotated -> str, but the command declares Signature.Output"):
        _command_with_declared_output(Plan)


def test_a_result_annotated_with_the_declared_output_or_a_response_or_not_at_all_loads():
    for annotation in (Declared, fastworkflow.CommandResponse, "Declared", None):
        class Plan:
            def find(self, input) -> FoundBooks:
                return [run("find_book", query=input.title)]

            def __result__(self, input, find):
                return Declared(found=True)

        if annotation is not None:
            Plan.__result__.__annotations__["return"] = annotation
        assert _command_with_declared_output(Plan).uses_result is True


def test_a_result_annotated_with_another_model_for_a_declared_output_does_not_load():
    from fastworkflow.planning.planning_commands import PlanningCommandFormatError

    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        def __result__(self, input, find) -> Other:
            return Other(found=True)

    with pytest.raises(PlanningCommandFormatError, match="annotated -> Other"):
        _command_with_declared_output(Plan)
