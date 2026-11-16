"""A split turn returns its parts as data next to the unchanged answer text (the artifact ``fw_sub_tasks``).

Driven like tests/test_subtask_execution.py: a real execution context on the todo workflow, the planner's split and the
agent's runs scripted (the seams that file already uses). The answer text is the one the headings have always made.
"""
import pytest

from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import (
    TURN_PARTS_ARTIFACT, WorkflowExecutionContext, _heading_title)
import fastworkflow
from tests.test_subtask_execution import _agent_with_steps, _split  # noqa: F401
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    _make_agent_ctx,
    _set_agents,
    initialized_fastworkflow,
    todo_workflow_path,
)


@pytest.fixture(autouse=True)
def no_review_gate(monkeypatch):
    monkeypatch.setattr(WorkflowExecutionContext, "_subtask_gate", lambda self, subtasks: None)


def _recording(ctx, agent):
    """The scripted agent also records one command output per run, as the real agent's tool calls do
    (``append_turn_output``): the parts travel on the turn's last command output."""
    scripted = agent.side_effect

    def run(**kwargs):
        ctx.append_turn_output(fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ran"), command_name="find"))
        return scripted(**kwargs)

    agent.side_effect = run
    return agent


def _parts(result):
    final = result.command_outputs[-1].command_response
    return final.artifacts.get(TURN_PARTS_ARTIFACT)


def _body(answer, part):
    return answer[part["start"]:part["end"]]


def test_every_part_answered_lists_each_part_and_the_text_is_unchanged(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find the very long thing Y: with a clause, then more words", "find Z"])
    _set_agents(ctx, _recording(ctx, _agent_with_steps(["X is here", "Y is there\nsecond line", "Z is gone"])))

    result = ctx.process_turn("find X, Y and Z")

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == (
        "### 1. find X\nX is here\n\n"
        f"### 2. {_heading_title('find the very long thing Y: with a clause, then more words')}\n"
        "Y is there\nsecond line\n\n"
        "### 3. find Z\nZ is gone")
    parts = _parts(result)
    assert [(p["number"], p["answered"]) for p in parts] == [(1, True), (2, True), (3, True)]
    assert [p["text"] for p in parts] == ["find X", "find the very long thing Y: with a clause, then more words", "find Z"]
    assert [p["title"] for p in parts] == ["find X", "find the very long thing Y", "find Z"]
    assert [_body(result.answer, p) for p in parts] == ["X is here", "Y is there\nsecond line", "Z is gone"]
    for part in parts:     # the heading the title makes sits right before the body
        assert result.answer[:part["start"]].endswith(f"### {part['number']}. {part['title']}\n")


def test_a_part_with_no_answer_is_listed_unanswered_and_has_no_heading(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find Y", "find Z"])
    _set_agents(ctx, _recording(ctx, _agent_with_steps(["X is here", "", "Z is gone"])))

    result = ctx.process_turn("find X, Y and Z")

    assert result.answer == "### 1. find X\nX is here\n\n### 3. find Z\nZ is gone"
    parts = _parts(result)
    assert [(p["number"], p["text"], p["answered"]) for p in parts] == [
        (1, "find X", True), (2, "find Y", False), (3, "find Z", True)]
    assert (parts[1]["start"], parts[1]["end"], parts[1]["title"]) == (None, None, "find Y")
    assert [_body(result.answer, p) for p in (parts[0], parts[2])] == ["X is here", "Z is gone"]


def test_one_answered_part_of_a_split_turn_is_the_whole_text_without_a_heading(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    _set_agents(ctx, _recording(ctx, _agent_with_steps(["", "Y is there"])))

    result = ctx.process_turn("find X and Y")

    assert result.answer == "Y is there"
    parts = _parts(result)
    assert [(p["number"], p["answered"]) for p in parts] == [(1, False), (2, True)]
    assert _body(result.answer, parts[1]) == "Y is there"


def test_a_part_that_was_handed_off_keeps_its_labelled_runs_in_its_span(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    from tests.test_subtask_execution import _fake_progress_check
    _fake_progress_check(monkeypatch)
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    _set_agents(ctx, _recording(ctx, _agent_with_steps(["X part one", "X part two", "Y is there"], exhaust=lambda n: n == 1)))

    result = ctx.process_turn("find X and Y")

    parts = _parts(result)
    assert [p["number"] for p in parts] == [1, 2]
    assert _body(result.answer, parts[0]) == "Part 1 of 2:\nX part one\n\nPart 2 of 2:\nX part two"
    assert _body(result.answer, parts[1]) == "Y is there"


def test_a_single_sub_task_turn_has_no_parts_and_a_plain_answer(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X"])
    _set_agents(ctx, _recording(ctx, _agent_with_steps(["X is here"])))

    result = ctx.process_turn("find X")

    assert result.answer == "X is here"
    assert _parts(result) is None
    assert all(TURN_PARTS_ARTIFACT not in o.command_response.artifacts for o in result.command_outputs)


def test_a_turn_with_no_command_output_carries_no_parts(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    _set_agents(ctx, _agent_with_steps(["X is here", "Y is there"]))

    result = ctx.process_turn("find X and Y")

    assert result.answer == "### 1. find X\nX is here\n\n### 2. find Y\nY is there"
    assert result.command_outputs == []


def test_parts_are_not_listed_when_the_answer_was_changed_after_they_were_read(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    """The offsets belong to the combined text; a cleaned or replaced answer gets none rather than wrong ones."""
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    _set_agents(ctx, _recording(ctx, _agent_with_steps(["X is here [[ ## completed ## ]]", "Y is there"])))

    result = ctx.process_turn("find X and Y")

    assert "[[" not in result.answer
    assert _parts(result) is None


def test_a_continued_part_lists_its_runs_as_offsets_that_agree_with_the_labels_in_the_text(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    from tests.test_subtask_execution import _fake_progress_check
    _fake_progress_check(monkeypatch)
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    _set_agents(ctx, _recording(ctx, _agent_with_steps(["X part one", "X part two", "Y is there"], exhaust=lambda n: n == 1)))

    result = ctx.process_turn("find X and Y")

    parts = _parts(result)
    runs = parts[0]["runs"]
    assert [(r["index"], r["of"]) for r in runs] == [(1, 2), (2, 2)]
    assert [result.answer[r["start"]:r["end"]] for r in runs] == ["X part one", "X part two"]
    for run in runs:       # the label line the text carries sits right before the run's answer
        assert result.answer[:run["start"]].endswith(f"Part {run['index']} of {run['of']}:\n")
        assert parts[0]["start"] <= run["start"] <= run["end"] <= parts[0]["end"]
    assert "runs" not in parts[1], "a part that was not continued has no runs"
    assert result.answer.count("Part ") == len(runs)


def test_a_repeated_run_is_dropped_from_the_text_and_from_the_runs(
        initialized_fastworkflow, todo_workflow_path, monkeypatch):
    from fastworkflow.workflow_execution_context import _join_parts, _part_runs
    for parts in (["a", "a"], ["a", "b", "b"], ["a"], ["a", "b", "a"]):
        text, runs = _join_parts(parts), _part_runs(parts)
        assert [text[r["start"]:r["end"]] for r in runs] == (
            [] if len({*parts}) == 1 or len(parts) == 1 else [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]])
        assert text.count("Part ") == len(runs)
