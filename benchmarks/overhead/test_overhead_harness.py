"""Focused unit tests for the overhead-benchmark harness itself.

These never start a fastWorkflow session, never train, never touch a model
provider. They exercise the stub LM against real dspy adapters (offline) and
the driver's pure helpers.

Run from this folder:
    /home/drawal/rl/fastworkflow/.venv/bin/python -m pytest -q -p no:cacheprovider test_overhead_harness.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any, Literal

import dspy
import pytest
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compare  # noqa: E402
import run_overhead  # noqa: E402
from stub_lm import (  # noqa: E402
    NetworkCallAttempted,
    NetworkTripwire,
    ScriptedStubLM,
    default_value_for,
    literal_choices,
    parse_output_fields,
    OutputField,
)


class ReactStep(dspy.Signature):
    """Mimics fastWorkflowReAct's per-step signature."""
    user_query: str = dspy.InputField()
    trajectory: str = dspy.InputField()
    next_thought: str = dspy.OutputField()
    next_tool_name: Literal["what_can_i_do", "execute_workflow_query", "ask_user", "finish"] = dspy.OutputField()
    next_tool_args: dict[str, Any] = dspy.OutputField()


class Extract(dspy.Signature):
    user_query: str = dspy.InputField()
    trajectory: str = dspy.InputField()
    final_answer: str = dspy.OutputField()


class Planner(dspy.Signature):
    user_query: str = dspy.InputField()
    next_steps: str = dspy.OutputField()


class Summary(dspy.Signature):
    user_query: str = dspy.InputField()
    conversation_summary: str = dspy.OutputField()


class Typed(dspy.Signature):
    question: str = dspy.InputField()
    flag: bool = dspy.OutputField()
    count: int = dspy.OutputField()
    entries: list[str] = dspy.OutputField()
    choice: Literal["alpha", "beta"] = dspy.OutputField()
    text: str = dspy.OutputField()


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

def test_parse_chat_adapter_fields():
    tail = (
        "Respond with the corresponding output fields, starting with the field "
        "`[[ ## next_thought ## ]]`, then `[[ ## next_tool_name ## ]]` (must be formatted as a valid "
        "Python Literal['execute_workflow_query', 'finish']), then `[[ ## next_tool_args ## ]]` "
        "(must be formatted as a valid Python dict[str, Any]), and then ending with the marker for "
        "`[[ ## completed ## ]]`."
    )
    fields = parse_output_fields("[[ ## user_query ## ]]\nq\n\n" + tail)
    assert [f.name for f in fields] == ["next_thought", "next_tool_name", "next_tool_args"]
    assert fields[1].type_hint == "Literal['execute_workflow_query', 'finish']"
    assert fields[2].type_hint == "dict[str, Any]"
    assert fields[0].type_hint is None


def test_parse_json_adapter_fields():
    tail = (
        "Respond with a JSON object in the following order of fields: `next_thought`, then "
        "`next_tool_name` (must be formatted as a valid Python Literal['a', 'b']), then "
        "`next_tool_args` (must be formatted as a valid Python dict[str, Any])."
    )
    fields = parse_output_fields("[[ ## trajectory ## ]]\n\n" + tail)
    assert [f.name for f in fields] == ["next_thought", "next_tool_name", "next_tool_args"]
    assert literal_choices(fields[1].type_hint) == ["a", "b"]


def test_parse_unknown_prompt_is_empty():
    assert parse_output_fields("just some text") == []


def test_default_values_by_hint():
    assert default_value_for(OutputField("x", "bool")) is True
    assert default_value_for(OutputField("x", "int")) == 0
    assert default_value_for(OutputField("x", "float")) == 0.0
    assert default_value_for(OutputField("x", "list[str]")) == []
    assert default_value_for(OutputField("x", "dict[str, Any]")) == {}
    assert default_value_for(OutputField("x", "Literal['p', 'q']")) == "p"
    assert isinstance(default_value_for(OutputField("x", None)), str)


# ---------------------------------------------------------------------------
# Scripted ReAct turn through real dspy adapters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("adapter", [ChatAdapter(), JSONAdapter()])
def test_scripted_turn_end_to_end(adapter):
    script = ["cmd_one <a>1</a>", "cmd_two <b>2</b>", "cmd_three"]
    stub = ScriptedStubLM(script)
    react = dspy.Predict(ReactStep)
    extract = dspy.ChainOfThought(Extract)
    planner = dspy.ChainOfThought(Planner)
    summary = dspy.ChainOfThought(Summary)

    with dspy.context(lm=stub, adapter=adapter):
        plan = planner(user_query="q")
        assert "1." in plan.next_steps
        issued = []
        for _ in range(10):
            step = react(user_query="q", trajectory=json.dumps(issued))
            if step.next_tool_name == "finish":
                assert step.next_tool_args == {}
                break
            assert step.next_tool_name == "execute_workflow_query"
            issued.append(step.next_tool_args["command"])
        else:
            pytest.fail("finish never issued")
        final = extract(user_query="q", trajectory=json.dumps(issued))
        assert final.final_answer
        summ = summary(user_query="q")
        assert summ.conversation_summary

    assert issued == script
    s = stub.summary()
    assert s["total_commands_issued"] == 3
    assert s["unparsed_prompts"] == 0
    assert s["anomalies"] == []
    assert len(s["turns"]) == 1
    turn = s["turns"][0]
    assert turn["finished"] is True
    assert turn["calls_by_kind"] == {"planner": 1, "react_tool": 3, "react_finish": 1, "extract": 1, "summary": 1}
    assert turn["model_calls"] == 7
    # One gap per tool call: measured between a tool-selecting step and the next step.
    assert len(turn["command_gaps_s"]) == 3
    assert all(g >= 0 for g in turn["command_gaps_s"])


def test_two_turns_are_separated_and_script_restarts():
    stub = ScriptedStubLM(["c1", "c2"])
    react = dspy.Predict(ReactStep)
    planner = dspy.ChainOfThought(Planner)

    def one_turn():
        out = []
        with dspy.context(lm=stub, adapter=ChatAdapter()):
            planner(user_query="q")
            while True:
                step = react(user_query="q", trajectory="")
                if step.next_tool_name == "finish":
                    return out
                out.append(step.next_tool_args["command"])

    assert one_turn() == ["c1", "c2"]
    assert one_turn() == ["c1", "c2"]
    s = stub.summary()
    assert len(s["turns"]) == 2
    assert [t["commands_issued"] for t in s["turns"]] == [2, 2]


def test_turn_without_planner_still_restarts_after_finish():
    stub = ScriptedStubLM(["c1"])
    react = dspy.Predict(ReactStep)
    with dspy.context(lm=stub, adapter=ChatAdapter()):
        assert react(user_query="q", trajectory="").next_tool_name == "execute_workflow_query"
        assert react(user_query="q", trajectory="").next_tool_name == "finish"
        assert react(user_query="q", trajectory="").next_tool_name == "execute_workflow_query"
    assert len(stub.turns) == 2


def test_set_script_between_turns_and_close():
    stub = ScriptedStubLM(["c1"])
    react = dspy.Predict(ReactStep)
    with dspy.context(lm=stub, adapter=ChatAdapter()):
        react(user_query="q", trajectory="")
        with pytest.raises(RuntimeError):
            stub.set_script(["x"])
        react(user_query="q", trajectory="")  # finish
    stub.set_script(["n1", "n2", "n3"])
    stub.close_turn()
    with dspy.context(lm=stub, adapter=ChatAdapter()):
        cmds = []
        while True:
            step = react(user_query="q", trajectory="")
            if step.next_tool_name == "finish":
                break
            cmds.append(step.next_tool_args["command"])
    assert cmds == ["n1", "n2", "n3"]


def test_typed_fields_parse_under_both_adapters():
    for adapter in (ChatAdapter(), JSONAdapter()):
        stub = ScriptedStubLM([])
        with dspy.context(lm=stub, adapter=adapter):
            pred = dspy.Predict(Typed)(question="?")
        assert pred.flag is True
        assert pred.count == 0
        assert pred.entries == []
        assert pred.choice == "alpha"
        assert isinstance(pred.text, str)


def test_tool_not_offered_finishes_and_records_anomaly():
    class NoExec(dspy.Signature):
        user_query: str = dspy.InputField()
        next_thought: str = dspy.OutputField()
        next_tool_name: Literal["other_tool", "finish"] = dspy.OutputField()
        next_tool_args: dict[str, Any] = dspy.OutputField()

    stub = ScriptedStubLM(["c1"])
    with dspy.context(lm=stub, adapter=ChatAdapter()):
        step = dspy.Predict(NoExec)(user_query="q")
    assert step.next_tool_name == "finish"
    assert stub.anomalies


# ---------------------------------------------------------------------------
# Tripwire
# ---------------------------------------------------------------------------

def test_tripwire_blocks_real_lm_and_litellm():
    import litellm
    from dspy.clients.lm import LM as RealLM

    original = litellm.completion
    tw = NetworkTripwire().install()
    try:
        with pytest.raises(NetworkCallAttempted):
            litellm.completion(model="x", messages=[])
        with pytest.raises(NetworkCallAttempted):
            RealLM("openai/never-called").forward(prompt="hi")
        assert tw.trips == 2
        # The stub is a DummyLM subclass and stays usable under the tripwire.
        stub = ScriptedStubLM([])
        with dspy.context(lm=stub, adapter=ChatAdapter()):
            assert dspy.ChainOfThought(Planner)(user_query="q").next_steps
    finally:
        tw.uninstall()
    assert litellm.completion is original


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------

def test_percentile_and_stats():
    vals = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert run_overhead.percentile(vals, 50) == 3.0
    assert run_overhead.percentile(vals, 90) == 5.0
    assert run_overhead.percentile(vals, 0) == 1.0
    assert run_overhead.percentile([], 50) is None
    s = run_overhead.stats(vals)
    assert s["n"] == 5 and s["median"] == 3.0 and s["min"] == 1.0 and s["max"] == 5.0
    assert run_overhead.stats([])["median"] is None


def test_build_script_cycles_and_substitutes():
    out = run_overhead.build_script(["a {i} {j}", "b {n}"], 5)
    assert out == ["a 1 2", "b 1", "a 3 4", "b 3", "a 5 6"]


def test_db_helpers_on_synthetic_store(tmp_path):
    db = tmp_path / "observability.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE spans (id INTEGER PRIMARY KEY, name TEXT, status TEXT, duration_ms REAL)")
    conn.execute("CREATE TABLE turns (id INTEGER PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO spans (name, status, duration_ms) VALUES (?, ?, ?)",
        [("fw.command.execute", "ok", 10.0), ("fw.command.execute", "ok", 30.0), ("fw.agent.step", "ok", 1.0)],
    )
    conn.execute("INSERT INTO turns DEFAULT VALUES")
    conn.commit()
    conn.close()

    rows = run_overhead.db_row_counts(str(db))
    assert rows["tables"] == {"spans": 3, "turns": 1}
    assert rows["span_names"] == {"fw.agent.step": 1, "fw.command.execute": 2}
    assert rows["spans_by_status"] == {"ok": 3}
    assert rows["command_execute_span_ms"]["n"] == 2
    assert rows["command_execute_span_ms"]["median"] == 20.0
    sizes = run_overhead.db_size_bytes(str(db))
    assert sizes["main"] > 0 and sizes["total"] >= sizes["main"]
    assert "error" in run_overhead.db_row_counts(str(tmp_path / "missing.sqlite3"))


def test_parse_args_defaults_and_env():
    ns = run_overhead.parse_args(["--commit", "x", "--env", "A=1", "--env", "B=2"])
    assert ns.commit == "x" and ns.turns == 10 and ns.commands_per_turn == 48
    assert ns.env == ["A=1", "B=2"] and ns.observability == "on"


def test_one_line_summary_and_compare_render():
    report = {
        "commit_label": "fork",
        "observability": "on",
        "metrics": {
            "turn_wall_ms": {"n": 2, "median": 100.0, "p90": 120.0},
            "command_gap_ms": {"n": 4, "median": 1.5, "p90": 2.0},
            "commands_per_turn": {"median": 2.0},
            "model_calls_per_turn": {"median": 6.0},
            "process_cpu_s": 0.5,
            "process_cpu_per_turn_s": 0.25,
            "peak_rss_bytes": 300 * 1024 * 1024,
        },
        "observability_db_bytes": {"total": 4096},
        "observability_rows": {"tables": {"spans": 12, "turns": 2}},
        "tripwire_trips": 0,
        "error": None,
    }
    line = run_overhead.one_line(report)
    assert "[fork]" in line and "spans=12" in line and "tripwire=0" in line
    other = json.loads(json.dumps(report))
    other["commit_label"] = "port"
    other["metrics"]["turn_wall_ms"]["median"] = 110.0
    table = compare.render([report, other])
    assert "fork" in table and "port" in table and "+10.0%" in table
