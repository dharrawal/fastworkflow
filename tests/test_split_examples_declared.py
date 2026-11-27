"""A workflow's splitter examples are declared as data and REPLACE the built-in generic examples (bd fix-qjne.26 stage 3).

The ``class Context`` of ``_commands/_workflow.py`` may set ``split_examples = [("<request>", ["<part 1>", ...]), ...]``: the splitter then reads
the built-in rules alone (``SPLIT_RULES``) as its instructions, and each example as a demo before the real request,
carrying the same job list the real request gets. A workflow without examples keeps today's prompt byte for byte. The
framework's rules are not replaceable: a ``Signature`` in it (the retired replacement prompt) fails the load, and
so does the removed ``split_guidance``, each naming ``split_examples``. Real components, no mocks: the planner's model is the scripted ``DummyLM`` the neighbouring split
tests use, and the prompt it receives is read from its own history.
"""
from __future__ import annotations

from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

from fastworkflow.planning import command_preview
from fastworkflow.agent import workflow_agent
from fastworkflow.planning.planning_commands import PlanningCommandFormatError
from fastworkflow.agent.chat_adapter import CommandsSystemPreludeAdapter
from fastworkflow.agent.workflow_agent import SPLIT_GENERIC_EXAMPLES, SPLIT_RULES, SplitExample, SplitIntoSubtasks
from tests.workflow_settings_file import write_workflow_file

pytestmark = pytest.mark.usefixtures("no_planner_key")

LIBRARY = "fastworkflow/examples/library"
#: The built-in prompt's digest before this change: a workflow without examples must keep it.
BUILT_IN_DIGEST = "7d0cc28ae114"
REQUEST = "Find the book Dune. Find the book Emma."
PARTS = ["Find the book Dune.", "Find the book Emma."]
EXAMPLES = [
    ("Count the books on shelf S-12 and on shelf S-14.",
     ["Count the books on shelf S-12.", "Count the books on shelf S-14."]),
    ("Show the member Ann Lee with her loans and fines.", ["Show the member Ann Lee with her loans and fines."]),
]
DECLARED = f"split_examples = {EXAMPLES!r}\n"
REPLACING = '''
class Signature:
    """Split a library request into its independent jobs, one numbered line each."""
'''
#: A planning command, so the job list is not "(none)": the demos must carry the same text as the real request.
SHELF_COMMAND = ('class Signature:\n    __doc__ = "Count the books on one shelf. One job per shelf."\n\n\n'
                 'class PlanGenerator:\n    def steps(self):\n        return ["count the shelf"]\n')


def _workflow(folder: Path, split_file: str | None, *, planning: bool = True) -> str:
    (folder / "_commands").mkdir(parents=True, exist_ok=True)
    if planning:
        (folder / "_commands" / "shelf_count.py").write_text(SHELF_COMMAND, encoding="utf-8")
    if split_file is not None:
        write_workflow_file(folder / "_commands", split_file)
    return str(folder)


class _PromptSeeingLM(DummyLM):
    """The scripted planner model, keeping the messages of each call: the splitter calls a copy of it
    (``workflow_agent._split_lm``), so the list is shared by the class, not held on the instance."""

    seen: list = []

    def forward(self, prompt=None, messages=None, **kwargs):
        _PromptSeeingLM.seen.append(messages or [])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def _split(folder: str) -> tuple[list[str], list[dict], dict]:
    """The merged parts, the messages the model received, and the split record."""
    _PromptSeeingLM.seen.clear()
    lm = _PromptSeeingLM([{"reasoning": "r", "subtasks": "\n".join(f"{n}. {p}" for n, p in enumerate(PARTS, 1))}])
    with workflow_agent.recording_split() as record:
        parts = workflow_agent.split_into_subtasks(REQUEST, planner_lm=lm, workflow_folderpath=folder)
    assert len(_PromptSeeingLM.seen) == 1
    return parts, _PromptSeeingLM.seen[0], record


def _flat(text: str) -> str:
    """*text* with each line's surrounding blanks removed: the adapter indents the instructions it writes."""
    return "\n".join(line.strip() for line in text.splitlines())


def test_the_examples_are_demos_with_the_calls_job_list_in_order_before_the_request(tmp_path):
    folder = _workflow(tmp_path / "declared", DECLARED)
    parts, messages, _ = _split(folder)
    assert parts == PARTS
    job_types = workflow_agent._job_types_for(folder)
    assert "shelf_count" in job_types
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    for (request, example_parts), asked, answered in zip(EXAMPLES, messages[1:5:2], messages[2:5:2]):
        assert job_types in asked["content"] and request in asked["content"]
        assert "\n".join(f"{n}. {p}" for n, p in enumerate(example_parts, 1)) in answered["content"]
        assert REQUEST not in asked["content"]
    assert job_types in messages[-1]["content"] and REQUEST in messages[-1]["content"]


def test_the_examples_replace_the_generic_ones_and_keep_the_built_in_rules(tmp_path):
    folder = _workflow(tmp_path / "declared", DECLARED)
    system = _flat(_split(folder)[1][0]["content"])
    assert _flat(SPLIT_RULES) in system
    assert "Example 1" not in system and "Queue stalls at night" not in system
    assert workflow_agent._split_signature(folder).instructions == SPLIT_RULES
    assert SplitIntoSubtasks.instructions == f"{SPLIT_RULES}\n\n{SPLIT_GENERIC_EXAMPLES}"


@pytest.mark.parametrize("split_file", [None, "subject_hints = 'a book'\n"])
def test_a_workflow_without_examples_gets_todays_prompt_byte_for_byte(tmp_path, split_file):
    folder = _workflow(tmp_path / "plain", split_file)
    assert workflow_agent._split_signature(folder).instructions == SplitIntoSubtasks.instructions
    assert workflow_agent.split_prompt(folder) == workflow_agent.split_prompt(None) == \
        workflow_agent.split_prompt(LIBRARY) == {"digest": BUILT_IN_DIGEST, "source": "built_in"}
    assert workflow_agent.split_demos(folder, "jobs") == []
    _, messages, record = _split(folder)
    expected = CommandsSystemPreludeAdapter().format(
        dspy.ChainOfThought(SplitIntoSubtasks).predict.signature, [],
        {"job_types": workflow_agent._job_types_for(folder), "user_query": REQUEST})
    assert [(m["role"], m["content"]) for m in messages] == [(m["role"], m["content"]) for m in expected]
    assert (record["prompt_digest"], record["prompt_source"]) == (BUILT_IN_DIGEST, "built_in")


def test_a_signature_in_the_split_file_is_one_workflow_wide_load_problem_naming_split_examples(tmp_path):
    """The replacement prompt is retired (bd fix-qjne.26): a ``Signature`` in ``_workflow.py`` stops the workflow with one
    line naming the file and what to write instead; the splitter's rules stay the framework's."""
    folder = _workflow(tmp_path / "replacing", REPLACING, planning=False)
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1, problems
    assert problems[0].workflow_wide
    line = str(problems[0])
    assert "_workflow.py" in line and "split_examples" in line and "Signature" in line
    assert "rules are not replaceable" in line
    with pytest.raises(PlanningCommandFormatError, match="a Signature"):
        workflow_agent.split_prompt(folder)


def test_the_split_record_carries_the_added_source_and_a_digest_of_the_examples(tmp_path):
    folder = _workflow(tmp_path / "declared", DECLARED)
    prompt = workflow_agent.split_prompt(folder)
    assert prompt["source"] == workflow_agent.SPLIT_PROMPT_BUILT_IN_AND_WORKFLOW == "built_in+workflow"
    assert prompt["digest"] != BUILT_IN_DIGEST
    record = _split(folder)[2]
    assert (record["prompt_digest"], record["prompt_source"]) == (prompt["digest"], "built_in+workflow")
    reworded = [(EXAMPLES[0][0], ["Count the books on S-12.", EXAMPLES[0][1][1]]), EXAMPLES[1]]
    other = _workflow(tmp_path / "reworded", f"split_examples = {reworded!r}\n")
    assert workflow_agent.split_prompt(other)["digest"] != prompt["digest"]   # the digest covers the parts
    more_jobs = _workflow(tmp_path / "more_jobs", DECLARED)
    (Path(more_jobs) / "_commands" / "shelf_walk.py").write_text(
        SHELF_COMMAND.replace("Count the books on one shelf", "Walk one shelf"), encoding="utf-8")
    assert workflow_agent.split_prompt(more_jobs)["digest"] == prompt["digest"]   # not the job list


def test_split_examples_returns_the_declared_examples(tmp_path):
    folder = _workflow(tmp_path / "declared", DECLARED)
    own = tuple(SplitExample(request, tuple(parts)) for request, parts in EXAMPLES)
    assert workflow_agent.split_examples(folder, workflow_only=True) == own
    assert workflow_agent.split_examples(folder) == own   # they replace the generic ones
    assert len(workflow_agent.split_examples(None)) == 3


def _parts(count: int) -> list[str]:
    return [f"Count the books on shelf S-{n}." for n in range(count)]


@pytest.mark.parametrize("split_file, match", [
    (REPLACING, "a Signature \\(a replacement splitter prompt\\) is removed: write split_examples"),
    (DECLARED + REPLACING, "a Signature \\(a replacement splitter prompt\\) is removed: write split_examples"),
    ("split_guidance = 'a rule'\n", "split_guidance is removed: write split_examples"),
    ("split_examples = []\n", "split_examples must be a non-empty list of"),
    ("split_examples = 'Find Dune.'\n", "split_examples must be a non-empty list of"),
    ("split_examples = [('Find Dune.', ['Find Dune.'])] * 13\n", "13 examples, more than 12"),
    ("split_examples = [['Find Dune.', ['Find Dune.']]]\n", "example 1 must be a \\(request, parts\\) pair"),
    ("split_examples = [('Find Dune.',)]\n", "example 1 must be a \\(request, parts\\) pair"),
    ("split_examples = [('  ', ['Find Dune.'])]\n", "example 1: the request must be a non-empty str"),
    ("split_examples = [('Find Dune.', [])]\n", "example 1: the parts must be a list of at least one"),
    ("split_examples = [('Find Dune.', 'Find Dune.')]\n", "example 1: the parts must be a list of at least one"),
    ("split_examples = [('Find Dune.', ['Find Dune.', ' '])]\n", "example 1: the parts must be a list of at least one"),
    (f"split_examples = [('Find Dune.', ['Find Dune.']), ('Count.', {_parts(11)!r})]\n",
     "example 2: 11 parts, more than the 10"),
    ("split_examples = [('Find Dune.', ['1. Find Dune.'])]\n", "without a list number"),
    ("split_examples = [('Find Dune.', ['2) Find Dune.'])]\n", "without a list number"),
    ("split_examples = [('Find Dune.', ['Find\\nDune.'])]\n", "must be one line"),
])
def test_a_malformed_split_file_fails_the_workflow_load(tmp_path, split_file, match):
    folder = _workflow(tmp_path / "bad", split_file, planning=False)
    with pytest.raises(PlanningCommandFormatError, match=match):
        workflow_agent.split_settings(folder)
    problems = command_preview.load_problems(folder)
    assert any(match.replace("\\", "") in str(problem) for problem in problems), problems
    assert all(problem.workflow_wide for problem in problems if "_workflow.py" in str(problem))


def test_the_most_examples_and_parts_allowed_load(tmp_path):
    examples = [(f"Count shelves {n}.", _parts(workflow_agent.MAX_SUBTASKS))
                for n in range(workflow_agent.MAX_SPLIT_EXAMPLES)]
    folder = _workflow(tmp_path / "full", f"split_examples = {examples!r}\n", planning=False)
    assert len(workflow_agent.split_settings(folder).examples) == workflow_agent.MAX_SPLIT_EXAMPLES == 12


# ---------------------------------------------------------------- the workflow's settings file, _workflow.py

def _one_workflow_wide_problem(folder: str) -> str:
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and problems[0].workflow_wide, problems
    return str(problems[0])


def test_a_split_file_is_the_removed_spelling_and_is_never_read(tmp_path):
    """``_commands/_split.py`` (the settings as module attributes) fails the load with one line naming both files,
    even beside a ``_workflow.py``: nothing is read from it."""
    folder = _workflow(tmp_path / "old", None, planning=False)
    (Path(folder) / "_commands" / "_split.py").write_text(DECLARED, encoding="utf-8")
    line = _one_workflow_wide_problem(folder)
    assert "_commands/_split.py: _split.py is removed: move its settings" in line
    assert "class Context in _commands/_workflow.py" in line
    with pytest.raises(PlanningCommandFormatError, match="_split.py is removed"):
        workflow_agent.split_settings(folder)
    write_workflow_file(Path(folder) / "_commands", DECLARED)
    with pytest.raises(PlanningCommandFormatError, match="_split.py is removed"):
        workflow_agent.split_examples(folder)


def test_a_workflow_settings_file_without_class_context_fails_the_load(tmp_path):
    folder = _workflow(tmp_path / "no_class", None, planning=False)
    (Path(folder) / "_commands" / "_workflow.py").write_text(DECLARED, encoding="utf-8")
    line = _one_workflow_wide_problem(folder)
    assert "_workflow.py: defines no class Context" in line and "split_examples outside a class" in line


def test_a_setting_written_beside_class_context_fails_the_load(tmp_path):
    folder = _workflow(tmp_path / "beside", None, planning=False)
    (Path(folder) / "_commands" / "_workflow.py").write_text(
        "class Context:\n    subject_hints = 'a book'\n\n\n" + DECLARED, encoding="utf-8")
    assert "split_examples is set outside class Context" in _one_workflow_wide_problem(folder)


@pytest.mark.parametrize("member, match", [
    ("split_exmaples = []", "Context.split_exmaples is not a workflow setting; did you mean 'split_examples'?"),
    ("notes = 'x'", "Context.notes is not a workflow setting. The settings are split_examples, identifier_patterns, "
                    "subject_hints; a helper starts with _"),
    ("def get_parent(self, command_context_object):\n    return None",
     "Context.get_parent is a context hook, but the workflow is not a record context and has no parent"),
    ("instance_label_attr = 'name'", "Context.instance_label_attr is a context hook"),
])
def test_class_context_holds_only_the_workflow_settings(tmp_path, member, match):
    folder = _workflow(tmp_path / "member", "subject_hints = 'a book'\n" + member + "\n", planning=False)
    assert match in _one_workflow_wide_problem(folder)
    with pytest.raises(PlanningCommandFormatError, match=r"Context\."):
        workflow_agent.split_settings(folder)


def test_a_private_helper_on_class_context_is_fine(tmp_path):
    folder = _workflow(tmp_path / "helper", "_ID = 'acct'\nsubject_hints = 'a book'\n", planning=False)
    assert command_preview.load_problems(folder) == []
    assert workflow_agent.split_settings(folder).subject_hints == "a book"


def test_no_workflow_settings_file_keeps_the_built_in_prompt_and_digest(tmp_path):
    folder = _workflow(tmp_path / "none", None)
    assert workflow_agent.split_settings(folder) == workflow_agent.SplitSettings()
    assert workflow_agent.split_prompt(folder) == {"digest": BUILT_IN_DIGEST, "source": "built_in"}
    assert workflow_agent._split_signature(folder).instructions == SplitIntoSubtasks.instructions
    assert workflow_agent.split_examples(folder) == workflow_agent.split_examples(None)
