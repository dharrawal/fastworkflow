"""Q1 (ido-z4w): the answer-verification block must reach the main agent.

One wording change, tested offline. ``WorkflowAgentSignature``'s docstring is
what the ReAct loop and the final-answer extractor are both built from
(``fastworkflow/utils/react.py``: ``instr = [signature.instructions]`` for the
react predictor, and ``fallback_signature`` carries ``signature.instructions``
verbatim for ``self.extract``). These assertions pin that the block is present
in *both* prompts, so a run cannot silently lose it. No model call, no server.
"""

import dspy

from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.workflow_agent import WorkflowAgentSignature

Q1_LINES = (
    "Before finishing, verify every claim in your final answer against the data "
    "you retrieved in this turn:",
    "State a fact only if an observation you retrieved supports it, and cite "
    "that observation's O-number.",
    "If retrieved data contradicts a claim in the request, say the request was "
    "wrong and give the evidence.",
    "If a requested item was not retrieved, name it as unresolved instead of "
    "filling it in.",
    "Distinguish claims the retrieved data contradicts from claims you could "
    "not verify.",
)


def _a_tool(query: str) -> str:
    """A stand-in tool so the agent can be constructed offline."""
    return query


def test_signature_instructions_carry_the_block():
    instructions = dspy.ensure_signature(WorkflowAgentSignature).instructions
    for line in Q1_LINES:
        assert line in instructions


def test_react_and_extract_prompts_both_carry_the_block():
    agent = fastWorkflowReAct(WorkflowAgentSignature, tools=[_a_tool], max_iters=2)

    react_instructions = agent.react.signature.instructions
    extract_signature = getattr(agent.extract, "signature", None)
    if extract_signature is None:  # ChainOfThought wraps a Predict
        extract_signature = agent.extract.predict.signature
    extract_instructions = extract_signature.instructions

    for line in Q1_LINES:
        assert line in react_instructions, f"missing from react prompt: {line}"
        assert line in extract_instructions, f"missing from extract prompt: {line}"


def test_the_original_objective_is_unchanged():
    instructions = dspy.ensure_signature(WorkflowAgentSignature).instructions
    assert (
        "Carefully review the user request, then execute the next steps using "
        "available tools for building the final answer." in instructions
    )
    assert (
        "Every user intent must be fully addressed before returning the final "
        "answer." in instructions
    )
