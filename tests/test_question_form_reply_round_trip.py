"""What the chat's question form sends reaches the server as ordinary chat text (fix-weih.11).

The form of fix-weih.11 replies with one line per answered field: ``"<field name>: <value>"`` on path A, ``"<Label>: <value>"`` on path B. These tests drive
the real suspensions the form answers and ask what the server does with that exact text; no browser.

Path A, a refused planning command input, with an agent model. ``_resume_parameter_loop`` does NOT bind the reply
in code on this path: it puts the reply after the part's request (``"<request>\\n<reply>"``) and runs that part's
typed call again (the part, not the turn: fix-yu3l.4), and the
planner's typed call reads that whole text through the shared parameter extractor (a model, on
LLM_PARAM_EXTRACTION), whose argument the guards then check. There is no ``name=value`` or ``Label:`` binding
in code on this path, so a name prefix is read by the extractor as part of the request, exactly as a plain
bare reply would be. The tests pin what the code does (the exact message the re-run receives, and that the
guards accept either spelling of the value), so a change to that behaviour is seen here.

Path B, an agent ``ask_user``. The reply is the observation the agent resumes with, verbatim, and the open
``ask_user`` entry is completed with it.

Real WorkflowExecutionContext and real agent on the annotated library copy (the ``todo`` fixture of the
approval tests); only the model's step choices and the parameter extractor are scripted, and a model call
that is not scripted fails the test.
"""
from __future__ import annotations

import dspy
import pytest

from fastworkflow import planning_commands as pc
from fastworkflow import workflow_agent
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the library copy, and the guards
    annotated_library, grant_root, no_live_model, no_model_call_for_parameters, todo)
from tests.test_pending_prompt_description import (  # noqa: F401
    REAL_BUILD_QUERY, _command, _last_artifacts, no_draft_extraction)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import script_matcher

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

#: What the form sends for the library's ``book_by_title`` input ``title`` (path A sends the schema field name, not the pretty label).
REPLY = "title: Dune"
REQUEST = "who wrote that book"


@pytest.fixture(autouse=True)
def no_reply_intent_model(monkeypatch):
    """The reply's intent is read by Jev when a key is set; here it never is, so the reply is an answer."""
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda *args, **kwargs: None)


def _suspend_on_refusal(todo, monkeypatch):
    """The refused planning command suspension of test_pending_prompt_description, up to the open question."""
    command = _command()
    with pytest.raises(pc.PlanningCommandInputError) as refused:
        pc.check_arguments(command, {p.name: None for p in command.params}, REQUEST)

    def refuse(*args, **kwargs):
        raise refused.value

    script_matcher(monkeypatch, lambda text: {command.name: 0.9, "none": 0.0})
    monkeypatch.setattr(workflow_agent, "typed_planning_call", refuse)
    unused_lm = dspy.LM("openai/unused-planner", api_base="http://127.0.0.1:9/v1", api_key="unused", num_retries=0)
    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps",
                        lambda *args, **kwargs: REAL_BUILD_QUERY(*args, **{**kwargs, "planner_lm": unused_lm}))
    out = todo.ctx.process_turn(REQUEST)
    assert out.status is TurnStatus.AWAITING_USER
    assert _last_artifacts(out)["question_kind"] == "planning_parameter"
    return command, out


def test_path_a_the_form_reply_is_appended_to_the_request_and_not_parsed(todo, monkeypatch):
    command, out = _suspend_on_refusal(todo, monkeypatch)
    assert _last_artifacts(out)["parameter_request"]["missing"] == ["title"]

    seen = {}

    def call_agent(query, planner_user_query, **kwargs):
        seen["message"], seen["planner_message"], seen["planning_name"] = query, planner_user_query, kwargs.get(
            "planning_name")
        return dspy.Prediction(final_answer="done", exhausted=False)

    monkeypatch.setattr(todo.ctx, "_call_agent_for_query", call_agent)
    monkeypatch.setattr(todo.ctx, "_run_agent", lambda message: pytest.fail("the turn was run again from the top"))
    todo.ctx.process_turn(REPLY)

    # The part's call receives the request, a newline, and the form's text as typed: the label prefix is in it. The
    # command is the one that asked, so the matcher is not asked again.
    assert seen["message"] == f"{REQUEST}\n{REPLY}" == seen["planner_message"]
    assert seen["planning_name"] == command.name
    # The question's exchange is completed with the same text.
    entry = next(o for o in todo.ctx._turn_outputs if o.is_ask_user)
    assert entry.command_response.success is True
    assert entry.user_reply == REPLY


def test_path_a_the_extractor_reads_the_whole_message_and_the_guards_accept_the_clean_value(todo, monkeypatch):
    """The only reader of the reply is the extractor, called on the message the re-run builds. A scripted extractor
    stands in for the model: it returns what a careful one would, the text after the label."""
    command, _ = _suspend_on_refusal(todo, monkeypatch)
    from fastworkflow.utils.signatures import InputForParamExtraction

    queries = []

    def extract(self, model_class=None, name=None, folder=None):
        queries.append(self.command)
        return model_class.model_construct(title=self.command.rsplit("title:", 1)[1].strip())

    monkeypatch.setattr(InputForParamExtraction, "extract_parameters", extract)
    message = f"{REQUEST}\n{REPLY}"
    raw = pc.extracted_arguments(command, message)
    assert queries == [message]
    assert raw == {"title": "Dune"}
    args, _notes = pc.check_arguments(command, raw, message)
    assert args["title"] == "Dune"


def test_path_a_nothing_in_code_strips_a_label_a_careless_extractor_copies(todo, monkeypatch):
    """The strict check of a marked str argument requires it to be written in the request at word boundaries (an
    unmarked one is not compared with the request at all), and the label line is part of the request, so an extractor
    that copies the whole line is accepted too: the code does not protect the value from its label. This is why the finding is a risk of the extractor, not a parse the form can rely on."""
    command = _command()
    message = f"{REQUEST}\n{REPLY}"
    args, _notes = pc.check_arguments(command, {"title": REPLY}, message)
    assert args["title"] == REPLY
    bare, _ = pc.check_arguments(command, {"title": "Dune"}, f"{REQUEST}\nDune")
    assert bare["title"] == "Dune"


def test_path_b_a_multiline_form_reply_reaches_the_agent_verbatim_and_completes_the_entry(todo, monkeypatch):
    todo.script.steps = [("ask_user", {"clarification_request": "Which order, and which items?"})]
    out = todo.ctx.process_turn("exchange my items")
    assert out.status is TurnStatus.AWAITING_USER
    assert _last_artifacts(out)["question_kind"] == "agent"

    reply = ("Order: #W1994898\nItems to exchange: 1008292230, 4983901480\n"
             "Payment method: credit_card_4421890")
    trajectories = []
    agent = todo.ctx._workflow_tool_agent
    scripted = agent._call_with_potential_trajectory_truncation

    def recording(module, trajectory, **inputs):
        trajectories.append(dict(trajectory))
        return scripted(module, trajectory, **inputs)

    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", recording)
    todo.script.steps = [("finish", {})]
    done = todo.ctx.process_turn(reply)
    assert done.status is TurnStatus.COMPLETED

    # The reply is the observation of the ask_user call, whole and in order (the planner stub returns it as is).
    observations = [v for t in trajectories for k, v in t.items() if k.startswith("observation_")]
    assert reply in observations
    entry = next(o for o in done.command_outputs if o.is_ask_user)
    assert entry.command_response.success is True
    assert entry.command_response.response == reply
    assert entry.user_reply == reply
    assert entry.question == "Which order, and which items?"
