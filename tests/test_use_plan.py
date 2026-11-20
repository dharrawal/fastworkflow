"""``use_plan``: a client's choice of the planning command for one request (fix-1ma8.6).

The request is one part with the plan pinned (no split, no matcher call), and from there it follows the path of a part the
matcher picked that plan for: the round-1 draft for that plan (arguments from the shared parameter extractor), the review
under the usual rules, then the run. The choice names the plan, not its arguments, so a required value the draft leaves
unbound is shown in the review and, when it is still unbound after the review, asked for.

Driven as tests/test_no_agent_review.py is: the real agent context on the annotated library copy with its planning
commands, no agent model. The scripted seams are the planner's split, the shared extractor's model, and the matcher
(``resolution.part_intent``), which a pinned turn must never ask. No test here can reach a model: every model call and every
network connection is refused (``_offline``)."""
from __future__ import annotations

import socket

import pytest

from fastworkflow import approval_binding, no_agent_mode
from fastworkflow.approval_binding import Surface
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import NOT_RUN_ARTIFACT, UsePlanError, WorkflowExecutionContext
from tests.scripted_extractor import ScriptedExtractor, script_extractor
from tests.test_approval_binding import annotated_library, todo  # noqa: F401 - the real agent on the library copy
from tests.test_round1_gate import EMMA, JOKE, _split_into, gated  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401
from tests.conftest import script_matcher

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

PLAN = "Library/book_by_title"
HOST = Surface(kind="host", verified=True, uid="user-1")
UNVERIFIED = Surface(kind="host", verified=False, uid=None)
REQUEST = "tell Ann a joke about cats"
JOKE_EMMA = "tell Ann a joke about Emma"
HAIKU = "write me a haiku about the weather in Paris"
GIVEN = {"reasoning": "the title", "title": "Emma"}
NONE_GIVEN = {"reasoning": "no title", "title": "NOT_FOUND"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """The earlier version of this file reached the configured provider once (a stray ``monkeypatch.undo()`` took the
    no-agent fixture away). Now nothing can: litellm's calls and every non-local socket connection raise."""
    import litellm

    def refuse(*args, **kwargs):
        raise AssertionError("a model call or a network connection was attempted from a use_plan test")

    monkeypatch.setattr(litellm, "completion", refuse)
    monkeypatch.setattr(litellm, "acompletion", refuse)
    real_connect = socket.socket.connect

    def connect(self, address):
        if self.family == socket.AF_UNIX or (isinstance(address, tuple) and str(address[0]) in ("127.0.0.1", "::1", "localhost")):
            return real_connect(self, address)
        refuse()

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture(autouse=True)
def _drafting_on(monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "on")


NO_PARAMETERS = "branch_names"
NO_PARAMETERS_SOURCE = '''"""branch_names: the library's branches, a root-level planning command with no parameters."""
from fastworkflow.planning import run

from .Library.list_branches import Signature as ListBranches


class Signature:
    """Name the library's branches."""

    plain_utterances = ["which branches do we have"]


class PlanGenerator:
    def branches(self) -> ListBranches.Output:
        return [run("list_branches")]

    def __result__(self, branches):
        return "Branches: " + ", ".join(branches.names) + "."
'''


@pytest.fixture(autouse=True)
def _root_plan_added(annotated_library):  # noqa: F811
    """A root-level planning command with no parameters, in the library copy before the workflow is built."""
    (annotated_library / "_commands" / f"{NO_PARAMETERS}.py").write_text(NO_PARAMETERS_SOURCE, encoding="utf-8")


@pytest.fixture(autouse=True)
def _no_agent(_root_plan_added, gated, no_agent_model, no_planner_key, monkeypatch):
    def never(self):
        raise AssertionError("the agent LM was built in no-agent mode")

    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context", never)
    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps",
                        lambda user_query, session, **kwargs: setattr(session, "_initial_plan_text", "") or user_query)


class _ByReply(ScriptedExtractor):
    """The shared extractor's model: the title is found only once the text it reads ends with the user's own reply."""

    def forward(self, prompt=None, messages=None, **kwargs):
        text = "\n".join(str(m.get("content", "")) for m in (messages or [])) or str(prompt)
        self.answers = iter([{"reasoning": "scripted", **(GIVEN if ("title: Emma" in text or "cats\nEmma" in text) else NONE_GIVEN)}])
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def extractor_finds_title_only_in_a_reply(monkeypatch) -> ScriptedExtractor:
    from fastworkflow.utils import signatures

    lm = _ByReply([])
    original = signatures.dspy_utils.get_lm
    monkeypatch.setattr(signatures.dspy_utils, "get_lm",
                        lambda name, *a, **k: lm if name == "LLM_PARAM_EXTRACTION" else original(name, *a, **k))
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    return lm


class Matcher:
    """The planning-command match: scripted to pick nothing, or to pick ``PLAN`` for a text; every ask is recorded."""

    def __init__(self, pick: str | None = None):
        self.pick, self.asked = pick, []

    def __call__(self, text):
        """The remote decision model's answer for *text*: ``pick`` confidently, or no answer at all."""
        return None if self.pick is None else {self.pick: 0.9, "none": 0.0}

    def install(self, monkeypatch) -> "Matcher":
        self.asked = script_matcher(monkeypatch, self)
        return self


@pytest.fixture
def matcher(monkeypatch):
    return Matcher().install(monkeypatch)


def _reply(gated, message):
    # The host is given its token once, when a review or an approval opens; every reply carries that one.
    if getattr(gated, "token", None) is None:
        delivery = approval_binding.take_delivery(gated.wf)
        gated.token = delivery["token"] if delivery else None
    token = gated.token
    return gated.ctx.process_turn(message, approval_surface=HOST, approval_token=token)


def _request(out) -> dict:
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") == "planning_parameter" and "review" not in artifacts, artifacts
    return artifacts["parameter_request"]


def _ran_with(gated) -> dict:
    return gated.ctx._turn_subtask_bindings[-1]["ran"]


def _review(out):
    artifacts = out.command_outputs[-1].command_response.artifacts
    assert artifacts.get("question_kind") == "review", artifacts
    return artifacts["review"]


def _ran(out) -> bool:
    return any(o.command_name.startswith("Library/") for o in out.command_outputs)


def _stops(out):
    return out.command_outputs[-1].command_response.artifacts.get(NOT_RUN_ARTIFACT)


# ---------------------------------------------------------------- a pinned plan asks for its parameters, not for a review

def test_a_chosen_plan_shows_its_parameters_to_confirm_and_never_the_review(gated, matcher, monkeypatch):
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda *a, **k: pytest.fail("a chosen plan's message is one part: it is not split"))
    lm = script_extractor(monkeypatch, *[GIVEN] * 12)
    out = gated.ctx.process_turn(EMMA, use_plan=PLAN, approval_surface=HOST)

    assert out.status is TurnStatus.AWAITING_USER
    request = _request(out)                      # question_kind planning_parameter, no review artifact
    assert request["command"] == PLAN and request["description"].startswith("Show who wrote a book")
    assert request["confirm"] is True and request["missing"] == [] and request["invalid"] == []
    assert request["values"] == {"title": "Emma"}
    assert [(p["name"], p["required"], p["value"], p["state"]) for p in request["parameters"]] == [
        ("title", True, "Emma", None)]
    assert "Sub-tasks" not in out.answer and EMMA not in out.answer and "step" not in out.answer.lower().split("reply")[0]
    assert "- title (required): Emma" in out.answer and "yes to run with the values shown" in out.answer
    assert matcher.asked == [] and len(lm.prompts) >= 1

    done = _reply(gated, "yes")
    assert done.status is TurnStatus.COMPLETED and _ran(done) and not _stops(done)
    assert _ran_with(gated) == {"title": "Emma"} and matcher.asked == []


def test_a_changed_value_runs_with_the_change(gated, matcher, monkeypatch):
    script_extractor(monkeypatch, *[GIVEN] * 12)
    gated.ctx.process_turn(EMMA, use_plan=PLAN, approval_surface=HOST)
    done = _reply(gated, "title: Dune")

    assert done.status is TurnStatus.COMPLETED and _ran(done)
    assert _ran_with(gated) == {"title": "Dune"} and matcher.asked == []


def test_a_missing_required_value_is_marked_missing_in_the_same_form_and_a_bare_reply_fills_it(gated, matcher, monkeypatch):
    extractor_finds_title_only_in_a_reply(monkeypatch)
    out = gated.ctx.process_turn(JOKE, use_plan=PLAN, approval_surface=HOST)

    request = _request(out)
    assert request["missing"] == ["title"] and request["values"] == {}
    assert [(p["name"], p["required"], p["value"], p["state"]) for p in request["parameters"]] == [
        ("title", True, None, "missing")]
    assert "- title (required): not given" in out.answer
    done = _reply(gated, "Emma")
    assert done.status is TurnStatus.COMPLETED and _ran(done) and _ran_with(gated) == {"title": "Emma"}
    assert gated.ctx._turn_use_plan == PLAN and matcher.asked == []


def test_a_reply_that_is_not_a_value_yes_or_cancel_is_asked_again_once_then_the_part_is_not_run(gated, matcher, monkeypatch):
    script_extractor(monkeypatch, *[GIVEN] * 12)
    gated.ctx.process_turn(EMMA, use_plan=PLAN, approval_surface=HOST)
    again = _reply(gated, "hmm, let me think")

    assert again.status is TurnStatus.AWAITING_USER and again.answer.startswith("That did not give a value I could use.")
    assert _request(again)["values"] == {"title": "Emma"}
    stopped = _reply(gated, "still thinking")
    assert stopped.status is TurnStatus.COMPLETED
    (stop,) = _stops(stopped)
    assert stop["command"] == PLAN and "not confirmed" in stop["reason"]


def test_cancel_declines_the_chosen_plan(gated, matcher, monkeypatch):
    script_extractor(monkeypatch, *[GIVEN] * 12)
    gated.ctx.process_turn(EMMA, use_plan=PLAN, approval_surface=HOST)
    out = _reply(gated, "cancel")

    assert out.status is TurnStatus.COMPLETED and not _ran(out)
    (stop,) = _stops(out)
    assert stop["command"] == PLAN and "declined" in stop["reason"]


def test_a_plan_with_no_parameters_runs_without_a_question(gated, matcher):
    out = gated.ctx.process_turn("which branches do we have", use_plan=NO_PARAMETERS, approval_surface=HOST)

    assert out.status is TurnStatus.COMPLETED and "Branches:" in out.answer
    assert not _stops(out) and matcher.asked == []


def test_an_unattended_turn_runs_as_drafted_and_cannot_be_asked_about_what_is_missing(gated, matcher, monkeypatch):
    script_extractor(monkeypatch, *[GIVEN] * 12)
    ran = gated.ctx.process_turn(EMMA, use_plan=PLAN, approval_surface=HOST, unattended=True)
    assert ran.status is TurnStatus.COMPLETED and _ran(ran) and _ran_with(gated) == {"title": "Emma"}


def test_an_unattended_turn_with_a_missing_value_is_not_run_naming_it(gated, matcher, monkeypatch):
    extractor_finds_title_only_in_a_reply(monkeypatch)
    out = gated.ctx.process_turn(JOKE, use_plan=PLAN, approval_surface=HOST, unattended=True)

    assert out.status is TurnStatus.COMPLETED
    (stop,) = _stops(out)
    assert stop["command"] == PLAN and "missing required argument title" in stop["reason"]


# ---------------------------------------------------------------- a matched part keeps its review

def test_a_normally_matched_planning_command_gets_the_same_review(gated, monkeypatch):
    """The path a pinned part must equal: the matcher picks the plan, the draft binds it, the review opens."""
    match = Matcher(PLAN).install(monkeypatch)
    script_extractor(monkeypatch, *[GIVEN] * 12)
    _split_into(monkeypatch, [JOKE_EMMA])
    out = gated.ctx.process_turn(JOKE_EMMA, approval_surface=HOST)

    (item,) = _review(out)["items"]
    assert item["means"] == {"kind": "planning", "name": PLAN}
    assert [(i["name"], i["value"], i["drafted"]) for i in item["inputs"]] == [("title", "Emma", True)]
    assert match.asked


def test_a_matched_part_with_a_required_value_unbound_is_asked_for_it_before_the_review(gated, monkeypatch):
    """Rewritten for bd fix-o1lg S4/S5. Before, the review showed the title unbound, approving it asked for the title,
    and the reply ran the part. Now the title is asked for FIRST; the review then shows it given (nothing missing), and
    approving it runs the part with no further question."""
    Matcher(PLAN).install(monkeypatch)
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks", lambda user_query, **kwargs: [user_query])
    extractor_finds_title_only_in_a_reply(monkeypatch)
    asked = gated.ctx.process_turn(JOKE, approval_surface=HOST)

    assert asked.status is TurnStatus.AWAITING_USER and "Which title should I use?" in asked.answer
    assert asked.command_outputs[-1].command_response.artifacts["question_kind"] == "planning_parameter"
    review = _review(_reply(gated, "Emma"))
    (item,) = review["items"]
    assert [(i["name"], i["value"], i["drafted"]) for i in item["inputs"]] == [("title", "Emma", False)]
    done = _reply(gated, "yes")
    assert done.status is TurnStatus.COMPLETED and _ran(done) and not _stops(done)


# ---------------------------------------------------------------- the choice persists, and is validated

def test_the_choice_and_the_question_are_saved_with_a_pending_parameter_question(gated, matcher, monkeypatch):
    script_extractor(monkeypatch, *[GIVEN] * 12)
    gated.ctx.process_turn(EMMA, use_plan=PLAN, approval_surface=HOST)
    state = gated.ctx.serialize_state(channel_id="c")

    loop = state["waiting"]["payload"]["loop"]
    assert state["use_plan"] == PLAN and "pending" not in loop and loop["confirm"] is True
    assert loop["request_text"] == EMMA and loop["values"] == {"title": "Emma"}
    assert loop["request"]["confirm"] is True and loop["request"]["command"] == PLAN


@pytest.mark.parametrize("name", ["Library/no_such_plan", "book_by_title", "review_subtasks", "find_member"],
                         ids=["unknown", "unqualified", "framework", "plain-command"])
def test_a_name_that_is_not_a_plan_of_the_context_is_refused_before_the_turn_begins(gated, matcher, name):
    with pytest.raises(UsePlanError, match="not a planning command available here"):
        gated.ctx.process_turn(EMMA, use_plan=name)

    assert matcher.asked == [] and gated.ctx._turn_use_plan is None


def test_a_plan_of_a_scope_off_the_lookup_path_is_refused(gated, matcher, monkeypatch):
    """The set is the matcher's: with the library's scope off the path (tests/test_plan_scopes_listing.py has the
    ancestor and root cases that are on it), its plans are refused."""
    from fastworkflow import workflow_agent

    monkeypatch.setattr(workflow_agent, "planning_scopes", lambda workflow: ("Member", ""))
    assert PLAN not in no_agent_mode.plan_names(gated.wf)
    with pytest.raises(UsePlanError, match="not a planning command available here"):
        gated.ctx.process_turn(EMMA, use_plan=PLAN)


def test_a_reply_to_a_waiting_question_cannot_carry_a_plan(gated, matcher, monkeypatch):
    script_extractor(monkeypatch, *[GIVEN] * 12)
    gated.ctx.process_turn(EMMA, use_plan=PLAN, approval_surface=HOST)         # the parameter question is waiting
    with pytest.raises(UsePlanError, match="question is waiting"):
        gated.ctx.process_turn("yes", use_plan=PLAN, approval_surface=HOST)


def test_without_use_plan_nothing_changes(gated, matcher, monkeypatch):
    _split_into(monkeypatch, [JOKE])
    asked = gated.ctx.process_turn(JOKE, approval_surface=UNVERIFIED)

    # Without a pinned plan the part is matched as always. No plan matches it, so an attended turn asks which one is
    # meant (fix-yu3l.5); "none" ends it as the part that is not run.
    assert asked.status is TurnStatus.AWAITING_USER and matcher.asked
    assert asked.command_outputs[-1].command_response.artifacts["question_kind"] == "plan_choice"
    assert gated.ctx._turn_use_plan is None
    out = gated.ctx.process_turn("none", approval_surface=UNVERIFIED)

    assert out.status is TurnStatus.COMPLETED and len(_stops(out)) == 1
    assert gated.ctx._turn_use_plan is None


# ---------------------------------------------------------------- the entry carries the request

def test_a_split_turn_entry_carries_its_own_request_and_a_single_part_its_message(gated, matcher, monkeypatch):
    # Unattended turns: the entry of a part that is not run is the subject (an attended turn asks which plan is meant
    # first, fix-yu3l.5).
    _split_into(monkeypatch, [JOKE, HAIKU])
    out = gated.ctx.process_turn(f"{JOKE} and {HAIKU}", approval_surface=UNVERIFIED, unattended=True)

    first, second = _stops(out)
    assert (first["request"], second["request"]) == (JOKE, HAIKU)

    _split_into(monkeypatch, [JOKE])
    out = gated.ctx.process_turn(JOKE, approval_surface=UNVERIFIED, unattended=True)
    (single,) = _stops(out)
    assert single["request"] == JOKE


def test_an_entry_that_is_not_ad_hoc_has_no_request():
    entry = no_agent_mode.stop_entry(command="Library/x", phase=None, open_step=None, done=[], facts={}, remaining=[],
                                     suggested_plan="", means="planning")
    assert "request" not in entry
