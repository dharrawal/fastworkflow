"""THE intent loop (bd fix-o1lg S7): the CME (the deterministic path) and the plan path (``plan_metadata_extraction/resolution.py``) decide
a request with ONE function (``intent_loop.resolve``) and read the reply to "which of these is meant?" with ONE reader
(``intent_loop.read`` / ``answer``), keeping the pending question as ONE piece of plain data and writing a settled
decision through ONE learning hook.

Real components throughout (the todo workflow copy with a live item -> list -> manager chain, the real CME workflow, the real
executor). The only doubles are the classifier (placed behind the trainer's ``CommandRouter`` name, with a head marker on
disk for each scope) and the remote decision model (``script_matcher``), and every model call is refused otherwise.

The differences that remain between the two paths are the only ones, and each is pinned here (or named, where an existing
test already pins it): ``core`` commands (CME only), one candidate (the plan path binds it and the review confirms; the CME
asks "Did you mean the command X?"), and what follows two unreadable replies (the plan path drops the part; the CME ends
the question with the ``what_can_i_do`` listing).
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

import fastworkflow
from fastworkflow.intent import intent_clarification, intent_detector, intent_loop
from fastworkflow.intent import router as intent_router
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow._workflows.command_metadata_extraction.intent_detection import CommandNamePrediction
from fastworkflow._workflows.plan_metadata_extraction import resolution
from tests.conftest import script_matcher
from tests.todo_list_workflow.application.todo_item import TodoItem
from tests.todo_list_workflow.application.todo_list import TodoList
from tests.todo_list_workflow.application.todo_manager import TodoListManager

TODO = os.path.join(os.path.dirname(__file__), "todo_list_workflow")


def _heads(app, *contexts: str) -> None:
    """A trained-looking head for each context: the router double answers for it (a scope with no head has no local model)."""
    for context in contexts:
        folder = intent_detector.head_folder(app.folderpath, context)
        os.makedirs(os.path.join(folder, "tinymodel.pth"), exist_ok=True)
        for name in ("label_encoder.pkl", "threshold.json"):
            with open(os.path.join(folder, name), "w") as marker:
                marker.write("{}")


def _router_by_context(labels_by_context: dict, asked: list):
    class ByContextRouter:
        def __init__(self, model_artifact_path):
            self.context = os.path.basename(model_artifact_path)
            self.modelpipeline = None

        def predict(self, command):
            return self.predict_with_details(command)[0]

        def predict_with_details(self, command):
            asked.append(self.context)
            return list(labels_by_context[self.context]), {}

    return ByContextRouter


def _stand(tmp_path, *, rooted_at_manager: bool = True):
    """An app workflow on an item of a list (of a manager when *rooted_at_manager*), and the real CME workflow."""
    folder = tmp_path / "todo_list_workflow"
    shutil.copytree(TODO, folder, ignore=shutil.ignore_patterns(
        "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    manager = TodoListManager(file_path=str(tmp_path / "lists.json"), load_from_file=False)
    todo_list = TodoList(1, "groceries")
    item = TodoItem(1, "milk")
    item.parent = todo_list
    todo_list.parent = manager if rooted_at_manager else None
    app = fastworkflow.Workflow.create(workflow_folderpath=str(folder), workflow_id_str=f"intent-loop-{tmp_path.name}")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id, workflow_context={"app_workflow": app})
    app.root = manager if rooted_at_manager else todo_list
    app.record = item
    return app, cme, item, todo_list, manager


@pytest.fixture
def chain(tmp_path, setup_test_environment):
    return _stand(tmp_path)


@pytest.fixture
def chain_without_the_manager(tmp_path, setup_test_environment):
    return _stand(tmp_path, rooted_at_manager=False)


# ======================================================================================================================
# ONE decision: the same text is decided alike on both paths
# ======================================================================================================================

@pytest.mark.usefixtures("no_planner_key")
@pytest.mark.parametrize("text, decided_by", [
    ("create_todo_list my list", "exact_name"),          # a command of the root scope, found from the item
    ("get_all_childrn", "fuzzy_name"),                    # a misspelt name: the plan path gains this layer
])
def test_both_paths_decide_a_name_alike(chain, monkeypatch, text, decided_by):
    app, cme, *_ = chain
    monkeypatch.setattr(intent_router, "CommandRouter", _router_by_context({}, []))
    plan_path = resolution.part_intent(app, text)
    cme_path = CommandNamePrediction(cme).predict(text)
    assert (plan_path.outcome, plan_path.decided_by) == ("match", decided_by)
    for each in ("outcome", "command", "kind", "decided_by", "context"):
        assert getattr(plan_path, each) == getattr(cme_path, each), each
    assert plan_path.record is cme_path.record


@pytest.mark.usefixtures("no_planner_key")
@pytest.mark.parametrize("text", ["get_all", "get", "g", "get all of them please"])
def test_a_prefix_of_a_command_name_is_not_a_name_on_either_path(chain, monkeypatch, text):
    """bd fix-ftdl.9: the fuzzy name layer compares the WHOLE request with each whole name, so a request that only
    begins like a name (``get_all`` for ``get_all_children``) is not the user naming it: the chain decides."""
    app, cme, *_ = chain
    monkeypatch.setattr(intent_router, "CommandRouter", _router_by_context({}, []))
    for decided in (resolution.part_intent(app, text), CommandNamePrediction(cme).predict(text)):
        assert decided.decided_by not in (intent_loop.BY_EXACT_NAME, intent_loop.BY_FUZZY_NAME), (text, decided)


def test_a_fuzzy_name_is_not_a_named_decider():
    """A misspelt name is the closest command, not the user naming it: a command that changes data found that way is
    confirmed (``unconfirmed_pick``), and one that changes nothing still runs directly (the fast path)."""
    assert intent_loop.BY_FUZZY_NAME not in intent_loop.NAMED_DECIDERS
    assert intent_loop.BY_FUZZY_NAME in intent_loop.FAST_PATH_DECIDERS


@pytest.mark.usefixtures("no_planner_key")
def test_both_paths_refuse_a_foreign_name_with_the_same_hint_and_ask_no_model(chain_without_the_manager, monkeypatch):
    """The foreign-name guard is shared: the plan path gains it (a known name whose owner is not on the path is not sent
    to a model)."""
    app, cme, *_ = chain_without_the_manager
    asked: list = []
    monkeypatch.setattr(intent_router, "CommandRouter", _router_by_context({}, asked))
    plan_path = resolution.part_intent(app, "create_todo_list my list")
    cme_path = CommandNamePrediction(cme).predict("create_todo_list my list")
    assert plan_path.command is None and plan_path.decided_by == "foreign_name"
    assert plan_path.hint == cme_path.hint and "TodoListManager context" in plan_path.hint
    assert plan_path.owners == cme_path.owners == ("TodoListManager",)
    assert asked == []


@pytest.mark.usefixtures("no_planner_key")
def test_the_local_model_decides_over_the_whole_walk_before_any_model_is_asked(chain, monkeypatch):
    """A scope the local model is unsure of does not send the request to the remote model while a farther scope on the walk
    is confident: ``local over the whole walk first, then the chain``, on both paths."""
    app, cme, *_ = chain
    _heads(app, "TodoItem", "TodoList", "TodoListManager")
    asked: list = []
    monkeypatch.setattr(intent_router, "CommandRouter", _router_by_context(
        {"TodoItem": ["wildcard"], "TodoList": ["TodoList/get_all_children"], "TodoListManager": ["wildcard"]}, asked))
    remote = script_matcher(monkeypatch, {})          # a text the remote model is asked about would fail the call

    for decide in (lambda: resolution.part_intent(app, "show me the children"),
                   lambda: CommandNamePrediction(cme).predict("show me the children")):
        intent = decide()
        assert (intent.command, intent.decided_by, intent.context) == ("TodoList/get_all_children", "classifier", "TodoList")
    assert remote == [], "the remote model was not asked"


def test_the_remote_model_is_never_offered_a_command_that_only_enters_a_context_on_either_path(chain, monkeypatch):
    app, cme, *_ = chain
    entry = "TodoListManager/get_todo_list"
    monkeypatch.setattr(intent_loop, "_entry_only", lambda workflow: frozenset({entry}))
    monkeypatch.setattr(intent_detector, "matcher_available", lambda: True)
    monkeypatch.setattr(intent_detector, "has_head", lambda folder: False)
    offered: list = []

    def answer(folder, scope_path, utterance, exclude=frozenset()):
        offered.append({c.name for _, group in intent_detector._matcher_groups(folder, scope_path, exclude)
                        for c in group})
        return {"scores": {"none": 1.0}, "model": "scripted", "latency_ms": 0.0, "calls": 1}

    monkeypatch.setattr(intent_detector, "_matcher_distribution", answer)
    resolution.part_intent(app, "get the list with id 3")
    CommandNamePrediction(cme).predict("fetch the list that has id 4")      # another text: no memo is shared
    assert len(offered) == 2 and all(entry not in names and names for names in offered)


def test_one_hook_records_a_settled_decision_from_both_paths(chain, monkeypatch):
    app, cme, *_ = chain
    written: list = []
    from fastworkflow.intent import intent_learning

    monkeypatch.setattr(intent_learning, "record", lambda folder, **row: written.append(row) or True)
    intent = CommandNamePrediction(cme).predict("create_todo_list")
    wildcard._settle(app, intent, "create_todo_list")                      # the CME's call
    resolution.record_part(app, "create_todo_list", intent.command, resolution.how_settled(intent.decided_by),
                           intent.evidence)                                 # the plan path's call
    assert [(row["command"], row["how"], row["context"]) for row in written] == [
        ("TodoListManager/create_todo_list", "name", "TodoListManager")] * 2


# ======================================================================================================================
# ONE reader
# ======================================================================================================================

NAMES = ["Library/find_book", "Library/find_member", "Record/add_note"]


def test_a_number_picks_on_both_paths_and_none_and_cancel_are_answers():
    assert intent_loop.read("2", NAMES).name == "Library/find_member"
    assert intent_loop.read("none of these", NAMES).kind == "none"
    assert intent_loop.read("cancel", NAMES).kind == "abort"
    assert intent_loop.read("7", NAMES).kind == "unclear"


def test_a_fuzzy_tie_is_unclear_never_the_first():
    assert intent_loop.read("find", ["Library/find_book", "Library/find_member"]).kind == "unclear"
    assert intent_loop.read("find_bok", ["Library/find_book", "Library/find_member"]).name == "Library/find_book"


@pytest.mark.parametrize("reply", ["a", "ad", "add", "find_m", "Library/f", "addnote please"])
def test_a_prefix_of_a_name_is_not_a_misspelt_name(reply):
    """bd fix-ftdl.9: a reply is compared with each WHOLE name (and its leaf), so only a misspelling of the whole name
    picks; the opening letters of a name do not."""
    assert intent_loop.read(reply, NAMES).kind == "unclear"


def test_a_misspelt_whole_name_still_picks():
    assert intent_loop.read("add_notte", NAMES).name == "Record/add_note"
    assert intent_loop.read("find membr", NAMES).name == "Library/find_member"
    assert intent_loop.read("Library/find_bok", NAMES).name == "Library/find_book"


@pytest.mark.usefixtures("setup_test_environment")
@pytest.mark.parametrize("reply", ["Terminator", "Change", "Wrong", "Never", "endless", "none of this"])
def test_the_core_aliases_match_a_whole_reply_only(reply):
    """The aliases (abort, you_misunderstood, what_can_i_do) are fixed phrases: a reply that only begins like one is not
    it (bd fix-ftdl.9)."""
    assert intent_loop.read(reply, NAMES, aliases=intent_loop.core_aliases()).kind == "unclear"


def test_a_reply_whose_first_word_names_a_candidate_returns_the_pick_and_the_rest():
    reading = intent_loop.read("find_member Ann Lee 2", NAMES)
    assert (reading.kind, reading.name, reading.rest) == ("pick", "Library/find_member", "Ann Lee 2")
    assert intent_loop.read("find_member", NAMES).rest == ""


def test_the_plain_utterances_of_the_core_commands_are_aliases_of_the_one_reader():
    aliases = intent_loop.core_aliases()
    assert intent_loop.read("abort", NAMES, aliases=aliases).kind == "abort"
    assert intent_loop.read("That is not what I meant", NAMES, aliases=aliases).kind == "none"
    assert intent_loop.read("what can i do", NAMES, aliases=aliases).kind == "list"
    # a candidate's name wins over an alias, and without aliases none of them is an answer
    assert intent_loop.read("find_book", NAMES, aliases=aliases).name == "Library/find_book"
    assert intent_loop.read("what can i do", NAMES).kind == "unclear"


def test_one_candidate_is_answered_by_yes():
    assert intent_loop.read("yes", ["Library/find_book"]).name == "Library/find_book"
    assert intent_loop.read("no", ["Library/find_book"]).kind == "none"


def test_an_unreadable_reply_is_asked_once_more_then_exhausted_on_both_paths():
    record = intent_loop.open_question("find it", [{"name": n, "description": "", "closest": True} for n in NAMES])
    intent_loop.ask(record)
    first = intent_loop.answer(record, "hmm")
    assert first.kind == intent_loop.ASK_AGAIN and first.question.startswith(intent_clarification.RETRY)
    assert intent_loop.answer(record, "hmm").kind == intent_loop.EXHAUSTED


# ======================================================================================================================
# ONE question: the client shape and the state
# ======================================================================================================================

def test_the_question_text_and_data_are_one_function_and_one_shape():
    choices = [{"name": n, "description": "d", "closest": True} for n in NAMES[:2]]
    record = intent_loop.open_question("find it", choices, sub_task=2, ambiguous=True)
    text = intent_loop.ask(record)
    assert text.startswith("More than one command fits part 2: 'find it'. Which one is meant?\n1. Library/find_book: d")
    assert intent_loop.prompt_artifacts(record) == {
        "question_kind": "plan_choice",
        "plan_choice": {"request": "find it", "sub_task": 2, "choices": choices, "none_reply": "none",
                        "decline": "cancel", "ambiguous": True, "misunderstood": False}}
    one = intent_loop.open_question("find it", choices[:1], ambiguous=True)
    assert intent_loop.ask(one).startswith("Did you mean the command 'find_book'?")
    assert json.loads(json.dumps(record)) == record


def test_the_cme_pending_question_is_one_json_record_saved_with_the_session(tmp_path, setup_test_environment):
    """The CME keeps the which-command question under ``intent_loop`` in its context: plain data, saved whole with the
    session and restored on the next process (no file is written for it)."""
    from fastworkflow.engine.workflow_execution_context import WorkflowExecutionContext

    folder = tmp_path / "todo_list_workflow"
    shutil.copytree(TODO, folder, ignore=shutil.ignore_patterns(
        "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"))
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key="intent-loop-save")
    ctx.bind_app_workflow(fastworkflow.Workflow.create(str(folder), workflow_id_str="intent-loop-save"))
    record = intent_loop.open_question(
        "find it", [{"name": "TodoListManager/create_todo_list", "description": "", "closest": True}],
        ambiguous=True, listing_hint=True)
    record["evidence"] = {"head": None, "remote": None}
    intent_loop.ask(record)
    context = ctx._cme_workflow.context
    intent_loop.store(context, record)
    context["command"] = "find it"

    blob = json.loads(json.dumps(ctx.serialize_state(channel_id="intent-loop-save")))
    ctx.close()
    assert blob["waiting"]["kind"] == "intent" and blob["waiting"]["payload"]["record"] == record
    assert blob["waiting"]["payload"]["command"] == "find it"

    restored = WorkflowExecutionContext(run_as_agent=False, session_key="intent-loop-save")
    restored.bind_app_workflow(fastworkflow.Workflow.create(str(folder), workflow_id_str="intent-loop-save"))
    restored.apply_serialized_state(blob)
    assert intent_loop.pending_record(restored._cme_workflow.context) == record
    assert restored.waiting.kind == "intent"
    assert restored._cme_workflow.context["command"] == "find it"
    restored.close()
    assert not list(tmp_path.rglob("cache*.sqlite3"))


# ======================================================================================================================
# The differences that remain
# ======================================================================================================================

@pytest.mark.usefixtures("no_planner_key")
def test_difference_1_core_commands_are_answerable_on_the_cme_only(chain, monkeypatch):
    app, cme, *_ = chain
    monkeypatch.setattr(intent_router, "CommandRouter", _router_by_context({}, []))
    for typed in ("what_can_i_do", "abort", "you_misunderstood", "go_up"):
        on_the_cme = intent_loop.resolve(app, typed, core=True)
        assert (on_the_cme.outcome, on_the_cme.kind) == ("match", "core"), typed
        on_a_plan = intent_loop.resolve(app, typed, core=False)
        assert on_a_plan.kind != "core" and on_a_plan.command is None, typed
        assert resolution.part_intent(app, typed).kind != "core"
    # the plain utterances of you_misunderstood are the CME's name layer, not a plan part's
    assert intent_loop.resolve(app, "That is not what I meant", core=True).command == "ErrorCorrection/you_misunderstood"
    assert intent_loop.resolve(app, "That is not what I meant", core=False).command is None


def test_difference_2_one_candidate_the_cme_asks_did_you_mean_and_yes_picks_it(chain, monkeypatch):
    """The CME has no review, so a single candidate is confirmed by a question; the plan path binds it and the turn's one
    review confirms (``test_an_ambiguity_with_one_candidate_is_not_asked_but_bound_and_reviewed``)."""
    app, cme, *_ = chain
    one = intent_loop.Intent(intent_loop.AMBIGUOUS, decided_by=intent_loop.BY_QUESTION,
                             ambiguous=("TodoListManager/create_todo_list",))
    monkeypatch.setattr(CommandNamePrediction, "predict", lambda self, command: one)
    asked = wildcard.ResponseGenerator()(cme, "make a list", None)
    assert asked.routed is False and "Did you mean the command 'create_todo_list'?" in asked.response
    assert [c["name"] for c in asked.plan_choice["choices"]] == ["TodoListManager/create_todo_list"]
    picked = wildcard._answer_question(cme, app, "yes", CommandNamePrediction(cme))
    assert picked is None and cme.context["command_name"] == "TodoListManager/create_todo_list"
    assert intent_loop.pending_record(cme.context) is None
    assert cme.context["command"] == "make a list"            # no rest in the reply: the request is the parameter text


def test_a_pick_with_more_words_in_the_reply_keeps_the_request_as_the_parameter_text(chain, monkeypatch):
    """A reply answers its question and nothing else (bd fix-ftdl.7.8, D5): the words after the picked name are not the
    command's values on the "/" path, as they are not on the plan path; the request that was asked about is."""
    app, cme, *_ = chain
    two = intent_loop.Intent(intent_loop.AMBIGUOUS, decided_by=intent_loop.BY_QUESTION,
                             ambiguous=("TodoListManager/create_todo_list", "TodoListManager/get_todo_list"))
    monkeypatch.setattr(CommandNamePrediction, "predict", lambda self, command: two)
    wildcard.ResponseGenerator()(cme, "do the list thing", None)
    wildcard._answer_question(cme, app, "get_todo_list 3", CommandNamePrediction(cme))
    assert cme.context["command_name"] == "TodoListManager/get_todo_list"
    assert cme.context["command"] == "do the list thing"


def test_difference_3_after_two_unreadable_replies_the_cme_lists_the_commands(chain, monkeypatch):
    """The plan path drops the part instead (``test_two_unreadable_replies_drop_the_part_on_the_plan_path``)."""
    app, cme, *_ = chain
    two = intent_loop.Intent(intent_loop.AMBIGUOUS, decided_by=intent_loop.BY_QUESTION,
                             ambiguous=("TodoListManager/create_todo_list", "TodoListManager/get_todo_list"))
    monkeypatch.setattr(CommandNamePrediction, "predict", lambda self, command: two)
    wildcard.ResponseGenerator()(cme, "do the list thing", None)
    again = wildcard.ResponseGenerator()(cme, "zzz", None)
    assert again.routed is False and "That did not name one of them." in again.response
    ended = wildcard.ResponseGenerator()(cme, "zzz", None)
    assert ended.handled is True and ended.response.startswith("Workflow Definition:")
    assert intent_loop.pending_record(cme.context) is None


@pytest.mark.usefixtures("initialized_fastworkflow")
def test_two_unreadable_replies_drop_the_part_on_the_plan_path(ambiguous_joke):
    from fastworkflow.core.turn import TurnStatus
    from fastworkflow.engine.workflow_execution_context import NOT_RUN_ARTIFACT
    from tests.test_parameter_loop import HOST, _reply
    from tests.test_round1_gate import JOKE

    env = ambiguous_joke
    env.parts = [JOKE]
    asked = env.ctx.process_turn(JOKE, approval_surface=HOST)
    assert asked.command_outputs[-1].command_response.artifacts["question_kind"] == "plan_choice"
    retried = _reply(env, "hmm?")
    assert retried.status is TurnStatus.AWAITING_USER and "That did not name one of them." in retried.answer
    done = _reply(env, "hmm?")
    assert done.status is TurnStatus.COMPLETED, done.answer
    (stop,) = [stop for stop in done.command_outputs[-1].command_response.artifacts[NOT_RUN_ARTIFACT]
               if stop["command"] is None]
    assert stop["request"] == JOKE


@pytest.mark.parametrize("reply, command_name", [
    ("abort", "ErrorCorrection/abort"), ("never mind", "ErrorCorrection/abort"),
    ("That is not what I meant", "ErrorCorrection/you_misunderstood")])
def test_a_reply_that_is_the_plain_utterance_of_abort_or_you_misunderstood_leaves_a_pending_parameter_question(
        chain, monkeypatch, reply, command_name):
    """While a command's parameters are being asked for, the CME's reply reader still hears these two before it reads a
    value (an agent aborts a refused extraction this way; a person types it)."""
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import park

    app, cme, *_ = chain
    name = "TodoListManager/create_todo_list"
    params = fastworkflow.RoutingRegistry.get_definition(app.folderpath).get_command_class(
        name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
    context = cme.context
    context["command"], context["command_name"] = "create a todo list", name
    park(cme, name, params.model_construct(description="NOT_FOUND"), request_text="create a todo list")
    ran: list = []
    monkeypatch.setattr(wildcard.CommandExecutor, "perform_action", staticmethod(
        lambda workflow, action: ran.append(action.command_name) or fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="ok", success=True))))

    out = wildcard.ResponseGenerator()(cme, reply, None)

    assert ran == [command_name] and out.response == "ok"


# the fixtures of the plan-path tests above
from tests.test_resolve_before_review import ambiguous_joke  # noqa: E402,F401 - fixture
from tests.test_parameter_loop import _offline, no_agent  # noqa: E402,F401 - fixtures the plan-path fixtures use
from tests.test_round1_gate import gated  # noqa: E402,F401
from tests.test_approval_binding import annotated_library, todo  # noqa: E402,F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: E402,F401


@pytest.mark.usefixtures("no_planner_key")
def test_a_bare_no_as_a_new_message_is_decided_like_any_request_and_never_runs_a_command(chain, monkeypatch):
    """With no question pending, a bare "no" is a request: the reply reader (whose none-words now need candidates) is not
    consulted at all. It is out of scope, so the CME opens the misunderstanding question (what ``_detect`` did before)."""
    app, cme, *_ = chain
    monkeypatch.setattr(intent_router, "CommandRouter", _router_by_context({}, []))
    ran: list = []
    monkeypatch.setattr(wildcard.CommandExecutor, "perform_action", staticmethod(
        lambda workflow, action: ran.append(action.command_name) or fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response="listing", success=True))))
    out = wildcard.ResponseGenerator()(cme, "no", None)
    assert ran == ["ErrorCorrection/you_misunderstood"] and out.response == "listing"
    assert intent_loop.pending_record(cme.context) is not None


def test_the_which_command_question_still_reads_none_no_and_neither_as_none_of_these():
    for word in ("none", "no", "neither", "n", "something else"):
        assert intent_loop.read(word, NAMES).kind == "none", word
