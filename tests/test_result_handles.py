"""Observation compaction: the agent reads a page, evidence keeps the payload.

`fastworkflow/result_handles.py` states the design; this file pins the four
claims that would be worth nothing as prose.

**Off path is byte-identical.** The mechanism is opt-in, and "opt-in" is only
credible if a command that did not opt in produces the same observation, the same
span attributes and the same turn record it produced before the feature existed.
`test_a_command_that_did_not_opt_in_is_byte_identical` runs the same command
twice — once with both framework seams neutered, standing in for the old build —
and diffs what came out. Comparing key SETS would pass while a value changed, so
the comparison is on canonical JSON with only the genuinely per-run fields
(call ids, timestamps, the child-call ledger) removed.

**Evidence is not the agent's context.** This is the whole point, so it is
asserted from both sides: the `fw.command.execute` span and the turn record's
`command_outputs` carry the full 23k-character payload, and the string the ReAct
loop received does not.

**Paging is stable and the cursor is versioned.** A cursor is handed to a model
and can come back many steps later, possibly after a suspend/resume. Pages must
partition the result exactly (no overlap, no gap), and a cursor from another
version must be refused rather than reinterpreted as an offset.

**The store survives suspension.** A resumed turn continues the SAME logical
turn, so its trajectory still cites handles issued before the suspension.

The fake CME hop is the same one `tests/test_span_contract_versioning.py` and
`tests/test_context_handle_capture.py` use: the test workflow ships no trained
intent models, so routing is stood in for while the real
`CommandExecutor.invoke_command` — which mints the call id, opens the span and
writes the store — runs for real.
"""

from __future__ import annotations

import json
import uuid
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow import result_handles, tracing
from fastworkflow.binding_normalizers import CONTROL_ALIAS_RESOLVER_V1
from fastworkflow.capture_policy import is_capture_envelope
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.plan_execution import (
    PlanExecutionArm,
    PlanExecutionBinding,
    PlanExecutionScope,
)
from fastworkflow.result_handles import (
    InvalidResultCursor,
    ResultHandleSpec,
    ResultHandleStore,
    StoredResult,
    UnknownResultHandle,
)
from fastworkflow.workflow_agent import _execute_workflow_query
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

COMMAND_NAME = "TodoListManager/list_todo_lists"

# Shaped like the Gate 4 v4 evidence: `show_holders` returned all 477 holders in
# one 23k-character string, and that string then rode every following prompt.
HOLDERS = tuple(f"identity_{index:03d}  Holder Number {index}" for index in range(477))
FULL_RESPONSE = "477 holder(s).\n" + "\n".join(HOLDERS)


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


class RecordingTraceSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []
        self.turn_records: list[dict] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        self.turn_records.append(record)
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass

    def named(self, name: str) -> list[tracing.Span]:
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def sink() -> RecordingTraceSink:
    return RecordingTraceSink()


def _holder_spec(**overrides) -> ResultHandleSpec:
    fields = {
        "kind": "holders",
        "summary": "477 holder(s) of this permission.",
        "items": list(HOLDERS),
        "ordering": "backend view order (ido_permissiondetail_identity)",
        "total": 477,
        "page_size": 20,
        "classification": "user-text",
    }
    fields.update(overrides)
    return ResultHandleSpec(**fields)


def _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch) -> WorkflowExecutionContext:
    """Agent-mode context on the real todo workflow; only the LLM seams are faked.

    Same seams as `tests/test_turn_result_capture.py::_make_agent_ctx` — the
    planner prelude, the command catalogue and the summary extractor each make a
    provider call this test has no business making.
    """
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"handles-{uuid.uuid4().hex}"
    )
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, with_agent_inputs_and_trajectory=False,
        planning_insights=None, planner_lm=None, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do", lambda session: "commands"
    )
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        ctx,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )
    return ctx


def _cme_hop(monkeypatch, artifacts: dict, response: str = FULL_RESPONSE) -> None:
    """Route every dispatch to one canned output, through the real invoke_command."""

    def hop(cls, workflow, action):
        return fastworkflow.CommandOutput(
            command_name=COMMAND_NAME,
            command_response=fastworkflow.CommandResponse(
                response=response,
                artifacts={**artifacts, "command_handled": True},
            ),
        )

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(hop))


def _run_turn(ctx, command: str = "who holds this permission") -> str:
    """Drive one real agent turn whose single tool call is `command`.

    Returns the observation `_execute_workflow_query` handed the ReAct loop —
    which is the string the trajectory carries and the prompt re-sends.
    """
    seen: list[str] = []

    def fake_forward(**kwargs):
        seen.append(_execute_workflow_query(command, ctx))
        return SimpleNamespace(final_answer="done")

    ctx._workflow_tool_agent = MagicMock(side_effect=fake_forward)
    ctx._intent_clarification_agent = MagicMock()
    ctx.process_turn(command)
    return seen[0]


# ----------------------------------------------------------------------
# Off path: byte-identical
# ----------------------------------------------------------------------

# Per-run by construction: a uuid, a wall clock, or a ledger of ids. Comparing
# them would fail on two runs of the SAME build, which would say nothing.
_PER_RUN_ATTRIBUTES = frozenset(
    {
        tracing.ATTR_COMMAND_CALL_ID,
        tracing.ATTR_PARENT_CALL_ID,
        tracing.ATTR_CHILD_CALLS,
        tracing.ATTR_CONTEXT_BEFORE,
        tracing.ATTR_CONTEXT_AFTER,
        # A minted key carrying a wall clock, and the ids derived from it.
        "turn_key",
        "channel_id",
        "conversation_id",
        "suspended_ms",
    }
)


def _canonical_spans(sink: RecordingTraceSink) -> str:
    return json.dumps(
        [
            [
                span.name,
                {
                    key: value
                    for key, value in sorted(span.attributes.items())
                    if key not in _PER_RUN_ATTRIBUTES
                },
            ]
            for span in sink.spans
        ],
        sort_keys=True,
    )


def _canonical_outputs(ctx) -> str:
    return json.dumps(
        [
            {
                key: value
                for key, value in output.model_dump(mode="json").items()
                if key not in {"command_call_id", "started_at", "duration_ms"}
            }
            for output in ctx._turn_outputs
        ],
        sort_keys=True,
    )


def test_a_command_that_did_not_opt_in_is_byte_identical(
    initialized_fastworkflow, todo_workflow_path, tmp_path, monkeypatch
):
    """The claim the whole mechanism rests on, checked rather than asserted.

    The `disabled` run neuters the two seams this feature adds, which is exactly
    the build that existed before it. If either seam ever touched a command that
    did not declare a handle, the canonical diff below would show it.
    """
    captured: dict[str, tuple[str, str, str]] = {}
    for label in ("disabled", "enabled"):
        with monkeypatch.context() as patch:
            if label == "disabled":
                patch.setattr(
                    result_handles,
                    "store_from_command_output",
                    lambda *a, **k: (None, ()),
                )
                patch.setattr(
                    result_handles, "compact_observation_for", lambda *a, **k: None
                )
            run_sink = RecordingTraceSink()
            ctx = _make_ctx(todo_workflow_path, tmp_path, run_sink, patch)
            # No handle declaration anywhere in the artifacts: the off path.
            _cme_hop(patch, {"rows": 477})
            try:
                observation = _run_turn(ctx)
                captured[label] = (
                    observation,
                    _canonical_spans(run_sink),
                    _canonical_outputs(ctx),
                )
                assert not run_sink.named(tracing.SPAN_COMMAND_EXECUTE)[0].attributes.get(
                    tracing.ATTR_RESULT_HANDLE_ID
                )
            finally:
                with suppress(Exception):
                    ctx.close()

    assert captured["enabled"][0] == captured["disabled"][0] == FULL_RESPONSE
    assert captured["enabled"][1] == captured["disabled"][1]
    assert captured["enabled"][2] == captured["disabled"][2]


def test_an_off_path_command_stores_nothing(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, {"rows": 477})
    try:
        _run_turn(ctx)
        assert len(ctx.result_handles) == 0
    finally:
        with suppress(Exception):
            ctx.close()


# ----------------------------------------------------------------------
# Opt-in round trip
# ----------------------------------------------------------------------


def test_opting_in_compacts_the_observation_and_stores_the_payload(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    try:
        observation = _run_turn(ctx)

        # The observation contract, field by field.
        assert observation.startswith("477 holder(s) of this permission.")
        assert "kind=holders" in observation
        assert "contract=v2" in observation
        assert "total=477 materialized=477 source_complete=true" in observation
        assert "matched=477 matched_complete=true" in observation
        assert "has_more=true" in observation
        assert "next_cursor=v2:20:" in observation
        assert "ordering=backend view order" in observation
        assert "filters=none" in observation

        handle_id = next(iter(result_handles.iter_handles(ctx)))
        assert f"result_handle={handle_id}" in observation

        # The compaction is the point: 20 rows instead of 477.
        assert observation.count("Holder Number") == 20
        assert len(observation) < len(FULL_RESPONSE) / 10

        # The payload is stored whole under the debug (default) profile.
        stored = result_handles.get_result(handle_id, host=ctx)
        assert stored.item_list == list(HOLDERS)
        assert stored.response == FULL_RESPONSE
        assert stored.command_name == COMMAND_NAME
    finally:
        with suppress(Exception):
            ctx.close()


def test_the_handle_is_the_producing_command_call_id(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """Aligned with `plan.py`'s captured binding, which keys on the same id."""
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    try:
        _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))
        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        assert execute.attributes[tracing.ATTR_COMMAND_CALL_ID] == handle_id
        assert ctx._turn_outputs[0].command_call_id == handle_id
    finally:
        with suppress(Exception):
            ctx.close()


def test_pages_partition_the_result_and_the_cursor_walks_it(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec(page_size=100)))
    try:
        _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))

        walked: list[str] = []
        cursor = None
        pages = 0
        while True:
            page = result_handles.fetch_page(handle_id, cursor=cursor, host=ctx)
            walked.extend(page.items)
            pages += 1
            assert page.total == 477
            if not page.has_more:
                assert page.next_cursor is None
                break
            cursor = page.next_cursor

        assert pages == 5  # 477 rows at 100 a page
        assert walked == list(HOLDERS)  # no overlap, no gap, order preserved
    finally:
        with suppress(Exception):
            ctx.close()


def test_backend_total_beyond_materialized_rows_never_claims_completion():
    """The end of the stored list is not the end of the backend population."""
    record = StoredResult(
        handle_id="partial-handle",
        command_name="show_holders",
        kind="holders",
        summary="100 holder(s).",
        ordering="backend order",
        total=100,
        # Even an inconsistent producer claim cannot override the observable
        # backend total/materialized mismatch.
        source_complete=True,
        page_size=20,
        classification="user-text",
        items=list(HOLDERS[:30]),
    )

    first = result_handles.page_of(record)
    last = result_handles.page_of(record, cursor=first.next_cursor)

    assert first.has_more is True
    assert first.next_cursor is not None
    assert last.items == HOLDERS[20:30]
    assert last.has_more is True
    assert last.next_cursor is None
    assert last.source_complete is False
    assert last.matched_complete is False
    assert last.continuation == "source-incomplete"
    assert last.incomplete_reason == result_handles.SOURCE_INCOMPLETE_REASON
    assert "source_complete=false" in last.as_observation()
    assert "treat this result as partial" in last.as_observation()


def test_cursor_is_bound_to_handle_and_normalized_filter_identity():
    record = StoredResult(
        handle_id="handle-a",
        command_name="show_holders",
        kind="holders",
        summary="60 holder(s).",
        ordering="backend order",
        total=60,
        page_size=20,
        classification="user-text",
        items=list(HOLDERS[:60]),
    )
    other = record.model_copy(update={"handle_id": "handle-b"})

    filtered = result_handles.page_of(record, contains=" HOLDER NUMBER ")
    assert filtered.next_cursor is not None
    with pytest.raises(InvalidResultCursor, match="normalized filter"):
        result_handles.page_of(record, cursor=filtered.next_cursor)
    with pytest.raises(InvalidResultCursor, match="result handle"):
        result_handles.page_of(
            other,
            cursor=filtered.next_cursor,
            contains="holder number",
        )

    # Equivalent case and surrounding whitespace normalize to one identity.
    continued = result_handles.page_of(
        record,
        cursor=filtered.next_cursor,
        contains="holder number",
    )
    assert continued.offset == 20
    # A new filter without a cursor intentionally starts over.
    restarted = result_handles.page_of(record, contains="Number 47")
    assert restarted.offset == 0
    assert restarted.items == ("identity_047  Holder Number 47",)


def test_a_filter_narrows_the_page_and_is_echoed(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    try:
        _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))

        page = result_handles.fetch_page(
            handle_id, contains="identity_04", page_size=100, host=ctx
        )
        assert page.matched == 10  # identity_040 .. identity_049
        assert page.items == tuple(HOLDERS[40:50])
        assert page.filters == {"contains": "identity_04"}
        # `total` is ALWAYS the producer's backend total and `matched` is always
        # the post-filter count, on every page. Collapsing them loses one fact
        # either way round: `total` alone lets the agent read a filtered page as
        # the whole population; `matched` alone hides from an evaluator how much
        # of the population the filter excluded.
        assert page.total == 477
        observation = page.as_observation()
        assert "total=477 materialized=477 source_complete=true" in observation
        assert "matched=10 matched_complete=true" in observation
        assert "filters=contains=identity_04" in observation
    finally:
        with suppress(Exception):
            ctx.close()


def test_producer_filters_ride_the_observation(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """A command that already filtered must say so, or the agent reads a
    filtered page as the whole population."""
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(
        monkeypatch,
        result_handles.declare(_holder_spec(filters={"department": "finance"})),
    )
    try:
        observation = _run_turn(ctx)
        assert "filters=department=finance" in observation
    finally:
        with suppress(Exception):
            ctx.close()


# ----------------------------------------------------------------------
# Evidence keeps the full payload
# ----------------------------------------------------------------------


def test_the_full_payload_survives_in_the_span_and_the_turn_record(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    try:
        observation = _run_turn(ctx)

        execute = sink.named(tracing.SPAN_COMMAND_EXECUTE)[0]
        assert execute.attributes["response_text"] == FULL_RESPONSE
        assert execute.attributes[tracing.ATTR_RESULT_HANDLE_ID] == next(
            iter(result_handles.iter_handles(ctx))
        )

        # The turn record's command_outputs — what an independent evaluator
        # scores — is untouched.
        assert ctx._turn_outputs[0].command_response.response == FULL_RESPONSE
        assert sink.turn_records
        record = sink.turn_records[-1]
        outputs = record.turn_output.command_outputs
        assert outputs[0].command_response.response == FULL_RESPONSE

        # ...and the agent did not see it.
        assert "identity_476" not in observation
    finally:
        with suppress(Exception):
            ctx.close()


def test_an_eviction_is_recorded_on_the_span_that_caused_it(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """An eviction nobody recorded makes a later handle miss undiagnosable.

    ido-mn1.6.6 resolves cited handles at the END of a turn, so the failure it
    would produce is a missing table in a finished answer — arriving long after
    the dispatch that dropped the payload, with nothing tying the two together.
    The span of the command whose storage displaced them names them.
    """
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    # A budget small enough that the second stored payload displaces the first.
    monkeypatch.setenv(result_handles.MAX_STORED_BYTES_VAR, "2000")
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    try:
        _run_turn(ctx)
        first = next(iter(result_handles.iter_handles(ctx)))
        _run_turn(ctx)

        executes = sink.named(tracing.SPAN_COMMAND_EXECUTE)
        assert executes[0].attributes.get(tracing.ATTR_RESULT_HANDLES_EVICTED) is None
        assert executes[1].attributes[tracing.ATTR_RESULT_HANDLES_EVICTED] == [first]

        with pytest.raises(UnknownResultHandle):
            result_handles.fetch_page(first, host=ctx)
    finally:
        with suppress(Exception):
            ctx.close()


def test_the_live_turn_keeps_its_handles_across_many_commands(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """The Gate 4 Arm C shape: many opted-in commands inside ONE turn.

    Every handle the turn issued must still resolve when the turn ends, because
    that is when composition reads the ones the agent cited.
    """
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(
        monkeypatch,
        result_handles.declare(_holder_spec(items=list(HOLDERS[:20]), total=20)),
    )

    def fake_forward(**kwargs):
        for _ in range(40):
            _execute_workflow_query("open_portrait", ctx)
        return SimpleNamespace(final_answer="done")

    ctx._workflow_tool_agent = MagicMock(side_effect=fake_forward)
    ctx._intent_clarification_agent = MagicMock()
    try:
        ctx.process_turn("walk the roster")
        handles = list(result_handles.iter_handles(ctx))
        assert len(handles) == 40
        for handle_id in handles:
            assert result_handles.fetch_page(handle_id, host=ctx).matched == 20
        assert not [
            span
            for span in sink.named(tracing.SPAN_COMMAND_EXECUTE)
            if span.attributes.get(tracing.ATTR_RESULT_HANDLES_EVICTED)
        ]
    finally:
        with suppress(Exception):
            ctx.close()


def test_the_agent_tool_call_span_still_records_the_full_response(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """The span the agent seam closes is evidence too, not a copy of the prompt."""
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    try:
        _run_turn(ctx)
        tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
        assert tool_call.attributes["response_text"] == FULL_RESPONSE
        assert ctx.action_log[0]["response"] == FULL_RESPONSE
    finally:
        with suppress(Exception):
            ctx.close()


# ----------------------------------------------------------------------
# Capture policy
# ----------------------------------------------------------------------


def test_the_capture_policy_applies_to_the_stored_payload(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """Default-deny reaches the agent-visible copy too (FW-REQ-002B, P1).

    The stored payload's destination IS the prompt, so it is projected with
    `for_prompt=True` under the process profile. Under `evidence` an
    unclassified payload is withheld — and the observation SAYS withheld rather
    than reporting an empty result, which an agent would relay as "no holders".
    """
    from fastworkflow import observability_store as obs

    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    try:
        observation = _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))
        stored = result_handles.get_result(handle_id, host=ctx)

        assert is_capture_envelope(stored.items)
        assert is_capture_envelope(stored.summary)
        assert is_capture_envelope(stored.ordering)
        assert is_capture_envelope(stored.filters)
        assert stored.payload_withheld is True
        assert "withheld by the capture policy" in observation
        assert "Holder Number" not in observation
        # The envelope is not silence: it still carries size and digest.
        assert stored.items["original_bytes"] > 0
        # ...and the evidence path is unaffected by the prompt-side withholding.
        assert sink.named(tracing.SPAN_COMMAND_EXECUTE)[0].attributes[
            "response_text"
        ] == FULL_RESPONSE
    finally:
        with suppress(Exception):
            ctx.close()


def test_a_declared_classification_is_honored(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """`controlled-vocabulary` is bounded rather than omitted under evidence."""
    from fastworkflow import observability_store as obs

    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(
        monkeypatch,
        result_handles.declare(
            _holder_spec(
                items=["active", "revoked"],
                total=2,
                classification="controlled-vocabulary",
            )
        ),
    )
    try:
        observation = _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))
        stored = result_handles.get_result(handle_id, host=ctx)
        assert stored.item_list == ["active", "revoked"]
        assert "active" in observation
    finally:
        with suppress(Exception):
            ctx.close()


# ----------------------------------------------------------------------
# The seam a workflow's own fetch command sits on
# ----------------------------------------------------------------------


def test_a_command_can_reach_the_store_through_the_ambient_host(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """What `ido-mn1.6.2`'s `fetch_result_page` command will actually do.

    fastWorkflow registers no core `fetch_result_page`: a core command joins
    EVERY workflow's command surface, its `what_can_i_do` output and its trained
    intent model, which would make the off-path parity claim false for workflows
    that never opt in. A workflow declares its own command instead, and reaches
    the store the same way the NLU emitters reach the trace sink — through the
    host `CommandExecutor.invoke_command` binds for the duration of the call. So
    this runs `fetch_page` with no explicit host, from inside a dispatch.
    """
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    try:
        _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))

        fetched: list[str] = []

        def paging_hop(cls, workflow, action):
            # No `host=`: exactly what a ResponseGenerator can do, holding only
            # a Workflow.
            page = result_handles.fetch_page(
                handle_id,
                cursor=result_handles.encode_cursor(
                    40,
                    handle_id=handle_id,
                ),
            )
            fetched.append(page.as_observation())
            return fastworkflow.CommandOutput(
                command_name="fetch_result_page",
                command_response=fastworkflow.CommandResponse(
                    response=page.as_observation(),
                    artifacts={"command_handled": True},
                ),
            )

        monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(paging_hop))
        observation = _run_turn(ctx, "fetch_result_page")

        assert fetched, "the command never reached the store"
        assert observation == fetched[0]
        assert "identity_040" in observation
        assert "identity_059" in observation
        assert "identity_060" not in observation
    finally:
        with suppress(Exception):
            ctx.close()


# ----------------------------------------------------------------------
# Suspend / resume
# ----------------------------------------------------------------------


def test_the_handle_store_survives_suspend_and_resume(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """A resumed turn continues the same trajectory, which still cites handles."""
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, result_handles.declare(_holder_spec()))
    channel = f"chan-{uuid.uuid4().hex[:8]}"
    try:
        _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))
        # The agent is a MagicMock here, and `serialize_state` asks it for its
        # suspended blob; a Mock's auto-attribute is not JSON-native and the
        # strict encoder rightly refuses it. Nothing about this test concerns
        # the ReAct blob.
        ctx._workflow_tool_agent = None
        blob = ctx.serialize_state(channel_id=channel)
    finally:
        with suppress(Exception):
            ctx.close()

    # Through JSON, as the store writes it — an in-memory dict would not prove
    # the blob is encodable.
    from fastworkflow.state_serialization import encode_state

    round_tripped = json.loads(encode_state(blob))
    assert round_tripped["result_handles"]

    restored = _make_ctx(todo_workflow_path, tmp_path, RecordingTraceSink(), monkeypatch)
    try:
        restored.apply_serialized_state(round_tripped)
        page = result_handles.fetch_page(
            handle_id,
            cursor=result_handles.encode_cursor(
                20,
                handle_id=handle_id,
            ),
            host=restored,
        )
        assert page.items == tuple(HOLDERS[20:40])
        assert page.total == 477
    finally:
        with suppress(Exception):
            restored.close()


def test_large_full_artifacts_survive_the_persisted_session_state(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """Both evaluator evidence and the prompt-side handle survive strict JSON."""
    large_tail = "tail-fact-" + "z" * 300_000
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(
        monkeypatch,
        result_handles.declare(
            _holder_spec(
                items=list(HOLDERS[:3]),
                total=3,
                detail={"tail": large_tail},
            )
        ),
    )
    channel = f"chan-large-{uuid.uuid4().hex[:8]}"
    try:
        _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))
        ctx._workflow_tool_agent = None
        blob = ctx.serialize_state(channel_id=channel)
    finally:
        with suppress(Exception):
            ctx.close()

    from fastworkflow.state_serialization import encode_state

    persisted = json.loads(encode_state(blob))
    declaration = persisted["turn"]["outputs"][0]["command_response"][
        "artifacts"
    ][result_handles.RESULT_HANDLE_ARTIFACT_KEY]
    assert declaration["detail"]["tail"] == large_tail
    stored_state = persisted["result_handles"][0]
    assert stored_state["detail"]["tail"] == large_tail

    restored = _make_ctx(
        todo_workflow_path,
        tmp_path,
        RecordingTraceSink(),
        monkeypatch,
    )
    try:
        restored.apply_serialized_state(persisted)
        stored = result_handles.get_result(handle_id, host=restored)
        assert stored.detail["tail"] == large_tail
        assert stored.item_list == list(HOLDERS[:3])
    finally:
        with suppress(Exception):
            restored.close()


def test_a_schema_seven_blob_restores_with_an_empty_store(
    initialized_fastworkflow, todo_workflow_path, tmp_path, sink, monkeypatch
):
    """Degradation, not corruption: the agent is told to re-run the command."""
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    try:
        blob = ctx.serialize_state(channel_id=f"chan-{uuid.uuid4().hex[:8]}")
        blob["schema_version"] = 7
        blob.pop("result_handles")
        ctx.apply_serialized_state(blob)
        assert len(ctx.result_handles) == 0
        with pytest.raises(UnknownResultHandle):
            result_handles.fetch_page("gone", host=ctx)
    finally:
        with suppress(Exception):
            ctx.close()


# ----------------------------------------------------------------------
# The unit-level contract
# ----------------------------------------------------------------------


def test_a_cursor_from_another_version_is_refused_not_reinterpreted():
    """The failure a versionless cursor would produce is silent and wrong."""
    with pytest.raises(InvalidResultCursor, match="version"):
        result_handles.decode_cursor(
            "v1:40:legacy",
            handle_id="h",
        )
    with pytest.raises(InvalidResultCursor, match="not a cursor"):
        result_handles.decode_cursor("40", handle_id="h")
    with pytest.raises(InvalidResultCursor):
        result_handles.decode_cursor(
            result_handles.encode_cursor(-1, handle_id="h"),
            handle_id="h",
        )
    assert result_handles.decode_cursor(None, handle_id="h") == 0
    cursor = result_handles.encode_cursor(40, handle_id="h")
    assert result_handles.decode_cursor(cursor, handle_id="h") == 40
    assert cursor.startswith("v2:40:")


def test_an_unknown_handle_is_an_error_not_an_empty_page():
    store = ResultHandleStore()
    with pytest.raises(UnknownResultHandle, match="Re-run the command"):
        store.get("never-issued")


def _stored(handle_id: str, *, turn_key=None, stored_bytes=0) -> StoredResult:
    return StoredResult(
        handle_id=handle_id,
        command_name="c",
        kind="k",
        summary="s",
        ordering="o",
        total=0,
        page_size=20,
        classification="controlled-vocabulary",
        turn_key=turn_key,
        stored_bytes=stored_bytes,
    )


def test_the_count_ceiling_never_evicts_a_handle_from_the_live_turn():
    """The rule the old count-based bound got wrong.

    ido-mn1.6.6 resolves the handles the agent CITED at composition time, at the
    END of a turn. A handle evicted mid-turn is therefore not a retry the agent
    can make — it is a silently missing table in the final answer. So the count
    ceiling, which is only a backstop, must skip the live turn entirely even when
    that leaves the store above its nominal entry count.
    """
    store = ResultHandleStore(max_entries=2, max_bytes=10**9)
    store.put(_stored("old0", turn_key="t0"), current_turn_key="t1")
    store.put(_stored("old1", turn_key="t0"), current_turn_key="t1")
    for index in range(4):
        evicted = store.put(_stored(f"live{index}", turn_key="t1"), current_turn_key="t1")

    # Both stale handles went; not one live one did, and the store is knowingly
    # over its entry count as a result.
    assert set(store.handle_ids()) == {
        "live0", "live1", "live2", "live3"
    }
    assert len(store) == 4 > 2
    assert evicted == ()  # nothing left to take by the time the last one landed


def test_the_byte_budget_is_hard_and_may_reach_into_the_live_turn():
    """The one bound that cannot be exceeded, because the alternative is
    unbounded memory. It still never drops the record just stored."""
    store = ResultHandleStore(max_entries=10_000, max_bytes=250)
    store.put(_stored("a", turn_key="t1", stored_bytes=100), current_turn_key="t1")
    store.put(_stored("b", turn_key="t1", stored_bytes=100), current_turn_key="t1")
    evicted = store.put(
        _stored("c", turn_key="t1", stored_bytes=100), current_turn_key="t1"
    )

    assert evicted == ("a",)  # oldest first, live turn or not
    assert set(store.handle_ids()) == {"b", "c"}
    assert store.total_bytes <= 250
    with pytest.raises(UnknownResultHandle):
        store.get("a")


def test_the_byte_budget_never_drops_the_record_just_stored():
    """Dropping the payload whose observation the agent is about to read would
    make the very next step unresolvable."""
    store = ResultHandleStore(max_entries=10_000, max_bytes=10)
    store.put(_stored("a", stored_bytes=5))
    store.put(_stored("b", stored_bytes=5_000))
    assert list(store.handle_ids()) == ["b"]
    assert store.total_bytes > 10  # knowingly over: the alternative is nothing


def test_stale_turns_go_before_the_live_one_under_the_byte_budget():
    store = ResultHandleStore(max_entries=10_000, max_bytes=250)
    store.put(_stored("live", turn_key="t1", stored_bytes=100), current_turn_key="t1")
    store.put(_stored("stale", turn_key="t0", stored_bytes=100), current_turn_key="t1")
    evicted = store.put(
        _stored("newest", turn_key="t1", stored_bytes=100), current_turn_key="t1"
    )
    # `stale` is NOT the oldest by insertion, but it is the one not from the live
    # turn, and that ordering is the whole point.
    assert evicted == ("stale",)
    assert set(store.handle_ids()) == {"live", "newest"}


def test_the_byte_budget_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(result_handles.MAX_STORED_BYTES_VAR, "512")
    assert result_handles.max_stored_bytes() == 512
    # A mis-set variable must not disable the bound.
    monkeypatch.setenv(result_handles.MAX_STORED_BYTES_VAR, "0")
    assert result_handles.max_stored_bytes() == result_handles.DEFAULT_MAX_STORED_BYTES
    monkeypatch.setenv(result_handles.MAX_STORED_BYTES_VAR, "not a number")
    assert result_handles.max_stored_bytes() == result_handles.DEFAULT_MAX_STORED_BYTES
    monkeypatch.delenv(result_handles.MAX_STORED_BYTES_VAR)
    assert result_handles.max_stored_bytes() == result_handles.DEFAULT_MAX_STORED_BYTES


def test_the_default_bound_holds_a_whole_arm_c_cell():
    """The sizing claim, checked rather than asserted in a comment.

    Gate 4 v4's worst Arm C cell ran 177 agent steps; ~27 `open_portrait` calls
    at ~4 KB each plus the listing commands. The default budget must hold every
    handle such a cell produces, or the bound reintroduces exactly the mid-turn
    eviction it was rewritten to prevent.
    """
    store = ResultHandleStore()
    payload = "x" * 4_000
    for index in range(200):
        store.put(
            _stored(f"h{index}", turn_key="t1", stored_bytes=len(payload)),
            current_turn_key="t1",
        )
    assert len(store) == 200
    assert store.total_bytes < result_handles.DEFAULT_MAX_STORED_BYTES


def test_a_malformed_declaration_is_ignored_rather_than_raised():
    """A command whose compaction metadata is wrong must still return its answer."""
    malformed = result_handles.spec_from_artifacts({"__fw_result_handle__": 7})
    assert malformed is not None
    assert malformed.classification is None
    assert malformed.kind == "withheld"
    assert result_handles.spec_from_artifacts({"other": 1}) is None
    assert result_handles.spec_from_artifacts(None) is None


def test_page_size_is_clamped():
    """A producer asking for 10_000 rows a page has opted out while looking in."""
    record = StoredResult(
        handle_id="h",
        command_name="c",
        kind="k",
        summary="s",
        ordering="o",
        total=500,
        page_size=20,
        classification="user-text",
        items=[f"row {index}" for index in range(500)],
    )
    assert len(result_handles.page_of(record, page_size=10_000).items) == (
        result_handles.MAX_PAGE_SIZE
    )
    assert len(result_handles.page_of(record, page_size=0).items) == 1


def test_an_unreadable_stored_record_is_dropped_not_fatal():
    store = ResultHandleStore()
    store.apply_state([{"handle_id": "h", "nonsense": True}, "not a dict"])
    assert len(store) == 0


def test_missing_classification_fails_closed_even_under_debug(
    initialized_fastworkflow,
    todo_workflow_path,
    tmp_path,
    sink,
    monkeypatch,
):
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    spec = _holder_spec(
        classification=None,
        summary="SECRET SUMMARY",
        ordering="SECRET ORDER",
        filters={"tenant": "SECRET FILTER"},
        detail={"secret": "SECRET DETAIL"},
    )
    _cme_hop(monkeypatch, result_handles.declare(spec))
    try:
        observation = _run_turn(ctx)
        handle_id = next(iter(result_handles.iter_handles(ctx)))
        stored = result_handles.get_result(handle_id, host=ctx)

        for value in (
            stored.kind,
            stored.summary,
            stored.ordering,
            stored.filters,
            stored.items,
            stored.detail,
            stored.response,
        ):
            assert is_capture_envelope(value)
        assert "SECRET" not in observation
        assert "Holder Number" not in observation
        assert sink.named(tracing.SPAN_COMMAND_EXECUTE)[0].attributes[
            "response_text"
        ] == FULL_RESPONSE

        resolved = result_handles.resolve_for_presentation(
            cited=[handle_id],
            host=ctx,
        )
        assert "SECRET" not in resolved.text
        assert resolved.entries[0].withheld is True
    finally:
        with suppress(Exception):
            ctx.close()


def test_invalid_classification_is_closed_and_never_restores_raw_observation(
    initialized_fastworkflow,
    todo_workflow_path,
    tmp_path,
    sink,
    monkeypatch,
):
    with pytest.raises(ValueError):
        _holder_spec(classification="not-a-classification")

    raw = _holder_spec().model_dump(mode="json")
    raw["classification"] = "not-a-classification"
    raw["summary"] = "SECRET INVALID SUMMARY"
    _cme_hop(
        monkeypatch,
        {result_handles.RESULT_HANDLE_ARTIFACT_KEY: raw},
    )
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    try:
        observation = _run_turn(ctx)
        stored = result_handles.get_result(
            next(iter(result_handles.iter_handles(ctx))),
            host=ctx,
        )
        assert stored.classification is None
        assert result_handles._is_capture_envelope(stored.summary)
        assert "SECRET INVALID SUMMARY" not in observation
        assert FULL_RESPONSE not in observation
    finally:
        with suppress(Exception):
            ctx.close()


def test_legacy_unclassified_handle_state_is_reprojected_before_restore():
    store = ResultHandleStore()
    store.apply_state(
        [
            {
                "handle_id": "legacy",
                "command_name": "show_holders",
                "kind": "holders",
                "summary": "SECRET SUMMARY",
                "ordering": "SECRET ORDER",
                "total": 1,
                "page_size": 20,
                "filters": {"tenant": "SECRET FILTER"},
                "items": ["SECRET ROW"],
                "detail": {"secret": "SECRET DETAIL"},
                "response": "SECRET RESPONSE",
            }
        ]
    )

    restored = store.get("legacy")
    for value in (
        restored.kind,
        restored.summary,
        restored.ordering,
        restored.filters,
        restored.items,
        restored.detail,
        restored.response,
    ):
        assert result_handles._is_capture_envelope(value)
    assert "SECRET" not in json.dumps(store.to_state())


def test_compaction_preserves_resolver_feedback_but_not_in_raw_evidence(
    initialized_fastworkflow,
    todo_workflow_path,
    tmp_path,
    sink,
    monkeypatch,
):
    artifacts = result_handles.declare(
        _holder_spec(items=list(HOLDERS[:2]), total=2)
    )
    artifacts.update(
        {
            "control_codes": ["rule-one", "rule-two"],
            "labels": [
                "Vendor with past end date and active account",
                "External worker whose manager left",
            ],
        }
    )
    ctx = _make_ctx(todo_workflow_path, tmp_path, sink, monkeypatch)
    _cme_hop(monkeypatch, artifacts)
    scope = PlanExecutionScope(
        arm=PlanExecutionArm.B,
        task_goal_id="task-one",
        task_bindings=(
            (
                "rule_query",
                PlanExecutionBinding(
                    value=(
                        "Enabled vendor identities with expired ending date "
                        "and owning active accounts"
                    ),
                    source="utterance",
                    kind="exact_text",
                    resolver=CONTROL_ALIAS_RESOLVER_V1,
                ),
            ),
        ),
    )
    seen: list[str] = []

    def fake_forward(**_kwargs):
        ctx._turn_leaf_scope = scope
        try:
            seen.append(_execute_workflow_query("list rules", ctx))
        finally:
            ctx._turn_leaf_scope = None
        return SimpleNamespace(final_answer="done")

    ctx._workflow_tool_agent = MagicMock(side_effect=fake_forward)
    ctx._intent_clarification_agent = MagicMock()
    try:
        ctx.process_turn("inspect the vendor control")
        observation = seen[0]
        assert "result_handle=" in observation
        assert observation.count("Holder Number") == 2
        assert "Private deterministic binding resolution" in observation
        assert '"handle":"rule-one"' in observation
        assert FULL_RESPONSE not in observation

        tool_call = sink.named(tracing.SPAN_AGENT_TOOL_CALL)[0]
        assert tool_call.attributes["response_text"] == FULL_RESPONSE
        assert ctx.action_log[0]["response"] == FULL_RESPONSE
        assert (
            "Private deterministic binding resolution"
            not in tool_call.attributes["response_text"]
        )
    finally:
        with suppress(Exception):
            ctx.close()
