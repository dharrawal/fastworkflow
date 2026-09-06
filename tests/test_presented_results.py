"""Composition resolves the handles the answer must present (ido-mn1.6.6).

EXP-028's completion endpoint rates what the final answer PRESENTS. Observation
compaction (ido-mn1.6.1) moved the rows out of the ReAct trajectory, and the
trajectory is exactly what the extraction step composes from — arm A returns its
``final_answer`` verbatim, arms B and C concatenate one per leaf with no further
model call. So without a repair the answer would fail "presented" predicates for
a measurement reason: the model never received the rows.

The repair is a dedicated ``presented_results`` input on the EXTRACT signature,
filled at extraction time from session state. These tests pin the three things
that make it a bound rather than a re-inflation:

* **Selection is runtime-checkable.** A handle is resolved because the agent
  CITED it on its closing step, or because the executing skill DECLARED the
  producing command a presentation output (``presents:``). Never because it is
  large, and never merely because it exists.
* **The view is the request's, not the population's.** A handle whose first page
  the agent read resolves to that page; a filtered fetch resolves the filtered
  rows; the 477-holder population is resolved only if the agent walked it.
* **The cap is arm-invariant and says when it bit.** One constant, one env
  override, and a trim that is stated in the field the model reads AND in the
  evidence a reader scores against.

Arm invariance is checked structurally as well as behaviourally: the three
``_call_agent`` sites in ``workflow_execution_context.py`` are scanned for the
parameter, because "all three arms go through it" is the claim, and three
agreeing copies of a call is how that claim rots.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow import result_handles
from fastworkflow.result_handles import (
    REASON_CITED,
    REASON_PRESENTS,
    SCOPE_FETCHED,
    SCOPE_PRODUCER_FILTERED,
    ResultHandleSpec,
    ResultHandleStore,
    StoredResult,
)
from fastworkflow.skill_catalog import SkillCatalogError, load_skill_catalog
from fastworkflow.turn_budget import LogicalTurnBudget
from fastworkflow.utils.react import (
    CLOSING_THOUGHT_STEPS,
    MAX_NEXT_THOUGHT_CHARS,
    PRESENTED_RESULTS_FIELD,
    THOUGHT_TRUNCATION_NOTICE,
    fastWorkflowReAct,
)
from fastworkflow.workflow_agent import WorkflowAgentSignature
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

# The Gate 4 v4 shape: `show_holders` returned all 477 holders in one 23k string.
HOLDERS = tuple(f"identity_{index:03d}  Holder Number {index}" for index in range(477))
PORTRAIT = tuple(f"field_{index}: value {index}" for index in range(12))


class _Host:
    """A trace host that owns a handle store, which is all `store_for` reads."""

    def __init__(self, store: ResultHandleStore) -> None:
        self.result_handles = store


def _record(
    handle_id: str,
    *,
    command_name: str = "show_holders",
    items: tuple[str, ...] = HOLDERS,
    total: int | None = None,
    page_size: int = 20,
    filters: dict[str, str] | None = None,
    detail: object | None = None,
    items_override: object = None,
    presentation: bool = False,
    source_complete: bool | None = None,
) -> StoredResult:
    return StoredResult(
        handle_id=handle_id,
        command_name=command_name,
        kind="holders",
        summary=f"{total if total is not None else len(items)} holder(s).",
        ordering="backend view order",
        total=total if total is not None else len(items),
        source_complete=source_complete,
        page_size=page_size,
        filters=dict(filters or {}),
        presentation=presentation,
        classification="user-text",
        items=items_override if items_override is not None else list(items),
        detail=detail or {},
        response="\n".join(items),
        turn_key="t1",
        stored_bytes=1,
    )


@pytest.fixture
def host() -> _Host:
    return _Host(ResultHandleStore())


def _entry(resolved, handle_id):
    return next(entry for entry in resolved.entries if entry.handle_id == handle_id)


# ----------------------------------------------------------------------
# 1. Selection: cited, or declared. Never "large".
# ----------------------------------------------------------------------


def test_a_cited_handle_is_resolved_and_an_uncited_larger_one_is_not(host):
    """The claim the whole bound rests on: size does not select."""
    host.result_handles.put(_record("a" * 32, command_name="open_portrait",
                                    items=PORTRAIT, page_size=50))
    host.result_handles.put(_record("b" * 32))  # 477 rows, uncited

    resolved = result_handles.resolve_for_presentation(cited=["a" * 32], host=host)

    assert [entry.handle_id for entry in resolved.entries] == ["a" * 32]
    assert _entry(resolved, "a" * 32).reasons == (REASON_CITED,)
    assert "Holder Number" not in resolved.text
    assert "value 3" in resolved.text


def test_a_declared_presentation_output_is_resolved_without_a_citation(host):
    host.result_handles.put(_record("c" * 32, filters={"department": "Engineering"},
                                    items=HOLDERS[:30], total=30))

    resolved = result_handles.resolve_for_presentation(
        presented=["c" * 32], host=host
    )

    assert _entry(resolved, "c" * 32).reasons == (REASON_PRESENTS,)


def test_a_handle_that_is_both_cited_and_declared_resolves_once_carrying_both(host):
    host.result_handles.put(_record("d" * 32, items=HOLDERS[:5], page_size=20))

    resolved = result_handles.resolve_for_presentation(
        cited=["d" * 32], presented=["d" * 32], host=host
    )

    assert len(resolved.entries) == 1
    assert _entry(resolved, "d" * 32).reasons == (REASON_CITED, REASON_PRESENTS)


def test_a_cited_handle_the_store_no_longer_holds_is_recorded_as_unresolved(host):
    resolved = result_handles.resolve_for_presentation(cited=["e" * 32], host=host)

    assert resolved.entries == ()
    assert resolved.unresolved == ("e" * 32,)
    assert resolved.as_evidence()["unresolved"] == ["e" * 32]
    assert "resolution incomplete" in resolved.text
    assert "do not imply" in resolved.text


def test_an_unresolved_notice_is_included_in_the_typed_total_cap(host):
    """The notice must not make a nominally untrimmed field exceed the allocator."""
    handle_id = "e1" + "0" * 30
    unavailable = "e2" + "0" * 30
    host.result_handles.put(
        _record(handle_id, items=tuple("x" * 30 for _ in range(200)), page_size=200)
    )
    result_handles.fetch_page(handle_id, page_size=200, host=host)
    exact_fit = result_handles.resolve_for_presentation(
        cited=[handle_id],
        host=host,
    ).field_bytes

    resolved = result_handles.resolve_for_presentation(
        cited=[handle_id, unavailable],
        host=host,
        max_bytes=exact_fit,
    )

    assert resolved.field_bytes <= exact_fit
    assert resolved.unresolved == (unavailable,)
    assert resolved.trimmed is True
    assert (
        resolved.truncation_classification
        == result_handles.PRESENTATION_TRUNCATION_CLASSIFICATION
    )
    assert "classification=infrastructure-truncated" in resolved.text
    assert "resolution incomplete" in resolved.text


def test_citations_are_case_normalized(host):
    """A model that retypes an id sometimes upcases it; that is not a miss."""
    host.result_handles.put(_record("f" * 32, items=HOLDERS[:3]))

    resolved = result_handles.resolve_for_presentation(cited=["F" * 32], host=host)

    assert [entry.handle_id for entry in resolved.entries] == ["f" * 32]


def test_nothing_selected_resolves_to_the_runtime_sentence_not_an_empty_field(host):
    resolved = result_handles.resolve_for_presentation(host=host)

    assert resolved.entries == ()
    assert resolved.text == result_handles.PRESENTED_RESULTS_NONE
    assert resolved.text.strip()


# ----------------------------------------------------------------------
# 2. Parsing a citation is the inverse of rendering one
# ----------------------------------------------------------------------


def test_parse_handle_ids_reads_the_observation_the_module_renders(host):
    record = _record("1" * 32)
    host.result_handles.put(record)
    observation = result_handles.page_of(record).as_observation()

    assert result_handles.parse_handle_ids(observation) == ("1" * 32,)


def test_parse_handle_ids_reads_a_bare_id_and_keeps_first_mention_order():
    text = f"done; results are in {'a' * 32} and result_handle={'b' * 32}"

    assert result_handles.parse_handle_ids(text) == ("a" * 32, "b" * 32)


def test_parse_handle_ids_ignores_a_longer_digest():
    assert result_handles.parse_handle_ids("sha256:" + "c" * 64) == ()


# ----------------------------------------------------------------------
# 3. The view is the request's, not the population's
# ----------------------------------------------------------------------


def test_the_first_page_the_agent_was_shown_is_what_a_citation_resolves(host):
    """The whole opt-in path, end to end: a command declares, the store keeps the
    477 rows, the agent is shown 20 — and citing the handle brings back the 20.
    """
    handle_id = "2" * 32
    command_output = fastworkflow.CommandOutput(
        command_name="show_holders",
        command_response=fastworkflow.CommandResponse(
            response="477 holder(s).\n" + "\n".join(HOLDERS),
            artifacts=result_handles.declare(
                ResultHandleSpec(
                    kind="holders",
                    summary="477 holder(s) of this permission.",
                    items=list(HOLDERS),
                    total=477,
                    page_size=20,
                    classification="user-text",
                )
            ),
        ),
    )
    command_output.command_call_id = handle_id

    stored, evicted = result_handles.store_from_command_output(host, command_output)
    observation = result_handles.compact_observation_for(host, command_output)

    assert (stored, evicted) == (handle_id, ())
    assert observation.count("Holder Number") == 20

    resolved = result_handles.resolve_for_presentation(cited=[handle_id], host=host)
    entry = _entry(resolved, handle_id)

    assert entry.scope == SCOPE_FETCHED
    assert entry.rows == HOLDERS[:20]
    assert entry.total == 477
    assert entry.rows_available == 20
    # The population is in the store the whole time; the bound is the view.
    assert len(host.result_handles.get(handle_id).item_list) == 477


def test_the_union_of_the_pages_actually_fetched_is_resolved_in_producer_order(host):
    handle_id = "3" * 32
    host.result_handles.put(_record(handle_id))
    page = result_handles.fetch_page(handle_id, host=host)
    page = result_handles.fetch_page(handle_id, cursor=page.next_cursor, host=host)
    result_handles.fetch_page(handle_id, cursor=page.next_cursor, host=host)

    entry = _entry(
        result_handles.resolve_for_presentation(cited=[handle_id], host=host),
        handle_id,
    )

    assert entry.rows == HOLDERS[:60]
    assert entry.scope == SCOPE_FETCHED


def test_a_page_fetched_under_a_filter_resolves_the_filtered_rows_only(host):
    handle_id = "4" * 32
    host.result_handles.put(_record(handle_id))

    result_handles.fetch_page(handle_id, contains="Holder Number 4", host=host)
    entry = _entry(
        result_handles.resolve_for_presentation(cited=[handle_id], host=host),
        handle_id,
    )

    assert entry.rows
    assert all("Holder Number 4" in row for row in entry.rows)
    # `total` still says what the population is: a filtered page must never read
    # as coverage of the whole.
    assert entry.total == 477


def test_a_record_with_no_recorded_view_falls_back_to_its_first_page(host):
    """What a handle restored from a state written before views were recorded means."""
    handle_id = "5" * 32
    host.result_handles.put(_record(handle_id))

    entry = _entry(
        result_handles.resolve_for_presentation(cited=[handle_id], host=host),
        handle_id,
    )

    assert entry.rows == HOLDERS[:20]


def test_re_fetching_the_same_page_records_one_view(host):
    handle_id = "6" * 32
    host.result_handles.put(_record(handle_id))

    result_handles.fetch_page(handle_id, host=host)
    result_handles.fetch_page(handle_id, host=host)

    assert len(host.result_handles.get(handle_id).views) == 1


def test_recorded_views_are_bounded(host):
    handle_id = "7" * 32
    host.result_handles.put(_record(handle_id, page_size=1))
    for offset in range(result_handles.MAX_RECORDED_VIEWS + 20):
        result_handles.fetch_page(
            handle_id,
            cursor=result_handles.encode_cursor(
                offset,
                handle_id=handle_id,
            ),
            host=host,
        )

    assert (
        len(host.result_handles.get(handle_id).views)
        == result_handles.MAX_RECORDED_VIEWS
    )


def test_reading_a_page_does_not_make_a_handle_younger(host):
    """`put`'s rule: the store bounds memory, and reading does not lower a cost."""
    host.result_handles.put(_record("8" * 32, items=HOLDERS[:3]))
    host.result_handles.put(_record("9" * 32, items=HOLDERS[:3]))

    result_handles.fetch_page("8" * 32, host=host)

    assert host.result_handles.handle_ids() == ("8" * 32, "9" * 32)


# ----------------------------------------------------------------------
# 4. `presents:` widens to the producer's own filtering, and no further
# ----------------------------------------------------------------------


def test_a_declared_output_resolves_every_row_its_producer_filtered_to(host):
    """The command was called with the request's filters, so its whole result is
    bounded by the request even where the agent stopped paging."""
    handle_id = "a1" + "0" * 30
    host.result_handles.put(
        _record(handle_id, items=HOLDERS[:30], total=30,
                filters={"department": "Engineering"})
    )
    result_handles.fetch_page(handle_id, host=host)  # the agent saw 20 of 30

    entry = _entry(
        result_handles.resolve_for_presentation(presented=[handle_id], host=host),
        handle_id,
    )

    assert entry.scope == SCOPE_PRODUCER_FILTERED
    assert len(entry.rows) == 30


def test_a_declared_output_with_no_producer_filter_stays_at_what_was_fetched(host):
    """The 477-holder population is not presentable merely by being declared."""
    handle_id = "a2" + "0" * 30
    host.result_handles.put(_record(handle_id))
    result_handles.fetch_page(handle_id, host=host)

    entry = _entry(
        result_handles.resolve_for_presentation(presented=[handle_id], host=host),
        handle_id,
    )

    assert entry.scope == SCOPE_FETCHED
    assert len(entry.rows) == 20


def test_citing_a_handle_never_widens_it(host):
    """A citation says where the result is, not that the answer may list more."""
    handle_id = "a3" + "0" * 30
    host.result_handles.put(
        _record(handle_id, items=HOLDERS[:30], total=30,
                filters={"department": "Engineering"})
    )
    result_handles.fetch_page(handle_id, host=host)

    entry = _entry(
        result_handles.resolve_for_presentation(cited=[handle_id], host=host),
        handle_id,
    )

    assert entry.scope == SCOPE_FETCHED
    assert len(entry.rows) == 20


def test_the_default_selector_is_the_producing_commands_own_flag(host):
    """The arm-invariance fix: no catalogue is consulted, so arm A reads it too."""
    host.result_handles.put(
        _record("b1" + "0" * 30, command_name="show_holders", presentation=True)
    )
    host.result_handles.put(_record("b2" + "0" * 30, command_name="list_groups"))

    found = result_handles.presentation_handles_for(
        ["b1" + "0" * 30, "b2" + "0" * 30], host=host
    )

    assert found == ("b1" + "0" * 30,)


def test_a_skill_override_narrows_the_flagged_set(host):
    """A skill that only reports holders must not also drag in every portrait."""
    host.result_handles.put(
        _record("b3" + "0" * 30, command_name="show_holders", presentation=True)
    )
    host.result_handles.put(
        _record("b4" + "0" * 30, command_name="open_portrait", presentation=True)
    )

    found = result_handles.presentation_handles_for(
        ["b3" + "0" * 30, "b4" + "0" * 30], ["show_holders"], host=host
    )

    assert found == ("b3" + "0" * 30,)


def test_a_skill_override_adds_a_command_the_producer_did_not_flag(host):
    host.result_handles.put(_record("b5" + "0" * 30, command_name="list_groups"))

    found = result_handles.presentation_handles_for(
        ["b5" + "0" * 30], ["list_groups"], host=host
    )

    assert found == ("b5" + "0" * 30,)


def test_an_unflagged_command_with_no_override_is_never_selected(host):
    host.result_handles.put(_record("b6" + "0" * 30, command_name="list_groups"))

    assert result_handles.presentation_handles_for(["b6" + "0" * 30], host=host) == ()


def test_the_flag_round_trips_from_the_command_declaration_to_the_store(host):
    """`ResultHandleSpec.presentation` is what a command sets; nothing else."""
    command_output = fastworkflow.CommandOutput(
        command_name="show_holders",
        command_response=fastworkflow.CommandResponse(
            response="rows",
            artifacts=result_handles.declare(
                ResultHandleSpec(
                    kind="holders",
                    summary="s",
                    items=list(HOLDERS[:3]),
                    presentation=True,
                )
            ),
        ),
    )
    command_output.command_call_id = "b7" + "0" * 30
    result_handles.store_from_command_output(host, command_output)

    assert host.result_handles.get("b7" + "0" * 30).presentation is True
    assert ResultHandleSpec(kind="k", summary="s").presentation is False


# ----------------------------------------------------------------------
# 5. The cap
# ----------------------------------------------------------------------


def test_the_cap_trims_and_says_so_in_the_field_and_in_the_evidence(host):
    handle_id = "c1" + "0" * 30
    host.result_handles.put(_record(handle_id, page_size=200))
    result_handles.fetch_page(handle_id, page_size=200, host=host)

    resolved = result_handles.resolve_for_presentation(
        cited=[handle_id], host=host, max_bytes=1024
    )
    entry = _entry(resolved, handle_id)

    assert entry.trimmed is True
    assert 0 < len(entry.rows) < entry.rows_available
    assert entry.bytes <= 1024
    assert len(resolved.text.encode("utf-8")) <= 1024
    assert "classification=infrastructure-truncated" in resolved.text
    assert resolved.as_evidence()["trimmed"] is True
    assert resolved.as_evidence()["handles"][0]["trimmed"] is True
    assert resolved.as_evidence()["max_bytes"] == 1024
    assert (
        resolved.as_evidence()["truncation_classification"]
        == result_handles.PRESENTATION_TRUNCATION_CLASSIFICATION
    )


def test_the_cap_allocates_to_citations_before_declarations(host):
    """Order is the allocation, so a declaration cannot starve a citation."""
    cited_id, declared_id = "c2" + "0" * 30, "c3" + "0" * 30
    host.result_handles.put(_record(cited_id, items=HOLDERS[:20]))
    host.result_handles.put(_record(declared_id, items=HOLDERS[:20]))

    resolved = result_handles.resolve_for_presentation(
        cited=[cited_id], presented=[declared_id], host=host, max_bytes=1200
    )

    assert _entry(resolved, cited_id).rows
    # The starved entry is still LISTED: the answer must know the result exists.
    # It is listed in independent evidence rather than forced into a field whose
    # hard byte cap cannot fit even the entry header.
    assert all(entry.handle_id != declared_id for entry in resolved.entries)
    # The lower-priority handle remains independently visible in evidence even
    # when its header cannot fit into the model-facing field.
    assert resolved.omitted_handles == (declared_id,)
    assert resolved.as_evidence()["omitted_handles"] == [declared_id]


def test_field_bytes_measures_the_whole_field_for_the_per_call_limit(host):
    """ido-mn1.6.10 sizes the extraction completion limit from this number, so it
    must be the FIELD, not just the rows the cap counted."""
    handle_id = "c5" + "0" * 30
    host.result_handles.put(_record(handle_id, items=HOLDERS[:3]))

    resolved = result_handles.resolve_for_presentation(cited=[handle_id], host=host)

    assert resolved.field_bytes == len(resolved.text.encode("utf-8"))
    assert resolved.field_bytes > resolved.total_bytes > 0
    assert resolved.as_evidence()["field_bytes"] == resolved.field_bytes


def test_the_total_cap_includes_headers_and_producer_metadata(host):
    """Rows were bounded before this fix while a pathological header was not."""
    handle_id = "c6" + "0" * 30
    record = _record(handle_id, items=HOLDERS[:3])
    record.summary = "summary-" + "x" * 40_000
    record.ordering = "ordering-" + "y" * 40_000
    record.filters = {"field-" + "k" * 4000: "value-" + "v" * 40_000}
    host.result_handles.put(record)

    resolved = result_handles.resolve_for_presentation(
        cited=[handle_id],
        host=host,
        max_bytes=result_handles.DEFAULT_PRESENTED_MAX_BYTES,
    )
    entry = _entry(resolved, handle_id)

    assert resolved.field_bytes <= result_handles.DEFAULT_PRESENTED_MAX_BYTES
    assert entry.metadata_trimmed is True
    assert resolved.trimmed is True
    assert (
        resolved.truncation_classification
        == result_handles.PRESENTATION_TRUNCATION_CLASSIFICATION
    )
    assert "classification=infrastructure-truncated" in resolved.text


def test_incomplete_source_is_explicit_in_the_extraction_field(host):
    handle_id = "c7" + "0" * 30
    host.result_handles.put(
        _record(
            handle_id,
            items=HOLDERS[:20],
            total=100,
            source_complete=False,
        )
    )

    resolved = result_handles.resolve_for_presentation(
        cited=[handle_id],
        host=host,
    )
    entry = _entry(resolved, handle_id)

    assert entry.source_complete is False
    assert entry.matched_complete is False
    assert entry.incomplete_reason == result_handles.SOURCE_INCOMPLETE_REASON
    assert "source_complete=false" in resolved.text
    assert "Treat the result as partial" in resolved.text
    assert (
        resolved.as_evidence()["handles"][0]["incomplete_reason"]
        == result_handles.SOURCE_INCOMPLETE_REASON
    )


def test_field_bytes_is_defined_when_nothing_resolved(host):
    resolved = result_handles.resolve_for_presentation(host=host)

    assert resolved.total_bytes == 0
    assert resolved.field_bytes == len(result_handles.PRESENTED_RESULTS_NONE.encode())


def test_the_cap_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(result_handles.PRESENTED_MAX_BYTES_VAR, "4096")
    assert result_handles.presented_max_bytes() == 4096
    monkeypatch.setenv(result_handles.PRESENTED_MAX_BYTES_VAR, "0")
    assert (
        result_handles.presented_max_bytes()
        == result_handles.DEFAULT_PRESENTED_MAX_BYTES
    )
    monkeypatch.setenv(result_handles.PRESENTED_MAX_BYTES_VAR, "not a number")
    assert (
        result_handles.presented_max_bytes()
        == result_handles.DEFAULT_PRESENTED_MAX_BYTES
    )
    monkeypatch.delenv(result_handles.PRESENTED_MAX_BYTES_VAR)
    assert (
        result_handles.presented_max_bytes()
        == result_handles.DEFAULT_PRESENTED_MAX_BYTES
    )


def test_the_default_cap_holds_the_whole_projected_holder_listing(host):
    """The sizing claim, checked rather than asserted in a comment."""
    handle_id = "c4" + "0" * 30
    host.result_handles.put(_record(handle_id, page_size=200))
    for offset in (0, 200, 400):
        result_handles.fetch_page(
            handle_id,
            cursor=result_handles.encode_cursor(
                offset,
                handle_id=handle_id,
            ),
            page_size=200,
            host=host,
        )

    entry = _entry(
        result_handles.resolve_for_presentation(cited=[handle_id], host=host),
        handle_id,
    )

    assert entry.trimmed is False
    assert len(entry.rows) == 477


# ----------------------------------------------------------------------
# 6. The capture policy reaches this field too
# ----------------------------------------------------------------------


def test_a_withheld_payload_is_reported_as_withheld_not_as_an_empty_result(host):
    """The rows passed the `for_prompt=True` projection on the way IN, so an
    envelope arrives here — and an agent told "no rows" would relay "no holders".
    """
    envelope = {"__fw_capture__": "withheld", "original_bytes": 23_000}
    handle_id = "d1" + "0" * 30
    host.result_handles.put(_record(handle_id, items_override=envelope))

    resolved = result_handles.resolve_for_presentation(cited=[handle_id], host=host)
    entry = _entry(resolved, handle_id)

    assert entry.withheld is True
    assert entry.rows == ()
    assert "withheld by the capture policy" in resolved.text
    assert resolved.as_evidence()["handles"][0]["withheld"] is True


def test_the_resolver_never_reaches_past_the_store_into_a_raw_response(host):
    """`response` is evidence, not prompt material; only projected rows resolve."""
    handle_id = "d2" + "0" * 30
    record = _record(handle_id, items=HOLDERS[:2])
    record.response = "SECRET RAW BODY"
    host.result_handles.put(record)

    resolved = result_handles.resolve_for_presentation(cited=[handle_id], host=host)

    assert "SECRET RAW BODY" not in resolved.text


# ----------------------------------------------------------------------
# 7. The ReAct seam: the EXTRACT signature, and only it
# ----------------------------------------------------------------------


def _tool() -> str:
    """A tool."""
    return "x"


@pytest.fixture(scope="module")
def agent():
    return fastWorkflowReAct(WorkflowAgentSignature, tools=[_tool])


def test_the_extraction_signature_carries_the_field_and_the_react_step_does_not(agent):
    """ido-mn1.6.4 pins the react prompt; this change is to EXTRACT only."""
    assert PRESENTED_RESULTS_FIELD in agent.extract.predict.signature.input_fields
    assert PRESENTED_RESULTS_FIELD not in agent.react.signature.input_fields
    assert PRESENTED_RESULTS_FIELD not in agent.react.signature.output_fields


def test_the_field_description_tells_the_model_what_it_may_and_may_not_do(agent):
    field = agent.extract.predict.signature.input_fields[PRESENTED_RESULTS_FIELD]
    desc = (field.json_schema_extra or {}).get("desc") or ""
    assert "do not invent rows" in desc
    assert "trimmed" in desc


def test_final_answer_is_still_unbounded(agent):
    """The deliverable must not acquire a bound from this change."""
    field = agent.extract.predict.signature.output_fields["final_answer"]
    desc = (field.json_schema_extra or {}).get("desc") or ""
    assert "at most" not in desc


class _RecordingExtract:
    """Stands in for `self.extract`, capturing the kwargs it was called with."""

    def __init__(self, answer: str = "composed") -> None:
        self.calls: list[dict] = []
        self.answer = answer

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"final_answer": self.answer, "reasoning": "r"}


def _loop_agent(host, presentation_commands=frozenset()):
    """A fastWorkflowReAct with no dspy wiring — `_finish` is what is under test."""
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent._budget = LogicalTurnBudget(iteration_limit=5)
    agent._step_seals = {}
    agent._exhausted_last_run = False
    agent._safety_envelope = None
    agent.presentation_commands = presentation_commands
    agent.decision_point = None
    agent.extract = _RecordingExtract()
    agent._format_trajectory = lambda trajectory: str(trajectory)
    agent._consult_finish_policy = lambda extract, trajectory, input_args: None
    agent._turn_partial = lambda trajectory, budget: None
    return agent


def _trajectory(*handle_ids, closing_thought: str) -> dict:
    trajectory: dict = {}
    for index, handle_id in enumerate(handle_ids):
        trajectory[f"thought_{index}"] = "looking"
        trajectory[f"tool_name_{index}"] = "show_holders"
        trajectory[f"tool_args_{index}"] = {}
        trajectory[f"observation_{index}"] = (
            f"summary\nresult_handle={handle_id} kind=holders\nrow"
        )
    last = len(handle_ids)
    trajectory[f"thought_{last}"] = closing_thought
    trajectory[f"tool_name_{last}"] = "finish"
    trajectory[f"tool_args_{last}"] = {}
    trajectory[f"observation_{last}"] = "Completed."
    return trajectory


def test_finish_hands_the_resolved_rows_to_the_extraction_call(host, monkeypatch):
    monkeypatch.setattr(result_handles.tracing, "current_host", lambda: host)
    cited, uncited = "e1" + "0" * 30, "e2" + "0" * 30
    host.result_handles.put(_record(cited, items=HOLDERS[:4]))
    host.result_handles.put(_record(uncited, items=HOLDERS[:4]))
    agent = _loop_agent(host)
    trajectory = _trajectory(
        cited, uncited, closing_thought=f"done; results in result_handle={cited}"
    )

    prediction = agent._finish(
        trajectory, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5)
    )

    field = agent.extract.calls[0][PRESENTED_RESULTS_FIELD]
    assert cited in field
    assert uncited not in field
    assert prediction.final_answer == "composed"
    assert prediction.presented_results["handles"][0]["handle_id"] == cited
    assert prediction.presented_results["handles"][0]["reason"] == REASON_CITED


def test_a_flagged_command_is_resolved_from_the_trajectory_without_a_citation(
    host, monkeypatch
):
    """With NO skill override — i.e. exactly what arm A runs."""
    monkeypatch.setattr(result_handles.tracing, "current_host", lambda: host)
    declared, other = "e3" + "0" * 30, "e4" + "0" * 30
    host.result_handles.put(
        _record(declared, command_name="open_portrait", items=PORTRAIT,
                presentation=True)
    )
    host.result_handles.put(_record(other, command_name="list_groups"))
    agent = _loop_agent(host)
    trajectory = _trajectory(declared, other, closing_thought="done")

    agent._finish(trajectory, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5))

    field = agent.extract.calls[0][PRESENTED_RESULTS_FIELD]
    assert declared in field
    assert other not in field


def test_the_flat_arm_and_a_plan_leaf_resolve_the_same_handles(host, monkeypatch):
    """The claim the arm-invariance fix exists for.

    Arm A cannot supply a `presents:` list at all — `off` never opens
    `_skills/` — so both overrides here are computed the way the runtime
    computes them: `None` for the flat site, a real leaf whose skill declares
    nothing for the plan site. If the DEFAULT lived on the skill rather than on
    the producing command, the flat field would be strictly poorer than the leaf
    field and the endpoint would score the difference as an effect of
    decomposition.
    """
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    monkeypatch.setattr(result_handles.tracing, "current_host", lambda: host)
    flagged, plain = "ea" + "0" * 30, "eb" + "0" * 30

    flat_ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    flat_ctx._turn_plan = None

    leaf = SimpleNamespace(goal_id="leaf", skill="department-roster-walk",
                           parent_goal_id=None)
    plan_ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    plan_ctx._turn_plan = SimpleNamespace(node=lambda goal_id: None)
    plan_ctx._catalog_for_plan_execution = lambda plan: SimpleNamespace(
        get=lambda name: SimpleNamespace(presents=())  # the skill declares nothing
    )

    overrides = [
        flat_ctx._presentation_commands_for(None),
        plan_ctx._presentation_commands_for(leaf),
    ]
    assert overrides == [frozenset(), frozenset()]

    fields = []
    for override in overrides:
        host.result_handles.clear()
        host.result_handles.put(
            _record(flagged, command_name="show_holders", items=HOLDERS[:4],
                    presentation=True)
        )
        host.result_handles.put(
            _record(plain, command_name="list_groups", items=HOLDERS[:4])
        )
        agent = _loop_agent(host, presentation_commands=override)
        agent._finish(
            _trajectory(flagged, plain, closing_thought="done"),
            {"user_query": "q"},
            LogicalTurnBudget(iteration_limit=5),
        )
        fields.append(agent.extract.calls[0][PRESENTED_RESULTS_FIELD])

    assert fields[0] == fields[1]
    assert flagged in fields[0]
    assert plain not in fields[0]


def test_a_handle_cited_long_before_the_closing_step_is_not_resolved(host, monkeypatch):
    """"Do not resolve everything" erodes exactly here if the window is the run."""
    monkeypatch.setattr(result_handles.tracing, "current_host", lambda: host)
    stale = "e5" + "0" * 30
    host.result_handles.put(_record(stale, items=HOLDERS[:4]))
    agent = _loop_agent(host)
    trajectory = {}
    for index in range(CLOSING_THOUGHT_STEPS + 3):
        trajectory[f"thought_{index}"] = (
            f"using result_handle={stale}" if index == 0 else "working"
        )
        trajectory[f"tool_name_{index}"] = "show_holders"
        trajectory[f"observation_{index}"] = "opaque"

    agent._finish(trajectory, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5))

    assert stale not in agent.extract.calls[0][PRESENTED_RESULTS_FIELD]


def test_the_field_is_resolved_once_and_reused_across_extraction_retries(
    host, monkeypatch
):
    """The snapshot is sealed; the payload the retries compose from must be too."""
    monkeypatch.setattr(result_handles.tracing, "current_host", lambda: host)
    handle_id = "e6" + "0" * 30
    host.result_handles.put(_record(handle_id, items=HOLDERS[:4]))
    agent = _loop_agent(host)
    calls: list[int] = []

    def once(extract, trajectory, input_args):
        calls.append(1)
        return None if len(calls) > 1 else SimpleNamespace(rewrite="try again")

    agent._consult_finish_policy = once
    trajectory = _trajectory(handle_id, closing_thought=f"done {handle_id}")

    agent._finish(trajectory, {"user_query": "q"}, LogicalTurnBudget(iteration_limit=5))

    fields = [call[PRESENTED_RESULTS_FIELD] for call in agent.extract.calls]
    assert len(fields) == 2
    assert fields[0] == fields[1]


def test_a_citation_survives_the_closing_thought_bound(host, monkeypatch):
    """`MAX_NEXT_THOUGHT_CHARS` truncates mid-token, so an id can be cut in half.

    The ids are parsed at the bounding choke point BEFORE `_bound_text` runs, so
    the citation reaches the resolver even though the trajectory no longer holds
    a readable id.
    """
    monkeypatch.setattr(result_handles.tracing, "current_host", lambda: host)
    handle_id = "f1" + "0" * 30
    host.result_handles.put(_record(handle_id, items=HOLDERS[:4]))
    agent = _loop_agent(host)
    # Placed so the truncation lands INSIDE the id: 16 of its 32 characters
    # survive the cut, which is the failure mode — a half id is not a citation
    # and re-reading the bounded thought would find nothing.
    cut = MAX_NEXT_THOUGHT_CHARS - len(THOUGHT_TRUNCATION_NOTICE)
    citation = f"result_handle={handle_id}"
    pred = SimpleNamespace(
        # ...and a tail long enough that the thought is over the bound at all.
        next_thought=(
            "x" * (cut - len(citation) + 16) + citation + " and then " + "y" * 200
        ),
    )

    agent._bound_prediction_thought(pred)

    assert handle_id[:16] in pred.next_thought  # half of it is still there...
    assert handle_id not in pred.next_thought  # ...which is not a citation
    assert result_handles.parse_handle_ids(pred.next_thought) == ()
    assert agent._cited_handle_ids({"thought_0": pred.next_thought}) == (handle_id,)

    agent._finish(
        {"thought_0": pred.next_thought, "observation_0": "opaque"},
        {"user_query": "q"},
        LogicalTurnBudget(iteration_limit=5),
    )

    assert handle_id in agent.extract.calls[0][PRESENTED_RESULTS_FIELD]


def test_an_unbounded_thought_records_the_same_ids_it_shows(host):
    agent = _loop_agent(host)
    handle_id = "f2" + "0" * 30
    pred = SimpleNamespace(next_thought=f"done; result_handle={handle_id}")

    agent._bound_prediction_thought(pred)

    assert pred.next_thought == f"done; result_handle={handle_id}"
    assert agent._step_citations == [(handle_id,)]


def test_the_closing_window_is_the_last_steps_of_the_recorded_citations(host):
    agent = _loop_agent(host)
    for index in range(CLOSING_THOUGHT_STEPS + 2):
        agent._bound_prediction_thought(
            SimpleNamespace(next_thought=f"step result_handle={chr(97 + index) * 32}")
        )

    cited = agent._cited_handle_ids({})

    assert len(cited) == CLOSING_THOUGHT_STEPS
    assert cited[-1] == chr(97 + CLOSING_THOUGHT_STEPS + 1) * 32
    assert "a" * 32 not in cited


def test_recorded_citations_fall_back_to_the_trajectory_when_absent(host):
    """A cross-process resume of a blob written before this record existed."""
    agent = _loop_agent(host)
    handle_id = "f3" + "0" * 30

    assert agent._cited_handle_ids({"thought_0": f"done {handle_id}"}) == (handle_id,)


def test_recorded_citations_cross_a_process_boundary(host):
    """`export_suspended` carries them, so a resumed turn keeps the unbounded ids."""
    agent = _loop_agent(host)
    handle_id = "f4" + "0" * 30
    agent._bound_prediction_thought(
        SimpleNamespace(next_thought=f"done result_handle={handle_id}")
    )
    agent._suspended = {
        "trajectory": {"thought_0": "cut"},
        "idx": 0,
        "input_args": {"user_query": "q"},
        "max_iters": 5,
        "clarification": "which?",
    }
    agent._budget = None
    agent.presentation_commands = frozenset(
        {"show_holders", "open_portrait"}
    )

    blob = agent.export_suspended()

    assert blob["step_citations"] == [[handle_id]]
    assert blob["presentation_commands"] == [
        "open_portrait",
        "show_holders",
    ]

    restored = _loop_agent(host)
    restored.max_iters = 5
    restored.import_suspended(blob)

    assert restored._cited_handle_ids({}) == (handle_id,)
    assert restored.presentation_commands == frozenset(
        {"show_holders", "open_portrait"}
    )


def test_resolution_failure_degrades_the_answer_and_never_the_turn(host, monkeypatch):
    monkeypatch.setattr(
        result_handles,
        "resolve_for_presentation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    agent = _loop_agent(host)

    prediction = agent._finish(
        _trajectory("e7" + "0" * 30, closing_thought="done"),
        {"user_query": "q"},
        LogicalTurnBudget(iteration_limit=5),
    )

    assert prediction.final_answer == "composed"
    assert agent.extract.calls[0][PRESENTED_RESULTS_FIELD] == (
        result_handles.PRESENTED_RESULTS_NONE
    )


def test_a_resumed_turn_resolves_the_handles_its_suspended_half_issued(
    host, monkeypatch
):
    """`resume` finishes through the same `_finish`, and the store survives."""
    monkeypatch.setattr(result_handles.tracing, "current_host", lambda: host)
    handle_id = "e8" + "0" * 30
    host.result_handles.put(_record(handle_id, items=HOLDERS[:4]))
    # Round-trip the store the way a cross-process suspension does.
    restored = ResultHandleStore()
    restored.apply_state(host.result_handles.to_state())
    host.result_handles = restored

    agent = _loop_agent(host)
    agent._suspended = {
        "trajectory": _trajectory(handle_id, closing_thought=f"done {handle_id}"),
        "idx": 1,
        "input_args": {"user_query": "q"},
        "max_iters": 5,
        "clarification": "which?",
    }
    agent.tools = {"finish": SimpleNamespace(func=lambda: "Completed.")}
    agent.current_trajectory = {}
    agent._on_step_complete = None
    agent._censored_last_run = False
    agent._run_loop = lambda *args, **kwargs: None

    agent.resume("the answer")

    assert handle_id in agent.extract.calls[0][PRESENTED_RESULTS_FIELD]


def test_recorded_views_survive_a_state_round_trip(host):
    handle_id = "e9" + "0" * 30
    host.result_handles.put(_record(handle_id))
    result_handles.fetch_page(handle_id, host=host)
    result_handles.fetch_page(
        handle_id,
        cursor=result_handles.encode_cursor(
            20,
            handle_id=handle_id,
        ),
        host=host,
    )

    restored = ResultHandleStore()
    restored.apply_state(host.result_handles.to_state())

    assert len(restored.get(handle_id).views) == 2
    entry = _entry(
        result_handles.resolve_for_presentation(
            cited=[handle_id], host=_Host(restored)
        ),
        handle_id,
    )
    assert entry.rows == HOLDERS[:40]


def test_legacy_schema_eight_view_offsets_migrate_to_exact_indices(host):
    handle_id = "f9" + "0" * 30
    state = _record(handle_id).model_dump(mode="json")
    state["views"] = [
        {"offset": 20, "count": 5, "contains": None},
        {"offset": 0, "count": 3, "contains": "identity_04"},
    ]

    restored = ResultHandleStore()
    restored.apply_state([state])
    views = restored.get(handle_id).views

    assert views[0].item_indices == tuple(range(20, 25))
    assert views[0].matched == len(HOLDERS)
    assert views[1].item_indices == (40, 41, 42)
    assert views[1].matched == 10
    assert all(view.contains is None for view in views)


def test_filtered_resolution_reports_view_filters_and_post_filter_match_count(host):
    handle_id = "fa" + "0" * 30
    host.result_handles.put(_record(handle_id, page_size=4))

    first = result_handles.fetch_page(
        handle_id,
        contains="identity_00",
        host=host,
    )
    result_handles.fetch_page(
        handle_id,
        cursor=first.next_cursor,
        contains="IDENTITY_00",
        host=host,
    )

    entry = _entry(
        result_handles.resolve_for_presentation(
            cited=[handle_id],
            host=host,
        ),
        handle_id,
    )

    assert entry.rows == HOLDERS[:8]
    assert entry.matched == 10
    assert len(entry.view_filters) == 1
    assert entry.view_filters[0]["matched"] == 10
    assert entry.view_filters[0]["filters"] == {
        "contains": "identity_00"
    }
    assert '"matched":10' in entry.as_text()
    assert "identity_009" not in entry.as_text()


def test_several_view_filters_union_rows_in_producer_order_and_survive_state(host):
    handle_id = "fb" + "0" * 30
    host.result_handles.put(_record(handle_id, page_size=20))

    result_handles.fetch_page(
        handle_id,
        contains="identity_02",
        host=host,
    )
    result_handles.fetch_page(
        handle_id,
        contains="identity_00",
        host=host,
    )

    restored = ResultHandleStore()
    restored.apply_state(host.result_handles.to_state())
    entry = _entry(
        result_handles.resolve_for_presentation(
            cited=[handle_id],
            host=_Host(restored),
        ),
        handle_id,
    )

    assert entry.rows == (*HOLDERS[:10], *HOLDERS[20:30])
    assert entry.matched == 20
    assert len(entry.view_filters) == 2
    assert {
        view["filters"]["contains"] for view in entry.view_filters
    } == {"identity_00", "identity_02"}
    restored_views = restored.get(handle_id).views
    assert all(view.contains is None for view in restored_views)
    assert all(view.item_indices for view in restored_views)
    assert all(view.matched_indices for view in restored_views)


# ----------------------------------------------------------------------
# 8. Arm invariance: one road, three call sites
# ----------------------------------------------------------------------

_WEC = Path(__file__).resolve().parents[1] / "fastworkflow" / "workflow_execution_context.py"


def _agent_call_sites() -> list[ast.Call]:
    """Every `self._call_agent(...)` whose lambda invokes the tool agent.

    Structural rather than behavioural on purpose: "the same code path runs for
    A, B and C" is a claim about the call sites, and three call sites that agree
    today are exactly what drifts.
    """
    tree = ast.parse(_WEC.read_text(encoding="utf-8"))
    sites: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "_call_agent"):
            continue
        source = ast.unparse(node)
        if "self._workflow_tool_agent(" in source:
            sites.append(node)
    return sites


def test_all_three_arm_call_sites_pass_presentation_commands():
    sites = _agent_call_sites()

    assert len(sites) == 3, (
        "the flat (A), shadow and plan-leaf (B/C) sites are the three that run "
        f"the tool agent; found {len(sites)}"
    )
    for site in sites:
        keywords = {keyword.arg for keyword in site.keywords}
        assert "presentation_commands" in keywords, ast.unparse(site)


def test_the_resume_path_keeps_the_declaration_its_suspended_half_ran_under():
    """A resumed run continues the same leaf; clearing it would drop the rows the
    second half of that leaf's answer presents."""
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    tree = ast.parse(_WEC.read_text(encoding="utf-8"))
    resume_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_call_agent"
        and "self._workflow_tool_agent.resume(" in ast.unparse(node)
    ]
    assert len(resume_sites) == 1
    assert "presentation_commands" not in {
        keyword.arg for keyword in resume_sites[0].keywords
    }

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    agent.presentation_commands = frozenset({"show_holders"})
    ctx._workflow_tool_agent = agent
    ctx._turn_presented_results = []
    ctx._turn_active_leaf = "leaf-1"
    ctx._agent_dspy_context = lambda: (SimpleNamespace(model="m"), None)
    import fastworkflow.workflow_execution_context as wec

    original_start, original_end = wec.tracing.start_span, wec.tracing.end_span
    wec.tracing.start_span = lambda *a, **k: None
    wec.tracing.end_span = lambda *a, **k: None
    try:
        ctx._call_agent(lambda: SimpleNamespace(final_answer="x"), resumed=True)
    finally:
        wec.tracing.start_span, wec.tracing.end_span = original_start, original_end

    assert agent.presentation_commands == frozenset({"show_holders"})


def test_call_agent_applies_the_declaration_to_the_agent_it_runs():
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    agent = fastWorkflowReAct.__new__(fastWorkflowReAct)
    ctx._workflow_tool_agent = agent
    ctx._turn_presented_results = []
    ctx._turn_active_leaf = None
    seen: list[frozenset] = []

    def fake_call():
        seen.append(agent.presentation_commands)
        return SimpleNamespace(final_answer="x")

    ctx._agent_dspy_context = lambda: (SimpleNamespace(model="m"), None)
    ctx._trace_sink = None
    import fastworkflow.workflow_execution_context as wec

    original_start, original_end = wec.tracing.start_span, wec.tracing.end_span
    wec.tracing.start_span = lambda *a, **k: None
    wec.tracing.end_span = lambda *a, **k: None
    try:
        ctx._call_agent(fake_call, presentation_commands={"show_holders"})
    finally:
        wec.tracing.start_span, wec.tracing.end_span = original_start, original_end

    assert seen == [frozenset({"show_holders"})]


def test_the_flat_arms_pass_an_empty_declaration_because_off_reads_no_skills():
    """`FW_PLAN_DECOMPOSITION=off` never opens `_skills/`; empty is honest here."""
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._turn_plan = None

    assert ctx._presentation_commands_for(None) == frozenset()


def test_presentation_commands_walk_the_leaf_up_to_the_skill_that_presents():
    """A command-sequence leaf has no skill of its own; its parent's declaration
    is the one that governs the answer."""
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    leaf = SimpleNamespace(goal_id="leaf", skill=None, parent_goal_id="root")
    root = SimpleNamespace(goal_id="root", skill="unit-review-packet",
                           parent_goal_id=None)
    plan = SimpleNamespace(node=lambda goal_id: {"leaf": leaf, "root": root}.get(goal_id))
    catalog = {
        "unit-review-packet": SimpleNamespace(presents=("show_holders", "open_portrait"))
    }

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._turn_plan = plan
    ctx._catalog_for_plan_execution = lambda p: SimpleNamespace(get=catalog.get)

    assert ctx._presentation_commands_for(leaf) == frozenset(
        {"show_holders", "open_portrait"}
    )


def test_an_unreadable_catalogue_costs_rows_and_not_the_turn():
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    leaf = SimpleNamespace(goal_id="leaf", skill="missing", parent_goal_id=None)
    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._turn_plan = SimpleNamespace(node=lambda goal_id: None)
    ctx._catalog_for_plan_execution = lambda p: (_ for _ in ()).throw(RuntimeError())

    assert ctx._presentation_commands_for(leaf) == frozenset()


def test_the_turn_record_and_the_span_both_carry_the_resolution():
    """Two readers, two roads: an evaluator reads the turn record, a trace reader
    reads the span, and neither can reconstruct the other's."""
    from fastworkflow.workflow_execution_context import (
        WorkflowExecutionContext,
        _agent_result_attributes,
    )

    evidence = {
        "handles": [{"handle_id": "f" * 32, "reason": "cited", "rows": 20,
                     "bytes": 400, "trimmed": True}],
        "bytes": 400,
        "max_bytes": 512,
        "trimmed": True,
        "unresolved": [],
    }
    attributes = _agent_result_attributes(
        SimpleNamespace(final_answer="a", presented_results=evidence), 1
    )

    assert attributes["presented_result_handles"] == evidence["handles"]
    assert attributes["presented_result_bytes"] == 400
    assert attributes["presented_result_trimmed"] is True

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._turn_presented_results = []
    ctx._turn_active_leaf = "leaf-2"
    ctx._remember_presented_results(
        SimpleNamespace(presented_results=evidence)
    )

    assert ctx._turn_presented_results == [{"leaf_goal_id": "leaf-2", **evidence}]


def test_a_turn_that_resolved_nothing_writes_no_provenance():
    """Absent, not an empty dict: v3 and v4 rows agree for a workflow that never
    opted in."""
    from fastworkflow.workflow_execution_context import (
        WorkflowExecutionContext,
        _agent_result_attributes,
    )

    attributes = _agent_result_attributes(SimpleNamespace(final_answer="a"), 1)
    assert "presented_result_handles" not in attributes

    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._turn_presented_results = []
    ctx._turn_active_leaf = None
    ctx._remember_presented_results(
        SimpleNamespace(
            presented_results={"handles": [], "bytes": 0, "max_bytes": 1,
                               "trimmed": False, "unresolved": []}
        )
    )

    assert ctx._turn_presented_results == []


def test_a_fully_omitted_handle_still_writes_turn_provenance():
    ctx = WorkflowExecutionContext.__new__(WorkflowExecutionContext)
    ctx._turn_presented_results = []
    ctx._turn_active_leaf = "leaf-omitted"
    evidence = {
        "handles": [],
        "bytes": 0,
        "field_bytes": 128,
        "max_bytes": 128,
        "trimmed": True,
        "omitted_handles": ["a" * 32],
        "truncation_classification": "infrastructure-truncated",
        "unresolved": [],
    }

    ctx._remember_presented_results(
        SimpleNamespace(presented_results=evidence)
    )

    assert ctx._turn_presented_results == [
        {"leaf_goal_id": "leaf-omitted", **evidence}
    ]


def test_the_span_contract_declares_the_keys_the_emitter_writes():
    from fastworkflow import tracing

    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_AGENT_EXECUTE]
    assert tracing.SPAN_CONTRACT_VERSION == 5
    assert contract.version >= 6
    assert {
        "presented_result_handles",
        "presented_result_bytes",
        "presented_result_field_bytes",
        "presented_result_trimmed",
        "presented_result_omitted_handles",
        "presented_result_truncation_classification",
    } <= set(contract.attributes)


# ----------------------------------------------------------------------
# 9. `presents:` in the skill schema
# ----------------------------------------------------------------------


def _manifest(commands=("show_holders", "open_portrait", "known")):
    return SimpleNamespace(
        features={"skills_v1": "enforce"},
        commands={f"Context/{name}": SimpleNamespace() for name in commands},
        skills_fingerprint=None,
    )


def _write_skill(workflow: Path, name: str, *, presents: str = "") -> Path:
    directory = workflow / "_skills" / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Exercise {name}.\n"
        "level: atomic\n"
        f"{presents}"
        "---\n\n"
        f"# {name}\n\n1. `known`\n",
        encoding="utf-8",
    )
    return path


def test_presents_parses_as_a_scalar_list(tmp_path):
    _write_skill(
        tmp_path, "leaf", presents="presents:\n  - show_holders\n  - open_portrait\n"
    )

    catalog = load_skill_catalog(str(tmp_path), _manifest())

    assert catalog["leaf"].presents == ("show_holders", "open_portrait")


def test_presents_is_optional_and_absent_means_absent(tmp_path):
    _write_skill(tmp_path, "leaf")

    assert load_skill_catalog(str(tmp_path), _manifest())["leaf"].presents == ()


def test_presents_deduplicates_in_declaration_order(tmp_path):
    _write_skill(
        tmp_path,
        "leaf",
        presents="presents:\n  - show_holders\n  - open_portrait\n  - show_holders\n",
    )

    catalog = load_skill_catalog(str(tmp_path), _manifest())

    assert catalog["leaf"].presents == ("show_holders", "open_portrait")


def test_presents_must_name_a_command_the_manifest_declares(tmp_path):
    path = _write_skill(tmp_path, "leaf", presents="presents:\n  - no_such_command\n")

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert "presents-names-a-declared-command" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_presents_may_not_name_framework_vocabulary(tmp_path):
    """A parameter hint is not a producer: it can never issue a handle."""
    _write_skill(tmp_path, "leaf", presents="presents:\n  - available_from\n")

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert "presents-names-a-declared-command" in str(excinfo.value)


def test_presents_entries_must_be_bare_command_names(tmp_path):
    _write_skill(tmp_path, "leaf", presents="presents:\n  - Context/show_holders\n")

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert "presents-shape" in str(excinfo.value)


def test_presents_must_use_indented_entries(tmp_path):
    _write_skill(tmp_path, "leaf", presents="presents: show_holders\n")

    with pytest.raises(SkillCatalogError) as excinfo:
        load_skill_catalog(str(tmp_path), _manifest())

    assert "must use indented" in str(excinfo.value)


def test_presents_is_not_on_the_selector_card(tmp_path):
    """FW-REQ-006 clause 3: the selector sees a card, and this is execution
    metadata the selector has no use for."""
    _write_skill(tmp_path, "leaf", presents="presents:\n  - show_holders\n")

    card = load_skill_catalog(str(tmp_path), _manifest())["leaf"].card().as_dict()

    assert "presents" not in card
