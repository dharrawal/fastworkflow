"""Generic contracts for the task-first structured selector."""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import dspy
import pytest
from dspy.clients import lm as dspy_lm
from dspy.utils import DummyLM
from dspy.utils.exceptions import LMConfigurationError, LMUnexpectedError
from litellm import ModelResponse

from fastworkflow import workflow_agent
from fastworkflow.plan import (
    InvocationEvidence,
    PlanConfigurationError,
    PlanMode,
    SourceSpan,
    expand,
)
from fastworkflow.skill_catalog import Skill, SkillCatalog, Slot, parse_steps
from fastworkflow.utils.dspy_utils import (
    SELECTOR_ADAPTER_IDENTITY,
    SingleCallJSONAdapter,
)
from fastworkflow.workflow_agent import _plan_decomposition_point, select_skills
from fastworkflow.workflow_agent import SkillSelectionSignature


def _skill(
    name: str,
    level: str,
    *,
    slot_name: str | None = "subject",
    binding_kind: str = "exact_text",
    normalizer: str | None = None,
    is_list: bool = False,
    uses: tuple[str, ...] = (),
) -> Skill:
    slots = (
        ()
        if slot_name is None
        else (
            Slot(
                name=slot_name,
                required=True,
                on_repeat="ask once, then use a declared default",
                binding_kind=binding_kind,
                normalizer=normalizer,
                list=is_list,
            ),
        )
    )
    body = "1. `perform_work`"
    return Skill(
        name=name,
        description=f"Complete the operator task named {name}.",
        level=level,
        goal=(
            f"{{{slot_name}}} is handled."
            if level != "atomic" and slot_name
            else "The task is handled."
        ),
        slots=slots,
        uses=uses,
        body=body,
        path=f"/generic/_skills/{name}/SKILL.md",
        content_hash=f"sha256:{name}",
        steps=parse_steps(body, uses=uses, path=name),
    )


@pytest.fixture
def task_catalog() -> SkillCatalog:
    review = _skill("review-subject", "task")
    audit = _skill("audit-resource", "task", slot_name="resource")
    atomic = _skill("inspect-subject", "atomic")
    composite = _skill(
        "review-packet",
        "composite",
        uses=("review-subject",),
    )
    return SkillCatalog(
        {skill.name: skill for skill in (review, audit, atomic, composite)},
        fingerprint="sha256:generic",
        mode="enforce",
    )


def _invocation(
    skill_name: str,
    slot_name: str,
    value: str,
    *,
    source_text: str | None = None,
) -> dict:
    return {
        "skill_name": skill_name,
        "bindings": [
            {
                "slot_name": slot_name,
                "value": value,
                "source_text": source_text or value,
            }
        ],
    }


def _dummy_lm(*answers: dict) -> DummyLM:
    return DummyLM(
        list(answers),
        adapter=SingleCallJSONAdapter(),
    )


def _model_response(payload: dict) -> ModelResponse:
    return ModelResponse(
        model="generic-selector-model",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload),
                },
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    )


def test_first_attempt_valid_selection_has_exact_typed_provenance(task_catalog):
    lm = _dummy_lm({"invocations": [_invocation("review-subject", "subject", "Casey")]})

    selected = select_skills("Review Casey.", task_catalog, lm)

    assert selected.application_attempt_count == 1
    assert selected.provider_call_count == 1
    assert selected.provider_response_count == 1
    assert selected.adapter_identity == SELECTOR_ADAPTER_IDENTITY
    assert len(lm.history) == 1
    assert selected[0].provenance == {
        "subject": InvocationEvidence(
            kind="exact_text",
            source_spans=(SourceSpan(start=7, end=12, text="Casey"),),
        )
    }


def test_one_schema_validation_retry_then_valid_is_exactly_two_calls(task_catalog):
    lm = _dummy_lm(
        {"unexpected": "not a selector response"},
        {"invocations": [_invocation("review-subject", "subject", "Casey")]},
    )

    selected = select_skills("Review Casey.", task_catalog, lm)

    assert selected.application_attempt_count == 2
    assert selected.provider_call_count == 2
    assert selected.provider_response_count == 2
    assert selected.validation_retry_used is True
    assert len(selected.validation_errors) == 1
    assert selected.validation_errors[0].startswith("AdapterParseError:")
    assert [attempt["provider_calls"] for attempt in selected.attempts] == [1, 1]
    assert [attempt["provider_responses"] for attempt in selected.attempts] == [1, 1]
    assert len(lm.history) == 2


def test_second_invalid_response_fails_without_a_third_call(task_catalog):
    lm = _dummy_lm(
        {"unexpected": "first invalid response"},
        {"unexpected": "second invalid response"},
    )

    with pytest.raises(ValueError) as caught:
        select_skills("Review Casey.", task_catalog, lm)

    assert getattr(caught.value, "selection_attempt_count") == 2
    assert getattr(caught.value, "selection_provider_call_count") == 2
    assert getattr(caught.value, "selection_provider_response_count") == 2
    assert getattr(caught.value, "selection_adapter_identity") == (
        SELECTOR_ADAPTER_IDENTITY
    )
    assert len(getattr(caught.value, "selection_attempts")) == 2
    assert len(lm.history) == 2


def test_no_automatic_adapter_fallback_or_hidden_lm_call(task_catalog):
    lm = _dummy_lm(
        {"invocations": "wrong top-level type"},
        {"invocations": [_invocation("review-subject", "subject", "Casey")]},
    )

    selected = select_skills("Review Casey.", task_catalog, lm)

    assert selected.application_attempt_count == selected.provider_call_count == 2
    assert selected.provider_response_count == 2
    assert len(lm.history) == 2
    assert all(attempt["provider_calls"] == 1 for attempt in selected.attempts)
    assert all(attempt["provider_responses"] == 1 for attempt in selected.attempts)
    assert all(attempt["raw_response"] for attempt in selected.attempts)


def test_dspy_lm_num_retries_is_not_forwarded_twice(
    task_catalog,
    monkeypatch,
):
    dispatches = []

    def fake_completion(*, request, num_retries, cache):
        dispatches.append(
            {
                "request": request,
                "num_retries": num_retries,
                "cache": cache,
            }
        )
        return _model_response(
            {
                "invocations": [
                    _invocation("review-subject", "subject", "Casey")
                ]
            }
        )

    monkeypatch.setattr(dspy_lm, "litellm_completion", fake_completion)
    lm = dspy.LM(
        "openai/generic-selector-model",
        cache=False,
        num_retries=0,
        temperature=0.0,
        max_tokens=256,
    )

    selected = select_skills("Review Casey.", task_catalog, lm)

    assert len(dispatches) == 1
    assert dispatches[0]["num_retries"] == 0
    assert "num_retries" not in dispatches[0]["request"]
    assert selected.application_attempt_count == 1
    assert selected.provider_call_count == 1
    assert selected.provider_response_count == 1


def test_cached_selector_result_is_not_counted_as_provider_traffic(
    task_catalog,
    monkeypatch,
):
    response = _model_response(
        {
            "invocations": [
                _invocation("review-subject", "subject", "Casey")
            ]
        }
    )
    response.cache_hit = True
    monkeypatch.setattr(
        dspy_lm,
        "litellm_completion",
        lambda **_kwargs: response,
    )
    lm = dspy.LM(
        "openai/generic-selector-model",
        cache=False,
        num_retries=0,
        temperature=0.0,
        max_tokens=256,
    )

    selected = select_skills("Review Casey.", task_catalog, lm)

    assert selected.application_attempt_count == 1
    assert selected.provider_call_count == 0
    assert selected.provider_response_count == 0
    assert len(lm.history) == 1


def test_duplicate_num_retries_fails_before_provider_and_is_not_counted(
    task_catalog,
    monkeypatch,
):
    provider_entries = 0

    def provider_boundary(**_kwargs):
        nonlocal provider_entries
        provider_entries += 1
        return _model_response({})

    def duplicate_sensitive_completion(*, request, num_retries, cache):
        return provider_boundary(
            cache=cache,
            num_retries=num_retries,
            **request,
        )

    monkeypatch.setattr(
        dspy_lm,
        "litellm_completion",
        duplicate_sensitive_completion,
    )
    lm = dspy.LM(
        "openai/generic-selector-model",
        cache=False,
        num_retries=0,
        temperature=0.0,
        max_tokens=256,
    )
    adapter = SingleCallJSONAdapter()
    selector = dspy.Predict(SkillSelectionSignature)

    with dspy.context(lm=lm, adapter=adapter):
        with pytest.raises(
            LMUnexpectedError,
            match="multiple values for keyword argument 'num_retries'",
        ):
            selector(
                utterance="Review Casey.",
                catalogue=json.dumps(
                    [card.as_dict() for card in task_catalog.cards()]
                ),
                validation_feedback="",
                config={"num_retries": 0},
            )

    assert provider_entries == 0
    assert adapter.provider_call_count == 0
    assert adapter.provider_response_count == 0


def test_provider_failure_is_counted_once_and_never_validation_retried(
    task_catalog,
    monkeypatch,
):
    provider_entries = 0

    def failing_completion(*, request, num_retries, cache):
        nonlocal provider_entries
        provider_entries += 1
        for callback in request["failure_callback"]:
            callback.log_failure_event(
                kwargs={"first_api_call_start_time": object()},
                response_obj=None,
                start_time=None,
                end_time=None,
            )
        raise RuntimeError("generic provider failure")

    monkeypatch.setattr(dspy_lm, "litellm_completion", failing_completion)
    lm = dspy.LM(
        "openai/generic-selector-model",
        cache=False,
        num_retries=0,
        temperature=0.0,
        max_tokens=256,
    )

    with pytest.raises(LMUnexpectedError) as caught:
        select_skills("Review Casey.", task_catalog, lm)

    assert provider_entries == 1
    assert getattr(caught.value, "selection_attempt_count") == 1
    assert getattr(caught.value, "selection_provider_call_count") == 1
    assert getattr(caught.value, "selection_provider_response_count") == 0
    assert len(getattr(caught.value, "selection_attempts")) == 1


def test_selector_rejects_hidden_provider_retries_before_dispatch(
    task_catalog,
    monkeypatch,
):
    monkeypatch.setattr(
        dspy_lm,
        "litellm_completion",
        lambda **_kwargs: pytest.fail("provider dispatch should not occur"),
    )
    lm = dspy.LM(
        "openai/generic-selector-model",
        cache=False,
        num_retries=1,
        temperature=0.0,
        max_tokens=256,
    )

    with pytest.raises(LMConfigurationError) as caught:
        select_skills("Review Casey.", task_catalog, lm)

    assert getattr(caught.value, "selection_attempt_count") == 1
    assert getattr(caught.value, "selection_provider_call_count") == 0
    assert getattr(caught.value, "selection_provider_response_count") == 0


def test_source_text_is_validated_and_offsets_are_derived(task_catalog):
    invalid = _invocation(
        "review-subject",
        "subject",
        "Casey",
        source_text="Riley",
    )
    lm = _dummy_lm(
        {"invocations": [invalid]},
        {"invocations": [invalid]},
    )

    with pytest.raises(PlanConfigurationError, match="not copied verbatim"):
        select_skills("Review Casey, not Riley.", task_catalog, lm)

    assert len(lm.history) == 2


def test_binding_error_guides_one_retry_to_valid_provenance(task_catalog):
    lm = _dummy_lm(
        {
            "invocations": [
                _invocation(
                    "review-subject",
                    "subject",
                    "Casey",
                    source_text="Riley",
                )
            ]
        },
        {"invocations": [_invocation("review-subject", "subject", "Casey")]},
    )

    selected = select_skills("Review Casey, not Riley.", task_catalog, lm)

    assert selected.validation_retry_used is True
    assert "not copied verbatim" in selected.validation_errors[0]
    assert selected[0].provenance["subject"].source_spans == (
        SourceSpan(start=7, end=12, text="Casey"),
    )
    assert selected.provider_call_count == 2


def test_task_cards_only_and_atomic_or_composite_names_are_rejected(
    task_catalog,
    monkeypatch,
):
    calls = []

    class ForbiddenSelector:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            forbidden_name = "inspect-subject" if len(calls) == 1 else "review-packet"
            return SimpleNamespace(
                invocations=[_invocation(forbidden_name, "subject", "Casey")]
            )

    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda _signature: ForbiddenSelector(),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )

    with pytest.raises(PlanConfigurationError, match="must name a task skill"):
        select_skills(
            "Review Casey.",
            task_catalog,
            SimpleNamespace(model="selector-model"),
        )

    cards = json.loads(calls[0]["catalogue"])
    assert {card["name"] for card in cards} == {
        "review-subject",
        "audit-resource",
    }
    assert all(card["level"] == "task" for card in cards)
    assert len(calls) == 2


def test_multi_task_coverage_is_preserved_and_duplicates_are_rejected(
    task_catalog,
):
    utterance = "Review Casey, then audit Atlas."
    response = {
        "invocations": [
            _invocation("review-subject", "subject", "Casey"),
            _invocation("audit-resource", "resource", "Atlas"),
        ]
    }
    selected = select_skills(utterance, task_catalog, _dummy_lm(response))
    plan = expand(
        task_catalog,
        selected,
        utterance,
        mode=PlanMode.ENFORCE,
        selection_model=selected.selection_model,
    )

    assert [invocation.skill_name for invocation in selected] == [
        "review-subject",
        "audit-resource",
    ]
    assert plan.requested_public_task_keys == plan.compiled_public_task_keys
    assert len(plan.compiled_public_task_keys) == 2

    duplicate = {
        "invocations": [
            _invocation("review-subject", "subject", "Casey"),
            _invocation("review-subject", "subject", "Casey"),
        ]
    }
    duplicate_lm = _dummy_lm(duplicate, duplicate)
    with pytest.raises(
        PlanConfigurationError,
        match="repeats canonical invocation",
    ):
        select_skills("Review Casey.", task_catalog, duplicate_lm)
    assert len(duplicate_lm.history) == 2


def test_uncovered_sequenced_task_gets_one_bounded_validation_retry(
    task_catalog,
):
    utterance = "Review Casey, then audit Atlas."
    lm = _dummy_lm(
        {
            "invocations": [
                _invocation("review-subject", "subject", "Casey"),
            ]
        },
        {
            "invocations": [
                _invocation("review-subject", "subject", "Casey"),
                _invocation("audit-resource", "resource", "Atlas"),
            ]
        },
    )

    selected = select_skills(utterance, task_catalog, lm)

    assert [item.skill_name for item in selected] == [
        "review-subject",
        "audit-resource",
    ]
    assert selected.validation_retry_used is True
    assert "coverage unproven" in selected.validation_errors[0]
    assert selected.coverage_segments == ((0, 13), (19, 31))
    assert len(lm.history) == 2
    retry_input = json.dumps(lm.history[1], default=str)
    assert "coverage unproven" in retry_input
    assert "review-subject" not in selected.validation_errors[0]
    assert "audit-resource" not in selected.validation_errors[0]


def test_verbatim_coverage_can_bind_a_slotless_followup_segment(task_catalog):
    utterance = "Review Casey; present the result."
    response = {
        "invocations": [
            {
                **_invocation("review-subject", "subject", "Casey"),
                "coverage": ["present the result."],
            }
        ]
    }

    selected = select_skills(utterance, task_catalog, _dummy_lm(response))

    assert len(selected) == 1
    assert selected.coverage_segments == ((0, 12), (14, 33))


def test_repeated_list_slot_bindings_preserve_source_order_and_spans():
    batch = _skill(
        "review-batch",
        "task",
        slot_name="subjects",
        is_list=True,
    )
    catalog = SkillCatalog(
        {batch.name: batch},
        fingerprint="sha256:list",
        mode="enforce",
    )
    response = {
        "invocations": [
            {
                "skill_name": "review-batch",
                "bindings": [
                    {
                        "slot_name": "subjects",
                        "value": "Casey",
                        "source_text": "Casey",
                    },
                    {
                        "slot_name": "subjects",
                        "value": "Riley",
                        "source_text": "Riley",
                    },
                ],
            }
        ]
    }

    selected = select_skills(
        "Review Casey and Riley.",
        catalog,
        _dummy_lm(response),
    )

    assert selected[0].slots == {"subjects": ["Casey", "Riley"]}
    assert selected[0].provenance["subjects"].source_spans == (
        SourceSpan(start=7, end=12, text="Casey"),
        SourceSpan(start=17, end=22, text="Riley"),
    )


def test_off_mode_does_not_load_cards_or_construct_selector(
    monkeypatch,
):
    monkeypatch.setenv("FW_PLAN_DECOMPOSITION", "off")
    monkeypatch.setattr(
        workflow_agent,
        "load_skill_catalog",
        lambda *_args, **_kwargs: pytest.fail("off mode loaded task cards"),
    )
    monkeypatch.setattr(
        workflow_agent.dspy,
        "Predict",
        lambda *_args, **_kwargs: pytest.fail("off mode constructed selector"),
    )

    mode, catalog = _plan_decomposition_point(
        SimpleNamespace(app_workflow=SimpleNamespace(folderpath="/generic/workflow"))
    )

    assert mode is PlanMode.OFF
    assert len(catalog) == 0
