"""EXP-013: every external call is bounded, and the bound crosses the thread.

FW-REQ-008 clause 1. Deadlines were per-call-site defaults chosen by whoever
wrote the call site — `timeout=60` on one client, `timeout=30` on a workflow, a
900-second poll under both — none of them related to the turn they belonged to.
A turn could exceed any bound its caller believed it had by composing calls that
were each individually inside one.

Two properties carry the weight here, and both are easy to implement wrongly in
a way that looks right:

* the deadline is **absolute**, so a nested call cannot restart it;
* it is **explicitly copied** into executor threads, because a ContextVar does
  not cross a thread boundary and every blocking backend call runs on the far
  side of one.
"""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from fastworkflow.external_operations import (
    DEFAULT_DEADLINES,
    ExternalOperationContext,
    clamp_timeout,
    copy_into_thread,
    current_operation,
    operation,
    remaining_seconds,
    require_time,
    retry_owner,
)
from fastworkflow.typed_failure import CODE_BACKEND_TIMEOUT, TurnFailedError


@pytest.fixture
def todo_workflow_path() -> str:
    from pathlib import Path

    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    import fastworkflow
    from fastworkflow.command_routing import RoutingRegistry

    fastworkflow.init({})
    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


# ----------------------------------------------------------------------
# Per-class deadline enforcement
# ----------------------------------------------------------------------


def test_every_deadline_class_has_a_default_and_a_retry_owner():
    """Arch §13.2/§13.2.1: a class with no owner for either is not a class."""
    for kind in DEFAULT_DEADLINES:
        assert DEFAULT_DEADLINES[kind] > 0
        assert retry_owner(kind) is not None


def test_a_write_may_only_be_retried_by_the_operation_contract():
    """FW-REQ-008B clause 1, expressed in the retry matrix.

    Not the phase caller and not a read policy: after a write is dispatched,
    nothing may retry it unless the contract guarantees deduplication under the
    same operation ID or reconciliation proves it safe.
    """
    assert retry_owner("backend.write") == "operation_contract"
    assert retry_owner("backend.read") == "read_policy"
    # Dispatch authority is consumed at the call, so a policy decision is not
    # re-askable.
    assert retry_owner("policy") == "none"
    assert retry_owner("something_nobody_declared") == "none"


def test_the_clamp_only_ever_lowers():
    with operation("backend.read", seconds=5):
        assert clamp_timeout(60) <= 5.0
        assert clamp_timeout(2) == 2.0


def test_outside_an_operation_the_callers_own_timeout_is_returned():
    """What makes the generated clients correct in and out of a bounded turn."""
    assert current_operation() is None
    assert clamp_timeout(60) == 60
    assert remaining_seconds() is None
    assert remaining_seconds(30) == 30


def test_a_nested_operation_cannot_outlive_its_parent():
    """The composition defect, closed.

    `backend.polling` defaults to 900 seconds. Nested inside a five-second
    parent it gets five, because a sub-call that outlives the call containing it
    is exactly how a bounded operation becomes an unbounded one.
    """
    with operation("backend.read", seconds=5):
        with operation("backend.polling") as inner:
            assert inner.remaining() <= 5.0
            assert inner.kind == "backend.polling"


def test_a_nested_operation_may_shorten_but_not_lengthen():
    with operation("backend.read", seconds=10):
        with operation("backend.read", seconds=2) as inner:
            assert inner.remaining() <= 2.0
        with operation("backend.read", seconds=1000) as inner:
            assert inner.remaining() <= 10.0


def test_the_context_is_restored_when_the_block_exits():
    with operation("backend.read", seconds=5):
        assert current_operation() is not None
    assert current_operation() is None


def test_correlation_is_inherited_by_nested_operations():
    with operation("backend.read", seconds=5, turn_key="t1", security_scope="tenant-a"):
        with operation("backend.polling") as inner:
            assert inner.turn_key == "t1"
            assert inner.security_scope == "tenant-a"


# ----------------------------------------------------------------------
# What a timeout MEANS is different for reads and writes
# ----------------------------------------------------------------------


def test_a_read_timeout_is_transient_and_a_write_timeout_is_unknown():
    """The whole of FW-REQ-008B in one distinction.

    A read that timed out left nothing behind. A write that timed out may have
    landed, and calling that a failure is the conversion nothing is allowed to
    make without evidence.
    """
    read = ExternalOperationContext(
        kind="backend.read", deadline_monotonic=time.monotonic() - 1
    )
    write = ExternalOperationContext(
        kind="backend.write", deadline_monotonic=time.monotonic() - 1
    )
    assert read.failure().disposition == "transient"
    assert write.failure().disposition == "outcome-unknown"
    assert write.failure().code == CODE_BACKEND_TIMEOUT
    # Reconciliation and compensation are the same shape as a write: they touch
    # the world.
    for kind in ("reconciliation", "compensation"):
        context = ExternalOperationContext(
            kind=kind, deadline_monotonic=time.monotonic() - 1
        )
        assert context.failure().disposition == "outcome-unknown"


def test_an_expired_deadline_refuses_before_dispatch():
    """Checked before the call, not only after it.

    Starting a backend call with no time left produces an outcome nobody can
    use and, for a write, an outcome nobody can classify.
    """
    with operation("backend.read", seconds=0.0001):
        time.sleep(0.01)
        with pytest.raises(TurnFailedError) as caught:
            require_time("about to read")
        assert caught.value.failure.code == CODE_BACKEND_TIMEOUT


def test_require_time_is_a_no_op_outside_an_operation():
    require_time("nothing in force")


# ----------------------------------------------------------------------
# The thread copy — where this would silently stop applying
# ----------------------------------------------------------------------


def test_the_deadline_does_not_cross_a_thread_on_its_own():
    """Stated as a test because it is the reason `copy_into_thread` exists.

    If this ever starts passing without the wrapper, the wrapper is dead code
    and somebody should find out from a failing test rather than by reading.
    """
    with operation("backend.read", seconds=5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            seen = pool.submit(current_operation).result()
    assert seen is None


def test_copy_into_thread_carries_the_deadline_into_the_worker():
    with operation("backend.read", seconds=5) as outer:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            seen = pool.submit(copy_into_thread(current_operation)).result()
            clamped = pool.submit(copy_into_thread(clamp_timeout), 60).result()
    assert seen is not None
    assert seen.deadline_monotonic == outer.deadline_monotonic
    assert clamped <= 5.0


def test_the_wrapper_preserves_the_callable_it_wraps():
    def work(a, b=2):
        """A docstring worth keeping."""
        return a + b

    wrapped = copy_into_thread(work)
    assert wrapped(1, b=3) == 4
    assert wrapped.__doc__ == "A docstring worth keeping."


# ----------------------------------------------------------------------
# Provider policy (arch §13.2.1)
# ----------------------------------------------------------------------


def test_model_calls_get_a_role_timeout_and_a_pinned_retry_count():
    from fastworkflow.utils.dspy_utils import _apply_role_policy

    kwargs: dict = {}
    _apply_role_policy(kwargs, "LLM_AGENT")
    assert kwargs["timeout"] > 0
    # Pinned rather than left to the provider default, so a library upgrade
    # cannot silently change how many times a turn's model call is retried.
    assert kwargs["num_retries"] == 3


def test_a_role_timeout_is_clamped_to_the_remaining_deadline():
    from fastworkflow.utils.dspy_utils import _apply_role_policy

    with operation("model.agent", seconds=7):
        kwargs: dict = {}
        _apply_role_policy(kwargs, "LLM_AGENT")
        assert kwargs["timeout"] <= 7.0


def test_an_explicit_caller_timeout_still_wins():
    from fastworkflow.utils.dspy_utils import _apply_role_policy

    kwargs = {"timeout": 5}
    _apply_role_policy(kwargs, "LLM_AGENT")
    assert kwargs["timeout"] == 5


def test_an_exhausted_deadline_never_asks_for_a_zero_second_call():
    """Most clients read `timeout=0` as "no timeout" — the opposite of the intent."""
    from fastworkflow.utils.dspy_utils import _apply_role_policy

    with operation("model.agent", seconds=0.0001):
        time.sleep(0.01)
        kwargs: dict = {}
        _apply_role_policy(kwargs, "LLM_AGENT")
        assert kwargs["timeout"] >= 1.0


# ----------------------------------------------------------------------
# Distillation isolation (arch §13.5)
# ----------------------------------------------------------------------


def _register_workflow_manifest(workflow_path, commands):
    """Register metadata built the way startup builds it.

    Through `merge_and_gate`, not by constructing `RuntimeMetadata` directly:
    the merge is what folds in the CORE manifest, which declares `go_up`,
    `what_can_i_do` and the rest of the framework's own commands `read_only`.
    Built by hand, those arrive as `unknown` — and since `unknown` is never
    `read_only` (arch §7.3), the guard would refuse every workflow on the
    strength of the framework's own navigation commands.
    """
    from fastworkflow.runtime_manifest import (
        RuntimeManifest,
        merge_and_gate,
        register_runtime_metadata,
    )

    manifest = RuntimeManifest(
        schema_version=1, manifest_version="1.0.0", commands=commands
    )
    register_runtime_metadata(
        workflow_path, merge_and_gate(manifest, deployment_features={})
    )


def test_distillation_refuses_a_surface_that_is_not_proven_read_only(
    todo_workflow_path, initialized_fastworkflow
):
    """Arch §13.5 clause 1.

    Distillation runs the same turn twice against the same backend. If any
    enabled command can write, the second pass applies a second effect and the
    comparison is between one world and a different one — and restoring local
    workflow state afterwards does not reverse that (clause 3), it only makes
    the tree look as if it had.
    """
    from types import SimpleNamespace

    from fastworkflow.command_context_model import CommandContextModel
    from fastworkflow.distillation import DistillationRefused, _refuse_unless_read_only
    from fastworkflow.runtime_manifest import (
        CommandDeclaration,
        EffectContract,
        clear_runtime_metadata,
    )

    session = SimpleNamespace(
        get_active_workflow=lambda: SimpleNamespace(folderpath=todo_workflow_path)
    )
    model = CommandContextModel.load(todo_workflow_path)
    surface = {
        capability.definition.definition_id
        for context_name in model.occupiable_contexts()
        for capability in model.effective_capabilities(context_name)
    }
    assert surface, "the fixture workflow must have a surface to judge"

    read_only = {
        definition_id: CommandDeclaration(effect=EffectContract(kind="read_only"))
        for definition_id in surface
    }

    try:
        # Everything declared read-only: distillation is a comparison, so it runs.
        _register_workflow_manifest(todo_workflow_path, read_only)
        _refuse_unless_read_only(session)

        # One writer is enough to make the second pass a second effect.
        writer = sorted(surface)[0]
        with_write = dict(
            read_only, **{writer: CommandDeclaration(effect=EffectContract(kind="write"))}
        )
        _register_workflow_manifest(todo_workflow_path, with_write)
        with pytest.raises(DistillationRefused) as caught:
            _refuse_unless_read_only(session)
        assert writer in str(caught.value)

        # ...unless the backend behind it is declared disposable.
        import fastworkflow

        fastworkflow._env_vars["FW_DISTILLATION_DISPOSABLE_BACKEND"] = "1"
        try:
            _refuse_unless_read_only(session)
        finally:
            fastworkflow._env_vars.pop("FW_DISTILLATION_DISPOSABLE_BACKEND", None)
    finally:
        clear_runtime_metadata()


def test_an_undeclared_command_is_treated_as_write_capable(
    todo_workflow_path, initialized_fastworkflow
):
    """Arch §7.3: absent is `unknown`, and `unknown` is never `read_only`.

    The conservative direction, and the reason a manifest that declares only
    some of its commands does not get a pass on the rest.
    """
    from types import SimpleNamespace

    from fastworkflow.distillation import DistillationRefused, _refuse_unless_read_only
    from fastworkflow.runtime_manifest import clear_runtime_metadata

    session = SimpleNamespace(
        get_active_workflow=lambda: SimpleNamespace(folderpath=todo_workflow_path)
    )
    try:
        _register_workflow_manifest(todo_workflow_path, {})
        with pytest.raises(DistillationRefused):
            _refuse_unless_read_only(session)
    finally:
        clear_runtime_metadata()


def test_a_workflow_with_no_manifest_warns_rather_than_refusing(
    todo_workflow_path, initialized_fastworkflow
):
    """The compatibility edge, recorded rather than enforced.

    Nothing has declared anything, so the guard cannot verify the surface —
    but refusing here would break every workflow that distills today and has no
    manifest, which is a compatibility break this slice does not own (arch §7.1).
    """
    from types import SimpleNamespace

    from fastworkflow.distillation import _refuse_unless_read_only
    from fastworkflow.runtime_manifest import clear_runtime_metadata

    clear_runtime_metadata()
    session = SimpleNamespace(
        get_active_workflow=lambda: SimpleNamespace(folderpath=todo_workflow_path)
    )
    _refuse_unless_read_only(session)


def test_the_disposable_backend_declaration_is_explicit(monkeypatch):
    """The safe answer cannot be inferred, so it has to be declared.

    A workflow cannot tell a disposable backend from a production one by
    looking at it.
    """
    import fastworkflow
    from fastworkflow.distillation import DISPOSABLE_BACKEND_ENV_VAR

    assert DISPOSABLE_BACKEND_ENV_VAR == "FW_DISTILLATION_DISPOSABLE_BACKEND"


def test_insight_extraction_has_its_own_deadline_class():
    """Arch §13.5 clause 4: it cannot kill the shared worker.

    Extraction runs after the turn's real work is finished, on the shared
    worker, so its bound is its own rather than borrowed from the turn.
    """
    assert "distillation.insight_extraction" in DEFAULT_DEADLINES
    assert retry_owner("distillation.insight_extraction") == "distillation_phase"
