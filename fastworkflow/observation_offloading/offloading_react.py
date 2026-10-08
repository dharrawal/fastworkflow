"""ReAct with observation offloading: per-turn scope and archive."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Mapping, Optional

import dspy

from fastworkflow.observation_offloading.archive import RuntimeHandleScope
from fastworkflow.utils.dspy_logger import DSPyForward
from fastworkflow.utils.react import fastWorkflowReAct

DEFAULT_MAX_ITERS = 25


class OffloadingReAct(fastWorkflowReAct):
    """Tool ReAct with per-turn handle scope and observation offloading hooks.

    ``scope_factory`` is called once per ``forward`` so the handle scope (and
    with it the ``O{n}`` alias namespace) belongs to the turn being run, not to
    the turn the agent happened to be constructed in. The bound scope travels
    with the suspended state so an ask_user resume, in this process or another,
    keeps writing and reading the same handles.
    """

    def __init__(
        self,
        *args: Any,
        scope_factory: Optional[Callable[[], RuntimeHandleScope]] = None,
        turn_runtime: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.continuation_scope: RuntimeHandleScope | None = None
        self.continuation_scope_id: str | None = None
        self._scope_factory = scope_factory
        self.turn_runtime = turn_runtime

    def bind_scope(self) -> RuntimeHandleScope | None:
        """Resolve the scope for the turn that is starting; reclaim the one it replaces.

        Binding a different scope is this agent saying the previous turn is
        over, which makes it the earliest honest moment to release that turn's
        process-local state: its hot payloads, its archive registry and its
        context clauses. Everything released is still on disk; only residency
        goes.

        A suspension is the one thing that is not over, so a still-suspended
        agent reclaims nothing. ``forward`` clears the suspension before it gets
        here, so the guard is for a caller that binds a scope by hand.

        That release includes the raw in-flight copies of the previous turn's
        redacted evidence, because "the agent has bound the next turn" is the
        strongest statement this process can make that the previous one is
        finished -- its summary is recorded, its answer is delivered, and no
        read of it can still be part of it. A suspension keeps them, by the one
        condition below.
        """
        import logging

        logger = logging.getLogger(__name__)
        factory = getattr(self, "_scope_factory", None)
        if factory is None:
            return getattr(self, "continuation_scope", None)
        previous = getattr(self, "continuation_scope", None)
        scope = factory()
        runtime = getattr(self, "turn_runtime", None)
        if runtime is None:
            from fastworkflow.agent_runtime import build_turn_runtime

            runtime = build_turn_runtime(
                previous or scope,
                archive=getattr(self, "observation_archive", None),
            )
            self.turn_runtime = runtime
        if (
            previous is not None
            and previous != scope
            and getattr(self, "_suspended", None) is None
        ):
            try:
                runtime.finish_scope(previous)
            except Exception as exc:  # noqa: BLE001
                # Same shape as the sibling in
                # WorkflowExecutionContext._reclaim_offloading_scope: reclaiming
                # the PREVIOUS turn's scope must not abort the turn that is
                # starting.
                logger.debug(
                    "bind_scope: could not reclaim the previous "
                    f"offloading scope ({type(exc).__name__}: {exc})"
                )
        self.continuation_scope = scope
        self.continuation_scope_id = scope.scope_id
        runtime.bind_scope(scope)
        return scope

    def export_suspended(self) -> dict[str, Any] | None:
        data = super().export_suspended()
        if data is not None:
            scope = getattr(self, "continuation_scope", None)
            if scope is not None:
                data["continuation_scope"] = asdict(scope)
        return data

    def import_suspended(self, data: dict[str, Any]) -> None:
        super().import_suspended(data)
        raw_scope = data.get("continuation_scope")
        if isinstance(raw_scope, Mapping):
            scope = RuntimeHandleScope(**raw_scope)
            self.continuation_scope = scope
            self.continuation_scope_id = scope.scope_id
            runtime = getattr(self, "turn_runtime", None)
            if runtime is None:
                from fastworkflow.agent_runtime import build_turn_runtime

                runtime = build_turn_runtime(
                    scope, archive=getattr(self, "observation_archive", None)
                )
                self.turn_runtime = runtime
            else:
                runtime.bind_scope(scope)

    @DSPyForward.intercept
    def forward(self, **input_args: Any) -> dspy.Prediction:
        self.clear_suspension()
        self.current_trajectory = {}
        self.iteration_counter = 0
        self.bind_scope()
        return super().forward(**input_args)
