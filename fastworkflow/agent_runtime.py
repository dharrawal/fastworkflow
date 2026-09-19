"""Composition and lifecycle owner for one workflow tool agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from fastworkflow.observation_offloading.archive import RuntimeHandleScope


def build_turn_runtime(
    scope: RuntimeHandleScope, *, archive: Any = None, result_store: Any = None
) -> "TurnRuntime":
    """Create the one valid runtime owner for a standalone agent."""
    from fastworkflow import auto_navigation
    from fastworkflow.observation_offloading import state
    from fastworkflow.result_handles import paging

    selected_archive = archive if archive is not None else state.archive()
    path = getattr(selected_archive, "db_path", "")
    factory = (lambda: paging.store_for_path(path)) if result_store is None and path else None
    return TurnRuntime(
        scope=scope, archive=selected_archive, result_store=result_store,
        observation_component=state, result_component=paging,
        navigation_component=auto_navigation, record_event=state.record_event,
        result_store_factory=factory,
    )


@dataclass
class TurnRuntime:
    """Bind one agent's scoped stores and coordinate finished-scope cleanup.

    Suspension remains owned by the agent and WEC guards.  This object is called
    only after those guards decide a scope is finished, so it does not maintain
    a second lifecycle state or suppress retries after a failed seal.
    """

    scope: RuntimeHandleScope
    archive: Any
    result_store: Any
    observation_component: Any
    result_component: Any
    navigation_component: Any
    record_event: Callable[[dict[str, Any]], None]
    result_store_factory: Optional[Callable[[], Any]] = None

    def bind_scope(self, scope: RuntimeHandleScope) -> None:
        self.scope = scope

    def emit(self, event: dict[str, Any]) -> None:
        self.record_event(event)

    def get_result_store(self) -> Any:
        if self.result_store is None and self.result_store_factory is not None:
            self.result_store = self.result_store_factory()
        return self.result_store

    def finish_scope(self, scope: RuntimeHandleScope) -> None:
        """Seal then release a scope already proven finished by its caller."""
        self.observation_component.seal_scope(
            scope, selected_archive=self.archive
        )
        self.release_scope(scope)

    def release_scope(self, scope: RuntimeHandleScope | str) -> None:
        """Ask each component to release only the resources it owns."""
        scope_id = str(getattr(scope, "scope_id", scope))
        self.observation_component.release_scope(scope_id)
        self.result_component.release_scope(scope_id)
        self.navigation_component.forget_scope(scope_id)


def reclaim_scope(scope: RuntimeHandleScope | str) -> None:
    """Released aggregate cleanup for callers without an agent runtime."""
    from fastworkflow import auto_navigation
    from fastworkflow.observation_offloading import state
    from fastworkflow.result_handles import paging

    scope_id = str(getattr(scope, "scope_id", scope))
    state.release_scope(scope_id)
    paging.release_scope(scope_id)
    auto_navigation.forget_scope(scope_id)


def reset_runtime_state() -> None:
    """Released aggregate reset for tests and standalone integrations."""
    from fastworkflow import auto_navigation
    from fastworkflow.observation_offloading import state
    from fastworkflow.result_handles import paging

    state.reset_observation_state()
    paging.reset_result_handle_state()
    auto_navigation.reset_auto_navigation_state()
