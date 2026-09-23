"""Composition and lifecycle owner for one workflow tool agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from fastworkflow.observation_offloading.archive import RuntimeHandleScope


def build_turn_runtime(
    scope: RuntimeHandleScope, *, archive: Any = None
) -> "TurnRuntime":
    """Create the one valid runtime owner for a standalone agent."""
    from fastworkflow.observation_offloading import state

    selected_archive = archive if archive is not None else state.archive()
    return TurnRuntime(
        scope=scope, archive=selected_archive,
        observation_component=state, record_event=state.record_event,
    )


@dataclass
class TurnRuntime:
    """Bind one agent's scoped store and coordinate finished-scope cleanup.

    Suspension remains owned by the agent and WEC guards.  This object is called
    only after those guards decide a scope is finished, so it does not maintain
    a second lifecycle state or suppress retries after a failed seal.
    """

    scope: RuntimeHandleScope
    archive: Any
    observation_component: Any
    record_event: Callable[[dict[str, Any]], None]

    def bind_scope(self, scope: RuntimeHandleScope) -> None:
        self.scope = scope

    def emit(self, event: dict[str, Any]) -> None:
        self.record_event(event)

    def finish_scope(self, scope: RuntimeHandleScope) -> None:
        """Seal then release a scope already proven finished by its caller."""
        self.observation_component.seal_scope(
            scope, selected_archive=self.archive
        )
        self.release_scope(scope)

    def release_scope(self, scope: RuntimeHandleScope | str) -> None:
        """Ask the observation component to release the resources it owns."""
        scope_id = str(getattr(scope, "scope_id", scope))
        self.observation_component.release_scope(scope_id)


def reclaim_scope(scope: RuntimeHandleScope | str) -> None:
    """Released aggregate cleanup for callers without an agent runtime."""
    from fastworkflow.observation_offloading import state

    state.release_scope(str(getattr(scope, "scope_id", scope)))


def reset_runtime_state() -> None:
    """Released aggregate reset for tests and standalone integrations."""
    from fastworkflow.observation_offloading import state

    state.reset_observation_state()
