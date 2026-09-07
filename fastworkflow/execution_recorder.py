"""Turn-scoped execution ledger (arch §12.1, §12.2).

Phase 0 records correlation refs only: ``command_call_id``, parent link,
ordinal, and optional ``span_id``. The span attributes carry the full
``ExecutionRecord`` field set; this list is the durable skeleton on
``TurnResult`` for executions that may never produce a ``CommandOutput``.
"""

from __future__ import annotations

from typing import Optional

from fastworkflow.turn import ExecutionRecordRef


class ExecutionRecorder:
    """Collect ``ExecutionRecordRef`` rows for one logical turn."""

    def __init__(self) -> None:
        self._records: list[ExecutionRecordRef] = []
        self._seen: set[str] = set()

    def begin(
        self,
        *,
        command_call_id: str,
        parent_call_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> None:
        """Record one dispatch. Idempotent per ``command_call_id``."""
        if command_call_id in self._seen:
            return
        self._seen.add(command_call_id)
        self._records.append(
            ExecutionRecordRef(
                command_call_id=command_call_id,
                parent_call_id=parent_call_id,
                command_ordinal=len(self._records),
                span_id=span_id,
            )
        )

    def complete(self, *, command_call_id: str) -> None:
        """Reserved for a future status/failure field; Phase 0 is begin-only."""

    def record_child_calls(self, child_calls: Optional[list]) -> None:
        """File inner CME/core hops that have no span of their own."""
        if not child_calls:
            return
        for entry in child_calls:
            if not isinstance(entry, dict):
                continue
            call_id = entry.get("call_id")
            if not call_id:
                continue
            self.begin(
                command_call_id=call_id,
                parent_call_id=entry.get("parent_call_id"),
                span_id=None,
            )

    def records(self) -> tuple[ExecutionRecordRef, ...]:
        return tuple(self._records)


def recorder_for(host) -> Optional[ExecutionRecorder]:
    """The turn's recorder on a trace host, or None when absent."""
    return getattr(host, "_execution_recorder", None)


def record_execution(
    recorder: Optional[ExecutionRecorder],
    *,
    command_call_id: str,
    parent_call_id: Optional[str],
    span_id: Optional[str],
    child_calls: Optional[list] = None,
) -> None:
    """Append one completed dispatch and any nested child calls."""
    if recorder is None:
        return
    recorder.begin(
        command_call_id=command_call_id,
        parent_call_id=parent_call_id,
        span_id=span_id,
    )
    recorder.record_child_calls(child_calls)
    recorder.complete(command_call_id=command_call_id)
