"""Shared dispatch choke point for execution-record correlation (arch §12.1).

Eligibility, authorization, and strict-write gates belong to later slices.
Phase 0 routes every migrated path through these record helpers so
``TurnResult.execution_records`` is populated from the same ledger the spans
join on via ``command_call_id``.
"""

from __future__ import annotations

from typing import Optional

from fastworkflow.execution_recorder import ExecutionRecorder, record_execution


class CommandDispatcher:
    """Phase 0 dispatcher surface: record-only, no control-flow reads."""

    @staticmethod
    def record_completed_dispatch(
        recorder: Optional[ExecutionRecorder],
        *,
        command_call_id: str,
        parent_call_id: Optional[str],
        span_id: Optional[str],
        child_calls: Optional[list] = None,
    ) -> None:
        record_execution(
            recorder,
            command_call_id=command_call_id,
            parent_call_id=parent_call_id,
            span_id=span_id,
            child_calls=child_calls,
        )
