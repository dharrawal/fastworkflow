"""Worker supervision and request envelopes (arch §13.3, FW-REQ-008).

The failure this closes: `ChatWorker.run` calls `_run_workflow_loop` inside a
bare `try/finally`, so an unhandled exception from any turn propagates out of
the `while` loop, kills the only worker thread, and every caller blocked on
`command_output_queue.get()` waits forever. FW-REQ-008 clauses 3, 4 and 5 are
all about that one shape: a turn exception must produce a terminal failed turn
rather than terminating the worker; worker health must be observable without
reading thread stacks; and a queued caller must fail promptly when the worker is
dead.

Two things live here:

* ``WorkerHealth`` — the observable state, heartbeat, and last classified
  failure, with the poisoning rule architecture §13.3 requires: a stuck worker
  and its ownership stay **visibly** poisoned, and a timed-out ``join()`` clears
  nothing and claims no termination.
* ``TurnRequest`` — a request envelope with completion/failure delivery, so a
  submission that the worker never gets to is failed rather than lost. The raw
  queue contract still works: ``unwrap_request`` accepts either.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from fastworkflow.typed_failure import (
    CODE_WORKER_DEAD,
    CODE_WORKER_FAILED,
    CODE_WORKER_STUCK,
    TypedFailure,
)


class WorkerState(str, enum.Enum):
    """Where the worker is, as a value rather than as a thread stack."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    # The outer loop terminated on a failure. The worker is gone; submissions
    # are rejected promptly rather than queued for nobody.
    FAILED = "failed"
    # Asked to stop and did not. NOT the same as stopped: the thread may still
    # be running and still holding the workflow, so ownership is retained and
    # the state says so (arch §13.3).
    STUCK = "stuck"


_DEAD_STATES = frozenset({WorkerState.FAILED, WorkerState.STUCK, WorkerState.STOPPED})
_POISONED_STATES = frozenset({WorkerState.FAILED, WorkerState.STUCK})


class WorkerDeadError(RuntimeError):
    """A submission reached a worker that cannot run it.

    Raised at submission time so the caller fails immediately instead of
    blocking on a reply that will never come — FW-REQ-008 clause 5, and the
    reason its acceptance criterion says "no test waits for a generic timeout
    after worker death".
    """

    def __init__(self, health: "WorkerHealth"):
        self.health = health
        self.failure = health.rejection_failure()
        super().__init__(self.failure.as_observation())


@dataclass
class WorkerHealth:
    """Observable worker state. Every field is read under the lock."""

    state: WorkerState = WorkerState.STOPPED
    # Monotonic, because this is used for liveness arithmetic and a wall clock
    # that steps backwards would make a live worker look stalled.
    heartbeat_monotonic: float = field(default_factory=time.monotonic)
    heartbeat_wall: float = field(default_factory=time.time)
    last_failure: Optional[TypedFailure] = None
    # The turn that was in flight when the worker failed, kept so a reader can
    # attribute the failure to a turn rather than to the session as a whole.
    failed_turn_key: Optional[str] = None
    turns_failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def beat(self, state: Optional[WorkerState] = None) -> None:
        """Record liveness, optionally moving state.

        Refuses to un-poison: once FAILED or STUCK, a heartbeat is evidence
        that something is still running, not permission to call it healthy.
        A poisoned worker is cleared by replacing it, never by beating.
        """
        with self._lock:
            now_monotonic, now_wall = time.monotonic(), time.time()
            self.heartbeat_monotonic = now_monotonic
            self.heartbeat_wall = now_wall
            if state is not None and self.state not in _POISONED_STATES:
                self.state = state

    def set_state(self, state: WorkerState) -> None:
        with self._lock:
            if self.state in _POISONED_STATES and state not in _POISONED_STATES:
                # Recording the attempt rather than making it: `stop_workflow`
                # calls this on its way out, and a poisoned worker that reported
                # itself STOPPED is precisely the claim §13.3 forbids.
                return
            self.state = state

    def record_turn_failure(
        self, failure: TypedFailure, turn_key: Optional[str] = None
    ) -> None:
        """A turn failed and the worker survived it."""
        with self._lock:
            self.last_failure = failure
            self.failed_turn_key = turn_key
            self.turns_failed += 1

    def record_worker_failure(
        self, failure: TypedFailure, turn_key: Optional[str] = None
    ) -> None:
        """The outer loop terminated. The worker is gone."""
        with self._lock:
            self.last_failure = failure
            self.failed_turn_key = turn_key
            self.state = WorkerState.FAILED

    def mark_stuck(self, detail: str) -> None:
        """Asked to stop, did not. Ownership is retained deliberately."""
        with self._lock:
            self.state = WorkerState.STUCK
            self.last_failure = TypedFailure(
                # Not `permanent`: the thread may still be inside a call that
                # returns. What is certain is that we do not know, and
                # `outcome-unknown` is the disposition that says so without
                # inviting anything to be retried on top of it.
                disposition="outcome-unknown",
                code=CODE_WORKER_STUCK,
                detail=detail,
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        with self._lock:
            return self.state not in _DEAD_STATES

    @property
    def is_poisoned(self) -> bool:
        """FAILED or STUCK: visible, and not cleared by a stop that timed out."""
        with self._lock:
            return self.state in _POISONED_STATES

    def seconds_since_heartbeat(self) -> float:
        with self._lock:
            return time.monotonic() - self.heartbeat_monotonic

    def rejection_failure(self) -> TypedFailure:
        """The classified reason a submission is being refused."""
        with self._lock:
            state, last = self.state, self.last_failure
        if state == WorkerState.STUCK:
            return last or TypedFailure(
                disposition="outcome-unknown",
                code=CODE_WORKER_STUCK,
                detail="worker is stuck; ownership retained",
            )
        return TypedFailure(
            disposition="permanent",
            code=CODE_WORKER_DEAD if state == WorkerState.STOPPED else CODE_WORKER_FAILED,
            detail=(
                f"worker is {state.value}"
                + (f": {last.detail}" if last is not None and last.detail else "")
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        """A JSON-safe view — the "without inspecting thread stacks" of clause 4."""
        with self._lock:
            return {
                "state": self.state.value,
                "heartbeat_age_seconds": round(
                    time.monotonic() - self.heartbeat_monotonic, 3
                ),
                "heartbeat_at": self.heartbeat_wall,
                "turns_failed": self.turns_failed,
                "failed_turn_key": self.failed_turn_key,
                "last_failure": (
                    self.last_failure.to_state() if self.last_failure else None
                ),
            }


@dataclass
class TurnRequest:
    """One submission, with completion and failure delivery.

    The raw-queue contract put a bare message on the queue and had no way to
    tell the submitter that it was never going to be processed. An envelope
    carries its own completion, so worker death fails every queued request
    instead of leaving them to time out one by one.
    """

    payload: Any
    request_id: str = ""
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _result: Any = field(default=None, repr=False)
    _failure: Optional[TypedFailure] = field(default=None, repr=False)

    def complete(self, result: Any) -> None:
        self._result = result
        self._done.set()

    def fail(self, failure: TypedFailure) -> None:
        self._failure = failure
        self._done.set()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    @property
    def failure(self) -> Optional[TypedFailure]:
        return self._failure

    def wait(self, timeout: Optional[float] = None) -> Any:
        """Block for the result. Raises ``TurnFailedError`` on a failed request.

        A timeout here is the caller's own patience running out, not evidence
        about the turn, so it raises ``TimeoutError`` and leaves the envelope
        undelivered rather than marking it failed.
        """
        from fastworkflow.typed_failure import TurnFailedError

        if not self._done.wait(timeout):
            raise TimeoutError(f"turn request {self.request_id!r} not completed")
        if self._failure is not None:
            raise TurnFailedError(self._failure)
        return self._result


def unwrap_request(item: Any) -> tuple[Any, Optional[TurnRequest]]:
    """Normalize a queue item to ``(payload, envelope_or_None)``.

    The legacy adapter architecture §13.3 keeps: a raw message on the queue is
    still a valid submission, it just has nowhere to deliver a failure. Every
    consumer goes through this so neither shape has to be handled twice.
    """
    if isinstance(item, TurnRequest):
        return item.payload, item
    return item, None
