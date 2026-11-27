"""The CLI transport's three queues, and what they add to a plain ``queue.Queue``.

``fastworkflow run_cli`` drives the engine through queues: a worker thread runs a turn, and the terminal reads the
trace and output queues until the turn's sentinel (``None``). The engine cannot abort a turn in flight (a model
call or a command's own code runs to its end), so "stop waiting" is built from two things here:

* **Discard.** :meth:`TurnGate.abandon` marks the running turn abandoned. While any abandoned turn is unfinished,
  the trace and output queues drop what the worker puts, so nothing of it reaches a later turn's prompt.
  The worker loop calls :meth:`TurnGate.turn_finished` after each turn; the last abandoned turn finishing lifts the
  discard. The turn itself still runs to its end (its commands' side effects happen); only its output is dropped.
* **No question to nobody.** An abandoned turn that asks ``ask_user`` would block for an answer nobody knows is
  wanted. :class:`MessageQueue.get` (the blocking read ``ask_user`` makes) raises the engine's own
  ``CommandCancelledError`` instead, which ``WorkflowExecutionContext._execute_message`` already turns into a
  failed "Command cancelled" output and a cleared suspension. The same error ends a question the user cancels
  (``//cancel``): :meth:`MessageQueue.post_cancel`.

With no interrupt and no cancel, none of this is on the path beyond one uncontended lock per queue put and one flag
test per ``ask_user`` read.
"""
from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Any

#: Posted to the user message queue to end the question a blocked ``ask_user`` is waiting on.
CANCEL_QUESTION = object()


class TurnGate:
    """The abandon counter shared by the CLI transport's queues."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._discarding = 0
        #: Sentinels (``None``) put on the trace queue since ``begin_turn``, dropped ones included.
        self.sentinels = 0
        #: True from the worker taking a message to the turn's end (an ask_user wait is inside a turn).
        self.in_turn = False
        #: A question an abandoned turn left pending was cancelled unseen (read once, for the notice).
        self.cancelled_unseen = False

    def begin_turn(self) -> None:
        self.sentinels = 0
        self.in_turn = True

    @property
    def muted(self) -> bool:
        """True while an abandoned turn has not finished: the queues drop what the worker puts."""
        return self._discarding > 0

    def abandon(self, trace_queue: Queue, output_queue: Queue) -> bool:
        """Abandon the turn the terminal is waiting on. False when it has already ended (its sentinel is queued):
        nothing is abandoned and the caller reads the result as usual. True: what the turn put so far is dropped
        and what it puts later is dropped until it finishes."""
        with self.lock:
            if not self.in_turn:
                return False  # the worker is between turns: nothing is running to abandon
            with trace_queue.mutex:
                if any(item is None for item in trace_queue.queue):
                    return False
                trace_queue.queue.clear()
            with output_queue.mutex:
                output_queue.queue.clear()
            self._discarding += 1
            return True

    def turn_finished(self) -> None:
        """The worker finished a turn: one fewer abandoned turn to drop."""
        with self.lock:
            self.in_turn = False
            if self._discarding:
                self._discarding -= 1


class GatedQueue(Queue):
    """A queue that drops puts while its gate is muted."""

    def __init__(self, gate: TurnGate) -> None:
        super().__init__()
        self._gate = gate

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        with self._gate.lock:
            if item is None:
                self._gate.sentinels += 1
            if self._gate.muted:
                return
            super().put(item, block, timeout)


class MessageQueue(Queue):
    """The user message queue. ``get`` is the blocking read ``ask_user`` makes; ``idle_get`` is the worker loop's
    wait for the next message."""

    def __init__(self, gate: TurnGate) -> None:
        super().__init__()
        self._gate = gate
        self.question_pending = False

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        from fastworkflow.workflow_execution_context import CommandCancelledError

        if self._gate.muted:
            raise CommandCancelledError("the turn was interrupted, so its question was not asked")
        self.question_pending = True
        try:
            item = super().get(block, timeout)
        finally:
            self.question_pending = False
        if item is CANCEL_QUESTION:
            raise CommandCancelledError("the question was cancelled")
        return item

    def idle_get(self) -> Any:
        """The worker loop's read between turns; a stray cancel marker is not a message."""
        while True:
            item = Queue.get(self)
            if item is not CANCEL_QUESTION:
                return item

    def post_cancel(self, wait: float = 0.5) -> bool:
        """End the question a blocked ``ask_user`` waits on. False when none is waiting.

        ``ask_user`` queues its question and the trace sentinel, then records the entry, then blocks here, so a
        caller that read the sentinel and cancels at once can be a few microseconds early. While the worker is
        still inside a turn, up to ``wait`` seconds are given for the read to begin; with no turn running there is
        no wait.
        """
        end = time.monotonic() + wait
        while not self.question_pending:
            if not self._gate.in_turn or time.monotonic() >= end:
                return False
            time.sleep(0.005)
        self.put(CANCEL_QUESTION)
        return True
