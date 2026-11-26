"""Run the independent per-sub-task model calls of one turn side by side, with the outcome of running them in order (fix-awg8).

After a request is split, three stages do one model-backed step per sub-task: the referent, the match to a planning
command and the draft of the arguments. Each step reads only its own sub-task's text (and, for a draft, the MEANS of the
earlier sub-tasks, which the match stage has already fixed), so the steps of one stage can wait on the model together.
``map_in_order`` is that one pooling pattern:

- one worker per call, at most ``FW_SUBTASK_WORKERS`` (default 4) at a time; with one call, or the setting at 1, nothing
  is pooled and the calls run in the caller's thread in order, exactly as before;
- each call runs under its own copy of the caller's context (``tracing.traced_call``), the pattern the argument
  checks already use, so the turn's observability host, progress listener and per-turn memo are the ones the caller has,
  and each call's spans are parented by the span that was current when it was submitted, never by a sibling call's;
- results come back in call order whatever the completion order;
- every call is awaited before the function returns or raises (no thread outlives it); the exception of the first call
  in order is raised and the other calls' exceptions are logged, not dropped.

Worst case of model calls in flight at once for n sub-tasks: ``min(n, workers)`` times the widest stage of one call. A
draft's widest stage is its argument-check pool (one call per filled argument, or per verified argument), so with the
default of 4 workers that is four times that width, plus the one extraction of a draft that has not reached its checks.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence, TypeVar

from fastworkflow import tracing
from fastworkflow.utils.logging import logger

T = TypeVar("T")

SUBTASK_WORKERS_ENV_VAR = "FW_SUBTASK_WORKERS"
DEFAULT_SUBTASK_WORKERS = 4


def worker_limit() -> int:
    """``FW_SUBTASK_WORKERS`` (default 4, at least 1). A value that is not a positive integer logs a warning and the
    default applies. 1 runs every call in order in the caller's thread."""
    import fastworkflow

    raw = fastworkflow.get_env_var(SUBTASK_WORKERS_ENV_VAR, default=str(DEFAULT_SUBTASK_WORKERS))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = 0
    if value < 1:
        logger.warning(f"{SUBTASK_WORKERS_ENV_VAR}={raw!r} is not a positive integer; using {DEFAULT_SUBTASK_WORKERS}")
        return DEFAULT_SUBTASK_WORKERS
    return value


def is_parallel(count: int) -> bool:
    """Whether *count* calls would run side by side (more than one call and more than one worker)."""
    return count > 1 and worker_limit() > 1


def map_in_order(calls: Sequence[Callable[[], T]], *, before: Optional[Callable[[int], None]] = None) -> list[T]:
    """The results of *calls*, in call order. ``before(index)`` (optional, 0-based) is called in the caller's thread
    just before call ``index`` starts, in order: right before it runs when sequential, in a row ahead of the pool when
    not, so the stage's start events are one per call and in order either way.

    Sequential (one call, or one worker): each call runs in the caller's thread and an exception stops the rest, as a
    loop would. Pooled: all calls are started, all are awaited, and the first exception in call order is raised.
    """
    if not is_parallel(len(calls)):
        out = []
        for index, call in enumerate(calls):
            if before is not None:
                before(index)
            out.append(call())
        return out
    with ThreadPoolExecutor(min(len(calls), worker_limit())) as pool:
        futures = []
        for index, call in enumerate(calls):
            if before is not None:
                before(index)
            futures.append(pool.submit(tracing.traced_call(call)))
        results: list[T] = []
        failures: list[BaseException] = []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:  # noqa: BLE001 - re-raised below, in call order
                failures.append(error)
    if failures:
        for later in failures[1:]:
            logger.warning(f"a sub-task call failed after an earlier one did ({type(later).__name__}: {later})")
        raise failures[0]
    return results
