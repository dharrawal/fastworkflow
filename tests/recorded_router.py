"""A recorded answer of the consultation ProposalReview head, installed where production caches its router.

The review's reply reader asks the head one question per free-text reply: ``CommandRouter(path).predict_with_details``
(``CommandNamePrediction._predict_impl``). Whether the reader then confirms, declines, edits or asks again is the
reader's code; which command the head named is the head's prediction. The head is a local, gitignored artifact, so a
test of the reader that needs a live head skips on a clean checkout, and goes red when a retrain moves a prediction.

This module records the head's prediction once per reply string (``tests/fixtures/review_reply_router_answers.json``:
labels and details exactly as ``predict_with_details`` returned them, with the head version id and the date) and
replays it, the way other tests replay recorded model outputs. Nothing of the reader is patched or stubbed: the
recording is placed in ``CommandRouter._instances_cache`` under the key the reader looks up, which is the cache
production itself uses. The router refuses a reply that has no recording instead of guessing, and the fixture restores
the cache entry it displaced, so ordering does not matter.

Re-record after a retrain (needs the head), over every file that uses the fixture::

    pytest --record-review-router tests/test_consultation_reply_policy.py tests/test_built_in_vague_replies.py \\
        tests/test_round1_handoff.py

``test_recorded_router_matches_the_head`` (marked ``requires_consultation_head``) reports which replies the current
head now answers differently.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from _pytest.stash import StashKey

from fastworkflow.model_pipeline_training import GLOBAL_CONTEXT_FOLDER, CommandRouter
from tests import consultation_head as head

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "review_reply_router_answers.json"
#: The string the reader builds: ``f"{app_workflow_folderpath}/___command_info/{context}"``.
ROUTER_PATH = f"{head.CONSULTATION}/___command_info/{head.HEAD_CONTEXT}"
_RECORD_OPTION = "record_review_router"
_STASH = StashKey[dict]()


class UnrecordedUtterance(AssertionError):
    """The reader asked the head about a reply the recording does not hold."""


def load_recording(path: Path = FIXTURE_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class RecordedRouter(CommandRouter):
    """A ``CommandRouter`` whose prediction is looked up in a recording. Built without loading a model."""

    def __init__(self, answers: dict):  # noqa: D107 - deliberately not CommandRouter.__init__
        pass

    @classmethod
    def from_recording(cls, recording: dict) -> "RecordedRouter":
        router = object.__new__(cls)
        router._initialised = True
        router.modelpipeline = router
        router.answers = recording["answers"]
        router.asked: list[str] = []
        router.unrecorded: list[str] = []
        return router

    def predict_with_details(self, command: str) -> tuple[list[str], dict]:
        self.asked.append(command)
        answer = self.answers.get(command)
        if answer is None:
            self.unrecorded.append(command)
            raise UnrecordedUtterance(
                f"no recorded head answer for the reply {command!r}. A recorded router never guesses. "
                f"Re-record with `pytest --record-review-router <the reader test files>` (needs the trained head); "
                f"see tests/recorded_router.py."
            )
        return list(answer["labels"]), copy.deepcopy(answer["details"])


class RecordingRouter(CommandRouter):
    """Record mode only: asks the REAL head and keeps its answer for the fixture file."""

    def __init__(self, real: CommandRouter):  # noqa: D107
        pass

    @classmethod
    def around(cls, real: CommandRouter) -> "RecordingRouter":
        router = object.__new__(cls)
        router._initialised = True
        router.real = real
        router.modelpipeline = real.modelpipeline
        router.seen = {}
        return router

    def predict_with_details(self, command: str) -> tuple[list[str], dict]:
        labels, details = self.real.predict_with_details(command)
        self.seen[command] = {"labels": list(labels), "details": copy.deepcopy(details)}
        return labels, details


@contextmanager
def installed(router: CommandRouter, router_path: str = ROUTER_PATH):
    """Place *router* in the process-level router cache under *router_path*; put back what was there."""
    key = router_path.replace("*", GLOBAL_CONTEXT_FOLDER)
    cache = CommandRouter._instances_cache
    absent = object()
    prior = cache.get(key, absent)
    cache[key] = router
    try:
        yield router
    finally:
        if prior is absent:
            cache.pop(key, None)
        else:
            cache[key] = prior


def _merge_into_fixture(seen: dict) -> None:
    recording = load_recording() if FIXTURE_PATH.is_file() else {"answers": {}}
    recording["answers"].update(seen)
    recording["answers"] = dict(sorted(recording["answers"].items()))
    recording["metadata"] = {
        "head": f"consultation/{head.HEAD_CONTEXT}",
        "head_version_id": head.head_version_id(),
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d"),
        "what": "CommandRouter.predict_with_details(reply) of the real head: labels and details as returned.",
        "re_record": "pytest --record-review-router <the files that use recorded_review_router>",
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as handle:
        json.dump(recording, handle, indent=1, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


@pytest.fixture
def recorded_review_router(request):
    """Reader tests: the head's recorded answers for the consultation ProposalReview context, for this test only.

    With ``--record-review-router`` the real head answers instead and the answers are written to the fixture file.
    """
    if request.config.getoption(_RECORD_OPTION, default=False):
        if not head.head_trained():
            pytest.skip(head.missing_head_message())
        recorder = RecordingRouter.around(CommandRouter(head.HEAD_DIR))
        with installed(recorder):
            yield recorder
        store = request.config.stash.get(_STASH, None)
        if store is None:
            store = request.config.stash[_STASH] = {}
        store.update(recorder.seen)
        return
    router = RecordedRouter.from_recording(load_recording())
    with installed(router):
        yield router
    assert not router.unrecorded, f"replies without a recording reached the head: {router.unrecorded}"


def write_recording_if_requested(config) -> None:
    seen = config.stash.get(_STASH, None)
    if config.getoption(_RECORD_OPTION, default=False) and seen:
        _merge_into_fixture(seen)
