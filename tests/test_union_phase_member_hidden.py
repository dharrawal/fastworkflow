"""The member tag of a union phase's saved facts (bd fix-ftdl.4.1) stays in the saved state: a stop's facts
(``fw_not_run``) and the answer text built from them never show it. No model is called (no-agent mode)."""
import json

from fastworkflow.planning import run, when
from tests.test_no_agent_mode import _no_agent, _plan, _stops  # noqa: F401 - autouse fixture and helpers
from tests.test_turn_result_capture import initialized_fastworkflow, no_review_gate  # noqa: F401
from tests.test_step_helpers import (
    FoundBooks, OpenedBook, _build_planning_command, library_env)  # noqa: F401


class UnionThenOpenStep:
    def find(self, input) -> FoundBooks | OpenedBook:
        return [run("find_book", query=input.title)]

    def book(self, input, find) -> OpenedBook:
        return [when("a book matched", run("open_book", named=input.title))]

    def __result__(self, input, find, book):
        return "done"


def test_a_stop_after_a_union_phase_shows_no_member_tag(library_env, monkeypatch):
    library_env.planning_command = _build_planning_command(UnionThenOpenStep)
    _plan(monkeypatch, library_env)
    result = library_env.ctx.process_turn("look up the book Dune", unattended=True)

    (stop,) = _stops(result)
    assert stop["phase"] == "book" and list(stop["facts"]) == ["find"]
    assert "__member__" not in json.dumps(stop) and "__member__" not in result.answer
    assert "book_uids" in stop["facts"]["find"]
