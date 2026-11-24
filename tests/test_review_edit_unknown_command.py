"""A review edit whose means names no command is a failed edit, never an ad hoc step (bd fix-ftdl.11).

The free-text reader of the sub-task review can return an ``add`` whose goal line carries a means, or a ``change`` of a
step's means, naming a command the workflow does not have (a name made up from the request's words). Such an edit
fails as an edit naming an input the step does not have fails: it is read once more with the reason
(``RepairSubTaskEdits``), and when it is still wrong the user is asked again with the commands a step can name; nothing
of the reply is applied and no step becomes ad hoc. An ``add`` with no means is decided as before, and a means naming a
real command in another spelling (case, the leaf of a qualified name) is resolved to that command.

Integration: a temporary copy of ``planning_tutorial_workflow`` with one planning command added, the real review engine
and reader, and the planner's model scripted (``DummyLM``), as ``tests/test_review_reader_repairs.py`` does.
"""
from __future__ import annotations

import pytest

from fastworkflow import review_engine as re_
from fastworkflow.review_engine import proposal_of, reply_to_proposal

from tests.test_review_reader_repairs import _Reader, _review, _step, workflow  # noqa: F401

UNKNOWN = "planning:audit_linked_rights"
REPLY = "also audit the rights linked to Ann Lee"


def _means(record) -> list[str]:
    return [item.means.render() for item in proposal_of(record).items]


@pytest.mark.parametrize("edit", [
    {"op": "add", "after": 1, "value": f"audit the rights linked to Ann Lee (inputs: none; means: {UNKNOWN})"},
    {"op": "change", "index": 1, "field": "means", "value": UNKNOWN},
])
def test_a_means_that_names_no_command_is_asked_again_with_the_commands(workflow, edit):
    reader = _Reader([edit], [edit])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, REPLY)
    assert not review.done and record.state == re_.AWAITING
    assert "'audit_linked_rights' is not a command available here" in review.text
    assert "The commands are: author_books, " in review.text and "linked_records, and 7 more." in review.text
    assert _means(record) == ["planning:linked_records"]              # nothing applied, no ad hoc step
    assert len(reader.prompts) == 2                                     # the one repair read, as for any failed edit
    assert "is not a command available here" in reader.prompts[1]


def test_the_other_edits_of_a_reply_with_an_unknown_means_are_not_applied(workflow):
    edits = [{"op": "change", "index": 1, "field": "kind", "value": "member"},
             {"op": "add", "after": 1, "value": f"audit the rights linked to Ann Lee (inputs: none; means: {UNKNOWN})"}]
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=_Reader(edits, edits))
    review = reply_to_proposal(host, store, record, "Dune is a member; " + REPLY)
    assert not review.done and record.state == re_.AWAITING
    assert [(i.name, i.value) for i in proposal_of(record).items[0].inputs] == [("name", "Dune"), ("kind", "book")]
    assert _means(record) == ["planning:linked_records"]


def test_the_repair_read_that_names_a_real_command_applies(workflow):
    reader = _Reader([{"op": "change", "index": 1, "field": "means", "value": UNKNOWN}],
                     [{"op": "change", "index": 1, "field": "means", "value": "planning:list_books"}])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    reply_to_proposal(host, store, record, "use the book list instead")
    assert _means(record) == ["planning:list_books"]


def test_an_add_with_no_means_is_decided_as_before(workflow):
    reader = _Reader([{"op": "add", "after": 1, "value": "audit the rights linked to Ann Lee"}])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, REPLY)
    assert record.state == re_.REVISED, review.text
    assert _means(record) == ["planning:linked_records", "from_scratch"]
    assert len(reader.prompts) == 1


@pytest.mark.parametrize("written", ["planning:list_books", "List_Books", "plain:list books"])
def test_a_means_naming_a_real_command_in_another_spelling_applies(workflow, written):
    reader = _Reader([{"op": "change", "index": 1, "field": "means", "value": written}])
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, "use the book list instead")
    assert record.state == re_.REVISED, review.text
    assert _means(record) == ["planning:list_books"]
    assert len(reader.prompts) == 1


# ---- the typed form: one rule, no model call ------------------------------------------------------------------------

def test_a_typed_means_that_names_no_command_is_refused_with_the_commands(workflow):
    reader = _Reader()                                          # no answer scripted: the typed path calls no model
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, f"change 1: means={UNKNOWN}")
    assert not review.done and record.state == re_.AWAITING
    assert "'audit_linked_rights' is not a command available here" in review.text
    assert "linked_records, and 7 more." in review.text
    assert _means(record) == ["planning:linked_records"]
    assert reader.prompts == []


@pytest.mark.parametrize("written", ["planning:list_books", "List_Books", "plain:list books"])
def test_a_typed_means_naming_a_real_command_in_another_spelling_applies(workflow, written):
    reader = _Reader()
    host, store, record = _review(workflow, _step(name="Dune", kind="book"), reader=reader)
    review = reply_to_proposal(host, store, record, f"change 1: means={written}")
    assert record.state == re_.REVISED, review.text
    assert _means(record) == ["planning:list_books"]
    assert reader.prompts == []
