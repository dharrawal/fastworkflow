"""F9 (fix-iq53.2.14): the boundary surface, and the one hole left in it.

F9's brief was "delete the superseded backend restrictions and verify generic
paging". The named targets were three: the rule that a descriptor's
``ordering`` had to be ``UNSORTED_OFFSET`` (ido-gqv.6 B0), the rule that its
``timeslot`` had to be ``None`` (ido-986.14.1), and ``view`` being required as
a positional. **All three were already gone**, removed by F1 (fix-iq53.2.5)
along with the fields they policed -- there was nothing left for F9 to delete,
and against a standing "delete nothing as part of this preparation" it deleted
nothing.

What is worth having instead is the guard that says so and keeps saying it.
Those rules were not removed to make the type smaller; they were removed
because a framework that can NAME an ordering, a snapshot pin or a SQL view is
one workflow's query object, and the guarantee is stronger when there is no
field to put one in. A field re-added quietly would re-import the restriction
with it, and nothing else in the suite would notice.

So this module pins the shape of the boundary rather than any one walk over
it:

* the descriptor's six fields, the batch request's four and the terminal
  request's four, by name;
* that the descriptor's constructor refuses exactly two things, and that both
  are about being usable at all rather than about one backend's policy;
* and that the package reads inside the adapter's opaque ``state`` in exactly
  ONE place -- the origin backstop, which is a known, named, tracked boundary
  violation with a scheduled removal, and the only one.

That last test is the useful one. "``state`` is never inspected" is the
sentence the whole descriptor reduction rests on, it is not quite true today,
and the honest way to hold it is to count the exceptions rather than to repeat
the claim.

Generic paging itself is verified by walking it, not by inspecting it:
``tests/test_result_handles_non_database.py`` runs the same contract over
in-memory objects with no query language, and
``tests/test_result_handle_callback_budget.py`` runs it over the portal
fixture and pins the callback profile of both.
"""
from __future__ import annotations

import ast
import pathlib
import unittest

from fastworkflow.result_handles import (
    ResultHandleError,
    SourceDescriptor,
    SourceRequest,
    TerminalRequest,
)

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "fastworkflow" / "result_handles"

#: Where the package is allowed to name the adapter's ``state`` key, and why.
#: A function not in here that reads ``state`` is a new boundary violation.
STATE_READERS = {
    # The pass-through. ``as_dict`` copies the mapping so the stored JSON owns
    # its own dict; it does not look at a single key inside it.
    "models.py": {"SourceDescriptor"},
    # (ido-oon / fix-iq53.2.8) THE backstop, and the only one. `_issue_terminal`
    # short-circuits on a partway origin before making any callback, reading
    # `state["start_offset"]` to do it. It stays until IDO's ido-0rk.2.1 I4
    # makes the adapter answer `offset_origin_not_zero` itself, because
    # deleting it first would leave a handle that covers part of a relation
    # free to be reported as covering all of it -- and that failure is silent.
    "paging.py": {"_origin_offset"},
}


def state_readers() -> dict[str, dict[str, list[int]]]:
    """Every function in the package that names the ``"state"`` key, by module.

    Read off the AST rather than by grepping, so a mention inside a docstring
    or a comment -- of which there are many, because this is the field the
    whole design argument is about -- is not mistaken for a read of it.
    """
    found: dict[str, dict[str, list[int]]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                for child in ast.walk(node):
                    child.__dict__.setdefault("_owner", node.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "state":
                owner = getattr(node, "_owner", "<module>")
                found.setdefault(path.name, {}).setdefault(
                    owner, []).append(node.lineno)
    return found


class BoundaryTypeSurfaceTests(unittest.TestCase):
    """Six fields, four, and four. Counted by name, not by arithmetic."""

    def test_the_descriptor_carries_exactly_the_six_generic_fields(self):
        self.assertEqual(
            list(SourceDescriptor.__dataclass_fields__),
            ["resolver", "uid_field", "label_fields", "batch_size",
             "filter_columns", "state"])

    def test_the_batch_request_carries_exactly_four(self):
        self.assertEqual(
            list(SourceRequest.__dataclass_fields__),
            ["descriptor", "continuation", "limit", "contains"])

    def test_the_terminal_request_carries_exactly_four_and_no_batch_count(self):
        """``batches_read`` was ruled out, and the ruling is asserted here.

        It had no named consumer on the adapter side, which is the shape two
        earlier revisions of this design were rejected for. The number still
        reaches the observability store on ``result_handle_reconciled``, which
        is framework-side and needs no boundary crossing to get there.
        """
        self.assertEqual(
            list(TerminalRequest.__dataclass_fields__),
            ["descriptor", "continuation", "contains", "distinct_uids"])
        self.assertNotIn("batches_read", TerminalRequest.__dataclass_fields__)

    def test_the_descriptor_refuses_exactly_two_things(self):
        """Both about being usable at all; neither about one backend's policy.

        ``tests/test_result_handles.py`` already asserts that an ordering, a
        timeslot and a view cannot be CONSTRUCTED. What is asserted here is the
        complement: nothing took their place. Two rules, both generic -- a
        resolver that can be looked up, and a batch size that can be asked for.
        """
        with self.assertRaises(ResultHandleError):
            SourceDescriptor(resolver="")
        with self.assertRaises(ResultHandleError):
            SourceDescriptor(resolver="probe", batch_size=0)
        with self.assertRaises(ResultHandleError):
            SourceDescriptor(resolver="probe", batch_size=-1)

        source = ast.parse((PACKAGE / "models.py").read_text(encoding="utf-8"))
        post_init = next(
            node for node in ast.walk(source)
            if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
        raises = [node for node in ast.walk(post_init) if isinstance(node, ast.Raise)]
        self.assertEqual(len(raises), 2,
                         "a constructor rule was added to the descriptor")

    def test_everything_the_deleted_rules_policed_is_carried_in_state(self):
        """The values did not stop existing; they stopped being the framework's.

        An adapter that needs a sort, a snapshot pin or a view name puts it
        here, where this package carries it verbatim and cannot act on it.
        """
        carried = SourceDescriptor(
            resolver="probe",
            state={"ordering": "sorted-offset", "timeslot": "2026-09-14",
                   "view": "ido_groupDetail_identity", "start_offset": 100},
        )
        self.assertEqual(carried.state["ordering"], "sorted-offset")
        self.assertEqual(carried.as_dict()["state"]["view"],
                         "ido_groupDetail_identity")


class OpaqueStateTests(unittest.TestCase):
    """How many holes are in "never inspected", and where they are."""

    def test_the_package_reads_inside_state_in_exactly_one_place(self):
        """The origin backstop, and nothing else.

        If this fails with a reader you did not expect, a domain fact has
        crossed back over the boundary and the descriptor reduction is being
        undone one key at a time.

        If it fails because ``_origin_offset`` is GONE, that is IDO's I4
        landing and it is good news: drop ``paging.py`` from ``STATE_READERS``
        entirely, and the claim that this package never inspects the adapter's
        state becomes true without an exception attached.
        """
        found = state_readers()
        self.assertEqual(
            {module: set(owners) for module, owners in found.items()},
            STATE_READERS,
            "the set of functions that read the adapter's opaque state changed")

    def test_the_backstop_is_the_only_thing_the_walk_reads_an_origin_for(self):
        """One read, on one line, and it is the short-circuit's own.

        Counted because "one function" and "one read" are different claims:
        a backstop that grew a second lookup would still pass the test above.
        """
        found = state_readers()
        self.assertEqual(len(found["paging.py"]["_origin_offset"]), 1)


if __name__ == "__main__":
    unittest.main()
