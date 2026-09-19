"""F9 (fix-iq53.2.14): the boundary surface, and the hole that used to be in it.

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
  ZERO places. It was ONE until IDO's ido-0rk.2.1 I4 landed: the origin
  backstop, a known, named, tracked boundary violation with a scheduled
  removal. The removal happened, and it was the last one.

Those last tests are the useful ones. "``state`` is never inspected" is the
sentence the whole descriptor reduction rests on; it was not quite true while
the backstop stood, and the honest way to hold it then was to count the
exceptions rather than to repeat the claim. The count is zero now, so the two
tests hold the claim from both ends: one bounds WHICH functions may name the
key at all, and the other asserts that none of them -- present or added later
-- reaches a value out from inside it. Naming and reading are different acts,
and only one of them is the violation: ``as_dict`` names the key to copy the
mapping whole, which is the pass-through the whole boundary is built on.

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
#: A function not in here that names ``state`` is a new boundary violation.
STATE_READERS = {
    # The pass-through, and now the only entry. ``as_dict`` copies the mapping
    # so the stored JSON owns its own dict; it does not look at a single key
    # inside it.
    #
    # (ido-oon / fix-iq53.2.8 / ido-0rk.2.1 I4) `paging.py` used to be here too,
    # for `_origin_offset`: `_issue_terminal` short-circuited on a partway
    # origin before making any callback, reading `state["start_offset"]` to do
    # it. It was kept deliberately until IDO's I4 made the adapter answer
    # `offset_origin_not_zero` itself, because deleting it first would have left
    # a handle that covers part of a relation free to be reported as covering
    # all of it -- and that failure is silent. I4 landed, so the entry is gone
    # and the claim needs no exception attached to it.
    "models.py": {"SourceDescriptor"},
}

#: Mapping reads that would pull a value out from inside ``state``.
STATE_INTERIOR_CALLS = frozenset(
    {"get", "keys", "values", "items", "pop", "setdefault"})


def _tag_owners(tree: ast.AST) -> None:
    """Label every node with the function or class that encloses it.

    Outermost wins: ``setdefault`` over a walk that reaches a ``ClassDef``
    before its own methods reports ``SourceDescriptor`` rather than ``as_dict``,
    which is the granularity the allow-list is written at.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            for child in ast.walk(node):
                child.__dict__.setdefault("_owner", node.name)


def state_readers() -> dict[str, dict[str, list[int]]]:
    """Every function in the package that names the ``"state"`` key, by module.

    Read off the AST rather than by grepping, so a mention inside a docstring
    or a comment -- of which there are many, because this is the field the
    whole design argument is about -- is not mistaken for a read of it.
    """
    found: dict[str, dict[str, list[int]]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _tag_owners(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "state":
                owner = getattr(node, "_owner", "<module>")
                found.setdefault(path.name, {}).setdefault(
                    owner, []).append(node.lineno)
    return found


def _names_state_key(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "state"


def _is_state_mapping(node: ast.AST) -> bool:
    """Does this expression evaluate to the adapter's ``state`` mapping itself?

    Three ways to get hold of it: ``descriptor["state"]``,
    ``descriptor.get("state")``, and the dataclass's own ``.state``. Getting
    hold of it is not the violation -- ``as_dict`` does exactly that to copy
    the whole mapping into the payload verbatim. Reaching a key WITHIN it is.
    """
    if isinstance(node, ast.Attribute) and node.attr == "state":
        return True
    if isinstance(node, ast.Subscript) and _names_state_key(node.slice):
        return True
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and bool(node.args)
            and _names_state_key(node.args[0]))


def _state_bound_names(tree: ast.AST) -> set[str]:
    """Locals holding the state mapping, so a two-step read is still caught.

    ``state = descriptor.get("state")`` and then ``state.get("start_offset")``
    is how the deleted backstop was written, and either line alone is
    innocent. A parameter named ``state`` counts as bound too: handing the
    mapping to a helper is the obvious way to move a read somewhere a
    single-expression scan would not look for it.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg == "state":
            names.add(node.arg)
            continue
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        if node.value is None:
            continue
        if not any(_is_state_mapping(inner) for inner in ast.walk(node.value)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def state_interior_reads() -> dict[str, dict[str, list[int]]]:
    """Every place the package pulls a value out from INSIDE ``state``.

    A stricter question than :func:`state_readers` asks, and the one the
    boundary actually rests on. Naming the key is allowed and necessary.
    Subscripting the mapping, or calling a mapping reader on it, means the
    framework is acting on a domain fact it agreed not to understand -- and
    there is no longer anywhere it may do that.

    Off the AST for the same reason :func:`state_readers` is: ``state`` is the
    word this design argument is conducted in, so it appears in dozens of
    comments and docstrings that read nothing at all.
    """
    found: dict[str, dict[str, list[int]]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _tag_owners(tree)
        bound = _state_bound_names(tree)
        for node in ast.walk(tree):
            if _is_state_mapping(node):
                # Getting hold of the mapping, not reaching inside it.
                continue
            if isinstance(node, ast.Subscript):
                reached = node.value
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr in STATE_INTERIOR_CALLS):
                reached = node.func.value
            else:
                continue
            if not any(_is_state_mapping(inner)
                       or (isinstance(inner, ast.Name) and inner.id in bound)
                       for inner in ast.walk(reached)):
                continue
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
    """How many holes are in "never inspected", and where they are. None, now."""

    def test_the_package_reads_inside_state_in_zero_places(self):
        """Nothing in the package reaches a key out of the adapter's state.

        The direct form of the claim the descriptor reduction rests on, and the
        one that has teeth: it does not ask which functions are allowed to
        mention ``state``, it asks whether any code anywhere in the package
        subscripts the mapping or calls a mapping reader on it. The backstop
        this replaces would have been caught by it -- ``_origin_offset`` bound
        the mapping to a local and then asked it for ``start_offset``, which is
        the two-step shape :func:`_state_bound_names` exists to follow.

        A failure here is a domain fact crossing back over the boundary, and
        the fix is never to widen this test. It is to move the read to the
        adapter, which owns ``state``, knows what its keys mean, and can answer
        from them without the framework learning one backend's vocabulary.
        """
        self.assertEqual(
            state_interior_reads(), {},
            "the package reached inside the adapter's opaque state")

    def test_only_the_pass_through_names_the_state_key_at_all(self):
        """The outer bound, kept because it catches what the inner one cannot.

        A function that names ``state`` without reading inside it is not a
        violation yet, but it is where one would appear, and the allow-list is
        short enough to be worth pinning by name: a new entry is a new place
        for the next key lookup to land. This is the test that was asserting
        ONE reader, the origin backstop; IDO's ido-0rk.2.1 I4 took that rule
        over, so the backstop went and the count is the pass-through alone.
        """
        found = state_readers()
        self.assertEqual(
            {module: set(owners) for module, owners in found.items()},
            STATE_READERS,
            "the set of functions that name the adapter's opaque state changed")

    def test_the_pass_through_copies_the_mapping_and_asks_it_nothing(self):
        """One mention, on one line, and it is the payload copy's own.

        Counted because "one function" and "one mention" are different claims:
        a pass-through that grew a key lookup beside the copy would still pass
        the test above, and this is the line that would have to change for it
        to.
        """
        found = state_readers()
        self.assertEqual(len(found["models.py"]["SourceDescriptor"]), 1)


if __name__ == "__main__":
    unittest.main()
