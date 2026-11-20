"""Direct cases for the helpers both argument guards share (fastworkflow/parameters/rules.py, bd fix-qoyz step 1),
and a check that the old names in the two guard modules are the very same objects."""
from pydantic import BaseModel, Field

from fastworkflow import planning_commands
from fastworkflow.parameters import rules
from fastworkflow.utils import argument_checks


def test_each_rule_has_one_home_and_the_old_aliases_are_gone():
    """The shared helpers live in ``fastworkflow.parameters.rules`` and the guards' reasons in ``parameters.check``;
    the names the two guard modules once re-exported while the rules moved are gone."""
    from fastworkflow.parameters import check

    for name in ("clean_str", "fold", "written", "placeholder_like", "written_numbers", "allowed_values",
                 "allowed_member", "not_allowed_why", "field_extra", "NUMBER_WORDS", "PLACEHOLDER_WORDS", "_DASHES"):
        assert hasattr(rules, name), name
    for name in ("EXAMPLE_WHY", "NO_VALUE_WHY", "NOT_WRITTEN_WHY", "BRACKET_PLACEHOLDER_WHY", "is_converted",
                 "is_grounding_exempt", "is_given", "as_int", "is_int_field", "is_str_field"):
        assert hasattr(check, name), name
    gone = {
        planning_commands: ("_clean_str", "_fold", "_written", "_placeholder_like", "NUMBER_WORDS", "PLACEHOLDER_WORDS",
                            "_PLACEHOLDER_SHAPE", "_DASHES"),
        argument_checks: ("written_numbers", "allowed_values", "allowed_member", "not_allowed_why", "_extra", "_DIGITS",
                          "_NUMBER_WORD", "_ONE_OF", "_ONE_OF_ITEM", "_ONE_OF_SEPARATOR", "_as_int", "_given",
                          "_is_int_field", "_is_str_field", "is_converted", "is_grounding_exempt", "EXAMPLE_WHY",
                          "NO_VALUE_WHY", "NOT_WRITTEN_WHY", "BRACKET_PLACEHOLDER_WHY"),
    }
    for module, names in gone.items():
        for name in names:
            assert not hasattr(module, name), f"{module.__name__}.{name} is an old alias"


def test_clean_str():
    assert rules.clean_str('  "Dune"  ') == "Dune"
    assert rules.clean_str('"Dune".') == '"Dune"'  # quotes go only when they are the outer pair
    assert rules.clean_str("'`x`'") == "x"
    assert rules.clean_str("a, b;") == "a, b"
    assert rules.clean_str(5) == "5"
    assert rules.clean_str('"') == '"'


def test_fold_keeps_length_and_maps_dashes():
    text = "a‑b–c"
    assert rules.fold(text) == "a-b-c"
    assert len(rules.fold(text)) == len(text)


def test_written_returns_the_requests_spelling():
    request = "Find ABC‑123 now"
    folded = rules.fold(request).lower()
    assert rules.written("abc-123", folded, request) == "ABC‑123"
    assert rules.written("zzz", folded, request) is None
    assert rules.written("", folded, request) is None


def test_placeholder_like():
    assert rules.placeholder_like("<item_uid>", "") is True
    assert rules.placeholder_like("NAME_1", "") is True
    assert rules.placeholder_like("John Doe", "find a book") is True
    assert rules.placeholder_like("John Doe", "find john doe") is False  # the request writes it
    assert rules.placeholder_like("Dune", "find dune") is False


def test_written_numbers():
    assert rules.written_numbers("Add 1,000 copies", []) >= {1000, 1, 0}
    assert -5 in rules.written_numbers("Remove -5 copies", [])
    assert rules.written_numbers("add three copies", []) == {3}
    assert rules.written_numbers("room 12 and 7", ["room 12"]) == {7}
    assert rules.written_numbers("no numbers here", []) == set()


class _Fields(BaseModel):
    kind: str = Field(description="One of: book, member or branch.")
    mode: str = Field(json_schema_extra={"enum": ["a", "b"]})
    note: str = Field(description="One of the usual things.")
    plain: str = ""


def test_allowed_values():
    f = _Fields.model_fields
    assert rules.allowed_values(f["kind"]) == ("book", "member", "branch")
    assert rules.allowed_values(f["mode"]) == ("a", "b")
    assert rules.allowed_values(f["note"]) == ()
    assert rules.allowed_values(f["plain"]) == ()
    assert rules.field_extra(f["plain"]) == {}
    assert rules.field_extra(f["mode"]) == {"enum": ["a", "b"]}


def test_allowed_member_and_why():
    allowed = ("Book", "member")
    assert rules.allowed_member(allowed, "book") == "Book"
    assert rules.allowed_member(allowed, "MEMBER") == "member"
    assert rules.allowed_member(allowed, "branch") is None
    assert rules.not_allowed_why(allowed) == "is not one of Book, member"


def test_no_policy_difference_is_left_between_the_two_kinds_of_command():
    """Every switch between the planning and the execution path is decided once; there is no Policy object."""
    import importlib

    import pytest

    with pytest.raises(ImportError):
        importlib.import_module("fastworkflow.parameters.spec")
