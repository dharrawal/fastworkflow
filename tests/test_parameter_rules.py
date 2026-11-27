"""Direct cases for the helpers both argument guards share (fastworkflow/parameters/rules.py, bd fix-qoyz step 1),
and a check that the old names in the two guard modules are the very same objects."""
import pytest
from pydantic import BaseModel, Field

from fastworkflow import planning_commands
from fastworkflow.parameters import rules
from fastworkflow.parameters import argument_checks


def test_each_rule_has_one_home_and_the_old_aliases_are_gone():
    """The shared helpers live in ``fastworkflow.parameters.rules`` and the guards' reasons in ``parameters.check``;
    the names the two guard modules once re-exported while the rules moved are gone."""
    from fastworkflow.parameters import check

    for name in ("clean_str", "fold", "written", "placeholder_like", "written_numbers", "field_extra", "NUMBER_WORDS", "_DASHES",
                 "literal_values", "allowed_member", "not_allowed_why"):
        assert hasattr(rules, name), name
    # the stand-in word list is gone (bd fix-ftdl.6.1): a placeholder is a value that is entirely one template token
    assert not hasattr(rules, "PLACEHOLDER_WORDS") and not hasattr(rules, "_PLACEHOLDER_SHAPE")
    # the ``enum`` key and the ``One of:`` words are gone, and with them their reader and its patterns; a ``Literal``'s
    # values (``literal_values``) are read through ``allowed_member`` / ``not_allowed_why``, which live here only
    for name in ("allowed_values", "_ONE_OF", "_ONE_OF_ITEM", "_ONE_OF_SEPARATOR"):
        assert not hasattr(rules, name), name
    for name in ("NO_VALUE_WHY", "NOT_WRITTEN_WHY", "BRACKET_PLACEHOLDER_WHY", "is_marked",
                 "not_written_fields", "is_given", "as_int", "is_int_field", "is_str_field"):
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
    # at word boundaries (bd fix-ftdl.6.6): a part of a written word is not written
    assert rules.written("abc-12", folded, request) is None and rules.written("nd", folded, request) is None


def test_placeholder_like():
    assert rules.placeholder_like("<item_uid>") is True
    # an upper-case id with a number is a value (bd fix-ftdl.6.1): "SKU_42", "ORDER_123"
    assert rules.placeholder_like("NAME_1") is False and rules.placeholder_like("SKU_42") is False
    # a bracketed value by its shape alone: a variable name is a token, a value in brackets is not
    assert rules.placeholder_like("<ID>") is True and rules.placeholder_like("<Dune>") is False
    assert rules.placeholder_like("{author}") is True
    # a value is judged by its own shape only (bd fix-ftdl.6.1): no stand-in words, no request
    assert rules.placeholder_like("John Doe") is False
    assert rules.placeholder_like("a<b") is False and rules.placeholder_like("use {name} here") is False
    assert rules.placeholder_like("Dune") is False


def test_written_numbers():
    assert rules.written_numbers("Add 1,000 copies", []) >= {1000, 1, 0}
    assert -5 in rules.written_numbers("Remove -5 copies", [])
    assert rules.written_numbers("add three copies", []) == {3}
    assert rules.written_numbers("room 12 and 7", ["room 12"]) == {7}
    assert rules.written_numbers("no numbers here", []) == set()


@pytest.mark.parametrize("request_text, texts, expected", [
    # unchanged: digits, thousands separators, a minus sign, single words zero to twenty, the texts exclusion
    ("show 3 rows", [], {3}),
    ("add 1,200 copies", [], {1200, 1, 200}),  # ambiguous with a list: stays as it is
    ("remove -5 copies", [], {5, -5}),
    ("room 12 and 7", ["room 12"], {7}),
    ("zero", [], {0}), ("seven", [], {7}), ("twenty", [], {20}), ("Show NINE rows", [], {9}),
    ("a2 and x7", [], set()),  # digits glued to letters or codes
    ("the 2nd and second", [], set()), ("a dozen, a couple, both", [], set()),
    # number words that sit together are ONE number: tens + unit (1 to 9), space or hyphen; a tens word alone
    ("page twenty five", [], {25}), ("twenty-one copies", [], {21}), ("thirty one", [], {31}),
    ("Ninety-NINE", [], {99}), ("thirty", [], {30}), ("show forty rows", [], {40}),
    # a stretch the small grammar cannot join soundly yields nothing for its words
    ("one hundred", [], set()), ("one hundred and five", [], set()), ("two thousand", [], set()),
    ("two dozen", [], set()), ("half a dozen eggs", [], set()), ("two three", [], set()),
    ("twenty ten", [], set()), ("twenty twenty", [], set()), ("twenty zero", [], set()),
    ("one two three", [], set()), ("twenty-five hundred", [], set()),
    # separate numbers written with a separator stay separate
    ("two, three", [], {2, 3}), ("two and three", [], {2, 3}), ("two or three", [], {2, 3}),
    ("twenty five and three", [], {25, 3}), ("2 three", [], {2, 3}),
    # a decimal or a dotted token is not a whole number
    ("last 3.5 days", [], set()), ("3.0", [], set()), ("v2.1", [], set()), ("version 1.2.3", [], set()),
    ("host 192.168.0.1", [], set()), ("-3.5", [], set()), ("show 3.", [], {3}), ("show 3. Then 4.", [], {3, 4}),
    ("1,200.50", [], set()),
    # the texts exclusion holds for a compound
    ("the book twenty five", ["the book twenty five"], set()),
    ("the book twenty five, page 7", ["the book twenty five"], {7}),
    ("twenty five copies of the book twenty five", ["the book twenty five"], {25}),
])
def test_written_numbers_reads_a_whole_number_or_nothing(request_text, texts, expected):
    assert rules.written_numbers(request_text, texts) == expected


class _Fields(BaseModel):
    mode: str = Field(json_schema_extra={"copied_from_command_text": True})
    plain: str = ""


def test_field_extra():
    f = _Fields.model_fields
    assert rules.field_extra(f["plain"]) == {}
    assert rules.field_extra(f["mode"]) == {"copied_from_command_text": True}


def test_no_policy_difference_is_left_between_the_two_kinds_of_command():
    """Every switch between the planning and the execution path is decided once; there is no Policy object."""
    import importlib

    import pytest

    with pytest.raises(ImportError):
        importlib.import_module("fastworkflow.parameters.spec")
