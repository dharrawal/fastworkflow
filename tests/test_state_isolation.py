"""The per-test state-root isolation contract (`fix-s21`).

`state_paths._optional_env` reads `fastworkflow._env_vars` BEFORE `os.environ`,
and `fastworkflow.init()` REPLACES that dict wholesale. So a single test calling
``init({"FASTWORKFLOW_STATE_ROOT": ...})`` used to pin every later test in the
process to its own tmp directory, silently overriding the OS variable
``conftest.isolate_state_root`` sets — and because nothing restored the dict, the
override lasted for the rest of the session.

The symptom was not a failure in the offending test. It was 82 ERRORs in an
unrelated module's fixture (`assert conv_id == 1` failing with 2, 3, 4…, because
every test was minting conversations into one leaked database), reachable only
when the two files shared a process in a particular order. A run that did not
hit that order looked completely green while those 82 tests had never executed.

These tests pin the contract from both directions. The order inside this file is
load-bearing: `test_a_pins_a_state_root_through_the_env_dict` must run before
`test_b_...`, which is what pytest guarantees for tests in one file.
"""

from __future__ import annotations

import os

import fastworkflow
from fastworkflow import state_paths


LEAKY_MARKER = "s21-leaked-root"


def test_a_pins_a_state_root_through_the_env_dict(tmp_path):
    """Do exactly what the offending fixture does, deliberately."""
    pinned = tmp_path / LEAKY_MARKER
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(pinned)})
    # The dict really does win over the OS environment — that is the documented
    # behaviour this test is not trying to change.
    assert state_paths.state_root() == str(pinned.resolve())


def test_b_the_pin_did_not_escape_into_this_test():
    """The next test must see the fixture's root, not the previous test's.

    If this fails, `conftest.isolate_state_root` has stopped restoring
    `fastworkflow._env_vars` and every test after a state-root-pinning one is
    sharing a database again.
    """
    assert "FASTWORKFLOW_STATE_ROOT" not in fastworkflow._env_vars
    root = state_paths.state_root()
    assert LEAKY_MARKER not in root
    assert root == os.path.abspath(os.environ["FASTWORKFLOW_STATE_ROOT"])


# The state root's PATH is deliberately recycled between tests: the fixture
# rmtree's it at teardown, so `tmp_path_factory.mktemp` hands out the same
# `fw_state_root0` name again. What must not be recycled is its CONTENTS — a
# leftover observability.sqlite3 from a previous test is exactly what turned
# `assert conv_id == 1` into 2, 3, 4… So the property to pin is emptiness at
# setup, not uniqueness of the path.


def test_c_writes_into_the_state_root():
    root = state_paths.state_root()
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "leftover.sqlite3"), "w") as handle:
        handle.write("a previous test's database")
    assert os.path.exists(os.path.join(root, "leftover.sqlite3"))


def test_d_starts_from_an_empty_state_root():
    """The next test must not inherit the previous test's files."""
    root = state_paths.state_root()
    leftovers = os.listdir(root) if os.path.isdir(root) else []
    assert "leftover.sqlite3" not in leftovers
    assert leftovers == []
