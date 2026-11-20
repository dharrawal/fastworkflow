"""The shared ``env_files`` fixture and the two conftest rules around it (fix-xd2s, fix-iq53.1.17, .19, .20).

* ``env_files`` is written from the tracked template and never reads ``env/.env`` or ``passwords/.env``, so every
  server-layer test runs the same on a clean checkout and on a developer's box.
* ``requires_llm_key`` means what it says: the test skips unless a real key is configured.
* A server-layer test (one asking for ``env_files`` or ``app_module``) that skips without that marker is a failure.

The rules are proved by running real pytest on small generated test files (``pytester``), with no mock.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from tests import llm_credentials

pytest_plugins = ("pytester",)

_CONFTEST = "from tests.conftest import pytest_runtest_makereport, pytest_runtest_setup  # noqa: F401\n"


def test_env_files_are_written_from_the_template_with_placeholders_only(env_files, tmp_path):
    env_file, passwords_file = (Path(p) for p in env_files)
    assert env_file.is_file() and passwords_file.is_file()
    assert tmp_path in env_file.parents and tmp_path in passwords_file.parents

    passwords = dotenv_values(passwords_file)
    template = dotenv_values(Path(fastworkflow.get_fastworkflow_package_path()) / "examples" / "fastworkflow.passwords.env")
    template_keys = {name for name in template if name.startswith("LITELLM_API_KEY_")}
    assert template_keys and set(passwords) == template_keys
    assert "JEV_API_KEY" not in passwords
    assert not any(llm_credentials._looks_like_real_key(value) for value in passwords.values())

    env = dotenv_values(env_file)
    for role in (name[len("LITELLM_API_KEY_"):] for name in template_keys):
        assert env[f"LLM_{role}"] == "mistral/mistral-small-latest", f"LLM_{role} is not the template default"


def test_env_files_do_not_depend_on_the_developers_own_files(env_files):
    """No secret of ``env/.env`` or ``passwords/.env`` is in the pair (values are compared, never shown)."""
    root = Path(__file__).resolve().parent.parent
    own = {
        value
        for name in ("env/.env", "passwords/.env")
        for key, value in dotenv_values(root / name).items()
        if value and key.startswith(("LITELLM_API_KEY_", "JEV_API_KEY", "LITELLM_PROXY_API_KEY"))
    }
    written = {value for path in env_files for value in dotenv_values(path).values() if value}
    assert not own & written


def test_a_server_layer_skip_without_the_marker_is_a_failure(pytester):
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def app_module():
            pytest.skip("a silent hole")

        def test_unmarked(app_module):
            pass

        @pytest.mark.requires_llm_key
        def test_marked(app_module):
            pass
        """
    )
    result = pytester.runpytest("-p", "no:cacheprovider", "-rs")
    outcomes = result.parseoutcomes()
    # A skip raised while setting up a fixture is reported as an error: it is a failure either way, not a skip.
    assert outcomes.get("errors", 0) + outcomes.get("failed", 0) == 1 and outcomes.get("skipped") == 1, result.stdout.str()
    result.stdout.fnmatch_lines(["*server-layer test skipped without the requires_llm_key marker*a silent hole*"])


def test_a_skip_in_the_test_body_stays_a_skip(pytester):
    """The jsdom DOM checks skip in their own body, with their own gate; only a setup skip is a hole."""
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def app_module():
            return None

        def test_declared_condition(app_module):
            pytest.skip("TEST_JSDOM_ROOT is not set")
        """
    )
    assert pytester.runpytest("-p", "no:cacheprovider").parseoutcomes().get("skipped") == 1


def test_a_skip_outside_the_server_layer_stays_a_skip(pytester):
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(
        """
        import pytest

        def test_plain():
            pytest.skip("not a server test")
        """
    )
    assert pytester.runpytest("-p", "no:cacheprovider").parseoutcomes().get("skipped") == 1


@pytest.mark.parametrize("have_key", [False, True])
def test_requires_llm_key_skips_unless_a_real_key_is_present(pytester, monkeypatch, have_key):
    monkeypatch.setattr(llm_credentials, "real_llm_credentials_available", lambda: have_key)
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.requires_llm_key
        def test_live():
            pass
        """
    )
    outcomes = pytester.runpytest("-p", "no:cacheprovider").parseoutcomes()
    assert outcomes.get("passed", 0) == int(have_key) and outcomes.get("skipped", 0) == int(not have_key)
