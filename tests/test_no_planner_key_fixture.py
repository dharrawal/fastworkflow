"""The ``no_planner_key`` fixture: the key is gone from both places and the caches are clear, inside and after."""
import os

import fastworkflow
from fastworkflow import workflow_agent


def test_the_key_and_the_cache_are_gone_inside_the_fixture(monkeypatch, request):
    monkeypatch.setenv("JEV_API_KEY", "injected-not-a-real-key")
    monkeypatch.setitem(fastworkflow._env_vars, "JEV_API_KEY", "injected-not-a-real-key")
    workflow_agent._jev_lm.cache_clear()   # an earlier test file may have left its own answer cached
    assert workflow_agent._jev_lm() is not None and workflow_agent._jev_lm.cache_info().currsize == 1
    request.getfixturevalue("no_planner_key")
    assert "JEV_API_KEY" not in os.environ and "JEV_API_KEY" not in fastworkflow._env_vars
    assert workflow_agent._jev_lm() is None


# Runs after the test above (file order), whose fixture teardown has just run.
def test_the_cache_is_clear_again_after_the_fixture():
    assert workflow_agent._jev_lm.cache_info().currsize == 0
