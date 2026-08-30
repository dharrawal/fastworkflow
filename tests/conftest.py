"""
Pytest configuration and shared fixtures for FastWorkflow tests.
"""

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up environment for tests
os.environ.setdefault("PYTEST_RUNNING", "1")

# Imported after the sys.path insert above so the LOCAL package is the one whose
# process-global `_env_vars` the isolation fixture snapshots and restores.
import fastworkflow  # noqa: E402


@pytest.fixture(autouse=True, scope="function")
def isolate_state_root(tmp_path_factory):
    """Point FASTWORKFLOW_STATE_ROOT at a private temp dir for every test.

    Persistent state (conversations, suspended sessions, checkpoints, function
    caches) is rooted at FASTWORKFLOW_STATE_ROOT, defaulting to
    ~/.local/state/fastworkflow. Without this fixture a test that touches disk
    state would write into the developer's real home directory and could observe
    another test's records. Tests that need a specific root still override it via
    their own init dict (the dict wins over the OS environment).

    **The env dict is restored too, and that half is load-bearing** (`fix-s21`).
    `state_paths._optional_env` reads `fastworkflow._env_vars` BEFORE
    `os.environ`, and `fastworkflow.init()` REPLACES that dict wholesale
    (`__init__.py:253`) — so a single test calling
    `init({"FASTWORKFLOW_STATE_ROOT": ...})` pinned every later test in the
    process to its own tmp directory, silently overriding the OS variable this
    fixture sets. The symptom was 82 ERRORs in an unrelated module's fixture
    (`assert conv_id == 1` failing with 2, 3, 4… because every test was minting
    into one leaked database) — visible only when the two files shared a process
    in an order that put the offender first. Collection order is alphabetical
    here, and `test_run_chatbot_server` happens to sort BEFORE
    `test_turn_and_cme_continuation`, which is the only reason the full suite
    stayed green while the leak was live for everything sorting after it.

    Restoring here keeps the documented "dict wins" behaviour inside a test
    while making it impossible for that win to escape one. Note the ordering
    that makes this safe: pytest instantiates module- and session-scoped
    fixtures BEFORE function-scoped ones, so the snapshot below is taken after
    any higher-scoped `init()` has run and the restore preserves it. Only what a
    function-scoped fixture or the test body sets is reverted.
    """
    previous = os.environ.get("FASTWORKFLOW_STATE_ROOT")
    previous_env_vars = dict(fastworkflow._env_vars)
    root = tmp_path_factory.mktemp("fw_state_root")
    os.environ["FASTWORKFLOW_STATE_ROOT"] = str(root)
    try:
        yield str(root)
    finally:
        if previous is None:
            os.environ.pop("FASTWORKFLOW_STATE_ROOT", None)
        else:
            os.environ["FASTWORKFLOW_STATE_ROOT"] = previous
        # Rebind rather than mutate: `init` rebinds the module global, so the
        # object this fixture saw at setup may no longer be the live one.
        fastworkflow._env_vars = previous_env_vars
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def setup_test_environment():
    """Set up the test environment."""
    # Ensure we're using the local fastworkflow module
    import fastworkflow
    
    # Initialize with minimal configuration for all tests
    fastworkflow.init({})
    
    yield
    
    # Cleanup after all tests


@pytest.fixture(scope="session", autouse=True)
def add_workflow_paths_to_syspath():
    """
    Add workflow paths to sys.path to enable relative imports in workflow modules.
    This simulates what Workflow class does in production code.
    """
    # Store original sys.path to restore later
    original_sys_path = list(sys.path)
    
    # Add common workflow paths used in tests
    workflow_paths = [
        Path(project_root) / "fastworkflow" / "examples" / "retail_workflow",
        Path(project_root) / "fastworkflow" / "examples" / "simple_workflow_template",
        Path(project_root) / "fastworkflow" / "examples" / "hello_world",
        # Add any other workflow paths used in tests
    ]
    
    for path in workflow_paths:
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    yield
    
    # Restore original sys.path after tests complete
    sys.path = original_sys_path


@pytest.fixture
def add_temp_workflow_path():
    """
    Fixture for tests that create temporary workflow paths.
    Usage: Call this fixture with the temporary path to add it to sys.path.
    
    Example:
        def test_something(tmp_path, add_temp_workflow_path):
            add_temp_workflow_path(tmp_path)
            # Now tmp_path is in sys.path for this test
    """
    original_sys_path = list(sys.path)
    added_paths = []
    
    def _add_path(path):
        path_str = str(Path(path).resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            added_paths.append(path_str)
    
    yield _add_path
    
    # Restore original sys.path after test completes
    for path in added_paths:
        if path in sys.path:
            sys.path.remove(path)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might take longer
        if "mcp_server" in item.nodeid:
            item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    # Every test that drives a real `train_workflow` carries both markers, so
    # `-m "not slow"` deselects the full training runs (minutes each) and
    # `-m "not requires_llm_key"` deselects everything that needs a real
    # LITELLM_API_KEY_* to do anything but skip. Before these existed there was no way
    # to run the suite without the ~7 full trains. bd fix-k0i.42.
    config.addinivalue_line(
        "markers",
        "requires_llm_key: mark test as needing a real LLM API key to do more than skip",
    )


@pytest.fixture(autouse=True, scope="function")
def cleanup_background_threads():
    """Ensure background ChatWorker threads finish before the next test.

    Only waits when a ChatWorker is actually alive. A blanket 0.5s sleep on
    every test cost ~13 minutes on a ~1500-test suite; only a handful of
    tests start ChatWorker threads.
    """
    yield
    workers = [
        t
        for t in threading.enumerate()
        if type(t).__name__ == "ChatWorker" and t.is_alive()
    ]
    if not workers:
        return
    # Join briefly first; fall back to a short sleep if a worker is stubborn.
    for t in workers:
        t.join(timeout=0.5)
    if any(t.is_alive() for t in workers):
        time.sleep(0.5)
