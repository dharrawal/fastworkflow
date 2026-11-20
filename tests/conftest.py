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

import litellm
import pytest

import fastworkflow
from fastworkflow import context_budget
from fastworkflow.observability import store as observability_store
from tests import plan_head, recorded_router
from tests.recorded_router import recorded_review_router  # noqa: F401  (fixture, shared by the reader tests)
# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up environment for tests
os.environ.setdefault("PYTEST_RUNNING", "1")


@pytest.fixture(autouse=True, scope="module")
def restore_fastworkflow_environment():
    """Prevent module-scoped fastworkflow.init calls leaking into later modules."""
    previous = dict(fastworkflow._env_vars)
    yield
    fastworkflow._env_vars.clear()
    fastworkflow._env_vars.update(previous)


@pytest.fixture(autouse=True, scope="function")
def isolate_state_root(tmp_path_factory):
    """Point FASTWORKFLOW_STATE_ROOT at a private temp dir for every test.

    Persistent state (conversations, suspended sessions, checkpoints, function
    caches) is rooted at FASTWORKFLOW_STATE_ROOT, defaulting to
    ~/.local/state/fastworkflow. Without this fixture a test that touches disk
    state would write into the developer's real home directory and could observe
    another test's records. Tests that need a specific root still override it via
    their own init dict (the dict wins over the OS environment).
    """
    previous = os.environ.get("FASTWORKFLOW_STATE_ROOT")
    root = tmp_path_factory.mktemp("fw_state_root")
    os.environ["FASTWORKFLOW_STATE_ROOT"] = str(root)
    try:
        yield str(root)
    finally:
        if previous is None:
            os.environ.pop("FASTWORKFLOW_STATE_ROOT", None)
        else:
            os.environ["FASTWORKFLOW_STATE_ROOT"] = previous
        # Execution contexts open their workflow's observability sink on their
        # own, so a test that never asked for one may still hold a writer in
        # this root. Close those before the root is removed.
        observability_store.close_sinks_under(str(root))
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


def pytest_addoption(parser):
    parser.addoption(
        "--record-review-router", action="store_true", default=False, dest="record_review_router",
        help="re-record tests/fixtures/review_reply_router_answers.json from the real plan head "
             "(see tests/recorded_router.py)",
    )


def pytest_report_header(config):
    """Name the plan head's current version in the session header, so a red after a retrain is attributable."""
    return plan_head.head_status_line()


def pytest_sessionfinish(session, exitstatus):
    recorded_router.write_recording_if_requested(session.config)


def _head_marker_and_absent(item) -> bool:
    return item.get_closest_marker("requires_plan_head") is not None and not plan_head.head_trained()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """``requires_plan_head``: skip, before any fixture, when the local head is absent.

    With ``FW_REQUIRE_HEADS=1`` the test is not skipped: ``pytest_runtest_call`` fails it, so it reports FAILED
    (not an error in setup) and its message names what is missing.
    """
    if _head_marker_and_absent(item) and not plan_head.heads_required():
        pytest.skip(plan_head.missing_head_message())
    if _needs_real_key_and_has_none(item):
        pytest.skip("requires_llm_key: no real LITELLM_API_KEY_* (or bedrock model) is configured")


SERVER_LAYER_FIXTURES = ("env_files", "app_module")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A server-layer test (one that asks for ``env_files`` or ``app_module``) may skip only because it needs a live key.

    ``env_files`` is written from the tracked template and never skips, so a skip there means a silent hole in the
    suite (a missing env file, a missing workflow). It is reported as a failure naming the skip reason. A test that
    carries ``requires_llm_key`` is the one allowed skip. Only a skip while the test is being SET UP counts: that is
    where a missing env file or workflow shows (the fixtures), while a skip in the test body is the test's own
    declared condition (the jsdom DOM checks skip there unless ``TEST_JSDOM_ROOT`` is set, and have their own gate,
    ``python -m tests.browser_validation``).
    """
    outcome = yield
    report = outcome.get_result()
    if (
        report.skipped
        and report.when == "setup"
        and not hasattr(report, "wasxfail")
        and item.get_closest_marker("requires_llm_key") is None
        and any(name in item.fixturenames for name in SERVER_LAYER_FIXTURES)
    ):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = (
            f"server-layer test skipped without the requires_llm_key marker, which is a failure "
            f"(it must run everywhere): {reason}"
        )


def _needs_real_key_and_has_none(item) -> bool:
    from tests import llm_credentials

    return item.get_closest_marker("requires_llm_key") is not None and not llm_credentials.real_llm_credentials_available()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    if _head_marker_and_absent(item) and plan_head.heads_required():
        pytest.fail(
            f"{plan_head.REQUIRE_HEADS_ENV} is set and {plan_head.missing_head_message()}",
            pytrace=False,
        )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_plan_head: asserts the built-in plan ProposalReview intent head's own quality; "
        "skips when the local (gitignored) head is not trained, fails instead when FW_REQUIRE_HEADS=1",
    )
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


@pytest.fixture
def env_files(tmp_path, request):
    """``(env_file, passwords_file)`` for server-layer tests, written from the tracked template.

    Same on every machine: never reads ``env/.env`` or ``passwords/.env``, holds placeholder keys only (read as no key
    by ``llm_credentials``) and no ``JEV_API_KEY``, and sets every role's model to the template's default. Nothing a
    test does with it reaches a model unless the test installs a live one.
    """
    from tests.fastapi_hermetic import write_placeholder_env

    # A server-layer test reaches a model only if it carries ``requires_llm_key``: the real litellm entry points
    # refuse the call, so a code path that would send one (the topic label, a parameter extraction) fails inside the
    # app, as it would offline, instead of sending a request with a placeholder key. A test that installs its own
    # model stand-in is left alone.
    # The patch has its own ``MonkeyPatch`` (not the test's), so a test's own ``monkeypatch`` is undone first.
    guard = pytest.MonkeyPatch()
    if request.node.get_closest_marker("requires_llm_key") is None:
        def refuse(*args, **kwargs):
            raise AssertionError("a live model call was attempted from a test that does not carry requires_llm_key")

        for name in ("completion", "acompletion"):
            if getattr(getattr(litellm, name), "__module__", "") == "litellm.main":
                guard.setattr(litellm, name, refuse)
    try:
        yield write_placeholder_env(tmp_path / "env_files")
    finally:
        guard.undo()


@pytest.fixture
def reference_agent_window(monkeypatch):
    """Pin the agent model to the budget calibration reference.

    The budgets derive from the agent model's context window, so a test that
    asserts the reference values must not depend on whatever ``LLM_AGENT`` the
    machine (or the repo-root ``.env`` litellm loads) names. This pins it to
    ``context_budget``'s calibration model, whose window is the reference.
    """
    # ``context_budget.env_value`` reads the loaded env files before the process
    # environment, and a developer's ``LLM_AGENT``, window or budget override
    # (``FW_TRAJECTORY_MAX_BYTES`` ...) may sit in either, so none of them may
    # reach a test that asserts the reference values.
    pinned_names = {context_budget.MODEL_CONTEXT_TOKENS_ENV, context_budget.AGENT_MODEL_ENV,
                    *(spec.override_env for spec in context_budget.BUDGETS if spec.override_env)}
    for name in pinned_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fastworkflow, "_env_vars",
                        {name: value for name, value in fastworkflow._env_vars.items() if name not in pinned_names})
    monkeypatch.setenv(context_budget.AGENT_MODEL_ENV, "cerebras/gpt-oss-120b")
    context_budget.reset_cache()
    yield
    context_budget.reset_cache()


@pytest.fixture
def no_planner_key(monkeypatch):
    """A process with no optional planner (Jev) key, for tests that assert behaviour without one.

    ``fastworkflow.get_env_var`` reads the loaded env files before the process environment, so the key is removed
    from both; the planner LM is ``lru_cache``d, so its cache is dropped
    before the test (an earlier test that had a key may have filled them) and after it (so this test leaks nothing).
    When the planner becomes mandatory, the tests that use this fixture are retired.
    """
    from fastworkflow import workflow_agent
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr(fastworkflow, "_env_vars",
                        {name: value for name, value in fastworkflow._env_vars.items() if name != "JEV_API_KEY"})
    def drop_caches():
        # A test may have stubbed either with a plain function (monkeypatch, undone after this fixture), which has no cache.
        getattr(workflow_agent._jev_lm, "cache_clear", lambda: None)()

    drop_caches()
    yield
    drop_caches()


def script_matcher(monkeypatch, distributions) -> list[str]:
    """Script the remote decision model's answer as intent detection asks it, with no model call.

    Makes the matcher available (``intent_detector.matcher_available``) and answers its one call per part
    (``intent_detector._matcher_distribution``) with ``distributions[text]``: ``{option: score}`` over the commands of
    the scope path and ``none``. *distributions* may also be a function of the text. A text with no scripted answer is
    a failed call, so the local model's answer stands (as when the real call fails). Returns the list of the texts the
    matcher was asked about, in order. Narrower than ``matcher_model`` (which scripts every ``dspy.Predict`` of the
    process): only intent detection's matcher is scripted, so a turn's other model calls are whatever the test makes
    them.
    """
    from fastworkflow import intent_detector

    asked: list[str] = []

    def answer(workflow_folderpath, scope_path, utterance, exclude=frozenset()):
        asked.append(utterance)
        scores = distributions(utterance) if callable(distributions) else distributions.get(utterance)
        if scores is None:
            return {"scores": {}, "model": "scripted", "latency_ms": 0.0, "calls": 0, "error": "not scripted"}
        return {"scores": {str(k): float(v) for k, v in dict(scores).items()}, "model": "scripted",
                "latency_ms": 0.0, "calls": 1}

    monkeypatch.setattr(intent_detector, "matcher_available", lambda: True)
    monkeypatch.setattr(intent_detector, "_matcher_distribution", answer)
    return asked


def no_local_model(monkeypatch) -> None:
    """No context has a trained local intent model (the state of a workflow that has not been trained): the detection
    chain is the remote decision model alone, so a test's scripted answer (``script_matcher``) is the whole decision."""
    from fastworkflow import intent_detector, run_listing

    monkeypatch.delenv("FW_REQUIRE_HEADS", raising=False)
    monkeypatch.setattr(intent_detector, "has_head", lambda folder: False)
    monkeypatch.setattr(run_listing, "head_answers", lambda workflow, sub_task, command_info_root=None: {})


def script_plan(monkeypatch, pick, *, decided_by: str = "exact_name") -> None:
    """Script which plan a part gets (``resolution.part_intent``), for a harness that builds its planning commands in
    memory: ``pick(text)`` returns the planning command object the part's text matches, or None (the part is ad hoc).
    The picked commands are added to ``workflow_agent.load_planning_command_map`` so the run finds them by name. A
    pinned plan (``use_plan``) is still decided by the product code. *decided_by* defaults to an exact command name, the
    one decider a single planning part takes the fast path on (a local model's planning pick is checked, fix-o1lg.2)."""
    from fastworkflow import workflow_agent
    from fastworkflow._workflows.plan_metadata_extraction import resolution

    real_intent, real_map = resolution.part_intent, workflow_agent.load_planning_command_map
    picked: dict = {}

    def part_intent(workflow, text, *, pinned=None, exclude=()):
        if pinned:
            return real_intent(workflow, text, pinned=pinned, exclude=exclude)
        from fastworkflow import turn_cache

        command = turn_cache.memo(("script_plan", text), lambda: pick(text))
        if command is None or command.name in set(exclude):
            return resolution.PartIntent(resolution.NONE, decided_by=decided_by)
        picked[command.name] = command
        return resolution.PartIntent(resolution.MATCH, command.name, resolution.KIND_PLANNING,
                                     (resolution.Ranked(command.name, resolution.KIND_PLANNING, 0.9),), decided_by)

    monkeypatch.setattr(resolution, "part_intent", part_intent)
    monkeypatch.setattr(workflow_agent, "load_planning_command_map", lambda folder: {**real_map(folder), **picked})


@pytest.fixture
def scripted_matcher(monkeypatch):
    """``scripted_matcher(distributions)``: ``script_matcher`` for this test."""
    return lambda distributions: script_matcher(monkeypatch, distributions)


@pytest.fixture
def no_agent_model(monkeypatch):
    """A process with no ReAct agent model (``LLM_AGENT``), for tests that assert no-agent mode.

    ``fastworkflow.get_env_var`` reads the loaded env files before the process environment, so ``LLM_AGENT`` is removed
    from both; and the execution contexts these tests build do not ask for the mode (an embedder that scripts its own
    agent LM must not be switched into it), so the class default that asks for it is switched on for the test. Nothing
    is cached, so nothing is dropped. With the agent model set, this fixture is not used: the rest of the suite is
    that proof.
    """
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    monkeypatch.delenv("LLM_AGENT", raising=False)
    monkeypatch.setattr(fastworkflow, "_env_vars",
                        {name: value for name, value in fastworkflow._env_vars.items() if name != "LLM_AGENT"})
    monkeypatch.setattr(WorkflowExecutionContext, "detect_no_agent_mode", True)
    yield


@pytest.fixture
def restore_litellm_model_cost():
    """Undo a test's ``litellm.register_model`` when it ends.

    Registration is process-wide: it writes ``litellm.model_cost`` and the
    provider's model-name set. Both are restored in place, since other modules
    hold references to them, and the window caches that read them are cleared.
    """
    model_cost = {key: dict(value) for key, value in litellm.model_cost.items()}
    openai_models = set(litellm.open_ai_chat_completion_models)
    try:
        yield
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(model_cost)
        litellm.open_ai_chat_completion_models.clear()
        litellm.open_ai_chat_completion_models.update(openai_models)
        # Private, and absent from some litellm versions: the lowercase map
        # and model-info caches built from ``model_cost``.
        invalidate = getattr(litellm.utils, "_invalidate_model_cost_lowercase_map", None)
        if invalidate is not None:
            invalidate()
        context_budget.reset_cache()


@pytest.fixture(autouse=True, scope="function")
def cleanup_background_threads():
    """Give the ChatWorker threads a test started a moment to finish before the next test.

    Only workers started during this test are waited for, and for one second in all. A worker left idle by
    an earlier test (it waits on its queue for the life of the process) is not waited for again: joining every
    leftover worker for 0.5 s each cost 10 s per test once twenty had been left behind, and more as a run went on.
    """
    before = set(threading.enumerate())
    yield
    workers = [
        t
        for t in threading.enumerate()
        if type(t).__name__ == "ChatWorker" and t.is_alive() and t not in before
    ]
    deadline = time.monotonic() + 1.0
    for t in workers:
        t.join(timeout=max(0.0, deadline - time.monotonic()))


@pytest.fixture
def review_nouns_workflow():
    """The review reply guards read the input names of the workflow a reply is for (``review_proposals.workflow_scope``).
    Review tests that build proposals in memory from a recorded exchange use inputs such as ``unit``, ``right`` and
    ``person``; ``tests/review_nouns_workflow`` declares them."""
    from pathlib import Path
    from fastworkflow.review_proposals import workflow_scope

    folder = str(Path(__file__).parent.joinpath("review_nouns_workflow").resolve())
    with workflow_scope(folder):
        yield folder

