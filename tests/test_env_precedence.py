"""One precedence rule for every setting (bd fix-ygxo): the process environment wins over the env files ``init``
loaded, and the env files win over the code default. An empty process value counts as set. ``LOG_LEVEL`` has one
reader; ``build`` and ``refine`` load the env files like ``train``."""
from __future__ import annotations

import argparse
import logging

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.utils import logging as fw_logging

NAME = "FW_TEST_PRECEDENCE_SETTING"


@pytest.fixture
def files(monkeypatch):
    """The env-file layer, empty, with the test's variable absent from the process."""
    env: dict = {}
    monkeypatch.setattr(fastworkflow, "_env_vars", env)
    monkeypatch.delenv(NAME, raising=False)
    return env


def test_a_process_value_beats_the_env_file(files, monkeypatch):
    files[NAME] = "from-file"
    monkeypatch.setenv(NAME, "from-process")

    assert fastworkflow.get_env_var(NAME, default="from-default") == "from-process"
    assert fastworkflow.env_source(NAME) == fastworkflow.SOURCE_PROCESS


def test_an_env_file_value_beats_the_code_default(files):
    files[NAME] = "from-file"

    assert fastworkflow.get_env_var(NAME, default="from-default") == "from-file"
    assert fastworkflow.env_source(NAME) == fastworkflow.SOURCE_ENV_FILE


def test_the_code_default_applies_when_neither_layer_sets_it(files):
    assert fastworkflow.get_env_var(NAME, default="from-default") == "from-default"
    assert fastworkflow.raw_env_var(NAME) is None
    assert fastworkflow.env_source(NAME) is None


def test_an_empty_process_value_counts_as_set_and_hides_the_env_file(files, monkeypatch):
    """As ``os.environ`` has it: an exported empty value is a value. Readers that treat a blank as "use the default"
    then use the default, never the env file's value."""
    files[NAME] = "from-file"
    monkeypatch.setenv(NAME, "")

    assert fastworkflow.raw_env_var(NAME) == ""
    assert fastworkflow.get_env_var(NAME, default="from-default") == ""
    assert fastworkflow.env_source(NAME) == fastworkflow.SOURCE_PROCESS


def test_a_former_process_only_setting_now_works_from_an_env_file(files, monkeypatch):
    from fastworkflow.intent import intent_detector

    monkeypatch.delenv("FW_REQUIRE_HEADS", raising=False)
    assert intent_detector.heads_required() is False
    files["FW_REQUIRE_HEADS"] = "1"
    assert intent_detector.heads_required() is True


def test_the_server_resolvers_follow_the_rule_and_name_the_layer(files, monkeypatch):
    from fastworkflow.run_fastapi_mcp.utils import resolve_max_live_sessions

    monkeypatch.delenv("MAX_LIVE_SESSIONS", raising=False)
    files["MAX_LIVE_SESSIONS"] = "7"
    assert resolve_max_live_sessions() == (7, fastworkflow.SOURCE_ENV_FILE)
    monkeypatch.setenv("MAX_LIVE_SESSIONS", "9")
    assert resolve_max_live_sessions() == (9, fastworkflow.SOURCE_PROCESS)


# ---------------------------------------------------------------- LOG_LEVEL: one reader


def test_log_level_is_read_in_any_case():
    assert fw_logging.parse_log_level("info") == logging.INFO
    assert fw_logging.parse_log_level(" Debug ") == logging.DEBUG
    assert fw_logging.parse_log_level("WARNING") == logging.WARNING
    assert fw_logging.parse_log_level("verbose") is None


def test_a_bad_log_level_warns_and_keeps_the_level(monkeypatch):
    seen: list[str] = []

    class Sink(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    sink = Sink(level=logging.WARNING)
    fw_logging.logger.addHandler(sink)
    before = fw_logging.LOG_LEVEL
    try:
        fw_logging.reconfigure_log_level("verbose")
    finally:
        fw_logging.logger.removeHandler(sink)
    assert fw_logging.LOG_LEVEL == before
    assert any("Invalid LOG_LEVEL 'verbose'" in message for message in seen)


def test_init_applies_log_level_from_the_env_files_under_the_rule(monkeypatch):
    before = fw_logging.LOG_LEVEL
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({"LOG_LEVEL": "error"})
        assert fw_logging.LOG_LEVEL == logging.ERROR
        monkeypatch.setenv("LOG_LEVEL", "warning")   # the process wins over the file
        fastworkflow.init({"LOG_LEVEL": "error"})
        assert fw_logging.LOG_LEVEL == logging.WARNING
    finally:
        fastworkflow.init(previous)
        fw_logging.reconfigure_log_level(logging.getLevelName(before))


# ---------------------------------------------------------------- build and refine load the env files


def _workflow_with_env_files(tmp_path, model="test/metadata-model"):
    folder = tmp_path / "wf"
    (folder / "_commands").mkdir(parents=True)
    (folder / "fastworkflow.env").write_text(f"LLM_COMMAND_METADATA_GEN={model}\n{NAME}=from-env-file\n")
    (folder / "fastworkflow.passwords.env").write_text("LITELLM_API_KEY_COMMANDMETADATA_GEN=metadata-key\n")
    return folder


def test_build_loads_the_workflow_env_files_like_train(tmp_path, monkeypatch):
    """Before, ``build`` called ``init({})``: a metadata model could be set only by a shell export."""
    from fastworkflow.build.__main__ import build_main

    monkeypatch.delenv(NAME, raising=False)
    folder = _workflow_with_env_files(tmp_path)
    args = argparse.Namespace(app_dir=str(tmp_path / "no-such-app"), workflow_folderpath=str(folder),
                              overwrite=False, stub_commands=None, no_startup=True,
                              env_file_path=None, passwords_file_path=None)
    with pytest.raises(SystemExit):   # the app folder does not exist: build stops right after loading the env
        build_main(args)

    assert fastworkflow.get_env_var(NAME) == "from-env-file"
    assert fastworkflow.get_env_var("LITELLM_API_KEY_COMMANDMETADATA_GEN") == "metadata-key"


def test_refine_loads_the_named_env_files(tmp_path, monkeypatch):
    from fastworkflow.refine.__main__ import refine_main
    from fastworkflow.utils import dspy_utils

    monkeypatch.delenv(NAME, raising=False)
    monkeypatch.delenv("LLM_COMMAND_METADATA_GEN", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY_COMMANDMETADATA_GEN", raising=False)
    folder = _workflow_with_env_files(tmp_path)
    named = tmp_path / "elsewhere.env"
    named.write_text(f"{NAME}=from-named-file\nLLM_COMMAND_METADATA_GEN=test/named-model\n")
    args = argparse.Namespace(workflow_folderpath=str(folder), env_file_path=str(named),
                              passwords_file_path=str(folder / "fastworkflow.passwords.env"))

    assert refine_main(args) == 0   # no command files: nothing to refine, no model call

    assert fastworkflow.get_env_var(NAME) == "from-named-file"
    resolved = dspy_utils.resolve_role("LLM_COMMAND_METADATA_GEN")
    assert (resolved.model, resolved.key) == ("test/named-model", "metadata-key")


def test_a_named_env_file_that_does_not_exist_is_an_error(tmp_path):
    from fastworkflow.utils.env import init_from_env_files

    with pytest.raises(FileNotFoundError, match="missing.env"):
        init_from_env_files(str(tmp_path), str(tmp_path / "missing.env"))


def test_the_cli_offers_the_env_file_options_on_build_and_refine():
    from fastworkflow import cli

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    cli.add_build_parser(subparsers)
    cli.add_refine_parser(subparsers)
    args = parser.parse_args(["refine", "-w", "wf", "--env-file-path", "a.env", "--passwords-file-path", "b.env"])
    assert (args.env_file_path, args.passwords_file_path) == ("a.env", "b.env")
    args = parser.parse_args(["build", "-s", "app", "-w", "wf"])
    assert (args.env_file_path, args.passwords_file_path) == (None, None)


# ---------------------------------------------------------------- the generation counts


def test_the_generation_counts_default_to_the_template_values(monkeypatch):
    """4, 5 and 1: the values the shipped template sets, so a workflow that relied on them keeps its cache keys."""
    from fastworkflow.train import generate_synthetic as gs

    template = dotenv_values(f"{fastworkflow.get_fastworkflow_package_path()}/examples/fastworkflow.env")
    monkeypatch.setattr(fastworkflow, "_env_vars", {})
    for name, default in gs.GENERATION_COUNT_DEFAULTS.items():
        monkeypatch.delenv(name, raising=False)
        assert gs._resolve_generation_count(None, name) == default
        assert int(template[name]) == default
    assert gs.GENERATION_COUNT_DEFAULTS == {
        "SYNTHETIC_UTTERANCE_GEN_NUMOF_PERSONAS": 4,
        "SYNTHETIC_UTTERANCE_GEN_UTTERANCES_PER_PERSONA": 5,
        "SYNTHETIC_UTTERANCE_GEN_PERSONAS_PER_BATCH": 1,
    }
