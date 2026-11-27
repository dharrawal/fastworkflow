"""`fastworkflow run_fastapi_mcp` forwards every public server option (fix-qavj).

The real `cli.main()` builds the server command line; the server's own `load_args` parses it. Only the
subprocess launch is stood in for. The parity guard compares the two parsers so a server option added
later cannot be left unreachable from the documented command.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from fastworkflow import cli

TODO = Path(__file__).parent / "todo_list_workflow"

# Server options the wrapper deliberately does not take under the same name.
#   workflow_path / env_file_path / passwords_file_path: positionals on the wrapper.
#   parent_pid: internal (run_chatbot's launcher), hidden with argparse.SUPPRESS.
POSITIONAL_ON_WRAPPER = {"workflow_path", "env_file_path", "passwords_file_path"}
INTERNAL = {"parent_pid"}


def _capture_server_command(tmp_path, monkeypatch, given):
    env_file = tmp_path / "e.env"
    pw_file = tmp_path / "p.env"
    env_file.write_text("")
    pw_file.write_text("")
    seen: list = []
    monkeypatch.setattr(cli, "_require_server_extra", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(cmd) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(sys, "argv", ["fastworkflow", "run_fastapi_mcp", str(TODO), str(env_file), str(pw_file), *given])
    try:
        cli.main()
    except SystemExit:
        pass
    assert len(seen) == 1
    return seen[0][3:]  # after: python -m fastworkflow.run_fastapi_mcp


def _server_args(monkeypatch, server_args):
    monkeypatch.setattr(sys, "argv", ["pytest", *server_args])
    import fastworkflow.run_fastapi_mcp.__main__ as main
    return importlib.reload(main).ARGS


@pytest.mark.parametrize("flag", ["expect_encrypted_jwt", "cors_loopback_only", "enable_admin_endpoints",
                                  "allow_request_cache_toggle"])
@pytest.mark.parametrize("present", [False, True])
def test_store_true_options_reach_the_server_and_absent_means_the_server_default(
        tmp_path, monkeypatch, flag, present):
    server_args = _capture_server_command(tmp_path, monkeypatch, [f"--{flag}"] if present else [])
    assert (f"--{flag}" in server_args) is present
    args = _server_args(monkeypatch, server_args)
    assert getattr(args, flag) is present
    # Absent: the whole parsed result equals what the server gives for the same inputs run directly.
    if not present:
        direct = _server_args(monkeypatch, server_args)
        assert vars(args) == vars(direct)


def test_all_three_together_and_the_values_are_not_swallowed_as_option_arguments(tmp_path, monkeypatch):
    given = ["--expect_encrypted_jwt", "--cors_loopback_only", "--enable_admin_endpoints", "--port", "9123"]
    args = _server_args(monkeypatch, _capture_server_command(tmp_path, monkeypatch, given))
    assert (args.expect_encrypted_jwt, args.cors_loopback_only, args.enable_admin_endpoints, args.port) == (
        True, True, True, 9123)


def _public_options(parser: argparse.ArgumentParser) -> dict:
    return {a.dest: a for a in parser._actions
            if a.dest != "help" and a.help is not argparse.SUPPRESS}


def test_every_public_server_option_is_forwardable_by_the_wrapper_with_the_same_default(monkeypatch):
    """The durable guard: a server option the wrapper cannot take (or takes with another default) fails here."""
    captured: dict = {}
    real_parse = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        captured["parser"] = self
        return real_parse(self, *a, **k)

    monkeypatch.setattr(sys, "argv", ["pytest", "--workflow_path", str(TODO)])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    import fastworkflow.run_fastapi_mcp.__main__ as main
    importlib.reload(main)
    server = _public_options(captured["parser"])

    top = argparse.ArgumentParser()
    cli.add_run_fastapi_mcp_parser(top.add_subparsers(dest="command"))
    wrapper = _public_options(top._subparsers._group_actions[0].choices["run_fastapi_mcp"])

    missing = sorted(set(server) - set(wrapper) - POSITIONAL_ON_WRAPPER - INTERNAL)
    assert not missing, f"server options the wrapper cannot forward: {missing}"
    for dest, action in server.items():
        if dest in POSITIONAL_ON_WRAPPER or dest in INTERNAL:
            continue
        w = wrapper[dest]
        assert w.default == action.default, dest
        assert type(w) is type(action), dest
        assert w.type == action.type and w.choices == action.choices, dest
    assert INTERNAL <= {a.dest for a in captured["parser"]._actions}, "stale exclusion list"
