import os
from typing import Optional

#: A workflow's two env files, in its folder, when none is named.
ENV_FILENAME = "fastworkflow.env"
PASSWORDS_FILENAME = "fastworkflow.passwords.env"


def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """
    Helper function to get the environment variable or raise exception.
    Used inside the container applications. DO NOT REMOVE!
    """
    try:
        return os.environ[var_name]
    except KeyError as exc:
        if default is not None:
            return default
        error_msg = f"The environment variable {var_name} was missing, abort..."

        # dynamically import logger
        try:
            from fastworkflow.utils.logging import logger
        except ImportError:
            import logging

            logger = logging.getLogger("fastWorkflow")
        logger.critical("%s", error_msg)

        raise EnvironmentError(error_msg) from exc


def read_env_files(env_file_path, passwords_file_path) -> dict:
    """The settings of a workflow's env file and passwords file, the passwords file winning on a name both set: the
    dict ``fastworkflow.init`` takes (``fastworkflow train``, ``run``, ``build`` and ``refine`` load it the same way)."""
    from dotenv import dotenv_values

    return {**dotenv_values(env_file_path), **dotenv_values(passwords_file_path)}


def init_from_env_files(workflow_folderpath: str, env_file_path: Optional[str] = None,
                        passwords_file_path: Optional[str] = None) -> dict:
    """``fastworkflow.init`` with the workflow's env files, for a command that may run before the workflow has them
    (``fastworkflow build`` and ``refine``): a file named here must exist (``FileNotFoundError`` otherwise); with
    none named, ``fastworkflow.env`` and ``fastworkflow.passwords.env`` in the workflow folder are read when present.
    Returns the loaded settings."""
    import fastworkflow

    from dotenv import dotenv_values

    env_vars: dict = {}
    for named, default_name in ((env_file_path, ENV_FILENAME), (passwords_file_path, PASSWORDS_FILENAME)):
        if named and not os.path.isfile(named):
            raise FileNotFoundError(f"env file not found: {named}")
        path = named or os.path.join(workflow_folderpath, default_name)
        # Never `dotenv_values(None)`: with no path it searches upward for a `.env` and would read that instead.
        if os.path.isfile(path):
            env_vars.update(dotenv_values(path))
    fastworkflow.init(env_vars=env_vars)
    return env_vars


def add_env_file_arguments(parser) -> None:
    """``--env-file-path`` and ``--passwords-file-path`` of ``build`` and ``refine`` (``init_from_env_files``)."""
    parser.add_argument('--env-file-path', default=None,
                        help="Path to the environment file (default: fastworkflow.env in the workflow folder, if any)")
    parser.add_argument('--passwords-file-path', default=None,
                        help="Path to the passwords file (default: fastworkflow.passwords.env in the workflow folder, "
                             "if any)")
