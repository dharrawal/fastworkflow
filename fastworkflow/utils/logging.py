""" utility functions """

import logging
import os
from datetime import datetime, timezone
from time import time_ns


def format_ns(time_in_ns):
    """convert nanoseconds to text format"""
    formatted_time_upto_seconds = datetime.fromtimestamp(
        time_in_ns / 1e9, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S")

    fractional_sec = (time_in_ns // 10**9) * 10**9
    nanoseconds_string = f"{fractional_sec}"[:9]

    return f"{formatted_time_upto_seconds}.{nanoseconds_string}Z"


# For time in nanoseconds
# https://stackoverflow.com/questions/31328300/python-logging-module-logging-timestamp-to-include-microsecond
class LogRecordNs(logging.LogRecord):  # pylint: disable=too-few-public-methods
    """class that returns nanoseconds"""

    def __init__(self, *args, **kwargs):
        self.created_ns = time_ns()  # Fetch precise timestamp
        super().__init__(*args, **kwargs)


logging.basicConfig(level=logging.NOTSET)


class FormatterNs(logging.Formatter):
    """nanosecond log formatter"""

    default_nsec_format = "%Y-%m-%dT%H:%M:%S.%09dZ"

    def formatTime(self, record, datefmt=None):
        if datefmt is not None:  # Do not handle custom formats here ...
            return super().formatTime(
                record, datefmt
            )  # ... leave to original implementation
        return format_ns(record.created_ns)


logging.setLogRecordFactory(LogRecordNs)

LOG_FORMAT = "%(levelname)s:     %(message)s - %(asctime)s - %(filename)s-%(funcName)s"
log_formatter = FormatterNs(LOG_FORMAT)

#: The values ``LOG_LEVEL`` takes, in any case.
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
DEFAULT_LOG_LEVEL = logging.INFO

#: A bad ``LOG_LEVEL`` seen before the logger exists, warned about once it does.
_invalid_log_level_at_import: str | None = None


def parse_log_level(text) -> int | None:
    """The one reader of ``LOG_LEVEL``: one of ``LOG_LEVELS`` in any case and spacing, else None (a bad value)."""
    return LOG_LEVELS.get(str(text or "").strip().upper())


# Read when this module is imported, which is before ``fastworkflow.init`` loads the env files, so only the process
# environment can set it here; ``init`` applies it again under the one precedence rule (``fastworkflow.get_env_var``:
# the process, then the env files), and ``reconfigure_log_level`` is the same reader.
_log_level_text = os.environ.get("LOG_LEVEL", "")
LOG_LEVEL = parse_log_level(_log_level_text) if _log_level_text.strip() else DEFAULT_LOG_LEVEL
if LOG_LEVEL is None:
    _invalid_log_level_at_import = _log_level_text
    LOG_LEVEL = DEFAULT_LOG_LEVEL

# create logger
logger = logging.getLogger("fastWorkflow")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # otherwise you will see duplicate log entries

# # clear any existing handlers for our logger
logger.handlers.clear()

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)

# create and add formatter to ch
ch.setFormatter(log_formatter)

# add ch to logger
logger.addHandler(ch)

if _invalid_log_level_at_import is not None:
    logger.warning(
        f"Invalid LOG_LEVEL {_invalid_log_level_at_import!r}, must be one of {list(LOG_LEVELS)} (any case); "
        f"using {logging.getLevelName(DEFAULT_LOG_LEVEL)}.")


# create a separate logger for pytest_logger to log assertions
# with a slightly different format
pytest_assertion_logger = logging.getLogger("PyTest_Logger")
pytest_assertion_logger.setLevel(logging.DEBUG)
pytest_assertion_logger.propagate = (
    False  # otherwise you will see duplicate log entries
)
pytest_assertion_logger.handlers.clear()
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(FormatterNs("%(levelname)s:     %(message)s - %(asctime)s"))
pytest_assertion_logger.addHandler(ch)

logging.getLogger("dspy").setLevel(logging.ERROR)
logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("speedict").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def reconfigure_log_level(log_level_str: str) -> None:
    """Set the fastWorkflow logger's level from a ``LOG_LEVEL`` value after import (``fastworkflow.init`` calls it once
    the env files are loaded). Read by ``parse_log_level``; a bad value logs a warning and keeps the current level."""
    global LOG_LEVEL

    new_level = parse_log_level(log_level_str)
    if new_level is None:
        logger.warning(
            f"Invalid LOG_LEVEL {log_level_str!r}, must be one of {list(LOG_LEVELS)} (any case). Keeping current level."
        )
        return

    # Only reconfigure if the level is actually different
    if LOG_LEVEL == new_level:
        return

    LOG_LEVEL = new_level
    logger.setLevel(LOG_LEVEL)

    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(LOG_LEVEL)

    logger.info(f"Log level reconfigured to {logging.getLevelName(new_level)}")


# some testing code
if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")
    logger.critical("critical message")

# To disable __debug__ and set the log level to INFO, use the -O option as shown below
# python3 -O utils.py
