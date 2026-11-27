"""Wording the library's planning commands share.

Imported by the Library commands as `from .._shared.wording import copies`: _commands/_shared/ is the
workflow's helper package, which is never read as commands.
"""
from fastworkflow.planning import count


def copies(number):
    """How many copies the library owns, in words: "1 copy", "3 copies"."""
    return count(number, "copy", "copies")
