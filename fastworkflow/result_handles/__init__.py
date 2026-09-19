"""Public result-handle API.

Implementation is split by responsibility; this module preserves the released
``from fastworkflow import result_handles`` surface.
"""
from fastworkflow.result_handles.paging import *  # noqa: F401,F403
from fastworkflow.result_handles.paging import __all__
