"""The signals that suspend a turn's agent and the error of a resume with nothing to resume.

They belong to the turn engine, which catches and raises them; the ReAct loop raises ``AskUserSuspend`` and imports it
from here. Standard library only.
"""


class AskUserSuspend(BaseException):
    """
    Raised by ask_user when no user_message_queue is configured (Topology B).

    Subclasses BaseException so fastWorkflowReAct's ``except Exception`` does not
    swallow it; the loop catches this explicitly and returns a suspended sentinel.
    """

    def __init__(self, clarification_request: str):
        self.clarification_request = clarification_request
        super().__init__(clarification_request)


class NoSuspendedAgentStateError(RuntimeError):
    """Resume requested but no suspended ReAct trajectory exists.

    Happens when ``_awaiting_user`` is set after the trajectory was already
    consumed (e.g. a deferred resume still in flight that later failed, or a
    restored blob that lost ``react``) so a second message cannot honestly
    continue the turn. Embedders map this to HTTP 409 Conflict — not 500.
    """
