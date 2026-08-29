"""fix-ajv.18: a failed command's diagnostics must not inherit its success policy.

`fix-ajv.16` started naming commands that fail after routing. That was right — 130
failures in one collection had been unattributable — but it moved their records from
`command.unknown.*` to `command.<name>.*`, and `CapturePolicy.apply` returns a value
WHOLE when a declared policy is not gated for this sink. So a rule written about a
command's normal response ("X's output is benign, keep it") silently began releasing
X's *failure* text too: an exception repr, an error message, a traceback.

The direction is what makes it worth a test. A bug that withholds too much announces
itself; this one runs toward less withholding, produces a clean-looking record, and
only bites a deployment that has declared per-command rules — which is the one that
cared enough to declare them.
"""

from fastworkflow.capture_policy import (
    CaptureFieldPolicy,
    evidence_policy,
    is_capture_envelope,
)
from fastworkflow.observability_store import _apply_capture_policy


def _record(command_name, *, success, response="the response", artifacts=None,
            parameters=None):
    return {
        "turn_output": {
            "command_outputs": [
                {
                    "command_name": command_name,
                    "command_parameters": parameters,
                    "command_response": {
                        "response": response,
                        "success": success,
                        "artifacts": artifacts if artifacts is not None else {},
                    },
                }
            ]
        }
    }


def _output(record):
    return record["turn_output"]["command_outputs"][0]


def _keep_whole(field_path):
    """A declared policy that does NOT redact for the trace sink.

    Note the disposition says `omit` and the value still comes back WHOLE: `apply`
    consults `redact_before_trace` first and returns early, so an ungated policy
    never reaches its own disposition. That is the short-circuit this bug rides on,
    so the helper states it rather than hiding it behind a friendlier disposition.
    """
    return CaptureFieldPolicy(
        field_path=field_path,
        classification="user-text",
        disposition="omit",
        redact_before_trace=False,
    )


def test_success_response_stays_whole_under_its_declared_policy():
    """The rule must keep doing its job — otherwise the fix is just breakage."""
    policy = evidence_policy((_keep_whole("command.Ctrl/open.response"),))
    record = _record("Ctrl/open", success=True, response="finding 12 opened")

    _apply_capture_policy(record, policy)

    assert _output(record)["command_response"]["response"] == "finding 12 opened"


def test_failure_response_is_withheld_despite_that_same_policy():
    """The defect: one rule about success text also released failure text."""
    policy = evidence_policy((_keep_whole("command.Ctrl/open.response"),))
    record = _record(
        "Ctrl/open",
        success=False,
        response="Execution error: KeyError('ssn-441-88-2019')",
    )

    _apply_capture_policy(record, policy)

    assert is_capture_envelope(_output(record)["command_response"]["response"])


def test_failure_artifacts_are_withheld_despite_that_same_policy():
    """artifacts.error_message and .traceback are the larger of the two leaks."""
    policy = evidence_policy((_keep_whole("command.Ctrl/open.artifacts.*"),))
    record = _record(
        "Ctrl/open",
        success=False,
        artifacts={"error_message": "no such identity: 441-88-2019"},
    )

    _apply_capture_policy(record, policy)

    assert is_capture_envelope(
        _output(record)["command_response"]["artifacts"]["error_message"]
    )


def test_a_wildcard_over_commands_does_not_reach_failures_either():
    """`command.*.response` is 3 segments; the error path is 4, so it cannot match.

    Pinned because the fix depends on that arithmetic rather than on a check.
    """
    policy = evidence_policy((_keep_whole("command.*.response"),))
    record = _record("Ctrl/open", success=False, response="Execution error: boom")

    _apply_capture_policy(record, policy)

    assert is_capture_envelope(_output(record)["command_response"]["response"])


def test_error_text_can_still_be_released_by_declaring_it():
    """Opt-in must remain possible: the fix moves the path, it does not forbid it."""
    policy = evidence_policy((_keep_whole("command.Ctrl/open.error.response"),))
    record = _record("Ctrl/open", success=False, response="Execution error: boom")

    _apply_capture_policy(record, policy)

    assert _output(record)["command_response"]["response"] == "Execution error: boom"


def test_ask_user_is_not_a_failure_and_keeps_the_ordinary_path():
    """[A7] role inversion: success=False means UNANSWERED, not failed.

    An ask_user response is the user's answer — ordinary user text. Routing it
    through the error namespace would make a declared ask_user rule stop matching
    the very field it was written for.
    """
    policy = evidence_policy((_keep_whole("command.ask_user.response"),))
    record = _record("ask_user", success=False, response="my order is W123")

    _apply_capture_policy(record, policy)

    assert _output(record)["command_response"]["response"] == "my order is W123"


def test_parameters_of_a_failed_command_keep_their_declared_gating():
    """The deliberate asymmetry, pinned so it is not 'tidied' into symmetry.

    A failure's parameters are the same values the success path carries, so a
    rule that gates them must keep applying. Moving them under `.error.` would
    stop the rule matching and fall through to the profile default — which under
    `debug` returns the value whole, un-gating the one field group the success
    policy is right about.
    """
    gated = CaptureFieldPolicy(
        field_path="command.Ctrl/open.parameters.*",
        classification="user-text",
        disposition="digest",
        redact_before_trace=True,
    )
    policy = evidence_policy((gated,))
    record = _record("Ctrl/open", success=False, parameters={"ssn": "441-88-2019"})

    _apply_capture_policy(record, policy)

    assert is_capture_envelope(_output(record)["command_parameters"]["ssn"])
