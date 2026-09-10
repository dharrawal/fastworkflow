"""What may be recorded about an execution, and in what form.

Architecture §6.6 (compiled `CapturePolicy`) and §6.7 (`ContextHandle`),
satisfying FW-REQ-002 clauses 2-4 and 6.

The two contracts live together because they answer the same question. A capture
policy decides what happens to a *field*; a context handle is what happens to a
*context object* — an opaque reference standing in for something the record must
identify but must not expose. §6.7's handle is the `opaque-ref` disposition
applied to context, so separating them would put one disposition in a different
module from its four siblings.

**The substrate's current posture is verbatim.** The 3.2.0 observability store
persists parameters, responses, prompts and completions in full, with a
credential-shape scrub at the sink boundary (`observability_store.Redactor`) and
nothing else. That is the right default for a single developer debugging a local
workflow against fake data, and it is not acceptable for evaluation evidence or
for any deployment touching real tenant data. Hence two profiles rather than one
global switch, per architecture §12.0 delta 3:

* `debug` — today's behavior, unchanged. Every field is captured; the credential
  scrub still runs. Chosen so that installing this module changes nothing until a
  deployment asks it to (EXP-003 is a Phase 0 slice: no behavior change).
* `evidence` — default-deny. A field nobody classified is omitted, because the
  alternative is that adding a parameter to a command silently starts recording
  it. Unclassified means unreviewed, not harmless.

The profile in effect is recorded in run provenance, so a bundle can never be
read without knowing which posture produced it.

**Redaction is never silent.** §12.0 delta 3 requires a removed field to render
as a classification badge plus digest and size, so the chatbot debug UI degrades
into "there was a 4 KiB user-text value here, digest abc123" rather than showing
nothing. An absent key and a withheld key look identical, and the second is the
one an operator needs to see, so every disposition except `omit`-of-nothing
leaves an envelope behind.

Two rules that are easy to state and easy to get wrong:

**Identifiers are never silently truncated (§6.6).** A truncated uid is not a
smaller uid, it is a different uid that still looks like one — it will join
against the wrong row, or against nothing, and no error is raised. So an oversize
identifier is omitted whole, with its size and digest, and never cut. This is why
`bounded-text` refuses the `identifier` classification outright rather than
trusting each policy author to notice.

**A digest of an identifier still joins.** Digesting is not the same as dropping:
the same uid always digests to the same value, so evidence stays correlatable
across records while the uid itself never lands in the bundle. That is why
`identifier` degrades to `digest` under the evidence profile instead of `omit`.

Leaf module (arch §22): standard library, Pydantic, and `runtime_manifest` only.
`DataClassification` is imported rather than restated so a classification a
workflow can declare and one this module can act on cannot drift apart.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fastworkflow.runtime_manifest import DataClassification

# Version of the policy *contract* — the field set, the disposition vocabulary,
# and the profile defaults below. Recorded on every produced envelope and in
# provenance, because a bundle whose omissions were decided by different rules is
# not comparable with one that came before it.
CAPTURE_POLICY_VERSION = "1"

CaptureProfile = Literal["debug", "evidence"]

CaptureDisposition = Literal["omit", "digest", "bounded-text", "opaque-ref"]

# §12.4's lifecycle classes. Carried per field so that reclamation can exclude
# protected records without re-deriving why they were protected.
RetentionClass = Literal[
    "active-task",
    "reconciliation",
    "baseline",
    "diagnostic",
    "tombstone",
]

# Marks a value the policy acted on. Prefixed and dunder-ish for the same reason
# as `__fw_artifact_ref__` in `observability_store.serialize_turn_result`: a
# reader needs to distinguish a policy envelope from workflow data that happens
# to be a dict with a `size` key.
CAPTURE_ENVELOPE_MARKER = "__fw_capture__"

# What each disposition does when the classification is not overridden by an
# explicit policy. `user-text` is the free-text category and therefore the one
# that can contain anything, including an entity's name, an address, or an
# injected instruction; it is dropped under the evidence profile rather than
# bounded, because a bounded prefix of arbitrary text is still arbitrary text.
_EVIDENCE_DEFAULTS: dict[DataClassification, CaptureDisposition] = {
    # Digest, not omit: the same uid digests identically, so joins survive.
    "identifier": "digest",
    # Nobody can say what is inside an opaque payload, which is the whole
    # meaning of the classification.
    "opaque-payload": "omit",
    # A closed vocabulary carries no entity content by construction.
    "controlled-vocabulary": "bounded-text",
    "user-text": "omit",
}

# What an unclassified field gets under the evidence profile. This single value is
# the whole default-deny decision.
_EVIDENCE_UNCLASSIFIED_DISPOSITION: CaptureDisposition = "omit"

# The debug profile has no defaults at all — see `default_disposition`.
_PROFILE_DEFAULTS: dict[CaptureProfile, dict[DataClassification, CaptureDisposition]] = {
    "debug": {},
    "evidence": _EVIDENCE_DEFAULTS,
}

# Generous, because bounding is not the safety mechanism here — classification
# is. A cap exists so one pathological value cannot dominate a trace row, and it
# matches `tracing._DEFAULT_MAX_ATTR_BYTES` so the two layers do not disagree
# about what "too big" means.
DEFAULT_MAX_BYTES = 16384

# Small, because a controlled vocabulary that needs more than this is not a
# controlled vocabulary.
_VOCABULARY_MAX_BYTES = 256

HMAC_KEY_VAR = "FW_CAPTURE_HMAC_KEY"


def _digest_bytes(payload: bytes, length: int = 16) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()[:length]}"


def _path_matches(pattern: str, field_path: str) -> bool:
    """Whether a dotted field path matches a pattern with `*` segments.

    A trailing `*` is greedy over the remaining segments (`p.*` covers
    `p.a` and `p.a.b`) because a policy author writing it means "everything under
    here"; an interior `*` matches exactly one segment, so `command.*.parameters`
    cannot accidentally reach into a nested structure the author did not name.
    """
    pattern_segments = pattern.split(".")
    path_segments = field_path.split(".")

    if pattern_segments and pattern_segments[-1] == "*":
        head = pattern_segments[:-1]
        if len(path_segments) <= len(head):
            return False
        path_segments = path_segments[: len(head)]
        pattern_segments = head

    if len(pattern_segments) != len(path_segments):
        return False
    return all(
        expected in ("*", actual)
        for expected, actual in zip(pattern_segments, path_segments)
    )


def _encode(value: Any) -> bytes:
    """Serialize a value the same way the sink will, so sizes agree.

    `observability_store` measures artifacts with `json.dumps(..., ensure_ascii=
    False).encode("utf-8")`; measuring differently here would let a value pass
    this policy's cap and then blow the sink's.
    """
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")


class CaptureFieldPolicy(BaseModel):
    """What happens to one field. Architecture §6.6's field list, unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_path: str
    classification: DataClassification
    disposition: CaptureDisposition
    max_bytes: int = Field(default=DEFAULT_MAX_BYTES, gt=0)
    max_tokens: Optional[int] = None
    retention_class: RetentionClass = "diagnostic"
    redact_before_prompt: bool = True
    redact_before_trace: bool = False

    @model_validator(mode="after")
    def _identifiers_are_not_truncatable(self) -> "CaptureFieldPolicy":
        """§6.6: identifiers are never silently truncated.

        Refused at policy-compile time rather than at capture time, because a
        policy that only misbehaves on an oversize value is a policy that passes
        every test until production. A truncated identifier is indistinguishable
        from a real one, so there is no safe way to honor this combination.
        """
        if self.classification == "identifier" and self.disposition == "bounded-text":
            raise ValueError(
                f"field '{self.field_path}' is an identifier and cannot use "
                "'bounded-text': a truncated identifier still looks like an "
                "identifier and will join against the wrong row. Use 'digest' to "
                "keep it correlatable, or 'omit' to drop it."
            )
        return self


class CapturedValue(BaseModel):
    """What the policy left behind in place of a value it acted on.

    Never merely absent: §12.0 delta 3 requires a viewer to be able to say "a
    value was here and was withheld, this is what class it was and how big" so
    that a redacted trace degrades instead of going dark.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    classification: Optional[DataClassification]
    disposition: CaptureDisposition
    original_bytes: int
    digest: str
    reason: str
    policy_version: str = CAPTURE_POLICY_VERSION
    retention_class: RetentionClass = "diagnostic"
    # Present only for `bounded-text` that actually had to cut. A reader can tell
    # a complete short value (no envelope at all) from a cut one (envelope with a
    # prefix) from a withheld one (envelope, no prefix).
    prefix: Optional[str] = None

    def to_envelope(self) -> dict[str, Any]:
        """The serialized form the sink persists."""
        envelope = {
            CAPTURE_ENVELOPE_MARKER: True,
            "classification": self.classification,
            "disposition": self.disposition,
            "original_bytes": self.original_bytes,
            "digest": self.digest,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "retention_class": self.retention_class,
        }
        if self.prefix is not None:
            envelope["prefix"] = self.prefix
        return envelope


class CapturePolicy(BaseModel):
    """A compiled set of field policies plus the profile that fills the gaps.

    "Compiled" means the field paths are resolved into an exact-match table and a
    wildcard table once, at construction, rather than re-scanned per field per
    turn — capture runs on every command of every turn and its overhead is an
    EXP-003 stop condition (FW-NFR-005).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: CaptureProfile
    field_policies: tuple[CaptureFieldPolicy, ...] = ()
    policy_version: str = CAPTURE_POLICY_VERSION

    @model_validator(mode="after")
    def _paths_are_unique(self) -> "CapturePolicy":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for policy in self.field_policies:
            if policy.field_path in seen:
                duplicates.add(policy.field_path)
            seen.add(policy.field_path)
        if duplicates:
            raise ValueError(
                f"duplicate field_path in capture policy: {sorted(duplicates)}; two "
                "policies for one path means the effective one depends on "
                "declaration order"
            )
        return self

    def policy_for(self, field_path: str) -> Optional[CaptureFieldPolicy]:
        """The most specific declared policy for this path, or None.

        Paths are dot-separated. A `*` segment matches exactly one segment, and a
        trailing `*` matches all remaining segments, so a deployment can write
        `command.*.parameters.*` for "every parameter of every command" and then
        override one field with an exact path.

        "Most specific" is the count of literal segments, which makes
        `command.get_user.parameters.*` beat `command.*.parameters.*` beat
        `command.*`. An exact match always wins. Ties cannot happen: two patterns
        with the same literal segments in the same positions are the same
        pattern, and `_paths_are_unique` rejects those.
        """
        best: Optional[CaptureFieldPolicy] = None
        best_specificity = -1
        for policy in self.field_policies:
            if policy.field_path == field_path:
                return policy
            if not _path_matches(policy.field_path, field_path):
                continue
            segments = policy.field_path.split(".")
            specificity = len(segments) - segments.count("*")
            if specificity > best_specificity:
                best, best_specificity = policy, specificity
        return best

    def default_disposition(
        self, classification: Optional[DataClassification]
    ) -> Optional[CaptureDisposition]:
        """What an undeclared field gets, or None for "nothing happens to it".

        A `classification` of None is the default-deny case: under the evidence
        profile a field nobody classified is omitted, because unclassified means
        unreviewed.

        **The debug profile has no defaults, and that is not the same as having
        lenient ones.** An earlier version of this module defaulted it to
        `bounded-text`, which looked equivalent and was not: capture runs before
        the artifact-offload pass in `observability_store.serialize_turn_result`,
        so a 520 KB artifact was bounded to 16 KB here and then never offloaded —
        it fell under the inline threshold. The result was a silently truncated
        artifact under the profile whose entire purpose is to change nothing.
        Measured, not hypothesized; it is why this returns None instead.
        """
        if self.profile == "debug":
            return None
        if classification is None:
            return _EVIDENCE_UNCLASSIFIED_DISPOSITION
        return _PROFILE_DEFAULTS[self.profile][classification]

    def apply(
        self,
        field_path: str,
        value: Any,
        *,
        classification: Optional[DataClassification] = None,
        for_prompt: bool = False,
    ) -> Any:
        """Return what may be recorded in place of `value`.

        `classification` is the workflow's declaration, normally from
        `RuntimeMetadata.capture_classification()`. A declared policy for the
        path overrides both it and the profile default.

        `for_prompt=True` selects `redact_before_prompt` instead of
        `redact_before_trace`, which is how one policy serves both the trace sink
        and the P1 agent-exposure path of FW-REQ-002B without a second table.
        """
        if value is None:
            return None

        declared = self.policy_for(field_path)
        if declared is not None:
            classification = declared.classification
            disposition = declared.disposition
            max_bytes = declared.max_bytes
            retention_class = declared.retention_class
            gated = declared.redact_before_prompt if for_prompt else declared.redact_before_trace
            # An explicit policy that does not redact for this sink keeps the
            # value whole, whatever its disposition would otherwise have been.
            if not gated:
                return value
        else:
            default = self.default_disposition(classification)
            if default is None:
                # No policy applies. The debug profile's whole contract.
                return value
            disposition = default
            max_bytes = (
                _VOCABULARY_MAX_BYTES
                if classification == "controlled-vocabulary"
                else DEFAULT_MAX_BYTES
            )
            retention_class = "diagnostic"

        encoded = _encode(value)
        reason_suffix = (
            "declared policy" if declared is not None else f"{self.profile} profile default"
        )

        if disposition == "omit":
            return CapturedValue(
                classification=classification,
                disposition="omit",
                original_bytes=len(encoded),
                digest=_digest_bytes(encoded),
                reason=f"omitted by {reason_suffix}",
                retention_class=retention_class,
            ).to_envelope()

        if disposition == "digest":
            return CapturedValue(
                classification=classification,
                disposition="digest",
                original_bytes=len(encoded),
                digest=_digest_bytes(encoded),
                reason=f"digested by {reason_suffix}",
                retention_class=retention_class,
            ).to_envelope()

        if disposition == "opaque-ref":
            # A reference is only meaningful if something can resolve it, and P0
            # introduces no dereference (§6.7). Producing a handle here would
            # invent a projector; the honest answer is the envelope, which still
            # carries size and digest.
            return CapturedValue(
                classification=classification,
                disposition="opaque-ref",
                original_bytes=len(encoded),
                digest=_digest_bytes(encoded),
                reason=(
                    f"opaque-ref by {reason_suffix}; P0 defines no dereference "
                    "(arch §6.7), so no handle is projected here"
                ),
                retention_class=retention_class,
            ).to_envelope()

        # bounded-text. Under the cap the value passes through untouched, which
        # is what keeps the debug profile identical to today's behavior.
        if len(encoded) <= max_bytes:
            return value

        # No identifier can reach here. §6.6's "never silently truncated" rule is
        # enforced at the two — and only two — places an identifier can acquire
        # `bounded-text`: `CaptureFieldPolicy` refuses it outright, and no profile
        # default maps `identifier` to it (asserted by test). Keeping a guard here
        # as well would be an untestable branch, which is false comfort rather
        # than defense.

        # Cut on a character boundary, not a byte one: slicing UTF-8 bytes can
        # land mid-codepoint and produce a value that will not decode.
        prefix = encoded[:max_bytes].decode("utf-8", errors="ignore")
        return CapturedValue(
            classification=classification,
            disposition="bounded-text",
            original_bytes=len(encoded),
            digest=_digest_bytes(encoded),
            reason=f"bounded to {max_bytes} bytes by {reason_suffix}",
            retention_class=retention_class,
            prefix=prefix,
        ).to_envelope()


class CaptureProfileError(ValueError):
    """A configured capture profile name is not one this engine knows."""


def debug_policy() -> CapturePolicy:
    """Today's behavior: capture everything, credential scrub still applies."""
    return CapturePolicy(profile="debug")


def evidence_policy(
    field_policies: tuple[CaptureFieldPolicy, ...] = (),
) -> CapturePolicy:
    """Default-deny, for evaluation runs and any deployment with tenant data."""
    return CapturePolicy(profile="evidence", field_policies=field_policies)


def policy_for_profile(
    name: str, field_policies: tuple[CaptureFieldPolicy, ...] = ()
) -> CapturePolicy:
    """Resolve a configured profile name, raising on anything unrecognized.

    **Deliberately not tolerant.** Falling back to `debug` on an unknown name
    would mean an operator who typed `evidnce` gets verbatim capture of tenant
    data and no indication that the profile they asked for is not in effect —
    the failure would be silent, permanent, and discovered by reading a bundle
    afterwards. Refusing to start is the recoverable direction.
    """
    known = get_args(CaptureProfile)
    if name not in known:
        raise CaptureProfileError(
            f"unknown capture profile '{name}'; expected one of {list(known)}. "
            "Refusing to fall back to a default: a typo here would silently "
            "capture tenant data verbatim."
        )
    return CapturePolicy(profile=name, field_policies=field_policies)


def is_capture_envelope(value: Any) -> bool:
    """Whether a persisted value is a policy envelope rather than workflow data."""
    return isinstance(value, dict) and value.get(CAPTURE_ENVELOPE_MARKER) is True


# ----------------------------------------------------------------------
# Context handles (arch §6.7)
# ----------------------------------------------------------------------


class ContextHandle(BaseModel):
    """An opaque stand-in for a context instance. Architecture §6.7.

    `handle_id` never authorizes access, and P0 defines no dereference — a handle
    exists so that two records can be known to concern the same context instance
    without either recording what that instance is.

    A handle whose `instance_fingerprint` is None is **type-only**: it says which
    context type was active and nothing about which instance. §6.7 permits that
    only as feature-off legacy behavior, and such a handle cannot contribute to
    G2A/G2B, so `concrete` is checked rather than assumed by anything reporting
    against those gates.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    handle_id: str
    context_type: str
    security_scope_ref: str
    # None means type-only; see the class docstring.
    instance_fingerprint: Optional[str] = None
    hmac_key_version: Optional[str] = None
    projector_id: str
    projector_version: str
    # §6.7: present only under an explicit allowlist, because a label is the one
    # field on this contract that can carry entity content.
    display_label: Optional[str] = None
    issued_at: datetime
    expires_at: Optional[datetime] = None

    @property
    def concrete(self) -> bool:
        """Whether this handle identifies an instance rather than only a type."""
        return self.instance_fingerprint is not None

    @model_validator(mode="after")
    def _fingerprint_carries_its_key_version(self) -> "ContextHandle":
        """A fingerprint without a key version cannot survive key rotation.

        The pair is what makes the value verifiable later: after a rotation, a
        fingerprint whose key nobody recorded is not comparable with anything,
        and its silence looks exactly like a match failure.
        """
        if (self.instance_fingerprint is None) != (self.hmac_key_version is None):
            raise ValueError(
                "instance_fingerprint and hmac_key_version are recorded together "
                "or not at all; a fingerprint whose key version is unknown cannot "
                "be compared across a rotation"
            )
        return self


def _hmac_key(env: Optional[Mapping[str, str]] = None) -> Optional[bytes]:
    source = env if env is not None else os.environ
    key = source.get(HMAC_KEY_VAR)
    return key.encode("utf-8") if key else None


def project_context_handle(
    *,
    context_type: str,
    instance_key: Optional[str],
    security_scope_ref: str,
    projector_id: str,
    projector_version: str,
    display_label: Optional[str] = None,
    hmac_key_version: str = "1",
    env: Optional[Mapping[str, str]] = None,
    issued_at: Optional[datetime] = None,
) -> ContextHandle:
    """Project a handle for a context instance.

    Keyed rather than plain-hashed on purpose. A context instance key is usually a
    low-entropy identifier — `sara_doe_496`, an order number — and an unkeyed
    SHA-256 of one of those is reversible by anyone who can enumerate the space,
    which is everyone. An HMAC under a key that does not leave the deployment is
    not, so the fingerprint stays joinable without becoming the identifier again.

    Degrades to a **type-only** handle, rather than failing or falling back to an
    unkeyed hash, when no key is configured or no instance key is available. That
    is §6.7's feature-off legacy behavior; the caller decides whether it is
    admissible, which is what `concrete` is for.
    """
    key = _hmac_key(env)
    fingerprint: Optional[str] = None
    key_version: Optional[str] = None
    if key and instance_key:
        digest = hmac.new(key, instance_key.encode("utf-8"), hashlib.sha256).hexdigest()
        fingerprint = f"hmac-sha256:{digest[:32]}"
        key_version = hmac_key_version

    # Derived from the fingerprint when there is one, so the same instance yields
    # the same handle id across records and processes. Random ids would make a
    # handle unjoinable, which is the only thing P0 uses handles for.
    basis = fingerprint or f"type-only:{context_type}"
    handle_id = _digest_bytes(f"{projector_id}|{basis}".encode("utf-8"), length=24)

    return ContextHandle(
        handle_id=handle_id,
        context_type=context_type,
        security_scope_ref=security_scope_ref,
        instance_fingerprint=fingerprint,
        hmac_key_version=key_version,
        projector_id=projector_id,
        projector_version=projector_version,
        display_label=display_label,
        issued_at=issued_at or datetime.now(timezone.utc),
    )
