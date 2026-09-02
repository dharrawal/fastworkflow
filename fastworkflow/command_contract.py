"""What a command does to the world, declared (arch §14.3, FW-REQ-008B).

A write is replay-unsafe unless somebody has said, in advance and in writing,
what makes a retry of it safe. That is not something the runtime can infer: the
question "would dispatching this twice apply it twice?" is a property of the
backend, and the only honest default is that nobody knows.

So the declarations here are **required to permit a write, and their absence is
a refusal rather than a permission**. `EffectKind.unknown` is not a shrug; it is
the state in which the strict write gate says no.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

# Arch §7.3. Absent or invalid reads as `unknown` — never `read_only`.
EffectKind = Literal["read_only", "write", "unknown"]

# How a repeated dispatch of the same logical call behaves at the backend.
IdempotencyGuarantee = Literal[
    # The backend deduplicates on the operation ID or a durable idempotency key
    # it accepts from us. The only guarantee that makes a same-ID retry safe.
    "backend_deduplicated",
    # Applying it twice is indistinguishable from applying it once *by nature*
    # (setting a value, not incrementing one). Weaker than deduplication: it
    # says a second apply is harmless, not that the first is recoverable.
    "naturally_idempotent",
    # Nobody has said. The default, and a refusal.
    "unknown",
]


class ReconciliationDeclaration(BaseModel):
    """How an unknown outcome for this command can be resolved (arch §14.6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str
    strategy_version: str
    # The source that can be asked what actually happened. A command whose only
    # evidence is the response we already lost has no reconciliation.
    authoritative_source: str
    # How long after the effect the source can still be trusted to answer.
    consistency_window_seconds: float = Field(gt=0)
    # Whether a receipt survives losing the response — the distinction §14.6
    # draws between a recoverable receipt and "a UUID that exists only in a
    # lost POST response".
    receipt_is_recoverable: bool = False


class CommandEffectContract(BaseModel):
    """The declared effect surface of one command definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    definition_id: str
    effect_kind: EffectKind = "unknown"
    # The adapter-declared effect key: what this command *does*, independent of
    # the parameters it does it with. Part of the logical-call key, so two
    # different effects on the same target never collide.
    effect_key: Optional[str] = None
    idempotency: IdempotencyGuarantee = "unknown"
    reconciliation: Optional[ReconciliationDeclaration] = None
    # A command declaring compensation says what undoes it; compensation is its
    # own authorized operation (arch §14.7), never an implicit rollback.
    compensating_effect_key: Optional[str] = None
    contract_version: str = "1"

    @property
    def is_read_only(self) -> bool:
        return self.effect_kind == "read_only"

    @property
    def may_dispatch_writes(self) -> bool:
        """Whether this contract carries what a write needs to be permitted.

        Every clause is load-bearing:

        * a declared **write** — an `unknown` effect kind is a command nobody
          has classified, and G1R denies it (arch §7.3);
        * an **effect key** — without one, two different effects on the same
          target derive the same logical-call key and the journal would join
          them as one operation;
        * a **reconciliation declaration** — an unknown outcome with no way to
          resolve it is a permanent unknown, and FW-REQ-008B forbids converting
          one into a known outcome without evidence. A command that can fail
          into an unresolvable state cannot be permitted to fail.
        """
        return (
            self.effect_kind == "write"
            and bool(self.effect_key)
            and self.reconciliation is not None
        )

    def refusal_reason(self) -> Optional[str]:
        """Why a write would be refused, or None when it would be permitted."""
        if self.effect_kind != "write":
            return (
                f"{self.definition_id} declares effect kind {self.effect_kind!r}; "
                "a write must be declared as one"
            )
        if not self.effect_key:
            return f"{self.definition_id} declares no adapter effect key"
        if self.reconciliation is None:
            return (
                f"{self.definition_id} declares no reconciliation strategy, so an "
                "unknown outcome could never be resolved"
            )
        return None


class ContractRegistry:
    """The contracts in force, by command definition ID.

    A lookup miss returns an `unknown` contract rather than raising: the caller
    must handle "nobody declared this" anyway, and returning a contract that
    says exactly that keeps every reader on one code path.
    """

    def __init__(self, contracts: Mapping[str, CommandEffectContract] = ()):
        self._contracts: dict[str, CommandEffectContract] = dict(contracts or {})

    def register(self, contract: CommandEffectContract) -> None:
        self._contracts[contract.definition_id] = contract

    def get(self, definition_id: str) -> CommandEffectContract:
        contract = self._contracts.get(definition_id)
        if contract is not None:
            return contract
        return CommandEffectContract(definition_id=definition_id, effect_kind="unknown")

    def __contains__(self, definition_id: str) -> bool:
        return definition_id in self._contracts

    def __len__(self) -> int:
        return len(self._contracts)


def binding_digest(
    *,
    definition_id: str,
    effect_key: str,
    target: Any,
    parameters: Any,
    security_scope: Optional[str] = None,
) -> str:
    """A stable digest of everything that makes this call *this* call.

    Two dispatches with the same logical-call key but different bindings are an
    idempotency conflict, not a retry (arch §14.3 step 5) — which only works if
    the binding covers the target and the parameters and not merely the name.
    """
    payload = json.dumps(
        {
            "definition_id": definition_id,
            "effect_key": effect_key,
            "target": target,
            "parameters": parameters,
            "security_scope": security_scope,
        },
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logical_call_key(
    *,
    scope_digest: str,
    turn_key: Optional[str],
    step_index: Optional[int],
    definition_id: str,
    effect_key: str,
    target: Any,
) -> str:
    """The identity a repeated dispatch of the same call derives again.

    Deliberately excludes the parameters that are not the target: a retry of the
    same step of the same turn against the same target IS the same logical call,
    and deriving a different key for it would let the journal mint a second
    operation for an effect that may already exist.

    The binding digest, which does include the parameters, is what catches a
    caller reusing the key for a different call.
    """
    payload = json.dumps(
        {
            "scope": scope_digest,
            "turn": turn_key or "",
            "step": step_index,
            "definition": definition_id,
            "effect": effect_key,
            "target": target,
        },
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
