"""The client-visible description of the question a turn is waiting on (bd fix-fpyb.8, ``docs/pending_state_design.md``).

``payload_of(ctx, held)`` returns ``(question_kind, payload)`` for the turn's waiting record *held*: the kind is the
record's (``waiting.TABLE``), the payload is what its kind's builder reads from the record and the records it points at
(the approval store's record of an approval or a review, the plan resolution's question, the paused decision step).
``("", {})`` when the kind has no payload (the agent's own question). Raises on a lookup failure; the caller
(``WorkflowExecutionContext._describe_pending_prompt``) tags "unknown" then.
"""
from __future__ import annotations

import json
from typing import Any

from fastworkflow.core import no_agent_mode
from fastworkflow.core import waiting as waiting_module


def _json_safe(value):
    return json.loads(json.dumps(value, default=str))


def _approval(ctx, held) -> dict[str, Any]:
    from fastworkflow.core import approval_annotations
    from fastworkflow.review import approval_grants, review_engine

    store = review_engine.store_for(ctx._app_workflow)
    record = store.get(held.payload["record_id"])
    # A REQUIRED_ALWAYS command asks every time: no standing approval is offered for it (bd fix-qjne.11).
    can_always = (record.annotation != approval_annotations.REQUIRED_ALWAYS
                  and approval_grants.identity_key(store.surface, store.channel) is not None)
    return {"approval": _json_safe({
        "command": record.command,
        "parameters": dict(record.binding.get("params") or {}),
        "origin": record.origin,
        # No "irreversible": whether a change can be undone is the author's to say (bd fix-ftdl.7.6).
        "can_always": bool(can_always),
    })}


def _review(ctx, held) -> dict[str, Any]:
    from fastworkflow.review import review_engine, review_proposals

    record = review_engine.store_for(ctx._app_workflow).get(held.payload["record_id"])
    proposal = review_engine.proposal_of(record)
    # The one definition of each part's badge (``SubTaskProposal.badges``): the text shows the same.
    badges = (proposal.badges(ctx._app_workflow.folderpath if ctx._app_workflow else None)
              if isinstance(proposal, review_proposals.SubTaskProposal) else [""] * len(proposal.items))
    # True for a part that needs the explicit yes: it changes data, or (ad hoc) may change it.
    changes_data = [badge in (review_proposals.CHANGES_DATA_BADGE, review_proposals.MAY_CHANGE_DATA_BADGE)
                    for badge in badges]
    items_data = [item.model_dump(mode="json") for item in proposal.items]
    if ctx.no_agent_mode:
        # No agent model: an ad hoc item (means from_scratch) will not be run here; it carries the planner's text
        # plan (``plan_preview``, "" when there is none) so the client can carry it out. Others run.
        previews = ctx._turn_plan_previews
        for data, item in zip(items_data, proposal.items):
            ad_hoc = item.means.kind == "from_scratch"
            data["not_run_here"] = ad_hoc
            if ad_hoc:
                # An item an edit made ad hoc has no preview yet: made now, once.
                if item.goal not in previews:
                    previews[item.goal] = ctx._plan_preview(item.goal)
                data["plan_preview"] = previews[item.goal]
    return {"review": _json_safe({
        "kind": record.kind,
        "title": proposal.title,
        "revision": record.revision,
        "items": items_data,
        "changes_data": changes_data,
        "badges": badges,
        # The replies the shown prompt's hint promises (the text stays as it is).
        "reply_forms": review_engine.reply_forms(
            getattr(record, "rendering", "") or "", proposal, structured=record.turns_seen >= 1),
    })}


def _decision(ctx, held) -> dict[str, Any]:
    """A planning command paused at a step a client decides (no-agent mode, ``_suspend_for_decision``)."""
    decision = held.payload["decision"]
    return {no_agent_mode.PHASE_STEP_ARTIFACT: _json_safe(
        no_agent_mode.phase_step_payload(decision["stop"], decision, decision.get("facts")))}


def payload_of(ctx, held) -> tuple[str, dict[str, Any]]:
    """``(question_kind, payload)`` of *held*; ``("", {})`` when its kind has none."""
    if held.kind in (waiting_module.KIND_INTENT, waiting_module.KIND_PARAMETERS):
        # A question of the turn's resolution: which command a part gets, or a bound plan's missing values.
        from fastworkflow._workflows.plan_metadata_extraction import resolution

        kind, payload = resolution.prompt(held)
        return kind, _json_safe(payload)
    builder = {waiting_module.KIND_APPROVAL: _approval, waiting_module.KIND_REVIEW: _review,
               waiting_module.KIND_DECISION: _decision}.get(held.kind)
    return (held.question_kind, builder(ctx, held)) if builder is not None else ("", {})
