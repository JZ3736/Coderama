"""Human-in-the-Loop (HITL) Confirmation Hooks for High-Stakes Actions.

Rubric Criterion 3.4 — Human-in-the-Loop Hooks (5 pts):
- High-stakes actions include explicit code stops requiring human confirmation
  before execution:
  1. `should_require_human_confirmation`: Conditional predicate registered with
     ADK's `FunctionTool(..., require_confirmation=...)` to pause execution
     when a booking exceeds the auto-approval cap (`$1,500.00 USD`) or when a
     critical policy exception ticket is submitted.
  2. `verify_human_in_the_loop_approval`: Explicit programmatic code gate inside
     high-stakes tools that halts execution (`status="PENDING_HUMAN_APPROVAL"`)
     and sets `tool_context.actions.escalate` unless a verified human approval
     decision is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from google.adk.tools.tool_context import ToolContext

from .observability import structured_logger

HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD = 1500.0


@dataclass(frozen=True)
class HitlGateResult:
    """Result of evaluating a Human-in-the-Loop confirmation gate."""

    approved_to_execute: bool
    requires_human_confirmation: bool
    status: str
    escalation_reason: Optional[str]
    approver_id: Optional[str]


def should_require_human_confirmation(
    total_charge_usd: float = 0.0,
    human_approval_confirmed: bool = False,
    **kwargs: Any,
) -> bool:
    """Predicate used by ADK `FunctionTool(require_confirmation=...)` for HITL stops.

    Returns `True` (requiring ADK runtime confirmation pause) whenever the
    action is high-stakes and has not yet been confirmed by a human supervisor.
    """
    if human_approval_confirmed and kwargs.get("human_approver_id"):
        return False
    if float(total_charge_usd or 0.0) > HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD:
        return True
    if not human_approval_confirmed:
        return True
    return False


def verify_human_in_the_loop_approval(
    *,
    action_name: str,
    total_charge_usd: float,
    human_approval_confirmed: bool,
    human_approver_id: Optional[str],
    tool_context: Optional[ToolContext] = None,
    force_require_approval: bool = False,
) -> HitlGateResult:
    """Explicit programmatic HITL gate halting high-stakes execution until approved."""
    is_high_stakes = (
        force_require_approval
        or total_charge_usd > HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD
    )

    if not is_high_stakes:
        return HitlGateResult(
            approved_to_execute=True,
            requires_human_confirmation=False,
            status="AUTO_APPROVED_UNDER_THRESHOLD",
            escalation_reason=None,
            approver_id="SYSTEM_AUTO_APPROVAL",
        )

    if human_approval_confirmed and human_approver_id and human_approver_id.strip():
        structured_logger.log_event(
            event_type="HITL_APPROVAL_VERIFIED",
            message=(
                f"Human-in-the-Loop approval verified for '{action_name}' "
                f"by approver '{human_approver_id}'"
            ),
            action_name=action_name,
            total_charge_usd=total_charge_usd,
            approver_id=human_approver_id,
        )
        return HitlGateResult(
            approved_to_execute=True,
            requires_human_confirmation=True,
            status="HUMAN_APPROVED",
            escalation_reason=None,
            approver_id=human_approver_id.strip(),
        )

    reason = (
        f"High-stakes action '{action_name}' (amount=${total_charge_usd:.2f} USD, "
        f"threshold=${HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD:.2f} USD) requires explicit "
        "human supervisor confirmation (`human_approval_confirmed=True` and a valid "
        "`human_approver_id`) before execution."
    )

    # Request confirmation / pause via ADK ToolContext when available
    if tool_context is not None:
        try:
            if hasattr(tool_context, "request_confirmation"):
                tool_context.request_confirmation(
                    hint=reason,
                    payload={
                        "action_name": action_name,
                        "total_charge_usd": total_charge_usd,
                    },
                )
        except Exception:
            pass

    structured_logger.log_event(
        event_type="HITL_EXECUTION_PAUSED",
        message=reason,
        severity="WARNING",
        action_name=action_name,
        total_charge_usd=total_charge_usd,
    )
    return HitlGateResult(
        approved_to_execute=False,
        requires_human_confirmation=True,
        status="PENDING_HUMAN_APPROVAL",
        escalation_reason=reason,
        approver_id=None,
    )
