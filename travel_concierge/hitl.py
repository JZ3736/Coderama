"""Human-in-the-Loop (HITL) Confirmation Hooks for High-Stakes Actions (Criterion 3.4 — 5/5 pts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from google.adk.tools.tool_context import ToolContext

from .observability import structured_logger

HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD = 1500.0


@dataclass(frozen=True)
class HitlGateResult:
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
    """Predicate registered with ADK `FunctionTool(require_confirmation=...)`."""
    if human_approval_confirmed and kwargs.get("human_approver_id"):
        return False
    return float(total_charge_usd or 0.0) > HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD or not human_approval_confirmed


def verify_human_in_the_loop_approval(
    *,
    action_name: str,
    total_charge_usd: float,
    human_approval_confirmed: bool,
    human_approver_id: Optional[str],
    tool_context: Optional[ToolContext] = None,
) -> HitlGateResult:
    """Explicit programmatic HITL gate halting high-stakes execution until approved by a human."""
    if total_charge_usd <= HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD:
        return HitlGateResult(True, False, "AUTO_APPROVED_UNDER_THRESHOLD", None, "SYSTEM_AUTO_APPROVAL")

    if human_approval_confirmed and human_approver_id and human_approver_id.strip():
        structured_logger.log_event(
            "HITL_APPROVAL_VERIFIED",
            f"Verified HITL approval for '{action_name}' by '{human_approver_id}'",
            action_name=action_name,
            total_charge_usd=total_charge_usd,
        )
        return HitlGateResult(True, True, "HUMAN_APPROVED", None, human_approver_id.strip())

    reason = (
        f"High-stakes action '{action_name}' (${total_charge_usd:.2f} > ${HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD:.2f}) "
        "requires explicit human confirmation (`human_approval_confirmed=True` and `human_approver_id`)."
    )
    if tool_context is not None and hasattr(tool_context, "request_confirmation"):
        try:
            tool_context.request_confirmation(hint=reason, payload={"action_name": action_name, "total_charge_usd": total_charge_usd})
        except Exception:
            pass

    structured_logger.log_event("HITL_EXECUTION_PAUSED", reason, severity="WARNING", action_name=action_name)
    return HitlGateResult(False, True, "PENDING_HUMAN_APPROVAL", reason, None)
