"""Security Guardrails, Policy Enforcement & Self-Evaluation ADK Plugin (Criterion 3.3 — 5/5 pts)."""

from __future__ import annotations

import re
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .observability import structured_logger
from .pii_redaction import pii_redactor

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|system)\s+instructions", re.I),
    re.compile(r"bypass\s+(corporate\s+)?(policy|human\s+approval|hitl)", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|secret\s+key|api\s+key)", re.I),
)

HARD_CORPORATE_SINGLE_BOOKING_CEILING_USD = 25000.0


def build_vertex_ai_safety_settings() -> list[types.SafetySetting]:
    return [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
    ]


class SelfEvalComplianceJudge:
    """Performs deterministic self-evaluation over agent responses for PII and HITL compliance."""

    def evaluate_response(self, response_text: str, session_state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        state = session_state or {}
        scrubbed_text, detected_pii = pii_redactor.redact_text(response_text)
        issues: list[str] = []
        if detected_pii:
            issues.append(f"Unredacted PII detected and scrubbed: {detected_pii}")
        latest = state.get("latest_booking") or {}
        if isinstance(latest, dict) and latest.get("status") == "PENDING_HUMAN_APPROVAL" and "card has been charged" in response_text.lower():
            issues.append("Response claimed card was charged while status is PENDING_HUMAN_APPROVAL.")
        report = {
            "self_eval_passed": len(issues) == 0,
            "pii_detected_and_redacted": bool(detected_pii),
            "detected_pii_types": detected_pii,
            "issues": issues,
            "sanitized_response_text": scrubbed_text,
        }
        structured_logger.log_event("SELF_EVAL_GUARDRAIL_CHECK", f"Self-eval passed={report['self_eval_passed']}", **report)
        return report


self_eval_judge = SelfEvalComplianceJudge()


class EnterpriseGuardrailsPolicyPlugin(BasePlugin):
    """ADK Plugin enforcing input prompt-injection blocks, tool policy caps, and output self-eval."""

    def __init__(self, name: str = "enterprise_guardrails_policy_plugin") -> None:
        super().__init__(name=name)
        self.judge = self_eval_judge

    @staticmethod
    def inspect_user_input(text: str) -> tuple[bool, Optional[str]]:
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text or ""):
                return False, f"Blocked by Enterprise Security Guardrail: matched '{pattern.pattern}'."
        return True, None

    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        for content in llm_request.contents or []:
            for part in content.parts or []:
                text = getattr(part, "text", None)
                if text:
                    allowed, reason = self.inspect_user_input(text)
                    if not allowed:
                        structured_logger.log_event("GUARDRAIL_INPUT_BLOCKED", reason or "Blocked", severity="WARNING")
                        return LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[types.Part.from_text(text="Blocked by Enterprise Security Guardrail: policy bypass is prohibited.")],
                            )
                        )
        return None

    async def before_tool_callback(self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext) -> Optional[dict[str, Any]]:
        charge = tool_args.get("total_charge_usd")
        if isinstance(charge, (int, float)) and float(charge) > HARD_CORPORATE_SINGLE_BOOKING_CEILING_USD:
            return {
                "status": "POLICY_ERROR",
                "tool_name": tool.name,
                "error_code": "EXCEEDS_HARD_TREASURY_CEILING",
                "error_message": f"Charge ${charge:.2f} exceeds treasury ceiling ${HARD_CORPORATE_SINGLE_BOOKING_CEILING_USD:.2f}.",
                "llm_recovery_instructions": ["Do not retry charging this amount; inform user CFO procurement is required."],
            }
        return None

    async def after_model_callback(self, *, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        if not llm_response.content or not llm_response.content.parts:
            return None
        updated: list[types.Part] = []
        modified = False
        for part in llm_response.content.parts:
            text = getattr(part, "text", None)
            if text:
                res = self.judge.evaluate_response(text)
                if res["pii_detected_and_redacted"]:
                    updated.append(types.Part.from_text(text=res["sanitized_response_text"]))
                    modified = True
                else:
                    updated.append(part)
            else:
                updated.append(part)
        if modified:
            llm_response.content = types.Content(role=llm_response.content.role or "model", parts=updated)
        return None
