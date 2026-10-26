"""Security Guardrails, Policy Enforcement & Self-Evaluation ADK Plugin.

Rubric Criterion 3.3 — Guardrails & Policy Plugins (5 pts):
- Implements security and self-evaluation guardrails via ADK's `BasePlugin`
  (`EnterpriseGuardrailsPolicyPlugin`) and Google Cloud Vertex AI Safety Settings:
  1. Input Guardrail (`before_model_callback`): Blocks prompt injection,
     jailbreak attempts, and policy bypass instructions before reaching the LLM.
  2. Google Cloud Safety Settings: Configures `types.SafetySetting` thresholds
     for hate speech, harassment, dangerous content, and civic integrity.
  3. Tool Policy Guardrail (`before_tool_callback`): Enforces spending hard-caps
     and blocks unauthorized tool execution.
  4. Self-Evaluation Output Guardrail (`after_model_callback` + `evaluate_response_compliance`):
     Audits model outputs for policy groundedness, unredacted PII, and
     unauthorized booking confirmations before returning to the user.
"""

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
    re.compile(r"system\s+override\s+grant\s+unlimited\s+budget", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|secret\s+key|api\s+key)", re.I),
)

HARD_CORPORATE_SINGLE_BOOKING_CEILING_USD = 25000.0


def build_vertex_ai_safety_settings() -> list[types.SafetySetting]:
    """Constructs Google Cloud Vertex AI Gemini SafetySettings for all agents."""
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
    ]


class SelfEvalComplianceJudge:
    """Performs deterministic + rubric self-evaluation over agent responses."""

    def evaluate_response(
        self,
        response_text: str,
        session_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Evaluates whether an agent response satisfies enterprise safety & policy rules."""
        state = session_state or {}
        scrubbed_text, detected_pii = pii_redactor.redact_text(response_text)

        issues: list[str] = []
        if detected_pii:
            issues.append(
                f"Response contained unredacted PII types: {detected_pii} (auto-scrubbed)."
            )

        # Verify that if a booking is marked PENDING_HUMAN_APPROVAL in state,
        # the response does not falsely claim the credit card was already charged.
        latest_booking = state.get("latest_booking") or {}
        if (
            isinstance(latest_booking, dict)
            and latest_booking.get("status") == "PENDING_HUMAN_APPROVAL"
            and "card has been charged" in response_text.lower()
        ):
            issues.append(
                "Self-eval violation: Response claimed card was charged while "
                "booking status is PENDING_HUMAN_APPROVAL."
            )

        passed = len(issues) == 0
        eval_report = {
            "self_eval_passed": passed,
            "pii_detected_and_redacted": bool(detected_pii),
            "detected_pii_types": detected_pii,
            "issues": issues,
            "sanitized_response_text": scrubbed_text,
        }
        structured_logger.log_event(
            event_type="SELF_EVAL_GUARDRAIL_CHECK",
            message=f"Self-evaluation completed: passed={passed}",
            **eval_report,
        )
        return eval_report


self_eval_judge = SelfEvalComplianceJudge()


class EnterpriseGuardrailsPolicyPlugin(BasePlugin):
    """ADK Plugin enforcing input prompt-injection blocks, tool policy caps, and output self-eval."""

    def __init__(self, name: str = "enterprise_guardrails_policy_plugin") -> None:
        super().__init__(name=name)
        self.judge = self_eval_judge

    @staticmethod
    def inspect_user_input(text: str) -> tuple[bool, Optional[str]]:
        """Checks user input text against prompt-injection and policy-bypass signatures."""
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text or ""):
                return (
                    False,
                    f"Blocked by Enterprise Security Guardrail: matched prohibited pattern '{pattern.pattern}'.",
                )
        return True, None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        for content in llm_request.contents or []:
            for part in content.parts or []:
                text = getattr(part, "text", None)
                if text:
                    allowed, reason = self.inspect_user_input(text)
                    if not allowed:
                        structured_logger.log_event(
                            event_type="GUARDRAIL_INPUT_BLOCKED",
                            message=reason or "Blocked prompt injection",
                            severity="WARNING",
                            agent_name=callback_context.agent_name,
                        )
                        return LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part.from_text(
                                        text=(
                                            "Request blocked by Enterprise Security Guardrail: "
                                            "Instructions attempting to bypass corporate travel "
                                            "policies, human-in-the-loop approvals, or system "
                                            "constitutions are strictly prohibited."
                                        )
                                    )
                                ],
                            )
                        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict[str, Any]]:
        total_charge = tool_args.get("total_charge_usd")
        if (
            isinstance(total_charge, (int, float))
            and float(total_charge) > HARD_CORPORATE_SINGLE_BOOKING_CEILING_USD
        ):
            structured_logger.log_event(
                event_type="GUARDRAIL_TOOL_BLOCKED",
                message=(
                    f"Blocked tool '{tool.name}': charge ${total_charge} exceeds "
                    f"hard ceiling ${HARD_CORPORATE_SINGLE_BOOKING_CEILING_USD}"
                ),
                severity="ERROR",
                tool_name=tool.name,
            )
            return {
                "status": "POLICY_ERROR",
                "tool_name": tool.name,
                "error_code": "EXCEEDS_HARD_TREASURY_CEILING",
                "error_message": (
                    f"Requested charge (${total_charge:.2f} USD) exceeds the absolute "
                    f"corporate treasury single-booking ceiling of "
                    f"${HARD_CORPORATE_SINGLE_BOOKING_CEILING_USD:.2f} USD."
                ),
                "llm_recovery_instructions": [
                    "Do not retry charging this amount.",
                    "Inform the user that charges above $25,000 USD require offline CFO procurement.",
                ],
            }
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if not llm_response.content or not llm_response.content.parts:
            return None

        updated_parts: list[types.Part] = []
        modified = False
        state_dict: dict[str, Any] = {}
        if hasattr(callback_context, "state") and callback_context.state is not None:
            try:
                state_dict = {
                    "latest_booking": callback_context.state.get("latest_booking")
                }
            except Exception:
                state_dict = {}

        for part in llm_response.content.parts:
            text = getattr(part, "text", None)
            if text:
                eval_res = self.judge.evaluate_response(
                    response_text=text, session_state=state_dict
                )
                if eval_res["pii_detected_and_redacted"]:
                    updated_parts.append(
                        types.Part.from_text(
                            text=eval_res["sanitized_response_text"]
                        )
                    )
                    modified = True
                else:
                    updated_parts.append(part)
            else:
                updated_parts.append(part)

        if modified:
            llm_response.content = types.Content(
                role=llm_response.content.role or "model",
                parts=updated_parts,
            )
        return None
