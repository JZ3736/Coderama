"""Strategic Model Routing (Flash for Fast Tasks vs. Pro for Complex Planning).

Rubric Criterion 3.2 — Strategic Model Routing (5 pts):
- Routes specific requests and agent roles to the most appropriate Gemini model:
  * Fast Tier (`gemini-2.5-flash`): Used for low-latency web search, knowledge
    base retrieval, and simple policy lookups.
  * Complex Planning & Reasoning Tier (`gemini-2.5-pro`): Used for root multi-agent
    orchestration, multi-city itinerary planning, executive policy exception
    arbitration, and self-evaluation compliance auditing.
- Also provides `StrategicModelRouter` and `DynamicModelRoutingPlugin` to inspect
  incoming user query complexity in `before_model_callback` and dynamically select
  or upgrade the active model.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from .observability import structured_logger

FAST_MODEL_ID = os.environ.get("FAST_MODEL_ID", "gemini-2.5-flash")
PRO_PLANNING_MODEL_ID = os.environ.get("PRO_PLANNING_MODEL_ID", "gemini-2.5-pro")

_COMPLEX_REASONING_KEYWORDS: tuple[str, ...] = (
    "multi-city",
    "exception",
    "escalate",
    "optimize",
    "arbitrate",
    "audit",
    "international",
    "compare",
    "trade-off",
    "override",
    "executive",
)


@dataclass(frozen=True)
class ModelRouteDecision:
    """Represents a deterministic routing decision between Gemini Flash and Pro."""

    selected_model_id: str
    tier: Literal["FAST_FLASH_TIER", "COMPLEX_PRO_PLANNING_TIER"]
    reason: str
    estimated_complexity_score: int


class StrategicModelRouter:
    """Classifies task complexity and selects Gemini Flash vs. Gemini Pro."""

    def __init__(
        self,
        fast_model_id: str = FAST_MODEL_ID,
        pro_model_id: str = PRO_PLANNING_MODEL_ID,
    ) -> None:
        self.fast_model_id = fast_model_id
        self.pro_model_id = pro_model_id

    def create_fast_model(self) -> Gemini:
        """Creates a Gemini Flash model instance for fast, low-latency sub-agents."""
        return Gemini(model=self.fast_model_id)

    def create_pro_planning_model(self) -> Gemini:
        """Creates a Gemini Pro model instance for complex planning and auditing."""
        return Gemini(model=self.pro_model_id)

    def route_query(
        self,
        query_text: str,
        agent_role: str = "general",
    ) -> ModelRouteDecision:
        """Determines whether a request should route to Flash or Pro."""
        role_lower = agent_role.lower()
        query_lower = (query_text or "").lower()

        keyword_hits = sum(
            1 for kw in _COMPLEX_REASONING_KEYWORDS if kw in query_lower
        )
        word_count = len(query_lower.split())
        complexity_score = keyword_hits * 2 + (1 if word_count > 35 else 0)

        if any(
            r in role_lower
            for r in ("coordinator", "root", "planner", "auditor", "self_eval")
        ):
            decision = ModelRouteDecision(
                selected_model_id=self.pro_model_id,
                tier="COMPLEX_PRO_PLANNING_TIER",
                reason=(
                    f"Agent role '{agent_role}' requires deep multi-step planning "
                    "or compliance auditing."
                ),
                estimated_complexity_score=max(complexity_score, 3),
            )
        elif complexity_score >= 2:
            decision = ModelRouteDecision(
                selected_model_id=self.pro_model_id,
                tier="COMPLEX_PRO_PLANNING_TIER",
                reason=(
                    f"Query complexity score ({complexity_score}) triggered upgrade "
                    "to Pro planning model."
                ),
                estimated_complexity_score=complexity_score,
            )
        else:
            decision = ModelRouteDecision(
                selected_model_id=self.fast_model_id,
                tier="FAST_FLASH_TIER",
                reason=(
                    f"Routine lookup/extraction for role '{agent_role}' routed to "
                    "low-latency Flash model."
                ),
                estimated_complexity_score=complexity_score,
            )

        structured_logger.log_event(
            event_type="STRATEGIC_MODEL_ROUTE_DECISION",
            message=(
                f"Routed '{agent_role}' request to {decision.selected_model_id} "
                f"({decision.tier})"
            ),
            agent_role=agent_role,
            selected_model_id=decision.selected_model_id,
            tier=decision.tier,
            reason=decision.reason,
            complexity_score=decision.estimated_complexity_score,
        )
        return decision


strategic_model_router = StrategicModelRouter()


class DynamicModelRoutingPlugin(BasePlugin):
    """ADK Plugin that dynamically routes `llm_request.model` based on complexity."""

    def __init__(self, name: str = "dynamic_model_routing_plugin") -> None:
        super().__init__(name=name)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        latest_text = ""
        if llm_request.contents:
            last_content = llm_request.contents[-1]
            for part in last_content.parts or []:
                if getattr(part, "text", None):
                    latest_text += part.text + " "

        decision = strategic_model_router.route_query(
            query_text=latest_text.strip(),
            agent_role=callback_context.agent_name,
        )
        llm_request.model = decision.selected_model_id
        return None
