"""Agent2Agent (A2A) Protocol Interoperability Descriptor & Task Handler.

References from Resource List:
- A2A Protocol Specification: https://a2a-protocol.org/latest/
- A2A Python SDK: https://github.com/a2aproject/a2a-python
- Google Cloud Agent Starter Pack: https://github.com/GoogleCloudPlatform/agent-starter-pack

Exposes an A2A-compliant `AgentCard` (`/.well-known/agent.json`) and task
invocation handler so peer enterprise agents (e.g., HR Onboarding Agent,
Corporate Finance ERP Agent) can discover and delegate tasks to
`travel_concierge_root`.
"""

from __future__ import annotations

from typing import Any

from .observability import structured_logger
from .tools import (
    draft_compliant_corporate_trip_itinerary,
    verify_corporate_travel_policy_compliance,
)


def get_a2a_agent_card(base_url: str = "https://travel-concierge.enterprise.internal") -> dict[str, Any]:
    """Returns the A2A Protocol `AgentCard` metadata (`/.well-known/agent.json`)."""
    return {
        "name": "Enterprise Corporate Travel & Expense Concierge",
        "description": (
            "ADK Multi-Agent system for corporate travel policy verification, "
            "live destination intelligence, and HITL-gated itinerary booking."
        ),
        "url": f"{base_url}/a2a",
        "version": "2.0.0",
        "protocolVersion": "0.2.1",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "authentication": {
            "schemes": ["Bearer", "GoogleCloudIAM"],
        },
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": [
            {
                "id": "verify-corporate-travel-policy",
                "name": "Verify Corporate Travel Policy Compliance",
                "description": (
                    "Checks proposed nightly hotel rates and flight cabin classes "
                    "against L1-L5, L6, and L7+ policy caps."
                ),
                "tags": ["travel", "policy", "compliance", "finance"],
                "examples": [
                    "Check if an L6 Director can book a $350/night hotel and Economy Plus to Chicago for 3 nights."
                ],
            },
            {
                "id": "draft-corporate-trip-itinerary",
                "name": "Draft Compliant Corporate Trip Itinerary",
                "description": (
                    "Drafts a structured business travel itinerary and flags whether "
                    "Human-in-the-Loop approval is required."
                ),
                "tags": ["travel", "booking", "hitl", "itinerary"],
                "examples": [
                    "Draft a 3-night trip from SFO to Chicago for an L6 Director staying at Marriott."
                ],
            },
        ],
    }


def handle_a2a_jsonrpc_request(request_payload: dict[str, Any]) -> dict[str, Any]:
    """Handles an incoming A2A JSON-RPC 2.0 `tasks/send` request."""
    rpc_id = request_payload.get("id", "a2a-req-1")
    method = request_payload.get("method", "tasks/send")
    params = request_payload.get("params") or {}

    corr_id = structured_logger.log_intent(
        action_category="A2A_PROTOCOL_REQUEST",
        target_name=method,
        intended_action=f"Process A2A peer agent request '{method}'",
        inputs=params,
    )

    if method == "agent/authenticatedExtendedCard":
        card = get_a2a_agent_card()
        structured_logger.log_outcome(
            correlation_id=corr_id,
            action_category="A2A_PROTOCOL_REQUEST",
            target_name=method,
            intended_action="Return A2A AgentCard",
            actual_outcome_status="SUCCESS",
            outputs=card,
            latency_ms=1.0,
        )
        return {"jsonrpc": "2.0", "id": rpc_id, "result": card}

    skill_id = params.get("skill_id", "verify-corporate-travel-policy")
    arguments = params.get("arguments") or {}

    if skill_id == "verify-corporate-travel-policy":
        result = verify_corporate_travel_policy_compliance(
            employee_level=arguments.get("employee_level", "L6 Director"),
            destination_city=arguments.get("destination_city", "Chicago"),
            nightly_hotel_rate_usd=float(arguments.get("nightly_hotel_rate_usd", 340.0)),
            flight_class=arguments.get("flight_class", "Economy Plus"),
            nights=int(arguments.get("nights", 3)),
        )
    else:
        result = draft_compliant_corporate_trip_itinerary(
            origin_airport_iata=arguments.get("origin_airport_iata", "SFO"),
            destination_city=arguments.get("destination_city", "Chicago"),
            travel_dates=arguments.get("travel_dates", "2026-11-10 to 2026-11-13"),
            nights=int(arguments.get("nights", 3)),
            flight_class=arguments.get("flight_class", "Economy Plus"),
            seat_preference=arguments.get("seat_preference", "aisle"),
            hotel_chain=arguments.get("hotel_chain", "Marriott"),
            nightly_hotel_rate_usd=float(arguments.get("nightly_hotel_rate_usd", 340.0)),
            employee_level=arguments.get("employee_level", "L6 Director"),
        )

    structured_logger.log_outcome(
        correlation_id=corr_id,
        action_category="A2A_PROTOCOL_REQUEST",
        target_name=skill_id,
        intended_action=f"Execute A2A skill '{skill_id}'",
        actual_outcome_status=result.get("status", "SUCCESS"),
        outputs=result,
        latency_ms=2.5,
    )

    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": {
            "taskId": params.get("id", "task-a2a-001"),
            "status": {"state": "completed"},
            "artifacts": [{"parts": [{"type": "data", "data": result}]}],
        },
    }
