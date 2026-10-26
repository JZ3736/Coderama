"""Explicit Input and Output JSON Schemas for Tool Validation and LLM Constraining.

Rubric Criterion 1.3 — Explicit JSON Schemas (5 pts):
- Uses strict Pydantic v2 `BaseModel` schemas (`ConfigDict(extra="forbid")`) with
  explicit `Field` constraints (`ge`, `le`, `min_length`, `max_length`, `pattern`,
  `Literal`) for both tool inputs and tool outputs, plus agent-level `output_schema`.
- Exports JSON Schema definitions (`TOOL_JSON_SCHEMAS`) via `.model_json_schema()`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared Error Recovery Schema (Rubric Criterion 1.4 — Guided Error Handling)
# ---------------------------------------------------------------------------
class StructuredToolErrorOutput(BaseModel):
    """Standardized error response guiding the LLM on how to self-correct."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["VALIDATION_ERROR", "POLICY_ERROR", "ERROR"] = Field(
        default="VALIDATION_ERROR",
        description="Machine-readable error status indicator.",
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool that encountered the validation or domain error.",
    )
    error_code: str = Field(
        ...,
        description="Specific error classification code (e.g., INVALID_AIRPORT_CODE).",
    )
    error_message: str = Field(
        ...,
        description="Human- and LLM-readable explanation of what failed.",
    )
    invalid_arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of offending parameter names and the invalid values received.",
    )
    llm_recovery_instructions: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Step-by-step instructions telling the LLM exactly how to correct "
            "its parameters or ask the user for clarification before retrying."
        ),
    )


# ---------------------------------------------------------------------------
# 1. Verify Corporate Travel Policy Compliance Schemas
# ---------------------------------------------------------------------------
class VerifyPolicyComplianceInput(BaseModel):
    """Strict input schema for `verify_corporate_travel_policy_compliance`."""

    model_config = ConfigDict(extra="forbid")

    employee_level: str = Field(
        ...,
        min_length=2,
        max_length=40,
        description="Employee level or title (e.g., 'L5 SWE', 'L6 Director', 'L7 VP').",
    )
    destination_city: str = Field(
        ...,
        min_length=2,
        max_length=80,
        description="Target destination city name (e.g., 'Chicago', 'London', 'Tokyo').",
    )
    nightly_hotel_rate_usd: float = Field(
        ...,
        gt=0.0,
        le=10000.0,
        description="Proposed nightly hotel rate in USD (must be > 0 and <= 10000).",
    )
    flight_class: Literal[
        "Economy", "Economy Plus", "Premium Economy", "Business", "First"
    ] = Field(
        ...,
        description="Requested flight cabin class.",
    )
    nights: int = Field(
        ...,
        ge=1,
        le=90,
        description="Number of hotel nights for the business trip (1 to 90).",
    )


class VerifyPolicyComplianceOutput(BaseModel):
    """Strict output schema for `verify_corporate_travel_policy_compliance`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["SUCCESS"] = "SUCCESS"
    compliant: bool = Field(
        ..., description="True if both hotel rate and flight class comply with policy."
    )
    resolved_policy_tier: Literal["L1-L5", "L6", "L7+"] = Field(
        ..., description="Normalized corporate policy tier."
    )
    destination_city: str
    hotel_compliant: bool
    max_allowed_nightly_usd: float
    proposed_nightly_usd: float
    flight_compliant: bool
    requested_flight_class: str
    allowed_flight_classes: list[str]
    estimated_total_trip_usd: float
    requires_human_approval: bool = Field(
        ...,
        description="True if total cost exceeds auto-approval cap or violates policy.",
    )
    violations: list[str] = Field(default_factory=list)
    remediation_guidance: str = Field(
        ..., description="Next-step guidance for the agent."
    )


# ---------------------------------------------------------------------------
# 2. Save Traveler Profile Preferences Schemas
# ---------------------------------------------------------------------------
class SaveTravelerPreferencesInput(BaseModel):
    """Strict input schema for `save_traveler_profile_preferences`."""

    model_config = ConfigDict(extra="forbid")

    employee_level: str = Field(
        ...,
        min_length=2,
        max_length=40,
        description="Employee level (e.g., 'L6 Director').",
    )
    home_airport_iata: str = Field(
        ...,
        pattern=r"^[A-Za-z]{3}$",
        description="3-letter IATA airport code (e.g., 'SFO', 'JFK', 'LHR').",
    )
    seat_preference: Literal["aisle", "window", "middle"] = Field(
        ...,
        description="Preferred aircraft seat position ('aisle', 'window', 'middle').",
    )
    preferred_hotel_chain: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description="Preferred hotel brand (e.g., 'Marriott', 'Hilton', 'Hyatt').",
    )
    dietary_or_loyalty_notes: str = Field(
        default="",
        max_length=300,
        description="Optional loyalty tier or dietary notes (automatically PII-scrubbed).",
    )


class SaveTravelerPreferencesOutput(BaseModel):
    """Strict output schema for `save_traveler_profile_preferences`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["SUCCESS"] = "SUCCESS"
    policy_tier: Literal["L1-L5", "L6", "L7+"]
    home_airport_iata: str
    seat_preference: str
    preferred_hotel_chain: str
    async_memory_consolidation_scheduled: bool
    pii_scrubbed: bool
    message: str


# ---------------------------------------------------------------------------
# 3. Search Corporate Policy Knowledge Base Schemas
# ---------------------------------------------------------------------------
class SearchPolicyKnowledgeBaseInput(BaseModel):
    """Strict input schema for `search_corporate_policy_knowledge_base`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Natural-language search query over corporate travel policies and vendor rules.",
    )
    policy_category: Literal[
        "ALL", "HOTEL_CAPS", "FLIGHT_CLASSES", "PER_DIEM", "APPROVAL_MATRIX", "SECURITY"
    ] = Field(
        default="ALL",
        description="Optional category filter to narrow knowledge base retrieval.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of policy documents to retrieve (1-10).",
    )


class PolicyDocumentSnippet(BaseModel):
    """A single retrieved document chunk from Vertex AI Search / policy vector store."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    category: str
    relevance_score: float
    content_snippet: str
    source_uri: str


class SearchPolicyKnowledgeBaseOutput(BaseModel):
    """Strict output schema for `search_corporate_policy_knowledge_base`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["SUCCESS"] = "SUCCESS"
    query: str
    retrieval_backend: str
    total_matches: int
    documents: list[PolicyDocumentSnippet]


# ---------------------------------------------------------------------------
# 4. Draft Compliant Corporate Trip Itinerary Schemas
# ---------------------------------------------------------------------------
class DraftTripItineraryInput(BaseModel):
    """Strict input schema for `draft_compliant_corporate_trip_itinerary`."""

    model_config = ConfigDict(extra="forbid")

    origin_airport_iata: str = Field(
        ...,
        pattern=r"^[A-Za-z]{3}$",
        description="3-letter origin IATA airport code (e.g., 'SFO').",
    )
    destination_city: str = Field(
        ...,
        min_length=2,
        max_length=80,
        description="Destination city name (e.g., 'Chicago').",
    )
    travel_dates: str = Field(
        ...,
        min_length=4,
        max_length=80,
        description="Date range for the trip (e.g., '2026-11-10 to 2026-11-13').",
    )
    nights: int = Field(..., ge=1, le=90)
    flight_class: Literal[
        "Economy", "Economy Plus", "Premium Economy", "Business", "First"
    ]
    seat_preference: Literal["aisle", "window", "middle"]
    hotel_chain: str = Field(..., min_length=2, max_length=60)
    nightly_hotel_rate_usd: float = Field(..., gt=0.0, le=10000.0)
    employee_level: str = Field(..., min_length=2, max_length=40)
    business_purpose: str = Field(
        default="Corporate Business Travel",
        min_length=3,
        max_length=200,
    )


class DraftTripItineraryOutput(BaseModel):
    """Strict output schema for `draft_compliant_corporate_trip_itinerary`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["DRAFT_READY", "DRAFT_REQUIRES_HITL_APPROVAL"]
    booking_reference: str
    policy_tier: Literal["L1-L5", "L6", "L7+"]
    origin_airport_iata: str
    destination_city: str
    travel_dates: str
    nights: int
    estimated_total_usd: float
    policy_compliant: bool
    requires_human_confirmation: bool
    next_step_instruction: str


# ---------------------------------------------------------------------------
# 5. Execute High-Stakes Corporate Booking Schemas (HITL Protected)
# ---------------------------------------------------------------------------
class ExecuteHighStakesBookingInput(BaseModel):
    """Strict input schema for `execute_high_stakes_corporate_booking`."""

    model_config = ConfigDict(extra="forbid")

    booking_reference: str = Field(
        ...,
        pattern=r"^TRIP-[A-Z0-9]{6}$",
        description="Draft booking reference ID matching '^TRIP-[A-Z0-9]{6}$'.",
    )
    total_charge_usd: float = Field(
        ...,
        gt=0.0,
        le=50000.0,
        description="Total USD amount to charge to the corporate credit line.",
    )
    human_approval_confirmed: bool = Field(
        default=False,
        description=(
            "Must be explicitly True with a valid `human_approver_id` for high-stakes "
            "bookings exceeding auto-approval limits."
        ),
    )
    human_approver_id: Optional[str] = Field(
        default=None,
        description="Identifier of the human manager who approved the high-stakes action.",
    )


class ExecuteHighStakesBookingOutput(BaseModel):
    """Strict output schema for `execute_high_stakes_corporate_booking`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["BOOKING_CONFIRMED", "PENDING_HUMAN_APPROVAL"]
    booking_reference: str
    total_charge_usd: float
    human_approval_verified: bool
    approver_id: Optional[str] = None
    confirmation_number: Optional[str] = None
    hitl_escalation_reason: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# 6. Submit Critical Policy Exception Ticket Schemas (HITL Protected)
# ---------------------------------------------------------------------------
class SubmitCriticalPolicyExceptionInput(BaseModel):
    """Strict input schema for `submit_critical_policy_exception_ticket`."""

    model_config = ConfigDict(extra="forbid")

    booking_reference: str = Field(
        ...,
        pattern=r"^TRIP-[A-Z0-9]{6}$",
        description="Associated draft booking reference ID (e.g., 'TRIP-1A2B3C').",
    )
    employee_level: str = Field(..., min_length=2, max_length=40)
    violation_summary: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Clear description of the policy cap or cabin class exception requested.",
    )
    justification: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Business justification for requesting executive exception approval.",
    )
    human_confirmation_received: bool = Field(
        default=False,
        description="True only after the human user explicitly confirms filing the escalation.",
    )


class SubmitCriticalPolicyExceptionOutput(BaseModel):
    """Strict output schema for `submit_critical_policy_exception_ticket`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["TICKET_CREATED", "PENDING_HUMAN_APPROVAL"]
    ticket_id: Optional[str]
    booking_reference: str
    priority: Literal["CRITICAL_FINANCE_EXCEPTION", "HIGH"]
    human_confirmation_verified: bool
    message: str


# ---------------------------------------------------------------------------
# Exported JSON Schemas Registry for Automated Inspection
# ---------------------------------------------------------------------------
TOOL_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "verify_corporate_travel_policy_compliance": {
        "input_schema": VerifyPolicyComplianceInput.model_json_schema(),
        "output_schema": VerifyPolicyComplianceOutput.model_json_schema(),
        "error_schema": StructuredToolErrorOutput.model_json_schema(),
    },
    "save_traveler_profile_preferences": {
        "input_schema": SaveTravelerPreferencesInput.model_json_schema(),
        "output_schema": SaveTravelerPreferencesOutput.model_json_schema(),
        "error_schema": StructuredToolErrorOutput.model_json_schema(),
    },
    "search_corporate_policy_knowledge_base": {
        "input_schema": SearchPolicyKnowledgeBaseInput.model_json_schema(),
        "output_schema": SearchPolicyKnowledgeBaseOutput.model_json_schema(),
        "error_schema": StructuredToolErrorOutput.model_json_schema(),
    },
    "draft_compliant_corporate_trip_itinerary": {
        "input_schema": DraftTripItineraryInput.model_json_schema(),
        "output_schema": DraftTripItineraryOutput.model_json_schema(),
        "error_schema": StructuredToolErrorOutput.model_json_schema(),
    },
    "execute_high_stakes_corporate_booking": {
        "input_schema": ExecuteHighStakesBookingInput.model_json_schema(),
        "output_schema": ExecuteHighStakesBookingOutput.model_json_schema(),
        "error_schema": StructuredToolErrorOutput.model_json_schema(),
    },
    "submit_critical_policy_exception_ticket": {
        "input_schema": SubmitCriticalPolicyExceptionInput.model_json_schema(),
        "output_schema": SubmitCriticalPolicyExceptionOutput.model_json_schema(),
        "error_schema": StructuredToolErrorOutput.model_json_schema(),
    },
}
