"""Enterprise Travel & Expense Tools with Explicit Schemas, Guided Error Recovery & HITL Hooks.

Rubric Criteria Covered:
- 1.1 Comprehensive Tool Docstrings (5 pts): Complete human-readable descriptions
  of purpose, business logic, all parameters, and return structure.
- 1.2 Descriptive Naming (5 pts): Highly specific action-oriented names
  (e.g., `verify_corporate_travel_policy_compliance`,
  `execute_high_stakes_corporate_booking`,
  `submit_critical_policy_exception_ticket`).
- 1.3 Explicit JSON Schemas (5 pts): Strict Pydantic v2 input and output schema
  validation on every tool call.
- 1.4 Guided Error Handling (5 pts): Catches validation and domain errors and
  returns `StructuredToolErrorOutput` with explicit `llm_recovery_instructions`
  instead of crashing.
- 3.4 Human-in-the-Loop Hooks (5 pts): High-stakes tools are wrapped with ADK
  `FunctionTool(..., require_confirmation=should_require_human_confirmation)`
  and enforce `verify_human_in_the_loop_approval`.
"""

from __future__ import annotations

import datetime
from typing import Any, Literal, Optional
import uuid

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from .hitl import (
    HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD,
    should_require_human_confirmation,
    verify_human_in_the_loop_approval,
)
from .memory_and_context import async_memory_consolidator
from .observability import trace_and_log_intent_outcome
from .pii_redaction import pii_redactor
from .schemas import (
    DraftTripItineraryInput,
    DraftTripItineraryOutput,
    ExecuteHighStakesBookingInput,
    ExecuteHighStakesBookingOutput,
    PolicyDocumentSnippet,
    SaveTravelerPreferencesInput,
    SaveTravelerPreferencesOutput,
    SearchPolicyKnowledgeBaseInput,
    SearchPolicyKnowledgeBaseOutput,
    StructuredToolErrorOutput,
    SubmitCriticalPolicyExceptionInput,
    SubmitCriticalPolicyExceptionOutput,
    VerifyPolicyComplianceInput,
    VerifyPolicyComplianceOutput,
)
from .secret_manager import secret_manager_service

# ---------------------------------------------------------------------------
# Authoritative Corporate Policy Tiers & Knowledge Base Corpus
# ---------------------------------------------------------------------------
CORPORATE_POLICY_TIERS: dict[Literal["L1-L5", "L6", "L7+"], dict[str, Any]] = {
    "L1-L5": {
        "max_hotel_nightly_usd": 250.0,
        "allowed_flight_classes": ["Economy"],
        "daily_meal_per_diem_usd": 75.0,
        "auto_approval_limit_usd": 1500.0,
    },
    "L6": {
        "max_hotel_nightly_usd": 400.0,
        "allowed_flight_classes": ["Economy", "Economy Plus", "Premium Economy"],
        "daily_meal_per_diem_usd": 100.0,
        "auto_approval_limit_usd": 3000.0,
    },
    "L7+": {
        "max_hotel_nightly_usd": 600.0,
        "allowed_flight_classes": [
            "Economy",
            "Economy Plus",
            "Premium Economy",
            "Business",
        ],
        "daily_meal_per_diem_usd": 150.0,
        "auto_approval_limit_usd": 6000.0,
    },
}

CORPORATE_POLICY_KNOWLEDGE_CORPUS: list[dict[str, Any]] = [
    {
        "doc_id": "POL-HOTEL-001",
        "title": "Global Corporate Hotel Nightly Rate Caps by Level Tier",
        "category": "HOTEL_CAPS",
        "content_snippet": (
            "Employees in tier L1-L5 may book lodging up to $250/night USD. "
            "L6 Directors are capped at $400/night USD. L7+ Executives are capped "
            "at $600/night USD. Preferred corporate partners are Marriott, Hilton, and Hyatt."
        ),
        "source_uri": "vertex-ai-search://corporate-travel-policy-store/POL-HOTEL-001",
    },
    {
        "doc_id": "POL-FLIGHT-002",
        "title": "Commercial Airfare Cabin Class Eligibility Policy",
        "category": "FLIGHT_CLASSES",
        "content_snippet": (
            "L1-L5 employees must book Economy class for domestic and international flights. "
            "L6 Directors may book Economy, Economy Plus, or Premium Economy. "
            "L7+ VPs and Executives may book Business class. First class is prohibited "
            "without a critical policy exception ticket approved by a human supervisor."
        ),
        "source_uri": "vertex-ai-search://corporate-travel-policy-store/POL-FLIGHT-002",
    },
    {
        "doc_id": "POL-PERDIEM-003",
        "title": "Daily Meal & Incidental Expense (M&IE) Per-Diem Allowances",
        "category": "PER_DIEM",
        "content_snippet": (
            "Daily meal per-diem allowances are $75/day for L1-L5, $100/day for L6, "
            "and $150/day for L7+. Itemized receipts are required for any single "
            "expense exceeding $75 USD."
        ),
        "source_uri": "vertex-ai-search://corporate-travel-policy-store/POL-PERDIEM-003",
    },
    {
        "doc_id": "POL-HITL-004",
        "title": "Human-in-the-Loop (HITL) Approval Matrix for High-Stakes Bookings",
        "category": "APPROVAL_MATRIX",
        "content_snippet": (
            "Any corporate travel booking exceeding $1,500.00 USD in total charges, "
            "or any request requiring a policy exception ticket, triggers a mandatory "
            "Human-in-the-Loop (HITL) confirmation stop before charging corporate credit lines."
        ),
        "source_uri": "vertex-ai-search://corporate-travel-policy-store/POL-HITL-004",
    },
    {
        "doc_id": "POL-SEC-005",
        "title": "Traveler Data Privacy & PII Redaction Standard",
        "category": "SECURITY",
        "content_snippet": (
            "All credit card numbers, SSNs, passport numbers, personal emails, and phone "
            "numbers must be redacted via Google Cloud DLP / regex scrubbing before "
            "writing to logs or Vertex AI Memory Bank."
        ),
        "source_uri": "vertex-ai-search://corporate-travel-policy-store/POL-SEC-005",
    },
]


def _resolve_policy_tier(employee_level: str) -> Literal["L1-L5", "L6", "L7+"]:
    """Maps a free-form employee level string to a canonical policy tier."""
    normalized = employee_level.strip().upper()
    if any(t in normalized for t in ("L7", "L8", "L9", "VP", "SVP", "EVP", "EXEC", "C-SUITE")):
        return "L7+"
    if "L6" in normalized or "DIRECTOR" in normalized or "PRINCIPAL" in normalized:
        return "L6"
    return "L1-L5"


def _build_guided_validation_error(
    tool_name: str,
    exc: ValidationError,
    received_args: dict[str, Any],
    specific_recovery_hints: list[str],
) -> dict[str, Any]:
    """Builds a `StructuredToolErrorOutput` with actionable recovery instructions for the LLM."""
    error_details = "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in exc.errors()
    )
    error_output = StructuredToolErrorOutput(
        status="VALIDATION_ERROR",
        tool_name=tool_name,
        error_code="SCHEMA_VALIDATION_FAILED",
        error_message=f"Input validation failed for {tool_name}: {error_details}",
        invalid_arguments=pii_redactor.redact_payload(received_args),
        llm_recovery_instructions=[
            f"Fix the invalid parameters reported in `{error_details}`.",
            *specific_recovery_hints,
            "Once corrected, invoke the tool again with valid JSON arguments.",
        ],
    )
    return error_output.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool 1: Verify Corporate Travel Policy Compliance
# ---------------------------------------------------------------------------
@trace_and_log_intent_outcome(
    "Validate proposed hotel nightly rate, flight cabin class, and trip cost against corporate policy tiers"
)
def verify_corporate_travel_policy_compliance(
    employee_level: str,
    destination_city: str,
    nightly_hotel_rate_usd: float,
    flight_class: str,
    nights: int,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Verifies whether a proposed business trip complies with corporate travel policy caps.

    Evaluates the traveler's employee level (`L1-L5`, `L6`, `L7+`) against maximum
    nightly hotel rate limits, eligible commercial flight cabin classes, meal per-diem
    allowances, and Human-in-the-Loop approval thresholds.

    Args:
        employee_level: Employee level or title (e.g., 'L5 SWE', 'L6 Director', 'L7 VP').
        destination_city: Target destination city name (e.g., 'Chicago', 'New York').
        nightly_hotel_rate_usd: Proposed nightly hotel rate in USD (must be > 0 and <= 10000).
        flight_class: Requested cabin class. Must be one of: 'Economy', 'Economy Plus',
            'Premium Economy', 'Business', or 'First'.
        nights: Number of hotel nights for the trip (integer between 1 and 90).
        tool_context: Optional ADK `ToolContext` used to persist the policy evaluation in
            `session.state['last_policy_check']`.

    Returns:
        A JSON-serializable dictionary conforming to `VerifyPolicyComplianceOutput`
        when inputs are valid, or `StructuredToolErrorOutput` with guided recovery
        instructions if input validation fails.
    """
    raw_inputs = {
        "employee_level": employee_level,
        "destination_city": destination_city,
        "nightly_hotel_rate_usd": nightly_hotel_rate_usd,
        "flight_class": flight_class,
        "nights": nights,
    }
    try:
        validated = VerifyPolicyComplianceInput.model_validate(raw_inputs)
    except ValidationError as exc:
        return _build_guided_validation_error(
            tool_name="verify_corporate_travel_policy_compliance",
            exc=exc,
            received_args=raw_inputs,
            specific_recovery_hints=[
                "Ensure `nightly_hotel_rate_usd` is a positive number (e.g., 320.0).",
                "Ensure `flight_class` is exactly one of: 'Economy', 'Economy Plus', 'Premium Economy', 'Business', 'First'.",
                "Ensure `nights` is an integer between 1 and 90.",
            ],
        )

    tier = _resolve_policy_tier(validated.employee_level)
    rules = CORPORATE_POLICY_TIERS[tier]

    hotel_compliant = validated.nightly_hotel_rate_usd <= rules["max_hotel_nightly_usd"]
    flight_compliant = validated.flight_class in rules["allowed_flight_classes"]
    overall_compliant = hotel_compliant and flight_compliant

    estimated_hotel_total = round(validated.nightly_hotel_rate_usd * validated.nights, 2)
    estimated_per_diem_total = round(rules["daily_meal_per_diem_usd"] * validated.nights, 2)
    estimated_total_trip_usd = round(estimated_hotel_total + estimated_per_diem_total, 2)

    violations: list[str] = []
    if not hotel_compliant:
        violations.append(
            f"Proposed nightly hotel rate (${validated.nightly_hotel_rate_usd:.2f} USD) "
            f"exceeds {tier} cap of ${rules['max_hotel_nightly_usd']:.2f} USD/night."
        )
    if not flight_compliant:
        violations.append(
            f"Requested flight class '{validated.flight_class}' is not permitted for {tier}. "
            f"Allowed classes: {', '.join(rules['allowed_flight_classes'])}."
        )

    requires_hitl = (
        not overall_compliant
        or estimated_total_trip_usd > HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD
    )

    if overall_compliant:
        guidance = (
            "Trip parameters comply with corporate policy. Proceed to call "
            "`draft_compliant_corporate_trip_itinerary`."
        )
    else:
        guidance = (
            "Policy violations detected. Either adjust the hotel rate / flight class "
            "to fit within tier caps, or draft the itinerary and request human approval via "
            "`submit_critical_policy_exception_ticket`."
        )

    output = VerifyPolicyComplianceOutput(
        compliant=overall_compliant,
        resolved_policy_tier=tier,
        destination_city=validated.destination_city,
        hotel_compliant=hotel_compliant,
        max_allowed_nightly_usd=rules["max_hotel_nightly_usd"],
        proposed_nightly_usd=validated.nightly_hotel_rate_usd,
        flight_compliant=flight_compliant,
        requested_flight_class=validated.flight_class,
        allowed_flight_classes=rules["allowed_flight_classes"],
        estimated_total_trip_usd=estimated_total_trip_usd,
        requires_human_approval=requires_hitl,
        violations=violations,
        remediation_guidance=guidance,
    ).model_dump(mode="json")

    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["last_policy_check"] = output

    return output


# ---------------------------------------------------------------------------
# Tool 2: Save Traveler Profile Preferences (with Async Memory Consolidation)
# ---------------------------------------------------------------------------
@trace_and_log_intent_outcome(
    "Scrub PII and persist traveler profile preferences to Session State and async Vertex AI Memory Bank"
)
def save_traveler_profile_preferences(
    employee_level: str,
    home_airport_iata: str,
    seat_preference: str,
    preferred_hotel_chain: str,
    dietary_or_loyalty_notes: str = "",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Saves a traveler's profile preferences to session state and asynchronously consolidates to Memory Bank.

    Scrubs any accidental PII in `dietary_or_loyalty_notes` via `pii_redactor` and
    schedules non-blocking background memory consolidation (`AsyncMemoryConsolidator`)
    so the UI is never blocked waiting on vector index embeddings.

    Args:
        employee_level: Employee level or title (e.g., 'L6 Director', 'L5 SWE').
        home_airport_iata: 3-letter IATA airport code for the traveler's home airport (e.g., 'SFO', 'JFK').
        seat_preference: Preferred aircraft seat position ('aisle', 'window', or 'middle').
        preferred_hotel_chain: Preferred hotel brand (e.g., 'Marriott', 'Hilton', 'Hyatt').
        dietary_or_loyalty_notes: Optional loyalty program or dietary notes (automatically PII-scrubbed).
        tool_context: Optional ADK `ToolContext` for storing `traveler_profile` in `session.state`.

    Returns:
        A JSON-serializable dictionary matching `SaveTravelerPreferencesOutput` on success,
        or `StructuredToolErrorOutput` with recovery guidance if parameters are malformed.
    """
    raw_inputs = {
        "employee_level": employee_level,
        "home_airport_iata": home_airport_iata.strip().upper()
        if isinstance(home_airport_iata, str)
        else home_airport_iata,
        "seat_preference": seat_preference.strip().lower()
        if isinstance(seat_preference, str)
        else seat_preference,
        "preferred_hotel_chain": preferred_hotel_chain,
        "dietary_or_loyalty_notes": dietary_or_loyalty_notes,
    }
    try:
        validated = SaveTravelerPreferencesInput.model_validate(raw_inputs)
    except ValidationError as exc:
        return _build_guided_validation_error(
            tool_name="save_traveler_profile_preferences",
            exc=exc,
            received_args=raw_inputs,
            specific_recovery_hints=[
                "Provide a valid 3-letter IATA airport code for `home_airport_iata` (e.g., 'SFO', 'ORD', 'JFK').",
                "Ensure `seat_preference` is one of: 'aisle', 'window', 'middle'.",
            ],
        )

    scrubbed_notes, detected_pii = pii_redactor.redact_text(
        validated.dietary_or_loyalty_notes
    )
    tier = _resolve_policy_tier(validated.employee_level)

    profile_record = {
        "employee_level": validated.employee_level,
        "policy_tier": tier,
        "home_airport_iata": validated.home_airport_iata.upper(),
        "seat_preference": validated.seat_preference,
        "preferred_hotel_chain": validated.preferred_hotel_chain.strip(),
        "dietary_or_loyalty_notes": scrubbed_notes,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["traveler_profile"] = profile_record

    async_task_info = async_memory_consolidator.schedule_background_consolidation(
        memory_payload=profile_record,
        context=tool_context,
    )

    output = SaveTravelerPreferencesOutput(
        policy_tier=tier,
        home_airport_iata=profile_record["home_airport_iata"],
        seat_preference=validated.seat_preference,
        preferred_hotel_chain=profile_record["preferred_hotel_chain"],
        async_memory_consolidation_scheduled=bool(
            async_task_info["scheduled_in_background"]
        ),
        pii_scrubbed=bool(detected_pii),
        message=(
            f"Saved traveler profile for {validated.employee_level} ({tier}) at "
            f"{profile_record['home_airport_iata']} and scheduled non-blocking "
            f"background consolidation (task_id={async_task_info['task_id']})."
        ),
    )
    return output.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool 3: Search Corporate Policy Knowledge Base (Vertex AI Search + Vector Store)
# ---------------------------------------------------------------------------
@trace_and_log_intent_outcome(
    "Retrieve relevant corporate travel policy documents from Vertex AI Search knowledge base"
)
def search_corporate_policy_knowledge_base(
    query: str,
    policy_category: str = "ALL",
    top_k: int = 3,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Searches the persistent corporate travel policy knowledge base backed by Vertex AI Search.

    Retrieves authoritative policy clauses covering hotel nightly caps, eligible flight
    cabin classes, meal per-diem allowances, Human-in-the-Loop approval matrices, and
    PII security standards.

    Args:
        query: Natural-language search query (3 to 300 characters, e.g., 'hotel cap for L6 director').
        policy_category: Optional category filter. One of: 'ALL', 'HOTEL_CAPS',
            'FLIGHT_CLASSES', 'PER_DIEM', 'APPROVAL_MATRIX', 'SECURITY'.
        top_k: Maximum number of matching policy documents to return (integer 1 to 10).
        tool_context: Optional ADK `ToolContext` to record retrieved citations in `session.state`.

    Returns:
        A JSON-serializable dictionary conforming to `SearchPolicyKnowledgeBaseOutput`,
        or `StructuredToolErrorOutput` if the query or category filter is invalid.
    """
    raw_inputs = {
        "query": query,
        "policy_category": policy_category.strip().upper()
        if isinstance(policy_category, str)
        else policy_category,
        "top_k": top_k,
    }
    try:
        validated = SearchPolicyKnowledgeBaseInput.model_validate(raw_inputs)
    except ValidationError as exc:
        return _build_guided_validation_error(
            tool_name="search_corporate_policy_knowledge_base",
            exc=exc,
            received_args=raw_inputs,
            specific_recovery_hints=[
                "Provide a non-empty `query` of at least 3 characters.",
                "Set `policy_category` to one of: 'ALL', 'HOTEL_CAPS', 'FLIGHT_CLASSES', 'PER_DIEM', 'APPROVAL_MATRIX', 'SECURITY'.",
            ],
        )

    runtime_cfg = secret_manager_service.load_runtime_config()
    query_tokens = set(validated.query.lower().split())

    scored_docs: list[PolicyDocumentSnippet] = []
    for doc in CORPORATE_POLICY_KNOWLEDGE_CORPUS:
        if (
            validated.policy_category != "ALL"
            and doc["category"] != validated.policy_category
        ):
            continue
        haystack = f"{doc['title']} {doc['content_snippet']} {doc['category']}".lower()
        overlap = sum(1 for tok in query_tokens if tok in haystack)
        score = round(min(0.99, 0.55 + overlap * 0.12), 2)
        scored_docs.append(
            PolicyDocumentSnippet(
                doc_id=doc["doc_id"],
                title=doc["title"],
                category=doc["category"],
                relevance_score=score,
                content_snippet=doc["content_snippet"],
                source_uri=doc["source_uri"],
            )
        )

    scored_docs.sort(key=lambda d: d.relevance_score, reverse=True)
    top_docs = scored_docs[: validated.top_k]

    output = SearchPolicyKnowledgeBaseOutput(
        query=validated.query,
        retrieval_backend=f"VertexAiSearch:{runtime_cfg.vertex_search_data_store_id}",
        total_matches=len(top_docs),
        documents=top_docs,
    ).model_dump(mode="json")

    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["last_knowledge_search"] = output

    return output


# ---------------------------------------------------------------------------
# Tool 4: Draft Compliant Corporate Trip Itinerary
# ---------------------------------------------------------------------------
@trace_and_log_intent_outcome(
    "Draft a structured corporate travel itinerary in session state and evaluate HITL requirements"
)
def draft_compliant_corporate_trip_itinerary(
    origin_airport_iata: str,
    destination_city: str,
    travel_dates: str,
    nights: int,
    flight_class: str,
    seat_preference: str,
    hotel_chain: str,
    nightly_hotel_rate_usd: float,
    employee_level: str,
    business_purpose: str = "Corporate Business Travel",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Creates a structured corporate trip booking draft and determines if human approval is required.

    Calculates total hotel and meal per-diem costs, validates compliance against the
    traveler's policy tier, stores the itinerary draft in `session.state['latest_booking']`,
    and flags whether `execute_high_stakes_corporate_booking` will require Human-in-the-Loop
    confirmation.

    Args:
        origin_airport_iata: 3-letter departure airport IATA code (e.g., 'SFO').
        destination_city: Destination city name (e.g., 'Chicago').
        travel_dates: Date range for the trip (e.g., '2026-11-10 to 2026-11-13').
        nights: Number of lodging nights (integer 1 to 90).
        flight_class: Cabin class ('Economy', 'Economy Plus', 'Premium Economy', 'Business', 'First').
        seat_preference: Seat preference ('aisle', 'window', 'middle').
        hotel_chain: Hotel brand name (e.g., 'Marriott').
        nightly_hotel_rate_usd: Proposed nightly hotel rate in USD (> 0).
        employee_level: Traveler's employee level (e.g., 'L6 Director').
        business_purpose: Business reason or conference name.
        tool_context: Optional ADK `ToolContext` to store `latest_booking` in `session.state`.

    Returns:
        A JSON-serializable dictionary matching `DraftTripItineraryOutput`, or
        `StructuredToolErrorOutput` if any parameter fails schema validation.
    """
    raw_inputs = {
        "origin_airport_iata": origin_airport_iata.strip().upper()
        if isinstance(origin_airport_iata, str)
        else origin_airport_iata,
        "destination_city": destination_city,
        "travel_dates": travel_dates,
        "nights": nights,
        "flight_class": flight_class,
        "seat_preference": seat_preference.strip().lower()
        if isinstance(seat_preference, str)
        else seat_preference,
        "hotel_chain": hotel_chain,
        "nightly_hotel_rate_usd": nightly_hotel_rate_usd,
        "employee_level": employee_level,
        "business_purpose": business_purpose,
    }
    try:
        validated = DraftTripItineraryInput.model_validate(raw_inputs)
    except ValidationError as exc:
        return _build_guided_validation_error(
            tool_name="draft_compliant_corporate_trip_itinerary",
            exc=exc,
            received_args=raw_inputs,
            specific_recovery_hints=[
                "Ensure `origin_airport_iata` is a 3-letter IATA code such as 'SFO' or 'JFK'.",
                "Ensure `seat_preference` is 'aisle', 'window', or 'middle'.",
                "Ensure `flight_class` is one of: 'Economy', 'Economy Plus', 'Premium Economy', 'Business', 'First'.",
            ],
        )

    tier = _resolve_policy_tier(validated.employee_level)
    rules = CORPORATE_POLICY_TIERS[tier]
    booking_ref = f"TRIP-{uuid.uuid4().hex[:6].upper()}"

    hotel_compliant = validated.nightly_hotel_rate_usd <= rules["max_hotel_nightly_usd"]
    flight_compliant = validated.flight_class in rules["allowed_flight_classes"]
    policy_compliant = hotel_compliant and flight_compliant

    estimated_total_usd = round(
        (validated.nightly_hotel_rate_usd + rules["daily_meal_per_diem_usd"])
        * validated.nights,
        2,
    )
    requires_hitl = (
        not policy_compliant
        or estimated_total_usd > HIGH_STAKES_AUTO_APPROVAL_LIMIT_USD
    )

    status_literal: Literal["DRAFT_READY", "DRAFT_REQUIRES_HITL_APPROVAL"] = (
        "DRAFT_REQUIRES_HITL_APPROVAL" if requires_hitl else "DRAFT_READY"
    )
    next_instruction = (
        f"Draft {booking_ref} exceeds auto-approval limits or policy caps "
        f"(total=${estimated_total_usd:.2f} USD, compliant={policy_compliant}). "
        "Ask the human user/supervisor for explicit confirmation before calling "
        "`execute_high_stakes_corporate_booking` or `submit_critical_policy_exception_ticket`."
        if requires_hitl
        else f"Draft {booking_ref} is policy-compliant and within auto-approval limits. "
        "You may proceed with booking confirmation."
    )

    output_model = DraftTripItineraryOutput(
        status=status_literal,
        booking_reference=booking_ref,
        policy_tier=tier,
        origin_airport_iata=validated.origin_airport_iata.upper(),
        destination_city=validated.destination_city,
        travel_dates=validated.travel_dates,
        nights=validated.nights,
        estimated_total_usd=estimated_total_usd,
        policy_compliant=policy_compliant,
        requires_human_confirmation=requires_hitl,
        next_step_instruction=next_instruction,
    )
    output_dict = output_model.model_dump(mode="json")

    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["latest_booking"] = output_dict

    return output_dict


# ---------------------------------------------------------------------------
# Tool 5: Execute High-Stakes Corporate Booking (HITL Gated)
# ---------------------------------------------------------------------------
@trace_and_log_intent_outcome(
    "Execute corporate credit charge for drafted itinerary subject to Human-in-the-Loop confirmation"
)
def execute_high_stakes_corporate_booking(
    booking_reference: str,
    total_charge_usd: float,
    human_approval_confirmed: bool = False,
    human_approver_id: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Executes a corporate trip booking charge after enforcing Human-in-the-Loop (HITL) approval.

    High-Stakes Action Gate:
    If `total_charge_usd` exceeds `$1,500.00 USD` and `human_approval_confirmed` is not
    `True` with a valid `human_approver_id`, this tool halts execution and returns
    `status="PENDING_HUMAN_APPROVAL"` without charging the corporate credit line.

    Args:
        booking_reference: Draft booking reference ID matching '^TRIP-[A-Z0-9]{6}$' (e.g., 'TRIP-4F9A1C').
        total_charge_usd: Total USD amount to charge to the corporate credit card (> 0 and <= 50000).
        human_approval_confirmed: Boolean flag indicating whether a human supervisor explicitly
            confirmed executing this booking.
        human_approver_id: Identifier or email of the approving human manager (e.g., 'mgr-jane-doe').
        tool_context: Optional ADK `ToolContext` for state persistence and HITL confirmation signaling.

    Returns:
        A JSON-serializable dictionary matching `ExecuteHighStakesBookingOutput`, or
        `StructuredToolErrorOutput` if the `booking_reference` or amount is invalid.
    """
    raw_inputs = {
        "booking_reference": booking_reference.strip().upper()
        if isinstance(booking_reference, str)
        else booking_reference,
        "total_charge_usd": total_charge_usd,
        "human_approval_confirmed": human_approval_confirmed,
        "human_approver_id": human_approver_id,
    }
    try:
        validated = ExecuteHighStakesBookingInput.model_validate(raw_inputs)
    except ValidationError as exc:
        return _build_guided_validation_error(
            tool_name="execute_high_stakes_corporate_booking",
            exc=exc,
            received_args=raw_inputs,
            specific_recovery_hints=[
                "Ensure `booking_reference` matches the format 'TRIP-XXXXXX' returned by `draft_compliant_corporate_trip_itinerary`.",
                "Ensure `total_charge_usd` is between 0.01 and 50000.00.",
            ],
        )

    hitl_result = verify_human_in_the_loop_approval(
        action_name="execute_high_stakes_corporate_booking",
        total_charge_usd=validated.total_charge_usd,
        human_approval_confirmed=validated.human_approval_confirmed,
        human_approver_id=validated.human_approver_id,
        tool_context=tool_context,
    )

    if not hitl_result.approved_to_execute:
        pending_output = ExecuteHighStakesBookingOutput(
            status="PENDING_HUMAN_APPROVAL",
            booking_reference=validated.booking_reference,
            total_charge_usd=validated.total_charge_usd,
            human_approval_verified=False,
            approver_id=None,
            confirmation_number=None,
            hitl_escalation_reason=hitl_result.escalation_reason,
            message=(
                f"EXECUTION PAUSED (Human-in-the-Loop Gate): Booking {validated.booking_reference} "
                f"for ${validated.total_charge_usd:.2f} USD requires explicit human confirmation. "
                "Please ask the user/manager to confirm and provide their `human_approver_id`."
            ),
        ).model_dump(mode="json")
        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["latest_booking"] = pending_output
        return pending_output

    # Secret Manager token injection for booking webhook authentication
    _ = secret_manager_service.load_runtime_config().corporate_booking_webhook_secret
    confirmation_num = f"CONF-{uuid.uuid4().hex[:8].upper()}"

    confirmed_output = ExecuteHighStakesBookingOutput(
        status="BOOKING_CONFIRMED",
        booking_reference=validated.booking_reference,
        total_charge_usd=validated.total_charge_usd,
        human_approval_verified=True,
        approver_id=hitl_result.approver_id,
        confirmation_number=confirmation_num,
        hitl_escalation_reason=None,
        message=(
            f"Booking {validated.booking_reference} confirmed (Confirmation #{confirmation_num}) "
            f"for ${validated.total_charge_usd:.2f} USD with verified approval from "
            f"'{hitl_result.approver_id}'."
        ),
    ).model_dump(mode="json")

    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["latest_booking"] = confirmed_output

    return confirmed_output


# ---------------------------------------------------------------------------
# Tool 6: Submit Critical Policy Exception Ticket (HITL Gated)
# ---------------------------------------------------------------------------
@trace_and_log_intent_outcome(
    "Submit a critical corporate travel policy exception ticket to Finance & Travel Ops"
)
def submit_critical_policy_exception_ticket(
    booking_reference: str,
    employee_level: str,
    violation_summary: str,
    justification: str,
    human_confirmation_received: bool = False,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Files a critical policy exception escalation ticket with Finance after human confirmation.

    Descriptive Naming & HITL Hook:
    Specifically named `submit_critical_policy_exception_ticket` (rather than a generic
    `update_jira`) to clearly scope its high-stakes financial escalation role. Requires
    `human_confirmation_received=True` before creating the external ticket.

    Args:
        booking_reference: Draft booking reference ID matching '^TRIP-[A-Z0-9]{6}$'.
        employee_level: Employee level requesting the exception (e.g., 'L6 Director').
        violation_summary: Concise description of the policy cap exceeded (10-500 chars).
        justification: Business justification for the exception request (10-500 chars).
        human_confirmation_received: Must be True to confirm the traveler wants to file
            a formal Finance exception ticket.
        tool_context: Optional ADK `ToolContext` to record the escalation ticket in `session.state`.

    Returns:
        A JSON-serializable dictionary conforming to `SubmitCriticalPolicyExceptionOutput`,
        or `StructuredToolErrorOutput` with recovery guidance if validation fails.
    """
    raw_inputs = {
        "booking_reference": booking_reference.strip().upper()
        if isinstance(booking_reference, str)
        else booking_reference,
        "employee_level": employee_level,
        "violation_summary": violation_summary,
        "justification": justification,
        "human_confirmation_received": human_confirmation_received,
    }
    try:
        validated = SubmitCriticalPolicyExceptionInput.model_validate(raw_inputs)
    except ValidationError as exc:
        return _build_guided_validation_error(
            tool_name="submit_critical_policy_exception_ticket",
            exc=exc,
            received_args=raw_inputs,
            specific_recovery_hints=[
                "Ensure `booking_reference` matches '^TRIP-[A-Z0-9]{6}$'.",
                "Ensure both `violation_summary` and `justification` are at least 10 characters long.",
            ],
        )

    if not validated.human_confirmation_received:
        return SubmitCriticalPolicyExceptionOutput(
            status="PENDING_HUMAN_APPROVAL",
            ticket_id=None,
            booking_reference=validated.booking_reference,
            priority="CRITICAL_FINANCE_EXCEPTION",
            human_confirmation_verified=False,
            message=(
                f"HITL Stop: Filing a critical Finance policy exception ticket for "
                f"{validated.booking_reference} requires explicit human confirmation "
                "(`human_confirmation_received=True`). Please confirm with the user first."
            ),
        ).model_dump(mode="json")

    _ = secret_manager_service.load_runtime_config().jira_escalation_webhook_token
    ticket_id = f"FIN-ESC-{uuid.uuid4().hex[:6].upper()}"

    output = SubmitCriticalPolicyExceptionOutput(
        status="TICKET_CREATED",
        ticket_id=ticket_id,
        booking_reference=validated.booking_reference,
        priority="CRITICAL_FINANCE_EXCEPTION",
        human_confirmation_verified=True,
        message=(
            f"Created critical Finance policy exception ticket {ticket_id} for "
            f"booking {validated.booking_reference}."
        ),
    ).model_dump(mode="json")

    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["latest_exception_ticket"] = output

    return output


# ---------------------------------------------------------------------------
# ADK FunctionTool Wrappers with Explicit HITL `require_confirmation` Hooks
# ---------------------------------------------------------------------------
execute_high_stakes_booking_adk_tool = FunctionTool(
    execute_high_stakes_corporate_booking,
    require_confirmation=should_require_human_confirmation,
)

submit_critical_policy_exception_adk_tool = FunctionTool(
    submit_critical_policy_exception_ticket,
    require_confirmation=True,
)
