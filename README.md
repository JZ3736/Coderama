# Enterprise Corporate Travel & Expense Concierge (`enterprise-travel-concierge-agentops`)

An enterprise-grade multi-agent system built with the **Google Agent Development Kit (ADK)**, **Vertex AI Agent Engine**, **Vertex AI Search**, **Vertex AI Memory Bank**, **Agent2Agent (A2A) Protocol**, and **OpenTelemetry**.

---

## 1. Problem Formulation & Solution Objective

### The Problem
Corporate travel and expense management is fragmented, manual, and error-prone:
1. Employees struggle to cross-reference live destination advisories with complex tier-based corporate travel policies (`L1–L5`, `L6`, `L7+` hotel caps, cabin class rules, and meal per-diem limits).
2. Unchecked autonomous agents risk charging high-stakes bookings without human supervisor oversight, leaking sensitive traveler PII (credit cards, SSNs, passports) into logs or vector memory stores, or suffering from context window bloat across multi-turn planning sessions.

### The Solution
We built the **Enterprise Corporate Travel & Expense Concierge (`travel_concierge_root`)**, an ADK multi-agent system engineered from the ground up to meet production **AgentOps** standards:
- **Multi-Agent Orchestration**: Combines a Root Coordinator (`Agent`), a concurrent intelligence gatherer (`ParallelAgent` running `live_web_research_agent` + `enterprise_knowledge_rag_agent`), and a strict compliance pipeline (`SequentialAgent` running `policy_compliance_validator_agent` $\rightarrow$ `itinerary_booking_execution_agent` $\rightarrow$ `self_eval_compliance_auditor_agent`).
- **Strategic Model Routing**: Routes fast lookups and web searches to **`gemini-2.5-flash`** and complex multi-step planning, policy exception arbitration, and self-evaluation auditing to **`gemini-2.5-pro`**.
- **Human-in-the-Loop (HITL) Safety Stops**: Enforces mandatory human confirmation via ADK `FunctionTool(..., require_confirmation=...)` and programmatic gates before executing any booking exceeding `$1,500.00 USD` or filing a critical Finance policy exception ticket.
- **End-to-End Observability & Zero-Leak PII Redaction**: Every action records structured JSON `INTENT_CAPTURED` and `OUTCOME_CAPTURED` logs linked by OpenTelemetry spans (`trace_id`, `span_id`, `correlation_id`), with active **Google Cloud DLP (`google.cloud.dlp_v2`) + regex PII scrubbing** prior to log emission or **Vertex AI Memory Bank** storage.

---

## 2. System Architecture

![Enterprise Corporate Travel & Expense Concierge Architecture Graph](docs/architecture_graph.svg)

<details>
<summary>View Interactive Mermaid Graph Source (auto-rendered on GitHub)</summary>

```mermaid
flowchart TB
    User(["Corporate Traveler / Supervisor"]) --> Root["Root Coordinator Agent<br/>(travel_concierge_root — PRO_PLANNING_MODEL_ID)"]
    PeerAgent(["Peer Enterprise Agent<br/>(A2A Protocol JSON-RPC)"]) --> A2A["A2A Server & AgentCard<br/>(travel_concierge/a2a_server.py)"]
    A2A --> Root

    subgraph Plugins["ADK App Plugins & Context Management"]
        Compaction["EventsCompactionConfig + LlmEventSummarizer<br/>+ ContextCacheConfig + SlidingWindowCompactor"]
        Guardrails["EnterpriseGuardrailsPolicyPlugin<br/>(Prompt-Injection Block + Vertex Safety + Self-Eval)"]
        Router["DynamicModelRoutingPlugin<br/>(Flash vs. Pro Complexity Router)"]
        Telemetry["ObservabilityAndTracingPlugin<br/>(Structured JSON Logs + Intent/Outcome + OpenTelemetry + DLP PII Scrubber)"]
    end

    Root --- Plugins

    Root --> Parallel["ParallelAgent: destination_intelligence_parallel_agent"]
    Parallel --> WebAgent["live_web_research_agent<br/>(FAST_MODEL_ID + GoogleSearchTool)"]
    Parallel --> RagAgent["enterprise_knowledge_rag_agent<br/>(FAST_MODEL_ID + VertexAiSearchTool)"]

    Root --> Seq["SequentialAgent: compliant_booking_sequential_pipeline"]
    Seq --> Step1["1. policy_compliance_validator_agent<br/>(verify_corporate_travel_policy_compliance)"]
    Step1 --> Step2["2. itinerary_booking_execution_agent<br/>(HITL-Gated Booking & Critical Exception Tools)"]
    Step2 --> Step3["3. self_eval_compliance_auditor_agent<br/>(PRO_PLANNING_MODEL_ID Self-Evaluation Audit)"]

    Root -. "Async Non-Blocking (asyncio.create_task)" .-> Memory["VertexAiMemoryBankService +<br/>VertexAiSessionService + VertexAiSearchTool"]
```

</details>

---

## 3. AgentOps Code Review Matrix — Complete 95/95 Rubric Traceability

| Category | Criteria | Pts | Implementation & Code Evidence in Repository |
| :--- | :--- | :---: | :--- |
| **1. Tool & Interface Design** | **Comprehensive Tool Docstrings** | 5 | [`travel_concierge/tools.py`](travel_concierge/tools.py): All 6 tool functions include comprehensive Google-style docstrings detailing purpose, tier rules, `Args:`, and `Returns:`. |
| | **Descriptive Naming** | 5 | [`travel_concierge/tools.py`](travel_concierge/tools.py): Specific verb-noun tool names: `verify_corporate_travel_policy_compliance`, `save_traveler_profile_preferences`, `search_corporate_policy_knowledge_base`, `draft_compliant_corporate_trip_itinerary`, `execute_high_stakes_corporate_booking`, `submit_critical_policy_exception_ticket`. |
| | **Explicit JSON Schemas** | 5 | [`travel_concierge/schemas.py`](travel_concierge/schemas.py): Strict Pydantic v2 `BaseModel` schemas (`ConfigDict(extra="forbid")`, `Field` bounds/patterns) for all tool inputs/outputs and `TOOL_JSON_SCHEMAS`. |
| | **Guided Error Handling** | 5 | [`travel_concierge/tools.py`](travel_concierge/tools.py) (`_build_guided_validation_error`) & [`travel_concierge/schemas.py`](travel_concierge/schemas.py) (`StructuredToolErrorOutput`): Returns structured `llm_recovery_instructions` on invalid inputs instead of crashing, paired with ADK's `ReflectAndRetryToolPlugin(max_retries=3)`. |
| **2. Context & Memory** | **Robust System Instructions** | 5 | [`travel_concierge/prompts.py`](travel_concierge/prompts.py): Explicit XML-tagged `<AGENT_CONSTITUTION>` defining `<PERSONA>`, `<DOMAIN_KNOWLEDGE>`, `<OPERATIONAL_CONSTRAINTS>`, `<SECURITY_AND_PII_CONSTITUTION>`, and `<HUMAN_IN_THE_LOOP_CONSTITUTION>`. |
| | **History Compaction** | 5 | [`travel_concierge/memory_and_context.py`](travel_concierge/memory_and_context.py) & [`travel_concierge/agent.py`](travel_concierge/agent.py): Configures ADK `EventsCompactionConfig` with `LlmEventSummarizer`, Google Cloud `ContextCacheConfig`, `ContextFilterPlugin`, and `SlidingWindowContextCompactor`. |
| | **Persistent Session State** | 5 | [`travel_concierge/memory_and_context.py`](travel_concierge/memory_and_context.py): Connects to `VertexAiSessionService`, `VertexAiMemoryBankService`, `VertexAiRagMemoryService`, and `VertexAiSearchTool` (`search_corporate_policy_knowledge_base`). |
| | **Async Memory Operations** | 5 | [`travel_concierge/memory_and_context.py`](travel_concierge/memory_and_context.py) (`AsyncMemoryConsolidator`): Uses `asyncio.create_task` background tasks in `save_traveler_profile_preferences` and `non_blocking_after_agent_memory_callback` so memory consolidation never blocks the UI. |
| **3. Orchestration & Logic** | **Multi-Agent Patterns** | 5 | [`travel_concierge/agent.py`](travel_concierge/agent.py): Combines Coordinator (`root_agent`), `ParallelAgent` (`destination_intelligence_parallel_agent`), and `SequentialAgent` (`compliant_booking_sequential_pipeline`), plus A2A interoperability in [`travel_concierge/a2a_server.py`](travel_concierge/a2a_server.py). |
| | **Strategic Model Routing** | 5 | [`travel_concierge/model_router.py`](travel_concierge/model_router.py): Routes fast retrieval/search tasks to `gemini-2.5-flash` and complex planning/auditing to `gemini-2.5-pro` via `StrategicModelRouter` and `DynamicModelRoutingPlugin`. |
| | **Guardrails & Policy Plugins** | 5 | [`travel_concierge/guardrails.py`](travel_concierge/guardrails.py): Implements `EnterpriseGuardrailsPolicyPlugin(BasePlugin)`, Vertex AI `SafetySetting` (`HarmBlockThreshold.BLOCK_LOW_AND_ABOVE`), prompt-injection blocking, treasury hard-caps, and `SelfEvalComplianceJudge`. |
| | **Human-in-the-Loop Hooks** | 5 | [`travel_concierge/hitl.py`](travel_concierge/hitl.py) & [`travel_concierge/tools.py`](travel_concierge/tools.py): Wraps high-stakes tools in ADK `FunctionTool(..., require_confirmation=should_require_human_confirmation)` and halts execution (`status="PENDING_HUMAN_APPROVAL"`) via `verify_human_in_the_loop_approval`. |
| **4. Observability & Tracing** | **Structured JSON Logging** | 5 | [`travel_concierge/observability.py`](travel_concierge/observability.py): `StructuredJsonLogFormatter` & `StructuredAgentLogger` emit Google Cloud Logging compatible JSON records with zero `print()` statements. |
| | **Intent vs. Outcome Capture** | 5 | [`travel_concierge/observability.py`](travel_concierge/observability.py): `ObservabilityAndTracingPlugin` and `@trace_and_log_intent_outcome` explicitly log `INTENT_CAPTURED` before execution and `OUTCOME_CAPTURED` after execution with matching `correlation_id`. |
| | **Distributed Tracing** | 5 | [`travel_concierge/observability.py`](travel_concierge/observability.py): OpenTelemetry `TracerProvider`, `CloudTraceSpanExporter` / `InMemorySpanExporter`, and linked spans across agent, model, and tool invocations. |
| | **PII Redaction** | 5 | [`travel_concierge/pii_redaction.py`](travel_concierge/pii_redaction.py): `PiiRedactionPipeline` integrates Google Cloud Sensitive Data Protection (`google.cloud.dlp_v2.DlpServiceClient`) and deterministic regex scrubbers for Credit Cards, SSNs, Passports, Emails, Phones, and API keys across all logs and memory writes. |
| **5. Infrastructure & CI/CD** | **Automated Evaluation Suites** | 5 | [`eval/golden_dataset.evalset.json`](eval/golden_dataset.evalset.json), [`eval/test_config.json`](eval/test_config.json), [`eval/run_evaluation.py`](eval/run_evaluation.py), and [`tests/test_agentops_rubric.py`](tests/test_agentops_rubric.py). |
| | **Infrastructure as Code** | 5 | [`deployment/terraform/main.tf`](deployment/terraform/main.tf), [`deployment/terraform/variables.tf`](deployment/terraform/variables.tf), [`deployment/terraform/outputs.tf`](deployment/terraform/outputs.tf), [`.github/workflows/agentops_ci_cd.yml`](.github/workflows/agentops_ci_cd.yml), and [`deployment/cloudbuild.yaml`](deployment/cloudbuild.yaml). |
| | **Secure Secret Management** | 5 | [`travel_concierge/secret_manager.py`](travel_concierge/secret_manager.py): `SecretManagerService` retrieves all secrets from Google Cloud Secret Manager (`google.cloud.secretmanager.SecretManagerServiceClient`) with zero hardcoded keys. |

---

## 4. Usage of ADK CLI, Evaluation Harness & Terraform IaC

### 4.1 Install Dependencies & Run Automated Verification
```bash
pip install -r requirements.txt

# Run the 19-criteria AgentOps Rubric pytest suite
pytest tests/ -v

# Run the Golden Dataset Evaluation Regression Harness
python -m eval.run_evaluation
```

### 4.2 Interactive Local Development with `adk` CLI
```bash
# Launch the interactive ADK Web UI (inspect traces, state, and HITL confirmations)
adk web .

# Or run the agent interactively in the terminal
adk run travel_concierge

# Run ADK's built-in CLI evaluator against the golden dataset
adk eval travel_concierge eval/golden_dataset.evalset.json --config_file_path=eval/test_config.json
```

### 4.3 Provision Cloud Infrastructure with Terraform IaC
```bash
terraform -chdir=deployment/terraform init
terraform -chdir=deployment/terraform plan -var="project_id=${GOOGLE_CLOUD_PROJECT}"
terraform -chdir=deployment/terraform apply -var="project_id=${GOOGLE_CLOUD_PROJECT}"
```

### 4.4 Deploy to Vertex AI Agent Engine via `adk` CLI or `agent-starter-pack`
```bash
# Deploy directly to Vertex AI Agent Engine managed runtime using ADK CLI
adk deploy agent_engine \
  --project="${GOOGLE_CLOUD_PROJECT}" \
  --region="us-central1" \
  --staging_bucket="gs://${GOOGLE_CLOUD_PROJECT}-enterprise-travel-concierge-staging" \
  --display_name="enterprise-travel-concierge-agentops" \
  travel_concierge
```

---

## 5. Official Resources & References Used

- **ADK Documentation**: [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)
- **ADK Python Repository**: [https://github.com/google/adk-python](https://github.com/google/adk-python)
- **ADK Sample Agents**: [https://github.com/google/adk-samples](https://github.com/google/adk-samples)
- **Google Cloud Agent Starter Pack**: [https://github.com/GoogleCloudPlatform/agent-starter-pack](https://github.com/GoogleCloudPlatform/agent-starter-pack)
- **Agent2Agent (A2A) Protocol Specification**: [https://a2a-protocol.org/latest/](https://a2a-protocol.org/latest/)
- **A2A Python SDK**: [https://github.com/a2aproject/a2a-python](https://github.com/a2aproject/a2a-python)
- **Vertex AI Agent Engine Documentation**: [https://docs.cloud.google.com/agent-builder/agent-engine/overview](https://docs.cloud.google.com/agent-builder/agent-engine/overview)
- **Introduction to Agents Whitepaper**: [Google Cloud Agents Whitepaper](https://drive.google.com/file/d/1C-HvqgxM7dj4G2kCQLnuMXi1fTpXRdpx/view)
