variable "project_id" {
  description = "Google Cloud Project ID hosting Vertex AI Agent Engine and Search."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Vertex AI Agent Engine runtime."
  type        = string
  default     = "us-central1"
}

variable "agent_name" {
  description = "Canonical identifier for the ADK multi-agent system."
  type        = string
  default     = "enterprise-travel-concierge"
}
