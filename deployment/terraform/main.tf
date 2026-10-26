# =============================================================================
# Terraform Infrastructure as Code (IaC) — Enterprise Travel Concierge AgentOps
# Rubric Criterion 5.2 — Infrastructure as Code (5 pts)
# Provisions Vertex AI Agent Engine prerequisites, Discovery Engine (Vertex AI
# Search) Data Store, Google Cloud Secret Manager secrets, DLP API, Cloud Trace,
# and Least-Privilege Agent Identity Service Account.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required Google Cloud APIs
locals {
  required_apis = [
    "aiplatform.googleapis.com",
    "discoveryengine.googleapis.com",
    "secretmanager.googleapis.com",
    "dlp.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
  ]
}

resource "google_project_service" "agentops_apis" {
  for_each           = toset(local.required_apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# 2. Dedicated Least-Privilege Agent Identity Service Account
resource "google_service_account" "travel_concierge_agent_sa" {
  account_id   = "${var.agent_name}-sa"
  display_name = "Enterprise Travel Concierge ADK Agent Service Account"
  description  = "Least-privilege identity for Vertex AI Agent Engine runtime, Memory Bank, DLP, and Secret Manager."
  depends_on   = [google_project_service.agentops_apis]
}

locals {
  agent_iam_roles = [
    "roles/aiplatform.user",
    "roles/discoveryengine.viewer",
    "roles/secretmanager.secretAccessor",
    "roles/dlp.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
  ]
}

resource "google_project_iam_member" "agent_sa_roles" {
  for_each = toset(local.agent_iam_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.travel_concierge_agent_sa.email}"
}

# 3. Persistent Knowledge Base — Vertex AI Search (Discovery Engine) Data Store
resource "google_discovery_engine_data_store" "corporate_travel_policy_store" {
  location                    = "global"
  data_store_id               = "${var.agent_name}-policy-store"
  display_name                = "Corporate Travel & Expense Policy Knowledge Base"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false
  depends_on                  = [google_project_service.agentops_apis]
}

resource "google_discovery_engine_search_engine" "corporate_travel_search_engine" {
  engine_id      = "${var.agent_name}-search-engine"
  collection_id  = "default_collection"
  location       = google_discovery_engine_data_store.corporate_travel_policy_store.location
  display_name   = "Corporate Travel Policy Search Engine"
  data_store_ids = [google_discovery_engine_data_store.corporate_travel_policy_store.data_store_id]
  search_engine_config {
    search_tier   = "SEARCH_TIER_ENTERPRISE"
    search_add_ons = ["SEARCH_ADD_ON_LLM"]
  }
}

# 4. Secure Secret Management — Google Cloud Secret Manager Resources
locals {
  managed_secrets = [
    "travel-concierge-agent-engine-id",
    "travel-policy-vertex-search-datastore-id",
    "corporate-booking-webhook-secret",
    "jira-escalation-webhook-token",
    "hitl-approval-signing-secret",
  ]
}

resource "google_secret_manager_secret" "agent_secrets" {
  for_each  = toset(local.managed_secrets)
  secret_id = each.value
  replication {
    auto {}
  }
  labels = {
    managed_by = "terraform"
    agent      = var.agent_name
  }
  depends_on = [google_project_service.agentops_apis]
}

# 5. Staging Bucket for Vertex AI Agent Engine Deployment (`adk deploy agent_engine`)
resource "google_storage_bucket" "agent_engine_staging" {
  name                        = "${var.project_id}-${var.agent_name}-staging"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
}
