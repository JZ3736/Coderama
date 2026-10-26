output "agent_service_account_email" {
  description = "Email of the least-privilege Service Account for the ADK Agent Engine."
  value       = google_service_account.travel_concierge_agent_sa.email
}

output "vertex_search_data_store_id" {
  description = "Vertex AI Search Data Store ID for corporate travel policy retrieval."
  value       = google_discovery_engine_data_store.corporate_travel_policy_store.name
}

output "staging_bucket_uri" {
  description = "GCS staging bucket URI used by `adk deploy agent_engine`."
  value       = "gs://${google_storage_bucket.agent_engine_staging.name}"
}
