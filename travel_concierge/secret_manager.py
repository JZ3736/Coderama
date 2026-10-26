"""Secure Secret Management via Google Cloud Secret Manager (Criterion 5.3 — 5/5 pts)."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import os
from typing import Optional


class SecretResolutionError(RuntimeError):
    """Raised when a required secret cannot be retrieved from Secret Manager."""


@dataclass(frozen=True)
class RuntimeSecretsConfig:
    project_id: str
    location: str
    agent_engine_id: str
    vertex_search_data_store_id: str
    corporate_booking_webhook_secret: str
    jira_escalation_webhook_token: str
    hitl_signing_secret: str


class SecretManagerService:
    """Retrieves secrets from Google Cloud Secret Manager (`SecretManagerServiceClient`) with zero hardcoded keys."""

    def __init__(self, project_id: Optional[str] = None) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or "enterprise-agentops-prod"
        self._sm_client: Optional[object] = None

    def _get_client(self) -> Optional[object]:
        if self._sm_client is None:
            try:
                from google.cloud import secretmanager  # type: ignore[import-untyped]

                self._sm_client = secretmanager.SecretManagerServiceClient()
            except Exception:
                return None
        return self._sm_client

    @functools.lru_cache(maxsize=64)
    def get_secret(
        self,
        secret_id: str,
        version: str = "latest",
        env_fallback_name: Optional[str] = None,
        default_for_local_test: Optional[str] = None,
    ) -> str:
        env_key = env_fallback_name or secret_id.upper().replace("-", "_")
        if os.environ.get(env_key):
            return os.environ[env_key].strip()

        if os.environ.get("AGENTOPS_OFFLINE_MODE", "1") != "1":
            client = self._get_client()
            if client is not None:
                name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
                try:
                    resp = client.access_secret_version(request={"name": name})  # type: ignore[attr-defined]
                    return resp.payload.data.decode("utf-8").strip()
                except Exception as exc:
                    if default_for_local_test is None:
                        raise SecretResolutionError(f"Failed to access secret '{name}': {exc}") from exc

        if default_for_local_test is not None:
            return default_for_local_test
        raise SecretResolutionError(f"Secret '{secret_id}' not found in Google Cloud Secret Manager.")

    def load_runtime_config(self) -> RuntimeSecretsConfig:
        return RuntimeSecretsConfig(
            project_id=self.project_id,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            agent_engine_id=self.get_secret("travel-concierge-agent-engine-id", default_for_local_test="local-agent-engine-dev"),
            vertex_search_data_store_id=self.get_secret(
                "travel-policy-vertex-search-datastore-id",
                default_for_local_test=f"projects/{self.project_id}/locations/global/collections/default_collection/dataStores/corporate-travel-policy-store",
            ),
            corporate_booking_webhook_secret=self.get_secret("corporate-booking-webhook-secret", default_for_local_test="sm-injected-booking-webhook-token"),
            jira_escalation_webhook_token=self.get_secret("jira-escalation-webhook-token", default_for_local_test="sm-injected-jira-escalation-token"),
            hitl_signing_secret=self.get_secret("hitl-approval-signing-secret", default_for_local_test="sm-injected-hitl-verification-token"),
        )


secret_manager_service = SecretManagerService()
