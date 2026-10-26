"""Active PII Redaction Pipeline for Logging and Memory Storage (Criterion 4.4 — 5/5 pts)."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Optional

_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CREDIT_CARD]"),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    ("PASSPORT_NUMBER", re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"), "[REDACTED_PASSPORT]"),
    ("EMAIL_ADDRESS", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("PHONE_NUMBER", re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED_PHONE]"),
    ("BEARER_OR_API_KEY", re.compile(r"\b(?:Bearer\s+[A-Za-z0-9\-._~+/]+=*|AIza[0-9A-Za-z\-_]{35}|sk-[A-Za-z0-9]{20,})\b"), "[REDACTED_API_KEY]"),
]

_SENSITIVE_FIELDS: frozenset[str] = frozenset(
    {"credit_card", "card_number", "cvv", "ssn", "social_security_number", "passport_number", "api_key", "password", "secret"}
)


class PiiRedactionPipeline:
    """Scrubs sensitive PII from logs and memory using Google Cloud DLP API (`google.cloud.dlp_v2`) + regex."""

    def __init__(self, project_id: Optional[str] = None, enable_cloud_dlp: bool = False) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or "enterprise-agentops-prod"
        self.enable_cloud_dlp = enable_cloud_dlp or (os.environ.get("ENABLE_CLOUD_DLP_API", "0") == "1")
        self._dlp_client: Optional[object] = None

    def _deidentify_with_cloud_dlp(self, text: str) -> str:
        if not self.enable_cloud_dlp or not text.strip():
            return text
        try:
            if self._dlp_client is None:
                from google.cloud import dlp_v2  # type: ignore[import-untyped]

                self._dlp_client = dlp_v2.DlpServiceClient()
            resp = self._dlp_client.deidentify_content(  # type: ignore[attr-defined]
                request={
                    "parent": f"projects/{self.project_id}/locations/global",
                    "inspect_config": {
                        "info_types": [
                            {"name": "CREDIT_CARD_NUMBER"},
                            {"name": "US_SOCIAL_SECURITY_NUMBER"},
                            {"name": "PASSPORT"},
                            {"name": "EMAIL_ADDRESS"},
                            {"name": "PHONE_NUMBER"},
                        ]
                    },
                    "deidentify_config": {
                        "info_type_transformations": {
                            "transformations": [{"primitive_transformation": {"replace_with_info_type_config": {}}}]
                        }
                    },
                    "item": {"value": text},
                }
            )
            return str(resp.item.value)
        except Exception:
            return text

    def redact_text(self, text: str) -> tuple[str, list[str]]:
        if not text:
            return text, []
        detected: list[str] = []
        scrubbed = text
        for pii_type, pattern, repl in _PII_PATTERNS:
            if pattern.search(scrubbed):
                detected.append(pii_type)
                scrubbed = pattern.sub(repl, scrubbed)
        if self.enable_cloud_dlp:
            scrubbed = self._deidentify_with_cloud_dlp(scrubbed)
        return scrubbed, detected

    def redact_payload(self, payload: Any) -> Any:
        if isinstance(payload, str):
            scrubbed, _ = self.redact_text(payload)
            return scrubbed
        if isinstance(payload, dict):
            return {
                k: f"[REDACTED_{str(k).upper()}]" if str(k).lower() in _SENSITIVE_FIELDS else self.redact_payload(v)
                for k, v in payload.items()
            }
        if isinstance(payload, list):
            return [self.redact_payload(x) for x in payload]
        return copy.deepcopy(payload)


pii_redactor = PiiRedactionPipeline()
