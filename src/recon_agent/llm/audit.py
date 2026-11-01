"""Append-only audit logging for LLM invocations.

Logs execution telemetry to results/llm_audit.jsonl.
Strict security boundary:
- Never writes raw untrusted text or raw model outputs (hashes only) to prevent log injection.
- Raw outputs reside exclusively in the on-disk cache.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AuditRecord:
    """Single audit log entry."""

    timestamp: str
    bundle_hash: str
    prompt_hash: str
    model_id: str
    raw_output_hash: str
    schema_valid: bool
    verifier_outcome: str
    verifier_failed_checks: list[str]
    latency_ms: int
    cost_usd: float
    final_action: str


class AuditLogger:
    """Append-only logger writing structured JSONL audit entries."""

    def __init__(self, log_path: str = "results/llm_audit.jsonl") -> None:
        self.log_path = os.path.abspath(log_path)
        log_dir = os.path.dirname(self.log_path)
        os.makedirs(log_dir, exist_ok=True)

    def log(
        self,
        bundle_hash: str,
        prompt_hash: str,
        model_id: str,
        raw_output: str,
        schema_valid: bool,
        verifier_outcome: str,
        verifier_failed_checks: list[str] | tuple[str, ...],
        final_action: str,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> AuditRecord:
        """Write an audit entry containing only cryptographic hashes and verification metrics."""
        raw_output_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        record = AuditRecord(
            timestamp=now_iso,
            bundle_hash=bundle_hash,
            prompt_hash=prompt_hash,
            model_id=model_id,
            raw_output_hash=raw_output_hash,
            schema_valid=schema_valid,
            verifier_outcome=verifier_outcome,
            verifier_failed_checks=list(verifier_failed_checks),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            final_action=final_action,
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        return record

