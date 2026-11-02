"""Append-only audit logging for LLM invocations.

Logs execution telemetry to results/llm_audit.jsonl.
Strict security boundary:
- Never writes raw untrusted text or raw model outputs (hashes only) to prevent log injection.
- Raw outputs reside exclusively in the on-disk cache.
- Applies secret sanitization to guarantee no API keys or credentials reach the log.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import asdict, dataclass

from src.recon_agent.guard import sanitize_error


@dataclass(frozen=True)
class AuditRecord:
    """Single audit log entry."""

    timestamp: str
    bundle_hash: str
    prompt_hash: str
    model_id: str
    thinking_level: str
    raw_output_hash: str
    schema_valid: bool
    verifier_outcome: str
    verifier_failed_checks: list[str]
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    latency_ms: int
    cost_usd: float | None
    final_action: str
    estimated_cost_low_usd: float | None = None
    estimated_cost_high_usd: float | None = None
    usage_raw: dict[str, int | float] | None = None
    usage_flagged: bool = False
    raw_response_shape: dict[str, Any] | list[Any] | str | None = None


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
        thinking_level: str = "low",
        input_tokens: int | None = 0,
        output_tokens: int | None = 0,
        thinking_tokens: int | None = 0,
        latency_ms: int = 0,
        cost_usd: float | None = 0.0,
        estimated_cost_low_usd: float | None = None,
        estimated_cost_high_usd: float | None = None,
        usage_raw: dict[str, int | float] | None = None,
        usage_flagged: bool = False,
        raw_response_shape: dict[str, Any] | list[Any] | str | None = None,
    ) -> AuditRecord:
        """Write an audit entry containing only cryptographic hashes and verification metrics."""
        # Sanitize raw output before hashing just in case
        sanitized_raw = sanitize_error(raw_output)
        raw_output_hash = hashlib.sha256(sanitized_raw.encode("utf-8")).hexdigest()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # If estimated_cost_high_usd is provided, synchronize cost_usd
        if cost_usd == 0.0 and estimated_cost_high_usd is not None:
            effective_cost = estimated_cost_high_usd
        elif input_tokens is None or output_tokens is None:
            effective_cost = None
        else:
            effective_cost = cost_usd

        record = AuditRecord(
            timestamp=now_iso,
            bundle_hash=sanitize_error(bundle_hash),
            prompt_hash=sanitize_error(prompt_hash),
            model_id=sanitize_error(model_id),
            thinking_level=sanitize_error(thinking_level),
            raw_output_hash=raw_output_hash,
            schema_valid=schema_valid,
            verifier_outcome=verifier_outcome,
            verifier_failed_checks=list(verifier_failed_checks),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            latency_ms=latency_ms,
            cost_usd=effective_cost,
            final_action=final_action,
            estimated_cost_low_usd=estimated_cost_low_usd,
            estimated_cost_high_usd=estimated_cost_high_usd,
            usage_raw=usage_raw,
            usage_flagged=usage_flagged,
            raw_response_shape=raw_response_shape,
        )

        record_json = json.dumps(asdict(record))
        # Ensure final written line is sanitized
        record_json_sanitized = sanitize_error(record_json)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(record_json_sanitized + "\n")

        return record
