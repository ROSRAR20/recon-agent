"""Finalization pipeline separating raw LLM outputs from rule-based verification.

Transforms raw LLM text into audited B2 (unverified) and B3 (verified) decisions.
Results are persisted to results/finalized/<verifier_version>/ keyed by (bundle_id, run_id).
Cache entries are never modified by finalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from typing import Any

from src.recon_agent.guard import sanitize_error
from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.schema import validate_llm_output
from src.recon_agent.llm.verifier import Verifier, VerifierOutcome, VerifierResult


def compute_verifier_version() -> str:
    """Compute SHA-256 hash of verifier.py and schema.py combined."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    verifier_path = os.path.join(current_dir, "verifier.py")
    schema_path = os.path.join(current_dir, "schema.py")

    with open(verifier_path, "rb") as f:
        verifier_bytes = f.read()
    with open(schema_path, "rb") as f:
        schema_bytes = f.read()

    combined = verifier_bytes + schema_bytes
    return hashlib.sha256(combined).hexdigest()


@dataclass(frozen=True)
class FinalDecision:
    """Final reconciliation decision isolating raw, B2, and B3 actions."""

    bundle_id: str
    raw_decision: str | None
    raw_parse_status: str  # "ok" | "non_json" | "schema_invalid" | "empty_output"
    verifier_result: VerifierResult | None
    b2_action: str
    b3_action: str
    b3_explanation: str
    predicted_class: str | None = None
    raw_output_hash: str = ""
    verifier_version: str = ""
    run_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "bundle_id": self.bundle_id,
            "raw_decision": self.raw_decision,
            "raw_parse_status": self.raw_parse_status,
            "verifier_outcome": self.verifier_result.outcome.value if self.verifier_result else None,
            "verifier_reason_code": self.verifier_result.reason_code if self.verifier_result else None,
            "verifier_failed_checks": list(self.verifier_result.failed_checks) if self.verifier_result else None,
            "b2_action": self.b2_action,
            "b3_action": self.b3_action,
            "b3_explanation": self.b3_explanation,
            "predicted_class": self.predicted_class,
            "raw_output_hash": self.raw_output_hash,
            "verifier_version": self.verifier_version,
            "run_id": self.run_id,
        }


def finalize(
    bundle: CaseBundle,
    raw_text: str | None,
    verifier_version: str | None = None,
    verifier: Verifier | None = None,
    run_id: int = 0,
) -> FinalDecision:
    """Transform raw model output text into deterministic B2 and B3 decisions."""
    raw_str = raw_text or ""
    raw_output_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    if verifier_version is None:
        verifier_version = compute_verifier_version()
    if verifier is None:
        verifier = Verifier()

    # Empty or blocked model output
    if raw_text is None or not raw_text.strip():
        return FinalDecision(
            bundle_id=bundle.bundle_id,
            raw_decision=None,
            raw_parse_status="empty_output",
            verifier_result=None,
            b2_action="escalate_human",
            b3_action="escalate_human",
            b3_explanation="Model returned empty or blocked output; escalating for human review.",
            predicted_class=None,
            raw_output_hash=raw_output_hash,
            verifier_version=verifier_version,
            run_id=run_id,
        )

    # Check JSON validity
    try:
        json.loads(raw_text)
    except json.JSONDecodeError:
        return FinalDecision(
            bundle_id=bundle.bundle_id,
            raw_decision=None,
            raw_parse_status="non_json",
            verifier_result=None,
            b2_action="escalate_human",
            b3_action="escalate_human",
            b3_explanation="Model output is non-JSON; escalating for human review.",
            predicted_class=None,
            raw_output_hash=raw_output_hash,
            verifier_version=verifier_version,
            run_id=run_id,
        )

    # Validate output schema
    is_valid, parsed, errors = validate_llm_output(raw_text)
    if not is_valid or parsed is None:
        err_msg = "; ".join(errors) if errors else "SCHEMA_INVALID"
        return FinalDecision(
            bundle_id=bundle.bundle_id,
            raw_decision=None,
            raw_parse_status="schema_invalid",
            verifier_result=None,
            b2_action="escalate_human",
            b3_action="escalate_human",
            b3_explanation=f"Model output violates schema ({err_msg}); escalating for human review.",
            predicted_class=None,
            raw_output_hash=raw_output_hash,
            verifier_version=verifier_version,
            run_id=run_id,
        )

    # Valid schema: B2 accepts raw decision
    raw_decision = parsed.decision
    b2_action = raw_decision
    pred_class = parsed.predicted_class

    # Run verifier for B3
    ver_res = verifier.verify(bundle, parsed)
    if ver_res.outcome == VerifierOutcome.ACCEPT:
        b3_action = parsed.decision
        b3_explanation = parsed.explanation
    else:
        b3_action = "escalate_human"
        b3_explanation = f"Reconciliation discrepancy escalated for human review ({ver_res.reason_code})."

    return FinalDecision(
        bundle_id=bundle.bundle_id,
        raw_decision=raw_decision,
        raw_parse_status="ok",
        verifier_result=ver_res,
        b2_action=b2_action,
        b3_action=b3_action,
        b3_explanation=b3_explanation,
        predicted_class=pred_class,
        raw_output_hash=raw_output_hash,
        verifier_version=verifier_version,
        run_id=run_id,
    )


def get_finalized_filepath(
    verifier_version: str,
    run_id: int,
    population: str,
    base_dir: str = "results/finalized",
) -> str:
    """Derive standard filepath for finalized results."""
    return os.path.join(base_dir, verifier_version, f"finalized_{population}_run{run_id}.jsonl")


def write_finalized_results(
    decisions: list[FinalDecision],
    verifier_version: str,
    run_id: int,
    population: str,
    base_dir: str = "results/finalized",
) -> str:
    """Atomically write finalized results for a population and run_id.

    Raises ValueError if duplicate (bundle_id, run_id) keys are found.
    Never appends across invocations; writes to temp file then replaces.
    """
    seen_keys: set[tuple[str, int]] = set()
    for d in decisions:
        key = (d.bundle_id, d.run_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate key ({d.bundle_id}, {d.run_id}) in finalized decisions.")
        seen_keys.add(key)

    filepath = get_finalized_filepath(verifier_version, run_id, population, base_dir=base_dir)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    temp_filepath = f"{filepath}.tmp"
    with open(temp_filepath, "w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    os.replace(temp_filepath, filepath)
    return filepath


def load_finalized_results(
    verifier_version: str,
    run_id: int,
    population: str,
    base_dir: str = "results/finalized",
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load finalized results into a dictionary keyed by (bundle_id, run_id).

    Raises ValueError if duplicate keys are detected or if file is missing.
    """
    filepath = get_finalized_filepath(verifier_version, run_id, population, base_dir=base_dir)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Finalized results not found at {filepath}")

    results: dict[tuple[str, int], dict[str, Any]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            entry = json.loads(line_str)
            bundle_id = entry.get("bundle_id")
            r_id = entry.get("run_id", run_id)
            key = (bundle_id, r_id)
            if key in results:
                raise ValueError(
                    f"Duplicate key ({bundle_id}, {r_id}) at line {line_num} in {filepath}"
                )
            results[key] = entry

    return results
