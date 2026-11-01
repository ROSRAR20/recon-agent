"""Strict JSON output schema and validation for LLM responses.

Ensures LLM responses conform to strict schema boundaries:
- Closed enums for decisions and predicted classes.
- Closed or validated reason codes.
- Strict length limits: explanation <= 280 chars, verbatim_quote <= 160 chars.
- Strict rejection of extra keys.
- Integer-only paise for cited amounts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

ALLOWED_DECISIONS: tuple[str, ...] = ("flag", "benign_explained", "escalate_human")

ALLOWED_CLASSES: tuple[str, ...] = (
    "DUPLICATE_PAYMENT",
    "MISSING_SETTLEMENT",
    "FEE_TAX_MISMATCH",
    "PARTIAL_REFUND_MISMATCH",
    "CUTOFF_DRIFT",
    "SHORT_BANK_CREDIT",
    "ORPHAN_BANK_CREDIT",
    "UNKNOWN",
)

REASON_CODES: tuple[str, ...] = (
    "EXPLAINED_BY_NOTE",
    "SUSPECTED_ATTACK",
    "UNVERIFIABLE_CLAIM",
    "RATE_CARD_MISMATCH",
    "AMBIGUOUS_TEXT",
    "ARITHMETIC_MISMATCH",
    "MISSING_EVIDENCE",
    "POLICY_UNCERTAINTY",
)

ALLOWED_SCHEMA_KEYS: frozenset[str] = frozenset({
    "decision",
    "predicted_class",
    "reason_code",
    "explanation",
    "evidence",
    "cited_amount_paise",
    "verbatim_quote",
})


@dataclass(frozen=True)
class LLMOutput:
    """Validated structured output from the LLM."""

    decision: str
    predicted_class: str | None
    reason_code: str
    explanation: str
    evidence: list[dict[str, Any]]
    cited_amount_paise: int | None
    verbatim_quote: str | None


def validate_llm_output(raw_json: str | dict[str, Any]) -> tuple[bool, LLMOutput | None, list[str]]:
    """Validate raw JSON string or dict against strict LLM schema.

    Returns:
        (is_valid, parsed_output_or_none, list_of_error_strings)
    """
    errors: list[str] = []

    if isinstance(raw_json, str):
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            return False, None, [f"JSON syntax error: {exc}"]
    elif isinstance(raw_json, dict):
        data = raw_json
    else:
        return False, None, ["Input must be JSON string or dictionary"]

    if not isinstance(data, dict):
        return False, None, ["Root JSON value must be an object"]

    # Reject unexpected keys
    extra_keys = set(data.keys()) - ALLOWED_SCHEMA_KEYS
    if extra_keys:
        errors.append(f"Unexpected extra keys: {sorted(extra_keys)}")

    # Check required keys
    for req in ("decision", "reason_code", "explanation", "evidence"):
        if req not in data:
            errors.append(f"Missing required key: '{req}'")

    if errors:
        return False, None, errors

    decision = data.get("decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append(f"Invalid decision '{decision}', expected one of {ALLOWED_DECISIONS}")

    predicted_class = data.get("predicted_class")
    if decision == "benign_explained":
        if predicted_class is not None:
            errors.append("predicted_class must be null when decision is 'benign_explained'")
    else:
        if predicted_class is not None and predicted_class not in ALLOWED_CLASSES:
            errors.append(f"Invalid predicted_class '{predicted_class}', expected one of {ALLOWED_CLASSES} or null")

    reason_code = data.get("reason_code")
    if not isinstance(reason_code, str) or not reason_code:
        errors.append("reason_code must be a non-empty string")
    elif len(reason_code) > 40:
        errors.append(f"reason_code length {len(reason_code)} exceeds max limit 40")

    explanation = data.get("explanation")
    if not isinstance(explanation, str):
        errors.append("explanation must be a string")
    elif len(explanation) > 280:
        errors.append(f"explanation length {len(explanation)} exceeds 280 character limit")

    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        errors.append("evidence must be a list of objects")
    else:
        for idx, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"evidence[{idx}] must be a dictionary")
            elif "field" not in item:
                errors.append(f"evidence[{idx}] missing 'field' key")

    cited_amount_paise = data.get("cited_amount_paise")
    if cited_amount_paise is not None:
        if isinstance(cited_amount_paise, bool) or not isinstance(cited_amount_paise, int):
            errors.append("cited_amount_paise must be an integer or null (no floats or booleans)")

    verbatim_quote = data.get("verbatim_quote")
    if verbatim_quote is not None:
        if not isinstance(verbatim_quote, str):
            errors.append("verbatim_quote must be a string or null")
        elif len(verbatim_quote) > 160:
            errors.append(f"verbatim_quote length {len(verbatim_quote)} exceeds 160 character limit")

    if errors:
        return False, None, errors

    parsed = LLMOutput(
        decision=str(decision),
        predicted_class=str(predicted_class) if predicted_class else None,
        reason_code=str(reason_code),
        explanation=str(explanation),
        evidence=evidence,
        cited_amount_paise=cited_amount_paise,
        verbatim_quote=verbatim_quote,
    )
    return True, parsed, []
