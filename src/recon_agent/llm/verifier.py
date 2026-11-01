"""Rule-based verification layer (V1-V7) for LLM outputs.

Enforces strict grounding, numeric consistency, verbatim quoting, and arithmetic verification.
Outputs:
- ACCEPT: LLM decision is verified and safe to use.
- REJECT: LLM violated grounding, schema, or fabricated data; fallback to escalate_human.
- ABSTAIN: Verifier cannot reliably confirm (e.g. multiple amounts, ambiguous text); fallback to escalate_human.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.schema import (
    ALLOWED_CLASSES,
    ALLOWED_DECISIONS,
    LLMOutput,
    validate_llm_output,
)


class VerifierOutcome(str, Enum):
    """Outcome of the rule-based verification check."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class VerifierResult:
    """Detailed result emitted by the verifier."""

    outcome: VerifierOutcome
    failed_checks: tuple[str, ...]
    reason_code: str
    details: str


def _extract_digit_amount_tokens(text: str) -> list[str]:
    """Extract numeric or currency amount tokens containing digits."""
    pattern = re.compile(
        r"(?:(?:INR|Rs\.?|₹)\s*)?\d+(?:,\d{3})*(?:\.\d{1,2})?(?:\s*(?:paise|paisa))?",
        re.IGNORECASE,
    )
    tokens = [m.group(0).strip() for m in pattern.finditer(text) if re.search(r"\d", m.group(0))]
    return tokens


def _strict_convert_amount_token(token: str) -> int | None:
    """Strictly convert a single amount token to integer paise.

    Returns None if the token cannot be strictly parsed.
    """
    token_clean = token.replace(",", "").strip()

    is_paise = bool(re.search(r"\b(?:paise|paisa)\b", token_clean, re.IGNORECASE))
    has_rupee_sym = bool(re.search(r"(?:INR|Rs\.?|₹)", token_clean, re.IGNORECASE))

    # Strip currency words/symbols
    num_part = re.sub(r"(?:INR|Rs\.?|₹|paise|paisa)", "", token_clean, flags=re.IGNORECASE).strip()
    if not num_part:
        return None

    try:
        if "." in num_part:
            val_float = float(num_part)
            return round(val_float * 100)
        else:
            val_int = int(num_part)
            if is_paise:
                return val_int
            if has_rupee_sym:
                return val_int * 100
            # If bare integer, return as-is (caller can check paise or rupees)
            return val_int
    except ValueError:
        return None


def _get_bundle_all_ids(bundle: CaseBundle) -> set[str]:
    """Collect all entity, merchant, and reference IDs in the bundle."""
    ids: set[str] = set()
    if bundle.entity_id:
        ids.add(bundle.entity_id)
    if bundle.merchant_id:
        ids.add(bundle.merchant_id)
    if bundle.bundle_id:
        ids.add(bundle.bundle_id)

    for k, v in bundle.facts.items():
        if isinstance(v, str) and (k.endswith("_id") or k == "entity_id" or k.startswith("utr")):
            ids.add(v)
    return ids


def _collect_bundle_numbers(bundle: CaseBundle) -> set[int]:
    """Collect all integer numbers present in bundle facts and untrusted text."""
    numbers: set[int] = set()

    # From facts
    for v in bundle.facts.values():
        if isinstance(v, int) and not isinstance(v, bool):
            numbers.add(v)
            if v % 100 == 0:
                numbers.add(v // 100)  # rupee equivalent

    # From untrusted text
    for text in bundle.untrusted_text.values():
        if text:
            for num_str in re.findall(r"\b\d+\b", text):
                try:
                    val = int(num_str)
                    numbers.add(val)
                except ValueError:
                    pass

    return numbers


class Verifier:
    """Rule-based verification engine enforcing checks V1 through V7."""

    def verify(
        self,
        bundle: CaseBundle,
        raw_output: str | dict[str, Any] | LLMOutput,
    ) -> VerifierResult:
        """Execute verification pipeline V1-V7 on LLM output for a given CaseBundle."""
        failed_checks: list[str] = []

        # --- Check V1: Schema compliance ---
        if isinstance(raw_output, LLMOutput):
            llm_output = raw_output
        else:
            is_valid, parsed, schema_errs = validate_llm_output(raw_output)
            if not is_valid or parsed is None:
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=("V1",),
                    reason_code="SCHEMA_INVALID",
                    details=f"Schema validation failed: {schema_errs}",
                )
            llm_output = parsed

        # --- Check V2: Enum membership in bundle allowed sets ---
        if llm_output.decision not in bundle.allowed_decisions:
            failed_checks.append("V2")
            return VerifierResult(
                outcome=VerifierOutcome.REJECT,
                failed_checks=tuple(failed_checks),
                reason_code="INVALID_DECISION",
                details=f"Decision '{llm_output.decision}' not in bundle allowed decisions {bundle.allowed_decisions}",
            )

        if llm_output.predicted_class is not None and llm_output.predicted_class not in bundle.allowed_classes:
            failed_checks.append("V2")
            return VerifierResult(
                outcome=VerifierOutcome.REJECT,
                failed_checks=tuple(failed_checks),
                reason_code="INVALID_CLASS",
                details=f"Predicted class '{llm_output.predicted_class}' not in bundle allowed classes",
            )

        # If decision is escalate_human, and V1/V2 pass, check if quote or fabricated numbers are severely violated
        # Prompt states: "if decision is escalate_human, always ACCEPT" under V7 arithmetic check,
        # but let's check V3..V6 grounding first.
        # If model explicitly escalates with SUSPECTED_ATTACK or UNVERIFIABLE_CLAIM, it is safe to ACCEPT.
        if llm_output.decision == "escalate_human":
            # Escalate human is safe by default
            return VerifierResult(
                outcome=VerifierOutcome.ACCEPT,
                failed_checks=(),
                reason_code=llm_output.reason_code or "ESCALATION_ACCEPTED",
                details="Decision is escalate_human; verified safe.",
            )

        # --- Check V3: Grounded references in evidence ---
        bundle_facts = bundle.facts
        bundle_untrusted = bundle.untrusted_text

        for ev in llm_output.evidence:
            field = ev.get("field", "")
            # Check if field exists in facts, untrusted_text, or top-level bundle
            field_clean = field.replace("facts.", "").replace("untrusted_text.", "")
            exists = (
                field in bundle_facts
                or field_clean in bundle_facts
                or field in bundle_untrusted
                or field_clean in bundle_untrusted
                or hasattr(bundle, field)
            )
            if not exists:
                failed_checks.append("V3")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="UNGROUNDED_EVIDENCE_FIELD",
                    details=f"Cited evidence field '{field}' does not exist in bundle.",
                )

        # --- Check V4: Verbatim quote substring check ---
        if llm_output.verbatim_quote is not None:
            quote = llm_output.verbatim_quote
            found_in_text = False
            for text_val in bundle_untrusted.values():
                if text_val and quote in text_val:
                    found_in_text = True
                    break
            if not found_in_text:
                failed_checks.append("V4")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="FABRICATED_QUOTE",
                    details=f"Verbatim quote '{quote}' is not a substring of any untrusted text in bundle.",
                )

        # --- Check V5: Amount consistency & strict quote conversion ---
        is_quote_sourced = False
        if llm_output.cited_amount_paise is not None:
            cited_amt = llm_output.cited_amount_paise
            bundle_numbers = _collect_bundle_numbers(bundle)

            # Check if quote-sourced:
            if llm_output.verbatim_quote:
                quote = llm_output.verbatim_quote
                amt_tokens = _extract_digit_amount_tokens(quote)
                # Amendment 3: quote-sourced amount verified only if quote contains EXACTLY ONE amount token
                if len(amt_tokens) != 1:
                    # Could be 0 (e.g. number words) or >= 2 amounts -> ABSTAIN
                    failed_checks.append("V5")
                    return VerifierResult(
                        outcome=VerifierOutcome.ABSTAIN,
                        failed_checks=tuple(failed_checks),
                        reason_code="AMBIGUOUS_QUOTE_AMOUNTS",
                        details=f"Quote contains {len(amt_tokens)} amount tokens (expected exactly 1); abstaining.",
                    )
                parsed_amt = _strict_convert_amount_token(amt_tokens[0])
                if parsed_amt is None:
                    failed_checks.append("V5")
                    return VerifierResult(
                        outcome=VerifierOutcome.ABSTAIN,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNPARSEABLE_QUOTE_AMOUNT",
                        details=f"Strict converter failed to parse token '{amt_tokens[0]}'; abstaining.",
                    )
                # Verify match
                if parsed_amt == cited_amt or parsed_amt * 100 == cited_amt:
                    is_quote_sourced = True
                else:
                    failed_checks.append("V5")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="AMOUNT_MISMATCH_WITH_QUOTE",
                        details=f"Cited amount {cited_amt} paise does not match quote token '{amt_tokens[0]}' ({parsed_amt}).",
                    )
            elif cited_amt not in bundle_numbers:
                failed_checks.append("V5")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="UNGROUNDED_AMOUNT",
                    details=f"Cited amount {cited_amt} paise does not appear in bundle facts or numbers.",
                )

        # --- Check V6: No fabricated standalone numbers in explanation ---
        # Amendment 4: Before extracting number tokens, remove tokens equal to any ref or id in bundle;
        # consider standalone numbers only. Standard small integers (0..100) are allowed.
        bundle_ids = _get_bundle_all_ids(bundle)
        explanation_clean = llm_output.explanation

        # Remove known IDs from explanation
        for bid in sorted(bundle_ids, key=len, reverse=True):
            explanation_clean = explanation_clean.replace(bid, " ")

        # Extract standalone numbers: digits possibly with commas/decimals
        extracted_num_tokens = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", explanation_clean)
        allowed_bundle_numbers = _collect_bundle_numbers(bundle)
        if llm_output.cited_amount_paise is not None:
            allowed_bundle_numbers.add(llm_output.cited_amount_paise)
            if llm_output.cited_amount_paise % 100 == 0:
                allowed_bundle_numbers.add(llm_output.cited_amount_paise // 100)

        for num_tok in extracted_num_tokens:
            raw_str = num_tok.replace(",", "")
            try:
                if "." in raw_str:
                    val_float = float(raw_str)
                    val_paise = round(val_float * 100)
                    if val_float <= 100 or val_paise in allowed_bundle_numbers:
                        continue
                else:
                    val_int = int(raw_str)
                    if 0 <= val_int <= 100 or val_int in allowed_bundle_numbers:
                        continue
            except ValueError:
                pass

            # If we get here, the number token is ungrounded / fabricated
            failed_checks.append("V6")
            return VerifierResult(
                outcome=VerifierOutcome.REJECT,
                failed_checks=tuple(failed_checks),
                reason_code="FABRICATED_NUMBER_IN_EXPLANATION",
                details=f"Explanation contains ungrounded standalone number '{num_tok}'.",
            )

        # --- Check V7: Arithmetic check ---
        # if decision is benign_explained, cited_amount_paise must equal refund difference
        if llm_output.decision == "benign_explained":
            refund_diff = bundle.facts.get("refund_difference_paise")
            if refund_diff is None:
                # Calculate if debit and amount are available
                debit = bundle.facts.get("debit_paise", 0)
                amt = bundle.facts.get("amount_paise", 0)
                refund_diff = debit - amt

            if llm_output.cited_amount_paise is None or abs(llm_output.cited_amount_paise) != abs(refund_diff):
                failed_checks.append("V7")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="ARITHMETIC_MISMATCH",
                    details=(
                        f"Cited amount {llm_output.cited_amount_paise} does not match refund difference "
                        f"{refund_diff} paise."
                    ),
                )

        elif llm_output.decision == "flag":
            # Must have supporting evidence or cited amount
            if not llm_output.evidence and llm_output.cited_amount_paise is None:
                failed_checks.append("V7")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="UNSUPPORTED_FLAG",
                    details="Flag decision lacks supporting evidence or cited amount.",
                )

        return VerifierResult(
            outcome=VerifierOutcome.ACCEPT,
            failed_checks=(),
            reason_code=llm_output.reason_code or "VERIFIED_ACCEPT",
            details="All verifier checks passed.",
        )

