"""Rule-based verification layer (V1-V7) for LLM outputs.

Enforces strict grounding, numeric consistency, verbatim quoting, and arithmetic verification.
Outputs:
- ACCEPT: LLM decision is verified and safe to use.
- REJECT: LLM violated grounding, schema, or fabricated data; fallback to escalate_human.
- ABSTAIN: Verifier cannot reliably confirm (e.g. multiple amounts, ambiguous text, bare numbers without units);
  fallback to escalate_human.

All checks V1-V6 run on EVERY decision, including escalate_human.
On REJECT or ABSTAIN, the pipeline keeps the final action as escalate_human, but the explanation
is replaced by a deterministic verifier template to prevent presenting fabricated text to human reviewers.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import Enum
import re
from typing import Any

from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.schema import (
    LLMOutput,
    validate_llm_output,
)


class VerifierOutcome(str, Enum):
    """Outcome of the rule-based verification check."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class VerifierResult:
    """Detailed result emitted by the verifier."""

    def __init__(
        self,
        outcome: VerifierOutcome,
        failed_checks: tuple[str, ...] = (),
        reason_code: str = "",
        details: str = "",
    ) -> None:
        self.outcome = outcome
        self.failed_checks = failed_checks
        self.reason_code = reason_code
        self.details = details

    def __repr__(self) -> str:
        return (
            f"VerifierResult(outcome={self.outcome.value}, "
            f"failed_checks={self.failed_checks}, reason_code='{self.reason_code}')"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VerifierResult):
            return False
        return (
            self.outcome == other.outcome
            and self.failed_checks == other.failed_checks
            and self.reason_code == other.reason_code
            and self.details == other.details
        )


# Unit-bearing amount pattern: explicitly requires currency symbol/code (Rs, INR, ₹) OR paise unit
AMOUNT_UNIT_PATTERN = re.compile(
    r"(?:(?:INR|Rs\.?|₹)\s*\d+(?:,\d{3})*(?:\.\d{1,2})?|\b\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:paise|paisa)\b)",
    re.IGNORECASE,
)

# Pattern to detect any digit sequence (for bare-number checks)
DIGIT_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")


def _extract_unit_bearing_amount_tokens(text: str) -> list[str]:
    """Extract amount tokens that have an explicit currency or paise unit."""
    return [m.group(0).strip() for m in AMOUNT_UNIT_PATTERN.finditer(text)]


def _extract_all_number_tokens(text: str) -> list[str]:
    """Extract all digit-bearing tokens (with or without units)."""
    return [m.group(0).strip() for m in DIGIT_PATTERN.finditer(text)]


def _strict_convert_amount_token(token: str) -> tuple[int | None, str]:
    """Strictly convert a single unit-bearing amount token to integer paise using Decimal.

    Returns (paise_value, unit_type), where unit_type is 'paise' or 'rupee'.
    If the token lacks an explicit unit or has fractional paise, returns (None, 'none').
    """
    token_clean = token.replace(",", "").strip()

    is_paise = bool(re.search(r"\b(?:paise|paisa)\b", token_clean, re.IGNORECASE))
    has_rupee_sym = bool(re.search(r"(?:INR|Rs\.?|₹)", token_clean, re.IGNORECASE))

    if not is_paise and not has_rupee_sym:
        # Bare number without unit
        return None, "none"

    num_part = re.sub(r"(?:INR|Rs\.?|₹|paise|paisa)", "", token_clean, flags=re.IGNORECASE).strip()
    if not num_part:
        return None, "none"

    try:
        dec = Decimal(num_part)
    except InvalidOperation:
        return None, "none"

    if is_paise:
        # Paise unit: must be exact integer paise, never multiplied
        if dec != dec.to_integral_value():
            return None, "paise"
        return int(dec), "paise"

    if has_rupee_sym:
        # Rupee unit: multiplied by 100 once
        dec_paise = dec * Decimal(100)
        if dec_paise != dec_paise.to_integral_value():
            return None, "rupee"
        return int(dec_paise), "rupee"

    return None, "none"


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


def _collect_allowed_numbers(
    bundle: CaseBundle,
    cited_amount_paise: int | None,
    verified_quote: str | None = None,
) -> set[str | int]:
    """Collect strictly allowed standalone numbers for V6 explanation verification.

    Permitted numbers:
    1. Fact integers and their absolute values
    2. Their rupee renderings (e.g. 1500 -> 15, '15.0', '15.00')
    3. The cited amount (if present) and its rupee renderings
    4. Digits found inside the verified quote
    5. Neutral constants: {0, 1, 100} and standard decimal representations
    """
    allowed: set[str | int] = {0, 1, 100, "0", "1", "100", "0.0", "1.0", "100.0", "0.00", "1.00", "100.00"}

    def add_num_and_renderings(n: int) -> None:
        allowed.add(n)
        allowed.add(str(n))
        abs_n = abs(n)
        allowed.add(abs_n)
        allowed.add(str(abs_n))

        # Rupee rendering
        r_float = abs_n / 100.0
        allowed.add(f"{r_float:.2f}")
        allowed.add(f"{r_float:.1f}")
        if abs_n % 100 == 0:
            allowed.add(abs_n // 100)
            allowed.add(str(abs_n // 100))

    # 1. Fact integers and their absolute values & rupee renderings
    for v in bundle.facts.values():
        if isinstance(v, int) and not isinstance(v, bool):
            add_num_and_renderings(v)

    # 2. Cited amount
    if cited_amount_paise is not None:
        add_num_and_renderings(cited_amount_paise)

    # 3. Digits inside the verified quote
    if verified_quote:
        for m in re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", verified_quote):
            raw_tok = m.group(0).replace(",", "")
            allowed.add(raw_tok)
            try:
                if "." in raw_tok:
                    f_val = float(raw_tok)
                    allowed.add(f"{f_val:.2f}")
                    allowed.add(f"{f_val:.1f}")
                else:
                    i_val = int(raw_tok)
                    allowed.add(i_val)
                    allowed.add(str(i_val))
            except ValueError:
                pass

    return allowed


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

        bundle_facts = bundle.facts
        bundle_untrusted = bundle.untrusted_text

        # --- Check V3: Grounded references and matching values in evidence ---
        for ev in llm_output.evidence:
            raw_field = ev.get("field", "")
            # Strip ONLY leading "facts." or "untrusted_text."
            if raw_field.startswith("facts."):
                field = raw_field[len("facts.") :]
            elif raw_field.startswith("untrusted_text."):
                field = raw_field[len("untrusted_text.") :]
            else:
                field = raw_field

            ev_value = ev.get("value")

            if field in bundle_facts:
                fact_val = bundle_facts[field]
                # An evidence value must have exactly the type of the referenced fact.
                # In Python, bool is an int subclass; exact type equality catches bool vs int.
                if type(ev_value) is not type(fact_val):
                    failed_checks.append("V3")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="EVIDENCE_VALUE_MISMATCH",
                        details=f"Evidence field '{raw_field}' type {type(ev_value).__name__} != fact type {type(fact_val).__name__}.",
                    )
                if fact_val != ev_value:
                    failed_checks.append("V3")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="EVIDENCE_VALUE_MISMATCH",
                        details=f"Evidence field '{raw_field}' value {ev_value!r} != fact value {fact_val!r}.",
                    )

            elif field in bundle_untrusted:
                untrusted_val = bundle_untrusted[field]
                if untrusted_val is None or not isinstance(ev_value, str) or not ev_value:
                    failed_checks.append("V3")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="EVIDENCE_VALUE_MISMATCH",
                        details=f"Evidence field '{raw_field}' references empty or missing text.",
                    )
                if ev_value not in untrusted_val:
                    failed_checks.append("V3")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="EVIDENCE_VALUE_MISMATCH",
                        details=f"Evidence value '{ev_value}' is not a substring of '{raw_field}'.",
                    )

            else:
                # Field not in facts and not in untrusted_text (hasattr removed)
                failed_checks.append("V3")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="UNGROUNDED_EVIDENCE_FIELD",
                    details=f"Cited evidence field '{raw_field}' does not exist in bundle facts or untrusted text.",
                )

        # --- Check V4: Verbatim quote substring & length check ---
        verified_quote: str | None = None
        if llm_output.verbatim_quote is not None:
            quote = llm_output.verbatim_quote
            # Minimum quote length requirement (>= 8 characters)
            if len(quote.strip()) < 8:
                failed_checks.append("V4")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="QUOTE_TOO_SHORT",
                    details=f"Verbatim quote '{quote}' is shorter than 8 characters.",
                )

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
            verified_quote = quote

        # --- Check V5: Amount consistency, strict conversion & grounding ---
        is_quote_sourced = False
        parsed_quote_amount_paise: int | None = None

        if verified_quote:
            unit_tokens = _extract_unit_bearing_amount_tokens(verified_quote)
            all_num_tokens = _extract_all_number_tokens(verified_quote)

            # If quote contains numbers but NONE has an explicit unit, bare number is never an amount -> ABSTAIN
            if len(unit_tokens) == 0 and len(all_num_tokens) > 0:
                failed_checks.append("V5")
                return VerifierResult(
                    outcome=VerifierOutcome.ABSTAIN,
                    failed_checks=tuple(failed_checks),
                    reason_code="BARE_NUMBER_NOT_AMOUNT",
                    details="Quote contains numbers but lacks explicit currency/paise unit; bare numbers are not amounts.",
                )

            if len(unit_tokens) == 0 and len(all_num_tokens) == 0:
                failed_checks.append("V5")
                return VerifierResult(
                    outcome=VerifierOutcome.ABSTAIN,
                    failed_checks=tuple(failed_checks),
                    reason_code="NO_AMOUNT_IN_QUOTE",
                    details="Quote contains no numeric amount tokens.",
                )

            # Exactly one unit-bearing token required
            if len(unit_tokens) > 1:
                failed_checks.append("V5")
                return VerifierResult(
                    outcome=VerifierOutcome.ABSTAIN,
                    failed_checks=tuple(failed_checks),
                    reason_code="AMBIGUOUS_QUOTE_AMOUNTS",
                    details=f"Quote contains {len(unit_tokens)} unit-bearing amount tokens (expected exactly 1); abstaining.",
                )

            parsed_amt, unit_type = _strict_convert_amount_token(unit_tokens[0])
            if parsed_amt is None:
                failed_checks.append("V5")
                return VerifierResult(
                    outcome=VerifierOutcome.ABSTAIN,
                    failed_checks=tuple(failed_checks),
                    reason_code="UNPARSEABLE_QUOTE_AMOUNT",
                    details=f"Failed to strictly parse unit-bearing amount token '{unit_tokens[0]}'.",
                )
            parsed_quote_amount_paise = parsed_amt

            # Match cited_amount_paise with parsed quote amount
            if llm_output.cited_amount_paise is not None:
                if llm_output.cited_amount_paise != parsed_amt:
                    failed_checks.append("V5")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="AMOUNT_MISMATCH_WITH_QUOTE",
                        details=(
                            f"Cited amount {llm_output.cited_amount_paise} paise does not match quote token "
                            f"'{unit_tokens[0]}' ({parsed_amt} paise)."
                        ),
                    )
                is_quote_sourced = True

        elif llm_output.cited_amount_paise is not None:
            # V5 without a quote: cited_amount_paise must equal a fact integer value or |value|
            cited = llm_output.cited_amount_paise
            fact_ints = set()
            for v in bundle_facts.values():
                if isinstance(v, int) and not isinstance(v, bool):
                    fact_ints.add(v)
                    fact_ints.add(abs(v))

            if cited not in fact_ints:
                failed_checks.append("V5")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="UNGROUNDED_AMOUNT",
                    details=f"Cited amount {cited} paise does not match any fact value in bundle.",
                )

        # --- Check V6: No fabricated standalone numbers in explanation ---
        bundle_ids = _get_bundle_all_ids(bundle)
        explanation_clean = llm_output.explanation

        # Remove known bundle IDs from explanation text
        for bid in sorted(bundle_ids, key=len, reverse=True):
            explanation_clean = explanation_clean.replace(bid, " ")

        # Extract standalone numbers from explanation
        extracted_num_tokens = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", explanation_clean)
        allowed_numbers = _collect_allowed_numbers(
            bundle=bundle,
            cited_amount_paise=llm_output.cited_amount_paise,
            verified_quote=verified_quote if is_quote_sourced else None,
        )

        for num_tok in extracted_num_tokens:
            clean_tok = num_tok.replace(",", "")
            # Check match against allowed_numbers
            matched = False
            if clean_tok in allowed_numbers:
                matched = True
            else:
                try:
                    if "." in clean_tok:
                        f_val = float(clean_tok)
                        if f_val in allowed_numbers or clean_tok in allowed_numbers:
                            matched = True
                    else:
                        i_val = int(clean_tok)
                        if i_val in allowed_numbers:
                            matched = True
                except ValueError:
                    pass

            if not matched:
                failed_checks.append("V6")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="FABRICATED_NUMBER_IN_EXPLANATION",
                    details=f"Explanation contains ungrounded standalone number '{num_tok}'.",
                )

        # If decision is escalate_human and V1-V6 passed, it is verified safe
        if llm_output.decision == "escalate_human":
            return VerifierResult(
                outcome=VerifierOutcome.ACCEPT,
                failed_checks=(),
                reason_code=llm_output.reason_code or "ESCALATION_ACCEPTED",
                details="Decision is escalate_human; verified safe.",
            )

        # --- Check V7: Arithmetic, class guard table & policy checks ---
        if llm_output.decision == "benign_explained":
            # benign_explained REQUIRES verified quote sourcing the amount
            if not verified_quote:
                failed_checks.append("V4")
                failed_checks.append("V7")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="MISSING_VERBATIM_QUOTE",
                    details="Decision 'benign_explained' requires a verified verbatim quote sourcing the amount.",
                )

            if not is_quote_sourced or parsed_quote_amount_paise is None:
                failed_checks.append("V5")
                failed_checks.append("V7")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="QUOTE_NOT_SOURCING_AMOUNT",
                    details="Decision 'benign_explained' requires amount to be strictly sourced by quote.",
                )

            # Arithmetic check: recompute refund difference directly from facts
            ledger_ref = bundle_facts.get("ledger_refund_paise")
            debit = bundle_facts.get("debit_paise")
            if ledger_ref is not None and debit is not None:
                recomputed_diff = ledger_ref - debit
            elif "refund_difference_paise" in bundle_facts:
                recomputed_diff = bundle_facts["refund_difference_paise"]
            else:
                amt = bundle_facts.get("amount_paise", 0)
                recomputed_diff = (debit if debit is not None else 0) - amt

            if llm_output.cited_amount_paise is None or abs(llm_output.cited_amount_paise) != abs(recomputed_diff):
                failed_checks.append("V7")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="ARITHMETIC_MISMATCH",
                    details=(
                        f"Cited amount {llm_output.cited_amount_paise} paise does not match refund difference "
                        f"{recomputed_diff} paise."
                    ),
                )

        elif llm_output.decision == "flag":
            pred_class = llm_output.predicted_class
            if not pred_class:
                failed_checks.append("V7")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="FLAG_MISSING_PREDICTED_CLASS",
                    details="Decision 'flag' requires a non-null predicted_class.",
                )

            # Class-to-facts guard table: verify supporting facts for the class
            if pred_class == "FEE_TAX_MISMATCH":
                # Must be a fee_tax check with fee_paise in facts
                if bundle.check_family != "fee_tax" or "fee_paise" not in bundle_facts:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNSUPPORTED_FLAG_CLASS",
                        details="FEE_TAX_MISMATCH requires fee_tax family and fee_paise in facts.",
                    )
                err_thresh = bundle_facts.get("error_threshold_paise", 10)
                fee_delta = bundle_facts.get("fee_delta_paise")
                tax_delta = bundle_facts.get("tax_delta_paise")
                if fee_delta is not None or tax_delta is not None:
                    delta = max(abs(fee_delta or 0), abs(tax_delta or 0))
                    has_mismatch = delta >= err_thresh
                else:
                    amt = bundle_facts.get("amount_paise", 0)
                    fee = bundle_facts.get("fee_paise", 0)
                    credit = bundle_facts.get("credit_paise", 0)
                    tax = bundle_facts.get("tax_paise", 0)
                    diff1 = abs(amt - fee - credit)
                    diff2 = abs(amt - fee - tax - credit)
                    has_mismatch = (
                        abs(bundle_facts.get("fee_difference_paise", 0)) >= err_thresh
                        or min(diff1, diff2) >= err_thresh
                    )
                if not has_mismatch:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNSUPPORTED_FLAG_CLASS",
                        details=f"FEE_TAX_MISMATCH requires fee/tax discrepancy >= {err_thresh} paise error threshold in facts.",
                    )

            elif pred_class == "PARTIAL_REFUND_MISMATCH":
                if bundle.check_family != "refund":
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNSUPPORTED_FLAG_CLASS",
                        details="PARTIAL_REFUND_MISMATCH requires refund check family.",
                    )
                ref_tol = bundle_facts.get("refund_tolerance_paise", 0)
                refund_diff = bundle_facts.get("refund_difference_paise")
                ledger_ref = bundle_facts.get("ledger_refund_paise")
                debit = bundle_facts.get("debit_paise")
                if ledger_ref is not None and debit is not None:
                    refund_diff = ledger_ref - debit
                elif refund_diff is None and "amount_paise" in bundle_facts:
                    refund_diff = (debit or 0) - bundle_facts["amount_paise"]
                if refund_diff is None or abs(refund_diff) <= ref_tol:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNSUPPORTED_FLAG_CLASS",
                        details=f"PARTIAL_REFUND_MISMATCH requires refund difference exceeding {ref_tol} paise tolerance in facts.",
                    )

            elif pred_class == "MISSING_SETTLEMENT":
                has_settlement_facts = "settled" in bundle_facts or "settlement_id" in bundle_facts
                is_missing = (
                    has_settlement_facts
                    and (
                        bundle_facts.get("settled") is False
                        or bundle_facts.get("settlement_id") is None
                        or bundle_facts.get("settlement_settled_at") is None
                    )
                )
                if not is_missing:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNSUPPORTED_FLAG_CLASS",
                        details="MISSING_SETTLEMENT requires facts showing unsettled status or missing settlement.",
                    )

            elif pred_class in ("SHORT_BANK_CREDIT", "ORPHAN_BANK_CREDIT"):
                if bundle.check_family != "bank" and "bank_credit_paise" not in bundle_facts:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNSUPPORTED_FLAG_CLASS",
                        details=f"{pred_class} requires bank check family and bank credit facts.",
                    )
                if pred_class == "SHORT_BANK_CREDIT":
                    credit = bundle_facts.get("bank_credit_paise", 0)
                    expected = bundle_facts.get("expected_credit_paise") or bundle_facts.get("settlement_amount_paise")
                    if expected is None or credit >= expected:
                        failed_checks.append("V7")
                        return VerifierResult(
                            outcome=VerifierOutcome.REJECT,
                            failed_checks=tuple(failed_checks),
                            reason_code="UNSUPPORTED_FLAG_CLASS",
                            details="SHORT_BANK_CREDIT requires bank credit < expected in facts.",
                        )
                elif pred_class == "ORPHAN_BANK_CREDIT":
                    is_orphan = bundle_facts.get("orphan") is True or bundle_facts.get("matched") is False
                    if not is_orphan:
                        failed_checks.append("V7")
                        return VerifierResult(
                            outcome=VerifierOutcome.REJECT,
                            failed_checks=tuple(failed_checks),
                            reason_code="UNSUPPORTED_FLAG_CLASS",
                            details="ORPHAN_BANK_CREDIT requires orphan or unmatched status in facts.",
                        )

            elif pred_class == "DUPLICATE_PAYMENT":
                is_dup = bundle_facts.get("is_duplicate") is True or "duplicate_payment_id" in bundle_facts
                if not is_dup:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNSUPPORTED_FLAG_CLASS",
                        details="DUPLICATE_PAYMENT requires duplicate payment indication in facts.",
                    )

            elif pred_class == "CUTOFF_DRIFT":
                has_drift = bundle_facts.get("cutoff_drift") is True or bundle_facts.get("drift_days", 0) > 0
                if not has_drift:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="UNSUPPORTED_FLAG_CLASS",
                        details="CUTOFF_DRIFT requires timing drift indication in facts.",
                    )

            else:
                failed_checks.append("V7")
                return VerifierResult(
                    outcome=VerifierOutcome.REJECT,
                    failed_checks=tuple(failed_checks),
                    reason_code="UNSUPPORTED_FLAG_CLASS",
                    details=f"Flagged class '{pred_class}' is unsupported or has no guard table verification.",
                )

        elif llm_output.decision == "no_issue":
            # Guard no_issue using verifiable supporting facts directly
            # 1. Fee/tax check family: requires fee_delta_paise and tax_delta_paise in facts
            if bundle.check_family == "fee_tax":
                fee_delta = bundle_facts.get("fee_delta_paise")
                tax_delta = bundle_facts.get("tax_delta_paise")
                if fee_delta is None or tax_delta is None:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.ABSTAIN,
                        failed_checks=tuple(failed_checks),
                        reason_code="MISSING_DELTA_FACTS",
                        details="Decision 'no_issue' for fee_tax requires fee_delta_paise and tax_delta_paise in facts.",
                    )
                round_tol = bundle_facts.get("rounding_tolerance_paise", 2)
                delta = max(abs(fee_delta), abs(tax_delta))
                if delta > round_tol:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="POLICY_VIOLATION_TOLERANCE",
                        details=f"Payment fee/tax delta ({delta} paise) exceeds {round_tol} paise rounding tolerance.",
                    )

            # 2. Refund check family: requires refund_difference within refund tolerance (0 paise)
            elif bundle.check_family == "refund":
                ledger_ref = bundle_facts.get("ledger_refund_paise")
                debit = bundle_facts.get("debit_paise")
                if ledger_ref is not None and debit is not None:
                    diff = ledger_ref - debit
                elif "refund_difference_paise" in bundle_facts:
                    diff = bundle_facts["refund_difference_paise"]
                else:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.ABSTAIN,
                        failed_checks=tuple(failed_checks),
                        reason_code="MISSING_REFUND_DIFF_FACTS",
                        details="Decision 'no_issue' for refund requires verifiable refund difference facts.",
                    )
                ref_tol = bundle_facts.get("refund_tolerance_paise", 0)
                if abs(diff) > ref_tol:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="POLICY_VIOLATION_REFUND_DIFF",
                        details=f"Refund has difference ({diff} paise) exceeding tolerance ({ref_tol} paise); no_issue rejected.",
                    )

            # 3. Bank check family: requires unambiguous match
            elif bundle.check_family == "bank":
                has_ambiguity_facts = (
                    "ambiguous_candidates" in bundle_facts
                    or "candidates_count" in bundle_facts
                    or "unambiguous_match" in bundle_facts
                )
                if not has_ambiguity_facts:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.ABSTAIN,
                        failed_checks=tuple(failed_checks),
                        reason_code="MISSING_BANK_MATCH_FACTS",
                        details="Decision 'no_issue' for bank requires candidate ambiguity facts.",
                    )
                is_ambiguous = (
                    bundle_facts.get("ambiguous_candidates") is True
                    or bundle_facts.get("candidates_count", 1) > 1
                    or bundle_facts.get("unambiguous_match") is False
                    or bundle.rule_id == "B1_BANK_AMBIGUOUS_CANDIDATES"
                )
                if is_ambiguous:
                    failed_checks.append("V7")
                    return VerifierResult(
                        outcome=VerifierOutcome.REJECT,
                        failed_checks=tuple(failed_checks),
                        reason_code="POLICY_VIOLATION_AMBIGUOUS_BANK",
                        details="Bank statement item has ambiguous candidates; no_issue rejected.",
                    )

        return VerifierResult(
            outcome=VerifierOutcome.ACCEPT,
            failed_checks=(),
            reason_code=llm_output.reason_code or "VERIFIED_ACCEPT",
            details="All verifier checks passed.",
        )
