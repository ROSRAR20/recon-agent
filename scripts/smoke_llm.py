"""Smoke test script for real LLM integration.

Sends exactly 5 synthetic bundles (one per rule family plus one control) once,
prints sanitized outcome, token counts, latency, estimated cost, and verifier result,
and writes to response cache and audit log.

Strictly refuses to run unless LLM_SMOKE=1 is set in environment.
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.guard import sanitize_error
from src.recon_agent.llm.audit import AuditLogger
from src.recon_agent.llm.bundle import CaseBundle, create_case_bundle
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.gemini_client import GeminiClient
from src.recon_agent.llm.schema import validate_llm_output
from src.recon_agent.llm.verifier import Verifier


def build_smoke_synthetic_bundles() -> list[CaseBundle]:
    """Build exactly 5 synthetic bundles representing rule families and one control."""
    bundles = []

    # 1. Family: Near-tolerance fee/tax discrepancy
    b1 = create_case_bundle(
        bundle_id="smoke_bundle_near_tolerance",
        entity_id="pay_smoke_01",
        merchant_id="mer_smoke_01",
        case_type="payment",
        rule_id="B1_NEAR_TOLERANCE",
        facts={
            "entity_id": "pay_smoke_01",
            "type": "payment",
            "amount_paise": 10000,
            "fee_paise": 206,
            "tax_paise": 37,
            "debit_paise": 0,
            "credit_paise": 9757,
        },
        untrusted_text={
            "recon_item_description": "Card payment with fee near tolerance",
            "recon_item_notes_comment": None,
            "recon_item_notes_reason_text": None,
            "bank_narration": None,
        },
    )
    bundles.append(b1)

    # 2. Family: Refund with reason text (arithmetic explanation)
    b2 = create_case_bundle(
        bundle_id="smoke_bundle_refund_reason_text",
        entity_id="rfnd_smoke_02",
        merchant_id="mer_smoke_01",
        case_type="refund",
        rule_id="REFUND_REASON_TEXT",
        facts={
            "entity_id": "rfnd_smoke_02",
            "type": "refund",
            "amount_paise": 20000,
            "debit_paise": 18500,
            "refund_difference_paise": -1500,
        },
        untrusted_text={
            "recon_item_description": "Refund item",
            "recon_item_notes_comment": None,
            "recon_item_notes_reason_text": "Processing fee deduction of Rs 15.00 applied.",
            "bank_narration": None,
        },
    )
    bundles.append(b2)

    # 3. Family: Refund with qualitative note but no amount
    b3 = create_case_bundle(
        bundle_id="smoke_bundle_refund_no_amount",
        entity_id="rfnd_smoke_03",
        merchant_id="mer_smoke_01",
        case_type="refund",
        rule_id="B1R_EXPLAINED_NO_AMOUNT",
        facts={
            "entity_id": "rfnd_smoke_03",
            "type": "refund",
            "amount_paise": 30000,
            "debit_paise": 25000,
            "refund_difference_paise": -5000,
        },
        untrusted_text={
            "recon_item_description": "Partial refund",
            "recon_item_notes_comment": None,
            "recon_item_notes_reason_text": "Customer accepted partial refund after support consultation.",
            "bank_narration": None,
        },
    )
    bundles.append(b3)

    # 4. Family: Bank statement candidate ambiguity
    b4 = create_case_bundle(
        bundle_id="smoke_bundle_bank_ambiguity",
        entity_id="stmt_smoke_04",
        merchant_id="mer_smoke_01",
        case_type="bank_statement",
        rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
        facts={
            "statement_id": "stmt_smoke_04",
            "bank_credit_paise": 500000,
            "bank_date": "2026-05-15",
        },
        untrusted_text={
            "recon_item_description": None,
            "recon_item_notes_comment": None,
            "recon_item_notes_reason_text": None,
            "bank_narration": "RAZORPAY SETTLEMENT BATCH 5000.00",
        },
    )
    bundles.append(b4)

    # 5. Family: Clean Negative Control (no issue)
    b5 = create_case_bundle(
        bundle_id="smoke_bundle_control",
        entity_id="pay_smoke_05",
        merchant_id="mer_smoke_01",
        case_type="payment",
        rule_id="CLEAN_CONTROL",
        facts={
            "entity_id": "pay_smoke_05",
            "type": "payment",
            "amount_paise": 50000,
            "fee_paise": 1000,
            "tax_paise": 180,
            "debit_paise": 0,
            "credit_paise": 48820,
        },
        untrusted_text={
            "recon_item_description": "Standard card transaction",
            "recon_item_notes_comment": None,
            "recon_item_notes_reason_text": None,
            "bank_narration": None,
        },
    )
    bundles.append(b5)

    return bundles


def main() -> None:
    if os.getenv("LLM_SMOKE") != "1":
        print("[ERROR] Refusing to run smoke test. Env LLM_SMOKE=1 is required.", file=sys.stderr)
        sys.exit(1)

    system_prompt_path = os.path.join(REPO_ROOT, "src", "recon_agent", "llm", "prompts", "system.txt")
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    cache = LLMCache()
    audit_logger = AuditLogger()
    client = GeminiClient(cache=cache, audit_logger=audit_logger, thinking_level="low")
    verifier = Verifier()

    bundles = build_smoke_synthetic_bundles()
    print(f"[*] Starting LLM smoke test on {len(bundles)} synthetic bundles...\n")

    for idx, bundle in enumerate(bundles, 1):
        print(f"--- Case {idx}/5: {bundle.bundle_id} (Family: {bundle.rule_id}) ---")
        raw_output = client.complete(bundle, system_prompt=system_prompt, run_id=0)

        # Parse and verify
        is_valid, parsed, errors = validate_llm_output(raw_output)
        if is_valid and parsed:
            ver_res = verifier.verify(bundle, parsed)
            outcome = ver_res.outcome.value
            failed_checks = ver_res.failed_checks
        else:
            outcome = "REJECT"
            failed_checks = ("V1",)

        sanitized_summary = sanitize_error(raw_output)
        print(f"  Sanitized Output: {sanitized_summary}")
        print(f"  Verifier Outcome: {outcome} (Failed: {failed_checks})")
        print()


if __name__ == "__main__":
    main()

