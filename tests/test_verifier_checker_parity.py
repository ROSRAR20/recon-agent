"""Parity tests between independent checker and verifier unit grammars.

Asserts:
1. Every unit-bearing form accepted by independent_checker's regex is recognized
   by verifier's AMOUNT_UNIT_PATTERN with identical converted paise value.
2. Conflicting unit forms (e.g. prefix and postfix together) are rejected by both.
3. Bare numbers (no unit) are not treated as unit-bearing amounts by either.
4. Red-team test: explanation containing "1 rupee" with refund_difference_paise = 346
   is strictly REJECTed by verifier with FABRICATED_NUMBER_IN_EXPLANATION.
"""

from __future__ import annotations

from decimal import Decimal
from dataclasses import asdict
import itertools
import pytest

from src.recon_agent.eval.independent_checker import _NUM as CHECKER_NUM, _dec, _paise, check_output
from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.schema import LLMOutput
from src.recon_agent.llm.verifier import (
    AMOUNT_UNIT_PATTERN,
    Verifier,
    VerifierOutcome,
    _extract_unit_bearing_amount_tokens,
    _strict_convert_amount_token,
)


def test_unit_grammar_parity_generated_forms():
    prefixes = ["INR", "Rs.", "Rs", "₹"]
    postfixes = ["rupee", "rupees", "paise", "paisa"]
    numbers = ["1", "5", "10", "250", "1,234", "5.50", "12.34"]

    # 1. Prefix-only forms: e.g. "INR 10", "Rs. 5.50"
    for pre, num in itertools.product(prefixes, numbers):
        for form in [f"{pre} {num}", f"{pre}{num}"]:
            # Checker
            m = CHECKER_NUM.search(form)
            assert m is not None, f"Checker failed to match {form}"
            chk_has_unit, chk_paise = _paise(m)
            assert chk_has_unit is True

            # Verifier
            v_matches = _extract_unit_bearing_amount_tokens(form)
            assert len(v_matches) == 1, f"Verifier failed to extract token from {form}: got {v_matches}"
            v_paise, v_unit = _strict_convert_amount_token(v_matches[0])
            assert v_paise is not None, f"Verifier failed to parse {v_matches[0]}"
            assert Decimal(v_paise) == chk_paise, f"Mismatch on {form}: verifier={v_paise} vs checker={chk_paise}"

    # 2. Postfix-only forms: e.g. "10 rupees", "1 rupee", "250 paise", "1 paisa"
    for num, post in itertools.product(numbers, postfixes):
        form = f"{num} {post}"
        m = CHECKER_NUM.search(form)
        assert m is not None, f"Checker failed to match {form}"
        chk_has_unit, chk_paise = _paise(m)
        assert chk_has_unit is True

        v_matches = _extract_unit_bearing_amount_tokens(form)
        assert len(v_matches) == 1, f"Verifier failed to extract token from {form}: got {v_matches}"
        v_paise, v_unit = _strict_convert_amount_token(v_matches[0])

        if post in ("paise", "paisa") and "." in num:
            # Fractional paise: both reject or treat as non-integer
            assert v_paise is None
        else:
            assert v_paise is not None, f"Verifier failed to parse {v_matches[0]}"
            assert Decimal(v_paise) == chk_paise, f"Mismatch on {form}: verifier={v_paise} vs checker={chk_paise}"


def test_conflicting_units_rejected_by_both():
    conflicting_forms = [
        "Rs 10 rupees",
        "INR 500 paise",
        "₹ 25 paisa",
        "Rs. 100 rupees",
    ]
    for form in conflicting_forms:
        m = CHECKER_NUM.search(form)
        assert m is not None
        chk_has_unit, chk_paise = _paise(m)
        assert chk_has_unit is True
        assert chk_paise is None  # Conflicting units in checker return None

        v_matches = _extract_unit_bearing_amount_tokens(form)
        if v_matches:
            v_paise, v_unit = _strict_convert_amount_token(v_matches[0])
            assert v_paise is None
            assert v_unit == "conflicting"


def test_bare_numbers_not_unit_bearing():
    bare_forms = ["1", "42", "100", "1234", "5.50"]
    for form in bare_forms:
        m = CHECKER_NUM.search(form)
        assert m is not None
        chk_has_unit, chk_paise = _paise(m)
        assert chk_has_unit is False

        v_matches = _extract_unit_bearing_amount_tokens(form)
        assert len(v_matches) == 0


def test_red_team_one_rupee_rejected():
    """Red-team: explanation '1 rupee' with refund difference 346 -> REJECT."""
    bundle = CaseBundle(
        bundle_id="bundle_redteam_01",
        check_family="refund",
        entity_id="rfnd_rt_01",
        merchant_id="mer_rt_01",
        facts={
            "amount_paise": 10000,
            "debit_paise": 9654,
            "ledger_refund_paise": 10000,
            "refund_difference_paise": 346,
            "refund_tolerance_paise": 0,
        },
        untrusted_text={
            "recon_item_notes_reason_text": "Customer complained about delay.",
        },
    )

    # Output claiming 1 rupee deduction, which is 100 paise != 346 paise
    output = LLMOutput(
        decision="escalate_human",
        predicted_class=None,
        reason_code="UNVERIFIABLE_CLAIM",
        explanation="Customer requested fee waiver of 1 rupee.",
        evidence=[],
        cited_amount_paise=None,
        verbatim_quote=None,
    )

    res = Verifier().verify(bundle, output)
    assert res.outcome == VerifierOutcome.REJECT
    assert "V6" in res.failed_checks
    assert res.reason_code == "FABRICATED_NUMBER_IN_EXPLANATION"


def test_verifier_accepts_faithful_mismatch_flag():
    """Verify that V5-V7 accept a faithful PARTIAL_REFUND_MISMATCH flag."""
    bundle = CaseBundle(
        bundle_id="bundle_rfnd_syn_06",
        check_family="refund",
        entity_id="rfnd_syn_06",
        merchant_id="mer_syn_01",
        facts={
            "amount_paise": 50000,
            "debit_paise": 45000,
            "ledger_refund_paise": 50000,
            "refund_difference_paise": 5000,
            "refund_tolerance_paise": 0,
        },
        untrusted_text={
            "bank_narration": None,
            "recon_item_description": "Partial refund processed",
            "recon_item_notes_comment": None,
            "recon_item_notes_reason_text": "Processing fee of Rs 30.00 retained per agreement.",
        },
    )

    output = LLMOutput(
        decision="flag",
        predicted_class="PARTIAL_REFUND_MISMATCH",
        reason_code="DISCREPANCY_FOUND",
        explanation="Refund difference of 5000 paise does not match cited note deduction.",
        evidence=[
            {
                "field": "facts.refund_difference_paise",
                "value": 5000,
            }
        ],
        cited_amount_paise=3000,
        verbatim_quote="Processing fee of Rs 30.00 retained per agreement.",
    )

    # 1. Verifier V1-V7 verification
    res = Verifier().verify(bundle, output)
    assert res.outcome == VerifierOutcome.ACCEPT
    assert res.failed_checks == ()
    assert res.reason_code == "DISCREPANCY_FOUND"

    # 2. Independent checker verification
    violations = check_output(bundle.to_dict(), asdict(output))
    assert violations == []

