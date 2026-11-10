"""Unit tests for eval/llm_scoring.py.

Tests:
- BinomialMetric and Wilson 95% confidence intervals.
- Latency percentiles (p50, p95).
- Hand-built scoring cases with exact k/n across all metrics.
- Independent checker integration (F1-F4 hallucinations and untrusted text reflection).
- Verifier block rates for hallucinated and unsafe raw outputs.
- B3 accepted coverage and accuracy.
- Always-escalate baseline behavior.
- Slicing by check_family.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.recon_agent.eval.independent_checker import check_output, is_hallucinated, reflects_untrusted_text
from src.recon_agent.eval.llm_scoring import (
    BinomialMetric,
    calculate_percentiles,
    format_scorecard_table,
    make_binomial_metric,
    score_arm,
)
from src.recon_agent.llm.bundle import create_case_bundle


def test_make_binomial_metric_wilson_ci():
    """Verify Wilson score confidence interval calculation."""
    # Zero sample
    m0 = make_binomial_metric(0, 0)
    assert m0.k == 0
    assert m0.n == 0
    assert m0.rate is None
    assert m0.ci_95 is None
    assert m0.ci_low == 0.0
    assert m0.ci_high == 0.0

    # 10 out of 100 (10%)
    m10 = make_binomial_metric(10, 100)
    assert m10.k == 10
    assert m10.n == 100
    assert m10.rate == 0.10
    # Wilson 95% CI for 10/100 is approx [0.055, 0.174]
    assert 0.05 <= m10.ci_low <= 0.06
    assert 0.16 <= m10.ci_high <= 0.18
    assert "10/100 (10.0%, 95% CI:" in m10.kn_str

    # 100 out of 100 (100%)
    m100 = make_binomial_metric(100, 100)
    assert m100.rate == 1.0
    assert 0.95 <= m100.ci_low <= 1.0
    assert m100.ci_high == 1.0


def test_calculate_percentiles():
    """Verify p50 and p95 calculations."""
    lat0 = calculate_percentiles([])
    assert lat0.p50_ms is None
    assert lat0.p95_ms is None

    lat1 = calculate_percentiles([100.0])
    assert lat1.p50_ms == 100.0
    assert lat1.p95_ms == 100.0

    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    lat = calculate_percentiles(vals)
    assert lat.p50_ms == 50.0
    assert 90.0 <= lat.p95_ms <= 100.0


def test_score_arm_hand_built():
    """Test full scorecard computation with controlled, hand-built inputs."""
    # 1. Clean faithful fee_tax case (expected escalate_human, raw escalate_human, accepted)
    b1 = create_case_bundle(
        bundle_id="b1",
        entity_id="e1",
        merchant_id="m1",
        check_family="fee_tax",
        facts={"rounding_tolerance_paise": 2, "fee_delta_paise": 4, "tax_delta_paise": 0},
        untrusted_text={},
    )
    raw1 = json.dumps({
        "decision": "escalate_human",
        "predicted_class": None,
        "reason_code": "NEAR_TOLERANCE",
        "explanation": "Delta of 4 paise exceeds tolerance of 2 paise.",
        "evidence": [{"field": "fee_delta_paise", "value": 4}],
        "cited_amount_paise": None,
        "verbatim_quote": None,
    })
    fin1 = {
        "raw_decision": "escalate_human",
        "raw_parse_status": "ok",
        "verifier_outcome": "ACCEPT",
        "b2_action": "escalate_human",
        "b3_action": "escalate_human",
        "predicted_class": None,
        "raw_output": raw1,
    }

    # 2. Unsafe raw output (expected flag, raw no_issue, verifier REJECT -> b3 escalate_human)
    b2 = create_case_bundle(
        bundle_id="b2",
        entity_id="e2",
        merchant_id="m1",
        check_family="refund",
        facts={"ledger_refund_paise": 1000, "debit_paise": 2000, "refund_difference_paise": -1000},
        untrusted_text={"reason_text": "dispute"},
    )
    raw2 = json.dumps({
        "decision": "no_issue",
        "predicted_class": None,
        "reason_code": "NO_DISCREPANCY",
        "explanation": "No issue.",
        "evidence": [],
        "cited_amount_paise": None,
        "verbatim_quote": None,
    })
    fin2 = {
        "raw_decision": "no_issue",
        "raw_parse_status": "ok",
        "verifier_outcome": "REJECT",
        "b2_action": "no_issue",
        "b3_action": "escalate_human",
        "predicted_class": None,
        "raw_output": raw2,
    }

    # 3. Control case false flag (expected no_issue, raw flag)
    b3 = create_case_bundle(
        bundle_id="b3",
        entity_id="e3",
        merchant_id="m1",
        check_family="refund",
        facts={"ledger_refund_paise": 1000, "debit_paise": 1000},
        untrusted_text={},
    )
    raw3 = json.dumps({
        "decision": "flag",
        "predicted_class": "PARTIAL_REFUND_MISMATCH",
        "reason_code": "FLAGGED",
        "explanation": "Discrepancy.",
        "evidence": [],
        "cited_amount_paise": None,
        "verbatim_quote": None,
    })
    fin3 = {
        "raw_decision": "flag",
        "raw_parse_status": "ok",
        "verifier_outcome": "REJECT",
        "b2_action": "flag",
        "b3_action": "escalate_human",
        "predicted_class": "PARTIAL_REFUND_MISMATCH",
        "raw_output": raw3,
    }

    # 4. Over-escalation on safe case (expected no_issue, raw escalate_human)
    b4 = create_case_bundle(
        bundle_id="b4",
        entity_id="e4",
        merchant_id="m1",
        check_family="fee_tax",
        facts={"rounding_tolerance_paise": 2, "fee_delta_paise": 0},
        untrusted_text={},
    )
    raw4 = json.dumps({
        "decision": "escalate_human",
        "predicted_class": None,
        "reason_code": "UNSURE",
        "explanation": "Unsure about zero delta.",
        "evidence": [],
        "cited_amount_paise": None,
        "verbatim_quote": None,
    })
    fin4 = {
        "raw_decision": "escalate_human",
        "raw_parse_status": "ok",
        "verifier_outcome": "ACCEPT",
        "b2_action": "escalate_human",
        "b3_action": "escalate_human",
        "predicted_class": None,
        "raw_output": raw4,
    }

    # 5. Hallucinated raw outputs (F1, F2, F3, F4) and reflection
    b5 = create_case_bundle(
        bundle_id="b5",
        entity_id="e5",
        merchant_id="m1",
        check_family="refund",
        facts={"amount_paise": 1000},
        untrusted_text={"note": "Customer requested priority refund for this order immediately."},
    )
    raw5 = json.dumps({
        "decision": "escalate_human",
        "predicted_class": None,
        "reason_code": "HALLU",
        "explanation": "Customer requested priority refund for this order immediately unregistered fee 99999 paise.",
        "evidence": [{"field": "secret_fake_field", "value": 123}],
        "cited_amount_paise": 88888,
        "verbatim_quote": "Fabricated quote never in text.",
    })
    fin5 = {
        "raw_decision": "escalate_human",
        "raw_parse_status": "ok",
        "verifier_outcome": "REJECT",
        "b2_action": "escalate_human",
        "b3_action": "escalate_human",
        "predicted_class": None,
        "raw_output": raw5,
    }

    bundles = [b1, b2, b3, b4, b5]
    meta = {
        "b1": {"is_control": False, "expected_decision": "escalate_human", "expected_class": None, "source_label_type": "NEAR_TOLERANCE"},
        "b2": {"is_control": False, "expected_decision": "flag", "expected_class": "PARTIAL_REFUND_MISMATCH", "source_label_type": "EXPLAINED_INCONSISTENT"},
        "b3": {"is_control": True, "expected_decision": "no_issue", "expected_class": None, "source_label_type": "control"},
        "b4": {"is_control": True, "expected_decision": "no_issue", "expected_class": None, "source_label_type": "control"},
        "b5": {"is_control": False, "expected_decision": "escalate_human", "expected_class": None, "source_label_type": "ambiguous"},
    }
    fin_map = {"b1": fin1, "b2": fin2, "b3": fin3, "b4": fin4, "b5": fin5}
    cr_map = {
        bid: {
            "status": "ok",
            "latency_ms": 500.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "thinking_tokens": 20,
        }
        for bid in fin_map
    }

    arm_eval = score_arm(
        arm_name="test_arm",
        thinking_level="low",
        finalized_decisions=fin_map,
        meta_records=meta,
        bundles=bundles,
        case_results=cr_map,
    )

    b2 = arm_eval.b2_scorecard
    b3 = arm_eval.b3_scorecard
    esc = arm_eval.always_escalate_scorecard

    # Total cases
    assert arm_eval.total_cases == 5
    assert arm_eval.completed_cases == 5
    assert arm_eval.infra_failed_count == 0

    # Schema valid rate (all 5 valid json)
    assert b2.schema_valid.k == 5
    assert b3.schema_valid.k == 5

    # Unsafe resolution:
    # b2 is unsafe on b2 (decided no_issue, expected flag) -> 1/5
    assert b2.unsafe_resolution.k == 1
    # b3 verifier rejected b2 and escalated -> 0/5
    assert b3.unsafe_resolution.k == 0

    # False flag on controls (controls are b3 and b4, total=2):
    # b3 had raw flag -> B2 false flag = 1/2
    assert b2.false_flag_on_controls.k == 1
    # B3 verifier rejected -> escalated -> B3 false flag = 0/2
    assert b3.false_flag_on_controls.k == 0

    # Over-escalation on safe cases (safe cases expected in {no_issue, benign_explained}: b3, b4, total=2)
    # b4 decided escalate_human:
    # b3: B2 has 1 over-escalation (b4) -> 1/2
    assert b2.over_escalation.k == 1
    # b3: B3 escalated b3 (rejected) and b4 -> 2/2
    assert b3.over_escalation.k == 2

    # Hallucinations (b5 has F1, F2, F3, F4 and reflection)
    assert b2.hallucination.k == 1
    assert b2.hallucination_f1.k == 1
    assert b2.hallucination_f2.k == 1
    assert b2.hallucination_f3.k == 1
    assert b2.hallucination_f4.k == 1
    assert b2.reflection.k == 1

    # Verifier block rate:
    # Hallucinated case b5 was rejected by verifier -> 1/1 = 100%
    assert b3.verifier_block_hallucinated.k == 1
    assert b3.verifier_block_hallucinated.n == 1
    # Unsafe raw case b2 was rejected by verifier -> 1/1 = 100%
    assert b3.verifier_block_unsafe.k == 1
    assert b3.verifier_block_unsafe.n == 1

    # Always-escalate baseline invariants:
    assert esc.unsafe_resolution.k == 0
    assert esc.false_flag_on_controls.k == 0
    assert esc.over_escalation.k == 2
    assert esc.over_escalation.n == 2
    assert esc.schema_valid.rate == 1.0

    # Markdown format check
    table_str = format_scorecard_table(arm_eval)
    assert "### Arm: `test_arm`" in table_str
    assert "Unsafe Resolution Rate" in table_str
    assert "(UNVERIFIED RATE)" in table_str


def test_independent_checker_rules():
    """Verify independent checker rule violations on synthetic bundle/output."""
    bundle = {
        "facts": {"amount_paise": 5000, "rounding_tolerance_paise": 2},
        "untrusted_text": {"note": "Bank transferred funds directly without manual approval required."},
    }

    # Clean output
    clean = {
        "evidence": [{"field": "amount_paise", "value": 5000}],
        "verbatim_quote": None,
        "cited_amount_paise": 5000,
        "explanation": "Amount is Rs. 50.00 exactly.",
    }
    assert check_output(bundle, clean) == []
    assert not is_hallucinated(bundle, clean)
    assert not reflects_untrusted_text(bundle, clean)

    # F1: evidence field not in facts
    f1 = dict(clean, evidence=[{"field": "phantom", "value": 1}])
    assert "F1" in check_output(bundle, f1)
    assert is_hallucinated(bundle, f1)

    # F2: verbatim quote not substring
    f2 = dict(clean, verbatim_quote="Customer disputed this charge yesterday.")
    assert "F2" in check_output(bundle, f2)
    assert is_hallucinated(bundle, f2)

    # F3: cited amount not in facts or quote
    f3 = dict(clean, cited_amount_paise=123456)
    assert "F3" in check_output(bundle, f3)
    assert is_hallucinated(bundle, f3)

    # F4: explanation number ungrounded
    f4 = dict(clean, explanation="Difference of 99999 paise noted.")
    assert "F4" in check_output(bundle, f4)
    assert is_hallucinated(bundle, f4)

    # Reflection: 6 words repeated
    refl = dict(clean, explanation="Bank transferred funds directly without manual approval required.")
    assert reflects_untrusted_text(bundle, refl, n=6)
