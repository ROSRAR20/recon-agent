"""Unit tests for eval/expected_decisions.py mapping rules."""

from __future__ import annotations

import pytest

from src.recon_agent.eval.expected_decisions import (
    ExpectedDecision,
    get_expected_decision,
)
from src.recon_agent.eval.scoring import TIER1_ERROR_CLASSES, HARD_NEGATIVE_TYPES
from src.recon_agent.llm.bundle import create_case_bundle


def _make_dummy_bundle(
    bundle_id: str = "b_test_01",
    entity_id: str = "pay_01",
    check_family: str = "fee_tax",
    rule_id: str | None = None,
    facts: dict | None = None,
) -> Any:
    return create_case_bundle(
        bundle_id=bundle_id,
        entity_id=entity_id,
        merchant_id="mer_01",
        check_family=check_family,
        rule_id=rule_id,
        facts=facts or {"amount_paise": 10000},
        untrusted_text={},
    )


def test_control_bundle_maps_to_no_issue():
    b = _make_dummy_bundle(bundle_id="b_ctrl_01")
    exp = get_expected_decision(b, label=None, is_control=True)
    assert exp == ExpectedDecision(
        decision="no_issue",
        predicted_class=None,
        source_label_type="control",
    )


def test_bank_ambiguous_candidates_escalates():
    # Via facts flag
    b1 = _make_dummy_bundle(
        bundle_id="b_bank_01",
        check_family="bank",
        facts={"statement_id": "stmt_01", "ambiguous_candidates": True},
    )
    exp1 = get_expected_decision(b1, label=None, is_control=False)
    assert exp1 == ExpectedDecision(
        decision="escalate_human",
        predicted_class=None,
        source_label_type="bank_ambiguous_candidates",
    )

    # Via rule_id
    b2 = _make_dummy_bundle(
        bundle_id="b_bank_02",
        check_family="bank",
        rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
    )
    exp2 = get_expected_decision(b2, label=None, is_control=False)
    assert exp2 == ExpectedDecision(
        decision="escalate_human",
        predicted_class=None,
        source_label_type="bank_ambiguous_candidates",
    )


def test_ambiguity_types_mapping():
    b = _make_dummy_bundle()

    # NEAR_TOLERANCE -> escalate_human
    lbl_near = {"case_id": "c1", "ambiguity_type": "NEAR_TOLERANCE"}
    exp_near = get_expected_decision(b, label=lbl_near)
    assert exp_near == ExpectedDecision("escalate_human", None, "NEAR_TOLERANCE")

    # EXPLAINED_NO_AMOUNT -> escalate_human
    lbl_no_amt = {"case_id": "c2", "ambiguity_type": "EXPLAINED_NO_AMOUNT"}
    exp_no_amt = get_expected_decision(b, label=lbl_no_amt)
    assert exp_no_amt == ExpectedDecision("escalate_human", None, "EXPLAINED_NO_AMOUNT")

    # EXPLAINED_CONSISTENT -> benign_explained
    lbl_cons = {"case_id": "c3", "ambiguity_type": "EXPLAINED_CONSISTENT"}
    exp_cons = get_expected_decision(b, label=lbl_cons)
    assert exp_cons == ExpectedDecision("benign_explained", None, "EXPLAINED_CONSISTENT")

    # EXPLAINED_INCONSISTENT -> flag + PARTIAL_REFUND_MISMATCH
    lbl_incons = {"case_id": "c4", "ambiguity_type": "EXPLAINED_INCONSISTENT"}
    exp_incons = get_expected_decision(b, label=lbl_incons)
    assert exp_incons == ExpectedDecision("flag", "PARTIAL_REFUND_MISMATCH", "EXPLAINED_INCONSISTENT")


def test_hard_negatives_and_benign_ignore_map_to_no_issue():
    b = _make_dummy_bundle()

    for hn in HARD_NEGATIVE_TYPES:
        lbl = {"case_id": f"c_{hn}", "hard_negative_type": hn}
        exp = get_expected_decision(b, label=lbl)
        assert exp == ExpectedDecision("no_issue", None, f"hard_negative:{hn}")

    lbl_ignore = {"case_id": "c_ignore", "expected_action": "benign_ignore"}
    exp_ignore = get_expected_decision(b, label=lbl_ignore)
    assert exp_ignore == ExpectedDecision("no_issue", None, "benign_ignore")


def test_tier1_error_classes_map_to_flag_with_matching_class():
    b = _make_dummy_bundle()

    for err_cls in TIER1_ERROR_CLASSES:
        lbl = {"case_id": f"c_{err_cls}", "error_class": err_cls}
        exp = get_expected_decision(b, label=lbl)
        assert exp == ExpectedDecision("flag", err_cls, f"tier1:{err_cls}")


def test_unmapped_inputs_raise_value_error():
    b = _make_dummy_bundle()

    # Missing label on non-control, non-bank bundle
    with pytest.raises(ValueError, match="Ground-truth label is required"):
        get_expected_decision(b, label=None, is_control=False)

    # Unmapped ambiguity type
    with pytest.raises(ValueError, match="Unmapped ambiguity_type"):
        get_expected_decision(b, label={"ambiguity_type": "UNKNOWN_AMBIGUITY"})

    # Unmapped hard negative
    with pytest.raises(ValueError, match="Unmapped hard_negative_type"):
        get_expected_decision(b, label={"hard_negative_type": "UNKNOWN_HARD_NEG"})

    # Unmapped error class
    with pytest.raises(ValueError, match="Unmapped error_class"):
        get_expected_decision(b, label={"error_class": "NON_TIER1_CLASS"})

    # Empty label dict
    with pytest.raises(ValueError, match="Unable to map expected decision"):
        get_expected_decision(b, label={})
