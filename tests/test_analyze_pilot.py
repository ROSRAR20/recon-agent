"""Tests for scripts/analyze_pilot.py error analysis functions."""

from __future__ import annotations

from decimal import Decimal
import os
import pytest

from scripts.analyze_pilot import (
    build_confusion_matrix,
    extract_f4_offending_tokens,
    format_confusion_matrix,
    run_error_analysis,
)


def test_extract_f4_offending_tokens():
    bundle_dict = {
        "facts": {
            "amount_paise": 5000,
            "refund_difference_paise": 250,
            "candidates_count": 2,
        },
        "untrusted_text": {
            "notes": "Customer requested refund. Deducted ₹2.50 fee.",
        },
    }

    # 1. Grounded tokens: Rs 2.50 (250 paise) and 2 candidates
    valid_output = {
        "explanation": "Deducted Rs 2.50 matching the refund difference across 2 candidates.",
        "cited_amount_paise": 250,
        "verbatim_quote": "Deducted ₹2.50 fee.",
    }
    offending_valid = extract_f4_offending_tokens(bundle_dict, valid_output)
    assert offending_valid == []

    # 2. Fabricated / ungrounded amount token: "Rs 99.99"
    invalid_output = {
        "explanation": "Calculated extra charge of Rs 99.99 and 42 random items.",
        "cited_amount_paise": None,
        "verbatim_quote": None,
    }
    offending_invalid = extract_f4_offending_tokens(bundle_dict, invalid_output)
    assert "Rs 99.99" in offending_invalid
    assert "42" in offending_invalid


def test_confusion_matrix_helpers():
    expected = ["no_issue", "flag", "escalate_human", "benign_explained"]
    predicted = ["no_issue", "escalate_human", "escalate_human", "benign_explained"]

    cm = build_confusion_matrix(expected, predicted)
    assert cm["no_issue"]["no_issue"] == 1
    assert cm["flag"]["escalate_human"] == 1
    assert cm["escalate_human"]["escalate_human"] == 1
    assert cm["benign_explained"]["benign_explained"] == 1
    assert cm["flag"]["flag"] == 0

    table = format_confusion_matrix(cm)
    assert "| Expected \\ Predicted |" in table
    assert "| **flag** |" in table


def test_run_error_analysis_real():
    """Run read-only error analysis on dev seed 1 (run 0) across all three arms."""
    findings = run_error_analysis(
        seed=1,
        arms=["default", "low", "high"],
        n_residual=100,
        n_control=20,
    )
    assert "default" in findings
    assert "low" in findings
    assert "high" in findings

    # Default arm assertions
    assert len(findings["default"]["f4_cases"]) == 21
    assert len(findings["default"]["unblocked_hallucinations"]) == 1
    assert len(findings["default"]["unsafe_raw_cases"]) == 9
    assert len(findings["default"]["accepted_wrong_cases"]) == 2
    assert len(findings["default"]["v8_rejections"]) == 1
    assert len(findings["default"]["no_amount_in_quote_cases"]) == 10

    # Low arm assertions
    assert len(findings["low"]["f4_cases"]) == 21
    assert len(findings["low"]["unblocked_hallucinations"]) == 0
    assert len(findings["low"]["unsafe_raw_cases"]) == 9
    assert len(findings["low"]["accepted_wrong_cases"]) == 2
    assert len(findings["low"]["v8_rejections"]) == 7
    assert len(findings["low"]["no_amount_in_quote_cases"]) == 9

    # High arm assertions
    assert len(findings["high"]["f4_cases"]) == 22
    assert len(findings["high"]["unblocked_hallucinations"]) == 0
    assert len(findings["high"]["unsafe_raw_cases"]) == 0
    assert len(findings["high"]["accepted_wrong_cases"]) == 3
    assert len(findings["high"]["v8_rejections"]) == 4
    assert len(findings["high"]["no_amount_in_quote_cases"]) == 0
