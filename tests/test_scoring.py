"""Unit tests for the scoring harness and evaluation contract."""

import pytest

from src.recon_agent.eval.scoring import (
    score_predictions,
    wilson_ci,
)
from src.recon_agent.pipeline.types import BankSettlementLink, Prediction, SystemOutput


def test_wilson_ci_exact_values() -> None:
    """Test Wilson score CI against analytical reference values."""
    # N=0 returns (0, 0)
    assert wilson_ci(0, 0) == (0.0, 0.0)

    # 50/50: center ~ 0.9643, margin ~ 0.0357 -> lower ~ 0.9287, upper = 1.0
    low, high = wilson_ci(50, 50)
    assert 0.928 <= low <= 0.930
    assert high == 1.0

    # 0/50: center ~ 0.0357, margin ~ 0.0357 -> lower = 0.0, upper ~ 0.0713
    low, high = wilson_ci(0, 50)
    assert low == 0.0
    assert 0.070 <= high <= 0.073

    # 25/50: center ~ 0.5000, margin ~ 0.1335 -> lower ~ 0.3665, upper ~ 0.6335
    low, high = wilson_ci(25, 50)
    assert 0.365 <= low <= 0.368
    assert 0.632 <= high <= 0.635


def test_perfect_predictions_give_precision_and_recall_one() -> None:
    """Perfect predictions on Tier 1 labels yield Precision=1.0, Recall=1.0, F1=1.0."""
    labels = [
        {
            "case_id": "case_01",
            "merchant_id": "mer_01",
            "error_class": "DUPLICATE_PAYMENT",
            "entity_ids": ["pay_01", "pay_02"],
            "expected_action": "flag_duplicate_payment",
        },
        {
            "case_id": "case_02",
            "merchant_id": "mer_01",
            "error_class": "MISSING_SETTLEMENT",
            "entity_ids": ["pay_03"],
            "expected_action": "flag_missing_settlement",
        },
    ]

    preds = [
        Prediction(
            merchant_id="mer_01",
            predicted_class="DUPLICATE_PAYMENT",
            action="flag",
            entity_ids=("pay_01",),
        ),
        Prediction(
            merchant_id="mer_01",
            predicted_class="MISSING_SETTLEMENT",
            action="flag",
            entity_ids=("pay_03",),
        ),
    ]

    res = score_predictions(preds, labels, truth_links={})

    dup_res = res.tier1_class_correct["DUPLICATE_PAYMENT"]
    assert dup_res.tp == 1
    assert dup_res.fp == 0
    assert dup_res.fn == 0
    assert dup_res.precision == 1.0
    assert dup_res.recall == 1.0
    assert dup_res.f1 == 1.0

    mis_res = res.tier1_class_correct["MISSING_SETTLEMENT"]
    assert mis_res.tp == 1
    assert mis_res.fp == 0
    assert mis_res.fn == 0
    assert mis_res.precision == 1.0
    assert mis_res.recall == 1.0
    assert mis_res.f1 == 1.0

    assert res.clean_false_positives == 0


def test_extra_flag_on_clean_data_is_false_positive() -> None:
    """An unassigned prediction that flags clean data counts as clean FP and class FP."""
    labels = [
        {
            "case_id": "case_01",
            "merchant_id": "mer_01",
            "error_class": "MISSING_SETTLEMENT",
            "entity_ids": ["pay_01"],
            "expected_action": "flag_missing_settlement",
        }
    ]

    preds = [
        # Correct prediction
        Prediction(
            merchant_id="mer_01",
            predicted_class="MISSING_SETTLEMENT",
            action="flag",
            entity_ids=("pay_01",),
        ),
        # Extra spurious prediction on clean entity pay_clean
        Prediction(
            merchant_id="mer_01",
            predicted_class="MISSING_SETTLEMENT",
            action="flag",
            entity_ids=("pay_clean",),
        ),
    ]

    res = score_predictions(preds, labels, truth_links={})

    mis_res = res.tier1_class_correct["MISSING_SETTLEMENT"]
    assert mis_res.tp == 1
    assert mis_res.fp == 1
    assert mis_res.fn == 0
    assert mis_res.precision == 0.5  # 1 / (1 + 1)
    assert mis_res.recall == 1.0

    assert res.clean_false_positives == 1
    assert res.confusion_matrix["CLEAN"]["MISSING_SETTLEMENT"] == 1


def test_flag_on_hard_negative_counts_against_hn_type() -> None:
    """A prediction flagging a hard negative counts as an FP on that HN type."""
    labels = [
        {
            "case_id": "case_hn_01",
            "merchant_id": "mer_01",
            "hard_negative_type": "ROUNDING",
            "entity_ids": ["pay_round_01"],
            "expected_action": "benign_ignore",
        }
    ]

    preds = [
        Prediction(
            merchant_id="mer_01",
            predicted_class="FEE_TAX_MISMATCH",
            action="flag",
            entity_ids=("pay_round_01",),
        )
    ]

    res = score_predictions(preds, labels, truth_links={})

    hn_res = res.hard_negatives["ROUNDING"]
    assert hn_res.total == 1
    assert hn_res.flagged == 1
    assert hn_res.fpr == 1.0

    # Also counts as FP for the predicted error class
    fee_res = res.tier1_class_correct["FEE_TAX_MISMATCH"]
    assert fee_res.fp == 1
    assert fee_res.tp == 0

    assert res.confusion_matrix["HARD_NEGATIVE"]["FEE_TAX_MISMATCH"] == 1


def test_duplicate_predictions_do_not_double_count() -> None:
    """Multiple predictions for the same label: only one matches, the duplicate is clean FP."""
    labels = [
        {
            "case_id": "case_01",
            "merchant_id": "mer_01",
            "error_class": "MISSING_SETTLEMENT",
            "entity_ids": ["pay_01"],
            "expected_action": "flag_missing_settlement",
        }
    ]

    preds = [
        Prediction(
            merchant_id="mer_01",
            predicted_class="MISSING_SETTLEMENT",
            action="flag",
            entity_ids=("pay_01",),
            rule_id="RULE_A",
        ),
        Prediction(
            merchant_id="mer_01",
            predicted_class="MISSING_SETTLEMENT",
            action="flag",
            entity_ids=("pay_01",),
            rule_id="RULE_B",
        ),
    ]

    res = score_predictions(preds, labels, truth_links={})

    mis_res = res.tier1_class_correct["MISSING_SETTLEMENT"]
    assert mis_res.tp == 1
    assert mis_res.fp == 1  # Duplicate prediction cannot match assigned label -> clean FP
    assert mis_res.fn == 0
    assert res.clean_false_positives == 1


def test_right_entity_wrong_class_is_agnostic_tp_and_class_specific_fp_fn() -> None:
    """Right entity with wrong class is class-agnostic TP, but class-specific FP + FN."""
    labels = [
        {
            "case_id": "case_01",
            "merchant_id": "mer_01",
            "error_class": "DUPLICATE_PAYMENT",
            "entity_ids": ["pay_01"],
            "expected_action": "flag_duplicate_payment",
        }
    ]

    # Model flags the entity, but mistakenly classifies it as MISSING_SETTLEMENT
    preds = [
        Prediction(
            merchant_id="mer_01",
            predicted_class="MISSING_SETTLEMENT",
            action="flag",
            entity_ids=("pay_01",),
        )
    ]

    res = score_predictions(preds, labels, truth_links={})

    # Class-Agnostic: Anomaly was flagged on pay_01 -> TP for DUPLICATE_PAYMENT
    dup_agnostic = res.tier1_class_agnostic["DUPLICATE_PAYMENT"]
    assert dup_agnostic.tp == 1
    assert dup_agnostic.fn == 0

    # Class-Correct:
    # DUPLICATE_PAYMENT was not classified correctly -> FN
    dup_correct = res.tier1_class_correct["DUPLICATE_PAYMENT"]
    assert dup_correct.tp == 0
    assert dup_correct.fn == 1

    # MISSING_SETTLEMENT was predicted erroneously -> FP
    mis_correct = res.tier1_class_correct["MISSING_SETTLEMENT"]
    assert mis_correct.fp == 1
    assert mis_correct.tp == 0

    assert res.confusion_matrix["DUPLICATE_PAYMENT"]["MISSING_SETTLEMENT"] == 1


def test_one_to_one_assignment_greedy_by_overlap_and_tiebreak() -> None:
    """Assignment is strictly one-to-one, preferring larger overlap size."""
    labels = [
        {
            "case_id": "case_small",
            "merchant_id": "mer_01",
            "error_class": "DUPLICATE_PAYMENT",
            "entity_ids": ["pay_01"],
            "expected_action": "flag",
        },
        {
            "case_id": "case_large",
            "merchant_id": "mer_01",
            "error_class": "DUPLICATE_PAYMENT",
            "entity_ids": ["pay_01", "pay_02"],
            "expected_action": "flag",
        },
    ]

    # Pred 1 has overlap of 2 with case_large, and overlap of 1 with case_small
    preds = [
        Prediction(
            merchant_id="mer_01",
            predicted_class="DUPLICATE_PAYMENT",
            action="flag",
            entity_ids=("pay_01", "pay_02"),
        )
    ]

    res = score_predictions(preds, labels, truth_links={})
    # Pred 1 should match case_large due to larger overlap
    assert res.tier1_class_correct["DUPLICATE_PAYMENT"].tp == 1
    assert res.tier1_class_correct["DUPLICATE_PAYMENT"].fn == 1


def test_matching_metrics_and_ambiguity_tier() -> None:
    """Test bank-to-settlement matching metrics and ambiguity tier accuracy."""
    truth_links = {
        "stmt_01": "setl_01",
        "stmt_02": "setl_02",
        "stmt_03": None,  # orphan
    }

    bank_links = [
        BankSettlementLink("stmt_01", "setl_01"),  # correct
        BankSettlementLink("stmt_03", "setl_02"),  # wrong match (stmt_03 is orphan)
    ]

    labels = [
        {
            "case_id": "case_amb_01",
            "merchant_id": "mer_01",
            "ambiguity_type": "NEAR_TOLERANCE",
            "entity_ids": ["pay_near_01"],
            "expected_action": "escalate_human",
        },
        {
            "case_id": "case_amb_02",
            "merchant_id": "mer_01",
            "ambiguity_type": "EXPLAINED_CONSISTENT",
            "entity_ids": ["rfnd_exp_01"],
            "expected_action": "benign_explained",
        },
    ]

    preds = [
        # Predicted escalate_human for NEAR_TOLERANCE -> correct
        Prediction(
            merchant_id="mer_01",
            predicted_class="UNKNOWN",
            action="escalate_human",
            entity_ids=("pay_near_01",),
        ),
        # Nothing predicted for EXPLAINED_CONSISTENT -> defaults to ignore -> wrong (expected benign_explained)
    ]

    system_output = SystemOutput(predictions=preds, bank_links=bank_links)
    res = score_predictions(system_output, labels, truth_links)

    # Matching: 2 links made, 1 correct, 1 wrong
    assert res.matching.links_made == 2
    assert res.matching.correct_links == 1
    assert res.matching.wrong_links == 1
    assert res.matching.false_match_rate == 0.5  # 1/2
    assert res.matching.match_rate == 0.5  # 1/2 (total true settlements is 2: stmt_01 and stmt_02)

    # Ambiguity
    assert res.ambiguity_tier["NEAR_TOLERANCE"].accuracy == 1.0  # 1/1
    assert res.ambiguity_tier["EXPLAINED_CONSISTENT"].accuracy == 0.0  # 0/1 (ignore != benign_explained)

