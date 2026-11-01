"""Expectation tests for System B0 baseline on synthetic reconciliation benchmarks."""

import pytest

from src.recon_agent.eval.scoring import load_ground_truth, score_predictions
from src.recon_agent.pipeline.loader import load_dataset
from src.recon_agent.synth.config import create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator
from src.recon_agent.systems.b0 import run_b0


def _assert_b0_expectations(dataset_dir: str, seed: int) -> None:
    dataset = load_dataset(dataset_dir)
    sys_out = run_b0(dataset)
    labels, truth_links = load_ground_truth(dataset_dir)
    res = score_predictions(sys_out, labels, truth_links, system_id="b0", split="dev", seed=seed)

    # 1. ORPHAN_BANK_CREDIT: recall = 100% (both class-correct and class-agnostic)
    orphan_cc = res.tier1_class_correct["ORPHAN_BANK_CREDIT"]
    orphan_ca = res.tier1_class_agnostic["ORPHAN_BANK_CREDIT"]
    assert (orphan_cc.tp + orphan_cc.fn) > 0, "ORPHAN_BANK_CREDIT ground truth total must be > 0"
    assert orphan_cc.recall == 1.0, f"Expected 100% class-correct recall for ORPHAN, got {orphan_cc.recall}"
    assert orphan_ca.recall == 1.0, f"Expected 100% class-agnostic recall for ORPHAN, got {orphan_ca.recall}"

    # 2. MISSING_SETTLEMENT: recall = 100% (both class-correct and class-agnostic)
    missing_cc = res.tier1_class_correct["MISSING_SETTLEMENT"]
    missing_ca = res.tier1_class_agnostic["MISSING_SETTLEMENT"]
    assert (missing_cc.tp + missing_cc.fn) > 0, "MISSING_SETTLEMENT ground truth total must be > 0"
    assert missing_cc.recall == 1.0, f"Expected 100% class-correct recall for MISSING, got {missing_cc.recall}"
    assert missing_ca.recall == 1.0, f"Expected 100% class-agnostic recall for MISSING, got {missing_ca.recall}"

    # 3. DUPLICATE, FEE_TAX, PARTIAL_REFUND, CUTOFF: recall = 0%
    zero_recall_classes = [
        "DUPLICATE_PAYMENT",
        "FEE_TAX_MISMATCH",
        "PARTIAL_REFUND_MISMATCH",
        "CUTOFF_DRIFT",
    ]
    for c in zero_recall_classes:
        cc = res.tier1_class_correct[c]
        ca = res.tier1_class_agnostic[c]
        assert (cc.tp + cc.fn) > 0, f"{c} ground truth total must be > 0"
        assert cc.recall == 0.0, f"Expected 0% class-correct recall for {c}, got {cc.recall}"
        assert ca.recall == 0.0, f"Expected 0% class-agnostic recall for {c}, got {ca.recall}"

    # 4. NO_UTR_NARRATION: flagged = 100%
    no_utr = res.hard_negatives["NO_UTR_NARRATION"]
    assert no_utr.total > 0, "NO_UTR_NARRATION total must be > 0"
    assert no_utr.flagged == no_utr.total, (
        f"Expected 100% NO_UTR_NARRATION flagged, got {no_utr.flagged}/{no_utr.total}"
    )

    # 5. PARTIAL_REFUND_CONSISTENT, RETRY_AFTER_FAIL, FAILED_PAYMENT: flagged = 0
    zero_flagged_hard_negatives = [
        "PARTIAL_REFUND_CONSISTENT",
        "RETRY_AFTER_FAIL",
        "FAILED_PAYMENT",
    ]
    for hn_name in zero_flagged_hard_negatives:
        hn = res.hard_negatives[hn_name]
        assert hn.total > 0, f"{hn_name} total must be > 0"
        assert hn.flagged == 0, f"Expected 0 flagged for {hn_name}, got {hn.flagged}/{hn.total}"


def test_b0_expectations_small_profile(tmp_path) -> None:
    """Verify B0 baseline detection and false positive expectations on small dev profile."""
    seed = 42
    config = create_config(profile="small", split="dev", seed=seed, inject_errors=True)
    dir_path = str(tmp_path / "dataset_small")
    SyntheticDataGenerator(config).write_to_directory(dir_path)
    _assert_b0_expectations(dir_path, seed)


@pytest.mark.slow
def test_b0_expectations_full_profile(tmp_path) -> None:
    """Verify B0 baseline detection and false positive expectations on full dev seed 1."""
    seed = 1
    config = create_config(profile="full", split="dev", seed=seed, inject_errors=True)
    dir_path = str(tmp_path / "dataset_full")
    SyntheticDataGenerator(config).write_to_directory(dir_path)
    _assert_b0_expectations(dir_path, seed)

