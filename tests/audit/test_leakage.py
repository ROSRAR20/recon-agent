"""Tests for statistical feature leakage audit using Mann-Whitney U / AUC."""

import pytest

from src.recon_agent.synth.config import create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator
from tests.audit.leakage import (
    SKIPPED_FEATURES,
    WHITELISTED_FEATURES,
    audit_leakage,
    compute_auc,
)


def test_compute_auc_deterministic() -> None:
    """Verify Mann-Whitney U / AUC calculation on known analytical cases."""
    # Complete separation: all positives > all negatives -> AUC = 1.0
    assert compute_auc([10, 20, 30], [1, 2, 3]) == 1.0
    # Complete inverse separation: all positives < all negatives -> AUC = 0.0
    assert compute_auc([1, 2, 3], [10, 20, 30]) == 0.0
    # Identical values: ties -> AUC = 0.5
    assert compute_auc([5, 5], [5, 5]) == 0.5
    # Interleaved: [1, 2, 3, 4] with pos=[2, 4], neg=[1, 3] -> AUC = 0.75
    assert compute_auc([2, 4], [1, 3]) == 0.75


def test_leakage_audit_full_profile() -> None:
    """Assert that for all error classes, every non-whitelisted feature has 0.30 <= AUC <= 0.70."""
    config = create_config(profile="full", split="dev", seed=1, inject_errors=True)
    generator = SyntheticDataGenerator(config)
    data = generator.generate()

    leakage_res = audit_leakage(data)
    results = leakage_res["results"]

    print("\n" + leakage_res["summary_table"])

    # Ensure all 7 classes are evaluated
    assert len(results) == 7, f"Expected 7 evaluated classes, got {len(results)}"

    failures = []
    for err_cls, features in results.items():
        for feat_name, auc in features.items():
            if (err_cls, feat_name) in WHITELISTED_FEATURES:
                continue
            if (err_cls, feat_name) in SKIPPED_FEATURES:
                continue

            if not (0.30 <= auc <= 0.70):
                failures.append(
                    f"Leakage violation: {err_cls} feature '{feat_name}' has AUC {auc:.4f} "
                    f"outside bounds [0.30, 0.70]"
                )

    assert not failures, "\n".join(failures)
