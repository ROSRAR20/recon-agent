"""Tests for statistical feature leakage audit using Mann-Whitney U / AUC."""

import ast
import os
import pytest

from src.recon_agent.synth.config import create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator
from tests.audit.leakage import (
    SKIPPED_FEATURES,
    TIER_1_ERROR_CLASSES,
    TIER_2_AMBIGUITY_TYPES,
    WHITELISTED_FEATURES,
    audit_leakage,
    compute_auc,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_leakage_isolation_no_synth_import() -> None:
    """Guarantee that tests/audit/leakage.py imports NO modules from recon_agent.synth."""
    leakage_path = os.path.join(REPO_ROOT, "tests", "audit", "leakage.py")
    with open(leakage_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="leakage.py")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("recon_agent"), f"Illegal import in leakage.py: {alias.name}"
                assert not alias.name.startswith("src.recon_agent"), f"Illegal import in leakage.py: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("recon_agent"), f"Illegal import-from in leakage.py: {mod}"
            assert not mod.startswith("src.recon_agent"), f"Illegal import-from in leakage.py: {mod}"


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
    """Assert that for all classes, every non-whitelisted feature has 0.40 <= AUC <= 0.60."""
    config = create_config(profile="full", split="dev", seed=1, inject_errors=True)
    generator = SyntheticDataGenerator(config)
    data = generator.generate()

    # 1. Before Audit: Naive comparison against broad un-eligible populations
    before_res = audit_leakage(data, use_eligible_population=False)
    print("\n" + "=" * 80)
    print("BEFORE LEAKAGE AUDIT TABLE (Naive comparison against broad populations)")
    print("=" * 80)
    print(before_res["summary_table"])

    # 2. After Audit: Comparison strictly against eligible clean populations
    after_res = audit_leakage(data, use_eligible_population=True)
    print("\n" + "=" * 80)
    print("AFTER LEAKAGE AUDIT TABLE (Comparison against same eligible populations)")
    print("=" * 80)
    print(after_res["summary_table"])

    results = after_res["results"]
    pop_sizes = after_res["population_sizes"]
    concentrations = after_res["merchant_concentrations"]

    # Verify all 7 Tier 1 error classes + 4 Tier 2 ambiguity types are evaluated
    expected_classes = set(TIER_1_ERROR_CLASSES + TIER_2_AMBIGUITY_TYPES)
    assert set(results.keys()) == expected_classes, f"Mismatch in evaluated classes: {set(results.keys()) ^ expected_classes}"

    failures = []
    # Check AUC bounds [0.40, 0.60]
    for err_cls, features in results.items():
        for feat_name, auc in features.items():
            if (err_cls, feat_name) in WHITELISTED_FEATURES:
                continue
            if (err_cls, feat_name) in SKIPPED_FEATURES:
                continue

            if not (0.40 <= auc <= 0.60):
                failures.append(
                    f"Leakage violation: {err_cls} feature '{feat_name}' has AUC {auc:.4f} "
                    f"outside bounds [0.40, 0.60] (N_pos={pop_sizes[err_cls]['pos']}, N_neg={pop_sizes[err_cls]['neg']})"
                )

    # Check merchant concentration for Tier 1 classes
    for err_cls in TIER_1_ERROR_CLASSES:
        conc = concentrations.get(err_cls, {})
        if not conc.get("passes", False):
            failures.append(
                f"Concentration violation: {err_cls} max merchant count {conc.get('max')} "
                f"> 3x uniform share {conc.get('max_allowed')}"
            )

    assert not failures, "\n".join(failures)
    assert after_res["all_passed"] is True
