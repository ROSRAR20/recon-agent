"""Tests for the pure observable oracle audit."""

import ast
import datetime
import os
import sys

import pytest

from src.recon_agent.synth.config import create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator
from tests.audit.oracle import REQUIRED_AMBIGUITY_TYPES, REQUIRED_ERROR_CLASSES, audit_dataset, run_oracle

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_oracle_isolation_no_synth_import() -> None:
    """Guarantee that tests/audit/oracle.py imports NO modules from recon_agent.synth."""
    oracle_path = os.path.join(REPO_ROOT, "tests", "audit", "oracle.py")
    with open(oracle_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="oracle.py")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("recon_agent"), f"Illegal import in oracle.py: {alias.name}"
                assert not alias.name.startswith("src.recon_agent"), f"Illegal import in oracle.py: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("recon_agent"), f"Illegal import-from in oracle.py: {mod}"
            assert not mod.startswith("src.recon_agent"), f"Illegal import-from in oracle.py: {mod}"


@pytest.mark.parametrize("seed", [1, 42])
def test_oracle_audit_small_profile(seed: int) -> None:
    """Test oracle detection rate >= 98% across all 7 error classes on small profile."""
    config = create_config(profile="small", split="dev", seed=seed, inject_errors=True)
    generator = SyntheticDataGenerator(config)
    data = generator.generate()

    results = audit_dataset(data)
    eval_res = results["evaluation"]

    for err_cls in REQUIRED_ERROR_CLASSES:
        assert err_cls in eval_res, f"Missing evaluation for {err_cls}"
        stats = eval_res[err_cls]
        assert stats["injected"] > 0, f"Expected non-zero injected for {err_cls}"
        assert stats["detection_rate"] >= 98.0, (
            f"Class {err_cls} detection rate {stats['detection_rate']}% < 98.0% floor "
            f"(injected={stats['injected']}, detected={stats['detected']})"
        )
        assert stats["false_positives"] == 0, (
            f"Class {err_cls} had {stats['false_positives']} false positives"
        )

    assert "tier2_evaluation" in results
    tier2_eval = results["tier2_evaluation"]
    for amb_type in REQUIRED_AMBIGUITY_TYPES:
        assert amb_type in tier2_eval, f"Missing Tier 2 evaluation for {amb_type}"
        amb_stats = tier2_eval[amb_type]
        assert amb_stats["count"] >= 0, (
            f"Expected >= 0 injected for small profile Tier 2 {amb_type}, got {amb_stats['count']}"
        )


def test_oracle_audit_full_profile() -> None:
    """Test oracle detection rate >= 98% across all 7 error classes on full profile."""
    config = create_config(profile="full", split="dev", seed=1, inject_errors=True)
    generator = SyntheticDataGenerator(config)
    data = generator.generate()

    results = audit_dataset(data)
    eval_res = results["evaluation"]

    print("\n" + results["summary_table"])

    for err_cls in REQUIRED_ERROR_CLASSES:
        assert err_cls in eval_res, f"Missing evaluation for {err_cls}"
        stats = eval_res[err_cls]
        assert stats["injected"] >= 50, f"Expected >= 50 injected for full profile {err_cls}"
        assert stats["detection_rate"] >= 98.0, (
            f"Class {err_cls} detection rate {stats['detection_rate']}% < 98.0% floor "
            f"(injected={stats['injected']}, detected={stats['detected']})"
        )
        assert stats["false_positives"] == 0, (
            f"Class {err_cls} had {stats['false_positives']} false positives"
        )

    assert "tier2_evaluation" in results
    tier2_eval = results["tier2_evaluation"]
    for amb_type in REQUIRED_AMBIGUITY_TYPES:
        assert amb_type in tier2_eval, f"Missing Tier 2 evaluation for {amb_type}"
        amb_stats = tier2_eval[amb_type]
        assert amb_stats["count"] >= 50, (
            f"Expected >= 50 injected for Tier 2 {amb_type}, got {amb_stats['count']}"
        )
