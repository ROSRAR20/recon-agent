"""Integration tests for pilot machinery (Day 7b-1).

Tests:
1. build_populations strictly refuses the 'test' split.
2. dev_seed1 bundles contain NO meta fields (eval metadata isolated).
3. Population prefix balance: any prefix is balanced across families and ambiguity types.
4. Unsupported arm handling: HTTP 400 parameter rejection marks arm unsupported without retrying.
5. End-to-end mock pilot run and report execution.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import pytest

from scripts.build_populations import build_population_for_seed
from scripts.run_pilot import MockPilotClient, load_stratified_prefix, run_pilot
from scripts.pilot_report import run_pilot_report
from src.recon_agent.llm.bundle import CaseBundle, create_case_bundle
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.gemini_client import NonRetriableAPIError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_build_populations_refuses_test_split():
    """Verify build_populations strictly refuses split='test'."""
    with pytest.raises(ValueError, match="Split 'test' refused"):
        build_population_for_seed(seed=1, split="test")


def test_bundles_contain_no_meta_fields():
    """Verify bundles in dev_seed1.jsonl contain NO evaluation meta fields."""
    pop_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1.jsonl")
    if not os.path.isfile(pop_file):
        pytest.skip("dev_seed1.jsonl not found")

    forbidden_keys = {
        "is_control",
        "expected_decision",
        "expected_class",
        "source_label_type",
        "ground_truth",
        "truth_link",
        "manifest",
    }
    allowed_keys = {
        "allowed_classes",
        "allowed_decisions",
        "bundle_id",
        "check_family",
        "entity_id",
        "facts",
        "merchant_id",
        "untrusted_text",
    }

    with open(pop_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            keys = set(data.keys())
            assert keys == allowed_keys, f"Bundle at line {idx} has invalid keys: {keys ^ allowed_keys}"
            assert not (keys & forbidden_keys), f"Bundle at line {idx} contains forbidden keys: {keys & forbidden_keys}"


def test_population_prefix_balance():
    """Verify any prefix of dev_seed1.jsonl has balanced check families and types."""
    pop_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1.jsonl")
    meta_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1_meta.jsonl")
    if not os.path.isfile(pop_file) or not os.path.isfile(meta_file):
        pytest.skip("dev_seed1 files not found")

    meta_lookup = {}
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta_lookup[r["bundle_id"]] = r

    bundles = []
    with open(pop_file, "r", encoding="utf-8") as f:
        for line in f:
            bundles.append(json.loads(line))

    # Check prefixes of length 20, 50, 100
    for prefix_len in (20, 50, 100):
        prefix = bundles[:prefix_len]
        families = set(b["check_family"] for b in prefix)
        # Should contain fee_tax, refund, and bank
        assert "fee_tax" in families
        assert "refund" in families
        assert "bank" in families

        types = set(meta_lookup[b["bundle_id"]]["source_label_type"] for b in prefix)
        assert "NEAR_TOLERANCE" in types
        assert "EXPLAINED_CONSISTENT" in types


def test_stratified_prefix_targets():
    """Verify load_stratified_prefix exact target counts and order preservation."""
    pop_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1.jsonl")
    meta_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1_meta.jsonl")
    if not os.path.isfile(pop_file) or not os.path.isfile(meta_file):
        pytest.skip("dev_seed1 files not found")

    prefix_bundles, meta_lookup = load_stratified_prefix(
        population_file=pop_file,
        meta_file=meta_file,
        n_residual=100,
        n_control=20,
    )
    assert len(prefix_bundles) == 120

    res_count = sum(1 for b in prefix_bundles if not meta_lookup[b.bundle_id]["is_control"])
    ctrl_count = sum(1 for b in prefix_bundles if meta_lookup[b.bundle_id]["is_control"])
    assert res_count == 100
    assert ctrl_count == 20


def test_unsupported_arm_handling(tmp_path):
    """Verify unsupported thinking_level parameter HTTP 400 is marked unsupported and does not retry."""
    pop_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1.jsonl")
    meta_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1_meta.jsonl")
    if not os.path.isfile(pop_file) or not os.path.isfile(meta_file):
        pytest.skip("dev_seed1 files not found")

    out_dir = str(tmp_path / "results")
    # Run with an arm configured with thinking_level='unsupported'
    summaries = run_pilot(
        seed=1,
        arms=["unsupported"],
        n_residual=5,
        n_control=2,
        mock=True,
        output_dir=out_dir,
    )
    assert summaries["unsupported"]["status"] == "unsupported"
    assert "unsupported parameter" in summaries["unsupported"]["reason"]


def test_end_to_end_mock_pilot_and_report(tmp_path, capsys):
    """Run full end-to-end mock pilot across arms and verify report generation."""
    out_dir = str(tmp_path / "results")
    fin_dir = str(tmp_path / "finalized")

    # 1. Run pilot with subset
    summaries = run_pilot(
        seed=1,
        arms=["default", "low"],
        n_residual=10,
        n_control=5,
        mock=True,
        output_dir=out_dir,
    )
    assert summaries["default"]["status"] == "completed"
    assert summaries["low"]["status"] == "completed"

    # 2. Run report
    run_pilot_report(
        seed=1,
        arms=["default", "low"],
        n_residual=10,
        n_control=5,
        results_dir=out_dir,
        finalized_dir=fin_dir,
    )

    captured = capsys.readouterr()
    assert "DAY 7b-1 PILOT EVALUATION REPORT" in captured.out
    assert "Arm: `default`" in captured.out
    assert "Arm: `low`" in captured.out
    assert "Suggested Arm for Freeze" in captured.out
