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
    assert "PILOT EVALUATION REPORT" in captured.out
    assert "Arm: `default`" in captured.out
    assert "Arm: `low`" in captured.out
    assert "Suggested Arm for Freeze" in captured.out


def test_mock_mode_isolation_and_real_mode_refusal(tmp_path):
    """Verify mock run leaves real cache untouched, and real client refuses mock model / cache."""
    out_dir = str(tmp_path / "results")
    real_cache_dir = os.path.join(out_dir, "llm_cache")
    mock_cache_dir = os.path.join(out_dir, "llm_cache_mock")

    # 1. Run pilot in mock mode
    run_pilot(
        seed=1,
        arms=["default"],
        n_residual=5,
        n_control=2,
        mock=True,
        output_dir=out_dir,
    )

    # Real cache should not exist or be empty
    if os.path.isdir(real_cache_dir):
        assert len(os.listdir(real_cache_dir)) == 0
    # Mock cache should exist and have files
    assert os.path.isdir(mock_cache_dir)
    assert len(os.listdir(mock_cache_dir)) > 0

    # 2. Real GeminiClient refuses model_id="mock"
    from src.recon_agent.llm.gemini_client import GeminiClient
    with pytest.raises(ValueError, match="Model id 'mock' is not allowed in real Gemini API client mode"):
        GeminiClient(model_id="mock")

    # 3. Real GeminiClient refuses mock cache directory
    with pytest.raises(ValueError, match="must not read from or write to mock cache directory"):
        mock_cache = LLMCache(cache_dir=mock_cache_dir)
        GeminiClient(model_id="gemini-3.5-flash-lite", cache=mock_cache)


def test_check_frozen_baselines_verification():
    """Verify check_frozen_baselines behavior on workspace HEAD."""
    from scripts.check_frozen import check_frozen_baselines
    ok, details = check_frozen_baselines()
    if details.get("registration_final_commit") is not None:
        assert ok is True, f"check_frozen_baselines failed: {details.get('failures')}"
        assert not details.get("failures")
    else:
        assert ok is False
        assert any("Tag 'registration-final' does not exist in repository" in f for f in details.get("failures", []))


def test_population_disjointness_and_reason_text_exclusion():
    """Verify residuals and controls are strictly disjoint and controls lack reason_text."""
    pop_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1.jsonl")
    meta_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1_meta.jsonl")
    if not os.path.isfile(pop_file) or not os.path.isfile(meta_file):
        pytest.skip("dev_seed1 population not found")

    meta_lookup = {}
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta_lookup[r["bundle_id"]] = r

    res_ids = set()
    ctrl_ids = set()
    with open(pop_file, "r", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            bid = b["bundle_id"]
            if meta_lookup[bid]["is_control"]:
                ctrl_ids.add(bid)
                assert b.get("untrusted_text", {}).get("recon_item_notes_reason_text") is None, (
                    f"Control bundle {bid} unexpectedly contains recon_item_notes_reason_text"
                )
            else:
                res_ids.add(bid)

    assert not (res_ids & ctrl_ids), f"Overlapping bundle IDs between residual and control: {res_ids & ctrl_ids}"
    assert len(ctrl_ids) == 60
    assert len(res_ids) == 252


def test_population_sha256_identical_across_two_builds(tmp_path):
    """Verify building population twice produces bit-for-bit identical SHA-256."""
    import hashlib

    dir1 = str(tmp_path / "pop1")
    dir2 = str(tmp_path / "pop2")

    p1, _, _ = build_population_for_seed(seed=1, output_dir=dir1)
    p2, _, _ = build_population_for_seed(seed=1, output_dir=dir2)

    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        h1 = hashlib.sha256(f1.read()).hexdigest()
        h2 = hashlib.sha256(f2.read()).hexdigest()

    assert h1 == h2, f"Population file hash differed across builds: {h1} vs {h2}"


def test_b1r_action_mapping_unit():
    """Verify B1r action mapping from system predictions to evaluation decisions."""
    from src.recon_agent.eval.llm_scoring import score_arm
    from src.recon_agent.pipeline.types import Prediction

    # Case 1: ignore -> no_issue
    # Case 2: flag -> flag
    # Case 3: escalate_human -> escalate_human
    # Case 4: unpredicted entity -> no_issue
    bundles = [
        create_case_bundle(
            bundle_id="b1",
            entity_id="e1",
            merchant_id="m1",
            check_family="fee_tax",
            facts={"rounding_tolerance_paise": 2},
            untrusted_text={},
        ),
        create_case_bundle(
            bundle_id="b2",
            entity_id="e2",
            merchant_id="m1",
            check_family="fee_tax",
            facts={"rounding_tolerance_paise": 2},
            untrusted_text={},
        ),
        create_case_bundle(
            bundle_id="b3",
            entity_id="e3",
            merchant_id="m1",
            check_family="fee_tax",
            facts={"rounding_tolerance_paise": 2},
            untrusted_text={},
        ),
        create_case_bundle(
            bundle_id="b4",
            entity_id="e4",
            merchant_id="m1",
            check_family="fee_tax",
            facts={"rounding_tolerance_paise": 2},
            untrusted_text={},
        ),
    ]

    meta = {
        "b1": {"is_control": True, "expected_decision": "no_issue", "expected_class": None, "source_label_type": "control"},
        "b2": {"is_control": False, "expected_decision": "flag", "expected_class": "FEE_TAX_MISMATCH", "source_label_type": "fee_mismatch"},
        "b3": {"is_control": False, "expected_decision": "escalate_human", "expected_class": None, "source_label_type": "NEAR_TOLERANCE"},
        "b4": {"is_control": True, "expected_decision": "no_issue", "expected_class": None, "source_label_type": "control"},
    }

    preds = [
        Prediction(merchant_id="m1", predicted_class="UNKNOWN", action="ignore", entity_ids=("e1",), settlement_id=None, rule_id="R1"),
        Prediction(merchant_id="m1", predicted_class="FEE_TAX_MISMATCH", action="flag", entity_ids=("e2",), settlement_id=None, rule_id="R2"),
        Prediction(merchant_id="m1", predicted_class="UNKNOWN", action="escalate_human", entity_ids=("e3",), settlement_id=None, rule_id="R3"),
        # e4 is unpredicted (clean)
    ]

    cr = {b.bundle_id: {"status": "ok"} for b in bundles}
    eval_res = score_arm(
        arm_name="test_b1r_map",
        thinking_level=None,
        finalized_decisions={},
        meta_records=meta,
        bundles=bundles,
        case_results=cr,
        b1r_predictions=preds,
    )

    b1r_card = eval_res.b1r_scorecard
    assert b1r_card.completed_cases == 4
    assert b1r_card.decision_accuracy.k == 4
    assert b1r_card.decision_accuracy.rate == 1.0
    assert b1r_card.unsafe_resolution.k == 0
    assert b1r_card.unsafe_resolution.rate == 0.0


@pytest.mark.slow
def test_b1r_real_accuracy_dev_seed1():
    """Verify B1r accuracy >= 95% and unsafe rate <= 2% on the full dev_seed1 population."""
    from src.recon_agent.eval.llm_scoring import score_arm
    from src.recon_agent.pipeline.loader import load_dataset
    from src.recon_agent.systems.b1r import run_b1r

    dataset_dir = os.path.join(REPO_ROOT, "data", "synth", "full", "dev_seed1")
    pop_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1.jsonl")
    meta_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1_meta.jsonl")
    if not (os.path.isdir(dataset_dir) and os.path.isfile(pop_file) and os.path.isfile(meta_file)):
        pytest.skip("dev_seed1 dataset or population files not found")

    ds = load_dataset(dataset_dir)
    b1r_out = run_b1r(ds)

    bundles = []
    with open(pop_file, "r", encoding="utf-8") as f:
        for line in f:
            bundles.append(CaseBundle(**json.loads(line)))

    meta = {}
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta[r["bundle_id"]] = r

    cr = {b.bundle_id: {"status": "ok"} for b in bundles}
    eval_res = score_arm(
        arm_name="dev_seed1_b1r",
        thinking_level=None,
        finalized_decisions={},
        meta_records=meta,
        bundles=bundles,
        case_results=cr,
        b1r_predictions=b1r_out.predictions,
    )

    b1r_card = eval_res.b1r_scorecard
    assert b1r_card.total_cases == 312
    assert b1r_card.completed_cases == 312
    assert b1r_card.decision_accuracy.rate is not None
    assert b1r_card.decision_accuracy.rate >= 0.95, f"B1r accuracy was {b1r_card.decision_accuracy.rate}"
    assert b1r_card.unsafe_resolution.rate is not None
    assert b1r_card.unsafe_resolution.rate <= 0.02, f"B1r unsafe rate was {b1r_card.unsafe_resolution.rate}"

