"""Tests for Day 8 main-run tooling: population building, main & twin runners, report, artifacts, and rescoring.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.build_main_population import build_main_population, get_pilot_excluded_ids
from scripts.export_artifacts import compute_file_sha256, export_artifacts
from scripts.main_report import generate_main_report
from scripts.rescore_from_archive import extract_numbers_from_report, rescore_from_archive
from scripts.run_main import get_consistency_subset_bundle_ids, run_main
from scripts.run_twin import compute_flips, run_twin
from scripts.startup_checks import (
    ensure_startup_checks,
    load_freeze_config,
    verify_freeze_hashes,
)


def test_startup_checks():
    """Verify startup checks succeed with allow_dirty=True and freeze config is valid."""
    cfg = load_freeze_config(repo_root=REPO_ROOT)
    assert cfg["model"] == "gemini-3.5-flash-lite"
    assert cfg["arm"] == "default"
    assert "prompt_hash" in cfg
    assert "verifier_version" in cfg

    verify_freeze_hashes(cfg, repo_root=REPO_ROOT)
    info = ensure_startup_checks(repo_root=REPO_ROOT, allow_dirty=True, verify_hashes=True)
    assert info["llm_frozen_commit"] is not None


def test_build_main_population_and_disjointness():
    """Verify main population builder creates 816 bundles and strictly excludes 120 pilot bundles."""
    with tempfile.TemporaryDirectory() as tmp_pop_dir:
        summary = build_main_population(pop_dir=tmp_pop_dir, allow_dirty=True)
        assert summary["total_bundles"] == 816
        assert summary["residual_total"] == 656
        assert summary["control_total"] == 160
        assert summary["excluded_pilot_count"] == 120

        pop_file = summary["output_population_path"]
        meta_file = summary["output_metadata_path"]
        exc_file = summary["excluded_ids_path"]

        with open(pop_file, "r", encoding="utf-8") as f:
            bundles = [json.loads(line) for line in f]
        with open(meta_file, "r", encoding="utf-8") as f:
            metas = [json.loads(line) for line in f]
        with open(exc_file, "r", encoding="utf-8") as f:
            exc_data = json.load(f)

        assert len(bundles) == 816
        assert len(metas) == 816

        pop_ids = {b["bundle_id"] for b in bundles}
        exc_ids = set(exc_data["excluded_bundle_ids"])
        assert len(exc_ids) == 120
        # Disjointness assert
        assert not (pop_ids & exc_ids)

        # Control text assert
        for b, m in zip(bundles, metas):
            if m["is_control"]:
                untrusted = b.get("untrusted_text") or {}
                assert untrusted.get("recon_item_notes_reason_text") is None


def test_consistency_subset_determinism():
    """Verify consistency subset returns exactly 200 deterministic bundles."""
    subset_1 = get_consistency_subset_bundle_ids(pop_dir="data/populations", target_count=200, seed=42)
    subset_2 = get_consistency_subset_bundle_ids(pop_dir="data/populations", target_count=200, seed=42)
    assert len(subset_1) == 200
    assert subset_1 == subset_2


def test_run_main_mock_and_resume():
    """Verify run_main in mock mode executes cleanly and supports resumability."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_dir = os.path.join(tmp_dir, "results")

        # Run consistency scope (200 bundles) with quota 50
        summary1 = run_main(
            run_id=1,
            scope="consistency",
            resume=False,
            max_requests_per_day=50,
            mock=True,
            allow_dirty=True,
            pop_dir="data/populations",
            output_dir=res_dir,
        )
        assert summary1["counts"]["ok"] == 50
        assert summary1["counts"]["pending"] == 150
        assert summary1["status"] == "partial"

        # Resume with increased quota 250
        summary2 = run_main(
            run_id=1,
            scope="consistency",
            resume=True,
            max_requests_per_day=250,
            mock=True,
            allow_dirty=True,
            pop_dir="data/populations",
            output_dir=res_dir,
        )
        assert summary2["counts"]["ok"] == 200
        assert summary2["counts"]["pending"] == 0
        assert summary2["status"] == "completed"


def test_run_twin_mock_and_flips():
    """Verify run_twin in mock mode runs across seeds 2 and 3 and calculates flip metrics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_dir = os.path.join(tmp_dir, "results")
        os.makedirs(res_dir, exist_ok=True)

        # Create mock base run 0 state with 10 cases to compute flips
        base_cases = {
            f"bundle_rfnd_case_{i}": {
                "bundle_id": f"bundle_rfnd_case_{i}",
                "status": "ok",
                "raw_output": '{"decision": "escalate_human", "predicted_class": null, "reason_code": "DISCREPANCY", "explanation": "Escalated", "evidence": [], "cited_amount_paise": null, "verbatim_quote": null}',
            }
            for i in range(10)
        }
        with open(os.path.join(res_dir, "main_run_state_run0_full.json"), "w", encoding="utf-8") as f:
            json.dump({"cases": base_cases, "run_id": 0, "scope": "full"}, f)

        summary = run_twin(
            seeds=(2,),
            run_id=0,
            resume=False,
            max_requests_per_day=450,
            mock=True,
            allow_dirty=True,
            output_dir=res_dir,
        )
        assert 2 in summary["seed_summaries"]
        assert summary["seed_summaries"][2]["ok"] == 180


def test_export_artifacts_and_rescore():
    """Verify artifact export compresses files, generates valid manifest, and rescore_from_archive matches report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_dir = os.path.join(tmp_dir, "results")
        art_dir = os.path.join(tmp_dir, "artifacts")
        rep_path = os.path.join(tmp_dir, "results_main.md")
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(art_dir, exist_ok=True)

        # Create dummy completed run 0 state file
        mock_cases = {
            f"bundle_dummy_{i}": {
                "bundle_id": f"bundle_dummy_{i}",
                "status": "ok",
                "raw_output": '{"decision": "no_issue", "predicted_class": null, "reason_code": "UNAMBIGUOUS_MATCH", "explanation": "Matched perfectly.", "evidence": [], "cited_amount_paise": null, "verbatim_quote": null}',
            }
            for i in range(5)
        }
        with open(os.path.join(res_dir, "main_run_state_run0_full.json"), "w", encoding="utf-8") as f:
            json.dump({"cases": mock_cases, "run_id": 0, "scope": "full", "is_mock": True}, f)

        # Export artifacts
        manifest = export_artifacts(
            run_ids=(0,),
            results_dir=res_dir,
            artifacts_dir=art_dir,
            allow_dirty=True,
        )
        assert len(manifest["archives"]) == 1
        arch = manifest["archives"][0]
        assert arch["archive_file"] == "main_run0_full.jsonl.gz"
        assert arch["case_count"] == 5

        # Verify gz content
        gz_path = os.path.join(art_dir, arch["archive_file"])
        with gzip.open(gz_path, "rt", encoding="utf-8") as gz:
            lines = gz.readlines()
        assert len(lines) == 5

        # Generate main report
        rep_text = generate_main_report(
            run_ids=(0,),
            out_path=rep_path,
            pop_dir="data/populations",
            results_dir=res_dir,
            mock=True,
        )
        assert "DAY 8 MAIN EVALUATION REPORT" in rep_text
        assert "## Pre-Registered Hypotheses" in rep_text

        # Rescore from archive and assert numerical match
        rescore_ok = rescore_from_archive(
            artifacts_dir=art_dir,
            report_path=rep_path,
            pop_dir="data/populations",
            mock=True,
            allow_dirty=True,
        )
        assert rescore_ok is True
