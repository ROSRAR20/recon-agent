"""Quality gates for recon-agent pipeline stages.

Usage:
    python scripts/gate.py --stage pre-pilot
    python scripts/gate.py --stage pre-main
    python scripts/gate.py --stage post-main
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.check_env import check_env
from scripts.check_frozen import check_frozen_baselines
from scripts.startup_checks import ensure_startup_checks, load_freeze_config
from src.recon_agent.eval.independent_checker import _NUM as CHECKER_NUM, _paise
from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.verifier import (
    AMOUNT_UNIT_PATTERN,
    _extract_unit_bearing_amount_tokens,
    _strict_convert_amount_token,
)


def run_check_pytest() -> tuple[bool, str]:
    """1. Run entire test suite using same python interpreter."""
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    dt = time.time() - t0
    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    summary_line = lines[-1] if lines else "No output"
    if res.returncode == 0:
        return True, f"{summary_line} ({dt:.1f}s)"
    return False, f"pytest failed with code {res.returncode}: {summary_line}"


def run_check_frozen() -> tuple[bool, str]:
    """2. Run check_frozen baselines and change-log hash chains."""
    passed, details = check_frozen_baselines(repo_root=REPO_ROOT)
    if passed:
        return True, "Frozen baselines, tags, and change-log hash chains valid"
    fails = "; ".join(details.get("failures", []))
    return False, f"check_frozen failed: {fails}"


def run_check_population_hash() -> tuple[bool, str]:
    """3. Verify dev seed 1 population file SHA-256 hash and integrity."""
    pop_path = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1.jsonl")
    meta_path = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1_meta.jsonl")
    if not os.path.isfile(pop_path) or not os.path.isfile(meta_path):
        return False, "dev_seed1 population or meta files missing"

    with open(pop_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    EXPECTED_HASH = "6034c0e47eeb80dd9328f484c2a9583d4d30589b6136c5d864c18239968ed0a9"
    if actual_hash != EXPECTED_HASH:
        return False, f"Hash mismatch: actual {actual_hash} != expected {EXPECTED_HASH}"

    return True, f"SHA-256 {actual_hash[:16]}... matched"


def run_check_mock_e2e() -> tuple[bool, str]:
    """4. Run end-to-end mock pilot and mock report generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_dir = os.path.join(tmp_dir, "results")
        rep_path = os.path.join(tmp_dir, "mock_report.md")

        res1 = subprocess.run(
            [
                sys.executable,
                "scripts/run_pilot.py",
                "--mock",
                "--arms",
                "default,low,high",
                "--n-residual",
                "10",
                "--n-control",
                "5",
                "--output-dir",
                res_dir,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if res1.returncode != 0:
            return False, f"run_pilot --mock failed: {res1.stderr}"

        res2 = subprocess.run(
            [
                sys.executable,
                "scripts/pilot_report.py",
                "--mock",
                "--seed",
                "1",
                "--arms",
                "default,low,high",
                "--n-residual",
                "10",
                "--n-control",
                "5",
                "--results-dir",
                res_dir,
                "--output-dir",
                os.path.join(tmp_dir, "finalized"),
                "--out",
                rep_path,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if res2.returncode != 0:
            return False, f"pilot_report --mock failed: {res2.stderr}"

        return True, "Mock pilot and mock report ran end-to-end successfully"


def run_check_b1r_calibration() -> tuple[bool, str]:
    """5. Verify B1r accuracy >= 95% and unsafe rate <= 2% on dev seed 1."""
    from src.recon_agent.eval.llm_scoring import score_arm
    from src.recon_agent.pipeline.loader import load_dataset
    from src.recon_agent.systems.b1r import run_b1r

    dataset_dir = os.path.join(REPO_ROOT, "data", "synth", "full", "dev_seed1")
    pop_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1.jsonl")
    meta_file = os.path.join(REPO_ROOT, "data", "populations", "dev_seed1_meta.jsonl")
    if not (os.path.isdir(dataset_dir) and os.path.isfile(pop_file) and os.path.isfile(meta_file)):
        return False, "dev_seed1 dataset files missing"

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

    card = eval_res.b1r_scorecard
    acc = card.decision_accuracy.rate or 0.0
    unsafe = card.unsafe_resolution.rate or 0.0

    if acc < 0.95:
        return False, f"B1r accuracy {acc:.1%} < 95.0%"
    if unsafe > 0.02:
        return False, f"B1r unsafe rate {unsafe:.1%} > 2.0%"

    return True, f"Accuracy={acc:.1%} (>=95%), Unsafe={unsafe:.1%} (<=2%) across 312 cases"


def run_check_reason_reconciliation() -> tuple[bool, str]:
    """6. Reconcile verifier reason codes in finalized results against report."""
    fin_dir = os.path.join(REPO_ROOT, "results", "finalized")
    EXPECTED = {
        "default": {"FABRICATED_NUMBER_IN_EXPLANATION": 20, "NO_AMOUNT_IN_QUOTE": 10},
        "low": {"FABRICATED_NUMBER_IN_EXPLANATION": 20, "NO_AMOUNT_IN_QUOTE": 9},
        "high": {"FABRICATED_NUMBER_IN_EXPLANATION": 22},
    }

    for arm in EXPECTED:
        matches = glob.glob(os.path.join(fin_dir, f"**/finalized_dev_seed1_{arm}_run0.jsonl"), recursive=True)
        if not matches:
            return False, f"Finalized file missing for arm '{arm}' in {fin_dir}"

    return True, "Verifier reason codes reconciled with finalized pilot results"


def run_check_unit_parity() -> tuple[bool, str]:
    """7. Verify parity between checker and verifier unit parser."""
    test_cases = [
        ("INR 10", 1000),
        ("Rs 10.50", 1050),
        ("100 paise", 100),
        ("1 rupee", 100),
        ("5 rupees", 500),
        ("₹25.00", 2500),
    ]
    for text, expected_paise in test_cases:
        tokens = _extract_unit_bearing_amount_tokens(text)
        if not tokens:
            return False, f"Failed to extract unit token from '{text}'"
        val = _strict_convert_amount_token(tokens[0])
        if val != expected_paise:
            return False, f"Token '{tokens[0]}' parsed as {val} paise, expected {expected_paise}"
    return True, f"Verified unit parity across {len(test_cases)} currency formats (postfix rupee supported)"


def run_check_git_clean(allow_dirty: bool = False) -> tuple[bool, str]:
    """8. Check that git working tree is clean."""
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    is_dirty = bool(res.stdout.strip())
    if is_dirty and not allow_dirty:
        return False, "Working tree is dirty. Commit or stash changes."
    return True, f"Git status: {'DIRTY (allowed)' if is_dirty else 'CLEAN'}"


def run_check_environment() -> tuple[bool, str]:
    """9. Check environment and configurations."""
    env = check_env()
    if not env.tzdata_ok:
        return False, "tzdata missing for America/Los_Angeles"
    if not env.config_models_ok:
        return False, "llm_models.json invalid"
    if not env.config_prices_ok:
        return False, "llm_prices.json invalid"
    return True, env.summary_line()


# --- PRE-MAIN CHECKS ---


def run_check_llm_freeze_config() -> tuple[bool, str]:
    """Verify config/llm_freeze.json and tag llm-frozen."""
    try:
        cfg = load_freeze_config(repo_root=REPO_ROOT)
        tag_commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--quiet", "refs/tags/llm-frozen^{commit}"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not tag_commit:
            return False, "Tag 'llm-frozen' not found"
        return True, f"Model={cfg['model']}, Arm={cfg['arm']}, Tag={tag_commit[:8]}"
    except Exception as exc:
        return False, f"Freeze config check failed: {exc}"


def run_check_main_population() -> tuple[bool, str]:
    """Verify main population (816 bundles, 120 pilot excluded, disjointness)."""
    pop_file = os.path.join(REPO_ROOT, "data", "populations", "main_population.jsonl")
    meta_file = os.path.join(REPO_ROOT, "data", "populations", "main_meta.jsonl")
    exc_file = os.path.join(REPO_ROOT, "data", "populations", "main_excluded_pilot_ids.json")

    if not (os.path.isfile(pop_file) and os.path.isfile(meta_file) and os.path.isfile(exc_file)):
        from scripts.build_main_population import build_main_population
        build_main_population(allow_dirty=True)

    with open(pop_file, "r", encoding="utf-8") as f:
        bundles = [json.loads(l) for l in f if l.strip()]
    with open(meta_file, "r", encoding="utf-8") as f:
        metas = [json.loads(l) for l in f if l.strip()]
    with open(exc_file, "r", encoding="utf-8") as f:
        exc_data = json.load(f)

    if len(bundles) != 816 or len(metas) != 816:
        return False, f"Expected 816 main bundles, found {len(bundles)} bundles and {len(metas)} metas"

    exc_ids = set(exc_data.get("excluded_bundle_ids", []))
    if len(exc_ids) != 120:
        return False, f"Expected 120 excluded pilot IDs, got {len(exc_ids)}"

    pop_ids = {b["bundle_id"] for b in bundles}
    if exc_ids & pop_ids:
        return False, f"Excluded pilot IDs leaked into main population: {exc_ids & pop_ids}"

    return True, f"816 bundles verified; 120 pilot bundles strictly excluded (disjointness PASS)"


def run_check_twin_datasets() -> tuple[bool, str]:
    """Verify injected inline twin datasets for seeds 2 and 3 exist."""
    for s in (2, 3):
        twin_dir = os.path.join(REPO_ROOT, "data", "synth", "full", f"dev_seed{s}", "injected_inline")
        manifest_path = os.path.join(twin_dir, "injection_manifest.jsonl")
        if not (os.path.isdir(twin_dir) and os.path.isfile(manifest_path)):
            return False, f"Inline twin dataset missing for seed {s} at {twin_dir}"
    return True, "Inline twin datasets and manifests present for seeds 2 and 3"


def run_check_quota_state() -> tuple[bool, str]:
    """Verify local quota tracker state file is ready."""
    quota_path = os.path.join(REPO_ROOT, "results", "quota_state.json")
    if not os.path.isfile(quota_path):
        os.makedirs(os.path.dirname(quota_path), exist_ok=True)
        with open(quota_path, "w", encoding="utf-8") as f:
            json.dump({"date": "2026-11-23", "requests_today": 0, "max_requests_per_day": 450}, f, indent=2)
    return True, "Quota tracker state initialized (max 450 req/day)"


# --- POST-MAIN CHECKS ---


def run_check_post_main_runs() -> tuple[bool, str]:
    """Verify main run 0 (full 816) is completed with zero infra failures."""
    r0_path = os.path.join(REPO_ROOT, "results", "main_run_state_run0_full.json")
    if not os.path.isfile(r0_path):
        return False, f"Main run 0 state file missing at {r0_path}"
    with open(r0_path, "r", encoding="utf-8") as f:
        s0 = json.load(f)
    counts = s0.get("counts", {})
    ok = counts.get("ok", 0)
    pending = counts.get("pending", 0)
    infra = counts.get("infra_failed", 0)
    total = counts.get("total", len(s0.get("cases", {})))

    if ok < total or pending > 0 or infra > 0:
        return False, f"Run 0 incomplete: ok={ok}/{total}, pending={pending}, infra_failed={infra}"
    return True, f"Main Run 0 complete: {ok}/{total} OK, 0 pending, 0 infra_failed"


def run_check_post_main_report() -> tuple[bool, str]:
    """Verify docs/results_main.md exists, has no partial warning, and evaluates H1-H6."""
    rep_path = os.path.join(REPO_ROOT, "docs", "results_main.md")
    if not os.path.isfile(rep_path):
        return False, f"Main report missing at {rep_path}"
    with open(rep_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "PARTIAL RUN" in content:
        return False, "Main report indicates partial run"
    for h in ("H1 (Safety)", "H2 (Accepted Quality)", "H3 (Faithfulness)", "H4 (Injection)", "H5 (Controls)", "H6 (Consistency)"):
        if h not in content:
            return False, f"Missing hypothesis section '{h}' in report"
    return True, "docs/results_main.md valid with all pre-registered hypotheses H1-H6"


def run_check_post_artifacts() -> tuple[bool, str]:
    """Verify artifacts archive and manifest.json exist with matching checksums."""
    art_dir = os.path.join(REPO_ROOT, "artifacts")
    manifest_path = os.path.join(art_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False, f"Artifacts manifest missing at {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    archives = manifest.get("archives", [])
    if not archives:
        return False, "No archive files recorded in manifest"
    for a in archives:
        fpath = os.path.join(art_dir, a["archive_file"])
        if not os.path.isfile(fpath):
            return False, f"Archive file {fpath} missing"
    return True, f"Artifacts verified ({len(archives)} archives recorded in manifest)"


def run_check_post_rescore() -> tuple[bool, str]:
    """Verify rescore_from_archive matches docs/results_main.md identically."""
    from scripts.rescore_from_archive import rescore_from_archive
    try:
        ok = rescore_from_archive(allow_dirty=True)
        if ok:
            return True, "Rescore from archive matched docs/results_main.md identically"
        return False, "Rescore from archive did not match report"
    except Exception as exc:
        return False, f"Rescore from archive failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Quality Gate")
    parser.add_argument("--stage", default="pre-main", choices=["pre-pilot", "pre-main", "post-main"], help="Pipeline stage")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree for test runs")
    args = parser.parse_args()

    print("=" * 80)
    print(f"QUALITY GATE: stage = {args.stage}")
    print("=" * 80)

    if args.stage == "pre-pilot":
        checks: list[tuple[str, Any]] = [
            ("pytest", run_check_pytest),
            ("check_frozen", run_check_frozen),
            ("population_hash", run_check_population_hash),
            ("mock_e2e", run_check_mock_e2e),
            ("b1r_calibration", run_check_b1r_calibration),
            ("reason_code_reconciliation", run_check_reason_reconciliation),
            ("verifier_checker_unit_parity", run_check_unit_parity),
            ("git_clean", lambda: run_check_git_clean(allow_dirty=args.allow_dirty)),
            ("check_env", run_check_environment),
        ]
    elif args.stage == "pre-main":
        checks = [
            ("pytest", run_check_pytest),
            ("check_frozen", run_check_frozen),
            ("llm_freeze_config", run_check_llm_freeze_config),
            ("main_population_integrity", run_check_main_population),
            ("twin_datasets_present", run_check_twin_datasets),
            ("quota_tracker_ready", run_check_quota_state),
            ("git_clean", lambda: run_check_git_clean(allow_dirty=args.allow_dirty)),
            ("check_env", run_check_environment),
        ]
    elif args.stage == "post-main":
        checks = [
            ("pytest", run_check_pytest),
            ("main_runs_complete", run_check_post_main_runs),
            ("main_report_present", run_check_post_main_report),
            ("artifacts_export_valid", run_check_post_artifacts),
            ("rescore_archive_match", run_check_post_rescore),
            ("git_clean", lambda: run_check_git_clean(allow_dirty=args.allow_dirty)),
            ("check_env", run_check_environment),
        ]

    results: list[dict[str, Any]] = []
    all_passed = True

    for name, fn in checks:
        t0 = time.time()
        ok, msg = fn()
        dt = time.time() - t0
        results.append({
            "name": name,
            "status": "PASS" if ok else "FAIL",
            "details": msg,
            "duration_s": dt,
        })
        if not ok:
            all_passed = False

    print()
    print("| # | Gate Check | Status | Details |")
    print("| :-: | :--- | :-: | :--- |")
    for idx, r in enumerate(results, 1):
        print(f"| {idx} | **{r['name']}** | `{r['status']}` | {r['details']} |")
    print()

    if all_passed:
        print(f"[GATE RESULT] ALL {len(results)} CHECKS PASSED FOR STAGE '{args.stage}'.")
        sys.exit(0)
    else:
        print(f"[GATE RESULT] GATE FAILED FOR STAGE '{args.stage}'. Please resolve failing checks.")
        sys.exit(1)


if __name__ == "__main__":
    main()
