"""Pre-pilot quality gate verifying system integrity, test suite, and baselines.

Usage:
    python scripts/gate.py --stage pre-pilot
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from scripts.check_env import check_env
from scripts.check_frozen import check_frozen_baselines
from src.recon_agent.eval.independent_checker import _NUM as CHECKER_NUM, _paise
from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.schema import LLMOutput
from src.recon_agent.llm.verifier import (
    AMOUNT_UNIT_PATTERN,
    Verifier,
    VerifierOutcome,
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

    # Verify disjointness and control text exclusion
    meta_lookup = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta_lookup[r["bundle_id"]] = r

    res_ids, ctrl_ids = set(), set()
    with open(pop_path, "r", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            bid = b["bundle_id"]
            if meta_lookup[bid]["is_control"]:
                ctrl_ids.add(bid)
                if b.get("untrusted_text", {}).get("recon_item_notes_reason_text") is not None:
                    return False, f"Control bundle {bid} contains reason_text"
            else:
                res_ids.add(bid)

    if res_ids & ctrl_ids:
        return False, f"Overlap between residual and control sets: {len(res_ids & ctrl_ids)} bundles"

    return True, f"SHA-256 {actual_hash[:16]}... matched; {len(res_ids)} res / {len(ctrl_ids)} ctrl disjoint"


def run_check_mock_e2e() -> tuple[bool, str]:
    """4. Run end-to-end mock pilot and mock report generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res_dir = os.path.join(tmp_dir, "results")
        rep_path = os.path.join(tmp_dir, "mock_report.md")

        # run_pilot --mock
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

        # pilot_report --mock --out
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

        if not os.path.isfile(rep_path):
            return False, "mock_report.md was not generated"

        with open(rep_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "MOCK DATA - NOT RESULTS" not in content:
            return False, "mock banner 'MOCK DATA - NOT RESULTS' missing in report"

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
    from collections import Counter
    fin_dir = os.path.join(REPO_ROOT, "results", "finalized")

    EXPECTED = {
        "default": {
            "FABRICATED_NUMBER_IN_EXPLANATION": 20,
            "NO_AMOUNT_IN_QUOTE": 10,
            "REFLECTED_UNTRUSTED_TEXT": 1,
            "UNGROUNDED_AMOUNT": 1,
            "FABRICATED_QUOTE": 1,
        },
        "low": {
            "FABRICATED_NUMBER_IN_EXPLANATION": 20,
            "NO_AMOUNT_IN_QUOTE": 9,
            "REFLECTED_UNTRUSTED_TEXT": 7,
            None: 1,
            "AMOUNT_MISMATCH_WITH_QUOTE": 1,
            "FABRICATED_QUOTE": 1,
        },
        "high": {
            "FABRICATED_NUMBER_IN_EXPLANATION": 22,
            "REFLECTED_UNTRUSTED_TEXT": 4,
        },
    }

    import glob
    for arm, exp_counts in EXPECTED.items():
        matches = glob.glob(os.path.join(fin_dir, f"**/finalized_dev_seed1_{arm}_run0.jsonl"), recursive=True)
        if not matches:
            return False, f"Finalized file missing for arm '{arm}' in {fin_dir}"
        fin_path = matches[0]

        counts: Counter[str | None] = Counter()
        with open(fin_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("verifier_outcome") != "ACCEPT":
                    counts[rec.get("verifier_reason_code")] += 1

        if dict(counts) != exp_counts:
            return False, f"Arm '{arm}' reason codes mismatch: actual={dict(counts)} vs expected={exp_counts}"

    return True, "All non-ACCEPT reason codes across default, low, high arms reconciled exactly"


def run_check_unit_parity() -> tuple[bool, str]:
    """7. Check verifier and independent checker unit token parity & red team."""
    import itertools
    prefixes = ["INR", "Rs.", "Rs", "₹"]
    postfixes = ["rupee", "rupees", "paise", "paisa"]
    numbers = ["1", "5", "10", "250", "1,234", "5.50"]

    # Prefix forms
    for pre, num in itertools.product(prefixes, numbers):
        form = f"{pre} {num}"
        m = CHECKER_NUM.search(form)
        chk_unit, chk_p = _paise(m)
        v_tokens = _extract_unit_bearing_amount_tokens(form)
        if not v_tokens:
            return False, f"Verifier failed to extract token for {form}"
        v_p, _ = _strict_convert_amount_token(v_tokens[0])
        if v_p is None or Decimal(v_p) != chk_p:
            return False, f"Unit parity mismatch for {form}: verifier={v_p} vs checker={chk_p}"

    # Postfix forms
    for num, post in itertools.product(numbers, postfixes):
        if "." in num and post in ("paise", "paisa"):
            continue
        form = f"{num} {post}"
        m = CHECKER_NUM.search(form)
        chk_unit, chk_p = _paise(m)
        v_tokens = _extract_unit_bearing_amount_tokens(form)
        if not v_tokens:
            return False, f"Verifier failed to extract token for {form}"
        v_p, _ = _strict_convert_amount_token(v_tokens[0])
        if v_p is None or Decimal(v_p) != chk_p:
            return False, f"Unit parity mismatch for {form}: verifier={v_p} vs checker={chk_p}"

    # Red team test: "1 rupee" with diff 346 paise -> REJECT
    bundle = CaseBundle(
        bundle_id="b_rt",
        check_family="refund",
        entity_id="e_rt",
        merchant_id="m_rt",
        facts={"amount_paise": 10000, "debit_paise": 9654, "ledger_refund_paise": 10000, "refund_difference_paise": 346, "refund_tolerance_paise": 0},
        untrusted_text={"recon_item_notes_reason_text": "Note without amount."},
    )
    out = LLMOutput(
        decision="escalate_human",
        predicted_class=None,
        reason_code="UNVERIFIABLE_CLAIM",
        explanation="Customer asked for fee waiver of 1 rupee.",
        evidence=[],
        cited_amount_paise=None,
        verbatim_quote=None,
    )
    res = Verifier().verify(bundle, out)
    if res.outcome != VerifierOutcome.REJECT or "V6" not in res.failed_checks:
        return False, f"Red-team '1 rupee' test failed to reject in V6: outcome={res.outcome}"

    return True, "Generated unit forms match checker exactly; red-team 1-rupee test rejected"


def run_check_git_clean() -> tuple[bool, str]:
    """8. Verify git working directory is clean."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as exc:
        return False, f"Failed to check git status: {exc}"

    if not out:
        return True, "Working tree clean (0 uncommitted files)"
    lines = [l for l in out.splitlines() if l.strip()]
    return False, f"Working tree dirty ({len(lines)} uncommitted file/changes: {', '.join([l.split()[-1] for l in lines[:5]])})"


def run_check_environment() -> tuple[bool, str]:
    """9. Run environment check (Python, tzdata, configs)."""
    try:
        env = check_env()
    except Exception as exc:
        return False, f"check_env failed: {exc}"

    if not env.tzdata_ok:
        return False, "tzdata missing for America/Los_Angeles"
    if not env.config_models_ok:
        return False, "models.json invalid"
    if not env.config_prices_ok:
        return False, "prices.json invalid"

    return True, env.summary_line()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-pilot Quality Gate")
    parser.add_argument("--stage", default="pre-pilot", help="Pipeline stage (default: pre-pilot)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"QUALITY GATE: stage = {args.stage}")
    print("=" * 80)

    checks: list[tuple[str, Any]] = [
        ("pytest", run_check_pytest),
        ("check_frozen", run_check_frozen),
        ("population_hash", run_check_population_hash),
        ("mock_e2e", run_check_mock_e2e),
        ("b1r_calibration", run_check_b1r_calibration),
        ("reason_code_reconciliation", run_check_reason_reconciliation),
        ("verifier_checker_unit_parity", run_check_unit_parity),
        ("git_clean", run_check_git_clean),
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
        print("[GATE RESULT] ALL CHECKS PASSED.")
        print()
        print("NEXT COMMANDS:")
        print("  1. Run real pilot execution (dev_seed1, runs=0, arms=default,low,high):")
        print("     python scripts/run_pilot.py --seed 1 --arms default,low,high --n-residual 100 --n-control 20")
        print("  2. Finalize runs:")
        print("     python scripts/finalize_runs.py --seed 1 --arms default,low,high")
        print("  3. Generate pilot report:")
        print("     python scripts/pilot_report.py --seed 1 --arms default,low,high --n-residual 100 --n-control 20 --out docs/pilot_run2_report.md")
        sys.exit(0)
    else:
        print("[GATE RESULT] GATE FAILED. Please resolve the failing checks above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
