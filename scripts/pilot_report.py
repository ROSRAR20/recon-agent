"""Generate Day 7b pilot evaluation report and pre-registered arm selection.

Computes llm_scoring across evaluated arms, prints scorecards with Wilson 95% CIs,
and applies pre-registered selection logic to suggest the arm for freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_pilot import load_stratified_prefix
from src.recon_agent.eval.llm_scoring import ArmEvaluation, format_scorecard_table, score_arm
from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.finalize import (
    FinalDecision,
    compute_verifier_version,
    finalize,
    write_finalized_results,
)
from src.recon_agent.llm.gemini_client import (
    get_schema_hash,
    load_model_config,
)


def get_git_info() -> tuple[str, bool, str, bool]:
    """Retrieve git commit, dirty flag, checker hash, and whether checker matches checker-v1."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        is_dirty = bool(status)
    except Exception:
        is_dirty = False

    try:
        checker_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD:src/recon_agent/eval/independent_checker.py"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        checker_hash = "unknown"

    try:
        diff_tag = subprocess.check_output(
            ["git", "diff", "checker-v1", "--", "src/recon_agent/eval/independent_checker.py"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        checker_matches_tag = (len(diff_tag) == 0)
    except Exception:
        checker_matches_tag = True

    return commit, is_dirty, checker_hash, checker_matches_tag


def run_pilot_report(
    seed: int = 1,
    arms: list[str] | None = None,
    n_residual: int = 100,
    n_control: int = 20,
    run_id: int = 0,
    population_dir: str = "data/populations",
    results_dir: str = "results",
    finalized_dir: str = "results/finalized",
    system_prompt_path: str = "src/recon_agent/llm/prompts/system.txt",
    prices_path: str = "config/llm_prices.json",
    model_id: str | None = None,
) -> None:
    """Finalize runs, calculate scores, and print pilot evaluation report."""
    arms = arms or ["default", "low", "high"]
    resolved_pop_dir = population_dir if os.path.isabs(population_dir) else os.path.join(REPO_ROOT, population_dir)
    resolved_res_dir = results_dir if os.path.isabs(results_dir) else os.path.join(REPO_ROOT, results_dir)
    resolved_fin_dir = finalized_dir if os.path.isabs(finalized_dir) else os.path.join(REPO_ROOT, finalized_dir)

    pop_file = os.path.join(resolved_pop_dir, f"dev_seed{seed}.jsonl")
    meta_file = os.path.join(resolved_pop_dir, f"dev_seed{seed}_meta.jsonl")

    bundles, meta_lookup = load_stratified_prefix(
        population_file=pop_file,
        meta_file=meta_file,
        n_residual=n_residual,
        n_control=n_control,
    )

    prompt_full_path = system_prompt_path if os.path.isabs(system_prompt_path) else os.path.join(REPO_ROOT, system_prompt_path)
    with open(prompt_full_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    schema_hash = get_schema_hash()
    verifier_version = compute_verifier_version()

    model_cfg = load_model_config()
    resolved_model_id = model_id or model_cfg.get("primary", "gemini-3.5-flash-lite")

    cache_dir = os.path.join(resolved_res_dir, "llm_cache")
    cache = LLMCache(cache_dir=cache_dir)

    arm_thinking_map: dict[str, str | None] = {
        "default": None,
        "low": "low",
        "medium": "medium",
        "high": "high",
    }

    eval_results: dict[str, ArmEvaluation] = {}
    unsupported_arms: dict[str, str] = {}
    served_models: set[str] = set()
    total_completed = 0
    total_cases_all = 0
    total_infra_failed = 0

    for arm in arms:
        thinking_level = arm_thinking_map.get(arm, arm)
        state_path = os.path.join(resolved_res_dir, f"pilot_run_state_{arm}_seed{seed}.json")
        if not os.path.isfile(state_path):
            print(f"[REPORT] Run state for arm '{arm}' not found at {state_path}, skipping.")
            continue

        with open(state_path, "r", encoding="utf-8") as f:
            run_state = json.load(f)

        if run_state.get("status") == "unsupported" or run_state.get("unsupported"):
            unsupported_arms[arm] = run_state.get("reason", "Rejected by API (HTTP 400 parameter error)")
            continue

        cases_map = run_state.get("cases", {})
        decisions: list[FinalDecision] = []

        for bundle in bundles:
            entry = cache.get_entry(
                bundle_hash=bundle.bundle_hash,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                model_id=resolved_model_id,
                thinking_level=thinking_level,
                run_id=run_id,
            )
            raw_text = entry.get("raw_output") if entry else None
            if entry and entry.get("served_model"):
                served_models.add(entry["served_model"])

            dec = finalize(
                bundle=bundle,
                raw_text=raw_text,
                verifier_version=verifier_version,
                run_id=run_id,
            )
            decisions.append(dec)

        # Write finalized results atomically
        pop_name = f"dev_seed{seed}_{arm}"
        write_finalized_results(
            decisions=decisions,
            verifier_version=verifier_version,
            run_id=run_id,
            population=pop_name,
            base_dir=resolved_fin_dir,
        )

        fin_dict = {d.bundle_id: d.to_dict() for d in decisions}
        arm_eval = score_arm(
            arm_name=arm,
            thinking_level=thinking_level,
            finalized_decisions=fin_dict,
            meta_records=meta_lookup,
            bundles=bundles,
            case_results=cases_map,
            model_id=resolved_model_id,
            prices_path=prices_path,
        )
        eval_results[arm] = arm_eval

        total_completed += arm_eval.completed_cases
        total_cases_all += arm_eval.total_cases
        total_infra_failed += arm_eval.infra_failed_count

    # Git metadata & checker verification
    git_commit, is_dirty, checker_hash, checker_matches_tag = get_git_info()
    overall_infra_rate = (total_infra_failed / total_cases_all) if total_cases_all > 0 else 0.0

    print("=" * 80)
    print("DAY 7b-1 PILOT EVALUATION REPORT")
    print("=" * 80)
    print(f"Git Commit        : {git_commit} {'(DIRTY)' if is_dirty else '(CLEAN)'}")
    print(f"System Prompt Hash: {prompt_hash}")
    print(f"Verifier Version  : {verifier_version}")
    print(f"Checker Git Hash  : {checker_hash}")
    if not checker_matches_tag:
        print("⚠️ WARNING: independent_checker.py differs from frozen tag checker-v1!")
    else:
        print("Checker Tag Status: Matches frozen tag checker-v1 (CLEAN)")
    print(f"Requested Model   : {resolved_model_id}")
    print(f"Served Model(s)   : {', '.join(sorted(served_models)) if served_models else resolved_model_id}")
    print(f"Run ID(s)         : [{run_id}]")
    print(f"Completed Cases   : {total_completed}/{total_cases_all} across {len(eval_results)} evaluated arm(s)")
    print(f"Infra-Failure Rate: {overall_infra_rate:.1%}")
    if unsupported_arms:
        print(f"Unsupported Arms  : {unsupported_arms}")
    print("=" * 80)
    print()

    # Print Scorecards per arm
    for arm, arm_eval in eval_results.items():
        print(format_scorecard_table(arm_eval))
        print()

    # Pre-registered selection logic
    print("## Pre-Registered Selection Logic")
    print()
    print("| Arm | Thinking Level | Schema-Valid (>=95%) | B3 Unsafe Res (k/n, 95% CI) | Latency p50 | Eligibility / Status |")
    print("| :--- | :---: | :---: | :---: | :---: | :--- |")

    eligible_arms: list[tuple[str, ArmEvaluation]] = []

    for arm, arm_eval in eval_results.items():
        b3 = arm_eval.b3_scorecard
        schema_rate = b3.schema_valid.rate
        is_eligible = schema_rate >= 0.95
        elig_str = "✅ ELIGIBLE" if is_eligible else "❌ INELIGIBLE (Schema < 95%)"

        p50_str = f"{b3.latency.p50_ms:.1f} ms" if b3.latency.p50_ms is not None else "N/A"
        unsafe_str = b3.unsafe_resolution.kn_str

        print(
            f"| `{arm}` | `{arm_eval.thinking_level}` | {b3.schema_valid.kn_str} | "
            f"**{unsafe_str}** | {p50_str} | {elig_str} |"
        )
        if is_eligible:
            eligible_arms.append((arm, arm_eval))

    for arm, reason in unsupported_arms.items():
        print(f"| `{arm}` | N/A | N/A | N/A | N/A | ❌ UNSUPPORTED ({reason}) |")

    print()

    # Apply selection logic
    suggested_arm: str | None = None
    selection_reason: str = ""

    if not eligible_arms:
        suggested_arm = None
        selection_reason = "No evaluated arms met the schema-validity requirement of >= 95%."
    else:
        # Sort by:
        # 1. B3 unsafe resolution count (ascending)
        # 2. Check ties within sampling error:
        #    If top candidate and another arm have overlapping 95% CIs on unsafe rate,
        #    break tie using lower p50 latency.
        min_unsafe_count = min(a[1].b3_scorecard.unsafe_resolution.k for a in eligible_arms)
        best_unsafe_arm, best_unsafe_eval = next(a for a in eligible_arms if a[1].b3_scorecard.unsafe_resolution.k == min_unsafe_count)

        ci_low_best = best_unsafe_eval.b3_scorecard.unsafe_resolution.ci_low
        ci_high_best = best_unsafe_eval.b3_scorecard.unsafe_resolution.ci_high

        # Find all arms within sampling error (overlapping CIs)
        tied_arms: list[tuple[str, ArmEvaluation]] = []
        for a_name, a_eval in eligible_arms:
            u_ci_low = a_eval.b3_scorecard.unsafe_resolution.ci_low
            u_ci_high = a_eval.b3_scorecard.unsafe_resolution.ci_high
            # Check CI overlap
            if max(ci_low_best, u_ci_low) <= min(ci_high_best, u_ci_high):
                tied_arms.append((a_name, a_eval))

        if len(tied_arms) == 1:
            suggested_arm = best_unsafe_arm
            selection_reason = (
                f"Lowest B3 unsafe resolution count ({min_unsafe_count}/{best_unsafe_eval.completed_cases}) "
                f"with non-overlapping CI."
            )
        else:
            # Tie breaker: lower p50 latency
            tied_names = [t[0] for t in tied_arms]
            tied_arms_sorted = sorted(
                tied_arms,
                key=lambda x: (x[1].b3_scorecard.latency.p50_ms if x[1].b3_scorecard.latency.p50_ms is not None else 999999.0)
            )
            suggested_arm = tied_arms_sorted[0][0]
            lat_p50 = tied_arms_sorted[0][1].b3_scorecard.latency.p50_ms
            selection_reason = (
                f"Tied on B3 unsafe resolution rate within sampling error among {tied_names}. "
                f"Selected arm `{suggested_arm}` with lower p50 latency ({lat_p50:.1f} ms)."
            )

    print("### Suggested Arm for Freeze")
    if suggested_arm:
        print(f"> **`{suggested_arm}`** (SUGGESTION ONLY)")
        print(f"> *Reason*: {selection_reason}")
    else:
        print(f"> None: {selection_reason}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pilot report and arm selection")
    parser.add_argument("--seed", type=int, default=1, help="Seed index (default: 1)")
    parser.add_argument("--arms", default="default,low,high", help="Comma-separated arms (default: default,low,high)")
    parser.add_argument("--n-residual", type=int, default=100, help="Number of residual cases (default: 100)")
    parser.add_argument("--n-control", type=int, default=20, help="Number of control cases (default: 20)")
    parser.add_argument("--run-id", type=int, default=0, help="Run ID (default: 0)")
    parser.add_argument("--population-dir", default="data/populations", help="Population directory")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="results/finalized", help="Finalized output directory")
    parser.add_argument("--prices-path", default="config/llm_prices.json", help="Pricing config path")
    parser.add_argument("--model", default=None, help="Model ID override")
    args = parser.parse_args()

    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    run_pilot_report(
        seed=args.seed,
        arms=arm_list,
        n_residual=args.n_residual,
        n_control=args.n_control,
        run_id=args.run_id,
        population_dir=args.population_dir,
        results_dir=args.results_dir,
        finalized_dir=args.output_dir,
        prices_path=args.prices_path,
        model_id=args.model,
    )


if __name__ == "__main__":
    main()
