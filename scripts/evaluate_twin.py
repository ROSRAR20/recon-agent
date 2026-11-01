"""Twin evaluation script for baseline systems (B1 and B1r).

Runs the system on base data and on the injected twin datasets (separate and inline),
computes flips, unsafe flips, and ambiguity-tier action accuracy under attack,
saves results to results/twin_<system>_<split>_pooled.json, and prints Markdown tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.eval.scoring import load_ground_truth
from src.recon_agent.eval.twin_eval import (
    PlacementResult,
    evaluate_twin_single,
    load_injection_manifest,
    pool_placement_results,
)
from src.recon_agent.pipeline.loader import load_dataset
from src.recon_agent.systems.b1 import run_b1
from src.recon_agent.systems.b1r import run_b1r


def print_field_occupancy_and_reading_info(system_name: str) -> None:
    """Print field occupancy of attack types and which fields systems read."""
    print("=" * 78)
    print("FIELD OCCUPANCY & SYSTEM INPUT MAPPINGS")
    print("=" * 78)
    print("1. Injected Attack Fields (Separate Twin):")
    print("   - instruction_override : recon_items.description")
    print("   - fake_authority       : recon_items.notes['comment']")
    print("   - fake_evidence_ids    : recon_items.notes['comment']")
    print("   - fake_numbers         : recon_items.description")
    print("   - json_payload         : recon_items.description")
    print("   - obfuscated_variant   : recon_items.notes['comment']")
    print()
    print("2. Injected Attack Fields (Inline Twin):")
    print("   - ALL attack types     : recon_items.notes['reason_text'] (appended after original sentence)")
    print()
    print("3. System Fields Read:")
    print("   - B1 reads  : settlements.utr, bank_statement.description, orders_ledger.booked_date,")
    print("                 payments.captured_at, numeric amounts/fees/taxes/IDs.")
    print("                 -> B1 reads ZERO recon_items.notes and ZERO recon_items.description.")
    print("   - B1r reads : All fields read by B1, plus recon_items.notes['reason_text'] via regex")
    print("                 on refund items with arithmetic discrepancy.")
    print("                 -> B1r reads ZERO recon_items.notes['comment'] and ZERO recon_items.description.")
    print(f"Target system under evaluation: {system_name.upper()}")
    print("=" * 78 + "\n")


def format_markdown_table(res: PlacementResult, system_name: str, placement: str) -> str:
    """Generate Markdown tables for a placement evaluation result."""
    lines: list[str] = []
    lines.append(f"### Twin Evaluation Results: System `{system_name}` ({placement.upper()} Placement)\n")
    lines.append(f"**Total Injected Cases:** {res.total_cases} | **Total Flips:** {res.total_flips} | **Total Unsafe Flips:** {res.total_unsafe_flips}\n")

    # Table 1: By Attack Type
    lines.append("| Attack Type | Total | Flips (k/n) | Flip Rate | 95% Wilson CI | Unsafe Flips (u/n) | Ambiguity Acc Under Attack |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for atk, m in sorted(res.by_attack_type.items()):
        ci_str = f"[{m.flip_ci[0]:.4f}, {m.flip_ci[1]:.4f}]"
        amb_str = f"{m.ambiguity_correct}/{m.ambiguity_total} ({m.ambiguity_accuracy:.1%})" if m.ambiguity_total > 0 and m.ambiguity_accuracy is not None else "N/A"
        lines.append(
            f"| `{atk}` | {m.total} | {m.flips}/{m.total} | {m.flip_rate:.1%} | {ci_str} | {m.unsafe_flips}/{m.total} ({m.unsafe_flip_rate:.1%}) | {amb_str} |"
        )
    lines.append("")

    # Table 2: By Field
    lines.append(f"#### Flips by Field ({placement.upper()} Placement)\n")
    lines.append("| Field | Total Cases | Flips | Flip Rate | Unsafe Flips |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for fld, fc in sorted(res.by_field.items()):
        frate = fc["flips"] / fc["total"] if fc["total"] > 0 else 0.0
        lines.append(f"| `{fld}` | {fc['total']} | {fc['flips']} | {frate:.1%} | {fc['unsafe_flips']} |")
    lines.append("")

    # Table 3: By Underlying Label Type
    lines.append(f"#### Flips by Underlying Label Type ({placement.upper()} Placement)\n")
    lines.append("| Underlying Label | Total Cases | Flips | Flip Rate | Unsafe Flips |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for lbl, lc in sorted(res.by_label_type.items()):
        frate = lc["flips"] / lc["total"] if lc["total"] > 0 else 0.0
        lines.append(f"| `{lbl}` | {lc['total']} | {lc['flips']} | {frate:.1%} | {lc['unsafe_flips']} |")
    lines.append("")

    return "\n".join(lines)


def run_twin_eval(
    system_name: str,
    seeds: list[int],
    profile: str = "full",
    split: str = "dev",
    results_dir: str = "results",
) -> dict[str, Any]:
    """Execute twin evaluation across specified seeds and save pooled results."""
    print_field_occupancy_and_reading_info(system_name)

    runner = run_b1 if system_name.lower() == "b1" else run_b1r

    separate_single_results: list[PlacementResult] = []
    inline_single_results: list[PlacementResult] = []

    for seed in seeds:
        base_dir = os.path.join(REPO_ROOT, "data", "synth", profile, f"{split}_seed{seed}")
        sep_dir = os.path.join(base_dir, "injected")
        inline_dir = os.path.join(base_dir, "injected_inline")

        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Base dataset not found at {base_dir}")
        if not os.path.isdir(sep_dir):
            raise FileNotFoundError(f"Separate twin dataset not found at {sep_dir}")
        if not os.path.isdir(inline_dir):
            raise FileNotFoundError(f"Inline twin dataset not found at {inline_dir}")

        print(f"[*] Running System {system_name.upper()} on Seed {seed} (Base, Separate Twin, Inline Twin)...")

        # Load datasets
        base_ds = load_dataset(base_dir)
        sep_ds = load_dataset(sep_dir)
        inline_ds = load_dataset(inline_dir)

        # Ground truth labels
        labels, _ = load_ground_truth(base_dir)

        # Injection manifests
        sep_manifest = load_injection_manifest(sep_dir)
        inline_manifest = load_injection_manifest(inline_dir)

        # Run system
        base_out = runner(base_ds)
        sep_out = runner(sep_ds)
        inline_out = runner(inline_ds)

        # Evaluate
        sep_res = evaluate_twin_single(base_out, sep_out, sep_manifest, labels, placement="separate")
        inline_res = evaluate_twin_single(base_out, inline_out, inline_manifest, labels, placement="inline")

        separate_single_results.append(sep_res)
        inline_single_results.append(inline_res)

        print(f"  Seed {seed} Separate: {sep_res.total_flips}/{sep_res.total_cases} flips, {sep_res.total_unsafe_flips} unsafe")
        print(f"  Seed {seed} Inline  : {inline_res.total_flips}/{inline_res.total_cases} flips, {inline_res.total_unsafe_flips} unsafe")

    # Pool results
    pooled_separate = pool_placement_results(separate_single_results, placement="separate")
    pooled_inline = pool_placement_results(inline_single_results, placement="inline")

    # Print markdown tables
    print("\n" + "=" * 78)
    print("POOLED TWIN EVALUATION REPORT")
    print("=" * 78 + "\n")
    md_sep = format_markdown_table(pooled_separate, system_name, "separate")
    md_inline = format_markdown_table(pooled_inline, system_name, "inline")
    print(md_sep)
    print(md_inline)

    # Save output JSON
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"twin_{system_name}_{split}_pooled.json")
    pooled_data = {
        "system": system_name,
        "split": split,
        "seeds": seeds,
        "profile": profile,
        "placements": {
            "separate": asdict(pooled_separate),
            "inline": asdict(pooled_inline),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pooled_data, f, indent=2)
    print(f"[+] Saved pooled results to {out_path}")

    return pooled_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate system robustness on injected twin datasets.")
    parser.add_argument("--system", choices=["b1", "b1r"], required=True, help="System to evaluate")
    parser.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds (default: 1,2,3)")
    parser.add_argument("--profile", default="full", help="Dataset profile (default: full)")
    parser.add_argument("--split", default="dev", help="Dataset split (default: dev)")
    parser.add_argument("--results-dir", default="results", help="Directory to save JSON results")

    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_twin_eval(
        system_name=args.system,
        seeds=seeds,
        profile=args.profile,
        split=args.split,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()

