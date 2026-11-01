#!/usr/bin/env python3
"""CLI script to evaluate reconciliation systems on benchmark datasets."""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.eval.scoring import (
    compute_pooled_results,
    load_ground_truth,
    score_predictions,
)
from src.recon_agent.pipeline.loader import load_dataset
from src.recon_agent.synth.config import create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator
from src.recon_agent.systems.b0 import run_b0
from src.recon_agent.systems.b1 import run_b1
from src.recon_agent.systems.b1r import run_b1r

SYSTEMS = {
    "b0": run_b0,
    "b1": run_b1,
    "b1r": run_b1r,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and score reconciliation systems on synthetic datasets."
    )
    parser.add_argument(
        "--system",
        choices=list(SYSTEMS.keys()),
        default="b0",
        help="System identifier to evaluate (default: b0)",
    )
    parser.add_argument(
        "--profile",
        choices=["small", "full"],
        default="full",
        help="Volume profile (small: 10 merchants, full: 60 merchants; default: full)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split (dev only; test split is embargoed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used for dataset generation (default: 1)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated random seeds (e.g. 1,2,3) for multi-seed pooled evaluation",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to pre-generated dataset directory (optional)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save output evaluation JSON (default: results)",
    )

    args = parser.parse_args()

    # Safety check: strictly refuse test split before Day 12
    if args.split == "test":
        print(
            "[ERROR] Evaluation on the held-out 'test' split is strictly prohibited before Day 12 (embargo active). Execution refused.",
            file=sys.stderr,
        )
        sys.exit(1)

    system_fn = SYSTEMS[args.system]

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    results_dir = os.path.abspath(os.path.join(REPO_ROOT, args.results_dir))
    os.makedirs(results_dir, exist_ok=True)

    results_list = []

    for seed in seeds:
        print(f"\n{'='*70}\n[>] Evaluating System '{args.system}' on Seed {seed}\n{'='*70}")

        # Resolve dataset directory for this seed
        if args.data_dir:
            if len(seeds) == 1:
                dataset_dir = os.path.abspath(args.data_dir)
            else:
                dataset_dir = os.path.abspath(os.path.join(args.data_dir, f"seed_{seed}"))
        else:
            if len(seeds) == 1 and not args.seeds:
                dataset_dir = os.path.abspath(os.path.join(REPO_ROOT, "data", "synth", args.profile, args.split))
            else:
                dataset_dir = os.path.abspath(os.path.join(REPO_ROOT, "data", "synth", args.profile, f"{args.split}_seed{seed}"))

        # If dataset does not exist on disk or lacks eval files, generate it deterministically
        has_dataset = (
            os.path.isdir(dataset_dir)
            and os.path.isfile(os.path.join(dataset_dir, "manifest.json"))
            and os.path.isfile(os.path.join(dataset_dir, "labels.jsonl"))
            and os.path.isfile(os.path.join(dataset_dir, "truth_links.jsonl"))
        )
        if not has_dataset:
            print(f"[*] Dataset not found or incomplete at {dataset_dir}. Generating (profile={args.profile}, split={args.split}, seed={seed})...")
            config = create_config(
                profile=args.profile,
                split=args.split,
                seed=seed,
                inject_errors=True,
            )
            generator = SyntheticDataGenerator(config)
            generator.write_to_directory(dataset_dir)
            print(f"[+] Dataset written to {dataset_dir}")

        # 1. Load observable files via pipeline loader
        print(f"[*] Loading observable dataset from {dataset_dir}...")
        dataset = load_dataset(dataset_dir)

        # 2. Execute reconciliation system
        print(f"[*] Executing system '{args.system}'...")
        system_output = system_fn(dataset)

        # 3. Load ground truth (eval-only)
        print("[*] Loading ground-truth labels and truth links for evaluation...")
        labels, truth_links = load_ground_truth(dataset_dir)

        # 4. Score predictions
        print("[*] Scoring predictions against ground truth...")
        result = score_predictions(
            system_output=system_output,
            labels=labels,
            truth_links=truth_links,
            system_id=args.system,
            split=args.split,
            seed=seed,
        )

        # 5. Save single-seed JSON results
        out_json_path = os.path.join(results_dir, f"{args.system}_{args.split}_{seed}.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[+] Seed {seed} results saved to {out_json_path}")

        # Save residuals if generated
        if system_output.residuals:
            residuals_path = os.path.join(results_dir, f"{args.system}_{args.split}_{seed}_residuals.json")
            with open(residuals_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in system_output.residuals], f, indent=2)
            print(f"[+] Seed {seed} residuals saved to {residuals_path} ({len(system_output.residuals)} cases)")

        results_list.append(result)

    # 6. Report outputs
    if len(seeds) == 1:
        print("\n" + results_list[0].summary_markdown + "\n")
    else:
        pooled = compute_pooled_results(results_list)
        pooled_json_path = os.path.join(results_dir, f"{args.system}_{args.split}_pooled.json")
        with open(pooled_json_path, "w", encoding="utf-8") as f:
            json.dump(pooled, f, indent=2)
        print(f"[+] Multi-seed pooled results saved to {pooled_json_path}")
        print("\n" + pooled["summary_markdown"] + "\n")


if __name__ == "__main__":
    main()
