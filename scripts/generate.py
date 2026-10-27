#!/usr/bin/env python3
"""CLI script to generate deterministic synthetic reconciliation datasets."""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.synth import SyntheticDataGenerator, create_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic datasets for recon-agent."
    )
    parser.add_argument(
        "--profile",
        choices=["small", "full"],
        default="small",
        help="Merchant volume profile (small: 10 merchants, full: 60 merchants)",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split (dev or test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Integer seed for deterministic generation",
    )
    parser.add_argument(
        "--out",
        default="data/synth",
        help="Base output directory (default: data/synth)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and display counts only, write nothing to disk",
    )

    args = parser.parse_args()

    # Safety check for test split embargo
    if args.split == "test" and not args.dry_run:
        if os.getenv("FINAL_RUN") != "1":
            print(
                "[ERROR] Refusing to generate 'test' split. Embargo active unless FINAL_RUN=1 is set.",
                file=sys.stderr,
            )
            sys.exit(1)

    config = create_config(profile=args.profile, split=args.split, seed=args.seed)
    generator = SyntheticDataGenerator(config)

    target_dir = os.path.join(args.out, args.profile, args.split)

    if args.dry_run:
        print(f"[*] Executing dry-run generation (Profile: {args.profile}, Split: {args.split}, Seed: {args.seed})...")
        data = generator.generate()
        counts = data["manifest"]["counts"]
        print("[*] Dry-run complete. Counts summary:")
        print(json.dumps(counts, indent=2))
        return

    print(f"[*] Generating synthetic dataset -> {target_dir} (Seed: {args.seed})...")
    paths = generator.write_to_directory(target_dir)
    print(f"[*] Successfully generated {len(paths)} dataset files:")
    for name, path in paths.items():
        size_bytes = os.path.getsize(path)
        print(f"  - {name}: {path} ({size_bytes} bytes)")


if __name__ == "__main__":
    main()
