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

REQUIRED_ERROR_CLASSES = [
    "DUPLICATE_PAYMENT",
    "MISSING_SETTLEMENT",
    "FEE_TAX_MISMATCH",
    "PARTIAL_REFUND_MISMATCH",
    "CUTOFF_DRIFT",
    "SHORT_BANK_CREDIT",
    "ORPHAN_BANK_CREDIT",
]

REQUIRED_HARD_NEGATIVES = [
    "ROUNDING",
    "LEGIT_ADJUSTMENT",
    "LATE_SETTLEMENT",
    "RETRY_AFTER_FAIL",
    "MULTI_MERCHANT_SAME_UTR",
    "NOISY_BANK_NARRATION",
    "UNRELATED_BANK_CREDIT",
]


def verify_floors(error_counts: dict[str, int], hard_neg_counts: dict[str, int]) -> None:
    """Verify that in full profile, all classes meet minimum threshold floors."""
    errors = []
    for cls_name in REQUIRED_ERROR_CLASSES:
        cnt = error_counts.get(cls_name, 0)
        if cnt < 50:
            errors.append(f"Error class {cls_name} count {cnt} < 50 floor")

    for hn_name in REQUIRED_HARD_NEGATIVES:
        cnt = hard_neg_counts.get(hn_name, 0)
        if cnt < 30:
            errors.append(f"Hard negative {hn_name} count {cnt} < 30 floor")

    if errors:
        for err in errors:
            print(f"[ERROR] Floor violation: {err}", file=sys.stderr)
        sys.exit(1)
    print("[OK] All error classes (>= 50) and hard-negative types (>= 30) meet required volume floors.")


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
        default=None,
        help="Integer seed for deterministic generation (default: 1 for dev, 9999 for test)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Generate clean data without error injection (default: False)",
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

    inject_errors = not args.clean
    config = create_config(
        profile=args.profile,
        split=args.split,
        seed=args.seed,
        inject_errors=inject_errors,
    )
    generator = SyntheticDataGenerator(config)

    target_dir = os.path.join(args.out, args.profile, args.split)

    if args.dry_run:
        print(
            f"[*] Executing dry-run generation (Profile: {args.profile}, Split: {args.split}, "
            f"Seed: {config.seed}, Inject: {inject_errors})..."
        )
        data = generator.generate()
        counts = data["manifest"]["counts"]
        print("[*] Dry-run complete. Counts summary:")
        print(json.dumps(counts, indent=2))

        if args.profile == "full" and inject_errors:
            err_counts = counts.get("error_class_counts", {})
            hn_counts = counts.get("hard_negative_counts", {})
            verify_floors(err_counts, hn_counts)
        return

    print(
        f"[*] Generating synthetic dataset -> {target_dir} "
        f"(Profile: {args.profile}, Split: {args.split}, Seed: {config.seed}, Inject: {inject_errors})..."
    )
    paths = generator.write_to_directory(target_dir)
    print(f"[*] Successfully generated {len(paths)} dataset files:")
    for name, path in paths.items():
        size_bytes = os.path.getsize(path)
        print(f"  - {name}: {path} ({size_bytes} bytes)")

    if args.profile == "full" and inject_errors:
        with open(paths["manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        err_counts = manifest["counts"].get("error_class_counts", {})
        hn_counts = manifest["counts"].get("hard_negative_counts", {})
        verify_floors(err_counts, hn_counts)


if __name__ == "__main__":
    main()
