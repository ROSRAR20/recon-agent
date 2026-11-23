"""Build main evaluation population for dev seeds 1-3.

Combines dev seeds 1-3 (residual + controls) in registered order, strictly
excluding the 120 pilot bundles (first 100 residual and first 20 control of seed 1).
Writes:
- data/populations/main_population.jsonl
- data/populations/main_meta.jsonl
- data/populations/main_excluded_pilot_ids.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.build_populations import build_population_for_seed
from scripts.startup_checks import ensure_startup_checks


def get_pilot_excluded_ids(pop_dir: str = "data/populations") -> list[str]:
    """Retrieve the exact 120 pilot bundles: first 100 residual and first 20 control of dev seed 1."""
    meta_path = os.path.join(REPO_ROOT, pop_dir, "dev_seed1_meta.jsonl")
    if not os.path.isfile(meta_path):
        # Build dev_seed1 if not present
        build_population_for_seed(seed=1, output_dir=pop_dir)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_records = [json.loads(line) for line in f if line.strip()]

    excluded_ids: list[str] = []
    res_count = 0
    ctrl_count = 0
    for m in meta_records:
        bid = m["bundle_id"]
        is_ctrl = m.get("is_control", False)
        if is_ctrl and ctrl_count < 20:
            excluded_ids.append(bid)
            ctrl_count += 1
        elif not is_ctrl and res_count < 100:
            excluded_ids.append(bid)
            res_count += 1

        if res_count >= 100 and ctrl_count >= 20:
            break

    if len(excluded_ids) != 120 or res_count != 100 or ctrl_count != 20:
        raise ValueError(
            f"Failed to identify pilot prefix: expected 100 res + 20 ctrl (120 total), got {len(excluded_ids)} (res={res_count}, ctrl={ctrl_count})"
        )
    return excluded_ids


def build_main_population(
    pop_dir: str = "data/populations",
    seeds: tuple[int, ...] = (1, 2, 3),
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Build the main population (dev seeds 1-3 excluding seed 1 pilot prefix)."""
    # 0. Startup checks
    ensure_startup_checks(repo_root=REPO_ROOT, allow_dirty=allow_dirty)

    resolved_pop_dir = os.path.join(REPO_ROOT, pop_dir)
    os.makedirs(resolved_pop_dir, exist_ok=True)

    # 1. Identify excluded pilot IDs
    excluded_ids = get_pilot_excluded_ids(pop_dir=pop_dir)
    excluded_set = set(excluded_ids)
    excluded_json_bytes = json.dumps(excluded_ids, indent=2).encode("utf-8")
    excluded_hash = hashlib.sha256(excluded_json_bytes).hexdigest()

    excluded_file = os.path.join(resolved_pop_dir, "main_excluded_pilot_ids.json")
    with open(excluded_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Dev seed 1 pilot bundles (first 100 residual + first 20 control) excluded from main population",
                "count": len(excluded_ids),
                "sha256": excluded_hash,
                "excluded_bundle_ids": excluded_ids,
            },
            f,
            indent=2,
        )

    # 2. Collect bundles and metadata across seeds 1-3
    main_bundles_lines: list[str] = []
    main_meta_lines: list[str] = []

    seed_stats: dict[int, dict[str, Any]] = {}
    total_family_counts: Counter[str] = Counter()
    residual_bundle_ids: set[str] = set()
    control_bundle_ids: set[str] = set()

    for seed in seeds:
        pop_path = os.path.join(resolved_pop_dir, f"dev_seed{seed}.jsonl")
        meta_path = os.path.join(resolved_pop_dir, f"dev_seed{seed}_meta.jsonl")

        if not os.path.isfile(pop_path) or not os.path.isfile(meta_path):
            build_population_for_seed(seed=seed, output_dir=pop_dir)

        with open(pop_path, "r", encoding="utf-8") as f_pop, open(meta_path, "r", encoding="utf-8") as f_meta:
            b_lines = [l.strip() for l in f_pop if l.strip()]
            m_lines = [l.strip() for l in f_meta if l.strip()]

        if len(b_lines) != len(m_lines):
            raise ValueError(f"Mismatch in seed {seed}: {len(b_lines)} bundles vs {len(m_lines)} metadata records")

        seed_res_count = 0
        seed_ctrl_count = 0
        seed_fam_counts: Counter[str] = Counter()

        for b_str, m_str in zip(b_lines, m_lines):
            b_obj = json.loads(b_str)
            m_obj = json.loads(m_str)
            bid = b_obj["bundle_id"]
            check_family = b_obj["check_family"]
            is_ctrl = m_obj.get("is_control", False)

            # Exclude pilot bundles for seed 1
            if seed == 1 and bid in excluded_set:
                continue

            # Invariant check: control bundle must not contain notes reason text
            if is_ctrl:
                control_bundle_ids.add(bid)
                seed_ctrl_count += 1
                untrusted = b_obj.get("untrusted_text") or {}
                if untrusted.get("recon_item_notes_reason_text") is not None:
                    raise AssertionError(f"Control bundle {bid} contains recon_item_notes_reason_text")
            else:
                residual_bundle_ids.add(bid)
                seed_res_count += 1

            seed_fam_counts[check_family] += 1
            total_family_counts[check_family] += 1
            main_bundles_lines.append(b_str)
            main_meta_lines.append(m_str)

        seed_stats[seed] = {
            "residual_count": seed_res_count,
            "control_count": seed_ctrl_count,
            "total_count": seed_res_count + seed_ctrl_count,
            "family_counts": dict(seed_fam_counts),
        }

    # 3. Assert disjointness invariants
    # Invariant A: No excluded pilot ID in main population
    overlap_pilot = excluded_set & (residual_bundle_ids | control_bundle_ids)
    if overlap_pilot:
        raise AssertionError(f"Excluded pilot IDs leaked into main population: {overlap_pilot}")

    # Invariant B: Residual and control sets must be strictly disjoint
    overlap_res_ctrl = residual_bundle_ids & control_bundle_ids
    if overlap_res_ctrl:
        raise AssertionError(f"Bundle IDs appear in both residual and control sets: {overlap_res_ctrl}")

    # 4. Write output files atomically
    out_pop_file = os.path.join(resolved_pop_dir, "main_population.jsonl")
    out_meta_file = os.path.join(resolved_pop_dir, "main_meta.jsonl")

    tmp_pop = f"{out_pop_file}.tmp"
    tmp_meta = f"{out_meta_file}.tmp"

    with open(tmp_pop, "w", encoding="utf-8") as f:
        for line in main_bundles_lines:
            f.write(line + "\n")

    with open(tmp_meta, "w", encoding="utf-8") as f:
        for line in main_meta_lines:
            f.write(line + "\n")

    os.replace(tmp_pop, out_pop_file)
    os.replace(tmp_meta, out_meta_file)

    total_bundles = len(main_bundles_lines)
    summary = {
        "total_bundles": total_bundles,
        "residual_total": len(residual_bundle_ids),
        "control_total": len(control_bundle_ids),
        "excluded_pilot_count": len(excluded_ids),
        "excluded_pilot_hash": excluded_hash,
        "family_counts": dict(total_family_counts),
        "seed_breakdown": seed_stats,
        "output_population_path": out_pop_file,
        "output_metadata_path": out_meta_file,
        "excluded_ids_path": excluded_file,
    }

    # Print report
    print("====================================================================")
    print("MAIN EVALUATION POPULATION BUILD SUMMARY")
    print("====================================================================")
    print(f"Total Bundles       : {total_bundles} (Residual: {len(residual_bundle_ids)}, Control: {len(control_bundle_ids)})")
    print(f"Excluded Pilot IDs  : {len(excluded_ids)} (SHA-256: {excluded_hash[:16]}...)")
    print(f"Disjointness Checks : PASS (pilot overlap=0, residual/control overlap=0)")
    print("--------------------------------------------------------------------")
    print("Counts per Check Family (Overall):")
    for fam, cnt in sorted(total_family_counts.items()):
        print(f"  {fam:<15}: {cnt}")
    print("--------------------------------------------------------------------")
    print("Seed Breakdown:")
    for s, stats in sorted(seed_stats.items()):
        print(f"  Seed {s}: total={stats['total_count']} (res={stats['residual_count']}, ctrl={stats['control_count']}) | families={stats['family_counts']}")
    print("====================================================================")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build main evaluation population for dev seeds 1-3")
    parser.add_argument("--pop-dir", default="data/populations", help="Populations directory")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree for test fixtures")
    args = parser.parse_args()

    build_main_population(pop_dir=args.pop_dir, allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()
