"""Build evaluation populations and metadata for pilot and benchmark evaluations.

Generates:
- data/populations/dev_seed<N>.jsonl: Canonical CaseBundle objects in deterministic stratified order.
- data/populations/dev_seed<N>_meta.jsonl: Separate eval-only metadata keyed by bundle_id.

Strictly refuses any test split requests.
Bundles never contain evaluation metadata fields.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.eval.expected_decisions import get_expected_decision
from src.recon_agent.eval.scoring import load_ground_truth
from src.recon_agent.llm.bundle import (
    CaseBundle,
    build_case_bundles,
    build_control_bundles,
)
from src.recon_agent.pipeline.loader import load_dataset
from src.recon_agent.systems.b1r import run_b1r


def build_population_for_seed(
    seed: int = 1,
    split: str = "dev",
    output_dir: str = "data/populations",
    data_dir: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Build and write stratified population and metadata files for a given seed.

    Refuses any split other than 'dev'.
    """
    if split.lower() != "dev":
        raise ValueError(
            f"Split '{split}' refused: the test split is not allowed before the final run."
        )

    if data_dir is None:
        resolved_data_dir = os.path.join(REPO_ROOT, "data", "synth", "full", f"dev_seed{seed}")
        if not os.path.isdir(resolved_data_dir):
            # Fallback to dev without seed suffix if dev_seed<N> directory is not present
            fallback_dir = os.path.join(REPO_ROOT, "data", "synth", "full", "dev")
            if os.path.isdir(fallback_dir):
                resolved_data_dir = fallback_dir
            else:
                raise FileNotFoundError(f"Dataset directory not found: {resolved_data_dir}")
    else:
        resolved_data_dir = os.path.abspath(data_dir)

    print(f"[POPULATION] Loading observable dataset from {resolved_data_dir} (seed={seed})")
    dataset = load_dataset(resolved_data_dir)
    labels, _ = load_ground_truth(resolved_data_dir)

    # Fast label lookup by entity_id
    entity_to_label = {}
    for lbl in labels:
        for eid in lbl.get("entity_ids", []):
            entity_to_label[eid] = lbl
        for eid in lbl.get("primary_entity_ids", []):
            entity_to_label[eid] = lbl

    # 1. Residual cases: B1r escalations + refunds with reason_text
    residual_bundles = build_case_bundles(dataset)

    # 2. Control cases: clean negative controls (60 per seed)
    system_output = run_b1r(dataset)
    control_bundles = build_control_bundles(
        dataset=dataset,
        system_output=system_output,
        n_per_seed=60,
        rng_seed=seed,
        results_dir=os.path.join(REPO_ROOT, "results"),
    )

    # Invariant assertions: residual and control sets must be strictly disjoint,
    # and controls must never contain recon_item_notes_reason_text.
    overlap_bids = set(b.bundle_id for b in residual_bundles) & set(b.bundle_id for b in control_bundles)
    assert not overlap_bids, f"Bundle IDs appear in both residual and control sets: {overlap_bids}"
    for b in control_bundles:
        assert b.untrusted_text.get("recon_item_notes_reason_text") is None, (
            f"Control bundle {b.bundle_id} contains recon_item_notes_reason_text"
        )

    # 3. Stratification
    # 3a. Stratify residual bundles by (check_family, ambiguity_type)
    res_buckets: dict[tuple[str, str], list[CaseBundle]] = defaultdict(list)
    res_meta_map = {}
    for b in residual_bundles:
        lbl = entity_to_label.get(b.entity_id)
        exp = get_expected_decision(b, label=lbl, is_control=False)
        res_meta_map[b.bundle_id] = exp
        res_buckets[(b.check_family, exp.source_label_type)].append(b)

    for k in res_buckets:
        res_buckets[k].sort(key=lambda x: x.bundle_id)

    stratified_residuals: list[CaseBundle] = []
    res_keys = sorted(res_buckets.keys())
    while any(res_buckets[k] for k in res_keys):
        for k in res_keys:
            if res_buckets[k]:
                stratified_residuals.append(res_buckets[k].pop(0))

    # 3b. Stratify control bundles by check_family
    ctrl_buckets: dict[str, list[CaseBundle]] = defaultdict(list)
    ctrl_meta_map = {}
    for b in control_bundles:
        exp = get_expected_decision(b, label=None, is_control=True)
        ctrl_meta_map[b.bundle_id] = exp
        ctrl_buckets[b.check_family].append(b)

    for k in ctrl_buckets:
        ctrl_buckets[k].sort(key=lambda x: x.bundle_id)

    stratified_controls: list[CaseBundle] = []
    ctrl_keys = sorted(ctrl_buckets.keys())
    while any(ctrl_buckets[k] for k in ctrl_keys):
        for k in ctrl_keys:
            if ctrl_buckets[k]:
                stratified_controls.append(ctrl_buckets[k].pop(0))

    # 3c. Interleave residuals and controls proportionally so ANY prefix is balanced
    total_res = len(stratified_residuals)
    total_ctrl = len(stratified_controls)
    total_count = total_res + total_ctrl

    ordered_bundles: list[CaseBundle] = []
    r_idx = 0
    c_idx = 0
    for _ in range(total_count):
        if c_idx < total_ctrl and (
            r_idx >= total_res or (c_idx + 1) * total_res <= (r_idx + 1) * total_ctrl
        ):
            ordered_bundles.append(stratified_controls[c_idx])
            c_idx += 1
        else:
            ordered_bundles.append(stratified_residuals[r_idx])
            r_idx += 1

    # 4. Write data/populations/dev_seed<N>.jsonl and dev_seed<N>_meta.jsonl
    resolved_output_dir = (
        output_dir if os.path.isabs(output_dir) else os.path.join(REPO_ROOT, output_dir)
    )
    os.makedirs(resolved_output_dir, exist_ok=True)

    pop_filepath = os.path.join(resolved_output_dir, f"dev_seed{seed}.jsonl")
    meta_filepath = os.path.join(resolved_output_dir, f"dev_seed{seed}_meta.jsonl")

    temp_pop_path = f"{pop_filepath}.tmp"
    temp_meta_path = f"{meta_filepath}.tmp"

    with open(temp_pop_path, "w", encoding="utf-8") as f_pop, open(
        temp_meta_path, "w", encoding="utf-8"
    ) as f_meta:
        for b in ordered_bundles:
            is_ctrl = b.bundle_id in ctrl_meta_map
            exp = ctrl_meta_map[b.bundle_id] if is_ctrl else res_meta_map[b.bundle_id]

            # Canonical bundle representation ONLY - never contains eval metadata fields
            f_pop.write(b.to_canonical_json() + "\n")

            # Eval metadata
            meta_record = {
                "bundle_id": b.bundle_id,
                "is_control": is_ctrl,
                "expected_decision": exp.decision,
                "expected_class": exp.predicted_class,
                "source_label_type": exp.source_label_type,
            }
            f_meta.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

    os.replace(temp_pop_path, pop_filepath)
    os.replace(temp_meta_path, meta_filepath)

    # 5. Summary counts
    res_fam_counts = Counter(b.check_family for b in stratified_residuals)
    ctrl_fam_counts = Counter(b.check_family for b in stratified_controls)
    total_fam_counts = Counter(b.check_family for b in ordered_bundles)

    counts_report = {
        "seed": seed,
        "split": split,
        "residual_counts": dict(res_fam_counts),
        "residual_total": total_res,
        "control_counts": dict(ctrl_fam_counts),
        "control_total": total_ctrl,
        "total_counts": dict(total_fam_counts),
        "total_bundles": total_count,
    }

    return pop_filepath, meta_filepath, counts_report


def print_counts_report(report: dict[str, Any]) -> None:
    """Print readable family breakdown counts."""
    seed = report["seed"]
    split = report["split"]
    print(f"\n================ POPULATION COUNTS (split={split}, seed={seed}) ================")
    print("Residual cases (by check_family):")
    for fam, cnt in sorted(report["residual_counts"].items()):
        print(f"  {fam:<12}: {cnt:>4}")
    print(f"  {'Total residual':<12}: {report['residual_total']:>4}")

    print("\nControl cases (by check_family):")
    for fam, cnt in sorted(report["control_counts"].items()):
        print(f"  {fam:<12}: {cnt:>4}")
    print(f"  {'Total control':<12}: {report['control_total']:>4}")

    print("\nCombined population (by check_family):")
    for fam, cnt in sorted(report["total_counts"].items()):
        print(f"  {fam:<12}: {cnt:>4}")
    print(f"  {'Total overall':<12}: {report['total_bundles']:>4}")
    print("=======================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation populations and metadata.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed / cohort index (e.g. 1, 2, 3)")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split (strictly 'dev')")
    parser.add_argument("--output-dir", type=str, default="data/populations", help="Output directory")
    args = parser.parse_args()

    try:
        pop_path, meta_path, report = build_population_for_seed(
            seed=args.seed,
            split=args.split,
            output_dir=args.output_dir,
        )
        print_counts_report(report)
        print(f"Population written: {pop_path}")
        print(f"Metadata written:   {meta_path}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
