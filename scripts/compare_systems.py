#!/usr/bin/env python3
"""Multi-system comparison tool for pooled reconciliation benchmarks.

Usage:
    python scripts/compare_systems.py results/b0_dev_pooled.json results/b1_dev_pooled.json results/b1r_dev_pooled.json
"""

import argparse
import json
import os
import sys
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.eval.scoring import (
    AMBIGUITY_TYPES,
    HARD_NEGATIVE_TYPES,
    TIER1_ERROR_CLASSES,
)


def load_pooled_json(path: str) -> dict[str, Any]:
    """Load pooled JSON evaluation result from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_rows(systems_data: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """Construct comparison table rows (metric_label, [val_sys1, val_sys2, ...])."""
    rows: list[tuple[str, list[str]]] = []

    # 1. Tier 1 Class-Correct Precision / Recall / F1 (21 rows)
    for c in TIER1_ERROR_CLASSES:
        # Precision
        p_cells = []
        for d in systems_data:
            m = d["tier1_class_correct"][c]
            if m["precision"] is not None:
                p_cells.append(f"{m['precision'] * 100:.2f}% ({m['precision_kn']})")
            else:
                p_cells.append(f"N/A ({m['precision_kn']})")
        rows.append((f"Tier 1 Correct: {c} Precision", p_cells))

        # Recall
        r_cells = []
        for d in systems_data:
            m = d["tier1_class_correct"][c]
            if m["recall"] is not None:
                r_cells.append(f"{m['recall'] * 100:.2f}% ({m['recall_kn']})")
            else:
                r_cells.append(f"N/A ({m['recall_kn']})")
        rows.append((f"Tier 1 Correct: {c} Recall", r_cells))

        # F1
        f_cells = []
        for d in systems_data:
            m = d["tier1_class_correct"][c]
            if m["f1"] is not None:
                f_cells.append(f"{m['f1']:.4f}")
            else:
                f_cells.append("N/A")
        rows.append((f"Tier 1 Correct: {c} F1", f_cells))

    # 2. Tier 1 Overall Micro Precision / Recall / F1 (3 rows)
    micro_p_cells = []
    for d in systems_data:
        m = d["tier1_class_correct_micro"]
        denom = m["tp"] + m["fp"]
        if m["precision"] is not None:
            micro_p_cells.append(f"{m['precision'] * 100:.2f}% ({m['tp']}/{denom})")
        else:
            micro_p_cells.append(f"N/A (0/{denom})")
    rows.append(("Tier 1 Overall Micro Precision", micro_p_cells))

    micro_r_cells = []
    for d in systems_data:
        m = d["tier1_class_correct_micro"]
        denom = m["tp"] + m["fn"]
        if m["recall"] is not None:
            micro_r_cells.append(f"{m['recall'] * 100:.2f}% ({m['tp']}/{denom})")
        else:
            micro_r_cells.append(f"N/A (0/{denom})")
    rows.append(("Tier 1 Overall Micro Recall", micro_r_cells))

    micro_f_cells = []
    for d in systems_data:
        m = d["tier1_class_correct_micro"]
        if m["f1"] is not None:
            micro_f_cells.append(f"{m['f1']:.4f}")
        else:
            micro_f_cells.append("N/A")
    rows.append(("Tier 1 Overall Micro F1", micro_f_cells))

    # 3. Macro-F1 (1 row)
    macro_f_cells = []
    for d in systems_data:
        macro_f_cells.append(f"{d['macro_f1_undefined_as_zero']:.4f}")
    rows.append(("Macro-F1 (undefined as 0)", macro_f_cells))

    # 4. Tier 1 Class-Agnostic Overall Micro Precision / Recall / F1 (3 rows)
    agnostic_p_cells = []
    for d in systems_data:
        m = d["tier1_class_agnostic_micro"]
        denom = m["tp"] + m["fp"]
        if m["precision"] is not None:
            agnostic_p_cells.append(f"{m['precision'] * 100:.2f}% ({m['tp']}/{denom})")
        else:
            agnostic_p_cells.append(f"N/A (0/{denom})")
    rows.append(("Tier 1 Agnostic Micro Precision", agnostic_p_cells))

    agnostic_r_cells = []
    for d in systems_data:
        m = d["tier1_class_agnostic_micro"]
        denom = m["tp"] + m["fn"]
        if m["recall"] is not None:
            agnostic_r_cells.append(f"{m['recall'] * 100:.2f}% ({m['tp']}/{denom})")
        else:
            agnostic_r_cells.append(f"N/A (0/{denom})")
    rows.append(("Tier 1 Agnostic Micro Recall", agnostic_r_cells))

    agnostic_f_cells = []
    for d in systems_data:
        m = d["tier1_class_agnostic_micro"]
        if m["f1"] is not None:
            agnostic_f_cells.append(f"{m['f1']:.4f}")
        else:
            agnostic_f_cells.append("N/A")
    rows.append(("Tier 1 Agnostic Micro F1", agnostic_f_cells))

    # 5. Hard Negatives False Positive Rate for each of the 10 types (10 rows)
    for hn in HARD_NEGATIVE_TYPES:
        hn_cells = []
        for d in systems_data:
            m = d["hard_negatives"][hn]
            flg = m["flagged"]
            tot = m["total"]
            fpr = m["fpr"]
            hn_cells.append(f"{fpr * 100:.2f}% ({flg}/{tot})")
        rows.append((f"Hard Negative: {hn} FPR", hn_cells))

    # 6. Hard Negatives Overall (1 row)
    hn_overall_cells = []
    for d in systems_data:
        tot_flg = sum(d["hard_negatives"][hn]["flagged"] for hn in HARD_NEGATIVE_TYPES)
        tot_tot = sum(d["hard_negatives"][hn]["total"] for hn in HARD_NEGATIVE_TYPES)
        overall_fpr = round(tot_flg / tot_tot, 4) if tot_tot > 0 else 0.0
        hn_overall_cells.append(f"{overall_fpr * 100:.2f}% ({tot_flg}/{tot_tot})")
    rows.append(("Hard Negatives Overall FPR", hn_overall_cells))

    # 7. Ambiguity Tier Action Accuracy for each of the 4 types (4 rows)
    for amb in AMBIGUITY_TYPES:
        amb_cells = []
        for d in systems_data:
            m = d["ambiguity_tier"][amb]
            amb_cells.append(f"{m['accuracy'] * 100:.2f}% ({m['accuracy_kn']})")
        rows.append((f"Ambiguity: {amb} Accuracy", amb_cells))

    # 8. Bank Matching: Match Rate, False-Match Rate (2 rows)
    mr_cells = []
    for d in systems_data:
        m = d["matching"]
        if m["match_rate"] is not None:
            mr_cells.append(f"{m['match_rate'] * 100:.2f}%")
        else:
            mr_cells.append("N/A")
    rows.append(("Bank Match Rate", mr_cells))

    fmr_cells = []
    for d in systems_data:
        m = d["matching"]
        if m["false_match_rate"] is not None:
            fmr_cells.append(f"{m['false_match_rate'] * 100:.2f}% ({m['wrong_links']}/{m['links_made']})")
        else:
            fmr_cells.append("N/A")
    rows.append(("Bank False-Match Rate", fmr_cells))

    # 9. Operational (5 rows)
    clean_fp_cells = [str(d["clean_false_positives"]) for d in systems_data]
    rows.append(("Clean Data False Positives (Flagged)", clean_fp_cells))

    clean_esc_cells = [str(d.get("clean_escalated", 0)) for d in systems_data]
    rows.append(("Clean Data Escalations", clean_esc_cells))

    tot_flg_cells = [str(d["total_flagged_predictions"]) for d in systems_data]
    rows.append(("Total Flagged Predictions", tot_flg_cells))

    tot_esc_cells = [str(d.get("total_escalated_predictions", 0)) for d in systems_data]
    rows.append(("Total Escalated Predictions", tot_esc_cells))

    esc_share_cells = [f"{d.get('escalation_share', 0.0) * 100:.2f}%" for d in systems_data]
    rows.append(("Escalation Share", esc_share_cells))

    return rows


def generate_comparison_table(json_paths: list[str]) -> str:
    """Generate markdown comparison table from multiple pooled JSON result paths."""
    systems_data = [load_pooled_json(p) for p in json_paths]
    system_names = [d.get("system_id", os.path.basename(p).split("_")[0]).upper() for d, p in zip(systems_data, json_paths)]

    rows = build_comparison_rows(systems_data)

    headers = ["Metric"] + system_names
    aligns = [":---"] + ["---:"] * len(system_names)

    lines = [
        "# Reconciliation Systems Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]

    for metric_name, cells in rows:
        row_str = "| " + metric_name + " | " + " | ".join(cells) + " |"
        lines.append(row_str)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pooled evaluation metrics across multiple reconciliation systems."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="Paths to pooled JSON evaluation files (e.g. results/b0_dev_pooled.json results/b1_dev_pooled.json results/b1r_dev_pooled.json)",
    )
    args = parser.parse_args()

    table_markdown = generate_comparison_table(args.json_files)
    print(table_markdown)


if __name__ == "__main__":
    main()

