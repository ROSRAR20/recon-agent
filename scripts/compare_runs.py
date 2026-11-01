#!/usr/bin/env python3
"""CLI utility to compare two evaluation results JSON files and print changed metrics row by row."""

import argparse
import json
import os
import sys
from typing import Any


def format_val(val: Any) -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, str)):
        return str(val)
    return str(val)


def compute_delta(v1: Any, v2: Any) -> str:
    """Compute difference string between two values."""
    if v1 is None and v2 is None:
        return "-"
    if v1 is None:
        return f"+{format_val(v2)}"
    if v2 is None:
        return f"-{format_val(v1)}"
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        diff = v2 - v1
        if isinstance(v1, int) and isinstance(v2, int):
            return f"{diff:+d}"
        return f"{diff:+.4f}"
    if v1 != v2:
        return "changed"
    return "-"


def extract_metrics(d: dict[str, Any]) -> dict[str, Any]:
    """Extract flattened dictionary of metric paths from results JSON."""
    metrics: dict[str, Any] = {}

    # 1. Tier 1: Class-Correct
    for cls, stats in d.get("tier1_class_correct", {}).items():
        for k in ["tp", "fp", "fn", "precision", "recall", "f1"]:
            if k in stats:
                metrics[f"Tier 1 (Class-Correct) > {cls} > {k}"] = stats[k]

    # 2. Tier 1: Class-Agnostic
    for cls, stats in d.get("tier1_class_agnostic", {}).items():
        for k in ["tp", "fp", "fn", "precision", "recall", "f1"]:
            if k in stats:
                metrics[f"Tier 1 (Class-Agnostic) > {cls} > {k}"] = stats[k]

    # 3. Overall / Headline metrics
    for micro_key, label in [
        ("tier1_class_correct_micro", "Tier 1 (Class-Correct Micro)"),
        ("tier1_class_agnostic_micro", "Tier 1 (Class-Agnostic Micro)"),
        ("tier1_class_correct_macro", "Tier 1 (Class-Correct Macro)"),
        ("tier1_class_agnostic_macro", "Tier 1 (Class-Agnostic Macro)"),
    ]:
        if micro_key in d:
            for k in ["tp", "fp", "fn", "precision", "recall", "f1"]:
                if k in d[micro_key]:
                    metrics[f"{label} > {k}"] = d[micro_key][k]

    for k in [
        "macro_f1_undefined_as_zero",
        "macro_agnostic_f1_undefined_as_zero",
    ]:
        if k in d:
            metrics[f"Overall > {k}"] = d[k]

    # 4. Hard Negatives
    for hn, stats in d.get("hard_negatives", {}).items():
        for k in ["flagged", "total", "fpr"]:
            if k in stats:
                metrics[f"Hard Negatives > {hn} > {k}"] = stats[k]

    # 5. Ambiguity Tier
    for amb, stats in d.get("ambiguity_tier", {}).items():
        for k in ["correct", "total", "accuracy"]:
            if k in stats:
                metrics[f"Ambiguity Tier > {amb} > {k}"] = stats[k]

    # 6. Matching
    for k in ["links_made", "correct_links", "wrong_links", "false_match_rate", "match_rate"]:
        if k in d.get("matching", {}):
            metrics[f"Matching > {k}"] = d["matching"][k]

    # 7. Totals & Global counts
    for k in ["clean_false_positives", "total_flagged_predictions", "total_labels"]:
        if k in d:
            metrics[f"Totals > {k}"] = d[k]

    return metrics


def compare_results(file1_path: str, file2_path: str) -> list[tuple[str, Any, Any, str]]:
    """Compare two result files and return list of (metric_name, val1, val2, delta) for all differences."""
    with open(file1_path, "r", encoding="utf-8") as f:
        d1 = json.load(f)
    with open(file2_path, "r", encoding="utf-8") as f:
        d2 = json.load(f)

    m1 = extract_metrics(d1)
    m2 = extract_metrics(d2)

    all_keys = sorted(set(m1.keys()) | set(m2.keys()))
    diffs = []

    for k in all_keys:
        v1 = m1.get(k)
        v2 = m2.get(k)

        is_different = False
        if v1 is None and v2 is not None:
            is_different = True
        elif v1 is not None and v2 is None:
            is_different = True
        elif isinstance(v1, float) and isinstance(v2, float):
            is_different = abs(v1 - v2) > 1e-5
        elif v1 != v2:
            is_different = True

        if is_different:
            delta = compute_delta(v1, v2)
            diffs.append((k, v1, v2, delta))

    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two reconciliation results JSON files.")
    parser.add_argument("file1", help="Path to first (baseline / prior run) results JSON file")
    parser.add_argument("file2", help="Path to second (new run) results JSON file")

    args = parser.parse_args()

    if not os.path.isfile(args.file1):
        print(f"Error: File not found: {args.file1}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.file2):
        print(f"Error: File not found: {args.file2}", file=sys.stderr)
        sys.exit(1)

    diffs = compare_results(args.file1, args.file2)

    name1 = os.path.basename(args.file1)
    name2 = os.path.basename(args.file2)

    print(f"\n# Results Comparison: {name1} vs {name2}")
    if not diffs:
        print("\nNo metric changes detected. Both runs are identical.")
        return

    print(f"\nTotal metrics changed: {len(diffs)}\n")
    headers = ["Metric Path", f"{name1}", f"{name2}", "Delta"]
    table_rows = []
    for k, v1, v2, delta in diffs:
        table_rows.append([k, format_val(v1), format_val(v2), delta])

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    # Format Markdown Table
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|:" + "-|-:".join("-" * (col_widths[i] - 1) for i in range(len(headers))) + ":|"
    print(header_line)
    print(sep_line)
    for row in table_rows:
        line = f"| {row[0].ljust(col_widths[0])} | {row[1].rjust(col_widths[1])} | {row[2].rjust(col_widths[2])} | {row[3].rjust(col_widths[3])} |"
        print(line)
    print()


if __name__ == "__main__":
    main()

