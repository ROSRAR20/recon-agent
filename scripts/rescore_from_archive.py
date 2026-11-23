"""Regenerate docs/results_main.md offline from exported artifacts archive.

Reads artifacts/*.jsonl.gz and manifest.json, re-scores decisions completely offline,
regenerates docs/results_main.md, and asserts strictly identical numbers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import tempfile
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.main_report import generate_main_report
from scripts.startup_checks import ensure_startup_checks


def extract_numbers_from_report(report_text: str) -> list[str]:
    """Extract all markdown table cells, rates, and numbers from report text."""
    lines = report_text.splitlines()
    data_lines = [l for l in lines if l.strip().startswith("|") or l.strip().startswith("- **")]
    # Normalize whitespace
    return [l.strip() for l in data_lines]


def rescore_from_archive(
    artifacts_dir: str = "artifacts",
    report_path: str = "docs/results_main.md",
    pop_dir: str = "data/populations",
    allow_dirty: bool = False,
    mock: bool = False,
) -> bool:
    """Regenerate main report from archive and assert numerical equivalence."""
    resolved_art_dir = os.path.join(REPO_ROOT, artifacts_dir)
    manifest_path = os.path.join(resolved_art_dir, "manifest.json")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Artifact manifest missing at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Stage temporary results directory from archive contents
    with tempfile.TemporaryDirectory() as tmp_res_dir:
        archives = manifest.get("archives", [])
        run_ids_found: set[int] = set()

        # Unpack archives back to run state json format in tmp_res_dir
        for arch in archives:
            arch_file = arch["archive_file"]
            arch_path = os.path.join(resolved_art_dir, arch_file)
            arch_type = arch.get("type")
            run_id = arch.get("run_id", 0)
            run_ids_found.add(run_id)

            cases: dict[str, Any] = {}
            with gzip.open(arch_path, "rt", encoding="utf-8") as gz:
                for line in gz:
                    if line.strip():
                        item = json.loads(line)
                        bid = item["bundle_id"]
                        cases[bid] = item

            if arch_type == "main_run":
                scope = arch.get("scope", "full")
                state_file = os.path.join(tmp_res_dir, f"main_run_state_run{run_id}_{scope}.json")
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump({"cases": cases, "run_id": run_id, "scope": scope}, f, indent=2)

            elif arch_type == "twin_run":
                seed = arch.get("seed", 2)
                t_state_file = os.path.join(tmp_res_dir, f"twin_run_state_seed{seed}_run{run_id}.json")
                with open(t_state_file, "w", encoding="utf-8") as f:
                    json.dump({"cases": cases, "run_id": run_id, "seed": seed}, f, indent=2)

        # Also copy twin_evaluation_summary if present in original results
        orig_twin_sum = os.path.join(REPO_ROOT, "results", "twin_evaluation_summary_run0.json")
        if os.path.isfile(orig_twin_sum):
            with open(orig_twin_sum, "r", encoding="utf-8") as f_in, open(
                os.path.join(tmp_res_dir, "twin_evaluation_summary_run0.json"), "w", encoding="utf-8"
            ) as f_out:
                f_out.write(f_in.read())

        # Regenerate report in temporary file
        tmp_report_path = os.path.join(tmp_res_dir, "results_main_rescored.md")
        sorted_run_ids = tuple(sorted(run_ids_found)) or (0, 1, 2)

        regenerated_text = generate_main_report(
            run_ids=sorted_run_ids,
            out_path=tmp_report_path,
            pop_dir=pop_dir,
            results_dir=tmp_res_dir,
            mock=mock,
        )

        resolved_report_path = os.path.join(REPO_ROOT, report_path)
        if not os.path.isfile(resolved_report_path):
            raise FileNotFoundError(f"Existing report not found at {resolved_report_path}")

        with open(resolved_report_path, "r", encoding="utf-8") as f:
            existing_text = f.read()

        # Extract data rows/numbers
        existing_rows = extract_numbers_from_report(existing_text)
        regenerated_rows = extract_numbers_from_report(regenerated_text)

        # Assert identical rows
        if existing_rows != regenerated_rows:
            diff_lines = []
            for idx, (ex, reg) in enumerate(zip(existing_rows, regenerated_rows)):
                if ex != reg:
                    diff_lines.append(f"Row {idx}:\n  Existing   : {ex}\n  Regenerated: {reg}")
            if len(existing_rows) != len(regenerated_rows):
                diff_lines.append(f"Row count mismatch: existing={len(existing_rows)}, regenerated={len(regenerated_rows)}")

            raise AssertionError("Rescored report numbers DO NOT match existing report:\n" + "\n".join(diff_lines[:10]))

        print("====================================================================")
        print("[RE-SCORE MATCH] PASS")
        print(f"All {len(existing_rows)} metric and hypothesis data rows match existing report identically.")
        print("====================================================================")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore and verify main report from exported archive")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--report", default="docs/results_main.md", help="Existing report path")
    parser.add_argument("--pop-dir", default="data/populations", help="Populations directory")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree")
    args = parser.parse_args()

    rescore_from_archive(
        artifacts_dir=args.artifacts_dir,
        report_path=args.report,
        pop_dir=args.pop_dir,
        mock=args.mock,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
