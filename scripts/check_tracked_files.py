#!/usr/bin/env python3
"""Audit tracked files in git to report and flag any files under data/ or results/."""
from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_tracked_files() -> list[str]:
    cmd = ["git", "ls-files"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def audit_tracked_files() -> dict[str, list[str]]:
    files = get_tracked_files()
    data_files = [f for f in files if f.startswith("data/") or f.startswith("data\\")]
    results_files = [f for f in files if f.startswith("results/") or f.startswith("results\\")]
    return {
        "data": data_files,
        "results": results_files,
        "total_tracked": files,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    report = audit_tracked_files()
    print("=" * 80)
    print("GIT TRACKED FILES REPORT (DATA & RESULTS AUDIT)")
    print("=" * 80)
    print(f"Total Tracked Files in Git: {len(report['total_tracked'])}")
    print(f"Tracked Files under data/   : {len(report['data'])}")
    print(f"Tracked Files under results/: {len(report['results'])}")
    print("=" * 80)

    if report["data"]:
        print("\n[FLAGGED] Files tracked under data/:")
        for f in report["data"]:
            print(f"  - {f}")

    if report["results"]:
        print("\n[FLAGGED] Files tracked under results/:")
        for f in report["results"][:20]:
            print(f"  - {f}")
        if len(report["results"]) > 20:
            print(f"  ... and {len(report['results']) - 20} more files.")

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
