"""Integrity checker for frozen baselines and registration tags.

Verifies:
- Required tags (b1-frozen, checker-v1, prereg) exist and are ancestors of HEAD.
- Frozen baseline files are unchanged versus their registration tags:
  - systems/ unchanged versus b1-frozen
  - independent_checker.py unchanged versus checker-v1
  - verifier.py unchanged versus prereg
- Accurately reports any differences versus registration tags.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FROZEN_TAGS = ("b1-frozen", "checker-v1", "prereg")


def is_ancestor(ancestor_tag: str, head_ref: str = "HEAD") -> bool:
    """Check if ancestor_tag is an ancestor of head_ref."""
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_tag, head_ref],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def get_git_diff(tag: str, path: str) -> str:
    """Return git diff output between tag and HEAD for a given path."""
    try:
        out = subprocess.check_output(
            ["git", "diff", tag, "--", path],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception as exc:
        return f"ERROR: Unable to compute diff: {exc}"


def check_frozen_baselines() -> tuple[bool, dict[str, Any]]:
    """Run full verification of frozen tags and files."""
    results: dict[str, Any] = {
        "ancestors": {},
        "diffs": {},
        "all_passed": True,
        "failures": [],
    }

    # 1. Ancestor verification
    for tag in FROZEN_TAGS:
        ok = is_ancestor(tag)
        results["ancestors"][tag] = ok
        if not ok:
            results["all_passed"] = False
            results["failures"].append(f"Tag '{tag}' is not an ancestor of HEAD")

    # 2. Frozen systems check
    diff_systems = get_git_diff("b1-frozen", "src/recon_agent/systems/")
    results["diffs"]["systems_vs_b1_frozen"] = len(diff_systems) == 0
    if diff_systems:
        results["all_passed"] = False
        results["failures"].append(f"src/recon_agent/systems/ differs from tag b1-frozen ({len(diff_systems)} bytes diff)")

    # 3. Frozen verifier check
    diff_verifier = get_git_diff("prereg", "src/recon_agent/llm/verifier.py")
    results["diffs"]["verifier_vs_prereg"] = len(diff_verifier) == 0
    if diff_verifier:
        results["all_passed"] = False
        results["failures"].append(f"src/recon_agent/llm/verifier.py differs from tag prereg ({len(diff_verifier)} bytes diff)")

    # 4. Frozen independent checker check versus checker-v1
    diff_checker = get_git_diff("checker-v1", "src/recon_agent/eval/independent_checker.py")
    results["diffs"]["checker_vs_checker_v1"] = len(diff_checker) == 0
    if diff_checker:
        results["all_passed"] = False
        results["failures"].append(f"src/recon_agent/eval/independent_checker.py differs from tag checker-v1 ({len(diff_checker)} bytes diff)")

    # 5. Diagnostic diff of checker versus prereg (prereg had 0-byte file, checker-v1 had registered 166 lines)
    diff_checker_prereg = get_git_diff("prereg", "src/recon_agent/eval/independent_checker.py")
    results["diffs"]["checker_vs_prereg_note"] = (
        "identical" if len(diff_checker_prereg) == 0 else "differs (+166 lines registered in commit 73b8ce1 / tag checker-v1)"
    )

    return results["all_passed"], results


def main() -> None:
    passed, details = check_frozen_baselines()
    print("=" * 60)
    print("FROZEN BASELINES INTEGRITY CHECK")
    print("=" * 60)
    for tag, is_anc in details["ancestors"].items():
        status = "PASS (ancestor of HEAD)" if is_anc else "FAIL (not an ancestor)"
        print(f"Tag {tag:<12}: {status}")

    print("-" * 60)
    print(f"systems/ vs b1-frozen            : {'PASS (0 diff)' if details['diffs']['systems_vs_b1_frozen'] else 'FAIL'}")
    print(f"verifier.py vs prereg            : {'PASS (0 diff)' if details['diffs']['verifier_vs_prereg'] else 'FAIL'}")
    print(f"independent_checker vs checker-v1: {'PASS (0 diff)' if details['diffs']['checker_vs_checker_v1'] else 'FAIL'}")
    print(f"independent_checker vs prereg    : NOTE ({details['diffs']['checker_vs_prereg_note']})")
    print("=" * 60)

    if passed:
        print("[FROZEN CHECK] PASS")
        sys.exit(0)
    else:
        print("[FROZEN CHECK] FAIL")
        for fail in details["failures"]:
            print(f"  - {fail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
