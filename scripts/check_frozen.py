"""Integrity checker for frozen baselines and registration tags.

Verifies:
- Required tag b1-frozen exists and is an ancestor of HEAD.
- Required tag registration-final exists and is an ancestor of HEAD.
- Frozen files are unchanged versus tag registration-final:
  - src/recon_agent/eval/independent_checker.py
  - src/recon_agent/llm/verifier.py
  - src/recon_agent/systems/
  - docs/preregistration.md
- src/recon_agent/systems/ is also unchanged versus tag b1-frozen.
- Reports git commit for registration-final and git blob hashes for all frozen files.
- Fails with a clear message if registration-final does not exist.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_tag_commit(tag: str, cwd: str = REPO_ROOT) -> str | None:
    """Return the commit SHA for a given tag, or None if tag does not exist."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out if out else None
    except Exception:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"],
                cwd=cwd,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return out if out else None
        except Exception:
            return None


def is_ancestor(ancestor_ref: str, head_ref: str = "HEAD", cwd: str = REPO_ROOT) -> bool:
    """Check if ancestor_ref is an ancestor of head_ref."""
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_ref, head_ref],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def get_git_diff(tag: str, path: str, cwd: str = REPO_ROOT) -> str:
    """Return git diff output between tag and HEAD for a given path."""
    try:
        out = subprocess.check_output(
            ["git", "diff", tag, "--", path],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception as exc:
        return f"ERROR: Unable to compute diff: {exc}"


def get_git_object_hash(ref_path: str, cwd: str = REPO_ROOT) -> str | None:
    """Return git object hash (blob or tree) for ref:path."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", ref_path],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out if out else None
    except Exception:
        return None


def get_directory_blob_hashes(tree_path: str, cwd: str = REPO_ROOT) -> dict[str, str]:
    """Return dict of {relative_path: blob_hash} for all files under tree_path."""
    try:
        out = subprocess.check_output(
            ["git", "ls-tree", "-r", "HEAD", tree_path],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        file_map: dict[str, str] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "blob":
                b_hash = parts[2]
                f_path = parts[3]
                file_map[f_path] = b_hash
        return file_map
    except Exception:
        return {}


def check_frozen_baselines(repo_root: str | None = None) -> tuple[bool, dict[str, Any]]:
    """Run full verification of frozen tags and files."""
    cwd = repo_root or REPO_ROOT
    results: dict[str, Any] = {
        "registration_final_commit": None,
        "b1_frozen_commit": None,
        "ancestors": {},
        "diffs": {},
        "file_hashes": {},
        "all_passed": True,
        "failures": [],
    }

    # 1. Retrieve blob hashes of frozen files in HEAD
    checker_hash = get_git_object_hash("HEAD:src/recon_agent/eval/independent_checker.py", cwd=cwd)
    verifier_hash = get_git_object_hash("HEAD:src/recon_agent/llm/verifier.py", cwd=cwd)
    prereg_doc_hash = get_git_object_hash("HEAD:docs/preregistration.md", cwd=cwd)
    systems_tree_hash = get_git_object_hash("HEAD:src/recon_agent/systems", cwd=cwd)
    systems_files = get_directory_blob_hashes("src/recon_agent/systems", cwd=cwd)

    results["file_hashes"] = {
        "src/recon_agent/eval/independent_checker.py": checker_hash,
        "src/recon_agent/llm/verifier.py": verifier_hash,
        "docs/preregistration.md": prereg_doc_hash,
        "src/recon_agent/systems/": systems_tree_hash,
        "systems_files": systems_files,
    }

    # 2. Check tag b1-frozen
    b1_commit = get_tag_commit("b1-frozen", cwd=cwd)
    results["b1_frozen_commit"] = b1_commit
    if not b1_commit:
        results["all_passed"] = False
        results["ancestors"]["b1-frozen"] = False
        results["failures"].append("Tag 'b1-frozen' does not exist in repository.")
    else:
        is_b1_anc = is_ancestor("b1-frozen", cwd=cwd)
        results["ancestors"]["b1-frozen"] = is_b1_anc
        if not is_b1_anc:
            results["all_passed"] = False
            results["failures"].append("Tag 'b1-frozen' is not an ancestor of HEAD.")

    # 3. Check tag registration-final
    reg_commit = get_tag_commit("registration-final", cwd=cwd)
    results["registration_final_commit"] = reg_commit
    if not reg_commit:
        results["all_passed"] = False
        results["ancestors"]["registration-final"] = False
        results["failures"].append(
            "Tag 'registration-final' does not exist in repository. (Expected user-created pre-registration tag)"
        )
    else:
        is_reg_anc = is_ancestor("registration-final", cwd=cwd)
        results["ancestors"]["registration-final"] = is_reg_anc
        if not is_reg_anc:
            results["all_passed"] = False
            results["failures"].append("Tag 'registration-final' is not an ancestor of HEAD.")

    # 4. Systems vs b1-frozen diff
    if b1_commit:
        diff_systems_b1 = get_git_diff("b1-frozen", "src/recon_agent/systems/", cwd=cwd)
        systems_b1_ok = (len(diff_systems_b1) == 0)
        results["diffs"]["systems_vs_b1_frozen"] = systems_b1_ok
        if not systems_b1_ok:
            results["all_passed"] = False
            results["failures"].append(
                f"src/recon_agent/systems/ differs from tag b1-frozen ({len(diff_systems_b1)} bytes diff)"
            )
    else:
        results["diffs"]["systems_vs_b1_frozen"] = False

    # 5. File diffs vs registration-final
    if reg_commit:
        # independent_checker.py
        diff_checker = get_git_diff("registration-final", "src/recon_agent/eval/independent_checker.py", cwd=cwd)
        checker_ok = (len(diff_checker) == 0)
        results["diffs"]["checker_vs_registration_final"] = checker_ok
        if not checker_ok:
            results["all_passed"] = False
            results["failures"].append(
                f"src/recon_agent/eval/independent_checker.py differs from tag registration-final ({len(diff_checker)} bytes diff)"
            )

        # verifier.py
        diff_verifier = get_git_diff("registration-final", "src/recon_agent/llm/verifier.py", cwd=cwd)
        verifier_ok = (len(diff_verifier) == 0)
        results["diffs"]["verifier_vs_registration_final"] = verifier_ok
        if not verifier_ok:
            results["all_passed"] = False
            results["failures"].append(
                f"src/recon_agent/llm/verifier.py differs from tag registration-final ({len(diff_verifier)} bytes diff)"
            )

        # systems/
        diff_systems_reg = get_git_diff("registration-final", "src/recon_agent/systems/", cwd=cwd)
        systems_reg_ok = (len(diff_systems_reg) == 0)
        results["diffs"]["systems_vs_registration_final"] = systems_reg_ok
        if not systems_reg_ok:
            results["all_passed"] = False
            results["failures"].append(
                f"src/recon_agent/systems/ differs from tag registration-final ({len(diff_systems_reg)} bytes diff)"
            )

        # docs/preregistration.md
        diff_prereg = get_git_diff("registration-final", "docs/preregistration.md", cwd=cwd)
        prereg_ok = (len(diff_prereg) == 0)
        results["diffs"]["prereg_doc_vs_registration_final"] = prereg_ok
        if not prereg_ok:
            results["all_passed"] = False
            results["failures"].append(
                f"docs/preregistration.md differs from tag registration-final ({len(diff_prereg)} bytes diff)"
            )
    else:
        results["diffs"]["checker_vs_registration_final"] = False
        results["diffs"]["verifier_vs_registration_final"] = False
        results["diffs"]["systems_vs_registration_final"] = False
        results["diffs"]["prereg_doc_vs_registration_final"] = False

    return results["all_passed"], results


def main() -> None:
    passed, details = check_frozen_baselines()
    reg_commit = details["registration_final_commit"] or "NONE (tag missing)"
    b1_commit = details["b1_frozen_commit"] or "NONE (tag missing)"

    print("=" * 68)
    print("FROZEN BASELINES INTEGRITY CHECK (registration-final)")
    print("=" * 68)
    print(f"Tag b1-frozen         : {b1_commit} {'(ANCESTOR)' if details['ancestors'].get('b1-frozen') else '(NOT ANCESTOR)'}")
    print(f"Tag registration-final: {reg_commit} {'(ANCESTOR)' if details['ancestors'].get('registration-final') else '(NOT ANCESTOR)'}")
    print("-" * 68)
    print("Frozen File Hashes (HEAD):")
    f_hashes = details["file_hashes"]
    print(f"  independent_checker.py: {f_hashes.get('src/recon_agent/eval/independent_checker.py')}")
    print(f"  verifier.py           : {f_hashes.get('src/recon_agent/llm/verifier.py')}")
    print(f"  docs/preregistration.md: {f_hashes.get('docs/preregistration.md')}")
    print(f"  systems/ (tree)       : {f_hashes.get('src/recon_agent/systems/')}")
    for s_path, s_hash in sorted(f_hashes.get("systems_files", {}).items()):
        print(f"    {s_path}: {s_hash}")
    print("-" * 68)
    print("Diffs vs registration-final:")
    print(f"  independent_checker.py: {'PASS (0 diff)' if details['diffs'].get('checker_vs_registration_final') else 'FAIL'}")
    print(f"  verifier.py           : {'PASS (0 diff)' if details['diffs'].get('verifier_vs_registration_final') else 'FAIL'}")
    print(f"  docs/preregistration.md: {'PASS (0 diff)' if details['diffs'].get('prereg_doc_vs_registration_final') else 'FAIL'}")
    print(f"  systems/              : {'PASS (0 diff)' if details['diffs'].get('systems_vs_registration_final') else 'FAIL'}")
    print("Diffs vs b1-frozen:")
    print(f"  systems/              : {'PASS (0 diff)' if details['diffs'].get('systems_vs_b1_frozen') else 'FAIL'}")
    print("=" * 68)

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
