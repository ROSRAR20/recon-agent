"""Startup integrity checks required for Day 8 main-run tooling.

Refuses execution unless:
1. config/llm_freeze.json exists and is valid.
2. Git tag llm-frozen exists and is an ancestor of HEAD (or exists in repo).
3. scripts/check_frozen.py check_frozen_baselines() passes.
4. Git working tree is clean (unless explicitly bypassed via allow_dirty).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.check_frozen import check_frozen_baselines, get_tag_commit


def load_freeze_config(repo_root: str = REPO_ROOT) -> dict[str, Any]:
    """Load and validate config/llm_freeze.json."""
    freeze_path = os.path.join(repo_root, "config", "llm_freeze.json")
    if not os.path.isfile(freeze_path):
        raise RuntimeError(f"Startup check failed: Freeze configuration missing at {freeze_path}")
    try:
        with open(freeze_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Startup check failed: Cannot parse {freeze_path}: {exc}")

    required_keys = {"model", "arm", "thinking_level", "prompt_hash", "verifier_version"}
    missing = required_keys - set(data.keys())
    if missing:
        raise RuntimeError(f"Startup check failed: {freeze_path} missing required keys: {sorted(missing)}")
    return data


def verify_freeze_hashes(freeze_config: dict[str, Any], repo_root: str = REPO_ROOT) -> None:
    """Verify that current system prompt hash and verifier version match llm_freeze.json."""
    prompt_file = os.path.join(repo_root, "src", "recon_agent", "llm", "prompts", "system.txt")
    if not os.path.isfile(prompt_file):
        raise RuntimeError(f"System prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        actual_prompt_hash = hashlib.sha256(f.read().encode("utf-8")).hexdigest()

    expected_prompt_hash = freeze_config["prompt_hash"]
    if actual_prompt_hash != expected_prompt_hash:
        raise RuntimeError(
            f"Prompt hash mismatch with freeze: current={actual_prompt_hash}, frozen={expected_prompt_hash}"
        )

    # Verifier version check
    try:
        from src.recon_agent.llm.finalize import compute_verifier_version
        actual_verifier_version = compute_verifier_version()
    except Exception as exc:
        raise RuntimeError(f"Cannot compute verifier version: {exc}")

    expected_verifier_version = freeze_config["verifier_version"]
    if actual_verifier_version != expected_verifier_version:
        raise RuntimeError(
            f"Verifier version mismatch with freeze: current={actual_verifier_version}, frozen={expected_verifier_version}"
        )


def check_git_clean(repo_root: str = REPO_ROOT) -> bool:
    """Check if the git working tree is clean."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return len(out) == 0
    except Exception:
        return False


def ensure_startup_checks(
    repo_root: str = REPO_ROOT,
    allow_dirty: bool = False,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Execute all mandatory startup checks. Refuses (raises RuntimeError) on any failure."""
    errors = []

    # 1. config/llm_freeze.json exists
    freeze_cfg = None
    try:
        freeze_cfg = load_freeze_config(repo_root=repo_root)
    except Exception as exc:
        errors.append(str(exc))

    # 2. Tag llm-frozen exists
    llm_frozen_commit = get_tag_commit("llm-frozen", cwd=repo_root)
    if not llm_frozen_commit:
        errors.append("Startup check failed: Tag 'llm-frozen' does not exist in repository.")

    # 3. check_frozen.py passes
    passed, frozen_info = check_frozen_baselines(repo_root=repo_root)
    if not passed:
        failures = "; ".join(frozen_info.get("failures", []))
        errors.append(f"Startup check failed: check_frozen baselines failed: {failures}")

    # 4. Git working tree is clean
    if not allow_dirty:
        if not check_git_clean(repo_root=repo_root):
            errors.append(
                "Startup check failed: Git working tree is dirty. "
                "Commit or stash changes before running main tooling."
            )

    # 5. Hashes match freeze
    if verify_hashes and freeze_cfg is not None:
        try:
            verify_freeze_hashes(freeze_cfg, repo_root=repo_root)
        except Exception as exc:
            errors.append(str(exc))

    if errors:
        msg = "\n  * " + "\n  * ".join(errors)
        raise RuntimeError(f"STARTUP CHECKS FAILED:{msg}")

    return {
        "freeze_config": freeze_cfg,
        "llm_frozen_commit": llm_frozen_commit,
        "frozen_info": frozen_info,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Day 8 startup checks")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty git working tree")
    args = parser.parse_args()

    try:
        info = ensure_startup_checks(allow_dirty=args.allow_dirty)
        print("STARTUP CHECKS: PASS")
        print(f"  Frozen tag commit: {info['llm_frozen_commit']}")
        print(f"  Frozen model: {info['freeze_config']['model']}")
        print(f"  Frozen arm: {info['freeze_config']['arm']}")
        print(f"  Prompt hash: {info['freeze_config']['prompt_hash'][:16]}...")
        print(f"  Verifier version: {info['freeze_config']['verifier_version'][:16]}...")
        sys.exit(0)
    except RuntimeError as exc:
        print(f"STARTUP CHECKS: FAIL\n{exc}", file=sys.stderr)
        sys.exit(1)
