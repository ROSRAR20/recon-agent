"""Export finalized raw LLM outputs to compressed archives in artifacts/.

Compresses raw outputs for specified run IDs to artifacts/ as .jsonl.gz
with SHA-256 checksums and a manifest.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.check_env import get_git_state
from scripts.startup_checks import ensure_startup_checks, load_freeze_config


def compute_file_sha256(path: str) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_artifacts(
    run_ids: tuple[int, ...] = (0, 1, 2),
    results_dir: str = "results",
    artifacts_dir: str = "artifacts",
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Export raw run outputs into gzipped jsonl files in artifacts/ with manifest.json."""
    ensure_startup_checks(repo_root=REPO_ROOT, allow_dirty=allow_dirty, verify_hashes=False)
    freeze_cfg = load_freeze_config(repo_root=REPO_ROOT)
    git_commit, is_dirty = get_git_state()

    resolved_res_dir = os.path.join(REPO_ROOT, results_dir)
    resolved_art_dir = os.path.join(REPO_ROOT, artifacts_dir)
    os.makedirs(resolved_art_dir, exist_ok=True)

    exported_files: list[dict[str, Any]] = []

    # 1. Export main run states
    scopes = ["full", "consistency"]
    for r_id in run_ids:
        for scope in scopes:
            state_file = os.path.join(resolved_res_dir, f"main_run_state_run{r_id}_{scope}.json")
            if not os.path.isfile(state_file):
                continue

            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            cases = state_data.get("cases", {})
            if not cases:
                continue

            archive_filename = f"main_run{r_id}_{scope}.jsonl.gz"
            archive_path = os.path.join(resolved_art_dir, archive_filename)

            # Write .jsonl.gz
            with gzip.open(archive_path, "wt", encoding="utf-8") as gz:
                for bid in sorted(cases.keys()):
                    case_payload = {
                        "run_id": r_id,
                        "scope": scope,
                        **cases[bid],
                    }
                    gz.write(json.dumps(case_payload, ensure_ascii=False) + "\n")

            sha256_hash = compute_file_sha256(archive_path)
            file_size = os.path.getsize(archive_path)

            exported_files.append({
                "archive_file": archive_filename,
                "type": "main_run",
                "run_id": r_id,
                "scope": scope,
                "case_count": len(cases),
                "sha256": sha256_hash,
                "size_bytes": file_size,
            })
            print(f"[EXPORT] Created {archive_filename} ({len(cases)} cases, SHA-256: {sha256_hash[:16]}...)")

    # 2. Export twin run states
    for s in (2, 3):
        for r_id in run_ids:
            twin_file = os.path.join(resolved_res_dir, f"twin_run_state_seed{s}_run{r_id}.json")
            if not os.path.isfile(twin_file):
                continue

            with open(twin_file, "r", encoding="utf-8") as f:
                t_data = json.load(f)

            t_cases = t_data.get("cases", {})
            if not t_cases:
                continue

            t_archive_filename = f"twin_seed{s}_run{r_id}.jsonl.gz"
            t_archive_path = os.path.join(resolved_art_dir, t_archive_filename)

            with gzip.open(t_archive_path, "wt", encoding="utf-8") as gz:
                for bid in sorted(t_cases.keys()):
                    t_payload = {
                        "run_id": r_id,
                        "seed": s,
                        **t_cases[bid],
                    }
                    gz.write(json.dumps(t_payload, ensure_ascii=False) + "\n")

            t_sha256 = compute_file_sha256(t_archive_path)
            t_size = os.path.getsize(t_archive_path)

            exported_files.append({
                "archive_file": t_archive_filename,
                "type": "twin_run",
                "run_id": r_id,
                "seed": s,
                "case_count": len(t_cases),
                "sha256": t_sha256,
                "size_bytes": t_size,
            })
            print(f"[EXPORT] Created {t_archive_filename} ({len(t_cases)} cases, SHA-256: {t_sha256[:16]}...)")

    # 3. Write manifest.json
    manifest_path = os.path.join(resolved_art_dir, "manifest.json")
    manifest_payload = {
        "git_commit": git_commit,
        "git_dirty": is_dirty,
        "prompt_hash": freeze_cfg.get("prompt_hash"),
        "verifier_version": freeze_cfg.get("verifier_version"),
        "model": freeze_cfg.get("model"),
        "archives": exported_files,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    print(f"\n[EXPORT COMPLETE] Manifest written to {manifest_path} ({len(exported_files)} archives).")
    return manifest_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export finalized raw LLM outputs to artifacts/")
    parser.add_argument("--run-ids", default="0,1,2", help="Comma-separated run IDs to export (default: 0,1,2)")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree for test fixtures")
    args = parser.parse_args()

    r_ids = tuple(int(r.strip()) for r in args.run_ids.split(",") if r.strip())
    export_artifacts(
        run_ids=r_ids,
        results_dir=args.results_dir,
        artifacts_dir=args.artifacts_dir,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
