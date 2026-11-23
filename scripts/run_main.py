"""Runner for Day 8 main evaluation runs (Run 0: Full population, Runs 1 & 2: Consistency subset).

Reads frozen configuration from config/llm_freeze.json.
Enforces startup checks, quota tracking, resumability, and raw outputs only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import random
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.build_main_population import build_main_population
from scripts.run_pilot import MockPilotClient
from scripts.startup_checks import (
    ensure_startup_checks,
    load_freeze_config,
    verify_freeze_hashes,
)
from src.recon_agent.guard import sanitize_error
from src.recon_agent.llm.bundle import CaseBundle, create_case_bundle
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.gemini_client import (
    DailyQuotaExhaustedError,
    GeminiClient,
    get_schema_hash,
)
from src.recon_agent.llm.runner import QuotaTracker, Runner


def get_consistency_subset_bundle_ids(
    pop_dir: str = "data/populations",
    target_count: int = 200,
    seed: int = 42,
) -> list[str]:
    """Deterministically select a fixed stratified 200-bundle subset from main_population."""
    resolved_pop_dir = os.path.join(REPO_ROOT, pop_dir)
    pop_file = os.path.join(resolved_pop_dir, "main_population.jsonl")
    meta_file = os.path.join(resolved_pop_dir, "main_meta.jsonl")

    if not os.path.isfile(pop_file) or not os.path.isfile(meta_file):
        build_main_population(pop_dir=pop_dir, allow_dirty=True)

    with open(pop_file, "r", encoding="utf-8") as f_pop, open(meta_file, "r", encoding="utf-8") as f_meta:
        bundles = [json.loads(line) for line in f_pop if line.strip()]
        metas = [json.loads(line) for line in f_meta if line.strip()]

    # Stratify by (check_family, is_control)
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for b, m in zip(bundles, metas):
        bid = b["bundle_id"]
        fam = b["check_family"]
        is_ctrl = m.get("is_control", False)
        strata[(fam, is_ctrl)].append(bid)

    for k in strata:
        strata[k].sort()

    total_pop = len(bundles)
    selected_bids: list[str] = []
    rng = random.Random(seed)

    # Allocate counts proportionally
    allocated: dict[tuple[str, bool], int] = {}
    current_total = 0
    for k, bids in strata.items():
        share = len(bids) / total_pop
        count = max(1, round(share * target_count))
        allocated[k] = min(count, len(bids))
        current_total += allocated[k]

    # Adjust rounding differences
    diff = target_count - current_total
    strata_keys = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)
    idx = 0
    while diff != 0:
        k = strata_keys[idx % len(strata_keys)]
        if diff > 0 and allocated[k] < len(strata[k]):
            allocated[k] += 1
            diff -= 1
        elif diff < 0 and allocated[k] > 1:
            allocated[k] -= 1
            diff += 1
        idx += 1

    for k in sorted(strata.keys()):
        bids = list(strata[k])
        # Deterministic sample
        sampled = rng.sample(bids, allocated[k])
        selected_bids.extend(sampled)

    # Sort selected bundle IDs to preserve population ordering
    pop_order = {b["bundle_id"]: i for i, b in enumerate(bundles)}
    selected_bids.sort(key=lambda x: pop_order.get(x, 0))

    if len(selected_bids) != target_count:
        raise ValueError(f"Expected {target_count} bundles for consistency subset, got {len(selected_bids)}")
    return selected_bids


def load_main_bundles(
    scope: str = "full",
    pop_dir: str = "data/populations",
    allow_dirty: bool = False,
) -> tuple[list[CaseBundle], dict[str, dict[str, Any]]]:
    """Load CaseBundle objects and metadata for the requested scope."""
    resolved_pop_dir = os.path.join(REPO_ROOT, pop_dir)
    pop_file = os.path.join(resolved_pop_dir, "main_population.jsonl")
    meta_file = os.path.join(resolved_pop_dir, "main_meta.jsonl")

    if not os.path.isfile(pop_file) or not os.path.isfile(meta_file):
        build_main_population(pop_dir=pop_dir, allow_dirty=allow_dirty)

    with open(pop_file, "r", encoding="utf-8") as f_pop, open(meta_file, "r", encoding="utf-8") as f_meta:
        raw_bundles = [json.loads(line) for line in f_pop if line.strip()]
        raw_metas = [json.loads(line) for line in f_meta if line.strip()]

    meta_lookup = {m["bundle_id"]: m for m in raw_metas}

    if scope == "consistency":
        subset_ids = set(get_consistency_subset_bundle_ids(pop_dir=pop_dir))
        raw_bundles = [b for b in raw_bundles if b["bundle_id"] in subset_ids]

    case_bundles: list[CaseBundle] = []
    for b in raw_bundles:
        bundle = create_case_bundle(
            bundle_id=b["bundle_id"],
            entity_id=b["entity_id"],
            merchant_id=b["merchant_id"],
            check_family=b["check_family"],
            facts=b.get("facts", {}),
            untrusted_text=b.get("untrusted_text", {}),
        )
        case_bundles.append(bundle)

    return case_bundles, meta_lookup


def run_main(
    run_id: int = 0,
    scope: str = "full",
    resume: bool = False,
    max_requests_per_day: int = 450,
    mock: bool = False,
    allow_dirty: bool = False,
    pop_dir: str = "data/populations",
    output_dir: str = "results",
) -> dict[str, Any]:
    """Execute main evaluation run."""
    # 0. Startup integrity checks
    startup_info = ensure_startup_checks(repo_root=REPO_ROOT, allow_dirty=allow_dirty, verify_hashes=True)
    freeze_cfg = startup_info["freeze_config"]

    resolved_model_id = "mock" if mock else freeze_cfg["model"]
    thinking_level = freeze_cfg["thinking_level"]
    arm_name = freeze_cfg.get("arm", "default")

    if not mock and resolved_model_id == "mock":
        raise ValueError("Real execution refused with model id 'mock'.")

    # Load system prompt
    prompt_path = os.path.join(REPO_ROOT, "src", "recon_agent", "llm", "prompts", "system.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Load bundles
    bundles, meta_lookup = load_main_bundles(scope=scope, pop_dir=pop_dir, allow_dirty=allow_dirty)
    print(f"\n[MAIN RUN] Loaded {len(bundles)} bundles for scope='{scope}' (run_id={run_id})")

    # Setup Cache & Client
    resolved_out_dir = os.path.join(REPO_ROOT, output_dir)
    os.makedirs(resolved_out_dir, exist_ok=True)
    cache_dir = os.path.join(resolved_out_dir, "llm_cache_mock" if mock else "llm_cache")
    cache = LLMCache(cache_dir=cache_dir)

    if mock:
        client = MockPilotClient(
            model_id=resolved_model_id,
            thinking_level=thinking_level,
            cache=cache,
            arm_name=arm_name,
            meta_lookup=meta_lookup,
        )
    else:
        client = GeminiClient(
            model_id=resolved_model_id,
            thinking_level=thinking_level,
            cache=cache,
        )

    # Setup Quota Tracker & State Path
    quota_path = os.path.join(resolved_out_dir, "quota_state.json")
    quota_tracker = QuotaTracker(quota_state_path=quota_path, max_requests_per_day=max_requests_per_day)

    run_state_path = os.path.join(resolved_out_dir, f"main_run_state_run{run_id}_{scope}.json")
    runner = Runner(
        client=client,
        quota_tracker=quota_tracker,
        system_prompt=system_prompt,
        run_state_path=run_state_path,
    )

    print(f"[MAIN RUN] Starting execution for run_id={run_id}, scope={scope}, model={resolved_model_id} (mock={mock})...")
    results = runner.run_cases(bundles, resume=resume, run_id=run_id)

    # Update state with is_mock and extra metadata
    n_ok = sum(1 for c in results.values() if c.status == "ok")
    n_infra = sum(1 for c in results.values() if c.status == "infra_failed")
    n_pending = sum(1 for c in results.values() if c.status == "pending")

    try:
        with open(run_state_path, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        state_data["is_mock"] = mock
        state_data["scope"] = scope
        state_data["run_id"] = run_id
        state_data["prompt_hash"] = freeze_cfg["prompt_hash"]
        state_data["verifier_version"] = freeze_cfg["verifier_version"]
        with open(run_state_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)
    except Exception:
        pass

    print("====================================================================")
    print(f"[MAIN RUN] Summary: run_id={run_id}, scope={scope}")
    print(f"  Total Cases : {len(bundles)}")
    print(f"  OK          : {n_ok}")
    print(f"  Infra Failed: {n_infra}")
    print(f"  Pending     : {n_pending}")
    print(f"  State File  : {run_state_path}")
    print("====================================================================")

    mock_flag = " --mock" if mock else ""
    dirty_flag = " --allow-dirty" if allow_dirty else ""
    resume_cmd = f"python scripts/run_main.py --run-id {run_id} --scope {scope} --resume{mock_flag}{dirty_flag}"
    print(f"Resume command : {resume_cmd}\n")

    return {
        "status": "completed" if n_pending == 0 and n_infra == 0 else "partial",
        "run_id": run_id,
        "scope": scope,
        "counts": {"ok": n_ok, "infra_failed": n_infra, "pending": n_pending, "total": len(bundles)},
        "run_state_path": run_state_path,
        "resume_command": resume_cmd,
        "is_mock": mock,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main evaluation runs")
    parser.add_argument("--run-id", type=int, default=0, help="Run ID (0=full, 1|2=consistency)")
    parser.add_argument("--scope", choices=["full", "consistency"], default=None, help="Evaluation scope")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    parser.add_argument("--max-requests-per-day", type=int, default=450, help="Daily request quota limit (default: 450)")
    parser.add_argument("--mock", action="store_true", help="Use MockClient instead of real Gemini API")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree for test fixtures")
    parser.add_argument("--pop-dir", default="data/populations", help="Populations directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    # Default scope based on run-id if omitted
    resolved_scope = args.scope
    if resolved_scope is None:
        resolved_scope = "full" if args.run_id == 0 else "consistency"

    run_main(
        run_id=args.run_id,
        scope=resolved_scope,
        resume=args.resume,
        max_requests_per_day=args.max_requests_per_day,
        mock=args.mock,
        allow_dirty=args.allow_dirty,
        pop_dir=args.pop_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
