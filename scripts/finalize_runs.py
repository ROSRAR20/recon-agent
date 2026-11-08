"""Read cached LLM outputs for a population and run_id, finalizing B2 and B3 results.

Outputs are written atomically to results/finalized/<verifier_version>/finalized_<population>_run<run_id>.jsonl
keyed by (bundle_id, run_id). Never modifies or overwrites the LLM cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.llm.bundle import CaseBundle, build_case_bundles
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.finalize import (
    FinalDecision,
    compute_verifier_version,
    finalize,
    write_finalized_results,
)
from src.recon_agent.llm.gemini_client import get_schema_hash, load_model_config
from src.recon_agent.pipeline.loader import load_dataset


def finalize_run(
    population: str = "dev",
    run_id: int = 0,
    model_id: str | None = None,
    thinking_level: str | None = "low",
    verifier_version: str | None = None,
    cache_dir: str = "results/llm_cache",
    output_dir: str = "results/finalized",
    system_prompt_path: str = "src/recon_agent/llm/prompts/system.txt",
    data_dir: str | None = None,
) -> str:
    """Finalize all cached cases for a given population and run_id."""
    model_cfg = load_model_config()
    resolved_model_id = model_id or model_cfg.get("primary", model_cfg.get("primary_model", "gemini-3.5-flash-lite"))
    resolved_verifier_version = verifier_version or compute_verifier_version()

    # Load system prompt for prompt hash
    sys_prompt_full_path = os.path.join(REPO_ROOT, system_prompt_path)
    with open(sys_prompt_full_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    schema_hash = get_schema_hash()

    # Determine data directory for population
    if data_dir is None:
        candidate_paths = [
            os.path.join(REPO_ROOT, "data", "synth", "small", population),
            os.path.join(REPO_ROOT, "data", "synth", "full", f"{population}_seed1"),
        ]
        resolved_data_dir = None
        for p in candidate_paths:
            if os.path.isdir(p):
                resolved_data_dir = p
                break
        if resolved_data_dir is None:
            raise FileNotFoundError(f"Could not find dataset directory for population '{population}'")
    else:
        resolved_data_dir = os.path.abspath(data_dir)

    dataset = load_dataset(resolved_data_dir)
    bundles = build_case_bundles(dataset)
    print(f"Loaded {len(bundles)} case bundles for population '{population}' from {resolved_data_dir}")

    resolved_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(REPO_ROOT, cache_dir)
    cache = LLMCache(cache_dir=resolved_cache_dir)

    decisions: list[FinalDecision] = []
    missing_cache = 0

    for bundle in bundles:
        entry = cache.get_entry(
            bundle_hash=bundle.bundle_hash,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            model_id=resolved_model_id,
            thinking_level=thinking_level,
            run_id=run_id,
        )
        if entry is None:
            missing_cache += 1
            # Empty/missing output treated as empty_output model failure
            dec = finalize(
                bundle=bundle,
                raw_text=None,
                verifier_version=resolved_verifier_version,
                run_id=run_id,
            )
        else:
            raw_text = entry.get("raw_output")
            dec = finalize(
                bundle=bundle,
                raw_text=raw_text,
                verifier_version=resolved_verifier_version,
                run_id=run_id,
            )
        decisions.append(dec)

    if missing_cache > 0:
        print(f"[WARNING] {missing_cache}/{len(bundles)} cases missing from cache (finalized as empty_output)")

    resolved_output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(REPO_ROOT, output_dir)
    out_path = write_finalized_results(
        decisions=decisions,
        verifier_version=resolved_verifier_version,
        run_id=run_id,
        population=population,
        base_dir=resolved_output_dir,
    )

    # Print summary
    b2_counts: dict[str, int] = {}
    b3_counts: dict[str, int] = {}
    ver_counts: dict[str, int] = {}
    for d in decisions:
        b2_counts[d.b2_action] = b2_counts.get(d.b2_action, 0) + 1
        b3_counts[d.b3_action] = b3_counts.get(d.b3_action, 0) + 1
        v_out = d.verifier_result.outcome.value if d.verifier_result else d.raw_parse_status
        ver_counts[v_out] = ver_counts.get(v_out, 0) + 1

    print("\n[FINALIZATION SUMMARY]")
    print(f"Verifier Version : {resolved_verifier_version}")
    print(f"Output File      : {out_path}")
    print(f"Total Cases      : {len(decisions)}")
    print(f"B2 Actions       : {b2_counts}")
    print(f"B3 Actions       : {b3_counts}")
    print(f"Verifier Outcomes: {ver_counts}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize cached LLM outputs into B2 and B3 results")
    parser.add_argument("--population", default="dev", help="Dataset population name (default: dev)")
    parser.add_argument("--run-id", type=int, default=0, help="Run ID (default: 0)")
    parser.add_argument("--model", default=None, help="Model ID")
    parser.add_argument("--thinking-level", default="low", help="Thinking level (default: low)")
    parser.add_argument("--verifier-version", default=None, help="Verifier version override")
    parser.add_argument("--data-dir", default=None, help="Explicit path to data directory")
    args = parser.parse_args()

    finalize_run(
        population=args.population,
        run_id=args.run_id,
        model_id=args.model,
        thinking_level=args.thinking_level,
        verifier_version=args.verifier_version,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
