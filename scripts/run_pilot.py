"""Runner for Day 7b pilot evaluation across model thinking levels.

Executes stratified prefix (e.g. 100 residual + 20 control) for specified arms
('default', 'low', 'high') collecting RAW outputs only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.guard import sanitize_error
from scripts.check_env import check_env, check_tzdata
from scripts.check_frozen import check_frozen_baselines
from src.recon_agent.llm.bundle import CaseBundle, create_case_bundle
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.gemini_client import (
    DailyQuotaExhaustedError,
    GeminiClient,
    InfraError,
    NonRetriableAPIError,
    get_schema_hash,
    load_model_config,
    load_price_rates,
)
from src.recon_agent.llm.runner import CaseResult, QuotaTracker, Runner


class MockPilotClient:
    """Mock Gemini client simulating faithful, hallucinated, unsafe, and malformed outputs."""

    def __init__(
        self,
        model_id: str = "mock",
        thinking_level: str | None = None,
        cache: LLMCache | None = None,
        arm_name: str = "default",
        meta_lookup: dict[str, dict[str, Any]] | None = None,
        tier: str = "free",
        prices_path: str = "config/llm_prices.json",
    ) -> None:
        self.model_id = model_id
        self.thinking_level = thinking_level
        self.cache = cache
        self.arm_name = arm_name
        self.meta_lookup = meta_lookup or {}
        self.tier = tier
        self.prices_path = prices_path
        self.call_count = 0

        # Telemetry
        self.last_tokens: dict[str, int | None] = {}
        self.last_cost_low: float = 0.0
        self.last_cost_high: float = 0.0
        self.last_latency_ms: float = 0.0
        self.last_served_model: str = model_id
        self.last_service_tier: str = tier

    def complete(self, bundle: CaseBundle, system_prompt: str, run_id: int = 0) -> str:
        """Produce mock raw LLM output and populate cache/telemetry."""
        self.call_count += 1

        # Simulate rejection of unsupported thinking_level parameter (HTTP 400)
        if self.thinking_level == "unsupported":
            raise NonRetriableAPIError(
                "Non-retriable Gemini API HTTP 400 Bad Request: Invalid argument: "
                "thinking_level is an unsupported parameter for this model."
            )

        bundle_hash = bundle.bundle_hash
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        schema_hash = get_schema_hash()

        # Check cache
        if self.cache is not None:
            cached_entry = self.cache.get_entry(
                bundle_hash=bundle_hash,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                model_id=self.model_id,
                thinking_level=self.thinking_level,
                run_id=run_id,
            )
            if cached_entry is not None and cached_entry.get("raw_output") is not None:
                self.last_tokens = cached_entry.get("tokens") or {
                    "input_tokens": 120,
                    "output_tokens": 60,
                    "thinking_tokens": 0,
                }
                self.last_cost_low = cached_entry.get("cost_low", 0.0)
                self.last_cost_high = cached_entry.get("cost_high", 0.0)
                self.last_served_model = cached_entry.get("served_model", self.model_id)
                self.last_service_tier = self.tier
                self.last_latency_ms = 400.0 if self.thinking_level is None else (700.0 if self.thinking_level == "low" else 1200.0)
                return cached_entry["raw_output"]

        # Deterministic generation based on bundle_id and arm
        meta = self.meta_lookup.get(bundle.bundle_id, {})
        source_type = meta.get("source_label_type", "")
        expected_dec = meta.get("expected_decision", "no_issue")
        is_ctrl = meta.get("is_control", False)

        bid_hash_int = int(hashlib.sha256(f"{bundle.bundle_id}:{self.arm_name}".encode("utf-8")).hexdigest(), 16)
        mod_val = bid_hash_int % 100

        # Establish arm behavior:
        # default: 2 malformed, 4 hallucinated, 1 reflection, 2 unsafe
        # low: 1 malformed, 3 hallucinated, 1 reflection, 1 unsafe
        # high: 1 malformed, 2 hallucinated, 0 reflection, 0 unsafe
        raw_output_text: str

        if self.arm_name == "default" and mod_val == 3:
            raw_output_text = "I reviewed the case and recommend escalating to a human operator."
        elif mod_val == 7:
            raw_output_text = json.dumps({"decision": "invalid_choice", "predicted_class": None, "explanation": "invalid"})
        elif mod_val == 15 and self.arm_name in ("default", "low", "high"):
            # F1 hallucination: ungrounded evidence
            raw_output_text = json.dumps({
                "decision": "escalate_human",
                "predicted_class": None,
                "reason_code": "NEAR_TOLERANCE",
                "explanation": "Delta requires escalation.",
                "evidence": [{"field": "fabricated_evidence_field", "value": 999}],
                "cited_amount_paise": None,
                "verbatim_quote": None,
            })
        elif mod_val == 25 and self.arm_name in ("default", "low", "high"):
            # F2 hallucination: fabricated quote
            raw_output_text = json.dumps({
                "decision": "escalate_human",
                "predicted_class": None,
                "reason_code": "AMBIGUOUS_TEXT",
                "explanation": "Customer service promised a special concession.",
                "evidence": [],
                "cited_amount_paise": None,
                "verbatim_quote": "Customer service promised a special concession.",
            })
        elif mod_val == 35 and self.arm_name in ("default", "low"):
            # F3 hallucination: ungrounded cited amount
            raw_output_text = json.dumps({
                "decision": "escalate_human",
                "predicted_class": None,
                "reason_code": "NEAR_TOLERANCE",
                "explanation": "Discrepancy noted.",
                "evidence": [],
                "cited_amount_paise": 888888,
                "verbatim_quote": None,
            })
        elif mod_val == 45 and self.arm_name == "default":
            # F4 hallucination: ungrounded explanation number
            raw_output_text = json.dumps({
                "decision": "escalate_human",
                "predicted_class": None,
                "reason_code": "NEAR_TOLERANCE",
                "explanation": "Observed unregistered fee discrepancy of 77777 paise.",
                "evidence": [],
                "cited_amount_paise": None,
                "verbatim_quote": None,
            })
        elif mod_val == 55 and self.arm_name in ("default", "low") and bundle.untrusted_text:
            # Reflection of >= 6 words from untrusted text
            untrusted_val = next((v for v in bundle.untrusted_text.values() if isinstance(v, str) and len(v.split()) >= 6), None)
            if untrusted_val:
                words = untrusted_val.split()[:7]
                refl_phrase = " ".join(words)
                raw_output_text = json.dumps({
                    "decision": "escalate_human",
                    "predicted_class": None,
                    "reason_code": "AMBIGUOUS_TEXT",
                    "explanation": f"Transaction note states {refl_phrase} which requires review.",
                    "evidence": [],
                    "cited_amount_paise": None,
                    "verbatim_quote": None,
                })
            else:
                raw_output_text = self._generate_faithful_output(bundle, source_type, expected_dec, is_ctrl)
        elif mod_val == 65 and self.arm_name in ("default", "low") and source_type == "EXPLAINED_INCONSISTENT":
            # Unsafe resolution: outputs benign_explained / no_issue where ground truth is flag
            raw_output_text = json.dumps({
                "decision": "no_issue",
                "predicted_class": None,
                "reason_code": "NO_DISCREPANCY",
                "explanation": "No issue detected upon automated review.",
                "evidence": [],
                "cited_amount_paise": None,
                "verbatim_quote": None,
            })
        elif mod_val == 75 and self.arm_name == "default" and source_type == "NEAR_TOLERANCE":
            # Unsafe resolution: outputs no_issue where ground truth is escalate_human
            raw_output_text = json.dumps({
                "decision": "no_issue",
                "predicted_class": None,
                "reason_code": "NO_DISCREPANCY",
                "explanation": "Difference is minor; resolving as no_issue.",
                "evidence": [],
                "cited_amount_paise": None,
                "verbatim_quote": None,
            })
        else:
            # Faithful output
            raw_output_text = self._generate_faithful_output(bundle, source_type, expected_dec, is_ctrl)

        # Telemetry & Cost calculation
        base_latency = 400.0 if self.thinking_level is None else (650.0 if self.thinking_level == "low" else 1150.0)
        self.last_latency_ms = round(base_latency + (bid_hash_int % 100) - 50, 1)

        in_tok = 150 + (bid_hash_int % 30)
        out_tok = 60 + (bid_hash_int % 20)
        think_tok = 0 if self.thinking_level is None else (40 if self.thinking_level == "low" else 100)

        self.last_tokens = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "thinking_tokens": think_tok,
        }
        if self.model_id == "mock":
            rates = {
                "input_cost_per_million": 0.075,
                "output_cost_per_million": 0.30,
                "thinking_cost_per_million": 0.30,
            }
        else:
            try:
                rates = load_price_rates(self.prices_path, self.model_id)
            except Exception:
                rates = {
                    "input_cost_per_million": 0.075,
                    "output_cost_per_million": 0.30,
                    "thinking_cost_per_million": 0.30,
                }
        cost = (
            in_tok * rates.get("input_cost_per_million", 0.0)
            + out_tok * rates.get("output_cost_per_million", 0.0)
            + think_tok * rates.get("thinking_cost_per_million", 0.0)
        ) / 1_000_000.0

        self.last_cost_low = round(cost, 6)
        self.last_cost_high = round(cost, 6)
        self.last_served_model = self.model_id
        self.last_service_tier = self.tier

        # Write to cache
        if self.cache is not None:
            self.cache.put(
                bundle_hash=bundle_hash,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                model_id=self.model_id,
                thinking_level=self.thinking_level,
                run_id=run_id,
                raw_output=raw_output_text,
                served_model=self.model_id,
                tokens=self.last_tokens,
                cost_low=self.last_cost_low,
                cost_high=self.last_cost_high,
                usage_raw=self.last_tokens,
                raw_response_bytes=len(raw_output_text.encode("utf-8")),
            )

        return raw_output_text

    def _generate_faithful_output(
        self,
        bundle: CaseBundle,
        source_type: str,
        expected_dec: str,
        is_ctrl: bool,
    ) -> str:
        facts = bundle.facts

        if is_ctrl or expected_dec == "no_issue":
            return json.dumps({
                "decision": "no_issue",
                "predicted_class": None,
                "reason_code": "CONTROL_CLEAN",
                "explanation": "All observed facts match expected state; no discrepancies.",
                "evidence": [],
                "cited_amount_paise": None,
                "verbatim_quote": None,
            })

        if bundle.check_family == "fee_tax":
            if source_type == "NEAR_TOLERANCE":
                fee_delta = facts.get("fee_delta_paise", 0)
                tax_delta = facts.get("tax_delta_paise", 0)
                total_delta = abs(fee_delta + tax_delta)
                tol = facts.get("rounding_tolerance_paise", 2)
                return json.dumps({
                    "decision": "escalate_human",
                    "predicted_class": None,
                    "reason_code": "NEAR_TOLERANCE",
                    "explanation": f"Delta of {total_delta} paise exceeds tolerance of {tol} paise.",
                    "evidence": [
                        {"field": "fee_delta_paise", "value": fee_delta},
                        {"field": "tax_delta_paise", "value": tax_delta},
                        {"field": "rounding_tolerance_paise", "value": tol},
                    ],
                    "cited_amount_paise": None,
                    "verbatim_quote": None,
                })
            else:
                return json.dumps({
                    "decision": "no_issue",
                    "predicted_class": None,
                    "reason_code": "NO_DISCREPANCY",
                    "explanation": "No discrepancies found.",
                    "evidence": [],
                    "cited_amount_paise": None,
                    "verbatim_quote": None,
                })

        elif bundle.check_family == "refund":
            if source_type == "EXPLAINED_CONSISTENT":
                reason_text = bundle.untrusted_text.get("recon_item_notes_reason_text") or ""
                # Extract quote pattern like 'Rs. X.XX' or 'Rs X.XX'
                m = re.search(r"(?:Rs\.?|INR)\s*(\d+(?:\.\d+)?)", reason_text, re.IGNORECASE)
                if m:
                    quote_str = m.group(0)
                    quote_val = float(m.group(1))
                    cited_paise = round(quote_val * 100)
                else:
                    quote_str = None
                    cited_paise = abs(facts.get("refund_difference_paise", 0))

                ledger_ref = facts.get("ledger_refund_paise")
                debit_p = facts.get("debit_paise")
                return json.dumps({
                    "decision": "benign_explained",
                    "predicted_class": None,
                    "reason_code": "EXPLAINED_CONSISTENT",
                    "explanation": f"Difference of {cited_paise} paise explained by quote {quote_str}.",
                    "evidence": [
                        {"field": "ledger_refund_paise", "value": ledger_ref},
                        {"field": "debit_paise", "value": debit_p},
                    ],
                    "cited_amount_paise": cited_paise,
                    "verbatim_quote": quote_str,
                })

            elif source_type == "EXPLAINED_INCONSISTENT":
                return json.dumps({
                    "decision": "flag",
                    "predicted_class": "PARTIAL_REFUND_MISMATCH",
                    "reason_code": "PARTIAL_REFUND_MISMATCH",
                    "explanation": "Refund difference not fully covered by reason note amount.",
                    "evidence": [
                        {"field": "ledger_refund_paise", "value": facts.get("ledger_refund_paise")},
                        {"field": "debit_paise", "value": facts.get("debit_paise")},
                    ],
                    "cited_amount_paise": None,
                    "verbatim_quote": None,
                })

            elif source_type == "EXPLAINED_NO_AMOUNT":
                return json.dumps({
                    "decision": "escalate_human",
                    "predicted_class": None,
                    "reason_code": "EXPLAINED_NO_AMOUNT",
                    "explanation": "Reason note lacks monetary amount; human review required.",
                    "evidence": [
                        {"field": "ledger_refund_paise", "value": facts.get("ledger_refund_paise")},
                        {"field": "debit_paise", "value": facts.get("debit_paise")},
                    ],
                    "cited_amount_paise": None,
                    "verbatim_quote": None,
                })

        elif bundle.check_family == "bank":
            if source_type == "bank_ambiguous_candidates":
                return json.dumps({
                    "decision": "escalate_human",
                    "predicted_class": None,
                    "reason_code": "AMBIGUOUS_CANDIDATES",
                    "explanation": "Multiple bank statement candidates matched; escalating for review.",
                    "evidence": [{"field": "candidates_count", "value": facts.get("candidates_count", 2)}],
                    "cited_amount_paise": None,
                    "verbatim_quote": None,
                })

        # Fallback safe escalate
        return json.dumps({
            "decision": "escalate_human",
            "predicted_class": None,
            "reason_code": "AMBIGUOUS_TEXT",
            "explanation": "Escalating for human review.",
            "evidence": [],
            "cited_amount_paise": None,
            "verbatim_quote": None,
        })


def load_stratified_prefix(
    population_file: str,
    meta_file: str,
    n_residual: int = 100,
    n_control: int = 20,
) -> tuple[list[CaseBundle], dict[str, dict[str, Any]]]:
    """Load stratified prefix of canonical bundles preserving interleaved order."""
    if not os.path.isfile(meta_file):
        raise FileNotFoundError(f"Meta file not found: {meta_file}")
    if not os.path.isfile(population_file):
        raise FileNotFoundError(f"Population file not found: {population_file}")

    meta_lookup: dict[str, dict[str, Any]] = {}
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                meta_lookup[row["bundle_id"]] = row

    selected_bundles: list[CaseBundle] = []
    res_count = 0
    ctrl_count = 0

    with open(population_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            bid = data["bundle_id"]
            m = meta_lookup.get(bid)
            if not m:
                continue

            is_ctrl = m.get("is_control", False)
            if is_ctrl and ctrl_count < n_control:
                bundle = create_case_bundle(
                    bundle_id=data["bundle_id"],
                    entity_id=data["entity_id"],
                    merchant_id=data["merchant_id"],
                    check_family=data["check_family"],
                    facts=data.get("facts", {}),
                    untrusted_text=data.get("untrusted_text", {}),
                )
                selected_bundles.append(bundle)
                ctrl_count += 1
            elif not is_ctrl and res_count < n_residual:
                bundle = create_case_bundle(
                    bundle_id=data["bundle_id"],
                    entity_id=data["entity_id"],
                    merchant_id=data["merchant_id"],
                    check_family=data["check_family"],
                    facts=data.get("facts", {}),
                    untrusted_text=data.get("untrusted_text", {}),
                )
                selected_bundles.append(bundle)
                res_count += 1

            if res_count >= n_residual and ctrl_count >= n_control:
                break

    return selected_bundles, meta_lookup


def run_pilot(
    seed: int = 1,
    arms: list[str] | None = None,
    n_residual: int = 100,
    n_control: int = 20,
    max_requests_per_day: int = 450,
    mock: bool = False,
    resume: bool = False,
    population_dir: str = "data/populations",
    output_dir: str = "results",
    system_prompt_path: str = "src/recon_agent/llm/prompts/system.txt",
    model_id: str | None = None,
    run_id: int = 0,
) -> dict[str, Any]:
    """Execute pilot evaluation across arms, collecting raw outputs."""
    check_tzdata()
    frozen_ok, frozen_details = check_frozen_baselines()
    reg_commit = frozen_details.get("registration_final_commit") or "MISSING (tag registration-final not found)"
    frozen_status = "PASS" if frozen_ok else "FAIL: " + "; ".join(frozen_details.get("failures", []))
    f_hashes = frozen_details.get("file_hashes", {})

    print("=" * 80)
    print("PILOT EXECUTION" if not mock else "PILOT EXECUTION (MOCK MODE)")
    print("=" * 80)
    print(f"Registration-Final: {reg_commit}")
    print(f"Frozen Baselines  : {frozen_status}")
    print("Frozen File Hashes (HEAD):")
    print(f"  independent_checker.py: {f_hashes.get('src/recon_agent/eval/independent_checker.py')}")
    print(f"  verifier.py           : {f_hashes.get('src/recon_agent/llm/verifier.py')}")
    print(f"  docs/preregistration.md: {f_hashes.get('docs/preregistration.md')}")
    print(f"  systems/ (tree)       : {f_hashes.get('src/recon_agent/systems/')}")
    print("=" * 80)

    # Real API mode guard: refuse to start unless frozen check passes and git tree is clean
    if not mock:
        if not frozen_ok:
            raise RuntimeError(
                f"Frozen baseline check failed; refusing to run in real API mode: "
                f"{'; '.join(frozen_details.get('failures', []))}"
            )
        env_info = check_env()
        if env_info.git_dirty:
            raise RuntimeError(
                "Git tracked working tree is dirty; refusing to run in real API mode. "
                "Commit or stash tracked changes first."
            )

    arms = arms or ["default", "low", "high"]
    resolved_pop_dir = population_dir if os.path.isabs(population_dir) else os.path.join(REPO_ROOT, population_dir)
    resolved_out_dir = output_dir if os.path.isabs(output_dir) else os.path.join(REPO_ROOT, output_dir)
    os.makedirs(resolved_out_dir, exist_ok=True)

    pop_file = os.path.join(resolved_pop_dir, f"dev_seed{seed}.jsonl")
    meta_file = os.path.join(resolved_pop_dir, f"dev_seed{seed}_meta.jsonl")

    # Load prefix bundles
    bundles, meta_lookup = load_stratified_prefix(
        population_file=pop_file,
        meta_file=meta_file,
        n_residual=n_residual,
        n_control=n_control,
    )
    print(f"[PILOT] Loaded {len(bundles)} prefix bundles (residuals target={n_residual}, controls target={n_control})")

    # Load system prompt
    prompt_full_path = system_prompt_path if os.path.isabs(system_prompt_path) else os.path.join(REPO_ROOT, system_prompt_path)
    with open(prompt_full_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Shared quota tracker: mock mode isolates quota tracking to quota_state_mock.json
    if mock:
        quota_state_path = os.path.join(resolved_out_dir, "quota_state_mock.json")
        if not resume and os.path.isfile(quota_state_path):
            try:
                os.remove(quota_state_path)
            except OSError:
                pass
    else:
        quota_state_path = os.path.join(resolved_out_dir, "quota_state.json")
    quota_tracker = QuotaTracker(quota_state_path=quota_state_path, max_requests_per_day=max_requests_per_day)

    # Cache and model resolution: mock mode strictly uses llm_cache_mock and model 'mock'
    if mock:
        cache_dir = os.path.join(resolved_out_dir, "llm_cache_mock")
        resolved_model_id = model_id or "mock"
    else:
        if model_id == "mock":
            raise ValueError("Model id 'mock' is not allowed in real Gemini API client mode; use MockClient")
        cache_dir = os.path.join(resolved_out_dir, "llm_cache")
        model_cfg = load_model_config()
        resolved_model_id = model_id or model_cfg.get("primary", "gemini-3.5-flash-lite")

    cache = LLMCache(cache_dir=cache_dir)

    arm_thinking_map: dict[str, str | None] = {
        "default": None,
        "low": "low",
        "medium": "medium",
        "high": "high",
    }

    run_summaries: dict[str, Any] = {}

    for arm in arms:
        thinking_level = arm_thinking_map.get(arm, arm)
        print(f"\n[PILOT] Starting execution for arm '{arm}' (thinking_level: {thinking_level})...")

        run_state_path = os.path.join(resolved_out_dir, f"pilot_run_state_{arm}_seed{seed}.json")

        if mock:
            client = MockPilotClient(
                model_id=resolved_model_id,
                thinking_level=thinking_level,
                cache=cache,
                arm_name=arm,
                meta_lookup=meta_lookup,
            )
        else:
            client = GeminiClient(
                model_id=resolved_model_id,
                thinking_level=thinking_level,
                cache=cache,
            )

        runner = Runner(
            client=client,
            quota_tracker=quota_tracker,
            system_prompt=system_prompt,
            run_state_path=run_state_path,
        )

        try:
            results = runner.run_cases(bundles, resume=resume, run_id=run_id)
            n_ok = sum(1 for c in results.values() if c.status == "ok")
            n_infra = sum(1 for c in results.values() if c.status == "infra_failed")
            n_pending = sum(1 for c in results.values() if c.status == "pending")
            print(f"[PILOT] Arm '{arm}' completed: ok={n_ok}, infra_failed={n_infra}, pending={n_pending}")

            # Record mock flag in state file
            try:
                with open(run_state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                state_data["is_mock"] = mock
                with open(run_state_path, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=2)
            except Exception:
                pass

            run_summaries[arm] = {
                "status": "completed",
                "counts": {"ok": n_ok, "infra_failed": n_infra, "pending": n_pending, "total": len(results)},
                "run_state_path": run_state_path,
                "is_mock": mock,
            }
        except NonRetriableAPIError as exc:
            err_str = str(exc).lower()
            if "400" in err_str and any(k in err_str for k in ("parameter", "thinking_level", "thinking", "generation_config")):
                print(f"[PILOT] Arm '{arm}' rejected by API (unsupported parameter): {sanitize_error(str(exc))}")
                # Mark arm unsupported, record state, and do NOT retry
                unsupported_payload = {
                    "arm": arm,
                    "status": "unsupported",
                    "reason": sanitize_error(str(exc)),
                    "counts": {"ok": 0, "infra_failed": 0, "pending": 0, "total": 0},
                    "cases": {},
                }
                with open(run_state_path, "w", encoding="utf-8") as f:
                    json.dump(unsupported_payload, f, indent=2)
                run_summaries[arm] = {
                    "status": "unsupported",
                    "reason": sanitize_error(str(exc)),
                    "run_state_path": run_state_path,
                }
                continue
            else:
                raise

    return run_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pilot evaluation across thinking levels")
    parser.add_argument("--seed", type=int, default=1, help="Seed index for population (default: 1)")
    parser.add_argument("--arms", default="default,low,high", help="Comma-separated list of arms (default: default,low,high)")
    parser.add_argument("--n-residual", type=int, default=100, help="Number of residual cases (default: 100)")
    parser.add_argument("--n-control", type=int, default=20, help="Number of control cases (default: 20)")
    parser.add_argument("--max-requests-per-day", type=int, default=450, help="Daily request quota limit (default: 450)")
    parser.add_argument("--mock", action="store_true", help="Use MockClient instead of real Gemini API")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run")
    parser.add_argument("--population-dir", default="data/populations", help="Population directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--model", default=None, help="Model ID override")
    parser.add_argument("--run-id", type=int, default=0, help="Run ID (default: 0)")
    args = parser.parse_args()

    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    run_pilot(
        seed=args.seed,
        arms=arm_list,
        n_residual=args.n_residual,
        n_control=args.n_control,
        max_requests_per_day=args.max_requests_per_day,
        mock=args.mock,
        resume=args.resume,
        population_dir=args.population_dir,
        output_dir=args.output_dir,
        model_id=args.model,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
