"""Resumable and quota-aware runner for LLM evaluations.

Manages execution of CaseBundles through the Gemini API client with:
- Local daily request budget tracking in Pacific time (America/Los_Angeles, midnight reset).
- Deterministic stratified ordering across (seed, family) strata.
- Explicit case status tracking ('ok', 'infra_failed', 'model_invalid', 'pending').
- Resumable runs (--resume) that retry only 'infra_failed' and 'pending' cases.
- Safe termination upon budget exhaustion with clean resume instructions.
- Strict scoring refusal on unresolved infra_failed cases unless --allow-infra-failures is passed.
- Free tier cost reporting ($0.00 actual + estimated paid-equivalent cost).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
import json
import os
import random
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.recon_agent.guard import sanitize_error
from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.gemini_client import (
    DailyQuotaExhaustedError,
    GeminiClient,
    InfraError,
    ModelMismatch,
    load_price_rates,
)
from src.recon_agent.llm.schema import validate_llm_output
from src.recon_agent.llm.verifier import Verifier, VerifierOutcome

LA_TZ = ZoneInfo("America/Los_Angeles")


class QuotaTracker:
    """Local quota tracker for daily request limits keyed by Pacific time."""

    def __init__(
        self,
        quota_state_path: str = "results/quota_state.json",
        max_requests_per_day: int = 450,
        now_fn: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self.quota_state_path = os.path.abspath(quota_state_path)
        self.max_requests_per_day = max_requests_per_day
        self.now_fn = now_fn or (lambda: datetime.datetime.now(LA_TZ))
        os.makedirs(os.path.dirname(self.quota_state_path), exist_ok=True)

    def get_pacific_date(self) -> str:
        """Return the current date string in Pacific Time (America/Los_Angeles)."""
        now = self.now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)
        now_la = now.astimezone(LA_TZ)
        return now_la.strftime("%Y-%m-%d")

    def _load_state(self) -> dict[str, Any]:
        """Load quota state from disk, initializing if absent."""
        if not os.path.isfile(self.quota_state_path):
            return {
                "date": self.get_pacific_date(),
                "requests_today": 0,
                "max_requests_per_day": self.max_requests_per_day,
            }
        try:
            with open(self.quota_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        current_date = self.get_pacific_date()
        if data.get("date") != current_date:
            # Midnight reset
            data["date"] = current_date
            data["requests_today"] = 0
        data["max_requests_per_day"] = self.max_requests_per_day
        return data

    def _save_state(self, state: dict[str, Any]) -> None:
        """Save quota state atomically."""
        temp_path = f"{self.quota_state_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_path, self.quota_state_path)

    def can_request(self) -> bool:
        """Check if remaining daily quota allows another request."""
        state = self._load_state()
        return int(state.get("requests_today", 0)) < self.max_requests_per_day

    def record_request(self) -> int:
        """Record an outbound request, resetting on midnight rollover."""
        state = self._load_state()
        if int(state.get("requests_today", 0)) >= self.max_requests_per_day:
            raise DailyQuotaExhaustedError(
                f"Daily request budget of {self.max_requests_per_day} requests reached for {state['date']}."
            )
        state["requests_today"] = int(state.get("requests_today", 0)) + 1
        self._save_state(state)
        return state["requests_today"]

    def requests_remaining(self) -> int:
        """Return the number of requests remaining today."""
        state = self._load_state()
        return max(0, self.max_requests_per_day - int(state.get("requests_today", 0)))


def order_cases_stratified(cases: list[CaseBundle], rng_seed: int = 42) -> list[CaseBundle]:
    """Derive a fixed pseudo-random order stratified by (seed, family).

    Round-robin interleaves cases across strata so ANY prefix is an unbiased sample.
    """
    if not cases:
        return []

    strata: dict[tuple[str, str], list[CaseBundle]] = {}
    for case in cases:
        seed_val = str(case.facts.get("seed", case.facts.get("split_seed", 0)))
        family_val = str(case.rule_id)
        key = (seed_val, family_val)
        strata.setdefault(key, []).append(case)

    # Sort each stratum deterministically and shuffle with pseudo-random RNG derived from seed
    sorted_strata_keys = sorted(strata.keys())
    shuffled_strata: dict[tuple[str, str], list[CaseBundle]] = {}
    for k in sorted_strata_keys:
        bucket = sorted(strata[k], key=lambda b: b.bundle_id)
        rnd = random.Random(f"{rng_seed}:{k[0]}:{k[1]}")
        rnd.shuffle(bucket)
        shuffled_strata[k] = bucket

    # Interleave round-robin across strata
    ordered: list[CaseBundle] = []
    active = True
    idx = 0
    while active:
        active = False
        for k in sorted_strata_keys:
            bucket = shuffled_strata[k]
            if idx < len(bucket):
                ordered.append(bucket[idx])
                active = True
        idx += 1

    return ordered


@dataclass
class CaseResult:
    """Per-case evaluation outcome."""

    bundle_id: str
    status: str  # 'ok' (raw obtained) | 'infra_failed' | 'pending' (or 'model_invalid' for legacy)
    decision: str | None = None
    predicted_class: str | None = None
    raw_output: str | None = None
    error_message: str | None = None
    cost_usd: float | None = None
    estimated_cost_low_usd: float | None = None
    estimated_cost_high_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    served_model: str | None = None
    service_tier: str | None = None


class Runner:
    """Manages resumable, quota-aware execution of CaseBundles collecting raw outputs only."""

    def __init__(
        self,
        client: GeminiClient,
        quota_tracker: QuotaTracker,
        system_prompt: str,
        run_state_path: str = "results/run_state.json",
        verifier: Verifier | None = None,
    ) -> None:
        self.client = client
        self.quota_tracker = quota_tracker
        self.system_prompt = system_prompt
        self.run_state_path = os.path.abspath(run_state_path)
        # Note: verifier is never called by Runner; kept for interface compatibility
        self.verifier = verifier or Verifier()
        os.makedirs(os.path.dirname(self.run_state_path), exist_ok=True)

    def load_state(self) -> dict[str, CaseResult]:
        """Load prior run state if available."""
        if not os.path.isfile(self.run_state_path):
            return {}
        try:
            with open(self.run_state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = {}
            for bid, res_data in data.get("cases", {}).items():
                results[bid] = CaseResult(**res_data)
            return results
        except Exception:
            return {}

    def save_state(self, results: dict[str, CaseResult], extra_meta: dict[str, Any] | None = None) -> None:
        """Persist current run state to disk."""
        cases_dict = {bid: asdict(cr) for bid, cr in results.items()}
        counts = {
            "ok": sum(1 for c in results.values() if c.status == "ok"),
            "infra_failed": sum(1 for c in results.values() if c.status == "infra_failed"),
            "model_invalid": sum(1 for c in results.values() if c.status == "model_invalid"),
            "pending": sum(1 for c in results.values() if c.status == "pending"),
            "total": len(results),
        }
        payload = {
            "model_id": self.client.model_id,
            "counts": counts,
            "cases": cases_dict,
            "meta": extra_meta or {},
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        temp_path = f"{self.run_state_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_path, self.run_state_path)

    def run_cases(
        self,
        cases: list[CaseBundle],
        resume: bool = False,
        run_id: int = 0,
    ) -> dict[str, CaseResult]:
        """Run reconciliation on the provided cases with resume and quota handling."""
        existing_results = self.load_state() if resume else {}
        results: dict[str, CaseResult] = {}

        # Initialize status for all cases
        for case in cases:
            bid = case.bundle_id
            if resume and bid in existing_results:
                prev = existing_results[bid]
                # Only re-run infra_failed or pending cases on resume
                if prev.status in ("ok", "model_invalid"):
                    results[bid] = prev
                    continue
            results[bid] = CaseResult(bundle_id=bid, status="pending")

        self.save_state(results)

        # Execute pending or retriable cases
        stopped_reason: str | None = None
        for case in cases:
            bid = case.bundle_id
            current_res = results[bid]
            if current_res.status in ("ok", "model_invalid"):
                continue

            # Check daily quota budget before making call
            if not self.quota_tracker.can_request():
                stopped_reason = (
                    f"Daily budget reached ({self.quota_tracker.max_requests_per_day} requests). "
                    "Paused run cleanly. Resume tomorrow with --resume."
                )
                print(f"[QUOTA] {stopped_reason}")
                break

            try:
                self.quota_tracker.record_request()
                raw_output = self.client.complete(case, self.system_prompt, run_id=run_id)

                toks = getattr(self.client, "last_tokens", None) or {}
                cost_low = getattr(self.client, "last_cost_low", None)
                cost_high = getattr(self.client, "last_cost_high", None)
                served_m = getattr(self.client, "last_served_model", self.client.model_id)
                svc_tier = getattr(self.client, "last_service_tier", self.client.tier)
                eff_cost = 0.0 if svc_tier == "free" and cost_high is not None else cost_high

                # Runner collects RAW outputs only: status is ok once raw_output is obtained.
                # Parse status and verification are handled strictly by finalize.
                results[bid] = CaseResult(
                    bundle_id=bid,
                    status="ok",
                    raw_output=raw_output,
                    cost_usd=eff_cost,
                    estimated_cost_low_usd=cost_low,
                    estimated_cost_high_usd=cost_high,
                    input_tokens=toks.get("input_tokens"),
                    output_tokens=toks.get("output_tokens"),
                    thinking_tokens=toks.get("thinking_tokens"),
                    served_model=served_m,
                    service_tier=svc_tier,
                )

            except DailyQuotaExhaustedError as exc:
                stopped_reason = f"Gemini API 429 daily quota exhausted. Paused run cleanly. Resume tomorrow with --resume."
                print(f"[QUOTA] {stopped_reason}")
                break

            except InfraError as exc:
                # Infrastructure failure produces NO model output
                results[bid] = CaseResult(
                    bundle_id=bid,
                    status="infra_failed",
                    decision=None,
                    predicted_class=None,
                    raw_output=None,
                    error_message=sanitize_error(str(exc)),
                )

            except ModelMismatch:
                # Fatal error, never counted as LLM failure, stop immediately
                self.save_state(results)
                raise

            self.save_state(results)

        self.save_state(results, extra_meta={"stopped_reason": stopped_reason} if stopped_reason else None)
        return results


def score_llm_run(
    results: dict[str, CaseResult],
    total_expected: int | None = None,
    allow_infra_failures: bool = False,
    tier: str = "free",
    model_id: str = "gemini-3.5-flash-lite",
    prices_path: str = "config/llm_prices.json",
) -> dict[str, Any]:
    """Score evaluation results, refusing if unresolved infra_failed cases remain.

    When tier='free', actual cost is reported as 0.0, and 'estimated paid-equivalent cost'
    is calculated from config/llm_prices.json.
    """
    total = total_expected if total_expected is not None else len(results)
    n_ok = sum(1 for c in results.values() if c.status == "ok")
    n_infra_failed = sum(1 for c in results.values() if c.status == "infra_failed")
    n_model_invalid = sum(1 for c in results.values() if c.status == "model_invalid")
    n_pending = sum(1 for c in results.values() if c.status == "pending")
    completed = n_ok + n_model_invalid

    if n_infra_failed > 0 and not allow_infra_failures:
        raise RuntimeError(
            f"Run contains {n_infra_failed} infra_failed cases. Refusing to score. "
            f"Re-run with --resume or pass --allow-infra-failures to map them to escalate_human."
        )

    # If allow_infra_failures is passed, map them to escalate_human for scoring
    scored_decisions: dict[str, str] = {}
    for bid, res in results.items():
        if res.status == "ok" and res.decision is not None:
            scored_decisions[bid] = res.decision
        elif res.status == "model_invalid":
            scored_decisions[bid] = "escalate_human"
        elif res.status == "infra_failed" and allow_infra_failures:
            scored_decisions[bid] = "escalate_human"

    infra_failure_rate = (n_infra_failed / total) if total > 0 else 0.0

    # Load price rates for paid equivalent calculation
    rates = load_price_rates(prices_path=prices_path, model_id=model_id)

    # Cost calculations
    tot_in_tokens = 0
    tot_out_tokens = 0
    tot_think_tokens = 0
    tokens_available = True

    for res in results.values():
        if res.status in ("ok", "model_invalid"):
            if res.input_tokens is not None and res.output_tokens is not None:
                tot_in_tokens += res.input_tokens
                tot_out_tokens += res.output_tokens
                tot_think_tokens += res.thinking_tokens or 0
            else:
                tokens_available = False

    if tokens_available and completed > 0:
        in_cost = (tot_in_tokens * rates["input_cost_per_million"]) / 1_000_000.0
        out_cost = (tot_out_tokens * rates["output_cost_per_million"]) / 1_000_000.0
        think_cost = (tot_think_tokens * rates["thinking_cost_per_million"]) / 1_000_000.0
        paid_equiv_low = round(in_cost + out_cost, 6)
        paid_equiv_high = round(in_cost + out_cost + think_cost, 6)
    else:
        paid_equiv_low = None
        paid_equiv_high = None

    rate_verified = rates.get("verified", False)
    if tier == "free":
        actual_cost_usd = 0.0 if completed > 0 else 0.0
        report_cost_str = "$0.00 (free tier)"
    else:
        actual_cost_usd = paid_equiv_high
        base_cost = f"${actual_cost_usd:.4f}" if actual_cost_usd is not None else "None"
        report_cost_str = f"{base_cost} (UNVERIFIED RATE)" if not rate_verified and actual_cost_usd is not None else base_cost

    header_warning = None
    if allow_infra_failures and n_infra_failed > 0:
        header_warning = (
            f"WARNING: Scored with --allow-infra-failures. Completed: {completed}/{total}. "
            f"Infra-failure rate: {infra_failure_rate:.1%}"
        )

    return {
        "header_warning": header_warning,
        "total_cases": total,
        "completed_cases": completed,
        "ok_cases": n_ok,
        "model_invalid_cases": n_model_invalid,
        "infra_failed_cases": n_infra_failed,
        "pending_cases": n_pending,
        "infra_failure_rate": infra_failure_rate,
        "actual_cost_usd": actual_cost_usd,
        "report_cost_str": report_cost_str,
        "estimated_paid_equivalent_cost_usd": paid_equiv_high,
        "estimated_paid_equivalent_cost_low_usd": paid_equiv_low,
        "scored_decisions": scored_decisions,
    }

