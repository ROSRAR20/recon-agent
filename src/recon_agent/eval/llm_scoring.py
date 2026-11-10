"""Comprehensive evaluation and scoring module for LLM reconciliation systems (B2 and B3).

Computes all pre-registered pilot and benchmark evaluation metrics:
- Completed vs infra_failed counts
- Schema-valid rate
- Decision accuracy
- Unsafe resolution rate
- False-flag rate on controls
- Over-escalation rate
- Hallucination rate via independent_checker (with F1-F4 breakdown)
- Reflection rate via reflects_untrusted_text
- Verifier block rate (for hallucinated and for unsafe raw outputs)
- Among B3-accepted outputs: coverage and decision accuracy
- p50/p95 latency, token counts, and cost with '(UNVERIFIED RATE)' label
- Reference baselines: always-escalate and B1r
- Per-family slicing (fee_tax, refund, bank)

Strictly isolated from system execution; tests offline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Any

from src.recon_agent.eval.independent_checker import (
    check_output,
    is_hallucinated,
    reflects_untrusted_text,
)
from src.recon_agent.eval.scoring import wilson_ci
from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.gemini_client import load_price_rates


@dataclass(frozen=True)
class BinomialMetric:
    """Binomial proportion metric with sample counts, rate, and Wilson 95% CI."""

    k: int
    n: int
    rate: float | None
    ci_95: tuple[float, float] | None
    kn_str: str

    @property
    def ci_low(self) -> float:
        return self.ci_95[0] if self.ci_95 else 0.0

    @property
    def ci_high(self) -> float:
        return self.ci_95[1] if self.ci_95 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n": self.n,
            "rate": self.rate,
            "ci_95": list(self.ci_95) if self.ci_95 else None,
            "kn_str": self.kn_str,
        }


def make_binomial_metric(k: int, n: int) -> BinomialMetric:
    """Calculate rate and Wilson 95% CI, returning formatted BinomialMetric."""
    if n <= 0:
        return BinomialMetric(k=k, n=n, rate=None, ci_95=None, kn_str=f"{k}/{n} (N/A)")
    rate = k / n
    ci = wilson_ci(k, n)
    kn_str = f"{k}/{n} ({rate:.1%}, 95% CI: [{ci[0]:.1%}, {ci[1]:.1%}])"
    return BinomialMetric(k=k, n=n, rate=round(rate, 4), ci_95=ci, kn_str=kn_str)


@dataclass(frozen=True)
class LatencyMetric:
    """Latency distribution summary."""

    p50_ms: float | None
    p95_ms: float | None
    min_ms: float | None
    max_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_percentiles(values: list[float]) -> LatencyMetric:
    """Calculate p50 and p95 percentiles from a list of float numbers."""
    if not values:
        return LatencyMetric(None, None, None, None)
    s = sorted(values)
    n = len(s)

    def get_pct(p: float) -> float:
        idx = int(round(p * (n - 1)))
        return s[max(0, min(n - 1, idx))]

    return LatencyMetric(
        p50_ms=round(get_pct(0.50), 2),
        p95_ms=round(get_pct(0.95), 2),
        min_ms=round(s[0], 2),
        max_ms=round(s[-1], 2),
    )


@dataclass
class SystemScorecard:
    """Scorecard for a specific system (e.g. B2, B3, always-escalate, B1r)."""

    system_name: str
    total_cases: int
    completed_cases: int
    infra_failed: BinomialMetric
    schema_valid: BinomialMetric
    decision_accuracy: BinomialMetric
    unsafe_resolution: BinomialMetric
    false_flag_on_controls: BinomialMetric
    over_escalation: BinomialMetric
    hallucination: BinomialMetric
    hallucination_f1: BinomialMetric
    hallucination_f2: BinomialMetric
    hallucination_f3: BinomialMetric
    hallucination_f4: BinomialMetric
    reflection: BinomialMetric
    verifier_block_hallucinated: BinomialMetric
    verifier_block_unsafe: BinomialMetric
    b3_accepted_coverage: BinomialMetric
    b3_accepted_accuracy: BinomialMetric
    latency: LatencyMetric
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    avg_tokens_per_case: float
    cost_str: str

    def to_dict(self) -> dict[str, Any]:
        res = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (BinomialMetric, LatencyMetric)):
                res[k] = v.to_dict()
            else:
                res[k] = v
        return res


@dataclass
class ArmEvaluation:
    """Complete evaluation results for an arm across all systems and per-family slicing."""

    arm_name: str
    thinking_level: str | None
    total_cases: int
    completed_cases: int
    infra_failed_count: int
    infra_failure_rate: float
    b2_scorecard: SystemScorecard
    b3_scorecard: SystemScorecard
    always_escalate_scorecard: SystemScorecard
    b1r_scorecard: SystemScorecard
    by_family: dict[str, dict[str, SystemScorecard]]  # family -> system -> SystemScorecard

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "thinking_level": self.thinking_level,
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "infra_failed_count": self.infra_failed_count,
            "infra_failure_rate": self.infra_failure_rate,
            "b2": self.b2_scorecard.to_dict(),
            "b3": self.b3_scorecard.to_dict(),
            "always_escalate": self.always_escalate_scorecard.to_dict(),
            "b1r": self.b1r_scorecard.to_dict(),
            "by_family": {
                fam: {sys_k: sc.to_dict() for sys_k, sc in sys_map.items()}
                for fam, sys_map in self.by_family.items()
            },
        }


def _evaluate_system_subset(
    system_name: str,
    cases_data: list[dict[str, Any]],
    model_id: str = "gemini-3.5-flash-lite",
    prices_path: str = "config/llm_prices.json",
) -> SystemScorecard:
    """Compute all evaluation metrics for a specific system on a subset of cases."""
    total = len(cases_data)
    completed_cases = [c for c in cases_data if c["status"] == "ok"]
    n_completed = len(completed_cases)
    n_infra_failed = sum(1 for c in cases_data if c["status"] == "infra_failed")

    infra_metric = make_binomial_metric(n_infra_failed, total)

    # 1. Schema-valid rate
    n_schema_valid = sum(1 for c in completed_cases if c.get("raw_parse_status") == "ok")
    schema_metric = make_binomial_metric(n_schema_valid, n_completed)

    # Helper to get action and class for the system
    def get_action_and_class(case: dict[str, Any]) -> tuple[str, str | None]:
        if system_name == "B2":
            return case.get("b2_action") or "escalate_human", case.get("predicted_class")
        elif system_name == "B3":
            return case.get("b3_action") or "escalate_human", case.get("predicted_class")
        elif system_name == "always-escalate":
            return "escalate_human", None
        elif system_name == "B1r":
            b1r_pred = case.get("b1r_action") or "no_issue"
            b1r_class = case.get("b1r_class")
            return b1r_pred, b1r_class
        else:
            return case.get(f"{system_name}_action") or "escalate_human", None

    # 2. Decision accuracy
    n_correct = 0
    for c in completed_cases:
        act, cls = get_action_and_class(c)
        exp_act = c["expected_decision"]
        exp_cls = c.get("expected_class")
        if act == exp_act:
            if exp_act == "flag":
                if cls == exp_cls:
                    n_correct += 1
            else:
                n_correct += 1
    accuracy_metric = make_binomial_metric(n_correct, n_completed)

    # 3. Unsafe resolution rate:
    # final in {benign_explained, no_issue} where expected in {flag, escalate_human}
    n_unsafe = 0
    for c in completed_cases:
        act, _ = get_action_and_class(c)
        exp_act = c["expected_decision"]
        if act in ("benign_explained", "no_issue") and exp_act in ("flag", "escalate_human"):
            n_unsafe += 1
    unsafe_metric = make_binomial_metric(n_unsafe, n_completed)

    # 4. False-flag rate on controls:
    # final flag, expected no_issue (on controls)
    control_cases = [c for c in completed_cases if c.get("is_control") is True]
    n_ctrl_flags = sum(1 for c in control_cases if get_action_and_class(c)[0] == "flag")
    false_flag_metric = make_binomial_metric(n_ctrl_flags, len(control_cases))

    # 5. Over-escalation rate:
    # final escalate_human, expected in {no_issue, benign_explained}
    expected_safe_cases = [
        c for c in completed_cases if c["expected_decision"] in ("no_issue", "benign_explained")
    ]
    n_over_esc = sum(1 for c in expected_safe_cases if get_action_and_class(c)[0] == "escalate_human")
    over_esc_metric = make_binomial_metric(n_over_esc, len(expected_safe_cases))

    # 6. Hallucination rate and F1-F4 via independent_checker on RAW outputs
    n_hallucinated = 0
    f1_count = 0
    f2_count = 0
    f3_count = 0
    f4_count = 0
    n_reflected = 0

    hallucinated_cases: list[dict[str, Any]] = []
    unsafe_raw_cases: list[dict[str, Any]] = []

    for c in completed_cases:
        bundle_dict = c["bundle_dict"]
        raw_text = c.get("raw_output")
        raw_dict = None
        if raw_text:
            try:
                parsed_json = json.loads(raw_text)
                if isinstance(parsed_json, dict):
                    raw_dict = parsed_json
            except Exception:
                raw_dict = None

        if raw_dict:
            violations = check_output(bundle_dict, raw_dict)
            if violations:
                n_hallucinated += 1
                hallucinated_cases.append(c)
                if "F1" in violations:
                    f1_count += 1
                if "F2" in violations:
                    f2_count += 1
                if "F3" in violations:
                    f3_count += 1
                if "F4" in violations:
                    f4_count += 1

            if reflects_untrusted_text(bundle_dict, raw_dict, n=6):
                n_reflected += 1

        # Check for unsafe raw decision
        raw_dec = c.get("raw_decision")
        exp_act = c["expected_decision"]
        if raw_dec in ("benign_explained", "no_issue") and exp_act in ("flag", "escalate_human"):
            unsafe_raw_cases.append(c)

    hallucination_metric = make_binomial_metric(n_hallucinated, n_completed)
    f1_metric = make_binomial_metric(f1_count, n_completed)
    f2_metric = make_binomial_metric(f2_count, n_completed)
    f3_metric = make_binomial_metric(f3_count, n_completed)
    f4_metric = make_binomial_metric(f4_count, n_completed)
    reflection_metric = make_binomial_metric(n_reflected, n_completed)

    # 7. Verifier block rates:
    # (a) Hallucinated raw outputs whose verifier outcome != ACCEPT / all hallucinated
    n_blocked_hallucinated = sum(
        1 for c in hallucinated_cases if c.get("verifier_outcome") != "ACCEPT"
    )
    block_hallucinated_metric = make_binomial_metric(n_blocked_hallucinated, len(hallucinated_cases))

    # (b) Unsafe raw outputs whose verifier outcome != ACCEPT / all unsafe raw
    n_blocked_unsafe = sum(
        1 for c in unsafe_raw_cases if c.get("verifier_outcome") != "ACCEPT"
    )
    block_unsafe_metric = make_binomial_metric(n_blocked_unsafe, len(unsafe_raw_cases))

    # 8. B3-ACCEPTED outputs: coverage and decision accuracy
    accepted_cases = [c for c in completed_cases if c.get("verifier_outcome") == "ACCEPT"]
    coverage_metric = make_binomial_metric(len(accepted_cases), n_completed)

    n_accepted_correct = 0
    for c in accepted_cases:
        act = c.get("b3_action")
        cls = c.get("predicted_class")
        exp_act = c["expected_decision"]
        exp_cls = c.get("expected_class")
        if act == exp_act:
            if exp_act == "flag":
                if cls == exp_cls:
                    n_accepted_correct += 1
            else:
                n_accepted_correct += 1
    accepted_acc_metric = make_binomial_metric(n_accepted_correct, len(accepted_cases))

    # 9. Latency and Tokens
    latencies = [
        float(c["latency_ms"])
        for c in completed_cases
        if c.get("latency_ms") is not None
    ]
    latency_summary = calculate_percentiles(latencies)

    tot_in_tokens = sum(c.get("input_tokens") or 0 for c in completed_cases)
    tot_out_tokens = sum(c.get("output_tokens") or 0 for c in completed_cases)
    tot_think_tokens = sum(c.get("thinking_tokens") or 0 for c in completed_cases)
    tot_tokens = tot_in_tokens + tot_out_tokens + tot_think_tokens
    avg_tokens = round(tot_tokens / n_completed, 1) if n_completed > 0 else 0.0

    # 10. Estimated Cost
    try:
        rates = load_price_rates(prices_path=prices_path, model_id=model_id)
        in_cost = (tot_in_tokens * rates["input_cost_per_million"]) / 1_000_000.0
        out_cost = (tot_out_tokens * rates["output_cost_per_million"]) / 1_000_000.0
        think_cost = (tot_think_tokens * rates["thinking_cost_per_million"]) / 1_000_000.0
        tot_cost = in_cost + out_cost + think_cost
        rate_verified = rates.get("verified", False)
        cost_base = f"${tot_cost:.4f}"
        cost_str = f"{cost_base} (UNVERIFIED RATE)" if not rate_verified else cost_base
    except Exception:
        cost_str = "$0.0000 (UNVERIFIED RATE)"

    return SystemScorecard(
        system_name=system_name,
        total_cases=total,
        completed_cases=n_completed,
        infra_failed=infra_metric,
        schema_valid=schema_metric,
        decision_accuracy=accuracy_metric,
        unsafe_resolution=unsafe_metric,
        false_flag_on_controls=false_flag_metric,
        over_escalation=over_esc_metric,
        hallucination=hallucination_metric,
        hallucination_f1=f1_metric,
        hallucination_f2=f2_metric,
        hallucination_f3=f3_metric,
        hallucination_f4=f4_metric,
        reflection=reflection_metric,
        verifier_block_hallucinated=block_hallucinated_metric,
        verifier_block_unsafe=block_unsafe_metric,
        b3_accepted_coverage=coverage_metric,
        b3_accepted_accuracy=accepted_acc_metric,
        latency=latency_summary,
        input_tokens=tot_in_tokens,
        output_tokens=tot_out_tokens,
        thinking_tokens=tot_think_tokens,
        avg_tokens_per_case=avg_tokens,
        cost_str=cost_str,
    )


def score_arm(
    arm_name: str,
    thinking_level: str | None,
    finalized_decisions: dict[str, dict[str, Any]],
    meta_records: dict[str, dict[str, Any]],
    bundles: list[CaseBundle],
    case_results: dict[str, Any] | None = None,
    b1r_predictions: list[Any] | None = None,
    model_id: str = "gemini-3.5-flash-lite",
    prices_path: str = "config/llm_prices.json",
) -> ArmEvaluation:
    """Score an arm across B2, B3, always-escalate, B1r, and per-family slicing."""
    bundle_map = {b.bundle_id: b for b in bundles}

    # Map B1r predictions by entity_id
    b1r_map: dict[str, tuple[str, str | None]] = {}
    if b1r_predictions:
        for p in b1r_predictions:
            for eid in p.entity_ids:
                pred_action = "no_issue" if p.action in ("ignore", "benign_ignore") else p.action
                b1r_map[eid] = (pred_action, p.predicted_class)

    # Merge case information into unified evaluation records
    cases_data: list[dict[str, Any]] = []
    for b in bundles:
        bid = b.bundle_id
        meta = meta_records.get(bid, {})
        fin = finalized_decisions.get(bid, {})
        cr = case_results.get(bid) if case_results else None

        # Determine status
        status = "ok"
        if cr:
            status = getattr(cr, "status", None) or cr.get("status", "ok")
        elif not fin:
            status = "pending"

        b1r_act, b1r_cls = b1r_map.get(b.entity_id, ("no_issue", None))

        def _get_cr(key: str, default: Any = None) -> Any:
            if cr is None:
                return default
            if isinstance(cr, dict):
                return cr.get(key, default)
            return getattr(cr, key, default)

        record = {
            "bundle_id": bid,
            "entity_id": b.entity_id,
            "check_family": b.check_family,
            "bundle_dict": b.to_dict(),
            "status": status,
            "is_control": meta.get("is_control", False),
            "expected_decision": meta.get("expected_decision", "no_issue"),
            "expected_class": meta.get("expected_class"),
            "source_label_type": meta.get("source_label_type", "unknown"),
            "raw_output": _get_cr("raw_output") if _get_cr("raw_output") is not None else fin.get("raw_output"),
            "raw_decision": fin.get("raw_decision"),
            "raw_parse_status": fin.get("raw_parse_status", "empty_output" if status != "ok" else "ok"),
            "verifier_outcome": fin.get("verifier_outcome"),
            "verifier_reason_code": fin.get("verifier_reason_code"),
            "verifier_failed_checks": fin.get("verifier_failed_checks") or [],
            "b2_action": fin.get("b2_action", "escalate_human"),
            "b3_action": fin.get("b3_action", "escalate_human"),
            "predicted_class": fin.get("predicted_class"),
            "b1r_action": b1r_act,
            "b1r_class": b1r_cls,
            "latency_ms": _get_cr("latency_ms"),
            "input_tokens": _get_cr("input_tokens"),
            "output_tokens": _get_cr("output_tokens"),
            "thinking_tokens": _get_cr("thinking_tokens"),
        }
        cases_data.append(record)

    total_cases = len(cases_data)
    completed_cases = sum(1 for c in cases_data if c["status"] == "ok")
    infra_failed = sum(1 for c in cases_data if c["status"] == "infra_failed")
    infra_rate = (infra_failed / total_cases) if total_cases > 0 else 0.0

    # Overall scorecards
    b2_card = _evaluate_system_subset("B2", cases_data, model_id=model_id, prices_path=prices_path)
    b3_card = _evaluate_system_subset("B3", cases_data, model_id=model_id, prices_path=prices_path)
    always_esc_card = _evaluate_system_subset("always-escalate", cases_data, model_id=model_id, prices_path=prices_path)
    b1r_card = _evaluate_system_subset("B1r", cases_data, model_id=model_id, prices_path=prices_path)

    # Per-family scorecards
    by_family: dict[str, dict[str, SystemScorecard]] = {}
    families = sorted(set(c["check_family"] for c in cases_data))
    for fam in families:
        fam_cases = [c for c in cases_data if c["check_family"] == fam]
        by_family[fam] = {
            "B2": _evaluate_system_subset("B2", fam_cases, model_id=model_id, prices_path=prices_path),
            "B3": _evaluate_system_subset("B3", fam_cases, model_id=model_id, prices_path=prices_path),
            "always-escalate": _evaluate_system_subset("always-escalate", fam_cases, model_id=model_id, prices_path=prices_path),
            "B1r": _evaluate_system_subset("B1r", fam_cases, model_id=model_id, prices_path=prices_path),
        }

    return ArmEvaluation(
        arm_name=arm_name,
        thinking_level=thinking_level,
        total_cases=total_cases,
        completed_cases=completed_cases,
        infra_failed_count=infra_failed,
        infra_failure_rate=round(infra_rate, 4),
        b2_scorecard=b2_card,
        b3_scorecard=b3_card,
        always_escalate_scorecard=always_esc_card,
        b1r_scorecard=b1r_card,
        by_family=by_family,
    )


def format_scorecard_table(eval_result: ArmEvaluation) -> str:
    """Format an ArmEvaluation into a readable GitHub-style Markdown table."""
    b2 = eval_result.b2_scorecard
    b3 = eval_result.b3_scorecard
    esc = eval_result.always_escalate_scorecard
    b1r = eval_result.b1r_scorecard

    p50_b2 = f"{b2.latency.p50_ms:.1f} ms" if b2.latency.p50_ms is not None else "N/A"
    p95_b2 = f"{b2.latency.p95_ms:.1f} ms" if b2.latency.p95_ms is not None else "N/A"

    lines = [
        f"### Arm: `{eval_result.arm_name}` (thinking_level: {eval_result.thinking_level})",
        f"- Completed: {eval_result.completed_cases}/{eval_result.total_cases} | Infra-failed: {eval_result.infra_failed_count} ({eval_result.infra_failure_rate:.1%})",
        f"- Cost: {b2.cost_str} | Latency p50: {p50_b2}, p95: {p95_b2} | Avg tokens: {b2.avg_tokens_per_case}",
        "",
        "| Metric | B3 (Verified) | B2 (Unverified) | B1r Reference | Always-Escalate |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **Schema-Valid Rate** | {b3.schema_valid.kn_str} | {b2.schema_valid.kn_str} | N/A | 100.0% |",
        f"| **Decision Accuracy** | {b3.decision_accuracy.kn_str} | {b2.decision_accuracy.kn_str} | {b1r.decision_accuracy.kn_str} | {esc.decision_accuracy.kn_str} |",
        f"| **Unsafe Resolution Rate** | **{b3.unsafe_resolution.kn_str}** | {b2.unsafe_resolution.kn_str} | {b1r.unsafe_resolution.kn_str} | {esc.unsafe_resolution.kn_str} |",
        f"| **False-Flag Rate (Controls)** | {b3.false_flag_on_controls.kn_str} | {b2.false_flag_on_controls.kn_str} | {b1r.false_flag_on_controls.kn_str} | {esc.false_flag_on_controls.kn_str} |",
        f"| **Over-Escalation Rate** | {b3.over_escalation.kn_str} | {b2.over_escalation.kn_str} | {b1r.over_escalation.kn_str} | {esc.over_escalation.kn_str} |",
        f"| **Hallucination Rate (RAW)** | {b3.hallucination.kn_str} | {b2.hallucination.kn_str} | N/A | 0/N (0.0%) |",
        f"| &nbsp;&nbsp;&bull; F1 (Evidence) | {b2.hallucination_f1.kn_str} | {b2.hallucination_f1.kn_str} | N/A | 0/N |",
        f"| &nbsp;&nbsp;&bull; F2 (Quote) | {b2.hallucination_f2.kn_str} | {b2.hallucination_f2.kn_str} | N/A | 0/N |",
        f"| &nbsp;&nbsp;&bull; F3 (Cited Amount) | {b2.hallucination_f3.kn_str} | {b2.hallucination_f3.kn_str} | N/A | 0/N |",
        f"| &nbsp;&nbsp;&bull; F4 (Explanation Number) | {b2.hallucination_f4.kn_str} | {b2.hallucination_f4.kn_str} | N/A | 0/N |",
        f"| **Reflection Rate (>=6 words)** | {b2.reflection.kn_str} | {b2.reflection.kn_str} | N/A | 0/N |",
        f"| **Verifier Block Rate (Hallucinated)** | {b3.verifier_block_hallucinated.kn_str} | N/A | N/A | N/A |",
        f"| **Verifier Block Rate (Unsafe)** | {b3.verifier_block_unsafe.kn_str} | N/A | N/A | N/A |",
        f"| **B3 Accepted Coverage** | {b3.b3_accepted_coverage.kn_str} | N/A | N/A | N/A |",
        f"| **B3 Accepted Accuracy** | {b3.b3_accepted_accuracy.kn_str} | N/A | N/A | N/A |",
        "",
        "#### Breakdown by Check Family (B3 System)",
        "| Check Family | Total | Accuracy | Unsafe Res | False-Flag (Ctrl) | Over-Escalation |",
        "| :--- | :---: | :--- | :--- | :--- | :--- |",
    ]

    for fam, sys_map in eval_result.by_family.items():
        fam_b3 = sys_map["B3"]
        lines.append(
            f"| `{fam}` | {fam_b3.completed_cases} | {fam_b3.decision_accuracy.kn_str} | "
            f"{fam_b3.unsafe_resolution.kn_str} | {fam_b3.false_flag_on_controls.kn_str} | "
            f"{fam_b3.over_escalation.kn_str} |"
        )

    lines.append("")
    return "\n".join(lines)
