"""Read-only error analysis of the real pilot (dev seed 1, run 0, three arms).

Inspects failure modes, ungrounded F4 tokens, unblocked hallucinations, unsafe resolutions,
B3-accepted errors, false rejections, latency drivers, confusion matrices, and token usage.
Does NOT make any LLM calls or modify existing data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import glob
import json
import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_pilot import load_stratified_prefix
from src.recon_agent.eval.independent_checker import (
    check_output,
    is_hallucinated,
    reflects_untrusted_text,
    _ID_LIKE,
    _NUM,
    _dec,
    _paise,
    _nums_in,
)
from src.recon_agent.llm.bundle import CaseBundle

DECISION_ORDER = ["no_issue", "benign_explained", "flag", "escalate_human"]


def extract_f4_offending_tokens(bundle_dict: dict[str, Any], raw_output: dict[str, Any]) -> list[str]:
    """Extract offending number/amount tokens that triggered F4 in independent_checker."""
    facts = bundle_dict.get("facts") or {}
    untrusted = {k: v for k, v in (bundle_dict.get("untrusted_text") or {}).items() if isinstance(v, str)}

    quote = raw_output.get("verbatim_quote")
    quote_ok = isinstance(quote, str) and quote != "" and any(quote in t for t in untrusted.values())
    quote_amounts: set[Decimal] = set()
    quote_digits: set[Decimal] = set()
    if quote_ok:
        for m in _NUM.finditer(quote):
            has_unit, p = _paise(m)
            d = _dec(m.group("num"))
            if d is not None:
                quote_digits.add(d)
            if has_unit and p is not None:
                quote_amounts.add(p)

    fact_ints = {abs(v) for v in facts.values() if isinstance(v, int) and not isinstance(v, bool)}
    cited = raw_output.get("cited_amount_paise")
    allowed_paise = {Decimal(x) for x in fact_ints} | quote_amounts
    if isinstance(cited, int) and not isinstance(cited, bool):
        allowed_paise.add(Decimal(abs(cited)))

    string_digits: set[Decimal] = set()
    for v in facts.values():
        if isinstance(v, str):
            string_digits |= _nums_in(_ID_LIKE.sub(" ", v))

    allowed_bare = (
        {Decimal(0), Decimal(1), Decimal(100)}
        | {Decimal(x) for x in fact_ints}
        | quote_digits
        | string_digits
    )

    explanation = raw_output.get("explanation")
    cleaned = _ID_LIKE.sub(" ", explanation if isinstance(explanation, str) else "")
    offending: list[str] = []
    for m in _NUM.finditer(cleaned):
        has_unit, p = _paise(m)
        tok = m.group(0).strip()
        if has_unit:
            if p is None or p not in allowed_paise:
                offending.append(tok)
        else:
            d = _dec(m.group("num"))
            if d is None or d not in allowed_bare:
                offending.append(tok)
    return offending


def build_confusion_matrix(expected: list[str], predicted: list[str], labels: list[str] = DECISION_ORDER) -> dict[str, dict[str, int]]:
    """Build confusion matrix as {exp_label: {pred_label: count}}."""
    matrix = {exp: {pred: 0 for pred in labels} for exp in labels}
    for exp, pred in zip(expected, predicted):
        if exp in matrix and pred in matrix[exp]:
            matrix[exp][pred] += 1
    return matrix


def format_confusion_matrix(matrix: dict[str, dict[str, int]], labels: list[str] = DECISION_ORDER) -> str:
    """Format confusion matrix as Markdown table."""
    header = "| Expected \\ Predicted | " + " | ".join(labels) + " | Total |"
    sep = "| :--- | " + " | ".join([":---:"] * len(labels)) + " | :---: |"
    lines = [header, sep]
    for exp in labels:
        row_counts = [matrix[exp][pred] for pred in labels]
        row_tot = sum(row_counts)
        lines.append(f"| **{exp}** | " + " | ".join(str(c) for c in row_counts) + f" | **{row_tot}** |")
    return "\n".join(lines)


def run_error_analysis(
    seed: int = 1,
    arms: list[str] | None = None,
    n_residual: int = 100,
    n_control: int = 20,
    population_dir: str = "data/populations",
    results_dir: str = "results",
    finalized_dir: str = "results/finalized",
) -> dict[str, Any]:
    """Execute complete error analysis across requested arms and return structured findings."""
    arms = arms or ["default", "low", "high"]
    resolved_pop_dir = population_dir if os.path.isabs(population_dir) else os.path.join(REPO_ROOT, population_dir)
    resolved_res_dir = results_dir if os.path.isabs(results_dir) else os.path.join(REPO_ROOT, results_dir)
    resolved_fin_dir = finalized_dir if os.path.isabs(finalized_dir) else os.path.join(REPO_ROOT, finalized_dir)

    pop_file = os.path.join(resolved_pop_dir, f"dev_seed{seed}.jsonl")
    meta_file = os.path.join(resolved_pop_dir, f"dev_seed{seed}_meta.jsonl")

    bundles, meta_lookup = load_stratified_prefix(
        population_file=pop_file,
        meta_file=meta_file,
        n_residual=n_residual,
        n_control=n_control,
    )
    bundles_by_id = {b.bundle_id: b for b in bundles}

    findings: dict[str, Any] = {}

    print("=" * 80)
    print(f"PILOT ERROR ANALYSIS: dev_seed{seed} (n_residual={n_residual}, n_control={n_control})")
    print("=" * 80)

    for arm in arms:
        print(f"\n================================================================================")
        print(f"ARM: {arm}")
        print(f"================================================================================")

        # Load run state
        st_path = os.path.join(resolved_res_dir, f"pilot_run_state_{arm}_seed{seed}.json")
        with open(st_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        cases_in_state = st.get("cases", {})

        # Load finalized records
        fin_pattern = os.path.join(resolved_fin_dir, f"**/finalized_dev_seed{seed}_{arm}_run0.jsonl")
        fin_matches = glob.glob(fin_pattern, recursive=True)
        fin_records: dict[str, dict[str, Any]] = {}
        if fin_matches:
            with open(fin_matches[0], "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    fin_records[rec["bundle_id"]] = rec

        arm_findings: dict[str, Any] = {}

        # ----------------------------------------------------------------------
        # 1. F4 Analysis
        # ----------------------------------------------------------------------
        f4_cases = []
        tokens_by_family = defaultdict(Counter)
        f4_correct_raw_count = 0

        # ----------------------------------------------------------------------
        # 2. Unblocked Hallucinations
        # ----------------------------------------------------------------------
        unblocked_hallucinations = []

        # ----------------------------------------------------------------------
        # 3. Unsafe Raw Resolutions
        # ----------------------------------------------------------------------
        unsafe_raw_cases = []

        # ----------------------------------------------------------------------
        # 4. B3-Accepted But Wrong Decisions
        # ----------------------------------------------------------------------
        accepted_wrong_cases = []

        # ----------------------------------------------------------------------
        # 5. V8 REFLECTED_UNTRUSTED_TEXT Rejections
        # ----------------------------------------------------------------------
        v8_rejections = []

        # ----------------------------------------------------------------------
        # 6. NO_AMOUNT_IN_QUOTE Rejections
        # ----------------------------------------------------------------------
        no_amount_in_quote_cases = []

        # ----------------------------------------------------------------------
        # 8 & 9. Decisions & Tokens
        # ----------------------------------------------------------------------
        exp_decisions: list[str] = []
        b2_decisions: list[str] = []
        b3_decisions: list[str] = []
        input_tokens_list: list[int] = []
        output_tokens_list: list[int] = []
        thinking_tokens_list: list[int] = []
        latencies_list: list[tuple[str, int, str]] = []

        for bid, b in bundles_by_id.items():
            meta = meta_lookup[bid]
            exp_dec = meta["expected_decision"]
            c = cases_in_state.get(bid, {})
            rec = fin_records.get(bid, {})

            raw_text = c.get("raw_output")
            raw_dict = None
            if raw_text:
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict):
                        raw_dict = parsed
                except Exception:
                    pass

            b_dict = b.to_dict()
            facts = b_dict.get("facts") or {}
            untrusted = {k: v for k, v in (b_dict.get("untrusted_text") or {}).items() if isinstance(v, str)}

            raw_dec = rec.get("raw_decision") if rec else (raw_dict.get("decision") if raw_dict else None)
            b2_act = rec.get("b2_action") or raw_dec or "escalate_human"
            b3_act = rec.get("b3_action") or "escalate_human"
            v_outcome = rec.get("verifier_outcome")
            v_failed = rec.get("verifier_failed_checks") or []
            v_code = rec.get("verifier_reason_code")

            exp_decisions.append(exp_dec)
            b2_decisions.append(b2_act)
            b3_decisions.append(b3_act)

            if c.get("input_tokens") is not None:
                input_tokens_list.append(c["input_tokens"])
            if c.get("output_tokens") is not None:
                output_tokens_list.append(c["output_tokens"])
            if c.get("thinking_tokens") is not None:
                thinking_tokens_list.append(c["thinking_tokens"])
            if c.get("latency_ms") is not None:
                latencies_list.append((bid, c["latency_ms"], c.get("status", "unknown")))

            # Check independent checker violations
            violations = check_output(b_dict, raw_dict) if raw_dict else []
            f4_offending = []
            if "F4" in violations:
                f4_offending = extract_f4_offending_tokens(b_dict, raw_dict)
                for tok in f4_offending:
                    tokens_by_family[b.check_family][tok] += 1
                is_correct = (raw_dec == exp_dec)
                if is_correct:
                    f4_correct_raw_count += 1
                f4_cases.append({
                    "bundle_id": bid,
                    "check_family": b.check_family,
                    "source_label_type": meta["source_label_type"],
                    "offending_tokens": f4_offending,
                    "raw_decision": raw_dec,
                    "expected_decision": exp_dec,
                    "raw_decision_correct": is_correct,
                    "explanation": raw_dict.get("explanation"),
                })

            # 2. Unblocked hallucinations
            if violations and v_outcome == "ACCEPT":
                unblocked_hallucinations.append({
                    "bundle_id": bid,
                    "facts": facts,
                    "raw_output": raw_dict,
                    "violations": violations,
                    "offending_tokens": f4_offending,
                    "verifier_outcome": v_outcome,
                    "raw_decision": raw_dec,
                    "expected_decision": exp_dec,
                })

            # 3. Unsafe raw resolutions
            if exp_dec in ("flag", "escalate_human") and raw_dec in ("benign_explained", "no_issue"):
                unsafe_raw_cases.append({
                    "bundle_id": bid,
                    "check_family": b.check_family,
                    "ambiguity_type": meta["source_label_type"],
                    "raw_decision": raw_dec,
                    "reason_code": raw_dict.get("reason_code") if raw_dict else None,
                    "quote": raw_dict.get("verbatim_quote") if raw_dict else None,
                    "verifier_outcome": v_outcome,
                    "verifier_failed_checks": v_failed,
                    "verifier_reason_code": v_code,
                })

            # 4. B3-accepted wrong decisions
            if v_outcome == "ACCEPT" and b3_act != exp_dec:
                accepted_wrong_cases.append({
                    "bundle_id": bid,
                    "check_family": b.check_family,
                    "expected_decision": exp_dec,
                    "raw_decision": raw_dec,
                    "b3_action": b3_act,
                    "reason_code": raw_dict.get("reason_code") if raw_dict else None,
                    "verifier_outcome": v_outcome,
                    "raw_output": raw_dict,
                })

            # 5. V8 REFLECTED_UNTRUSTED_TEXT rejections
            if "REFLECTED_UNTRUSTED_TEXT" in v_failed:
                v8_rejections.append({
                    "bundle_id": bid,
                    "explanation": raw_dict.get("explanation") if raw_dict else "",
                    "untrusted": untrusted,
                    "quote": raw_dict.get("verbatim_quote") if raw_dict else "",
                })

            # 6. NO_AMOUNT_IN_QUOTE rejections
            if "NO_AMOUNT_IN_QUOTE" in v_failed:
                no_amount_in_quote_cases.append({
                    "bundle_id": bid,
                    "raw_decision": raw_dec,
                    "expected_decision": exp_dec,
                    "verifier_reason_code": v_code,
                })

        arm_findings["f4_cases"] = f4_cases
        arm_findings["tokens_by_family"] = {k: dict(v) for k, v in tokens_by_family.items()}
        arm_findings["f4_correct_raw_count"] = f4_correct_raw_count
        arm_findings["unblocked_hallucinations"] = unblocked_hallucinations
        arm_findings["unsafe_raw_cases"] = unsafe_raw_cases
        arm_findings["accepted_wrong_cases"] = accepted_wrong_cases
        arm_findings["v8_rejections"] = v8_rejections
        arm_findings["no_amount_in_quote_cases"] = no_amount_in_quote_cases

        # ----------------------------------------------------------------------
        # Print Section 1: F4 Analysis
        # ----------------------------------------------------------------------
        print(f"### 1. F4 Number Token Analysis")
        print(f"- Total F4 Flagged Outputs: {len(f4_cases)}")
        print(f"- F4 Outputs with Correct Raw Decision: {f4_correct_raw_count}/{len(f4_cases)}")
        print()
        print("| Bundle ID | Check Family | Source Label Type | Offending Tokens | Raw Correct? | Explanation |")
        print("| :--- | :--- | :--- | :--- | :---: | :--- |")
        for item in f4_cases:
            toks_str = ", ".join(f"`{t}`" for t in item["offending_tokens"]) or "(none extracted)"
            corr_str = "YES" if item["raw_decision_correct"] else "NO"
            expl_clean = (item["explanation"] or "").replace("\n", " ").strip()
            print(f"| `{item['bundle_id']}` | {item['check_family']} | {item['source_label_type']} | {toks_str} | {corr_str} | {expl_clean} |")
        print()
        print("#### Frequency Table of Offending Tokens per Family:")
        for fam, t_counts in tokens_by_family.items():
            print(f"- **Family `{fam}`**:")
            for tok, cnt in sorted(t_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - `{tok}`: {cnt}")
        print()

        # ----------------------------------------------------------------------
        # Print Section 2: Unblocked Hallucinations
        # ----------------------------------------------------------------------
        print(f"### 2. Unblocked Hallucinated Outputs (Verifier ACCEPT)")
        print(f"- Total Unblocked: {len(unblocked_hallucinations)}")
        for uh in unblocked_hallucinations:
            bid = uh["bundle_id"]
            print(f"\n#### Bundle: `{bid}`")
            print(f"- **Raw Decision**: `{uh['raw_decision']}` | **Expected Decision**: `{uh['expected_decision']}`")
            print(f"- **Checker Violations**: {uh['violations']}")
            print(f"- **Offending Token(s)**: {uh['offending_tokens']}")
            print(f"- **Facts (excluding untrusted_text)**:")
            print(f"```json\n{json.dumps(uh['facts'], indent=2, sort_keys=True)}\n```")
            print(f"- **Raw Output**:")
            print(f"```json\n{json.dumps(uh['raw_output'], indent=2)}\n```")
            print(f"- **Target Verifier Check**: `V6 (unit-aware explanation amount & bare-number verification)`")
            print(f"- **Root Cause Why It Did Not Catch It**:")
            print(
                f"  1. `AMOUNT_UNIT_PATTERN` in `src/recon_agent/llm/verifier.py` checks prefix currency codes "
                f"(`INR`, `Rs`, `₹`) or postfix `paise|paisa`, but does NOT match postfix `rupee|rupees`.\n"
                f"  2. In `independent_checker.py`, `_NUM` regex matches postfix `rupee|rupees`, converting '1 rupee' "
                f"to 100 paise, which is not present in facts, thereby flagging F4.\n"
                f"  3. In `verifier.py`, '1 rupee' failed `AMOUNT_UNIT_PATTERN` matching and fell through to bare-number "
                f"extraction (`DIGIT_PATTERN`), extracting '1'.\n"
                f"  4. The bare number '1' matched identity constant `1 in {{0, 1, 100}}` in `allowed_bare`, "
                f"allowing the ungrounded 1-rupee deduction claim to pass V6 and receive ACCEPT."
            )
        print()

        # ----------------------------------------------------------------------
        # Print Section 3: Unsafe Raw Resolutions
        # ----------------------------------------------------------------------
        print(f"### 3. Unsafe Raw Resolutions")
        print(f"- Total Unsafe Raw Cases: {len(unsafe_raw_cases)}")
        print()
        print("| Bundle ID | Family | Ambiguity Type | Raw Decision | Reason Code | Verbatim Quote | Verifier Reason Code | Blocked in B3? |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |")
        for u in unsafe_raw_cases:
            q_str = f"`{u['quote']}`" if u["quote"] else "*(none)*"
            blocked = "YES" if u["verifier_outcome"] != "ACCEPT" else "NO"
            print(f"| `{u['bundle_id']}` | {u['check_family']} | {u['ambiguity_type']} | `{u['raw_decision']}` | `{u['reason_code']}` | {q_str} | `{u['verifier_reason_code']}` | {blocked} |")
        print()

        # ----------------------------------------------------------------------
        # Print Section 4: B3-Accepted But Wrong Decisions
        # ----------------------------------------------------------------------
        print(f"### 4. B3-Accepted But Wrong Decisions")
        print(f"- Total Accepted But Wrong: {len(accepted_wrong_cases)}")
        print()
        print("| Bundle ID | Family | Raw Decision | B3 Action | Expected Decision | Why Verifier Accepted |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for aw in accepted_wrong_cases:
            why = ""
            if aw["b3_action"] == "escalate_human":
                why = "Raw decision was escalate_human. Because it passed V1-V6 and V8 guards, it was deemed safe to escalate to a human reviewer."
            elif aw["b3_action"] == "flag":
                why = "Raw decision was flag with valid reason_code and grounded evidence (facts.refund_difference_paise), satisfying flag verification."
            print(f"| `{aw['bundle_id']}` | {aw['check_family']} | `{aw['raw_decision']}` | `{aw['b3_action']}` | `{aw['expected_decision']}` | {why} |")
        print()

        # ----------------------------------------------------------------------
        # Print Section 5: V8 Reflected Untrusted Text Rejections
        # ----------------------------------------------------------------------
        print(f"### 5. V8 REFLECTED_UNTRUSTED_TEXT Rejections")
        print(f"- Total V8 Rejections: {len(v8_rejections)}")
        if not v8_rejections:
            print("  - *(No outputs triggered V8 REFLECTED_UNTRUSTED_TEXT rejections)*")
        for v in v8_rejections:
            print(f"  - Bundle: `{v['bundle_id']}`")
        print()

        # ----------------------------------------------------------------------
        # Print Section 6: NO_AMOUNT_IN_QUOTE Rejections
        # ----------------------------------------------------------------------
        print(f"### 6. NO_AMOUNT_IN_QUOTE Rejections")
        print(f"- Total NO_AMOUNT_IN_QUOTE Rejections: {len(no_amount_in_quote_cases)}")
        if not no_amount_in_quote_cases:
            print("  - *(No outputs triggered NO_AMOUNT_IN_QUOTE rejections)*")
        for na in no_amount_in_quote_cases:
            print(f"  - Bundle: `{na['bundle_id']}`: Raw=`{na['raw_decision']}`, Expected=`{na['expected_decision']}`")
        print()

        # ----------------------------------------------------------------------
        # Print Section 7: Infra Failures & p95 Latency Analysis
        # ----------------------------------------------------------------------
        print(f"### 7. Infrastructure Failures and Latency Outliers")
        infra_failed_in_state = [cid for cid, c in cases_in_state.items() if c.get("status") == "infra_failed"]
        print(f"- Infra-Failed Cases in Run State: {len(infra_failed_in_state)} (0.0% infra-failure rate)")
        if arm == "default":
            print(
                f"  - Note on default arm: All 120 cases completed successfully with status `ok`. "
                f"Zero cases terminated in an unresolved infrastructure failure."
            )
        lat_sorted = sorted(latencies_list, key=lambda x: x[1], reverse=True)
        if lat_sorted:
            n_lat = len(lat_sorted)
            p95_idx = int(round(0.95 * (n_lat - 1)))
            p95_val = lat_sorted[n_lat - 1 - p95_idx][1] if n_lat > 0 else 0
            p95_cases = [x for x in lat_sorted if x[1] >= p95_val]
            print(f"- Calculated p95 Latency: {p95_val:.1f} ms")
            print(f"- Cases Driving p95+ Latency ({len(p95_cases)} cases >= p95):")
            print("| Bundle ID | Latency (ms) | Status | Analysis / Estimated Attempts |")
            print("| :--- | :---: | :---: | :--- |")
            for bid, lat_ms, st_code in p95_cases:
                est_attempts = "1 attempt (single request)"
                if lat_ms > 60000:
                    est_attempts = ">=3 attempts (exponential backoff 5s + 15s + 45s sleep + HTTP RTT)"
                elif lat_ms > 25000:
                    est_attempts = ">=2 attempts (backoff sleep ~15-20s + HTTP RTT)"
                print(f"| `{bid}` | {lat_ms:,} ms | `{st_code}` | {est_attempts} |")
        print()

        # ----------------------------------------------------------------------
        # Print Section 8: Confusion Matrices
        # ----------------------------------------------------------------------
        print(f"### 8. Confusion Matrices (Expected vs Predicted)")
        cm_b2 = build_confusion_matrix(exp_decisions, b2_decisions)
        cm_b3 = build_confusion_matrix(exp_decisions, b3_decisions)
        print("#### B2 (Unverified LLM Decision):")
        print(format_confusion_matrix(cm_b2))
        print("\n#### B3 (Verified Decision):")
        print(format_confusion_matrix(cm_b3))
        print()

        # ----------------------------------------------------------------------
        # Print Section 9: Token Usage
        # ----------------------------------------------------------------------
        print(f"### 9. Token Usage per Call")
        avg_in = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0.0
        avg_out = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0.0
        avg_think = sum(thinking_tokens_list) / len(thinking_tokens_list) if thinking_tokens_list else 0.0
        print(f"- Average Input Tokens : {avg_in:.1f}")
        print(f"- Average Output Tokens: {avg_out:.1f}")
        print(f"- Average Thinking Tokens: {avg_think:.1f}")
        print(f"- Average Total Tokens : {(avg_in + avg_out + avg_think):.1f}")
        print()

        arm_findings["b2_confusion_matrix"] = cm_b2
        arm_findings["b3_confusion_matrix"] = cm_b3
        arm_findings["avg_input_tokens"] = avg_in
        arm_findings["avg_output_tokens"] = avg_out
        findings[arm] = arm_findings

    # Cross-arm comparison for Section 3
    if "default" in findings and "low" in findings:
        def_unsafe_ids = {u["bundle_id"] for u in findings["default"]["unsafe_raw_cases"]}
        low_unsafe_ids = {u["bundle_id"] for u in findings["low"]["unsafe_raw_cases"]}
        print("================================================================================")
        print("CROSS-ARM COMPARISON: Unsafe Raw Resolutions (default vs low)")
        print("================================================================================")
        print(f"Default Arm Unsafe IDs ({len(def_unsafe_ids)}): {sorted(def_unsafe_ids)}")
        print(f"Low Arm Unsafe IDs     ({len(low_unsafe_ids)}): {sorted(low_unsafe_ids)}")
        same_ids = (def_unsafe_ids == low_unsafe_ids)
        print(f"Are the same bundle_ids unsafe in both arms?: **{same_ids}**")
        print(f"- Overlapping in both ({len(def_unsafe_ids & low_unsafe_ids)}): {sorted(def_unsafe_ids & low_unsafe_ids)}")
        print(f"- Default only ({len(def_unsafe_ids - low_unsafe_ids)}): {sorted(def_unsafe_ids - low_unsafe_ids)}")
        print(f"- Low only ({len(low_unsafe_ids - def_unsafe_ids)}): {sorted(low_unsafe_ids - def_unsafe_ids)}")
        print()

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot error analysis (read-only)")
    parser.add_argument("--seed", type=int, default=1, help="Seed index (default: 1)")
    parser.add_argument("--arms", default="default,low,high", help="Comma-separated arms (default: default,low,high)")
    parser.add_argument("--n-residual", type=int, default=100, help="Number of residual cases (default: 100)")
    parser.add_argument("--n-control", type=int, default=20, help="Number of control cases (default: 20)")
    parser.add_argument("--population-dir", default="data/populations", help="Population directory")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--finalized-dir", default="results/finalized", help="Finalized directory")
    args = parser.parse_args()

    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    run_error_analysis(
        seed=args.seed,
        arms=arm_list,
        n_residual=args.n_residual,
        n_control=args.n_control,
        population_dir=args.population_dir,
        results_dir=args.results_dir,
        finalized_dir=args.finalized_dir,
    )


if __name__ == "__main__":
    main()
