# Evaluation Protocol: recon-agent

This document defines the formal evaluation methodology, baseline hierarchy, two-tier evaluation structure, data-split hygiene, and reporting requirements for benchmarking reconciliation accuracy and verifier efficacy.

---

## 1. Systems Compared (Baselines & Ablations)

To rigorously isolate the contribution of each architectural component, all evaluations compare five distinct configurations on identical transaction sets:

| System ID | Name | Ingestion & Rules | LLM Residual Reasoning | Verification Layer | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B0** | Baseline Exact-ID | Exact ID matching only | None | None | Exact identifier matching only (`payments <-> recon` by `payment_id`, `recon <-> settlements` by `settlement_id`, and `settlements <-> bank` by full UTR as an exact case-sensitive substring of narration without normalization). Unmatched outputs: captured payment without recon item -> `MISSING_SETTLEMENT`; Razorpay bank credit without settlement match -> `ORPHAN_BANK_CREDIT`; settlement without bank match -> `UNKNOWN`. Zero arithmetic checks, ledger checks, or tolerances. |
| **B1** | Deterministic Rules | Extended rules & invariants | None | None | Matches on secondary keys, amount matching (in integer paise), timestamp windows, and fee calculations without any LLM. |
| **B1r** | Rules + Regex Parser | Extended rules & invariants + Regex parser | None | None | Extends B1 with regex pattern matching over free-text fields (`notes.reason_text`, bank narrations) to extract explicit numeric amounts and deduction keywords without an LLM. Establishes the non-LLM text extraction benchmark. |
| **B2** | Rules + Unverified LLM | Extended rules & invariants | Active for residual cases | None | Residual exceptions are routed to an LLM. Model explanations and resolutions are accepted without algorithmic verification. |
| **B3** | Full System (Verified) | Extended rules & invariants | Active for residual cases | **Active** | Residual exceptions are reasoned by LLM but passed through the independent verification layer before acceptance. Failures escalate to human queue. |

---

## 2. Held-Out Test Set Protocols

To prevent data contamination and overfitting:

1. **Independent Seed & Mix:** The held-out test suite is generated using a distinct pseudorandom seed and a distinct injected error distribution from the development set.
2. **Disjoint Text Phrasings:** Free-text templates (for Tier 2 explained refunds) are strictly disjoint between dev and held-out test splits. Dev data uses only dev templates; test data uses only test templates.
3. **Strict Embargo:** The held-out test suite is **never opened, inspected, or executed before Day 12** of the 14-day development cycle.
4. **Single Execution Run:** The final benchmark on the held-out dataset is executed **exactly once** to prevent iterative tuning against the test distribution.

---

## 3. Statistical Power and Reporting Standards

- **Raw Counts & Confidence Intervals Mandatory:** All metrics must be reported with explicit integer counts ($k/n$), not solely percentages (e.g., `"49/50 (98.0%)"` instead of `"98%"`). Pooled multi-seed benchmarks report 95% Wilson score confidence intervals and per-seed range bounds `[min, max]`.
- **Zero-Denominator Semantics:** When a metric has a zero denominator (e.g., zero predictions for a class yielding $TP + FP = 0$), precision, recall, and F1 are reported as `N/A` without a confidence interval, and are excluded from summary macro-averages.
- **Sample Size Floor:** Each evaluated Tier 1 error class and Tier 2 ambiguity type must contain a **minimum of 50 test cases** in the full benchmark profile to ensure statistical validity.
- **Reproducibility & Variance:**
  - Multi-seed pooled benchmarks run across seeds 1, 2, 3.
  - Model `gemini-3.8-flash` offers no sampling control (`temperature`, `top_p`, and `top_k` are deprecated/removed), so reproducibility is achieved via the deterministic on-disk response cache plus 3 independent reruns (`run_id` included in the cache key) with run-to-run classification agreement reported.
  - `thinking_level` is chosen on dev only and frozen before any test run.

---

## 4. Two-Tier Evaluation Architecture

Evaluation is structured into two non-overlapping tiers to maintain clear separation between core deterministic reconciliation integrity and nuanced text/policy reasoning:

### Tier 1: Deterministic Anomaly Classes (Hard Pass/Fail Floors)
Tier 1 evaluates the system's ability to catch unambiguous structural and financial discrepancies. The pass/fail benchmark criteria require **$\ge 98.0\%$ detection rate** and **zero false positives** on clean control data and hard negatives:

1. `DUPLICATE_PAYMENT`: Multiple captured payments for one order with identical amount.
2. `MISSING_SETTLEMENT`: Captured payment absent from recon items where expected settlement date + `missing_after_days` $\le$ `as_of`.
3. `FEE_TAX_MISMATCH`: Recon payment item where fee or tax differs from rate card calculation by $\ge \text{error\_threshold\_paise}$ (10 paise; Band 3).
4. `PARTIAL_REFUND_MISMATCH`: Recon refund debit $\ne$ ledger refund amount, or ledger status == "captured" with no refund entry.
5. `CUTOFF_DRIFT`: Ledger `booked_date` != IST date of payment capture.
6. `SHORT_BANK_CREDIT`: Bank credit amount $<$ settlement net amount for matching UTR.
7. `ORPHAN_BANK_CREDIT`: Bank credit with Razorpay narration matching no settlement batch.

### Tier 2: Ambiguity & Text Reasoning Tier (Informational Accuracy & Headroom)
Tier 2 evaluates policy boundary compliance, human escalation logic, and free-text comprehension. **Tier 2 metrics are never blended into Tier 1 pass/fail rates**; they are reported independently:

- **Tier-1 Exclusion Rule (Day 5b)**: Every prediction assigned to an ambiguity-tier label (regardless of predicted class or action) is scored **exclusively in the ambiguity tier** and is strictly excluded from Tier-1 TP, FP, and FN across all classes. For example, flagging an `EXPLAINED_INCONSISTENT` refund is the correct action in the ambiguity tier and does not incur a Tier-1 `PARTIAL_REFUND_MISMATCH` false positive. Conversely, spurious flags on hard negatives continue to generate Tier-1 false positives.
- **Reporting Requirement**: Reports must explicitly emit a separate summary line:
  `predictions assigned to ambiguity labels: n (correct action: k)`

1. `NEAR_TOLERANCE`: Fee or tax variance in the 3–9 paise escalation band ($3 \le |\Delta| \le 9$ paise). Evaluates whether the system properly abstains and routes to `escalate_human` rather than forcing an arbitrary binary decision.
2. `EXPLAINED_CONSISTENT`: Partial refund difference justified by free-text notes where stated deduction matches numerical ledger deficit. Evaluates ability to reconcile qualitative assertions with quantitative records (`benign_explained`).
3. `EXPLAINED_INCONSISTENT`: Partial refund where notes claim an explanation but the stated amount contradicts actual ledger arithmetic. Evaluates ability to catch deceptive or inaccurate textual claims (`flag_partial_refund_mismatch`).
4. `EXPLAINED_NO_AMOUNT`: Partial refund with qualitative text explanation but no explicit numeric figure. Evaluates ability to detect qualitative ambiguity and escalate to human review (`escalate_human`).

---

## 5. Control Groups and Hard Negatives

Systems are evaluated against legitimate operational variations that must NOT be flagged as errors:
- `ROUNDING`: Fee/tax variance bounded by $|\Delta| \le 2$ paise (`rounding_tolerance_paise: 2`; Band 1).
- `LEGIT_ADJUSTMENT`: Authorized fee dispute adjustments and manual gateway corrections.
- `LATE_SETTLEMENT`: Multi-day settlement shifts (due to weekends or holidays) where items and payout shift consistently.
- `RETRY_AFTER_FAIL`: Legitimate failed payment authorization followed by successful retry for the same order.
- `NOISY_BANK_NARRATION`: Casing irregularities, whitespace padding, or truncated UTR in bank statement.
- `NO_UTR_NARRATION`: Valid settlement credit without UTR in bank narration, matched via date and net amount.
- `UNRELATED_BANK_CREDIT`: External non-gateway credits (payroll, interest, other payment processors).
- `FAILED_PAYMENT`: Standard failed payment authorization attempt.
- `PARTIAL_REFUND_CONSISTENT`: Compliant partial refund matching internal order ledger.
- `IN_FLIGHT`: Captured payments and refunds whose scheduled settlement date falls after `as_of`. These transactions legitimately have no recon items, settlement batches, or bank credits and must be ignored by reconciliation systems (`benign_ignore`). Naive systems that omit `as_of` / settlement-lag awareness suffer massive false-positive rates on in-flight transactions (e.g. System B0 flags >93% of in-flight items, collapsing `MISSING_SETTLEMENT` precision to ~7%).

---

## 6. Adversarial Twin Dataset Benchmark

To assess resilience against indirect prompt injection, systems B2 and B3 are benchmarked on the twin `<split>/injected/` datasets:
- Measures degradation in detection accuracy compared to base dataset.
- Quantifies false negative rate induced by adversarial directives (`instruction_override`, `fake_authority`, `fake_evidence_ids`, `fake_numbers`, `json_payload`, `obfuscated_variant`).
- Measures verification rejection rate of the verifier layer in B3.

---

## 7. LLM Discrepancy Routing & Final-Action Mapping (B2 & B3)

The evaluation architecture enforces strict architectural separation between inference, caching, verification, and scoring:

```
client -> raw cache -> finalize (verifier_version) -> B2/B3 -> scoring
```

1. **Client Execution (`GeminiClient`)**: Calls the model and receives raw response text. The client does not validate or verify output and does not mutate responses.
2. **Raw Cache (`results/llm_cache/`)**: Stores exact raw strings with execution telemetry. Cache keys are immutable and content-hashed with SHA-256.
3. **Finalization (`scripts/finalize_runs.py` / `finalize()`)**: Loads cached raw outputs and maps them to deterministic actions under a specific `verifier_version` (SHA-256 of `verifier.py` + `schema.py`):
   - **System B2 (Unverified LLM)**: Validates JSON schema compliance. If valid, `b2_action` = raw decision; otherwise `escalate_human`.
   - **System B3 (Full Verified System)**: Validates JSON schema and runs rule-based checks V1–V7. If `ACCEPT`, `b3_action` = raw decision; otherwise `b3_action` = `escalate_human` with deterministic fallback explanation.
   - Results are written atomically to `results/finalized/<verifier_version>/finalized_<population>_run<run_id>.jsonl`.
4. **Scoring (`src/recon_agent/evaluation/`)**: Scores finalized B2 and B3 actions against ground truth labels.

### System B2 (Rules + Unverified LLM)
- The raw output from the LLM is validated strictly for JSON schema compliance (`V1`).
- If schema validation **passes**:
  - `final_action` = LLM decision (`flag`, `benign_explained`, or `escalate_human`).
  - `predicted_class` = LLM predicted class (or `None` for `benign_explained`).
- If schema validation **fails**:
  - Fail-safe fallback: `final_action` = `escalate_human`.

### System B3 (Full System with Verification Layer)
- The raw output from the LLM passes through both the JSON schema check and the deterministic rule-based verification layer (`V1` through `V7`).
- The verifier emits a `VerifierOutcome` of `ACCEPT`, `REJECT`, or `ABSTAIN`:
  - **`ACCEPT`**: All grounding, verbatim quote, amount consistency, number fabrication, and arithmetic checks succeed.
    - `final_action` = LLM decision (`flag`, `benign_explained`, or `escalate_human`).
    - `predicted_class` = LLM predicted class.
  - **`REJECT`**: A check failed (schema invalid, ungrounded reference, fabricated quote/number, arithmetic mismatch).
    - Fail-safe fallback: `final_action` = `escalate_human`.
  - **`ABSTAIN`**: The verifier cannot reliably confirm (e.g. multiple amount tokens in quote, ambiguous phrasing, non-numeric number words).
    - Fail-safe fallback: `final_action` = `escalate_human`.

