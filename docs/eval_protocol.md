# Evaluation Protocol: recon-agent

This document defines the formal evaluation methodology, baseline hierarchy, data-split hygiene, and reporting requirements for benchmarking reconciliation accuracy and verifier efficacy.

---

## 1. Systems Compared (Baselines & Ablations)

To rigorously isolate the contribution of each architectural component, all evaluations compare four distinct configurations on identical transaction sets:

| System ID | Name | Ingestion & Rules | LLM Residual Reasoning | Verification Layer | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B0** | Baseline Exact-ID | Exact ID matching only | None | None | Matches records solely if transaction ID (`pay_...`, `order_...`) is identical in gateway and internal ledger. |
| **B1** | Deterministic Rules | Extended rules & invariants | None | None | Matches on secondary keys, amount matching (in integer paise), timestamp windows, and fee calculations without any LLM. |
| **B2** | Rules + Unverified LLM | Extended rules & invariants | Active for residual cases | None | Residual exceptions are routed to an LLM. Model explanations and resolutions are accepted without algorithmic verification. |
| **B3** | Full System (Verified) | Extended rules & invariants | Active for residual cases | **Active** | Residual exceptions are reasoned by LLM but passed through the independent verification layer before acceptance. Failures escalate to human queue. |

---

## 2. Held-Out Test Set Protocols

To prevent data contamination and overfitting:

1. **Independent Seed & Mix:** The held-out test suite is generated using a distinct pseudorandom seed and a distinct injected error distribution from the development set.
2. **Strict Embargo:** The held-out test suite is **never opened, inspected, or executed before Day 12** of the 14-day development cycle.
3. **Single Execution Run:** The final benchmark on the held-out dataset is executed **exactly once** to prevent iterative tuning against the test distribution.

---

## 3. Statistical Power and Reporting Standards

- **Raw Counts Mandatory:** All metrics must be reported with explicit integer counts ($k/n$), not solely percentages (e.g., `"49/50 (98.0%)"` instead of `"98%"`).
- **Sample Size Floor:** Each evaluated error class (e.g., fee discrepancy, settlement delay, missing order, partial refund) must contain a **minimum of 50 test cases** to ensure statistical validity.
- **Reproducibility & Variance:**
  - All LLM evaluations require **3 complete reruns at temperature 0** using a pinned model snapshot.
  - Run-to-run classification consistency and variance must be documented across all three iterations.

---

## 4. Error Classes Under Evaluation

1. **Exact Matches (Control Group):** Standard 1-to-1 settlements and payments.
2. **Benign Hard Negatives:** Multi-day settlement lag due to bank holidays or weekends, compliant fee rounding [DOCS: https://razorpay.com/docs/api/settlements/].
3. **Fee/Tax Discrepancies:** Injected differences between expected merchant fee schedule and deducted gateway fee in paise.
4. **Missing Gateway Transactions:** Internal ledger records with no corresponding payment/settlement in the gateway.
5. **Missing Ledger Entries:** Gateway charges with no corresponding internal order record.
6. **Multi-Part / Partial Refunds:** Complex split refund allocations [DOCS: https://razorpay.com/docs/api/refunds/].
