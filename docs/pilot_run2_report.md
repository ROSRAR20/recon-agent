================================================================================
DAY 7b-5 PILOT EVALUATION REPORT
================================================================================
Environment       : Python 3.13.0 on Windows | tzdata OK | Commit a2253fe6 (DIRTY)
Registration-Final: f09646f5283791344ee6b9ef931b3b57e83ade43
Frozen Baselines  : PASS
Frozen File Hashes (HEAD):
  independent_checker.py: 5def01030115807749e77f0375e93a4e59ef9f35
  verifier.py           : 74412e3d7e991366eb6afe04ff008546daf0ff71
  docs/preregistration.md: a690d23686806bac3899a9b389561e02d2fea7a2
  systems/ (tree)       : 3d7c50658ddefd38f3bf4e21a4276a6c32ed235a
Git Commit        : a2253fe691ca2d7e4877ea62b5ef030edcbad0ed (DIRTY)
System Prompt Hash: a92e1b8026684afc584502fb2d7853a301348f6ef400708da18bbaff5eafd52f
Verifier Version  : 568c7f01c13d1df49a1bf8c13fd36f813a089f0e829ad6685d4ecd200e76f1f6
Requested Model   : gemini-3.5-flash-lite
Served Model(s)   : gemini-3.5-flash-lite
Run ID(s)         : [0]
Completed Cases   : 120/120 across 1 evaluated arm(s)
Infra-Failure Rate: 0.0%
================================================================================

### Arm: `default` (thinking_level: None)
- Completed: 120/120 | Infra-failed: 0 (0.0%)
- Cost: $0.0471 (UNVERIFIED RATE) | Latency p50: 9402.0 ms, p95: 17169.0 ms | Avg tokens: 4836.4

| Metric | B3 (Verified) | B2 (Unverified) | B1r Reference | Always-Escalate |
| :--- | :--- | :--- | :--- | :--- |
| **Schema-Valid Rate** | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | N/A | 100.0% |
| **Decision Accuracy** | 114/120 (95.0%, 95% CI: [89.5%, 97.7%]) | 117/120 (97.5%, 95% CI: [92.9%, 99.2%]) | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | 51/120 (42.5%, 95% CI: [34.0%, 51.4%]) |
| **Unsafe Resolution Rate** | **0/120 (0.0%, 95% CI: [0.0%, 3.1%])** | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) |
| **False-Flag Rate (Controls)** | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) |
| **Over-Escalation Rate** | 3/45 (6.7%, 95% CI: [2.3%, 17.9%]) | 1/45 (2.2%, 95% CI: [0.4%, 11.6%]) | 0/45 (0.0%, 95% CI: [0.0%, 7.9%]) | 45/45 (100.0%, 95% CI: [92.1%, 100.0%]) |
| **Hallucination Rate (Schema-Valid)** | 2/120 (1.7%, 95% CI: [0.5%, 5.9%]) | 2/120 (1.7%, 95% CI: [0.5%, 5.9%]) | N/A | 0/N (0.0%) |
| &nbsp;&nbsp;&bull; Rate (All Outputs) | 2/120 (1.7%, 95% CI: [0.5%, 5.9%]) | 2/120 (1.7%, 95% CI: [0.5%, 5.9%]) | N/A | 0/N (0.0%) |
| &nbsp;&nbsp;&bull; F1 (Evidence) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F2 (Quote) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F3 (Cited Amount) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F4 (Explanation Number) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | N/A | 0/N |
| **Reflection Rate (>=6 words)** | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | N/A | 0/N |
| **Verifier Block Rate (Hallucinated)** | 2/2 (100.0%, 95% CI: [34.2%, 100.0%]) | N/A | N/A | N/A |
| **Verifier Block Rate (Unsafe)** | 0/0 (N/A) | N/A | N/A | N/A |
| **B3 Accepted Coverage** | 116/120 (96.7%, 95% CI: [91.7%, 98.7%]) | N/A | N/A | N/A |
| **B3 Accepted Accuracy** | 114/116 (98.3%, 95% CI: [93.9%, 99.5%]) | N/A | N/A | N/A |

#### Verifier-Only Rejections (False Rejections)
- **Overall False-Reject Rate**: 2/65 (3.1%, 95% CI: [0.9%, 10.5%]) of faithful correct non-escalations (2/120 (1.7%, 95% CI: [0.5%, 5.9%]) of all completed)
- **Top False-Reject Reason Codes**: `NO_AMOUNT_IN_QUOTE` (2)
- **By Check Family**:
  - `bank`: 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) (0/9 (0.0%, 95% CI: [0.0%, 29.9%]) overall) | Top: None
  - `fee_tax`: 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) (0/32 (0.0%, 95% CI: [0.0%, 10.7%]) overall) | Top: None
  - `refund`: 2/51 (3.9%, 95% CI: [1.1%, 13.2%]) (2/79 (2.5%, 95% CI: [0.7%, 8.8%]) overall) | Top: `NO_AMOUNT_IN_QUOTE` (2)

#### B3 Non-ACCEPT Outcomes by Reason Code
| Reason Code | Hallucinated (Checker) | Not Hallucinated (Faithful) | Total |
| :--- | :---: | :---: | :---: |
| `NO_AMOUNT_IN_QUOTE` | 1 | 2 | 3 |
| `FABRICATED_NUMBER_IN_EXPLANATION` | 1 | 0 | 1 |

#### Paired B2 vs B3 Analysis (Cost vs Benefit of Verification)
| Category | Cases (k/n) | Percentage | Interpretation |
| :--- | :---: | :---: | :--- |
| **Cost** | 3/120 | 2.5% | B2 was correct, B3 escalated (false reject) |
| **Benefit** | 0/120 | 0.0% | B2 was unsafe, B3 safe (prevented unsafe resolution) |
| **Unchanged** | 117/120 | 97.5% | B2 and B3 took equivalent actions |

#### Breakdown by Check Family (B3 System)
| Check Family | Total | Accuracy | Unsafe Res | False-Flag (Ctrl) | Over-Escalation |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `bank` | 9 | 9/9 (100.0%, 95% CI: [70.1%, 100.0%]) | 0/9 (0.0%, 95% CI: [0.0%, 29.9%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) |
| `fee_tax` | 32 | 32/32 (100.0%, 95% CI: [89.3%, 100.0%]) | 0/32 (0.0%, 95% CI: [0.0%, 10.7%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) |
| `refund` | 79 | 73/79 (92.4%, 95% CI: [84.4%, 96.5%]) | 0/79 (0.0%, 95% CI: [0.0%, 4.6%]) | 0/6 (0.0%, 95% CI: [0.0%, 39.0%]) | 3/31 (9.7%, 95% CI: [3.4%, 24.9%]) |


## Pre-Registered Selection Logic

| Arm | Thinking Level | Schema-Valid (>=95%) | B3 Unsafe Res (k/n, 95% CI) | Latency p50 | Eligibility / Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `default` | `None` | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | **0/120 (0.0%, 95% CI: [0.0%, 3.1%])** | 9402.0 ms | ✅ ELIGIBLE |

### Suggested Arm for Freeze
> **`default`** (SUGGESTION ONLY)
> *Reason*: Lowest B3 unsafe resolution count (0/120) with non-overlapping CI.

