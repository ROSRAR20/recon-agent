================================================================================
DAY 7b-5 PILOT EVALUATION REPORT
================================================================================
Environment       : Python 3.13.0 on Windows | tzdata OK | Commit 480d120c (CLEAN)
Registration-Final: f09646f5283791344ee6b9ef931b3b57e83ade43
Frozen Baselines  : PASS
Frozen File Hashes (HEAD):
  independent_checker.py: 5def01030115807749e77f0375e93a4e59ef9f35
  verifier.py           : 06929945500e4818b38ce07e58aa8a0c2915556e
  docs/preregistration.md: a690d23686806bac3899a9b389561e02d2fea7a2
  systems/ (tree)       : 3d7c50658ddefd38f3bf4e21a4276a6c32ed235a
Git Commit        : 480d120c5fed2a671122451c66a62a2274df34ac (CLEAN)
System Prompt Hash: 20026d1fd2cee1efd10abe9e7e5c9815406f3547d422a612156db5dee1fa4434
Verifier Version  : bd2a4e01e29520a8c19669402abbe414c23413f31bfa61a56745dc65c562185d
Requested Model   : gemini-3.5-flash-lite
Served Model(s)   : gemini-3.5-flash-lite
Run ID(s)         : [0]
Completed Cases   : 360/360 across 3 evaluated arm(s)
Infra-Failure Rate: 0.0%
================================================================================

### Arm: `default` (thinking_level: None)
- Completed: 120/120 | Infra-failed: 0 (0.0%)
- Cost: $0.0406 (UNVERIFIED RATE) | Latency p50: 3063.0 ms, p95: 29349.0 ms | Avg tokens: 4091.3

| Metric | B3 (Verified) | B2 (Unverified) | B1r Reference | Always-Escalate |
| :--- | :--- | :--- | :--- | :--- |
| **Schema-Valid Rate** | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | N/A | 100.0% |
| **Decision Accuracy** | 93/120 (77.5%, 95% CI: [69.2%, 84.0%]) | 91/120 (75.8%, 95% CI: [67.5%, 82.6%]) | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | 51/120 (42.5%, 95% CI: [34.0%, 51.4%]) |
| **Unsafe Resolution Rate** | **0/120 (0.0%, 95% CI: [0.0%, 3.1%])** | 9/120 (7.5%, 95% CI: [4.0%, 13.6%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) |
| **False-Flag Rate (Controls)** | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) |
| **Over-Escalation Rate** | 3/45 (6.7%, 95% CI: [2.3%, 17.9%]) | 0/45 (0.0%, 95% CI: [0.0%, 7.9%]) | 0/45 (0.0%, 95% CI: [0.0%, 7.9%]) | 45/45 (100.0%, 95% CI: [92.1%, 100.0%]) |
| **Hallucination Rate (Schema-Valid)** | 23/120 (19.2%, 95% CI: [13.1%, 27.1%]) | 23/120 (19.2%, 95% CI: [13.1%, 27.1%]) | N/A | 0/N (0.0%) |
| &nbsp;&nbsp;&bull; Rate (All Outputs) | 23/120 (19.2%, 95% CI: [13.1%, 27.1%]) | 23/120 (19.2%, 95% CI: [13.1%, 27.1%]) | N/A | 0/N (0.0%) |
| &nbsp;&nbsp;&bull; F1 (Evidence) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F2 (Quote) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F3 (Cited Amount) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F4 (Explanation Number) | 21/120 (17.5%, 95% CI: [11.7%, 25.3%]) | 21/120 (17.5%, 95% CI: [11.7%, 25.3%]) | N/A | 0/N |
| **Reflection Rate (>=6 words)** | 3/120 (2.5%, 95% CI: [0.9%, 7.1%]) | 3/120 (2.5%, 95% CI: [0.9%, 7.1%]) | N/A | 0/N |
| **Verifier Block Rate (Hallucinated)** | 22/23 (95.7%, 95% CI: [79.0%, 99.2%]) | N/A | N/A | N/A |
| **Verifier Block Rate (Unsafe)** | 9/9 (100.0%, 95% CI: [70.1%, 100.0%]) | N/A | N/A | N/A |
| **B3 Accepted Coverage** | 87/120 (72.5%, 95% CI: [63.9%, 79.7%]) | N/A | N/A | N/A |
| **B3 Accepted Accuracy** | 85/87 (97.7%, 95% CI: [92.0%, 99.4%]) | N/A | N/A | N/A |

#### Verifier-Only Rejections (False Rejections)
- **Overall False-Reject Rate**: 3/45 (6.7%, 95% CI: [2.3%, 17.9%]) of faithful correct non-escalations (3/120 (2.5%, 95% CI: [0.9%, 7.1%]) of all completed)
- **Top False-Reject Reason Codes**: `NO_AMOUNT_IN_QUOTE` (2), `REFLECTED_UNTRUSTED_TEXT` (1)
- **By Check Family**:
  - `bank`: 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) (0/9 (0.0%, 95% CI: [0.0%, 29.9%]) overall) | Top: None
  - `fee_tax`: 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) (0/32 (0.0%, 95% CI: [0.0%, 10.7%]) overall) | Top: None
  - `refund`: 3/31 (9.7%, 95% CI: [3.4%, 24.9%]) (3/79 (3.8%, 95% CI: [1.3%, 10.6%]) overall) | Top: `NO_AMOUNT_IN_QUOTE` (2), `REFLECTED_UNTRUSTED_TEXT` (1)

#### B3 Non-ACCEPT Outcomes by Reason Code
| Reason Code | Hallucinated (Checker) | Not Hallucinated (Faithful) | Total |
| :--- | :---: | :---: | :---: |
| `FABRICATED_NUMBER_IN_EXPLANATION` | 20 | 0 | 20 |
| `NO_AMOUNT_IN_QUOTE` | 0 | 10 | 10 |
| `FABRICATED_QUOTE` | 1 | 0 | 1 |
| `REFLECTED_UNTRUSTED_TEXT` | 0 | 1 | 1 |
| `UNGROUNDED_AMOUNT` | 1 | 0 | 1 |

#### Paired B2 vs B3 Analysis (Cost vs Benefit of Verification)
| Category | Cases (k/n) | Percentage | Interpretation |
| :--- | :---: | :---: | :--- |
| **Cost** | 6/120 | 5.0% | B2 was correct, B3 escalated (false reject) |
| **Benefit** | 9/120 | 7.5% | B2 was unsafe, B3 safe (prevented unsafe resolution) |
| **Unchanged** | 105/120 | 87.5% | B2 and B3 took equivalent actions |

#### Breakdown by Check Family (B3 System)
| Check Family | Total | Accuracy | Unsafe Res | False-Flag (Ctrl) | Over-Escalation |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `bank` | 9 | 9/9 (100.0%, 95% CI: [70.1%, 100.0%]) | 0/9 (0.0%, 95% CI: [0.0%, 29.9%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) |
| `fee_tax` | 32 | 32/32 (100.0%, 95% CI: [89.3%, 100.0%]) | 0/32 (0.0%, 95% CI: [0.0%, 10.7%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) |
| `refund` | 79 | 52/79 (65.8%, 95% CI: [54.9%, 75.3%]) | 0/79 (0.0%, 95% CI: [0.0%, 4.6%]) | 0/6 (0.0%, 95% CI: [0.0%, 39.0%]) | 3/31 (9.7%, 95% CI: [3.4%, 24.9%]) |


### Arm: `low` (thinking_level: low)
- Completed: 120/120 | Infra-failed: 0 (0.0%)
- Cost: $0.0462 (UNVERIFIED RATE) | Latency p50: 3242.0 ms, p95: 6922.0 ms | Avg tokens: 4247.8

| Metric | B3 (Verified) | B2 (Unverified) | B1r Reference | Always-Escalate |
| :--- | :--- | :--- | :--- | :--- |
| **Schema-Valid Rate** | 119/120 (99.2%, 95% CI: [95.4%, 99.9%]) | 119/120 (99.2%, 95% CI: [95.4%, 99.9%]) | N/A | 100.0% |
| **Decision Accuracy** | 87/120 (72.5%, 95% CI: [63.9%, 79.7%]) | 90/120 (75.0%, 95% CI: [66.6%, 81.9%]) | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | 51/120 (42.5%, 95% CI: [34.0%, 51.4%]) |
| **Unsafe Resolution Rate** | **0/120 (0.0%, 95% CI: [0.0%, 3.1%])** | 9/120 (7.5%, 95% CI: [4.0%, 13.6%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) |
| **False-Flag Rate (Controls)** | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) |
| **Over-Escalation Rate** | 8/45 (17.8%, 95% CI: [9.3%, 31.3%]) | 0/45 (0.0%, 95% CI: [0.0%, 7.9%]) | 0/45 (0.0%, 95% CI: [0.0%, 7.9%]) | 45/45 (100.0%, 95% CI: [92.1%, 100.0%]) |
| **Hallucination Rate (Schema-Valid)** | 22/119 (18.5%, 95% CI: [12.5%, 26.4%]) | 22/119 (18.5%, 95% CI: [12.5%, 26.4%]) | N/A | 0/N (0.0%) |
| &nbsp;&nbsp;&bull; Rate (All Outputs) | 22/120 (18.3%, 95% CI: [12.4%, 26.2%]) | 22/120 (18.3%, 95% CI: [12.4%, 26.2%]) | N/A | 0/N (0.0%) |
| &nbsp;&nbsp;&bull; F1 (Evidence) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F2 (Quote) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | 1/120 (0.8%, 95% CI: [0.1%, 4.6%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F3 (Cited Amount) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F4 (Explanation Number) | 21/120 (17.5%, 95% CI: [11.7%, 25.3%]) | 21/120 (17.5%, 95% CI: [11.7%, 25.3%]) | N/A | 0/N |
| **Reflection Rate (>=6 words)** | 10/120 (8.3%, 95% CI: [4.6%, 14.7%]) | 10/120 (8.3%, 95% CI: [4.6%, 14.7%]) | N/A | 0/N |
| **Verifier Block Rate (Hallucinated)** | 22/22 (100.0%, 95% CI: [85.1%, 100.0%]) | N/A | N/A | N/A |
| **Verifier Block Rate (Unsafe)** | 9/9 (100.0%, 95% CI: [70.1%, 100.0%]) | N/A | N/A | N/A |
| **B3 Accepted Coverage** | 81/120 (67.5%, 95% CI: [58.7%, 75.2%]) | N/A | N/A | N/A |
| **B3 Accepted Accuracy** | 79/81 (97.5%, 95% CI: [91.4%, 99.3%]) | N/A | N/A | N/A |

#### Verifier-Only Rejections (False Rejections)
- **Overall False-Reject Rate**: 8/45 (17.8%, 95% CI: [9.3%, 31.3%]) of faithful correct non-escalations (8/120 (6.7%, 95% CI: [3.4%, 12.6%]) of all completed)
- **Top False-Reject Reason Codes**: `REFLECTED_UNTRUSTED_TEXT` (6), `NO_AMOUNT_IN_QUOTE` (2)
- **By Check Family**:
  - `bank`: 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) (0/9 (0.0%, 95% CI: [0.0%, 29.9%]) overall) | Top: None
  - `fee_tax`: 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) (0/32 (0.0%, 95% CI: [0.0%, 10.7%]) overall) | Top: None
  - `refund`: 8/31 (25.8%, 95% CI: [13.7%, 43.2%]) (8/79 (10.1%, 95% CI: [5.2%, 18.7%]) overall) | Top: `REFLECTED_UNTRUSTED_TEXT` (6), `NO_AMOUNT_IN_QUOTE` (2)

#### B3 Non-ACCEPT Outcomes by Reason Code
| Reason Code | Hallucinated (Checker) | Not Hallucinated (Faithful) | Total |
| :--- | :---: | :---: | :---: |
| `FABRICATED_NUMBER_IN_EXPLANATION` | 20 | 0 | 20 |
| `NO_AMOUNT_IN_QUOTE` | 0 | 9 | 9 |
| `REFLECTED_UNTRUSTED_TEXT` | 0 | 7 | 7 |
| `AMOUNT_MISMATCH_WITH_QUOTE` | 0 | 1 | 1 |
| `FABRICATED_QUOTE` | 1 | 0 | 1 |
| `schema_invalid` | 1 | 0 | 1 |

#### Paired B2 vs B3 Analysis (Cost vs Benefit of Verification)
| Category | Cases (k/n) | Percentage | Interpretation |
| :--- | :---: | :---: | :--- |
| **Cost** | 10/120 | 8.3% | B2 was correct, B3 escalated (false reject) |
| **Benefit** | 9/120 | 7.5% | B2 was unsafe, B3 safe (prevented unsafe resolution) |
| **Unchanged** | 101/120 | 84.2% | B2 and B3 took equivalent actions |

#### Breakdown by Check Family (B3 System)
| Check Family | Total | Accuracy | Unsafe Res | False-Flag (Ctrl) | Over-Escalation |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `bank` | 9 | 9/9 (100.0%, 95% CI: [70.1%, 100.0%]) | 0/9 (0.0%, 95% CI: [0.0%, 29.9%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) |
| `fee_tax` | 32 | 32/32 (100.0%, 95% CI: [89.3%, 100.0%]) | 0/32 (0.0%, 95% CI: [0.0%, 10.7%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) |
| `refund` | 79 | 46/79 (58.2%, 95% CI: [47.2%, 68.5%]) | 0/79 (0.0%, 95% CI: [0.0%, 4.6%]) | 0/6 (0.0%, 95% CI: [0.0%, 39.0%]) | 8/31 (25.8%, 95% CI: [13.7%, 43.2%]) |


### Arm: `high` (thinking_level: high)
- Completed: 120/120 | Infra-failed: 0 (0.0%)
- Cost: $0.0921 (UNVERIFIED RATE) | Latency p50: 5802.0 ms, p95: 7283.0 ms | Avg tokens: 5520.1

| Metric | B3 (Verified) | B2 (Unverified) | B1r Reference | Always-Escalate |
| :--- | :--- | :--- | :--- | :--- |
| **Schema-Valid Rate** | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | N/A | 100.0% |
| **Decision Accuracy** | 91/120 (75.8%, 95% CI: [67.5%, 82.6%]) | 96/120 (80.0%, 95% CI: [72.0%, 86.2%]) | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | 51/120 (42.5%, 95% CI: [34.0%, 51.4%]) |
| **Unsafe Resolution Rate** | **0/120 (0.0%, 95% CI: [0.0%, 3.1%])** | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) |
| **False-Flag Rate (Controls)** | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) | 0/20 (0.0%, 95% CI: [0.0%, 16.1%]) |
| **Over-Escalation Rate** | 6/45 (13.3%, 95% CI: [6.3%, 26.2%]) | 2/45 (4.4%, 95% CI: [1.2%, 14.8%]) | 0/45 (0.0%, 95% CI: [0.0%, 7.9%]) | 45/45 (100.0%, 95% CI: [92.1%, 100.0%]) |
| **Hallucination Rate (Schema-Valid)** | 22/120 (18.3%, 95% CI: [12.4%, 26.2%]) | 22/120 (18.3%, 95% CI: [12.4%, 26.2%]) | N/A | 0/N (0.0%) |
| &nbsp;&nbsp;&bull; Rate (All Outputs) | 22/120 (18.3%, 95% CI: [12.4%, 26.2%]) | 22/120 (18.3%, 95% CI: [12.4%, 26.2%]) | N/A | 0/N (0.0%) |
| &nbsp;&nbsp;&bull; F1 (Evidence) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F2 (Quote) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F3 (Cited Amount) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | 0/120 (0.0%, 95% CI: [0.0%, 3.1%]) | N/A | 0/N |
| &nbsp;&nbsp;&bull; F4 (Explanation Number) | 22/120 (18.3%, 95% CI: [12.4%, 26.2%]) | 22/120 (18.3%, 95% CI: [12.4%, 26.2%]) | N/A | 0/N |
| **Reflection Rate (>=6 words)** | 4/120 (3.3%, 95% CI: [1.3%, 8.3%]) | 4/120 (3.3%, 95% CI: [1.3%, 8.3%]) | N/A | 0/N |
| **Verifier Block Rate (Hallucinated)** | 22/22 (100.0%, 95% CI: [85.1%, 100.0%]) | N/A | N/A | N/A |
| **Verifier Block Rate (Unsafe)** | 0/0 (N/A) | N/A | N/A | N/A |
| **B3 Accepted Coverage** | 94/120 (78.3%, 95% CI: [70.2%, 84.8%]) | N/A | N/A | N/A |
| **B3 Accepted Accuracy** | 91/94 (96.8%, 95% CI: [91.0%, 98.9%]) | N/A | N/A | N/A |

#### Verifier-Only Rejections (False Rejections)
- **Overall False-Reject Rate**: 4/44 (9.1%, 95% CI: [3.6%, 21.2%]) of faithful correct non-escalations (4/120 (3.3%, 95% CI: [1.3%, 8.3%]) of all completed)
- **Top False-Reject Reason Codes**: `REFLECTED_UNTRUSTED_TEXT` (4)
- **By Check Family**:
  - `bank`: 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) (0/9 (0.0%, 95% CI: [0.0%, 29.9%]) overall) | Top: None
  - `fee_tax`: 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) (0/32 (0.0%, 95% CI: [0.0%, 10.7%]) overall) | Top: None
  - `refund`: 4/30 (13.3%, 95% CI: [5.3%, 29.7%]) (4/79 (5.1%, 95% CI: [2.0%, 12.3%]) overall) | Top: `REFLECTED_UNTRUSTED_TEXT` (4)

#### B3 Non-ACCEPT Outcomes by Reason Code
| Reason Code | Hallucinated (Checker) | Not Hallucinated (Faithful) | Total |
| :--- | :---: | :---: | :---: |
| `FABRICATED_NUMBER_IN_EXPLANATION` | 22 | 0 | 22 |
| `REFLECTED_UNTRUSTED_TEXT` | 0 | 4 | 4 |

#### Paired B2 vs B3 Analysis (Cost vs Benefit of Verification)
| Category | Cases (k/n) | Percentage | Interpretation |
| :--- | :---: | :---: | :--- |
| **Cost** | 5/120 | 4.2% | B2 was correct, B3 escalated (false reject) |
| **Benefit** | 0/120 | 0.0% | B2 was unsafe, B3 safe (prevented unsafe resolution) |
| **Unchanged** | 115/120 | 95.8% | B2 and B3 took equivalent actions |

#### Breakdown by Check Family (B3 System)
| Check Family | Total | Accuracy | Unsafe Res | False-Flag (Ctrl) | Over-Escalation |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `bank` | 9 | 9/9 (100.0%, 95% CI: [70.1%, 100.0%]) | 0/9 (0.0%, 95% CI: [0.0%, 29.9%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) |
| `fee_tax` | 32 | 32/32 (100.0%, 95% CI: [89.3%, 100.0%]) | 0/32 (0.0%, 95% CI: [0.0%, 10.7%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) | 0/7 (0.0%, 95% CI: [0.0%, 35.4%]) |
| `refund` | 79 | 50/79 (63.3%, 95% CI: [52.3%, 73.1%]) | 0/79 (0.0%, 95% CI: [0.0%, 4.6%]) | 0/6 (0.0%, 95% CI: [0.0%, 39.0%]) | 6/31 (19.4%, 95% CI: [9.2%, 36.3%]) |


## Pre-Registered Selection Logic

| Arm | Thinking Level | Schema-Valid (>=95%) | B3 Unsafe Res (k/n, 95% CI) | Latency p50 | Eligibility / Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `default` | `None` | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | **0/120 (0.0%, 95% CI: [0.0%, 3.1%])** | 3063.0 ms | ✅ ELIGIBLE |
| `low` | `low` | 119/120 (99.2%, 95% CI: [95.4%, 99.9%]) | **0/120 (0.0%, 95% CI: [0.0%, 3.1%])** | 3242.0 ms | ✅ ELIGIBLE |
| `high` | `high` | 120/120 (100.0%, 95% CI: [96.9%, 100.0%]) | **0/120 (0.0%, 95% CI: [0.0%, 3.1%])** | 5802.0 ms | ✅ ELIGIBLE |

### Suggested Arm for Freeze
> **`default`** (SUGGESTION ONLY)
> *Reason*: Tied on B3 unsafe resolution rate within sampling error among ['default', 'low', 'high']. Selected arm `default` with lower p50 latency (3063.0 ms).

