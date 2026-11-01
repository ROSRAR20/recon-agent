# Reconciliation System B1 & B1r Rules Specification

This document specifies the deterministic rule engine (**B1**) and the regex-enhanced text parser (**B1r**) for the `recon-agent` pipeline.

---

## 1. Reconciled Constants & Rate Card Parameters

All numerical thresholds and policy parameters are read dynamically at runtime from `rate_card.json` (via `ReconDataset.rate_card`).

| Parameter | Value | Source / Tag | Description |
|:---|:---:|:---:|:---|
| `card_bps` | 200 | `[POLICY]` | Card payment fee rate (2.00%, 200 basis points) |
| `upi_bps` | 0 | `[POLICY]` | UPI payment fee rate (0.00%, 0 basis points) |
| `netbanking_bps` | 150 | `[POLICY]` | Netbanking fee rate (1.50%, 150 basis points) *(reconciled: plan draft erroneously stated 180; actual rate card and generator use 150)* |
| `gst_bps` | 1800 | `[POLICY]` | Goods & Services Tax rate (18.00%, 1800 basis points) |
| `fee_includes_tax` | `true` | `[ASSUMED]` | Merchant pricing model: fee quoted includes 18% GST component |
| `payment_settlement_lag_days` | 2 | `[POLICY]` | T+2 working days settlement cycle for captured payments |
| `refund_settlement_lag_days` | 1 | `[POLICY]` | T+1 working days settlement cycle for processed refunds *(reconciled: plan draft erroneously stated 3; actual rate card and generator use 1)* |
| `missing_after_days` | 7 | `[POLICY]` | Calendar-day grace period past scheduled settlement before flagging missing payout |
| `late_grace_working_days` | 2 | `[POLICY]` | Working-day grace window allowed for late settlement payouts |
| `rounding_tolerance_paise` | 2 | `[ASSUMED]` | Maximum deviation in paise considered benign floating-point/tax rounding error |
| `escalate_tolerance_paise` | 9 | `[ASSUMED]` | Maximum deviation in paise for near-tolerance human escalation |
| `error_threshold_paise` | 10 | `[ASSUMED]` | Minimum deviation in paise classified as a definitive fee/tax error |
| `t2_min_utr_length` | 10 | `[POLICY]` | Minimum length of truncated UTR required for Tier 2 normalized matching |

---

## 2. Bank-to-Settlement Matching Protocol (One-to-One Tiers)

Matching proceeds in four strict sequential tiers across bank statement credit entries (`credit > 0`) and settlement batches for each merchant. Once an entity is matched, it is removed from candidate pools for lower tiers.

- **Tier 1 (T1: Exact UTR Substring)**:
  - Exact case-sensitive substring match: `settlement.utr in bank_row.description`.
  - Same merchant, credit $> 0$.
- **Tier 2 (T2: Normalized & Truncated UTR)**:
  - Punctuation-insensitive and space-insensitive matching: alphanumeric characters only, case-folded.
  - Handles truncated UTRs ($\ge 10$ characters) produced by bank truncation.
  - Candidate match must be unique within the merchant and settlement window. If 2 or more candidate bank rows or settlements match, the system escalates to human (`action="escalate_human"`, `rule_id="B1_BANK_AMBIGUOUS_CANDIDATES"`), never guessing.
- **Tier 3 (T3: No-UTR Exact Amount Match)**:
  - Bank row has Razorpay-style narration (`"razorpay"` in description, case-insensitive).
  - Bank date is within $[settle\_date, settle\_date + late\_grace\_working\_days]$.
  - Exact net match: `bank_row.credit == settlement.amount`.
  - If unique, link them; if ambiguous ($\ge 2$ candidates), escalate to human.
- **Tier 4 (T4: No-UTR Short Credit Match)**:
  - Remaining unmatched Razorpay-style bank credits in the same date window with `bank_row.credit < settlement.amount`.
  - Allowed if and only if there is exactly 1 candidate settlement and 1 candidate bank row on that merchant-window.
  - If ambiguous ($\ge 2$ candidates), escalate to human.
- **Post-Matching Residual Handling**:
  - Remaining unmatched Razorpay-style bank credit $\rightarrow$ flag `ORPHAN_BANK_CREDIT` (`action="flag"`).
  - Non-Razorpay bank credits (rent, payroll, other processor payouts) $\rightarrow$ never flag, never escalate.
  - Remaining unmatched settlement past grace window ($as\_of > settle\_date + late\_grace\_working\_days$) $\rightarrow$ escalate `UNKNOWN` (`action="escalate_human"`).

---

## 3. Deterministic Anomaly Detectors (System B1)

Each prediction carries a distinct `rule_id`:

1. **`B1_DUPLICATE_PAYMENT`**:
   - Multiple captured payments with identical `(merchant_id, order_id, amount)`.
   - The earliest payment by `(created_at, payment_id)` is accepted; all subsequent ones are flagged `action="flag"`.
2. **`B1_MISSING_SETTLEMENT`**:
   - Captured payment with no recon item AND $as\_of > scheduled\_settle\_date + missing\_after\_days$.
   - Payments within the settlement lag or grace window are considered in-flight and are never flagged.
3. **`B1_FEE_TAX_MISMATCH` & `B1_NEAR_TOLERANCE`**:
   - Expected fee and tax computed with integer half-up rounding:
     $$\text{base\_fee} = \lfloor(\text{amount} \times \text{bps} + 5000) / 10000\rfloor$$
     $$\text{expected\_tax} = \lfloor(\text{base\_fee} \times \text{gst\_bps} + 5000) / 10000\rfloor$$
     $$\text{expected\_fee} = \text{base\_fee} + \text{expected\_tax} \quad (\text{if fee\_includes\_tax else base\_fee})$$
   - $\Delta = \max(|\text{actual\_fee} - \text{expected\_fee}|, |\text{actual\_tax} - \text{expected\_tax}|)$.
   - $\Delta \le \text{rounding\_tolerance\_paise}$ (2 paise): benign rounding, ignore.
   - $3 \le \Delta \le \text{escalate\_tolerance\_paise}$ (9 paise): `B1_NEAR_TOLERANCE`, `action="escalate_human"`.
   - $\Delta \ge \text{error\_threshold\_paise}$ (10 paise): `B1_FEE_TAX_MISMATCH`, `action="flag"`.
4. **`B1_PARTIAL_REFUND_MISMATCH`**:
   - Refund recon item debit differs from `order_ledger.refund_amount` by $> \text{rounding\_tolerance\_paise}$ (2 paise), or order has no refund entry.
   - In B1 (pure arithmetic): flagged with `action="flag"`.
5. **`B1_CUTOFF_DRIFT`**:
   - Captured payment where order `booked_date` differs from the capture date in IST timezone.
   - Flagged with `action="flag"`, `entity_ids=(payment_id, order_id)`.
6. **`B1_SHORT_BANK_CREDIT`**:
   - Bank statement row matched to a settlement batch where `bank_row.credit < settlement.amount - rounding_tolerance_paise`.
   - Flagged with `action="flag"`, `entity_ids=(statement_id,)`.
7. **Never-Flag Invariants**:
   - Legitimate adjustments (`adj_...`) $\rightarrow$ no flag.
   - Failed payments $\rightarrow$ no flag.
   - In-flight payments and in-flight refunds $\rightarrow$ no flag.
   - Late settlements within the 2-working-day grace window $\rightarrow$ no flag.

---

## 4. Text Reasoning Enhancement (System B1r)

System **B1r** extends B1 by inspecting `notes.reason_text` on refund recon items when arithmetic mismatch is detected:
- **Parser Patterns**:
  - Generic currency symbols and abbreviations: `(?:Rs\.?|INR|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?|\.\d+)`
  - Explicit paise values: `(\d+)\s*paise`
  - Generic number-word parser: converts word numbers (`one`, `twenty-five`, `seventy-three`, `hundred`, `thousand`) into integer paise for phrases like `"four rupees and seventy-three paise"`.
- **Decision Logic**:
  - If parsed stated amount equals the actual arithmetic difference $|\text{debit} - \text{order\_refund\_amount}|$:
    $\rightarrow$ `action="benign_explained"`, `rule_id="B1R_EXPLAINED_MATCH"`.
  - If parsed stated amount differs from the actual difference:
    $\rightarrow$ `action="flag"`, `rule_id="B1R_EXPLAINED_MISMATCH"`.
  - If no numeric or word amount is found in `reason_text`:
    $\rightarrow$ `action="escalate_human"`, `rule_id="B1R_EXPLAINED_NO_AMOUNT"`.
- **Validation Notice & Frozen Status**:
  - B1r's parser patterns were derived from analysis of the dev split `reason_text` samples and verified against dev ground truth labels:
    - 15/15 `EXPLAINED_CONSISTENT` cases matched the exact actual difference.
    - 15/15 `EXPLAINED_INCONSISTENT` cases parsed an amount that differed from the actual difference.
    - 15/15 `EXPLAINED_NO_AMOUNT` cases returned no parsed amount and escalated to human.
  - Because patterns were validated against dev data, dev performance figures are optimistic.
  - The B1r parser code and patterns are **strictly frozen** after Day 5.
