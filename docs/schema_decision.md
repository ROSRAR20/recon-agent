# Schema Decision: recon-agent

> **STATUS: TEMPLATE ONLY - DO NOT FILL TODAY**  
> This document will be completed after executing `scripts/probe_razorpay.py` and reviewing the resulting test-mode responses in `docs/api_probe_report.md`.
>
> All assertions regarding Razorpay API behavior must strictly cite tags:  
> - `[OBSERVED]` (verified in real API probe output)  
> - `[DOCS]` (verified in official documentation at razorpay.com/docs)  
> - `[UNVERIFIED]` (assumed or pending empirical validation)

---

## 1. Test Mode Recon Combined Behavior
- **Does `/v1/settlements/recon/combined` return data in test mode?**  
  `[ ] Yes  |  [ ] No  |  [ ] Empty items collection`  
  *Tag: [UNVERIFIED - pending execution of scripts/probe_razorpay.py]*  
  *Official Docs Context: The API endpoint specification is documented at [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon].*

---

## 2. Fee and Tax Accounting Semantics
- **Does `fee` include `tax` or are they additive?**  
  *Tag: [UNVERIFIED - to be confirmed from probe output]*  
  *Formula hypothesis: Does $\text{deduction} = \text{fee}$ or does $\text{deduction} = \text{fee} + \text{tax}$?*  
  *Reference: Payment entity schema defines `fee` and `tax` in paise [DOCS: https://razorpay.com/docs/api/payments/]. Recon documentation also lists `fee` and `tax` as distinct fields [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon].*

---

## 3. Monetary Units & Minor Currency Standard
- **Design rule (ours): integer paise, no floats:** All internal balances, transaction amounts, fee schedules, invariant checks, and ledger entries strictly use integer minor currency units (paise for INR; 1 INR = 100 paise). Floating-point arithmetic for currency is strictly prohibited across the codebase.
- **API units: [UNVERIFIED until observed]:** Official documentation specifies transaction amounts, fees, and taxes in currency subunits (paise for INR) [DOCS: https://razorpay.com/docs/api/payments/], [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon]. However, empirical response unit behavior from the live probe remains unverified until observed in actual test-mode payloads.

---

## 4. Entity Field List & Normalized Schema
*(To be populated from `docs/api_probe_report.md` JSON type schemas once probe is run)*

- **Payment Entity Fields:**  
  `[Pending probe output]`
- **Settlement Entity Fields:**  
  `[Pending probe output]`
- **Recon Item Entity Fields:**  
  `[Pending probe output]`
- **Refund Entity Fields:**  
  `[Pending probe output]`
- **Order Entity Fields:**  
  `[Pending probe output]`

### Record Linkage Fields
*(Identifies keys linking entities across ledger and gateway; tagged [UNVERIFIED] until observed in probe responses)*
- **`settlement_id`:** [UNVERIFIED until observed] Links settlement entities to combined recon line items (`setl_...`).
- **`payment_id` / `entity_id`:** [UNVERIFIED until observed] Links payment records to recon line items, orders, and refund transactions (`pay_...`).
- **`order_id`:** [UNVERIFIED until observed] Links internal merchant orders to payment capture records and recon line items (`order_...`).

---

## 5. Architectural Data Source Decision
- **Selected Strategy:**  
  `[ ] REAL  |  [ ] SYNTHETIC  |  [ ] HYBRID`

---

## 6. Engineering Reasoning
*(To be justified based on whether test-mode endpoints yield populated settlements and recon line items, or require synthetic augmentation modeled on Razorpay's documented schemas).*
