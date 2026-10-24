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
  *Official Docs Context: The API endpoint specification is documented at [DOCS: https://razorpay.com/docs/api/settlements/#fetch-settlement-recon-details]. Test-mode behavior on non-activated accounts is to be confirmed empirically.*

---

## 2. Fee and Tax Accounting Semantics
- **Does `fee` include `tax` or are they additive?**  
  *Tag: [UNVERIFIED - to be confirmed from probe output]*  
  *Formula hypothesis: Does $\text{deduction} = \text{fee}$ or does $\text{deduction} = \text{fee} + \text{tax}$?*  
  *Reference: Payment entity schema defines `fee` and `tax` in paise [DOCS: https://razorpay.com/docs/api/payments/].*

---

## 3. Monetary Units & Minor Currency Standard
- **Units:** Integer minor units strictly (paise for INR). 1 INR = 100 paise.  
  *Floating-point numbers are prohibited for all balance and transaction calculations across the pipeline.*

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

---

## 5. Architectural Data Source Decision
- **Selected Strategy:**  
  `[ ] REAL  |  [ ] SYNTHETIC  |  [ ] HYBRID`

---

## 6. Engineering Reasoning
*(To be justified based on whether test-mode endpoints yield populated settlements and recon line items, or require synthetic augmentation modeled on Razorpay's documented schemas).*
