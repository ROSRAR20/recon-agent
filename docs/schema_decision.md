# Schema Decision: recon-agent

All assertions regarding Razorpay API behavior strictly cite:
- `[OBSERVED]` (verified in real API probe output from `docs/api_probe_report.md`)
- `[DOCS]` (verified in official documentation at `https://razorpay.com/docs/api/settlements/fetch-recon`)
- `[UNVERIFIED]` (assumed or pending empirical validation)

---

## 1. Test Mode Recon Combined Behavior
- **Does `/v1/settlements/recon/combined` return data in test mode?**  
  `[ ] Yes  |  [ ] No  |  [x] Empty items collection`  
  - For periods `2026-09` and `2026-10`, `/v1/settlements/recon/combined` returned HTTP status `200` with an empty items collection (`count: 0`, `items: []`) [OBSERVED].
  - Context & Caveat: This account has no transactions [OBSERVED], so this empty response is not evidence about whether test activity yields recon rows [UNVERIFIED].
  - Official Docs Reference: The API endpoint specification is documented at [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon].

---

## 2. Fee and Tax Accounting Semantics
- **Does `fee` include `tax` or are they additive?**  
  *Tag: [UNVERIFIED]*  
  - Empirical verification: The relationship between `fee` and `tax` remains [UNVERIFIED] because the API probe yielded empty items collections with no transaction-level records [OBSERVED].
  - Documented example arithmetic [DOCS sample]: In the official documentation example at [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon], a transfer item displays:
    - `amount`: 100000 (in paise)
    - `fee`: 296 (in paise)
    - `tax`: 46 (in paise)
    - `debit`: 100296 (in paise)
    The arithmetic $\text{debit} = \text{amount} + \text{fee} = 100000 + 296 = 100296$ suggests that `fee` may include `tax` [DOCS sample].
  - However, whether `fee` includes `tax` or whether they are additive across transaction types remains [UNVERIFIED] until confirmed with populated live or test data.
  - Recon documentation defines `fee` and `tax` as distinct integer fields in currency subunits (paise) [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon]. Payment entity fee/tax definitions remain [UNVERIFIED].

---

## 3. Monetary Units & Minor Currency Standard
- **Design rule (ours): integer paise, no floats:** All internal balances, transaction amounts, fee schedules, invariant checks, and ledger entries strictly use integer minor currency units (paise for INR; 1 INR = 100 paise). Floating-point arithmetic for currency is strictly prohibited across the codebase.
- **API units:** According to official documentation, `debit`, `credit`, `amount`, `fee`, and `tax` are represented in currency subunits (integer paise for INR) [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon]. Payment-entity specific units remain [UNVERIFIED] due to empty probe responses [OBSERVED].

---

## 4. Entity Field List & Normalized Schema

- **Payment Entity Fields:**  
  Empty items collection returned by API probe (`items: []`) [OBSERVED]. Field list is [UNVERIFIED].
- **Settlement Entity Fields:**  
  Empty items collection returned by API probe (`items: []`) [OBSERVED]. Field list is [UNVERIFIED].
- **Refund Entity Fields:**  
  Empty items collection returned by API probe (`items: []`) [OBSERVED]. Field list is [UNVERIFIED].
- **Order Entity Fields:**  
  Empty items collection returned by API probe (`items: []`) [OBSERVED]. Field list is [UNVERIFIED].

- **Recon Item Entity Fields:**  
  Populated from official documentation [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon]:
  - `entity_id`: string — Unique identifier of the transaction (e.g., `pay_...`, `rfnd_...`, `trf_...`, `adj_...`) [DOCS]
  - `type`: string — Type of transaction (`payment`, `refund`, `transfer`, `adjustment`) [DOCS]
  - `debit`: integer — Debited amount in currency subunits (paise) [DOCS]
  - `credit`: integer — Credited amount in currency subunits (paise) [DOCS]
  - `amount`: integer — Total transaction amount in currency subunits (paise) [DOCS]
  - `currency`: string — 3-letter ISO currency code (e.g., `INR`) [DOCS]
  - `fee`: integer — Processing fee in currency subunits (paise) [DOCS]
  - `tax`: integer — Tax on processing fee in currency subunits (paise) [DOCS]
  - `on_hold`: boolean — Indicates whether settlement is on hold [DOCS]
  - `settled`: boolean — Indicates whether the transaction has been settled [DOCS]
  - `created_at`: integer — Unix timestamp of creation [DOCS]
  - `settled_at`: integer — Unix timestamp when settled [DOCS]
  - `settlement_id`: string — Identifier of the settlement batch (`setl_...`) [DOCS]
  - `description`: string / null — Description of the transaction [DOCS]
  - `payment_id`: string / null — Associated payment ID (for refunds/transfers) [DOCS]
  - `settlement_utr`: string / null — UTR of the settlement transfer [DOCS]
  - `notes`: object / null — Key-value pairs of user-provided notes [DOCS]

### Record Linkage Fields
- **`entity_id`:** ID of the settled transaction (`pay_` / `rfnd_` / `trf_` / `adj_`) [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon].
- **`payment_id`:** Linked payment for refund/transfer, null for payments [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon].
- **`settlement_id`:** Links settlement entities to combined recon line items (`setl_...`) [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon].
- **`settlement_utr`:** Reference number (UTR) for the bank settlement [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon].
- **`order_id`:** Links internal merchant orders to payment capture records [UNVERIFIED in recon combined endpoint docs; observed empty in probe [OBSERVED]].

---

## 5. Architectural Data Source Decision
- **Selected Strategy:**  
  `[ ] REAL  |  [x] SYNTHETIC  |  [ ] HYBRID`

  - **Selected: SYNTHETIC**
    - Synthetic data generation is selected for settlement recon (`/v1/settlements/recon/combined`), settlement batches (`/v1/settlements`), and external bank statement records.
  - **HYBRID Option (Optional Extension):**
    - A HYBRID strategy is noted as an optional extension if a real payments/orders slice is added later through active test-mode transaction generation.

---

## 6. Engineering Reasoning
1. **Empty Probe Observations:** The read-only API probe across all endpoints (`/v1/settlements/recon/combined`, `/v1/settlements`, `/v1/payments`, `/v1/orders`, `/v1/refunds`) returned HTTP 200 with empty items collections for both 2026-09 and 2026-10 [OBSERVED]. Because this account has no transactions [OBSERVED], no live records or schemas were returned, providing no evidence about whether test activity yields recon rows [UNVERIFIED].
2. **Recon Schema Grounding:** The official documentation at [DOCS: https://razorpay.com/docs/api/settlements/fetch-recon] provides the complete field specification and types for recon line items in integer minor units (paise).
3. **Controlled Invariant Benchmarking:** Evaluating deterministic rules, verifier constraints, and LLM reasoning requires known ground-truth datasets with injected anomalies (e.g. fee/tax arithmetic mismatches, timing lags, partial refunds). Given that test-mode recon data was empty, a SYNTHETIC data generation approach modeled strictly after the documented Razorpay schema ensures reproducible evaluation.
4. **Extensibility:** If real test-mode transactions are generated in the future, the pipeline can flexibly adopt a HYBRID model by ingesting real payments/orders alongside synthetic recon settlements.
