# Synthetic Data Specification: recon-agent

This specification defines the deterministic generator for clean (error-free) synthetic reconciliation data.

All parameters, entity schemas, and behavioral distributions are explicitly tagged with evidence tags:
- `[DOCS]` — Grounded in official Razorpay API documentation at `https://razorpay.com/docs/api/settlements/fetch-recon`.
- `[ASSUMED]` — Synthetic modeling parameter or internal system architecture assumption (not Razorpay behavior).
- `[UNVERIFIED]` — Empirical behavior observed empty or pending real-world confirmation.

---

## 1. Monetary and Arithmetic Invariants

- **Integer Paise Representation:** All monetary values (`amount`, `credit`, `debit`, `fee`, `tax`, `balance`) are strictly non-negative integers in minor currency units (paise; 1 INR = 100 paise) `[ASSUMED / DESIGN RULE]`. Floating-point currency arithmetic is prohibited.
- **Integer Rounding Helper:** Half-up rounding is implemented via pure integer arithmetic:  
  $\text{mul\_div\_round}(x, \text{num}, \text{den}) = \lfloor \frac{2 \cdot x \cdot \text{num} + \text{den}}{2 \cdot \text{den}} \rfloor$ `[ASSUMED / DESIGN RULE]`.
- **Fee and Tax Semantics:**
  - $\text{base\_fee} = \text{mul\_div\_round}(\text{amount}, \text{rate\_bps}, 10000)$ `[ASSUMED]`
  - $\text{tax} = \text{mul\_div\_round}(\text{base\_fee}, 1800, 10000)$ `[ASSUMED]`
  - If `FEE_INCLUDES_TAX = True`: $\text{fee} = \text{base\_fee} + \text{tax}$ and $\text{credit} = \text{amount} - \text{fee}$ `[DOCS sample arithmetic; UNVERIFIED for general live data]`
  - If `FEE_INCLUDES_TAX = False`: $\text{fee} = \text{base\_fee}$ and $\text{credit} = \text{amount} - \text{fee} - \text{tax}$ `[ASSUMED alternate]`
  - For refund items: $\text{debit} = \text{refund\_amount}$, $\text{credit} = 0$, $\text{fee} = 0$, $\text{tax} = 0$ `[DOCS sample]`

---

## 2. Temporal & Calendar Specification

- **Timezone:** Indian Standard Time (IST, UTC+05:30) `[ASSUMED / STANDARD]`.
- **Timestamps:** Standard integer Unix seconds `[DOCS / ASSUMED]`.
- **Simulation Window:** 22 working days (Monday through Friday) starting Monday, 2026-08-03 `[ASSUMED]`.
- **Settlement Lag:**
  - Captured payments settle at $C + 2$ working days `[ASSUMED standard T+2 settlement]`.
  - Refunds settle at $R + 1$ working day after processing `[ASSUMED]`.
  - Settlements are batched once per merchant per settlement date `[ASSUMED]`.
  - Refund debits are deferred to the subsequent batch if including them would cause batch net to be $\le 0$ `[ASSUMED / INVARIANT I8]`.

---

## 3. Configuration Parameters

| Parameter | Value | Tag | Description |
| :--- | :--- | :--- | :--- |
| `start_date` | `2026-08-03` | `[ASSUMED]` | Window start date (Monday). |
| `working_days` | `22` | `[ASSUMED]` | Count of Monday–Friday business days. |
| `profile: small` | 10 merchants | `[ASSUMED]` | Fast test profile for dev iteration. |
| `profile: full` | 60 merchants | `[ASSUMED]` | Full benchmarking profile per split. |
| `avg_payments_per_day` | 20 (Poisson mean) | `[ASSUMED]` | Mean captured attempts per merchant-day. |
| `method_mix` | Card: 45%, UPI: 40%, Netbanking: 15% | `[ASSUMED]` | Payment instrument distribution. |
| `card_fee_bps` | 200 bps (2.0%) | `[ASSUMED]` | Illustrative placeholder rate (not Razorpay pricing). |
| `upi_fee_bps` | 0 bps (0.0%) | `[ASSUMED]` | Illustrative placeholder rate. |
| `netbanking_fee_bps` | 150 bps (1.5%) | `[ASSUMED]` | Illustrative placeholder rate. |
| `gst_bps` | 1800 bps (18.0%) | `[ASSUMED]` | Standard Indian GST rate on processing fee. |
| `fee_includes_tax` | `True` | `[DOCS sample / UNVERIFIED]` | Sample arithmetic in fetch-recon docs suggests fee includes tax. |
| `failed_rate_bps` | 600 bps (~6%) | `[ASSUMED]` | Transaction attempts failing authorization. |
| `refund_rate_bps` | 400 bps (~4%) | `[ASSUMED]` | Captured transactions subsequently refunded. |
| `refund_split` | 70% full / 30% partial | `[ASSUMED]` | Ratio of full vs. partial refund amounts. |
| `adjustment_prob` | ~3.2% per batch | `[ASSUMED]` | Rare dispute or goodwill fee adjustment item (yielding >= 30 in full profile). |
| `bank_narration_mix` | 70% clean / 20% noisy / 10% no UTR | `[ASSUMED]` | Bank statement narration realistic format variation. |
| `unrelated_bank_rows` | ~30 rows / merchant-month | `[ASSUMED]` | External bank statement noise (payroll, rent, utility). |

---

## 4. Entity Schemas & File Outputs

Datasets are output to `data/synth/<profile>/<split>/`:

### 1. `recon_items.jsonl`
Fields strictly match the documented Razorpay recon endpoint `[DOCS: https://razorpay.com/docs/api/settlements/fetch-recon]`, plus internal routing attribute `merchant_id` `[ASSUMED]`:
- `entity_id`: string (e.g. `pay_...`, `rfnd_...`, `adj_...`) `[DOCS]`
- `type`: string (`payment`, `refund`, `adjustment`) `[DOCS]`
- `debit`: integer paise `[DOCS]`
- `credit`: integer paise `[DOCS]`
- `amount`: integer paise `[DOCS]`
- `currency`: string (`INR`) `[DOCS]`
- `fee`: integer paise `[DOCS]`
- `tax`: integer paise `[DOCS]`
- `on_hold`: boolean `[DOCS]`
- `settled`: boolean `[DOCS]`
- `created_at`: integer unix seconds `[DOCS]`
- `settled_at`: integer unix seconds `[DOCS]`
- `settlement_id`: string (`setl_...`) `[DOCS]`
- `description`: string / null `[DOCS]`
- `notes`: object / null `[DOCS]`
- `payment_id`: string / null `[DOCS]`
- `settlement_utr`: string (16 alphanumeric) `[DOCS sample / ASSUMED]`
- `order_id`: string / null `[DOCS]`
- `order_receipt`: string / null `[DOCS]`
- `method`: string / null `[DOCS]`
- `card_network`: string / null `[DOCS]`
- `card_issuer`: string / null `[DOCS]`
- `card_type`: string / null `[DOCS]`
- `dispute_id`: string / null `[DOCS]`
- `merchant_id`: string `[ASSUMED: internal evaluation routing only, NOT a Razorpay field]`

### 2. Auxiliary Datasets `[ASSUMED SHAPES]`
- `payments.jsonl`: Gateway transactions including failed payment attempts (`payment_id`, `order_id`, `merchant_id`, `amount`, `status`, etc.) `[ASSUMED]`.
- `orders_ledger.csv`: Internal merchant order management ledger (`order_id`, `merchant_id`, `booked_date`, `created_at`, `expected_amount`, `currency`, `order_receipt`, `status`, `payment_id`, `refund_id`, `refund_amount`) `[ASSUMED]`. In clean data, `booked_date == IST date of payment capture` `[ASSUMED]`.
- `settlements.jsonl`: Lump-sum gateway payout batches (`settlement_id`, `merchant_id`, `amount`, `utr`, `status`, `created_at`, `settled_at`) `[ASSUMED]`.
- `bank_statement.csv`: Merchant bank statement records containing settlement credits with noisy narrations and unrelated non-gateway transactions extending up to `as_of` `[ASSUMED]`.
- `rate_card.json`: Active fee schedules and tax basis points `[ASSUMED]`.
- `manifest.json`: Deterministic seed, configuration hash (SHA-256), generator version, `as_of` timestamp (`last_settlement_date + 10 calendar days`), and dataset row/class counts `[ASSUMED]`.
- `labels.jsonl`: Evaluation ground-truth labels (`case_id`, `merchant_id`, `error_class`, `hard_negative_type`, `entity_ids`, `settlement_id`, `expected_action`) `[ASSUMED]`.

---

## 5. Error Injection Classes `[ASSUMED ERROR MODELS]`

Each entity receives at most one error class. Batch-level classes select only batches free of entity-level errors. All injections preserve settlement conservation (I2) except `SHORT_BANK_CREDIT`:

1. `DUPLICATE_PAYMENT` (1.0% of captured orders) `[ASSUMED]`: Second captured payment for same order and amount with new `payment_id`, settled in same batch; ledger unchanged. I2 preserved.
2. `MISSING_SETTLEMENT` (0.8% of captured payments, eligible where settle date + 7 days $\le$ `as_of`) `[ASSUMED]`: Captured payment excluded from recon, settlement, and bank credit. I2 preserved.
3. `FEE_TAX_MISMATCH` (1.2% of card/netbanking payments; never UPI) `[ASSUMED]`: 60% fee_off (+5% to +25%, $\ge +10$ paise vs rate card), 40% tax_off ($\ge 5$ paise). Credit follows the erroneous value so settlement and bank tie. I2 preserved.
4. `PARTIAL_REFUND_MISMATCH` (15% of refund items) `[ASSUMED]`: 80% refund debit differs from ledger `refund_amount` by $\ge 100$ paise; 20% refund with no ledger entry. I2 preserved.
5. `CUTOFF_DRIFT` (1.5% of payments, Mon–Thu orders) `[ASSUMED]`: Created 23:40–23:59 IST on day D, captured 00:05–00:35 on D+1; recon `created_at` on D+1, ledger `booked_date = D`; item moves to D+1 cohort. I2 preserved.
6. `SHORT_BANK_CREDIT` (6% of eligible batches) `[ASSUMED]`: Bank credit = settlement net minus 1%–5% of net ($\ge 100$ paise); same UTR. I2 violated here only.
7. `ORPHAN_BANK_CREDIT` (extra rows = 6% of batch count) `[ASSUMED]`: Razorpay-style bank credit matching no settlement.

---

## 6. Hard Negatives `[ASSUMED BEHAVIORAL MODELS]`

Hard negatives represent legitimate operational variations that must NOT be flagged as errors (labeled with `expected_action: "benign_ignore"`):

1. `ROUNDING` (3% of card/netbanking items) `[ASSUMED]`: Fee or tax off by $\pm 1$ or $\pm 2$ paise due to rounding variance; credit adjusted so I2 holds.
2. `LEGIT_ADJUSTMENT` `[ASSUMED]`: Fee dispute credits or manual adjustments that are documented in recon (`type == "adjustment"`).
3. `LATE_SETTLEMENT` (2%–3% of eligible batches) `[ASSUMED]`: Captured Friday settles Tuesday (standard C+2) or delayed settlement batch shifted by 1–2 working days; I2 holds.
4. `RETRY_AFTER_FAIL` (50% of failed attempts) `[ASSUMED]`: Failed attempt shares `order_id` with a subsequent captured payment.
5. `MULTI_MERCHANT_SAME_UTR` `[ASSUMED]`: Rare bank-side batching of payouts across merchants under the same settlement UTR.
6. `NOISY_BANK_NARRATION` `[ASSUMED]`: Lowercase, trailing spaces, or truncated UTR in bank narration.
7. `UNRELATED_BANK_CREDIT` `[ASSUMED]`: External other-gateway payouts (Stripe, Paytm), payroll, or interest credits present in bank statement.

---

## 7. Invariants Tested

- **I1 (Item Arithmetic):** $\text{base\_fee} = \text{mul\_div\_round}(\text{amount}, \text{rate\_bps}, 10000)$, $\text{tax} = \text{mul\_div\_round}(\text{base\_fee}, 1800, 10000)$, $\text{credit} = \text{amount} - \text{fee}$ (or $\text{amount} - \text{fee} - \text{tax}$). For refunds: $\text{debit} = \text{amount}, \text{credit} = 0, \text{fee} = 0, \text{tax} = 0$.
- **I2 (Settlement Conservation):** $\sum \text{credit} - \sum \text{debit} = \text{settlement\_net} = \text{bank\_credit\_amount}$ (holds for all clean batches and all injections except `SHORT_BANK_CREDIT`).
- **I3 (Referential Integrity):** Every captured payment has exactly one recon payment item; every recon item links to a valid `settlement_id`.
- **I4 (Ledger Consistency):** Order expected amount equals payment amount; refund amounts align with orders ledger.
- **I5 (Type & ID Hygiene):** Strictly zero floating-point numbers in emitted JSON files; all identifiers conform to unique `prefix + 14 base62` characters.
- **I6 (Determinism):** Same seed produces bit-identical files; distinct seeds diverge.
- **I7 (Architectural Boundary):** No module outside `src/recon_agent/synth/`, `tests/`, and `scripts/generate.py` imports `recon_agent.synth`.
- **I8 (Positive Settlement Batch Net):** Every settlement batch satisfies $\text{net} > 0$.
- **I9 (Schema Conformance):** All emitted files conform to column, type, and nullability rules defined in `docs/schemas.json`.
- **I10 (Disjointness of Error Entities):** Each entity participates in at most one error class.

