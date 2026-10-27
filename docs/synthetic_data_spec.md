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
| `adjustment_prob` | ~0.3% per batch | `[ASSUMED]` | Rare dispute or goodwill fee adjustment item. |
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
- `settlement_utr`: string (16 alphanumeric) `[DOCS]`
- `order_id`: string / null `[DOCS]`
- `order_receipt`: string / null `[DOCS]`
- `method`: string / null `[DOCS]`
- `card_network`: string / null `[DOCS]`
- `card_issuer`: string / null `[DOCS]`
- `card_type`: string / null `[DOCS]`
- `dispute_id`: string / null `[DOCS]`
- `merchant_id`: string `[ASSUMED: internal evaluation routing only, NOT a Razorpay field]`

### 2. Auxiliary Datasets `[ASSUMED SHAPES]`
- `payments.jsonl`: Gateway transactions including failed payment attempts (`payment_id`, `order_id`, `amount`, `status`, etc.).
- `orders_ledger.csv`: Internal merchant order management ledger (`order_id`, `expected_amount`, `status`, `refund_id`, `refund_amount`).
- `settlements.jsonl`: Lump-sum gateway payout batches (`settlement_id`, `amount`, `utr`, `status`, `settled_at`).
- `bank_statement.csv`: Merchant bank statement records containing settlement credits with noisy narrations and unrelated non-gateway transactions.
- `rate_card.json`: Active fee schedules and tax basis points.
- `manifest.json`: Deterministic seed, configuration hash (SHA-256), generator version, and dataset row counts.

---

## 5. Invariants Tested

- **I1 (Item Arithmetic):** $\text{base\_fee} = \text{mul\_div\_round}(\text{amount}, \text{rate\_bps}, 10000)$, $\text{tax} = \text{mul\_div\_round}(\text{base\_fee}, 1800, 10000)$, $\text{credit} = \text{amount} - \text{fee}$ (or $\text{amount} - \text{fee} - \text{tax}$). For refunds: $\text{debit} = \text{amount}, \text{credit} = 0, \text{fee} = 0, \text{tax} = 0$.
- **I2 (Settlement Conservation):** $\sum \text{credit} - \sum \text{debit} = \text{settlement\_net} = \text{bank\_credit\_amount}$.
- **I3 (Referential Integrity):** Every captured payment has exactly one recon payment item; every recon item links to a valid `settlement_id`.
- **I4 (Ledger Consistency):** Order expected amount equals payment amount; refund amounts align with orders ledger.
- **I5 (Type & ID Hygiene):** Strictly zero floating-point numbers in emitted JSON files; all identifiers conform to unique `prefix + 14 base62` characters.
- **I6 (Determinism):** Same seed produces bit-identical files; distinct seeds diverge.
- **I7 (Architectural Boundary):** No module outside `src/recon_agent/synth/`, `tests/`, and `scripts/generate.py` imports `recon_agent.synth`.
- **I8 (Positive Settlement Batch Net):** Every settlement batch satisfies $\text{net} > 0$.
