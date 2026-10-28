"""Error injection and hard-negative generator for recon-agent synthetic datasets."""

import datetime
import random
from typing import Any

from src.recon_agent.synth.config import SynthConfig
from src.recon_agent.synth.generator import IST_TZ, _gen_base62_id, _gen_utr, finalize_bank_statements
from src.recon_agent.synth.math_utils import mul_div_round


def inject_errors_and_labels(
    data: dict[str, Any],
    config: SynthConfig,
    seen_ids: set[str],
    seen_utrs: set[str],
    working_days: list[datetime.date],
    as_of_date: datetime.date,
) -> list[dict[str, Any]]:
    """Inject controlled errors and generate comprehensive ground-truth evaluation labels.

    Uses dedicated per-class RNG streams: random.Random(seed * 1000 + class_index).
    """
    # Per-class RNG streams
    rng_dup = random.Random(config.seed * 1000 + 1)
    rng_mis = random.Random(config.seed * 1000 + 2)
    rng_fee = random.Random(config.seed * 1000 + 3)
    rng_ref = random.Random(config.seed * 1000 + 4)
    rng_cut = random.Random(config.seed * 1000 + 5)
    rng_sho = random.Random(config.seed * 1000 + 6)
    rng_orp = random.Random(config.seed * 1000 + 7)
    rng_rnd = random.Random(config.seed * 1000 + 8)
    rng_lat = random.Random(config.seed * 1000 + 9)
    rng_ret = random.Random(config.seed * 1000 + 10)

    used_entities: set[str] = set()
    used_batches: set[str] = set()
    labels: list[dict[str, Any]] = []

    def get_rate(name: str, base_rate: float) -> float:
        factor = config.rate_factors.get(name, 1.0)
        return base_rate * factor

    # Fast indexing structures
    payments_by_id = {p["payment_id"]: p for p in data["payments"]}
    orders_by_id = {o["order_id"]: o for o in data["orders_ledger"]}
    recon_by_entity_id = {r["entity_id"]: r for r in data["recon_items"]}
    settlements_by_id = {s["settlement_id"]: s for s in data["settlements"]}

    # Index settlements by (merchant_id, date)
    settlements_by_merchant_date: dict[tuple[str, str], dict[str, Any]] = {}
    for s in data["settlements"]:
        settle_dt = datetime.datetime.fromtimestamp(s["settled_at"], tz=IST_TZ).date().isoformat()
        settlements_by_merchant_date[(s["merchant_id"], settle_dt)] = s

    # Index bank rows by settlement_id
    bank_rows_by_settlement: dict[str, dict[str, Any]] = {}
    for b in data["bank_statement"]:
        for s in data["settlements"]:
            if (
                b["merchant_id"] == s["merchant_id"]
                and b["date"] == datetime.datetime.fromtimestamp(s["settled_at"], tz=IST_TZ).date().isoformat()
                and (
                    s["utr"] in b["description"]
                    or s["utr"].lower() in b["description"].lower()
                    or s["utr"][:12] in b["description"]
                    or "RAZORPAY PAYMENT SERVICES" in b["description"]
                )
            ):
                bank_rows_by_settlement[s["settlement_id"]] = b
                break

    # Helper to create label entries conforming to docs/schemas.json
    def add_label(
        merchant_id: str,
        error_class: str | None,
        hard_negative_type: str | None,
        entity_ids: list[str],
        settlement_id: str | None,
        expected_action: str,
    ) -> None:
        labels.append(
            {
                "case_id": f"case_{len(labels)+1:06d}",
                "merchant_id": merchant_id,
                "error_class": error_class,
                "hard_negative_type": hard_negative_type,
                "entity_ids": entity_ids,
                "settlement_id": settlement_id,
                "expected_action": expected_action,
            }
        )

    # =========================================================================
    # Error Injection Classes (7 classes)
    # =========================================================================

    # 1. DUPLICATE_PAYMENT (1.0% of captured orders, random 1-300s gap)
    captured_orders = [
        o
        for o in data["orders_ledger"]
        if o["status"] == "captured"
        and o["order_id"] not in used_entities
        and o["payment_id"] not in used_entities
    ]
    rng_dup.shuffle(captured_orders)
    dup_count = int(round(len(captured_orders) * get_rate("DUPLICATE_PAYMENT", 0.010)))

    for order in captured_orders[:dup_count]:
        p1 = payments_by_id[order["payment_id"]]
        r1 = recon_by_entity_id.get(p1["payment_id"])
        if not r1:
            continue
        stl = settlements_by_id.get(r1["settlement_id"])
        if not stl or stl["settlement_id"] in used_batches:
            continue
        bk_row = bank_rows_by_settlement.get(stl["settlement_id"])
        if not bk_row:
            continue

        p2_id = _gen_base62_id(rng_dup, "pay_", seen_ids)
        p2 = dict(p1)
        p2["payment_id"] = p2_id
        gap = rng_dup.randint(1, 300)
        p2["created_at"] = p1["created_at"] + gap
        p2["captured_at"] = (p1["captured_at"] or p1["created_at"]) + gap
        data["payments"].append(p2)
        payments_by_id[p2_id] = p2

        r2 = dict(r1)
        r2["entity_id"] = p2_id
        r2["created_at"] = p2["created_at"]
        data["recon_items"].append(r2)
        recon_by_entity_id[p2_id] = r2

        # Settlement batch net and bank credit increase by duplicate credit (I2 preserved)
        stl["amount"] += r2["credit"]
        bk_row["credit"] += r2["credit"]
        bk_row["balance"] += r2["credit"]

        used_entities.update([p1["payment_id"], p2_id, order["order_id"]])
        used_batches.add(stl["settlement_id"])

        add_label(
            merchant_id=p1["merchant_id"],
            error_class="DUPLICATE_PAYMENT",
            hard_negative_type=None,
            entity_ids=[p1["payment_id"], p2_id, order["order_id"]],
            settlement_id=stl["settlement_id"],
            expected_action="flag_duplicate_payment",
        )

    # 2. MISSING_SETTLEMENT (0.8% of captured payments where expected settle date + 7 days <= as_of)
    eligible_missing = []
    for p in data["payments"]:
        if p["status"] != "captured" or p["payment_id"] in used_entities:
            continue
        r = recon_by_entity_id.get(p["payment_id"])
        if not r:
            continue
        stl = settlements_by_id.get(r["settlement_id"])
        if not stl or stl["settlement_id"] in used_batches:
            continue
        settle_date = datetime.datetime.fromtimestamp(stl["settled_at"], tz=IST_TZ).date()
        if settle_date + datetime.timedelta(days=7) <= as_of_date:
            eligible_missing.append((p, r, stl))

    rng_mis.shuffle(eligible_missing)
    missing_count = int(round(len(eligible_missing) * get_rate("MISSING_SETTLEMENT", 0.008)))

    for p, r, stl in eligible_missing[:missing_count]:
        bk_row = bank_rows_by_settlement.get(stl["settlement_id"])
        if not bk_row or stl["amount"] - r["credit"] <= 0:
            continue

        # Exclude from recon, settlement, and bank statement (I2 preserved)
        data["recon_items"].remove(r)
        recon_by_entity_id.pop(r["entity_id"], None)
        stl["amount"] -= r["credit"]
        bk_row["credit"] -= r["credit"]
        bk_row["balance"] -= r["credit"]

        used_entities.update([p["payment_id"], p["order_id"]])
        used_batches.add(stl["settlement_id"])

        add_label(
            merchant_id=p["merchant_id"],
            error_class="MISSING_SETTLEMENT",
            hard_negative_type=None,
            entity_ids=[p["payment_id"]],
            settlement_id=stl["settlement_id"],
            expected_action="flag_missing_settlement",
        )

    # 3. FEE_TAX_MISMATCH (1.2% of card/netbanking payments; never UPI)
    eligible_feetax = [
        r
        for r in data["recon_items"]
        if r["type"] == "payment"
        and r["method"] in ("card", "netbanking")
        and r["entity_id"] not in used_entities
        and r["settlement_id"] not in used_batches
    ]
    rng_fee.shuffle(eligible_feetax)
    feetax_count = int(round(len(eligible_feetax) * get_rate("FEE_TAX_MISMATCH", 0.012)))

    for idx, r in enumerate(eligible_feetax[:feetax_count]):
        stl = settlements_by_id.get(r["settlement_id"])
        bk_row = bank_rows_by_settlement.get(r["settlement_id"])
        if not stl or not bk_row:
            continue

        is_fee_off = (idx % 10) < 6  # 60% fee_off, 40% tax_off
        if is_fee_off:
            # Fee raised by 5-25%, at least +10 paise vs rate card
            delta_fee = max(10, mul_div_round(r["fee"], rng_fee.randint(500, 2500), 10000))
            if stl["amount"] - delta_fee <= 0 or r["credit"] - delta_fee <= 0:
                continue
            r["fee"] += delta_fee
            r["credit"] -= delta_fee
            stl["amount"] -= delta_fee
            bk_row["credit"] -= delta_fee
            bk_row["balance"] -= delta_fee
        else:
            # Tax differs from expected tax by at least 5 paise
            delta_tax = max(5, mul_div_round(max(1, r["tax"]), rng_fee.randint(1000, 3000), 10000))
            if stl["amount"] - delta_tax <= 0 or r["credit"] - delta_tax <= 0:
                continue
            r["tax"] += delta_tax
            if config.fee_includes_tax:
                r["fee"] += delta_tax
            r["credit"] -= delta_tax
            stl["amount"] -= delta_tax
            bk_row["credit"] -= delta_tax
            bk_row["balance"] -= delta_tax

        used_entities.add(r["entity_id"])
        used_batches.add(stl["settlement_id"])

        add_label(
            merchant_id=r["merchant_id"],
            error_class="FEE_TAX_MISMATCH",
            hard_negative_type=None,
            entity_ids=[r["entity_id"]],
            settlement_id=stl["settlement_id"],
            expected_action="flag_fee_tax_mismatch",
        )

    # 4. PARTIAL_REFUND_MISMATCH (15% of refund items; random sign; resulting refund in [1, payment_amount]; |diff| >= 100)
    eligible_refunds = [
        r
        for r in data["recon_items"]
        if r["type"] == "refund"
        and r["entity_id"] not in used_entities
        and r["settlement_id"] not in used_batches
    ]
    rng_ref.shuffle(eligible_refunds)
    refund_mismatch_count = int(round(len(eligible_refunds) * get_rate("PARTIAL_REFUND_MISMATCH", 0.150)))

    for idx, r in enumerate(eligible_refunds[:refund_mismatch_count]):
        stl = settlements_by_id.get(r["settlement_id"])
        bk_row = bank_rows_by_settlement.get(r["settlement_id"])
        order = orders_by_id.get(r["order_id"])
        p_cap = payments_by_id.get(r["payment_id"])
        if not stl or not bk_row or not order or not p_cap:
            continue

        is_debit_diff = (idx % 10) < 8  # 80% debit differs by >= 100 paise; 20% missing in ledger
        if is_debit_diff:
            p_amt = p_cap["amount"]
            r_amt = r["amount"]

            available_signs: list[int] = []
            if p_amt - r_amt >= 100:
                available_signs.append(+1)
            if r_amt - 1 >= 100:
                available_signs.append(-1)

            if not available_signs:
                continue

            sign = rng_ref.choice(available_signs)
            if sign == +1:
                max_diff = p_amt - r_amt
                diff_paise = rng_ref.randint(100, max(100, min(max_diff, mul_div_round(r_amt, 35, 100))))
                r_new = r_amt + diff_paise
            else:
                max_diff = r_amt - 1
                diff_paise = rng_ref.randint(100, max(100, min(max_diff, mul_div_round(r_amt, 35, 100))))
                r_new = r_amt - diff_paise

            delta_debit = r_new - r_amt
            if stl["amount"] - delta_debit <= 0:
                continue

            r["amount"] = r_new
            r["debit"] = r_new
            stl["amount"] -= delta_debit
            bk_row["credit"] -= delta_debit
            bk_row["balance"] -= delta_debit
        else:
            # Refund exists in recon/settlement but has no ledger entry
            order["refund_id"] = ""
            order["refund_amount"] = 0
            order["status"] = "captured"

        used_entities.update([r["entity_id"], r["payment_id"], r["order_id"]])
        used_batches.add(stl["settlement_id"])

        add_label(
            merchant_id=r["merchant_id"],
            error_class="PARTIAL_REFUND_MISMATCH",
            hard_negative_type=None,
            entity_ids=[r["entity_id"], r["payment_id"]],
            settlement_id=stl["settlement_id"],
            expected_action="flag_partial_refund_mismatch",
        )

    # 5. CUTOFF_DRIFT (1.5% of payments, orders Mon-Thu)
    eligible_cutoff = []
    for p in data["payments"]:
        if p["status"] != "captured" or p["payment_id"] in used_entities:
            continue
        tx_dt = datetime.datetime.fromtimestamp(p["created_at"], tz=IST_TZ)
        if tx_dt.weekday() >= 4:  # Only Mon-Thu (0=Mon, 3=Thu)
            continue
        r = recon_by_entity_id.get(p["payment_id"])
        if not r or r["settlement_id"] in used_batches:
            continue
        order = orders_by_id.get(p["order_id"])
        if not order:
            continue
        eligible_cutoff.append((p, r, order, tx_dt))

    rng_cut.shuffle(eligible_cutoff)
    cutoff_count = int(round(len(eligible_cutoff) * get_rate("CUTOFF_DRIFT", 0.015)))

    for p, r, order, tx_dt in eligible_cutoff[:cutoff_count]:
        day_D = tx_dt.date()
        try:
            d_idx = working_days.index(day_D)
        except ValueError:
            continue

        if d_idx + 3 >= len(working_days):
            continue

        day_D1 = working_days[d_idx + 1]
        stl_old_date = working_days[d_idx + 2].isoformat()
        stl_new_date = working_days[d_idx + 3].isoformat()

        stl_old = settlements_by_merchant_date.get((p["merchant_id"], stl_old_date))
        stl_new = settlements_by_merchant_date.get((p["merchant_id"], stl_new_date))

        if not stl_old or not stl_new:
            continue
        if stl_old["settlement_id"] in used_batches or stl_new["settlement_id"] in used_batches:
            continue
        if stl_old["amount"] - r["credit"] <= 0:
            continue

        bk_old = bank_rows_by_settlement.get(stl_old["settlement_id"])
        bk_new = bank_rows_by_settlement.get(stl_new["settlement_id"])
        if not bk_old or not bk_new:
            continue

        # Modify timing: created 23:40-23:59 on day D, captured 00:05-00:35 on D+1
        time_D = datetime.datetime.combine(day_D, datetime.time(23, rng_cut.randint(40, 59), rng_cut.randint(0, 59)), tzinfo=IST_TZ)
        time_D1 = datetime.datetime.combine(day_D1, datetime.time(0, rng_cut.randint(5, 35), rng_cut.randint(0, 59)), tzinfo=IST_TZ)

        order["booked_date"] = day_D.isoformat()
        order["created_at"] = int(time_D.timestamp())
        p["created_at"] = int(time_D.timestamp())
        p["captured_at"] = int(time_D1.timestamp())
        r["created_at"] = int(time_D1.timestamp())

        # Move item from D+2 settlement cohort to D+3 settlement cohort
        stl_old["amount"] -= r["credit"]
        bk_old["credit"] -= r["credit"]
        bk_old["balance"] -= r["credit"]

        stl_new["amount"] += r["credit"]
        bk_new["credit"] += r["credit"]
        bk_new["balance"] += r["credit"]

        r["settlement_id"] = stl_new["settlement_id"]
        r["settlement_utr"] = stl_new["utr"]
        r["settled_at"] = stl_new["settled_at"]

        used_entities.update([p["payment_id"], order["order_id"]])
        used_batches.update([stl_old["settlement_id"], stl_new["settlement_id"]])

        add_label(
            merchant_id=p["merchant_id"],
            error_class="CUTOFF_DRIFT",
            hard_negative_type=None,
            entity_ids=[p["payment_id"], order["order_id"]],
            settlement_id=r["settlement_id"],
            expected_action="flag_cutoff_drift",
        )

    # 6. SHORT_BANK_CREDIT (6% of eligible batches; restricted to batches with UTR in narration)
    variants = data.get("bank_narration_variants", {})
    eligible_short_batches = [
        s
        for s in data["settlements"]
        if s["settlement_id"] not in used_batches
        and bank_rows_by_settlement.get(s["settlement_id"]) is not None
        and variants.get(bank_rows_by_settlement[s["settlement_id"]]["statement_id"]) in ("clean", "noisy")
    ]
    rng_sho.shuffle(eligible_short_batches)
    short_count = int(round(len(data["settlements"]) * get_rate("SHORT_BANK_CREDIT", 0.060)))

    for stl in eligible_short_batches[:short_count]:
        bk_row = bank_rows_by_settlement.get(stl["settlement_id"])
        if not bk_row or stl["amount"] <= 200:
            continue

        short_pct = rng_sho.randint(100, 500)  # 1.00% to 5.00%
        short_paise = max(100, mul_div_round(stl["amount"], short_pct, 10000))
        bk_row["credit"] = stl["amount"] - short_paise
        bk_row["balance"] -= short_paise

        used_batches.add(stl["settlement_id"])
        used_entities.add(bk_row["statement_id"])

        add_label(
            merchant_id=stl["merchant_id"],
            error_class="SHORT_BANK_CREDIT",
            hard_negative_type=None,
            entity_ids=[bk_row["statement_id"]],
            settlement_id=stl["settlement_id"],
            expected_action="flag_short_bank_credit",
        )

    # 7. ORPHAN_BANK_CREDIT (extra rows = 6% of the number of batches; 70/20/10 narration mix)
    merchants = sorted(list({s["merchant_id"] for s in data["settlements"]}))
    settlement_dates = sorted(list({
        datetime.datetime.fromtimestamp(s["settled_at"], tz=IST_TZ).date()
        for s in data["settlements"]
    }))
    if not settlement_dates:
        settlement_dates = working_days[:config.working_days]
    orphan_count = int(round(len(data["settlements"]) * get_rate("ORPHAN_BANK_CREDIT", 0.060)))

    for _ in range(orphan_count):
        fake_utr = _gen_utr(rng_orp, seen_utrs)
        m_id = rng_orp.choice(merchants)
        rnd_date = rng_orp.choice(settlement_dates)

        # Sample from merchant's actual settlement nets (+/-30% noise)
        m_settlement_amounts = [s["amount"] for s in data["settlements"] if s["merchant_id"] == m_id]
        if m_settlement_amounts:
            base_amt = rng_orp.choice(m_settlement_amounts)
            noise_pct = rng_orp.randint(0, 30)
            delta = mul_div_round(base_amt, noise_pct, 100)
            sign = rng_orp.choice([1, -1])
            orphan_amt = max(100, base_amt + sign * delta)
        else:
            orphan_amt = rng_orp.randint(50000, 500000)

        stmt_id = f"stmt_orp_{len(data['bank_statement']) + 1:07d}"

        orp_roll = rng_orp.random()
        if orp_roll < 0.70:
            orp_variant = "clean"
            narration = f"CMS/RAZORPAY/{fake_utr}/SETTLEMENT"
        elif orp_roll < 0.90:
            orp_variant = "noisy"
            noise_choice = rng_orp.choice(["lower", "spaces", "trunc"])
            if noise_choice == "lower":
                narration = f"cms/razorpay/{fake_utr.lower()}/settlement"
            elif noise_choice == "spaces":
                narration = f"CMS / RAZORPAY / {fake_utr} / SETTLEMENT"
            else:
                narration = f"RAZORPAY CMS {fake_utr[:12]}"
        else:
            orp_variant = "no_utr"
            narration = "RAZORPAY PAYMENT SERVICES PRIVATE LIMITED CR"

        if "bank_narration_variants" in data:
            data["bank_narration_variants"][stmt_id] = orp_variant

        data["bank_statement"].append(
            {
                "statement_id": stmt_id,
                "merchant_id": m_id,
                "date": rnd_date.isoformat(),
                "description": narration,
                "debit": 0,
                "credit": orphan_amt,
                "balance": 0,
                "_orig_seq": len(data["bank_statement"]),
            }
        )

        add_label(
            merchant_id=m_id,
            error_class="ORPHAN_BANK_CREDIT",
            hard_negative_type=None,
            entity_ids=[stmt_id],
            settlement_id=None,
            expected_action="flag_orphan_bank_credit",
        )

    # =========================================================================
    # Hard Negatives (9 types)
    # =========================================================================

    # 1. ROUNDING (3% of card/netbanking items; <= 2 paise)
    eligible_rounding = [
        r
        for r in data["recon_items"]
        if r["type"] == "payment"
        and r["method"] in ("card", "netbanking")
        and r["entity_id"] not in used_entities
        and r["settlement_id"] not in used_batches
    ]
    rng_rnd.shuffle(eligible_rounding)
    rounding_count = int(round(len(eligible_rounding) * get_rate("ROUNDING", 0.030)))

    for r in eligible_rounding[:rounding_count]:
        stl = settlements_by_id.get(r["settlement_id"])
        bk_row = bank_rows_by_settlement.get(r["settlement_id"])
        if not stl or not bk_row:
            continue

        delta = rng_rnd.choice([-2, -1, 1, 2])
        if stl["amount"] - delta <= 0 or r["credit"] - delta <= 0:
            continue

        r["fee"] += delta
        r["credit"] -= delta
        stl["amount"] -= delta
        bk_row["credit"] -= delta
        bk_row["balance"] -= delta

        used_entities.add(r["entity_id"])

        add_label(
            merchant_id=r["merchant_id"],
            error_class=None,
            hard_negative_type="ROUNDING",
            entity_ids=[r["entity_id"]],
            settlement_id=stl["settlement_id"],
            expected_action="benign_ignore",
        )

    # 2. LEGIT_ADJUSTMENT (documented adjustment credits)
    for r in data["recon_items"]:
        if r["type"] == "adjustment":
            add_label(
                merchant_id=r["merchant_id"],
                error_class=None,
                hard_negative_type="LEGIT_ADJUSTMENT",
                entity_ids=[r["entity_id"]],
                settlement_id=r["settlement_id"],
                expected_action="benign_ignore",
            )

    # 3. LATE_SETTLEMENT (guaranteed actual 1-2 working-day shift; skip if cannot apply)
    eligible_late_batches = [
        s for s in data["settlements"] if s["settlement_id"] not in used_batches
    ]
    rng_lat.shuffle(eligible_late_batches)
    late_count = int(round(len(data["settlements"]) * get_rate("LATE_SETTLEMENT", 0.035)))

    for stl in eligible_late_batches[:late_count]:
        bk_row = bank_rows_by_settlement.get(stl["settlement_id"])
        if not bk_row:
            continue

        old_dt = datetime.datetime.fromtimestamp(stl["settled_at"], tz=IST_TZ).date()
        try:
            old_idx = working_days.index(old_dt)
        except ValueError:
            continue

        shift_days = rng_lat.choice([1, 2])
        new_idx = old_idx + shift_days
        if new_idx >= len(working_days):
            continue
        new_date = working_days[new_idx]
        if new_date > as_of_date:
            continue

        new_settle_dt = datetime.datetime.combine(new_date, datetime.time(18, 0, 0), tzinfo=IST_TZ)
        new_timestamp = int(new_settle_dt.timestamp())

        stl["settled_at"] = new_timestamp
        bk_row["date"] = new_date.isoformat()

        # Update recon items in this batch
        for r in data["recon_items"]:
            if r["settlement_id"] == stl["settlement_id"]:
                r["settled_at"] = new_timestamp

        used_batches.add(stl["settlement_id"])

        add_label(
            merchant_id=stl["merchant_id"],
            error_class=None,
            hard_negative_type="LATE_SETTLEMENT",
            entity_ids=[stl["settlement_id"]],
            settlement_id=stl["settlement_id"],
            expected_action="benign_ignore",
        )

    # 4. RETRY_AFTER_FAIL (same order, equal amount, created 30-600s before captured payment, max 1 failed per captured order)
    captured_orders_for_retry = [
        o for o in data["orders_ledger"]
        if o["status"] in ("captured", "refunded")
        and o["order_id"] not in used_entities
    ]
    rng_ret.shuffle(captured_orders_for_retry)
    retry_count = int(round(len(captured_orders_for_retry) * get_rate("RETRY_AFTER_FAIL", 0.030)))

    retry_failed_pids: set[str] = set()
    for order in captured_orders_for_retry[:retry_count]:
        p_cap = payments_by_id[order["payment_id"]]
        fail_id = _gen_base62_id(rng_ret, "pay_", seen_ids)
        gap = rng_ret.randint(30, 600)

        p_fail = {
            "payment_id": fail_id,
            "order_id": order["order_id"],
            "merchant_id": order["merchant_id"],
            "amount": order["expected_amount"],  # Exactly equals order expected amount (satisfies I4)
            "currency": "INR",
            "status": "failed",
            "method": p_cap["method"],
            "created_at": p_cap["created_at"] - gap,
            "captured_at": None,
            "error_code": "BAD_REQUEST_ERROR",
            "error_description": "Payment authorization failed",
        }
        data["payments"].append(p_fail)
        payments_by_id[fail_id] = p_fail
        retry_failed_pids.add(fail_id)

        add_label(
            merchant_id=order["merchant_id"],
            error_class=None,
            hard_negative_type="RETRY_AFTER_FAIL",
            entity_ids=[fail_id, p_cap["payment_id"], order["order_id"]],
            settlement_id=None,
            expected_action="benign_ignore",
        )

    # 5. NOISY_BANK_NARRATION & 6. NO_UTR_NARRATION (derived strictly from side table, no string sniffing)
    variants = data.get("bank_narration_variants", {})
    for s in data["settlements"]:
        if s["settlement_id"] in used_batches:
            continue
        bk = bank_rows_by_settlement.get(s["settlement_id"])
        if not bk:
            continue
        var = variants.get(bk["statement_id"])
        if var == "noisy":
            add_label(
                merchant_id=s["merchant_id"],
                error_class=None,
                hard_negative_type="NOISY_BANK_NARRATION",
                entity_ids=[bk["statement_id"]],
                settlement_id=s["settlement_id"],
                expected_action="benign_ignore",
            )
        elif var == "no_utr":
            add_label(
                merchant_id=s["merchant_id"],
                error_class=None,
                hard_negative_type="NO_UTR_NARRATION",
                entity_ids=[bk["statement_id"]],
                settlement_id=s["settlement_id"],
                expected_action="benign_ignore",
            )

    # 7. UNRELATED_BANK_CREDIT (external credits recorded with 'unrelated' variant)
    for b in data["bank_statement"]:
        if b["credit"] > 0 and variants.get(b["statement_id"]) == "unrelated":
            add_label(
                merchant_id=b["merchant_id"],
                error_class=None,
                hard_negative_type="UNRELATED_BANK_CREDIT",
                entity_ids=[b["statement_id"]],
                settlement_id=None,
                expected_action="benign_ignore",
            )

    # 8. FAILED_PAYMENT (independent failed authorization attempts not part of retry)
    for p in data["payments"]:
        if p["status"] == "failed" and p["payment_id"] not in retry_failed_pids:
            add_label(
                merchant_id=p["merchant_id"],
                error_class=None,
                hard_negative_type="FAILED_PAYMENT",
                entity_ids=[p["payment_id"]],
                settlement_id=None,
                expected_action="benign_ignore",
            )

    # 9. PARTIAL_REFUND_CONSISTENT (legitimate partial refund matching ledger)
    for r in data["recon_items"]:
        if r["type"] == "refund" and r["entity_id"] not in used_entities:
            add_label(
                merchant_id=r["merchant_id"],
                error_class=None,
                hard_negative_type="PARTIAL_REFUND_CONSISTENT",
                entity_ids=[r["entity_id"], r["payment_id"]],
                settlement_id=r["settlement_id"],
                expected_action="benign_ignore",
            )

    # =========================================================================
    # Finalize Canonical Ordering: Sort Payments, Recon Items, and Bank Statements
    # =========================================================================
    data["labels"] = labels

    rng_order = random.Random(config.seed * 1000 + 99)
    data["payments"].sort(key=lambda x: (x["created_at"], rng_order.random()))
    data["recon_items"].sort(key=lambda x: (x["created_at"], rng_order.random()))
    finalize_bank_statements(data, rng_order)

    return labels
