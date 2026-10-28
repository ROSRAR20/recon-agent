"""Error injection and hard-negative generator for recon-agent synthetic datasets."""

import datetime
import random
from typing import Any

from src.recon_agent.synth.config import SynthConfig
from src.recon_agent.synth.generator import IST_TZ, _gen_base62_id, _gen_utr
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

    Uses a dedicated, isolated RNG stream seeded distinctly from clean generation
    so enabling/disabling injection does not perturb clean data structure.
    """
    rng = random.Random(config.seed + 10000)

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

    # 1. DUPLICATE_PAYMENT (1.0% of captured orders)
    captured_orders = [
        o
        for o in data["orders_ledger"]
        if o["status"] == "captured"
        and o["order_id"] not in used_entities
        and o["payment_id"] not in used_entities
    ]
    rng.shuffle(captured_orders)
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

        p2_id = _gen_base62_id(rng, "pay_", seen_ids)
        p2 = dict(p1)
        p2["payment_id"] = p2_id
        p2["created_at"] = p1["created_at"] + 1
        p2["captured_at"] = (p1["captured_at"] or p1["created_at"]) + 1
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

    rng.shuffle(eligible_missing)
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
    rng.shuffle(eligible_feetax)
    feetax_count = int(round(len(eligible_feetax) * get_rate("FEE_TAX_MISMATCH", 0.012)))

    for idx, r in enumerate(eligible_feetax[:feetax_count]):
        stl = settlements_by_id.get(r["settlement_id"])
        bk_row = bank_rows_by_settlement.get(r["settlement_id"])
        if not stl or not bk_row:
            continue

        is_fee_off = (idx % 10) < 6  # 60% fee_off, 40% tax_off
        if is_fee_off:
            # Fee raised by 5-25%, at least +10 paise vs rate card
            delta_fee = max(10, mul_div_round(r["fee"], rng.randint(500, 2500), 10000))
            if stl["amount"] - delta_fee <= 0 or r["credit"] - delta_fee <= 0:
                continue
            r["fee"] += delta_fee
            r["credit"] -= delta_fee
            stl["amount"] -= delta_fee
            bk_row["credit"] -= delta_fee
            bk_row["balance"] -= delta_fee
        else:
            # Tax differs from expected tax by at least 5 paise
            delta_tax = max(5, mul_div_round(max(1, r["tax"]), rng.randint(1000, 3000), 10000))
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

    # 4. PARTIAL_REFUND_MISMATCH (15% of refund items)
    eligible_refunds = [
        r
        for r in data["recon_items"]
        if r["type"] == "refund"
        and r["entity_id"] not in used_entities
        and r["settlement_id"] not in used_batches
    ]
    rng.shuffle(eligible_refunds)
    refund_mismatch_count = int(round(len(eligible_refunds) * get_rate("PARTIAL_REFUND_MISMATCH", 0.150)))

    for idx, r in enumerate(eligible_refunds[:refund_mismatch_count]):
        stl = settlements_by_id.get(r["settlement_id"])
        bk_row = bank_rows_by_settlement.get(r["settlement_id"])
        order = orders_by_id.get(r["order_id"])
        if not stl or not bk_row or not order:
            continue

        is_debit_diff = (idx % 10) < 8  # 80% debit differs by >= 100 paise; 20% missing in ledger
        if is_debit_diff:
            diff_paise = max(100, mul_div_round(r["amount"], rng.randint(15, 35), 100))
            if stl["amount"] - diff_paise <= 0:
                continue
            r["amount"] += diff_paise
            r["debit"] += diff_paise
            stl["amount"] -= diff_paise
            bk_row["credit"] -= diff_paise
            bk_row["balance"] -= diff_paise
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

    rng.shuffle(eligible_cutoff)
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
        time_D = datetime.datetime.combine(day_D, datetime.time(23, rng.randint(40, 59), rng.randint(0, 59)), tzinfo=IST_TZ)
        time_D1 = datetime.datetime.combine(day_D1, datetime.time(0, rng.randint(5, 35), rng.randint(0, 59)), tzinfo=IST_TZ)

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

    # 6. SHORT_BANK_CREDIT (6% of eligible batches)
    eligible_short_batches = [
        s for s in data["settlements"] if s["settlement_id"] not in used_batches
    ]
    rng.shuffle(eligible_short_batches)
    short_count = int(round(len(data["settlements"]) * get_rate("SHORT_BANK_CREDIT", 0.060)))

    for stl in eligible_short_batches[:short_count]:
        bk_row = bank_rows_by_settlement.get(stl["settlement_id"])
        if not bk_row or stl["amount"] <= 200:
            continue

        short_pct = rng.randint(100, 500)  # 1.00% to 5.00%
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

    # 7. ORPHAN_BANK_CREDIT (extra rows = 6% of the number of batches)
    merchants = sorted(list({s["merchant_id"] for s in data["settlements"]}))
    orphan_count = int(round(len(data["settlements"]) * get_rate("ORPHAN_BANK_CREDIT", 0.060)))

    for _ in range(orphan_count):
        fake_utr = _gen_utr(rng, seen_utrs)
        m_id = rng.choice(merchants)
        rnd_date = rng.choice(working_days)
        orphan_amt = rng.randint(50000, 500000)
        stmt_id = f"stmt_{len(data['bank_statement']) + 1:07d}"

        narration = f"CMS/RAZORPAY/{fake_utr}/SETTLEMENT"
        data["bank_statement"].append(
            {
                "statement_id": stmt_id,
                "merchant_id": m_id,
                "date": rnd_date.isoformat(),
                "description": narration,
                "debit": 0,
                "credit": orphan_amt,
                "balance": 15000000 + orphan_amt,
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
    # Hard Negatives (7 types)
    # =========================================================================

    # 1. ROUNDING (3% of card/netbanking items)
    eligible_rounding = [
        r
        for r in data["recon_items"]
        if r["type"] == "payment"
        and r["method"] in ("card", "netbanking")
        and r["entity_id"] not in used_entities
        and r["settlement_id"] not in used_batches
    ]
    rng.shuffle(eligible_rounding)
    rounding_count = int(round(len(eligible_rounding) * get_rate("ROUNDING", 0.030)))

    for r in eligible_rounding[:rounding_count]:
        stl = settlements_by_id.get(r["settlement_id"])
        bk_row = bank_rows_by_settlement.get(r["settlement_id"])
        if not stl or not bk_row:
            continue

        delta = rng.choice([-2, -1, 1, 2])
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

    # 3. LATE_SETTLEMENT (batches shifted by 1-2 working days, remaining consistent)
    eligible_late_batches = [
        s for s in data["settlements"] if s["settlement_id"] not in used_batches
    ]
    rng.shuffle(eligible_late_batches)
    late_count = int(round(len(data["settlements"]) * get_rate("LATE_SETTLEMENT", 0.035)))

    for stl in eligible_late_batches[:late_count]:
        bk_row = bank_rows_by_settlement.get(stl["settlement_id"])
        if not bk_row:
            continue

        shift_days = rng.choice([1, 2])
        old_dt = datetime.datetime.fromtimestamp(stl["settled_at"], tz=IST_TZ).date()
        try:
            old_idx = working_days.index(old_dt)
            new_idx = min(old_idx + shift_days, len(working_days) - 1)
            new_date = working_days[new_idx]
        except ValueError:
            new_date = old_dt + datetime.timedelta(days=shift_days)

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

    # 4. RETRY_AFTER_FAIL (failed attempt followed by success for same order)
    failed_payments = [p for p in data["payments"] if p["status"] == "failed"]
    captured_by_merchant: dict[str, list[dict[str, Any]]] = {}
    for p in data["payments"]:
        if p["status"] == "captured":
            captured_by_merchant.setdefault(p["merchant_id"], []).append(p)

    rng.shuffle(failed_payments)
    retry_count = len(failed_payments) // 2

    for p_fail in failed_payments[:retry_count]:
        c_list = captured_by_merchant.get(p_fail["merchant_id"], [])
        if not c_list:
            continue
        target_captured = rng.choice(c_list)
        p_fail["order_id"] = target_captured["order_id"]

        add_label(
            merchant_id=p_fail["merchant_id"],
            error_class=None,
            hard_negative_type="RETRY_AFTER_FAIL",
            entity_ids=[p_fail["payment_id"], target_captured["payment_id"], target_captured["order_id"]],
            settlement_id=None,
            expected_action="benign_ignore",
        )

    # 5. MULTI_MERCHANT_SAME_UTR (bank-side batching of payouts across merchants under same UTR)
    # Group available settlement batches by date
    batches_by_date: dict[str, list[dict[str, Any]]] = {}
    for s in data["settlements"]:
        if s["settlement_id"] not in used_batches:
            dt_str = datetime.datetime.fromtimestamp(s["settled_at"], tz=IST_TZ).date().isoformat()
            batches_by_date.setdefault(dt_str, []).append(s)

    # Collect distinct merchant pairs on same date
    eligible_multi_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for dt_str, b_list in batches_by_date.items():
        if len(b_list) >= 2:
            for i in range(len(b_list) - 1):
                s1 = b_list[i]
                s2 = b_list[i + 1]
                if s1["merchant_id"] != s2["merchant_id"]:
                    eligible_multi_pairs.append((s1, s2))

    rng.shuffle(eligible_multi_pairs)
    # Target at least 25 pairs (= 50 labels) in full profile, or proportional in small profile
    target_multi_pairs = min(
        len(eligible_multi_pairs),
        max(5, int(round(25 * get_rate("MULTI_MERCHANT_SAME_UTR", 1.0))))
    )

    for s1, s2 in eligible_multi_pairs[:target_multi_pairs]:
        if s1["settlement_id"] in used_batches or s2["settlement_id"] in used_batches:
            continue
        bk1 = bank_rows_by_settlement.get(s1["settlement_id"])
        bk2 = bank_rows_by_settlement.get(s2["settlement_id"])
        if not bk1 or not bk2:
            continue

        shared_utr = s1["utr"]
        s2["utr"] = shared_utr

        # Update recon items for s2
        for r in data["recon_items"]:
            if r["settlement_id"] == s2["settlement_id"]:
                r["settlement_utr"] = shared_utr

        # Update bank narration for s2
        bk2["description"] = f"CMS/RAZORPAY/{shared_utr}/SETTLEMENT"

        used_batches.update([s1["settlement_id"], s2["settlement_id"]])

        # Label both merchants' settlements as MULTI_MERCHANT_SAME_UTR
        add_label(
            merchant_id=s1["merchant_id"],
            error_class=None,
            hard_negative_type="MULTI_MERCHANT_SAME_UTR",
            entity_ids=[s1["settlement_id"], s2["settlement_id"]],
            settlement_id=s1["settlement_id"],
            expected_action="benign_ignore",
        )
        add_label(
            merchant_id=s2["merchant_id"],
            error_class=None,
            hard_negative_type="MULTI_MERCHANT_SAME_UTR",
            entity_ids=[s2["settlement_id"], s1["settlement_id"]],
            settlement_id=s2["settlement_id"],
            expected_action="benign_ignore",
        )

    # 6. NOISY_BANK_NARRATION (lowercase, trailing spaces, truncated UTR in bank narration)
    for s in data["settlements"]:
        if s["settlement_id"] in used_batches:
            continue
        bk = bank_rows_by_settlement.get(s["settlement_id"])
        if not bk:
            continue
        desc = bk["description"]
        is_noisy = (
            desc.startswith("cms/razorpay")
            or " / " in desc
            or (s["utr"][:12] in desc and s["utr"] not in desc)
            or desc.endswith(" CR")
        )
        if is_noisy:
            add_label(
                merchant_id=s["merchant_id"],
                error_class=None,
                hard_negative_type="NOISY_BANK_NARRATION",
                entity_ids=[bk["statement_id"]],
                settlement_id=s["settlement_id"],
                expected_action="benign_ignore",
            )

    # 7. UNRELATED_BANK_CREDIT (other-gateway payouts, payroll, interest present in statement)
    for b in data["bank_statement"]:
        if (
            b["credit"] > 0
            and "RAZORPAY" not in b["description"]
            and "razorpay" not in b["description"].lower()
        ):
            add_label(
                merchant_id=b["merchant_id"],
                error_class=None,
                hard_negative_type="UNRELATED_BANK_CREDIT",
                entity_ids=[b["statement_id"]],
                settlement_id=None,
                expected_action="benign_ignore",
            )

    return labels
