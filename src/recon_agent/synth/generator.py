"""Deterministic synthetic data generator for clean Razorpay recon data."""

import csv
import datetime
import json
import math
import os
import random
from typing import Any

from src.recon_agent.synth.config import SynthConfig
from src.recon_agent.synth.math_utils import mul_div_round

BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
ALPHANUM_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def _gen_base62_id(rng: random.Random, prefix: str, seen: set[str]) -> str:
    """Generate unique ID with prefix + 14 base62 characters."""
    while True:
        token = "".join(rng.choices(BASE62_ALPHABET, k=14))
        cand = f"{prefix}{token}"
        if cand not in seen:
            seen.add(cand)
            return cand


def _gen_utr(rng: random.Random, seen: set[str]) -> str:
    """Generate unique 16-character alphanumeric UTR."""
    while True:
        cand = "".join(rng.choices(ALPHANUM_UPPER, k=16))
        if cand not in seen:
            seen.add(cand)
            return cand


def _poisson_knuth(rng: random.Random, lam: float) -> int:
    """Sample from Poisson distribution using Knuth's algorithm."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def get_working_days(start_date_str: str, count: int) -> list[datetime.date]:
    """Derive Mon-Fri working calendar days starting from start_date_str."""
    curr = datetime.date.fromisoformat(start_date_str)
    days: list[datetime.date] = []
    while len(days) < count:
        if curr.weekday() < 5:  # Monday=0, Friday=4
            days.append(curr)
        curr += datetime.timedelta(days=1)
    return days


class SyntheticDataGenerator:
    """Clean deterministic generator for recon-agent synthetic datasets."""

    def __init__(self, config: SynthConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.seen_ids: set[str] = set()
        self.seen_utrs: set[str] = set()

    def generate(self) -> dict[str, Any]:
        """Execute full data generation pipeline and return structured datasets."""
        # 1. Merchants & Calendar
        merchants = [f"mer_{i+1:02d}" for i in range(self.config.num_merchants)]
        # Generate extra working days so settlements past day 22 have defined dates
        working_days = get_working_days(self.config.start_date, self.config.working_days + 15)

        payments: list[dict[str, Any]] = []
        orders_ledger: list[dict[str, Any]] = []
        settlements: list[dict[str, Any]] = []
        recon_items: list[dict[str, Any]] = []
        bank_statements: list[dict[str, Any]] = []

        total_adjustment_count = 0

        # Method choices and weights
        methods = ["card", "upi", "netbanking"]
        method_weights = [
            self.config.method_mix["card"],
            self.config.method_mix["upi"],
            self.config.method_mix["netbanking"],
        ]

        for m_idx, merchant_id in enumerate(merchants):
            # Track pending items per settlement day: day_idx -> list of recon items
            pending_settlement_items: dict[int, list[dict[str, Any]]] = {}
            running_bank_balance = 10000000 + m_idx * 500000  # Initial balance in paise (~₹100,000)

            # Generate payments & orders across 22 working days
            for day_idx in range(self.config.working_days):
                work_date = working_days[day_idx]
                num_attempts = _poisson_knuth(self.rng, self.config.avg_payments_per_day)

                for _ in range(num_attempts):
                    # Operating timestamp during day: 09:00:00 to 20:59:59 IST
                    sec_offset = self.rng.randint(9 * 3600, 21 * 3600 - 1)
                    tx_time = datetime.datetime.combine(
                        work_date,
                        datetime.time(0, 0, 0),
                        tzinfo=IST_TZ,
                    ) + datetime.timedelta(seconds=sec_offset)
                    tx_timestamp = int(tx_time.timestamp())

                    order_id = _gen_base62_id(self.rng, "order_", self.seen_ids)
                    payment_id = _gen_base62_id(self.rng, "pay_", self.seen_ids)
                    receipt_num = f"rcpt_{self.rng.randint(100000, 999999)}"

                    # Lognormal integer paise amount
                    raw_amount = self.rng.lognormvariate(
                        self.config.lognormal_mu, self.config.lognormal_sigma
                    )
                    amount = max(self.config.min_amount_paise, int(round(raw_amount)))

                    method = self.rng.choices(methods, weights=method_weights, k=1)[0]
                    card_network = None
                    card_issuer = None
                    card_type = None

                    if method == "card":
                        card_network = self.rng.choice(["MasterCard", "Visa", "RuPay"])
                        card_issuer = self.rng.choice(["HDFC", "ICICI", "SBI", "AXIS"])
                        card_type = self.rng.choice(["credit", "debit"])

                    # Determine failure (~6% failed rate)
                    is_failed = self.rng.randint(1, 10000) <= self.config.failed_rate_bps

                    if is_failed:
                        payments.append(
                            {
                                "payment_id": payment_id,
                                "order_id": order_id,
                                "merchant_id": merchant_id,
                                "amount": amount,
                                "currency": "INR",
                                "status": "failed",
                                "method": method,
                                "created_at": tx_timestamp,
                                "captured_at": None,
                                "error_code": "BAD_REQUEST_ERROR",
                                "error_description": "Payment authorization failed",
                            }
                        )
                        orders_ledger.append(
                            {
                                "order_id": order_id,
                                "merchant_id": merchant_id,
                                "created_at": tx_timestamp,
                                "expected_amount": amount,
                                "currency": "INR",
                                "order_receipt": receipt_num,
                                "status": "failed",
                                "payment_id": payment_id,
                                "refund_id": "",
                                "refund_amount": 0,
                            }
                        )
                        # Failed payments NEVER enter recon or settlements
                        continue

                    # Captured payment
                    captured_at = tx_timestamp + self.rng.randint(2, 60)
                    payments.append(
                        {
                            "payment_id": payment_id,
                            "order_id": order_id,
                            "merchant_id": merchant_id,
                            "amount": amount,
                            "currency": "INR",
                            "status": "captured",
                            "method": method,
                            "created_at": tx_timestamp,
                            "captured_at": captured_at,
                            "error_code": None,
                            "error_description": None,
                        }
                    )

                    # Compute fee and tax in minor units
                    rate_bps = self.config.method_fee_bps[method]
                    base_fee = mul_div_round(amount, rate_bps, 10000)
                    tax = mul_div_round(base_fee, self.config.gst_bps, 10000)

                    if self.config.fee_includes_tax:
                        fee = base_fee + tax
                        credit = amount - fee
                    else:
                        fee = base_fee
                        credit = amount - fee - tax

                    # Recon item for payment (settlement_id and settlement_utr assigned during batching)
                    recon_payment_item = {
                        "entity_id": payment_id,
                        "type": "payment",
                        "debit": 0,
                        "credit": credit,
                        "amount": amount,
                        "currency": "INR",
                        "fee": fee,
                        "tax": tax,
                        "on_hold": False,
                        "settled": True,
                        "created_at": tx_timestamp,
                        "settled_at": 0,  # Updated in batch
                        "settlement_id": "",
                        "description": f"Payment via {method.capitalize()}",
                        "notes": {"merchant_order_ref": receipt_num},
                        "payment_id": None,
                        "settlement_utr": None,
                        "order_id": order_id,
                        "order_receipt": receipt_num,
                        "method": method,
                        "card_network": card_network,
                        "card_issuer": card_issuer,
                        "card_type": card_type,
                        "dispute_id": None,
                        "merchant_id": merchant_id,
                    }

                    # Payment settles at C + 2 working days
                    payment_settle_day = day_idx + self.config.payment_settlement_lag_days
                    pending_settlement_items.setdefault(payment_settle_day, []).append(
                        recon_payment_item
                    )

                    # Check for refund (~4% of captured payments)
                    is_refunded = self.rng.randint(1, 10000) <= self.config.refund_rate_bps
                    refund_id_for_ledger = ""
                    refund_amt_for_ledger = 0

                    if is_refunded:
                        is_full = self.rng.randint(1, 100) <= self.config.full_refund_pct
                        if is_full or amount <= 100:
                            refund_amount = amount
                        else:
                            pct = self.rng.randint(20, 80)
                            refund_amount = max(100, mul_div_round(amount, pct, 100))
                            if refund_amount >= amount:
                                refund_amount = amount - 50

                        refund_id = _gen_base62_id(self.rng, "rfnd_", self.seen_ids)
                        refund_id_for_ledger = refund_id
                        refund_amt_for_ledger = refund_amount

                        # Refund processed after payment: within 1 working day
                        refund_time = tx_timestamp + self.rng.randint(3600, 24 * 3600)

                        recon_refund_item = {
                            "entity_id": refund_id,
                            "type": "refund",
                            "debit": refund_amount,
                            "credit": 0,
                            "amount": refund_amount,
                            "currency": "INR",
                            "fee": 0,
                            "tax": 0,
                            "on_hold": False,
                            "settled": True,
                            "created_at": refund_time,
                            "settled_at": 0,
                            "settlement_id": "",
                            "description": "Refund processed",
                            "notes": None,
                            "payment_id": payment_id,
                            "settlement_utr": None,
                            "order_id": order_id,
                            "order_receipt": receipt_num,
                            "method": method,
                            "card_network": card_network,
                            "card_issuer": card_issuer,
                            "card_type": card_type,
                            "dispute_id": None,
                            "merchant_id": merchant_id,
                        }

                        # Refund settles 1 working day after processing
                        refund_settle_day = min(
                            day_idx + 1 + self.config.refund_settlement_lag_days,
                            len(working_days) - 1,
                        )
                        pending_settlement_items.setdefault(refund_settle_day, []).append(
                            recon_refund_item
                        )

                    orders_ledger.append(
                        {
                            "order_id": order_id,
                            "merchant_id": merchant_id,
                            "created_at": tx_timestamp,
                            "expected_amount": amount,
                            "currency": "INR",
                            "order_receipt": receipt_num,
                            "status": "refunded" if is_refunded else "captured",
                            "payment_id": payment_id,
                            "refund_id": refund_id_for_ledger,
                            "refund_amount": refund_amt_for_ledger,
                        }
                    )

            # Process settlement batches day-by-day for this merchant
            # Guarantee every refund debit is processed and deferred if net would be <= 0
            deferred_refunds: list[dict[str, Any]] = []
            max_settle_day = max(pending_settlement_items.keys()) if pending_settlement_items else 0

            for s_day in range(max_settle_day + 1):
                day_items = pending_settlement_items.get(s_day, [])
                payments_in_batch = [item for item in day_items if item["type"] == "payment"]
                refunds_in_batch = deferred_refunds + [
                    item for item in day_items if item["type"] == "refund"
                ]
                deferred_refunds = []

                if not payments_in_batch and not refunds_in_batch:
                    continue

                # Legitimate adjustment (~0.3% chance per batch)
                adjustment_items: list[dict[str, Any]] = []
                if payments_in_batch and self.rng.randint(1, 10000) <= self.config.adjustment_prob_bps:
                    adj_id = _gen_base62_id(self.rng, "adj_", self.seen_ids)
                    adj_amount = self.rng.randint(100, 2500)  # ₹1 to ₹25 adjustment credit
                    adj_item = {
                        "entity_id": adj_id,
                        "type": "adjustment",
                        "debit": 0,
                        "credit": adj_amount,
                        "amount": adj_amount,
                        "currency": "INR",
                        "fee": 0,
                        "tax": 0,
                        "on_hold": False,
                        "settled": True,
                        "created_at": 0,
                        "settled_at": 0,
                        "settlement_id": "",
                        "description": "Fee dispute credit adjustment",
                        "notes": None,
                        "payment_id": None,
                        "settlement_utr": None,
                        "order_id": None,
                        "order_receipt": None,
                        "method": None,
                        "card_network": None,
                        "card_issuer": None,
                        "card_type": None,
                        "dispute_id": None,
                        "merchant_id": merchant_id,
                    }
                    adjustment_items.append(adj_item)
                    total_adjustment_count += 1

                total_credits = sum(item["credit"] for item in payments_in_batch) + sum(
                    item["credit"] for item in adjustment_items
                )

                # Defer refund debits if net would be <= 0
                accepted_refunds: list[dict[str, Any]] = []
                current_debit = 0
                for r_item in refunds_in_batch:
                    if total_credits - (current_debit + r_item["debit"]) > 0:
                        accepted_refunds.append(r_item)
                        current_debit += r_item["debit"]
                    else:
                        deferred_refunds.append(r_item)

                if not payments_in_batch and not accepted_refunds and not adjustment_items:
                    continue

                # Finalize batch
                final_batch_items = payments_in_batch + accepted_refunds + adjustment_items
                settlement_net = sum(item["credit"] for item in final_batch_items) - sum(
                    item["debit"] for item in final_batch_items
                )

                settle_date = working_days[s_day]
                settle_dt = datetime.datetime.combine(
                    settle_date,
                    datetime.time(18, 0, 0),
                    tzinfo=IST_TZ,
                )
                settle_timestamp = int(settle_dt.timestamp())

                settlement_id = _gen_base62_id(self.rng, "setl_", self.seen_ids)
                settlement_utr = _gen_utr(self.rng, self.seen_utrs)

                for item in final_batch_items:
                    item["settlement_id"] = settlement_id
                    item["settlement_utr"] = settlement_utr
                    item["settled_at"] = settle_timestamp
                    if item["created_at"] == 0:
                        item["created_at"] = settle_timestamp
                    recon_items.append(item)

                settlements.append(
                    {
                        "settlement_id": settlement_id,
                        "merchant_id": merchant_id,
                        "amount": settlement_net,
                        "utr": settlement_utr,
                        "status": "processed",
                        "created_at": settle_timestamp,
                        "settled_at": settle_timestamp,
                    }
                )

                # Bank statement entry for this settlement (one credit per settlement on settle date)
                # Narration mix: 70% clean UTR, 20% noisy UTR, 10% no UTR
                narration_roll = self.rng.random()
                if narration_roll < 0.70:
                    narration = f"CMS/RAZORPAY/{settlement_utr}/SETTLEMENT"
                elif narration_roll < 0.90:
                    noise_choice = self.rng.choice(["lower", "spaces", "trunc"])
                    if noise_choice == "lower":
                        narration = f"cms/razorpay/{settlement_utr.lower()}/settlement"
                    elif noise_choice == "spaces":
                        narration = f"CMS / RAZORPAY / {settlement_utr} / SETTLEMENT"
                    else:
                        narration = f"RAZORPAY CMS {settlement_utr[:12]}"
                else:
                    narration = "RAZORPAY PAYMENT SERVICES PRIVATE LIMITED CR"

                stmt_id = f"stmt_{len(bank_statements) + 1:07d}"
                running_bank_balance += settlement_net
                bank_statements.append(
                    {
                        "statement_id": stmt_id,
                        "merchant_id": merchant_id,
                        "date": settle_date.isoformat(),
                        "description": narration,
                        "debit": 0,
                        "credit": settlement_net,
                        "balance": running_bank_balance,
                    }
                )

            # Add ~30 unrelated bank rows per merchant-month (salary, rent, other gateway)
            unrelated_descriptions = [
                ("OFFICE RENT LEASE AUG", 1500000, 0),
                ("EMPLOYEE PAYROLL HDFC", 3500000, 0),
                ("ELECTRICITY UTILITY BESCOM", 45000, 0),
                ("CLOUD SERVICES AWS INDIA", 120000, 0),
                ("VENDOR PAYOUT TECH CORPS", 280000, 0),
                ("STRIPE PAYMENTS SETTLEMENT", 0, 1850000),
                ("PAYTM PG SETTLEMENT", 0, 920000),
                ("INTEREST CREDIT FIXED DEP", 0, 35000),
            ]
            for _ in range(self.config.unrelated_bank_rows_per_merchant):
                rnd_day = self.rng.choice(working_days[: self.config.working_days])
                desc, d_amt, c_amt = self.rng.choice(unrelated_descriptions)
                noise_amt = self.rng.randint(-5000, 5000)
                if d_amt > 0:
                    d_final = max(1000, d_amt + noise_amt)
                    c_final = 0
                else:
                    d_final = 0
                    c_final = max(1000, c_amt + noise_amt)

                running_bank_balance = running_bank_balance + c_final - d_final
                stmt_id = f"stmt_{len(bank_statements) + 1:07d}"
                bank_statements.append(
                    {
                        "statement_id": stmt_id,
                        "merchant_id": merchant_id,
                        "date": rnd_day.isoformat(),
                        "description": desc,
                        "debit": d_final,
                        "credit": c_final,
                        "balance": running_bank_balance,
                    }
                )

        # Sort datasets chronologically where appropriate
        payments.sort(key=lambda x: x["created_at"])
        orders_ledger.sort(key=lambda x: x["created_at"])
        settlements.sort(key=lambda x: (x["settled_at"], x["settlement_id"]))
        recon_items.sort(key=lambda x: (x["settled_at"], x["created_at"], x["entity_id"]))
        bank_statements.sort(key=lambda x: (x["date"], x["statement_id"]))

        # Build manifest
        captured_count = sum(1 for p in payments if p["status"] == "captured")
        failed_count = sum(1 for p in payments if p["status"] == "failed")
        refund_count = sum(1 for r in recon_items if r["type"] == "refund")

        counts = {
            "merchant_count": len(merchants),
            "payment_count": len(payments),
            "captured_count": captured_count,
            "failed_count": failed_count,
            "refund_count": refund_count,
            "settlement_count": len(settlements),
            "recon_item_count": len(recon_items),
            "bank_statement_count": len(bank_statements),
            "adjustment_count": total_adjustment_count,
        }

        manifest = {
            "seed": self.config.seed,
            "config_hash": self.config.config_hash(),
            "generator_version": self.config.generator_version,
            "counts": counts,
        }

        rate_card = {
            "card_bps": self.config.method_fee_bps["card"],
            "upi_bps": self.config.method_fee_bps["upi"],
            "netbanking_bps": self.config.method_fee_bps["netbanking"],
            "gst_bps": self.config.gst_bps,
            "fee_includes_tax": self.config.fee_includes_tax,
        }

        return {
            "recon_items": recon_items,
            "payments": payments,
            "orders_ledger": orders_ledger,
            "settlements": settlements,
            "bank_statement": bank_statements,
            "rate_card": rate_card,
            "manifest": manifest,
        }

    def write_to_directory(self, out_dir: str) -> dict[str, str]:
        """Generate and save all dataset files to specified directory."""
        data = self.generate()
        os.makedirs(out_dir, exist_ok=True)
        paths: dict[str, str] = {}

        # 1. recon_items.jsonl
        recon_path = os.path.join(out_dir, "recon_items.jsonl")
        with open(recon_path, "w", encoding="utf-8") as f:
            for item in data["recon_items"]:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        paths["recon_items"] = recon_path

        # 2. payments.jsonl
        payments_path = os.path.join(out_dir, "payments.jsonl")
        with open(payments_path, "w", encoding="utf-8") as f:
            for item in data["payments"]:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        paths["payments"] = payments_path

        # 3. settlements.jsonl
        settlements_path = os.path.join(out_dir, "settlements.jsonl")
        with open(settlements_path, "w", encoding="utf-8") as f:
            for item in data["settlements"]:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        paths["settlements"] = settlements_path

        # 4. orders_ledger.csv
        orders_path = os.path.join(out_dir, "orders_ledger.csv")
        if data["orders_ledger"]:
            keys = list(data["orders_ledger"][0].keys())
            with open(orders_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data["orders_ledger"])
        paths["orders_ledger"] = orders_path

        # 5. bank_statement.csv
        bank_path = os.path.join(out_dir, "bank_statement.csv")
        if data["bank_statement"]:
            keys = list(data["bank_statement"][0].keys())
            with open(bank_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data["bank_statement"])
        paths["bank_statement"] = bank_path

        # 6. rate_card.json
        rate_card_path = os.path.join(out_dir, "rate_card.json")
        with open(rate_card_path, "w", encoding="utf-8") as f:
            json.dump(data["rate_card"], f, indent=2)
        paths["rate_card"] = rate_card_path

        # 7. manifest.json
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data["manifest"], f, indent=2)
        paths["manifest"] = manifest_path

        return paths
