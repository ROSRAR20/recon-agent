"""Deterministic oracle audit for synthetic reconciliation datasets.

Pure observable file audit (imports csv, datetime, json, os, sys only; never imports recon_agent.synth).
Evaluates 7 error classes directly from observable files:
  1. DUPLICATE_PAYMENT
  2. MISSING_SETTLEMENT
  3. FEE_TAX_MISMATCH
  4. PARTIAL_REFUND_MISMATCH
  5. CUTOFF_DRIFT
  6. SHORT_BANK_CREDIT
  7. ORPHAN_BANK_CREDIT
"""

import csv
import datetime
import json
import os
import sys
from typing import Any

IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

REQUIRED_ERROR_CLASSES = [
    "DUPLICATE_PAYMENT",
    "MISSING_SETTLEMENT",
    "FEE_TAX_MISMATCH",
    "PARTIAL_REFUND_MISMATCH",
    "CUTOFF_DRIFT",
    "SHORT_BANK_CREDIT",
    "ORPHAN_BANK_CREDIT",
]


def mul_div_round(x: int, num: int, den: int) -> int:
    """Integer half-up multiplication and division: round(x * num / den)."""
    if den == 0:
        raise ZeroDivisionError("den cannot be zero")
    return (2 * x * num + den) // (2 * den)


def add_working_days(start_date: datetime.date, n: int) -> datetime.date:
    """Add n Mon-Fri working days to start_date."""
    curr = start_date
    added = 0
    while added < n:
        curr += datetime.timedelta(days=1)
        if curr.weekday() < 5:
            added += 1
    return curr


def load_dataset_from_directory(dir_path: str) -> dict[str, Any]:
    """Load all recon-agent dataset files from a directory."""
    data: dict[str, Any] = {}

    # 1. manifest.json
    manifest_path = os.path.join(dir_path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data["manifest"] = json.load(f)
    else:
        data["manifest"] = {}

    # 2. rate_card.json
    rate_card_path = os.path.join(dir_path, "rate_card.json")
    if os.path.exists(rate_card_path):
        with open(rate_card_path, "r", encoding="utf-8") as f:
            data["rate_card"] = json.load(f)
    else:
        data["rate_card"] = {}

    # 3. payments.jsonl
    payments_path = os.path.join(dir_path, "payments.jsonl")
    data["payments"] = []
    if os.path.exists(payments_path):
        with open(payments_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data["payments"].append(json.loads(line))

    # 4. recon_items.jsonl
    recon_path = os.path.join(dir_path, "recon_items.jsonl")
    data["recon_items"] = []
    if os.path.exists(recon_path):
        with open(recon_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data["recon_items"].append(json.loads(line))

    # 5. settlements.jsonl
    settlements_path = os.path.join(dir_path, "settlements.jsonl")
    data["settlements"] = []
    if os.path.exists(settlements_path):
        with open(settlements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data["settlements"].append(json.loads(line))

    # 6. orders_ledger.csv
    orders_path = os.path.join(dir_path, "orders_ledger.csv")
    data["orders_ledger"] = []
    if os.path.exists(orders_path):
        with open(orders_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Type conversions
                row["created_at"] = int(row["created_at"]) if row.get("created_at") else 0
                row["expected_amount"] = int(row["expected_amount"]) if row.get("expected_amount") else 0
                row["refund_amount"] = int(row["refund_amount"]) if row.get("refund_amount") else 0
                data["orders_ledger"].append(row)

    # 7. bank_statement.csv
    bank_path = os.path.join(dir_path, "bank_statement.csv")
    data["bank_statement"] = []
    if os.path.exists(bank_path):
        with open(bank_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["credit"] = int(row["credit"]) if row.get("credit") else 0
                row["debit"] = int(row["debit"]) if row.get("debit") else 0
                row["balance"] = int(row["balance"]) if row.get("balance") else 0
                data["bank_statement"].append(row)

    # 8. labels.jsonl (if present)
    labels_path = os.path.join(dir_path, "labels.jsonl")
    data["labels"] = []
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data["labels"].append(json.loads(line))

    return data


# =============================================================================
# Detection Rules (Pure Observable Analysis)
# =============================================================================


def detect_duplicate_payments(payments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule 1: Multiple captured payments for one order with same amount."""
    by_order: dict[str, list[dict[str, Any]]] = {}
    for p in payments:
        if p.get("status") == "captured":
            by_order.setdefault(p["order_id"], []).append(p)

    detected: list[dict[str, Any]] = []
    for order_id, plist in by_order.items():
        if len(plist) > 1:
            # Group by amount
            by_amount: dict[int, list[dict[str, Any]]] = {}
            for p in plist:
                by_amount.setdefault(p["amount"], []).append(p)
            for amt, dups in by_amount.items():
                if len(dups) > 1:
                    pids = [p["payment_id"] for p in dups]
                    detected.append({
                        "order_id": order_id,
                        "payment_ids": pids,
                        "merchant_id": dups[0]["merchant_id"],
                        "amount": amt,
                        "entity_ids": set(pids) | {order_id},
                    })
    return detected


def detect_missing_settlements(
    payments: list[dict[str, Any]],
    recon_items: list[dict[str, Any]],
    rate_card: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rule 2: Captured payment with no recon item and expected_settle_date + missing_after_days <= as_of."""
    recon_pids = {
        r.get("entity_id")
        for r in recon_items
        if r.get("type") == "payment"
    }

    as_of_str = manifest.get("as_of")
    if not as_of_str:
        return []
    as_of_date = datetime.date.fromisoformat(as_of_str)

    settle_lag = rate_card.get("payment_settlement_lag_days", 2)
    missing_after_days = rate_card.get("missing_after_days", 7)

    detected: list[dict[str, Any]] = []
    for p in payments:
        if p.get("status") == "captured" and p["payment_id"] not in recon_pids:
            cap_ts = p.get("captured_at") or p.get("created_at")
            cap_date = datetime.datetime.fromtimestamp(cap_ts, tz=IST_TZ).date()
            expected_settle_date = add_working_days(cap_date, settle_lag)
            if expected_settle_date + datetime.timedelta(days=missing_after_days) <= as_of_date:
                detected.append({
                    "payment_id": p["payment_id"],
                    "order_id": p.get("order_id"),
                    "merchant_id": p.get("merchant_id"),
                    "entity_ids": {p["payment_id"]},
                })
    return detected


def detect_fee_tax_mismatch(
    recon_items: list[dict[str, Any]],
    rate_card: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rule 3: Recon payment item where fee/tax differs from rate card by > rounding_tolerance_paise."""
    tolerance = rate_card.get("rounding_tolerance_paise", 2)
    gst_bps = rate_card.get("gst_bps", 1800)
    fee_includes_tax = rate_card.get("fee_includes_tax", True)

    detected: list[dict[str, Any]] = []
    for r in recon_items:
        if r.get("type") == "payment" and r.get("method") in ("card", "netbanking"):
            method = r["method"]
            rate_bps = rate_card.get(f"{method}_bps")
            if rate_bps is None:
                continue

            amount = r["amount"]
            base_fee = mul_div_round(amount, rate_bps, 10000)
            expected_tax = mul_div_round(base_fee, gst_bps, 10000)
            expected_fee = base_fee + expected_tax if fee_includes_tax else base_fee

            if abs(r["fee"] - expected_fee) > tolerance or abs(r["tax"] - expected_tax) > tolerance:
                detected.append({
                    "entity_id": r["entity_id"],
                    "merchant_id": r.get("merchant_id"),
                    "settlement_id": r.get("settlement_id"),
                    "observed_fee": r["fee"],
                    "expected_fee": expected_fee,
                    "observed_tax": r["tax"],
                    "expected_tax": expected_tax,
                    "entity_ids": {r["entity_id"]},
                })
    return detected


def detect_partial_refund_mismatch(
    recon_items: list[dict[str, Any]],
    orders_ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rule 4: Refund debit != ledger refund amount or ledger status == 'captured' with no refund entry."""
    orders_by_id = {o["order_id"]: o for o in orders_ledger}
    detected: list[dict[str, Any]] = []

    for r in recon_items:
        if r.get("type") == "refund":
            order = orders_by_id.get(r.get("order_id"))
            if not order:
                detected.append({
                    "entity_id": r["entity_id"],
                    "payment_id": r.get("payment_id"),
                    "order_id": r.get("order_id"),
                    "merchant_id": r.get("merchant_id"),
                    "entity_ids": {r["entity_id"], r.get("payment_id", "")},
                })
                continue

            order_status = order.get("status")
            has_refund_id = bool(order.get("refund_id"))
            order_refund_amt = int(order.get("refund_amount", 0))

            if order_status == "captured" or not has_refund_id or order_refund_amt != r["debit"]:
                detected.append({
                    "entity_id": r["entity_id"],
                    "payment_id": r.get("payment_id"),
                    "order_id": r.get("order_id"),
                    "merchant_id": r.get("merchant_id"),
                    "entity_ids": {r["entity_id"], r.get("payment_id", "")},
                })
    return detected


def detect_cutoff_drift(
    payments: list[dict[str, Any]],
    orders_ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rule 5: Ledger booked_date != IST date of payment capture."""
    orders_by_id = {o["order_id"]: o for o in orders_ledger}
    detected: list[dict[str, Any]] = []

    for p in payments:
        if p.get("status") == "captured":
            order = orders_by_id.get(p.get("order_id"))
            if not order:
                continue
            cap_ts = p.get("captured_at") or p.get("created_at")
            cap_date = datetime.datetime.fromtimestamp(cap_ts, tz=IST_TZ).date().isoformat()
            booked_date = order.get("booked_date")
            if booked_date and cap_date != booked_date:
                detected.append({
                    "payment_id": p["payment_id"],
                    "order_id": p.get("order_id"),
                    "merchant_id": p.get("merchant_id"),
                    "entity_ids": {p["payment_id"], p.get("order_id", "")},
                })
    return detected


def detect_short_bank_credit(
    settlements: list[dict[str, Any]],
    bank_statement: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rule 6: Bank credit < settlement net for UTR match."""
    bank_credits = [b for b in bank_statement if int(b.get("credit", 0)) > 0]
    detected: list[dict[str, Any]] = []

    for s in settlements:
        utr = s.get("utr", "")
        if not utr:
            continue
        m_id = s.get("merchant_id")
        for b in bank_credits:
            if b.get("merchant_id") == m_id:
                desc = b.get("description", "")
                if utr in desc or utr.lower() in desc.lower() or utr[:12] in desc:
                    b_credit = int(b.get("credit", 0))
                    s_amount = int(s.get("amount", 0))
                    if b_credit < s_amount:
                        detected.append({
                            "statement_id": b["statement_id"],
                            "settlement_id": s["settlement_id"],
                            "merchant_id": m_id,
                            "bank_credit": b_credit,
                            "settlement_amount": s_amount,
                            "short_amount": s_amount - b_credit,
                            "entity_ids": {b["statement_id"]},
                        })
                    break
    return detected


def detect_orphan_bank_credits(
    settlements: list[dict[str, Any]],
    bank_statement: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rule 7: Bank credit with Razorpay narration matching no settlement."""
    bank_credits = [b for b in bank_statement if int(b.get("credit", 0)) > 0]
    rzp_credits = [b for b in bank_credits if "razorpay" in b.get("description", "").lower()]

    matched_stmt_ids: set[str] = set()
    matched_settle_ids: set[str] = set()

    # Step 1: match with settlements by UTR
    for s in settlements:
        utr = s.get("utr", "")
        if not utr:
            continue
        m_id = s.get("merchant_id")
        for b in rzp_credits:
            if b["statement_id"] in matched_stmt_ids:
                continue
            if b.get("merchant_id") == m_id:
                desc = b.get("description", "")
                if utr in desc or utr.lower() in desc.lower() or utr[:12] in desc:
                    matched_stmt_ids.add(b["statement_id"])
                    matched_settle_ids.add(s["settlement_id"])
                    break

    # Step 2: for settlements with no UTR in narration, match remaining by (merchant, date, amount)
    for s in settlements:
        if s["settlement_id"] in matched_settle_ids:
            continue
        m_id = s.get("merchant_id")
        s_date = datetime.datetime.fromtimestamp(s["settled_at"], tz=IST_TZ).date().isoformat()
        s_amt = int(s.get("amount", 0))
        for b in rzp_credits:
            if b["statement_id"] in matched_stmt_ids:
                continue
            if b.get("merchant_id") == m_id and b.get("date") == s_date and int(b.get("credit", 0)) == s_amt:
                matched_stmt_ids.add(b["statement_id"])
                matched_settle_ids.add(s["settlement_id"])
                break

    detected: list[dict[str, Any]] = []
    for b in rzp_credits:
        if b["statement_id"] not in matched_stmt_ids:
            detected.append({
                "statement_id": b["statement_id"],
                "merchant_id": b.get("merchant_id"),
                "date": b.get("date"),
                "credit": int(b.get("credit", 0)),
                "description": b.get("description", ""),
                "entity_ids": {b["statement_id"]},
            })
    return detected


# =============================================================================
# Master Oracle Audit Engine
# =============================================================================


def audit_dataset(data: dict[str, Any]) -> dict[str, Any]:
    """Run all 7 oracle detection rules over loaded dataset dictionaries."""
    payments = data.get("payments", [])
    recon_items = data.get("recon_items", [])
    orders_ledger = data.get("orders_ledger", [])
    settlements = data.get("settlements", [])
    bank_statement = data.get("bank_statement", [])
    rate_card = data.get("rate_card", {})
    manifest = data.get("manifest", {})
    labels = data.get("labels", [])

    detected_cases: dict[str, list[dict[str, Any]]] = {
        "DUPLICATE_PAYMENT": detect_duplicate_payments(payments),
        "MISSING_SETTLEMENT": detect_missing_settlements(payments, recon_items, rate_card, manifest),
        "FEE_TAX_MISMATCH": detect_fee_tax_mismatch(recon_items, rate_card),
        "PARTIAL_REFUND_MISMATCH": detect_partial_refund_mismatch(recon_items, orders_ledger),
        "CUTOFF_DRIFT": detect_cutoff_drift(payments, orders_ledger),
        "SHORT_BANK_CREDIT": detect_short_bank_credit(settlements, bank_statement),
        "ORPHAN_BANK_CREDIT": detect_orphan_bank_credits(settlements, bank_statement),
    }

    evaluation: dict[str, dict[str, Any]] = {}
    if labels:
        # Group injected labels by error_class
        labels_by_class: dict[str, list[dict[str, Any]]] = {}
        for lbl in labels:
            err = lbl.get("error_class")
            if err:
                labels_by_class.setdefault(err, []).append(lbl)

        for err_cls in REQUIRED_ERROR_CLASSES:
            class_labels = labels_by_class.get(err_cls, [])
            injected_count = len(class_labels)
            class_detections = detected_cases.get(err_cls, [])

            # Collect all labeled entities for this class
            all_labeled_entities: set[str] = {
                eid for lbl in class_labels for eid in lbl.get("entity_ids", [])
            }

            # Collect all detected entities for this class
            all_detected_entities: set[str] = {
                eid for det in class_detections for eid in det.get("entity_ids", set())
            }

            # Count how many injected labels were detected
            detected_label_cases = sum(
                1 for lbl in class_labels
                if set(lbl.get("entity_ids", [])) & all_detected_entities
            )

            # Count false positives (detections not matching any injected label of this class)
            false_positives = sum(
                1 for det in class_detections
                if not (det.get("entity_ids", set()) & all_labeled_entities)
            )

            rate = (detected_label_cases / injected_count * 100.0) if injected_count > 0 else 100.0

            evaluation[err_cls] = {
                "injected": injected_count,
                "detected": detected_label_cases,
                "detection_rate": round(rate, 2),
                "false_positives": false_positives,
            }

    summary_table = format_oracle_table(evaluation)
    return {
        "detected_cases": detected_cases,
        "evaluation": evaluation,
        "summary_table": summary_table,
    }


def format_oracle_table(evaluation: dict[str, dict[str, Any]]) -> str:
    """Format oracle audit evaluation into an ASCII / Markdown table."""
    lines = [
        "| Error Class | Injected | Detected | Detection Rate (%) | False Positives | Status |",
        "|:---|---:|---:|---:|---:|:---:|",
    ]
    for err_cls in REQUIRED_ERROR_CLASSES:
        if err_cls in evaluation:
            res = evaluation[err_cls]
            inj = res["injected"]
            det = res["detected"]
            rate = res["detection_rate"]
            fp = res["false_positives"]
            status = "PASS" if rate >= 98.0 and fp == 0 else "FAIL"
            lines.append(f"| {err_cls:<23} | {inj:>8} | {det:>8} | {rate:>17.2f}% | {fp:>15} | {status:^6} |")
        else:
            lines.append(f"| {err_cls:<23} |      N/A |      N/A |               N/A |             N/A |   N/A  |")
    return "\n".join(lines)


def run_oracle(dir_path: str) -> dict[str, Any]:
    """Execute complete oracle audit on a directory containing reconciliation files."""
    data = load_dataset_from_directory(dir_path)
    return audit_dataset(data)
