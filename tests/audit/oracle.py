"""Deterministic oracle audit for synthetic reconciliation datasets.

Pure observable file audit (imports csv, datetime, json, os, sys only; never imports recon_agent.synth).
Evaluates Tier 1 (7 error classes) and Tier 2 (4 ambiguity types) directly from observable files.
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

REQUIRED_AMBIGUITY_TYPES = [
    "NEAR_TOLERANCE",
    "EXPLAINED_CONSISTENT",
    "EXPLAINED_INCONSISTENT",
    "EXPLAINED_NO_AMOUNT",
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
    data["manifest"] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data["manifest"] = json.load(f)

    # 2. rate_card.json
    rate_card_path = os.path.join(dir_path, "rate_card.json")
    data["rate_card"] = {}
    if os.path.exists(rate_card_path):
        with open(rate_card_path, "r", encoding="utf-8") as f:
            data["rate_card"] = json.load(f)

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
    """Rule 3: Recon payment item where fee/tax error >= error_threshold_paise (10 paise; Amendment 1)."""
    error_threshold = rate_card.get("error_threshold_paise", 10)
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

            delta_fee = abs(r["fee"] - expected_fee)
            delta_tax = abs(r["tax"] - expected_tax)

            if delta_fee >= error_threshold or delta_tax >= error_threshold:
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


def detect_near_tolerance(
    recon_items: list[dict[str, Any]],
    rate_card: dict[str, Any],
) -> list[dict[str, Any]]:
    """Tier 2 Rule: Card/netbanking payment items with fee/tax variance in 3-9 paise policy gap."""
    round_tol = rate_card.get("rounding_tolerance_paise", 2)
    esc_tol = rate_card.get("escalate_tolerance_paise", 9)
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

            max_delta = max(abs(r["fee"] - expected_fee), abs(r["tax"] - expected_tax))
            if round_tol < max_delta <= esc_tol:
                detected.append({
                    "entity_id": r["entity_id"],
                    "merchant_id": r.get("merchant_id"),
                    "settlement_id": r.get("settlement_id"),
                    "max_delta": max_delta,
                    "entity_ids": {r["entity_id"]},
                    "suggested_action": "escalate_human",
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


def match_settlements_and_bank(
    settlements: list[dict[str, Any]],
    bank_statement: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match settlements and bank credits conforming strictly to Amendment 2.

    Filter to Razorpay-style narrations. Then:
      - Rows with matching UTR are matched directly.
      - On remaining (merchant, date):
        * A row whose amount equals settlement net is the match.
        * Any additional Razorpay row on that merchant-date is an orphan.
        * A lone row with a different amount is a short credit.
        * Never match on 'amount < net' alone.
    """
    bank_credits = [b for b in bank_statement if int(b.get("credit", 0)) > 0]
    rzp_credits = [b for b in bank_credits if "razorpay" in b.get("description", "").lower()]

    matched_stmt_ids: set[str] = set()
    matched_settle_ids: set[str] = set()
    short_bank_credits: list[dict[str, Any]] = []

    # Step 1: Match by UTR
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
                    b_credit = int(b.get("credit", 0))
                    s_amount = int(s.get("amount", 0))
                    if b_credit < s_amount:
                        short_bank_credits.append({
                            "statement_id": b["statement_id"],
                            "settlement_id": s["settlement_id"],
                            "merchant_id": m_id,
                            "bank_credit": b_credit,
                            "settlement_amount": s_amount,
                            "short_amount": s_amount - b_credit,
                            "entity_ids": {b["statement_id"]},
                        })
                    break

    # Step 2: Unmatched settlements and bank rows (no_utr cohorts) grouped by (merchant, date)
    unmatched_stls_by_m_d: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in settlements:
        if s["settlement_id"] not in matched_settle_ids:
            s_date = datetime.datetime.fromtimestamp(s["settled_at"], tz=IST_TZ).date().isoformat()
            unmatched_stls_by_m_d.setdefault((s["merchant_id"], s_date), []).append(s)

    unmatched_rzp_by_m_d: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for b in rzp_credits:
        if b["statement_id"] not in matched_stmt_ids:
            unmatched_rzp_by_m_d.setdefault((b["merchant_id"], b.get("date", "")), []).append(b)

    for (m_id, d_str), brows in unmatched_rzp_by_m_d.items():
        stls = unmatched_stls_by_m_d.get((m_id, d_str), [])
        if not stls:
            continue

        # A row whose amount equals settlement net is the match
        for s in list(stls):
            matched_b = None
            for b in brows:
                if b["statement_id"] not in matched_stmt_ids and int(b.get("credit", 0)) == int(s.get("amount", 0)):
                    matched_b = b
                    break
            if matched_b:
                matched_stmt_ids.add(matched_b["statement_id"])
                matched_settle_ids.add(s["settlement_id"])
                stls.remove(s)
                brows.remove(matched_b)

        # A lone row with a different amount is a short credit
        if len(stls) == 1 and len(brows) == 1:
            s = stls[0]
            b = brows[0]
            b_credit = int(b.get("credit", 0))
            s_amount = int(s.get("amount", 0))
            if b_credit < s_amount:
                matched_stmt_ids.add(b["statement_id"])
                matched_settle_ids.add(s["settlement_id"])
                short_bank_credits.append({
                    "statement_id": b["statement_id"],
                    "settlement_id": s["settlement_id"],
                    "merchant_id": m_id,
                    "bank_credit": b_credit,
                    "settlement_amount": s_amount,
                    "short_amount": s_amount - b_credit,
                    "entity_ids": {b["statement_id"]},
                })
                stls.remove(s)
                brows.remove(b)

    # Any additional Razorpay row on that merchant-date is an orphan
    orphan_bank_credits: list[dict[str, Any]] = []
    for b in rzp_credits:
        if b["statement_id"] not in matched_stmt_ids:
            orphan_bank_credits.append({
                "statement_id": b["statement_id"],
                "merchant_id": b.get("merchant_id"),
                "date": b.get("date"),
                "credit": int(b.get("credit", 0)),
                "description": b.get("description", ""),
                "entity_ids": {b["statement_id"]},
            })

    return short_bank_credits, orphan_bank_credits


# =============================================================================
# Master Oracle Audit Engine
# =============================================================================


def audit_dataset(data: dict[str, Any]) -> dict[str, Any]:
    """Run all Tier 1 and Tier 2 detection rules over loaded dataset dictionaries."""
    payments = data.get("payments", [])
    recon_items = data.get("recon_items", [])
    orders_ledger = data.get("orders_ledger", [])
    settlements = data.get("settlements", [])
    bank_statement = data.get("bank_statement", [])
    rate_card = data.get("rate_card", {})
    manifest = data.get("manifest", {})
    labels = data.get("labels", [])

    short_credits, orphan_credits = match_settlements_and_bank(settlements, bank_statement)

    detected_cases: dict[str, list[dict[str, Any]]] = {
        "DUPLICATE_PAYMENT": detect_duplicate_payments(payments),
        "MISSING_SETTLEMENT": detect_missing_settlements(payments, recon_items, rate_card, manifest),
        "FEE_TAX_MISMATCH": detect_fee_tax_mismatch(recon_items, rate_card),
        "PARTIAL_REFUND_MISMATCH": detect_partial_refund_mismatch(recon_items, orders_ledger),
        "CUTOFF_DRIFT": detect_cutoff_drift(payments, orders_ledger),
        "SHORT_BANK_CREDIT": short_credits,
        "ORPHAN_BANK_CREDIT": orphan_credits,
        "NEAR_TOLERANCE": detect_near_tolerance(recon_items, rate_card),
    }

    tier1_evaluation: dict[str, dict[str, Any]] = {}
    tier2_evaluation: dict[str, dict[str, Any]] = {}

    if labels:
        # Tier 1 Evaluation (Injected Error Classes)
        labels_by_err: dict[str, list[dict[str, Any]]] = {}
        labels_by_amb: dict[str, list[dict[str, Any]]] = {}
        all_labeled_entities: set[str] = {
            eid for lbl in labels for eid in lbl.get("entity_ids", [])
        }

        for lbl in labels:
            err = lbl.get("error_class")
            if err:
                labels_by_err.setdefault(err, []).append(lbl)
            amb = lbl.get("ambiguity_type")
            if amb:
                labels_by_amb.setdefault(amb, []).append(lbl)

        for err_cls in REQUIRED_ERROR_CLASSES:
            class_labels = labels_by_err.get(err_cls, [])
            injected_count = len(class_labels)
            class_detections = detected_cases.get(err_cls, [])

            all_detected_entities: set[str] = {
                eid for det in class_detections for eid in det.get("entity_ids", set())
            }

            detected_label_cases = sum(
                1 for lbl in class_labels
                if set(lbl.get("entity_ids", [])) & all_detected_entities
            )
            # True false positives: detections on clean data (entities with no label in any tier)
            false_positives = sum(
                1 for det in class_detections
                if not (det.get("entity_ids", set()) & all_labeled_entities)
            )

            rate = (detected_label_cases / injected_count * 100.0) if injected_count > 0 else 100.0
            tier1_evaluation[err_cls] = {
                "injected": injected_count,
                "detected": detected_label_cases,
                "detection_rate": round(rate, 2),
                "false_positives": false_positives,
            }

        # Tier 2 Evaluation (Ambiguity Types: Informational Headroom Assessment)
        # Note: Oracle may NOT parse text.
        refund_mismatch_detected_entities = {
            eid for det in detected_cases["PARTIAL_REFUND_MISMATCH"] for eid in det.get("entity_ids", set())
        }
        near_tol_detected_entities = {
            eid for det in detected_cases["NEAR_TOLERANCE"] for eid in det.get("entity_ids", set())
        }

        for amb_type in REQUIRED_AMBIGUITY_TYPES:
            amb_labels = labels_by_amb.get(amb_type, [])
            count = len(amb_labels)

            if amb_type == "NEAR_TOLERANCE":
                # Detected numerically by 3-band rule -> escalate_human matches ground truth
                correct = sum(
                    1 for lbl in amb_labels
                    if set(lbl.get("entity_ids", [])) & near_tol_detected_entities
                )
            elif amb_type == "EXPLAINED_CONSISTENT":
                # Ground truth is benign_explained. Oracle flags as refund mismatch without text parsing -> 0% accuracy
                correct = 0
            elif amb_type == "EXPLAINED_INCONSISTENT":
                # Ground truth is flag_partial_refund_mismatch. Oracle flags as refund mismatch -> matches!
                correct = sum(
                    1 for lbl in amb_labels
                    if set(lbl.get("entity_ids", [])) & refund_mismatch_detected_entities
                )
            elif amb_type == "EXPLAINED_NO_AMOUNT":
                # Ground truth is escalate_human. Oracle flags as refund mismatch -> 0% accuracy
                correct = 0
            else:
                correct = 0

            acc = (correct / count * 100.0) if count > 0 else 100.0
            tier2_evaluation[amb_type] = {
                "count": count,
                "oracle_correct": correct,
                "oracle_accuracy": round(acc, 2),
                "oracle_behavior": "flag_mismatch (text unparsed)" if "EXPLAINED" in amb_type else "3-band escalate",
            }

    summary_table = format_oracle_table(tier1_evaluation, tier2_evaluation)
    return {
        "detected_cases": detected_cases,
        "evaluation": tier1_evaluation,
        "tier2_evaluation": tier2_evaluation,
        "summary_table": summary_table,
    }


def format_oracle_table(
    tier1_eval: dict[str, dict[str, Any]],
    tier2_eval: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Format oracle audit evaluation into structured Markdown tables."""
    lines = [
        "### Tier 1: Deterministic Error Classes (Pass/Fail Floors)",
        "| Error Class | Injected | Detected | Detection Rate (%) | False Positives | Status |",
        "|:---|---:|---:|---:|---:|:---:|",
    ]
    for err_cls in REQUIRED_ERROR_CLASSES:
        if err_cls in tier1_eval:
            res = tier1_eval[err_cls]
            inj = res["injected"]
            det = res["detected"]
            rate = res["detection_rate"]
            fp = res["false_positives"]
            status = "PASS" if rate >= 98.0 and fp == 0 else "FAIL"
            lines.append(f"| {err_cls:<23} | {inj:>8} | {det:>8} | {rate:>17.2f}% | {fp:>15} | {status:^6} |")
        else:
            lines.append(f"| {err_cls:<23} |      N/A |      N/A |               N/A |             N/A |   N/A  |")

    if tier2_eval:
        lines.extend([
            "",
            "### Tier 2: Ambiguity Tier (Informational Accuracy / Headroom)",
            "| Ambiguity Type | Injected | Oracle Correct | Oracle Accuracy (%) | Oracle Text Blindness Effect |",
            "|:---|---:|---:|---:|:---|",
        ])
        for amb_type in REQUIRED_AMBIGUITY_TYPES:
            if amb_type in tier2_eval:
                res = tier2_eval[amb_type]
                cnt = res["count"]
                cor = res["oracle_correct"]
                acc = res["oracle_accuracy"]
                beh = res["oracle_behavior"]
                lines.append(f"| {amb_type:<23} | {cnt:>8} | {cor:>14} | {acc:>18.2f}% | {beh:<28} |")
            else:
                lines.append(f"| {amb_type:<23} |      N/A |            N/A |                N/A | N/A                          |")

    return "\n".join(lines)


def run_oracle(dir_path: str) -> dict[str, Any]:
    """Execute complete oracle audit on a directory containing reconciliation files."""
    data = load_dataset_from_directory(dir_path)
    return audit_dataset(data)
