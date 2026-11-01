"""Observable data loader for reconciliation pipeline.

Loads strictly observable reconciliation files (orders_ledger, payments, recon_items,
settlements, bank_statement, rate_card, and manifest.as_of) into frozen dataclasses.
Never opens, reads, or touches labels, truth links, or injection manifests.
"""

import csv
import json
import os
from typing import Any

from src.recon_agent.pipeline.types import (
    BankStatementRow,
    OrderLedgerRow,
    PaymentRow,
    RateCard,
    ReconDataset,
    ReconItemRow,
    SettlementRow,
)


def _ensure_int(val: Any, field_name: str) -> int:
    """Validate and convert value to int, strictly forbidding floats."""
    if isinstance(val, float):
        raise ValueError(f"Floating-point value forbidden for {field_name}: {val}")
    if isinstance(val, str):
        if "." in val:
            raise ValueError(f"Floating-point string forbidden for {field_name}: {val}")
        return int(val)
    if isinstance(val, int):
        return val
    raise TypeError(f"Expected int or int-compatible str for {field_name}, got {type(val)}")


def _optional_int(val: Any, field_name: str) -> int | None:
    """Validate and convert optional value to int or None, forbidding floats."""
    if val is None or val == "":
        return None
    return _ensure_int(val, field_name)


def load_dataset(dir_path: str) -> ReconDataset:
    """Load observable reconciliation files from a dataset directory into a ReconDataset."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Dataset directory not found: {dir_path}")

    # 1. orders_ledger.csv
    orders_ledger_path = os.path.join(dir_path, "orders_ledger.csv")
    if not os.path.isfile(orders_ledger_path):
        raise FileNotFoundError(f"Missing required observable file: {orders_ledger_path}")
    orders_ledger_rows: list[OrderLedgerRow] = []
    with open(orders_ledger_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            orders_ledger_rows.append(
                OrderLedgerRow(
                    order_id=r["order_id"],
                    merchant_id=r["merchant_id"],
                    booked_date=r["booked_date"],
                    created_at=_ensure_int(r["created_at"], "orders_ledger.created_at"),
                    expected_amount=_ensure_int(r["expected_amount"], "orders_ledger.expected_amount"),
                    currency=r["currency"],
                    order_receipt=r["order_receipt"],
                    status=r["status"],
                    payment_id=r["payment_id"] if r["payment_id"] else None,
                    refund_id=r["refund_id"] if r["refund_id"] else None,
                    refund_amount=_optional_int(r["refund_amount"], "orders_ledger.refund_amount"),
                )
            )

    # 2. payments.jsonl
    payments_path = os.path.join(dir_path, "payments.jsonl")
    if not os.path.isfile(payments_path):
        raise FileNotFoundError(f"Missing required observable file: {payments_path}")
    payment_rows: list[PaymentRow] = []
    with open(payments_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            p = json.loads(line_str)
            payment_rows.append(
                PaymentRow(
                    payment_id=p["payment_id"],
                    order_id=p["order_id"],
                    merchant_id=p["merchant_id"],
                    amount=_ensure_int(p["amount"], "payments.amount"),
                    currency=p["currency"],
                    status=p["status"],
                    method=p["method"],
                    created_at=_ensure_int(p["created_at"], "payments.created_at"),
                    captured_at=_optional_int(p.get("captured_at"), "payments.captured_at"),
                    error_code=p.get("error_code"),
                    error_description=p.get("error_description"),
                )
            )

    # 3. recon_items.jsonl
    recon_items_path = os.path.join(dir_path, "recon_items.jsonl")
    if not os.path.isfile(recon_items_path):
        raise FileNotFoundError(f"Missing required observable file: {recon_items_path}")
    recon_item_rows: list[ReconItemRow] = []
    with open(recon_items_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            r = json.loads(line_str)
            recon_item_rows.append(
                ReconItemRow(
                    entity_id=r["entity_id"],
                    type=r["type"],
                    debit=_ensure_int(r["debit"], "recon_items.debit"),
                    credit=_ensure_int(r["credit"], "recon_items.credit"),
                    amount=_ensure_int(r["amount"], "recon_items.amount"),
                    currency=r["currency"],
                    fee=_ensure_int(r["fee"], "recon_items.fee"),
                    tax=_ensure_int(r["tax"], "recon_items.tax"),
                    on_hold=bool(r["on_hold"]),
                    settled=bool(r["settled"]),
                    created_at=_ensure_int(r["created_at"], "recon_items.created_at"),
                    settled_at=_optional_int(r.get("settled_at"), "recon_items.settled_at"),
                    settlement_id=r.get("settlement_id"),
                    description=r.get("description"),
                    notes=r.get("notes"),
                    payment_id=r.get("payment_id"),
                    settlement_utr=r.get("settlement_utr"),
                    order_id=r.get("order_id"),
                    order_receipt=r.get("order_receipt"),
                    method=r.get("method"),
                    card_network=r.get("card_network"),
                    card_issuer=r.get("card_issuer"),
                    card_type=r.get("card_type"),
                    dispute_id=r.get("dispute_id"),
                    merchant_id=r["merchant_id"],
                )
            )

    # 4. settlements.jsonl
    settlements_path = os.path.join(dir_path, "settlements.jsonl")
    if not os.path.isfile(settlements_path):
        raise FileNotFoundError(f"Missing required observable file: {settlements_path}")
    settlement_rows: list[SettlementRow] = []
    with open(settlements_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            s = json.loads(line_str)
            settlement_rows.append(
                SettlementRow(
                    settlement_id=s["settlement_id"],
                    merchant_id=s["merchant_id"],
                    amount=_ensure_int(s["amount"], "settlements.amount"),
                    utr=s["utr"],
                    status=s["status"],
                    created_at=_ensure_int(s["created_at"], "settlements.created_at"),
                    settled_at=_ensure_int(s["settled_at"], "settlements.settled_at"),
                )
            )

    # 5. bank_statement.csv
    bank_path = os.path.join(dir_path, "bank_statement.csv")
    if not os.path.isfile(bank_path):
        raise FileNotFoundError(f"Missing required observable file: {bank_path}")
    bank_statement_rows: list[BankStatementRow] = []
    with open(bank_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for b in reader:
            bank_statement_rows.append(
                BankStatementRow(
                    statement_id=b["statement_id"],
                    merchant_id=b["merchant_id"],
                    date=b["date"],
                    description=b["description"],
                    debit=_ensure_int(b["debit"], "bank_statement.debit"),
                    credit=_ensure_int(b["credit"], "bank_statement.credit"),
                    balance=_ensure_int(b["balance"], "bank_statement.balance"),
                )
            )

    # 6. rate_card.json
    rate_card_path = os.path.join(dir_path, "rate_card.json")
    if not os.path.isfile(rate_card_path):
        raise FileNotFoundError(f"Missing required observable file: {rate_card_path}")
    with open(rate_card_path, "r", encoding="utf-8") as f:
        rc = json.load(f)
    rate_card = RateCard(
        card_bps=_ensure_int(rc["card_bps"], "rate_card.card_bps"),
        upi_bps=_ensure_int(rc["upi_bps"], "rate_card.upi_bps"),
        netbanking_bps=_ensure_int(rc["netbanking_bps"], "rate_card.netbanking_bps"),
        gst_bps=_ensure_int(rc["gst_bps"], "rate_card.gst_bps"),
        fee_includes_tax=bool(rc["fee_includes_tax"]),
        payment_settlement_lag_days=_ensure_int(rc["payment_settlement_lag_days"], "rate_card.payment_settlement_lag_days"),
        refund_settlement_lag_days=_ensure_int(rc["refund_settlement_lag_days"], "rate_card.refund_settlement_lag_days"),
        missing_after_days=_ensure_int(rc["missing_after_days"], "rate_card.missing_after_days"),
        rounding_tolerance_paise=_ensure_int(rc["rounding_tolerance_paise"], "rate_card.rounding_tolerance_paise"),
        escalate_tolerance_paise=_ensure_int(rc["escalate_tolerance_paise"], "rate_card.escalate_tolerance_paise"),
        error_threshold_paise=_ensure_int(rc["error_threshold_paise"], "rate_card.error_threshold_paise"),
    )

    # 7. manifest.json (strictly for as_of)
    manifest_path = os.path.join(dir_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing required observable file: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    as_of = str(m["as_of"])

    return ReconDataset(
        orders_ledger=tuple(orders_ledger_rows),
        payments=tuple(payment_rows),
        recon_items=tuple(recon_item_rows),
        settlements=tuple(settlement_rows),
        bank_statement=tuple(bank_statement_rows),
        rate_card=rate_card,
        as_of=as_of,
    )

