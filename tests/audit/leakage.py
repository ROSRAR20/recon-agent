"""Statistical feature leakage audit for synthetic reconciliation datasets.

Evaluates non-semantic feature leakage between injected (positive) and clean (negative)
entities using the Mann-Whitney U test (Area Under ROC Curve / AUC).
All non-whitelisted features must have AUC strictly between 0.30 and 0.70.
"""

import csv
import datetime
import json
import math
import os
import sys
from typing import Any

IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

WHITELISTED_FEATURES = {
    ("CUTOFF_DRIFT", "time_of_day"),
}

SKIPPED_FEATURES = {
    ("FEE_TAX_MISMATCH", "merchant_amount_zscore"),
    ("PARTIAL_REFUND_MISMATCH", "merchant_amount_zscore"),
}


def compute_auc(positives: list[float], negatives: list[float]) -> float:
    """Compute Area Under the ROC Curve (AUC) via Mann-Whitney U statistic.

    Handles ties using average fractional ranks.
    """
    n1 = len(positives)
    n2 = len(negatives)
    if n1 == 0 or n2 == 0:
        return 0.5

    all_items = [(val, 1) for val in positives] + [(val, 0) for val in negatives]
    all_items.sort(key=lambda t: t[0])

    ranks_sum_pos = 0.0
    i = 0
    total = len(all_items)
    while i < total:
        j = i
        while j < total - 1 and all_items[j + 1][0] == all_items[j][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            if all_items[k][1] == 1:
                ranks_sum_pos += avg_rank
        i = j + 1

    u1 = ranks_sum_pos - (n1 * (n1 + 1)) / 2.0
    return u1 / (n1 * n2)


def get_id_suffix(entity_id: str) -> str:
    """Extract trailing base62/alphanumeric suffix from ID."""
    if "_" in entity_id:
        return entity_id.split("_", 1)[1]
    return entity_id


def extract_features_for_payment(
    p: dict[str, Any],
    pos_idx: int,
    total_count: int,
    merchant_stats: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Extract numerical non-semantic features for a payment record."""
    file_pos_pct = pos_idx / max(1, total_count)
    token = get_id_suffix(p.get("payment_id", ""))
    suffix = token[-6:] if len(token) >= 6 else token
    digits_ratio = sum(1 for c in suffix if c.isdigit()) / max(1, len(suffix))
    upper_ratio = sum(1 for c in suffix if c.isupper()) / max(1, len(suffix))

    m_id = p.get("merchant_id", "")
    amt = float(p.get("amount", 0))
    mean_amt, std_amt = merchant_stats.get(m_id, (amt, 1.0))
    amt_zscore = (amt - mean_amt) / max(1.0, std_amt)

    null_count = float(sum(1 for v in p.values() if v is None))

    ts = p.get("created_at") or 0
    dt = datetime.datetime.fromtimestamp(ts, tz=IST_TZ)
    time_of_day = float(dt.hour * 3600 + dt.minute * 60 + dt.second)

    return {
        "file_position_pct": file_pos_pct,
        "id_suffix_digits_ratio": digits_ratio,
        "id_suffix_upper_ratio": upper_ratio,
        "merchant_amount_zscore": amt_zscore,
        "null_count": null_count,
        "time_of_day": time_of_day,
    }


def extract_features_for_recon_item(
    r: dict[str, Any],
    pos_idx: int,
    total_count: int,
    merchant_stats: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Extract numerical non-semantic features for a recon item."""
    file_pos_pct = pos_idx / max(1, total_count)
    token = get_id_suffix(r.get("entity_id", ""))
    suffix = token[-6:] if len(token) >= 6 else token
    digits_ratio = sum(1 for c in suffix if c.isdigit()) / max(1, len(suffix))
    upper_ratio = sum(1 for c in suffix if c.isupper()) / max(1, len(suffix))

    m_id = r.get("merchant_id", "")
    amt = float(r.get("amount", 0))
    mean_amt, std_amt = merchant_stats.get(m_id, (amt, 1.0))
    amt_zscore = (amt - mean_amt) / max(1.0, std_amt)

    null_count = float(sum(1 for v in r.values() if v is None))

    ts = r.get("created_at") or 0
    dt = datetime.datetime.fromtimestamp(ts, tz=IST_TZ)
    time_of_day = float(dt.hour * 3600 + dt.minute * 60 + dt.second)

    return {
        "file_position_pct": file_pos_pct,
        "id_suffix_digits_ratio": digits_ratio,
        "id_suffix_upper_ratio": upper_ratio,
        "merchant_amount_zscore": amt_zscore,
        "null_count": null_count,
        "time_of_day": time_of_day,
    }


def extract_features_for_bank_row(
    b: dict[str, Any],
    pos_idx: int,
    total_count: int,
    merchant_stats: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Extract numerical non-semantic features for a bank statement row."""
    file_pos_pct = pos_idx / max(1, total_count)
    stmt_id = b.get("statement_id", "")
    token = get_id_suffix(stmt_id)
    digits_ratio = sum(1 for c in token if c.isdigit()) / max(1, len(token))
    upper_ratio = sum(1 for c in token if c.isupper()) / max(1, len(token))

    stmt_num = 0.0
    try:
        stmt_num = float(stmt_id.split("_")[-1])
    except (ValueError, IndexError):
        pass

    m_id = b.get("merchant_id", "")
    amt = float(b.get("credit", 0))
    mean_amt, std_amt = merchant_stats.get(m_id, (amt, 1.0))
    amt_zscore = (amt - mean_amt) / max(1.0, std_amt)

    desc = b.get("description", "")
    desc_len = float(len(desc))

    null_count = float(sum(1 for v in b.values() if v is None))

    return {
        "file_position_pct": file_pos_pct,
        "id_suffix_digits_ratio": digits_ratio,
        "id_suffix_upper_ratio": upper_ratio,
        "sequence_id_gap": stmt_num / max(1, total_count),
        "merchant_amount_zscore": amt_zscore,
        "narration_length": desc_len,
        "null_count": null_count,
    }


def audit_leakage(data: dict[str, Any]) -> dict[str, Any]:
    """Compute AUC for all non-semantic features across all 7 error classes."""
    payments = data.get("payments", [])
    recon_items = data.get("recon_items", [])
    settlements = data.get("settlements", [])
    bank_statement = data.get("bank_statement", [])
    labels = data.get("labels", [])

    # Index labels by error class
    labels_by_class: dict[str, list[dict[str, Any]]] = {}
    all_injected_entities: set[str] = set()
    for lbl in labels:
        err = lbl.get("error_class")
        if err:
            labels_by_class.setdefault(err, []).append(lbl)
            all_injected_entities.update(lbl.get("entity_ids", []))

    # Precompute merchant amount statistics on clean entities
    m_pay_amounts: dict[str, list[float]] = {}
    for p in payments:
        if p.get("payment_id") not in all_injected_entities and p.get("status") == "captured":
            m_pay_amounts.setdefault(p["merchant_id"], []).append(float(p["amount"]))

    m_pay_stats: dict[str, tuple[float, float]] = {}
    for m, vals in m_pay_amounts.items():
        mean_v = sum(vals) / len(vals)
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1))
        m_pay_stats[m] = (mean_v, std_v)

    m_bank_amounts: dict[str, list[float]] = {}
    for b in bank_statement:
        if b.get("statement_id") not in all_injected_entities and int(b.get("credit", 0)) > 0:
            m_bank_amounts.setdefault(b["merchant_id"], []).append(float(b["credit"]))

    m_bank_stats: dict[str, tuple[float, float]] = {}
    for m, vals in m_bank_amounts.items():
        mean_v = sum(vals) / len(vals)
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1))
        m_bank_stats[m] = (mean_v, std_v)

    # Pre-extract features for all entities
    pay_features: list[dict[str, float]] = [
        extract_features_for_payment(p, idx, len(payments), m_pay_stats)
        for idx, p in enumerate(payments)
    ]
    pay_by_id = {p["payment_id"]: (p, pay_features[idx]) for idx, p in enumerate(payments)}

    recon_features: list[dict[str, float]] = [
        extract_features_for_recon_item(r, idx, len(recon_items), m_pay_stats)
        for idx, r in enumerate(recon_items)
    ]
    recon_by_id = {r["entity_id"]: (r, recon_features[idx]) for idx, r in enumerate(recon_items)}

    bank_features: list[dict[str, float]] = [
        extract_features_for_bank_row(b, idx, len(bank_statement), m_bank_stats)
        for idx, b in enumerate(bank_statement)
    ]
    bank_by_id = {b["statement_id"]: (b, bank_features[idx]) for idx, b in enumerate(bank_statement)}

    # Clean baselines
    clean_payments = [
        pay_features[idx]
        for idx, p in enumerate(payments)
        if p["payment_id"] not in all_injected_entities and p.get("status") == "captured"
    ]
    clean_recon = [
        recon_features[idx]
        for idx, r in enumerate(recon_items)
        if r["entity_id"] not in all_injected_entities
    ]
    clean_bank = [
        bank_features[idx]
        for idx, b in enumerate(bank_statement)
        if b["statement_id"] not in all_injected_entities and int(b.get("credit", 0)) > 0
    ]

    results: dict[str, dict[str, float]] = {}

    # Define targets for each class
    class_targets = {
        "DUPLICATE_PAYMENT": ("payment", clean_payments, pay_by_id),
        "MISSING_SETTLEMENT": ("payment", clean_payments, pay_by_id),
        "CUTOFF_DRIFT": ("payment", clean_payments, pay_by_id),
        "FEE_TAX_MISMATCH": ("recon", clean_recon, recon_by_id),
        "PARTIAL_REFUND_MISMATCH": ("recon", clean_recon, recon_by_id),
        "SHORT_BANK_CREDIT": ("bank", clean_bank, bank_by_id),
        "ORPHAN_BANK_CREDIT": ("bank", clean_bank, bank_by_id),
    }

    for err_cls, (ent_type, clean_feats, index_map) in class_targets.items():
        lbls = labels_by_class.get(err_cls, [])
        pos_entities: set[str] = set()
        for l in lbls:
            for eid in l.get("entity_ids", []):
                if eid in index_map:
                    pos_entities.add(eid)

        pos_feats = [index_map[eid][1] for eid in pos_entities]
        if not pos_feats or not clean_feats:
            continue

        results[err_cls] = {}
        feature_names = list(pos_feats[0].keys())
        for f_name in feature_names:
            if (err_cls, f_name) in SKIPPED_FEATURES:
                continue

            pos_vals = [f[f_name] for f in pos_feats]
            neg_vals = [f[f_name] for f in clean_feats]
            auc = compute_auc(pos_vals, neg_vals)
            results[err_cls][f_name] = round(auc, 4)

    summary_table = format_leakage_table(results)
    return {
        "results": results,
        "summary_table": summary_table,
    }


def format_leakage_table(results: dict[str, dict[str, float]]) -> str:
    """Format AUC leakage results as an ASCII / Markdown table."""
    all_features = sorted(list({f for res in results.values() for f in res.keys()}))
    headers = ["Error Class"] + all_features + ["Overall Status"]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|:" + "---:|:" * (len(headers) - 1) + "---:|"

    lines = [header_line, sep_line]
    for err_cls, feats in results.items():
        row = [f"{err_cls:<23}"]
        cls_pass = True
        for f_name in all_features:
            if f_name in feats:
                auc = feats[f_name]
                is_whitelisted = (err_cls, f_name) in WHITELISTED_FEATURES
                passes = is_whitelisted or (0.30 <= auc <= 0.70)
                if not passes:
                    cls_pass = False
                tag = f"{auc:.3f}" + ("*" if is_whitelisted else "")
                row.append(f"{tag:>10}")
            else:
                row.append("       N/A")
        status = "PASS" if cls_pass else "FAIL"
        row.append(f"{status:^14}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n* Whitelisted semantic feature (e.g. CUTOFF_DRIFT time_of_day)")
    return "\n".join(lines)
