"""Statistical feature leakage audit for synthetic reconciliation datasets.

Evaluates non-semantic feature leakage between injected (positive) and clean (negative)
entities from the SAME ELIGIBLE POPULATION using the Mann-Whitney U test (Area Under ROC Curve / AUC).
All non-whitelisted features must have AUC strictly between 0.40 and 0.60.
Whitelist stays strictly CUTOFF_DRIFT time_of_day only.
"""

import csv
import datetime
import json
import math
import os
import sys
from typing import Any

IST_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def add_working_days(start_date: datetime.date, n: int) -> datetime.date:
    """Add n Mon-Fri working days to start_date."""
    curr = start_date
    added = 0
    while added < n:
        curr += datetime.timedelta(days=1)
        if curr.weekday() < 5:
            added += 1
    return curr


WHITELISTED_FEATURES = {
    ("CUTOFF_DRIFT", "time_of_day"),
}

# Day 3C: strictly no skipped features
SKIPPED_FEATURES: set[tuple[str, str]] = set()

TIER_1_ERROR_CLASSES = [
    "DUPLICATE_PAYMENT",
    "MISSING_SETTLEMENT",
    "FEE_TAX_MISMATCH",
    "PARTIAL_REFUND_MISMATCH",
    "CUTOFF_DRIFT",
    "SHORT_BANK_CREDIT",
    "ORPHAN_BANK_CREDIT",
]

TIER_2_AMBIGUITY_TYPES = [
    "NEAR_TOLERANCE",
    "EXPLAINED_CONSISTENT",
    "EXPLAINED_INCONSISTENT",
    "EXPLAINED_NO_AMOUNT",
]


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
    merchant_pay_stats: dict[str, tuple[float, float]],
    merchant_ref_stats: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Extract numerical non-semantic features for a recon item."""
    file_pos_pct = pos_idx / max(1, total_count)
    token = get_id_suffix(r.get("entity_id", ""))
    suffix = token[-6:] if len(token) >= 6 else token
    digits_ratio = sum(1 for c in suffix if c.isdigit()) / max(1, len(suffix))
    upper_ratio = sum(1 for c in suffix if c.isupper()) / max(1, len(suffix))

    m_id = r.get("merchant_id", "")
    amt = float(r.get("amount", 0))
    if r.get("type") == "refund":
        mean_amt, std_amt = merchant_ref_stats.get(m_id, (amt, 1.0))
    else:
        mean_amt, std_amt = merchant_pay_stats.get(m_id, (amt, 1.0))
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
    batch_info: dict[str, Any] | None = None,
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

    feats = {
        "file_position_pct": file_pos_pct,
        "id_suffix_digits_ratio": digits_ratio,
        "id_suffix_upper_ratio": upper_ratio,
        "sequence_id_gap": stmt_num / max(1, total_count),
        "merchant_amount_zscore": amt_zscore,
        "narration_length": desc_len,
        "null_count": null_count,
    }

    if batch_info:
        feats["batch_item_count"] = batch_info["item_count"]
        feats["batch_net_amount_zscore"] = batch_info["net_amount_zscore"]
        feats["batch_day_of_week"] = batch_info["day_of_week"]

    return feats


def audit_leakage(data: dict[str, Any], use_eligible_population: bool = True) -> dict[str, Any]:
    """Compute AUC for non-semantic features across all error and ambiguity classes."""
    payments = data.get("payments", [])
    recon_items = data.get("recon_items", [])
    settlements = data.get("settlements", [])
    bank_statement = data.get("bank_statement", [])
    labels = data.get("labels", [])
    variants = data.get("bank_narration_variants", {})
    manifest = data.get("manifest", {})
    rate_card = data.get("rate_card", {})
    as_of_str = manifest.get("as_of")
    if as_of_str:
        as_of_date = datetime.date.fromisoformat(as_of_str)
    else:
        last_stl = max(s["settled_at"] for s in settlements) if settlements else 0
        as_of_date = datetime.datetime.fromtimestamp(last_stl, tz=IST_TZ).date() + datetime.timedelta(days=10)

    # Index labels by error class or ambiguity type
    labels_by_class: dict[str, list[dict[str, Any]]] = {}
    injected_entities: set[str] = set()
    injected_batches: set[str] = set()

    for lbl in labels:
        err = lbl.get("error_class") or lbl.get("ambiguity_type")
        if err:
            labels_by_class.setdefault(err, []).append(lbl)
            injected_entities.update(lbl.get("entity_ids", []))
            if lbl.get("settlement_id"):
                injected_batches.add(lbl["settlement_id"])

    merchants = sorted(list({p["merchant_id"] for p in payments}))
    num_merchants = max(1, len(merchants))

    # Fast indexing
    recon_by_entity_id = {r["entity_id"]: r for r in recon_items}
    settlements_by_id = {s["settlement_id"]: s for s in settlements}
    items_by_settlement: dict[str, list[dict[str, Any]]] = {}
    for r in recon_items:
        items_by_settlement.setdefault(r["settlement_id"], []).append(r)

    # Index settlement bank rows
    bank_rows_by_settlement: dict[str, dict[str, Any]] = {}
    settlement_by_bank_stmt: dict[str, dict[str, Any]] = {}
    for b in bank_statement:
        for s in settlements:
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
                settlement_by_bank_stmt[b["statement_id"]] = s
                break

    # Precompute merchant stats on clean entities
    m_pay_amounts: dict[str, list[float]] = {}
    for p in payments:
        if p.get("payment_id") not in injected_entities and p.get("status") == "captured":
            m_pay_amounts.setdefault(p["merchant_id"], []).append(float(p["amount"]))
    m_pay_stats: dict[str, tuple[float, float]] = {}
    for m, vals in m_pay_amounts.items():
        mean_v = sum(vals) / len(vals)
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1))
        m_pay_stats[m] = (mean_v, std_v)

    m_ref_amounts: dict[str, list[float]] = {}
    for r in recon_items:
        if r.get("entity_id") not in injected_entities and r.get("type") == "refund":
            m_ref_amounts.setdefault(r["merchant_id"], []).append(float(r["amount"]))
    m_ref_stats: dict[str, tuple[float, float]] = {}
    for m, vals in m_ref_amounts.items():
        mean_v = sum(vals) / len(vals)
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1))
        m_ref_stats[m] = (mean_v, std_v)

    m_stl_amounts: dict[str, list[float]] = {}
    for s in settlements:
        if s["settlement_id"] not in injected_batches:
            m_stl_amounts.setdefault(s["merchant_id"], []).append(float(s["amount"]))
    m_stl_stats: dict[str, tuple[float, float]] = {}
    for m, vals in m_stl_amounts.items():
        mean_v = sum(vals) / len(vals)
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1))
        m_stl_stats[m] = (mean_v, std_v)

    m_bank_amounts: dict[str, list[float]] = {}
    for b in bank_statement:
        if b.get("statement_id") not in injected_entities and int(b.get("credit", 0)) > 0:
            m_bank_amounts.setdefault(b["merchant_id"], []).append(float(b["credit"]))
    m_bank_stats: dict[str, tuple[float, float]] = {}
    for m, vals in m_bank_amounts.items():
        mean_v = sum(vals) / len(vals)
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1))
        m_bank_stats[m] = (mean_v, std_v)

    # Pre-extract features
    pay_features = [
        extract_features_for_payment(p, idx, len(payments), m_pay_stats)
        for idx, p in enumerate(payments)
    ]
    pay_by_id = {p["payment_id"]: (p, pay_features[idx]) for idx, p in enumerate(payments)}

    recon_features = [
        extract_features_for_recon_item(r, idx, len(recon_items), m_pay_stats, m_ref_stats)
        for idx, r in enumerate(recon_items)
    ]
    recon_by_id = {r["entity_id"]: (r, recon_features[idx]) for idx, r in enumerate(recon_items)}

    def get_batch_info_for_bank_stmt(stmt_id: str) -> dict[str, Any] | None:
        stl = settlement_by_bank_stmt.get(stmt_id)
        if not stl:
            return None
        m_id = stl["merchant_id"]
        amt = float(stl["amount"])
        mean_amt, std_amt = m_stl_stats.get(m_id, (amt, 1.0))
        z = (amt - mean_amt) / max(1.0, std_amt)
        item_cnt = float(len(items_by_settlement.get(stl["settlement_id"], [])))
        dt = datetime.datetime.fromtimestamp(stl["settled_at"], tz=IST_TZ)
        dow = float(dt.weekday())
        return {
            "item_count": item_cnt,
            "net_amount_zscore": z,
            "day_of_week": dow,
        }

    bank_features = [
        extract_features_for_bank_row(
            b,
            idx,
            len(bank_statement),
            m_bank_stats,
            batch_info=get_batch_info_for_bank_stmt(b["statement_id"]),
        )
        for idx, b in enumerate(bank_statement)
    ]
    bank_by_id = {b["statement_id"]: (b, bank_features[idx]) for idx, b in enumerate(bank_statement)}

    # Base clean entity sets
    clean_captured_payments = [
        pay_features[idx]
        for idx, p in enumerate(payments)
        if p["payment_id"] not in injected_entities and p.get("status") == "captured"
    ]
    clean_all_recon = [
        recon_features[idx]
        for idx, r in enumerate(recon_items)
        if r["entity_id"] not in injected_entities
    ]
    clean_all_bank = [
        bank_features[idx]
        for idx, b in enumerate(bank_statement)
        if b["statement_id"] not in injected_entities and int(b.get("credit", 0)) > 0
    ]

    # Specific eligible clean subsets
    clean_payments_missing = []
    clean_payments_cutoff = []
    settle_lag = rate_card.get("payment_settlement_lag_days", 2)
    missing_after = rate_card.get("missing_after_days", 7)
    for idx, p in enumerate(payments):
        if p["payment_id"] in injected_entities or p.get("status") != "captured":
            continue
        # Missing settlement eligibility: scheduled settle date + 7 days <= as_of
        cap_ts = p.get("captured_at") or p.get("created_at") or 0
        cap_date = datetime.datetime.fromtimestamp(cap_ts, tz=IST_TZ).date()
        sched_settle = add_working_days(cap_date, settle_lag)
        if sched_settle + datetime.timedelta(days=missing_after) <= as_of_date:
            clean_payments_missing.append(pay_features[idx])
        # Cutoff drift eligibility: Mon-Thu creation
        dt = datetime.datetime.fromtimestamp(p.get("created_at") or 0, tz=IST_TZ)
        if dt.weekday() < 4:
            clean_payments_cutoff.append(pay_features[idx])

    clean_recon_feetax = [
        recon_features[idx]
        for idx, r in enumerate(recon_items)
        if r["entity_id"] not in injected_entities
        and r.get("type") == "payment"
        and r.get("method") in ("card", "netbanking")
    ]
    clean_recon_refunds = [
        recon_features[idx]
        for idx, r in enumerate(recon_items)
        if r["entity_id"] not in injected_entities and r.get("type") == "refund"
    ]
    clean_bank_settlements = [
        bank_features[idx]
        for idx, b in enumerate(bank_statement)
        if b["statement_id"] not in injected_entities
        and variants.get(b["statement_id"]) in ("clean", "noisy", "no_utr")
    ]

    # Target specifications
    if use_eligible_population:
        target_map = {
            "DUPLICATE_PAYMENT": clean_captured_payments,
            "MISSING_SETTLEMENT": clean_payments_missing,
            "CUTOFF_DRIFT": clean_payments_cutoff,
            "FEE_TAX_MISMATCH": clean_recon_feetax,
            "PARTIAL_REFUND_MISMATCH": clean_recon_refunds,
            "SHORT_BANK_CREDIT": clean_bank_settlements,
            "ORPHAN_BANK_CREDIT": clean_bank_settlements,
            "NEAR_TOLERANCE": clean_recon_feetax,
            "EXPLAINED_CONSISTENT": clean_recon_refunds,
            "EXPLAINED_INCONSISTENT": clean_recon_refunds,
            "EXPLAINED_NO_AMOUNT": clean_recon_refunds,
        }
    else:
        # Before / naive baseline
        target_map = {
            "DUPLICATE_PAYMENT": clean_captured_payments,
            "MISSING_SETTLEMENT": clean_captured_payments,
            "CUTOFF_DRIFT": clean_captured_payments,
            "FEE_TAX_MISMATCH": clean_all_recon,
            "PARTIAL_REFUND_MISMATCH": clean_all_recon,
            "SHORT_BANK_CREDIT": clean_all_bank,
            "ORPHAN_BANK_CREDIT": clean_all_bank,
            "NEAR_TOLERANCE": clean_all_recon,
            "EXPLAINED_CONSISTENT": clean_all_recon,
            "EXPLAINED_INCONSISTENT": clean_all_recon,
            "EXPLAINED_NO_AMOUNT": clean_all_recon,
        }

    results: dict[str, dict[str, float]] = {}
    population_sizes: dict[str, dict[str, int]] = {}
    merchant_concentrations: dict[str, dict[str, Any]] = {}
    narration_distributions: dict[str, dict[str, float]] = {}
    all_passed = True

    eval_classes = [c for c in TIER_1_ERROR_CLASSES + TIER_2_AMBIGUITY_TYPES if c in labels_by_class]

    for err_cls in eval_classes:
        clean_feats = target_map.get(err_cls, [])
        lbls = labels_by_class.get(err_cls, [])

        # Collect positive entities
        pos_entities: set[str] = set()
        m_counts: dict[str, int] = {}
        for l in lbls:
            m_id = l.get("merchant_id", "")
            m_counts[m_id] = m_counts.get(m_id, 0) + 1
            for eid in l.get("entity_ids", []):
                pos_entities.add(eid)

        # Map to features
        if err_cls in ("DUPLICATE_PAYMENT", "MISSING_SETTLEMENT", "CUTOFF_DRIFT"):
            pos_feats = [pay_by_id[eid][1] for eid in pos_entities if eid in pay_by_id]
        elif err_cls in ("SHORT_BANK_CREDIT", "ORPHAN_BANK_CREDIT"):
            pos_feats = [bank_by_id[eid][1] for eid in pos_entities if eid in bank_by_id]
        elif err_cls in (
            "PARTIAL_REFUND_MISMATCH",
            "EXPLAINED_CONSISTENT",
            "EXPLAINED_INCONSISTENT",
            "EXPLAINED_NO_AMOUNT",
        ):
            pos_feats = [
                recon_by_id[eid][1]
                for eid in pos_entities
                if eid in recon_by_id and recon_by_id[eid][0].get("type") == "refund"
            ]
        else:
            pos_feats = [
                recon_by_id[eid][1]
                for eid in pos_entities
                if eid in recon_by_id and recon_by_id[eid][0].get("type") == "payment"
            ]

        population_sizes[err_cls] = {
            "pos": len(pos_feats),
            "neg": len(clean_feats),
        }

        # Concentration check
        total_cases = len(lbls)
        uniform_share = total_cases / num_merchants
        max_allowed = 3.0 * uniform_share
        all_m_counts = [m_counts.get(m, 0) for m in merchants]
        all_m_counts.sort()
        min_c = min(all_m_counts)
        med_c = all_m_counts[num_merchants // 2]
        max_c = max(all_m_counts)
        # Injected error classes strictly enforce <= 3x uniform share
        conc_pass = max_c <= max_allowed if err_cls in TIER_1_ERROR_CLASSES else True
        merchant_concentrations[err_cls] = {
            "min": min_c,
            "median": med_c,
            "max": max_c,
            "uniform_share": round(uniform_share, 2),
            "max_allowed": round(max_allowed, 2),
            "passes": conc_pass,
        }
        if not conc_pass:
            all_passed = False

        # Narration distribution for bank classes
        if err_cls in ("SHORT_BANK_CREDIT", "ORPHAN_BANK_CREDIT"):
            var_counts: dict[str, int] = {}
            for eid in pos_entities:
                v = variants.get(eid, "unknown")
                var_counts[v] = var_counts.get(v, 0) + 1
            tot = max(1, len(pos_entities))
            narration_distributions[err_cls] = {
                "clean": round(100.0 * var_counts.get("clean", 0) / tot, 1),
                "noisy": round(100.0 * var_counts.get("noisy", 0) / tot, 1),
                "no_utr": round(100.0 * var_counts.get("no_utr", 0) / tot, 1),
            }

        if not pos_feats or not clean_feats:
            continue

        results[err_cls] = {}
        # Keys present in all pos and all clean items
        feature_names = [
            k
            for k in pos_feats[0].keys()
            if all(k in f for f in pos_feats) and all(k in f for f in clean_feats)
        ]
        for f_name in feature_names:
            if (err_cls, f_name) in SKIPPED_FEATURES:
                continue

            pos_vals = [f[f_name] for f in pos_feats]
            neg_vals = [f[f_name] for f in clean_feats]
            auc = compute_auc(pos_vals, neg_vals)
            results[err_cls][f_name] = round(auc, 4)

            is_whitelisted = (err_cls, f_name) in WHITELISTED_FEATURES
            if not is_whitelisted and not (0.40 <= auc <= 0.60):
                all_passed = False

    summary_table = format_leakage_table(
        results,
        population_sizes=population_sizes,
        merchant_concentrations=merchant_concentrations,
        narration_distributions=narration_distributions,
    )

    return {
        "results": results,
        "population_sizes": population_sizes,
        "merchant_concentrations": merchant_concentrations,
        "narration_distributions": narration_distributions,
        "summary_table": summary_table,
        "all_passed": all_passed,
    }


def format_leakage_table(
    results: dict[str, dict[str, float]],
    population_sizes: dict[str, dict[str, int]] | None = None,
    merchant_concentrations: dict[str, dict[str, Any]] | None = None,
    narration_distributions: dict[str, dict[str, float]] | None = None,
) -> str:
    """Format AUC leakage results as an ASCII / Markdown table."""
    all_features = sorted(list({f for res in results.values() for f in res.keys()}))
    headers = ["Class", "N_pos", "N_neg"] + all_features + ["Conc (min/med/max)", "Status"]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|:" + "---:|:" * (len(headers) - 1) + "---:|"

    lines = [header_line, sep_line]
    for err_cls, feats in results.items():
        row = [f"{err_cls:<23}"]
        if population_sizes and err_cls in population_sizes:
            row.append(f"{population_sizes[err_cls]['pos']:>5}")
            row.append(f"{population_sizes[err_cls]['neg']:>5}")
        else:
            row.extend(["  N/A", "  N/A"])

        cls_pass = True
        for f_name in all_features:
            if f_name in feats:
                auc = feats[f_name]
                is_whitelisted = (err_cls, f_name) in WHITELISTED_FEATURES
                passes = is_whitelisted or (0.40 <= auc <= 0.60)
                if not passes:
                    cls_pass = False
                tag = f"{auc:.3f}" + ("*" if is_whitelisted else "")
                row.append(f"{tag:>10}")
            else:
                row.append("       N/A")

        if merchant_concentrations and err_cls in merchant_concentrations:
            mc = merchant_concentrations[err_cls]
            conc_str = f"{mc['min']}/{mc['median']}/{mc['max']} (<= {mc['max_allowed']:.1f})"
            if not mc["passes"]:
                cls_pass = False
            row.append(f"{conc_str:>19}")
        else:
            row.append("                N/A")

        status = "PASS" if cls_pass else "FAIL"
        row.append(f"{status:^8}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n* Whitelisted semantic feature (CUTOFF_DRIFT time_of_day only)")
    if narration_distributions:
        lines.append("\n** Bank Narration Distributions (clean / noisy / no_utr):")
        for cls_name, dist in narration_distributions.items():
            lines.append(f"  - {cls_name}: {dist['clean']}% clean, {dist['noisy']}% noisy, {dist['no_utr']}% no_utr")

    return "\n".join(lines)
