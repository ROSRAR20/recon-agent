"""System B1: Deterministic Rule-Based Reconciliation Engine.

Implements multi-tiered bank-to-settlement matching and deterministic anomaly detectors.
Reads all rates, tolerances, and policy thresholds dynamically from dataset.rate_card.
Uses documented policy constants from docs/b1_rules.md for matching grace and thresholds.

Zero text parsing or LLM dependencies.
"""

import datetime
import re
from typing import Any

from src.recon_agent.pipeline.types import (
    BankSettlementLink,
    Prediction,
    ReconDataset,
    ResidualCase,
    SystemOutput,
)

# Documented policy constants from docs/b1_rules.md [POLICY]
LATE_GRACE_WORKING_DAYS = 2
T2_MIN_UTR_LENGTH = 10
IST_OFFSET = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def _add_working_days(d: datetime.date, n: int) -> datetime.date:
    """Add n business days (Monday-Friday) to date d."""
    curr = d
    added = 0
    while added < n:
        curr += datetime.timedelta(days=1)
        if curr.weekday() < 5:
            added += 1
    return curr


def _normalize_text(text: str) -> str:
    """Strip all punctuation and whitespace, converting to lower-case alphanumeric."""
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def _to_ist_date(timestamp: int) -> datetime.date:
    """Convert unix epoch timestamp to IST calendar date."""
    return datetime.datetime.fromtimestamp(timestamp, tz=IST_OFFSET).date()


def run_b1(dataset: ReconDataset) -> SystemOutput:
    """Execute System B1 deterministic rule engine on observable reconciliation dataset."""
    predictions: list[Prediction] = []
    bank_links: list[BankSettlementLink] = []
    residuals: list[ResidualCase] = []

    rc = dataset.rate_card
    as_of_date = datetime.date.fromisoformat(dataset.as_of)

    # Index orders by order_id
    orders_map = {o.order_id: o for o in dataset.orders_ledger}

    # =========================================================================
    # 1. Tiered Bank-to-Settlement Matching (One-to-One)
    # =========================================================================
    # Group settlements and credit bank rows by merchant
    stls_by_merchant: dict[str, list[Any]] = {}
    for s in dataset.settlements:
        stls_by_merchant.setdefault(s.merchant_id, []).append(s)

    bank_by_merchant: dict[str, list[Any]] = {}
    for b in dataset.bank_statement:
        if b.credit > 0:
            bank_by_merchant.setdefault(b.merchant_id, []).append(b)

    all_merchants = set(stls_by_merchant.keys()) | set(bank_by_merchant.keys())

    matched_statement_ids: set[str] = set()
    matched_settlement_ids: set[str] = set()
    matched_links_map: dict[str, Any] = {}  # stmt_id -> settlement

    for m_id in all_merchants:
        m_stls = list(stls_by_merchant.get(m_id, []))
        m_bank = list(bank_by_merchant.get(m_id, []))

        # ---------------------------------------------------------------------
        # Tier 1: Exact UTR substring match
        # ---------------------------------------------------------------------
        for s in list(m_stls):
            if not s.utr:
                continue
            for b in list(m_bank):
                if s.utr in b.description:
                    bank_links.append(BankSettlementLink(statement_id=b.statement_id, settlement_id=s.settlement_id))
                    matched_statement_ids.add(b.statement_id)
                    matched_settlement_ids.add(s.settlement_id)
                    matched_links_map[b.statement_id] = s
                    m_stls.remove(s)
                    m_bank.remove(b)
                    break

        # ---------------------------------------------------------------------
        # Tier 2: Normalized & Truncated UTR match
        # ---------------------------------------------------------------------
        # Precompute normalized descriptions and dates for available bank rows
        for s in list(m_stls):
            if not s.utr:
                continue
            s_date = _to_ist_date(s.settled_at)
            max_date = _add_working_days(s_date, LATE_GRACE_WORKING_DAYS)
            norm_utr = _normalize_text(s.utr)
            trunc_utr = norm_utr[:12] if len(norm_utr) >= 12 else (norm_utr[:T2_MIN_UTR_LENGTH] if len(norm_utr) >= T2_MIN_UTR_LENGTH else "")

            candidates: list[Any] = []
            for b in m_bank:
                b_date = datetime.date.fromisoformat(b.date)
                if s_date <= b_date <= max_date:
                    norm_desc = _normalize_text(b.description)
                    if norm_utr in norm_desc or (trunc_utr and trunc_utr in norm_desc):
                        candidates.append(b)

            if len(candidates) == 1:
                b_cand = candidates[0]
                # Check uniqueness from bank row's perspective
                b_date = datetime.date.fromisoformat(b_cand.date)
                norm_desc = _normalize_text(b_cand.description)
                reverse_stls = [
                    s_other for s_other in m_stls
                    if _to_ist_date(s_other.settled_at) <= b_date <= _add_working_days(_to_ist_date(s_other.settled_at), LATE_GRACE_WORKING_DAYS)
                    and (_normalize_text(s_other.utr) in norm_desc or (_normalize_text(s_other.utr)[:12] in norm_desc if len(_normalize_text(s_other.utr)) >= 12 else False))
                ]
                if len(reverse_stls) == 1:
                    bank_links.append(BankSettlementLink(statement_id=b_cand.statement_id, settlement_id=s.settlement_id))
                    matched_statement_ids.add(b_cand.statement_id)
                    matched_settlement_ids.add(s.settlement_id)
                    matched_links_map[b_cand.statement_id] = s
                    m_stls.remove(s)
                    m_bank.remove(b_cand)
                else:
                    # Ambiguous settlements matching this bank row -> escalate
                    predictions.append(
                        Prediction(
                            merchant_id=m_id,
                            predicted_class="UNKNOWN",
                            action="escalate_human",
                            entity_ids=tuple(s_rev.settlement_id for s_rev in reverse_stls),
                            rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
                        )
                    )
                    residuals.append(
                        ResidualCase(
                            merchant_id=m_id,
                            entity_ids=tuple(s_rev.settlement_id for s_rev in reverse_stls),
                            rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
                            reason=f"Ambiguous Tier 2 candidate matches ({len(reverse_stls)} settlements for bank row {b_cand.statement_id})",
                        )
                    )
            elif len(candidates) >= 2:
                # Ambiguous bank candidates for this settlement -> escalate
                predictions.append(
                    Prediction(
                        merchant_id=m_id,
                        predicted_class="UNKNOWN",
                        action="escalate_human",
                        entity_ids=tuple(b.statement_id for b in candidates) + (s.settlement_id,),
                        settlement_id=s.settlement_id,
                        rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
                    )
                )
                residuals.append(
                    ResidualCase(
                        merchant_id=m_id,
                        entity_ids=tuple(b.statement_id for b in candidates) + (s.settlement_id,),
                        rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
                        reason=f"Ambiguous Tier 2 candidate matches ({len(candidates)} bank rows for settlement {s.settlement_id})",
                    )
                )

        # ---------------------------------------------------------------------
        # Tier 3: No-UTR Exact Amount match
        # ---------------------------------------------------------------------
        # Candidate bank rows: Razorpay narration within date window, amount == settlement amount
        for s in list(m_stls):
            s_date = _to_ist_date(s.settled_at)
            max_date = _add_working_days(s_date, LATE_GRACE_WORKING_DAYS)
            candidates = [
                b for b in m_bank
                if "razorpay" in b.description.lower()
                and s_date <= datetime.date.fromisoformat(b.date) <= max_date
                and b.credit == s.amount
            ]

            if len(candidates) == 1:
                b_cand = candidates[0]
                b_date = datetime.date.fromisoformat(b_cand.date)
                reverse_stls = [
                    s_other for s_other in m_stls
                    if _to_ist_date(s_other.settled_at) <= b_date <= _add_working_days(_to_ist_date(s_other.settled_at), LATE_GRACE_WORKING_DAYS)
                    and s_other.amount == b_cand.credit
                ]
                if len(reverse_stls) == 1:
                    bank_links.append(BankSettlementLink(statement_id=b_cand.statement_id, settlement_id=s.settlement_id))
                    matched_statement_ids.add(b_cand.statement_id)
                    matched_settlement_ids.add(s.settlement_id)
                    matched_links_map[b_cand.statement_id] = s
                    m_stls.remove(s)
                    m_bank.remove(b_cand)
                else:
                    predictions.append(
                        Prediction(
                            merchant_id=m_id,
                            predicted_class="UNKNOWN",
                            action="escalate_human",
                            entity_ids=tuple(s_rev.settlement_id for s_rev in reverse_stls),
                            rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
                        )
                    )
                    residuals.append(
                        ResidualCase(
                            merchant_id=m_id,
                            entity_ids=tuple(s_rev.settlement_id for s_rev in reverse_stls),
                            rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
                            reason=f"Ambiguous Tier 3 candidate matches ({len(reverse_stls)} settlements for bank row {b_cand.statement_id})",
                        )
                    )
            elif len(candidates) >= 2:
                predictions.append(
                    Prediction(
                        merchant_id=m_id,
                        predicted_class="UNKNOWN",
                        action="escalate_human",
                        entity_ids=tuple(b.statement_id for b in candidates) + (s.settlement_id,),
                        settlement_id=s.settlement_id,
                        rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
                    )
                )
                residuals.append(
                    ResidualCase(
                        merchant_id=m_id,
                        entity_ids=tuple(b.statement_id for b in candidates) + (s.settlement_id,),
                        rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
                        reason=f"Ambiguous Tier 3 candidate matches ({len(candidates)} bank rows for settlement {s.settlement_id})",
                    )
                )

        # ---------------------------------------------------------------------
        # Tier 4: No-UTR Short Credit match (amount below net)
        # ---------------------------------------------------------------------
        # Remaining unmatched Razorpay-style rows with amount below net, exactly one candidate pair
        for s in list(m_stls):
            s_date = _to_ist_date(s.settled_at)
            max_date = _add_working_days(s_date, LATE_GRACE_WORKING_DAYS)
            candidates = [
                b for b in m_bank
                if "razorpay" in b.description.lower()
                and s_date <= datetime.date.fromisoformat(b.date) <= max_date
                and b.credit < s.amount
            ]

            if len(candidates) == 1:
                b_cand = candidates[0]
                b_date = datetime.date.fromisoformat(b_cand.date)
                reverse_stls = [
                    s_other for s_other in m_stls
                    if _to_ist_date(s_other.settled_at) <= b_date <= _add_working_days(_to_ist_date(s_other.settled_at), LATE_GRACE_WORKING_DAYS)
                    and b_cand.credit < s_other.amount
                ]
                if len(reverse_stls) == 1:
                    bank_links.append(BankSettlementLink(statement_id=b_cand.statement_id, settlement_id=s.settlement_id))
                    matched_statement_ids.add(b_cand.statement_id)
                    matched_settlement_ids.add(s.settlement_id)
                    matched_links_map[b_cand.statement_id] = s
                    m_stls.remove(s)
                    m_bank.remove(b_cand)

        # ---------------------------------------------------------------------
        # Unmatched Settlements: Escalate if expired past grace
        # ---------------------------------------------------------------------
        for s in m_stls:
            s_date = _to_ist_date(s.settled_at)
            grace_expiry = _add_working_days(s_date, LATE_GRACE_WORKING_DAYS)
            if as_of_date > grace_expiry:
                predictions.append(
                    Prediction(
                        merchant_id=m_id,
                        predicted_class="UNKNOWN",
                        action="escalate_human",
                        entity_ids=(s.settlement_id,),
                        settlement_id=s.settlement_id,
                        rule_id="B1_UNMATCHED_SETTLEMENT_EXPIRED",
                    )
                )
                residuals.append(
                    ResidualCase(
                        merchant_id=m_id,
                        entity_ids=(s.settlement_id,),
                        rule_id="B1_UNMATCHED_SETTLEMENT_EXPIRED",
                        reason=f"Settlement {s.settlement_id} settled on {s_date} remains unmatched past grace window (as_of {as_of_date})",
                    )
                )

        # ---------------------------------------------------------------------
        # Unmatched Bank Credits: ORPHAN_BANK_CREDIT if Razorpay narration
        # ---------------------------------------------------------------------
        for b in m_bank:
            if "razorpay" in b.description.lower():
                predictions.append(
                    Prediction(
                        merchant_id=m_id,
                        predicted_class="ORPHAN_BANK_CREDIT",
                        action="flag",
                        entity_ids=(b.statement_id,),
                        rule_id="B1_ORPHAN_BANK_CREDIT",
                    )
                )

    # =========================================================================
    # 2. Detector: SHORT_BANK_CREDIT (matched bank amount < settlement net)
    # =========================================================================
    for b in dataset.bank_statement:
        if b.statement_id in matched_links_map:
            s = matched_links_map[b.statement_id]
            if b.credit < (s.amount - rc.rounding_tolerance_paise):
                predictions.append(
                    Prediction(
                        merchant_id=b.merchant_id,
                        predicted_class="SHORT_BANK_CREDIT",
                        action="flag",
                        entity_ids=(b.statement_id,),
                        settlement_id=s.settlement_id,
                        rule_id="B1_SHORT_BANK_CREDIT",
                    )
                )

    # =========================================================================
    # 3. Detector: DUPLICATE_PAYMENT (2+ captured payments with same merchant, order, amount)
    # =========================================================================
    captured_groups: dict[tuple[str, str, int], list[Any]] = {}
    for p in dataset.payments:
        if p.status == "captured":
            key = (p.merchant_id, p.order_id, p.amount)
            captured_groups.setdefault(key, []).append(p)

    for (m_id, ord_id, amt), p_list in captured_groups.items():
        if len(p_list) > 1:
            p_sorted = sorted(p_list, key=lambda x: (x.created_at, x.payment_id))
            # The earliest payment is considered legitimate; subsequent ones are duplicates
            for dup_p in p_sorted[1:]:
                predictions.append(
                    Prediction(
                        merchant_id=m_id,
                        predicted_class="DUPLICATE_PAYMENT",
                        action="flag",
                        entity_ids=(dup_p.payment_id,),
                        rule_id="B1_DUPLICATE_PAYMENT",
                    )
                )

    # =========================================================================
    # 4. Detector: MISSING_SETTLEMENT (captured payment, no recon item, past grace)
    # =========================================================================
    recon_payment_ids: set[str] = set()
    for r in dataset.recon_items:
        if r.type == "payment":
            if r.payment_id:
                recon_payment_ids.add(r.payment_id)
            recon_payment_ids.add(r.entity_id)

    for p in dataset.payments:
        if p.status == "captured" and p.payment_id not in recon_payment_ids:
            if p.captured_at is not None:
                capture_dt = _to_ist_date(p.captured_at)
                sched_settle = _add_working_days(capture_dt, rc.payment_settlement_lag_days)
                grace_expiry = sched_settle + datetime.timedelta(days=rc.missing_after_days)
                if as_of_date > grace_expiry:
                    predictions.append(
                        Prediction(
                            merchant_id=p.merchant_id,
                            predicted_class="MISSING_SETTLEMENT",
                            action="flag",
                            entity_ids=(p.payment_id,),
                            rule_id="B1_MISSING_SETTLEMENT",
                        )
                    )

    # =========================================================================
    # 5. Detector: FEE_TAX_MISMATCH & NEAR_TOLERANCE
    # =========================================================================
    for r in dataset.recon_items:
        if r.type == "payment":
            method = r.method or "card"
            if method == "card":
                rate_bps = rc.card_bps
            elif method == "netbanking":
                rate_bps = rc.netbanking_bps
            elif method == "upi":
                rate_bps = rc.upi_bps
            else:
                rate_bps = rc.card_bps

            # Integer half-up rounding: (amount * bps + 5000) // 10000
            base_fee = (r.amount * rate_bps + 5000) // 10000
            expected_tax = (base_fee * rc.gst_bps + 5000) // 10000
            expected_fee = base_fee + expected_tax if rc.fee_includes_tax else base_fee

            fee_delta = abs(r.fee - expected_fee)
            tax_delta = abs(r.tax - expected_tax)
            delta = max(fee_delta, tax_delta)

            if delta <= rc.rounding_tolerance_paise:
                pass  # Benign rounding tolerance, never flag
            elif delta <= rc.escalate_tolerance_paise:
                predictions.append(
                    Prediction(
                        merchant_id=r.merchant_id,
                        predicted_class="FEE_TAX_MISMATCH",
                        action="escalate_human",
                        entity_ids=(r.entity_id,),
                        settlement_id=r.settlement_id,
                        rule_id="B1_NEAR_TOLERANCE",
                    )
                )
                residuals.append(
                    ResidualCase(
                        merchant_id=r.merchant_id,
                        entity_ids=(r.entity_id,),
                        rule_id="B1_NEAR_TOLERANCE",
                        reason=f"Fee/tax deviation {delta} paise falls within escalation band (3-9 paise)",
                    )
                )
            elif delta >= rc.error_threshold_paise:
                predictions.append(
                    Prediction(
                        merchant_id=r.merchant_id,
                        predicted_class="FEE_TAX_MISMATCH",
                        action="flag",
                        entity_ids=(r.entity_id,),
                        settlement_id=r.settlement_id,
                        rule_id="B1_FEE_TAX_MISMATCH",
                    )
                )

    # =========================================================================
    # 6. Detector: PARTIAL_REFUND_MISMATCH
    # =========================================================================
    for r in dataset.recon_items:
        if r.type == "refund":
            order = orders_map.get(r.order_id)
            if order is None or order.refund_amount is None or order.refund_id is None:
                predictions.append(
                    Prediction(
                        merchant_id=r.merchant_id,
                        predicted_class="PARTIAL_REFUND_MISMATCH",
                        action="flag",
                        entity_ids=(r.entity_id,),
                        settlement_id=r.settlement_id,
                        rule_id="B1_PARTIAL_REFUND_MISMATCH",
                    )
                )
            else:
                diff = abs(r.debit - order.refund_amount)
                if diff > rc.rounding_tolerance_paise:
                    predictions.append(
                        Prediction(
                            merchant_id=r.merchant_id,
                            predicted_class="PARTIAL_REFUND_MISMATCH",
                            action="flag",
                            entity_ids=(r.entity_id,),
                            settlement_id=r.settlement_id,
                            rule_id="B1_PARTIAL_REFUND_MISMATCH",
                        )
                    )

    # =========================================================================
    # 7. Detector: CUTOFF_DRIFT (ledger booked_date != IST date of capture)
    # =========================================================================
    for p in dataset.payments:
        if p.status == "captured" and p.captured_at is not None:
            capture_date_str = _to_ist_date(p.captured_at).isoformat()
            order = orders_map.get(p.order_id)
            if order is not None and order.booked_date and order.booked_date != capture_date_str:
                predictions.append(
                    Prediction(
                        merchant_id=p.merchant_id,
                        predicted_class="CUTOFF_DRIFT",
                        action="flag",
                        entity_ids=(p.payment_id, order.order_id),
                        rule_id="B1_CUTOFF_DRIFT",
                    )
                )

    return SystemOutput(
        predictions=tuple(predictions),
        bank_links=tuple(bank_links),
        residuals=tuple(residuals),
    )
