"""Unit tests for System B1r regex & number-word reason parser engine."""

import ast
import os
import pytest

from src.recon_agent.pipeline.types import (
    OrderLedgerRow,
    RateCard,
    ReconDataset,
    ReconItemRow,
)
from src.recon_agent.systems.b1r import parse_reason_amount, run_b1r

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sample_rate_card() -> RateCard:
    return RateCard(
        card_bps=200,
        upi_bps=0,
        netbanking_bps=150,
        gst_bps=1800,
        fee_includes_tax=True,
        payment_settlement_lag_days=2,
        refund_settlement_lag_days=1,
        missing_after_days=7,
        rounding_tolerance_paise=2,
        escalate_tolerance_paise=9,
        error_threshold_paise=10,
    )


def _make_recon_item(
    entity_id: str,
    type: str,
    debit: int = 0,
    credit: int = 0,
    amount: int = 0,
    fee: int = 0,
    tax: int = 0,
    method: str | None = "card",
    settlement_id: str | None = "setl_01",
    merchant_id: str = "mer_01",
    payment_id: str | None = None,
    order_id: str | None = None,
    notes: dict | None = None,
) -> ReconItemRow:
    return ReconItemRow(
        entity_id=entity_id,
        type=type,
        debit=debit,
        credit=credit,
        amount=amount,
        currency="INR",
        fee=fee,
        tax=tax,
        on_hold=False,
        settled=True,
        created_at=1000,
        settled_at=1780308000,
        settlement_id=settlement_id,
        description=None,
        notes=notes,
        payment_id=payment_id,
        settlement_utr=None,
        order_id=order_id,
        order_receipt=None,
        method=method,
        card_network=None,
        card_issuer=None,
        card_type=None,
        dispute_id=None,
        merchant_id=merchant_id,
    )


def test_b1r_boundary_isolation() -> None:
    """Guarantee that systems/b1r.py does NOT import synth or read eval files."""
    b1r_path = os.path.join(REPO_ROOT, "src", "recon_agent", "systems", "b1r.py")
    with open(b1r_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename="b1r.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "synth" not in alias.name, f"Illegal synth import in b1r.py: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "synth" not in mod, f"Illegal synth import-from in b1r.py: {mod}"

    forbidden = ["labels.jsonl", "truth_links.jsonl", "injection_manifest.jsonl"]
    for word in forbidden:
        assert word not in src, f"b1r.py must not reference eval file: {word}"


def test_parse_reason_amount_formats() -> None:
    """Test regex and number word parser across diverse monetary formats."""
    # 1. Standard currency prefixes
    assert parse_reason_amount("Adjustment of Rs 50 applied") == 5000
    assert parse_reason_amount("Fee waiver of Rs. 15.50 approved") == 1550
    assert parse_reason_amount("Difference INR 100 refunded") == 10000
    assert parse_reason_amount("Reversed \u20b9 45.25 for customer") == 4525
    assert parse_reason_amount("Credited Rs 1,250.00 to account") == 125000

    # 2. Explicit numeric paise
    assert parse_reason_amount("Rounding delta 50 paise forgiven") == 50
    assert parse_reason_amount("Difference of 75 paise") == 75

    # 3. English number-words
    assert parse_reason_amount("Currency conversion variance of four rupees and ten paise absorbed") == 410
    assert parse_reason_amount("Goodwill discount of fifty rupees applied") == 5000
    assert parse_reason_amount("Adjustment of one hundred rupees") == 10000
    assert parse_reason_amount("Fractional fee of twenty-five paise") == 25

    # 4. Word boundary protection (e.g. 'rs' in conversion must not trigger)
    assert parse_reason_amount("Customer requested waiver without specifying any number") is None
    assert parse_reason_amount("Conversion error occurred") is None


def test_b1r_classification_decisions() -> None:
    """Test B1r logic for benign_explained, flag mismatch, and escalate_human."""
    rc = _sample_rate_card()
    m_id = "mer_01"

    # Order refund amount is 40,000 paise (Rs 400.00)
    orders = (
        OrderLedgerRow(order_id="ord_01", merchant_id=m_id, booked_date="2026-06-01", created_at=1000, expected_amount=100000, currency="INR", order_receipt="r1", status="paid", payment_id="p1", refund_id="rfnd_1", refund_amount=40000),
        OrderLedgerRow(order_id="ord_02", merchant_id=m_id, booked_date="2026-06-01", created_at=1000, expected_amount=100000, currency="INR", order_receipt="r2", status="paid", payment_id="p2", refund_id="rfnd_2", refund_amount=40000),
        OrderLedgerRow(order_id="ord_03", merchant_id=m_id, booked_date="2026-06-01", created_at=1000, expected_amount=100000, currency="INR", order_receipt="r3", status="paid", payment_id="p3", refund_id="rfnd_3", refund_amount=40000),
        OrderLedgerRow(order_id="ord_04", merchant_id=m_id, booked_date="2026-06-01", created_at=1000, expected_amount=100000, currency="INR", order_receipt="r4", status="paid", payment_id="p4", refund_id="rfnd_4", refund_amount=40000),
    )

    recon_items = (
        # 1. Consistent: debit is 45,000 (diff 5,000 paise = Rs 50). Note states Rs 50 -> benign_explained
        _make_recon_item(entity_id="rfnd_match", type="refund", debit=45000, amount=45000, payment_id="p1", order_id="ord_01", notes={"reason_text": "Goodwill Rs 50 approved"}),
        # 2. Inconsistent: debit is 45,000 (diff 5,000 paise). Note states Rs 20 (2,000 paise) -> flag
        _make_recon_item(entity_id="rfnd_mismatch", type="refund", debit=45000, amount=45000, payment_id="p2", order_id="ord_02", notes={"reason_text": "Fee waiver of Rs 20 approved"}),
        # 3. No amount: debit is 45,000. Note has no amount -> escalate_human + residual
        _make_recon_item(entity_id="rfnd_no_amt", type="refund", debit=45000, amount=45000, payment_id="p3", order_id="ord_03", notes={"reason_text": "Special exception applied per manager"}),
        # 4. Unannotated: debit is 45,000. No reason_text -> remains standard B1 flag
        _make_recon_item(entity_id="rfnd_unannotated", type="refund", debit=45000, amount=45000, payment_id="p4", order_id="ord_04", notes=None),
    )

    dataset = ReconDataset(
        orders_ledger=orders,
        payments=(),
        recon_items=recon_items,
        settlements=(),
        bank_statement=(),
        rate_card=rc,
        as_of="2026-06-15",
    )

    out = run_b1r(dataset)
    pred_map = {p.entity_ids[0]: p for p in out.predictions}

    assert pred_map["rfnd_match"].action == "benign_explained"
    assert pred_map["rfnd_match"].rule_id == "B1R_EXPLAINED_MATCH"

    assert pred_map["rfnd_mismatch"].action == "flag"
    assert pred_map["rfnd_mismatch"].rule_id == "B1R_EXPLAINED_MISMATCH"

    assert pred_map["rfnd_no_amt"].action == "escalate_human"
    assert pred_map["rfnd_no_amt"].rule_id == "B1R_EXPLAINED_NO_AMOUNT"

    assert pred_map["rfnd_unannotated"].action == "flag"
    assert pred_map["rfnd_unannotated"].rule_id == "B1_PARTIAL_REFUND_MISMATCH"

    # Check residual recorded for rfnd_no_amt
    residual_eids = {eid for r in out.residuals for eid in r.entity_ids}
    assert "rfnd_no_amt" in residual_eids
