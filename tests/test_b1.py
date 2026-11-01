"""Unit tests for System B1 deterministic reconciliation engine."""

import ast
import os
import pytest

from src.recon_agent.pipeline.types import (
    BankSettlementLink,
    BankStatementRow,
    OrderLedgerRow,
    PaymentRow,
    RateCard,
    ReconDataset,
    ReconItemRow,
    SettlementRow,
)
from src.recon_agent.systems.b1 import run_b1

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


def test_b1_boundary_isolation() -> None:
    """Guarantee that systems/b1.py does NOT import synth or read eval files."""
    b1_path = os.path.join(REPO_ROOT, "src", "recon_agent", "systems", "b1.py")
    with open(b1_path, "r", encoding="utf-8") as f:
        src = f.read()

    # Disallow synth imports
    tree = ast.parse(src, filename="b1.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "synth" not in alias.name, f"Illegal synth import in b1.py: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "synth" not in mod, f"Illegal synth import-from in b1.py: {mod}"

    # Disallow references to eval-only files
    forbidden = ["labels.jsonl", "truth_links.jsonl", "injection_manifest.jsonl"]
    for word in forbidden:
        assert word not in src, f"b1.py must not reference eval file: {word}"


def test_b1_bank_matching_tiers_and_ambiguity() -> None:
    """Test bank matching across T1 (exact), T2 (norm/trunc), T3 (amount), T4 (short credit) and ambiguity."""
    rc = _sample_rate_card()
    m_id = "mer_01"

    # Base timestamp: 2026-06-01 10:00:00 UTC (15:30 IST) -> 1780308000
    # Settle date IST: 2026-06-01
    settled_at = 1780308000

    settlements = (
        SettlementRow(settlement_id="setl_t1", merchant_id=m_id, amount=100000, utr="UTR_EXACT_12345", status="settled", created_at=1000, settled_at=settled_at),
        SettlementRow(settlement_id="setl_t2", merchant_id=m_id, amount=200000, utr="UTR_LONG_NORM_9988776655", status="settled", created_at=1000, settled_at=settled_at),
        SettlementRow(settlement_id="setl_t3", merchant_id=m_id, amount=300000, utr="UTR_MISC_333", status="settled", created_at=1000, settled_at=settled_at),
        SettlementRow(settlement_id="setl_t4", merchant_id=m_id, amount=400000, utr="UTR_MISC_444", status="settled", created_at=1000, settled_at=settled_at),
        SettlementRow(settlement_id="setl_amb_a", merchant_id="mer_amb", amount=350000, utr="UTR_AMB_555", status="settled", created_at=1000, settled_at=settled_at),
        SettlementRow(settlement_id="setl_amb_b", merchant_id="mer_amb", amount=350000, utr="UTR_AMB_666", status="settled", created_at=1000, settled_at=settled_at),
    )

    bank_statement = (
        # T1: Exact UTR
        BankStatementRow(statement_id="stmt_t1", merchant_id=m_id, date="2026-06-01", description="ACH CR RAZORPAY UTR_EXACT_12345", debit=0, credit=100000, balance=1000000),
        # T2: Normalized & truncated (UTR_LONG_NORM_9988776655 -> utrlongnorm99)
        BankStatementRow(statement_id="stmt_t2", merchant_id=m_id, date="2026-06-02", description="ACH CR RAZORPAY / utr-long-norm-9988776655-extra", debit=0, credit=200000, balance=1200000),
        # T3: No-UTR exact amount
        BankStatementRow(statement_id="stmt_t3", merchant_id=m_id, date="2026-06-02", description="ACH CR RAZORPAY SETTLEMENT NO UTR", debit=0, credit=300000, balance=1500000),
        # T4: No-UTR short credit
        BankStatementRow(statement_id="stmt_t4", merchant_id=m_id, date="2026-06-02", description="ACH CR RAZORPAY DEDUCTED", debit=0, credit=395000, balance=1895000),
        # Ambiguity: two bank rows matching setl_amb_a and setl_amb_b
        BankStatementRow(statement_id="stmt_amb_1", merchant_id="mer_amb", date="2026-06-02", description="ACH CR RAZORPAY BATCH 1", debit=0, credit=350000, balance=2395000),
        BankStatementRow(statement_id="stmt_amb_2", merchant_id="mer_amb", date="2026-06-02", description="ACH CR RAZORPAY BATCH 2", debit=0, credit=350000, balance=2895000),
    )

    dataset = ReconDataset(
        orders_ledger=(),
        payments=(),
        recon_items=(),
        settlements=settlements,
        bank_statement=bank_statement,
        rate_card=rc,
        as_of="2026-06-15",
    )

    out = run_b1(dataset)
    link_map = {link.statement_id: link.settlement_id for link in out.bank_links}

    # T1 match
    assert link_map.get("stmt_t1") == "setl_t1"
    # T2 match
    assert link_map.get("stmt_t2") == "setl_t2"
    # T3 match
    assert link_map.get("stmt_t3") == "setl_t3"
    # T4 match
    assert link_map.get("stmt_t4") == "setl_t4"

    # Ambiguity: stmt_amb_1 and stmt_amb_2 should NOT be matched blindly
    assert "stmt_amb_1" not in link_map
    assert "stmt_amb_2" not in link_map
    amb_preds = [p for p in out.predictions if p.rule_id == "B1_BANK_AMBIGUOUS_CANDIDATES"]
    assert len(amb_preds) > 0
    for p in amb_preds:
        assert p.action == "escalate_human"


def test_b1_duplicate_payment_detector() -> None:
    """Duplicate payment flags second attempt while treating earliest as legitimate."""
    rc = _sample_rate_card()
    m_id = "mer_01"

    payments = (
        PaymentRow(payment_id="pay_01", order_id="ord_01", merchant_id=m_id, amount=50000, currency="INR", status="captured", method="card", created_at=1000, captured_at=1050, error_code=None, error_description=None),
        PaymentRow(payment_id="pay_02", order_id="ord_01", merchant_id=m_id, amount=50000, currency="INR", status="captured", method="card", created_at=2000, captured_at=2050, error_code=None, error_description=None),
    )

    dataset = ReconDataset(
        orders_ledger=(),
        payments=payments,
        recon_items=(),
        settlements=(),
        bank_statement=(),
        rate_card=rc,
        as_of="2026-06-15",
    )

    out = run_b1(dataset)
    dup_preds = [p for p in out.predictions if p.rule_id == "B1_DUPLICATE_PAYMENT"]
    assert len(dup_preds) == 1
    assert dup_preds[0].entity_ids == ("pay_02",)
    assert dup_preds[0].action == "flag"


def test_b1_fee_tax_and_near_tolerance_detector() -> None:
    """Fee & tax calculation differentiates rounding, near-tolerance escalation, and flag."""
    rc = _sample_rate_card()
    m_id = "mer_01"
    # Card bps: 200 (2.0%), GST: 1800 (18.0%), fee_includes_tax: True
    # For amount 100,000 paise (Rs 1,000):
    # base_fee = (100000 * 200 + 5000) // 10000 = 2000
    # expected_tax = (2000 * 1800 + 5000) // 10000 = 360
    # expected_fee = 2000 + 360 = 2360

    recon_items = (
        # 1. Exact match (delta 0) -> no flag
        _make_recon_item(entity_id="recon_exact", type="payment", credit=100000, amount=100000, fee=2360, tax=360, method="card", payment_id="p1", order_id="o1"),
        # 2. Benign rounding (delta 1 <= 2 paise) -> no flag
        _make_recon_item(entity_id="recon_round", type="payment", credit=100000, amount=100000, fee=2361, tax=360, method="card", payment_id="p2", order_id="o2"),
        # 3. Near tolerance (delta 5, within 3..9 paise) -> escalate_human
        _make_recon_item(entity_id="recon_near", type="payment", credit=100000, amount=100000, fee=2365, tax=360, method="card", payment_id="p3", order_id="o3"),
        # 4. Error threshold (delta 100 >= 10 paise) -> flag
        _make_recon_item(entity_id="recon_err", type="payment", credit=100000, amount=100000, fee=2460, tax=360, method="card", payment_id="p4", order_id="o4"),
    )

    dataset = ReconDataset(
        orders_ledger=(),
        payments=(),
        recon_items=recon_items,
        settlements=(),
        bank_statement=(),
        rate_card=rc,
        as_of="2026-06-15",
    )

    out = run_b1(dataset)
    pred_map = {p.entity_ids[0]: p for p in out.predictions}

    assert "recon_exact" not in pred_map
    assert "recon_round" not in pred_map
    assert pred_map["recon_near"].action == "escalate_human"
    assert pred_map["recon_near"].rule_id == "B1_NEAR_TOLERANCE"
    assert pred_map["recon_err"].action == "flag"
    assert pred_map["recon_err"].rule_id == "B1_FEE_TAX_MISMATCH"


def test_b1_partial_refund_and_cutoff_drift() -> None:
    """Test partial refund discrepancy and cutoff drift detectors."""
    rc = _sample_rate_card()
    m_id = "mer_01"

    orders = (
        OrderLedgerRow(order_id="ord_prm", merchant_id=m_id, booked_date="2026-06-01", created_at=1000, expected_amount=100000, currency="INR", order_receipt="rcpt_1", status="paid", payment_id="pay_prm", refund_id="rfnd_01", refund_amount=40000),
        OrderLedgerRow(order_id="ord_drift", merchant_id=m_id, booked_date="2026-06-01", created_at=1000, expected_amount=50000, currency="INR", order_receipt="rcpt_2", status="paid", payment_id="pay_drift", refund_id=None, refund_amount=None),
    )

    # ord_prm had refund_amount 40,000 paise; recon debit is 50,000 paise (diff 10,000)
    recon_items = (
        _make_recon_item(entity_id="rfnd_01", type="refund", debit=50000, amount=50000, payment_id="pay_prm", order_id="ord_prm"),
    )

    # ord_drift: captured on 2026-06-02 IST (timestamp 1780374000) but booked_date in ledger is 2026-06-01 -> drift
    payments = (
        PaymentRow(payment_id="pay_drift", order_id="ord_drift", merchant_id=m_id, amount=50000, currency="INR", status="captured", method="card", created_at=1780374000, captured_at=1780374000, error_code=None, error_description=None),
    )

    dataset = ReconDataset(
        orders_ledger=orders,
        payments=payments,
        recon_items=recon_items,
        settlements=(),
        bank_statement=(),
        rate_card=rc,
        as_of="2026-06-15",
    )

    out = run_b1(dataset)
    pred_map = {p.rule_id: p for p in out.predictions}

    assert "B1_PARTIAL_REFUND_MISMATCH" in pred_map
    assert pred_map["B1_PARTIAL_REFUND_MISMATCH"].entity_ids == ("rfnd_01",)

    assert "B1_CUTOFF_DRIFT" in pred_map
    assert pred_map["B1_CUTOFF_DRIFT"].entity_ids == ("pay_drift", "ord_drift")


def test_b1_flags_zero_in_flight_on_dev_seed1() -> None:
    """Dev-data check: B1 flags exactly 0 IN_FLIGHT payments and 0 IN_FLIGHT refunds."""
    from src.recon_agent.eval.scoring import load_ground_truth
    from src.recon_agent.pipeline.loader import load_dataset

    dev_dir = os.path.join(REPO_ROOT, "data", "synth", "full", "dev_seed1")
    if not os.path.isdir(dev_dir):
        pytest.skip(f"Dev dataset not found at {dev_dir}")

    dataset = load_dataset(dev_dir)
    labels, _ = load_ground_truth(dev_dir)
    out = run_b1(dataset)

    inflight_labels = [l for l in labels if l.get("hard_negative_type") == "IN_FLIGHT"]
    assert len(inflight_labels) > 0, "Non-vacuous check: IN_FLIGHT labels must exist"

    inflight_primary_ids = {
        eid for l in inflight_labels for eid in l.get("primary_entity_ids", [])
    }
    flagged_ids = {
        eid for p in out.predictions if p.action == "flag" for eid in p.entity_ids
    }

    overlap = inflight_primary_ids & flagged_ids
    assert len(overlap) == 0, f"B1 erroneously flagged IN_FLIGHT entities: {overlap}"
