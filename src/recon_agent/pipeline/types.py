"""Immutable data models and prediction types for recon-agent pipeline."""

from dataclasses import dataclass
from typing import Any

ALLOWED_PREDICTED_CLASSES = frozenset({
    "DUPLICATE_PAYMENT",
    "MISSING_SETTLEMENT",
    "FEE_TAX_MISMATCH",
    "PARTIAL_REFUND_MISMATCH",
    "CUTOFF_DRIFT",
    "SHORT_BANK_CREDIT",
    "ORPHAN_BANK_CREDIT",
    "UNKNOWN",
})

ALLOWED_ACTIONS = frozenset({
    "flag",
    "escalate_human",
    "benign_explained",
    "ignore",
})


@dataclass(frozen=True)
class Prediction:
    """Standardized prediction emitted by a reconciliation system."""

    merchant_id: str
    predicted_class: str
    action: str
    entity_ids: tuple[str, ...]
    settlement_id: str | None = None
    rule_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.merchant_id, str):
            raise TypeError(f"merchant_id must be str, got {type(self.merchant_id)}")
        if self.predicted_class not in ALLOWED_PREDICTED_CLASSES:
            raise ValueError(
                f"predicted_class '{self.predicted_class}' not in allowed classes: {sorted(ALLOWED_PREDICTED_CLASSES)}"
            )
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"action '{self.action}' not in allowed actions: {sorted(ALLOWED_ACTIONS)}"
            )
        if not isinstance(self.entity_ids, tuple):
            # Normalization helper for tuple conversion
            object.__setattr__(self, "entity_ids", tuple(self.entity_ids))
        for eid in self.entity_ids:
            if not isinstance(eid, str):
                raise TypeError(f"All entity_ids must be str, got {type(eid)}")
        if self.settlement_id is not None and not isinstance(self.settlement_id, str):
            raise TypeError(f"settlement_id must be str or None, got {type(self.settlement_id)}")
        if not isinstance(self.rule_id, str):
            raise TypeError(f"rule_id must be str, got {type(self.rule_id)}")


@dataclass(frozen=True)
class BankSettlementLink:
    """Link matching a bank statement credit row to a settlement batch."""

    statement_id: str
    settlement_id: str


@dataclass(frozen=True)
class SystemOutput:
    """Standardized output collection from a reconciliation system."""

    predictions: tuple[Prediction, ...]
    bank_links: tuple[BankSettlementLink, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.predictions, tuple):
            object.__setattr__(self, "predictions", tuple(self.predictions))
        if not isinstance(self.bank_links, tuple):
            object.__setattr__(self, "bank_links", tuple(self.bank_links))


# =============================================================================
# Observable Dataset Types (frozen, integers only)
# =============================================================================


@dataclass(frozen=True)
class OrderLedgerRow:
    """Internal merchant order ledger entry."""

    order_id: str
    merchant_id: str
    booked_date: str
    created_at: int
    expected_amount: int
    currency: str
    order_receipt: str
    status: str
    payment_id: str | None
    refund_id: str | None
    refund_amount: int | None


@dataclass(frozen=True)
class PaymentRow:
    """Payment gateway transaction attempt record."""

    payment_id: str
    order_id: str
    merchant_id: str
    amount: int
    currency: str
    status: str
    method: str
    created_at: int
    captured_at: int | None
    error_code: str | None
    error_description: str | None


@dataclass(frozen=True)
class ReconItemRow:
    """Reconciliation item returned by payment gateway recon endpoint."""

    entity_id: str
    type: str
    debit: int
    credit: int
    amount: int
    currency: str
    fee: int
    tax: int
    on_hold: bool
    settled: bool
    created_at: int
    settled_at: int | None
    settlement_id: str | None
    description: str | None
    notes: dict[str, Any] | None
    payment_id: str | None
    settlement_utr: str | None
    order_id: str | None
    order_receipt: str | None
    method: str | None
    card_network: str | None
    card_issuer: str | None
    card_type: str | None
    dispute_id: str | None
    merchant_id: str


@dataclass(frozen=True)
class SettlementRow:
    """Lump-sum gateway payout settlement batch record."""

    settlement_id: str
    merchant_id: str
    amount: int
    utr: str
    status: str
    created_at: int
    settled_at: int


@dataclass(frozen=True)
class BankStatementRow:
    """Merchant bank statement transaction line."""

    statement_id: str
    merchant_id: str
    date: str
    description: str
    debit: int
    credit: int
    balance: int


@dataclass(frozen=True)
class RateCard:
    """Merchant fee schedule and reconciliation policy parameters."""

    card_bps: int
    upi_bps: int
    netbanking_bps: int
    gst_bps: int
    fee_includes_tax: bool
    payment_settlement_lag_days: int
    refund_settlement_lag_days: int
    missing_after_days: int
    rounding_tolerance_paise: int
    escalate_tolerance_paise: int
    error_threshold_paise: int


@dataclass(frozen=True)
class ReconDataset:
    """Collection of all observable reconciliation inputs for a merchant cohort."""

    orders_ledger: tuple[OrderLedgerRow, ...]
    payments: tuple[PaymentRow, ...]
    recon_items: tuple[ReconItemRow, ...]
    settlements: tuple[SettlementRow, ...]
    bank_statement: tuple[BankStatementRow, ...]
    rate_card: RateCard
    as_of: str

