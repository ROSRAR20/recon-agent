"""System B0: Baseline Exact-ID reconciliation engine.

Performs exact identifier matching only:
- payments <-> recon_items by payment_id / entity_id.
- recon_items <-> settlements by settlement_id.
- settlements <-> bank_statement by the full UTR as an exact substring of the bank narration (no normalization).

Unmatched entity rules:
- Captured payment with no recon item -> MISSING_SETTLEMENT.
- Razorpay-style bank credit matching no settlement -> ORPHAN_BANK_CREDIT.
- Settlement batch with no bank statement match -> UNKNOWN.

Zero arithmetic checks, ledger checks, or tolerance logic.
"""

from src.recon_agent.pipeline.types import (
    BankSettlementLink,
    Prediction,
    ReconDataset,
    SystemOutput,
)


def run_b0(dataset: ReconDataset) -> SystemOutput:
    """Execute System B0 exact-ID baseline on observable reconciliation dataset."""
    predictions: list[Prediction] = []
    bank_links: list[BankSettlementLink] = []

    # 1. Index recon payment items by payment_id
    recon_payment_ids: set[str] = set()
    for r in dataset.recon_items:
        if r.type == "payment":
            if r.payment_id:
                recon_payment_ids.add(r.payment_id)
            recon_payment_ids.add(r.entity_id)

    # 2. Check captured payments <-> recon items
    for p in dataset.payments:
        if p.status == "captured":
            if p.payment_id not in recon_payment_ids:
                predictions.append(
                    Prediction(
                        merchant_id=p.merchant_id,
                        predicted_class="MISSING_SETTLEMENT",
                        action="flag",
                        entity_ids=(p.payment_id,),
                        rule_id="B0_MISSING_SETTLEMENT",
                    )
                )

    # 3. Match settlements <-> bank statement by exact UTR substring (no normalization)
    matched_statement_ids: set[str] = set()
    for s in dataset.settlements:
        matched_stmt_id: str | None = None
        for b in dataset.bank_statement:
            if b.merchant_id == s.merchant_id and b.credit > 0 and b.statement_id not in matched_statement_ids:
                # Full UTR as an exact substring of the narration (case-sensitive, no normalization)
                if s.utr in b.description:
                    matched_stmt_id = b.statement_id
                    break

        if matched_stmt_id is not None:
            matched_statement_ids.add(matched_stmt_id)
            bank_links.append(
                BankSettlementLink(
                    statement_id=matched_stmt_id,
                    settlement_id=s.settlement_id,
                )
            )
        else:
            # Settlement with no bank match -> UNKNOWN
            predictions.append(
                Prediction(
                    merchant_id=s.merchant_id,
                    predicted_class="UNKNOWN",
                    action="flag",
                    entity_ids=(s.settlement_id,),
                    settlement_id=s.settlement_id,
                    rule_id="B0_UNMATCHED_SETTLEMENT",
                )
            )

    # 4. Razorpay-style bank credit matching no settlement -> ORPHAN_BANK_CREDIT
    for b in dataset.bank_statement:
        if b.credit > 0 and "razorpay" in b.description.lower():
            if b.statement_id not in matched_statement_ids:
                predictions.append(
                    Prediction(
                        merchant_id=b.merchant_id,
                        predicted_class="ORPHAN_BANK_CREDIT",
                        action="flag",
                        entity_ids=(b.statement_id,),
                        rule_id="B0_ORPHAN_BANK_CREDIT",
                    )
                )

    return SystemOutput(
        predictions=tuple(predictions),
        bank_links=tuple(bank_links),
    )

