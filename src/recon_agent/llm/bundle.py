"""CaseBundle extraction and serialization for recon-agent.

Extracts structured facts and untrusted text for ambiguous cases and escalations
into a deterministic, frozen dataclass with canonical JSON serialization and
stable SHA-256 bundle hashing.

Strict boundaries:
- Contains ZERO evaluation or ground-truth fields (no labels, truth links, injection manifests).
- All monetary values strictly in integer paise.
- All untrusted text fields tagged explicitly as UNTRUSTED.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from src.recon_agent.pipeline.types import ReconDataset, ReconItemRow
from src.recon_agent.systems.b1r import run_b1r

ALLOWED_DECISIONS: tuple[str, ...] = ("flag", "benign_explained", "escalate_human")

ALLOWED_CLASSES: tuple[str, ...] = (
    "DUPLICATE_PAYMENT",
    "MISSING_SETTLEMENT",
    "FEE_TAX_MISMATCH",
    "PARTIAL_REFUND_MISMATCH",
    "CUTOFF_DRIFT",
    "SHORT_BANK_CREDIT",
    "ORPHAN_BANK_CREDIT",
    "UNKNOWN",
)


@dataclass(frozen=True)
class CaseBundle:
    """Frozen case bundle containing observable facts and untrusted text for an entity."""

    bundle_id: str
    entity_id: str
    merchant_id: str
    case_type: str
    rule_id: str | None
    facts: dict[str, Any]
    untrusted_text: dict[str, str | None]
    allowed_decisions: tuple[str, ...] = ALLOWED_DECISIONS
    allowed_classes: tuple[str, ...] = ALLOWED_CLASSES
    bundle_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert bundle to dictionary representation."""
        data = asdict(self)
        return data

    def to_canonical_json(self) -> str:
        """Produce canonical JSON representation excluding bundle_hash."""
        data = self.to_dict()
        data.pop("bundle_hash", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_bundle_hash(bundle_canonical_json: str) -> str:
    """Compute SHA-256 hash over canonical JSON string."""
    return hashlib.sha256(bundle_canonical_json.encode("utf-8")).hexdigest()


def create_case_bundle(
    bundle_id: str,
    entity_id: str,
    merchant_id: str,
    case_type: str,
    rule_id: str | None,
    facts: dict[str, Any],
    untrusted_text: dict[str, str | None],
) -> CaseBundle:
    """Create a CaseBundle instance with automatically computed bundle_hash."""
    temp_bundle = CaseBundle(
        bundle_id=bundle_id,
        entity_id=entity_id,
        merchant_id=merchant_id,
        case_type=case_type,
        rule_id=rule_id,
        facts=facts,
        untrusted_text=untrusted_text,
        allowed_decisions=ALLOWED_DECISIONS,
        allowed_classes=ALLOWED_CLASSES,
        bundle_hash="",
    )
    b_hash = compute_bundle_hash(temp_bundle.to_canonical_json())
    return CaseBundle(
        bundle_id=bundle_id,
        entity_id=entity_id,
        merchant_id=merchant_id,
        case_type=case_type,
        rule_id=rule_id,
        facts=facts,
        untrusted_text=untrusted_text,
        allowed_decisions=ALLOWED_DECISIONS,
        allowed_classes=ALLOWED_CLASSES,
        bundle_hash=b_hash,
    )


def build_case_bundles(dataset: ReconDataset) -> list[CaseBundle]:
    """Build case bundles for all B1r escalations and all refunds with notes.reason_text.

    Never accesses evaluation files or ground truth.
    """
    # 1. Run B1r on the observable dataset to identify escalations
    b1r_output = run_b1r(dataset)
    escalations = [p for p in b1r_output.predictions if p.action == "escalate_human"]

    escalated_entity_rules: dict[str, str] = {}
    for p in escalations:
        for eid in p.entity_ids:
            escalated_entity_rules[eid] = p.rule_id or "B1R_ESCALATION"

    # Index dataset tables for fast lookup
    recon_by_entity: dict[str, ReconItemRow] = {r.entity_id: r for r in dataset.recon_items}
    payments_by_id = {p.payment_id: p for p in dataset.payments}
    settlements_by_id = {s.settlement_id: s for s in dataset.settlements}
    orders_by_id = {o.order_id: o for o in dataset.orders_ledger}
    bank_by_id = {b.statement_id: b for b in dataset.bank_statement}

    # Settlements by UTR for bank matching
    settlements_by_utr = {s.utr: s for s in dataset.settlements if s.utr}

    bundles: list[CaseBundle] = []
    processed_entities: set[str] = set()

    # Entities to bundle: all escalated entities + all refunds with reason_text
    candidate_entities: dict[str, str | None] = {}

    for eid, rule_id in escalated_entity_rules.items():
        candidate_entities[eid] = rule_id

    for r in dataset.recon_items:
        if r.type == "refund" and r.notes and r.notes.get("reason_text"):
            if r.entity_id not in candidate_entities:
                candidate_entities[r.entity_id] = "REFUND_REASON_TEXT"

    for eid, rule_id in sorted(candidate_entities.items()):
        if eid in processed_entities:
            continue
        processed_entities.add(eid)

        # Case A: entity is a recon_item
        if eid in recon_by_entity:
            item = recon_by_entity[eid]
            mer_id = item.merchant_id

            facts: dict[str, Any] = {
                "entity_id": eid,
                "type": item.type,
                "amount_paise": item.amount,
                "fee_paise": item.fee,
                "tax_paise": item.tax,
                "debit_paise": item.debit,
                "credit_paise": item.credit,
                "settled": item.settled,
                "created_at": item.created_at,
                "settled_at": item.settled_at,
                "settlement_id": item.settlement_id,
                "payment_id": item.payment_id,
            }

            # Link payment facts
            pay = payments_by_id.get(item.payment_id) if item.payment_id else None
            if pay:
                facts["payment_amount_paise"] = pay.amount
                facts["payment_method"] = pay.method
                facts["payment_captured_at"] = pay.captured_at
                facts["order_id"] = pay.order_id

                # Link order facts
                ord_row = orders_by_id.get(pay.order_id) if pay.order_id else None
                if ord_row:
                    facts["order_booked_date"] = ord_row.booked_date
                    facts["order_amount_paise"] = ord_row.expected_amount

            # Link settlement facts
            setl = settlements_by_id.get(item.settlement_id) if item.settlement_id else None
            if setl:
                facts["settlement_amount_paise"] = setl.amount
                facts["settlement_utr"] = setl.utr
                facts["settlement_settled_at"] = setl.settled_at

            # For refund items, compute refund_difference_paise
            if item.type == "refund":
                # Expected debit is payment amount or original refund amount
                # In recon items, debit is what was actually settled
                # If difference exists:
                facts["refund_difference_paise"] = item.debit - item.amount

            # Untrusted text
            untrusted_text = {
                "recon_item_description": item.description,
                "recon_item_notes_comment": item.notes.get("comment") if item.notes else None,
                "recon_item_notes_reason_text": item.notes.get("reason_text") if item.notes else None,
                "bank_narration": None,
            }

            bundle = create_case_bundle(
                bundle_id=f"bundle_{eid}",
                entity_id=eid,
                merchant_id=mer_id,
                case_type=item.type,
                rule_id=rule_id,
                facts=facts,
                untrusted_text=untrusted_text,
            )
            bundles.append(bundle)

        # Case B: entity is a bank_statement row
        elif eid in bank_by_id:
            bank_row = bank_by_id[eid]
            mer_id = bank_row.merchant_id

            facts = {
                "statement_id": bank_row.statement_id,
                "bank_credit_paise": bank_row.credit,
                "bank_date": bank_row.date,
            }

            untrusted_text = {
                "recon_item_description": None,
                "recon_item_notes_comment": None,
                "recon_item_notes_reason_text": None,
                "bank_narration": bank_row.description,
            }

            bundle = create_case_bundle(
                bundle_id=f"bundle_{eid}",
                entity_id=eid,
                merchant_id=mer_id,
                case_type="bank_statement",
                rule_id=rule_id,
                facts=facts,
                untrusted_text=untrusted_text,
            )
            bundles.append(bundle)

        # Case C: entity is a payment
        elif eid in payments_by_id:
            pay = payments_by_id[eid]
            facts = {
                "payment_id": pay.payment_id,
                "payment_amount_paise": pay.amount,
                "payment_method": pay.method,
                "payment_captured_at": pay.captured_at,
                "order_id": pay.order_id,
            }
            bundle = create_case_bundle(
                bundle_id=f"bundle_{eid}",
                entity_id=eid,
                merchant_id=pay.merchant_id,
                case_type="payment",
                rule_id=rule_id,
                facts=facts,
                untrusted_text={
                    "recon_item_description": None,
                    "recon_item_notes_comment": None,
                    "recon_item_notes_reason_text": None,
                    "bank_narration": None,
                },
            )
            bundles.append(bundle)

    return bundles
