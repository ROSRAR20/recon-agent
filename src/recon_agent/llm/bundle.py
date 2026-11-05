"""CaseBundle extraction and serialization for recon-agent.

Extracts structured facts and untrusted text for ambiguous cases, escalations,
and negative controls into a deterministic, frozen dataclass with canonical JSON
serialization and stable SHA-256 bundle hashing.

Strict neutrality & boundaries:
- Contains ZERO evaluation or ground-truth fields (no labels, truth links, injection manifests).
- Negative control status is recorded ONLY in results/controls_<seed>.json, never in bundles.
- All monetary values strictly in integer paise.
- All untrusted text fields tagged explicitly as UNTRUSTED.
- Neutral check_family ('fee_tax' | 'refund' | 'bank') across all bundles.
- Canonical JSON (LLM-visible) excludes rule_id and case_type to ensure residual and control
  bundles are indistinguishable by key set, key order, and neutral-field value domain.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

from src.recon_agent.pipeline.types import (
    BankStatementRow,
    PaymentRow,
    ReconDataset,
    ReconItemRow,
    SystemOutput,
)
from src.recon_agent.systems.b1r import run_b1r

ALLOWED_DECISIONS: tuple[str, ...] = ("flag", "benign_explained", "escalate_human", "no_issue")

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
    check_family: str  # "fee_tax" | "refund" | "bank"
    facts: dict[str, Any]
    untrusted_text: dict[str, str | None]
    rule_id: str | None = None  # Kept ONLY in audit/eval metadata; NOT in LLM-visible canonical JSON
    allowed_decisions: tuple[str, ...] = ALLOWED_DECISIONS
    allowed_classes: tuple[str, ...] = ALLOWED_CLASSES
    bundle_hash: str = ""

    @property
    def case_type(self) -> str:
        """Backward compatible accessor for case_type."""
        if self.check_family == "fee_tax":
            return "payment"
        elif self.check_family == "bank":
            return "bank_statement"
        return "refund"

    def to_dict(self, include_eval_metadata: bool = False) -> dict[str, Any]:
        """Convert bundle to dictionary representation."""
        data = asdict(self)
        if not include_eval_metadata:
            data.pop("rule_id", None)
            data.pop("bundle_hash", None)
        return data

    def to_canonical_json(self) -> str:
        """Produce neutral canonical JSON representation for LLM and bundle hashing.

        Excludes bundle_hash, rule_id, and case_type to ensure bundle neutrality.
        Keys are sorted lexicographically.
        """
        payload = {
            "allowed_classes": list(self.allowed_classes),
            "allowed_decisions": list(self.allowed_decisions),
            "bundle_id": self.bundle_id,
            "check_family": self.check_family,
            "entity_id": self.entity_id,
            "facts": self.facts,
            "merchant_id": self.merchant_id,
            "untrusted_text": self.untrusted_text,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_bundle_hash(bundle_canonical_json: str) -> str:
    """Compute SHA-256 hash over canonical JSON string."""
    return hashlib.sha256(bundle_canonical_json.encode("utf-8")).hexdigest()


def create_case_bundle(
    bundle_id: str,
    entity_id: str,
    merchant_id: str,
    case_type: str | None = None,
    rule_id: str | None = None,
    facts: dict[str, Any] | None = None,
    untrusted_text: dict[str, str | None] | None = None,
    check_family: str | None = None,
) -> CaseBundle:
    """Create a CaseBundle instance with automatically computed bundle_hash."""
    # Determine neutral check_family
    if check_family is None:
        if case_type in ("payment", "fee_tax"):
            check_family = "fee_tax"
        elif case_type in ("bank_statement", "bank"):
            check_family = "bank"
        else:
            check_family = "refund"

    temp_bundle = CaseBundle(
        bundle_id=bundle_id,
        entity_id=entity_id,
        merchant_id=merchant_id,
        check_family=check_family,
        rule_id=rule_id,
        facts=facts or {},
        untrusted_text=untrusted_text or {},
        allowed_decisions=ALLOWED_DECISIONS,
        allowed_classes=ALLOWED_CLASSES,
        bundle_hash="",
    )
    b_hash = compute_bundle_hash(temp_bundle.to_canonical_json())
    return CaseBundle(
        bundle_id=bundle_id,
        entity_id=entity_id,
        merchant_id=merchant_id,
        check_family=check_family,
        rule_id=rule_id,
        facts=facts or {},
        untrusted_text=untrusted_text or {},
        allowed_decisions=ALLOWED_DECISIONS,
        allowed_classes=ALLOWED_CLASSES,
        bundle_hash=b_hash,
    )


def _build_recon_item_bundle(
    item: ReconItemRow,
    rule_id: str | None,
    payments_by_id: dict[str, PaymentRow],
    settlements_by_id: dict[str, Any],
    orders_by_id: dict[str, Any],
) -> CaseBundle:
    """Helper to build a CaseBundle from a ReconItemRow with exact field ordering."""
    eid = item.entity_id
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
        facts["refund_difference_paise"] = item.debit - item.amount

    # Untrusted text
    untrusted_text = {
        "recon_item_description": item.description,
        "recon_item_notes_comment": item.notes.get("comment") if item.notes else None,
        "recon_item_notes_reason_text": item.notes.get("reason_text") if item.notes else None,
        "bank_narration": None,
    }

    check_family = "fee_tax" if item.type == "payment" else "refund"

    return create_case_bundle(
        bundle_id=f"bundle_{eid}",
        entity_id=eid,
        merchant_id=mer_id,
        check_family=check_family,
        rule_id=rule_id,
        facts=facts,
        untrusted_text=untrusted_text,
    )


def _build_bank_statement_bundle(
    bank_row: BankStatementRow,
    rule_id: str | None,
) -> CaseBundle:
    """Helper to build a CaseBundle from a BankStatementRow with exact field ordering."""
    eid = bank_row.statement_id
    mer_id = bank_row.merchant_id

    facts: dict[str, Any] = {
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

    return create_case_bundle(
        bundle_id=f"bundle_{eid}",
        entity_id=eid,
        merchant_id=mer_id,
        check_family="bank",
        rule_id=rule_id,
        facts=facts,
        untrusted_text=untrusted_text,
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

    bundles: list[CaseBundle] = []
    processed_entities: set[str] = set()

    candidate_entities: dict[str, str | None] = {}

    for eid, rule_id in escalated_entity_rules.items():
        candidate_entities[eid] = rule_id

    for r in dataset.recon_items:
        if r.type == "refund" and r.notes and r.notes.get("reason_text"):
            if r.entity_id not in candidate_entities:
                candidate_entities[r.entity_id] = "REFUND_REASON_TEXT"

    # Stable deterministic sort of entity IDs
    for eid in sorted(candidate_entities.keys()):
        if eid in processed_entities:
            continue

        rule_id = candidate_entities[eid]

        if eid in recon_by_entity:
            item = recon_by_entity[eid]
            bundle = _build_recon_item_bundle(
                item=item,
                rule_id=rule_id,
                payments_by_id=payments_by_id,
                settlements_by_id=settlements_by_id,
                orders_by_id=orders_by_id,
            )
            bundles.append(bundle)
            processed_entities.add(eid)

        elif eid in bank_by_id:
            bank_row = bank_by_id[eid]
            bundle = _build_bank_statement_bundle(
                bank_row=bank_row,
                rule_id=rule_id,
            )
            bundles.append(bundle)
            processed_entities.add(eid)

    return bundles


def build_control_bundles(
    dataset: ReconDataset,
    system_output: SystemOutput,
    n_per_seed: int = 15,
    rng_seed: int = 42,
    results_dir: str = "results",
) -> list[CaseBundle]:
    """Sample clean, un-flagged entities symmetric to the ambiguous families as negative controls.

    Identifies:
    - fee_tax: payments where fee and tax match expected values within tolerance
    - refunds: refunds where debit == amount (refund_difference == 0)
    - bank_matches: bank statement items with exact 1:1 match to settlement
    """
    flagged_or_escalated = set()
    for pred in system_output.predictions:
        if pred.action in ("flag", "escalate_human"):
            flagged_or_escalated.update(pred.entity_ids)

    payments_by_id = {p.payment_id: p for p in dataset.payments}
    settlements_by_id = {s.settlement_id: s for s in dataset.settlements}
    orders_by_id = {o.order_id: o for o in dataset.orders_ledger}

    # 1. Family: fee_tax (clean payments)
    fee_tax_candidates: list[ReconItemRow] = [
        r for r in dataset.recon_items
        if r.type == "payment" and r.entity_id not in flagged_or_escalated
    ]

    # 2. Family: refunds (clean refunds, debit == amount)
    refund_candidates: list[ReconItemRow] = [
        r for r in dataset.recon_items
        if r.type == "refund" and r.entity_id not in flagged_or_escalated and r.debit == r.amount
    ]

    # 3. Family: bank_matches (clean bank rows matched to settlements)
    matched_bank_ids = {link.statement_id for link in system_output.bank_links}
    bank_candidates: list[BankStatementRow] = [
        b for b in dataset.bank_statement
        if b.statement_id in matched_bank_ids and b.statement_id not in flagged_or_escalated
    ]

    rng = random.Random(rng_seed)
    # Deterministic sort before shuffle
    fee_tax_candidates.sort(key=lambda r: r.entity_id)
    refund_candidates.sort(key=lambda r: r.entity_id)
    bank_candidates.sort(key=lambda b: b.statement_id)

    rng.shuffle(fee_tax_candidates)
    rng.shuffle(refund_candidates)
    rng.shuffle(bank_candidates)

    n_per_family = max(1, n_per_seed // 3)
    selected_fee = fee_tax_candidates[:n_per_family]
    selected_refund = refund_candidates[:n_per_family]
    selected_bank = bank_candidates[:n_per_family]

    bundles: list[CaseBundle] = []
    control_records: dict[str, dict[str, str]] = {}

    for item in selected_fee:
        b = _build_recon_item_bundle(
            item=item,
            rule_id="B1_ROUNDING_ACCEPT",
            payments_by_id=payments_by_id,
            settlements_by_id=settlements_by_id,
            orders_by_id=orders_by_id,
        )
        bundles.append(b)
        control_records[b.entity_id] = {
            "family": "fee_tax",
            "is_control": "true",
            "case_type": "payment",
        }

    for item in selected_refund:
        b = _build_recon_item_bundle(
            item=item,
            rule_id="PARTIAL_REFUND_CONSISTENT",
            payments_by_id=payments_by_id,
            settlements_by_id=settlements_by_id,
            orders_by_id=orders_by_id,
        )
        bundles.append(b)
        control_records[b.entity_id] = {
            "family": "refunds",
            "is_control": "true",
            "case_type": "refund",
        }

    for bank_row in selected_bank:
        b = _build_bank_statement_bundle(
            bank_row=bank_row,
            rule_id="B1_BANK_T1_EXACT",
        )
        bundles.append(b)
        control_records[b.entity_id] = {
            "family": "bank_matches",
            "is_control": "true",
            "case_type": "bank_statement",
        }

    # Save to results/controls_<seed>.json (eval-only tracking)
    os.makedirs(results_dir, exist_ok=True)
    controls_file = os.path.join(results_dir, f"controls_{rng_seed}.json")
    out_payload = {
        "seed": rng_seed,
        "n_requested": n_per_seed,
        "counts": {
            "fee_tax": len(selected_fee),
            "refunds": len(selected_refund),
            "bank_matches": len(selected_bank),
            "total": len(bundles),
        },
        "controls": control_records,
    }
    with open(controls_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    return bundles
