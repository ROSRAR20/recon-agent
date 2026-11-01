"""System B1r: Rule Engine with Generic Regex and Number-Word Reason Parser.

Extends System B1 by parsing textual explanations in `notes.reason_text` on
reconciliation refund items. Categorizes partial refund discrepancies into:
- B1R_EXPLAINED_MATCH (benign_explained): parsed amount matches the discrepancy.
- B1R_EXPLAINED_MISMATCH (flag): parsed amount contradicts the discrepancy.
- B1R_EXPLAINED_NO_AMOUNT (escalate_human): text exists but has no parseable amount.

Residual cases that cannot be resolved deterministically are recorded for
downstream escalation to human or text models.
"""

import re
from typing import Any

from src.recon_agent.pipeline.types import (
    Prediction,
    ReconDataset,
    ResidualCase,
    SystemOutput,
)
from src.recon_agent.systems.b1 import run_b1

SMALL_NUMBERS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def parse_words_to_int(words_str: str) -> int | None:
    """Parse number words up to 99,999 into an integer."""
    tokens = words_str.lower().replace("-", " ").split()
    total = 0
    current = 0
    for tok in tokens:
        if tok in SMALL_NUMBERS:
            current += SMALL_NUMBERS[tok]
        elif tok == "hundred":
            current = (current if current else 1) * 100
        elif tok == "thousand":
            total += (current if current else 1) * 1000
            current = 0
        elif tok in ("and",):
            continue
        else:
            return None
    return total + current


def parse_reason_amount(text: str) -> int | None:
    """Parse monetary amount in paise from reason text.

    Supports:
    - Currency prefix (Rs, Rs., INR, ₹) followed by numbers with optional commas and decimals.
    - Explicit paise values (e.g., '50 paise', '100 paise').
    - English number-words for rupees and paise (e.g. 'fifty rupees and twenty paise').

    Returns amount in integer paise, or None if no amount could be extracted.
    """
    if not text:
        return None

    # 1. Currency prefix followed by numeric value: Rs 100, Rs. 15.50, INR 50, ₹ 20.00
    m_curr = re.search(
        r"(?:\bRs\.?|\bINR|\u20b9)\s*(\d+(?:,\d+)*(?:\.\d+)?|\.\d+)", text, re.IGNORECASE
    )
    if m_curr:
        val_str = m_curr.group(1).replace(",", "")
        if "." in val_str:
            parts = val_str.split(".")
            rupees = int(parts[0]) if parts[0] else 0
            paise_str = (parts[1] + "00")[:2]
            return rupees * 100 + int(paise_str)
        else:
            return int(val_str) * 100

    # 2. Explicit numeric paise: e.g. "50 paise", "75 paise"
    m_paise = re.search(r"(\d+)\s*\bpaise\b", text, re.IGNORECASE)
    if m_paise:
        return int(m_paise.group(1))

    # 3. Number words with rupees and optional paise: e.g. "fifty rupees and twenty paise"
    m_words = re.search(
        r"([a-z\s-]+?)\s*(?:\brupees?|\brs|\binr|\u20b9)\b(?:\s+and\s+([a-z\s-]+?)\s*\bpaise\b)?",
        text,
        re.IGNORECASE,
    )
    if m_words:
        raw_words = m_words.group(1).replace("-", " ").split()
        num_toks: list[str] = []
        for w in reversed(raw_words):
            if w in SMALL_NUMBERS or w in ("hundred", "thousand", "and"):
                num_toks.insert(0, w)
            else:
                break
        if num_toks:
            rup_val = parse_words_to_int(" ".join(num_toks))
            if rup_val is not None:
                total = rup_val * 100
                if m_words.group(2):
                    pai_val = parse_words_to_int(m_words.group(2).strip())
                    if pai_val is not None:
                        total += pai_val
                return total

    # 4. Number words with paise only: e.g. "fifty paise"
    m_words_pai = re.search(r"([a-z\s-]+?)\s*\bpaise\b", text, re.IGNORECASE)
    if m_words_pai:
        raw_words = m_words_pai.group(1).replace("-", " ").split()
        num_toks = []
        for w in reversed(raw_words):
            if w in SMALL_NUMBERS or w in ("hundred", "thousand", "and"):
                num_toks.insert(0, w)
            else:
                break
        if num_toks:
            pai_val = parse_words_to_int(" ".join(num_toks))
            if pai_val is not None:
                return pai_val

    return None


def run_b1r(dataset: ReconDataset) -> SystemOutput:
    """Execute System B1r on observable reconciliation dataset.

    Inherits all B1 matches and detections, then evaluates refund items with
    `notes.reason_text` using regex and number-word parsing.
    """
    b1_output = run_b1(dataset)
    rc = dataset.rate_card

    # Index recon_items by entity_id and orders by order_id
    recon_map = {r.entity_id: r for r in dataset.recon_items}
    orders_map = {o.order_id: o for o in dataset.orders_ledger}

    new_predictions: list[Prediction] = []
    new_residuals: list[ResidualCase] = list(b1_output.residuals)

    for pred in b1_output.predictions:
        if pred.rule_id == "B1_PARTIAL_REFUND_MISMATCH":
            entity_id = pred.entity_ids[0]
            recon_item = recon_map.get(entity_id)
            notes = recon_item.notes if (recon_item and recon_item.notes) else {}
            reason_text = notes.get("reason_text")

            if reason_text is not None:
                order = orders_map.get(recon_item.order_id) if recon_item else None
                order_rfnd = order.refund_amount if (order and order.refund_amount is not None) else 0
                actual_diff = abs(recon_item.debit - order_rfnd) if recon_item else 0

                parsed_amt = parse_reason_amount(reason_text)

                if parsed_amt is not None:
                    if abs(parsed_amt - actual_diff) <= rc.rounding_tolerance_paise:
                        # Stated amount explains the discrepancy
                        new_predictions.append(
                            Prediction(
                                merchant_id=pred.merchant_id,
                                predicted_class="PARTIAL_REFUND_MISMATCH",
                                action="benign_explained",
                                entity_ids=pred.entity_ids,
                                settlement_id=pred.settlement_id,
                                rule_id="B1R_EXPLAINED_MATCH",
                            )
                        )
                    else:
                        # Stated amount contradicts actual difference -> flag
                        new_predictions.append(
                            Prediction(
                                merchant_id=pred.merchant_id,
                                predicted_class="PARTIAL_REFUND_MISMATCH",
                                action="flag",
                                entity_ids=pred.entity_ids,
                                settlement_id=pred.settlement_id,
                                rule_id="B1R_EXPLAINED_MISMATCH",
                            )
                        )
                else:
                    # Reason text present but no parseable amount -> escalate to human
                    new_predictions.append(
                        Prediction(
                            merchant_id=pred.merchant_id,
                            predicted_class="PARTIAL_REFUND_MISMATCH",
                            action="escalate_human",
                            entity_ids=pred.entity_ids,
                            settlement_id=pred.settlement_id,
                            rule_id="B1R_EXPLAINED_NO_AMOUNT",
                        )
                    )
                    new_residuals.append(
                        ResidualCase(
                            merchant_id=pred.merchant_id,
                            entity_ids=pred.entity_ids,
                            rule_id="B1R_EXPLAINED_NO_AMOUNT",
                            reason=f"Refund discrepancy of {actual_diff} paise has explanation text without parseable amount: {reason_text}",
                        )
                    )
            else:
                # No reason_text: unannotated partial refund mismatch
                new_predictions.append(pred)
        else:
            new_predictions.append(pred)

    return SystemOutput(
        predictions=tuple(new_predictions),
        bank_links=b1_output.bank_links,
        residuals=tuple(new_residuals),
    )
