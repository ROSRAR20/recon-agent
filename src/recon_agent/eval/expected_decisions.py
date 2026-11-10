"""Ground truth mapping to expected reconciliation decisions and classes.

Strictly reserved for evaluation infrastructure. Pipeline and LLM runner code
must never import or access this module.

Deterministic mapping rules:
1. Controls and benign/hard-negative items -> no_issue
2. Bank bundles B1 escalated for ambiguous candidates -> escalate_human
3. Ambiguity types:
   - NEAR_TOLERANCE -> escalate_human
   - EXPLAINED_NO_AMOUNT -> escalate_human
   - EXPLAINED_CONSISTENT -> benign_explained
   - EXPLAINED_INCONSISTENT -> flag + PARTIAL_REFUND_MISMATCH
4. Tier-1 error classes -> flag + matching error class
5. Any unmapped case raises ValueError (never silently defaults).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.recon_agent.eval.scoring import (
    HARD_NEGATIVE_TYPES,
    TIER1_ERROR_CLASSES,
)
from src.recon_agent.llm.bundle import CaseBundle


@dataclass(frozen=True)
class ExpectedDecision:
    """Expected evaluation decision, optional predicted class, and source label type."""

    decision: str
    predicted_class: str | None
    source_label_type: str


def get_expected_decision(
    bundle: CaseBundle,
    label: dict[str, Any] | None = None,
    is_control: bool = False,
) -> ExpectedDecision:
    """Map a bundle and its ground-truth label to the pre-registered expected decision.

    Raises:
        ValueError: If the combination of bundle and label cannot be deterministically mapped.
    """
    # 1. Controls are clean/negative controls by definition -> no_issue
    if is_control:
        return ExpectedDecision(
            decision="no_issue",
            predicted_class=None,
            source_label_type="control",
        )

    # 2. Bank statement bundles escalated by B1 for ambiguous candidates -> escalate_human
    is_ambiguous_bank = (
        bundle.check_family == "bank"
        and (
            bundle.facts.get("ambiguous_candidates") is True
            or bundle.rule_id == "B1_BANK_AMBIGUOUS_CANDIDATES"
        )
    )
    if is_ambiguous_bank:
        return ExpectedDecision(
            decision="escalate_human",
            predicted_class=None,
            source_label_type="bank_ambiguous_candidates",
        )

    if label is None:
        raise ValueError(
            f"Ground-truth label is required for non-control, non-ambiguous-bank bundle '{bundle.bundle_id}'."
        )

    # 3. Ambiguity tier items
    ambig_type = label.get("ambiguity_type")
    if ambig_type:
        if ambig_type == "NEAR_TOLERANCE":
            return ExpectedDecision(
                decision="escalate_human",
                predicted_class=None,
                source_label_type="NEAR_TOLERANCE",
            )
        elif ambig_type == "EXPLAINED_NO_AMOUNT":
            return ExpectedDecision(
                decision="escalate_human",
                predicted_class=None,
                source_label_type="EXPLAINED_NO_AMOUNT",
            )
        elif ambig_type == "EXPLAINED_CONSISTENT":
            return ExpectedDecision(
                decision="benign_explained",
                predicted_class=None,
                source_label_type="EXPLAINED_CONSISTENT",
            )
        elif ambig_type == "EXPLAINED_INCONSISTENT":
            return ExpectedDecision(
                decision="flag",
                predicted_class="PARTIAL_REFUND_MISMATCH",
                source_label_type="EXPLAINED_INCONSISTENT",
            )
        else:
            raise ValueError(f"Unmapped ambiguity_type: '{ambig_type}' for bundle '{bundle.bundle_id}'.")

    # 4. Hard negatives and benign items -> no_issue
    hard_neg_type = label.get("hard_negative_type")
    if hard_neg_type:
        if hard_neg_type in HARD_NEGATIVE_TYPES:
            return ExpectedDecision(
                decision="no_issue",
                predicted_class=None,
                source_label_type=f"hard_negative:{hard_neg_type}",
            )
        else:
            raise ValueError(
                f"Unmapped hard_negative_type: '{hard_neg_type}' for bundle '{bundle.bundle_id}'."
            )

    expected_action = label.get("expected_action")
    if expected_action == "benign_ignore":
        return ExpectedDecision(
            decision="no_issue",
            predicted_class=None,
            source_label_type="benign_ignore",
        )

    # 5. Tier-1 error classes -> flag + matching class
    error_class = label.get("error_class")
    if error_class:
        if error_class in TIER1_ERROR_CLASSES:
            return ExpectedDecision(
                decision="flag",
                predicted_class=error_class,
                source_label_type=f"tier1:{error_class}",
            )
        else:
            raise ValueError(
                f"Unmapped error_class: '{error_class}' for bundle '{bundle.bundle_id}'."
            )

    # 6. Unmapped case
    raise ValueError(
        f"Unable to map expected decision for bundle '{bundle.bundle_id}' with label {label}."
    )
