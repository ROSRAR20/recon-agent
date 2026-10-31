"""Scoring harness and evaluation contract for recon-agent systems.

Computes one-to-one greedy assignment between predictions and ground-truth labels,
evaluates Tier 1 error classes (class-correct and class-agnostic with Wilson 95% CIs),
hard-negative false-positive rates, ambiguity tier action accuracy,
bank-to-settlement match rates, clean-data false positives, and confusion matrix.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any

from src.recon_agent.pipeline.types import BankSettlementLink, Prediction, SystemOutput

TIER1_ERROR_CLASSES = (
    "DUPLICATE_PAYMENT",
    "MISSING_SETTLEMENT",
    "FEE_TAX_MISMATCH",
    "PARTIAL_REFUND_MISMATCH",
    "CUTOFF_DRIFT",
    "SHORT_BANK_CREDIT",
    "ORPHAN_BANK_CREDIT",
)

HARD_NEGATIVE_TYPES = (
    "ROUNDING",
    "LEGIT_ADJUSTMENT",
    "LATE_SETTLEMENT",
    "RETRY_AFTER_FAIL",
    "NOISY_BANK_NARRATION",
    "NO_UTR_NARRATION",
    "UNRELATED_BANK_CREDIT",
    "FAILED_PAYMENT",
    "PARTIAL_REFUND_CONSISTENT",
    "IN_FLIGHT",
)

AMBIGUITY_TYPES = (
    "NEAR_TOLERANCE",
    "EXPLAINED_CONSISTENT",
    "EXPLAINED_INCONSISTENT",
    "EXPLAINED_NO_AMOUNT",
)


def load_ground_truth(dataset_dir: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load evaluation ground-truth files (labels.jsonl and truth_links.jsonl) from dataset directory.

    Strictly reserved for eval infrastructure; pipeline and systems code must never call this.
    """
    import json
    import os

    labels_path = os.path.join(dataset_dir, "labels.jsonl")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing evaluation ground-truth file: {labels_path}")
    labels: list[dict[str, Any]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                labels.append(json.loads(line_str))

    truth_path = os.path.join(dataset_dir, "truth_links.jsonl")
    if not os.path.isfile(truth_path):
        raise FileNotFoundError(f"Missing evaluation truth links file: {truth_path}")
    truth_links: list[dict[str, Any]] = []
    with open(truth_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                truth_links.append(json.loads(line_str))

    return labels, truth_links


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate the Wilson score 95% confidence interval for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% normal quantile
    denominator = n + z * z
    center = (k + 0.5 * z * z) / denominator
    margin = (z * math.sqrt((k * (n - k) / n) + (z * z / 4.0))) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


@dataclass(frozen=True)
class Tier1Metric:
    """Evaluation metrics for a Tier 1 error class."""

    tp: int
    fp: int
    fn: int
    precision: float | None
    precision_ci: tuple[float, float] | None
    precision_kn: str
    recall: float | None
    recall_ci: tuple[float, float] | None
    recall_kn: str
    f1: float | None


@dataclass(frozen=True)
class HardNegativeMetric:
    """Evaluation metrics for a hard-negative category."""

    flagged: int
    total: int
    fpr: float
    fpr_ci: tuple[float, float]
    fpr_kn: str


@dataclass(frozen=True)
class AmbiguityMetric:
    """Evaluation metrics for an ambiguity category."""

    correct: int
    total: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    accuracy_kn: str


@dataclass(frozen=True)
class MatchingMetric:
    """Evaluation metrics for bank-to-settlement matching."""

    links_made: int
    correct_links: int
    wrong_links: int
    false_match_rate: float | None
    false_match_ci: tuple[float, float] | None
    match_rate: float | None
    match_rate_ci: tuple[float, float] | None


@dataclass
class ScoringResult:
    """Complete structured scoring results for a system run."""

    system_id: str
    split: str
    seed: int
    tier1_class_correct: dict[str, Tier1Metric]
    tier1_class_agnostic: dict[str, Tier1Metric]
    hard_negatives: dict[str, HardNegativeMetric]
    ambiguity_tier: dict[str, AmbiguityMetric]
    matching: MatchingMetric
    clean_false_positives: int
    total_flagged_predictions: int
    total_labels: int
    confusion_matrix: dict[str, dict[str, int]]
    summary_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result object into a JSON-serializable dictionary."""
        data = asdict(self)
        return data


def score_predictions(
    system_output: SystemOutput | list[Prediction],
    labels: list[dict[str, Any]],
    truth_links: list[dict[str, Any]] | dict[str, str | None],
    system_id: str = "system",
    split: str = "dev",
    seed: int = 1,
) -> ScoringResult:
    """Evaluate system predictions and bank matches against ground-truth labels and truth links."""
    if isinstance(system_output, SystemOutput):
        predictions = list(system_output.predictions)
        bank_links = list(system_output.bank_links)
    elif isinstance(system_output, list):
        predictions = list(system_output)
        bank_links = []
    else:
        raise TypeError(f"Expected SystemOutput or list[Prediction], got {type(system_output)}")

    # Normalize truth links to mapping {statement_id -> settlement_id}
    truth_map: dict[str, str | None] = {}
    if isinstance(truth_links, dict):
        truth_map = dict(truth_links)
    elif isinstance(truth_links, list):
        for item in truth_links:
            truth_map[item["statement_id"]] = item.get("settlement_id")

    # =========================================================================
    # 1. One-to-one greedy assignment between predictions and labels
    # =========================================================================
    # Candidate matches: merchant_id equal and entity_ids intersect
    candidate_pairs: list[tuple[int, int, int, str]] = []  # (-overlap, l_idx, p_idx, case_id)
    for p_idx, pred in enumerate(predictions):
        p_entities = set(pred.entity_ids)
        for l_idx, lbl in enumerate(labels):
            if pred.merchant_id == lbl.get("merchant_id"):
                l_entities = set(lbl.get("entity_ids", []))
                overlap = len(p_entities & l_entities)
                if overlap > 0:
                    case_id = str(lbl.get("case_id", f"case_{l_idx:06d}"))
                    candidate_pairs.append((-overlap, l_idx, p_idx, case_id))

    # Sort candidates greedily: largest overlap first, tiebreaker by case_id then p_idx
    candidate_pairs.sort(key=lambda x: (x[0], x[3], x[2]))

    assigned_preds: dict[int, int] = {}  # p_idx -> l_idx
    assigned_labels: dict[int, int] = {}  # l_idx -> p_idx

    for neg_ov, l_idx, p_idx, _ in candidate_pairs:
        if p_idx not in assigned_preds and l_idx not in assigned_labels:
            assigned_preds[p_idx] = l_idx
            assigned_labels[l_idx] = p_idx

    # =========================================================================
    # 2. Confusion Matrix & Flagging Action Tracking
    # =========================================================================
    matrix_rows = list(TIER1_ERROR_CLASSES) + ["HARD_NEGATIVE", "AMBIGUITY", "CLEAN"]
    matrix_cols = list(TIER1_ERROR_CLASSES) + ["UNKNOWN", "NONE"]
    confusion_matrix: dict[str, dict[str, int]] = {
        row: {col: 0 for col in matrix_cols} for row in matrix_rows
    }

    # Tracking for Tier 1 metrics
    t1_correct_tp = {c: 0 for c in TIER1_ERROR_CLASSES}
    t1_correct_fp = {c: 0 for c in TIER1_ERROR_CLASSES}
    t1_correct_fn = {c: 0 for c in TIER1_ERROR_CLASSES}

    t1_agnostic_tp = {c: 0 for c in TIER1_ERROR_CLASSES}
    t1_agnostic_fp = {c: 0 for c in TIER1_ERROR_CLASSES}
    t1_agnostic_fn = {c: 0 for c in TIER1_ERROR_CLASSES}

    clean_false_positives = 0
    total_flagged_predictions = 0

    # Categorize unmatched predictions
    for p_idx, pred in enumerate(predictions):
        is_flagged = pred.action in ("flag", "escalate_human")
        if is_flagged:
            total_flagged_predictions += 1

        if p_idx not in assigned_preds:
            if is_flagged:
                clean_false_positives += 1
                pred_c = pred.predicted_class if pred.predicted_class in matrix_cols else "UNKNOWN"
                confusion_matrix["CLEAN"][pred_c] += 1
                if pred.predicted_class in TIER1_ERROR_CLASSES:
                    t1_correct_fp[pred.predicted_class] += 1
                    t1_agnostic_fp[pred.predicted_class] += 1

    # Categorize labels
    for l_idx, lbl in enumerate(labels):
        err_cls = lbl.get("error_class")
        hn_type = lbl.get("hard_negative_type")
        amb_type = lbl.get("ambiguity_type")

        # Determine row in confusion matrix
        if err_cls:
            row_key = err_cls
        elif hn_type:
            row_key = "HARD_NEGATIVE"
        elif amb_type:
            row_key = "AMBIGUITY"
        else:
            row_key = "CLEAN"

        matched_p_idx = assigned_labels.get(l_idx)
        if matched_p_idx is not None:
            pred = predictions[matched_p_idx]
            is_flagged = pred.action in ("flag", "escalate_human")
            pred_class = pred.predicted_class if is_flagged else "NONE"
            col_key = pred_class if pred_class in matrix_cols else "UNKNOWN"
        else:
            pred = None
            is_flagged = False
            col_key = "NONE"

        confusion_matrix[row_key][col_key] += 1

        # Tier 1 classification accounting
        if err_cls and err_cls in TIER1_ERROR_CLASSES:
            if is_flagged and pred is not None:
                # Class-agnostic detection is TP whenever flagged
                t1_agnostic_tp[err_cls] += 1

                # Class-correct detection requires matching class
                if pred.predicted_class == err_cls:
                    t1_correct_tp[err_cls] += 1
                else:
                    t1_correct_fn[err_cls] += 1
                    if pred.predicted_class in TIER1_ERROR_CLASSES:
                        t1_correct_fp[pred.predicted_class] += 1
            else:
                t1_correct_fn[err_cls] += 1
                t1_agnostic_fn[err_cls] += 1

        # If a hard negative or ambiguity label was flagged by a Tier 1 class prediction, that class gets an FP
        if (hn_type or amb_type) and is_flagged and pred is not None:
            if pred.predicted_class in TIER1_ERROR_CLASSES:
                t1_correct_fp[pred.predicted_class] += 1
                t1_agnostic_fp[pred.predicted_class] += 1

    # =========================================================================
    # 3. Compute Tier 1 Precision / Recall / F1 with Wilson CIs
    # =========================================================================
    tier1_class_correct: dict[str, Tier1Metric] = {}
    tier1_class_agnostic: dict[str, Tier1Metric] = {}

    for c in TIER1_ERROR_CLASSES:
        # Class-Correct
        c_tp = t1_correct_tp[c]
        c_fp = t1_correct_fp[c]
        c_fn = t1_correct_fn[c]
        p_denom = c_tp + c_fp
        r_denom = c_tp + c_fn
        prec = (c_tp / p_denom) if p_denom > 0 else None
        prec_ci = wilson_ci(c_tp, p_denom) if p_denom > 0 else None
        rec = (c_tp / r_denom) if r_denom > 0 else None
        rec_ci = wilson_ci(c_tp, r_denom) if r_denom > 0 else None
        f1 = (
            round(2 * prec * rec / (prec + rec), 4)
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else None
        )

        tier1_class_correct[c] = Tier1Metric(
            tp=c_tp,
            fp=c_fp,
            fn=c_fn,
            precision=round(prec, 4) if prec is not None else None,
            precision_ci=prec_ci,
            precision_kn=f"{c_tp}/{p_denom}",
            recall=round(rec, 4) if rec is not None else None,
            recall_ci=rec_ci,
            recall_kn=f"{c_tp}/{r_denom}",
            f1=f1,
        )

        # Class-Agnostic
        a_tp = t1_agnostic_tp[c]
        a_fp = t1_agnostic_fp[c]
        a_fn = t1_agnostic_fn[c]
        ap_denom = a_tp + a_fp
        ar_denom = a_tp + a_fn
        a_prec = (a_tp / ap_denom) if ap_denom > 0 else None
        a_prec_ci = wilson_ci(a_tp, ap_denom) if ap_denom > 0 else None
        a_rec = (a_tp / ar_denom) if ar_denom > 0 else None
        a_rec_ci = wilson_ci(a_tp, ar_denom) if ar_denom > 0 else None
        a_f1 = (
            round(2 * a_prec * a_rec / (a_prec + a_rec), 4)
            if (a_prec is not None and a_rec is not None and (a_prec + a_rec) > 0)
            else None
        )

        tier1_class_agnostic[c] = Tier1Metric(
            tp=a_tp,
            fp=a_fp,
            fn=a_fn,
            precision=round(a_prec, 4) if a_prec is not None else None,
            precision_ci=a_prec_ci,
            precision_kn=f"{a_tp}/{ap_denom}",
            recall=round(a_rec, 4) if a_rec is not None else None,
            recall_ci=a_rec_ci,
            recall_kn=f"{a_tp}/{ar_denom}",
            f1=a_f1,
        )

    # =========================================================================
    # 4. Hard Negatives False-Positive Rate with Wilson CIs
    # =========================================================================
    hard_negatives: dict[str, HardNegativeMetric] = {}
    hn_counts: dict[str, int] = {hn: 0 for hn in HARD_NEGATIVE_TYPES}
    hn_flagged: dict[str, int] = {hn: 0 for hn in HARD_NEGATIVE_TYPES}

    for l_idx, lbl in enumerate(labels):
        hn = lbl.get("hard_negative_type")
        if hn and hn in hn_counts:
            hn_counts[hn] += 1
            p_idx = assigned_labels.get(l_idx)
            if p_idx is not None:
                pred = predictions[p_idx]
                if pred.action in ("flag", "escalate_human"):
                    hn_flagged[hn] += 1

    for hn in HARD_NEGATIVE_TYPES:
        tot = hn_counts[hn]
        flg = hn_flagged[hn]
        fpr = (flg / tot) if tot > 0 else 0.0
        hard_negatives[hn] = HardNegativeMetric(
            flagged=flg,
            total=tot,
            fpr=round(fpr, 4),
            fpr_ci=wilson_ci(flg, tot),
            fpr_kn=f"{flg}/{tot}",
        )

    # =========================================================================
    # 5. Ambiguity Tier Action Accuracy with Wilson CIs
    # =========================================================================
    ambiguity_tier: dict[str, AmbiguityMetric] = {}
    amb_counts: dict[str, int] = {amb: 0 for amb in AMBIGUITY_TYPES}
    amb_correct: dict[str, int] = {amb: 0 for amb in AMBIGUITY_TYPES}

    for l_idx, lbl in enumerate(labels):
        amb = lbl.get("ambiguity_type")
        if amb and amb in amb_counts:
            amb_counts[amb] += 1
            expected = lbl.get("expected_action", "ignore")
            p_idx = assigned_labels.get(l_idx)
            if p_idx is not None:
                observed = predictions[p_idx].action
            else:
                observed = "ignore"  # unpredicted defaults to ignore

            if observed == expected:
                amb_correct[amb] += 1

    for amb in AMBIGUITY_TYPES:
        tot = amb_counts[amb]
        cor = amb_correct[amb]
        acc = (cor / tot) if tot > 0 else 0.0
        ambiguity_tier[amb] = AmbiguityMetric(
            correct=cor,
            total=tot,
            accuracy=round(acc, 4),
            accuracy_ci=wilson_ci(cor, tot),
            accuracy_kn=f"{cor}/{tot}",
        )

    # =========================================================================
    # 6. Bank-to-Settlement Matching Metrics
    # =========================================================================
    links_made = len(bank_links)
    correct_links = 0
    wrong_links = 0
    for link in bank_links:
        true_s_id = truth_map.get(link.statement_id)
        if true_s_id == link.settlement_id:
            correct_links += 1
        else:
            wrong_links += 1

    false_match_rate = (wrong_links / links_made) if links_made > 0 else None
    false_match_ci = wilson_ci(wrong_links, links_made) if links_made > 0 else None

    total_true_settlement_rows = sum(1 for sid in truth_map.values() if sid is not None)
    match_rate = (correct_links / total_true_settlement_rows) if total_true_settlement_rows > 0 else None
    match_rate_ci = wilson_ci(correct_links, total_true_settlement_rows) if total_true_settlement_rows > 0 else None

    matching_metric = MatchingMetric(
        links_made=links_made,
        correct_links=correct_links,
        wrong_links=wrong_links,
        false_match_rate=round(false_match_rate, 4) if false_match_rate is not None else None,
        false_match_ci=false_match_ci,
        match_rate=round(match_rate, 4) if match_rate is not None else None,
        match_rate_ci=match_rate_ci,
    )

    result = ScoringResult(
        system_id=system_id,
        split=split,
        seed=seed,
        tier1_class_correct=tier1_class_correct,
        tier1_class_agnostic=tier1_class_agnostic,
        hard_negatives=hard_negatives,
        ambiguity_tier=ambiguity_tier,
        matching=matching_metric,
        clean_false_positives=clean_false_positives,
        total_flagged_predictions=total_flagged_predictions,
        total_labels=len(labels),
        confusion_matrix=confusion_matrix,
    )
    result.summary_markdown = format_markdown_report(result)
    return result


def format_markdown_report(res: ScoringResult) -> str:
    """Format scoring results into structured markdown tables."""
    lines = [
        f"# Evaluation Report: {res.system_id.upper()} (Split: {res.split}, Seed: {res.seed})",
        "",
        "## 1. Tier 1: Deterministic Error Classes (Class-Correct Detection)",
        "| Error Class | TP | FP | FN | Precision (k/n) | 95% CI | Recall (k/n) | 95% CI | F1 |",
        "|:---|---:|---:|---:|---:|:---:|---:|:---:|---:|",
    ]
    valid_c_p = [m.precision for m in res.tier1_class_correct.values() if m.precision is not None]
    valid_c_r = [m.recall for m in res.tier1_class_correct.values() if m.recall is not None]
    valid_c_f = [m.f1 for m in res.tier1_class_correct.values() if m.f1 is not None]
    for c, m in res.tier1_class_correct.items():
        if m.precision is not None:
            prec_str = f"{m.precision * 100:.1f}% ({m.precision_kn})"
            prec_ci = f"[{m.precision_ci[0]*100:.1f}%, {m.precision_ci[1]*100:.1f}%]" if m.precision_ci else "N/A"
        else:
            prec_str = f"N/A ({m.precision_kn})"
            prec_ci = "N/A"

        if m.recall is not None:
            rec_str = f"{m.recall * 100:.1f}% ({m.recall_kn})"
            rec_ci = f"[{m.recall_ci[0]*100:.1f}%, {m.recall_ci[1]*100:.1f}%]" if m.recall_ci else "N/A"
        else:
            rec_str = f"N/A ({m.recall_kn})"
            rec_ci = "N/A"

        f1_str = f"{m.f1:.3f}" if m.f1 is not None else "N/A"
        lines.append(
            f"| {c:<23} | {m.tp:>4} | {m.fp:>4} | {m.fn:>4} | {prec_str:>16} | {prec_ci:^16} | {rec_str:>16} | {rec_ci:^16} | {f1_str:>6} |"
        )
    macro_p = f"{sum(valid_c_p)/len(valid_c_p)*100:.1f}%" if valid_c_p else "N/A"
    macro_r = f"{sum(valid_c_r)/len(valid_c_r)*100:.1f}%" if valid_c_r else "N/A"
    macro_f = f"{sum(valid_c_f)/len(valid_c_f):.3f}" if valid_c_f else "N/A"
    lines.append(
        f"| {'Macro-Average':<23} | {'-':>4} | {'-':>4} | {'-':>4} | {macro_p:>16} | {'-':^16} | {macro_r:>16} | {'-':^16} | {macro_f:>6} |"
    )

    lines.extend([
        "",
        "## 2. Tier 1: Anomaly Detection (Class-Agnostic Detection)",
        "| Error Class | TP | FP | FN | Precision (k/n) | 95% CI | Recall (k/n) | 95% CI | F1 |",
        "|:---|---:|---:|---:|---:|:---:|---:|:---:|---:|",
    ])
    valid_a_p = [m.precision for m in res.tier1_class_agnostic.values() if m.precision is not None]
    valid_a_r = [m.recall for m in res.tier1_class_agnostic.values() if m.recall is not None]
    valid_a_f = [m.f1 for m in res.tier1_class_agnostic.values() if m.f1 is not None]
    for c, m in res.tier1_class_agnostic.items():
        if m.precision is not None:
            prec_str = f"{m.precision * 100:.1f}% ({m.precision_kn})"
            prec_ci = f"[{m.precision_ci[0]*100:.1f}%, {m.precision_ci[1]*100:.1f}%]" if m.precision_ci else "N/A"
        else:
            prec_str = f"N/A ({m.precision_kn})"
            prec_ci = "N/A"

        if m.recall is not None:
            rec_str = f"{m.recall * 100:.1f}% ({m.recall_kn})"
            rec_ci = f"[{m.recall_ci[0]*100:.1f}%, {m.recall_ci[1]*100:.1f}%]" if m.recall_ci else "N/A"
        else:
            rec_str = f"N/A ({m.recall_kn})"
            rec_ci = "N/A"

        f1_str = f"{m.f1:.3f}" if m.f1 is not None else "N/A"
        lines.append(
            f"| {c:<23} | {m.tp:>4} | {m.fp:>4} | {m.fn:>4} | {prec_str:>16} | {prec_ci:^16} | {rec_str:>16} | {rec_ci:^16} | {f1_str:>6} |"
        )
    macro_ap = f"{sum(valid_a_p)/len(valid_a_p)*100:.1f}%" if valid_a_p else "N/A"
    macro_ar = f"{sum(valid_a_r)/len(valid_a_r)*100:.1f}%" if valid_a_r else "N/A"
    macro_af = f"{sum(valid_a_f)/len(valid_a_f):.3f}" if valid_a_f else "N/A"
    lines.append(
        f"| {'Macro-Average':<23} | {'-':>4} | {'-':>4} | {'-':>4} | {macro_ap:>16} | {'-':^16} | {macro_ar:>16} | {'-':^16} | {macro_af:>6} |"
    )

    lines.extend([
        "",
        "## 3. Hard Negatives (False Positive Resistance)",
        "| Hard Negative Type | Flagged / Total | False Positive Rate (%) | 95% CI |",
        "|:---|---:|---:|:---:|",
    ])
    for hn, m in res.hard_negatives.items():
        fpr_str = f"{m.fpr * 100:.2f}%"
        ci_str = f"[{m.fpr_ci[0]*100:.1f}%, {m.fpr_ci[1]*100:.1f}%]"
        lines.append(f"| {hn:<25} | {m.fpr_kn:>15} | {fpr_str:>23} | {ci_str:^16} |")

    lines.extend([
        "",
        "## 4. Ambiguity Tier (Action Accuracy)",
        "| Ambiguity Type | Correct / Total | Action Accuracy (%) | 95% CI |",
        "|:---|---:|---:|:---:|",
    ])
    for amb, m in res.ambiguity_tier.items():
        acc_str = f"{m.accuracy * 100:.2f}%"
        ci_str = f"[{m.accuracy_ci[0]*100:.1f}%, {m.accuracy_ci[1]*100:.1f}%]"
        lines.append(f"| {amb:<25} | {m.accuracy_kn:>15} | {acc_str:>19} | {ci_str:^16} |")

    lines.extend([
        "",
        "## 5. Bank Statement to Settlement Matching",
        "| Metric | Count / Value | 95% CI |",
        "|:---|---:|:---:|",
        f"| Links Made | {res.matching.links_made} | N/A |",
        f"| Correct Links | {res.matching.correct_links} | N/A |",
        f"| Wrong Links | {res.matching.wrong_links} | N/A |",
    ])
    if res.matching.false_match_rate is not None and res.matching.false_match_ci is not None:
        fmr_str = f"{res.matching.false_match_rate*100:.2f}% ({res.matching.wrong_links}/{res.matching.links_made})"
        fmr_ci = f"[{res.matching.false_match_ci[0]*100:.1f}%, {res.matching.false_match_ci[1]*100:.1f}%]"
    else:
        fmr_str = f"N/A ({res.matching.wrong_links}/{res.matching.links_made})"
        fmr_ci = "N/A"

    if res.matching.match_rate is not None and res.matching.match_rate_ci is not None:
        mr_str = f"{res.matching.match_rate*100:.2f}%"
        mr_ci = f"[{res.matching.match_rate_ci[0]*100:.1f}%, {res.matching.match_rate_ci[1]*100:.1f}%]"
    else:
        mr_str = "N/A"
        mr_ci = "N/A"

    lines.extend([
        f"| False-Match Rate | {fmr_str} | {fmr_ci} |",
        f"| Match Rate | {mr_str} | {mr_ci} |",
        "",
        f"**Clean Data False Positives:** {res.clean_false_positives}",
        f"**Total Flagged Predictions:** {res.total_flagged_predictions}",
        f"**Total Ground-Truth Labels:** {res.total_labels}",
        "",
        "## 6. Confusion Matrix",
    ])

    cols = list(TIER1_ERROR_CLASSES) + ["UNKNOWN", "NONE"]
    short_cols = [c[:8] for c in cols]
    header = "| True \\ Pred | " + " | ".join(short_cols) + " |"
    sep = "|:---| " + " | ".join(["---:"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)
    for r_name, r_vals in res.confusion_matrix.items():
        row_str = f"| {r_name:<11} | " + " | ".join(f"{r_vals[c]:>8}" for c in cols) + " |"
        lines.append(row_str)

    return "\n".join(lines)


def compute_pooled_results(results: list[ScoringResult]) -> dict[str, Any]:
    """Pool evaluation counts across multiple seeds with Wilson CIs and per-seed metric ranges."""
    if not results:
        raise ValueError("Cannot pool empty results list")

    system_id = results[0].system_id
    split = results[0].split
    seeds = [r.seed for r in results]

    # 1. Tier 1 Class-Correct
    t1_correct: dict[str, dict[str, Any]] = {}
    valid_c_p_pooled: list[float] = []
    valid_c_r_pooled: list[float] = []
    valid_c_f_pooled: list[float] = []

    for c in TIER1_ERROR_CLASSES:
        tp = sum(r.tier1_class_correct[c].tp for r in results)
        fp = sum(r.tier1_class_correct[c].fp for r in results)
        fn = sum(r.tier1_class_correct[c].fn for r in results)
        p_denom = tp + fp
        r_denom = tp + fn

        prec = round(tp / p_denom, 4) if p_denom > 0 else None
        prec_ci = wilson_ci(tp, p_denom) if p_denom > 0 else None
        rec = round(tp / r_denom, 4) if r_denom > 0 else None
        rec_ci = wilson_ci(tp, r_denom) if r_denom > 0 else None
        f1 = (
            round(2 * prec * rec / (prec + rec), 4)
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else None
        )

        seed_precs = [
            r.tier1_class_correct[c].precision
            for r in results
            if r.tier1_class_correct[c].precision is not None
        ]
        seed_recs = [
            r.tier1_class_correct[c].recall
            for r in results
            if r.tier1_class_correct[c].recall is not None
        ]
        seed_f1s = [
            r.tier1_class_correct[c].f1
            for r in results
            if r.tier1_class_correct[c].f1 is not None
        ]

        p_range = (min(seed_precs), max(seed_precs)) if seed_precs else None
        r_range = (min(seed_recs), max(seed_recs)) if seed_recs else None
        f_range = (min(seed_f1s), max(seed_f1s)) if seed_f1s else None

        if prec is not None:
            valid_c_p_pooled.append(prec)
        if rec is not None:
            valid_c_r_pooled.append(rec)
        if f1 is not None:
            valid_c_f_pooled.append(f1)

        t1_correct[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "precision_ci": prec_ci,
            "precision_kn": f"{tp}/{p_denom}",
            "precision_range": p_range,
            "recall": rec,
            "recall_ci": rec_ci,
            "recall_kn": f"{tp}/{r_denom}",
            "recall_range": r_range,
            "f1": f1,
            "f1_range": f_range,
        }

    macro_c_p = round(sum(valid_c_p_pooled) / len(valid_c_p_pooled), 4) if valid_c_p_pooled else None
    macro_c_r = round(sum(valid_c_r_pooled) / len(valid_c_r_pooled), 4) if valid_c_r_pooled else None
    macro_c_f = round(sum(valid_c_f_pooled) / len(valid_c_f_pooled), 4) if valid_c_f_pooled else None

    # 2. Tier 1 Class-Agnostic
    t1_agnostic: dict[str, dict[str, Any]] = {}
    valid_a_p_pooled: list[float] = []
    valid_a_r_pooled: list[float] = []
    valid_a_f_pooled: list[float] = []

    for c in TIER1_ERROR_CLASSES:
        tp = sum(r.tier1_class_agnostic[c].tp for r in results)
        fp = sum(r.tier1_class_agnostic[c].fp for r in results)
        fn = sum(r.tier1_class_agnostic[c].fn for r in results)
        p_denom = tp + fp
        r_denom = tp + fn

        prec = round(tp / p_denom, 4) if p_denom > 0 else None
        prec_ci = wilson_ci(tp, p_denom) if p_denom > 0 else None
        rec = round(tp / r_denom, 4) if r_denom > 0 else None
        rec_ci = wilson_ci(tp, r_denom) if r_denom > 0 else None
        f1 = (
            round(2 * prec * rec / (prec + rec), 4)
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else None
        )

        seed_precs = [
            r.tier1_class_agnostic[c].precision
            for r in results
            if r.tier1_class_agnostic[c].precision is not None
        ]
        seed_recs = [
            r.tier1_class_agnostic[c].recall
            for r in results
            if r.tier1_class_agnostic[c].recall is not None
        ]
        seed_f1s = [
            r.tier1_class_agnostic[c].f1
            for r in results
            if r.tier1_class_agnostic[c].f1 is not None
        ]

        p_range = (min(seed_precs), max(seed_precs)) if seed_precs else None
        r_range = (min(seed_recs), max(seed_recs)) if seed_recs else None
        f_range = (min(seed_f1s), max(seed_f1s)) if seed_f1s else None

        if prec is not None:
            valid_a_p_pooled.append(prec)
        if rec is not None:
            valid_a_r_pooled.append(rec)
        if f1 is not None:
            valid_a_f_pooled.append(f1)

        t1_agnostic[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "precision_ci": prec_ci,
            "precision_kn": f"{tp}/{p_denom}",
            "precision_range": p_range,
            "recall": rec,
            "recall_ci": rec_ci,
            "recall_kn": f"{tp}/{r_denom}",
            "recall_range": r_range,
            "f1": f1,
            "f1_range": f_range,
        }

    macro_a_p = round(sum(valid_a_p_pooled) / len(valid_a_p_pooled), 4) if valid_a_p_pooled else None
    macro_a_r = round(sum(valid_a_r_pooled) / len(valid_a_r_pooled), 4) if valid_a_r_pooled else None
    macro_a_f = round(sum(valid_a_f_pooled) / len(valid_a_f_pooled), 4) if valid_a_f_pooled else None

    # 3. Hard Negatives
    hard_negatives: dict[str, dict[str, Any]] = {}
    for hn in HARD_NEGATIVE_TYPES:
        flg = sum(r.hard_negatives[hn].flagged for r in results)
        tot = sum(r.hard_negatives[hn].total for r in results)
        fpr = round(flg / tot, 4) if tot > 0 else 0.0
        fpr_ci = wilson_ci(flg, tot) if tot > 0 else (0.0, 0.0)
        seed_fprs = [r.hard_negatives[hn].fpr for r in results]
        hard_negatives[hn] = {
            "flagged": flg,
            "total": tot,
            "fpr": fpr,
            "fpr_ci": fpr_ci,
            "fpr_kn": f"{flg}/{tot}",
            "fpr_range": (min(seed_fprs), max(seed_fprs)),
        }

    # 4. Ambiguity Tier
    ambiguity_tier: dict[str, dict[str, Any]] = {}
    for amb in AMBIGUITY_TYPES:
        cor = sum(r.ambiguity_tier[amb].correct for r in results)
        tot = sum(r.ambiguity_tier[amb].total for r in results)
        acc = round(cor / tot, 4) if tot > 0 else 0.0
        acc_ci = wilson_ci(cor, tot) if tot > 0 else (0.0, 0.0)
        seed_accs = [r.ambiguity_tier[amb].accuracy for r in results]
        ambiguity_tier[amb] = {
            "correct": cor,
            "total": tot,
            "accuracy": acc,
            "accuracy_ci": acc_ci,
            "accuracy_kn": f"{cor}/{tot}",
            "accuracy_range": (min(seed_accs), max(seed_accs)),
        }

    # 5. Matching
    links_made = sum(r.matching.links_made for r in results)
    correct_links = sum(r.matching.correct_links for r in results)
    wrong_links = sum(r.matching.wrong_links for r in results)
    fmr = round(wrong_links / links_made, 4) if links_made > 0 else None
    fmr_ci = wilson_ci(wrong_links, links_made) if links_made > 0 else None
    seed_fmrs = [
        r.matching.false_match_rate
        for r in results
        if r.matching.false_match_rate is not None
    ]
    fmr_range = (min(seed_fmrs), max(seed_fmrs)) if seed_fmrs else None

    total_true_settlements = sum(
        int(round(r.matching.correct_links / r.matching.match_rate))
        for r in results
        if r.matching.match_rate is not None and r.matching.match_rate > 0
    )
    mr = round(correct_links / total_true_settlements, 4) if total_true_settlements > 0 else None
    mr_ci = wilson_ci(correct_links, total_true_settlements) if total_true_settlements > 0 else None
    seed_mrs = [
        r.matching.match_rate
        for r in results
        if r.matching.match_rate is not None
    ]
    mr_range = (min(seed_mrs), max(seed_mrs)) if seed_mrs else None

    matching = {
        "links_made": links_made,
        "correct_links": correct_links,
        "wrong_links": wrong_links,
        "false_match_rate": fmr,
        "false_match_ci": fmr_ci,
        "false_match_range": fmr_range,
        "match_rate": mr,
        "match_rate_ci": mr_ci,
        "match_rate_range": mr_range,
    }

    pooled_dict = {
        "system_id": system_id,
        "split": split,
        "seeds": seeds,
        "tier1_class_correct": t1_correct,
        "tier1_class_correct_macro": {
            "precision": macro_c_p,
            "recall": macro_c_r,
            "f1": macro_c_f,
        },
        "tier1_class_agnostic": t1_agnostic,
        "tier1_class_agnostic_macro": {
            "precision": macro_a_p,
            "recall": macro_a_r,
            "f1": macro_a_f,
        },
        "hard_negatives": hard_negatives,
        "ambiguity_tier": ambiguity_tier,
        "matching": matching,
        "clean_false_positives": sum(r.clean_false_positives for r in results),
        "total_flagged_predictions": sum(r.total_flagged_predictions for r in results),
        "total_labels": sum(r.total_labels for r in results),
    }

    markdown_report = format_pooled_markdown_report(pooled_dict)
    pooled_dict["summary_markdown"] = markdown_report
    return pooled_dict


def format_pooled_markdown_report(pooled: dict[str, Any]) -> str:
    """Format pooled multi-seed evaluation results into structured markdown tables."""
    seeds_str = ", ".join(str(s) for s in pooled["seeds"])
    lines = [
        f"# Multi-Seed Evaluation Report: {pooled['system_id'].upper()} (Split: {pooled['split']}, Seeds: {seeds_str})",
        "",
        "## 1. Tier 1: Deterministic Error Classes (Class-Correct Detection, Pooled)",
        "| Error Class | TP | FP | FN | Precision (k/n) | 95% CI | Seed Range | Recall (k/n) | 95% CI | Seed Range | F1 | Seed Range |",
        "|:---|---:|---:|---:|---:|:---:|:---:|---:|:---:|:---:|---:|:---:|",
    ]
    t1_c = pooled["tier1_class_correct"]
    for c in TIER1_ERROR_CLASSES:
        m = t1_c[c]
        if m["precision"] is not None:
            prec_str = f"{m['precision'] * 100:.1f}% ({m['precision_kn']})"
            prec_ci = f"[{m['precision_ci'][0]*100:.1f}%, {m['precision_ci'][1]*100:.1f}%]" if m["precision_ci"] else "N/A"
            p_range_str = f"[{m['precision_range'][0]*100:.1f}%, {m['precision_range'][1]*100:.1f}%]" if m["precision_range"] else "N/A"
        else:
            prec_str = f"N/A ({m['precision_kn']})"
            prec_ci = "N/A"
            p_range_str = "N/A"

        if m["recall"] is not None:
            rec_str = f"{m['recall'] * 100:.1f}% ({m['recall_kn']})"
            rec_ci = f"[{m['recall_ci'][0]*100:.1f}%, {m['recall_ci'][1]*100:.1f}%]" if m["recall_ci"] else "N/A"
            r_range_str = f"[{m['recall_range'][0]*100:.1f}%, {m['recall_range'][1]*100:.1f}%]" if m["recall_range"] else "N/A"
        else:
            rec_str = f"N/A ({m['recall_kn']})"
            rec_ci = "N/A"
            r_range_str = "N/A"

        f1_str = f"{m['f1']:.3f}" if m["f1"] is not None else "N/A"
        f_range_str = f"[{m['f1_range'][0]:.3f}, {m['f1_range'][1]:.3f}]" if m["f1_range"] else "N/A"

        lines.append(
            f"| {c:<23} | {m['tp']:>4} | {m['fp']:>4} | {m['fn']:>4} | {prec_str:>16} | {prec_ci:^16} | {p_range_str:^16} | {rec_str:>16} | {rec_ci:^16} | {r_range_str:^16} | {f1_str:>6} | {f_range_str:^14} |"
        )

    macro_c = pooled["tier1_class_correct_macro"]
    m_p = f"{macro_c['precision']*100:.1f}%" if macro_c["precision"] is not None else "N/A"
    m_r = f"{macro_c['recall']*100:.1f}%" if macro_c["recall"] is not None else "N/A"
    m_f = f"{macro_c['f1']:.3f}" if macro_c["f1"] is not None else "N/A"
    lines.append(
        f"| {'Macro-Average':<23} | {'-':>4} | {'-':>4} | {'-':>4} | {m_p:>16} | {'-':^16} | {'-':^16} | {m_r:>16} | {'-':^16} | {'-':^16} | {m_f:>6} | {'-':^14} |"
    )

    lines.extend([
        "",
        "## 2. Tier 1: Anomaly Detection (Class-Agnostic Detection, Pooled)",
        "| Error Class | TP | FP | FN | Precision (k/n) | 95% CI | Seed Range | Recall (k/n) | 95% CI | Seed Range | F1 | Seed Range |",
        "|:---|---:|---:|---:|---:|:---:|:---:|---:|:---:|:---:|---:|:---:|",
    ])
    t1_a = pooled["tier1_class_agnostic"]
    for c in TIER1_ERROR_CLASSES:
        m = t1_a[c]
        if m["precision"] is not None:
            prec_str = f"{m['precision'] * 100:.1f}% ({m['precision_kn']})"
            prec_ci = f"[{m['precision_ci'][0]*100:.1f}%, {m['precision_ci'][1]*100:.1f}%]" if m["precision_ci"] else "N/A"
            p_range_str = f"[{m['precision_range'][0]*100:.1f}%, {m['precision_range'][1]*100:.1f}%]" if m["precision_range"] else "N/A"
        else:
            prec_str = f"N/A ({m['precision_kn']})"
            prec_ci = "N/A"
            p_range_str = "N/A"

        if m["recall"] is not None:
            rec_str = f"{m['recall'] * 100:.1f}% ({m['recall_kn']})"
            rec_ci = f"[{m['recall_ci'][0]*100:.1f}%, {m['recall_ci'][1]*100:.1f}%]" if m["recall_ci"] else "N/A"
            r_range_str = f"[{m['recall_range'][0]*100:.1f}%, {m['recall_range'][1]*100:.1f}%]" if m["recall_range"] else "N/A"
        else:
            rec_str = f"N/A ({m['recall_kn']})"
            rec_ci = "N/A"
            r_range_str = "N/A"

        f1_str = f"{m['f1']:.3f}" if m["f1"] is not None else "N/A"
        f_range_str = f"[{m['f1_range'][0]:.3f}, {m['f1_range'][1]:.3f}]" if m["f1_range"] else "N/A"

        lines.append(
            f"| {c:<23} | {m['tp']:>4} | {m['fp']:>4} | {m['fn']:>4} | {prec_str:>16} | {prec_ci:^16} | {p_range_str:^16} | {rec_str:>16} | {rec_ci:^16} | {r_range_str:^16} | {f1_str:>6} | {f_range_str:^14} |"
        )

    macro_a = pooled["tier1_class_agnostic_macro"]
    ma_p = f"{macro_a['precision']*100:.1f}%" if macro_a["precision"] is not None else "N/A"
    ma_r = f"{macro_a['recall']*100:.1f}%" if macro_a["recall"] is not None else "N/A"
    ma_f = f"{macro_a['f1']:.3f}" if macro_a["f1"] is not None else "N/A"
    lines.append(
        f"| {'Macro-Average':<23} | {'-':>4} | {'-':>4} | {'-':>4} | {ma_p:>16} | {'-':^16} | {'-':^16} | {ma_r:>16} | {'-':^16} | {'-':^16} | {ma_f:>6} | {'-':^14} |"
    )

    lines.extend([
        "",
        "## 3. Hard Negatives (False Positive Resistance, Pooled)",
        "| Hard Negative Type | Flagged / Total | False Positive Rate (%) | 95% CI | Per-Seed Range |",
        "|:---|---:|---:|:---:|:---:|",
    ])
    for hn, m in pooled["hard_negatives"].items():
        fpr_str = f"{m['fpr'] * 100:.2f}%"
        ci_str = f"[{m['fpr_ci'][0]*100:.1f}%, {m['fpr_ci'][1]*100:.1f}%]"
        range_str = f"[{m['fpr_range'][0]*100:.1f}%, {m['fpr_range'][1]*100:.1f}%]"
        lines.append(f"| {hn:<25} | {m['fpr_kn']:>15} | {fpr_str:>23} | {ci_str:^16} | {range_str:^16} |")

    lines.extend([
        "",
        "## 4. Ambiguity Tier (Action Accuracy, Pooled)",
        "| Ambiguity Type | Correct / Total | Action Accuracy (%) | 95% CI | Per-Seed Range |",
        "|:---|---:|---:|:---:|:---:|",
    ])
    for amb, m in pooled["ambiguity_tier"].items():
        acc_str = f"{m['accuracy'] * 100:.2f}%"
        ci_str = f"[{m['accuracy_ci'][0]*100:.1f}%, {m['accuracy_ci'][1]*100:.1f}%]"
        range_str = f"[{m['accuracy_range'][0]*100:.1f}%, {m['accuracy_range'][1]*100:.1f}%]"
        lines.append(f"| {amb:<25} | {m['accuracy_kn']:>15} | {acc_str:>19} | {ci_str:^16} | {range_str:^16} |")

    m_match = pooled["matching"]
    lines.extend([
        "",
        "## 5. Bank Statement to Settlement Matching (Pooled)",
        "| Metric | Count / Value | 95% CI | Per-Seed Range |",
        "|:---|---:|:---:|:---:|",
        f"| Links Made | {m_match['links_made']} | N/A | N/A |",
        f"| Correct Links | {m_match['correct_links']} | N/A | N/A |",
        f"| Wrong Links | {m_match['wrong_links']} | N/A | N/A |",
    ])
    if m_match["false_match_rate"] is not None and m_match["false_match_ci"] is not None:
        fmr_str = f"{m_match['false_match_rate']*100:.2f}% ({m_match['wrong_links']}/{m_match['links_made']})"
        fmr_ci = f"[{m_match['false_match_ci'][0]*100:.1f}%, {m_match['false_match_ci'][1]*100:.1f}%]"
        fmr_rng = f"[{m_match['false_match_range'][0]*100:.1f}%, {m_match['false_match_range'][1]*100:.1f}%]" if m_match["false_match_range"] else "N/A"
    else:
        fmr_str = f"N/A ({m_match['wrong_links']}/{m_match['links_made']})"
        fmr_ci = "N/A"
        fmr_rng = "N/A"

    if m_match["match_rate"] is not None and m_match["match_rate_ci"] is not None:
        mr_str = f"{m_match['match_rate']*100:.2f}%"
        mr_ci = f"[{m_match['match_rate_ci'][0]*100:.1f}%, {m_match['match_rate_ci'][1]*100:.1f}%]"
        mr_rng = f"[{m_match['match_rate_range'][0]*100:.1f}%, {m_match['match_rate_range'][1]*100:.1f}%]" if m_match["match_rate_range"] else "N/A"
    else:
        mr_str = "N/A"
        mr_ci = "N/A"
        mr_rng = "N/A"

    lines.extend([
        f"| False-Match Rate | {fmr_str} | {fmr_ci} | {fmr_rng} |",
        f"| Match Rate | {mr_str} | {mr_ci} | {mr_rng} |",
        "",
        f"**Clean Data False Positives (Pooled):** {pooled['clean_false_positives']}",
        f"**Total Flagged Predictions (Pooled):** {pooled['total_flagged_predictions']}",
        f"**Total Ground-Truth Labels (Pooled):** {pooled['total_labels']}",
    ])

    return "\n".join(lines)
