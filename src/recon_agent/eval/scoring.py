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
    precision: float
    precision_ci: tuple[float, float]
    precision_kn: str
    recall: float
    recall_ci: tuple[float, float]
    recall_kn: str
    f1: float


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
    false_match_rate: float
    false_match_ci: tuple[float, float]
    match_rate: float
    match_rate_ci: tuple[float, float]


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
        prec = (c_tp / p_denom) if p_denom > 0 else 0.0
        rec = (c_tp / r_denom) if r_denom > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        tier1_class_correct[c] = Tier1Metric(
            tp=c_tp,
            fp=c_fp,
            fn=c_fn,
            precision=round(prec, 4),
            precision_ci=wilson_ci(c_tp, p_denom),
            precision_kn=f"{c_tp}/{p_denom}",
            recall=round(rec, 4),
            recall_ci=wilson_ci(c_tp, r_denom),
            recall_kn=f"{c_tp}/{r_denom}",
            f1=round(f1, 4),
        )

        # Class-Agnostic
        a_tp = t1_agnostic_tp[c]
        a_fp = t1_agnostic_fp[c]
        a_fn = t1_agnostic_fn[c]
        ap_denom = a_tp + a_fp
        ar_denom = a_tp + a_fn
        a_prec = (a_tp / ap_denom) if ap_denom > 0 else 0.0
        a_rec = (a_tp / ar_denom) if ar_denom > 0 else 0.0
        a_f1 = (2 * a_prec * a_rec / (a_prec + a_rec)) if (a_prec + a_rec) > 0 else 0.0

        tier1_class_agnostic[c] = Tier1Metric(
            tp=a_tp,
            fp=a_fp,
            fn=a_fn,
            precision=round(a_prec, 4),
            precision_ci=wilson_ci(a_tp, ap_denom),
            precision_kn=f"{a_tp}/{ap_denom}",
            recall=round(a_rec, 4),
            recall_ci=wilson_ci(a_tp, ar_denom),
            recall_kn=f"{a_tp}/{ar_denom}",
            f1=round(a_f1, 4),
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

    false_match_rate = (wrong_links / links_made) if links_made > 0 else 0.0
    false_match_ci = wilson_ci(wrong_links, links_made)

    total_true_settlement_rows = sum(1 for sid in truth_map.values() if sid is not None)
    match_rate = (correct_links / total_true_settlement_rows) if total_true_settlement_rows > 0 else 0.0
    match_rate_ci = wilson_ci(correct_links, total_true_settlement_rows)

    matching_metric = MatchingMetric(
        links_made=links_made,
        correct_links=correct_links,
        wrong_links=wrong_links,
        false_match_rate=round(false_match_rate, 4),
        false_match_ci=false_match_ci,
        match_rate=round(match_rate, 4),
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
    for c, m in res.tier1_class_correct.items():
        prec_str = f"{m.precision * 100:.1f}% ({m.precision_kn})"
        prec_ci = f"[{m.precision_ci[0]*100:.1f}%, {m.precision_ci[1]*100:.1f}%]"
        rec_str = f"{m.recall * 100:.1f}% ({m.recall_kn})"
        rec_ci = f"[{m.recall_ci[0]*100:.1f}%, {m.recall_ci[1]*100:.1f}%]"
        lines.append(
            f"| {c:<23} | {m.tp:>4} | {m.fp:>4} | {m.fn:>4} | {prec_str:>16} | {prec_ci:^16} | {rec_str:>16} | {rec_ci:^16} | {m.f1:>6.3f} |"
        )

    lines.extend([
        "",
        "## 2. Tier 1: Anomaly Detection (Class-Agnostic Detection)",
        "| Error Class | TP | FP | FN | Precision (k/n) | 95% CI | Recall (k/n) | 95% CI | F1 |",
        "|:---|---:|---:|---:|---:|:---:|---:|:---:|---:|",
    ])
    for c, m in res.tier1_class_agnostic.items():
        prec_str = f"{m.precision * 100:.1f}% ({m.precision_kn})"
        prec_ci = f"[{m.precision_ci[0]*100:.1f}%, {m.precision_ci[1]*100:.1f}%]"
        rec_str = f"{m.recall * 100:.1f}% ({m.recall_kn})"
        rec_ci = f"[{m.recall_ci[0]*100:.1f}%, {m.recall_ci[1]*100:.1f}%]"
        lines.append(
            f"| {c:<23} | {m.tp:>4} | {m.fp:>4} | {m.fn:>4} | {prec_str:>16} | {prec_ci:^16} | {rec_str:>16} | {rec_ci:^16} | {m.f1:>6.3f} |"
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
        f"| False-Match Rate | {res.matching.false_match_rate*100:.2f}% ({res.matching.wrong_links}/{res.matching.links_made}) | [{res.matching.false_match_ci[0]*100:.1f}%, {res.matching.false_match_ci[1]*100:.1f}%] |",
        f"| Match Rate | {res.matching.match_rate*100:.2f}% | [{res.matching.match_rate_ci[0]*100:.1f}%, {res.matching.match_rate_ci[1]*100:.1f}%] |",
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
