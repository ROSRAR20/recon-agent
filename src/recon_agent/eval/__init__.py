"""Evaluation package for recon-agent: scoring, metrics, and ground-truth auditing."""

from src.recon_agent.eval.scoring import (
    ScoringResult,
    format_markdown_report,
    load_ground_truth,
    score_predictions,
    wilson_ci,
)

__all__ = [
    "ScoringResult",
    "format_markdown_report",
    "load_ground_truth",
    "score_predictions",
    "wilson_ci",
]
