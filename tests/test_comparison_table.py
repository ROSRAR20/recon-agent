"""Unit tests for the multi-system comparison table generator."""

import json
import os
import subprocess
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.compare_systems import (
    build_comparison_rows,
    generate_comparison_table,
    load_pooled_json,
)
from src.recon_agent.eval.scoring import (
    AMBIGUITY_TYPES,
    HARD_NEGATIVE_TYPES,
    TIER1_ERROR_CLASSES,
)


def test_compare_runs_script_not_modified() -> None:
    """Verify that scripts/compare_runs.py was untouched."""
    res = subprocess.run(
        ["git", "diff", "HEAD", "--", "scripts/compare_runs.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "", "scripts/compare_runs.py must not be modified!"


def test_comparison_table_matches_pooled_json() -> None:
    """Assert every cell value in comparison table matches the exact pooled JSON values."""
    results_dir = os.path.join(REPO_ROOT, "results")
    json_paths = [
        os.path.join(results_dir, "b0_dev_pooled.json"),
        os.path.join(results_dir, "b1_dev_pooled.json"),
        os.path.join(results_dir, "b1r_dev_pooled.json"),
    ]

    for p in json_paths:
        if not os.path.isfile(p):
            pytest.skip(f"Pooled result {p} not found; run evaluation first.")

    systems_data = [load_pooled_json(p) for p in json_paths]
    rows = build_comparison_rows(systems_data)

    # Convert rows to mapping {metric_name: [val_b0, val_b1, val_b1r]}
    row_map = {metric_name: vals for metric_name, vals in rows}

    for sys_idx, data in enumerate(systems_data):
        # 1. Tier 1 Class-Correct Precision / Recall / F1
        for c in TIER1_ERROR_CLASSES:
            m = data["tier1_class_correct"][c]
            # Precision
            p_cell = row_map[f"Tier 1 Correct: {c} Precision"][sys_idx]
            if m["precision"] is not None:
                assert f"{m['precision'] * 100:.2f}%" in p_cell
                assert m["precision_kn"] in p_cell
            else:
                assert f"N/A ({m['precision_kn']})" in p_cell

            # Recall
            r_cell = row_map[f"Tier 1 Correct: {c} Recall"][sys_idx]
            if m["recall"] is not None:
                assert f"{m['recall'] * 100:.2f}%" in r_cell
                assert m["recall_kn"] in r_cell
            else:
                assert f"N/A ({m['recall_kn']})" in r_cell

            # F1
            f_cell = row_map[f"Tier 1 Correct: {c} F1"][sys_idx]
            if m["f1"] is not None:
                assert f"{m['f1']:.4f}" in f_cell
            else:
                assert f_cell == "N/A"

        # 2. Tier 1 Overall Micro
        micro_c = data["tier1_class_correct_micro"]
        micro_p_cell = row_map["Tier 1 Overall Micro Precision"][sys_idx]
        if micro_c["precision"] is not None:
            assert f"{micro_c['precision'] * 100:.2f}%" in micro_p_cell
            assert f"{micro_c['tp']}/{micro_c['tp'] + micro_c['fp']}" in micro_p_cell
        else:
            assert "N/A" in micro_p_cell

        micro_r_cell = row_map["Tier 1 Overall Micro Recall"][sys_idx]
        if micro_c["recall"] is not None:
            assert f"{micro_c['recall'] * 100:.2f}%" in micro_r_cell
            assert f"{micro_c['tp']}/{micro_c['tp'] + micro_c['fn']}" in micro_r_cell
        else:
            assert "N/A" in micro_r_cell

        micro_f_cell = row_map["Tier 1 Overall Micro F1"][sys_idx]
        if micro_c["f1"] is not None:
            assert f"{micro_c['f1']:.4f}" in micro_f_cell
        else:
            assert micro_f_cell == "N/A"

        # 3. Macro-F1
        macro_f_cell = row_map["Macro-F1 (undefined as 0)"][sys_idx]
        assert f"{data['macro_f1_undefined_as_zero']:.4f}" in macro_f_cell

        # 4. Tier 1 Agnostic Micro
        micro_a = data["tier1_class_agnostic_micro"]
        agno_p_cell = row_map["Tier 1 Agnostic Micro Precision"][sys_idx]
        if micro_a["precision"] is not None:
            assert f"{micro_a['precision'] * 100:.2f}%" in agno_p_cell
            assert f"{micro_a['tp']}/{micro_a['tp'] + micro_a['fp']}" in agno_p_cell

        agno_r_cell = row_map["Tier 1 Agnostic Micro Recall"][sys_idx]
        if micro_a["recall"] is not None:
            assert f"{micro_a['recall'] * 100:.2f}%" in agno_r_cell
            assert f"{micro_a['tp']}/{micro_a['tp'] + micro_a['fn']}" in agno_r_cell

        agno_f_cell = row_map["Tier 1 Agnostic Micro F1"][sys_idx]
        if micro_a["f1"] is not None:
            assert f"{micro_a['f1']:.4f}" in agno_f_cell

        # 5. Hard Negatives per-type
        for hn in HARD_NEGATIVE_TYPES:
            m = data["hard_negatives"][hn]
            hn_cell = row_map[f"Hard Negative: {hn} FPR"][sys_idx]
            assert f"{m['fpr'] * 100:.2f}%" in hn_cell
            assert f"({m['flagged']}/{m['total']})" in hn_cell

        # 6. Hard Negatives Overall
        hn_overall_cell = row_map["Hard Negatives Overall FPR"][sys_idx]
        tot_flg = sum(data["hard_negatives"][hn]["flagged"] for hn in HARD_NEGATIVE_TYPES)
        tot_tot = sum(data["hard_negatives"][hn]["total"] for hn in HARD_NEGATIVE_TYPES)
        expected_overall_fpr = round(tot_flg / tot_tot, 4) if tot_tot > 0 else 0.0
        assert f"{expected_overall_fpr * 100:.2f}%" in hn_overall_cell
        assert f"({tot_flg}/{tot_tot})" in hn_overall_cell

        # Check hard negative overall denominator equals sum of 10 per-type totals
        overall_denom = int(hn_overall_cell.split("(")[1].split("/")[1].rstrip(")"))
        sum_of_10_totals = sum(data["hard_negatives"][hn]["total"] for hn in HARD_NEGATIVE_TYPES)
        assert overall_denom == sum_of_10_totals

        # 7. Ambiguity Tier
        for amb in AMBIGUITY_TYPES:
            m = data["ambiguity_tier"][amb]
            amb_cell = row_map[f"Ambiguity: {amb} Accuracy"][sys_idx]
            assert f"{m['accuracy'] * 100:.2f}%" in amb_cell
            assert f"({m['accuracy_kn']})" in amb_cell

        # 8. Bank Matching
        m_match = data["matching"]
        mr_cell = row_map["Bank Match Rate"][sys_idx]
        if m_match["match_rate"] is not None:
            assert f"{m_match['match_rate'] * 100:.2f}%" in mr_cell
        fmr_cell = row_map["Bank False-Match Rate"][sys_idx]
        if m_match["false_match_rate"] is not None:
            assert f"{m_match['false_match_rate'] * 100:.2f}%" in fmr_cell
            assert f"({m_match['wrong_links']}/{m_match['links_made']})" in fmr_cell

        # 9. Operational
        assert row_map["Clean Data False Positives (Flagged)"][sys_idx] == str(data["clean_false_positives"])
        assert row_map["Clean Data Escalations"][sys_idx] == str(data.get("clean_escalated", 0))
        assert row_map["Total Flagged Predictions"][sys_idx] == str(data["total_flagged_predictions"])
        assert row_map["Total Escalated Predictions"][sys_idx] == str(data.get("total_escalated_predictions", 0))
        assert row_map["Escalation Share"][sys_idx] == f"{data.get('escalation_share', 0.0) * 100:.2f}%"


def test_comparison_table_row_count() -> None:
    """Assert the comparison table produces exactly 50 metric rows."""
    results_dir = os.path.join(REPO_ROOT, "results")
    p = os.path.join(results_dir, "b0_dev_pooled.json")
    if not os.path.isfile(p):
        pytest.skip(f"Pooled result {p} not found; run evaluation first.")

    systems_data = [load_pooled_json(p)]
    rows = build_comparison_rows(systems_data)
    assert len(rows) == 50, f"Expected exactly 50 rows in comparison table, got {len(rows)}"

