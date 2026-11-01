"""Unit tests for System B0 baseline."""

import os
import pytest

from src.recon_agent.pipeline.loader import load_dataset
from src.recon_agent.synth.config import create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator
from src.recon_agent.systems.b0 import run_b0


def test_b0_execution_on_synthetic_data(tmp_path) -> None:
    """Verify System B0 runs on observable dataset and emits valid predictions and bank links."""
    config = create_config(profile="small", split="dev", seed=42, inject_errors=True)
    dir_path = str(tmp_path / "dataset")
    SyntheticDataGenerator(config).write_to_directory(dir_path)

    dataset = load_dataset(dir_path)
    output = run_b0(dataset)

    assert len(output.predictions) > 0
    assert len(output.bank_links) > 0

    classes_emitted = {p.predicted_class for p in output.predictions}
    # B0 only emits MISSING_SETTLEMENT, ORPHAN_BANK_CREDIT, UNKNOWN
    assert classes_emitted.issubset({"MISSING_SETTLEMENT", "ORPHAN_BANK_CREDIT", "UNKNOWN"})

    # Check that bank links match valid statement IDs and settlement IDs
    all_stmt_ids = {b.statement_id for b in dataset.bank_statement}
    all_setl_ids = {s.settlement_id for s in dataset.settlements}
    for link in output.bank_links:
        assert link.statement_id in all_stmt_ids
        assert link.settlement_id in all_setl_ids

