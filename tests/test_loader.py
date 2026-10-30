"""Unit tests for the observable dataset loader."""

import os
import pytest

from src.recon_agent.pipeline.loader import load_dataset
from src.recon_agent.synth.config import create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator


def test_load_dataset_success(tmp_path) -> None:
    """Loader successfully parses observable files into frozen dataclasses with integers only."""
    config = create_config(profile="small", split="dev", seed=42, inject_errors=True)
    generator = SyntheticDataGenerator(config)
    dir_path = str(tmp_path / "dataset")
    generator.write_to_directory(dir_path)

    dataset = load_dataset(dir_path)

    assert len(dataset.orders_ledger) > 0
    assert len(dataset.payments) > 0
    assert len(dataset.recon_items) > 0
    assert len(dataset.settlements) > 0
    assert len(dataset.bank_statement) > 0
    assert dataset.as_of.startswith("2026-")

    # Check types and no floats
    for p in dataset.payments:
        assert isinstance(p.amount, int)
        assert not isinstance(p.amount, float)
    for b in dataset.bank_statement:
        assert isinstance(b.credit, int)
        assert isinstance(b.debit, int)
        assert isinstance(b.balance, int)
        assert not isinstance(b.credit, float)

    # Immutability check
    with pytest.raises(Exception):
        dataset.payments[0].amount = 999  # type: ignore[misc]


def test_load_dataset_missing_file_raises(tmp_path) -> None:
    """Missing observable file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path / "non_existent"))

