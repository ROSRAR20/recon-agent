"""Schema conformance tests for all synthetic dataset outputs against docs/schemas.json."""

import csv
import datetime
import json
import os
from typing import Any

import pytest

from src.recon_agent.synth.config import create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCHEMAS_PATH = os.path.join(REPO_ROOT, "docs", "schemas.json")


def load_schemas() -> dict[str, Any]:
    with open(SCHEMAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_type(val: Any, allowed_types: list[str], field_name: str) -> None:
    matched = False
    assert not isinstance(val, float), f"Field {field_name} must never be a float, got: {val}"

    for t in allowed_types:
        if t == "int":
            if type(val) is int:
                matched = True
                break
        elif t == "bool":
            if type(val) is bool:
                matched = True
                break
        elif t == "str":
            if isinstance(val, str):
                matched = True
                break
        elif t == "dict":
            if isinstance(val, dict):
                matched = True
                break
        elif t == "list":
            if isinstance(val, list):
                matched = True
                break
        elif t == "NoneType":
            if val is None:
                matched = True
                break
        else:
            raise ValueError(f"Unknown type specifier: {t}")

    assert matched, (
        f"Field '{field_name}' value {val!r} (type: {type(val).__name__}) "
        f"does not match allowed types: {allowed_types}"
    )


def test_schema_file_exists() -> None:
    """Verify docs/schemas.json exists and defines all 8 dataset files."""
    schemas = load_schemas()
    expected_files = {
        "recon_items.jsonl",
        "payments.jsonl",
        "orders_ledger.csv",
        "settlements.jsonl",
        "bank_statement.csv",
        "rate_card.json",
        "manifest.json",
        "labels.jsonl",
    }
    assert set(schemas.keys()) == expected_files


@pytest.mark.parametrize("inject_errors", [False, True])
def test_schema_conformance(tmp_path: Any, inject_errors: bool) -> None:
    """Validate all generated files conform to schemas.json for both clean and injected modes."""
    schemas = load_schemas()
    config = create_config("small", "dev", seed=42, inject_errors=inject_errors)
    gen = SyntheticDataGenerator(config)
    paths = gen.write_to_directory(str(tmp_path))

    # 1. recon_items.jsonl
    schema = schemas["recon_items.jsonl"]
    with open(paths["recon_items"], "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) > 0
    for item in lines:
        for req in schema["required_keys"]:
            assert req in item, f"Missing required key {req} in recon_item"
        for field, allowed in schema["types"].items():
            validate_type(item[field], allowed, f"recon_item.{field}")

    # 2. payments.jsonl
    schema = schemas["payments.jsonl"]
    with open(paths["payments"], "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) > 0
    for item in lines:
        for req in schema["required_keys"]:
            assert req in item, f"Missing required key {req} in payment"
        for field, allowed in schema["types"].items():
            validate_type(item[field], allowed, f"payment.{field}")

    # 3. settlements.jsonl
    schema = schemas["settlements.jsonl"]
    with open(paths["settlements"], "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) > 0
    for item in lines:
        for req in schema["required_keys"]:
            assert req in item, f"Missing required key {req} in settlement"
        for field, allowed in schema["types"].items():
            validate_type(item[field], allowed, f"settlement.{field}")

    # 4. orders_ledger.csv
    schema = schemas["orders_ledger.csv"]
    with open(paths["orders_ledger"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == schema["required_columns"]
        rows = list(reader)
    assert len(rows) > 0
    for r in rows:
        assert datetime.date.fromisoformat(r["booked_date"])
        assert int(r["created_at"]) >= 0
        assert int(r["expected_amount"]) > 0
        assert "." not in r["expected_amount"]
        if r["refund_amount"]:
            assert int(r["refund_amount"]) >= 0
            assert "." not in r["refund_amount"]

    # 5. bank_statement.csv
    schema = schemas["bank_statement.csv"]
    with open(paths["bank_statement"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == schema["required_columns"]
        rows = list(reader)
    assert len(rows) > 0
    for r in rows:
        assert datetime.date.fromisoformat(r["date"])
        assert int(r["debit"]) >= 0
        assert int(r["credit"]) >= 0
        assert int(r["balance"]) >= 0
        assert "." not in r["debit"]
        assert "." not in r["credit"]
        assert "." not in r["balance"]

    # 6. rate_card.json
    schema = schemas["rate_card.json"]
    with open(paths["rate_card"], "r", encoding="utf-8") as f:
        rate_card = json.load(f)
    for req in schema["required_keys"]:
        assert req in rate_card, f"Missing required key {req} in rate_card"
    for field, allowed in schema["types"].items():
        validate_type(rate_card[field], allowed, f"rate_card.{field}")

    # 7. manifest.json
    schema = schemas["manifest.json"]
    with open(paths["manifest"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for req in schema["required_keys"]:
        assert req in manifest, f"Missing required key {req} in manifest"
    for field, allowed in schema["types"].items():
        validate_type(manifest[field], allowed, f"manifest.{field}")
    # Verify as_of is valid ISO date
    assert datetime.date.fromisoformat(manifest["as_of"])

    # 8. labels.jsonl
    if inject_errors:
        assert "labels" in paths
        schema = schemas["labels.jsonl"]
        with open(paths["labels"], "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0
        for item in lines:
            for req in schema["required_keys"]:
                assert req in item, f"Missing required key {req} in label"
            for field, allowed in schema["types"].items():
                validate_type(item[field], allowed, f"label.{field}")
    else:
        assert "labels" not in paths
