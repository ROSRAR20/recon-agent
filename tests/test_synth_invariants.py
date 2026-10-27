"""Invariant tests (I1 through I8) for clean synthetic data generator."""

import ast
import hashlib
import json
import os
import re
from typing import Any

import pytest

from src.recon_agent.synth.config import BASE62_ALPHABET, create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator
from src.recon_agent.synth.math_utils import mul_div_round

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEEDS = [1, 2, 3, 42]
FEE_MODES = [True, False]


def _assert_no_floats(obj: Any, path: str = "root") -> None:
    """Recursively assert that an object contains strictly no float values."""
    if isinstance(obj, float):
        raise AssertionError(f"Float detected at {path}: {obj}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _assert_no_floats(item, f"{path}[{idx}]")


# =========================================================================
# I1: Per-Item Arithmetic
# =========================================================================


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fee_includes_tax", FEE_MODES)
def test_invariant_i1_per_item_arithmetic(seed: int, fee_includes_tax: bool) -> None:
    """I1: per-item arithmetic for credit, debit, fee, and tax."""
    config = create_config("small", "dev", seed=seed, fee_includes_tax=fee_includes_tax)
    gen = SyntheticDataGenerator(config)
    data = gen.generate()

    for item in data["recon_items"]:
        item_type = item["type"]
        amount = item["amount"]
        fee = item["fee"]
        tax = item["tax"]
        credit = item["credit"]
        debit = item["debit"]

        assert isinstance(amount, int) and amount > 0
        assert isinstance(fee, int) and fee >= 0
        assert isinstance(tax, int) and tax >= 0
        assert isinstance(credit, int) and credit >= 0
        assert isinstance(debit, int) and debit >= 0

        if item_type == "payment":
            assert debit == 0
            method = item["method"]
            rate_bps = config.method_fee_bps[method]
            base_fee = mul_div_round(amount, rate_bps, 10000)
            expected_tax = mul_div_round(base_fee, config.gst_bps, 10000)

            assert tax == expected_tax

            if fee_includes_tax:
                assert fee == base_fee + expected_tax
                assert credit == amount - fee
            else:
                assert fee == base_fee
                assert credit == amount - fee - tax

        elif item_type == "refund":
            assert credit == 0
            assert fee == 0
            assert tax == 0
            assert debit == amount
            assert item["payment_id"] is not None and item["payment_id"].startswith("pay_")

        elif item_type == "adjustment":
            assert fee == 0
            assert tax == 0
            assert credit == amount or debit == amount


# =========================================================================
# I2: Settlement Conservation & Bank Credit
# =========================================================================


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fee_includes_tax", FEE_MODES)
def test_invariant_i2_settlement_net_and_bank_credit(
    seed: int, fee_includes_tax: bool
) -> None:
    """I2: sum(credit) - sum(debit) == settlement net == bank credit amount."""
    config = create_config("small", "dev", seed=seed, fee_includes_tax=fee_includes_tax)
    gen = SyntheticDataGenerator(config)
    data = gen.generate()

    # Index recon items by settlement_id
    batch_items: dict[str, list[dict[str, Any]]] = {}
    for item in data["recon_items"]:
        batch_items.setdefault(item["settlement_id"], []).append(item)

    # Index bank settlement credits by merchant and amount
    bank_credits = [
        row for row in data["bank_statement"] if "razorpay" in row["description"].lower()
    ]

    for stl in data["settlements"]:
        stl_id = stl["settlement_id"]
        stl_net = stl["amount"]
        assert stl_id in batch_items

        items = batch_items[stl_id]
        total_credit = sum(it["credit"] for it in items)
        total_debit = sum(it["debit"] for it in items)

        # sum(credit) - sum(debit) == settlement net
        assert total_credit - total_debit == stl_net

        # Exactly matches the bank credit amount for this settlement
        matching_bank_rows = [
            b
            for b in bank_credits
            if b["merchant_id"] == stl["merchant_id"]
            and b["credit"] == stl_net
            and (
                stl["utr"] in b["description"]
                or stl["utr"].lower() in b["description"].lower()
                or stl["utr"][:12] in b["description"]
                or "RAZORPAY PAYMENT SERVICES" in b["description"]
            )
        ]
        assert len(matching_bank_rows) >= 1


# =========================================================================
# I3: Referential Integrity
# =========================================================================


@pytest.mark.parametrize("seed", SEEDS)
def test_invariant_i3_referential_integrity(seed: int) -> None:
    """I3: captured payment has exactly one recon item; valid settlement_id."""
    config = create_config("small", "dev", seed=seed)
    gen = SyntheticDataGenerator(config)
    data = gen.generate()

    settlement_ids = {s["settlement_id"] for s in data["settlements"]}
    payment_recon_map: dict[str, list[dict[str, Any]]] = {}

    for item in data["recon_items"]:
        assert item["settlement_id"] in settlement_ids
        if item["type"] == "payment":
            payment_recon_map.setdefault(item["entity_id"], []).append(item)

    for p in data["payments"]:
        pid = p["payment_id"]
        if p["status"] == "captured":
            # Exactly one recon payment item
            assert len(payment_recon_map.get(pid, [])) == 1
        elif p["status"] == "failed":
            # Failed payment never enters recon
            assert pid not in payment_recon_map


# =========================================================================
# I4: Ledger Consistency
# =========================================================================


@pytest.mark.parametrize("seed", SEEDS)
def test_invariant_i4_ledger_consistency(seed: int) -> None:
    """I4: ledger expected_amount == payment amount; refunds match ledger."""
    config = create_config("small", "dev", seed=seed)
    gen = SyntheticDataGenerator(config)
    data = gen.generate()

    orders_map = {o["order_id"]: o for o in data["orders_ledger"]}

    for p in data["payments"]:
        if p["status"] == "captured":
            order = orders_map[p["order_id"]]
            assert order["expected_amount"] == p["amount"]

    for r in data["recon_items"]:
        if r["type"] == "refund":
            order = orders_map[r["order_id"]]
            assert order["refund_id"] == r["entity_id"]
            assert order["refund_amount"] == r["amount"]


# =========================================================================
# I5: Type Hygiene & ID Structure
# =========================================================================


@pytest.mark.parametrize("seed", SEEDS)
def test_invariant_i5_no_floats_and_valid_unique_ids(seed: int) -> None:
    """I5: strictly no floats in JSON output; IDs are unique and well-formed."""
    config = create_config("small", "dev", seed=seed)
    gen = SyntheticDataGenerator(config)
    data = gen.generate()

    # 1. No floats
    _assert_no_floats(data["recon_items"])
    _assert_no_floats(data["payments"])
    _assert_no_floats(data["settlements"])
    _assert_no_floats(data["rate_card"])
    _assert_no_floats(data["manifest"])

    # 2. ID patterns and global uniqueness
    seen_all_ids: set[str] = set()
    b62_charset = set(BASE62_ALPHABET)

    def _verify_id(identifier: str, expected_prefix: str) -> None:
        assert identifier.startswith(expected_prefix)
        token = identifier[len(expected_prefix) :]
        assert len(token) == 14
        assert all(c in b62_charset for c in token)
        assert identifier not in seen_all_ids
        seen_all_ids.add(identifier)

    for p in data["payments"]:
        _verify_id(p["payment_id"], "pay_")

    for o in data["orders_ledger"]:
        _verify_id(o["order_id"], "order_")

    for s in data["settlements"]:
        _verify_id(s["settlement_id"], "setl_")
        assert len(s["utr"]) == 16

    for r in data["recon_items"]:
        if r["type"] == "refund":
            _verify_id(r["entity_id"], "rfnd_")
        elif r["type"] == "adjustment":
            _verify_id(r["entity_id"], "adj_")


# =========================================================================
# I6: Determinism
# =========================================================================


def test_invariant_i6_determinism_and_divergence(tmp_path) -> None:
    """I6: same seed gives byte-identical files; different seeds differ."""
    config_a1 = create_config("small", "dev", seed=42)
    config_a2 = create_config("small", "dev", seed=42)
    config_b = create_config("small", "dev", seed=43)

    dir_a1 = str(tmp_path / "a1")
    dir_a2 = str(tmp_path / "a2")
    dir_b = str(tmp_path / "b")

    SyntheticDataGenerator(config_a1).write_to_directory(dir_a1)
    SyntheticDataGenerator(config_a2).write_to_directory(dir_a2)
    SyntheticDataGenerator(config_b).write_to_directory(dir_b)

    file_names = [
        "recon_items.jsonl",
        "payments.jsonl",
        "orders_ledger.csv",
        "settlements.jsonl",
        "bank_statement.csv",
        "manifest.json",
        "rate_card.json",
    ]

    for fname in file_names:
        content_a1 = (tmp_path / "a1" / fname).read_bytes()
        content_a2 = (tmp_path / "a2" / fname).read_bytes()
        content_b = (tmp_path / "b" / fname).read_bytes()

        # Identical runs with same seed
        assert hashlib.sha256(content_a1).hexdigest() == hashlib.sha256(content_a2).hexdigest()

        # Divergent runs with different seeds (except static rate_card)
        if fname != "rate_card.json":
            assert hashlib.sha256(content_a1).hexdigest() != hashlib.sha256(content_b).hexdigest()


# =========================================================================
# I7: Import Boundary
# =========================================================================


def test_invariant_i7_import_boundary() -> None:
    """I7: no module outside synth/, tests/, and scripts/generate.py imports recon_agent.synth."""
    src_dir = os.path.join(REPO_ROOT, "src")
    scripts_dir = os.path.join(REPO_ROOT, "scripts")

    allowed_paths = [
        os.path.abspath(os.path.join(src_dir, "recon_agent", "synth")),
        os.path.abspath(os.path.join(REPO_ROOT, "tests")),
        os.path.abspath(os.path.join(scripts_dir, "generate.py")),
    ]

    violations: list[str] = []

    for search_root in (src_dir, scripts_dir):
        for root, _, files in os.walk(search_root):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.abspath(os.path.join(root, file))

                # Check if file is permitted
                is_permitted = any(
                    file_path.startswith(allowed) or file_path == allowed
                    for allowed in allowed_paths
                )
                if is_permitted:
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if "synth" in alias.name:
                                violations.append(f"{file_path}: import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and "synth" in node.module:
                            violations.append(f"{file_path}: from {node.module} import ...")

    assert not violations, (
        "Architectural violation: unauthorized imports of recon_agent.synth detected:\n"
        + "\n".join(violations)
    )


# =========================================================================
# I8: Positive Batch Net
# =========================================================================


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fee_includes_tax", FEE_MODES)
def test_invariant_i8_positive_batch_net(seed: int, fee_includes_tax: bool) -> None:
    """I8: no batch net <= 0."""
    config = create_config("small", "dev", seed=seed, fee_includes_tax=fee_includes_tax)
    gen = SyntheticDataGenerator(config)
    data = gen.generate()

    for s in data["settlements"]:
        assert s["amount"] > 0, f"Batch net must be strictly positive: {s}"
