"""Invariant tests (I1 through I8) for clean synthetic data generator."""

import ast
import datetime
import hashlib
import json
import os
import re
from typing import Any

import pytest

from src.recon_agent.synth.config import BASE62_ALPHABET, create_config
from src.recon_agent.synth.generator import IST_TZ, SyntheticDataGenerator
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


# =========================================================================
# Day 3A Invariant Tests: Error Injection, Disjointness, and Labels
# =========================================================================

REQUIRED_ERROR_CLASSES = [
    "DUPLICATE_PAYMENT",
    "MISSING_SETTLEMENT",
    "FEE_TAX_MISMATCH",
    "PARTIAL_REFUND_MISMATCH",
    "CUTOFF_DRIFT",
    "SHORT_BANK_CREDIT",
    "ORPHAN_BANK_CREDIT",
]

REQUIRED_HARD_NEGATIVES = [
    "ROUNDING",
    "LEGIT_ADJUSTMENT",
    "LATE_SETTLEMENT",
    "RETRY_AFTER_FAIL",
    "NOISY_BANK_NARRATION",
    "NO_UTR_NARRATION",
    "UNRELATED_BANK_CREDIT",
    "FAILED_PAYMENT",
    "PARTIAL_REFUND_CONSISTENT",
]


def test_injected_determinism_by_seed() -> None:
    """Injection ON: identical seed yields bitwise identical data and labels; different seed diverges."""
    cfg1 = create_config("small", "dev", seed=777, inject_errors=True)
    cfg2 = create_config("small", "dev", seed=777, inject_errors=True)
    cfg3 = create_config("small", "dev", seed=888, inject_errors=True)

    data1 = SyntheticDataGenerator(cfg1).generate()
    data2 = SyntheticDataGenerator(cfg2).generate()
    data3 = SyntheticDataGenerator(cfg3).generate()

    assert json.dumps(data1, sort_keys=True) == json.dumps(data2, sort_keys=True)
    assert json.dumps(data1, sort_keys=True) != json.dumps(data3, sort_keys=True)


def test_injected_entity_level_disjointness() -> None:
    """Each entity gets at most one error class; no entity ID appears in >1 error class."""
    config = create_config("small", "dev", seed=101, inject_errors=True)
    data = SyntheticDataGenerator(config).generate()
    labels = data["labels"]

    entities_by_class: dict[str, set[str]] = {}
    for lbl in labels:
        err_cls = lbl["error_class"]
        if err_cls:
            entities_by_class.setdefault(err_cls, set())
            for eid in lbl["entity_ids"]:
                # Exclude batch settlement IDs or generic statement IDs from entity-level check
                if eid.startswith(("pay_", "order_", "rfnd_")):
                    entities_by_class[err_cls].add(eid)

    # Check pairwise disjointness
    classes = list(entities_by_class.keys())
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            c1 = classes[i]
            c2 = classes[j]
            overlap = entities_by_class[c1].intersection(entities_by_class[c2])
            assert not overlap, f"Entity overlap detected between {c1} and {c2}: {overlap}"


def test_injected_error_classes_spot_checks() -> None:
    """Spot check verification for each of the 7 injected error classes."""
    config = create_config("small", "dev", seed=101, inject_errors=True)
    data = SyntheticDataGenerator(config).generate()

    payments_map = {p["payment_id"]: p for p in data["payments"]}
    recon_map = {r["entity_id"]: r for r in data["recon_items"]}
    orders_map = {o["order_id"]: o for o in data["orders_ledger"]}
    settlements_map = {s["settlement_id"]: s for s in data["settlements"]}
    bank_rows = data["bank_statement"]

    labels_by_class: dict[str, list[dict[str, Any]]] = {}
    for lbl in data["labels"]:
        if lbl["error_class"]:
            labels_by_class.setdefault(lbl["error_class"], []).append(lbl)

    # Verify all 7 error classes are present
    for err_cls in REQUIRED_ERROR_CLASSES:
        assert len(labels_by_class.get(err_cls, [])) > 0, f"Error class {err_cls} missing in generated labels"

    # 1. DUPLICATE_PAYMENT spot-check
    dup_case = labels_by_class["DUPLICATE_PAYMENT"][0]
    p1_id, p2_id, ord_id = dup_case["entity_ids"][:3]
    p1 = payments_map[p1_id]
    p2 = payments_map[p2_id]
    assert p1["order_id"] == p2["order_id"] == ord_id
    assert p1["amount"] == p2["amount"]
    assert 1 <= p2["created_at"] - p1["created_at"] <= 300
    assert p1_id in recon_map and p2_id in recon_map
    assert recon_map[p1_id]["settlement_id"] == recon_map[p2_id]["settlement_id"]
    assert dup_case["expected_action"] == "flag_duplicate_payment"

    # 2. MISSING_SETTLEMENT spot-check
    missing_case = labels_by_class["MISSING_SETTLEMENT"][0]
    m_pid = missing_case["entity_ids"][0]
    assert m_pid in payments_map
    assert payments_map[m_pid]["status"] == "captured"
    assert m_pid not in recon_map
    assert missing_case["expected_action"] == "flag_missing_settlement"

    # 3. FEE_TAX_MISMATCH spot-check
    feetax_case = labels_by_class["FEE_TAX_MISMATCH"][0]
    ft_eid = feetax_case["entity_ids"][0]
    recon_item = recon_map[ft_eid]
    method = recon_item["method"]
    rate_bps = config.method_fee_bps[method]
    expected_base_fee = mul_div_round(recon_item["amount"], rate_bps, 10000)
    expected_tax = mul_div_round(expected_base_fee, config.gst_bps, 10000)
    expected_total_fee = expected_base_fee + expected_tax if config.fee_includes_tax else expected_base_fee
    assert recon_item["fee"] != expected_total_fee or recon_item["tax"] != expected_tax
    assert feetax_case["expected_action"] == "flag_fee_tax_mismatch"

    # 4. PARTIAL_REFUND_MISMATCH spot-check
    refund_case = labels_by_class["PARTIAL_REFUND_MISMATCH"][0]
    rfnd_id = refund_case["entity_ids"][0]
    rfnd_recon = recon_map[rfnd_id]
    ord_obj = orders_map[rfnd_recon["order_id"]]
    p_orig = payments_map[rfnd_recon["payment_id"]]
    assert 1 <= rfnd_recon["amount"] <= p_orig["amount"]
    assert (abs(rfnd_recon["debit"] - ord_obj["refund_amount"]) >= 100) or (ord_obj["refund_id"] == "")
    assert refund_case["expected_action"] == "flag_partial_refund_mismatch"

    # 5. CUTOFF_DRIFT spot-check
    cutoff_case = labels_by_class["CUTOFF_DRIFT"][0]
    c_pid, c_oid = cutoff_case["entity_ids"]
    c_pay = payments_map[c_pid]
    c_ord = orders_map[c_oid]
    assert c_ord["booked_date"] != ""
    assert c_pay["captured_at"] > c_pay["created_at"]
    assert cutoff_case["expected_action"] == "flag_cutoff_drift"

    # 6. SHORT_BANK_CREDIT spot-check
    short_case = labels_by_class["SHORT_BANK_CREDIT"][0]
    short_sid = short_case["settlement_id"]
    stl = settlements_map[short_sid]
    matching_bk = [
        b
        for b in bank_rows
        if b["merchant_id"] == stl["merchant_id"]
        and (stl["utr"] in b["description"] or stl["utr"].lower() in b["description"].lower() or stl["utr"][:12] in b["description"])
    ]
    assert matching_bk, "Matching bank row not found for short credit settlement"
    assert matching_bk[0]["credit"] < stl["amount"]
    assert short_case["expected_action"] == "flag_short_bank_credit"

    # 7. ORPHAN_BANK_CREDIT spot-check
    orphan_case = labels_by_class["ORPHAN_BANK_CREDIT"][0]
    orphan_stmt_id = orphan_case["entity_ids"][0]
    orphan_bk = next(b for b in bank_rows if b["statement_id"] == orphan_stmt_id)
    all_settlement_utrs = {s["utr"] for s in settlements_map.values()}
    assert not any(utr in orphan_bk["description"] for utr in all_settlement_utrs)
    assert orphan_case["expected_action"] == "flag_orphan_bank_credit"


def test_injected_i2_conservation_except_short_bank_credit() -> None:
    """Settlement conservation I2 holds for all batches except SHORT_BANK_CREDIT."""
    config = create_config("small", "dev", seed=202, inject_errors=True)
    data = SyntheticDataGenerator(config).generate()

    short_credit_settlement_ids = {
        lbl["settlement_id"]
        for lbl in data["labels"]
        if lbl["error_class"] == "SHORT_BANK_CREDIT"
    }

    # Group recon items by settlement_id
    recon_by_batch: dict[str, list[dict[str, Any]]] = {}
    for r in data["recon_items"]:
        recon_by_batch.setdefault(r["settlement_id"], []).append(r)

    # Bank rows by settlement UTR and date
    for stl in data["settlements"]:
        sid = stl["settlement_id"]
        stl_date = datetime.datetime.fromtimestamp(stl["settled_at"], tz=IST_TZ).date().isoformat()
        batch_items = recon_by_batch.get(sid, [])
        recon_net = sum(item["credit"] for item in batch_items) - sum(
            item["debit"] for item in batch_items
        )

        # 1. Recon net must equal settlement amount for ALL batches
        assert recon_net == stl["amount"], f"Recon net != settlement amount for {sid}"

        # 2. Bank credit equals settlement amount EXCEPT for SHORT_BANK_CREDIT batches
        exact_utr_matches = [
            b
            for b in data["bank_statement"]
            if b["merchant_id"] == stl["merchant_id"]
            and b["date"] == stl_date
            and (
                stl["utr"] in b["description"]
                or stl["utr"].lower() in b["description"].lower()
                or stl["utr"][:12] in b["description"]
            )
            and b["credit"] > 0
        ]
        if exact_utr_matches:
            matching_bk = exact_utr_matches
        else:
            matching_bk = [
                b
                for b in data["bank_statement"]
                if b["merchant_id"] == stl["merchant_id"]
                and b["date"] == stl_date
                and "RAZORPAY PAYMENT SERVICES" in b["description"]
                and (b["credit"] == stl["amount"] or sid in short_credit_settlement_ids)
            ]

        assert matching_bk, f"No matching bank row for settlement {sid}"
        bk_credit = matching_bk[0]["credit"]

        if sid in short_credit_settlement_ids:
            assert bk_credit < stl["amount"], f"SHORT_BANK_CREDIT expected for {sid}"
        else:
            assert bk_credit == stl["amount"], f"I2 bank conservation failed for {sid}"


def test_full_profile_volume_floors_dev_and_test() -> None:
    """Full profile meets floors: >= 50 for every error class, >= 30 for every hard negative."""
    for split, seed in [("dev", 1), ("test", 9999)]:
        config = create_config("full", split, seed=seed, inject_errors=True)
        data = SyntheticDataGenerator(config).generate()
        counts = data["manifest"]["counts"]
        err_counts = counts.get("error_class_counts", {})
        hn_counts = counts.get("hard_negative_counts", {})

        for err_name in REQUIRED_ERROR_CLASSES:
            cnt = err_counts.get(err_name, 0)
            assert cnt >= 50, f"[{split}] Error class {err_name} count {cnt} < 50 floor"

        for hn_name in REQUIRED_HARD_NEGATIVES:
            cnt = hn_counts.get(hn_name, 0)
            assert cnt >= 30, f"[{split}] Hard negative {hn_name} count {cnt} < 30 floor"


def test_labels_filename_boundary() -> None:
    """Verify literal string 'labels.jsonl' does not appear outside synth/, tests/, scripts/, and docs/."""
    src_dir = os.path.join(REPO_ROOT, "src")
    synth_dir = os.path.abspath(os.path.join(src_dir, "recon_agent", "synth"))

    violations: list[str] = []
    for root, _, files in os.walk(src_dir):
        if os.path.abspath(root).startswith(synth_dir):
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "labels.jsonl" in content:
                        violations.append(path)

    assert not violations, (
        f"Pipeline code outside synth/ references 'labels.jsonl': {violations}"
    )


def test_statement_ids_strictly_increase_per_merchant() -> None:
    """Statement IDs strictly increase with (date, row order) per merchant."""
    for inject_errors in [False, True]:
        config = create_config("small", "dev", seed=42, inject_errors=inject_errors)
        data = SyntheticDataGenerator(config).generate()
        rows_by_merchant: dict[str, list[dict[str, Any]]] = {}
        for r in data["bank_statement"]:
            rows_by_merchant.setdefault(r["merchant_id"], []).append(r)

        for m_id, rows in rows_by_merchant.items():
            assert len(rows) > 0, f"No bank statement rows for {m_id}"
            for i in range(1, len(rows)):
                assert rows[i]["date"] >= rows[i - 1]["date"], f"Date decreased for {m_id} at row {i}"
                curr_id = int(rows[i]["statement_id"].split("_")[1])
                prev_id = int(rows[i - 1]["statement_id"].split("_")[1])
                assert curr_id > prev_id, (
                    f"Statement ID did not strictly increase for {m_id}: "
                    f"{rows[i - 1]['statement_id']} -> {rows[i]['statement_id']}"
                )


def test_bank_balances_continuous() -> None:
    """Final balances are continuous: balance[k] == balance[k-1] + credit[k] - debit[k]."""
    for inject_errors in [False, True]:
        config = create_config("small", "dev", seed=42, inject_errors=inject_errors)
        data = SyntheticDataGenerator(config).generate()
        rows_by_merchant: dict[str, list[dict[str, Any]]] = {}
        for r in data["bank_statement"]:
            rows_by_merchant.setdefault(r["merchant_id"], []).append(r)

        for m_id, rows in rows_by_merchant.items():
            for i in range(1, len(rows)):
                expected = rows[i - 1]["balance"] + rows[i]["credit"] - rows[i]["debit"]
                assert rows[i]["balance"] == expected, (
                    f"Discontinuous bank balance for {m_id} at row {i}: "
                    f"expected {expected}, got {rows[i]['balance']}"
                )


def test_failed_payments_satisfy_i4() -> None:
    """Every failed payment satisfies I4: p['amount'] == order['expected_amount']."""
    for inject_errors in [False, True]:
        config = create_config("small", "dev", seed=42, inject_errors=inject_errors)
        data = SyntheticDataGenerator(config).generate()
        orders_map = {o["order_id"]: o for o in data["orders_ledger"]}
        failed_payments = [p for p in data["payments"] if p["status"] == "failed"]
        assert len(failed_payments) > 0
        for p in failed_payments:
            order = orders_map[p["order_id"]]
            assert p["amount"] == order["expected_amount"], (
                f"Failed payment amount mismatch for {p['payment_id']}: "
                f"payment {p['amount']} != order expected {order['expected_amount']}"
            )


def test_every_refund_le_payment_amount() -> None:
    """Every refund in recon items is between 1 paise and the original payment amount."""
    for inject_errors in [False, True]:
        config = create_config("small", "dev", seed=42, inject_errors=inject_errors)
        data = SyntheticDataGenerator(config).generate()
        payments_map = {p["payment_id"]: p for p in data["payments"]}
        refund_items = [r for r in data["recon_items"] if r["type"] == "refund"]
        assert len(refund_items) > 0
        for r in refund_items:
            p = payments_map[r["payment_id"]]
            assert 1 <= r["amount"] <= p["amount"], (
                f"Refund amount {r['amount']} not in [1, {p['amount']}] for {r['entity_id']}"
            )
            assert 1 <= r["debit"] <= p["amount"], (
                f"Refund debit {r['debit']} not in [1, {p['amount']}] for {r['entity_id']}"
            )


def test_utrs_unique_across_dataset() -> None:
    """All settlement UTRs are strictly unique across the dataset."""
    for inject_errors in [False, True]:
        config = create_config("small", "dev", seed=42, inject_errors=inject_errors)
        data = SyntheticDataGenerator(config).generate()
        settlement_utrs = [s["utr"] for s in data["settlements"]]
        assert len(settlement_utrs) == len(set(settlement_utrs)), "Settlement UTRs are not unique"
        recon_utrs = [r["settlement_utr"] for r in data["recon_items"] if r["settlement_utr"]]
        assert set(recon_utrs).issubset(set(settlement_utrs)), "Recon UTR not found in settlements"


# =========================================================================
# Day 3B Invariant Tests: Boundary Checks & Twin Dataset Byte Identity
# =========================================================================


def test_injection_manifest_boundary() -> None:
    """Verify 'injection_manifest.jsonl' never appears in code outside synth/, tests/, scripts/, docs/."""
    allowed_dirs = {"synth", "tests", "scripts", "docs", ".gemini", ".git", ".pytest_cache", ".venv"}
    violations = []

    for root, dirs, files in os.walk(REPO_ROOT):
        # Exclude hidden or build directories
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__", ".pytest_cache"}]

        rel_root = os.path.relpath(root, REPO_ROOT)
        top_dir = rel_root.split(os.sep)[0] if rel_root != "." else "."

        # Check if inside allowed path
        norm_rel = rel_root.replace("\\", "/")
        is_allowed = (
            any(norm_rel.startswith(ad) or norm_rel == ad for ad in allowed_dirs)
            or "src/recon_agent/synth" in norm_rel
        )

        if not is_allowed:
            for file_name in files:
                if file_name.endswith((".py", ".json", ".md", ".sh", ".ps1")):
                    f_path = os.path.join(root, file_name)
                    with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if "injection_manifest.jsonl" in content:
                            violations.append(f_path)

    assert not violations, (
        "Boundary violation: 'injection_manifest.jsonl' detected outside permitted synth/tests/scripts/docs paths:\n"
        + "\n".join(violations)
    )


def test_twin_dataset_byte_identity_except_free_text(tmp_path: Any) -> None:
    """Twin dataset in <split>/injected/ is byte-identical to base except recon_items.description and notes."""
    from src.recon_agent.synth.text_inject import (
        REQUIRED_ATTACK_TYPES,
        apply_text_injections,
        write_twin_dataset,
    )

    config = create_config("small", "dev", seed=1, inject_errors=True)
    generator = SyntheticDataGenerator(config)
    data = generator.generate()

    # Write base dataset
    base_dir = str(tmp_path / "base")
    generator.write_to_directory(base_dir, data=data)

    # Apply text injection and write twin dataset
    twin_data, inj_manifest = apply_text_injections(data, seed=config.seed)
    injected_dir = str(tmp_path / "base" / "injected")
    write_twin_dataset(twin_data, inj_manifest, injected_dir)

    # 1. Compare auxiliary files byte-for-byte
    identical_files = [
        "payments.jsonl",
        "settlements.jsonl",
        "orders_ledger.csv",
        "bank_statement.csv",
        "rate_card.json",
        "manifest.json",
        "labels.jsonl",
    ]
    for fname in identical_files:
        base_f = os.path.join(base_dir, fname)
        twin_f = os.path.join(injected_dir, fname)
        assert os.path.exists(twin_f), f"Missing twin file: {fname}"
        with open(base_f, "rb") as f1, open(twin_f, "rb") as f2:
            assert f1.read() == f2.read(), f"File {fname} is not byte-identical in twin dataset!"

    # 2. Compare recon_items.jsonl line-by-line
    base_recon = []
    with open(os.path.join(base_dir, "recon_items.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            base_recon.append(json.loads(line))

    twin_recon = []
    with open(os.path.join(injected_dir, "recon_items.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            twin_recon.append(json.loads(line))

    assert len(base_recon) == len(twin_recon), "recon_items count mismatch between base and twin"

    attacked_entity_ids = {entry["item_entity_id"] for entry in inj_manifest}
    attack_count = 0

    for b_item, t_item in zip(base_recon, twin_recon):
        eid = b_item["entity_id"]
        assert eid == t_item["entity_id"]

        # Check all non-free-text fields are strictly identical
        for k in b_item:
            if k not in ("description", "notes"):
                assert b_item[k] == t_item[k], f"Non-free-text field '{k}' mutated on {eid}!"

        # Check free-text fields
        if eid in attacked_entity_ids:
            attack_count += 1
            # Either description or notes mutated
            assert (b_item["description"] != t_item["description"]) or (b_item["notes"] != t_item["notes"]), (
                f"Attacked item {eid} was not mutated in free text fields"
            )
        else:
            assert b_item["description"] == t_item["description"], f"Unattacked description mutated on {eid}"
            assert b_item["notes"] == t_item["notes"], f"Unattacked notes mutated on {eid}"

    assert attack_count == len(inj_manifest), "Mismatch in applied attack counts"

    # 3. Verify injection_manifest.jsonl presence and schema
    manifest_f = os.path.join(injected_dir, "injection_manifest.jsonl")
    assert os.path.exists(manifest_f), "Missing injection_manifest.jsonl in injected/"
    manifest_rows = []
    with open(manifest_f, "r", encoding="utf-8") as f:
        for line in f:
            manifest_rows.append(json.loads(line))

    assert len(manifest_rows) == len(inj_manifest)
    observed_attack_types = set()
    for row in manifest_rows:
        for req_k in ("case_id", "item_entity_id", "field", "attack_type", "template_id"):
            assert req_k in row, f"Missing required key {req_k} in injection_manifest row"
        observed_attack_types.add(row["attack_type"])

    assert observed_attack_types == set(REQUIRED_ATTACK_TYPES)



