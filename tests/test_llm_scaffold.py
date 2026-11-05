"""Unit tests for LLM scaffold, bundle extraction, schema, verifier, cache, and audit logger.

Covers all required specifications and amendments:
- Bundle extraction, stable hashing, and no ground-truth leakage.
- Strict schema validation (valid acceptance, extra keys rejection, enum rejection, length limits).
- Verifier rules V1-V7 (clean accept, unquoted reject, bad amount reject, fabricated numbers reject, escalate pass-through).
- Amendment 3: Quote with two amounts and number-word quote triggers ABSTAIN.
- Amendment 4: Bundle IDs allowed in explanation; fabricated 4,321 rejected.
- Cache get/put hit verification.
- Audit logger privacy and formatting (no raw untrusted text).
- Boundary isolation (no synth imports, no eval file references).
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
import tempfile
import pytest
import requests
import requests.exceptions

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from src.recon_agent.llm.audit import AuditLogger
from src.recon_agent.llm.bundle import (
    CaseBundle,
    build_case_bundles,
    build_control_bundles,
    create_case_bundle,
)
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.client import MockClient
from src.recon_agent.llm.gemini_client import (
    FatalAPIError,
    GeminiClient,
    NonRetriableAPIError,
)
from src.recon_agent.llm.schema import (
    ALLOWED_CLASSES,
    ALLOWED_DECISIONS,
    LLMOutput,
    get_llm_json_schema,
    validate_llm_output,
)
from src.recon_agent.llm.verifier import (
    Verifier,
    VerifierOutcome,
    VerifierResult,
)
from src.recon_agent.pipeline.loader import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_bundle(
    entity_id: str = "rfnd_001",
    amount_paise: int = 10000,
    debit_paise: int = 8500,
    refund_diff_paise: int = -1500,
    reason_text: str = "Customer requested discount. Fee waived Rs 15.00 applied.",
    extra_facts: dict | None = None,
) -> CaseBundle:
    facts = {
        "entity_id": entity_id,
        "type": "refund",
        "amount_paise": amount_paise,
        "debit_paise": debit_paise,
        "refund_difference_paise": refund_diff_paise,
        "payment_id": "pay_Ab12Cd",
        "settlement_id": "setl_Xy98Zz",
    }
    if extra_facts:
        facts.update(extra_facts)

    untrusted_text = {
        "recon_item_description": "Refund transaction",
        "recon_item_notes_comment": None,
        "recon_item_notes_reason_text": reason_text,
        "bank_narration": None,
    }

    return create_case_bundle(
        bundle_id=f"bundle_{entity_id}",
        entity_id=entity_id,
        merchant_id="mer_test_01",
        case_type="refund",
        rule_id="B1R_EXPLAINED_NO_AMOUNT",
        facts=facts,
        untrusted_text=untrusted_text,
    )


# ---------------------------------------------------------------------------
# 1. Bundle Extraction & Properties
# ---------------------------------------------------------------------------

def test_bundle_extraction() -> None:
    """Extract bundles for dev dataset; counts match escalations + notes."""
    dev_dir = os.path.join(REPO_ROOT, "data", "synth", "full", "dev_seed1")
    if not os.path.isdir(dev_dir):
        pytest.skip(f"Dev dataset not found at {dev_dir}")

    dataset = load_dataset(dev_dir)
    bundles = build_case_bundles(dataset)

    assert len(bundles) > 0
    # Every bundle must have valid attributes and integer paise
    for b in bundles:
        assert b.bundle_id.startswith("bundle_")
        assert b.entity_id
        assert b.merchant_id
        assert b.bundle_hash
        assert len(b.bundle_hash) == 64
        assert b.allowed_decisions == ALLOWED_DECISIONS
        assert b.allowed_classes == ALLOWED_CLASSES
        for k, v in b.facts.items():
            if k.endswith("_paise"):
                assert isinstance(v, int) and not isinstance(v, bool), f"Fact {k} must be int paise: {v}"


def test_bundle_stable_hash() -> None:
    """Same input bundle yields byte-identical SHA-256 hash."""
    b1 = _make_test_bundle("rfnd_hash_test")
    b2 = _make_test_bundle("rfnd_hash_test")

    assert b1.bundle_hash == b2.bundle_hash
    assert len(b1.bundle_hash) == 64

    # Changing any fact changes the hash
    b3 = _make_test_bundle("rfnd_hash_test", debit_paise=8000)
    assert b1.bundle_hash != b3.bundle_hash


def test_bundle_no_leakage() -> None:
    """Bundle contains NO label, truth link, or injection manifest fields."""
    forbidden_keys = {
        "label", "labels", "truth_link", "truth_links", "injection_manifest",
        "error_class", "hard_negative_type", "ambiguity_type", "attack_type", "template_id"
    }

    bundle = _make_test_bundle()
    b_dict = bundle.to_dict()

    top_keys = set(b_dict.keys())
    fact_keys = set(b_dict["facts"].keys())
    untrusted_keys = set(b_dict["untrusted_text"].keys())

    assert not (top_keys & forbidden_keys)
    assert not (fact_keys & forbidden_keys)
    assert not (untrusted_keys & forbidden_keys)


# ---------------------------------------------------------------------------
# 2. Schema Validation
# ---------------------------------------------------------------------------

def test_schema_valid_accept() -> None:
    """Conforming output passes schema validation."""
    valid_payload = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Valid fee waiver note matches arithmetic gap.",
        "evidence": [{"field": "facts.refund_difference_paise", "value": -1500}],
        "cited_amount_paise": 1500,
        "verbatim_quote": "Fee waived Rs 15.00",
    }

    is_valid, parsed, errors = validate_llm_output(json.dumps(valid_payload))
    assert is_valid
    assert parsed is not None
    assert errors == []
    assert parsed.decision == "benign_explained"
    assert parsed.cited_amount_paise == 1500


def test_schema_extra_keys_reject() -> None:
    """Unexpected extra keys cause schema rejection."""
    payload = {
        "decision": "escalate_human",
        "predicted_class": None,
        "reason_code": "AMBIGUOUS_TEXT",
        "explanation": "Needs human check.",
        "evidence": [],
        "cited_amount_paise": None,
        "verbatim_quote": None,
        "malicious_injected_key": "DROP TABLE",
    }

    is_valid, parsed, errors = validate_llm_output(payload)
    assert not is_valid
    assert parsed is None
    assert any("malicious_injected_key" in err for err in errors)


def test_schema_enum_reject() -> None:
    """Invalid decision or invalid predicted_class fails validation."""
    payload_bad_decision = {
        "decision": "approve_immediately",
        "predicted_class": None,
        "reason_code": "ADMIN_OVERRIDE",
        "explanation": "Overridden.",
        "evidence": [],
        "cited_amount_paise": None,
        "verbatim_quote": None,
    }
    is_valid, _, errors = validate_llm_output(payload_bad_decision)
    assert not is_valid
    assert any("Invalid decision" in err for err in errors)

    payload_bad_class = {
        "decision": "flag",
        "predicted_class": "NON_EXISTENT_ERROR_CLASS",
        "reason_code": "FLAGGED",
        "explanation": "Bad class.",
        "evidence": [],
        "cited_amount_paise": None,
        "verbatim_quote": None,
    }
    is_valid2, _, errors2 = validate_llm_output(payload_bad_class)
    assert not is_valid2
    assert any("Invalid predicted_class" in err for err in errors2)


def test_schema_length_reject() -> None:
    """Explanation > 280 chars or quote > 160 chars fails validation."""
    long_explanation = "A" * 281
    payload = {
        "decision": "escalate_human",
        "predicted_class": None,
        "reason_code": "TOO_LONG",
        "explanation": long_explanation,
        "evidence": [],
        "cited_amount_paise": None,
        "verbatim_quote": None,
    }
    is_valid, _, errors = validate_llm_output(payload)
    assert not is_valid
    assert any("280 character limit" in err for err in errors)

    long_quote = "Q" * 161
    payload2 = {
        "decision": "escalate_human",
        "predicted_class": None,
        "reason_code": "TOO_LONG",
        "explanation": "Valid length explanation.",
        "evidence": [],
        "cited_amount_paise": None,
        "verbatim_quote": long_quote,
    }
    is_valid2, _, errors2 = validate_llm_output(payload2)
    assert not is_valid2
    assert any("160 character limit" in err for err in errors2)


# ---------------------------------------------------------------------------
# 3. Verifier Rules V1-V7
# ---------------------------------------------------------------------------

def test_verifier_accept_clean() -> None:
    """Valid benign_explained passes all verifier checks with ACCEPT."""
    bundle = _make_test_bundle(
        amount_paise=10000,
        debit_paise=8500,
        refund_diff_paise=-1500,
        reason_text="Fee deduction of Rs 15.00 authorized.",
    )
    output = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "The deduction of 1500 paise is grounded by Rs 15.00 fee authorized.",
        "evidence": [{"field": "facts.refund_difference_paise", "value": -1500}],
        "cited_amount_paise": 1500,
        "verbatim_quote": "Rs 15.00",
    }
    verifier = Verifier()
    res = verifier.verify(bundle, output)
    assert res.outcome == VerifierOutcome.ACCEPT
    assert res.failed_checks == ()


def test_verifier_reject_unquoted() -> None:
    """Fabricated quote not present in untrusted text fails V4 (REJECT)."""
    bundle = _make_test_bundle(reason_text="Actual text is completely different.")
    output = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Explanation claiming waiver.",
        "evidence": [],
        "cited_amount_paise": 1500,
        "verbatim_quote": "Hallucinated waiver Rs 15.00",
    }
    verifier = Verifier()
    res = verifier.verify(bundle, output)
    assert res.outcome == VerifierOutcome.REJECT
    assert "V4" in res.failed_checks


def test_verifier_reject_bad_amount() -> None:
    """Cited amount != arithmetic refund difference fails V7 (REJECT)."""
    bundle = _make_test_bundle(
        amount_paise=10000,
        debit_paise=8500,
        refund_diff_paise=-1500,
        reason_text="Deduction Rs 20.00.",
    )
    output = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Claiming 2000 paise deduction.",
        "evidence": [],
        "cited_amount_paise": 2000,
        "verbatim_quote": "Rs 20.00",
    }
    verifier = Verifier()
    res = verifier.verify(bundle, output)
    assert res.outcome == VerifierOutcome.REJECT
    assert "V7" in res.failed_checks


def test_verifier_reject_fabricated_numbers() -> None:
    """Explanation containing ungrounded numbers fails V6 (REJECT)."""
    bundle = _make_test_bundle(
        amount_paise=10000,
        debit_paise=8500,
        refund_diff_paise=-1500,
        reason_text="Fee waived Rs 15.00.",
    )
    # 987654 is an ungrounded fabricated number not in bundle or 0..100
    output = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Fee waived Rs 15.00 confirmed under ticket 987654.",
        "evidence": [],
        "cited_amount_paise": 1500,
        "verbatim_quote": "Rs 15.00",
    }
    verifier = Verifier()
    res = verifier.verify(bundle, output)
    assert res.outcome == VerifierOutcome.REJECT
    assert "V6" in res.failed_checks


def test_verifier_accept_escalate() -> None:
    """escalate_human decision passes through safely."""
    bundle = _make_test_bundle()
    output = {
        "decision": "escalate_human",
        "predicted_class": None,
        "reason_code": "SUSPECTED_ATTACK",
        "explanation": "Untrusted text attempts prompt injection.",
        "evidence": [{"field": "entity_id", "value": bundle.entity_id}],
        "cited_amount_paise": None,
        "verbatim_quote": None,
    }
    verifier = Verifier()
    res = verifier.verify(bundle, output)
    assert res.outcome == VerifierOutcome.ACCEPT


# ---------------------------------------------------------------------------
# 4. Amendments 3 & 4 Tests
# ---------------------------------------------------------------------------

def test_verifier_v5_two_amounts_in_quote_abstain() -> None:
    """Amendment 3: Quote containing two amount tokens causes V5 to ABSTAIN."""
    bundle = _make_test_bundle(
        reason_text="Fee Rs 10.00 and tax Rs 5.00 waived.",
    )
    output = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Multiple fee waivers cited.",
        "evidence": [],
        "cited_amount_paise": 1500,
        "verbatim_quote": "Fee Rs 10.00 and tax Rs 5.00",
    }
    verifier = Verifier()
    res = verifier.verify(bundle, output)
    assert res.outcome == VerifierOutcome.ABSTAIN
    assert "V5" in res.failed_checks
    assert res.reason_code == "AMBIGUOUS_QUOTE_AMOUNTS"


def test_verifier_v5_number_word_quote_abstain() -> None:
    """Amendment 3: Number-word quote ('fifteen rupees') without digits triggers ABSTAIN."""
    bundle = _make_test_bundle(
        reason_text="Customer accepted fifteen rupees fee deduction.",
    )
    output = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Note states fifteen rupees deduction.",
        "evidence": [],
        "cited_amount_paise": 1500,
        "verbatim_quote": "fifteen rupees fee deduction",
    }
    verifier = Verifier()
    res = verifier.verify(bundle, output)
    assert res.outcome == VerifierOutcome.ABSTAIN
    assert "V5" in res.failed_checks


def test_verifier_v6_bundle_id_and_fabricated_number() -> None:
    """Amendment 4: Bundle ID 'pay_Ab12Cd' in explanation is allowed; fabricated 4,321 REJECTs."""
    bundle = _make_test_bundle(
        amount_paise=10000,
        debit_paise=8500,
        refund_diff_paise=-1500,
        reason_text="Fee waived Rs 15.00 applied.",
    )

    # Valid case: mentions payment_id pay_Ab12Cd and allowed amount 1500
    valid_output = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Verified deduction for pay_Ab12Cd matching 1500 paise gap.",
        "evidence": [],
        "cited_amount_paise": 1500,
        "verbatim_quote": "Rs 15.00",
    }
    verifier = Verifier()
    res_valid = verifier.verify(bundle, valid_output)
    assert res_valid.outcome == VerifierOutcome.ACCEPT

    # Invalid case: contains fabricated 4,321
    bad_output = {
        "decision": "benign_explained",
        "predicted_class": None,
        "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Verified deduction for pay_Ab12Cd with reference 4,321.",
        "evidence": [],
        "cited_amount_paise": 1500,
        "verbatim_quote": "Rs 15.00",
    }
    res_bad = verifier.verify(bundle, bad_output)
    assert res_bad.outcome == VerifierOutcome.REJECT
    assert "V6" in res_bad.failed_checks


# ---------------------------------------------------------------------------
# 5. Cache & Audit Logger
# ---------------------------------------------------------------------------

def test_cache_hit() -> None:
    """Second complete call hits disk cache; client call count does not increase."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = LLMCache(cache_dir=tmp_dir)
        client = MockClient(cache=cache)
        bundle = _make_test_bundle()

        # First call: cache miss, client increments call_count
        resp1 = client.complete(bundle, prompt_hash="p1")
        assert client.call_count == 1
        assert resp1 is not None

        # Direct call to cache should return the string
        cached_val = cache.get(bundle.bundle_hash, "p1", client.model_id)
        assert cached_val == resp1

        # Second call with another client pointing to the same cache
        client2 = MockClient(cache=cache)
        resp2 = client2.complete(bundle, prompt_hash="p1")
        # client2 did not increment because it hit cache
        assert client2.call_count == 1
        assert resp2 == resp1


def test_audit_log_format() -> None:
    """Audit log contains required fields and NEVER contains raw untrusted text."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "llm_audit.jsonl")
        logger = AuditLogger(log_path=log_file)

        raw_output = json.dumps({"decision": "escalate_human", "text": "MALICIOUS INJECTION TEXT"})

        record = logger.log(
            bundle_hash="abc123bundle",
            prompt_hash="def456prompt",
            model_id="mock-v1",
            raw_output=raw_output,
            schema_valid=True,
            verifier_outcome="ACCEPT",
            verifier_failed_checks=[],
            final_action="escalate_human",
        )

        assert os.path.isfile(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 1
        entry = lines[0]
        assert entry["bundle_hash"] == "abc123bundle"
        assert entry["prompt_hash"] == "def456prompt"
        assert entry["verifier_outcome"] == "ACCEPT"
        assert entry["final_action"] == "escalate_human"

        # Privacy guarantee: raw untrusted text MUST NOT appear in the audit file
        with open(log_file, "r", encoding="utf-8") as f:
            raw_content = f.read()
            assert "MALICIOUS INJECTION TEXT" not in raw_content
            assert record.raw_output_hash in raw_content


# ---------------------------------------------------------------------------
# 6. Boundary Isolation Test
# ---------------------------------------------------------------------------

def test_llm_boundary_no_synth_or_eval() -> None:
    """Guarantee src/recon_agent/llm/ never imports recon_agent.synth or references eval files."""
    llm_dir = os.path.join(REPO_ROOT, "src", "recon_agent", "llm")
    assert os.path.isdir(llm_dir)

    eval_filenames = ("labels.jsonl", "truth_links.jsonl", "injection_manifest.jsonl")

    for root, _, files in os.walk(llm_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # No eval filenames in source
            for ef in eval_filenames:
                assert ef not in content, f"File {path} references eval file '{ef}'"

            # AST walk: no synth imports
            tree = ast.parse(content, filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "synth" not in alias.name, f"Illegal synth import in {path}: {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        assert "synth" not in node.module, f"Illegal synth import-from in {path}: {node.module}"


# ---------------------------------------------------------------------------
# 7. Day 7a Tests: no_issue, Negative Controls & GeminiClient Offline Tests
# ---------------------------------------------------------------------------

def test_schema_no_issue_accept() -> None:
    """Valid no_issue output with predicted_class=None passes schema validation."""
    payload = {
        "decision": "no_issue",
        "predicted_class": None,
        "explanation": "Calculated fee and tax match expected values within 2 paise tolerance.",
        "evidence": [{"field": "fee_paise", "value": 2360}, {"field": "tax_paise", "value": 360}],
        "verbatim_quote": None,
        "cited_amount_paise": None,
        "reason_code": "WITHIN_TOLERANCE",
    }
    is_valid, parsed, errors = validate_llm_output(json.dumps(payload))
    assert is_valid, f"Validation failed with: {errors}"
    assert parsed is not None
    assert parsed.decision == "no_issue"
    assert parsed.predicted_class is None


def test_schema_no_issue_reject_with_class() -> None:
    """no_issue with non-null predicted_class fails schema validation."""
    payload = {
        "decision": "no_issue",
        "predicted_class": "FEE_TAX_MISMATCH",
        "explanation": "Calculated fee and tax match within tolerance.",
        "evidence": [],
        "verbatim_quote": None,
        "cited_amount_paise": None,
        "reason_code": "WITHIN_TOLERANCE",
    }
    is_valid, parsed, errors = validate_llm_output(json.dumps(payload))
    assert not is_valid
    assert any("must be null" in err.lower() for err in errors)


def test_schema_json_single_source_of_truth() -> None:
    """get_llm_json_schema() enum choices and structure agree with validate_llm_output."""
    schema = get_llm_json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "decision" in props
    assert set(props["decision"]["enum"]) == set(ALLOWED_DECISIONS)
    assert "predicted_class" in props
    assert props["predicted_class"].get("nullable") is True
    assert set(props["predicted_class"]["enum"]) == set(ALLOWED_CLASSES)


def test_verifier_no_issue_within_tolerance() -> None:
    """no_issue decision on payment within tolerance ACCEPTs."""
    verifier = Verifier()
    bundle = create_case_bundle(
        bundle_id="bundle_pay_ctrl",
        entity_id="pay_ctrl_1",
        merchant_id="mer_01",
        case_type="payment",
        rule_id="B1_MATCH_EXACT",
        facts={
            "entity_id": "pay_ctrl_1",
            "type": "payment",
            "amount_paise": 100000,
            "fee_paise": 2360,
            "tax_paise": 360,
        },
        untrusted_text={
            "recon_item_description": "Clean transaction",
            "recon_item_notes_comment": None,
            "recon_item_notes_reason_text": None,
            "bank_narration": None,
        },
    )
    llm_out = LLMOutput(
        decision="no_issue",
        predicted_class=None,
        explanation="Payment fee and tax are verified correct within policy tolerance.",
        evidence=[{"field": "fee_paise", "value": 2360}, {"field": "tax_paise", "value": 360}],
        verbatim_quote=None,
        cited_amount_paise=None,
        reason_code="VERIFIED_TOLERANCE",
    )
    res = verifier.verify(bundle, llm_out)
    assert res.outcome == VerifierOutcome.ACCEPT


def test_verifier_no_issue_rejects_discrepancies() -> None:
    """no_issue on NEAR_TOLERANCE payment, non-zero refund diff, or ambiguous bank match REJECTs (V7)."""
    verifier = Verifier()
    llm_out = LLMOutput(
        decision="no_issue",
        predicted_class=None,
        explanation="Everything looks normal.",
        evidence=[],
        verbatim_quote=None,
        cited_amount_paise=None,
        reason_code=None,
    )

    # 1. NEAR_TOLERANCE payment
    pay_bundle = create_case_bundle(
        bundle_id="bundle_pay_disc",
        entity_id="pay_disc_1",
        merchant_id="mer_01",
        case_type="payment",
        rule_id="B1_NEAR_TOLERANCE",
        facts={"entity_id": "pay_disc_1", "type": "payment", "amount_paise": 50000, "fee_paise": 1180},
        untrusted_text={},
    )
    res_pay = verifier.verify(pay_bundle, llm_out)
    assert res_pay.outcome == VerifierOutcome.REJECT
    assert "V7" in res_pay.failed_checks
    assert res_pay.reason_code == "POLICY_VIOLATION_TOLERANCE"

    # 2. Refund with non-zero difference
    rfnd_bundle = create_case_bundle(
        bundle_id="bundle_rfnd_disc",
        entity_id="rfnd_disc_1",
        merchant_id="mer_01",
        case_type="refund",
        rule_id="B1R_EXPLAINED_NO_AMOUNT",
        facts={
            "entity_id": "rfnd_disc_1",
            "type": "refund",
            "amount_paise": 5000,
            "debit_paise": 4000,
            "refund_difference_paise": -1000,
        },
        untrusted_text={},
    )
    res_rfnd = verifier.verify(rfnd_bundle, llm_out)
    assert res_rfnd.outcome == VerifierOutcome.REJECT
    assert "V7" in res_rfnd.failed_checks
    assert res_rfnd.reason_code == "POLICY_VIOLATION_REFUND_DIFF"

    # 3. Bank match with ambiguous candidates
    bank_bundle = create_case_bundle(
        bundle_id="bundle_bank_disc",
        entity_id="stmt_disc_1",
        merchant_id="mer_01",
        case_type="bank_statement",
        rule_id="B1_BANK_AMBIGUOUS_CANDIDATES",
        facts={"entity_id": "stmt_disc_1", "type": "bank_statement", "credit_paise": 100000},
        untrusted_text={},
    )
    res_bank = verifier.verify(bank_bundle, llm_out)
    assert res_bank.outcome == VerifierOutcome.REJECT
    assert "V7" in res_bank.failed_checks
    assert res_bank.reason_code == "POLICY_VIOLATION_AMBIGUOUS_BANK"


def test_control_bundles_symmetry_and_persistence() -> None:
    """build_control_bundles produces symmetric bundles with identical key sets and writes results/controls_<seed>.json."""
    from src.recon_agent.systems.b1 import run_b1

    dev_dir = os.path.join(REPO_ROOT, "data", "synth", "full", "dev_seed1")
    if not os.path.isdir(dev_dir):
        pytest.skip(f"Dev dataset not found at {dev_dir}")

    dataset = load_dataset(dev_dir)
    output = run_b1(dataset)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundles = build_control_bundles(
            dataset=dataset,
            system_output=output,
            n_per_seed=15,
            rng_seed=42,
            results_dir=tmp_dir,
        )
        assert len(bundles) == 15  # 5 * 3 families
        out_files = [f for f in os.listdir(tmp_dir) if f.startswith("controls_") and f.endswith(".json")]
        assert len(out_files) == 1
        with open(os.path.join(tmp_dir, out_files[0]), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "counts" in data
        assert data["counts"]["fee_tax"] == 5
        assert data["counts"]["refunds"] == 5
        assert data["counts"]["bank_matches"] == 5

        # Check bundle schema symmetry
        first_untrusted_keys = list(bundles[0].untrusted_text.keys())
        for b in bundles:
            assert list(b.untrusted_text.keys()) == first_untrusted_keys


def test_gemini_client_fatal_errors_offline() -> None:
    """HTTP 401, 402, 403 raise FatalAPIError immediately without retrying."""
    from unittest.mock import MagicMock, patch

    client = GeminiClient(api_key="test-api-key", sleep_fn=lambda _: None)
    bundle = _make_test_bundle()

    for code in (401, 402, 403):
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = code
            mock_resp.text = f"Error {code} Unauthorized/Forbidden"
            mock_post.return_value = mock_resp

            with pytest.raises(FatalAPIError) as exc_info:
                client.complete(bundle, system_prompt="System prompt", run_id=0)
            assert f"Fatal Gemini API HTTP error {code}" in str(exc_info.value)
            assert mock_post.call_count == 1


def test_gemini_client_non_retriable_400_offline() -> None:
    """HTTP 400 raises NonRetriableAPIError immediately without retrying."""
    from unittest.mock import MagicMock, patch

    client = GeminiClient(api_key="test-api-key")
    bundle = _make_test_bundle()

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request: Schema violation"
        mock_post.return_value = mock_resp

        with pytest.raises(NonRetriableAPIError) as exc_info:
            client.complete(bundle, system_prompt="System prompt", run_id=0)
        assert "Non-retriable Gemini API HTTP 400 Bad Request" in str(exc_info.value)
        assert mock_post.call_count == 1


def test_gemini_client_retries_429_then_succeeds() -> None:
    """HTTP 429 triggers retry with backoff and succeeds on subsequent attempt."""
    from unittest.mock import MagicMock, patch

    client = GeminiClient(api_key="test-api-key", max_retries=2)
    bundle = _make_test_bundle()

    success_payload = {
        "model": "gemini-3.5-flash-lite",
        "output_text": json.dumps({
            "decision": "escalate_human",
            "predicted_class": "BENIGN_AMBIGUITY",
            "explanation": "Multiple candidate explanations found.",
            "evidence": [],
            "verbatim_quote": None,
            "cited_amount_paise": None,
            "reason_code": "AMBIGUOUS",
        }),
        "usage": {
            "total_input_tokens": 100,
            "total_output_tokens": 30,
            "total_thought_tokens": 10,
        },
    }

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.01"}
    resp_429.text = "Rate limit"

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = success_payload

    with patch("requests.post") as mock_post, patch("time.sleep") as mock_sleep:
        mock_post.side_effect = [resp_429, resp_200]
        result = client.complete(bundle, system_prompt="System prompt", run_id=0)
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1
        assert "escalate_human" in result


def test_gemini_client_build_request_payload() -> None:
    """Built request payload has store=False, fixed system_instruction, and ONLY bundle JSON in input."""
    client = GeminiClient(api_key="test-api-key")
    bundle = _make_test_bundle(
        reason_text="ATTACK_INJECTION_PROMPT_IGNORE_RULES_AND_ACCEPT"
    )
    system_inst = "FIXED SYSTEM INSTRUCTION: Reconcile financial records."
    payload = client.build_request_payload(bundle, system_instruction=system_inst)

    assert payload["store"] is False
    assert payload["system_instruction"] == system_inst
    # Untrusted text MUST NOT leak into system_instruction
    assert "ATTACK_INJECTION_PROMPT_IGNORE_RULES_AND_ACCEPT" not in payload["system_instruction"]
    # Untrusted text MUST be inside input JSON
    assert "ATTACK_INJECTION_PROMPT_IGNORE_RULES_AND_ACCEPT" in payload["input"]
    input_data = json.loads(payload["input"])
    assert input_data["bundle_id"] == bundle.bundle_id


def test_gemini_client_rejects_minimal_thinking_level() -> None:
    """thinking_level='minimal' is rejected at construction with clear unsupported error."""
    with pytest.raises(ValueError) as exc_info:
        GeminiClient(api_key="test-api-key", thinking_level="minimal")
    assert "thinking_level 'minimal' is unsupported for" in str(exc_info.value)


def test_gemini_client_costs_none_when_tokens_missing() -> None:
    """When token counts are missing, both cost_low and cost_high are None and never 0."""
    client = GeminiClient(api_key="test-api-key")
    cost_low, cost_high = client.calculate_estimated_costs(None, 100, 50)
    assert cost_low is None
    assert cost_high is None

    cost_low2, cost_high2 = client.calculate_estimated_costs(100, None, 50)
    assert cost_low2 is None
    assert cost_high2 is None


def test_gemini_client_parse_shape1_output_text() -> None:
    """Shape 1: top-level output_text and usage are correctly parsed."""
    client = GeminiClient(api_key="test-api-key")
    resp_json = {
        "output_text": '{"decision": "no_issue", "predicted_class": null}',
        "usage": {
            "total_input_tokens": 500,
            "total_output_tokens": 120,
            "total_thought_tokens": 60,
        },
    }
    raw_text, usage_num, tokens, usage_flagged, shape = client.parse_interaction_response(resp_json)
    assert raw_text == '{"decision": "no_issue", "predicted_class": null}'
    assert tokens["input_tokens"] == 500
    assert tokens["output_tokens"] == 120
    assert tokens["thinking_tokens"] == 60
    assert usage_flagged is False
    assert shape == {"output_text": "str", "usage": {"total_input_tokens": "int", "total_output_tokens": "int", "total_thought_tokens": "int"}}


def test_gemini_client_parse_shape2_model_output_steps() -> None:
    """Shape 2: steps of type model_output concatenated, thought steps strictly ignored, tolerant usage keys."""
    client = GeminiClient(api_key="test-api-key")
    resp_json = {
        "steps": [
            {
                "type": "thought",
                "content": {
                    "parts": [{"type": "thought", "text": "INTERNAL THOUGHT: NEVER EXTRACT THIS"}]
                },
            },
            {
                "type": "model_output",
                "content": {
                    "parts": [
                        {"type": "text", "text": '{"decision": "escalate_human", '},
                        {"type": "text", "text": '"predicted_class": null}'},
                    ]
                },
            },
        ],
        "usage": {
            "prompt_tokens": 300,
            "candidates_tokens": 80,
            "thought_tokens": 40,
        },
    }
    raw_text, usage_num, tokens, usage_flagged, shape = client.parse_interaction_response(resp_json)
    assert raw_text == '{"decision": "escalate_human", "predicted_class": null}'
    assert "INTERNAL THOUGHT" not in (raw_text or "")
    assert tokens["input_tokens"] == 300
    assert tokens["output_tokens"] == 80
    assert tokens["thinking_tokens"] == 40
    assert usage_flagged is False


def test_gemini_client_parse_unparseable_shape() -> None:
    """Any shape other than Shape 1 or Shape 2 maps to escalate_human and records key structure."""
    from unittest.mock import MagicMock, patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        audit_file = os.path.join(tmp_dir, "audit.jsonl")
        logger = AuditLogger(log_path=audit_file)
        client = GeminiClient(api_key="test-api-key", audit_logger=logger)
        bundle = _make_test_bundle()

        # Legacy candidates-only shape (must NOT be accepted as valid)
        unparseable_payload = {
            "model": "gemini-3.5-flash-lite",
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"decision": "no_issue"}'}]
                    }
                }
            ],
            "usage": {"total_input_tokens": 100, "total_output_tokens": 50},
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = unparseable_payload

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_resp
            res = client.complete(bundle, system_prompt="system prompt", run_id=0)

        res_data = json.loads(res)
        assert res_data["decision"] == "escalate_human"
        assert res_data["reason_code"] == "UNPARSEABLE_RESPONSE_SHAPE"

        # Verify audit log contains key structure (names and types only)
        with open(audit_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 1
        entry = lines[0]
        assert entry["final_action"] == "escalate_human"
        assert entry["verifier_failed_checks"] == ["UNPARSEABLE_RESPONSE_SHAPE"]
        assert "raw_response_shape" in entry
        # Structure has names and types only
        expected_shape = {
            "model": "str",
            "candidates": [{"content": {"parts": [{"text": "str"}]}}],
            "usage": {"total_input_tokens": "int", "total_output_tokens": "int"},
        }
        assert entry["raw_response_shape"] == expected_shape


def test_gemini_client_read_timeout_max_1_retry_and_audit() -> None:
    """Read timeout allows at most 1 retry, then records duplicate billing in audit and raises InfraError."""
    from unittest.mock import patch
    import requests.exceptions
    from src.recon_agent.llm.gemini_client import InfraError

    with tempfile.TemporaryDirectory() as tmp_dir:
        audit_file = os.path.join(tmp_dir, "audit_timeout.jsonl")
        logger = AuditLogger(log_path=audit_file)
        client = GeminiClient(api_key="test-api-key", audit_logger=logger)
        bundle = _make_test_bundle()

        # Attempt 1: ReadTimeout, Attempt 2 (Retry 1): ReadTimeout -> raises InfraError (produces NO model output)
        with patch("requests.post") as mock_post, patch("time.sleep"):
            mock_post.side_effect = [
                requests.exceptions.ReadTimeout("Server took too long to send data"),
                requests.exceptions.ReadTimeout("Server took too long to send data"),
            ]
            with pytest.raises(InfraError) as exc_info:
                client.complete(bundle, system_prompt="System prompt", run_id=0)
            assert "possible duplicate billing" in str(exc_info.value)

        assert mock_post.call_count == 2

        # Check audit entry
        with open(audit_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 1
        entry = lines[0]
        assert "possible duplicate billing" in entry["verifier_failed_checks"]
        assert "attempt_count=2" in entry["verifier_failed_checks"]
        assert entry["cost_usd"] is None


def test_gemini_client_connect_error_retries_up_to_max() -> None:
    """Connect errors retry up to max_retries (4) before failing."""
    from unittest.mock import patch
    import requests.exceptions

    client = GeminiClient(api_key="test-api-key", max_retries=3)
    bundle = _make_test_bundle()

    with patch("requests.post") as mock_post, patch("time.sleep"):
        mock_post.side_effect = requests.exceptions.ConnectTimeout("Connection timed out")
        with pytest.raises(RuntimeError) as exc_info:
            client.complete(bundle, system_prompt="System prompt", run_id=0)
        assert mock_post.call_count == 3
        assert "Exceeded max retries on connect error" in str(exc_info.value)


def test_gemini_client_utf8_rupee_symbol_and_no_content_length() -> None:
    """Request body preserves non-ASCII text like the rupee symbol ₹ and sets no manual Content-Length header."""
    from unittest.mock import MagicMock, patch

    client = GeminiClient(api_key="test-api-key", read_timeout=120.0, connect_timeout=10.0)
    bundle = _make_test_bundle(
        reason_text="Fee of ₹15.00 applied and waived."
    )
    system_inst = "System prompt ₹ testing."

    payload = client.build_request_payload(bundle, system_instruction=system_inst)
    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # Indian Rupee symbol is U+20B9 -> UTF-8 bytes b'\xe2\x82\xb9'
    assert b"\xe2\x82\xb9" in body_bytes
    decoded = json.loads(body_bytes.decode("utf-8"))
    assert "₹" in decoded["input"]
    assert "₹" in decoded["system_instruction"]

    success_payload = {
        "model": "gemini-3.5-flash-lite",
        "output_text": json.dumps({
            "decision": "no_issue",
            "predicted_class": None,
            "explanation": "Verified without issues.",
            "evidence": [],
            "verbatim_quote": None,
            "cited_amount_paise": None,
            "reason_code": "WITHIN_TOLERANCE",
        }),
        "usage": {"total_input_tokens": 100, "total_output_tokens": 20},
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = success_payload

    with patch("requests.post") as mock_post:
        mock_post.return_value = mock_resp
        client.complete(bundle, system_prompt=system_inst, run_id=0)

        assert mock_post.call_count == 1
        _args, kwargs = mock_post.call_args
        headers = kwargs["headers"]
        # No manual Content-Length header!
        assert "Content-Length" not in headers
        assert "content-length" not in headers
        assert headers["Content-Type"] == "application/json; charset=utf-8"
        assert headers["x-goog-api-key"] == "test-api-key"
        # Body passed as explicit UTF-8 bytes
        assert kwargs["data"] == body_bytes
        # Timeout configured as (connect_timeout, read_timeout)
        assert kwargs["timeout"] == (10.0, 120.0)


def test_verifier_v1_rejects_overlength_explanation() -> None:
    """V1 client-side validation strictly rejects explanation exceeding 280 characters with SCHEMA_INVALID."""
    verifier = Verifier()
    bundle = _make_test_bundle()

    overlength_text = "A" * 285
    payload = {
        "decision": "escalate_human",
        "predicted_class": None,
        "reason_code": "AMBIGUOUS",
        "explanation": overlength_text,
        "evidence": [],
        "verbatim_quote": None,
        "cited_amount_paise": None,
    }

    res = verifier.verify(bundle, json.dumps(payload))
    assert res.outcome == VerifierOutcome.REJECT
    assert res.failed_checks == ("V1",)
    assert res.reason_code == "SCHEMA_INVALID"
    assert "280" in res.details


# ---------------------------------------------------------------------------
# 8. Day 7a-3 & Day 7a-4 Tests: Model Config, Served-Model Check & Runner
# ---------------------------------------------------------------------------

def test_model_config_loading_and_alias_rejection() -> None:
    """Primary model defaults to gemini-3.5-flash-lite, alias-style -latest models are rejected."""
    client = GeminiClient(api_key="test-api-key")
    assert client.model_id == "gemini-3.5-flash-lite"
    assert client.tier == "free"
    assert client.rpm == 4.0

    # Reject -latest alias
    with pytest.raises(ValueError) as exc:
        GeminiClient(api_key="test-api-key", model_id="gemini-3.5-flash-latest")
    assert "Alias-style model id" in str(exc.value)

    # Reject unallowed model
    with pytest.raises(ValueError) as exc2:
        GeminiClient(api_key="test-api-key", model_id="gpt-4o")
    assert "not in allowed models list" in str(exc2.value)


def test_optional_thinking_level_omits_generation_config() -> None:
    """When thinking_level is None, payload omits generation_config; includes it when provided."""
    bundle = _make_test_bundle()

    # None thinking level
    client_none = GeminiClient(api_key="test-api-key", thinking_level=None)
    payload_none = client_none.build_request_payload(bundle, system_instruction="system prompt")
    assert "generation_config" not in payload_none

    # Low thinking level
    client_low = GeminiClient(api_key="test-api-key", thinking_level="low")
    payload_low = client_low.build_request_payload(bundle, system_instruction="system prompt")
    assert "generation_config" in payload_low
    assert payload_low["generation_config"]["thinking_level"] == "low"


def test_served_model_check_matching_and_mismatch() -> None:
    """Served model matching succeeds; mismatch raises fatal ModelMismatch (never cached)."""
    from unittest.mock import MagicMock, patch
    from src.recon_agent.llm.gemini_client import ModelMismatch

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = LLMCache(cache_dir=tmp_dir)
        client = GeminiClient(api_key="test-api-key", model_id="gemini-3.5-flash-lite", cache=cache, mode="eval", sleep_fn=lambda _: None)
        bundle = _make_test_bundle()

        # Case 1: Matching served model succeeds
        matching_resp = MagicMock()
        matching_resp.status_code = 200
        matching_resp.json.return_value = {
            "model": "gemini-3.5-flash-lite",
            "output_text": json.dumps({"decision": "no_issue", "predicted_class": None}),
            "usage": {"total_input_tokens": 100, "total_output_tokens": 20},
        }
        with patch("requests.post") as mock_post:
            mock_post.return_value = matching_resp
            res = client.complete(bundle, system_prompt="test", run_id=0)
            assert "no_issue" in res

        # Case 2: Mismatched served model raises fatal ModelMismatch
        mismatched_resp = MagicMock()
        mismatched_resp.status_code = 200
        mismatched_resp.json.return_value = {
            "model": "gemini-3.8-flash",  # different from requested gemini-3.5-flash-lite!
            "output_text": json.dumps({"decision": "no_issue", "predicted_class": None}),
            "usage": {"total_input_tokens": 100, "total_output_tokens": 20},
        }
        # Clear cache for next test by using a new run_id
        with patch("requests.post") as mock_post:
            mock_post.return_value = mismatched_resp
            with pytest.raises(ModelMismatch) as exc:
                client.complete(bundle, system_prompt="test", run_id=1)
            assert "does not match requested model" in str(exc.value)

        # Never cached under requested model ID
        assert cache.get(bundle.bundle_hash, hashlib.sha256(b"test").hexdigest(), model_id="gemini-3.5-flash-lite", run_id=1) is None


def test_served_model_missing_mode_check() -> None:
    """Missing 'model' field raises ModelMismatch in eval mode; emits warning in smoke mode."""
    from unittest.mock import MagicMock, patch
    import warnings
    from src.recon_agent.llm.gemini_client import ModelMismatch

    bundle = _make_test_bundle()
    resp_no_model = MagicMock()
    resp_no_model.status_code = 200
    resp_no_model.json.return_value = {
        "output_text": json.dumps({"decision": "no_issue", "predicted_class": None}),
        "usage": {"total_input_tokens": 100, "total_output_tokens": 20},
    }

    # Eval mode -> fatal ModelMismatch
    client_eval = GeminiClient(api_key="test-api-key", mode="eval")
    with patch("requests.post") as mock_post:
        mock_post.return_value = resp_no_model
        with pytest.raises(ModelMismatch) as exc:
            client_eval.complete(bundle, system_prompt="test", run_id=10)
        assert "missing 'model' field" in str(exc.value)

    # Smoke mode -> emits warning, does not crash
    client_smoke = GeminiClient(api_key="test-api-key", mode="smoke")
    with patch("requests.post") as mock_post, warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mock_post.return_value = resp_no_model
        res = client_smoke.complete(bundle, system_prompt="test", run_id=11)
        assert "no_issue" in res
        assert any("Response missing 'model' field" in str(item.message) for item in w)


def test_observed_interactions_response_parsing_with_and_without_id() -> None:
    """Wire shape observed from Interactions API without output_text parses correctly with or without id."""
    client = GeminiClient(api_key="test-api-key")

    # Wire shape WITH id
    wire_with_id = {
        "id": "interactions_12345",
        "status": "completed",
        "model": "gemini-3.5-flash-lite",
        "service_tier": "default",
        "steps": [
            {
                "type": "model_output",
                "content": {
                    "parts": [
                        {"type": "text", "text": json.dumps({"decision": "benign_explained", "predicted_class": "ROUNDING_TOLERANCE"})}
                    ]
                },
            }
        ],
        "usage": {
            "prompt_token_count": 420,
            "candidates_token_count": 85,
            "thinking_token_count": 0,
        },
    }
    raw_text, usage_numbers, tokens, flagged, shape = client.parse_interaction_response(wire_with_id)
    assert raw_text is not None
    assert "benign_explained" in raw_text
    assert tokens["input_tokens"] == 420
    assert tokens["output_tokens"] == 85
    assert tokens["thinking_tokens"] == 0
    assert flagged is False

    # Wire shape WITHOUT id (store: false)
    wire_no_id = dict(wire_with_id)
    del wire_no_id["id"]
    raw_text2, _, tokens2, _, _ = client.parse_interaction_response(wire_no_id)
    assert raw_text2 == raw_text
    assert tokens2["input_tokens"] == 420


def test_client_side_rate_limiter_injectable() -> None:
    """Rate limiter enforces min_interval between calls using injectable clock and sleep without delay."""
    current_time = [1000.0]
    sleep_calls = []

    def mock_now() -> float:
        return current_time[0]

    def mock_sleep(sec: float) -> None:
        sleep_calls.append(sec)
        current_time[0] += sec

    from unittest.mock import MagicMock, patch

    client = GeminiClient(
        api_key="test-api-key",
        rpm=6.0,  # 6 RPM -> min_interval = 10.0 seconds
        time_fn=mock_now,
        sleep_fn=mock_sleep,
    )
    bundle = _make_test_bundle()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "gemini-3.5-flash-lite",
        "output_text": json.dumps({"decision": "no_issue", "predicted_class": None}),
        "usage": {"total_input_tokens": 100, "total_output_tokens": 20},
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value = mock_resp
        # Call 1: immediate
        client.complete(bundle, system_prompt="test", run_id=1)
        assert len(sleep_calls) == 0

        # Advance clock by only 3 seconds
        current_time[0] += 3.0
        # Call 2: should sleep 7.0 seconds to meet 10.0 second interval
        client.complete(bundle, system_prompt="test", run_id=2)
        assert len(sleep_calls) == 1
        assert abs(sleep_calls[0] - 7.0) < 0.01


def test_http_503_backoff_schedule() -> None:
    """HTTP 503 triggers backoff (~5s, 15s, 45s) using injectable sleep."""
    from unittest.mock import MagicMock, patch
    from src.recon_agent.llm.gemini_client import InfraError

    sleeps = []
    client = GeminiClient(
        api_key="test-api-key",
        max_retries=3,
        sleep_fn=lambda s: sleeps.append(s),
    )
    bundle = _make_test_bundle()

    resp_503 = MagicMock()
    resp_503.status_code = 503
    resp_503.text = "Service Unavailable"

    with patch("requests.post") as mock_post:
        mock_post.return_value = resp_503
        with pytest.raises(InfraError) as exc:
            client.complete(bundle, system_prompt="test", run_id=0)
        assert "Exceeded max retries (3) on HTTP 503" in str(exc.value)

    assert len(sleeps) == 2  # attempt 1 sleep, attempt 2 sleep, attempt 3 fails
    # Check backoff ranges: attempt 1 >= 5s, attempt 2 >= 15s
    assert 5.0 <= sleeps[0] <= 6.0
    assert 15.0 <= sleeps[1] <= 16.0


def test_quota_tracker_dst_and_midnight_rollover() -> None:
    """QuotaTracker resets request count at midnight Pacific time across DST boundaries."""
    from zoneinfo import ZoneInfo
    from src.recon_agent.llm.runner import QuotaTracker
    from src.recon_agent.llm.gemini_client import DailyQuotaExhaustedError

    la_tz = ZoneInfo("America/Los_Angeles")
    # Test across autumn DST transition date: 2026-11-01
    simulated_now = [datetime.datetime(2026, 11, 1, 23, 59, 50, tzinfo=la_tz)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = os.path.join(tmp_dir, "quota_state.json")
        tracker = QuotaTracker(
            quota_state_path=state_file,
            max_requests_per_day=3,
            now_fn=lambda: simulated_now[0],
        )

        assert tracker.get_pacific_date() == "2026-11-01"
        assert tracker.can_request() is True

        tracker.record_request()
        tracker.record_request()
        tracker.record_request()
        assert tracker.can_request() is False

        # Budget exhausted raises
        with pytest.raises(DailyQuotaExhaustedError):
            tracker.record_request()

        # Advance past midnight to next Pacific date (2026-11-02 00:00:10)
        simulated_now[0] = datetime.datetime(2026, 11, 2, 0, 0, 10, tzinfo=la_tz)
        assert tracker.get_pacific_date() == "2026-11-02"
        # Midnight rollover reset!
        assert tracker.can_request() is True
        assert tracker.requests_remaining() == 3
        new_cnt = tracker.record_request()
        assert new_cnt == 1


def test_stratified_case_ordering() -> None:
    """order_cases_stratified interleaves (seed, family) strata so any prefix is balanced."""
    from src.recon_agent.llm.runner import order_cases_stratified

    cases = []
    # Create 3 seeds * 3 families = 9 strata, each with 4 cases = 36 cases
    for s in (1, 2, 3):
        for f in ("fee_tax", "refund", "bank"):
            for i in range(4):
                bundle = create_case_bundle(
                    bundle_id=f"bundle_s{s}_{f}_{i}",
                    entity_id=f"ent_{s}_{f}_{i}",
                    merchant_id="mer_01",
                    case_type="payment",
                    rule_id=f,
                    facts={"seed": s, "type": "payment"},
                    untrusted_text={"recon_item_description": None},
                )
                cases.append(bundle)

    ordered = order_cases_stratified(cases, rng_seed=123)
    assert len(ordered) == 36

    # Any prefix of length 9 should contain multiple distinct families and seeds
    prefix9 = ordered[:9]
    families = {b.rule_id for b in prefix9}
    seeds = {b.facts["seed"] for b in prefix9}
    assert len(families) == 3
    assert len(seeds) == 3


def test_resumable_runner_flow() -> None:
    """Runner manages ok, infra_failed, model_invalid, and pending states with --resume."""
    from unittest.mock import MagicMock, patch
    from src.recon_agent.llm.runner import Runner, QuotaTracker, score_llm_run
    from src.recon_agent.llm.gemini_client import InfraError

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = os.path.join(tmp_dir, "run_state.json")
        quota_file = os.path.join(tmp_dir, "quota.json")

        tracker = QuotaTracker(quota_state_path=quota_file, max_requests_per_day=10)
        client = GeminiClient(api_key="test-api-key", sleep_fn=lambda _: None)

        # 3 test cases
        b1 = create_case_bundle(
            bundle_id="case_1_ok",
            entity_id="pay_1",
            merchant_id="mer_1",
            case_type="payment",
            rule_id="B1_MATCH_EXACT",
            facts={"entity_id": "pay_1", "type": "payment", "amount_paise": 10000, "fee_paise": 200, "tax_paise": 36},
            untrusted_text={"recon_item_description": None},
        )
        b2 = create_case_bundle(
            bundle_id="case_2_infra",
            entity_id="pay_2",
            merchant_id="mer_1",
            case_type="payment",
            rule_id="B1_MATCH_EXACT",
            facts={"entity_id": "pay_2", "type": "payment", "amount_paise": 20000, "fee_paise": 400, "tax_paise": 72},
            untrusted_text={"recon_item_description": None},
        )
        b3 = create_case_bundle(
            bundle_id="case_3_invalid",
            entity_id="pay_3",
            merchant_id="mer_1",
            case_type="payment",
            rule_id="B1_MATCH_EXACT",
            facts={"entity_id": "pay_3", "type": "payment", "amount_paise": 30000, "fee_paise": 600, "tax_paise": 108},
            untrusted_text={"recon_item_description": None},
        )

        cases = [b1, b2, b3]

        runner = Runner(
            client=client,
            quota_tracker=tracker,
            system_prompt="prompt",
            run_state_path=state_file,
        )

        # Mock responses: case 1 ok, case 2 infra fail, case 3 model invalid
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {
            "model": "gemini-3.5-flash-lite",
            "output_text": json.dumps({
                "decision": "no_issue",
                "predicted_class": None,
                "explanation": "Valid",
                "evidence": [],
                "reason_code": "MATCH",
            }),
            "usage": {"total_input_tokens": 100, "total_output_tokens": 20},
        }

        resp_invalid = MagicMock()
        resp_invalid.status_code = 200
        resp_invalid.json.return_value = {
            "model": "gemini-3.5-flash-lite",
            "output_text": "NOT_JSON_AT_ALL",
            "usage": {"total_input_tokens": 100, "total_output_tokens": 5},
        }

        with patch("requests.post") as mock_post:
            mock_post.side_effect = [
                resp_ok,
                requests.exceptions.ConnectTimeout("Connect failed"),
                requests.exceptions.ConnectTimeout("Connect failed"),
                requests.exceptions.ConnectTimeout("Connect failed"),
                requests.exceptions.ConnectTimeout("Connect failed"),
                resp_invalid,
            ]
            results = runner.run_cases(cases, resume=False)

        assert results["case_1_ok"].status == "ok"
        assert results["case_2_infra"].status == "infra_failed"
        assert results["case_2_infra"].raw_output is None  # NO model output on infra fail!
        assert results["case_3_invalid"].status == "model_invalid"

        # Scoring without --allow-infra-failures MUST refuse
        with pytest.raises(RuntimeError) as exc:
            score_llm_run(results, allow_infra_failures=False)
        assert "Refusing to score" in str(exc.value)

        # Scoring WITH --allow-infra-failures maps infra_failed to escalate_human and warns in header
        score_res = score_llm_run(results, allow_infra_failures=True, tier="free")
        assert score_res["header_warning"] is not None
        assert "WARNING: Scored with --allow-infra-failures" in score_res["header_warning"]
        assert score_res["actual_cost_usd"] == 0.0
        assert score_res["report_cost_str"] == "$0.00 (free tier)"
        assert score_res["estimated_paid_equivalent_cost_usd"] is not None

        # Test --resume: re-running with resume retries only case_2_infra
        with patch("requests.post") as mock_post:
            mock_post.return_value = resp_ok
            resumed_results = runner.run_cases(cases, resume=True)

        assert resumed_results["case_1_ok"].status == "ok"
        assert resumed_results["case_2_infra"].status == "ok"  # Recovered!
        assert resumed_results["case_3_invalid"].status == "model_invalid"  # Not re-run!




