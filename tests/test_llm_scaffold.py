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
import json
import os
import tempfile
import urllib.error
import pytest

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

    client = GeminiClient(api_key="test-api-key")
    bundle = _make_test_bundle()

    for code in (401, 402, 403):
        with patch("urllib.request.urlopen") as mock_url:
            mock_err = urllib.error.HTTPError(
                url="https://generativelanguage.googleapis.com/v1beta/interactions",
                code=code,
                msg="Forbidden",
                hdrs=MagicMock(),
                fp=None,
            )
            mock_url.side_effect = mock_err
            with pytest.raises(FatalAPIError) as exc_info:
                client.complete(bundle, system_prompt="System prompt", run_id=0)
            assert f"Fatal Gemini API HTTP error {code}" in str(exc_info.value)
            assert mock_url.call_count == 1


def test_gemini_client_non_retriable_400_offline() -> None:
    """HTTP 400 raises NonRetriableAPIError immediately without retrying."""
    from unittest.mock import MagicMock, patch

    client = GeminiClient(api_key="test-api-key")
    bundle = _make_test_bundle()

    with patch("urllib.request.urlopen") as mock_url:
        mock_err = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/v1beta/interactions",
            code=400,
            msg="Bad Request",
            hdrs=MagicMock(),
            fp=None,
        )
        mock_url.side_effect = mock_err
        with pytest.raises(NonRetriableAPIError):
            client.complete(bundle, system_prompt="System prompt", run_id=0)
        assert mock_url.call_count == 1


def test_gemini_client_retries_429_then_succeeds() -> None:
    """HTTP 429 triggers retry with backoff and succeeds on subsequent attempt."""
    from unittest.mock import MagicMock, patch

    client = GeminiClient(api_key="test-api-key", max_retries=2)
    bundle = _make_test_bundle()

    success_payload = {
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

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(success_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    mock_err_429 = urllib.error.HTTPError(
        url="https://generativelanguage.googleapis.com/v1beta/interactions",
        code=429,
        msg="Rate limit",
        hdrs={"Retry-After": "0.01"},
        fp=None,
    )

    with patch("urllib.request.urlopen") as mock_url, patch("time.sleep") as mock_sleep:
        mock_url.side_effect = [mock_err_429, mock_resp]
        result = client.complete(bundle, system_prompt="System prompt", run_id=0)
        assert mock_url.call_count == 2
        assert mock_sleep.call_count == 1
        assert "escalate_human" in result


