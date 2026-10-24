"""Tests for masking and redaction helpers."""

from src.recon_agent.guard import extract_schema, mask_key_id, redact_id


def test_mask_key_id_length_and_prefix():
    """Key ID masking must output at most 9 characters and never expose full ID."""
    test_key = "rzp_test_9876543210abcdef"
    masked = mask_key_id(test_key)
    assert masked == "rzp_test_"
    assert len(masked) <= 9
    assert "9876543210abcdef" not in masked

    live_key = "rzp_live_secretkeyhere123"
    masked_live = mask_key_id(live_key)
    assert masked_live == "rzp_live_"
    assert len(masked_live) <= 9
    assert "secretkeyhere123" not in masked_live


def test_mask_key_id_handles_empty():
    """Empty or None key IDs return safe placeholder."""
    assert mask_key_id("") == "[EMPTY_KEY]"
    assert mask_key_id(None) == "[EMPTY_KEY]"


def test_redact_id_entity_prefixes():
    """Entity IDs must be redacted to prefix + '****'."""
    assert redact_id("pay_K83jdn291La0") == "pay_****"
    assert redact_id("order_Jk9281nla00") == "order_****"
    assert redact_id("rfnd_0192hjs817a") == "rfnd_****"
    assert redact_id("setl_910283ha910") == "setl_****"
    assert redact_id("cust_1234567890") == "cust_****"

    # Non-standard or missing prefix
    assert redact_id("invalidid") == "****"
    assert redact_id("") == "****"
    assert redact_id(None) == "****"


def test_extract_schema_contains_strictly_no_values():
    """Schema extractor must only produce type names and never leak data values."""
    sample_payload = {
        "id": "pay_SecretId9999",
        "entity": "payment",
        "amount": 54000,
        "currency": "INR",
        "status": "captured",
        "order_id": "order_SecretOrder11",
        "invoice_id": None,
        "international": False,
        "method": "upi",
        "notes": {
            "merchant_order_ref": "ORD_REF_881273",
            "internal_flag": 1,
        },
        "acquirer_data": {
            "rrn": "981273918231",
            "upi_transaction_id": "UPI_918239128",
        },
    }

    schema = extract_schema(sample_payload)

    # Verify field types
    assert schema["id"] == "str"
    assert schema["entity"] == "str"
    assert schema["amount"] == "int"
    assert schema["currency"] == "str"
    assert schema["status"] == "str"
    assert schema["invoice_id"] == "NoneType"
    assert schema["international"] == "bool"
    assert schema["notes"]["merchant_order_ref"] == "str"
    assert schema["notes"]["internal_flag"] == "int"
    assert schema["acquirer_data"]["rrn"] == "str"

    # Verify no raw values or sensitive strings leak into schema
    schema_str = str(schema)
    forbidden_values = [
        "pay_SecretId9999",
        "54000",
        "INR",
        "captured",
        "order_SecretOrder11",
        "ORD_REF_881273",
        "981273918231",
        "UPI_918239128",
    ]
    for val in forbidden_values:
        assert val not in schema_str
