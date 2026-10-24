"""Security, live-key guard, and redaction helpers for recon-agent.

Ensures strict test-mode safety, masks credentials, and redacts sensitive entity identifiers.
"""

from typing import Any


class LiveKeySecurityError(ValueError):
    """Raised when a live Razorpay credential or non-test key is detected."""


def validate_key_id(key_id: str | None) -> None:
    """Validate that the Razorpay Key ID strictly belongs to test mode.

    Aborts immediately before any network call if the key id does not start
    with 'rzp_test_'. Explicitly rejects 'rzp_live_' and empty/missing keys.
    """
    if not key_id or not isinstance(key_id, str):
        raise LiveKeySecurityError(
            "Missing or empty Razorpay Key ID. Must start with 'rzp_test_'."
        )

    if key_id.startswith("rzp_live_"):
        raise LiveKeySecurityError(
            f"LIVE KEY DETECTED! Network operations prohibited for live credentials. "
            f"Key prefix: {mask_key_id(key_id)}"
        )

    if not key_id.startswith("rzp_test_"):
        raise LiveKeySecurityError(
            f"Invalid Key ID prefix. Expected 'rzp_test_', got: {mask_key_id(key_id)}"
        )


def mask_key_id(key_id: str | None) -> str:
    """Mask Key ID to at most the first 9 characters (never reveals full id).

    For 'rzp_test_1234567890', returns 'rzp_test_'.
    For 'rzp_live_1234567890', returns 'rzp_live_'.
    Never outputs the API secret.
    """
    if not key_id:
        return "[EMPTY_KEY]"
    return key_id[:9]


def redact_id(entity_id: str | None) -> str:
    """Redact an entity ID to its prefix (e.g. 'pay_****', 'order_****').

    Ensures full entity identifiers are not leaked into reports or logs.
    """
    if not entity_id or not isinstance(entity_id, str):
        return "****"

    if "_" in entity_id:
        prefix = entity_id.split("_", 1)[0]
        return f"{prefix}_****"
    return "****"


def extract_schema(item: Any) -> Any:
    """Extract field name to type name schema with strictly NO values.

    Maps fields to Python/JSON type names (e.g. 'str', 'int', 'bool', 'dict', 'list').
    """
    if item is None:
        return "NoneType"

    if isinstance(item, dict):
        return {k: extract_schema(v) for k, v in item.items()}

    if isinstance(item, list):
        if not item:
            return []
        # Return list schema representation based on first entry structure
        return [extract_schema(item[0])]

    return type(item).__name__
