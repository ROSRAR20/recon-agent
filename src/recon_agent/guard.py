"""Security, live-key guard, and redaction helpers for recon-agent.

Ensures strict test-mode safety, masks credentials, and redacts sensitive entity identifiers.
"""

import os
import re
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


def _url_encode_simple(text: str) -> str:
    """URL encode characters without importing urllib to maintain strict network module isolation."""
    safe_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~")
    return "".join(c if c in safe_chars else f"%{ord(c):02X}" for c in text)


def sanitize_error(
    text_or_exc: Any,
    gemini_key: str | None = None,
    razorpay_key_id: str | None = None,
    razorpay_key_secret: str | None = None,
) -> str:
    """Sanitize error messages or exception strings.

    Scrubs:
    1. Exact loaded GEMINI_API_KEY value and its URL-encoded forms
    2. Exact loaded RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET
    3. Pattern AIza[0-9A-Za-z-_]{30,}
    4. x-goog-api-key header text
    5. rzp_(test|live)_[A-Za-z0-9]+
    6. Authorization headers and tokens
    7. Query strings in URLs
    """
    text = str(text_or_exc)

    # 1. Scrub exact loaded GEMINI_API_KEY value and its URL-encoded form first
    gkey = gemini_key or os.getenv("GEMINI_API_KEY")
    if gkey:
        text = text.replace(gkey, "[REDACTED_GEMINI_KEY]")
        text = text.replace(_url_encode_simple(gkey), "[REDACTED_GEMINI_KEY]")

    # Also scrub exact Razorpay keys from environment if present
    rk_id = razorpay_key_id or os.getenv("RAZORPAY_KEY_ID")
    if rk_id:
        text = text.replace(rk_id, "")
        text = text.replace(_url_encode_simple(rk_id), "")
    rk_sec = razorpay_key_secret or os.getenv("RAZORPAY_KEY_SECRET")
    if rk_sec:
        text = text.replace(rk_sec, "")
        text = text.replace(_url_encode_simple(rk_sec), "")

    # 2. Scrub pattern AIza... (standard Google API key format, >= 30 chars)
    text = re.sub(r"AIza[0-9A-Za-z\-_]{30,}", "[REDACTED_KEY]", text)

    # 3. Scrub x-goog-api-key header text
    text = re.sub(r"""(?i)['"]?x-goog-api-key['"]?\s*:\s*['"][^'"]*['"]""", "[REDACTED_HEADER]", text)
    text = re.sub(r"""(?i)\bx-goog-api-key\s*:[^\r\n,;}\]]*""", "[REDACTED_HEADER]", text)
    text = re.sub(r"""(?i)\bx-goog-api-key\s*=[^\r\n,;}\]]*""", "[REDACTED_HEADER]", text)
    text = re.sub(r"""(?i)\bx-goog-api-key\b""", "[REDACTED_HEADER]", text)

    # 4. Scrub rzp_(test|live)_[A-Za-z0-9]+
    text = re.sub(r"rzp_(?:test|live)_[A-Za-z0-9]+", "", text)

    # 5. Scrub Authorization header text
    text = re.sub(r"""(?i)['"]?Authorization['"]?\s*:\s*['"][^'"]*['"]""", "", text)
    text = re.sub(r"""(?i)\bAuthorization\s*:[^\r\n,;}\]]*""", "", text)
    text = re.sub(r"""(?i)\bAuthorization\s*=[^\r\n,;}\]]*""", "", text)
    text = re.sub(r"""(?i)\b(?:Basic|Bearer)\s+[A-Za-z0-9\-._~+/=]+""", "", text)
    text = re.sub(r"""(?i)\bAuthorization\b""", "", text)

    # 6. Scrub query strings in URLs
    text = re.sub(r"((?:https?://|/)[^\s\"'<>)\],?#]+)\?[^\s\"'<>)\],}]*", r"\1", text)
    text = re.sub(r"\?[a-zA-Z0-9_.~%+-]+=[^\s\"'<>)\],}]*", "", text)

    # Clean up excess spaces while preserving newlines
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


