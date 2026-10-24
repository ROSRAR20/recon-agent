"""Tests for Razorpay live-key guard."""

import pytest
from src.recon_agent.guard import LiveKeySecurityError, validate_key_id


def test_guard_rejects_live_key():
    """Live-key guard must reject any key starting with 'rzp_live_'."""
    with pytest.raises(LiveKeySecurityError, match="LIVE KEY DETECTED"):
        validate_key_id("rzp_live_abc1234567890abcdef")


def test_guard_accepts_test_key():
    """Live-key guard must accept keys starting with 'rzp_test_'."""
    # Should not raise any exception
    validate_key_id("rzp_test_abc1234567890abcdef")
    validate_key_id("rzp_test_key1")


@pytest.mark.parametrize(
    "invalid_key",
    [
        "",
        None,
        "live_12345",
        "test_12345",
        "rzp_prod_12345",
        "rzp_test",  # Missing trailing underscore
        "12345",
    ],
)
def test_guard_rejects_invalid_or_missing_keys(invalid_key):
    """Guard must reject empty, None, or improperly formatted key IDs."""
    with pytest.raises(LiveKeySecurityError):
        validate_key_id(invalid_key)
