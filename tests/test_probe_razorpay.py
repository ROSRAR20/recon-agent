"""Tests for scripts/probe_razorpay.py sanitizer, retries, and preflight."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.probe_razorpay import (
    BASE_URL,
    MAX_RETRIES,
    execute_get_request,
    run_preflight,
    sanitize_error,
    write_probe_report,
)


# =========================================================================
# Sanitizer Tests
# =========================================================================

def test_sanitizer_removes_rzp_test_and_live_keys():
    """Sanitizer must remove anything matching rzp_(test|live)_[A-Za-z0-9]+."""
    raw = (
        "Failed connection using rzp_test_9876543210abcdef and backup "
        "rzp_live_secret1234567890abcdef."
    )
    sanitized = sanitize_error(raw)

    assert "rzp_test_9876543210abcdef" not in sanitized
    assert "rzp_live_secret1234567890abcdef" not in sanitized
    assert "rzp_test_" not in sanitized
    assert "rzp_live_" not in sanitized
    assert "rzp_" not in sanitized


def test_sanitizer_removes_key_id_and_secret_from_env(monkeypatch):
    """Sanitizer must remove exact values of RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET."""
    fake_key = "rzp_test_customkey123"
    fake_secret = "super_secret_shhh_9999"

    monkeypatch.setenv("RAZORPAY_KEY_ID", fake_key)
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", fake_secret)

    raw = f"Error with credentials key={fake_key} and secret={fake_secret}"
    sanitized = sanitize_error(raw)

    assert fake_key not in sanitized
    assert fake_secret not in sanitized
    assert "rzp_" not in sanitized


def test_sanitizer_removes_authorization_header_text():
    """Sanitizer must remove any Authorization header text in various formats."""
    samples = [
        "Header: Authorization: Basic dGVzdF9rZXk6dGVzdF9zZWNyZXQ=",
        "headers={'Authorization': 'Basic dGVzdF9rZXk6dGVzdF9zZWNyZXQ='}",
        'headers={"Authorization": "Bearer some_jwt_token_12345"}',
        "Authorization: Bearer my_secret_token",
        "Failed with Authorization: token_here",
        "Authorization = token_12345",
        "Request failed with Basic dGVzdF9rZXk6dGVzdF9zZWNyZXQ=",
    ]

    for sample in samples:
        sanitized = sanitize_error(sample)
        assert "Authorization" not in sanitized
        assert "dGVzdF9rZXk6dGVzdF9zZWNyZXQ=" not in sanitized
        assert "some_jwt_token_12345" not in sanitized
        assert "my_secret_token" not in sanitized
        assert "Basic" not in sanitized
        assert "Bearer" not in sanitized


def test_sanitizer_removes_query_strings_in_urls():
    """Sanitizer must remove query strings from URLs."""
    raw = (
        "Max retries exceeded with url: "
        "https://api.razorpay.com/v1/payments?count=10&from=1728000000&to=1730000000"
    )
    sanitized = sanitize_error(raw)

    assert "https://api.razorpay.com/v1/payments" in sanitized
    assert "?count=" not in sanitized
    assert "from=1728000000" not in sanitized
    assert "to=1730000000" not in sanitized

    relative_raw = "Call failed at /v1/settlements/recon/combined?year=2026&month=10&count=10"
    sanitized_rel = sanitize_error(relative_raw)
    assert "/v1/settlements/recon/combined" in sanitized_rel
    assert "?year=" not in sanitized_rel


def test_sanitizer_on_exception_message(monkeypatch):
    """Sanitizer must scrub rzp_ keys, secrets, auth headers, and query params from exceptions."""
    fake_key = "rzp_test_mock_123456"
    fake_secret = "my_top_secret_key"
    monkeypatch.setenv("RAZORPAY_KEY_ID", fake_key)
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", fake_secret)

    exc = requests.exceptions.ConnectionError(
        f"Failed to connect to https://api.razorpay.com/v1/payments?count=10&skip=0 "
        f"using key {fake_key}, secret {fake_secret}, and Authorization: Basic dXNlcjpwYXNz"
    )

    sanitized = sanitize_error(exc)

    assert fake_key not in sanitized
    assert fake_secret not in sanitized
    assert "rzp_" not in sanitized
    assert "Authorization" not in sanitized
    assert "Basic" not in sanitized
    assert "dXNlcjpwYXNz" not in sanitized
    assert "?count=10" not in sanitized
    assert "https://api.razorpay.com/v1/payments" in sanitized


def test_sanitizer_handles_plain_text():
    """Sanitizer preserves innocuous error messages."""
    text = "Connection refused by peer at port 443"
    assert sanitize_error(text) == text


# =========================================================================
# Preflight Tests
# =========================================================================

def test_preflight_success_401(capsys):
    """Preflight prints 'network OK (401 expected)' when endpoint returns 401."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("requests.get", return_value=mock_resp) as mock_get:
        run_preflight()
        mock_get.assert_called_once_with(f"{BASE_URL}/v1/payments", timeout=15)

    captured = capsys.readouterr()
    assert "network OK (401 expected)" in captured.out


def test_preflight_unexpected_status_prints_sanitized_error(capsys):
    """Preflight prints sanitized error and continues if status is not 401."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error with key rzp_test_leak123"

    with patch("requests.get", return_value=mock_resp):
        run_preflight()

    captured = capsys.readouterr()
    assert "network OK (401 expected)" not in captured.out
    assert "500" in captured.out
    assert "rzp_" not in captured.out


def test_preflight_exception_prints_sanitized_error_and_continues(capsys):
    """Preflight prints sanitized error and does not raise if an exception occurs."""
    exc = requests.exceptions.ConnectionError("DNS failure with rzp_live_key999")

    with patch("requests.get", side_effect=exc):
        # Should not raise
        run_preflight()

    captured = capsys.readouterr()
    assert "network OK (401 expected)" not in captured.out
    assert "ConnectionError" in captured.out
    assert "rzp_" not in captured.out


# =========================================================================
# Retry & Attempt Count Tests
# =========================================================================

def test_execute_get_request_retries_connection_error():
    """execute_get_request retries on ConnectionError up to MAX_RETRIES."""
    session = MagicMock()
    session.get.side_effect = [
        requests.exceptions.ConnectionError("err1"),
        requests.exceptions.ConnectionError("err2"),
        MagicMock(status_code=200),
    ]

    with patch("time.sleep"):
        resp, attempts = execute_get_request(
            session=session,
            url=f"{BASE_URL}/v1/payments",
            params={},
            auth=("k", "s"),
            max_retries=4,
        )

    assert resp.status_code == 200
    assert attempts == 3
    assert session.get.call_count == 3


def test_execute_get_request_retries_timeout():
    """execute_get_request retries on Timeout up to MAX_RETRIES."""
    session = MagicMock()
    session.get.side_effect = [
        requests.exceptions.Timeout("timed out"),
        MagicMock(status_code=200),
    ]

    with patch("time.sleep"):
        resp, attempts = execute_get_request(
            session=session,
            url=f"{BASE_URL}/v1/payments",
            params={},
            auth=("k", "s"),
            max_retries=4,
        )

    assert resp.status_code == 200
    assert attempts == 2
    assert session.get.call_count == 2


def test_execute_get_request_exhausts_retries_and_records_attempts():
    """execute_get_request records attempts count on exception when max retries exceeded."""
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("fatal connection error")

    with patch("time.sleep"):
        with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
            execute_get_request(
                session=session,
                url=f"{BASE_URL}/v1/payments",
                params={},
                auth=("k", "s"),
                max_retries=4,
            )

    assert exc_info.value.attempts == 4
    assert session.get.call_count == 4


def test_execute_get_request_retries_429_and_5xx():
    """execute_get_request retries on 429 and 5xx status codes."""
    session = MagicMock()
    resp_429 = MagicMock(status_code=429)
    resp_503 = MagicMock(status_code=503)
    resp_200 = MagicMock(status_code=200)

    session.get.side_effect = [resp_429, resp_503, resp_200]

    with patch("time.sleep"):
        resp, attempts = execute_get_request(
            session=session,
            url=f"{BASE_URL}/v1/payments",
            params={},
            auth=("k", "s"),
            max_retries=4,
        )

    assert resp.status_code == 200
    assert attempts == 3
    assert session.get.call_count == 3


# =========================================================================
# Report Generation Tests
# =========================================================================

def test_write_probe_report_headers_and_status(tmp_path, monkeypatch):
    """Report must include base host, proxy settings (yes/no), attempts, and exception class."""
    monkeypatch.setenv("HTTP_PROXY", "http://internal-proxy:8080")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)

    report_file = tmp_path / "probe_report.md"

    results = [
        {
            "endpoint": "/v1/payments",
            "period": "2026-10",
            "http_status": None,
            "exception_class": "ConnectionError",
            "attempts": 4,
            "top_level_keys": [],
            "item_count": 0,
            "schema": None,
            "error": {"code": "ConnectionError", "description": "Failed to connect"},
        },
        {
            "endpoint": "/v1/orders",
            "period": "2026-10",
            "http_status": 200,
            "exception_class": None,
            "attempts": 1,
            "top_level_keys": ["entity", "items"],
            "item_count": 5,
            "schema": None,
            "error": None,
        },
    ]

    empty_or_errored = [
        {
            "endpoint": "/v1/payments",
            "period": "2026-10",
            "status": "ConnectionError",
            "attempts": 4,
            "reason": "ConnectionError: Failed to connect",
        }
    ]

    write_probe_report(results, empty_or_errored, str(report_file))

    content = report_file.read_text(encoding="utf-8")

    # Base URL Host and Proxy presence (yes/no, never values)
    assert "**Base URL Host:** api.razorpay.com" in content
    assert "**HTTP_PROXY:** yes" in content
    assert "**HTTPS_PROXY:** no" in content
    assert "**ALL_PROXY:** no" in content
    assert "http://internal-proxy:8080" not in content

    # Status column shows exception class, not just "ERROR"
    assert "| `/v1/payments` | 2026-10 | ConnectionError | 4 | `-` | 0 |" in content
    assert "| `/v1/orders` | 2026-10 | 200 | 1 | `entity, items` | 5 |" in content

    # Error table includes attempts and detail
    assert "| `/v1/payments` | 2026-10 | ConnectionError | 4 | ConnectionError: Failed to connect |" in content
