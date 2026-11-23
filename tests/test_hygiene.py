"""Repo hygiene and secret scanning tests.

Ensures no live API keys, private keys, or credentials are committed to git tracking.
"""
from __future__ import annotations

import os
import re
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Regex patterns for secret detection
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),               # Google API Key
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),                  # OpenAI API Key
    re.compile(r"rzp_live_[a-zA-Z0-9]{14,}"),            # Razorpay live key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),    # Private keys
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),                  # GitHub PAT
    re.compile(r"github_pat_[a-zA-Z0-9_]{50,}"),         # Fine-grained GitHub PAT
]

BINARY_EXTENSIONS = {".gz", ".zip", ".tar", ".png", ".jpg", ".jpeg", ".pyc", ".ico"}


def get_tracked_files() -> list[str]:
    cmd = ["git", "ls-files"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def test_no_secrets_in_tracked_files():
    tracked = get_tracked_files()
    assert len(tracked) > 0, "Git must track files"

    violations: list[str] = []

    # Mock keys strictly permitted within tests/ as negative-guard test fixtures
    test_mock_whitelist = {
        "rzp_live_abc1234567890abcdef",
        "rzp_live_secret1234567890abcdef",
        "rzp_live_secretkeyhere123",
        "AIzaSyDummySecretKeyForTesting9876543210",
    }

    for rel_path in tracked:
        # Skip binary extensions
        _, ext = os.path.splitext(rel_path)
        if ext.lower() in BINARY_EXTENSIONS:
            continue

        full_path = os.path.join(REPO_ROOT, rel_path.replace("/", os.sep))
        if not os.path.exists(full_path):
            continue

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue

        for pattern in SECRET_PATTERNS:
            matches = pattern.findall(content)
            for m in matches:
                if rel_path.startswith("tests/") and any(allowed.startswith(m) or m in allowed for allowed in test_mock_whitelist):
                    continue
                violations.append(f"Secret detected in {rel_path}: pattern {pattern.pattern} matched '{m}'")

    assert not violations, "Secret scan failed:\n" + "\n".join(violations)


def test_tracked_files_audit():
    from scripts.check_tracked_files import audit_tracked_files

    report = audit_tracked_files()
    assert isinstance(report["total_tracked"], list)
    assert isinstance(report["data"], list)
    assert isinstance(report["results"], list)
    # Ensure audit successfully finds and reports tracked files
    assert len(report["total_tracked"]) > 0
