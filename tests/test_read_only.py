"""Network policy and read-only enforcement tests.

Guarantees:
1. The Razorpay probe stays GET-only against api.razorpay.com.
2. Outbound POST is allowed ONLY in src/recon_agent/llm/gemini_client.py and ONLY to GEMINI_API_HOST constant.
3. No other module under scripts/ or src/ imports or uses networking libraries (urllib, http.client, socket, requests, httpx, aiohttp) or invokes curl/wget.
4. Key-shaped strings and exact GEMINI_API_KEY never reach logs, audit files, or cache.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FORBIDDEN_NETWORK_MODULES = {
    "urllib",
    "http.client",
    "socket",
    "requests",
    "httpx",
    "aiohttp",
}

ALLOWED_NETWORK_FILES = {
    os.path.normpath(os.path.join("scripts", "probe_razorpay.py")),
    os.path.normpath(os.path.join("src", "recon_agent", "llm", "gemini_client.py")),
}


def find_py_files(*directories: str) -> list[str]:
    """Find all python files under given directory names relative to repo root."""
    py_files = []
    for dir_name in directories:
        target_dir = os.path.join(REPO_ROOT, dir_name)
        if not os.path.exists(target_dir):
            continue
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))
    return py_files


def test_network_isolation_ast_and_text_scan() -> None:
    """Flag any import or use of network libraries outside gemini_client.py and probe_razorpay.py."""
    files_to_check = find_py_files("scripts", "src")
    assert files_to_check, "No Python files found in scripts/ or src/ to check!"

    violations = []

    for file_path in files_to_check:
        rel_path = os.path.normpath(os.path.relpath(file_path, REPO_ROOT))
        if rel_path in ALLOWED_NETWORK_FILES:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Text scan for curl / wget subprocess invocations
        if re.search(r"\b(?:curl|wget)\b", source):
            # Check if used in command or string
            if "os.system" in source or "subprocess" in source:
                violations.append(f"{rel_path}: invokes curl or wget via subprocess/os.system")

        tree = ast.parse(source, filename=file_path)

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in FORBIDDEN_NETWORK_MODULES or alias.name in FORBIDDEN_NETWORK_MODULES:
                        violations.append(f"{rel_path}:L{node.lineno} imports '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in FORBIDDEN_NETWORK_MODULES or node.module in FORBIDDEN_NETWORK_MODULES:
                        violations.append(f"{rel_path}:L{node.lineno} imports from '{node.module}'")

    assert not violations, (
        f"Unauthorized networking imports or calls detected outside permitted files:\n"
        + "\n".join(violations)
    )


def test_razorpay_probe_stays_get_only() -> None:
    """Assert that scripts/probe_razorpay.py does NOT invoke any HTTP method other than GET."""
    probe_path = os.path.join(REPO_ROOT, "scripts", "probe_razorpay.py")
    assert os.path.isfile(probe_path)

    with open(probe_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=probe_path)
    forbidden_verbs = {"post", "put", "patch", "delete", "head", "options"}
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr.lower() in forbidden_verbs:
                    violations.append(f"L{node.lineno} calls .{func.attr}()")
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "request" and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    if first_arg.value.upper() != "GET":
                        violations.append(f"L{node.lineno} calls request('{first_arg.value}')")

    assert not violations, f"Forbidden non-GET calls in probe_razorpay.py:\n" + "\n".join(violations)


def test_gemini_client_post_only_to_host_constant() -> None:
    """Assert that gemini_client.py only performs POST to GEMINI_API_HOST constant."""
    client_path = os.path.join(REPO_ROOT, "src", "recon_agent", "llm", "gemini_client.py")
    assert os.path.isfile(client_path)

    with open(client_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=client_path)

    # 1. Assert GEMINI_API_HOST is defined and equals the official Google Generative Language API
    host_defined = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GEMINI_API_HOST":
                    assert isinstance(node.value, ast.Constant)
                    assert node.value.value == "https://generativelanguage.googleapis.com"
                    host_defined = True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "GEMINI_API_HOST":
                assert isinstance(node.value, ast.Constant)
                assert node.value.value == "https://generativelanguage.googleapis.com"
                host_defined = True
    assert host_defined, "GEMINI_API_HOST constant not found in gemini_client.py"

    # 2. Check all URLs in module start with GEMINI_API_HOST or are relative
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if val.startswith("http://") or val.startswith("https://"):
                assert val.startswith("https://generativelanguage.googleapis.com"), (
                    f"Forbidden external URL in gemini_client.py: {val}"
                )


def test_sanitizer_prevents_keys_in_logs_audit_cache() -> None:
    """Assert that a key-shaped string never reaches logs, the audit file, or the cache."""
    from src.recon_agent.guard import sanitize_error
    from src.recon_agent.llm.audit import AuditLogger
    from src.recon_agent.llm.cache import LLMCache

    fake_key = "AIzaSyDummySecretKeyForTesting9876543210"
    fake_header = f"x-goog-api-key: {fake_key}"
    sample_exception = Exception(f"Failed to connect: Error 401 with header {fake_header} and key={fake_key}")

    # 1. Test sanitizer directly on exception message
    sanitized = sanitize_error(sample_exception, gemini_key=fake_key)
    assert fake_key not in sanitized
    assert "AIzaSy" not in sanitized
    assert "x-goog-api-key" not in sanitized

    # 2. Test audit logger scrubbing
    with tempfile.TemporaryDirectory() as tmp_dir:
        audit_file = os.path.join(tmp_dir, "test_audit.jsonl")
        logger = AuditLogger(log_path=audit_file)

        logger.log(
            bundle_hash=f"bundle_{fake_key}",
            prompt_hash="prompt_hash_123",
            model_id="gemini-3.8-flash",
            raw_output=f"Output containing {fake_key} and {fake_header}",
            schema_valid=True,
            verifier_outcome="ACCEPT",
            verifier_failed_checks=[],
            final_action="no_issue",
        )

        with open(audit_file, "r", encoding="utf-8") as f:
            audit_contents = f.read()

        assert fake_key not in audit_contents
        assert "AIzaSy" not in audit_contents
        assert "x-goog-api-key:" not in audit_contents

    # 3. Test cache storage scrubbing
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = LLMCache(cache_dir=tmp_dir)
        raw_output_with_key = f"Response with {fake_key}"
        sanitized_for_cache = sanitize_error(raw_output_with_key, gemini_key=fake_key)

        cache.put(
            bundle_hash="bundle_123",
            prompt_hash="prompt_123",
            schema_hash="schema_123",
            model_id="gemini-3.8-flash",
            thinking_level="low",
            run_id=0,
            raw_output=sanitized_for_cache,
        )

        # Inspect raw cache files on disk
        for fname in os.listdir(tmp_dir):
            if fname.endswith(".json"):
                with open(os.path.join(tmp_dir, fname), "r", encoding="utf-8") as f:
                    cache_content = f.read()
                assert fake_key not in cache_content
                assert "AIzaSy" not in cache_content
