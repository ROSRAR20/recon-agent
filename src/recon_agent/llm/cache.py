"""Deterministic on-disk response cache for LLM calls.

Keyed by (bundle_hash, prompt_hash, model_id) under results/llm_cache/.
Ensures reproducibility and eliminates redundant LLM invocations.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


class LLMCache:
    """Disk cache for LLM completions."""

    def __init__(self, cache_dir: str = "results/llm_cache") -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, bundle_hash: str, prompt_hash: str, model_id: str) -> str:
        """Derive deterministic cache filename key."""
        composite = f"{bundle_hash}:{prompt_hash}:{model_id}".encode("utf-8")
        return hashlib.sha256(composite).hexdigest()

    def get(self, bundle_hash: str, prompt_hash: str, model_id: str) -> str | None:
        """Retrieve cached raw LLM response if present, else None."""
        key = self._get_cache_key(bundle_hash, prompt_hash, model_id)
        filepath = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("raw_output")
        except Exception:
            return None

    def put(self, bundle_hash: str, prompt_hash: str, model_id: str, raw_output: str) -> None:
        """Write raw LLM response to disk cache."""
        key = self._get_cache_key(bundle_hash, prompt_hash, model_id)
        filepath = os.path.join(self.cache_dir, f"{key}.json")
        payload = {
            "bundle_hash": bundle_hash,
            "prompt_hash": prompt_hash,
            "model_id": model_id,
            "raw_output": raw_output,
        }
        temp_filepath = f"{filepath}.tmp"
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_filepath, filepath)
