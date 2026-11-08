"""Deterministic on-disk response cache for LLM calls.

Keyed by (bundle_hash, prompt_hash, schema_hash, model_id, thinking_level, run_id)
under results/llm_cache/. Ensures reproducibility and eliminates redundant LLM invocations.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


class CacheCorruptionError(Exception):
    """Raised when on-disk cache content fails SHA-256 hash integrity check."""


class CacheKeyExistsError(FileExistsError):
    """Raised when an attempt is made to overwrite an immutable cache entry."""


class LLMCache:
    """Disk cache for LLM completions."""

    def __init__(self, cache_dir: str = "results/llm_cache") -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(
        self,
        bundle_hash: str,
        prompt_hash: str,
        schema_hash: str = "",
        model_id: str = "",
        thinking_level: str | None = "low",
        run_id: int = 0,
    ) -> str:
        """Derive deterministic cache filename key."""
        composite = (
            f"{bundle_hash}:{prompt_hash}:{schema_hash}:{model_id}:{thinking_level}:{run_id}".encode("utf-8")
        )
        return hashlib.sha256(composite).hexdigest()

    def get_entry(
        self,
        bundle_hash: str,
        prompt_hash: str,
        schema_hash: str = "",
        model_id: str = "",
        thinking_level: str | None = "low",
        run_id: int = 0,
    ) -> dict[str, Any] | None:
        """Retrieve full cached entry dictionary if present, else None."""
        if model_id == "" and schema_hash != "" and len(schema_hash) != 64:
            model_id = schema_hash
            schema_hash = ""

        key = self._get_cache_key(
            bundle_hash=bundle_hash,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            model_id=model_id,
            thinking_level=thinking_level,
            run_id=run_id,
        )
        filepath = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise CacheCorruptionError(f"Cache entry '{key}' is invalid JSON: {exc}") from exc
        except Exception:
            return None

        raw_output = data.get("raw_output")
        if raw_output is not None:
            expected_hash = data.get("content_hash") or data.get("raw_output_hash")
            if expected_hash is not None:
                actual_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
                if actual_hash != expected_hash:
                    raise CacheCorruptionError(
                        f"Cache entry '{key}' is corrupted: expected {expected_hash}, got {actual_hash}"
                    )
        return data

    def get(
        self,
        bundle_hash: str,
        prompt_hash: str,
        schema_hash: str = "",
        model_id: str = "",
        thinking_level: str | None = "low",
        run_id: int = 0,
    ) -> str | None:
        """Retrieve cached raw LLM response if present, else None."""
        entry = self.get_entry(
            bundle_hash=bundle_hash,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            model_id=model_id,
            thinking_level=thinking_level,
            run_id=run_id,
        )
        if entry is not None:
            return entry.get("raw_output")
        return None

    def put(
        self,
        bundle_hash: str,
        prompt_hash: str,
        schema_hash: str = "",
        model_id: str = "",
        thinking_level: str | None = "low",
        run_id: int = 0,
        raw_output: str = "",
        served_model: str | None = None,
        tokens: dict[str, int | None] | None = None,
        cost_low: float | None = None,
        cost_high: float | None = None,
        usage_flagged: bool = False,
        usage_raw: dict[str, int] | None = None,
        raw_response_bytes: int = 0,
        raw_response_shape: Any = None,
    ) -> None:
        """Write raw LLM response and metadata to disk cache."""
        # Backward compatibility for legacy 4-arg positional call: put(bundle_hash, prompt_hash, model_id, raw_output)
        if raw_output == "" and model_id != "" and (model_id.startswith("{") or model_id.startswith("[")):
            raw_output = model_id
            model_id = schema_hash
            schema_hash = ""
        key = self._get_cache_key(
            bundle_hash=bundle_hash,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            model_id=model_id,
            thinking_level=thinking_level,
            run_id=run_id,
        )
        filepath = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.isfile(filepath):
            raise CacheKeyExistsError(
                f"Cache entry for key '{key}' already exists and is immutable."
            )
        content_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        payload = {
            "bundle_hash": bundle_hash,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "model_id": model_id,
            "thinking_level": thinking_level,
            "run_id": run_id,
            "raw_output": raw_output,
            "content_hash": content_hash,
            "raw_output_hash": content_hash,
            "served_model": served_model,
            "tokens": tokens,
            "cost_low": cost_low,
            "cost_high": cost_high,
            "usage_flagged": usage_flagged,
            "usage_raw": usage_raw,
            "raw_response_bytes": raw_response_bytes,
            "raw_response_shape": raw_response_shape,
        }
        temp_filepath = f"{filepath}.tmp"
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_filepath, filepath)
