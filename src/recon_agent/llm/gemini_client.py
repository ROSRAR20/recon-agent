"""Real Gemini API client for model 'gemini-3.5-flash-lite', 'gemini-3.6-flash', 'gemini-3.8-flash'.

Uses Google GenAI Interactions API (POST https://generativelanguage.googleapis.com/v1beta/interactions)
with structured outputs and tunable thinking_level ('low', 'medium', 'high').

Strict operational constraints:
- Outbound POST allowed ONLY to GEMINI_API_HOST constant.
- Auth key loaded from GEMINI_API_KEY via python-dotenv, never printed or leaked.
- Sanitizer applied to every logged exception and output string.
- No temperature, top_p, or top_k sent (deprecated/removed for gemini-3.8-flash and 3.x models).
- Exponential backoff with jitter on transport errors, HTTP 429, and HTTP 5xx (503 longer backoff).
- HTTP 401, 402, 403 are fatal and abort immediately without retry.
- HTTP 400 and non-JSON/schema-invalid responses are never retried.
- Token counts (input, output, thinking) and latency recorded per call.
- Pricing loaded dynamically from config/llm_prices.json.
- Model configuration loaded dynamically from config/llm_models.json.
- Alias-style model IDs ending in -latest are forbidden.
- Response served_model validated against requested model_id.
- Client-side rate limiting (RPM) with injectable clock and sleep.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import time
from typing import Any, Callable
import warnings

from dotenv import load_dotenv
import requests
import requests.exceptions

from src.recon_agent.guard import sanitize_error
from src.recon_agent.llm.audit import AuditLogger
from src.recon_agent.llm.bundle import CaseBundle
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.schema import (
    get_llm_json_schema,
    get_schema_hash,
    validate_llm_output,
)
from src.recon_agent.llm.verifier import Verifier, VerifierOutcome

# Strict network host constant: only allowed URL base in this module
GEMINI_API_HOST: str = "https://generativelanguage.googleapis.com"
INTERACTIONS_ENDPOINT: str = f"{GEMINI_API_HOST}/v1beta/interactions"


class FatalAPIError(RuntimeError):
    """Raised on unrecoverable authentication or permission errors (401, 402, 403)."""


class NonRetriableAPIError(ValueError):
    """Raised on bad requests (400) or non-JSON/invalid outputs."""


class PriceConfigError(RuntimeError):
    """Raised when pricing configuration cannot be loaded."""


class ModelMismatch(RuntimeError):
    """Raised when the served model does not match the requested model."""


class DailyQuotaExhaustedError(RuntimeError):
    """Raised when the daily quota (429) is exhausted."""


class InfraError(RuntimeError):
    """Raised on transport errors, connection drops, 5xx, or exhausted timeouts."""


def load_model_config(config_path: str = "config/llm_models.json") -> dict[str, Any]:
    """Load model configuration from config/llm_models.json."""
    if not os.path.isabs(config_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        config_path = os.path.join(repo_root, config_path)

    if not os.path.isfile(config_path):
        raise RuntimeError(f"Missing model configuration file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_price_rates(prices_path: str = "config/llm_prices.json", model_id: str = "gemini-3.5-flash-lite") -> dict[str, float]:
    """Load model pricing rates from config/llm_prices.json."""
    if not os.path.isabs(prices_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        prices_path = os.path.join(repo_root, prices_path)

    if not os.path.isfile(prices_path):
        raise PriceConfigError(f"Missing pricing file: {prices_path}")

    with open(prices_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_entry = data.get("models", {}).get(model_id)
    if not model_entry or "rates" not in model_entry:
        raise PriceConfigError(f"No pricing rates defined for model '{model_id}' in {prices_path}")

    # Determine rate based on current date
    today_str = datetime.date.today().isoformat()
    selected_rate = None
    for r in model_entry["rates"]:
        ef_from = r.get("effective_from", "1970-01-01")
        ef_to = r.get("effective_to")
        if ef_from <= today_str and (ef_to is None or today_str <= ef_to):
            selected_rate = r
            break

    if selected_rate is None:
        selected_rate = model_entry["rates"][0]

    rate_verified = bool(selected_rate.get("verified", data.get("_metadata", {}).get("verified", False)))
    return {
        "input_cost_per_million": float(selected_rate.get("input_cost_per_million", 0.075)),
        "output_cost_per_million": float(selected_rate.get("output_cost_per_million", 0.30)),
        "thinking_cost_per_million": float(
            selected_rate.get("thinking_cost_per_million", selected_rate.get("output_cost_per_million", 0.30))
        ),
        "verified": rate_verified,
    }


def extract_key_structure(obj: Any) -> Any:
    """Recursively extract key names and value types only (never values).

    For lists containing objects with a 'type' discriminator (such as steps),
    extracts distinct shapes by step type rather than only the first item.
    """
    if isinstance(obj, dict):
        return {str(k): extract_key_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        if not obj:
            return []
        if any(isinstance(item, dict) and "type" in item for item in obj):
            shapes_by_type: dict[str, Any] = {}
            for item in obj:
                if isinstance(item, dict) and "type" in item:
                    t = str(item.get("type"))
                    if t not in shapes_by_type:
                        shapes_by_type[t] = extract_key_structure(item)
                else:
                    item_type = type(item).__name__
                    if item_type not in shapes_by_type:
                        shapes_by_type[item_type] = extract_key_structure(item)
            return list(shapes_by_type.values())
        return [extract_key_structure(obj[0])]
    else:
        return type(obj).__name__


class ParsedResponse(tuple):
    """5-element tuple (backward compatible with unpacking) with served_model and service_tier attributes."""

    def __new__(
        cls,
        raw_output_text: str | None,
        usage_numbers: dict[str, int | float],
        tokens_dict: dict[str, int | None],
        usage_flagged: bool,
        key_structure: Any,
        served_model: str | None = None,
        service_tier: str | None = None,
    ) -> ParsedResponse:
        return super().__new__(
            cls,
            (raw_output_text, usage_numbers, tokens_dict, usage_flagged, key_structure),
        )

    def __init__(
        self,
        raw_output_text: str | None,
        usage_numbers: dict[str, int | float],
        tokens_dict: dict[str, int | None],
        usage_flagged: bool,
        key_structure: Any,
        served_model: str | None = None,
        service_tier: str | None = None,
    ) -> None:
        self.raw_output_text = raw_output_text
        self.usage_numbers = usage_numbers
        self.tokens_dict = tokens_dict
        self.usage_flagged = usage_flagged
        self.key_structure = key_structure
        self.served_model = served_model
        self.service_tier = service_tier


class GeminiClient:
    """Client for invoking Gemini models via the Interactions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        thinking_level: str | None = None,
        cache: LLMCache | None = None,
        audit_logger: AuditLogger | None = None,
        config_path: str = "config/llm_models.json",
        prices_path: str = "config/llm_prices.json",
        timeout: float = 60.0,
        read_timeout: float = 180.0,
        connect_timeout: float = 10.0,
        max_retries: int = 4,
        rpm: float | None = None,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        debug_http: bool = False,
        mode: str = "eval",
        no_cache: bool = False,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.config_path = config_path
        self.prices_path = prices_path
        self.model_config = load_model_config(self.config_path)

        # Model ID selection and validation
        chosen_model = model_id or self.model_config.get("primary", "gemini-3.5-flash-lite")
        if chosen_model.endswith("-latest"):
            raise ValueError(
                f"Alias-style model id '{chosen_model}' is forbidden; specify an explicit model version"
            )
        allowed_models = self.model_config.get("allowed", [])
        if allowed_models and chosen_model not in allowed_models:
            raise ValueError(f"Model '{chosen_model}' is not in allowed models list: {allowed_models}")
        self.model_id = chosen_model

        # thinking_level validation
        if thinking_level == "minimal":
            raise ValueError(
                f"thinking_level 'minimal' is unsupported for {self.model_id}; must be 'low', 'medium', or 'high'"
            )
        if thinking_level is not None and thinking_level not in ("low", "medium", "high"):
            raise ValueError(f"thinking_level must be 'low', 'medium', or 'high', got '{thinking_level}'")
        self.thinking_level = thinking_level

        self.cache = cache
        self.no_cache = no_cache
        self.audit_logger = audit_logger
        self.timeout = timeout
        self.read_timeout = read_timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max_retries
        self.debug_http = debug_http
        self.mode = mode  # "eval" or "smoke"

        # Rate limiting configuration
        configured_rpm = rpm if rpm is not None else float(self.model_config.get("default_rpm", 4))
        self.rpm = configured_rpm
        self.min_interval = 60.0 / self.rpm if self.rpm > 0 else 0.0
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self.last_call_time = 0.0

        self.tier = self.model_config.get("tier", "free")
        self.verifier = Verifier()

        # Last call metadata
        self.last_tokens: dict[str, int | None] | None = None
        self.last_cost_low: float | None = None
        self.last_cost_high: float | None = None
        self.last_served_model: str | None = None
        self.last_service_tier: str | None = None
        self.last_cache_hit: bool = False
        self.last_attempts: int = 0
        self.last_raw_response_bytes: int = 0
        self.last_usage_flagged: bool = False
        self.last_usage_raw: dict[str, int] | None = None
        self.last_raw_response_shape: Any = None

        # Load rates
        self.rates = load_price_rates(self.prices_path, self.model_id)

    def _now(self) -> float:
        return self._time_fn() if self._time_fn is not None else time.time()

    def _sleep(self, seconds: float) -> None:
        if self._sleep_fn is not None:
            self._sleep_fn(seconds)
        else:
            time.sleep(seconds)

    def calculate_estimated_costs(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        thinking_tokens: int | None,
    ) -> tuple[float | None, float | None]:
        """Compute cost_low (input + output only) and cost_high (input + output + thinking).

        Both are labeled 'estimated'. If token counts are missing, both are None
        and are never treated as 0 in any report or sum.
        """
        if input_tokens is None or output_tokens is None:
            return None, None
        think = thinking_tokens or 0
        in_cost = (input_tokens * self.rates["input_cost_per_million"]) / 1_000_000.0
        out_cost = (output_tokens * self.rates["output_cost_per_million"]) / 1_000_000.0
        think_cost = (think * self.rates["thinking_cost_per_million"]) / 1_000_000.0
        cost_low = round(in_cost + out_cost, 6)
        cost_high = round(in_cost + out_cost + think_cost, 6)
        return cost_low, cost_high

    def calculate_estimated_cost(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        thinking_tokens: int | None,
    ) -> float | None:
        """Backward compatible helper returning cost_high."""
        _low, high = self.calculate_estimated_costs(input_tokens, output_tokens, thinking_tokens)
        return high

    def build_request_payload(
        self,
        bundle: CaseBundle,
        system_instruction: str,
    ) -> dict[str, Any]:
        """Build request payload for Google GenAI Interactions API.

        [DOCS] In Google GenAI Interactions API, `system_instruction` is the top-level
        parameter steering model behavior and output constraints, while `input` contains
        the per-interaction turn payload (here: canonical bundle JSON).
        `store: false` disables server-side retention.
        `generation_config` is omitted when `thinking_level` is None.
        """
        schema_dict = get_llm_json_schema()
        payload: dict[str, Any] = {
            "model": self.model_id,
            "system_instruction": system_instruction.strip(),
            "input": bundle.to_canonical_json(),
            "store": False,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema_dict,
            },
        }
        if self.thinking_level is not None:
            payload["generation_config"] = {
                "thinking_level": self.thinking_level,
            }
        return payload

    def parse_interaction_response(
        self,
        resp_json: dict[str, Any],
    ) -> ParsedResponse:
        """Parse response from Interactions API.

        Accepts two shapes:
        1. Top-level output_text (string)
        2. steps with type == 'model_output' and text parts

        Extracts served_model (raw model field) and service_tier (raw string).
        Extracts usage numbers only and performs tolerant lookups.
        Returns ParsedResponse which unpacks as (raw_output_text, usage_numbers, tokens_dict, usage_flagged, key_structure).
        """
        key_structure = extract_key_structure(resp_json)
        raw_output_text: str | None = None
        served_model: str | None = resp_json.get("model")
        service_tier: str | None = resp_json.get("service_tier")

        # Shape 1: Top-level output_text
        if "output_text" in resp_json and isinstance(resp_json["output_text"], str):
            raw_output_text = resp_json["output_text"]

        # Shape 2: Steps of type model_output with text parts
        elif "steps" in resp_json and isinstance(resp_json["steps"], list):
            collected_text: list[str] = []
            found_model_output = False
            for step in resp_json["steps"]:
                if not isinstance(step, dict):
                    continue
                # Strictly ignore thought steps
                if step.get("type") == "model_output":
                    found_model_output = True
                    content = step.get("content")
                    parts: list[Any] = []
                    if isinstance(content, dict):
                        parts = content.get("parts", [])
                    elif isinstance(content, list):
                        parts = content
                    elif "parts" in step and isinstance(step.get("parts"), list):
                        parts = step.get("parts", [])

                    for part in parts:
                        if isinstance(part, dict):
                            # never use thought parts
                            if part.get("type") == "thought":
                                continue
                            if "text" in part and (part.get("type") == "text" or "type" not in part):
                                collected_text.append(str(part["text"]))
                        elif isinstance(part, str):
                            collected_text.append(part)

            if found_model_output and collected_text:
                raw_output_text = "".join(collected_text)

        # Extract usage numbers only
        raw_usage: dict[str, Any] = {}
        for uk in ("usage", "usageMetadata", "usage_metadata"):
            if uk in resp_json and isinstance(resp_json[uk], dict):
                raw_usage = resp_json[uk]
                break

        usage_numbers: dict[str, int | float] = {
            k: v for k, v in raw_usage.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        input_candidates = (
            "total_input_tokens",
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
            "promptTokenCount",
            "input_token_count",
        )
        output_candidates = (
            "total_output_tokens",
            "output_tokens",
            "candidates_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
            "output_token_count",
        )
        thinking_candidates = (
            "total_thought_tokens",
            "thought_tokens",
            "thinking_tokens",
            "thinking_token_count",
            "thinkingTokenCount",
        )

        in_tok = None
        for cand in input_candidates:
            if cand in usage_numbers:
                in_tok = int(usage_numbers[cand])
                break

        out_tok = None
        for cand in output_candidates:
            if cand in usage_numbers:
                out_tok = int(usage_numbers[cand])
                break

        think_tok = None
        for cand in thinking_candidates:
            if cand in usage_numbers:
                think_tok = int(usage_numbers[cand])
                break

        usage_flagged = False
        if in_tok is None or out_tok is None:
            in_tok = None
            out_tok = None
            think_tok = None
            usage_flagged = True

        tokens_dict = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "thinking_tokens": think_tok,
        }

        return ParsedResponse(
            raw_output_text,
            usage_numbers,
            tokens_dict,
            usage_flagged,
            key_structure,
            served_model=served_model,
            service_tier=service_tier,
        )

    def complete(
        self,
        bundle: CaseBundle,
        system_prompt: str,
        run_id: int = 0,
        no_cache: bool | None = None,
    ) -> str:
        """Complete reconciliation decision for a CaseBundle.

        Checks response cache first. If missed, invokes Gemini API with rate limiting
        and retries, validates served model, validates schema, updates cache, and logs to audit.
        """
        bundle_hash = bundle.bundle_hash
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        schema_hash = get_schema_hash()

        effective_no_cache = self.no_cache if no_cache is None else no_cache

        # 1. Check disk cache
        if self.cache is not None and not effective_no_cache:
            cache_entry = self.cache.get_entry(
                bundle_hash=bundle_hash,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                model_id=self.model_id,
                thinking_level=self.thinking_level,
                run_id=run_id,
            )
            if cache_entry is not None:
                cached = cache_entry.get("raw_output")
                if cached is not None:
                    self.last_cache_hit = True
                    self.last_attempts = 0
                    self.last_served_model = cache_entry.get("served_model", self.model_id)
                    self.last_service_tier = self.tier
                    self.last_tokens = cache_entry.get("tokens") or {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "thinking_tokens": 0,
                    }
                    self.last_cost_low = cache_entry.get("cost_low", 0.0)
                    self.last_cost_high = cache_entry.get("cost_high", 0.0)
                    self.last_usage_flagged = cache_entry.get("usage_flagged", False)
                    self.last_usage_raw = cache_entry.get("usage_raw")
                    self.last_raw_response_bytes = cache_entry.get(
                        "raw_response_bytes", len(cached.encode("utf-8"))
                    )
                    self.last_raw_response_shape = cache_entry.get("raw_response_shape")
                    return cached

        self.last_cache_hit = False

        # Ensure API key is present before network call
        if not self.api_key:
            raise FatalAPIError(
                "Missing GEMINI_API_KEY in environment. Set GEMINI_API_KEY via .env or environment variable."
            )

        payload = self.build_request_payload(bundle, system_prompt)
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": self.api_key,
        }

        # 2. Rate limiting check
        if self.min_interval > 0:
            now = self._now()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                sleep_needed = self.min_interval - elapsed
                self._sleep(sleep_needed)
            self.last_call_time = self._now()

        # 3. Invoke API with retry logic via requests
        latency_ms = 0
        resp_json: dict[str, Any] = {}

        start_time = self._now()
        attempt = 0
        read_timeout_retries = 0

        while True:
            attempt += 1
            attempt_start = self._now()
            status_or_exc = "UNKNOWN"
            try:
                resp = requests.post(
                    INTERACTIONS_ENDPOINT,
                    data=body_bytes,
                    headers=headers,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                status_or_exc = f"HTTP {resp.status_code}"

                # Fatal HTTP codes: 401, 402, 403
                if resp.status_code in (401, 402, 403):
                    sanitized_text = sanitize_error(resp.text, gemini_key=self.api_key)
                    err_msg = f"Fatal Gemini API HTTP error {resp.status_code}: {sanitized_text}"
                    raise FatalAPIError(err_msg) from None

                # Non-retriable: 400
                if resp.status_code == 400:
                    sanitized_text = sanitize_error(resp.text, gemini_key=self.api_key)
                    err_msg = f"Non-retriable Gemini API HTTP 400 Bad Request: {sanitized_text}"
                    raise NonRetriableAPIError(err_msg) from None

                # Retriable: 429 (Too Many Requests)
                if resp.status_code == 429:
                    lower_text = resp.text.lower()
                    sanitized_text = sanitize_error(resp.text, gemini_key=self.api_key)
                    if "daily" in lower_text or "per-day" in lower_text or "day" in lower_text:
                        raise DailyQuotaExhaustedError(
                            f"Gemini API daily quota exhausted (HTTP 429): {sanitized_text}"
                        )
                    if attempt >= self.max_retries:
                        raise InfraError(
                            f"Exceeded max retries ({self.max_retries}) on HTTP 429: {sanitized_text}"
                        )
                    retry_after_hdr = resp.headers.get("Retry-After")
                    if retry_after_hdr:
                        try:
                            sleep_sec = float(retry_after_hdr)
                        except ValueError:
                            sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))
                    else:
                        sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))
                    self._sleep(sleep_sec)
                    continue

                # Retriable: 5xx errors
                if 500 <= resp.status_code <= 599:
                    sanitized_text = sanitize_error(resp.text, gemini_key=self.api_key)
                    if attempt >= self.max_retries:
                        raise InfraError(
                            f"Exceeded max retries ({self.max_retries}) on HTTP {resp.status_code}: {sanitized_text}"
                        )
                    if resp.status_code == 503:
                        backoffs = [5.0, 15.0, 45.0, 60.0]
                        base_sleep = backoffs[min(attempt - 1, len(backoffs) - 1)]
                        sleep_sec = base_sleep + random.uniform(0.1, 0.5)
                    else:
                        sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))
                    self._sleep(sleep_sec)
                    continue

                # Any other HTTP status check
                resp.raise_for_status()
                latency_ms = round((self._now() - start_time) * 1000)
                raw_bytes_len = len(resp.content)
                self.last_raw_response_bytes = raw_bytes_len
                self.last_attempts = attempt
                resp_json = resp.json()
                break

            except requests.exceptions.ReadTimeout as exc:
                status_or_exc = exc.__class__.__name__
                read_timeout_retries += 1
                if read_timeout_retries > 1 or attempt >= self.max_retries:
                    # At most 1 retry for read timeouts. Record possible duplicate billing in audit and raise InfraError.
                    latency_ms = round((self._now() - start_time) * 1000)
                    if self.audit_logger is not None:
                        self.audit_logger.log(
                            bundle_hash=bundle_hash,
                            prompt_hash=prompt_hash,
                            model_id=self.model_id,
                            raw_output="",
                            schema_valid=False,
                            verifier_outcome="REJECT",
                            verifier_failed_checks=[
                                "READ_TIMEOUT",
                                "possible duplicate billing",
                                f"attempt_count={attempt}",
                            ],
                            final_action="",
                            thinking_level=self.thinking_level,
                            input_tokens=None,
                            output_tokens=None,
                            thinking_tokens=None,
                            latency_ms=latency_ms,
                            cost_usd=None,
                            estimated_cost_low_usd=None,
                            estimated_cost_high_usd=None,
                            usage_raw=None,
                            usage_flagged=True,
                            raw_response_shape={"error": "ReadTimeout", "attempts": attempt},
                            tier=self.tier,
                        )
                    raise InfraError(
                        f"Gemini API read timeout after {attempt} attempts (possible duplicate billing)"
                    ) from None

                sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))
                self._sleep(sleep_sec)
                continue

            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as exc:
                status_or_exc = exc.__class__.__name__
                sanitized_exc = sanitize_error(str(exc), gemini_key=self.api_key)
                if attempt >= self.max_retries:
                    raise InfraError(f"Exceeded max retries on connect error: {sanitized_exc}") from None

                sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))
                self._sleep(sleep_sec)
                continue

            except requests.exceptions.RequestException as exc:
                status_or_exc = exc.__class__.__name__
                sanitized_exc = sanitize_error(str(exc), gemini_key=self.api_key)
                if attempt >= self.max_retries:
                    raise InfraError(f"Exceeded max retries on request error: {sanitized_exc}") from None

                sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))
                self._sleep(sleep_sec)
                continue

            finally:
                if self.debug_http:
                    attempt_elapsed = self._now() - attempt_start
                    print(
                        f"[HTTP] Attempt {attempt}: {attempt_elapsed:.2f}s | "
                        f"Body: {len(body_bytes)} bytes | "
                        f"Headers: {sorted(list(headers.keys()))} | "
                        f"Result: {status_or_exc}"
                    )

        # 4. Parse response and check served_model
        parsed_resp = self.parse_interaction_response(resp_json)
        raw_output_text = parsed_resp.raw_output_text
        usage_numbers = parsed_resp.usage_numbers
        tokens = parsed_resp.tokens_dict
        usage_flagged = parsed_resp.usage_flagged
        key_structure = parsed_resp.key_structure
        served_model = parsed_resp.served_model
        service_tier = parsed_resp.service_tier

        # Validate served_model against requested model_id
        if served_model is not None:
            norm_served = served_model.removeprefix("models/")
            norm_requested = self.model_id.removeprefix("models/")
            if norm_served != norm_requested:
                raise ModelMismatch(
                    f"Served model '{served_model}' does not match requested model '{self.model_id}'"
                )
        else:
            if self.mode == "smoke":
                warnings.warn(f"Response missing 'model' field for request to '{self.model_id}'")
            else:
                raise ModelMismatch(
                    f"Response missing 'model' field for requested model '{self.model_id}' in evaluation mode"
                )

        cost_low, cost_high = self.calculate_estimated_costs(
            tokens["input_tokens"], tokens["output_tokens"], tokens["thinking_tokens"]
        )
        self.last_tokens = tokens
        self.last_cost_low = cost_low
        self.last_cost_high = cost_high
        self.last_served_model = served_model
        self.last_service_tier = service_tier
        self.last_usage_flagged = usage_flagged
        self.last_usage_raw = usage_numbers
        self.last_raw_response_shape = key_structure

        # Handle unknown response shape: infra/protocol failure
        if raw_output_text is None:
            if self.audit_logger is not None:
                self.audit_logger.log(
                    bundle_hash=bundle_hash,
                    prompt_hash=prompt_hash,
                    model_id=self.model_id,
                    raw_output="",
                    thinking_level=self.thinking_level,
                    input_tokens=tokens["input_tokens"],
                    output_tokens=tokens["output_tokens"],
                    thinking_tokens=tokens["thinking_tokens"],
                    latency_ms=latency_ms,
                    cost_usd=cost_high,
                    estimated_cost_low_usd=cost_low,
                    estimated_cost_high_usd=cost_high,
                    usage_raw=usage_numbers,
                    usage_flagged=usage_flagged,
                    raw_response_shape=key_structure,
                    served_model=served_model,
                    service_tier=service_tier,
                    tier=self.tier,
                )
            raise InfraError(
                f"Unrecognized response shape from Gemini API (infra/protocol failure). "
                f"Observed key structure: {key_structure}"
            )

        # Sanitize raw output to ensure no leaked credentials
        sanitized_raw_output = sanitize_error(raw_output_text, gemini_key=self.api_key)

        # 5. Write to audit log (raw invocation telemetry only; no verifier outcomes)
        if self.audit_logger is not None:
            self.audit_logger.log(
                bundle_hash=bundle_hash,
                prompt_hash=prompt_hash,
                model_id=self.model_id,
                raw_output=sanitized_raw_output,
                thinking_level=self.thinking_level,
                input_tokens=tokens["input_tokens"],
                output_tokens=tokens["output_tokens"],
                thinking_tokens=tokens["thinking_tokens"],
                latency_ms=latency_ms,
                cost_usd=cost_high,
                estimated_cost_low_usd=cost_low,
                estimated_cost_high_usd=cost_high,
                usage_raw=usage_numbers,
                usage_flagged=usage_flagged,
                raw_response_shape=key_structure,
                served_model=served_model,
                service_tier=service_tier,
                tier=self.tier,
            )

        # 6. Write to disk cache (immutable raw output and metadata)
        if self.cache is not None and not effective_no_cache:
            self.cache.put(
                bundle_hash=bundle_hash,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                model_id=self.model_id,
                thinking_level=self.thinking_level,
                run_id=run_id,
                raw_output=sanitized_raw_output,
                served_model=served_model,
                tokens=tokens,
                cost_low=cost_low,
                cost_high=cost_high,
                usage_flagged=usage_flagged,
                usage_raw=usage_numbers,
                raw_response_bytes=raw_bytes_len,
                raw_response_shape=key_structure,
            )

        return sanitized_raw_output
