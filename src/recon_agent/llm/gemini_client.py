"""Real Gemini API client for model 'gemini-3.8-flash'.

Uses Google GenAI Interactions API (POST https://generativelanguage.googleapis.com/v1beta/interactions)
with structured outputs and tunable thinking_level ('low', 'medium', 'high').

Strict operational constraints:
- Outbound POST allowed ONLY to GEMINI_API_HOST constant.
- Auth key loaded from GEMINI_API_KEY via python-dotenv, never printed or leaked.
- Sanitizer applied to every logged exception and output string.
- No temperature, top_p, or top_k sent (deprecated/removed for gemini-3.8-flash).
- Exponential backoff with jitter on transport errors, HTTP 429, and HTTP 5xx.
- HTTP 401, 402, 403 are fatal and abort immediately without retry.
- HTTP 400 and non-JSON/schema-invalid responses are never retried.
- Token counts (input, output, thinking) and latency recorded per call.
- Pricing loaded dynamically from config/llm_prices.json.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv

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


def load_price_rates(prices_path: str = "config/llm_prices.json", model_id: str = "gemini-3.8-flash") -> dict[str, float]:
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

    return {
        "input_cost_per_million": float(selected_rate.get("input_cost_per_million", 0.75)),
        "output_cost_per_million": float(selected_rate.get("output_cost_per_million", 3.75)),
        "thinking_cost_per_million": float(selected_rate.get("thinking_cost_per_million", 3.75)),
    }


class GeminiClient:
    """Client for invoking Gemini 3.8 Flash via the Interactions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "gemini-3.8-flash",
        thinking_level: str = "low",
        cache: LLMCache | None = None,
        audit_logger: AuditLogger | None = None,
        prices_path: str = "config/llm_prices.json",
        timeout: float = 60.0,
        max_retries: int = 4,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_id = model_id
        if thinking_level not in ("low", "medium", "high"):
            raise ValueError(f"thinking_level must be 'low', 'medium', or 'high', got '{thinking_level}'")
        self.thinking_level = thinking_level
        self.cache = cache
        self.audit_logger = audit_logger
        self.prices_path = prices_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.verifier = Verifier()

        # Load rates
        self.rates = load_price_rates(self.prices_path, self.model_id)

    def calculate_estimated_cost(self, input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
        """Calculate estimated cost (USD) based on loaded rates."""
        in_cost = (input_tokens * self.rates["input_cost_per_million"]) / 1_000_000.0
        out_cost = (output_tokens * self.rates["output_cost_per_million"]) / 1_000_000.0
        think_cost = (thinking_tokens * self.rates["thinking_cost_per_million"]) / 1_000_000.0
        return round(in_cost + out_cost + think_cost, 6)

    def _format_prompt(self, bundle: CaseBundle, system_prompt: str) -> str:
        """Format prompt combining system prompt and case bundle JSON."""
        return f"{system_prompt.strip()}\n\nCASE BUNDLE:\n{bundle.to_canonical_json()}"

    def complete(
        self,
        bundle: CaseBundle,
        system_prompt: str,
        run_id: int = 0,
    ) -> str:
        """Complete reconciliation decision for a CaseBundle.

        Checks response cache first. If missed, invokes Gemini API with retries,
        validates, updates cache, and logs to audit.
        """
        bundle_hash = bundle.bundle_hash
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        schema_hash = get_schema_hash()

        # 1. Check disk cache
        if self.cache is not None:
            cached = self.cache.get(
                bundle_hash=bundle_hash,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                model_id=self.model_id,
                thinking_level=self.thinking_level,
                run_id=run_id,
            )
            if cached is not None:
                return cached

        # Ensure API key is present before network call
        if not self.api_key:
            raise FatalAPIError(
                "Missing GEMINI_API_KEY in environment. Set GEMINI_API_KEY via .env or environment variable."
            )

        formatted_input = self._format_prompt(bundle, system_prompt)
        schema_dict = get_llm_json_schema()

        # Build request payload for Interactions API (strictly NO temperature, top_p, top_k)
        payload = {
            "model": self.model_id,
            "input": formatted_input,
            "generation_config": {
                "thinking_level": self.thinking_level,
            },
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema_dict,
            },
        }

        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        req = urllib.request.Request(
            url=INTERACTIONS_ENDPOINT,
            data=body_bytes,
            headers=headers,
            method="POST",
        )

        # 2. Invoke API with retry logic
        raw_output_text = ""
        input_tokens = 0
        output_tokens = 0
        thinking_tokens = 0
        latency_ms = 0

        start_time = time.time()
        attempt = 0

        while True:
            attempt += 1
            call_start = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    call_latency = round((time.time() - call_start) * 1000)
                    latency_ms = round((time.time() - start_time) * 1000)
                    resp_bytes = response.read()
                    resp_json = json.loads(resp_bytes.decode("utf-8"))

                    # Extract output text from Interactions API
                    if "output_text" in resp_json:
                        raw_output_text = str(resp_json["output_text"])
                    elif "candidates" in resp_json:
                        parts = resp_json["candidates"][0].get("content", {}).get("parts", [])
                        raw_output_text = parts[0].get("text", "") if parts else ""
                    else:
                        raw_output_text = json.dumps(resp_json)

                    # Extract usage
                    usage = resp_json.get("usage", {})
                    input_tokens = int(usage.get("total_input_tokens") or resp_json.get("usageMetadata", {}).get("promptTokenCount", 0))
                    output_tokens = int(usage.get("total_output_tokens") or resp_json.get("usageMetadata", {}).get("candidatesTokenCount", 0))
                    thinking_tokens = int(usage.get("total_thought_tokens") or resp_json.get("usageMetadata", {}).get("thinkingTokenCount", 0))
                    break

            except urllib.error.HTTPError as exc:
                sanitized_reason = sanitize_error(str(exc.reason), gemini_key=self.api_key)
                code = exc.code

                # Fatal HTTP codes: 401, 402, 403
                if code in (401, 402, 403):
                    err_msg = f"Fatal Gemini API HTTP error {code}: {sanitized_reason}"
                    raise FatalAPIError(err_msg) from exc

                # Non-retriable: 400
                if code == 400:
                    err_msg = f"Non-retriable Gemini API HTTP 400 Bad Request: {sanitized_reason}"
                    raise NonRetriableAPIError(err_msg) from exc

                # Retriable: 429 (Too Many Requests) and 5xx
                if code == 429 or 500 <= code <= 599:
                    if attempt > self.max_retries or (time.time() - start_time) > self.timeout:
                        err_msg = f"Exceeded max retries ({self.max_retries}) on HTTP {code}: {sanitized_reason}"
                        raise RuntimeError(err_msg) from exc

                    # Honor Retry-After header if present
                    retry_after_hdr = exc.headers.get("Retry-After") if hasattr(exc, "headers") else None
                    if retry_after_hdr:
                        try:
                            sleep_sec = float(retry_after_hdr)
                        except ValueError:
                            sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))
                    else:
                        sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))

                    time.sleep(sleep_sec)
                    continue

                # Any other HTTP error
                raise RuntimeError(f"Unhandled HTTP error {code}: {sanitized_reason}") from exc

            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                sanitized_exc = sanitize_error(str(exc), gemini_key=self.api_key)
                if attempt > self.max_retries or (time.time() - start_time) > self.timeout:
                    raise RuntimeError(f"Exceeded max retries on transport error: {sanitized_exc}") from exc

                sleep_sec = min(60.0, (2 ** attempt) + random.uniform(0.1, 0.5))
                time.sleep(sleep_sec)
                continue

        # Sanitize raw output to ensure no leaked credentials
        sanitized_raw_output = sanitize_error(raw_output_text, gemini_key=self.api_key)

        # 3. Validate schema (do NOT retry on schema violation)
        is_schema_valid, parsed, schema_errs = validate_llm_output(sanitized_raw_output)

        # 4. Verify outcome
        if is_schema_valid and parsed is not None:
            ver_res = self.verifier.verify(bundle, parsed)
            final_action = parsed.decision if ver_res.outcome == VerifierOutcome.ACCEPT else "escalate_human"
        else:
            ver_res = None
            final_action = "escalate_human"

        estimated_cost = self.calculate_estimated_cost(input_tokens, output_tokens, thinking_tokens)

        # 5. Write to audit log
        if self.audit_logger is not None:
            self.audit_logger.log(
                bundle_hash=bundle_hash,
                prompt_hash=prompt_hash,
                model_id=self.model_id,
                raw_output=sanitized_raw_output,
                schema_valid=is_schema_valid,
                verifier_outcome=ver_res.outcome.value if ver_res else "REJECT",
                verifier_failed_checks=ver_res.failed_checks if ver_res else ["V1"],
                final_action=final_action,
                thinking_level=self.thinking_level,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                latency_ms=latency_ms,
                cost_usd=estimated_cost,
            )

        # 6. Write to disk cache
        if self.cache is not None:
            getattr(self.cache, "put")(
                bundle_hash=bundle_hash,
                prompt_hash=prompt_hash,
                schema_hash=schema_hash,
                model_id=self.model_id,
                thinking_level=self.thinking_level,
                run_id=run_id,
                raw_output=sanitized_raw_output,
            )

        return sanitized_raw_output

