#!/usr/bin/env python3
"""Read-only Razorpay API probe for recon-agent.

Queries settlement, recon, payment, order, and refund endpoints in TEST mode
for the current and previous month (calculated in IST).
Enforces live-key guard before any network access.
Uses HTTP GET requests exclusively.
"""

import datetime
import json
import os
import random
import sys
import time
import zoneinfo
from typing import Any

import requests
from dotenv import load_dotenv

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.recon_agent.guard import (
    extract_schema,
    mask_key_id,
    redact_id,
    validate_key_id,
)

BASE_URL = "https://api.razorpay.com"
TIMEOUT_SECONDS = 15
MAX_RETRIES = 4


def get_ist_periods() -> list[dict[str, Any]]:
    """Derive year and month details for current and previous month in IST."""
    ist_tz = zoneinfo.ZoneInfo("Asia/Kolkata")
    now_ist = datetime.datetime.now(ist_tz)

    current_year = now_ist.year
    current_month = now_ist.month

    if current_month == 1:
        prev_year = current_year - 1
        prev_month = 12
    else:
        prev_year = current_year
        prev_month = current_month - 1

    periods = []
    for yr, mo in [(current_year, current_month), (prev_year, prev_month)]:
        start_dt = datetime.datetime(yr, mo, 1, 0, 0, 0, tzinfo=ist_tz)
        if mo == 12:
            next_start_dt = datetime.datetime(yr + 1, 1, 1, 0, 0, 0, tzinfo=ist_tz)
        else:
            next_start_dt = datetime.datetime(yr, mo + 1, 1, 0, 0, 0, tzinfo=ist_tz)
        end_dt = next_start_dt - datetime.timedelta(seconds=1)

        periods.append(
            {
                "year": yr,
                "month": mo,
                "from_timestamp": int(start_dt.timestamp()),
                "to_timestamp": int(end_dt.timestamp()),
                "label": f"{yr}-{mo:02d}",
            }
        )
    return periods


def execute_get_request(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    auth: tuple[str, str],
) -> requests.Response:
    """Execute a single HTTP GET request with exponential backoff plus jitter.

    Retries on HTTP 429 and 5xx status codes up to MAX_RETRIES.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                params=params,
                auth=auth,
                timeout=TIMEOUT_SECONDS,
            )
            if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                sleep_sec = (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                time.sleep(sleep_sec)
                continue
            return response
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES:
                sleep_sec = (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                time.sleep(sleep_sec)
                continue
            raise


def sanitize_filename(endpoint: str, label: str) -> str:
    """Convert endpoint path and period label into a safe filename."""
    cleaned = endpoint.strip("/").replace("/", "_")
    return f"{cleaned}_{label}.json"


def run_probe() -> None:
    """Execute the read-only probe across all specified endpoints."""
    # Step 1: Load environment variables
    env_file = os.path.join(REPO_ROOT, ".env")
    load_dotenv(dotenv_path=env_file)
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")

    # Step 2: Enforce Live-Key Guard BEFORE any network call
    validate_key_id(key_id)
    masked_key = mask_key_id(key_id)
    print(f"[*] Live-key guard passed. Key ID verified: {masked_key}")

    if not key_secret:
        raise ValueError("Missing RAZORPAY_KEY_SECRET in environment.")

    auth = (key_id, key_secret)

    # Step 3: Prepare output directories
    raw_data_dir = os.path.join(REPO_ROOT, "data", "raw")
    docs_dir = os.path.join(REPO_ROOT, "docs")
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    periods = get_ist_periods()

    endpoints_to_probe = [
        "/v1/settlements/recon/combined",
        "/v1/settlements",
        "/v1/payments",
        "/v1/orders",
        "/v1/refunds",
    ]

    results: list[dict[str, Any]] = []
    empty_or_errored: list[dict[str, Any]] = []

    session = requests.Session()

    for period in periods:
        yr = period["year"]
        mo = period["month"]
        label = period["label"]

        for ep in endpoints_to_probe:
            url = f"{BASE_URL}{ep}"

            if ep == "/v1/settlements/recon/combined":
                params = {
                    "year": yr,
                    "month": f"{mo:02d}",
                    "count": 10,
                    "skip": 0,
                }
            else:
                params = {
                    "count": 10,
                    "from": period["from_timestamp"],
                    "to": period["to_timestamp"],
                    "skip": 0,
                }

            record: dict[str, Any] = {
                "endpoint": ep,
                "period": label,
                "params": params,
                "http_status": None,
                "top_level_keys": [],
                "item_count": 0,
                "schema": None,
                "error": None,
            }

            try:
                response = execute_get_request(session, url, params, auth)
                record["http_status"] = response.status_code

                try:
                    payload = response.json()
                except Exception:
                    payload = {"raw_text_length": len(response.text)}

                # Save raw response to data/raw/ (gitignored)
                out_path = os.path.join(raw_data_dir, sanitize_filename(ep, label))
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

                if response.status_code >= 400:
                    err_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
                    err_code = err_obj.get("code", f"HTTP_{response.status_code}")
                    err_desc = err_obj.get("description", response.reason)
                    record["error"] = {"code": err_code, "description": err_desc}
                    empty_or_errored.append(
                        {
                            "endpoint": ep,
                            "period": label,
                            "status": response.status_code,
                            "reason": f"Error: {err_code} - {err_desc}",
                        }
                    )
                elif isinstance(payload, dict):
                    record["top_level_keys"] = sorted(list(payload.keys()))
                    items = payload.get("items")
                    if isinstance(items, list):
                        record["item_count"] = len(items)
                        if len(items) > 0 and isinstance(items[0], dict):
                            record["schema"] = extract_schema(items[0])
                        elif len(items) == 0:
                            empty_or_errored.append(
                                {
                                    "endpoint": ep,
                                    "period": label,
                                    "status": response.status_code,
                                    "reason": "Empty items collection",
                                }
                            )
                    else:
                        record["item_count"] = 1
                        record["schema"] = extract_schema(payload)
                else:
                    record["item_count"] = len(payload) if isinstance(payload, list) else 1

            except Exception as exc:
                record["error"] = {
                    "code": "REQUEST_EXCEPTION",
                    "description": type(exc).__name__,
                }
                empty_or_errored.append(
                    {
                        "endpoint": ep,
                        "period": label,
                        "status": "FAILED",
                        "reason": f"Exception: {type(exc).__name__}",
                    }
                )

            results.append(record)

    # Step 4: Write docs/api_probe_report.md
    write_probe_report(results, empty_or_errored, os.path.join(docs_dir, "api_probe_report.md"))
    print("[*] Probe completed. Report written to docs/api_probe_report.md")


def write_probe_report(
    results: list[dict[str, Any]],
    empty_or_errored: list[dict[str, Any]],
    report_path: str,
) -> None:
    """Generate Markdown report containing status, shape, schema, and empty/error list."""
    ist_tz = zoneinfo.ZoneInfo("Asia/Kolkata")
    now_str = datetime.datetime.now(ist_tz).strftime("%Y-%m-%d %H:%M:%S IST")

    lines = [
        "# Razorpay API Probe Report",
        "",
        f"**Timestamp:** {now_str}",
        "**Environment:** TEST Mode",
        "**Mode:** Read-only probe (HTTP GET only)",
        "",
        "## Endpoints Summary",
        "",
        "| Endpoint | Period | Status | Top-Level Keys | Item Count |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        status_str = str(r["http_status"]) if r["http_status"] is not None else "ERROR"
        keys_str = ", ".join(r["top_level_keys"]) if r["top_level_keys"] else "-"
        lines.append(
            f"| `{r['endpoint']}` | {r['period']} | {status_str} | `{keys_str}` | {r['item_count']} |"
        )

    lines.extend([
        "",
        "## First-Item Field/Type Schemas",
        "",
        "Extracted JSON type signatures with strictly NO values:",
        "",
    ])

    for r in results:
        if r["schema"]:
            lines.append(f"### `{r['endpoint']}` ({r['period']})")
            lines.append("```json")
            lines.append(json.dumps(r["schema"], indent=2))
            lines.append("```")
            lines.append("")

    lines.extend([
        "## Endpoints Returning Empty or Errored",
        "",
    ])

    if empty_or_errored:
        lines.append("| Endpoint | Period | Status | Detail |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for e in empty_or_errored:
            lines.append(f"| `{e['endpoint']}` | {e['period']} | {e['status']} | {e['reason']} |")
    else:
        lines.append("No endpoints returned empty collections or errors.")

    lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_probe()

