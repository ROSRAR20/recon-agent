"""Environment and configuration integrity checker.

Verifies:
- Python version and runtime environment
- Timezone data availability for 'America/Los_Angeles' (Pacific time)
- Git repository cleanliness and commit state
- Presence and validity of required configuration files (pricing, models)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import platform
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass(frozen=True)
class EnvironmentInfo:
    """Environment telemetry record."""

    python_version: str
    os_platform: str
    os_release: str
    tzdata_ok: bool
    git_commit: str
    git_dirty: bool
    config_models_ok: bool
    config_prices_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        dirty_flag = "DIRTY" if self.git_dirty else "CLEAN"
        tz_flag = "tzdata OK" if self.tzdata_ok else "tzdata MISSING"
        return (
            f"Python {self.python_version} on {self.os_platform} | {tz_flag} | "
            f"Commit {self.git_commit[:8]} ({dirty_flag})"
        )


def check_tzdata() -> bool:
    """Check that America/Los_Angeles timezone data is available."""
    try:
        ZoneInfo("America/Los_Angeles")
        return True
    except (ZoneInfoNotFoundError, Exception):
        raise RuntimeError(
            "Missing required timezone data for 'America/Los_Angeles'. "
            "Please run: pip install tzdata"
        )


def get_git_state() -> tuple[str, bool]:
    """Retrieve current git commit hash and tracked dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        is_dirty = bool(status)
    except Exception:
        is_dirty = False

    return commit, is_dirty


def check_configs() -> tuple[bool, bool]:
    """Verify presence and validity of required configuration files."""
    models_path = os.path.join(REPO_ROOT, "config", "llm_models.json")
    prices_path = os.path.join(REPO_ROOT, "config", "llm_prices.json")

    models_ok = False
    if os.path.isfile(models_path):
        try:
            with open(models_path, "r", encoding="utf-8") as f:
                json.load(f)
            models_ok = True
        except Exception:
            models_ok = False

    prices_ok = False
    if os.path.isfile(prices_path):
        try:
            with open(prices_path, "r", encoding="utf-8") as f:
                json.load(f)
            prices_ok = True
        except Exception:
            prices_ok = False

    return models_ok, prices_ok


def check_env() -> EnvironmentInfo:
    """Run full environment integrity check."""
    tz_ok = check_tzdata()
    commit, is_dirty = get_git_state()
    models_ok, prices_ok = check_configs()

    return EnvironmentInfo(
        python_version=platform.python_version(),
        os_platform=platform.system(),
        os_release=platform.release(),
        tzdata_ok=tz_ok,
        git_commit=commit,
        git_dirty=is_dirty,
        config_models_ok=models_ok,
        config_prices_ok=prices_ok,
    )


def main() -> None:
    env_info = check_env()
    print("=" * 60)
    print("ENVIRONMENT INTEGRITY CHECK")
    print("=" * 60)
    print(f"OS Platform     : {env_info.os_platform} ({env_info.os_release})")
    print(f"Python Version  : {env_info.python_version}")
    print(f"Timezone Data   : {'PASS (America/Los_Angeles available)' if env_info.tzdata_ok else 'FAIL'}")
    print(f"Git Commit      : {env_info.git_commit}")
    print(f"Git Tracked State: {'DIRTY' if env_info.git_dirty else 'CLEAN'}")
    print(f"Models Config   : {'PASS' if env_info.config_models_ok else 'FAIL'}")
    print(f"Prices Config   : {'PASS' if env_info.config_prices_ok else 'FAIL'}")
    print("=" * 60)
    print(f"Summary: {env_info.summary_line()}")


if __name__ == "__main__":
    main()
