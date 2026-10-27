"""Configuration dataclasses for deterministic synthetic generation."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"



@dataclass(frozen=True)
class SynthConfig:
    """Configuration for synthetic data generation."""

    seed: int
    profile: str = "small"  # "small" (10 merchants) or "full" (60 merchants)
    split: str = "dev"  # "dev" or "test"
    num_merchants: int = 10
    start_date: str = "2026-08-03"  # Monday
    working_days: int = 22  # Mon-Fri business days
    avg_payments_per_day: float = 20.0  # Poisson mean
    method_mix: dict[str, int] = field(
        default_factory=lambda: {"card": 45, "upi": 40, "netbanking": 15}
    )
    method_fee_bps: dict[str, int] = field(
        default_factory=lambda: {"card": 200, "upi": 0, "netbanking": 150}
    )
    gst_bps: int = 1800  # 18.00%
    fee_includes_tax: bool = True
    failed_rate_bps: int = 600  # ~6%
    refund_rate_bps: int = 400  # ~4% of captured
    full_refund_pct: int = 70  # 70% full refunds
    partial_refund_pct: int = 30  # 30% partial refunds
    payment_settlement_lag_days: int = 2  # C+2 working days
    refund_settlement_lag_days: int = 1  # R+1 working day
    adjustment_prob_bps: int = 30  # ~0.3% of batches
    unrelated_bank_rows_per_merchant: int = 30
    lognormal_mu: float = 9.5
    lognormal_sigma: float = 1.0
    min_amount_paise: int = 100
    generator_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return asdict(self)

    def config_hash(self) -> str:
        """Compute deterministic SHA-256 hash of configuration."""
        canonical_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def create_config(
    profile: str,
    split: str,
    seed: int,
    fee_includes_tax: bool = True,
    **kwargs: Any,
) -> SynthConfig:
    """Factory to create a validated SynthConfig instance.

    Args:
        profile: "small" or "full".
        split: "dev" or "test".
        seed: Pseudorandom integer seed.
        fee_includes_tax: Whether fee includes GST or is additive.
        kwargs: Overrides for configuration fields.

    Returns:
        SynthConfig instance.
    """
    if profile not in ("small", "full"):
        raise ValueError(f"Unknown profile: {profile}. Must be 'small' or 'full'.")
    if split not in ("dev", "test"):
        raise ValueError(f"Unknown split: {split}. Must be 'dev' or 'test'.")

    num_merchants = 10 if profile == "small" else 60

    return SynthConfig(
        seed=seed,
        profile=profile,
        split=split,
        num_merchants=num_merchants,
        fee_includes_tax=fee_includes_tax,
        **kwargs,
    )
