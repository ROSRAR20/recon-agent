"""Synthetic data generation subpackage for recon-agent."""

from src.recon_agent.synth.config import SynthConfig, create_config
from src.recon_agent.synth.generator import SyntheticDataGenerator
from src.recon_agent.synth.math_utils import mul_div_round

__all__ = [
    "SynthConfig",
    "create_config",
    "SyntheticDataGenerator",
    "mul_div_round",
]
