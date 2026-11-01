"""LLM scaffold, verification layer, caching, and audit logging for recon-agent.

This package provides:
- CaseBundle: structured facts and untrusted text for ambiguous cases.
- Schema: strict JSON output schema and validation.
- Verifier: rule-based consistency and grounding verifier (V1-V7).
- Cache: deterministic on-disk response caching.
- Audit: append-only, privacy-preserving audit logging.
- Client: protocol and mock client implementations.
"""

from src.recon_agent.llm.audit import AuditLogger, AuditRecord
from src.recon_agent.llm.bundle import CaseBundle, build_case_bundles
from src.recon_agent.llm.cache import LLMCache
from src.recon_agent.llm.client import LLMClient, MockClient
from src.recon_agent.llm.schema import (
    ALLOWED_CLASSES,
    ALLOWED_DECISIONS,
    REASON_CODES,
    LLMOutput,
    validate_llm_output,
)
from src.recon_agent.llm.verifier import Verifier, VerifierOutcome, VerifierResult

__all__ = [
    "ALLOWED_CLASSES",
    "ALLOWED_DECISIONS",
    "REASON_CODES",
    "AuditLogger",
    "AuditRecord",
    "CaseBundle",
    "LLMCache",
    "LLMClient",
    "LLMOutput",
    "MockClient",
    "Verifier",
    "VerifierOutcome",
    "VerifierResult",
    "build_case_bundles",
    "validate_llm_output",
]

