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
from src.recon_agent.llm.cache import CacheCorruptionError, CacheKeyExistsError, LLMCache
from src.recon_agent.llm.client import LLMClient, MockClient
from src.recon_agent.llm.finalize import (
    FinalDecision,
    compute_verifier_version,
    finalize,
    get_finalized_filepath,
    load_finalized_results,
    write_finalized_results,
)
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
    "CacheCorruptionError",
    "CacheKeyExistsError",
    "CaseBundle",
    "FinalDecision",
    "LLMCache",
    "LLMClient",
    "LLMOutput",
    "MockClient",
    "Verifier",
    "VerifierOutcome",
    "VerifierResult",
    "build_case_bundles",
    "compute_verifier_version",
    "finalize",
    "get_finalized_filepath",
    "load_finalized_results",
    "validate_llm_output",
    "write_finalized_results",
]

