# Pre-Freeze Change Log

This document records all pre-freeze modifications applied to code and configuration assets after tag `registration-final` (`112e5b099772982809f818d4853e9406fd53a833`).
Integrity checker `scripts/check_frozen.py` validates that `verifier.py` and `system.txt` match the latest recorded hash and maintain an unbroken cryptographic hash chain from `registration-final`.

---

## Entry 1: Verifier Unit Grammar Extension (Postfix Rupee Support)
- **Date**: 2026-11-22
- **File**: `src/recon_agent/llm/verifier.py`
- **Old Git Blob Hash**: `06929945500e4818b38ce07e58aa8a0c2915556e`
- **New Git Blob Hash**: `74412e3d7e991366eb6afe04ff008546daf0ff71`
- **Reason**: The verifier unit grammar (`AMOUNT_UNIT_PATTERN` and `_strict_convert_amount_token`) only recognized prefix currency codes (`INR`, `Rs`, `₹`) or postfix paise, while `independent_checker.py` also recognized postfix `rupee`/`rupees`. In default arm bundle `bundle_rfnd_K1jO63oR433kS8`, an ungrounded claim of "1 rupee" evaded unit extraction, fell through to bare-number extraction, matched neutral constant `1`, and was accepted by the verifier (resulting in 1 unblocked hallucination). Extended verifier unit grammar to cover all forms accepted by the independent checker (including postfix rupee/rupees) and strictly reject conflicting or ungrounded amounts.
- **BEFORE Numbers** (from `docs/pilot_run1_report.md`):
  - Hallucination Rate (Schema-Valid): default 23/120 (19.2%), low 22/119 (18.5%), high 23/120 (19.2%)
  - Verifier Block Rate (Hallucinated): default 22/23 (95.7%, 1 unblocked: `bundle_rfnd_K1jO63oR433kS8`), low 22/22 (100.0%), high 23/23 (100.0%)
  - Unsafe Resolution Rate: default 0/120 (0.0%), low 0/120 (0.0%), high 0/120 (0.0%)
  - B3 Decision Accuracy: default 93/120 (77.5%), low 87/120 (72.5%), high 88/120 (73.3%)
  - B3 Non-ACCEPT Verifier Outcomes: default 33 (20 FABRICATED_NUMBER_IN_EXPLANATION, 10 NO_AMOUNT_IN_QUOTE, 1 REFLECTED_UNTRUSTED_TEXT, 1 UNGROUNDED_AMOUNT, 1 FABRICATED_QUOTE)
- **AFTER Numbers**: pending re-pilot

---

## Entry 2: System Prompt Specification & Mismatch Few-Shot Addition
- **Date**: 2026-11-22
- **File**: `src/recon_agent/llm/prompts/system.txt`
- **Old Git Blob Hash**: `e278987b6f8a21fb347981a6d1bb7f4d4af4dca7`
- **New Git Blob Hash**: `0d62ce08635b322dd6b73df4de7e0bccbe1e3995`
- **Reason**: Aligned system prompt with B1r refund-note rule: (1) one unit-bearing amount equal to |refund_difference_paise| -> benign_explained with that sentence as verbatim_quote and cited_amount_paise; (2) one unit-bearing amount that DIFFERS -> flag PARTIAL_REFUND_MISMATCH with that sentence as verbatim_quote and cited_amount_paise = the note's amount; (3) no amount or words only -> escalate_human; (4) never mention an amount from the note in the explanation unless inside verbatim_quote. Added hand-authored few-shot Example 6 demonstrating the mismatch case (`PARTIAL_REFUND_MISMATCH`). Recomputed prompt hash to `a92e1b8026684afc584502fb2d7853a301348f6ef400708da18bbaff5eafd52f`.
- **BEFORE Numbers** (from `docs/pilot_run1_report.md`):
  - B3 Decision Accuracy: default 93/120 (77.5%), low 87/120 (72.5%), high 88/120 (73.3%)
  - B2 Decision Accuracy: default 91/120 (75.8%), low 90/120 (75.0%), high 91/120 (75.8%)
  - B3 Over-Escalation Rate: default 3/45 (6.7%), low 8/45 (17.8%), high 7/45 (15.6%)
  - False-Reject Rate (Faithful Correct Non-Escalations): default 3/45 (6.7%), low 8/45 (17.8%), high 7/45 (15.6%)
  - Non-ACCEPT Outcomes: default 33 (10 NO_AMOUNT_IN_QUOTE), low 39 (9 NO_AMOUNT_IN_QUOTE), high 26 (0 NO_AMOUNT_IN_QUOTE)
- **AFTER Numbers**: pending re-pilot
