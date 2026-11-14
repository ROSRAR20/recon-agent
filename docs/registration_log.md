# Registration log (written before the pilot; history recorded as it happened)

- 9aa5571 "Independent faithfulness checker" (tag prereg points here): src/recon_agent/eval/independent_checker.py was committed as an EMPTY file (0 bytes).
- 73b8ce1 "Pre-registration" (tag checker-v1 points here; the agent moved this tag there during Day 7b-1): contains docs/preregistration.md and the complete 166-line checker.
- No real LLM evaluation output existed before 73b8ce1. Only smoke tests on 5 synthetic bundles had been run, and the cache was purged before the pilot.
- The final registration state is the commit tagged registration-final: this log plus Addendum A on top of 73b8ce1. The registered checker is the file at 73b8ce1 (git blob hash 5def01030115807749e77f0375e93a4e59ef9f35). The registered verifier is unchanged since commit b470193.
- No tag is moved after this point.

## Addendum A (before the pilot)
- The pilot bundles (the first 100 residual and the first 20 control bundles of dev seed 1, in the registered population order) are EXCLUDED from all tests of H1-H6.
- Any change to the prompt, bundle format, schema or verifier after the pilot is allowed on dev only, is logged with before/after numbers, and invalidates the pilot data: the chosen arm is re-run before the main evaluation. Pilot outputs are never reused in the main analysis.
- The false-reject rate (correct decisions the verifier escalates for lack of verifiable evidence) is reported next to H1-H6 as the cost of verification. It has no pass/fail threshold.
- Arms whose thinking level the API rejects are dropped without retry.