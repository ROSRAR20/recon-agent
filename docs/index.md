# Documentation Index & Sitemap

Welcome to the Recon-Agent technical documentation. Below is a guide to all research protocols, specifications, and reports.

---

## Architecture & Systems
- [docs/b1_rules.md](file:///c:/Users/rosra/recon-agent/docs/b1_rules.md): Specification of the deterministic baseline matcher (B1).
- [docs/verifier.md](file:///c:/Users/rosra/recon-agent/docs/verifier.md): Rule-based verification pipeline (V1–V8) enforcing schema, grounding, and arithmetic.
- [docs/schemas.json](file:///c:/Users/rosra/recon-agent/docs/schemas.json): Canonical JSON schemas for transaction entities and LLM inputs/outputs.
- [docs/schema_decision.md](file:///c:/Users/rosra/recon-agent/docs/schema_decision.md): Architecture decision record (ADR) on schema design.

---

## Experimental Protocol & Registration
- [docs/preregistration.md](file:///c:/Users/rosra/recon-agent/docs/preregistration.md): Pre-registered hypotheses (H1–H6), acceptance criteria, and methodology.
- [docs/registration_log.md](file:///c:/Users/rosra/recon-agent/docs/registration_log.md): Immutable commit hashes and tags for preregistration snapshots.
- [docs/eval_protocol.md](file:///c:/Users/rosra/recon-agent/docs/eval_protocol.md): Evaluation protocol across baseline engines and verified LLM pipelines.

---

## Synthetic Data & Benchmarks
- [docs/synthetic_data_spec.md](file:///c:/Users/rosra/recon-agent/docs/synthetic_data_spec.md): Parameterization of transaction generators, error classes, and edge cases.
- [docs/api_probe_report.md](file:///c:/Users/rosra/recon-agent/docs/api_probe_report.md): Initial API probe and rate limit characterization.

---

## Evaluation Reports
- [docs/results_main.md](file:///c:/Users/rosra/recon-agent/docs/results_main.md): Primary evaluation report for Run 0 (Main Population) covering hypotheses H1–H6.
- [docs/pilot_run1_report.md](file:///c:/Users/rosra/recon-agent/docs/pilot_run1_report.md): Pilot Run 1 evaluation report across default, low, and high temperature arms.
- [docs/pilot_run2_report.md](file:///c:/Users/rosra/recon-agent/docs/pilot_run2_report.md): Pilot Run 2 evaluation report validating prompt and verifier fixes.
- [docs/change_log.md](file:///c:/Users/rosra/recon-agent/docs/change_log.md): Change log tracking iterations and before/after metrics.
