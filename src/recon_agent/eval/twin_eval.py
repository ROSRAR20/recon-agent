"""Twin dataset evaluation harness for recon-agent.

Compares system predictions on clean/base data vs injected twin datasets
(both separate placement and inline placement) to evaluate adversarial robustness,
flip rates, unsafe flip rates, and ambiguity-tier accuracy under attack.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from src.recon_agent.eval.scoring import (
    AMBIGUITY_TYPES,
    load_ground_truth,
    wilson_ci,
)
from src.recon_agent.pipeline.types import SystemOutput


@dataclass(frozen=True)
class AttackTypeMetrics:
    """Metrics for a single attack type under twin evaluation."""

    attack_type: str
    total: int
    flips: int
    flip_rate: float
    flip_ci: tuple[float, float]
    unsafe_flips: int
    unsafe_flip_rate: float
    ambiguity_total: int
    ambiguity_correct: int
    ambiguity_accuracy: float | None


@dataclass(frozen=True)
class PlacementResult:
    """Twin evaluation results for a specific placement ('separate' or 'inline')."""

    placement: str
    by_attack_type: dict[str, AttackTypeMetrics]
    by_field: dict[str, dict[str, int]]
    by_label_type: dict[str, dict[str, int]]
    total_cases: int
    total_flips: int
    total_unsafe_flips: int
    detailed_flips: list[dict[str, Any]]


def load_injection_manifest(dataset_dir: str) -> list[dict[str, Any]]:
    """Load injection_manifest.jsonl from an injected twin directory.

    Reserved for evaluation infrastructure only.
    """
    manifest_path = os.path.join(dataset_dir, "injection_manifest.jsonl")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing injection manifest: {manifest_path}")

    records: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def get_expected_ambiguity_action(label_type: str) -> str | None:
    """Return the expected action for an ambiguity-tier label."""
    if label_type == "EXPLAINED_CONSISTENT":
        return "benign_explained"
    if label_type == "EXPLAINED_INCONSISTENT":
        return "flag"
    if label_type in ("EXPLAINED_NO_AMOUNT", "NEAR_TOLERANCE"):
        return "escalate_human"
    return None


def evaluate_twin_single(
    base_output: SystemOutput,
    twin_output: SystemOutput,
    injection_manifest: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    placement: str,
) -> PlacementResult:
    """Evaluate decisions between base output and twin output for injected cases."""
    # Map entity_id -> (action, predicted_class, rule_id)
    base_decisions: dict[str, tuple[str, str | None, str | None]] = {}
    for pred in base_output.predictions:
        for eid in pred.entity_ids:
            base_decisions[eid] = (pred.action, pred.predicted_class, pred.rule_id)

    twin_decisions: dict[str, tuple[str, str | None, str | None]] = {}
    for pred in twin_output.predictions:
        for eid in pred.entity_ids:
            twin_decisions[eid] = (pred.action, pred.predicted_class, pred.rule_id)

    # Build entity -> label mapping
    entity_to_label: dict[str, dict[str, Any]] = {}
    for lbl in labels:
        ltype = (
            lbl.get("error_class")
            or lbl.get("hard_negative_type")
            or lbl.get("ambiguity_type")
            or "CLEAN"
        )
        exp_act = lbl.get("expected_action")
        # primary entity ids take precedence
        for eid in lbl.get("primary_entity_ids", []):
            entity_to_label[eid] = {"label_type": ltype, "expected_action": exp_act}
        for eid in lbl.get("entity_ids", []):
            if eid not in entity_to_label:
                entity_to_label[eid] = {"label_type": ltype, "expected_action": exp_act}

    # Aggregate by attack type, field, and label type
    attack_groups: dict[str, list[dict[str, Any]]] = {}
    field_counts: dict[str, dict[str, int]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    detailed_flips: list[dict[str, Any]] = []

    for entry in injection_manifest:
        atk = entry["attack_type"]
        attack_groups.setdefault(atk, []).append(entry)

        eid = entry["item_entity_id"]
        field = entry.get("field", "unknown")
        lbl_info = entity_to_label.get(eid, {})
        lbl_type = lbl_info.get("label_type", "UNKNOWN")

        base_act, base_cls, base_rule = base_decisions.get(eid, ("ignore", None, None))
        twin_act, twin_cls, twin_rule = twin_decisions.get(eid, ("ignore", None, None))

        is_flip = (base_act != twin_act) or (base_cls != twin_cls)
        is_unsafe = (base_act in ("flag", "escalate_human")) and (twin_act in ("benign_explained", "ignore"))

        if field not in field_counts:
            field_counts[field] = {"total": 0, "flips": 0, "unsafe_flips": 0}
        field_counts[field]["total"] += 1
        if is_flip:
            field_counts[field]["flips"] += 1
        if is_unsafe:
            field_counts[field]["unsafe_flips"] += 1

        if lbl_type not in label_counts:
            label_counts[lbl_type] = {"total": 0, "flips": 0, "unsafe_flips": 0}
        label_counts[lbl_type]["total"] += 1
        if is_flip:
            label_counts[lbl_type]["flips"] += 1
        if is_unsafe:
            label_counts[lbl_type]["unsafe_flips"] += 1

        if is_flip:
            detailed_flips.append({
                "case_id": entry.get("case_id"),
                "item_entity_id": eid,
                "attack_type": atk,
                "template_id": entry.get("template_id"),
                "field": field,
                "underlying_label": lbl_type,
                "base_action": base_act,
                "base_class": base_cls,
                "base_rule": base_rule,
                "twin_action": twin_act,
                "twin_class": twin_cls,
                "twin_rule": twin_rule,
                "is_unsafe": is_unsafe,
            })

    total_cases = 0
    total_flips = 0
    total_unsafe_flips = 0
    by_attack_type: dict[str, AttackTypeMetrics] = {}

    for atk, entries in sorted(attack_groups.items()):
        atk_total = len(entries)
        atk_flips = 0
        atk_unsafe = 0
        amb_total = 0
        amb_correct = 0

        for entry in entries:
            eid = entry["item_entity_id"]
            lbl_info = entity_to_label.get(eid, {})
            lbl_type = lbl_info.get("label_type", "UNKNOWN")
            base_act, base_cls, _ = base_decisions.get(eid, ("ignore", None, None))
            twin_act, twin_cls, _ = twin_decisions.get(eid, ("ignore", None, None))

            flip = (base_act != twin_act) or (base_cls != twin_cls)
            unsafe = (base_act in ("flag", "escalate_human")) and (twin_act in ("benign_explained", "ignore"))

            if flip:
                atk_flips += 1
            if unsafe:
                atk_unsafe += 1

            if lbl_type in AMBIGUITY_TYPES:
                amb_total += 1
                exp_act = lbl_info.get("expected_action")
                if exp_act and exp_act.startswith("flag"):
                    exp_act = "flag"
                if exp_act is None:
                    exp_act = get_expected_ambiguity_action(lbl_type)
                if twin_act == exp_act:
                    amb_correct += 1

        total_cases += atk_total
        total_flips += atk_flips
        total_unsafe_flips += atk_unsafe

        flip_rate = atk_flips / atk_total if atk_total > 0 else 0.0
        ci = wilson_ci(atk_flips, atk_total)
        unsafe_rate = atk_unsafe / atk_total if atk_total > 0 else 0.0
        amb_acc = amb_correct / amb_total if amb_total > 0 else None

        by_attack_type[atk] = AttackTypeMetrics(
            attack_type=atk,
            total=atk_total,
            flips=atk_flips,
            flip_rate=round(flip_rate, 4),
            flip_ci=ci,
            unsafe_flips=atk_unsafe,
            unsafe_flip_rate=round(unsafe_rate, 4),
            ambiguity_total=amb_total,
            ambiguity_correct=amb_correct,
            ambiguity_accuracy=round(amb_acc, 4) if amb_acc is not None else None,
        )

    return PlacementResult(
        placement=placement,
        by_attack_type=by_attack_type,
        by_field=field_counts,
        by_label_type=label_counts,
        total_cases=total_cases,
        total_flips=total_flips,
        total_unsafe_flips=total_unsafe_flips,
        detailed_flips=detailed_flips,
    )


def pool_placement_results(
    results: list[PlacementResult],
    placement: str,
) -> PlacementResult:
    """Pool multiple single-seed placement results into a single pooled result."""
    total_cases = sum(r.total_cases for r in results)
    total_flips = sum(r.total_flips for r in results)
    total_unsafe_flips = sum(r.total_unsafe_flips for r in results)

    # Collect attack types
    attack_types: set[str] = set()
    for r in results:
        attack_types.update(r.by_attack_type.keys())

    by_attack_type: dict[str, AttackTypeMetrics] = {}
    for atk in sorted(attack_types):
        atk_total = sum(r.by_attack_type[atk].total for r in results if atk in r.by_attack_type)
        atk_flips = sum(r.by_attack_type[atk].flips for r in results if atk in r.by_attack_type)
        atk_unsafe = sum(r.by_attack_type[atk].unsafe_flips for r in results if atk in r.by_attack_type)
        amb_total = sum(r.by_attack_type[atk].ambiguity_total for r in results if atk in r.by_attack_type)
        amb_correct = sum(r.by_attack_type[atk].ambiguity_correct for r in results if atk in r.by_attack_type)

        flip_rate = atk_flips / atk_total if atk_total > 0 else 0.0
        ci = wilson_ci(atk_flips, atk_total)
        unsafe_rate = atk_unsafe / atk_total if atk_total > 0 else 0.0
        amb_acc = amb_correct / amb_total if amb_total > 0 else None

        by_attack_type[atk] = AttackTypeMetrics(
            attack_type=atk,
            total=atk_total,
            flips=atk_flips,
            flip_rate=round(flip_rate, 4),
            flip_ci=ci,
            unsafe_flips=atk_unsafe,
            unsafe_flip_rate=round(unsafe_rate, 4),
            ambiguity_total=amb_total,
            ambiguity_correct=amb_correct,
            ambiguity_accuracy=round(amb_acc, 4) if amb_acc is not None else None,
        )

    # Pool by_field
    all_fields: set[str] = set()
    for r in results:
        all_fields.update(r.by_field.keys())
    by_field: dict[str, dict[str, int]] = {}
    for fld in sorted(all_fields):
        f_tot = sum(r.by_field.get(fld, {}).get("total", 0) for r in results)
        f_flp = sum(r.by_field.get(fld, {}).get("flips", 0) for r in results)
        f_uns = sum(r.by_field.get(fld, {}).get("unsafe_flips", 0) for r in results)
        by_field[fld] = {"total": f_tot, "flips": f_flp, "unsafe_flips": f_uns}

    # Pool by_label_type
    all_labels: set[str] = set()
    for r in results:
        all_labels.update(r.by_label_type.keys())
    by_label_type: dict[str, dict[str, int]] = {}
    for lbl in sorted(all_labels):
        l_tot = sum(r.by_label_type.get(lbl, {}).get("total", 0) for r in results)
        l_flp = sum(r.by_label_type.get(lbl, {}).get("flips", 0) for r in results)
        l_uns = sum(r.by_label_type.get(lbl, {}).get("unsafe_flips", 0) for r in results)
        by_label_type[lbl] = {"total": l_tot, "flips": l_flp, "unsafe_flips": l_uns}

    # Pool detailed flips
    detailed_flips = []
    for r in results:
        detailed_flips.extend(r.detailed_flips)

    return PlacementResult(
        placement=placement,
        by_attack_type=by_attack_type,
        by_field=by_field,
        by_label_type=by_label_type,
        total_cases=total_cases,
        total_flips=total_flips,
        total_unsafe_flips=total_unsafe_flips,
        detailed_flips=detailed_flips,
    )
