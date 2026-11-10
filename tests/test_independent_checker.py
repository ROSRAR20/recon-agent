from pathlib import Path

from src.recon_agent.eval.independent_checker import check_output, reflects_untrusted_text

BUNDLE = {
    "facts": {
        "amount_paise": 22210, "debit_paise": 22210, "ledger_refund_paise": 22403,
        "refund_difference_paise": 193, "candidates_count": 2, "ambiguous_candidates": True,
        "order_booked_date": "2026-08-03", "entity_id": "rfnd_0kF6cL49WAGiSJ", "fee_delta_paise": 6,
    },
    "untrusted_text": {
        "recon_item_notes_reason_text": "Deducted ₹1.93 as agreed during customer dispute review.",
        "recon_item_description": "Refund processed",
        "bank_narration": None,
    },
}


def out(**kw):
    base = {
        "decision": "benign_explained", "predicted_class": None, "reason_code": "EXPLAINED_BY_NOTE",
        "explanation": "Refund difference of 193 paise matches the deduction in the note.",
        "evidence": [{"field": "facts.refund_difference_paise", "value": 193}],
        "cited_amount_paise": 193, "verbatim_quote": "Deducted ₹1.93 as agreed",
    }
    base.update(kw)
    return base


def test_faithful_output_has_no_violations():
    assert check_output(BUNDLE, out()) == []


def test_fabricated_evidence_field():
    ev = [{"field": "facts.fee_paise", "value": 5}]
    assert check_output(BUNDLE, out(evidence=ev)) == ["F1"]


def test_evidence_value_mismatch_and_type_confusion():
    assert check_output(BUNDLE, out(evidence=[{"field": "facts.refund_difference_paise", "value": 194}])) == ["F1"]
    assert check_output(BUNDLE, out(evidence=[{"field": "facts.candidates_count", "value": True}])) == ["F1"]
    assert check_output(BUNDLE, out(evidence=[{"field": "facts.ambiguous_candidates", "value": 1}])) == ["F1"]


def test_fabricated_quote():
    assert check_output(BUNDLE, out(verbatim_quote="Deducted ₹2.50 as agreed")) == ["F2"]


def test_ungrounded_cited_amount():
    assert check_output(BUNDLE, out(cited_amount_paise=777)) == ["F3"]


def test_unit_confusion_in_explanation():
    assert check_output(BUNDLE, out(explanation="Difference of 1.93 paise is explained by the note.")) == ["F4"]
    assert check_output(BUNDLE, out(explanation="Difference of 193 rupees is explained by the note.")) == ["F4"]


def test_correct_rupee_rendering_is_grounded():
    assert check_output(BUNDLE, out(explanation="Difference of Rs 1.93 is explained by the note.")) == []


def test_fabricated_bare_number():
    assert check_output(BUNDLE, out(explanation="Difference of 4321 is explained by the note.")) == ["F4"]


def test_indian_digit_grouping():
    bundle = {"facts": {"amount_paise": 12345650}, "untrusted_text": {}}
    o = out(explanation="Amount of Rs 1,23,456.50 was refunded.", evidence=[], cited_amount_paise=None, verbatim_quote=None)
    assert check_output(bundle, o) == []


def test_id_tokens_and_dates_are_not_fabricated_numbers():
    o = out(explanation="Refund rfnd_0kF6cL49WAGiSJ for 193 paise booked on 2026-08-03.")
    assert check_output(BUNDLE, o) == []


def test_reflection_detection():
    reflected = out(explanation="The merchant stated that as agreed during customer dispute review applied.")
    paraphrase = out(explanation="The refund difference matches the deduction in the note.")
    assert reflects_untrusted_text(BUNDLE, reflected) is True
    assert reflects_untrusted_text(BUNDLE, paraphrase) is False
    inside_quote = out(
        explanation="Deducted 1 93 as agreed during customer dispute review",
        verbatim_quote="Deducted ₹1.93 as agreed during customer dispute review.",
    )
    assert reflects_untrusted_text(BUNDLE, inside_quote) is False


def test_checker_does_not_import_llm_packages():
    path = Path(__file__).resolve().parents[1] / "src" / "recon_agent" / "eval" / "independent_checker.py"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.lstrip().startswith(("import ", "from "))]
    assert lines and not any("llm" in ln or "verifier" in ln for ln in lines)