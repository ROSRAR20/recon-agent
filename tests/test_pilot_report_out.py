"""Test pilot_report.py UTF-8 output and --out PATH parameter."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import pytest

from scripts.pilot_report import _TeeStream, run_pilot_report


def test_tee_stream_unicode():
    """Verify _TeeStream handles non-ASCII characters and writes UTF-8 file properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "report.txt")
        buffer = io.StringIO()
        tee = _TeeStream(buffer, out_path=out_file)

        non_ascii_text = "Evaluation Report: ₹100.50 fee, ±0.05% error, naïve check: PASS\n"
        tee.write(non_ascii_text)
        assert tee.get_text() == non_ascii_text
        assert buffer.getvalue() == non_ascii_text

        # Write directly to file as done in run_pilot_report
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(tee.get_text())

        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert content == non_ascii_text
        assert "₹" in content


def test_run_pilot_report_with_out(tmp_path):
    """Verify run_pilot_report in mock mode writes UTF-8 report to --out path."""
    out_file = str(tmp_path / "mock_report.md")
    fin_dir = str(tmp_path / "finalized")
    # Run in mock mode on small sample (dev seed 1, mock=True)
    run_pilot_report(
        seed=1,
        arms=["default"],
        n_residual=5,
        n_control=2,
        mock=True,
        finalized_dir=fin_dir,
        out_path=out_file,
    )

    assert os.path.isfile(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "MOCK DATA - NOT RESULTS" in content
    assert "### Arm: `default`" in content
