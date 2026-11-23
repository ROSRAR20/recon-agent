"""Test that every benchmark number cited in README.md carries a valid source tag

and that the referenced numbers match the source document.
"""
from __future__ import annotations

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_readme_source_tags_exist_and_point_to_valid_files():
    readme_path = os.path.join(REPO_ROOT, "README.md")
    assert os.path.exists(readme_path), "README.md must exist"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all source tags: <!-- source: <path> -->
    source_tags = re.findall(r"<!--\s*source:\s*([^\s>]+)\s*-->", content)
    assert len(source_tags) >= 2, f"Expected multiple source tags in README, found {len(source_tags)}"

    for src in source_tags:
        target_path = os.path.join(REPO_ROOT, src.replace("/", os.sep))
        assert os.path.exists(target_path), f"Tagged source file '{src}' does not exist on disk"


def test_readme_results_numbers_match_results_main():
    readme_path = os.path.join(REPO_ROOT, "README.md")
    results_path = os.path.join(REPO_ROOT, "docs", "results_main.md")

    assert os.path.exists(readme_path)
    assert os.path.exists(results_path)

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    with open(results_path, "r", encoding="utf-8") as f:
        results_text = f.read()

    # Extract all sections or blocks tagged with docs/results_main.md
    blocks = re.split(r"<!--\s*source:\s*docs/results_main\.md\s*-->", readme_text)
    # blocks[1:] are the blocks governed by docs/results_main.md
    assert len(blocks) >= 2, "README must contain sections tagged with docs/results_main.md"

    # In each governed block, find key ratios and percentages (e.g., 518/527, 98.3%, 0/555, 0.0%)
    for block in blocks[1:]:
        # Stop at the next major section marker if any (level 2 heading ## N.)
        section_content = re.split(r"\n##\s+\d", block)[0]

        # Extract metric patterns like '518/527', '98.3%', '0/555', '95.0%', '[96.8%, 99.1%]'
        patterns = re.findall(r"\b\d+/\d+\b|\b\d+\.\d+%\b|\[\d+\.\d+%,\s*\d+\.\d+%\]", section_content)
        assert len(patterns) > 0, "Tagged section should contain verifiable numbers"

        for token in patterns:
            # Token must appear verbatim in docs/results_main.md
            assert token in results_text, f"Token '{token}' from README not found in docs/results_main.md"
