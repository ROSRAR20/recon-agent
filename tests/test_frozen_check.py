"""Tests for frozen baseline and registration-final integrity checks (Day 7b-5).

Uses temporary git repositories to thoroughly verify:
- Missing 'registration-final' tag fails with clear message.
- Tag present and files unchanged passes cleanly.
- Tampering with any frozen file (independent_checker.py, verifier.py, systems/,
  docs/preregistration.md) fails verification.
- Non-mock pilot refuses to run if frozen check fails or git tree is dirty.
- Mock pilot proceeds even if registration-final tag is missing.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from scripts.check_frozen import check_frozen_baselines
from scripts.run_pilot import run_pilot


def git_cmd(cmd: list[str], cwd: str) -> str:
    """Helper to run git commands in temp directory."""
    res = subprocess.run(
        ["git"] + cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


@pytest.fixture
def temp_git_repo(tmp_path):
    """Fixture creating a fully valid git repository with frozen files and b1-frozen tag."""
    repo = tmp_path / "repo"
    repo.mkdir()
    repo_str = str(repo)

    git_cmd(["init"], cwd=repo_str)
    git_cmd(["config", "user.name", "Test User"], cwd=repo_str)
    git_cmd(["config", "user.email", "test@example.com"], cwd=repo_str)

    # Create directory tree
    (repo / "src" / "recon_agent" / "eval").mkdir(parents=True)
    (repo / "src" / "recon_agent" / "llm" / "prompts").mkdir(parents=True)
    (repo / "src" / "recon_agent" / "systems").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)

    # Create frozen files
    (repo / "src" / "recon_agent" / "eval" / "independent_checker.py").write_text("# independent checker v1\n", encoding="utf-8")
    (repo / "src" / "recon_agent" / "llm" / "verifier.py").write_text("# verifier code\n", encoding="utf-8")
    (repo / "src" / "recon_agent" / "llm" / "prompts" / "system.txt").write_text("# system prompt\n", encoding="utf-8")
    (repo / "src" / "recon_agent" / "systems" / "__init__.py").write_text("# systems init\n", encoding="utf-8")
    (repo / "src" / "recon_agent" / "systems" / "b0.py").write_text("# b0\n", encoding="utf-8")
    (repo / "src" / "recon_agent" / "systems" / "b1.py").write_text("# b1\n", encoding="utf-8")
    (repo / "src" / "recon_agent" / "systems" / "b1r.py").write_text("# b1r\n", encoding="utf-8")
    (repo / "docs" / "preregistration.md").write_text("# preregistration document\n", encoding="utf-8")

    git_cmd(["add", "."], cwd=repo_str)
    git_cmd(["commit", "-m", "Initial commit with frozen files"], cwd=repo_str)
    git_cmd(["tag", "b1-frozen"], cwd=repo_str)

    return repo_str


def test_missing_registration_final_tag_fails(temp_git_repo):
    """Verify check fails with clear message when registration-final tag is missing."""
    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is False
    assert details["registration_final_commit"] is None
    assert any("Tag 'registration-final' does not exist in repository" in f for f in details["failures"])
    assert details["ancestors"]["b1-frozen"] is True


def test_clean_pass_with_both_tags(temp_git_repo):
    """Verify check passes when registration-final tag exists and files match."""
    git_cmd(["tag", "registration-final"], cwd=temp_git_repo)

    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is True
    assert details["registration_final_commit"] is not None
    assert not details["failures"]
    assert details["ancestors"]["b1-frozen"] is True
    assert details["ancestors"]["registration-final"] is True
    assert details["diffs"]["systems_vs_b1_frozen"] is True
    assert details["diffs"]["checker_vs_registration_final"] is True
    assert details["diffs"]["verifier_vs_registration_final"] is True
    assert details["diffs"]["systems_vs_registration_final"] is True
    assert details["diffs"]["prereg_doc_vs_registration_final"] is True

    # Blob hashes must be non-empty
    f_hashes = details["file_hashes"]
    assert f_hashes["src/recon_agent/eval/independent_checker.py"] is not None
    assert f_hashes["src/recon_agent/llm/verifier.py"] is not None
    assert f_hashes["docs/preregistration.md"] is not None
    assert f_hashes["src/recon_agent/systems/"] is not None
    assert len(f_hashes["systems_files"]) == 4


def test_diff_on_independent_checker_fails(temp_git_repo):
    """Verify modifying independent_checker.py triggers failure."""
    git_cmd(["tag", "registration-final"], cwd=temp_git_repo)

    # Modify independent_checker.py
    checker_path = os.path.join(temp_git_repo, "src", "recon_agent", "eval", "independent_checker.py")
    with open(checker_path, "a", encoding="utf-8") as f:
        f.write("# unauthorized modification\n")
    git_cmd(["commit", "-am", "Tamper with checker"], cwd=temp_git_repo)

    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is False
    assert details["diffs"]["checker_vs_registration_final"] is False
    assert any("independent_checker.py differs" in f for f in details["failures"])


def test_diff_on_verifier_fails(temp_git_repo):
    """Verify modifying verifier.py triggers failure."""
    git_cmd(["tag", "registration-final"], cwd=temp_git_repo)

    # Modify verifier.py
    verifier_path = os.path.join(temp_git_repo, "src", "recon_agent", "llm", "verifier.py")
    with open(verifier_path, "a", encoding="utf-8") as f:
        f.write("# unauthorized verifier edit\n")
    git_cmd(["commit", "-am", "Tamper with verifier"], cwd=temp_git_repo)

    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is False
    assert details["diffs"]["verifier_vs_registration_final"] is False
    assert any("verifier.py differs" in f for f in details["failures"])


def test_diff_on_preregistration_doc_fails(temp_git_repo):
    """Verify modifying docs/preregistration.md triggers failure."""
    git_cmd(["tag", "registration-final"], cwd=temp_git_repo)

    # Modify docs/preregistration.md
    doc_path = os.path.join(temp_git_repo, "docs", "preregistration.md")
    with open(doc_path, "a", encoding="utf-8") as f:
        f.write("# unauthorized prereg edit\n")
    git_cmd(["commit", "-am", "Tamper with preregistration doc"], cwd=temp_git_repo)

    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is False
    assert details["diffs"]["prereg_doc_vs_registration_final"] is False
    assert any("preregistration.md differs" in f for f in details["failures"])


def test_diff_on_systems_fails(temp_git_repo):
    """Verify modifying systems/ triggers failure against registration-final and b1-frozen."""
    git_cmd(["tag", "registration-final"], cwd=temp_git_repo)

    # Modify systems/b1.py
    sys_path = os.path.join(temp_git_repo, "src", "recon_agent", "systems", "b1.py")
    with open(sys_path, "a", encoding="utf-8") as f:
        f.write("# unauthorized system edit\n")
    git_cmd(["commit", "-am", "Tamper with system"], cwd=temp_git_repo)

    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is False
    assert details["diffs"]["systems_vs_registration_final"] is False
    assert details["diffs"]["systems_vs_b1_frozen"] is False
    assert any("systems/ differs from tag registration-final" in f for f in details["failures"])
    assert any("systems/ differs from tag b1-frozen" in f for f in details["failures"])


def test_non_mock_pilot_refuses_when_frozen_fails(tmp_path):
    """Verify non-mock pilot run refuses to start when frozen check fails."""
    fake_details = {
        "registration_final_commit": None,
        "failures": ["Tag 'registration-final' does not exist in repository."],
        "file_hashes": {},
    }
    with patch("scripts.run_pilot.check_frozen_baselines", return_value=(False, fake_details)):
        with pytest.raises(RuntimeError, match="Frozen baseline check failed"):
            run_pilot(seed=1, arms=["default"], n_residual=5, n_control=2, mock=False, output_dir=str(tmp_path))


def test_non_mock_pilot_refuses_when_git_dirty(tmp_path):
    """Verify non-mock pilot run refuses to start when git tree is dirty."""
    fake_details = {
        "registration_final_commit": "12345678",
        "failures": [],
        "file_hashes": {},
    }
    fake_env = MagicMock()
    fake_env.git_dirty = True

    with patch("scripts.run_pilot.check_frozen_baselines", return_value=(True, fake_details)):
        with patch("scripts.run_pilot.check_env", return_value=fake_env):
            with pytest.raises(RuntimeError, match="Git tracked working tree is dirty"):
                run_pilot(seed=1, arms=["default"], n_residual=5, n_control=2, mock=False, output_dir=str(tmp_path))


def test_mock_pilot_runs_despite_missing_registration_tag(tmp_path):
    """Verify mock pilot proceeds cleanly even when registration-final tag is missing."""
    out_dir = str(tmp_path / "mock_res")
    summaries = run_pilot(
        seed=1,
        arms=["default"],
        n_residual=5,
        n_control=2,
        mock=True,
        output_dir=out_dir,
    )
    assert summaries["default"]["status"] == "completed"


def test_change_log_chain_valid_and_broken(temp_git_repo):
    """Verify change log hash chaining: valid chain passes, broken chain fails."""
    git_cmd(["tag", "registration-final"], cwd=temp_git_repo)

    # 1. Get initial hash of verifier.py
    v_path = os.path.join(temp_git_repo, "src", "recon_agent", "llm", "verifier.py")
    orig_hash = git_cmd(["hash-object", "src/recon_agent/llm/verifier.py"], cwd=temp_git_repo)

    # 2. Modify verifier.py
    with open(v_path, "a", encoding="utf-8") as f:
        f.write("# modification 1\n")
    new_hash = git_cmd(["hash-object", "src/recon_agent/llm/verifier.py"], cwd=temp_git_repo)

    # Without change_log entry, check fails
    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is False
    assert details["diffs"]["verifier_chain_valid"] is False

    # 3. Create docs/change_log.md with unbroken chain
    cl_path = os.path.join(temp_git_repo, "docs", "change_log.md")
    valid_content = f"""# Change Log
## Entry 1: Verifier update
- **Date**: 2026-11-22
- **File**: `src/recon_agent/llm/verifier.py`
- **Old Git Blob Hash**: `{orig_hash}`
- **New Git Blob Hash**: `{new_hash}`
"""
    with open(cl_path, "w", encoding="utf-8") as f:
        f.write(valid_content)

    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is True
    assert details["diffs"]["verifier_chain_valid"] is True

    # 4. Corrupt the old_hash in change_log to break the chain
    broken_content = valid_content.replace(orig_hash, "0000000000000000000000000000000000000000")
    with open(cl_path, "w", encoding="utf-8") as f:
        f.write(broken_content)

    ok, details = check_frozen_baselines(repo_root=temp_git_repo)
    assert ok is False
    assert details["diffs"]["verifier_chain_valid"] is False
    assert any("Broken hash chain" in f for f in details["failures"])

