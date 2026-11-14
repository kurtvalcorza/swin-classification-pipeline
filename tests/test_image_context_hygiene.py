"""Regression tests for the image build-context hygiene gate (issue #13, finding R1).

The gate must refuse a pinned worker revision whose `.dockerignore` is missing or
does not exclude `.git`, must report setup errors distinctly from hygiene
defects, and must stay wired into both `verify_release.py` and the
`verify-image-release` workflow.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "verify_image_context_hygiene", SCRIPTS / "verify_image_context_hygiene.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hygiene = _load()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.invalid", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _worker(tmp_path: Path, dockerignore: str | None) -> tuple[Path, str]:
    """A throwaway worker repo with one commit; returns (repo, revision)."""
    repo = tmp_path / "worker"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / "Dockerfile").write_text("FROM scratch\nCOPY . /opt/worker/src\n")
    if dockerignore is not None:
        (repo / ".dockerignore").write_text(dockerignore)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    "dockerignore",
    [
        ".git\n",
        "# keep history out\n.git/\n__pycache__\n",
        "/.git\n",
        "./.git\n",
        "  .git  \n",
        ".git\n!src/.git\n",  # anchored at the root: cannot reach the top-level .git
    ],
)
def test_dockerignore_excluding_git_passes(tmp_path, monkeypatch, dockerignore):
    repo, revision = _worker(tmp_path, dockerignore)
    monkeypatch.setenv("VALIDATOR_SRC", str(repo))
    _, ok, detail, _ = hygiene.check_worker("validator", "VALIDATOR_SRC", revision)
    assert ok, detail


@pytest.mark.parametrize(
    "dockerignore",
    [
        "__pycache__\n.venv\n",  # required pattern missing
        "# .git\n",  # only a comment
        ".github\n.gitignore\n",  # near-misses are not .git
        ".git\n!.git\n",  # later negation re-includes it
        ".git\n!.git/**\n",
        # Any negation that can match .git or a path under it re-includes it.
        ".git\n!.git/objects\n",
        ".git\n!**/.git\n",
        ".git\n!.git*\n",
        ".git\n!*\n",
        ".git\n! .git\n",
    ],
)
def test_dockerignore_not_excluding_git_fails(tmp_path, monkeypatch, dockerignore):
    repo, revision = _worker(tmp_path, dockerignore)
    monkeypatch.setenv("VALIDATOR_SRC", str(repo))
    _, ok, detail, _ = hygiene.check_worker("validator", "VALIDATOR_SRC", revision)
    assert not ok
    assert "does not exclude .git" in detail


def test_missing_dockerignore_fails_as_hygiene_defect(tmp_path, monkeypatch):
    repo, revision = _worker(tmp_path, None)
    monkeypatch.setenv("VALIDATOR_SRC", str(repo))
    _, ok, detail, setup_error = hygiene.check_worker("validator", "VALIDATOR_SRC", revision)
    assert not ok
    assert not setup_error
    assert "no .dockerignore at" in detail


def test_unresolvable_revision_is_a_setup_error_not_a_hygiene_defect(tmp_path, monkeypatch):
    repo, _ = _worker(tmp_path, ".git\n")
    monkeypatch.setenv("VALIDATOR_SRC", str(repo))
    _, ok, detail, setup_error = hygiene.check_worker("validator", "VALIDATOR_SRC", "0" * 40)
    assert not ok
    assert setup_error
    assert "not resolvable in VALIDATOR_SRC" in detail


def test_unset_checkout_fails_closed(monkeypatch):
    monkeypatch.delenv("FINETUNER_SRC", raising=False)
    _, ok, detail, setup_error = hygiene.check_worker("finetuner", "FINETUNER_SRC", "a" * 40)
    assert not ok
    assert setup_error
    assert "set FINETUNER_SRC" in detail


def test_check_release_reads_the_committed_pins(tmp_path, monkeypatch):
    """A re-pin to a pre-.dockerignore revision is refused end to end."""
    good, good_rev = _worker(tmp_path / "good", ".git\n")
    bad, bad_rev = _worker(tmp_path / "bad", None)
    staged = tmp_path / "staged"
    (staged / "release").mkdir(parents=True)
    for role, rev in (("validator", good_rev), ("finetuner", bad_rev)):
        release = json.loads((ROOT / f"release/worker-release-{role}.json").read_text())
        release["sourceRevision"] = rev
        (staged / f"release/worker-release-{role}.json").write_text(json.dumps(release))
    monkeypatch.setenv("VALIDATOR_SRC", str(good))
    monkeypatch.setenv("FINETUNER_SRC", str(bad))
    results = hygiene.check_release(staged)
    assert [r.ok for r in results] == [True, False]


def test_main_exit_code_reflects_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VALIDATOR_SRC", raising=False)
    monkeypatch.delenv("FINETUNER_SRC", raising=False)
    assert hygiene.main() == 1
    captured = capsys.readouterr()
    assert "0/2 checks passed" in captured.out
    # Missing checkouts are a setup error: no rebuild advice.
    assert "setup error" in captured.err
    assert "Rebuild the worker image" not in captured.err


def test_main_gives_remediation_only_for_a_hygiene_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        hygiene,
        "check_release",
        lambda: [
            hygiene.Result("validator ok", True, "fine"),
            hygiene.Result("finetuner bad", False, "no .dockerignore at abc1234"),
        ],
    )
    assert hygiene.main() == 1
    captured = capsys.readouterr()
    assert "1/2 checks passed" in captured.out
    assert "Rebuild the worker image" in captured.err
    assert "setup error" not in captured.err


@pytest.mark.skipif(
    not (os.environ.get("VALIDATOR_SRC") and os.environ.get("FINETUNER_SRC")),
    reason="needs VALIDATOR_SRC and FINETUNER_SRC worker checkouts",
)
def test_committed_release_passes_against_real_worker_checkouts():
    results = hygiene.check_release()
    assert all(r.ok for r in results), results


def test_hygiene_is_wired_into_verify_release():
    source = (SCRIPTS / "verify_release.py").read_text(encoding="utf-8")
    assert "from verify_image_context_hygiene import check_worker" in source


def test_hygiene_is_wired_into_ci_with_both_worker_checkouts():
    workflow = (ROOT / ".github/workflows/verify-image-release.yml").read_text(encoding="utf-8")
    assert "python scripts/verify_image_context_hygiene.py" in workflow
    assert "repository: kurtvalcorza/swin-classification-dataset-validator" in workflow
    assert "repository: kurtvalcorza/swin-classification-finetuner" in workflow
    assert "fetch-depth: 0" in workflow
