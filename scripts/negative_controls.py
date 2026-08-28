"""Discriminating negative controls for `scripts/verify_release.py`.

A verifier is only worth its exit code if it refuses the cases it claims to
refuse. Each control here constructs a scenario that a weaker verifier accepts,
runs the real verifier against it, and asserts the specific refusal. Every
scenario is built in a temporary directory; nothing in this repository or in
the caller's checkouts is modified.

Usage (same three checkouts `verify_release.py` needs):

    ML_WORKER_SRC=<ml-worker checkout> \
    VALIDATOR_SRC=<validator checkout> \
    FINETUNER_SRC=<finetuner checkout> \
        python scripts/negative_controls.py

Exit 0 means every control discriminated.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify_release.py"
CONTRACT_REVISION = "0f0c221222402721ee7716edf01378604cbd6ef3"

# Module-level code carrying an observable side effect. A contract checkout is
# imported by the verifier, so a checkout that is never proved is a checkout
# whose code must never run.
SENTINEL_SOURCE = """

import os as _os
import pathlib as _pathlib

_marker = _os.environ.get("CONTRACT_IMPORT_SENTINEL")
if _marker:
    _pathlib.Path(_marker).write_text("unverified contract code executed\\n")
"""


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def run_verifier(
    repo: Path, contract: Path, sentinel: Path | None = None
) -> tuple[int, str]:
    env = dict(os.environ)
    env["ML_WORKER_SRC"] = str(contract)
    if sentinel is not None:
        env["CONTRACT_IMPORT_SENTINEL"] = str(sentinel)
    else:
        env.pop("CONTRACT_IMPORT_SENTINEL", None)
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "verify_release.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo),
    )
    return completed.returncode, completed.stdout + completed.stderr


def clone_contract(source: str, into: Path) -> Path:
    subprocess.run(
        ["git", "clone", "-q", "--shared", source, str(into)],
        check=True,
        capture_output=True,
    )
    git(into, "checkout", "-q", "--detach", CONTRACT_REVISION)
    return into


def copy_repo(into: Path) -> Path:
    shutil.copytree(
        ROOT,
        into,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    return into


def main() -> int:
    contract_src = os.environ.get("ML_WORKER_SRC")
    if not contract_src:
        print("FAIL: set ML_WORKER_SRC, VALIDATOR_SRC and FINETUNER_SRC")
        return 1

    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # Control 0 -- the release as committed must still pass, or every
        # refusal below proves nothing.
        code, out = run_verifier(ROOT, Path(contract_src))
        record(
            "baseline: committed release verifies",
            code == 0 and "checks passed" in out,
            f"exit {code}",
        )

        # Control 1 -- a contract checkout at the wrong revision must be
        # refused BEFORE its code is imported, not merely reported afterwards.
        poisoned = clone_contract(contract_src, tmpdir / "poisoned")
        init = poisoned / "src" / "ml_worker_contract" / "__init__.py"
        init.write_text(init.read_text() + SENTINEL_SOURCE)
        git(poisoned, "add", "-A")
        git(poisoned, "-c", "user.email=nc@example.invalid", "-c", "user.name=nc",
            "commit", "-qm", "import side effect")
        sentinel = tmpdir / "sentinel-wrong-revision"
        code, out = run_verifier(ROOT, poisoned, sentinel)
        record(
            "contract at wrong revision is refused",
            code != 0 and "not the pinned contract revision" in out,
            f"exit {code}",
        )
        record(
            "...and its code never executed",
            not sentinel.exists(),
            "sentinel absent" if not sentinel.exists() else "SENTINEL WRITTEN",
        )

        # Control 2 -- a dirty checkout at the right revision is still unproved.
        dirty = clone_contract(contract_src, tmpdir / "dirty")
        (dirty / "STRAY.txt").write_text("uncommitted\n")
        code, out = run_verifier(ROOT, dirty)
        record(
            "dirty contract worktree is refused",
            code != 0 and "dirty worktree" in out,
            f"exit {code}",
        )

        # Control 3 -- a tree that is not a git checkout has no identity at all.
        nogit = tmpdir / "nogit"
        shutil.copytree(clone_contract(contract_src, tmpdir / "src-copy"), nogit)
        shutil.rmtree(nogit / ".git")
        init = nogit / "src" / "ml_worker_contract" / "__init__.py"
        init.write_text(init.read_text() + SENTINEL_SOURCE)
        sentinel = tmpdir / "sentinel-nogit"
        code, out = run_verifier(ROOT, nogit, sentinel)
        record(
            "non-git contract tree is refused",
            code != 0 and "not a git checkout" in out,
            f"exit {code}",
        )
        record(
            "...and its code never executed",
            not sentinel.exists(),
            "sentinel absent" if not sentinel.exists() else "SENTINEL WRITTEN",
        )

        # Control 4 -- a source manifest that stdlib JSON accepts but the
        # contract rejects (DOC-001: duplicate object keys are invalid) must
        # not be digested as though it were a valid contract document.
        dupworker = tmpdir / "dupworker"
        dupworker.mkdir()
        git_init = subprocess.run(
            ["git", "init", "-q", str(dupworker)], capture_output=True, text=True
        )
        if git_init.returncode != 0:
            record("duplicate-key source manifest is refused", False, "git init failed")
        else:
            (dupworker / "worker-manifest.json").write_text(
                '{\n  "schemaVersion": "1.0",\n'
                '  "workerId": "org.valcorza.swin-classification-dataset-validator",\n'
                '  "role": "validator",\n  "role": "validator",\n'
                '  "capabilities": {\n'
                '    "taskProfiles": ["core.task.vision.image-classification"],\n'
                '    "datasetRepresentations": ["core.dataset.vision.image-folder"]\n'
                "  }\n}\n"
            )
            git(dupworker, "add", "-A")
            git(dupworker, "-c", "user.email=nc@example.invalid", "-c", "user.name=nc",
                "commit", "-qm", "duplicate-key manifest")
            dup_revision = git(dupworker, "rev-parse", "HEAD").strip()

            staged = copy_repo(tmpdir / "staged")
            release_path = staged / "release" / "worker-release-validator.json"
            release = json.loads(release_path.read_text())
            release["sourceRevision"] = dup_revision
            release_path.write_text(json.dumps(release, indent=2) + "\n")

            env_backup = os.environ.get("VALIDATOR_SRC")
            os.environ["VALIDATOR_SRC"] = str(dupworker)
            code, out = run_verifier(staged, Path(contract_src))
            if env_backup is not None:
                os.environ["VALIDATOR_SRC"] = env_backup
            record(
                "duplicate-key source manifest is refused",
                code != 0 and "is strict contract JSON" in out,
                f"exit {code}",
            )

    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name} ({detail})")
    failed = [name for name, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} controls discriminated")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
