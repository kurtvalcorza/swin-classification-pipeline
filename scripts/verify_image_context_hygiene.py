"""Check that the pinned worker images were built from build-context-clean source.

`scripts/verify_release.py` proves that the release's *bindings* are coherent:
every digest re-derives, and each committed manifest agrees with the manifest at
its declared `sourceRevision`. That is a statement about identity, not about the
quality of the artifact the identity names. A release can therefore pin an image
whose digest chain is perfectly coherent and whose contents are still wrong.

This script checks one such property that no digest agreement can detect.

Each worker Dockerfile copies its build context wholesale (`COPY . /opt/worker/src`),
and `.gitignore` does not filter Docker build contexts. Without a `.dockerignore`
excluding `.git`, the published image embeds the full repository history, and its
contents depend on whose workspace built it rather than on the pinned revision.
An image built from a revision that predates that exclusion ships exactly that
defect, sealed under an immutable digest.

The check is deliberately kept out of `verify_release.py` and out of CI: the
release's binding claims are true and should stay green, while this is a
separate, currently-failing claim about the artifacts those bindings name. Once
both worker images are rebuilt from post-`.dockerignore` revisions and the
releases are repinned, this check passes and can be folded into the release gate.

Usage (needs the same worker checkouts `verify_release.py` uses):

    VALIDATOR_SRC=<validator checkout> \
    FINETUNER_SRC=<finetuner checkout> \
        python scripts/verify_image_context_hygiene.py

Exit 0 only when every pinned image source excludes repository metadata.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKERS = [
    ("validator", "VALIDATOR_SRC", "release/worker-release-validator.json"),
    ("finetuner", "FINETUNER_SRC", "release/worker-release-finetuner.json"),
]

# Excluding `.git` is the property that actually matters: it is what carries
# repository history and builder-workspace dependence into the image.
REQUIRED_PATTERN = ".git"


def git_output(repo: Path, *args: str) -> str | None:
    """Return stdout of a git command in `repo`, or None if it cannot run."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def dockerignore_patterns(blob: str) -> set[str]:
    """The meaningful exclusion patterns in a .dockerignore, comments removed."""
    return {
        line.strip()
        for line in blob.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def check_worker(role: str, env_name: str, release_path: str) -> tuple[str, bool, str]:
    release = json.loads((ROOT / release_path).read_text())
    revision = release["sourceRevision"]
    short = revision[:7]
    label = f"{role} image source {short} excludes repository metadata"

    source = os.environ.get(env_name)
    if not source:
        return label, False, f"set {env_name} to a {role} checkout"

    blob = git_output(Path(source), "show", f"{revision}:.dockerignore")
    if blob is None:
        # Distinguish "no such revision" from "revision has no .dockerignore",
        # so a missing checkout is never reported as a hygiene defect.
        if git_output(Path(source), "cat-file", "-e", f"{revision}^{{commit}}") is None:
            return label, False, f"{short} not resolvable in {env_name}"
        return label, False, f"no .dockerignore at {short}; build context embeds .git"

    if REQUIRED_PATTERN not in dockerignore_patterns(blob):
        return label, False, f".dockerignore at {short} does not exclude {REQUIRED_PATTERN}"

    return label, True, f".dockerignore at {short} excludes {REQUIRED_PATTERN}"


def main() -> int:
    results = [check_worker(*worker) for worker in WORKERS]

    for label, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {label} ({detail})")

    failed = [label for label, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")

    if failed:
        print(
            "\nThe pinned images were built from revisions predating their "
            "`.dockerignore`, so they embed repository history. Rebuild both "
            "worker images from post-`.dockerignore` revisions, re-run the image "
            "smoke, and repin release/worker-release-*.json.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
