"""Check that the pinned worker images were built from build-context-clean source.

`scripts/verify_release.py` proves that the release's *bindings* are coherent:
every digest re-derives, and each committed manifest agrees with the manifest at
its declared `sourceRevision`. That is a statement about identity, not about the
artifact the identity names. A release can pin an image whose digest chain is
perfectly coherent and whose contents are still wrong.

This script checks one such property that no digest agreement can detect.

Each worker Dockerfile copies its build context wholesale (`COPY . /opt/worker/src`),
and `.gitignore` does not filter Docker build contexts. Without a `.dockerignore`
excluding `.git`, the image embeds the full repository history, and its contents
depend on whose workspace built it rather than on the pinned revision.

Review finding R1 (PR #4) was that the 0.1.0 pins predated both workers'
`.dockerignore`. Release 0.1.1 (#11) re-pinned to post-`.dockerignore` revisions,
so this check passes and is part of the release gate: `verify_release.py` runs it
for both workers, and the `verify-image-release` workflow runs it against public
checkouts of both worker repositories, so a re-pin to a revision whose
`.dockerignore` is missing or does not exclude `.git` fails CI.

Usage (needs the same worker checkouts `verify_release.py` uses):

    VALIDATOR_SRC=<validator checkout> \
    FINETUNER_SRC=<finetuner checkout> \
        python scripts/verify_image_context_hygiene.py

Exit 0 only when every pinned image source excludes repository metadata.
A missing checkout and an unresolvable revision are reported distinctly from a
hygiene defect, and both fail closed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_release import git_output  # noqa: E402  (shared helper, stdlib-only module scope)

ROOT = Path(__file__).resolve().parents[1]

WORKERS = [
    ("validator", "VALIDATOR_SRC", "release/worker-release-validator.json"),
    ("finetuner", "FINETUNER_SRC", "release/worker-release-finetuner.json"),
]

# Excluding `.git` is the property that matters: it is what carries repository
# history and builder-workspace dependence into the image.
REQUIRED_PATTERN = ".git"


def _normalise(pattern: str) -> str:
    """`.git`, `.git/`, `/.git` and `./.git` match the same top-level directory."""
    return pattern.removeprefix("./").strip("/")


def excludes_required(blob: str) -> bool:
    """True when the .dockerignore excludes `.git` and no later line re-includes it.

    Docker applies .dockerignore lines in order and the last match wins, so a
    later `!.git` (or `!.git/**`) undoes an earlier exclusion. Comments and blank
    lines are ignored.
    """
    excluded = False
    for line in blob.splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        if pattern.startswith("!"):
            if _normalise(pattern[1:]) in (REQUIRED_PATTERN, f"{REQUIRED_PATTERN}/**"):
                excluded = False
        elif _normalise(pattern) == REQUIRED_PATTERN:
            excluded = True
    return excluded


def check_worker(role: str, env_name: str, revision: str) -> tuple[str, bool, str]:
    """Check one worker's pinned source revision; never raises for setup errors."""
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

    if not excludes_required(blob):
        return label, False, f".dockerignore at {short} does not exclude {REQUIRED_PATTERN}"

    return label, True, f".dockerignore at {short} excludes {REQUIRED_PATTERN}"


def check_release(root: Path = ROOT) -> list[tuple[str, bool, str]]:
    """Hygiene results for every worker release committed under `root`."""
    results = []
    for role, env_name, release_path in WORKERS:
        release = json.loads((root / release_path).read_text(encoding="utf-8"))
        results.append(check_worker(role, env_name, release["sourceRevision"]))
    return results


def main() -> int:
    results = check_release()

    for label, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {label} ({detail})")

    failed = [label for label, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")

    if failed:
        print(
            "\nA pinned image source does not exclude `.git` from its build "
            "context, so the image embeds repository history. Rebuild the "
            "worker image from a revision whose `.dockerignore` excludes `.git`, "
            "re-run the image smoke, and repin release/worker-release-*.json.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
