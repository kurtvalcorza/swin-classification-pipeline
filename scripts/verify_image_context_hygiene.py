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

import fnmatch
import json
import os
import posixpath
import sys
from pathlib import Path
from typing import NamedTuple

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


class Result(NamedTuple):
    label: str
    ok: bool
    detail: str
    # True when the check could not run (no checkout, unresolvable revision),
    # as opposed to a real hygiene defect at a resolvable revision.
    setup_error: bool = False


def _normalise(pattern: str) -> str:
    """Docker's normalisation: trim, drop a leading `/`, then `filepath.Clean`.

    So `.git`, `.git/`, `/.git`, `./.git` and ` .git ` all become `.git`.
    """
    pattern = pattern.strip().lstrip("/")
    return posixpath.normpath(pattern) if pattern else pattern


def _segment_may_match_git(segment: str) -> bool:
    """Could one Go `filepath.Match` path segment match the name `.git`?

    Conservative: `**` matches any number of segments (including the `.git`
    one), and a segment using `\\` escapes, which `fnmatch` does not model,
    is assumed to match.
    """
    if segment == "**" or "\\" in segment:
        return True
    # Go negates a character class with `[^...]`, fnmatch with `[!...]`.
    return fnmatch.fnmatchcase(REQUIRED_PATTERN, segment.replace("[^", "[!"))


def reincludes_git(negated: str) -> bool:
    """Could the `!` pattern `negated` re-include `.git` or any path under it?

    Docker patterns are anchored at the context root, so the first path segment
    decides whether the pattern can reach the top-level `.git` directory:
    `!.git/objects`, `!**/.git`, `!.git*` and `!*` all can; `!src/.git` cannot.
    """
    first = _normalise(negated).split("/", 1)[0]
    return bool(first) and first != "." and _segment_may_match_git(first)


def excludes_required(blob: str) -> bool:
    """True when the .dockerignore excludes `.git` and no later line re-includes it.

    Docker applies .dockerignore lines in order and the last match wins, so any
    later `!` line that can match `.git` or a path under it undoes an earlier
    exclusion. Comments and blank lines are ignored.
    """
    excluded = False
    for line in blob.splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        if pattern.startswith("!"):
            if reincludes_git(pattern[1:]):
                excluded = False
        elif _normalise(pattern) == REQUIRED_PATTERN:
            excluded = True
    return excluded


def check_worker(role: str, env_name: str, revision: str) -> Result:
    """Check one worker's pinned source revision; never raises for setup errors."""
    short = revision[:7]
    label = f"{role} image source {short} excludes repository metadata"

    source = os.environ.get(env_name)
    if not source:
        return Result(label, False, f"set {env_name} to a {role} checkout", True)

    blob = git_output(Path(source), "show", f"{revision}:.dockerignore")
    if blob is None:
        # Distinguish "no such revision" from "revision has no .dockerignore",
        # so a missing checkout is never reported as a hygiene defect.
        if git_output(Path(source), "cat-file", "-e", f"{revision}^{{commit}}") is None:
            return Result(label, False, f"{short} not resolvable in {env_name}", True)
        return Result(label, False, f"no .dockerignore at {short}; build context embeds .git")

    if not excludes_required(blob):
        return Result(
            label, False, f".dockerignore at {short} does not exclude {REQUIRED_PATTERN}"
        )

    return Result(label, True, f".dockerignore at {short} excludes {REQUIRED_PATTERN}")


def check_release(root: Path = ROOT) -> list[Result]:
    """Hygiene results for every worker release committed under `root`."""
    results = []
    for role, env_name, release_path in WORKERS:
        release = json.loads((root / release_path).read_text(encoding="utf-8"))
        results.append(check_worker(role, env_name, release["sourceRevision"]))
    return results


SETUP_MESSAGE = (
    "\nThe check could not run for every worker: set VALIDATOR_SRC and "
    "FINETUNER_SRC to git checkouts of the worker repositories that contain the "
    "pinned sourceRevision. This is a setup error, not a hygiene verdict."
)
REMEDIATION_MESSAGE = (
    "\nA pinned image source does not exclude `.git` from its build context, so "
    "the image embeds repository history. Rebuild the worker image from a "
    "revision whose `.dockerignore` excludes `.git`, re-run the image smoke, and "
    "repin release/worker-release-*.json."
)


def main() -> int:
    results = check_release()

    for result in results:
        print(f"{'PASS' if result.ok else 'FAIL'}  {result.label} ({result.detail})")

    failed = [result for result in results if not result.ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")

    if any(result.setup_error for result in failed):
        print(SETUP_MESSAGE, file=sys.stderr)
    if any(not result.setup_error for result in failed):
        print(REMEDIATION_MESSAGE, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
