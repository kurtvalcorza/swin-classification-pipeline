"""Re-derive and verify every release binding in this repository.

Usage:
    ML_WORKER_SRC=<path to ml-worker checkout at the pinned contractRevision> \
    VALIDATOR_SRC=<path to swin-classification-dataset-validator checkout> \
    FINETUNER_SRC=<path to swin-classification-finetuner checkout> \
        python scripts/verify_release.py

Exits 0 only when every digest in the committed release documents reproduces
from the committed inputs and the pinned contract: worker-manifest digests,
worker-release digests, the pipeline-manifest digest, a freshly re-run
composition (must be COMPATIBLE and byte-agree with the committed report),
the contract-release digest, and schema validation of every document.

The verifier also proves the external identities it names rather than trusting
the caller: the contract checkout must be a clean git tree at the pinned
revision, and each committed worker-manifest copy must agree with
`worker-manifest.json` at that release's declared `sourceRevision` in the
worker repository. Those source checkouts are required; a missing checkout is
reported as a failed check, never as a silent pass.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_REVISION = "0f0c221222402721ee7716edf01378604cbd6ef3"


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


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


def main() -> int:
    contract_src = os.environ.get("ML_WORKER_SRC")
    if not contract_src:
        fail("set ML_WORKER_SRC to an ml-worker tree at the pinned contract revision")

    # Importing from ML_WORKER_SRC executes module-level code out of a
    # caller-supplied tree. Prove that tree's identity FIRST: a check that runs
    # after the import cannot undo code that has already executed, and that code
    # can influence every verification that follows. These are gates, not
    # checks -- they refuse before the trust boundary is crossed.
    contract_root = Path(contract_src)
    head = git_output(contract_root, "rev-parse", "HEAD")
    if head is None:
        fail(
            f"{contract_src} is not a git checkout; the contract identity cannot "
            "be established and its code will not be imported"
        )
    if head.strip() != CONTRACT_REVISION:
        fail(
            f"ML_WORKER_SRC is at {head.strip()[:7]}, not the pinned contract "
            f"revision {CONTRACT_REVISION[:7]}; refusing to import from it"
        )
    contract_status = git_output(contract_root, "status", "--porcelain")
    if contract_status is None:
        fail("cannot read the contract checkout's git status; refusing to import from it")
    if contract_status.strip():
        fail(
            "the contract checkout has a dirty worktree; refusing to import "
            "unverified contract code"
        )

    sys.path.insert(0, str(contract_root / "src"))
    from ml_worker_contract.composition import compose
    from ml_worker_contract.identity import digest_document
    from ml_worker_contract.jsonio import ContractJSONError, load_strict, loads_strict
    from ml_worker_contract.release import build_contract_release
    from ml_worker_contract.schemas import validate as validate_document

    def load(relative: str):
        return load_strict(ROOT / relative)

    def verify_worker_source(
        role: str, env_name: str, release_document: dict, committed_manifest: dict
    ) -> list[tuple[str, bool]]:
        """Prove the committed manifest copy equals the manifest at sourceRevision."""
        revision = release_document["sourceRevision"]
        short = revision[:7]
        source = os.environ.get(env_name)
        if not source:
            return [
                (
                    f"{role} source revision {short} resolvable "
                    f"(set {env_name} to a {role} checkout)",
                    False,
                )
            ]
        blob = git_output(Path(source), "show", f"{revision}:worker-manifest.json")
        if blob is None:
            return [
                (f"{role} source revision {short} resolvable in {env_name}", False)
            ]
        try:
            # Contract interchange semantics, not stdlib JSON: DOC-001 makes
            # duplicate object keys invalid, while json.loads silently accepts
            # them last-key-wins and would digest the result as a valid document.
            source_manifest = loads_strict(blob)
        except (ContractJSONError, ValueError):
            return [
                (f"{role} source revision {short} resolvable in {env_name}", True),
                (f"{role} manifest at {short} is strict contract JSON", False),
            ]
        return [
            (f"{role} source revision {short} resolvable in {env_name}", True),
            (
                f"{role} committed manifest agrees with manifest at {short}",
                digest_document(source_manifest) == digest_document(committed_manifest),
            ),
        ]

    checks: list[tuple[str, bool]] = []

    documents = {
        "worker-manifest.schema.json": [
            "workers/validator/worker-manifest.json",
            "workers/finetuner/worker-manifest.json",
        ],
        "worker-release.schema.json": [
            "release/worker-release-validator.json",
            "release/worker-release-finetuner.json",
        ],
        "pipeline-manifest.schema.json": ["pipeline-manifest.json"],
        "composition-report.schema.json": ["release/composition-report.json"],
        "pipeline-release.schema.json": ["release/pipeline-release.json"],
    }
    for schema_name, paths in documents.items():
        for path in paths:
            errors = validate_document(load(path), schema_name)
            checks.append((f"schema {path}", not errors))

    # Proven above, before the import; recorded here so the report is complete.
    checks.append(("contract source is a git checkout (gated before import)", True))
    checks.append(
        (
            f"contract source at pinned revision {CONTRACT_REVISION[:7]} "
            "(gated before import)",
            True,
        )
    )
    checks.append(("contract source tree clean (gated before import)", True))

    validator_manifest = load("workers/validator/worker-manifest.json")
    finetuner_manifest = load("workers/finetuner/worker-manifest.json")
    validator_release = load("release/worker-release-validator.json")
    finetuner_release = load("release/worker-release-finetuner.json")

    checks.append(
        (
            "validator manifest digest",
            digest_document(validator_manifest)
            == validator_release["workerManifestDigest"],
        )
    )
    checks.append(
        (
            "finetuner manifest digest",
            digest_document(finetuner_manifest)
            == finetuner_release["workerManifestDigest"],
        )
    )

    checks.extend(
        verify_worker_source(
            "validator", "VALIDATOR_SRC", validator_release, validator_manifest
        )
    )
    checks.extend(
        verify_worker_source(
            "finetuner", "FINETUNER_SRC", finetuner_release, finetuner_manifest
        )
    )

    # Digest agreement alone does not prove the documents describe the same
    # logical worker: a coherent repin can pair a release with a foreign
    # manifest. Bind the logical identities explicitly.
    checks.append(
        (
            "validator workerId agreement",
            validator_release["workerId"] == validator_manifest["workerId"],
        )
    )
    checks.append(
        (
            "finetuner workerId agreement",
            finetuner_release["workerId"] == finetuner_manifest["workerId"],
        )
    )
    checks.append(
        (
            "validator manifest declares validator role",
            validator_manifest["role"] == "validator",
        )
    )
    checks.append(
        (
            "finetuner manifest declares finetuner role",
            finetuner_manifest["role"] == "finetuner",
        )
    )

    manifest = load("pipeline-manifest.json")
    checks.append(
        (
            "validator release digest in manifest",
            digest_document(validator_release)
            == manifest["validatorWorkerReleaseDigest"],
        )
    )
    checks.append(
        (
            "finetuner release digest in manifest",
            digest_document(finetuner_release)
            == manifest["finetunerWorkerReleaseDigest"],
        )
    )

    report = compose(manifest, validator_manifest, finetuner_manifest)
    committed_report = load("release/composition-report.json")
    checks.append(("composition status COMPATIBLE", report["status"] == "COMPATIBLE"))
    checks.append(
        (
            "composition report reproduces",
            digest_document(report) == digest_document(committed_report),
        )
    )
    checks.append(
        (
            "composition report binds the committed worker manifests",
            committed_report["validatorWorkerManifestDigest"]
            == digest_document(validator_manifest)
            and committed_report["finetunerWorkerManifestDigest"]
            == digest_document(finetuner_manifest),
        )
    )
    checks.append(
        (
            "composition report binds the committed pipeline manifest",
            committed_report["pipelineManifestDigest"] == digest_document(manifest),
        )
    )

    release = load("release/pipeline-release.json")
    checks.append(
        (
            "pipelineId agreement across manifest and release",
            manifest["pipelineId"] == release["pipelineId"],
        )
    )
    checks.append(
        (
            "pipeline manifest digest in release",
            digest_document(manifest) == release["pipelineManifestDigest"],
        )
    )
    checks.append(
        (
            "composition report digest in release",
            digest_document(committed_report) == release["compositionReportDigest"],
        )
    )
    checks.append(
        (
            "worker release digests in release",
            release["validatorWorkerReleaseDigest"]
            == manifest["validatorWorkerReleaseDigest"]
            and release["finetunerWorkerReleaseDigest"]
            == manifest["finetunerWorkerReleaseDigest"],
        )
    )
    checks.append(
        (
            "contract release digest",
            digest_document(build_contract_release())
            == release["contractReleaseDigest"],
        )
    )
    checks.append(
        (
            "contract release digest consistent across all documents",
            validator_release["contractReleaseDigest"]
            == finetuner_release["contractReleaseDigest"]
            == release["contractReleaseDigest"],
        )
    )

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed "
          f"(contract {CONTRACT_REVISION[:7]})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
