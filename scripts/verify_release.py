"""Re-derive and verify every release binding in this repository.

Usage:
    ML_WORKER_SRC=<path to ml-worker checkout at the pinned contractRevision> \
        python scripts/verify_release.py

Exits 0 only when every digest in the committed release documents reproduces
from the committed inputs and the pinned contract: worker-manifest digests,
worker-release digests, the pipeline-manifest digest, a freshly re-run
composition (must be COMPATIBLE and byte-agree with the committed report),
the contract-release digest, and schema validation of every document.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_REVISION = "0f0c221222402721ee7716edf01378604cbd6ef3"


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    contract_src = os.environ.get("ML_WORKER_SRC")
    if not contract_src:
        fail("set ML_WORKER_SRC to an ml-worker tree at the pinned contract revision")
    sys.path.insert(0, str(Path(contract_src) / "src"))
    from ml_worker_contract.composition import compose
    from ml_worker_contract.identity import digest_document
    from ml_worker_contract.jsonio import load_strict
    from ml_worker_contract.release import build_contract_release
    from ml_worker_contract.schemas import validate as validate_document

    def load(relative: str):
        return load_strict(ROOT / relative)

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

    validator_release = load("release/worker-release-validator.json")
    finetuner_release = load("release/worker-release-finetuner.json")
    checks.append(
        (
            "validator manifest digest",
            digest_document(load("workers/validator/worker-manifest.json"))
            == validator_release["workerManifestDigest"],
        )
    )
    checks.append(
        (
            "finetuner manifest digest",
            digest_document(load("workers/finetuner/worker-manifest.json"))
            == finetuner_release["workerManifestDigest"],
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

    report = compose(
        manifest,
        load("workers/validator/worker-manifest.json"),
        load("workers/finetuner/worker-manifest.json"),
    )
    committed_report = load("release/composition-report.json")
    checks.append(("composition status COMPATIBLE", report["status"] == "COMPATIBLE"))
    checks.append(
        (
            "composition report reproduces",
            digest_document(report) == digest_document(committed_report),
        )
    )

    release = load("release/pipeline-release.json")
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
