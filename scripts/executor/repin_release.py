"""Re-pin the swin-classification-pipeline release to freshly built worker images.

Builder step, run from the pipeline repo root on a release branch. Uses the ml-worker contract
library at the pinned contract revision (0f0c221) for every digest and for the composition
solver, so nothing here re-implements contract identity.

Inputs (all written by the Executor scripts in this directory):
  out/images.txt            image ids (containerd manifest digests) for both images
  out/validator-summary.json, out/finetuner-summary.json, out/versions.txt, out/gpu.txt, out/context-hygiene.txt
  VALIDATOR_REV, FINETUNER_REV
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

X = Path(r"C:\swin-executor\rebuild-2026-09-11")
CONTRACT_SRC = Path(r"C:\swin-executor\mlw-0f0c221")
CONTRACT_REV = "0f0c221222402721ee7716edf01378604cbd6ef3"
VALIDATOR_REPO = Path(r"C:\Users\Kurt Valcorza\Projects\swin-classification-dataset-validator")
FINETUNER_REPO = Path(r"C:\Users\Kurt Valcorza\Projects\swin-classification-finetuner")
ROOT = Path.cwd()
NEW_VERSION = "0.1.1"

head = subprocess.run(["git", "-C", str(CONTRACT_SRC), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
assert head == CONTRACT_REV, f"contract worktree at {head}, expected {CONTRACT_REV}"
sys.path.insert(0, str(CONTRACT_SRC / "src"))
from ml_worker_contract.composition import compose  # noqa: E402
from ml_worker_contract.identity import digest_document  # noqa: E402
from ml_worker_contract.jsonio import load_strict, loads_strict  # noqa: E402
from ml_worker_contract.release import build_contract_release  # noqa: E402
from ml_worker_contract.schemas import validate as validate_document  # noqa: E402

vrev = (X / "VALIDATOR_REV").read_text().strip()
frev = (X / "FINETUNER_REV").read_text().strip()
images = {}
for line in (X / "out" / "images.txt").read_text().splitlines():
    m = re.match(r"^(\S+) id=(sha256:[0-9a-f]{64}) repoDigests=(\S+) size=(\d+) created=(\S+)", line)
    if m:
        images[m.group(1).split(":")[0]] = {"tag": m.group(1), "id": m.group(2), "size": int(m.group(4)), "created": m.group(5)}
vimg, fimg = images["swin-classification-validator"], images["swin-classification-finetuner"]
vsum = json.loads((X / "out" / "validator-summary.json").read_text())
fsum = json.loads((X / "out" / "finetuner-summary.json").read_text())
versions = (X / "out" / "versions.txt").read_text().strip()
gpu = (X / "out" / "gpu.txt").read_text().strip()
hygiene = (X / "out" / "context-hygiene.txt").read_text()
assert not any(f"PRESENT {k}" in hygiene for k in (".git", ".gitignore", ".gitattributes", ".github")), "build context leaked repo metadata into an image:\n" + hygiene
assert vsum["state"] == "SUCCEEDED" and fsum["state"] == "SUCCEEDED"


def git_show(repo: Path, rev: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True, check=True).stdout


def tree_id(repo: Path, rev: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", f"{rev}^{{tree}}"], capture_output=True, text=True, check=True).stdout.strip()


# 1. worker manifests at the pinned revisions (byte-for-byte copies)
vmanifest_text = git_show(VALIDATOR_REPO, vrev, "worker-manifest.json")
fmanifest_text = git_show(FINETUNER_REPO, frev, "worker-manifest.json")
(ROOT / "workers/validator/worker-manifest.json").write_text(vmanifest_text, encoding="utf-8", newline="\n")
(ROOT / "workers/finetuner/worker-manifest.json").write_text(fmanifest_text, encoding="utf-8", newline="\n")
vmanifest = loads_strict(vmanifest_text)
fmanifest = loads_strict(fmanifest_text)

# 2. image smoke evidence
timm_v = re.search(r"timm (\S+)", versions).group(1)
torch_v = re.search(r"torch (\S+)", versions).group(1)
evidence = {
    "schemaVersion": "1.0",
    "status": "PASSED",
    "description": "End-to-end smoke executed from the runnable worker images themselves: fixture generated in the validator image, validator image produced the handoff, finetuner image trained/published/reloaded from its baked-in weights. Both stages ran with --network none; the finetuner ran with --gpus all. Procedure: C:\\swin-executor\\rebuild-2026-09-11\\{build_images.sh,image_smoke.sh} (Executor host nvidia-container-swin, a clone of the nvidia-container-base golden WSL image).",
    "images": {
        "validator": {"image": vimg["tag"], "imageDigest": vimg["id"], "sourceRevision": vrev, "sourceTree": tree_id(VALIDATOR_REPO, vrev), "imageSizeBytes": vimg["size"], "buildContext": "post-.dockerignore: .git/.github/tests absent from /opt/worker/src"},
        "finetuner": {"image": fimg["tag"], "imageDigest": fimg["id"], "sourceRevision": frev, "sourceTree": tree_id(FINETUNER_REPO, frev), "imageSizeBytes": fimg["size"], "buildContext": "post-.dockerignore: .git/.github/tests absent from /opt/worker/src", "weightStaging": "downloaded at pinned HF revisions during image build and SHA-256 verified against the catalog (scripts/stage_weights.py); a mismatch fails the build; no Hub cache left in the image"},
    },
    "environment": {
        "baseImage": "pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385",
        "timm": timm_v,
        "torch": torch_v,
        "runtimeVersions": versions,
        "device": f"cuda:0 ({gpu}, sm_120)",
        "host": "WSL2 distro nvidia-container-swin, Docker 29.7.2 with the containerd image store, NVIDIA Container Toolkit 1.20.0",
        "network": "none for both worker stages",
    },
    "results": {
        "validator": {
            "state": vsum["state"],
            "validatedDatasetManifestDigest": vsum["observed"]["validatedDatasetManifestDigest"],
            "dataPlanDigest": vsum["observed"].get("dataPlanDigest"),
            "runManifestDigest": vsum["runManifestDigest"],
            "l3Findings": vsum["l3Findings"],
        },
        "finetuner": {
            "state": fsum["state"],
            "validationAccuracy": fsum["metrics"]["core.metric.classification.accuracy"],
            "validationCrossEntropy": fsum["metrics"]["org.valcorza.metric.classification.cross-entropy"],
            "runManifestDigest": fsum["runManifestDigest"],
            "artifactBundleDigest": fsum["artifactBundleDigest"],
            "artifactMembers": fsum["artifactMembers"],
            "modelManifestEmitted": fsum["modelManifest"] is not None,
            "inputImageForms": fsum["inputImageForms"],
            "reproducibility": fsum["reproducibility"],
        },
    },
    "limitations": [
        "Bounded 8-image fixture smoke from the worker images; not production-scale.",
        "Image digests are the local containerd manifest digests; they become registry digests unchanged on push.",
        "Training remains REEXECUTABLE: cross-entropy is not bitwise reproducible across runs.",
    ],
}
(ROOT / "release/image-smoke-evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
evidence_digest = digest_document(load_strict(ROOT / "release/image-smoke-evidence.json"))

# 3. worker releases
SOURCE_SMOKE = "sha256:a3cffa345f5c023e5a203ad566dc1d971115ee48278b2d582476239d27456dce"  # finetuner blackwell-training-smoke-cf3f429 evidenceDigest
RUNTIME_PACKET = "sha256:230e094ee7fa4eb2e9cab7a74f0a45ce31c35bdeb9df68f83ac87e912059e522"  # finetuner blackwell-timm-1.0.28 packetManifestDigest
contract_release_digest = digest_document(build_contract_release())
old_v = load_strict(ROOT / "release/worker-release-validator.json")
old_f = load_strict(ROOT / "release/worker-release-finetuner.json")
assert old_v["contractReleaseDigest"] == contract_release_digest == old_f["contractReleaseDigest"]
vrelease = {**old_v, "version": NEW_VERSION, "imageDigest": vimg["id"], "workerManifestDigest": digest_document(vmanifest), "sourceRevision": vrev, "conformanceEvidenceDigests": [SOURCE_SMOKE, evidence_digest]}
frelease = {**old_f, "version": NEW_VERSION, "imageDigest": fimg["id"], "workerManifestDigest": digest_document(fmanifest), "sourceRevision": frev, "conformanceEvidenceDigests": [SOURCE_SMOKE, RUNTIME_PACKET, evidence_digest]}
for name, doc in (("validator", vrelease), ("finetuner", frelease)):
    _errs = validate_document(doc, "worker-release.schema.json"); assert not _errs, _errs
    (ROOT / f"release/worker-release-{name}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")

# 4. pipeline manifest, composition, pipeline release
manifest = load_strict(ROOT / "pipeline-manifest.json")
manifest["validatorWorkerReleaseDigest"] = digest_document(vrelease)
manifest["finetunerWorkerReleaseDigest"] = digest_document(frelease)
_errs = validate_document(manifest, "pipeline-manifest.schema.json"); assert not _errs, _errs
(ROOT / "pipeline-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
report = compose(manifest, vmanifest, fmanifest)
_errs = validate_document(report, "composition-report.schema.json"); assert not _errs, _errs
assert report["status"] == "COMPATIBLE", report["status"]
(ROOT / "release/composition-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
release = load_strict(ROOT / "release/pipeline-release.json")
release.update({
    "version": NEW_VERSION,
    "pipelineManifestDigest": digest_document(manifest),
    "validatorWorkerReleaseDigest": digest_document(vrelease),
    "finetunerWorkerReleaseDigest": digest_document(frelease),
    "contractReleaseDigest": contract_release_digest,
    "compositionReportDigest": digest_document(report),
})
_errs = validate_document(release, "pipeline-release.schema.json"); assert not _errs, _errs
(ROOT / "release/pipeline-release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8", newline="\n")

# 5. pipeline-metadata.json (Pipeline Spec 1.0 declaration) follows the release
meta_path = ROOT / "pipeline-metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["pipeline"]["version"] = NEW_VERSION
    for component in meta["components"]:
        rel = {"validator": vrelease, "finetuner": frelease}[component["role"]]
        component["source_revision"] = rel["sourceRevision"]
        component["release_digest"] = digest_document(rel)
        component["image_digest"] = rel["imageDigest"]
    meta["composition"]["report_digest"] = digest_document(report)
    meta["composition"]["status"] = report["status"]
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

print(json.dumps({
    "validator": {"sourceRevision": vrev, "imageDigest": vimg["id"], "workerReleaseDigest": digest_document(vrelease)},
    "finetuner": {"sourceRevision": frev, "imageDigest": fimg["id"], "workerReleaseDigest": digest_document(frelease)},
    "imageSmokeEvidenceDigest": evidence_digest,
    "compositionStatus": report["status"], "compositionChecks": len(report.get("checks", [])),
    "pipelineReleaseVersion": NEW_VERSION,
}, indent=2))
