#!/usr/bin/env python3
"""Verify pipeline-metadata.json against the DIMER Pipeline Specification 1.0 and this repository's release inputs.

Checks, all fail-closed:

- lifecycle_status, implementation_topology and capability_modes are declared with spec values (§3, GEN8);
- a `release` lifecycle is refused while any release gate is not PASS (GEN15, TEST21, REL17);
- every component's source revision and release digest agree with release/worker-release-*.json,
  and the composition digest agrees with release/composition-report.json (ARCH14/16, DOC13);
- every model entry agrees with pipeline-manifest.json and provenance/open-weights.json on identifier,
  immutable revision and SHA-256, and carries a machine-readable redistribution_status (LIC4);
- redistribution_status `unknown`/`prohibited` fail the hosting gate; `conditional` fails unless a
  recorded `condition_satisfied` block exists (LIC6/LIC7, TEST22); `permitted` requires a license and
  license_source (LIC2, LIC17);
- self-consistency of the two model identifiers per entry and no unreferenced/duplicate models.

Pass `--as-file PATH` to verify an alternative metadata document (used by the negative controls).
This validates metadata; it is not execution evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LIFECYCLES = ("scaffold", "candidate", "release")
TOPOLOGIES = ("PACKAGE", "COMPOSED-WORKERS")
MODES = ("GRADIENT-ADAPTATION", "CONTEXT-CONDITIONING", "PRETRAINED-INFERENCE", "MULTI-CAPABILITY-INFERENCE")
REDISTRIBUTION = ("permitted", "conditional", "prohibited", "unknown")
GATE_STATUSES = ("PASS", "OPEN", "FAIL", "NOT_APPLICABLE")


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_digest(document: dict) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verify(meta: dict, root: Path) -> list[str]:
    errors: list[str] = []
    manifest = load(root / "pipeline-manifest.json")
    provenance = load(root / "provenance" / "open-weights.json")
    pipeline_release = load(root / "release" / "pipeline-release.json")

    pipeline = meta.get("pipeline", {})
    if meta.get("pipeline_spec") != "1.0":
        errors.append("pipeline_spec must be '1.0'")
    if pipeline.get("id") != manifest.get("pipelineId"):
        errors.append("pipeline.id does not match pipeline-manifest.json pipelineId")
    if pipeline.get("version") != pipeline_release.get("version"):
        errors.append("pipeline.version does not match release/pipeline-release.json version")
    lifecycle = pipeline.get("lifecycle_status")
    if lifecycle not in LIFECYCLES:
        errors.append(f"lifecycle_status must be one of {LIFECYCLES}, got {lifecycle!r}")
    if pipeline.get("implementation_topology") not in TOPOLOGIES:
        errors.append(f"implementation_topology must be one of {TOPOLOGIES}")
    modes = pipeline.get("capability_modes") or []
    if not modes or any(mode not in MODES for mode in modes):
        errors.append(f"capability_modes must be a non-empty subset of {MODES}")
    if pipeline.get("task_profile") != manifest.get("taskProfile"):
        errors.append("pipeline.task_profile does not match pipeline-manifest.json taskProfile")
    if sorted(pipeline.get("dataset_representations") or []) != sorted(manifest.get("datasetRepresentations") or []):
        errors.append("pipeline.dataset_representations do not match pipeline-manifest.json")
    if "GRADIENT-ADAPTATION" in modes and "core.training.supervised-finetuning" not in manifest.get("trainingMethods", []):
        errors.append("GRADIENT-ADAPTATION declared but the manifest lists no supervised-finetuning training method")

    # Release gates and lifecycle truthfulness.
    gates = meta.get("release_gates") or {}
    if not gates:
        errors.append("release_gates must be declared")
    for name, gate in gates.items():
        if not isinstance(gate, dict) or gate.get("status") not in GATE_STATUSES or not gate.get("evidence"):
            errors.append(f"release gate {name!r} needs a status in {GATE_STATUSES} and an evidence string")
    open_gates = sorted(name for name, gate in gates.items() if isinstance(gate, dict) and gate.get("status") not in ("PASS", "NOT_APPLICABLE"))
    if lifecycle == "release" and open_gates:
        errors.append(f"lifecycle_status 'release' is refused while release gates are open: {open_gates}")
    if lifecycle == "scaffold" and pipeline.get("implementation_topology") == "COMPOSED-WORKERS" and meta.get("components"):
        errors.append("a scaffold must not declare bound components; promote to candidate or drop the components")

    # Components against the release documents.
    if pipeline.get("implementation_topology") == "COMPOSED-WORKERS":
        components = meta.get("components") or []
        if not components:
            errors.append("COMPOSED-WORKERS requires at least one component")
        for component in components:
            role = component.get("role")
            release_file = root / str(component.get("release_file", ""))
            if not release_file.is_file():
                errors.append(f"component {role!r}: release_file missing: {component.get('release_file')}")
                continue
            release = load(release_file)
            if component.get("source_revision") != release.get("sourceRevision") or not HEX40.fullmatch(str(component.get("source_revision", ""))):
                errors.append(f"component {role!r}: source_revision does not match {release_file.name} sourceRevision")
            if component.get("worker_id") != release.get("workerId"):
                errors.append(f"component {role!r}: worker_id does not match {release_file.name} workerId")
            declared = component.get("release_digest", "")
            actual = canonical_digest(release)
            manifest_key = f"{role}WorkerReleaseDigest"
            if not DIGEST.fullmatch(declared) or declared != actual:
                errors.append(f"component {role!r}: release_digest {declared!r} does not equal the canonical digest of {release_file.name} ({actual})")
            if manifest.get(manifest_key) != declared:
                errors.append(f"component {role!r}: pipeline-manifest.json {manifest_key} disagrees with release_digest")
            if "image_digest" in component and component["image_digest"] != release.get("imageDigest"):
                errors.append(f"component {role!r}: image_digest disagrees with {release_file.name} imageDigest")
        composition = meta.get("composition") or {}
        report_path = root / str(composition.get("report_file", ""))
        if not report_path.is_file():
            errors.append("composition.report_file missing")
        else:
            report = load(report_path)
            if composition.get("status") != report.get("status"):
                errors.append("composition.status disagrees with release/composition-report.json")
            if composition.get("report_digest") != canonical_digest(report) or pipeline_release.get("compositionReportDigest") != composition.get("report_digest"):
                errors.append("composition.report_digest does not equal the composition report's canonical digest bound in pipeline-release.json")
        contract = meta.get("contract") or {}
        if not HEX40.fullmatch(str(contract.get("revision", ""))):
            errors.append("contract.revision must be an immutable 40-hex commit")
        if contract.get("release_digest") != pipeline_release.get("contractReleaseDigest"):
            errors.append("contract.release_digest disagrees with release/pipeline-release.json contractReleaseDigest")

    # Models against manifest + provenance, with the redistribution gate.
    provenance_by_key = {entry["catalogKey"]: entry for entry in provenance.get("operationalWeights", [])}
    manifest_by_id = {entry["id"]: entry for entry in manifest.get("modelDescriptors", [])}
    seen_keys: set[str] = set()
    for model in meta.get("models") or []:
        key = model.get("catalog_key")
        if key in seen_keys:
            errors.append(f"duplicate model entry {key!r}")
        seen_keys.add(key)
        entry = provenance_by_key.get(key)
        if entry is None:
            errors.append(f"model {key!r}: no provenance entry in provenance/open-weights.json")
            continue
        descriptor = manifest_by_id.get(model.get("model_descriptor_id"))
        if descriptor is None:
            errors.append(f"model {key!r}: model_descriptor_id not in pipeline-manifest.json")
        elif descriptor.get("version") != model.get("revision"):
            errors.append(f"model {key!r}: revision disagrees with pipeline-manifest.json modelDescriptors.version")
        if model.get("id") != entry.get("distributionRepository") or model.get("revision") != entry.get("modelDescriptorVersion"):
            errors.append(f"model {key!r}: id/revision disagree with provenance/open-weights.json")
        if not HEX40.fullmatch(str(model.get("revision", ""))):
            errors.append(f"model {key!r}: revision is not an immutable 40-hex commit")
        files = model.get("files") or []
        weight = next((f for f in files if f.get("path") == entry.get("file")), None)
        if weight is None or weight.get("sha256") != entry.get("sha256") or not HEX64.fullmatch(str(weight.get("sha256", ""))):
            errors.append(f"model {key!r}: weight file sha256 disagrees with provenance/open-weights.json")
        if weight is not None and not isinstance(weight.get("size"), int):
            errors.append(f"model {key!r}: weight file size must be recorded as an integer byte count")
        if model.get("remote_code_required") is not False:
            errors.append(f"model {key!r}: remote_code_required must be false for this pipeline")
        status = model.get("redistribution_status")
        if status not in REDISTRIBUTION:
            errors.append(f"model {key!r}: redistribution_status must be one of {REDISTRIBUTION}")
        elif status in ("unknown", "prohibited"):
            errors.append(f"model {key!r}: redistribution_status {status!r} fails the DIMER hosting gate")
        elif status == "conditional" and not model.get("condition_satisfied"):
            errors.append(f"model {key!r}: conditional redistribution requires a recorded condition_satisfied block")
        if not model.get("license") or not model.get("license_source"):
            errors.append(f"model {key!r}: license and license_source are required (LIC2/LIC17)")
        review = model.get("license_review") or {}
        if not all(review.get(field) for field in ("reviewer", "date", "scope")):
            errors.append(f"model {key!r}: license_review needs reviewer, date and scope")
    unreferenced = sorted(set(provenance_by_key) - seen_keys)
    if unreferenced:
        errors.append(f"provenance models without a metadata entry: {unreferenced}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-file", type=Path, default=ROOT / "pipeline-metadata.json")
    args = parser.parse_args(argv)
    meta = load(args.as_file)
    errors = verify(meta, ROOT)
    if errors:
        for error in errors:
            print(f"FAIL  {error}")
        return 1
    pipeline = meta["pipeline"]
    print(
        f"PASS  pipeline-metadata: lifecycle={pipeline['lifecycle_status']} topology={pipeline['implementation_topology']} "
        f"modes={pipeline['capability_modes']} models={len(meta['models'])} components={len(meta.get('components', []))}; "
        "metadata consistency only, not execution evidence"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
