#!/usr/bin/env python3
"""Verify that pipeline model descriptors are bound to immutable open-weight provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REV_RE = re.compile(r"^[0-9a-f]{40}$")


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    manifest = load(ROOT / "pipeline-manifest.json")
    provenance = load(ROOT / "provenance" / "open-weights.json")

    errors: list[str] = []
    if provenance.get("pipelineId") != manifest.get("pipelineId"):
        errors.append("pipelineId mismatch")

    by_id = {
        entry["modelDescriptorId"]: entry
        for entry in provenance.get("operationalWeights", [])
    }

    for descriptor in manifest.get("modelDescriptors", []):
        model_id = descriptor.get("id")
        entry = by_id.get(model_id)
        if entry is None:
            errors.append(f"missing provenance for {model_id}")
            continue
        if entry.get("modelDescriptorVersion") != descriptor.get("version"):
            errors.append(f"version mismatch for {model_id}")
        revision = entry.get("modelDescriptorVersion", "")
        if not REV_RE.fullmatch(revision):
            errors.append(f"non-immutable revision for {model_id}: {revision!r}")
        digest = entry.get("sha256", "")
        if not SHA256_RE.fullmatch(digest):
            errors.append(f"invalid sha256 for {model_id}: {digest!r}")
        if entry.get("status") != "qualified":
            errors.append(f"unqualified provenance entry for {model_id}")

    manifest_ids = {item.get("id") for item in manifest.get("modelDescriptors", [])}
    extra = sorted(set(by_id) - manifest_ids)
    if extra:
        errors.append(f"provenance has unreferenced model descriptors: {extra}")

    policy = provenance.get("policy", {})
    if policy.get("runtimeNetworkFetch") != "DENY":
        errors.append("runtimeNetworkFetch must be DENY")
    if policy.get("offCatalogSelection") != "REFUSE":
        errors.append("offCatalogSelection must be REFUSE")

    if errors:
        for error in errors:
            print(f"FAIL  {error}")
        return 1

    print(f"PASS  {len(manifest_ids)} model descriptors have immutable, digest-pinned provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
