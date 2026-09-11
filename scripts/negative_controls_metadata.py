#!/usr/bin/env python3
"""Prove that verify_pipeline_metadata.py refuses what it claims to refuse.

Each control copies pipeline-metadata.json into a temporary file, applies one mutation, and asserts
the verifier exits 1 with a message naming the mutated condition. The baseline (unmodified) document
must pass. Nothing in the repository is modified.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify_pipeline_metadata.py"


def run(document: dict) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(document, handle)
        path = handle.name
    try:
        result = subprocess.run([sys.executable, str(VERIFIER), "--as-file", path], capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    base = json.loads((ROOT / "pipeline-metadata.json").read_text(encoding="utf-8"))
    code, out = run(base)
    if code != 0:
        print("FAIL  baseline metadata does not verify:\n" + out)
        return 1
    print("ok    baseline verifies (exit 0)")

    def mutate(label: str, expect: str, fn) -> bool:
        document = copy.deepcopy(base)
        fn(document)
        code, out = run(document)
        good = code == 1 and expect in out
        print(f"{'ok   ' if good else 'FAIL '} {label}: exit {code}, {'names' if expect in out else 'MISSING'} {expect!r}")
        return good

    results = [
        mutate("release lifecycle with open gates", "is refused while release gates are open",
               lambda d: d["pipeline"].__setitem__("lifecycle_status", "release")),
        mutate("redistribution unknown", "fails the DIMER hosting gate",
               lambda d: d["models"][0].__setitem__("redistribution_status", "unknown")),
        mutate("redistribution prohibited", "fails the DIMER hosting gate",
               lambda d: d["models"][1].__setitem__("redistribution_status", "prohibited")),
        mutate("conditional without recorded satisfaction", "requires a recorded condition_satisfied block",
               lambda d: d["models"][0].__setitem__("redistribution_status", "conditional")),
        mutate("tampered weight sha256", "weight file sha256 disagrees",
               lambda d: d["models"][0]["files"][0].__setitem__("sha256", "0" * 64)),
        mutate("tampered component source revision", "source_revision does not match",
               lambda d: d["components"][1].__setitem__("source_revision", "f" * 40)),
        mutate("tampered component release digest", "does not equal the canonical digest",
               lambda d: d["components"][0].__setitem__("release_digest", "sha256:" + "1" * 64)),
        mutate("invalid topology", "implementation_topology must be one of",
               lambda d: d["pipeline"].__setitem__("implementation_topology", "NOTEBOOK")),
        mutate("missing license source", "license and license_source are required",
               lambda d: d["models"][1].pop("license_source")),
        mutate("model dropped (provenance unreferenced)", "provenance models without a metadata entry",
               lambda d: d["models"].pop()),
        mutate("mutable model revision", "revision is not an immutable 40-hex commit",
               lambda d: d["models"][0].__setitem__("revision", "main")),
    ]
    passed = sum(results)
    print(f"{passed}/{len(results)} negative controls discriminated")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
