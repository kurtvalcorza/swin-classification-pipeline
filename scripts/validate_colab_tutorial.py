"""Static checks for the Swin classification DIMER Notebook Spec 1.0 tutorial.

This validates source structure only. It must never be cited as REL1/REL5 execution evidence.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "swin_classification_colab.ipynb"
PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME)\b", re.IGNORECASE)


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _text(nb: dict, kind: str) -> str:
    return "\n".join(_source(c) for c in nb["cells"] if c.get("cell_type") == kind)


def main() -> int:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    meta = nb.get("metadata", {}).get("dimer", {})
    assert meta.get("notebook_profile") == "E2E", meta
    # NOTEBOOK_SPEC requires the normative profile declaration but does not prescribe a
    # metadata key for the spec version. Preserve this repository's `notebook_spec` key
    # while accepting the `_version` spelling used by sibling repositories.
    spec_value = meta.get("notebook_spec", meta.get("notebook_spec_version"))
    assert spec_value == "1.0", meta

    code = _text(nb, "code")
    markdown = _text(nb, "markdown")
    all_text = code + "\n" + markdown
    assert not PLACEHOLDER.search(all_text), "unresolved placeholder in tutorial"

    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None, f"cell {index} has execution_count"
        assert not cell.get("outputs"), f"cell {index} persists outputs"
        python_lines = [
            line
            for line in _source(cell).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        ast.parse("\n".join(python_lines))

    required_code = (
        'PIPELINE_REF = "e73a8aee9881093861bbd934b8476626fad23f12"',
        "def resolve_github_token(",
        "def private_git_env(",
        '"GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader"',
        "del PRIVATE_TOKEN, PRIVATE_GIT_ENV",
        "def checkout_pinned(",
        'worker_cli("swin-classification-validate")',
        'worker_cli("swin-classification-train")',
        "hf_hub_download(",
        "hashlib.sha256",
        '"--expected-accelerator", "cuda:0"',
        "torch.cuda.is_available()",
        # BYOD (DAT7-DAT14) and archive safety (SEC1-SEC6)
        "USE_BYOD_DATASET = False  # @param",
        "USE_BYOD_IMAGE = False  # @param",
        "def safe_extract_zip(",
        "symlink archive member rejected",
        "archive member escapes extraction root",
        "MAX_EXPANDED_MIB",
        # fresh-boundary verification (VER1-VER8) using the worker's own preprocessing semantics
        'FRESH_DIR = WORK / "fresh-reload"',
        "from swin_classification_finetuner.trainer import NORMALIZATION_MEAN, NORMALIZATION_STD, load_visual_image",
        "load_file(",
        "strict=True",
        '"accuracyMatches"',
        '"crossEntropyMatches"',
        # baseline computed from the validator's data plan, not hard-coded
        "majority_baseline_accuracy = label_counts[majority_class] / len(validation_labels)",
        # machine-readable exports
        "validation-predictions.csv",
        "metrics.json",
        "prediction.json",
        "provenance.json",
    )
    missing = [marker for marker in required_code if marker not in code]
    assert not missing, f"missing required source markers: {missing}"

    forbidden = (
        "https://x-access-token:",
        "https://${GITHUB_TOKEN}@",
        "print(PRIVATE_TOKEN)",
        "print(GITHUB_TOKEN)",
        "trust_remote_code=True",
        "pickle.load",
        "torch.load(",
        "pretrained=True",
        "extractall(",
    )
    present = [marker for marker in forbidden if marker in code]
    assert not present, f"forbidden/insecure source markers: {present}"

    # Colab form parameters must default to the gated-off sample path (UX4/DAT11).
    for gate in ("USE_BYOD_DATASET", "USE_BYOD_IMAGE"):
        assert f"{gate} = False  # @param" in code, f"{gate} must default to False"

    required_markdown = (
        "**Profile:** `E2E`",
        "Notebook spec:** 1.0",
        "GITHUB_TOKEN",
        "private",
        "By the end of this notebook you will be able to",
        "does not demonstrate",
        "Expected input schema",
        "Do not upload confidential",
        "uncalibrated",
        "argmax",
        "majority-class baseline",
        "single holdout",
        "not bitwise reproducible",
        "Fresh-boundary verification",
        "Interpretation and limits",
        "does **not** prove",
        "Troubleshooting",
        "Next experiments",
    )
    missing_md = [marker for marker in required_markdown if marker not in markdown]
    assert not missing_md, f"missing required learner-facing markers: {missing_md}"

    # Every code cell should be preceded by an explanatory markdown cell (G5).
    cells = nb["cells"]
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            assert index > 0 and cells[index - 1].get("cell_type") == "markdown", f"code cell {index} lacks a preceding explanation"

    print("Static Swin classification Notebook Spec 1.0 checks pass.")
    print("NOTE: this is source validation only, not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
