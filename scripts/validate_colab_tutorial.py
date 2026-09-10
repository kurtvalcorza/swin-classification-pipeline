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
        'PIPELINE_REF="f68176e95577d6106538a4d8f8ea592a6d5ca848"',
        "private_github_token(",
        "private_git_env(",
        'GIT_CONFIG_KEY_0":"http.https://github.com/.extraHeader"',
        "del PRIVATE_TOKEN, PRIVATE_GIT_ENV",
        "swin-classification-validate",
        "swin-classification-train",
        "hf_hub_download(",
        "hashlib.sha256",
        '"--expected-accelerator","cuda:0"',
        "torch.cuda.is_available()",
        "load_file(",
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
    )
    present = [marker for marker in forbidden if marker in code]
    assert not present, f"forbidden/insecure source markers: {present}"

    required_markdown = (
        "**Profile:** `E2E`",
        "Notebook spec:** 1.0",
        "GITHUB_TOKEN",
        "private",
        "DAT7",
        "uncalibrated",
        "Interpretation and limits",
        "does **not** prove",
    )
    missing_md = [marker for marker in required_markdown if marker not in markdown]
    assert not missing_md, f"missing required learner-facing markers: {missing_md}"

    print("Static Swin classification Notebook Spec 1.0 checks pass.")
    print("NOTE: this is source validation only, not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
