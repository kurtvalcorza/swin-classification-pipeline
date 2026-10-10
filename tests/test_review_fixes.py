"""Regression tests for the 2026-10-05 notebook review findings (SWC-M1..M3, SWC-m1, SWC-m3).

They need only CI's lightweight dependencies (Pillow, pytest): the carried module's data helpers run on the real
synthetic sample, and the generated notebook is checked statically. No torch, no weights, no GPU; none of this is
model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from swin_classification_pipeline import (
    ValidationFailure,
    evaluation_report,
    generate_synthetic_sample,
    inspect_dataset,
    mean_colour_baseline,
    safe_extract_zip,
    validate_inputs,
)
from swin_classification_pipeline.pipeline import locate_dataset_root

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"swc_fix_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# ---- SWC-M1: isolated runtime, no restart --------------------------------------------------------------------------


def test_swc_M1_no_kernel_install_and_no_restart_instruction(notebook: dict) -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text
    sources = [c["source"] for c in _code(notebook)]
    assert not any("[sys.executable, '-m', 'pip'" in s for s in sources)
    kernel = [s for s in sources if "# dimer: kernel cell" in s]
    assert len(kernel) == 2, "exactly the isolated install and the router run in the kernel"
    install = next(s for s in kernel if "LOCK_TEXT = r'''" in s)
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in "\n".join(kernel)


def test_swc_M1_carried_lock_is_the_committed_hash_lock_of_the_pins() -> None:
    lock = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT, TEMPLATE), lock)
    assert "--only-binary :all:" in lock.splitlines()[1] and "x86_64-manylinux" in lock.splitlines()[1]


# ---- SWC-M2: no leakage, a non-neural baseline, an honest statement -----------------------------------------------


def test_swc_M2_default_sample_has_no_duplicate_content_across_splits(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    origin = generate_synthetic_sample(dataset)
    manifest = validate_inputs(dataset)
    assert manifest["findings"] == [] and origin["distinct_images"] is True
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dataset.rglob("*.png"))]
    assert len(digests) == 24 and len(set(digests)) == 24


def test_swc_M2_mean_colour_baseline_is_reported_and_saturates_the_default_sample(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generate_synthetic_sample(dataset)
    plan = inspect_dataset(dataset)["dataPlan"]
    colour = mean_colour_baseline(dataset, plan["assignments"], ["cool", "warm"])
    assert colour["accuracy"] == 1.0 and colour["n_validation"] == 8 and colour["fittedOn"] == "train"
    report = evaluation_report({"core.metric.classification.accuracy": 1.0}, baseline={"accuracy": 0.5, "majorityClass": "cool"}, colour_baseline=colour)
    assert [b["id"] for b in report["baselines"]] == ["majority_class", "mean_colour_nearest_centroid"]


def test_swc_M2_notebook_states_what_the_default_result_can_and_cannot_show(notebook: dict) -> None:
    md = _markdown(notebook)
    assert "**What the default result can and cannot show:**" in md and "**cannot** show what fine-tuning adds" in md
    code = "\n".join(c["source"] for c in _code(notebook))
    assert "colour_baseline = mean_colour_baseline(DATASET_DIR, data_plan['assignments'], class_names)" in code
    assert "if sample_kind == 'synthetic' and any(f['code'] == 'VISION_DUPLICATE_CONTENT_ACROSS_SPLITS'" in code
    assert "jittered generator can produce" not in md


# ---- SWC-M3: the guided layer --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    ["## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model/System → Output", "## Roadmap", "<strong>Glossary</strong>", "**Predict before running:**", "**What to notice", "<summary>Check your reasoning</summary>", "## 11. Activity: change one thing", "## Troubleshooting", "## Conclusion (your notes)"],
)
def test_swc_M3_guided_layer_marker_is_present(notebook: dict, marker: str) -> None:
    assert marker in _markdown(notebook)


def test_swc_M3_infrastructure_cells_are_collapsed(notebook: dict) -> None:
    code = _code(notebook)
    learner_start = next(i for i, c in enumerate(code) if c["source"].startswith("USE_BYOD = False"))
    for cell in code[:learner_start]:
        assert cell["metadata"].get("cellView") == "form", cell["source"][:60]
    assert _markdown(notebook).count("> **Infrastructure.**") >= 3
    assert _markdown(notebook).count("**Predict before running:**") >= 5


# ---- SWC-m1: the recorded revision carries the module whose digest is recorded -------------------------------------


def test_swc_m1_recorded_revision_contains_the_carried_module(notebook: dict) -> None:
    generated = notebook["metadata"]["dimer"]["generated_from"]
    revision, module = generated["revision"], generated["module"]
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{revision}:{module}"], capture_output=True)
    if shown.returncode:
        pytest.skip(f"revision {revision[:12]} is not in this clone (shallow checkout)")
    assert hashlib.sha256(shown.stdout.replace(b"\r\n", b"\n")).hexdigest() == generated["module_sha256"], "regenerate the notebook from a commit that contains the carried module"


# ---- SWC-m3: macOS archive metadata and refusal messages -----------------------------------------------------------


def _zip_dataset(source: Path, target: Path, extra: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(target, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
        for name, data in extra.items():
            archive.writestr(name, data)
    return target


def test_swc_m3_macos_zip_is_accepted_and_the_skipped_entries_are_reported(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generate_synthetic_sample(dataset, per_split={"train": 2, "val": 1})
    archive = _zip_dataset(dataset, tmp_path / "finder.zip", {"__MACOSX/train/._cool": b"x", "__MACOSX/._train": b"x", "train/cool/.DS_Store": b"x"})
    skipped: list[str] = []
    safe_extract_zip(archive, tmp_path / "out", skipped=skipped)
    assert sorted(skipped) == ["__MACOSX/._train", "__MACOSX/train/._cool", "train/cool/.DS_Store"]
    manifest = validate_inputs(locate_dataset_root(tmp_path / "out"))
    assert manifest["verdict"] == "accepted"


def test_swc_m3_refusal_names_the_offending_entry(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generate_synthetic_sample(dataset, per_split={"train": 2, "val": 1})
    (dataset / "notes.txt").write_text("stray", encoding="utf-8")
    with pytest.raises(ValidationFailure, match=r"Found: \['notes\.txt'\]"):
        validate_inputs(dataset)
