"""Offline tests for the carried pipeline module: snapshot helpers (injected downloader), the in-notebook
image-folder validator on a temporary synthetic dataset, the artifact-bundle publication/verification, and the
public validation / evaluation stage helpers (DAT24 / EVAL21). No weights, no torch, no timm."""
# ruff: noqa: E501  -- single-line fixtures and assertion messages

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from swin_classification_pipeline import (
    BASE_MODEL_CATALOG,
    DECISION_RULE,
    DEFAULT_WEIGHTS_DIR,
    INPUT_SCHEMA,
    METRIC_IDS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    WEIGHTS_SHA256,
    ValidationFailure,
    evaluation_report,
    generate_synthetic_sample,
    inspect_dataset,
    majority_class_baseline,
    safe_extract_zip,
    stage_missing_files,
    validate_dataset,
    validate_inputs,
    verify_artifact_generation,
    verify_snapshot,
)
from swin_classification_pipeline import pipeline as api

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PAYLOAD = b"not-the-real-safetensors"
CONFIG_PAYLOAD = b'{"architecture": "swinv2_tiny_window8_256"}'
WDIGEST = hashlib.sha256(WEIGHTS_PAYLOAD).hexdigest()
CDIGEST = hashlib.sha256(CONFIG_PAYLOAD).hexdigest()


def _manifest(wsha: str = WDIGEST) -> dict:
    return {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": api.CONFIG_FILE, "bytes": len(CONFIG_PAYLOAD), "sha256": CDIGEST},
            {"path": api.WEIGHTS_FILE, "bytes": len(WEIGHTS_PAYLOAD), "sha256": wsha},
        ],
        "totalBytes": len(WEIGHTS_PAYLOAD) + len(CONFIG_PAYLOAD),
    }


def _snapshot(tmp_path: Path, *, manifest: dict | None = None, write_files: bool = True) -> Path:
    root = tmp_path / "weights" / MODEL_KEY
    root.mkdir(parents=True)
    (root / api.MANIFEST_NAME).write_text(json.dumps(manifest or _manifest()), encoding="utf-8")
    if write_files:
        (root / api.WEIGHTS_FILE).write_bytes(WEIGHTS_PAYLOAD)
        (root / api.CONFIG_FILE).write_bytes(CONFIG_PAYLOAD)
    return root


@pytest.fixture
def constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "WEIGHTS_SHA256", WDIGEST)


def test_identity_constants_and_catalog_agree() -> None:
    entry = BASE_MODEL_CATALOG["entries"][MODEL_KEY]
    assert (entry["source"]["repoId"], entry["source"]["revision"]) == (MODEL_ID, MODEL_REVISION)
    assert entry["source"]["files"][0]["digest"] == "sha256:" + WEIGHTS_SHA256
    assert len(MODEL_REVISION) == 40 and all(c in "0123456789abcdef" for c in MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == ROOT / "weights" / MODEL_KEY


def test_committed_manifest_matches_the_catalog_digest() -> None:
    manifest_path = DEFAULT_WEIGHTS_DIR / api.MANIFEST_NAME
    if not manifest_path.is_file():
        pytest.skip("snapshot manifest is not staged in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"]) == (MODEL_ID, MODEL_REVISION)
    digests = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    assert digests[api.WEIGHTS_FILE] == WEIGHTS_SHA256


def test_verify_snapshot_and_rejections(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    assert [f["path"] for f in verify_snapshot(root)["files"]] == [api.CONFIG_FILE, api.WEIGHTS_FILE]
    (root / api.WEIGHTS_FILE).write_bytes(WEIGHTS_PAYLOAD[:-1] + b"!")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)
    with pytest.raises(ValueError, match="catalog WEIGHTS_SHA256"):
        verify_snapshot(_snapshot(tmp_path / "bad", manifest=_manifest(wsha="0" * 64)))
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(_snapshot(tmp_path / "foreign", manifest={**_manifest(), "modelId": "x/y"}))
    with pytest.raises(FileNotFoundError, match="snapshot file missing"):
        verify_snapshot(_snapshot(tmp_path / "missing", write_files=False))


def test_stage_missing_files_uses_injected_downloader(tmp_path: Path, constants: None) -> None:
    assert stage_missing_files(_snapshot(tmp_path)) == []
    root = _snapshot(tmp_path / "empty", write_files=False)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls: list[str] = []

    def downloader(relative_path: str, destination: Path) -> None:
        calls.append(relative_path)
        (destination / relative_path).write_bytes(WEIGHTS_PAYLOAD if relative_path == api.WEIGHTS_FILE else CONFIG_PAYLOAD)

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [api.CONFIG_FILE, api.WEIGHTS_FILE]
    assert calls == [api.CONFIG_FILE, api.WEIGHTS_FILE]
    verify_snapshot(root)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(_snapshot(tmp_path / "rev", manifest={**_manifest(), "revision": "a" * 40}, write_files=False), allow_download=True, downloader=lambda *a: None)


def test_from_pretrained_verifies_without_building_a_model(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    pipe = api.SwinClassificationPipeline.from_pretrained(weights_dir=root, device="cpu")
    assert pipe.weights_path == root / api.WEIGHTS_FILE and pipe.source == "local-snapshot" and pipe.device == "cpu"
    with pytest.raises(RuntimeError, match="no fine-tuned classifier is loaded"):
        pipe.classify([root / api.CONFIG_FILE])
    with pytest.raises(FileNotFoundError):
        api.SwinClassificationPipeline.from_pretrained(weights_dir=_snapshot(tmp_path / "u", write_files=False), device="cpu")


def test_synthetic_sample_validates_into_an_input_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    origin = generate_synthetic_sample(dataset, seed=1, per_split={"train": 3, "val": 2})
    assert origin["seed"] == 1 and sorted(p.name for p in dataset.iterdir()) == ["train", "val"]
    manifest = validate_inputs(dataset, names=["sample"])
    assert manifest["verdict"] == "accepted"
    # the jittered synthetic generator may draw byte-identical images in two splits: a WARNING, never fatal
    assert all(f["verdict"] == "warning" for f in manifest["findings"])
    assert manifest["schema"] == INPUT_SCHEMA and manifest["schema"]["decision_rule"] == DECISION_RULE == "argmax"
    (entry,) = manifest["inputs"]
    assert entry["id"] == "sample" and entry["samples"] == 10
    assert entry["split_counts"] == {"train": 6, "validation": 4} and entry["classes"] == ["cool", "warm"]
    assert entry["class_counts"]["warm"] == {"train": 3, "validation": 2}
    assert entry["logicalDatasetDigest"].startswith("sha256:")
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert inspect_dataset(dataset)["logicalDatasetDigest"] == entry["logicalDatasetDigest"]
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(dataset, names=["a", "b"])


def test_validate_inputs_raises_the_validator_findings(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generate_synthetic_sample(dataset, per_split={"train": 2, "val": 1})
    (dataset / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValidationFailure) as info:
        validate_inputs(dataset)
    assert info.value.finding["code"] == "VISION_UNKNOWN_ROOT_ENTRY"
    (dataset / "stray.txt").unlink()
    (dataset / "train" / "cool" / "broken.png").write_bytes(b"not an image")
    with pytest.raises(ValidationFailure) as info:
        validate_inputs(dataset)
    assert info.value.finding["code"] == "VISION_IMAGE_DECODE_FAILED"
    (dataset / "train" / "cool" / "broken.png").unlink()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(dataset / "train" / "cool" / "renamed.jpg", format="PNG")
    with pytest.raises(ValidationFailure) as info:
        validate_inputs(dataset)
    assert info.value.finding["code"] == "VISION_IMAGE_EXTENSION_FORMAT_MISMATCH"
    (dataset / "train" / "cool" / "renamed.jpg").unlink()
    single = tmp_path / "single"
    (single / "train" / "only").mkdir(parents=True)
    (single / "val" / "only").mkdir(parents=True)
    for split in ("train", "val"):
        Image.new("RGB", (8, 8)).save(single / split / "only" / "a.png")
    with pytest.raises(ValidationFailure) as info:
        validate_inputs(single)
    assert info.value.finding["code"] == "VISION_TOO_FEW_CLASSES"
    no_val = tmp_path / "noval"
    (no_val / "train" / "a").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(no_val / "train" / "a" / "a.png")
    with pytest.raises(ValidationFailure) as info:
        inspect_dataset(no_val)
    assert info.value.finding["code"] == "VISION_MISSING_VALIDATION_SPLIT"


def test_validate_dataset_writes_the_handoff_and_reports_warnings(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    generate_synthetic_sample(dataset, per_split={"train": 2, "val": 1})
    # duplicate content across splits -> WARNING finding, not fatal
    (dataset / "val" / "cool" / "dup.png").write_bytes((dataset / "train" / "cool" / "cool-00.png").read_bytes())
    out = tmp_path / "validated"
    result = validate_dataset(dataset, out, worker_release_digest="sha256:" + "0" * 64)
    assert (out / "result.json").is_file() and json.loads((out / "result.json").read_text())["state"] == "SUCCEEDED"
    for name in ("logical-dataset-manifest.json", "data-plan.json", "semantic-dataset-schema.json", "validated-dataset-manifest.json", "run-manifest.json"):
        assert (out / name).is_file(), name
    assert result["classNames"] == ["cool", "warm"]
    assert [w["code"] for w in result["warnings"]] == ["VISION_DUPLICATE_CONTENT_ACROSS_SPLITS"]
    manifest = validate_inputs(dataset)
    assert [f["code"] for f in manifest["findings"]] == ["VISION_DUPLICATE_CONTENT_ACROSS_SPLITS"]
    with pytest.raises(ValidationFailure):
        validate_dataset(tmp_path / "does-not-exist", tmp_path / "out2", worker_release_digest="sha256:" + "0" * 64)
    assert json.loads((tmp_path / "out2" / "result.json").read_text())["state"] == "FAILED"


def test_safe_extract_zip_and_dataset_root(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("wrapper/train/a/x.png", b"x")
        zf.writestr("wrapper/val/a/y.png", b"y")
    expanded = safe_extract_zip(archive, tmp_path / "out")
    assert expanded == 2 and api.locate_dataset_root(tmp_path / "out") == tmp_path / "out" / "wrapper"
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", b"x")
    with pytest.raises(ValueError, match="traversing"):
        safe_extract_zip(evil, tmp_path / "evil-out")
    with pytest.raises(ValueError, match="expands beyond"):
        safe_extract_zip(archive, tmp_path / "small", max_expanded_bytes=1)


def test_artifact_bundle_publication_and_verification(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "model.safetensors").write_bytes(b"weights")
    config = {"schemaVersion": "1.0", "modelKey": MODEL_KEY, "timmModelName": api.TIMM_MODEL_NAME, "numClasses": 2, "classNames": ["cool", "warm"], "input": {"height": 256, "width": 256}, "transforms": {"exifOrientation": api.EXIF_ORIENTATION, "train": [], "validation": [{"id": "org.torchvision.resize", "size": [256, 256]}, {"id": "org.torchvision.to-tensor"}, {"id": "org.torchvision.normalize", "mean": list(api.NORMALIZATION_MEAN), "std": list(api.NORMALIZATION_STD)}], "stochasticAugmentation": False}}
    (stage / "model-config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest_doc = api.build_model_manifest(config)
    assert manifest_doc["class_names"] == ["cool", "warm"] and manifest_doc["preprocessing"]["crop_mode"] == "squash"
    (stage / "model_manifest.json").write_text(json.dumps(manifest_doc), encoding="utf-8")
    manifest, generation = api.publish_artifact_bundle(tmp_path / "artifact", (api.ArtifactMemberSource("core.artifact.model.weights", stage / "model.safetensors", "application/vnd.safetensors"), api.ArtifactMemberSource("org.valcorza.swin-classification.model-config", stage / "model-config.json", "application/json"), api.ArtifactMemberSource("org.valcorza.timm.model-manifest", stage / "model_manifest.json", "application/json")))
    assert generation.name == manifest["bundleDigest"].split(":", 1)[1]
    assert (tmp_path / "artifact" / "CURRENT").read_text().strip() == generation.name
    verified = verify_artifact_generation(generation)
    assert [m["role"] for m in verified] == ["core.artifact.model.weights", "org.valcorza.swin-classification.model-config", "org.valcorza.timm.model-manifest"]
    (generation / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        verify_artifact_generation(generation)


def test_majority_baseline_and_evaluation_report() -> None:
    baseline = majority_class_baseline(["cool", "warm", "warm"], ["cool", "warm"])
    assert baseline == {"majorityClass": "warm", "labelCounts": {"cool": 1, "warm": 2}, "accuracy": pytest.approx(2 / 3)}
    with pytest.raises(ValueError, match="must not be empty"):
        majority_class_baseline([])
    metrics = {METRIC_IDS[0]: 1.0, METRIC_IDS[1]: 0.25}
    report = evaluation_report(metrics, baseline=baseline, n_validation=3, class_names=["cool", "warm"], sample_kind="synthetic", reload_check={"accuracyMatches": True})
    assert report["verdict"] == "sample-sanity" and report["decision_rule"] == "argmax"
    assert [m["id"] for m in report["metrics"]] == list(METRIC_IDS)
    assert report["metrics"][1]["higher_is_better"] is False and report["metrics"][1]["units"] == "nats"
    assert report["baselines"][0]["id"] == "majority_class" and report["baselines"][0]["majorityClass"] == "warm"
    assert report["reload_check"] == {"accuracyMatches": True} and report["class_labels"] == ["cool", "warm"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    missing = evaluation_report(None, sample_kind="BYOD")
    assert missing["verdict"] == "not-measurable" and "majority_class_baseline" in missing["needs"]
    with pytest.raises(ValueError, match="unknown metric ids"):
        evaluation_report({"accuracy": 1.0})
