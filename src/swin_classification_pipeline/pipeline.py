"""SwinV2 image classification (DIMER swin-classification pipeline): the tutorial pipeline module.

This module is what the standalone tutorial carries verbatim (NOTEBOOK_SPEC 1.1 §3.6). It holds the
pinned base-model identity and manifest-driven snapshot verification, the in-notebook image-folder
validator (the DIMER handoff documents), the in-kernel supervised fine-tuning loop with content-addressed
artifact publication, the fresh-boundary reload and inference path, and the public ``validate_inputs`` /
``evaluation_report`` stage helpers. Everything was extracted from the repository's Notebook Spec 2.0
tutorial (Gemini, PR #15) without changing its semantics; heavy libraries (``torch``, ``timm``,
``torchvision``, ``safetensors``) are imported lazily inside the functions that need them.
"""
# ruff: noqa: E501  -- finding messages and contract strings are kept on one line

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import unicodedata
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

MODEL_ID = "timm/swinv2_tiny_window8_256.ms_in1k"
MODEL_REVISION = "650d02aabf05e8adbd060a739ab39e39f53da639"
MODEL_LICENSE = "mit"
MODEL_KEY = "swinv2-tiny-window8-256-ms-in1k"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
WEIGHTS_SHA256 = "c47f52b4556ff4436aa9502f5efbc93aac77fdab42d9d70dd845931757ff5d65"
TIMM_MODEL_NAME = "swinv2_tiny_window8_256.ms_in1k"
MODEL_DESCRIPTOR_ID = "org.timm.swinv2-tiny-window8-256-ms-in1k"

# The pipeline's allowlisted base-model catalog (both qualified entries). The standalone tutorial pins
# MODEL_KEY above and carries that snapshot's manifest; the small model is listed for reference and is
# selectable only by regenerating the notebook from a template that pins it.
BASE_MODEL_CATALOG: dict[str, Any] = {
    "schemaVersion": "1.0",
    "catalogVersion": "0.2",
    "entries": {
        "swinv2-tiny-window8-256-ms-in1k": {
            "modelDescriptorId": MODEL_DESCRIPTOR_ID,
            "modelDescriptorVersion": MODEL_REVISION,
            "timmModelName": TIMM_MODEL_NAME,
            "source": {"repoId": MODEL_ID, "revision": MODEL_REVISION, "files": [{"path": WEIGHTS_FILE, "digest": "sha256:" + WEIGHTS_SHA256}]},
            "license": {"spdx": "MIT", "datasetTerms": "ImageNet-1k dataset terms apply to the pretrained weights."},
            "input": {"height": 256, "width": 256},
            "qualification": {"status": "QUALIFIED"},
        },
        "swinv2-small-window8-256-ms-in1k": {
            "modelDescriptorId": "org.timm.swinv2-small-window8-256-ms-in1k",
            "modelDescriptorVersion": "0c9500fcde4c689e97ff51954debae59c158af0d",
            "timmModelName": "swinv2_small_window8_256.ms_in1k",
            "source": {"repoId": "timm/swinv2_small_window8_256.ms_in1k", "revision": "0c9500fcde4c689e97ff51954debae59c158af0d", "files": [{"path": WEIGHTS_FILE, "digest": "sha256:7e793c2f3576d20b5f746ac5dc6b7271745d612f9ce06ccabe382c36f39f1caa"}]},
            "license": {"spdx": "MIT", "datasetTerms": "ImageNet-1k dataset terms apply to the pretrained weights."},
            "input": {"height": 256, "width": 256},
            "qualification": {"status": "QUALIFIED"},
        },
    },
}
CATALOG_ENTRY = BASE_MODEL_CATALOG["entries"][MODEL_KEY]

# Release identity of the pipeline the tutorial mirrors (pipeline-manifest.json at the generating revision): the
# validator's handoff documents record which worker release the in-notebook validation stands in for.
PIPELINE_ID = "org.valcorza.swin-classification"
VALIDATOR_WORKER_RELEASE_DIGEST = "sha256:4085c0b620eed34c186da78f9bdf0baa7524fa6cd0b2e5b36078105c75453aa4"
FINETUNER_WORKER_RELEASE_DIGEST = "sha256:a03808550b69e432a15fb6b9f92bfb0ac8ca49fdc19062be1349eec906a845af"

# Preprocessing constants (the finetuner's contract; mirrored into the artifact's model-config.json).
INPUT_SIZE = 256
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
EXIF_ORIENTATION = "transpose-to-visual-orientation"
EVAL_INTERPOLATION = "bicubic"

# Image-folder validator contract.
TASK = "core.task.vision.image-classification"
REP = "core.dataset.vision.image-folder"
ALG = "org.valcorza.swin-classification-validator.v1"
EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff", ".webp": "image/webp"}
IMAGE_EXTENSIONS = frozenset(EXT)
SPLITS = {"train": "train", "val": "validation", "valid": "validation", "test": "test"}
MAX_EXPANDED_BYTES = 2048 * 1024 * 1024  # BYOD ZIP expansion ceiling (2 GiB) unless the learner raises it
MIN_CLASSES = 2
DECISION_RULE = "argmax"  # argmax(logits); softmax scores are uncalibrated and no threshold is shipped
METRIC_IDS = ("core.metric.classification.accuracy", "org.valcorza.metric.classification.cross-entropy")
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 20260910
EXPECTED_ACCELERATOR = "cuda:0"  # the finetuner is fail-closed: no silent CPU fallback


# ---------------------------------------------------------------------------
# Digests, snapshot verification and staging (MOD6-MOD8, ST3/ST4)
# ---------------------------------------------------------------------------


def sha256_hex(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """``sha256:<hex>`` form used by the DIMER artifact contracts."""
    return "sha256:" + sha256_hex(path)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch. The manifest's
    ``model.safetensors`` digest must equal the catalog's ``WEIGHTS_SHA256`` so the two cannot diverge."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    declared = {entry["path"]: entry["sha256"] for entry in manifest.get("files", [])}
    if declared.get(WEIGHTS_FILE) != WEIGHTS_SHA256:
        raise ValueError(f"manifest {WEIGHTS_FILE} sha256 {declared.get(WEIGHTS_FILE)!r} != catalog WEIGHTS_SHA256 {WEIGHTS_SHA256!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_hex(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; ``verify_snapshot`` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage")
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}")
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


# ---------------------------------------------------------------------------
# Datasets: safe ZIP extraction, the synthetic sample, image decoding
# ---------------------------------------------------------------------------


def safe_extract_zip(zip_path: str | Path, destination: str | Path, max_expanded_bytes: int = MAX_EXPANDED_BYTES) -> int:
    """Extract an image-folder ZIP with the archive-safety rules of DIMER Notebook Spec section 19 (member by
    member, never ``extractall``); returns the expanded byte count."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    expanded = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename
            posix = PurePosixPath(name)
            if "\\" in name:
                raise ValueError(f"backslash-ambiguous archive member rejected: {name!r}")
            if posix.is_absolute() or name.startswith("/") or ".." in posix.parts:
                raise ValueError(f"absolute or traversing archive member rejected: {name!r}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"symlink archive member rejected: {name!r}")
            expanded += info.file_size
            if expanded > max_expanded_bytes:
                raise ValueError(f"archive expands beyond {max_expanded_bytes} bytes; raise MAX_EXPANDED_MIB only if you trust the file")
            target = (root / posix).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member escapes extraction root: {name!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return expanded


def locate_dataset_root(extracted: str | Path) -> Path:
    """Accept either ``train/`` at the top level or exactly one wrapper directory around it."""
    extracted = Path(extracted)
    if (extracted / "train").is_dir():
        return extracted
    children = [p for p in extracted.iterdir() if p.is_dir() and not p.name.startswith(("__MACOSX", "."))]
    if len(children) == 1 and (children[0] / "train").is_dir():
        return children[0]
    raise ValueError("ZIP must contain train/ (and val/ or valid/) at the top level or inside one wrapper folder")


def generate_synthetic_sample(dataset_dir: str | Path, seed: int = DEFAULT_SEED, per_split: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The deterministic two-class tutorial sample (``cool`` squares on blue, ``warm`` circles on red)."""
    dataset_dir = Path(dataset_dir)
    rng = random.Random(seed)
    for split, per_class in (per_split or {"train": 8, "val": 4}).items():
        for class_name in ("cool", "warm"):
            out = dataset_dir / split / class_name
            out.mkdir(parents=True, exist_ok=True)
            for index in range(per_class):
                background = (35, 70, 190) if class_name == "cool" else (190, 65, 35)
                image = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), background)
                draw = ImageDraw.Draw(image)
                jitter = rng.randint(-15, 15)
                if class_name == "cool":
                    draw.rectangle((60 + jitter, 60, 196 + jitter, 196), outline=(230, 240, 255), width=10)
                else:
                    draw.ellipse((60 + jitter, 60, 196 + jitter, 196), outline=(255, 240, 220), width=10)
                image.save(out / f"{class_name}-{index:02d}.png")
    return {"type": "deterministic synthetic tutorial sample", "seed": seed, "generator": "swin_classification_pipeline.generate_synthetic_sample"}


def synthetic_new_image(path: str | Path) -> Path:
    """A freshly drawn ``warm``-style image that is in neither split (the default new-image input)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (190, 65, 35))
    ImageDraw.Draw(image).ellipse((68, 68, 188, 188), outline=(255, 240, 220), width=12)
    image.save(path)
    return path


def load_visual_image(path: str | Path) -> Image.Image:
    """Decode with the declared EXIF semantic: visual orientation."""
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def dataset_inventory(dataset_dir: str | Path) -> dict[str, dict[str, int]]:
    """Per-class, per-split image counts for a layout sanity check before validation."""
    dataset_dir = Path(dataset_dir)
    inventory: dict[str, dict[str, int]] = {}
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            parts = path.relative_to(dataset_dir).parts
            if len(parts) < 2:
                continue
            split, class_name = parts[:2]
            inventory.setdefault(class_name, {}).setdefault(split, 0)
            inventory[class_name][split] += 1
    return inventory


# ---------------------------------------------------------------------------
# In-notebook image-folder validator (DIMER handoff documents)
# ---------------------------------------------------------------------------


class ValidationFailure(RuntimeError):
    def __init__(self, finding: dict) -> None:
        super().__init__(finding["message"])
        self.finding = finding


def cbytes(v: Any) -> bytes:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def dbytes(v: bytes) -> str:
    return "sha256:" + hashlib.sha256(v).hexdigest()


def djson(v: Any) -> str:
    return dbytes(cbytes(v))


def fdigest(p: Path) -> str:
    return sha256_file(p)


def digest_json(document: Any) -> str:
    """Canonical JSON SHA-256 in the ``sha256:<hex>`` form the workers use."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, v: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cbytes(v) + b"\n"
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dbytes(data[:-1])


def finding(code, severity, layer, scope, message, observed, expected, evidence=None) -> dict:
    return {"code": code, "severity": severity, "layer": layer, "scope": scope, "location": None, "message": message, "observed": observed, "expected": expected, "evidence": evidence or {}}


def fail(code, layer, scope, message, observed, expected, **evidence):
    raise ValidationFailure(finding(code, "FATAL", layer, scope, message, observed, expected, evidence))


def safe_rel(root: Path, p: Path) -> str:
    if p.is_symlink():
        fail("VISION_SYMLINK_REJECTED", "L0", "dataset.path", "Symbolic links are forbidden.", str(p), "regular path")
    try:
        return p.resolve(strict=True).relative_to(root).as_posix()
    except ValueError:
        fail("VISION_PATH_ESCAPE", "L0", "dataset.path", "Path resolves outside dataset root.", str(p), str(root))
    return ""  # pragma: no cover - fail() always raises


def root_checked(root: Path) -> Path:
    if not root.exists():
        fail("VISION_DATASET_ROOT_MISSING", "L0", "dataset.root", "Dataset root does not exist.", str(root), "existing directory")
    if root.is_symlink():
        fail("VISION_SYMLINK_REJECTED", "L0", "dataset.root", "Dataset root must not be a symlink.", str(root), "non-symlink directory")
    if not root.is_dir():
        fail("VISION_DATASET_ROOT_NOT_DIRECTORY", "L0", "dataset.root", "Dataset root is not a directory.", str(root), "directory")
    return root.resolve(strict=True)


def source_digest(root: Path) -> str:
    rows = []
    for cur, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs.sort()
        files.sort()
        cp = Path(cur)
        for n in dirs:
            safe_rel(root, cp / n)
        for n in files:
            p = cp / n
            rows.append({"path": safe_rel(root, p), "digest": fdigest(p)})
    return djson({"algorithmId": ALG + ".source-snapshot", "files": rows})


def decode_image_file(p: Path, media: str) -> None:
    try:
        with Image.open(p) as im:
            fmt = im.format
            im.verify()
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            im.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        fail("VISION_IMAGE_DECODE_FAILED", "L1", "dataset.image", "Image decode failed; synthetic replacement pixels are forbidden.", str(p), "decodable image", error=type(e).__name__)
    actual = {"JPEG": "image/jpeg", "PNG": "image/png", "BMP": "image/bmp", "TIFF": "image/tiff", "WEBP": "image/webp"}.get(fmt or "")
    if actual != media:
        fail("VISION_IMAGE_EXTENSION_FORMAT_MISMATCH", "L1", "dataset.image", "Decoded image format contradicts extension.", actual, media, path=str(p))


def inspect_dataset(root: Path) -> dict:
    """The validator's structural + content inspection; raises ``ValidationFailure`` on the first fatal finding."""
    root = root_checked(Path(root))
    src = source_digest(root)
    entries = sorted(root.iterdir(), key=lambda p: p.name)
    unknown = [p.name for p in entries if p.name not in SPLITS]
    if unknown:
        fail("VISION_UNKNOWN_ROOT_ENTRY", "L1", "dataset.structure", "Dataset root contains unsupported entries.", unknown, sorted(SPLITS))
    present = {p.name for p in entries if p.is_dir()}
    if "val" in present and "valid" in present:
        fail("VISION_AMBIGUOUS_VALIDATION_SPLIT", "L1", "dataset.splits", "Both val/ and valid/ exist; no precedence is defined.", ["val", "valid"], "exactly one validation directory")
    if "train" not in present:
        fail("VISION_MISSING_TRAIN_SPLIT", "L2", "dataset.splits", "train/ is required.", sorted(present), "train present")
    if not ({"val", "valid"} & present):
        fail("VISION_MISSING_VALIDATION_SPLIT", "L2", "dataset.splits", "A validation split is required; test/ is never reinterpreted as validation.", sorted(present), "val/ or valid/ present")

    samples, assets, assignments = [], [], []
    classes: set[str] = set()
    class_presence: dict[str, set[str]] = {}
    warnings = []

    for sd in ("train", "val", "valid", "test"):
        sp = root / sd
        if not sp.exists():
            continue
        safe_rel(root, sp)
        logical = SPLITS[sd]
        cdirs = sorted(sp.iterdir(), key=lambda p: p.name)
        if not cdirs:
            fail("VISION_EMPTY_SPLIT", "L1", "dataset.splits", "Split directory is empty.", sd, "one or more class directories")
        seen: dict[str, str] = {}
        for cd in cdirs:
            safe_rel(root, cd)
            if not cd.is_dir():
                fail("VISION_UNASSIGNED_SPLIT_FILE", "L1", "dataset.structure", "Files directly inside split directories are not assignable to a class.", cd.name, "class directory")
            label = cd.name
            key = unicodedata.normalize("NFC", label).casefold()
            if key in seen and seen[key] != label:
                fail("VISION_CLASS_NAME_COLLISION", "L2", "dataset.classes", "Class names collide under normalization/case folding.", [seen[key], label], "distinct normalized labels")
            seen[key] = label
            classes.add(label)
            class_presence.setdefault(label, set()).add(logical)
            ims = sorted(cd.iterdir(), key=lambda p: p.name)
            if not ims:
                fail("VISION_EMPTY_CLASS_DIRECTORY", "L1", "dataset.classes", "Class directory is empty.", cd.as_posix(), "one or more image files")
            for p in ims:
                rel = safe_rel(root, p)
                if p.is_dir():
                    fail("VISION_NESTED_CLASS_DIRECTORY", "L1", "dataset.structure", "Nested directories below a class directory are forbidden.", rel, "image file")
                media = EXT.get(p.suffix.lower())
                if not media:
                    fail("VISION_UNRECOGNIZED_IMAGE_EXTENSION", "L1", "dataset.image", "Unrecognized image extension; files are never silently skipped.", p.suffix.lower(), sorted(EXT), path=rel)
                decode_image_file(p, media)
                dg = fdigest(p)
                samples.append({"sampleId": rel, "assetIds": [rel], "sourceLocator": rel})
                assets.append({"assetId": rel, "digest": dg, "mediaType": media})
                assignments.append({"sampleId": rel, "split": logical, "reason": "directory-mapping"})

    if not samples:
        fail("VISION_NO_SAMPLES", "L1", "dataset.samples", "No image samples were discovered.", 0, ">= 1")

    seen_ids: dict[str, str] = {}
    for sid in [s["sampleId"] for s in samples]:
        k = unicodedata.normalize("NFC", sid).casefold()
        if k in seen_ids and seen_ids[k] != sid:
            fail("VISION_SAMPLE_ID_COLLISION", "L1", "dataset.samples", "Logical sample IDs collide after canonicalization.", [seen_ids[k], sid], "unique canonical sample IDs")
        seen_ids[k] = sid

    amap = {a["sampleId"]: a["split"] for a in assignments}
    dig = {a["assetId"]: a["digest"] for a in assets}
    logical_rows = [
        {"sampleId": s["sampleId"], "split": amap[s["sampleId"]], "className": Path(s["sampleId"]).parts[1], "contentDigest": dig[s["sampleId"]]}
        for s in sorted(samples, key=lambda x: x["sampleId"])
    ]
    logical = djson({"algorithmId": ALG + ".logical-dataset", "representationProfile": REP, "samples": logical_rows})
    cls = sorted(classes)
    semantic = {
        "schemaVersion": "1.0",
        "taskProfile": TASK,
        "fields": [
            {"id": "image", "sourceField": None, "logicalType": "core.type.image", "semanticRole": "core.role.image", "nullable": False},
            {"id": "target.class", "sourceField": None, "logicalType": "core.type.class-label", "semanticRole": "core.role.target.class", "nullable": False},
        ],
        "labelMap": {str(i): n for i, n in enumerate(cls)},
    }
    plan = {"schemaVersion": "1.0", "logicalDatasetDigest": logical, "assignments": sorted(assignments, key=lambda x: x["sampleId"]), "seedPolicy": None}
    manifest = {"schemaVersion": "1.0", "sourceArtifactDigest": src, "representationProfile": REP, "logicalDatasetDigest": logical, "samples": sorted(samples, key=lambda x: x["sampleId"]), "assets": sorted(assets, key=lambda x: x["assetId"])}

    for c in cls:
        missing = sorted({"train", "validation"} - class_presence.get(c, set()))
        if missing:
            warnings.append(finding("VISION_CLASS_ABSENT_FROM_REQUIRED_SPLIT", "WARNING", "L3", "dataset.classes", "A class is absent from one or more required splits.", {"className": c, "missingSplits": missing}, "class represented in train and validation", {"className": c}))

    by_digest: dict[str, list[str]] = {}
    for a in assets:
        by_digest.setdefault(a["digest"], []).append(a["assetId"])
    leaked = []
    for dg, ids in sorted(by_digest.items()):
        splits = sorted({amap[i] for i in ids})
        if len(splits) > 1:
            leaked.append({"contentDigest": dg, "sampleIds": sorted(ids), "splits": splits})
    if leaked:
        warnings.append(finding("VISION_DUPLICATE_CONTENT_ACROSS_SPLITS", "WARNING", "L3", "dataset.splits", "Byte-identical images appear in more than one split; held-out metrics on those samples are not independent evidence.", {"duplicateGroups": len(leaked), "groups": leaked}, "each image content present in at most one split"))

    return {"sourceArtifactDigest": src, "logicalDatasetDigest": logical, "logicalManifest": manifest, "semanticSchema": semantic, "dataPlan": plan, "warnings": warnings, "classNames": cls}


def validate_dataset(root: Path, out: Path, worker_release_digest: str, job_id: str = "tutorial-classification", attempt_id: str = "attempt-1", effective_job_spec_digest: str = "", admission_record_digest: str = "", security_grant_digest: str = "") -> dict:
    """Standalone in-notebook dataset validator producing verified DIMER handoff artifacts."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"algorithmId": ALG, "taskProfile": TASK, "representationProfile": REP, "acceptedImageExtensions": sorted(EXT), "exifOrientation": EXIF_ORIENTATION, "splitMapping": SPLITS, "requireTrain": True, "requireValidation": True, "testAsValidation": False}
    cfgd = djson(cfg)
    ep = {"schemaVersion": "1.0", "jobId": job_id, "attemptId": attempt_id, "role": "validator", "workerReleaseDigest": worker_release_digest, "effectiveJobSpecDigest": effective_job_spec_digest, "resourceBindingDigests": [], "admissionRecordDigest": admission_record_digest, "securityGrantDigest": security_grant_digest}
    epd = atomic_write_json(out / "execution-plan.json", ep)

    try:
        x = inspect_dataset(root)
    except ValidationFailure as e:
        atomic_write_json(out / "validation-findings.json", {"schemaVersion": "1.0", "algorithmId": ALG, "findings": [e.finding]})
        rm = {"schemaVersion": "1.0", "jobId": job_id, "attemptId": attempt_id, "executionPlanDigest": epd, "workerReleaseDigest": worker_release_digest, "outcome": "FAILED", "observed": {"findingCodes": [e.finding["code"]]}, "artifactManifestDigest": None, "evaluationReportDigest": None, "reproducibility": "IDENTIFIED"}
        rmd = atomic_write_json(out / "run-manifest.json", rm)
        atomic_write_json(out / "result.json", {"schemaVersion": "1.0", "jobId": job_id, "attemptId": attempt_id, "state": "FAILED", "failure": {"code": e.finding["code"], "category": "INPUT", "retryable": False, "origin": ALG, "details": {"finding": e.finding}}, "runManifestDigest": rmd, "artifactManifestDigest": None})
        raise

    lmd = atomic_write_json(out / "logical-dataset-manifest.json", x["logicalManifest"])
    ssd = atomic_write_json(out / "semantic-dataset-schema.json", x["semanticSchema"])
    dpd = atomic_write_json(out / "data-plan.json", x["dataPlan"])

    ed, prev = [], []
    for layer, finds in [("L0", []), ("L1", []), ("L2", []), ("L3", x["warnings"]), ("L4", [])]:
        ev = {"schemaVersion": "1.0", "logicalDatasetDigest": x["logicalDatasetDigest"], "layer": layer, "algorithmId": ALG + "." + layer.lower(), "outcome": "FAIL" if any(f["severity"] in {"ERROR", "FATAL"} for f in finds) else "PASS", "findings": finds, "dependencies": list(prev), "cacheability": "REUSABLE"}
        dg = atomic_write_json(out / "evidence" / (layer.lower() + ".json"), ev)
        ed.append(dg)
        prev = [dg]
    if x["warnings"]:
        atomic_write_json(out / "validation-findings.json", {"schemaVersion": "1.0", "algorithmId": ALG, "findings": x["warnings"]})

    vm = {"schemaVersion": "1.0", "logicalDatasetDigest": x["logicalDatasetDigest"], "datasetProfileDigest": None, "semanticSchemaDigest": ssd, "dataPlanDigest": dpd, "validationEvidenceDigests": ed, "validatorWorkerReleaseDigest": worker_release_digest, "effectiveValidationConfigDigest": cfgd, "policySetDigest": None}
    vmd = atomic_write_json(out / "validated-dataset-manifest.json", vm)
    vi = {"schemaVersion": "1.0", "algorithmId": ALG + ".validated-dataset-identity", "validatedDatasetManifestDigest": vmd, "logicalDatasetDigest": x["logicalDatasetDigest"], "effectiveValidationConfigDigest": cfgd, "digest": djson({"manifest": vmd, "logical": x["logicalDatasetDigest"], "config": cfgd, "worker": worker_release_digest, "semantic": ssd, "dataPlan": dpd, "evidence": ed})}
    vid = atomic_write_json(out / "validated-dataset-identity.json", vi)
    rm = {"schemaVersion": "1.0", "jobId": job_id, "attemptId": attempt_id, "executionPlanDigest": epd, "workerReleaseDigest": worker_release_digest, "outcome": "SUCCEEDED", "observed": {"sourceArtifactDigest": x["sourceArtifactDigest"], "logicalDatasetManifestDigest": lmd, "validatedDatasetManifestDigest": vmd, "validatedDatasetIdentityDigest": vid, "sampleCount": len(x["logicalManifest"]["samples"]), "classNames": x["classNames"]}, "artifactManifestDigest": None, "evaluationReportDigest": None, "reproducibility": "REEXECUTABLE"}
    rmd = atomic_write_json(out / "run-manifest.json", rm)
    atomic_write_json(out / "result.json", {"schemaVersion": "1.0", "jobId": job_id, "attemptId": attempt_id, "state": "SUCCEEDED", "failure": None, "runManifestDigest": rmd, "artifactManifestDigest": None})
    return {"validatedDatasetManifestDigest": vmd, "validatedDatasetIdentityDigest": vid, "logicalDatasetDigest": x["logicalDatasetDigest"], "dataPlanDigest": dpd, "semanticDatasetSchemaDigest": ssd, "runManifestDigest": rmd, "classNames": x["classNames"], "warnings": x["warnings"]}


# ---------------------------------------------------------------------------
# In-kernel supervised fine-tuning and the content-addressed artifact
# ---------------------------------------------------------------------------


def build_model_manifest(model_config: dict[str, Any], checkpoint: str = WEIGHTS_FILE) -> dict[str, Any]:
    """Derive model_manifest.json (dimer-inference-service-timm contract) from model-config.json."""
    validation = model_config["transforms"]["validation"]
    resize = next(t for t in validation if t["id"] == "org.torchvision.resize")
    normalize = next(t for t in validation if t["id"] == "org.torchvision.normalize")
    height, width = resize["size"]
    class_names = list(model_config["classNames"])
    return {
        "schema_version": 1,
        "framework": "timm",
        "model": model_config["timmModelName"],
        "num_classes": int(model_config["numClasses"]),
        "class_names": class_names,
        "checkpoint": checkpoint,
        "use_ema": False,
        "preprocessing": {"input_size": [3, int(height), int(width)], "mean": [float(v) for v in normalize["mean"]], "std": [float(v) for v in normalize["std"]], "interpolation": EVAL_INTERPOLATION, "crop_pct": 1.0, "crop_mode": "squash"},
    }


@dataclass(frozen=True)
class ArtifactMemberSource:
    role: str
    path: Path
    media_type: str
    required: bool = True


def publish_artifact_bundle(artifact_root: str | Path, members: tuple[ArtifactMemberSource, ...]) -> tuple[dict[str, Any], Path]:
    """Atomically publish a generation directory (content-addressed by the bundle digest) and the CURRENT pointer."""
    artifact_root = Path(artifact_root)
    generations = artifact_root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    stage = artifact_root / f".staging-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        records = []
        for source in members:
            target = stage / source.path.name
            shutil.copyfile(source.path, target)
            records.append({"role": source.role, "digest": sha256_file(target), "mediaType": source.media_type, "required": source.required, "relationships": [{"type": "org.valcorza.bundle.member-path", "path": source.path.name}]})
        records.sort(key=lambda item: item["role"])
        manifest: dict[str, Any] = {"schemaVersion": "1.0", "members": records}
        bundle_identity = digest_json({"algorithmId": "org.valcorza.swin-classification-finetuner.v1.artifact-bundle", "schemaVersion": manifest["schemaVersion"], "members": manifest["members"], "requiredMemberDigests": sorted(m["digest"] for m in manifest["members"] if m["required"])})
        manifest["bundleDigest"] = bundle_identity
        (stage / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        generation_dir = generations / bundle_identity.split(":", 1)[1]
        if generation_dir.exists():
            shutil.rmtree(stage)
        else:
            os.replace(stage, generation_dir)
        (artifact_root / "CURRENT").write_text(generation_dir.name + "\n", encoding="utf-8")
        return manifest, generation_dir
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def member_path_of(member: Mapping[str, Any]) -> str:
    """Each manifest member names its file through a ``member-path`` relationship."""
    return next(r["path"] for r in member["relationships"] if r["type"] == "org.valcorza.bundle.member-path")


def verify_artifact_generation(generation_dir: str | Path) -> list[dict[str, str]]:
    """Re-verify every artifact member's SHA-256 from ``artifact-manifest.json`` (AINF3); returns the members."""
    generation_dir = Path(generation_dir)
    artifact_manifest = json.loads((generation_dir / "artifact-manifest.json").read_text(encoding="utf-8"))
    verified = []
    for member in artifact_manifest["members"]:
        member_file = generation_dir / member_path_of(member)
        if not member_file.is_file():
            raise RuntimeError(f"artifact member missing after copy: {member_file.name} ({member['role']})")
        if sha256_file(member_file) != member["digest"]:
            raise RuntimeError(f"artifact member digest mismatch: {member_file.name} ({member['role']})")
        verified.append({"role": member["role"], "file": member_file.name, "digest": member["digest"]})
    for required in (WEIGHTS_FILE, "model-config.json"):
        if not (generation_dir / required).is_file():
            raise RuntimeError(f"artifact generation is missing {required}")
    return verified


def make_preprocess(model_config: Mapping[str, Any]):
    """The validation preprocessing the artifact records (resize size, normalisation), cross-checked
    against the module's constants so a foreign artifact cannot silently change them."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    recorded = {step["id"]: step for step in model_config["transforms"]["validation"]}
    resize_size = tuple(recorded["org.torchvision.resize"]["size"])
    normalize = recorded["org.torchvision.normalize"]
    if resize_size != (model_config["input"]["height"], model_config["input"]["width"]):
        raise RuntimeError("artifact resize/input size disagree")
    if tuple(normalize["mean"]) != tuple(NORMALIZATION_MEAN) or tuple(normalize["std"]) != tuple(NORMALIZATION_STD):
        raise RuntimeError("artifact normalisation differs from the module's normalisation constants")
    if model_config["transforms"]["exifOrientation"] != EXIF_ORIENTATION:
        raise RuntimeError(f"artifact EXIF policy {model_config['transforms']['exifOrientation']!r} != {EXIF_ORIENTATION!r}")
    return transforms.Compose([transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True), transforms.ToTensor(), transforms.Normalize(normalize["mean"], normalize["std"])])


def train_model(
    dataset_dir: Path,
    handoff_dir: Path,
    staged_weight_path: Path,
    output_dir: Path,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    device: Any = None,
    model_key: str = MODEL_KEY,
    catalog_entry: Mapping[str, Any] = CATALOG_ENTRY,
) -> dict[str, Any]:
    """In-kernel supervised fine-tuning: replace the SwinV2 head with one output per validated class, train all
    parameters with AdamW + cross-entropy on the frozen ``train`` split, publish a content-addressed artifact
    generation, then free the model and evaluate the frozen ``validation`` split from the persisted bytes."""
    import timm
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, Dataset

    dataset_dir, handoff_dir, staged_weight_path = Path(dataset_dir), Path(handoff_dir), Path(staged_weight_path)
    device = torch.device(device if device is not None else EXPECTED_ACCELERATOR)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logical_manifest = json.loads((handoff_dir / "logical-dataset-manifest.json").read_text(encoding="utf-8"))
    data_plan = json.loads((handoff_dir / "data-plan.json").read_text(encoding="utf-8"))
    semantic_schema = json.loads((handoff_dir / "semantic-dataset-schema.json").read_text(encoding="utf-8"))
    label_map = semantic_schema["labelMap"]
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    class_to_target = {name: i for i, name in enumerate(class_names)}
    assignments = {a["sampleId"]: a["split"] for a in data_plan["assignments"]}

    transforms_meta = {
        "exifOrientation": EXIF_ORIENTATION,
        "train": [{"id": "org.torchvision.resize", "size": [INPUT_SIZE, INPUT_SIZE]}, {"id": "org.torchvision.to-tensor"}, {"id": "org.torchvision.normalize", "mean": list(NORMALIZATION_MEAN), "std": list(NORMALIZATION_STD)}],
        "validation": [{"id": "org.torchvision.resize", "size": [INPUT_SIZE, INPUT_SIZE]}, {"id": "org.torchvision.to-tensor"}, {"id": "org.torchvision.normalize", "mean": list(NORMALIZATION_MEAN), "std": list(NORMALIZATION_STD)}],
        "stochasticAugmentation": False,
    }
    model_config = {"schemaVersion": "1.0", "modelKey": model_key, "timmModelName": catalog_entry["timmModelName"], "numClasses": len(class_names), "classNames": class_names, "input": {"height": INPUT_SIZE, "width": INPUT_SIZE}, "transforms": transforms_meta}
    transform = make_preprocess(model_config)

    class ImageDataset(Dataset):
        def __init__(self, sample_ids):
            self.sample_ids = sample_ids

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, index):
            sid = self.sample_ids[index]
            tensor = transform(load_visual_image(dataset_dir / sid))
            return tensor, class_to_target[Path(sid).parts[1]], sid

    train_ids = [s["sampleId"] for s in logical_manifest["samples"] if assignments[s["sampleId"]] == "train"]
    validation_ids = [s["sampleId"] for s in logical_manifest["samples"] if assignments[s["sampleId"]] == "validation"]
    if not train_ids or not validation_ids:
        raise RuntimeError("train and validation splits must both be non-empty")

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(ImageDataset(train_ids), batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(ImageDataset(validation_ids), batch_size=batch_size, shuffle=False)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = timm.create_model(catalog_entry["timmModelName"], pretrained=False, num_classes=len(class_names))
    staged_sd = load_file(str(staged_weight_path))  # the digest-verified base checkpoint; backbone only, new head
    backbone_sd = {k: v for k, v in staged_sd.items() if not k.startswith("head.fc.")}
    model.load_state_dict(backbone_sd, strict=False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for inputs, targets, _ids in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * targets.numel()
            train_count += targets.numel()
        model.eval()
        val_loss_sum, val_correct, val_count = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets, _ids in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = functional.cross_entropy(outputs, targets)
                val_loss_sum += float(loss.detach().cpu()) * targets.numel()
                val_correct += int((outputs.argmax(dim=1) == targets).sum())
                val_count += targets.numel()
        history.append({"epoch": epoch + 1, "trainLoss": train_loss_sum / train_count, "validationLoss": val_loss_sum / val_count, "validationAccuracy": val_correct / val_count})

    with tempfile.TemporaryDirectory(prefix=".artifact-stage-", dir=output) as stage_dir:
        stage_path = Path(stage_dir)
        weights_path = stage_path / WEIGHTS_FILE
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(weights_path))
        model_config_path = stage_path / "model-config.json"
        model_config_path.write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")
        model_manifest_path = stage_path / "model_manifest.json"
        model_manifest_path.write_text(json.dumps(build_model_manifest(model_config), indent=2) + "\n", encoding="utf-8")
        artifact_manifest, generation = publish_artifact_bundle(output / "artifact", (ArtifactMemberSource("core.artifact.model.weights", weights_path, "application/vnd.safetensors"), ArtifactMemberSource("org.valcorza.swin-classification.model-config", model_config_path, "application/json"), ArtifactMemberSource("org.valcorza.timm.model-manifest", model_manifest_path, "application/json")))

    # Fresh persisted reload verification in-kernel: reported metrics come from the persisted bytes.
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fresh_model = timm.create_model(catalog_entry["timmModelName"], pretrained=False, num_classes=len(class_names)).to(device)
    fresh_model.load_state_dict(load_file(str(generation / WEIGHTS_FILE), device=str(device)), strict=True)
    fresh_model.eval()
    eval_loss_sum, eval_correct, eval_count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets, _ids in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = fresh_model(inputs)
            loss = functional.cross_entropy(outputs, targets)
            eval_loss_sum += float(loss.detach().cpu()) * targets.numel()
            eval_correct += int((outputs.argmax(dim=1) == targets).sum())
            eval_count += targets.numel()
    del fresh_model

    evaluation = {"schemaVersion": "1.0", "artifactManifestDigest": digest_json(artifact_manifest), "metrics": [{"id": METRIC_IDS[0], "version": "1", "value": eval_correct / eval_count}, {"id": METRIC_IDS[1], "version": "1", "value": eval_loss_sum / eval_count}]}
    (output / "evaluation-report.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    run_manifest = {"schemaVersion": "1.0", "outcome": "SUCCEEDED", "observed": {"artifactGeneration": generation.name, "artifactBundleDigest": artifact_manifest["bundleDigest"], "splitCounts": {"train": len(train_ids), "validation": len(validation_ids)}, "classNames": class_names, "history": history, "device": str(device), "timmVersion": timm.__version__, "torchVersion": torch.__version__}, "reproducibility": "REEXECUTABLE"}
    (output / "run-manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps({"schemaVersion": "1.0", "state": "SUCCEEDED"}, indent=2) + "\n", encoding="utf-8")
    return {"state": "SUCCEEDED", "generation": generation.name, "generationDir": str(generation), "artifactBundleDigest": artifact_manifest["bundleDigest"], "metrics": evaluation["metrics"], "runManifest": run_manifest, "evaluationReport": evaluation, "classNames": class_names, "validationSampleIds": validation_ids}


# ---------------------------------------------------------------------------
# Pipeline: snapshot -> fit (in-kernel) -> artifact -> reload -> predict
# ---------------------------------------------------------------------------


@dataclass
class SwinClassificationPipeline:
    """The verified base snapshot plus, after ``fit`` or ``from_artifact``, a reloaded fine-tuned classifier.

    ``from_pretrained`` stages and digest-verifies the pinned snapshot (no model is built yet; the head is
    replaced at fit time). ``fit`` runs ``train_model`` (fail-closed on the expected accelerator) and reloads the
    published generation; ``from_artifact`` reloads an existing generation; ``predict`` applies the artifact's
    recorded preprocessing and the ``argmax`` rule and reports uncalibrated softmax scores in class order.
    """

    weights_path: Path
    device: str = "cpu"
    source: str = "local-snapshot"
    generation_dir: Path | None = None
    model_config: dict[str, Any] | None = None
    class_names: list[str] | None = None
    _model: Any = None
    _preprocess: Any = None

    @classmethod
    def from_pretrained(cls, device: str | None = None, weights_dir: str | Path | None = None, allow_download: bool = False) -> SwinClassificationPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        if device is None:
            import torch

            device = EXPECTED_ACCELERATOR if torch.cuda.is_available() else "cpu"
        return cls(weights_path=root / WEIGHTS_FILE, device=device)

    def fit(self, dataset_dir: str | Path, handoff_dir: str | Path, output_dir: str | Path, *, epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE, learning_rate: float = DEFAULT_LEARNING_RATE, weight_decay: float = DEFAULT_WEIGHT_DECAY, seed: int = DEFAULT_SEED, expected_accelerator: str = EXPECTED_ACCELERATOR) -> dict[str, Any]:
        import torch

        if expected_accelerator != EXPECTED_ACCELERATOR:
            raise RuntimeError(f"Expected accelerator must be {EXPECTED_ACCELERATOR!r}, got {expected_accelerator!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("ACCELERATOR_UNAVAILABLE: No CUDA accelerator observed; silent CPU fallback is forbidden.")
        self.device = expected_accelerator
        result = train_model(Path(dataset_dir), Path(handoff_dir), self.weights_path, Path(output_dir), epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay, seed=seed, device=expected_accelerator)
        self.load_artifact(result["generationDir"])
        self.source = "fine-tuned"
        return result

    @classmethod
    def from_artifact(cls, generation_dir: str | Path, device: str | None = None) -> SwinClassificationPipeline:
        """Rebuild the classifier from a published generation directory alone (fresh-boundary reload)."""
        if device is None:
            import torch

            device = EXPECTED_ACCELERATOR if torch.cuda.is_available() else "cpu"
        pipe = cls(weights_path=Path(generation_dir) / WEIGHTS_FILE, device=device, source="artifact")
        pipe.load_artifact(generation_dir)
        return pipe

    def load_artifact(self, generation_dir: str | Path) -> list[dict[str, str]]:
        import timm
        from safetensors.torch import load_file

        generation_dir = Path(generation_dir)
        verified = verify_artifact_generation(generation_dir)
        model_config = json.loads((generation_dir / "model-config.json").read_text(encoding="utf-8"))
        model = timm.create_model(model_config["timmModelName"], pretrained=False, num_classes=model_config["numClasses"])
        model.load_state_dict(load_file(str(generation_dir / WEIGHTS_FILE)), strict=True)
        self._model = model.to(self.device).eval()
        self._preprocess = make_preprocess(model_config)
        self.model_config = model_config
        self.class_names = list(model_config["classNames"])
        self.generation_dir = generation_dir
        return verified

    def _require_model(self) -> None:
        if self._model is None:
            raise RuntimeError("no fine-tuned classifier is loaded; call fit(...) or from_artifact(...) first")

    def classify(self, image_paths: Sequence[str | Path]):
        """(predicted indices, softmax scores, logits) on CPU for a batch of image paths."""
        import torch
        import torch.nn.functional as functional

        self._require_model()
        batch = torch.stack([self._preprocess(load_visual_image(Path(p))) for p in image_paths]).to(self.device)
        with torch.no_grad():
            logits = self._model(batch)
        return logits.argmax(dim=1).cpu(), functional.softmax(logits, dim=1).cpu(), logits.cpu()

    def predict(self, image_paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
        """Classify images with the ``argmax`` rule; scores are uncalibrated softmax outputs in class order."""
        paths = [Path(image_paths)] if isinstance(image_paths, (str, Path)) else [Path(p) for p in image_paths]
        _check_images(paths)
        predicted, scores, _logits = self.classify(paths)
        assert self.class_names is not None
        return [
            {"input": path.name, "inputSha256": sha256_file(path), "decisionRule": "argmax(logits)", "predictedClass": self.class_names[int(predicted[i])], "uncalibratedSoftmaxScores": {name: float(scores[i][k]) for k, name in enumerate(self.class_names)}, "classOrder": list(self.class_names)}
            for i, path in enumerate(paths)
        ]

    def evaluate_split(self, dataset_dir: str | Path, sample_ids: Sequence[str], batch_size: int = 16) -> dict[str, Any]:
        """Score frozen validation samples with the reloaded artifact: accuracy, cross-entropy, per-sample rows."""
        import torch
        import torch.nn.functional as functional

        self._require_model()
        assert self.class_names is not None
        dataset_dir = Path(dataset_dir)
        per_sample, correct, loss_sum = [], 0, 0.0
        ids_all = list(sample_ids)
        for start in range(0, len(ids_all), batch_size):
            ids = ids_all[start : start + batch_size]
            predicted, scores, logits = self.classify([dataset_dir / sample_id for sample_id in ids])
            targets = torch.tensor([self.class_names.index(Path(sample_id).parts[1]) for sample_id in ids])
            loss_sum += float(functional.cross_entropy(logits, targets, reduction="sum"))
            correct += int((predicted == targets).sum())
            for i, sample_id in enumerate(ids):
                per_sample.append({"sampleId": sample_id, "label": self.class_names[int(targets[i])], "predicted": self.class_names[int(predicted[i])], **{f"score_{name}": float(scores[i][k]) for k, name in enumerate(self.class_names)}})
        return {METRIC_IDS[0]: correct / len(ids_all), METRIC_IDS[1]: loss_sum / len(ids_all), "perSample": per_sample}


# ---------------------------------------------------------------------------
# Role stages (DAT24 / EVAL21)
# ---------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "input": "image-folder dataset root: train/<class>/*, val/ or valid/<class>/* (exactly one), optional test/<class>/*",
    "image_extensions": sorted(EXT),
    "classes": [MIN_CLASSES, None],
    "byod_zip_expanded_bytes": [1, MAX_EXPANDED_BYTES],
    "structure": "only split directories at the root, only class directories in a split, only decodable images in a class directory; symlinks, nested folders, stray files, unknown extensions and undecodable images are fatal",
    "class_names": "directory names; names that collide after NFC normalisation + case folding are refused",
    "splits": "train and validation are required; test is never reinterpreted as validation; duplicate image content across splits is a warning",
    "preprocessing": f"EXIF {EXIF_ORIENTATION}, RGB, resize to {INPUT_SIZE}x{INPUT_SIZE} ({EVAL_INTERPOLATION}), ImageNet normalisation",
    "decision_rule": DECISION_RULE,
}


def _check_images(paths: Sequence[Path]) -> None:
    if not paths:
        raise ValueError("at least one image path is required")
    for path in paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension {path.suffix!r}; accepted: {sorted(IMAGE_EXTENSIONS)}")
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")


def validate_inputs(dataset_root: str | Path, *, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Validation stage: the input manifest for an image-folder dataset root.

    Runs exactly the inspection ``validate_dataset`` runs (``inspect_dataset``): a fatal finding raises
    ``ValidationFailure`` with the same finding document the validator would write; warnings (a class missing
    from a required split, duplicate content across splits) are carried as non-fatal findings. Also enforces
    ``MIN_CLASSES``, which the finetuner needs to train a classifier head."""
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (the dataset's id)")
    dataset_root = Path(dataset_root)
    x = inspect_dataset(dataset_root)
    if len(x["classNames"]) < MIN_CLASSES:
        fail("VISION_TOO_FEW_CLASSES", "L2", "dataset.classes", "A classifier needs at least MIN_CLASSES classes.", len(x["classNames"]), f">= {MIN_CLASSES}")
    split_counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for assignment in x["dataPlan"]["assignments"]:
        split_counts[assignment["split"]] = split_counts.get(assignment["split"], 0) + 1
        class_name = Path(assignment["sampleId"]).parts[1]
        class_counts.setdefault(class_name, {}).setdefault(assignment["split"], 0)
        class_counts[class_name][assignment["split"]] += 1
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "dataset-0", "mode": "fit", "samples": len(x["logicalManifest"]["samples"]), "split_counts": split_counts, "classes": x["classNames"], "class_counts": class_counts, "logicalDatasetDigest": x["logicalDatasetDigest"], "sourceArtifactDigest": x["sourceArtifactDigest"]}],
        "verdict": "accepted",
        "findings": [{"input": names[0] if names else "dataset-0", "verdict": "warning", "code": w["code"], "message": w["message"], "observed": w["observed"]} for w in x["warnings"]],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def majority_class_baseline(labels: Sequence[str], class_names: Sequence[str] | None = None) -> dict[str, Any]:
    """The accuracy of always predicting the most frequent class of the scored split."""
    labels = list(labels)
    if not labels:
        raise ValueError("labels must not be empty")
    names = list(class_names) if class_names is not None else sorted(set(labels))
    counts = {name: labels.count(name) for name in names}
    majority = max(counts, key=lambda name: counts[name])
    return {"majorityClass": majority, "labelCounts": counts, "accuracy": counts[majority] / len(labels)}


def evaluation_report(
    metrics: Mapping[str, float] | None,
    *,
    baseline: Mapping[str, Any] | None = None,
    n_validation: int | None = None,
    class_names: Sequence[str] | None = None,
    sample_kind: str = "synthetic",
    reload_check: Mapping[str, Any] | None = None,
    estimation: str = "single frozen validation holdout (validator-assigned); no dispersion estimate",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``metrics`` maps the repository's metric ids (``core.metric.classification.accuracy``,
    ``org.valcorza.metric.classification.cross-entropy`` — the finetuner's evaluation-report ids) to values on
    the frozen validation split; ``baseline`` is ``majority_class_baseline``; the verdict is ``sample-sanity``.
    Without metrics the verdict is ``not-measurable`` and the report says what labelled data would make the
    task measurable."""
    base: dict[str, Any] = {
        "task": "image classification by supervised fine-tuning of a pinned SwinV2 backbone",
        "decision_rule": DECISION_RULE,
        "score_semantics": "softmax over the fine-tuned head's logits; uncalibrated class scores, no threshold shipped",
        "sample_kind": sample_kind,
        "n_validation": n_validation,
        "class_labels": None if class_names is None else list(class_names),
        "baselines": [],
        "reload_check": dict(reload_check) if reload_check else None,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if metrics is None:
        return {**base, "metrics": [], "verdict": "not-measurable", "reason": "no labelled validation split was scored", "needs": "a labelled image-folder validation split from the deployment domain (every class present), scored with the finetuner's accuracy and cross-entropy against majority_class_baseline; repeated held-out runs for any dispersion estimate"}
    unknown = sorted(set(metrics) - set(METRIC_IDS))
    if unknown:
        raise ValueError(f"unknown metric ids {unknown}; the finetuner reports {list(METRIC_IDS)}")
    entries = [{"id": metric_id, "value": float(metrics[metric_id]), "units": "nats" if metric_id.endswith("cross-entropy") else "unitless", "higher_is_better": not metric_id.endswith("cross-entropy"), "estimation": estimation} for metric_id in METRIC_IDS if metric_id in metrics]
    baselines = [] if baseline is None else [{"id": "majority_class", "metrics": [{"id": METRIC_IDS[0], "value": float(baseline["accuracy"]), "units": "unitless", "higher_is_better": True}], "majorityClass": baseline.get("majorityClass")}]
    rows = "an unstated number of" if n_validation is None else str(n_validation)
    return {**base, "metrics": entries, "baselines": baselines, "verdict": "sample-sanity", "reason": f"{rows} validation image(s) from one frozen holdout; tutorial evidence, not a benchmark", "needs": "a labelled, domain-representative test set with every class present and repeated runs for any generalisable accuracy claim; the softmax scores are uncalibrated"}
