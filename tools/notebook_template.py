"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier) — E2E.

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline module
(``src/swin_classification_pipeline/pipeline.py``) and the base-model pin/stage/verify cell are produced by the
generator from repository sources so they cannot drift from the package. The in-kernel fine-tuning semantics of
the previous notebook (PR #15) are kept: dataset validation, AdamW fine-tuning with head replacement, artifact
publication, fresh-boundary reload and new-image inference all run inside this kernel through the carried module.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "swin-classification-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/swin_classification_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fswinv2__tiny__window8__256.ms__in1k-ffcc4d?style=flat",
        "https://huggingface.co/timm/swinv2_tiny_window8_256.ms_in1k",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-microsoft%2FSwin--Transformer-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/microsoft/Swin-Transformer",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2111.09883-b31b1b.svg", "https://arxiv.org/abs/2111.09883"),
]

TEMPLATE = {
    "package": "swin_classification_pipeline",
    "repo_name": REPO,
    "stem": "swin_classification",
    "notebook_name": "swin_classification_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # NOTEBOOK_SPEC 2.2 §5 (SWC-M1): the pins are installed into an isolated uv environment and every later cell runs in
    # a persistent worker there, so a hosted runtime's preloaded packages never force a restart. The lock is compiled with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "This notebook needs an NVIDIA GPU exposed as `cuda:0` (Colab: Runtime > Change runtime type > T4 GPU); the finetuner is fail-closed and stops with a clear message otherwise (RUN11). Once that runtime is selected, **Run all** builds an isolated Python environment from the hash-locked pins (the kernel's own packages are left alone, so no restart is needed; a second Run all reuses it), stages and digest-verifies the pinned SwinV2 snapshot, generates the deterministic synthetic two-class sample (24 PNGs, seeded, no download), validates the dataset into an input manifest, performs a bounded fine-tuning run in the notebook (1 epoch, batch 4, AdamW, seed 20260910) with a replaced classification head, exports the artifact and reloads it across a fresh boundary, evaluates the reloaded model on the held-out split against a majority-class and a mean-colour baseline and writes the evaluation report, classifies a new image with the reloaded model, and exports machine-readable results and provenance. No repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5). No hosted run of this revision has been recorded yet; the hosted check is described in `docs/release-verification.md`."
    ),
    "byod": (
        "Two optional branches, both off by default and never part of the default path: `USE_BYOD = True` (or `BYOD_ZIP_PATH`) in Section 4 supplies a ZIP of your own image folders (`train/<class>/…`, `val/`, optional `test/`) that enters the same validation, fine-tuning, export, fresh-reload and evaluation cells as the synthetic sample (DAT14); `USE_BYOD_IMAGE = True` in Section 9 classifies one image of your own with the reloaded model. Expected layout, size ceiling (`MAX_EXPANDED_MIB`) and privacy guidance are stated in the Prerequisites and in those cells; uploads stay inside this runtime."
    ),
    "pipeline_class": "SwinClassificationPipeline",
    "weights_key": "swinv2-tiny-window8-256-ms-in1k",
    "runtime_imports": ["torch", "torchvision", "timm"],
    "title": "SwinV2 Image Classification — DIMER E2E fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "image-folder validation → supervised SwinV2 fine-tuning in this notebook → content-addressed artifact publication → fresh-boundary reload → new-image inference, with the pinned `timm/swinv2_tiny_window8_256.ms_in1k` backbone",
    "intro": (
        "The DIMER Swin classification pipeline fine-tunes an ImageNet-1k SwinV2 backbone on your own image folder. "
        "Everything runs **100 % in this notebook** — in its isolated Python environment, through the carried module: the in-notebook validator freezes the "
        "dataset's logical identity (sample IDs, class labels, split ownership, per-file digests) into DIMER handoff "
        "documents; the finetuner replaces the classifier head with one output per class and trains **all "
        "parameters** with AdamW and cross-entropy — this is gradient adaptation, not in-context conditioning — then "
        "publishes a content-addressed artifact generation and evaluates the *persisted* model, not the in-memory one. "
        "The carried module adds the pinned snapshot scheme, the allowlisted base-model catalog, the archive-safety "
        "rules, the artifact contract and the `validate_inputs` / `majority_class_baseline` / `evaluation_report` "
        "helpers. The default sample is a deterministic two-class synthetic set drawn in code; its metrics are "
        "tutorial sanity evidence, not a benchmark or production claim. `predict` applies `argmax(logits)`; the softmax "
        "scores are **uncalibrated class scores** and the pipeline ships no confidence threshold."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried module guarantees, resolve and digest-verify the immutable "
        "backbone snapshot, generate the synthetic sample or supply an image-folder ZIP through the archive-safety "
        "rules, validate it into an input manifest and the DIMER handoff documents, fine-tune SwinV2 on "
        "`cuda:0` (fail-closed, no silent CPU fallback), read accuracy and cross-entropy against a majority-class and a "
        "mean-colour baseline and say what they can and cannot show, prove the published artifact reloads across a fresh boundary and reproduces the reported metrics, "
        "classify an unseen image under the `argmax` rule, and export machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "object detection, segmentation, multi-label tagging, zero-shot or in-context classification, serving through "
        "`dimer-inference-service-timm`, and any base model other than the pinned `swinv2-tiny-window8-256-ms-in1k` "
        "snapshot (the small variant is listed in the carried catalog for reference only). Fine-tuning runs only on "
        "`cuda:0`; the default synthetic data is tiny by design."
    ),
    "guided_opening": [
        (
            "## How to use this notebook\n\n"
            "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab or Kaggle), run cells in order and read "
            "short Python, and who want to see how a pretrained image classifier is fine-tuned on a small image folder, exported and "
            "checked honestly. No prior experience with Swin Transformers or fine-tuning is assumed: each term is explained where it is "
            "first needed and again in the glossary below. A T4 GPU is required. The **Prerequisites** give the details.\n\n"
            "**Running it.** In Colab choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default path needs no "
            "edit, no upload, no account, no token and no runtime restart. Section 1 builds the isolated environment (the CUDA build of "
            "torch is several GB, so it is the slowest step); a second Run all in the same runtime reuses it. You can also run the "
            "notebook one cell at a time with *Shift + Enter*.\n\n"
            "**Where the code runs.** The first two code cells run in the notebook kernel: they build the environment and start one "
            "Python process inside it. Every later cell is sent to that process, so the pinned `torch`, `torchvision` and `timm` are used "
            "without replacing anything the hosted runtime had already loaded. Printed output, figures and errors come back to the "
            "notebook as usual, and variables persist from cell to cell.\n\n"
            "**Two kinds of cell.** *Infrastructure cells* (Sections 1–3: the isolated install and router, the carried pipeline module "
            "and the pinned-backbone staging) are collapsed and labelled **Infrastructure**; you may run them without studying their "
            "implementation. *Learner cells* (Sections 4–10) are the machine-learning workflow.\n\n"
            "**Form controls.** Some learner cells start with fields Colab renders as a form: `USE_BYOD`, `BYOD_ZIP_PATH`, "
            "`MAX_EXPANDED_MIB` and `SAMPLE_SEED` (Section 4); `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `SEED` and "
            "`EXPECTED_ACCELERATOR_REQUESTED` (Section 6); `USE_BYOD_IMAGE` and `BYOD_IMAGE_PATH` (Section 9). Leave them at their "
            "defaults for the first run: the notes and sample answers describe the default path.\n\n"
            "**Section tags.** Each learner heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation "
            "practice]** — how the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and "
            "packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an "
            "expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives a "
            "worked answer. Write your own answer first, then open it. No hosted run of this revision has been recorded yet, so the "
            "answers describe the shape of a normal result; the one number they quote — the mean-colour baseline — was computed offline "
            "on the default sample, which needs no model."
        ),
        (
            "## The task: Input → Model/System → Output\n\n"
            "| Stage | Input | Model / system | Output |\n"
            "|---|---|---|---|\n"
            "| **Validate** | an image folder: `train/<class>/`, `val/` (or `valid/`), optional `test/` | the carried validator (`validate_inputs`, `validate_dataset`) | an input manifest, a frozen label map and split, and the DIMER handoff documents |\n"
            "| **Fine-tune** | the `train` split, 256 × 256 RGB, ImageNet normalisation | SwinV2-Tiny (window 8, ImageNet-1k weights) with a new classification head; AdamW, cross-entropy, all parameters | a content-addressed artifact: `model.safetensors`, `model-config.json`, `model_manifest.json`, `artifact-manifest.json` |\n"
            "| **Evaluate** | the frozen `validation` split | the artifact reloaded in a fresh location | accuracy and cross-entropy, beside a majority-class and a mean-colour baseline |\n"
            "| **Predict** | one new image | the reloaded artifact, `argmax(logits)` | a class and uncalibrated per-class scores |\n\n"
            "## Roadmap\n\n"
            "| Section | Tag | What happens | What you read |\n"
            "|---|---|---|---|\n"
            "| 1. Install the pinned runtime | [Engineering] | isolated environment built; later cells routed to it | versions, CUDA |\n"
            "| 2. Pipeline code | [Engineering] | the repository's module, carried verbatim | nothing to run by hand |\n"
            "| 3. Pin, stage and verify the model | [Engineering] | backbone downloaded and digest-checked | the verified files |\n"
            "| 4. Prepare the dataset | [Evaluation practice] | the synthetic sample (or your ZIP) | per-class, per-split counts |\n"
            "| 5. Validate | [Evaluation practice] | input manifest, refusal probe, handoff | findings, label map |\n"
            "| 6. Fine-tune | [Concept] | one bounded run on `cuda:0` | the loss history and reported metrics |\n"
            "| 7. Fresh-boundary reload | [Engineering] | the artifact rebuilt from its files | `accuracyMatches`, `crossEntropyMatches` |\n"
            "| 8. Evaluate | [Evaluation practice] | the report with two baselines | the principal result and its limits |\n"
            "| 9. Classify a new image | [Concept] | one unseen image | the predicted class and scores |\n"
            "| 10. Export | [Engineering] | CSV and JSON outputs with provenance | the file list |\n"
            "| 11. Activity (optional) | [Concept] | change `EPOCHS` and compare | your comparison |\n"
            "| Troubleshooting, Glossary, Conclusion | — | recovery, terms, your notes | when needed |\n\n"
            "**Fast path.** Short on time? Run all, then read Sections 5, 8 and 9 and the conclusion."
        ),
        (
            "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
            "| Term | Meaning in this notebook |\n"
            "|---|---|\n"
            "| **Image folder** | A dataset laid out as `split/class/image`; the class name is the folder name. |\n"
            "| **Split** | `train` (the model learns from it), `validation` (held out, scored), optional `test` (never reinterpreted as validation). |\n"
            "| **Logical dataset digest** | A SHA-256 over every sample's id, class, split and file digest: the dataset's identity, frozen before training. |\n"
            "| **Handoff documents** | The validator's JSON records (`logical-dataset-manifest.json`, `data-plan.json`, …) that the finetuner consumes as is, never re-splitting a sample. |\n"
            "| **Data plan** | The record of which sample belongs to which split, and why. |\n"
            "| **Backbone / head** | The pretrained SwinV2 network that turns an image into features, and the final layer that turns features into one score per class; the head is replaced for your classes. |\n"
            "| **Fine-tuning** | Further gradient training of a pretrained network on new labelled data; here every parameter is updated. |\n"
            "| **Cross-entropy** | The mean negative log-probability the model gives the true class; lower is better, and it moves before accuracy does on tiny data. |\n"
            "| **Accuracy** | The share of scored images whose highest-scoring class is the true class. |\n"
            "| **Majority-class baseline** | Always predicting the most frequent class; 0.5 on a balanced two-class split. |\n"
            "| **Mean-colour baseline** | A non-neural rule: average each image's colour, fit one average per class on `train`, and give each validation image the class of the nearest average. |\n"
            "| **Leakage** | The same (or a byte-identical) image in training and validation, so the validation score is partly a memory test. |\n"
            "| **Saturated evaluation** | A test so easy that every reasonable method scores at the ceiling, so it cannot rank them. |\n"
            "| **Content-addressed artifact** | Files named and checked by their SHA-256, so the bytes on disk are exactly the bytes recorded. |\n"
            "| **Fresh boundary** | Reloading the exported files in a new location with nothing kept in memory, as a downstream user would. |\n"
            "| **Uncalibrated scores** | Softmax outputs that rank classes but are not probabilities of being right. |\n"
            "| **Isolated environment** | A separate Python built from hash-locked pins, in which every learner cell runs. |\n"
            "| **BYOD** | Bring Your Own Data: the optional switches that run the same cells on your ZIP or image. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime with an NVIDIA GPU exposed as `cuda:0` (Colab **T4** or better; Kaggle T4 also works). The kernel's own Python version does not matter: Section 1 builds a separate environment with CPython 3.12.12 from the hash-locked pins and every later cell runs there; the finetuner is **fail-closed**: it refuses to run without the expected accelerator instead of silently falling back to CPU. The CUDA build of `torch==2.14.0` is the largest download of the run (several GB of disk; not measured for this notebook).",
        "- **Knowledge:** basic Python, the `train/<class>/` image-folder convention, and what accuracy / cross-entropy mean.",
        "- **Data:** the default path generates a deterministic synthetic two-class sample (24 PNGs) in code and needs no download and no private data; a gated BYOD path accepts a ZIP of your own image folder (`train/<class>/*.png|jpg|…`, `val/` or `valid/`, optional `test/`). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
        "- **Credentials:** none. The pinned backbone is public.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Prepare the dataset: default synthetic sample or bring your own · [Evaluation practice]\n\n"
                "The expected input is the image-folder representation: `<root>/train/<class>/*`, `val/` (or `valid/` — "
                "exactly one) and an optional `test/` that is **never** reinterpreted as validation. Class names are "
                "directory names; two names that collide after Unicode normalisation and case folding (`Cat` and `cat`) "
                "are refused. The dataset root may contain only split directories, split directories only class "
                "directories of images: stray files, nested folders, symlinks, unrecognised extensions and undecodable "
                "images are fatal validator findings, and the refusal names the offending entry. The one exception is operating-system "
                "metadata a ZIP tool adds on its own — `__MACOSX/`, `._*` files, `.DS_Store`, `Thumbs.db` — which is skipped at "
                "extraction and listed under `skipped_archive_metadata`, so a ZIP made with macOS Finder works.\n\n"
                "With `USE_BYOD = False` the module draws the deterministic synthetic sample (`generate_synthetic_sample`: "
                "`cool` squares on blue, `warm` circles on red; 8 training and 4 validation images per class, each drawn at its own offset so "
                "no image appears twice or in two splits). **Read this sample for what it is:** the two classes differ in background "
                "colour as well as shape, so a rule that only looks at colour separates them perfectly, and Section 8 shows that next "
                "to the network. With "
                "`USE_BYOD = True` a ZIP is taken from `BYOD_ZIP_PATH` (an executor places it there) or uploaded, and "
                "extracted member by member by `safe_extract_zip` with the archive-safety rules (no absolute or "
                "traversing paths, no symlinks, `MAX_EXPANDED_MIB` ceiling; never `extractall`). Look for the per-class, "
                "per-split inventory.\n\n"
                "**Predict before running:** how many images will each class have in `train` and in `val`? And on this sample, do you "
                "expect a colour-only rule to tell `cool` from `warm`?"
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "MAX_EXPANDED_MIB = 2048  # @param {{type:\"integer\"}}\n"
                "SAMPLE_SEED = 20260910  # @param {{type:\"integer\"}}\n\n"
                "WORK = Path('work')\n"
                "shutil.rmtree(WORK, ignore_errors=True)\n"
                "WORK.mkdir(parents=True)\n"
                "DATASET_DIR = WORK / 'dataset'\n"
                "if USE_BYOD:\n"
                "    if BYOD_ZIP_PATH:\n"
                "        zip_path = Path(BYOD_ZIP_PATH)\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError('Upload exactly one ZIP file.')\n"
                "        zip_path = WORK / next(iter(uploaded))\n"
                "        zip_path.write_bytes(next(iter(uploaded.values())))\n"
                "    extracted = WORK / 'byod-extracted'\n"
                "    skipped_metadata = []\n"
                "    expanded = safe_extract_zip(zip_path, extracted, MAX_EXPANDED_MIB * 1024 * 1024, skipped=skipped_metadata)\n"
                "    shutil.copytree(locate_dataset_root(extracted), DATASET_DIR)\n"
                "    dataset_origin = {{'type': 'user-supplied image folder (BYOD)', 'zip': zip_path.name, 'expandedBytes': expanded, 'skipped_archive_metadata': skipped_metadata}}\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    dataset_origin = generate_synthetic_sample(DATASET_DIR, seed=SAMPLE_SEED)\n"
                "    sample_kind = 'synthetic'\n"
                "inventory = dataset_inventory(DATASET_DIR)\n"
                "splits = sorted({{s for counts in inventory.values() for s in counts}})\n"
                "print(f\"{{'class':<24}}\" + ''.join(f'{{s:>8}}' for s in splits))\n"
                "for class_name, counts in inventory.items():\n"
                "    print(f'{{class_name:<24}}' + ''.join(f'{{counts.get(s, 0):>8}}' for s in splits))\n"
                "print(json.dumps({{'sample_kind': sample_kind, **dataset_origin}}, indent=2))"
            ),
        },
        {
            "md": (
                "**What to notice (Section 4):** `cool` and `warm` each have 8 `train` and 4 `val` images on the default path, and the "
                "origin record says `distinct_images: true`.\n\n"
                "<details><summary>Check your reasoning</summary>8 and 4 per class, 24 images in all. Yes: `cool` images sit on a blue "
                "background and `warm` images on a red one, so the average colour alone separates them — the shape is a second, redundant "
                "cue. Keep that in mind when you read the accuracy in Section 8.</details>\n\n"
                "## 5. Validate the dataset → input manifest and the DIMER handoff · [Evaluation practice]\n\n"
                "`validate_inputs` is the module's public validation stage: it runs exactly the inspection the "
                "in-notebook validator runs (`inspect_dataset`) — structure, class-name collisions, every image "
                "decoded, extension/format agreement, per-file SHA-256, split ownership — plus the `MIN_CLASSES` floor a "
                "classifier head needs, and returns an **input manifest** naming the schema, the sample and per-class "
                "split counts, the frozen `logicalDatasetDigest` and the verdict; validator *warnings* (a class absent "
                "from a required split, byte-identical images in two splits) are carried as non-fatal findings. On the default "
                "sample there must be none: the cell stops if the synthetic sample ever shows duplicate content across splits. On "
                "your own data a duplicate-content warning means some validation images are also training images, so the validation "
                "score is partly a memory test — remove the duplicates before reading Section 8. It is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a probe "
                "dataset with a stray file at its root and records the validator's own finding.\n\n"
                "`validate_dataset` then writes the complete DIMER handoff — `logical-dataset-manifest.json`, "
                "`data-plan.json`, `semantic-dataset-schema.json`, `validated-dataset-manifest.json`, the evidence "
                "layers and `result.json` — which the finetuner consumes *as is*, verifying every digest and never "
                "re-splitting, dropping or mutating a sample. The ceilings are printed before any model runs.\n\n"
                "**Predict before running:** the probe dataset has a stray `notes.txt` at its root. Which rule will reject it, and will "
                "the message tell you which file to remove?"
            ),
            "code": (
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_CLASSES': MIN_CLASSES, 'MAX_EXPANDED_BYTES': MAX_EXPANDED_BYTES, 'INPUT_SIZE': INPUT_SIZE, 'IMAGE_EXTENSIONS': sorted(IMAGE_EXTENSIONS)}}, 'decision_rule': DECISION_RULE}})\n"
                "input_manifest = validate_inputs(DATASET_DIR, names=[dataset_origin['type']])\n"
                "# Demonstrate rejection on a probe that breaks the layout rules; the finding is recorded, not swallowed.\n"
                "probe_root = WORK / 'probe-dataset'\n"
                "shutil.copytree(DATASET_DIR, probe_root)\n"
                "(probe_root / 'notes.txt').write_text('stray file at the dataset root', encoding='utf-8')\n"
                "try:\n"
                "    validate_inputs(probe_root)\n"
                "except ValidationFailure as exc:\n"
                "    input_manifest['findings'].append({{'input': 'stray-root-file-probe', 'verdict': 'rejected', 'code': exc.finding['code'], 'message': str(exc), 'observed': exc.finding.get('observed')}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest['inputs'][0], indent=2))\n"
                "print('findings:', json.dumps(input_manifest['findings'], indent=2))\n"
                "if sample_kind == 'synthetic' and any(f['code'] == 'VISION_DUPLICATE_CONTENT_ACROSS_SPLITS' for f in input_manifest['findings']):\n"
                "    raise RuntimeError('The synthetic sample has duplicate images across splits; the held-out evaluation would not be independent. Regenerate the notebook from the repository.')\n\n"
                "HANDOFF_DIR = WORK / 'validated'\n"
                "binding = {{'job': digest_json({{'kind': 'tutorial-job', 'task': TASK}}), 'admission': digest_json({{'kind': 'tutorial-admission'}}), 'security': digest_json({{'kind': 'tutorial-security', 'networkDuringRunning': 'DENY'}})}}\n"
                "handoff = validate_dataset(DATASET_DIR, HANDOFF_DIR, worker_release_digest=VALIDATOR_WORKER_RELEASE_DIGEST, job_id='tutorial-classification', attempt_id='attempt-1', effective_job_spec_digest=binding['job'], admission_record_digest=binding['admission'], security_grant_digest=binding['security'])\n"
                "validation_result = json.loads((HANDOFF_DIR / 'result.json').read_text(encoding='utf-8'))\n"
                "if validation_result.get('state') != 'SUCCEEDED':\n"
                "    raise RuntimeError('Dataset validation failed. Fix the dataset layout and rerun Sections 4 and 5.')\n"
                "semantic_schema = json.loads((HANDOFF_DIR / 'semantic-dataset-schema.json').read_text(encoding='utf-8'))\n"
                "data_plan = json.loads((HANDOFF_DIR / 'data-plan.json').read_text(encoding='utf-8'))\n"
                "logical_manifest = json.loads((HANDOFF_DIR / 'logical-dataset-manifest.json').read_text(encoding='utf-8'))\n"
                "label_map = semantic_schema['labelMap']\n"
                "class_names = [label_map[key] for key in sorted(label_map, key=int)]  # index -> class name, frozen by the validator\n"
                "split_counts = input_manifest['inputs'][0]['split_counts']\n"
                "print(json.dumps({{'state': validation_result['state'], 'logicalDatasetDigest': logical_manifest['logicalDatasetDigest'], 'labelMap': label_map, 'splitCounts': split_counts, 'warnings': [w['code'] for w in handoff['warnings']]}}, indent=2))\n"
                "if split_counts.get('validation', 0) == 0:\n"
                "    raise RuntimeError('No validation samples were assigned; the evaluation below would be empty.')"
            ),
        },
        {
            "md": (
                "## 6. Fine-tune SwinV2 in this notebook (fail-closed on `cuda:0`) · [Concept]\n\n"
                "`pipe.fit` is the module's training entry point and runs **inside this notebook's isolated environment**: it requires "
                "`EXPECTED_ACCELERATOR = 'cuda:0'` to match the observed device (**no silent CPU fallback**), loads the "
                "frozen `train` and `validation` splits according to `data-plan.json`, builds the SwinV2 network from the "
                "digest-verified backbone of Section 3 with a fresh classifier head (`num_classes = len(class_names)`), "
                "fine-tunes **all parameters** with AdamW and cross-entropy for `EPOCHS`, publishes a content-addressed "
                "artifact generation (`model.safetensors` + `model-config.json` + `model_manifest.json` — the "
                "`dimer-inference-service-timm` serving manifest — + `artifact-manifest.json` with per-member digests), "
                "then frees the trained model, reloads the persisted artifact and evaluates the frozen `validation` split "
                "with it — the reported metrics come from the persisted bytes, not the in-memory model. **Reproducibility "
                "boundary:** `SEED` drives the loader shuffle and the head initialisation; bitwise-identical results "
                "across GPUs and library builds are not promised. Look for the per-epoch history and the two reported "
                "metrics.\n\n"
                "**Predict before running:** after one epoch on 16 training images, will validation accuracy be near 0.5 (chance), near "
                "1.0, or in between? Will the cross-entropy be close to zero?"
            ),
            "code": (
                "EPOCHS = 1  # @param {{type:\"integer\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "SEED = 20260910  # @param {{type:\"integer\"}}\n"
                "EXPECTED_ACCELERATOR_REQUESTED = 'cuda:0'  # @param {{type:\"string\"}}\n\n"
                "TRAINING_DIR = WORK / 'training-output'\n"
                "if not torch.cuda.is_available():\n"
                "    raise RuntimeError('ACCELERATOR_UNAVAILABLE: No CUDA accelerator observed; silent CPU fallback is forbidden. In Colab choose Runtime > Change runtime type > GPU, then Run all.')\n"
                "training_result = pipe.fit(DATASET_DIR, HANDOFF_DIR, TRAINING_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, expected_accelerator=EXPECTED_ACCELERATOR_REQUESTED)\n"
                "run_manifest = training_result['runManifest']\n"
                "reported_metrics = {{metric['id']: metric['value'] for metric in training_result['metrics']}}\n"
                "if pipe.class_names != class_names:\n"
                "    raise RuntimeError('artifact class order differs from the validator label map')\n"
                "print(json.dumps({{'state': training_result['state'], 'device': run_manifest['observed']['device'], 'splitCounts': run_manifest['observed']['splitCounts'], 'history': run_manifest['observed']['history'], 'reportedMetrics': reported_metrics, 'artifactGeneration': training_result['generation'], 'artifactBundleDigest': training_result['artifactBundleDigest'], 'reproducibility': run_manifest['reproducibility'], 'source': pipe.source}}, indent=2))"
            ),
        },
        {
            "md": (
                "**What to notice (Section 5):** the stray-file probe is `rejected` with `VISION_UNKNOWN_ROOT_ENTRY`, and its message names "
                "`notes.txt`; the default sample has no warnings. **(Section 6):** `device` is `cuda:0`, one history row per epoch, and "
                "`reportedMetrics` with accuracy and cross-entropy.\n\n"
                "<details><summary>Check your reasoning</summary>The root may hold only split directories, so `notes.txt` breaks the layout "
                "rule; the message now ends with `Found: ['notes.txt']`, the entry to remove. For Section 6 there is no recorded hosted "
                "number for this revision, but the shape is predictable: the classes are separable by colour alone, so a pretrained "
                "backbone with a fresh head usually reaches accuracy at or near 1.0 within one epoch, while cross-entropy stays clearly "
                "above zero because one short epoch does not make the head confident. Cross-entropy is the number that still moves with "
                "more training.</details>\n\n"
                "## 7. Fresh-boundary verification of the exported artifact · [Engineering]\n\n"
                "A working in-memory model proves nothing about the bytes on disk. This section reproduces what a "
                "downstream consumer would do with the artifact and nothing else: copy the published generation to a "
                "**fresh location**, re-verify every member's SHA-256 from `artifact-manifest.json` "
                "(`verify_artifact_generation`), rebuild the network from `model-config.json` alone and load "
                "`model.safetensors` with `strict=True` (`SwinClassificationPipeline.from_artifact`), then score the "
                "frozen validation split with the preprocessing the artifact records (`evaluate_split`, cross-checked "
                "against the module's normalisation constants and EXIF policy). Accuracy must match the reported value "
                "exactly and cross-entropy within `1e-4`; a mismatch stops the notebook — do not ship such an artifact."
            ),
            "code": (
                "FRESH_DIR = WORK / 'fresh-reload'\n"
                "shutil.copytree(TRAINING_DIR / 'artifact' / 'generations' / training_result['generation'], FRESH_DIR)\n"
                "fresh = SwinClassificationPipeline.from_artifact(FRESH_DIR, device=pipe.device)\n"
                "verified_members = verify_artifact_generation(FRESH_DIR)\n"
                "print('artifact members verified:', json.dumps(verified_members, indent=2))\n"
                "if fresh.class_names != class_names:\n"
                "    raise RuntimeError('artifact class order differs from the validator label map')\n"
                "validation_sample_ids = sorted(a['sampleId'] for a in data_plan['assignments'] if a['split'] == 'validation')\n"
                "reloaded = fresh.evaluate_split(DATASET_DIR, validation_sample_ids)\n"
                "per_sample = reloaded['perSample']\n"
                "reload_check = {{'reloadedAccuracy': reloaded[METRIC_IDS[0]], 'reportedAccuracy': reported_metrics[METRIC_IDS[0]], 'reloadedCrossEntropy': reloaded[METRIC_IDS[1]], 'reportedCrossEntropy': reported_metrics[METRIC_IDS[1]], 'crossEntropyTolerance': 1e-4}}\n"
                "reload_check['accuracyMatches'] = reload_check['reloadedAccuracy'] == reload_check['reportedAccuracy']\n"
                "reload_check['crossEntropyMatches'] = abs(reload_check['reloadedCrossEntropy'] - reload_check['reportedCrossEntropy']) <= reload_check['crossEntropyTolerance']\n"
                "print(json.dumps(reload_check, indent=2))\n"
                "if not (reload_check['accuracyMatches'] and reload_check['crossEntropyMatches']):\n"
                "    raise RuntimeError('The reloaded artifact does not reproduce the reported metrics. Do not ship this artifact.')\n"
                "print(f'Fresh-boundary verification PASSED: artifact reproduces the reported metrics on {{len(per_sample)}} validation samples.')"
            ),
        },
        {
            "md": (
                "## 8. Evaluate → evaluation report · [Evaluation practice]\n\n"
                "Two metrics are reported, both on the frozen `validation` split as a **single holdout** — no "
                "cross-validation, no repeated runs, therefore no dispersion estimate: "
                "`core.metric.classification.accuracy` (fraction of validation images whose `argmax` class equals the "
                "label; blind to *how* confident the wrong answers were) and "
                "`org.valcorza.metric.classification.cross-entropy` (mean negative log-likelihood of the true class; it "
                "moves before accuracy does on tiny datasets). `majority_class_baseline` is the accuracy of always "
                "predicting the most frequent validation class, computed from the validator's data plan so it is "
                "correct for BYOD datasets too (0.5 on the balanced sample). `evaluation_report` is the module's public "
                "evaluation stage: it carries both metrics under the finetuner's own ids with the verdict "
                "`sample-sanity`, the baseline and the fresh-boundary reload check; without a scored validation split "
                "the verdict is `not-measurable`. It is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "A second, stronger baseline sits beside it: `mean_colour_baseline` averages each image's colour, fits one average per "
                "class on `train` and labels each validation image by the nearest average — no neural network at all. **What the "
                "default result can and cannot show:** on the synthetic sample the mean-colour rule scores **1.0** (computed offline on "
                "the default sample; it needs no model), so a fine-tuned accuracy of 1.0 here shows that the pipeline trains, exports, "
                "reloads and scores correctly, and **cannot** show what fine-tuning adds over a colour rule. These are sanity metrics for "
                "the plumbing, not evidence of image-classification skill. On your own data the comparison becomes informative: a network "
                "that does not beat the mean-colour rule has learned nothing that colour does not already give.\n\n"
                "**Predict before running:** will the fine-tuned model beat the mean-colour baseline on the default sample?"
            ),
            "code": (
                "validation_labels = [Path(sample_id).parts[1] for sample_id in validation_sample_ids]  # class = 2nd path component\n"
                "baseline = majority_class_baseline(validation_labels, class_names)\n"
                "colour_baseline = mean_colour_baseline(DATASET_DIR, data_plan['assignments'], class_names)\n"
                "report = evaluation_report(reported_metrics, baseline=baseline, n_validation=len(validation_labels), class_names=class_names, sample_kind=sample_kind, reload_check=reload_check, colour_baseline=colour_baseline)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{key: report[key] for key in ('verdict', 'reason', 'decision_rule', 'n_validation', 'metrics', 'baselines')}}, indent=2))\n"
                "print(json.dumps({{'mean_colour_baseline': colour_baseline}}, indent=2))\n"
                "if reported_metrics[METRIC_IDS[0]] < baseline['accuracy']:\n"
                "    print('WARNING: the fine-tuned model does not beat the majority baseline on this holdout. More epochs or more data are needed before drawing any conclusion.')\n"
                "if reported_metrics[METRIC_IDS[0]] <= colour_baseline['accuracy']:\n"
                "    print('NOTE: the fine-tuned model does not beat the mean-colour baseline; this holdout cannot show what fine-tuning adds over a colour rule.')"
            ),
        },
        {
            "md": (
                "**What to notice (Section 7):** `accuracyMatches` and `crossEntropyMatches` are both `true`. **(Section 8):** the "
                "verdict `sample-sanity`, two baselines (majority 0.5, mean colour 1.0 on the default sample) and the fine-tuned metrics, "
                "followed by the NOTE when the network does not beat the colour rule.\n\n"
                "<details><summary>Check your reasoning</summary>It cannot beat it: the colour rule is already at 1.0, the ceiling. If the "
                "network also reaches 1.0, the two tie, and the evaluation says nothing about whether fine-tuning learned shape, colour or "
                "both. That is the honest reading of the default run: the plumbing works and the reload reproduces the reported metrics. To "
                "measure what fine-tuning contributes you need data where colour is not enough — your own folder through `USE_BYOD`, with "
                "both baselines printed beside the network.</details>\n\n"
                "## 9. Classify a new image · [Concept]\n\n"
                "Real use means images the model has never seen. The default is a freshly drawn `warm`-style image that is "
                "in neither split (`synthetic_new_image`); set `USE_BYOD_IMAGE = True` to supply one of your own instead "
                "(`BYOD_IMAGE_PATH` for an executor, or the upload dialog; any accepted extension; EXIF orientation is "
                "honoured exactly as during training). `predict` applies the reloaded artifact's recorded preprocessing "
                "and the `argmax(logits)` rule; the softmax scores are **uncalibrated class scores** — larger means the "
                "model favours that class, but 0.9 does not mean \"90 % chance of being right\". The pipeline ships no "
                "confidence threshold; a downstream system that needs one must calibrate it on labelled data from its "
                "deployment domain. Look for `predictedClass` and the per-class scores, which sum to 1.\n\n"
                "**Predict before running:** the default new image is a red-background circle. Which class will it get, and would the "
                "mean-colour rule agree?"
            ),
            "code": (
                "USE_BYOD_IMAGE = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_IMAGE_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "NEW_IMAGE_DIR = WORK / 'new-images'\n"
                "NEW_IMAGE_DIR.mkdir(exist_ok=True)\n"
                "if USE_BYOD_IMAGE:\n"
                "    if BYOD_IMAGE_PATH:\n"
                "        new_image_path = Path(BYOD_IMAGE_PATH)\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError('Upload exactly one image.')\n"
                "        new_image_path = NEW_IMAGE_DIR / next(iter(uploaded))\n"
                "        new_image_path.write_bytes(next(iter(uploaded.values())))\n"
                "    new_image_origin = 'user-supplied image (BYOD)'\n"
                "else:\n"
                "    new_image_path = synthetic_new_image(NEW_IMAGE_DIR / 'synthetic-new-warm.png')\n"
                "    new_image_origin = 'synthetic new image drawn by the carried module (not in train/val)'\n"
                "(prediction,) = fresh.predict([new_image_path])\n"
                "prediction['inputOrigin'] = new_image_origin\n"
                "print(json.dumps(prediction, indent=2))"
            ),
        },
        {
            "md": (
                "**What to notice (Section 9):** `predictedClass` and the two scores.\n\n"
                "<details><summary>Check your reasoning</summary>`warm`, on both counts: the image has the `warm` background colour and "
                "shape, and the mean-colour rule labels it `warm` as well, so agreement here is expected rather than impressive. A score "
                "near 1.0 is a ranking, not a probability of being right; try an image of something that is neither class (Section 11) to "
                "see that it still receives a label.</details>\n\n"
                "## 10. Export machine-readable results and provenance · [Engineering]\n\n"
                "`outputs/{stem}_validation_predictions.csv` holds one row per validation sample (`sampleId`, `label`, "
                "`predicted`, one `score_<class>` column per class in `classOrder`, so class ordering survives downstream "
                "use); `outputs/{stem}_result.json` records the reported metrics, the majority and mean-colour baselines, the evaluation "
                "report, the fresh-boundary reload comparison, the new-image prediction, the input manifest, the dataset "
                "identity (`logicalDatasetDigest`, class names, split counts), the artifact generation and bundle digest "
                "with its verified members, the training configuration, the notebook's source (repository, revision, "
                "module digest, generator), the pinned backbone identity, revision and licence, and the runtime. The "
                "deployable artifact itself is the generation directory under `work/training-output/artifact/generations/`; "
                "`model_manifest.json` inside it is the serving contract `dimer-inference-service-timm` consumes. No "
                "credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "with open('outputs/{stem}_validation_predictions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(per_sample[0].keys()))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(per_sample)\n"
                "payload = {{\n"
                "    'reported_metrics': reported_metrics,\n"
                "    'majority_class_baseline': baseline,\n"
                "    'mean_colour_baseline': colour_baseline,\n"
                "    'evaluation_report': report,\n"
                "    'fresh_boundary_reload': reload_check,\n"
                "    'prediction': prediction,\n"
                "    'input_manifest': input_manifest,\n"
                "    'dataset': {{**dataset_origin, 'sample_kind': sample_kind, 'logicalDatasetDigest': logical_manifest['logicalDatasetDigest'], 'classNames': class_names, 'splitCounts': split_counts, 'validatorWorkerReleaseDigest': VALIDATOR_WORKER_RELEASE_DIGEST, 'handoff': handoff}},\n"
                "    'artifact': {{'generation': training_result['generation'], 'bundleDigest': training_result['artifactBundleDigest'], 'members': verified_members, 'format': 'safetensors + model-config.json + model_manifest.json + artifact-manifest.json', 'finetunerWorkerReleaseDigest': FINETUNER_WORKER_RELEASE_DIGEST}},\n"
                "    'training': {{'method': 'core.training.supervised-finetuning (full network, AdamW, cross-entropy)', 'epochs': EPOCHS, 'batchSize': BATCH_SIZE, 'learningRate': LEARNING_RATE, 'seed': SEED, 'reproducibility': run_manifest['reproducibility'], 'device': run_manifest['observed']['device'], 'history': run_manifest['observed']['history']}},\n"
                "    'base_model': {{'modelKey': MODEL_KEY, 'timmModelName': TIMM_MODEL_NAME, 'modelDescriptorId': MODEL_DESCRIPTOR_ID, 'catalog': CATALOG_ENTRY}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'timm': timm.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## 11. Activity: change one thing — the number of epochs · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**. It changes nothing unless you do it.\n\n"
        "1. **Predict:** with `EPOCHS = 3` instead of 1, what happens to validation accuracy, and what happens to cross-entropy?\n"
        "2. **Change:** in Section 6 set `EPOCHS = 3`. Change nothing else.\n"
        "3. **Run:** select the Section 6 cell and choose *Runtime → Run after*. Section 6 starts again from the verified backbone "
        "with a fresh head and writes a new artifact generation; Sections 7–10 re-run on it. (The outputs of the first run are "
        "replaced; download `outputs/` first if you want to keep them.)\n"
        "4. **Observe:** the history now has three rows; compare `reportedMetrics` with your first run.\n"
        "5. **Explain:** in one sentence, why can cross-entropy change when accuracy cannot?\n\n"
        "<details><summary>Check your reasoning</summary>Accuracy is already at or near its ceiling on this sample, so three epochs "
        "can hardly move it; cross-entropy keeps falling as the head becomes more confident in answers that were already right. "
        "Accuracy only counts which class wins; cross-entropy also measures by how much. Because the mean-colour rule already scores "
        "1.0, neither number tells you what the network learned beyond colour — that needs data where colour is not enough.</details>\n\n"
        "## Interpretation and limits\n\n"
        "The predicted class is `argmax(logits)` over the classes the validator froze from your folder names; the "
        "softmax scores are uncalibrated and the pipeline ships no threshold. The evaluation report's `sample-sanity` "
        "verdict names what it is: one frozen validation holdout with no dispersion estimate — on the synthetic sample "
        "a plumbing check, and even BYOD metrics are a single holdout that must not be generalised to a domain, a "
        "camera or a class distribution. Datasets whose classes are absent from a required split, duplicate images "
        "across splits (the validator warns), class imbalance, images far from ImageNet statistics and more epochs on "
        "tiny data all change results in ways these two metrics do not measure. On the default sample the mean-colour baseline is "
        "already perfect, so the default evaluation is saturated: it checks the plumbing and cannot show what fine-tuning "
        "contributes. Digest equality proves the backbone "
        "bytes are the ones the catalog allowlisted; it does not by itself prove who published them — that trust rests "
        "on the catalog review that produced the pin.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this "
        "notebook, can acquire and digest-verify the pinned backbone, validate the demonstrated image folder into the "
        "DIMER handoff, fine-tune SwinV2 on `cuda:0` in an isolated, hash-locked environment, publish a content-addressed artifact, reload it "
        "across a fresh boundary and reproduce the reported metrics, classify an unseen image, and emit the shown "
        "machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, ImageNet-level or real-domain accuracy, calibration, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a small image folder of your own (two to five classes, a few dozen "
        "images per class, classes that do not differ by colour alone) and compare accuracy against both baselines; classify an image of a class the model never saw and observe that it still "
        "receives a label; hand the generation directory to `dimer-inference-service-timm` through "
        "`model_manifest.json`.\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab or Kaggle; the lock holds manylinux x86_64 wheels. |\n"
        "| Section 1 fails while downloading, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the Section 1 install cell again; a repeated mismatch means the download is being altered — never edit the digest. |\n"
        "| `holds Python …, not 3.12.12` in Section 1 | an older `dimer_isolated_env/` folder from another notebook version | Delete that folder (or start a fresh runtime) and run the install cell again. |\n"
        "| `The isolated environment's Python process exited` | the worker ran out of memory, usually during training | Lower `BATCH_SIZE` in Section 6, then choose *Runtime → Run all* again. |\n"
        "| `ACCELERATOR_UNAVAILABLE: No CUDA accelerator observed` in Section 6 | no GPU is attached | Choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. |\n"
        "| A Hub download fails in Section 3, or `sha256 … != manifest` | a transient Hugging Face failure, or a changed file | Run the Section 3 cell again; a digest mismatch is never loaded — do not edit the manifest. |\n"
        "| `archive expands beyond … bytes` in Section 4 | the ZIP is larger than `MAX_EXPANDED_MIB` | Raise `MAX_EXPANDED_MIB` only if you trust the file, or use a smaller ZIP. |\n"
        "| `ZIP must contain train/ …` | the ZIP has no `train/` at its top level or inside one wrapper folder | Re-create the ZIP from the folder that holds `train/` and `val/`. |\n"
        "| `ValidationFailure: … Found: […]` in Section 5 | a layout rule failed; `Found` names the entry | Remove or fix that entry (a stray file, a nested folder, an unsupported extension, an undecodable image) and re-run Sections 4–5. macOS metadata is skipped automatically. |\n"
        "| `VISION_DUPLICATE_CONTENT_ACROSS_SPLITS` warning on your data | the same image is in `train` and `val` | Remove the duplicates; until then the validation score is partly a memory test. |\n"
        "| `The reloaded artifact does not reproduce the reported metrics` | the artifact files changed, or a nondeterministic kernel | Re-run Sections 6–7; do not ship the artifact if it repeats. |\n"
        "| `Upload exactly one ZIP file.` / `Upload exactly one image.` | the upload dialog was cancelled or several files were chosen | Run the cell again and choose one file, or set `BYOD_ZIP_PATH` / `BYOD_IMAGE_PATH`. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. After ___ epoch(s), validation accuracy was ___ and cross-entropy ___, against a majority baseline of ___ and a mean-colour baseline of ___.\n"
        "2. The reloaded artifact reproduced the reported metrics: ___ (yes/no).\n"
        "3. What this result can show about fine-tuning on this sample, and what it cannot: ___.\n"
        "4. The new image was classified as ___ with a score of ___, which means ___ (and does not mean ___).\n"
        "5. What I would need before trusting this classifier on my own images: ___.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/microsoft/Swin-Transformer\n"
        "- timm: https://github.com/huggingface/pytorch-image-models\n"
        "- Swin Transformer V2: https://arxiv.org/abs/2111.09883"
    ),
}
