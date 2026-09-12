---
license: mit
model_card_spec: "1.1"
pipeline_spec: "1.0"
base_model:
  - timm/swinv2_tiny_window8_256.ms_in1k
  - timm/swinv2_small_window8_256.ms_in1k
base_model_revision:
  - 650d02aabf05e8adbd060a739ab39e39f53da639
  - 0c9500fcde4c689e97ff51954debae59c158af0d
base_model_weights_license: MIT (timm pretrained-config licence field); ImageNet-1k dataset terms apply to the pretraining data
pipeline_id: org.valcorza.swin-classification
pipeline_version: "0.1.0"
lifecycle_status: candidate
implementation_topology: COMPOSED-WORKERS
capability_modes:
  - GRADIENT-ADAPTATION
task_profile: core.task.vision.image-classification
---

# SwinV2 Tiny/Small (org.valcorza.swin-classification 0.1.0) — Image Classification (Fine-Tuning & Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fswinv2__tiny__window8__256.ms__in1k-ffcc4d?style=flat)](https://huggingface.co/timm/swinv2_tiny_window8_256.ms_in1k)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fswinv2__small__window8__256.ms__in1k-ffcc4d?style=flat)](https://huggingface.co/timm/swinv2_small_window8_256.ms_in1k)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-microsoft%2FSwin--Transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/Swin-Transformer)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2111.09883-b31b1b.svg)](https://arxiv.org/abs/2111.09883)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pipeline](https://img.shields.io/badge/Pipeline-swin--classification--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/swin-classification-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook demonstrating the full composed-worker path — image-folder validation, supervised SwinV2 fine-tuning, content-addressed artifact publication, fresh-boundary reload, and new-image inference:

- **End-to-End Pipeline Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin-classification-pipeline/blob/main/tutorials/swin_classification_colab.ipynb) [`swin_classification_colab.ipynb`](https://github.com/kurtvalcorza/swin-classification-pipeline/blob/main/tutorials/swin_classification_colab.ipynb)  
  *Validate an image folder, fine-tune SwinV2 Tiny/Small from the pinned ImageNet-1k weights through the two pinned workers, publish a content-addressed artifact, reload it across a fresh boundary, and classify new images.*

> [!NOTE]
> Needs an NVIDIA GPU exposed as `cuda:0` (Colab Tesla T4 or better; Kaggle T4 also works); training one epoch on the 16-image sample takes seconds on a T4-class GPU.

---

###### Description

This pipeline packages **SwinV2 Tiny** and **SwinV2 Small** image-classification checkpoints from the Microsoft Swin Transformer V2 family — *Swin Transformer V2: Scaling Up Capacity and Resolution* (Liu et al., 2022, arXiv:2111.09883) — in the `timm` distribution `swinv2_tiny_window8_256.ms_in1k` (Hugging Face revision `650d02aabf05…`) and `swinv2_small_window8_256.ms_in1k` (revision `0c9500fcde4c…`), both pretrained on ImageNet-1k at 256×256 with an 8×8 attention window. A Swin transformer is a hierarchical vision transformer: the image is cut into 4×4 patches, self-attention is computed inside local windows that shift between layers, and patches are merged stage by stage so the network builds a multi-scale feature map ending in one feature vector per image.

At inference time the fine-tuned network maps one 256×256 RGB image to one logit per user class; the decision rule is `argmax`. **Adaptation is full-network gradient fine-tuning** (AdamW, cross-entropy) on the user's own labelled image folder; the ImageNet head is discarded and replaced with a head sized to the user's classes. There is no zero-shot or in-context use: without fine-tuning the pipeline produces nothing.

The upstream weights supply the pretrained backbone. This repository adds the DIMER composition: a digest-pinned allowlist (catalog) of exactly two base checkpoints, a dataset **validator** worker that freezes the image folder's identity, a **finetuner** worker that trains, publishes a content-addressed `safetensors` artifact, reloads it and evaluates it, contract validation binding both workers to the `ml-worker` contract at an immutable revision, and fail-closed verification scripts. The model architecture itself is used through `timm` unchanged.

#### Intended Use and Limitations

The uses envisioned during development are narrow: supervised adaptation of a general-purpose ImageNet backbone to a user's own single-label image classes inside DIMER.

###### Primary Intended Uses

**Task:** single-label image classification by supervised fine-tuning. Input is an image-folder dataset — `train/<class>/` and `val/<class>/` directories of JPEG, PNG, BMP, TIFF or WebP files — plus a choice of base checkpoint. Output is a deployable artifact (`model.safetensors` + `model-config.json` + `artifact-manifest.json`) that maps a 256×256 RGB image to one logit per class, an evaluation report on the frozen validation split, and provenance manifests.

**Application domains envisioned:** small-to-medium image-category problems where a few hundred to a few thousand labelled photographs per class exist and a strong pretrained backbone is more useful than training from scratch — product or part recognition, document-type sorting, scene or land-cover tiles at 256×256, quality-control pass/fail imagery. It was built as the first NATIVE `ml-worker` vision pipeline for DIMER, so its structural role is a **reference fine-tuning pipeline**: the validator/finetuner split, the pinned catalog and the artifact contract are meant to be reused by the sibling segmentation and detection pipelines.

**Role in a larger system:** the pipeline *produces* a fine-tuned artifact for DIMER to serve. It is not itself an inference service; serving reconstruction is defined by the artifact contract and demonstrated in `tutorials/swin_classification_colab.ipynb`.

###### Primary Intended Users

Intended users are **machine-learning engineers and data engineers operating DIMER deployments**, and analysts who can assemble a labelled image folder and read an evaluation report. The deployment setting is an internal enterprise or research platform where the operator controls the data, the GPU and the downstream use of predictions.

Users are assumed to understand: how a train/validation split is formed and why the pipeline refuses to invent one; that ImageNet pretraining transfers well to natural photographs and poorly to modalities it never saw (radiology, spectrograms, multispectral rasters); that accuracy on a small validation folder is a noisy estimate; that softmax scores are not calibrated probabilities; and that a GPU exposed as `cuda:0` is required. The pipeline is not built for hobbyist or public-facing self-service use: it fails closed with typed refusal codes rather than guessing, and expects the operator to read them.

###### Out-of-scope use cases

Capability boundaries:

- Not for object detection, instance or semantic segmentation, image captioning, OCR, retrieval or embedding export. Detection and semantic segmentation are the sibling `swin-detection-pipeline` and `swin-segmentation-pipeline`; neither is runtime-complete yet.
- Not for multi-label images (an image belonging to several classes at once) — the head is single-label cross-entropy with `argmax`.
- Not for zero-shot classification: the pipeline emits no prediction without a fine-tuning run on labelled data.

Input boundaries:

- Not for datasets without both a `train/` and a `val/` (or `valid/`) split; `test/` is never reinterpreted as validation and the run is refused (`VISION_MISSING_VALIDATION_SPLIT`).
- Not for image folders containing anything but class directories of images: nested folders, stray files, unknown root entries, symlinks, unrecognised extensions or undecodable files are fatal (`VISION_*` codes below).
- Not for non-square or high-resolution imagery where geometry matters: every image is resized to exactly 256×256 without preserving aspect ratio, so elongated inputs are anisotropically stretched and fine detail above 256 pixels is lost.
- Not for images whose channel structure carries meaning: any decodable image is converted to 3-channel RGB (alpha dropped, grayscale replicated, palette expanded) before the network sees it. No channel-count validation exists.
- Not for CPU-only environments: the finetuner refuses to run without `cuda:0` and offers no CPU fallback.
- Not for class sets whose names collide after Unicode normalisation and case folding (`Cat` and `cat` are refused, `VISION_CLASS_NAME_COLLISION`).

Decision boundaries:

- Not for autonomous decisions affecting people — medical triage, security screening, hiring, credit, law enforcement — with or without a human in the loop; the pipeline has not been validated for any of them.
- Not for any deployment that treats a softmax score as a probability of correctness or applies a threshold that has not been calibrated on the operator's own labelled data.

#### Factors

The pipeline's behaviour varies with the data the operator supplies far more than with anything fixed in this repository; the subsections below record what was and was not considered.

###### Groups

The pipeline is **not human-centric by design**: it classifies whatever image classes the operator's folder defines, and nothing in the validator, finetuner or artifact contract knows whether an image depicts a person. No group-level evaluation was performed and no demographic or phenotypic attributes are collected or inferred.

Two facts transfer an obligation to the operator. First, the upstream pretraining corpus, ImageNet-1k, is a web-scraped photograph collection that contains people and has documented demographic and geographic skews; the `timm` weights are **not group-audited**, and the features the fine-tuned head builds on inherit whatever those skews encode. Second, the fine-tuned model's behaviour on any group is determined by the operator's own training folder. An operator whose images depict people, or whose classes correlate with people's characteristics, must run their own group-disaggregated evaluation — at minimum per-group accuracy and per-group confusion on a labelled holdout that they control — before any deployment. The pipeline does not perform that audit and will not warn that it is needed.

###### Instrumentation

Training and evaluation images are **user-supplied files**; the pipeline has no knowledge of the camera, scanner, rendering engine or export pipeline that produced them, and records only the bytes: every file's SHA-256, its media type as declared by extension and confirmed by decoding (`VISION_IMAGE_EXTENSION_FORMAT_MISMATCH` otherwise), and its EXIF orientation, which is normalised to visual orientation (`transpose-to-visual-orientation`) in both the validator and the finetuner.

Instrument characteristics that reach the model are therefore resolution (everything becomes 256×256), colour encoding (everything becomes 8-bit RGB), JPEG compression artefacts, and orientation metadata. Upstream, the ImageNet-1k pretraining images are consumer photographs of mixed provenance and quality.

Instrument error propagates as feature error without detection: a change of camera, lighting rig, export colour profile or compression level between training and inference shifts the input distribution and the pipeline has no drift detector. It does detect three things mechanically — undecodable files (fatal, never substituted with blank or random pixels: `VISION_IMAGE_DECODE_FAILED`), extension/format contradictions, and files whose bytes changed between validation and training (`HANDOFF_*` digest checks) — and nothing else.

###### Environment

**Operating environment.** Training requires an NVIDIA GPU exposed as `cuda:0`; the finetuner is fail-closed on `--expected-accelerator` (`ACCELERATOR_UNAVAILABLE`, `ACCELERATOR_MISMATCH`) and has no CPU path. Training and inference run in full FP32 with no mixed precision, quantisation, compilation or deterministic-kernel setting. The formally qualified envelope is the finetuner's recorded packet: torch 2.8.0+cu128, `timm 1.0.28`, Pillow 12.3.0, on an RTX 5070 Ti Laptop GPU, with 1724 MiB (Tiny) and 2732 MiB (Small) peak reserved VRAM at batch 8 and 256×256 input (`catalog/base-model-catalog.json`, `qualification.measuredEnvelope`). Runs on other CUDA GPUs (Colab/Kaggle T4-class, torch 2.9) have executed the tutorial path but are execution evidence for those runtimes, not an extension of qualification. No outbound network is permitted while a worker is RUNNING (`securityRequirements.runningOutbound: DENY`); base weights must be staged before execution.

**Data environment.** The pipeline assumes inference images are drawn from the same distribution as the operator's `train/` folder — same subject matter, capture conditions, resolution regime and label semantics — and that `train/` and `val/` are independent samples of that distribution. It assumes photographs or photograph-like renderings in the RGB domain the ImageNet backbone was pretrained on. When those assumptions fail — a new site, season, sensor, or a class that was rare in training — accuracy degrades silently and softmax scores typically remain high; the artifact carries no out-of-distribution detector. Because the resize does not preserve aspect ratio, a change in image geometry between training and inference is itself a distribution shift.

#### Metrics

Metrics were chosen for a single-label classifier whose serving consumer is DIMER: one discrete correctness measure, one likelihood measure, and an explicit statement of what is *not* estimated.

###### Performance Measures

The finetuner reports exactly two metrics in `evaluation-report.json`, computed on the frozen validation split by the **persisted, reloaded** artifact rather than the in-memory model:

- `core.metric.classification.accuracy` — fraction of validation images whose `argmax` class equals the label. Discrete correctness; it is what a downstream consumer of hard labels experiences, and it is blind to how confident the wrong answers were.
- `org.valcorza.metric.classification.cross-entropy` — mean negative log-likelihood of the true class under the softmax output. It moves before accuracy does on small folders and penalises confident errors, so it is the earlier signal of over- or under-fitting.

Both are reported per epoch in the run manifest's `history` (as `validationAccuracy`, `validationLoss`, together with `trainLoss`) and once, from the reloaded artifact, in the evaluation report. Reading accuracy alone hides confident mistakes; reading cross-entropy alone hides the operating point. Top-5 accuracy, precision/recall per class, ROC-AUC and calibration error are **not** computed by the pipeline; the notebook exports per-sample predictions and scores (`validation-predictions.csv`) so an operator can compute them. No ImageNet benchmark number is measured or reproduced by this repository; upstream figures belong to the upstream papers.

###### Decision thresholds

The default decision rule is **`argmax` over the class logits** — an implicit threshold that always emits exactly one class, even for an image that resembles none of them. No acceptance threshold on accuracy or cross-entropy was set during development: the finetuner publishes the artifact whatever the validation metrics are, and records them; the tutorial's only guard is a warning when accuracy fails to beat the majority-class baseline.

No score threshold, abstention rule or reject option is shipped. It is withheld because the softmax outputs are uncalibrated and the pipeline has no knowledge of the operator's error costs. **Calibration and thresholding are owned by the downstream operator**: choose the threshold on a labelled holdout from the deployment domain, weighing the cost of a false positive (acting on a wrong class) against a false negative (routing an image to manual review), and re-tune it whenever the model, the classes or the data source change.

###### Approaches to uncertainty and variability

Reported metrics come from a **single holdout**: the validator-assigned `validation` split, evaluated once per training run. No cross-validation, repeated runs, bootstrap or confidence interval is computed, and the evaluation report carries no dispersion field; a single number must not be read as a stable population estimate.

Run-to-run variability comes from head initialisation, data-loader shuffling, and GPU kernel selection and floating-point reduction order. The first two are controlled by the explicit `--seed` (default `20260828`, applied to `torch.manual_seed`, `torch.cuda.manual_seed_all` and the data-loader generator; Python's `random` and NumPy are not seeded because the worker does not use them); the last is not, so the finetuner records `reproducibility: REEXECUTABLE`, meaning identical inputs reproduce the same validation accuracy on the tutorial sample while cross-entropy drifts in later decimals and the artifact digest differs between runs. Data order is otherwise deterministic — no augmentation is applied.

Softmax scores are **uncalibrated**. They are exposed as raw class scores and must not be presented as probabilities; an operator needing calibrated confidence must fit a calibrator (temperature scaling or isotonic regression) on a labelled holdout they control.

#### Ethical considerations and biases

No external ethics board reviewed this pipeline and no clearance testing with a specific population was performed; the considerations below were made by the developers during design and are recorded so that an operator can judge whether more is needed for their deployment.

###### Data

**Pretraining.** Both base checkpoints were pretrained on ImageNet-1k, as disclosed by the `timm` pretrained configuration (`ms_in1k`) and the Swin V2 authors. ImageNet-1k is a web-scraped photograph corpus of roughly 1.28 million training images across 1000 classes; it contains images of people, was collected without individual consent, and its licence restricts use to non-commercial research. The upstream authors do not enumerate its contents beyond the published class list, so whether it contains personal or otherwise sensitive images cannot be ruled out — it is **known to contain images of people**, and beyond that its sensitivity is **not characterised**.

**Fine-tuning data** is supplied entirely by the operator and never leaves the worker's local filesystem; the pipeline's runtime network policy is `DENY`.

**What this repository distributes:** manifests, provenance records, verification scripts and the tutorial. It distributes **no weights** — base checkpoints are fetched by the operator from the pinned Hugging Face revision and verified by SHA-256 — and **no sample dataset**; the tutorial generates its 24-image synthetic sample at run time. The finetuner's qualification image bakes the two pinned checkpoints in for offline execution; that image is a private artifact.

**Operator obligation.** The pipeline performs no audit of the operator's images for personal data, consent, licensing or confidentiality, and the fine-tuned artifact — although it contains only weights and configuration — is derived from those images and inherits their governance. The operator must establish their right to train on, retain and redistribute both the data and the resulting artifact.

###### Human Life

The pipeline is **not intended** for decisions in health, safety, criminal justice, employment, credit, housing, education or any other matter central to human life or flourishing, and it has **not been validated** for any of them by anyone: the only evaluations that exist are the finetuner's qualification smoke on a synthetic fixture and the tutorial's synthetic and three-class BYOD runs, none of which measure anything about real-world decisions.

Because it is a generic image classifier, use in such a domain is foreseeable — sorting medical images, screening documents for eligibility, flagging people in footage. Any such use would be admissible only with all of: an independent, domain-specific clinical or regulatory validation study on representative data; a human decision-maker who reviews every consequential output rather than an automated action; the calibration and threshold work described under *Decision thresholds*; the group-disaggregated evaluation described under *Groups*; and whatever regulatory clearance the domain requires. None of this is provided or implied by this repository.

###### Mitigations

Every item below names a mechanism present at the pinned worker revisions (`swin-classification-dataset-validator@52d1fd0`, `swin-classification-finetuner@0621a11`) or in this repository.

*Supply-chain integrity.* Base models are an allowlisted catalog of two entries, each pinned to an immutable Hugging Face commit and a `model.safetensors` SHA-256; an off-catalog key is a typed refusal (`BASE_MODEL_OFF_CATALOG`), a missing or non-matching staged file is refused before loading (`BASE_MODEL_FILE_MISSING`, `BASE_MODEL_DIGEST_MISMATCH`), and unqualified entries are refused (`BASE_MODEL_NOT_RUNTIME_QUALIFIED`). Weights are loaded from the staged safetensors file only (`pretrained_cfg_overlay={"file": …}`); no Hub fetch happens at run time and no pickle/`torch.load` path exists. The pipeline release binds worker manifests, worker releases, the contract release and the composition report by digest, and `scripts/verify_release.py` refuses a contract or worker checkout whose tree is not clean at the declared revision — with `scripts/negative_controls.py` proving each refusal.

*Input integrity.* The validator decodes every image and fails on the first undecodable one rather than substituting pixels (`VISION_IMAGE_DECODE_FAILED`); rejects extension/format contradictions, unrecognised extensions, symlinks, path escapes, nested or stray entries and colliding class names; requires `train/` and `val|valid/`, never reinterprets `test/`, and warns when a class is absent from a required split. It emits a digested handoff (logical dataset, data plan, semantic schema) that the finetuner re-verifies (`HANDOFF_*_DIGEST_MISMATCH`, `HANDOFF_SAMPLE_SET_MUTATION`, `HANDOFF_DUPLICATE_SAMPLE_ID`, `HANDOFF_LABEL_MAP_INVALID`); the finetuner cannot add, drop or re-split samples.

*Statistical mitigations.* None are implemented: there is no class re-weighting, balancing or augmentation. The tutorial computes a majority-class baseline from the data plan so an operator can see when the model has learned nothing.

*Reproducibility.* Explicit `--seed`; pinned `timm==1.0.28` and `Pillow==12.3.0`; effective training configuration (`training-config`, its digest, the transforms) recorded in `model-config.json` and the run manifest; the artifact is content-addressed with per-member digests and is **reloaded from disk before the reported metrics are computed**.

*Refusals.* No CPU fallback (`ACCELERATOR_UNAVAILABLE`/`ACCELERATOR_MISMATCH`, `EXPECTED_ACCELERATOR_MISSING`); no outbound network and no custom code during RUNNING (`securityRequirements`); safetensors only.

###### Risks and harms

- **Silent distribution shift.** Mechanism: a fine-tuned classifier keeps emitting confident `argmax` labels on images unlike its training folder. Bearer: whoever acts on the label, and any data subject the image depicts. Realised whenever camera, site, season or class mix changes; likely under normal use over time. Magnitude: from wasted review effort to wrong operational decisions — bounded only by what the operator wires the label to.
- **Overconfidence read as probability.** Mechanism: uncalibrated softmax scores near 1.0 on wrong answers. Bearer: operator and downstream users who threshold on them. Likely wherever a score is displayed as "confidence". Magnitude: mis-prioritised review queues, automation of errors.
- **Inherited pretraining bias.** Mechanism: ImageNet-1k features encode demographic and geographic skews; classes correlated with people's characteristics inherit them. Bearer: data subjects in under-represented groups. Likelihood unknown — not measured here. Magnitude: systematically worse accuracy for some groups.
- **Label leakage into evaluation.** Mechanism: near-duplicate images placed in both `train/` and `val/` by the operator; the validator identifies samples by path and digests each file, but never compares digests across paths, so identical or near-identical images in both splits pass silently. Bearer: operator, through an inflated accuracy that misleads deployment decisions. Likely with scraped or burst-captured data.
- **Small-holdout noise.** Mechanism: a single holdout of a few dozen images yields accuracy with wide sampling error and no interval. Bearer: operator making go/no-go decisions. Very likely with small datasets.
- **Automation bias.** Mechanism: human reviewers defer to the model's label. Bearer: data subjects and third parties. Use-context risk independent of model quality.
- **Geometry distortion.** Mechanism: anisotropic resize to 256×256 destroys aspect-ratio cues and fine detail. Bearer: operator, as unexplained failures on elongated or high-resolution inputs.
- **Artifact governance.** Mechanism: the fine-tuned artifact is a derivative of the operator's data and may be shared as if it were a neutral model. Bearer: data owners whose licence or confidentiality terms are breached. Magnitude: legal and privacy exposure.

###### Use cases

The developers consider the following uses unacceptable even where the pipeline would function:

1. **Surveillance, biometric or demographic profiling, and social scoring** — classifying people by identity, ethnicity, gender, age, religion, health status, disability, or any protected characteristic, or scoring individuals for behaviour; the ImageNet backbone's known people-related biases make this doubly unacceptable.
2. **Unlawful discrimination** — using image classes as a proxy for eligibility decisions in employment, housing, credit, insurance, education or access to healthcare.
3. **Deceptive, manipulative or predatory applications** — content moderation or targeting systems designed to exploit users, or presenting the model's uncalibrated scores as verified facts to people affected by them.
4. **Uses prohibited by the upstream terms** — the `timm` weights are MIT-licensed, but they were pretrained on ImageNet-1k, whose access terms restrict the dataset to non-commercial research and education. Whether that restriction binds derived weights is contested; operators intending commercial deployment must make their own determination and MUST NOT rely on this repository for it. Any deployment must also honour the operator's own data licences and DIMER's platform terms.

---

## Model details

| Item | Value |
|---|---|
| Pipeline id / version | `org.valcorza.swin-classification` / `0.1.0` (`release/pipeline-release.json`) |
| Task profile | `core.task.vision.image-classification` |
| Dataset representation | `core.dataset.vision.image-folder` |
| Training method | `core.training.supervised-finetuning` (full-network AdamW, cross-entropy; defaults: 1 epoch, batch 8, lr 1e-4, weight decay 0.01, seed 20260828) |
| Base models | `timm/swinv2_tiny_window8_256.ms_in1k` @ `650d02aabf05e8adbd060a739ab39e39f53da639`, `model.safetensors` sha256 `c47f52b4556ff4436aa9502f5efbc93aac77fdab42d9d70dd845931757ff5d65`; `timm/swinv2_small_window8_256.ms_in1k` @ `0c9500fcde4c689e97ff51954debae59c158af0d`, sha256 `7e793c2f3576d20b5f746ac5dc6b7271745d612f9ce06ccabe382c36f39f1caa` |
| Weight licence | MIT per the `timm` pretrained-config licence field, recorded in the finetuner catalog; ImageNet-1k dataset terms apply to the pretraining data. Machine-readable `redistribution_status` is not yet declared (Pipeline Spec LIC4) — see *Known gaps* |
| Components | validator `org.valcorza.swin-classification-dataset-validator` 0.1.0 @ `52d1fd0ca9af5bfd413b0cd7cc1e3f441560fa95`; finetuner `org.valcorza.swin-classification-finetuner` 0.1.0 @ `0621a11c1cdb6749984aaaca6919086e5c2a7d65`; contract `kurtvalcorza/ml-worker` @ `0f0c221222402721ee7716edf01378604cbd6ef3` |
| Input contract | image folder: `train/<class>/*`, `val|valid/<class>/*`, optional `test/<class>/*`; extensions `.jpg .jpeg .png .bmp .tif .tiff .webp`; EXIF orientation normalised; no channel-count check (converted to RGB); no image-count ceiling |
| Preprocessing | resize to 256×256 (bicubic, antialias, aspect not preserved) → tensor → normalise mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225); identical for training, evaluation and inference |
| Output contract | per image: one logit per class in `classNames` order (frozen by the validator's `labelMap`, index → name); `argmax` decision; softmax scores uncalibrated |
| Artifact | generation directory `model.safetensors` + `model-config.json` (`schemaVersion 1.0`: `timmModelName`, `numClasses`, `classNames`, `input`, `transforms`, `trainingConfigDigest`, `baseModel`, `handoff`) + `artifact-manifest.json` (per-member SHA-256, `bundleDigest`); contains no training images |
| Metrics | `core.metric.classification.accuracy`, `org.valcorza.metric.classification.cross-entropy` on the validation split, from the reloaded artifact |
| Runtime | qualified image: Python 3.11.13, torch 2.8.0+cu128, `timm==1.0.28`, `Pillow==12.3.0` (`qualification/image-ff4297dd/pip-freeze.txt` in the finetuner); `cuda:0` required; FP32 |
| Lifecycle | `candidate` — implementation and composition verified; open gates listed below |

## Known gaps (recorded, not hidden)

- **No public load-and-predict operation** in the finetuner release (Pipeline Spec API7/ARCH2). Serving reconstruction follows the artifact contract, as demonstrated in the tutorial; the recipe's two unrecorded parameters (bicubic interpolation, antialias) come from `trainer.py` at the pinned revision (PRE11/ART22).
- **No channel-count or minimum-size validation** before model execution (Pipeline Spec §21.9); images are converted to RGB and resized unconditionally.
- **No duplicate-content detection across splits** (SPL9).
- **`redistribution_status` not declared machine-readably** for the two base checkpoints (LIC4); the licence determination above is recorded in prose and in the finetuner catalog only.
- **No DIMER serving integration and no clean-runtime tutorial execution recorded yet** (SRV*, REL5/REL6 for the tutorial path); see `tutorials/RELEASE_VERIFICATION.md`.

## References

- Liu, Z. et al. *Swin Transformer V2: Scaling Up Capacity and Resolution.* CVPR 2022. arXiv:2111.09883.
- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. arXiv:2103.14030.
- Microsoft Swin Transformer: https://github.com/microsoft/Swin-Transformer
- `timm` weights: https://huggingface.co/timm/swinv2_tiny_window8_256.ms_in1k · https://huggingface.co/timm/swinv2_small_window8_256.ms_in1k
- Repository provenance: `provenance/open-weights.json`, `release/`, `evidence/release-verification-kaggle.json`
- Tutorial: `tutorials/swin_classification_colab.ipynb` (DIMER Notebook Spec 1.0, `E2E`, candidate)
