# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Artifact | Release status |
|---|---|---|---|---|---|---|
| `swin_classification_colab.ipynb` | `E2E` | Image-folder validation → supervised SwinV2 fine-tuning → content-addressed artifact publication → fresh-boundary reload → new-image inference | CUDA GPU (`cuda:0`; Colab/Kaggle T4 class) | ZIP image folder + single image, gated off by default | `model.safetensors` + `model-config.json` + `model_manifest.json` + `artifact-manifest.json` generation (release 0.1.1 pins) | **Candidate / not public-ready** — static checks and local pre-flight pass; public launch remains blocked by exact-revision clean supported GPU execution (REL1/REL5) |

## Conformance notes

- The notebook exercises the pipeline release's pinned validator and finetuner worker CLIs (`swin-classification-validate`, `swin-classification-train`) rather than reimplementing validation or training. New-image inference rebuilds the model from the published artifact contract (`model-config.json` + `model.safetensors`) and imports the finetuner's own image-decoding function and normalisation constants, because the worker ships no standalone predictor CLI.
- **Public access:** the pipeline, validator and finetuner repositories are public, so the notebook clones the exact pinned worker revisions anonymously and a fresh user needs no repository-specific authorization (this closes the public-bootstrap gate that issue #12 tracked). The `GITHUB_TOKEN` fallback is retained only for a private fork or mirror; when it is used, credentials travel through an ephemeral Git HTTP header and are not printed, placed in clone URLs, persisted in Git config, or written to any export.
- The default dataset is deterministic synthetic tutorial data (24 PNGs, two classes). Its metrics are sanity evidence only, not ImageNet or production evidence.
- **BYOD (DAT7–DAT14):** `USE_BYOD_DATASET` accepts one ZIP laid out as `train/<class>/`, `val/<class>/`; extraction rejects absolute paths, `..` traversal, backslash-ambiguous names, symlinks, root escapes, and archives above `MAX_EXPANDED_MIB`. `USE_BYOD_IMAGE` accepts a single new image. Both default to `False` so the sample path never opens an upload dialog. Uploaded data is only read by the local workers.
- **Fresh-boundary verification (VER1–VER8):** the published generation is copied to a fresh directory, every manifest member digest is re-verified, the model is rebuilt from the artifact alone, and the frozen validation split is re-scored; accuracy must equal the worker's report exactly and cross-entropy must agree within `1e-4`.
- Seeds are explicit (`SAMPLE_SEED`, `SEED`); the finetuner records `reproducibility: REEXECUTABLE`. The notebook states that GPU kernel selection still causes small loss drift and a different artifact digest between runs.
- PyTorch/torchvision are supplied by the accelerator runtime and reported at execution; worker-level dependencies are pinned (`timm==1.0.28`, `Pillow==12.3.0`, `huggingface_hub==1.29.0`, `safetensors==0.8.0`). The notebook must not claim formal hardware qualification outside the finetuner's recorded qualification packet.
- **Serving reconstruction (DIMER Pipeline Specification 1.0 API7/PRE11):** release 0.1.1's pinned finetuner emits `model_manifest.json` (timm identifier, class order, bicubic interpolation, `crop_pct 1.0`/`squash`, normalisation), the contract consumed by `dimer-inference-service-timm`, DIMER's serving path. The notebook's Sections 8–9 still reconstruct from `model-config.json` + `model.safetensors` and mirror the resize interpolation from `trainer.py` (the service repository is private and not installed by the notebook); switching Section 8 to the service's own loader remains a follow-up.
- `MODEL_CARD.md` **exists** in this repository and is part of the release documentation. The current notebook revision does not yet link it directly; this G16 `SHOULD` deviation is recorded here and should be removed when the notebook is next materially revised.
- CI/static validation (`scripts/validate_colab_tutorial.py`) does not satisfy clean-runtime execution requirements. A release review must confirm a clean supported Colab/Kaggle T4 run for the exact notebook revision under review before the registry status is promoted to `release-grade`.

## Public-release gates

1. ~~**Public worker bootstrap — issue #12.**~~ Closed: the pinned validator and finetuner repositories are public and the notebook clones them anonymously at the exact pinned revisions.
2. **Clean supported GPU execution — REL1/REL5.** Execute the exact release-candidate notebook top-to-bottom in a fresh supported Colab/Kaggle T4-class runtime and record commit/blob identity, environment, and outcome in `RELEASE_VERIFICATION.md`.

Gate 2 is still required. Passing only static CI or local pre-flight does not make the notebook public-ready.
