# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | Release status |
|---|---|---|---|---|
| `swin_classification_colab.ipynb` | `E2E` | Image-folder validation → supervised SwinV2 fine-tuning → artifact publication/reload → new-image inference | CUDA GPU (`cuda:0`) | **Candidate** — static source checks pass; clean-runtime execution evidence is still required before marking release-grade |

## Conformance notes

- The notebook exercises the pipeline release's pinned validator and finetuner worker CLIs rather than reimplementing the training path.
- The validator and finetuner source repositories are private. The notebook requires a `GITHUB_TOKEN` Colab Secret/environment variable with read access and passes it through an ephemeral Git HTTP header; credentials are not printed, placed in clone URLs, or persisted in Git config.
- The default dataset is deterministic synthetic tutorial data. Its metrics are sanity evidence only, not ImageNet or production evidence.
- **DAT7 SHOULD deviation:** a Colab BYOD folder-upload path is not included in this first notebook revision. The production representation is a split/class directory tree; a secure and low-friction folder-transfer UX should be added separately rather than teaching an unsafe archive shortcut.
- PyTorch/torchvision are supplied by the accelerator runtime and reported at execution. The notebook pins the worker's principal Python dependencies but must not claim formal hardware qualification outside the repository's recorded qualification packet.
- CI/static validation does not satisfy clean-runtime execution requirements. A release review must record an actual clean GPU run for the notebook revision before the registry status is promoted to `release-grade`.
