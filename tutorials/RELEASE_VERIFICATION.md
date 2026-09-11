# Swin Classification Tutorial Release Verification

DIMER Notebook Specification 1.0 requires clean-runtime execution evidence (REL1/REL5) in addition to static source conformance. This file is the durable release-gate record for `tutorials/swin_classification_colab.ipynb` (`E2E`).

## Automatic coverage (static, every pull request)

`verify-image-release.yml` runs `scripts/validate_colab_tutorial.py`, which checks:

- notebook JSON parses; every Python cell compiles; no persisted outputs or execution counts; no `TODO`/`TBD`/`FIXME`;
- declared `E2E` profile and Notebook Spec `1.0` identity in notebook metadata;
- the immutable pipeline anchor (`PIPELINE_REF`), pinned-worker checkout, anonymous public clone with a secure token fallback (ephemeral `extraHeader`, token deleted after use) and rejection of credential-in-URL, `trust_remote_code`, pickle/`torch.load`, `pretrained=True` and `extractall` patterns;
- worker CLI usage, checkpoint acquisition with SHA-256 verification, the `cuda:0` fail-closed requirement;
- gated-off BYOD flags, archive-safety function and its rejection messages, expanded-size cap;
- fresh-directory reload with the finetuner's own preprocessing imports, `strict=True` loading, and the accuracy/cross-entropy equivalence checks;
- data-derived majority baseline, the four machine-readable exports, and the required learner-facing sections (objectives, exclusions, schema, uncalibrated scores, `argmax`, single holdout, non-bitwise reproducibility, interpretation, troubleshooting, next experiments);
- every code cell is preceded by an explanatory markdown cell.

These are source/provenance checks. They are **not** REL1/REL5 execution evidence.

## Executor paths

| Path | Runtime | Token delivery | Role |
|---|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 class) | None (public repositories); Colab Secret `GITHUB_TOKEN` only for a private fork | The runtime the tutorial is written for; a clean top-to-bottom run here is the promotion evidence |
| Kaggle CLI kernel | Kaggle `NvidiaTeslaT4`, Python 3.12 image | None (public repositories); Kaggle Secret `GITHUB_TOKEN` only for a private fork | Reproducible clean-room executor that matches the Colab GPU class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and `/content` |
| `.github/workflows/notebook-release-gpu.yml` | Self-hosted `linux`/`x64`/`gpu` runner; optional repository secret `DIMER_WORKER_READ_TOKEN` for a private fork | Environment variable (optional) | Manually dispatched executor; runs the committed notebook with `nbconvert`, asserts the exports and the fresh-boundary equivalence flags, uploads evidence |
| Local WSL harness (pre-flight only) | Workstation GPU, `run_nb.py` sequential cell executor with a `google.colab` shim | Environment variable | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the tutorial registry from `Candidate` to `release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or the Kaggle T4 executor above);
3. provide no credential (the `swin-classification-dataset-validator` and `swin-classification-finetuner` repositories are public); a `GITHUB_TOKEN` secret is needed only when verifying a private fork or mirror;
4. run the notebook top-to-bottom without editing implementation cells (form parameters at defaults for the sample path);
5. verify the immutable pipeline/worker revisions reported by Section 1 match the release files at `PIPELINE_REF`;
6. verify the checkpoint digest in Section 5 equals the catalog pin before the model is built;
7. verify dataset validation, fine-tuning on `cuda:0`, evaluation plus majority baseline, fresh-boundary reload with `accuracyMatches` and `crossEntropyMatches` both `true`, and new-image inference all complete;
8. verify the four exports exist and the interpretation section matches the observed path;
9. record the notebook SHA-256, commit, runtime (platform, Python, torch, torchvision, CUDA, GPU), worker/model revisions, outcome and any warnings in the table below;
10. record no access tokens or other secrets.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/swin_classification_colab.ipynb` (verify with `git rev-parse <commit>:tutorials/swin_classification_colab.ipynb`) together with the SHA-256 of the committed LF bytes. The local harness executed the Windows working copy (CRLF), which is JSON-identical to the committed blob. Wall times are the sum of per-cell times reported by the harness and include installs and the checkpoint download; they are measurements for the stated runtime, not general estimates.

Pre-flight runtime: WSL2 Ubuntu 24.04.4 (kernel 6.18.33), Python 3.12.3, torch 2.9.1+cu128, torchvision 0.24.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 5070 Ti Laptop GPU (driver 610.88), timm 1.0.28, Pillow 12.3.0 in-kernel. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Notebook blob / SHA-256 prefix | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `a2025fb784c5` / `76924c5b05bbd2ec` | Local WSL harness | Default sample path, all 10 code cells, token clone | 35.3 s | PASS — validator `SUCCEEDED`; checkpoint digest matched (114,918,618 bytes); 1 epoch accuracy 1.0 / CE 0.007642; fresh reload `accuracyMatches=true`, `crossEntropyMatches=true` (|Δ| 1.4e-8); new image → `warm`; 4 exports |
| 2026-09-11 | `a2025fb784c5` / `76924c5b05bbd2ec` | Local WSL harness | BYOD: `USE_BYOD_DATASET=True` (27-file ZIP, wrapper folder, 3 classes, EXIF-orientation-6 JPEGs mixed with PNGs), `EPOCHS=3`, `USE_BYOD_IMAGE=True` via `BYOD_IMAGE_PATH` | 54.1 s | PASS — expanded 192,077 bytes; validator `SUCCEEDED` with 3 classes; majority baseline 0.333 from the data plan; accuracy 1.0; fresh reload PASSED on 9 samples; new image → `dots` |
| 2026-09-11 | `a2025fb784c5` / `76924c5b05bbd2ec` | Local negative controls (`safe_extract_zip` source lifted verbatim from the notebook), CPython 3.12.3 | `train/../../evil.txt`, `/tmp/evil.txt`, symlink member, `train\..\evil.txt`, 1000-byte cap vs 192 KB archive | — | PASS — all five rejected with the expected message; good archive extracted 27 files |
| 2026-09-11 | `4875c6419407` / `01d0440018fcb778` | Local WSL harness | Default path after the effective-version restart-boundary check | 83.3 s | PASS — `pillowInKernel` 12.3.0 = pinned; reload equivalence PASSED |
| 2026-09-11 | `604df01ec060` / `bd64d9e86244249d` | Local WSL harness, **no `GITHUB_TOKEN` in the environment** | Default path with worker checkouts pre-staged from all-refs git bundles (`workerSourceMode` = pre-staged) | 27.0 s | PASS — pinned revisions enforced by `checkout_pinned`; reload equivalence PASSED |
| 2026-09-11 | `604df01ec060` / `bd64d9e86244249d` | Local WSL harness | Default path, token clone (`workerSourceMode` = cloned with ephemeral token) | 34.4 s | PASS |
| 2026-09-11 | `14b3830967c1` / `92937036fad81557` | Local WSL harness | Default path after preprocessing is read from `model-config.json → transforms.validation` | 30.6 s | PASS — artifact normalisation equals the worker constants; reload equivalence PASSED |
| 2026-09-11 | `bc5ee58f278f` / `9f453373495a0003` | Local WSL harness | `PIPELINE_REF` re-pointed to `3e4087ab` (release 0.1.1: validator `7c6e77f`, finetuner `7063849`); default path, token clone; the only code change vs the previous blob is the anchor constant, prose updated for the four-member artifact | 122.6 s | PASS — workers cloned at the 0.1.1 pins; artifact now carries `model_manifest.json`; reload `accuracyMatches=true`, `crossEntropyMatches=true` |
| 2026-09-11 | `91fd64b7d651` / `8f54b46ee75594f1` | — (markdown-only revision) | Section 3 schema wording corrected to the validator's actual rules (class-name collision, root-entry and nesting refusals, RGB conversion, aspect-ratio stretch) and two troubleshooting rows added; **all 10 code cells byte-identical** to blob `14b3830967c1`, so its execution evidence carries over | — | n/a |
| 2026-09-11 | `4875c6419407` / `01d0440018fcb778` | Kaggle kernel `kurtvalcorza/swin-cls-tutorial-t4-verify` v2 (Python 3.12.13, Linux 6.12.90) | Section 1 only | — | Stopped as designed at the fail-closed `GITHUB_TOKEN` check (no secret attached); the message is the UX6 corrective text. Not execution evidence. |
| 2026-09-11 | `8546802af945` / `b2d2cfb10dae61fd` | Windows 11 host, CPython 3.12, git 2.x, **no `GITHUB_TOKEN`/`GH_TOKEN` in the environment** | Section 1 bootstrap only (code cell 3 up to the pip installs), after the worker repositories became public: anonymous clone path | — | PASS — pipeline `3e4087ab`, validator `7c6e77fd`, finetuner `70638493` checked out at the pinned SHAs; `workerSourceMode` = cloned anonymously from public GitHub; unauthenticated `curl` of both repositories' `git-upload-pack` refs returned 200. **Code cell 3 changed in this revision, so earlier local-harness rows do not carry over; a full run of this blob is still outstanding.** |

## Current status

Static CI, the archive-safety controls, and the local pre-flight are preparatory evidence. The tutorial remains **Candidate** until a clean supported-class GPU run (Colab or the Kaggle T4 executor) for the exact notebook blob under review is appended to the table and reviewed.
