# Release verification

`tutorials/swin_classification_colab.ipynb` (`E2E`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported GPU runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 1.1. This file is the durable release-gate record for the notebook; it supersedes
`tutorials/RELEASE_VERIFICATION.md`, whose rows for the previous (non-standalone) notebook revision are kept below as
history.

## Automatic coverage (static, every pull request)

CI (`.github/workflows/standalone-notebook.yml`; `verify-image-release.yml` runs the same validator through
`scripts/validate_colab_tutorial.py`) runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and `generated_from`
  (repository, module commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install, repository import, worker CLI or
  subprocess on the primary path; exactly one cell tagged `embedded_module` equal to
  `src/swin_classification_pipeline/pipeline.py` after the generator's documented rewrites; the inline `MANIFEST` equal
  to the committed snapshot manifest and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook
  byte-identical to `tools/build_notebook.py` output; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions beyond the
  allowlisted worker/contract/catalog pins;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `SwinClassificationPipeline.from_pretrained(weights_dir=...)`, `generate_synthetic_sample` / `safe_extract_zip`,
  `dataset_inventory`, `validate_inputs` with a rejected probe, `validate_dataset`, `pipe.fit(...)` with the
  fail-closed CUDA check, `from_artifact`, `verify_artifact_generation`, `evaluate_split`, the reload equivalence
  checks, `majority_class_baseline`, `evaluation_report`, `predict`), the ceiling print (`MIN_CLASSES`,
  `MAX_EXPANDED_BYTES`, `INPUT_SIZE`, `IMAGE_EXTENSIONS`), the four exports, the learner-facing statements
  (gradient adaptation, 100 % in-kernel, no silent CPU fallback, reproducibility boundary, persisted-bytes metrics,
  uncalibrated scores, no threshold, single holdout, `sample-sanity` / `not-measurable`) and the gated-off BYOD
  defaults (`USE_BYOD`, `USE_BYOD_IMAGE`); forbidden patterns (credential-in-URL, any `git clone` / `github.com` /
  repository import on the primary path, a mutable `revision='main'`, direct `timm.create_model` / `from timm import` /
  `from torchvision import` / `from huggingface_hub import` / `torch.optim` / `load_state_dict` / `safetensors` /
  `zipfile.ZipFile` / `Image.open` use **outside the carried module cell**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`, `worker.run(` / `subprocess.run([` outside the install cell);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (YAML-list `base_model` naming `timm/swinv2_tiny_window8_256.ms_in1k`), single H1, required heading order, and the
  `## Model details` provenance table.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/test_role_helpers.py`,
`tests/test_notebook_parity.py`; no torch, no weights, no GPU — the module imports torch/timm lazily inside the
training and inference functions). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 class), `cuda:0` | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle `NvidiaTeslaT4`, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| `.github/workflows/notebook-release-gpu.yml` | Self-hosted `linux`/`x64`/`gpu` runner | Manually dispatched executor for the committed notebook (`nbconvert`); it predates the standalone carrier and must be re-validated against the regenerated notebook before its output is cited |
| Local harness (pre-flight only) | Workstation GPU, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or the Kaggle T4 executor above) with **no
   repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults for the
   sample path: `USE_BYOD = False`, `USE_BYOD_IMAGE = False`, `EPOCHS = 1`, `SEED = 20260910`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`; note the pinned `torch==2.14.0` / `torchvision==0.29.0` replace the Colab-provided wheels and
   the install guard may require one runtime restart);
5. verify every default-path stage completes:
   - the carried module cell executes (defines the pipeline class and helpers) with no import of the repository package;
   - the inline `MANIFEST` is asserted against the module identity and written to
     `weights/swinv2-tiny-window8-256-ms-in1k/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports both
     manifest entries (`config.json`, `model.safetensors`) on a clean runtime, `verify_snapshot` returns the manifest
     dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports `source == 'local-snapshot'` on `cuda:0`;
   - the synthetic 24-image two-class sample is generated in code and the inventory printed;
   - `validate_inputs` writes `outputs/swin_classification_input_manifest.json` (verdict `accepted`; the
     `VISION_DUPLICATE_CONTENT_ACROSS_SPLITS` warning may appear as a non-fatal finding; one recorded rejection from
     the stray-root-file probe) and `validate_dataset` reports `state == 'SUCCEEDED'` with `labelMap` `cool`/`warm`;
   - `pipe.fit` runs on `cuda:0`, publishes one generation with four members and reports accuracy / cross-entropy from
     the reloaded artifact;
   - the fresh-boundary reload prints `accuracyMatches: true` and `crossEntropyMatches: true`;
   - `evaluation_report` writes `outputs/swin_classification_evaluation_report.json` with verdict `sample-sanity`
     and the majority baseline `0.5`;
   - the synthetic new image is classified (`predictedClass` printed with scores summing to 1);
   - `outputs/swin_classification_result.json` and `outputs/swin_classification_validation_predictions.csv` written
     with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, torchvision, timm, CUDA, GPU), model
   identifier and immutable revision, whether the model cache was clean, outcome, produced outputs, and any warning
   or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/swin_classification_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/swin_classification_colab.ipynb`). Wall times, when recorded, are the sum of per-cell times
reported by the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

### Manual clean-runtime evidence (standalone notebook)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — queued to the GPU lane |

### History: previous (non-standalone) notebook revision

The rows below were recorded for the PR #15 notebook, which cloned this repository at `PIPELINE_REF` and carried the
validator/finetuner code inline. The regenerated standalone notebook shares that code (now
`src/swin_classification_pipeline/pipeline.py`) but is a different blob with different Sections 1–3, so **none of
these rows carry over**; they are kept as provenance for the extracted code only.
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

No clean-runtime execution of the standalone notebook has been recorded yet; the run is **pending** and queued to
the GPU lane. Static validation (`tools/validate_release_assets.py`), the generator parity check, a `compile()` sweep
over every code cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is
necessary but not sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run
against the notebook blob under review and an integrator promotes it; promotion is not performed by the builder.
Facts a reviewer should weigh: the pinned `torch==2.14.0` / `torchvision==0.29.0` wheels have never been installed on
a Colab/Kaggle T4 by this notebook (the previous revision used the runtime-provided torch); the extracted module's
training path (`train_model`) was executed only inside the previous notebook revision's local pre-flight runs, never
as a package; the standalone carrier — executing the carried module cell in a runtime with no repository checkout —
has been validated statically and by a CPU carrier probe (module cells + identity assertion, no fetch), never run
end to end; and `stage_missing_files` with a real Hub download has not been executed for this manifest.
