# Release verification

`tutorials/swin_classification_colab.ipynb` (`E2E`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported GPU runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 2.2. This file is the durable release-gate record for the notebook; it supersedes
`tutorials/RELEASE_VERIFICATION.md`, whose rows for the previous (non-standalone) notebook revision are kept below as
history.

## Automatic coverage (static, every pull request)

CI (`.github/workflows/standalone-notebook.yml`; `verify-image-release.yml` runs the same validator through
`scripts/validate_colab_tutorial.py`) runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, `standalone: true` and `generated_from`
  (repository, module commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install, repository import, worker CLI or
  subprocess on the primary path; exactly one cell tagged `embedded_module` equal to
  `src/swin_classification_pipeline/pipeline.py` after the generator's documented rewrites; the inline `MANIFEST` equal
  to the committed snapshot manifest and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook
  byte-identical to `tools/build_notebook.py` output; exactly two kernel cells — the isolated install (pinned `uv` by size and
  SHA-256, managed CPython, the hash lock with `--require-hashes --only-binary :all:`) and the router — no `pip install` into the
  kernel and no restart instruction anywhere;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions beyond the
  allowlisted worker/contract/catalog pins;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `SwinClassificationPipeline.from_pretrained(weights_dir=...)`, `generate_synthetic_sample` / `safe_extract_zip`,
  `dataset_inventory`, `validate_inputs` with a rejected probe, `validate_dataset`, `pipe.fit(...)` with the
  fail-closed CUDA check, `from_artifact`, `verify_artifact_generation`, `evaluate_split`, the reload equivalence
  checks, `majority_class_baseline`, `mean_colour_baseline`, `evaluation_report`, `predict`), the ceiling print (`MIN_CLASSES`,
  `MAX_EXPANDED_BYTES`, `INPUT_SIZE`, `IMAGE_EXTENSIONS`), the four exports, the learner-facing statements
  (gradient adaptation, 100 % in this notebook, no silent CPU fallback, reproducibility boundary, persisted-bytes metrics,
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
4. verify that Section 1 builds the isolated environment and routes the later cells to it with **no runtime restart**
   (record `restarted: false`; any restart fails the gate), that it reports `NOTEBOOK_SOURCE.repository_revision` equal
   to the module commit recorded in `metadata.dimer.generated_from` (and that `git show <revision>:src/swin_classification_pipeline/pipeline.py`
   hashes to `module_sha256`), and that the imported core package versions equal the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - the carried module cell executes (defines the pipeline class and helpers) with no import of the repository package;
   - the inline `MANIFEST` is asserted against the module identity and written to
     `weights/swinv2-tiny-window8-256-ms-in1k/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports both
     manifest entries (`config.json`, `model.safetensors`) on a clean runtime, `verify_snapshot` returns the manifest
     dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports `source == 'local-snapshot'` on `cuda:0`;
   - the synthetic 24-image two-class sample is generated in code and the inventory printed;
   - `validate_inputs` writes `outputs/swin_classification_input_manifest.json` (verdict `accepted`; **no**
     `VISION_DUPLICATE_CONTENT_ACROSS_SPLITS` warning — the sample draws distinct images per split and the cell stops if
     one appears; one recorded rejection from the stray-root-file probe, naming `notes.txt`) and `validate_dataset` reports `state == 'SUCCEEDED'` with `labelMap` `cool`/`warm`;
   - `pipe.fit` runs on `cuda:0`, publishes one generation with four members and reports accuracy / cross-entropy from
     the reloaded artifact;
   - the fresh-boundary reload prints `accuracyMatches: true` and `crossEntropyMatches: true`;
   - `evaluation_report` writes `outputs/swin_classification_evaluation_report.json` with verdict `sample-sanity`
     and two baselines: majority `0.5` and mean colour `1.0` (the default sample is separable by colour alone, so the
     default evaluation cannot show what fine-tuning adds);
   - the synthetic new image is classified (`predictedClass` printed with scores summing to 1);
   - `outputs/swin_classification_result.json` and `outputs/swin_classification_validation_predictions.csv` written
     with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, torchvision, timm, CUDA, GPU), model
   identifier and immutable revision, whether the model cache was clean, `restarted: true/false`, the reported
   metrics and both baselines, the reload check, outcome, produced outputs, and any warning
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
| 2026-10-29 | `179e8235f6471729430e8e67dc861e6fb0051998` (PR #24 head) / `swin_classification_colab.ipynb` blob `b1e3e4fb99a0edba1b58091d8173e93a804a4507` (current blob) | Google Colab CLI sequential execution (`colab exec -f`), fresh Colab Tesla T4 (session `suite-swin-179e823-52d2`): every code cell run in order in one kernel; not a browser `Run all`, and the CLI records no execution counts, so order is evidenced by `exec.log` (cells 1–12 in order); kernel Python 3.13.16, isolated CPython 3.12.12 (45 locked packages, 56 s); the committed blob fetched at the 40-character SHA and Git-blob verified, input unchanged | Default path only, form fields at their defaults, no overrides | 100.2 s | **One pass, no restart, 0 errors**; 12/12 code cells; VM released (`stop_verified`); the executed notebook's SHA-256 equals `run_summary.json` `executed_sha256`. 2 of 2 snapshot files (114,919,250 bytes) verified, `cuda:0`; tutorial dataset cool / warm, 8 + 8 train, 4 + 4 validation; 1 epoch, train loss 0.2483745296485722, validation accuracy 1.0, cross-entropy 0.007677703339140862; fresh-boundary reload: accuracy 1.0, cross-entropy 0.007677674293518066 (within the 0.0001 tolerance), `Fresh-boundary verification PASSED` on 8 validation samples; majority-class baseline 0.5, mean-colour baseline 1.0 (the notebook notes that the fine-tuned model does not beat the mean-colour baseline on this holdout). Evidence `docs/execution-evidence/2026-10-29-179e823/`: SHA-256 captured → committed: `swin_classification_colab_179e823_colab-cli-t4_default.ipynb` `97ecb0215835320047023bbdddc0bcec079c74838dde48ab8dd4f992f2f2f7f3` → unchanged (no local path); `swin_classification_colab_179e823_colab-cli-t4_default_exec.log` `0450c19c6f198608ae21fb61a4b5e7808e60dd879061e052d107cb12201f3d59` → `833e6f8ec98ae64a931357525fbc928558b663ebb9807d3f234e1c2327c2a67b` (2 local-path replacements); `swin_classification_colab_179e823_colab-cli-t4_default_run_summary.json` `920dba84c4aa3ea26b08f5af19c1325f959680780d955e8163e7a92a885d58db` → unchanged (no local path). Not exercised: an accepted BYOD input, optional switches off their defaults, and the form widgets |
| 2026-10-29 | same | same, fresh VM (session `suite-swin-179e823-52d2`) | Scripted DAT20 refusal: `USE_BYOD = True`; `BYOD_ZIP_PATH = '/etc/hostname'` (set by `--set` on a copy; the run copy differs from the blob in exactly those 2 source lines), and the run copy cut right after the refusing cell `swin_classification-16`, dropping 13 cells (6 code cells), which reproduces `Run all` stopping at the first error | 82.3 s | **Refused as designed** — 6/6 code cells of the cut copy. The 5 earlier code cells completed without an error; the refusing cell printed exactly one `Refused:` line, with no traceback: `Refused: hostname: File is not a zip file; fix the input and run this cell again (the earlier results of this session and their outputs/ files were removed).`. The suite recorded the halting error as `IsolatedCellRefusal`. `verdict` `PASS`, input unchanged, VM released. Cut copy `run_copy.ipynb` SHA-256 `84e25e003ecdf6dfa81f41c719424f8df9ffe47aa463424ac0de77c95995b3b2` (Git blob `40edcdb5`), equal to the summary's `run_copy_sha256`. This run says nothing about the 13 cut cells, which did not run in it. Evidence `docs/execution-evidence/2026-10-29-179e823/`: SHA-256 captured → committed: `swin_classification_colab_179e823_colab-cli-t4_refusal.ipynb` `74e49d8673797f363ddc97812ee1b4b8e55cb46a049c5d3e5402b8f63b674df8` → unchanged (no local path); `swin_classification_colab_179e823_colab-cli-t4_refusal_exec.log` `55cca256795462e9b9a4bebaf07e8446cb0807687881c55272115dd2f544338f` → `28882f3f287a0255379de84f6949732052e40ce4f179e01e92d035f9389a761f` (2 local-path replacements); `swin_classification_colab_179e823_colab-cli-t4_refusal_run_summary.json` `330236bad14f2bbcd6d7d74d61cb9529abb1bfac986b8393c9730de47e1f948d` → unchanged (no local path); `swin_classification_colab_179e823_colab-cli-t4_refusal_run_copy.ipynb` `84e25e003ecdf6dfa81f41c719424f8df9ffe47aa463424ac0de77c95995b3b2` → unchanged (no local path) |
| 2026-09-14 | `b1382d1` / `18aad3a06bb6` | Kaggle T4 (`kurtvalcorza/dimer-nb2-swin-classification` v1) | Default sample path | 172.2 s | **PASSED** — 10/10 code cells executed cleanly, 1 weights staged. Applies to blob `18aad3a06bb6` only (the in-kernel install revision); it records no metrics, library versions, device or restart status. |
| 2026-10-10 (10:57:20 UTC start) | `5f584090e3022d5ed72d672255c24934607eb2ab` / `fdef15099d9d81cd66a06635b2259e9d436f45fb` (`NOTEBOOK_SOURCE.repository_revision` `52b357227ef8`, `module_sha256` `fc0cf7e7fb70…`, generator `build_notebook.py/2.1-swc`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-swin-5f58409-52d2`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (45 locked packages, setup 50 s), `torch 2.14.0+cu130`, `timm 1.0.28`, `cuda:0` | Default sample path, every form field at its default (BYOD off) | 134.9 s | **PASSED** — one pass, no restart, 0 errors; 12/12 code cells in order (cell 4, the carried module, prints nothing); 2 snapshot files digest-verified at `650d02aabf05`; 8 train / 4 validation per class; `notes.txt` root-entry probe rejected; validator `SUCCEEDED`; 1 epoch: validation accuracy 1.0, cross-entropy 0.007678; artifact members verified, fresh reload accuracy 1.0 and cross-entropy within 1e-4; baselines majority 0.5, mean-colour nearest centroid 1.0; new image → `warm` (0.9966). Evidence in `docs/execution-evidence/2026-10-10-5f58409/`: executed notebook SHA-256 `12bc3b789f06d066666fd93fa3929763f9bcb515e34f86e0032c88300f19296d`, `run_summary.json` `5bce32c118157a8e43a685720f636169ae7c03723e4778dfed31ba16199ce812`, `exec.log` `8ac3e4a4b0e5f5da8c95f69bc22d476b6b0f9a63625d5c48f2f6cd0e2abf0548`. Not exercised: BYOD, the optional activity, a browser Run all |
| 2026-10-25 (07:19:46 UTC start) | `558c10a2b0db6dcff184c957b5cc6e60c91b638e` (PR #23 head) / `e868a656f74f9ca67e28e64b621fbefa0ad93309` (tier C review fixes, generated at `b742539`) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-swin-558c10a-52d2`), committed blob fetched at the commit and checked before the VM was allocated | Default sample path, every form field at its default (BYOD off) | 81.5 s | **PASSED** — one pass, no restart, 0 errors; 12/12 code cells in order in `exec.log`; root-entry probe (`notes.txt`) rejected; validator `SUCCEEDED`; 1 epoch: train loss 0.2484, validation accuracy 1.0, cross-entropy 0.007678 (`0.007677703339140862`); fresh reload accuracy 1.0, cross-entropy 0.007677674 (`accuracyMatches: true`, `crossEntropyMatches: true`, tolerance 1e-4); baselines majority 0.5, mean-colour 1.0 (the notebook prints that the fine-tuned model does not beat the mean-colour baseline); new image → `warm` (0.9966). Every number equals the 2026-10-10 run and the 0.0077 the worked answers quote. Evidence in `docs/execution-evidence/2026-10-25-558c10a/`: SHA-256 captured → committed: executed notebook `swin_classification_colab_558c10a_colab-cli-t4_default.ipynb` `96dcdaf660eb55508d61c1d90fea8f27fb4cf3255e67ab55902f34fad50df362` → unchanged (no local path); `run_summary.json` `4e9702d8ee34ec2f443c69fdb36d4da4df51dca16249052a14774b1d3f2487e6` → unchanged; `exec.log` `6fd07aa6e8ac0becc69e3e1cda2fc4d92bf936281ea6c66223052ac598ffad9e` → `d6baea476147932afdb8ea5ea70929f76be56ed32f5b2628da6775f78a8b0180` (2 local-path replacements). |

The current notebook (blob `e868a656`, tier C review fixes) is recorded in the 2026-10-25 Colab T4 row above; the 2026-10-10 row is history for the previous blob `fdef1509` (REL14).

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

**Current status (2026-10-25): Candidate.** `tutorials/swin_classification_colab.ipynb` blob `e868a656f74f9ca67e28e64b621fbefa0ad93309` (PR #23 head `558c10a`, generated at `b742539`, T85-1..T85-4: markdown, the use notice, `metadata.dimer` and the `NOTEBOOK_SOURCE` record only) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 12/12 code cells, 81.5 s; validation accuracy 1.0 and cross-entropy 0.0077 after one epoch, majority baseline 0.5, mean-colour baseline 1.0, fresh reload matches; evidence in `docs/execution-evidence/2026-10-25-558c10a/`). This covers the **default path** only. Remaining gates: REL12 (a BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). Under REL14 the runs of earlier blobs are history.

The current blob `b1e3e4fb` (code commit `5a7a40a`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); on 2026-10-29 its default path (12/12 code cells, one pass, no restart, 0 errors) and its scripted DAT20 refusal (`USE_BYOD=True;BYOD_ZIP_PATH='/etc/hostname'`: exactly one `Refused:` line at cell `swin_classification-16`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI sequential execution, not a browser `Run all`; see "Recorded executions" in `docs/release-verification.md`), and the hosted runs of earlier blobs, including `e868a656f74f`, are history for it (REL14).

Static validation (`tools/validate_release_assets.py`), the generator parity check, a `compile()` sweep
over every code cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is
necessary but not sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run
against the notebook blob under review and an integrator promotes it; promotion is not performed by the builder.
Facts a reviewer should weigh: the first hosted run (Kaggle T4, 2026-09-14) executed an earlier blob that installed
the pins into the kernel, and recorded no metrics, versions or restart status; the notebook installs the pins
into an isolated `uv` environment, and its previous blob `fdef15099d9d` (commit `5f58409`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-10 (Colab CLI 0.7.4 sequential execution, 12/12 code cells, 134.9 s; validation accuracy 1.0 and cross-entropy 0.0077 after one epoch, majority baseline 0.5, mean-colour baseline 1.0, fresh reload matches).