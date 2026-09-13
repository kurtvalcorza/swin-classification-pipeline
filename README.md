# swin-classification-pipeline

Pipeline umbrella for the first **NATIVE ml-worker** vision pipeline: SwinV2
image classification on the [ml-worker contract](https://github.com/kurtvalcorza/ml-worker).
It binds the dataset validator and finetuner workers into one composed,
digest-pinned, contract-validated release.

## Upstream alignment

This pipeline corresponds to the **Image Classification** downstream task of the
original Microsoft Swin Transformer project.

- **Upstream project:** [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- **Upstream task:** Image Classification
- **Canonical benchmark:** ImageNet-1K
- **Reference architecture:** Swin/SwinV2 backbone + image-classification head
- **Pipeline task boundary:** image → class label
- **Primary metrics:** top-1 accuracy; top-5 accuracy where applicable
- **This implementation:** digest-pinned SwinV2 Tiny and Small ImageNet-1K checkpoints through the pipeline's allowlisted base-model catalog

The repo family deliberately mirrors the three canonical Swin downstream task
families: **Image Classification**, **Semantic Segmentation**, and **Object
Detection**. Each pipeline shares Swin backbone lineage while keeping a separate
task head, dataset contract, metrics surface, and finetuning path.

## Open weights / checkpoint provenance

Microsoft Swin Transformer is the architecture/source-of-record. The operational
weights used by this pipeline are the exact `timm` Hugging Face artifacts bound in
`provenance/open-weights.json` by immutable repository revision and
`model.safetensors` SHA-256.

Run the self-contained provenance check with:

```sh
python scripts/verify_open_weight_provenance.py
```

This verifies that every `modelDescriptor` in `pipeline-manifest.json` has a
matching immutable, digest-pinned provenance entry and that runtime network fetch
and off-catalog selection remain fail-closed.

| Component | Repository | Pinned revision |
| :--- | :--- | :--- |
| Dataset validator | [swin-classification-dataset-validator](https://github.com/kurtvalcorza/swin-classification-dataset-validator) | `7c6e77fd72a79e7debe9ef811fc6c3227d44ce92` |
| Finetuner | [swin-classification-finetuner](https://github.com/kurtvalcorza/swin-classification-finetuner) | `70638493d081501501506108b23c6937eff770ef` |
| Contract | [ml-worker](https://github.com/kurtvalcorza/ml-worker) `build/dimer-v1-freeze` | `0f0c221222402721ee7716edf01378604cbd6ef3` |

Worker source revisions are exact immutable commits, independent of branch state; the
values above are the `sourceRevision` fields of `release/worker-release-*.json`, which are
the source of truth if this table ever disagrees.

## Pipeline surface

- **Task profile:** `core.task.vision.image-classification`
- **Dataset representation:** `core.dataset.vision.image-folder` (`train/`+`val|valid/`
  class directories; split ownership belongs to the validator; `test/` is never
  reinterpreted as validation)
- **Training method:** `core.training.supervised-finetuning`
- **Base models** (digest-pinned catalog, off-catalog selection is a typed refusal;
  no runtime Hub access): SwinV2 Tiny and Small, window8 256, ImageNet-1k
- **Exposed config:** `baseModel` — one of the two catalog keys, projected to the
  finetuner only
- **Security:** no outbound network during RUNNING, no custom code

## Documents

| File | Contract schema |
| :--- | :--- |
| `pipeline-manifest.json` | `pipeline-manifest.schema.json` |
| `workers/{validator,finetuner}/worker-manifest.json` | `worker-manifest.schema.json` (byte-pinned copies at the revisions above) |
| `release/worker-release-{validator,finetuner}.json` | `worker-release.schema.json` |
| `release/composition-report.json` | `composition-report.schema.json` — solver output, status `COMPATIBLE` (15/15 checks) |
| `release/pipeline-release.json` | `pipeline-release.schema.json` — binds every digest above plus the contract release |
| `pipeline-metadata.json` | DIMER Pipeline Specification 1.0 §28 (repository-owned; not a contract schema) — lifecycle, topology, capability modes, component/contract identities, per-model licence and `redistribution_status`, release gates |

The worker releases (0.1.1) pin the **runnable worker images** built from the
revisions above — validator `sha256:bcd0f816…`, finetuner `sha256:bbcbe1ff…`
(containerd manifest digests; unchanged on push) — and cite as conformance
evidence the finetuner repo's executed packets
(`qualification/blackwell-training-smoke-cf3f429.json` — source-bound GPU smoke —
and `qualification/blackwell-timm-1.0.28.json` — Blackwell runtime qualification)
plus `release/image-smoke-evidence.json`: the end-to-end smoke run **from those
images** on 2026-09-11 (fixture in the validator image → validator handoff →
finetuner train/publish/reload from baked-in weights, `--network none`, `--gpus all`,
RTX 5070 Ti; 14/14 emitted documents schema-valid against `ml-worker@0f0c221`).
Both images were built from post-`.dockerignore` revisions (`.git`/`.github`
absent from `/opt/worker/src`), which closes review finding R1. The exact
Executor procedure is committed under `scripts/executor/`.

## Verify

Every binding re-derives from the committed inputs and the pinned contract:

```sh
git -C <ml-worker> worktree add /tmp/mlw 0f0c221222402721ee7716edf01378604cbd6ef3
ML_WORKER_SRC=/tmp/mlw \
VALIDATOR_SRC=<swin-classification-dataset-validator checkout> \
FINETUNER_SRC=<swin-classification-finetuner checkout> \
    python scripts/verify_release.py
```

Exit 0 means: all 7 documents schema-valid, every digest chain intact, a fresh
composition run reproduces the committed `COMPATIBLE` report, and every external
and logical identity the release names is proven rather than assumed —

- the contract checkout is a **clean git tree at the pinned revision**, not
  merely whatever path `ML_WORKER_SRC` points at;
- each committed worker-manifest copy **agrees with `worker-manifest.json` at
  that release's declared `sourceRevision`** in the worker repository;
- `pipelineId` agrees across the manifest and the release, each `workerId`
  agrees with its worker manifest, and each manifest declares its expected role.

All three checkouts are required. A missing worker checkout is reported as a
failed check and a non-zero exit, never as a silent pass. The narrower
`verify_image_release.py` and `verify_open_weight_provenance.py` checks are
self-contained and do not require the external source trees.

The contract-identity checks are **gates, not checks**: they run before
`ML_WORKER_SRC/src` is placed on `sys.path`, because importing contract code
executes module-level code out of a caller-supplied tree, and a check that runs
after the import cannot undo it. Source manifests are parsed with the contract's
own strict loader, so a source document that stdlib JSON would accept but the
contract rejects (DOC-001: duplicate object keys) is never digested as valid.

`scripts/negative_controls.py` is the executable proof that the verifier refuses
what it claims to refuse. It builds each scenario in a temporary directory --
nothing in this repository or in your checkouts is modified -- and asserts the
specific refusal, including that an unproved contract checkout's code never runs:

```sh
ML_WORKER_SRC=/tmp/mlw \
VALIDATOR_SRC=<validator checkout> \
FINETUNER_SRC=<finetuner checkout> \
    python scripts/negative_controls.py
```

### Cloud verification (Kaggle clean room)

The three verification gates were executed end-to-end in an isolated Kaggle cloud container:

- **Kernel:** [`kurtvalcorza/swin-classification-verify`](https://www.kaggle.com/code/kurtvalcorza/swin-classification-verify)
- **Status:** `KernelWorkerStatus.COMPLETE` (Exit Code 0)
- **Environment:** Linux `6.12.90+` x86_64, Python `3.12.13`
- **Gate 1 (`scripts/verify_image_release.py`):** 7/7 checks passed (0.12s) — smoke status `PASSED`, validator and finetuner image digests bound to evidence.
- **Gate 2 (`scripts/verify_release.py`):** 32/32 checks passed (0.53s) — contract verified clean at pinned revision `0f0c221`, validator at `52d1fd0`, finetuner at `0621a11`, manifests and composition report reproduced.
- **Gate 3 (`scripts/negative_controls.py`):** 7/7 controls discriminated (1.36s) — baseline verifies (`exit 0`), unproved/dirty/non-git contract refused (`exit 1`), duplicate keys refused (`exit 1`).
- **Machine-readable evidence:** [`evidence/release-verification-kaggle.json`](file:///evidence/release-verification-kaggle.json)

## Model card and specifications

`MODEL_CARD.md` (DIMER Model Card Specification 1.0) records intended use, factors,
metrics, mitigations, risks and prohibited uses for this pipeline, with every claim
tied to the pinned worker revisions above. `tutorials/README.md` is the notebook
registry (DIMER Notebook Specification 2.0).

## Status

**Lifecycle:** `candidate` — implementation topology `COMPOSED-WORKERS`, capability mode
`GRADIENT-ADAPTATION` (DIMER Pipeline Specification 1.0 §3), declared machine-readably in
[`pipeline-metadata.json`](pipeline-metadata.json) together with the per-checkpoint weight
licence and `redistribution_status` (`permitted`, MIT, reviewed 2026-09-11; the ImageNet-1k
pretraining-data terms are recorded as a use-restriction note for operators), the component
identities, the composition digest and the release gates. `scripts/verify_pipeline_metadata.py`
cross-checks every value against `pipeline-manifest.json`, `provenance/open-weights.json` and
`release/*.json`, refuses a `release` lifecycle while any gate is open, and fails `unknown`,
`prohibited` and unsatisfied `conditional` redistribution states; `scripts/negative_controls_metadata.py`
proves those refusals (11 controls). Both run in CI (`verify-pipeline-metadata.yml`).

Open release gates: DIMER serving integration, a public load-and-predict operation in the
finetuner, clean-runtime execution evidence for the tutorial, model-card content review. The
repository is not release-ready and must not be represented as such.


Layer-1 (worker repos + this umbrella) of the three-layer freeze program. The
layer-2 contract PR (image-folder representation profile, required-audits entries,
NATIVE audits) and layer-3 freeze-ledger extension live in the ml-worker stack
and are tracked in the spec (`#1`).

The currently pinned worker images still predate their `.dockerignore` fixes;
rebuilding and repinning those images remains an Executor/worker-repo dependency
tracked by PR #4 and is intentionally not papered over by this pipeline-only pass.
