# swin-classification-pipeline

Pipeline umbrella for the first **NATIVE ml-worker** vision pipeline: SwinV2
image classification on the [ml-worker contract](https://github.com/kurtvalcorza/ml-worker).
It binds the dataset validator and finetuner workers into one composed,
digest-pinned, contract-validated release.

| Component | Repository | Pinned revision |
| :--- | :--- | :--- |
| Dataset validator | [swin-classification-dataset-validator](https://github.com/kurtvalcorza/swin-classification-dataset-validator) | `818d88a4a829bc810a4f054a61e32b627035646c` |
| Finetuner | [swin-classification-finetuner](https://github.com/kurtvalcorza/swin-classification-finetuner) | `7025c875847d7e75dcad2a1f7f15fd1275a8d9f0` |
| Contract | [ml-worker](https://github.com/kurtvalcorza/ml-worker) `build/dimer-v1-freeze` | `0f0c221222402721ee7716edf01378604cbd6ef3` |

Worker source revisions are exact immutable commits, independent of branch state.

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

The worker releases pin the shared qualification image
(`sha256:ff4297dd81798d91c8616dc96fcb61de5cbdeca102afb964562fd52c62b43d47`) and
reference the finetuner repo's executed evidence packets
(`qualification/blackwell-training-smoke-cf3f429.json` — source-bound GPU smoke —
and `qualification/blackwell-timm-1.0.28.json` — Blackwell runtime qualification)
as conformance evidence.

## Verify

Every binding re-derives from the committed inputs and the pinned contract:

```sh
git -C <ml-worker> worktree add /tmp/mlw 0f0c221222402721ee7716edf01378604cbd6ef3
ML_WORKER_SRC=/tmp/mlw python scripts/verify_release.py
```

Exit 0 means: all 7 documents schema-valid, every digest chain intact, and a
fresh composition run reproduces the committed `COMPATIBLE` report.

## Status

Layer-1 (worker repos + this umbrella) of the three-layer freeze program. The
layer-2 contract PR (image-folder representation profile, required-audits entries,
NATIVE audits) and layer-3 freeze-ledger extension live in the ml-worker stack
and are tracked in the spec (`#1`).
