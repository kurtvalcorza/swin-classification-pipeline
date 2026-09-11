# Executor procedure — worker images, image smoke, release re-pin

These are the exact scripts that produced `release/image-smoke-evidence.json` and the
0.1.1 worker-release pins on 2026-09-11. They are committed for provenance (Pipeline
Spec REL5/REL8), not as a portable tool: host paths (`C:\swin-executor\...`,
`/mnt/c/...`) are hard-coded and the host is a WSL2 clone of the
`nvidia-container-base` golden image (Docker 29 with the containerd image store +
NVIDIA Container Toolkit).

| Step | Script | Runs where | Network |
|---|---|---|---|
| 0 | export `git archive` trees of the two worker revisions + `schemas/` from `ml-worker@0f0c221` into `src/` | Windows | — |
| 1 | `build_images.sh` — pull the digest-pinned base, build both images, record image ids, context hygiene, runtime versions, staged-weight digests | distro | allowed (pip, HF weight staging) |
| 2 | `image_smoke.sh` — fixture in the validator image → validator handoff → finetuner train/publish/reload | distro | `--network none` for both worker stages; `--gpus all` for the finetuner |
| 2b | `validate_docs.py` — every emitted contract document against the pinned schemas | distro | — |
| 3 | `repin_release.py` — worker manifests, image-smoke evidence, worker releases, pipeline manifest, composition report (contract solver), pipeline release, `pipeline-metadata.json` | Windows, pipeline repo root, contract worktree at `0f0c221` | — |
| 4 | `verify_release.py` 32/32, `negative_controls.py` 7/7, `verify_image_release.py` 7/7, `verify_open_weight_provenance.py`, `verify_pipeline_metadata.py` + 12 negative controls, hygiene oracle 2/2 | Windows | — |

Image digests are the local containerd manifest digests (`docker image inspect --format '{{.Id}}'`); they become registry digests unchanged on push.
