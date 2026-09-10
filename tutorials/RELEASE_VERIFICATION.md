# Swin Classification Tutorial Release Verification

DIMER Notebook Specification 1.0 requires clean-runtime execution evidence in addition to static source conformance. This file is the durable release-gate record for `tutorials/swin_classification_colab.ipynb` (`E2E`).

## Automatic coverage

Pull-request CI currently verifies:

- runnable image-release evidence and its binding to the pinned validator/finetuner WorkerReleases;
- open-weight/model provenance;
- notebook JSON/source structure, declared `E2E` profile and Notebook Spec 1.0 identity;
- clean committed notebook state and Python-cell compilation;
- secure private-source bootstrap markers and rejection of credential-in-URL patterns;
- immutable pipeline binding, worker CLI usage, pinned model acquisition/digest verification, CUDA requirement, fresh artifact reload, and machine-readable prediction/provenance export markers.

These are source/provenance checks. They are **not** REL1/REL5 execution evidence.

A manually dispatched supplemental workflow, `.github/workflows/notebook-release-gpu.yml`, is also available for a self-hosted runner labelled `linux`, `x64`, and `gpu`. It requires repository secret `DIMER_WORKER_READ_TOKEN`, executes the committed notebook top-to-bottom, checks required outputs, and retains an executed notebook plus machine-readable evidence. A run of that workflow is useful executor evidence, but it does not silently broaden the user-facing runtime claim beyond the documented supported Colab path.

## Supported release verification

Before changing the tutorial registry from `Candidate` to `release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new Google Colab CUDA runtime;
3. configure a `GITHUB_TOKEN` Colab Secret with read access to the private `swin-classification-dataset-validator` and `swin-classification-finetuner` repositories;
4. run the notebook top-to-bottom without editing implementation cells;
5. verify the immutable pipeline/worker revisions reported by the notebook;
6. verify the allowlisted upstream checkpoint identity and SHA-256 before model use;
7. verify dataset validation, supervised training, evaluation plus majority baseline, artifact publication, fresh reconstruction, and separate new-image inference all complete;
8. verify the exported prediction/provenance files exist and the interpretation/limits section matches the observed path;
9. record the exact notebook SHA, Colab/Python/Torch/torchvision/CUDA/GPU identity, worker/model revisions, result paths, outcome, and any warnings in a durable issue/PR/release record;
10. do not record access tokens or other secrets.

## Current status

No clean supported Colab GPU PASS is asserted by this repository record yet. Static CI and the supplemental GPU executor are preparatory controls; the tutorial remains **Candidate** until the supported-runtime evidence is independently reviewed.
