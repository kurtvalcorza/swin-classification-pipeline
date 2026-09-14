# Weight provenance and DIMER hosting

- Upstream: `timm/swinv2_tiny_window8_256.ms_in1k` (SwinV2 Tiny, window 8, 256×256, ImageNet-1k; Microsoft Swin Transformer weights repackaged by `timm`)
- Immutable revision: `650d02aabf05e8adbd060a739ab39e39f53da639`
- Weight format: SafeTensors (`model.safetensors`, 114918618 bytes, SHA-256 `c47f52b4556ff4436aa9502f5efbc93aac77fdab42d9d70dd845931757ff5d65`)
- Upstream weight license: MIT (timm pretrained-config licence field); ImageNet-1k dataset terms apply to the pretraining data
- Local snapshot: `weights/swinv2-tiny-window8-256-ms-in1k/` with `dimer-base-manifest.json` (per-file bytes + SHA-256 for `config.json` and `model.safetensors`, `totalBytes` 114919250); the Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is ignored; `config.json` and the manifest are committable).
- Load-time check: `verify_snapshot()` in `src/swin_classification_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch; `stage_missing_files(..., allow_download=True)` fetches only absent entries, from the Hub at the immutable revision (never `main`).
- Catalog: `BASE_MODEL_CATALOG` in the module also lists `timm/swinv2_small_window8_256.ms_in1k` @ `0c9500fcde4c689e97ff51954debae59c158af0d` (`model.safetensors` SHA-256 `7e793c2f3576d20b5f746ac5dc6b7271745d612f9ce06ccabe382c36f39f1caa`) for reference; the standalone notebook pins and stages the Tiny snapshot only. Both catalog entries mirror `provenance/open-weights.json` (checked by `scripts/verify_open_weight_provenance.py`).
- DIMER hosting: MIT permits use, modification, distribution and commercial use subject to the licence notice; DIMER may mirror the pinned checkpoint in its model store under the upstream licence. The fine-tuned artifact the pipeline publishes contains no training images.
- Loader trust boundary: `timm==1.0.28` built-in `swinv2_tiny_window8_256` architecture built with `pretrained=False`; weights loaded from the verified file with `safetensors` (`strict=True`); no `torch.load`, no remote code. Fine-tuning runs only on `cuda:0` (fail-closed).
