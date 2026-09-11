#!/usr/bin/env bash
# Executor step 1: build both worker images from git-archived source trees.
# Host: nvidia-container-swin (WSL2, native Docker 29 + NVIDIA toolkit). Network allowed (pip + HF weight staging).
set -uo pipefail
X=/mnt/c/swin-executor/rebuild-2026-09-11
R=/root/swin-rebuild
VREV=$(tr -d '\r\n' < "$X/VALIDATOR_REV"); FREV=$(tr -d '\r\n' < "$X/FINETUNER_REV")
VTAG="swin-classification-validator:${VREV:0:7}"; FTAG="swin-classification-finetuner:${FREV:0:7}"
OUT="$X/out"; mkdir -p "$OUT"
rm -rf "$R"; mkdir -p "$R"; cp -r "$X/src/validator" "$X/src/finetuner" "$X/src/schemas" "$R/"
# strip CRLF that Windows git may have applied to text files in the archive (archives from a Windows clone keep LF; belt and braces)
find "$R" -type f \( -name '*.py' -o -name '*.txt' -o -name '*.json' -o -name '*.toml' -o -name 'Dockerfile' -o -name '.dockerignore' \) -exec sed -i 's/\r$//' {} +
echo "docker $(docker version --format '{{.Server.Version}}') | storage: $(docker info --format '{{.Driver}} {{range .DriverStatus}}{{index . 0}}={{index . 1}} {{end}}')"
echo "validator src: $VREV  finetuner src: $FREV"

echo "=== pull base (pinned digest) ==="
docker pull pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385 > "$OUT/pull.log" 2>&1; echo "PULL_EXIT=$?"; tail -2 "$OUT/pull.log"

echo "=== build validator $VTAG ==="
( cd "$R/validator" && docker build --pull=false -t "$VTAG" . ) > "$OUT/build-validator.log" 2>&1; echo "VALIDATOR_BUILD_EXIT=$?"; tail -3 "$OUT/build-validator.log"
echo "=== build finetuner $FTAG ==="
( cd "$R/finetuner" && docker build --pull=false -t "$FTAG" . ) > "$OUT/build-finetuner.log" 2>&1; echo "FINETUNER_BUILD_EXIT=$?"; tail -3 "$OUT/build-finetuner.log"
grep -E "staged|MISMATCH|sha256" "$OUT/build-finetuner.log" | grep -v "^#[0-9]* \[" | head -6

echo "=== image identities ==="
for t in "$VTAG" "$FTAG"; do
  docker image inspect "$t" --format "$t id={{.Id}} repoDigests={{json .RepoDigests}} size={{.Size}} created={{.Created}}"
done | tee "$OUT/images.txt"
echo "=== context hygiene inside images ==="
for t in "$VTAG" "$FTAG"; do
  docker run --rm --entrypoint sh "$t" -c 'cd /opt/worker/src && for f in .git .gitignore .gitattributes .github tests; do [ -e "$f" ] && echo "PRESENT $f" || echo "absent  $f"; done' | sed "s/^/$t /"
done | tee "$OUT/context-hygiene.txt"
echo "=== finetuner runtime versions + staged weights ==="
docker run --rm --entrypoint python "$FTAG" -c 'import PIL, safetensors, timm, huggingface_hub, torch, torchvision; print("pillow", PIL.__version__, "| safetensors", safetensors.__version__, "| timm", timm.__version__, "| huggingface_hub", huggingface_hub.__version__, "| torch", torch.__version__, "| torchvision", torchvision.__version__)' | tee "$OUT/versions.txt"
docker run --rm --entrypoint sh "$FTAG" -c 'find /opt/worker/weights -type f -exec sha256sum {} \;; ls -d /root/.cache/huggingface 2>/dev/null || echo NO-HUB-CACHE' | tee "$OUT/weights.txt"
echo "BUILD_DONE"
