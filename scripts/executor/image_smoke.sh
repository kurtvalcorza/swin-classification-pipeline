#!/usr/bin/env bash
# Executor step 2: end-to-end image smoke from the runnable worker images themselves.
#   fixture (validator image, Pillow) -> validator image handoff (--network none)
#   -> finetuner image train/publish/reload from baked-in weights (--network none --gpus all)
# Mirrors qualification/harness/run_smoke.py stage semantics but runs each stage in its own image.
set -uo pipefail
X=/mnt/c/swin-executor/rebuild-2026-09-11
R=/root/swin-rebuild
VREV=$(tr -d '\r\n' < "$X/VALIDATOR_REV"); FREV=$(tr -d '\r\n' < "$X/FINETUNER_REV")
VTAG="swin-classification-validator:${VREV:0:7}"; FTAG="swin-classification-finetuner:${FREV:0:7}"
OUT="$X/out"; W="$R/smoke"; rm -rf "$W"; mkdir -p "$W" "$OUT"
cp "$R/finetuner/qualification/harness/generate_fixture.py" "$W/generate_fixture.py"

sd() { printf 'sha256:%s' "$(printf 'smoke:%s' "$1" | sha256sum | cut -d' ' -f1)"; }
bind() { local role=$1; echo --job-id "smoke-$role" --attempt-id attempt-1 --worker-release-digest "$(sd $role-release)" --effective-job-spec-digest "$(sd $role-job-spec)" --admission-record-digest "$(sd $role-admission)" --security-grant-digest "$(sd $role-grant)"; }

echo "=== fixture (inside $VTAG, --network none) ==="
docker run --rm --network none -v "$W:/smoke" --entrypoint python "$VTAG" /smoke/generate_fixture.py /smoke/fixture | tee "$OUT/fixture.txt"
find "$W/fixture" -type f | sort | tee "$OUT/fixture-files.txt" | wc -l

echo "=== validator stage ($VTAG, --network none) ==="
docker run --rm --network none -v "$W:/smoke" "$VTAG" /smoke/fixture /smoke/handoff $(bind validator) > "$OUT/validator.log" 2>&1; echo "VALIDATOR_EXIT=$?"
python3 - "$W/handoff" <<'PY' | tee "$OUT/validator-summary.json"
import json,sys
from pathlib import Path
h=Path(sys.argv[1]); r=json.loads((h/"result.json").read_text()); rm=json.loads((h/"run-manifest.json").read_text())
l3=json.loads((h/"evidence"/"l3.json").read_text())
print(json.dumps({"state":r["state"],"runManifestDigest":r["runManifestDigest"],"observed":rm["observed"],"l3Findings":[f["code"] for f in l3["findings"]]},indent=2))
PY

echo "=== finetuner stage ($FTAG, --gpus all --network none) ==="
docker run --rm --gpus all --network none -v "$W:/smoke" "$FTAG" \
  /smoke/fixture /smoke/handoff /opt/worker/weights /opt/worker/base-model-catalog.json /smoke/training \
  --model-key swinv2-tiny-window8-256-ms-in1k --expected-accelerator cuda:0 $(bind finetuner) \
  --epochs 1 --batch-size 4 --seed 20260828 > "$OUT/finetuner.log" 2>&1; echo "FINETUNER_EXIT=$?"
grep -E "WARNING" "$OUT/finetuner.log" | head -3
python3 - "$W/training" <<'PY' | tee "$OUT/finetuner-summary.json"
import json,sys
from pathlib import Path
t=Path(sys.argv[1]); r=json.loads((t/"result.json").read_text()); rm=json.loads((t/"run-manifest.json").read_text()); ev=json.loads((t/"evaluation-report.json").read_text())
gen=(t/"artifact"/"CURRENT").read_text().strip(); members=sorted(p.name for p in (t/"artifact"/"generations"/gen).iterdir())
mm=json.loads((t/"artifact"/"generations"/gen/"model_manifest.json").read_text()) if (t/"artifact"/"generations"/gen/"model_manifest.json").exists() else None
o=rm["observed"]
print(json.dumps({"state":r["state"],"runManifestDigest":r["runManifestDigest"],"artifactBundleDigest":o["artifactBundleDigest"],"device":o["device"],"timmVersion":o["timmVersion"],"torchVersion":o["torchVersion"],
 "metrics":{m["id"]:m["value"] for m in ev["metrics"]},"splitCounts":o["splitCounts"],"inputImageForms":o.get("inputImageForms"),"artifactMembers":members,"modelManifest":mm,"reproducibility":rm["reproducibility"]},indent=2))
PY

echo "=== contract document validation (schemas from ml-worker@0f0c221) ==="
python3 - "$R/schemas" "$W" <<'PY' | tee "$OUT/schema-validation.txt"
import json,sys,importlib
from pathlib import Path
try:
    import jsonschema
except ImportError:
    print("jsonschema unavailable on host; skipped (run inside an image if needed)"); sys.exit(0)
schemas=Path(sys.argv[1]); w=Path(sys.argv[2])
mapping={"handoff/logical-dataset-manifest.json":"logical-dataset-manifest.schema.json","handoff/data-plan.json":"data-plan.schema.json","handoff/semantic-dataset-schema.json":"semantic-dataset-schema.schema.json","handoff/validated-dataset-manifest.json":"validated-dataset-manifest.schema.json","handoff/result.json":"result.schema.json","handoff/run-manifest.json":"run-manifest.schema.json","training/result.json":"result.schema.json","training/run-manifest.json":"run-manifest.schema.json","training/evaluation-report.json":"evaluation-report.schema.json","training/artifact-manifest.json":"artifact-manifest.schema.json","training/execution-plan.json":"execution-plan.schema.json"}
ok=0
for doc,sch in mapping.items():
    d=w/doc; s=schemas/sch
    if not d.exists() or not s.exists(): print("SKIP", doc, "(missing doc or schema)"); continue
    try: jsonschema.Draft202012Validator(json.loads(s.read_text())).validate(json.loads(d.read_text())); print("PASS", doc); ok+=1
    except Exception as e: print("FAIL", doc, str(e)[:160])
print(f"{ok} documents valid")
PY

echo "=== environment ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
docker image inspect "$VTAG" "$FTAG" --format '{{.Id}}' | tee "$OUT/image-ids.txt"
echo "SMOKE_DONE"
