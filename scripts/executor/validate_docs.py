import json, sys
from pathlib import Path
import jsonschema
schemas = Path("/root/swin-rebuild/schemas/schemas"); w = Path("/root/swin-rebuild/smoke")
mapping = {"handoff/logical-dataset-manifest.json":"logical-dataset-manifest.schema.json","handoff/data-plan.json":"data-plan.schema.json","handoff/semantic-dataset-schema.json":"semantic-dataset-schema.schema.json","handoff/validated-dataset-manifest.json":"validated-dataset-manifest.schema.json","handoff/result.json":"result.schema.json","handoff/run-manifest.json":"run-manifest.schema.json","handoff/execution-plan.json":"execution-plan.schema.json","training/result.json":"result.schema.json","training/run-manifest.json":"run-manifest.schema.json","training/evaluation-report.json":"evaluation-report.schema.json","training/artifact-manifest.json":"artifact-manifest.schema.json","training/execution-plan.json":"execution-plan.schema.json","training/evaluation-plan.json":"evaluation-plan.schema.json","training/resolved-base-model-manifest.json":"resolved-base-model-manifest.schema.json"}
ok=0; n=0
for doc, sch in mapping.items():
    d=w/doc; s=schemas/sch
    if not d.exists(): print("ABSENT", doc); continue
    if not s.exists(): print("NOSCHEMA", sch); continue
    n+=1
    try:
        jsonschema.Draft202012Validator(json.loads(s.read_text())).validate(json.loads(d.read_text())); print("PASS", doc); ok+=1
    except Exception as e: print("FAIL", doc, str(e)[:200])
print(f"{ok}/{n} documents valid against ml-worker@0f0c221 schemas")
print("handoff files:", sorted(p.name for p in (w/"handoff").iterdir())); print("training files:", sorted(p.name for p in (w/"training").iterdir()))
