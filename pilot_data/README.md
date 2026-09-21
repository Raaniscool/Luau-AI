# DukeOTR Real Model Pilot source briefs

`real_model_pilot_v1.jsonl` is a small, fixed, project-authored set of 12 `pilot-...` briefs for the Windows-local **Qwen3-4B / `qwen3:4b`** baseline/discovery workflow.

It is intentionally separate from `raw_data/` and from `evaluation_data/`:

- every brief has `source.kind: "project_authored"` and `source.pilot_only: true`;
- the runner rejects duplicate/non-`pilot-` IDs and runs a local held-out wording-isolation check before any Ollama request;
- raw and reviewed model output remains in ignored `reports/real_model_pilots/<run-id>/` and is not automatically source data or final training data;
- do not add held-out evaluation text, rubrics, answers, or score artifacts here.

Run the no-model audit and the eventual Windows-local pilot only through the documented coordinator:

```powershell
py -3 -m scripts.run_real_model_pilot --run-id win-qwen3-4b-pilot-YYYYMMDD --dry-run
```

See [`docs/REAL_MODEL_PILOT.md`](../docs/REAL_MODEL_PILOT.md) for safety boundaries, preflight, resume, artifact inspection, and limitations.
