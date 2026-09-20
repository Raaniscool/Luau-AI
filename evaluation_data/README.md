# Held-out evaluation data

`roblox_luau_eval.jsonl` is a project-authored, rubric-driven evaluation suite. Its
prompts are intentionally not read by the training generator. Keep the task wording,
rubrics, and model outputs separate from training data to avoid benchmark leakage.

`coverage_plan.json` tracks the deliberate expansion path from the current 24 authored tasks
toward a mature 100–300-task suite. It contains only coverage taxonomy and planned batches—not
future task wording, rubrics, reference answers, base answers, or scores. Run:

```powershell
python .\scripts\audit_evaluation_suite.py --strict
```

The normal strict audit confirms the current suite did not regress below its 24-task floor. It
does **not** claim that the 100-task mature target has been reached. Use
`--require-mature-target` only as a future release gate.
