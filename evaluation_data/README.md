# Held-out evaluation data

`roblox_luau_eval.jsonl` is DukeOTR's project-authored, rubric-driven, permanently held-out
suite. It contains 100 tasks across short-answer and deep-reasoning forms, beginner through
advanced difficulty, and the configured Luau/Roblox coverage tracks. Prompts, rubrics,
reference answers, model outputs, and score reports must never enter training generation,
Code Book retrieval, final training data, or role context.

`coverage_plan.json` records the completed 100-task mature-suite authoring floor and the
continuing 100–300 stewardship range. It is an inventory/coverage record only—not a model
run, score, baseline result, training result, or release claim.

```powershell
# Validates count, coverage tracks, task-form/depth/difficulty taxonomy, and suite isolation.
# It does not generate answers or score a model.
python .\scripts\audit_evaluation_suite.py --strict --require-mature-target
```
