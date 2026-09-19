# Baseline, evaluation, scoring, and comparison

## Why baseline comes first

A fine-tuned adapter has value only if it improves the capabilities that matter without
regressing security or general instruction following. Run commands from the repository root
(the directory containing `scripts/`), then record the untouched base tag before training or
creating a custom Ollama model:

```powershell
Test-Path .\scripts\run_baseline.py  # Must print True.
# Verification only: runs `ollama list`; it does not download or run a model.
python .\scripts\check_ollama.py --model qwen3:4b
python .\scripts\run_baseline.py --model qwen3:4b --output .\reports\evaluations\qwen3_4b_baseline.jsonl
```

`run_baseline.py` sends only task prompts (not hidden rubrics) to Ollama, with deterministic
settings by default (`temperature=0`, seed `3407`). It stores the complete original answer
for each task together with task ID, model tag, available model fingerprint, settings,
timing, and an error record when a task cannot run.

The first task is exactly:

> Explain what a RemoteEvent is in Roblox and show a secure example.

For an incremental proof before the full 24-task capture, run:

```powershell
python .\scripts\run_first_poc.py --model qwen3:4b
```

It first runs `ollama list`, audits the Code Book, captures only this task, and scores it.
It uses streamed responses so a slow local CPU does not wait for one giant HTTP response,
with compact 900-token baseline / 700-token judge budgets, a 4K context, and a 30-minute
per-chunk timeout by default. It writes the raw answer and score under `reports/poc/` and
refuses to replace an existing raw POC answer unless you explicitly add `--overwrite`. It
does not generate a training corpus or start fine-tuning.

The baseline runner cannot prove a user-created tag was unmodified. Use the downloaded
`qwen3:4b` tag before any `ollama create` work, and retain the raw output file.

## Held-out suite

`evaluation_data/roblox_luau_eval.jsonl` contains 24 rubric-driven tasks spanning:

- RemoteEvents, RemoteFunctions, secure shop/currency flows, and Instance validation;
- client/server combat, tools, UI inputs, leaderstats, rounds, and matchmaking;
- DataStore failure safety, gamepasses, developer product receipts, and architecture;
- Luau tables, modules/OOP cleanup, NPC pathfinding, debugging, and performance; and
- advanced streaming and future builder/verifier design.

It is a tracked, project-authored benchmark. Its prompts/rubrics are intentionally never
loaded by `generate_examples.py` or `train_qlora.py`.

## Score a run

```powershell
python .\scripts\score_evaluation.py --answers .\reports\evaluations\qwen3_4b_baseline.jsonl --judge-model qwen3:4b
```

The scorer gives the judge the answer and rubric only after generation is complete. It
requires all rubric criteria to be scored, recomputes the total from criterion points, and
records critical failures separately. It also runs narrow deterministic regression checks for
the RemoteEvent baseline task: an answer that says clients cannot call `FireServer`, that
the server must call `FireServer`, or that RemoteEvents are inherently/automatically secure
receives a release-fail verdict even if the judge misses it. It labels the remaining method
honestly: **LLM-as-judge is repeatable triage, not ground truth.**

For higher confidence, use a different qualified judge model and/or independent human
review of all security tasks. Do not optimize solely for a single judge score.

## Evaluate an actual candidate

After a successful real adapter export/import, run the same suite without changing its
prompts:

```powershell
python .\scripts\run_evaluation.py --model luau-ai-qwen3-4b --output .\reports\evaluations\luau_ai_candidate.jsonl

python .\scripts\score_evaluation.py --answers .\reports\evaluations\luau_ai_candidate.jsonl --judge-model qwen3:4b

python .\scripts\compare_reports.py --baseline .\reports\evaluations\qwen3_4b_baseline.scored.jsonl --candidate .\reports\evaluations\luau_ai_candidate.scored.jsonl
```

The comparison report includes per-task score deltas, category means, missing tasks,
critical-failure counts, and an explicit interpretation warning. It never manufactures an
improvement result. A credible claim should include:

1. complete baseline and candidate coverage of the same tasks;
2. no unresolved regressions in remote/security/persistence tasks;
3. raw-answer inspection for material wins and failures;
4. the model tags, base/adapter version, generation settings, dataset manifest hash, and
   judge method; and
5. ideally a human review sample or a second independent judge.

## Regression policy suggestion

Do not promote a candidate merely because mean score increases. Set a project release gate
such as: no new critical security failures; no meaningful decline in persistence or remote
security tasks; and a review of every task with a large positive or negative delta. Keep
baseline and candidate report files outside training inputs forever.
