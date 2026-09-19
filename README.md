# Luau-AI — Roblox/Luau specialization pipeline

This project specializes an **already pretrained** language model for Roblox development;
it does **not** train a language model or English from scratch. The intended base for the
first experiment is the local Ollama tag `qwen3:4b`, with `Qwen/Qwen3-4B` as the matching
Hugging Face training base when an actual adapter-training environment is available.

> **Current status — Phase 1 infrastructure is implemented; no fine-tuning has been run.**
> The repository contains 70 curated training task briefs and 24 separate held-out
> evaluation tasks. Generated model answers, reviews, datasets, checkpoints, baseline
> outputs, and training reports are intentionally local/ignored artifacts until someone
> runs the pipeline. See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).

## What is included

- A broad, project-authored Roblox/Luau seed catalog covering syntax, APIs, remotes,
  client/server design, exploit resistance, UI, persistence, systems, debugging, and
  advanced architecture.
- A staged, auditable workflow:
  **generation → validation → correction → re-validation → deduplication → final dataset**.
- Static guards plus a separate structured LLM review. A record cannot enter final data
  without a static pass, a recorded accepting reviewer, deduplication, and a held-out
  prompt leakage check.
- A 24-task held-out rubric suite, including the requested prompt:
  _“Explain what a RemoteEvent is in Roblox and show a secure example.”_
- Baseline/candidate runners for the local Ollama API, LLM-rubric scoring, and a factual
  comparison report.
- QLoRA/LoRA preparation with a CUDA hardware guard; it refuses to present a CPU or small
  integrated-GPU attempt as a real fine-tuning run.
- An interface contract for a future builder → verifier → reviewer system, without making
  that larger system block the foundational pipeline.

## Repository layout

```text
raw_data/          Curated source briefs and provenance (tracked)
generated_data/    Stage 1 local candidate outputs (ignored)
validated_data/    Review, correction, and dedupe artifacts (ignored)
training_data/     Quality-gated final/train/development JSONL (ignored)
evaluation_data/   Held-out prompts and scoring rubrics (tracked; never training input)
scripts/           Dependency-light pipeline, evaluation, and training entry points
models/            Local adapters/checkpoints/export artifacts (ignored)
configs/           Pipeline and LoRA/QLoRA configuration (tracked)
reports/           Local baseline/evaluation/training reports (ignored)
legacy_data/       Archive location for compatible previous material
```

## Quick start on the Windows machine with Ollama

Open PowerShell in the cloned repository. The data/evaluation stages use only Python's
standard library; no PyTorch install is needed for them.

```powershell
# Optional but recommended: isolate Python tooling.
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Verify the curated catalog and the held-out split before spending model time.
python -m scripts.audit_catalog --fail-on-missing

# Confirm the untouched base model is present.
ollama list
ollama run qwen3:4b "Reply with: ready"

# Capture the baseline BEFORE adapter training or creating a specialized tag.
python -m scripts.run_baseline --model qwen3:4b `
  --output reports/evaluations/qwen3_4b_baseline.jsonl

# Score it (the judge model is recorded; review raw answers too).
python -m scripts.score_evaluation `
  --answers reports/evaluations/qwen3_4b_baseline.jsonl `
  --judge-model qwen3:4b
```

The baseline command records raw answers, settings, timing, model fingerprint when
available, and failures. It does not fabricate an answer if Ollama is unavailable.

### First data pilot

Start with a small pilot, inspect its artifacts, then scale deliberately:

```powershell
# Generate eight varied source briefs, validate them, repair revise/fail cases,
# revalidate repairs, deduplicate, and build only quality-gated records.
python -m scripts.run_pipeline --model qwen3:4b --limit 8 --variants 1

# Inspect the local reports and accepted final records before generating all 70 briefs.
Get-Content training_data\dataset_manifest.json
```

For explicit stage control:

```powershell
python -m scripts.generate_examples --model qwen3:4b --limit 8
python -m scripts.validate_examples --input generated_data/generated_examples.jsonl --model qwen3:4b
python -m scripts.correct_examples --input validated_data/validated_examples.jsonl --model qwen3:4b
python -m scripts.validate_examples --input validated_data/corrected_candidates.jsonl `
  --output validated_data/corrected_validated_examples.jsonl --model qwen3:4b
python -m scripts.deduplicate_examples `
  --input validated_data/validated_examples.jsonl `
  --input validated_data/corrected_validated_examples.jsonl
python -m scripts.build_datasets --strict
```

If no records need correction, the correction artifact is intentionally empty; skip its
re-validation input in that case. `run_pipeline.py` handles this automatically.

## Quality rules

A generated answer is **not** automatically training data. Final construction enforces:

1. canonical chat-record schema and source seed provenance;
2. deterministic safety/structure/API heuristics with no blocking finding;
3. an accepting structured LLM review, or a documented human acceptance;
4. exact/near-duplicate exclusion; and
5. held-out evaluation prompt collision exclusion.

The LLM reviewer is an additional check, not a substitute for expert review. Use
`python -m scripts.record_human_review` to record named human decisions in an auditable
way. See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).

## Hardware and fine-tuning reality check

The stated machine (about 11.7 GB system RAM, Intel integrated graphics with about 2 GB
shared/available VRAM, Windows) is useful for **local Ollama inference and every data /
evaluation stage above**. It is not a dependable environment for the supplied 4B QLoRA
training configuration.

```powershell
python -m scripts.preflight_hardware --config configs/qlora_sft.json --require-suitable
python -m scripts.train_qlora  # writes a plan only; it does not train
```

Use a suitable CUDA/cloud machine for actual adapter training. The shipped config is
parameter-efficient QLoRA: 4-bit frozen base weights plus small trainable LoRA adapters,
not full-model modification. See [docs/HARDWARE_AND_FINETUNING.md](docs/HARDWARE_AND_FINETUNING.md)
for the local/cloud boundary, guarded training command, and Ollama import caveats.

## Measuring improvement after a real training run

Do not claim a capability gain just because an adapter trained without error. Run the same
held-out suite, score it, and compare it to the saved baseline:

```powershell
python -m scripts.run_evaluation --model your-specialized-ollama-tag `
  --output reports/evaluations/specialized.jsonl
python -m scripts.score_evaluation `
  --answers reports/evaluations/specialized.jsonl `
  --judge-model qwen3:4b
python -m scripts.compare_reports `
  --baseline reports/evaluations/qwen3_4b_baseline.scored.jsonl `
  --candidate reports/evaluations/specialized.scored.jsonl
```

Read the raw outputs and all critical failures, especially security tasks, before relying
on an aggregate score. Details: [docs/EVALUATION.md](docs/EVALUATION.md).

## Development checks

```bash
python -m compileall -q scripts
python -m scripts.audit_catalog --fail-on-missing
python -m unittest discover -v
```

## Next incremental milestone

Once a reviewed dataset, a recorded baseline, and a real adapter evaluation show a useful
improvement, implement the future role contract in
[docs/FUTURE_BUILDER_VERIFIER_REVIEWER.md](docs/FUTURE_BUILDER_VERIFIER_REVIEWER.md):
Builder → Verifier → Reviewer → Builder revision. Do not make this more complicated
system a substitute for high-quality data and measured evaluation.
