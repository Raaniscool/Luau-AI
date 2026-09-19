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

Open PowerShell **in the cloned repository root**, not in your home directory. The
data/evaluation stages use only Python's standard library; no PyTorch install is needed for
them. Replace the example path below with the actual location of your clone.

```powershell
# This must be the folder that contains README.md, scripts, configs, and evaluation_data.
Set-Location "C:\path\to\Luau-AI"
Test-Path .\scripts\run_baseline.py  # Must print True before continuing.
# If it prints False, you are in the wrong folder or have not checked out the pipeline revision.

# Optional but recommended: isolate Python tooling.
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Verify the curated catalog and the held-out split before spending model time.
python .\scripts\audit_catalog.py --fail-on-missing

# Confirm the untouched base model is present.
ollama list
ollama run qwen3:4b "Reply with: ready"

# Capture the baseline BEFORE adapter training or creating a specialized tag.
python .\scripts\run_baseline.py --model qwen3:4b --output .\reports\evaluations\qwen3_4b_baseline.jsonl

# Score it (the judge model is recorded; review raw answers too).
python .\scripts\score_evaluation.py --answers .\reports\evaluations\qwen3_4b_baseline.jsonl --judge-model qwen3:4b
```

The baseline command records raw answers, settings, timing, model fingerprint when
available, and failures. It does not fabricate an answer if Ollama is unavailable.

### First data pilot

Start with a small pilot, inspect its artifacts, then scale deliberately:

```powershell
# Generate eight varied source briefs, validate them, repair revise/fail cases,
# revalidate repairs, deduplicate, and build only quality-gated records.
python .\scripts\run_pipeline.py --model qwen3:4b --limit 8 --variants 1

# Inspect the local reports and accepted final records before generating all 70 briefs.
Get-Content .\training_data\dataset_manifest.json
```

For explicit stage control:

```powershell
python .\scripts\generate_examples.py --model qwen3:4b --limit 8
python .\scripts\validate_examples.py --input .\generated_data\generated_examples.jsonl --model qwen3:4b
python .\scripts\correct_examples.py --input .\validated_data\validated_examples.jsonl --model qwen3:4b
python .\scripts\validate_examples.py --input .\validated_data\corrected_candidates.jsonl --output .\validated_data\corrected_validated_examples.jsonl --model qwen3:4b
python .\scripts\deduplicate_examples.py --input .\validated_data\validated_examples.jsonl --input .\validated_data\corrected_validated_examples.jsonl
python .\scripts\build_datasets.py --strict
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
`python .\scripts\record_human_review.py` to record named human decisions in an auditable
way (when run from the repository root). See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).

## Hardware and fine-tuning reality check

The stated machine (about 11.7 GB system RAM, Intel integrated graphics with about 2 GB
shared/available VRAM, Windows) is useful for **local Ollama inference and every data /
evaluation stage above**. It is not a dependable environment for the supplied 4B QLoRA
training configuration.

```powershell
python .\scripts\preflight_hardware.py --config .\configs\qlora_sft.json --require-suitable
python .\scripts\train_qlora.py  # writes a plan only; it does not train
```

Use a suitable CUDA/cloud machine for actual adapter training. The shipped config is
parameter-efficient QLoRA: 4-bit frozen base weights plus small trainable LoRA adapters,
not full-model modification. See [docs/HARDWARE_AND_FINETUNING.md](docs/HARDWARE_AND_FINETUNING.md)
for the local/cloud boundary, guarded training command, and Ollama import caveats.

## Measuring improvement after a real training run

Do not claim a capability gain just because an adapter trained without error. Run the same
held-out suite, score it, and compare it to the saved baseline:

```powershell
python .\scripts\run_evaluation.py --model your-specialized-ollama-tag --output .\reports\evaluations\specialized.jsonl
python .\scripts\score_evaluation.py --answers .\reports\evaluations\specialized.jsonl --judge-model qwen3:4b
python .\scripts\compare_reports.py --baseline .\reports\evaluations\qwen3_4b_baseline.scored.jsonl --candidate .\reports\evaluations\specialized.scored.jsonl
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
