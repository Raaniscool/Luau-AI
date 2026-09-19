# DukeOTR — Roblox/Luau specialization pipeline

**DukeOTR** is a staged Roblox-development and Luau specialization built on the existing
pretrained local Ollama model `qwen3:4b` (matching Hugging Face training base:
`Qwen/Qwen3-4B`). It does **not** train a language model—or basic English—from scratch.

> The checked-out GitHub repository and local clone directory may retain their historical
> names for continuity. The permanent project, dataset, adapter, Code Book, and model
> identity is **DukeOTR**.

> **Current status — foundation and curriculum work only; no DukeOTR adapter is trained.**
> The repository contains 70 broad project-authored Roblox/Luau source briefs, a dedicated
> 45-brief DukeOTR Phase-1 Luau-fundamentals curriculum, a 24-task held-out evaluation
> suite, and a 15-card source-checked Code Book foundation. None of those source
> briefs is a completed training dataset. Generated candidates, accepted records, adapters,
> and model claims require the documented gates and evidence.

Read the current non-claims in [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) and the
staged curriculum in [docs/DUKEOTR_ROADMAP.md](docs/DUKEOTR_ROADMAP.md).

## DukeOTR sequence

```text
Qwen3-4B
  → verified Roblox/Luau source data + DukeOTR Code Book
  → Builder → Reviewer → Fixer
  → reviewed data / measured adapter candidate
  → held-out evaluation → repeat
```

The Builder/Reviewer/Fixer loop is a future interface boundary, not an implemented autonomous
self-training system. The present work intentionally starts with a high-quality curriculum,
provenance, validation, review, correction, deduplication, final-dataset gates, and an
untouched evaluation boundary.

## What is included

- **DukeOTR identity/version contract** in `configs/dukeotr_project.json`:
  planned `dukeotr_dataset_v1`, `dukeotr_v1`, and `dukeotr-v1-qwen3-4b` names are clearly
  marked as planned—not created.
- **Phase-1 Luau fundamentals curriculum** in
  `raw_data/dukeotr_phase1_luau_seed_tasks.jsonl`, covering values/types/scope, operators,
  conditionals, numeric/generic/while/repeat-until loops, functions, tables, strings,
  math/randomness, runtime checks, annotations, modules, errors, events/connections,
  closures, and callbacks.
- Multiple instructional modes: Q&A, natural language → code, code explanation, broken-code
  repair, requirements implementation, review, output prediction, refactoring, and trade-off
  analysis across beginner, intermediate, and advanced briefs.
- A staged, auditable workflow:
  **generation → validation → review → correction → re-validation → deduplication → final dataset**.
- A 24-task held-out rubric suite and original base-model baseline path that never become
  training input.
- A cited, manually curated DukeOTR Code Book proof of concept; it is reference context,
  not automatic SFT material.
- Conservative LoRA/QLoRA configuration for a suitable CUDA/cloud environment, with model
  binaries, checkpoints, generated corpora, and local reports ignored by Git.

## Repository layout

```text
raw_data/          Project-authored source briefs, including the DukeOTR Phase-1 catalog
legacy_data/       Archive location for compatible historic material; do not discard useful data
generated_data/    Local candidate outputs (ignored; never automatically final)
validated_data/    Local review/correction/dedupe artifacts (ignored)
training_data/     Local quality-gated dataset versions only (ignored)
evaluation_data/   Held-out prompts/rubrics (tracked; permanently excluded from training)
code_book/         Source-attributed Roblox/Luau cards (tracked; manual curation only)
configs/           DukeOTR, pipeline, and adapter-planning configurations
scripts/           Dependency-light pipeline, evaluation, and training entry points
models/            Ignored adapters/checkpoints/export artifacts
reports/           Ignored baseline/evaluation/training reports
```

## Start safely on the Windows machine

Open PowerShell **in the cloned repository root**—the folder containing `README.md`,
`scripts`, `configs`, and `evaluation_data`. The directory can retain its historical clone
name; it does not rename DukeOTR.

```powershell
Set-Location "C:\path\to\your-existing-clone"
Test-Path .\scripts\audit_dukeotr_curriculum.py  # Must print True.

# Optional but recommended for local Python tooling.
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# These are static audits; no model is contacted.
python .\scripts\audit_dukeotr_curriculum.py --strict
python .\scripts\audit_catalog.py --fail-on-missing
python .\scripts\audit_code_book.py --strict
```

### Required preflight before any model-facing command

The local Ollama installation and the **exact existing** tag must be checked first. Do not
pull or download another Qwen copy just because no model file appears in this repository.

```powershell
ollama list
python .\scripts\check_ollama.py --model qwen3:4b
```

All local model-facing scripts also enforce this registration check before generation,
review, scoring, baseline capture, or evaluation. It never performs `ollama pull`.

## Preserve and score the baseline

The raw base-model answer is an evaluation artifact, not training material. Never overwrite
a captured POC answer merely to retry scoring. If the one-task raw baseline already exists,
run the schema-constrained scorer against it with a **new** output name:

```powershell
python .\scripts\score_evaluation.py `
  --answers .\reports\poc\qwen3_4b_remoteevent_baseline.jsonl `
  --judge-model qwen3:4b `
  --output .\reports\poc\qwen3_4b_remoteevent_baseline.scored_schema_retry.jsonl `
  --report .\reports\poc\qwen3_4b_remoteevent_baseline.scored_schema_retry.score_report.json `
  --num-predict 700 --num-ctx 4096 --timeout-seconds 1800
```

Keep raw answers, judge responses, options, timing, and failures. A failed judge does not
license a fabricated score. See [docs/EVALUATION.md](docs/EVALUATION.md) for the held-out
contract and score interpretation.

If no baseline exists yet, capture it only after the preflight above. The baseline runner
uses the original `qwen3:4b` tag and records raw output/settings; it is not a training step.

```powershell
python .\scripts\run_baseline.py --model qwen3:4b --output .\reports\evaluations\qwen3_4b_baseline.jsonl
```

## Phase-1 curriculum and small data pilot

First inspect the planned workflow without running Ollama:

```powershell
# Defaults to raw_data/dukeotr_phase1_luau_seed_tasks.jsonl.
# Use a unique run id; it keeps ignored artifacts separate and will not overwrite prior work.
python .\scripts\run_pipeline.py --model qwen3:4b --limit 8 --variants 1 --run-id dukeotr_phase1_plan_001 --dry-run
```

After auditing the plan, passing `ollama list`, and deciding to spend local model time, run a
small pilot—not thousands of superficial variants:

```powershell
# Candidate generation only; this is still not a final dataset or a fine-tune.
python .\scripts\run_pipeline.py --model qwen3:4b --limit 8 --variants 1 --run-id dukeotr_phase1_pilot_001
```

The pipeline preserves the stage sequence:

1. `generate_examples.py` emits candidate conversations with source/model provenance.
2. `validate_examples.py` runs deterministic checks and structured review.
3. `correct_examples.py` repairs revisable candidates while preserving parent provenance.
4. Re-validation and `deduplicate_examples.py` reject unresolved or too-similar records.
5. `build_datasets.py` permits only quality-gated, evaluation-isolated rows into final data.

A run ID writes separate ignored paths such as
`generated_data/dukeotr_phase1_pilot_001.generated.jsonl` and
`training_data/dukeotr_phase1_pilot_001/`. The orchestrator refuses to overwrite prior paths;
choose a new run ID or archive compatible material in `legacy_data/` before any deliberate
overwrite.

The meaningful first corpus target is **1,000–5,000 high-quality reviewed examples**, not a
bulk dump. The 45 dedicated Phase-1 briefs plus the prior broad catalog are curated starting
points; generated output counts do not count toward a dataset version until every quality
and isolation gate passes.

For explicit stage control, pick a unique output prefix deliberately:

```powershell
$RunId = "dukeotr_phase1_manual_001"
python .\scripts\generate_examples.py --seeds .\raw_data\dukeotr_phase1_luau_seed_tasks.jsonl --model qwen3:4b --limit 8 --output ".\generated_data\$RunId.generated.jsonl"
python .\scripts\validate_examples.py --input ".\generated_data\$RunId.generated.jsonl" --output ".\validated_data\$RunId.validated.jsonl" --model qwen3:4b
python .\scripts\correct_examples.py --input ".\validated_data\$RunId.validated.jsonl" --output ".\validated_data\$RunId.corrected.jsonl" --model qwen3:4b
python .\scripts\validate_examples.py --input ".\validated_data\$RunId.corrected.jsonl" --output ".\validated_data\$RunId.corrected_validated.jsonl" --model qwen3:4b
python .\scripts\deduplicate_examples.py --input ".\validated_data\$RunId.validated.jsonl" --input ".\validated_data\$RunId.corrected_validated.jsonl" --output ".\validated_data\$RunId.deduplicated.jsonl"
python .\scripts\build_datasets.py --input ".\validated_data\$RunId.deduplicated.jsonl" --output-dir ".\training_data\$RunId" --strict
```

If correction produces no records, skip the corrected-candidate revalidation input rather
than treating a missing file as a final dataset. `run_pipeline.py` handles that branch.

## Code Book and Roblox security

The Code Book is source-attributed reference context for future Builder/Reviewer/Fixer work.
It is not silently copied into SFT data. Audit or query it without calling a model:

```powershell
python .\scripts\audit_code_book.py --strict
python .\scripts\query_code_book.py --query "secure RemoteEvent shop purchase" --format markdown
```

DukeOTR's later client/server curriculum keeps these rules explicit:

- a LocalScript can call `RemoteEvent:FireServer`;
- a RemoteEvent is not authorization or security by default;
- servers validate type, range, permission/ownership, state, cooldown, and rate limits; and
- sensitive currency, inventory, combat, purchase, and progression state remains server-owned.

Use authoritative Roblox documentation for API claims. Do not invent API names or treat a
plausible-looking method as verified.

## Fine-tuning boundary

The stated Windows system (about 11.7 GB RAM with Intel integrated graphics and about 2 GB
shared/available VRAM) is suitable for local Ollama inference and data/evaluation work. It
is **not** a dependable Qwen3-4B QLoRA training host. Train adapters only on a suitable
CUDA/cloud environment using the exact matching `Qwen/Qwen3-4B` base.

```powershell
python .\scripts\preflight_hardware.py --config .\configs\qlora_sft.json --require-suitable
python .\scripts\train_qlora.py  # plan/validation only; does not train without --execute
```

`configs/qlora_sft.json` and `configs/lora_sft.json` describe planned output names under
`dukeotr_v1`; they do not mean an adapter exists. Never commit large model binaries or
checkpoints without an explicit reason and storage plan.

## Measuring a real DukeOTR candidate

Only after a versioned final dataset, actual compatible adapter training/import, and a
candidate Ollama tag exist, evaluate the candidate against the same held-out suite and
compare raw outputs and critical failures:

```powershell
python .\scripts\run_evaluation.py --model dukeotr-v1-qwen3-4b --output .\reports\evaluations\dukeotr_v1.jsonl
python .\scripts\score_evaluation.py --answers .\reports\evaluations\dukeotr_v1.jsonl --judge-model qwen3:4b
python .\scripts\compare_reports.py `
  --baseline .\reports\evaluations\qwen3_4b_baseline.scored.jsonl `
  --candidate .\reports\evaluations\dukeotr_v1.scored.jsonl
```

Do not claim improvement because training exited successfully. Review the raw held-out
answers, score traces, and security failures before describing any capability change.

## Development checks

```bash
python -m compileall -q scripts
python scripts/audit_dukeotr_curriculum.py --strict
python scripts/audit_catalog.py --fail-on-missing
python scripts/audit_code_book.py --strict
python -m unittest discover -v
```
