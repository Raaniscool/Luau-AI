# Phase 1 implementation plan

## Repository assessment

The repository was inspected before implementation on 2026-09-19. It contained only
`README.md`; there were no existing datasets, model files, scripts, or legacy artifacts
to move. The `legacy_data/` directory is retained as the future archive location so that
useful prior material is never overwritten or silently mixed into a new corpus.

## Scope and non-goals

This phase specializes an **already pretrained** `qwen3:4b` model. It does not train a
language model from scratch and it does not represent a completed fine-tuning run. The
first deliverable is an auditable data, evaluation, and fine-tuning preparation pipeline.

## Deliverables and implementation order

1. **Project contract and safe storage**
   - Create the requested data, model, configuration, script, and report directories.
   - Keep model weights, checkpoints, generated bulk data, and local Ollama files out of
     Git.
   - Define one versioned JSONL schema for all training examples and one for held-out
     evaluation tasks.

2. **Diverse source specifications**
   - Write a varied catalog of Roblox/Luau task briefs across syntax, APIs, networking,
     security, systems, debugging, and architecture.
   - Keep evaluation briefs in `evaluation_data/` from the beginning. Generation scripts
     only read training briefs by default.
   - Include task type, difficulty, constraints, expected evidence, tags, and source
     provenance, rather than producing near-duplicate prompts.

3. **Staged data pipeline**
   - `generate_examples.py`: asks a local Ollama model for a structured instructional
     example from a source brief.
   - `validate_examples.py`: performs deterministic schema/safety/API heuristics and a
     separate structured LLM review. An example cannot be eligible for final training
     data without recorded validation.
   - `correct_examples.py`: sends only failed or revision-needed examples and their
     review findings through a correction prompt; corrected records preserve parent
     provenance and must be validated again.
   - `deduplicate_examples.py`: removes exact and near duplicates with a documented,
     deterministic similarity method.
   - `build_datasets.py`: makes the canonical chat-format training JSONL only from
     quality-gated, deduplicated records and emits a manifest/contamination report.

4. **Evaluation before and after training**
   - Provide a held-out rubric suite, including the requested secure `RemoteEvent`
     question.
   - `run_baseline.py` captures the untouched local Ollama model's raw answers with
     version, parameters, timestamps, and failures.
   - `score_evaluation.py` and `compare_reports.py` create per-task and aggregate
     baseline-vs-candidate reports. Results are labelled as unavailable until actually
     run; no synthetic score is substituted.

5. **Fine-tuning preparation**
   - Ship a QLoRA/LoRA SFT configuration and an explicit hardware preflight command.
   - Use a frozen base model and train adapters only when a suitable accelerator exists.
   - Prepare an optional export path for an adapter or merged model to Ollama, while
     requiring compatibility verification against the exact base model.

6. **Verification and future architecture seam**
   - Add unit tests for schema gates, split isolation, deduplication, and report
     comparison.
   - Add interfaces and prompts for a future builder → verifier → reviewer loop, but do
     not make it a prerequisite for the foundational data/evaluation pipeline.

## Quality gates

A generated record is **not** final merely because a model produced it. It must have:

- a valid canonical schema and provenance;
- static checks with no blocking issue;
- an independently recorded LLM reviewer decision of `accept` (or a documented human
  approval); and
- no exact or above-threshold near duplicate in its split or the held-out evaluation
  prompts.

Records needing correction stay in stage-specific artifacts. `build_datasets.py` refuses
to include unreviewed, rejected, unresolved, or duplicate records.

## Hardware decision

The stated Windows machine (about 11.7 GB RAM and Intel integrated graphics with about
2 GB shared/available VRAM) is appropriate for local Ollama inference and the data/
evaluation pipeline, but is not a practical target for a dependable Qwen3-4B fine-tuning
run. The project will therefore support:

- **local:** generation, review, validation, dataset assembly, baseline/candidate
  inference and scoring through Ollama;
- **GPU/cloud:** QLoRA/LoRA adapter training, checkpointing, and model merging/export.

The code will detect the environment and fail safely rather than falling back to a
misleading full CPU training attempt.
