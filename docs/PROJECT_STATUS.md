# Project status

**Updated:** 2026-09-19

## Completed in this repository

- Project structure and storage rules for raw, generated, validated, training,
  evaluation, model, report, and legacy data.
- A 70-item project-authored training seed catalog with broad Roblox/Luau coverage.
- A separate 24-item project-authored held-out evaluation suite with rubrics, including explicit
  regression criteria for the reported RemoteEvent baseline errors.
- A 12-card source-checked Code Book proof of concept with schema, source policy, audit, and
  deterministic retrieval; it is not auto-training data.
- Generation, validation, correction, deduplication, final dataset, baseline,
  scoring/comparison, a one-task non-training proof-of-concept runner, hardware preflight,
  and guarded adapter-training scripts.
- Unit tests for key quality, split-isolation, deduplication, scoring, and dataset-build
  invariants.

## Deliberately not completed or claimed

- No generated candidates were produced by `qwen3:4b` in this repository checkout.
- No local baseline answer was captured, because the user's Windows Ollama installation is
  not available inside this repository workspace.
- No dataset has passed the complete live LLM-review pipeline yet.
- No LoRA or QLoRA adapter has been trained.
- No specialized model has been imported into Ollama.
- No improvement has been measured or claimed.

The latter items require the commands in the README to be executed against the real local
Ollama service and, for training, a suitable CUDA/cloud environment. Generated artifacts
are ignored by Git on purpose and must be preserved locally or in approved external
storage with their manifests.
