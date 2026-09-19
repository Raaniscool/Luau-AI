# DukeOTR project status

**Updated:** 2026-09-19

DukeOTR is the permanent project/model identity for this Roblox/Luau specialization. It starts
from the existing pretrained `qwen3:4b` Ollama model and matching `Qwen/Qwen3-4B` training
base; it is not a language model trained from scratch.

## Completed in the tracked repository

- Storage rules that keep model weights, checkpoints, bulk generated data, local reports,
  and local Ollama artifacts out of Git by default.
- The original 70-item broad, project-authored Roblox/Luau source-brief catalog and a
  separate 24-item held-out evaluation suite.
- A 45-item DukeOTR Phase-1 Luau-fundamentals source-brief catalog plus a static curriculum
  audit covering required concepts, instructional modes, difficulty range, schema validity,
  duplicate prompts, and held-out prompt collisions.
- A 15-card source-checked DukeOTR Code Book foundation, source policy, audit, and retrieval.
  It is reference context and manual-curation input—not automatic training data.
- Generation, validation/review, correction, deduplication, final-dataset, baseline,
  evaluation/scoring/comparison, hardware-preflight, and guarded adapter-training scripts.
- A strict per-task JSON Schema path for the LLM evaluation scorer, with unit coverage for
  the observed RemoteEvent scoring-contract failure.
- Planned identity/version configuration for `dukeotr_dataset_v1`, `dukeotr_v1`, and
  `dukeotr-v1-qwen3-4b`; all remain explicitly planned.

## Local evaluation evidence (not tracked here)

The user captured one raw `qwen3:4b` response to the permanent RemoteEvent task in their
ignored Windows `reports/poc/` directory. That raw baseline must be preserved and never
regenerated/overwritten just to retry scoring.

Earlier score attempts did not produce a valid rubric score: the JSON-mode judge response
was syntactically JSON but emitted an answer-shaped `model_answer` object rather than the
required score object. The repository now contains a stricter schema-constrained retry path,
but its result on the user's live Ollama server has not yet been observed. A judge failure is
recorded evidence, not a reason to invent a score.

## Deliberately not completed or claimed

- No Phase-1 or broad source brief is a finished reviewed training example by itself.
- No complete generation → review → correction → deduplication → final-dataset run has
  produced a versioned DukeOTR dataset.
- No `dukeotr_dataset_v1` artifact exists.
- No LoRA or QLoRA adapter has been trained.
- No `dukeotr_v1` adapter or `dukeotr-v1-qwen3-4b` Ollama model has been created.
- No DukeOTR capability improvement has been measured or claimed.
- The future Builder → Reviewer → Fixer design is not an implemented autonomous loop.

The Windows system is appropriate for local Ollama inference, auditing, curation, and
assessment; actual 4B adapter training should occur on a suitable CUDA/cloud machine. See
[DUKEOTR_ROADMAP.md](DUKEOTR_ROADMAP.md) for staged exit gates and
[HARDWARE_AND_FINETUNING.md](HARDWARE_AND_FINETUNING.md) for the hardware boundary.
