# DukeOTR project status

**Updated:** 2026-09-19

DukeOTR is the permanent public project/model identity for this Roblox/Luau specialization.
It is planned to derive from the existing pretrained `qwen3:4b` Ollama model and matching
`Qwen/Qwen3-4B` training base; it is not a language model trained from scratch. Those Qwen
names are technical provenance, not the public identity of a completed model.

## Completed in the tracked repository

- Storage rules that keep model weights, checkpoints, bulk generated data, local reports,
  and local Ollama artifacts out of Git by default.
- An 80-item broad, project-authored Roblox/Luau source-brief catalog and a separate 24-item
  held-out evaluation suite, plus an audited expansion plan toward a mature 100–300-test suite
  that intentionally does not pretend the target is already met.
- A 51-item DukeOTR Phase-1 Luau-fundamentals source-brief catalog plus a static curriculum
  audit covering required concepts, instructional modes, difficulty range, schema validity,
  duplicate prompts, and held-out prompt collisions.
- A combined 131-brief source-portfolio audit that verifies explicit task-form evidence
  (including critique, completion, subtle-bug, runtime, and insecurity analysis), conservative
  task-type concentration, and wording-level held-out isolation.
- A 25-card source-checked DukeOTR Code Book foundation, source policy, retrieval, and
  granular coverage audit for the requested Luau/Roblox concept floor. It is reference context
  and manual-curation input—not automatic training data.
- Generation, validation/review, correction, deduplication, final-dataset, baseline,
  evaluation/scoring/comparison, hardware-preflight, and guarded adapter-training scripts.
- A strict per-task JSON Schema path for the LLM evaluation scorer, with unit coverage for
  the observed RemoteEvent scoring-contract failure.
- A strict Ollama JSON Schema response contract for candidate generation, plus bounded raw
  response diagnostics when a model still produces malformed output.
- Versioned deterministic static checks that invalidate older validation results after a
  material rule change, rather than allowing a stale `static` pass into a later dataset build.
- Planned identity/version configuration for `dukeotr_dataset_v1`, `dukeotr_v1`, versioned
  candidate tag `dukeotr-v1`, and stable release alias `dukeotr`; all remain explicitly
  planned.
- A bounded Builder → Reviewer → Fixer trace runner with a configurable multi-round limit,
  structured correctness/security/API/requirements/English/code-quality findings, source-attributed
  Code Book context, evaluation-wording isolation, and explicit no-auto-promotion policy.
- A future-training handoff contract: versioned final datasets carry file hashes, an ignored
  transfer bundle can preserve data/config/provenance outside Git, and the guarded trainer
  rechecks its final-data manifest before real execution.

## Execution boundary

The Windows workstation retains its existing `qwen3:4b` Ollama model for inference, prompt
pilots, baseline/evaluation, and eventual DukeOTR candidate evaluation. Arena develops the
repository and quality gates but cannot access that local model or perform QLoRA training.
GitHub stores the reproducible source project, while the actual QLoRA run belongs on a suitable
CUDA/cloud machine using the matching Hugging Face `Qwen/Qwen3-4B` base. The final reviewed
JSONL stays ignored by Git and transfers to that machine separately with its manifest/hashes.
See [TRAINING_MACHINE_RUNBOOK.md](TRAINING_MACHINE_RUNBOOK.md).

## Local evaluation evidence (not tracked here)

The user captured one raw `qwen3:4b` response to the permanent RemoteEvent task in their
ignored Windows `reports/poc/` directory. That raw baseline must be preserved and never
regenerated/overwritten just to retry scoring.

The strict-schema scoring retry completed on the user's live Ollama server: the preserved
base-model record scored **30.0/100** with verdict `fail`, no deterministic flags, and a
322.316-second judge elapsed time. This is a valid measurement of the untouched base model,
not evidence of DukeOTR improvement.

The first one-item live generation pilot (`dukeotr_phase1_pilot_001`) failed before validation
because the base model response was not parseable JSON. The fresh `dukeotr_phase1_pilot_002`
run then verified the schema repair: generation, review, deduplication, and the local pilot
build all completed. That is evidence that the **structured-generation transport repair works**;
it is not evidence that its generated answer is suitable training data.

Manual inspection found that the single pilot answer incorrectly connected `OnServerEvent` in
a client-labeled section and passed `LocalPlayer` to `GetPlayerFromCharacter`; it also did not
actually show every requested basic type. This exposed a static-validation blind spot. The
current static checker now blocks those directionality/API errors, requires a current checker
version at deduplication/final-build time, and tightens the affected pure-Luau seed. Preserve
the ignored pilot artifacts as diagnostic evidence, but do not use its old
`training_data/dukeotr_phase1_pilot_002/` output for training or call it a dataset version.

## Deliberately not completed or claimed

- No Phase-1 or broad source brief is a finished reviewed training example by itself.
- No complete generation → review → correction → deduplication → final-dataset run has
  produced a versioned DukeOTR dataset.
- No `dukeotr_dataset_v1` artifact exists.
- No LoRA or QLoRA adapter has been trained.
- No `dukeotr_v1` adapter, `dukeotr-v1` versioned candidate, or `dukeotr` stable Ollama
  model has been created.
- No DukeOTR capability improvement has been measured or claimed.
- The bounded Builder → Reviewer → Fixer trace runner is not an autonomous self-training loop,
  not a source of automatic training-data promotion, and not evidence of a trained DukeOTR model.

The Windows system is appropriate for local Ollama inference, auditing, curation, and
assessment; actual 4B adapter training should occur on a suitable CUDA/cloud machine. See
[DUKEOTR_ROADMAP.md](DUKEOTR_ROADMAP.md) for staged exit gates and
[HARDWARE_AND_FINETUNING.md](HARDWARE_AND_FINETUNING.md) for the hardware boundary.
