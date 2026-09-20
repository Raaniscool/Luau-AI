# DukeOTR eight-month readiness assessment

**Assessment date:** 2026-09-19

**Scope:** tracked repository readiness for a future GPU/cloud LoRA/QLoRA phase. This is not a
training report and does not claim a DukeOTR model, adapter, dataset version, or score exists.

## Current evidence in the repository

| Readiness area | Current evidence | What it does **not** prove yet |
|---|---|---|
| Project identity and provenance | DukeOTR identity/version policy; `qwen3:4b` / `Qwen/Qwen3-4B` technical provenance; guarded future training/export documentation. | That `dukeotr`, `dukeotr-v1`, or `dukeotr_v1` exists. |
| Code Book | 25 source-checked, source-attributed cards; a strict audit verifies schema, source hosts, a 25-card floor, and a granular requested-concept matrix. | Human Studio verification of every card, exhaustive API coverage, or training-data eligibility. |
| Candidate source briefs | 51 dedicated Phase-1 Luau briefs and 80 broader project-authored briefs (131 total), with curriculum/catalog/portfolio audits and explicitly varied task forms. | A large, diverse, quality-gated candidate corpus; briefs are specifications, not examples. |
| Quality pipeline | Generation, static validation, structured review, correction, revalidation, deduplication, split-isolation, final-dataset build, manifests, and transfer-bundle scripts. | That a final quality-gated `dukeotr_dataset_v1` has been built. |
| Builder → Reviewer → Fixer | Deterministic conservative simple/normal/complex routing, task-appropriate tester/reviewer scopes, early-pass stopping, bounded deeper repair/retest rounds, provenance, Code Book context, and no-auto-promotion policy. | That any trace output is approved for SFT, that a configured pass is best possible, or that a model loop replaces human review. |
| Held-out evaluation | 100 permanently held-out rubric-driven tasks, deterministic RemoteEvent regression guards, baseline/evaluation/score/compare workflow, and a strict mature-floor/coverage audit. | Complete baseline capture, any new model score, an improvement claim, or a release decision. |
| Training handoff | Future-CUDA configs, hardware preflight, hash-verified final-data bundle, guarded QLoRA training script, adapter/export plan, and runbook. | A compatible adapter, GPU training run, or Ollama import. |

## What this change adds

1. **A real bounded adaptive quality-loop contract.** `scripts/run_builder_reviewer_fixer.py`
   deterministically routes a selected project-authored `train` brief before Builder work:
   narrow low-risk tasks use a minimal tester path, normal Luau/API tasks use standard review,
   and ambiguous/high-risk tasks use deeper bounded checks and repair/retest rounds. It records
   concrete correctness, security, API, requirement, English, code-quality, style, testability,
   and Code Book findings where relevant. A configured passing response stops early without a
   default Fixer call; it is never called best possible, never becomes a final training dataset,
   and keeps prior answers as correction parents.
2. **Evaluation isolation at the role boundary.** The quality loop performs a local
   wording-level collision check before inference. Evaluation prompt/rubric/answer text is
   never passed to a Builder, Reviewer, or Fixer. A collision blocks the trace.
3. **Broader source-checked Code Book coverage.** New cards cover values/scope/nil/operators,
   strings/math/randomness, typed tables/unions/callbacks, players/characters, GUI/input,
   replication containers, raycasting/physics validation, TweenService/RunService, API
   hallucination review, and hierarchy/service access. Existing cards retain their history;
   touched taxonomy revisions are explicit.
4. **A broader and auditable source-brief portfolio.** Six new Phase-1 and ten broader
   project-authored briefs add explicit bad-answer critique, completion, subtle-bug analysis,
   runtime reasoning, diagnosis/correction, API-misuse diagnosis, and insecurity analysis.
   `audit_source_portfolio.py` ensures all requested task forms have evidence without claiming
   that a model has generated or approved examples.
5. **A mature held-out-suite stewardship audit.** `evaluation_data/coverage_plan.json` and
   `scripts/audit_evaluation_suite.py` now verify 100 independently authored tasks, coverage
   floors, task forms, short/deep depth, and difficulty taxonomy without fabricating a model
   score or leaking hidden evaluation content.

## Deliberate gaps before summer GPU access

### 1. Curated source breadth and corpus scale

The 131 tracked source briefs are a useful base, but they are not the requested large,
diverse candidate corpus. Expand by documented gaps, learner level, task form, game context,
and failure mode—not synonym swaps. Keep natural English quality and code/API review as first
class gates. Use small inspected local inference pilots only; do not make slow laptop bulk
inference a prerequisite.

### 2. Code Book depth and human verification

The 25-card floor proves only traceable coverage labels and source attribution. Add narrow
cards when a real gap appears (for example DataStore lifecycle, physics/network ownership,
or current API changes), then record named human verification for high-risk cards after
checking current docs and Studio behavior. Do not auto-convert cards into SFT rows.

### 3. Held-out evaluation stewardship and baseline discipline

The independently authored 100-task suite reaches the current mature authoring floor, while
the 100–300 stewardship range remains open for genuinely new coverage gaps. Preserve its
permanent separation: prompt wording, rubrics, expected answers, baseline answers, and score
reports stay out of every source brief, Code Book role context, and training artifact. The
completed authoring audit is not baseline evidence; Windows-side model evaluation remains a
separate deliberate activity.

### 4. Baseline completeness and comparison discipline

Preserve the untouched `qwen3:4b` baseline workflow and local raw outputs. Capture more
baseline tasks only through deliberate Windows-side evaluation, preserving exact model/options
and never overwriting original evidence. Do not call a model better based on one task or a
single judge score.

### 5. Final-data and training evidence

No final `dukeotr_dataset_v1`, LoRA/QLoRA adapter, `dukeotr_v1`, `dukeotr-v1`, or stable
`dukeotr` tag exists. Only after a real manifest-gated dataset, suitable CUDA/cloud training,
compatible adapter/export, full held-out comparison, and human release decision may those
names advance beyond plans.

## Recommended order of work before GPU access

1. Run the offline audits and inspect their reports:
   `audit_dukeotr_curriculum.py`, `audit_catalog.py`, `audit_code_book.py --strict`, and
   `audit_evaluation_suite.py --strict --require-mature-target`.
2. Use `run_builder_reviewer_fixer.py --dry-run` to inspect trace planning. On the Windows
   machine, run only a few manually inspected `qwen3:4b` quality-loop pilots after `ollama
   list` confirms the existing tag.
3. Convert observed reviewer/static failure patterns into narrowly authored source briefs,
   Code Book revisions, and deterministic checks where appropriate.
4. Add a permanently held-out task only for a documented new coverage gap; do not dilute the suite with superficial variants.
5. Build a final dataset only after all documented gates are met, then package it with its
   manifest for a suitable future CUDA/cloud host.

## Non-claims

- No local Ollama inference, bulk generation, model download, adapter training, model export,
  or DukeOTR evaluation was performed merely by adding the readiness infrastructure described
  here.
- A trace status of `accepted_for_manual_review_only` is not a dataset, an SFT approval, or a
  model-quality claim.
- The planned user-facing command remains `ollama run dukeotr` only after a real approved
  release; today the technical local base remains `ollama run qwen3:4b` where it is available.
