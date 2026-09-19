# DukeOTR staged development roadmap

DukeOTR is the public identity of a planned Roblox/Luau specialization derived from the
existing pretrained `qwen3:4b` Ollama base model. It is not a language model trained from
scratch. The project name is intentional and must remain **DukeOTR** in model, dataset, Code
Book, report, adapter, and future Ollama-release identities; Qwen names remain technical
provenance only. See [MODEL_IDENTITY.md](MODEL_IDENTITY.md).

> Repository and clone paths may retain a historical directory name. That path is not the
> DukeOTR product/model name and is not a reason to rename the checked-out repository root
> or the existing Git history.

## Evidence rules

- A local `qwen3:4b` tag is the technical starting inference model; do not pull a second copy
  merely because a repository contains no weights.
- The Windows baseline artifact is local/ignored. It must be retained under `reports/` with
  its options and raw answer; Git should not receive model outputs by default.
- No statement may say `dukeotr_v1`, `dukeotr-v1`, or `dukeotr` was trained, imported into
  Ollama, released, or improved until a compatible adapter artifact and held-out comparison
  report exist.
- The Code Book is cited reference context, not automatic SFT material.

## Current repository foundation

The repository already has a staged candidate pipeline, 70 broad project-authored briefs,
24 held-out tasks, a source-checked Code Book POC, static safety gates, LoRA/QLoRA plans,
and baseline/evaluation runners. It does **not** yet contain a completed reviewed corpus or
a trained DukeOTR adapter.

The first local Windows POC has captured the one-task raw base-model answer under the user's
ignored `reports/poc/` directory. Its strict-schema score is a completed base-model baseline,
not evidence of a DukeOTR model. Candidate generation likewise uses a schema-constrained
envelope before any quality gate sees a response.

## Version identities

| Artifact | Planned identifier | Status |
|---|---|---|
| Curated first corpus | `dukeotr_dataset_v1` | Curriculum/seed expansion in progress |
| First adapter | `dukeotr_v1` | Not trained |
| Versioned candidate Ollama tag | `dukeotr-v1` | Not created |
| Stable public Ollama release tag | `dukeotr` | Not created; reserved until release gate |

Use a new manifest and changelog entry whenever one of these identifiers is advanced. A
version exists only after its referenced artifact is actually created.

## Stage 0 — reliable baseline and evaluation

1. Preserve the raw `qwen3:4b` answer to the permanent RemoteEvent test.
2. Score it with a schema-constrained judge where possible, while retaining deterministic
   RemoteEvent regression flags and raw judge failures for audit.
3. Capture the full 24-task baseline only after the one-task path is reliable on the local
   machine.
4. Keep all answer/score artifacts outside training input.

**Exit gate:** raw base answers exist locally with recorded model/options; no baseline score
is invented when a judge fails.

## Stage 1 — DukeOTR Luau fundamentals curriculum

`raw_data/dukeotr_phase1_luau_seed_tasks.jsonl` is the dedicated Phase-1 source-brief
catalog. It covers values/scope, operators/control flow, functions/returns, tables,
strings/math/randomness, types/annotations, ModuleScripts/`require`, error handling,
events/connections, callbacks, and closures.

The curriculum deliberately uses many instructional modes:

- question → answer;
- natural language → code;
- code → explanation;
- broken code → diagnosis/correction;
- requirements → implementation;
- code review;
- output prediction;
- beginner, intermediate, and advanced explanations;
- trade-off analysis; and
- refactoring/performance/lifecycle review.

Run small pilots first; do not request thousands of superficial variable-renaming variants.
The initial final-corpus target is **1,000–5,000 quality-gated examples**, reached through
careful source expansion, varied scenarios, human review samples, correction, and
measurement—not by treating every generated candidate as training data.

### Scale path without bulk repetition

1. Pilot one answer per selected brief and inspect failure/revision patterns before adding
   variants.
2. Add source briefs only to fill a recorded concept, mode, difficulty, system, or placement
   gap; vary requests, constraints, expected evidence, and realistic game context rather than
   identifiers alone.
3. Use small, deliberately different variants only where a brief can teach a genuinely
   distinct approach or learner need. Record provenance for every candidate.
4. Review accepted/revised/rejected rates by topic, then author targeted new briefs for weak
   areas. Do not extrapolate candidate count from prompt count as a quality claim.
5. Build a version candidate only after the final gate has sufficient breadth and a manifest;
   stop between 1,000 and 5,000 accepted examples when additional rows are no longer adding
   meaningful coverage or quality.

**Exit gate:** every required Phase-1 concept and instructional mode is represented in the
source catalog; generated records pass validation/review/dedupe; held-out tasks remain
separate.

## Stage 2 — Roblox API and placement fundamentals

Expand curated briefs for services, Instances, hierarchy/lifetime, UI, Character/Humanoid,
Tools, prompts/click detectors, CollectionService, TweenService, RunService,
UserInputService, ContextActionService, MarketplaceService, DataStoreService, Debris,
Teams, Lighting, Camera, and client/server script placement.

Every API example should state where its code belongs and why. An unfamiliar method/property
must be checked against official Roblox documentation before it is accepted as knowledge.

## Stage 3 — client/server architecture and security

Teach server authority, RemoteEvents/RemoteFunctions, `FireServer`, `OnServerEvent`,
`FireClient`, `OnClientEvent`, invoke boundaries, type/range/permission validation,
cooldowns, rate limits, ownership, and exploit resistance.

The following are permanent negative claims to identify and correct, never positive training
guidance:

- clients cannot call `RemoteEvent:FireServer`;
- RemoteEvents are secure by default or prevent exploits by themselves;
- the server should trust client price, currency, damage, target, or ownership data;
- the server must call `FireServer`; and
- a plausible-sounding API can be used without verification.

## Stage 4 — real game systems

Progress from small systems (doors, checkpoints, coins, leaderstats, prompts, tools, GUI)
to inventories, currencies, shops, NPCs, dialogue, rounds, waves, combat, quests,
DataStores, scalable UI, matchmaking, trading, optimization, and large modular
architectures. Complexity should increase only after the underlying authority and lifecycle
patterns are well represented.

## Later — Builder → Reviewer → Fixer

The future loop is:

```text
DukeOTR Builder → Reviewer → Fixer → reviewed revision → measured evaluation
```

It is intentionally not an autonomous self-training loop today. First require structured
traces, Code Book card revisions/citations, schema validation, human inspection points, and
strict isolation of held-out evaluation material. Only accepted, provenance-preserving
records can ever be proposed for manual training-data curation.

## Fine-tuning hardware boundary

The current Windows system is appropriate for local Ollama inference, Code Book work, data
review, and evaluation. It is not a dependable 4B QLoRA training host with integrated Intel
graphics and about 11.7 GB system RAM. Train only LoRA/QLoRA adapters on a suitable CUDA or
cloud environment using the matching Hugging Face `Qwen/Qwen3-4B` base, then verify
adapter/base compatibility before any Ollama import.
