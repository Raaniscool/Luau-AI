# DukeOTR implementation plan

## Repository assessment

The existing repository was inspected before this staged expansion. It already contains a
foundation pipeline, a broad 70-brief source catalog, 24 held-out evaluation tasks, a
15-card Code Book source-checked foundation, adapter configurations, and quality-gate tests. It does
not contain a completed reviewed corpus, a trained adapter, or a measured DukeOTR model.

The historical repository directory/name may remain in place; **DukeOTR** is the required
product/model identity. Compatible historic inputs remain in place rather than being deleted;
future compatible legacy material belongs in `legacy_data/` with provenance.

## Scope and non-goals

DukeOTR is the public identity of a specialization derived from the existing pretrained
`qwen3:4b` / `Qwen/Qwen3-4B` base. It does not train English or a language model from scratch.
The Qwen references are required technical provenance, not the finished assistant's public
name. The current implementation target is a clean, staged curriculum/data/evaluation
foundation—not a premature autonomous Builder/Reviewer/Fixer system and not a claimed
fine-tune.

## Staged deliverables

### 0. Baseline and evaluation integrity

- Preserve the original local base-model response and options as an evaluation artifact.
- Keep `evaluation_data/` prompts/rubrics, base answers, candidate answers, and score reports
  permanently out of generation and SFT inputs.
- Score the captured RemoteEvent answer only against the preserved raw answer, using a new
  output filename and the strict-schema scorer; the latest strict retry is a recorded
  30.0/100 base-model result, not a training result.
- Capture remaining base-model tasks only after the one-task judge path is reliable.

**Exit evidence:** recorded raw answers/options; no fabricated score when an evaluator fails.

### 1. Luau fundamentals curriculum (current)

- Maintain `raw_data/dukeotr_phase1_luau_seed_tasks.jsonl` as a source-brief catalog, not a
  completed corpus.
- Cover variables/types/scope, operators, conditionals, numeric/generic/while/repeat-until
  loops, functions, tables, strings, math/randomness, `type`/`typeof`, annotations,
  ModuleScripts/`require`, errors, events/connections, closures, and callbacks.
- Require varied modes: Q&A, natural-language-to-code, explanation, diagnosis/fix,
  requirements implementation, review, output prediction, refactoring, and trade-off
  discussion across learner levels.
- Run small pilot generations only after exact local `ollama list` preflight; generation uses
  Ollama JSON Schema mode for the response envelope, and malformed output is retained as a
  bounded diagnostic excerpt in the ignored report. Inspect failures and reviews before
  expanding anything.

**Exit evidence:** source audit passes; candidates have passed validation/review/correction and
dedupe; final records remain isolated from held-out evaluation.

### 2. Roblox placement and API fundamentals

Add source-checked curriculum for Instances/services, hierarchy/lifetime, LocalScripts,
Scripts, ModuleScripts, ReplicatedStorage, ServerStorage, ServerScriptService, StarterGui,
and StarterPlayerScripts. Verify unfamiliar Roblox API names and behavior against authoritative
documentation before accepting them into the Code Book or curated data.

### 3. Client/server architecture and security

Teach server authority as a core behavior, not a disclaimer:

- LocalScripts can call `RemoteEvent:FireServer`.
- RemoteEvents do not secure, authorize, or validate client input by themselves.
- Servers validate type, range, state, permission/ownership, cooldowns, and rate limits.
- Sensitive rewards, purchases, currency, inventory, combat, and progression state are
  server-owned.

Code Book revisions should trace these claims to official Roblox sources and reviewer checks.

### 4. Real game systems

Progress deliberately through small gameplay systems, inventory/currency/shop patterns,
NPCs/dialogue/combat, round/wave/quest systems, persistence, GUI, matchmaking/trading,
optimization, and modular architectures. Do not substitute breadth with lightly renamed
synthetic prompts.

### 5. Dataset version and adapter workflow

- Aim for approximately **1,000–5,000** quality-gated, diverse, reviewed examples before
  calling the first corpus `dukeotr_dataset_v1`.
- Create a dataset manifest/changelog only when the artifact actually exists.
- Run LoRA/QLoRA only on suitable CUDA/cloud hardware with the matching Hugging Face base.
- Record exact base revision, config, dataset manifest/hash, adapter output, training report,
  and held-out result before calling an artifact `dukeotr_v1`.
- Validate import compatibility before creating the planned versioned DukeOTR Ollama tag
  `dukeotr-v1`; only after held-out evaluation and human release approval may it receive the
  stable public alias `dukeotr`. Never commit model binaries by default.

### 6. Future Builder → Reviewer → Fixer

Keep the current role-contract configuration as a future seam. Before implementing an active
loop, require structured traces, citation-aware Code Book changes, approval checkpoints,
provenance-preserving revisions, and strict evaluation isolation. It must not automatically
promote its own outputs to SFT data.

## Quality gates

A generated record is not final simply because a model produced it. It requires:

1. canonical schema and project-authored source provenance;
2. deterministic structural/safety/API checks with no blocking finding;
3. recorded accepting LLM review or documented human acceptance;
4. exact/near-duplicate exclusion; and
5. held-out evaluation prompt collision exclusion.

Records requiring repair retain their parent IDs and stage metadata. `build_datasets.py`
refuses unresolved, unreviewed, rejected, duplicate, or evaluation-contaminated records.

## Hardware decision

The stated Windows machine (about 11.7 GB RAM and Intel integrated graphics with about 2 GB
shared/available VRAM) supports local Ollama inference, data curation, review, and evaluation.
It is not a dependable target for Qwen3-4B adapter training. Use a suitable CUDA/cloud host
for LoRA/QLoRA; the local scripts fail safely rather than turning a CPU/iGPU attempt into a
misleading training claim.
