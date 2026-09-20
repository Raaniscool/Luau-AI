# DukeOTR Builder → Reviewer → Fixer quality loop

> Historical filename retained for links. The repository now implements a **bounded,
> trace-producing quality-loop orchestrator**; it is not an autonomous agent, a training-data
> promotion path, or evidence that DukeOTR has been trained.

## What is implemented

`configs/builder_verifier_reviewer.json` is the machine-readable contract and
`python -m scripts.run_builder_reviewer_fixer` executes one carefully selected **train** brief
through these roles:

```text
project-authored train brief
  ↓  (held-out wording isolation check; no evaluation text reaches a role)
deterministic conservative route: simple | normal | complex
  ↓
Builder → task-appropriate deterministic tester → 0 / 1 / 2 independent Reviewer pass(es)
  ↓ pass: stop early, with no default Fixer or extra review
  ↓ revise only, bounded by the configured route/round count
Fixer → re-tester → re-review for the same route
  ↓
trace artifact: accepted-for-manual-review | needs-human-review | rejected | error
```

Simple low-risk greetings, short definitions, and straightforward conversational wording use a
minimal one-round tester path with no model Reviewer. Normal Luau/API/moderate debugging work
uses one independent Reviewer and up to two total rounds. Ambiguous, advanced, many-requirement,
client/server, remote/security, persistence/economy, spatial/combat, architecture/lifecycle,
or complex-debugging work uses two independent Reviewer passes and a default of three bounded
candidate rounds (hard cap four). The deterministic route, selected/skipped checks, context
budget, reviewer depth, and bounds are recorded in every trace. Each run records source hashes,
model/options, Code Book card revisions/source URLs where relevant, static findings, structured
reviewer findings, policy overrides, Fixer provenance, terminal status, and a clear
non-promotion statement.

An accepted trace means only that the candidate passed its configured task-appropriate quality
checks. It is not described as best possible, is **not** added to `training_data/`, is not a
final dataset row, is not an adapter, and is not a DukeOTR model release. The ordinary
validation, deduplication, evaluation-isolation, human-review, and explicit dataset-build gates
still apply separately.

## Safe usage

Plan a trace with no model or Ollama preflight:

```powershell
python -m scripts.run_builder_reviewer_fixer `
  --seed-id dukeotr-phase1-001 `
  --dry-run `
  --output reports\brf-phase1-001.plan.json
```

On the Windows machine only, after confirming the existing base tag with `ollama list`, a
small deliberate pilot can use the current base model:

```powershell
python -m scripts.run_builder_reviewer_fixer `
  --seed-id dukeotr-phase1-001 `
  --model qwen3:4b `
  --max-rounds 2 `
  --output reports\brf-phase1-001.trace.json
```

The command performs the project’s exact local model-presence preflight before inference. It
never downloads, modifies, replaces, trains, exports, or imports a model. Do not turn this
into bulk laptop generation: use it for a few inspected quality-loop pilots or a future
suitable inference environment.

## Role contracts

### Builder

**Inputs:** a project-authored train brief and route-bounded source-attributed Code Book
context (none for the minimal simple route).

**Required JSON:** a self-contained `assistant_response`, explicit assumptions, security
notes, a test plan, and the relevant Code Book card IDs. The Builder must use natural English,
avoid invented APIs and fake execution claims, and keep pure-Luau tasks free of irrelevant
Roblox networking.

### Reviewer

**Inputs:** the original train brief, candidate answer, deterministic static findings, the route
selected check scopes, and the same route-bounded Code Book context. Candidate text is quoted as
untrusted data. The Reviewer checks relevant API/security/lifecycle scopes without inventing
Roblox concerns for a simple wording task.

**Required JSON:** a decision plus 1–5 scores for correctness, security, API validity,
requirements, English, and code quality. Every issue is structured as:

- `id` and category (`correctness`, `security`, `api`, `requirements`, `english`,
  `code_quality`, `style`, `testability`, or `code_book`);
- severity (`block`, `major`, `minor`, or `advisory`);
- concrete message and observable evidence; and
- required correction.

A Reviewer `accept` is mechanically downgraded to `revise` if it contains a block/major
finding or any configured dimension is below its policy minimum. A static validation failure
also prevents effective acceptance.

### Fixer

**Inputs:** the original train brief, prior answer, all Reviewer findings, static findings,
route-selected check scopes, and Code Book context. Deeper reviewer-pass findings are
namespaced before acknowledgement so correction provenance remains unambiguous.

**Required JSON:** revised answer, changes made, remaining assumptions, unresolved risks, and
the IDs of Reviewer findings addressed. A new candidate record is created with immutable
parent/correction provenance; the old response is never overwritten.

## Isolation and security invariants

- Only a `train` source brief is accepted. The command performs a local wording-level
  train-versus-evaluation collision check before the first role runs.
- Evaluation prompts, rubrics, baseline outputs, scores, and expected answers are never
  included in Builder, Reviewer, or Fixer prompts. The trace contains only collision IDs and
  similarity numbers if it blocks a run.
- Code Book context is source-attributed reference material, not authority to skip API
  verification and not automatic SFT data.
- Sensitive state remains server-owned. A LocalScript can call `RemoteEvent:FireServer`; a
  RemoteEvent does not authenticate, validate, or authorize client-controlled input.
- A reviewer error, parser error, rejection, static block, or exhausted repair budget stops
  automatic progress. Nothing is silently accepted or promoted.

## Deliberate non-features

This loop does not execute Luau, edit Roblox places, use held-out tasks as feedback, run
training, manufacture an adapter, claim benchmark improvement, or call `ollama run dukeotr`.
Real LoRA/QLoRA work remains a future suitable CUDA/cloud-machine step after an actual,
quality-gated dataset and baseline/evaluation evidence exist.
