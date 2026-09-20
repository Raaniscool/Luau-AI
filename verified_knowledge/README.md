# DukeOTR Verified Roblox Knowledge / Fast-Answer Layer

This directory is a **separate, source-checked reference library** for narrow Roblox/Luau
questions that can safely be answered without model generation. It is not a model, adapter,
training dataset, Code Book replacement, benchmark, or automatic data-promotion path.

The initial library contains **27 curated concepts** backed by **19 official Roblox Creator Hub
source records**. It covers the initial requested foundation: events and connections; remotes;
Script, LocalScript, and ModuleScript; common data-model containers; player/character objects;
Instances and hierarchy lookups; Vector3/CFrame/Color3; and common timing/input services.

## Files

| Path | Purpose |
| --- | --- |
| `roblox_luau_fast_answers.jsonl` | One structured, source-checked entry per concept. This is the library, not training data. |
| `sources.json` | Normalized official-source catalog with URLs, source type/version wording, and verification dates. |
| `schema.json` | Published JSON Schema for entry shape; the strict audit also enforces cross-entry and policy checks. |

Each entry contains a stable ID, category/concept, canonical name, normalized aliases and match
terms, authored Quick/Normal/Deep explanations, separately approved fast-answer depths, verified
Luau examples, misconceptions, named API claims, official source references, difficulty, Code
Book cross-references, and last-verification metadata. It explicitly carries:

```text
training_policy: not_training_data_or_auto_promotion
evaluation_policy: not_sourced_from_held_out_evaluation
```

Those fields are audit requirements, not suggestions.

## Deliberately conservative runtime flow

```text
user request
  → deterministic alias + definition-intent match against this library
  → pure source-checked entry answer only when confidence/currentness/depth are sufficient
  → otherwise existing effort_routing.classify_task(...)
  → existing adaptive Builder → Reviewer/Tester → Fixer pipeline
```

`scripts/lib/request_routing.py` implements that boundary. It first calls the verified matcher;
on every no-match or escalation it calls the existing `classify_task()` unchanged and reports the
adaptive route. A caller can then invoke the existing Builder/Reviewer/Tester/Fixer runner.
The verified layer **does not** invoke Ollama, modify model files, create candidates, or replace
the quality architecture.

A fast answer is only eligible when all of these are true:

- the request is a short, explicit definition/explanation request for one curated concept;
- normalized aliases/match terms give a high-confidence, unambiguous match;
- the entry is source-checked, stable, and within its review window;
- the requested Quick, Normal, or Deep level is explicitly approved by that entry; and
- the returned answer is rendered verbatim from curated fields, with source attribution.

The matcher supports ordinary phrasing variants such as:

- `What is a Roblox event?`
- `What’s an event in Roblox?`
- `Explain Roblox events.`

It fails closed for custom implementation, debugging, architecture, security, version-sensitive,
ambiguous, multi-concept, long, or detailed client/server requests. For example, a detailed
RemoteEvent client/server question is deliberately routed to adaptive quality work rather than
answered with a canned remote snippet. Entries for RemoteEvent and RemoteFunction currently
approve only Quick/Normal fast answers for that reason.

## Query without model inference

From the repository root:

```powershell
# Prints a JSON route/result. This does not contact Ollama.
python .\scripts\query_verified_knowledge.py --prompt "What’s an event in Roblox?"

# Quick entry depth selected from wording; source-attributed Markdown result.
python .\scripts\query_verified_knowledge.py `
  --prompt "Briefly explain Roblox events." `
  --format markdown

# This reports an adaptive route; it does not run the downstream model pipeline itself.
python .\scripts\query_verified_knowledge.py `
  --prompt "Explain detailed RemoteEvent behavior between the client and server."
```

Use `--as-of YYYY-MM-DD` for deterministic freshness testing. Optional output paths are checked
against the protected held-out evaluation file before any write. The CLI refuses
`evaluation_data/roblox_luau_eval.jsonl` even if supplied via an equivalent resolved path.

## Strict audit and evaluation isolation

```powershell
python .\scripts\audit_verified_knowledge.py --strict
```

The audit validates, without model calls:

- required entry metadata, entry/source IDs, valid official HTTPS hosts, source references, and
  source type/version/check-date records;
- exact/near duplicate concepts and alias collisions;
- approved API allowlist membership, explicitly rejected invented names, and visible
  example-to-API/source correspondence where a code example is present;
- Code Book cross-reference IDs while keeping Code Book cards separate and unpromoted;
- stale/future verification dates (staleness is a strict-audit failure);
- the no-training/no-auto-promotion policies and forbidden held-out-path markers; and
- local wording-level collisions between entry aliases and the held-out suite.

The collision report retains only library IDs, held-out task IDs, and similarity scores. It never
emits held-out prompt text, rubrics, expected answers, baseline answers, or scoring material, and
it never supplies that material to a model. The report writer uses the same protected-output
check as the query CLI.

## Maintenance rules

1. Curate an entry from official Roblox/Luau documentation first; do not infer undocumented API
   behavior from a model response or a tutorial.
2. Add/refresh a normalized source record before referencing it from an entry.
3. Keep the entry outside `raw_data/`, generated candidates, validated candidates, and
   `training_data/` unless a separate, explicit human-reviewed governance decision creates a
   different artifact. There is no automatic conversion mechanism here.
4. Run `python scripts/audit_verified_knowledge.py --strict` and relevant tests after every
   edit. Re-verify an entry by its `review_by` date before allowing it to remain a fast path.
5. If an answer needs adaptation, project-specific code, security reasoning, current-version
   claims, or additional concepts, do not stretch this library—route it through the existing
   adaptive quality system.

## Current execution boundary

The structured library, deterministic matcher, routing adapter, CLI, and strict audit are
implemented repository tooling. They are not a deployed chat service and do not claim that a
DukeOTR model has been trained, fine-tuned, evaluated, exported, or released. Qwen3-4B remains
technical provenance for future model work; no local or remote model action is required for this
library.
