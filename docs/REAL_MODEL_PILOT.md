# DukeOTR Real Model Pilot (Windows-local, Qwen3-4B)

This is a small **data-discovery and pipeline-baseline** workflow. It is not training, an adapter build, an Ollama import, or a claim that Qwen3-4B has become a DukeOTR model.

The selected baseline for this v1 pilot is the exact pre-existing local Ollama tag **`qwen3:4b`** (Qwen3-4B). The runner uses the repository's existing `OllamaClient` and existing Builder → Tester/Reviewer → Fixer trace runner; it does not scrape terminal output or implement a second HTTP client.

## What existed before this pilot

- `scripts/run_builder_reviewer_fixer.py` already produced one bounded, non-promoting Builder → Tester/Reviewer → Fixer trace from a project-authored train brief.
- `scripts/lib/ollama.py` already used local Ollama's API and required `ollama list` plus `/api/tags` model presence checking for a local host.
- The Training Factory already supplied deterministic deduplication/held-out wording checks, failure recording, lineage ledgers, and evidence-only failure analysis.
- Final dataset construction (`scripts/build_datasets.py`) remained a separate explicit boundary.

The pilot coordinates those components; it does not replace them or rebuild the desktop application.

## Safety and scope boundaries

1. The predefined source briefs are in `pilot_data/real_model_pilot_v1.jsonl`, separate from ordinary train briefs and held-out evaluation material. Every ID begins with `pilot-` and requires `source.pilot_only: true`.
2. Before any model request, the runner validates all source briefs and performs a local held-out wording comparison. It serializes only collision IDs/scores if blocked; it never passes held-out prompts, rubrics, answers, or scores to model roles.
3. On a real run, `OllamaClient.assert_model_present("qwen3:4b")` executes `ollama list` for local hosts and confirms `/api/tags`. It never pulls, deletes, replaces, or modifies the model.
4. Work is sequential (`parallelism = 1`) and each B/R/F trace is saved before the runner starts the next task.
5. Raw Builder output, correction lineage, terminal review/test state, post-deduplication/isolation state, and quality-gate-eligible state are written separately. A skipped or failed Reviewer remains visibly ineligible; nothing is silently upgraded. A Fixer answer with no non-empty actual `changes_made`/`reason` evidence is preserved but is explicitly unlinked and ineligible; the coordinator never invents an explanation from a diff or review text.
6. The runner **never invokes `scripts.build_datasets.py`** and never writes `training_data/`. “Training eligible” in the report means only that the existing final quality predicate currently accepts an evidence record; it is not a promotion decision.
7. Failure recommendations come only from the existing `analyze_failures` sidecar over actual stored pilot failure rows and its configured threshold. No report makes an improvement score or a broad weakness claim.

## Windows prerequisites

Open PowerShell in your cloned repository, for example:

```powershell
cd C:\Users\Raani\Luau-AI
git status --short
ollama list
```

Do not continue unless `ollama list` shows the exact tag `qwen3:4b`. Do **not** run `ollama pull`, delete a model, or substitute a lookalike tag for this pilot.

Use a current Python 3 installation. The repository uses only its existing dependency-free Python tooling for this flow.

## First: no-model static audit

This checks the 12 pilot briefs, duplicate/prefix/provenance rules, run directory safety, and held-out wording isolation. It does **not** call Ollama, B/R/F, or any model:

```powershell
py -3 -m scripts.run_real_model_pilot `
  --run-id win-qwen3-4b-pilot-20260920 `
  --dry-run
```

Inspect:

```powershell
Get-Content reports\real_model_pilots\win-qwen3-4b-pilot-20260920\pilot_state.json
Get-Content reports\real_model_pilots\win-qwen3-4b-pilot-20260920\pilot_report.json
```

You may reuse the same run ID for its real run because the dry-run checkpoint contains no model response; use a different ID if you want planning and real-run artifacts kept in separate directories. In either case, a run ID binds catalog/config and held-out-evaluation fingerprints so incompatible evidence cannot be mixed.

## Real local run

After confirming `ollama list`, run a small first slice. This is intentionally sequential and persists after each task:

```powershell
py -3 -m scripts.run_real_model_pilot `
  --run-id win-qwen3-4b-pilot-20260920-real `
  --max-tasks 3 `
  --timeout-seconds 900
```

Then resume the same run ID for its remaining pending or interrupted tasks **while its pipeline/config fingerprint remains unchanged**:

```powershell
py -3 -m scripts.run_real_model_pilot `
  --run-id win-qwen3-4b-pilot-20260920-real `
  --timeout-seconds 900
```

The real run performs the same no-download preflight itself, even after you manually checked `ollama list`. It is local-loopback-only; a remote `--host` or remote `OLLAMA_HOST` is refused before inference so prompts/history are not sent off-machine. It is pinned to `qwen3:4b`; attempting another `--model` is rejected so a future provider/model pilot must be explicitly versioned instead of being mixed into this baseline.

Every B/R/F call requests Ollama `think: false` through the existing `OllamaClient`. If a newer Ollama API nevertheless returns a separate `thinking` field, the client keeps it separate from final `response` content and the trace records only bounded presence/hash metadata—not the reasoning text. Text that the model itself places inside the structured `assistant_response` remains actual final-output evidence; the pilot does not heuristically strip it, turn it into a success, or invent a replacement answer.

### Failures, timeouts, and interruption

- Each Ollama HTTP response/stream chunk uses `--timeout-seconds`; increase it for slow local hardware instead of changing the model.
- If a B/R/F task returns a model/transport/parser error, its trace and state are retained as `generation_failed`. The invocation ends with a non-zero exit status so automation cannot call it clean success. It is **not retried by default**, because retrying would generate another nondeterministic answer.
- Inspect the trace first, then retry failed tasks explicitly (a new `attempt-N` trace is created; the old one is preserved):

```powershell
py -3 -m scripts.run_real_model_pilot `
  --run-id win-qwen3-4b-pilot-20260920-real `
  --retry-failed `
  --timeout-seconds 900
```

- `Ctrl+C` is recorded as `interrupted`. Re-run the same command/run ID to resume pending and interrupted tasks. If B/R/F atomically finished a terminal trace just before the interruption, the coordinator recovers that trace rather than sampling a second answer; otherwise it creates the next preserved attempt. Do not delete the run directory to “fix” an interruption.
- A missing/unreachable local Ollama service or missing exact tag stops at preflight and writes a safe `blocked_preflight` state/report without any candidate response.

### Safe recovery of a persisted `sidecar_error` (no model call)

If a real run has already preserved its traces/candidate files but stops with `state: "sidecar_error"`—for example, an actual Fixer answer has an empty `changes_made` list—**do not retry, delete, overwrite, or normally resume it first**. Pull the coordinator repair, then use the explicit no-inference repair mode:

```powershell
cd "$HOME\Luau-AI"
git pull --ff-only origin arena/01a0bb76-luau-ai

py -3 -m scripts.run_real_model_pilot `
  --run-id win-qwen3-4b-pilot-20260920-real `
  --repair-sidecars `
  --timeout-seconds 900
```

`--repair-sidecars` does **not** instantiate `OllamaClient`, run `ollama list`, contact Ollama, invoke Builder/Reviewer/Fixer, retry a task, or rewrite `raw_builder_candidates.jsonl`, `corrected_candidates.jsonl`, or `final_candidates.jsonl`. It first copies the pre-repair state/reports/derived artifacts to `repair_snapshots/<timestamp>/`, records immutable trace hashes, and then regenerates only derived lineage/deduplication/failure-analysis artifacts. It requires those three persisted source-candidate files; it refuses to reconstruct missing model evidence.

The normal model-generation resume guard remains exact: after a code/config fingerprint change, a standard invocation still refuses the old run ID. The repair exception is deliberately limited to post-processing already-persisted evidence, and the report/state records both the stored and repair-coordinator fingerprints. Do not use `--retry-failed` for this recovery. If you later choose to generate new model output under changed code/config, inspect this repaired evidence first and start a **new run ID** rather than mixing provenance into the existing baseline.

A successful repair command exits `0` because the sidecar operation completed; this does **not** mean the pilot became clean. Its report/state still retain any actual `generation_failed`, rejected, interrupted, or pending tasks. `failure_analysis.json` appears only if the existing sidecars complete over the real stored evidence.

## Artifact locations and inspection

All generated pilot evidence remains under one ignored run directory:

```text
reports/real_model_pilots/<run-id>/
  pilot_state.json
  pilot_manifest.json
  pilot_report.json
  traces/<pilot-task-id>.attempt-<N>.trace.json
  raw_builder_candidates.jsonl
  corrected_candidates.jsonl
  linkable_corrections.jsonl              # explanation-backed subset only
  correction_linkage_report.json          # preserved unlinked-correction evidence
  final_candidates.jsonl
  deduplication_input_candidates.jsonl    # derived, excludes unsupported corrections
  deduplicated_candidates.jsonl
  training_eligible_candidates.jsonl
  training_factory_ledger.jsonl
  observed_failures.jsonl
  failure_analysis.json
  repair_snapshots/<timestamp>/           # pre-sidecar-repair copies/manifests, if used
```

Useful PowerShell inspection commands:

```powershell
$run = "reports\real_model_pilots\win-qwen3-4b-pilot-20260920-real"
Get-Content "$run\pilot_report.json"
Get-Content "$run\pilot_state.json"
Get-Content "$run\pilot_manifest.json"
Get-ChildItem "$run\traces"
Get-Content "$run\failure_analysis.json"
Get-Content "$run\correction_linkage_report.json"
Get-Content "$run\training_factory_ledger.jsonl"
Get-ChildItem "$run\repair_snapshots" -ErrorAction SilentlyContinue
```

`pilot_report.json` contains attempted/completed/generation-failure status, B/R/F terminal states, actual review/static/fixer counts, explicit unlinked-correction evidence, verification/failure category/severity summaries, repeated patterns only when stored evidence repeats, quality eligibility, model/pipeline/dataset-version provenance, sidecar-repair audit history when applicable, and evidence-backed recommendations. It deliberately has no “improvement score.”

## What this cannot establish yet

- It cannot prove a general model quality level from roughly 12 predefined tasks.
- It cannot claim a Qwen3-4B fine-tune, adapter, DukeOTR model, executable integration, or Roblox Studio automation.
- It cannot replace independent human review, factual validation against current Roblox documentation, final dataset construction, actual training, or held-out evaluation.
- Arena does not run your Windows-local Ollama model. Results exist only after you run the command on your machine, and every report is bounded to the actual stored local traces.
