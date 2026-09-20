# DukeOTR Training Factory

The Training Factory is reproducible **data-curation infrastructure** for DukeOTR. It extends
the existing Builder → Reviewer/Tester → Fixer pipeline; it does not replace the canonical
candidate schema, source portfolio, Code Book, verified knowledge layer, final dataset builder,
or permanently held-out 100-task evaluation suite.

> **Status:** Factory tooling and planned-version metadata are implemented. No DukeOTR corpus,
> adapter, weights, Ollama import, benchmark run, or separate trained model is created by this
> directory or its scripts.

## Contracts and boundaries

| Location | Role | May automatically become training data? |
|---|---|---:|
| `raw_data/*.jsonl` | curated project-authored source briefs | No; normal candidate pipeline and gates still apply |
| `knowledge_base/` and Code Book artifacts | verified/reference material | No |
| `generated_data/` / `validated_data/` | canonical candidate traces | No; final dataset build is explicit |
| `validated_data/training_factory/` | Builder/reviewer/fixer audit ledger | No |
| `failure_data/` | observed-failure database | Never |
| `raw_data/training_factory_drafts/` | human-review-required targeted source drafts | Never directly |
| `training_data/` | explicit final train/development artifacts | Only after normal final gates |
| `evaluation_data/` | permanent held-out evaluation prompts/rubrics | Never |
| `reports/` | local output, scoring, and audit artifacts | Never |

All factory outputs have protected-path guards: they cannot write into held-out evaluation,
final training-data, verified-knowledge, Code Book, or active source-catalog paths. The only
local use of evaluation prompts is a wording-level collision check after source-brief
construction; evaluation content is not sent to a generator, reviewer, planner, dataset
formatter, or model.

## Sidecar record formats

[`schema.json`](schema.json) documents four JSON contracts, while the dependency-free runtime
validators live in [`scripts/lib/training_factory.py`](../scripts/lib/training_factory.py):

1. **Failure record** — prompt/task context; actually known producing agent/model metadata;
   original answer; controlled failure categories and severity; static/LLM/human review evidence;
   actual corrected solution if present; revalidation state; provenance; and timestamps.
2. **Lineage ledger record** — joins original Builder output, reviewer/tester findings, corrected
   output, correction explanation, failure analysis, quality status, provenance, and version
   context without changing or promoting the canonical candidate.
3. **Targeted draft** — an answer-free, human-review-required task specification with category,
   Beginner/Intermediate/Advanced difficulty, Short/Normal/Deep depth rationale, requirements,
   evidence expectations, provenance, and the exact measured failure category/count that justified
   it.
4. **Version registry** — metadata for `dukeotr_dataset_v1`, `dukeotr_v1`, `dukeotr-v1`, and
   reserved `dukeotr`. Planned names are not model-existence claims. A future trained/evaluated
   state requires actual artifact/evaluation paths and SHA-256 evidence fields.

`failure_taxonomy.json` is the controlled category vocabulary. `targeted_brief_templates.json`
contains task-specification templates only—not answers—and `configs/training_factory.json`
contains paths, coverage tracks, thresholds, and non-promotion policy.

## Normal evidence-first workflow

Run normal candidate generation/validation/correction/deduplication exactly as documented in the
repository. After a pilot, create local sidecars from candidate artifacts:

```powershell
python -m scripts.build_training_ledger `
  --input validated_data\pilot.validated.jsonl `
  --corrections validated_data\pilot.corrected.validated.jsonl `
  --output validated_data\training_factory\pilot.lineage.jsonl

python -m scripts.record_failures `
  --input validated_data\pilot.validated.jsonl `
  --corrections validated_data\pilot.corrected.validated.jsonl `
  --output failure_data\pilot.failures.jsonl

python -m scripts.analyze_failures `
  --input failure_data\pilot.failures.jsonl `
  --output reports\training_factory\pilot.failure_analysis.json
```

These commands do not call Ollama, generate answers, or promote rows. A reviewer outage by
itself is not counted as a model weakness. The recorder creates failure rows only when a canonical
candidate has stored static-fail, reviewer-revise/reject, duplicate, or split-isolation evidence.
Unknown model/agent identity stays `known: false`; it is never guessed.

For a normal live pilot, `run_pipeline.py --record-factory-sidecars` performs the two separate
sidecar stages after deduplication. It remains opt-in, does not alter the final-dataset opt-in,
and does not invoke failure analysis/targeted planning automatically. Use a new run ID (or the
existing reviewed overwrite mechanism) just as for every other pipeline artifact.

### Measured weakness → targeted task planning

Only choose a failure category that appears in the stored database. For example, this command
refuses to run unless `insecure_remote_handling` has at least one matching stored failure row:

```powershell
python -m scripts.generate_targeted_briefs `
  --failure-input failure_data\pilot.failures.jsonl `
  --failure-category insecure_remote_handling `
  --beginner 4 --intermediate 4 --advanced 2 `
  --seed 3407 --created-at 2026-09-20T00:00:00Z `
  --output raw_data\training_factory_drafts\remote_security.pilot.jsonl
```

The output is deterministic for the same input fingerprint, category, counts, seed, and
`--created-at`. It creates only source briefs with distinct scenarios/task forms, then checks
local wording collision against the held-out suite. It does not mass-generate superficial
paraphrases and it never generates an answer.

A named reviewer must decide every draft. A decision JSONL row has exactly:

```json
{"draft_id":"draft-example","decision":"accept","reviewer":"reviewer-name","notes":"Why this task adds a distinct, factual, useful learning objective."}
```

Then make a **new**, separate catalog (never an in-place append to active catalogs):

```powershell
python -m scripts.promote_targeted_briefs `
  --drafts raw_data\training_factory_drafts\remote_security.pilot.jsonl `
  --decisions raw_data\training_factory_drafts\remote_security.decisions.jsonl `
  --against-catalog raw_data\dukeotr_phase1_luau_seed_tasks.jsonl `
  --against-catalog raw_data\roblox_luau_seed_tasks.jsonl `
  --output raw_data\reviewed_targeted_remote_security.pilot.jsonl
```

The promoter rejects incomplete decisions, malformed drafts, near duplicates, and held-out
wording collisions. A resulting source brief retains its targeted-draft and reviewer provenance,
but it still only becomes a canonical candidate after the normal pipeline is run explicitly.

## Quality gates and their limits

The factory validates required fields, controlled task category/type/difficulty/depth values,
human-review-required status, correction/review relationships, known-versus-unknown agent
provenance, taxonomy labels, controlled severity, duplicate IDs, model-version metadata, source
provenance, and held-out boundaries. The canonical pipeline additionally validates schema, static safety/content signals,
review acceptance, correction lineage, exact/near duplicates, assistant length, and final
train/development partitioning. Final dataset manifests now report category and task-depth
coverage when labels are available.

These checks are intentionally **not an oracle of factual correctness**. A fact can still require
an authoritative Roblox/Luau source and human review; a clean gate cannot make an unsupported API
claim true. Likewise, an observed failure count reflects only the stored artifacts, not the true
rate of a weakness.

## Broad coverage plan

`configs/training_factory.json` declares planned tracks for natural English/reasoning, Luau,
Roblox, client/server security, UI responsiveness, and future 3D/model presentation. The factory
does not falsely mark a track as covered merely because it is planned. Source/audit reports expose
actual category/depth counts and legacy fields as `unspecified` rather than invent labels.

Target templates currently include evidence-driven profiles for remote security, CFrame/spatial
reasoning, future 3D/model presentation, UI responsiveness, Roblox/Luau API correctness, Luau
language reasoning, DataStore safety, lifecycle/performance, physics reasoning, and
requirement/English reasoning. New profiles
must be supplied through the template/taxonomy configuration and pass the factory audit rather
than by hardcoding one-off routing rules.

## Version and future QLoRA handoff

Validate planned metadata without touching a model:

```powershell
python -m scripts.record_model_version --check
python -m scripts.audit_training_factory --strict
```

Future training remains a separate CUDA/cloud operation using compatible Hugging Face
`Qwen/Qwen3-4B` weights. If a future metadata update claims a trained/evaluated/released state,
`record_model_version.py` requires `--verify-evidence-files` on the machine holding the actual
adapter/evaluation files and compares their SHA-256 values before writing the registry. Use the existing `prepare_training_bundle.py`, `train_qlora.py`,
`configs/qlora_sft.json`, and [training-machine runbook](../docs/TRAINING_MACHINE_RUNBOOK.md)
after a real final manifest exists. Do not modify, move, replace, re-download, or otherwise
model-face the existing Windows `qwen3:4b` installation from this factory; any future Windows
Ollama action starts with `ollama list` and follows the repository's existing preparation gates.

## Audit

```powershell
python -m scripts.audit_training_factory --strict
```

Optional `--failure-input`, `--draft-input`, and `--ledger-input` arguments extend the audit to
local sidecars. Reports intentionally avoid copying held-out prompt text. A strict failure means
an actual contract/boundary issue; it is separate from expected held-out-overwrite refusal tests.
