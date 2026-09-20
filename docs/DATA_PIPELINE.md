# DukeOTR data pipeline and quality contract

## Design goal

DukeOTR is the public identity of a specialization derived from a pretrained Qwen3 base for
**Roblox/Luau engineering**, not basic language. Each row is an auditable instructional
conversation with a source task, not an unreviewed dump of model text. Natural English
remains expected in every curated answer; Qwen is preserved as technical provenance.

The repository has two tracked project-authored source-brief catalogs:

- `raw_data/dukeotr_phase1_luau_seed_tasks.jsonl` has 45 dedicated Luau-fundamentals
  briefs. `scripts/audit_dukeotr_curriculum.py --strict` verifies its required concepts,
  instructional modes, difficulty range, duplicate prompts, and held-out collisions.
- `raw_data/roblox_luau_seed_tasks.jsonl` has 70 broad Roblox/Luau briefs across
  beginner, intermediate, and advanced work. `scripts/audit_catalog.py --fail-on-missing`
  verifies its broader coverage themes and exact duplicate source briefs.

The broad catalog currently includes 23 code-generation briefs, 14 architecture briefs,
8 bug fixes, 6 security reviews, 6 explanations, 5 API tasks, 4 code reviews, 2
optimizations, and 2 natural-language conversions; its difficulty distribution is 14
beginner, 29 intermediate, and 27 advanced. A separate 24-task held-out evaluation file is
maintained throughout.

These counts describe **source briefs**, not a falsely claimed completed training corpus.
Each brief can produce one or more diverse, reviewed candidates.

## Canonical candidate shape

Every candidate uses JSONL with this essential structure:

```json
{
  "schema_version": "1.0",
  "record_id": "gen-train-...",
  "source_seed_id": "train-...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "split": "train",
    "task_type": "code_generation",
    "difficulty": "intermediate",
    "topics": ["RemoteEvents", "security"],
    "requirements": ["..."],
    "expected_evidence": ["..."],
    "source": {"kind": "project_authored"}
  },
  "quality": {"static": {}, "llm_review": {}, "human_review": {}, "deduplication": {}}
}
```

Correction creates a **new** record with `parent_record_id`, correction round, and a
snapshot of the findings that led to the repair. It never overwrites the original candidate.

## Required stages

### 1. Generation

`generate_examples.py` defaults to
`raw_data/dukeotr_phase1_luau_seed_tasks.jsonl` and can take either curated catalog through
`--seeds`. It sends a structured task brief to local Ollama only after the exact local model
registration preflight, and demands a JSON envelope containing the answer, covered concepts,
and self-check claims. The request uses Ollama JSON Schema mode, not only a prose prompt, so
small local models are constrained to the required envelope. Prompt variants are
scenario-driven and source briefs use different task types and wording to avoid repetitive
answers. Pure Luau-fundamentals briefs explicitly forbid irrelevant Roblox services,
RemoteEvents, and client/server placement details unless the source brief requires them.

Generation failures are written to a report rather than converted into guessed records. If a
response was received but cannot be parsed, the ignored report retains a bounded
`model_response_excerpt`, its SHA-256, truncation status, and elapsed time for diagnosis.
No generated artifact is final data.

### 2. Validation / review

`validate_examples.py` applies two independent layers:

- **static checks:** schema, code-fence balance, hidden reasoning markup, obviously unsafe
  dynamic/exploit APIs, common event API mistakes, labeled client/server RemoteEvent direction,
  `LocalPlayer`/Character misuse, busy loops, DataStore error-handling warnings, potential
  client-side DataStore access, source-authored required code evidence, and advisory
  requirement evidence checks. The checker is versioned: a material rule update invalidates
  older static passes until the record is revalidated;
- **structured reviewer pass:** an Ollama prompt that separately judges API plausibility,
  security, requirement coverage, and pedagogy, returning `accept`, `revise`, or `reject`
  plus concrete findings.

The reviewer prompt treats candidate contents as untrusted quoted data to reduce prompt
injection risk. A reported `accept` is downgraded to `revise` if its accuracy, security,
or pedagogy score is below the configured policy floor.

`--skip-llm-review` is useful for offline diagnostics only. It marks records ineligible for
final construction; it is not a loophole around review.

### 3. Correction

`correct_examples.py` selects static failures and reviewer `revise` records. It gives a
separate correction prompt the original request, answer, requirements, deterministic
findings, and review result. Corrected output is never trusted automatically: rerun
`validate_examples.py` on it.

`reject` is excluded by default. Use `--include-rejected` only when an expert deliberately
wants an additional repair attempt.

### 4. Deduplication and split isolation

`deduplicate_examples.py` considers only schema-valid, **current-checker** static-pass,
accepting-reviewer records by default. It records normalized exact hashes and transparent
token 3-gram Jaccard similarity. The default near-duplicate threshold is `0.82` and is saved
in every artifact.

It also compares **training prompts only** against held-out evaluation prompts at the
stricter default threshold `0.93`. Conceptual overlap (for example, both sets testing
RemoteEvents) is expected; near-identical wording is not. Potential collisions are
recorded and `build_datasets.py` excludes them again as a final defense.

### 5. Final dataset creation

`build_datasets.py` creates, within an explicitly selected output directory:

- `<output-dir>/final_dataset.jsonl` — every accepted final conversation;
- `<output-dir>/train.jsonl` — deterministic stratified training split; and
- `<output-dir>/validation.jsonl` — a small development split, distinct from the
  held-out evaluation suite.

`run_pipeline.py` stops after deduplication for a pilot by default. It supplies a per-run
`training_data/<new-run-id>/` directory only with explicit `--build-final-dataset` opt-in and
refuses to reuse it without an explicit reviewed overwrite. The failed
`dukeotr_phase1_pilot_001` ID and the quality-diagnostic `dukeotr_phase1_pilot_002` ID must
not be reused. A final version name such as
`dukeotr_dataset_v1` is reserved until its manifest and quality evidence actually exist.

It refuses each record lacking any of these: valid schema, a pass from the current static
checker, accepting LLM or human review, `unique` dedupe status, and evaluation isolation. A
manifest records
coverage, source path, rejected IDs/reasons, hashes, and split counts.

## Human review

For spot checks or high-stakes examples, make a decisions JSONL file such as:

```json
{"record_id":"gen-...","decision":"accept","reviewer":"alice","notes":"Verified API names and server authority."}
```

Then apply it without hand-editing candidate data:

```bash
python -m scripts.record_human_review \
  --input validated_data/dukeotr_phase1_candidates.validated.jsonl \
  --decisions my_decisions.jsonl \
  --output validated_data/human_reviewed_examples.jsonl
```

Human acceptance can supplement an LLM reviewer result, but cannot bypass a static failure,
duplicate mark, or held-out leakage check.

## Data governance

The tracked task briefs are project-authored. Do not add copied/scraped Roblox code, paid
assets, private game scripts, or content with unclear license/permission status. Keep any
future external source in a manifest with URL, license, revision/date, allowed use,
transformation, and reviewer. See [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md).
