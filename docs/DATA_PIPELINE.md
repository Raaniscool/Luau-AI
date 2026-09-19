# Data pipeline and quality contract

## Design goal

The corpus is intended to teach a pretrained Qwen3 model **Roblox/Luau specialization**,
not basic language. Each row is an auditable instructional conversation with a source task,
not an unreviewed dump of model text.

The tracked seed catalog has 70 varied task specifications across beginner,
intermediate, and advanced work. `scripts.audit_catalog --fail-on-missing` verifies all
required coverage themes and checks exact duplicate source briefs. It currently reports:

- 23 code-generation briefs, 14 architecture briefs, 8 bug fixes, 6 security reviews,
  6 explanations, 5 API tasks, 4 code reviews, 2 optimizations, and 2 natural-language
  conversions;
- 14 beginner, 29 intermediate, and 27 advanced briefs; and
- a separate 24-task held-out evaluation file.

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

`generate_examples.py` reads only `raw_data/roblox_luau_seed_tasks.jsonl`. It sends a
structured task brief to local Ollama and demands a JSON envelope containing the answer,
covered concepts, and self-check claims. Prompt variants are scenario-driven and source
briefs use different task types and wording to avoid repetitive answers.

Generation failures are written to a report rather than converted into guessed records.
No generated artifact is final data.

### 2. Validation / review

`validate_examples.py` applies two independent layers:

- **static checks:** schema, code-fence balance, hidden reasoning markup, obviously unsafe
  dynamic/exploit APIs, common event API mistakes, RemoteEvent callback shape, busy loops,
  DataStore error-handling warnings, potential client-side DataStore access, and advisory
  requirement evidence checks;
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

`deduplicate_examples.py` considers only schema-valid, static-pass, accepting-reviewer
records by default. It records normalized exact hashes and transparent token 3-gram
Jaccard similarity. The default near-duplicate threshold is `0.82` and is saved in every
artifact.

It also compares **training prompts only** against held-out evaluation prompts at the
stricter default threshold `0.93`. Conceptual overlap (for example, both sets testing
RemoteEvents) is expected; near-identical wording is not. Potential collisions are
recorded and `build_datasets.py` excludes them again as a final defense.

### 5. Final dataset creation

`build_datasets.py` creates:

- `training_data/final_dataset.jsonl` — every accepted final conversation;
- `training_data/train.jsonl` — deterministic stratified training split; and
- `training_data/validation.jsonl` — a small development split, distinct from the
  held-out evaluation suite.

It refuses each record lacking any of these: valid schema, static pass, accepting LLM or
human review, `unique` dedupe status, and evaluation isolation. A manifest records
coverage, source path, rejected IDs/reasons, hashes, and split counts.

## Human review

For spot checks or high-stakes examples, make a decisions JSONL file such as:

```json
{"record_id":"gen-...","decision":"accept","reviewer":"alice","notes":"Verified API names and server authority."}
```

Then apply it without hand-editing candidate data:

```bash
python -m scripts.record_human_review \
  --input validated_data/validated_examples.jsonl \
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
