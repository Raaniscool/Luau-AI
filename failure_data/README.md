# Local failure database boundary

`failure_data/` holds local, machine-readable records created by
`python -m scripts.record_failures`. A failure row may contain prompts, candidate answers,
review findings, correction text, provenance, and verification state, so it is deliberately
separate from:

- `raw_data/` project-authored source briefs;
- `generated_data/`, `validated_data/`, and `training_data/` candidate/final-dataset stages;
- `knowledge_base/` verified reference material and the Code Book; and
- `evaluation_data/` permanently held-out tasks and score artifacts.

The directory ignores JSON/JSONL/CSV outputs by default. It is **not** an automatic source of
training examples: record failures with actual pipeline evidence, analyze them, plan review-
required briefs if warranted, then use the existing human review and candidate gates. Never put
held-out prompts, rubrics, expected answers, baseline answers, or scores here as generation
context.
