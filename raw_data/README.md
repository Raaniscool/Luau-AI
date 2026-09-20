# Raw data

This directory contains **curated source specifications**, not blindly trusted training
examples. `roblox_luau_seed_tasks.jsonl` is a broad, project-authored task catalog used
only as input to `scripts.generate_examples`. `dukeotr_phase1_luau_seed_tasks.jsonl` is the
separate Phase-1 Luau-fundamentals catalog. Together they contain 131 source briefs across
implementation, explanation, output prediction, diagnosis/correction, bad-answer critique,
review, security/insecurity analysis, refactor, optimization, API-misuse diagnosis,
architecture, requirements, trade-offs, completion, subtle-bug analysis, and runtime reasoning.
They remain specifications—not generated examples or a completed corpus.

Do not put scraped Roblox source code here without recording its license, source URL,
revision, and permission status. Do not feed held-out files from `evaluation_data/` into
generation.

Run `python .\scripts\audit_source_portfolio.py --strict` to check the combined source-brief
floor, task-form diversity, task-type concentration, exact duplicate prompts, and local
wording-level held-out collisions. A passing report is a source-brief quality signal only; it
does not claim model generation, review, deduplication, or dataset eligibility.
