# Raw data

This directory contains **curated source specifications**, not blindly trusted training
examples. `roblox_luau_seed_tasks.jsonl` is a broad, project-authored task catalog used
only as input to `scripts.generate_examples`.

Do not put scraped Roblox source code here without recording its license, source URL,
revision, and permission status. Do not feed held-out files from `evaluation_data/` into
generation.
