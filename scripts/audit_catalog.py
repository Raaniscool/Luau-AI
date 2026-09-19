"""Audit curated seed/evaluation coverage before spending local model time."""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import sys
from collections import Counter
from typing import Any

from scripts.lib.dedupe import normalized_tokens
from scripts.lib.evaluation import validate_evaluation_tasks
from scripts.lib.io_utils import read_jsonl, utc_now, write_json_atomic
from scripts.lib.schema import validate_seed

REQUIRED_THEMES: dict[str, tuple[str, ...]] = {
    "Luau syntax": ("luau-syntax",),
    "Roblox APIs": ("roblox-api", "api-usage"),
    "RemoteEvents": ("remoteevents",),
    "RemoteFunctions": ("remotefunctions",),
    "client/server architecture": ("client-server", "architecture"),
    "security and exploit resistance": ("security",),
    "Instances": ("instances",),
    "services": ("services",),
    "events": ("events",),
    "functions": ("functions",),
    "tables": ("tables",),
    "metatables": ("metatables",),
    "ModuleScripts": ("modules",),
    "OOP patterns": ("oop",),
    "UI": ("ui",),
    "leaderstats": ("leaderstats",),
    "data saving": ("datastore", "persistence"),
    "tools": ("tools",),
    "NPCs": ("npcs",),
    "combat": ("combat",),
    "inventories": ("inventory",),
    "currencies": ("currency",),
    "shops": ("shops",),
    "round systems": ("rounds",),
    "matchmaking": ("matchmaking",),
    "gamepasses": ("gamepasses",),
    "developer products": ("developer products",),
    "debugging": ("debugging",),
    "optimization": ("optimization",),
    "common beginner mistakes": ("beginner",),
    "advanced architecture": ("advanced-architecture", "architecture"),
    "code explanations": ("explanations", "code-review"),
    "natural-language conversion": ("natural-language-to-code", "natural-language"),
    "fixing/reviewing/improving code": ("bug-fix", "code-review", "optimization"),
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Audit seed variety and held-out evaluation isolation")
    value.add_argument("--seeds", default="raw_data/roblox_luau_seed_tasks.jsonl")
    value.add_argument("--evaluation", default="evaluation_data/roblox_luau_eval.jsonl")
    value.add_argument("--output", default="reports/catalog_audit.json")
    value.add_argument("--fail-on-missing", action="store_true")
    return value


def record_terms(record: dict[str, Any]) -> set[str]:
    metadata = [*record.get("tags", []), *record.get("concepts", [])]
    return {str(value).lower() for value in metadata}


def run(arguments: argparse.Namespace) -> int:
    seeds = list(read_jsonl(arguments.seeds))
    tasks = list(read_jsonl(arguments.evaluation))
    seed_errors = {
        str(seed.get("id")): validate_seed(seed)
        for seed in seeds
        if validate_seed(seed)
    }
    validate_evaluation_tasks(tasks)
    ids = [str(seed.get("id")) for seed in seeds]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    prompt_keys: dict[str, list[str]] = {}
    for seed in seeds:
        prompt_keys.setdefault(" ".join(normalized_tokens(str(seed.get("user_request", "")))), []).append(str(seed.get("id")))
    duplicate_prompts = {key: values for key, values in prompt_keys.items() if len(values) > 1}
    all_terms = set().union(*(record_terms(seed) for seed in seeds)) if seeds else set()
    missing = {
        theme: alternatives
        for theme, alternatives in REQUIRED_THEMES.items()
        if not any(term in all_terms for term in alternatives)
    }
    task_types = Counter(str(seed.get("task_type")) for seed in seeds)
    difficulty = Counter(str(seed.get("difficulty")) for seed in seeds)
    report = {
        "stage": "catalog_audit",
        "created_at": utc_now(),
        "seed_file": arguments.seeds,
        "evaluation_file": arguments.evaluation,
        "seed_count": len(seeds),
        "held_out_evaluation_count": len(tasks),
        "task_type_counts": dict(sorted(task_types.items())),
        "difficulty_counts": dict(sorted(difficulty.items())),
        "covered_themes": sorted(set(REQUIRED_THEMES) - set(missing)),
        "missing_themes": missing,
        "invalid_seed_records": seed_errors,
        "duplicate_seed_ids": duplicate_ids,
        "exact_duplicate_seed_prompts": list(duplicate_prompts.values()),
        "evaluation_ids_in_seed_catalog": sorted(set(ids) & {str(task.get("id")) for task in tasks}),
        "status": "pass" if not (missing or seed_errors or duplicate_ids or duplicate_prompts) else "fail",
    }
    write_json_atomic(arguments.output, report)
    print(
        f"Catalog audit: {report['status']} — {len(seeds)} seeds, {len(tasks)} held-out tasks, "
        f"{len(missing)} missing themes. Report: {arguments.output}"
    )
    if arguments.fail_on_missing and report["status"] != "pass":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"catalog audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
