"""Capture the pre-fine-tuning baseline of the original local Ollama model.

This intentionally delegates to the generic held-out evaluator but fixes the run label to
`baseline`. It does not and cannot prove a user-supplied Ollama tag has never been modified;
run it against the untouched `qwen3:4b` tag before creating any specialized tag.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import sys

from scripts.run_evaluation import main as evaluation_main


def main(argv: list[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if "--help" not in supplied and "-h" not in supplied:
        print("Baseline mode: evaluate the untouched original qwen3:4b tag before fine-tuning.")
    # argparse uses the last occurrence, so an explicit user --model after this default
    # intentionally overrides it for a differently named original base tag.
    return evaluation_main(["--run-kind", "baseline", "--model", "qwen3:4b", *supplied])


if __name__ == "__main__":
    raise SystemExit(main())
