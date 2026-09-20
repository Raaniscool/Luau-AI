# DukeOTR Roblox/Luau Code Book

The DukeOTR Code Book is a compact, versioned, source-attributed knowledge layer for the project.
It is deliberately different from the fine-tuning corpus:

- **Cards are factual reference material**, not automatically supervised-training rows.
- **Source links and caveats travel with each claim** so a future Reviewer can challenge an
  API assertion instead of repeating it confidently.
- **Source-checked is not human-verified.** Cards begin as `source_checked`; promote a card
  to `human_verified` only after a named reviewer checks current official documentation and
  practical Studio behavior.
- **Cards may become stale.** Create a revision or mark the old card `deprecated`; do not
  silently overwrite historical claims.

## Files

- `schema.json` — versioned card contract.
- `roblox_luau_cards.jsonl` — the 25-card source-checked foundation; the strict audit also
  checks a granular DukeOTR readiness coverage matrix.
- `sources.md` — primary documentation source policy and links.

## Commands

Run these from the repository root:

```powershell
python .\scripts\audit_code_book.py --strict
python .\scripts\query_code_book.py --query "secure RemoteEvent shop purchase" --format markdown
```

The query command produces cited reference context only. It does not call Ollama, generate
code, modify training data, or establish that a proposed solution is safe.

## Promotion workflow

```text
draft → source_checked → human_verified → deprecated
```

A `human_verified` card must name a reviewer and verification date. The initial cards are
`source_checked` based on official Roblox / Luau sources and retain caveats that a Reviewer
should check against the current documentation.
