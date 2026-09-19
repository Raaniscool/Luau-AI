# Code Book: source-attributed Roblox/Luau reference context

## What it is

The Code Book is a compact, structured reference layer for facts and review checklists that
are especially important for Roblox/Luau reliability. It is intentionally separate from:

- **training data** — cards are not automatically turned into supervised fine-tuning rows;
- **evaluation data** — held-out prompts/rubrics never enter card retrieval for training;
- **model output** — a generated answer is not a source; and
- **live Studio testing** — a source citation does not prove a game's specific code works.

The current source-checked foundation stores 15 cards in
`code_book/roblox_luau_cards.jsonl`. The cards cover:

- RemoteEvents, RemoteFunctions, client/server responsibility, and security;
- Luau types, tables, functions, control flow, standard-library iteration/type/error
  behavior, services, Instances, events/connections, and ModuleScripts/OOP;
- UI, DataStores, inventories, currencies, shops, combat, Tools, NPCs, rounds, and
  matchmaking;
- developer-product receipt handling; and
- debugging, optimization, common mistakes, and hallucination-resistant API review.

## Reliability model

Each card has a versioned schema, a status, explicit claims, caveats, implementation
patterns, common pitfalls, reviewer checks, and source URLs. In practice, the summary/title
state the concept and purpose; patterns carry correct/practical use; pitfalls capture
incorrect use and when-not-to-use guidance; claims/caveats carry API and security limits;
and domains/keywords identify related APIs or concepts. Do not infer an unstated API fact
from this structure.

| Status | Meaning | Default retrieval? |
|---|---|---:|
| `draft` | Editorial proposal; not ready for Builder/Reviewer context. | No |
| `source_checked` | Claim/source mapping was curated against cited official documentation. It is **not** a claim of human Studio verification. | Yes |
| `human_verified` | A named person recorded a verification date and review. | Yes |
| `deprecated` | Superseded or stale; kept for history. | No |

The initial cards use `source_checked`, not `human_verified`, so the repository does not
pretend that a human already validated every card in Studio.

## Audit before use

```powershell
# Run from the repository root.
python .\scripts\audit_code_book.py --strict
```

The audit verifies card schema, source-host policy, duplicate IDs, source references, claim
references, training policy, and required topic coverage. It cannot prove that an external
web page still says the same thing; re-check sources when an API changes or a Reviewer finds
a discrepancy.

## Deterministic retrieval proof of concept

```powershell
python .\scripts\query_code_book.py --query "secure RemoteEvent shop purchase" --format markdown
```

The result is lexical retrieval with source-preserving excerpts. It makes no Ollama request,
does not generate code, and does not automatically influence a training example. Its future
role is intentionally narrow:

```text
User request
  ↓
Builder receives relevant Code Book context with citations
  ↓
Builder proposes placement, code, and assumptions
  ↓
Reviewer checks against card pitfalls/checklists and current official docs
  ↓
Fixer revises only concrete findings
```

A future model prompt should identify card IDs/revisions used and explicitly say that cards
are reference context rather than infallible truth.

## Baseline RemoteEvent regression guard

The held-out task `eval-remoteevent-secure-001` now explicitly rejects both reported
baseline errors:

1. saying that clients cannot call `RemoteEvent:FireServer`; and
2. saying that a RemoteEvent automatically validates or secures client input.

This information stays in `evaluation_data/`, never in generated training prompts. The
source-checked RemoteEvent card cites the official RemoteEvent/remote networking and
security guidance that a future Reviewer should use to catch those claims.

## Curation rules

1. Prefer official Roblox Creator Hub / API reference and official Luau documentation.
2. Make an API fact an `official_api` or `official_security_guidance` claim only when its
   cited source supports it.
3. Mark project design advice as `engineering_pattern`; give caveats instead of presenting
   a design choice as a platform guarantee.
4. When an API changes, increment a card revision or deprecate it. Never silently rewrite
   a historical claim.
5. Do not add private game code, exploit scripts, unclear-license content, personal data,
   or raw model answers as authoritative knowledge.
6. Promote to `human_verified` only with a named reviewer and date.

See `code_book/README.md`, `code_book/schema.json`, and
[CODE_BOOK_PLAN.md](CODE_BOOK_PLAN.md) for the concrete contract and proof-of-concept exit
criteria.
