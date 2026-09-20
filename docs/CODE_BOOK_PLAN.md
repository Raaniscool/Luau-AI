# DukeOTR Code Book proof-of-concept and expansion plan

## Purpose

The DukeOTR Code Book is a versioned, source-attributed knowledge layer for the Roblox/Luau
specialization project. It is **not** automatically treated as fine-tuning data and it is
not an excuse to trust generated code. Its first job is to give future Builder, Reviewer,
and Fixer roles a small set of independently auditable facts and implementation checklists.

## Scope of this proof of concept

1. Create a `code_book/` directory with a documented card schema and provenance rules.
2. Add a deliberately small, reviewed starter set of cards covering the highest-risk
   areas: remote communication, server authority, Luau types/tables, module boundaries,
   data persistence, UI, inventory/shop transactions, combat/NPC validation, debugging,
   and optimization.
3. Require an official primary source URL, a verification date, a clear scope, known
   pitfalls, and reviewer checks on every `verified` card.
4. Add an offline audit command that validates card structure, source policy, duplicate
   IDs, claim coverage, and status values.
5. Add an offline lexical retrieval proof of concept that produces only source-attributed
   context for a query. It does not call Ollama, modify training data, or claim an answer
   is correct.
6. Strengthen the held-out RemoteEvent task with explicit regression checks for the known
   baseline errors: clients **can** call `FireServer`, and a RemoteEvent does **not** make
   client input secure.

## Curated expansion status and next batches

The original 12-card starter set is now a **25-card source-checked foundation**. It includes
narrow cards for values/scope/nil/operators; strings/math/randomness; annotations, typed
tables, unions, and callbacks; players/character lifecycle; GUI/input boundaries;
client/server replication containers; raycasting/physics validation; TweenService/RunService
performance; API-hallucination defenses; and Instance/hierarchy/service access. The older
control-flow/function, standard-library/error, event-connection, module, DataStore, remote,
and game-system cards remain preserved and have explicit granular-domain tags where needed.

`audit_code_book.py --strict` now checks an explicit coverage matrix for the requested Luau
and Roblox concepts and enforces a 25-card source-checked floor. It records card IDs for every
row so a missing concept is actionable instead of hidden behind a broad "Roblox APIs" label.

Next batches should deepen—not merely rename—the shallowest rows: exact API/version changes,
RaycastParams edge cases, DataStore retry/session behavior, UI accessibility/input modes,
physics/network-ownership trade-offs, and realistic debugging/refactoring cases. Add these in
small source-checked batches and raise the configured floor only with a documented curation
reason.

Each new card should make its **purpose**, syntax/pattern, practical use, common mistakes,
reviewer checks, related APIs/concepts, appropriate difficulty, and any security caveat
explicit. Only claims with a cited authoritative source may be described as official API or
language behavior; design advice must remain an `engineering_pattern`. Where a topic has no
meaningful security angle or "when not to use" rule, say so rather than inventing one.

## Card lifecycle

```text
draft → source_checked → human_verified → deprecated
```

Only `source_checked` and `human_verified` cards are returned by default retrieval. A card
becomes `human_verified` only after an identified reviewer confirms that its claims match its
cited, current source. Changes to Roblox APIs should create a new revision or mark an old card
deprecated rather than silently rewriting historical facts.

## Guardrails

- Prefer official Roblox Creator Hub / API reference and official Luau documentation.
- Do not copy private game code, leaked content, exploit code, or unlicensed third-party
  tutorials into the Code Book.
- Do not use a Code Book card as a substitute for server-side validation or live Studio
  testing.
- Do not feed held-out evaluation prompts, rubrics, baseline answers, or score reports into
  Code Book retrieval during model training.
- Do not automatically convert cards into SFT rows. A future curation step must decide which
  cards are suitable for instructional examples.

## Exit criteria for the proof of concept

- `python .\scripts\audit_code_book.py --strict` succeeds from a Windows repository root.
- `python .\scripts\query_code_book.py --query "secure RemoteEvent shop purchase"` returns
  only eligible, cited card excerpts with source-check status preserved.
- The baseline RemoteEvent evaluation explicitly rejects the two observed factual errors.
- No model download, inference generation, baseline run, or fine-tuning is required to
  create or audit the Code Book structure.
