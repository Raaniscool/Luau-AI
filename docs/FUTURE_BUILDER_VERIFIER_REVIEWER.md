# DukeOTR future Builder → Reviewer → Fixer architecture

This is a **future interface contract**, not a prematurely implemented autonomous system.
The curriculum, data-quality gates, baseline, and held-out evaluation path must work first.

## Intended loop

```text
User request
  ↓
DukeOTR Builder produces solution + assumptions + test/security notes
  ↓
DukeOTR Reviewer checks requirements, Roblox API usage, authority boundaries, and code risks
  ↓
DukeOTR Fixer applies concrete review findings
  ↓
Reviewed revision + visible caveats
  ↓
Measured held-out evaluation (never training input)
```

`configs/builder_verifier_reviewer.json` retains its historical filename but now defines the
first machine-readable Builder/Reviewer/Fixer seam. It is not an orchestrator and does not
run models, write training data, or auto-promote a revision.

## Role contracts

### Builder

**Input:** user request, constraints, prior feedback, and explicitly selected Code Book card
IDs/revisions.

**Output must include:**

- answer/code and file/script placement;
- explicit assumptions;
- client/server authority map;
- security notes for remotes, state, currency, inventory, combat, or purchases;
- test plan and a structured summary for review; and
- consulted Code Book card IDs/revisions with caveats preserved.

The Builder must not claim an unrun Roblox test passed or present a Code Book excerpt as a
substitute for current API verification.

### Reviewer

**Input:** original request, Builder output, and cited Code Book card revisions.

**Review checks include:**

- each stated requirement and unresolved ambiguity;
- Luau syntax/idiom plausibility and Roblox API names;
- Script/LocalScript/ModuleScript placement;
- untrusted RemoteEvent/RemoteFunction inputs;
- server authority for economic, inventory, combat, access, and purchase decisions;
- Instance/class/ancestry/distance/ownership validation where relevant;
- event connection, lifetime, yielding, and performance risks; and
- testability and failure paths.

**Output:** machine-readable findings with severity, observable evidence, recommendation,
priority, Code Book citations/caveats, and a verdict. The Reviewer does not silently replace
the solution; that responsibility remains visible in a Fixer revision.

### Fixer

**Input:** original request, Builder solution, Reviewer report, and cited Code Book context.

**Output:** a revised self-contained solution, changes made, any remaining assumptions, and
unresolved risks. The Fixer must address each concrete blocking finding or state why a
requirement remains unresolved. It must not turn reviewer prose into an unsupported claim or
silently discard an authority boundary.

## First safe incremental implementation

Only after a model has measurable evaluation results, implement a **single-turn prototype**
that emits schema-validated JSON traces for these three roles and logs all inputs/outputs
locally. Do not auto-execute Luau, publish models, mutate game files, or allow any role to
see held-out evaluation rubrics, base answers, or score artifacts. Add multi-round feedback
only after human inspection shows that traces, citations, correction provenance, and
isolation gates are reliable.
