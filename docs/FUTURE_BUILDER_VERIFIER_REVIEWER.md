# Future builder → verifier → reviewer architecture

This is a **future interface contract**, not a prematurely implemented autonomous system.
The data/evaluation pipeline must work first.

## Intended loop

```text
User request
  ↓
Builder produces solution + assumptions + test/security notes
  ↓
Verifier checks requirements, Roblox API usage, authority boundaries, and code risks
  ↓
Reviewer prioritizes weaknesses and revision requirements
  ↓
Builder receives structured feedback and revises
  ↓
Final solution + visible caveats
```

`configs/builder_verifier_reviewer.json` defines the first machine-readable input/output
shape and invariants.

## Role contracts

### Builder

Input: user request, constraints, prior feedback.

Output must include:

- answer/code and file/script placement;
- explicit assumptions;
- client/server authority map;
- security notes for remotes, state, currency, inventory, combat, or purchases;
- test plan; and
- a structured summary suitable for a verifier.

The Builder must not claim an unrun Roblox test passed.

### Verifier

Input: original request and Builder output.

It should check:

- each stated requirement and unaddressed ambiguity;
- Luau syntax/idiom plausibility and Roblox API names;
- Script/LocalScript/ModuleScript placement;
- untrusted RemoteEvent/RemoteFunction inputs;
- server authority for economic, inventory, combat, access, and purchase decisions;
- Instance/class/ancestry/distance/ownership validation where relevant;
- event connection, lifetime, yielding, and performance risks; and
- testability and failure paths.

Output: machine-readable finding list with severity, evidence, recommendation, and verdict.
It should not write the final solution silently; that belongs to a visible Builder revision.

### Reviewer

Input: original request, candidate, and verifier report.

Output: strengths, prioritized weaknesses, missing requirements, regression concerns, and
an explicit `revision_required` flag. The reviewer focuses on usefulness, maintainability,
and whether the verifier may have missed a user-facing flaw.

## First safe incremental implementation

After a model has measurable evaluation results, implement a **single-turn prototype** that
emits JSON for the three roles and logs all inputs/outputs locally. Do not auto-execute
Lua, publish models, mutate game files, or let the Builder see held-out evaluation rubrics.
Only add multi-round feedback once role outputs are schema-validated and a human can inspect
the trace.
