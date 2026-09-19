# DukeOTR dataset governance

## Allowed tracked inputs in Phase 1

- Project-authored task briefs in `raw_data/`.
- Project-authored held-out tasks and rubrics in `evaluation_data/`.
- Small, intentionally reviewed test fixtures.

## Before adding an external source

Record all of the following in a manifest adjacent to the source:

1. source URL/repository and pinned revision or retrieval date;
2. license and whether it permits training, derivative works, and redistribution;
3. whether the author/game owner gave permission;
4. whether it contains private, personal, paid, leaked, or credential-bearing material;
5. transformation steps and attribution obligations; and
6. a reviewer decision.

Do not copy private Roblox experiences, plugin source with unclear rights, chat logs,
Roblox credentials/cookies, exploit scripts, or other unsafe code into any pipeline stage.

## Generated data and version claims

Synthetic output inherits all the usual quality risks: hallucinated API calls, obsolete
patterns, security mistakes, repetitive language, and benchmark contamination. It must
retain source-model/settings metadata and pass the documented review gates. Synthetic data
is not evidence that code was executed or works in Roblox Studio.

Do not label an artifact `dukeotr_dataset_v1` merely because it was generated. That version
name is reserved for a final, quality-gated, deduplicated corpus with a recorded manifest,
provenance, and evaluation-isolation evidence. Likewise, no adapter/model can be called
`dukeotr_v1` without its actual compatible training artifact and held-out evaluation record.

## Evaluation integrity

Never feed the evaluation prompt file, rubrics, baseline answers, candidate answers, or
scoring reports to the generator or SFT trainer. Evaluation concepts may overlap training
concepts, but wording and target scenarios must remain held out.
