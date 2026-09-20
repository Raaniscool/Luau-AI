# Review-required targeted source briefs

`generate_targeted_briefs.py` writes deterministic source-task **drafts** here only after a
stored failure database contains evidence for the requested category. Drafts contain no answer,
are not active source catalogs, and cannot be fed directly to candidate generation because their
provenance is intentionally `training_factory_targeted_draft`.

A named human reviewer must make an explicit accept/reject decision for every draft. Use
`promote_targeted_briefs.py` to create a **separate** reviewed seed catalog after local
near-duplicate and held-out wording checks. Promotion does not append to either active catalog
and the promoted source brief must still pass the existing Builder → Reviewer/Tester → Fixer,
deduplication, and final dataset gates.

Generated draft artifacts are ignored by Git. Do not place evaluation prompts, rubrics, expected
answers, baseline answers, score reports, verified knowledge, or Code Book text here.
