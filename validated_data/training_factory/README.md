# Training Factory lineage sidecars

This local-only directory is for `scripts.build_training_ledger` output. Each ledger row joins
an original Builder candidate with its static/reviewer evidence, any Fixer correction,
correction explanation, failure labels, and verification status. It is an audit companion to,
not a replacement for, canonical candidates.

Ledger files cannot be used as training input and do not automatically promote a candidate.
Canonical final-data eligibility remains enforced by the normal static, reviewer, deduplication,
and held-out-isolation gates in `scripts.build_datasets`.
