# Training data (local, ignored)

`build_datasets.py` writes canonical chat JSONL files here. A real named release should use
its own ignored directory, for example `training_data/dukeotr_dataset_v1/`:

- `final_dataset.jsonl`: all accepted training records;
- `train.jsonl`: deterministic training split;
- `validation.jsonl`: deterministic development split, distinct from the held-out
  evaluation suite; and
- `dataset_manifest.json`: quality-gate detail, partition counts, and SHA-256 values for all
  three JSONL files.

A named dataset is created only by explicitly passing
`--dataset-version dukeotr_dataset_v1` after review. It remains a local artifact—not a Git
commit. Use `scripts/prepare_training_bundle.py` to create a separately transferable,
hash-verified bundle for a future CUDA/cloud training machine. Never manually add evaluation
prompts or baseline answers here.
