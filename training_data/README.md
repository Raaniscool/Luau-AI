# Training data (local, ignored)

`build_datasets.py` writes canonical chat JSONL files here:

- `final_dataset.jsonl`: all accepted training records;
- `train.jsonl`: deterministic training split;
- `validation.jsonl`: deterministic development split, distinct from the held-out
  evaluation suite.

Never manually add evaluation prompts or baseline answers here.
