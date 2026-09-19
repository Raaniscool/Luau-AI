# DukeOTR models and adapters (local only)

Do not commit downloaded base models, Ollama blobs, checkpoints, adapters, merged weights, or
GGUF files. This directory is the local destination for DukeOTR training and export artifacts.

The planned versioned candidate is `dukeotr-v1`; the stable user-facing release alias is
`dukeotr`, intended for `ollama run dukeotr` **only after** compatible export, held-out
evaluation, and human release approval. Neither tag exists merely because this directory or a
configuration file exists.

Keep the exact Qwen base revision, conversion details, license/notice material, adapter
configuration, dataset manifest hash, and evaluation report alongside any real artifact. See
[MODEL_IDENTITY.md](../docs/MODEL_IDENTITY.md) and
[HARDWARE_AND_FINETUNING.md](../docs/HARDWARE_AND_FINETUNING.md) for the naming and
compatibility requirements.
