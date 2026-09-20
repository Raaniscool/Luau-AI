# DukeOTR hardware decision and QLoRA/LoRA preparation

> **Execution boundary:** the Windows workstation owns its existing `qwen3:4b` Ollama
> inference/evaluation workflow; Arena develops the repository but cannot use that local model
> or run QLoRA; real adapter training belongs to a suitable CUDA/cloud host. The exact
> handoff commands, data-hash contract, and output/evaluation procedure are in
> [TRAINING_MACHINE_RUNBOOK.md](TRAINING_MACHINE_RUNBOOK.md).

## Decision for the stated Windows machine

| Work | Suitable locally? | Why |
|---|---:|---|
| Run `ollama run qwen3:4b` / answer prompts | Yes | The model has already run through Ollama on the PC. |
| Generate candidate examples | Yes, slowly is acceptable | Uses the local Ollama inference API. |
| Static validation, LLM review, deduplication, dataset assembly | Yes | The core pipeline is standard-library Python plus existing local inference. |
| Capture baseline / candidate evaluation | Yes | Uses the same local Ollama inference service. |
| Train a reliable Qwen3-4B LoRA/QLoRA adapter | **No, not recommended** | About 11.7 GB system RAM and an Intel integrated GPU with about 2 GB shared/available VRAM do not provide a practical, validated training budget. |
| Merge/export a large trained base and adapter | Generally cloud/suitable GPU | Requires matching full weights, conversion tools, and additional memory/storage. |

The fact that a quantized model can perform local inference does **not** mean it can be
fine-tuned there. Training also needs activations, gradients for adapters, optimizer state,
sequence buffers, and framework/kernel support. The supplied training entry point
intentionally requires an NVIDIA CUDA device with at least the configured **16 GiB** VRAM
recommendation; 24 GiB gives more practical headroom for experiments. This is a conservative
project guardrail, not a claim that every conceivable low-memory research experiment is
impossible.

Current bitsandbytes documentation lists experimental/alternative Intel backends, but that
does not make a roughly 2 GB shared-memory integrated GPU an appropriate target for this
4B SFT configuration. The project does not silently use an untested CPU fallback and call
it a successful training run.

Run the actual inspection on the destination machine:

```powershell
# Run from the repository root, where .\scripts\preflight_hardware.py exists.
python .\scripts\preflight_hardware.py --config .\configs\qlora_sft.json --print-json
python .\scripts\preflight_hardware.py --config .\configs\qlora_sft.json --require-suitable
```

## Recommended training method

Start with **supervised fine-tuning (SFT) using QLoRA**:

- Public specialization identity: **DukeOTR**; planned adapter/output version: `dukeotr_v1`.
- Technical base: `Qwen/Qwen3-4B` from Hugging Face, not an opaque Ollama quantized blob.
- Base weights: loaded in 4-bit NF4 and frozen.
- Trainable parameters: LoRA adapters on attention and MLP projections only.
- Defaults: rank 32, alpha 64, dropout 0.05, 2,048-token maximum, batch size 1, gradient
  accumulation 16, gradient checkpointing, and two epochs.
- Dataset: only `training_data/dukeotr_dataset_v1/train.jsonl`; development checks use
  `training_data/dukeotr_dataset_v1/validation.jsonl`; the versioned manifest hashes both
  files and held-out evaluation never enters SFT.

This is appropriate for the first domain-specialization experiment because it minimizes
trainable state and lets the base general capabilities remain mostly intact. It is not an
argument for full fine-tuning, pretraining, or training from scratch.

`configs/qlora_sft.json` is the default. `configs/lora_sft.json` is an unquantized adapter
alternative for a larger GPU; QLoRA is preferred first.

## Suitable cloud/CUDA workflow

Use a CUDA-capable environment with enough VRAM (project baseline: 16 GiB minimum;
preferably 24 GiB or more), **32 GiB host RAM recommended**, and adequate storage (plan at
least 50 GiB free for caches/checkpoints/outputs). Linux-based cloud environments tend to be
the least fragile route for the current Transformers/PEFT/bitsandbytes stack. The versioned
final dataset is not committed to Git; transfer a hash-verified bundle after its manual data
release gate, as documented in [TRAINING_MACHINE_RUNBOOK.md](TRAINING_MACHINE_RUNBOOK.md).

```bash
# On the suitable cloud/CUDA machine, from this repository:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/training.txt

# Move only reviewed, quality-gated data and config to this environment.
python -m scripts.preflight_hardware --require-suitable
python -m scripts.train_qlora --config configs/qlora_sft.json
# The previous command writes a plan only.
python -m scripts.train_qlora --config configs/qlora_sft.json --execute
```

The trainer validates canonical message roles, refuses missing/empty data, renders the
model's chat template, masks system/user tokens from completion loss, drops overlength rows
rather than silently losing assistant endings, and writes adapter-only output beneath
`models/adapters/`. It creates a completion report **only after** training returns.

Start with a small, reviewed pilot (`--max-train-samples`) before a full run. Inspect:

- count of rows dropped for sequence length;
- training/evaluation loss for instability or overfit signals;
- adapter config and exact base revision;
- held-out answers and critical security findings; and
- comparison to the pre-training baseline.

A successful process exit is not a capability claim.

## Ollama export compatibility

Ollama can import model/adapters through a `Modelfile`, but the adapter must match the
exact base model it was trained against. Ollama's own import documentation warns that
mismatched bases lead to erratic behavior and that non-QLoRA adapters are generally the
safer import path. A local `qwen3:4b` tag is commonly a quantized inference artifact; it
is **not automatically the same weight representation** used by HF QLoRA training.

Do not point an adapter at `FROM qwen3:4b` just because the names look similar. Instead:

1. retain the exact `Qwen/Qwen3-4B` revision/adaptor config from training;
2. choose and test a supported conversion/merge path (often matching HF base → merged or
   verified adapter → GGUF);
3. ensure the final base representation and adapter are compatible; and
4. evaluate the imported Ollama model on the held-out suite before use.

After that compatibility work, this helper writes a reviewed template but does not execute
`ollama create` for you:

```powershell
python .\scripts\prepare_ollama_modelfile.py --base C:\models\matching-qwen3-4b.gguf --adapter C:\models\verified-dukeotr-adapter.gguf --name dukeotr-v1 --release-alias dukeotr
```

The helper writes a DukeOTR-branded system prompt plus a provenance manifest, but it still
does not create either Ollama tag. Evaluate `dukeotr-v1` first. Only after the held-out
comparison and a human release decision may the stable public `dukeotr` alias be created,
making `ollama run dukeotr` valid.

`models/` is ignored: do not commit weights, adapter blobs, GGUFs, checkpoints, or Ollama
files into Git by default.

## License and source references

Verify terms before distribution, especially after combining third-party datasets or
weights. As of the project research date, Qwen's Qwen3 open-weight models including
Qwen3-4B are published under Apache-2.0; preserve notices and verify the exact revision.

Primary references consulted on 2026-09-19:

- [Qwen3 release and license statement](https://qwen.ai/blog?id=qwen3)
- [Qwen/Qwen3-4B Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE)
- [Hugging Face TRL PEFT/QLoRA integration](https://huggingface.co/docs/trl/peft_integration)
- [Hugging Face bitsandbytes hardware compatibility](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [Ollama model/adapter import documentation](https://github.com/ollama/ollama/blob/main/docs/import.mdx)
- [Ollama Modelfile adapter documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx)
