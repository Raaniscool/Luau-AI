# Fine-tuning research notes (2026-09-19)

## Decision summary

The appropriate first training method is adapter-based SFT, specifically QLoRA where a
suitable CUDA GPU is available. QLoRA freezes a 4-bit quantized base and trains small
low-rank adapters, substantially reducing memory compared with full-model fine-tuning.
This matches the project's specialization goal and avoids retraining general language
ability from zero.

It is **not** a recommendation to run QLoRA on the stated 2 GB integrated GPU. The project
separates local inference/data work from cloud/suitable-CUDA adapter training and enforces
that separation in code.

## Source observations

1. The Qwen3 release states that dense Qwen3 models including 4B are open-weighted under
   Apache-2.0. The exact model repository carries the Apache-2.0 license text.
2. Hugging Face's TRL documentation describes QLoRA as 4-bit frozen base weights plus
   LoRA adapters, and documents NF4/double quantization and PEFT setup.
3. Hugging Face's bitsandbytes documentation describes CUDA support and alternative Intel
   backends. Backend availability alone is not a memory/performance guarantee for an
   integrated shared-memory GPU.
4. Ollama's current import documentation supports base models/adapters through
   `FROM`/`ADAPTER`, while warning that adapters must use the same base and that non-QLoRA
   adapters are generally safest because quantization representations differ.

## Links

- https://qwen.ai/blog?id=qwen3
- https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE
- https://huggingface.co/docs/trl/peft_integration
- https://huggingface.co/docs/transformers/quantization/bitsandbytes
- https://github.com/ollama/ollama/blob/main/docs/import.mdx
- https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx

Always re-check current upstream compatibility/version notes immediately before an actual
cloud training or export run.
