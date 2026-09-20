# DukeOTR execution boundary and training-machine runbook

## Purpose and current status

This document separates **repository preparation** from **actual model training**. DukeOTR
has no trained adapter, no `dukeotr-v1` Ollama candidate, and no `dukeotr` release alias at
this time. Nothing in this document, the configuration files, or a generated plan is evidence
that a training run occurred.

The project is deliberately designed so an eventual CUDA/cloud machine is the major missing
execution resource—not a hidden dependency on Arena or on the user's local Windows Ollama
storage.

## Execution boundary

| Environment | Permitted role | Must not be assumed to do |
|---|---|---|
| **Windows workstation** | Keep the existing `qwen3:4b` Ollama tag intact; run local inference, baseline capture, prompt experiments, live candidate generation/review, and pre/post-training Ollama evaluation. | QLoRA/LoRA training, CUDA conversion work, or access by Arena through `localhost`. |
| **Arena** | Develop and test repository code, source briefs, Code Book, static validators, dataset/evaluation formats, training scripts, configs, documentation, and GitHub changes. | Access the Windows Ollama service, download/use its local model cache, run real QLoRA, or claim DukeOTR training. |
| **GitHub** | Store the reproducible source project, configs, tests, and documentation. | Store downloaded base weights, adapters, checkpoints, generated corpora, or private/local reports by default. |
| **CUDA/cloud training machine** | Obtain the matching Hugging Face base, receive a hash-verified final dataset separately, run actual LoRA/QLoRA, retain outputs/reports, and perform compatible conversion work. | Treat an Ollama quantized blob as the Hugging Face training base or publish a DukeOTR tag before evaluation. |

Arena can improve the **generator and dataset pipeline**, but a live call to `qwen3:4b` must
run on the Windows workstation (or another explicitly configured inference host). The Arena
sandbox has no route to the Windows machine's `localhost` service.

## Artifact flow

```text
Arena / GitHub
  source briefs, validators, evaluation suite, training code, configs, tests
                    │
                    ▼
Windows workstation
  existing qwen3:4b → small live generation/review pilots → human inspection
                    │
                    ▼
quality-gated, versioned local final dataset + manifest + SHA-256 hashes
                    │
                    ▼
ignored transfer bundle (not committed to Git)
                    │
                    ▼
CUDA/cloud training machine
  matching Qwen/Qwen3-4B HF base → QLoRA adapter → report/checkpoints
                    │
                    ▼
verified compatible conversion/export
                    │
                    ▼
Windows Ollama
  dukeotr-v1 candidate evaluation → human release approval → dukeotr alias
```

The final JSONL is intentionally ignored by Git. That avoids accidentally publishing a large,
reviewable corpus or mixing local artifacts with source code. It must be transferred to the
training host separately with its manifest and hashes.

## Training input contract

A real first training dataset must be explicitly built as `dukeotr_dataset_v1`; a pilot output
or a source-brief catalog is not enough.

Required directory layout on the training machine after bundle extraction:

```text
training_data/dukeotr_dataset_v1/
  dataset_manifest.json
  final_dataset.jsonl
  train.jsonl
  validation.jsonl
```

`dataset_manifest.json` must state:

- `dataset_version: "dukeotr_dataset_v1"` and `dataset_version_status: "created"`;
- a current static-checker/reviewer/dedupe/held-out-isolation quality-gate description;
- train/development/final partition counts; and
- SHA-256 values for `train.jsonl`, `validation.jsonl`, and `final_dataset.jsonl`.

Each record is canonical chat JSONL:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "split": "train",
    "dataset_partition": "train"
  },
  "quality": {"static": {}, "llm_review": {}, "deduplication": {}}
}
```

The trainer rechecks that every row remains quality-gated, that train/development partition
labels are correct, that no evaluation source ID is present, and that all dataset-file hashes
match the final-dataset manifest. It will not train from arbitrary chat-shaped JSONL.

Training Factory failure records, Builder/reviewer/fixer lineage ledgers, targeted source-task
drafts, verified knowledge entries, Code Book cards, baseline answers, scoring reports, and
held-out evaluation material are explicitly **not** bundle inputs. They remain separate until a
human-reviewed canonical candidate independently clears every final-data gate. See
[`training_factory/README.md`](../training_factory/README.md).

## Build and package only after the data release gate

Do not run these commands for `_001`, `_002`, an unreviewed pilot, or an unversioned local
output. Once a human has reviewed a true final-data candidate and deliberately approves the
first dataset version, build it explicitly on the dataset assembly machine:

```powershell
python .\scripts\build_datasets.py `
  --input .\validated_data\<reviewed-run>.deduplicated.jsonl `
  --evaluation .\evaluation_data\roblox_luau_eval.jsonl `
  --output-dir .\training_data\dukeotr_dataset_v1 `
  --dataset-version dukeotr_dataset_v1 `
  --strict

python .\scripts\prepare_training_bundle.py `
  --config .\configs\qlora_sft.json `
  --output .\artifacts\dukeotr_v1_training_bundle.zip
```

The second command verifies the dataset manifest hashes and creates an **ignored** ZIP with:

- the three final JSONL artifacts and their manifest;
- `configs/qlora_sft.json`;
- `requirements/training.txt` and `pyproject.toml`; and
- a transfer manifest containing the dataset/config SHA-256 values and Git revision.

It does not contact Ollama, download Qwen, train an adapter, create a checkpoint, or create an
Ollama model. Transfer the ZIP through a deliberate private channel appropriate for the data
(for example, controlled object storage or `scp`); do not commit it to Git.

## CUDA/cloud machine requirements

The repository's conservative QLoRA profile in `configs/qlora_sft.json` expects:

| Resource | Project expectation |
|---|---:|
| GPU | NVIDIA CUDA-capable GPU |
| VRAM | 16 GiB minimum guard; 24 GiB or more preferred |
| Host RAM | 32 GiB recommended for a less fragile first run |
| Storage | Plan at least 50 GiB free for Python environments, Hugging Face cache, base files, checkpoints, reports, and the transfer bundle; more is prudent for repeated runs |
| Training base | `Qwen/Qwen3-4B` Hugging Face/Transformers weights, not `qwen3:4b` Ollama blobs |
| Adapter method | QLoRA first: frozen 4-bit NF4 base plus PEFT LoRA adapters |

The exact usable capacity depends on driver, CUDA, PyTorch, sequence lengths, checkpointing,
and other processes. `scripts/preflight_hardware.py` is a guardrail and evidence record, not
a guarantee that every GPU/software combination will work. `configs/lora_sft.json` is the
unquantized LoRA alternative for a larger-memory setup; it uses the same versioned-data
manifest contract, but QLoRA remains the first planned experiment.

## Reproducible environment setup

On the future CUDA/cloud machine, clone the recorded branch/revision first. The bundle manifest
records the source Git revision used when the data handoff was prepared.

```bash
# Clone the approved DukeOTR repository, then check out the exact revision recorded in the
# transfer bundle's training_bundle_manifest.json.
git clone https://github.com/Raaniscool/Luau-AI.git DukeOTR
cd DukeOTR
git checkout <recorded-git-revision>

# Inspect the bundle manifest, then extract at the repository root.
unzip /secure/path/dukeotr_v1_training_bundle.zip

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CUDA-compatible PyTorch wheel selected for the destination driver's CUDA version
using the official PyTorch installation instructions. Then install the remaining pinned-range
training dependencies:

```bash
pip install -r requirements/training.txt
```

Before any model download or execution, verify source, data, and hardware:

```bash
python -m compileall -q scripts
python scripts/audit_dukeotr_curriculum.py --strict
python scripts/audit_catalog.py --fail-on-missing
python scripts/audit_code_book.py --strict
python -m scripts.audit_training_factory --strict
python -m unittest discover -v

python -m scripts.preflight_hardware \
  --config configs/qlora_sft.json \
  --require-suitable \
  --output reports/hardware_preflight.json

# Plan/validate only. This must say not_executed.
python -m scripts.train_qlora \
  --config configs/qlora_sft.json \
  --report reports/training/dukeotr_v1/training_plan.json
```

The plan is not training. It records whether the versioned manifest, train JSONL, validation
JSONL, and hardware are present. Inspect it before opting into `--execute`.

## Base-model and tokenization contract

- **Training base:** `Qwen/Qwen3-4B` from Hugging Face.
- **Requested revision:** configured in `configs/qlora_sft.json`. The trainer records the
  resolved immutable commit hash in the real training report. Preserve that hash and use it
  for exact reruns; do not describe an unrecorded moving `main` revision as fully reproducible.
- **Model format:** Hugging Face Transformers model/tokenizer files (`safetensors` where
  supplied by the upstream base) plus a PEFT adapter output. An Ollama quantized model cache
  is not the training input.
- **Chat formatting:** the base tokenizer's own `apply_chat_template` is used with
  `enable_thinking=False` where supported.
- **Loss:** system and user tokens are masked; only assistant-completion tokens contribute to
  SFT loss.
- **Length policy:** rows beyond `max_seq_length` are excluded whole rather than truncating
  the assistant ending. The training report records drop counts.

## QLoRA command sequence

Start with a small, reviewed pilot only after the plan and hardware preflight pass:

```bash
python -m scripts.train_qlora \
  --config configs/qlora_sft.json \
  --max-train-samples 64 \
  --max-eval-samples 16 \
  --execute \
  --report reports/training/dukeotr_v1/pilot_training.json
```

After inspecting the pilot's loss, overlength-drop counts, adapter files, dependency versions,
base revision, and held-out behavior, a deliberately approved full run uses the same config
without sample limits:

```bash
python -m scripts.train_qlora \
  --config configs/qlora_sft.json \
  --execute \
  --report reports/training/dukeotr_v1/full_training.json
```

The configured output location is:

```text
models/adapters/dukeotr_v1-qlora/
```

This is a PEFT adapter-only output. The training report records base/tokenizer revisions,
dataset/config hashes, source Git revision, runtime/dependency versions, hardware,
configuration, tokenization policy, metrics, and output path. A successful exit still does
not establish that DukeOTR is better than the base model.

## Evaluation and Ollama packaging

1. Preserve the real adapter, report, config, dataset manifest, base revision, and conversion
   notes outside Git.
2. Run compatible conversion or merge work using the **exact** Hugging Face base and actual
   PEFT adapter. A QLoRA adapter is not automatically compatible with the existing Windows
   `qwen3:4b` Ollama quantization.
3. Only after independently verifying a matching GGUF/adapter or merged export may you use:

   ```powershell
   python .\scripts\prepare_ollama_modelfile.py `
     --base C:\models\matching-qwen3-4b.gguf `
     --adapter C:\models\verified-dukeotr-adapter.gguf `
     --name dukeotr-v1 `
     --release-alias dukeotr
   ```

4. Review the generated Modelfile and run `ollama create dukeotr-v1 -f Modelfile` manually on
   the Windows machine. Then evaluate the candidate against the unchanged held-out suite.
5. Only after held-out comparison, no unresolved security regression, and human release
   approval may `dukeotr` be created as the stable public alias for `ollama run dukeotr`.

Never create a placeholder adapter, fabricate a training report, or call a planned name a
trained DukeOTR model.
