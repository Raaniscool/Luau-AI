"""Optional CUDA/cloud QLoRA/LoRA supervised fine-tuning entry point.

This script is intentionally conservative: without `--execute` it only validates data and
writes a plan. With `--execute`, it refuses the shipped QLoRA configuration unless a
suitable NVIDIA CUDA device was detected (unless an explicit unsafe override is supplied).
It never trains from an Ollama quantized model blob; it loads the matching Hugging Face base
model specified in the config and writes adapter weights only.
"""

from __future__ import annotations

# Support both `python -m scripts.name` and `python scripts/name.py` from the repo.
if __package__ in {None, ""}:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.lib.io_utils import read_json, read_jsonl, utc_now, write_json_atomic
from scripts.preflight_hardware import assess_hardware


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Plan or run Qwen3-4B LoRA/QLoRA SFT on suitable CUDA hardware")
    value.add_argument("--config", default="configs/qlora_sft.json")
    value.add_argument("--execute", action="store_true", help="Actually begin training; otherwise write a plan only")
    value.add_argument("--force-unsafe-hardware", action="store_true", help="Bypass CUDA/VRAM guard (not recommended)")
    value.add_argument("--max-train-samples", type=int, default=0, help="Pilot subset; 0 means all")
    value.add_argument("--max-eval-samples", type=int, default=0)
    value.add_argument("--resume-from-checkpoint", default=None)
    value.add_argument("--report", default=None)
    return value


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_training_records(path: str | Path, label: str) -> list[dict[str, Any]]:
    records = list(read_jsonl(path))
    if not records:
        raise ValueError(f"{label} dataset is empty: {path}")
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        metadata = record.get("metadata", {})
        messages = record.get("messages")
        if metadata.get("split") != "train":
            errors.append(f"{label}:{index} does not have metadata.split=train")
        if not isinstance(messages, list) or len(messages) < 3:
            errors.append(f"{label}:{index} does not have canonical messages")
            continue
        roles = [message.get("role") if isinstance(message, dict) else None for message in messages[:3]]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"{label}:{index} first message roles are not system/user/assistant")
        if any(str(message.get("content", "")).strip() == "" for message in messages[:3] if isinstance(message, dict)):
            errors.append(f"{label}:{index} has blank message content")
        if str(record.get("source_seed_id", "")).startswith("eval-"):
            errors.append(f"{label}:{index} appears to include an evaluation source id")
    if errors:
        raise ValueError("Training JSONL validation failed:\n- " + "\n- ".join(errors[:30]))
    return records


def format_messages(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Render Qwen's own chat template while supporting minor transformers API changes."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def build_tokenized_dataset(dataset: Any, tokenizer: Any, max_length: int) -> tuple[Any, int]:
    """Tokenize messages and mask system/user tokens so SFT learns assistant completions.

    Overlong rows are removed rather than silently truncating the assistant answer. The
    drop count is returned and must be inspected in the final training report.
    """
    def convert(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]
        prefix = format_messages(tokenizer, messages[:-1], add_generation_prompt=True)
        full = format_messages(tokenizer, messages, add_generation_prompt=False)
        full_ids = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
        prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
        # A malformed/custom template can make the prefix not align perfectly. Mask at
        # least the prefix length bounded by the full sequence; reviewer data remains
        # inspectable in the report rather than causing arbitrary model execution.
        prefix_length = min(len(prefix_ids), len(full_ids))
        labels = [-100] * prefix_length + full_ids[prefix_length:]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "_usable": len(full_ids) <= max_length and any(token != -100 for token in labels),
            "_sequence_length": len(full_ids),
        }

    tokenized = dataset.map(convert, remove_columns=dataset.column_names, desc="Applying Qwen chat template")
    before = len(tokenized)
    tokenized = tokenized.filter(lambda row: row["_usable"], desc=f"Dropping rows over {max_length} tokens")
    dropped = before - len(tokenized)
    tokenized = tokenized.remove_columns(["_usable", "_sequence_length"])
    if not len(tokenized):
        raise ValueError(f"No usable examples remain at max_seq_length={max_length}")
    return tokenized, dropped


def make_collator(tokenizer: Any) -> Any:
    """Pad input/attention through the tokenizer and labels with -100."""
    import torch

    class CompletionCollator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            labels = [list(feature["labels"]) for feature in features]
            inputs = [{"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]} for feature in features]
            batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
            target_length = batch["input_ids"].shape[1]
            padded: list[list[int]] = []
            for label in labels:
                missing = target_length - len(label)
                if tokenizer.padding_side == "left":
                    padded.append([-100] * missing + label)
                else:
                    padded.append(label + [-100] * missing)
            batch["labels"] = torch.tensor(padded, dtype=torch.long)
            return batch

    return CompletionCollator()


def execute_training(config: dict[str, Any], arguments: argparse.Namespace, report_path: Path) -> int:
    method = config.get("method")
    if method not in {"qlora_sft", "lora_sft"}:
        raise ValueError(f"Unsupported method {method!r}")
    if arguments.max_train_samples < 0 or arguments.max_eval_samples < 0:
        raise ValueError("sample limits must be zero or positive")
    assessment = assess_hardware(float(config.get("minimum_recommended_cuda_vram_gb", 16)))
    if not assessment["suitable_for_configured_qlora"] and not arguments.force_unsafe_hardware:
        write_json_atomic(
            report_path,
            {
                "stage": "training_preflight",
                "created_at": utc_now(),
                "status": "blocked_by_hardware",
                "config": config,
                "hardware": assessment,
                "message": "No training was run. Use a suitable CUDA/cloud system; do not mistake this for a completed fine-tuning job.",
            },
        )
        raise RuntimeError("Hardware guard blocked training. See report; use cloud/suitable CUDA hardware.")

    train_records = validate_training_records(config["training_file"], "train")
    eval_records = validate_training_records(config["validation_file"], "validation")
    if arguments.max_train_samples:
        train_records = train_records[: arguments.max_train_samples]
    if arguments.max_eval_samples:
        eval_records = eval_records[: arguments.max_eval_samples]
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are absent. On a suitable CUDA/cloud system install `pip install -r requirements/training.txt`."
        ) from exc

    if not torch.cuda.is_available() and not arguments.force_unsafe_hardware:
        raise RuntimeError("CUDA is unavailable; refusing CPU full-model/QLoRA training.")
    set_seed(int(config.get("seed", 3407)))
    use_bf16 = bool(config.get("bf16", True)) and bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config.get("base_model_revision"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "revision": config.get("base_model_revision"),
        "torch_dtype": compute_dtype,
        "device_map": {"": int(os.environ.get("LOCAL_RANK", "0"))},
    }
    if method == "qlora_sft":
        quant = config.get("quantization", {})
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=bool(quant.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quant.get("use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], **model_kwargs)
    resolved_base_revision = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False
    if method == "qlora_sft":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)))
    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora.get("lora_dropout", 0.05)),
        bias=str(lora.get("bias", "none")),
        task_type=str(lora.get("task_type", "CAUSAL_LM")),
        target_modules=list(lora["target_modules"]),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset, dropped_train = build_tokenized_dataset(Dataset.from_list(train_records), tokenizer, int(config["max_seq_length"]))
    eval_dataset, dropped_eval = build_tokenized_dataset(Dataset.from_list(eval_records), tokenizer, int(config["max_seq_length"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(config["num_train_epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_strategy": config.get("save_strategy", "steps"),
        "save_steps": int(config.get("save_steps", 100)),
        "eval_steps": int(config.get("eval_steps", 100)),
        "save_total_limit": int(config.get("save_total_limit", 2)),
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "tf32": bool(config.get("tf32", False)),
        "optim": config.get("optim", "paged_adamw_8bit"),
        "report_to": [],
        "remove_unused_columns": False,
        "seed": int(config.get("seed", 3407)),
    }
    # Transformers changed this keyword name. Supporting both avoids a needless code
    # edit when reproducing from a frozen requirements file.
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["eval_strategy"] = config.get("eval_strategy", "steps")
    else:  # pragma: no cover - for older training environments
        training_kwargs["evaluation_strategy"] = config.get("eval_strategy", "steps")
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collator(tokenizer),
    )
    train_result = trainer.train(resume_from_checkpoint=arguments.resume_from_checkpoint)
    metrics = dict(train_result.metrics)
    metrics.update({f"eval_{key}": value for key, value in trainer.evaluate().items()})
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    config_copy = output_dir / "training_config.json"
    config_copy.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "stage": "training",
        "created_at": utc_now(),
        "completed_at": utc_now(),
        "status": "completed",
        "method": method,
        "base_model": config["base_model"],
        "base_model_revision_requested": config.get("base_model_revision"),
        "base_model_revision_resolved": resolved_base_revision,
        "output_dir": str(output_dir),
        "hardware": assessment,
        "effective_precision": "bfloat16" if use_bf16 else "float16",
        "train_records_requested": len(train_records),
        "validation_records_requested": len(eval_records),
        "dropped_overlength_train_records": dropped_train,
        "dropped_overlength_validation_records": dropped_eval,
        "train_records_used": len(train_dataset),
        "validation_records_used": len(eval_dataset),
        "metrics": metrics,
        "warning": "A completed training run is not evidence of improvement. Run held-out candidate evaluation and compare it to the recorded baseline.",
    }
    write_json_atomic(report_path, report)
    print(f"Training completed; adapter saved to {output_dir}. Evaluate it before claiming improvement.")
    return 0


def run(arguments: argparse.Namespace) -> int:
    config = read_json(arguments.config)
    report_path = Path(arguments.report or Path(config.get("report_dir", "reports/training")) / f"training_{_timestamp()}.json")
    if not arguments.execute:
        assessment = assess_hardware(float(config.get("minimum_recommended_cuda_vram_gb", 16)))
        train_exists = Path(config["training_file"]).exists()
        validation_exists = Path(config["validation_file"]).exists()
        write_json_atomic(
            report_path,
            {
                "stage": "training_plan",
                "created_at": utc_now(),
                "status": "not_executed",
                "reason": "--execute was not supplied; no model training has happened.",
                "config": config,
                "training_file_exists": train_exists,
                "validation_file_exists": validation_exists,
                "hardware": assessment,
                "next_step": "Run on suitable CUDA/cloud hardware with --execute only after baseline and final dataset review.",
            },
        )
        print(f"Wrote training plan (no training executed): {report_path}")
        return 0
    return execute_training(config, arguments, report_path)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parser().parse_args(argv))
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
