# DukeOTR model identity and provenance

## Public identity

The finished assistant is named **DukeOTR**. Its user-facing identity must not be presented as
Qwen after the DukeOTR adapter/export has actually been trained, imported, evaluated, and
approved.

| Purpose | Planned value | Current status |
|---|---|---|
| Display/model name | `DukeOTR` | Reserved; no trained model exists |
| Versioned candidate Ollama tag | `dukeotr-v1` | Planned; not created |
| Stable user-facing Ollama tag | `dukeotr` | Planned release alias; not created |
| Dataset version | `dukeotr_dataset_v1` | Planned; no final dataset exists |
| Adapter version | `dukeotr_v1` | Planned; not trained |

The intended released interaction is:

```text
ollama run dukeotr
```

That command must not be documented as available until a real DukeOTR artifact has passed its
release gate and the `dukeotr` alias has actually been created.

## Technical provenance

DukeOTR remains technically derived from these base artifacts:

- local starting inference tag: `qwen3:4b`;
- training base: `Qwen/Qwen3-4B`; and
- any exact base revision/quantization/conversion recorded in a future training and export
  manifest.

Those Qwen references are required in technical metadata, license/notice material, training
configuration, compatibility checks, and reproducibility records. They are **provenance**, not
the public identity of the finished assistant.

## Naming policy

1. Use **DukeOTR** in user-facing system prompts, documentation, exports, reports, training
   run names, adapter names, and dataset names. Preserve Qwen names in the explicitly
   technical base-model baseline, compatibility, licensing, and reproducibility records.
2. Use `dukeotr-vN` for a measured version candidate and reserve `dukeotr` for the approved
   stable user-facing release alias.
3. Keep Qwen names only where a command must operate on the actual untouched base model or
   where base architecture/revision/license compatibility must be recorded.
4. Never create `dukeotr`, `dukeotr-v1`, or claim a DukeOTR improvement before training,
   compatible export, held-out evaluation, and explicit release approval evidence exist.
5. A user-facing name does not erase the base-model license, attribution, or technical
   provenance requirements.

## Release gate for the `dukeotr` alias

Create the stable `dukeotr` tag only after all of the following are recorded:

1. a quality-gated DukeOTR dataset manifest;
2. a completed compatible adapter training/export record;
3. held-out comparison against the preserved `qwen3:4b` baseline;
4. no unresolved critical security regression; and
5. a human release decision.

Until then, a versioned candidate such as `dukeotr-v1` is only a planned or evaluated
candidate—not a claim that DukeOTR is already finished.
