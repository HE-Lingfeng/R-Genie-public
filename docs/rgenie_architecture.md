# R-Genie Code Architecture

This repo keeps the editing path aligned with the paper's three-stage logic:

1. Multimodal tokenization: `training/rgenie_pipeline.py` builds the tokenizer, prompting helper, VQ tokenizer, and masked target tokens.
2. Reasoning-to-visual modulation: `models/rgenie_components.py` implements HRM and RAB on hidden embeddings, not on vocabulary logits.
3. Discrete diffusion reconstruction: `training/train_edit.py` trains masked target-token prediction, conditioned on source-image tokens and the instruction.

## Main Files

- `training/train_edit.py`: thin training entrypoint. It owns orchestration only: config, loading, DeepSpeed, epoch loop, logging, checkpointing.
- `training/rgenie_pipeline.py`: reusable training pipeline utilities: config validation, model loading, LoRA policy, DeepSpeed config, batch preparation.
- `training/edit_dataset.py`: editing dataset adapter for `imgs/`, `gt/`, and `editing_instruction_dict.json`.
- `models/RGenie.py`: model wrapper around Show-o/Phi. It extracts hidden states, applies reasoning modulation, projects to logits, and computes reconstruction loss.
- `models/rgenie_components.py`: paper-specific HRM/RAB module.

## Data Contract

The editing dataset expects:

```text
data/
  imgs/
  gt/
  editing_instruction_dict.json
```

Image filenames in `imgs/` and `gt/` must match. The JSON keys may use the raw image id, the id with leading zeros removed, or the full image stem.

## Training Contract

The training step follows masked latent modeling:

```text
source image -> source visual tokens -> conditioning tokens for RAB/HRM
target image -> target visual tokens -> mask schedule -> masked target input + labels
instruction + masked target -> Show-o sequence -> R-Genie logits -> CE on masked target tokens
```

LoRA is applied to the configured LLM projection layers. R-Genie's reasoning modulation layers remain trainable through the `reasoning_modulator` name prefix.

