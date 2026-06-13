import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from models import MAGVITv2, RGenieModel, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import mask_or_random_replace_tokens


CHECKPOINT_WEIGHT_NAMES = (
    "pytorch_model.bin",
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
)

RGENIE_TRAINABLE_KEYWORDS = (
    "lora",
    "reasoning_modulator",
)


def validate_editing_config(config):
    required_sections = ("model", "dataset", "training", "experiment")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    if config.model.vq_model.type != "magvitv2":
        raise ValueError(f"Unsupported VQ model type: {config.model.vq_model.type}")

    if config.training.precision not in ("fp16", "bf16", "fp32"):
        raise ValueError("training.precision must be one of: fp16, bf16, fp32")

    if not 0.0 <= float(config.training.min_masking_rate) <= 1.0:
        raise ValueError("training.min_masking_rate must be in [0, 1].")

    if config.training.noise_type not in ("mask", "random_replace"):
        raise ValueError("training.noise_type must be either 'mask' or 'random_replace'.")

    data_path = config.dataset.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    required_data_entries = ("imgs", "gt", "editing_instruction_dict.json")
    missing_entries = [entry for entry in required_data_entries if not os.path.exists(os.path.join(data_path, entry))]
    if missing_entries:
        raise FileNotFoundError(
            f"Dataset path {data_path} is missing: {', '.join(missing_entries)}. "
            "Expected imgs/, gt/, and editing_instruction_dict.json."
        )


def has_checkpoint_weights(path):
    return os.path.isdir(path) and any(os.path.exists(os.path.join(path, name)) for name in CHECKPOINT_WEIGHT_NAMES)


def build_tokenizer_and_prompting(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
    )
    return tokenizer, prompting


def load_vq_model(config, device):
    if config.model.vq_model.type != "magvitv2":
        raise ValueError(f"Unsupported VQ model type: {config.model.vq_model.type}")

    if config.model.vq_model.get("pretrained_model_path", None):
        model = MAGVITv2().to(device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path, map_location="cpu")["model"]
        model.load_state_dict(state_dict)
    else:
        model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)

    model.eval()
    model.requires_grad_(False)
    return model


def load_rgenie_model(config, device):
    rgenie_path = config.model.RGenie.pretrained_model_path
    if has_checkpoint_weights(rgenie_path):
        model = RGenieModel.from_pretrained(rgenie_path, low_cpu_mem_usage=False)
    else:
        model = RGenieModel(config)
    return model.to(device)


def find_lora_target_layers(model, target_modules):
    targets = [target.strip() for target in target_modules.split(",") if target.strip()]
    return sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any(target in name for target in targets)
    )


def apply_rgenie_lora(model, config):
    if config.training.lora_r <= 0:
        return model

    target_modules = find_lora_target_layers(model, config.training.lora_target_modules)
    if not target_modules:
        raise ValueError(
            "No LoRA target modules matched. Check training.lora_target_modules "
            f"({config.training.lora_target_modules})."
        )

    lora_config = LoraConfig(
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.training.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        param.requires_grad = any(keyword in name for keyword in RGENIE_TRAINABLE_KEYWORDS)
    if not any(param.requires_grad for param in model.parameters()):
        raise RuntimeError("No trainable parameters remain after applying LoRA/freeze policy.")
    model.print_trainable_parameters()
    return model


def trainable_parameters(model):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return params


def build_deepspeed_config(config, steps_per_epoch):
    precision = config.training.precision
    return {
        "train_micro_batch_size_per_gpu": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.training.lr,
                "weight_decay": 0.0,
                "betas": (config.training.beta1, config.training.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": config.training.epochs * steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": config.training.lr,
                "warmup_num_steps": config.training.get("warmup_num_steps", 100),
                "warmup_type": "linear",
            },
        },
        "fp16": {"enabled": precision == "fp16"},
        "bf16": {"enabled": precision == "bf16"},
        "gradient_clipping": config.training.get("gradient_clipping", 1.0),
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }


def build_mask_schedule(config):
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        return get_mask_chedule(schedule, **args)
    return get_mask_chedule(config.training.get("mask_schedule", "cosine"))


@torch.no_grad()
def prepare_editing_batch(batch, vq_model, uni_prompting, config, device, mask_schedule):
    _, _, instructions, images, targets = batch
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    text_vocab_size = len(uni_prompting.text_tokenizer)
    source_tokens = vq_model.get_code(images) + text_vocab_size
    target_tokens = vq_model.get_code(targets) + text_vocab_size
    mask_id = config.model.showo.vocab_size - 1

    masked_target_tokens, target_labels, _, _ = mask_or_random_replace_tokens(
        target_tokens,
        mask_id,
        config,
        mask_schedule=mask_schedule,
        is_train=True,
    )

    input_ids, _, labels = uni_prompting((instructions, masked_target_tokens, target_labels), "t2i")
    attention_mask = create_attention_mask_predict_next(
        input_ids,
        pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
        soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        rm_pad_in_image=True,
    )
    return input_ids, attention_mask, labels, source_tokens, images.size(0)
