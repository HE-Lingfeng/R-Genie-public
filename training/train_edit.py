# This training file is to fintune the show-o model in order to empower the mllm model with editing capability

import os
import sys
# script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/opt/data/private/helingfeng/RIE/Show-o-main")
# sys.path.append('/root/miniconda3/envs/RIE/lib/python3.10/site-packages/')
# print(sys.executable)
# print("Current Working Directory:", os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import Showo, MAGVITv2, get_mask_chedule, RGenieModel
from models.sampling import cosine_schedule, mask_by_random_topk
from training.edit_dataset import EditingDataset, AverageMeter, ProgressMeter
from training.utils import get_config
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
# from llava.llava import conversation as conversation_lib
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import deepspeed
import time

# process images into discrete tokens
def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def gumbel_softmax(logits, tau=1.0):
    gumbels = -torch.empty_like(logits).exponential_().log()  # Gumbel noise
    gumbel_logits = (logits + gumbels) / tau
    return torch.softmax(gumbel_logits, dim=-1)

def main():
    config = get_config()
    log_dir = os.path.join(config.training.log_base_dir, config.experiment.name)
    if config.training.local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"config.model.showo.llm_model_path is {config.model.showo.llm_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    
    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # If you train vq model, please uncomment the following lines
    # vq_model.train()
    # vq_model.requires_grad_(True)

    # model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model = RGenieModel.from_pretrained(config.model.RGenie.pretrained_model_path).to(device)
    # model = RGenieModel(config).to(device)

    # model.eval()
    # print(model)

    lora_r = config.training.lora_r

    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            # print(f"In function: lora_target_modules are {lora_target_modules}")
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                # print(f"name={name}, module={module}")
                if (
                    isinstance(module, cls) and
                    any(x in name for x in lora_target_modules)  # Include only specific layers
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))


        lora_alpha = config.training.lora_alpha
        lora_dropout = config.training.lora_dropout
        lora_target_modules = find_linear_layers(
            model, config.training.lora_target_modules.split(",")
        )
        # print(f"lora_target_modules are {lora_target_modules}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            # task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        for name, param in model.named_parameters():
            # print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
            if 'lora' not in name:
                param.requires_grad = False
        model.print_trainable_parameters()

        # Prepare training dataset
        train_dataset = EditingDataset(data_path=config.dataset.data_path, tokenizer=tokenizer)

        steps_per_epoch = len(train_dataset) // config.training.batch_size
        config.training.steps_per_epoch = steps_per_epoch
        ds_config = {
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
                    "warmup_num_steps": 100,
                    "warmup_type": "linear",
                },
            },
            "fp16": {
                "enabled": config.training.precision == "fp16",
            },
            "bf16": {
                "enabled": config.training.precision == "bf16",
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
        }

        # train_dataloader = DataLoader(train_dataloader, batch_size=config.training.batch_size, shuffle=True, num_workers=2, collate_fn=train_dataset.collate_fn)
        # optimizer = optim.AdamW(
        #     model.parameters(), 
        #     lr=0.001,        # Learning rate
        #     weight_decay=0.01 # Decoupled weight decay for L2 regularization
        # )
        world_size = torch.cuda.device_count()
        config.training.distributed = world_size > 1
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=train_dataset.collate_fn,
            config=ds_config,
        )
        train_iter = iter(train_loader)
        # print(f"train_iter is {train_iter}")


        for epoch in range(config.training.epochs):
            # train for one epoch
            train_iter = train(
                train_loader,
                model_engine,
                uni_prompting,
                vq_model,
                device,
                epoch,
                scheduler,
                writer,
                train_iter,
                config,
            )

            save_dir = os.path.join(log_dir, "ckpt_model")
            torch.distributed.barrier()
            if epoch % 5 == 0:
                model_engine.save_checkpoint(save_dir)


def train(train_loader,
          model,
          uni_prompting,
          vq_model,
          device,
          epoch,
          scheduler,
          writer,
          train_iter,
          config):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")

    progress = ProgressMeter(
        config.training.steps_per_epoch,
        [
            batch_time,
            losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in tqdm(range(config.training.steps_per_epoch), desc="Training Epoch"):
        for i in range(config.training.gradient_accumulation_steps):
            try:
                img_names, img_ids, instructions, images, targets = next(train_iter)
            except:
                train_iter = iter(train_loader)
                img_names, img_ids, instructions, images, targets = next(train_iter)

            data_time.update(time.time() - end)
            images = images.to(device)
            targets = targets.to(device)
            # print(f"images type is {type(images)}, device is {device}")
            # instructions.to(device)
            image_tokens = vq_model.get_code(images) + len(uni_prompting.text_tokenizer)
            targets_tokens = vq_model.get_code(targets)
            targets_tokens = targets_tokens + len(uni_prompting.text_tokenizer) # torch.Size([B, 1024])
            
            input_ids, _, labels = uni_prompting((instructions, image_tokens, targets_tokens), 't2i')
            # print(f"label size is {labels.size()}")

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(instructions), image_tokens), 't2i')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                uncond_input_ids = None
            
            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
            
            logits, loss_t2i = model(input_ids, attention_mask=attention_mask, labels=labels, batch_size_t2i=config.training.batch_size, image_tokens=image_tokens)
            
            # print(f"logits size: {logits.size()}")
            # print(f"loss_t2i: {loss_t2i}")
            
            losses.update(loss_t2i.item(), images.size(0))
            model.backward(loss_t2i)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % 5 == 0:
            if config.training.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()

            if config.training.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )
            
            batch_time.reset()
            data_time.reset()
            losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if config.training.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)
    print(loss_t2i)

    return train_iter

        

if __name__ == "__main__":
    # print("Current Working Directory:", os.getcwd())
    main()
