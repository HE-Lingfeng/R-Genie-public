import math
import os
import time

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.edit_dataset import AverageMeter, EditingDataset, ProgressMeter
from training.rgenie_pipeline import (
    apply_rgenie_lora,
    build_deepspeed_config,
    build_mask_schedule,
    build_tokenizer_and_prompting,
    load_rgenie_model,
    load_vq_model,
    prepare_editing_batch,
    trainable_parameters,
    validate_editing_config,
)
from training.utils import get_config


def main():
    config = get_config()
    validate_editing_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(config.training.log_base_dir, config.experiment.name)
    writer = build_writer(config, log_dir)

    tokenizer, uni_prompting = build_tokenizer_and_prompting(config)
    vq_model = load_vq_model(config, device)
    model = apply_rgenie_lora(load_rgenie_model(config, device), config)

    train_dataset = EditingDataset(
        data_path=config.dataset.data_path,
        tokenizer=tokenizer,
        resolution=config.dataset.preprocessing.resolution,
    )
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / config.training.batch_size))
    config.training.steps_per_epoch = steps_per_epoch

    model_engine, _, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=trainable_parameters(model),
        training_data=train_dataset,
        collate_fn=train_dataset.collate_fn,
        config=build_deepspeed_config(config, steps_per_epoch),
    )

    config.training.distributed = torch.cuda.device_count() > 1
    train_iter = iter(train_loader)
    mask_schedule = build_mask_schedule(config)

    for epoch in range(config.training.epochs):
        train_iter = train_one_epoch(
            train_loader=train_loader,
            model=model_engine,
            uni_prompting=uni_prompting,
            vq_model=vq_model,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            writer=writer,
            train_iter=train_iter,
            config=config,
            mask_schedule=mask_schedule,
        )
        save_checkpoint_if_needed(model_engine, log_dir, epoch)


def build_writer(config, log_dir):
    if config.training.local_rank != 0:
        return None
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def train_one_epoch(
    train_loader,
    model,
    uni_prompting,
    vq_model,
    device,
    epoch,
    scheduler,
    writer,
    train_iter,
    config,
    mask_schedule,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        config.training.steps_per_epoch,
        [batch_time, losses],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    end = time.time()
    last_loss = None

    for global_step in tqdm(range(config.training.steps_per_epoch), desc="Training Epoch"):
        for _ in range(config.training.gradient_accumulation_steps):
            batch, train_iter = next_batch(train_loader, train_iter)
            data_time.update(time.time() - end)

            input_ids, attention_mask, labels, source_tokens, batch_size = prepare_editing_batch(
                batch=batch,
                vq_model=vq_model,
                uni_prompting=uni_prompting,
                config=config,
                device=device,
                mask_schedule=mask_schedule,
            )

            _, loss = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                batch_size_t2i=batch_size,
                image_tokens=source_tokens,
                max_seq_length=config.dataset.preprocessing.max_seq_length,
            )

            losses.update(loss.item(), batch_size)
            model.backward(loss)
            model.step()
            last_loss = loss

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress(global_step, batch_time, data_time, losses, progress, scheduler, writer, config)

    if last_loss is not None and config.training.local_rank == 0:
        print(last_loss)
    return train_iter


def next_batch(train_loader, train_iter):
    try:
        return next(train_iter), train_iter
    except StopIteration:
        train_iter = iter(train_loader)
        return next(train_iter), train_iter


def log_progress(global_step, batch_time, data_time, losses, progress, scheduler, writer, config):
    if global_step % 5 == 0:
        if config.training.distributed:
            batch_time.all_reduce()
            data_time.all_reduce()
            losses.all_reduce()

        if writer is not None:
            progress.display(global_step + 1)
            writer.add_scalar("train/loss", losses.avg, global_step)
            writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
            writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

        batch_time.reset()
        data_time.reset()
        losses.reset()

    if global_step != 0 and writer is not None:
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)


def save_checkpoint_if_needed(model_engine, log_dir, epoch):
    save_dir = os.path.join(log_dir, "ckpt_model")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if epoch % 5 == 0:
        model_engine.save_checkpoint(save_dir)


if __name__ == "__main__":
    main()
