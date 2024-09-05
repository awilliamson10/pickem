import argparse
import itertools
import logging
import os
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm.auto import tqdm
# from pbp.model import LlamaForCausalLM
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pbp.dataloader import collate_fn, LengthBasedBatchSampler, tokenize_collate
from pbp.config import PBPConfig, parse_yaml_to_config
from pbp.scheduler import cosine_lr, constant_lr
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate(model, dataloader, config, tokenizer):
    out = {}
    model.eval()
    losses = torch.zeros(config.eval_steps)
    for k in range(config.eval_steps):
        batch = next(iter(dataloader))
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"], team_embeddings=batch["teams"])
        losses[k] = outputs.loss.item()
    out["eval_loss"] = losses.mean()
    model.train()
    return out


def init_model(config: PBPConfig):
    model = LlamaForCausalLM(config=config).to(config.device)
    if config.compile:
        model.compile()
    return model


def main(config: PBPConfig):
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.wandb else None,
        mixed_precision=config.precision if config.precision != "fp32" else "no",
        rng_types=["torch"]
    )

    if accelerator.is_main_process:
        accelerator.print()
        if config.output_dir is not None:
            accelerator.print(f"Output directory: {config.output_dir}")
            os.makedirs(config.output_dir, exist_ok=True)

        if config.wandb:
            accelerator.init_trackers(
                project_name=config.wandb_project if config.wandb_project else None,
            )

    if config.seed is not None:
        seed = config.seed
        accelerator.print(f"Using seed {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = init_model(config)

    train_dataset = load_dataset(path=config.train_dataset, split="train")
    eval_dataset = load_dataset(path=config.eval_dataset, split="test")

    if config.length_sample:
        accelerator.print("Using length-based sampling.")
        train_sampler = LengthBasedBatchSampler(train_dataset, batch_size=config.batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=tokenize_collate,
            num_workers=4,
            pin_memory=True
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            collate_fn=tokenize_collate,
            num_workers=4,
            shuffle=False,
            pin_memory=True
        )
    else:
        accelerator.print("Using standard sampling.")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=tokenize_collate,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            collate_fn=tokenize_collate,
            num_workers=4,
            shuffle=False,
            pin_memory=True
        )
    assert len(train_dataloader), "No data found, please check your data location."

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if isinstance(config.learning_rate, str):
        config.learning_rate = float(config.learning_rate)

    params_to_optimize = [
        {
            "params": itertools.chain(model.parameters()),
            "lr": config.learning_rate,
        },
    ]

    optimizer = optimizer_class(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )

    # create scheduler if train
    total_steps = len(train_dataloader) * config.epochs
    # if args.warmup is float, it is a percentage of total_steps
    if isinstance(config.warmup, float):
        assert (
            0 <= config.warmup <= 1
        ), "Warmup must be between 0 and 1 if not a fixed number of steps."
        config.warmup = int(config.warmup * total_steps)

    scheduler = cosine_lr(optimizer, config.learning_rate, config.warmup, total_steps)
    # scheduler = constant_lr(optimizer, config.learning_rate, config.warmup, total_steps)

    model, optimizer, scheduler, train_dataloader, eval_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, eval_dataloader
        )
    )

    print("***** Running training *****")
    print(f"  Num Iters = {len(train_dataloader)}")
    print(f"  Num Epochs = {config.epochs}")
    print(f"  Instantaneous batch size per device = {config.batch_size}")
    print(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    # print number of parameters in model
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Trainable Parameters: {total_trainable_params:,}")
    print(f"  Total Parameters: {total_params:,}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(config.epochs * len(train_dataloader)),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0
    best_val_loss = float("inf")

    tokenizer = PreTrainedTokenizerFast.from_pretrained("pickem_tokenizer")

    for epoch in range(config.epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if accelerator.is_local_main_process:
                if global_step % config.eval_interval == 0:
                    if accelerator.is_local_main_process:
                        eval_results = evaluate(model, eval_dataloader, config, tokenizer)
                        accelerator.log(
                            {
                                "eval_loss": eval_results["eval_loss"]
                            },
                            step=global_step
                        )
                        progress_bar.write(
                            f"Step: {global_step}, Eval loss: {eval_results['eval_loss']}"
                        )
                        if eval_results["eval_loss"] < best_val_loss:
                            best_val_loss = eval_results["eval_loss"]
                            save_path = os.path.join(
                                config.output_dir, "best_model"
                            )
                            model.save_pretrained(save_path)

                if global_step % config.save_interval == 0:
                    save_path = os.path.join(
                        config.output_dir, f"checkpoint-{global_step}"
                    )
                    model.save_pretrained(save_path)

            input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            teams = batch["teams"]
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, team_embeddings=teams).loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, 1.0)
            optimizer.step()
            scheduler(global_step)
            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step": global_step,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        save_path = os.path.join(config.output_dir)
        model.save_pretrained(save_path)

    accelerator.print("\n\nTraining completed.\n\n")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="The path to the yaml file containing the training configuration.",
    )
    config = parse_yaml_to_config(parser.parse_args().config)
    main(config)