#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import os
from functools import partial

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Model cache paths replicated from scripts/test-pd-offline.py
MODEL_PATHS = {
    "llama-2-7b": "/huggingface-cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
    "qwen3-4b": "/huggingface-cache/hub/models--Qwen--Qwen3-4B/snapshots/82d62bb073771e7a1ea59435f548908540217d1f",
    "qwen3-8b": "/huggingface-cache/hub/models--Qwen--Qwen3-8B/snapshots/2069b3fae1114555f3c020c81410e51fa0f656f2",
    "gpt2-l": "/huggingface-cache/hub/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 2 7B with Hugging Face Transformers")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-l",
        choices=sorted(MODEL_PATHS.keys()),
        help="Model alias from test-pd-offline.py mapping.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model path on disk (otherwise resolved via --model).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store checkpoints and logs. Defaults to outputs/<model>.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="🤗 Datasets hub name (e.g., wikitext, json).")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration (if required).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Column name containing raw text. Defaults to the first string column.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional maximum number of training examples for quick experiments.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps to reach an effective batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Logging frequency (in steps).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Checkpoint save frequency (in steps).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 mixed precision.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision (ignored if bf16 is set).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def pick_text_column(dataset, user_choice: str | None) -> str:
    if user_choice:
        if user_choice not in dataset.column_names:
            raise ValueError(f"Column '{user_choice}' not in dataset columns: {dataset.column_names}")
        return user_choice

    # fall back to first column with string dtype
    for column in dataset.column_names:
        if isinstance(dataset[column][0], str):
            return column
    raise ValueError("No string/text column found; provide --text-column explicitly.")


def tokenize_text(examples, tokenizer, text_column: str, max_length: int):
    texts = examples[text_column]
    output = tokenizer(texts, truncation=True, max_length=max_length)
    return output


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model_path = args.model_path or MODEL_PATHS.get(args.model)
    if model_path is None:
        raise ValueError(f"Unknown model alias '{args.model}'.")

    output_dir = args.output_dir or os.path.join("outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading tokenizer for %s from %s", args.model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Loading dataset %s/%s (%s split)", args.dataset_name, args.dataset_config, args.dataset_split
    )
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    if args.max_train_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))

    text_column = pick_text_column(dataset, args.text_column)

    tokenize_fn = partial(tokenize_text, tokenizer=tokenizer, text_column=text_column, max_length=args.max_seq_length)
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    logger.info("Loading model weights from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    gpu_count = max(1, torch.cuda.device_count())
    total_steps_per_epoch = math.ceil(len(tokenized_dataset) / (args.per_device_train_batch_size * gpu_count))
    logger.info(
        "Starting training with %d samples (~%d steps per epoch).",
        len(tokenized_dataset),
        total_steps_per_epoch,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=not args.bf16 and args.fp16,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
