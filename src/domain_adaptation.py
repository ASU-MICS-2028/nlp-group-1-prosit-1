"""
Utilities for domain adaptation and fine-tuning of English language models.
Supports causal language modeling, Hugging Face transformers, and parameter-efficient fine-tuning (LoRA).
"""

from typing import Dict, Any, Optional
import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset


def load_model_and_tokenizer(model_name: str = "distilgpt2", use_lora: bool = True, lora_r: int = 8):
    """
    Loads base model and tokenizer, optionally applying LoRA for parameter-efficient adaptation.

    Args:
        model_name: Hugging Face model identifier (e.g. 'distilgpt2', 'gpt2', 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T').
        use_lora: Whether to wrap model with LoRA adapters.
        lora_r: Rank of LoRA update matrices.

    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
    )

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=32,
                lora_dropout=0.05,
            )
            model = get_peft_model(model, peft_config)
            print("LoRA adapter attached. Trainable parameters:")
            model.print_trainable_parameters()
        except ImportError:
            print("peft library not installed, proceeding with full fine-tuning.")

    return model, tokenizer


def prepare_dataset(texts, tokenizer, max_length: int = 256) -> Dataset:
    """
    Tokenizes raw text corpus into a Hugging Face Dataset ready for causal LM training.
    """
    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    return tokenized_dataset


def evaluate_perplexity(model, tokenizer, eval_dataset, batch_size: int = 4) -> float:
    """
    Evaluates causal language model perplexity on an evaluation dataset.

    Args:
        model: Hugging Face CausalLM.
        tokenizer: Pretrained tokenizer.
        eval_dataset: Tokenized dataset.
        batch_size: Batch size for evaluation.

    Returns:
        Perplexity score (float).
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    return perplexity
