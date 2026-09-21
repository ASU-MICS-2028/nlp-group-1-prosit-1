"""
Script to execute domain adaptation of distilgpt2 on the real Agricultural Q&A dataset using LoRA.
Logs empirical baseline vs post-adaptation metrics, prompt completions, and saves plots.
"""

import os
import math
import json
import time
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "domain_english"
MODELS_DIR = REPO_ROOT / "models" / "domain_adapted_checkpoint"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPO_ROOT / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_split(path: Path, max_samples: int = None):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    examples = [e.strip() for e in text.split("\n\n") if e.strip()]
    if max_samples:
        examples = examples[:max_samples]
    return examples


def evaluate_ppl(model, tokenizer, texts, batch_size=8, max_length=96):
    model.eval()
    device = "cpu"
    model.to(device)
    
    total_loss = 0.0
    total_batches = 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1
            
    avg_loss = total_loss / max(total_batches, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 40):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run_experiment():
    print("=== Domain-Specific LLM Adaptation Experiment (Agro-Extension) ===")
    
    # 1. Load Data Splits
    train_texts = load_split(PROCESSED_DIR / "train.txt", max_samples=500)
    val_texts = load_split(PROCESSED_DIR / "val.txt", max_samples=100)
    test_texts = load_split(PROCESSED_DIR / "test.txt", max_samples=100)
    print(f"Data Loaded: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # 2. Load Base Model & Tokenizer
    base_model_name = "distilgpt2"
    print(f"\nLoading Pre-trained Base Model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # 3. Evaluate Zero-Shot Baseline
    print("\nEvaluating Zero-Shot Base Model on Test Split...")
    base_loss, base_ppl = evaluate_ppl(base_model, tokenizer, test_texts)
    print(f"  -> Base Model Test Loss: {base_loss:.4f}")
    print(f"  -> Base Model Test Perplexity: {base_ppl:.2f}")
    
    prompts = [
        "Question: why is crop rotation important in farming?\nAnswer:",
        "Question: What farming practice helps prevent soil erosion?\nAnswer:",
    ]
    
    base_completions = {}
    print("\nSampling Zero-Shot Completions from Base Model:")
    for p in prompts:
        comp = generate_completion(base_model, tokenizer, p)
        base_completions[p] = comp
        print(f"\nPrompt:\n{p}\nBase Completion:\n{comp}")

    # 4. Attach LoRA Adapter
    print("\nConfiguring LoRA Adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        fan_in_fan_out=True,
    )
    adapted_model = get_peft_model(base_model, lora_config)
    print("Trainable Parameters:")
    adapted_model.print_trainable_parameters()

    # 5. Prepare Tokenized Datasets
    def tokenize_func(batch):
        return tokenizer(batch["text"], truncation=True, max_length=96, padding="max_length")

    train_ds = Dataset.from_dict({"text": train_texts}).map(tokenize_func, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_dict({"text": val_texts}).map(tokenize_func, batched=True, remove_columns=["text"])

    # 6. Fine-Tuning Execution
    print("\nStarting LoRA Fine-Tuning (3 Epochs)...")
    training_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "models" / "temp_train"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=adapted_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    start_time = time.time()
    train_result = trainer.train()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Final Training Loss: {train_result.training_loss:.4f}")

    # Save trained LoRA adapter weights
    adapted_model.save_pretrained(str(MODELS_DIR))
    tokenizer.save_pretrained(str(MODELS_DIR))
    print(f"Saved LoRA adapter checkpoint to: {MODELS_DIR}")

    # 7. Post-Adaptation Evaluation
    print("\nEvaluating LoRA-Adapted Model on Test Split...")
    adapt_loss, adapt_ppl = evaluate_ppl(adapted_model, tokenizer, test_texts)
    ppl_reduction = (base_ppl - adapt_ppl) / base_ppl * 100
    print(f"  -> Adapted Model Test Loss: {adapt_loss:.4f}")
    print(f"  -> Adapted Model Test Perplexity: {adapt_ppl:.2f}")
    print(f"  -> Relative Perplexity Improvement: {ppl_reduction:.2f}%")

    adapted_completions = {}
    print("\nSampling Completions from Adapted Model:")
    for p in prompts:
        comp = generate_completion(adapted_model, tokenizer, p)
        adapted_completions[p] = comp
        print(f"\nPrompt:\n{p}\nAdapted Completion:\n{comp}")

    # 8. Save Metrics Artifact
    results = {
        "domain": "Tropical Agriculture (Agro-Extension / KisanVaani QA)",
        "base_model": base_model_name,
        "lora_parameters": {
            "rank": 8,
            "alpha": 32,
            "dropout": 0.05,
            "target_layer": "Conv1D (attention projections)",
            "trainable_params": 147456,
            "total_params": 82060032,
            "trainable_pct": 0.1797
        },
        "dataset_split_sizes": {
            "train": len(train_texts),
            "val": len(val_texts),
            "test": len(test_texts)
        },
        "training_metrics": {
            "epochs": 3,
            "total_runtime_seconds": round(total_time, 2),
            "final_train_loss": round(train_result.training_loss, 4),
        },
        "evaluation_metrics": {
            "base_zero_shot_loss": round(base_loss, 4),
            "base_zero_shot_perplexity": round(base_ppl, 2),
            "lora_adapted_loss": round(adapt_loss, 4),
            "lora_adapted_perplexity": round(adapt_ppl, 2),
            "relative_perplexity_reduction_pct": round(ppl_reduction, 2)
        },
        "qualitative_completions": [
            {
                "prompt": p,
                "base_completion": base_completions[p],
                "adapted_completion": adapted_completions[p]
            }
            for p in prompts
        ]
    }

    results_file = REPORTS_DIR / "domain_adaptation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved empirical results to: {results_file}")

    # 9. Plot Comparison Figure
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    models = ["Base Model (Zero-Shot)", "LoRA Adapted ($r=8, \\alpha=32$)"]
    perplexities = [base_ppl, adapt_ppl]
    colors = ["#4A90E2", "#50E3C2"]
    
    bars = plt.bar(models, perplexities, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    plt.ylabel("Perplexity (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Domain Adaptation Perplexity: Agro-Extension Test Benchmark", fontsize=14, fontweight="bold", pad=15)
    
    for bar, ppl in zip(bars, perplexities):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{ppl:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    plt.ylim(0, max(perplexities) * 1.25)
    plot_path = FIGURES_DIR / "domain_adaptation_perplexity.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved benchmark figure to: {plot_path}")
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    run_experiment()
