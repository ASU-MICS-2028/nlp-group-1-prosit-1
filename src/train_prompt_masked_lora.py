"""
Ablation Experiment: Prompt Loss Masking vs Standard Causal Language Modeling.
Trains DistilGPT2 with LoRA using prompt-loss masking:
Only tokens in the 'Answer:' section receive cross-entropy loss gradients,
while prompt ('Question: ...') tokens are masked with label -100.
Saves empirical comparison to reports/prompt_masking_ablation_results.json.
"""

import json
import math
import time
from pathlib import Path
import torch
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
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_CHECKPOINT = REPO_ROOT / "models" / "prompt_masked_lora_checkpoint"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CHECKPOINT.mkdir(parents=True, exist_ok=True)


def load_split(path: Path, max_samples: int = None):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    examples = [e.strip() for e in text.split("\n\n") if e.strip()]
    if max_samples:
        examples = examples[:max_samples]
    return examples


def tokenize_with_prompt_masking(batch, tokenizer, max_length=96):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for text in batch["text"]:
        # Find the boundary where the answer begins
        parts = text.split("\nAnswer:")
        if len(parts) == 2:
            prompt_str = parts[0] + "\nAnswer:"
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            prompt_len = len(prompt_ids)
        else:
            prompt_len = 0
            
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"][0].tolist()
        attention_mask = enc["attention_mask"][0].tolist()
        labels = list(input_ids)
        
        # Mask prompt tokens and pad tokens with -100 (ignored by PyTorch loss)
        for i in range(len(labels)):
            if i < prompt_len or labels[i] == tokenizer.pad_token_id:
                labels[i] = -100
                
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
        
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list
    }


def evaluate_answer_ppl(model, tokenizer, texts, batch_size=8, max_length=96):
    """Evaluate perplexity strictly on answer tokens (ignoring prompt tokens)."""
    model.eval()
    device = "cpu"
    model.to(device)
    total_loss, total_batches = 0.0, 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_dict = {"text": batch_texts}
        enc = tokenize_with_prompt_masking(batch_dict, tokenizer, max_length=max_length)
        input_ids = torch.tensor(enc["input_ids"]).to(device)
        attention_mask = torch.tensor(enc["attention_mask"]).to(device)
        labels = torch.tensor(enc["labels"]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1
            
    avg_loss = total_loss / max(total_batches, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def run_ablation():
    print("=== Ablation: Prompt Loss Masking vs Standard LoRA ===")
    
    train_texts = load_split(PROCESSED_DIR / "train.txt", max_samples=400)
    val_texts = load_split(PROCESSED_DIR / "val.txt", max_samples=100)
    test_texts = load_split(PROCESSED_DIR / "test.txt", max_samples=100)
    print(f"Dataset: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    base_model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Base Model Zero-Shot Answer PPL
    print("\nEvaluating Base Zero-Shot on Answer Tokens...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_ans_loss, base_ans_ppl = evaluate_answer_ppl(base_model, tokenizer, test_texts)
    print(f"Base Zero-Shot Answer Loss: {base_ans_loss:.4f} | Perplexity: {base_ans_ppl:.2f}")

    # 2. Attach LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        fan_in_fan_out=True,
    )
    model = get_peft_model(base_model, lora_config)

    # 3. Prepare Prompt-Masked Datasets
    print("\nTokenizing with Prompt-Loss Masking (labels[:prompt_len] = -100)...")
    train_ds = Dataset.from_dict({"text": train_texts}).map(
        lambda b: tokenize_with_prompt_masking(b, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    val_ds = Dataset.from_dict({"text": val_texts}).map(
        lambda b: tokenize_with_prompt_masking(b, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "models" / "temp_prompt_mask_train"),
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
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("\nTraining LoRA with Prompt-Loss Masking (3 Epochs)...")
    t0 = time.time()
    train_res = trainer.train()
    train_duration = time.time() - t0
    print(f"Training completed in {train_duration:.2f}s. Final Train Loss: {train_res.training_loss:.4f}")

    # 5. Evaluate Adapted Model on Answer Tokens
    print("\nEvaluating Prompt-Masked Model on Test Split (Answer Tokens)...")
    masked_ans_loss, masked_ans_ppl = evaluate_answer_ppl(model, tokenizer, test_texts)
    print(f"Prompt-Masked LoRA Answer Loss: {masked_ans_loss:.4f} | Perplexity: {masked_ans_ppl:.2f}")
    
    # Save checkpoint
    model.save_pretrained(str(OUTPUT_CHECKPOINT))

    # 6. Sample Generations with Conservative Agronomic Strategy
    sample_prompts = [
        "Question: why is crop rotation important in farming?\nAnswer:",
        "Question: What farming practice helps prevent soil erosion?\nAnswer:",
        "Question: How can farmers control fall armyworm in maize?\nAnswer:",
    ]
    
    completions = []
    model.eval()
    for p in sample_prompts:
        input_ids = tokenizer.encode(p, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=40,
                temperature=0.35,
                top_p=0.85,
                repetition_penalty=1.25,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        ans = tokenizer.decode(out[0], skip_special_tokens=True)[len(p):].strip()
        completions.append({"prompt": p.strip(), "answer": ans})

    # Save results artifact
    results = {
        "experiment": "Prompt Loss Masking vs Zero-Shot Base",
        "masking_rule": "labels[:prompt_len] = -100 (Gradients exclusively on Answer tokens)",
        "train_samples": len(train_texts),
        "epochs": 3,
        "training_time_seconds": round(train_duration, 2),
        "metrics": {
            "base_zero_shot_answer_loss": round(base_ans_loss, 4),
            "base_zero_shot_answer_ppl": round(base_ans_ppl, 2),
            "prompt_masked_answer_loss": round(masked_ans_loss, 4),
            "prompt_masked_answer_ppl": round(masked_ans_ppl, 2),
            "relative_perplexity_reduction_pct": round((base_ans_ppl - masked_ans_ppl) / base_ans_ppl * 100, 2)
        },
        "sample_completions": completions
    }

    out_json = REPORTS_DIR / "prompt_masking_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation completed successfully! Results written to: {out_json}")


if __name__ == "__main__":
    run_ablation()
