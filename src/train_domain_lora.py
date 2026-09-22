"""
Section C experiment: adapt distilgpt2 to agricultural Q&A with LoRA, trained two ways on identical data.

    standard   loss on every token of "Question: ...\\nAnswer: ..."
    masked     loss on the answer tokens only (question tokens get label -100)

The base model and both adapters are scored on the same held-out test questions (none of them occur in
training, see src/prepare_domain_data.py) with three token-weighted perplexities:

    full_ppl       every token of the test Q&A text
    answer_ppl     only the answer tokens, given the question
    wikitext_ppl   200 WikiText-2 test paragraphs: general English, i.e. what the adaptation costs

Run from the repo root after src/prepare_domain_data.py (about 15 minutes on a laptop CPU):
    python src/train_domain_lora.py
Writes reports/domain_adaptation_results.json and figures/domain_adaptation_perplexity.png.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.prepare_domain_data import load_split, normalize_question  # noqa: E402

RANDOM_SEED = 42
BASE_MODEL = "distilgpt2"
MAX_LENGTH = 96
TRAIN_PAIRS = 500  # ponytail: CPU budget (~5 min per adapter); a GPU can take all 1,769 training pairs
MODEL_DIRS = {
    "standard": REPO_ROOT / "models" / "domain_adapted_checkpoint",
    "masked": REPO_ROOT / "models" / "prompt_masked_lora_checkpoint",
}
PROMPTS = [
    "Question: why is crop rotation important in farming?\nAnswer:",
    "Question: What farming practice helps prevent soil erosion?\nAnswer:",
    "Question: How can farmers control fall armyworm in maize?\nAnswer:",
]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


def lora_config() -> LoraConfig:
    # c_attn is GPT-2's fused query/key/value projection, stored as a Conv1D (hence fan_in_fan_out)
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05, fan_in_fan_out=True)


def prompt_len(text: str) -> int:
    """Tokens up to and including 'Answer:', the part the masked model gets no loss on."""
    return len(tokenizer(text[: text.index("\nAnswer:") + len("\nAnswer:")]).input_ids)


def perplexity(model, texts, answer_only=False, max_length=MAX_LENGTH) -> float:
    """Token-weighted: total negative log-likelihood of all scored tokens / number of those tokens."""
    model.eval()
    nll = count = 0
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").input_ids
        labels = ids.clone()
        if answer_only:
            labels[:, : prompt_len(text)] = -100
        n = int((labels[:, 1:] != -100).sum())  # the model predicts token t+1 from tokens up to t
        if n == 0:
            continue
        with torch.no_grad():
            nll += model(input_ids=ids, labels=labels).loss.item() * n
        count += n
    return round(math.exp(nll / count), 2)


def encode(texts, mask_prompt: bool) -> Dataset:
    enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    labels = []
    for text, ids, attention in zip(texts, enc["input_ids"], enc["attention_mask"]):
        row = [t if a else -100 for t, a in zip(ids, attention)]  # padding is never a target
        if mask_prompt:
            k = prompt_len(text)
            row[:k] = [-100] * len(row[:k])
        labels.append(row)
    return Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels})


def sample(model, prompt: str) -> str:
    set_seed(RANDOM_SEED)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def evaluate(model, test_texts, wiki) -> dict:
    return {
        "full_ppl": perplexity(model, test_texts),
        "answer_ppl": perplexity(model, test_texts, answer_only=True),
        "wikitext_ppl": perplexity(model, wiki, max_length=256),
        "samples": [sample(model, p) for p in PROMPTS],
    }


def train(mask_prompt: bool, train_texts, val_texts):
    set_seed(RANDOM_SEED)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(BASE_MODEL), lora_config())
    args = TrainingArguments(
        output_dir=str(REPO_ROOT / "models" / "temp_train"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        seed=RANDOM_SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encode(train_texts, mask_prompt),
        eval_dataset=encode(val_texts, mask_prompt),
        data_collator=default_data_collator,
    )
    t0 = time.time()
    result = trainer.train()
    return model, {
        "train_seconds": round(time.time() - t0, 1),
        "mean_train_loss": round(result.training_loss, 4),  # averaged over all steps, early ones included
        "val_loss_per_epoch": [round(h["eval_loss"], 4) for h in trainer.state.log_history if "eval_loss" in h],
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("full_ppl", "Full Q&A text"), ("answer_ppl", "Answer tokens only"), ("wikitext_ppl", "General English\n(WikiText-2)")]
    models = [("base", "Base distilgpt2"), ("standard", "LoRA, loss on all tokens"), ("masked", "LoRA, loss on answer only")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3, validated light mode
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    width = 0.26
    for i, ((key, label), color) in enumerate(zip(models, colors)):
        xs = [m + (i - 1) * (width + 0.02) for m in range(len(metrics))]
        bars = ax.bar(xs, [results[key][m] for m, _ in metrics], width, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color="#52514e", padding=2)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics], color="#0b0b0b")
    ax.set_ylabel("Perplexity (lower is better)", color="#52514e")
    ax.set_title("distilgpt2 on 222 held-out agricultural questions, before and after LoRA", color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", length=0)
    ax.legend(frameon=False, labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "domain_adaptation_perplexity.png", dpi=200)


def main():
    train_texts, val_texts, test_texts = load_split("train", TRAIN_PAIRS), load_split("val"), load_split("test")
    wiki = [
        t.strip()
        for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
        if len(t.split()) >= 40 and not t.strip().startswith("=")
    ][:200]
    question = lambda t: normalize_question(t.split("\nAnswer:")[0].removeprefix("Question: "))
    seen = {question(t) for t in load_split("train") + val_texts}

    results = {"base": evaluate(AutoModelForCausalLM.from_pretrained(BASE_MODEL), test_texts, wiki)}
    for name, mask_prompt in (("standard", False), ("masked", True)):
        model, info = train(mask_prompt, train_texts, val_texts)
        model.save_pretrained(str(MODEL_DIRS[name]))
        results[name] = {**info, **evaluate(model, test_texts, wiki)}
        print(name, {k: v for k, v in results[name].items() if k != "samples"}, flush=True)

    report = {
        "dataset": "KisanVaani/agriculture-qa-english-only, one row per distinct question (src/prepare_domain_data.py)",
        "base_model": BASE_MODEL,
        "lora": {"r": 8, "alpha": 32, "dropout": 0.05, "target_modules": ["c_attn"], "epochs": 3, "lr": 5e-4, "batch_size": 8},
        "split_sizes": {"train_used": len(train_texts), "val": len(val_texts), "test": len(test_texts), "wikitext_paragraphs": len(wiki)},
        "test_questions_seen_in_train_or_val": sum(question(t) in seen for t in test_texts),
        "max_length_tokens": MAX_LENGTH,
        "sampling": "seed 42 before each prompt; do_sample, temperature 0.7, top_p 0.9, 40 new tokens",
        "prompts": PROMPTS,
        "results": results,
    }
    with open(REPO_ROOT / "reports" / "domain_adaptation_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    plot(results)


if __name__ == "__main__":
    main()
