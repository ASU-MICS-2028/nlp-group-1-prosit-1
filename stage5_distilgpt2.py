"""Section C, second experiment: distilgpt2 (82M parameters) adapted to agricultural Q&A with LoRA on a laptop CPU.

The main Section C experiment (stage5_domain.py) adapts Qwen2.5-0.5B with MLX on the GPU. This one asks three questions
that experiment does not:
    1. Does it matter WHICH tokens carry the loss?  "standard" = loss on every token of "Question: ...\nAnswer: ...";
       "masked" = loss on the answer tokens only (question labels set to -100, which the loss ignores).
    2. How much of the gain is the domain, and how much is the question template?  Three perplexities on the same
       222 held-out questions: full text, answer tokens only (given the question), and WikiText-2 (general English).
    3. Do decoding settings fix repetition, and do they fix correctness?  Distinct-3 (unique word trigrams / all word
       trigrams) under five decoding strategies.

The corpus repeats questions (22,615 rows, about 2,200 distinct questions), so one row per question is kept BEFORE
the split: no test question is also a training question.

    uv run python stage5_distilgpt2.py prepare     # splits -> data/domain/distilgpt2/ (seconds)
    uv run python stage5_distilgpt2.py train       # both adapters -> models/distilgpt2_agriculture/ (~15 min, CPU)
    uv run python stage5_distilgpt2.py decoding    # -> results/distilgpt2_decoding.json

Needs torch, transformers, peft, datasets and pandas at the versions in requirements-distilgpt2.txt (the recorded
results were produced with them; newer transformers renamed some Trainer arguments).
"""
import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
RAW_PARQUET = ROOT / "data" / "domain" / "raw" / "agri.parquet"   # KisanVaani/agriculture-qa-english-only (Apache-2.0)
PROCESSED_DIR = ROOT / "data" / "domain" / "distilgpt2"
BASE_MODEL = "distilgpt2"
MAX_LENGTH = 96
TRAIN_PAIRS = 500  # CPU budget (~5 min per adapter); a GPU can take all 1,769 training pairs
MODELS = ROOT / "models" / "distilgpt2_agriculture"
MODEL_DIRS = {"standard": MODELS / "standard", "masked": MODELS / "masked"}
RESULTS = ROOT / "results"
PROMPTS = [
    "Question: why is crop rotation important in farming?\nAnswer:",
    "Question: What farming practice helps prevent soil erosion?\nAnswer:",
    "Question: How can farmers control fall armyworm in maize?\nAnswer:",
]


# ---------------------------------------------------------------- data (no ML libraries needed)
def normalize_question(question):
    """Lowercase, punctuation to spaces, collapsed whitespace: the key used to find repeated questions."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", question.lower())).strip()


def prepare():
    import pyarrow.parquet as pq
    table = pq.read_table(RAW_PARQUET).to_pydict()
    pairs, seen = [], set()
    for question, answer in zip(table["question"], table["answers"]):
        question, answer = str(question).strip(), str(answer).strip()
        key = normalize_question(question)
        if question and answer and key not in seen:  # first occurrence of each question only
            seen.add(key)
            pairs.append({"question": question, "answer": answer})
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    n_train, n_val = int(0.8 * len(pairs)), int(0.1 * len(pairs))
    splits = {"train": pairs[:n_train], "val": pairs[n_train:n_train + n_val], "test": pairs[n_train + n_val:]}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(PROCESSED_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    stats = {"raw_rows": len(table["question"]), "unique_questions": len(pairs), **{f"{k}_pairs": len(v) for k, v in splits.items()}}
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(stats)
    return stats


def load_split(name, max_samples=None):
    """One split as training strings: 'Question: ...\\nAnswer: ...'."""
    with open(PROCESSED_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows[:max_samples]]


# ---------------------------------------------------------------- model (imports only when training or decoding)
def _ml():
    global torch, tokenizer, Dataset, load_dataset, LoraConfig, TaskType, get_peft_model, PeftModel
    global AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator, set_seed
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              default_data_collator, set_seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------- training and scoring
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
        output_dir=str(MODELS / "trainer_tmp"),  # scratch: save_strategy="no" keeps no checkpoints
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
    fig.savefig(RESULTS / "distilgpt2_perplexity.png", dpi=200)


def train_all():
    _ml()
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
        "dataset": "KisanVaani/agriculture-qa-english-only, one row per distinct question (stage5_distilgpt2.py prepare)",
        "base_model": BASE_MODEL,
        "lora": {"r": 8, "alpha": 32, "dropout": 0.05, "target_modules": ["c_attn"], "epochs": 3, "lr": 5e-4, "batch_size": 8},
        "split_sizes": {"train_used": len(train_texts), "val": len(val_texts), "test": len(test_texts), "wikitext_paragraphs": len(wiki)},
        "test_questions_seen_in_train_or_val": sum(question(t) in seen for t in test_texts),
        "max_length_tokens": MAX_LENGTH,
        "sampling": "seed 42 before each prompt; do_sample, temperature 0.7, top_p 0.9, 40 new tokens",
        "prompts": PROMPTS,
        "results": results,
    }
    with open(RESULTS / "distilgpt2_lora.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    plot(results)


# ---------------------------------------------------------------- decoding benchmark
def calculate_repetition_metrics(text: str):
    """Calculate distinct-1, distinct-2, and distinct-3 ratios to quantify repetition."""
    words = text.lower().split()
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0, "word_count": 0}
    
    unigrams = words
    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    
    d1 = len(set(unigrams)) / max(len(unigrams), 1)
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    d3 = len(set(trigrams)) / max(len(trigrams), 1)
    
    return {
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "distinct_3": round(d3, 4),
        "word_count": len(words)
    }


def decoding():
    _ml()
    print("=== Benchmarking Decoding Strategies for LoRA-Adapted LLM ===")
    
    print(f"Loading checkpoint from: {MODEL_DIRS["standard"]}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    adapted_model = PeftModel.from_pretrained(base_model, str(MODEL_DIRS["standard"]))
    adapted_model.eval()

    prompts = [
        "Question: why is crop rotation important in farming?\nAnswer:",
        "Question: What farming practice helps prevent soil erosion?\nAnswer:",
        "Question: How can farmers control fall armyworm in maize?\nAnswer:",
        "Question: What are the signs of nitrogen deficiency in crops?\nAnswer:",
    ]

    strategies = {
        "unpenalized_sampling": {
            "name": "Unpenalized Sampling (Baseline)",
            "params": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "do_sample": True}
        },
        "repetition_penalty_only": {
            "name": "Repetition Penalty (r=1.3)",
            "params": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.3, "no_repeat_ngram_size": 0, "do_sample": True}
        },
        "ngram_blocking": {
            "name": "Strict N-gram Blocking (N=3)",
            "params": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.0, "no_repeat_ngram_size": 3, "do_sample": True}
        },
        "conservative_agronomic": {
            "name": "Low temperature + penalty + 3-gram block",
            "params": {"temperature": 0.35, "top_p": 0.85, "repetition_penalty": 1.25, "no_repeat_ngram_size": 3, "do_sample": True}
        },
        "deterministic_greedy": {
            "name": "Deterministic Greedy Search",
            "params": {"do_sample": False, "repetition_penalty": 1.25, "no_repeat_ngram_size": 3}
        }
    }

    benchmark_data = {
        "model": "distilgpt2 + LoRA (r=8, alpha=32), models/distilgpt2_agriculture/standard",
        "prompts_evaluated": len(prompts),
        "samples_per_prompt": N_SAMPLES,
        "note": "Distinct-3 measures repetition, not correctness. no_repeat_ngram_size=3 makes it ~1 by construction.",
        "results": []
    }

    for prompt in prompts:
        prompt_entry = {
            "prompt": prompt.strip(),
            "evaluations": {}
        }
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        for strat_id, strat in strategies.items():
            samples = []
            for i in range(N_SAMPLES if strat["params"]["do_sample"] else 1):
                set_seed(RANDOM_SEED + i)
                with torch.no_grad():
                    output = adapted_model.generate(
                        input_ids,
                        max_new_tokens=45,
                        pad_token_id=tokenizer.eos_token_id,
                        **strat["params"]
                    )
                answer = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                samples.append({"generated_answer": answer, "metrics": calculate_repetition_metrics(answer)})

            prompt_entry["evaluations"][strat_id] = {
                "strategy_name": strat["name"],
                "samples": samples,
                "mean_distinct_3": round(sum(x["metrics"]["distinct_3"] for x in samples) / len(samples), 4),
            }
            
        benchmark_data["results"].append(prompt_entry)

    # Compute average distinct-3 across all prompts per strategy
    strategy_averages = {}
    for strat_id, strat in strategies.items():
        d3_scores = [x["metrics"]["distinct_3"] for p in benchmark_data["results"] for x in p["evaluations"][strat_id]["samples"]]
        strategy_averages[strat_id] = {
            "name": strat["name"],
            "avg_distinct_3": round(sum(d3_scores) / len(d3_scores), 4)
        }
    benchmark_data["strategy_averages"] = strategy_averages

    out_file = RESULTS / "distilgpt2_decoding.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nBenchmark completed successfully! Saved to: {out_file}")
    print("\nSummary of Distinct-3 Ratios (Higher = Less Repetitive):")
    for s_id, s_info in strategy_averages.items():
        print(f"  - {s_info['name']}: {s_info['avg_distinct_3'] * 100:.1f}% unique trigrams")


if __name__ == "__main__":
    {"prepare": prepare, "train": train_all, "decoding": decoding}[sys.argv[1]]()
