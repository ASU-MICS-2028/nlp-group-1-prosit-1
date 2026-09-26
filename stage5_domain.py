"""Stage 5: adapt a small pretrained LLM to a domain, several ways, and measure what each way costs elsewhere.

Fan-out: 2 domains (health, agriculture) x 5 methods (LoRA rank 2/8/32, DoRA, full fine-tuning) on one base model,
plus a second base model as a check. Every model is scored on held-out HEALTH, AGRICULTURE and GENERAL text, so the
table shows both "did adaptation work?" and "what did it forget?".

Run `uv run python stage5_domain.py`; results land in results/stage5_domain.json and the notebook only reads that file.
"""
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

# Small BASE models (not instruction-tuned), stored locally. HuggingFace's large-file servers were unusably slow from
# this network (18 KB/s), so the weights come from ModelScope and are verified against HuggingFace's published sha256.
MODELS = ["data/models/Qwen2.5-0.5B", "data/models/SmolLM2-135M"]
METHODS = {  # name: (fine-tune type, LoRA rank or None)
    "lora_r2": ("lora", 2), "lora_r8": ("lora", 8), "lora_r32": ("lora", 32), "dora_r8": ("dora", 8), "full": ("full", None),
}
# Learning rate is tuned, not guessed: a first attempt at 1e-4 made rank 8 WORSE in-domain than rank 2 (too big a step).
# The sweep runs LoRA rank 8 on health at each rate and keeps the one with the best VALIDATION perplexity (never test).
LR_SWEEP = [1e-5, 3e-5, 1e-4]
FULL_LR = 1e-5  # every weight moves in full fine-tuning, so it always gets the gentlest rate
SECOND_MODEL_METHODS = ["lora_r8"]  # the second model only repeats the middle setting, as a sanity check
DOMAINS = ["health", "agriculture"]
EVAL_SETS = ["health", "agriculture", "general"]
ITERS, BATCH, MAX_LEN, LAYERS = 600, 4, 512, 16
N_TRAIN, N_VALID, N_TEST = 8000, 300, 500

ROOT = Path("data/domain")
HF = "https://huggingface.co/api/datasets/{}/parquet/{}/{}/0.parquet"
SOURCES = {"pubmed": HF.format("qiaojin/PubMedQA", "pqa_unlabeled", "train"),
           "agri": HF.format("KisanVaani/agriculture-qa-english-only", "default", "train"),
           "wiki_test": HF.format("Salesforce/wikitext", "wikitext-2-raw-v1", "test")}
RESULTS = Path("results/stage5_domain.json")


def download():
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = ROOT / "raw" / f"{name}.parquet"
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)


def write_jsonl(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")


def split_docs(docs):
    """Dedupe, then split BY DOCUMENT using a hash of its text, so nothing is in both train and test.
    (md5 of the text mod 100 gives a bucket 0-99; the same document always lands in the same split.)"""
    out = {"train": [], "valid": [], "test": []}
    for text in dict.fromkeys(" ".join(d.split()) for d in docs if d and d.strip()):
        bucket = int(hashlib.md5(text.encode()).hexdigest(), 16) % 100
        out["test" if bucket < 5 else "valid" if bucket < 10 else "train"].append(text)
    return {"train": out["train"][:N_TRAIN], "valid": out["valid"][:N_VALID], "test": out["test"][:N_TEST]}


def prepare():
    download()
    pubmed = pq.read_table(ROOT / "raw" / "pubmed.parquet").to_pylist()
    agri = pq.read_table(ROOT / "raw" / "agri.parquet").to_pylist()
    corpora = {
        "health": split_docs(" ".join(r["context"]["contexts"]) + " " + r["long_answer"] for r in pubmed),  # PubMed abstracts
        "agriculture": split_docs(f"{r['question']} {r['answers']}" for r in agri),                             # farming Q&A
    }
    wiki = [t.strip() for t in pq.read_table(ROOT / "raw" / "wiki_test.parquet")["text"].to_pylist()]
    wiki = [t for t in wiki if len(t) > 200 and not t.startswith("=")][:N_TEST]  # real paragraphs, not headings
    corpora["general"] = {"train": wiki, "valid": wiki, "test": wiki}  # only test is used; the tool wants all 3 files
    info = {}
    for name, splits in corpora.items():
        for split, texts in splits.items():
            write_jsonl(ROOT / name / f"{split}.jsonl", texts)
        words = sum(len(t.split()) for t in splits["train"])
        info[name] = {"train_docs": len(splits["train"]), "train_words": words, "test_docs": len(splits["test"]),
                      "example": splits["test"][0][:240]}
    return info


def mlx(model, *args):
    """Run `python -m mlx_lm lora ...` as a subprocess and return its printed output (training and testing both go through it)."""
    out = subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--model", model, *args], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    return out.stdout + out.stderr


def test_all(model, adapter=None):
    scores = {}
    for data in EVAL_SETS:
        args = ["--data", str(ROOT / data), "--test", "--test-batches", "-1", "--batch-size", str(BATCH), "--max-seq-length", str(MAX_LEN),
                "--adapter-path", str(adapter) if adapter else ""]  # an empty path tells mlx_lm to test the untouched base model
        scores[data] = float(re.search(r"Test ppl ([\d.]*\d)", mlx(model, *args)).group(1))
    return scores


def valid_ppl(model, domain, adapter):
    """Perplexity on the domain's VALIDATION split: the number used to choose settings, so test stays untouched."""
    folder = ROOT / f"{domain}_valid"
    folder.mkdir(exist_ok=True)
    for split in ("train", "valid", "test"):
        (folder / f"{split}.jsonl").write_text((ROOT / domain / "valid.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    args = ["--data", str(folder), "--test", "--test-batches", "-1", "--batch-size", str(BATCH), "--max-seq-length", str(MAX_LEN), "--adapter-path", str(adapter)]
    return float(re.search(r"Test ppl ([\d.]*\d)", mlx(model, *args)).group(1))


def adapt(results, model, domain, name, lr, seed=0):
    """Train one adapter (skipped if already in the results file) and score it on every test set plus validation."""
    runs = results["runs"][model]
    tag = f"{name}@{lr:g}" + (f"#seed{seed}" if seed else "")
    key = f"{domain}/{tag}"
    if key not in runs:
        kind, rank = METHODS[name]
        adapter = ROOT / "adapters" / model.split("/")[-1] / domain / tag
        args = ["--train", "--data", str(ROOT / domain), "--fine-tune-type", kind, "--iters", str(ITERS), "--batch-size", str(BATCH),
                "--num-layers", str(LAYERS), "--max-seq-length", str(MAX_LEN), "--learning-rate", str(lr),
                "--adapter-path", str(adapter), "--steps-per-eval", "300", "--seed", str(seed)]
        if rank:
            config = ROOT / f"rank{rank}.yaml"
            config.write_text(f"lora_parameters:\n  rank: {rank}\n  scale: 20.0\n  dropout: 0.0\n")
            args += ["-c", str(config)]
        t0 = time.time()
        log = mlx(model, *args)
        trainable = re.search(r"Trainable parameters: ([\d.]+)% \(([\d.]+)M/([\d.]+)M\)", log)
        runs[key] = {**test_all(model, adapter), "valid": valid_ppl(model, domain, adapter), "lr": lr, "seed": seed,
                     "train_seconds": round(time.time() - t0),
                     "trainable_pct": float(trainable.group(1)) if trainable else None,
                     "trainable_millions": float(trainable.group(2)) if trainable else None}
        print(model, key, runs[key], flush=True)
        save(results)
    elif "valid" not in runs[key]:  # a run kept from the first attempt: add its validation score
        adapter = ROOT / "adapters" / model.split("/")[-1] / domain / tag
        runs[key].update(valid=valid_ppl(model, domain, adapter), lr=lr)
        save(results)
    return runs[key]


def save(results):
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))


def run():
    info = prepare()
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"runs": {}}
    results.update(data=info, settings={"iters": ITERS, "batch": BATCH, "max_len": MAX_LEN, "layers": LAYERS, "lr_sweep": LR_SWEEP, "full_lr": FULL_LR})
    main, second = MODELS
    for model in MODELS:
        runs = results["runs"].setdefault(model, {})
        if "base" not in runs:
            runs["base"] = test_all(model)
            print(model, "base", runs["base"], flush=True)
            save(results)

    sweep = {lr: adapt(results, main, "health", "lora_r8", lr)["valid"] for lr in LR_SWEEP}  # 1. choose the learning rate
    best_lr = min(sweep, key=sweep.get)
    results["lr_choice"] = {"validation_perplexity_by_lr": {f"{k:g}": v for k, v in sweep.items()}, "chosen": best_lr}
    print("learning-rate sweep (validation perplexity):", sweep, "-> chosen", best_lr, flush=True)
    save(results)
    for domain in DOMAINS:                                                                    # 2. the method fan-out
        for name in METHODS:
            adapt(results, main, domain, name, FULL_LR if name == "full" else best_lr)
    for domain in DOMAINS:                                                                    # 3. the second model
        for name in SECOND_MODEL_METHODS:
            adapt(results, second, domain, name, best_lr)
    return results


def run_seeds(seeds=(1, 2), settings=(("agriculture", "lora_r2"), ("agriculture", "lora_r8"), ("agriculture", "dora_r8"), ("agriculture", "full"), ("health", "lora_r8"), ("health", "lora_r2"))):
    """Repeat the most-quoted settings with other random seeds, so we can tell noise from real differences."""
    results = json.loads(RESULTS.read_text())
    lr = results["lr_choice"]["chosen"]
    for domain, name in settings:
        for seed in seeds:
            adapt(results, MODELS[0], domain, name, FULL_LR if name == "full" else lr, seed=seed)
    return results


if __name__ == "__main__":
    import sys
    run_seeds() if "seeds" in sys.argv else run()
