"""
Prepares the English agriculture Q&A corpus (KisanVaani/agriculture-qa-english-only) for Section C.

The raw corpus has 22,615 rows but only about 2,200 distinct questions: most rows are repeats. We keep
one row per question (its first occurrence) BEFORE shuffling, so a test question can never also sit
in training, then split 80/10/10 with seed 42.

Splits are JSONL, one {"question", "answer"} object per line. (Some answers contain blank lines, which
broke the old blank-line-separated text files: one Q&A pair could turn into several fragments.)

Run from the repo root:  python src/prepare_domain_data.py
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PARQUET = REPO_ROOT / "data" / "raw" / "domain_english" / "agriculture_qa.parquet"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "domain_english"


def normalize_question(question: str) -> str:
    """Lowercase, punctuation to spaces, collapsed whitespace: the key used to find repeated questions."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", question.lower())).strip()


def load_raw() -> pd.DataFrame:
    if not RAW_PARQUET.exists():
        from datasets import load_dataset

        RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        load_dataset("KisanVaani/agriculture-qa-english-only", split="train").to_parquet(str(RAW_PARQUET))
    return pd.read_parquet(RAW_PARQUET)


def prepare_data(seed: int = RANDOM_SEED) -> dict:
    df = load_raw()
    pairs, seen = [], set()
    for question, answer in zip(df["question"], df["answers"]):
        question, answer = str(question).strip(), str(answer).strip()
        key = normalize_question(question)
        if question and answer and key not in seen:
            seen.add(key)
            pairs.append({"question": question, "answer": answer})

    random.seed(seed)
    random.shuffle(pairs)
    n_train, n_val = int(0.8 * len(pairs)), int(0.1 * len(pairs))
    splits = {"train": pairs[:n_train], "val": pairs[n_train : n_train + n_val], "test": pairs[n_train + n_val :]}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(PROCESSED_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    stats = {"raw_rows": len(df), "unique_questions": len(pairs), **{f"{k}_pairs": len(v) for k, v in splits.items()}}
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(stats)
    return stats


def load_split(name: str, max_samples: int = None) -> list:
    """One split as training strings: 'Question: ...\\nAnswer: ...'."""
    with open(PROCESSED_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows[:max_samples]]


if __name__ == "__main__":
    prepare_data()
