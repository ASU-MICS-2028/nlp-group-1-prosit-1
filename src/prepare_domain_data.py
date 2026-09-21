"""
Script to prepare and split the English Domain-Specific Agriculture Dataset (KisanVaani Agriculture QA).
Saves raw parquet to data/raw/domain_english/ and formatted text splits to data/processed/domain_english/.
"""

import os
from pathlib import Path
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "domain_english"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "domain_english"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data(
    train_size: int = 4000,
    val_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
):
    print("Loading KisanVaani/agriculture-qa-english-only from Hugging Face...")
    ds = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")
    print(f"Total raw examples downloaded: {len(ds):,}")

    # Save raw dataset locally for reproducibility and offline access
    raw_parquet_path = RAW_DIR / "agriculture_qa.parquet"
    if not raw_parquet_path.exists():
        ds.to_parquet(str(raw_parquet_path))
        print(f"Saved raw parquet to: {raw_parquet_path}")

    # Shuffle deterministically
    shuffled_ds = ds.shuffle(seed=seed)

    # Slice partitions
    total_needed = train_size + val_size + test_size
    assert len(shuffled_ds) >= total_needed, "Dataset size smaller than requested partitions"

    train_data = shuffled_ds.select(range(0, train_size))
    val_data = shuffled_ds.select(range(train_size, train_size + val_size))
    test_data = shuffled_ds.select(range(train_size + val_size, total_needed))

    def format_and_save(dataset_split, output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset_split:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answers", "")).strip()
                if q and a:
                    formatted = f"Question: {q}\nAnswer: {a}\n\n"
                    f.write(formatted)
        
        # Calculate summary statistics
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
            num_words = len(content.split())
            num_lines = content.count("\n")
        print(f"  -> {output_path.name}: {len(dataset_split)} examples, {num_words:,} words")

    print("\nFormatting and writing causal language modeling splits:")
    format_and_save(train_data, PROCESSED_DIR / "train.txt")
    format_and_save(val_data, PROCESSED_DIR / "val.txt")
    format_and_save(test_data, PROCESSED_DIR / "test.txt")
    print("\nData preparation complete!")


if __name__ == "__main__":
    prepare_data()
