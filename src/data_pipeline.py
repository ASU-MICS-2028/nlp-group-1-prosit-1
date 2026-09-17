"""
Multi-source dataset ingestion, normalization, deduplication, and merging pipeline
tailored for low-resource Ewe (Èʋegbe) corpora.
"""

from pathlib import Path
import re
import random
import unicodedata
from typing import List, Dict, Any, Tuple, Optional


def clean_and_normalize_ewe_sentence(text: str, min_words: int = 2) -> Optional[str]:
    """
    Cleans and normalizes a single Ewe sentence:
    1. Strips HTML/XML tags and URLs.
    2. Enforces Unicode NFC normalization to bind combining tone diacritics.
    3. Preserves all Ewe Latin letters (ɖ, ƒ, ɣ, ŋ, ɔ, ɛ, ʋ).
    4. Filters out noise, numbers-only lines, or single-character fragments.
    """
    if not text:
        return None

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Minimum word count check
    words = text.split()
    if len(words) < min_words:
        return None

    # Verify presence of valid alphabetic characters (avoid lines of pure punctuation/digits)
    if not re.search(r"[a-zA-ZɖƒɣŋɔɛʋƉƑƔŊƆƐƲ]", text):
        return None

    return text


def load_dataset_source(file_path: Path) -> List[str]:
    """
    Loads text from a local source file supporting:
    - Plain text (.txt)
    - CSV (.csv with text/sentence/ee column)
    - JSONL (.jsonl with text/sentence/ee field)
    """
    if not file_path.exists():
        print(f"Warning: File {file_path} not found.")
        return []

    ext = file_path.suffix.lower()
    raw_lines = []

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

    elif ext == ".csv":
        import csv
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            # Find the best candidate text column
            possible_cols = ["text", "sentence", "ee", "ewe", "target", "content"]
            fieldnames = [c.lower() for c in (reader.fieldnames or [])]
            target_col = None
            for col in possible_cols:
                if col in fieldnames:
                    # Retrieve matching original casing
                    target_col = reader.fieldnames[fieldnames.index(col)]
                    break

            if target_col:
                for row in reader:
                    val = row.get(target_col, "")
                    if val:
                        raw_lines.append(val.strip())
            else:
                # Fallback to first column
                f.seek(0)
                reader_raw = csv.reader(f)
                for row in reader_raw:
                    if row:
                        raw_lines.append(row[0].strip())

    elif ext == ".jsonl":
        import json
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        val = (
                            data.get("text")
                            or data.get("sentence")
                            or data.get("ee")
                            or data.get("translation", {}).get("ee")
                        )
                        if val:
                            raw_lines.append(str(val).strip())
                    except json.JSONDecodeError:
                        continue

    return raw_lines


def merge_and_harmonize_datasets(
    source_files: List[Path],
    output_dir: Path,
    min_words: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Ingests up to 4 (or more) disparate Ewe datasets, cleans, normalizes,
    deduplicates, merges, and splits them into clean train/val/test partitions.
    """
    random.seed(seed)
    stats_per_source = {}
    seen_hashes = set()
    unified_sentences = []

    print(f"--- Starting Multi-Source Dataset Harmonization ({len(source_files)} sources) ---")

    for idx, src_path in enumerate(source_files, start=1):
        source_name = src_path.stem
        raw_lines = load_dataset_source(src_path)
        valid_lines = 0
        duplicates = 0

        for line in raw_lines:
            cleaned = clean_and_normalize_ewe_sentence(line, min_words=min_words)
            if cleaned:
                # Deduplication key based on normalized lower string
                norm_key = cleaned.lower()
                if norm_key in seen_hashes:
                    duplicates += 1
                else:
                    seen_hashes.add(norm_key)
                    unified_sentences.append(cleaned)
                    valid_lines += 1

        stats_per_source[f"Source {idx} ({source_name})"] = {
            "path": str(src_path),
            "raw_lines": len(raw_lines),
            "unique_valid_lines": valid_lines,
            "duplicates_removed": duplicates,
        }
        print(f"Source {idx} [{source_name}]: {len(raw_lines)} raw -> {valid_lines} retained ({duplicates} duplicates removed)")

    # Shuffle for stratified splitting
    random.shuffle(unified_sentences)
    total_unified = len(unified_sentences)

    n_train = int(train_ratio * total_unified)
    n_val = int(val_ratio * total_unified)

    train_data = unified_sentences[:n_train]
    val_data = unified_sentences[n_train : n_train + n_val]
    test_data = unified_sentences[n_train + n_val :]

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    unified_path = output_dir / "ewe_unified_corpus.txt"
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    test_path = output_dir / "test.txt"

    with open(unified_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unified_sentences) + "\n")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_data) + "\n")

    summary = {
        "sources_count": len(source_files),
        "source_breakdown": stats_per_source,
        "total_unique_sentences": total_unified,
        "train_sentences": len(train_data),
        "val_sentences": len(val_data),
        "test_sentences": len(test_data),
        "unified_corpus_path": str(unified_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }

    print(f"\n--- Harmonization Complete ---")
    print(f"Total Unified Corpus: {total_unified} sentences")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Exported to: {output_dir}")

    return summary
