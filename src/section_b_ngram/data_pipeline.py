"""
Multi-source dataset ingestion, normalization, deduplication, and merging pipeline
tailored for low-resource Ewe (Èʋegbe) corpora.
"""

from pathlib import Path
import re
import random
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

# Lookalike letters typed in place of Ewe letters. Capital eth Ð looks identical to African D Ɖ, but
# lowercases to ð instead of ɖ, so "Ðe" and "Ɖe" would become two different words. Greek ε stands in for ɛ.
EWE_LOOKALIKES = str.maketrans({"Ð": "Ɖ", "ð": "ɖ", "ε": "ɛ"})


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

    # Drop corrupted rows: Dataset 1's CSV holds a few binary blobs (control bytes, "\E7"-style escapes)
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|\\[0-9A-F]{2}", text):
        return None
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # NFC normalization, then drop zero-width characters (they are not whitespace, so \s+ misses them)
    text = unicodedata.normalize("NFC", text).translate(EWE_LOOKALIKES)
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
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
    sources = {f"Source {i} ({p.stem})": load_dataset_source(p) for i, p in enumerate(source_files, start=1)}
    return harmonize_sentences(sources, output_dir, min_words, train_ratio, val_ratio, test_ratio, seed)


def harmonize_sentences(
    sources: Dict[str, List[str]],
    output_dir: Path,
    min_words: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Same pipeline as merge_and_harmonize_datasets, for sentences already in memory
    ({source name: raw lines}). Used by src/section_b_ngram/build_datasets.py.
    """
    random.seed(seed)
    stats_per_source = {}
    seen_hashes = set()
    unified_sentences = []

    print(f"--- Starting Multi-Source Dataset Harmonization ({len(sources)} sources) ---")

    for source_name, raw_lines in sources.items():
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

        stats_per_source[source_name] = {
            "raw_lines": len(raw_lines),
            "unique_valid_lines": valid_lines,
            "duplicates_removed": duplicates,
        }
        print(f"{source_name}: {len(raw_lines)} raw -> {valid_lines} retained ({duplicates} duplicates removed)")

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

    words = lambda lines: sum(len(line.split()) for line in lines)
    summary = {
        "sources_count": len(sources),
        "source_breakdown": stats_per_source,
        "total_unique_sentences": total_unified,
        "train_sentences": len(train_data),
        "val_sentences": len(val_data),
        "test_sentences": len(test_data),
        "train_words": words(train_data),
        "val_words": words(val_data),
        "test_words": words(test_data),
    }

    print(f"\n--- Harmonization Complete ---")
    print(f"Total Unified Corpus: {total_unified} sentences")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Exported to: {output_dir}")

    return summary
